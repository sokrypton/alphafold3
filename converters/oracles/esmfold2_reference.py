"""A complete, self-contained JAX reference implementation of ESMFold2.

This is the consolidation of the per-module gates into one runnable model:
features + ESM-C hidden states -> folded coordinates.  It is deliberately plain
jnp (no haiku) so it can be read as the SPEC for the graph work, and so every
constant in it is one a gate has already checked against native.

Not included: the ESM-C 6B tower (hidden states are an input), and the MSA
encoder (single-sequence path only).

Gated pieces, all against a deterministic fp32 native reference:
  trunk, raw features -> distogram   corr 0.99961
  diffusion denoise step             corr 0.99999767
  confidence head                    corr >= 0.99999981
"""
import numpy as np
import jax
import jax.numpy as jnp

MAX_Z, CHARV, MAXC = 128, 64, 4
SIGMA_DATA = 16.0
NRB, NCB = 32, 2


# ── primitives ──────────────────────────────────────────────────────────────

def layer_norm(x, scale=1.0, offset=0.0, eps=1e-5):
  m = x.mean(-1, keepdims=True)
  v = x.var(-1, keepdims=True)
  return (x - m) * jax.lax.rsqrt(v + eps) * scale + offset


def rms(x):
  # torch F.rms_norm(eps=None) uses finfo(dtype).eps, NOT 1e-5
  return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + np.finfo(np.float32).eps)


def swiglu(x, w_in, w_out):
  h = x @ w_in
  n = h.shape[-1] // 2
  return (jax.nn.silu(h[..., :n]) * h[..., n:]) @ w_out


# ── pair-only trunk block ───────────────────────────────────────────────────

def tri_mul(z, p, mask, outgoing):
  zn = layer_norm(z, p['left_norm_input/scale'], p['left_norm_input/offset'])
  routed = (zn @ p['projection/weights']) * jax.nn.sigmoid(zn @ p['gate/weights'])
  routed = routed * mask[..., None]
  a, b = routed[..., 0::2], routed[..., 1::2]
  # AF3's equations; the converter has already swapped a/b for incoming, because
  # ESMFold2 contracts left[k,i]*right[k,j] where AF3 contracts a[k,j]*b[k,i]
  c = jnp.einsum('ikd,jkd->ijd', a, b) if outgoing else jnp.einsum('kjd,kid->ijd', a, b)
  c = layer_norm(c, p['center_norm/scale'], p['center_norm/offset'])
  return (c @ p['output_projection/weights']) * jax.nn.sigmoid(zn @ p['gating_linear/weights'])


def pair_block(z, p, mask):
  sub = lambda q: {k[len(q):]: v for k, v in p.items() if k.startswith(q)}
  z = z + tri_mul(z, sub('triangle_multiplication_outgoing/'), mask, True)
  z = z + tri_mul(z, sub('triangle_multiplication_incoming/'), mask, False)
  t = sub('pair_transition/')
  z = z + swiglu(layer_norm(z, t['input_layer_norm/scale'], t['input_layer_norm/offset']),
                 t['transition1/weights'], t['transition2/weights'])
  return z


def pair_stack(z, p, prefix, n, mask):
  q = {k[len(prefix):]: v for k, v in p.items() if k.startswith(prefix)}
  for i in range(n):
    z = pair_block(z, {k: v[i] for k, v in q.items()}, mask)
  return z


# ── SWA / 3D-RoPE atom transformer ──────────────────────────────────────────

def rope_inv_freq(n, base):
  return (1.0 / (base ** (np.arange(n, dtype=np.float32) / n))).astype(np.float32)


def build_rope(ref_pos, uid, head_dim, n_sp=2, n_uid=10, sp_base=20.0, uid_base=10000.0):
  fs = (ref_pos[..., None] * rope_inv_freq(n_sp, sp_base)).reshape(ref_pos.shape[0], -1)
  fu = uid[:, None] * rope_inv_freq(n_uid, uid_base)
  fr = jnp.concatenate([fs, fu], -1)
  half = head_dim // 2
  if fr.shape[-1] < half:
    fr = jnp.concatenate([fr, jnp.zeros((fr.shape[0], half - fr.shape[-1]))], -1)
  return jnp.cos(fr), jnp.sin(fr)


def apply_rope(x, cos, sin):
  # native TILES cos/sin ([c|c]) to pair with rotate_half's split-into-halves;
  # interleaving instead silently reads corr 0.88
  ro = cos.shape[-1] * 2
  c = jnp.concatenate([cos, cos], -1)[:, None]
  s = jnp.concatenate([sin, sin], -1)[:, None]
  a, b = jnp.split(x[..., :ro], 2, axis=-1)
  return jnp.concatenate([x[..., :ro] * c + jnp.concatenate([-b, a], -1) * s, x[..., ro:]], -1)


def swa_attn(x, p, cos, sin, valid, half_window, n_heads):
  N, d = x.shape
  dh = d // n_heads
  qkv = (x @ p['qkv/weights']).reshape(N, 3, n_heads, dh)
  q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
  q, k = apply_rope(rms(q), cos, sin), apply_rope(rms(k), cos, sin)
  rank = jnp.cumsum(valid) - 1                       # window over RANK among valid atoms
  ok = (jnp.abs(rank[:, None] - rank[None, :]) <= half_window) \
       & (valid[:, None] > 0) & (valid[None, :] > 0)
  ok = ok | jnp.eye(N, dtype=bool)
  logits = jnp.where(ok[None], jnp.einsum('ihd,jhd->hij', q, k) * dh ** -0.5, -1e9)
  o = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(N, d)
  o = o * valid[:, None] * jax.nn.sigmoid(x @ p['attn_gate/weights'])
  return o @ p['attn_out/weights']


def swa_block(x, c, p, cos, sin, valid, half_window, n_heads):
  sh_a, sc_a, g_a, sh_f, sc_f, g_f = jnp.split(jax.nn.silu(c) @ p['adaln/weights'], 6, -1)
  x = x + g_a * swa_attn(rms(x) * (1 + sc_a) + sh_a, p, cos, sin, valid, half_window, n_heads)
  x = x + g_f * swiglu(rms(x) * (1 + sc_f) + sh_f, p['ffn_up/weights'], p['ffn_down/weights'])
  return x


def atom_stack(q, c, p, prefix, n, cos, sin, mask, n_heads=4, half_window=64):
  b = {k[len(prefix):]: v for k, v in p.items() if k.startswith(prefix)}
  for i in range(n):
    q = swa_block(q, c, {k: v[i] for k, v in b.items()}, cos, sin, mask, half_window, n_heads)
  return q


def scatter_mean(x, seg, n, w):
  num = jax.ops.segment_sum(x * w[:, None], seg, n)
  den = jax.ops.segment_sum(w, seg, n)[:, None]
  return num / jnp.maximum(den, 1e-9)


# ── features ────────────────────────────────────────────────────────────────

def atom_features(f, mask):
  elem = jax.nn.one_hot(f['ref_element'].astype(int), MAX_Z) * mask[:, None]
  chars = (jax.nn.one_hot(f['ref_atom_name_chars'].astype(int), CHARV)
           * mask[:, None, None]).reshape(-1, MAXC * CHARV)
  return jnp.concatenate([f['ref_pos'], f['ref_charge'][:, None], mask[:, None], elem, chars], -1)


def rel_pos_features(residue_index, asym_id, sym_id, entity_id, token_index):
  """Concat order is [rel_pos, rel_token, same_entity, rel_chain] -- entity BEFORE
  chain -- and the chain bucket is INVERTED (same-chain goes to the OOB bin)."""
  same_chain = asym_id[:, None] == asym_id[None, :]
  same_res = residue_index[:, None] == residue_index[None, :]
  same_ent = entity_id[:, None] == entity_id[None, :]
  d_res = np.where(same_chain, np.clip(residue_index[:, None] - residue_index[None, :] + NRB, 0, 2*NRB), 2*NRB+1)
  d_tok = np.where(same_chain & same_res, np.clip(token_index[:, None] - token_index[None, :] + NRB, 0, 2*NRB), 2*NRB+1)
  d_ch = np.where(same_chain, 2*NCB+1, np.clip(sym_id[:, None] - sym_id[None, :] + NCB, 0, 2*NCB))
  oh = lambda x, n: np.eye(n, dtype=np.float32)[x]
  return np.concatenate([oh(d_res, 2*NRB+2), oh(d_tok, 2*NRB+2),
                         same_ent.astype(np.float32)[..., None], oh(d_ch, 2*NCB+2)], -1)


# ── the LM shim ─────────────────────────────────────────────────────────────

def lm_shim(hidden, p):
  x = layer_norm(hidden, p['language_model/lm_norm/scale'], p['language_model/lm_norm/offset'])
  x = jnp.einsum('k,lkc->lc', p['language_model/combine'], x @ p['language_model/lm_projection/weights'])
  x = x @ p['language_model/downproject/weights'] + p['language_model/downproject/bias']
  z = jnp.concatenate([x[:, None] * x[None, :], x[:, None] - x[None, :]], -1)
  z = z @ p['language_model/pair_mlp_1/weights'] + p['language_model/pair_mlp_1/bias']
  z = jax.nn.gelu(z, approximate=False) @ p['language_model/pair_mlp_2/weights'] + p['language_model/pair_mlp_2/bias']
  return layer_norm(z, p['language_model/pair_norm/scale'], p['language_model/pair_norm/offset'])


# ── the ESM-C 6B tower ──────────────────────────────────────────────────────
#
# 80 pre-norm blocks, d_model 2560 / 40 heads / head_dim 64, ESM3 residual
# scaling by sqrt(n_layers/36).  ESMFold2 consumes all 81 hidden states, and
# HF's LAST one is POST the stack's final LayerNorm while the other 80 are the
# raw residual stream -- using the pre-norm value there reads corr 0.909 against
# 0.9998 everywhere else.
#
# Gated vs torch ESM-C: all 81 layers, corr 0.99985508 (the residual is native
# running bf16 against our fp32).

def _rope(x, pos, base=10000.0):
  d = x.shape[-1]
  inv = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
  fr = pos[:, None] * inv[None, :]
  c, sn = jnp.cos(fr)[:, None], jnp.sin(fr)[:, None]
  x1, x2 = x[..., :d//2], x[..., d//2:]
  return jnp.concatenate([x1 * c - x2 * sn, x1 * sn + x2 * c], -1)


def esmc_hidden_states(ids, p, dims):
  """[L] token ids (already BOS/EOS wrapped) -> [n_layers+1, L, d_model]."""
  L = ids.shape[0]
  H = dims['n_heads']
  dh = dims['d_model'] // H
  scale = dims['residual_scale']
  x = jnp.asarray(p['embed/weights'])[ids]
  states = [x]
  pos = jnp.arange(L, dtype=jnp.float32)
  B = {k[len('blocks/'):]: v for k, v in p.items() if k.startswith('blocks/')}
  for i in range(dims['n_layers']):
    b = {k: jnp.asarray(v[i]) for k, v in B.items()}
    h = layer_norm(x, b['attn_norm/scale'], b['attn_norm/offset']) @ b['qkv/weights']
    q, k, v = jnp.split(h, 3, -1)
    q = layer_norm(q, b['q_norm/scale'])          # over the FULL d_model, not per head
    k = layer_norm(k, b['k_norm/scale'])
    q = _rope(q.reshape(L, H, dh), pos)
    k = _rope(k.reshape(L, H, dh), pos)
    v = v.reshape(L, H, dh)
    logits = jnp.einsum('ihd,jhd->hij', q, k) * dh ** -0.5
    ctx = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(L, -1)
    x = x + (ctx @ b['attn_out/weights']) / scale
    h = layer_norm(x, b['ffn_norm/scale'], b['ffn_norm/offset']) @ b['fc1/weights']
    n = h.shape[-1] // 2
    x = x + ((jax.nn.silu(h[..., :n]) * h[..., n:]) @ b['fc2/weights']) / scale
    states.append(x)
  states[-1] = layer_norm(states[-1], p['final_norm/scale'])
  return jnp.stack(states)


# ── the MSA encoder ─────────────────────────────────────────────────────────
#
# NOT optional.  msa_encoder_overwrite=True means this REPLACES the injected
# pair representation rather than adding to it, and native scores 3.4 A with it
# against 20-22 A without -- even on a depth-1 self MSA.  Layout is TOKEN-major
# [L, M, c], unlike AF3's [M, L, c].

def outer_product_mean(m, mmask, q):
  """NOTE the divide order: Wout(outer)/n_valid, so the BIAS is scaled too."""
  mn = layer_norm(m, q['layer_norm/scale'], q['layer_norm/offset'])
  a = (mn @ q['left_projection/weights']) * mmask[..., None]
  b = (mn @ q['right_projection/weights']) * mmask[..., None]
  n_valid = jnp.maximum(mmask @ mmask.T, 1.0)[..., None]
  outer = jnp.einsum('imc,jmd->ijcd', a, b).reshape(a.shape[0], b.shape[0], -1)
  return (outer @ q['output/weights'] + q['output/bias']) / n_valid


def msa_pair_weighted_averaging(m, z, pair_mask, q):
  mn = layer_norm(m, q['msa_norm/scale'], q['msa_norm/offset'])
  zn = layer_norm(z, q['pair_norm/scale'], q['pair_norm/offset'])
  bias = jnp.where(pair_mask[..., None] > 0, zn @ q['bias/weights'], -1e5)
  attn = jax.nn.softmax(bias, axis=-2)
  L, M = m.shape[0], m.shape[1]
  h = bias.shape[-1]; dh = q['value/weights'].shape[1] // h
  v = (mn @ q['value/weights']).reshape(L, M, h, dh)
  g = jax.nn.sigmoid(mn @ q['gate/weights']).reshape(L, M, h, dh)
  o = jnp.einsum('ijh,jmhd,imhd->imhd', attn, v, g)
  return o.reshape(L, M, h * dh) @ q['output/weights']


def msa_encoder(z, s_inputs, msa_oh, has_deletion, deletion_value, mmask, p, dims):
  q = {k[len('msa_encoder/'):]: v for k, v in p.items() if k.startswith('msa_encoder/')}
  feat = jnp.concatenate([msa_oh, has_deletion[..., None], deletion_value[..., None]], -1)
  m = feat @ q['embed/weights'] + (s_inputs @ q['project_inputs/weights'])[:, None]
  tok = mmask[:, 0]
  pair_mask = tok[:, None] * tok[None, :]
  n = dims['n_msa']
  for i in range(n):
    sub = lambda pre, d: {k[len(pre):]: v for k, v in d.items() if k.startswith(pre)}
    blk = ({k[len('blocks/'):]: v[i] for k, v in q.items() if k.startswith('blocks/')}
           if i < n - 1 else sub('final_block/', q))
    z = z + outer_product_mean(m, mmask, sub('outer_product_mean/', blk))
    if i < n - 1:                      # the LAST block drops the MSA update entirely
      m = m + msa_pair_weighted_averaging(m, z, pair_mask, sub('msa_pair_weighted_averaging/', blk))
      t = sub('msa_transition/', blk)
      m = m + swiglu(layer_norm(m, t['input_layer_norm/scale'], t['input_layer_norm/offset']),
                     t['transition1/weights'], t['transition2/weights'])
    z = pair_block(z, blk, pair_mask)
  return z


# ── the trunk ───────────────────────────────────────────────────────────────

# Stage taps, mirroring evoformer.ESM_TRUNK_TAPS so the two can be compared
# stage by stage rather than only at the end. Always on: this file is a spec,
# not a hot path.
TAPS = {}


def trunk(f, lm_hidden, p, dims, n_loops=3, key=None, lm_dropout=0.25, msa=None):
  """features + ESM-C hidden states -> (z, s_inputs, rel_pos).

  The parcae recurrence replaces AF3 recycling: a diagonal SSM over the loop
  axis whose initial state is RANDOM (trunc normal, std sqrt(2/(5c))).
  """
  mask = f['atom_attention_mask']
  L = f['res_type'].shape[0]
  a2t = f['atom_to_token'].astype(int) * mask.astype(int)

  ae = {k[len('inputs_embedder/'):]: v for k, v in p.items() if k.startswith('inputs_embedder/')}
  c0 = layer_norm(atom_features(f, mask) @ ae['atom_linear/weights'],
                  ae['atom_norm/scale'], ae['atom_norm/offset'])
  cos, sin = build_rope(f['ref_pos'], f['ref_space_uid'], c0.shape[-1] // 4)
  q = atom_stack(c0, c0, ae, 'blocks/', dims['n_input_atom'], cos, sin, mask)
  a = scatter_mean(jax.nn.relu(q @ ae['atom_to_token/weights']), jnp.asarray(a2t), L, mask)

  res_oh = jax.nn.one_hot(f['res_type'].astype(int), 33) * f['token_attention_mask'][:, None]
  s_inputs = jnp.concatenate([a, res_oh, res_oh, jnp.zeros((L, 1))], -1)   # no MSA: profile = res_type

  rp = jnp.asarray(rel_pos_features(f['residue_index'].astype(int), f['asym_id'].astype(int),
                                    f['sym_id'].astype(int), f['entity_id'].astype(int),
                                    f['token_index'].astype(int))) @ p['rel_pos/weights']
  z_pair0 = ((s_inputs @ p['z_init_1/weights'])[:, None]
             + (s_inputs @ p['z_init_2/weights'])[None, :])
  TAPS.setdefault('s_inputs', []).append(s_inputs)
  TAPS.setdefault('z_pair0', []).append(z_pair0)
  TAPS.setdefault('z_relpos', []).append(z_pair0 + rp)
  z_init = (z_pair0 + rp + f['token_bonds'] @ p['token_bonds/weights'])
  TAPS.setdefault('z_init', []).append(z_init)

  pm = jnp.ones((L, L))
  # lm_hidden=None reproduces native's no-LM path exactly: `lm_z is None` means
  # refined_lm_z is never computed and z_inject is z_init alone. Passing zeros
  # instead would NOT be equivalent -- the shim's biases make lm_shim(0) nonzero.
  lm_z = None if lm_hidden is None else lm_shim(lm_hidden, p)

  key = jax.random.PRNGKey(0) if key is None else key
  key, k_init = jax.random.split(key)
  std = np.sqrt(2.0 / (5.0 * z_init.shape[-1]))
  z = jax.random.truncated_normal(k_init, -3.0, 3.0, z_init.shape) * std
  av, bT = p['parcae_a'], p['parcae_b/weights']
  for _ in range(n_loops + 1):
    # ESMFold2 keeps 25% dropout on the LM pair rep at INFERENCE, resampled every
    # loop (config.lm_encoder.per_loop_lm_dropout).  It is not optional polish:
    # disabling it costs ~18 A on 6MRR.
    lm_ref = None
    if lm_z is not None:
      lm_i = lm_z
      if lm_dropout > 0:
        key, k_do = jax.random.split(key)
        keep = jax.random.bernoulli(k_do, 1.0 - lm_dropout, lm_z.shape)
        lm_i = lm_z * keep / (1.0 - lm_dropout)
      lm_ref = pair_stack(lm_i, p, 'lm_encoder/', dims['n_lm_encoder'], pm)
    z_inject = z_init
    if msa is not None:
      # OVERWRITE, not add (config.msa_encoder_overwrite)
      z_inject = msa_encoder(z_inject, s_inputs, msa['oh'], msa['has_deletion'],
                             msa['deletion_value'], msa['mask'], p, dims)
    TAPS.setdefault('z_inject', []).append(z_inject)
    inj = layer_norm(z_inject if lm_ref is None else z_inject + lm_ref,
                     p['parcae_input_norm/scale'], p['parcae_input_norm/offset'])
    z = av * z + inj @ bT
    TAPS.setdefault('z_parcae', []).append(z)
    z = pair_stack(z, p, 'folding_trunk/', dims['n_trunk'], pm)
  z = pair_stack(z @ p['parcae_readout/weights'], p, 'parcae_coda/', dims['n_coda'], pm)
  return z, s_inputs, rp


# ── the diffusion module ────────────────────────────────────────────────────

def diffusion_conditioning(s_inputs, z_trunk, rel_pos, t_hat, p):
  c = {k[len('conditioning/'):]: v for k, v in p.items() if k.startswith('conditioning/')}
  z = jnp.concatenate([z_trunk, rel_pos], -1)
  z = layer_norm(z, c['z_input_norm/scale'], c['z_input_norm/offset']) @ c['z_projection/weights']
  def stack(x, pre):
    q = {k[len(pre):]: v for k, v in c.items() if k.startswith(pre)}
    for i in range(q['transition1/weights'].shape[0]):
      x = x + swiglu(layer_norm(x, q['input_layer_norm/scale'][i], q['input_layer_norm/offset'][i]),
                     q['transition1/weights'][i], q['transition2/weights'][i])
    return x
  z = stack(z, 'z_transitions/')
  s = layer_norm(s_inputs, c['s_input_norm/scale'], c['s_input_norm/offset']) @ c['s_projection/weights']
  t_noise = 0.25 * jnp.log(jnp.maximum(t_hat / SIGMA_DATA, 1e-20))
  n = jnp.cos(2 * jnp.pi * (t_noise * c['fourier_w'] + c['fourier_b']))
  s = s + (layer_norm(n, c['noise_norm/scale'], c['noise_norm/offset']) @ c['noise_projection/weights'])[None]
  return stack(s, 's_transitions/'), z


def adaln(a, s, p):
  s_n = layer_norm(s, p['adaln/s_norm/scale'], 0.0)      # scale, NO offset
  return jax.nn.sigmoid(s_n @ p['adaln/gate/weights'] + p['adaln/gate/bias']) * layer_norm(a) \
         + s_n @ p['adaln/shift/weights']


def token_attn(a, s, z, p, n_heads):
  L, D = a.shape
  dh = D // n_heads
  x = adaln(a, s, p)
  q = (x @ p['q/weights'] + p['q/bias']).reshape(L, n_heads, dh)
  k, v = jnp.split(x @ p['kv/weights'], 2, -1)
  bias = layer_norm(z, p['pair_norm/scale'], p['pair_norm/offset']) @ p['pair_bias/weights']
  logits = jnp.einsum('ihd,jhd->ijh', q, k.reshape(L, n_heads, dh)) * dh ** -0.5 + bias
  ctx = jnp.einsum('ijh,jhd->ihd', jax.nn.softmax(logits, axis=-2), v.reshape(L, n_heads, dh))
  g = jax.nn.sigmoid(x @ p['g/weights']).reshape(L, n_heads, dh)
  out = (g * ctx).reshape(L, D) @ p['out/weights']
  return jax.nn.sigmoid(s @ p['out_gate/weights'] + p['out_gate/bias']) * out


def token_transition(a, s, p):
  out = swiglu(adaln(a, s, p), p['swish/weights'], p['out/weights'])
  return jax.nn.sigmoid(s @ p['out_gate/weights'] + p['out_gate/bias']) * out


def denoise(x_noisy, t_hat, f, s_inputs, z_trunk, rel_pos, p, dims, n_heads=16):
  """One EDM denoise step -> x_denoised."""
  d = {k[len('diffusion/'):]: v for k, v in p.items() if k.startswith('diffusion/')}
  s, z = diffusion_conditioning(s_inputs, z_trunk, rel_pos, t_hat, d)
  mask = f['atom_attention_mask']
  a2t = jnp.asarray(f['atom_to_token'].astype(int) * mask.astype(int))
  L = s_inputs.shape[0]

  ae = {k[len('atom_encoder/'):]: v for k, v in d.items() if k.startswith('atom_encoder/')}
  c0 = layer_norm(atom_features(f, mask) @ ae['atom_linear/weights'],
                  ae['atom_norm/scale'], ae['atom_norm/offset'])
  cos, sin = build_rope(f['ref_pos'], f['ref_space_uid'], c0.shape[-1] // 4)
  r_noisy = x_noisy / jnp.sqrt(t_hat ** 2 + SIGMA_DATA ** 2)            # c_in
  q = c0 + jnp.concatenate([r_noisy, jnp.zeros_like(r_noisy)], -1) @ ae['coords_linear/weights']
  q = atom_stack(q, c0, ae, 'blocks/', dims['n_diff_atom'], cos, sin, mask)
  a = scatter_mean(jax.nn.relu(q @ ae['atom_to_token/weights']), a2t, L, mask)

  a = a + layer_norm(s, d['s_step_norm/scale'], d['s_step_norm/offset']) @ d['s_to_token/weights']
  ta = {k[len('token_attn/'):]: v for k, v in d.items() if k.startswith('token_attn/')}
  tt = {k[len('token_transition/'):]: v for k, v in d.items() if k.startswith('token_transition/')}
  for i in range(ta['q/weights'].shape[0]):
    a = a + token_attn(a, s, z, {k: v[i] for k, v in ta.items()}, n_heads)
    a = a + token_transition(a, s, {k: v[i] for k, v in tt.items()})
  a = layer_norm(a, d['token_norm/scale'], d['token_norm/offset'])

  ad = {k[len('atom_decoder/'):]: v for k, v in d.items() if k.startswith('atom_decoder/')}
  qd = q + (a @ ad['token_to_atom/weights'])[a2t]
  qd = atom_stack(qd, c0, ad, 'blocks/', dims['n_diff_atom'], cos, sin, mask)
  r_update = layer_norm(qd, ad['norm/scale'], ad['norm/offset']) @ ad['output/weights']

  s2, t2 = SIGMA_DATA ** 2, t_hat ** 2
  return (s2 / (s2 + t2)) * x_noisy + (SIGMA_DATA * t_hat / jnp.sqrt(s2 + t2)) * r_update


# ── the EDM sampler (Algorithm 18) ──────────────────────────────────────────

INFER = dict(steps=14, s_max=160.0, s_min=4e-4, p=7.0, gamma_0=0.8, gamma_min=1.0,
             noise_scale=1.003, step_scale=1.5, max_inference_sigma=256.0)


def noise_schedule(steps, s_max, s_min, p, sigma_data=SIGMA_DATA):
  k = np.arange(steps, dtype=np.float32)
  base = s_max ** (1/p) + (k / (steps - 1)) * (s_min ** (1/p) - s_max ** (1/p))
  return np.concatenate([sigma_data * base ** p, [0.0]])


def _kabsch(mob, ref, w):
  """Weighted rigid align of `mob` onto `ref` (the sampler realigns every step)."""
  wm = w[:, None]
  mc = (mob * wm).sum(0) / wm.sum()
  rc = (ref * wm).sum(0) / wm.sum()
  a, b = mob - mc, ref - rc
  u, _, vt = jnp.linalg.svd((a * wm).T @ b)
  d = jnp.sign(jnp.linalg.det(u @ vt))
  r = u @ jnp.diag(jnp.array([1.0, 1.0, d])) @ vt
  return a @ r + rc


def sample(f, s_inputs, z_trunk, rel_pos, p, dims, key, cfg=None, augment=True):
  cfg = {**INFER, **(cfg or {})}
  sched = noise_schedule(cfg['steps'], cfg['s_max'], cfg['s_min'], cfg['p'])
  sched = sched[sched <= cfg['max_inference_sigma']]
  sched = np.concatenate([[cfg['max_inference_sigma']], sched])
  gammas = np.where(sched > cfg['gamma_min'], cfg['gamma_0'], 0.0)
  mask = f['atom_attention_mask']

  key, sub = jax.random.split(key)
  x = sched[0] * jax.random.normal(sub, (mask.shape[0], 3))
  for i in range(len(sched) - 1):
    sigma_tm, sigma_t, gamma = float(sched[i]), float(sched[i+1]), float(gammas[i+1])
    if augment:                                   # Algorithm 19: centre, rotate, translate
      key, sub, sub2 = jax.random.split(key, 3)
      x = x - (x * mask[:, None]).sum(0) / mask.sum()
      # a PROPER rotation: u @ vt from a random matrix has det +/-1, and the -1
      # half is a REFLECTION -- applying those mid-trajectory inverts chirality
      u, _, vt = jnp.linalg.svd(jax.random.normal(sub, (3, 3)))
      r = u @ jnp.diag(jnp.array([1.0, 1.0, jnp.sign(jnp.linalg.det(u @ vt))])) @ vt
      x = x @ r + jax.random.normal(sub2, (1, 3))
    t_hat = sigma_tm * (1.0 + gamma)
    eps_std = cfg['noise_scale'] * max(t_hat ** 2 - sigma_tm ** 2, 0.0) ** 0.5
    key, sub = jax.random.split(key)
    x_noisy = x + eps_std * jax.random.normal(sub, x.shape)
    x_den = denoise(x_noisy, t_hat, f, s_inputs, z_trunk, rel_pos, p, dims)
    # the sampler realigns the NOISY coords onto the denoised before the Euler step
    x_noisy = _kabsch(x_noisy, x_den, mask)
    x = x_noisy + cfg['step_scale'] * (sigma_t - t_hat) * (x_noisy - x_den) / t_hat
  return x


def self_msa(f):
  """The depth-1 'MSA' of the query alone, which is what a single-sequence run uses."""
  L = f['res_type'].shape[0]
  return dict(oh=jax.nn.one_hot(f['res_type'].astype(int), 33)[:, None],
              has_deletion=jnp.zeros((L, 1)), deletion_value=jnp.zeros((L, 1)),
              mask=f['token_attention_mask'][:, None])


def fold(f, lm_hidden, p, dims, seed=0, n_loops=3, lm_dropout=0.25, msa='self', **kw):
  key = jax.random.PRNGKey(seed)
  key, k_trunk = jax.random.split(key)
  msa = self_msa(f) if msa == 'self' else msa
  z, s_inputs, rel_pos = trunk(f, lm_hidden, p, dims, n_loops=n_loops, key=k_trunk,
                               lm_dropout=lm_dropout, msa=msa)
  return sample(f, s_inputs, z, rel_pos, p, dims, key, **kw)
