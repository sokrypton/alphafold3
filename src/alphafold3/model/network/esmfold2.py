"""ESMFold2's graph.

ESMFold2 diverges from AF3 far enough that reshaping it into the shared
Evoformer would be mostly `if model == 'esmfold2'` branches, so it gets its own
module tree and `Model.__call__` dispatches to it -- the packaging plegadx uses
for template embedders, and the same judgement made for OpenDDE's structural
stage. AF3's own graph is untouched.

Three things make it unlike the rest of this package:

  * THE TRUNK HAS NO SINGLE TRACK. It is 48 pair-only blocks (tri_mul_out /
    tri_mul_in / transition); there is no triangle attention and no pairformer
    single update, and the structure head is handed s_trunk=None. The only
    single representation anywhere is s_inputs (451-d).

  * RECYCLING IS A LINEAR RECURRENCE. "Parcae" is a discretised diagonal SSM
    over the loop axis: `z = a * z + linear(norm(z_inject), b)`. a and b are
    input-independent so they fold to plain arrays at conversion time. The
    initial state is RANDOM (trunc normal, std sqrt(2/(5c))), which makes the
    model stochastic before diffusion even starts.

  * ATOM ATTENTION IS SLIDING-WINDOW WITH 3D ROPE, not AF3's 32-query/128-key
    window with a pair bias. The positional signal is entirely rotary, built
    from the reference conformer: ref_pos on 3 axes x 2 pairs at base 20, plus
    ref_space_uid x 10 pairs at base 10000 -- 3*2 + 10 = 16 = head_dim/2.

Every constant here has a gate behind it in converters/oracles/esmfold2_*.py;
converters/oracles/esmfold2_reference.py is the plain-jnp spec this mirrors.
"""

from alphafold3.common import base_config
from alphafold3.model.components import haiku_modules as hm
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


MAX_ATOMIC_NUMBER = 128
CHAR_VOCAB_SIZE = 64
MAX_CHARS = 4
NUM_RES_TYPES = 33
SIGMA_DATA = 16.0


def _rms(x):
  """Affine-free RMSNorm. torch's F.rms_norm(eps=None) uses finfo(dtype).eps."""
  eps = float(np.finfo(np.float32).eps)
  return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def _swiglu(x, w_in, w_out):
  h = x @ w_in
  n = h.shape[-1] // 2
  return (jax.nn.silu(h[..., :n]) * h[..., n:]) @ w_out


class PairTransition(hk.Module):
  """LayerNorm + fused SwiGLU, the transition every ESMFold2 stack uses."""

  def __init__(self, num_channels, *, expansion=4, name='pair_transition'):
    super().__init__(name=name)
    self.num_channels = num_channels
    self.expansion = expansion

  def __call__(self, x):
    # ffn.w12 is (2*hidden, c) with hidden = expansion * c, so transition1
    # widens to 2*expansion*c and the SwiGLU halves are each expansion*c.
    h = hm.LayerNorm(name='input_layer_norm')(x)
    h = hm.Linear(2 * self.expansion * self.num_channels, name='transition1')(h)
    n = h.shape[-1] // 2
    return hm.Linear(x.shape[-1], name='transition2')(
        jax.nn.silu(h[..., :n]) * h[..., n:])


class TriangleMultiplication(hk.Module):
  """ESMFold2 packs both projections and both gates into one proj_bundle.

  The converter splits it and re-fuses as an interleave, and pre-swaps a/b for
  the incoming direction -- ESMFold2 contracts left[k,i]*right[k,j] where AF3
  contracts a[k,j]*b[k,i], so AF3's (a, b) IS ESMFold2's (right, left).
  """

  def __init__(self, num_channels, *, outgoing, name):
    super().__init__(name=name)
    self.num_channels = num_channels
    self.outgoing = outgoing

  def __call__(self, z, mask):
    zn = hm.LayerNorm(name='left_norm_input')(z)
    proj = hm.Linear(self.num_channels * 2, name='projection')(zn)
    gate = jax.nn.sigmoid(hm.Linear(self.num_channels * 2, name='gate')(zn))
    routed = proj * gate * mask[..., None]
    a, b = routed[..., 0::2], routed[..., 1::2]
    eq = 'ikd,jkd->ijd' if self.outgoing else 'kjd,kid->ijd'
    c = jnp.einsum(eq, a, b)
    c = hm.LayerNorm(name='center_norm')(c)
    out = hm.Linear(z.shape[-1], name='output_projection')(c)
    return out * jax.nn.sigmoid(hm.Linear(z.shape[-1], name='gating_linear')(zn))


class PairBlock(hk.Module):
  """tri_mul_out + tri_mul_in + transition. No attention, no single track."""

  def __init__(self, num_channels, *, name='pair_block'):
    super().__init__(name=name)
    self.num_channels = num_channels

  def __call__(self, z, mask):
    z = z + TriangleMultiplication(
        self.num_channels, outgoing=True,
        name='triangle_multiplication_outgoing')(z, mask)
    z = z + TriangleMultiplication(
        self.num_channels, outgoing=False,
        name='triangle_multiplication_incoming')(z, mask)
    return z + PairTransition(self.num_channels)(z)


class PairStack(hk.Module):
  """A named stack of PairBlocks: `<name>/__layer_stack_no_per_layer/block/...`.

  Wrapped in a module on purpose. hk.experimental.layer_stack names its OWN
  scope `__layer_stack_no_per_layer` and uniquifies repeats as `_1`, `_2`, so
  three bare stacks would be told apart only by CONSTRUCTION ORDER -- the
  converter would have to know that lm_encoder is stack 0 and folding_trunk is
  stack 1. Nesting under a named module makes the path self-describing instead.

  Construct ONCE and call repeatedly: parcae runs the same trunk on every
  recycle, and building the block inside the loop gives each recycle its own
  independent weights (222 parameters against the converter's 149).
  """

  def __init__(self, num_channels, num_layer, *, name):
    super().__init__(name=name)
    self.num_channels = num_channels
    self.num_layer = num_layer

  def __call__(self, z, mask):
    if not self.num_layer:
      return z
    block = PairBlock(self.num_channels, name='block')
    return hk.experimental.layer_stack(self.num_layer)(
        lambda x: block(x, mask))(z)


def pair_stack(z, mask, num_channels, num_layer, name):
  return PairStack(num_channels, num_layer, name=name)(z, mask)


# ── SWA / 3D-RoPE atom transformer ──────────────────────────────────────────

def rope_inv_freq(n_pairs, base):
  return (1.0 / (base ** (np.arange(n_pairs, dtype=np.float32) / n_pairs))).astype(np.float32)


def build_rope(ref_pos, uid, head_dim, n_sp=2, n_uid=10, sp_base=20.0, uid_base=10000.0):
  """3D rotary from the reference conformer: 3 axes x n_sp, plus ref_space_uid."""
  fs = (ref_pos[..., None] * rope_inv_freq(n_sp, sp_base)).reshape(ref_pos.shape[0], -1)
  fu = uid[:, None] * rope_inv_freq(n_uid, uid_base)
  fr = jnp.concatenate([fs, fu], -1)
  half = head_dim // 2
  if fr.shape[-1] < half:
    fr = jnp.concatenate([fr, jnp.zeros((fr.shape[0], half - fr.shape[-1]))], -1)
  return jnp.cos(fr), jnp.sin(fr)


def apply_rope(x, cos, sin):
  """cos/sin are TILED ([c|c]) to pair with rotate_half's split-into-halves.

  Interleaving instead is silent: it reads corr 0.88 on the atom encoder, high
  enough to look like noise and low enough to ruin the fold.
  """
  ro = cos.shape[-1] * 2
  c = jnp.concatenate([cos, cos], -1)[:, None]
  s = jnp.concatenate([sin, sin], -1)[:, None]
  a, b = jnp.split(x[..., :ro], 2, axis=-1)
  rot = jnp.concatenate([-b, a], -1)
  return jnp.concatenate([x[..., :ro] * c + rot * s, x[..., ro:]], -1)


class SWAAtomBlock(hk.Module):
  """adaLN-Zero over non-affine RMSNorm + sliding-window 3D-RoPE attention."""

  def __init__(self, num_channels, num_heads, half_window, *, name='swa_block'):
    super().__init__(name=name)
    self.c = num_channels
    self.h = num_heads
    self.half_window = half_window

  def __call__(self, x, c, cos, sin, valid):
    mod = hm.Linear(6 * self.c, name='adaln')(jax.nn.silu(c))
    sh_a, sc_a, g_a, sh_f, sc_f, g_f = jnp.split(mod, 6, axis=-1)

    # inlined rather than a helper method: haiku scopes a submodule created in a
    # method under `~_<method>/`, which would put every attention weight at
    # blocks/~_attn/... and no longer match what the converter emits.
    xa = _rms(x) * (1 + sc_a) + sh_a
    n, d = xa.shape
    dh = d // self.h
    qkv = hm.Linear(3 * d, name='qkv')(xa).reshape(n, 3, self.h, dh)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q, k = apply_rope(_rms(q), cos, sin), apply_rope(_rms(k), cos, sin)
    rank = jnp.cumsum(valid) - 1              # window over RANK among valid atoms
    ok = (jnp.abs(rank[:, None] - rank[None, :]) <= self.half_window)
    ok = ok & (valid[:, None] > 0) & (valid[None, :] > 0)
    ok = ok | jnp.eye(n, dtype=bool)          # the diagonal is always allowed
    logits = jnp.where(ok[None], jnp.einsum('ihd,jhd->hij', q, k) * dh ** -0.5, -1e9)
    o = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(n, d)
    o = o * valid[:, None] * jax.nn.sigmoid(hm.Linear(d, name='attn_gate')(xa))
    x = x + g_a * hm.Linear(d, name='attn_out')(o)

    h = _rms(x) * (1 + sc_f) + sh_f
    up = hm.Linear(self._ffn_hidden(), name='ffn_up')(h)
    nh = up.shape[-1] // 2
    return x + g_f * hm.Linear(self.c, name='ffn_down')(
        jax.nn.silu(up[..., :nh]) * up[..., nh:])

  def _ffn_hidden(self):
    # SwiGLUFFN rounds up to a multiple of 256: ((2*(d//3)*2)+255)//256*256
    return 2 * (((2 * (self.c // 3) * 2) + 255) // 256 * 256) // 2 * 2


def atom_stack(q, c, cos, sin, valid, num_channels, num_heads, half_window,
               num_layer, name):
  def block(x):
    return SWAAtomBlock(num_channels, num_heads, half_window, name=name)(
        x, c, cos, sin, valid)
  return hk.experimental.layer_stack(num_layer)(block)(q)


def atom_features(f, mask):
  elem = jax.nn.one_hot(f['ref_element'].astype(jnp.int32), MAX_ATOMIC_NUMBER) * mask[:, None]
  chars = (jax.nn.one_hot(f['ref_atom_name_chars'].astype(jnp.int32), CHAR_VOCAB_SIZE)
           * mask[:, None, None]).reshape(-1, MAX_CHARS * CHAR_VOCAB_SIZE)
  return jnp.concatenate(
      [f['ref_pos'], f['ref_charge'][:, None], mask[:, None], elem, chars], -1)


def scatter_mean(x, seg, n, w):
  num = jax.ops.segment_sum(x * w[:, None], seg, n)
  den = jax.ops.segment_sum(w, seg, n)[:, None]
  return num / jnp.maximum(den, 1e-9)


class AtomEncoder(hk.Module):
  """atom_linear -> atom_norm -> [+ coords] -> SWA stack -> relu -> mean scatter."""

  def __init__(self, c_atom, out_dim, num_blocks, num_heads, half_window,
               *, structure=False, name='atom_encoder'):
    super().__init__(name=name)
    self.c_atom, self.out_dim = c_atom, out_dim
    self.num_blocks, self.num_heads, self.half_window = num_blocks, num_heads, half_window
    self.structure = structure

  def __call__(self, f, mask, a2t, num_tokens, r_noisy=None):
    c0 = hm.LayerNorm(name='atom_norm')(
        hm.Linear(self.c_atom, name='atom_linear')(atom_features(f, mask)))
    cos, sin = build_rope(f['ref_pos'], f['ref_space_uid'], self.c_atom // self.num_heads)
    q = c0
    if self.structure and r_noisy is not None:
      pair = jnp.concatenate([r_noisy, jnp.zeros_like(r_noisy)], -1)
      q = q + hm.Linear(self.c_atom, name='coords_linear')(pair)
    q = atom_stack(q, c0, cos, sin, mask, self.c_atom, self.num_heads,
                   self.half_window, self.num_blocks, 'blocks')
    a = scatter_mean(jax.nn.relu(hm.Linear(self.out_dim, name='atom_to_token')(q)),
                     a2t, num_tokens, mask)
    return a, q, c0, (cos, sin)


# ── the LM shim: 81 ESM-C hidden states -> a pair representation ─────────────

class LanguageModelShim(hk.Module):
  """Learned softmax mix over ESM-C's layers, then SingleToPair.

  The mix peaks on the LAST layers (79/80/78 hold 58% of the mass), so the tower
  cannot be truncated -- but only ~31 of 81 states carry 99%, which bounds what
  has to be materialised.
  """

  def __init__(self, c_z, *, name='language_model'):
    super().__init__(name=name)
    self.c_z = c_z

  def __call__(self, hidden):
    n_layers = hidden.shape[-2]
    combine = hk.get_parameter('combine', [n_layers], init=hk.initializers.Constant(0.))
    x = hm.LayerNorm(name='lm_norm')(hidden)
    x = hm.Linear(self.c_z, name='lm_projection')(x)
    x = jnp.einsum('k,lkc->lc', combine, x)          # combine is pre-softmaxed
    x = hm.Linear(self.c_z, use_bias=True, name='downproject')(x)
    z = jnp.concatenate([x[:, None] * x[None, :], x[:, None] - x[None, :]], -1)
    z = hm.Linear(self.c_z, use_bias=True, name='pair_mlp_1')(z)
    z = jax.nn.gelu(z, approximate=False)            # torch nn.GELU is exact
    z = hm.Linear(self.c_z, use_bias=True, name='pair_mlp_2')(z)
    return hm.LayerNorm(name='pair_norm')(z)


# ── MSA encoder (token-major [L, M, c], unlike AF3's [M, L, c]) ──────────────

class OuterProductMean(hk.Module):
  """NOTE the divide order: Wout(outer) / n_valid, so the BIAS is scaled too."""

  def __init__(self, c_hidden, c_z, *, name='outer_product_mean'):
    super().__init__(name=name)
    self.c_hidden, self.c_z = c_hidden, c_z

  def __call__(self, m, mmask):
    mn = hm.LayerNorm(name='layer_norm')(m)
    a = hm.Linear(self.c_hidden, name='left_projection')(mn) * mmask[..., None]
    b = hm.Linear(self.c_hidden, name='right_projection')(mn) * mmask[..., None]
    n_valid = jnp.maximum(mmask @ mmask.T, 1.0)[..., None]
    outer = jnp.einsum('imc,jmd->ijcd', a, b).reshape(a.shape[0], b.shape[0], -1)
    return hm.Linear(self.c_z, use_bias=True, name='output')(outer) / n_valid


class MSAPairWeightedAveraging(hk.Module):

  def __init__(self, num_heads, head_width, c_m, *, name='msa_pair_weighted_averaging'):
    super().__init__(name=name)
    self.h, self.dh, self.c_m = num_heads, head_width, c_m

  def __call__(self, m, z, pair_mask):
    mn = hm.LayerNorm(name='msa_norm')(m)
    zn = hm.LayerNorm(name='pair_norm')(z)
    bias = jnp.where(pair_mask[..., None] > 0, hm.Linear(self.h, name='bias')(zn), -1e5)
    attn = jax.nn.softmax(bias, axis=-2)             # over the second token axis
    ell, mm = m.shape[0], m.shape[1]
    v = hm.Linear(self.h * self.dh, name='value')(mn).reshape(ell, mm, self.h, self.dh)
    g = jax.nn.sigmoid(hm.Linear(self.h * self.dh, name='gate')(mn)).reshape(
        ell, mm, self.h, self.dh)
    o = jnp.einsum('ijh,jmhd,imhd->imhd', attn, v, g)
    return hm.Linear(self.c_m, name='output')(o.reshape(ell, mm, self.h * self.dh))


class MSABlock(hk.Module):
  """OPM into pair, then (unless final) the MSA update, then the pair block."""

  def __init__(self, cfg, is_final, *, name='msa_block'):
    super().__init__(name=name)
    self.cfg, self.is_final = cfg, is_final

  def __call__(self, m, z, mmask, pair_mask):
    c = self.cfg
    z = z + OuterProductMean(c['c_hidden'], c['c_z'])(m, mmask)
    if not self.is_final:
      m = m + MSAPairWeightedAveraging(c['msa_heads'], c['msa_head_width'],
                                       c['c_m'])(m, z, pair_mask)
      m = m + PairTransition(c['c_m'], name='msa_transition')(m)
    # inlined so the tri-mul/transition scopes sit directly under the block,
    # matching what the converter emits for msa_encoder blocks
    z = _pair_block_inline(z, pair_mask, c['c_z'])
    return m, z


def _pair_block_inline(z, mask, c_z):
  z = z + TriangleMultiplication(c_z, outgoing=True,
                                 name='triangle_multiplication_outgoing')(z, mask)
  z = z + TriangleMultiplication(c_z, outgoing=False,
                                 name='triangle_multiplication_incoming')(z, mask)
  return z + PairTransition(c_z)(z)


class MSAEncoder(hk.Module):
  """The last block is pair-only: its MSA-update params are absent upstream."""

  def __init__(self, cfg, num_layer, *, name='msa_encoder'):
    super().__init__(name=name)
    self.cfg, self.num_layer = cfg, num_layer

  def __call__(self, z, s_inputs, msa_oh, has_deletion, deletion_value, mmask):
    c = self.cfg
    feat = jnp.concatenate([msa_oh, has_deletion[..., None], deletion_value[..., None]], -1)
    m = (hm.Linear(c['c_m'], name='embed')(feat)
         + hm.Linear(c['c_m'], name='project_inputs')(s_inputs)[:, None])
    tok = mmask[:, 0]
    pair_mask = tok[:, None] * tok[None, :]

    def block(x):
      return MSABlock(c, False, name='blocks')(x[0], x[1], mmask, pair_mask)

    if self.num_layer > 1:
      m, z = hk.experimental.layer_stack(self.num_layer - 1)(block)((m, z))
    _, z = MSABlock(c, True, name='final_block')(m, z, mmask, pair_mask)
    return z


# ── diffusion ───────────────────────────────────────────────────────────────

class AdaptiveLayerNorm(hk.Module):
  """s is normalised with a learned SCALE and NO offset; the gate carries a bias."""

  def __init__(self, *, name='adaln'):
    super().__init__(name=name)

  def __call__(self, a, s):
    a_n = hm.LayerNorm(name='a_norm', create_scale=False, create_offset=False)(a)
    s_n = hm.LayerNorm(name='s_norm', create_offset=False)(s)
    gate = jax.nn.sigmoid(hm.Linear(a.shape[-1], use_bias=True, name='gate')(s_n))
    return gate * a_n + hm.Linear(a.shape[-1], name='shift')(s_n)


class DiffusionAttention(hk.Module):

  def __init__(self, num_heads, *, name='token_attn'):
    super().__init__(name=name)
    self.h = num_heads

  def __call__(self, a, s, z):
    ell, d = a.shape
    dh = d // self.h
    x = AdaptiveLayerNorm()(a, s)
    q = hm.Linear(d, use_bias=True, name='q')(x).reshape(ell, self.h, dh)
    kv = hm.Linear(2 * d, name='kv')(x)
    k, v = jnp.split(kv, 2, -1)
    bias = hm.Linear(self.h, name='pair_bias')(hm.LayerNorm(name='pair_norm')(z))
    logits = jnp.einsum('ihd,jhd->ijh', q, k.reshape(ell, self.h, dh)) * dh ** -0.5 + bias
    ctx = jnp.einsum('ijh,jhd->ihd', jax.nn.softmax(logits, axis=-2),
                     v.reshape(ell, self.h, dh))
    g = jax.nn.sigmoid(hm.Linear(d, name='g')(x)).reshape(ell, self.h, dh)
    out = hm.Linear(d, name='out')((g * ctx).reshape(ell, d))
    return jax.nn.sigmoid(hm.Linear(d, use_bias=True, name='out_gate')(s)) * out


class DiffusionTransition(hk.Module):

  def __init__(self, multiplier=2, *, name='token_transition'):
    super().__init__(name=name)
    self.multiplier = multiplier

  def __call__(self, a, s):
    d = a.shape[-1]
    x = AdaptiveLayerNorm()(a, s)
    sw = hm.Linear(2 * self.multiplier * d, name='swish')(x)
    n = sw.shape[-1] // 2
    out = hm.Linear(d, name='out')(jax.nn.silu(sw[..., :n]) * sw[..., n:])
    return jax.nn.sigmoid(hm.Linear(d, use_bias=True, name='out_gate')(s)) * out


class DiffusionConditioning(hk.Module):
  """No s_trunk anywhere: s comes from s_inputs alone, the trunk only via z."""

  def __init__(self, cfg, *, name='conditioning'):
    super().__init__(name=name)
    self.cfg = cfg

  def __call__(self, s_inputs, z_trunk, rel_pos, t_hat):
    c = self.cfg
    z = jnp.concatenate([z_trunk, rel_pos], -1)
    z = hm.Linear(c['c_z'], name='z_projection')(hm.LayerNorm(name='z_input_norm')(z))
    for i in range(c['n_z_transitions']):
      z = z + _block_transition(z, c['c_z'], 'z_transitions', i)
    s = hm.Linear(c['c_token'], name='s_projection')(
        hm.LayerNorm(name='s_input_norm')(s_inputs))
    t_noise = 0.25 * jnp.log(jnp.maximum(t_hat / SIGMA_DATA, 1e-20))
    w = hk.get_parameter('fourier_w', [c['fourier_dim']], init=jnp.zeros)
    b = hk.get_parameter('fourier_b', [c['fourier_dim']], init=jnp.zeros)
    n = jnp.cos(2.0 * jnp.pi * (t_noise * w + b))
    n = hm.Linear(c['c_token'], name='noise_projection')(
        hm.LayerNorm(name='noise_norm')(n))
    s = s + n[None]
    for i in range(c['n_s_transitions']):
      s = s + _block_transition(s, c['c_token'], 's_transitions', i)
    return s, z


def _block_transition(x, c, stack_name, i):
  """A stacked, unconditioned SwiGLU transition with SEPARATE a/b projections.

  The diffusion conditioning transitions use a_proj/b_proj rather than the
  trunk's fused ffn.w12, which is why the converter carries a second dialect.
  """
  scope = '%s_%d' % (stack_name, i)
  h = hm.LayerNorm(name='%s/input_layer_norm' % scope)(x)
  h = hm.Linear(2 * 2 * c, name='%s/transition1' % scope)(h)
  n = h.shape[-1] // 2
  return hm.Linear(c, name='%s/transition2' % scope)(
      jax.nn.silu(h[..., :n]) * h[..., n:])


class ConfidenceHead(hk.Module):
  """Reads the PREDICTED structure: distance bins of the sampled coordinates."""

  def __init__(self, cfg, *, name='confidence'):
    super().__init__(name=name)
    self.cfg = cfg

  def __call__(self, s_inputs, z, rel_pos, token_bonds_enc, x_pred, rep_idx,
               a2t, intra_idx, tok_mask):
    c = self.cfg
    sn = hm.LayerNorm(name='s_inputs_norm')(s_inputs)
    zz = hm.LayerNorm(name='z_norm')(z) + rel_pos + token_bonds_enc
    zz = zz + hm.Linear(c['c_z'], name='s_to_z')(sn)[:, None]
    zz = zz + hm.Linear(c['c_z'], name='s_to_z_transpose')(sn)[None, :]
    zz = zz + hm.Linear(c['c_z'], name='s_to_z_prod_out')(
        hm.Linear(c['c_z'], name='s_to_z_prod_in1')(sn)[:, None]
        * hm.Linear(c['c_z'], name='s_to_z_prod_in2')(sn)[None, :])
    rep = x_pred[rep_idx]
    dist = jnp.sqrt(jnp.maximum(
        ((rep[:, None] - rep[None, :]) ** 2).sum(-1), 1e-10))
    bounds = hk.get_parameter('boundaries', [c['num_dist_bins'] - 1], init=jnp.zeros)
    bins = (dist[..., None] > bounds).sum(-1)
    emb = hk.get_parameter('dist_bin_embed', [c['num_dist_bins'], c['c_z']],
                           init=jnp.zeros)
    zz = zz + emb[bins]
    pm = tok_mask[:, None] * tok_mask[None, :]
    # ONE MORE residual around the WHOLE stack, on top of the per-block ones
    zz = zz + pair_stack(zz, pm, c['c_z'], c['n_blocks'], 'folding_trunk')
    scores = jnp.where(tok_mask[None, :] > 0,
                       hm.Linear(1, name='row_pool_attn')(zz)[..., 0], -1e9)
    single = hm.Linear(c['c_single'], name='row_pool_out')(
        jnp.einsum('nm,nmd->nd', jax.nn.softmax(scores, -1), zz))
    s_at = single[a2t]
    plddt_w = hk.get_parameter('plddt_weight',
                               [c['max_atoms_per_token'], c['c_single'], c['num_plddt_bins']],
                               init=jnp.zeros)
    res_w = hk.get_parameter('resolved_weight',
                             [c['max_atoms_per_token'], c['c_single'], 2], init=jnp.zeros)
    out = {
        'pae_logits': hm.Linear(c['num_pae_bins'], name='pae')(
            hm.LayerNorm(name='pae_norm')(zz)),
        'pde_logits': hm.Linear(c['num_pde_bins'], name='pde')(
            hm.LayerNorm(name='pde_norm')(zz)),
        'plddt_logits': jnp.einsum(
            'ac,acb->ab', hm.LayerNorm(name='plddt_norm')(s_at), plddt_w[intra_idx]),
        'resolved_logits': jnp.einsum(
            'ac,acb->ab', hm.LayerNorm(name='resolved_norm')(s_at), res_w[intra_idx]),
    }
    return out


class DiffusionModule(hk.Module):
  """One EDM denoise step. Reports r_update so the mix cannot flatter it."""

  def __init__(self, cfg, *, name='diffusion'):
    super().__init__(name=name)
    self.cfg = cfg

  def __call__(self, x_noisy, t_hat, f, mask, a2t, num_tokens,
               s_inputs, z_trunk, rel_pos):
    c = self.cfg
    s, z = DiffusionConditioning(c)(s_inputs, z_trunk, rel_pos, t_hat)
    r_noisy = x_noisy / jnp.sqrt(t_hat ** 2 + SIGMA_DATA ** 2)          # c_in
    enc = AtomEncoder(c['c_atom'], c['c_token'], c['n_atom_blocks'],
                      c['atom_heads'], c['half_window'], structure=True)
    a, q, c0, (cos, sin) = enc(f, mask, a2t, num_tokens, r_noisy=r_noisy)
    a = a + hm.Linear(c['c_token'], name='s_to_token')(
        hm.LayerNorm(name='s_step_norm')(s))
    for i in range(c['n_token_blocks']):
      a = a + DiffusionAttention(c['token_heads'], name='token_attn_%d' % i)(a, s, z)
      a = a + DiffusionTransition(name='token_transition_%d' % i)(a, s)
    a = hm.LayerNorm(name='token_norm')(a)
    qd = q + hm.Linear(c['c_atom'], name='atom_decoder/token_to_atom')(a)[a2t]
    qd = atom_stack(qd, c0, cos, sin, mask, c['c_atom'], c['atom_heads'],
                    c['half_window'], c['n_atom_blocks'], 'atom_decoder/blocks')
    r_update = hm.Linear(3, name='atom_decoder/output')(
        hm.LayerNorm(name='atom_decoder/norm')(qd))
    s2, t2 = SIGMA_DATA ** 2, t_hat ** 2
    return (s2 / (s2 + t2)) * x_noisy + (SIGMA_DATA * t_hat / jnp.sqrt(s2 + t2)) * r_update


class ESMFold2(hk.Module):
  """features + ESM-C hidden states -> trunk pair, distogram, and a denoiser."""

  class Config(base_config.BaseConfig):
    c_z: int = 256
    c_single: int = 384
    c_atom: int = 128
    c_token: int = 768
    s_inputs: int = 451
    n_trunk: int = 48
    n_lm_encoder: int = 4
    n_coda: int = 2
    n_msa: int = 4
    n_input_atom: int = 3
    num_bins: int = 64
    n_loops: int = 3
    lm_dropout: float = 0.25

  def __init__(self, config, global_config, *, name='esmfold2'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, f, lm_hidden, rel_pos_feats, msa=None, key=None):
    c = self.config
    mask = f['atom_attention_mask']
    num_tokens = f['res_type'].shape[0]
    a2t = f['atom_to_token'].astype(jnp.int32) * mask.astype(jnp.int32)

    a, _, _, _ = AtomEncoder(c.c_atom, c.c_token // 2, c.n_input_atom, 4, 64,
                             name='inputs_embedder')(f, mask, a2t, num_tokens)
    res_oh = jax.nn.one_hot(f['res_type'].astype(jnp.int32), NUM_RES_TYPES) \
        * f['token_attention_mask'][:, None]
    profile = res_oh if msa is None else msa['profile']
    s_inputs = jnp.concatenate(
        [a, res_oh, profile, jnp.zeros((num_tokens, 1))], -1)

    rel_pos = hm.Linear(c.c_z, name='rel_pos')(rel_pos_feats)
    z_init = (hm.Linear(c.c_z, name='z_init_1')(s_inputs)[:, None]
              + hm.Linear(c.c_z, name='z_init_2')(s_inputs)[None, :]
              + rel_pos
              + hm.Linear(c.c_z, name='token_bonds')(f['token_bonds']))

    pm = jnp.ones((num_tokens, num_tokens))
    lm_z = LanguageModelShim(c.c_z)(lm_hidden)

    # every module the recycle loop reuses is constructed ONCE, here
    lm_encoder = PairStack(c.c_z, c.n_lm_encoder, name='lm_encoder')
    trunk = PairStack(c.c_z, c.n_trunk, name='folding_trunk')
    parcae_norm = hm.LayerNorm(name='parcae_input_norm')
    parcae_b = hm.Linear(c.c_z, name='parcae_b')
    msa_encoder = MSAEncoder(msa['cfg'], c.n_msa) if msa is not None else None
    a_vec = hk.get_parameter('parcae_a', [c.c_z], init=jnp.zeros)
    key = jax.random.PRNGKey(0) if key is None else key
    key, k_init = jax.random.split(key)
    std = float(np.sqrt(2.0 / (5.0 * c.c_z)))
    z = jax.random.truncated_normal(k_init, -3.0, 3.0, z_init.shape) * std
    for _ in range(c.n_loops + 1):
      lm_i = lm_z
      if c.lm_dropout > 0:
        # 25% dropout on the LM pair rep EVERY loop, at INFERENCE -- deliberate
        # stochastic ensembling, and worth ~18 A when wrongly disabled.
        key, k_do = jax.random.split(key)
        keep = jax.random.bernoulli(k_do, 1.0 - c.lm_dropout, lm_z.shape)
        lm_i = lm_z * keep / (1.0 - c.lm_dropout)
      lm_ref = lm_encoder(lm_i, pm)
      z_inject = z_init
      if msa is not None:
        # OVERWRITE, not add: msa_encoder_overwrite=True. Removing this costs
        # ~18 A even on a depth-1 self MSA.
        z_inject = msa_encoder(
            z_inject, s_inputs, msa['oh'], msa['has_deletion'],
            msa['deletion_value'], msa['mask'])
      inj = parcae_norm(z_inject + lm_ref)
      z = a_vec * z + parcae_b(inj)
      z = trunk(z, pm)
    z = pair_stack(hm.Linear(c.c_z, name='parcae_readout')(z), pm, c.c_z,
                   c.n_coda, 'parcae_coda')
    logits = hm.Linear(c.num_bins, use_bias=True, name='distogram')(
        z + jnp.swapaxes(z, 0, 1))
    return {'pair': z, 's_inputs': s_inputs, 'rel_pos': rel_pos,
            'distogram_logits': logits}
