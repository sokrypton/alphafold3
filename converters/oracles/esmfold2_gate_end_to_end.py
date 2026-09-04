import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'/home/ubuntu/alphafold3')
from converters import esmfold2 as E

def rms(x):                                       # affine-free RMSNorm
    # torch F.rms_norm(eps=None) uses finfo(dtype).eps, NOT 1e-5
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + np.finfo(np.float32).eps)

def layer_norm(x, p, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + eps) * p['scale'] + p['offset']

def rotate_half(x):
    a, b = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-b, a], -1)

def apply_rope(x, cos, sin):                      # x [N,H,D]
    ro = cos.shape[-1] * 2
    # native: cos.unsqueeze(2).repeat(1,1,1,2) TILES [c|c]; it pairs with
    # rotate_half's split-into-halves.  Interleaving here is a silent 0.88-corr bug.
    c = jnp.concatenate([cos, cos], -1)[:, None]
    s = jnp.concatenate([sin, sin], -1)[:, None]
    return jnp.concatenate([x[..., :ro] * c + rotate_half(x[..., :ro]) * s, x[..., ro:]], -1)

def build_rope(ref_pos, uid, head_dim, n_sp=2, n_uid=10, sp_base=20.0, uid_base=10000.0):
    sp = E.rope_inv_freq(n_sp, sp_base); ui = E.rope_inv_freq(n_uid, uid_base)
    fs = (ref_pos[..., None] * sp).reshape(ref_pos.shape[0], -1)     # [N, 3*n_sp]
    fu = uid[:, None] * ui                                           # [N, n_uid]
    fr = jnp.concatenate([fs, fu], -1)
    half = head_dim // 2
    if fr.shape[-1] < half:
        fr = jnp.concatenate([fr, jnp.zeros((fr.shape[0], half - fr.shape[-1]))], -1)
    # native casts cos/sin to bfloat16 even on the fp32 path
    return jnp.cos(fr), jnp.sin(fr)

def swa_attn(x, q_p, cos, sin, valid, half_window, n_heads):
    N, d = x.shape
    dh = d // n_heads
    qkv = (x @ q_p['qkv/weights']).reshape(N, 3, n_heads, dh)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q, k = rms(q), rms(k)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rank = jnp.cumsum(valid) - 1
    allowed = (jnp.abs(rank[:, None] - rank[None, :]) <= half_window) \
              & (valid[:, None] > 0) & (valid[None, :] > 0)
    allowed = allowed | jnp.eye(N, dtype=bool)
    logits = jnp.einsum('ihd,jhd->hij', q, k) * (dh ** -0.5)
    logits = jnp.where(allowed[None], logits, -1e9)
    o = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(N, d)
    o = o * valid[:, None]
    o = o * jax.nn.sigmoid(x @ q_p['attn_gate/weights'])
    return o @ q_p['attn_out/weights']

def swa_block(x, c, q_p, cos, sin, valid, half_window, n_heads):
    mod = jax.nn.silu(c) @ q_p['adaln/weights']
    sh_a, sc_a, g_a, sh_f, sc_f, g_f = jnp.split(mod, 6, axis=-1)
    x = x + g_a * swa_attn(rms(x) * (1 + sc_a) + sh_a, q_p, cos, sin, valid, half_window, n_heads)
    h = rms(x) * (1 + sc_f) + sh_f
    up = h @ q_p['ffn_up/weights']
    n = up.shape[-1] // 2
    x = x + g_f * ((jax.nn.silu(up[..., :n]) * up[..., n:]) @ q_p['ffn_down/weights'])
    return x

def tri_mul(z, p, mask, outgoing):
    zn = layer_norm(z, {'scale': p['left_norm_input/scale'], 'offset': p['left_norm_input/offset']})
    proj = zn @ p['projection/weights']          # (..., 2h) interleaved a/b
    gate = jax.nn.sigmoid(zn @ p['gate/weights'])
    routed = proj * gate * mask[..., None]
    a = routed[..., 0::2]                        # interleave convention
    b = routed[..., 1::2]
    if outgoing:
        c = jnp.einsum('ikd,jkd->ijd', a, b)
    else:
        c = jnp.einsum('kjd,kid->ijd', a, b)   # AF3's incoming equation (weights pre-swapped)
    c = layer_norm(c, {'scale': p['center_norm/scale'], 'offset': p['center_norm/offset']})
    out = c @ p['output_projection/weights']
    return out * jax.nn.sigmoid(zn @ p['gating_linear/weights'])

def transition(x, p):
    h = layer_norm(x, {'scale': p['input_layer_norm/scale'], 'offset': p['input_layer_norm/offset']})
    h = h @ p['transition1/weights']
    n = h.shape[-1] // 2
    return (jax.nn.silu(h[..., :n]) * h[..., n:]) @ p['transition2/weights']

def pair_block(z, p, mask):
    z = z + tri_mul(z, {k[len('triangle_multiplication_outgoing/'):]: v
                        for k, v in p.items() if k.startswith('triangle_multiplication_outgoing/')}, mask, True)
    z = z + tri_mul(z, {k[len('triangle_multiplication_incoming/'):]: v
                        for k, v in p.items() if k.startswith('triangle_multiplication_incoming/')}, mask, False)
    z = z + transition(z, {k[len('pair_transition/'):]: v
                           for k, v in p.items() if k.startswith('pair_transition/')})
    return z


def run_stack(z, p, pre, n, mask):
    for i in range(n):
        z = pair_block(z, {k[len(pre):]: v[i] for k, v in p.items() if k.startswith(pre)}, mask)
    return z

def lm_shim(hidden, p):
    """[L, 81, 2560] ESM-C hidden states -> [L, L, 256] pair."""
    x = layer_norm(hidden, {'scale': p['language_model/lm_norm/scale'],
                            'offset': p['language_model/lm_norm/offset']})
    x = x @ p['language_model/lm_projection/weights']                 # [L,81,256]
    x = jnp.einsum('k,lkc->lc', p['language_model/combine'], x)       # learned layer mix
    x = x @ p['language_model/downproject/weights'] + p['language_model/downproject/bias']
    pair = jnp.concatenate([x[:, None] * x[None, :], x[:, None] - x[None, :]], -1)
    pair = pair @ p['language_model/pair_mlp_1/weights'] + p['language_model/pair_mlp_1/bias']
    pair = jax.nn.gelu(pair, approximate=False)                       # torch nn.GELU is exact
    pair = pair @ p['language_model/pair_mlp_2/weights'] + p['language_model/pair_mlp_2/bias']
    return layer_norm(pair, {'scale': p['language_model/pair_norm/scale'],
                             'offset': p['language_model/pair_norm/offset']})



NRB, NCB = 32, 2
def rel_pos_feats(residue_index, asym_id, sym_id, entity_id, token_index):
    same_chain = asym_id[:, None] == asym_id[None, :]
    same_res   = residue_index[:, None] == residue_index[None, :]
    same_ent   = entity_id[:, None] == entity_id[None, :]
    d_res = np.clip(residue_index[:, None] - residue_index[None, :] + NRB, 0, 2*NRB)
    d_res = np.where(same_chain, d_res, 2*NRB + 1)
    d_tok = np.clip(token_index[:, None] - token_index[None, :] + NRB, 0, 2*NRB)
    d_tok = np.where(same_chain & same_res, d_tok, 2*NRB + 1)
    d_ch  = np.clip(sym_id[:, None] - sym_id[None, :] + NCB, 0, 2*NCB)
    d_ch  = np.where(same_chain, 2*NCB + 1, d_ch)      # NOTE: inverted vs the others
    oh = lambda x, n: np.eye(n, dtype=np.float32)[x]
    # concat order is [rel_pos, rel_token, same_entity, rel_chain] -- entity
    # BEFORE chain, which the class docstring's arithmetic does not tell you
    return np.concatenate([oh(d_res, 2*NRB+2), oh(d_tok, 2*NRB+2),
                           same_ent.astype(np.float32)[..., None], oh(d_ch, 2*NCB+2)], -1)

sd = dict(np.load('esmfold2_sd.npz'))
f  = dict(np.load('esmfold2_dump.npz'))
r  = dict(np.load('esmfold2_det_fp32.npz'))
dims = E.derive_dims(sd)
p = E.map_trunk(sd, dims)
pa = E.atom_encoder(sd, 'inputs_embedder.atom_attention_encoder', 3)

def rep(tag, ours, native, warn=3e-2):
    ours=np.asarray(ours,np.float64); native=np.asarray(np.squeeze(native),np.float64)
    e=np.abs(ours-native).max()/max(np.abs(native).max(),1e-9)
    c=np.corrcoef(ours.ravel(),native.ravel())[0,1]
    print("%-36s relerr %.3e  corr %.8f %s"%(tag,e,c,'' if e<warn else ' <-- CHECK')); return c

# ---- atom encoder -> a -----------------------------------------------------
MAX_Z, CHARV, MAXC = 128, 64, 4
ref_pos = jnp.asarray(f['feat.ref_pos'][0]); mask = jnp.asarray(f['feat.atom_attention_mask'][0])
elem  = jax.nn.one_hot(f['feat.ref_element'][0].astype(int), MAX_Z)*mask[:,None]
chars = (jax.nn.one_hot(f['feat.ref_atom_name_chars'][0].astype(int), CHARV)*mask[:,None,None]).reshape(-1, MAXC*CHARV)
a2t = f['feat.atom_to_token'][0].astype(int)*f['feat.atom_attention_mask'][0].astype(int)
atom_feats = jnp.concatenate([ref_pos, jnp.asarray(f['feat.ref_charge'][0])[:,None], mask[:,None], elem, chars], -1)
c0 = layer_norm(atom_feats @ pa['atom_linear/weights'], {'scale':pa['atom_norm/scale'],'offset':pa['atom_norm/offset']})
cos,sin = build_rope(ref_pos, jnp.asarray(f['feat.ref_space_uid'][0]), c0.shape[-1]//4)
q = c0
for i in range(3):
    q = swa_block(q, c0, {k[len('blocks/'):]:v[i] for k,v in pa.items() if k.startswith('blocks/')}, cos,sin,mask,64,4)
qa = jax.nn.relu(q @ pa['atom_to_token/weights'])
L = r['x_inputs.0'].shape[1]
num = jax.ops.segment_sum(qa*mask[:,None], jnp.asarray(a2t), L); den = jax.ops.segment_sum(mask, jnp.asarray(a2t), L)[:,None]
a = num/jnp.maximum(den,1e-9)

res_oh = jax.nn.one_hot(f['feat.res_type'][0].astype(int), 33)*jnp.asarray(f['feat.token_attention_mask'][0])[:,None]
x_in = jnp.concatenate([a, res_oh, res_oh, jnp.zeros((L,1))], -1)   # no MSA -> profile = res_type_oh
rep('x_inputs (FROM FEATURES)', x_in, r['x_inputs.0'])

# ---- rel_pos ---------------------------------------------------------------
rp = jnp.asarray(rel_pos_feats(f['feat.residue_index'][0].astype(int), f['feat.asym_id'][0].astype(int),
                               f['feat.sym_id'][0].astype(int), f['feat.entity_id'][0].astype(int),
                               f['feat.token_index'][0].astype(int))) @ p['rel_pos/weights']
rep('rel_pos (FROM FEATURES)', rp, r['rel_pos.0'])

# ---- the whole trunk -------------------------------------------------------
mask2 = jnp.ones((L,L))
tb = jnp.asarray(f['feat.token_bonds'][0]) @ p['token_bonds/weights']
z_init = (x_in @ p['z_init_1/weights'])[:,None] + (x_in @ p['z_init_2/weights'])[None,:] + rp + tb
lm_z = lm_shim(jnp.asarray(r['lm_hidden'][0]), p)
lm_ref = run_stack(lm_z, p, 'lm_encoder/', dims['n_lm_encoder'], mask2)
z = jnp.asarray(r['parcae.init_state'][0])
av = jnp.asarray(p['parcae_a']); bT = jnp.asarray(p['parcae_b/weights'])
for i in range(4):
    inj = layer_norm(z_init+lm_ref, {'scale':p['parcae_input_norm/scale'],'offset':p['parcae_input_norm/offset']})
    z = av*z + inj@bT
    z = run_stack(z, p, 'folding_trunk/', dims['n_trunk'], mask2)
z = z @ p['parcae_readout/weights']
z = run_stack(z, p, 'parcae_coda/', dims['n_coda'], mask2)
logits = (z + z.transpose(1,0,2)) @ p['distogram/weights'] + p['distogram/bias']
print()
c = rep('distogram (FEATURES -> LOGITS)', logits, r['out.distogram_logits'], warn=1e-1)
print("\nFULL TRUNK FROM FEATURES: corr %.6f" % c)
