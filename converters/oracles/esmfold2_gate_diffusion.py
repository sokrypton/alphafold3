import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'/home/ubuntu/alphafold3')
from converters import esmfold2 as E

def rms(x):                                       # affine-free RMSNorm
    # torch F.rms_norm(eps=None) uses finfo(dtype).eps, NOT 1e-5
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + np.finfo(np.float32).eps)

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


def layer_norm(x, s, o, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + eps) * s + o

def adaln(a, s, q):
    a_n = layer_norm(a, 1.0, 0.0)
    s_n = layer_norm(s, q['adaln/s_norm/scale'], 0.0)
    return jax.nn.sigmoid(s_n @ q['adaln/gate/weights'] + q['adaln/gate/bias']) * a_n \
           + s_n @ q['adaln/shift/weights']

def attn_block(a, s, z, q, n_heads):
    L, D = a.shape; dh = D // n_heads
    x = adaln(a, s, q)
    qq = (x @ q['q/weights'] + q['q/bias']).reshape(L, n_heads, dh)
    kv = x @ q['kv/weights']; k, v = jnp.split(kv, 2, -1)
    k = k.reshape(L, n_heads, dh); v = v.reshape(L, n_heads, dh)
    bias = layer_norm(z, q['pair_norm/scale'], q['pair_norm/offset']) @ q['pair_bias/weights']
    logits = jnp.einsum('ihd,jhd->ijh', qq, k) * (dh ** -0.5) + bias
    attn = jax.nn.softmax(logits, axis=-2)
    ctx = jnp.einsum('ijh,jhd->ihd', attn, v)
    g = jax.nn.sigmoid(x @ q['g/weights']).reshape(L, n_heads, dh)
    out = (g * ctx).reshape(L, D) @ q['out/weights']
    return jax.nn.sigmoid(s @ q['out_gate/weights'] + q['out_gate/bias']) * out

def trans_block(a, s, q):
    x = adaln(a, s, q)
    sw = x @ q['swish/weights']; n = sw.shape[-1] // 2
    out = (jax.nn.silu(sw[..., :n]) * sw[..., n:]) @ q['out/weights']
    return jax.nn.sigmoid(s @ q['out_gate/weights'] + q['out_gate/bias']) * out

sd = dict(np.load('esmfold2_sd.npz'))
r  = dict(np.load('esmfold2_diff.npz'))
dims = E.derive_dims(sd)
P = E.map_diffusion(sd, dims)
pick = lambda pre, src=None: {k[len(pre):]: v for k, v in (src or P).items() if k.startswith(pre)}
idx  = lambda q, i: {k: v[i] for k, v in q.items()}

def rep(tag, ours, native, warn=3e-2):
    ours=np.asarray(ours,np.float64); native=np.asarray(np.squeeze(native),np.float64)
    e=np.abs(ours-native).max()/max(np.abs(native).max(),1e-9)
    cc=np.corrcoef(ours.ravel(),native.ravel())[0,1]
    print("%-32s relerr %.3e  corr %.8f %s"%(tag,e,cc,'' if e<warn else ' <-- CHECK')); return cc

# ---- conditioning ----------------------------------------------------------
c = pick('conditioning/')
SIGMA = 16.0
t_hat = float(r['in_t_hat'][0])
z = jnp.concatenate([jnp.asarray(r['in_z_trunk'][0]), jnp.asarray(r['in_relative_position_encoding'][0])], -1)
z = layer_norm(z, c['z_input_norm/scale'], c['z_input_norm/offset']) @ c['z_projection/weights']
def tr(x, q, i):
    h = layer_norm(x, q['input_layer_norm/scale'][i], q['input_layer_norm/offset'][i]) @ q['transition1/weights'][i]
    n = h.shape[-1]//2
    return (jax.nn.silu(h[..., :n]) * h[..., n:]) @ q['transition2/weights'][i]
zt = pick('z_transitions/', c)
for i in range(zt['transition1/weights'].shape[0]): z = z + tr(z, zt, i)
s = layer_norm(jnp.asarray(r['in_s_inputs'][0]), c['s_input_norm/scale'], c['s_input_norm/offset']) @ c['s_projection/weights']
t_noise = 0.25*np.log(max(t_hat/SIGMA, 1e-20))
n_emb = jnp.cos(2*jnp.pi*(t_noise*c['fourier_w'] + c['fourier_b']))
s = s + (layer_norm(n_emb, c['noise_norm/scale'], c['noise_norm/offset']) @ c['noise_projection/weights'])[None]
st = pick('s_transitions/', c)
for i in range(st['transition1/weights'].shape[0]): s = s + tr(s, st, i)

# ---- atom encoder ----------------------------------------------------------
ae = pick('atom_encoder/')
mask = jnp.asarray(r['in_ref_mask'][0]); ref_pos = jnp.asarray(r['in_ref_pos'][0])
feats = jnp.concatenate([ref_pos, jnp.asarray(r['in_ref_charge'][0])[:,None], mask[:,None],
                         jnp.asarray(r['in_ref_element'][0]),
                         jnp.asarray(r['in_ref_atom_name_chars'][0]).reshape(len(mask), -1)], -1)
c_base = layer_norm(feats @ ae['atom_linear/weights'], ae['atom_norm/scale'], ae['atom_norm/offset'])
r_noisy = jnp.asarray(r['in_x_noisy'][0]) / np.sqrt(t_hat**2 + SIGMA**2)
qv = c_base + jnp.concatenate([r_noisy, jnp.zeros_like(r_noisy)], -1) @ ae['coords_linear/weights']
cos, sin = build_rope(ref_pos, jnp.asarray(r['in_ref_space_uid'][0]), c_base.shape[-1]//4)
ab = pick('blocks/', ae)
for i in range(dims['n_diff_atom']):
    qv = swa_block(qv, c_base, idx(ab, i), cos, sin, mask, 64, 4)
rep('diffusion atom encoder q', qv, r['enc_q'])
a2t = jnp.asarray(r['in_tok_idx'][0].astype(int))
L = r['in_s_inputs'].shape[1]
qa = jax.nn.relu(qv @ ae['atom_to_token/weights'])
a = jax.ops.segment_sum(qa*mask[:,None], a2t, L) / jnp.maximum(jax.ops.segment_sum(mask, a2t, L)[:,None], 1e-9)

# ---- token transformer -----------------------------------------------------
a = a + layer_norm(s, P['s_step_norm/scale'], P['s_step_norm/offset']) @ P['s_to_token/weights']
ta, tt2 = pick('token_attn/'), pick('token_transition/')
for i in range(ta['q/weights'].shape[0]):
    a = a + attn_block(a, s, z, idx(ta, i), 16)
    a = a + trans_block(a, s, idx(tt2, i))
rep('token transformer (12 blk)', a, r['tok_out'])
a = layer_norm(a, P['token_norm/scale'], P['token_norm/offset'])

# ---- atom decoder ----------------------------------------------------------
ad = pick('atom_decoder/')
qd = qv + (a @ ad['token_to_atom/weights'])[a2t]
db = pick('blocks/', ad)
for i in range(dims['n_diff_atom']):
    qd = swa_block(qd, c_base, idx(db, i), cos, sin, mask, 64, 4)
r_up = layer_norm(qd, ad['norm/scale'], ad['norm/offset']) @ ad['output/weights']
rep('r_update (EDM UNDONE)', r_up, r['r_update'])

x_den = (SIGMA**2/(SIGMA**2+t_hat**2))*jnp.asarray(r['in_x_noisy'][0]) \
        + (SIGMA*t_hat/np.sqrt(SIGMA**2+t_hat**2))*r_up
cc = rep('x_denoised (FULL STEP)', x_den, r['out_x_denoised'])
print("\nt_hat = %.4g" % t_hat)
