import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
from converters import esmfold2 as E

MAX_Z, CHARV, MAXC = 128, 64, 4
sd = dict(np.load('esmfold2_sd.npz'))
f  = dict(np.load('esmfold2_dump.npz'))          # holds feat.* (same sequence)
r  = dict(np.load('esmfold2_det_fp32.npz'))      # fp32 deterministic x_inputs
dims = E.derive_dims(sd)
p = E.atom_encoder(sd, 'inputs_embedder.atom_attention_encoder', 3)

def rms(x):                                       # affine-free RMSNorm
    # torch F.rms_norm(eps=None) uses finfo(dtype).eps, NOT 1e-5
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + np.finfo(np.float32).eps)

def layer_norm(x, s, o, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + eps) * s + o

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

# ---- features (exactly as the model builds them) ---------------------------
ref_pos  = jnp.asarray(f['feat.ref_pos'][0])
mask     = jnp.asarray(f['feat.atom_attention_mask'][0])
charge   = jnp.asarray(f['feat.ref_charge'][0])
elem     = jax.nn.one_hot(f['feat.ref_element'][0].astype(int), MAX_Z) * mask[:, None]
chars    = (jax.nn.one_hot(f['feat.ref_atom_name_chars'][0].astype(int), CHARV)
            * mask[:, None, None]).reshape(-1, MAXC * CHARV)
a2t      = (f['feat.atom_to_token'][0].astype(int) * f['feat.atom_attention_mask'][0].astype(int))
uid      = jnp.asarray(f['feat.ref_space_uid'][0])

atom_feats = jnp.concatenate([ref_pos, charge[:, None], mask[:, None], elem, chars], -1)
print("atom_feats", atom_feats.shape, "expected", dims['atom_in'])

c = layer_norm(atom_feats @ p['atom_linear/weights'],
               p['atom_norm/scale'], p['atom_norm/offset'])
n_heads, half_window = 4, 128 // 2
cos, sin = build_rope(ref_pos, uid, c.shape[-1] // n_heads)
q = c
for i in range(3):
    q = swa_block(q, c, {k[len('blocks/'):]: v[i] for k, v in p.items() if k.startswith('blocks/')},
                  cos, sin, mask, half_window, n_heads)
q_to_a = jax.nn.relu(q @ p['atom_to_token/weights'])

L = r['x_inputs.0'].shape[1]
seg = jnp.asarray(a2t)
w = jnp.asarray(mask)
num = jax.ops.segment_sum(q_to_a * w[:, None], seg, L)
den = jax.ops.segment_sum(w, seg, L)[:, None]
a = num / jnp.maximum(den, 1e-9)

native_a = r['x_inputs.0'][0][:, :a.shape[-1]]
e = np.abs(np.asarray(a) - native_a).max() / np.abs(native_a).max()
cc = np.corrcoef(np.asarray(a).ravel(), native_a.ravel())[0, 1]
print("ATOM ENCODER -> a   relerr %.3e   corr %.8f" % (e, cc))
