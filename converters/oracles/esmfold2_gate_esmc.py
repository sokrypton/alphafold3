"""Gate the ESM-C tower: our JAX forward vs the saved torch hidden states."""
import sys, os, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
jax.config.update('jax_platform_name', 'cpu')      # 25 GB fp32 won't fit on the A10
from converters import esmc as C

SNAP = os.path.expanduser('~/.cache/huggingface/hub/models--biohub--ESMC-6B/snapshots')
snap = os.path.join(SNAP, os.listdir(SNAP)[0])
t0 = time.time()
from converters.esmfold2 import load_esmfold2_checkpoint
sd = load_esmfold2_checkpoint(snap)
print('loaded %d tensors in %.0f s' % (len(sd), time.time() - t0))
dims = C.derive_dims(sd)
print('dims:', dims)
p = C.map_esmc_to_af3(sd, dims)
del sd

def layer_norm(x, scale, offset=0.0, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + eps) * scale + offset

def rope(x, pos, base=10000.0):
    # x [L, H, D]; RotaryEmbedding(head_dim), interleaved-half convention
    D = x.shape[-1]
    inv = 1.0 / (base ** (np.arange(0, D, 2, dtype=np.float32) / D))
    fr = pos[:, None] * inv[None, :]
    c, s = jnp.cos(fr)[:, None], jnp.sin(fr)[:, None]
    x1, x2 = x[..., :D//2], x[..., D//2:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)

def run(ids, p, dims):
    L = ids.shape[0]
    H, dh = dims['n_heads'], dims['d_model'] // dims['n_heads']
    scale = dims['residual_scale']
    x = jnp.asarray(p['embed/weights'])[ids]
    states = [x]
    pos = jnp.arange(L, dtype=jnp.float32)
    B = {k[len('blocks/'):]: v for k, v in p.items() if k.startswith('blocks/')}
    for i in range(dims['n_layers']):
        b = {k: jnp.asarray(v[i]) for k, v in B.items()}
        h = layer_norm(x, b['attn_norm/scale'], b['attn_norm/offset']) @ b['qkv/weights']
        q, k, v = jnp.split(h, 3, -1)
        q = layer_norm(q, b['q_norm/scale'])          # over the FULL d_model
        k = layer_norm(k, b['k_norm/scale'])
        q = rope(q.reshape(L, H, dh), pos); k = rope(k.reshape(L, H, dh), pos)
        v = v.reshape(L, H, dh)
        logits = jnp.einsum('ihd,jhd->hij', q, k) * dh ** -0.5
        ctx = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(L, -1)
        x = x + (ctx @ b['attn_out/weights']) / scale
        h = layer_norm(x, b['ffn_norm/scale'], b['ffn_norm/offset']) @ b['fc1/weights']
        n = h.shape[-1] // 2
        x = x + ((jax.nn.silu(h[..., :n]) * h[..., n:]) @ b['fc2/weights']) / scale
        states.append(x)
    # HF's LAST hidden state is POST the stack's final LayerNorm; the other 80
    # are the raw residual stream.  Using the pre-norm value here reads corr
    # 0.909 on layer 80 while every other layer is >= 0.9998.
    states[-1] = layer_norm(states[-1], p['final_norm/scale'])
    return jnp.stack(states), states[-1]

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
d = dict(np.load(S + 'esmfold2_dump.npz'))
tok = d['feat.input_ids'][0].astype(np.int64)
ids = C.lm_input_ids(tok)
print('lm input len', len(ids), '(BOS + %d + EOS)' % len(tok))
t0 = time.time()
states, normed = run(jnp.asarray(ids), p, dims)
states.block_until_ready()
print('forward: %.0f s   states %s' % (time.time() - t0, states.shape))

nat = d['feat.input_ids'] if False else np.load(S + 'esmfold2_det.npz')['lm_hidden'][0]  # [L,81,D]
ours = np.asarray(states)[:, 1:-1, :].transpose(1, 0, 2)                                 # strip BOS/EOS
print('ours', ours.shape, 'native', nat.shape)
for li in (0, 1, 40, 78, 79, 80):
    o, n = ours[:, li], nat[:, li]
    print('  layer %-3d relerr %.3e  corr %.8f' % (
        li, np.abs(o - n).max() / max(np.abs(n).max(), 1e-9),
        np.corrcoef(o.ravel(), n.ravel())[0, 1]))
print('ALL layers corr %.8f' % np.corrcoef(ours.ravel(), nat.ravel())[0, 1])
