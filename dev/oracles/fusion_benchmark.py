"""One (case, variant, size) per PROCESS: peak_bytes_in_use never resets."""
import os, sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import time, numpy as np, jax, jax.numpy as jnp
case, variant, L, c = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dev = jax.devices()[0]; assert dev.platform == 'gpu'
key = jax.random.PRNGKey(0)

if case == 'trimul':
    z = jax.random.normal(key, (L, L, c), jnp.float32)
    wp = jax.random.normal(key, (c, 2*c), jnp.float32) * 0.02
    wg = jax.random.normal(key, (c, 2*c), jnp.float32) * 0.02
    if variant == 'split':
        f = jax.jit(lambda z, a, b: (z @ a) * jax.nn.sigmoid(z @ b)); args = (z, wp, wg)
    else:
        wb = jnp.concatenate([wp, wg], -1)
        def _f(z, w):
            h = z @ w; n = h.shape[-1] // 2
            return h[..., :n] * jax.nn.sigmoid(h[..., n:])
        f = jax.jit(_f); args = (z, wb)
elif case == 'kv':
    x = jax.random.normal(key, (L, c), jnp.float32)
    wk = jax.random.normal(key, (c, c), jnp.float32) * 0.02
    wv = jax.random.normal(key, (c, c), jnp.float32) * 0.02
    if variant == 'split':
        f = jax.jit(lambda x, a, b: (x @ a, x @ b)); args = (x, wk, wv)
    else:
        wkv = jnp.concatenate([wk, wv], -1)
        def _f(x, w):
            h = x @ w; n = h.shape[-1] // 2
            return h[..., :n], h[..., n:]
        f = jax.jit(_f); args = (x, wkv)

o = f(*args); jax.block_until_ready(o)
ts = []
for _ in range(30):
    t0 = time.perf_counter(); o = f(*args); jax.block_until_ready(o)
    ts.append(time.perf_counter() - t0)
print('%s %s L=%d c=%d  %.3f ms  peak %.1f MiB' % (
    case, variant, L, c, np.median(ts)*1e3,
    dev.memory_stats()['peak_bytes_in_use']/2**20))
