"""Are the volta LayerNorm / gated-dual-projection arms wired correctly?

Two checks, and the first one runs on ANY device:

  1. LAYOUT. Substitute reference implementations for the two kernels and
     compare TriangleMultiplication against its normal (tokamax GLU) path. This
     is what catches the thing a T4 benchmark would not: the a/b split here is
     INTERLEAVED (`projection` is transposed to [2h, N, N] and reshaped to
     [h, 2, N, N]), not the contiguous split Milot's own wrapper does, and the
     kernel output has to come back channel-major. Needs matmul precision
     pinned -- tf32 alone is 1.5e-3 here and would hide a real error.

  2. KERNELS. Only where colabfold-legacy-kernels actually loads (sm_70/75):
     the two kernels against an XLA reference, at the tri-mul's own shapes.

  PYTHONPATH=src:. python dev/oracles/volta_ops_check.py [N]
"""
import sys

import jax

jax.config.update('jax_default_matmul_precision', 'highest')

import haiku as hk  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from alphafold3.model import model_config  # noqa: E402
from alphafold3.model.components import volta_attn  # noqa: E402
from alphafold3.model.network import modules  # noqa: E402


def ref_gdp(x, w_proj, w_gate, mask, *, cc=None):
  h = w_proj.shape[1] // 2
  spatial = tuple(x.shape[:-1])
  xf = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
  m = jnp.reshape(mask, (-1,)).astype(jnp.float32)[:, None]
  out = []
  for off in (0, 1):
    proj = xf @ w_proj[:, off::2].astype(jnp.float32)
    gate = xf @ w_gate[:, off::2].astype(jnp.float32)
    o = m * proj * jax.nn.sigmoid(gate)
    out.append(jnp.swapaxes(o, 0, 1).reshape((h,) + spatial).astype(x.dtype))
  return out[0], out[1]


def ref_ln(x, scale, offset, *, eps=1e-5, cc=None):
  xf = x.astype(jnp.float32)
  mu = jnp.mean(xf, -1, keepdims=True)
  var = jnp.mean((xf - mu) ** 2, -1, keepdims=True)
  return ((xf - mu) * jax.lax.rsqrt(var + eps) * scale + offset).astype(x.dtype)


def _trimul(impl, n, c):
  gc = model_config.GlobalConfig(
      flash_attention_implementation=impl, bfloat16='none')
  cfg = modules.TriangleMultiplication.Config(equation='ikc,jkc->ijc')
  fwd = hk.transform(
      lambda a, m: modules.TriangleMultiplication(cfg, gc, name='tm')(a, m))
  key = jax.random.PRNGKey(0)
  act = jax.random.normal(jax.random.PRNGKey(1), (n, n, c))
  mask = (jax.random.uniform(jax.random.PRNGKey(2), (n, n)) > 0.2).astype(
      jnp.float32)
  params = fwd.init(key, act, mask)
  # final_init='zeros' leaves half the weights zero, which would hide a wrong
  # split entirely -- give every matrix real numbers.
  params = jax.tree.map(
      lambda v: jax.random.normal(
          jax.random.PRNGKey(abs(hash(v.shape)) % 9999), v.shape) * 0.3
      if v.ndim > 1 else v, params)
  return np.asarray(fwd.apply(params, key, act, mask))


def check_layout(n=24, c=32):
  base = _trimul('xla', n, c)
  hits = []
  real_ln, real_gdp = volta_attn.layer_norm, volta_attn.gated_dual_proj
  volta_attn.layer_norm = lambda *a, **k: (hits.append('ln'), ref_ln(*a, **k))[1]
  volta_attn.gated_dual_proj = lambda *a, **k: (hits.append('gdp'),
                                                ref_gdp(*a, **k))[1]
  try:
    got = _trimul('volta', n, c)
  finally:
    volta_attn.layer_norm, volta_attn.gated_dual_proj = real_ln, real_gdp
  err = float(np.abs(base - got).max())
  ok = 'ln' in hits and 'gdp' in hits and err < 1e-4
  print(f'layout: max|d| {err:.2e} vs {np.abs(base).max():.2f}  '
        f'arms fired {sorted(set(hits))}  {"OK" if ok else "FAIL"}')
  return ok


def check_kernels(n=384, c=128):
  cc = volta_attn.device_cc()
  if not volta_attn.ops_available(cc):
    print(f'kernels: colabfold-legacy-kernels not loadable here (cc {cc}) '
          '-- skipped')
    return True
  x = jax.random.normal(jax.random.PRNGKey(3), (n, n, c), jnp.float32)
  scale = jax.random.normal(jax.random.PRNGKey(4), (c,)) * 0.1 + 1.0
  offset = jax.random.normal(jax.random.PRNGKey(5), (c,)) * 0.1
  ln = volta_attn.layer_norm(x, scale, offset)
  d_ln = float(np.abs(np.asarray(ln) - np.asarray(ref_ln(x, scale, offset))).max())

  wp = jax.random.normal(jax.random.PRNGKey(6), (c, 2 * c)) * 0.1
  wg = jax.random.normal(jax.random.PRNGKey(7), (c, 2 * c)) * 0.1
  mask = (jax.random.uniform(jax.random.PRNGKey(8), (n, n)) > 0.2).astype(
      jnp.float32)
  a, b = volta_attn.gated_dual_proj(ln, wp, wg, mask)
  ra, rb = ref_gdp(ln, wp, wg, mask)
  d_gdp = max(float(np.abs(np.asarray(a) - np.asarray(ra)).max()),
              float(np.abs(np.asarray(b) - np.asarray(rb)).max()))
  # float16 kernels: these are tolerances, not equalities.
  ok = d_ln < 5e-3 and d_gdp < 5e-2
  print(f'kernels (cc {cc}, N={n}): layer_norm max|d| {d_ln:.2e}   '
        f'gated_dual_proj max|d| {d_gdp:.2e}  {"OK" if ok else "FAIL"}')
  return ok


if __name__ == '__main__':
  size = int(sys.argv[1]) if len(sys.argv) > 1 else 24
  good = check_layout(size) & check_kernels()
  print('RESULT:', 'PASS' if good else 'FAIL')
  sys.exit(0 if good else 1)
