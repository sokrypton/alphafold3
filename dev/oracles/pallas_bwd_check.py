"""Does the wrapper drive the Pallas attention BACKWARD, and is it worth it?

colabfold-kernels 0.4.0 added a Pallas flash backward (dQ/dK/dV + dBias), so
`pallas` can now serve a gradient. `tests/test_pallas_vjp.py` in that repo is
the KERNEL test; this is the WIRING -- `pallas_attn.attention` under jax.grad,
at this fork's calling convention (b, S, h, c with a [1, h, Sq, Sk] bias and a
boolean mask) -- plus the timing that decides whether the platform table should
prefer it to Triton on a gradient path.

  PYTHONPATH=src:<colabfold-kernels>/src python dev/oracles/pallas_bwd_check.py

TIME EVERY TOKAMAX ARM TWICE: it autotunes the forward and the VJP separately,
and pass 1 reads ~30x slow. This harness reports pass 2.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.model.components import attention as attn_mod
from alphafold3.model.components import pallas_attn

TOL = 3e-2            # bf16 kernel against an fp32 XLA reference


def _reference(q, k, v, mask, bias, scale):
  logits = jnp.einsum('bqhc,bkhc->bhqk', q.astype(jnp.float32),
                      k.astype(jnp.float32)) * scale
  if bias is not None:
    logits += bias.astype(jnp.float32)
  logits = jnp.where(mask[:, None, None, :], logits, -1e9)
  w = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum('bhqk,bkhc->bqhc', w, v.astype(jnp.float32))


def _rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if not np.isfinite(a).all():
    return float('inf')
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-3))


def _inputs(b, sq, sk, h, c, seed=0):
  rng = np.random.default_rng(seed)
  f = lambda *s: jnp.asarray(rng.standard_normal(s) * 0.5, jnp.float32)
  q, k, v = f(b, sq, h, c), f(b, sk, h, c), f(b, sk, h, c)
  bias = f(1, h, sq, sk)
  mask = jnp.asarray(rng.random((b, sk)) > 0.15)
  return q, k, v, bias, mask


def _loss(fn, q, k, v, bias, mask, scale, dout):
  out = fn(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16),
           v.astype(jnp.bfloat16), mask=mask, bias=bias.astype(jnp.bfloat16),
           scale=scale)
  return jnp.sum(out.astype(jnp.float32) * dout)


def correctness():
  print(f'pallas installed={pallas_attn.installed()} '
        f'bwd={pallas_attn.bwd_installed()}')
  bad = 0
  for b, sq, sk, h, c in ((2, 96, 96, 4, 32), (1, 128, 128, 4, 64),
                          (2, 70, 83, 8, 16), (1, 96, 96, 4, 48),
                          (1, 64, 64, 16, 96)):
    q, k, v, bias, mask = _inputs(b, sq, sk, h, c)
    scale = float(c) ** -0.5
    rng = np.random.default_rng(1)
    dout = jnp.asarray(rng.standard_normal((b, sq, h, c)) * 0.5, jnp.float32)

    ours = lambda *a, **kw: pallas_attn.attention(*a, **kw)
    out = jax.jit(lambda *a: ours(a[0].astype(jnp.bfloat16),
                                  a[1].astype(jnp.bfloat16),
                                  a[2].astype(jnp.bfloat16),
                                  mask=mask, bias=a[3].astype(jnp.bfloat16),
                                  scale=scale))(q, k, v, bias)
    if out is None:
      print(f'  b={b} sq={sq} sk={sk} h={h} c={c:2d}   SKIP (kernel declined)')
      continue
    g = jax.jit(jax.grad(lambda *a: _loss(ours, *a, mask, scale, dout),
                         argnums=(0, 1, 2, 3)))(q, k, v, bias)
    ref = _reference(q, k, v, mask, bias, scale)
    rg = jax.jit(jax.grad(lambda *a: jnp.sum(
        _reference(a[0], a[1], a[2], mask, a[3], scale) * dout),
        argnums=(0, 1, 2, 3)))(q, k, v, bias)
    rows = [('out', out, ref)] + list(zip(('dq', 'dk', 'dv', 'dbias'), g, rg))
    worst = max(_rel(x, y) for _, x, y in rows)
    bad += worst >= TOL
    print(f'  b={b} sq={sq} sk={sk} h={h} c={c:2d}   '
          + '  '.join(f'{n} {_rel(x, y):.1e}' for n, x, y in rows)
          + f"   {'OK' if worst < TOL else 'FAIL'}")
  return bad


def _time(fn, *args, n=5):
  best = []
  for _ in range(n):
    t = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    best.append(time.perf_counter() - t)
  return min(best) * 1e3


def timing():
  print('\nms per call, min of 5, pass 2 (tokamax autotunes on pass 1).')
  print('Triangle attention shape: [N rows, N keys, 4 heads, 32] -- the batch'
        ' axis is N,')
  print('which is what makes this op 77% of the trunk. A b=1 probe of the same'
        ' N is')
  print('1000x smaller and every backend looks identical on it.')
  print(f"{'N':>5} {'impl':>8} {'fwd':>9} {'fwd+bwd':>9}")
  for n in (256, 384, 512):
    for impl in ('xla', 'cudnn', 'triton', 'pallas'):
      q, k, v, bias, mask = _inputs(n, n, n, 4, 32)
      scale = float(32) ** -0.5
      rng = np.random.default_rng(1)
      dout = jnp.asarray(rng.standard_normal((n, n, 4, 32)) * 0.5, jnp.float32)
      call = lambda *a: attn_mod.dot_product_attention(
          a[0].astype(jnp.bfloat16), a[1].astype(jnp.bfloat16),
          a[2].astype(jnp.bfloat16), mask=mask[:, None, None, :],
          bias=a[3].astype(jnp.bfloat16), implementation=impl, scale=scale)
      try:
        fwd = jax.jit(call)
        gr = jax.jit(jax.grad(lambda *a: jnp.sum(call(*a).astype(jnp.float32)
                                                 * dout), argnums=(0, 1, 2, 3)))
        fwd(q, k, v, bias)          # compile + autotune
        gr(q, k, v, bias)
        tf = _time(lambda: fwd(q, k, v, bias))
        tb = _time(lambda: gr(q, k, v, bias))
        print(f'{n:>5} {impl:>8} {tf:>9.3f} {tb:>9.3f}')
      except Exception as e:  # pylint: disable=broad-except
        print(f'{n:>5} {impl:>8}   {type(e).__name__}: {str(e)[:60]}')


if __name__ == '__main__':
  jax.config.update('jax_default_matmul_precision', 'highest')
  bad = correctness()
  timing()
  print('\nRESULT:', 'PASS' if not bad else f'FAIL ({bad})')
  sys.exit(0 if not bad else 1)
