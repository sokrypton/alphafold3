"""Which GLU should a DESIGN run take: tokamax, pallas, or none at all?

The datacenter row hands out `glu: pallas` only when differentiable=False --
a gate written when the Pallas GLU had no VJP at all. It has one now (ours
since 5301bab, upstream's in colabfold_kernels.grads since 0.4.0), so the
question is no longer "can it" but "is it faster than tokamax's real VJP when
the backward is a RECOMPUTATION in XLA".

Arms are the same TriangleMultiplication with only the GLU switched, forward
and forward+backward, timed TWICE (tokamax autotunes on pass 1).

ANSWERED, and the answer is no -- ms, min of 2 passes x 5 calls:

                A100 (cc 8.0)          A10 (cc 8.6)
                  fwd   fwd+bwd            fwd   fwd+bwd
      N=384   tokamax   1.046     3.105    2.432     6.895
              pallas    0.881     4.246    1.943    10.325
              off       0.997     3.027    2.527     7.427
      N=768   tokamax   4.076    12.115   10.988    31.207
              pallas    3.384    15.799    8.756    44.563
              off       3.831    11.514   10.940    32.658
      N=1024  tokamax   7.458    21.498   19.821    56.766
              pallas    6.183    27.342   17.203    80.465
              off       7.041    21.181   19.671    59.277

The Pallas GLU has the fastest FORWARD everywhere (1.19-1.25x tokamax) and the
slowest gradient everywhere (1.27-1.37x on the A100, 1.42-1.50x on the A10),
because its
backward is a recomputation of the forward in XLA where tokamax ships a real
VJP kernel. So the platform table stays as it is: `glu: pallas` for prediction,
tokamax for a differentiable caller. Note also that NO GLU KERNEL AT ALL is
within 2.5% of tokamax under a gradient and beats it at N=768/1024 on the
A100 -- the same "tokamax's GLU is worth nothing here" result the forward
sweep found, and not worth a switch of its own.

The gradients agree BIT-EXACTLY across all three arms, which is not a broken
harness: both fused GLUs take their backward from XLA, so only the forward
differs and only speed is being compared here.

  PYTHONPATH=src python dev/bench/glu_grad.py [N ...]
"""
import sys, time

# CAPTURE THE SIZES BEFORE absl EATS argv. `sys.argv = ['bench']` below is how
# every harness here stops absl parsing pytest/ssh arguments, and reading
# sys.argv[1:] after it silently gives the defaults -- which is exactly what
# the first run of this did.
_SIZES = [int(x) for x in sys.argv[1:]]
sys.argv = ['bench']
from absl import flags as _af
if not _af.FLAGS.is_parsed():
  _af.FLAGS(sys.argv)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.model import model_config
from alphafold3.model.network import modules


def build(n, c, glu, want_grad):
  gc = model_config.GlobalConfig(flash_attention_implementation='triton',
                                 glu_kernel=glu, bfloat16='all')
  cfg = modules.TriangleMultiplication.Config(equation='ikc,jkc->ijc',
                                              use_glu_kernel=glu != 'off')
  fwd = hk.transform(lambda a, m: modules.TriangleMultiplication(
      cfg, gc, name='tm')(a, m))
  act = (jax.random.normal(jax.random.PRNGKey(1), (n, n, c)) * 0.5).astype(jnp.bfloat16)
  mask = (jax.random.uniform(jax.random.PRNGKey(2), (n, n)) > 0.2).astype(jnp.bfloat16)
  p = fwd.init(jax.random.PRNGKey(0), act, mask)
  # final_init is zeros, so an untouched init gives a zero output AND a zero
  # gradient -- a timing that measures nothing and an accuracy check that
  # passes on anything.
  p = jax.tree.map(lambda v: (jax.random.normal(
      jax.random.PRNGKey(abs(hash(v.shape)) % 9999), v.shape) * 0.3).astype(v.dtype)
      if v.ndim > 1 else v, p)
  w = (jax.random.normal(jax.random.PRNGKey(3), (n, n, c)) * 0.1).astype(jnp.float32)
  loss = lambda p, a, m: jnp.sum(fwd.apply(p, None, a, m).astype(jnp.float32) * w)
  f = jax.jit(jax.grad(loss, argnums=(0, 1))) if want_grad else jax.jit(
      lambda p, a, m: fwd.apply(p, None, a, m))
  return f, p, act, mask


def t(f, *a, iters=5):
  o = f(*a)
  jax.block_until_ready(o)
  s = time.time()
  for _ in range(iters):
    o = f(*a)
  jax.block_until_ready(o)
  return (time.time() - s) / iters * 1e3, o


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))


if __name__ == '__main__':
  d = jax.devices()[0]
  print(d, 'cc', d.compute_capability)
  sizes = _SIZES or [384, 768, 1024]
  # 'off' = no GLU kernel at all (plain XLA), the floor tokamax lost to on the
  # forward at N >= 384.
  arms = ('tokamax', 'pallas', 'off')
  for n in sizes:
    print(f'\nTriangleMultiplication N={n} c=128  (ms, min of 2 passes x 5 calls)')
    print(f"  {'glu':10s} {'fwd':>9} {'fwd+bwd':>9}   {'d(grad) vs tokamax':>18}")
    ref = None
    for glu in arms:
      try:
        row = []
        for want_grad in (False, True):
          f, p, a, m = build(n, 128, glu, want_grad)
          ms1, o = t(f, p, a, m)
          ms2, o = t(f, p, a, m)
          row.append(min(ms1, ms2))
        g = np.asarray(jax.tree.leaves(o)[0], np.float32)
        if ref is None:
          ref, dtxt = g, '(reference)'
        else:
          dtxt = f'{rel(g, ref):.2e}'
        print(f'  {glu:10s} {row[0]:9.3f} {row[1]:9.3f}   {dtxt:>18}')
      except Exception as e:    # pylint: disable=broad-except
        print(f'  {glu:10s} FAILED: {type(e).__name__}: {str(e)[:90]}')
