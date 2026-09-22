"""Does the wrapper drive the legacy attention BACKWARD, on both families?

Not a kernel test -- `tests/test_attn_bwd.py` in the kernels repo is that. This
is the WIRING: `volta_attn.attention` under `jax.grad`, at this fork's calling
convention (b, S, h, c with a [1, h, Sq, Sk] bias), against an fp32 XLA
reference. It is the check that caught `bwd_available` asking one library for
two symbols -- a bug a kernel test cannot see, because it registers the symbols
itself.

  PYTHONPATH=src python dev/oracles/volta_bwd_check.py [70 [75 ...]]

ON A CARD THAT IS NEITHER. The published wheels are cubin-only -- their sm_75
libraries raise `no kernel image is available for execution on the device` on
anything newer -- but the SOURCE builds for any arch, so build it for the local
one and shim it in under the name of the arch you want to test:

  cd ~/colabfold-legacy-kernels          # main, which is upstream's
  ALLOW_NEW_FFI=1 ARCH=86 bash scripts/build_kernels.sh /tmp/clk_build
  mkdir -p /tmp/clk_shim/colabfold_legacy_kernels/kernels
  cp src/colabfold_legacy_kernels/*.py /tmp/clk_shim/colabfold_legacy_kernels/
  cp -r /tmp/clk_build/sm86 /tmp/clk_shim/colabfold_legacy_kernels/kernels/sm70
  cp -r /tmp/clk_build/sm86 /tmp/clk_shim/colabfold_legacy_kernels/kernels/sm75
  CLK_SHIM=/tmp/clk_shim PYTHONPATH=src python dev/oracles/volta_bwd_check.py 70 75

`colabfold_kernels` must also be importable: since 0.4.0 the dlopen, the FFI
registration and the custom_vjp live in `colabfold_kernels.volta`, and this
file's `volta_attn` is only the adapter over it.
"""
import os
import sys

if os.environ.get('CLK_SHIM'):
  sys.path.insert(0, os.environ['CLK_SHIM'])

import jax

jax.config.update('jax_default_matmul_precision', 'highest')

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from alphafold3.model.components import volta_attn  # noqa: E402

NEG = -1.0e4          # the kernel's own masking convention
TOL = 5e-3            # float16 kernel vs float32 XLA


def reference(q, k, v, mask, bias, scale):
  logits = scale * jnp.einsum('bqhc,bkhc->bhqk', q, k) + bias
  logits = jnp.where(mask[:, None, None, :], logits, NEG)
  return jnp.einsum('bhqk,bkhc->bqhc', jax.nn.softmax(logits, -1), v)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))


def run(cc, b=2, sq=96, sk=96, h=4, c=32, seed=0):
  ks = jax.random.split(jax.random.PRNGKey(seed), 5)
  q = jax.random.normal(ks[0], (b, sq, h, c), jnp.float32) * .5
  k = jax.random.normal(ks[1], (b, sk, h, c), jnp.float32) * .5
  v = jax.random.normal(ks[2], (b, sk, h, c), jnp.float32) * .5
  bias = jax.random.normal(ks[3], (1, h, sq, sk), jnp.float32) * .5
  mask = jax.random.uniform(ks[4], (b, sk)) > 0.15
  scale = c ** -0.5
  w = jax.random.normal(jax.random.PRNGKey(7), (b, sq, h, c), jnp.float32)

  def kernel(q, k, v, bias):
    out = volta_attn.attention(q, k, v, mask=mask, bias=bias, scale=scale, cc=cc)
    if out is None:
      raise AssertionError('the wrapper refused these shapes')
    return out

  ref = lambda q, k, v, bias: reference(q, k, v, mask, bias, scale)
  loss = lambda f: lambda *a: jnp.sum(f(*a) * w)
  rows = [('out', kernel(q, k, v, bias), ref(q, k, v, bias))]
  rows += list(zip(('dq', 'dk', 'dv', 'dbias'),
                   jax.grad(loss(kernel), (0, 1, 2, 3))(q, k, v, bias),
                   jax.grad(loss(ref), (0, 1, 2, 3))(q, k, v, bias)))
  worst = 0.
  for name, got, want in rows:
    r = rel(got, want)
    worst = max(worst, r)
    print(f'    {name:6s} rel {r:.2e}')
  return worst


def main(caps):
  bad = 0
  for cc in caps:
    print(f'cc {cc}: installed={volta_attn.installed()} '
          f'bwd_installed={volta_attn.bwd_installed(cc)} '
          f'bwd_available={volta_attn.bwd_available(cc)}')
    if not volta_attn.bwd_available(cc):
      print('  no backward library for this capability -- SKIP')
      continue
    for kwargs in (dict(c=32), dict(c=64), dict(c=16, sq=70, sk=83), dict(c=8)):
      print('  ', kwargs)
      worst = run(cc, **kwargs)
      ok = worst < TOL
      bad += not ok
      print('   ->', 'OK' if ok else 'FAIL', f'(worst {worst:.2e})')
  print('RESULT:', 'PASS' if not bad else f'FAIL ({bad})')
  return bad


if __name__ == '__main__':
  caps = [int(a) for a in sys.argv[1:]] or [70, 75]
  sys.exit(1 if main(caps) else 0)
