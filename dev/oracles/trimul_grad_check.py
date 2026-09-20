"""Is TriangleMultiplication DIFFERENTIABLE on every fused row, and correct?

The attention kernels all have a backward now (XLA/cuDNN natively, triton via
tokamax's VJP, volta via the fork). THE GLU AND THE FUSED LAYERNORM DO NOT --
they are a bare `ffi_call` and a bare `pallas_call`, neither of which jax can
differentiate at all. `glu_kernel='auto'` follows the attention pick, so the
moment a pre-Ampere card answers 'volta' to a differentiable caller, a design
step reaches one and dies:

    ValueError: The FFI call to `VoltaLayerNorm` cannot be differentiated.

Both now carry a `custom_vjp` whose backward is `jax.vjp` of a plain-XLA twin.
This checks BOTH halves of that: the fused forward still matches XLA, and the
gradient it now produces does too.

  PYTHONPATH=src python dev/oracles/trimul_grad_check.py [volta70 volta75 pallas]

`volta70`/`volta75` need the kernels for that capability; on any other card,
build them for the local arch and shim them in under the target's name -- see
dev/oracles/volta_bwd_check.py, which documents the recipe (CLK_SHIM).
"""
import contextlib
import os
import sys

if os.environ.get('CLK_SHIM'):
  sys.path.insert(0, os.environ['CLK_SHIM'])

import jax

jax.config.update('jax_default_matmul_precision', 'highest')

import haiku as hk  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from alphafold3.model import model_config  # noqa: E402
from alphafold3.model.components import pallas_attn  # noqa: E402
from alphafold3.model.components import volta_attn  # noqa: E402
from alphafold3.model.network import modules  # noqa: E402

TOL = 2e-2      # a float16 kernel against float32 XLA, through a whole module

# The PALLAS GLU takes only bf16/f16 activations and returns None on f32, so a
# float32 harness would never reach it and would report a serene OK for the XLA
# fallback. Each row names the dtype it must be exercised in.
ROWS = {'volta70': ('volta', 70, jnp.float32),
        'volta75': ('volta', 75, jnp.float32),
        'pallas': ('pallas', None, jnp.bfloat16)}


# THE REFERENCE IS THE SAME MODULE WITH THE KERNELS SUBSTITUTED, not a
# differently-configured one: `glu_kernel='xla'` routes to tokamax, which
# promotes to float32, so comparing a bf16 row against it measures the dtype
# change and not the kernel (it read 1.49 relative before this was fixed).
# Swapping just the two entry points keeps every other op identical.
def _ln_reference(x, scale, offset, *, eps=1e-5, cc=None):
  return volta_attn._ln_ref(  # pylint: disable=protected-access
      x, jnp.reshape(scale, (x.shape[-1],)),
      jnp.reshape(offset, (x.shape[-1],)), eps)


def _gdp_reference(x, w_proj, w_gate, mask, **kwargs):
  return volta_attn.gdp_reference(x, w_proj, w_gate, mask)


@contextlib.contextmanager
def substituted():
  saved = (volta_attn.layer_norm, volta_attn.gated_dual_proj,
           pallas_attn.gated_dual_proj)
  volta_attn.layer_norm = _ln_reference
  volta_attn.gated_dual_proj = _gdp_reference
  pallas_attn.gated_dual_proj = _gdp_reference
  try:
    yield
  finally:
    (volta_attn.layer_norm, volta_attn.gated_dual_proj,
     pallas_attn.gated_dual_proj) = saved


def build(glu, dtype=jnp.float32, n=64, c=128, seed=0):
  gc = model_config.GlobalConfig(
      flash_attention_implementation='xla', glu_kernel=glu,
      bfloat16='none' if dtype == jnp.float32 else 'all')
  cfg = modules.TriangleMultiplication.Config(equation='ikc,jkc->ijc')
  f = hk.transform(
      lambda a, m: modules.TriangleMultiplication(cfg, gc, name='tm')(a, m))
  key = jax.random.PRNGKey(seed)
  # The dtype must be the ACTIVATION's, not just a config flag: the module
  # passes whatever it is given straight to the kernel, and the pallas GLU
  # refuses float32.
  act = (jax.random.normal(key, (n, n, c), jnp.float32) * 0.1).astype(dtype)
  mask = jnp.ones((n, n), jnp.float32)
  params = f.init(key, act, mask)
  # `final_init` is zeros, so an untouched init gives an all-zero output and a
  # zero gradient -- a test that passes on any implementation, including a
  # broken one. Give every leaf real values.
  # NOT `hash(path)`: python randomises string hashing per process, so that
  # would draw different weights in every run and make the numbers below
  # irreproducible (they would still compare correctly WITHIN a run, which is
  # how this went unnoticed once already).
  leaves, treedef = jax.tree_util.tree_flatten(params)
  keys = jax.random.split(key, len(leaves))
  params = jax.tree_util.tree_unflatten(
      treedef, [0.1 * jax.random.normal(k, x.shape, x.dtype)
                for k, x in zip(keys, leaves)])
  loss = lambda p, a: jnp.sum(f.apply(p, key, a, mask) ** 2)
  return f.apply(params, key, act, mask), jax.grad(loss, argnums=(0, 1))(
      params, act)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))


def main(rows):
  bad = 0
  for row in rows:
    glu, cc, dtype = ROWS[row]
    with substituted():
      ref_out, (ref_dp, ref_da) = build(glu, dtype)
    if cc is not None:
      volta_attn.device_cc = lambda cc=cc: cc     # the shim answers for it
    # A ROW WHOSE KERNEL IS NOT INSTALLED FALLS BACK SILENTLY, and would then
    # print a confident OK for a path it never ran. Refuse instead.
    have = (volta_attn.ops_available(cc) if cc is not None
            else pallas_attn.installed())
    if not have:
      print(f'{row}: kernel not installed here -- SKIP (not a pass)')
      continue
    print(f'{row}: ', end='')
    try:
      out, (dparams, dact) = build(glu, dtype)
    except Exception as e:                        # noqa: BLE001
      print('FAILED', f'{type(e).__name__}: {str(e).splitlines()[0][:90]}')
      bad += 1
      continue
    dp = max(rel(g, r) for g, r in zip(jax.tree_util.tree_leaves(dparams),
                                       jax.tree_util.tree_leaves(ref_dp)))
    worst = max(rel(out, ref_out), rel(dact, ref_da), dp)
    # EXACTLY the reference means the kernel never ran: a fused low-precision
    # kernel cannot agree bit for bit with the XLA path. This is the check that
    # catches a silent fallback, which `installed()` alone does not.
    if worst == 0.0:
      print('fell back to XLA -- the kernel never ran   -> FAIL')
      bad += 1
      continue
    ok = worst < TOL
    bad += not ok
    print(f'out {rel(out, ref_out):.2e}  dact {rel(dact, ref_da):.2e}  '
          f'dparams {dp:.2e}   -> {"OK" if ok else "FAIL"}')
  print('RESULT:', 'PASS' if not bad else f'FAIL ({bad})')
  return bad


if __name__ == '__main__':
  sys.exit(1 if main(sys.argv[1:] or list(ROWS)) else 0)
