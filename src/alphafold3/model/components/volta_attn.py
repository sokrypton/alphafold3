"""sm_70 / sm_75 fused attention: the only one a T4 or V100 can run.

THE KERNELS AND THE WRAPPER THIS IS ADAPTED FROM ARE MILOT MIRDITA'S.
`colabfold-legacy-kernels` (MIT, (c) 2026 The ColabFold Development Team) ships
prebuilt sm_70/sm_75 CUDA kernels registered as XLA FFI targets, and
`alphafold/model/volta_attn.py` in alphafold-colabfold 2.3.20 (Apache-2.0)
drives them; ColabFold's AF2 path selects them below sm_80 as `cuda_legacy`.
This file is that wrapper, narrowed to the one op this fork needs and adapted
to its calling convention.

WHY IT EXISTS HERE. On compute capability 7.x every fused path we have refuses:
cuDNN's SDPA wants SM80 ("SDPA FP16/BF16 requires SM80 (Ampere) or newer"),
tokamax has no kernel, and XLA gates Pallas/Triton at sm_80
("Triton support is only enabled for Ampere GPUs ... but got compute capability
7.5"). So the commonest Colab card runs the materialising XLA attention, and
triangle attention is 77% of a pairformer pass. Measured on a Colab T4 at this
fork's own triangle-attention shape, against that XLA path:

    N=256   1.787 ms vs  5.764 ms   3.22x   max|d| 0.0001
    N=384   7.294 ms vs 21.965 ms   3.01x   max|d| 0.0001
    N=512  17.230 ms vs 57.740 ms   3.35x   max|d| 0.0001

FORWARD ONLY -- THIS CANNOT BE DIFFERENTIATED. The libraries export no backward
symbol, and jax says so plainly:

    ValueError: The FFI call to `VoltaMma` cannot be differentiated.

The design path backprops through the trunk, and GridSelfAttention is in the
trunk, so this must never be selected for a differentiable call. That is why
`platform.attention_config` takes `differentiable=` and never answers 'volta'
when it is set.

FLOAT16, not bfloat16: Volta and Turing tensor cores have no bf16. Inputs are
cast in and the result is cast back to the caller's dtype.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.util
import os

import jax
import jax.numpy as jnp
import numpy as np

# Head dims the kernels instantiate (Milot's _HEAD_DIMS_MMA / _HEAD_DIMS_WMMA).
_HEAD_DIMS = (8, 16, 32, 64)
_LOADED = {}


def installed() -> bool:
  """True if the kernel package is importable. No jax, no dlopen."""
  return importlib.util.find_spec('colabfold_legacy_kernels') is not None


def _symbol(cc: int) -> str:
  """sm_75+ gets the CUTLASS mma kernel; sm_70 the wmma one."""
  return 'VoltaMma' if cc >= 75 else 'VoltaWmma'


def available(cc) -> bool:
  """dlopen the library for this device and register its FFI target."""
  if cc is None:
    return False
  cc = int(cc)
  sym = _symbol(cc)
  key = (cc, sym)
  if key in _LOADED:
    return _LOADED[key]
  ok = False
  try:
    import colabfold_legacy_kernels as clk

    path = clk.library_path('attention', cc)
    if os.path.exists(path):
      lib = ctypes.cdll.LoadLibrary(path)
      jax.ffi.register_ffi_target(
          sym, jax.ffi.pycapsule(getattr(lib, sym)), platform='CUDA')
      ok = True
  except Exception:  # pylint: disable=broad-except
    ok = False
  _LOADED[key] = ok
  return ok


@functools.partial(jax.jit, static_argnames=('sym', 'sm_scale', 'block_q', 'block_k'))
def _call(q, k, v, bias, kmask, *, sym, sm_scale, block_q, block_k):
  n, h, sq, d = q.shape
  # sequential: the handlers take a fixed rank, so a vmap must not add one.
  return jax.ffi.ffi_call(sym, jax.ShapeDtypeStruct((n, h, sq, d), jnp.float16),
                          vmap_method='sequential')(
      q, k, v, bias, kmask,
      scale=np.float32(sm_scale),
      block_q=np.int64(block_q), block_k=np.int64(block_k))


def attention(q, k, v, *, mask, bias, scale, cc, block_q=64, block_k=32):
  """This fork's attention contract, on the legacy kernel. None if it cannot.

  q/k/v are (batch, seq, heads, dim); bias is (1, heads, q, k); mask is
  boolean and broadcasts over the key axis. Returns None -- never a wrong
  answer -- when the shapes are outside what the kernel instantiates, so the
  caller can fall back to XLA.
  """
  if bias is None or mask is None or q.ndim != 4:
    return None
  b, sq, h, c = q.shape
  sk = k.shape[1]
  if c not in _HEAD_DIMS or c != v.shape[-1]:
    return None
  # A broadcast bias (AF2's MSA column attention passes (batch, 1, 1, seq))
  # is not what the kernel indexes; it wants one [h, sq, sk] slab.
  if bias.shape[-3:] != (h, sq, sk) or bias.shape[0] != 1:
    return None
  if mask.shape[-1] != sk:
    return None
  if not available(cc):
    return None
  if cc >= 75 and c == 64 and (block_q, block_k) == (64, 32):
    block_k = 64      # the mma kernel has no (64, 64, 32) instantiation
  in_dtype = q.dtype
  f16 = lambda t: jnp.swapaxes(t, 1, 2).astype(jnp.float16)   # -> [b, h, S, c]
  kmask = jnp.broadcast_to(
      jnp.reshape(mask.astype(jnp.bool_), mask.shape[:1] + (-1,))[:, -sk:],
      (b, sk)).astype(jnp.uint8)
  out = _call(f16(q), f16(k), f16(v),
              jnp.reshape(bias, (h, sq, sk)).astype(jnp.float16), kmask,
              sym=_symbol(int(cc)), sm_scale=float(scale),
              block_q=block_q, block_k=block_k)
  return jnp.swapaxes(out, 1, 2).astype(in_dtype)     # back to [b, S, h, c]
