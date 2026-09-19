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


def _load(kernel: str, cc: int, symbols) -> bool:
  """dlopen one kernel library for this device and register its FFI targets."""
  key = (kernel, int(cc), tuple(symbols))
  if key in _LOADED:
    return _LOADED[key]
  ok = False
  try:
    import colabfold_legacy_kernels as clk

    path = clk.library_path(kernel, int(cc))
    if path and os.path.exists(path):
      lib = ctypes.cdll.LoadLibrary(path)
      for sym in symbols:
        jax.ffi.register_ffi_target(
            sym, jax.ffi.pycapsule(getattr(lib, sym)), platform='CUDA')
      ok = True
  except Exception:  # pylint: disable=broad-except
    ok = False
  _LOADED[key] = ok
  return ok


def available(cc) -> bool:
  """dlopen the attention library for this device and register its target."""
  if cc is None:
    return False
  return _load('attention', int(cc), (_symbol(int(cc)),))


def ops_available(cc) -> bool:
  """dlopen the LayerNorm and gated-dual-projection libraries.

  The LayerNorm kernel is Volta-capable; the CUTLASS gated dual projection is
  sm_75+, and sm_70 gets a separate wmma build (Milot's `ops_available`).
  """
  if cc is None:
    return False
  cc = int(cc)
  syms = ('VoltaLayerNorm', 'VoltaGdp') if cc >= 75 else ('VoltaLayerNorm',)
  ok = _load('layer_norm', cc, syms)
  if ok and cc < 75:
    ok = _load('gated_dual_proj', cc, ('VoltaGdpWmma',))
  return ok


def device_cc():
  """The compute capability of device 0 as an integer (75), or None."""
  try:
    cap = getattr(jax.devices()[0], 'compute_capability', None)
  except Exception:  # pylint: disable=broad-except
    return None
  return None if cap is None else int(float(cap) * 10)


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


# ---------------------------------------------------------------------------
# The other two kernels the same package ships: a fused LayerNorm and a masked,
# sigmoid-gated dual projection. Both are Milot Mirdita's; the wrappers below
# are his `volta_layer_norm` / `volta_gated_dual_proj`, adapted to this fork's
# TriangleMultiplication, which is where they land: at N=384 that module spends
# 0.321 ms in its input LayerNorm (39% of the card's bandwidth peak) and
# 0.869 ms in the GLU, and tokamax's GLU -- which a T4 cannot launch anyway --
# beats plain XLA there by only 7%.
#
# FORWARD ONLY, float16, like the attention kernel above.


@functools.partial(jax.jit, static_argnames=('eps',))
def _ln_call(x, scale, offset, *, eps):
  m, c = x.shape
  return jax.ffi.ffi_call(
      'VoltaLayerNorm', jax.ShapeDtypeStruct((m, c), jnp.float16),
      vmap_method='sequential')(x, scale, offset, eps=np.float32(eps))


def layer_norm(x, scale, offset, *, eps=1e-5, cc=None):
  """LayerNorm over the LAST axis. None -- never a wrong answer -- if it cannot.

  scale/offset are [c] vectors; the kernel wants them in float32 and the
  activations in float16, and gives back the caller's dtype.
  """
  if x.ndim < 2 or scale is None or offset is None:
    return None
  c = x.shape[-1]
  if scale.size != c or offset.size != c:
    return None
  cc = device_cc() if cc is None else cc
  if not ops_available(cc):
    return None
  out = _ln_call(x.reshape(-1, c).astype(jnp.float16),
                 jnp.reshape(scale, (c,)).astype(jnp.float32),
                 jnp.reshape(offset, (c,)).astype(jnp.float32), eps=float(eps))
  return out.reshape(x.shape).astype(x.dtype)


@functools.partial(jax.jit, static_argnames=('sym',))
def _gdp_call(x, wp, bp, wg, bg, mask, *, sym):
  m, n = x.shape[0], wp.shape[1]
  return jax.ffi.ffi_call(sym, jax.ShapeDtypeStruct((m, n), jnp.float16),
                          vmap_method='sequential')(x, wp, bp, wg, bg, mask)


def gated_dual_proj(x, w_proj, w_gate, mask, *, cc=None):
  """TriangleMultiplication's GLU, fused, split and already channel-major.

  Computes `mask * (x @ w_proj) * sigmoid(x @ w_gate)` -- the projection and
  the gate in one kernel, so the [M, 2h] intermediates are never materialised
  -- and returns the a/b pair the triangle einsum consumes, each [h, *spatial].

  The split is INTERLEAVED, not contiguous: this fork transposes the [.., 2h]
  projection to [2h, ..] and reshapes it to [h, 2, ..], so `a` is the even
  output channels and `b` the odd ones. (Milot's own wrapper splits
  contiguously, which is AF2's layout.) Neither linear has a bias here --
  hm.Linear is use_bias=False -- so the kernel's bias operands are zeros.

  Returns None for anything outside what the kernel instantiates.
  """
  if x.ndim < 2 or w_proj.shape != w_gate.shape:
    return None
  c = x.shape[-1]
  if w_proj.shape[0] != c or w_proj.shape[1] % 2:
    return None
  h = w_proj.shape[1] // 2
  if c % 8 or h % 8:      # fp16 tensor-core alignment
    return None
  cc = device_cc() if cc is None else cc
  if not ops_available(cc):
    return None
  sym = 'VoltaGdp' if cc >= 75 else 'VoltaGdpWmma'
  spatial = tuple(x.shape[:-1])
  m = int(np.prod(spatial))
  f16 = lambda t: t.astype(jnp.float16)
  xf = f16(x.reshape(m, c))
  maskf = f16(jnp.reshape(mask, (m,)))
  zeros = jnp.zeros((h,), jnp.float16)
  out = []
  for off in (0, 1):
    half = _gdp_call(xf, f16(w_proj[:, off::2]), zeros,
                     f16(w_gate[:, off::2]), zeros, maskf, sym=sym)
    # [M, h] -> [h, *spatial], which is what the 'cik,cjk->cij' einsum wants.
    out.append(jnp.swapaxes(half, 0, 1).reshape((h,) + spatial).astype(x.dtype))
  return out[0], out[1]
