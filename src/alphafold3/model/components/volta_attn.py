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

UPSTREAM IS FORWARD ONLY. Its libraries export no backward symbol, and jax says
so plainly:

    ValueError: The FFI call to `VoltaMma` cannot be differentiated.

The design path backprops through the trunk and GridSelfAttention is in the
trunk, so on such a wheel this must never be selected for a differentiable
call -- which is why `platform.attention_config` takes `differentiable=` and
consults `bwd_installed` before it answers 'volta'.

The FORK's wheel does have one, for both families: `VoltaMmaBwd` (sm_75) and
`VoltaWmmaBwd` (sm_70), returning dQ, dK, dV and dBias. dBias is the point --
AF3 reaches the pair representation through the attention bias. `attention()`
below takes the differentiable path whenever those symbols load, and the
forward-only one otherwise, so a caller never has to know which wheel is
installed.

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


def _symbols_diff(cc: int) -> tuple[str, str]:
  """The forward-with-lse and backward symbols of this card's family."""
  base = _symbol(cc)
  return base + 'Fwd', base + 'Bwd'


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


def bwd_installed(cc) -> bool:
  """True if this wheel carries a backward library for this card. No dlopen.

  `platform.attention_config` is called from notebook cells that want an answer
  in milliseconds and before anything has claimed the GPU, so it cannot dlopen
  to find out -- the same reason `installed()` is a find_spec.
  """
  if cc is None or int(cc) < 70:
    return False
  try:
    import colabfold_legacy_kernels as clk

    return os.path.exists(clk.library_path('attention_bwd', int(cc)))
  except Exception:  # pylint: disable=broad-except
    return False


def bwd_available(cc) -> bool:
  """dlopen the backward library of this card's family, if the wheel has one.

  UPSTREAM HAS NO BACKWARD -- the libraries export no such symbol and jax says
  `The FFI call to VoltaMma cannot be differentiated`, which is why every
  gradient path on a T4 falls back to XLA. `VoltaMmaBwd` (dQ, dK, dV and
  dBias) and `VoltaMmaFwd` (the forward plus the softmax statistic the backward
  needs) come from the fork:

      github.com/sokrypton/colabfold-legacy-kernels, branch `backward`

  Both families have one: `VoltaWmmaBwd` is the sm_70 twin, which cannot reuse
  the mma trick of feeding a computed tile straight into the next GEMM and
  round-trips P and dS through shared memory instead. A wheel without these
  symbols simply answers False and the caller keeps XLA.
  """
  if cc is None or int(cc) < 70:
    return False
  cc = int(cc)
  # TWO LIBRARIES, not one. VoltaMmaFwd is in the attention library (it is the
  # same kernel with one more result); VoltaMmaBwd is its own. Asking for both
  # from 'attention' makes getattr fail, this return False, and the wrapper
  # silently take the forward-only path -- which is exactly what a real T4
  # did, with `The FFI call to VoltaMma cannot be differentiated` landing in
  # the caller. It passed locally only because that test patched `_load` with
  # a loader that opened both files: it was testing the kernels, not the
  # wiring.
  sym_fwd, sym_bwd = _symbols_diff(cc)
  return (_load('attention', cc, (sym_fwd,))
          and _load('attention_bwd', cc, (sym_bwd,)))


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


@functools.partial(jax.jit,
                   static_argnames=('sym', 'sm_scale', 'block_q', 'block_k'))
def _call_fwd(q, k, v, bias, kmask, *, sym, sm_scale, block_q, block_k):
  """Volta{Mma,Wmma}Fwd: the forward, plus `lse` in the log2 domain."""
  n, h, sq, d = q.shape
  return jax.ffi.ffi_call(
      sym,
      (jax.ShapeDtypeStruct((n, h, sq, d), jnp.float16),
       jax.ShapeDtypeStruct((n, h, sq), jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, scale=np.float32(sm_scale),
          block_q=np.int64(block_q), block_k=np.int64(block_k))


@functools.partial(jax.jit,
                   static_argnames=('sym', 'sm_scale', 'block_q', 'block_k'))
def _call_bwd(q, k, v, bias, kmask, dout, lse, delta, *, sym, sm_scale, block_q,
              block_k):
  """Volta{Mma,Wmma}Bwd: dQ, dK, dV and dBias (summed over the batch)."""
  n, h, sq, d = q.shape
  sk = k.shape[2]
  return jax.ffi.ffi_call(
      sym,
      (jax.ShapeDtypeStruct(q.shape, jnp.float16),
       jax.ShapeDtypeStruct(k.shape, jnp.float16),
       jax.ShapeDtypeStruct(v.shape, jnp.float16),
       jax.ShapeDtypeStruct((h, sq, sk), jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, dout, lse, delta, scale=np.float32(sm_scale),
          block_q=np.int64(block_q), block_k=np.int64(block_k))


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _attention_diff(q, k, v, bias, kmask, sym, sm_scale, block_q, block_k):
  """[b, h, S, c] float16 attention that can be differentiated."""
  out, _ = _call_fwd(q, k, v, bias, kmask, sym=sym + 'Fwd', sm_scale=sm_scale,
                     block_q=block_q, block_k=block_k)
  return out


def _attention_diff_fwd(q, k, v, bias, kmask, sym, sm_scale, block_q, block_k):
  out, lse = _call_fwd(q, k, v, bias, kmask, sym=sym + 'Fwd', sm_scale=sm_scale,
                       block_q=block_q, block_k=block_k)
  return out, (q, k, v, bias, kmask, out, lse)


def _attention_diff_bwd(sym, sm_scale, block_q, block_k, res, dout):
  q, k, v, bias, kmask, out, lse = res
  # rowsum(dout * out): one line of XLA, so the kernel does not spend a launch
  # on it and the backward keeps a single entry point.
  delta = jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32), axis=-1)
  dq, dk, dv, dbias = _call_bwd(
      q, k, v, bias, kmask, dout.astype(jnp.float16), lse, delta,
      sym=sym + 'Bwd', sm_scale=sm_scale, block_q=block_q, block_k=block_k)
  # dBias comes back [h, Sq, Sk] and summed over the batch, which is the
  # gradient of a bias that WAS broadcast over it -- so it re-enters with the
  # leading axis the caller passed in.
  return dq, dk, dv, dbias.reshape(bias.shape).astype(bias.dtype), None


_attention_diff.defvjp(_attention_diff_fwd, _attention_diff_bwd)


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
  fbias = jnp.reshape(bias, (h, sq, sk)).astype(jnp.float16)
  # WHENEVER THE WHEEL HAS A BACKWARD, take the differentiable path. It is the
  # same forward kernel with one extra [n, h, sq] float output (the softmax
  # statistic), so a prediction-only caller pays almost nothing for it, and no
  # caller has to declare in advance whether someone will differentiate the
  # graph it builds.
  if bwd_available(cc):
    out = _attention_diff(f16(q), f16(k), f16(v), fbias, kmask,
                          _symbol(int(cc)), float(scale), block_q, block_k)
  else:
    out = _call(f16(q), f16(k), f16(v), fbias, kmask,
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
# float16, like the attention kernel above. Neither has a backward kernel, so
# both take their gradient from an XLA recomputation -- see `xla_vjp`.


def xla_vjp(ref):
  """A backward that is `jax.vjp` of a plain-XLA twin of the kernel.

  Public because pallas_attn's GLU needs exactly the same treatment, and
  `gdp_reference` below is the same math for both packages.

  THE FUSED GLU AND LAYERNORM HAVE NO BACKWARD KERNEL, in either of Milot's
  packages (his Pallas GLU is a bare `pallas_call`, these are bare `ffi_call`s).
  A raw call is not differentiable at all, so a design run that reached one died
  with `The FFI call to VoltaLayerNorm cannot be differentiated` -- and
  `glu_kernel='auto'` FOLLOWS the attention pick, so enabling the attention
  backward on a pre-Ampere card armed exactly that.

  Recomputing the forward in XLA to get the gradient is what a remat would do
  anyway, and it keeps the fused kernel on the forward pass inside a design
  step. It also makes the gradient the exact one of the fp32 reference rather
  than of the fp16 kernel, which is the better of the two.
  """
  def bwd(*args):
    res, cotangent = args[-2], args[-1]
    static = args[:-2]
    _, vjp = jax.vjp(lambda *a: ref(*a, *static), *res)
    return vjp(cotangent)
  return bwd


def _ln_ref(x, scale, offset, eps):
  xf = x.astype(jnp.float32)
  mu = jnp.mean(xf, -1, keepdims=True)
  var = jnp.mean((xf - mu) ** 2, -1, keepdims=True)
  norm = (xf - mu) * jax.lax.rsqrt(var + eps)
  return (norm * scale + offset).astype(x.dtype)


@functools.partial(jax.jit, static_argnames=('eps',))
def _ln_call(x, scale, offset, *, eps):
  m, c = x.shape
  return jax.ffi.ffi_call(
      'VoltaLayerNorm', jax.ShapeDtypeStruct((m, c), jnp.float16),
      vmap_method='sequential')(x, scale, offset, eps=np.float32(eps))


def _ln_impl(x, scale, offset, eps):
  c = x.shape[-1]
  out = _ln_call(x.reshape(-1, c).astype(jnp.float16),
                 scale.astype(jnp.float32), offset.astype(jnp.float32), eps=eps)
  return out.reshape(x.shape).astype(x.dtype)


_ln_diff = jax.custom_vjp(_ln_impl, nondiff_argnums=(3,))
_ln_diff.defvjp(lambda x, s, o, eps: (_ln_impl(x, s, o, eps), (x, s, o)),
                xla_vjp(_ln_ref))


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
  return _ln_diff(x, jnp.reshape(scale, (c,)), jnp.reshape(offset, (c,)),
                  float(eps))


@functools.partial(jax.jit, static_argnames=('sym',))
def _gdp_call(x, wp, bp, wg, bg, mask, *, sym):
  m, n = x.shape[0], wp.shape[1]
  return jax.ffi.ffi_call(sym, jax.ShapeDtypeStruct((m, n), jnp.float16),
                          vmap_method='sequential')(x, wp, bp, wg, bg, mask)


def gdp_reference(x, w_proj, w_gate, mask):
  """The kernel's own math in plain XLA: mask * proj * sigmoid(gate)."""
  c = x.shape[-1]
  h = w_proj.shape[1] // 2
  spatial = tuple(x.shape[:-1])
  xf = x.reshape(-1, c).astype(jnp.float32)
  m = jnp.reshape(mask, (-1,)).astype(jnp.float32)[:, None]
  out = []
  for off in (0, 1):
    proj = xf @ w_proj[:, off::2].astype(jnp.float32)
    gate = xf @ w_gate[:, off::2].astype(jnp.float32)
    o = m * proj * jax.nn.sigmoid(gate)
    out.append(jnp.swapaxes(o, 0, 1).reshape((h,) + spatial).astype(x.dtype))
  return out[0], out[1]


def _gdp_impl(x, w_proj, w_gate, mask, sym):
  c = x.shape[-1]
  h = w_proj.shape[1] // 2
  spatial = tuple(x.shape[:-1])
  f16 = lambda t: t.astype(jnp.float16)
  xf = f16(x.reshape(-1, c))
  maskf = f16(jnp.reshape(mask, (-1,)))
  zeros = jnp.zeros((h,), jnp.float16)
  out = []
  for off in (0, 1):
    half = _gdp_call(xf, f16(w_proj[:, off::2]), zeros,
                     f16(w_gate[:, off::2]), zeros, maskf, sym=sym)
    # [M, h] -> [h, *spatial], which is what the 'cik,cjk->cij' einsum wants.
    out.append(jnp.swapaxes(half, 0, 1).reshape((h,) + spatial).astype(x.dtype))
  return out[0], out[1]


_gdp_diff = jax.custom_vjp(_gdp_impl, nondiff_argnums=(4,))
_gdp_diff.defvjp(
    lambda x, wp, wg, m, sym: (_gdp_impl(x, wp, wg, m, sym), (x, wp, wg, m)),
    xla_vjp(lambda x, wp, wg, m, sym: gdp_reference(x, wp, wg, m)))


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
  return _gdp_diff(x, w_proj, w_gate, mask, sym)
