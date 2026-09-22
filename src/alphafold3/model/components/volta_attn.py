"""sm_70 / sm_75 fused attention: the only one a T4 or V100 can run.

THE KERNELS AND THE WRAPPER BELOW THEM ARE MILOT MIRDITA'S. Two packages, both
MIT: `colabfold-legacy-kernels` ships the prebuilt sm_70/sm_75 CUDA libraries
(registered as XLA FFI targets), and `colabfold-kernels` (pure Python) carries
`colabfold_kernels.volta`, the wrapper that dlopens them, registers the
symbols and holds the `custom_vjp`. THIS FILE IS ONLY THE ADAPTER between that
wrapper and this fork's calling convention -- the plumbing it used to duplicate
now lives upstream, which is where head dims, tilings and symbol names are
decided.

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

THE BACKWARD IS UPSTREAM AS OF 0.4.0 (2026-09-22), for both families:
`VoltaMmaBwd` (sm_75) and `VoltaWmmaBwd` (sm_70), each giving dQ, dK, dV and
dBias. dBias is the point -- AF3 reaches the pair representation through the
attention bias, so a design step on a T4 needs it and ColabFold never did. It
began here (this fork wrote both kernels, and both commits are in
mirditalab/colabfold-legacy-kernels main), and it is now released, refactored
and tested there; 0.4.0 also widened the head dims to AF3's 24/48/96 and pads
on the wmma side, so more of this model's shapes take the kernel than before.

An OLDER wheel has no backward at all, and jax says so plainly --
`ValueError: The FFI call to VoltaMma cannot be differentiated`. The design
path backprops through the trunk and GridSelfAttention is in the trunk, so on
such a wheel this must never be selected for a differentiable call, which is
why `platform.attention_config` takes `differentiable=` and consults
`bwd_installed` before it answers 'volta'.

FLOAT16, not bfloat16: Volta and Turing tensor cores have no bf16. Inputs are
cast in and the result is cast back to the caller's dtype.
"""

from __future__ import annotations

import importlib.util
import os

import jax
import jax.numpy as jnp


def _volta():
  """Milot's wrapper module. Imports jax, so never at module scope."""
  from colabfold_kernels import volta  # pylint: disable=g-import-not-at-top
  return volta


def _grads():
  from colabfold_kernels import grads  # pylint: disable=g-import-not-at-top
  return grads


def installed() -> bool:
  """True if both packages are importable. No jax, no dlopen.

  BOTH: the libraries live in `colabfold_legacy_kernels` and the wrapper that
  loads them in `colabfold_kernels`. The second is pure Python and tiny, but a
  wheel-only install (which is what ColabFold's own AF2 path had) satisfies the
  first and not the second.
  """
  return all(importlib.util.find_spec(m) is not None
             for m in ('colabfold_legacy_kernels', 'colabfold_kernels'))


def available(cc) -> bool:
  """dlopen the attention library for this device and register its target."""
  if cc is None or not installed():
    return False
  return bool(_volta().available(int(cc)))


def bwd_installed(cc) -> bool:
  """True if this wheel carries a backward library for this card. No dlopen.

  `platform.attention_config` is called from notebook cells that want an answer
  in milliseconds and before anything has claimed the GPU, so it cannot dlopen
  to find out -- the same reason `installed()` is a find_spec. An 0.2.0 wheel
  has no `attention_bwd` entry at all and raises KeyError here, not
  FileNotFoundError, which is why both are caught.
  """
  if cc is None or int(cc) < 70 or not installed():
    return False
  try:
    import colabfold_legacy_kernels as clk  # pylint: disable=g-import-not-at-top

    return os.path.exists(clk.library_path('attention_bwd', int(cc)))
  except Exception:  # pylint: disable=broad-except
    return False


def bwd_available(cc) -> bool:
  """dlopen the backward library of this card's family, if the wheel has one.

  A wheel without those symbols answers False and the caller keeps XLA. Note
  that 0.4.0 folded the forward-with-lse into the FORWARD symbol (one
  `want_lse` attribute) rather than exporting a second one, so there is nothing
  to load here but the backward library itself.
  """
  if cc is None or int(cc) < 70 or not installed():
    return False
  return bool(_volta().bwd_available(int(cc)))


def ops_available(cc) -> bool:
  """dlopen the LayerNorm and gated-dual-projection libraries.

  The LayerNorm kernel is Volta-capable; the CUTLASS gated dual projection is
  sm_75+, and sm_70 gets a separate wmma build.
  """
  if cc is None or not installed():
    return False
  return bool(_volta().ops_available(int(cc)))


def device_cc():
  """The compute capability of device 0 as an integer (75), or None."""
  try:
    cap = getattr(jax.devices()[0], 'compute_capability', None)
  except Exception:  # pylint: disable=broad-except
    return None
  return None if cap is None else int(float(cap) * 10)


def attention(q, k, v, *, mask, bias, scale, cc, block_q=64, block_k=32):
  """This fork's attention contract, on the legacy kernel. None if it cannot.

  q/k/v are (batch, seq, heads, dim); bias is (1, heads, q, k); mask is
  boolean and broadcasts over the key axis. Returns None -- never a wrong
  answer -- when the shapes are outside what the kernel instantiates, so the
  caller can fall back to XLA.

  The kernel is differentiable whenever the backward library loads, and the
  forward pays for it only by writing one extra [n, h, sq] float (the softmax
  statistic), so no caller has to declare in advance whether someone will
  differentiate the graph it builds.
  """
  if bias is None or mask is None or q.ndim != 4:
    return None
  b, sq, h, c = q.shape
  sk = k.shape[1]
  if c != v.shape[-1] or not available(cc):
    return None
  # WHICH HEAD DIMS EXIST IS THE PACKAGE'S ANSWER, NOT OURS. 0.4.0 instantiates
  # AF3's 24/48/96 as well as AF2's 8/16/32/64, and on the wmma side pads up to
  # the next one it has; a hardcoded list here went stale the day it shipped.
  if not _volta().supports(int(c), int(cc)):
    return None
  # A broadcast bias (AF2's MSA column attention passes (batch, 1, 1, seq))
  # is not what the kernel indexes; it wants one [h, sq, sk] slab.
  if bias.shape[-3:] != (h, sq, sk) or bias.shape[0] != 1:
    return None
  if mask.shape[-1] != sk:
    return None
  in_dtype = q.dtype
  f16 = lambda t: jnp.swapaxes(t, 1, 2).astype(jnp.float16)   # -> [b, h, S, c]
  # Their entry point takes AF2's additive mask_bias [b, 1, 1, sk] and reduces
  # it back to a boolean itself (`> -1e3`); ours is the boolean.
  m = jnp.broadcast_to(
      jnp.reshape(mask.astype(jnp.bool_), mask.shape[:1] + (-1,))[:, -sk:],
      (b, sk))
  mask_bias = jnp.where(m, 0.0, -1e4).astype(jnp.float16)[:, None, None, :]
  out = _volta().volta_attention(
      f16(q), f16(k), f16(v), mask_bias,
      jnp.reshape(bias, (h, sq, sk)).astype(jnp.float16), float(scale),
      int(cc), block_q=block_q, block_k=block_k)
  return jnp.swapaxes(out, 1, 2).astype(in_dtype)     # back to [b, S, h, c]


# ---------------------------------------------------------------------------
# The other two kernels the same package ships: a fused LayerNorm and a masked,
# sigmoid-gated dual projection. Both are Milot Mirdita's; the wrappers below
# adapt them to this fork's TriangleMultiplication, which is where they land:
# at N=384 that module spends 0.321 ms in its input LayerNorm (39% of the
# card's bandwidth peak) and 0.869 ms in the GLU, and tokamax's GLU -- which a
# T4 cannot launch anyway -- beats plain XLA there by only 7%.
#
# float16, like the attention kernel. NEITHER HAS A BACKWARD KERNEL in either
# of Milot's packages, so both take their gradient from a recomputation in XLA
# -- `colabfold_kernels.grads` now holds that wrapper (it used to be `xla_vjp`
# here), and it is what makes a design run on a pre-Ampere card survive
# `glu_kernel='auto'` following the attention pick: a raw ffi_call is not
# differentiable at all, and a design run that reached one died with
# `The FFI call to VoltaLayerNorm cannot be differentiated`.


def xla_vjp(ref):
  """A backward that is `jax.vjp` of a plain-XLA twin of the kernel.

  Kept because pallas_attn's GLU wraps its own forward (an interleaved,
  channel-major split Milot's wrapper does not have) and so cannot use
  `colabfold_kernels.grads` directly.

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


def ln_reference(x, scale, offset, eps=1e-5):
  """The LayerNorm kernel's math in plain XLA, in float32.

  Public for the same reason `gdp_reference` is: `dev/oracles/trimul_grad_check`
  substitutes it for the kernel to get a reference that differs from the real
  module in nothing BUT the kernel.
  """
  xf = x.astype(jnp.float32)
  mu = jnp.mean(xf, -1, keepdims=True)
  var = jnp.mean((xf - mu) ** 2, -1, keepdims=True)
  norm = (xf - mu) * jax.lax.rsqrt(var + eps)
  return (norm * scale + offset).astype(x.dtype)


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
  fn = _grads().differentiable_layer_norm(_volta().volta_layer_norm)
  out = fn(x.reshape(-1, c), jnp.reshape(scale, (c,)),
           jnp.reshape(offset, (c,)), eps=float(eps))
  return out.reshape(x.shape)


def gdp_reference(x, w_proj, w_gate, mask):
  """The kernel's own math in plain XLA: mask * proj * sigmoid(gate).

  Still here because pallas_attn's GLU takes its gradient from it.
  """
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


def gated_dual_proj(x, w_proj, w_gate, mask, *, cc=None):
  """TriangleMultiplication's GLU, fused, split and already channel-major.

  Computes `mask * (x @ w_proj) * sigmoid(x @ w_gate)` -- the projection and
  the gate in one kernel, so the [M, 2h] intermediates are never materialised
  -- and returns the a/b pair the triangle einsum consumes, each [h, *spatial].

  The split is INTERLEAVED, not contiguous: this fork transposes the [.., 2h]
  projection to [2h, ..] and reshapes it to [h, 2, ..], so `a` is the even
  output channels and `b` the odd ones. (Milot's own `split=True` is
  contiguous, which is AF2's layout, so this calls his kernel once per half
  with a strided weight slice instead -- the slicing is an ordinary jnp op and
  transposes for free.) Neither linear has a bias here -- hm.Linear is
  use_bias=False -- so the kernel's bias operands are zeros.

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
  fwd = lambda *a, **kw: _volta().volta_gated_dual_proj(*a, cc=int(cc), **kw)
  fn = _grads().differentiable_gated_dual_proj(fwd, jax.nn.sigmoid)
  spatial = tuple(x.shape[:-1])
  xf = x.reshape(-1, c)
  maskf = jnp.reshape(mask, (-1,))
  zeros = jnp.zeros((h,), x.dtype)
  out = []
  for off in (0, 1):
    half = fn(xf, w_proj[:, off::2], zeros, w_gate[:, off::2], zeros, maskf)
    # [M, h] -> [h, *spatial], which is what the 'cik,cjk->cij' einsum wants.
    out.append(jnp.swapaxes(half, 0, 1).reshape((h,) + spatial).astype(x.dtype))
  return out[0], out[1]
