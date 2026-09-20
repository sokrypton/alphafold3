"""Milot Mirdita's Pallas kernels, for the cards the legacy ones are not for.

THE KERNELS THIS DRIVES ARE MILOT MIRDITA'S. `colabfold-kernels` (MIT, pure
Python) is the Ampere-and-newer half of the pair whose other half is
`colabfold-legacy-kernels` (see volta_attn.py): Pallas/Triton flash attention
with a NON-BATCHED bias, a fused LayerNorm, and a masked sigmoid-gated dual
projection. This file adapts them to this fork's calling convention.

WHY, GIVEN WE ALREADY HAVE cuDNN AND tokamax. Two reasons, both measured here
on an A10 (compute capability 8.6) at this model's own triangle-attention
shape, N=384, 4 heads, head dim 32:

    XLA          8.635 ms
    cuDNN        2.409 ms     what platform.py picks for this card today
    tokamax      FAILS        'Not supported on NVIDIA A10'
    this         0.723 ms     3.3x cuDNN, 11.9x XLA

tokamax gates Triton on `compute_capability >= 8.0` and then asks for more
shared memory than an Ada or consumer-Ampere card has; this kernel sizes its
blocks against the device instead, so it runs where tokamax refuses. And the
gated dual projection is 0.512 ms against tokamax's 0.872 and XLA's 0.941 at
M=147456, K=128 -- tokamax's GLU is worth only 7% over plain XLA there.

The fused LayerNorm is deliberately NOT wired: 0.164 ms against XLA's 0.166 on
the same tensor. XLA already reaches the card's bandwidth.

FORWARD ONLY. A Pallas call has no VJP, so `platform.attention_config` never
answers 'pallas' to a differentiable caller -- the same rule as the sm_70/75
kernels.
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp

from alphafold3.model.components import volta_attn

_NEG = -1e9      # the kernel reads mask_bias > -1e3 as "this key is real"


def installed() -> bool:
  """True if the kernel package is importable. Imports nothing."""
  return importlib.util.find_spec('colabfold_kernels') is not None


def attention(q, k, v, *, mask, bias, scale):
  """This fork's attention contract, on the Pallas kernel. None if it cannot.

  q/k/v are (batch, seq, heads, dim); bias is (1, heads, q, k) or None; mask is
  boolean and broadcasts over the key axis. Returns None -- never a wrong
  answer -- for a shape the kernel does not take, so the caller falls back.
  """
  if not installed() or q.ndim != 4 or mask is None:
    return None
  if q.dtype not in (jnp.bfloat16, jnp.float16):
    return None
  b, sq, h, c = q.shape
  sk = k.shape[1]
  # One BlockSpec serves k and v, so the two must share a channel count.
  if k.shape[-1] != c or v.shape[-1] != c:
    return None
  if bias is not None and (bias.shape[-3:] != (h, sq, sk) or bias.shape[0] != 1):
    return None

  from colabfold_kernels import tri_flash  # pylint: disable=g-import-not-at-top

  # The kernel takes AF2's additive mask_bias [b, 1, 1, sk], which it reduces
  # back to a boolean; ours is the boolean itself, broadcast over the key axis.
  m = jnp.broadcast_to(
      jnp.reshape(mask.astype(jnp.bool_), mask.shape[:1] + (-1,))[:, -sk:],
      (b, sk))
  mask_bias = jnp.where(m, 0.0, _NEG).astype(jnp.float32)[:, None, None, :]
  nb_bias = None if bias is None else jnp.reshape(bias, (h, sq, sk))
  out = tri_flash.pallas_attention(q, k, v, mask_bias, nb_bias, float(scale),
                                   seq_major=True)
  return out.astype(q.dtype)


_GDP_BLOCK_M = 64      # tri_mul.gated_dual_proj's own default


def _gdp_fits(k, c, itemsize):
  """Will the GLU kernel's shared memory fit on this device?

  IT DOES NOT ALWAYS. The kernel stages both weight matrices whole, so its
  shared memory grows as the product of the channel counts while the activation
  tile stays fixed -- and a Pallas launch failure is NOT catchable at trace
  time, exactly like the tokamax problem this backend exists to route around.
  protenix2 folds through a 256-wide pair stack (x [N,N,256], weights
  [256,512]) and asked for 294912 B against an A10's 101376: the whole fold
  died with RESOURCE_EXHAUSTED, while every other model here is 128-wide and
  sits under the limit.

  Measured on an A10 by probing the kernel directly -- the model is
  `2*K*c + block_m*K` elements, and it predicts every observed request:

      K=128 c=256  147456 B (measured 147456)     K=256 c=128  163840 (164096)
      K=256 c=256  294912 B (measured 294912)     K=512 c=128  327680 (327936)

  and the shapes that fit (K=c=64, K=c=128, K=256 c=64, K=64 c=256) all land
  under the limit. The 256 B discrepancy is the bias operands; the +1024 below
  covers it.
  """
  import jax  # pylint: disable=g-import-not-at-top

  try:
    limit = min(d.shared_memory_per_block_optin for d in jax.local_devices()
                if d.platform == 'gpu')
  except Exception:  # pylint: disable=broad-except
    return False
  if not limit:
    return False
  return itemsize * (2 * k * c + _GDP_BLOCK_M * k) + 1024 <= limit


def _pow2(n):
  """Triton lowers only power-of-two shapes; K=192 and K=160 raise at trace."""
  return n > 0 and (n & (n - 1)) == 0


def _gdp_impl(x, w_proj, w_gate, mask):
  from colabfold_kernels import tri_mul  # pylint: disable=g-import-not-at-top

  c = x.shape[-1]
  h = w_proj.shape[1] // 2
  pack = lambda w: jnp.concatenate([w[:, 0::2], w[:, 1::2]], axis=-1)
  zeros = jnp.zeros((2 * h,), x.dtype)
  spatial = tuple(x.shape[:-1])
  left, right = tri_mul.gated_dual_proj(
      x.reshape(-1, c), pack(w_proj), zeros, pack(w_gate), zeros,
      jnp.reshape(mask, (-1,)).astype(x.dtype), split=True, channel_major=True)
  return (left.reshape((h,) + spatial), right.reshape((h,) + spatial))


# A bare `pallas_call` HAS NO VJP, so this would die under jax.grad exactly as
# the volta one did. Same remedy, and the same reasoning: see
# `volta_attn.xla_vjp`. It costs one recomputation of a forward that is
# cheaper than the XLA path it replaces.
_gdp_diff = jax.custom_vjp(_gdp_impl)
_gdp_diff.defvjp(
    lambda x, wp, wg, m: (_gdp_impl(x, wp, wg, m), (x, wp, wg, m)),
    volta_attn.xla_vjp(volta_attn.gdp_reference))


def gated_dual_proj(x, w_proj, w_gate, mask):
  """TriangleMultiplication's GLU, fused, split and already channel-major.

  Same contract as volta_attn.gated_dual_proj -- see the note there on why the
  split is INTERLEAVED here and contiguous in Milot's own wrapper. The weights
  are re-packed into his layout (a [c, 2h] copy, which is weight-sized, not
  activation-sized), so his `split=True, channel_major=True` path writes the
  two halves transposed straight out of the kernel and the triangle einsum
  consumes them with no further copy.
  """
  if not installed() or x.ndim < 2 or w_proj.shape != w_gate.shape:
    return None
  c = x.shape[-1]
  if w_proj.shape[0] != c or w_proj.shape[1] % 2:
    return None
  if x.dtype not in (jnp.bfloat16, jnp.float16):
    return None
  h = w_proj.shape[1] // 2
  if not (_pow2(c) and _pow2(h)):
    return None
  if not _gdp_fits(c, h, jnp.dtype(x.dtype).itemsize):
    return None

  return _gdp_diff(x, w_proj, w_gate, mask)
