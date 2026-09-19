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

import jax.numpy as jnp

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

  from colabfold_kernels import tri_mul  # pylint: disable=g-import-not-at-top

  pack = lambda w: jnp.concatenate([w[:, 0::2], w[:, 1::2]], axis=-1)
  zeros = jnp.zeros((2 * h,), x.dtype)
  spatial = tuple(x.shape[:-1])
  left, right = tri_mul.gated_dual_proj(
      x.reshape(-1, c), pack(w_proj), zeros, pack(w_gate), zeros,
      jnp.reshape(mask, (-1,)).astype(x.dtype), split=True, channel_major=True)
  return (left.reshape((h,) + spatial), right.reshape((h,) + spatial))
