'''flash-attention dispatch with a cuDNN odd-length backprop guard.

cuDNN FlashAttention with an additive bias rejects ODD query/key sequence lengths
UNDER BACKPROP -- it raises "vjp not implemented" (which reads as if there is no
gradient at all, but the forward pass is fine). Prediction/scoring (forward-only)
is unaffected at any length; only the gradient-design path hits it, and only at odd
lengths. Since a receptor+binder complex is frequently an odd token count, and cuDNN
is the default attention on Ampere/consumer GPUs (see platform.py), the design graph
would crash without this guard.

Fix (as in BindCraft2): pad the odd axis up to even, mask the padded key column with
a large negative bias so real queries ignore it, run the kernel, then slice the padded
query rows back off. Numerically identical to the unpadded result (padded weights are
0); verified grad matches the XLA path to bf16 round-off. Even lengths and non-cuDNN
backends are a straight pass-through (zero overhead).

Layout matches tokamax.dot_product_attention: q,k,v = (..., seq, heads, dim);
bias = (..., heads, q, k); mask broadcasts over the key axis (last dim).
'''
from __future__ import annotations

import jax.numpy as jnp
import tokamax

_NEG = 1e8


def dot_product_attention(q, k, v, *, mask=None, bias=None, implementation=None,
                          scale=None):
  '''tokamax.dot_product_attention, guarded so cuDNN backprop works at odd lengths.'''
  if implementation == 'cudnn':
    Q, K = q.shape[-3], k.shape[-3]
    padq, padk = Q % 2, K % 2
    if padq or padk:
      zc = lambda n: [(0, 0)] * n
      q = jnp.pad(q, zc(q.ndim - 3) + [(0, padq), (0, 0), (0, 0)])
      k = jnp.pad(k, zc(k.ndim - 3) + [(0, padk), (0, 0), (0, 0)])
      v = jnp.pad(v, zc(v.ndim - 3) + [(0, padk), (0, 0), (0, 0)])
      if bias is not None:
        bias = jnp.pad(bias, zc(bias.ndim - 2) + [(0, padq), (0, padk)])
        if padk:
          bias = bias.at[..., K:].set(jnp.asarray(-_NEG, bias.dtype))
      if mask is not None:
        mask = jnp.pad(mask, zc(mask.ndim - 1) + [(0, padk)])  # pad key axis False
      out = tokamax.dot_product_attention(
          q, k, v, mask=mask, bias=bias, implementation=implementation, scale=scale)
      return out[..., :Q, :, :]
  return tokamax.dot_product_attention(
      q, k, v, mask=mask, bias=bias, implementation=implementation, scale=scale)
