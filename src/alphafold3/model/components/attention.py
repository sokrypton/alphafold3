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


_TOKAMAX_ADA = {'done': False}


def _enable_tokamax_on_ada():
  """Let tokamax's Triton kernels run on Ada / consumer Ampere.

  See the note at the call site. Idempotent, and a no-op on any card tokamax
  already accepts, so a datacenter run is untouched.
  """
  _TOKAMAX_ADA['done'] = True
  try:
    import jax

    from tokamax._src import gpu_utils

    caps = [float(d.compute_capability) for d in jax.local_devices()
            if getattr(d, 'compute_capability', None)]
    if not caps or all(c == 8.0 or c >= 9.0 for c in caps):
      return                       # tokamax accepts these already
    real = gpu_utils.has_triton_support

    def _allow(device=None):
      if real(device):
        return True
      dev = device
      if dev is None:
        try:
          dev = jax.local_devices()[0]
        except Exception:  # pylint: disable=broad-except
          return False
      cap = getattr(dev, 'compute_capability', None)
      return cap is not None and 8.0 <= float(cap) < 9.0
    gpu_utils.has_triton_support = _allow
    # the op reads it through its own module reference
    from tokamax._src.ops.attention import pallas_triton as _pt
    _pt.gpu_utils.has_triton_support = _allow
  except Exception:  # pylint: disable=broad-except
    pass


def dot_product_attention(q, k, v, *, mask=None, bias=None, implementation=None,
                          scale=None):
  '''tokamax.dot_product_attention, guarded so cuDNN backprop works at odd lengths.'''
  # cuDNN's fused attention takes fp16/bf16 only, and rejects fp32 from several
  # frames inside jax with "Q must be fp16/bf16/fp8_e4m3fn/fp8_e5m2" -- a message
  # that names neither the caller nor the tensor. Dispatching to it in fp32 is
  # never right, so fall back to XLA, which is what a non-Ada machine would have
  # picked anyway. Surfaced by folding with a real MSA at use_bfloat16=False:
  # the MSA axis is deep enough to cross the flash-attention length threshold,
  # which single-sequence design never does.
  # sm_70/sm_75 have no fused attention at all otherwise -- cuDNN wants SM80,
  # tokamax has no kernel, XLA gates Pallas/Triton at sm_80. `volta` is Milot
  # Mirdita's colabfold-legacy-kernels, 3x the XLA path on a T4. It returns
  # None rather than a wrong answer when the shapes are outside what it
  # instantiates (a broadcast bias, an odd head dim), so this falls through.
  # FORWARD ONLY: it cannot be differentiated, which is why nothing asks for
  # it on a gradient path -- see platform.attention_config(differentiable=).
  # Milot Mirdita's Pallas kernel (colabfold-kernels): a flash attention with a
  # non-batched bias that sizes its blocks against the device, so it runs on the
  # Ada / consumer-Ampere cards where tokamax's Triton arm answers 'Not
  # supported on NVIDIA A10' -- and at 0.723 ms against cuDNN's 2.409 at
  # N=384 it is 3.3x the backend those cards use today. Forward only (a Pallas
  # call has no VJP), which is why platform.attention_config never answers
  # 'pallas' to a differentiable caller. Falls through to XLA on any shape it
  # does not take.
  # TOKAMAX REFUSES ADA BY NAME, AND THE PREMISE IS WRONG FOR OUR SHAPES.
  # `gpu_utils.has_triton_support` answers `cc == 8.0 or cc >= 9.0`, commented
  # "Ada/L4 lack shared memory". Every shape this model uses launches there --
  # see platform.attention_config's differentiable branch for the map and the
  # numbers -- and a GRADIENT on Ada needs it: cuDNN's backward wants 22.96 GiB
  # at 384 residues where the flash backward runs in 4.2 s.
  #
  # Patched once per process, and only when Triton was actually asked for on a
  # card tokamax would refuse. The refusal is a NotImplementedError at TRACE
  # time, so the caller below can fall back; a shared-memory failure at LAUNCH
  # could not be caught, which is why the shapes were mapped first.
  if implementation == 'triton' and not _TOKAMAX_ADA['done']:
    _enable_tokamax_on_ada()

  if implementation == 'pallas':
    from alphafold3.model.components import pallas_attn
    out = pallas_attn.attention(
        q, k, v, mask=mask, bias=bias,
        scale=scale if scale is not None else q.shape[-1] ** -0.5)
    if out is not None:
      return out
    implementation = 'xla'

  if implementation == 'volta':
    from alphafold3.model.components import volta_attn
    import jax
    try:
      cc = getattr(jax.devices()[0], 'compute_capability', None)
    except Exception:  # pylint: disable=broad-except
      cc = None
    out = volta_attn.attention(
        q, k, v, mask=mask, bias=bias,
        scale=scale if scale is not None else q.shape[-1] ** -0.5,
        cc=None if cc is None else int(float(cc) * 10))
    if out is not None:
      return out
    implementation = 'xla'

  if implementation == 'cudnn' and q.dtype == jnp.float32:
    implementation = 'xla'
  if implementation == 'cudnn':
    Q, K = q.shape[-3], k.shape[-3]
    padq, padk = Q % 2, K % 2
    if padq or padk:
      zc = lambda n: [(0, 0)] * n
      q = jnp.pad(q, zc(q.ndim - 3) + [(0, padq), (0, 0), (0, 0)])
      k = jnp.pad(k, zc(k.ndim - 3) + [(0, padk), (0, 0), (0, 0)])
      v = jnp.pad(v, zc(v.ndim - 3) + [(0, padk), (0, 0), (0, 0)])
      # Pad a bias axis ONLY if it is full-size. A bias is allowed to be
      # BROADCAST along the query or key axis -- AF2's MSA column attention
      # passes (batch, 1, 1, num_seq) -- and padding a length-1 axis turns it
      # into length 2, which no longer broadcasts against the logits. The
      # failure then arrives from inside tokamax as a jaxtyping dump naming
      # neither this function nor the caller, and only at ODD lengths, which is
      # why it went unseen: it needs a broadcast bias and an odd sequence length
      # at once (here, an MSA of 513).
      if bias is not None:
        bq = padq if bias.shape[-2] == Q else 0
        bk = padk if bias.shape[-1] == K else 0
        if bq or bk:
          bias = jnp.pad(bias, zc(bias.ndim - 2) + [(0, bq), (0, bk)])
        # Only mask out the padded keys where the bias actually indexes them; a
        # key-broadcast bias cannot express "this one key is padding", and does
        # not need to -- `mask` below carries it.
        if bk:
          bias = bias.at[..., K:].set(jnp.asarray(-_NEG, bias.dtype))
      if mask is not None:
        if mask.shape[-1] == K:
          mask = jnp.pad(mask, zc(mask.ndim - 1) + [(0, padk)])  # pad key False
      out = tokamax.dot_product_attention(
          q, k, v, mask=mask, bias=bias, implementation=implementation, scale=scale)
      return out[..., :Q, :, :]
  return tokamax.dot_product_attention(
      q, k, v, mask=mask, bias=bias, implementation=implementation, scale=scale)
