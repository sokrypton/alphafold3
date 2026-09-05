"""ESM-C (biohub/ESMC-6B) -> AF3 haiku parameter conversion.

The protein language model ESMFold2 is conditioned on.  6.352B params / 25.4 GB
fp32 across 80 blocks of d_model 2560 / 40 heads; 67% of the weight is FFN.
ESMFold2 consumes ALL 81 hidden states (embedding + 80 blocks) and mixes them
with a learned softmax, so the tower cannot be truncated -- the mix peaks on the
LAST layers (79/80/78 hold 58% of the mass).

Block layout (pre-norm, ESM3 residual scaling):
    h  = LN(x; layernorm_qkv.layer_norm_*) @ layernorm_qkv.weight   -> [q|k|v]
    q,k = LN(q; q_ln, NO bias), LN(k; k_ln)      # over the FULL d_model, not per head
    q,k = RoPE(q, k)                             # head_dim 64, base 10000
    x  = x + out_proj(attn(q,k,v)) / sqrt(n_layers / 36)
    x  = x + fc2(swiglu(fc1(LN(x; ffn.layer_norm_*)))) / sqrt(n_layers / 36)
and a final LayerNorm with NO bias.

`layernorm_qkv` and the ffn keep the TransformerEngine parameter NAMES
(`layer_norm_weight`, `weight`, `fc1_weight`) even on the pure-PyTorch path, so
the state dict layout is the same either way.
"""

import numpy as np

from . import common
from .common import _arr, t, nest, stack_blocks

BOS, PAD, EOS, MASK = 0, 1, 2, 32


def derive_dims(sd):
  n = len({k[len('esmc.transformer.blocks.'):].split('.')[0]
           for k in sd if k.startswith('esmc.transformer.blocks.')})
  d_model = sd['esmc.embed.weight'].shape[1]
  return dict(
      n_layers=n,
      d_model=d_model,
      vocab=sd['esmc.embed.weight'].shape[0],
      ffn_hidden=sd['esmc.transformer.blocks.0.ffn.fc1_weight'].shape[0] // 2,
      n_heads=d_model // 64,                     # RotaryEmbedding(d_model // n_heads) = 64
      residual_scale=float(np.sqrt(n / 36.0)),   # the ESM3 scheme
  )


def block(sd, prefix):
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  return {
      'attn_norm/scale': _arr(g('attn.layernorm_qkv.layer_norm_weight')),
      'attn_norm/offset': _arr(g('attn.layernorm_qkv.layer_norm_bias')),
      'qkv/weights': t(g('attn.layernorm_qkv.weight')),
      'q_norm/scale': _arr(g('attn.q_ln.weight')),
      'k_norm/scale': _arr(g('attn.k_ln.weight')),
      'attn_out/weights': t(g('attn.out_proj.weight')),
      'ffn_norm/scale': _arr(g('ffn.layer_norm_weight')),
      'ffn_norm/offset': _arr(g('ffn.layer_norm_bias')),
      'fc1/weights': t(g('ffn.fc1_weight')),
      'fc2/weights': t(g('ffn.fc2_weight')),
  }


def map_esmc_to_af3(sd, dims=None):
  dims = dims or derive_dims(sd)
  p = {'embed/weights': _arr(sd['esmc.embed.weight']),
       'final_norm/scale': _arr(sd['esmc.transformer.norm.weight'])}
  p.update(nest('blocks', stack_blocks(
      lambda i: block(sd, 'esmc.transformer.blocks.%d' % i), dims['n_layers'])))
  return p


def lm_input_ids(token_ids):
  """Wrap one chain's residue token ids as ESMFold2 does: [BOS, ids..., EOS].

  Multi-chain inserts [EOS, BOS] between chains; padding is PAD and
  `sequence_id = cumsum(ids == BOS) - 1`, set to -1 on PAD, restricts attention
  to within a chain.
  """
  return np.concatenate([[BOS], np.asarray(token_ids, np.int64), [EOS]])


def convert_esmc_weights(checkpoint, output_dir, scheme='int8'):
  """Convert biohub/ESMC-6B (sharded safetensors dir) to an AF3-haiku blob.

  int8 by default, and not only to save bytes: float32 CANNOT be written at all.
  The record header packs the payload length as a signed 32-bit int and the
  tower's fused fc1 stack is a single 5.27 GiB tensor, so an fp32 write dies
  partway through with a struct.error. int8 puts it at 1.32 GiB, and the whole
  tower at ~6.4 GB -- which also fits a 23 GB GPU, where the fp32 tower never
  did (the ESM-C gate had been running on CPU for exactly that reason).

  Quantisation happens HERE rather than through converters.quantise's
  requantise_blob, which reads an existing blob: there is no fp32 blob to read.
  """
  from pathlib import Path
  from . import quantise
  from .esmfold2 import load_esmfold2_checkpoint
  sd = load_esmfold2_checkpoint(checkpoint)
  dims = derive_dims(sd)
  params = {}
  common.populate(params, 'esmc', map_esmc_to_af3(sd, dims))
  del sd
  records = split_stacked_blocks(common.tree_to_records(params), dims['n_layers'])
  del params
  records = quantise.quantise_records(records, scheme)
  return Path(common.write_records_blob(
      Path(output_dir) / 'esmc.bin.zst', records, identifier='esmc'))


def split_stacked_blocks(records, n_layers):
  """Write the 80 transformer blocks one record each, not one stacked array.

  The stacked fc1 weight is 17.8 GB in float32 and still 4.4 GB in int8, over
  the 2 GiB a record header can describe (`'<5i'`, a signed 32-bit length). ESM-C
  is a separate graph with its own loader, so its blob layout is ours to choose:
  splitting on the layer axis puts the largest record at ~56 MB and costs
  nothing, because esmc_embed.load restacks them. It also makes the int8 scales
  PER BLOCK rather than shared across all 80, which is strictly more accurate.
  """
  out = []
  for scope, name, arr in records:
    arr = np.asarray(arr)
    if '/blocks/' in scope + '/' and arr.ndim >= 1 and arr.shape[0] == n_layers:
      for i in range(n_layers):
        out.append(('%s/%d' % (scope, i), name, arr[i]))
    else:
      out.append((scope, name, arr))
  return out
