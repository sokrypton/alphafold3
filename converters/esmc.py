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


def convert_esmc_weights(checkpoint, output_dir):
  """Convert biohub/ESMC-6B (sharded safetensors dir) to an AF3-haiku blob."""
  from pathlib import Path
  from .esmfold2 import load_esmfold2_checkpoint
  sd = load_esmfold2_checkpoint(checkpoint)
  params = {}
  common.populate(params, 'esmc', map_esmc_to_af3(sd))
  return Path(common.write_params_blob(output_dir, 'esmc.bin.zst', params, add_meta=True))
