"""ESM2 / ESM-C -> AF3-haiku weight conversion.

The towers two of the ported models fold from: chai-1 from ESM2 3B, ESMFold2
from ESM-C. This maps their weights; `alphafold3.model.esm` RUNS them, and the
split is the same one every other converter observes -- conversion is an offline
step, and loading a published blob must not depend on this package.

ESM2 arrives as a traced TorchScript archive rather than a state dict, which is
why chai-1's language model stayed in torch for so long. It is not opaque:
`state_dict()` gives the 621 tensors under their ordinary ESM2 names.

The architecture, and the five places the two towers differ, are documented in
`alphafold3.model.esm` alongside the forward that implements them. The
vocabularies live there too -- they are needed to RUN a tower, and the native
gates read them from a torch environment.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

from alphafold3.model.esm import VOCABS  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Weight maps.
# ---------------------------------------------------------------------------

def derive_dims(sd, family):
  """Read the tower's shape off the checkpoint rather than hard-coding it."""
  if family == 'esmc':
    pre = 'esmc.transformer.blocks.'
    n = len({k[len(pre):].split('.')[0] for k in sd if k.startswith(pre)})
    d_model = sd['esmc.embed.weight'].shape[1]
    return dict(
        family='esmc', n_layers=n, d_model=d_model,
        vocab=sd['esmc.embed.weight'].shape[0],
        ffn_hidden=sd[pre + '0.ffn.fc1_weight'].shape[0] // 2,
        n_heads=d_model // 64,                    # RotaryEmbedding(head_dim) = 64
        residual_scale=float(np.sqrt(n / 36.0)),  # the ESM3 scheme
    )
  n = len({k[len('layers.'):].split('.')[0] for k in sd if k.startswith('layers.')})
  d_model = sd['embed_tokens.weight'].shape[1]
  return dict(
      family='esm2', n_layers=n, d_model=d_model,
      vocab=sd['embed_tokens.weight'].shape[0],
      ffn_hidden=sd['layers.0.fc1.weight'].shape[0],
      n_heads=d_model // 64,
      residual_scale=1.0,                         # ESM2 does not scale residuals
  )


def _esmc_block(sd, prefix):
  from .common import _arr, t
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


def _esm2_block(sd, prefix):
  from .common import _arr, t
  g = lambda leaf: sd['%s.%s' % (prefix, leaf)]
  return {
      'attn_norm/scale': _arr(g('self_attn_layer_norm.weight')),
      'attn_norm/offset': _arr(g('self_attn_layer_norm.bias')),
      'q/weights': t(g('self_attn.q_proj.weight')),
      'q/bias': _arr(g('self_attn.q_proj.bias')),
      'k/weights': t(g('self_attn.k_proj.weight')),
      'k/bias': _arr(g('self_attn.k_proj.bias')),
      'v/weights': t(g('self_attn.v_proj.weight')),
      'v/bias': _arr(g('self_attn.v_proj.bias')),
      'attn_out/weights': t(g('self_attn.out_proj.weight')),
      'attn_out/bias': _arr(g('self_attn.out_proj.bias')),
      'ffn_norm/scale': _arr(g('final_layer_norm.weight')),
      'ffn_norm/offset': _arr(g('final_layer_norm.bias')),
      'fc1/weights': t(g('fc1.weight')),
      'fc1/bias': _arr(g('fc1.bias')),
      'fc2/weights': t(g('fc2.weight')),
      'fc2/bias': _arr(g('fc2.bias')),
  }


def map_tower(sd, family, dims=None):
  from .common import _arr, nest, stack_blocks
  dims = dims or derive_dims(sd, family)
  if family == 'esmc':
    p = {'embed/weights': _arr(sd['esmc.embed.weight']),
         'final_norm/scale': _arr(sd['esmc.transformer.norm.weight'])}
    blocks = lambda i: _esmc_block(sd, 'esmc.transformer.blocks.%d' % i)
  else:
    p = {'embed/weights': _arr(sd['embed_tokens.weight']),
         'final_norm/scale': _arr(sd['emb_layer_norm_after.weight']),
         'final_norm/offset': _arr(sd['emb_layer_norm_after.bias'])}
    blocks = lambda i: _esm2_block(sd, 'layers.%d' % i)
  p.update(nest('blocks', stack_blocks(blocks, dims['n_layers'])))
  return p


# ---------------------------------------------------------------------------
# Conversion.
# ---------------------------------------------------------------------------

def load_esm2_checkpoint(path):
  """chai ships ESM2 as a TRACED TorchScript archive, not a state dict.

  That is why chai-1 ran its language model in torch for so long. The archive is
  not opaque, though: `state_dict()` gives the 621 tensors under their ordinary
  ESM2 names, so it converts like any other checkpoint.
  """
  import torch
  module = torch.jit.load(os.path.expanduser(str(path)), map_location='cpu')
  return {k: v.detach().cpu().float().numpy()
          for k, v in module.state_dict().items()}


# The towers are int8, full stop -- not a default that can be overridden the way
# --weights_precision overrides a model's. For ESM-C it is the only thing that
# works at all: float32 CANNOT be written, because the record header packs the
# payload length as a signed 32-bit int and the fused fc1 stack is a single
# 5.27 GiB tensor, so an fp32 write dies partway through with a struct.error.
# For both, int8 is what makes the tower GPU-resident inside the scan (6.35 GB
# instead of 25.4 for the 6B) rather than merely smaller to download, and the
# cost is measured, not assumed: ESM2 reads corr 0.9999498 against the traced
# torch model, and ESM-C's hidden states are >= 0.9998 per layer.
#
# A precision knob here would therefore only offer choices that are broken
# (fp32 for ESM-C) or pointless (fp16, which neither halves the residency nor
# improves on 1e-3), so there is none.
QUANT_SCHEME = 'int8'


def convert_tower(checkpoint, output_dir, family, tower=None):
  """Convert a language-model tower to an int8 AF3-haiku blob.

  Quantisation happens HERE rather than through converters.quantise's
  requantise_blob, which reads an existing blob: there is no fp32 blob to read,
  and for ESM-C there could not be one.
  """
  from . import common, quantise
  if family == 'esmc':
    from .esmfold2 import load_esmfold2_checkpoint
    sd = load_esmfold2_checkpoint(checkpoint)
  else:
    sd = load_esm2_checkpoint(checkpoint)
  dims = derive_dims(sd, family)
  params = {}
  common.populate(params, family, map_tower(sd, family, dims))
  del sd
  records = split_stacked_blocks(common.tree_to_records(params), dims['n_layers'])
  del params
  records = quantise.quantise_records(records, QUANT_SCHEME)
  # Named after the TOWER, not the family: esmc/esmc_300m/esmc_600m are three
  # trained towers of one architecture, and the blob has to carry which one it
  # is -- both so a directory is self-describing and so the local name matches
  # the published one.
  tower = tower or family
  return pathlib.Path(common.write_records_blob(
      pathlib.Path(output_dir) / ('%s.bin.zst' % tower), records,
      identifier=family))


def convert_esmc_weights(checkpoint, output_dir, tower=None):
  return convert_tower(checkpoint, output_dir, 'esmc', tower)


def convert_esm2_weights(checkpoint, output_dir):
  return convert_tower(checkpoint, output_dir, 'esm2')


def split_stacked_blocks(records, n_layers):
  """Write the transformer blocks one record each, not one stacked array.

  ESM-C's stacked fc1 weight is 17.8 GB in float32 and still 4.4 GB in int8,
  over the 2 GiB a record header can describe (`'<5i'`, a signed 32-bit length).
  These towers are separate graphs with their own loader, so the blob layout is
  ours to choose: splitting on the layer axis puts the largest record at ~56 MB
  and costs nothing, because `load` restacks them. It also makes the int8 scales
  PER BLOCK rather than shared across all layers, which is strictly more
  accurate.
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


