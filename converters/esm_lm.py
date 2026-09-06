"""The protein language models the ports fold from: ESM2 and ESM-C, one file.

Two of the ported models are conditioned on a protein language model rather than
(chai-1) or instead of (ESMFold2) an MSA, and neither model can be run at full
fidelity without one:

  ESM2 3B    chai-1's TOKEN feature stream is mostly ESM2. Zeroing it drops the
             token embedding to corr 0.327 of chai's and folds a natural protein
             to 5.70 A where chai reaches 0.642.
  ESM-C      ESMFold2's alternative to an MSA, not an extra on top of one. It
             consumes ALL n+1 hidden states and mixes them with a learned
             softmax, so the tower cannot be truncated -- the mix peaks on the
             LAST layers (79/80/78 hold 58% of the mass for the 6B).

Both are pre-LayerNorm transformer towers with rotary attention and the same
head_dim of 64, so the scan machinery below is shared. They differ in exactly
five places, which is the whole reason one file can carry both:

                    ESM2 (chai-1)              ESM-C (ESMFold2)
  qkv               three Linears, with bias   one fused Linear, no bias
  q/k norm          none                       LayerNorm over the FULL d_model
  FFN               gelu, with biases          SwiGLU, no biases
  residual          x + f(x)                   x + f(x) / sqrt(n_layers / 36)
  embedding         scaled by 1 - 0.15*0.8     used as-is
  read out as       the LAST hidden state      ALL n+1 hidden states

Both towers convert to int8 and run under a `lax.scan`, and for ESM-C both are
load-bearing rather than a size optimisation -- see `convert_tower` and
`forward`.

    python -m converters.esm_lm --family esm2 --sequence MKT... --out esm.npz
    python -m converters.esm_lm --family esmc --pdb ~/6MRR.pdb --out hidden.npz

jax is imported softly on purpose: the vocabularies below are the one thing that
must not drift between our side and the native one, and the native gates run in
a torch environment that has no jax. Importing this module for a token table
therefore has to work without it.
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

try:
  import jax
  import jax.numpy as jnp
except ImportError:  # torch-only environments still need the vocabularies
  jax = jnp = None

# `converters.common` is imported inside the conversion functions, not here, for
# the same reason jax is soft above: it pulls in alphafold3 (for the blob record
# format), which the torch-only gate environment does not have. Reading a token
# table must not require either.


# ---------------------------------------------------------------------------
# Vocabularies. NumPy only.
# ---------------------------------------------------------------------------

# The two alphabets are the same 33 tokens in the same order and differ in
# exactly one slot: index 31 is ESM2's <null_1> and ESM-C's chain separator '|'.
# Neither appears in a single-chain input, but they are kept apart rather than
# shared because a silent one-token drift is precisely the failure that would
# survive every shape check.
_COMMON = '<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . -'
ESM2_VOCAB = (_COMMON + ' <null_1> <mask>').split()
ESMC_VOCAB = (_COMMON + ' | <mask>').split()

BOS, PAD, EOS, MASK = 0, 1, 2, 32

VOCABS = {'esm2': ESM2_VOCAB, 'esmc': ESMC_VOCAB}


def sequence_ids(seq, family='esmc'):
  """One-letter sequence -> residue token ids (no BOS/EOS)."""
  table = {tok: i for i, tok in enumerate(VOCABS[family])}
  unk = table['<unk>']
  return np.array([table.get(c.upper(), unk) for c in seq], np.int64)


def lm_input_ids(token_ids):
  """Wrap one chain as both towers do: [BOS, ids..., EOS].

  Multi-chain inserts [EOS, BOS] between chains; padding is PAD and
  `sequence_id = cumsum(ids == BOS) - 1`, set to -1 on PAD, restricts attention
  to within a chain.
  """
  return np.concatenate([[BOS], np.asarray(token_ids, np.int64), [EOS]])


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


def convert_tower(checkpoint, output_dir, family, scheme='int8'):
  """Convert a language-model tower to an AF3-haiku blob. int8 by default.

  For ESM-C int8 is not an optimisation, it is the only thing that works:
  float32 CANNOT be written at all, because the record header packs the payload
  length as a signed 32-bit int and the fused fc1 stack is a single 5.27 GiB
  tensor, so an fp32 write dies partway through with a struct.error. int8 puts
  the 6B tower at ~6.4 GB, which also fits a 23 GB card where fp32 (25.4 GB)
  never did.

  Quantisation happens HERE rather than through converters.quantise's
  requantise_blob, which reads an existing blob: there is no fp32 blob to read.
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
  records = quantise.quantise_records(records, scheme)
  return pathlib.Path(common.write_records_blob(
      pathlib.Path(output_dir) / ('%s.bin.zst' % family), records,
      identifier=family))


def convert_esmc_weights(checkpoint, output_dir, scheme='int8'):
  return convert_tower(checkpoint, output_dir, 'esmc', scheme)


def convert_esm2_weights(checkpoint, output_dir, scheme='int8'):
  return convert_tower(checkpoint, output_dir, 'esm2', scheme)


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


# ---------------------------------------------------------------------------
# Forward. Shared scan, one block body per family.
# ---------------------------------------------------------------------------

def _require_jax():
  if jax is None:
    raise ImportError(
        'running a tower needs jax; this module imports without it so that the '
        'vocabularies can be read from a torch-only environment')


def _layer_norm(x, scale, offset=0.0, eps=1e-5):
  m = x.mean(-1, keepdims=True)
  v = x.var(-1, keepdims=True)
  return (x - m) * jax.lax.rsqrt(v + eps) * scale + offset


def _rope(x, pos, base=10000.0):
  """x [L, H, D]; RotaryEmbedding(head_dim), split-halves convention.

  Both towers use it identically: ESM-C's RotaryEmbedding and ESM2's rot_emb
  both build inv_freq as base**(-i/(D/2)) and rotate halves, which is why the
  checkpoint's own `rot_emb.inv_freq` reproduces to [1, 0.75, 0.5625, ...].
  """
  d = x.shape[-1]
  inv = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
  fr = pos[:, None] * inv[None, :]
  c, s = jnp.cos(fr)[:, None], jnp.sin(fr)[:, None]
  x1, x2 = x[..., :d // 2], x[..., d // 2:]
  return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def _deq(w, k):
  """int8 weights are dequantised inside the scan body, not on load."""
  if k + '__q_scale' in w:
    return w[k].astype(jnp.float32) * w[k + '__q_scale']
  return w[k].astype(jnp.float32)


def _attend(q, k, v, pos, heads, head_dim):
  n_len = q.shape[0]
  q = _rope(q.reshape(n_len, heads, head_dim), pos)
  k = _rope(k.reshape(n_len, heads, head_dim), pos)
  v = v.reshape(n_len, heads, head_dim)
  logits = jnp.einsum('ihd,jhd->hij', q, k) * head_dim ** -0.5
  return jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(n_len, -1)


def _esmc_forward_block(x, w, pos, heads, head_dim, scale):
  h = _layer_norm(x, w['attn_norm/scale'], w['attn_norm/offset']) @ _deq(w, 'qkv/weights')
  q, k, v = jnp.split(h, 3, -1)
  q = _layer_norm(q, w['q_norm/scale'])       # over the FULL d_model, not per head
  k = _layer_norm(k, w['k_norm/scale'])
  ctx = _attend(q, k, v, pos, heads, head_dim)
  x = x + (ctx @ _deq(w, 'attn_out/weights')) / scale
  h = _layer_norm(x, w['ffn_norm/scale'], w['ffn_norm/offset']) @ _deq(w, 'fc1/weights')
  n = h.shape[-1] // 2
  return x + ((jax.nn.silu(h[..., :n]) * h[..., n:]) @ _deq(w, 'fc2/weights')) / scale


def _esm2_forward_block(x, w, pos, heads, head_dim, scale):
  del scale  # ESM2 does not scale its residuals
  h = _layer_norm(x, w['attn_norm/scale'], w['attn_norm/offset'])
  q = h @ _deq(w, 'q/weights') + w['q/bias']
  k = h @ _deq(w, 'k/weights') + w['k/bias']
  v = h @ _deq(w, 'v/weights') + w['v/bias']
  ctx = _attend(q, k, v, pos, heads, head_dim)
  x = x + ctx @ _deq(w, 'attn_out/weights') + w['attn_out/bias']
  h = _layer_norm(x, w['ffn_norm/scale'], w['ffn_norm/offset'])
  h = jax.nn.gelu(h @ _deq(w, 'fc1/weights') + w['fc1/bias'], approximate=False)
  return x + h @ _deq(w, 'fc2/weights') + w['fc2/bias']


_BLOCKS = {'esmc': _esmc_forward_block, 'esm2': _esm2_forward_block}

# ESM2 is trained with token_dropout, which rescales the embedding by
# (1 - mask_ratio_train) / (1 - mask_ratio_observed). At inference there are no
# <mask> tokens, so the observed ratio is 0 and this collapses to a constant
# 1 - 0.15*0.8. It is a plain scalar on the tower's INPUT and easy to miss;
# dropping it shifts every hidden state.
_ESM2_TOKEN_DROPOUT = 1.0 - 0.15 * 0.8


def forward(ids, p, dims, all_states=None):
  """-> hidden states, BOS/EOS still attached.

  A lax.scan over the blocks, NOT a python loop. The loop traced all 80 of
  ESM-C's blocks into one graph and moved each block's weights to the device as
  it went, so the whole 25 GB tower ended up resident and a 23 GB card ran out
  on the last few layers -- native torch never hits this because it executes
  eagerly and frees as it goes. Scanning compiles ONE block body and keeps the
  weights as the scanned axis, which is also why they stay int8 on the device
  and are dequantised inside the body: 6.35 GB resident instead of 25.4 GB.

  `all_states` defaults to what the consumer needs. ESMFold2 mixes ALL n+1
  states with a learned softmax, so none can be skipped, and the LAST is post
  the stack's final LayerNorm while the other n are the raw residual stream --
  returning the pre-norm value there reads corr 0.909 against native on layer 80
  where every other layer is >= 0.9998. chai-1 reads only the final state.
  """
  _require_jax()
  family = dims.get('family', 'esmc')
  if all_states is None:
    all_states = family == 'esmc'
  ids = jnp.asarray(ids)
  n_len = ids.shape[0]
  heads, head_dim = dims['n_heads'], dims['d_model'] // dims['n_heads']
  scale = dims.get('residual_scale', 1.0)
  embed = jnp.asarray(p['embed/weights']).astype(jnp.float32)[ids]
  if family == 'esm2':
    embed = embed * _ESM2_TOKEN_DROPOUT
  pos = jnp.arange(n_len, dtype=jnp.float32)
  xs = {k[len('blocks/'):]: jnp.asarray(v)
        for k, v in p.items() if k.startswith('blocks/')}
  block = _BLOCKS[family]

  def body(carry, w):
    out = block(carry, w, pos, heads, head_dim, scale)
    return out, (out if all_states else None)

  x, states = jax.lax.scan(body, embed, xs)
  final = _layer_norm(x, p['final_norm/scale'], p.get('final_norm/offset', 0.0))
  if not all_states:
    return final
  states = jnp.concatenate([embed[None], states], axis=0)
  return states.at[-1].set(final)


# ---------------------------------------------------------------------------
# Loading and the CLI.
# ---------------------------------------------------------------------------

def load(model_dir=None, family='esmc'):
  """-> (params, dims). Weights stay INT8; the scan body dequantises per block.

  Deliberately not `params.get_model_haiku_params`, which dequantises on load
  and would put 25.4 GB of float32 in host memory on the way to a card that
  holds 23.
  """
  import collections
  import re
  from .common import read_blob

  model_dir = model_dir or ('~/ported/%s' % family)
  path = os.path.join(os.path.expanduser(str(model_dir)), '%s.bin.zst' % family)
  if not os.path.exists(path):
    # A tower has no ModelSpec -- it is a separate graph, not an AF3 model -- so
    # ensure_weights does not know about it. Fetch on demand: 5.5 GB is not
    # something to pull for a run that folds from an MSA instead.
    from alphafold3.model import model_registry, weights
    os.makedirs(os.path.dirname(path), exist_ok=True)
    repo = model_registry.get('esmfold2' if family == 'esmc' else 'chai1')
    weights._download(weights._HF_URL.format(repo=repo.weights_repo,
                                             file='%s.bin.zst' % family), path)
  per_block = collections.defaultdict(dict)
  p = {}
  for scope, name, arr in read_blob(path):
    if scope.startswith('__meta__'):
      continue
    sub = scope[len(family) + 1:] if scope.startswith(family + '/') else scope
    m = re.match(r'(blocks/.+)/(\d+)$', sub)
    if m:
      per_block['%s/%s' % (m.group(1), name)][int(m.group(2))] = arr
    else:
      p['%s/%s' % (sub, name) if sub else name] = arr
  for key, layers in per_block.items():
    p[key] = np.stack([layers[i] for i in sorted(layers)])
  # Only the BLOCK weights stay int8, because only they are scanned and only
  # they are big enough to matter. Everything else -- notably embed/weights, the
  # input to the whole stack -- is dequantised here. Leaving the embedding as
  # raw int8 codes silently scaled the tower's input by its per-channel scales
  # and read corr 0.19 against native, with layer 0 already at 0.89.
  from .quantise import SCALE_SUFFIX, dequantise_int8
  for key, arr in list(p.items()):
    if key.endswith(SCALE_SUFFIX):
      continue
    scale_key = key + SCALE_SUFFIX
    if key.startswith('blocks/'):
      if arr.dtype != np.int8:
        p[key] = np.asarray(arr, np.float32)
      continue
    if scale_key in p:
      p[key] = dequantise_int8(np.asarray(arr), np.asarray(p.pop(scale_key)))
    else:
      p[key] = np.asarray(arr, np.float32)
  n_layers = (p['blocks/qkv/weights'] if family == 'esmc'
              else p['blocks/q/weights']).shape[0]
  d_model = p['embed/weights'].shape[1]
  dims = dict(family=family, n_layers=n_layers, d_model=d_model,
              n_heads=d_model // 64,
              residual_scale=(float(np.sqrt(n_layers / 36.0))
                              if family == 'esmc' else 1.0))
  return p, dims


def embed(sequences, model_dir=None, family='esmc'):
  """-> the array the consumer wants, BOS/EOS stripped.

  ESM-C: (L, n_layers + 1, d_model), every state, for ESMFold2's layer mix.
  ESM2:  (sum(len(s)), d_model), the last state only, chains concatenated in
         order -- the rows land on the batch's protein tokens in order, so they
         have to be concatenated the way the featuriser lays those tokens out.
  """
  _require_jax()
  if isinstance(sequences, str):
    sequences = [sequences]
  p, dims = load(model_dir, family)
  rows = []
  for seq in sequences:
    ids = lm_input_ids(sequence_ids(seq, family))
    out = np.asarray(forward(ids, p, dims))
    if family == 'esmc':
      rows.append(out[:, 1:-1, :].transpose(1, 0, 2))
    else:
      out = out[1:-1]
      if out.shape[0] != len(seq):
        raise ValueError(f'{out.shape[0]} rows for a {len(seq)}-residue chain')
      rows.append(out)
  if family == 'esmc':
    if len(rows) > 1:
      raise ValueError('ESM-C multi-chain wrapping is [EOS, BOS] separated; '
                       'pass one chain')
    return rows[0]
  return np.concatenate(rows, axis=0).astype(np.float32)


def protein_sequences(json_path):
  """The protein chain sequences of an AlphaFold 3 input JSON, in chain order."""
  from alphafold3.common import folding_input

  fold_input = folding_input.Input.from_json(
      pathlib.Path(json_path).read_text())
  return [chain.sequence for chain in fold_input.chains
          if isinstance(chain, folding_input.ProteinChain)]


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  ap.add_argument('--family', choices=sorted(VOCABS), default='esmc')
  ap.add_argument('--sequence', action='append', default=[],
                  help='a protein sequence, repeatable, in chain order')
  ap.add_argument('--json', help='the fold input JSON to read sequences from')
  ap.add_argument('--pdb', help='take the sequence from this structure instead')
  ap.add_argument('--model-dir', default=None)
  ap.add_argument('--out', required=True)
  a = ap.parse_args(argv)

  seqs = list(a.sequence)
  if a.json:
    seqs += protein_sequences(a.json)
  if a.pdb:
    from .pdb import parse_ca
    seqs.append(parse_ca(os.path.expanduser(a.pdb))[0])
  if not seqs:
    raise SystemExit('pass one of --sequence, --json or --pdb')

  rows = embed(seqs, a.model_dir, a.family)
  out = os.path.expanduser(a.out)
  if a.family == 'esmc':
    np.savez_compressed(out, lm_hidden=rows.astype(np.float32))
  else:
    np.savez_compressed(out, esm=rows,
                        sequences=np.array(seqs, dtype=object),
                        allow_pickle=True)
  print('wrote %s  %s' % (out, (rows.shape,)))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
