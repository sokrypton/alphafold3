"""The protein language models two of the ported models fold from: ESM2, ESM-C.

Running a tower is inference, not conversion, so it lives here rather than in
`converters/`: loading a published blob must never depend on the conversion
package being installed, for the same reason `params._dequantise_records` does
not import it. `converters/esm_lm.py` maps the weights; this runs them.

  ESM2 3B    chai-1's TOKEN feature stream is mostly ESM2. Zeroing it drops the
             token embedding to corr 0.327 of chai's and folds a natural protein
             to 5.70 A where chai reaches 0.642.
  ESM-C      ESMFold2's alternative to an MSA, not an extra on top of one. It
             consumes ALL n+1 hidden states and mixes them with a learned
             softmax, so the tower cannot be truncated -- the mix peaks on the
             LAST layers (79/80/78 hold 58% of the mass for the 6B).

Both are pre-LayerNorm transformer towers with rotary attention and the same
head_dim of 64, so the scan below is shared. They differ in exactly five places,
which is the whole reason one file can carry both:

                    ESM2 (chai-1)              ESM-C (ESMFold2)
  qkv               three Linears, with bias   one fused Linear, no bias
  q/k norm          none                       LayerNorm over the FULL d_model
  FFN               gelu, with biases          SwiGLU, no biases
  residual          x + f(x)                   x + f(x) / sqrt(n_layers / 36)
  embedding         scaled by 1 - 0.15*0.8     used as-is
  read out as       the LAST hidden state      ALL n+1 hidden states

    python -m alphafold3.model.esm --family esm2 --sequence MKT... --out esm.npz
    python -m alphafold3.model.esm --family esmc --json fold.json --out hid.npz

jax is imported softly on purpose: the vocabularies below are the one thing that
must not drift between our side and the native one, and the native gates run in
a torch environment that has no jax. Importing this module for a token table
therefore has to work without it -- which also means nothing heavier than numpy
may be imported at module level (`alphafold3.model` is a namespace package, so
only this module's own imports matter).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
  import jax
  import jax.numpy as jnp
except ImportError:  # torch-only environments still need the vocabularies
  jax = jnp = None


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
# Forward. Shared scan, one block body per family.
# ---------------------------------------------------------------------------

def _require_jax():
  if jax is None:
    raise ImportError(
        'running a tower needs jax; this module imports without it so that the '
        'vocabularies can be read from a torch-only environment')


def _layer_norm(x, scale, offset=0.0, eps=1e-5):
  dtype = x.dtype
  x = x.astype(jnp.float32)                      # the reduction, always in f32
  m = x.mean(-1, keepdims=True)
  v = x.var(-1, keepdims=True)
  out = (x - m) * jax.lax.rsqrt(v + eps) * scale + offset
  return out.astype(dtype)


def _rope(x, pos, base=10000.0):
  """x [L, H, D]; RotaryEmbedding(head_dim), split-halves convention.

  Both towers use it identically: ESM-C's RotaryEmbedding and ESM2's rot_emb
  both build inv_freq as base**(-i/(D/2)) and rotate halves, which is why the
  checkpoint's own `rot_emb.inv_freq` reproduces to [1, 0.75, 0.5625, ...].
  """
  d = x.shape[-1]
  inv = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
  fr = pos[:, None] * inv[None, :]
  # cast to x's dtype: the angles are computed in float32 (they must be -- the
  # position index is exact there and not in bfloat16), but leaving them float32
  # would promote the residual stream back up and break the scan carry.
  c = jnp.cos(fr)[:, None].astype(x.dtype)
  s = jnp.sin(fr)[:, None].astype(x.dtype)
  x1, x2 = x[..., :d // 2], x[..., d // 2:]
  return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


# float32, and bfloat16 measured and rejected. Native runs ESM-C at bf16, and
# the accuracy would have been acceptable (lm_z against native 0.999969 at bf16
# vs 0.999976 at fp32), but it buys nothing here: at the lengths these towers
# see, the forward is WEIGHT-bound, not compute-bound. The 6B streams 6.36 GB of
# int8 through the scan and dequantises it, against attention over a few hundred
# tokens -- so peak VRAM was identical either way (7.93 GB, all weights) and
# bf16 only added a cast to the dequantise. Set this to jnp.bfloat16 if a much
# longer sequence ever makes the activations matter.
#
# LayerNorm is float32 regardless: it takes a mean and a variance over the full
# d_model, which is the reduction bf16 is worst at, for a vanishing share of the
# FLOPs.
COMPUTE_DTYPE = jnp.float32 if jnp is not None else None


def _deq(w, k, dtype):
  """int8 weights are dequantised inside the scan body, not on load."""
  if k + '__q_scale' in w:
    return (w[k].astype(jnp.float32) * w[k + '__q_scale']).astype(dtype)
  return w[k].astype(dtype)


def _attend(q, k, v, pos, heads, head_dim):
  n_len = q.shape[0]
  q = _rope(q.reshape(n_len, heads, head_dim), pos)
  k = _rope(k.reshape(n_len, heads, head_dim), pos)
  v = v.reshape(n_len, heads, head_dim)
  logits = jnp.einsum('ihd,jhd->hij', q, k) * head_dim ** -0.5
  return jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(n_len, -1)


def _esmc_forward_block(x, w, pos, heads, head_dim, scale, dtype):
  h = _layer_norm(x, w['attn_norm/scale'], w['attn_norm/offset']) @ _deq(w, 'qkv/weights', dtype)
  q, k, v = jnp.split(h, 3, -1)
  q = _layer_norm(q, w['q_norm/scale'])       # over the FULL d_model, not per head
  k = _layer_norm(k, w['k_norm/scale'])
  ctx = _attend(q, k, v, pos, heads, head_dim)
  x = x + (ctx @ _deq(w, 'attn_out/weights', dtype)) / scale
  h = _layer_norm(x, w['ffn_norm/scale'], w['ffn_norm/offset']) @ _deq(w, 'fc1/weights', dtype)
  n = h.shape[-1] // 2
  return x + ((jax.nn.silu(h[..., :n]) * h[..., n:]) @ _deq(w, 'fc2/weights', dtype)) / scale


def _esm2_forward_block(x, w, pos, heads, head_dim, scale, dtype):
  del scale  # ESM2 does not scale its residuals
  h = _layer_norm(x, w['attn_norm/scale'], w['attn_norm/offset'])
  q = h @ _deq(w, 'q/weights', dtype) + w['q/bias'].astype(dtype)
  k = h @ _deq(w, 'k/weights', dtype) + w['k/bias'].astype(dtype)
  v = h @ _deq(w, 'v/weights', dtype) + w['v/bias'].astype(dtype)
  ctx = _attend(q, k, v, pos, heads, head_dim)
  x = x + ctx @ _deq(w, 'attn_out/weights', dtype) + w['attn_out/bias'].astype(dtype)
  h = _layer_norm(x, w['ffn_norm/scale'], w['ffn_norm/offset'])
  h = jax.nn.gelu(h @ _deq(w, 'fc1/weights', dtype) + w['fc1/bias'].astype(dtype), approximate=False)
  return x + h @ _deq(w, 'fc2/weights', dtype) + w['fc2/bias'].astype(dtype)


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
  dtype = COMPUTE_DTYPE
  n_len = ids.shape[0]
  heads, head_dim = dims['n_heads'], dims['d_model'] // dims['n_heads']
  scale = dims.get('residual_scale', 1.0)
  embed = jnp.asarray(p['embed/weights']).astype(dtype)[ids]
  if family == 'esm2':
    embed = embed * _ESM2_TOKEN_DROPOUT
  pos = jnp.arange(n_len, dtype=jnp.float32)
  xs = {k[len('blocks/'):]: jnp.asarray(v)
        for k, v in p.items() if k.startswith('blocks/')}
  block = _BLOCKS[family]

  def body(carry, w):
    out = block(carry, w, pos, heads, head_dim, scale, dtype)
    return out, (out if all_states else None)

  x, states = jax.lax.scan(body, embed, xs)
  final = _layer_norm(x, p['final_norm/scale'], p.get('final_norm/offset', 0.0))
  if not all_states:
    return final.astype(jnp.float32)
  states = jnp.concatenate([embed[None], states], axis=0)
  return states.at[-1].set(final).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Loading and the CLI.
# ---------------------------------------------------------------------------

# Weights kept on the device between calls, keyed by blob path. OFF by default,
# and the default is the point: uploading them costs ~0.6 s of a ~2.6 s call,
# while HOLDING them costs 6.36 GB for the 6B tower -- on a 23 GB card that is
# memory the fold itself needs, and the tower runs once per fold input, so the
# trade is 2% of one fold against a materially higher chance of an OOM in it.
#
# It is worth it when the tower is called repeatedly and no fold is competing
# for the card: a design loop, or scoring many sequences. Then the upload
# happens once instead of per call.
#
#     AF3_ESM_DEVICE_CACHE=1        keep them resident
#     esm.release_device_cache()    hand the memory back
_DEVICE_CACHE = {}


def release_device_cache():
  """Drop any device-resident tower weights. Safe to call when none are held."""
  _DEVICE_CACHE.clear()


def _to_device(params, key):
  if os.environ.get('AF3_ESM_DEVICE_CACHE') != '1':
    return params
  if key not in _DEVICE_CACHE:
    _DEVICE_CACHE[key] = {k: jax.device_put(np.asarray(v))
                          for k, v in params.items()}
  return _DEVICE_CACHE[key]


def _cache_dir(path):
  return path[: -len('.bin.zst')] + '.unpacked' if path.endswith('.bin.zst') else None


def _read_cache(cache):
  """-> the params memory-mapped, or None if the cache is absent/incomplete.

  Memory-mapped rather than read: the scan uploads each block's weights to the
  device as it reaches them, so the pages are wanted exactly once and in order,
  and the OS is better at that than we are. Opening costs nothing measurable.
  """
  stamp = os.path.join(cache, 'MANIFEST')
  if not os.path.exists(stamp):
    return None
  keys = open(stamp).read().split('\n')
  out = {}
  for key in keys:
    if not key:
      continue
    f = os.path.join(cache, key.replace('/', '__') + '.npy')
    if not os.path.exists(f):
      return None
    out[key] = np.load(f, mmap_mode='r')
  return out


def _write_cache(cache, params):
  """Store the DECOMPRESSED weights beside the blob, once.

  zstd buys 14% on top of int8 -- 5.5 GB against 6.4 -- and costs 13 s of
  single-threaded decompression on every load, which was 59% of a 21 s load and
  by far the largest fixed cost in the language-model path. Decompression of one
  frame cannot be parallelised, so the only way not to pay it repeatedly is not
  to repeat it.

  Written to a temporary directory and renamed, so an interrupted write leaves
  no half-cache for the next run to trust.
  """
  import shutil
  tmp = cache + '.partial'
  shutil.rmtree(tmp, ignore_errors=True)
  os.makedirs(tmp, exist_ok=True)
  try:
    for key, arr in params.items():
      np.save(os.path.join(tmp, key.replace('/', '__') + '.npy'), np.asarray(arr))
    with open(os.path.join(tmp, 'MANIFEST'), 'w') as fh:
      fh.write('\n'.join(params))
    shutil.rmtree(cache, ignore_errors=True)
    os.rename(tmp, cache)
  except OSError as err:                      # a full disk is not a fold error
    shutil.rmtree(tmp, ignore_errors=True)
    print('esm: could not write the unpacked cache (%s); '
          'the blob will be decompressed each run' % err)


def load(model_dir=None, family='esmc', tower=None):
  """-> (params, dims). Weights stay INT8; the scan body dequantises per block.

  Deliberately not `params.get_model_haiku_params`, which dequantises on load
  and would put 25.4 GB of float32 in host memory on the way to a card that
  holds 23.
  """
  import collections
  import io
  import re
  import zstandard
  from alphafold3.model.params import read_records, _Q_SCALE_SUFFIX

  # `family` is the ARCHITECTURE and names the blob; `tower` is which trained
  # tower of that architecture, and names the directory and the published file.
  # ESMFold2's variants are trained against step-matched ESM-C snapshots, so
  # esmc/esmc_300m/esmc_600m are three different towers of one family.
  tower = tower or family
  model_dir = model_dir or ('~/ported/%s' % tower)
  path = os.path.join(os.path.expanduser(str(model_dir)), '%s.bin.zst' % tower)
  if not os.path.exists(path):
    directory = os.path.dirname(path)
    others = ([f for f in os.listdir(directory) if f.endswith('.bin.zst')]
              if os.path.isdir(directory) else [])
    if others:
      # Refuse rather than fetch. A directory holding a DIFFERENT tower's blob
      # means the caller meant that one and named the wrong tower; downloading
      # the default over the top of it runs a different model at a different
      # width, which shows up as nothing worse than a shape mismatch several
      # steps later -- or as a silently worse fold if the widths happen to
      # agree.
      raise FileNotFoundError(
          '%s not found, but %s holds %s -- pass tower=%r (or --tower) to use '
          'it' % (os.path.basename(path), directory, ', '.join(sorted(others)),
                  sorted(others)[0][: -len('.bin.zst')]))
    # A tower has no ModelSpec -- it is a separate graph, not an AF3 model -- so
    # ensure_weights does not know about it. Fetch on demand: 5.5 GB is not
    # something to pull for a run that folds from an MSA instead.
    from alphafold3.model import model_registry, weights
    os.makedirs(os.path.dirname(path), exist_ok=True)
    repo = model_registry.get('esmfold2' if family == 'esmc' else 'chai1')
    weights._download(weights._HF_URL.format(repo=repo.weights_repo,
                                             file='%s.bin.zst' % tower), path)
  # The unpacked cache, if a previous run left one. Everything below this point
  # -- decompress, parse, restack, dequantise the small tensors -- produces
  # exactly what it stores.
  cache = _cache_dir(path)
  cached = _read_cache(cache) if (cache and os.environ.get('AF3_ESM_CACHE', '1')
                                  != '0') else None
  if cached is not None:
    return _to_device(cached, path), _dims_from(cached, family)

  per_block = collections.defaultdict(dict)
  p = {}
  with open(path, 'rb') as fh:
    raw = (zstandard.ZstdDecompressor().stream_reader(fh).read()
           if path.endswith('.zst') else fh.read())
  for scope, name, arr in read_records(io.BytesIO(raw)):
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
  for key, arr in list(p.items()):
    if key.endswith(_Q_SCALE_SUFFIX):
      continue
    scale_key = key + _Q_SCALE_SUFFIX
    if key.startswith('blocks/'):
      if arr.dtype != np.int8:
        p[key] = np.asarray(arr, np.float32)
      continue
    if scale_key in p:
      q, sc = np.asarray(arr), np.asarray(p.pop(scale_key))
      p[key] = (q.reshape(-1, q.shape[-1]).astype(np.float32)
                * sc).reshape(q.shape)
    else:
      p[key] = np.asarray(arr, np.float32)
  if cache and os.environ.get('AF3_ESM_CACHE', '1') != '0':
    _write_cache(cache, p)
  return _to_device(p, path), _dims_from(p, family)


def _dims_from(p, family):
  n_layers = (p['blocks/qkv/weights'] if family == 'esmc'
              else p['blocks/q/weights']).shape[0]
  d_model = p['embed/weights'].shape[1]
  return dict(family=family, n_layers=n_layers, d_model=d_model,
              n_heads=d_model // 64,
              residual_scale=(float(np.sqrt(n_layers / 36.0))
                              if family == 'esmc' else 1.0))


def embed(sequences, model_dir=None, family='esmc', tower=None):
  """-> the array the consumer wants, BOS/EOS stripped.

  ESM-C: (L, n_layers + 1, d_model), every state, for ESMFold2's layer mix.
  ESM2:  (sum(len(s)), d_model), the last state only, chains concatenated in
         order -- the rows land on the batch's protein tokens in order, so they
         have to be concatenated the way the featuriser lays those tokens out.
  """
  _require_jax()
  if isinstance(sequences, str):
    sequences = [sequences]
  p, dims = load(model_dir, family, tower)
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
  ap.add_argument('--tower', default=None,
                  help='which trained tower of the family: esmc (6B, the '
                       'default), esmc_300m, esmc_600m, esm2. Names the blob '
                       'and the directory, because ESMFold2 variants are '
                       'trained against step-matched ESM-C snapshots.')
  ap.add_argument('--model-dir', default=None)
  ap.add_argument('--out', required=True)
  a = ap.parse_args(argv)

  seqs = list(a.sequence)
  if a.json:
    seqs += protein_sequences(a.json)
  if not seqs:
    raise SystemExit('pass --sequence or --json')

  rows = embed(seqs, a.model_dir, a.family, a.tower)
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


# ---------------------------------------------------------------------------
# The LanguageModelShim: ESM-C's hidden states -> ESMFold2's pair rep.
# ---------------------------------------------------------------------------
# This belongs to ESMFold2 rather than to the tower, but it runs at fold time,
# which is what decides where it lives. The forward above stops at ESM-C's
# hidden states; the last few layers that turn those into a PAIR representation
# are ESMFold2's own -- a layer mix over all n+1 states, a downprojection, and
# an outer product. None of it has an AlphaFold 3 counterpart to reuse, and
# every release trains its own, which is why the weights are written per model
# beside the blob rather than shared.
#
# It is numpy, not jax: it runs once per fold on a (L, n+1, width) array and is
# nowhere near the hot path, so a second graph would buy nothing.


def load_shim_params(model_dir, model_name=None):
  """-> the shim weights written beside the blob by convert_esmfold2_weights.

  Named after the MODEL: each ESMFold2 release trains its own shim, and using
  another's is not a small error -- corr 0.026 against native, and a fold that
  goes from 0.96 A to 8.8.
  """
  import pathlib
  root = pathlib.Path(os.path.expanduser(str(model_dir)))
  name = model_name or root.name
  for cand in (root / ('%s.lm.npz' % name), root / 'esmfold2.lm.npz'):
    if cand.exists():
      return {k.replace('.', '/'): v for k, v in np.load(cand).items()}
  raise FileNotFoundError(
      f'no shim in {root} -- rerun convert_esmfold2_weights, or run without ESM-C')


def shim(hidden, params):
  """(num_tokens, num_layers, lm_width) -> (num_tokens, num_tokens, c_z).

  `combine` is already softmaxed at conversion time: the layer mix is a constant,
  so it folds to a plain (num_layers,) array. It peaks on the LAST ESM-C layers
  -- 79/80/78 hold 58% of the mass -- so the tower cannot be truncated from the
  top, though ~31 of the 81 states carry 99% of it.

  Runs on the accelerator, like the tower above it. Every step is a matmul or an
  elementwise op over (L, L, c_z), so on the host this cost more than the 6B
  tower's forward pass and grew as L^2: at 340 tokens it was 2.4 s, of which the
  erf in the gelu and two LayerNorms were 86%. Nothing about it wanted to be on
  a CPU -- it was numpy only because it started life beside the converters.

  The layer mix is applied BEFORE the projection rather than after. `lm_projection`
  is linear and shared across layers, so sum_k c_k (LN_k @ W) == (sum_k c_k LN_k) @ W
  exactly (7e-7 relative, pure float associativity), and the second form does one
  matmul where the first does num_layers of them.
  """
  _require_jax()
  # float32 matmuls, explicitly. The default on a modern NVIDIA card is tf32,
  # which took this to 7e-4 relative -- and the trunk amplifies an injection
  # error by ~50x, so that lands at the same magnitude as the relative-chain bug
  # this port already had. The precision costs nothing measurable here.
  with jax.default_matmul_precision('float32'):
    return _shim(hidden, params)


def _shim(hidden, params):
  p = {k: jnp.asarray(v) for k, v in params.items()}
  hidden = jnp.asarray(hidden, jnp.float32)
  if hidden.ndim == 4 and hidden.shape[0] == 1:
    hidden = hidden[0]
  x = _layer_norm(hidden, p['lm_norm/scale'], p['lm_norm/offset'])
  x = jnp.einsum('k,lkc->lc', p['combine'], x) @ p['lm_projection/weights']
  x = x @ p['downproject/weights'] + p['downproject/bias']
  # SingleToPair: the outer product carries BOTH a product and a difference, so
  # the pair rep sees magnitude and direction, not just agreement.
  z = jnp.concatenate([x[:, None] * x[None, :], x[:, None] - x[None, :]], -1)
  z = z @ p['pair_mlp_1/weights'] + p['pair_mlp_1/bias']
  z = jax.nn.gelu(z, approximate=False) @ p['pair_mlp_2/weights'] + p['pair_mlp_2/bias']
  return np.asarray(
      _layer_norm(z, p['pair_norm/scale'], p['pair_norm/offset']))
