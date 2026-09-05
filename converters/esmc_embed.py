"""Run the converted ESM-C 6B tower and emit ESMFold2's 81 hidden states.

ESM-C is ESMFold2's other input -- the alternative to an MSA, not an optional
extra on top of one. It is a separate graph from AF3's (converters/esmc.py maps
the weights; this runs them), so it lives here rather than in the model package,
and its output feeds converters/esmfold2_lm.py's shim to become a pair
representation.

    python -m converters.esmc_embed --seq MKT... --out hidden.npz
    python -m converters.esmc_embed --pdb ~/6MRR.pdb --out hidden.npz

int8 storage is what makes this runnable at all on a 23 GB card: the float32
tower is 25.4 GB and could not even be WRITTEN (see convert_esmc_weights).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

from . import esmc as C

# ESM-C's alphabet, in id order. Index 0/1/2/32 are BOS/PAD/EOS/MASK, which is
# what converters.esmc asserts; the rest is the standard ESM residue ordering.
VOCAB = ('<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B '
         'U Z O . - | <mask>').split()
_TOKENS = {t: i for i, t in enumerate(VOCAB)}
UNK = _TOKENS['<unk>']


def sequence_ids(seq):
  """One-letter sequence -> ESM-C residue token ids (no BOS/EOS)."""
  return np.array([_TOKENS.get(c.upper(), UNK) for c in seq], np.int64)


def _layer_norm(x, scale, offset=0.0, eps=1e-5):
  m = x.mean(-1, keepdims=True)
  v = x.var(-1, keepdims=True)
  return (x - m) * jax.lax.rsqrt(v + eps) * scale + offset


def _rope(x, pos, base=10000.0):
  """x [L, H, D]; RotaryEmbedding(head_dim), split-halves convention."""
  d = x.shape[-1]
  inv = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
  fr = pos[:, None] * inv[None, :]
  c, s = jnp.cos(fr)[:, None], jnp.sin(fr)[:, None]
  x1, x2 = x[..., :d // 2], x[..., d // 2:]
  return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def _block(x, w, pos, heads, head_dim, scale):
  """One ESM-C transformer block, with int8 weights dequantised in place."""
  deq = lambda k: (w[k].astype(jnp.float32) * w[k + '__q_scale']
                   if k + '__q_scale' in w else w[k].astype(jnp.float32))
  n_len = x.shape[0]
  h = _layer_norm(x, w['attn_norm/scale'], w['attn_norm/offset']) @ deq('qkv/weights')
  q, k, v = jnp.split(h, 3, -1)
  q = _layer_norm(q, w['q_norm/scale'])       # over the FULL d_model, not per head
  k = _layer_norm(k, w['k_norm/scale'])
  q = _rope(q.reshape(n_len, heads, head_dim), pos)
  k = _rope(k.reshape(n_len, heads, head_dim), pos)
  v = v.reshape(n_len, heads, head_dim)
  logits = jnp.einsum('ihd,jhd->hij', q, k) * head_dim ** -0.5
  ctx = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(n_len, -1)
  x = x + (ctx @ deq('attn_out/weights')) / scale
  h = _layer_norm(x, w['ffn_norm/scale'], w['ffn_norm/offset']) @ deq('fc1/weights')
  n = h.shape[-1] // 2
  return x + ((jax.nn.silu(h[..., :n]) * h[..., n:]) @ deq('fc2/weights')) / scale


def forward(ids, p, dims):
  """-> (n_layers + 1, L, d_model) hidden states, BOS/EOS still attached.

  A lax.scan over the blocks, NOT a python loop. The loop traced all 80 blocks
  into one graph and moved each block's weights to the device as it went, so the
  whole 25 GB tower ended up resident and a 23 GB card ran out on the last few
  layers -- the native torch implementation never hits this because it executes
  eagerly and frees as it goes. Scanning compiles ONE block body and keeps the
  weights as the scanned axis, which is also why they stay int8 on the device
  and are dequantised inside the body: 6.35 GB resident instead of 25.4 GB. That
  is the real reason for int8 here; the download size is a side benefit.

  ESMFold2 consumes ALL 81 states -- embedding plus 80 blocks -- and mixes them
  with a learned softmax, so none can be skipped. The LAST is post the stack's
  final LayerNorm while the other 80 are the raw residual stream; returning the
  pre-norm value there reads corr 0.909 against native on layer 80 where every
  other layer is >= 0.9998.
  """
  ids = jnp.asarray(ids)
  n_len = ids.shape[0]
  heads, head_dim = dims['n_heads'], dims['d_model'] // dims['n_heads']
  scale = dims['residual_scale']
  x = jnp.asarray(p['embed/weights']).astype(jnp.float32)[ids]
  pos = jnp.arange(n_len, dtype=jnp.float32)
  xs = {k[len('blocks/'):]: jnp.asarray(v)
        for k, v in p.items() if k.startswith('blocks/')}

  def body(carry, w):
    out = _block(carry, w, pos, heads, head_dim, scale)
    return out, out

  x, states = jax.lax.scan(body, x, xs)
  states = jnp.concatenate([jnp.asarray(p['embed/weights']).astype(jnp.float32)[ids][None],
                            states], axis=0)
  return states.at[-1].set(_layer_norm(states[-1], p['final_norm/scale']))


def load(model_dir='~/ported/esmc'):
  """-> (params, dims). Weights stay INT8; the scan body dequantises per block.

  Deliberately not `params.get_model_haiku_params`, which dequantises on load
  and would put 25.4 GB of float32 in host memory on the way to a card that
  holds 23.
  """
  import collections
  import re
  from converters.common import read_blob

  path = os.path.join(os.path.expanduser(str(model_dir)), 'esmc.bin.zst')
  if not os.path.exists(path):
    # ESM-C has no ModelSpec -- it is a separate graph, not an AF3 model -- so
    # ensure_weights does not know about it. Fetch it here, on demand: 5.5 GB is
    # not something to pull for a run that folds from an MSA instead.
    from alphafold3.model import model_registry, weights
    os.makedirs(os.path.dirname(path), exist_ok=True)
    weights._download(
        weights._HF_URL.format(repo=model_registry.get('esmfold2').weights_repo,
                               file='esmc.bin.zst'), path)
  per_block = collections.defaultdict(dict)
  p = {}
  for scope, name, arr in read_blob(path):
    if scope.startswith('__meta__'):
      continue
    sub = scope[len('esmc/'):] if scope.startswith('esmc/') else scope
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
  from converters.quantise import SCALE_SUFFIX, dequantise_int8
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
  n_layers = p['blocks/qkv/weights'].shape[0]
  d_model = p['embed/weights'].shape[1]
  dims = dict(n_layers=n_layers, d_model=d_model,
              n_heads=d_model // 64,
              residual_scale=float(np.sqrt(n_layers / 36.0)))
  return p, dims


def embed(seq, model_dir='~/ported/esmc'):
  """sequence -> (L, n_layers + 1, d_model), BOS/EOS stripped."""
  p, dims = load(model_dir)
  ids = C.lm_input_ids(sequence_ids(seq))
  states = np.asarray(forward(ids, p, dims))
  return states[:, 1:-1, :].transpose(1, 0, 2)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  ap.add_argument('--seq')
  ap.add_argument('--pdb', help='take the sequence from this structure instead')
  ap.add_argument('--model-dir', default='~/ported/esmc')
  ap.add_argument('--out', required=True)
  a = ap.parse_args(argv)
  seq = a.seq
  if not seq:
    from converters.oracles.fold_check import parse_ca
    seq, _ = parse_ca(os.path.expanduser(a.pdb))
  h = embed(seq, a.model_dir)
  np.savez_compressed(os.path.expanduser(a.out), lm_hidden=h.astype(np.float32))
  print('wrote %s  %s' % (a.out, (h.shape,)))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
