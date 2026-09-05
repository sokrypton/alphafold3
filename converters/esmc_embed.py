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


def forward(ids, p, dims):
  """-> (n_layers + 1, L, d_model) hidden states, BOS/EOS still attached.

  ESMFold2 consumes ALL 81 -- embedding plus 80 blocks -- and mixes them with a
  learned softmax, so none can be skipped. The LAST state is post the stack's
  final LayerNorm while the other 80 are the raw residual stream; returning the
  pre-norm value there reads corr 0.909 against native on layer 80 where every
  other layer is >= 0.9998.
  """
  ids = jnp.asarray(ids)
  n_len = ids.shape[0]
  heads, head_dim = dims['n_heads'], dims['d_model'] // dims['n_heads']
  scale = dims['residual_scale']
  x = jnp.asarray(p['embed/weights'])[ids]
  states = [x]
  pos = jnp.arange(n_len, dtype=jnp.float32)
  blocks = {k[len('blocks/'):]: v for k, v in p.items() if k.startswith('blocks/')}
  for i in range(dims['n_layers']):
    b = {k: jnp.asarray(v[i]) for k, v in blocks.items()}
    h = _layer_norm(x, b['attn_norm/scale'], b['attn_norm/offset']) @ b['qkv/weights']
    q, k, v = jnp.split(h, 3, -1)
    q = _layer_norm(q, b['q_norm/scale'])       # over the FULL d_model, not per head
    k = _layer_norm(k, b['k_norm/scale'])
    q = _rope(q.reshape(n_len, heads, head_dim), pos)
    k = _rope(k.reshape(n_len, heads, head_dim), pos)
    v = v.reshape(n_len, heads, head_dim)
    logits = jnp.einsum('ihd,jhd->hij', q, k) * head_dim ** -0.5
    ctx = jnp.einsum('hij,jhd->ihd', jax.nn.softmax(logits, -1), v).reshape(n_len, -1)
    x = x + (ctx @ b['attn_out/weights']) / scale
    h = _layer_norm(x, b['ffn_norm/scale'], b['ffn_norm/offset']) @ b['fc1/weights']
    n = h.shape[-1] // 2
    x = x + ((jax.nn.silu(h[..., :n]) * h[..., n:]) @ b['fc2/weights']) / scale
    states.append(x)
  states[-1] = _layer_norm(states[-1], p['final_norm/scale'])
  return jnp.stack(states)


def load(model_dir='~/ported/esmc'):
  """-> (params dict keyed 'embed/weights' etc, dims). Dequantises int8."""
  from alphafold3.model import params as afp
  raw = afp.get_model_haiku_params(model_dir=os.path.expanduser(str(model_dir)))
  import collections
  import re
  p = {}
  per_block = collections.defaultdict(dict)
  for scope, leaves in raw.items():
    if scope.startswith('__meta__'):
      continue
    sub = scope[len('esmc/'):] if scope.startswith('esmc/') else scope
    # the blocks are stored one record per layer (see split_stacked_blocks);
    # restack them on the leading axis, which is what forward() indexes.
    m = re.match(r'(blocks/.+)/(\d+)$', sub)
    for name, arr in leaves.items():
      if m:
        per_block['%s/%s' % (m.group(1), name)][int(m.group(2))] = arr
      else:
        p['%s/%s' % (sub, name) if sub else name] = arr
  for key, layers in per_block.items():
    p[key] = np.stack([layers[i] for i in sorted(layers)])
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
