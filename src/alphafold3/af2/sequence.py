'''
the continuous relaxation of sequence

This is ColabDesign's core trick and it is model-agnostic: a free parameter is
turned into a distribution over amino acids, sharpened by a schedule, and fed to
whatever model is downstream. v1 keeps it in shared/model.py as soft_seq.

The schedule knobs, all traced so they can change between steps without a
recompile:

  alpha   logit scale
  temp    softmax temperature
  soft    interpolate raw logits -> softmax
  hard    interpolate that -> straight-through argmax

The straight-through estimator (`hard`) is what lets a discrete sequence receive
a gradient: forward is one-hot, backward is the softmax.
'''

from __future__ import annotations

import jax
import jax.numpy as jnp


def soft_seq(x, bias=None, opt=None, key=None, num_seq=None, shuffle_first=True):
  '''parameters -> a dict of sequence representations

  Ported from v1's shared/model.py:soft_seq. Returns input / logits / pssm /
  soft / hard / pseudo, where `pseudo` is the one the model consumes.
  '''
  opt = {} if opt is None else opt
  seq = {'input': x}

  # shuffle which row is the query, when there is more than one
  if x.ndim == 3 and x.shape[0] > 1 and key is not None:
    key, sub_key = jax.random.split(key)
    if num_seq is None or x.shape[0] == num_seq:
      if shuffle_first:
        n = jax.random.randint(sub_key, [], 0, x.shape[0])
        seq['input'] = seq['input'].at[0].set(seq['input'][n]).at[n].set(seq['input'][0])
    else:
      n = jnp.arange(x.shape[0])
      n = (jax.random.permutation(sub_key, n) if shuffle_first
           else jnp.append(0, jax.random.permutation(sub_key, n[1:])))
      seq['input'] = seq['input'][n[:num_seq]]

  seq['logits'] = seq['input'] * opt.get('alpha', 2.0)
  if bias is not None:
    seq['logits'] = seq['logits'] + bias

  seq['pssm'] = jax.nn.softmax(seq['logits'])
  seq['soft'] = jax.nn.softmax(seq['logits'] / opt.get('temp', 1.0))
  hard = jax.nn.one_hot(seq['soft'].argmax(-1), seq['soft'].shape[-1])
  # straight-through: one-hot forward, softmax gradient backward
  seq['hard'] = jax.lax.stop_gradient(hard - seq['soft']) + seq['soft']

  s, h = opt.get('soft', 0.0), opt.get('hard', 0.0)
  seq['pseudo'] = s * seq['soft'] + (1 - s) * seq['input']
  seq['pseudo'] = h * seq['hard'] + (1 - h) * seq['pseudo']
  return seq


def pin_fixed(seq, wt_aatype, seq_fixed, alphabet_size=20):
  '''overwrite the designed sequence with wildtype wherever seq_fixed is set

  v1 does this as `jnp.where(fix_pos[:,None], wt_seq, x)` inside _update_seq.
  Keeping params full-length and overwriting (rather than parameterising only the
  free positions) means seq_fixed can change between steps without a recompile.
  '''
  if seq_fixed is None:
    return seq
  ref = jax.nn.one_hot(jnp.maximum(wt_aatype, 0), alphabet_size)
  m = jnp.asarray(seq_fixed, bool)

  def fix(x):
    pad = x.shape[-1] - ref.shape[-1]
    r = jnp.pad(ref, [[0, 0], [0, pad]]) if pad > 0 else ref[..., :x.shape[-1]]
    return jnp.where(m[:, None], r, x)

  return jax.tree_util.tree_map(fix, seq)


def expand_copies(x, copies=1, block_diag=True, x_default=0):
  '''tile a sequence over copies, optionally block-diagonalising the MSA

  block_diag builds an MSA where each copy's sequences appear only in its own
  block, with a gap elsewhere -- v1 uses this so AF2 does not see the copies as
  one long aligned sequence.
  '''
  if copies == 1:
    return x
  if x.ndim == 1:
    return jnp.tile(x, copies)
  if x.ndim == 2:
    n, length = x.shape
    a = 1
    x = x[:, :, None]
    new_shape = (n * copies + 1, length * copies) if block_diag else (n, length * copies)
  else:
    n, length, a = x.shape
    new_shape = ((n * copies + 1, length * copies, a) if block_diag
                 else (n, length * copies, a))

  y = jnp.tile(x, [1, copies, 1])
  if block_diag:
    y_ = y.reshape(n, copies, length, a)
    i = jnp.arange(copies)
    diag = jnp.full((n, copies, copies, length, a), x_default, dtype=y.dtype)
    diag = diag.at[:, i, i].set(y_)
    diag = diag.swapaxes(0, 1).reshape(n * copies, copies * length, a)
    y = jnp.concatenate([y[:1], diag], 0)
  return y.reshape(new_shape)


def designed(outputs, kind='pseudo'):
  """the designed sequence from a runner's outputs, as (L, 20)

  Both runners publish soft_seq's full dict under outputs['seq'] -- AF2 always
  did, AF3 does since it was found publishing only the blended array, which made
  a consumer asking for 'hard' silently receive the raw parameters.

  Reading it in one place rather than three: when the contract changed, the
  three call sites that had each open-coded it did not all get updated, and the
  one that did not failed with "argmax requires ndarray, got dict" a long way
  from the change.

    kind='pseudo'  what the model consumed
    kind='hard'    straight-through one-hot, for anything wanting a discrete call
  """
  import jax.numpy as jnp

  seq = outputs.get('seq') if hasattr(outputs, 'get') else None
  if seq is None:
    return None
  x = seq[kind] if isinstance(seq, dict) else seq
  x = jnp.asarray(x)
  return x[0] if x.ndim == 3 else x


def aa_bias(length, rm_aa=None, bias=None, positions=None):
  '''a static (length, 20) logit bias over the amino-acid alphabet

  AF2 hallucination has a well-known composition pathology -- it reaches for
  cysteine and tryptophan because they buy contacts and confidence cheaply, and
  the resulting sequences are not synthesisable. BindCraft omits cysteine
  outright. Before this there was no way to say so: `inputs['bias']` was read by
  both runners but only ever written by a module, so a user had no route to it.

  rm_aa: e.g. 'C' or 'C,W' -- excluded by a large negative logit, not a hard
         mask, so the straight-through estimator still has a gradient.
  bias:  {'A': +1.5, ...} to nudge without excluding.
  positions: bool array; restrict the bias to these positions (default: all).
  '''
  import numpy as np
  from .pdb import AA1
  out = np.zeros((length, len(AA1)), dtype=np.float32)
  row = np.zeros(len(AA1), dtype=np.float32)
  if rm_aa:
    for aa in str(rm_aa).replace(',', ''):
      aa = aa.strip().upper()
      if not aa:
        continue
      if aa not in AA1:
        raise ValueError(f'rm_aa: {aa!r} is not one of {AA1}')
      row[AA1.index(aa)] = -1e8
  for aa, v in (bias or {}).items():
    if aa.upper() not in AA1:
      raise ValueError(f'bias: {aa!r} is not one of {AA1}')
    row[AA1.index(aa.upper())] += float(v)
  out[:] = row
  if positions is not None:
    out = np.where(np.asarray(positions, bool)[:, None], out, 0.0)
  return out
