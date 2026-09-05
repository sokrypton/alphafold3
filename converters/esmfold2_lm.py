"""ESMFold2's LanguageModelShim, as its own graph.

ESMFold2 folds from ESM-C 6B's hidden states, not from an MSA. The tower is a
6.35B-parameter artifact of its own and the shim is the last few layers of it:
a layer-mix over 81 hidden states, a downprojection, and an outer product that
turns the resulting single representation into a PAIR one. None of that has an
AlphaFold 3 counterpart to reuse, so it lives here instead of in the shared
graph -- the same reasoning that keeps chai-1's ESM2 in converters/esm_embed.py.

What the AF3 graph does with the result is ordinary pair-stack work and stays
there: the 25% per-loop dropout and the four-block lm_encoder, which is the
PairFormerIteration-with-zeroed-attention identity the trunk already rides.

    python -m converters.esmfold2_lm --hidden hidden.npz --model-dir ~/ported/esmfold2 \
        --out lm_pair.npz

`hidden` is (num_tokens, num_layers, lm_width) ESM-C hidden states, in the
featuriser's token order. Run without ESM-C and ESMFold2 still folds -- 6MRR at
1.719 A natively -- it just folds without its language model.
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np


def load_params(model_dir):
  """-> the shim weights written beside the blob by convert_esmfold2_weights."""
  p = pathlib.Path(os.path.expanduser(str(model_dir))) / 'esmfold2.lm.npz'
  if not p.exists():
    raise FileNotFoundError(
        f'{p} not found -- rerun convert_esmfold2_weights, or run without ESM-C')
  return {k.replace('.', '/'): v for k, v in np.load(p).items()}


def _layer_norm(x, scale, offset, eps=1e-5):
  m = x.mean(-1, keepdims=True)
  v = x.var(-1, keepdims=True)
  return (x - m) / np.sqrt(v + eps) * scale + offset


def _gelu(x):
  from scipy.special import erf
  return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def shim(hidden, params):
  """(num_tokens, num_layers, lm_width) -> (num_tokens, num_tokens, c_z).

  `combine` is already softmaxed at conversion time: the layer mix is a constant,
  so it folds to a plain (num_layers,) array. It peaks on the LAST ESM-C layers
  -- 79/80/78 hold 58% of the mass -- so the tower cannot be truncated from the
  top, though ~31 of the 81 states carry 99% of it.
  """
  p = params
  hidden = np.asarray(hidden, np.float32)
  if hidden.ndim == 4 and hidden.shape[0] == 1:
    hidden = hidden[0]
  x = _layer_norm(hidden, p['lm_norm/scale'], p['lm_norm/offset'])
  x = np.einsum('k,lkc->lc', p['combine'], x @ p['lm_projection/weights'])
  x = x @ p['downproject/weights'] + p['downproject/bias']
  # SingleToPair: the outer product carries BOTH a product and a difference, so
  # the pair rep sees magnitude and direction, not just agreement.
  z = np.concatenate([x[:, None] * x[None, :], x[:, None] - x[None, :]], -1)
  z = z @ p['pair_mlp_1/weights'] + p['pair_mlp_1/bias']
  z = _gelu(z) @ p['pair_mlp_2/weights'] + p['pair_mlp_2/bias']
  return _layer_norm(z, p['pair_norm/scale'], p['pair_norm/offset'])


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  ap.add_argument('--hidden', required=True,
                  help='npz or npy of (num_tokens, num_layers, lm_width)')
  ap.add_argument('--model-dir', default='~/ported/esmfold2')
  ap.add_argument('--key', default=None,
                  help='array name inside --hidden, if it is an npz')
  ap.add_argument('--out', required=True)
  a = ap.parse_args(argv)
  h = np.load(os.path.expanduser(a.hidden))
  if hasattr(h, 'files'):
    h = h[a.key or ('lm_hidden' if 'lm_hidden' in h.files else h.files[0])]
  z = shim(h, load_params(a.model_dir))
  np.savez_compressed(os.path.expanduser(a.out), lm_pair=z.astype(np.float32))
  print('wrote %s  %s' % (a.out, (z.shape,)))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
