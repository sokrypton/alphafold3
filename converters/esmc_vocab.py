"""ESM-C's alphabet and chain wrapping. NumPy only, deliberately.

Split out of esmc_embed so it can be imported from an environment that has torch
but not jax: the native-side gate runs in ~/venv_esm, and importing the jax
forward there just to reach a token table would fail. One source of truth for
the vocabulary, which is the part that must not drift between the two sides --
verified byte-for-byte against ESMFold2's own `input_ids` for 6MRR.
"""

from __future__ import annotations

import numpy as np

# Index 0/1/2/32 are BOS/PAD/EOS/MASK, which converters.esmc asserts; the rest is
# the standard ESM residue ordering.
VOCAB = ('<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B '
         'U Z O . - | <mask>').split()
_TOKENS = {t: i for i, t in enumerate(VOCAB)}
BOS, PAD, EOS, MASK = 0, 1, 2, 32
UNK = _TOKENS['<unk>']


def sequence_ids(seq):
  """One-letter sequence -> ESM-C residue token ids (no BOS/EOS)."""
  return np.array([_TOKENS.get(c.upper(), UNK) for c in seq], np.int64)


def lm_input_ids(token_ids):
  """Wrap one chain's residue ids as ESMFold2 does: [BOS, ids..., EOS].

  Multi-chain inserts [EOS, BOS] between chains; padding is PAD and
  `sequence_id = cumsum(ids == BOS) - 1`, set to -1 on PAD, restricts attention
  to within a chain.
  """
  return np.concatenate([[BOS], np.asarray(token_ids, np.int64), [EOS]])
