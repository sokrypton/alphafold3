"""Does our structural-token ATOM AXIS match the residue-major one native uses?

    PYTHONPATH=src:.:dev/oracles python dev/oracles/struct_atom_axis_probe.py [seq ...]

opendde tokenises a standard protein residue into a BACKBONE token and a
SIDECHAIN token, and its backbone set includes OXT (data/tokenizer.py:21 -- ours
is copied from it). At a non-glycine chain terminus that makes the backbone set
NON-CONTIGUOUS in the residue's own atom order:

    dense, residue-major   N CA C O CB SG OXT
    our struct flat order  N CA C O OXT | CB SG      <- a 3-cycle

The two implementations then part company on what the ATOM AXIS is. Native
opendde keeps every atom in the ORIGINAL atom array and has each token carry
`atom_indices` INTO it (`_get_atom_to_token_idx` fills an array indexed by atom
i over range(n_atoms)), so its axis is residue-major. Ours re-packs atoms into
(token, slot) rows and flattens row-major, so our axis is token-major -- and the
atom-attention windows are cut on it.

WHETHER THAT BITES depends on whether a 32-atom query block straddles the
terminal residue: inside one block the permutation is invisible (the features
travel with the atoms), across a boundary it changes which atoms attend to which.
This prints both, per input.

Reported by chlee19990109-cloud (their D108, the same bug in their own tree);
reproduced and scoped here. Their note that our padded-window work was verified
on 1EHZ is right and is half the reason it survived: RNA puts the backbone atoms
first, so no permutation appears. Ubiquitin is blind too -- it ends in GLY, which
takes the single-token path and is never split.
"""

import sys

import numpy as np

sys.path.insert(0, 'dev/oracles')
import fold_check


def _dec(v):
  v = np.asarray(v)
  if v.ndim == 2:
    v = v.argmax(-1)
  return ''.join(chr(32 + int(i)) for i in v).strip()


def probe(seq, model='opendde', chains=None, tag=None):
  """-> (n_atoms, atoms at a different index, query blocks whose SET differs)."""
  b, _, _ = fold_check._fold_setup(model, seq, None, chains=chains)
  if 'struct/ref_atom_name_chars' not in b:
    raise SystemExit('%s has no structural-token layout' % model)
  snm = np.asarray(b['struct/ref_atom_name_chars'])
  smk = np.asarray(b['struct/pred_dense_atom_mask'])
  srix, sasym = np.asarray(b['struct/residue_index']), np.asarray(b['struct/asym_id'])
  ours = [(int(sasym[t]), int(srix[t]), _dec(snm[t, a]))
          for t in range(smk.shape[0]) for a in range(smk.shape[1]) if smk[t, a]]
  dnm, dmk = np.asarray(b['ref_atom_name_chars']), np.asarray(b['pred_dense_atom_mask'])
  drix, dasym = np.asarray(b['residue_index']), np.asarray(b['asym_id'])
  canon = [(int(dasym[t]), int(drix[t]), _dec(dnm[t, a]))
           for t in range(dmk.shape[0]) for a in range(dmk.shape[1]) if dmk[t, a]]
  n = len(ours)
  moved = sum(1 for x, y in zip(ours, canon) if x != y)
  # 32 is AF3's query block; a block whose SET is unchanged computes the same
  # thing under a permutation, because every per-atom feature moves with its atom
  diff = sum(1 for i in range(0, n, 32)
             if set(ours[i:i + 32]) != set(canon[i:i + 32]))
  print('%-34s %5d atoms  moved %3d  query blocks differing %d/%d%s'
        % (tag or seq[:30], n, moved, diff, (n + 31) // 32,
           '   <-- LIVE' if diff else '   (inert here)'))
  return n, moved, diff


if __name__ == '__main__':
  args = sys.argv[1:]
  if args:
    for s in args:
      probe(s)
  else:
    probe('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
          tag='ubiquitin (ends GLY)')
    probe(fold_check.parse_ca('/home/ubuntu/6MRR.pdb')[0], tag='6MRR')
    for L in (12, 18, 19, 24, 25):
      probe('A' * (L - 1) + 'W', tag='poly-A(%d) ending TRP' % L)
