"""The MSA alphabet gate for ESMFold2: every class, not just the residues.

  PYTHONPATH=src:. ~/venv/bin/python dev/oracles/esmfold2_msa_alphabet.py

`msa_parity.py` gates the MSA STACK, and it does so by injecting native's
already-EMBEDDED msa -- so the embedder, and with it the 33-class -> 32-class
alphabet remap, was never compared at all. That is the hole the inverted-MSA bug
walked through: `remap_msa_feat` sliced from `ESM_RESTYPE_OFFSET`, which is 2,
while `protein_utils.MSA_GAP_TOKEN_ID` is 1 -- so the trained GAP embedding was
dropped and AF3's gap slot filled with zeros. Depth-1 self-MSAs have no gaps, so
every fold gate passed; a real 2145-row alignment is mostly gaps, and 1STP went
0.476 -> 14.364 A with an MSA where native went 18.728 -> 3.184 A.

This is deliberately a STATIC check, for the reason `dna-parity-status` gives for
the nucleotide alphabet: an activation comparison on RANDOM msa rows cannot see
it, because random rows excite class 0 and class 1 as much as any other, so the
per-class correspondence averages out into a single corr. The mapping is a
permutation; test it as one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from alphafold3.constants import residue_names                  # noqa: E402
from converters import esmfold2                                 # noqa: E402

# ESMFold2's res_type ids, read off the vendor rather than retyped: 20 residues
# at 2..21, UNK at 22, the gap at 1. Class 0 is unused by the MSA featuriser.
# (transformers/models/esmfold2/protein_utils.py -- see PROTEIN_RESIDUE_TO_RES_TYPE,
# PROTEIN_UNK_RES_TYPE and MSA_GAP_TOKEN_ID.)
ESM_AA_FIRST, ESM_UNK, ESM_GAP = 2, 22, 1
N_ESM = 33


def esm_to_af3():
  """{ESMFold2 res_type -> AF3 msa one-hot column}, built from AF3's own order."""
  order = residue_names.POLYMER_TYPES_ORDER_WITH_ALL_UNKS_AND_GAP
  aa = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP[:20]
  m = {ESM_AA_FIRST + i: order[a] for i, a in enumerate(aa)}
  m[ESM_UNK] = order[residue_names.UNK]
  m[ESM_GAP] = order['-']
  # Everything above UNK is nucleic, in the same order in both models, so it
  # follows AF3's gap slot rather than being named one by one.
  for k in range(ESM_UNK + 1, N_ESM):
    m[k] = order['-'] + (k - ESM_UNK)
  return m


def main():
  want = esm_to_af3()
  # A marker column per ESM class: value c+1, so class 0 is distinguishable from
  # a slot that was filled with ZEROS -- which is exactly what the bug did.
  w = np.arange(1, N_ESM + 3, dtype=np.float32)[:, None]
  got = esmfold2.remap_msa_feat(w)[:, 0]
  n_af3 = len(residue_names.POLYMER_TYPES_WITH_ALL_UNKS_AND_GAP)
  print('ESMFold2 %d classes -> AF3 %d + 2 deletion columns' % (N_ESM, n_af3))
  if got.shape[0] != n_af3 + 2:
    raise SystemExit('remap_msa_feat gave %d rows, want %d'
                     % (got.shape[0], n_af3 + 2))

  bad = []
  for esm_c, af3_c in sorted(want.items()):
    if got[af3_c] != esm_c + 1:
      bad.append('  ESM class %2d -> AF3 %2d: holds %s, want ESM %d'
                 % (esm_c, af3_c, 'ZERO' if got[af3_c] == 0
                    else 'ESM %d' % (got[af3_c] - 1), esm_c))
  # Every AF3 class must be fed by SOME ESM class: a slot left at zero is a
  # feature the model can no longer read, which is silent rather than fatal.
  unfilled = [c for c in range(n_af3) if c not in set(want.values())]
  for c in unfilled:
    bad.append('  AF3 class %2d (%s) is fed by nothing'
               % (c, [k for k, v in
                      residue_names.POLYMER_TYPES_ORDER_WITH_ALL_UNKS_AND_GAP.items()
                      if v == c]))
  # The two deletion columns pass straight through, in ESMFold2's own order
  # (has_deletion then deletion_value) -- see create_msa_feat's default branch.
  for i in (0, 1):
    if got[n_af3 + i] != N_ESM + i + 1:
      bad.append('  deletion column %d misplaced (holds %g)' % (i, got[n_af3 + i]))

  if bad:
    print('MISMATCH:')
    print('\n'.join(bad))
    return 1
  print('  all %d ESMFold2 classes land on the right AF3 column '
        '(gap %d -> %d), both deletion columns in place'
        % (len(want), ESM_GAP, want[ESM_GAP]))

  # The SAME alphabet, in the two other places it is spelled out. s_inputs uses
  # AF3's NARROWER 31-class block (no DN), and `permute_s_inputs` keeps ESM's own
  # 33 widths for the diffusion conditioner -- where the widening happens in the
  # GRAPH instead, so the graph's version has to agree with the converter's.
  n31 = len(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
  perm = esmfold2.esm_class_of_af3(n31)
  bad = ['  s_inputs AF3 %2d -> ESM %2d, want %2d' % (c, perm[c], want_c)
         for c, want_c in sorted(want.items(), key=lambda kv: kv[1])
         if want_c < n31 and perm[want_c] != c]
  if bad:
    print('MISMATCH in the s_inputs blocks:')
    print('\n'.join(bad))
    return 1
  print('  the 31-class s_inputs blocks take the same permutation '
        '(DN, ESM class %d, has no AF3 slot and is dropped)' % (N_ESM - 1))

  # diffusion_head widens AF3's 31 columns back to ESM's 33 before the
  # conditioner's LayerNorm. Compare it against the converter's table by
  # WIDENING a marker vector and checking each ESM slot holds the AF3 column
  # that `esm_class_of_af3` says feeds it.
  import jax.numpy as jnp
  from alphafold3.model.network import diffusion_head              # noqa: F401
  block = jnp.arange(1, n31 + 1, dtype=jnp.float32)
  wide = np.asarray(_graph_widen(block, n31))
  bad = []
  for esm_c in range(N_ESM):
    af3_c = [c for c in range(n31) if perm[c] == esm_c]
    exp = af3_c[0] + 1 if af3_c else 0
    if wide[esm_c] != exp:
      bad.append('  ESM slot %2d holds AF3 %s, want %s'
                 % (esm_c, 'ZERO' if wide[esm_c] == 0 else int(wide[esm_c] - 1),
                    'ZERO' if not af3_c else af3_c[0]))
  if bad:
    print("MISMATCH between the graph's widening and the converter:")
    print('\n'.join(bad))
    return 1
  print("  diffusion_head's widening to %d agrees with the converter" % N_ESM)
  return 0


def _graph_widen(block, n):
  """diffusion_head's PAIR_ONLY_TRUNK widening, on one block.

  A copy rather than a call: the widening sits inside `_conditioning`, which
  needs a whole batch and a parameter scope to reach. Copied code is exactly
  what rots, so the numbers here are the CONVERTER's, and this only checks the
  SHAPE of the rearrangement -- if the graph's own line changes, the assertion
  below is what fails.
  """
  import inspect
  import jax.numpy as jnp
  from alphafold3.model.network import diffusion_head
  src = inspect.getsource(diffusion_head.DiffusionHead._conditioning)
  want = '[zero, block[..., 21:22], block[..., :21], block[..., 22:n], zero]'
  if want not in src:
    raise SystemExit('diffusion_head no longer widens with %s -- update this '
                     'gate to match, do not delete it' % want)
  zero = jnp.zeros_like(block[..., :1])
  return jnp.concatenate(
      [zero, block[..., 21:22], block[..., :21], block[..., 22:n], zero],
      axis=-1)


if __name__ == '__main__':
  sys.exit(main())
