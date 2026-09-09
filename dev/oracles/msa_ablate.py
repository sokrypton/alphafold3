"""Which MSA-derived feature breaks the fold? Ablate one at a time.

  PYTHONPATH=src:. ABLATE=profile ~/venv/bin/python \
    dev/oracles/msa_ablate.py esmfold2_exp ligand_1stp

A real alignment reaches an AF3-lineage model by THREE routes, not one:

  * the MSA track itself -- `msa.rows` into the msa module,
  * `msa.profile`, a per-token class distribution inside target_feat,
  * `msa.deletion_mean`, one more target_feat column,

and target_feat is `s_inputs`, which feeds z_init, BOTH diffusion conditioners
and the confidence head. So "the MSA makes the fold worse" does not localise to
the MSA module, and NO_MSA removes all three at once. This turns each off on its
own, with the alignment otherwise intact:

  ABLATE=profile   the profile becomes zeros (featurization.PROFILE_MODE)
  ABLATE=deletion  msa.deletion_mean and the deletion_matrix become zeros
  ABLATE=track     the msa module is skipped, profile and deletion_mean stay
  ABLATE=none      the control, so the wrapper itself is not the variable

Read at trace time, so each setting is its own compile.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                              # noqa: E402


def main():
  which = os.environ.get('ABLATE', 'none')
  from alphafold3.model.network import featurization
  import fold_check
  import modality_check

  if which == 'profile':
    featurization.PROFILE_MODE = 'zero'
  elif which == 'deletion':
    # Zero it on the BATCH rather than in the graph: deletion_mean is a plain
    # target_feat column and the deletion matrix is what create_msa_feat turns
    # into has_deletion/deletion_value, so both have to go or the ablation is
    # only half done.
    orig = fold_check._fold_setup

    def patched(*a, **kw):
      batch, cfg, md = orig(*a, **kw)
      for k in list(batch):
        if k.endswith('deletion_mean') or k.endswith('deletion_matrix'):
          batch[k] = np.zeros_like(np.asarray(batch[k]))
          print('  ABLATE deletion: zeroed %s' % k)
      return batch, cfg, md

    fold_check._fold_setup = patched
  elif which == 'track':
    orig = fold_check._fold_setup

    def patched(*a, **kw):
      batch, cfg, md = orig(*a, **kw)
      # num_layer 0 skips the CALL, which is how the msa-less releases run --
      # so this is a supported configuration, not a hole poked in the graph.
      cfg.evoformer.msa_stack.num_layer = 0
      print('  ABLATE track: msa_stack.num_layer = 0')
      return batch, cfg, md

    fold_check._fold_setup = patched
  elif which != 'none':
    raise SystemExit('ABLATE must be profile/deletion/track/none, got %r'
                     % which)
  print('ABLATE=%s' % which)
  return modality_check.main(sys.argv[1:])


if __name__ == '__main__':
  sys.exit(main())
