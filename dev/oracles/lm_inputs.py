"""Generate the LANGUAGE-MODEL inputs the L5/L6 sweep needs, per target.

Two of the ported families do not fold from sequence alone:

  * chai-1's token stream is mostly ESM2. With `esm=None` it folds A DIFFERENT
    MODEL -- 5.70 A where chai reaches 0.642 on a natural protein, and 3.9 A
    against 0.456 on 1STP+BTN. Every in-repo chai1 number measured without
    ESM_EMB is a no-ESM number.
  * ESMFold2 folds from ESM-C hidden states and has no MSA at all. Each release
    is trained against its OWN tower (esmc 6B / esmc_600m / esmc_300m -- see
    model_registry.ESMFOLD2_VARIANTS' `esmc` field) and against its own shim;
    crossing them reads corr 0.026.

So the sweep needs one file per (tower, case). Sequences come from
`modality_check.py --dump_seqs`, i.e. from the SAME chain construction the fold
uses, in chain order -- `esm.embed` concatenates chains and the rows land on the
batch's protein tokens in that order.

    PYTHONPATH=src:. python dev/oracles/lm_inputs.py            # all of them
    PYTHONPATH=src:. python dev/oracles/lm_inputs.py --tower esm2 --case ligand_1stp

Writes dev/oracles/lm_inputs/<tower>.<case>.npz, which run_all_parity.sh picks
up automatically (and says so in the log when it does, because a missing file
must not look like a real number).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT = os.path.join(_HERE, 'lm_inputs')

# esm2 for chai1; the three ESM-C tiers for the esmfold2 releases that use them.
TOWERS = ('esm2', 'esmc', 'esmc_600m', 'esmc_300m')
# Only the protein-bearing cases. rna_1ehz and dna_1lmb have no protein chain,
# and ptm_5k9p carries the same sequence as plain_5k9p (SEP is a modification of
# a residue already there), so it reuses that file.
CASES = ('protein_6mrr', 'plain_5k9p', 'ligand_1stp', 'complex_1lmb')


def sequences(case):
  """-> the case's protein sequences, in chain order, from modality_check."""
  env = dict(os.environ, PYTHONPATH='src:.:%s' % env_pp())
  out = subprocess.run(
      [sys.executable, 'dev/oracles/modality_check.py', 'boltz2', case,
       '--dump_seqs'],
      cwd=_ROOT, env=env, capture_output=True, text=True, timeout=900)
  if out.returncode != 0:
    raise SystemExit('--dump_seqs failed for %s:\n%s' % (case, out.stderr[-2000:]))
  return [l.split('\t', 1)[1] for l in out.stdout.splitlines()
          if l.startswith('SEQ\t')]


def env_pp():
  return os.environ.get('PYTHONPATH', '')


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--tower', action='append', default=[], choices=TOWERS)
  ap.add_argument('--case', action='append', default=[], choices=CASES)
  ap.add_argument('--force', action='store_true')
  a = ap.parse_args(argv)
  sys.argv = sys.argv[:1]                  # absl parses argv lazily
  towers = a.tower or list(TOWERS)
  cases = a.case or list(CASES)
  os.makedirs(OUT, exist_ok=True)

  from alphafold3.model import esm

  for case in cases:
    seqs = sequences(case)
    if not seqs:
      print('%-16s no protein chain, nothing to embed' % case)
      continue
    print('%-16s %d chain(s): %s' % (case, len(seqs), [len(s) for s in seqs]))
    for tower in towers:
      path = os.path.join(OUT, '%s.%s.npz' % (tower, case))
      if os.path.exists(path) and not a.force:
        print('    %-10s cached' % tower)
        continue
      family = 'esm2' if tower == 'esm2' else 'esmc'
      if family == 'esmc' and len(seqs) > 1:
        # `esm.embed` refuses multi-chain ESM-C: its wrapping is [EOS, BOS]
        # separated and it will not guess. No loss here -- the only multi-chain
        # case is complex_1lmb, which carries DNA and is outside ESMFold2's
        # scope (protein only) either way. Said out loud rather than crashing,
        # so a future protein-only complex is a known gap and not a traceback.
        print('    %-10s SKIPPED: %d chains, and ESM-C embedding is one chain '
              'at a time' % (tower, len(seqs)))
        continue
      rows = esm.embed(seqs, None, family, None if tower in ('esm2', 'esmc')
                       else tower)
      # The two consumers want different keys: fold_check reads `esm` for ESM2
      # and `lm_hidden` for ESM-C (which it then puts through THIS model's shim).
      key = 'esm' if tower == 'esm2' else 'lm_hidden'
      np.savez_compressed(path, **{key: np.asarray(rows, np.float32)})
      print('    %-10s %s -> %s' % (tower, (rows.shape,), os.path.basename(path)))
  return 0


if __name__ == '__main__':
  sys.exit(main())
