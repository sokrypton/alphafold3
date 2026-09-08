"""Fold a target with AlphaFold 2 inside this package -- the AF2 pipeline gate.

The AF2 network here is vendored from ColabDesign v1 by way of colabdesign2, so
there is an unusually strong gate available: the SAME code and the SAME DeepMind
parameters already run in colabdesign2, and must produce the same coordinates.
`--compare` runs both and reports the difference against this model's own
run-to-run noise, which on AF2 is not small (see below).

  PYTHONPATH=src:. python dev/oracles/af2_fold_check.py alphafold2_ptm ~/6MRR.pdb
  PYTHONPATH=src:. python dev/oracles/af2_fold_check.py alphafold2_ptm --compare

NOISE, and why every fold here gets its own PROCESS. Two calls inside one
process reuse one compiled executable and come back BIT-IDENTICAL, so an
in-process rerun measures nothing and reports a noise floor of exactly zero --
which then fails any real comparison. The variation lives in COMPILATION: XLA
picks kernels by timing them, so a fresh process autotunes afresh and the same
code on the same inputs moves up to 0.23 A on an atom (mean 0.012) at fp32 with
no recycling; bfloat16 with 3 recycles reaches 0.5 A.

Worse, compiling BOTH implementations in one process is not a fair comparison
either -- the second one autotunes against a warm, fuller GPU. So: one fold, one
process, and read the cross-implementation difference against the cross-process
rerun floor.

And read the per-atom difference as BIMODAL, never as a single number.
Autotuning picks one of a few kernel configurations; two processes that happen to
pick the same one agree to ~1e-3 A, two that pick differently land ~0.2 A apart.

That is why the GATE IS ON CA-RMSD, not on the per-atom maximum. An earlier
version compared one draw of max|dx| against one draw of the rerun floor, which
is not a test: both are single samples from the same bimodal distribution, so it
passes or fails roughly at random even when the two implementations are running
identical code. Observed over four runs of this exact comparison -- three passes
and one failure -- while CA-RMSD stayed within 0.001 A every time.

CA-RMSD is the steady statistic (0.0013 A spread across runs of both
implementations), and it is also sensitive enough for the job: any real
mis-wiring of the featurisation or the parameters moves it by far more than the
tolerance here. The per-atom numbers are still printed, as diagnostics.
"""
import argparse
import os
import sys

import numpy as np

from converters.pdb import parse_ca
from dev.oracles.fold_check import kabsch_rmsd


def build_inputs(seq):
  """A single protein chain through the shared front door.

  Note this runs AF3's OWN featuriser and then re-reads it -- there is one
  featurisation in the package, not one per engine.
  """
  from alphafold3.af2 import features as af2_features
  from alphafold3.common import folding_input

  fold_input = folding_input.Input(
      name='af2_fold_check',
      chains=[folding_input.ProteinChain(id='A', sequence=seq, ptms=[],
                                         unpaired_msa='', paired_msa='',
                                         templates=[])],
      rng_seeds=[0])
  inputs, chain_seq, _batch = af2_features.featurise_input(fold_input)
  return inputs, chain_seq


def fold(model_type, seq, num_recycle=0, bfloat16=False, data_dir='~',
         runner_cls=None):
  """-> atom37 positions (num_res, 37, 3)."""
  if runner_cls is None:
    from alphafold3.af2.runner import AF2Runner as runner_cls  # noqa: N813
  inputs, chain_seq = build_inputs(seq)
  runner = runner_cls(model_type=model_type, data_dir=os.path.expanduser(data_dir),
                      num_recycle=num_recycle, use_bfloat16=bfloat16)
  out = runner.predict(inputs, chain_seq)
  return np.asarray(out['structure_module']['final_atom_positions'])


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model_type', nargs='?', default='alphafold2_ptm')
  ap.add_argument('target', nargs='?', default='~/6MRR.pdb')
  ap.add_argument('--recycle', type=int, default=0)
  ap.add_argument('--bfloat16', action='store_true')
  ap.add_argument('--data_dir', default='~')
  ap.add_argument('--save', help='write atom37 positions here (child runs)')
  ap.add_argument('--impl', default='af2', choices=('af2', 'colabdesign2'),
                  help="which AF2Runner to use; 'colabdesign2' is the reference")
  ap.add_argument('--tolerance', type=float, default=0.01,
                  help='CA-RMSD agreement required of the two implementations, '
                       'in A. Ten times the observed run-to-run spread.')
  ap.add_argument('--compare', action='store_true',
                  help="also run colabdesign2's AF2Runner and diff, together "
                       'with this build\'s own run-to-run noise')
  args = ap.parse_args(argv)
  tolerance = args.tolerance

  seq, native = parse_ca(os.path.expanduser(args.target))
  kw = dict(num_recycle=args.recycle, bfloat16=args.bfloat16,
            data_dir=args.data_dir)
  # --compare does NOT fold here. The parent would hold its allocation for the
  # whole comparison, and the children then autotune against a GPU that is
  # already full -- which showed up first as CUDA_ERROR_OUT_OF_MEMORY retries
  # and then as a hundredfold inflation of every difference below (0.27 A where
  # uncontended runs agree to 0.0024). One fold at a time on this card.
  if not args.compare:
    if args.impl == 'colabdesign2':
      sys.path.insert(0, os.path.expanduser('~/ColabDesign2'))
      from colabdesign2.af2.runner import AF2Runner as cls
      kw['runner_cls'] = cls
    pos = fold(args.model_type, seq, **kw)
    if args.save:
      np.save(args.save, pos)
    print('RESULT %s  CA-RMSD %.4f' % (args.model_type,
                                       kabsch_rmsd(pos[:, 1, :], native)))
    return 0

  import subprocess
  import tempfile

  def run(impl, out_path):
    """One fold, one fresh process -- see the module docstring."""
    cmd = [sys.executable, os.path.abspath(__file__), args.model_type,
           args.target, '--recycle', str(args.recycle),
           '--data_dir', args.data_dir, '--save', out_path, '--impl', impl]
    if args.bfloat16:
      cmd.append('--bfloat16')
    env = dict(os.environ, PYTHONPATH='src:.')
    subprocess.run(cmd, check=True, cwd=os.getcwd(), env=env,
                   stdout=subprocess.DEVNULL)
    return np.load(out_path)

  with tempfile.TemporaryDirectory() as tmp:
    a = run('af2', os.path.join(tmp, 'a.npy'))
    b = run('af2', os.path.join(tmp, 'b.npy'))
    ref = run('colabdesign2', os.path.join(tmp, 'ref.npy'))

  rmsd = [kabsch_rmsd(x[:, 1, :], native) for x in (a, b, ref)]
  print('RESULT %s  CA-RMSD %.4f' % (args.model_type, rmsd[0]))
  noise = np.abs(a - b)
  diff = np.abs(a - ref)
  print('  rerun floor      max %.3e  mean %.3e  (diagnostic, bimodal)'
        % (noise.max(), noise.mean()))
  print('  vs colabdesign2  max %.3e  mean %.3e  (diagnostic, bimodal)'
        % (diff.max(), diff.mean()))
  print('  CA-RMSD  here %.4f / %.4f   colabdesign2 %.4f   spread %.4f'
        % (rmsd[0], rmsd[1], rmsd[2], max(rmsd) - min(rmsd)))
  delta = abs(rmsd[0] - rmsd[2])
  ok = delta <= tolerance
  print('  GATE %s: |CA-RMSD difference| %.4f A %s tolerance %.4f A'
        % ('PASS' if ok else 'FAIL', delta,
           'within' if ok else 'EXCEEDS', tolerance))
  return 0 if ok else 1


if __name__ == '__main__':
  raise SystemExit(main())
