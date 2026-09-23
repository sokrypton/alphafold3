"""How much of a design gradient comes through the MSA and profile channels?

The designed sequence reaches an AF3-family trunk three ways: the target
feature, the MSA one-hot, and the profile. BoltzDesign1 detaches the last two
(boltzdesign_utils.py:797) so only the first carries a gradient. Which is
better is measurable twice over -- by designing (done: boltz2 reads 0.935
attached against 0.914 detached) and, here, by looking at the gradient itself.

For one batch and one soft sequence, take d(loss)/d(soft_seq) under each
setting and report its norm and its angle to the full gradient. That says
whether the MSA paths add signal, add noise, or add a near-duplicate of what
the target-feature path already carries.

  PYTHONPATH=src:. python dev/oracles/msa_gradient_probe.py --model boltz2
"""
import argparse
import os
import sys

sys.argv_full = list(sys.argv)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--model', default='boltz2')
  ap.add_argument('--target', default='PD1')
  ap.add_argument('--binder', type=int, default=95)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--steps', type=int, default=0,
                  help='design this many steps first, so the probe sees a '
                       'sequence mid-optimisation rather than the init')
  a = ap.parse_args()
  sys.argv = [sys.argv[0]]
  from absl import flags as _af
  if not _af.FLAGS.is_parsed():
    _af.FLAGS(sys.argv)

  import numpy as np
  import jax
  import jax.numpy as jnp
  from alphafold3.model.network import featurization as F3
  sys.path.insert(0, os.path.expanduser('~/ColabDesign2_new'))
  sys.path.insert(0, os.path.expanduser('~/ColabDesign2'))
  from colabdesign2.app import af3

  pdb = os.path.join(os.path.expanduser(
      os.environ.get('BC_TARGETS', '~/bindcraft/targets')), f'{a.target}.pdb')
  m = af3.model(f'A:{a.binder}', pdb=pdb, weights=a.model, num_recycles=1,
                verbose=False)
  m.design(seeds=(a.seed,), iters=a.steps)
  d = m.results[-1]['design']

  def grad_under(mode):
    prev = F3.set_msa_gradient(mode)
    try:
      d.grad_fn.cache_clear() if hasattr(d.grad_fn, 'cache_clear') else None
      d._grad_stamp = None                   # force a retrace under this mode
      aux = d.run(backprop=True)
      return np.asarray(aux['grad']['seq'], np.float64), float(aux['loss'])
    finally:
      F3.set_msa_gradient(prev)

  full, loss = grad_under('both')
  print(f'{a.model}, {a.target}, binder {a.binder}, seed {a.seed}, '
        f'{a.steps} steps of design first')
  print(f'  loss {loss:.4f}   |g_full| {np.linalg.norm(full):.4e}\n')
  print(f"  {'channels':10s} {'|g|':>11s} {'|g|/|g_full|':>13s} "
        f"{'cos to full':>12s} {'|g_full - g|':>13s}")
  for mode in ('both', 'msa', 'profile', 'none'):
    g, _ = grad_under(mode)
    n = np.linalg.norm(g)
    cos = float((g * full).sum() / (n * np.linalg.norm(full) + 1e-12))
    print(f"  {mode:10s} {n:11.4e} {n / (np.linalg.norm(full) + 1e-12):13.3f} "
          f"{cos:12.4f} {np.linalg.norm(full - g):13.4e}")
  print('\n  `none` is BoltzDesign\'s setting: target-feature path only.')
  print('  cos ~1 with a smaller norm means the MSA paths mostly rescale;')
  print('  a cos well below 1 means they point somewhere else.')


if __name__ == '__main__':
  main()
