"""Is AlphaFold 2 differentiable in the sequence, through the AF3 batch?

The design loop's whole requirement is d(loss)/d(soft_seq). This checks it two
ways: the gradient is finite and non-zero, and it agrees with a central finite
difference on individual entries.

Finite differences are only meaningful because of a fact established for this
model earlier: two calls INSIDE one process reuse one compiled executable and
are bit-identical. Across processes XLA re-autotunes and the same fold moves up
to 0.2 A, which would swamp any epsilon worth using. So every evaluation here
must stay in one process.

  PYTHONPATH=src:. python dev/oracles/af2_grad_check.py
"""
import argparse
import os

import numpy as np


def build(target, model_type, num_recycles, use_cluster_profile=True):
  import jax

  from alphafold3.af2 import inference as af2_inference
  from alphafold3.common import folding_input
  from alphafold3.constants import decoded_ccd
  from alphafold3.data import featurisation
  from alphafold3.model import feat_batch, model_registry

  from converters.pdb import parse_ca

  seq, _native = parse_ca(os.path.expanduser(target))
  fold_input = folding_input.Input(
      name='af2_grad_check', rng_seeds=[0],
      chains=[folding_input.ProteinChain(
          id='A', sequence=seq, ptms=[], unpaired_msa='', paired_msa='',
          templates=[])])
  batch = feat_batch.Batch.from_data_dict(featurisation.featurise_input(
      fold_input=fold_input, ccd=decoded_ccd.get_ccd(), buckets=None)[0])

  runner = af2_inference.AF2ModelRunner(
      model_registry.get(model_type), device=jax.devices()[0],
      model_dir=os.path.expanduser('~'), num_recycles=num_recycles,
      use_bfloat16=False, use_cluster_profile=use_cluster_profile)
  return runner, batch, seq


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--model', default='af2_ptm')
  ap.add_argument('--target', default='~/6MRR.pdb')
  ap.add_argument('--recycle', type=int, default=0)
  ap.add_argument('--cluster_profile', default='on',
                  choices=('on', 'off'),
                  help="AF2's nearest_neighbor_clusters, whose st_one_hot is a "
                       'STRAIGHT-THROUGH estimator: hard one-hot forward, '
                       'identity backward. With it on, the analytic gradient is '
                       'deliberately not the derivative of the forward pass.')
  args = ap.parse_args(argv)

  import jax
  import jax.numpy as jnp

  runner, batch, seq = build(args.target, args.model, args.recycle,
                             use_cluster_profile=args.cluster_profile == 'on')
  print('cluster_profile %s' % args.cluster_profile)
  num_tokens = len(seq)

  def loss_fn(soft_seq):
    """mean predicted lDDT, as a design loop would read it.

    The bin-centre EXPECTATION of the pLDDT head, not an argmax over its logits:
    a score read off logits is not differentiable and is not a pLDDT either.
    """
    out = runner.forward(batch, soft_seq=soft_seq,
                         key=jax.random.PRNGKey(0))
    logits = out['predicted_lddt']['logits']
    centres = (jnp.arange(logits.shape[-1]) + 0.5) * (100.0 / logits.shape[-1])
    plddt = (jax.nn.softmax(logits, axis=-1) * centres).sum(-1)
    return -jnp.mean(plddt)

  # A blurred one-hot rather than the true sequence: an exact one-hot sits on a
  # corner of the simplex where the softmax inside the trunk is saturated, and a
  # vanishing gradient there would say nothing about the general case.
  rng = np.random.default_rng(0)
  x = np.full((num_tokens, 20), 0.02, np.float32)
  from alphafold3.af2.common import residue_constants as rc
  for i, a in enumerate(seq):
    x[i, rc.restype_order.get(a, 0)] = 0.62
  x += rng.normal(0, 0.01, x.shape).astype(np.float32)
  x = np.clip(x, 1e-4, None)
  x /= x.sum(-1, keepdims=True)
  x = jnp.asarray(x)

  value, grad = jax.value_and_grad(loss_fn)(x)
  grad = np.asarray(grad)
  print('loss (-mean pLDDT) %.6f' % float(value))
  print('grad: finite %s  nonzero %d/%d  max|g| %.3e  mean|g| %.3e'
        % (bool(np.isfinite(grad).all()), int((grad != 0).sum()), grad.size,
           np.abs(grad).max(), np.abs(grad).mean()))
  if not np.isfinite(grad).all() or not np.abs(grad).max() > 0:
    print('GATE FAIL: gradient is not finite and non-zero')
    return 1

  # A DIRECTIONAL derivative along one random direction, swept over epsilon --
  # not a per-coordinate difference. Two reasons. A single simplex entry here is
  # ~0.02, so even eps=1e-2 is a ~50% relative change and measures curvature as
  # much as slope; and one coordinate can sit in a locally flat or kinked spot
  # while the gradient is perfectly correct. Sweeping eps separates the two: if
  # the ratio converges to 1 as eps shrinks, the gradient is right and the
  # earlier disagreement was step size. If it plateaus away from 1, it is a bug.
  d = rng.normal(0, 1, x.shape).astype(np.float32)
  d /= np.linalg.norm(d)
  d = jnp.asarray(d)
  analytic = float(np.sum(grad * np.asarray(d)))
  print('  directional analytic  %.6f' % analytic)
  print('  %10s %14s %8s' % ('eps', 'finite-diff', 'ratio'))
  ratios = []
  for eps in (3e-2, 1e-2, 3e-3, 1e-3):
    fd = float((loss_fn(x + eps * d) - loss_fn(x - eps * d)) / (2 * eps))
    r = fd / analytic if analytic else float('nan')
    ratios.append(r)
    print('  %10.0e %14.6f %8.3f' % (eps, fd, r))
  # Judge on the SMALLEST step, where curvature is least; the larger ones are
  # printed to show the trend rather than to be graded.
  ratios = ratios[-1:]

  ok = all(0.9 < r < 1.11 for r in ratios)
  print('GATE %s: the directional finite difference %s the analytic gradient '
        'at the smallest step' % ('PASS' if ok else 'FAIL',
                                  'matches' if ok else 'DISAGREES with'))
  return 0 if ok else 1


if __name__ == '__main__':
  raise SystemExit(main())
