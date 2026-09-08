"""Is a model differentiable in the SEQUENCE? One check, either engine.

Design needs d(loss)/d(soft_seq). This asks that of any model in the registry --
the 22 on the AF3 graph and the two on the AlphaFold 2 network -- through the
same directional finite-difference test, so "differentiable" means the same
thing across the library.

  PYTHONPATH=src:. python dev/oracles/grad_check.py openfold3
  PYTHONPATH=src:. python dev/oracles/grad_check.py af2_ptm

READ THE EPSILON SWEEP FROM THE LARGE END. The forward is float32, so at small
eps the central difference f(+)-f(-) is a subtraction of two numbers of
magnitude ~10-100 that differ in their last few bits, and cancellation destroys
it: measured on af2_ptm, the ratio is 0.985 at eps=1e-1 and 1.744 at eps=1e-3.
Judging on the smallest step -- the instinct from exact arithmetic -- is exactly
backwards here, and reads a correct gradient as a failure. The gate therefore
scores the well-conditioned window and prints the rest as trend.

ONE PROCESS. Two evaluations inside a process are bit-identical (measured:
spread 0.000e+00 over five). Across processes XLA re-autotunes and the same loss
moves by ~1e-3, which is larger than the differences being measured -- so a
sweep split across processes compares nothing.
"""
import argparse
import os

import numpy as np

# Error against step size is a U: truncation dominates at large eps, rounding at
# small. The floor sits at a different eps for each model, because it depends on
# |grad| and on the loss scale -- af2_ptm bottoms out near 1e-2, openfold3 needs
# 1e-1. So the gate scores the BEST step rather than a fixed window; a wrong
# gradient would not match at any step.
_ALL_EPS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3)


def _blurred_one_hot(seq, rng):
  """A point INSIDE the simplex, not on a corner.

  An exact one-hot sits where the trunk's softmax is saturated; a small gradient
  there would say nothing about the general case a design run lives in.
  """
  from alphafold3.af2.common import residue_constants as rc

  x = np.full((len(seq), 20), 0.02, np.float32)
  for i, a in enumerate(seq):
    x[i, rc.restype_order.get(a, 0)] = 0.62
  x += rng.normal(0, 0.01, x.shape).astype(np.float32)
  x = np.clip(x, 1e-4, None)
  return x / x.sum(-1, keepdims=True)


def _project(contact_probs):
  """A fixed random linear functional of the contact map, not its mean.

  The MEAN contact probability is close to a conserved quantity -- a chain of a
  given length has roughly a given contact density whatever its sequence -- so
  it is very nearly flat in `soft_seq`. Measured on openfold3 the whole gradient
  came to max|g| 7.7e-04, which leaves a central difference of ~130 ULP and a
  check that reads pure float noise.

  A fixed random projection of the same map has no such conservation and probes
  the whole output rather than one aggregate. It is not a design objective and
  is not meant to be -- design targets SPECIFIC contacts (see the i_con work),
  which is exactly why their gradients are informative where the mean's is not.
  """
  import jax.numpy as jnp
  import numpy as _np

  w = _np.random.default_rng(1234).normal(0, 1, contact_probs.shape[-2:])
  w /= _np.linalg.norm(w)
  return jnp.sum(contact_probs * jnp.asarray(w, contact_probs.dtype))


def _contact_loss(out):
  """Mean P(contact) from the distogram -- the same scalar on either engine.

  Deliberately not pLDDT or pTM: two models in the registry ship no confidence
  head at all, and a gate that cannot run on them is not a gate on the library.

  The two engines expose the distogram differently. AF3's head returns
  `contact_probs` already reduced (distogram_head.py returns bin_edges +
  contact_probs, and the raw logits only under return_distogram); AF2 returns
  logits and bin_edges. Reducing AF2's to the same quantity keeps one number
  comparable across all 24 rather than two that cannot be read side by side.
  """
  import jax
  import jax.numpy as jnp

  dgram = out['distogram']
  if 'contact_probs' in dgram:
    return _project(dgram['contact_probs'])
  logits, breaks = dgram['logits'], dgram['bin_edges']
  probs = jax.nn.softmax(logits, axis=-1)
  # bin i covers [breaks[i-1], breaks[i]); the last is open-ended, so a bin is
  # a contact when its UPPER edge is under 8 A.
  under8 = jnp.concatenate([breaks, jnp.array([jnp.inf])]) < 8.0
  return _project((probs * under8).sum(-1))


def make_loss_fn(model_name, seq, model_dir=None, bfloat16='none'):
  """-> loss_fn(soft_seq), for whichever engine this model runs on."""
  import jax

  from alphafold3.model import model_registry

  spec = model_registry.get(model_name)
  if spec.engine == 'af2':
    from alphafold3.af2 import inference as af2_inference
    from alphafold3.common import folding_input
    from alphafold3.constants import decoded_ccd
    from alphafold3.data import featurisation
    from alphafold3.model import feat_batch

    fold_input = folding_input.Input(
        name='grad_check', rng_seeds=[0],
        chains=[folding_input.ProteinChain(
            id='A', sequence=seq, ptms=[], unpaired_msa='', paired_msa='',
            templates=[])])
    batch = feat_batch.Batch.from_data_dict(featurisation.featurise_input(
        fold_input=fold_input, ccd=decoded_ccd.get_ccd(), buckets=None)[0])
    runner = af2_inference.AF2ModelRunner(
        spec, device=jax.devices()[0],
        model_dir=os.path.expanduser(model_dir or '~'),
        num_recycles=0, use_bfloat16=False,
        # nearest_neighbor_clusters is a STRAIGHT-THROUGH estimator (hard
        # one-hot forward, identity backward). Off here so the check measures
        # the gradient rather than an estimator's bias -- and off for design
        # generally, since clustering a single sequence is meaningless.
        use_cluster_profile=False)

    def loss_fn(x):
      return _contact_loss(
          runner.forward(batch, soft_seq=x, key=jax.random.PRNGKey(0)))

    return loss_fn

  # ...the AF3 graph.
  import haiku as hk

  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp
  from dev.oracles.fold_check import _fold_setup

  # The SAME batch and config a fold gets, per-model featurisation conventions
  # included -- a gradient measured on a differently-built batch is a gradient
  # of a different model.
  batch, cfg, model_dir = _fold_setup(model_name, seq, model_dir)
  # num_recycles=0. AF3's default is 10, and the gradient backprops through
  # EVERY trunk pass (model.py:750), so the backward pass asked for 199 GiB of
  # activations on a 68-residue chain and XLA could not remat its way out.
  # A design loop makes the same choice for the same reason.
  cfg.num_recycles = 0
  # block_remat is NOT set here on purpose: the library's default is now True
  # (evoformer.PairformerConfig), and this harness relying on the default is
  # what keeps that default honest.
  # ...and truncate the MSA. The featuriser pads to a bucket -- 16384 rows even
  # for a single sequence -- and the evoformer's own default keeps 1024 of them.
  # For single-sequence design 1 is the real depth, and this is most of the
  # memory at design sizes.
  cfg.evoformer.num_msa = 1
  # ...and float32. The trunk defaults to bfloat16 ('all'), whose ~3e-3 relative
  # precision is coarse next to the differences a finite-difference check takes,
  # and it showed: openfold3's ratios swung 1.225 then 0.924 over one step of
  # eps, which is not how a smooth function behaves.
  cfg.global_config.bfloat16 = bfloat16

  @hk.transform
  def forward(b, soft_seq):
    # structure=False leaves the diffusion sampler and the confidence head out
    # of the graph: it is the design path, and the only one that fits in memory
    # with a backward pass attached.
    return af3_model.Model(cfg)(b, soft_seq=soft_seq, structure=False)

  weights = afp.get_model_haiku_params(model_dir=model_dir)

  def loss_fn(x):
    return _contact_loss(
        forward.apply(weights, jax.random.PRNGKey(0), batch, x))

  return loss_fn


def check(model_name, seq, model_dir=None, tolerance=0.10, bfloat16=None):
  import jax

  rng = np.random.default_rng(0)
  x = jax.numpy.asarray(_blurred_one_hot(seq, rng))

  # float32 for both engines. An earlier version forced AF3 to bfloat16 because
  # the backward pass would not fit -- but that was this harness failing to turn
  # on block_remat, not a property of the models. Precision is still chosen up
  # front rather than by retrying: a failed allocation does not release its
  # buffers, so an in-process fallback OOMs at 54 MiB on a card that just freed
  # 20 GiB.
  bf16 = bfloat16 or 'none'
  loss_fn = make_loss_fn(model_name, seq, model_dir, bfloat16=bf16)
  value, grad = jax.value_and_grad(loss_fn)(x)
  # Force it here: JAX dispatch is asynchronous, so without this an error
  # surfaces wherever the value is first used, pointing at an unrelated line.
  value, grad = jax.block_until_ready((value, grad))
  print('  precision %s' % ('float32' if bf16 == 'none' else 'bfloat16'))
  grad = np.asarray(grad)
  finite = bool(np.isfinite(grad).all())
  nonzero = int((grad != 0).sum())
  print('  loss %.6f   grad: finite %s  nonzero %d/%d  max|g| %.3e'
        % (float(value), finite, nonzero, grad.size, np.abs(grad).max()))
  if not finite or nonzero == 0:
    print('  GATE FAIL: gradient is not finite and non-zero')
    return False

  # Probe ALONG the gradient. In 1360 dimensions a random direction is nearly
  # orthogonal to it -- |g.d| ~ |g|/sqrt(1360) -- which shrinks the central
  # difference to a few hundred ULP and measures rounding, not slope. Along g
  # the directional derivative is |g| itself, the largest it can be, so this is
  # the best-conditioned probe available. It still tests the gradient: the
  # finite difference is taken on the true function, and only the comparison
  # value comes from g.
  norm = float(np.linalg.norm(grad))
  if norm == 0:
    print('  GATE FAIL: gradient is identically zero')
    return False
  d = jax.numpy.asarray(grad / norm)
  analytic = norm

  ratios = {}
  for eps in _ALL_EPS:
    fd = float((loss_fn(x + eps * d) - loss_fn(x - eps * d)) / (2 * eps))
    ratios[eps] = fd / analytic
  print('  |grad| %.3e;  fd/|grad| by eps: %s'
        % (analytic, '  '.join('%.0e=%.3f' % (e, ratios[e]) for e in _ALL_EPS)))

  best_eps = min(ratios, key=lambda e: abs(ratios[e] - 1.0))
  best = abs(ratios[best_eps] - 1.0)
  ok = best <= tolerance
  print('  GATE %s: best |fd/|grad| - 1| = %.3f at eps %.0e (tolerance %.2f)'
        % ('PASS' if ok else 'FAIL', best, best_eps, tolerance))
  return ok


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('models', nargs='+')
  ap.add_argument('--target', default='~/6MRR.pdb')
  ap.add_argument('--model_dir', default=None)
  ap.add_argument('--tolerance', type=float, default=0.10)
  ap.add_argument('--bfloat16', default=None,
                  choices=('none', 'all', 'intermediate'),
                  help='override the per-engine default')
  args = ap.parse_args(argv)

  from converters.pdb import parse_ca

  seq, _ = parse_ca(os.path.expanduser(args.target))
  results = {}
  for name in args.models:
    print('%s (%d residues)' % (name, len(seq)))
    try:
      results[name] = check(name, seq, args.model_dir, args.tolerance,
                            bfloat16=args.bfloat16)
    except Exception as exc:                     # keep the sweep going
      print('  ERROR %s: %s' % (type(exc).__name__, str(exc)[:200]))
      results[name] = False
  print()
  print('%d/%d differentiable: %s' % (
      sum(results.values()), len(results),
      ' '.join('%s=%s' % (k, 'ok' if v else 'FAIL') for k, v in results.items())))
  return 0 if all(results.values()) else 1


if __name__ == '__main__':
  raise SystemExit(main())
