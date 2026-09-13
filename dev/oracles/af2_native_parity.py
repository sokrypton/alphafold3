"""Compare this package's AlphaFold 2 modules against DEEPMIND'S OWN CODE.

Every other AF2 check here is a known-answer test against colabdesign2 -- which
is the same lineage (DeepMind -> ColabDesign v1 -> colabdesign2 -> here), so it
proves the copy was not broken in transit and NOTHING about whether the copy
matches the original. This is the missing direction: the original repo, its own
config, its own parameters, identical inputs.

  git clone https://github.com/google-deepmind/alphafold ~/af2_original
  PYTHONPATH=src:.:dev/oracles:/home/ubuntu/af2_original \
      python dev/oracles/af2_native_parity.py template

Gates (`--module`):
  template   the MONOMER template embedder -- vendored back from ColabDesign v1
             when templates were re-enabled, and the piece with no prior
             coverage against the original at all
  evoformer  one EvoformerIteration
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AF2_ORIGINAL = os.environ.get('AF2_ORIGINAL', '/home/ubuntu/af2_original')
PARAMS = os.path.expanduser(os.environ.get('AF2_PARAMS',
                                           '~/params/params_model_1_ptm.npz'))
PREFIX = 'alphafold/alphafold_iteration/evoformer/'


def _cmp(tag, got, ref):
  got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
  if got.shape != ref.shape:
    print('  %-18s SHAPE ours %s original %s' % (tag, got.shape, ref.shape))
    return False
  d = np.abs(got - ref)
  rms = float(np.sqrt((ref ** 2).mean()))
  corr = float(np.corrcoef(got.reshape(-1), ref.reshape(-1))[0, 1])
  print('  %-18s corr %.8f  max|d| %.3e  mean|d| %.3e  rms(orig) %.4f'
        % (tag, corr, d.max(), d.mean(), rms))
  return corr > 0.9999 and d.max() <= max(1e-3, 1e-3 * rms)


def _subtree(params, scope):
  """checkpoint arrays under `scope` -> a haiku param dict for a bare module."""
  out = {}
  for k in params.files:
    if not k.startswith(PREFIX + scope):
      continue
    mod, leaf = k.rsplit('//', 1)
    out.setdefault(mod[len(PREFIX):], {})[leaf] = np.asarray(params[k])
  return out


def template_gate(n=24, seed=0):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import config as o_config
  from alphafold.model import modules as o_modules

  from alphafold3.af2.model import modules as our_modules
  from alphafold3.af2.model import config as our_config

  # TWO PARAMETER SETS FOR ONE COMPARISON, deliberately. The original runs the
  # UNFUSED TriangleMultiplication and reads the checkpoint as shipped; this
  # package runs the FUSED variant, and its loader concatenates the projections
  # (left/right -> `projection` at 2x64, and `layer_norm_input` -> the fused
  # `left_norm_input`). The fusion is supposed to be an algebraic regrouping, so
  # feeding each side the parameters it expects is the comparison -- and whether
  # the regrouping is faithful is precisely what has never been checked.
  params = np.load(PARAMS, allow_pickle=True)
  p_orig = _subtree(params, 'template_embedding')
  if not p_orig:
    raise SystemExit('no template_embedding params in %s' % PARAMS)

  from alphafold3.af2.runner import load_params
  loaded = load_params([os.path.basename(PARAMS)[len('params_'):-len('.npz')]],
                       os.path.dirname(PARAMS), use_templates=True)[0]
  pre = 'alphafold/alphafold_iteration/evoformer/'
  p_ours = {k[len(pre):]: v for k, v in loaded.items()
            if k.startswith(pre + 'template_embedding')}

  rng = np.random.default_rng(seed)
  c_z = 128
  query = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)
  mask_2d = np.ones((n, n), np.float32)
  aatype = rng.integers(0, 20, size=(2, n)).astype(np.int32)
  pos = (rng.normal(size=(2, n, 37, 3)) * 5).astype(np.float32)
  atom_mask = (rng.random((2, n, 37)) > 0.3).astype(np.float32)
  if os.environ.get('FULLMASK'):
    # every atom present: the masks stop distinguishing the implementations, so
    # what is left is the arithmetic
    atom_mask = np.ones_like(atom_mask)
  pb = (rng.normal(size=(2, n, 3)) * 5).astype(np.float32)
  pb_mask = (rng.random((2, n)) > 0.2).astype(np.float32)
  if os.environ.get('FULLMASK'):
    pb_mask = np.ones_like(pb_mask)
  t_mask = np.ones(2, np.float32)
  batch = {'template_aatype': aatype, 'template_all_atom_positions': pos,
           'template_all_atom_mask': atom_mask, 'template_pseudo_beta': pb,
           'template_pseudo_beta_mask': pb_mask, 'template_mask': t_mask,
           # THE ORIGINAL SPELLS IT PLURAL. `template_all_atom_masks` is what
           # DeepMind's pipeline emits and what its modules read; somewhere down
           # the ColabDesign lineage it became singular. Both are supplied here
           # so the difference cannot silently decide the comparison.
           'template_all_atom_masks': atom_mask}

  o_cfg = o_config.model_config('model_1_ptm')
  o_tcfg = o_cfg.model.embeddings_and_evoformer.template
  o_gc = o_cfg.model.global_config

  def o_fwd():
    return o_modules.TemplateEmbedding(o_tcfg, o_gc)(
        jnp.asarray(query), {k: jnp.asarray(v) for k, v in batch.items()},
        jnp.asarray(mask_2d), is_training=False)

  o_f = hk.transform(o_fwd)
  ref = np.asarray(o_f.apply(p_orig, jax.random.PRNGKey(0)))

  u_cfg = our_config.model_config('model_1_ptm')
  u_tcfg = u_cfg.model.embeddings_and_evoformer.template
  u_gc = u_cfg.model.global_config
  # UNFUSE=1 runs our TriangleMultiplication on the original's arithmetic with
  # the original's parameters, which separates "the fusion is not equivalent"
  # from "the module differs elsewhere".
  if os.environ.get('UNFUSE'):
    for side in ('triangle_multiplication_outgoing',
                 'triangle_multiplication_incoming'):
      u_tcfg.template_pair_stack[side].fuse_projection_weights = False
    p_ours = p_orig

  def u_fwd():
    return our_modules.MonomerTemplateEmbedding(u_tcfg, u_gc)(
        query_embedding=jnp.asarray(query),
        template_batch={k: jnp.asarray(v) for k, v in batch.items()},
        mask_2d=jnp.asarray(mask_2d),
        multichain_mask_2d=jnp.ones((n, n), np.float32),
        use_dropout=False)

  u_f = hk.transform(u_fwd)
  got = np.asarray(u_f.apply(p_ours, jax.random.PRNGKey(0)))
  print('monomer template embedder, %d residues, 2 templates:' % n)
  return 0 if _cmp('template_embed', got, ref) else 1


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('module', nargs='?', default='template',
                  choices=('template',))
  ap.add_argument('--tokens', type=int, default=24)
  args = ap.parse_args(argv)
  if not os.path.isdir(AF2_ORIGINAL):
    raise SystemExit('clone the original first: git clone '
                     'https://github.com/google-deepmind/alphafold %s'
                     % AF2_ORIGINAL)
  return template_gate(n=args.tokens)


if __name__ == '__main__':
  raise SystemExit(main())
