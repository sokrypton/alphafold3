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


def template_multimer_gate(n=24, seed=0):
  """The MULTIMER template embedder, against modules_multimer's own."""
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import config as o_config
  from alphafold.model import modules_multimer as o_mm

  from alphafold3.af2.model import modules as our_modules
  from alphafold3.af2.model import config as our_config
  from alphafold3.af2.runner import load_params

  ckpt = os.path.expanduser(
      os.environ.get('AF2_MULTIMER_PARAMS',
                     '~/params/params_model_1_multimer_v3.npz'))
  params = np.load(ckpt, allow_pickle=True)
  p_orig = _subtree(params, 'template_embedding')
  if not p_orig:
    raise SystemExit('no template_embedding params in %s' % ckpt)
  loaded = load_params([os.path.basename(ckpt)[len('params_'):-len('.npz')]],
                       os.path.dirname(ckpt), use_templates=True,
                       use_multimer=True)[0]
  pre = 'alphafold/alphafold_iteration/evoformer/'
  p_ours = {k[len(pre):]: v for k, v in loaded.items()
            if k.startswith(pre + 'template_embedding')}

  rng = np.random.default_rng(seed)
  c_z = 128
  query = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)
  pad2d = np.ones((n, n), np.float32)
  # two chains, so the multichain mask is not trivially all ones
  asym = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
  multichain = (asym[:, None] == asym[None, :]).astype(np.float32)
  batch = {
      'template_aatype': rng.integers(0, 20, size=(2, n)).astype(np.int32),
      'template_all_atom_positions': (rng.normal(size=(2, n, 37, 3)) * 5
                                      ).astype(np.float32),
      'template_all_atom_mask': (rng.random((2, n, 37)) > 0.3).astype(np.float32),
      'template_mask': np.ones(2, np.float32),
  }
  if os.environ.get('FULLMASK'):
    batch['template_all_atom_mask'] = np.ones_like(
        batch['template_all_atom_mask'])

  o_cfg = o_config.model_config('model_1_multimer_v3')
  o_tcfg = o_cfg.model.embeddings_and_evoformer.template
  o_gc = o_cfg.model.global_config

  def o_fwd():
    return o_mm.TemplateEmbedding(o_tcfg, o_gc)(
        query_embedding=jnp.asarray(query),
        template_batch={k: jnp.asarray(v) for k, v in batch.items()},
        padding_mask_2d=jnp.asarray(pad2d),
        multichain_mask_2d=jnp.asarray(multichain),
        is_training=False)

  o_f = hk.transform(o_fwd)
  ref = np.asarray(o_f.apply(p_orig, jax.random.PRNGKey(0)))

  u_cfg = our_config.model_config('model_1_multimer_v3')
  u_tcfg = u_cfg.model.embeddings_and_evoformer.template
  u_gc = u_cfg.model.global_config

  def u_fwd():
    return our_modules.TemplateEmbedding(u_tcfg, u_gc)(
        query_embedding=jnp.asarray(query),
        template_batch={k: jnp.asarray(v) for k, v in batch.items()},
        padding_mask_2d=jnp.asarray(pad2d),
        multichain_mask_2d=jnp.asarray(multichain),
        use_dropout=False)

  u_f = hk.transform(u_fwd)
  got = np.asarray(u_f.apply(p_ours, jax.random.PRNGKey(0)))
  print('multimer template embedder, %d residues (2 chains), 2 templates:' % n)
  return 0 if _cmp('template_embed', got, ref) else 1


def template_1d_gate(n=24, seed=0):
  """`template_embedding_1d` -- the MULTIMER 1-D features appended to the MSA.

  A plain function rather than a module, but it is the other half of the
  multimer template path and it reads `template_aatype` through a chi-angle
  table, so a relabelled alphabet would show here and nowhere else.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import config as o_config
  from alphafold.model import modules_multimer as o_mm

  from alphafold3.af2.model import modules as our_modules
  from alphafold3.af2.model import config as our_config

  rng = np.random.default_rng(seed)
  batch = {
      'template_aatype': rng.integers(0, 20, size=(2, n)).astype(np.int32),
      'template_all_atom_positions': (rng.normal(size=(2, n, 37, 3)) * 5
                                      ).astype(np.float32),
      'template_all_atom_mask': (rng.random((2, n, 37)) > 0.3).astype(np.float32),
  }
  o_gc = o_config.model_config('model_1_multimer_v3').model.global_config
  u_gc = our_config.model_config('model_1_multimer_v3').model.global_config

  # it builds its own two Linears, so it needs their weights
  ckpt = os.path.expanduser(
      os.environ.get('AF2_MULTIMER_PARAMS',
                     '~/params/params_model_1_multimer_v3.npz'))
  raw = np.load(ckpt, allow_pickle=True)
  p1d = {}
  for scope in ('template_single_embedding', 'template_projection'):
    p1d.update(_subtree(raw, scope))
  if not p1d:
    raise SystemExit('no 1-D template params in %s' % ckpt)

  def run(fn, gc):
    def fwd():
      f, m = fn(batch={k: jnp.asarray(v) for k, v in batch.items()},
                num_channel=256, global_config=gc)
      return f, m
    t = hk.transform(fwd)
    return jax.tree.map(np.asarray, t.apply(p1d, jax.random.PRNGKey(0)))

  ref_f, ref_m = run(o_mm.template_embedding_1d, o_gc)
  got_f, got_m = run(our_modules.template_embedding_1d, u_gc)
  print('multimer template_embedding_1d, %d residues, 2 templates:' % n)
  ok = _cmp('features', got_f, ref_f)
  ok &= _cmp('mask', got_m, ref_m)
  return 0 if ok else 1


def evoformer_gate(n=24, n_seq=8, seed=0, extra=False):
  """One EvoformerIteration -- the trunk block, and most of the parameters.

  Run on the MONOMER config and parameters: our graph runs the multimer
  structure with converted weights, so this asks whether one block of it still
  computes what DeepMind's does.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import config as o_config
  from alphafold.model import modules as o_modules

  from alphafold3.af2.model import modules as our_modules
  from alphafold3.af2.model import config as our_config
  from alphafold3.af2.runner import load_params

  scope = 'extra_msa_stack' if extra else 'evoformer_iteration'
  params = np.load(PARAMS, allow_pickle=True)
  p_orig = _subtree(params, scope)
  if not p_orig:
    raise SystemExit('no %s params in %s' % (scope, PARAMS))
  # the checkpoint stacks the blocks; take the first
  p_orig = {k.replace('__layer_stack_no_state/', ''):
            {kk: vv[0] for kk, vv in v.items()}
            for k, v in p_orig.items()}

  loaded = load_params([os.path.basename(PARAMS)[len('params_'):-len('.npz')]],
                       os.path.dirname(PARAMS), use_templates=True)[0]
  pre = 'alphafold/alphafold_iteration/evoformer/'
  # Both sides take BLOCK 0 of the stack: the checkpoint stores the 48 trunk
  # blocks stacked on axis 0, and one block is the unit being compared.
  p_ours = {k[len(pre):].replace('__layer_stack_no_state/', ''):
            {a: b[0] for a, b in v.items()}
            for k, v in loaded.items() if k.startswith(pre + scope)}
  rng = np.random.default_rng(seed)
  o_cfg = o_config.model_config('model_1_ptm')
  c_m = (o_cfg.model.embeddings_and_evoformer.extra_msa_channel if extra
         else o_cfg.model.embeddings_and_evoformer.msa_channel)
  c_z = o_cfg.model.embeddings_and_evoformer.pair_channel
  act = {'msa': (rng.normal(size=(n_seq, n, c_m)) * 0.5).astype(np.float32),
         'pair': (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)}
  masks = {'msa': np.ones((n_seq, n), np.float32),
           'pair': np.ones((n, n), np.float32)}
  o_ecfg = o_cfg.model.embeddings_and_evoformer.evoformer
  o_gc = o_cfg.model.global_config

  def o_fwd():
    return o_modules.EvoformerIteration(o_ecfg, o_gc, is_extra_msa=extra,
                                        name=scope)(
        {k: jnp.asarray(v) for k, v in act.items()},
        {k: jnp.asarray(v) for k, v in masks.items()},
        is_training=False)

  ref = jax.tree.map(np.asarray,
                     hk.transform(o_fwd).apply(p_orig, jax.random.PRNGKey(0)))

  u_cfg = our_config.model_config('model_1_ptm')
  u_ecfg = u_cfg.model.embeddings_and_evoformer.evoformer
  u_gc = u_cfg.model.global_config

  def u_fwd():
    return our_modules.EvoformerIteration(u_ecfg, u_gc, is_extra_msa=extra,
                                          name=scope)(
        {k: jnp.asarray(v) for k, v in act.items()},
        {k: jnp.asarray(v) for k, v in masks.items()} |
        {'opm_first': jnp.float32(0.0)},
        use_dropout=False)

  got = jax.tree.map(np.asarray,
                     hk.transform(u_fwd).apply(p_ours, jax.random.PRNGKey(0)))
  print('%s, %d residues x %d sequences:'
        % ('extra_msa_stack' if extra else 'evoformer_iteration', n, n_seq))
  ok = _cmp('msa', got['msa'], ref['msa'])
  ok &= _cmp('pair', got['pair'], ref['pair'])
  return 0 if ok else 1


def ipa_gate(n=24, seed=0):
  """InvariantPointAttention -- the structure module's core.

  Compared against folding_multimer's, because that is the one this package
  runs: the monomer structure module was retired with the monomer graph and its
  weights are converted (convert.py splits the fused q/kv scalar and point
  projections), so this asks whether the split is faithful.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import config as o_config
  from alphafold.model import folding_multimer as o_folding
  from alphafold.model import geometry as o_geometry

  from alphafold3.af2.model import folding as our_folding
  from alphafold3.af2.model import geometry as our_geometry
  from alphafold3.af2.model import config as our_config
  from alphafold3.af2.runner import load_params

  ckpt = os.path.expanduser(
      os.environ.get('AF2_MULTIMER_PARAMS',
                     '~/params/params_model_1_multimer_v3.npz'))
  raw = np.load(ckpt, allow_pickle=True)
  pre = 'alphafold/alphafold_iteration/structure_module/fold_iteration/'
  p = {}
  for k in raw.files:
    if not k.startswith(pre + 'invariant_point_attention'):
      continue
    mod, leaf = k.rsplit('//', 1)
    p.setdefault(mod[len(pre):], {})[leaf] = np.asarray(raw[k])
  if not p:
    raise SystemExit('no invariant_point_attention params in %s' % ckpt)
  # OUR scalar projections always carry a bias so one graph serves monomer and
  # multimer; native multimer has none, and the loader supplies zeros (adding 0
  # is a no-op). So each side gets the parameters its own graph declares --
  # the same arrangement as the template gate.
  loaded = load_params([os.path.basename(ckpt)[len('params_'):-len('.npz')]],
                       os.path.dirname(ckpt), use_multimer=True)[0]
  p_ours = {k[len(pre):]: v for k, v in loaded.items()
            if k.startswith(pre + 'invariant_point_attention')}

  rng = np.random.default_rng(seed)
  o_cfg = o_config.model_config('model_1_multimer_v3')
  c_s = o_cfg.model.heads.structure_module.num_channel
  c_z = o_cfg.model.embeddings_and_evoformer.pair_channel
  s1d = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  s2d = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)
  mask = np.ones((n, 1), np.float32)
  trans = (rng.normal(size=(n, 3)) * 5).astype(np.float32)
  rot = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))

  def mk(geo):
    r = geo.Rot3Array.from_array(jnp.asarray(rot))
    t = geo.Vec3Array.from_array(jnp.asarray(trans))
    return geo.Rigid3Array(r, t)

  def o_fwd():
    return o_folding.InvariantPointAttention(
        o_cfg.model.heads.structure_module, o_cfg.model.global_config)(
            jnp.asarray(s1d), jnp.asarray(s2d), jnp.asarray(mask),
            mk(o_geometry))

  ref = np.asarray(hk.transform(o_fwd).apply(p, jax.random.PRNGKey(0)))

  u_cfg = our_config.model_config('model_1_multimer_v3')

  def u_fwd():
    return our_folding.InvariantPointAttention(
        u_cfg.model.heads.structure_module, u_cfg.model.global_config)(
            jnp.asarray(s1d), jnp.asarray(s2d), jnp.asarray(mask),
            mk(our_geometry))

  got = np.asarray(hk.transform(u_fwd).apply(p_ours, jax.random.PRNGKey(0)))
  print('invariant point attention, %d residues:' % n)
  return 0 if _cmp('ipa', got, ref) else 1


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('module', nargs='?', default='template',
                  choices=('template', 'template_multimer', 'template_1d',
                           'evoformer', 'extra_msa', 'ipa'))
  ap.add_argument('--tokens', type=int, default=24)
  args = ap.parse_args(argv)
  if not os.path.isdir(AF2_ORIGINAL):
    raise SystemExit('clone the original first: git clone '
                     'https://github.com/google-deepmind/alphafold %s'
                     % AF2_ORIGINAL)
  if args.module == 'ipa':
    return ipa_gate(n=args.tokens)
  if args.module in ('evoformer', 'extra_msa'):
    return evoformer_gate(n=args.tokens, extra=(args.module == 'extra_msa'))
  if args.module == 'template_1d':
    return template_1d_gate(n=args.tokens)
  if args.module == 'template_multimer':
    return template_multimer_gate(n=args.tokens)
  return template_gate(n=args.tokens)


if __name__ == '__main__':
  raise SystemExit(main())
