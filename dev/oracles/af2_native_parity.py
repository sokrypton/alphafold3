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

# The two paths. `params` is what each side's checkpoint is called, `config` the
# config name, and the original's modules live in different files -- monomer in
# modules.py/folding.py, multimer in modules_multimer.py/folding_multimer.py.
# THIS PACKAGE HAS ONE GRAPH for both (the multimer structure, with monomer
# weights converted at load), so for the monomer variant the comparison is
# genuinely cross-architecture and is exactly what tests convert.py.
VARIANTS = {
    'monomer': dict(params='~/params/params_model_1_ptm.npz',
                    config='model_1_ptm'),
    'multimer': dict(params='~/params/params_model_1_multimer_v3.npz',
                     config='model_1_multimer_v3'),
}


def _jax_compat():
  """Let DeepMind's 2021 code run on this JAX.

  `jnp.clip(x, a_min=, a_max=)` was removed; the original's
  `_relative_encoding` still calls it that way. The wrapper only TRANSLATES the
  two retired keyword names -- it does not change any value, and our own code
  never passes them, so it is transparent to our side of every comparison.
  Scoped to this oracle's process.
  """
  import jax.numpy as jnp
  if getattr(jnp.clip, '_af2_compat', False):
    return
  _clip = jnp.clip

  def clip(x, a_min=None, a_max=None, *a, **kw):
    if a or kw:
      return _clip(x, *a, **kw) if a_min is None and a_max is None else _clip(
          x, a_min, a_max)
    return _clip(x, a_min, a_max)

  clip._af2_compat = True
  jnp.clip = clip


def _variant(name):
  v = VARIANTS[name]
  return os.path.expanduser(v['params']), v['config']


def _load_ours(ckpt, use_templates=True):
  """params as THIS PACKAGE's loader produces them (fused, bias-normalised)."""
  from alphafold3.af2.runner import load_params
  p = load_params([os.path.basename(ckpt)[len('params_'):-len('.npz')]],
                  os.path.dirname(ckpt), use_templates=use_templates,
                  use_multimer=('multimer' in ckpt))[0]
  if 'multimer' not in ckpt:
    # AND CONVERTED, as the runner does before using them: this package has one
    # graph (the multimer one) and a monomer checkpoint is remapped onto it at
    # load -- fused q/kv scalar and point projections split, pair_activiations
    # and the single-side linears regrouped. Comparing without this step asks
    # our graph for parameters that only exist after it.
    from alphafold3.af2.convert import convert_monomer_params
    p = convert_monomer_params(p)
  return p


def _scope(params, prefix, scope, unstack=False):
  """arrays under prefix+scope -> a haiku param dict for a bare module.

  Takes either an .npz (keys are 'module//leaf') or this package's loader output
  (already {module: {leaf: array}}). An .npz also exposes `.items()`, so the two
  are told apart by `.files`, not by duck typing.
  """
  out = {}
  if hasattr(params, 'files'):
    for k in params.files:
      if not k.startswith(prefix + scope):
        continue
      mod, leaf = k.rsplit('//', 1)
      # the layer_stack prefix is part of the scope and is only removed when a
      # single block is being pulled OUT of the stack
      key = mod[len(prefix):]
      if unstack:
        key = key.replace('__layer_stack_no_state/', '')
      arr = np.asarray(params[k])
      out.setdefault(key, {})[leaf] = arr[0] if unstack else arr
  else:
    for k, v in params.items():
      if not k.startswith(prefix + scope):
        continue
      key = k[len(prefix):]
      if unstack:
        key = key.replace('__layer_stack_no_state/', '')
      out[key] = {a: (b[0] if unstack else b) for a, b in v.items()}
  return out


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
  p_orig = _scope(params, PREFIX, 'template_embedding')
  if not p_orig:
    raise SystemExit('no template_embedding params in %s' % PARAMS)
  p_ours = _scope(_load_ours(PARAMS), PREFIX, 'template_embedding')

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


def evoformer_gate(n=24, n_seq=8, seed=0, extra=False, variant='monomer'):
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
  ckpt, cfg_name = _variant(variant)
  raw = np.load(ckpt, allow_pickle=True)
  # BLOCK 0 of the stack on both sides, through the same helper -- an earlier
  # hand-rolled slice here disagreed with it and made the multimer block read
  # corr 0.01, which was the gate and not the port.
  p_orig = _scope(raw, PREFIX, scope, unstack=True)
  p_ours = _scope(_load_ours(ckpt), PREFIX, scope, unstack=True)
  if not p_orig:
    raise SystemExit('no %s params in %s' % (scope, ckpt))

  rng = np.random.default_rng(seed)
  o_cfg = o_config.model_config(cfg_name)
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

  u_cfg = our_config.model_config(cfg_name)
  u_ecfg = u_cfg.model.embeddings_and_evoformer.evoformer
  u_gc = u_cfg.model.global_config

  def u_fwd():
    return our_modules.EvoformerIteration(u_ecfg, u_gc, is_extra_msa=extra,
                                          name=scope)(
        {k: jnp.asarray(v) for k, v in act.items()},
        {k: jnp.asarray(v) for k, v in masks.items()} |
        # OPM ORDER IS A REGIME, NOT A CONSTANT. This package injects it as a
        # runtime scalar (the monomer runs the outer-product-mean AFTER the row
        # /column attention, native multimer runs it FIRST), so it has to come
        # from the config being compared -- hardcoding the monomer value made
        # the multimer Evoformer read corr 0.01 and look like a broken port.
        {'opm_first': jnp.float32(
            bool(o_ecfg.outer_product_mean.get('first', False)))},
        use_dropout=False)

  got = jax.tree.map(np.asarray,
                     hk.transform(u_fwd).apply(p_ours, jax.random.PRNGKey(0)))
  print('%s [%s], %d residues x %d sequences:'
        % (scope, variant, n, n_seq))
  ok = _cmp('msa', got['msa'], ref['msa'])
  ok &= _cmp('pair', got['pair'], ref['pair'])
  return 0 if ok else 1


def ipa_gate(n=24, seed=0, variant='multimer'):
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
  # the ORIGINAL has two structure modules; this package has one (the multimer
  # one), so on the monomer variant this compares across architectures -- which
  # is precisely what convert.py's fused q/kv split has to get right.
  if variant == 'multimer':
    from alphafold.model import folding_multimer as o_folding
  else:
    from alphafold.model import folding as o_folding
  from alphafold.model import geometry as o_geometry

  from alphafold3.af2.model import folding as our_folding
  from alphafold3.af2.model import geometry as our_geometry
  from alphafold3.af2.model import config as our_config
  from alphafold3.af2.runner import load_params

  ckpt, cfg_name = _variant(variant)
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
  loaded = _load_ours(ckpt)
  p_ours = {k[len(pre):]: v for k, v in loaded.items()
            if k.startswith(pre + 'invariant_point_attention')}

  rng = np.random.default_rng(seed)
  o_cfg = o_config.model_config(cfg_name)
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
    if variant == 'multimer':
      rigid = mk(o_geometry)
    else:
      # THE MONOMER ORIGINAL TAKES A QuatAffine, not a Rigid3Array -- a
      # different geometry API for the same transform. Built here from the same
      # rotation and translation so the two sides are posed identically; this
      # is the bridge that lets convert.py's fused q/kv split be compared at all.
      from alphafold.model import quat_affine as o_quat
      rigid = o_quat.QuatAffine(
          quaternion=o_quat.rot_to_quat(jnp.asarray(rot), unstack_inputs=True),
          translation=[jnp.asarray(trans[:, i]) for i in range(3)],
          rotation=[[jnp.asarray(rot[:, i, j]) for j in range(3)]
                    for i in range(3)],
          unstack_inputs=False)
    return o_folding.InvariantPointAttention(
        o_cfg.model.heads.structure_module, o_cfg.model.global_config)(
            jnp.asarray(s1d), jnp.asarray(s2d), jnp.asarray(mask), rigid)

  ref = np.asarray(hk.transform(o_fwd).apply(p, jax.random.PRNGKey(0)))

  u_cfg = our_config.model_config(cfg_name)

  def u_fwd():
    return our_folding.InvariantPointAttention(
        u_cfg.model.heads.structure_module, u_cfg.model.global_config)(
            jnp.asarray(s1d), jnp.asarray(s2d), jnp.asarray(mask),
            mk(our_geometry))

  got = np.asarray(hk.transform(u_fwd).apply(p_ours, jax.random.PRNGKey(0)))
  print('invariant point attention [%s], %d residues:' % (variant, n))
  return 0 if _cmp('ipa', got, ref) else 1


def heads_gate(variant='monomer', n=24, seed=0):
  """The five output heads. They live in modules.py on BOTH paths -- multimer
  reuses them -- so the variant only changes the checkpoint and the config."""
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import config as o_config
  from alphafold.model import modules as o_modules

  from alphafold3.af2.model import modules as our_modules
  from alphafold3.af2.model import config as our_config

  ckpt, cfg_name = _variant(variant)
  raw = np.load(ckpt, allow_pickle=True)
  # RAW ON BOTH SIDES. The heads are not transformed by convert.py -- with one
  # exception, `masked_msa_head`, whose 23 columns it truncates to 22 to fit the
  # multimer graph's alphabet. That head is TRAINING-ONLY and inference never
  # builds it, so converting here would compare a head the runtime never uses
  # against parameters reshaped for a graph it never reaches.
  ours_p = raw
  o_cfg = o_config.model_config(cfg_name)
  u_cfg = our_config.model_config(cfg_name)
  o_gc, u_gc = o_cfg.model.global_config, u_cfg.model.global_config

  rng = np.random.default_rng(seed)
  c_s = o_cfg.model.embeddings_and_evoformer.seq_channel
  c_z = o_cfg.model.embeddings_and_evoformer.pair_channel
  c_m = o_cfg.model.embeddings_and_evoformer.msa_channel
  reps = {
      'single': (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32),
      'pair': (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32),
      'msa': (rng.normal(size=(4, n, c_m)) * 0.5).astype(np.float32),
      'structure_module': (rng.normal(size=(n, 384)) * 0.5).astype(np.float32),
  }
  batch = {'aatype': rng.integers(0, 20, size=(n,)).astype(np.int32),
           'seq_mask': np.ones(n, np.float32)}

  # The checkpoint scope is `<name>_head`, which is also the haiku module name
  # AlphaFoldIteration gives it -- so the module has to be built under that name
  # or its parameters are not found.
  HEADS = [
      ('distogram_head', 'DistogramHead', o_cfg.model.heads.distogram),
      ('predicted_lddt_head', 'PredictedLDDTHead',
       o_cfg.model.heads.predicted_lddt),
      ('experimentally_resolved_head', 'ExperimentallyResolvedHead',
       o_cfg.model.heads.experimentally_resolved),
      ('masked_msa_head', 'MaskedMsaHead', o_cfg.model.heads.masked_msa),
  ]
  if 'predicted_aligned_error' in o_cfg.model.heads:
    HEADS.append(('predicted_aligned_error_head', 'PredictedAlignedErrorHead',
                  o_cfg.model.heads.predicted_aligned_error))

  print('heads [%s], %d residues:' % (variant, n))
  ok = True
  prefix = 'alphafold/alphafold_iteration/'
  for scope, cls_name, hcfg in HEADS:
    p_orig = _scope(raw, prefix, scope)
    p_ours = _scope(ours_p, prefix, scope)
    if not p_orig:
      print('  %-24s (not in this checkpoint)' % scope)
      continue

    def mk(mod_src, cfg, gc, cls=cls_name, sc=scope):
      def fwd():
        head = getattr(mod_src, cls)(cfg, gc, name=sc)
        return head({k: jnp.asarray(v) for k, v in reps.items()},
                    {k: jnp.asarray(v) for k, v in batch.items()},
                    is_training=False)
      return fwd

    # OURS TAKES NO `is_training`: the heads have no dropout, so this package
    # dropped the argument. Same computation, different signature.
    try:
      ref = hk.transform(mk(o_modules, hcfg, o_gc)).apply(
          p_orig, jax.random.PRNGKey(0))
    except Exception as e:                                   # noqa: BLE001
      print('  %-24s original raised: %s' % (scope, str(e)[:70]))
      ok = False
      continue

    def mk_ours(cfg=hcfg, cls=cls_name, sc=scope):
      def fwd():
        head = getattr(our_modules, cls)(cfg, u_gc, name=sc)
        return head({k: jnp.asarray(v) for k, v in reps.items()},
                    {k: jnp.asarray(v) for k, v in batch.items()})
      return fwd

    got = hk.transform(mk_ours()).apply(p_ours, jax.random.PRNGKey(0))
    key = 'logits' if 'logits' in ref else sorted(ref)[0]
    ok &= _cmp(scope, got[key], ref[key])
  return 0 if ok else 1


def msa_gate(n=20, n_seq=32, max_seq=8, seed=0, variant='multimer'):
  """The MSA feature pipeline: SUBSAMPLING, the msa/extra split, and the BERT
  masking that mutates residues.

  This is not the network -- it is where AlphaFold 2's stochasticity lives, and
  it decides which sequences the trunk ever sees. Multimer moved it INSIDE the
  jax model (modules_multimer), which is the version this package runs, so it
  can be compared directly rather than through the monomer's TensorFlow
  data_transforms.

  One real difference has to be held in mind: ours carries the MSA as a ONE-HOT
  so a soft, designed MSA can carry gradients; the original carries integer
  ids. So the comparison is on what each SELECTS and PRODUCES -- row order, mask
  positions, resulting residues -- not on dtype.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import modules_multimer as o_mm
  from alphafold.model import prng as o_prng
  import ml_collections

  from alphafold3.af2.model import msa as our_msa

  rng = np.random.default_rng(seed)
  msa_i = rng.integers(0, 21, size=(n_seq, n)).astype(np.int32)
  msa_mask = (rng.random((n_seq, n)) > 0.1).astype(np.float32)
  deletion = (rng.random((n_seq, n)) * 3).astype(np.float32)
  key = jax.random.PRNGKey(seed)

  def o_batch():
    b = {'msa': jnp.asarray(msa_i), 'msa_mask': jnp.asarray(msa_mask),
         'deletion_matrix': jnp.asarray(deletion)}
    b['msa_profile'] = o_mm.make_msa_profile(b)
    return b

  def u_batch():
    b = {'msa': jax.nn.one_hot(jnp.asarray(msa_i), 22),
         'msa_mask': jnp.asarray(msa_mask),
         'deletion_matrix': jnp.asarray(deletion)}
    our_msa.make_msa_profile(b)
    return b

  print('msa pipeline, %d sequences x %d residues, sampling %d:'
        % (n_seq, n, max_seq))
  ok = True

  # --- the profile both sides build from the same alignment
  ok &= _cmp('msa_profile', np.asarray(u_batch()['msa_profile']),
             np.asarray(o_batch()['msa_profile']))

  # --- SUBSAMPLING: which rows become msa, which become extra
  ob, ub = o_batch(), u_batch()
  o_out = o_mm.sample_msa(o_prng.SafeKey(key), ob, max_seq)
  our_msa.sample_msa(key, ub, max_seq)
  ok &= _cmp('sampled msa rows', np.asarray(ub['msa']).argmax(-1),
             np.asarray(o_out['msa']))
  ok &= _cmp('extra msa rows', np.asarray(ub['extra_msa']).argmax(-1),
             np.asarray(o_out['extra_msa']))
  ok &= _cmp('extra deletion', np.asarray(ub['extra_deletion_matrix']),
             np.asarray(o_out['extra_deletion_matrix']))

  # --- MASKING / MUTATING
  # EACH PATH'S OWN MASKING CONFIG. The multimer keeps it in the MODEL config;
  # the monomer keeps it in `data.common.masked_msa` with the fraction in
  # `data.eval`, because there it is applied by the TensorFlow data pipeline
  # rather than in the model. The values happen to agree (0.1/0.1/0.1, 0.15),
  # which is worth having checked rather than assumed.
  from alphafold.model import config as o_config
  oc = o_config.model_config(_variant(variant)[1])
  if variant == 'multimer':
    mcfg = oc.model.embeddings_and_evoformer.masked_msa
    cfg = ml_collections.ConfigDict(dict(
        replace_fraction=mcfg.replace_fraction, uniform_prob=mcfg.uniform_prob,
        profile_prob=mcfg.profile_prob, same_prob=mcfg.same_prob))
  else:
    mcfg = oc.data.common.masked_msa
    cfg = ml_collections.ConfigDict(dict(
        replace_fraction=oc.data.eval.masked_msa_replace_fraction,
        uniform_prob=mcfg.uniform_prob, profile_prob=mcfg.profile_prob,
        same_prob=mcfg.same_prob))
  print('  masking config [%s]: replace %.2f uniform %.2f profile %.2f same %.2f'
        % (variant, cfg.replace_fraction, cfg.uniform_prob, cfg.profile_prob,
           cfg.same_prob))
  ob, ub = o_batch(), u_batch()
  o_out = o_mm.make_masked_msa(ob, o_prng.SafeKey(key), cfg)
  our_msa.make_masked_msa(key, ub, dict(
      replace_fraction=cfg.replace_fraction, uniform_prob=cfg.uniform_prob,
      profile_prob=cfg.profile_prob, same_prob=cfg.same_prob))
  ok &= _cmp('bert_mask', np.asarray(ub['bert_mask']),
             np.asarray(o_out['bert_mask']))
  ok &= _cmp('masked msa', np.asarray(ub['msa']).argmax(-1),
             np.asarray(o_out['msa']))

  # --- CLUSTERING of the extra sequences onto the sampled ones
  ob, ub = o_batch(), u_batch()
  o_out = o_mm.sample_msa(o_prng.SafeKey(key), ob, max_seq)
  our_msa.sample_msa(key, ub, max_seq)
  # PIPELINE ORDER: ours pads the alphabet to 23 (the BERT mask column) before
  # clustering, which make_msa_feats does between the two steps.
  our_msa.pad_msa_A(ub)
  # the original RETURNS the two arrays; ours writes them into the batch
  o_prof, o_del = o_mm.nearest_neighbor_clusters(o_out)
  our_msa.nearest_neighbor_clusters(ub)
  ok &= _cmp('cluster_profile', np.asarray(ub['cluster_profile']),
             np.asarray(o_prof))
  ok &= _cmp('cluster_deletion_mean', np.asarray(ub['cluster_deletion_mean']),
             np.asarray(o_del))
  return 0 if ok else 1


def structure_gate(n=24, seed=0, variant='multimer'):
  """The WHOLE StructureModule -- FoldIteration x8, not just the IPA inside it.

  The `ipa` cell is exact on both paths, but a stack of eight iterations is
  more than its attention: each one also runs the backbone update, the
  transition, the sidechain torsion net, and -- the part no other cell reaches
  -- carries a RIGID from one iteration to the next through
  `stop_gradient_rot`. A per-iteration term that is wrong only after an update
  is invisible to a single-module gate.

  Compared on the FINAL frames (`traj[-1]`), which is what everything
  downstream reads, plus the sidechain atom positions.

  READ THE FLOOR LINE, NOT THE CORRELATION. On random s/z the recurrence runs
  far outside its trained distribution and amplifies hard: the final frames
  read corr ~0.92 against native, and perturbing NATIVE's own single by 1e-6
  moves NATIVE further than that (corr ~0.60). So the headline number is the
  cell's resolution, not a port difference -- what this gate actually asserts is
  that our mean|d| stays BELOW the floor at every one of the eight iterations,
  which it does by 5-10x.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  from alphafold.model import config as o_config
  from alphafold.model import folding_multimer as o_folding
  from alphafold.model import prng as o_prng

  from alphafold3.af2.model import folding as our_folding
  from alphafold3.af2.model import config as our_config

  if variant != 'multimer':
    # The monomer StructureModule was retired with the monomer graph -- a
    # monomer checkpoint runs on the MULTIMER structure module with converted
    # weights, which is what `ipa` compares across architectures. Running the
    # whole stack across two different module trees would compare the
    # ARCHITECTURES, not the port, so this cell declines rather than reporting
    # a number nobody can act on.
    raise SystemExit('N/A: the monomer StructureModule is retired; its IPA is '
                     'compared across architectures by the `ipa` cell and the '
                     'stack around it is the multimer one')

  ckpt, cfg_name = _variant(variant)
  raw = np.load(ckpt, allow_pickle=True)
  # The module NAMES ITSELF `structure_module`, so the scope stays in the key
  # and only the two enclosing levels come off. Stripping the scope too is the
  # first thing tried here and it fails loudly -- haiku cannot find
  # `structure_module/single_layer_norm` -- which is the good outcome.
  pre = 'alphafold/alphafold_iteration/'
  p = _scope(raw, pre, 'structure_module/')
  if not p:
    raise SystemExit('no structure_module params in %s' % ckpt)
  p_ours = _scope(_load_ours(ckpt), pre, 'structure_module/')

  rng = np.random.default_rng(seed)
  o_cfg = o_config.model_config(cfg_name)
  sm = o_cfg.model.heads.structure_module
  c_s = o_cfg.model.embeddings_and_evoformer.seq_channel
  c_z = o_cfg.model.embeddings_and_evoformer.pair_channel
  reps = {'single': (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32),
          'pair': (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)}
  batch = {'aatype': rng.integers(0, 20, size=(n,)).astype(np.int32),
           'seq_mask': np.ones((n,), np.float32)}
  # OUR fold_iteration reads `batch['use_dropout']` -- a traced flag, where the
  # original takes `is_training`. False is the inference path on both sides.
  our_batch = dict(batch, use_dropout=np.bool_(False))
  t = lambda d: {k: jnp.asarray(v) for k, v in d.items()}
  # ONE key, both sides. The stack splits it per iteration, and at inference
  # every dropout it feeds is identity -- but the split still has to be the
  # same split or the two runs are not the same run.
  key = jax.random.PRNGKey(0)

  def o_fwd():
    # compute_loss=True ONLY so the return carries `traj` and `sidechains`;
    # with the default it drops to three keys and the trajectory -- the part
    # this cell exists to compare -- is thrown away. No loss is computed.
    return o_folding.StructureModule(sm, o_cfg.model.global_config)(
        t(reps), t(batch), is_training=False,
        safe_key=o_prng.SafeKey(jax.random.PRNGKey(0)),
        compute_loss=True)

  ref = hk.transform(o_fwd).apply(p, key)

  # THE FLOOR, measured before any number here is believed. Eight iterations of
  # a rigid-body recurrence amplify: perturb NATIVE's own single by 1e-6 and
  # run NATIVE again, and whatever that moves is what this cell cannot resolve.
  # Without it the amplification below reads as a port bug -- which is the
  # mistake `trunk_gate_resolution` records for the trunk pairformer.
  #
  # RANDOM, not a constant. `single` goes straight into a LayerNorm, which
  # subtracts the mean, so a uniform +1e-6 is removed EXACTLY: the first
  # version of this control read max|d| 0.000e+00 at every iteration and looked
  # like proof the cell had full resolution.
  reps_eps = dict(reps, single=(reps['single'] + (1e-6 * rng.normal(
      size=reps['single'].shape)).astype(np.float32)))

  def o_fwd_eps():
    return o_folding.StructureModule(sm, o_cfg.model.global_config)(
        t(reps_eps), t(batch), is_training=False,
        safe_key=o_prng.SafeKey(jax.random.PRNGKey(0)), compute_loss=True)

  floor = hk.transform(o_fwd_eps).apply(p, key)

  u_cfg = our_config.model_config(cfg_name)
  from alphafold3.af2.model import prng as our_prng

  def u_fwd():
    return our_folding.StructureModule(
        u_cfg.model.heads.structure_module, u_cfg.model.global_config)(
            t(reps), t(our_batch),
            safe_key=our_prng.SafeKey(jax.random.PRNGKey(0)))

  got = hk.transform(u_fwd).apply(p_ours, key)
  print('structure module [%s], %d residues, %d fold iterations:'
        % (variant, n, sm.num_layer))
  # PER ITERATION, always. The whole point of this cell is the carry between
  # iterations, so a single final number cannot say whether one iteration is
  # wrong or the handover is.
  nt = np.asarray(ref['traj'])
  nf = np.asarray(floor['traj'])
  resolved = True
  for i in range(nt.shape[0]):
    d = np.abs(np.asarray(got['traj'][i], np.float64) - nt[i]).mean()
    f = np.abs(nf[i] - nt[i]).mean()
    _cmp('traj[%d]' % i, got['traj'][i], nt[i])
    _cmp('  floor[%d]' % i, nf[i], nt[i])
    if d > f:
      resolved = False
      print('    ^ RESOLVED and WORSE than the floor -- a real difference')
  print('  verdict: our mean|d| is %s the 1e-6 floor at every iteration'
        % ('BELOW' if resolved else 'NOT below'))
  # These four are printed, not asserted. `_cmp`'s corr > 0.9999 is the right
  # bar for a single module on a bounded input and the WRONG one for an
  # eight-step recurrence on noise -- native fails it against itself. The
  # assertion is `resolved`.
  _cmp('traj (final)', got['traj'][-1], np.asarray(ref['traj'])[-1])
  _cmp('atom14 positions', got['final_atom14_positions'],
       np.asarray(ref['final_atom14_positions']))
  _cmp('sidechain frames', got['sidechains']['frames'][-1],
       np.asarray(ref['sidechains']['frames'])[-1])
  _cmp('atom37 positions', got['final_atom_positions'],
       np.asarray(ref['final_atom_positions']))
  return 0 if resolved else 1


def trunk_gate(n=20, n_seq=16, seed=0, variant='multimer'):
  """EmbeddingsAndEvoformer END TO END -- the one cell that reaches the
  INPUT EMBEDDER and the RECYCLING adds.

  Everything else in this file compares ONE module on inputs the harness makes
  up. That leaves the wiring uncovered: `preprocess_1d` / `preprocess_msa`,
  `left_single` / `right_single`, the relative-position encoding, the three
  recycle adds (`prev_pos_linear`, `prev_msa_first_row_norm`,
  `prev_pair_norm`), `extra_msa_activations`, and the order the four stacks run
  in. A wrong term in any of those is invisible to a module gate and obvious
  here.

  ONE STRUCTURAL DIFFERENCE has to be bridged, and it is a design choice rather
  than drift: the original builds `msa_feat` and `extra_msa_feat` INSIDE the
  module from `aatype` and the raw alignment, while this package takes them
  PRECOMPUTED in the batch -- because a designed MSA has to arrive as a soft
  one-hot that can carry gradients. So native's own `create_msa_feat` and
  `create_extra_msa_feature` are captured as they run and handed to ours. That
  is legitimate because the pipeline that produces them is itself gated, by
  the `msa` cell, on both variants.

  Templates are OFF here. They have three cells of their own and leaving them
  on would fold a second uncovered path into this number.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  sys.path.insert(0, AF2_ORIGINAL)
  _jax_compat()
  from alphafold.model import config as o_config
  from alphafold.model import modules_multimer as o_mm
  from alphafold.model import prng as o_prng

  from alphafold3.af2.model import modules as our_modules
  from alphafold3.af2.model import config as our_config

  if variant != 'multimer':
    raise SystemExit('N/A: a monomer checkpoint runs on the MULTIMER trunk in '
                     'this package, so the monomer variant of this cell would '
                     'compare two different module trees end to end; the '
                     'per-module cells cross that boundary one piece at a time')

  ckpt, cfg_name = _variant(variant)
  raw = np.load(ckpt, allow_pickle=True)
  pre = 'alphafold/alphafold_iteration/'
  p = _scope(raw, pre, 'evoformer/')
  p_ours = _scope(_load_ours(ckpt, use_templates=False), pre, 'evoformer/')
  if not p:
    raise SystemExit('no evoformer params in %s' % ckpt)

  def _truncate(params, scope, depth):
    """Slice a layer_stack's leading axis to `depth`, on BOTH sides.

    The checkpoint holds 48 evoformer blocks and 4 extra-msa ones; asking the
    stack for fewer without slicing is an assertion, not a shorter run. Same
    rule as MSA_BLOCKS in the af3 gates: truncate both sides or neither.
    """
    out = {}
    for k, v in params.items():
      if scope in k:
        out[k] = {a: (b[:depth] if hasattr(b, 'shape') and b.ndim else b)
                  for a, b in v.items()}
      else:
        out[k] = v
    return out

  o_cfg = o_config.model_config(cfg_name)
  c = o_cfg.model.embeddings_and_evoformer
  c.template.enabled = False
  c.num_msa = 8
  c.num_extra_msa = 8
  c.evoformer_num_block = 2          # two blocks, not 48: this cell is about
  c.extra_msa_stack_num_block = 2    # the WIRING, and 48 blocks only amplify
  o_cfg.model.global_config.bfloat16 = False
  o_cfg.model.global_config.use_remat = False

  rng = np.random.default_rng(seed)
  aatype = rng.integers(0, 20, size=(n,)).astype(np.int32)
  c_m = c.msa_channel
  c_z = c.pair_channel
  batch = {
      'aatype': aatype,
      'seq_mask': np.ones((n,), np.float32),
      'residue_index': np.arange(n, dtype=np.int32),
      'asym_id': np.where(np.arange(n) < n // 2, 1, 2).astype(np.int32),
      'entity_id': np.ones((n,), np.int32),
      'sym_id': np.where(np.arange(n) < n // 2, 1, 2).astype(np.int32),
      'msa': rng.integers(0, 21, size=(n_seq, n)).astype(np.int32),
      'msa_mask': np.ones((n_seq, n), np.float32),
      'deletion_matrix': (rng.random((n_seq, n)) * 3).astype(np.float32),
      'cluster_bias_mask': np.concatenate(
          [np.ones(1), np.zeros(n_seq - 1)]).astype(np.float32),
      'prev_pos': (rng.normal(size=(n, 37, 3)) * 5).astype(np.float32),
      'prev_msa_first_row': (rng.normal(size=(n, c_m)) * 0.5).astype(np.float32),
      'prev_pair': (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32),
  }
  t = lambda d: {k: jnp.asarray(v) for k, v in d.items()}

  # CAPTURED FROM NATIVE'S OWN RUN, not rebuilt here: the two feature builders
  # are wrapped so ours starts from the identical tensors. Rebuilding them
  # outside would need native's key split reproduced by hand, which is the kind
  # of assumption this file has been burned by.
  caught = {}
  _o_msa_feat, _o_extra = o_mm.create_msa_feat, o_mm.create_extra_msa_feature

  def _cap_msa(b):
    out = _o_msa_feat(b)
    caught['msa_feat'] = out
    caught['msa_mask'] = b['msa_mask']
    return out

  def _cap_extra(b, k):
    out = _o_extra(b, k)
    caught['extra_msa_feat'], caught['extra_msa_mask'] = out
    return out

  def o_fwd():
    return o_mm.EmbeddingsAndEvoformer(c, o_cfg.model.global_config)(
        t(batch), is_training=False,
        safe_key=o_prng.SafeKey(jax.random.PRNGKey(0)))

  for _scope_name, _depth in (('extra_msa_stack', c.extra_msa_stack_num_block),
                              ('evoformer_iteration', c.evoformer_num_block)):
    p = _truncate(p, _scope_name, _depth)
    p_ours = _truncate(p_ours, _scope_name, _depth)

  o_mm.create_msa_feat, o_mm.create_extra_msa_feature = _cap_msa, _cap_extra
  try:
    ref = hk.transform(o_fwd).apply(p, jax.random.PRNGKey(0))
  finally:
    o_mm.create_msa_feat, o_mm.create_extra_msa_feature = _o_msa_feat, _o_extra
  if 'msa_feat' not in caught or 'extra_msa_feat' not in caught:
    raise SystemExit('the feature builders never ran -- the capture missed')

  u_cfg = our_config.model_config(cfg_name)
  uc = u_cfg.model.embeddings_and_evoformer
  uc.template.enabled = False
  uc.num_msa, uc.num_extra_msa = c.num_msa, c.num_extra_msa
  uc.evoformer_num_block = c.evoformer_num_block
  uc.extra_msa_stack_num_block = c.extra_msa_stack_num_block
  u_cfg.model.global_config.bfloat16 = False
  u_cfg.model.global_config.use_remat = False

  our_batch = dict(batch)
  # 20 wide, not 21: our wrapper pads [0,1] on the way in, which is the
  # convention convert.py's alphabet reduction is written against.
  our_batch['target_feat'] = np.eye(20, dtype=np.float32)[aatype]
  our_batch['msa_feat'] = np.asarray(caught['msa_feat'], np.float32)
  our_batch['msa_mask'] = np.asarray(caught['msa_mask'], np.float32)
  our_batch['extra_msa_feat'] = np.asarray(caught['extra_msa_feat'], np.float32)
  our_batch['extra_msa_mask'] = np.asarray(caught['extra_msa_mask'], np.float32)
  our_batch['use_dropout'] = np.bool_(False)

  def u_fwd():
    return our_modules.EmbeddingsAndEvoformer(
        uc, u_cfg.model.global_config)(t(our_batch))

  got = hk.transform(u_fwd).apply(p_ours, jax.random.PRNGKey(0))
  print('trunk end to end [%s], %d residues, %d msa rows, %d+%d blocks:'
        % (variant, n, n_seq, c.evoformer_num_block,
           c.extra_msa_stack_num_block))
  ok = True
  for k in ('single', 'pair', 'msa_first_row'):
    if k in got and k in ref:
      ok &= _cmp(k, got[k], np.asarray(ref[k]))
  return 0 if ok else 1


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('module', nargs='?', default='template',
                  choices=('template', 'template_multimer', 'template_1d',
                           'evoformer', 'extra_msa', 'ipa', 'heads', 'msa',
                           'structure', 'trunk'))
  ap.add_argument('--tokens', type=int, default=24)
  ap.add_argument('--variant', default='monomer',
                  choices=('monomer', 'multimer'))
  args = ap.parse_args(argv)
  if not os.path.isdir(AF2_ORIGINAL):
    raise SystemExit('clone the original first: git clone '
                     'https://github.com/google-deepmind/alphafold %s'
                     % AF2_ORIGINAL)
  if args.module == 'trunk':
    return trunk_gate(variant=args.variant)
  if args.module == 'structure':
    return structure_gate(n=args.tokens, variant=args.variant)
  if args.module == 'msa':
    return msa_gate(variant=args.variant)
  if args.module == 'heads':
    return heads_gate(variant=args.variant, n=args.tokens)
  if args.module == 'ipa':
    return ipa_gate(n=args.tokens, variant=args.variant)
  if args.module in ('evoformer', 'extra_msa'):
    return evoformer_gate(n=args.tokens, variant=args.variant,
                          extra=(args.module == 'extra_msa'))
  if args.module == 'template_1d':
    return template_1d_gate(n=args.tokens)
  if args.module == 'template_multimer':
    return template_multimer_gate(n=args.tokens)
  return template_gate(n=args.tokens)


if __name__ == '__main__':
  raise SystemExit(main())
