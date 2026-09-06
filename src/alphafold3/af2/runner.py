'''
the AlphaFold2 runner

Turns parameters into model outputs. Everything above this -- masks, losses,
schedules, the loop -- is model-agnostic; this is where AF2 specifically enters.

It is v1's `_get_model._model` with two changes:

  - it stops at `outputs`. v1 computes its losses inside the same function, which
    is why adding a loss meant editing loss.py or using a callback with a
    different signature. Here Design owns the objective, so a user's loss is a
    peer of the builtin ones and the gradient flows through both identically.
  - the protocol branches are gone. v1 special-cases binder in `_get_seq`
    (concatenating a fixed target sequence) and fixbb/hallucination/partial for
    copies. Both fall out of the spec: `seq_fixed` pins the target and `copies`
    drives expansion.

AF2's model code is vendored under `alphafold3.af2.model`, so this package can
run AlphaFold 2 without an installed ColabDesign. That matters for the same
reason it did there: sharing the `colabdesign` name with a user's own install
means imports resolve to whatever branch and jax pin they happen to have, and
results change with no error.

AF2 does NOT ride the AF3 graph. Every other model in this repo is AF3-lineage
-- pair+single trunk into a diffusion sampler -- so it is a weight remap plus a
`global_config.model` branch. AF2 shares neither end: its trunk is MSA row and
column attention where AF3 uses pair-weighted averaging, and its head is IPA
over backbone frames and torsions where AF3 diffuses coordinates. There is no
weight transform between those, so AF2 is a SIBLING network reached by its own
runner, not a branch inside `model.Model`.
'''

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .sequence import expand_copies, pin_fixed, soft_seq

ALPHABET_SIZE = 20


def _import_v1():
  '''AF2's model code, vendored under `alphafold3.af2`

  Copied from ColabDesign v1 main -- the branch every numerical comparison was
  made against -- with `colabdesign.af.alphafold` rewritten to `alphafold3.af2`
  and the three JAX-compat fixes applied. Self-contained: nothing in the tree
  reaches back into v1 or into colabdesign2.
  '''
  from .model import config, data, model
  return config, data, model


def load_params(model_names, data_dir, use_templates=False, use_multimer=False):
  '''load AF2 haiku parameters by name'''
  _config, data, _model = _import_v1()
  # gamma's loader takes rm_templates to drop the template weights; main's does
  # not. Adapt rather than assume a branch.
  import inspect
  extra = {}
  if 'rm_templates' in inspect.signature(data.get_model_haiku_params).parameters:
    extra['rm_templates'] = (not use_templates and not use_multimer)

  out = []
  for name in model_names:
    p = data.get_model_haiku_params(model_name=name, data_dir=data_dir,
                                    fuse=True, **extra)
    if p is not None:
      out.append(fit_to_multimer_graph(p))
  if not out:
    raise FileNotFoundError(
        f'no AF2 params loaded from {data_dir!r} for {model_names}; expected '
        f'{data_dir}/params/params_<name>.npz')
  return out


def fit_to_multimer_graph(params):
  '''normalize a checkpoint to the unified multimer graph, in memory only

  The graph gives the IPA scalar projections use_bias=True so it is identical
  for monomer and multimer -- switching weights is a value swap, not a
  recompile. Native multimer checkpoints have no scalar bias, so add zeros
  (adding 0 is a numerical no-op, so native multimer output is unchanged). A
  converted monomer supplies its real biases instead (see convert.py, Stage 2).

  Never writes to disk -- the params directory stays the canonical DeepMind
  .npz files; this runs each load.
  '''
  import numpy as np
  out = dict(params)
  for k, v in params.items():
    if k.endswith('_scalar_projection') and 'weights' in v and 'bias' not in v:
      w = v['weights']
      out[k] = {**v, 'bias': np.zeros(w.shape[1:], w.dtype)}
  return out


def make_config(model_type='alphafold2_ptm', use_templates=False,
                num_recycle=0, use_remat=True, use_bfloat16=True,
                subbatch_size=None, use_dgram=False, use_dgram_pred=False,
                flash_attention=None, heads=None):
  '''build the AF2 ml_collections config for a model type

  The non-ptm path is reachable here and is not in v1 main, whose every
  non-multimer branch is hardcoded to _ptm.
  '''
  config, _data, _model = _import_v1()
  if 'multimer' in model_type:
    cfg = config.model_config('model_1_multimer')
  elif 'ptm' in model_type:
    cfg = config.model_config('model_1_ptm' if use_templates else 'model_3_ptm')
  else:
    cfg = config.model_config('model_1' if use_templates else 'model_3')

  # v1's model_config shares nested mutable state across calls (mutating one
  # returned config's outer_product_mean.first or position_scale leaks into the
  # next). The unified graph mutates exactly those fields per regime, so isolate
  # every config with a deep copy -- otherwise an on_multimer_graph runner
  # silently corrupts a later native-multimer runner in the same process.
  import copy
  cfg = copy.deepcopy(cfg)

  # ColabDesign2: AF3's attention kernel on AF2's tensors. AF2 materialises the
  # full (batch, heads, q, k) logits matrix, which is quadratic in sequence
  # length; tokamax's flash kernels are linear. q/k/v and the bias are already
  # in tokamax's layout, so this is a dispatch rather than a rewrite.
  #
  # None means the platform decides -- and on this hardware that is 'xla',
  # because Triton is a launch-time crash on consumer Ada. 'none' keeps AF2's
  # own einsum, which is what the oracle tests compare against.
  if flash_attention is None:
    from alphafold3.model.components.platform import attention_config
    flash_attention = attention_config()['attention']
  cfg.model.global_config.flash_attention = flash_attention
  # below this the einsum wins anyway, and AF2's MSA column attention over a
  # single sequence is length 1, which the flash kernels reject
  cfg.model.global_config.flash_attention_min_len = 256

  # ColabDesign2: drop heads the objective does not read. AF2 already skips any
  # head whose config weight is 0 (modules.py:102) -- nobody had used that as a
  # speed control. A dgram_cce objective needs the distogram head alone; the
  # structure module's 8 IPA layers, the confidence heads and the
  # experimentally-resolved head are pure cost.
  #
  # Same idea as AF3's structure=False, one model down. Note predicted_lddt and
  # predicted_aligned_error read representations['structure_module'], so
  # dropping the structure module drops them too -- and recycling reads
  # prev_pos from it, so this is only safe at num_recycle=0.
  if heads is not None:
    if num_recycle:
      raise ValueError('dropping heads needs num_recycle=0: recycling reads '
                       "prev_pos from the structure module")
    keep = set(heads)
    for name in list(cfg.model.heads.keys()):
      if name not in keep and 'weight' in cfg.model.heads[name]:
        cfg.model.heads[name].weight = 0.0

  cfg.model.num_recycle = num_recycle
  gc = cfg.model.global_config
  gc.use_remat = use_remat
  gc.use_dgram = use_dgram          # read unconditionally by modules.py
  gc.bfloat16 = use_bfloat16
  gc.bfloat16_output = use_bfloat16
  gc.subbatch_size = subbatch_size
  # gamma-only knobs; set when present so one runner serves both branches
  # use_dgram_pred recycles the PREDICTED distogram rather than one derived from
  # the predicted coordinates. Both feed the same prev_pos_linear, so with it
  # off the two use_dgram paths are the same computation -- which is why
  # use_dgram=True alone changes nothing.
  for key, value in (('use_dgram_pred', use_dgram_pred),
                     ('bfloat16_output', use_bfloat16)):
    if key in gc:
      gc[key] = value
  return cfg


def default_model_names(model_type='alphafold2_ptm', use_templates=False):
  if 'multimer' in model_type:
    return [f'model_{k}_multimer_v3' for k in (1, 2, 3, 4, 5)]
  suffix = '_ptm' if 'ptm' in model_type else ''
  ks = (1, 2) if use_templates else (1, 2, 3, 4, 5)
  return [f'model_{k}{suffix}' for k in ks]


class AF2Runner:
  '''parameters -> AlphaFold2 outputs

  apply() is pure and differentiable, so Design can wrap it in
  jax.value_and_grad together with whatever losses are registered.
  '''

  def __init__(self, model_type='alphafold2_ptm', use_templates=False,
               data_dir='.', model_names=None, num_seq=1, copies=1,
               block_diag=None, shuffle_first=True, num_recycle=0,
               num_msa=512, num_extra_msa=1024, use_cluster_profile=False,
               use_remat=True, use_bfloat16=True, use_dgram=False,
               use_dgram_pred=False,
               model_params=None, cfg=None, wt_aatype=None, seq_fixed=None,
               flash_attention=None, heads=None, recycle_remat=True,
               recycle_backprop=False, sample_models=False,
               on_multimer_graph=False):
    _config, _data, model = _import_v1()

    self.model_type = model_type
    self.use_templates = use_templates
    self.use_ptm = 'ptm' in model_type or 'multimer' in model_type
    self.num_seq = num_seq
    self.copies = copies
    self.block_diag = (not 'multimer' in model_type) if block_diag is None else block_diag
    self.shuffle_first = shuffle_first
    self.num_msa = num_msa
    self.num_extra_msa = num_extra_msa
    self.use_cluster_profile = use_cluster_profile
    self.wt_aatype = wt_aatype
    self.seq_fixed = seq_fixed

    self.use_dgram = use_dgram
    # how the gradient crosses a recycle boundary:
    #   recycle_remat=True   (default) gradient through every pass, memory of
    #                        one, at the cost of recomputing each in the
    #                        backward -- what jax.checkpoint buys
    #   recycle_backprop     keep every pass's activations; memory grows with
    #                        the recycle count. v1's 'backprop' mode
    #   neither              detach `prev`; v1's default 'last' mode, cheapest,
    #                        and the sequence only affects the final pass
    self.recycle_remat = recycle_remat
    self.recycle_backprop = recycle_backprop

    # ONE graph: everything runs on the multimer network. A monomer model has
    # its weights converted at load (convert.py) and the config put in the
    # monomer regime (position_scale=10, outer_product_mean.first=False).
    # Verified bit-exact against the retired monomer graph's golden in
    # test_af2_convert. `on_multimer_graph` is now always-on for monomer types
    # and kept only as an accepted-but-ignored kwarg for back-compat.
    del on_multimer_graph
    self.on_multimer_graph = 'multimer' not in model_type
    use_multimer_graph = True

    if cfg is not None:
      self.cfg = cfg
    elif self.on_multimer_graph:
      if use_templates:
        raise ValueError('on_multimer_graph does not support templates: the '
                         'monomer and multimer template embedders differ and '
                         'monomer template weights cannot be converted. Use '
                         'native multimer weights for templates.')
      self.cfg = make_config('alphafold2_multimer_v3', use_templates=False,
                             num_recycle=num_recycle, use_remat=use_remat,
                             use_bfloat16=use_bfloat16, use_dgram=use_dgram,
                             use_dgram_pred=use_dgram_pred,
                             flash_attention=flash_attention, heads=heads)
      # position_scale and outer_product_mean.first are both injected as runtime
      # scalars (features()), not baked, so the graph is regime-agnostic. Only
      # template.enabled stays config (it gates whether the template embedder is
      # instantiated -- a genuine graph choice, not a per-model value).
      self.cfg.model.embeddings_and_evoformer.template.enabled = False
    else:
      self.cfg = make_config(
          model_type, use_templates, num_recycle, use_remat, use_bfloat16,
          use_dgram=use_dgram, use_dgram_pred=use_dgram_pred,
          flash_attention=flash_attention, heads=heads)

    # regime scalar injected at runtime (features()) so the graph is
    # position_scale-agnostic: monomer routing forces the monomer regime (10);
    # otherwise respect the config (native multimer 20, or an explicit cfg).
    self._position_scale = (
        10.0 if self.on_multimer_graph
        else float(self.cfg.model.heads.structure_module.position_scale))
    self._opm_first = (
        0.0 if self.on_multimer_graph
        else float(bool(self.cfg.model.embeddings_and_evoformer
                        .evoformer.outer_product_mean.first)))
    # opm_first is now a COMPILE-TIME setting (modules.py reads the config bool),
    # so bake the regime into the config before RunModel builds: routed monomer
    # needs first=False, native multimer keeps its config value. Recompiles on a
    # monomer<->multimer switch (accepted; future jax.lax.cond restores no-recompile).
    self.cfg.model.embeddings_and_evoformer.evoformer.outer_product_mean.first = \
        bool(self._opm_first)

    # v1-main forces pssm_hard=True for multimer (af/model.py:91-94). Mirror that
    # as the DEFAULT so a caller who never thinks about it gets the working value;
    # an explicit opt['pssm_hard'] still wins. See features() for the measurement.
    self.default_pssm_hard = not self.on_multimer_graph

    if model_params is None:
      names = model_names or default_model_names(model_type, use_templates)
      model_params = load_params(names, data_dir, use_templates,
                                 'multimer' in model_type)
      if self.on_multimer_graph:
        from .convert import convert_monomer_params
        model_params = [convert_monomer_params(p) for p in model_params]
      self.model_names = names[:len(model_params)]
    else:
      self.model_names = model_names or [f'model_{i}' for i in range(len(model_params))]
    self.model_params = model_params

    # BindCraft's sample_models: optimise against a different AF2 model each step
    # so the design is not overfit to one network -- the main reason its binders
    # transfer to prediction and the wet lab. Done v1's way (design.py:run):
    # HOST-SIDE. The param sets stay a plain list; sample_model_params() picks one
    # per step and it is passed to the jitted apply as an ARGUMENT, so only 1x
    # params ever live in the compiled graph and value-swapping never recompiles
    # (the sets share a shape). An earlier version jnp.stack'd all sets and
    # gathered with a traced index INSIDE the jit -- that put 5x params in the
    # graph and OOM'd big complexes at compile ("failed to load CUBIN"), even
    # though params are tiny (one multimer_v3 set is ~0.37 GB).
    self.sample_models = sample_models and len(model_params) > 1
    if self.sample_models:
      shp = lambda p: [x.shape for x in jax.tree_util.tree_leaves(p)]
      s0 = shp(model_params[0])
      if any(shp(p) != s0 for p in model_params[1:]):
        raise ValueError('sample_models needs identically-shaped param sets '
                         '(use one model type, e.g. all multimer_v3); otherwise '
                         'swapping the params argument would recompile.')

    self._runner = model.RunModel(self.cfg, use_multimer=use_multimer_graph)

  def sample_model_params(self, rng=None):
    '''host-side: params of one randomly chosen model for this step (v1's
    sample_models), or None when not sampling. Passed to apply() as an argument
    so only 1x params live in the compiled graph -- no stack, no recompile.'''
    if not self.sample_models:
      return None
    import numpy as _np
    r = rng if rng is not None else _np.random.default_rng()
    n = int(r.integers(0, len(self.model_params)))
    return self.model_params[n]

  # ----------------------------------------------------------------- features

  # gamma builds MSA features through its own pipeline rather than writing
  # msa_feat directly, which is what unlocks real MSAs, subsampling, clustering
  # and MLM. modules.py then reads extra_msa_feat, which main's feature dict does
  # not contain -- so the two branches have genuinely different input contracts.
  use_msa_pipeline = True

  def update_seq_gamma(self, seq, inputs, pssm=None):
    '''gamma's contract: hand the model an MSA, not a pre-baked msa_feat

    v1 gamma inputs.py:_update_seq writes msa / cluster_profile / target_feat /
    aatype / deletion_matrix / msa_mask, and model.py then calls make_msa_feats
    to turn them into msa_feat and extra_msa_feat.
    '''
    one_hot = jnp.pad(seq['pseudo'], [[0, 0], [0, 0],
                                      [0, 22 - seq['pseudo'].shape[-1]]])
    prf = one_hot if pssm is None else jnp.pad(
        pssm, [[0, 0], [0, 0], [0, 22 - pssm.shape[-1]]])

    # opt['profile_gap'] blends the profile channels toward the GAP token -- the
    # profile is meant to carry real MSA statistics, and during single-sequence
    # design there are none. Kept as an option but OFF by default: over 3 seeds on
    # RSO monomer hallucination (multimer, 50 steps) it is NOT an improvement --
    # mean plddt 0.941 (sequence) vs 0.964 (GAP), but that edge is entirely one
    # unlucky baseline seed, and GAP is worse on rg (3.03 vs 1.20), pae and con.
    # Traced, so it costs no recompile and can be scheduled.
    w = jnp.asarray(inputs['opt'].get('profile_gap', 0.0), prf.dtype)
    gap = jnp.zeros(prf.shape[-1], prf.dtype).at[21].set(1.0)
    prf = (1.0 - w) * prf + w * gap
    return {
        'msa': one_hot,
        'cluster_profile': prf,
        'target_feat': one_hot[0, :, :20],
        'aatype': one_hot[0].argmax(-1),
        'deletion_matrix': jnp.zeros(one_hot.shape[:2]),
        'msa_mask': jnp.ones(one_hot.shape[:2]),
    }

  def update_seq(self, seq, inputs, pssm=None):
    '''sequence -> msa_feat / target_feat, v1's update_seq

    The 49-channel msa_feat layout is 22 one-hot, 1 has_deletion,
    1 deletion_value, 22 profile, ... -- v1 writes the one-hot at 0:22 and the
    profile at 25:47.

    The profile is NOT seq['pssm']. v1 passes
    `jnp.where(opt["pssm_hard"], seq["hard"], seq["pseudo"])`, so by default the
    profile channels carry the same pseudo sequence as the one-hot channels.
    Using the softmax instead changes msa_feat by ~8e-2 and every loss with it.

    This is the same profile knob AF3 exposes as PROFILE_MODE (see af3
    .../network/featurization.py): pssm_hard=True == 'hard', pssm_hard=False ==
    'soft'. AF3 additionally has 'frozen' (profile ignores the sequence), which
    AF2 has no equivalent of because AF2 always tracks the sequence here.
    '''
    one_hot = seq['pseudo']
    pssm = seq['pseudo'] if pssm is None else pssm
    target_feat = one_hot[0, :, :20]
    pad = lambda x: jnp.pad(x, [[0, 0], [0, 0], [0, 22 - x.shape[-1]]])
    one_hot, pssm = pad(one_hot), pad(pssm)
    msa_feat = (jnp.zeros_like(inputs['msa_feat'])
                .at[..., 0:22].set(one_hot)
                .at[..., 25:47].set(pssm))
    return {'msa_feat': msa_feat, 'target_feat': target_feat}

  def update_aatype(self, aatype, inputs):
    '''per-residue atom layout tables, v1's update_aatype'''
    from .common import residue_constants as rc
    tables = {'atom14_atom_exists': rc.restype_atom14_mask,
              'atom37_atom_exists': rc.restype_atom37_mask,
              'residx_atom14_to_atom37': rc.restype_atom14_to_atom37,
              'residx_atom37_to_atom14': rc.restype_atom37_to_atom14}
    mask = inputs['seq_mask'][:, None]
    out = {k: jnp.where(mask, jnp.asarray(v)[aatype], 0) for k, v in tables.items()}
    out['aatype'] = aatype
    return out

  def init_prev(self, length, use_dgram=False, init='zeros', batch=None,
                dtype=None):
    """the recycling state AF2 carries between passes

    v1 builds this in _recycle with five interacting booleans
    (use_initial_guess x use_dgram x use_dgram_pred x use_batch_as_template x
    use_initial_atom_pos) covering up to 32 combinations for four real choices.
    Here it is one named choice.

    Note this is not optional: v1 main's RunModel.apply reads
    `self.config.global_config.use_dgram` when `prev` is absent, which raises --
    the attribute lives at config.model.global_config. The bug is latent only
    because v1 always pre-populates prev.

      zeros      start from nothing (v1's default)
      positions  start from the reference coordinates (initial guess)
      dgram      start from a distogram built from those coordinates
    """
    # v1 allocates these in bfloat16 when use_bfloat16 is on; the dtype
    # propagates into the trunk and shifts every downstream value by ~3e-3 if
    # it is float32 instead.
    if dtype is None:
      dtype = jnp.bfloat16 if self.cfg.model.global_config.bfloat16 else jnp.float32
    prev = {'prev_msa_first_row': np.zeros([length, 256], dtype),
            'prev_pair': np.zeros([length, length, 128], dtype)}
    if use_dgram:
      # the bin count comes from the model, not a constant. This was hardcoded
      # to 64 while c.prev_pos.num_bins is 15 and get_prev emits 15, so
      # use_dgram=True failed on the very first pass -- before any recycling --
      # because the initial state had the wrong width for prev_pos_linear.
      #
      # Both recycling paths feed that one Linear: prev_pos converted by
      # dgram_from_positions, or the predicted distogram directly when
      # use_dgram_pred is set. So the width is the model's to decide.
      nb = int(self.cfg.model.embeddings_and_evoformer.prev_pos.num_bins)
      prev['prev_dgram'] = np.zeros([length, length, nb], np.float32)
    else:
      prev['prev_pos'] = np.zeros([length, 37, 3], np.float32)

    if init == 'positions' and batch is not None:
      prev['prev_pos'] = np.asarray(batch['all_atom_positions'], np.float32)
    elif init == 'zeros' or batch is None:
      pass
    elif init != 'positions':
      raise ValueError(f'unknown prev init {init!r}: expected zeros/positions/dgram')
    return prev

  def features(self, params, inputs, key=None):
    '''parameters + static inputs -> the full AF2 input dict'''
    opt = inputs['opt']
    seq = soft_seq(params['seq'], inputs.get('bias'), opt, key,
                   num_seq=self.num_seq, shuffle_first=self.shuffle_first)

    # pin fixed positions to wildtype -- replaces v1's binder special case
    if self.seq_fixed is not None and self.wt_aatype is not None:
      seq = pin_fixed(seq, self.wt_aatype, self.seq_fixed, ALPHABET_SIZE)

    if self.copies > 1:
      seq = jax.tree_util.tree_map(
          lambda x: expand_copies(x, self.copies, self.block_diag), seq)

    # pssm_hard defaults to TRUE on the native multimer model, as v1-main does
    # (af/model.py:91-94 sets opt["pssm_hard"]=True whenever use_multimer). The
    # multimer trunk reads the profile channels as a real MSA profile, and a SOFT
    # profile derails design: measured on RSO monomer hallucination, multimer with
    # a soft profile never folds (plddt stuck ~0.29) where the same run with a hard
    # profile reaches 0.88, matching v1-main. Monomer/ptm keeps False (v1's default
    # there) -- the two model families genuinely want opposite values.
    pssm = jnp.where(opt.get('pssm_hard', self.default_pssm_hard),
                     seq['hard'], seq['pseudo'])
    inputs = dict(inputs)
    if self.use_msa_pipeline:
      from .model.msa import make_msa_feats
      inputs.update(self.update_seq_gamma(seq, inputs, pssm=pssm))
      inputs['seq_mask'] = jnp.asarray(inputs['seq_mask'])
      inputs = make_msa_feats(
          inputs, key if key is not None else jax.random.PRNGKey(0),
          num_msa=self.num_msa, num_extra_msa=self.num_extra_msa,
          use_mlm=False, mlm_opt=opt.get('mlm'),
          use_cluster_profile=self.use_cluster_profile)
    else:
      inputs.update(self.update_seq(seq, inputs, pssm=pssm))
    inputs.update(self.update_aatype(seq['pseudo'][0].argmax(-1), inputs))
    inputs['msa_mask'] = jnp.where(inputs['seq_mask'], inputs['msa_mask'], 0)
    inputs['seq'] = seq
    inputs['use_dropout'] = opt.get('dropout', False)
    inputs['mask_template_interchain'] = opt.get('template', {}).get('rm_ic', False)
    # position_scale as a TRACED runtime scalar rather than a baked config float,
    # so one compiled graph serves both regimes (monomer 10, multimer 20) with no
    # recompile on switch. The structure module reads batch['position_scale'].
    inputs['position_scale'] = jnp.asarray(self._position_scale, jnp.float32)
    inputs['opm_first'] = jnp.asarray(self._opm_first, jnp.float32)
    inputs.setdefault('batch', None)
    if 'prev' not in inputs:
      inputs['prev'] = self.init_prev(inputs['aatype'].shape[0],
                                      use_dgram=self.use_dgram)
    return inputs, seq

  # -------------------------------------------------------------------- apply

  def apply(self, params, inputs, key=None, model_params=None):
    """parameters -> outputs; pure and differentiable

    Recycling runs here rather than in the design loop. num_recycle extra passes
    feed `prev` forward, each detached, and only the final pass carries the
    gradient -- v1's default "last" mode (design.py:_recycle). Detaching matters:
    backpropagating through every pass costs memory linear in the count and is
    what v1's "backprop" mode does deliberately, not by default.

    Before this, num_recycle set a config field and nothing drove a loop, so
    num_recycle=1 gave a loss identical to num_recycle=0 to full precision. It
    was the third flag in this codebase that looked like a control and was not.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    # model_params is chosen HOST-SIDE per step (Design.run -> sample_model_params)
    # and passed in, so only this one set is in the graph. Falls back to the first
    # model when the caller does not sample (single-model or a plain predict).
    mp = model_params if model_params is not None else self.model_params[0]
    n = int(getattr(self.cfg.model, 'num_recycle', 0) or 0)

    def one_pass(params, inputs, key):
      k1, k2 = jax.random.split(key)
      full, seq = self.features(params, inputs, k1)
      return self._runner.apply(mp, k2, full), full, seq

    # rematerialise the recycle body: the gradient flows through every pass
    # while only one pass's activations are kept, at the cost of recomputing
    # each during the backward. Without it the choice is between v1's 'last'
    # mode (detach, and lose the sequence's effect on all but the final pass --
    # measured as a 13x smaller gradient at one recycle) and full backprop,
    # whose memory grows linearly in the recycle count.
    body = jax.checkpoint(one_pass) if self.recycle_remat else one_pass

    for _i in range(n):
      key, sub = jax.random.split(key)
      out, _full, _seq = body(params, inputs, sub)
      prev = out['prev']
      if not (self.recycle_remat or self.recycle_backprop):
        prev = jax.lax.stop_gradient(prev)
      inputs = {**inputs, 'prev': prev}

    key, sub = jax.random.split(key)
    outputs, full, seq = one_pass(params, inputs, sub)
    outputs['seq'] = seq
    outputs['inputs'] = full
    return outputs

  def __call__(self, params, inputs, key=None, backprop=True):
    '''Runner protocol shim -- Design prefers .apply and takes its own gradient'''
    return self.apply(params, inputs, key), None

  def predict(self, inputs, seq, key=None, model_params=None, opt=None):
    '''featurised inputs + a fixed sequence -> outputs, the AF2 analogue of
    AF3Runner.predict.

    A plain forward pass on a known sequence (not a design step): `seq` is the
    whole complex sequence (target + binder, length = the featurised length) and
    is written as a peaked one-hot into params['seq'], read out hard so soft and
    hard both give `seq`. The returned outputs carry AF2's confidence heads
    (predicted_aligned_error, predicted_lddt, ptm/iptm) that validate._pae /
    _iface_conf read the same way as AF3, so the two backends score by identical
    code. Templates and the model set follow how the runner was built --
    use_templates=True selects the template-capable ptm models (1, 2); the
    multimer graph carries templates on all five. Build the runner with
    seq_fixed=None so `seq` fully determines the sequence (no wildtype pinning).
    '''
    from .common import residue_constants as rc
    order = rc.restype_order                       # {aa1: 0..19}, matches ALPHABET_SIZE
    L = int(np.asarray(inputs['residue_index']).reshape(-1).shape[0])
    if len(seq) != L:
      raise ValueError(
          f'seq length {len(seq)} != featurised length {L}; predict() expects the '
          'whole complex sequence (target + binder)')
    x = np.zeros((self.num_seq, L, ALPHABET_SIZE), np.float32)
    for i, a in enumerate(seq):
      j = order.get(a)
      if j is not None:
        x[:, i, j] = 20.0                          # peaked: argmax and softmax give `a`
    o = {'weights': {}, 'alpha': 2.0, 'temp': 1.0, 'soft': 1.0, 'hard': 1.0}
    if opt:
      o.update(opt)
    inp = {**inputs, 'opt': {**inputs.get('opt', {}), **o}}
    if key is None:
      key = jax.random.PRNGKey(0)
    return self.apply({'seq': jnp.asarray(x)}, inp, key, model_params=model_params)

  # ------------------------------------------------------------------- params

  def init_seq(self, length, rng=None, mode='normal', alpha=2.0):
    '''initial sequence parameters, v1's set_seq defaults

    `normal` is v1's default and it means 0.01 * normal (shared/model.py:80) --
    a hundred times smaller than a unit-scale draw. The scale is load-bearing,
    not cosmetic: through stage 1 `soft` is still ramping from 0, and soft_seq's
    `pseudo` is the raw parameter, so what AF2 actually sees at the start is
    these numbers. Near zero they are a near-uniform residue that the gradient
    shapes gently; at unit scale they are already a confident random sequence and
    the run starts from an essentially arbitrary committed point. Measured at
    L=60, initialising with a unit gumbel drove `con` from 2.2 up to 2.8 over 75
    steps where v1 took it from 5.6 down to 1.6.

    `gumbel` is v1's opt-in mode, and its draw is divided by alpha
    (shared/model.py:118): soft_seq multiplies the parameters by alpha to form
    its logits, so dividing here makes the *logits* a standard gumbel whatever
    alpha is.
    '''
    rng = np.random.default_rng() if rng is None else rng
    shape = (self.num_seq, length, ALPHABET_SIZE)
    if mode == 'gumbel':
      x = rng.gumbel(size=shape) / alpha
    elif mode == 'zeros':
      x = np.zeros(shape)
    elif mode == 'normal':
      x = rng.normal(size=shape) * 0.01
    else:
      raise ValueError(f'unknown seq init {mode!r}: expected '
                       'normal/gumbel/zeros')
    return {'seq': jnp.asarray(x, jnp.float32)}
