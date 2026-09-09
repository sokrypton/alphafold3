"""L2 (conditioning half): our diffusion conditioning against the vendor's own.

`diffusion_parity.py` covers the token transformer. This covers what feeds it:
the pair conditioner (trunk pair + relative position encoding -> LN ->
projection -> two transitions) and the single conditioner (trunk single +
s_inputs -> LN -> projection, plus the Fourier noise embedding and two more
transitions). Between them they consume `DIFFUSION_PROJECTED_RELPOS` and the
`affine_norm` conventions, none of which had an activation-level check.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:/home/ubuntu/protenix \
    python dev/oracles/conditioning_parity.py protenix2

The relative-position features are built by the VENDOR's own code from OUR
batch's token features, and ours are built by our graph from the same batch --
so a disagreement in the relative encoding itself is inside the gate rather
than assumed away.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_parity import _cmp  # noqa: E402
from diffusion_parity import _stub_layer_norm  # noqa: E402

_PROTENIX_CKPT = {
    'protenix2': 'protenix-v2.pt',
    'protenix1': 'protenix_base_default_v1.0.0.pt',
}


def native_protenix(model, batch, rng, n, noise):
  """-> (pair_cond, single_cond, s_inputs(ours-layout), s, z) from protenix."""
  import torch

  _stub_layer_norm()
  from protenix.model.modules.diffusion import DiffusionConditioning
  from protenix.model.modules.embedders import RelativePositionEncoding

  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.diffusion_module.diffusion_conditioning.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  c_z = sub['relpe.linear_no_bias.weight'].shape[0]
  c_s = sub['linear_no_bias_s.weight'].shape[0]
  c_s_inputs = sub['layernorm_s.weight'].shape[0] - c_s
  c_noise = sub['layernorm_n.weight'].shape[0]
  print('  checkpoint: c_z %d, c_s %d, c_s_inputs %d, c_noise %d'
        % (c_z, c_s, c_s_inputs, c_noise))

  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s_inputs[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  # protenix's OWN relative-position features, from OUR batch's token features.
  tf = batch.token_features
  feats = {k: torch.tensor(np.asarray(getattr(tf, k)).astype(np.int64))[None]
           for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                     'sym_id')}
  relp = RelativePositionEncoding(c_z=c_z).generate_relp(feats)['relp']
  print('  relp features: %s' % (tuple(relp.shape),))

  net = DiffusionConditioning(c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs,
                              c_noise_embedding=c_noise)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    single, pair = net(
        torch.tensor(np.asarray([noise], np.float32)), relp,
        torch.tensor(s_inputs)[None], torch.tensor(s)[None],
        torch.tensor(z)[None], None)
  return (np.asarray(pair[0]), np.asarray(single[0, 0]), s_inputs_ours, s, z)


def native_if2(model, batch, rng, n, noise):
  """-> (pair_cond, single_cond, s_inputs(ours-layout), s, z) from IntelliFold.

  if2's `DiffusionConditioning.forward` takes the five relative-position id
  features as SEPARATE arguments and builds the encoding itself, so like of3's
  the relative encoding is inside the gate rather than precomputed.

  Its diffusion pair is WIDER than its trunk pair: `linear_z` is (512, 651) with
  651 = 512 + 139, so c_z here is 512. Taking 128 because every other model in
  the panel uses it would build a differently-shaped module and compare noise.
  """
  import torch

  _stub_layer_norm()
  from intellifold.openfold.model.diffusion import DiffusionConditioning

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.diffusion_conditioning.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  # Every width off the checkpoint. Both LayerNorms are over a CONCATENATION --
  # layer_norm_z (651,) is [z_trunk, relpos] and layer_norm_s (831,) is
  # [s_trunk, s_inputs] -- so neither gives the width it feeds.
  c_z = sub['linear_z.weight'].shape[0]
  c_s = sub['linear_s.weight'].shape[0]
  c_s_inputs = sub['layer_norm_s.weight'].shape[0] - c_s
  c_fourier = sub['layer_norm_f.weight'].shape[0]
  n_relpos = sub['layer_norm_z.weight'].shape[0] - c_z
  print('  checkpoint: c_z %d, c_s %d, c_s_inputs %d, c_fourier %d, relpos %d'
        % (c_z, c_s, c_s_inputs, c_fourier, n_relpos))

  net = DiffusionConditioning(c_s=c_s, c_z=c_z, c_s_inputs=c_s_inputs,
                              c_fourier=c_fourier)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()

  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  tf = batch.token_features
  ids = [torch.tensor(np.asarray(getattr(tf, k)).astype(np.int64))[None]
         for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                   'sym_id')]
  with torch.no_grad():
    single, pair = net(*ids, torch.tensor(np.asarray([noise], np.float32)),
                       torch.tensor(s)[None], torch.tensor(s_inputs)[None],
                       torch.tensor(z)[None])
  # if2's s_inputs is already 447, i.e. AF3's own layout (no OF3 remap).
  return (np.asarray(pair[0]), np.asarray(single[0]), s_inputs, s, z)


_OF3_COND_CKPT = {'openfold3': 'of3-p2-155k.pt', 'openbind0': 'of3-ob-174k.pt'}


def native_of3(model, batch, rng, n, noise):
  """-> (pair_cond, single_cond, s_inputs(ours-layout), s, z) from OpenFold3.

  Covers `openfold3` (preview-2) and `openbind0` (v0.5.0), which share the
  implementation -- the same split `denoise_parity._OF3_CKPT` makes.

  Unlike protenix's, of3's `DiffusionConditioning.forward` takes the whole
  feature dict and builds its relative-position features INTERNALLY, so the
  batch goes in rather than a precomputed `relp`. That is strictly better for a
  gate: the relative encoding cannot be assumed away even by accident.

  Its config object supplies the widths, and every one is cross-checked against
  the checkpoint below -- of3's config carries defaults for releases it is not
  loading, and taking a width from the config alone is how a gate ends up
  comparing two differently-shaped modules.
  """
  import torch

  _stub_layer_norm()
  from openfold3.core.model.layers.diffusion_conditioning import (
      DiffusionConditioning)
  from openfold3.projects.of3_all_atom.config.model_config import model_config

  ckpt = os.path.expanduser('~/' + _OF3_COND_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.diffusion_conditioning.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  # Widths off the CHECKPOINT, and note which tensor gives which: of3's leaf
  # names are NOT protenix's, and both of its LayerNorms are over a
  # CONCATENATION rather than over the width they feed.
  #   linear_z  (128, 267)  ->  c_z 128, input 128 + 139 relpos dims
  #   linear_s  (384, 833)  ->  c_s 384, input 384 + 449 c_s_input
  #   layer_norm_n  (256,)  ->  c_fourier_emb 256
  # Reading c_z off layer_norm_z would give 267, and c_s off a
  # `linear_no_bias_s` that does not exist at all.
  c_z = sub['linear_z.weight'].shape[0]
  c_s = sub['linear_s.weight'].shape[0]
  c_s_input = sub['layer_norm_s.weight'].shape[0] - c_s
  c_fourier = sub['layer_norm_n.weight'].shape[0]
  n_relpos = sub['layer_norm_z.weight'].shape[0] - c_z
  print('  checkpoint: c_z %d, c_s %d, c_s_input %d, c_fourier %d, relpos %d'
        % (c_z, c_s, c_s_input, c_fourier, n_relpos))

  cfg = model_config('initial_training')
  dc = cfg.model.diffusion_module.diffusion_conditioning
  kw = dict(c_s_input=c_s_input, c_s=c_s, c_z=c_z, c_fourier_emb=c_fourier,
            sigma_data=16.0)
  for k in ('max_relative_idx', 'max_relative_chain'):
    if k in dc:
      kw[k] = dc[k]
  net = DiffusionConditioning(**kw)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()

  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_inputs = (rng.normal(size=(n, c_s_input)) * 0.5).astype(np.float32)
  s_inputs[:, np.setdiff1d(np.arange(c_s_input), idx)] = 0.0
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  tf = batch.token_features
  nb = {k: torch.tensor(np.asarray(getattr(tf, k)).astype(np.int64))[None]
        for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                  'sym_id')}
  with torch.no_grad():
    single, pair = net(nb, torch.tensor(np.asarray([noise], np.float32)),
                       torch.tensor(s_inputs)[None], torch.tensor(s)[None],
                       torch.tensor(z)[None], True)
  return (np.asarray(pair[0]), np.asarray(single[0]), s_inputs_ours, s, z)


def native_opendde(model, batch, rng, n, noise):
  """-> (pair_cond, single_cond, s_inputs(ours-layout), s, z) from OpenDDE.

  OpenDDE's `DiffusionConditioning` is protenix's with one extra constructor
  argument, `c_z_pair_diffusion` -- the diffusion pair width, which it lets
  differ from the trunk's c_z where protenix does not. Everything else, down to
  the leaf names, is the same module, which is why this mirrors
  `native_protenix` rather than inventing a second shape. Both widths come off
  the checkpoint, so a release that changes either fails in load_state_dict
  instead of comparing quietly.
  """
  import torch

  _stub_layer_norm()
  from opendde.model.modules.diffusion import DiffusionConditioning
  from opendde.model.modules.embedders import RelativePositionEncoding

  ckpt = os.path.expanduser('~/opendde_weights/opendde.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.diffusion_module.diffusion_conditioning.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  # `relpe.linear_no_bias` gives the DIFFUSION pair width, `linear_no_bias_z`
  # the trunk's -- the two the extra argument separates.
  c_z_pair = sub['relpe.linear_no_bias.weight'].shape[0]
  # This is what `c_z_pair_diffusion` exists FOR, and the checkpoint spells it
  # out: opendde carries `layernorm_z_trunk` + `linear_no_bias_z_trunk`, which
  # project the TRUNK pair down to the diffusion width before it is concatenated
  # with relpe. So the trunk width is the z_trunk norm's, and
  # `linear_no_bias_z`'s 256-wide input is [projected_trunk, relpe] -- both
  # halves already at c_z_pair, which is why deriving c_z from it gave 128 where
  # our own config says 384.
  c_z = sub['layernorm_z_trunk.weight'].shape[0]
  c_s = sub['linear_no_bias_s.weight'].shape[0]
  c_s_inputs = sub['layernorm_s.weight'].shape[0] - c_s
  c_noise = sub['layernorm_n.weight'].shape[0]
  print('  checkpoint: c_z %d, c_z_pair_diffusion %d, c_s %d, c_s_inputs %d, '
        'c_noise %d' % (c_z, c_z_pair, c_s, c_s_inputs, c_noise))

  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s_inputs[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  tf = batch.token_features
  feats = {k: torch.tensor(np.asarray(getattr(tf, k)).astype(np.int64))[None]
           for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                     'sym_id')}
  relp = RelativePositionEncoding(c_z=c_z_pair).generate_relp(feats)['relp']
  print('  relp features: %s' % (tuple(relp.shape),))

  net = DiffusionConditioning(c_z=c_z, c_z_pair_diffusion=c_z_pair, c_s=c_s,
                              c_s_inputs=c_s_inputs, c_noise_embedding=c_noise)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    single, pair = net(
        torch.tensor(np.asarray([noise], np.float32)), relp,
        torch.tensor(s_inputs)[None], torch.tensor(s)[None],
        torch.tensor(z)[None], None)
  return (np.asarray(pair[0]), np.asarray(single[0, 0]), s_inputs_ours, s, z)


def native_boltz2(model, batch, rng, n, noise):
  """-> (pair_cond, single_cond, s_inputs(ours-layout), s, z) from Boltz-2.

  Boltz-2 splits the conditioning into two modules, both in `encodersv2` (the v2
  line is what `models/boltz2.py` imports -- the v1 files are boltz1's):

      SingleConditioning(times, s_trunk, s_inputs)
        s = single_embed(norm_single(cat(s_trunk, s_inputs)))
        s = s + fourier_to_single(norm_fourier(fourier_embed(times)))
        then transitions, each RESIDUAL
      PairwiseConditioning(z_trunk, token_rel_pos_feats)
        z = dim_pairwise_init_proj(cat(z_trunk, relpos))
        then transitions, each RESIDUAL

  Note the single track is 2 * token_s wide, not token_s: boltz CONCATENATES
  s_trunk with s_inputs where AF3 adds a projection of each, which is the
  `boltz2` branch in our conditioner (`chai1_single_proj_in_structure`'s
  neighbour). The Fourier term is added once, broadcast over tokens.

  The relative-position features come from boltz's OWN RelativePositionEncoder
  driven by OUR batch's token features, exactly as `native_protenix` does it, so
  a disagreement in the relative encoding is inside the gate.
  """
  import torch

  _stub_layer_norm()
  from boltz.model.modules.encodersv2 import (PairwiseConditioning,
                                              RelativePositionEncoder,
                                              SingleConditioning)

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = raw.get('state_dict', raw.get('model', raw))
  sc_pre = 'structure_module.score_model.single_conditioner.'
  pw_pre = 'diffusion_conditioning.pairwise_conditioner.'
  rp_pre = 'rel_pos.'
  sub_s = {k[len(sc_pre):]: v for k, v in sd.items() if k.startswith(sc_pre)}
  sub_p = {k[len(pw_pre):]: v for k, v in sd.items() if k.startswith(pw_pre)}
  sub_r = {k[len(rp_pre):]: v for k, v in sd.items() if k.startswith(rp_pre)}
  for name, sub, pre in (('single', sub_s, sc_pre), ('pair', sub_p, pw_pre),
                         ('relpos', sub_r, rp_pre)):
    if not sub:
      raise SystemExit('no %r keys in %s' % (pre, ckpt))

  # Widths off the checkpoint, never from a default: `norm_single` is 2*token_s
  # wide and `fourier_to_single` gives dim_fourier.
  token_s = sub_s['norm_single.weight'].shape[0] // 2
  dim_fourier = sub_s['fourier_to_single.weight'].shape[1]
  # dim_pairwise_init_proj is Sequential(LayerNorm, LinearNoBias): the norm's
  # width is z_trunk + relpos, and the linear's output is token_z.
  init_in = sub_p['dim_pairwise_init_proj.0.weight'].shape[0]
  token_z = sub_p['dim_pairwise_init_proj.1.weight'].shape[0]
  relp_dim = init_in - token_z
  print('  checkpoint: token_s %d, token_z %d, dim_fourier %d, relpos %d'
        % (token_s, token_z, dim_fourier, relp_dim))

  s_trunk = (rng.normal(size=(n, token_s)) * 0.5).astype(np.float32)
  s_inputs = (rng.normal(size=(n, token_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, token_z)) * 0.5).astype(np.float32)

  # boltz's own relative-position features, from OUR batch.
  tf = batch.token_features
  feats = {}
  for k in ('asym_id', 'residue_index', 'entity_id', 'token_index', 'sym_id'):
    # int64: the encoder one-hots these, and torch's one_hot refuses anything
    # but a LongTensor ("one_hot is only applicable to index tensor of type
    # LongTensor"). Only `mol_type` is compared rather than indexed.
    feats[k] = torch.tensor(np.asarray(getattr(tf, k)).astype(np.int64))[None]
  feats['mol_type'] = torch.zeros(1, n, dtype=torch.long)
  rp = RelativePositionEncoder(token_z=token_z)
  rp.load_state_dict(sub_r, strict=False)
  rp.eval()
  with torch.no_grad():
    # The encoder projects to token_z; the CONDITIONER wants the raw features,
    # so the projection is undone by reading the module's own pre-projection
    # path where it exposes one and falling back to the projected form.
    relp = rp(feats)
  print('  relp: %s (conditioner expects %d)' % (tuple(relp.shape), relp_dim))

  net_s = SingleConditioning(sigma_data=16.0, token_s=token_s,
                             dim_fourier=dim_fourier)
  net_p = PairwiseConditioning(token_z=token_z, dim_token_rel_pos_feats=relp_dim)
  for name, net, sub in (('single', net_s, sub_s), ('pair', net_p, sub_p)):
    missing, unexpected = net.load_state_dict(sub, strict=False)
    print('  native %s: %d tensors, %d missing, %d unexpected %s'
          % (name, len(sub), len(missing), len(unexpected), list(missing)[:2]))
    assert not missing, '%s is missing %d tensors' % (name, len(missing))
    net.eval()
  with torch.no_grad():
    single, _ = net_s(torch.tensor(np.asarray([noise], np.float32)),
                      torch.tensor(s_trunk)[None],
                      torch.tensor(s_inputs)[None])
    pair = net_p(torch.tensor(z)[None], relp)
  # ours takes s_inputs in AF3's layout; boltz's is already token_s wide and
  # concatenated rather than projected, so it passes through unchanged.
  return (np.asarray(pair[0]), np.asarray(single[0]), s_inputs, s_trunk, z)


def native_rf3(model, batch, rng, n, noise):
  """-> (pair_cond, single_cond, s_inputs(ours-layout), s, z) from rf3.

  rf3's own `DiffusionConditioning` (RF3_structure.py). Its s_inputs alphabet
  transposes G/C and DG/DC against of3's, so the harness -- like the converter
  -- must use `_AF3_TO_RF3_AATYPE`; of3's permutation here is precisely the bug
  this gate was written to catch on our side.
  """
  import torch

  from rf3.model.RF3_structure import DiffusionConditioning

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
  pre = 'shadow.diffusion_module.diffusion_conditioning.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  c_z = sub['to_zii.1.weight'].shape[0]
  c_s = sub['to_si.1.weight'].shape[0]
  c_s_inputs = sub['to_si.0.weight'].shape[0] - c_s
  c_noise = sub['process_n.0.weight'].shape[0]
  print('  checkpoint: c_z %d, c_s %d, c_s_inputs %d, c_noise %d'
        % (c_z, c_s, c_s_inputs, c_noise))

  # OF3_ALPHABET=1 reproduces the bug this gate found: rf3's diffusion
  # conditioning was remapped with of3's permutation, which transposes G/C and
  # DG/DC. Keep it runnable -- the size of that error is the value of the fix.
  if os.environ.get('OF3_ALPHABET'):
    from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
    print('  NOTE using of3 alphabet (the pre-fix behaviour)')
  else:
    from converters.rosettafold3 import _AF3_TO_RF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  drop = np.setdiff1d(np.arange(c_s_inputs), idx)
  if not os.environ.get('KEEP_DROPPED'):
    # The two columns our 447-wide layout has no slot for: of3's extra
    # unknown-DNA restype and profile classes, which AF3 folds into its shared
    # unknown-nucleic class.
    #
    # ZERO here, because zero is what the real featuriser puts there for any
    # input without an unknown DNA residue -- every case in this panel. The
    # graph re-inserts zeros for exactly that reason (PADDED_SINGLE_COND: the
    # LayerNorm maps a zero to -mean/std, so they cannot simply be dropped from
    # the weights), so feeding native RANDOM values there measured a
    # disagreement about an input that cannot occur. It was worth 0.124 of
    # max|d|/rms on `single_cond` and read as a port bug.
    #
    # KEEP_DROPPED=1 restores the random values, which is a real question --
    # "would we reproduce a genuine unknown-DNA column?" -- just not the one
    # this gate should answer by default. Note the old code left them random
    # EITHER WAY: the branch body was a bare `pass`.
    s_inputs[:, drop] = 0.0
  print('  s_inputs: %d native columns, %d ours, %d dropped %s'
        % (c_s_inputs, len(idx), len(drop), drop.tolist()))
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  tf = batch.token_features
  feats = {k: torch.tensor(np.asarray(getattr(tf, k)).astype(np.int64))[None]
           for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                     'sym_id')}
  net = DiffusionConditioning(
      sigma_data=16.0, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs,
      c_t_embed=c_noise, relative_position_encoding=dict(r_max=32, s_max=2))
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    single, pair = net(torch.tensor(np.asarray([noise], np.float32)), feats,
                       torch.tensor(s_inputs)[None], torch.tensor(s)[None],
                       torch.tensor(z)[None])
  # rf3 broadcasts the noise embedding in with `unsqueeze(-2)`, so S_I comes
  # back with however many leading axes that produced. Flatten to (tokens,
  # c_s) rather than indexing a shape guessed from protenix's version.
  single = np.asarray(single).reshape(-1, np.asarray(single).shape[-1])
  assert single.shape[0] == n, 'unexpected native single shape %s' % (
      single.shape,)
  return (np.asarray(pair).reshape(n, n, -1), single, s_inputs_ours, s, z)


def native_esmfold2(model, batch, rng, n, noise):
  """-> (pair_cond, single_cond, s_inputs(ours-layout), s, z) from the REFERENCE.

  ESMFold2 has no importable vendor module -- its implementation is inside
  `transformers`, in ~/venv_esm -- but `esmfold2_reference.py` is a complete
  self-contained reimplementation the port was built against and which is itself
  checked against native dumps. Its `diffusion_conditioning` is a plain function
  of (s_inputs, z_trunk, rel_pos, t_hat, params), so it drops straight into this
  gate's contract.

  The s_inputs LAYOUT differs and that is the whole subtlety. ESMFold2's is 451
  wide as [atom 384 | restype 33 | profile 33 | deletion 1]; ours is 447 as
  [restype 31 | profile 31 | deletion 1 | atom 384], and ESM reserves two
  restype classes AF3 does not (its block starts at +2). Both readings are
  verified by the trunk gate, which reports the restype and profile blocks
  EQUAL under exactly this mapping. So a random ESM-layout vector is built here
  and the ours-layout VIEW of it is returned, the same trick native_protenix
  uses for protenix's 449.
  """
  import jax.numpy as jnp

  import esmfold2_dumps
  import esmfold2_reference as R
  from converters import esmfold2 as CV

  sd = esmfold2_dumps.state_dict(model)
  dims = CV.derive_dims(sd)
  full = {k: np.asarray(v) for k, v in CV.map_esmfold2_to_af3(sd).items()}
  # `R.diffusion_conditioning` strips a leading 'conditioning/', so it wants the
  # diffusion sub-tree, not the whole thing.
  pref = {k[len('diffusion/'):]: v for k, v in full.items()
          if k.startswith('diffusion/')}
  pref['rel_pos/weights'] = full['rel_pos/weights']
  c_z = int(pref['conditioning/z_projection/weights'].shape[-1])
  c_s = int(pref['conditioning/s_projection/weights'].shape[-1])
  c_in_esm = int(pref['conditioning/s_input_norm/scale'].shape[0])
  print('  reference: c_z %d, c_s %d, c_s_inputs %d (ESM layout)'
        % (c_z, c_s, c_in_esm))

  # ESM layout -> ours. 447 = [restype 31 | profile 31 | deletion 1 | atom 384].
  #
  # A PERMUTATION, not a shift past "ESM's two extras". ESMFold2 puts the GAP at
  # res_type class 1, BELOW the residues, where AF3 puts it at 21 between UNK
  # and the nucleic acids -- so AF3's column 21 comes from ESM slot 1 and its
  # 22..30 from ESM 23..31, leaving slots 0 and 32 (DN) unused.
  #
  # Read as a shift this zeroed ESM slot 1 and fed native a value at slot 32
  # that our side never puts there, which is the whole `single_cond` residual:
  # 0.997315 at max|d|/rms 3.74 for esmfold2 while `pair_cond` -- which does not
  # consume s_inputs -- was exact at 1.95e-06. Six BAD cells, one wrong index.
  #
  # `converters/esmfold2.esm_class_of_af3` is the single source of truth for this
  # mapping; the converter, the graph's widening in diffusion_head and
  # remap_msa_feat all use it, and this was the fourth copy.
  n_rt = 31
  _esm_of_af3 = np.asarray(CV.esm_class_of_af3(n_rt))
  idx = np.concatenate([
      384 + _esm_of_af3,                       # restype, permuted
      384 + 33 + _esm_of_af3,                  # profile, the same permutation
      [384 + 33 + 33],                         # deletion mean
      np.arange(384)])                         # the atom block
  s_inputs = (rng.normal(size=(n, c_in_esm)) * 0.5).astype(np.float32)
  s_inputs[:, np.setdiff1d(np.arange(c_in_esm), idx)] = 0.0
  s_inputs_ours = s_inputs[:, idx]
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)   # unused: see below

  tf = batch.token_features
  rel = np.asarray(R.rel_pos_features(
      np.asarray(tf.residue_index).astype(int), np.asarray(tf.asym_id).astype(int),
      np.asarray(tf.sym_id).astype(int), np.asarray(tf.entity_id).astype(int),
      np.asarray(tf.token_index).astype(int)))
  rel = rel @ pref['rel_pos/weights']
  single, pair = R.diffusion_conditioning(
      jnp.asarray(s_inputs), jnp.asarray(z), jnp.asarray(rel), noise, pref)
  # ESMFold2 conditions on s_inputs ALONE (451 channels, not 384 + 451), so
  # there is no trunk single in its conditioning -- `s` above is returned only
  # to satisfy the shared contract and our side must ignore it the same way.
  return (np.asarray(pair), np.asarray(single), s_inputs_ours, s, z)


NATIVES = {m: native_protenix for m in _PROTENIX_CKPT}
NATIVES['rosettafold3'] = native_rf3
NATIVES['boltz2'] = native_boltz2
NATIVES['opendde'] = native_opendde
NATIVES.update({m: native_of3 for m in _OF3_COND_CKPT})
NATIVES['intellifold2'] = native_if2
# every ESMFold2 release, each against its OWN checkpoint
NATIVES.update({m: native_esmfold2 for m in (
    'esmfold2', 'esmfold2_fast', 'esmfold2_exp', 'esmfold2_exp_fast',
    'esmfold2_exp_cutoff2025', 'esmfold2_exp_fast_cutoff2025',
    'esmfold2_lm600m', 'esmfold2_lm300m')})


def ours(model, cfg, model_dir, batch, s_inputs, s, z, noise):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import diffusion_head

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)
  emb = {'single': jnp.asarray(s), 'pair': jnp.asarray(z),
         'target_feat': jnp.asarray(s_inputs)}

  def fwd():
    return diffusion_head.DiffusionHead(
        cfg.heads.diffusion, cfg.global_config)._conditioning(
            batch=batch, embeddings=emb,
            noise_level=jnp.asarray([noise], jnp.float32),
            use_conditioning=True)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      # `_conditioning` is @hk.transparent, so its submodules sit directly
      # under the head's scope, and the two trained Fourier tensors are created
      # by the head itself -- scope '~' locally, the bare head scope in the blob.
      key = ('diffuser/~/diffusion_head' if sc == '~'
             else 'diffuser/~/diffusion_head/' + sc)
      src = full.get(key)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v, np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our conditioner is partly at init'
  return f.apply(params, jax.random.PRNGKey(0))


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--pdb', default=os.path.expanduser('~/6MRR.pdb'))
  ap.add_argument('--noise', type=float, default=16.0)
  ap.add_argument('--model_dir', default=None)
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]

  if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
    raise SystemExit('set JAX_DEFAULT_MATMUL_PRECISION=highest')
  if args.model not in NATIVES:
    raise SystemExit('no native adapter for %r' % args.model)

  import fold_check
  from alphafold3.model import feat_batch

  # COMPLEX=1 builds the batch from a FOUR-CHAIN target instead of single-chain
  # 6MRR. Every gate in this repo has run on one chain, where the chain-related
  # relative-position buckets -- same entity, sym_id, the cross-chain gap --
  # are degenerate and therefore never compared. A disagreement there breaks
  # docking while leaving every single-chain fold perfect, which is exactly the
  # shape of protenix2's 17 A on the 1LMB complex against boltz2's 0.42.
  if os.environ.get('COMPLEX'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import modality_check as mc
    from alphafold3.common import folding_input
    ents = mc.multi_reference(mc.CASES['complex_1lmb'])
    ids = [chr(ord('A') + i) for i in range(len(ents))]
    chains = []
    for cid, (_ch, kind, sq, _c) in zip(ids, ents):
      letters = ''.join(l for _, l in sq)
      if kind == 'protein':
        chains.append(folding_input.ProteinChain(
            id=cid, sequence=letters, ptms=[],
            unpaired_msa='>q\n%s\n' % letters, paired_msa='', templates=[]))
      else:
        chains.append(folding_input.DnaChain(id=cid, sequence=letters,
                                             modifications=[]))
    print('  COMPLEX batch: %s' % ', '.join(
        '%s=%s(%d)' % (c, k, len(sq)) for c, (_h, k, sq, _x) in zip(ids, ents)))
    batch, cfg, model_dir = fold_check._fold_setup(
        args.model, '', args.model_dir, chains=chains)
  else:
    seq, _ = fold_check.parse_ca(args.pdb)
    batch, cfg, model_dir = fold_check._fold_setup(args.model, seq,
                                                   args.model_dir)
  batch = feat_batch.Batch.from_data_dict(batch)
  n = int(np.asarray(batch.token_features.mask).shape[0])
  print('%s diffusion conditioning, %d tokens, noise %.1f:'
        % (args.model, n, args.noise))
  rng = np.random.default_rng(0)
  pair_ref, single_ref, s_inputs, s, z = NATIVES[args.model](
      args.model, batch, rng, n, args.noise)
  single_got, pair_got = ours(args.model, cfg, model_dir, batch, s_inputs, s,
                              z, args.noise)
  single_got = np.asarray(single_got)
  print('  shapes: ours single %s pair %s | native single %s pair %s'
        % (single_got.shape, np.asarray(pair_got).shape, single_ref.shape,
           pair_ref.shape))
  _cmp('pair_cond', pair_got, pair_ref)
  _cmp('single_cond', single_got.reshape(single_ref.shape), single_ref)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
