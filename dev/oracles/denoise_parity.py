"""L3: one full denoise step, ours against the vendor's whole DiffusionModule.

This is the level that subsumes the others on the diffusion side: conditioning,
atom encoder, token transformer, atom decoder and the EDM scaling all run, so a
match here means the whole score model agrees for one step. It is also the last
level with no gate on any protenix model.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:/home/ubuntu/protenix \
    python dev/oracles/denoise_parity.py protenix2

Inputs come from a REAL featurised 6MRR batch (the atom windows and the
relative-position features cannot be synthesised), with the trunk activations
seeded random -- the same split conditioning_parity.py and atom_parity.py use.
The noisy coordinates are shared: our dense (token, slot) array masked
row-major IS native's flat atom list.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atom_parity import flat_atom_features                 # noqa: E402
from confidence_parity import _cmp                         # noqa: E402
from diffusion_parity import _stub_einx, _stub_layer_norm  # noqa: E402

_PROTENIX_CKPT = {
    'protenix2': 'protenix-v2.pt',
    'protenix1': 'protenix_base_default_v1.0.0.pt',
}



# Which vendor package implements a model's diffusion, and where its weights
# are. protenix and opendde share the implementation -- opendde is protenix's
# file with the package renamed -- so this is a row rather than a function, the
# same shape `atom_parity._ENC_SRC` uses.
_DIFF_SRC = {
    'protenix1': ('protenix', '~/protenix_weights/'
                  'protenix_base_default_v1.0.0.pt'),
    'protenix2': ('protenix', '~/protenix_weights/protenix-v2.pt'),
    'opendde': ('opendde', '~/opendde_weights/opendde.pt'),
}


def native_protenix(model, fb, feats, pos_noisy, noise, s_inputs_449, s, z):
  """-> x_denoised (flat atoms) from the vendor's own DiffusionModule.

  Serves protenix1, protenix2 and opendde off `_DIFF_SRC`. Every width below
  already came off the checkpoint, which is what makes the extra model free:
  opendde's `layernorm_s` is 833 = 384 + 449, its relpe is 128-wide, and its
  three stacks are 3 / 24 / 3, all read rather than defaulted.
  """
  import importlib

  import torch

  _stub_layer_norm()
  pkg, ckpt_path = _DIFF_SRC[model]
  DiffusionModule = getattr(
      importlib.import_module(pkg + '.model.modules.diffusion'),
      'DiffusionModule')
  rearrange_qk_to_dense_trunk = getattr(
      importlib.import_module(pkg + '.model.modules.transformer'),
      'rearrange_qk_to_dense_trunk')

  ckpt = os.path.expanduser(ckpt_path)
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd.get('state_dict', sd))
  pre = 'module.diffusion_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  dc = 'diffusion_conditioning.'
  c_z_pair = sub[dc + 'relpe.linear_no_bias.weight'].shape[0]
  # c_z is the width of the pair this module is FED. opendde is fed the 384-wide
  # trunk pair and compresses it; protenix is fed the same width it works in.
  c_z = (sub[dc + 'layernorm_z_trunk.weight'].shape[0]
         if dc + 'layernorm_z_trunk.weight' in sub else c_z_pair)
  extra = ({'c_z_pair_diffusion': c_z_pair} if c_z != c_z_pair else {})
  c_s = sub[dc + 'linear_no_bias_s.weight'].shape[0]
  c_s_inputs = sub[dc + 'layernorm_s.weight'].shape[0] - c_s
  ae = 'atom_attention_encoder.'
  c_atom = sub[ae + 'linear_no_bias_ref_pos.weight'].shape[0]
  c_atompair = sub[ae + 'linear_no_bias_d.weight'].shape[0]
  c_token = sub[ae + 'linear_no_bias_q.weight'].shape[0]
  n_dt = 1 + max(int(k.split('.')[2]) for k in sub
                 if k.startswith('diffusion_transformer.blocks.'))
  heads = sub['diffusion_transformer.blocks.0.attention_pair_bias.'
              'linear_nobias_z.weight'].shape[0]
  print('  checkpoint: c_z %d (pair width in the module %d), c_s %d, '
        'c_atom %d, c_atompair %d, c_token %d, %d token blocks, %d heads'
        % (c_z, c_z_pair, c_s, c_atom, c_atompair, c_token, n_dt, heads))

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  n_atom = feats['ref_pos'].shape[0]
  n_tok = np.asarray(fb.token_features.mask).shape[0]
  # ELEMENT INDEX CONVENTION. protenix (and the OF3 families) featurise an
  # element as `GetAtomicNum() - 1`; our batch stores AF3's 1-indexed
  # `GetAtomicNum()`, and the converter folds the shift into the embedding rows
  # (converters/common.py `fold_element_index_shift`). So NATIVE has to be fed
  # the shifted index -- feeding it ours read c_atom_cond corr 0.832 while the
  # position and charge features were exact at 1.000000, which is how this was
  # found.
  el = np.maximum(0, np.asarray(feats['element']).astype(int) - 1)
  ref_element = np.zeros((n_atom, 128), np.float32)
  ref_element[np.arange(n_atom), el] = 1.0
  chars = np.asarray(feats['atom_name_chars']).astype(int)
  ref_chars = np.zeros((n_atom, 4, 64), np.float32)
  for i in range(4):
    ref_chars[np.arange(n_atom), i, chars[:, i]] = 1.0

  # UNBATCHED here on purpose. protenix's own pipeline windows a 1-D
  # `atom_to_token_idx`, and `gather_pair_embedding_in_dense_trunk` asserts its
  # gather indices are exactly 2-D -- feed it a leading batch axis and that
  # assert is what fires, from inside the vendor's code.
  # BATCHED for opendde, UNBATCHED for protenix, and the vendors' own pipelines
  # are what say so. `update_input_feature_dict` (opendde/model/opendde.py) runs
  # this same construction on a BATCHED ref_pos, so its d_lm/v_lm carry a leading
  # axis and its local attention's `len(z.shape) == len(q.shape) + 2` holds.
  # protenix instead windows a 1-D atom_to_token_idx and its
  # `gather_pair_embedding_in_dense_trunk` asserts exactly 2-D gather indices, so
  # a leading axis fires THAT assert from inside the vendor. Hand-batching d_lm
  # and v_lm after the fact is not the same thing: `pad_info` is built here too,
  # and it then describes the wrong rank.
  _b = (lambda x: t(x)[None]) if model == 'opendde' else t
  q_list, k_list, pad_info = rearrange_qk_to_dense_trunk(
      q=[_b(feats['ref_pos']), _b(feats['ref_space_uid'])],
      k=[_b(feats['ref_pos']), _b(feats['ref_space_uid'])],
      dim_q=[-2, -1], dim_k=[-2, -1], n_queries=32, n_keys=128,
      compute_mask=True)
  d_lm = q_list[0][..., None, :] - k_list[0][..., None, :, :]
  v_lm = (q_list[1][..., None].int() == k_list[1][..., None, :].int()
          ).unsqueeze(dim=-1)

  # protenix's own relative-position features, from OUR batch's token features.
  RelativePositionEncoding = getattr(
      importlib.import_module(pkg + '.model.modules.embedders'),
      'RelativePositionEncoding')
  tf = fb.token_features
  ifd = {k: t(getattr(tf, k), torch.long)[None]
         for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                   'sym_id')}
  ifd = RelativePositionEncoding(c_z=c_z).generate_relp(ifd)
  ifd.update(
      ref_pos=t(feats['ref_pos'])[None],
      ref_charge=t(feats['ref_charge'])[None],
      ref_mask=torch.ones(1, n_atom),
      ref_element=t(ref_element)[None],
      ref_atom_name_chars=t(ref_chars)[None],
      ref_space_uid=t(feats['ref_space_uid'])[None],
      atom_to_token_idx=t(feats['atom_to_token_idx'], torch.long),
      d_lm=d_lm, v_lm=v_lm, pad_info=pad_info)

  # The atom encoder/decoder block counts default to 3, and mini/tiny have 1 --
  # which loads 290 of 766 tensors and reports 476 missing. Derive all three
  # stacks from the checkpoint, as every other harness here does.
  def _n(prefix):
    ks = [k for k in sub if k.startswith(prefix)]
    return 1 + max(int(k[len(prefix):].split('.')[0]) for k in ks) if ks else 0

  n_enc = _n('atom_attention_encoder.atom_transformer.diffusion_transformer.'
             'blocks.')
  n_dec = _n('atom_attention_decoder.atom_transformer.diffusion_transformer.'
             'blocks.')
  print('  atom stacks: encoder %d blocks, decoder %d blocks' % (n_enc, n_dec))
  net = DiffusionModule(c_atom=c_atom, c_atompair=c_atompair,
                        c_token=c_token, c_s=c_s, c_z=c_z, **extra,
                        c_s_inputs=c_s_inputs,
                        atom_encoder={'n_blocks': n_enc, 'n_heads': 4},
                        transformer={'n_blocks': n_dt, 'n_heads': heads},
                        atom_decoder={'n_blocks': n_dec, 'n_heads': 4})
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    x = net(t(pos_noisy)[None][None], t(np.asarray([noise], np.float32)), ifd,
            t(s_inputs_449)[None], t(s)[None], t(z)[None], None, None, None)
  return np.asarray(x).reshape(-1, 3)


def native_if2(model, fb, feats, pos_noisy, noise, s_inputs_449, s, z):
  """-> x_denoised (flat atoms) from IntelliFold-2's own DiffusionModule.

  Two things differ from the protenix adapter:
    * **if2's module returns `r_update`, not `x_denoised`** -- the EDM scaling
      lives OUTSIDE it, in `model.py::diffusion_edm_forward`. So this applies
      it here, with if2's own constants, or the comparison would be against a
      differently-scaled tensor that still correlates well.
    * its `s_inputs` is 447 wide, not 449: `layer_norm_s` is 831 = c_s 384 +
      447, so if2 is NOT one of `model_config.PADDED_SINGLE_COND`. The harness
      hands natives the 449 layout, and this one takes our 447 view of it.
  The config comes from if2's own `v2_inference_config`, with the same three
  `advanced_conversion` flags that file sets -- see the atom-encoder adapter in
  atom_parity.py for why that flag decides the atom layout.
  """
  import torch

  from intellifold.openfold.v2_inference_config import model_config
  from intellifold.openfold.model.diffusion import DiffusionModule

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  cfg = model_config(low_prec=False)
  cfg.globals.advanced_conversion = True
  cfg.diffusion.atom_attention_encoder.advanced_conversion = True
  cfg.diffusion.atom_attention_decoder.advanced_conversion = True
  cfg.globals.chunk_size = None
  cfg.globals.use_deepspeed_evo_attention = False

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  mask = feats['mask']
  n_atom = feats['ref_pos'].shape[0]
  el = np.clip(np.asarray(feats['element']).astype(int), 0, 127)
  ref_element = np.zeros((n_atom, 128), np.float32)
  ref_element[np.arange(n_atom), el] = 1.0
  chars = np.asarray(feats['atom_name_chars']).astype(int)
  ref_chars = np.zeros((n_atom, 4, 64), np.float32)
  for i in range(4):
    ref_chars[np.arange(n_atom), i, chars[:, i]] = 1.0
  tf = fb.token_features
  batch = {
      'ref_pos': t(feats['ref_pos'])[None],
      'ref_charge': t(feats['ref_charge'])[None],
      'ref_mask': torch.ones(1, n_atom),
      'ref_element': t(ref_element)[None],
      'ref_atom_name_chars': t(ref_chars)[None],
      'ref_space_uid': t(feats['ref_space_uid'], torch.long)[None],
      'aggregated_pred_dense_atom_mask': torch.ones(1, n_atom),
      'seq_mask': t(tf.mask)[None],
      'molecule_atom_lens': t(mask.sum(1), torch.long)[None],
  }
  for k in ('asym_id', 'residue_index', 'entity_id', 'token_index', 'sym_id'):
    batch[k] = t(getattr(tf, k), torch.long)[None]

  net = DiffusionModule(cfg)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()

  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s447 = np.asarray(s_inputs_449)[:, idx]

  sd_scale = float(cfg.diffusion.sigma_data)
  tt = torch.tensor([noise], dtype=torch.float32)
  scale_skip = sd_scale ** 2 / (tt ** 2 + sd_scale ** 2)
  scale_out = tt * sd_scale / ((tt ** 2 + sd_scale ** 2).sqrt())
  scale_in = 1 / ((sd_scale ** 2 + tt ** 2).sqrt())
  x_noisy = t(pos_noisy)[None]
  with torch.no_grad():
    r_update = net(x_noisy * scale_in, tt, batch, t(s447)[None], t(s)[None],
                   t(z)[None])
    x = scale_skip * x_noisy + scale_out * r_update
  return np.asarray(x).reshape(-1, 3)


def native_rf3(model, fb, feats, pos_noisy, noise, s_inputs_449, s, z):
  """-> x_denoised (flat atoms) from RoseTTAFold3's own DiffusionModule.

  rf3's module does the EDM scaling itself (`f_pred='edm'`) and returns X_out,
  so no scaling here -- unlike if2's. The feature dict is the union of what the
  conditioning needs (token ids) and what the atom encoder/decoder need (the
  reference features plus `atom_to_token_map`), built exactly as the two L2
  adapters build them, including:
    * the RF3 alphabet for s_inputs (`_AF3_TO_RF3_AATYPE`, which transposes G/C
      against of3's -- the bug conditioning_parity.py was written to catch);
    * `use_atom_level_embedding`, whose all-zero input still contributes a
      constant (see atom_parity.py::native_rf3);
    * `use_chiral_features` ON, with the centres remapped into native's packed
      atom indexing. The port DOES implement this term; running native without
      it cost 0.38 A/atom here and 2.01e-01 on the atom encoder.
  """
  import torch

  from rf3.model.RF3_structure import DiffusionModule
  from converters.rosettafold3 import _AF3_TO_RF3_AATYPE as _remap

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  raw = raw.get('state_dict', raw.get('model', raw))
  pre = 'shadow.diffusion_module.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))
  chiral = not os.environ.get('RF3_NO_CHIRAL')
  if not chiral:
    sub.pop('atom_attention_encoder.process_ch.weight', None)

  enc = 'atom_attention_encoder.'
  c_z = sub['diffusion_conditioning.to_zii.1.weight'].shape[0]
  c_s = sub['diffusion_conditioning.to_si.1.weight'].shape[0]
  c_s_inputs = sub['diffusion_conditioning.to_si.0.weight'].shape[0] - c_s
  c_noise = sub['diffusion_conditioning.process_n.0.weight'].shape[0]
  c_atom = sub[enc + 'process_input_features.weight'].shape[0]
  c_atompair = sub[enc + 'process_d.weight'].shape[0]
  c_token = sub[enc + 'process_q.0.weight'].shape[0]
  n_feat = sub[enc + 'process_input_features.weight'].shape[1]

  def _n(prefix):
    ks = [k for k in sub if k.startswith(prefix)]
    return 1 + max(int(k[len(prefix):].split('.')[0]) for k in ks) if ks else 0

  n_enc = _n(enc + 'atom_transformer.diffusion_transformer.blocks.')
  n_dec = _n('atom_attention_decoder.atom_transformer.diffusion_transformer.'
             'blocks.')
  n_dt = _n('diffusion_transformer.blocks.')
  heads = 16
  print('  checkpoint: c_atom %d, c_atompair %d, c_token %d, c_s %d, c_z %d, '
        'c_s_inputs %d, %d/%d/%d blocks (enc/token/dec)'
        % (c_atom, c_atompair, c_token, c_s, c_z, c_s_inputs, n_enc, n_dt,
           n_dec))

  ale = enc + 'process_atom_level_embedding.process_atom_level_embedding.0.weight'
  ale_dim = sub[ale].shape[1] if ale in sub else 384
  ale = ale in sub

  atom_tr = lambda nb, nh: dict(
      n_queries=32, n_keys=128,
      diffusion_transformer=dict(
          n_block=nb,
          diffusion_transformer_block=dict(
              n_head=nh,
              no_residual_connection_between_attention_and_transition=True,
              kq_norm=True)))
  net = DiffusionModule(
      sigma_data=16.0, c_atom=c_atom, c_atompair=c_atompair, c_token=c_token,
      c_s=c_s, c_z=c_z, f_pred='edm',
      diffusion_conditioning=dict(
          c_s_inputs=c_s_inputs, c_t_embed=c_noise,
          relative_position_encoding=dict(r_max=32, s_max=2)),
      atom_attention_encoder=dict(
          c_tokenpair=c_z, c_atom_1d_features=n_feat, use_inv_dist_squared=True,
          atom_1d_features=['ref_pos', 'ref_charge', 'ref_mask', 'ref_element',
                            'ref_atom_name_chars', 'ref_pos_ground_truth',
                            'has_atom_level_embedding'][
                                :None if n_feat == 393 else 5],
          atom_transformer=atom_tr(n_enc, 4),
          broadcast_trunk_feats_on_1dim_old=False, use_chiral_features=chiral,
          no_grad_on_chiral_center=False, use_atom_level_embedding=ale,
          atom_level_embedding_dim=ale_dim),
      diffusion_transformer=dict(
          n_block=n_dt,
          diffusion_transformer_block=dict(
              n_head=heads,
              no_residual_connection_between_attention_and_transition=True,
              kq_norm=True)),
      atom_attention_decoder=dict(atom_transformer=atom_tr(n_dec, 4)))

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  n_atom = feats['ref_pos'].shape[0]
  el = np.clip(np.asarray(feats['element']).astype(int), 0, 127)
  ref_element = np.zeros((n_atom, 128), np.float32)
  ref_element[np.arange(n_atom), el] = 1.0
  chars = np.asarray(feats['atom_name_chars']).astype(int)
  ref_chars = np.zeros((n_atom, 4, 64), np.float32)
  for i in range(4):
    ref_chars[np.arange(n_atom), i, chars[:, i]] = 1.0
  tf = fb.token_features
  f = {
      'ref_pos': t(feats['ref_pos']),
      'ref_charge': t(feats['ref_charge']).reshape(n_atom, 1),
      'ref_mask': torch.ones(n_atom, 1),
      'ref_element': t(ref_element),
      'ref_atom_name_chars': t(ref_chars),
      'ref_space_uid': t(feats['ref_space_uid'], torch.long),
      'atom_to_token_map': t(feats['atom_to_token_idx'], torch.long),
      'ref_pos_ground_truth': torch.zeros(n_atom, 3),
      'has_atom_level_embedding': torch.zeros(n_atom, 1),
  }
  # THE CHIRALITY CENTRES, in native's packed indexing -- the port implements
  # rf3's chirality term and running native without it compared two different
  # graphs. See the long note in `atom_parity.native_rf3`; it was worth 0.38
  # A/atom on this gate.
  if chiral:
    _m = np.asarray(feats['mask']).reshape(-1)
    _d2p = np.cumsum(_m) - 1
    _c = np.asarray(fb.chirals.centers).astype(int)
    assert _m[_c].all(), 'a chiral centre indexes a padding slot'
    f['chiral_centers'] = t(_d2p[_c], torch.long)
    f['chiral_center_dihedral_angles'] = t(np.asarray(fb.chirals.angles))
    print('  chiral: %d centres, remapped dense -> packed' % len(_c))
  if ale:
    f['atom_level_embedding'] = torch.zeros(8, n_atom, ale_dim)
  # UNBATCHED token ids, single and pair -- only the coordinates carry rf3's
  # leading dim. Batching the trunk tensors makes the conditioning emit a 4-D
  # Z_II, which the atom stack then unsqueezes to 5-D and indexes on the wrong
  # axis; see the same note in atom_parity.py::native_rf3.
  for k in ('asym_id', 'residue_index', 'entity_id', 'token_index', 'sym_id'):
    f[k] = t(getattr(tf, k), torch.long)

  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  # Every AttentionPairBiasDiffusion sets `force_bfloat16 = True` and casts the
  # activation to bf16 inside attention regardless of the surrounding dtype --
  # which on CPU is a hard dtype error, and everywhere is a precision floor we
  # do not share (we run bfloat16='none'). Switched off block by block, the same
  # way diffusion_parity.py does it for the token stack.
  n_off = 0
  for m in net.modules():
    if getattr(m, 'force_bfloat16', False):
      m.force_bfloat16 = False
      n_off += 1
  print('  force_bfloat16 switched off on %d attention blocks' % n_off)
  net.eval()

  # main() builds OUR 447-wide view of s_inputs with OF3's permutation, for
  # every model. rf3's alphabet is not of3's -- it transposes G/C and DG/DC --
  # so the native side has to be handed OUR vector scattered into RF3's
  # positions, NOT the vendor array read at rf3's positions. Those two differ
  # in exactly the transposed columns, and doing it the wrong way costs ~2% on
  # every token (it was worth 2.79 A -> see PARITY.md).
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _of3
  ours_idx = np.concatenate([384 + np.asarray(_of3), 416 + np.asarray(_of3),
                             [448], np.arange(384)])
  s447_ours = np.asarray(s_inputs_449)[:, ours_idx]
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_in = np.zeros((np.asarray(s).shape[0], c_s_inputs), np.float32)
  s_in[:, idx] = s447_ours
  with torch.no_grad():
    x = net(t(pos_noisy)[None], t(np.asarray([noise], np.float32)), f,
            t(s_in), t(s), t(z))
  return np.asarray(x).reshape(-1, 3)


_OF3_CKPT = {'openfold3': 'of3-p2-155k.pt', 'openbind0': 'of3-ob-174k.pt'}


def native_of3(model, fb, feats, pos_noisy, noise, s_inputs_449, s, z):
  """-> x_denoised (flat atoms) from OpenFold3's own DiffusionModule.

  Covers `openfold3` (preview-2) and `openbind0` (v0.5.0). of3's module does
  the EDM scaling itself and takes a config object, so the config comes from
  of3's own `model_config` rather than being retyped -- the one edit is the
  release split that `model_config.PER_BLOCK_PAIR_LAYER_NORM` encodes:
  preview-2 LayerNorms the pair conditioning inside every token block, v0.5.0
  runs it once for the stack. `~/openfold-3` only implements the per-block
  form, so for v0.5.0 the per-block LNs become Identity and the single LN is
  applied to the conditioned pair tensor on the way in -- the same surgery
  diffusion_parity.py does, moved inside the module by wrapping the token
  transformer's forward.
  """
  import copy

  import torch

  from openfold3.core.model.structure.diffusion_module import DiffusionModule
  from openfold3.projects.of3_all_atom.config.model_config import model_config

  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  def _find(cfg, key):
    """The diffusion subtree, located rather than assumed."""
    if hasattr(cfg, 'keys'):
      if key in cfg and 'atom_attn_dec' in cfg[key]:
        return cfg[key]
      for k in cfg:
        got = _find(cfg[k], key) if hasattr(cfg[k], 'keys') else None
        if got is not None:
          return got
    return None

  cfg = _find(copy.deepcopy(model_config), 'diffusion_module')
  if cfg is None:
    raise SystemExit('no diffusion_module subtree in of3 model_config')

  single_ln = 'diffusion_transformer.layer_norm_z.weight' in sub
  n_dt = 1 + max(int(k.split('.')[2]) for k in sub
                 if k.startswith('diffusion_transformer.blocks.'))
  print('  checkpoint: %d token blocks, %s pair LN'
        % (n_dt, 'single' if single_ln else 'per-block'))

  net = DiffusionModule(cfg)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  # Against v0.5.0's OWN tree the shared LayerNorm is part of the module, so
  # nothing is missing and the surgery below must not run. It exists only for
  # the case of a v0.5.0 checkpoint on main's code -- which is how openbind0's
  # transposed end-node pair bias went uncaught for four days.
  if single_ln and missing:
    expect = {'diffusion_transformer.blocks.%d.attention_pair_bias.'
              'layer_norm_z.weight' % i for i in range(n_dt)}
    assert set(missing) == expect, 'unexpected missing: %s' % sorted(
        set(missing) - expect)[:3]
    w = sub['diffusion_transformer.layer_norm_z.weight']
    c_z = w.shape[0]
    for blk in net.diffusion_transformer.blocks:
      blk.attention_pair_bias.layer_norm_z = torch.nn.Identity()
    inner = net.diffusion_transformer.forward

    def pre_ln(*args, **kw):
      # The module calls this with keywords, so find the pair tensor by name
      # or by position rather than assuming a signature.
      ln = lambda x: torch.nn.functional.layer_norm(
          x, (c_z,), weight=w, bias=None, eps=1e-5)
      for name in ('z', 'zij', 'z_ij'):
        if name in kw:
          kw[name] = ln(kw[name])
          return inner(*args, **kw)
      args = list(args)
      assert len(args) >= 3, 'pair tensor is neither a keyword nor positional'
      args[2] = ln(args[2])
      return inner(*args, **kw)

    net.diffusion_transformer.forward = pre_ln
  else:
    assert not missing, 'native is missing %d tensors' % len(missing)
  print('  native: %d tensors, %d missing (%s), %d unexpected %s'
        % (len(sub), len(missing), 'handled' if single_ln else 'none',
           len(unexpected), list(unexpected)[:2]))
  net.eval()

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  n_atom = feats['ref_pos'].shape[0]
  n_tok = np.asarray(fb.token_features.mask).shape[0]
  n_elem = sub['atom_attn_enc.ref_atom_feature_embedder.linear_ref_element.'
               'weight'].shape[1]
  el = np.clip(np.asarray(feats['element']).astype(int) - 1, 0, n_elem - 1)
  ref_element = np.zeros((n_atom, n_elem), np.float32)
  ref_element[np.arange(n_atom), el] = 1.0
  chars = np.asarray(feats['atom_name_chars']).astype(int)
  ref_chars = np.zeros((n_atom, 4, 64), np.float32)
  for i in range(4):
    ref_chars[np.arange(n_atom), i, chars[:, i]] = 1.0
  tf = fb.token_features
  batch = {
      'ref_pos': t(feats['ref_pos'])[None],
      'ref_mask': torch.ones(1, n_atom),
      'ref_element': t(ref_element)[None],
      'ref_charge': t(feats['ref_charge'])[None],
      'ref_atom_name_chars': t(ref_chars)[None],
      'ref_space_uid': t(feats['ref_space_uid'], torch.long)[None],
      'token_mask': torch.ones(1, n_tok),
      'atom_mask': torch.ones(1, n_atom),
      'num_atoms_per_token': t(feats['mask'].sum(1), torch.long)[None],
      'atom_to_token_index': t(feats['atom_to_token_idx'], torch.long)[None],
  }
  for k in ('asym_id', 'residue_index', 'entity_id', 'token_index', 'sym_id'):
    batch[k] = t(getattr(tf, k), torch.long)[None]

  with torch.no_grad():
    x = net(batch=batch, xl_noisy=t(pos_noisy)[None],
            token_mask=torch.ones(1, n_tok), atom_mask=torch.ones(1, n_atom),
            t=t(np.asarray([noise], np.float32)),
            si_input=t(s_inputs_449)[None], si_trunk=t(s)[None],
            zij_trunk=t(z)[None], use_conditioning=True)
  return np.asarray(x).reshape(-1, 3)


def native_boltz2(model, fb, feats, pos_noisy, noise, s_inputs_449, s, z):
  """-> x_denoised (flat real atoms) from Boltz-2's own score model + EDM.

  The last of the boltz2 module holes, and the deepest: it needs everything the
  atom and diffusion gates just gained, plus the two things only the whole step
  has -- the conditioning DICT and the EDM preconditioning.

  Three boltz2 facts:

    * `DiffusionModule.forward` does not take z. It takes a
      `diffusion_conditioning` dict that `DiffusionConditioning` produced
      (q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias), so that
      module is built and run here first, from `diffusion_conditioning.*`.
    * the EDM scaling lives OUTSIDE the score model, in
      `AtomDiffusion.preconditioned_network_forward`: the network is fed
      `c_in(sigma) * x` and `c_noise(sigma)`, and its output is combined as
      `c_skip * x + c_out * r_update`. Same constants AF3 uses, sigma_data 16,
      so the four coefficients are written out here rather than importing the
      sampler (which would also want a schedule and a device).
    * s_inputs is token_s wide (384), NOT AF3's 449. `norm_single` is 768 =
      2 * token_s, i.e. s_trunk concatenated with s_inputs -- which is why main
      feeds boltz2 its own width and hands the same array to both sides.
  """
  import numpy as _np
  import torch

  _stub_layer_norm()
  _stub_einx()
  from atom_parity import _boltz2_feats
  from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
  from boltz.model.modules.diffusionv2 import DiffusionModule
  from boltz.model.modules.encodersv2 import RelativePositionEncoder

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = raw.get('state_dict', raw.get('model', raw))
  take = lambda pre: {k[len(pre):]: v for k, v in sd.items()
                      if k.startswith(pre)}
  c_pre, m_pre, r_pre = ('diffusion_conditioning.',
                         'structure_module.score_model.', 'rel_pos.')
  cs, ms, rs = take(c_pre), take(m_pre), take(r_pre)
  for pre, sub in ((c_pre, cs), (m_pre, ms), (r_pre, rs)):
    if not sub:
      raise SystemExit('no %r keys in %s' % (pre, ckpt))

  atom_s, feat_dim = cs['atom_encoder.embed_atom_features.weight'].shape
  atom_z = cs['atom_encoder.embed_atompair_ref_pos.weight'].shape[0]
  token_s = cs['atom_encoder.s_to_c_trans.0.weight'].shape[0]
  token_z = cs['atom_encoder.z_to_p_trans.0.weight'].shape[0]
  dim_fourier = ms['single_conditioner.fourier_to_single.weight'].shape[1]
  depth = lambda sub, pre: 1 + max(int(k[len(pre):].split('.')[0])
                                   for k in sub if k.startswith(pre))
  n_enc = depth(cs, 'atom_enc_proj_z.')
  n_dec = depth(cs, 'atom_dec_proj_z.')
  n_tok_blocks = depth(cs, 'token_trans_proj_z.')
  h_enc = cs['atom_enc_proj_z.0.1.weight'].shape[0]
  h_dec = cs['atom_dec_proj_z.0.1.weight'].shape[0]
  h_tok = cs['token_trans_proj_z.0.1.weight'].shape[0]
  n_trans = depth(cs, 'pairwise_conditioner.transitions.')
  # relpos width from the init projection's norm: token_z + relpos.
  relp = cs['pairwise_conditioner.dim_pairwise_init_proj.0.weight'].shape[0] \
      - token_z
  print('  checkpoint: token_s %d, token_z %d, atom_s %d, atom_z %d, fourier '
        '%d, relpos %d, stacks %d/%d/%d, heads %d/%d/%d, transitions %d'
        % (token_s, token_z, atom_s, atom_z, dim_fourier, relp, n_enc,
           n_tok_blocks, n_dec, h_enc, h_tok, h_dec, n_trans))
  assert relp == token_z, (
      'this adapter feeds boltz its OWN relative-position features at token_z '
      'width; the checkpoint wants %d' % relp)

  # The two modules take DIFFERENT argument sets and only look alike: the
  # conditioner wants token_z and the atom-pair width (it builds the pair), the
  # score model wants dim_fourier (it builds the single conditioning) and no
  # token_z at all. Sharing one dict fires
  # `DiffusionModule.__init__() got an unexpected keyword argument 'token_z'`.
  kw = dict(token_s=token_s,
            atoms_per_window_queries=32, atoms_per_window_keys=128,
            atom_encoder_depth=n_enc, atom_encoder_heads=h_enc,
            token_transformer_depth=n_tok_blocks,
            token_transformer_heads=h_tok,
            atom_decoder_depth=n_dec, atom_decoder_heads=h_dec,
            conditioning_transition_layers=n_trans)
  cond = DiffusionConditioning(atom_s=atom_s, atom_z=atom_z, token_z=token_z,
                               atom_feature_dim=feat_dim, **kw)
  score = DiffusionModule(atom_s=atom_s, dim_fourier=dim_fourier, **kw)
  rp = RelativePositionEncoder(token_z=token_z)
  for mod, sub, label in ((cond, cs, 'DiffusionConditioning'),
                          (score, ms, 'score model'),
                          (rp, rs, 'RelativePositionEncoder')):
    missing, unexpected = mod.load_state_dict(sub, strict=False)
    print('  native %-24s %3d tensors, %d missing, %d unexpected %s'
          % (label, len(sub), len(missing), len(unexpected), list(missing)[:2]))
    assert not missing, '%s is missing %d tensors' % (label, len(missing))
    mod.eval()

  n_tok = _np.asarray(fb.token_features.mask).shape[0]
  bf, n_real, pad_to, r_dense = _boltz2_feats(fb, feats, pos_noisy, n_tok)
  # boltz's own relative-position features, from OUR batch -- the same rule
  # every adapter here follows, so a disagreement in the encoding is inside
  # the gate rather than attributed to the port.
  tf = fb.token_features
  rpf = {k: torch.tensor(_np.asarray(getattr(tf, k)).astype(_np.int64))[None]
         for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                   'sym_id')}
  rpf['mol_type'] = torch.zeros(1, n_tok, dtype=torch.long)
  t = lambda x: torch.tensor(_np.asarray(x, _np.float32))
  sigma_data = 16.0
  sigma = float(noise)
  c_in = 1.0 / _np.sqrt(sigma ** 2 + sigma_data ** 2)
  c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
  c_out = sigma * sigma_data / _np.sqrt(sigma_data ** 2 + sigma ** 2)
  c_noise = _np.log(sigma / sigma_data) * 0.25
  with torch.no_grad():
    relpos = rp(rpf)
    q, c, to_keys, aeb, adb, ttb = cond(
        s_trunk=t(s)[None], z_trunk=t(z)[None],
        relative_position_encoding=relpos, feats=bf)
    dc = {'q': q, 'c': c, 'to_keys': to_keys, 'atom_enc_bias': aeb,
          'atom_dec_bias': adb, 'token_trans_bias': ttb}
    r_update = score(
        s_inputs=t(s_inputs_449)[None], s_trunk=t(s)[None],
        r_noisy=r_dense * c_in,
        # `times` is Float[' b'] -- ONE per sample, not a broadcastable
        # (b, 1, 1). `FourierEmbedding.forward` does
        # `rearrange(times, "b -> b 1")`, which refuses anything else, and
        # `SingleConditioning` then broadcasts over tokens itself.
        times=torch.full((1,), c_noise, dtype=torch.float32),
        feats=bf, diffusion_conditioning=dc)
    x = c_skip * r_dense + c_out * r_update
  # The atom axis is packed in our order already; drop the window padding.
  return _np.asarray(x)[0][:n_real]


NATIVES = {m: native_protenix for m in _DIFF_SRC}
NATIVES['intellifold2'] = native_if2
NATIVES['rosettafold3'] = native_rf3
NATIVES.update({m: native_of3 for m in _OF3_CKPT})
NATIVES['boltz2'] = native_boltz2


def ours(model, cfg, model_dir, fb, pos_dense, noise, s_inputs, s, z):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import diffusion_head

  cfg.global_config.bfloat16 = os.environ.get('BF16', 'none')
  full = afp.get_model_haiku_params(model_dir=model_dir)
  emb = {'single': jnp.asarray(s), 'pair': jnp.asarray(z),
         'target_feat': jnp.asarray(s_inputs)}

  def fwd():
    return diffusion_head.DiffusionHead(
        cfg.heads.diffusion, cfg.global_config)(
            positions_noisy=jnp.asarray(pos_dense),
            noise_level=jnp.asarray(noise, jnp.float32),
            batch=fb, embeddings=emb, use_conditioning=True)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      # Calling DiffusionHead directly puts its own name at the FRONT of every
      # scope ('diffusion_head/relpe_projection'), where the conditioning
      # harness -- which calls the @hk.transparent _conditioning -- sees the
      # submodule alone. Strip it before prefixing, or all 124 scopes miss.
      tail = sc[len('diffusion_head/'):] if sc.startswith('diffusion_head/') \
          else sc
      key = ('diffuser/~/diffusion_head' if tail in ('~', 'diffusion_head')
             else 'diffuser/~/diffusion_head/' + tail)
      src = full.get(key)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v, np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our head is partly at init'
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

  seq, _ = fold_check.parse_ca(args.pdb)
  batch, cfg, model_dir = fold_check._fold_setup(args.model, seq,
                                                 args.model_dir)
  fb = feat_batch.Batch.from_data_dict(batch)
  feats = flat_atom_features(fb)
  n_tok = np.asarray(fb.token_features.mask).shape[0]
  max_atoms = feats['mask'].shape[1]
  print('%s one denoise step, %d tokens, %d atoms, noise %.1f:'
        % (args.model, n_tok, feats['ref_pos'].shape[0], args.noise))

  rng = np.random.default_rng(0)
  c_s, c_z = cfg.evoformer.seq_channel, cfg.evoformer.pair_channel
  s = (rng.normal(size=(n_tok, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n_tok, n_tok, c_z)) * 0.5).astype(np.float32)
  # s_inputs in BOTH layouts, as in conditioning_parity.
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s449 = (rng.normal(size=(n_tok, 449)) * 0.5).astype(np.float32)
  s449[:, np.setdiff1d(np.arange(449), idx)] = 0.0
  s447 = s449[:, idx]
  if args.model == 'boltz2':
    w = cfg.evoformer.seq_channel
    s449 = s447 = (rng.normal(size=(n_tok, w)) * 0.5).astype(np.float32)
    print('  s_inputs is %d wide for this model (s_trunk is concatenated with '
          'it, not with a 449-channel target_feat), and both sides get the '
          'same array' % w)
  # Noisy coordinates, dense and flat, the same numbers either way.
  pos_dense = (rng.normal(size=(n_tok, max_atoms, 3)) * args.noise
               ).astype(np.float32) * feats['mask'][..., None]
  pos_flat = pos_dense[feats['mask']]

  x_ref = NATIVES[args.model](args.model, fb, feats, pos_flat, args.noise,
                              s449, s, z)
  x_got = ours(args.model, cfg, model_dir, fb, pos_dense, args.noise, s447, s,
               z)
  x_got_flat = np.asarray(x_got)[feats['mask']]
  print('  shapes: ours %s -> flat %s | native %s'
        % (np.asarray(x_got).shape, x_got_flat.shape, x_ref.shape))
  _cmp('x_denoised', x_got_flat, x_ref)
  d = np.sqrt(((x_got_flat - x_ref) ** 2).sum(-1))
  print('  per-atom distance: mean %.4f A, max %.4f A, rms(native) %.2f'
        % (d.mean(), d.max(), np.sqrt((x_ref ** 2).sum(-1).mean())))
  if os.environ.get('DIAG'):
    # WHERE the disagreement sits, which separates the candidate causes:
    #  * concentrated at 32-atom window edges -> the windowed atom attention
    #    (which side pads, which clamps, and how the mask is built);
    #  * flat across positions but scaling with |x| -> a systematic term;
    #  * concentrated on particular tokens -> conditioning or pooling.
    # The window position must come from the DENSE layout, not from this flat
    # list. AtomCrossAtt windows over the flattened (num_token, max_atoms)
    # array INCLUDING its padding, and `feats['mask']` compresses that away --
    # so `arange(len(flat)) % 32` is the window position only if no token is
    # short of max_atoms, which is never true. Using it read edge/interior 1.54
    # for if2 and 1.74 for rf3 and sent both to the padded-key item; on the real
    # positions the ratio is what is printed below.
    n = d.shape[0]
    flat_mask = np.asarray(feats['mask'])
    if flat_mask.ndim > 1:
      pos = np.flatnonzero(flat_mask.ravel()) % 32
    else:
      pos = np.arange(d.shape[0]) % 32
    assert pos.shape == d.shape, (pos.shape, d.shape)
    edge = (pos < 4) | (pos >= 28)
    print('  DIAG window edge atoms (|pos mod 32 - 16| >= 12): mean %.4f  '
          'interior: mean %.4f  ratio %.2f'
          % (d[edge].mean(), d[~edge].mean(), d[edge].mean() / d[~edge].mean()))
    r = np.sqrt((x_ref ** 2).sum(-1))
    rel = d / np.maximum(r, 1e-6)
    print('  DIAG relative error: mean %.4f  median %.4f  '
          'corr(|d|, |x_native|) %.3f'
          % (rel.mean(), np.median(rel), np.corrcoef(d, r)[0, 1]))
    tok = np.asarray(feats['atom_to_token_idx'])
    per_tok = np.array([d[tok == i].mean() for i in np.unique(tok)])
    order = np.argsort(-per_tok)
    print('  DIAG worst tokens %s (mean %.3f) vs best %s (mean %.3f)'
          % (order[:5].tolist(), per_tok[order[:5]].mean(),
             order[-5:].tolist(), per_tok[order[-5:]].mean()))
    nat = np.asarray(fb.token_features.mask).shape[0]
    print('  DIAG per-token error spread: min %.3f  median %.3f  max %.3f '
          'over %d tokens' % (per_tok.min(), np.median(per_tok),
                              per_tok.max(), nat))
    # The LAST window is the only partially-padded one (n_atom is not a
    # multiple of 32), so if the two sides treat padded keys differently this
    # is where it shows. Dropping it separates that from a whole-model gap.
    last = (n // 32) * 32
    if last:
      print('  DIAG excluding the final partial window (atoms >= %d of %d): '
            'mean %.4f  vs %.4f inside it  (whole %.4f)'
            % (last, n, d[:last].mean(), d[last:].mean(), d.mean()))
    # Out-of-range KEYS occur at BOTH ends: every 32-atom query block attends to
    # 128 keys centred on it, so the first and last few blocks reach past the
    # sequence and the interior ones never do. Our windows clamp such a key onto
    # atom 0 (a real atom, repeated) where the vendors pad with zeros, both
    # sides masking those cells -- so if the mask is right the interior and the
    # ends agree equally, and if it is not, this split says so.
    if n > 320:
      inner = d[128:n - 128]
      ends = np.concatenate([d[:128], d[n - 128:]])
      print('  DIAG interior atoms [128, %d): mean %.4f  vs ends %.4f  '
            'ratio %.2f' % (n - 128, inner.mean(), ends.mean(),
                            ends.mean() / max(inner.mean(), 1e-9)))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
