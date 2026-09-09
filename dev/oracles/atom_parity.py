"""L2 (atom half): our atom cross-attention encoder against the vendor's own.

The last third of the L2 column. `conditioning_parity.py` covers the pair and
single conditioners and `diffusion_parity.py` the token transformer; this covers
the ATOM encoder, which is where `SWA_ROPE_ATOM_ATTENTION`, `ATOM_ROPE`,
`ATOM_ROPE_HALF_WINDOW`, `KEY_MASKED_ATOM_ATTENTION`, `NORMED_ATOM_FEATURES` and
`PER_BLOCK_ATOM_PAIR_LAYER_NORM` live -- six conventions with no
activation-level check on any protenix model.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:/home/ubuntu/protenix \
    python dev/oracles/atom_parity.py protenix2

The atom features and the window layout come from a REAL featurised 6MRR batch,
because a windowed local attention cannot be exercised on synthetic input: the
windows are built from `ref_space_uid` and the atom ordering. Our dense
(token, slot) layout flattens to native's flat atom list exactly -- AF3 builds
its own flat list as `token_atoms_layout[token_atoms_mask]`, i.e. row-major over
real atoms (features.py AtomCrossAtt.compute_features) -- so the two sides index
the same atoms, and the harness asserts that on ref_pos before comparing.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_parity import _cmp                      # noqa: E402
from diffusion_parity import _stub_layer_norm           # noqa: E402

_PROTENIX_CKPT = {
    'protenix2': 'protenix-v2.pt',
    'protenix1': 'protenix_base_default_v1.0.0.pt',
}

# Where each vendor's AtomAttentionENCODER lives -- the same table shape as
# `_DECODER_SRC` below, and for the same reason. opendde's encoder is
# protenix's: 98 tensors on both, identical leaf names, identical shape
# signature except c_z (128 against protenix2's 256), which the code reads off
# the checkpoint anyway. So it is one row, not a function.
_ENC_SRC = {
    'protenix1': ('protenix.model.modules.transformer',
                  '~/protenix_weights/protenix_base_default_v1.0.0.pt',
                  'module.diffusion_module.atom_attention_encoder.'),
    'protenix2': ('protenix.model.modules.transformer',
                  '~/protenix_weights/protenix-v2.pt',
                  'module.diffusion_module.atom_attention_encoder.'),
    'opendde': ('opendde.model.modules.transformer',
                '~/opendde_weights/opendde.pt',
                'module.diffusion_module.atom_attention_encoder.'),
}

# Tensors an adapter produces for an adapter downstream, NOT for a comparison.
# `native_if2` returns `p_lm` as None on purpose -- if2 windows the atom pair
# over an axis whose windows hold different atoms than ours, so comparing them
# would mislead -- but the DECODER consumes that same tensor as an input. Parked
# here rather than changing the return contract every caller reads.
_RAW = {}


def zero_other_features(fb, feats, keep):
  """Zero every per-atom reference feature except `keep`, on BOTH sides.

  The encoder's conditioning is a SUM of five independent projections
  (positions, charge, mask, element, atom-name chars), so zeroing four of them
  isolates the fifth: whichever one still disagrees is the one whose weights or
  convention are wrong. Beats reading five converter lines and guessing.
  """
  import dataclasses

  fields = {'pos': 'positions', 'charge': 'charge', 'mask': 'mask',
            'element': 'element', 'chars': 'atom_name_chars'}
  assert keep in fields, 'FEAT must be one of %s' % sorted(fields)
  rs = fb.ref_structure
  repl, out = {}, dict(feats)
  for k, field in fields.items():
    if k == keep:
      continue
    if field == 'mask':
      continue          # zeroing the mask would delete every atom
    repl[field] = np.zeros_like(np.asarray(getattr(rs, field)))
  fb = dataclasses.replace(fb, ref_structure=dataclasses.replace(rs, **repl))
  m = feats['mask']
  for k, field in fields.items():
    if k == keep or field == 'mask':
      continue
    src = {'positions': 'ref_pos', 'charge': 'ref_charge',
           'element': 'element', 'atom_name_chars': 'atom_name_chars'}[field]
    out[src] = np.zeros_like(out[src])
  print('  NOTE keeping only the %s feature' % keep)
  return fb, out


def flat_atom_features(fb):
  """-> dict of native-layout atom features, flattened from OUR batch."""
  rs = fb.ref_structure
  mask = np.asarray(rs.mask) > 0
  out = dict(
      ref_pos=np.asarray(rs.positions)[mask],
      ref_charge=np.asarray(rs.charge)[mask],
      ref_space_uid=np.asarray(rs.ref_space_uid)[mask],
      atom_name_chars=np.asarray(rs.atom_name_chars)[mask],
      element=np.asarray(rs.element)[mask],
  )
  n_tok = mask.shape[0]
  out['atom_to_token_idx'] = np.repeat(np.arange(n_tok), mask.sum(1))
  out['mask'] = mask
  return out


def native_protenix(model, fb, feats, pos_noisy, s, z, n_tok):
  """-> (a_token, q_l) from the vendor's own AtomAttentionEncoder.

  Serves protenix1, protenix2 and opendde off `_ENC_SRC` -- the module path is
  the only thing that differs, and `opendde.model.modules.transformer` is
  protenix's file with the package renamed.
  """
  import importlib

  import torch

  _stub_layer_norm()
  mod_name, ckpt_path, pre = _ENC_SRC[model]
  _vend = importlib.import_module(mod_name)
  AtomAttentionEncoder = _vend.AtomAttentionEncoder
  rearrange_qk_to_dense_trunk = _vend.rearrange_qk_to_dense_trunk

  ckpt = os.path.expanduser(ckpt_path)
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd.get('state_dict', sd))
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  c_atom = sub['linear_no_bias_ref_pos.weight'].shape[0]
  c_atompair = sub['linear_no_bias_d.weight'].shape[0]
  c_token = sub['linear_no_bias_q.weight'].shape[0]
  c_s = sub['linear_no_bias_s.weight'].shape[1]
  c_z = sub['linear_no_bias_z.weight'].shape[1]
  # The stack is nested one level deeper than the token transformer's:
  # `atom_transformer.diffusion_transformer.blocks.N....`. Derived by PRINTING
  # the keys -- guessing `atom_transformer.blocks.N` gave max() on an empty
  # sequence, which is what an assumption about a name looks like when it fails.
  bpre = 'atom_transformer.diffusion_transformer.blocks.'
  n_blocks = 1 + max(int(k[len(bpre):].split('.')[0]) for k in sub
                     if k.startswith(bpre))
  heads = sub[bpre + '0.attention_pair_bias.linear_nobias_z.weight'].shape[0]
  print('  checkpoint: c_atom %d, c_atompair %d, c_token %d, c_s %d, c_z %d, '
        '%d blocks, %d heads'
        % (c_atom, c_atompair, c_token, c_s, c_z, n_blocks, heads))

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  n_atom = feats['ref_pos'].shape[0]
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

  # protenix's own windowing, from protenix/model/protenix.py.
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

  # HOW MANY LEADING AXES the windowed atom-pair tensor and the trunk inputs
  # carry. Both vendors' local attention asserts
  # `len(z.shape) == len(q.shape) + 2` (transformer.py:714), where z is the
  # WINDOWED atom pair and q the per-atom activation -- but they build the atom
  # pair from d_lm/v_lm differently, so the same inputs give different ranks.
  # protenix expands it internally and wants two leading axes on the trunk
  # tensors; opendde does not, so the windowed pair has to be batched here and
  # the trunk tensors carry one. Derived by reading the assert, not by guessing:
  # 0, 1 and 2 leading axes on the trunk tensors alone all fire it.
  _n_lead = int(os.environ.get('N_LEAD', 1 if model == 'opendde' else 2))
  net = AtomAttentionEncoder(c_atom=c_atom, c_atompair=c_atompair,
                             c_token=c_token, c_s=c_s, c_z=c_z,
                             n_blocks=n_blocks, n_heads=heads,
                             n_queries=32, n_keys=128, has_coords=True)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    a, q_l, c_l, p_lm = net(
        atom_to_token_idx=t(feats['atom_to_token_idx'], torch.long),  # 1-D
        ref_pos=t(feats['ref_pos'])[None],
        ref_charge=t(feats['ref_charge'])[None],
        ref_mask=torch.ones(1, n_atom),
        ref_atom_name_chars=t(ref_chars)[None],
        ref_element=t(ref_element)[None],
        d_lm=d_lm, v_lm=v_lm, pad_info=pad_info,
        # HOW MANY LEADING AXES. protenix's encoder wants two (a diffusion-batch
        # axis inside a batch axis) and threads both through to the atom pair,
        # so its local attention's `len(z.shape) == len(q.shape) + 2` holds.
        # opendde carries the same assert but does NOT thread the second axis
        # into p_lm, so two axes fire it from inside the vendor's code
        # (transformer.py:714). One axis is what its own pipeline passes.
        **{k: (t(v)[None][None] if _n_lead == 2 else
               t(v)[None] if _n_lead == 1 else t(v))
           for k, v in (('r_l', pos_noisy), ('s', s), ('z', z))})
  # c_l and p_lm are the encoder's INPUTS -- the per-atom conditioning and the
  # windowed atom-pair features. Returned so the harness can tell "our features
  # differ" from "our atom stack differs", which the outputs alone cannot.
  # pad_info's mask says which (block, query, key) slots hold REAL atom pairs.
  # Without it the p_lm comparison is dominated by the out-of-range slots, where
  # AF3 clamps the key onto atom 0 (a real atom, repeated) and protenix pads --
  # different numbers in cells both sides mask out of the attention anyway.
  pm = None
  for k in ('mask_trunked', 'mask', 'pad_mask'):
    if isinstance(pad_info, dict) and k in pad_info:
      pm = np.asarray(pad_info[k])
      break
  return (np.asarray(a)[0, 0], np.asarray(q_l)[0, 0],
          np.asarray(c_l).reshape(-1, np.asarray(c_l).shape[-1]),
          np.asarray(p_lm).reshape(*np.asarray(p_lm).shape[-4:]), pm)


_OF3_CKPT = {'openfold3': 'of3-p2-155k.pt', 'openbind0': 'of3-ob-174k.pt'}


def _config_dict(**kw):
  import ml_collections
  return ml_collections.ConfigDict(kw)


def native_of3(model, fb, feats, pos_noisy, s, z, n_tok):
  """-> (a, q_l, c_l, p_lm, pad_mask) from OpenFold3's own AtomAttentionEncoder.

  Simpler to drive than protenix's: of3 builds its own windows inside forward,
  so there is no d_lm / v_lm / pad_info to reconstruct -- just a feature dict.
  Construction args come from of3's own model_config (c_hidden 32, 4 heads, 3
  blocks, n_query 32 / n_key 128, `c_atom_ref` element 119 + name_chars 256);
  the widths still come off the checkpoint, so a release that changed one fails
  in load_state_dict rather than comparing quietly.
  """
  import torch

  from openfold3.core.model.layers.sequence_local_atom_attention import (
      AtomAttentionEncoder)

  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.atom_attn_enc.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  bpre = 'atom_transformer.blocks.'
  n_blocks = 1 + max(int(k[len(bpre):].split('.')[0]) for k in sub
                     if k.startswith(bpre))
  c_atom_pair = sub['atom_transformer.layer_norm_z.weight'].shape[0]
  c_atom = sub[bpre + '0.attention_pair_bias.mha.linear_q.weight'].shape[0]
  # of3 names the atom->token projection `linear_q.0` (a Sequential), not
  # `linear_q_out`; the decoder is the one with `linear_q_out`. Printed off the
  # checkpoint, not guessed -- guessing gave KeyError on both releases.
  c_token = sub['linear_q.0.weight'].shape[0]
  c_s = sub['noisy_position_embedder.linear_s.weight'].shape[1]
  c_z = sub['noisy_position_embedder.linear_z.weight'].shape[1]
  heads = sub[bpre + '0.attention_pair_bias.linear_z.weight'].shape[0]
  n_elem = sub['ref_atom_feature_embedder.linear_ref_element.weight'].shape[1]
  print('  checkpoint: c_atom %d, c_atom_pair %d, c_token %d, c_s %d, c_z %d, '
        '%d blocks, %d heads, %d element classes'
        % (c_atom, c_atom_pair, c_token, c_s, c_z, n_blocks, heads, n_elem))

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  n_atom = feats['ref_pos'].shape[0]
  # of3 carries 119 element classes to AF3's 128, and its own `-1` indexing --
  # the same shift protenix uses, which `fold_element_index_shift` folds into
  # the converted rows.
  el = np.clip(np.asarray(feats['element']).astype(int) - 1, 0, n_elem - 1)
  ref_element = np.zeros((n_atom, n_elem), np.float32)
  ref_element[np.arange(n_atom), el] = 1.0
  chars = np.asarray(feats['atom_name_chars']).astype(int)
  ref_chars = np.zeros((n_atom, 4, 64), np.float32)
  for i in range(4):
    ref_chars[np.arange(n_atom), i, chars[:, i]] = 1.0
  mask = feats['mask']
  batch = {
      'ref_pos': t(feats['ref_pos'])[None],
      'ref_mask': torch.ones(1, n_atom),
      'ref_element': t(ref_element)[None],
      'ref_charge': t(feats['ref_charge'])[None],
      'ref_atom_name_chars': t(ref_chars)[None],
      'ref_space_uid': t(feats['ref_space_uid'], torch.long)[None],
      'token_mask': torch.ones(1, n_tok),
      'atom_mask': torch.ones(1, n_atom),
      'num_atoms_per_token': t(mask.sum(1), torch.long)[None],
      'atom_to_token_index': t(feats['atom_to_token_idx'], torch.long)[None],
  }
  net = AtomAttentionEncoder(
      # of3 reads c_atom_ref with attribute access, so a plain dict fails --
      # it wants an ml_collections ConfigDict.
      c_atom_ref=_config_dict(element=n_elem, name_chars=256), c_atom=c_atom,
      c_atom_pair=c_atom_pair, c_token=c_token, add_noisy_pos=True,
      c_hidden=32, no_heads=heads, no_blocks=n_blocks, n_transition=2,
      n_query=32, n_key=128, use_ada_layer_norm=True, c_s=c_s, c_z=c_z,
      blocks_per_ckpt=None, inf=1e9)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    a, q_l, c_l, p_lm = net(batch, rl=t(pos_noisy)[None],
                            si_trunk=t(s)[None], zij_trunk=t(z)[None])
  return (np.asarray(a)[0], np.asarray(q_l)[0],
          np.asarray(c_l).reshape(-1, np.asarray(c_l).shape[-1]),
          np.asarray(p_lm).reshape(*np.asarray(p_lm).shape[-4:]), None)


def native_if2(model, fb, feats, pos_noisy, s, z, n_tok):
  """-> (a, q_l, c_l, p_lm, pad_mask) from IntelliFold-2's own AtomAttentionEncoder.

  Driven like of3's -- if2 windows the atoms inside forward too -- with four
  differences that are vendor facts, not choices:
    * it has TWO atom layouts and the CONFIG picks one -- see `packed` below;
      the v2 inference config packs, which is ours;
    * it takes flat tensors, not a feature dict, plus `molecule_atom_lens`;
    * 128 element classes on AF3's own indexing, so NO `-1` shift here. That is
      why `converters/intellifold2.py` is the one converter in the family with
      no `fold_element_index_shift` call -- the two facts have to agree;
    * it applies `asinh` to the charge itself, so the raw charge goes in.
  Construction args come from if2's own config (c_ref 3+1+128+1+256, 3 blocks,
  4 heads, window 32x128); the widths still come off the checkpoint.
  """
  import torch

  from intellifold.openfold.model.diffusion import AtomAttentionEncoder

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.atom_attention_encoder.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  bpre = 'atom_transformer.blocks.'
  n_blocks = 1 + max(int(k[len(bpre):].split('.')[0]) for k in sub
                     if k.startswith(bpre))
  c_atom = sub['linear_ref_pos.weight'].shape[0]
  c_atompair = sub['linear_d.weight'].shape[0]
  c_token = sub['pool_q.linear_q.weight'].shape[0]
  c_s = sub['linear_s.weight'].shape[1]
  c_z = sub['linear_z.weight'].shape[1]
  n_elem = sub['linear_ref_element.weight'].shape[1]
  heads = 4
  print('  checkpoint: c_atom %d, c_atompair %d, c_token %d, c_s %d, c_z %d, '
        '%d blocks, %d element classes'
        % (c_atom, c_atompair, c_token, c_s, c_z, n_blocks, n_elem))

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  mask = feats['mask']
  # if2 has TWO atom->token broadcasts. The default `repeat_consecutive_with_lens`
  # ignores the lens and repeats each token 24 times, keeping the padding
  # interleaved; `_advanced` honours the lens and packs the real atoms, which is
  # AF3's layout and ours. `v2_inference_config.py` turns `advanced_conversion`
  # ON for the atom encoder, decoder and input embedder, so PACKED is what this
  # checkpoint actually runs -- and the default here. IF2_DENSE=1 selects the
  # other one, which is how the two were told apart: dense scores a_token 0.965,
  # packed 0.999999 on the same weights and the same inputs.
  packed = not os.environ.get('IF2_DENSE')
  # Built dense first -- every (token, slot) including padding, row-major --
  # then packed below when `packed`, because the two if2 layouts differ only in
  # whether the padding slots stay in.
  rs = fb.ref_structure
  n_atom = mask.size
  dense = lambda a: np.asarray(a).reshape(n_atom, *np.asarray(a).shape[2:])
  d_pos, d_charge = dense(rs.positions), dense(rs.charge)
  d_uid, d_mask = dense(rs.ref_space_uid), mask.reshape(-1).astype(np.float32)
  el = np.clip(dense(rs.element).astype(int), 0, n_elem - 1)
  ref_element = np.zeros((n_atom, n_elem), np.float32)
  ref_element[np.arange(n_atom), el] = 1.0
  chars = dense(rs.atom_name_chars).astype(int)
  ref_chars = np.zeros((n_atom, 4, 64), np.float32)
  for i in range(4):
    ref_chars[np.arange(n_atom), i, chars[:, i]] = 1.0
  # The noisy coordinates arrive packed (native protenix/of3 want them that
  # way); scatter them back into the dense slots for if2.
  d_r = np.zeros((n_atom, 3), np.float32)
  d_r[mask.reshape(-1)] = pos_noisy
  if packed:
    k = mask.reshape(-1)
    d_pos, d_charge, d_uid = d_pos[k], d_charge[k], d_uid[k]
    ref_element, ref_chars = ref_element[k], ref_chars[k]
    d_mask, d_r = d_mask[k], d_r[k]
    n_atom = int(k.sum())
    print('  packed layout (v2 inference config): %d atoms, ours on both sides'
          % n_atom)
  net = AtomAttentionEncoder(
      c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, c_s=c_s, c_z=c_z,
      c_ref=3 + 1 + n_elem + 1 + 4 * 64, no_blocks=n_blocks, no_heads=heads,
      window_size_row=32, window_size_col=128, initial=False,
      advanced_conversion=packed)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  # BLOCKS truncates BOTH sides (ours in `ours()`), which is what separates a
  # per-block difference from its accumulation: a tail difference in the key
  # window propagates one window backwards per block, so three blocks turn an
  # edge into what looks like a gradient.
  _nb = os.environ.get('BLOCKS')
  if _nb:
    net.atom_transformer.blocks = net.atom_transformer.blocks[:int(_nb)]
    print('  native atom stack truncated to %s block(s)' % _nb)
  net.eval()
  with torch.no_grad():
    a, q_l, c_l, p_lm = net(
        ref_pos=t(d_pos)[None], ref_charge=t(d_charge)[None],
        ref_mask=t(d_mask)[None], ref_element=t(ref_element)[None],
        ref_atom_name_chars=t(ref_chars)[None],
        ref_space_uid=t(d_uid, torch.long)[None],
        atom_mask=t(d_mask)[None], s_trunk=t(s)[None], z=t(z)[None],
        r=t(d_r)[None], molecule_atom_lens=t(mask.sum(1), torch.long)[None])
  q_l, c_l = np.asarray(q_l), np.asarray(c_l)
  # Back to OUR layout for the comparison. Ours packs the real atoms first and
  # pads at the end (which is why truncating to the real atom count is what the
  # packed-native adapters do); if2's dense axis has the padding interleaved,
  # 24 slots per token. Selecting its real slots row-major gives the same 574
  # atoms in the same order. `p_lm` is windowed over the DENSE axis, so its
  # windows hold different atoms than ours -- not comparable, and returned as
  # None rather than compared misleadingly.
  keep = np.ones(q_l.shape[-2], bool) if packed else mask.reshape(-1)
  _RAW['if2_p'] = np.asarray(p_lm)
  return (np.asarray(a)[0], q_l.reshape(-1, q_l.shape[-1])[keep],
          c_l.reshape(-1, c_l.shape[-1])[keep], None, None)


def native_rf3(model, fb, feats, pos_noisy, s, z, n_tok):
  """-> (a, q_l, c_l, None, None) from RoseTTAFold3's own AtomAttentionEncoderDiffusion.

  rf3's encoder is the packed-atom kind (`atom_to_token_map` drives everything),
  so no layout surgery -- but three vendor facts shape it:
    * **all 1d atom features are ONE fused linear of 393 columns**, in the order
      `converters/rosettafold3.py` splits them: pos(3), charge(1), mask(1),
      element(128), name chars(256), ref_pos_ground_truth(3),
      has_atom_level_embedding(1). The last two are not features we carry, and
      the converter drops those columns, so they are fed as zeros here -- the
      same thing the port does.
    * element is NOT shifted (128 classes on AF3's indexing), matching the
      absence of a `fold_element_index_shift` call in the rf3 converter.
    * `use_chiral_features` is ON in rf3's config and `process_ch` IS in the
      checkpoint -- it is the one rf3 term the port does not implement. This
      gate runs with it OFF and drops that tensor, so it measures everything
      else exactly; RF3_CHIRAL=1 turns it back on to size the missing term.
  Weights come from the `shadow.` EMA copy, which is what the converter reads.
  """
  import torch

  from rf3.model.layers.af3_diffusion_transformer import (
      AtomAttentionEncoderDiffusion)

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'shadow.diffusion_module.atom_attention_encoder.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  chiral = bool(os.environ.get('RF3_CHIRAL'))
  if not chiral:
    sub.pop('process_ch.weight', None)
  n_feat = sub['process_input_features.weight'].shape[1]
  c_atom = sub['process_input_features.weight'].shape[0]
  c_atompair = sub['process_d.weight'].shape[0]
  c_token = sub['process_q.0.weight'].shape[0]
  c_s = sub['process_s_trunk.1.weight'].shape[1]
  c_tokenpair = sub['process_z.1.weight'].shape[1]
  bpre = 'atom_transformer.diffusion_transformer.blocks.'
  n_blocks = 1 + max([int(k[len(bpre):].split('.')[0]) for k in sub
                      if k.startswith(bpre)] or [2])
  print('  checkpoint: c_atom %d, c_atompair %d, c_token %d, c_s %d, '
        'c_tokenpair %d, %d blocks, %d fused atom features, chiral %s'
        % (c_atom, c_atompair, c_token, c_s, c_tokenpair, n_blocks, n_feat,
           'ON' if chiral else 'OFF (dropped)'))

  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  n_atom = feats['ref_pos'].shape[0]
  el = np.clip(np.asarray(feats['element']).astype(int), 0, 127)
  ref_element = np.zeros((n_atom, 128), np.float32)
  ref_element[np.arange(n_atom), el] = 1.0
  chars = np.asarray(feats['atom_name_chars']).astype(int)
  ref_chars = np.zeros((n_atom, 4, 64), np.float32)
  for i in range(4):
    ref_chars[np.arange(n_atom), i, chars[:, i]] = 1.0
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
  # The checkpoint carries `process_atom_level_embedding.*` -- rf3's conformer
  # embedding, which the yaml in the repo does not switch on but these weights
  # have. Leaving it out is not neutral: the MLP has biases and a LayerNorm
  # tail, so on the all-zero conformer input it still emits a fixed NONZERO
  # vector, which `converters/rosettafold3.py::_conformer_embedding_bias` folds
  # into our embeddings as a constant. Omitting the module here made native
  # SMALLER than ours (c_atom_cond 0.939, rms ratio 2.44) -- the harness
  # missing a term the port has, the mirror image of the usual fault.
  ale = 'process_atom_level_embedding.process_atom_level_embedding.0.weight'
  ale_dim = sub[ale].shape[1] if ale in sub else 384
  ale = ale in sub
  if ale:
    f['atom_level_embedding'] = torch.zeros(8, n_atom, ale_dim)
  names = ['ref_pos', 'ref_charge', 'ref_mask', 'ref_element',
           'ref_atom_name_chars', 'ref_pos_ground_truth',
           'has_atom_level_embedding'][:None if n_feat == 393 else 5]
  net = AtomAttentionEncoderDiffusion(
      c_atom=c_atom, c_atompair=c_atompair, c_token=c_token,
      c_tokenpair=c_tokenpair, c_s=c_s, atom_1d_features=names,
      c_atom_1d_features=n_feat,
      atom_transformer=dict(
          n_queries=32, n_keys=128,
          diffusion_transformer=dict(
              n_block=n_blocks,
              diffusion_transformer_block=dict(
                  n_head=4,
                  no_residual_connection_between_attention_and_transition=True,
                  kq_norm=True))),
      broadcast_trunk_feats_on_1dim_old=False, use_chiral_features=chiral,
      no_grad_on_chiral_center=False, use_inv_dist_squared=True,
      use_atom_level_embedding=ale, atom_level_embedding_dim=ale_dim)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  if unexpected:
    print('  native UNEXPECTED (in checkpoint, not in the module we built): %s'
          % sorted(unexpected)[:12])
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    # The coordinates carry rf3's leading DIFFUSION-batch dim, the trunk
    # tensors do NOT: `atom_attention` normalises A_I (adds the dim if it is
    # missing) but unsqueezes the pair tensor unconditionally, `Z_II[None]`, so
    # a batched z arrives 5-D and the windowed gather indexes the batch axis
    # ("index 249 is out of bounds for dimension 0 with size 1").
    a, q_l, c_l, _ = net(f, t(pos_noisy)[None], t(s), t(z))
  # rf3 keeps the atom pair tensor dense over all L*L, ours is windowed --
  # not comparable, so it is not compared.
  sq = lambda x: np.asarray(x).reshape(np.asarray(x).shape[-2:])
  return (sq(a), sq(q_l), sq(c_l), None, None)


def native_esmfold2(model, fb, feats, pos_noisy, s, z, n_tok):
  """-> (a_token, enc_queries, c0, None, None) from ESMFold2's REFERENCE.

  ESMFold2 has no importable vendor module, so the oracle is
  `esmfold2_reference.py` -- the self-contained reimplementation the port was
  built against, itself validated on native dumps. Its atom encoder is
      c0 = LN(atom_features @ atom_linear)
      q  = c0 + [r_noisy | 0] @ coords_linear
      q  = atom_stack(q, c0, ...)                 # conditioned on c0 ALONE
      a  = scatter_mean(relu(q @ atom_to_token))
  which is exactly this gate's three tensors. There is no atom PAIR
  representation -- its atom attention is positional through rotary alone --
  hence the two Nones.

  The noisy coordinates come from the HARNESS, in our dense layout, and are
  scattered into the dump's flat layout through the name-based atom
  correspondence (esmfold2_dumps.atom_map). Aligning those two by position
  instead is what made the denoise localiser compare a permutation.
  """
  import jax
  import jax.numpy as jnp

  import esmfold2_dumps
  import esmfold2_reference as R
  from converters import esmfold2 as CV

  sd = esmfold2_dumps.state_dict(model)
  dims = CV.derive_dims(sd)
  dims['n_input_atom'] = 3
  # BLOCKS truncates the atom stack. TRUNCATE BOTH SIDES -- ours is sliced in
  # `ours()` below; here it is just a smaller loop count.
  if os.environ.get('BLOCKS'):
    dims['n_diff_atom'] = int(os.environ['BLOCKS'])
  pref = {k: jnp.asarray(v) for k, v in CV.map_esmfold2_to_af3(sd).items()}
  ae = {k[len('diffusion/atom_encoder/'):]: v for k, v in pref.items()
        if k.startswith('diffusion/atom_encoder/')}
  nat = esmfold2_dumps.native(model)
  f = {k[5:]: jnp.asarray(v[0]) for k, v in nat.items() if k.startswith('feat.')}
  mask = f['atom_attention_mask']
  ref_idx, our_flat = esmfold2_dumps.atom_map(fb, f)
  print('  reference: %d atom blocks, %d atoms matched by name'
        % (dims['n_diff_atom'], len(ref_idx)))

  # our dense (num_token, max_atoms, 3) noise -> the dump's flat layout
  dense = np.zeros(np.asarray(feats['mask']).shape + (3,), np.float32)
  dense[np.asarray(feats['mask'])] = pos_noisy
  x = np.zeros(np.asarray(mask).shape + (3,), np.float32)
  x.reshape(-1, 3)[ref_idx] = dense.reshape(-1, 3)[our_flat]

  c0 = R.layer_norm(R.atom_features(f, mask) @ ae['atom_linear/weights'],
                    ae['atom_norm/scale'], ae['atom_norm/offset'])
  cos, sin = R.build_rope(f['ref_pos'], f['ref_space_uid'], c0.shape[-1] // 4)
  # NO c_in HERE. This gate's convention is that `pos_noisy` is ALREADY the
  # scaled input -- `ours()` hands it to the encoder untouched, and every other
  # adapter does the same. Dividing by sqrt(sigma^2 + sigma_data^2) again made
  # the reference's coordinates 17.9x smaller than ours and read as a 3.1x
  # magnitude blow-up in our atom stack (q_atom rms 3.1219, a_token 0.7497).
  r_noisy = jnp.asarray(x)
  q = c0 + jnp.concatenate([r_noisy, jnp.zeros_like(r_noisy)], -1) @ ae['coords_linear/weights']
  q = R.atom_stack(q, c0, ae, 'blocks/', dims['n_diff_atom'], cos, sin, mask)
  a2t = jnp.asarray(np.asarray(f['atom_to_token']).astype(int)
                    * np.asarray(mask).astype(int))
  a = R.scatter_mean(jax.nn.relu(q @ ae['atom_to_token/weights']), a2t,
                     int(np.asarray(fb.token_features.mask).shape[0]), mask)
  sel = lambda v: np.asarray(v).reshape(-1, np.asarray(v).shape[-1])[ref_idx]
  return (np.asarray(a), sel(q), sel(c0), None, None)


_BOLTZ2_CKPT = '~/boltz2_weights/boltz2_conf.ckpt'


def _boltz2_modules(need_decoder=False):
  """-> the vendor's atom modules, loaded, plus the widths they were built from.

  Every width and depth is read off the checkpoint. Verified on CPU before the
  gate was wired: `AtomEncoder` 16 tensors, `AtomAttentionEncoder` 71,
  `AtomAttentionDecoder` 73, both bias stacks 9, and all five load with nothing
  missing and nothing unexpected.

  boltz2 splits AF3's atom encoder in two, and the split is the reason this is
  one helper rather than two adapters: `diffusion_conditioning.atom_encoder`
  (an `AtomEncoder`) builds q / c / p from the reference features, and
  `structure_module.score_model.atom_attention_encoder` (an
  `AtomAttentionEncoder`) runs the stack. The per-block pair BIAS is a third
  thing again -- `diffusion_conditioning.atom_enc_proj_z`, a ModuleList of
  (LayerNorm, Linear -> heads) applied to p and concatenated -- where AF3
  projects the pair inside the stack. Our graph fuses all three, so the gate
  has to assemble them here.
  """
  import numpy as _np
  import torch
  import torch.nn as nn

  from boltz.model.modules.encodersv2 import (AtomAttentionDecoder,
                                              AtomAttentionEncoder,
                                              AtomEncoder)

  ckpt = os.path.expanduser(_BOLTZ2_CKPT)
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = raw.get('state_dict', raw.get('model', raw))
  take = lambda pre: {k[len(pre):]: v for k, v in sd.items()
                      if k.startswith(pre)}
  enc_pre = 'diffusion_conditioning.atom_encoder.'
  aae_pre = 'structure_module.score_model.atom_attention_encoder.'
  aad_pre = 'structure_module.score_model.atom_attention_decoder.'
  bias_pre = 'diffusion_conditioning.atom_enc_proj_z.'
  dbias_pre = 'diffusion_conditioning.atom_dec_proj_z.'
  e, a, d = take(enc_pre), take(aae_pre), take(aad_pre)
  if not e:
    raise SystemExit('no %r keys in %s' % (enc_pre, ckpt))

  atom_s, feat_dim = e['embed_atom_features.weight'].shape
  atom_z = e['embed_atompair_ref_pos.weight'].shape[0]
  token_s = e['s_to_c_trans.0.weight'].shape[0]
  token_z = e['z_to_p_trans.0.weight'].shape[0]
  depth = lambda pre: 1 + max(int(k[len(pre):].split('.')[0])
                              for k in sd if k.startswith(pre))
  n_enc, n_dec = depth(bias_pre), depth(dbias_pre)
  heads = sd[bias_pre + '0.1.weight'].shape[0]
  d_heads = sd[dbias_pre + '0.1.weight'].shape[0]
  print('  checkpoint: atom_s %d, atom_z %d, token_s %d, token_z %d, '
        'atom_feature_dim %d, encoder %dx%d heads, decoder %dx%d heads'
        % (atom_s, atom_z, token_s, token_z, feat_dim, n_enc, heads,
           n_dec, d_heads))
  # The feature width says which optional atom features this release uses, and
  # all three flags default off: 3 + 1 + 128 + 4*64 = 388 is position, charge,
  # element and the four atom-name characters and nothing else. A release that
  # turned on `use_atom_backbone_feat` or `use_residue_feats_atoms` would widen
  # this, and then the feature vector built below would be wrong -- so it is
  # asserted rather than assumed.
  assert feat_dim == 3 + 1 + 128 + 4 * 64, (
      'boltz2 atom_feature_dim is %d, not the 388 this adapter builds; the '
      'release has an optional atom feature turned on' % feat_dim)

  enc = AtomEncoder(atom_s=atom_s, atom_z=atom_z, token_s=token_s,
                    token_z=token_z, atoms_per_window_queries=32,
                    atoms_per_window_keys=128, atom_feature_dim=feat_dim,
                    structure_prediction=True)
  aae = AtomAttentionEncoder(atom_s=atom_s, token_s=token_s,
                             atoms_per_window_queries=32,
                             atoms_per_window_keys=128,
                             atom_encoder_depth=n_enc, atom_encoder_heads=heads,
                             structure_prediction=True)
  mods = [(enc, e, 'AtomEncoder'), (aae, a, 'AtomAttentionEncoder')]
  aad = None
  if need_decoder:
    aad = AtomAttentionDecoder(atom_s=atom_s, token_s=token_s,
                               attn_window_queries=32, attn_window_keys=128,
                               atom_decoder_depth=n_dec,
                               atom_decoder_heads=d_heads)
    mods.append((aad, d, 'AtomAttentionDecoder'))

  def _bias(pre, n, h):
    stack = nn.ModuleList([nn.Sequential(nn.LayerNorm(atom_z),
                                         nn.Linear(atom_z, h, bias=False))
                           for _ in range(n)])
    mods.append((stack, take(pre), pre.rstrip('.').split('.')[-1]))
    return stack

  enc_bias = _bias(bias_pre, n_enc, heads)
  dec_bias = _bias(dbias_pre, n_dec, d_heads) if need_decoder else None

  for mod, sub, label in mods:
    missing, unexpected = mod.load_state_dict(sub, strict=False)
    print('  native %-22s %3d tensors, %d missing, %d unexpected %s'
          % (label, len(sub), len(missing), len(unexpected),
             list(missing)[:2]))
    assert not missing, '%s is missing %d tensors' % (label, len(missing))
    mod.eval()
  return dict(enc=enc, aae=aae, aad=aad, enc_bias=enc_bias,
              dec_bias=dec_bias, token_s=token_s, token_z=token_z,
              atom_s=atom_s, atom_z=atom_z)


def _boltz2_feats(fb, feats, pos_noisy, n_tok):
  """-> (boltz feature dict, real atom count, padded length, noisy coords).

  PACKED, not dense. boltz2's own featuriser emits a packed atom list -- all
  real atoms contiguous, then padding to a multiple of the query window -- and
  its `atom_to_token` matrix and `atom_pad_mask` are built from that. Feeding it
  the DENSE (token, slot) layout instead puts padding BETWEEN real atoms, so its
  windows hold different neighbours than ours even though the returned per-atom
  tensors can still be indexed back into agreement: q read corr 0.847 and the
  atom pair 0.609 that way, with c_atom_cond (which has no window) at 0.999999
  -- the signature of a layout difference rather than an arithmetic one.

  The one hazard is the window arithmetic: `AtomEncoder` computes K = N // W
  with W = 32 and reshapes, so N must be a MULTIPLE of 32. 6MRR happens to
  satisfy that (68 tokens * 24 slots = 1632 = 51 * 32) and 1STP does not
  (121 * 24 = 2904), so the dense axis is padded up here and the outputs are
  cut back down. boltz's own featuriser pads for the same reason.
  """
  import numpy as _np
  import torch

  rs = fb.ref_structure
  mask = _np.asarray(feats['mask'])
  n_slot = mask.shape[1]
  n_dense = n_tok * n_slot
  real = mask.reshape(-1)
  n_real = int(real.sum())
  # Dense first (every (token, slot), row-major), then PACKED by the mask, which
  # is the order our own queries layout uses. `AtomEncoder` computes K = N // 32
  # and reshapes, so the packed length is padded up to a whole window.
  keep = _np.flatnonzero(real)
  dense = lambda x: _np.asarray(x).reshape(n_dense,
                                           *_np.asarray(x).shape[2:])[keep]
  pad_to = -(-n_real // 32) * 32
  pad = pad_to - n_real
  print('  packed layout: %d real atoms padded to %d (boltz windows N // 32)'
        % (n_real, pad_to))

  def _p(x, fill=0.0):
    x = _np.asarray(x, _np.float32)
    if not pad:
      return x
    z = _np.full((pad,) + x.shape[1:], fill, _np.float32)
    return _np.concatenate([x, z], axis=0)

  el = _np.clip(dense(rs.element).astype(int), 0, 127)
  elem = _np.zeros((n_real, 128), _np.float32)
  elem[_np.arange(n_real), el] = 1.0
  chars = dense(rs.atom_name_chars).astype(int)
  ref_chars = _np.zeros((n_real, 4, 64), _np.float32)
  for i in range(4):
    ref_chars[_np.arange(n_real), i, chars[:, i]] = 1.0
  # The noisy coordinates already arrive packed, in this same order.
  d_r = _np.asarray(pos_noisy, _np.float32)
  # atom -> token as a one-hot MATRIX: boltz pools with a bmm against its
  # column-normalised transpose, not a gather through an index.
  a2t = _np.zeros((n_real, n_tok), _np.float32)
  a2t[_np.arange(n_real),
      _np.repeat(_np.arange(n_tok), n_slot)[keep]] = 1.0
  # A padding atom must not join a real window's `v`: uid equality is ANDed
  # with the masks, but a distinct uid makes it true by construction rather
  # than by the mask alone.
  uid = dense(rs.ref_space_uid).astype(_np.float32)
  t = lambda x, dt=torch.float32: torch.tensor(_np.asarray(x), dtype=dt)
  bf = {
      'ref_pos': t(_p(dense(rs.positions)))[None],
      'ref_charge': t(_p(dense(rs.charge)))[None],
      'ref_element': t(_p(elem))[None],
      'ref_atom_name_chars': t(_p(ref_chars))[None],
      'ref_space_uid': t(_p(uid, -1.0), torch.long)[None],
      'atom_pad_mask': t(_p(_np.ones(n_real, _np.float32)))[None],
      'atom_to_token': t(_p(a2t))[None],
  }
  return bf, n_real, pad_to, t(_p(d_r))[None]


def native_boltz2(model, fb, feats, pos_noisy, s, z, n_tok):
  """-> (a, q_l, c_l, p_lm, pad_mask) from Boltz-2's own atom encoder."""
  import numpy as _np
  import torch

  _stub_layer_norm()
  M = _boltz2_modules()
  bf, n_real, pad_to, r = _boltz2_feats(fb, feats, pos_noisy, n_tok)
  t = lambda x: torch.tensor(_np.asarray(x, _np.float32))
  with torch.no_grad():
    q, c, p, to_keys = M['enc'](feats=bf, s_trunk=t(s)[None], z=t(z)[None])
    bias = torch.cat([l(p) for l in M['enc_bias']], dim=-1)
    a, q_out, c_out, _ = M['aae'](feats=bf, q=q, c=c, atom_enc_bias=bias,
                                  to_keys=to_keys, r=r)
  _RAW['boltz2'] = dict(p=p, q=q_out, c=c_out, to_keys=to_keys,
                        bf=bf, n_real=n_real)
  # The atom axis is already packed in our order; drop the window padding.
  qn = _np.asarray(q_out)[0][:n_real]
  cn = _np.asarray(c_out)[0][:n_real]
  # WHICH (block, query, key) SLOTS HOLD A REAL ATOM PAIR. boltz clips and pads
  # its window and then masks, so the padded slots hold index 0 -- a real atom,
  # repeated -- on both sides but are excluded from attention on both sides. A
  # p_atom_pair comparison over all slots is dominated by numbers neither model
  # looks at; this is the same mask protenix's `pad_info` supplies, built here
  # from boltz's own `to_keys` so it cannot drift from the window it describes.
  with torch.no_grad():
    am = bf['atom_pad_mask']
    K = am.shape[-1] // 32
    mq = am.view(1, K, 32, 1)
    mk = to_keys(am.unsqueeze(-1).float()).view(1, K, 1, -1)
    pm = (mq * mk)[0]
  return (_np.asarray(a)[0], qn, cn,
          _np.asarray(p).reshape(*_np.asarray(p).shape[-4:]),
          _np.asarray(pm))


def _native_decoder_boltz2(model, feats, a, q_ref, c_ref, p_ref, n_atom,
                           c_token):
  """-> r_ref from Boltz-2's own AtomAttentionDecoder.

  Reuses the encoder invocation parked in `_RAW`, because boltz's decoder needs
  three things the gate's arguments cannot carry: the DENSE q and c (the gate
  hands back the real atoms only), the `to_keys` closure the encoder built
  (a partial over an indexing matrix, not a tensor), and the same feature dict.
  Each side is still fed its own encoder's skips, which is the rule this gate
  already follows.
  """
  import numpy as _np
  import torch

  if _RAW.get('boltz2') is None:
    raise SystemExit('the boltz2 decoder needs the encoder invocation; '
                     'native_boltz2 did not park one in _RAW')
  R = _RAW['boltz2']
  M = _boltz2_modules(need_decoder=True)
  t = lambda x: torch.tensor(_np.asarray(x, _np.float32))
  with torch.no_grad():
    dbias = torch.cat([l(R['p']) for l in M['dec_bias']], dim=-1)
    r = M['aad'](a=t(a)[None], q=R['q'], c=R['c'], atom_dec_bias=dbias,
                 feats=R['bf'], to_keys=R['to_keys'])
  return _np.asarray(r)[0][:R['n_real']][:n_atom]


NATIVES = {m: native_protenix for m in _ENC_SRC}
NATIVES.update({m: native_of3 for m in _OF3_CKPT})
NATIVES['intellifold2'] = native_if2
NATIVES['rosettafold3'] = native_rf3
NATIVES['boltz2'] = native_boltz2
NATIVES.update({m: native_esmfold2 for m in (
    'esmfold2', 'esmfold2_fast', 'esmfold2_lm600m', 'esmfold2_lm300m')})


def _truncate_atom_blocks(p, nb, which='diffusion_atom_transformer'):
  """Slice a stacked atom transformer to `nb` blocks. See the BLOCKS knob."""
  out = {}
  for k, v in p.items():
    if which not in k:
      out[k] = v; continue
    out[k] = {}
    for leaf, arr in v.items():
      a_ = np.asarray(arr)
      if 'pair_logits_projection' in k and a_.ndim == 3 and a_.shape[1] > nb:
        a_ = a_[:, :nb]
      elif '__layer_stack' in k and a_.ndim and a_.shape[0] > nb:
        a_ = a_[:nb]
      out[k][leaf] = a_
  return out


def ours(model, cfg, model_dir, fb, act_dense, s, z):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import atom_cross_attention as aca

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)
  _nb = os.environ.get('BLOCKS')
  if _nb:
    # TRUNCATE BOTH SIDES: the loop count AND the stacked params. Truncating
    # only the reference's loop reads 0.72 and means nothing -- it compares a
    # 1-block stack against a 3-block one.
    cfg.heads.diffusion.atom_transformer.num_blocks = int(_nb)
    full = _truncate_atom_blocks(full, int(_nb))
    print('  BLOCKS=%s: atom stack truncated on BOTH sides' % _nb)

  def fwd():
    enc = aca.atom_cross_att_encoder(
        token_atoms_act=jnp.asarray(act_dense),
        trunk_single_cond=jnp.asarray(s), trunk_pair_cond=jnp.asarray(z),
        config=cfg.heads.diffusion, global_config=cfg.global_config,
        batch=fb, name='diffusion')
    return (enc.token_act, enc.skip_connection, enc.queries_single_cond,
            enc.pair_cond)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
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
  assert not unmapped, 'our encoder is partly at init'
  return f.apply(params, jax.random.PRNGKey(0))


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--pdb', default=os.path.expanduser('~/6MRR.pdb'))
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
  n_atom = feats['ref_pos'].shape[0]
  print('%s atom encoder, %d tokens, %d real atoms:'
        % (args.model, n_tok, n_atom))

  rng = np.random.default_rng(0)
  c_s = cfg.evoformer.seq_channel
  c_z = cfg.heads.diffusion.conditioning.pair_channel
  if c_z != cfg.evoformer.pair_channel:
    print('  NOTE this model compresses the trunk pair for the diffusion: '
          '%d in the trunk, %d here' % (cfg.evoformer.pair_channel, c_z))
  s = (rng.normal(size=(n_tok, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n_tok, n_tok, c_z)) * 0.5).astype(np.float32)
  # The localisation ladder, same idea as confidence_parity's: switch off one
  # input at a time and see which one carries the disagreement. ZERO_COND
  # silences the trunk conditioning (leaving the per-atom REFERENCE features and
  # the coordinates), ZERO_POS the noisy coordinates.
  if os.environ.get('ZERO_COND'):
    s = np.zeros_like(s)
    z = np.zeros_like(z)
    print('  NOTE trunk conditioning zeroed on both sides')
  act_dense = (rng.normal(size=(n_tok, max_atoms, 3)) * 5.0).astype(np.float32)
  if os.environ.get('ZERO_POS'):
    act_dense = np.zeros_like(act_dense)
    print('  NOTE noisy coordinates zeroed on both sides')
  act_dense = act_dense * feats['mask'][..., None]
  pos_noisy = act_dense[feats['mask']]
  # The two sides must be indexing the SAME atoms: our flat list is the dense
  # array masked row-major, which is how AF3 builds its own.
  assert np.allclose(pos_noisy,
                     act_dense.reshape(-1, 3)[feats['mask'].reshape(-1)])

  keep = os.environ.get('FEAT')
  if keep:
    fb, feats = zero_other_features(fb, feats, keep)

  if os.environ.get('SAME_ATOM_SET') and args.model.startswith('esmfold2'):
    # COMPARE ON THE SAME ATOM SET. Our featuriser emits 574 atoms for 6MRR and
    # ESMFold2's 573: ours carries the terminal OXT, absent from its
    # PROTEIN_HEAVY_ATOMS table. The encoder pools atoms to tokens with
    # scatter_mean, so that one atom changes its token's output outright -- it is
    # the single largest term in this gate (token 67 max|d| 2.19 against a
    # median of 0.244). Masking it says how much of the residual is the port and
    # how much is the input.
    import dataclasses

    import jax.numpy as jnp

    import esmfold2_dumps
    _nat = esmfold2_dumps.native(args.model)
    _f = {k[5:]: v[0] for k, v in _nat.items() if k.startswith('feat.')}
    _, _our = esmfold2_dumps.atom_map(fb, _f)
    _m = np.zeros(np.asarray(feats['mask']).shape, bool).reshape(-1)
    _m[_our] = True
    _m = _m.reshape(np.asarray(feats['mask']).shape)
    _dropped = int(np.asarray(feats['mask']).sum() - _m.sum())
    feats = dict(feats); feats['mask'] = _m
    rs = dataclasses.replace(fb.ref_structure, mask=jnp.asarray(_m, jnp.float32))
    psi = dataclasses.replace(fb.predicted_structure_info,
                              atom_mask=jnp.asarray(_m, jnp.float32))
    fb = dataclasses.replace(fb, ref_structure=rs, predicted_structure_info=psi)
    act_dense = act_dense * _m[..., None]
    pos_noisy = act_dense[_m]
    print('  SAME_ATOM_SET: dropped %d atom(s) ours-only' % _dropped)

  if os.environ.get('NATIVE_REF_POS') and args.model.startswith('esmfold2'):
    # Give OUR side ESMFold2's own reference conformer, so both sides build the
    # SAME rotary tables. ref_pos differs by mean 3.31 A name-matched (a
    # local-FRAME difference: our CCD ideal against its PROTEIN_REF_POS table)
    # and it moves the rope hard -- cos corr 0.837 -- while feeding rotary,
    # whose phase DIFFERENCE is what reaches attention. This says whether the
    # broad residual is that or not.
    import dataclasses

    import jax.numpy as jnp

    import esmfold2_dumps
    _nat2 = esmfold2_dumps.native(args.model)
    _f2 = {k[5:]: v[0] for k, v in _nat2.items() if k.startswith('feat.')}
    _ri, _oi = esmfold2_dumps.atom_map(fb, _f2)
    _pos = np.array(np.asarray(fb.ref_structure.positions), copy=True)
    _pos.reshape(-1, 3)[_oi] = np.asarray(_f2['ref_pos']).reshape(-1, 3)[_ri]
    fb = dataclasses.replace(
        fb, ref_structure=dataclasses.replace(
            fb.ref_structure, positions=jnp.asarray(_pos)))
    print('  NATIVE_REF_POS: our ref_pos replaced by the dump\'s at %d atoms'
          % len(_ri))
  a_ref, q_ref, c_ref, p_ref, pad_mask = NATIVES[args.model](
      args.model, fb, feats, pos_noisy, s, z, n_tok)
  a_got, skip, c_got, p_got = ours(args.model, cfg, model_dir, fb, act_dense,
                                   s, z)
  # `a_token` is the comparison that matters and the only layout-free one: our
  # skip connection is (num_subsets, num_queries, ch) -- the WINDOWED layout --
  # where native returns a flat (num_atoms, c_atom), so comparing those two
  # directly would compare different tensors. a_token is per TOKEN on both
  # sides, and it is what the token transformer consumes, so the whole encoder
  # -- features, windows, local attention, pooling -- is behind it.
  if os.environ.get('DIAG'):
    # WHICH tokens disagree. corr 0.9999 with max|d| 2.7 and p99.9 1.08 is a
    # small TAIL, not a uniform error -- so the question is whether the tail
    # sits at chain termini, at 32-atom subset boundaries, or on particular
    # residues. Each points somewhere different.
    _a = np.asarray(a_got).reshape(np.asarray(a_ref).shape)
    _e = np.abs(_a - np.asarray(a_ref)).max(-1)
    _o = np.argsort(-_e)
    print('  DIAG per-token max|d|: worst %s' % [(int(i), round(float(_e[i]), 3))
                                                 for i in _o[:6]])
    print('  DIAG            median %.4f  p90 %.4f  n tokens %d'
          % (float(np.median(_e)), float(np.quantile(_e, 0.9)), len(_e)))
    _na = np.asarray(feats['mask']).sum(-1)
    print('  DIAG atoms per worst token: %s' % [int(_na[i]) for i in _o[:6]])
  print('  shapes: a ours %s native %s | skip ours %s native q_l %s'
        % (np.asarray(a_got).shape, a_ref.shape, np.asarray(skip).shape,
           q_ref.shape))
  _cmp('a_token', np.asarray(a_got), a_ref)
  # The per-ATOM output too, which splits "the atom stack disagrees" from "the
  # atom-to-token pooling disagrees". Our queries layout tiles the atoms in
  # order in blocks of 32, so flattening it and truncating to the real atom
  # count is native's flat layout -- the same correspondence the input side
  # asserts above.
  # Truncate to what the NATIVE returned, not to the real atom count: protenix
  # and of3 hand back a packed atom list (574 here), if2 the dense (token, 24)
  # one (1632). Ours is the dense list in both cases, so this compares
  # like for like instead of silently taking the first 574 dense slots.
  n_cmp = q_ref.shape[0]
  q_got = np.asarray(skip).reshape(-1, np.asarray(skip).shape[-1])[:n_cmp]
  _cmp('q_atom', q_got, q_ref)
  if os.environ.get('DIAG'):
    # WHICH ATOMS, in window terms. The per-token breakdown above says which
    # tokens, and a residual that is ~0 for most of a structure and rises at one
    # END is about the atom LIST or the key window, not the arithmetic -- the
    # lesson the ESMFold2 OXT taught. This says whether the atoms that disagree
    # are exactly the ones in the LAST window(s), where our flat axis is padded
    # to num_tokens * max_atoms and native's is not.
    _d = np.abs(np.asarray(q_got, np.float64)
                - np.asarray(q_ref, np.float64)).max(-1)
    _w = np.arange(n_cmp) // 32
    print('  DIAG q per-window max|d| (32 atoms each): %s'
          % [(int(w), round(float(_d[_w == w].max()), 4))
             for w in range(int(_w.max()) + 1)])
    _bad = np.flatnonzero(_d > 0.01 * max(_d.max(), 1e-9))
    print('  DIAG q atoms above 1%% of the worst: %d of %d, first %s last %s'
          % (len(_bad), n_cmp, _bad[:6].tolist(), _bad[-6:].tolist()))
  # The two INPUTS to the atom stack.
  c_flat = np.asarray(c_got).reshape(-1, np.asarray(c_got).shape[-1])[:n_cmp]
  print('  c_l ours %s -> %s | native %s'
        % (np.asarray(c_got).shape, c_flat.shape, c_ref.shape))
  _cmp('c_atom_cond', c_flat, c_ref[:n_cmp])
  if os.environ.get('DECODER'):
    return _decoder(args.model, cfg, model_dir, fb, feats, q_ref, c_ref,
                    p_ref, a_ref, act_dense, s, z, n_atom, n_tok)
  if p_ref is None:
    print('  p_lm: native windows the DENSE atom axis, ours the packed one -- '
          'different atoms per window, not compared')
    return 0
  pg, pr = np.asarray(p_got), np.asarray(p_ref)
  print('  p_lm ours %s | native %s' % (pg.shape, pr.shape))
  # Our flat atom axis is padded to num_tokens * max_atoms, so we build more
  # windows than native does over the real atoms alone (51 vs 18 here). The
  # leading windows hold the same atoms in the same order, so compare those.
  nw = min(pg.shape[0], pr.shape[0])
  if pg.shape[1:] == pr.shape[1:]:
    _cmp('p_atom_pair', pg[:nw], pr[:nw])
    # WHERE the disagreement lives, because the claim above -- "the leading
    # windows hold the same atoms in the same order" -- is an assumption, and
    # p_atom_pair reads corr 0.9678 for protenix2 while q_atom, which is
    # computed FROM p, reads 1.000000. Both cannot be true of the same tensor,
    # so either the comparison is misaligned or p is not what feeds q.
    #
    # Per WINDOW says whether it is the trailing windows (an alignment
    # artifact, since ours has more of them); per KEY POSITION says whether it
    # is the padded end of each window. The same breakdown found the ESMFold2
    # OXT in one run.
    if os.environ.get('DIAG'):
      d = np.abs(np.asarray(pg[:nw], np.float64)
                 - np.asarray(pr[:nw], np.float64))
      per_w = d.reshape(nw, -1).max(-1)
      order = np.argsort(-per_w)
      print('  DIAG p per-window max|d|: worst %s'
            % [(int(i), round(float(per_w[i]), 3)) for i in order[:6]])
      print('  DIAG p            median %.4f  n windows %d (native %d, ours %d)'
            % (float(np.median(per_w)), nw, pr.shape[0], pg.shape[0]))
      per_k = d.reshape(-1, d.shape[-2], d.shape[-1]).max(0).max(-1)
      print('  DIAG p per-key-position max|d|: first 8 %s  last 8 %s'
            % ([round(float(x), 3) for x in per_k[:8]],
               [round(float(x), 3) for x in per_k[-8:]]))
    if pad_mask is not None:
      m = np.asarray(pad_mask).reshape(pr.shape[:3])[:nw] > 0
      print('  pad mask keeps %d of %d (block, query, key) slots'
            % (int(m.sum()), m.size))
      _cmp('p_pair_valid', pg[:nw][m], pr[:nw][m])
  else:
    print('  p_lm trailing shapes differ, not compared')
  return 0


# Where each vendor's AtomAttentionDecoder lives, and which checkpoint holds
# its weights. Every vendor in the panel names the class the same thing, and
# protenix, opendde and openfold3 additionally agree on the leaf names and the
# call signature -- so one helper serves them and a new model is one row here
# rather than a new function.
_DECODER_SRC = {
    'protenix1': ('protenix.model.modules.transformer',
                  '~/protenix_weights/protenix_base_default_v1.0.0.pt',
                  'module.diffusion_module.atom_attention_decoder.'),
    'protenix2': ('protenix.model.modules.transformer',
                  '~/protenix_weights/protenix-v2.pt',
                  'module.diffusion_module.atom_attention_decoder.'),
    # opendde's decoder is protenix's: same prefix, same 84 tensors, same
    # 3 blocks, same constructor (n_blocks, n_heads, c_token, c_atom,
    # c_atompair, n_queries, n_keys).
    'opendde': ('opendde.model.modules.transformer',
                '~/opendde_weights/opendde.pt',
                'module.diffusion_module.atom_attention_decoder.'),
}


def _native_decoder(model, feats, a, q_ref, c_ref, p_ref, n_atom, c_token):
  """-> r_ref, the decoder's per-atom position update, from the vendor's own.

  Split out of `_decoder` so the gate is not protenix-only: its `ours` half was
  always model-generic (it runs `aca.atom_cross_att_encoder/decoder` off the
  config), and only this side was hardcoded. That was 14 of the 39 remaining
  holes -- the largest single group -- and most of them are one table row.

  Widths come off the CHECKPOINT, never a default: `c_atompair` is read from the
  reference pair tensor and `n_blocks` counted from the block keys, so a release
  that changes either fails in load_state_dict rather than comparing quietly.

  Who fits this helper and who does not, checked rather than assumed:

    protenix1, protenix2, opendde   same class, same prefix
                                    (`module.diffusion_module.atom_attention_decoder.`),
                                    same leaf names, same call signature.
    openfold3, openbind0            call it `atom_attn_dec`, not
                                    `atom_attention_decoder`, and its leaf
                                    names differ.
    intellifold2                    prefix matches
                                    (`diffusion_module.atom_attention_decoder.`)
                                    but the leaves are `linear_a`,
                                    `layer_norm_q`, `linear_q` where protenix
                                    has `linear_no_bias_a`.
    rosettafold3, boltz2            not yet looked at.

  So those five need a function each, not a row.
  """
  import importlib

  import numpy as _np
  import torch

  mod_name, ckpt_path, pre = _DECODER_SRC[model]
  AtomAttentionDecoder = getattr(importlib.import_module(mod_name),
                                 'AtomAttentionDecoder')
  ckpt = os.path.expanduser(ckpt_path)
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd.get('state_dict', sd))
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))
  c_atom = sub['linear_no_bias_a.weight'].shape[0]
  c_atompair = p_ref.shape[-1] if p_ref is not None else 16
  bpre = 'atom_transformer.diffusion_transformer.blocks.'
  n_blocks = 1 + max(int(k[len(bpre):].split('.')[0]) for k in sub
                     if k.startswith(bpre))
  print('  decoder checkpoint: c_atom %d, c_atompair %d, c_token %d, %d blocks'
        % (c_atom, c_atompair, c_token, n_blocks))
  net = AtomAttentionDecoder(c_token=c_token, c_atom=c_atom,
                             c_atompair=c_atompair, n_blocks=n_blocks,
                             n_heads=4)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native decoder is missing %d tensors' % len(missing)
  net.eval()
  t = lambda x, d=torch.float32: torch.tensor(_np.asarray(x), dtype=d)
  with torch.no_grad():
    r = net(atom_to_token_idx=t(feats['atom_to_token_idx'], torch.long),
            a=t(a)[None], q_skip=t(q_ref)[None], c_skip=t(c_ref)[None],
            p_skip=t(p_ref)[None])
  return _np.asarray(r).reshape(-1, 3)[:n_atom]


def _native_decoder_of3(model, feats, a, q_ref, c_ref, p_ref, n_atom, c_token):
  """-> r_ref from OpenFold3's own AtomAttentionDecoder (openfold3, openbind0).

  Not a `_DECODER_SRC` row because of3 differs from protenix in all three ways
  a row cannot express: the submodule is `atom_attn_dec` not
  `atom_attention_decoder`, the leaves are `linear_q_in`/`linear_q_out` not
  `linear_no_bias_a`/`linear_no_bias_q`, and `forward` takes a feature dict
  plus keyword tensors rather than an `atom_to_token_idx`.

  The atom->token broadcast is the one thing to get right: of3 does NOT gather
  through an index, it calls `broadcast_token_feat_to_atoms(token_mask,
  num_atoms_per_token, ...)`, which assumes each token's atoms are contiguous.
  Ours are -- that is what makes `c_atom_cond` exact under `[:n_atom]` slicing
  -- so `feats['mask'].sum(1)` is the right lens vector, and it sums to exactly
  the atom count `q_ref` carries. Asserted below rather than trusted.

  Construction comes from of3's own model_config subtree (the same `_find` walk
  `denoise_parity.native_of3` uses) so nothing is retyped; the widths are then
  overridden from the CHECKPOINT, and c_atom_pair from the reference pair
  tensor, so a release that moved either fails in load_state_dict.
  """
  import copy

  import numpy as _np
  import torch

  from openfold3.core.model.layers.sequence_local_atom_attention import (
      AtomAttentionDecoder)
  from openfold3.projects.of3_all_atom.config.model_config import model_config

  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.atom_attn_dec.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  def _find(cfg, key):
    if hasattr(cfg, 'keys'):
      if key in cfg and 'atom_attn_dec' in cfg[key]:
        return cfg[key]
      for k in cfg:
        got = _find(cfg[k], key) if hasattr(cfg[k], 'keys') else None
        if got is not None:
          return got
    return None

  dcfg = _find(copy.deepcopy(model_config), 'diffusion_module')
  if dcfg is None:
    raise SystemExit('no diffusion_module subtree in of3 model_config')
  kw = dict(dcfg['atom_attn_dec'])
  bpre = 'atom_transformer.blocks.'
  kw['no_blocks'] = 1 + max(int(k[len(bpre):].split('.')[0]) for k in sub
                            if k.startswith(bpre))
  kw['c_atom'], kw['c_token'] = sub['linear_q_in.weight'].shape
  kw['c_atom_pair'] = (p_ref.shape[-1] if p_ref is not None
                       else sub['atom_transformer.layer_norm_z.weight'].shape[0])
  print('  decoder checkpoint: c_atom %d, c_atom_pair %d, c_token %d, %d blocks'
        % (kw['c_atom'], kw['c_atom_pair'], kw['c_token'], kw['no_blocks']))
  net = AtomAttentionDecoder(**kw)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native decoder is missing %d tensors' % len(missing)
  net.eval()

  lens = _np.asarray(feats['mask']).sum(1).astype(int)
  n_flat = q_ref.shape[0]
  assert int(lens.sum()) == n_flat, (
      'of3 broadcasts by per-token atom COUNTS, so the lens must sum to the '
      'atom axis the skips carry: %d lens vs %d atoms'
      % (int(lens.sum()), n_flat))
  t = lambda x, d=torch.float32: torch.tensor(_np.asarray(x), dtype=d)
  batch = {
      'token_mask': torch.ones(1, len(lens)),
      'num_atoms_per_token': t(lens, torch.long)[None],
      'atom_mask': torch.ones(1, n_flat),
  }
  with torch.no_grad():
    r = net(batch, ai=t(a)[None], ql=t(q_ref)[None], cl=t(c_ref[:n_flat])[None],
            plm=t(p_ref)[None])
  return _np.asarray(r).reshape(-1, 3)[:n_atom]


def _native_decoder_if2(model, feats, a, q_ref, c_ref, p_ref, n_atom, c_token):
  """-> r_ref from IntelliFold-2's own AtomAttentionDecoder.

  Two vendor facts drive this one. First, the leaves are `linear_a`,
  `layer_norm_q`, `linear_q` where protenix has `linear_no_bias_a` -- same
  prefix, different names, so it cannot be a `_DECODER_SRC` row.

  Second, and the reason this needed a stash: `native_if2` deliberately returns
  `p_lm` as None, because if2 windows the atom pair over an axis whose windows
  hold different atoms than ours and comparing them would mislead. The DECODER
  still needs that tensor as `p_skip` -- it is an input, not a comparison -- so
  `native_if2` now also parks the raw tensor in `_RAW`. Feeding native its own
  p is exactly the rule the encoder gate already follows: each side gets its
  own encoder's skips, legitimate here because the encoder is gated exact.

  `advanced_conversion` must match the encoder's -- the packed layout of the v2
  inference config, see [[if2-atom-layout]] -- because the decoder does the
  same atom<-token broadcast the encoder does.
  """
  import numpy as _np
  import torch

  from intellifold.openfold.model.diffusion import AtomAttentionDecoder

  if _RAW.get('if2_p') is None:
    raise SystemExit(
        'the if2 decoder needs the encoder\'s p_lm and native_if2 did not park '
        'one in _RAW -- it returns None for the COMPARISON, which is correct, '
        'but the decoder consumes it as an input.')
  p_skip = _RAW['if2_p']
  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.atom_attention_decoder.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))
  bpre = 'atom_transformer.blocks.'
  n_blocks = 1 + max(int(k[len(bpre):].split('.')[0]) for k in sub
                     if k.startswith(bpre))
  c_atom, c_tok_ck = sub['linear_a.weight'].shape
  c_atompair = sub['atom_transformer.layer_norm_z.weight'].shape[0]
  print('  decoder checkpoint: c_atom %d, c_atompair %d, c_token %d, %d blocks'
        % (c_atom, c_atompair, c_tok_ck, n_blocks))
  assert c_tok_ck == c_token, (c_tok_ck, c_token)
  packed = not os.environ.get('IF2_DENSE')
  net = AtomAttentionDecoder(
      c_atom=c_atom, c_atompair=c_atompair, c_token=c_token,
      no_blocks=n_blocks, no_heads=4, window_size_row=32, window_size_col=128,
      advanced_conversion=packed)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native decoder is missing %d tensors' % len(missing)
  net.eval()
  t = lambda x, d=torch.float32: torch.tensor(_np.asarray(x), dtype=d)
  mask = _np.asarray(feats['mask'])
  n_flat = q_ref.shape[0]
  with torch.no_grad():
    r = net(a=t(a)[None], q_skip=t(q_ref)[None], c_skip=t(c_ref[:n_flat])[None],
            p_skip=t(p_skip), atom_mask=torch.ones(1, n_flat),
            molecule_atom_lens=t(mask.sum(1), torch.long)[None],
            chunk_size=None, use_deepspeed_evo_attention=False,
            inplace_safe=False)
  return _np.asarray(r).reshape(-1, 3)[:n_atom]

_DECODER_FN = {'boltz2': _native_decoder_boltz2,
               'openfold3': _native_decoder_of3,
               'openbind0': _native_decoder_of3,
               'intellifold2': _native_decoder_if2}


def _decoder(model, cfg, model_dir, fb, feats, q_ref, c_ref, p_ref, a_ref,
             act_dense, s, z, n_atom, n_tok):
  """DECODER=1: the atom cross-attention DECODER, which nothing else gates.

  It is covered elsewhere only in composition, by the models whose whole denoise
  step is exact -- real evidence, but unable to localise a decoder-only fault.

  Each side is fed its OWN encoder's outputs, which is legitimate here precisely
  because the encoder is already gated exact for this model (1.000000): the two
  sets of skips agree to ~1e-5, so anything larger in the output is the decoder.
  The token activation `a` IS shared -- it is per token, so it needs no layout
  translation.

  OUR side must run the encoder and decoder in ONE transform, on the REAL
  inputs. Splicing three fields into an encoder output computed from zeros
  reads r_update 0.976, because `keys_single_cond` -- which the decoder's
  cross-attention consumes -- then comes from the wrong invocation. That number
  cannot be a real decoder fault: protenix2's whole denoise step (L3) is exact
  at 0.0000 A/atom, and L3 runs this decoder.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp
  import numpy as _np
  import torch

  from alphafold3.model import params as afp
  from alphafold3.model.network import atom_cross_attention as aca

  from diffusion_parity import _stub_layer_norm

  rng = _np.random.default_rng(1)
  c_token = a_ref.shape[-1]
  a = (rng.normal(size=(n_tok, c_token)) * 0.5).astype(_np.float32)

  fn = _DECODER_FN.get(model)
  if fn is None and model in _DECODER_SRC:
    fn = _native_decoder
  if fn is None:
    raise SystemExit(
        'no native DECODER for %r; have %s. The `ours` half below is already '
        'model-generic -- what a new model needs is one _DECODER_SRC row (if it '
        'is protenix-shaped) or one _DECODER_FN entry, and its checkpoint '
        'prefix and widths can be checked without a GPU.'
        % (model, sorted(set(_DECODER_SRC) | set(_DECODER_FN))))
  _stub_layer_norm()
  r_ref = fn(model, feats, a, q_ref, c_ref, p_ref, n_atom, c_token)

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)

  def fwd():
    enc = aca.atom_cross_att_encoder(
        token_atoms_act=jnp.asarray(act_dense),
        trunk_single_cond=jnp.asarray(s), trunk_pair_cond=jnp.asarray(z),
        config=cfg.heads.diffusion, global_config=cfg.global_config,
        batch=fb, name='diffusion')
    return aca.atom_cross_att_decoder(
        token_act=jnp.asarray(a), enc=enc, config=cfg.heads.diffusion,
        global_config=cfg.global_config, batch=fb, name='diffusion')

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      key = ('diffuser/~/diffusion_head' if sc == '~'
             else 'diffuser/~/diffusion_head/' + sc)
      src = full.get(key)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = _np.asarray(v, _np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our decoder is partly at init'
  r_got = _np.asarray(f.apply(params, jax.random.PRNGKey(0)))
  r_got = r_got[_np.asarray(feats['mask'])][:n_atom]
  print('  shapes: ours %s native %s' % (r_got.shape, r_ref.shape))
  _cmp('r_update', r_got, r_ref)
  d = _np.sqrt(((r_got - r_ref) ** 2).sum(-1))
  print('  per-atom |dr|: mean %.6f, max %.6f, rms(native) %.4f'
        % (d.mean(), d.max(), _np.sqrt((r_ref ** 2).sum(-1).mean())))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
