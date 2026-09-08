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
  """-> (a_token, q_l) from protenix's own AtomAttentionEncoder."""
  import torch

  _stub_layer_norm()
  from protenix.model.modules.transformer import (AtomAttentionEncoder,
                                                  rearrange_qk_to_dense_trunk)

  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.diffusion_module.atom_attention_encoder.'
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
  q_list, k_list, pad_info = rearrange_qk_to_dense_trunk(
      q=[t(feats['ref_pos']), t(feats['ref_space_uid'])],
      k=[t(feats['ref_pos']), t(feats['ref_space_uid'])],
      dim_q=[-2, -1], dim_k=[-2, -1], n_queries=32, n_keys=128,
      compute_mask=True)
  d_lm = q_list[0][..., None, :] - k_list[0][..., None, :, :]
  v_lm = (q_list[1][..., None].int() == k_list[1][..., None, :].int()
          ).unsqueeze(dim=-1)

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
        r_l=t(pos_noisy)[None][None], s=t(s)[None][None], z=t(z)[None][None])
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


NATIVES = {m: native_protenix for m in _PROTENIX_CKPT}
NATIVES.update({m: native_of3 for m in _OF3_CKPT})
NATIVES['intellifold2'] = native_if2
NATIVES['rosettafold3'] = native_rf3


def ours(model, cfg, model_dir, fb, act_dense, s, z):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import atom_cross_attention as aca

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)

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
  c_z = cfg.evoformer.pair_channel
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
    if pad_mask is not None:
      m = np.asarray(pad_mask).reshape(pr.shape[:3])[:nw] > 0
      print('  pad mask keeps %d of %d (block, query, key) slots'
            % (int(m.sum()), m.size))
      _cmp('p_pair_valid', pg[:nw][m], pr[:nw][m])
  else:
    print('  p_lm trailing shapes differ, not compared')
  return 0


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
  from protenix.model.modules.transformer import AtomAttentionDecoder

  rng = _np.random.default_rng(1)
  c_token = a_ref.shape[-1]
  a = (rng.normal(size=(n_tok, c_token)) * 0.5).astype(_np.float32)

  _stub_layer_norm()
  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.diffusion_module.atom_attention_decoder.'
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
    r_ref = net(atom_to_token_idx=t(feats['atom_to_token_idx'], torch.long),
                a=t(a)[None], q_skip=t(q_ref)[None], c_skip=t(c_ref)[None],
                p_skip=t(p_ref)[None])
  r_ref = _np.asarray(r_ref).reshape(-1, 3)[:n_atom]

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
