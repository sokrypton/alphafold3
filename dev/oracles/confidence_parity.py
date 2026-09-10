"""L4 parity: our confidence head against the vendor's own module.

The README's confidence column reads "not measured" for openfold3, intellifold2,
the protenix family and rosettafold3 -- four of the seven ports, i.e. every one
that predates the injection-ladder method. This closes it the way
diffusion_parity.py closed L2: instantiate the vendor's own head, load the real
weights into it, and run BOTH heads on the SAME inputs.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:/home/ubuntu/protenix \
    python dev/oracles/confidence_parity.py protenix2

Unlike L1/L2 this cannot run on purely synthetic input: both heads consume an
ATOM LAYOUT (which slot of which token each atom is, and which atom represents
a token in the distance embedding). So the trunk activations stay synthetic --
s_inputs/s_trunk/z_trunk, seeded -- while the layout comes from a real
featurised 6MRR batch, and the native side's flat-atom indices are DERIVED from
that same batch rather than invented. The rep-atom coordinates the two sides
end up embedding are compared explicitly before anything else is trusted; if
that check fails the logits are being computed on different geometry and no
correlation below it means anything.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_PROTENIX_CKPT = {
    'protenix2': 'protenix-v2.pt',
    'protenix1': 'protenix_base_default_v1.0.0.pt',
}


def _cmp(tag, got, ref, mask=None, note=''):
  a = np.asarray(got, np.float64)
  b = np.asarray(ref, np.float64)
  if mask is not None:
    a, b = a[mask], b[mask]
  a, b = a.ravel(), b.ravel()
  rb = np.sqrt((b ** 2).mean())
  d = np.abs(a - b)
  # p99.9 next to max: these outputs are EXPECTATIONS over softmaxed bins, so a
  # 1e-6 logit difference at a bin boundary moves one entry by O(1) while every
  # other entry is exact. max|d| alone cannot tell that apart from a systematic
  # error; p99.9 >> 0 can.
  # `note` carries a tag the audit reads, e.g. '[superseded by p_pair_valid]'
  # for a comparison the gate makes for DIAGNOSIS and does not want graded.
  line = ('  %-10s corr %.6f  rms ours/native %.4f  max|d| %.5f  p99.9|d| %.2e'
          '  rms(native) %.3f  max|d|/rms %.2e'
          % (tag, np.corrcoef(a, b)[0, 1], np.sqrt((a ** 2).mean()) / rb,
             d.max(), np.percentile(d, 99.9), rb, d.max() / rb))
  print(line + ('  ' + note if note else ''))


def _truncate(stack, name='blocks'):
  """Truncate a native pairformer stack to BLOCKS, if set.

  MUST be called by every adapter. When only OUR side was truncated, the
  1-block run read corr -0.19 against native's 4 blocks and looked like a
  catastrophic port bug -- the tell being that it got WORSE with fewer blocks.
  Same mistake, same tell, as the L1 harness's `--blocks`.
  """
  nb = os.environ.get('BLOCKS')
  if nb is None:
    return
  nb = int(nb)
  setattr(stack, name, getattr(stack, name)[:nb])
  for attr in ('n_blocks', 'no_blocks'):
    if hasattr(stack, attr):
      setattr(stack, attr, nb)


def our_inputs(model, seq, model_dir=None):
  """-> (batch, cfg, dense_positions, rng-drawn s_inputs/s/z).

  Uses fold_check's own featurisation so the layout conventions are the model's
  real ones, not a second implementation of them.
  """
  import fold_check

  batch, cfg, model_dir = fold_check._fold_setup(model, seq, model_dir)
  # _fold_setup hands back a flat BatchDict; Model.__call__ is what normally
  # turns it into the structured Batch the head's arguments come from.
  from alphafold3.model import feat_batch
  batch = feat_batch.Batch.from_data_dict(batch)
  n = int(np.asarray(batch.token_features.mask).shape[0])
  max_atoms = int(np.asarray(batch.predicted_structure_info.atom_mask).shape[1])
  rng = np.random.default_rng(0)
  # Coordinates are only ever cdist'ed and one-hot binned, so a random cloud is
  # a valid input -- but it has to SPAN the distance bins (3.25 to 52 A) or the
  # one-hot lands in one column and the comparison stops testing the embedding.
  # POS_SCALE inflates the cloud. At a large scale every off-diagonal pair
  # lands in the final distance bin and the diagonal in none, so the distance
  # embedding is IDENTICAL on both sides by construction -- combined with
  # ZERO_SINPUTS that leaves only the stack and the logit heads in play.
  pos = (rng.normal(size=(n, max_atoms, 3))
         * float(os.environ.get('POS_SCALE', 12.0))).astype(np.float32)
  return batch, cfg, model_dir, pos, rng, n, max_atoms


def native_protenix(model, batch, pos, rng, n, max_atoms):
  """-> (dict of native logits, s_inputs, s, z, rep_coords)."""
  import torch

  from diffusion_parity import _stub_layer_norm
  _stub_layer_norm()
  from protenix.model.modules.confidence import ConfidenceHead

  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.confidence_head.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[2])
                     for k in sub if k.startswith('pairformer_stack.blocks.'))
  c_z = sub['linear_no_bias_pae.weight'].shape[1]
  c_s_inputs = sub['linear_no_bias_s1.weight'].shape[1]
  slots, c_s, b_plddt = sub['plddt_weight'].shape
  b_pae = sub['linear_no_bias_pae.weight'].shape[0]
  print('  checkpoint: %d pairformer blocks, c_s %d, c_z %d, c_s_inputs %d, '
        '%d slots, %d plddt bins' % (n_blocks, c_s, c_z, c_s_inputs, slots,
                                     b_plddt))
  if slots < max_atoms:
    raise SystemExit('batch has %d atom slots, checkpoint has %d'
                     % (max_atoms, slots))

  # s_inputs IS NOT SHARED. protenix's 449 columns are
  # [atom(384), restype(32), profile(32), del_mean(1)]; ours are 447,
  # [restype(31), profile(31), del_mean(1), atom(384)] with the residue alphabet
  # permuted -- converters/openfold3.py `_reorder_target_feat_weights` is what
  # the weights went through. So draw NATIVE's vector and derive ours with the
  # same row selection, and ZERO the two native columns our layout drops: they
  # are the nucleic classes, exactly zero on a protein input, and left random
  # they would feed native rows that have no counterpart on our side at all.
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  assert len(idx) == 447 and len(set(idx.tolist())) == 447
  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  drop = np.setdiff1d(np.arange(c_s_inputs), idx)
  print('  s_inputs: %d native columns, %d ours, zeroing %d dropped %s'
        % (c_s_inputs, len(idx), len(drop), drop.tolist()))
  s_inputs[:, drop] = 0.0
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  # The atom layout, DERIVED from the batch. Native works on a flat atom axis:
  # flattening our dense (token, slot) array in C order makes atom_to_token the
  # repeated token index and atom_to_tokatom the tiled slot index, so the two
  # sides index the same plddt_weight row for the same atom by construction.
  a2t = np.repeat(np.arange(n), max_atoms)
  a2ta = np.tile(np.arange(max_atoms), n)
  # The rep atom per token is whatever the batch's pseudo-beta gather picks.
  gi = batch.pseudo_beta_info.token_atoms_to_pseudo_beta
  flat_rep = np.asarray(gi.gather_idxs).reshape(-1)
  rep_mask = np.zeros(n * max_atoms, bool)
  rep_mask[flat_rep] = True
  assert rep_mask.sum() == n, 'pseudo-beta gather is not one atom per token'
  # ORDER matters: native selects with a boolean mask, i.e. in flat order. That
  # equals token order only if the gather indices increase, which they do
  # because each token's rep atom lies in that token's own slot block.
  assert (np.diff(flat_rep) > 0).all(), 'rep atoms are not in token order'

  net = ConfidenceHead(n_blocks=n_blocks, c_s=c_s, c_z=c_z,
                       c_s_inputs=c_s_inputs, b_pae=b_pae, b_pde=b_pae,
                       b_plddt=b_plddt, max_atoms_per_token=slots,
                       hidden_scale_up=True, stop_gradient=False)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  # BLOCKS truncates the confidence pairformer on BOTH sides: it isolates the
  # embedding + logits path from the stack, which localises a broad difference
  # (the stack is 4 blocks of code L1 already gates, so a divergence there
  # points at the confidence stack's own config, not the pairformer).
  _truncate(net.pairformer_stack)
  net.eval()
  feats = {
      'distogram_rep_atom_mask': torch.tensor(rep_mask),
      'atom_to_token_idx': torch.tensor(a2t),
      'atom_to_tokatom_idx': torch.tensor(a2ta),
  }
  with torch.no_grad():
    plddt, pae, pde, resolved = net(
        feats, torch.tensor(s_inputs)[None], torch.tensor(s)[None],
        torch.tensor(z)[None], torch.ones(1, n, n),
        torch.tensor(pos.reshape(-1, 3))[None, None])
  out = {'plddt': plddt[0, 0].numpy(), 'pae': pae[0, 0].numpy(),
         'pde': pde[0, 0].numpy(), 'resolved': resolved[0, 0].numpy()}
  return out, s_inputs_ours, s, z, pos.reshape(-1, 3)[rep_mask]


_OF3_CKPT = {'openfold3': 'of3-p2-155k.pt', 'openbind0': 'of3-ob-174k.pt'}


def native_of3(model, batch, pos, rng, n, max_atoms):
  """-> (native logits, s_inputs(ours-layout), s, z, rep coords) for of3.

  of3's `AuxiliaryHeadsAllAtom.forward` wants a whole feature dict and does its
  own representative-atom selection, so this drives the four sub-modules
  directly -- PairformerEmbedding, then pde/pae/plddt/resolved -- which is
  exactly the sequence that forward runs and skips the batch plumbing.

  Two differences from protenix worth knowing before reading the numbers: of3
  has NO unbinned distance term and NO clamp+LayerNorm on the trunk single, and
  its confidence distance embedding is AF3's squared-bin dgram
  (`(dij > bins**2) * (dij < upper)`), not a distance one-hot.
  """
  import torch

  from openfold3.core.model.heads import prediction_heads as ph

  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'aux_heads.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(
      int(k.split('.')[3]) for k in sub
      if k.startswith('pairformer_embedding.pairformer_stack.blocks.'))
  c_s_inputs, c_z = (sub['pairformer_embedding.linear_i.weight'].shape[1],
                     sub['pairformer_embedding.linear_i.weight'].shape[0])
  no_bin = sub['pairformer_embedding.linear_distance.weight'].shape[1]
  c_s = sub['plddt.layer_norm.weight'].shape[0]
  b_plddt = 50
  slots = sub['plddt.linear.weight'].shape[0] // b_plddt
  b_pae = sub['pae.linear.weight'].shape[0]
  print('  checkpoint: %d pairformer blocks, c_s %d, c_z %d, c_s_inputs %d, '
        '%d slots, %d distance bins' % (n_blocks, c_s, c_z, c_s_inputs, slots,
                                        no_bin))
  if slots != max_atoms:
    # of3 stores 23 slots where our dense layout has 24; the converter
    # zero-pads the extra one. Only the slots a real atom occupies are compared
    # below, and no protein residue reaches 23 heavy atoms, so this is a layout
    # difference rather than a missing head.
    print('  NOTE %d native slots vs %d dense slots; comparing the common %d'
          % (slots, max_atoms, min(slots, max_atoms)))

  # Same s_inputs story as protenix: 449 native columns in of3's own order.
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s_inputs[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  gi = batch.pseudo_beta_info.token_atoms_to_pseudo_beta
  flat_rep = np.asarray(gi.gather_idxs).reshape(-1)
  rep = pos.reshape(-1, 3)[flat_rep]

  emb = ph.PairformerEmbedding(
      pairformer=dict(c_s=c_s, c_z=c_z, c_hidden_pair_bias=24,
                      no_heads_pair_bias=16, c_hidden_mul=128,
                      c_hidden_pair_att=32, no_heads_pair=4,
                      no_blocks=n_blocks, transition_type='swiglu',
                      transition_n=4, pair_dropout=0.0,
                      fuse_projection_weights=False, blocks_per_ckpt=None,
                      inf=1e9),
      c_s_input=c_s_inputs, c_z=c_z, min_bin=3.25, max_bin=50.75,
      no_bin=no_bin, inf=1e8)
  pde = ph.PredictedDistanceErrorHead(c_z=c_z, c_out=b_pae)
  pae = ph.PredictedAlignedErrorHead(c_z=c_z, c_out=b_pae)
  plddt = ph.PerResidueLDDTAllAtom(c_s=c_s, c_out=b_plddt,
                                   max_atoms_per_token=slots)
  resolved = ph.ExperimentallyResolvedHeadAllAtom(
      c_s=c_s, c_out=2, max_atoms_per_token=slots)
  loaded = 0
  for name, mod in (('pairformer_embedding', emb), ('pde', pde), ('pae', pae),
                    ('plddt', plddt), ('experimentally_resolved', resolved)):
    part = {k[len(name) + 1:]: v for k, v in sub.items()
            if k.startswith(name + '.')}
    missing, unexpected = mod.load_state_dict(part, strict=False)
    assert not missing, '%s missing %s' % (name, list(missing)[:3])
    assert not unexpected, '%s unexpected %s' % (name, list(unexpected)[:3])
    mod.eval()
    loaded += len(part)
  _truncate(emb.pairformer_stack)
  print('  native: %d tensors loaded across 5 modules, 0 missing' % loaded)

  with torch.no_grad():
    si, zij = emb.pairformer_emb(
        si_input=torch.tensor(s_inputs)[None], si=torch.tensor(s)[None],
        zij=torch.tensor(z)[None], x_pred=torch.tensor(rep)[None],
        single_mask=torch.ones(1, n), pair_mask=torch.ones(1, n, n))
    # plddt/resolved BEFORE the flat select: their reshape to (token, slot, bin)
    # is what our dense layout already is, and the select would only drop the
    # padding slots we mask out anyway.
    lg = plddt.linear(plddt.layer_norm(si)).reshape(n, slots, b_plddt)
    rg = resolved.linear(resolved.layer_norm(si)).reshape(n, slots, 2)
    out = {'pae': pae(zij)[0].numpy(), 'pde': pde(zij)[0].numpy(),
           'plddt': lg.numpy(), 'resolved': rg.numpy()}
  return out, s_inputs_ours, s, z, rep


def native_if2(model, batch, pos, rng, n, max_atoms):
  """-> (native logits, s_inputs, s, z, rep coords) for intellifold2.

  if2's `c_s_inputs` is 384+31+31+1 = 447 -- OUR width, not the 449 of3/protenix
  use -- and its converter maps the two projections straight through, so no
  s_inputs remap is needed here. Its confidence `c_z` is 512, NOT the 128 its
  config file defaults to, so that is read off the checkpoint and overridden.

  Independent confirmation of two conventions while wiring this: if2's own
  pae/pde reduction is `max_bin=31, no_bins=64` and its plddt is 50 bins over
  [0, 1] -- the same numbers our shared head uses.
  """
  import torch

  from intellifold.openfold.config import model_config
  from intellifold.openfold.model.heads import ConfidenceHead

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'confidence_head.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  c_z = sub['linear_d.weight'].shape[0]
  c_s_inputs = sub['linear_s_inputs_row.weight'].shape[1]
  c_s = sub['plddt_head.layer_norm.weight'].shape[0]
  no_bins = sub['linear_d.weight'].shape[1]
  n_blocks = 1 + max(int(k.split('.')[2]) for k in sub
                     if k.startswith('pairformer_stack.blocks.'))
  cfg = model_config()
  cfg.confidence_head.c_z = c_z
  ps = cfg.confidence_head.pairformer_stack
  ps.c_z = c_z
  ps.pair_dropout = 0.0
  # Every hidden width scales with c_z here, and none of them follows from it by
  # a fixed ratio, so read them all off the checkpoint: the config's own 128/32/4
  # defaults were written for c_z 128 and fail load_state_dict with a wall of
  # size mismatches on this release.
  ps.c_hidden_mul = sub['pairformer_stack.blocks.0.pair_stack.tri_mul_out.'
                        'linear_ab_p.weight'].shape[0] // 2
  ps.no_heads_pair = sub['pairformer_stack.blocks.0.pair_stack.tri_att_start.'
                         'linear.weight'].shape[0]
  ps.c_hidden_pair_att = (
      sub['pairformer_stack.blocks.0.pair_stack.tri_att_start.mha.linear_q.'
          'weight'].shape[0] // ps.no_heads_pair)
  ps.no_heads_single = sub['pairformer_stack.blocks.0.attention_pair_bias.'
                           'linear_z.weight'].shape[0]
  ps.no_blocks = n_blocks
  ps.transition_n = (
      sub['pairformer_stack.blocks.0.pair_stack.pair_transition.linear.'
          'weight'].shape[0] // (2 * c_z))
  slots = cfg.confidence_head.max_num_atoms
  b_plddt = cfg.confidence_head.no_bin_plddt
  print('  checkpoint: %d pairformer blocks, c_s %d, c_z %d, c_s_inputs %d, '
        '%d slots, %d distance bins' % (n_blocks, c_s, c_z, c_s_inputs, slots,
                                        no_bins))

  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)
  # ZERO_SINPUTS silences the two s_inputs->pair projections on both sides. It
  # is the cheapest discriminator for an i/j orientation swap: those two
  # projections are the only asymmetric term added to z here, so if the
  # comparison goes to parity with them off, the fault is in their orientation
  # and not in the stack.
  if os.environ.get('ZERO_SINPUTS'):
    s_inputs = np.zeros_like(s_inputs)
    print('  NOTE s_inputs zeroed on both sides')

  gi = batch.pseudo_beta_info.token_atoms_to_pseudo_beta
  flat_rep = np.asarray(gi.gather_idxs).reshape(-1)
  rep = pos.reshape(-1, 3)[flat_rep]
  amask = (np.asarray(batch.predicted_structure_info.atom_mask) > 0)

  # MATCH THE STORAGE DTYPE. intellifold2's converter writes trunk-region
  # weights as bfloat16 on purpose -- "mirrors AF3's param dtype policy, which
  # the published blob follows" (converters/intellifold2.py `_record_dtype`) --
  # so our side runs bf16-rounded values while native's checkpoint is fp32.
  # Left alone that reads as a port bug: pae/pde 0.9867/0.9847 and plddt 0.614,
  # all of it storage. Rounding native the same way is the apples-to-apples
  # comparison; the LayerNorms and the four logit heads stay fp32 because the
  # blob keeps those in fp32 too.
  import ml_dtypes
  f32 = ('layer_norm', 'pae_head.linear', 'pde_head.linear',
         'plddt_head.linear', 'resolved_head.linear')
  nbf = 0
  for k in list(sub):
    if any(t in k for t in f32):
      continue
    sub[k] = torch.tensor(
        np.asarray(sub[k], np.float32).astype(ml_dtypes.bfloat16)
        .astype(np.float32))
    nbf += 1
  print('  rounded %d of %d native tensors to bfloat16 (blob storage dtype)'
        % (nbf, len(sub)))

  net = ConfidenceHead(cfg)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  _truncate(net.pairformer_stack)
  net.eval()
  feats = {
      'seq_mask': torch.ones(1, n),
      'atom_pseudo_beta_index': torch.tensor(flat_rep)[None],
      'pseudo_beta_mask': torch.ones(1, n),
      'pred_dense_atom_mask': torch.tensor(amask)[None],
      'frame_mask': torch.ones(1, n),
      'asym_id': torch.zeros(1, n, dtype=torch.long),
  }
  with torch.no_grad():
    out = net(torch.tensor(s_inputs)[None], torch.tensor(s)[None],
              torch.tensor(z)[None], torch.tensor(pos)[None], feats)
  ref = {'pae': out['pae_logits'][0].numpy(),
         'pde': out['pde_logits'][0].numpy(),
         'plddt': out['plddt_logits'][0].numpy(),
         'resolved': out['resolved_logits'][0].numpy()}
  return ref, s_inputs, s, z, rep


def native_rf3(model, batch, pos, rng, n, max_atoms):
  """-> (native logits, s_inputs(ours-layout), s, z, rep coords) for rf3.

  Two rf3-specific conventions our port implements and nothing has checked at
  activation level until now:
    * a PARAMETER-FREE LayerNorm over the WHOLE TENSOR applied to each detached
      trunk input (`layer_norm_along_feature_dimension` is False and the
      released config leaves it so) -- the same global norm that made pTM read
      0.04 under token padding (memory rf3-padding-confidence-bug);
    * the pair distance embedding is CA-CA bucketized into 40 classes (39
      boundaries 3.25..50.75, so class 0 is "closer than 3.25"), not AF3's
      39-bin CB dgram. `process_pred_distances` is (c_z, 40) in the checkpoint,
      which is what confirms the extra class.
  """
  import torch

  from rf3.model.layers.af3_auxiliary_heads import ConfidenceHead

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
  pre = 'shadow.confidence_head.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  c_z = sub['predict_pae.weight'].shape[1]
  c_s = sub['predict_plddt.weight'].shape[1]
  c_s_inputs = sub['process_s_inputs_left.weight'].shape[1]
  b_pae = sub['predict_pae.weight'].shape[0]
  b_plddt = 50
  slots = sub['predict_plddt.weight'].shape[0] // b_plddt
  n_blocks = 1 + max(int(k.split('.')[1]) for k in sub
                     if k.startswith('pairformer.'))
  print('  checkpoint: %d pairformer blocks, c_s %d, c_z %d, c_s_inputs %d, '
        '%d slots (NHEAVY), %d distance classes'
        % (n_blocks, c_s, c_z, c_s_inputs, slots,
           sub['process_pred_distances.weight'].shape[1]))

  # RF3'S OWN ALPHABET, not of3's: rf3 transposes G/C and DG/DC, and its
  # converter remaps these projections through `_rf3_target_feat` accordingly.
  # Built from of3's permutation instead, this harness read pae/pde corr 0.983
  # and looked like a port bug -- the tell being that restricting s_inputs to
  # the protein columns (PROTEIN_ONLY_SINPUTS=1) made it vanish, i.e. the
  # disagreement lived exactly in the residue classes the two alphabets order
  # differently. See memory rf3-rna-alphabet.
  from converters.rosettafold3 import _AF3_TO_RF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s_inputs[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  if os.environ.get('ZERO_SINPUTS'):
    s_inputs = np.zeros_like(s_inputs)
  if os.environ.get('PROTEIN_ONLY_SINPUTS'):
    # Keep only the columns a PROTEIN fold actually populates: the atom block,
    # del_mean, and the first 20 residue classes of both the restype and the
    # profile block. A random draw over all 449 columns exercises the nucleic
    # classes too, and those are where an alphabet-permutation difference
    # between vendors lives (rf3 swaps G/C against of3 -- memory
    # rf3-rna-alphabet). If parity appears only under this flag, the fault is
    # confined to columns a protein input never reaches.
    keep = np.zeros(c_s_inputs, bool)
    keep[:384] = True
    keep[448] = True
    keep[384 + np.asarray(_remap)[:20]] = True
    keep[416 + np.asarray(_remap)[:20]] = True
    s_inputs[:, ~keep] = 0.0
    print('  NOTE s_inputs restricted to %d protein columns' % keep.sum())
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  # rf3's rep atom for the confidence distance embedding is the token-centre CA
  # = dense slot 1, NOT the pseudo-beta gather our other adapters use.
  rep_idx = np.arange(n) * max_atoms + 1
  rep = pos.reshape(-1, 3)[rep_idx]

  net = ConfidenceHead(
      c_s=c_s, c_z=c_z, n_pairformer_layers=n_blocks,
      pairformer=dict(p_drop=0.0,
                      triangle_multiplication=dict(d_hidden=128),
                      triangle_attention=dict(n_head=4, d_hidden=32),
                      attention_pair_bias=dict(n_head=16)),
      n_bins_pae=b_pae, n_bins_pde=b_pae, n_bins_plddt=b_plddt,
      n_bins_exp_resolved=2,
      use_af3_style_binning_and_final_layer_norms=True)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  _truncate(net, 'pairformer')
  # rf3's attention modules carry `force_bfloat16 = True` and cast the
  # activation inside forward, which here raises outright (bf16 activation
  # against an fp32 weight) rather than quietly degrading. Off everywhere: it
  # is a speed switch, and our graph runs this head in fp32.
  nbf = 0
  for mod in net.modules():
    if getattr(mod, 'force_bfloat16', False):
      mod.force_bfloat16 = False
      nbf += 1
  print('  cleared force_bfloat16 on %d submodules' % nbf)
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(s_inputs)[None], torch.tensor(s)[None],
              torch.tensor(z)[None], torch.tensor(pos.reshape(-1, 3))[None],
              torch.zeros(1, n, dtype=torch.long),
              torch.tensor(rep_idx))
  ref = {'pae': out['pae_logits'][0].numpy(),
         'pde': out['pde_logits'][0].numpy(),
         'plddt': out['plddt_logits'][0].numpy().reshape(n, slots, b_plddt),
         'resolved': out['exp_resolved_logits'][0].numpy().reshape(n, slots, 2)}
  return ref, s_inputs_ours, s, z, rep


def native_opendde(model, batch, pos, rng, n, max_atoms):
  """-> (native logits, s_inputs(ours-layout), s, z, rep coords) for opendde.

  opendde is the one port with its OWN head module
  (`network/opendde_confidence.py`), because its confidence runs on the
  STRUCTURAL token set rather than residues, at c_s = c_z = 384. Both sides take
  the atom layout as explicit arguments, so unlike every other adapter here this
  one needs no featurised batch at all -- the layout is synthesised and handed
  to both. `ours_opendde` below is a separate path for the same reason.
  """
  import torch

  from opendde.model.modules.confidence import ConfidenceHead

  ckpt = os.path.expanduser('~/opendde_weights/opendde.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.confidence_head.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[2])
                     for k in sub if k.startswith('pairformer_stack.blocks.'))
  c_z = sub['linear_no_bias_pae.weight'].shape[1]
  c_s_inputs = sub['linear_no_bias_s1.weight'].shape[1]
  slots, c_s, b_plddt = sub['plddt_weight'].shape
  b_pae = sub['linear_no_bias_pae.weight'].shape[0]
  print('  checkpoint: %d pairformer blocks, c_s %d, c_z %d, c_s_inputs %d, '
        '%d slots, %d plddt bins'
        % (n_blocks, c_s, c_z, c_s_inputs, slots, b_plddt))

  from converters.opendde import RESTYPE_PERM as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s_inputs[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  s_inputs_ours = s_inputs[:, idx]
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  # A SYNTHESISED layout, the one model.py builds for the structural set:
  # atom_to_token repeats the token index, atom_to_tokatom tiles the slot index,
  # and the rep atom is slot 0 of each token.
  a2t = np.repeat(np.arange(n), max_atoms)
  a2ta = np.tile(np.arange(max_atoms), n)
  rep_mask = np.zeros(n * max_atoms, bool)
  rep_mask[np.arange(n) * max_atoms] = True
  rep = pos.reshape(-1, 3)[rep_mask]

  net = ConfidenceHead(n_blocks=n_blocks, c_s=c_s, c_z=c_z,
                       c_s_inputs=c_s_inputs, b_pae=b_pae, b_pde=b_pae,
                       b_plddt=b_plddt, max_atoms_per_token=slots,
                       hidden_scale_up=True)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  _truncate(net.pairformer_stack)
  net.eval()
  feats = {'distogram_rep_atom_mask': torch.tensor(rep_mask),
           'atom_to_token_idx': torch.tensor(a2t),
           'atom_to_tokatom_idx': torch.tensor(a2ta)}
  with torch.no_grad():
    plddt, pae, pde, resolved = net(
        feats, torch.tensor(s_inputs)[None], torch.tensor(s)[None],
        torch.tensor(z)[None], torch.ones(1, n, n),
        torch.tensor(pos.reshape(-1, 3))[None, None])
  ref = {'plddt': plddt[0, 0].numpy(), 'pae': pae[0, 0].numpy(),
         'pde': pde[0, 0].numpy(), 'resolved': resolved[0, 0].numpy()}
  return ref, s_inputs_ours, s, z, rep


def ours_opendde(cfg, model_dir, pos, s_inputs, s, z, n, max_atoms):
  """Our OpenDDEConfidenceHead on the same synthesised layout."""
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import opendde_confidence

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)
  a2t = jnp.asarray(np.repeat(np.arange(n), max_atoms))
  a2ta = jnp.asarray(np.tile(np.arange(max_atoms), n))
  rep = jnp.asarray(pos.reshape(-1, 3)[np.arange(n) * max_atoms])

  def fwd():
    return opendde_confidence.OpenDDEConfidenceHead(
        cfg.evoformer.seq_channel, cfg.evoformer.pair_channel,
        s_inputs.shape[-1], cfg.global_config)(
            jnp.asarray(s_inputs), jnp.asarray(s), jnp.asarray(z), rep,
            a2t, a2ta, jnp.ones(n, jnp.float32))

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  nb = os.environ.get('BLOCKS')
  nb = None if nb is None else int(nb)
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      src = full.get('diffuser/' + sc)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        v = np.asarray(v, np.float32)
        if nb is not None and 'pairformer_stack' in sc:
          v = v[:nb]
        params[sc][k] = v
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our head is partly at init -- comparison is noise'
  return f.apply(params, jax.random.PRNGKey(0))


_OVERRIDE = {}


def native_boltz2(model, batch, pos, rng, n, max_atoms):
  """-> (ref logits, s_inputs, s, z, rep_native) from Boltz-2's ConfidenceModule.

  THE LAST HOLE in the panel. Its converter has been structurally complete for a
  while (66 unported -> 0) and its three graph branches landed with it, but the
  VALUES had never been compared -- which is the state [[boltz2-confidence-port]]
  records as "structural coverage COMPLETE, values UNVERIFIED".

  Everything is read off the checkpoint's own `confidence_model_args`, so a
  release that changes its mind fails in load_state_dict rather than comparing
  quietly: 8 pairformer blocks / 16 heads, 64 distance bins to max_dist 22, and
  the three `add_*` flags that decide whether the re-embedding runs at all
  (all True here) plus `use_separate_heads` for the intra/inter split.
  """
  import torch

  from boltz.model.modules.confidencev2 import ConfidenceModule

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = raw.get('state_dict', raw.get('model', raw))
  hp = raw.get('hyper_parameters', {})
  cm = dict(hp.get('confidence_model_args', {}))
  pre = 'confidence_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  token_s = sub['s_to_z.weight'].shape[1]
  token_z = sub['s_to_z.weight'].shape[0]
  print('  checkpoint: %d tensors, token_s %d, token_z %d, %d pairformer blocks'
        % (len(sub), token_s, token_z,
           cm.get('pairformer_args', {}).get('num_blocks')))
  # `bond_type_feature` and `maximum_bond_distance` are NOT in the checkpoint's
  # `confidence_model_args`, and both change the forward. Derived from the
  # tensors instead: `token_bonds_type` exists only when the flag is on, and
  # `token_bonds`'s input width is 1 + maximum_bond_distance.
  bond_type = 'token_bonds_type.weight' in sub
  max_bond = int(sub['token_bonds.weight'].shape[1]) - 1
  print('  derived: bond_type_feature=%s, maximum_bond_distance=%d'
        % (bond_type, max_bond))
  net = ConfidenceModule(
      token_s=token_s, token_z=token_z,
      pairformer_args=dict(cm['pairformer_args'], dropout=0.0),
      num_dist_bins=cm.get('num_dist_bins', 64),
      max_dist=cm.get('max_dist', 22),
      add_s_to_z_prod=cm.get('add_s_to_z_prod', False),
      add_s_input_to_s=cm.get('add_s_input_to_s', False),
      add_z_input_to_z=cm.get('add_z_input_to_z', False),
      bond_type_feature=bond_type, maximum_bond_distance=max_bond,
      confidence_args=cm.get('confidence_args'),
      conditioning_cutoff_min=4.0, conditioning_cutoff_max=20.0,
      # FROM THE CHECKPOINT'S OWN hyper_parameters, not ConfidenceModule's
      # defaults. Both default False there and are True in this checkpoint, and
      # they change the RELATIVE POSITION ENCODING -- so leaving them unpassed
      # builds a native this checkpoint never was.
      #
      # This was the WHOLE of boltz2's L4 gap (2026-09-10). Our rel_pos matches
      # native BIT-EXACTLY (max|d| 0.00000) once the flags are right; with them
      # wrong it reads corr 0.809, and that term's max|d| 0.67429 and
      # per-channel constant 0.2211 were to the digit the entire re-embedding
      # error, and therefore the entire cell.
      fix_sym_check=hp.get('fix_sym_check', False),
      cyclic_pos_enc=hp.get('cyclic_pos_enc', False))
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d missing, %d unexpected %s%s'
        % (len(missing), len(unexpected), list(missing)[:3],
           list(unexpected)[:3]))
  assert not missing, 'native confidence is missing %d tensors' % len(missing)
  # AND NOT UNEXPECTED EITHER. A tensor the checkpoint has and the module did
  # not build means a FLAG is wrong, and the module then silently skips that
  # term: `bond_type_feature` defaulted False, `token_bonds_type` came back
  # unexpected, and native dropped a term ours was adding -- an nn.Embedding's
  # row 0 on every pair, which is a learned constant and not a no-op.
  assert not unexpected, (
      'native confidence did not build %d checkpoint tensors: %s'
      % (len(unexpected), sorted(unexpected)[:6]))
  # BLOCKS truncates the confidence pairformer on BOTH sides (ours in `ours()`),
  # which splits "the re-embedding differs" from "the stack differs". BLOCKS=0
  # compares the re-embedded z straight into the heads.
  _nb = os.environ.get('BLOCKS')
  if _nb is not None:
    import torch.nn as _nn
    _l = net.pairformer_stack.layers
    net.pairformer_stack.layers = _nn.ModuleList(list(_l)[:int(_nb)])
    print('  native pairformer truncated to %s block(s)' % _nb)
  net.eval()

  t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)
  rng2 = np.random.default_rng(0)
  s_inputs = (rng2.normal(size=(n, token_s)) * 0.5).astype(np.float32)
  s_tr = (rng2.normal(size=(n, token_s)) * 0.5).astype(np.float32)
  z = (rng2.normal(size=(n, n, token_z)) * 0.5).astype(np.float32)

  # boltz works on a PACKED atom axis with a token -> representative-atom
  # gather, and this harness synthesises a dense (token, slot) layout in which
  # every slot is a real atom. So the packed axis is that layout flattened, and
  # `token_to_rep_atom` is a one-hot picking slot 0 of each token -- which is
  # what main compares `rep_ours` against.
  posd = np.asarray(pos)
  flat = posd.reshape(-1, 3)
  n_at = flat.shape[0]
  # THE REPRESENTATIVE ATOM IS THE PSEUDO-BETA, not slot 0. Our head gathers
  # `pseudo_beta_info.token_atoms_to_pseudo_beta`, and main asserts the two
  # sides embed the same coordinates -- picking slot 0 here read 4.56e+01 and
  # fired that assert, which is the whole point of it.
  rep_flat = np.asarray(
      batch.pseudo_beta_info.token_atoms_to_pseudo_beta.gather_idxs).reshape(-1)
  assert rep_flat.shape == (n,), rep_flat.shape
  t2r = np.zeros((n, n_at), np.float32)
  t2r[np.arange(n), rep_flat] = 1.0
  a2t = np.repeat(np.arange(n), max_atoms)
  feats = {
      'token_pad_mask': torch.ones(1, n),
      'atom_pad_mask': torch.ones(1, n_at),
      'token_to_rep_atom': t(t2r)[None],
      'atom_to_token': t(np.eye(n, dtype=np.float32)[a2t])[None],
      'asym_id': t(np.asarray(batch.token_features.asym_id), torch.long)[None],
      'residue_index': t(np.asarray(batch.token_features.residue_index),
                         torch.long)[None],
      'entity_id': t(np.asarray(batch.token_features.entity_id),
                     torch.long)[None],
      'token_index': t(np.asarray(batch.token_features.token_index),
                       torch.long)[None],
      'sym_id': t(np.asarray(batch.token_features.sym_id), torch.long)[None],
      'mol_type': torch.zeros(1, n, dtype=torch.long),
      'token_bonds': torch.zeros(1, n, n, 1),
      'type_bonds': torch.zeros(1, n, n, dtype=torch.long),
      'contact_conditioning': torch.nn.functional.one_hot(
          torch.zeros(1, n, n, dtype=torch.long), 5).float(),
      'contact_threshold': torch.zeros(1, n, n),
      'target_pair_mask': None,
      # `cyclic_pos_enc` reads this; absent, the encoder KeyErrors.
      'cyclic_period': torch.zeros(1, n),
  }
  with torch.no_grad():
    out = net(s_inputs=t(s_inputs)[None], s=t(s_tr)[None], z=t(z)[None],
              x_pred=t(flat)[None], feats=feats,
              pred_distogram_logits=torch.zeros(1, n, n, 64))
  g = lambda k: np.asarray(out[k])[0]
  ref = {'pae': g('pae_logits'), 'pde': g('pde_logits'),
         'plddt': g('plddt_logits'), 'resolved': g('resolved_logits')}
  for k, v in ref.items():
    print('  native %-9s %s' % (k, v.shape))
  return ref, s_inputs, s_tr, z, flat[rep_flat]


def native_esmfold2(model, batch, pos, rng, n, max_atoms):
  """-> (ref logits, s_inputs, s, z, rep_native) from the ESMFold2 DUMP.

  THE ONE esmfold2 CELL THE REFERENCE CANNOT SERVE. `esmfold2_reference.py`
  stops at the diffusion; it has no confidence head, so this is the only module
  gate in the family that needs a native run -- which is why
  `esmfold2_oracle_6mrr.py` hooks `confidence_head` with `with_kwargs=True` and
  parks its 11 inputs and 14 outputs in the dump.

  Everything our head consumes is INJECTED from that dump, so both sides run on
  the tensors native actually used:

      z, s_inputs  the real trunk output for this fold (256- and 451-wide)
      x_pred       native's sampled coordinates, PACKED (576 slots, 574 real)

  Two layout translations, both of them the ones this family always needs:

    * s_inputs is 451 = [atom 384 | restype 33 | profile 33 | deletion 1] and
      ours is 447 = [restype 31 | profile 31 | deletion 1 | atom 384], with the
      restype blocks a PERMUTATION (ESM puts the MSA gap at class 1, below the
      residues -- `converters/esmfold2.esm_class_of_af3`).
    * the atom axis is packed on native's side and dense (token, slot) on ours,
      and the two do not even agree on the atom COUNT: ours carries the terminal
      OXT, ESMFold2's PROTEIN_HEAVY_ATOMS table does not. `esmfold2_dumps.atom_map`
      matches them BY NAME, which is the only way that is safe.
  """
  import esmfold2_dumps
  from converters import esmfold2 as CV

  io = esmfold2_dumps.module_io(model, tag='conf')
  g = lambda k: np.asarray(io[k])[0]
  nat = esmfold2_dumps.native(model)
  f = {k[5:]: v[0] for k, v in nat.items() if k.startswith('feat.')}
  ref_idx, our_flat = esmfold2_dumps.atom_map(batch, f)
  print('  dump: %d tokens, %d packed atoms, %d matched by name'
        % (g('in.z').shape[0], g('in.x_pred').shape[0], len(ref_idx)))

  # s_inputs: 451 (ESM) -> 447 (ours), the permutation not a slice.
  n_af3, n_esm = 31, 33
  si451 = g('in.s_inputs')
  cols = np.asarray(CV.esm_class_of_af3(n_af3))
  s_inputs = np.zeros((si451.shape[0], 2 * n_af3 + 1 + 384), np.float32)
  s_inputs[:, :n_af3] = si451[:, 384 + cols]
  s_inputs[:, n_af3:2 * n_af3] = si451[:, 384 + n_esm + cols]
  s_inputs[:, 2 * n_af3] = si451[:, 384 + 2 * n_esm]
  s_inputs[:, 2 * n_af3 + 1:] = si451[:, :384]

  # x_pred: packed -> our dense (token, slot). Unmatched slots stay zero and
  # are masked out of every comparison.
  x = g('in.x_pred')
  pos_new = np.zeros_like(np.asarray(pos))
  pos_new.reshape(-1, 3)[our_flat] = x[ref_idx]
  _OVERRIDE['pos'] = pos_new

  # The per-atom logits arrive on the packed axis too.
  def dense(v, last):
    out = np.zeros((n, max_atoms, last), np.float32)
    out.reshape(-1, last)[our_flat] = v[ref_idx]
    return out

  ref = {
      'pae': g('out.pae_logits'),
      'pde': g('out.pde_logits'),
      'plddt': dense(g('out.plddt_logits'), g('out.plddt_logits').shape[-1]),
      'resolved': dense(g('out.resolved_logits'), 2),
  }
  # THE REPRESENTATIVE ATOM. ESMFold2 hands its head an explicit
  # `distogram_atom_idx` per token; our head gathers the pseudo-beta. main
  # asserts the two land on the SAME coordinates, which is the check that a
  # geometry mismatch cannot hide behind a plausible correlation.
  rep = x[np.asarray(g('in.distogram_atom_idx')).astype(int)]
  # `s` is the single track, and ESMFold2 has none (PAIR_ONLY_TRUNK): its head
  # reads z and s_inputs only. Zeros, matching what our head is handed.
  s = np.zeros((n, 384), np.float32)
  return ref, s_inputs, s, g('in.z'), rep


NATIVES = {m: native_protenix for m in _PROTENIX_CKPT}
NATIVES.update({m: native_esmfold2 for m in ('esmfold2', 'esmfold2_fast')})
NATIVES['boltz2'] = native_boltz2
NATIVES.update({m: native_of3 for m in _OF3_CKPT})
NATIVES['intellifold2'] = native_if2
NATIVES['rosettafold3'] = native_rf3
NATIVES['opendde'] = native_opendde


def ours(model, cfg, model_dir, batch, pos, s_inputs, s, z):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import confidence_head, evoformer as ev

  # BF16=all runs OUR head in bfloat16. ESMFold2's own head casts its folding
  # trunk with `autocast(dtype=torch.bfloat16)` whenever the pair is on a GPU
  # (modeling_esmfold2.py ConfidenceHead.forward), and the dump was produced on
  # one -- so for that family the fp32 default compares two different
  # precisions, not two implementations.
  cfg.global_config.bfloat16 = os.environ.get('BF16', 'none')
  nb = os.environ.get('BLOCKS')
  nb = None if nb is None else int(nb)
  if nb is not None:
    cfg.heads.confidence.pairformer.num_layer = nb
  full = afp.get_model_haiku_params(model_dir=model_dir)

  def fwd():
    return confidence_head.ConfidenceHead(
        cfg.heads.confidence, cfg.global_config)(
            dense_atom_positions=jnp.asarray(pos),
            embeddings={'pair': jnp.asarray(z), 'single': jnp.asarray(s),
                        'target_feat': jnp.asarray(s_inputs)},
            seq_mask=batch.token_features.mask,
            token_atoms_to_pseudo_beta=(
                batch.pseudo_beta_info.token_atoms_to_pseudo_beta),
            asym_id=batch.token_features.asym_id,
            token_features=batch.token_features,
            bond_matrix=ev.token_bond_matrix(batch, symmetrize=True),
            bond_type_matrix=ev.token_bond_type_matrix(batch, symmetrize=True),
            atom_name_chars=batch.ref_structure.atom_name_chars)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      src = full.get('diffuser/' + sc)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        v = np.asarray(v, np.float32)
        # under BLOCKS the stacked scopes must be sliced to the same depth the
        # native side was truncated to; the leading axis is the block index.
        if nb is not None and '__layer_stack' in sc:
          v = v[:nb]
        params[sc][k] = v
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our head is partly at init -- comparison is noise'
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
    raise SystemExit('no native adapter for %r; have %s'
                     % (args.model, sorted(NATIVES)))

  import fold_check
  seq, _native = fold_check.parse_ca(args.pdb)
  print('%s confidence head, %d tokens:' % (args.model, len(seq)))
  batch, cfg, model_dir, pos, rng, n, max_atoms = our_inputs(
      args.model, seq, args.model_dir)
  ref, s_inputs, s, z, rep_native = NATIVES[args.model](
      args.model, batch, pos, rng, n, max_atoms)
  # An adapter may REPLACE the positions: the esmfold2 cell is dump-driven, so
  # both sides have to embed native's own sampled coordinates rather than this
  # harness's synthetic ones.
  pos = _OVERRIDE.get('pos', pos)

  # Which atom represents a token in the distance embedding is itself a
  # convention, and it differs: rosettafold3 uses the token-centre CA (dense
  # slot 1), everyone else the pseudo-beta gather. Our graph's choice is fixed
  # in confidence_head.py, so mirror it here and assert the two sides land on
  # the same coordinates -- bit-identical, not merely close. Without this a
  # geometry mismatch reads as a plausible correlation, and the first rf3 run
  # (CA on native, pseudo-beta here) tripped exactly this assert.
  if args.model == 'opendde':
    rep_ours = np.asarray(pos)[:, 0, :]          # slot 0, synthesised layout
  elif args.model == 'rosettafold3':
    rep_ours = np.asarray(pos)[:, 1, :]
  else:
    from alphafold3.model.atom_layout import atom_layout
    rep_ours = np.asarray(atom_layout.convert(
        batch.pseudo_beta_info.token_atoms_to_pseudo_beta, pos,
        layout_axes=(-3, -2)))
  d = np.abs(rep_ours - rep_native).max()
  print('  rep-atom coords agree to %.2e (must be 0)' % d)
  assert d == 0, 'the two heads are embedding different geometry'

  if args.model == 'opendde':
    out = ours_opendde(cfg, model_dir, pos, s_inputs, s, z, n, max_atoms)
    # opendde's head works on a FLAT atom axis and the layout here is
    # synthesised, so every slot is a real atom: reshape ours to the dense
    # (token, slot) shape the comparison below uses and mask nothing out.
    out = dict(out)
    for key in ('predicted_lddt', 'predicted_experimentally_resolved'):
      out[key] = np.asarray(out[key]).reshape(n, max_atoms)
    atom_mask = np.ones((n, max_atoms), bool)
  else:
    out = ours(args.model, cfg, model_dir, batch, pos, s_inputs, s, z)
    atom_mask = np.asarray(batch.predicted_structure_info.atom_mask) > 0
  # Our head returns EXPECTATIONS, not logits, so native's logits go through the
  # same reduction to be comparable. NOTE what that does NOT gate: both sides
  # use OUR bin centers, so a bin-convention error would cancel here. Those were
  # checked by reading both definitions instead, and they agree exactly --
  # protenix pae/pde are min 0 / max 32 / 64 bins with centers
  # `linspace(0, 32-w, 64) + w/2`, which is what our max_error_bin=31.0 plus the
  # catch-all bin produces (0.25 ... 31.75); plddt is 50 bins over [0, 1] on both
  # sides. Worth re-checking per vendor: confidence bugs here have been bin and
  # layout bugs, not weight bugs (memory rf3-padding-confidence-bug).
  def expectation(logits, max_bin, n_bins):
    breaks = np.linspace(0.0, max_bin, n_bins - 1)
    step = breaks[1] - breaks[0]
    centers = np.concatenate([breaks + step / 2, [breaks[-1] + 1.5 * step]])
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    return (p * centers).sum(-1)

  ccfg = cfg.heads.confidence
  pae_ref = expectation(ref['pae'], ccfg.pae.max_error_bin, ccfg.pae.num_bins)
  # NOT symmetrised again here: protenix symmetrises the PAIR before the LN and
  # the projection (`pde_ln(z + z.transpose)`), so ref['pde'] is already the
  # symmetric logit. Our graph symmetrises the LOGITS instead
  # (left + swap(left)), and LayerNorm makes those two different functions --
  # which is what the number below is measuring.
  pde_ref = expectation(ref['pde'], ccfg.max_error_bin, ccfg.num_bins)
  # FLOOR -- how much of the CLOSE band here is resolvable at all? These heads
  # sit on top of a 4-block pairformer fed s_inputs, s and z, and the outputs are
  # EXPECTATIONS over softmaxed bins, so a small input shift can move an entry by
  # a bin's worth. Perturb the injected inputs and see how far our own outputs
  # move; parity_audit grades a row at or below its own floor as FLOOR.
  eps = float(os.environ.get('FLOOR') or 0)
  if eps and not _OVERRIDE.get('dump_driven'):
    rngf = np.random.default_rng(1234)
    pert = lambda x: (np.asarray(x)
                      * (1 + eps * rngf.normal(size=np.shape(x)))).astype(
                          np.asarray(x).dtype)
    try:
      out_p = ours(args.model, cfg, model_dir, batch, pos,
                   pert(s_inputs), pert(s), pert(z))
    except AssertionError as e:
      # `ours()` IS NOT IDEMPOTENT for every model: called twice in one process
      # it builds 51 scopes for opendde where the first call built 52, and the
      # 66 it cannot fill are stock-AF3 names (`confidence_head/~_embed_features
      # /...`) rather than opendde's own module's -- so the second build is a
      # DIFFERENT head. Reported, never swallowed: a floor that cannot be
      # measured has to look different from one that came out small.
      print('  FLOOR UNAVAILABLE (%s): ours() is not idempotent for this model '
            '-- see HOLES.md' % e)
      out_p = None
    if out_p is not None:
      print('  FLOOR: our own outputs after a %g relative perturbation of the '
            'injected s_inputs / s / z' % eps)
      _cmp('full_pae_floor', out_p['full_pae'], out['full_pae'])
      _cmp('full_pde_floor', out_p['full_pde'], out['full_pde'])
      _OVERRIDE['floor_out'] = out_p

  _cmp('full_pae', out['full_pae'], pae_ref)
  _cmp('full_pde', out['full_pde'], pde_ref)
  # Compare on the slots both layouts have: of3 has 23, our dense layout 24.
  lddt = ref['plddt'].reshape(n, -1, ref['plddt'].shape[-1])
  k = min(lddt.shape[1], max_atoms)
  lddt = lddt[:, :k]
  amask = atom_mask[:, :k]
  p = np.exp(lddt - lddt.max(-1, keepdims=True)); p = p / p.sum(-1, keepdims=True)
  bw = 1.0 / p.shape[-1]
  if _OVERRIDE.get('floor_out') is not None:
    _f = _OVERRIDE['floor_out']
    _cmp('plddt_floor', _f['predicted_lddt'][:, :k],
         out['predicted_lddt'][:, :k])
    _cmp('resolved_floor', _f['predicted_experimentally_resolved'][:, :k],
         out['predicted_experimentally_resolved'][:, :k])
  _cmp('plddt', out['predicted_lddt'][:, :k],
       (p * np.arange(0.5 * bw, 1.0, bw)).sum(-1) * 100.0, amask)
  if os.environ.get('DIAG'):
    # PER-ATOM vs PER-TOKEN. A per-atom quantity can disagree two ways: the
    # heads differ, or the two atom layouts are misaligned WITHIN a token. The
    # per-token mean over matched atoms is invariant to the second, so if it
    # correlates while the per-atom does not, the fault is the mapping.
    _lo = (p * np.arange(0.5 * bw, 1.0, bw)).sum(-1) * 100.0
    _og = np.asarray(out['predicted_lddt'][:, :k])
    _w = amask.astype(np.float64)
    _tn = (_lo * _w).sum(1) / np.maximum(_w.sum(1), 1)
    _to = (_og * _w).sum(1) / np.maximum(_w.sum(1), 1)
    print('  DIAG plddt per-TOKEN mean: corr %.6f  (per-atom corr above)'
          % np.corrcoef(_to, _tn)[0, 1])
    _i = int(np.argmax(_w.sum(1)))
    print('  DIAG token %d, %d atoms: ours %s' % (
        _i, int(_w[_i].sum()),
        np.round(_og[_i][amask[_i]][:8], 1).tolist()))
    print('  DIAG                      native %s' % (
        np.round(_lo[_i][amask[_i]][:8], 1).tolist(),))
  res = ref['resolved'].reshape(n, -1, 2)[:, :k]
  pr = np.exp(res - res.max(-1, keepdims=True)); pr = pr / pr.sum(-1, keepdims=True)
  _cmp('resolved', out['predicted_experimentally_resolved'][:, :k],
       pr[..., 1], amask)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
