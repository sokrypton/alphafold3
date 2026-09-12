"""L1i -- the TRUNK PAIR INIT (z before the pairformer). Nothing else gates it.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:<vendor> \
    ~/venv/bin/python dev/oracles/trunk_init_parity.py boltz2

`trunk_parity.py` feeds the pairformer stack SYNTHETIC s and z, which is the
right way to gate 48 blocks of arithmetic and means it never sees the relative
position encoding, the bond embeddings, or anything else that builds z in the
first place. `conditioning_parity.py` gates the DIFFUSION conditioner's copy of
the relative encoding, not the trunk's.

That gap is not hypothetical. boltz2's relative-CHAIN bucket convention is
certified by its own module at the diffusion site (pair_cond 1.000000) and
CONTRADICTED by the fold at the trunk site (mean 0.775 against 0.539 over 15
paired samples, with three at ~1.5). Both call sites build the same features
from the same weights, so one of those two statements is measuring something
else -- and no cell could say which.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_parity import _cmp                      # noqa: E402
from diffusion_parity import _stub_layer_norm           # noqa: E402


def native_boltz2(model, batch, s_inputs, bonds, bond_types, n):
  """-> z_init from Boltz-2's own layers, on OUR features.

  boltz2 has no `z_init` MODULE: the five terms are assembled inline in
  `Boltz2.forward` (models/boltz2.py:429-438), so they are assembled here the
  same way from the same checkpoint tensors -- z_init_1/z_init_2 over s_inputs,
  the relative position encoding, the bond indicator, the bond ORDER embedding,
  and the contact conditioning. The last two contribute on EVERY pair even with
  nothing annotated (an nn.Embedding's row 0 and `encoding_unspecified`), which
  is why they cannot be skipped for a plain protein.
  """
  import torch

  from boltz.model.modules.encodersv2 import RelativePositionEncoder
  from boltz.model.modules.trunkv2 import ContactConditioning

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = raw.get('state_dict', raw.get('model', raw))
  g = lambda k: sd[k]
  token_z, token_s = g('z_init_1.weight').shape
  print('  checkpoint: token_s %d, token_z %d' % (token_s, token_z))

  t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)
  si = t(s_inputs)[None]
  lin = lambda w, x: torch.nn.functional.linear(x, g(w))
  z = lin('z_init_1.weight', si)[:, :, None] + lin('z_init_2.weight', si)[:, None, :]

  tf = batch.token_features
  feats = {k: t(np.asarray(getattr(tf, k)).astype(np.int64), torch.long)[None]
           for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                     'sym_id')}
  feats['mol_type'] = torch.zeros(1, n, dtype=torch.long)
  # FROM THE CHECKPOINT, not the class defaults: `fix_sym_check` defaults False
  # and is True in this checkpoint, and it changes the encoding even on a
  # MONOMER (max|d| 0.164 on 6MRR's features). Building native at the default
  # compares against a model this checkpoint never was -- which is exactly how
  # boltz2's L4 gap hid. B2_NO_SYM_FIX=1 restores the old (wrong) construction
  # for the A/B.
  _hp = raw.get('hyper_parameters', {}) if isinstance(raw, dict) else {}
  _fx = bool(_hp.get('fix_sym_check', False)) and not os.environ.get('B2_NO_SYM_FIX')
  print('  native RelativePositionEncoder(fix_sym_check=%s)' % _fx)
  rp = RelativePositionEncoder(token_z=token_z, fix_sym_check=_fx)
  rp.load_state_dict({k[len('rel_pos.'):]: v for k, v in sd.items()
                      if k.startswith('rel_pos.')}, strict=False)
  rp.eval()
  with torch.no_grad():
    z = z + rp(feats)
    # The bond terms, on OUR matrices -- the harness rule everywhere here: feed
    # the vendor our features so a disagreement is ours, not the featuriser's.
    z = z + lin('token_bonds.weight', t(bonds)[None][..., None])
    z = z + torch.nn.functional.embedding(
        t(bond_types, torch.long)[None], g('token_bonds_type.weight'))
    cc = ContactConditioning(token_z=token_z, cutoff_min=4.0, cutoff_max=20.0)
    missing, _ = cc.load_state_dict(
        {k[len('contact_conditioning.'):]: v for k, v in sd.items()
         if k.startswith('contact_conditioning.')}, strict=False)
    assert not missing, missing
    cc.eval()
    contact = torch.zeros(1, n, n, len(_CONTACT_CLASSES))
    contact[..., 0] = 1.0                      # UNSPECIFIED on every pair
    z = z + cc({'contact_conditioning': contact,
                'contact_threshold': torch.zeros(1, n, n)})
  return np.asarray(z)[0]


_CONTACT_CLASSES = range(5)


def native_boltz2_loop(model, batch, s_inputs, bonds, bond_types, n, passes):
  """-> (s, z) after `passes` of Boltz-2's own recycle loop, on OUR features.

  PASSES>=1 turns this gate from "the tensor the trunk starts from" into "the
  loop that consumes it", which is the one thing left un-gated when z-init, the
  MSA module, the pairformer stack and the diffusion conditioner are each exact
  on their own and the composed fold still prefers a z-init that is NOT exact.

  The loop is boltz's, verbatim (models/boltz2.py:461-494):

      s = s_init + s_recycle(s_norm(s))
      z = z_init + z_recycle(z_norm(z))
      z = z + msa_module(z, s_inputs, feats)          # templates skipped: 6MRR
      s, z = pairformer(s, z, mask, pair_mask)        # has none

  starting from s = z = ZERO, which matters: `s_norm` and `z_norm` are
  LayerNorms WITH a bias, so LayerNorm(0) is that bias and the first pass adds
  `recycle(bias)`, not nothing.
  """
  import torch

  from boltz.model.layers.pairformer import PairformerModule
  from boltz.model.modules.trunkv2 import MSAModule

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = raw.get('state_dict', raw.get('model', raw))
  hp = raw.get('hyper_parameters', {})
  g = lambda k: sd[k]
  t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)
  lin = lambda w, x: torch.nn.functional.linear(x, g(w))

  z_init = t(native_boltz2(model, batch, s_inputs, bonds, bond_types, n))[None]
  si = t(s_inputs)[None]
  s_init = lin('s_init.weight', si)
  token_s, token_z = s_init.shape[-1], z_init.shape[-1]

  # The MSA module, with the same feature construction msa_parity validates --
  # 6MRR has no alignment, so the "MSA" is the query and the deletions are zero.
  msub = {k[len('msa_module.'):]: v for k, v in sd.items()
          if k.startswith('msa_module.')}
  msa_s, in_dim = msub['msa_proj.weight'].shape
  # THE REAL ROWS ONLY. `batch.msa.rows` is padded to the bucket (16384 for
  # 6MRR, which has no alignment at all), and handing 16384 rows to the MSA
  # module is neither what our graph runs nor something that fits.
  raw_msa = np.asarray(batch.msa.rows)
  keep_rows = np.asarray(batch.msa.mask).max(-1) > 0
  raw_msa = raw_msa[keep_rows][:int(os.environ.get('N_MSA', 8))]
  n_seq = raw_msa.shape[0]
  dele = np.asarray(batch.msa.deletion_matrix)[keep_rows][:n_seq]
  # IN BOLTZ'S CLASS ORDER, not ours. Our converter reorders boltz's 33 restype
  # columns into our 31 classes (`BOLTZ2_RESTYPE_PERM`), so our side applies
  # REMAPPED weights to an AF3-ordered one-hot while native applies its ORIGINAL
  # weights -- our class j is boltz's column PERM[j]. Feeding native ours
  # instead read s corr 0.95 / z 0.93 at one pass, which is what a permuted
  # vocabulary looks like when most columns still land somewhere plausible.
  from converters.boltz2 import BOLTZ2_RESTYPE_PERM
  perm = np.asarray(BOLTZ2_RESTYPE_PERM)
  paired = (in_dim - 33) == 3
  onehot = np.zeros((n_seq, n, 33), np.float32)
  cols = perm[np.clip(raw_msa, 0, len(perm) - 1)]
  onehot[np.arange(n_seq)[:, None], np.arange(n)[None, :], cols] = 1.0
  has_del = (dele > 0).astype(np.float32)
  del_val = (2.0 / np.pi) * np.arctan(np.asarray(dele, np.float32) / 3.0)
  print('  msa: %d rows, 33 boltz classes (msa_proj in %d, paired %s), '
        '%d deletions' % (n_seq, in_dim, paired, int(has_del.sum())))
  n_blocks = 1 + max(int(k.split('.')[1]) for k in msub
                     if k.startswith('layers.'))
  msa_net = MSAModule(msa_s=msa_s, token_z=token_z, token_s=token_s,
                      msa_blocks=n_blocks, msa_dropout=0.0, z_dropout=0.0,
                      use_paired_feature=paired, subsample_msa=False)
  miss, _ = msa_net.load_state_dict(msub, strict=False)
  assert not miss, list(miss)[:3]
  zero_msa = np.zeros((n_seq, n), np.float32)
  # THE QUERY ROW IS MARKED PAIRED. boltz sets `is_paired` = 1 on row 0 and 0
  # elsewhere for an unpaired MSA, which our graph reproduces
  # (evoformer.py: `query_paired = 1.0` for boltz2, 0.0 for rf3, and the two
  # vendors genuinely disagree on what the flag means). Feeding zeros gives
  # native a different embedding for its query row than ours has.
  is_paired = np.zeros((n_seq, n), np.float32)
  is_paired[0] = 1.0
  feats = {'msa': t(onehot)[None], 'has_deletion': t(has_del)[None],
           'deletion_value': t(del_val)[None], 'msa_paired': t(is_paired)[None],
           'msa_mask': torch.ones(1, n_seq, n),
           'token_pad_mask': torch.ones(1, n),
           'target_pair_mask': None}

  psub = {k[len('pairformer_module.'):]: v for k, v in sd.items()
          if k.startswith('pairformer_module.')}
  pargs = hp.get('pairformer_args', {})
  # v2=True. The checkpoint's pairformer layers carry `pre_norm_s`, where the
  # default (v1) layer builds `attention.norm_s` -- 128 tensors missing, and
  # nothing else in the panel needed this flag. Read off the layer-0 key set,
  # not from the hparams, which do not mention it.
  pf = PairformerModule(token_s=token_s, token_z=token_z,
                        num_blocks=pargs.get('num_blocks', 64),
                        num_heads=pargs.get('num_heads', 16), dropout=0.0,
                        post_layer_norm=pargs.get('post_layer_norm', False),
                        v2=True)
  miss, _ = pf.load_state_dict(psub, strict=False)
  assert not miss, list(miss)[:3]
  print('  native: msa %d tensors, pairformer %d tensors, %d blocks'
        % (len(msub), len(psub), pargs.get('num_blocks', 64)))
  msa_net.eval(); pf.eval()

  mask = torch.ones(1, n)
  pair_mask = mask[:, :, None] * mask[:, None, :]
  s = torch.zeros_like(s_init)
  z = torch.zeros_like(z_init)
  with torch.no_grad():
    for i in range(passes):
      s = s_init + lin('s_recycle.weight',
                       torch.nn.functional.layer_norm(
                           s, (token_s,), g('s_norm.weight'), g('s_norm.bias')))
      z = z_init + lin('z_recycle.weight',
                       torch.nn.functional.layer_norm(
                           z, (token_z,), g('z_norm.weight'), g('z_norm.bias')))
      z = z + msa_net(z, si, feats)
      s, z = pf(s, z, mask=mask, pair_mask=pair_mask)
      print('  native pass %d: s rms %.4f, z rms %.4f'
            % (i + 1, float(s.pow(2).mean().sqrt()),
               float(z.pow(2).mean().sqrt())))
  return np.asarray(s)[0], np.asarray(z)[0]


_OURS = {}


def native_rf3(model, batch, s_inputs, bonds, bond_types, n):
  """-> z_init from RoseTTAFold3's own `FeatureInitializer`, on OUR features.

  Three lines of its forward, and the middle one is the trap:

      Z = to_z_init_i(S).unsqueeze(-3) + to_z_init_j(S).unsqueeze(-2)
      Z = Z + relative_position_encoding(f)
      Z = Z + process_token_bonds(f["token_bonds"][..., None])

  `unsqueeze(-3)` broadcasts along the ROW axis, so `to_z_init_i` is the COLUMN
  embedder and `to_z_init_j` the ROW one -- the opposite of what the names say.
  `converters/rosettafold3.py` already crosses them for that reason; this gate
  is what can prove it, since a transposed pair init is invisible on a symmetric
  input and this one is not symmetric.
  """
  import torch

  from rf3.model.layers.pairformer_layers import RelativePositionEncoding

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
  pre = 'shadow.feature_initializer.'
  g = lambda k: raw[pre + k]
  t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)
  lin = lambda w, x: torch.nn.functional.linear(x, g(w))

  c_z, c_s_inputs = g('to_z_init_i.weight').shape
  print('  checkpoint: c_z %d, c_s_inputs %d' % (c_z, c_s_inputs))
  # S_INPUTS IN RF3'S OWN LAYOUT, and the gate's synthetic 384-wide array is not
  # it: rf3's is 449 channels [a(384), restype(32), profile(32), deletion], with
  # its OWN restype alphabet (`_AF3_TO_RF3_AATYPE`, which transposes G/C against
  # of3's). Built here and handed back in OUR 447-wide layout so both sides see
  # the same numbers -- the same contract `conditioning_parity` uses.
  from converters.rosettafold3 import _AF3_TO_RF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  rng = np.random.default_rng(0)
  s449 = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  # The columns our 447-wide layout has no slot for are ZERO, which is what the
  # featuriser puts there for every input in this panel -- see the long note in
  # conditioning_parity.native_rf3.
  s449[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  _OURS['s_inputs'] = s449[:, idx]
  si = t(s449)
  # THE AXES, VERBATIM. rf3 writes
  #     Z = to_z_init_i(S).unsqueeze(-3) + to_z_init_j(S).unsqueeze(-2)
  # and on an [I, c] tensor `unsqueeze(-3)` gives [1, I, c] -- broadcast over
  # ROWS, so `to_z_init_i` is the COLUMN embedder -- while `unsqueeze(-2)` gives
  # [I, 1, c], the ROW one. Writing both with `[..., None, :]` (which is
  # unsqueeze(-2)) and then swapping the NAMES to compensate lands on the same
  # numbers and hides the reasoning; it also read corr 0.2799 until the swap,
  # which is what a transposed pair init looks like. Note this is the opposite
  # convention to protenix and of3, whose `[..., None, :]` term is the ROW.
  z = (lin('to_z_init_i.weight', si)[None, :, :]
       + lin('to_z_init_j.weight', si)[:, None, :])

  tf = batch.token_features
  f = {k: t(np.asarray(getattr(tf, k)).astype(np.int64), torch.long)
       for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                 'sym_id')}
  rp = RelativePositionEncoding(r_max=32, s_max=2, c_z=c_z)
  rp.load_state_dict({'linear.weight': g('relative_position_encoding.linear.weight')},
                     strict=False)
  rp.eval()
  with torch.no_grad():
    if not os.environ.get('NORELPE'):
      z = z + rp(f)
    else:
      print('  NORELPE: relative position encoding dropped on both sides')
    z = z + lin('process_token_bonds.weight', t(bonds)[..., None])
  return np.asarray(z)


_OF3_CKPT = {'openfold3': 'of3-p2-155k.pt', 'openbind0': 'of3-ob-174k.pt'}


def native_of3(model, batch, s_inputs, bonds, bond_types, n):
  """-> z_init from OpenFold3's own `InputEmbedder` tail (of3, openbind0).

      z = linear_z_i(s_input)[..., None, :] + linear_z_j(s_input)[..., None, :, :]
      z = z + linear_relpos(relpos_complex(batch, r_max, s_max))
      z = z + linear_token_bonds(token_bonds[..., None])

  The axes are the OPPOSITE of rf3's: here `[..., None, :]` on an [n, c] tensor
  is [n, 1, c], the ROW, and `[..., None, :, :]` the column -- where rf3's
  `unsqueeze(-3)` makes its first term the column. Same-looking code, mirrored
  meaning, which is exactly how the rf3 adapter got it wrong first.
  """
  import torch

  from openfold3.core.utils.relpos import relpos_complex

  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'input_embedder.'
  g = lambda k: sd[pre + k]
  t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)
  lin = lambda w, x: torch.nn.functional.linear(x, g(w))

  c_z, c_s_inputs = g('linear_z_i.weight').shape
  n_relpos = g('linear_relpos.weight').shape[1]
  print('  checkpoint: c_z %d, c_s_inputs %d, relpos %d'
        % (c_z, c_s_inputs, n_relpos))
  # of3's 449-channel s_inputs with ITS restype permutation, handed back in our
  # 447-wide layout -- the same contract conditioning_parity uses.
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  rng = np.random.default_rng(0)
  s449 = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s449[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  _OURS['s_inputs'] = s449[:, idx]
  si = t(s449)

  tf = batch.token_features
  b = {k: t(np.asarray(getattr(tf, k)).astype(np.int64), torch.long)
       for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                 'sym_id')}
  # r_max / s_max are of3's own config values, and 2*(2*r+2) + (2*s+2) + 1 has
  # to come out at the relpos linear's input width -- asserted, not assumed.
  r_max, s_max = 32, 2
  assert 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1 == n_relpos, n_relpos
  with torch.no_grad():
    z = (lin('linear_z_i.weight', si)[..., None, :]
         + lin('linear_z_j.weight', si)[..., None, :, :])
    if not os.environ.get('NORELPE'):
      z = z + lin('linear_relpos.weight',
                  relpos_complex(b, r_max, s_max).to(torch.float32))
    else:
      print('  NORELPE: relative position encoding dropped on both sides')
    z = z + lin('linear_token_bonds.weight', t(bonds)[..., None])
  return np.asarray(z)


_PROTENIX_SRC = {
    'protenix1': ('protenix', '~/protenix_weights/'
                  'protenix_base_default_v1.0.0.pt'),
    'protenix2': ('protenix', '~/protenix_weights/protenix-v2.pt'),
    'opendde': ('opendde', '~/opendde_weights/opendde.pt'),
}


def native_protenix(model, batch, s_inputs, bonds, bond_types, n):
  """-> z_init from protenix's own `Protenix.forward` head (also opendde).

      s_init = linear_no_bias_sinit(s_inputs)
      z_init = zinit1(s_init)[..., None, :] + zinit2(s_init)[..., None, :, :]
      z_init = z_init + relative_position_encoding(relp)
      z_init = z_init + linear_no_bias_token_bond(token_bonds[..., None])

  THE COMPOSE is what makes this family different from of3 and rf3: the two
  pair projections read `s_init`, not `s_inputs`, so the converter has to fold
  `zinit @ sinit` into one matrix against our single projection of target_feat.
  A gate on z_init is the only thing that can check that fold, and this is it.

  Axes as of3's: `[..., None, :]` is the ROW.
  """
  import importlib

  import torch

  _stub_layer_norm()
  pkg, ckpt_path = _PROTENIX_SRC[model]
  RelativePositionEncoding = getattr(
      importlib.import_module(pkg + '.model.modules.embedders'),
      'RelativePositionEncoding')

  sd = torch.load(os.path.expanduser(ckpt_path), map_location='cpu',
                  weights_only=False)
  sd = sd.get('model', sd.get('state_dict', sd))
  g = lambda k: sd['module.' + k]
  t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)
  lin = lambda w, x: torch.nn.functional.linear(x, g(w))

  c_s, c_s_inputs = g('linear_no_bias_sinit.weight').shape
  c_z = g('linear_no_bias_zinit1.weight').shape[0]
  print('  checkpoint: c_z %d, c_s %d, c_s_inputs %d' % (c_z, c_s, c_s_inputs))
  from converters.openfold3 import _AF3_TO_OF3_AATYPE as _remap
  idx = np.concatenate([384 + np.asarray(_remap), 416 + np.asarray(_remap),
                        [448], np.arange(384)])
  rng = np.random.default_rng(0)
  s449 = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
  s449[:, np.setdiff1d(np.arange(c_s_inputs), idx)] = 0.0
  _OURS['s_inputs'] = s449[:, idx]

  tf = batch.token_features
  ifd = {k: t(np.asarray(getattr(tf, k)).astype(np.int64), torch.long)[None]
         for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                   'sym_id')}
  rp = RelativePositionEncoding(c_z=c_z)
  rp.load_state_dict(
      {'linear_no_bias.weight': g('relative_position_encoding.linear_no_bias.weight')},
      strict=False)
  rp.eval()
  with torch.no_grad():
    s_init = lin('linear_no_bias_sinit.weight', t(s449))
    z = (lin('linear_no_bias_zinit1.weight', s_init)[..., None, :]
         + lin('linear_no_bias_zinit2.weight', s_init)[..., None, :, :])
    if not os.environ.get('NORELPE'):
      # `generate_relp` returns the feature DICT with a 'relp' entry added;
      # forward takes that tensor (protenix.py: `relative_position_encoding(
      # input_feature_dict["relp"])`).
      z = z + rp(rp.generate_relp(ifd)['relp'])[0]
    else:
      print('  NORELPE: relative position encoding dropped on both sides')
    z = z + lin('linear_no_bias_token_bond.weight', t(bonds)[..., None])
  return np.asarray(z)


def native_if2(model, batch, s_inputs, bonds, bond_types, n):
  """-> z_init from IntelliFold-2's own `InputEmbedder` (embedders.py:172-186).

      z = linear_z_i(s).unsqueeze(-2) + linear_z_j(s).unsqueeze(-3)
      z = z + relative_position_encoding(asym, residue, entity, token, sym)
      z = z + linear_token_bonds(token_bonds[..., None])

  A third spelling of the same axes: `unsqueeze(-2)` is the ROW, so i is the row
  and j the column -- of3's convention, rf3's mirrored. And if2's s_inputs is
  447 channels, OUR layout, so unlike every other model in this gate there is no
  restype permutation to undo.
  """
  import torch

  from intellifold.openfold.model.embedders import RelativePositionEncoding

  sd = torch.load(os.path.expanduser('~/model_v2/intellifold_v2.pt'),
                  map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'backbone_trunk.input_embedder.'
  g = lambda k: sd[pre + k]
  t = lambda x, d=torch.float32: torch.tensor(np.asarray(x), dtype=d)
  lin = lambda w, x: torch.nn.functional.linear(x, g(w))

  # ROUND THE NATIVE WEIGHTS TO THE BLOB'S STORAGE DTYPE. if2 is the only port
  # that stores its TRUNK in bfloat16, deliberately, mirroring AF3's own param
  # dtype policy -- `converters/intellifold2._record_dtype`, where it is also
  # measured fold-neutral (6MRR 1.517 against 1.519 for an fp32 blob). Its
  # checkpoint is entirely float32, so comparing our bf16-stored weights against
  # the raw checkpoint measures the STORAGE, not the port: 2.12e-02 on this
  # gate, reproduced exactly in numpy with the same formula and both weight
  # sets. The L4 confidence gate already rounds native for this reason.
  # NO_BF16_ROUND=1 shows the unrounded number.
  if not os.environ.get('NO_BF16_ROUND'):
    import ml_dtypes
    _raw = g
    def g(k, _raw=_raw):                       # noqa: E306
      v = _raw(k)
      return torch.tensor(
          np.asarray(v, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32))
    print('  native trunk weights rounded to bfloat16 (the blob\'s storage)')
  c_z, c_s_inputs = g('linear_z_i.weight').shape
  n_relpos = g('relative_position_encoding.linear_relpos.weight').shape[1]
  print('  checkpoint: c_z %d, c_s_inputs %d (OUR layout), relpos %d'
        % (c_z, c_s_inputs, n_relpos))
  # main's synthetic array is `evoformer.seq_channel` wide, which is right only
  # for boltz2; here the width is the 447-channel target_feat. Built at the
  # checkpoint's own width and handed back for our side to use.
  if s_inputs.shape[-1] != c_s_inputs:
    rng = np.random.default_rng(0)
    s_inputs = (rng.normal(size=(n, c_s_inputs)) * 0.5).astype(np.float32)
    _OURS['s_inputs'] = s_inputs
  si = t(s_inputs)
  tf = batch.token_features
  ids = [t(np.asarray(getattr(tf, k)).astype(np.int64), torch.long)[None]
         for k in ('asym_id', 'residue_index', 'entity_id', 'token_index',
                   'sym_id')]
  rp = RelativePositionEncoding(c_z=c_z, r_max=32, s_max=2)
  miss, _ = rp.load_state_dict(
      {'linear_relpos.weight': g('relative_position_encoding.linear_relpos.weight')},
      strict=False)
  assert not miss, list(miss)[:3]
  rp.eval()
  with torch.no_grad():
    z = (lin('linear_z_i.weight', si).unsqueeze(-2)
         + lin('linear_z_j.weight', si).unsqueeze(-3))
    if not os.environ.get('NORELPE'):
      z = z + rp(*ids, dtype=z.dtype)[0]
    else:
      print('  NORELPE: relative position encoding dropped on both sides')
    z = z + lin('linear_token_bonds.weight', t(bonds)[..., None])
  return np.asarray(z)


def native_esmfold2(model, batch, s_inputs, bonds, bond_types, n):
  """-> z_init from `esmfold2_reference`, the family's stand-in for a vendor.

  ESMFold2's own modules live in ~/venv_esm, so every other esmfold2 gate
  compares against `dev/oracles/esmfold2_reference.py` -- a self-contained JAX
  reimplementation whose fidelity `L1.trunk_ref` and `L3.denoise_ref` establish
  end to end. Its z-init is three lines (reference line 324-330):

      z_pair0 = (s @ z_init_1)[:, None] + (s @ z_init_2)[None, :]
      z       = z_pair0 + rel_pos_features(...) @ rel_pos
      z       = z + token_bonds @ token_bonds_w

  the same three terms our `_seq_pair_embedding` / `_relative_encoding` /
  `_embed_bonds` produce. Written out here rather than by running `R.trunk`,
  which would need the ESM-C hidden states and a native dump to reach the same
  point.

  Note the relative encoding: ESMFold2 puts same_entity BEFORE the chain block
  and INVERTS the chain bucket (same-chain goes to the out-of-bounds bin), which
  is the convention `model_config.CHAIN_BUCKET_ON_SAME_CHAIN` carries for this
  family -- so a mismatch here would be that flag, and the gate would say so.
  """
  import esmfold2_dumps
  import esmfold2_reference as R
  from converters import esmfold2 as CV

  sd = esmfold2_dumps.state_dict(model)
  p = {k: np.asarray(v) for k, v in CV.map_esmfold2_to_af3(sd).items()}
  w1 = p['z_init_1/weights']
  c_s_inputs, c_z = w1.shape
  print('  reference params: c_s_inputs %d, c_z %d' % (c_s_inputs, c_z))
  # TWO LAYOUTS OF THE SAME VECTOR. ESMFold2 concatenates
  # [atom 384 | restype 33 | profile 33 | deletion 1] = 451 and AF3
  # [restype 31 | profile 31 | deletion 1 | atom 384] = 447, and the restype
  # blocks are a PERMUTATION, not a slice -- ESM puts the MSA gap at class 1,
  # below the residues (`converters/esmfold2.esm_class_of_af3`). The converter's
  # `remap_s_inputs` reorders the WEIGHT rows for exactly this; here the same
  # permutation is applied to the FEATURE so both sides hold the same numbers in
  # their own order. Reading it as a slice put every consumer's atom block under
  # the restype weights and read z_init corr 0.008 when the converter got it
  # wrong, which is the recorded size of this mistake.
  n_af3, n_esm = 31, 33
  rng = np.random.default_rng(0)
  s447 = (rng.normal(size=(n, 2 * n_af3 + 1 + 384)) * 0.5).astype(np.float32)
  _OURS['s_inputs'] = s447
  cols = np.asarray(CV.esm_class_of_af3(n_af3))
  s_inputs = np.zeros((n, c_s_inputs), np.float32)
  s_inputs[:, :384] = s447[:, 2 * n_af3 + 1:]
  s_inputs[:, 384 + cols] = s447[:, :n_af3]
  s_inputs[:, 384 + n_esm + cols] = s447[:, n_af3:2 * n_af3]
  s_inputs[:, 384 + 2 * n_esm] = s447[:, 2 * n_af3]
  tf = batch.token_features
  ids = [np.asarray(getattr(tf, k)).astype(int)
         for k in ('residue_index', 'asym_id', 'sym_id', 'entity_id',
                   'token_index')]
  z = ((s_inputs @ w1)[:, None] + (s_inputs @ p['z_init_2/weights'])[None, :])
  if not os.environ.get('NORELPE'):
    z = z + np.asarray(R.rel_pos_features(*ids)) @ p['rel_pos/weights']
  else:
    print('  NORELPE: relative position encoding dropped on both sides')
  z = z + np.asarray(bonds)[..., None] @ p['token_bonds/weights']
  return np.asarray(z)


NATIVES = {'boltz2': native_boltz2, 'rosettafold3': native_rf3,
           'intellifold2': native_if2}
NATIVES.update({m: native_esmfold2 for m in (
    'esmfold2', 'esmfold2_fast', 'esmfold2_lm600m', 'esmfold2_lm300m')})
NATIVES.update({m: native_of3 for m in _OF3_CKPT})
NATIVES.update({m: native_protenix for m in _PROTENIX_SRC})
LOOPS = {'boltz2': native_boltz2_loop}


def ours(model, cfg, model_dir, batch, s_inputs):
  """Our z-init: _seq_pair_embedding -> _relative_encoding -> _embed_bonds."""
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import evoformer as evo

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)

  def fwd():
    ev = evo.Evoformer(cfg.evoformer, cfg.global_config, name='evoformer')
    single_act = None
    if cfg.global_config.model == 'opendde':
      # OpenDDE builds its pair init from the SINGLE EMBEDDING, not from
      # target_feat -- `_seq_pair_embedding`'s `single_act` argument, and the
      # reason its left/right projections are (384, 384) where everyone else's
      # are (447, 384). Building s_init here creates `single_activations` in
      # this transform, which the blob has.
      import haiku as _hk
      from alphafold3.model.components import haiku_modules as hm
      single_act = hm.Linear(cfg.evoformer.seq_channel,
                             name='single_activations')(jnp.asarray(s_inputs))
    # It returns (pair_activations, pair_mask) -- pair FIRST. Unpacking it the
    # other way round broadcasts a (n, n) mask against a (n, n, c) tensor and
    # says so, which is the good outcome.
    pair, _ = ev._seq_pair_embedding(  # pylint: disable=protected-access
        batch.token_features, jnp.asarray(s_inputs), single_act=single_act)
    pair = ev._relative_encoding(batch, pair)  # pylint: disable=protected-access
    return ev._embed_bonds(batch, pair)        # pylint: disable=protected-access

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      # `_seq_pair_embedding` and its neighbours are @hk.transparent, so their
      # parameters land at the TOP LEVEL of this transform ('left_single') where
      # the blob has them under the module that owns them
      # ('diffuser/evoformer/left_single'). The relative encoding keeps its own
      # sub-scope, which the converter writes as
      # `diffuser/evoformer/~_relative_encoding/position_activations`.
      if sc == '~':
        # A bare `hk.get_parameter` in a module-level helper called from a
        # @hk.transparent method lands in the OWNER's bundle, which for the
        # contact-conditioning constants is `diffuser/evoformer` itself.
        key = 'diffuser/evoformer'
      elif sc == 'evoformer' or sc.startswith('evoformer/'):
        key = 'diffuser/' + sc
      else:
        key = 'diffuser/evoformer/' + sc
      src = full.get(key)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v, np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:4]))
  assert not unmapped, 'our z-init is partly at init'
  if os.environ.get('NORELPE'):
    # Zero the relative-position projection on OUR side too. z-init is a sum of
    # independent terms, so dropping one from both sides says whether the
    # disagreement is in it or in what is left.
    for sc in params:
      if 'position_activations' in sc or 'relpe' in sc:
        params[sc] = {k: np.zeros_like(np.asarray(v))
                      for k, v in params[sc].items()}
  return np.asarray(f.apply(params, jax.random.PRNGKey(0)))


def ours_loop(model, cfg, model_dir, batch, s_inputs, passes):
  """-> (s, z) after `passes` calls of our own Evoformer, carrying prev.

  `Evoformer.__call__(batch, prev, target_feat, key)` IS the loop body, so the
  seam is the `prev` dict -- exactly what the model itself threads through its
  recycle scan. Running it pass by pass here (rather than through the scan) is
  what lets a divergence be attributed to a pass number.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import evoformer as evo

  cfg.global_config.bfloat16 = 'none'
  cfg.global_config.flash_attention_implementation = 'xla'
  full = afp.get_model_haiku_params(model_dir=model_dir)
  n = int(np.asarray(batch.token_features.mask).shape[0])
  c_s, c_z = cfg.evoformer.seq_channel, cfg.evoformer.pair_channel

  def fwd(prev):
    ev = evo.Evoformer(cfg.evoformer, cfg.global_config, name='evoformer')
    return ev(batch=batch, prev=prev, target_feat=jnp.asarray(s_inputs),
              key=jax.random.PRNGKey(0))

  f = hk.transform(fwd)
  prev0 = {'single': jnp.zeros((n, c_s), jnp.float32),
           'pair': jnp.zeros((n, n, c_z), jnp.float32)}
  init = f.init(jax.random.PRNGKey(0), prev0)
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      key = ('diffuser/' + sc if sc.startswith('evoformer')
             else 'diffuser/evoformer' if sc == '~'
             else 'diffuser/evoformer/' + sc)
      src = full.get(key)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v, np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:4]))
  assert not unmapped, 'our trunk is partly at init'
  prev = prev0
  for i in range(passes):
    out = f.apply(params, jax.random.PRNGKey(0), prev)
    prev = {'single': out['single'], 'pair': out['pair']}
    print('  ours pass %d: s rms %.4f, z rms %.4f'
          % (i + 1, float(jnp.sqrt(jnp.mean(out['single'] ** 2))),
             float(jnp.sqrt(jnp.mean(out['pair'] ** 2)))))
  return np.asarray(prev['single']), np.asarray(prev['pair'])


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--pdb', default=os.path.expanduser('~/6MRR.pdb'))
  ap.add_argument('--chains_json', default=None,
                  help='a fold-input JSON whose chains replace --pdb. THE ONLY '
                       'WAY TO REACH A CROSS-CHAIN CONVENTION: z-init is where '
                       'the relative-position encoding lives, and its '
                       'chain/entity/sym terms are constant on a monomer -- '
                       'both the boltz2 entity bucket and the esmfold2 chain '
                       'bucket were per-channel constants there. Every gate in '
                       'this directory otherwise runs one protein chain.')
  ap.add_argument('--model_dir', default=None)
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]

  if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
    raise SystemExit('set JAX_DEFAULT_MATMUL_PRECISION=highest')
  if args.model not in NATIVES:
    raise SystemExit('no native adapter for %r; have %s'
                     % (args.model, sorted(NATIVES)))

  import fold_check
  from alphafold3.model import feat_batch
  from alphafold3.model.network import evoformer as evo
  from alphafold3.model import model_config

  chains = None
  if args.chains_json:
    from alphafold3.common import folding_input
    chains = list(folding_input.Input.from_json(
        open(args.chains_json).read()).chains)
    seq = getattr(chains[0], 'sequence', '')
  else:
    seq, _ = fold_check.parse_ca(args.pdb)
  batch, cfg, model_dir = fold_check._fold_setup(args.model, seq,
                                                 args.model_dir, chains=chains)
  batch = feat_batch.Batch.from_data_dict(batch)
  n = int(np.asarray(batch.token_features.mask).shape[0])
  rng = np.random.default_rng(0)
  # s_inputs at the model's OWN width, and the SAME array to both sides.
  w = cfg.evoformer.seq_channel
  s_inputs = (rng.normal(size=(n, w)) * 0.5).astype(np.float32)
  sym = args.model in model_config.OPENFOLD3_LINEAGE
  bonds = np.asarray(evo.token_bond_matrix(batch, symmetrize=sym))
  bond_types = np.asarray(evo.token_bond_type_matrix(batch, symmetrize=sym))
  print('%s trunk z-init, %d tokens, s_inputs %d wide, %d bonded pairs:'
        % (args.model, n, w, int((bonds > 0).sum())))

  passes = int(os.environ.get('PASSES', 0))
  if passes:
    if args.model not in LOOPS:
      raise SystemExit('no native LOOP for %r; have %s'
                       % (args.model, sorted(LOOPS)))
    s_ref, z_ref = LOOPS[args.model](args.model, batch, s_inputs, bonds,
                                     bond_types, n, passes)
    s_got, z_got = ours_loop(args.model, cfg, model_dir, batch, s_inputs,
                             passes)
    print('  shapes: ours s %s z %s | native s %s z %s'
          % (s_got.shape, z_got.shape, s_ref.shape, z_ref.shape))
    _cmp('s_pass%d' % passes, s_got, s_ref)
    _cmp('z_pass%d' % passes, z_got, z_ref)
    return 0

  z_ref = NATIVES[args.model](args.model, batch, s_inputs, bonds, bond_types, n)
  # An adapter may REPLACE the synthetic s_inputs: the width and the alphabet
  # are per-model (boltz2 concatenates a token_s-wide one, the of3 lineage a
  # 449-channel one with its own restype permutation), and the two sides have to
  # see the same numbers in their own layouts.
  s_inputs = _OURS.get('s_inputs', s_inputs)
  z_got = ours(args.model, cfg, model_dir, batch, s_inputs)
  print('  shapes: ours %s native %s' % (z_got.shape, z_ref.shape))
  _cmp('z_init', z_got, z_ref)
  if os.environ.get('DIAG'):
    d = np.abs(np.asarray(z_got, np.float64) - np.asarray(z_ref, np.float64))
    print('  DIAG per-pair max|d|: median %.5f  p90 %.5f  max %.5f'
          % (float(np.median(d.max(-1))), float(np.quantile(d.max(-1), 0.9)),
             float(d.max())))
    off = d.mean(axis=(0, 1))
    print('  DIAG per-CHANNEL mean|d|: worst %s'
          % [(int(i), round(float(off[i]), 4))
             for i in np.argsort(-off)[:6]])
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
