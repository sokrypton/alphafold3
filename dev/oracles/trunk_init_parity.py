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
  rp = RelativePositionEncoder(token_z=token_z)
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


NATIVES = {'boltz2': native_boltz2}
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
    # It returns (pair_activations, pair_mask) -- pair FIRST. Unpacking it the
    # other way round broadcasts a (n, n) mask against a (n, n, c) tensor and
    # says so, which is the good outcome.
    pair, _ = ev._seq_pair_embedding(  # pylint: disable=protected-access
        batch.token_features, jnp.asarray(s_inputs))
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

  seq, _ = fold_check.parse_ca(args.pdb)
  batch, cfg, model_dir = fold_check._fold_setup(args.model, seq,
                                                 args.model_dir)
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
