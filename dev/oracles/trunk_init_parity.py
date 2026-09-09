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
NATIVES = {'boltz2': native_boltz2}


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
