"""Gates the DISTOGRAM head, which nothing had ever compared.

It appeared in confidence_parity.py only as a feature NAME
(`distogram_rep_atom_mask`), never as a compared tensor -- so a divergence here
was invisible to every number in PARITY.md. It matters out of proportion to its
size: the distogram is the head design gradients flow through (zero recycles,
backprop into the distogram and stop), so a silent error would corrupt the
design path while every structural RMSD stayed green.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:/home/ubuntu/protenix \
    python dev/oracles/dgram_parity.py protenix2

The head is one projection off the trunk pair, so the input is synthetic z --
no real batch needed, unlike the atom gates. What CAN go wrong here and does:
whether the vendor symmetrises before or after the projection, whether it
trains a bias (model_config.DISTOGRAM_BIAS), and whether the converter has to
halve that bias because native symmetrises first.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_parity import _cmp                         # noqa: E402


def native_protenix(model, z, n_bins):
  import torch

  from diffusion_parity import _stub_layer_norm
  _stub_layer_norm()
  from protenix.model.modules.head import DistogramHead

  from denoise_parity import _PROTENIX_CKPT
  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.distogram_head.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))
  c_z = sub['linear.weight'].shape[1]
  bins = sub['linear.weight'].shape[0]
  print('  checkpoint: c_z %d, %d bins, bias %s'
        % (c_z, bins, 'linear.bias' in sub))
  net = DistogramHead(c_z=c_z, no_bins=bins)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(np.asarray(z), dtype=torch.float32)[None])
  return np.asarray(out)[0], bins


def native_of3(model, z, n_bins):
  import torch

  from openfold3.core.model.heads.prediction_heads import DistogramHead

  from denoise_parity import _OF3_CKPT
  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  for pre in ('distogram_head.', 'aux_heads.distogram.', 'heads.distogram.'):
    sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    if sub:
      break
  if not sub:
    cands = sorted({k.split('.')[0] for k in sd if 'disto' in k.lower()})
    raise SystemExit('no distogram keys; candidates %s' % cands)
  print('  prefix %r, %d tensors: %s' % (pre, len(sub), sorted(sub)[:4]))
  w = [v for k, v in sub.items() if k.endswith('weight')][0]
  bins, c_z = w.shape[0], w.shape[1]
  print('  checkpoint: c_z %d, %d bins, bias %s'
        % (c_z, bins, any(k.endswith('bias') for k in sub)))
  # of3 names the bin count `c_out`, not `no_bins`, and its linear-init params
  # come from its own config -- passing the default is what the model does.
  net = DistogramHead(c_z=c_z, c_out=bins)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(np.asarray(z), dtype=torch.float32)[None])
  out = out['logits'] if isinstance(out, dict) else out
  return np.asarray(out)[0], bins


def native_if2(model, z, n_bins):
  import torch

  from intellifold.openfold.model.heads import DistogramHead

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  cands = sorted({k for k in sd if 'distogram' in k.lower()})
  pre = None
  for c in cands:
    if c.endswith('weight'):
      pre = c[:-len('linear.weight')] if c.endswith('linear.weight') else None
      break
  if pre is None:
    raise SystemExit('no distogram linear; candidates %s' % cands[:6])
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  bins, c_z = sub['linear.weight'].shape
  print('  prefix %r, c_z %d, %d bins, bias %s'
        % (pre, c_z, bins, any(k.endswith('bias') for k in sub)))
  net = DistogramHead(c_z=c_z, no_bins=bins)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(np.asarray(z), dtype=torch.float32)[None])
  return np.asarray(out)[0], bins


def native_rf3(model, z, n_bins):
  """rf3 symmetrises BEFORE the linear, which is what its halved bias is for.

  `predictor(Z + Z.transpose(-2, -3))` gives W(z+z^T) + b, where our graph --
  and protenix, and of3 -- compute `Linear(z) + Linear(z)^T` = W(z+z^T) + 2b.
  Identical for the weight, off by a factor of two on the BIAS, which is why
  converters/rosettafold3.py stores b/2. This gate is what checks that claim
  rather than trusting the comment.
  """
  import torch

  from rf3.model.RF3_structure import DistogramHead

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  raw = raw.get('state_dict', raw.get('model', raw))
  pre = 'shadow.distogram_head.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  bins, c_z = sub['predictor.weight'].shape
  print('  c_z %d, %d bins, bias %s (symmetrises BEFORE the linear)'
        % (c_z, bins, 'predictor.bias' in sub))
  net = DistogramHead(c_z=c_z, bins=bins)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(np.asarray(z), dtype=torch.float32)[None])
  return np.asarray(out)[0], bins


def native_boltz2(model, z, n_bins):
  """boltz symmetrises BEFORE the linear, so its bias is halved on conversion.

  `z = z + z.transpose(1, 2); self.distogram(z)` (trunkv2.py) gives W(z+z^T)+b
  where our graph computes W(z+z^T)+2b -- the same split rf3 is on, and
  `converters/boltz2.py` already halves it. This is what checks that.
  """
  import torch

  from boltz.model.modules.trunkv2 import DistogramModule

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd)
  pre = 'distogram_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  out_dim, token_z = sub['distogram.weight'].shape
  bins = n_bins
  n_dgram = out_dim // bins
  print('  checkpoint: token_z %d, %d bins x %d distogram(s), bias %s'
        % (token_z, bins, n_dgram, 'distogram.bias' in sub))
  net = DistogramModule(token_z=token_z, num_bins=bins,
                        num_distograms=n_dgram)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(np.asarray(z), dtype=torch.float32)[None])
  out = np.asarray(out)
  # (b, n, n, n_dgram, bins) -> our (n, n, bins); boltz2 carries one distogram.
  return out.reshape(z.shape[0], z.shape[1], -1)[..., :bins], bins


def native_opendde(model, z, n_bins):
  """opendde symmetrises AFTER the linear, so its bias is NOT halved."""
  import torch

  from opendde.model.modules.head import DistogramHead

  ckpt = os.path.expanduser('~/opendde_weights/opendde.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'module.distogram_head.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  bins, c_z = sub['linear.weight'].shape
  print('  checkpoint: c_z %d, %d bins, bias %s'
        % (c_z, bins, 'linear.bias' in sub))
  net = DistogramHead(c_z=c_z, no_bins=bins)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(np.asarray(z), dtype=torch.float32)[None])
  return np.asarray(out)[0], bins


NATIVES = {}
try:
  from denoise_parity import _PROTENIX_CKPT, _OF3_CKPT
  NATIVES.update({m: native_protenix for m in _PROTENIX_CKPT})
  NATIVES.update({m: native_of3 for m in _OF3_CKPT})
  NATIVES['intellifold2'] = native_if2
  NATIVES['rosettafold3'] = native_rf3
  NATIVES['boltz2'] = native_boltz2
  NATIVES['opendde'] = native_opendde
except ImportError:
  pass


def ours(model, cfg, model_dir, fb, z):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import distogram_head

  cfg.global_config.bfloat16 = 'none'
  full = afp.get_model_haiku_params(model_dir=model_dir)

  def fwd():
    return distogram_head.DistogramHead(
        cfg.heads.distogram, cfg.global_config)(
            batch=fb, embeddings={'pair': jnp.asarray(z)},
            return_distogram=True)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      # The blob keys carry the `diffuser/` prefix that calling the head
      # directly does not; same fixup denoise_parity.py needs.
      src = full.get('diffuser/' + sc, full.get(sc))
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v, np.float32)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our head is partly at init'
  return f.apply(params, jax.random.PRNGKey(0))['distogram']


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
    raise SystemExit('no native adapter for %r (have %s)'
                     % (args.model, sorted(NATIVES)))

  import fold_check
  from alphafold3.model import feat_batch

  seq, _ = fold_check.parse_ca(args.pdb)
  batch, cfg, model_dir = fold_check._fold_setup(args.model, seq,
                                                 args.model_dir)
  fb = feat_batch.Batch.from_data_dict(batch)
  n_tok = np.asarray(fb.token_features.mask).shape[0]
  c_z = cfg.evoformer.pair_channel
  print('%s distogram head, %d tokens, c_z %d:' % (args.model, n_tok, c_z))

  rng = np.random.default_rng(0)
  z = (rng.normal(size=(n_tok, n_tok, c_z)) * 0.5).astype(np.float32)
  ref, bins = NATIVES[args.model](args.model, z, cfg.heads.distogram.num_bins)
  got = np.asarray(ours(args.model, cfg, model_dir, fb, z))
  print('  shapes: ours %s native %s' % (got.shape, ref.shape))
  if got.shape != ref.shape:
    print('  SHAPE MISMATCH -- our num_bins %d vs native %d'
          % (got.shape[-1], ref.shape[-1]))
    return 1
  _cmp('distogram', got, ref)
  # Symmetry is the thing most likely to differ and least likely to show up in
  # a correlation: both sides symmetrise, but before or after the projection.
  for tag, a in (('ours', got), ('native', ref)):
    asym = np.abs(a - np.swapaxes(a, -2, -3)).max()
    print('  %-6s max asymmetry %.3e' % (tag, asym))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
