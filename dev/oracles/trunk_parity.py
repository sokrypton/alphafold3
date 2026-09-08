"""L1 trunk parity: our single/pair against the vendor's own torch module.

Lives HERE, in the library, deliberately. The equivalent harness in ColabDesign2
is bound to that repo's vendored 8-model registry, so it cannot gate the other
14 -- `MODEL=protenix1` there dies with "unknown weights 'protenix1'". The
models moved into this package and the gate has to follow them.

  PYTHONPATH=src:. python dev/oracles/trunk_parity.py protenix2

Both harness confounds must be off or the number is meaningless -- they cost six
false leads on protenix2, whose trunk read 0.9929/0.9376 with a port that turned
out exact. This script forces the first (bfloat16='none' before the config is
read) and asserts the second, so a caller who forgets the env var is told.

    JAX_DEFAULT_MATMUL_PRECISION=highest \
      PYTHONPATH=src:.:/home/ubuntu/protenix \
      python dev/oracles/trunk_parity.py protenix2

NATIVE ADAPTERS. One vendor implementation usually serves a whole family, which
is the leverage here: `~/protenix` covers all six protenix models and
`~/openfold-3` covers openfold3 + openbind0. Dims come from the CHECKPOINT, not
from constants -- the family shares an implementation but not its widths
(protenix2 is c_z 256, protenix1 c_z 128), and
hardcoding one model's shape is what left five with no gate.
"""
import argparse
import os
import sys

import numpy as np

# A scope that exists exactly once inside the trunk pairformer, used to FIND
# the blob's prefix rather than hardcode haiku's layer_stack numbering -- that
# numbering differs between models, and guessing it silently leaves parameters
# at random init, which reads as a port bug. Borrowed from
# tools/oracles/openfold3/cmp_trunk_parity.py, where the reasoning is recorded.
_SUFFIX = 'trunk_pairformer/pair_attention1/act_norm'

_PROTENIX_CKPT = {
    'protenix2': 'protenix-v2.pt',
    'protenix1': 'protenix_base_default_v1.0.0.pt',
}


def _cmp(tag, got, ref):
  a = np.asarray(got, np.float64).ravel()
  b = np.asarray(ref, np.float64).ravel()
  print('  %-4s corr %.6f  rms ours/native %.4f  max|d| %.5f'
        % (tag, np.corrcoef(a, b)[0, 1],
           np.sqrt((a ** 2).mean()) / np.sqrt((b ** 2).mean()),
           np.abs(a - b).max()))


def _stub_protenix_ext():
  """Stand in for protenix's fused LayerNorm CUDA kernel.

  Its layer_norm module JIT-COMPILES that kernel at import time with no flag to
  skip it, and this venv has CPU torch and no CUDA_HOME (and no ninja). An EMPTY
  stub is not enough -- the stack really does call the fused path -- so this
  gives a torch implementation of the same contract, returning
  (output, mean, invvar). That is plain LayerNorm; the kernel is a speed
  optimisation, not a different function, and getting it wrong would show up
  immediately as a mismatch against our own LayerNorm rather than passing
  quietly. Ported from tools/oracles/protenix2/cmp_trunk_parity.py, where it was
  worked out.
  """
  import types

  import torch as _t

  ext = types.ModuleType('fast_layer_norm_cuda_v2')

  def _ln(x, shape, w=None, b=None, eps=1e-5):
    dims = tuple(range(x.dim() - len(shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    inv = _t.rsqrt(var + eps)
    out = (x - mean) * inv
    if w is not None:
      out = out * w
    if b is not None:
      out = out + b
    return out, mean.squeeze(-1), inv.squeeze(-1)

  ext.forward_with_both_affine = lambda x, shp, w, b, eps: _ln(x, shp, w, b, eps)
  ext.forward_with_weight_affine = lambda x, shp, w, eps: _ln(x, shp, w, None, eps)
  ext.forward = lambda x, shp, eps: _ln(x, shp, None, None, eps)
  sys.modules.setdefault('fast_layer_norm_cuda_v2', ext)


def native_protenix(model, n, mask, blocks=None):
  """-> (s_ref, z_ref, dims). Covers all six protenix models."""
  import torch

  _stub_protenix_ext()
  from protenix.model.modules.pairformer import PairformerStack

  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.pairformer_stack.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  c_z = sub['blocks.0.pair_transition.layernorm1.weight'].shape[0]
  heads = sub['blocks.0.attention_pair_bias.linear_nobias_z.weight'].shape[0]
  c_s = 384
  print('  checkpoint: %d blocks, c_z %d, %d heads' % (n_blocks, c_z, heads))

  rng = np.random.default_rng(0)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  net = PairformerStack(n_blocks=n_blocks, n_heads=heads, c_z=c_z, c_s=c_s,
                        hidden_scale_up=True)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  # Truncate the NATIVE stack to the requested depth. Dropping this when porting
  # the harness made `--blocks 1` compare our 1-block output against native's
  # full 48 -- which read corr -0.019 and looked like a catastrophic port bug in
  # a small protenix release rather than a harness comparing two functions.
  if blocks is not None and blocks < n_blocks:
    net.blocks = net.blocks[:blocks]
    if hasattr(net, 'n_blocks'):
      net.n_blocks = blocks
    n_blocks = blocks
  net.eval()
  with torch.no_grad():
    s_ref, z_ref = net(torch.tensor(s)[None], torch.tensor(z)[None],
                       torch.tensor(mask)[None])
  return s, z, s_ref[0].numpy(), z_ref[0].numpy(), n_blocks


_OF3_CKPT = {'openfold3': 'of3-p2-155k.pt', 'openbind0': 'of3-ob-174k.pt'}


def native_of3(model, n, mask, blocks=None):
  """-> (s, z, s_ref, z_ref, n_blocks). Covers openfold3 and openbind0.

  Both releases run the same `~/openfold-3` implementation; openbind0 is v0.5.0
  weights over it. Block count and c_z come from the checkpoint, and every other
  kwarg is asserted by load_state_dict reporting zero missing -- so a release
  that widened something fails loudly here instead of comparing mismatched
  tensors.
  """
  import torch

  from openfold3.core.model.latent.pairformer import PairFormerStack

  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'pairformer_stack.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  # Key paths read off the checkpoint, not guessed: of3 nests the transition
  # under `pair_stack.`, and `linear_out` (c_z, 4*c_z) is what fixes
  # transition_n=4.
  c_z = sub['blocks.0.pair_stack.pair_transition.layer_norm.weight'].shape[0]
  c_s = sub['blocks.0.attn_pair_bias.layer_norm_a.weight'].shape[0]
  heads_bias = sub['blocks.0.attn_pair_bias.linear_z.weight'].shape[0]
  print('  checkpoint: %d blocks, c_z %d, c_s %d, %d pair-bias heads'
        % (n_blocks, c_z, c_s, heads_bias))

  rng = np.random.default_rng(0)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  net = PairFormerStack(c_s=c_s, c_z=c_z, c_hidden_pair_bias=24,
                        no_heads_pair_bias=heads_bias, c_hidden_mul=c_z,
                        c_hidden_pair_att=32, no_heads_pair=4,
                        no_blocks=n_blocks, transition_type='swiglu',
                        transition_n=4, pair_dropout=0.0,
                        fuse_projection_weights=False, blocks_per_ckpt=None,
                        inf=1e9)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native stack is partly at random init'
  # Truncate the NATIVE stack to the requested depth. Dropping this when porting
  # the harness made `--blocks 1` compare our 1-block output against native's
  # full 48 -- which read corr -0.019 and looked like a catastrophic port bug in
  # a small protenix release rather than a harness comparing two functions.
  if blocks is not None and blocks < n_blocks:
    net.blocks = net.blocks[:blocks]
    if hasattr(net, 'n_blocks'):
      net.n_blocks = blocks
    n_blocks = blocks
  net.eval()
  with torch.no_grad():
    s_ref, z_ref = net(torch.tensor(s)[None], torch.tensor(z)[None],
                       torch.ones(1, n), torch.tensor(mask)[None])
  return s, z, s_ref[0].numpy(), z_ref[0].numpy(), n_blocks


NATIVES = {m: native_protenix for m in _PROTENIX_CKPT}
NATIVES.update({m: native_of3 for m in _OF3_CKPT})


def ours(model, s, z, mask, n_blocks, model_dir=None):
  """Our trunk pairformer stack, weights from the converted blob."""
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import model as af3_model, model_registry
  from alphafold3.model import params as afp
  from alphafold3.model.network import modules

  cfg = af3_model.Model.Config()
  # BEFORE anything reads it: parameter dtypes are fixed by eval_shape, so
  # flipping this later is silently ineffective -- the mistake that left the
  # precision hypothesis untested for five leads.
  cfg.global_config.bfloat16 = 'none'
  cfg.global_config.flash_attention_implementation = 'xla'
  model_registry.get(model).configure(cfg)
  assert cfg.global_config.bfloat16 == 'none', 'a spec re-enabled bfloat16'

  full = afp.get_model_haiku_params(
      model_dir=model_dir or os.path.expanduser('~/ported/%s' % model))
  pf, n = cfg.evoformer.pairformer, s.shape[0]

  def fwd(s_, z_):
    def blk(x):
      z2, s2 = modules.PairFormerIteration(
          pf, cfg.global_config, with_single=True, name='trunk_pairformer'
      )(act=x[1], single_act=x[0], pair_mask=jnp.asarray(mask),
        seq_mask=jnp.ones(n, jnp.float32))
      return (s2, z2)
    return hk.experimental.layer_stack(n_blocks)(blk)((s_, z_))

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0), jnp.asarray(s), jnp.asarray(z))
  pres = [k[:-len(_SUFFIX)] for k in full
          if k.endswith(_SUFFIX) and k.startswith('diffuser/evoformer/')]
  assert len(pres) == 1, 'trunk scope not unique: %s' % pres
  pre = pres[0]
  print('  trunk scope: %r' % pre)

  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      # our local scope is '__layer_stack_.../trunk_pairformer/...'; the blob's
      # is `pre` + 'trunk_pairformer/...' with its own stack index.
      tail = (sc[sc.index('trunk_pairformer/'):]
              if 'trunk_pairformer/' in sc else None)
      src = None if tail is None else full.get(pre + tail)
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v)[:n_blocks]
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:2]))
  assert not unmapped, 'our stack has %d unmapped params' % len(unmapped)
  return f.apply(params, jax.random.PRNGKey(0),
                 jnp.asarray(s), jnp.asarray(z))


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--tokens', type=int, default=68)
  ap.add_argument('--blocks', type=int, default=None,
                  help='compare only the first N blocks; 48 of them compound '
                       'any per-block difference, so a low stack number means '
                       'little until ONE block is measured')
  ap.add_argument('--model_dir', default=None)
  args = ap.parse_args(argv)
  # Clear argv before anything imports tokamax: it parses sys.argv LAZILY
  # through absl.flags, which raises UnrecognizedFlagError on any flag it does
  # not own -- so `--blocks 1` died several frames from here, after the native
  # side had already run. fold_check.py does the same for the same reason.
  sys.argv = sys.argv[:1]

  if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
    raise SystemExit(
        'set JAX_DEFAULT_MATMUL_PRECISION=highest. XLA uses tf32 for float32 '
        'matmuls on this hardware (~5e-4 each) and it compounds over the '
        'stack; without it the number does not mean what it looks like.')
  if args.model not in NATIVES:
    raise SystemExit('no native adapter for %r; have %s'
                     % (args.model, sorted(NATIVES)))

  n = args.tokens
  mask = np.ones((n, n), np.float32)
  print('%s, %d tokens, identical synthetic input:' % (args.model, n))
  s, z, s_ref, z_ref, nb = NATIVES[args.model](
      args.model, n, mask, blocks=args.blocks)
  s_out, z_out = ours(args.model, s, z, mask, nb, args.model_dir)
  _cmp('s', s_out, s_ref)
  _cmp('z', z_out, z_ref)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
