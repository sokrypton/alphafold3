"""L2 parity: our token diffusion transformer against the vendor's own module.

The trunk gate (dev/oracles/trunk_parity.py) covers the pairformer stack and
nothing else. Nine `model_config` tables feed the DIFFUSION path alone, and for
most models not one of them has an activation-level check -- which is the
largest unexamined exposure in PARITY.md. This is the first of those levels.

  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:<vendor> \
    python dev/oracles/diffusion_parity.py <model>

Vendor overlay per model (they can be concatenated):

  protenix family   /home/ubuntu/protenix
  openfold3         /home/ubuntu/openfold-3
  openbind0         /home/ubuntu/openfold-3
  intellifold2      /home/ubuntu/IntelliFold
  rosettafold3      /home/ubuntu/rf3_extra:/home/ubuntu/foundry_rf3/src:\
                    /home/ubuntu/foundry_rf3/models/rf3/src

Same discipline as the trunk gate: dims come from the CHECKPOINT, native and
ours must both report zero missing/unmapped, bfloat16 is forced off BEFORE the
config is read, and the script refuses to run without the tf32 override.
"""
import argparse
import os
import sys
import types

import numpy as np

_PROTENIX_CKPT = {
    'protenix2': 'protenix-v2.pt',
    'protenix1': 'protenix_base_default_v1.0.0.pt',
}
_OF3_CKPT = {'openfold3': 'of3-p2-155k.pt', 'openbind0': 'of3-ob-174k.pt'}
_OURS_PREFIX = 'diffuser/~/diffusion_head/transformer/'


def _cmp(tag, got, ref):
  a = np.asarray(got, np.float64).ravel()
  b = np.asarray(ref, np.float64).ravel()
  # max|d| ALONE is unreadable: it tracks the reference's own scale and the
  # stack depth, not fidelity. Print rms(native) and the ratio next to it.
  rb = np.sqrt((b ** 2).mean())
  print('  %-4s corr %.6f  rms ours/native %.4f  max|d| %.5f  '
        'rms(native) %.3f  max|d|/rms %.2e'
        % (tag, np.corrcoef(a, b)[0, 1], np.sqrt((a ** 2).mean()) / rb,
           np.abs(a - b).max(), rb, np.abs(a - b).max() / rb))


def _stub_layer_norm():
  """protenix's fused LayerNorm kernel, in torch, dispatched by ARITY.

  The diffusion path calls MORE entry points than the trunk did -- it wants
  `forward_none_affine` as well as the three the pairformer used -- so
  enumerating them means discovering each by AttributeError. Dispatching on how
  many tensors arrive covers all of them. Plain LayerNorm is the right stand-in:
  the kernel is a speed optimisation of the same function, and a wrong one shows
  up immediately as a mismatch rather than passing quietly.
  """
  import torch

  def _ln(x, shape, w=None, b=None, eps=1e-5):
    dims = tuple(range(x.dim() - len(shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + eps)
    out = (x - mean) * inv
    if w is not None:
      out = out * w
    if b is not None:
      out = out + b
    return out, mean.squeeze(-1), inv.squeeze(-1)

  class _Ext(types.ModuleType):

    def __getattr__(self, name):
      if not name.startswith('forward'):
        raise AttributeError(name)

      def fn(x, shape, *rest):
        eps = rest[-1] if rest and isinstance(rest[-1], float) else 1e-5
        tens = [r for r in rest if torch.is_tensor(r)]
        return _ln(x, shape, tens[0] if tens else None,
                   tens[1] if len(tens) > 1 else None, eps)
      return fn

  sys.modules.setdefault('fast_layer_norm_cuda_v2',
                         _Ext('fast_layer_norm_cuda_v2'))


def native_protenix(model, n):
  """-> (a, s, z, ref, n_blocks) from protenix's own DiffusionTransformer."""
  import torch

  _stub_layer_norm()
  from protenix.model.modules.transformer import DiffusionTransformer

  ckpt = os.path.expanduser('~/protenix_weights/' + _PROTENIX_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd)
  pre = 'module.diffusion_module.diffusion_transformer.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  # Key paths PRINTED off the checkpoint, not guessed -- four successive
  # guesses at these were wrong while writing this.
  ab = 'blocks.0.attention_pair_bias.'
  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  c_a = sub[ab + 'attention.linear_q.weight'].shape[0]
  c_s = sub[ab + 'layernorm_a.layernorm_s.weight'].shape[0]
  c_z = sub[ab + 'layernorm_z.weight'].shape[0]
  heads = sub[ab + 'linear_nobias_z.weight'].shape[0]
  print('  checkpoint: %d blocks, c_a %d, c_s %d, c_z %d, %d heads'
        % (n_blocks, c_a, c_s, c_z, heads))

  rng = np.random.default_rng(0)
  a = (rng.normal(size=(n, c_a)) * 0.5).astype(np.float32)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  net = DiffusionTransformer(c_a=c_a, c_s=c_s, c_z=c_z, n_blocks=n_blocks,
                             n_heads=heads)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    ref = net(torch.tensor(a)[None], torch.tensor(s)[None],
              torch.tensor(z)[None])
  return a, s, z, np.asarray(ref[0]), n_blocks


def native_of3(model, n):
  """-> (a, s, z, ref, n_blocks) from OpenFold3's own DiffusionTransformer.

  Covers `openfold3` and `openbind0`: both run the `~/openfold-3`
  implementation, openbind0 on v0.5.0 weights. No LayerNorm stub is needed here
  -- of3's LayerNorm is plain torch, unlike protenix's fused CUDA kernel.
  """
  import torch

  from openfold3.core.model.layers.diffusion_transformer import (
      DiffusionTransformer)

  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.diffusion_transformer.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  # Key paths PRINTED off the checkpoint. of3 names the AdaLN sub-scope
  # `layer_norm_a.layer_norm_s` and the attention `mha.`, where protenix uses
  # `layernorm_a.layernorm_s` and `attention.` -- close enough to guess wrong.
  ab = 'blocks.0.attention_pair_bias.'
  ct = 'blocks.0.conditioned_transition.'
  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  c_a = sub[ab + 'mha.linear_q.weight'].shape[0]
  c_s = sub[ab + 'layer_norm_a.layer_norm_s.weight'].shape[0]
  heads, c_z = sub[ab + 'linear_z.weight'].shape
  # WHICH RELEASE: preview-2 LayerNorms the pair conditioning inside every
  # block (24 x `blocks.N.attention_pair_bias.layer_norm_z.weight`); v0.5.0
  # moved it out and runs it once for the stack (a single top-level
  # `layer_norm_z.weight`) to match the AF3 SI. That is exactly the split
  # `model_config.PER_BLOCK_PAIR_LAYER_NORM` encodes, and openbind0 is the
  # v0.5.0 side of it -- so the CHECKPOINT decides, nothing is remembered here.
  # `~/openfold-3` only implements the per-block form, so for v0.5.0 the
  # per-block LNs are replaced by Identity and the single LN is applied to z
  # up front, leaving our side to see the same raw z it sees in the real graph.
  single_ln = 'layer_norm_z.weight' in sub
  # c_hidden is PER HEAD (config comments it `c_token / no_heads`), and
  # n_transition is the swiglu widening. Both derived, so a release that
  # changed either fails in load_state_dict rather than comparing quietly.
  c_hidden = c_a // heads
  n_transition = sub[ct + 'swiglu.linear_a.weight'].shape[0] // c_a
  print('  checkpoint: %d blocks, c_a %d, c_s %d, c_z %d, %d heads, '
        'c_hidden %d, n_transition %d'
        % (n_blocks, c_a, c_s, c_z, heads, c_hidden, n_transition))

  rng = np.random.default_rng(0)
  a = (rng.normal(size=(n, c_a)) * 0.5).astype(np.float32)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  net = DiffusionTransformer(c_a=c_a, c_s=c_s, c_z=c_z, c_hidden=c_hidden,
                             no_heads=heads, no_blocks=n_blocks,
                             n_transition=n_transition,
                             use_ada_layer_norm=True, n_query=None,
                             n_key=None, inf=1e9, blocks_per_ckpt=None)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s  (%s pair LN)'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           'single' if single_ln else 'per-block'))
  z_in = torch.tensor(z)[None]
  if single_ln:
    expect = {'blocks.%d.attention_pair_bias.layer_norm_z.weight' % i
              for i in range(n_blocks)}
    assert set(missing) == expect, 'unexpected missing: %s' % sorted(
        set(missing) - expect)[:3]
    for blk in net.blocks:
      blk.attention_pair_bias.layer_norm_z = torch.nn.Identity()
    z_in = torch.nn.functional.layer_norm(
        z_in, (c_z,), weight=sub['layer_norm_z.weight'], bias=None, eps=1e-5)
  else:
    assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    ref = net(torch.tensor(a)[None], torch.tensor(s)[None], z_in,
              mask=torch.ones(1, n))
  return a, s, z, np.asarray(ref[0]), n_blocks


def native_if2(model, n):
  """-> (a, s, z, ref, n_blocks) from IntelliFold-2's own stack.

  if2's `DiffusionTransformerStack` LayerNorms the pair conditioning ONCE for
  the whole stack (`self.layer_norm_z`, bias=False) -- the AF3 convention, and
  why if2 is absent from `model_config.PER_BLOCK_PAIR_LAYER_NORM`. So unlike
  the of3 v0.5.0 case no surgery is needed: raw z goes into both sides.
  """
  import torch

  from intellifold.openfold.model.diffusion import DiffusionTransformerStack

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'diffusion_module.diffusion_transformer.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  # Printed off the checkpoint. if2 names the AdaLN `adaptive_layer_norm` with
  # separate `linear_s_gamma`/`linear_s_beta`, where of3 has `layer_norm_a` and
  # protenix `layernorm_a` -- three vendors, three spellings of one module.
  ab = 'blocks.0.attention_pair_bias.'
  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  c_a = sub[ab + 'mha.linear_q.weight'].shape[0]
  c_s = sub[ab + 'adaptive_layer_norm.layer_norm_s.weight'].shape[0]
  heads, c_z = sub[ab + 'linear_z.weight'].shape
  # swiglu: linear_a widens to 2 * transition_n * c_a, linear_b takes half.
  n_trans = sub['blocks.0.single_transition.linear_a.weight'].shape[0] // (
      2 * c_a)
  print('  checkpoint: %d blocks, c_a %d, c_s %d, c_z %d, %d heads, '
        'transition_n %d' % (n_blocks, c_a, c_s, c_z, heads, n_trans))

  rng = np.random.default_rng(0)
  a = (rng.normal(size=(n, c_a)) * 0.5).astype(np.float32)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  net = DiffusionTransformerStack(no_blocks=n_blocks, no_heads=heads, c_a=c_a,
                                  c_s=c_s, c_z=c_z, transition_n=n_trans)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  with torch.no_grad():
    ref = net(torch.tensor(a)[None], torch.tensor(s)[None],
              torch.tensor(z)[None], mask=torch.ones(1, n), chunk_size=None)
  return a, s, z, np.asarray(ref[0]), n_blocks


def native_rf3(model, n):
  """-> (a, s, z, ref, n_blocks) from RosettaFold3's own DiffusionTransformer.

  PARITY.md called rf3 "blocked: no native installed". It is not: foundry
  imports as a PYTHONPATH overlay with its deps in `~/rf3_extra`, the trick the
  rf3 featurisation oracles already use.

  Two rf3-specific things this gate has to handle:
    * `force_bfloat16 = True` on every AttentionPairBiasDiffusion -- native
      casts the token activation to bfloat16 inside attention regardless of
      input dtype. Left on, the comparison measures bf16 rounding (~1e-2
      relative), not the port. Turned OFF here; it is a speed switch, not a
      convention, and our own graph runs this path in fp32 under
      `bfloat16='none'`.
    * `Beta_II` must be None. Passing a bias routes the module into its
      windowed `atom_attention` branch instead of token attention.
  """
  import torch

  from rf3.model.layers.af3_diffusion_transformer import DiffusionTransformer

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
  # `shadow.*` is the EMA copy, which is what converters/rosettafold3.py ports.
  pre = 'shadow.diffusion_module.diffusion_transformer.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  ab = 'blocks.0.attention_pair_bias.'
  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  c_a = sub[ab + 'to_q.weight'].shape[0]
  c_s = sub[ab + 'ada_ln_1.ln_s.weight'].shape[0]
  heads, c_z = sub[ab + 'to_b.weight'].shape
  kq_norm = ab + 'key_layer_norm.weight' in sub
  print('  checkpoint: %d blocks, c_a %d, c_s %d, c_z %d, %d heads, '
        'kq_norm %s' % (n_blocks, c_a, c_s, c_z, heads, kq_norm))

  rng = np.random.default_rng(0)
  a = (rng.normal(size=(n, c_a)) * 0.5).astype(np.float32)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  # `no_residual_connection_between_attention_and_transition` is NOT derivable
  # from the weights -- it is rf3_net.yaml `true`, and our graph implements it
  # at diffusion_transformer.py (act + attn + transition(pre-attention act)).
  net = DiffusionTransformer(
      c_token=c_a, c_s=c_s, c_tokenpair=c_z, n_block=n_blocks,
      diffusion_transformer_block=dict(
          n_head=heads, kq_norm=kq_norm,
          no_residual_connection_between_attention_and_transition=True))
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  for blk in net.blocks:
    blk.attention_pair_bias.force_bfloat16 = False
  net.eval()
  with torch.no_grad():
    ref = net(torch.tensor(a)[None], torch.tensor(s)[None],
              torch.tensor(z)[None], None)
  return a, s, z, np.asarray(ref[0]), n_blocks


NATIVES = {m: native_protenix for m in _PROTENIX_CKPT}
NATIVES['rosettafold3'] = native_rf3
NATIVES['intellifold2'] = native_if2
NATIVES.update({m: native_of3 for m in _OF3_CKPT})


def ours(model, a, s, z, model_dir=None):
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import model as af3_model, model_registry
  from alphafold3.model import params as afp
  from alphafold3.model.network import diffusion_transformer

  cfg = af3_model.Model.Config()
  cfg.global_config.bfloat16 = 'none'          # before anything reads it
  cfg.global_config.flash_attention_implementation = 'xla'
  model_registry.get(model).configure(cfg)
  assert cfg.global_config.bfloat16 == 'none', 'a spec re-enabled bfloat16'

  full = afp.get_model_haiku_params(
      model_dir=model_dir or os.path.expanduser('~/ported/%s' % model))
  tcfg = cfg.heads.diffusion.transformer
  n = a.shape[0]
  print('  ours: %d blocks x super %d, %s heads'
        % (tcfg.num_blocks, tcfg.super_block_size,
           getattr(tcfg.attention, 'num_head', '?')))

  def fwd(a_, s_, z_):
    return diffusion_transformer.Transformer(tcfg, cfg.global_config)(
        act=a_, mask=jnp.ones(n, jnp.float32), single_cond=s_, pair_cond=z_)

  f = hk.transform(fwd)
  init = f.init(jax.random.PRNGKey(0), jnp.asarray(a), jnp.asarray(s),
                jnp.asarray(z))
  params, unmapped = {}, []
  for sc in init:
    params[sc] = {}
    for k in init[sc]:
      # our scope is '<stacks>/<name>'; the blob's is the same tail under
      # _OURS_PREFIX, and the leading (super, block) axes already match because
      # both sides build the same nested layer_stack.
      tail = sc.split('/')[-1]
      src = None
      for cand in full:
        if cand.startswith(_OURS_PREFIX) and cand.split('/')[-1] == tail:
          src = full[cand]
          break
      v = None if src is None else src.get(k)
      if v is None:
        unmapped.append('%s/%s' % (sc, k))
        params[sc][k] = init[sc][k]
      else:
        params[sc][k] = np.asarray(v)
  print('  ours: %d scopes, %d unmapped %s'
        % (len(init), len(unmapped), unmapped[:3]))
  assert not unmapped, 'our stack is partly at init -- comparison would be noise'
  return f.apply(params, jax.random.PRNGKey(0), jnp.asarray(a),
                 jnp.asarray(s), jnp.asarray(z))


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--tokens', type=int, default=68)
  ap.add_argument('--model_dir', default=None)
  args = ap.parse_args(argv)
  sys.argv = sys.argv[:1]          # absl parses argv lazily; see fold_check

  if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
    raise SystemExit('set JAX_DEFAULT_MATMUL_PRECISION=highest; tf32 is ~5e-4 '
                     'per matmul and compounds over the stack')
  if args.model not in NATIVES:
    raise SystemExit('no native adapter for %r; have %s'
                     % (args.model, sorted(NATIVES)))

  print('%s token diffusion transformer, %d tokens:' % (args.model, args.tokens))
  a, s, z, ref, _nb = NATIVES[args.model](args.model, args.tokens)
  out = ours(args.model, a, s, z, args.model_dir)
  _cmp('a', out, ref)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
