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

# This directory on the path, so the shared comparison below imports whether the
# gate is run from the repo root or anywhere else -- the same line atom_parity.py
# carries, and without it `from confidence_parity import _cmp` depends on the
# caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


# THE SHARED COMPARISON, not a local one. This file used to print corr, the rms
# ratio and max|d| -- and no rms(native), so no max|d|/rms. `parity_audit.py`
# grades on corr AND that ratio, and falls back to corr alone when the ratio is
# absent: L1.trunk runs for more models than any other gate, and every one of
# its rows was being graded on the one number [[correlation-hides-bias]] says is
# blind to a per-channel constant. A 48-block stack makes that worse, not
# better, because the magnitudes grow with depth -- of3's z reads max|d| 75.6 at
# corr 1.000000, which says nothing without the scale beside it.
from confidence_parity import _cmp                      # noqa: E402


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


def native_if2(model, n, mask, blocks=None):
  """-> (s, z, s_ref, z_ref, n_blocks) from IntelliFold-2's own PairformerStack.

  if2's trunk is the WIDENED tree, and every width here is read off the
  checkpoint for that reason: c_z is 512 (not AF3's 128), the triangle
  multiplication is 512 wide, and the pair attention runs 8 heads of 64 against
  the single track's 16. Hardcoding AF3's numbers fails in load_state_dict --
  the good outcome -- but the same trap already cost this repo a session on if2's
  TEMPLATE stack, so the derivation is spelled out rather than assumed:

      c_s               attention_pair_bias.layer_norm          (384,)
      c_z               attention_pair_bias.layer_norm_z        (512,)
      no_heads_single   attention_pair_bias.linear_z            (16, 512)
      no_heads_pair     pair_stack.tri_att_start.linear         (8, 512)
      c_hidden_pair_att mha.linear_q // no_heads_pair           512/8 = 64
      c_hidden_mul      tri_mul_out.linear_ab_p // 2            1024/2 = 512
  """
  import torch

  from intellifold.openfold.model.pairformer import PairformerStack

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  raw = raw.get('model', raw.get('state_dict', raw))
  pre = 'backbone_trunk.pairformer.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  b0 = 'blocks.0.'
  c_s = sub[b0 + 'attention_pair_bias.layer_norm.weight'].shape[0]
  c_z = sub[b0 + 'attention_pair_bias.layer_norm_z.weight'].shape[0]
  nh_single = sub[b0 + 'attention_pair_bias.linear_z.weight'].shape[0]
  nh_pair = sub[b0 + 'pair_stack.tri_att_start.linear.weight'].shape[0]
  c_pair_att = sub[b0 + 'pair_stack.tri_att_start.mha.linear_q.weight'].shape[0] // nh_pair
  c_mul = sub[b0 + 'pair_stack.tri_mul_out.linear_ab_p.weight'].shape[0] // 2
  # the transition's expansion, off the single track's SwiGLU: linear is
  # (2 * n * c_s, c_s), so n = shape[0] // (2 * c_s)
  trans_n = sub[b0 + 'single_transition.linear.weight'].shape[0] // (2 * c_s)
  print('  checkpoint: %d blocks, c_s %d, c_z %d, %d single heads, '
        '%d pair heads x %d, c_hidden_mul %d, transition_n %d'
        % (n_blocks, c_s, c_z, nh_single, nh_pair, c_pair_att, c_mul, trans_n))

  rng = np.random.default_rng(0)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  net = PairformerStack(c_s=c_s, c_z=c_z, c_hidden_mul=c_mul,
                        c_hidden_pair_att=c_pair_att, no_heads_pair=nh_pair,
                        no_heads_single=nh_single, no_blocks=n_blocks,
                        transition_n=trans_n, pair_dropout=0.0, inf=1e9,
                        eps=1e-8)
  # ROUND NATIVE'S WEIGHTS THE WAY THE BLOB STORES THEM. if2 is the only port
  # whose blob stores the trunk in bfloat16 -- a deliberate, measured policy
  # that mirrors AF3's own (see converters/intellifold2.py `_record_dtype`).
  # The gate already turns bf16 off in both FORWARD passes, so without this it
  # loads bf16-rounded weights and compares them against native's fp32
  # checkpoint: the difference is the STORAGE dtype, not the port. Measured
  # 2026-09-10 -- the shipped blob reads s 1.05e-02 / z 6.98e-02 while an
  # fp32-converted blob (IF2_FP32_BLOB=1) reads s 4.44e-06 / z 3.58e-05, so
  # the entire reading was the rounding.
  #
  # The rule is exactly the converter's: LayerNorm scale/offset stay fp32,
  # every other pairformer tensor rounds. Verified by count against a loaded
  # blob -- 20 fp32 and 31 bf16 per block, which is the 10 layer norms' 20
  # tensors and the remaining 31 (`linear_q.bias` among them, because the
  # converter keys on the haiku names 'scale'/'offset', not on being a bias).
  if not os.environ.get('IF2_NO_BF16_WEIGHTS'):
    n_round = 0
    for k in list(sub):
      if '.layer_norm' in k:
        continue
      sub[k] = sub[k].to(torch.bfloat16).float()
      n_round += 1
    print('  native: %d of %d tensors bf16-rounded to match the blob '
          "(set IF2_NO_BF16_WEIGHTS=1 to measure the storage policy instead)"
          % (n_round, len(sub)))

  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  # TRUNCATE THE NATIVE STACK TOO -- see native_protenix.
  if blocks is not None and blocks < n_blocks:
    net.blocks = net.blocks[:blocks]
    n_blocks = blocks
  net.eval()
  with torch.no_grad():
    out = net(torch.tensor(s)[None], torch.tensor(z)[None],
              torch.tensor(mask.max(-1))[None].float(),
              torch.tensor(mask)[None], chunk_size=None)
  s_ref, z_ref = out[0], out[1]
  return s, z, s_ref[0].numpy(), z_ref[0].numpy(), n_blocks


def native_boltz2(model, n, mask, blocks=None):
  """-> (s, z, s_ref, z_ref, n_blocks) from Boltz-2's own PairformerModule.

  boltz's module takes (s, z, mask, pair_mask) rather than protenix's
  (s, z, pair_mask), and its blocks live under `layers.` not `blocks.`. Widths
  come off the checkpoint: token_s and token_z from the layer norms, the pair
  head count and width from the triangle attention, since boltz's defaults
  (16 heads, 32-wide, 4 pair heads) are not universal across its releases.
  """
  import torch

  # `boltz.model.modules.trunk` pulls fairscale, which is not in this venv and
  # must not be installed into it. `boltz.model.layers.pairformer` holds the
  # same PairformerModule without that import -- the same route msa_parity.py
  # takes to reach MSAModule through trunkv2.
  from boltz.model.layers.pairformer import PairformerModule

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  raw = raw.get('state_dict', raw)
  pre = 'pairformer_module.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('layers.'))
  token_s = sub['layers.0.pre_norm_s.weight'].shape[0]
  token_z = sub['layers.0.tri_mul_out.norm_in.weight'].shape[0]
  # The triangle attention's head count and width, off its own projections:
  # proj_q is (heads*width, token_z). boltz's defaults (4 heads x 32) are not
  # universal across its releases, so they are derived, and a wrong pair fails
  # in load_state_dict rather than comparing quietly.
  pnh = int(sub['layers.0.tri_att_start.linear.weight'].shape[0])
  pw = int(sub['layers.0.tri_att_start.mha.linear_q.weight'].shape[0])
  head_w = pw // pnh
  print('  checkpoint: %d blocks, token_s %d, token_z %d, tri proj %d '
        '(%d heads x %d)' % (n_blocks, token_s, token_z, pw, pnh, head_w))

  rng = np.random.default_rng(0)
  s = (rng.normal(size=(n, token_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, token_z)) * 0.5).astype(np.float32)

  # v2=True: boltz2's blocks carry `pre_norm_s`, the v2 AttentionPairBias. The
  # v1 default builds `norm_s` instead and reports 96 missing tensors -- two per
  # block, which is the tell that the class is right and the variant is not.
  net = PairformerModule(token_s=token_s, token_z=token_z, num_blocks=n_blocks,
                         dropout=0.0, pairwise_head_width=head_w,
                         pairwise_num_heads=pnh, v2=True)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  # TRUNCATE THE NATIVE STACK TOO -- see native_protenix.
  if blocks is not None and blocks < n_blocks:
    net.layers = net.layers[:blocks]
    n_blocks = blocks
  net.eval()
  with torch.no_grad():
    s_ref, z_ref = net(torch.tensor(s)[None], torch.tensor(z)[None],
                       torch.tensor(mask.max(-1))[None].float(),
                       torch.tensor(mask)[None])
  return s, z, s_ref[0].numpy(), z_ref[0].numpy(), n_blocks


def native_opendde(model, n, mask, blocks=None):
  """-> (s, z, s_ref, z_ref, n_blocks) from OpenDDE's own PairformerStack.

  opendde is protenix-lineage and its stack takes the same arguments, so this is
  native_protenix with a different checkpoint and prefix. `hidden_scale_up` is
  read off the checkpoint rather than assumed: `PairformerBlock` builds its
  triangle hidden width from c_z only when that flag is set, so a wrong value
  fails in load_state_dict -- which is the good outcome -- but a value that
  loads is not automatically the right one, and protenix2 needed True where AF3
  needs False.
  """
  import torch

  from opendde.model.modules.pairformer import PairformerStack

  ckpt = os.path.expanduser('~/opendde_weights/opendde.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('model', sd.get('state_dict', sd))
  pre = 'module.pairformer_stack.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[1])
                     for k in sub if k.startswith('blocks.'))
  c_z = sub['blocks.0.pair_transition.layernorm1.weight'].shape[0]
  heads = sub['blocks.0.attention_pair_bias.linear_nobias_z.weight'].shape[0]
  c_s = sub['blocks.0.attention_pair_bias.layernorm_a.weight'].shape[0]
  # the triangle hidden width says which convention this checkpoint was built
  # with: c_z when scaled up, 128 otherwise
  tri_hidden = sub['blocks.0.tri_mul_out.linear_z.weight'].shape[0]
  hsu = tri_hidden == c_z
  print('  checkpoint: %d blocks, c_z %d, c_s %d, %d heads, tri hidden %d '
        '(hidden_scale_up %s)' % (n_blocks, c_z, c_s, heads, tri_hidden, hsu))

  rng = np.random.default_rng(0)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(np.float32)

  net = PairformerStack(n_blocks=n_blocks, n_heads=heads, c_z=c_z, c_s=c_s,
                        hidden_scale_up=hsu)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  # TRUNCATE THE NATIVE STACK TOO -- see native_protenix.
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


def native_rf3(model, n, mask, blocks=None):
  """-> (s, z, s_ref, z_ref, n_blocks) from RosettaFold3's own pairformer.

  rf3 was the single worst-covered model in the panel -- four of the twelve
  disagreeing cells plus no trunk adapter at all -- and the trunk was the
  biggest missing piece.

  Three rf3 facts shape this, and each of the first two is the same switch the
  diffusion adapter already had to throw:

    * `force_bfloat16 = True` on every AttentionPairBiasPairformerDeepspeed.
      Left on, the comparison measures bf16 rounding, not the port.
    * `use_cuequivariance=True` is hardcoded in `PairformerBlock.__init__` for
      both triangle multiplications and both triangle attentions. It is
      already inert on this card (`attention.SHOULD_USE_CUEQUIVARIANCE` is
      False), but "inert today" is not a property of the port, so it is turned
      off explicitly rather than relied upon.
    * THERE IS NO MASK. rf3's pairformer block takes only (S_I, Z_II) -- no
      pair mask, no seq mask -- so this adapter ignores the `mask` argument.
      That is sound only because the gate feeds an all-ones mask; a partial
      mask would compare a masked stack against an unmasked one and the number
      would be meaningless. Asserted below rather than left to a reader.

  There is also no `PairformerStack` class: rf3 builds an `nn.ModuleList` of 48
  blocks inline in `RF3_structure.py`, so the stack is assembled here the same
  way and every width is read off the checkpoint.
  """
  import numpy as _np
  import torch

  from rf3.model.layers import attention as _att
  from rf3.model.layers.pairformer_layers import PairformerBlock

  assert _np.all(_np.asarray(mask) == 1), (
      'rf3 pairformer blocks take no mask; this adapter is only valid for the '
      'all-ones mask the gate feeds')

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
  # `shadow.*` is the EMA copy, which is what converters/rosettafold3.py ports.
  pre = 'shadow.recycler.pairformer_stack.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys in %s' % (pre, ckpt))

  n_blocks = 1 + max(int(k.split('.')[0]) for k in sub)
  c_s = sub['0.attention_pair_bias.ln_1.weight'].shape[0]
  c_z = sub['0.tri_mul_outgoing.norm_in.weight'].shape[0]
  # `p_in` projects to BOTH triangle branches at once, so the hidden width is
  # half of it -- 128 here, which happens to equal c_z; reading it off the
  # tensor keeps that coincidence from being load-bearing.
  d_hidden = sub['0.tri_mul_outgoing.p_in.weight'].shape[0] // 2
  ta_heads = sub['0.tri_attn_start.to_b.weight'].shape[0]
  ta_hidden = sub['0.tri_attn_start.to_q.weight'].shape[0] // ta_heads
  apb_heads = sub['0.attention_pair_bias.to_b.weight'].shape[0]
  n_transition = sub['0.z_transition.linear_1.weight'].shape[0] // c_z
  print('  checkpoint: %d blocks, c_s %d, c_z %d, tri hidden %d, tri_attn '
        '%dx%d, apb %d heads, n_transition %d, cuEq available %s'
        % (n_blocks, c_s, c_z, d_hidden, ta_heads, ta_hidden, apb_heads,
           n_transition, getattr(_att, 'SHOULD_USE_CUEQUIVARIANCE', None)))

  keep = n_blocks if blocks is None else min(blocks, n_blocks)
  net = torch.nn.ModuleList([
      PairformerBlock(c_s=c_s, c_z=c_z, p_drop=0.25,
                      triangle_multiplication=dict(d_hidden=d_hidden),
                      triangle_attention=dict(n_head=ta_heads,
                                              d_hidden=ta_hidden),
                      attention_pair_bias=dict(n_head=apb_heads),
                      n_transition=n_transition)
      for _ in range(keep)])
  sub = {k: v for k, v in sub.items() if int(k.split('.')[0]) < keep}
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected'
        % (len(sub), len(missing), len(unexpected)))
  assert not missing, 'native is missing %d tensors' % len(missing)
  for b in net:
    b.attention_pair_bias.force_bfloat16 = False
    for m in (b.tri_mul_outgoing, b.tri_mul_incoming,
              b.tri_attn_start, b.tri_attn_end):
      m.use_cuequivariance = False
  net.eval()

  rng = _np.random.default_rng(0)
  s = (rng.normal(size=(n, c_s)) * 0.5).astype(_np.float32)
  z = (rng.normal(size=(n, n, c_z)) * 0.5).astype(_np.float32)
  with torch.no_grad():
    S, Z = torch.tensor(s)[None], torch.tensor(z)[None]
    for b in net:
      S, Z = b(S, Z)
  return s, z, S[0].numpy(), Z[0].numpy(), keep


NATIVES = {m: native_protenix for m in _PROTENIX_CKPT}
NATIVES.update({m: native_of3 for m in _OF3_CKPT})
NATIVES['opendde'] = native_opendde
NATIVES['boltz2'] = native_boltz2
NATIVES['intellifold2'] = native_if2
NATIVES['rosettafold3'] = native_rf3


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
                       'little until ONE block is measured. AND THE FULL DEPTH '
                       'MEANS LITTLE EITHER, on synthetic input: rosettafold3 '
                       'reads max|d|/rms 5.8e-04 on z at 1 block, 5.0e-04 at 4 '
                       '-- flat, so not compounding -- and 1.2e-01 at 48, where '
                       'the single track has grown to rms 2.7e4 and the pair '
                       'track has FALLEN to 24. A stack driven that far outside '
                       'its trained input distribution is an amplifier being '
                       'measured, not a port. Read 1-4 blocks for the port and '
                       'the full depth only as a smoke test.')
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
