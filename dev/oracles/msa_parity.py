"""L1b for models OTHER than protenix: the MSA module vs the vendor's own.

`prot_parity.py` gates the MSA module for protenix only, which is what closes
`CLAMPED_OPM_NORM` and `NO_MSA_ROW_UPDATE` -- two of the four trunk conventions
L1 explicitly does NOT cover. Ten models carry an MSA stack; this adds the rest,
starting with rosettafold3.

  JAX_DEFAULT_MATMUL_PRECISION=highest \
    PYTHONPATH=src:.:/home/ubuntu/rf3_extra:/home/ubuntu/foundry_rf3/src:/home/ubuntu/foundry_rf3/models/rf3/src \
    python dev/oracles/msa_parity.py rosettafold3

The MSA module updates BOTH representations -- msa -> pair through the outer
product, pair -> msa through pair-weighted averaging -- so the gate compares the
PAIR output, which is the half that survives into the trunk. Comparing only the
msa rows would miss a wrong outer-product normalisation entirely.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_parity import _cmp                          # noqa: E402


def native_rf3(model, msa, s_inputs, z, n_msa):
  import torch

  from rf3.model.layers.pairformer_layers import MSAModule

  ckpt = os.path.expanduser(
      '~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt')
  raw = torch.load(ckpt, map_location='cpu', weights_only=False)
  raw = raw.get('state_dict', raw.get('model', raw))
  pre = 'shadow.recycler.msa_module.'
  sub = {k[len(pre):]: v for k, v in raw.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  c_m, c_msa_feat = sub['msa_subsampler.emb_msa.weight'].shape
  c_s_inputs = sub['msa_subsampler.emb_S_inputs.weight'].shape[1]
  c_z = sub['outer_product.proj_out.weight'].shape[0]
  c_hidden_opm = sub['outer_product.proj_left.weight'].shape[0]
  # rf3's MSA keys carry NO block index -- its forward loops n_block times over
  # ONE set of submodules, so the checkpoint holds a single copy and the depth
  # cannot be derived from key names. 4 is rf3's own config
  # (rf3_net.yaml msa_module n_block). `converters/rosettafold3.py` already
  # mirrors the sharing: `_rosettafold3_msa_block` ignores its block index and
  # `_stack_blocks` replicates that one block across our 4 layers.
  n_block = int(os.environ.get('RF3_MSA_BLOCKS', 4))
  print('  checkpoint: c_m %d, c_msa_feat %d, c_s_inputs %d, c_z %d, '
        'opm hidden %d, %d blocks'
        % (c_m, c_msa_feat, c_s_inputs, c_z, c_hidden_opm, n_block))
  # Sub-config names and the parameter-free shapes are rf3's own
  # (configs/model/components/rf3_net.yaml `msa_module`); every WIDTH that
  # carries parameters is derived from the checkpoint above, so a release that
  # changed one fails in load_state_dict rather than comparing quietly.
  net = MSAModule(
      n_block=n_block, c_m=c_m, p_drop_msa=0.0, p_drop_pair=0.0,
      msa_subsample_embedder=dict(num_sequences=1024,
                                  dim_raw_msa=c_msa_feat,
                                  c_s_inputs=c_s_inputs, c_msa_embed=c_m),
      outer_product=dict(c_msa_embed=c_m, c_outer_product=c_hidden_opm,
                         c_out=c_z),
      msa_pair_weighted_averaging=dict(n_heads=8, c_weighted_average=32,
                                       c_msa_embed=c_m, c_z=c_z,
                                       separate_gate_for_every_channel=True),
      msa_transition=dict(n=4, c=c_m),
      triangle_multiplication_outgoing=dict(d_pair=c_z, d_hidden=128,
                                            bias=True),
      triangle_multiplication_incoming=dict(d_pair=c_z, d_hidden=128,
                                            bias=True),
      triangle_attention_starting=dict(d_pair=c_z, n_head=4, d_hidden=32,
                                       p_drop=0.0),
      triangle_attention_ending=dict(d_pair=c_z, n_head=4, d_hidden=32,
                                     p_drop=0.0),
      pair_transition=dict(n=4, c=c_z))
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  for m in net.modules():
    if getattr(m, 'force_bfloat16', False):
      m.force_bfloat16 = False
  net.eval()
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
  with torch.no_grad():
    # forward(f, Z_II, S_inputs_I) -- pair BEFORE s_inputs. Reversing them
    # feeds the 128-wide pair into a 449-wide projection and dies in the matmul
    # rather than comparing quietly, which is the good outcome.
    # The EMBEDDED msa is computed with native's own subsampler and handed
    # back, so OUR side starts from the identical activation. Feeding ours a
    # random c_m activation instead reads corr 0.468 -- not a finding, just two
    # different inputs. The embedding itself (emb_msa + emb_S_inputs) is left
    # to native because our equivalent lives in the input embedder, which
    # real_trunk_parity.py already gates end to end.
    msa_si = net.msa_subsampler(t(msa), t(s_inputs))
    out = net({'msa': t(msa)}, t(z), t(s_inputs))
  z_out = out[1] if isinstance(out, (tuple, list)) else out
  return (np.asarray(z_out).reshape(z.shape),
          np.asarray(msa_si.detach()))


def native_of3(model, msa, s_inputs, z, n_msa):
  """-> (pair out, embedded msa) from OpenFold3's own MSAModuleStack.

  of3 splits the module in two: `MSAModuleEmbedder` turns the raw rows into the
  c_m activation, and `MSAModuleStack` is the stack proper (AF3 Algorithm 8
  lines 5-15). This gate drives the STACK, and takes the embedded activation
  from of3's own embedder weights so both sides enter on the same tensor.
  """
  import copy

  import torch

  from openfold3.core.model.latent.msa_module import MSAModuleStack
  from openfold3.projects.of3_all_atom.config.model_config import model_config

  from denoise_parity import _OF3_CKPT
  ckpt = os.path.expanduser('~/' + _OF3_CKPT[model])
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'msa_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  emb = {k[len('msa_module_embedder.'):]: v for k, v in sd.items()
         if k.startswith('msa_module_embedder.')}
  if not sub or not emb:
    raise SystemExit('no msa_module / msa_module_embedder keys')

  def _find(cfg):
    if hasattr(cfg, 'keys'):
      if 'no_blocks' in cfg and 'c_hidden_opm' in cfg:
        return cfg
      for k in cfg:
        got = _find(cfg[k]) if hasattr(cfg[k], 'keys') else None
        if got is not None:
          return got
    return None

  mcfg = _find(copy.deepcopy(model_config))
  if mcfg is None:
    raise SystemExit('no msa_module subtree in of3 model_config')
  n_blocks = 1 + max(int(k.split('.')[1]) for k in sub
                     if k.startswith('blocks.'))
  c_m = emb['linear_m.weight'].shape[0]
  print('  checkpoint: c_m %d, %d blocks (config says %d)'
        % (c_m, n_blocks, mcfg['no_blocks']))
  args = {k: v for k, v in mcfg.items() if k != 'linear_init_params'}
  args['no_blocks'] = n_blocks
  args['c_m'] = c_m
  args['msa_dropout'] = 0.0
  args['pair_dropout'] = 0.0
  net = MSAModuleStack(**args)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
  # of3's embedder: linear_m(raw msa) + linear_s_input(s_inputs) broadcast.
  m_emb = (torch.nn.functional.linear(t(msa), emb['linear_m.weight'].float())
           + torch.nn.functional.linear(
               t(s_inputs), emb['linear_s_input.weight'].float())[None])
  n_tok = z.shape[0]
  with torch.no_grad():
    out = net(m=m_emb[None], z=t(z)[None],
              msa_mask=torch.ones(1, n_msa, n_tok),
              pair_mask=torch.ones(1, n_tok, n_tok))
  z_out = out[1] if isinstance(out, (tuple, list)) else out
  return (np.asarray(z_out).reshape(z.shape),
          np.asarray(m_emb.detach()))


def native_if2(model, msa, s_inputs, z, n_msa):
  """-> (pair out, embedded msa) from IntelliFold-2's own MSAModuleStack.

  Like of3, if2 splits embedding from the stack: `linear_no_bias_m` on the raw
  rows plus `linear_no_bias_s` on s_inputs. Widths come off the checkpoint; the
  parameter-free shapes are if2's own config (`msa_stack`), and its trunk is the
  widened full_fat tree so AF3's defaults would not fit.
  """
  import torch

  from intellifold.openfold.model.pairformer import MSAModuleStack

  ckpt = os.path.expanduser('~/model_v2/intellifold_v2.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  cands = sorted({k.split('msa_stack.')[0] + 'msa_stack.' for k in sd
                  if 'msa_stack.blocks.' in k})
  if not cands:
    raise SystemExit('no msa_stack.blocks in the checkpoint')
  pre = cands[0]
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  epre = pre[:pre.index('msa_stack.')]
  emb = {k[len(epre):]: v for k, v in sd.items()
         if k.startswith(epre) and 'msa_stack.' not in k}
  ek = {k for k in emb if k.endswith('.weight')}
  print('  prefix %r; embedder keys %s' % (pre, sorted(ek)[:4]))
  n_blocks = 1 + max(int(k.split('.')[1]) for k in sub
                     if k.startswith('blocks.'))
  # widths off the checkpoint
  b0 = 'blocks.0.'
  c_z = sub[b0 + 'pair_stack.tri_att_start.linear.weight'].shape[1] \
      if b0 + 'pair_stack.tri_att_start.linear.weight' in sub else z.shape[-1]
  # c_m from the MSA attention's own LayerNorm (256 for if2's widened tree, not
  # AF3's 64), and the msa head count from linear_z (heads, c_z).
  c_m = sub[b0 + 'msa_pair_weighted_averaging.layer_norm_m.weight'].shape[0]
  heads_msa = sub[b0 + 'msa_pair_weighted_averaging.linear_z.weight'].shape[0]
  c_msa_att = sub[b0 + 'msa_pair_weighted_averaging.mha.linear_v.weight'
                  ].shape[0] // heads_msa
  heads_pair = sub[b0 + 'pair_stack.tri_att_start.linear.weight'].shape[0] \
      if b0 + 'pair_stack.tri_att_start.linear.weight' in sub else 4
  c_att = sub[b0 + 'pair_stack.tri_att_start.mha.linear_q.weight'
              ].shape[0] // heads_pair \
      if b0 + 'pair_stack.tri_att_start.mha.linear_q.weight' in sub else 32
  c_mul = sub[b0 + 'pair_stack.tri_mul_out.linear_ab_p.weight'].shape[0] // 2 \
      if b0 + 'pair_stack.tri_mul_out.linear_ab_p.weight' in sub else 128
  print('  checkpoint: c_m %d, c_z %d, %d blocks, msa %d heads x %d, '
        'pair %d heads, c_hidden att %d / mul %d'
        % (c_m, c_z, n_blocks, heads_msa, c_msa_att, heads_pair, c_att, c_mul))
  net = MSAModuleStack(c_m=c_m, c_z=c_z, c_hidden_msa_att=c_msa_att,
                       c_hidden_opm=32, c_hidden_mul=c_mul,
                       c_hidden_pair_att=c_att, no_heads_msa=heads_msa,
                       no_heads_pair=heads_pair, no_blocks=n_blocks,
                       transition_n=4, msa_dropout=0.0, pair_dropout=0.0,
                       inf=1e9, eps=1e-10,
                       # v2_inference_config sets this True, and the checkpoint
                       # proves it: block 3 (the LAST) carries no
                       # msa_pair_weighted_averaging at all -- its msa output is
                       # never read, so if2 does not build those 12 tensors.
                       # Leaving it False asks for them and reports 12 missing.
                       skip_unused_modules=True)
  # ROUND NATIVE'S WEIGHTS THE WAY THE BLOB STORES THEM -- the same exposure
  # `trunk_parity` had, found the same day and for the same reason. if2 is the
  # only port whose blob keeps the trunk region in bfloat16 (a deliberate,
  # measured, fold-neutral policy: converters/intellifold2.py `_record_dtype`),
  # and this gate runs with bfloat16='none', so without this it compares
  # bf16-rounded weights against native's fp32 checkpoint and reports the
  # STORAGE dtype as a disagreement.
  #
  # Measured: the shipped blob reads max|d|/rms 1.05e-01 on this cell and an
  # fp32-converted one (IF2_FP32_BLOB=1) reads 1.06e-04, so the whole reading
  # was the rounding. LayerNorm scale/offset stay fp32 and every other tensor
  # rounds -- the converter's own rule. IF2_NO_BF16_WEIGHTS=1 measures the
  # storage policy instead.
  if not os.environ.get('IF2_NO_BF16_WEIGHTS'):
    n_round = 0
    for k in list(sub):
      if '.layer_norm' in k:
        continue
      sub[k] = sub[k].to(torch.bfloat16).float()
      n_round += 1
    print('  native: %d of %d tensors bf16-rounded to match the blob'
          % (n_round, len(sub)))
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  if missing:
    print('  MISSING: %s' % sorted(missing)[:6])
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
  # if2's embedder is `msa_embedder.linear_mf` (raw 34) + `linear_s_inputs`
  # (447 -- our 447 layout, NOT 449; if2 is not in PADDED_SINGLE_COND).
  wm = [v for k, v in emb.items() if k.endswith('msa_embedder.linear_mf.weight')]
  ws = [v for k, v in emb.items()
        if k.endswith('msa_embedder.linear_s_inputs.weight')]
  if not wm:
    raise SystemExit('no msa_embedder.linear_mf in %r' % epre)
  if ws and ws[0].shape[1] != s_inputs.shape[-1]:
    # trim/route the s_inputs layout to what THIS vendor's embedder expects
    from converters.openfold3 import _AF3_TO_OF3_AATYPE as _r
    idx = np.concatenate([384 + np.asarray(_r), 416 + np.asarray(_r), [448],
                          np.arange(384)])
    s_inputs = np.asarray(s_inputs)[:, idx][:, :ws[0].shape[1]]
  m_emb = torch.nn.functional.linear(t(msa), wm[0].float())
  if ws:
    m_emb = m_emb + torch.nn.functional.linear(t(s_inputs), ws[0].float())[None]
  n_tok = z.shape[0]
  with torch.no_grad():
    out = net(m=m_emb[None], z=t(z)[None],
              msa_mask=torch.ones(1, n_msa, n_tok),
              pair_mask=torch.ones(1, n_tok, n_tok), chunk_size=None)
  z_out = out[1] if isinstance(out, (tuple, list)) else out
  return (np.asarray(z_out).reshape(z.shape),
          np.asarray(m_emb.detach()))


def native_opendde(model, msa, s_inputs, z, n_msa):
  """-> (pair out, embedded msa) from OpenDDE's own MSAModule.

  opendde is protenix-lineage and keeps the embedder INSIDE the module
  (`_prepare_msa_sample`: one-hot the msa to 32 classes, concat has_deletion(1)
  and deletion_value(1) -> 34, `linear_no_bias_m`, plus
  `linear_no_bias_s(s_inputs)`). So this adapter hands it the raw integer msa
  and takes the embedded activation back out through the same call, keeping both
  sides on one tensor.

  `msa_configs` must define `msa_depth` or the constructor raises -- it is not
  a shape that shows up in the weights, so it comes from the harness.
  """
  import torch

  from opendde.model.modules.pairformer import MSAModule

  ckpt = os.path.expanduser('~/opendde_weights/opendde.pt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd.get('model', sd))
  pre = 'module.msa_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  c_m, n_feat = sub['linear_no_bias_m.weight'].shape
  c_s_inputs = sub['linear_no_bias_s.weight'].shape[1]
  n_blocks = 1 + max(int(k.split('.')[1]) for k in sub
                     if k.startswith('blocks.'))
  c_z = z.shape[-1]
  print('  checkpoint: c_m %d, %d raw features, c_s_inputs %d, c_z %d, '
        '%d blocks' % (c_m, n_feat, c_s_inputs, c_z, n_blocks))
  net = MSAModule(n_blocks=n_blocks, c_m=c_m, c_z=c_z,
                  c_s_inputs=c_s_inputs, blocks_per_ckpt=None,
                  msa_chunk_size=None, msa_configs={'msa_depth': n_msa},
                  hidden_scale_up=c_z > 128)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  t = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)
  n_tok = z.shape[0]
  # RAW integer msa (opendde one-hots it to 32 itself) plus the two deletion
  # channels it concatenates.
  rng = np.random.default_rng(3)
  msa_int = rng.integers(0, 31, size=(n_msa, n_tok))
  ifd = {'msa': t(msa_int, torch.long),
         'has_deletion': t(np.zeros((n_msa, n_tok), np.float32)),
         'deletion_value': t(np.zeros((n_msa, n_tok), np.float32))}
  s_in = np.zeros((n_tok, c_s_inputs), np.float32)
  s_in[:, :min(c_s_inputs, s_inputs.shape[-1])] = \
      np.asarray(s_inputs)[:, :min(c_s_inputs, s_inputs.shape[-1])]
  with torch.no_grad():
    m_emb = net._prepare_msa_sample(ifd, t(s_in), n_tok)
    out = net(ifd, t(z), t(s_in), torch.ones(n_tok, n_tok),
              triangle_multiplicative='torch', triangle_attention='torch')
  z_out = out[1] if isinstance(out, (tuple, list)) else out
  return (np.asarray(z_out).reshape(z.shape), np.asarray(m_emb.detach()))


def native_boltz2(model, msa, s_inputs, z, n_msa):
  """-> (pair out, embedded msa) from Boltz-2's own MSAModule.

  Its feature needs are modest -- `msa`, `has_deletion`, `deletion_value`,
  `msa_paired`, `msa_mask`, `token_pad_mask` -- NOT boltz's flat-atom layout,
  which is what made this look harder than it is. The embedder is inside the
  module (`msa_proj` on the concatenated features plus `s_proj(emb)`
  broadcast), so the embedded activation is reproduced here with those same two
  weights to keep both sides on one tensor.

  `use_paired_feature` decides whether `msa_paired` joins the concat, i.e.
  whether msa_proj is (raw+3) or (raw+2) wide -- read off the checkpoint rather
  than assumed.
  """
  import torch

  from boltz.model.modules.trunkv2 import MSAModule

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd)
  pre = 'msa_module.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  msa_s, in_dim = sub['msa_proj.weight'].shape
  token_s = sub['s_proj.weight'].shape[1]
  token_z = sub['layers.0.pair_weighted_averaging.proj_z.0.weight'].shape[0] \
      if 'layers.0.pair_weighted_averaging.proj_z.0.weight' in sub \
      else z.shape[-1]
  n_blocks = 1 + max(int(k.split('.')[1]) for k in sub
                     if k.startswith('layers.'))
  n_blocks = int(os.environ.get('MSA_BLOCKS', n_blocks))
  # in_dim = raw msa classes + has_deletion + deletion_value [+ is_paired]
  raw = msa.shape[-1]
  paired = (in_dim - raw) == 3
  print('  checkpoint: msa_s %d, msa_proj in %d (raw %d, paired %s), '
        'token_s %d, token_z %d, %d blocks'
        % (msa_s, in_dim, raw, paired, token_s, z.shape[-1], n_blocks))
  if in_dim - raw not in (2, 3):
    raise SystemExit('msa_proj wants %d columns and the raw msa is %d wide -- '
                     'neither +2 nor +3, so the feature set is not what this '
                     'adapter builds' % (in_dim, raw))
  net = MSAModule(msa_s=msa_s, token_z=z.shape[-1], token_s=token_s,
                  msa_blocks=n_blocks, msa_dropout=0.0, z_dropout=0.0,
                  use_paired_feature=paired, subsample_msa=False)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native: %d tensors, %d missing, %d unexpected %s %s'
        % (len(sub), len(missing), len(unexpected), list(missing)[:2],
           list(unexpected)[:2]))
  assert not missing, 'native is missing %d tensors' % len(missing)
  net.eval()
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
  n_tok = z.shape[0]
  zeros = np.zeros((n_msa, n_tok), np.float32)
  feats = {'msa': t(msa)[None], 'has_deletion': t(zeros)[None],
           'deletion_value': t(zeros)[None], 'msa_paired': t(zeros)[None],
           'msa_mask': torch.ones(1, n_msa, n_tok),
           'token_pad_mask': torch.ones(1, n_tok)}
  emb = np.zeros((n_tok, token_s), np.float32)
  emb[:, :min(token_s, s_inputs.shape[-1])] = \
      np.asarray(s_inputs)[:, :min(token_s, s_inputs.shape[-1])]
  cols = [t(msa)[None], t(zeros)[None][..., None], t(zeros)[None][..., None]]
  if paired:
    cols.append(t(zeros)[None][..., None])
  with torch.no_grad():
    m_cat = torch.cat(cols, dim=-1)
    m_emb = net.msa_proj(m_cat) + net.s_proj(t(emb)[None]).unsqueeze(1)
    out = net(t(z)[None], t(emb)[None], feats)
  z_out = out[1] if isinstance(out, (tuple, list)) else out
  return (np.asarray(z_out).reshape(z.shape),
          np.asarray(m_emb.detach())[0])



def path_of_dump(model):
  """The npz `native_esmfold2` would read, for callers that need it directly."""
  tag = model + ('_nonuniform' if os.environ.get('NONUNIFORM') else '')
  if os.environ.get('ESM_FINAL_UPDATE'):
    tag += '_finalupd'
  if os.environ.get('MSA_BLOCKS'):
    tag += '_b%d' % int(os.environ['MSA_BLOCKS'])
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'esmfold2_msa_%s.npz' % tag)


def native_esmfold2(model, msa, s_inputs, z, n_msa):
  """ESMFold2's MSAEncoder, read from an npz -- see esmfold2_msa_dump.py.

  The only adapter here that does NOT run the vendor in-process: ESMFold2's
  implementation ships inside `transformers`, which is installed in ~/venv_esm
  only. The dump writes its INPUTS as well as its output, and this adapter
  returns those inputs so both sides compare on the identical tensors rather
  than on two independently seeded draws.
  """
  tag = model + ('_nonuniform' if os.environ.get('NONUNIFORM') else '')
  if os.environ.get('ESM_FINAL_UPDATE'):
    tag += '_finalupd'
  # MSA_BLOCKS truncates OUR stacked leaves (see `ours`), so it has to select a
  # dump whose native stack was truncated to the same depth.
  if os.environ.get('MSA_BLOCKS'):
    tag += '_b%d' % int(os.environ['MSA_BLOCKS'])
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'esmfold2_msa_%s.npz' % tag)
  if not os.path.exists(path):
    raise SystemExit('run first:  ~/venv_esm/bin/python '
                     'dev/oracles/esmfold2_msa_dump.py %s%s%s%s'
                     % (model,
                        ' --nonuniform' if os.environ.get('NONUNIFORM') else '',
                        ' --keep_final_update'
                        if os.environ.get('ESM_FINAL_UPDATE') else '',
                        '' if not os.environ.get('MSA_BLOCKS')
                        else ' --blocks %s' % os.environ['MSA_BLOCKS']))
  d = np.load(path)
  print('  native npz: %s, %s class, %d blocks, final-block msa update %s'
        % (os.path.basename(path),
           'EXPERIMENTAL' if int(d.get('experimental', 0)) else 'released',
           int(d['n_layers']),
           'ON' if int(d['final_update']) else 'OFF (native default)'))
  # The harness's own z is discarded in favour of the dump's, so the two sides
  # cannot drift apart through two different default_rng streams.
  return d['pair_out'], d['msa_emb'], d['z'], d['mask']


NATIVES = {'rosettafold3': native_rf3, 'intellifold2': native_if2,
           'opendde': native_opendde, 'boltz2': native_boltz2,
           # ('esmfold2',) -- WITH THE COMMA. Without it this iterates the
           # STRING and registers eight single-character keys ('e', 's', 'm',
           # ...), so `esmfold2` had no adapter and the gate reported a HOLE for
           # a module that has been implemented all along.
           **{m: native_esmfold2 for m in ('esmfold2',)}}
try:
  from denoise_parity import _OF3_CKPT as _OF3
  NATIVES.update({m: native_of3 for m in _OF3})
except ImportError:
  pass


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('model')
  ap.add_argument('--pdb', default=os.path.expanduser('~/6MRR.pdb'))
  ap.add_argument('--model_dir', default=None)
  ap.add_argument('--num_msa', type=int, default=8)
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
  rng = np.random.default_rng(0)
  z = (rng.normal(size=(n_tok, n_tok, c_z)) * 0.5).astype(np.float32)
  print('%s MSA module (PAIR output), %d tokens, %d msa rows, c_z %d:'
        % (args.model, n_tok, args.num_msa, c_z))
  # The raw MSA feature width comes off the checkpoint inside the adapter; the
  # harness hands it random rows, as the trunk gates do for s/z.
  raw_w = {'rosettafold3': 35, 'intellifold2': 34, 'boltz2': 33,
           'opendde': 34, 'esmfold2': 35}.get(args.model, 34)
  msa = (rng.normal(size=(args.num_msa, n_tok, raw_w)) * 0.5).astype(np.float32)
  s_inputs = (rng.normal(size=(n_tok, 449)) * 0.5).astype(np.float32)
  if os.environ.get('LAYER'):
    return boltz2_layer_split(cfg, model_dir, args.num_msa, n_tok)
  if os.environ.get('COMPARE') == 'opm':
    return esmfold2_opm_split(args.model, cfg, model_dir, args.num_msa, n_tok)
  out = NATIVES[args.model](args.model, msa, s_inputs, z, args.num_msa)
  # The esmfold2 adapter also hands back the z it was actually run on (it comes
  # from a dump, not from this harness's rng), so ours runs on the same tensor.
  msa_mask = None
  if len(out) == 4:
    ref, msa_emb, z, msa_mask = out
  else:
    ref, msa_emb = out
  print('  embedded msa from native: %s' % (msa_emb.shape,))
  got = ours(args.model, cfg, model_dir, msa_emb, z, args.num_msa, n_tok,
             msa_mask=msa_mask)
  if os.environ.get('COMPARE') == 'msa':
    # The dump taps block 0's msa half: after the pair-weighted averaging, and
    # after the transition on top of it.
    d = np.load(path_of_dump(args.model))
    # Only the post-TRANSITION tap is comparable to a 1-block run of ours: our
    # EvoformerIteration does the pair-weighted averaging AND the transition,
    # and there is no way to stop between them. COMPARE_STEP=pwa is therefore
    # refused rather than silently compared -- reading a 2-step result against
    # a 1-step reference is the harness fault this file has hit five times.
    if os.environ.get('COMPARE_STEP') == 'pwa':
      raise SystemExit('COMPARE_STEP=pwa needs our transition ablated; a plain '
                       '1-block run has already applied it, so the comparison '
                       'would be 2 steps against 1')
    key = 'msa_after_transition'
    if key not in d:
      raise SystemExit('%s has no %s -- re-run the dump, it taps block 0 now'
                       % (os.path.basename(path_of_dump(args.model)), key))
    ref = d[key]
  print('  shapes: ours %s native %s' % (np.asarray(got).shape, ref.shape))
  _cmp('msa rows' if os.environ.get('COMPARE') == 'msa' else 'msa -> pair',
       np.asarray(got), ref)
  return 0


def ours(model, cfg, model_dir, msa_emb, z, n_msa, n_tok, msa_mask=None):
  """Our msa_stack's PAIR output, run the way prot_parity runs protenix's."""
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import model_config
  from alphafold3.model import params as afp
  from alphafold3.model.network import modules

  cfg.global_config.bfloat16 = 'none'
  p = afp.get_model_haiku_params(model_dir=model_dir)
  ms = cfg.evoformer.msa_stack
  # An adapter that ran a NON-UNIFORM msa mask hands it back, so ours runs the
  # same one -- an all-ones mask cannot distinguish two OPM normalisers, which
  # is how a wrong boltz2 fix passed once (model_config.OPM_ROW_COUNT_NORM).
  mm = (jnp.ones((n_msa, n_tok), jnp.float32) if msa_mask is None
        else jnp.asarray(np.asarray(msa_mask, np.float32)))
  masks = {'msa': mm, 'pair': jnp.ones((n_tok, n_tok), jnp.float32)}
  # The embedded MSA comes from NATIVE's own embedder (see the adapters), so
  # both sides enter the stack on the identical activation. Feeding ours a
  # random activation instead read corr 0.468 for rf3 -- not a finding, just two
  # different inputs.
  m_emb = np.asarray(msa_emb, np.float32)
  while m_emb.ndim > 3:
    m_emb = m_emb[0]

  mp = {k[len('diffuser/evoformer/'):]: v for k, v in p.items()
        if k.startswith('diffuser/evoformer/__layer_stack_no_per_layer/')}
  # TRUNCATE BOTH SIDES. Our params carry a leading layer_stack axis, so asking
  # for fewer blocks than the blob holds is a scan-length error rather than a
  # shorter run -- the same rule as every other block knob here.
  depth = int(os.environ.get('MSA_BLOCKS', ms.num_layer))
  if depth != ms.num_layer:
    mp = {k: {kk: (vv[:depth] if hasattr(vv, 'shape') and vv.ndim
                   and vv.shape[0] == ms.num_layer else vv)
              for kk, vv in v.items()} for k, v in mp.items()}

  def fwd(m_, z_):
    def blk(x):
      return modules.EvoformerIteration(
          ms, cfg.global_config, name='msa_stack',
          # ESMFold2's MSAEncoderBlock has no triangle ATTENTION -- only the two
          # triangle multiplications and the transition. Building it anyway
          # leaves those parameters at random init, which is a port bug the
          # graph already gates on (PAIR_ONLY_TRUNK); the gate has to match.
          with_pair_attention=(cfg.global_config.model
                               not in model_config.PAIR_ONLY_TRUNK),
      )(activations=x, masks=masks)
    out = hk.experimental.layer_stack(depth)(blk)({'msa': m_, 'pair': z_})
    # COMPARE=msa returns the MSA rows instead of the pair. Everything the MSA
    # side does reaches the pair only through the outer product, so the pair
    # output cannot say whether a residual came from the pair-weighted
    # averaging, the transition, or the outer product itself. The msa rows
    # separate the first two from the third.
    return out['msa'] if os.environ.get('COMPARE') == 'msa' else out['pair']

  return hk.transform(fwd).apply(mp, jax.random.PRNGKey(0),
                                 jnp.asarray(m_emb), jnp.asarray(z))


def esmfold2_opm_split(model, cfg, model_dir, n_msa, n_tok):
  """The outer product ALONE, against the dump's own tap. COMPARE=opm.

  The pair output cannot separate the outer product from the two triangle
  multiplications and the pair transition that follow it in the same block, and
  our EvoformerIteration offers no way to stop between them. So this runs
  `modules.OuterProductMean` on its own, on NATIVE's post-transition msa
  (`msa_after_transition` from the dump), and compares against native's own
  `pair_after_opm - z`. Both sides then see the identical msa rows and the
  identical mask, and the only thing under test is the outer product.

  This is the split `boltz2_layer_split` does in-process; ESMFold2 needs the npz
  because its implementation lives in ~/venv_esm.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp

  from alphafold3.model import params as afp
  from alphafold3.model.network import modules

  d = np.load(path_of_dump(model))
  for k in ('msa_after_transition', 'pair_after_opm', 'z', 'mask'):
    if k not in d:
      raise SystemExit('%s has no %s -- re-run the dump with --blocks 1'
                       % (os.path.basename(path_of_dump(model)), k))
  m2 = np.asarray(d['msa_after_transition'], np.float32)      # (M, L, c_m)
  ref = np.asarray(d['pair_after_opm'], np.float32) - np.asarray(d['z'],
                                                                 np.float32)
  msa_mask = np.asarray(d['mask'], np.float32)                # (M, L)
  print('  OPM alone: msa %s, mask %s, ref %s'
        % (m2.shape, msa_mask.shape, ref.shape))

  cfg.global_config.bfloat16 = 'none'
  p = afp.get_model_haiku_params(model_dir=model_dir)
  ms = cfg.evoformer.msa_stack

  def fwd(m_):
    return modules.OuterProductMean(
        ms.outer_product_mean, cfg.global_config,
        num_output_channel=cfg.evoformer.pair_channel,
        name='outer_product_mean')(m_, jnp.asarray(msa_mask))

  key = 'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/'
  op = {}
  for k, v in p.items():
    if k.startswith(key) and 'outer_product_mean' in k:
      # BLOCK 0 only: slice the layer_stack axis off, the same way the
      # block-truncating path above does.
      op[k[len(key):]] = {kk: (vv[0] if hasattr(vv, 'ndim') and vv.ndim
                               and vv.shape[0] == ms.num_layer else vv)
                          for kk, vv in v.items()}
  if not op:
    raise SystemExit('no outer_product_mean params under %r' % key)
  got = hk.transform(fwd).apply(op, jax.random.PRNGKey(0), jnp.asarray(m2))
  _cmp('OPM alone', np.asarray(got), ref)
  return 0


def boltz2_layer_split(cfg, model_dir, n_msa, n_tok):
  """LAYER=1: split boltz2's MSA layer into its m-side and z-side halves.

  The stack-level gate reads 0.974 and does NOT compound (one block 0.9676 vs
  four 0.9743), so the difference is inside ONE layer body. That leaves four
  candidates, and comparing `m` as well as `z` cuts them in two:

    * `m` differs  -> pair_weighted_averaging and/or msa_transition
    * `m` matches, `z` differs -> outer_product_mean and/or pairformer_layer

  Native's MSAModule returns only z, which is why the stack gate could not make
  this cut; `MSALayer` returns (z, m) and is what makes it possible.
  """
  import haiku as hk
  import jax
  import jax.numpy as jnp
  import torch

  from boltz.model.modules.trunkv2 import MSALayer

  from alphafold3.model import model_config
  from alphafold3.model import params as afp
  from alphafold3.model.network import modules

  ckpt = os.path.expanduser('~/boltz2_weights/boltz2_conf.ckpt')
  sd = torch.load(ckpt, map_location='cpu', weights_only=False)
  sd = sd.get('state_dict', sd)
  pre = 'msa_module.layers.0.'
  sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
  if not sub:
    raise SystemExit('no %r keys' % pre)
  msa_s = sub['msa_transition.fc1.weight'].shape[1] \
      if 'msa_transition.fc1.weight' in sub else 64
  token_z = sub['pair_weighted_averaging.proj_z.1.weight'].shape[1] \
      if 'pair_weighted_averaging.proj_z.1.weight' in sub \
      else cfg.evoformer.pair_channel
  print('  layer 0: msa_s %d, token_z %d' % (msa_s, token_z))
  net = MSALayer(msa_s=msa_s, token_z=cfg.evoformer.pair_channel,
                 msa_dropout=0.0, z_dropout=0.0)
  missing, unexpected = net.load_state_dict(sub, strict=False)
  print('  native layer: %d tensors, %d missing, %d unexpected %s'
        % (len(sub), len(missing), len(unexpected), sorted(missing)[:3]))
  assert not missing, 'native layer is missing %d tensors' % len(missing)
  net.eval()

  rng = np.random.default_rng(7)
  m0 = (rng.normal(size=(n_msa, n_tok, msa_s)) * 0.5).astype(np.float32)
  z0 = (rng.normal(size=(n_tok, n_tok, cfg.evoformer.pair_channel)) * 0.5
        ).astype(np.float32)
  t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
  # NONUNIFORM=1 masks out some rows for some tokens. An all-ones msa mask
  # cannot distinguish boltz's normaliser from ours: boltz divides by
  # `mask.sum(1)` -- the row count for token i, broadcast over j -- where AF3
  # divides by the PAIRWISE count einsum('abc,adc->bdc', mask, mask). Those are
  # equal only when every row covers every token.
  msa_mask_np = np.ones((n_msa, n_tok), np.float32)
  if os.environ.get('NONUNIFORM'):
    msa_mask_np[n_msa // 2:, :n_tok // 2] = 0.0
    print('  NONUNIFORM msa mask: %d of %d entries zero'
          % (int((msa_mask_np == 0).sum()), msa_mask_np.size))
  masks_msa = t(msa_mask_np)[None]
  with torch.no_grad():
    z_ref, m_ref = net(t(z0)[None], t(m0)[None],
                       torch.ones(1, n_tok, n_tok), masks_msa)
  z_ref = np.asarray(z_ref)[0]
  m_ref = np.asarray(m_ref)[0]

  cfg.global_config.bfloat16 = 'none'
  p = afp.get_model_haiku_params(model_dir=model_dir)
  ms = cfg.evoformer.msa_stack
  masks = {'msa': jnp.asarray(msa_mask_np),
           'pair': jnp.ones((n_tok, n_tok), jnp.float32)}

  def fwd(m_, z_):
    out = modules.EvoformerIteration(
        ms, cfg.global_config, name='msa_stack')(
            activations={'msa': m_, 'pair': z_}, masks=masks)
    return out['msa'], out['pair']

  mp = {}
  for k, v in p.items():
    key = 'diffuser/evoformer/__layer_stack_no_per_layer/'
    if k.startswith(key):
      # ONE block: slice the leading layer_stack axis off every parameter.
      mp[k[len('diffuser/evoformer/'):].replace(
          '__layer_stack_no_per_layer/', '')] = {
              kk: (vv[0] if hasattr(vv, 'ndim') and vv.ndim
                   and vv.shape[0] == ms.num_layer else vv)
              for kk, vv in v.items()}
  m_got, z_got = hk.transform(fwd).apply(mp, jax.random.PRNGKey(0),
                                         jnp.asarray(m0), jnp.asarray(z0))
  _cmp('m (msa side)', np.asarray(m_got), m_ref)
  _cmp('z (pair side)', np.asarray(z_got), z_ref)

  # z side splits again: OPM alone, then the pair stack. boltz's MSA layer uses
  # `PairformerNoSeqLayer` -- a DIFFERENT class from the trunk's
  # `PairformerLayer`, which is why the trunk being exact at 1.000000 does not
  # cover it.
  with torch.no_grad():
    opm_ref = np.asarray(net.outer_product_mean(t(m0)[None], masks_msa))[0]

  def opm_fwd(m_):
    return modules.OuterProductMean(
        ms.outer_product_mean, cfg.global_config,
        num_output_channel=cfg.evoformer.pair_channel,
        name='outer_product_mean')(m_, masks['msa'])

  op = {k: v for k, v in mp.items() if 'outer_product_mean' in k}
  op = {k.split('msa_stack/')[-1]: v for k, v in op.items()}
  try:
    opm_got = hk.transform(opm_fwd).apply(op, jax.random.PRNGKey(0),
                                          jnp.asarray(m0))
    _cmp('  OPM alone', np.asarray(opm_got), opm_ref)
    # PREDICTION, checked rather than asserted: ours computes (Wz + b)/n and
    # boltz computes W(z/n) + b, so native - ours must be a per-CHANNEL
    # constant equal to (1 - 1/n) * output_b. If the residual is not constant
    # across (i, j), the bias placement is not the whole story.
    d = np.asarray(opm_ref) - np.asarray(opm_got)
    per_ch = d.reshape(-1, d.shape[-1])
    spread = per_ch.std(0).max()
    b = np.asarray(op[[k for k in op if 'outer_product_mean' in k or True][0]]
                   .get('output_b', np.zeros(d.shape[-1])))
    pred = (1.0 - 1.0 / n_msa) * b
    print('    residual per channel: mean %.6f, spread across (i,j) %.3e'
          % (per_ch.mean(0).mean(), spread))
    print('    predicted (1-1/n)*output_b: mean %.6f | max|resid - pred| %.3e'
          % (pred.mean(), np.abs(per_ch.mean(0) - pred).max()))
  except Exception as e:  # noqa: BLE001
    print('  OPM split skipped: %s' % str(e)[:160])
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
