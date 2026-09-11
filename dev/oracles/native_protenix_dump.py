"""Dump NATIVE protenix's real trunk, per stage, from a real inference job.

    CASE=5k9p_plain CYCLES=1 \
    CKPT=~/protenix_weights/protenix_base_default_v1.0.0.pt \
    MODEL=protenix_base_default_v1.0.0 DUMP=out.npz \
    PYTHONPATH=~/px_stub:~/protenix ~/of3_venv/bin/python \
      dev/oracles/native_protenix_dump.py

`Protenix.get_pairformer_output` returns (s_inputs, s, z) -- Algorithm 1 after
recycling -- so wrapping it catches the trunk with no reimplementation, and the
template embedder, the MSA module and the pairformer stack get their own in/out
taps for the localisation ladder.

DEFAULTS TO fp32 (`--dtype fp32`). protenix autocasts to BF16, and its own
bf16-vs-fp32 trunk differs at corr 0.789 on ubiquitin -- a bf16 reference cannot
tell a port bug from rounding, and has faked several. DTYPE=bf16 measures what
that costs; it is not for leaving on.

~/px_stub exists because protenix pins biotite==1.4.0 and of3_venv carries
1.7.1: it symlinks 1.4.0 (plus optree) AHEAD of site-packages. Without it the
featuriser dies in `BondList._bonds`.
"""

import os, sys, types
import numpy as np
import torch

CASE = os.environ.get('CASE', '5k9p_plain')
CKPT = os.environ['CKPT']
MODEL = os.environ['MODEL']
DUMP = os.environ['DUMP']
JSONDIR = os.environ.get('JSONDIR', '/home/ubuntu/alphafold3/dev/oracles/native_json')
SEED = os.environ.get('SEED', '0')
CYCLES = os.environ.get('CYCLES', '10')
if os.environ.get('HIGHPREC'):
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision('highest')


def _ln(x, shape, w=None, b=None, eps=1e-5):
  dims = tuple(range(x.dim() - len(shape), x.dim()))
  mean = x.mean(dim=dims, keepdim=True)
  inv = torch.rsqrt(x.var(dim=dims, unbiased=False, keepdim=True) + eps)
  out = (x - mean) * inv
  if w is not None:
    out = out * w
  if b is not None:
    out = out + b
  return out, mean.squeeze(-1), inv.squeeze(-1)


class _E(types.ModuleType):
  def __getattr__(self, n):
    if not n.startswith('forward'):
      raise AttributeError(n)

    def f(x, shape, *rest):
      eps = rest[-1] if rest and isinstance(rest[-1], float) else 1e-5
      ts = [r for r in rest if torch.is_tensor(r)]
      return _ln(x, shape, ts[0] if ts else None,
                 ts[1] if len(ts) > 1 else None, eps)
    return f


sys.modules.setdefault('fast_layer_norm_cuda_v2', _E('fast_layer_norm_cuda_v2'))
_esm = types.ModuleType('esm')
_esm.FastaBatchedDataset = object
_esm.pretrained = object
sys.modules.setdefault('esm', _esm)

caught = {}


def _np(t):
  return t.detach().float().cpu().numpy() if torch.is_tensor(t) else None


import protenix.model.protenix as _P

_orig = _P.Protenix.get_pairformer_output
_installed = {}


def _tap(mod, name, in_keys=(0, 1)):
  orig = mod.forward

  def f(*a, **kw):
    ts = [x for x in a if torch.is_tensor(x)]
    for i, key in enumerate(in_keys):
      if isinstance(key, int) and key < len(ts):
        caught['%s_in%d' % (name, key)] = _np(ts[key])
    out = orig(*a, **kw)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    for i, o in enumerate(outs):
      if torch.is_tensor(o):
        caught['%s_out%d' % (name, i)] = _np(o)
    return out
  mod.forward = f


def hooked(self, *a, **kw):
  if not _installed:
    _tap(self.pairformer_stack, 'pf')
    if hasattr(self, 'msa_module'):
      _tap(self.msa_module, 'msa')
    if hasattr(self, 'template_embedder'):
      _tap(self.template_embedder, 'templ')
    _installed['yes'] = True
  s_inputs, s, z = _orig(self, *a, **kw)
  caught['s_inputs'] = _np(s_inputs)
  caught['s_trunk'] = _np(s)
  caught['z_trunk'] = _np(z)
  batch = kw.get('input_feature_dict') or (a[0] if a else None)
  if isinstance(batch, dict):
    # every MSA-side feature too: with a real MSA the MSA module does real work,
    # and its inputs are where a featurisation difference shows
    # the ATOM-level features too: with the trunk matching to 1e-6, what the
    # DIFFUSION is fed is the only thing left, and `ref_pos` is known to differ
    # (our conformers are a different torsion draw -- see PARITY.md's
    # featurisation diff)
    for k in list(batch):
      if any(t in k for t in ('ref_', 'atom_', 'is_', 'residue_index',
                              'asym_id', 'token_index', 'restype')):
        v = batch[k]
        if torch.is_tensor(v):
          caught['batch_' + k] = _np(v)
    for k in list(batch):
      if any(t in k for t in ('msa', 'profile', 'deletion', 'paired')):
        v = batch[k]
        if torch.is_tensor(v):
          caught['batch_' + k] = _np(v)
    for k in ('template_aatype', 'template_distogram', 'template_unit_vector',
              'template_pseudo_beta_mask', 'template_backbone_frame_mask',
              'token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id',
              'restype', 'is_protein', 'msa'):
      if k in batch and torch.is_tensor(batch[k]):
        caught['batch_' + k] = _np(batch[k])
  np.savez(DUMP, **caught)
  print('TRUNK DUMPED', {k: getattr(v, 'shape', v) for k, v in caught.items()},
        flush=True)
  if os.environ.get('DENOISE_STEP'):
    _hook_denoise(self)          # let the job run on into the sampler
    return s_inputs, s, z
  raise SystemExit(0)


# DENOISE_STEP=1: capture the FIRST denoise call of the real sampler -- the
# noisy coordinates it is given, the noise level, and what it returns -- so OUR
# diffusion head can be run on the SAME input from OUR OWN features. The L3 gate
# feeds our denoiser NATIVE's conditioning, so it cannot see a difference in the
# conditioning we build ourselves.
if os.environ.get('DENOISE_STEP'):
  _dm_calls = {'n': 0}

  def _hook_denoise(self):
    mod = self.diffusion_module
    orig = mod.forward

    want = int(os.environ.get('DENOISE_STEP', '1')) - 1   # 1-based, as read

    def f(*a, **kw):
      out = orig(*a, **kw)
      if _dm_calls['n'] == want:
        for k, v in list(kw.items()):
          if torch.is_tensor(v) and v.numel() < 4_000_000:
            caught['denoise_in_' + k] = _np(v)
        for i, v in enumerate(a):
          if torch.is_tensor(v) and v.numel() < 4_000_000:
            caught['denoise_in_arg%d' % i] = _np(v)
        caught['denoise_out'] = _np(out if torch.is_tensor(out) else out[0])
        np.savez(DUMP, **caught)
        print('DENOISE STEP DUMPED', sorted(k for k in caught if 'denoise' in k),
              flush=True)
        raise SystemExit(0)
      _dm_calls['n'] += 1
      return out
    mod.forward = f

_P.Protenix.get_pairformer_output = hooked

# MSA_DETERMINISTIC=1: protenix's MSAModule RANDOMLY SUBSAMPLES its rows on every
# forward pass, at INFERENCE too -- `sample_indices` draws
# sample_size ~ randint(1, n) and then randperm(n)[:sample_size] (model/utils.py,
# strategy "random"). So with a real MSA its trunk is stochastic per pass and per
# recycle, and no deterministic port can match one of its passes. This patch
# makes it take every row in order, which is the only way to compare like with
# like.
if os.environ.get('MSA_DETERMINISTIC'):
  import torch as _t

  from protenix.model import utils as _pxu
  _pxu.sample_indices = lambda n, device=None, lower_bound=1, strategy='random': (
      _t.arange(n, device=device))
  print('NATIVE MSA SUBSAMPLING DISABLED: every row, in order')

sys.argv = ['inference',
            '--input_json_path', '%s/%s.json' % (JSONDIR, CASE),
            '--dump_dir', '/tmp/px_dump_' + CASE,
            '--load_checkpoint_path', CKPT,
            '--model_name', MODEL,
            '--seeds', SEED,
            # USE_MSA: protenix reads unpairedMsa from the json ONLY with this
            # true. Left false while an MSA was supplied in the json, native
            # silently folded from the single sequence -- and its batch said so
            # (`msa` at (1, 76)), which is the only reason it was caught.
            '--use_msa', os.environ.get('USE_MSA', 'false'),
            '--triangle_multiplicative', 'torch',
            '--triangle_attention', 'torch',
            '--sample_diffusion.N_step', '200',
            '--model.N_cycle', CYCLES,
            # protenix's default eval precision is BF16 (configs_base
            # "dtype": "bf16", autocast in runner/inference.py:214). A bf16
            # reference cannot be told apart from a port bug -- it faked six of
            # them on protenix2 once -- so the dump defaults to fp32 and DTYPE
            # is there to measure what bf16 costs, not to leave it on.
            '--dtype', os.environ.get('DTYPE', 'fp32')]
from runner.inference import run
run()
