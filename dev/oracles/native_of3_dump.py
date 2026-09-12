"""Dump NATIVE openfold3/openbind0's REAL trunk output from a real inference job.

The of3 analogue of dev/oracles/native_trunk_dump.sh (which is protenix's).
One hook is enough here, and it is the clean seam: `OpenFold3.run_trunk`
returns exactly (s_input, s_trunk, z_trunk) -- Algorithm 1 lines 1-14, after
recycling -- so wrapping it catches the three tensors real_trunk_parity.py
wants with no reimplementation and no tracer problem.

The distogram head is hooked too, because the distogram is what named the
trunk as the suspect (contact precision 0.303 vs openfold3's 0.868).

  DUMP=<out.npz> CKPT=~/of3-ob-174k.pt TREE=~/openfold-3-v050 \
    ~/of3_venv/bin/python dump_trunk.py <query.json>
"""
import os, sys
import numpy as np
import torch

TREE = os.environ.get('TREE', '/home/ubuntu/openfold-3-v050')
CKPT = os.environ.get('CKPT', '/home/ubuntu/of3-ob-174k.pt')
DUMP = os.environ.get('DUMP', '/tmp/of3_trunk.npz')
OUTDIR = os.environ.get('OUTDIR', '/tmp/of3_trunk_out')
sys.path.insert(0, TREE)
sys.path.insert(0, os.environ.get('STUB', os.path.expanduser('~/of3_stub')))


# NATIVE'S OWN FLOOR. of3 calls torch.set_float32_matmul_precision("high") in
# `_torch_gpu_setup`, i.e. TF32 -- 10 mantissa bits. With HIGHPREC=1 the same
# job runs with TF32 off, and the difference between the two dumps is how much
# native's OWN arithmetic moves its trunk. A disagreement at or below that is
# not evidence of anything.
if os.environ.get('HIGHPREC'):
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision('highest')
  _sfmp = torch.set_float32_matmul_precision
  torch.set_float32_matmul_precision = lambda *a, **k: _sfmp('highest')

from openfold3.projects.of3_all_atom.model import OpenFold3

caught = {}
_orig = OpenFold3.run_trunk


def run_trunk(self, batch, num_cycles, inplace_safe=False):
  s_input, s, z = _orig(self, batch, num_cycles, inplace_safe)
  caught['s_inputs'] = s_input.detach().float().cpu().numpy()
  caught['s_trunk'] = s.detach().float().cpu().numpy()
  caught['z_trunk'] = z.detach().float().cpu().numpy()
  caught['num_cycles'] = np.asarray(num_cycles)
  # the ATOM-level features too, for dev/oracles/featurisation_diff.py -- the
  # direction L0-L4 cannot see, since they feed each module native's own features
  for k in list(batch):
    if any(t in k for t in ('ref_', 'atom_', 'template_', 'profile', 'deletion',
                            'token_bonds', 'is_')):
      v = batch[k]
      if torch.is_tensor(v):
        caught['batch_' + k] = _np(v)
  for k in ('token_mask', 'restype', 'asym_id', 'residue_index', 'entity_id',
            'sym_id', 'token_index', 'msa', 'is_protein'):
    if k in batch:
      v = batch[k]
      caught['batch_' + k] = (v.detach().cpu().numpy()
                              if torch.is_tensor(v) else np.asarray(v))
  return s_input, s, z



OpenFold3.run_trunk = run_trunk


def _np(t):
  return t.detach().float().cpu().numpy() if torch.is_tensor(t) else None


def _stage(mod_attr, name, take_in=(), take_out=None):
  """Wrap one sub-module of OpenFold3 and keep the LAST call's tensors.

  Last call = the final recycling cycle, which is the one whose output becomes
  z_trunk. Recording every cycle would be more data and no more information:
  what we need is a pair of tensors our own stack can be fed on the same input.
  """
  def install(self):
    mod = getattr(self, mod_attr)
    orig = mod.forward

    def fwd(*a, **kw):
      for key, src in take_in:
        v = kw.get(key)
        if v is None and isinstance(src, int) and src < len(a):
          v = a[src]
        if v is not None:
          caught['%s_in_%s' % (name, key)] = _np(v)
      out = orig(*a, **kw)
      t = out
      if isinstance(out, (tuple, list)):
        t = out[take_out] if take_out is not None else out[0]
      if torch.is_tensor(t):
        caught['%s_out' % name] = _np(t)
      elif isinstance(out, (tuple, list)):
        for i, o in enumerate(out):
          if torch.is_tensor(o):
            caught['%s_out%d' % (name, i)] = _np(o)
      return out

    mod.forward = fwd
  return install


_installs = [
    # input embedder -> (s_input, s_init, z_init)
    _stage('input_embedder', 'embed'),
    _stage('template_embedder', 'templ', take_in=[('z', 1)]),
    _stage('msa_module', 'msa', take_in=[('m', 0), ('z', 1)]),
    _stage('pairformer_stack', 'pf', take_in=[('s', 0), ('z', 1)]),
]
_orig_init_done = {}
_orig_run = OpenFold3.run_trunk


def run_trunk_installing(self, *a, **kw):
  if not _orig_init_done:
    for ins in _installs:
      try:
        ins(self)
      except Exception as e:
        print('hook failed:', e)
    # the embedder returns a 3-tuple; keep all three
    emb = self.input_embedder
    _emb_fwd = emb.forward

    def emb_fwd(*aa, **kk):
      out = _emb_fwd(*aa, **kk)
      for nm, t in zip(('s_input', 's_init', 'z_init'), out):
        if torch.is_tensor(t):
          caught['embed_' + nm] = _np(t)
      return out

    emb.forward = emb_fwd

    # Per-BLOCK pairformer taps: the bisect axis. Each block's output pair is
    # kept under pf_blk<k>_z, so our own stack run with BLOCKS=k has something
    # to be compared against.
    try:
      for k, blk in enumerate(self.pairformer_stack.blocks):
        def mk(k, blk):
          orig = blk.forward

          def f(*aa, **kk):
            out = orig(*aa, **kk)
            if isinstance(out, (tuple, list)):
              for nm, t in zip(('s', 'z'), out):
                if torch.is_tensor(t):
                  caught['pf_blk%d_%s' % (k, nm)] = _np(t)
            return out
          blk.forward = f
        mk(k, blk)
    except Exception as e:
      print('block hook failed:', e)
    _orig_init_done['yes'] = True
  return _orig_run(self, *a, **kw)


OpenFold3.run_trunk = run_trunk_installing

# The distogram: hooked on the head module so we get the logits as the model
# produces them, not re-derived from z.
try:
  from openfold3.core.model.heads.distogram import DistogramHead
except Exception:
  DistogramHead = None
if DistogramHead is not None:
  _dg = DistogramHead.forward

  def dg(self, *a, **kw):
    out = _dg(self, *a, **kw)
    t = out[0] if isinstance(out, (tuple, list)) else out
    if torch.is_tensor(t):
      caught['distogram_logits'] = t.detach().float().cpu().numpy()
    return out

  DistogramHead.forward = dg

sys.argv = ['run_openfold.py', 'predict',
            '--query_json', sys.argv[1],
            '--inference_ckpt_path', CKPT,
            '--num_diffusion_samples', '5',
            '--num_model_seeds', '1',
            '--runner_yaml', os.environ.get('RUNNER_YAML', ''), '--use_msa_server', 'false',
            '--use_templates', 'false',
            '--output_dir', OUTDIR]
os.chdir(TREE)
from openfold3.run_openfold import cli
try:
  cli(standalone_mode=False)
finally:
  if caught:
    np.savez(DUMP, **caught)
    print('dumped', DUMP, {k: getattr(v, "shape", v) for k, v in caught.items()})
  else:
    print('NOTHING CAUGHT -- run_trunk never ran')
