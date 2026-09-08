#!/bin/bash
# Dump NATIVE protenix's real trunk output: s_inputs, s_trunk and the RAW
# z_trunk, for dev/oracles/real_trunk_parity.py to compare against ours.
#
#   bash dev/oracles/native_trunk_dump.sh <case>      # case = a json in $JSONDIR
#
# Why a shell script and not an in-process adapter like every other oracle
# here: this one needs protenix's own RUNNER and FEATURISER, not one module, so
# it runs a real inference job and stops it the moment the trunk is done.
#
# Two hooks, and both are load-bearing:
#   * `Protenix.sample_diffusion` is patched on the CLASS. The module-level
#     `generator.sample_diffusion` is imported by value into protenix.py, so
#     patching THAT does nothing and the job runs to completion (11 s, no dump).
#   * `DiffusionConditioning.prepare_cache` is patched to catch the RAW pair.
#     protenix precomputes a CONDITIONED pair and passes `z_trunk=None`, so the
#     tensor arriving at the sampler under `pair_z` is NOT the trunk pair --
#     comparing ours against it reads corr 0.011 and means nothing.
#
# Runs on boltz_gpu_venv (torch+cu130) because ~/venv's torch is CPU-only and
# must stay that way; px_deps symlinks the few pure-python packages it lacks.
set -e
CASE=${1:?usage: native_trunk_dump.sh <case-name>   (reads $JSONDIR/<case>.json)}
JSONDIR=${JSONDIR:-$PWD/dev/oracles/native_json}
OUT=${OUT:-$PWD/dev/oracles/native_trunk_$CASE.npz}
DEPS=${PX_DEPS:?set PX_DEPS to a dir symlinking the pure-python packages boltz_gpu_venv lacks}
CKPT=${CKPT:-/home/ubuntu/protenix_weights/protenix-v2.pt}
MODEL=${MODEL:-protenix-v2}
SEED=${SEED:-101}
CYCLES=${CYCLES:-10}
cd ~/protenix
export PYTHONPATH=$DEPS:/home/ubuntu/protenix
echo "=== native $MODEL trunk dump: $CASE (seed $SEED, $CYCLES cycles)"
  ~/boltz_gpu_venv/bin/python -c "
import sys, types, torch
def _ln(x, shape, w=None, b=None, eps=1e-5):
    dims = tuple(range(x.dim() - len(shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    inv = torch.rsqrt(x.var(dim=dims, unbiased=False, keepdim=True) + eps)
    out = (x - mean) * inv
    if w is not None: out = out * w
    if b is not None: out = out + b
    return out, mean.squeeze(-1), inv.squeeze(-1)
class _E(types.ModuleType):
    def __getattr__(self, n):
        if not n.startswith('forward'): raise AttributeError(n)
        def f(x, shape, *rest):
            eps = rest[-1] if rest and isinstance(rest[-1], float) else 1e-5
            ts = [r for r in rest if torch.is_tensor(r)]
            return _ln(x, shape, ts[0] if ts else None, ts[1] if len(ts)>1 else None, eps)
        return f
sys.modules.setdefault('fast_layer_norm_cuda_v2', _E('fast_layer_norm_cuda_v2'))
_esm = types.ModuleType('esm'); _esm.FastaBatchedDataset = object; _esm.pretrained = object
sys.modules.setdefault('esm', _esm)
sys.argv = ['inference',
  '--input_json_path', '$JSONDIR/${CASE}.json',
  '--dump_dir', '/tmp/native_trunk_dump_${CASE}',
  '--load_checkpoint_path', '$CKPT',
  '--model_name', '$MODEL',
  '--seeds', '$SEED',
  '--use_msa', 'false',
  '--triangle_multiplicative', 'torch',
  '--triangle_attention', 'torch',
  '--sample_diffusion.N_step', '200',
  '--model.N_cycle', '$CYCLES']
import numpy as np
import protenix.model.protenix as _P
from protenix.model.modules.diffusion import DiffusionConditioning as _DC
_saved = {}
_pc = _DC.prepare_cache
def _pc_hook(self, relp, z, *a, **kw):
    _saved['z_trunk'] = z.detach().float().cpu().numpy()
    print('RAW z_trunk captured', z.shape, flush=True)
    return _pc(self, relp, z, *a, **kw)
_DC.prepare_cache = _pc_hook
def _hook(self, **kw):
    out = {k: v.detach().float().cpu().numpy()
           for k, v in kw.items() if hasattr(v, 'detach')}
    out.update(_saved)
    np.savez('/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/native_trunk_plain.npz', **out)
    print('TRUNK DUMPED', sorted(out), flush=True)
    raise SystemExit(0)
_P.Protenix.sample_diffusion = _hook
from runner.inference import run
run()
" 2>&1 | tail -6
echo "wrote $OUT"
