"""A native protenix FOLD (not a trunk dump), for comparing fold quality.

    DTYPE=bf16 CASE=5k9p_plain CKPT=... MODEL=... DUMP_DIR=out \
    PYTHONPATH=~/px_stub:~/protenix ~/of3_venv/bin/python \
      dev/oracles/native_protenix_fold.py

Same environment recipe as native_protenix_dump.py, without the trunk hook, so
the job runs to completion and writes structures. Score them with
dev/oracles/score_native_cif.py.

NO_TEMPLATE=1 makes native's TemplateEmbedder return zero -- what our port used
to do with no template supplied. That is how protenix1's 9 A ubiquitin gap was
confirmed: native drops from 1.855 A to 8.40 A, onto our number. Changing the
REFERENCE and nothing else is the strongest form the claim can take.
"""

import os, sys, types
import torch

CASE = os.environ.get('CASE', '5k9p_plain')
CKPT = os.environ['CKPT']
MODEL = os.environ['MODEL']
JSONDIR = os.environ.get('JSONDIR', '/home/ubuntu/alphafold3/dev/oracles/native_json')
DUMP = os.environ.get('DUMP_DIR', '/tmp/px_fold_' + CASE)
SEED = os.environ.get('SEED', '0')
CYCLES = os.environ.get('CYCLES', '10')
NSAMPLE = os.environ.get('NSAMPLE', '5')


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

sys.argv = ['inference',
            '--input_json_path', '%s/%s.json' % (JSONDIR, CASE),
            '--dump_dir', DUMP,
            '--load_checkpoint_path', CKPT,
            '--model_name', MODEL,
            '--seeds', SEED,
            '--use_msa', 'false',
            '--triangle_multiplicative', 'torch',
            '--triangle_attention', 'torch',
            '--sample_diffusion.N_sample', NSAMPLE,
            '--sample_diffusion.N_step', '200',
            '--model.N_cycle', CYCLES, '--dtype', os.environ.get('DTYPE','bf16')]
# NO_TEMPLATE=1 makes native's TemplateEmbedder return zero, which is what our
# port does when no template is supplied. If native then folds as badly as we
# do, the missing term IS the gap -- the strongest available confirmation, since
# it changes native and nothing else.
if os.environ.get('NO_TEMPLATE'):
  from protenix.model.modules.pairformer import TemplateEmbedder
  TemplateEmbedder.forward = lambda self, *a, **kw: 0
  print('NATIVE TEMPLATE EMBEDDER DISABLED')

from runner.inference import run
run()
print('PXFOLDDONE')
