"""Native RoseTTAFold3 steady-state time for one 6MRR prediction.

Matched to bench_ours.py as closely as the two APIs allow: same target, 3
recycles, 200 sampling steps, 1 diffusion sample, no MSA or template.

The engine's `run()` includes featurisation, which our jitted forward does not,
so the fair comparison on our side is (featurise + forward), not forward alone.
Both include the confidence head.

  PYTHONPATH=~/rf3_extra:~/foundry_rf3/src:~/foundry_rf3/models/rf3/src \
      ~/venv/bin/python bench_rf3_native.py <input> [reps]
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

INP = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/6MRR.pdb'
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CKPT = '/home/ubuntu/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt'
CFG_DIR = '/home/ubuntu/foundry_rf3/models/rf3/configs'

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

with initialize_config_dir(config_dir=CFG_DIR, version_base='1.3'):
  cfg = compose(config_name='inference',
                overrides=[f'inputs={INP}', 'out_dir=/tmp/_rf3_bench',
                           f'ckpt_path={CKPT}', 'inference_engine=rf3',
                           'n_recycles=3', 'num_steps=200'])
  from rf3.inference import run_inference
  import rf3.inference as RI

  # run_inference builds the engine and calls run(); wrap run() so the SAME
  # engine (weights loaded once, CUDA warm) serves every repetition.
  times = {}

  def timed_run(engine, *a, **kw):
    out = None
    t0 = time.time(); out = engine._orig_run(*a, **kw)
    import torch; torch.cuda.synchronize()
    times.setdefault('all', []).append(time.time() - t0)
    for _ in range(REPS - 1):
      t0 = time.time(); engine._orig_run(*a, **kw); torch.cuda.synchronize()
      times['all'].append(time.time() - t0)
    return out

  # Time the PIPELINE separately, so native's number can be split into
  # featurisation and everything else. Without this the comparison against our
  # forward-only timing is only an upper bound: `run()` also parses the cif,
  # featurises through atomworks and writes outputs, none of which our number
  # includes. Same wrapper point dump_native_batch.py uses.
  from foundry.inference_engines.base import BaseInferenceEngine
  feat_times = []
  _orig_cp = BaseInferenceEngine._construct_pipeline

  def construct(self, cfg_):
    _orig_cp(self, cfg_)
    built = self.pipeline

    def wrapped(x):
      t0 = time.time()
      out = built(x)
      feat_times.append(time.time() - t0)
      return out
    self.pipeline = wrapped
  BaseInferenceEngine._construct_pipeline = construct

  from rf3.inference_engines.rf3 import RF3InferenceEngine
  RF3InferenceEngine._orig_run = RF3InferenceEngine.run
  RF3InferenceEngine.run = timed_run
  run_inference(cfg)

ts = times.get('all', [])
if ts:
  steady = sorted(ts[1:])[len(ts[1:]) // 2] if len(ts) > 1 else ts[0]
  import statistics
  feat = statistics.median(feat_times[1:]) if len(feat_times) > 1 else (
      feat_times[0] if feat_times else float('nan'))
  print(f'NATIVE-RF3 first {ts[0]:6.2f}s | steady {steady:6.3f}s | '
        f'featurise {feat:6.3f}s | steady-minus-featurise {steady - feat:6.3f}s | '
        f'all {[round(t, 2) for t in ts]}')
