"""Native OpenFold-3 steady-state forward time.

  PYTHONPATH=~/if2_extra:~/openfold-3 ~/boltz_gpu_venv/bin/python \
      bench_of3_native.py <length> [reps]

Seam: `OpenFold3AllAtom.predict_step` calls `self(batch)` for the forward and
then `_compute_confidence_scores`. We patch predict_step, re-run `self(batch)`
REPS+1 times on a DEEP COPY of the batch (protenix2's predict consumed its
batch and raised KeyError on the second call -- assume the same here), and time
only the forward, so the number is comparable to our jitted forward.

Must run under a CUDA-torch venv: `~/venv` is torch+cpu and would silently
measure CPU. Prints NO_TIMINGS loudly if the patch never fires -- runpy-style
re-execution discards patches, which cost an entire run on protenix2.
"""
import os, sys, time, json, random, statistics, warnings
warnings.filterwarnings('ignore')

# Read from the ENVIRONMENT, not sys.argv. openfold-3's DataLoader spawns
# workers that RE-IMPORT this module as __main__ -- and by then sys.argv has been
# replaced with the openfold CLI args, so `int(sys.argv[1])` ran on the string
# 'predict' and every worker died with
#   ValueError: invalid literal for int() with base 10: 'predict'
# which surfaced only as "DataLoader worker (pid(s) ...) exited unexpectedly".
L = int(os.environ.get('OF3_L', sys.argv[1] if len(sys.argv) > 1
                       and sys.argv[1].isdigit() else 64))
REPS = int(os.environ.get('OF3_REPS', sys.argv[2] if len(sys.argv) > 2
                          and sys.argv[2].isdigit() else 2))
ROOT = '/home/ubuntu/openfold-3'
CKPT = '/home/ubuntu/of3-p2-155k.pt'
sys.path.insert(0, ROOT)

rng = random.Random(L)                      # same seed/alphabet as bench_sweep.py
SEQ = ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(L))
# openfold-3 has its OWN query schema (InferenceQuerySet): an OBJECT with a
# `queries` map of name -> {chains: [{molecule_type, chain_ids, sequence}]},
# NOT the AF3 fold-input list. See examples/example_inference_inputs/.
inp = f'/tmp/_of3_{L}.json'
json.dump({'seeds': [1],
           'queries': {f'L{L}': {'chains': [{'molecule_type': 'protein',
                                             'chain_ids': ['A'],
                                             'sequence': SEQ}]}}},
          open(inp, 'w'))

import torch
times = []

from openfold3.projects.of3_all_atom import runner as R

_orig_step = R.OpenFold3AllAtom.predict_step


def _timed_step(self, batch, batch_idx):
  import copy
  for i in range(REPS + 1):
    b = copy.deepcopy(batch)
    torch.cuda.synchronize()
    t0 = time.time()
    self(b)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
    print(f'  call {i}: {times[-1]:.3f}s', flush=True)
  return _orig_step(self, batch, batch_idx)


R.OpenFold3AllAtom.predict_step = _timed_step

# `--seed` belongs to the `train` subcommand, not `predict`; predict takes
# --query-json / --inference-ckpt-path / --output-dir / --num-diffusion-samples
# / --num-model-seeds / --runner-yaml / --use-msa-server / --use-templates.
sys.argv = ['run_openfold.py', 'predict', '--query-json', inp,
            '--inference-ckpt-path', CKPT, '--output-dir', f'/tmp/_of3_out_{L}',
            '--num-diffusion-samples', '1', '--num-model-seeds', '1']
try:
  from openfold3 import run_openfold
  run_openfold.cli(standalone_mode=False)
except SystemExit:
  pass
except Exception as e:
  print(f'DRIVER ERROR: {type(e).__name__}: {str(e)[:200]}', flush=True)

if times:
  cut = max(1, (2 * len(times)) // 3)
  tail = times[cut:] or times[-1:]
  warm = 'WARM' if len(times) > 1 else 'COLD-ONLY(not a steady state)'
  print(f'NATIVE-OPENFOLD3 L={L} {warm} first {times[0]:6.2f}s | '
        f'steady {statistics.median(tail):6.3f}s | all {[round(t, 2) for t in times]}')
else:
  print(f'NATIVE-OPENFOLD3 L={L} NO_TIMINGS (predict_step never called)')
