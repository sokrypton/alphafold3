"""Native chai-1 steady-state time per prediction, across lengths.

  CHAI_DOWNLOADS_DIR=~/chai1_weights ~/chai_venv/bin/python bench_chai_native.py <L> [reps]

chai exposes a Python API (`chai1.run_inference`), so the SAME process can fold
repeatedly: weights stay loaded and CUDA stays warm, which is what makes calls
2..n a steady state rather than a cold start.

`run_inference` also featurises and writes cif files each call, so this is
comparable to our (featurise + forward), not to our forward alone -- the same
caveat as native rosettafold3. Settings match bench_sweep.py: 3 recycles, 200
diffusion timesteps, 1 sample, no ESM, no MSA, no template.
"""
import os, sys, time, random, warnings, statistics
warnings.filterwarnings('ignore')
from pathlib import Path
os.environ.setdefault('CHAI_DOWNLOADS_DIR', os.path.expanduser('~/chai1_weights'))
# chai's tqdm bars emit a line per diffusion step -- 199 per call, which buried
# the actual result in tens of thousands of characters of progress output.
os.environ['TQDM_DISABLE'] = '1'

L = int(sys.argv[1])
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rng = random.Random(L)                      # same seed/alphabet as bench_sweep.py
SEQ = ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(L))

OUT = Path(f'/tmp/_chai_native_{L}')
OUT.mkdir(parents=True, exist_ok=True)
fasta = OUT / f'L{L}.fasta'
fasta.write_text(f'>protein|L{L}\n{SEQ}\n')

import torch
from chai_lab import chai1

# Split native's number the way the rosettafold3 harness does. chai calls every
# network through `ModuleWrapper.forward` (the modules themselves hold weights
# but have no forward methods, so torch hooks never fire -- see
# tools/oracles/chai1/run_native.py). Accumulating time across those calls gives
# MODEL time; whatever `run_inference` spends outside them is featurisation,
# conformer building and cif writing, none of which our forward-only number
# includes.
MODEL_T = {'t': 0.0, 'crop': None}
_orig_fwd = chai1.ModuleWrapper.forward


def _timed_fwd(self, *a, **kw):
  # `crop_size` is chai's BUCKET, and it is a real argument here
  # (ModuleWrapper.forward dispatches to `forward_{crop_size}`). Record it:
  # chai pads every input up to the next of
  # AVAILABLE_MODEL_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]
  # (data/collate/utils.py), so 64, 128 and 192 tokens ALL fold as 256. Timing
  # those against our unpadded 64/128/192 compares different problems and read
  # as a meaningless 9.9x/6.2x/3.9x. Only bucket BOUNDARIES are fair.
  crop = kw.get('crop_size', a[0] if a else None)
  if crop is not None:
    MODEL_T['crop'] = crop
  t0 = time.time()
  out = _orig_fwd(self, *a, **kw)
  torch.cuda.synchronize()
  MODEL_T['t'] += time.time() - t0
  return out


chai1.ModuleWrapper.forward = _timed_fwd

def one(i):
  d = OUT / f'pred{i}'
  if d.exists():
    import shutil
    shutil.rmtree(d)
  return chai1.run_inference(
      fasta_file=fasta, output_dir=d, use_esm_embeddings=False,
      use_msa_server=False, num_trunk_recycles=3, num_diffn_timesteps=200,
      num_diffn_samples=1, seed=0, device='cuda:0', low_memory=False)

times, model_times = [], []
for i in range(REPS + 1):
  MODEL_T['t'] = 0.0
  t0 = time.time()
  one(i)
  torch.cuda.synchronize()
  times.append(time.time() - t0)
  model_times.append(MODEL_T['t'])
  print(f'  call {i}: {times[-1]:.3f}s (model {model_times[-1]:.3f}s)', flush=True)

cut = max(1, (2 * len(times)) // 3)
tail, mtail = times[cut:] or times[-1:], model_times[cut:] or model_times[-1:]
total, modelt = statistics.median(tail), statistics.median(mtail)
print(f'NATIVE-CHAI1 L={L} bucket={MODEL_T["crop"]} first {times[0]:6.2f}s | '
      f'steady {total:6.3f}s | model {modelt:6.3f}s | '
      f'outside-model {total - modelt:6.3f}s | all {[round(t, 2) for t in times]}')
