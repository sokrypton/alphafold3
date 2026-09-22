"""Native protenix-v2 / OpenDDE steady-state forward time, across lengths.

  bench_px_native.py <protenix2|opendde> <length> [reps]

Both projects expose `InferenceRunner.predict(data)`, which takes an ALREADY
FEATURISED batch and runs the model. Wrapping it gives a forward-only number --
directly comparable to our jitted forward, with no featurisation or file IO
folded in (unlike native rosettafold3, whose run() bundles all three).

The wrapper re-runs predict REPS+1 times on the same data inside the one process,
so weights stay loaded and CUDA stays warm; call 0 is the warm-up.

Reuses the stubs from tools/oracles/<model>/run_native.py -- esm (imported
unconditionally by the dataloader, unused here) and the fast layer-norm CUDA
extension (JIT-built with ninja, absent) -- because ~/venv must not have
packages installed into it.
"""
import os, sys, time, json, random, statistics, warnings, importlib.util
warnings.filterwarnings('ignore')

MODEL = sys.argv[1]
L = int(sys.argv[2])
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ROOT = {'protenix2': '/home/ubuntu/protenix', 'opendde': '/home/ubuntu/OpenDDE'}[MODEL]
sys.path.insert(0, '/home/ubuntu/ColabDesign2')

# pull the stubs out of the existing oracle runner without executing its main()
spec = importlib.util.spec_from_file_location(
    f'_native_{MODEL}', f'/home/ubuntu/ColabDesign2/tools/oracles/{MODEL}/run_native.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._stub_esm()
mod._stub_fast_ln()

rng = random.Random(L)                      # same seed/alphabet as bench_sweep.py
SEQ = ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(L))
inp = f'/tmp/_px_{MODEL}_{L}.json'
json.dump([{'name': f'L{L}', 'sequences': [{'proteinChain': {
    'sequence': SEQ, 'count': 1}}]}], open(inp, 'w'))

sys.path.insert(0, ROOT)
times = []

# Import the module and patch it IN PLACE, then call its own run(). Do NOT use
# runpy.run_path: that re-executes inference.py under the name __main__, which
# builds a FRESH InferenceRunner class, so a patch applied to the previously
# imported class is silently discarded and `predict` is never timed (the harness
# ran to completion and reported "predict never called").
import runner.inference as _inf

_orig_predict = _inf.InferenceRunner.predict


def _timed_predict(self, data):
  # DEEP-COPY per call: predict() CONSUMES the batch it is handed (the second
  # call died with KeyError: 'profile'), so re-running it on the same dict gives
  # one usable timing and an exception that inference.py swallows as "L64
  # failed". One cold call is not a steady state -- it carries lazy CUDA and
  # kernel init that every other port's number excludes.
  import copy
  import torch
  out = None
  for i in range(REPS + 1):
    batch = copy.deepcopy(data)
    torch.cuda.synchronize()
    t0 = time.time()
    out = _orig_predict(self, batch)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
    print(f'  call {i}: {times[-1]:.3f}s', flush=True)
  return out


_inf.InferenceRunner.predict = _timed_predict
argv = ['inference.py', '--seeds', '101', '--dump_dir', f'/tmp/_px_dump_{MODEL}_{L}',
        '--input_json_path', inp, '--model.N_cycle', '3',
        '--sample_diffusion.N_sample', '1', '--sample_diffusion.N_step', '200']
if MODEL == 'protenix2':
  # protenix's OWN default is cuequivariance for both triangle ops
  # (configs/configs_base.py:129-130; options triattention, cuequivariance,
  # deepspeed, torch). The oracle forces `torch` only because those kernels are
  # not installed -- but that measures native with its optimisations OFF and
  # flatters us. PX_KERNELS selects: set it to `cuequivariance` once
  # cuequivariance-torch is present, and say which was used when quoting a ratio.
  kern = os.environ.get('PX_KERNELS', 'torch')
  argv += ['--model_name', 'protenix-v2', '--triangle_attention', kern,
           '--triangle_multiplicative', kern]
else:
  # opendde exposes the same knob (config/inference.py: TRIANGLE_KERNELS =
  # auto, cuequivariance, torch). Default `auto` resolves to cuequivariance when
  # cuequivariance-torch is importable -- which is why installing it for
  # protenix2 silently changed opendde's configuration too.
  argv += ['--load_checkpoint_path',
           os.path.expanduser('~/opendde_weights/opendde.pt')]
  kern = os.environ.get('PX_KERNELS', '')
  if kern:
    argv += ['--triangle_attention', kern, '--triangle_multiplicative', kern]
sys.argv = argv
try:
  _inf.run()
except SystemExit:
  pass

if times:
  tail = times[max(1, (2 * len(times)) // 3):] or times[-1:]
  warm = 'WARM' if len(times) > 1 else 'COLD-ONLY(1 call, not a steady state)'
  # Only protenix2 has its kernels forced here; opendde uses its OWN config
  # default (which resolves to cuequivariance now that cuequivariance-torch is
  # installed -- its log says "Resolved triangle kernels from ..."). Printing the
  # PX_KERNELS default for opendde claimed `torch` when cuEq was actually active.
  kern = os.environ.get('PX_KERNELS') or ('torch' if MODEL == 'protenix2' else 'model-default')
  print(f'NATIVE-{MODEL.upper()} L={L} kernels={kern} '
        f'{warm} first {times[0]:6.2f}s | steady {statistics.median(tail):6.3f}s | '
        f'all {[round(t, 2) for t in times]}')
else:
  print(f'NATIVE-{MODEL.upper()} L={L} NO_TIMINGS (predict never called)')
