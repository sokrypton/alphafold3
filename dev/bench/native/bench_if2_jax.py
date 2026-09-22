"""IntelliGen's OWN jax IntelliFold-2, timed the same way as ours.

  IF2_L=64 IF2_REPS=3 ~/venv/bin/python bench_if2_jax.py

IntelliFold ships a jax implementation alongside its torch one:
`intellifold/run_jax_inference.py` is a vendored, patched AlphaFold 3 runner
(cli.py calls it exactly that). Its ModelRunner mirrors AF3's -- `model_params`,
`hk.transform(forward_fn)`, `jax.jit(forward_fn.apply)`, `run_inference` -- so
this is the ONE comparison in the set that is jax against jax: two independent
ports of identical weights, rather than a framework difference.

Seam: `ModelRunner.run_inference(featurised_example, rng_key)`, patched to call
the jitted model REPS+1 times on the same featurised batch (a jax forward does
not consume its input, unlike protenix2's predict).
"""
import os, sys, time, statistics, warnings
warnings.filterwarnings('ignore')

L = int(os.environ.get('IF2_L', '64'))
REPS = int(os.environ.get('IF2_REPS', '3'))
ROOT = '/home/ubuntu/IntelliFold'
sys.path.insert(0, ROOT)

times = []
import jax
from intellifold import run_jax_inference as RJ

_orig_ri = RJ.ModelRunner.run_inference


def _timed_ri(self, featurised_example, rng_key):
  out = None
  for i in range(REPS + 1):
    t0 = time.time()
    out = _orig_ri(self, featurised_example, rng_key)
    jax.block_until_ready(out)
    times.append(time.time() - t0)
    print(f'  call {i}: {times[-1]:.3f}s', flush=True)
  return out


RJ.ModelRunner.run_inference = _timed_ri

# Their runner is AF3's, so it takes AF3 flags and an AF3 fold-input JSON.
import json, random
rng = random.Random(L)                     # same sequence as every other harness
SEQ = ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(L))
inp = f'/tmp/_if2jax_{L}.json'
json.dump({'name': f'L{L}',
           'sequences': [{'protein': {'id': 'A', 'sequence': SEQ,
                                      'unpairedMsa': '', 'pairedMsa': '',
                                      'templates': []}}],
           'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 4},
          open(inp, 'w'))

sys.argv = ['run_jax_inference.py',
            f'--json_path={inp}',
            f'--output_dir=/tmp/_if2jax_out_{L}',
            f'--model_dir={os.path.expanduser("~/model_v2")}',
            '--norun_data_pipeline',
            '--flash_attention_implementation=xla',   # A10 cannot use the others
            '--num_recycles=3', '--num_diffusion_samples=1']
try:
  from absl import app
  app.run(RJ.main, argv=sys.argv)
except SystemExit:
  pass
except Exception as e:
  print(f'DRIVER ERROR: {type(e).__name__}: {str(e)[:220]}', flush=True)

if times:
  cut = max(1, (2 * len(times)) // 3)
  tail = times[cut:] or times[-1:]
  warm = 'WARM' if len(times) > 1 else 'COLD-ONLY(not a steady state)'
  print(f'NATIVE-IF2-JAX L={L} {warm} first {times[0]:6.2f}s | steady '
        f'{statistics.median(tail):6.3f}s | all {[round(t, 2) for t in times]}')
else:
  print(f'NATIVE-IF2-JAX L={L} NO_TIMINGS (run_inference never called)')
