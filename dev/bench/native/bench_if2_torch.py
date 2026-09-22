"""Native IntelliFold-2 (TORCH) steady-state forward time.

  IF2_L=64 IF2_REPS=3 PYTHONPATH=~/if2_extra:~/IntelliFold \
      ~/boltz_gpu_venv/bin/python bench_if2_torch.py

IntelliFold ships BOTH a torch implementation (runner/intellifold_inference.py)
and a jax one (intellifold/run_jax_inference.py, a vendored patched AlphaFold 3).
This times the torch original; bench_if2_jax.py times theirs in jax. Together
with our own port that gives three numbers on identical weights -- and the
jax-vs-jax pair is the interesting one, two independent ports of the same model.

Seam: `predict_and_save` calls `model(input_features, diffusion_batch_size=...)`
under torch.no_grad(). We wrap the model object itself so only the forward is
timed, not featurisation or file writing.

`~/venv` is torch+CPU, so this MUST run under a CUDA venv (boltz_gpu_venv) with
IntelliFold's extra deps on PYTHONPATH -- see NATIVE_SETUP.md.
Parameters come from the ENVIRONMENT, never sys.argv: the harness rewrites argv
to drive the CLI and any DataLoader worker re-importing this module would then
parse the CLI's words as numbers.
"""
import os, sys, time, json, random, statistics, warnings
warnings.filterwarnings('ignore')

L = int(os.environ.get('IF2_L', '64'))
REPS = int(os.environ.get('IF2_REPS', '3'))
ROOT = '/home/ubuntu/IntelliFold'
sys.path.insert(0, ROOT)

rng = random.Random(L)                     # same sequence as every other harness
SEQ = ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(L))
inp = f'/tmp/_if2_{L}.json'
json.dump({'sequences': [{'protein': {'id': 'A', 'sequence': SEQ}}],
           'name': f'L{L}'}, open(inp, 'w'))

import torch
times = []
import runner.intellifold_inference as II

_orig_pas = II.predict_and_save


def _timed_pas(model, *a, **kw):
  """Wrap the MODEL, then let predict_and_save run once as usual."""
  if not getattr(model, '_timed', False):
    inner = model.__call__

    def timed_call(*ca, **ckw):
      out = None
      for i in range(REPS + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        out = inner(*ca, **ckw)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f'  call {i}: {times[-1]:.3f}s', flush=True)
      return out
    model.__call__ = timed_call
    model._timed = True
  return _orig_pas(model, *a, **kw)


II.predict_and_save = _timed_pas

sys.argv = ['run_intellifold.py', inp, '--out_dir', f'/tmp/_if2_out_{L}',
            '--cache', os.path.expanduser('~/model_v2'), '--model', 'v2',
            '--recycling_iters', '3', '--sampling_steps', '200',
            '--num_diffusion_samples', '1', '--override']
# run_intellifold builds its parser under `if __name__ == "__main__"`, so import
# the module and re-run that block's tail: construct the parser the same way and
# hand main() the parsed args. runpy is deliberately avoided -- it re-executes
# the module as __main__ and would discard the predict_and_save patch above
# (that silently cost a whole protenix2 run).
try:
  import run_intellifold
  import argparse as _ap
  _src = open(f'{ROOT}/run_intellifold.py').read().split('if __name__ ==')[1]
  _ns = {'argparse': _ap, '__name__': '__notmain__'}
  exec('\n'.join(l[4:] if l.startswith('    ') else l
                  for l in _src.split('\n')[1:]).replace('main(args)', 'pass'),
       run_intellifold.__dict__ | _ns, _ns)
  run_intellifold.main(_ns['args'])
except SystemExit:
  pass
except Exception as e:
  print(f'DRIVER ERROR: {type(e).__name__}: {str(e)[:220]}', flush=True)

if times:
  cut = max(1, (2 * len(times)) // 3)
  tail = times[cut:] or times[-1:]
  warm = 'WARM' if len(times) > 1 else 'COLD-ONLY(not a steady state)'
  print(f'NATIVE-IF2-TORCH L={L} {warm} first {times[0]:6.2f}s | steady '
        f'{statistics.median(tail):6.3f}s | all {[round(t, 2) for t in times]}')
else:
  print(f'NATIVE-IF2-TORCH L={L} NO_TIMINGS (model never called)')
