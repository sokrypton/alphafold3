"""Where does a fold's time actually go? Stage by stage, measured.

A 306-residue fold on an A10 is ~31 s warm. Fitting (recycles, samples) points
attributes ~14 s to the trunk and ~8 s to the sampler and leaves **~9.6 s, a
third of the fold, unaccounted for** -- neither trunk nor sampler. This times
each stage `staged.make_stages` exposes, so that third has a name.

  PYTHONPATH=src:. python dev/oracles/stage_budget.py <model> [length]
"""
import sys
import time

import jax
import numpy as np

sys.path.insert(0, '.')
from dev.oracles import fold_check            # noqa: E402  (batch builder)


def timeit(fn, *a, iters=5, **kw):
  o = fn(*a, **kw)
  jax.block_until_ready(o)
  t = time.time()
  for _ in range(iters):
    o = fn(*a, **kw)
  jax.block_until_ready(o)
  return (time.time() - t) / iters


def main(model_name, length):
  from alphafold3.model import staged
  unit = 'ACDEFGHIKLMNPQRSTVWY'
  seq = (unit * (length // 20 + 1))[:length]
  batch, params, config = fold_check.build_batch(model_name, seq)[:3]

  stages = staged.make_stages(config)
  rng = jax.random.PRNGKey(1)
  out = {}

  embed = timeit(stages['embed'], params, rng, batch)
  out['embed (input + template + MSA)'] = embed
  carry = stages['embed'](params, rng, batch)

  trunk = timeit(stages['trunk'], params, rng, batch, carry=carry)
  out['trunk, ONE recycle'] = trunk
  emb = stages['trunk'](params, rng, batch, carry=carry)

  cond = timeit(stages['diff_cond'], params, rng, batch, carry=emb)
  out['diffusion conditioning'] = cond

  print(f'\n=== {model_name}, {length} residues')
  for k, v in out.items():
    print(f'  {k:34s} {v * 1000:9.1f} ms')
  print(f'\n  a 10-recycle fold spends {trunk * 10:.2f} s in the trunk and '
        f'{embed + cond:.2f} s in embed+cond before a single sampler step.')


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else 'openbind0',
       int(sys.argv[2]) if len(sys.argv) > 2 else 306)
