"""Where does a fold's time actually go? Stage by stage, measured.

Fitting (recycles, samples) points to a 306-residue A10 fold attributes ~14 s
to the trunk and ~8 s to the sampler and leaves **~9.6 s, a third of the fold,
unaccounted for**. This names that third by timing the seams
`staged.staged_fold` drives, which are disjoint -- unlike the naive reading of
`make_stages`, where `heads` falls through into the WHOLE sampler and
`diff_cond` runs it too unless `sample_config.stepwise` is set. Measured that
way the pieces summed to 49.9 s of a 31 s fold, which is how the mistake
announces itself.

  PYTHONPATH=src:. python dev/oracles/stage_budget.py <model> [length]
"""
import sys
import time

import jax
import numpy as np

sys.path.insert(0, '.')
from dev.oracles import fold_check            # noqa: E402


def timeit(fn, *a, iters=3, **kw):
  o = fn(*a, **kw)
  jax.block_until_ready(o)
  t = time.time()
  for _ in range(iters):
    o = fn(*a, **kw)
  jax.block_until_ready(o)
  return (time.time() - t) / iters


def main(model_name, length):
  import haiku as hk
  import jax.numpy as jnp
  from alphafold3.model import model as af3_model
  from alphafold3.model import params as afp
  from alphafold3.model import staged

  unit = 'ACDEFGHIKLMNPQRSTVWY'
  seq = (unit * (length // 20 + 1))[:length]
  batch, config, model_dir = fold_check._fold_setup(model_name, seq)
  params = afp.get_model_haiku_params(model_dir=model_dir)
  # stepwise makes diff_cond HAND BACK the conditioning instead of sampling.
  config.heads.diffusion.eval.stepwise = True
  steps = config.heads.diffusion.eval.steps
  samples = config.heads.diffusion.eval.num_samples
  recycles = af3_model.num_trunk_passes(config.num_recycles,
                                        config.global_config.model)
  rng = jax.random.PRNGKey(1)

  def stage(name):
    @hk.transform
    def fn(b, carry=None, key=None, diffusion_state=None):
      return af3_model.Model(config)(
          b, key=key, use_dropout=False, recycle_carry=carry, stage=name,
          diffusion_state=diffusion_state)
    return jax.jit(fn.apply)

  embed, trunk, cond, score = (stage('embed'), stage('trunk'),
                               stage('diff_cond'), stage('score'))
  t_embed = timeit(embed, params, rng, batch)
  carry, key = embed(params, rng, batch)
  t_trunk = timeit(trunk, params, rng, batch, carry=carry, key=key)
  for _ in range(recycles):
    carry, key, _ = trunk(params, rng, batch, carry=carry, key=key)

  t_cond = timeit(cond, params, rng, batch, carry=carry, key=key)
  st = cond(params, rng, batch, carry=carry, key=key)

  # ONE denoise step, for all samples at once (the body is vmapped over them).
  arrays, static = staged._split_static(st['atom_cond'])   # noqa: SLF001
  @hk.transform
  def denoise_fn(b, carry, key, dcarry, noise_level, pair_cond, atom_arrays):
    return af3_model.Model(config)(
        b, key=key, use_dropout=False, recycle_carry=carry, stage='denoise',
        diffusion_state=(dcarry, noise_level, pair_cond,
                         {**atom_arrays, **static}))
  denoise = jax.jit(denoise_fn.apply)
  levels = st['noise_levels']
  xs = (levels[1], levels[0])
  t_step = timeit(denoise, params, rng, batch, carry, key, st['init'], xs,
                  st['pair_cond'], arrays)
  out = denoise(params, rng, batch, carry, key, st['init'], xs,
                st['pair_cond'], arrays)

  # Scoring alone: hand it coordinates so it does NOT sample again.
  pos = out['atom_positions']
  t_score = timeit(score, params, rng, batch, carry=carry, key=key,
                   diffusion_state=(pos,))

  rows = [('embed (input + template + MSA init)', t_embed, 1),
          ('trunk (one recycle)', t_trunk, recycles),
          ('diffusion conditioning (once)', t_cond, 1),
          ('denoise step (all samples)', t_step, steps),
          ('confidence head + output', t_score, 1)]
  print(f'\n=== {model_name}, {length} residues: '
        f'{recycles} trunk passes, {steps} steps, {samples} samples')
  total = 0.0
  for label, secs, n in rows:
    total += secs * n
    print(f'  {label:38s} {secs * 1000:9.2f} ms x{n:<4d} = {secs * n:6.2f} s')
  print(f'  {"":38s} {"":9s}          {"-" * 8}')
  print(f'  {"total":38s} {"":9s}          {total:6.2f} s')


if __name__ == '__main__':
  main(sys.argv[1] if len(sys.argv) > 1 else 'openbind0',
       int(sys.argv[2]) if len(sys.argv) > 2 else 306)
