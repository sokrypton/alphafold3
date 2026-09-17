"""The model as separable STAGES, driven from Python.

`staged_fold` returns `fold(rng, batch, on_frame=None, on_step=None)`, which
runs the trunk one pass at a time and the sampler one chunk at a time, calling
back after each. Three things this buys that the fused graph cannot:

* FRAMES. Every recycle's contact map and every denoise step's coordinates,
  as they are produced -- `return_trajectory` only yields them after the whole
  scan has finished, which is a replay.
* STEERING. `on_step(i, positions, sigma) -> positions` is denoised by the next
  step, so a restraint or a symmetry average is a few lines against the
  coordinates with no change to the graph and no recompile.
* PIECES. The stages are ordinary jax functions, so a caller can run only the
  trunk (the distogram is what most design objectives are built on), or
  differentiate one pass, without dragging the sampler and the confidence
  heads along.

WHY IT LIVES HERE and not in run_alphafold.py: that is a CLI script, and design
code cannot import a script. run_alphafold's live_model() is a thin wrapper
around this.

PREDICTION SEMANTICS, not bit-identical to the fused path. Each stage is its
own `apply`, so haiku's rng stream restarts per call where the fused graph
advances one stream throughout. The difference is a different DRAW from the
same distribution -- fused across seeds reads ptm 0.62-0.68 and 0.47-1.35 A
pairwise, and staged reads ptm 0.62 at 1.52 A. Parity gates must compare within
a mode, never across.

Cost, measured on a T4 (openbind0, 59 residues, 4 passes, 8 steps): warm 1.7 s
against the fused path's 1.65 s, i.e. +4%. Cold it pays one trace per stage.
"""

import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.model import model


def _preinit(config):
  """Enter tokamax's autotuning context before anything is traced.

  A lazy user context retraces the whole model on the second call; see
  run_alphafold._preinit_tokamax_context, which this mirrors.
  """
  try:
    import tokamax
    from tokamax._src.ops.experimental import dot_product_attention as _op  # noqa
  except Exception:
    return


def make_stages(config, *, jit=True, use_dropout=False):
  """The stages as plain callables, for code that wants pieces not a fold.

  Returns {'embed', 'trunk', 'diff_cond', 'score'}, each taking
  (params, rng, batch, carry=None, key=None, diffusion_state=None) like any
  haiku apply.

  `trunk` is ONE recycle pass and returns (embeddings, key, contact_probs), so
  a design objective built on the distogram can run the trunk alone -- no
  sampler, no confidence heads. Pass jit=False to compose or differentiate it
  yourself; a single pass is reverse-differentiable, which a fori_loop with a
  dynamic trip count would not be.
  """
  def _stage(name):
    @hk.transform
    def fn(batch, carry=None, key=None, diffusion_state=None):
      return model.Model(config)(
          batch, key=key, use_dropout=use_dropout,
          recycle_carry=carry, stage=name, diffusion_state=diffusion_state)
    return fn.apply if not jit else jax.jit(fn.apply)

  return {name: _stage(name)
          for name in ('embed', 'trunk', 'diff_cond', 'score')}


def staged_fold(config, params, *, jit=True, chunk=10,
                use_dropout=False, on_trunk=None):
  """Recycles AND diffusion steps driven from Python, one frame at a time.

  Returns `run(rng_key, batch, on_frame=None)`. `on_frame(kind, index, data)`
  is called after every trunk pass (kind='recycle', data=embeddings) and after
  every denoise step (kind='diffusion', data=atom_positions), so a notebook
  can draw the structure as it emerges rather than after it is finished.
  `return_trajectory` cannot do this: it yields its frames only once the whole
  scan has run.

  Compiled once each: trunk pass, conditioning, denoise step, scoring. Neither
  the recycle count nor the step count reaches a graph, so changing either
  costs nothing and one compile cache covers both.
  """
  def _stage(name):
    @hk.transform
    def fn(batch, carry=None, key=None, diffusion_state=None):
      return model.Model(config)(
          batch, key=key, use_dropout=use_dropout,
          recycle_carry=carry, stage=name, diffusion_state=diffusion_state)
    return fn.apply if (not jit) else jax.jit(fn.apply)

  embed, trunk = _stage('embed'), _stage('trunk')
  cond, score = _stage('diff_cond'), _stage('score')

  def _split_static(tree):
    """Arrays cross the jit boundary; everything else is closed over.

    The atom conditioning carries Python flags (swa_rope) that drive control
    flow inside the encoder, so passing them as traced values fails with
    TracerBoolConversionError. They are constant for a whole fold, so they
    belong in the closure, not in the signature.
    """
    arrays, static = {}, {}
    for k, v in tree.items():
      # A 0-d BOOL is a flag, not data -- swa_rope is a numpy bool scalar, so
      # `hasattr(v, 'shape')` filed it as an array and it came back a tracer.
      is_flag = isinstance(v, bool) or (
          np.ndim(v) == 0 and np.asarray(v).dtype == np.bool_)
      (static if is_flag or not hasattr(v, 'shape') else arrays)[k] = v
    return arrays, static

  # MEMOISED. Building this inside `run` re-traced the whole score model on
  # every fold: measured 4708 ms on the first chunk against 118 ms on the
  # next (10 steps, i.e. 11.8 ms a step against fused's 11.7). The per-step
  # cost was never the problem -- the trace was, and it is the only reason
  # live mode is slower at all.
  stage_cache = {}

  def _denoise_stage(static_atom_cond):
    # 0-d jax arrays are not hashable, so the key holds their VALUES --
    # `num_tokens` arrives as an ArrayImpl and raised
    # "unhashable type: ArrayImpl" straight out of the dict lookup.
    def _hashable(v):
      return v.item() if hasattr(v, 'item') else v
    ckey = tuple(sorted((k, _hashable(v)) for k, v in static_atom_cond.items()
                        if np.ndim(v) == 0))
    if ckey in stage_cache:
      return stage_cache[ckey]

    @hk.transform
    def fn(batch, carry, key, dcarry, noise_level, pair_cond, atom_arrays):
      return model.Model(config)(
          batch, key=key, use_dropout=use_dropout,
          recycle_carry=carry, stage='denoise',
          diffusion_state=(dcarry, noise_level, pair_cond,
                           {**atom_arrays, **static_atom_cond}))
    out = fn.apply if (not jit) else jax.jit(fn.apply)
    stage_cache[ckey] = out
    return out
  n = model.num_trunk_passes(config.num_recycles,
                             config.global_config.model)
  _preinit(config)   # tokamax context, before anything traces
  pass  # params are an argument here

  def run(rng_key, batch, on_frame=None, on_step=None):
    # The initial carry comes from its own cheap stage, so all n trunk calls
    # share ONE jit signature. With carry=None on the first call and a dict
    # on the rest, one computation compiled twice: 22.5 s then 20.2 s on a
    # cold T4. key=None here so the first key is drawn with hk.next_rng_key()
    # off the same haiku rng the fused path uses.
    carry, key = embed(params, rng_key, batch, None, None)
    for i in range(n):
      carry, key, contacts = trunk(params, rng_key, batch, carry, key)
      if on_frame:
        on_frame('recycle', i, {'embeddings': carry, 'contacts': contacts})

    st = cond(params, rng_key, batch, carry, key)
    dcarry, levels = st['init'], st['noise_levels']
    atom_arrays, atom_static = _split_static(st['atom_cond'])
    step = _denoise_stage(atom_static)
    # CHUNKED. `_LIVE_CHUNK` steps go in one dispatch and every frame comes
    # back, so the animation is unchanged and the per-call overhead is paid
    # once per chunk instead of once per step. The tail chunk is a second
    # (and last) shape, so at most two executables.
    # Steering can only act where the host sees the coordinates, so a chunk
    # of k steps means steering every k steps. With a steer callback the
    # chunk drops to 1; the cost of that is now 8% rather than the 14x it was
    # before the trace was memoised.
    k = 1 if on_step else max(1, chunk)
    todo = levels[1:]
    t = 0
    while t < len(todo):
      window = todo[t:t + k]   # NOT `chunk`: that is the parameter, and
                              # assigning it here made it local and unbound
      out = step(params, rng_key, batch, carry, key, dcarry,
                 window if len(window) > 1 else window[0],
                 st['pair_cond'], atom_arrays)
      dcarry = out['carry']
      if on_step is not None:
        # STEERING. The callback sees the coordinates this step produced and
        # returns them changed, or None to leave them alone; what it returns
        # is what the next step denoises. This is the seam a restraint, a
        # symmetry average or a pull towards a pocket goes through, and none
        # of it touches the graph -- which is the argument for driving the
        # loop from Python even where the animation is not wanted.
        #
        # `sigma` is passed because guidance strength is normally scaled by
        # the noise level: the same nudge is a large move early and a
        # distortion late.
        steered = on_step(t, dcarry[1], float(window[-1]))
        if steered is not None:
          steered = jnp.asarray(steered, dcarry[1].dtype)
          if steered.shape != dcarry[1].shape:
            raise ValueError(
                f'on_step returned {steered.shape}, expected '
                f'{dcarry[1].shape} (num_samples, num_tokens, max_atoms, 3)')
          dcarry = (dcarry[0], steered, dcarry[2])
      if on_frame:
        # The DENOISER'S PREDICTION, not the noisy state it hands on. See
        # make_denoising_body: the state is a cloud early (830 A radius of
        # gyration at step 0) where the prediction is already a structure.
        frames = out.get('denoised', out['atom_positions'])
        if len(window) > 1:
          for j in range(len(window)):
            on_frame('diffusion', t + j, frames[j])
        else:
          on_frame('diffusion', t, frames)
      t += len(window)
    return score(params, rng_key, batch, carry, key, (dcarry[1],))

  return run
