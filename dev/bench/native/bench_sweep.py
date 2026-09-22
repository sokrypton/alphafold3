"""Steady-state runtime of one ported model across LENGTH and MSA-ROW count.

  bench_sweep.py <model> <model_dir> <length> <num_msa> [calls]

Emits one TSV line: model, length, num_msa, tokens, compile, steady, all.

Two notes on what is being swept:

* LENGTH is the real scaling axis and is swept with a random sequence; only the
  shape matters for timing.
* NUM_MSA is the CONFIG's row count, not the input alignment's depth. AF3
  featurisation pads `msa` to 16384 rows whatever the input carries and the
  model truncates to config.evoformer.num_msa, so the input depth changes a
  scalar (`num_alignments`) and never a tensor shape -- sweeping it would give a
  flat line. The config knob is what costs time, and it is the one a user can
  actually turn down.
"""
import sys, os, time, functools, warnings, datetime, json, random
warnings.filterwarnings('ignore')
# AF3_SRC so this runs somewhere other than the machine it was written on. The
# default is the A10's checkout; the A100 has the package pip-installed and only
# run_alphafold.py on disk, and the hardcoded path failed there with a bare
# ModuleNotFoundError that named run_alphafold rather than the assumption.
_AF3 = os.environ.get('AF3_SRC', '/home/ubuntu/alphafold3')
sys.path.insert(0, os.path.join(_AF3, 'src'))
sys.path.insert(0, _AF3)
import numpy as np

model_name, model_dir = sys.argv[1], sys.argv[2]
length, num_msa = int(sys.argv[3]), int(sys.argv[4])
n_calls = int(sys.argv[5]) if len(sys.argv) > 5 else 3

import jax, jax.numpy as jnp, haiku as hk
# Both caches, the way run_alphafold.py wires them. Without these every process
# pays a full XLA compile (58-75 s) and, on a box where tokamax's Triton kernels
# are available, re-autotunes every kernel/shape pair from scratch. That was the
# whole of the A100 sweep's cost -- the package was already right, this harness
# was bypassing it.
_CACHE = os.environ.get('SWEEP_CACHE_DIR', '/tmp/alphafold_cache')
os.makedirs(os.path.join(_CACHE, 'jax'), exist_ok=True)
jax.config.update('jax_compilation_cache_dir', os.path.join(_CACHE, 'jax'))
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model import model, params, model_registry
from alphafold3.model.pipeline import model_features
from alphafold3.model.components import utils
import run_alphafold as RA

rng = random.Random(length)
SEQ = ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(length))
path = f'/tmp/_sweep_{length}.json'
json.dump({'name': f'L{length}',
           'sequences': [{'protein': {'id': 'A', 'sequence': SEQ,
                                      'unpairedMsa': '', 'pairedMsa': '',
                                      'templates': []}}],
           'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 4},
          open(path, 'w'))

fold_input = folding_input.load_fold_inputs_from_path(path).__next__()
ccd = decoded_ccd.get_ccd(user_ccd=fold_input.user_ccd)
spec = model_registry.get(model_name)
feat = functools.partial(
    featurisation.featurise_input, fold_input=fold_input, buckets=None, ccd=ccd,
    flatten_non_standard_residues=not spec.featurise.get('modified_as_one_token', False),
    ref_max_modified_date=datetime.date(2100, 1, 1))
batch = feat(verbose=False)[0]
if spec.featurise:
  batch = model_features.apply(batch, spec, refeaturise=lambda: feat(verbose=False),
                               model_dir=model_dir, esm=None, has_msa=False,
                               fold_input=fold_input, cyclic=False)

loaded = params.get_model_haiku_params(model_dir=model_dir)
# The shape-manifest fill is GONE. `params.read_shape_manifest` and
# `fill_from_manifest` were deleted with the manifests themselves (they never
# once fired), and this harness kept calling them -- so it raised AttributeError
# before measuring anything, exactly like `dev/audit_published.py` did. Nothing
# is lost: haiku fails loudly on a missing parameter, which is what the fill was
# insuring against.

cfg = RA.make_model_config(model_name=model_name,
                           num_recycles=int(os.environ.get('SWEEP_RECYCLES', '3')),
                           num_diffusion_samples=1,
                           flash_attention_implementation='xla')
cfg.evoformer.num_msa = num_msa
steps = int(os.environ.get('SWEEP_DIFF_STEPS', '0'))
if steps:
  cfg.heads.diffusion.eval.steps = steps

@hk.transform
def forward_fn(b):
  return model.Model(cfg)(b)
run = functools.partial(jax.jit(forward_fn.apply), loaded)

ex = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))

# tokamax autotuning, cached across processes and keyed per shape. On the A100
# the Triton kernels exist and autotuning them was showing up as a SECOND
# ~54 s "compile" on call 2 of every single run.
# NO explicit tokamax.autotune here, deliberately. The "Autotuning cache miss"
# warnings tokamax emits are NOT a cost: on a miss it logs and returns None
# (op.py:514-523), and the op falls back to its heuristics config. tokamax ships
# a precomputed cache keyed by device kind, so an unusual shape on an A100 simply
# misses it and runs on heuristics for free. Calling tokamax.autotune() instead
# does real search work for every unique shape -- which for a SWEEP, where each
# shape is visited once, is pure added cost with nothing to amortise it against.
# run_alphafold.py does call it, and there it pays off: a user folding many
# inputs of one size reuses the result through the on-disk cache.
# Create tokamax's JAX user context BEFORE tracing. Without this the model
# RETRACES on call 2 (the context is built lazily inside the first trace and
# JAX keys the jit cache on it), which cost ~45 s of every A100 measurement and
# is what made "steady state" read 27 s instead of 1.9 s. See
# run_alphafold.ModelRunner._preinit_tokamax_context.
try:
  from tokamax._src.ops import op as _tk_op
  _tk_op.get_autotuning_cache_overlay_state()
except Exception:
  pass

autotune = None

times = []
for i in range(n_calls):
  t0 = time.time()
  if autotune is not None:
    with autotune:
      out = run(jax.random.PRNGKey(i), ex)
  else:
    out = run(jax.random.PRNGKey(i), ex)
  jax.block_until_ready(out)
  times.append(time.time() - t0)

n_tok = int(np.asarray(batch['aatype']).reshape(-1).shape[0])
# Steady state = the LAST THIRD of the calls. On the A100 the second call
# recompiled (~54 s) as well as the first, so "everything after call 1" averaged
# a compile with a real call and reported 27 s where the truth was 1.9 s. Taking
# the tail is robust to however many warm-up compiles a box decides to do; the
# per-call list is printed so the shape of the warm-up stays visible.
tail = times[max(1, (2 * len(times)) // 3):] or times[-1:]
steady = float(np.median(tail))
# PEAK DEVICE MEMORY, from the allocator rather than nvidia-smi. nvidia-smi
# reports what the process RESERVED -- with JAX's default 75% pre-allocation
# that is a constant and says nothing -- while memory_stats()['peak_bytes_in_use']
# is what was actually live at the high-water mark. This matters more than the
# timings at large sizes: the interesting comparison at 768 tokens is not how
# much faster we are than native rosettafold3, it is that native OOMs there and
# we do not, and a claim like that needs a number attached.
peak = ''
try:
  st = jax.local_devices()[0].memory_stats() or {}
  pb = st.get('peak_bytes_in_use')
  if pb:
    peak = f'{pb / 2**30:.2f}'
except Exception:
  pass
print(f'{model_name}\t{length}\t{num_msa}\t{n_tok}\t{times[0]:.2f}\t{steady:.4f}\t'
      + ','.join(f'{t:.3f}' for t in times) + f'\t{peak}', flush=True)
