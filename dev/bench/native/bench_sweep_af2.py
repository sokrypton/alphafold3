"""Steady-state runtime of AlphaFold 2 across LENGTH and MSA-ROW count.

  bench_sweep_af2.py <af2_ptm|af2_multimer> <params_dir> <length> <num_msa> [calls]

Emits the same TSV line shape as bench_sweep.py: model, length, num_msa,
tokens, compile, steady, all -- so the two sit in one table.

WHY A SECOND SCRIPT. `bench_sweep.py` builds `model.Model(cfg)`, the AF3 graph.
AlphaFold 2 is a SIBLING NETWORK in this library, not a model on that graph, so
it has no config to set `evoformer.num_msa` on and cannot be driven through the
same harness at all. This goes through `AF2ModelRunner`, which is the same entry
point `run_alphafold.py --model af2_ptm` uses.

TWO AXES THAT MEAN WHAT THEY SAY HERE, unlike the AF3 side. AF2's MSA sizes are
constructor arguments (`num_msa`, `num_extra_msa`) that set real tensor shapes,
so sweeping them costs real time -- there is no 16384-row padding and no config
truncation to see through. `num_extra_msa` is held at AF2's own 1024 and only the
cluster count is swept, because that is the knob a user turns.
"""
import sys, os, time, warnings, json, random, datetime
warnings.filterwarnings('ignore')
_AF3 = os.environ.get('AF3_SRC', '/home/ubuntu/alphafold3')
sys.path.insert(0, os.path.join(_AF3, 'src'))
sys.path.insert(0, _AF3)
import numpy as np

model_name, model_dir = sys.argv[1], sys.argv[2]
length, num_msa = int(sys.argv[3]), int(sys.argv[4])
n_calls = int(sys.argv[5]) if len(sys.argv) > 5 else 3

import jax
_CACHE = os.environ.get('SWEEP_CACHE_DIR', '/tmp/alphafold_cache')
os.makedirs(os.path.join(_CACHE, 'jax'), exist_ok=True)
jax.config.update('jax_compilation_cache_dir', os.path.join(_CACHE, 'jax'))
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)

from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model import model_registry
from alphafold3.af2 import inference as af2_inference

rng = random.Random(length)
SEQ = ''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(length))
path = f'/tmp/_sweep_af2_{length}.json'
json.dump({'name': f'L{length}',
           'sequences': [{'protein': {'id': 'A', 'sequence': SEQ,
                                      'unpairedMsa': f'>q\n{SEQ}\n',
                                      'pairedMsa': '', 'templates': []}}],
           'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 4},
          open(path, 'w'))
fold_input = folding_input.load_fold_inputs_from_path(path).__next__()
ccd = decoded_ccd.get_ccd(user_ccd=fold_input.user_ccd)
spec = model_registry.get(model_name)
batch = featurisation.featurise_input(
    fold_input=fold_input, buckets=None, ccd=ccd,
    ref_max_modified_date=datetime.date(2100, 1, 1), verbose=False)[0]

runner = af2_inference.AF2ModelRunner(
    spec, device=jax.local_devices()[0], model_dir=model_dir,
    num_recycles=int(os.environ.get('SWEEP_RECYCLES', '3')),
    num_msa=num_msa, num_extra_msa=1024)
_ = runner.model_params            # load eagerly, outside the timing

times = []
for i in range(n_calls):
  t0 = time.time()
  out = runner.forward(batch, key=jax.random.PRNGKey(i))
  jax.block_until_ready(out)
  times.append(time.time() - t0)

n_tok = int(np.asarray(batch['token_index']).shape[-1])
compile_s, steady = times[0], float(np.mean(times[1:])) if len(times) > 1 else times[0]
print('%s\t%d\t%d\t%d\t%.2f\t%.4f\t%s'
      % (model_name, length, num_msa, n_tok, compile_s, steady,
         ','.join('%.3f' % t for t in times)), flush=True)
