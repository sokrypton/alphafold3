"""Does on_step actually steer? Pull two distant residues together."""
import os, sys, time
sys.path.insert(0, '/home/ubuntu/alphafold3/src'); sys.path.insert(0, '/home/ubuntu/alphafold3')
sys.path.insert(0, '/home/ubuntu/alphafold3/dev/live')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')
from absl import flags
import numpy as np, jax
import run_alphafold as RA
SP = os.path.dirname(os.path.abspath(__file__))
flags.FLAGS(['run_alphafold.py', '--flash_attention_implementation=xla',
             f'--cache_dir={SP}/cache_cost', '--norun_data_pipeline'])
import live_frames as LF
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model import model_registry
from alphafold3.model.components import utils as _u
from alphafold3.model.pipeline import model_features
WD = os.path.expanduser('~/ported/openbind0')
fi = folding_input.load_fold_inputs_from_path(f'{SP}/dyn.json').__next__()
ccd = decoded_ccd.get_ccd()
feat = lambda: featurisation.featurise_input(fold_input=fi, ccd=ccd, buckets=None, verbose=False)
raw = feat()[0]
spec = model_registry.get('openbind0')
if spec.featurise:
    raw = model_features.apply(raw, spec, refeaturise=lambda: feat()[0], model_dir=WD,
                               esm=None, has_msa=True, fold_input=fi, cyclic=False)
bobj = LF.as_batch(raw)
batch = jax.device_put(jax.tree.map(jax.numpy.asarray, _u.remove_invalidly_typed_feats(raw)))
cfg = RA.make_model_config(model_name='openbind0', num_recycles=3,
                           num_diffusion_samples=1, flash_attention_implementation='xla')
cfg.heads.diffusion.eval.steps = 50
cfg.heads.diffusion.eval.stepwise = True
run = RA.ModelRunner(config=cfg, device=jax.local_devices()[0],
                     model_dir=WD).live_model()

A, B = 2, 29                      # two residues far apart in the free fold
idx, _ = LF.rep_atom_index(bobj)
def gap(res):
    p = np.asarray(res['diffusion_samples']['atom_positions'])[0]
    return float(np.linalg.norm(p[B, idx[B]] - p[A, idx[A]]))

t = time.time(); free = run(jax.random.PRNGKey(1), batch)
print(f'free           CA{A}-CA{B} = {gap(free):6.2f} A   ({time.time()-t:.1f}s)', flush=True)

# a GLOBAL steer for comparison: translate the two halves of the chain together
def halves(pull=1.0):
    def steer(i, positions, sigma):
        pos = np.array(positions); n = pos.shape[1] // 2
        c1 = pos[:, :n].reshape(pos.shape[0], -1, 3).mean(1)
        c2 = pos[:, n:].reshape(pos.shape[0], -1, 3).mean(1)
        unit = (c2 - c1) / np.maximum(np.linalg.norm(c2 - c1, axis=-1, keepdims=True), 1e-6)
        w = pull * min(1.0, sigma / 16.0)
        pos[:, :n] += (unit * w)[:, None, None, :]
        pos[:, n:] -= (unit * w)[:, None, None, :]
        return pos
    return steer

for target, strength in ((6.0, 1.0), (6.0, 0.3)):
    steer = LF.pull_together(A, B, target=target, strength=strength, batch=bobj)
    calls = [0]
    def wrapped(i, pos, sigma):
        calls[0] += 1
        return steer(i, pos, sigma)
    t = time.time(); out = run(jax.random.PRNGKey(1), batch, on_step=wrapped)
    print(f'target {target} strength {strength}: CA{A}-CA{B} = {gap(out):6.2f} A'
          f'   ({calls[0]} steer calls, {time.time()-t:.1f}s)', flush=True)

for pull in (0.5, 2.0):
    t = time.time(); out = run(jax.random.PRNGKey(1), batch, on_step=halves(pull))
    print(f'halves pull {pull}:            CA{A}-CA{B} = {gap(out):6.2f} A'
          f'   ({time.time()-t:.1f}s)', flush=True)
