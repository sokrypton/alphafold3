"""Drive recycles AND diffusion steps from Python, printing every frame."""
import os, sys, time
sys.path.insert(0, '/home/ubuntu/alphafold3/src')
sys.path.insert(0, '/home/ubuntu/alphafold3')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')
from absl import flags
import numpy as np, jax
import run_alphafold as RA

SP = os.path.dirname(os.path.abspath(__file__))
flags.FLAGS(['run_alphafold.py', '--flash_attention_implementation=xla',
             f'--cache_dir={SP}/cache_live', '--norun_data_pipeline'])

from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model import model_registry

fi = folding_input.load_fold_inputs_from_path(f'{SP}/dyn.json').__next__()
cfg = RA.make_model_config(model_name='openbind0', num_recycles=3,
                           num_diffusion_samples=1,
                           flash_attention_implementation='xla')
cfg.heads.diffusion.eval.steps = 20          # short, so the demo is quick
cfg.heads.diffusion.eval.stepwise = True     # hand back cond + init
runner = RA.ModelRunner(config=cfg, device=jax.local_devices()[0],
                        model_dir=os.path.expanduser('~/ported/openbind0'))
ccd = decoded_ccd.get_ccd()
batch = featurisation.featurise_input(fold_input=fi, ccd=ccd, buckets=None,
                                      verbose=False)[0]
from alphafold3.model.pipeline import model_features
spec = model_registry.get('openbind0')
if spec.featurise:
    batch = model_features.apply(
        batch, spec, refeaturise=lambda: featurisation.featurise_input(
            fold_input=fi, ccd=ccd, buckets=None, verbose=False),
        model_dir=os.path.expanduser('~/ported/openbind0'), esm=None,
        has_msa=True, fold_input=fi, cyclic=False)
from alphafold3.model.components import utils as m_utils
batch = jax.device_put(jax.tree.map(
    __import__('jax').numpy.asarray,
    m_utils.remove_invalidly_typed_feats(batch)))

frames = []
t0 = time.time()
def on_frame(kind, i, data):
    if kind == 'recycle':
        # a cheap scalar per pass, to show the trunk converging
        print(f'  [{time.time()-t0:5.1f}s] recycle {i}: '
              f'|pair| {float(np.abs(np.asarray(data["pair"])).mean()):.4f}', flush=True)
    else:
        xyz = np.asarray(data)[0]
        frames.append(xyz)
        if i % 4 == 0 or i >= 18:
            print(f'  [{time.time()-t0:5.1f}s] diffusion step {i}: '
                  f'radius of gyration {np.sqrt(((xyz - xyz.mean(0))**2).sum(-1).mean()):.2f} A',
                  flush=True)

run = runner.live_model()
result = run(jax.random.PRNGKey(1), batch, on_frame=on_frame)
print(f'LIVE DONE in {time.time()-t0:.1f}s, {len(frames)} diffusion frames')
print('result keys:', sorted(k for k in result)[:8])
