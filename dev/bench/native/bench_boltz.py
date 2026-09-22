"""Native Boltz-2 forward time on GPU, matched to bench_ours.py.

Warmup call first: torch pays cudnn autotune + allocator growth on call 1, which is the
eager analogue of our XLA compile and does not belong in a steady-state number.
"""
import sys, time
from pathlib import Path
from dataclasses import asdict

import torch

sys.path.insert(0, '/home/ubuntu/BoltzDesign1/boltz2/src')
from boltz.data.types import Manifest
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.model.models.boltz2 import Boltz2
from boltz.main import (Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs,
                        BoltzSteeringParams)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RECYCLES = int(sys.argv[3]) if len(sys.argv) > 3 else 0
USE_KERNELS = (sys.argv[2] if len(sys.argv) > 2 else 'on') == 'on'
REPS = 3
proc = Path(sys.argv[4] if len(sys.argv) > 4
            else '/home/ubuntu/boltz2_6mrr/out/boltz_results_6mrr/processed')

dm = Boltz2InferenceDataModule(
    manifest=Manifest.load(proc / 'manifest.json'), target_dir=proc / 'structures',
    msa_dir=proc / 'msa', mol_dir=Path('/home/ubuntu/.boltz/mols'), num_workers=0,
    constraints_dir=proc / 'constraints', template_dir=proc / 'templates',
    extra_mols_dir=proc / 'mols', override_method=None)
dm.setup('predict')
batch = next(iter(dm.predict_dataloader()))
batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

st = BoltzSteeringParams(); st.fk_steering = False; st.physical_guidance_update = False
model = Boltz2.load_from_checkpoint(
    '/home/ubuntu/boltz2_weights/boltz2_conf.ckpt', strict=True, map_location='cpu',
    predict_args={'recycling_steps': RECYCLES, 'sampling_steps': STEPS, 'diffusion_samples': 1,
                  'max_parallel_samples': 1, 'write_confidence_summary': False,
                  'write_full_pae': False, 'write_full_pde': False},
    diffusion_process_args=asdict(Boltz2DiffusionParams()), ema=False,
    use_kernels=USE_KERNELS, pairformer_args=asdict(PairformerArgsV2()),
    msa_args=asdict(MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024,
                                  use_paired_feature=True)),
    steering_args=asdict(st))
model = model.eval().cuda()

n_tok = int(batch['token_pad_mask'].sum().item())
def run():
  with torch.no_grad():
    return model(batch, recycling_steps=RECYCLES, num_sampling_steps=STEPS,
                 diffusion_samples=1, run_confidence_sequentially=True)

t0 = time.time(); run(); torch.cuda.synchronize(); first = time.time() - t0
ts = []
for _ in range(REPS):
  t0 = time.time(); run(); torch.cuda.synchronize(); ts.append(time.time() - t0)
print(f'NATIVE L={n_tok} recycles={RECYCLES} steps={STEPS} kernels={USE_KERNELS}: '
      f'first {first:6.2f}s (incl warmup) | steady {min(ts):6.3f}s '
      f'(median {sorted(ts)[len(ts)//2]:.3f}, n={REPS})')
