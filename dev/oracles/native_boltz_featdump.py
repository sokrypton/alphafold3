"""Dump BOLTZ-2's own featuriser output for the 6MRR run, for a field-by-field
diff against our batch. Every port bug found on 2026-09-11/12 was an INPUT
convention, and the gates feed native OUR features, so this is the direction
nothing checks."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/BoltzDesign1/boltz2/src')
from boltz.data.types import Manifest
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule

# PROC lets the ligand / two-chain cases run: point it at any tree written by
# boltz's own process_inputs (see the prep script beside this one).
import os
proc = Path(os.environ.get(
    'PROC', '/home/ubuntu/boltz2_6mrr/out/boltz_results_6mrr/processed'))
dm = Boltz2InferenceDataModule(
    manifest=Manifest.load(proc / 'manifest.json'), target_dir=proc / 'structures',
    msa_dir=proc / 'msa', mol_dir=Path('/home/ubuntu/.boltz/mols'), num_workers=0,
    constraints_dir=proc / 'constraints', template_dir=proc / 'templates',
    extra_mols_dir=proc / 'mols', override_method=None)
dm.setup('predict')
batch = next(iter(dm.predict_dataloader()))
out = {}
for k, v in batch.items():
    if torch.is_tensor(v):
        out[k] = v.detach().cpu().numpy()
np.savez(sys.argv[1], **out)
print('boltz2 featuriser: %d tensor fields' % len(out))
for k in sorted(out):
    v = out[k]
    u = np.unique(v)
    print('   %-32s %-22s %s' % (k, tuple(v.shape),
                                 ('uniq %s' % u[:4]) if u.size <= 6 else
                                 'range [%.3g, %.3g]' % (v.min(), v.max())))
