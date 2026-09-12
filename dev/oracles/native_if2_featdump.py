"""Dump IntelliFold-2's own featuriser output, for featurisation_diff.py.

Drives if2's real path: `check_inputs` + `process_inputs` (its preprocessing,
which writes a processed/ tree) then `get_inference_dataloader`, exactly as
run_intellifold.py does -- no reimplementation.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/IntelliFold')
from intellifold.data.inference.data_tools import check_inputs, process_inputs
from intellifold.data.module.inference import get_inference_dataloader
from intellifold.data.types import Manifest

YAML = os.environ.get('INPUT', '/home/ubuntu/if2_6mrr_native/6mrr.yaml')
OUT = sys.argv[1]
out_dir = Path(os.environ.get('OUTDIR', '/tmp/if2_featdump'))
out_dir.mkdir(parents=True, exist_ok=True)

data = check_inputs(Path(YAML))
args = SimpleNamespace(return_similar_seq=False, use_msa_server=False,
                       msa_server_url='', msa_pairing_strategy='greedy',
                       no_pairing=True, use_template=False, num_workers=0)
process_inputs(args, data=data, out_dir=out_dir,
               ccd_path=Path(os.environ.get('CCD', '/home/ubuntu/if2_cache/ccd_v2.pkl')),
               use_msa_server=False, msa_server_url='',
               msa_pairing_strategy='greedy', max_msa_seqs=16384,
               use_pairing=False, use_template=False)
proc = out_dir / 'processed'
loader = get_inference_dataloader(
    args,
    manifest=Manifest.load(proc / 'manifest.json'),
    target_dir=proc / 'structures', msa_dir=proc / 'msa',
    constraints_dir=(proc / 'constraints') if (proc / 'constraints').exists() else None,
)
batch = next(iter(loader))
out = {k: v.detach().cpu().numpy() for k, v in batch.items() if torch.is_tensor(v)}
np.savez(OUT, **out)
print('intellifold2 featuriser: %d tensor fields' % len(out))
for k in sorted(out):
  print('   %-30s %s' % (k, tuple(out[k].shape)))
