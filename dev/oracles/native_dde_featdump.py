"""Dump OpenDDE's own featuriser output, for dev/oracles/featurisation_diff.py.

Its InferenceDataset takes the whole config object, so this builds the default
inference config and overrides only the input json -- the same path
`opendde` inference itself runs, rather than a reimplementation.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/OpenDDE')
from opendde.data.inference.infer_dataloader import InferenceDataset

CASE = os.environ.get('JSON', '/home/ubuntu/alphafold3/dev/oracles/native_json/5k9p_plain.json')
OUT = sys.argv[1]

# opendde's own config builder, driven exactly as its CLI does -- the overrides
# are passed as an arg string so the typed OpenDDEConfig view is built the same
# way inference builds it.
from opendde.config.inference import build_inference_config

cfg = build_inference_config(
    arg_str='--input_json_path %s --dump_dir /tmp/dde_dump --use_msa false '
            '--use_template false' % CASE)
ds = InferenceDataset(cfg)
data, atom_array, err = ds[0]
if err:
  raise SystemExit(err[:2000])
# the features live under 'input_feature_dict'; the top level carries only the
# N_* counts and the entity map
feats = data.get('input_feature_dict', data)
out = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
       for k, v in feats.items()
       if torch.is_tensor(v) or isinstance(v, np.ndarray)}
np.savez(OUT, **out)
print('opendde featuriser: %d tensor fields' % len(out))
for k in sorted(out):
  print('   %-30s %s' % (k, tuple(np.shape(out[k]))))
