"""Dump RoseTTAFold3's own featuriser output, for featurisation_diff.py.

rf3's inference engine builds an atomworks Transform pipeline from
`build_af3_transform_pipeline` and calls it on `InferenceInput.to_pipeline_input()`.
Both are plain constructors, so the featurisation runs without the engine, the
checkpoint or hydra.
"""
import os
import sys

import numpy as np
import torch

for p in ('/home/ubuntu/rf3_extra', '/home/ubuntu/foundry_rf3/src',
          '/home/ubuntu/foundry_rf3/models/rf3/src'):
  sys.path.insert(0, p)
from rf3.data.pipelines import build_af3_transform_pipeline
from rf3.utils.inference import InferenceInput

SEQ = os.environ.get('SEQ', 'GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE')
OUT = sys.argv[1]

spec = InferenceInput.from_json_dict(
    {'name': 'feat_dump', 'components': [{'seq': SEQ, 'chain_id': 'A'}]})
pipeline = build_af3_transform_pipeline(
    is_inference=True, protein_msa_dirs=[], rna_msa_dirs=[], n_recycles=1,
    residue_cache_dir=None)
out = pipeline(spec.to_pipeline_input())


def flatten(d, pre=''):
  flat = {}
  for k, v in d.items():
    key = '%s%s' % (pre, k)
    if isinstance(v, dict):
      flat.update(flatten(v, key + '.'))
    elif torch.is_tensor(v):
      flat[key] = v.detach().cpu().numpy()
    elif isinstance(v, np.ndarray) and v.dtype.kind in 'fiub':
      flat[key] = v
  return flat


feats = flatten(out if isinstance(out, dict) else {'out': out})
np.savez(OUT, **feats)
print('rosettafold3 featuriser: %d numeric fields' % len(feats))
for k in sorted(feats):
  print('   %-42s %s' % (k, tuple(np.shape(feats[k]))))
