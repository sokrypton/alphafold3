"""Does each int8 blob dequantise back to its fp32 parameters?

    PYTHONPATH=src:. python -m converters.int8_roundtrip_check protenix2

Run before publishing an int8 blob. A healthy model reports a worst relative
error of 0.0039 -- exactly 1/256, the int8 floor -- across every leaf.

READ THE ABSOLUTE ERROR, NOT JUST THE RELATIVE ONE. Some LayerNorm offsets in
these checkpoints hold denormal-scale values (protenix2 carries 4.9e-37,
opendde 2.5e-15). Quantising those to zero is correct, but dividing by that
scale reports a relative error of 1.0 and looks like a destroyed tensor. The
first run of this check raised exactly that false alarm on two of seven models.
Which is the same lesson as the distogram bias, pointing the other way: a
normalised statistic answers a different question than the one you asked.

Publishing a storage format nobody has read back is how a blob goes out broken.
This loads BOTH blobs through the real loader (which dequantises int8 records on
the way in) and compares every leaf.
"""
import sys, os, numpy as np
sys.path.insert(0, '/home/ubuntu/alphafold3/src'); sys.path.insert(0, '/home/ubuntu/alphafold3')
from alphafold3.model import params
S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad'
m = sys.argv[1]
fp32 = params.get_model_haiku_params(model_dir=os.path.expanduser(f'~/ported/{m}'))
q8 = params.get_model_haiku_params(model_dir=f'{S}/reported_int8/{m}')
missing = set(fp32) ^ set(q8)
worst, worst_name, worst_abs, n = 0.0, None, 0.0, 0
for scope in fp32:
  for name in fp32[scope]:
    a = np.asarray(fp32[scope][name], np.float32)
    b = np.asarray(q8[scope][name], np.float32)
    if a.shape != b.shape:
      print(f'  SHAPE MISMATCH {scope}/{name}: {a.shape} vs {b.shape}'); continue
    n += 1
    denom = np.abs(a).max()
    abs_err = float(np.abs(a - b).max())
    # Skip tensors that are numerically zero: a LayerNorm offset of 4.9e-37
    # quantises to 0.0, which is CORRECT, but dividing by that scale reports a
    # relative error of 1.0. Judge those by absolute error, like everything else
    # whose scale is meaningless.
    if denom < 1e-20:
      if abs_err > 1e-20:
        print(f'  near-zero tensor with real error: {scope}/{name} {abs_err:g}')
      continue
    rel = abs_err / denom
    if rel > worst:
      worst, worst_name, worst_abs = rel, f'{scope}/{name}', abs_err
print(f'{m:14s} leaves {n:4d}  scope-diff {len(missing)}  '
      f'worst rel {worst:.4f} (abs {worst_abs:.2e})  at {worst_name}')
