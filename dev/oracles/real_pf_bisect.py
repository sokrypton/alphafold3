"""Which pairformer BLOCK does our openbind0 start diverging at?

Runs our stack truncated to k blocks on native's real pf_in, against native's
own output after block k-1 (`pf_blk<k-1>_z`).
"""
import sys
import numpy as np
sys.path.insert(0, 'dev/oracles')
import trunk_parity as tp

model, npz = sys.argv[1], sys.argv[2]
ks = [int(x) for x in sys.argv[3].split(',')]
d = np.load(npz)
s = d['pf_in_s'][0].astype(np.float32)
z = d['pf_in_z'][0].astype(np.float32)
tok = d['batch_token_mask'][0].astype(np.float32)
mask = tok[:, None] * tok[None, :]
for k in ks:
  out_s, out_z = tp.ours(model, s, z, mask, k)
  for tag, ours, nat in (('z', out_z, d['pf_blk%d_z' % (k - 1)][0]),
                         ('s', out_s, d['pf_blk%d_s' % (k - 1)][0])):
    a = np.asarray(ours, np.float64).ravel()
    b = np.asarray(nat, np.float64).ravel()
    print('  blocks=%-3d %s corr %.9f  rms ours/nat %.5f  max|d| %.4f  rms(nat) %.3f'
          % (k, tag, np.corrcoef(a, b)[0, 1],
             np.sqrt((a**2).mean()) / np.sqrt((b**2).mean()),
             np.abs(a - b).max(), np.sqrt((b**2).mean())), flush=True)
