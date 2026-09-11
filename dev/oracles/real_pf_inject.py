"""Feed OUR trunk pairformer stack NATIVE's real (s, z) and compare its output.

The real-input version of the L1 trunk cell. L1 feeds the stack random tensors,
which for this family has no resolution at all (that cell grades FLOOR); here
the input is the one native actually used on ubiquitin, taken from
`dump_trunk.py`'s `pf_in_s` / `pf_in_z`, and the target is native's own
`z_trunk` / `s_trunk` out of the same call.
"""
import os, sys
import numpy as np
sys.path.insert(0, 'dev/oracles')
import trunk_parity as tp

model, npz = sys.argv[1], sys.argv[2]
d = np.load(npz)
s = d['pf_in_s'][0].astype(np.float32)
z = d['pf_in_z'][0].astype(np.float32)
tok = d['batch_token_mask'][0].astype(np.float32)
mask = tok[:, None] * tok[None, :]
n_blocks = int(os.environ.get('BLOCKS', 48))
print('%s: injecting native pf_in s%s z%s, %d blocks'
      % (model, s.shape, z.shape, n_blocks))
out_s, out_z = tp.ours(model, s, z, mask, n_blocks)


def cmp(tag, a, b):
  a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
  corr = np.corrcoef(a, b)[0, 1]
  ra, rb = np.sqrt((a**2).mean()), np.sqrt((b**2).mean())
  print('  %-10s corr %.8f  rms ours/native %.4f  max|d| %.5f  rms(nat) %.4f'
        % (tag, corr, ra / max(rb, 1e-9), np.abs(a - b).max(), rb))


cmp('pair', out_z, d['z_trunk'][0])
cmp('single', out_s, d['s_trunk'][0])
