"""Feed OUR MSA module NATIVE's real embedded MSA and pair, compare the pair out.

Real-input version of the L1b MSA cell. The tensors come from dump_trunk.py's
`msa_in_m` / `msa_in_z` (native's own embedder, final recycle) and the target is
native's `msa_out`, which in of3 IS the new z (the module returns it).
"""
import os, sys
import numpy as np
sys.path.insert(0, 'dev/oracles')
import fold_check
import msa_parity as mp

model, npz = sys.argv[1], sys.argv[2]
d = np.load(npz)
m = d['msa_in_m'][0].astype(np.float32)          # (n_msa, n_tok, c_m)
z = d['msa_in_z'][0].astype(np.float32)
ref = d['msa_out'][0].astype(np.float32)
tok = d['batch_token_mask'][0].astype(np.float32)
n_msa, n_tok = m.shape[0], m.shape[1]
seq = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
_, cfg, model_dir = fold_check._fold_setup(model, seq, None)
print('%s: msa %s z %s (%d rows, %d tokens)' % (model, m.shape, z.shape, n_msa, n_tok))
got = np.asarray(mp.ours(model, cfg, model_dir, m, z, n_msa, n_tok,
                         msa_mask=np.ones((n_msa, n_tok), np.float32)))
a = got.astype(np.float64).ravel(); b = ref.astype(np.float64).ravel()
print('  msa -> pair corr %.9f  rms ours/nat %.5f  max|d| %.4f  rms(nat) %.3f'
      % (np.corrcoef(a, b)[0, 1], np.sqrt((a**2).mean()) / np.sqrt((b**2).mean()),
         np.abs(a - b).max(), np.sqrt((b**2).mean())))
