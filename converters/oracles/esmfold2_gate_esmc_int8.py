"""Gate the int8 ESM-C tower against native's saved hidden states, ON GPU.

The float32 tower never fit a 23 GB card, which is why the original gate set
jax_platform_name='cpu' -- a standing rule violation that only existed because
of the size. At int8 the blob is 5.46 GB and the forward runs one block at a
time, so this runs where everything else does.
"""
import os
import sys
import time

import numpy as np
import jax

sys.path.insert(0, '/home/ubuntu/alphafold3')
sys.path.insert(0, '/home/ubuntu/alphafold3/src')
sys.argv = sys.argv[:1]
assert jax.devices()[0].platform == 'gpu', 'this must run on the GPU'

from converters import esmc_embed as E
from converters.oracles.fold_check import parse_ca

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
seq, _ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
nat = np.load(S + 'esmfold2_6mrr68.npz')['lm_hidden'][0]      # (L, 81, D) torch, fp32

t0 = time.time()
ours = E.embed(seq)                                            # (L, 81, D)
print('int8 tower: %s in %.0f s' % ((ours.shape,), time.time() - t0))
print('native     : %s' % (nat.shape,))
for li in (0, 1, 40, 78, 79, 80):
  o, n = ours[:, li], nat[:, li]
  print('  layer %-3d relerr %.3e  corr %.8f'
        % (li, np.abs(o - n).max() / max(np.abs(n).max(), 1e-9),
           np.corrcoef(o.ravel(), n.ravel())[0, 1]))
print('ALL layers corr %.8f' % np.corrcoef(ours.ravel(), nat.ravel())[0, 1])

# What actually matters is not the hidden states but the pair representation the
# trunk sees: the learned softmax mix concentrates on the last layers, so an
# error there counts for more than one at layer 3.
from converters import esmfold2_lm
p = esmfold2_lm.load_params('~/ported/esmfold2')
zo = esmfold2_lm.shim(ours, p)
zn = esmfold2_lm.shim(nat, p)
print('lm_pair    corr %.8f   std %.4f vs %.4f'
      % (np.corrcoef(np.asarray(zo).ravel(), np.asarray(zn).ravel())[0, 1],
         np.asarray(zo).std(), np.asarray(zn).std()))
np.savez_compressed(S + '6mrr_lm_pair_int8.npz', lm_pair=np.asarray(zo, np.float32))
