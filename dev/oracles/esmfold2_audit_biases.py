"""Which of ESMFold2's trained LayerNorm BIASES never reach the graph?

A scope diff cannot see this: the graph simply does not create the parameter,
so the converter has no slot to fill and nothing is reported missing. Rank
correlation is blind to a constant, so the numerical gates miss it too. The only
way to find it is to ask the CHECKPOINT what it has.
"""
import sys, numpy as np
sys.path.insert(0, '/home/ubuntu/alphafold3'); sys.path.insert(0, '/home/ubuntu/alphafold3/src')
from converters import esmfold2 as CV

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
sd = dict(np.load(S + 'esmfold2_sd.npz'))
flat = CV.map_esmfold2_to_af3_graph(sd)
have = [np.asarray(v).ravel() for v in flat.values()]

def present(a):
  a = np.asarray(a).ravel()
  for h in have:
    if h.size == a.size and np.array_equal(h, a):
      return True
    # stacked (layer_stack) parameters hold one row per block
    if h.size % a.size == 0 and h.size > a.size:
      r = h.reshape(-1, a.size)
      if (r == a).all(1).any():
        return True
  return False

missing = [k for k, v in sorted(sd.items())
           if k.endswith('.bias') and v.ndim == 1
           and k[:-len('.bias')] + '.weight' in sd
           and sd[k[:-len('.bias')] + '.weight'].shape == v.shape
           and not present(v)]
print('%d LayerNorm-shaped biases in the checkpoint never reach the graph:' % len(missing))
for k in missing:
  print('   %-70s %s' % (k, sd[k].shape))
