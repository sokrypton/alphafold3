"""Is a fold number a property of the port, or a per-process draw?

    for i in 1 2 3 4; do PYTHONPATH=src:.:dev/oracles \
      python dev/oracles/fold_repeat_probe.py <model> [seq]; done

Prints a hash of the sampled coordinates, so two runs can be told apart without
a reference structure. plain 5K9P is BISTABLE for the protenix lineage: the same
command, same seed, same precision gives 11.362 in one process and 1.555 in the
next, each reproducible to three decimals within its own process. Three
attributions died on that (matmul precision, a vendor PYTHONPATH overlay, and
bf16-vs-fp32), so repeat the process before believing any of them.
"""

import sys, hashlib
import numpy as np
sys.path.insert(0, 'dev/oracles')
import fold_check

_SEQ = ('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKEST'
        'LHLVLRLRGG')   # plain ubiquitin, the bistable case
out, batch = fold_check.fold(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else _SEQ, seed=0)
pos = np.asarray(out['diffusion_samples']['atom_positions'])
print('pos hash', hashlib.sha256(np.ascontiguousarray(pos).tobytes()).hexdigest()[:16],
      'mean |x|', float(np.abs(pos).mean()))
plddt = np.asarray(out['distogram']['bin_edges']).mean() if 'distogram' in out else 0
conf = out.get('predicted_lddt', out.get('confidences', {}))
try:
    p = np.asarray(out['plddt']); print('mean pLDDT %.1f' % p.mean())
except Exception:
    ks = [k for k in out if 'lddt' in k or 'conf' in k]
    print('conf keys', ks[:4])
