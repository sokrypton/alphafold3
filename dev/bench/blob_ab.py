import os, sys, numpy as np, jax
sys.path.insert(0, '/home/ubuntu/alphafold3')
from dev.oracles.fold_check import fold, parse_ca, kabsch_rmsd
d = sys.argv[1]; name = sys.argv[2]
seq, native = parse_ca(os.path.expanduser('~/6MRR.pdb'))
outs = [fold(name, seq, model_dir=d, seed=s) for s in (0,)]
pos = np.concatenate([np.asarray(o['diffusion_samples']['atom_positions']) for o,_ in outs])
rs = [kabsch_rmsd(pos[i,:,1,:], native) for i in range(pos.shape[0])]
st = jax.devices()[0].memory_stats()
print('RESULT %s  rmsd %s  peak %.3f GB' % (
    d.split('/')[-2] if d.endswith('/') else ('OLD' if 'ported' in d else 'NEW'),
    ' '.join('%.3f' % r for r in rs), st['peak_bytes_in_use']/1e9))
