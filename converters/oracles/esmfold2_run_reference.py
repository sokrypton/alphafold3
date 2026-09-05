import sys, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
from converters import esmfold2 as E
from converters.oracles import esmfold2_reference as R

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
sd = dict(np.load(S + 'esmfold2_sd.npz'))
dims = E.derive_dims(sd); dims['n_input_atom'] = 3
p = {k: jnp.asarray(v) for k, v in E.map_esmfold2_to_af3(sd).items()}
d = dict(np.load(S + 'esmfold2_dump.npz'))
f = {k[len('feat.'):]: jnp.asarray(v[0]) for k, v in d.items() if k.startswith('feat.')}
lm = jnp.asarray(np.load(S + 'esmfold2_det.npz')['lm_hidden'][0])
print('tokens %d  atoms %d' % (f['res_type'].shape[0], f['ref_pos'].shape[0]))

t0 = time.time()
x = R.fold(f, lm, p, dims, seed=0)
x.block_until_ready()
print('fold: %.1f s' % (time.time() - t0))

def ca(pdb):
    return np.array([[float(l[30+8*i:38+8*i]) for i in range(3)] for l in open(pdb)
                     if l.startswith('ATOM') and l[12:16].strip() == 'CA'])
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T @ b); dd = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt((((a @ (u @ np.diag([1,1,dd]) @ vt)) - b) ** 2).sum(1).mean()))

nat = ca('/home/ubuntu/6MRR.pdb')
rep = np.asarray(d['feat.distogram_atom_idx'][0], int)   # representative (CA) atom per token
ours = np.asarray(x)[rep]
print('OURS  vs native 6MRR structure : %.3f A' % rmsd(ours, nat))
nat_pred = np.asarray(d['out.sample_atom_coords'][0])[rep]
print('ESMFold2 native prediction     : %.3f A' % rmsd(nat_pred, nat))
print('OURS  vs ESMFold2 prediction   : %.3f A' % rmsd(ours, nat_pred))
