"""6MRR folded entirely in JAX: our ESM-C -> our ESMFold2 -> coordinates."""
import sys, os, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
jax.config.update('jax_platform_name', 'cpu')            # 25 GB of ESM-C
from converters import esmfold2 as E, esmc as C
from converters.oracles import esmfold2_reference as R
from converters.esmfold2 import load_esmfold2_checkpoint

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
d = dict(np.load(S + 'esmfold2_dump.npz'))
f = {k[len('feat.'):]: jnp.asarray(v[0]) for k, v in d.items() if k.startswith('feat.')}

snap = os.path.expanduser('~/.cache/huggingface/hub/models--biohub--ESMC-6B/snapshots')
t0 = time.time()
sd_c = load_esmfold2_checkpoint(os.path.join(snap, os.listdir(snap)[0]))
dims_c = C.derive_dims(sd_c)
p_c = C.map_esmc_to_af3(sd_c, dims_c); del sd_c
ids = C.lm_input_ids(d['feat.input_ids'][0].astype(np.int64))
hs = R.esmc_hidden_states(jnp.asarray(ids), p_c, dims_c)[:, 1:-1, :].transpose(1, 0, 2)
hs.block_until_ready(); del p_c
print('ESM-C (JAX, cpu): %.0f s   %s' % (time.time() - t0, hs.shape))

sd = load_esmfold2_checkpoint(S + 'esmfold2_sd.npz')
dims = E.derive_dims(sd); dims['n_input_atom'] = 3
p = {k: jnp.asarray(v) for k, v in E.map_esmfold2_to_af3(sd).items()}

def ca(pdb):
    return np.array([[float(l[30+8*i:38+8*i]) for i in range(3)] for l in open(pdb)
                     if l.startswith('ATOM') and l[12:16].strip() == 'CA'])
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T @ b); dd = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt((((a @ (u @ np.diag([1,1,dd]) @ vt)) - b) ** 2).sum(1).mean()))
nat = ca('/home/ubuntu/6MRR.pdb'); rep = np.asarray(d['feat.distogram_atom_idx'][0], int)

for seed in (0, 1, 2):
    t0 = time.time()
    x = R.fold(f, hs, p, dims, seed=seed); x.block_until_ready()
    print('FULL JAX seed %d: %6.3f A  (%.0f s)' % (seed, rmsd(np.asarray(x)[rep], nat), time.time()-t0))
print()
print('native (real ESM-C, as shipped): 1.87 - 2.02 A')
