"""THE test of the ESM-C path: a NATURAL protein, where the LM is worth 6.8 A."""
import sys, os, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
jax.config.update('jax_platform_name', 'cpu')          # 25 GB of ESM-C
from converters import esmfold2 as E, esmc as C
from converters.oracles import esmfold2_reference as R
from converters.esmfold2 import load_esmfold2_checkpoint
S = os.path.dirname(os.path.abspath(__file__)) + '/'
d = dict(np.load(S + 'esmfold2_1ubq.npz'))
f = {k[len('feat.'):]: jnp.asarray(v[0]) for k, v in d.items() if k.startswith('feat.')}
nat = d['native_ca']; rep = np.load(S + 'ca_idx_ubq.npy')
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T@b); s = np.sign(np.linalg.det(u@vt))
    return float(np.sqrt((((a@(u@np.diag([1,1,s])@vt))-b)**2).sum(1).mean()))

# OUR ESM-C, from the token ids -- not native's hidden states
snap = os.path.expanduser('~/.cache/huggingface/hub/models--biohub--ESMC-6B/snapshots')
sd_c = load_esmfold2_checkpoint(os.path.join(snap, os.listdir(snap)[0]))
dims_c = C.derive_dims(sd_c); p_c = C.map_esmc_to_af3(sd_c, dims_c); del sd_c
ids = C.lm_input_ids(d['feat.input_ids'][0].astype(np.int64))
ours_hs = R.esmc_hidden_states(jnp.asarray(ids), p_c, dims_c)[:, 1:-1, :].transpose(1, 0, 2)
ours_hs.block_until_ready(); del p_c
nat_hs = d['lm_hidden'][0]
print('our ESM-C vs native hidden states: corr %.8f' % np.corrcoef(
    np.asarray(ours_hs).ravel(), nat_hs.ravel())[0, 1])

sd = dict(np.load(S + 'esmfold2_sd.npz'))
dims = E.derive_dims(sd); dims['n_input_atom'] = 3
p = {k: jnp.asarray(v) for k, v in E.map_esmfold2_to_af3(sd).items()}
print()
print('%-34s %s' % ('1UBQ (natural protein)', 'CA-RMSD'))
for tag, hs in [('OURS, our ESM-C', ours_hs), ('OURS, native ESM-C replayed', jnp.asarray(nat_hs))]:
    rs = []
    for seed in (0, 1, 2):
        x = R.fold(f, hs, p, dims, seed=seed, cfg={'steps': 200}); x.block_until_ready()
        rs.append(rmsd(np.asarray(x)[rep], nat))
    print('  %-32s %s   mean %.3f' % (tag, ' '.join('%.3f' % v for v in rs), np.mean(rs)))
print('  %-32s %.3f   (pLDDT %.3f)' % ('NATIVE', rmsd(d['out.sample_atom_coords'][0][rep], nat),
                                       d['out.plddt'].mean()))
print('  %-32s %.3f' % ('native WITHOUT ESM-C (reference)', 7.624))
