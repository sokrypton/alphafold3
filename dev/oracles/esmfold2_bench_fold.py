"""Pass 2 (GPU): fold every target from saved hidden states, 3 seeds."""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
from converters import esmfold2 as E
from converters.oracles import esmfold2_reference as R
assert jax.devices()[0].platform == 'gpu'
tag = sys.argv[1]
meta = json.load(open('bench_meta.json'))
tg = dict(np.load('bench_targets.npz'))
hid = dict(np.load('hidden_%s.npz' % tag))
sd = dict(np.load('esmfold2_sd.npz'))
dims = E.derive_dims(sd); dims['n_input_atom'] = 3
p = {k: jnp.asarray(v) for k, v in E.map_esmfold2_to_af3(sd).items()}
def rmsd(a, b):
    n = min(len(a), len(b)); a, b = a[:n]-a[:n].mean(0), b[:n]-b[:n].mean(0)
    u, _, vt = np.linalg.svd(a.T@b); s = np.sign(np.linalg.det(u@vt))
    return float(np.sqrt((((a@(u@np.diag([1,1,s])@vt))-b)**2).sum(1).mean()))
res = json.load(open('q_bench.json')) if os.path.exists('q_bench.json') else {}
for n in meta:
    f = {k[len('%s.feat.' % n):]: jnp.asarray(v[0]) for k, v in tg.items()
         if k.startswith('%s.feat.' % n)}
    hs = jnp.asarray(hid[n], jnp.float32)
    rs = []
    for seed in (0, 1, 2):
        x = R.fold(f, hs, p, dims, seed=seed, cfg={'steps': 200}); x.block_until_ready()
        rs.append(rmsd(np.asarray(x)[tg['%s.rep' % n]], tg['%s.native_ca' % n]))
    res['%s.%s' % (tag, n)] = rs
    print("  %-6s %-6s %s" % (tag, n, " ".join("%.3f" % v for v in rs)), flush=True)
json.dump(res, open('q_bench.json', 'w'), indent=1)
