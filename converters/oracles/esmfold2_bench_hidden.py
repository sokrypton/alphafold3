"""Pass 1 (GPU): ESM-C hidden states for one scheme, all targets -> disk."""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys, gc, json, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/home/ubuntu/alphafold3')
from converters import esmc as C
from converters.oracles import esmfold2_reference as R
from converters.esmfold2 import load_esmfold2_checkpoint
assert jax.devices()[0].platform == 'gpu'
tag, bits, group = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def rtn(w, bits, group):
    if w.ndim != 2 or bits >= 16: return w
    I, O = w.shape
    g = min(group, I)
    if I % g: return w
    x = w.reshape(I // g, g, O).astype(np.float32)
    n = 2 ** bits - 1
    mn = x.min(1, keepdims=True); mx = x.max(1, keepdims=True)
    s = np.maximum((mx - mn) / n, 1e-12)
    return (np.clip(np.round((x - mn) / s), 0, n) * s + mn).reshape(I, O)

meta = json.load(open('bench_meta.json'))
tg = dict(np.load('bench_targets.npz'))
snap = os.path.expanduser('~/.cache/huggingface/hub/models--biohub--ESMC-6B/snapshots')
sd = load_esmfold2_checkpoint(os.path.join(snap, os.listdir(snap)[0]))
dims = C.derive_dims(sd)
p_host = C.map_esmc_to_af3(sd, dims); del sd; gc.collect()
p_dev = {}
for k, v in p_host.items():
    if bits < 16 and k.startswith('blocks/') and k.endswith('/weights') and v.ndim == 3:
        v = np.stack([rtn(v[i], bits, group) for i in range(v.shape[0])])
    p_dev[k] = jnp.asarray(v, jnp.bfloat16 if v.ndim >= 2 else jnp.float32)
del p_host; gc.collect()
out = {}
for n in meta:
    ids = jnp.asarray(C.lm_input_ids(tg['%s.feat.input_ids' % n][0].astype(np.int64)))
    hs = R.esmc_hidden_states(ids, p_dev, dims)
    out[n] = np.asarray(jax.device_get(hs), np.float16)[:, 1:-1, :].transpose(1, 0, 2)
np.savez('hidden_%s.npz' % tag, **out)
print("%s: wrote %d targets" % (tag, len(out)), flush=True)
