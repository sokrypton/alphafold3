"""Does ESMFold2 design actually DESCEND -- and does closing the LM chain help?

Objective: mean entropy of the distogram. Lower = a sharper, more confident
contact map, the standard hallucination proxy available with structure=False.
Two runs from the SAME start: the LM in the gradient chain, and the LM frozen
at a constant lm_pair (what the library does today).
"""
import os, sys, time
sys.path.insert(0, '/home/ubuntu/alphafold3/src'); sys.path.insert(0, '/home/ubuntu/alphafold3')
import numpy as np, jax, jax.numpy as jnp, haiku as hk
from converters.pdb import parse_ca
from dev.oracles.fold_check import _fold_setup
from dev.oracles.grad_check import _blurred_one_hot
from alphafold3.model import esm, model as af3_model, params as afp
from alphafold3.model import model_registry as mr
from alphafold3.af2.common import residue_constants as af2rc

model = sys.argv[1] if len(sys.argv) > 1 else 'esmfold2_lm300m'
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 25
seq, _ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
md = os.path.expanduser('~/ported/%s' % model)
tower = mr.ESMFOLD2_VARIANTS[model]['esmc']
tp, dims = esm.load(None, 'esmc', tower)
shim_p = esm.load_shim_params(md, model)
vocab = np.asarray(tp['embed/weights']).shape[0]
M = np.zeros((20, vocab), np.float32)
for i, aa in enumerate(af2rc.restypes):
    M[i, int(np.asarray(esm.sequence_ids(aa, 'esmc'))[0])] = 1.0
idf = np.asarray(esm.lm_input_ids(esm.sequence_ids(seq, 'esmc')))
bos, eos = int(idf[0]), int(idf[-1])
Mj = jnp.asarray(M)

batch, cfg, _ = _fold_setup(model, seq)
cfg.num_recycles = 0; cfg.global_config.bfloat16 = 'none'; cfg.evoformer.num_msa = 1
w = afp.get_model_haiku_params(model_dir=md)

def rows_from(soft):
    r = jnp.matmul(soft, Mj, precision='highest')
    oh = lambda i: jax.nn.one_hot(jnp.asarray([i]), vocab, dtype=r.dtype)
    return jnp.concatenate([oh(bos), r, oh(eos)], axis=0)

# the frozen lm_pair the library builds today, from the WILD-TYPE sequence
frozen = jax.lax.stop_gradient(esm.lm_pair_in_graph(idf, tp, dims, shim_p))

def entropy(out):
    p = out['distogram']['contact_probs']
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    return -jnp.mean(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))

@hk.transform
def fwd(b, soft, chain):
    lm = esm.lm_pair_in_graph(rows_from(soft), tp, dims, shim_p) if chain else frozen
    return af3_model.Model(cfg)({**b, 'lm_pair': lm}, soft_seq=soft,
                                structure=False)

def run(chain):
    f = jax.jit(lambda s: entropy(fwd.apply(w, jax.random.PRNGKey(0), batch, s, chain)))
    gf = jax.jit(jax.value_and_grad(
        lambda s: entropy(fwd.apply(w, jax.random.PRNGKey(0), batch, s, chain))))
    x = jnp.asarray(_blurred_one_hot(seq, np.random.default_rng(0)))
    traj, t0 = [], time.time()
    for i in range(steps):
        v, g = gf(x)
        traj.append(float(v))
        x = jax.nn.softmax(jnp.log(jnp.clip(x, 1e-8, None)) - 3.0 * g / (jnp.linalg.norm(g) + 1e-9) * 10)
    print('DESIGN %s chain=%-5s start %.5f  end %.5f  best %.5f  drop %.1f%%  %.0fs'
          % (model, chain, traj[0], traj[-1], min(traj), 100 * (traj[0] - min(traj)) / abs(traj[0]),
             time.time() - t0))
    print('   traj', ' '.join('%.4f' % t for t in traj[::max(1, steps // 8)]))

run(True)
run(False)
