"""End-to-end: soft_seq -> ESM-C tower -> shim -> trunk -> distogram, one grad.

Also measures the peak with tower and trunk weights CO-RESIDENT, which is the
number that decides whether Model should call the tower itself.
"""
import os, sys, time
sys.path.insert(0, '/home/ubuntu/alphafold3/src'); sys.path.insert(0, '/home/ubuntu/alphafold3')
import numpy as np, jax, jax.numpy as jnp, haiku as hk
from converters.pdb import parse_ca
from dev.oracles.fold_check import _fold_setup
from dev.oracles.grad_check import _blurred_one_hot, _contact_loss
from alphafold3.model import esm, model as af3_model, params as afp
from alphafold3.model import model_registry as mr
from alphafold3.af2.common import residue_constants as af2rc

model = sys.argv[1] if len(sys.argv) > 1 else 'esmfold2_lm300m'
seq, _ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
md = os.path.expanduser('~/ported/%s' % model)
tower = mr.ESMFOLD2_VARIANTS[model]['esmc']
tp, dims = esm.load(None, 'esmc', tower)
shim_p = esm.load_shim_params(md, model)
vocab = np.asarray(tp['embed/weights']).shape[0]

# The adapter the library would need: a (20, vocab) one-hot map from the design
# alphabet to ESM-C's, so a DISTRIBUTION over amino acids becomes a distribution
# over tower tokens. Built through the one-letter codes, never by index.
M = np.zeros((20, vocab), np.float32)
for i, aa in enumerate(af2rc.restypes):
    M[i, int(np.asarray(esm.sequence_ids(aa, 'esmc'))[0])] = 1.0
ids_full = np.asarray(esm.lm_input_ids(esm.sequence_ids(seq, 'esmc')))
bos, eos = int(ids_full[0]), int(ids_full[-1])

batch, cfg, _ = _fold_setup(model, seq)
cfg.num_recycles = 0
cfg.global_config.bfloat16 = 'none'
cfg.evoformer.num_msa = 1
w = afp.get_model_haiku_params(model_dir=md)

@hk.transform
def fwd(b, soft):
    rows = jnp.matmul(soft, jnp.asarray(M), precision='highest')   # (L, vocab)
    onehot = lambda i: jax.nn.one_hot(jnp.asarray([i]), vocab, dtype=rows.dtype)
    rows = jnp.concatenate([onehot(bos), rows, onehot(eos)], axis=0)
    lm = esm.lm_pair_in_graph(rows, tp, dims, shim_p)
    # inject into the DICT: Model does Batch.from_data_dict internally and
    # feat_batch reads batch.get('lm_pair'). A _replace-with-hasattr-fallback
    # silently injected NOTHING here, so the tower was dead-code-eliminated.
    return af3_model.Model(cfg)({**b, 'lm_pair': lm}, soft_seq=soft,
                                structure=False)

x = jnp.asarray(_blurred_one_hot(seq, np.random.default_rng(0)))
t0 = time.time()
v, g = jax.block_until_ready(jax.value_and_grad(
    lambda s: _contact_loss(fwd.apply(w, jax.random.PRNGKey(0), batch, s)))(x))
print('E2E %s compile+run %.1fs  peak %.2f GB  loss %.6f  |g| %.4e  finite %s nonzero %d/%d'
      % (model, time.time() - t0,
         jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9,
         float(v), float(jnp.linalg.norm(g)), bool(np.isfinite(g).all()),
         int((np.asarray(g) != 0).sum()), g.size))
