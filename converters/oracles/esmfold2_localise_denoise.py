"""One denoise step: the GRAPH's diffusion head against the reference's.

Both run on their own trunk (which agrees to corr 0.9887) but on the SAME noisy
coordinates and the same t_hat, so a divergence here is the score network, not
the sampler and not the trunk.
"""
import sys, os, numpy as np, jax, jax.numpy as jnp, haiku as hk
sys.path.insert(0, '/home/ubuntu/alphafold3'); sys.path.insert(0, '/home/ubuntu/alphafold3/src')
sys.argv = sys.argv[:1]
from alphafold3.model import model as af3_model, model_registry, params as afp
from alphafold3.model.components import utils
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model.pipeline import model_features
from alphafold3.model import feat_batch
from alphafold3.model.network import evoformer as ev, diffusion_head as dh
from converters.oracles.fold_check import parse_ca
from converters import esmfold2 as CV
from converters.oracles import esmfold2_reference as R

T_HAT = 8.0
seq, _ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
d = os.path.expanduser('~/ported/esmfold2')
spec = model_registry.get('esmfold2')
fi = folding_input.Input(name='x', chains=[folding_input.ProteinChain(
    id='A', sequence=seq, ptms=[], unpaired_msa='', paired_msa='', templates=[])], rng_seeds=[0])
ccd = decoded_ccd.get_ccd()
feat = lambda **kw: featurisation.featurise_input(fold_input=fi, ccd=ccd, buckets=None, **kw)
batch = feat()[0]
if spec.featurise:
    batch = model_features.apply(batch, spec, refeaturise=feat, model_dir=d, esm=None,
                                 has_msa=False, fold_input=fi)
cfg = af3_model.Model.Config(); cfg.global_config.flash_attention_implementation = 'xla'
cfg.global_config.bfloat16 = 'none'; spec.configure(cfg)

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
sd = dict(np.load(S + 'esmfold2_sd.npz')); dims = CV.derive_dims(sd); dims['n_input_atom'] = 3
dd = dict(np.load(S + 'esmfold2_6mrr68.npz'))
f = {k[5:]: jnp.asarray(v[0]) for k, v in dd.items() if k.startswith('feat.')}
pref = {k: jnp.asarray(v) for k, v in CV.map_esmfold2_to_af3(sd).items()}
rmask = np.asarray(f['atom_attention_mask']).astype(bool)   # 576 slots, 573 real
x_real = np.asarray(jax.random.normal(jax.random.PRNGKey(7), (int(rmask.sum()), 3))) * T_HAT
x_flat = np.zeros(rmask.shape + (3,), np.float32); x_flat[rmask] = x_real

zr, s_in, _ = R.trunk(f, None, pref, dims, n_loops=3, key=jax.random.PRNGKey(0),
                      lm_dropout=0.0, msa=R.self_msa(f))
rp = jnp.asarray(R.rel_pos_features(
    f['residue_index'].astype(int), f['asym_id'].astype(int), f['sym_id'].astype(int),
    f['entity_id'].astype(int), f['token_index'].astype(int))) @ pref['rel_pos/weights']
x_ref = np.asarray(R.denoise(jnp.asarray(x_flat), T_HAT, f, s_in, zr, rp, pref, dims))[rmask]

N_PASSES = 4
b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
fb0 = feat_batch.Batch.from_data_dict(b)
gmask = np.asarray(fb0.predicted_structure_info.atom_mask).astype(bool)
dense = np.zeros(gmask.shape + (3,), np.float32)
n = min(int(gmask.sum()), len(x_real))
buf = np.zeros((int(gmask.sum()), 3), np.float32); buf[:n] = x_real[:n]
dense[gmask] = buf


@hk.transform
def fwd(bb, x_noisy):
    fb = feat_batch.Batch.from_data_dict(bb)
    L = fb.token_features.mask.shape[0]
    c = cfg.evoformer.pair_channel
    prev = {'pair': jnp.zeros((L, L, c), jnp.float32),
            'pair_pre_coda': jnp.zeros((L, L, c), jnp.float32),
            'single': jnp.zeros((L, cfg.evoformer.seq_channel), jnp.float32)}
    tf = af3_model.create_target_feat_embedding(
        batch=fb, config=cfg.evoformer, global_config=cfg.global_config)
    mod = ev.Evoformer(cfg.evoformer, cfg.global_config)
    for _ in range(N_PASSES):
        emb = mod(batch=fb, prev=prev, target_feat=tf, key=jax.random.PRNGKey(0))
        prev = {**prev, **{k: v.astype(jnp.float32) for k, v in emb.items() if k in prev}}
    emb = {k: v.astype(jnp.float32) for k, v in emb.items()}
    out = dh.DiffusionHead(cfg.heads.diffusion, cfg.global_config)(
        positions_noisy=x_noisy, noise_level=jnp.asarray(T_HAT),
        batch=fb, embeddings=emb, use_conditioning=True)
    return out


_p = afp.get_model_haiku_params(model_dir=d)
# Model builds the head inside a method, so its scope is `diffuser/~/diffusion_head`;
# calling the module directly here drops both the outer scope and the `~/`.
def _strip(k):
  if k.startswith('diffuser/'):
    k = k[len('diffuser/'):]
  return k[len('~/'):] if k.startswith('~/') else k
_p = {_strip(k): v for k, v in _p.items()}
out = np.asarray(fwd.apply(_p, jax.random.PRNGKey(0), b, jnp.asarray(dense)))
x_graph = out[gmask][:n]
a, c = x_graph.ravel(), x_ref[:n].ravel()
print('x_denoised  GRAPH vs REFERENCE   (t_hat = %.3g, %d atoms)' % (T_HAT, n))
print('   corr %.6f   rms diff %.4f A' % (np.corrcoef(a, c)[0, 1],
                                          np.sqrt(((x_graph - x_ref[:n]) ** 2).sum(-1).mean())))
print('   graph std %.4f   ref std %.4f' % (x_graph.std(), x_ref[:n].std()))
