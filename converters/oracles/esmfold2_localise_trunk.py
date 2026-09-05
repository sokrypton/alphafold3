"""Split 'trunk is wrong' from 'diffusion/featurisation is wrong'."""
import sys, os, numpy as np, jax, jax.numpy as jnp, haiku as hk
sys.path.insert(0,'/home/ubuntu/alphafold3'); sys.path.insert(0,'/home/ubuntu/alphafold3/src')
sys.argv=sys.argv[:1]
from alphafold3.model import model as af3_model, model_registry, params as afp
from alphafold3.model.components import utils
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model.pipeline import model_features
from converters.oracles.fold_check import parse_ca
from converters import esmfold2 as CV
from converters.oracles import esmfold2_reference as R

seq,_ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
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
cfg = af3_model.Model.Config(); cfg.global_config.flash_attention_implementation='xla'
cfg.global_config.bfloat16='none'; spec.configure(cfg)
# tap the TRUNK, not the distogram: the previous run compared distogram outputs,
# which include the head, so a broken head masqueraded as a broken trunk.
from alphafold3.model.network import evoformer as ev
from alphafold3.model import feat_batch
@hk.transform
def fwd(b):
    fb = feat_batch.Batch.from_data_dict(b)
    emb = ev.Evoformer(cfg.evoformer, cfg.global_config)(
        batch=fb,
        prev={'pair': jnp.zeros((fb.token_features.mask.shape[0],)*2 + (cfg.evoformer.pair_channel,), jnp.float32),
              'single': jnp.zeros((fb.token_features.mask.shape[0], cfg.evoformer.seq_channel), jnp.float32)},
        target_feat=af3_model.create_target_feat_embedding(
            batch=fb, config=cfg.evoformer, global_config=cfg.global_config),
        key=jax.random.PRNGKey(0))
    return emb
b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
# calling Evoformer directly drops the Model's own `diffuser/` scope prefix
_p = afp.get_model_haiku_params(model_dir=d)
_p = {(k[len('diffuser/'):] if k.startswith('diffuser/') else k): v for k, v in _p.items()}
g = fwd.apply(_p, jax.random.PRNGKey(0), b)
z_graph = np.asarray(g['pair'])

S='/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
sd=dict(np.load(S+'esmfold2_sd.npz')); dims=CV.derive_dims(sd); dims['n_input_atom']=3
dd=dict(np.load(S+'esmfold2_6mrr68.npz'))
f={k[5:]: jnp.asarray(v[0]) for k,v in dd.items() if k.startswith('feat.')}
pref={k: jnp.asarray(v) for k,v in CV.map_esmfold2_to_af3(sd).items()}
msa=R.self_msa(f)
z,_,_ = R.trunk(f, None, pref, dims, n_loops=3, key=jax.random.PRNGKey(0),
                lm_dropout=0.0, msa=msa)
zr = np.asarray(z)
a, c = z_graph.ravel(), zr.ravel()
print('GRAPH trunk pair vs REFERENCE trunk pair (both no-LM, self-MSA):')
print('   shapes %s vs %s' % (z_graph.shape, zr.shape))
print('   corr %.6f   relerr %.3e' % (np.corrcoef(a, c)[0,1],
                                      np.abs(a-c).max()/max(np.abs(c).max(),1e-9)))
print('   graph  std %.4f  absmax %.3f' % (z_graph.std(), np.abs(z_graph).max()))
print('   ref    std %.4f  absmax %.3f' % (zr.std(), np.abs(zr).max()))
