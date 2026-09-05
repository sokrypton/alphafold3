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
# LM=1 runs the ESM-C branch on BOTH sides. The dropout is switched off in both:
# it is resampled every pass from a key the two implementations do not share, so
# leaving it on would compare two different random maskings, not two trunks.
from alphafold3.model import model_config as _mc
USE_LM = os.environ.get('LM') == '1'
if USE_LM:
    _mc.LM_PAIR_DROPOUT['esmfold2'] = 0.0
else:
    cfg.evoformer.lm_encoder.num_layer = 0
# tap the TRUNK, not the distogram: the previous run compared distogram outputs,
# which include the head, so a broken head masqueraded as a broken trunk.
from alphafold3.model.network import evoformer as ev
from alphafold3.model import feat_batch
# the reference runs n_loops + 1 = 4 parcae passes; one pass with a zero carry
# is a DIFFERENT function, so the harness has to recycle too or the comparison
# measures the loop count rather than the trunk.
N_PASSES = 4

@hk.transform
def fwd(b):
    fb = feat_batch.Batch.from_data_dict(b)
    L = fb.token_features.mask.shape[0]
    c = cfg.evoformer.pair_channel
    prev = {'pair': jnp.zeros((L, L, c), jnp.float32),
            'pair_pre_coda': jnp.zeros((L, L, c), jnp.float32),
            'single': jnp.zeros((L, cfg.evoformer.seq_channel), jnp.float32)}
    tf = af3_model.create_target_feat_embedding(
        batch=fb, config=cfg.evoformer, global_config=cfg.global_config)
    # ONE module instance, called N times -- constructing it in the loop gives
    # `evoformer_1`, `evoformer_2`, ... each wanting its own weights.
    mod = ev.Evoformer(cfg.evoformer, cfg.global_config)
    for i in range(N_PASSES):
        emb = mod(batch=fb, prev=prev, target_feat=tf,
                  key=jax.random.PRNGKey(0))
        prev = {**prev, **{k: v.astype(jnp.float32)
                           for k, v in emb.items() if k in prev}}
    return emb
S0='/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
_dd = dict(np.load(S0 + 'esmfold2_6mrr68.npz'))
lm_hidden = jnp.asarray(_dd['lm_hidden'][0]) if USE_LM else None
if USE_LM:
    from converters import esmfold2_lm
    from alphafold3.model.pipeline import model_features as _mf
    _mf._attach_lm_pair(batch, esmfold2_lm.shim(
        np.asarray(lm_hidden), esmfold2_lm.load_params(d)))

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
z,_,_ = R.trunk(f, lm_hidden, pref, dims, n_loops=3, key=jax.random.PRNGKey(0),
                lm_dropout=0.0, msa=msa)
zr = np.asarray(z)
# stage-by-stage, pass 0: which of z_init / z_inject / z_parcae first diverges
gt = ev.ESM_TRUNK_TAPS
for name in ('z_pair0', 'z_relpos', 'z_init', 'z_inject', 'z_parcae'):
    if name in gt and name in R.TAPS:
        for i in range(min(len(gt[name]), len(R.TAPS[name]), N_PASSES)):
            x = np.asarray(gt[name][i]).ravel()
            y = np.asarray(R.TAPS[name][i]).ravel()
            print('   %-9s pass %d  corr %.6f  std %.4f vs %.4f'
                  % (name, i, np.corrcoef(x, y)[0, 1], x.std(), y.std()))

tf_g = np.asarray(g['target_feat']); tf_r = np.asarray(R.TAPS['s_inputs'][0])
print('   target_feat  %s vs %s' % (tf_g.shape, tf_r.shape))
# blockwise: [atom 384 | restype | profile | deletion]. The widths differ (ESM
# reserves two restype classes AF3 does not), so compare the atom block
# directly and the restype blocks after dropping ESM's leading two columns.
print('     atom        corr %.6f  std %.4f vs %.4f'
      % (np.corrcoef(tf_g[:, 63:].ravel(), tf_r[:, :384].ravel())[0, 1],
         tf_g[:, 63:].std(), tf_r[:, :384].std()))
gr, rr = tf_g[:, 0:31], tf_r[:, 384+2:384+33]
print('     restype     corr %.6f   equal %s' % (np.corrcoef(gr.ravel(), rr.ravel())[0, 1],
                                                 np.allclose(gr, rr)))
print('     graph restype idx', np.argmax(tf_g[:, 0:31], -1)[:24].tolist())
print('     ref   restype idx', (np.argmax(tf_r[:, 384:384+33], -1) - 2)[:24].tolist())
print('     ref   res_type   ', np.asarray(f['res_type']).astype(int)[:24].tolist())
gp, rp_ = tf_g[:, 31:62], tf_r[:, 417+2:417+33]
print('     profile     corr %.6f   equal %s' % (np.corrcoef(gp.ravel(), rp_.ravel())[0, 1],
                                                 np.allclose(gp, rp_)))

a, c = z_graph.ravel(), zr.ravel()
print('GRAPH trunk pair vs REFERENCE trunk pair (LM %s, self-MSA):'
      % ('ON' if USE_LM else 'off'))
print('   shapes %s vs %s' % (z_graph.shape, zr.shape))
print('   corr %.6f   relerr %.3e' % (np.corrcoef(a, c)[0,1],
                                      np.abs(a-c).max()/max(np.abs(c).max(),1e-9)))
print('   graph  std %.4f  absmax %.3f' % (z_graph.std(), np.abs(z_graph).max()))
print('   ref    std %.4f  absmax %.3f' % (zr.std(), np.abs(zr).max()))
