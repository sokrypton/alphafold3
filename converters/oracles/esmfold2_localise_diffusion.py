"""Trunk or diffusion? Fold with the GRAPH's trunk and the REFERENCE's sampler.

The trunk now reproduces the reference at corr 0.9887 but the model still folds
6MRR at 15 A, so the fault is downstream -- or the trunk's last 1% matters.
Feed the graph's own z and s_inputs into the reference's `sample` and see which:
a ~1.7 A fold means the trunk is fine and the graph's diffusion is wrong.
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
from alphafold3.model.network import evoformer as ev
from converters.oracles.fold_check import parse_ca
from converters import esmfold2 as CV
from converters.oracles import esmfold2_reference as R

seq, xyz_true = parse_ca(os.path.expanduser('~/6MRR.pdb'))
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
    mod = ev.Evoformer(cfg.evoformer, cfg.global_config)
    for _ in range(N_PASSES):
        emb = mod(batch=fb, prev=prev, target_feat=tf, key=jax.random.PRNGKey(0))
        prev = {**prev, **{k: v.astype(jnp.float32) for k, v in emb.items() if k in prev}}
    return emb


b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
_p = afp.get_model_haiku_params(model_dir=d)
_p = {(k[len('diffuser/'):] if k.startswith('diffuser/') else k): v for k, v in _p.items()}
g = fwd.apply(_p, jax.random.PRNGKey(0), b)

S = '/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/'
sd = dict(np.load(S + 'esmfold2_sd.npz')); dims = CV.derive_dims(sd); dims['n_input_atom'] = 3
dd = dict(np.load(S + 'esmfold2_6mrr68.npz'))
f = {k[5:]: jnp.asarray(v[0]) for k, v in dd.items() if k.startswith('feat.')}
pref = {k: jnp.asarray(v) for k, v in CV.map_esmfold2_to_af3(sd).items()}


def to_ref_s_inputs(tf, n_af3=31, off=CV.ESM_RESTYPE_OFFSET, n_esm=33):
  """AF3's 447 [restype|profile|deletion|atom] -> ESMFold2's 451 [atom|...]."""
  pad = lambda blk: jnp.concatenate(
      [jnp.zeros((blk.shape[0], off)), blk,
       jnp.zeros((blk.shape[0], n_esm - off - n_af3))], -1)
  rest, prof = tf[:, :n_af3], tf[:, n_af3:2 * n_af3]
  deln, atom = tf[:, 2 * n_af3:2 * n_af3 + 1], tf[:, 2 * n_af3 + 1:]
  return jnp.concatenate([atom, pad(rest), pad(prof), deln], -1)


rp = jnp.asarray(R.rel_pos_features(
    f['residue_index'].astype(int), f['asym_id'].astype(int), f['sym_id'].astype(int),
    f['entity_id'].astype(int), f['token_index'].astype(int))) @ pref['rel_pos/weights']


def ca_rmsd(pred):
  a = np.asarray(pred) - np.asarray(pred).mean(0)
  bb = np.asarray(xyz_true) - np.asarray(xyz_true).mean(0)
  u, _, vt = np.linalg.svd(a.T @ bb)
  dd_ = np.sign(np.linalg.det(u @ vt))
  rot = u @ np.diag([1, 1, dd_]) @ vt
  return float(np.sqrt(((a @ rot - bb) ** 2).sum(1).mean()))


ca = np.where(np.asarray(f['distogram_atom_idx']).astype(int) >= 0,
              np.asarray(f['distogram_atom_idx']).astype(int), 0)
for tag, z, s_in in (
    ('reference trunk', None, None),
    ('GRAPH trunk    ', jnp.asarray(g['pair']), to_ref_s_inputs(jnp.asarray(g['target_feat']))),
):
  if z is None:
    zz, s_in, _ = R.trunk(f, None, pref, dims, n_loops=3, key=jax.random.PRNGKey(0),
                          lm_dropout=0.0, msa=R.self_msa(f))
  else:
    zz = z
  x = R.sample(f, s_in, zz, rp, pref, dims, jax.random.PRNGKey(0))
  x = np.asarray(x)
  a2t = np.asarray(f['atom_to_token']).astype(int)
  # CA per token: first atom of each residue is N, second CA in ESMFold2's order
  cas = np.stack([x[a2t == i][1] for i in range(int(a2t.max()) + 1)])
  print('%s  CA-RMSD %.3f A' % (tag, ca_rmsd(cas)))
