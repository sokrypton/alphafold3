"""Fold with NATIVE's trunk z through OUR diffusion. Trunk or diffusion?

esmfold2_exp_fast folds 8.8 A where native reaches 0.885, and its trunk already
agrees with native's at corr 0.988 (its working twin reads 0.9975). That gap is
real but small; this settles whether it is the cause. Hand our diffusion the
trunk NATIVE computed: a good fold means the residual trunk error was being
amplified, a bad one means the diffusion is wrong.

  MODEL=esmfold2_exp_fast NATIVE=<npz> PYTHONPATH=src:. python \
      converters/oracles/esmfold2_fold_from_native_z.py
"""
import os
import sys
import functools

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

sys.path.insert(0, '/home/ubuntu/alphafold3')
sys.path.insert(0, '/home/ubuntu/alphafold3/src')
sys.argv = sys.argv[:1]
from alphafold3.model import model as af3_model, model_registry, params as afp
from alphafold3.model.components import utils
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model.pipeline import model_features
from alphafold3.model import feat_batch
from alphafold3.model.network import diffusion_head
from converters.oracles.fold_check import parse_ca, kabsch_rmsd

MODEL = os.environ.get('MODEL', 'esmfold2_exp_fast')
S = ('/tmp/claude-1000/-home-ubuntu-ColabDesign2/'
     '77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/')
NATIVE = os.environ.get('NATIVE', S + 'exp_fast_native.npz')

nat = dict(np.load(NATIVE))
z_native = nat['trunk_out'][0].astype(np.float32)
lm_z = nat['lm_z'][0]

seq, xyz_true = parse_ca(os.path.expanduser('~/6MRR.pdb'))
d = os.path.expanduser('~/ported/%s' % MODEL)
spec = model_registry.get(MODEL)
fi = folding_input.Input(name='x', chains=[folding_input.ProteinChain(
    id='A', sequence=seq, ptms=[], unpaired_msa='', paired_msa='',
    templates=[])], rng_seeds=[0])
ccd = decoded_ccd.get_ccd()
feat = lambda **kw: featurisation.featurise_input(
    fold_input=fi, ccd=ccd, buckets=None, **kw)
batch = feat()[0]
if spec.featurise:
  batch = model_features.apply(batch, spec, refeaturise=feat, model_dir=d,
                               esm=None, has_msa=False, fold_input=fi,
                               lm_pair=lm_z)
cfg = af3_model.Model.Config()
cfg.global_config.flash_attention_implementation = 'xla'
spec.configure(cfg)


@hk.transform
def fwd(b, z):
  fb = feat_batch.Batch.from_data_dict(b)
  tf = af3_model.create_target_feat_embedding(
      batch=fb, config=cfg.evoformer, global_config=cfg.global_config)
  emb = {'pair': jnp.asarray(z),
         'single': jnp.zeros((tf.shape[0], cfg.evoformer.seq_channel),
                             jnp.float32),
         'target_feat': tf}
  head = diffusion_head.DiffusionHead(cfg.heads.diffusion, cfg.global_config)

  def denoise(x, noise_level):
    return head(positions_noisy=x, noise_level=noise_level, batch=fb,
                embeddings=emb, use_conditioning=True)

  return diffusion_head.sample(denoise, fb, hk.next_rng_key(),
                               cfg.heads.diffusion.eval, cfg.global_config)


b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
p = afp.get_model_haiku_params(model_dir=d)
p = {(k[len('diffuser/'):] if k.startswith('diffuser/') else k): v
     for k, v in p.items()}
p = {(k[len('~/'):] if k.startswith('~/') else k): v for k, v in p.items()}
out = fwd.apply(p, jax.random.PRNGKey(0), b, jnp.asarray(z_native))
pos = np.asarray(out['atom_positions'])
ca = pos[:, :, 1, :]
rs = [kabsch_rmsd(ca[i], xyz_true) for i in range(ca.shape[0])]
print('%s: OUR diffusion on NATIVE trunk z' % MODEL)
print('  %d samples, CA-RMSD: %s   best %.3f  mean %.3f'
      % (len(rs), ' '.join('%.3f' % r for r in rs), min(rs), float(np.mean(rs))))
