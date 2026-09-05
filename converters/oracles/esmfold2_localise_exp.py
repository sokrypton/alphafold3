"""Our trunk vs NATIVE's, for one experimental variant, with the LM injected.

esmfold2_exp_fast folds 8.8 A where native reaches 0.885, while its
architectural twin esmfold2_exp_fast_cutoff2025 folds 1.560 against native's
1.494 -- same code path, same key set, same shapes. So the divergence is not
structural and has to be localised on activations.

Native's own `lm_z` is fed in as our lm_pair, which takes ESM-C and the shim out
of the comparison entirely: what is left is the trunk.

  MODEL=esmfold2_exp_fast NATIVE=<npz> PYTHONPATH=src:. python converters/oracles/esmfold2_localise_exp.py
"""
import os
import sys

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
from alphafold3.model.network import evoformer as ev
from converters.oracles.fold_check import parse_ca

MODEL = os.environ.get('MODEL', 'esmfold2_exp_fast')
NATIVE = os.environ.get(
    'NATIVE', '/tmp/claude-1000/-home-ubuntu-ColabDesign2/'
    '77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad/exp_fast_native.npz')

nat = dict(np.load(NATIVE))
lm_z = nat['lm_z'][0]
z_native = nat['trunk_out'][0]

seq, _ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
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
cfg.global_config.bfloat16 = 'none'
spec.configure(cfg)
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
    prev = {**prev, **{k: v.astype(jnp.float32)
                       for k, v in emb.items() if k in prev}}
  return emb


b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
p = afp.get_model_haiku_params(model_dir=d)
p = {(k[len('diffuser/'):] if k.startswith('diffuser/') else k): v
     for k, v in p.items()}
g = fwd.apply(p, jax.random.PRNGKey(0), b)
z = np.asarray(g['pair'])
print('%s: OUR trunk vs NATIVE trunk (native lm_z injected)' % MODEL)
print('   shapes %s vs %s' % (z.shape, z_native.shape))
print('   corr %.6f   relerr %.3e' % (
    np.corrcoef(z.ravel(), z_native.ravel())[0, 1],
    np.abs(z - z_native).max() / max(np.abs(z_native).max(), 1e-9)))
print('   ours std %.4f  absmax %.3f' % (z.std(), np.abs(z).max()))
print('   nat  std %.4f  absmax %.3f' % (z_native.std(), np.abs(z_native).max()))

# The distogram is deterministic and reads straight off the trunk, so it says
# whether the remaining trunk gap MATTERS -- a contact map that agrees with
# native's argmax leaves only the diffusion to explain the fold.
from alphafold3.model.network import distogram_head as dh


@hk.transform
def disto(b, pair, tf):
  fb = feat_batch.Batch.from_data_dict(b)
  return dh.DistogramHead(cfg.heads.distogram, cfg.global_config)(
      fb, {'pair': jnp.asarray(pair), 'single': jnp.zeros(
          (pair.shape[0], cfg.evoformer.seq_channel), jnp.float32),
       'target_feat': tf}, return_distogram=True)


tf = np.asarray(g['target_feat'])
try:
  ours = disto.apply(p, jax.random.PRNGKey(0), b, z, jnp.asarray(tf))
  logits = np.asarray(ours['bin_edges'] if 'bin_edges' in ours else
                      ours.get('distogram', ours))
except Exception as err:
  print('   distogram head: %s: %s' % (type(err).__name__, err))
  logits = None
nat_logits = nat['distogram_logits'][0]
if logits is not None and getattr(logits, 'shape', ())[-1:] == nat_logits.shape[-1:]:
  a = logits.argmax(-1); bnat = nat_logits.argmax(-1)
  print('   distogram argmax agreement %.4f   corr %.6f'
        % ((a == bnat).mean(),
           np.corrcoef(np.asarray(logits).ravel(), nat_logits.ravel())[0, 1]))
else:
  # even without the head, the SYMMETRISED trunk is what the head reads
  sym = z + z.transpose(1, 0, 2)
  symn = z_native + z_native.transpose(1, 0, 2)
  print('   symmetrised trunk corr %.6f' % np.corrcoef(sym.ravel(), symn.ravel())[0, 1])
