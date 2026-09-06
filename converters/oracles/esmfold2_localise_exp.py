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
  per_pass = []
  for _ in range(N_PASSES):
    emb = mod(batch=fb, prev=prev, target_feat=tf, key=jax.random.PRNGKey(0))
    per_pass.append(emb['pair'])
    prev = {**prev, **{k: v.astype(jnp.float32)
                       for k, v in emb.items() if k in prev}}
  return emb, per_pass


b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
p = afp.get_model_haiku_params(model_dir=d)
p = {(k[len('diffuser/'):] if k.startswith('diffuser/') else k): v
     for k, v in p.items()}
# haiku names a layer_stack's scope after whether it carries per-layer inputs,
# so turning the tap on renames every trunk block param. The weights are
# identical -- only the scope moves -- so alias them rather than reconvert.
if os.environ.get('ESM_TAP_TRUNK_BLOCKS') == '1':
  p = {**p, **{k.replace('__layer_stack_no_per_layer',
                         '__layer_stack_with_per_layer'): v
               for k, v in p.items()
               if '__layer_stack_no_per_layer/trunk_pairformer' in k}}
ev.TRUNK_BLOCK_TAPS.clear()  # one entry per recycle pass is appended below
for _v in ev.ESM_TRUNK_TAPS.values():
  _v.clear()
g, per_pass = fwd.apply(p, jax.random.PRNGKey(0), b)
z = np.asarray(g['pair'])

# Per PASS, which says whether the divergence enters at the injection or grows
# inside the 24 blocks. native's `trunk_in{i}` is what our _add_prev produces
# (z_init + pair_loop_proj(z)); `trunk_pass{i}` is the block stack's output.
taps = ev.ESM_TRUNK_TAPS
# Decompose the injection. Native builds trunk_in0 (pass 0, where prev is zero)
# as outer(z_init_1, z_init_2) + relpos + bonds + lm_z, and lm_z is injected
# verbatim -- so any gap at the injection is in the other two, and this says
# which. The trunk amplifies an injection error ~50x, so a 1e-4 gap here is
# what a 3e-2 gap after 24 blocks is made of.
if 'z_init_1' in nat and taps.get('z_pair0'):
  z1, z2 = nat['z_init_1'][0], nat['z_init_2'][0]
  nat_outer = z1[:, None, :] + z2[None, :, :]
  our_outer = np.asarray(taps['z_pair0'][0])
  # The remainder of trunk_in0 is NOT just relpos: `pair_loop_proj` is a
  # LayerNorm+Linear, so on pass 0 it maps the zero carry to a nonzero constant
  # (LN(0) -> offset -> Linear) that lands in trunk_in0 too. Native's own relpos
  # is dumped, so compare against it rather than against a subtraction that
  # silently absorbs that constant.
  our_rem = np.asarray(taps['z_relpos'][0]) - our_outer
  pairs = [('outer(z_init_1,2)', our_outer, nat_outer)]
  if 'relpos' in nat:
    pairs.append(('relpos', our_rem, nat['relpos'][0]))
    pairs.append(('bonds (native=0)',
                  np.asarray(taps['z_init'][0]) - np.asarray(taps['z_relpos'][0]),
                  nat['bonds'][0]))
  for label, a, bb in pairs:
    print('   %-18s corr %.7f  std %.4f vs %.4f'
          % (label, np.corrcoef(a.ravel(), bb.ravel())[0, 1], a.std(), bb.std()))

for i in range(N_PASSES):
  for ours, key, label in ((taps.get('z_parcae', [None] * 9)[i]
                            if len(taps.get('z_parcae', [])) > i else None,
                            'trunk_in%d' % i, 'injection'),
                           (per_pass[i], 'trunk_pass%d' % i, 'after 24 blocks')):
    if ours is None or key not in nat:
      continue
    a, bb = np.asarray(ours), nat[key][0]
    print('   pass %d %-16s corr %.6f  std %.3f vs %.3f'
          % (i, label, np.corrcoef(a.ravel(), bb.ravel())[0, 1], a.std(), bb.std()))
# Per BLOCK inside pass 0, when both sides taped it. This is the test that
# separates a per-block convention from a mapping error: a corr that decays
# smoothly across the 24 is a convention every block shares, while a cliff at
# one j is that block's weights.
blocks = ev.TRUNK_BLOCK_TAPS
if blocks and 'block0' in nat:
  per_layer = np.asarray(blocks[0])  # pass 0, (num_layer, L, L, c) -- native
                                     # dumps pass 0's blocks, so pass 0 it is
  print('   per-block, pass 0:')
  for j in range(per_layer.shape[0]):
    key = 'block%d' % j
    if key not in nat:
      break
    a, bb = per_layer[j], nat[key][0]
    # The block's own UPDATE, not its output. A block whose output std shrinks
    # is mostly cancelling its input, which amplifies whatever error arrived --
    # so a low output corr there can be inherited rather than local. The update
    # is what the block itself computed, and only IT indicts the block.
    prev_a = per_layer[j - 1] if j else np.asarray(taps['z_parcae'][0])
    prev_b = nat['block%d' % (j - 1)][0] if j else nat['trunk_in0'][0]
    du, dn = a - prev_a, bb - prev_b
    print('     block %2d  corr %.6f  std %.3f vs %.3f   update corr %.6f'
          % (j, np.corrcoef(a.ravel(), bb.ravel())[0, 1], a.std(), bb.std(),
             np.corrcoef(du.ravel(), dn.ravel())[0, 1]))
elif 'block0' in nat:
  print('   per-block: set ESM_TAP_TRUNK_BLOCKS=1 to fill TRUNK_BLOCK_TAPS')

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
