"""Where does our trunk's PAIR first leave native's -- stage by stage, per cycle?

    PASSES=1 JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:dev/oracles \
      python dev/oracles/trunk_stage_parity.py protenix1 <native_dump>.npz [seq]

The module gates feed each module NATIVE's features, so they are blind to what
our featuriser hands it in a real fold: every protenix1 cell read corr 1.000000
while the fold was 9 A out. This compares OUR z at each stage against native's
own, which is what named the template stage in one run.

Builds `ev.Evoformer` DIRECTLY (the esmfold2_localise_trunk recipe): going
through `fold_check.fold` returns TRACERS, because recycling runs inside
model.py's fori_loop.

PASSES=n on this side must match --model.N_cycle n on the dump, and the
comparison takes our LAST pass -- the dump keeps native's last cycle. Reading
`taps[name][0]` instead made a cycle-2 recycle term look 37% small.
"""

import os, sys
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp

sys.path.insert(0, 'dev/oracles')
os.environ['AF3_ESM_TRUNK_TAPS'] = '1'

import fold_check
from alphafold3.model import feat_batch, model as af3_model
from alphafold3.model import params as afp
from alphafold3.model.network import evoformer as ev
from alphafold3.model.components import utils

model, npz = sys.argv[1], sys.argv[2]
seq = sys.argv[3] if len(sys.argv) > 3 else (
    'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG')
N_PASSES = int(os.environ.get('PASSES', 1))
batch, cfg, model_dir = fold_check._fold_setup(model, seq, None)
cfg.global_config.bfloat16 = 'none'


@hk.transform
def fwd(b):
  fb = feat_batch.Batch.from_data_dict(b)
  L = fb.token_features.mask.shape[0]
  c = cfg.evoformer.pair_channel
  # 'pair_pre_coda' ONLY for the pair-only (ESMFold2) trunk. Evoformer reads
  # `prev.get('pair_pre_coda', prev['pair'])`, so carrying the key on a stock
  # model feeds the recycle a tensor nothing ever updates -- zeros -- and the
  # recycling silently does nothing. That cost an hour: the pair rms was
  # identical to 8 digits across passes and it read as a model property.
  prev = {'pair': jnp.zeros((L, L, c), jnp.float32),
          'single': jnp.zeros((L, cfg.evoformer.seq_channel), jnp.float32)}
  tf = af3_model.create_target_feat_embedding(
      batch=fb, config=cfg.evoformer, global_config=cfg.global_config)
  mod = ev.Evoformer(cfg.evoformer, cfg.global_config)
  rms = []
  for _ in range(N_PASSES):
    emb = mod(batch=fb, prev=prev, target_feat=tf, key=jax.random.PRNGKey(0))
    rms.append(jnp.sqrt((emb['pair'].astype(jnp.float32) ** 2).mean()))
    prev = {**prev, **{k: v.astype(jnp.float32)
                       for k, v in emb.items() if k in prev}}
  return emb, jnp.stack(rms), jnp.asarray(sorted(emb.keys()) == sorted(emb.keys()))


b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
p = afp.get_model_haiku_params(model_dir=model_dir)
p = {(k[len('diffuser/'):] if k.startswith('diffuser/') else k): v
     for k, v in p.items()}
out, pass_rms, _ = fwd.apply(p, jax.random.PRNGKey(0), b)
print('  pair rms per pass:', ' '.join('%.4f' % float(x) for x in pass_rms))
print('  emb keys:', sorted(out.keys()))
taps = {k: [np.asarray(x, np.float32) for x in v]
        for k, v in ev.ESM_TRUNK_TAPS.items()}
print('%s: taps %s' % (model, {k: len(v) for k, v in taps.items()}))
for nm in ('z_before_prev', 'z_after_prev'):
  if nm in taps:
    print('  %-14s rms per pass: %s' % (nm, ' '.join(
        '%.4f' % float(np.sqrt((x.astype(np.float64) ** 2).mean())) for x in taps[nm])))

d = np.load(npz)


def cmp(tag, ours, nat):
  a = np.asarray(ours, np.float64).ravel(); bb = np.asarray(nat, np.float64).ravel()
  if a.size != bb.size:
    print('  %-28s SHAPE %s vs %s' % (tag, np.shape(ours), np.shape(nat)))
    return
  rb = np.sqrt((bb ** 2).mean())
  print('  %-28s corr %.8f  rms ours/nat %.5f  max|d| %.4f  rms(nat) %.4f'
        % (tag, np.corrcoef(a, bb)[0, 1], np.sqrt((a ** 2).mean()) / rb,
           np.abs(a - bb).max(), rb))


# native, cycle 1: templ_in0 is z entering the template embedder (== z_init with
# the recycle term zero), msa_in0 is z entering the MSA module, pf_in1 is z
# entering the pairformer.
pairs = [('z_init            vs templ_in0', 'z_init', 'templ_in0'),
         ('z_init_generic    vs templ_in0', 'z_init_generic', 'templ_in0'),
         ('z_after_template  vs templ_out0', 'z_after_template', 'templ_out0'),
         ('z_after_template  vs msa_in0', 'z_after_template', 'msa_in0'),
         ('z_after_msa       vs msa_out0', 'z_after_msa', 'msa_out0'),
         ('trunk_in_pair     vs pf_in1', 'trunk_in_pair', 'pf_in1'),
         ('trunk_out_pair    vs pf_out1', 'trunk_out_pair', 'pf_out1'),
         ('z_pre_coda        vs z_trunk', 'z_pre_coda', 'z_trunk')]
for tag, ok, nk in pairs:
  if ok in taps and nk in d.files:
    cmp(tag, taps[ok][-1], d[nk])   # LAST pass: the dump keeps native's last cycle
if 'pair' in out:
  cmp('final pair        vs z_trunk', np.asarray(out['pair']), d['z_trunk'])
