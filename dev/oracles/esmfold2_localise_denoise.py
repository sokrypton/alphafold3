"""One denoise step: the GRAPH's diffusion head against the reference's.

Both run on their own trunk (which agrees to corr 0.9887) but on the SAME noisy
coordinates and the same t_hat, so a divergence here is the score network, not
the sampler and not the trunk.
"""
import sys, os, numpy as np, jax, jax.numpy as jnp, haiku as hk
sys.path.insert(0, '/home/ubuntu/alphafold3'); sys.path.insert(0, '/home/ubuntu/alphafold3/src')
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esmfold2_dumps import state_dict as _sd_of, native as _native_of
import esmfold2_dumps as _native_dumps
# MODEL picks the release: every ESMFold2 variant has its own weights AND
# its own shim, and crossing them reads corr 0.026 against native.
MODEL = os.environ.get('MODEL', 'esmfold2')
from alphafold3.model import model as af3_model, model_registry, params as afp
from alphafold3.model.components import utils
from alphafold3.common import folding_input
from alphafold3.constants import decoded_ccd
from alphafold3.data import featurisation
from alphafold3.model.pipeline import model_features
from alphafold3.model import feat_batch
from alphafold3.model.network import evoformer as ev, diffusion_head as dh
from dev.oracles.fold_check import parse_ca
from converters import esmfold2 as CV
# the reference moved from converters/oracles/ to dev/oracles/ when the
# repo split what SHIPS from what only verifies; this import had not
# followed it, so every harness below had been unrunnable since.
import esmfold2_reference as R

T_HAT = 8.0
seq, _ = parse_ca(os.path.expanduser('~/6MRR.pdb'))
# MODEL selects the release on BOTH sides. It used to pick the reference's
# weights while the graph always loaded ~/ported/esmfold2, so every variant but
# the base one compared two different models -- esmfold2_fast read corr 0.069
# with the graph's std identical to the base model's, which is the tell.
d = os.path.expanduser('~/ported/%s' % MODEL)
spec = model_registry.get(MODEL)
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
# NB lets the atom stack be truncated to N blocks on both sides, to see whether
# the divergence appears in one block or accumulates over three.
NB = int(os.environ.get('NB', '0'))
if NB:
    cfg.heads.diffusion.atom_transformer.num_blocks = NB

sd = _sd_of(MODEL); dims = CV.derive_dims(sd); dims['n_input_atom'] = 3
dd = _native_of(MODEL)
f = {k[5:]: jnp.asarray(v[0]) for k, v in dd.items() if k.startswith('feat.')}
pref = {k: jnp.asarray(v) for k, v in CV.map_esmfold2_to_af3(sd).items()}
rmask = np.asarray(f['atom_attention_mask']).astype(bool)   # 576 slots, 573 real
x_real = np.asarray(jax.random.normal(jax.random.PRNGKey(7), (int(rmask.sum()), 3))) * T_HAT
x_flat = np.zeros(rmask.shape + (3,), np.float32); x_flat[rmask] = x_real

# ZERO_ATOM=1 makes the atom stacks EXACT IDENTITIES on both sides, by zeroing
# the two output gates of every atom block (`x + g*f(x)` with g = 0). Any
# residual then lives OUTSIDE the atom blocks -- in atom_linear, the
# atom->token aggregation, the token transformer or the decoder projection --
# which bisects the score network without writing a new harness. The gates are
# chunks 2 and 5 of the fused adaln modulation on the reference side and the
# `*adaptive_zero_cond` slots on ours.
# ZERO_ATOM=1 kills both gates (identity stacks); =attn keeps only the
# attention sublayer (kills the FFN gate); =ffn keeps only the FFN.
# ZERO_ATOM also selects WHICH stack: `enc` / `dec` zero one of the two, which
# is what separates the encoder from the decoder. The encoder is exact on its
# own gate (atom_parity.py esmfold2, SAME_ATOM_SET + NATIVE_REF_POS -> 1.000000),
# so a residual that survives here has to come from the decoder.
ZERO_ATOM = os.environ.get('ZERO_ATOM', '')
_CHUNKS = {'1': (2, 5), 'attn': (5,), 'ffn': (2,),
           'enc': (2, 5), 'dec': (2, 5)}.get(ZERO_ATOM, ())
_STACKS = {'enc': ('atom_encoder',), 'dec': ('atom_decoder',)}.get(
    ZERO_ATOM, ('atom_encoder', 'atom_decoder'))
if _CHUNKS:
  for _which in _STACKS:
    _k = 'diffusion/%s/blocks/adaln/weights' % _which
    _w = np.array(pref[_k])
    for _c in _CHUNKS:
      _w[:, :, _c * 128:(_c + 1) * 128] = 0.0
    pref[_k] = jnp.asarray(_w)
  print('   ZERO_ATOM=%s: zeroed adaln chunks %s (reference side)'
        % (ZERO_ATOM, _CHUNKS))

zr, s_in, _ = R.trunk(f, None, pref, dims, n_loops=3, key=jax.random.PRNGKey(0),
                      lm_dropout=0.0,
                      # see the note in esmfold2_localise_trunk.py: a *_fast
                      # release has no msa_encoder to build one with
                      msa=R.self_msa(f) if dims.get('n_msa') else None)
rp = jnp.asarray(R.rel_pos_features(
    f['residue_index'].astype(int), f['asym_id'].astype(int), f['sym_id'].astype(int),
    f['entity_id'].astype(int), f['token_index'].astype(int))) @ pref['rel_pos/weights']
if NB:
    dims = dict(dims); dims['n_diff_atom'] = NB

N_PASSES = 4
# SAME_INPUTS=1 removes the two FEATURISATION differences this gate otherwise
# measures, so what is left is the port:
#   * our terminal OXT, which ESMFold2's PROTEIN_HEAVY_ATOMS table does not have
#     (574 atoms against 573), and which the encoder's scatter_mean pooling
#     turns into a whole-token difference;
#   * our CCD ideal ref_pos against its PROTEIN_REF_POS table -- mean 3.31 A
#     apart in local frame, and it feeds the rotary embedding.
# With both removed the atom ENCODER reads exactly 1.000000
# (atom_parity.py esmfold2 SAME_ATOM_SET=1 NATIVE_REF_POS=1), so this says
# whether the whole denoise step follows.
SAME_INPUTS = os.environ.get('SAME_INPUTS') == '1'
if SAME_INPUTS:
  import dataclasses
  _fb_tmp = feat_batch.Batch.from_data_dict(
      jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch)))
  _ri, _oi = _native_dumps.atom_map(_fb_tmp, f)
  _m = np.zeros(np.asarray(_fb_tmp.ref_structure.mask).shape, bool).reshape(-1)
  _m[_oi] = True
  _m = _m.reshape(np.asarray(_fb_tmp.ref_structure.mask).shape)
  _pos = np.array(np.asarray(_fb_tmp.ref_structure.positions), copy=True)
  _pos.reshape(-1, 3)[_oi] = np.asarray(f['ref_pos']).reshape(-1, 3)[_ri]
  batch = dict(batch)
  # `pred_dense_atom_mask` is the predicted-structure mask the encoder pools
  # over; `ref_mask` the reference-structure one. Both have to lose the atom.
  for _k2, _v2 in (('ref_pos', _pos), ('ref_mask', _m.astype(np.float32)),
                   ('pred_dense_atom_mask', _m.astype(np.float32))):
    if _k2 in batch:
      batch[_k2] = _v2
  print('   SAME_INPUTS=1: %d atoms kept, ref_pos taken from the dump'
        % int(_m.sum()))

b = jax.tree_util.tree_map(jnp.asarray, utils.remove_invalidly_typed_feats(batch))
fb0 = feat_batch.Batch.from_data_dict(b)
gmask = np.asarray(fb0.predicted_structure_info.atom_mask).astype(bool)

# MATCH THE TWO ATOM LISTS BY NAME, NOT BY POSITION. Our featuriser emits 574
# atoms for 6MRR and ESMFold2's 573 -- ours carries the terminal OXT, its
# PROTEIN_HEAVY_ATOMS table does not -- so `buf[:n] = x_real[:n]` put every atom
# after that one against its neighbour. Measured before fixing: 567 of 573 atoms
# more than 0.5 A apart in ref_pos alone, mean 3.31 A. The gate was reporting a
# permutation (1.34 A/atom) and it looked like a port bug.
#
# Same lesson as the conformer Kabsch that read a flat ~0.9 A for everything:
# match on `ref_atom_name_chars` (see PARITY.md, "the atom correspondence"), per
# token, and compare only atoms both sides have.
def _names(chars):
  ch = np.asarray(chars).astype(int)
  if ch.ndim == 3 and ch.shape[-1] > 8:            # one-hot over 64 characters
    ch = ch.argmax(-1)
  return ch

_gn = _names(fb0.ref_structure.atom_name_chars)     # (tokens, slots, 4)
_rn = _names(f['ref_atom_name_chars'])              # (flat_atoms, 4)
_r2t = np.asarray(f['atom_to_token']).astype(int)
_nm = lambda v: ''.join(chr(c + 32) for c in v).strip()

_our_by_tok = {}
for _t in range(_gn.shape[0]):
  for _sl in range(_gn.shape[1]):
    if gmask[_t, _sl]:
      _our_by_tok.setdefault(_t, {})[_nm(_gn[_t, _sl])] = _t * _gn.shape[1] + _sl

_ref_idx, _our_flat = [], []
for _i in np.flatnonzero(rmask):
  _slot = _our_by_tok.get(int(_r2t[_i]), {}).get(_nm(_rn[_i]))
  if _slot is not None:
    _ref_idx.append(int(_i)); _our_flat.append(int(_slot))
_ref_idx = np.asarray(_ref_idx); _our_flat = np.asarray(_our_flat)
n = len(_ref_idx)
print('atoms: ours %d, reference %d, MATCHED BY NAME %d'
      % (int(gmask.sum()), int(rmask.sum()), n))

# One shared noise vector, placed at each side's own index for the same atom.
x_common = np.asarray(jax.random.normal(jax.random.PRNGKey(7), (n, 3))) * T_HAT
x_flat = np.zeros(rmask.shape + (3,), np.float32)
x_flat.reshape(-1, 3)[_ref_idx] = x_common
dense = np.zeros(gmask.shape + (3,), np.float32)
dense.reshape(-1, 3)[_our_flat] = x_common


# INJECT=1 feeds the REFERENCE's trunk output into our diffusion head instead of
# running our own trunk. Without it the two sides each run their own trunk, and
# the trunk agrees only to relerr 4.9e-03 on a z of std 33.6 -- which the score
# network then amplifies, so the headline mixes "our trunk" with "our
# denoiser". This is the seam that separates them, and it is the only way to
# read the denoise number as a statement about the score network.
INJECT = os.environ.get('INJECT') == '1'


@hk.transform
def fwd(bb, x_noisy, z_inj=None, s_inj=None):
    fb = feat_batch.Batch.from_data_dict(bb)
    L = fb.token_features.mask.shape[0]
    c = cfg.evoformer.pair_channel
    prev = {'pair': jnp.zeros((L, L, c), jnp.float32),
            'pair_pre_coda': jnp.zeros((L, L, c), jnp.float32),
            'single': jnp.zeros((L, cfg.evoformer.seq_channel), jnp.float32)}
    tf = af3_model.create_target_feat_embedding(
        batch=fb, config=cfg.evoformer, global_config=cfg.global_config)
    if z_inj is not None:
        # ESMFold2 has no single track (PAIR_ONLY_TRUNK), so the head reads
        # `pair` and `target_feat` only; `single` is carried to satisfy the dict.
        emb = {'pair': z_inj.astype(jnp.float32),
               'single': jnp.zeros((L, cfg.evoformer.seq_channel), jnp.float32),
               'target_feat': s_inj.astype(jnp.float32)}
    else:
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
_STACKS_SEL = ZERO_ATOM if ZERO_ATOM in ('enc', 'dec') else ''
if _CHUNKS:
  # the gate lives in the SCOPE name, not the leaf (haiku appends the module
  # name, so the leaf is just 'weights'). 'adaptive_zero_cond' is a SUBSTRING of
  # 'ffw_adaptive_zero_cond', so the attention gate has to be matched by its
  # absence.
  def _hit(k):
    if 'diffusion_atom_transformer' not in k:
      return False
    if 'enc' in _STACKS_SEL and 'encoder' not in k:
      return False
    if 'dec' in _STACKS_SEL and 'decoder' not in k:
      return False
    ffn = 'ffw_adaptive_zero_cond' in k
    attn = ('adaptive_zero_cond' in k) and not ffn
    return (ffn and 5 in _CHUNKS) or (attn and 2 in _CHUNKS)
  _p = {k: ({kk: (np.zeros_like(np.asarray(vv)) if _hit(k) else vv)
             for kk, vv in v.items()}) for k, v in _p.items()}
  _n = sum(1 for k in _p if _hit(k))
  print('   ZERO_ATOM=1: zeroed %d atom-block gates (our side)' % _n)

# TRUNCATE BOTH SIDES. NB shortens the atom stack, and the blob's params carry
# the full 3 blocks -- asking the graph for 1 is a shape error, not a shorter
# run ('pair_logits_projection/weights' with retrieved shape (16, 3, 4) does not
# match (16, 1, 4)). The same rule every other block knob in this repo obeys.
# Note the per-layer axis differs by scope: leading for the layer_stack'd
# weights, axis 1 for the per-super-block pair logits.
if NB:
  # The block count is read off the ARRAYS, not from `dims` -- `dims` has
  # already been set to NB by this point, so using it truncated nothing and the
  # shape error stood.
  _cut, _was = {}, None
  for k, v in _p.items():
    if 'diffusion_atom_transformer' not in k:
      _cut[k] = v; continue
    _cut[k] = {}
    for leaf, arr in v.items():
      a_ = np.asarray(arr)
      if 'pair_logits_projection' in k and a_.ndim == 3 and a_.shape[1] > NB:
        _was = _was or a_.shape[1]; a_ = a_[:, :NB]
      elif '__layer_stack' in k and a_.ndim and a_.shape[0] > NB:
        _was = _was or a_.shape[0]; a_ = a_[:NB]
      _cut[k][leaf] = a_
  _p = _cut
  print('   NB=%d: atom stack truncated on BOTH sides (was %s)' % (NB, _was))
# Both sides denoise the same atoms from the same coordinates, and each is read
# back at ITS OWN indices for those atoms.
x_ref_full = np.asarray(
    R.denoise(jnp.asarray(x_flat), T_HAT, f, s_in, zr, rp, pref, dims))
x_ref = x_ref_full.reshape(-1, 3)[_ref_idx]
# The reference's s_inputs is ESM's 451 layout; ours is 447 and the graph pads
# it back to 451 itself, so inject the OURS-layout view or the pad makes 455.
#     ESMFold2  [atom 384 | restype 33 | profile 33 | deletion 1]  = 451
#     AF3       [restype 31 | profile 31 | deletion 1 | atom 384]  = 447
# with ESM reserving two restype classes AF3 does not (hence the +2 offsets).
_idx = np.concatenate([384 + 2 + np.arange(31), 384 + 33 + 2 + np.arange(31),
                       [384 + 33 + 33], np.arange(384)])
_s_ours = np.asarray(s_in)[:, _idx] if INJECT else None
out = np.asarray(fwd.apply(
    _p, jax.random.PRNGKey(0), b, jnp.asarray(dense),
    *( (jnp.asarray(zr), jnp.asarray(_s_ours)) if INJECT else (None, None) )))
if INJECT:
    print('   INJECT=1: our denoiser on the REFERENCE trunk (score network only)')
x_graph = out.reshape(-1, 3)[_our_flat]
# THE PER-STAGE BREAKDOWN IS OPTIONAL, and the taps it reads are gone. The port
# instrumented `diffusion_head.DIFF_TAPS` and `atom_cross_attention.ATOM_TAPS`
# while it was being localised, and those were removed afterwards -- correctly,
# debug taps should not ship. The HEADLINE comparison below needs none of them,
# so the breakdown is skipped rather than crashing the gate with
# `module ... has no attribute 'DIFF_TAPS'`, which is what it did.
#
# To get the breakdown back, reinstate the taps in the library for the duration
# of the investigation; it is a localisation aid, not the measurement.
gt = getattr(dh, 'DIFF_TAPS', None)
if gt is None:
  print('   (per-stage taps not present in the library; headline only)')
else:
  for name in ('single_cond', 'pair_cond', 'token_act', 'act_pre_transformer',
               'act_post_transformer', 'r_update'):
      if name in gt and name in R.TAPS:
          x = np.asarray(gt[name][0]); y = np.asarray(R.TAPS[name][0])
          if name == 'r_update':                     # dense vs flat atom layout
              x = x[gmask][:n]; y = y[rmask][:n]
              globals()['r_ref_own'] = y
          if x.shape != y.shape:
              print('   %-20s SHAPE %s vs %s' % (name, x.shape, y.shape)); continue
          print('   %-20s corr %.6f  std %.4f vs %.4f'
                % (name, np.corrcoef(x.ravel(), y.ravel())[0, 1], x.std(), y.std()))

  from alphafold3.model.network import atom_cross_attention as aca
  nk = np.asarray(aca.ATOM_TAPS['diffusion_swa_nkeys'][0]).ravel()
  qr = np.asarray(aca.ATOM_TAPS['diffusion_q_rank'][0]).ravel()
  real = np.asarray(aca.ATOM_TAPS['diffusion_qmask'][0]).ravel().astype(bool)
  nr = nk[real]
  print('   swa keys per query: min %d  median %d  max %d over %d real queries'
        ' (ideal 129 in the interior)' % (nr.min(), int(np.median(nr)), nr.max(), real.sum()))
  import collections
  print('   histogram', sorted(collections.Counter(nr.tolist()).items())[:6], '...',
        sorted(collections.Counter(nr.tolist()).items())[-4:])
  # WHICH flat order is the queries layout? Compare the per-atom features, whose
  # value is fixed by the element and atom name, under both readings: dense
  # (token, slot) flattened, or the compact atom list.
  xa = np.asarray(aca.ATOM_TAPS['diffusion_atom_features'][0])
  ya = np.asarray(R.TAPS['atom_features'][0])[:n]
  for how, v in (('dense[gmask]', xa.reshape(-1, xa.shape[-1])[np.asarray(gmask).ravel()][:n]),
                 ('flat[:n]', xa.reshape(-1, xa.shape[-1])[:n])):
      print('   atom_features %-14s corr %.6f' % (how, np.corrcoef(v.ravel(), ya.ravel())[0, 1]))

  for gname, rname in (('diffusion_enc_queries_in', 'enc_queries_in'),
                       ('diffusion_enc_queries', 'enc_queries'),
                       ('diffusion_dec_queries', 'dec_queries')):
      if gname in aca.ATOM_TAPS and rname in R.TAPS:
          x = np.asarray(aca.ATOM_TAPS[gname][0])
          y = np.asarray(R.TAPS[rname][0])
          # the queries layout is the FLAT atom list in blocks of 32, not the
          # per-token dense layout gmask indexes -- reshaping and masking with
          # gmask would compare different atoms.
          x = x.reshape(-1, x.shape[-1])
          print('     (%s shapes %s vs %s; ref mask trailing=%s)'
                % (rname, x.shape, y.shape, bool(rmask[:int(rmask.sum())].all())))
          x, y = x[:n], y[:n]
          print('   %-20s corr %.6f  std %.4f vs %.4f'
                % (rname, np.corrcoef(x.ravel(), y.ravel())[0, 1], x.std(), y.std()))

# the headline's own operands, outside the optional breakdown above
a, c = x_graph.ravel(), x_ref.ravel()
print('x_denoised  GRAPH vs REFERENCE   (t_hat = %.3g, %d atoms)' % (T_HAT, n))
print('   corr %.6f   rms diff %.4f A' % (np.corrcoef(a, c)[0, 1],
                                          np.sqrt(((x_graph - x_ref) ** 2).sum(-1).mean())))
print('   graph std %.4f   ref std %.4f' % (x_graph.std(), x_ref.std()))
# WHERE the disagreement sits. Our featuriser emits one atom ESMFold2's does not
# (the terminal OXT), and a +/-64 rank window means that atom is a KEY for the
# last ~64 atoms only. If the error is concentrated there, it is an input
# difference and not the score network; if it is flat, it is the network.
_d = np.sqrt(((x_graph - x_ref) ** 2).sum(-1))
_k = 64
print('   per-atom |d|: first %d mean %.4f | middle mean %.4f | last %d mean %.4f'
      % (_k, _d[:_k].mean(), _d[_k:-_k].mean(), _k, _d[-_k:].mean()))
# IS IT A TRANSLATION? A centring convention (one side re-centres the noisy
# coordinates, the other does not) shifts every atom by the same vector, which
# leaves corr and std almost untouched and shows up as a near-uniform per-atom
# |d| -- exactly what the three means above look like. So take the mean offset
# out and see what is left.
_off = (x_graph - x_ref).mean(0)
_res = np.sqrt((((x_graph - _off) - x_ref) ** 2).sum(-1).mean())
print('   mean offset %s |%.4f| A -> rms after removing it %.4f A (was %.4f)'
      % (np.round(_off, 4), float(np.linalg.norm(_off)), _res,
         np.sqrt(((x_graph - x_ref) ** 2).sum(-1).mean())))
# IS IT A ROTATION? An augmentation/frame convention on one side only would
# leave a rigid-body difference. Kabsch the two and report what survives: if
# the aligned rms collapses the gap is a frame, if it does not the score network
# genuinely disagrees.
def _kab(a_, b_):
    ac, bc = a_ - a_.mean(0), b_ - b_.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    dsign = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, dsign]) @ vt
    return float(np.sqrt(((ac @ r - bc) ** 2).sum(-1).mean()))
print('   rigid-body aligned rms %.4f A (unaligned %.4f) -> %s'
      % (_kab(x_graph, x_ref), np.sqrt(((x_graph - x_ref) ** 2).sum(-1).mean()),
         'a FRAME difference' if _kab(x_graph, x_ref) < 0.3
         else 'NOT rigid-body'))
_q = np.quantile(_d, [0.5, 0.9, 0.99])
print('   per-atom |d| median %.4f  p90 %.4f  p99 %.4f  max %.4f'
      % (_q[0], _q[1], _q[2], _d.max()))

# How much of the residual is simply a DIFFERENT REFERENCE CONFORMER? Our
# featuriser builds ref_pos from the CCD/RDKit ideal, ESMFold2's dump carries
# its own, and the atom blocks read ref_pos through rotary embeddings -- so the
# atom path is the one place a conformer difference is not cosmetic. Re-run the
# reference denoise with OUR conformer substituted and see how far r_update
# moves on its own.
import copy
f2 = dict(f)
sub = np.asarray(f['ref_pos']).copy()
gpos_dense = np.asarray(fb0.ref_structure.positions)
gmask_ref = np.asarray(fb0.ref_structure.mask).astype(bool)
gflat = gpos_dense[gmask_ref]
sub.reshape(-1, 3)[_ref_idx] = gpos_dense.reshape(-1, 3)[_our_flat]
f2['ref_pos'] = jnp.asarray(sub)
# The REFERENCE's own taps are still there (it is our code); only the library's
# were removed. So take its r_update from this run before the swap clears them,
# rather than from the graph taps the optional breakdown above used to provide.
r_ref_own = np.asarray(R.TAPS['r_update'][0]).reshape(-1, 3)[_ref_idx]
R.TAPS.clear()
R.denoise(jnp.asarray(x_flat), T_HAT, f2, s_in, zr, rp, pref, dims)
r_swap = np.asarray(R.TAPS['r_update'][0]).reshape(-1, 3)[_ref_idx]
print('reference r_update, OUR conformer vs ITS own: corr %.6f'
      % np.corrcoef(r_swap.ravel(), np.asarray(r_ref_own).ravel())[0, 1])
