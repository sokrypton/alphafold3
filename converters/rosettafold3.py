"""RoseTTAFold3 (RF3, Baker lab / RosettaCommons foundry, files.ipd.uw.edu) -> AF3 haiku.

RF3 ADOPTED the AF3 architecture (pairformer + af3 diffusion + af3 losses) despite its
Baker-lab origin -- structurally IDENTICAL to protenix (pairformer 48 / msa / template 2 /
confidence 4 / diffusion 24 / atom 3), at STANDARD AF3 dims (c_z=128, no widening). So the
port is a protenix-clone with DIALECT_ROSETTAFOLD3 + a couple of rf3-gated forward flags.

Checkpoint: ~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt (LATEST, recommended;
`shadow.*` EMA copy, strip 'shadow.'). Source: ~/foundry_rf3 (RosettaCommons/foundry),
models/rf3/src/rf3/. State-dict fully dumped -> DIALECT_ROSETTAFOLD3 below is definitive.

Divergences from protenix (all recorded in memory rosettafold3-port.md):
  - leaf-name dialect: tri-mul = boltz2 block-fused (norm_in/out, p_in/g_in/p_out/g_out);
    transition = flat swiglu (layer_norm_1, linear_1/linear_2/linear_3); grid+single attn use
    `to_q/to_k/to_v/to_g/to_out` (tri) and `to_a` (single output) + `to_b` pair bias, `ln_0`/
    `ln_1`/`norm` layernorms -- NO `mha.` nesting; to_g is a Sequential (leaf `to_g.0`).
  - diffusion adaLN = `ada_ln`(ln_s/to_gain/to_bias); kq_norm (query_layer_norm/key_layer_norm
    on the diffusion attention q/k) -- a NEW rf3-gated forward branch; no_residual flag.
  - atom feature fused into `process_input_features` (393-d); template feature 66-d.
  - z-init standard AF3 (to_z_init_i/j direct from s_inputs; no zinit@sinit compose).

STATUS: DIALECT + loader written; the section map (map_rosettafold3_to_af3) is the next build --
mirror converters/protenix.py, swapping the dialect + the to_g.0 gate + to_out/to_a outputs.
NOT yet wired into __init__/registry (WIP).
"""

from pathlib import Path

import numpy as np

from . import common as C
from .common import Dialect

# ─── RF3 dialect (from the dumped state_dict; DEFINITIVE) ─────────────────────
DIALECT_ROSETTAFOLD3 = Dialect(
    # transition: flat SwiGLU (silu(linear_1)*linear_2 -> linear_3), norm layer_norm_1
    tr_ln='layer_norm_1', tr_mode='swiglu',
    tr_a='linear_1', tr_b='linear_2', tr_out='linear_3',
    # triangle multiplication: boltz2-style BLOCK-FUSED (p_in/g_in are c_z->2*c_hidden)
    tm_fused='block', tm_ln_in='norm_in', tm_ln_out='norm_out',
    tm_ab_p='p_in', tm_ab_g='g_in', tm_z='p_out', tm_g='g_out',
    # triangle attention: flat to_* (no mha nesting); tri-attn gate is a PLAIN linear
    # (to_g.weight, +bias -- note RF3 gate has a bias AF3's gating_query lacks; dropped).
    ga_ln='norm', ga_bias='to_b', ga_mha='',
    ga_q='to_q', ga_k='to_k', ga_v='to_v', ga_g='to_g', ga_o='to_out',
    # single attention (attention_pair_bias): ln_1(input)/ln_0(pair)/to_b(pair proj);
    # to_q/k/v, to_g Sequential (to_g.0) gate, to_a output. NO mha nesting.
    sa_ln_a='ln_1', sa_ln_z='ln_0', sa_z='to_b', sa_mha='',
    sa_q='to_q', sa_k='to_k', sa_v='to_v', sa_g='to_g.0', sa_o='to_a',
    # MSA pair-weighted averaging (recycler.msa_module.msa_pair_weighted_averaging): flat to_*
    msa_ln_m='norm_msa', msa_ln_z='norm_pair', msa_z='to_bias',
    msa_v='to_v', msa_g='to_gate', msa_o='to_out',
    # outer product mean (recycler.msa_module.outer_product): norm/proj_left/proj_right/proj_out(+bias)
    opm_ln='norm', opm_l='proj_left', opm_r='proj_right', opm_out='proj_out', opm_out_direct=False,
)
D = DIALECT_ROSETTAFOLD3


def _t(w):
  return C.t(w)


def load_rf3_checkpoint(ckpt_path, use_ema=True):
  """Load rf3_foundry_*.ckpt -> state dict. Prefer the `shadow.*` EMA copy (use_ema),
  else `model.*`; strip the prefix. Returns {leaf: np.ndarray-able tensor}."""
  import torch
  raw = torch.load(str(Path(ckpt_path).expanduser()), map_location='cpu',
                   weights_only=False)['model']
  pfx = 'shadow.' if (use_ema and any(k.startswith('shadow.') for k in raw)) else 'model.'
  return {k[len(pfx):]: v for k, v in raw.items() if k.startswith(pfx)}


# map_rosettafold3_to_af3 + convert_rosettafold3_weights: NEXT (protenix-clone; see module docstring +
# memory rosettafold3-port.md for the complete section-by-section spec).


# ─── section mappers (protenix-clone; reuse protenix's generic helpers) ───────
from .protenix2 import _set, _get, _has, _populate_scope, _stack_blocks  # noqa: E402


def _rosettafold3_pair_block(sd, prefix, pair_H, pair_D):
  """RF3 pair stack (tri_mul_outgoing/incoming + tri_attn_start/end + z_transition).

  RF3 sub-module names differ from protenix (tri_mul_out/in, tri_att_*, pair_transition),
  so we call the dialect primitives with RF3's exact sub-names onto the shared target tags.
  """
  out = {}
  for tag, name, outgoing in [('triangle_multiplication_outgoing', 'tri_mul_outgoing', True),
                              ('triangle_multiplication_incoming', 'tri_mul_incoming', False)]:
    for k, v in C.triangle_mul(sd, f'{prefix}.{name}', D, outgoing=outgoing).items():
      out[f'{tag}/{k}'] = v
  for tag, name in [('pair_attention1', 'tri_attn_start'), ('pair_attention2', 'tri_attn_end')]:
    for k, v in C.grid_attention(sd, f'{prefix}.{name}', D, pair_H, pair_D).items():
      out[f'{tag}/{k}'] = v
  for k, v in C.transition(sd, f'{prefix}.z_transition', D).items():
    out[f'pair_transition/{k}'] = v
  return out


def _rosettafold3_pairformer_block(sd, i, pair_H, pair_D, single_H, single_D):
  prefix = f'recycler.pairformer_stack.{i}'
  out = _rosettafold3_pair_block(sd, prefix, pair_H, pair_D)
  out.update(C.single_attention(sd, f'{prefix}.attention_pair_bias', D, single_H, single_D))
  out['single_attention_q_projection/bias'] = np.zeros((single_H, single_D), np.float32)
  for k, v in C.transition(sd, f'{prefix}.s_transition', D).items():
    out[f'single_transition/{k}'] = v
  return out


def map_pairformer_stack(sd, params, n_blocks=48, pair_H=4, pair_D=32, single_H=16, single_D=24):
  scope = 'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer'
  stacked = _stack_blocks(
      lambda i: _rosettafold3_pairformer_block(sd, i, pair_H, pair_D, single_H, single_D), n_blocks)
  _populate_scope(params, scope, stacked)


# ─── input embeddings + recycle + z-init ─────────────────────────────────────
# ─── RF3's residue alphabet is NOT OF3's: G and C are swapped ─────────────────
# atomworks (ml/encoding_definitions.py AF3_TOKENS) orders the nucleic classes
#   21 A, 22 C, 23 G, 24 U, 25 N, 26 DA, 27 DC, 28 DG, 29 DT, 30 DN, 31 GAP
# while OF3/protenix order them
#   21 A, 22 G, 23 C, 24 U, 25 N, 26 DA, 27 DG, 28 DC, 29 DT, 30 DN, 31 GAP
# -- identical everywhere EXCEPT that G/C and DG/DC are transposed. rf3 reused
# of3's remap, so every G was read as C and every C as G in RNA and DNA.
#
# Invisible to every gate this port has: protein indices (0-20) and ligands are
# untouched, so 6MRR (0.9-1.8 A), the biotin ligand ladder (0.099 A vs native
# 0.125) and 1STP+BTN (0.464/0.889) all passed while 1EHZ tRNA folded to 16.8 A
# against native rf3's 0.94 A on the same input.
#
# Verified against native's own featurised batch: for GCGGAUUUAGCU its `restype`
# argmax is [23 22 23 23 21 24 24 24 21 23 22 24], which decodes only under
# 21=A, 22=C, 23=G, 24=U.
_AF3_TO_RF3_AATYPE = np.array(
    list(range(21))            # 0-20 protein + UNK, identical
    + [31]                     # AF3 GAP -> rf3 31
    + [21, 23, 22, 24]         # A, G, C, U  (G and C swapped vs of3)
    + [26, 28, 27, 29]         # DA, DG, DC, DT  (likewise)
    + [25],                    # AF3 N -> rf3 UNKNOWN_RNA
    dtype=np.int32,
)
_AF3_TO_RF3_MSA = np.concatenate([_AF3_TO_RF3_AATYPE, [30]]).astype(np.int32)


def _rf3_target_feat(w):
  """449 -> 447 target_feat remap, with RF3's alphabet.

  Same layout surgery as of3's _reorder_target_feat_weights -- OF3 order
  [atom(384), aatype(32), profile(32), del_mean(1)] to AF3's
  [aatype(31), profile(31), del_mean(1), atom(384)] -- but through RF3's own
  permutation.
  """
  r = _AF3_TO_RF3_AATYPE
  return np.concatenate([w[384 + r], w[416 + r], w[448:449], w[0:384]], axis=0)


def _rf3_msa(w):
  """MSA embedder rows: [restype one-hot(32), has_deletion, deletion_value,
  is_paired]. Only the 32-class block is permuted; the tail passes through."""
  return np.concatenate([w[_AF3_TO_RF3_MSA], w[32:]], axis=0)


def map_input_embeddings(sd, params):
  """Recycle norms+proj, z-init (direct from s_inputs, 449->447 remap), rel-pos, token bond,
  MSA embedder. RF3 z-init is STANDARD AF3 (to_z_init_i/j direct) -- no zinit@sinit compose."""
  _rtf = _rf3_target_feat          # RF3's alphabet, not OF3's -- see above
  s = 'diffuser/evoformer'
  # recycle: process_zh = [LN, Linear] (pair), process_sh = [LN, Linear] (single)
  _set(params, f'{s}/prev_embedding_layer_norm', 'scale', _get(sd, 'recycler.process_zh.0.weight'))
  _set(params, f'{s}/prev_embedding_layer_norm', 'offset', _get(sd, 'recycler.process_zh.0.bias'))
  _set(params, f'{s}/prev_embedding', 'weights', _t(_get(sd, 'recycler.process_zh.1.weight')))
  _set(params, f'{s}/prev_single_embedding_layer_norm', 'scale', _get(sd, 'recycler.process_sh.0.weight'))
  _set(params, f'{s}/prev_single_embedding_layer_norm', 'offset', _get(sd, 'recycler.process_sh.0.bias'))
  _set(params, f'{s}/prev_single_embedding', 'weights', _t(_get(sd, 'recycler.process_sh.1.weight')))
  # single/pair init from s_inputs (target_feat), 449->447 remap.
  # RF3 z_init = to_z_init_i(S).unsqueeze(-3) + to_z_init_j(S).unsqueeze(-2), i.e.
  # Z[i,j] = to_z_init_i(S)[j] + to_z_init_j(S)[i] -> to_z_init_i is the COLUMN embedder
  # and to_z_init_j the ROW embedder. AF3's _seq_pair_embedding is Z[i,j]=left(tf)[i](row)
  # +right(tf)[j](col), so left_single <- to_z_init_j (row), right_single <- to_z_init_i (col).
  # (protenix/of3 name their z-init the other way, matching the forward directly -- no swap.)
  _set(params, f'{s}/single_activations', 'weights', _rtf(_t(_get(sd, 'feature_initializer.to_s_init.weight'))))
  _set(params, f'{s}/left_single', 'weights', _rtf(_t(_get(sd, 'feature_initializer.to_z_init_j.weight'))))
  _set(params, f'{s}/right_single', 'weights', _rtf(_t(_get(sd, 'feature_initializer.to_z_init_i.weight'))))
  # rel-pos + token bond
  _set(params, f'{s}/~_relative_encoding/position_activations', 'weights',
       _t(_get(sd, 'feature_initializer.relative_position_encoding.linear.weight')))
  _set(params, f'{s}/bond_embedding', 'weights', _t(_get(sd, 'feature_initializer.process_token_bonds.weight')))
  # MSA embedder: emb_msa (35-d MSA feat = 32 restype + has_del + del_val + is_paired)
  # + emb_S_inputs (449->447). The 32-class block is restype-indexed and takes the same
  # alphabet permutation as _rtf applies elsewhere here; the 3 trailing columns pass
  # through. It used to be left unpermuted ("no remap") -- silent, because the row count
  # matches either way and protein indices coincide, so only nucleics and gaps were wrong.
  _set(params, f'{s}/msa_activations', 'weights',
       _rf3_msa(_t(_get(sd, 'recycler.msa_module.msa_subsampler.emb_msa.weight'))))
  _set(params, f'{s}/extra_msa_target_feat', 'weights',
       _rtf(_t(_get(sd, 'recycler.msa_module.msa_subsampler.emb_S_inputs.weight'))))


# ─── MSA (SINGLE module -- RF3 does not stack MSA blocks) ─────────────────────
def _rosettafold3_msa_block(sd, i, msa_H, msa_D, pair_H, pair_D):
  """RF3 msa_module is a SINGLE (unstacked) module -- i ignored. Full pair stack:
  msa pair-weighted-avg + msa transition + OPM + tri_mult(+t)/tri_attn + pair_transition."""
  m = 'recycler.msa_module'
  out = {}
  for k, v in C.msa_attention(sd, f'{m}.msa_pair_weighted_averaging', D, msa_H, msa_D).items():
    out[f'msa_attention1/{k}'] = v
  for k, v in C.transition(sd, f'{m}.msa_transition', D).items():
    out[f'msa_transition/{k}'] = v
  for k, v in C.outer_product_mean(sd, f'{m}.outer_product', D, c_hidden=32, c_z=128,
                                        lr_bias=True).items():
    out[f'outer_product_mean:{k[2:]}' if k.startswith('::') else f'outer_product_mean/{k}'] = v
  # pair stack (RF3 msa uses tri_muLT -- extra 't' -- vs trunk's tri_mul)
  for tag, name, outgoing in [('triangle_multiplication_outgoing', 'tri_mult_outgoing', True),
                              ('triangle_multiplication_incoming', 'tri_mult_incoming', False)]:
    for k, v in C.triangle_mul(sd, f'{m}.{name}', D, outgoing=outgoing).items():
      out[f'{tag}/{k}'] = v
  for tag, name in [('pair_attention1', 'tri_attn_start'), ('pair_attention2', 'tri_attn_end')]:
    for k, v in C.grid_attention(sd, f'{m}.{name}', D, pair_H, pair_D).items():
      out[f'{tag}/{k}'] = v
  for k, v in C.transition(sd, f'{m}.pair_transition', D).items():
    out[f'pair_transition/{k}'] = v
  return out


def map_msa_stack(sd, params, n_blocks=1, msa_H=8, msa_D=32, pair_H=4, pair_D=32):
  scope = 'diffuser/evoformer/__layer_stack_no_per_layer/msa_stack'
  stacked = _stack_blocks(lambda i: _rosettafold3_msa_block(sd, i, msa_H, msa_D, pair_H, pair_D), n_blocks)
  regrouped = {}
  for k, v in stacked.items():
    if ':' in k and '/' not in k:
      sub, name = k.split(':', 1); regrouped[f'{sub}/{name}'] = v
    else:
      regrouped[k] = v
  _populate_scope(params, scope, regrouped)


# ─── distogram (RF3: 65 bins) ────────────────────────────────────────────────
def map_distogram_head(sd, params):
  _set(params, 'diffuser/distogram_head/half_logits', 'weights',
       _t(_get(sd, 'distogram_head.predictor.weight')))
  # HALVED, deliberately. RF3 symmetrises the pair features BEFORE the linear
  # (`predictor(Z + Z.transpose(-2, -3))`, RF3_structure.py:252), so its bias is
  # added once; our graph applies the linear first and then sums the two
  # orientations, which passes the bias through twice. b/2 makes the two agree
  # exactly and keeps this a weight transform rather than a forward branch.
  _set(params, 'diffuser/distogram_head/half_logits', 'bias',
       0.5 * _get(sd, 'distogram_head.predictor.bias'))


# ─── confidence head ─────────────────────────────────────────────────────────
def _rosettafold3_conf_block(sd, i, pair_H, pair_D, single_H, single_D):
  prefix = f'confidence_head.pairformer.{i}'
  out = _rosettafold3_pair_block(sd, prefix, pair_H, pair_D)
  out.update(C.single_attention(sd, f'{prefix}.attention_pair_bias', D, single_H, single_D))
  out['single_attention_q_projection/bias'] = np.zeros((single_H, single_D), np.float32)
  for k, v in C.transition(sd, f'{prefix}.s_transition', D).items():
    out[f'single_transition/{k}'] = v
  return out


def map_confidence_head(sd, params, n_layers=4, pair_H=4, pair_D=32, single_H=16, single_D=24):
  _rtf = _rf3_target_feat          # RF3's alphabet, not OF3's -- see above
  base = 'diffuser/confidence_head'
  _populate_scope(params, f'{base}/__layer_stack_no_per_layer/confidence_pairformer',
                  _stack_blocks(lambda i: _rosettafold3_conf_block(sd, i, pair_H, pair_D, single_H, single_D), n_layers))
  es = f'{base}/~_embed_features'
  _set(params, f'{es}/left_target_feat_project', 'weights', _rtf(_t(_get(sd, 'confidence_head.process_s_inputs_left.weight'))))
  _set(params, f'{es}/right_target_feat_project', 'weights', _rtf(_t(_get(sd, 'confidence_head.process_s_inputs_right.weight'))))
  # RF3 confidence distance feature = 40-bin (discretize CA-CA 3.25..50.75 / 39 -> one_hot 40);
  # the rf3 confidence forward branch feeds a 40-bin dgram, so keep all 40 bins.
  _set(params, f'{es}/distogram_feat_project', 'weights', _t(_get(sd, 'confidence_head.process_pred_distances.weight')))
  for tgt, src in [('logits_ln', 'layernorm_pde'), ('pae_logits_ln', 'layernorm_pae'),
                   ('plddt_logits_ln', 'layernorm_plddt'), ('experimentally_resolved_ln', 'layernorm_exp_resolved')]:
    _set(params, f'{base}/{tgt}', 'scale', _get(sd, f'confidence_head.{src}.weight'))
    _set(params, f'{base}/{tgt}', 'offset', _get(sd, f'confidence_head.{src}.bias'))
  # logit projections: pde/pae = linear(LN(z))->64 bins; plddt/exp_resolved = per-atom-type
  # (predict_plddt (1150,384)=23 atoms x 50 bins; predict_exp_resolved (46,384)=23 x 2).
  _set(params, f'{base}/left_half_distance_logits', 'weights', _t(_get(sd, 'confidence_head.predict_pde.weight')))
  _set(params, f'{base}/pae_logits', 'weights', _t(_get(sd, 'confidence_head.predict_pae.weight')))
  _set(params, f'{base}/plddt_logits', 'weights', _atom_logits(_get(sd, 'confidence_head.predict_plddt.weight'), 50))
  _set(params, f'{base}/experimentally_resolved_logits', 'weights',
       _atom_logits(_get(sd, 'confidence_head.predict_exp_resolved.weight'), 2))


def _atom_logits(w, bins, max_atoms=24):
  """(atoms*bins, c_s) torch -> (c_s, max_atoms, bins) haiku, atom-major, atom-padded."""
  atoms = w.shape[0] // bins
  logits = w.T.reshape(w.shape[1], atoms, bins)     # (c_s, atoms, bins) [atom-major assumed]
  if atoms < max_atoms:
    pad = np.zeros((w.shape[1], max_atoms - atoms, bins), dtype=logits.dtype)
    logits = np.concatenate([logits, pad], axis=1)
  return logits


# ─── template embedder (fused -> boltz2 scopes; weights only, feat builder = forward) ─
def _rosettafold3_template_pair_block(sd, i, templ_H, templ_D):
  return _rosettafold3_pair_block(sd, f'recycler.template_embedder.pairformer.{i}', templ_H, templ_D)


def map_template_embedder(sd, params, n_blocks=2, templ_H=4, templ_D=64):
  te = 'recycler.template_embedder'
  TE = 'diffuser/evoformer/template_embedding'
  _set(params, f'{TE}/z_norm', 'scale', _get(sd, f'{te}.norm_pair_before_pairformer.weight'))
  _set(params, f'{TE}/z_norm', 'offset', _get(sd, f'{te}.norm_pair_before_pairformer.bias'))
  _set(params, f'{TE}/v_norm', 'scale', _get(sd, f'{te}.norm_after_pairformer.weight'))
  _set(params, f'{TE}/v_norm', 'offset', _get(sd, f'{te}.norm_after_pairformer.bias'))
  _set(params, f'{TE}/z_proj', 'weights', _t(_get(sd, f'{te}.emb_pair.weight')))
  _set(params, f'{TE}/a_proj', 'weights', _t(_get(sd, f'{te}.emb_templ.weight')))
  _set(params, f'{TE}/u_proj', 'weights', _t(_get(sd, f'{te}.agg_emb.weight')))
  _populate_scope(params, f'{TE}/__layer_stack_no_per_layer',
                  _stack_blocks(lambda i: {f'tmpl_pairformer/{k}': v
                                           for k, v in _rosettafold3_template_pair_block(sd, i, templ_H, templ_D).items()}, n_blocks))



def _conformer_embedding_bias(sd, prefix, n_conformers=8):
  """Collapse RF3's ConformerEmbeddingWeightedAverage to the constant it emits.

  RF3 adds `process_atom_level_embedding(f['atom_level_embedding'])` to the atom
  single rep. Without conformer embeddings that input is all zeros -- but the MLP
  carries biases and its tail is a LayerNorm, so the output is a fixed NONZERO
  vector, identical for every atom (|x| ~ 0.31, two thirds of the ref-feature
  embedding). Dropping it as "inert" is what left our atom embeddings pointing the
  wrong way. Since the input is always zero here, the whole subtree is exactly this
  constant: push zeros through Linear+ReLU x4, tile to n_conformers, apply the
  bias-free Linear, then the LayerNorm. Dropout is inference-off.

  Returns the (c_atom,) vector, or None if the checkpoint lacks the subtree.
  """
  mlp = f'{prefix}.process_atom_level_embedding'
  head = f'{prefix}.conformers_to_atom_single_embedding'
  if not _has(sd, f'{head}.0.weight'):
    return None
  x = np.zeros(_get(sd, f'{mlp}.0.weight').shape[1], np.float32)
  for i, layer in enumerate((0, 3, 6, 9)):
    x = _get(sd, f'{mlp}.{layer}.weight') @ x + _get(sd, f'{mlp}.{layer}.bias')
    if layer != 9:                      # the last Linear has no ReLU after it
      x = np.maximum(x, 0.0)
  x = np.tile(x, n_conformers)
  x = _get(sd, f'{head}.0.weight') @ x
  if _has(sd, f'{head}.1.weight'):      # LayerNorm tail
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
    x = x * _get(sd, f'{head}.1.weight') + _get(sd, f'{head}.1.bias')
  return x.astype(np.float32)

def map_rosettafold3_to_af3(sd, *, n_pairformer=48, n_msa=4, n_template=2, n_confidence=4):
  """Convert RF3 (shadow/EMA) state dict -> AF3 haiku params. Trunk + heads + template
  (weights). Diffusion module is the next chunk (map_rf3_diffusion)."""
  params = {}
  map_input_embeddings(sd, params)
  map_msa_stack(sd, params, n_blocks=n_msa)
  map_pairformer_stack(sd, params, n_blocks=n_pairformer)
  map_confidence_head(sd, params, n_layers=n_confidence)
  map_distogram_head(sd, params)
  map_template_embedder(sd, params, n_blocks=n_template)
  map_rosettafold3_diffusion_conditioning_and_token(sd, params)
  map_rosettafold3_diffusion_atom(sd, params)
  map_rosettafold3_evoformer_conditioning(sd, params)
  # the conformer-embedding constant both atom encoders add (see
  # _conformer_embedding_bias); absent from the checkpoint -> leave at init (zeros)
  for pfx, scope, pname in (
      ('feature_initializer.input_feature_embedder.atom_attention_encoder',
       'diffuser', 'evoformer_conditioning_conformer_embedding_bias'),
      ('diffusion_module.atom_attention_encoder',
       'diffuser/~/diffusion_head', 'diffusion_conformer_embedding_bias')):
    bias = _conformer_embedding_bias(sd, f'{pfx}.process_atom_level_embedding')
    if bias is not None:
      params.setdefault(scope, {})[pname] = bias

  return params


# ─── diffusion: conditioning + token transformer (clean protenix-clone) ───────
# RF3 adaLN: ln_s (LN) + to_gain.0 (Linear+bias, sigmoid gate/scale) + to_bias (Linear, beta).
# Atom enc/dec path DEFERRED (RF3-specific conformer atom embedding; see memory).

def _rosettafold3_adaln(sd, prefix, name):
  return {
      f'{name}single_cond_layer_norm/scale': _get(sd, f'{prefix}.ln_s.weight'),
      f'{name}single_cond_scale/weights': _t(_get(sd, f'{prefix}.to_gain.0.weight')),
      f'{name}single_cond_scale/bias': _get(sd, f'{prefix}.to_gain.0.bias'),
      f'{name}single_cond_bias/weights': _t(_get(sd, f'{prefix}.to_bias.weight')),
  }


def _rosettafold3_diff_block(sd, i, prefix_base, name, H, D):
  """One RF3 diffusion (self-attn) transformer block -> of3 diffusion-block target tags.
  Includes kq_norm (query/key_layer_norm) params -- these need the rf3 forward branch."""
  pa = f'{prefix_base}.blocks.{i}.attention_pair_bias'
  ct = f'{prefix_base}.blocks.{i}.conditioned_transition_block'
  d = {}
  d.update(_rosettafold3_adaln(sd, f'{pa}.ada_ln_1', name))
  d['pair_input_layer_norm/scale'] = _get(sd, f'{pa}.ln_0.weight')
  d['pair_logits_projection/weights'] = _t(_get(sd, f'{pa}.to_b.weight'))
  d[f'{name}q_projection/weights'] = _get(sd, f'{pa}.to_q.weight').T.reshape(-1, H, D)
  d[f'{name}q_projection/bias'] = np.zeros((H, D), np.float32)   # RF3 to_q is no-bias
  d[f'{name}k_projection/weights'] = _get(sd, f'{pa}.to_k.weight').T.reshape(-1, H, D)
  d[f'{name}v_projection/weights'] = _get(sd, f'{pa}.to_v.weight').T.reshape(-1, H, D)
  d[f'{name}gating_query/weights'] = _t(_get(sd, f'{pa}.to_g.0.weight'))
  d[f'{name}transition2/weights'] = _t(_get(sd, f'{pa}.to_a.weight'))
  d[f'{name}adaptive_zero_cond/weights'] = _t(_get(sd, f'{pa}.linear_output_project.0.weight'))
  d[f'{name}adaptive_zero_cond/bias'] = _get(sd, f'{pa}.linear_output_project.0.bias')
  # kq_norm (RF3-only): LN on q and k pre-attention (needs rf3 forward branch + slots)
  d[f'{name}query_layer_norm/scale'] = _get(sd, f'{pa}.query_layer_norm.weight')
  d[f'{name}query_layer_norm/offset'] = _get(sd, f'{pa}.query_layer_norm.bias')
  d[f'{name}key_layer_norm/scale'] = _get(sd, f'{pa}.key_layer_norm.weight')
  d[f'{name}key_layer_norm/offset'] = _get(sd, f'{pa}.key_layer_norm.bias')
  # conditioned transition (swiglu linear_1/2 -> linear_3, + adaLN + output gate)
  d.update(_rosettafold3_adaln(sd, f'{ct}.ada_ln', f'{name}ffw_'))
  l1 = _get(sd, f'{ct}.linear_1.weight'); l2 = _get(sd, f'{ct}.linear_2.weight')
  d[f'{name}ffw_transition1/weights'] = np.concatenate([l1.T, l2.T], axis=-1)
  d[f'{name}ffw_transition2/weights'] = _t(_get(sd, f'{ct}.linear_3.weight'))
  d[f'{name}ffw_adaptive_zero_cond/weights'] = _t(_get(sd, f'{ct}.linear_output_project.0.weight'))
  d[f'{name}ffw_adaptive_zero_cond/bias'] = _get(sd, f'{ct}.linear_output_project.0.bias')
  return d


def _rosettafold3_cond_transition(sd, params, scope, prefix, name):
  """diffusion_conditioning transition (swiglu layer_norm_1/linear_1/2/3) -> ffw_* target."""
  tr = C.transition(sd, prefix, D)   # produces input_layer_norm/transition1/transition2
  _set(params, f'{scope}/{name}ffw_layer_norm', 'scale', tr['input_layer_norm/scale'])
  _set(params, f'{scope}/{name}ffw_layer_norm', 'offset', tr['input_layer_norm/offset'])
  _set(params, f'{scope}/{name}ffw_transition1', 'weights', tr['transition1/weights'])
  _set(params, f'{scope}/{name}ffw_transition2', 'weights', tr['transition2/weights'])


def map_rosettafold3_diffusion_conditioning_and_token(sd, params, *, n_token=24, n_super=6, diff_H=16, diff_D=48):
  """Diffusion conditioning + token transformer (clean). Atom enc/dec deferred."""
  dm = 'diffusion_module'; dc = f'{dm}.diffusion_conditioning'
  scope = 'diffuser/~/diffusion_head'
  # pair conditioning
  _set(params, f'{scope}/relpe_projection', 'weights', _t(_get(sd, f'{dc}.relative_position_encoding.linear.weight')))
  _set(params, f'{scope}/pair_cond_initial_norm', 'scale', _get(sd, f'{dc}.to_zii.0.weight'))
  _set(params, f'{scope}/pair_cond_initial_norm', 'offset', _get(sd, f'{dc}.to_zii.0.bias'))
  _set(params, f'{scope}/pair_cond_initial_projection', 'weights', _t(_get(sd, f'{dc}.to_zii.1.weight')))
  _rosettafold3_cond_transition(sd, params, scope, f'{dc}.transition_1.0', 'pair_transition_0')
  _rosettafold3_cond_transition(sd, params, scope, f'{dc}.transition_1.1', 'pair_transition_1')
  # single conditioning (to_si 833->384, remapped 833->831) + noise + fourier
  from .openfold3 import _reorder_features_1d as _of3_rf1
  # 831, not openfold3's 833: this graph's single_cond_initial_norm
  # spans the AF3-width block, so the two classes AF3 lacks are dropped.
  _rf1 = lambda a, **kw: _of3_rf1(a, pad_unk_dna=False, **kw)
  _set(params, f'{scope}/single_cond_initial_norm', 'scale', _rf1(_get(sd, f'{dc}.to_si.0.weight')))
  _set(params, f'{scope}/single_cond_initial_norm', 'offset', _rf1(_get(sd, f'{dc}.to_si.0.bias')))
  _set(params, f'{scope}/single_cond_initial_projection', 'weights', _rf1(_t(_get(sd, f'{dc}.to_si.1.weight'))))
  _set(params, f'{scope}/noise_embedding_initial_norm', 'scale', _get(sd, f'{dc}.process_n.0.weight'))
  _set(params, f'{scope}/noise_embedding_initial_norm', 'offset', _get(sd, f'{dc}.process_n.0.bias'))
  _set(params, f'{scope}/noise_embedding_initial_projection', 'weights', _t(_get(sd, f'{dc}.process_n.1.weight')))
  _set(params, scope, 'fourier_embedding_weight', _get(sd, f'{dc}.fourier_embedding.w').astype(np.float32))
  _set(params, scope, 'fourier_embedding_bias', _get(sd, f'{dc}.fourier_embedding.b').astype(np.float32))
  _rosettafold3_cond_transition(sd, params, scope, f'{dc}.transition_2.0', 'single_transition_0')
  _rosettafold3_cond_transition(sd, params, scope, f'{dc}.transition_2.1', 'single_transition_1')
  # token transformer (24 blocks -> nested (n_super, n_token//n_super))
  tr_base = f'{dm}.diffusion_transformer'; ss = n_token // n_super
  blocks = [_rosettafold3_diff_block(sd, i, tr_base, 'transformer', diff_H, diff_D) for i in range(n_token)]
  stacked = {}
  for k in blocks[0]:
    flat = np.stack([b[k] for b in blocks], axis=0)
    stacked[k] = flat.reshape((n_super, ss) + flat.shape[1:])
  _populate_scope(params, f'{scope}/transformer/__layer_stack_no_per_layer/__layer_stack_no_per_layer', stacked)
  # diffusion_module top-level: process_s = Seq[LN, Linear] (single cond embedding), layer_norm_1 = output_norm
  _set(params, f'{scope}/single_cond_embedding_norm', 'scale', _get(sd, f'{dm}.process_s.0.weight'))
  _set(params, f'{scope}/single_cond_embedding_norm', 'offset', _get(sd, f'{dm}.process_s.0.bias'))
  _set(params, f'{scope}/single_cond_embedding_projection', 'weights', _t(_get(sd, f'{dm}.process_s.1.weight')))
  _set(params, f'{scope}/output_norm', 'scale', _get(sd, f'{dm}.layer_norm_1.weight'))
  _set(params, f'{scope}/output_norm', 'offset', _get(sd, f'{dm}.layer_norm_1.bias'))


# ─── diffusion atom path (encoder/decoder + input-embedder atom encoder) ──────
# RF3 fuses all 1d atom features into ONE linear (process_input_features, 393-d):
#   [ref_pos(3), ref_charge(1), ref_mask(1), ref_element(128), ref_atom_name_chars(256),
#    ref_pos_ground_truth(3), has_atom_level_embedding(1)] = 393.
# AF3 has SEPARATE per-feature embedders that SUM -> equals a fused linear on the
# concatenation, so we split the fused (c_atom, 393) weight column-wise into AF3's
# embedders. The last 4 cols (ground-truth position leak + atom-level-embedding flag)
# have no AF3 slot and are DROPPED -- correct for de-novo folding (no gt-pos, flag=0,
# conformer atom-level embedding off).
def _rosettafold3_ref_embedders(sd, params, scope, enc, pfx):
  f = _get(sd, f'{enc}.process_input_features.weight')          # (c_atom, 393)
  _set(params, f'{scope}/{pfx}embed_ref_pos',       'weights', _t(f[:, 0:3]))
  _set(params, f'{scope}/{pfx}embed_ref_charge',    'weights', _t(f[:, 3:4]))
  _set(params, f'{scope}/{pfx}embed_ref_mask',      'weights', _t(f[:, 4:5]))
  _set(params, f'{scope}/{pfx}embed_ref_element',   'weights', _t(f[:, 5:133]))
  _set(params, f'{scope}/{pfx}embed_ref_atom_name', 'weights', _t(f[:, 133:389]))
  # pair features (+ the of3 "_1" duplicates that reuse the weight)
  for suf, leaf, dup in [('embed_pair_offsets', 'process_d', True),
                         ('embed_pair_distances', 'process_inverse_dist', True),
                         ('embed_pair_offsets_valid', 'process_valid_mask', False)]:
    w = _t(_get(sd, f'{enc}.{leaf}.weight'))
    _set(params, f'{scope}/{pfx}{suf}', 'weights', w)
    if dup:
      _set(params, f'{scope}/{pfx}{suf}_1', 'weights', w)
  for suf, leaf in [('single_to_pair_cond_row', 'process_single_l.1'),
                    ('single_to_pair_cond_col', 'process_single_m.1')]:
    w = _t(_get(sd, f'{enc}.{leaf}.weight'))
    _set(params, f'{scope}/{pfx}{suf}', 'weights', w)
    _set(params, f'{scope}/{pfx}{suf}_1', 'weights', w)
  _set(params, f'{scope}/{pfx}pair_mlp_1', 'weights', _t(_get(sd, f'{enc}.pair_mlp.1.weight')))
  _set(params, f'{scope}/{pfx}pair_mlp_2', 'weights', _t(_get(sd, f'{enc}.pair_mlp.3.weight')))
  _set(params, f'{scope}/{pfx}pair_mlp_3', 'weights', _t(_get(sd, f'{enc}.pair_mlp.5.weight')))
  _set(params, f'{scope}/{pfx}project_atom_features_for_aggr', 'weights',
       _t(_get(sd, f'{enc}.process_q.0.weight')))


def _rosettafold3_atom_block(sd, i, prefix_base, name, H=4, D=32):
  """One RF3 atom-transformer block -> AF3 CROSS-att per-block scopes. RF3's atom attn
  is SELF-attention (single ada_ln_1); AF3's cross-att applies adaLN to q and k
  separately, so we map RF3's single ada_ln_1 into BOTH the q and k adaLN slots
  (equivalent when q and k share the input, as in the atom encoder)."""
  pa = f'{prefix_base}.blocks.{i}.attention_pair_bias'
  ct = f'{prefix_base}.blocks.{i}.conditioned_transition_block'
  d = {}
  d.update(_rosettafold3_adaln(sd, f'{pa}.ada_ln_1', f'{name}q'))
  d.update(_rosettafold3_adaln(sd, f'{pa}.ada_ln_1', f'{name}k'))
  d['pair_input_layer_norm/scale'] = _get(sd, f'{pa}.ln_0.weight')
  d['pair_logits_projection/weights'] = _t(_get(sd, f'{pa}.to_b.weight'))
  d[f'{name}q_projection/weights'] = _get(sd, f'{pa}.to_q.weight').T.reshape(-1, H, D)
  d[f'{name}q_projection/bias'] = np.zeros((H, D), np.float32)     # RF3 to_q is no-bias
  d[f'{name}k_projection/weights'] = _get(sd, f'{pa}.to_k.weight').T.reshape(-1, H, D)
  d[f'{name}v_projection/weights'] = _get(sd, f'{pa}.to_v.weight').T.reshape(-1, H, D)
  d[f'{name}gating_query/weights'] = _t(_get(sd, f'{pa}.to_g.0.weight'))
  d[f'{name}transition2/weights'] = _t(_get(sd, f'{pa}.to_a.weight'))
  d[f'{name}adaptive_zero_cond/weights'] = _t(_get(sd, f'{pa}.linear_output_project.0.weight'))
  d[f'{name}adaptive_zero_cond/bias'] = _get(sd, f'{pa}.linear_output_project.0.bias')
  d[f'{name}query_layer_norm/scale'] = _get(sd, f'{pa}.query_layer_norm.weight')
  d[f'{name}query_layer_norm/offset'] = _get(sd, f'{pa}.query_layer_norm.bias')
  d[f'{name}key_layer_norm/scale'] = _get(sd, f'{pa}.key_layer_norm.weight')
  d[f'{name}key_layer_norm/offset'] = _get(sd, f'{pa}.key_layer_norm.bias')
  d.update(_rosettafold3_adaln(sd, f'{ct}.ada_ln', f'{name}ffw_'))
  l1 = _get(sd, f'{ct}.linear_1.weight'); l2 = _get(sd, f'{ct}.linear_2.weight')
  d[f'{name}ffw_transition1/weights'] = np.concatenate([l1.T, l2.T], axis=-1)
  d[f'{name}ffw_transition2/weights'] = _t(_get(sd, f'{ct}.linear_3.weight'))
  d[f'{name}ffw_adaptive_zero_cond/weights'] = _t(_get(sd, f'{ct}.linear_output_project.0.weight'))
  d[f'{name}ffw_adaptive_zero_cond/bias'] = _get(sd, f'{ct}.linear_output_project.0.bias')
  return d


def _rosettafold3_atom_transformer(sd, params, scope, base, name, n_blocks=3, H=4, D=32):
  """Stack RF3 atom-transformer blocks under the per-block (opendde-style) scope."""
  per = [_rosettafold3_atom_block(sd, i, base, name, H, D) for i in range(n_blocks)]
  stacked = {k: np.stack([b[k] for b in per], axis=0) for k in per[0]}
  _populate_scope(params, f'{scope}/__layer_stack_no_per_layer', stacked)


def map_rosettafold3_evoformer_conditioning(sd, params, *, n_atom=3, atom_H=4, atom_D=32):
  """Input-embedder atom encoder (feature_initializer) -> evoformer_conditioning_* scopes.
  No trunk/noisy-position cond (it computes s_inputs on the first pass)."""
  scope = 'diffuser'
  enc = 'feature_initializer.input_feature_embedder.atom_attention_encoder'
  _rosettafold3_ref_embedders(sd, params, scope, enc, 'evoformer_conditioning_')
  _rosettafold3_atom_transformer(sd, params, f'{scope}/evoformer_conditioning_atom_transformer_encoder',
                        f'{enc}.atom_transformer.diffusion_transformer',
                        'evoformer_conditioning_atom_transformer_encoder', n_atom, atom_H, atom_D)


def map_rosettafold3_diffusion_atom(sd, params, *, n_atom=3, atom_H=4, atom_D=32):
  """Diffusion atom encoder (ref embedders + trunk cond + noisy pos + atom transformer)
  and atom decoder -> diffusion_head diffusion_* scopes."""
  dm = 'diffusion_module'; scope = 'diffuser/~/diffusion_head'
  ae = f'{dm}.atom_attention_encoder'
  _rosettafold3_ref_embedders(sd, params, scope, ae, 'diffusion_')
  # Both trunk-conditioning norms are AFFINE in RF3 (`nn.LayerNorm(c_s)` /
  # `nn.LayerNorm(c_tokenpair)` inside process_s_trunk / process_z,
  # af3_diffusion_transformer.py:56-60), so each carries an offset the stock AF3
  # graph has no slot for. Dropping it is the same silent directional error the
  # chai/boltz note at atom_cross_attention.py:270 describes -- it survives the
  # affine-free LN downstream and reads as rounding.
  _set(params, f'{scope}/diffusion_embed_trunk_single_cond', 'weights', _t(_get(sd, f'{ae}.process_s_trunk.1.weight')))
  _set(params, f'{scope}/diffusion_lnorm_trunk_single_cond', 'scale', _get(sd, f'{ae}.process_s_trunk.0.weight'))
  _set(params, f'{scope}/diffusion_lnorm_trunk_single_cond', 'offset', _get(sd, f'{ae}.process_s_trunk.0.bias'))
  _set(params, f'{scope}/diffusion_embed_trunk_pair_cond', 'weights', _t(_get(sd, f'{ae}.process_z.1.weight')))
  _set(params, f'{scope}/diffusion_lnorm_trunk_pair_cond', 'scale', _get(sd, f'{ae}.process_z.0.weight'))
  _set(params, f'{scope}/diffusion_lnorm_trunk_pair_cond', 'offset', _get(sd, f'{ae}.process_z.0.bias'))
  _set(params, f'{scope}/diffusion_atom_positions_to_features', 'weights', _t(_get(sd, f'{ae}.process_r.weight')))
  # process_ch: the chirality-gradient embedder (use_chiral_features). Small next to
  # process_r (absmean 0.07 vs 2.14) -- it is zero-initialised and trained as a nudge
  # towards ideal tetrahedral geometry, not a primary coordinate path.
  if _has(sd, f'{ae}.process_ch.weight'):
    _set(params, f'{scope}/diffusion_atom_chiral_to_features', 'weights',
         _t(_get(sd, f'{ae}.process_ch.weight')))
  _rosettafold3_atom_transformer(sd, params, f'{scope}/diffusion_atom_transformer_encoder',
                        f'{ae}.atom_transformer.diffusion_transformer',
                        'diffusion_atom_transformer_encoder', n_atom, atom_H, atom_D)
  ad = f'{dm}.atom_attention_decoder'
  _set(params, f'{scope}/diffusion_project_token_features_for_broadcast', 'weights', _t(_get(sd, f'{ad}.linear_1.weight')))
  # to_r_update is `Sequential(nn.LayerNorm(c_atom), linearNoBias(c_atom, 3))`
  # (RF3_structure.py:40), so it too carries an offset. Its effect is a constant
  # 3-vector added to EVERY atom's position update, i.e. a rigid translation that
  # no superposed RMSD could ever see -- mapped anyway, because an exact
  # conversion is worth more than an argument about why a gap does not matter.
  _set(params, f'{scope}/diffusion_atom_features_layer_norm', 'scale', _get(sd, f'{ad}.to_r_update.0.weight'))
  _set(params, f'{scope}/diffusion_atom_features_layer_norm', 'offset', _get(sd, f'{ad}.to_r_update.0.bias'))
  _set(params, f'{scope}/diffusion_atom_features_to_position_update', 'weights', _t(_get(sd, f'{ad}.to_r_update.1.weight')))
  _rosettafold3_atom_transformer(sd, params, f'{scope}/diffusion_atom_transformer_decoder',
                        f'{ad}.atom_transformer.diffusion_transformer',
                        'diffusion_atom_transformer_decoder', n_atom, atom_H, atom_D)


def convert_rosettafold3_weights(checkpoint, output_dir):
  """Convert rf3_foundry_*.ckpt (EMA shadow weights) to a loadable AF3-haiku blob dir.

  Complete: trunk + MSA + template + confidence + distogram + diffusion (conditioning,
  token transformer, atom encoder/decoder). Structural gate is 0/0/0 vs the rf3-config
  graph. Confidence distance is 40-bin in RF3 (sliced to AF3's 39, confidence-only)."""
  sd = load_rf3_checkpoint(checkpoint)
  params = map_rosettafold3_to_af3(sd)
  return Path(C.write_params_blob(output_dir, 'rosettafold3.bin.zst',
                                  params, add_meta=True))
