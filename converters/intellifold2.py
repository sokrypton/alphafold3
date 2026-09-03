"""IntelliFold-v2 -> AF3-haiku weight converter (forward map, no schema).

IF2 is "AF3 except the hidden sizes" (the full_fat preset), so its module tree is
the DeepMind AF3 tree with wider channels. This converts the IF2 PyTorch checkpoint
straight to the haiku `.bin.zst` the AF3 loader reads, using the shared primitives in
common.py plus an IF2 dialect. It replaces the earlier schema-driven converter (which
read a pickled af3_schema.pkl); the correspondence now lives in code.

Validated bit-for-bit against IntelliGen's published intellifold_v2.bin.zst
(tests/test_af3.py::test_ifv2_converter_reproduces_the_published_weights).
"""

from __future__ import annotations

import os

import numpy as np

from . import common as C
from .common import DIALECT_INTELLIFOLD2 as D

# The __meta__/__identifier__ record the published blob carries (the af3.bin schema
# identifier). Fixed 64-byte ascii; reproduced so our output matches byte-for-byte.
_META_ID = 'f08da88ec964e3929b954fe6a042255eca5c52b3367b4f586577e622ac70125c'

# dims (full_fat)
_PAIR_H = 8          # triangle attention heads
_SINGLE_H, _SINGLE_D = 16, 24
_MSA_H, _MSA_VD = 8, 32
_ATOM_H, _ATOM_D = 4, 32
_DIFF_H, _DIFF_D = 16, 48
_N_TRUNK, _N_MSA, _N_TEMPL, _N_CONF = 48, 4, 2, 4
_N_ATOM, _N_DIFF, _N_SUPER = 3, 24, 6


def map_intellifold2_to_af3(sd):
  """Convert an IF2 state dict {name: ndarray} to AF3 haiku params {scope:{name:arr}}."""
  params = {}

  def S(scope, name, arr):
    params.setdefault(scope, {})[name] = arr

  def W(scope, ptkey):                       # a plain transposed linear weight
    S(scope, 'weights', C.t(sd[ptkey]))

  def LN(scope, ptprefix, scale_only=False):
    S(scope, 'scale', C._arr(sd[f'{ptprefix}.weight']))
    if not scale_only:
      S(scope, 'offset', C._arr(sd[f'{ptprefix}.bias']))

  EVO = 'diffuser/evoformer'
  DH = 'diffuser/~/diffusion_head'

  # --- input / recycling / msa embeddings ---------------------------------
  ie = 'backbone_trunk.input_embedder'
  W(f'{EVO}/single_activations', f'{ie}.linear_s_inputs.weight')
  W(f'{EVO}/left_single', f'{ie}.linear_z_i.weight')
  W(f'{EVO}/right_single', f'{ie}.linear_z_j.weight')
  W(f'{EVO}/bond_embedding', f'{ie}.linear_token_bonds.weight')
  W(f'{EVO}/~_relative_encoding/position_activations',
    f'{ie}.relative_position_encoding.linear_relpos.weight')
  W(f'{EVO}/msa_activations', 'backbone_trunk.msa_embedder.linear_mf.weight')
  W(f'{EVO}/extra_msa_target_feat', 'backbone_trunk.msa_embedder.linear_s_inputs.weight')
  re = 'backbone_trunk.recycling_embedder'
  W(f'{EVO}/prev_embedding', f'{re}.linear_z.weight')
  LN(f'{EVO}/prev_embedding_layer_norm', f'{re}.layer_norm_z')
  W(f'{EVO}/prev_single_embedding', f'{re}.linear_s.weight')
  LN(f'{EVO}/prev_single_embedding_layer_norm', f'{re}.layer_norm_s')
  W('diffuser/distogram_head/half_logits', 'backbone_trunk.aux_heads.distogram.linear.weight')

  # --- msa stack (4 blocks; the last is a "dead" pair-only block -- its MSA-update
  #     submodules are absent, guarded below, and stack_blocks zero-fills the slot) --
  def msa_block(i):
    b = f'backbone_trunk.msa_stack.blocks.{i}'
    out = C.nest('msa_stack', C.pair_block(sd, f'{b}.pair_stack', D, _PAIR_H, 64))
    out.update(C.nest('msa_stack/outer_product_mean',
                      C.outer_product_mean(sd, f'{b}.outer_product_mean', D)))
    if f'{b}.msa_pair_weighted_averaging.layer_norm_m.weight' in sd:
      out.update(C.nest('msa_stack/msa_attention1',
                        C.msa_attention(sd, f'{b}.msa_pair_weighted_averaging', D, _MSA_H, _MSA_VD)))
      out.update(C.nest('msa_stack/msa_transition', C.transition(sd, f'{b}.msa_transition', D)))
    return out
  C.populate(params, f'{EVO}/__layer_stack_no_per_layer', C.stack_blocks(msa_block, _N_MSA))

  # --- trunk pairformer (48 blocks) ----------------------------------------
  def pf_block(i):
    b = f'backbone_trunk.pairformer.blocks.{i}'
    return C.nest('trunk_pairformer', C.pairformer_block(
        sd, b, 'pair_stack', 'attention_pair_bias', D, _PAIR_H, 64, _SINGLE_H, _SINGLE_D))
  C.populate(params, f'{EVO}/__layer_stack_no_per_layer_1', C.stack_blocks(pf_block, _N_TRUNK))

  # --- template embedder ---------------------------------------------------
  te = 'backbone_trunk.template_embedder'
  ste = f'{EVO}/template_embedding/single_template_embedding'
  W(f'{EVO}/template_embedding/output_linear', f'{te}.linear_o.weight')
  LN(f'{ste}/query_embedding_norm', f'{te}.layer_norm_z')
  LN(f'{ste}/output_layer_norm', f'{te}.layer_norm_t')
  W(f'{ste}/template_pair_embedding_0', f'{te}.linear_d.weight')
  W(f'{ste}/template_pair_embedding_2', f'{te}.linear_aatype_col.weight')
  W(f'{ste}/template_pair_embedding_3', f'{te}.linear_aatype_row.weight')
  W(f'{ste}/template_pair_embedding_8', f'{te}.linear_z.weight')
  for idx, attr in [(1, 'linear_d_mask'), (4, 'linear_unit_vec_x'), (5, 'linear_unit_vec_y'),
                    (6, 'linear_unit_vec_z'), (7, 'linear_bb_mask')]:
    S(f'{ste}/template_pair_embedding_{idx}', 'weights',
      C._arr(sd[f'{te}.{attr}.weight']).squeeze(-1))

  def tmpl_block(i):
    return C.pair_block(sd, f'{te}.pairformer_stack.{i}', D, _PAIR_H, 32)
  C.populate(params, f'{ste}/__layer_stack_no_per_layer/template_embedding_iteration',
             C.stack_blocks(tmpl_block, _N_TEMPL))

  # --- confidence head -----------------------------------------------------
  cf = 'confidence_head'
  cfb = 'diffuser/confidence_head'
  def conf_block(i):
    b = f'{cf}.pairformer_stack.blocks.{i}'
    return C.nest('confidence_pairformer', C.pairformer_block(
        sd, b, 'pair_stack', 'attention_pair_bias', D, _PAIR_H, 64, _SINGLE_H, _SINGLE_D))
  C.populate(params, f'{cfb}/__layer_stack_no_per_layer', C.stack_blocks(conf_block, _N_CONF))
  W(f'{cfb}/~_embed_features/left_target_feat_project', f'{cf}.linear_s_inputs_col.weight')
  W(f'{cfb}/~_embed_features/right_target_feat_project', f'{cf}.linear_s_inputs_row.weight')
  W(f'{cfb}/~_embed_features/distogram_feat_project', f'{cf}.linear_d.weight')
  for head, ln_scope, log_scope in [
      ('pae_head', 'pae_logits_ln', 'pae_logits'),
      ('pde_head', 'logits_ln', 'left_half_distance_logits'),
      ('plddt_head', 'plddt_logits_ln', 'plddt_logits'),
      ('resolved_head', 'experimentally_resolved_ln', 'experimentally_resolved_logits')]:
    LN(f'{cfb}/{ln_scope}', f'{cf}.{head}.layer_norm')
  # pae/pde logits are plain transposes; plddt/resolved are per-atom (T_reshape -> (c_s, atoms, bins))
  W(f'{cfb}/pae_logits', f'{cf}.pae_head.linear.weight')
  W(f'{cfb}/left_half_distance_logits', f'{cf}.pde_head.linear.weight')
  for head, scope, bins in [('plddt_head', 'plddt_logits', 50),
                            ('resolved_head', 'experimentally_resolved_logits', 2)]:
    w = C._arr(sd[f'{cf}.{head}.linear.weight'])          # (atoms*bins, c_s)
    atoms = w.shape[0] // bins
    S(f'{cfb}/{scope}', 'weights', np.ascontiguousarray(w.T).reshape(-1, atoms, bins))

  # --- evoformer conditioning (trunk atom encoder) -------------------------
  ae = 'backbone_trunk.input_embedder.atom_attention_encoder'
  _atom_embed(sd, params, ae, 'diffuser/evoformer_conditioning_', S, W)
  _atom_transformer(sd, params, f'{ae}.atom_transformer',
                    'diffuser/evoformer_conditioning_atom_transformer_encoder',
                    'evoformer_conditioning_atom_transformer_encoder', cross=True)

  # --- diffusion head ------------------------------------------------------
  dm = 'diffusion_module'
  # atom encoder feature embedders + trunk conditioning
  aenc = f'{dm}.atom_attention_encoder'
  _atom_embed(sd, params, aenc, f'{DH}/diffusion_', S, W, sep='')
  W(f'{DH}/diffusion_embed_trunk_pair_cond', f'{aenc}.linear_z.weight')
  W(f'{DH}/diffusion_embed_trunk_single_cond', f'{aenc}.linear_s.weight')
  LN(f'{DH}/diffusion_lnorm_trunk_pair_cond', f'{aenc}.layer_norm_z', scale_only=True)
  LN(f'{DH}/diffusion_lnorm_trunk_single_cond', f'{aenc}.layer_norm_s', scale_only=True)
  W(f'{DH}/diffusion_atom_positions_to_features', f'{aenc}.linear_r.weight')
  # atom decoder
  adec = f'{dm}.atom_attention_decoder'
  LN(f'{DH}/diffusion_atom_features_layer_norm', f'{adec}.layer_norm_q', scale_only=True)
  W(f'{DH}/diffusion_atom_features_to_position_update', f'{adec}.linear_q.weight')
  W(f'{DH}/diffusion_project_token_features_for_broadcast', f'{adec}.linear_a.weight')
  # atom transformers (enc/dec, cross-attn)
  _atom_transformer(sd, params, f'{aenc}.atom_transformer',
                    f'{DH}/diffusion_atom_transformer_encoder',
                    'diffusion_atom_transformer_encoder', cross=True)
  _atom_transformer(sd, params, f'{adec}.atom_transformer',
                    f'{DH}/diffusion_atom_transformer_decoder',
                    'diffusion_atom_transformer_decoder', cross=True)
  # diffusion conditioning
  dc = f'{dm}.diffusion_conditioning'
  LN(f'{DH}/pair_cond_initial_norm', f'{dc}.layer_norm_z', scale_only=True)
  W(f'{DH}/pair_cond_initial_projection', f'{dc}.linear_z.weight')
  LN(f'{DH}/single_cond_initial_norm', f'{dc}.layer_norm_s', scale_only=True)
  W(f'{DH}/single_cond_initial_projection', f'{dc}.linear_s.weight')
  LN(f'{DH}/noise_embedding_initial_norm', f'{dc}.layer_norm_f', scale_only=True)
  W(f'{DH}/noise_embedding_initial_projection', f'{dc}.linear_f.weight')
  for i in (0, 1):
    LN(f'{DH}/pair_transition_{i}ffw_layer_norm', f'{dc}.pair_transitions.{i}.layer_norm')
    W(f'{DH}/pair_transition_{i}ffw_transition1', f'{dc}.pair_transitions.{i}.linear.weight')
    W(f'{DH}/pair_transition_{i}ffw_transition2', f'{dc}.pair_transitions.{i}.linear_o.weight')
    LN(f'{DH}/single_transition_{i}ffw_layer_norm', f'{dc}.single_transitions.{i}.layer_norm')
    W(f'{DH}/single_transition_{i}ffw_transition1', f'{dc}.single_transitions.{i}.linear.weight')
    W(f'{DH}/single_transition_{i}ffw_transition2', f'{dc}.single_transitions.{i}.linear_o.weight')
  LN(f'{DH}/single_cond_embedding_norm', f'{dm}.proj_s.0', scale_only=True)
  W(f'{DH}/single_cond_embedding_projection', f'{dm}.proj_s.1.weight')
  LN(f'{DH}/output_norm', f'{dm}.layer_norm_a', scale_only=True)
  # main diffusion transformer (24 blocks, nested super=6)
  dt = f'{dm}.diffusion_transformer'
  tr = f'{DH}/transformer'
  LN(f'{tr}/pair_input_layer_norm', f'{dt}.layer_norm_z', scale_only=True)
  def dtr_block(i):
    return _rekey(C.diff_attn_block(sd, f'{dt}.blocks.{i}', D, _DIFF_H, _DIFF_D, cross=False),
                  'transformer')
  C.populate(params, f'{tr}/__layer_stack_with_per_layer/__layer_stack_with_per_layer',
             C.stack_super(dtr_block, _N_DIFF, _N_SUPER))
  S(f'{tr}/__layer_stack_with_per_layer/pair_logits_projection', 'weights',
    _pair_logits_nested(sd, dt, _N_SUPER, _N_DIFF // _N_SUPER))

  # Fourier noise embedding (trained; travels in the blob like OF3's)
  for scope, name, arr in fourier_records(sd):
    S(scope, name, arr)
  return params


# --- helpers -----------------------------------------------------------------

def _rekey(block, stackname):
  """diff_attn_block 'X/param' -> '{stackname}{X}::param' for populate under a stack."""
  out = {}
  for k, v in block.items():
    sub, name = k.rsplit('/', 1)
    out[f'{stackname}{sub}::{name}'] = v
  return out


def _atom_embed(sd, params, base, prefix, S, W, sep='_'):
  """The per-atom reference-feature + pair MLP linears shared by every atom encoder.

  prefix already ends in the family tag (e.g. 'diffuser/evoformer_conditioning_' or
  'diffuser/~/diffusion_head/diffusion_'); each name is appended directly.
  """
  W(f'{prefix}embed_ref_pos', f'{base}.linear_ref_pos.weight')
  W(f'{prefix}embed_ref_mask', f'{base}.linear_ref_mask.weight')
  W(f'{prefix}embed_ref_element', f'{base}.linear_ref_element.weight')
  W(f'{prefix}embed_ref_charge', f'{base}.linear_ref_charge.weight')
  W(f'{prefix}embed_ref_atom_name', f'{base}.linear_ref_atom_name_chars.weight')
  W(f'{prefix}embed_pair_offsets', f'{base}.linear_d.weight')
  W(f'{prefix}embed_pair_offsets_1', f'{base}.linear_d.weight')
  W(f'{prefix}embed_pair_distances', f'{base}.linear_d_inv.weight')
  W(f'{prefix}embed_pair_distances_1', f'{base}.linear_d_inv.weight')
  W(f'{prefix}embed_pair_offsets_valid', f'{base}.linear_v.weight')
  W(f'{prefix}single_to_pair_cond_row', f'{base}.linear_c_row.weight')
  W(f'{prefix}single_to_pair_cond_row_1', f'{base}.linear_c_row.weight')
  W(f'{prefix}single_to_pair_cond_col', f'{base}.linear_c_col.weight')
  W(f'{prefix}single_to_pair_cond_col_1', f'{base}.linear_c_col.weight')
  W(f'{prefix}pair_mlp_1', f'{base}.linear_mlp_p_1.weight')
  W(f'{prefix}pair_mlp_2', f'{base}.linear_mlp_p_2.weight')
  W(f'{prefix}pair_mlp_3', f'{base}.linear_mlp_p_3.weight')
  W(f'{prefix}project_atom_features_for_aggr', f'{base}.pool_q.linear_q.weight')


def _atom_transformer(sd, params, base, scope, stackname, cross):
  """A 3-block atom transformer stack + its bare pair LN and pair-logits projection."""
  def blk(i):
    return _rekey(C.diff_attn_block(sd, f'{base}.blocks.{i}', D, _ATOM_H, _ATOM_D, cross=cross),
                  stackname)
  C.populate(params, f'{scope}/__layer_stack_with_per_layer', C.stack_blocks(blk, _N_ATOM))
  params.setdefault(f'{scope}/pair_input_layer_norm', {})['scale'] = C._arr(sd[f'{base}.layer_norm_z.weight'])
  params.setdefault(f'{scope}/pair_logits_projection', {})['weights'] = _pair_logits_flat(sd, base, _N_ATOM)


def _pair_logits_flat(sd, base, n):
  """Atom transformer pair bias: stack per-block linear_z^T -> (c_pair, n_blocks, H)."""
  return np.stack([C.t(sd[f'{base}.blocks.{j}.attention_pair_bias.linear_z.weight'])
                   for j in range(n)], axis=1)


def _pair_logits_nested(sd, base, n_super, inner):
  """Main diffusion transformer pair bias -> (n_super, c, inner, H)."""
  a = np.array([[C.t(sd[f'{base}.blocks.{g * inner + j}.attention_pair_bias.linear_z.weight'])
                 for j in range(inner)] for g in range(n_super)])   # (super, inner, c, H)
  return a.transpose(0, 2, 1, 3)


# --- fourier + checkpoint io (public; tests import these) ---------------------

_FOURIER_SCOPE = 'diffuser/~/diffusion_head'


def fourier_records(sd):
  """IF2's trained Fourier noise embedding as (scope, name, arr) records."""
  w = b = None
  for k, v in sd.items():
    if k.endswith('fourier_embedding.weight'):
      w = np.ravel(C._arr(v)).astype('float32')
    if k.endswith('fourier_embedding.bias'):
      b = np.ravel(C._arr(v)).astype('float32')
  assert w is not None and b is not None, 'fourier_embedding.{weight,bias} not found'
  return [(_FOURIER_SCOPE, 'fourier_embedding_weight', w),
          (_FOURIER_SCOPE, 'fourier_embedding_bias', b)]


def load_pt(path):
  """Load a .pt checkpoint -> {name: float32 ndarray}."""
  import torch
  sd = torch.load(os.path.expanduser(str(path)), map_location='cpu', weights_only=True)
  return {k: v.detach().cpu().float().numpy() for k, v in sd.items()}


read_bin = C.read_blob

# Trunk-region records that are float32 despite being weights: the aux-head output
# logits. Everything else in the trunk region is bfloat16 (layernorms excepted); the
# diffusion region is entirely float32. This mirrors AF3's param dtype policy, which
# the published blob follows.
_F32_TRUNK_WEIGHTS = {
    'diffuser/distogram_head/half_logits',
    'diffuser/confidence_head/pae_logits',
    'diffuser/confidence_head/left_half_distance_logits',
    'diffuser/confidence_head/plddt_logits',
    'diffuser/confidence_head/experimentally_resolved_logits',
}


def _record_dtype(scope, name):
  import ml_dtypes
  if 'diffusion_head' in scope or 'evoformer_conditioning' in scope:
    return np.float32
  if name in ('scale', 'offset') or scope in _F32_TRUNK_WEIGHTS:
    return np.float32
  return ml_dtypes.bfloat16


def convert_intellifold2_weights(checkpoint, output_dir):
  """Convert an IF2 .pt to a loadable AF3-haiku dir (intellifold2.bin.zst).

  Writes the trained Fourier folded in (no sidecar). Returns output_dir.
  """
  import zstandard
  from alphafold3.model.params import encode_record
  sd = load_pt(checkpoint)
  params = map_intellifold2_to_af3(sd)
  out_dir = os.path.expanduser(str(output_dir))
  os.makedirs(out_dir, exist_ok=True)
  # Named for the model, like every other converter here: the registry and the
  # published repo address it as <model>.bin.zst, and a file called something
  # else only works while someone renames it by hand.
  out_path = os.path.join(out_dir, 'intellifold2.bin.zst')
  with zstandard.ZstdCompressor(level=10).stream_writer(open(out_path, 'wb')) as comp:
    comp.write(encode_record('__meta__', '__identifier__',
                             np.frombuffer(_META_ID.encode('ascii'), dtype=np.uint8)))
    for scope in sorted(params):
      for name in sorted(params[scope]):
        arr = np.asarray(params[scope][name], dtype=_record_dtype(scope, name))
        comp.write(encode_record(scope, name, arr))
  return output_dir
