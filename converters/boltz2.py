"""Boltz-2 -> AF3-haiku weight converter (forward map, reuses converters/common).

Boltz-2 (github jwohlwend/boltz, MIT) is an independent AF3-family PyTorch reimpl
with its own Pairformer / triangle / diffusion primitives -- archetype 2, the same
class as OpenDDE. Per-residue tokenization (NO structural sub-token stage), EDM
diffusion, a trained-but-frozen Fourier noise embedding, and a Boltz-specific
affinity head (separate checkpoint, out of scope here). This module ports the
`boltz2_conf.ckpt` (confidence checkpoint) trunk + diffusion into our vendored AF3
haiku graph, reusing common.py's dialect-parameterized primitives.

Status: WORKING. Folds 6MRR to ~0.95 A CA-RMSD from single sequence (native Boltz-2 ~0.88 A),
self-contained on colabdesign2's featurise_spec. Ports the full structure path (input embedder,
trunk pairformer/msa, diffusion conditioners + atom encoder/decoder + token transformer, EDM
sampler), the template head and the confidence head, via the boltz2 forward branches
(global_config.model == 'boltz2'). The affinity head is NOT ported (separate checkpoint).
See boltz2-port.md for the 5 diffusion
bug fixes (atom encoder q!=c, per-block proj_z LN-bake, rel-pos, conditioner/atom LN offsets).
"""

from __future__ import annotations

import numpy as np

from . import common as C

# Boltz-2 dialect for the shared primitives (leaf names from BOLTZ2_RECON.md §2):
# - triangle attention matches the common/IF2 default (layer_norm / linear / mha.linear_*);
# - tri-mul is a single fused Linear(dim, 2*dim) per gate, chunk-split into BLOCK halves
#   (tm_fused='block' -> split + interleave, the crux vs our interleave-reading haiku);
# - transitions are SwiGLU (block-fused fc1/fc2 -> fc3), like OpenDDE's block mode;
# - single attention uses proj_q/k/v/g/o + proj_z.0(LN)/proj_z.1, with the s-LayerNorm
#   (pre_norm_s) living at the block scope; OPM output built from proj_o (not direct).
DIALECT_BOLTZ2 = C.Dialect(
    # transition (SwiGLU: silu(fc1)*fc2 -> fc3)
    tr_mode='block', tr_ln='norm', tr_a='fc1', tr_b='fc2', tr_out='fc3',
    # triangle multiplication (fused p_in/g_in, block-split)
    tm_fused='block', tm_ln_in='norm_in', tm_ln_out='norm_out',
    tm_ab_p='p_in', tm_ab_g='g_in', tm_z='p_out', tm_g='g_out',
    # triangle attention == common default (ga_ln='layer_norm', ga_bias='linear', ga_mha='mha')
    # single attention (pre_norm_s at block scope; proj_* under 'attention')
    sa_ln_a='pre_norm_s', sa_mha='attention',
    sa_q='proj_q', sa_k='proj_k', sa_v='proj_v', sa_g='proj_g', sa_o='proj_o',
    sa_ln_z='attention.proj_z.0', sa_z='attention.proj_z.1',
    # MSA pair-weighted averaging
    msa_ln_m='norm_m', msa_ln_z='norm_z', msa_z='proj_z',
    msa_v='proj_m', msa_g='proj_g', msa_o='proj_o',
    # outer product mean (output from proj_o, reshaped -- not stored direct)
    opm_out_direct=False, opm_ln='norm', opm_l='proj_a', opm_r='proj_b', opm_out='proj_o',
    # adaptive layer norm (diffusion)
    ada_lns='s_norm', ada_gamma='s_scale', ada_beta='s_bias',
)

D = DIALECT_BOLTZ2

# Boltz 33-token vocab (const.tokens) -> our 31-class POLYMER_TYPES_WITH_UNKNOWN_AND_GAP.
# PERM[our_idx] = boltz_idx, so a one-hot weight column reorders as w[:, PERM]. Verified
# by name: drops Boltz <pad>(0) and DN(32); our single nucleic-unk 'N'(30) <- Boltz N(27).
BOLTZ2_RESTYPE_PERM = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                       22, 1, 23, 24, 25, 26, 28, 29, 30, 31, 27]


def remap_restype_cols(w, start=0, n=33):
  """Reorder a weight's Boltz restype input columns [start:start+33] to our 31 classes."""
  import numpy as np
  w = np.asarray(w)
  cols = list(range(start)) + [start + i for i in BOLTZ2_RESTYPE_PERM] + list(range(start + n, w.shape[1]))
  return w[:, cols]


# dims (from hyper_parameters / BOLTZ2_RECON.md §1)
_PAIR_H, _PAIR_D = 4, 32          # triangle attention
_SINGLE_H, _SINGLE_D = 16, 24     # trunk single attention (c_s=384)
_MSA_H, _MSA_VD = 8, 32
_ATOM_H, _ATOM_D = 4, 32          # atom transformers (dim 128)
_DIFF_H, _DIFF_D = 16, 48         # token transformer (dim 768)
_N_TRUNK, _N_MSA, _N_TEMPL, _N_CONF = 64, 4, 2, 8
_N_DIFF, _N_ATOM = 24, 3


def pair_block(sd, prefix, pair_H=_PAIR_H, pair_D=_PAIR_D):
  """The pair-only stack shared by pairformer / msa / template blocks.

  prefix is the block scope; submodules are tri_mul_out/in, tri_att_start/end,
  transition_z (Boltz names). Returns flat keys under the block root.
  """
  out = {}
  out.update(C.nest('triangle_multiplication_outgoing',
                    C.triangle_mul(sd, C._pfx(prefix, 'tri_mul_out'), D, outgoing=True)))
  out.update(C.nest('triangle_multiplication_incoming',
                    C.triangle_mul(sd, C._pfx(prefix, 'tri_mul_in'), D, outgoing=False)))
  out.update(C.nest('pair_attention1', C.grid_attention(sd, C._pfx(prefix, 'tri_att_start'), D, pair_H, pair_D)))
  out.update(C.nest('pair_attention2', C.grid_attention(sd, C._pfx(prefix, 'tri_att_end'), D, pair_H, pair_D)))
  out.update(C.nest('pair_transition', C.transition(sd, C._pfx(prefix, 'transition_z'), D)))
  return out


def pairformer_block(sd, prefix, pair_H=_PAIR_H, pair_D=_PAIR_D,
                     single_H=_SINGLE_H, single_D=_SINGLE_D):
  """A full Boltz PairformerLayer: pair stack + single attention + single transition.

  pre_norm_s (the s LayerNorm) is at the block scope and single_attention reads it via
  sa_ln_a; the q/k/v/g/o + pair-bias projections live under `{prefix}.attention`.
  """
  out = pair_block(sd, prefix, pair_H, pair_D)
  out.update(C.single_attention(sd, prefix, D, single_H, single_D))
  out.update(C.nest('single_transition', C.transition(sd, C._pfx(prefix, 'transition_s'), D)))
  return out


def map_boltz2_to_af3(sd):
  """Assemble a Boltz-2 state dict into AF3 haiku params (the full STRUCTURE path).
  Covers: trunk pairformer/msa stacks, input embedder (summed s_inputs) + atom encoder,
  diffusion conditioners + atom encoder/decoder + token transformer, recycling, distogram.
  Folds 6MRR to ~0.95 A. Template, confidence and msa_activations are ALL mapped; the only
  thing left at model init is the affinity head (separate checkpoint).
  """
  params = {}
  map_trunk_stacks(sd, params)
  map_diffusion_transformers(sd, params)
  map_diffusion_conditioners(sd, params)
  map_input_embedder(sd, params)
  map_atom_encoder(sd, params)
  map_diffusion_atom_conditioning(sd, params)
  map_clean_oneoffs(sd, params)
  map_template(sd, params)
  map_confidence(sd, params)
  # Not mapped (left at model init): the affinity head only. The confidence head's branches
  # (token-level plddt/resolved, 64-bin distances, no per-head LayerNorm, the z
  # re-embedding) are implemented; the inter PAE/PDE heads still are not, which makes
  # confidence exact on monomers and approximate on complexes.
  return params


def _plain_transition(sd, prefix, name):
  """Boltz plain Transition (layers/transition.py: norm(w+b) -> silu(fc1)*fc2 -> fc3, no
  adaln, no a_to_b) -> our transition_block ffw_* params for the single_cond=None case
  (diffusion single/pair conditioners). No SwiGLU half-swap (plain silu(fc1)*fc2 already
  has a=fc1 in split[0], unlike ConditionedTransitionBlock which needed the swap)."""
  return {
      f'{name}ffw_layer_norm/scale': C._arr(sd[f'{prefix}.norm.weight']),
      f'{name}ffw_layer_norm/offset': C._arr(sd[f'{prefix}.norm.bias']),
      f'{name}ffw_transition1/weights': np.concatenate(
          [C.t(sd[f'{prefix}.fc1.weight']), C.t(sd[f'{prefix}.fc2.weight'])], axis=1),
      f'{name}ffw_transition2/weights': C.t(sd[f'{prefix}.fc3.weight']),
  }


def map_diffusion_conditioners(sd, params):
  """Map the diffusion single/pair conditioners into the diffusion_head (_conditioning is
  @hk.transparent, so params sit directly under diffusion_head/). Covers the shape-stable
  pieces; the INITIAL projections (single_cond_initial_* / pair_cond_initial_*) are deferred
  -- their input dims depend on target_feat=s_inputs(384) and Boltz rel-pos(128), which come
  from the Stage 3 input-embedder / rel-pos forward branches (AF3's featurise gives 831/267).

  NOTE (minor, documented): several AF3 LayerNorms here are create_offset=False while the
  Boltz LNs (norm_fourier, s_to_a_linear.0, a_norm) carry a bias. The dropped offset adds a
  per-channel constant to a per-structure/per-token-uniform tensor -> a small constant shift;
  accepted for a first fold, exact only with a create_offset branch.
  """
  DH = 'diffuser/~/diffusion_head'
  sm = 'structure_module.score_model'
  sc = f'{sm}.single_conditioner'
  pw = 'diffusion_conditioning.pairwise_conditioner'
  def P(local):
    C.populate(params, DH, local)
  # single conditioner transitions (dim 768) and pair conditioner transitions (dim 128)
  for i in range(2):
    P(_plain_transition(sd, f'{sc}.transitions.{i}', f'single_transition_{i}'))
    P(_plain_transition(sd, f'{pw}.transitions.{i}', f'pair_transition_{i}'))
  # noise / Fourier: trained_fourier weight+bias from Boltz's fixed proj; norm_fourier ->
  # noise_embedding_initial_norm (offset dropped); fourier_to_single -> projection.
  params.setdefault(DH, {})['fourier_embedding_weight'] = C._arr(sd[f'{sc}.fourier_embed.proj.weight'][:, 0])
  params.setdefault(DH, {})['fourier_embedding_bias'] = C._arr(sd[f'{sc}.fourier_embed.proj.bias'])
  # boltz2 keeps the conditioner LN OFFSETS (create_offset gated on boltz2 in diffusion_head):
  # these LNs feed non-softmax paths (Linear->transitions/added), so the bias matters (unlike the
  # per-block pair-bias LNs where it cancels). Dropping them capped COND pair at 0.947.
  P({'noise_embedding_initial_norm/scale': C._arr(sd[f'{sc}.norm_fourier.weight']),
     'noise_embedding_initial_norm/offset': C._arr(sd[f'{sc}.norm_fourier.bias'])})
  P({'noise_embedding_initial_projection/weights': C.t(sd[f'{sc}.fourier_to_single.weight'])})
  # a = a + s_to_a_linear(s): s_to_a_linear.0 (LN) -> single_cond_embedding_norm (offset
  # dropped), .1 (Linear 768->768) -> single_cond_embedding_projection.
  P({'single_cond_embedding_norm/scale': C._arr(sd[f'{sm}.s_to_a_linear.0.weight']),
     'single_cond_embedding_norm/offset': C._arr(sd[f'{sm}.s_to_a_linear.0.bias'])})
  P({'single_cond_embedding_projection/weights': C.t(sd[f'{sm}.s_to_a_linear.1.weight'])})
  # a_norm (post token transformer) -> output_norm (keep offset for boltz2).
  P({'output_norm/scale': C._arr(sd[f'{sm}.a_norm.weight']),
     'output_norm/offset': C._arr(sd[f'{sm}.a_norm.bias'])})
  return params


def convert_boltz2_weights(checkpoint, output_dir):
  """Convert a Boltz-2 .ckpt to an AF3-haiku dir (boltz2.bin.zst). Writes the full structure
  path (map_boltz2_to_af3); the affinity head is not ported and stays at the
  model's init values (they aren't needed for coordinates). The fold path (AF3Runner /
  test_boltz2_fold) uses init+overlay so the unported heads are init-filled."""
  import torch
  import os
  ck = torch.load(os.path.expanduser(str(checkpoint)), map_location='cpu', weights_only=False)
  sd = {k: v.numpy() for k, v in ck['state_dict'].items() if hasattr(v, 'numpy')}
  C.write_params_blob(output_dir, 'boltz2.bin.zst', map_boltz2_to_af3(sd))
  return output_dir


def map_input_embedder(sd, params):
  """Boltz-2 input embedder (trunkv2.InputEmbedder) -> the boltz2 s_inputs branch scopes.
  s_inputs = a + res_type_encoding(res) + msa_profile_encoding([profile,deletion]); s_init/
  z_init_1/z_init_2 then build the trunk single/pair from s_inputs. res_type/profile are
  Boltz's 33-class; our aatype/profile one-hots are 31-class, so remap the weight IN columns
  33->31 (remap_restype_cols). The atom-encoder `a` (evoformer_conditioning_* + atom
  transformer) is mapped separately (Stage 4). Also maps the diffusion single conditioner's
  INITIAL projection, whose input is now 768 (concat[s_trunk(384), s_inputs(384)])."""
  EVO = 'diffuser/evoformer'
  DH = 'diffuser/~/diffusion_head'
  def S(scope, name, arr): params.setdefault(scope, {})[name] = arr
  # summed input embedder: res_type_encoding (33->31) and msa_profile_encoding (33+1 -> 31+1)
  S('diffuser/boltz2_res_type_encoding', 'weights',
    C.t(remap_restype_cols(sd['input_embedder.res_type_encoding.weight'])))            # (31,384)
  S('diffuser/boltz2_msa_profile_encoding', 'weights',
    C.t(remap_restype_cols(sd['input_embedder.msa_profile_encoding.weight'], start=0, n=33)))  # (32,384)
  # conditioning inits: Boltz nn.Embedding weight (num_cls, seq) == our Linear-over-one-hot
  # weight (in=num_cls, out=seq) directly (Embedding[i]==one_hot(i)@W), so NO transpose.
  S('diffuser/boltz2_mol_type_conditioning', 'weights',
    C._arr(sd['input_embedder.mol_type_conditioning_init.weight']))    # (4,384)
  S('diffuser/boltz2_method_conditioning', 'weights',
    C._arr(sd['input_embedder.method_conditioning_init.weight']))      # (12,384)
  S('diffuser/boltz2_modified_conditioning', 'weights',
    C._arr(sd['input_embedder.modified_conditioning_init.weight']))    # (2,384)
  # trunk single/pair init from s_inputs (Boltz s_init / z_init_1 / z_init_2)
  S(f'{EVO}/single_activations', 'weights', C.t(sd['s_init.weight']))                  # (384,384)
  S(f'{EVO}/left_single', 'weights', C.t(sd['z_init_1.weight']))                       # (384,128)
  S(f'{EVO}/right_single', 'weights', C.t(sd['z_init_2.weight']))                      # (384,128)
  # bond embedding (token_bonds -> pair). token_bonds.weight is (128, n_bond_feat).
  S(f'{EVO}/bond_embedding', 'weights', C.t(sd['token_bonds.weight']))
  # the other two z-init terms boltz adds and AF3 has no slot for. TRUNK-level weights,
  # distinct from the identically-shaped set under confidence_module (see map_confidence).
  # nn.Embedding is already (in, out) for a one-hot matmul -- do NOT transpose, that
  # would mix bond types together.
  S(f'{EVO}/token_bonds_type_embed', 'weights', C._arr(sd['token_bonds_type.weight']))
  for _scope, _tn in (('contact_encoder', 'encoder'),
                      ('contact_fourier', 'fourier_embedding.proj')):
    params[f'{EVO}/{_scope}'] = {
        'weights': C._arr(sd[f'contact_conditioning.{_tn}.weight']).T,
        'bias': C._arr(sd[f'contact_conditioning.{_tn}.bias']),
    }
  # bare hk.get_parameter in a module-level helper called from @hk.transparent
  # _embed_bonds -> lands in the Evoformer's own bundle, not a submodule scope.
  params[EVO] = dict(params.get(EVO, {}), **{
      'contact_encoding_unspecified':
          C._arr(sd['contact_conditioning.encoding_unspecified']),
      'contact_encoding_unselected':
          C._arr(sd['contact_conditioning.encoding_unselected']),
  })
  # diffusion single conditioner INITIAL projection (input now 768 = [s_trunk, s_inputs]);
  # offset/bias dropped by the create_offset=False graph LN + bias-free projection (minor
  # per-token constant, documented in map_diffusion_conditioners).
  sc = 'structure_module.score_model.single_conditioner'
  S(f'{DH}/single_cond_initial_norm', 'scale', C._arr(sd[f'{sc}.norm_single.weight']))
  S(f'{DH}/single_cond_initial_norm', 'offset', C._arr(sd[f'{sc}.norm_single.bias']))
  S(f'{DH}/single_cond_initial_projection', 'weights', C.t(sd[f'{sc}.single_embed.weight']))
  S(f'{DH}/single_cond_initial_projection', 'bias',
    C._arr(sd[f'{sc}.single_embed.bias']))
  # relative position encoding: Boltz rel_pos.linear_layer (128,139) maps DIRECTLY onto AF3's
  # create_relative_encoding (same 139-d feature order -- verified: our relpe vs Boltz relpos
  # corr 1.0 with NO column reorder; a prior entity/chain swap was WRONG, gave 0.52). Same weight
  # feeds the trunk z-init (position_activations) and the diffusion pair conditioner (relpe).
  relw = sd['rel_pos.linear_layer.weight']              # (128, 139)
  S(f'{EVO}/~_relative_encoding/position_activations', 'weights', C.t(relw))   # (139,128)
  S(f'{DH}/relpe_projection', 'weights', C.t(relw))
  # diffusion pairwise_conditioner initial projection (input now 256 = [z_trunk, relpe])
  pw = 'diffusion_conditioning.pairwise_conditioner'
  S(f'{DH}/pair_cond_initial_norm', 'scale', C._arr(sd[f'{pw}.dim_pairwise_init_proj.0.weight']))
  S(f'{DH}/pair_cond_initial_norm', 'offset', C._arr(sd[f'{pw}.dim_pairwise_init_proj.0.bias']))
  S(f'{DH}/pair_cond_initial_projection', 'weights', C.t(sd[f'{pw}.dim_pairwise_init_proj.1.weight']))
  # MSA init: extra_msa_target_feat <- s_proj (s_inputs 384 -> msa_channel 64), clean.
  S(f'{EVO}/extra_msa_target_feat', 'weights', C.t(sd['msa_module.s_proj.weight']))   # (384,64)
  # msa_activations <- msa_proj (VALIDATED exact, corr 1.0). Boltz msa features =
  # [msa_onehot(33), has_deletion, deletion_value, is_paired]=36; ours (boltz2 branch in
  # evoformer) = [msa_onehot(32=31 POLYMER + 1 msa-mask), has_deletion, deletion_value,
  # is_paired]=35. Map: restype-perm the 31 POLYMER cols, zero the extra msa-mask col (rare in
  # the query), then has_del(33)/del_val(34)/is_paired(35). (Leaving this at random init was the
  # 12.68A-fold root cause -- random MSA rep -> OuterProductMean blows up z.)
  mp = np.asarray(sd['msa_module.msa_proj.weight'])          # (64, 36)
  msa_cols = list(BOLTZ2_RESTYPE_PERM) + [0, 33, 34, 35]     # 31 perm + [extra, has_del, del_val, is_paired]
  w_msa = mp[:, msa_cols].copy()                             # (64, 35)
  w_msa[:, 31] = 0.0                                          # zero the msa-mask/extra col (col 31)
  S(f'{EVO}/msa_activations', 'weights', C.t(w_msa))         # (35, 64)
  return params


def _atom_cross_block(sd, prefix, name, H, Dh):
  """One Boltz atom-transformer DiffusionTransformerLayer -> our CrossAttTransformer block
  (of3 is_cross_attn layout). Boltz has ONE adaln (self-attn with windowed keys); the
  cross-att graph wants SEPARATE q and k adaln -> duplicate Boltz's adaln into both. Keys use
  the '{name}q'/'{name}k' prefixes for the two adalns, '{name}' for the rest (of3 convention)."""
  g = lambda leaf: sd[C._pfx(prefix, leaf)]
  ada = dict(C.adaln(sd, C._pfx(prefix, 'adaln'), D))     # single_cond_layer_norm/scale, scale/w+b, bias/w
  out = {}
  for qk in ('q', 'k'):                                   # duplicate Boltz's single adaln into q and k
    out[f'{name}{qk}single_cond_layer_norm/scale'] = ada['single_cond_layer_norm/scale']
    out[f'{name}{qk}single_cond_scale/weights'] = ada['single_cond_scale/weights']
    out[f'{name}{qk}single_cond_scale/bias'] = ada['single_cond_scale/bias']
    out[f'{name}{qk}single_cond_bias/weights'] = ada['single_cond_bias/weights']
  out[f'{name}q_projection/weights'] = C.qk_std(g('pair_bias_attn.proj_q.weight'), H, Dh)
  out[f'{name}q_projection/bias'] = C._arr(g('pair_bias_attn.proj_q.bias')).reshape(H, Dh)
  out[f'{name}k_projection/weights'] = C.qk_std(g('pair_bias_attn.proj_k.weight'), H, Dh)
  out[f'{name}v_projection/weights'] = C.v_std(g('pair_bias_attn.proj_v.weight'), H, Dh)
  out[f'{name}gating_query/weights'] = C.t(g('pair_bias_attn.proj_g.weight'))
  out[f'{name}transition2/weights'] = C.t(g('pair_bias_attn.proj_o.weight'))
  out[f'{name}adaptive_zero_cond/weights'] = C.t(g('output_projection_linear.weight'))
  out[f'{name}adaptive_zero_cond/bias'] = C._arr(g('output_projection_linear.bias'))
  for k, v in cond_transition(sd, C._pfx(prefix, 'transition')).items():
    out[f'{name}{k}'] = v
  return out


def map_atom_encoder(sd, params):
  """Boltz-2 input-embedder atom encoder -> evoformer_conditioning_* (mirrors of3
  map_evoformer_conditioning). Produces the per-token atom feature `a` summed into s_inputs.
  WIP: architectural points to confirm via the s_inputs numerical gate -- (a) embed_pair_*
  and single_to_pair_cond_* are DOUBLE-created in the AF3 graph (_per_atom_conditioning +
  encoder body -> _1 variants); of3 maps both to the same Boltz weight (double-application).
  (b) the atom transformer's single adaln is duplicated to q/k. (c) gotchas: embed_ref_mask
  zeroed (Boltz has no mask feature), element NOT -1 shifted for boltz2, charge raw-vs-arcsinh."""
  P = 'diffuser'
  ae = 'input_embedder.atom_encoder'
  aae = 'input_embedder.atom_attention_encoder'
  pfx = 'evoformer_conditioning_'
  def S(leaf, param, arr): params.setdefault(f'{P}/{pfx}{leaf}', {})[param] = arr
  # --- per-atom single: column-slice the fused embed_atom_features (128,388):
  #     [ref_pos 0:3, charge 3:4, element 4:132, atom_name 132:388] ---
  W = sd[f'{ae}.embed_atom_features.weight']       # (128, 388)
  S('embed_ref_pos', 'weights', C.t(W[:, 0:3]))
  S('embed_ref_charge', 'weights', C.t(W[:, 3:4]))
  S('embed_ref_element', 'weights', C.t(W[:, 4:132]))
  S('embed_ref_atom_name', 'weights', C.t(W[:, 132:388]))
  S('embed_ref_mask', 'weights', np.zeros_like(C.t(W[:, 0:1])))   # Boltz has no mask feature
  # embed_atom_features is Boltz's `Linear` (WITH bias) over the fused 388-d vector; the
  # AF3 graph splits it into bias-free per-feature Linears, so the bias needs its own
  # parameter or it is silently dropped from every atom. See _per_atom_conditioning.
  # hk.get_parameter (not a Linear) -> a PARAMETER on the enclosing module, so it lands
  # in the 'diffuser' bundle under the prefixed name, not in a submodule scope.
  params.setdefault(P, {})[f'{pfx}embed_atom_features_bias'] = C._arr(
      sd[f'{ae}.embed_atom_features.bias'])
  # --- per-atom pair (double-applied: base + _1 to same Boltz weight, of3 convention) ---
  ofs = C.t(sd[f'{ae}.embed_atompair_ref_pos.weight'])   # (3,16)
  dst = C.t(sd[f'{ae}.embed_atompair_ref_dist.weight'])  # (1,16)
  msk = C.t(sd[f'{ae}.embed_atompair_mask.weight'])      # (1,16)
  S('embed_pair_offsets', 'weights', ofs);   S('embed_pair_offsets_1', 'weights', ofs)
  S('embed_pair_distances', 'weights', dst); S('embed_pair_distances_1', 'weights', dst)
  S('embed_pair_offsets_valid', 'weights', msk)
  rowk = C.t(sd[f'{ae}.c_to_p_trans_q.1.weight'])        # (128,16)  q->row
  colk = C.t(sd[f'{ae}.c_to_p_trans_k.1.weight'])        # (128,16)  k->col
  S('single_to_pair_cond_row', 'weights', rowk); S('single_to_pair_cond_row_1', 'weights', rowk)
  S('single_to_pair_cond_col', 'weights', colk); S('single_to_pair_cond_col_1', 'weights', colk)
  S('pair_mlp_1', 'weights', C.t(sd[f'{ae}.p_mlp.1.weight']))
  S('pair_mlp_2', 'weights', C.t(sd[f'{ae}.p_mlp.3.weight']))
  S('pair_mlp_3', 'weights', C.t(sd[f'{ae}.p_mlp.5.weight']))
  # atom->token aggregation projection (Boltz atom_to_token_trans.0)
  S('project_atom_features_for_aggr', 'weights', C.t(sd[f'{aae}.atom_to_token_trans.0.weight']))
  # --- atom cross-attention transformer x3 + shared pair bias (from atom_enc_proj_z) ---
  base = f'{aae}.atom_encoder.diffusion_transformer'
  name = f'{pfx}atom_transformer_encoder'
  scope = f'{P}/{name}'
  proj = 'input_embedder.atom_enc_proj_z'
  params.setdefault(f'{scope}/pair_input_layer_norm', {})['scale'] = C._arr(sd[f'{proj}.0.weight'])
  linz = sd[f'{proj}.1.weight']                          # (depth*heads=12, atom_z=16)
  params.setdefault(f'{scope}/pair_logits_projection', {})['weights'] = \
      linz.reshape(_N_ATOM, _ATOM_H, -1).transpose(2, 0, 1)   # (16, 3, 4)
  blocks = [_atom_cross_block(sd, f'{base}.layers.{i}', name, _ATOM_H, _ATOM_D) for i in range(_N_ATOM)]
  stacked = {}
  for k in blocks[0]:
    sub, param = k.rsplit('/', 1)
    stacked[f'{sub}::{param}'] = np.stack([b[k] for b in blocks], axis=0)
  C.populate(params, f'{scope}/__layer_stack_with_per_layer', stacked)
  return params


def map_diffusion_atom_conditioning(sd, params):
  """Diffusion score-model atom encoder+decoder conditioning -> diffuser/~/diffusion_head/
  diffusion_* (name='diffusion' in atom_cross_att_encoder/decoder). Mirrors map_atom_encoder
  (column-slice embed_atom_features, pair features, mask zeroed, offsets/single_to_pair double
  base+_1) plus the diffusion-only pieces: trunk single/pair cond embeddings (s_to_c/z_to_p),
  the noisy-coord projection (r_to_q_trans), atom<->token aggr/broadcast, and the final atom
  position update (decoder). Atom transformers themselves are mapped in map_diffusion_transformers."""
  DH = 'diffuser/~/diffusion_head'
  sm = 'structure_module.score_model'
  ac = 'diffusion_conditioning.atom_encoder'
  def S(leaf, param, arr): params.setdefault(f'{DH}/diffusion_{leaf}', {})[param] = arr
  W = sd[f'{ac}.embed_atom_features.weight']            # (128, 388)
  S('embed_ref_pos', 'weights', C.t(W[:, 0:3]))
  S('embed_ref_charge', 'weights', C.t(W[:, 3:4]))
  S('embed_ref_element', 'weights', C.t(W[:, 4:132]))
  S('embed_ref_atom_name', 'weights', C.t(W[:, 132:388]))
  S('embed_ref_mask', 'weights', np.zeros_like(C.t(W[:, 0:1])))
  params.setdefault(DH, {})['diffusion_embed_atom_features_bias'] = C._arr(
      sd[f'{ac}.embed_atom_features.bias'])
  ofs = C.t(sd[f'{ac}.embed_atompair_ref_pos.weight']); dst = C.t(sd[f'{ac}.embed_atompair_ref_dist.weight'])
  msk = C.t(sd[f'{ac}.embed_atompair_mask.weight'])
  S('embed_pair_offsets', 'weights', ofs);   S('embed_pair_offsets_1', 'weights', ofs)
  S('embed_pair_distances', 'weights', dst); S('embed_pair_distances_1', 'weights', dst)
  S('embed_pair_offsets_valid', 'weights', msk)
  rowk = C.t(sd[f'{ac}.c_to_p_trans_q.1.weight']); colk = C.t(sd[f'{ac}.c_to_p_trans_k.1.weight'])
  S('single_to_pair_cond_row', 'weights', rowk); S('single_to_pair_cond_row_1', 'weights', rowk)
  S('single_to_pair_cond_col', 'weights', colk); S('single_to_pair_cond_col_1', 'weights', colk)
  S('pair_mlp_1', 'weights', C.t(sd[f'{ac}.p_mlp.1.weight']))
  S('pair_mlp_2', 'weights', C.t(sd[f'{ac}.p_mlp.3.weight']))
  S('pair_mlp_3', 'weights', C.t(sd[f'{ac}.p_mlp.5.weight']))
  # trunk single/pair conditioning embeddings (diffusion-only): s_to_c / z_to_p (LN + Linear)
  S('embed_trunk_single_cond', 'weights', C.t(sd[f'{ac}.s_to_c_trans.1.weight']))   # (384,128)
  S('lnorm_trunk_single_cond', 'scale', C._arr(sd[f'{ac}.s_to_c_trans.0.weight']))
  S('lnorm_trunk_single_cond', 'offset', C._arr(sd[f'{ac}.s_to_c_trans.0.bias']))   # boltz2 keeps offset
  S('embed_trunk_pair_cond', 'weights', C.t(sd[f'{ac}.z_to_p_trans.1.weight']))     # (128,16)
  S('lnorm_trunk_pair_cond', 'scale', C._arr(sd[f'{ac}.z_to_p_trans.0.weight']))
  S('lnorm_trunk_pair_cond', 'offset', C._arr(sd[f'{ac}.z_to_p_trans.0.bias']))
  # noisy-coord projection into atom features (diffusion-only)
  S('atom_positions_to_features', 'weights', C.t(sd[f'{sm}.atom_attention_encoder.r_to_q_trans.weight']))  # (3,128)
  # atom<->token aggregation / broadcast + final position update (decoder)
  S('project_atom_features_for_aggr', 'weights', C.t(sd[f'{sm}.atom_attention_encoder.atom_to_token_trans.0.weight']))  # (128,768)
  S('project_token_features_for_broadcast', 'weights', C.t(sd[f'{sm}.atom_attention_decoder.a_to_q_trans.weight']))     # (768,128)
  S('atom_features_layer_norm', 'scale', C._arr(sd[f'{sm}.atom_attention_decoder.atom_feat_to_atom_pos_update.0.weight']))
  S('atom_features_layer_norm', 'offset', C._arr(sd[f'{sm}.atom_attention_decoder.atom_feat_to_atom_pos_update.0.bias']))
  S('atom_features_to_position_update', 'weights', C.t(sd[f'{sm}.atom_attention_decoder.atom_feat_to_atom_pos_update.1.weight']))  # (128,3)
  return params


def map_clean_oneoffs(sd, params):
  """One-off linears that map cleanly (same shape) into existing AF3 scopes: the
  recycling embedder (c_z/c_s dims match AF3) and the distogram head (64 bins, bias
  dropped). The other embeddings (single_activations/left_single/right_single/relpos)
  do NOT map cleanly -- they depend on Boltz's summed s_inputs (384) vs AF3's 449-dim
  target_feat, so they belong to the input-embedder forward branch, not here."""
  EVO = 'diffuser/evoformer'
  def S(scope, name, arr): params.setdefault(scope, {})[name] = arr
  # recycling embedder (pair z + single s). Boltz z_norm/s_norm are full LayerNorms
  # (weight+bias); the graph's recycle LNs keep both scale AND offset (unlike the
  # create_offset=False LNs elsewhere), so map the bias too.
  S(f'{EVO}/prev_embedding', 'weights', C.t(sd['z_recycle.weight']))
  S(f'{EVO}/prev_embedding_layer_norm', 'scale', C._arr(sd['z_norm.weight']))
  S(f'{EVO}/prev_embedding_layer_norm', 'offset', C._arr(sd['z_norm.bias']))
  S(f'{EVO}/prev_single_embedding', 'weights', C.t(sd['s_recycle.weight']))
  S(f'{EVO}/prev_single_embedding_layer_norm', 'scale', C._arr(sd['s_norm.weight']))
  S(f'{EVO}/prev_single_embedding_layer_norm', 'offset', C._arr(sd['s_norm.bias']))
  # distogram head. The bias is HALVED: boltz symmetrises the pair features
  # BEFORE the linear (`z = z + z.transpose(1, 2); self.distogram(z)`,
  # trunkv2.py:826) so its bias lands once, while our graph applies the linear
  # first and sums both orientations, doubling it. See model_config.DISTOGRAM_BIAS.
  S('diffuser/distogram_head/half_logits', 'weights', C.t(sd['distogram_module.distogram.weight']))
  S('diffuser/distogram_head/half_logits', 'bias',
    0.5 * C._arr(sd['distogram_module.distogram.bias']))
  return params


def map_template(sd, params):
  """Assemble Boltz's template_module into the boltz2 template scope
  diffuser/evoformer/template_embedding (Boltz2TemplateEmbedding). Forward + weights
  validated corr 1.0 vs Boltz's TemplateModule. z/v norms carry offsets; z/a/u_proj
  are no-bias linears (torch (out,in) -> C.t); the 2-block pair-only pairformer
  (pairwise_num_heads=4, pairwise_head_width=32) stacks under tmpl_pairformer."""
  TE = 'diffuser/evoformer/template_embedding'
  tm = 'template_module'
  params[f'{TE}/z_norm'] = {'scale': C._arr(sd[f'{tm}.z_norm.weight']),
                            'offset': C._arr(sd[f'{tm}.z_norm.bias'])}
  params[f'{TE}/v_norm'] = {'scale': C._arr(sd[f'{tm}.v_norm.weight']),
                            'offset': C._arr(sd[f'{tm}.v_norm.bias'])}
  params[f'{TE}/z_proj'] = {'weights': C.t(sd[f'{tm}.z_proj.weight'])}
  params[f'{TE}/a_proj'] = {'weights': C.t(sd[f'{tm}.a_proj.weight'])}
  params[f'{TE}/u_proj'] = {'weights': C.t(sd[f'{tm}.u_proj.weight'])}
  C.populate(params, f'{TE}/__layer_stack_no_per_layer',
             C.stack_blocks(lambda i: C.nest(
                 'tmpl_pairformer',
                 pair_block(sd, f'{tm}.pairformer.layers.{i}', 4, 32)), 2))
  return params


def map_trunk_stacks(sd, params):
  """Assemble the two validated trunk layer-stacks into the AF3 haiku scopes:
  pairformer x64 -> diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer,
  msa x4 -> diffuser/evoformer/__layer_stack_no_per_layer/msa_stack. WIP partial of
  map_boltz2_to_af3 (embeddings/template/diffusion/confidence still to add).
  """
  EVO = 'diffuser/evoformer'
  C.populate(params, f'{EVO}/__layer_stack_no_per_layer_1',
             C.stack_blocks(lambda i: C.nest(
                 'trunk_pairformer',
                 pairformer_block(sd, f'pairformer_module.layers.{i}')), _N_TRUNK))
  C.populate(params, f'{EVO}/__layer_stack_no_per_layer',
             C.stack_blocks(lambda i: msa_block(sd, f'msa_module.layers.{i}'), _N_MSA))
  return params


def _rekey(block, stackname):
  """diff_block 'X/param' -> '{stackname}{X}::param' for populate under a diffusion stack
  (our diffusion self_attention/transition_block name their params '{stackname}{X}')."""
  out = {}
  for k, v in block.items():
    sub, name = k.rsplit('/', 1)
    out[f'{stackname}{sub}::{name}'] = v
  return out


def _token_block(sd, sm, dc, i):
  """One token-transformer block in populate '{sub}::{param}' form. The DiffusionTransformerLayer
  params get the 'transformer' name prefix; the of3-mode per-block pair bias (bare
  pair_input_layer_norm / pair_logits_projection) comes from diffusion_conditioning's
  token_trans_proj_z[i] = Sequential(LayerNorm(z), Linear(z, heads, bias=False)). The LN
  offset is dropped (create_offset=False in the graph): it is a per-head constant added to
  every (i,j) logit and cancels in the softmax over keys (verified vs captured bias 1.1e-5)."""
  out = _rekey(diff_block(sd, f'{sm}.token_transformer.layers.{i}', 16, 48), 'transformer')
  out['pair_input_layer_norm::scale'] = C._arr(sd[f'{dc}.token_trans_proj_z.{i}.0.weight'])
  out['pair_logits_projection::weights'] = C.t(sd[f'{dc}.token_trans_proj_z.{i}.1.weight'])
  return out


def map_diffusion_transformers(sd, params, n_super=6):
  """Assemble the three diffusion transformer stacks (reusing diff_block) into the
  diffusion_head scopes: token transformer x24 (nested n_super x inner) + atom
  encoder/decoder x3 each. Boltz is in OPENFOLD3_LINEAGE, so the
  token transformer is a DOUBLE-nested __layer_stack_no_per_layer with a per-block pair LN +
  projection inside each block (GATE C validated: corr 0.99999981 vs captured Boltz output).
  The conditioners (single/pairwise) are added separately. WIP part of map_boltz2_to_af3."""
  DH = 'diffuser/~/diffusion_head'
  sm = 'structure_module.score_model'
  dc = 'diffusion_conditioning'
  # token transformer x24 (dim 768, 16 heads), of3-mode double-nested (n_super, 24//n_super)
  C.populate(params, f'{DH}/transformer/__layer_stack_no_per_layer/__layer_stack_no_per_layer',
             C.stack_super(lambda i: _token_block(sd, sm, dc, i), _N_DIFF, n_super))
  # atom encoder/decoder transformers x3 (dim 128, 4 heads): cross-att layout (like the input
  # atom encoder) -- _atom_cross_block duplicates Boltz's single adaln into q/k. Pair bias from
  # diffusion_conditioning.atom_{enc,dec}_proj_z, which here is PER-BLOCK (ModuleList of 3) --
  # the graph shares ONE pair_input_layer_norm, so we use block-0's LN scale (offset cancels in
  # softmax; per-block scale differences are a small approximation) + per-block flat projection.
  for role, pz in (('encoder', 'atom_enc_proj_z'), ('decoder', 'atom_dec_proj_z')):
    base = f'{sm}.atom_attention_{role}.atom_{role}.diffusion_transformer'
    name = f'diffusion_atom_transformer_{role}'
    scope = f'{DH}/{name}'
    proj = f'diffusion_conditioning.{pz}'
    # Boltz atom_{enc,dec}_proj_z is PER-BLOCK (LayerNorm + Linear), but the graph has ONE shared
    # pair_input_layer_norm. Make it exact by normalize-only shared LN (scale=1) and BAKING each
    # block's LN scale into that block's pair_logits Linear: logits[b] = Linear[b](scale[b]*z_norm)
    # = (Linear[b]*scale[b]) @ z_norm (the LN offset is a per-head constant -> cancels in softmax).
    az = np.asarray(sd[f'{proj}.0.1.weight']).shape[1]      # atom_z (=16)
    params.setdefault(f'{scope}/pair_input_layer_norm', {})['scale'] = np.ones(az, np.float32)
    blocks_linz = []
    for i in range(_N_ATOM):
      lin = np.asarray(sd[f'{proj}.{i}.1.weight'])          # (heads=4, atom_z=16)
      ln_scale = np.asarray(sd[f'{proj}.{i}.0.weight'])     # (atom_z=16,)
      blocks_linz.append(lin * ln_scale[None, :])           # bake LN scale into the Linear
    linz = np.stack(blocks_linz, axis=0)                    # (3, heads=4, atom_z=16)
    params.setdefault(f'{scope}/pair_logits_projection', {})['weights'] = linz.transpose(2, 0, 1)  # (16,3,4)
    blocks = [_atom_cross_block(sd, f'{base}.layers.{i}', name, _ATOM_H, _ATOM_D) for i in range(_N_ATOM)]
    stacked = {}
    for k in blocks[0]:
      sub, param = k.rsplit('/', 1)
      stacked[f'{sub}::{param}'] = np.stack([b[k] for b in blocks], axis=0)
    C.populate(params, f'{scope}/__layer_stack_with_per_layer', stacked)
  return params


def cond_transition(sd, prefix):
  """Boltz ConditionedTransitionBlock -> our transition_block ffw_* params (needs the
  global_config.boltz2_cond_transition graph branch for the extra a_to_b up-gate).

  b = SwiGLU(swish_gate(a)) * a_to_b(a); out = sigmoid(output_projection(s)) * b_to_a(b).
  """
  out = {f'ffw_{k}': v for k, v in C.adaln(sd, C._pfx(prefix, 'adaln'), D).items()}
  # Boltz SwiGLU gates on the SECOND half (silu(gates)*x, gates=chunk[1]); our
  # transition_block gates on the FIRST (swish(a)*b, a=split[0]). Swap the halves.
  sg = C.t(sd[C._pfx(prefix, 'swish_gate.0.weight')])       # (dim, 2*inner)
  inner = sg.shape[1] // 2
  out['ffw_transition1/weights'] = np.concatenate([sg[:, inner:], sg[:, :inner]], axis=1)
  out['ffw_a_to_b/weights'] = C.t(sd[C._pfx(prefix, 'a_to_b.weight')])
  out['ffw_transition2/weights'] = C.t(sd[C._pfx(prefix, 'b_to_a.weight')])
  out['ffw_adaptive_zero_cond/weights'] = C.t(sd[C._pfx(prefix, 'output_projection.0.weight')])
  out['ffw_adaptive_zero_cond/bias'] = C._arr(sd[C._pfx(prefix, 'output_projection.0.bias')])
  return out


def diff_self_attn(sd, prefix, H, Dh):
  """Boltz DiffusionTransformerLayer attention (adaln at block + pair_bias_attn.proj_*
  + block-scope output_projection_linear adaptive-zero gate) -> our self_attention params.
  The pair bias is external (per-block proj_z), supplied as pair_logits at forward time.
  """
  pa = C._pfx(prefix, 'pair_bias_attn')
  g = lambda leaf: sd[C._pfx(prefix, leaf)]
  out = dict(C.adaln(sd, C._pfx(prefix, 'adaln'), D))
  out['q_projection/weights'] = C.qk_std(g('pair_bias_attn.proj_q.weight'), H, Dh)
  out['q_projection/bias'] = C._arr(g('pair_bias_attn.proj_q.bias')).reshape(H, Dh)
  out['k_projection/weights'] = C.qk_std(g('pair_bias_attn.proj_k.weight'), H, Dh)
  out['v_projection/weights'] = C.v_std(g('pair_bias_attn.proj_v.weight'), H, Dh)
  out['gating_query/weights'] = C.t(g('pair_bias_attn.proj_g.weight'))
  out['transition2/weights'] = C.t(g('pair_bias_attn.proj_o.weight'))
  out['adaptive_zero_cond/weights'] = C.t(g('output_projection_linear.weight'))
  out['adaptive_zero_cond/bias'] = C._arr(g('output_projection_linear.bias'))
  return out


def diff_block(sd, prefix, H, Dh):
  """One full Boltz DiffusionTransformerLayer -> our (self_attention + transition_block).
  Shared by the token transformer, atom enc/dec transformers, and the input atom encoder.
  """
  out = diff_self_attn(sd, prefix, H, Dh)
  out.update(cond_transition(sd, C._pfx(prefix, 'transition')))
  return out


def msa_block(sd, prefix, msa_H=_MSA_H, msa_vd=_MSA_VD, pair_H=_PAIR_H, pair_D=_PAIR_D):
  """One Boltz MSALayer -> our EvoformerIteration (msa stack). Boltz forward order is
  PWA + msa_transition (MSA update) THEN outer_product_mean THEN the pair stack -- the
  update-then-OPM order our EvoformerIteration takes under the opendde/msa-order gate
  (a forward branch, not a weight difference). Emitted under a 'msa_stack/' namespace.
  """
  out = C.nest('msa_stack', pair_block(sd, C._pfx(prefix, 'pairformer_layer'), pair_H, pair_D))
  out.update(C.nest('msa_stack/msa_attention1',
                    C.msa_attention(sd, C._pfx(prefix, 'pair_weighted_averaging'), D, msa_H, msa_vd)))
  out.update(C.nest('msa_stack/msa_transition',
                    C.transition(sd, C._pfx(prefix, 'msa_transition'), D)))
  opm_pfx = C._pfx(prefix, 'outer_product_mean')
  c_hidden = np.asarray(sd[C._pfx(opm_pfx, 'proj_a.weight')]).shape[0]
  out.update(C.nest('msa_stack/outer_product_mean',
                    C.outer_product_mean(sd, opm_pfx, D, c_hidden=c_hidden)))
  return out


_N_CONF = 8                       # confidence pairformer depth (heads.confidence.pairformer.num_layer)


def map_confidence(sd, params):
  """Boltz-2's ConfidenceModule -> our AF3 ConfidenceHead scopes.

  What maps cleanly (done here):
    * the 8-layer confidence pairformer. It is a FULL PairformerLayer, identical in
      shape to the trunk's (pair 4x32, single 16x24=384, transition_s + pre_norm_s),
      so pairformer_block covers it verbatim -- 44 of the 66 unported arrays.
    * pae / pde logits, from the INTRA heads. Boltz splits each into intra + inter and
      chain-masks them:
          pae = to_pae_intra(z)*same_chain + to_pae_inter(z)*diff_chain
      On a monomer every pair is same-chain, so intra alone is exact; the inter head
      needs a graph branch before multimers are right (see below).
    * the s->z projections: s_to_z / s_to_z_transpose are our left/right
      target_feat_project (torch (out,in) -> transpose).

  What does NOT map, and why each needs a gated forward branch rather than a weight:
    1. pLDDT/resolved are TOKEN-level in boltz ((50,384) / (2,384)) but PER-ATOM in
       AF3 ([384,24,50] / [384,24,2]). Different output rank, not a reshape.
    2. distance bins: boltz embeds 64 bins (dist_bin_pairwise_embed + `boundaries`),
       AF3 projects 39 (distogram_feat_project). Different discretisation.
    3. boltz has no per-head LayerNorm before the logit heads -- it applies
       to_*_logits straight to z/s -- whereas AF3 has logits_ln / pae_logits_ln /
       plddt_logits_ln / experimentally_resolved_ln. These cannot be mapped to
       identity (a LayerNorm with scale=1/offset=0 still normalises); the graph has
       to skip them under the boltz2 gate.
    4. boltz re-embeds z from its own rel_pos / token_bonds / token_bonds_type /
       contact_conditioning (+ s_to_z_prod_in1/in2/out) after LayerNorming s and z.
       That is `ConfidenceHead._boltz2_reembed`, whose weights are mapped below.
  1-4 have all landed; `missing` is empty for this head.
  """
  CONF = 'diffuser/confidence_head'
  C.populate(params, f'{CONF}/__layer_stack_no_per_layer',
             C.stack_blocks(lambda i: C.nest(
                 'confidence_pairformer',
                 pairformer_block(
                     sd, f'confidence_module.pairformer_stack.layers.{i}')),
                 _N_CONF))
  cm = 'confidence_module'
  # Each of these is its OWN haiku scope with a single `weights` leaf -- not a key
  # inside a parent scope. torch Linear stores (out, in); haiku wants (in, out).
  def put(scope, torch_key, transpose=True):
    w = C._arr(sd[torch_key])
    params[scope] = {'weights': w.T if transpose else w}

  # The z re-embedding (`_boltz2_reembed`). Orientation of the two s->z projections is
  # the opposite of what the names suggest: boltz broadcasts s_to_z along j, i.e. it is
  # indexed by i, which is our RIGHT projection.
  RE = f'{CONF}/~_boltz2_reembed'
  put(f'{RE}/right_target_feat_project', f'{cm}.s_to_z.weight')
  put(f'{RE}/left_target_feat_project', f'{cm}.s_to_z_transpose.weight')
  put(f'{RE}/rel_pos_project', f'{cm}.rel_pos.linear_layer.weight')
  put(f'{RE}/token_bonds_project', f'{cm}.token_bonds.weight')
  put(f'{RE}/s_input_to_s', f'{cm}.s_input_to_s.weight')
  for k in ('s_to_z_prod_in1', 's_to_z_prod_in2', 's_to_z_prod_out'):
    put(f'{RE}/{k}', f'{cm}.{k}.weight')
  # nn.Embedding is already (in, out) for a one-hot matmul -- transposing it would
  # silently mix bond types together.
  put(f'{RE}/token_bonds_type_embed', f'{cm}.token_bonds_type.weight',
      transpose=False)
  for name, torch_name in (('z_norm', 'z_norm'), ('s_norm', 's_norm'),
                           ('s_inputs_norm', 's_inputs_norm')):
    params[f'{RE}/{name}'] = {
        'scale': C._arr(sd[f'{cm}.{torch_name}.weight']),
        'offset': C._arr(sd[f'{cm}.{torch_name}.bias']),
    }
  for scope, torch_name in (('contact_encoder', 'encoder'),
                            ('contact_fourier', 'fourier_embedding.proj')):
    params[f'{RE}/{scope}'] = {
        'weights': C._arr(sd[f'{cm}.contact_conditioning.{torch_name}.weight']).T,
        'bias': C._arr(sd[f'{cm}.contact_conditioning.{torch_name}.bias']),
    }
  # bare hk.get_parameter in a @hk.transparent helper -> the head's own bundle
  params[CONF] = dict(params.get(CONF, {}), **{
      'contact_encoding_unspecified':
          C._arr(sd[f'{cm}.contact_conditioning.encoding_unspecified']),
      'contact_encoding_unselected':
          C._arr(sd[f'{cm}.contact_conditioning.encoding_unselected']),
  })
  # intra/inter chain head pairs (use_separate_heads=True): each is hard-masked to its
  # half of the pair matrix, so the inter weights are dead on a monomer.
  put(f'{CONF}/pae_logits',
      f'{cm}.confidence_heads.to_pae_intra_logits.weight')
  put(f'{CONF}/pae_inter_logits',
      f'{cm}.confidence_heads.to_pae_inter_logits.weight')
  put(f'{CONF}/left_half_distance_logits',
      f'{cm}.confidence_heads.to_pde_intra_logits.weight')
  put(f'{CONF}/inter_half_distance_logits',
      f'{cm}.confidence_heads.to_pde_inter_logits.weight')
  # Token-level heads. Boltz predicts one pLDDT (and one resolved) per TOKEN;
  # AF3's head is per dense-atom slot. Broadcasting boltz's output across the 24
  # slots is the same function as a per-slot weight whose slots are all equal, so
  # tile it here and the forward graph needs no boltz2 branch. The prediction is
  # unchanged -- every atom of a token still reports that token's number.
  def put_tiled(scope, torch_key, max_atoms=24):
    w = C._arr(sd[torch_key]).T                        # (c_s, bins)
    params[scope] = {'weights': np.repeat(w[:, None, :], max_atoms, axis=1)}

  put_tiled(f'{CONF}/plddt_logits',
            f'{cm}.confidence_heads.to_plddt_logits.weight')
  put_tiled(f'{CONF}/experimentally_resolved_logits',
            f'{cm}.confidence_heads.to_resolved_logits.weight')
  # nn.Embedding stores (num_bins, dim) -- already (in, out) for a one-hot matmul,
  # so unlike the Linears this one is NOT transposed.
  put(f'{RE}/distogram_feat_project',
      f'{cm}.dist_bin_pairwise_embed.weight', transpose=False)
  return params
