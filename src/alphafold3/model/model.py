# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold3 model."""

from collections.abc import Iterable, Mapping
import concurrent
import copy
import dataclasses
import functools
import os
from typing import Any, TypeAlias

from absl import logging
from alphafold3 import structure
from alphafold3.common import base_config
from alphafold3.model import confidences
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.constants import residue_names
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import mapping
from alphafold3.model.components import utils
from .network import atom_cross_attention
from .network import bioemu2 as bioemu2_network
from .network import confidence_head
from .network import diffusion_head
from .network import diffusion_transformer
from .network import distogram_head
from .network import evoformer as evoformer_network
from .network import featurization
from .network import modules as pairformer_modules
from .network import opendde_confidence
from .network import structural_tokens
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


ModelResult: TypeAlias = Mapping[str, Any]


@dataclasses.dataclass(frozen=True, kw_only=True)
class InferenceResult:
  """Postprocessed model result.

  Attributes:
    predicted_structure: Predicted protein structure.
    numerical_data: Useful numerical data (scalars or arrays) to be saved at
      inference time.
    metadata: Smaller numerical data (usually scalar) to be saved as inference
      metadata.
    debug_outputs: Additional dict for debugging, e.g. raw outputs of a model
      forward pass.
    model_id: Model identifier.
  """

  predicted_structure: structure.Structure = dataclasses.field()
  numerical_data: Mapping[str, float | int | np.ndarray] = dataclasses.field(
      default_factory=dict
  )
  metadata: Mapping[str, float | int | np.ndarray] = dataclasses.field(
      default_factory=dict
  )
  debug_outputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  model_id: bytes = b''


def get_predicted_structure(
    result: ModelResult, batch: feat_batch.Batch
) -> structure.Structure:
  """Creates the predicted structure and ion preditions.

  Args:
    result: model output in a model specific layout
    batch: model input batch

  Returns:
    Predicted structure.
  """
  return predicted_structure_from_coords(
      result['diffusion_samples']['atom_positions'],
      batch,
      predicted_lddt=result.get('predicted_lddt'),
  )


def predicted_structure_from_coords(
    model_output_coords, batch: feat_batch.Batch, predicted_lddt=None
) -> structure.Structure:
  """Coordinates in the model-output atom layout -> a Structure.

  Split out of get_predicted_structure so that a model which is not an AF3
  diffusion model can reach it. Everything below is driven by the batch's atom
  layouts and knows nothing about how the coordinates were produced -- the only
  AF3-specific thing in the original was reading them out of
  `result['diffusion_samples']`. AlphaFold 2 arrives here from
  `alphafold3.af2.output`, which regathers its atom37 into the same layout, and
  so gets the same mmCIF, the same b-factors and the same missing-atom warning
  rather than a second output path that drifts.

  Args:
    model_output_coords: (..., num_tokens, max_atoms_per_token, 3).
    batch: model input batch, for its atom layouts and empty output structure.
    predicted_lddt: (..., num_tokens, max_atoms_per_token) or None; written to
      the b-factor column, as AlphaFold does.

  Returns:
    Predicted structure.
  """
  # Rearrange model output coordinates to the flat output layout.
  model_output_to_flat = atom_layout.compute_gather_idxs(
      source_layout=batch.convert_model_output.token_atoms_layout,
      target_layout=batch.convert_model_output.flat_output_layout,
  )
  pred_flat_atom_coords = atom_layout.convert(
      gather_info=model_output_to_flat,
      arr=model_output_coords,
      layout_axes=(-3, -2),
  )

  if predicted_lddt is not None:
    pred_flat_b_factors = atom_layout.convert(
        gather_info=model_output_to_flat,
        arr=predicted_lddt,
        layout_axes=(-2, -1),
    )
  else:
    # Handle models which don't have predicted_lddt outputs.
    pred_flat_b_factors = np.zeros(pred_flat_atom_coords.shape[:-1])

  (missing_atoms_indices,) = np.nonzero(model_output_to_flat.gather_mask == 0)
  if missing_atoms_indices.shape[0] > 0:
    missing_atoms_flat_layout = batch.convert_model_output.flat_output_layout[
        missing_atoms_indices
    ]
    missing_atoms_uids = list(
        zip(
            missing_atoms_flat_layout.chain_id,
            missing_atoms_flat_layout.res_id,
            missing_atoms_flat_layout.res_name,  # pyrefly: ignore[bad-argument-type]
            missing_atoms_flat_layout.atom_name,
        )
    )
    logging.warning(
        'Target %s: warning: %s atoms were not predicted by the '
        'model, setting their coordinates to (0, 0, 0). '
        'Missing atoms: %s',
        batch.convert_model_output.empty_output_struc.name,
        missing_atoms_indices.shape[0],
        missing_atoms_uids,
    )

  # Put them into a structure
  pred_struc = batch.convert_model_output.empty_output_struc
  pred_struc = pred_struc.copy_and_update_atoms(
      atom_x=pred_flat_atom_coords[..., 0],  # pyrefly: ignore[bad-argument-type]
      atom_y=pred_flat_atom_coords[..., 1],  # pyrefly: ignore[bad-argument-type]
      atom_z=pred_flat_atom_coords[..., 2],  # pyrefly: ignore[bad-argument-type]
      atom_b_factor=pred_flat_b_factors,  # pyrefly: ignore[bad-argument-type]
      atom_occupancy=np.ones(pred_flat_atom_coords.shape[:-1]),  # Always 1.0.
  )
  # Set manually/differently when adding metadata.
  pred_struc = pred_struc.copy_and_update_globals(release_date=None)
  return pred_struc


def create_target_feat_embedding(
    batch: feat_batch.Batch,
    config: evoformer_network.Evoformer.Config,
    global_config: model_config.GlobalConfig,
    soft_seq=None,
    design_mask=None,
) -> jnp.ndarray:
  """Create target feature embedding.

  Returns one tensor, or `(trunk, structure)` for the models in
  `model_config.SEPARATE_STRUCTURE_TARGET_FEAT`, which project the same input
  twice -- once for the trunk and once for the diffusion module.
  """

  dtype = utils.compute_dtype(global_config)

  with utils.bfloat16_context():
    target_feat = featurization.create_target_feat(
        batch,
        append_per_atom_features=False,
        soft_seq=soft_seq,
        design_mask=design_mask,
    ).astype(dtype)

    enc = atom_cross_attention.atom_cross_att_encoder(
        token_atoms_act=None,
        trunk_single_cond=None,
        trunk_pair_cond=None,
        config=config.per_atom_conditioning,
        global_config=global_config,
        batch=batch,
        name='evoformer_conditioning',
    )
    if global_config.model == 'boltz2':
      # Boltz-2 input embedder (trunkv2.InputEmbedder): s_inputs is a SUM, not AF3's concat.
      #   s_inputs(seq_channel) = a(atom-encoder token_act)
      #     + res_type_encoding(one_hot aatype)              Linear(31->seq_channel, no bias)
      #     + msa_profile_encoding(cat[profile, deletion])   Linear(32->seq_channel, no bias)
      #     (+ zero-init method/modified/cyclic/mol_type conditioning; omitted -- default
      #      monomer, all near-constant. Add via feats when non-default inputs are needed.)
      # Our aatype/profile are 31-class (POLYMER_TYPES_...GAP); Boltz's res_type/profile are
      # 33-class, so the converter remaps those weight columns 33->31 (remap_restype_cols).
      from alphafold3.model.components import haiku_modules as hm
      from alphafold3.constants import residue_names
      n_restype = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
      res_oh = jax.nn.one_hot(batch.token_features.aatype, n_restype)
      res_oh = featurization.blend_soft(res_oh, soft_seq, design_mask).astype(dtype)
      # THE PROFILE IS A SEQUENCE FEATURE TOO. Read raw it is the profile of
      # whatever placeholder the batch was built from -- poly-alanine, in
      # design -- while res_type above moves with soft_seq: the trunk is then
      # told two different sequences, which is AF3's own PROFILE_MODE problem
      # (featurization.py, 'frozen' == exactly this) and AF2's pssm_hard one.
      # Measured before the fix: boltz2's profile channel contributed EXACTLY
      # zero design gradient -- `both` and `msa`-only agreed to the last digit
      # over six seeds, as did `none` and `profile`-only.
      profile = featurization.blend_profile(
          batch.msa.profile, soft_seq, design_mask).astype(dtype)
      deletion = batch.msa.deletion_mean[..., None].astype(dtype)
      s_inputs = enc.token_act.astype(dtype)
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_res_type_encoding')(res_oh)
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_msa_profile_encoding')(
          jnp.concatenate([profile, deletion], axis=-1))
      # Boltz's DN (32), the DNA unknown - a DNA N, and a modified DNA residue's one token (get_unk_token) - which our
      # single nucleic unknown (30, Boltz's RNA N at 27) cannot tell from an RNA one: the converter keeps each
      # encoding's DN column less its N column, added on a DNA token by its class-30 weight. Zero for a blob without
      # them (the loader's fill), which folds as before.
      dn_res = hk.get_parameter('boltz2_dn_res_type', [config.seq_channel], jnp.float32, init=jnp.zeros)
      dn_prof = hk.get_parameter('boltz2_dn_msa_profile', [config.seq_channel], jnp.float32, init=jnp.zeros)
      on_dna = batch.token_features.is_dna.astype(dtype)
      s_inputs += (on_dna * res_oh[:, 30])[:, None] * dn_res.astype(dtype)
      s_inputs += (on_dna * profile[:, 30])[:, None] * dn_prof.astype(dtype)
      # conditioning inits (trained, non-zero): mol_type (PROTEIN=0/DNA=1/RNA=2/NONPOLYMER=3),
      # method (x-ray=1, the prediction default), modified=0. cyclic=0 -> Linear(0)=0, omitted.
      # These are Embeddings in Boltz; here Linear-over-one-hot (weight (num_cls, seq) == the
      # Embedding table). Validated: adding them lifts s_inputs corr 0.908 -> 0.974.
      tf = batch.token_features
      mol_type = (tf.is_dna.astype(jnp.int32) + 2 * tf.is_rna.astype(jnp.int32)
                  + 3 * tf.is_ligand.astype(jnp.int32))
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_mol_type_conditioning')(
          jax.nn.one_hot(mol_type, 4).astype(dtype))
      n_tok = s_inputs.shape[0]
      # boltz's cyclic flag: `cyclic_period.clamp(max=1.0)` -- present/absent,
      # NOT the period. Zero when nothing is cyclic, so this term vanishes on
      # every ordinary input and the trained weight only ever acts on a cyclic
      # one. Our cyclic support had gone into the relative-position wrap alone
      # (featurization._relative_encoding), which is AF3's mechanism; boltz2
      # carries BOTH, and this was the half we did not port.
      cyc = batch.token_features.cyclic_period
      cyc = (jnp.zeros((n_tok,), dtype) if cyc is None
             else jnp.clip(cyc.astype(dtype), 0.0, 1.0))
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_cyclic_conditioning')(cyc[..., None])
      method = jnp.full((n_tok,), 1, dtype=jnp.int32)   # x-ray diffraction
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_method_conditioning')(
          jax.nn.one_hot(method, 12).astype(dtype))
      # `modified` marks the tokens of a modified residue. Row 0 is a learned
      # non-zero vector, so a constant 0 was correct for every unmodified input
      # -- but on a PTM boltz2 needs the flag AND the unknown restype together:
      # either alone leaves the phosphate 6.4 A out with a 4.3 A OG-P bond,
      # both give 3.10 A and a correct 1.61 A bond (native boltz2: 1.77 / 1.64).
      modified = tf.is_modified
      modified = (jnp.zeros((n_tok,), jnp.int32) if modified is None
                  else modified.astype(jnp.int32))
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_modified_conditioning')(
          jax.nn.one_hot(modified, 2).astype(dtype))
      return s_inputs.astype(dtype)

    if global_config.model == 'chai1':
      # chai's token embedder: s_init = proj_in_trunk(cat[pooled_atoms, token_feats]).
      #
      # `token_feats` is chai's 384-d TOKEN stream. For a de novo single-chain
      # protein that whole 2638-wide stream collapses to one Linear over the
      # residue one-hot plus a constant -- five of its generators are identically
      # zero and three are constant one-hots (IsDistillation 0, TokenBFactor 2,
      # TokenPLDDT 3, all "not included" sentinels). Verified against chai's own
      # feature embedder at the bfloat16 floor, so the collapse is exact, not an
      # approximation. The converter folds the constant into this Linear's bias.
      #
      # Putting it here rather than in featurisation is deliberate: a Linear over
      # the residue one-hot is exactly where soft sequence has to blend.
      from alphafold3.model.components import haiku_modules as hm
      from alphafold3.constants import residue_names
      n_restype = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
      res_oh = jax.nn.one_hot(batch.token_features.aatype, n_restype)
      res_oh = featurization.blend_soft(res_oh, soft_seq, design_mask).astype(dtype)
      token_feats = hm.Linear(config.seq_channel, use_bias=True,
                              name='chai1_token_feature_embedding')(res_oh)
      # our N is chai's RX; on a DNA token chai has DX (a modified nucleotide's atomised tokens, an unknown base)
      dx = hk.get_parameter('chai1_token_dx', [config.seq_channel], jnp.float32, init=jnp.zeros)
      dna_unk = (batch.token_features.is_dna & (batch.token_features.aatype == n_restype - 1)).astype(dtype)
      token_feats += dna_unk[:, None] * dx.astype(dtype)
      # chai's TOKEN stream is dominated by ESM2: rebuilding the full 2638-wide
      # stream reproduces its s_init at corr 0.99999280, and zeroing ONLY the
      # ESM columns drops that to 0.327 (rms 0.183). Folding them into the bias
      # -- what we did while feeding zeros -- costs 5.70 A vs chai's 0.642 A on
      # a natural protein. Absent (None) it is simply not built, so every model
      # that does not supply ESM stays byte-identical.
      # the MSA profile and deletion mean, which chai carries in the SAME token
      # stream. Zero without an MSA (chai feeds an all-gap one), so this term
      # vanishes on the single-sequence path and leaves it byte-identical.
      token_feats += hm.Linear(
          config.seq_channel, use_bias=False,
          name='chai1_msa_profile_embedding')(
              jnp.concatenate([featurization.blend_profile(
                                   batch.msa.profile, soft_seq, design_mask),
                               batch.msa.deletion_mean[..., None]],
                              axis=-1).astype(dtype))
      esm = batch.token_features.esm_embeddings
      if esm is not None:
        token_feats += hm.Linear(
            config.seq_channel, use_bias=False,
            name='chai1_esm_embedding')(esm.astype(dtype))
      s_cat = jnp.concatenate([enc.token_act.astype(dtype), token_feats], axis=-1)
      # TWO projections of the same concatenation, not one. chai's
      # TokenInputEmbedding returns `(s_trunk, s_structure, z_init)`:
      #     s_structure = token_single_proj_in_structure(input13)
      #     s_trunk     = token_single_proj_in_trunk(input13)
      # and feeds the trunk the first and the DIFFUSION module the second. AF3
      # computes one `s_inputs` and hands it to both, so the port fed the
      # diffusion conditioning the trunk's projection -- a different 384-d
      # vector from the one those weights were trained against.
      #
      # Found by giving chai1 an L0 gate: `token_single_proj_in_structure.weight`
      # was one of five unaccounted tensors, and unlike the other four it is
      # called in EVERY forward_* of the shipped token_embedder graph, whose
      # return is `TupleConstruct(%s_trunk, %s_structure, %z_init)`.
      trunk = hm.Linear(config.seq_channel, use_bias=False,
                        name='chai1_single_proj_in_trunk')(s_cat).astype(dtype)
      # WHICH CONSUMER GETS WHICH, verified against chai's own captures rather
      # than inferred: the trunk and the CONFIDENCE head both take the trunk
      # projection (the confidence input's per-channel std correlates +0.744
      # with `proj_in_trunk`'s row norms and -0.07 with the structure one), and
      # the diffusion module takes the structure projection (+0.682 against ours
      # vs -0.013 for the trunk). So only the diffusion path is switched below.
      if global_config.model in model_config.SEPARATE_STRUCTURE_TARGET_FEAT:
        return trunk, hm.Linear(
            config.seq_channel, use_bias=False,
            name='chai1_single_proj_in_structure')(s_cat).astype(dtype)
      return trunk

    target_feat = jnp.concatenate([target_feat, enc.token_act], axis=-1).astype(
        dtype
    )
    if global_config.model in model_config.HF3_DN_CLASS:
      target_feat = hf3_dn_columns(target_feat, batch.token_features.is_dna)

  return target_feat


def hf3_dn_columns(target_feat, is_dna):
  """[447 | restype DN, profile DN]: HelixFold3's DNA unknown (DN, its class 26).

  Our nucleic unknown (30) is one class for RNA and DNA; HelixFold3 keeps them
  apart, and types a DNA N and every token of a modified DNA residue DN. So on a
  DNA token the class-30 entry of the restype and profile one-hots MOVES into
  two trailing columns, which the converter maps to HelixFold3's DN rows -- and
  which fill the two unknown-DNA columns its diffusion and confidence
  LayerNorms normalise over (model_config.HF3_DN_CLASS).
  """
  n = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
  dna = is_dna.astype(target_feat.dtype)[:, None]
  rest_dn = target_feat[:, n - 1:n] * dna
  prof_dn = target_feat[:, 2 * n - 1:2 * n] * dna
  target_feat = target_feat.at[:, n - 1:n].multiply(1 - dna)
  target_feat = target_feat.at[:, 2 * n - 1:2 * n].multiply(1 - dna)
  return jnp.concatenate([target_feat, rest_dn, prof_dn], axis=-1)


def _structure_only_results(result, pred_structure, num_tokens, chain_ids,
                            res_ids):
  """InferenceResults for a model with no confidence head.

  Every confidence-derived field is NaN, because there is nothing behind it: the
  checkpoint has no head, so the graph emits no logits (see the
  NO_CONFIDENCE_HEAD branch in __call__). The two things that ARE measured from
  the structure itself -- clashes and disorder -- are real, and so is the
  distogram, which is a separate head these models do have.

  Ranking is therefore arbitrary and says so: with a NaN score every `>`
  comparison is False, so the first sample stays the representative rather than
  an accidental winner.
  """
  nan = float('nan')
  contact_probs = result['distogram']['contact_probs'][:num_tokens, :num_tokens]
  pred_structures = pred_structure.unstack()
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=min(len(pred_structures), 32)
  ) as executor:
    has_clash = list(executor.map(confidences.has_clash, pred_structures))
    fraction_disordered = list(
        executor.map(confidences.fraction_disordered, pred_structures)
    )
  empty_pair = np.full((num_tokens, num_tokens), nan, dtype=np.float32)
  for idx, one in enumerate(pred_structures):
    yield InferenceResult(
        predicted_structure=one,
        numerical_data={
            'full_pde': empty_pair,
            'full_pae': empty_pair,
            'contact_probs': contact_probs,
        },
        metadata={
            'predicted_distance_error': nan,
            'ranking_score': nan,
            'fraction_disordered': fraction_disordered[idx],
            'has_clash': has_clash[idx],
            'predicted_tm_score': nan,
            'interface_predicted_tm_score': nan,
            'chain_pair_pde_mean': np.full((1, 1), nan),
            'chain_pair_pde_min': np.full((1, 1), nan),
            'chain_pair_pae_min': np.full((1, 1), nan),
            'ptm': nan,
            'iptm': nan,
            'ptm_iptm_average': nan,
            'intra_chain_single_pde': nan,
            'cross_chain_single_pde': nan,
            'pae_ichain': nan,
            'pae_xchain': nan,
            'ranking_confidence': nan,
            'ranking_confidence_pae': nan,
            'chain_pair_iptm': np.full((1, 1), nan),
            'iptm_ichain': nan,
            'iptm_xchain': nan,
            'token_chain_ids': chain_ids,
            'token_res_ids': res_ids,
        },
        model_id=result['__identifier__'],
        debug_outputs={},
    )


def _compute_ptm(
    result: ModelResult,
    num_tokens: int,
    asym_id: np.ndarray,
    pae_single_mask: np.ndarray,
    interface: bool,
) -> np.ndarray:
  """Computes the pTM metrics from PAE."""
  return np.stack(
      [
          confidences.predicted_tm_score(
              tm_adjusted_pae=tm_adjusted_pae[:num_tokens, :num_tokens],
              asym_id=asym_id,
              pair_mask=pae_single_mask[:num_tokens, :num_tokens],
              interface=interface,
          )
          for tm_adjusted_pae in result['tmscore_adjusted_pae_global']
      ],
      axis=0,
  )


def _compute_chain_pair_iptm(
    num_tokens: int,
    asym_ids: np.ndarray,
    mask: np.ndarray,
    tm_adjusted_pae: np.ndarray,
) -> np.ndarray:
  """Computes the chain pair ipTM metrics from PAE."""
  return np.stack(
      [
          confidences.chain_pairwise_predicted_tm_scores(
              tm_adjusted_pae=sample_tm_adjusted_pae[:num_tokens],
              asym_id=asym_ids[:num_tokens],
              pair_mask=mask[:num_tokens, :num_tokens],
          )
          for sample_tm_adjusted_pae in tm_adjusted_pae
      ],
      axis=0,
  )



def _structural_atoms_to_residue(positions, book):
  """OpenDDE's sampled atoms from structural back to residue tokens, exactly, plus the (n_res, max_atoms) atom mask."""
  gather = book['residue_atom_gather']            # (n_res, max_atoms), -1 = none
  n_res, max_atoms = gather.shape
  valid = gather >= 0
  flat = positions.reshape((positions.shape[0], -1, positions.shape[-1]))
  out = jnp.take(flat, jnp.where(valid, gather, 0).reshape(-1), axis=1)
  out = out.reshape((positions.shape[0], n_res, max_atoms, positions.shape[-1]))
  return jnp.where(valid[None, :, :, None], out, 0.0), valid


def _opendde_confidence(conf, atom_valid, seq_mask, asym_id,
                        num_plddt_bins, num_pae_bins, num_pde_bins, return_logits=False):
  """OpenDDE's residue-space confidence logits reduced to the outputs every other model reports."""
  n_res, max_atoms = atom_valid.shape

  def by_atom(arr):
    """(n_samples, n_res * max_atoms, *rest) -> (n_samples, n_res, max_atoms, *rest)."""
    rest = arr.shape[2:]
    out = arr.reshape((arr.shape[0], n_res, max_atoms) + rest)
    keep = atom_valid.reshape((1, n_res, max_atoms) + (1,) * len(rest))
    return jnp.where(keep, out, 0.0)

  def centers(min_bin, max_bin, n_bins):
    width = (max_bin - min_bin) / n_bins
    return min_bin + width * (jnp.arange(n_bins, dtype=jnp.float32) + 0.5)

  # Bin edges are OpenDDE's own: pLDDT over [0, 1] rescaled to 100, PAE and PDE
  # over [0, 32].
  plddt_logits = by_atom(conf['predicted_lddt_logits'])
  predicted_lddt = jnp.sum(
      jax.nn.softmax(plddt_logits, -1) * centers(0.0, 1.0, num_plddt_bins),
      -1) * 100.0
  resolved_logits = by_atom(conf['experimentally_resolved_logits'])

  pae_logits = conf['predicted_aligned_error_logits']
  pde_logits = conf['predicted_distance_error_logits']
  pae_centers = centers(0.0, 32.0, num_pae_bins)
  pae_probs = jax.nn.softmax(pae_logits, -1)
  mask = seq_mask.astype(bool)
  pair_mask = mask[:, None] * mask[None, :]
  full_pae = jnp.sum(pae_probs * pae_centers, -1) * pair_mask
  full_pde = jnp.sum(
      jax.nn.softmax(pde_logits, -1) * centers(0.0, 32.0, num_pde_bins),
      -1) * pair_mask
  average_pde = (jnp.sum(full_pde, axis=(-2, -1))
                 / jnp.maximum(jnp.sum(pair_mask), 1.0))

  tm_global, tm_interface = jax.vmap(
      lambda probs: confidence_head.tmscore_adjusted_pae(
          asym_id=asym_id, seq_mask=seq_mask, pair_mask=pair_mask,
          bin_centers=pae_centers, pae_probs=probs))(pae_probs)

  out = {
      'predicted_lddt': predicted_lddt,
      'full_pae': full_pae,
      'full_pde': full_pde,
      'average_pde': average_pde,
      'predicted_experimentally_resolved': jax.nn.softmax(
          resolved_logits, axis=-1)[..., 1],
      'tmscore_adjusted_pae_global': tm_global,
      'tmscore_adjusted_pae_interface': tm_interface,
  }
  if return_logits:
    out.update(predicted_lddt_logits=plddt_logits, predicted_aligned_error_logits=pae_logits,
               predicted_distance_error_logits=pde_logits, experimentally_resolved_logits=resolved_logits)
  return out


def num_trunk_passes(num_recycles: int, model: str) -> int:
  """Trunk passes for a `num_recycles` setting: AF's convention for every model, N recycles = N + 1 passes.

  Each model's own default pass count lives in model_registry (default_num_recycles), so `model` no longer changes
  the meaning; it stays in the signature for callers.
  """
  del model
  return max(0, num_recycles) + 1


class Chai1StructurePair(hk.Module):
  """chai-1's STRUCTURE token-pair features, the pair input of its diffusion module.

  chai-lab chunks its 512-wide TOKEN_PAIR input projection (and its bond projection) in two: the first halves
  build the trunk's z_init, the second halves `token_pair_structure_input_feats`, which its diffusion module
  conditions on beside z_trunk (chai1.py, run_folding_on_context). This repository's converter kept only the
  trunk halves and fed the diffusion the trunk's z_init instead -- relRMS 9.2 apart on 6MRR, measured on
  chai-lab's own tensors. The published chai1 blobs carry the second halves now, under this module's name
  (huggingface.co/sokrypton/af3-any-model 28141c7).

  The 163 inputs are chai's own columns (its generators, alphabetical): docking 0:6 (class 5: no
  constraint), relative chain 6:12, relative entity 12:15, residue separation 15:82, token separation
  82:149, the contact and pocket restraints 149:156 and 156:163 -- their mask columns 155 and 162 alone without
  constraints, chai's -1 sentinel; their RBF encodings (evoformer.chai1_restraint_feature) with. A one-hot that
  wide is never built: each pair adds the rows its classes select. The bond projection's structure half reads chai's declared-covalent-
  bond feature, which nothing here builds -- zero for every input without one, as chai-lab's is.

  Relative entity and chain read the raw ids, which is chai's dense ranking for every purpose here: entity
  enters only through same/greater/less (clipped to +-1), and sym ids within one entity are consecutive.
  """

  def __init__(self, num_channels, name='chai1_structure_token_pair'):
    super().__init__(name=name)
    self.num_channels = num_channels

  def __call__(self, tf, restraints=None):
    w = hk.get_parameter('weights', [163, self.num_channels], jnp.float32, init=jnp.zeros)
    b = hk.get_parameter('bias', [self.num_channels], jnp.float32, init=jnp.zeros)
    ri = tf.residue_index.astype(jnp.int32)
    ti = tf.token_index.astype(jnp.int32)
    asym, ent, sym = (x.astype(jnp.int32) for x in (tf.asym_id, tf.entity_id, tf.sym_id))
    same_chain = asym[:, None] == asym[None]
    rss = jnp.where(same_chain, jnp.clip(ri[:, None] - ri[None] + 33, 0, 65), 66)
    rts = jnp.where(same_chain & (ri[:, None] == ri[None]), jnp.clip(ti[:, None] - ti[None] + 32, 0, 65), 66)
    rel_e = ent[:, None] - ent[None]
    rchain = jnp.where(rel_e != 0, 5, jnp.clip(sym[:, None] - sym[None] + 2, 0, 4))
    rent = jnp.clip(rel_e + 1, 0, 2)
    out = ((b + w[5] + w[155] + w[162])[None, None] + w[6 + rchain] + w[12 + rent]
           + w[15 + rss] + w[82 + rts])
    if restraints is not None:
      out += evoformer_network.chai1_restraint_feature(*restraints) @ w[149:163]
    return out


class Model(hk.Module):
  """Full model. Takes in data batch and returns model outputs."""

  class HeadsConfig(base_config.BaseConfig):
    diffusion: diffusion_head.DiffusionHead.Config = base_config.autocreate()
    confidence: confidence_head.ConfidenceHead.Config = base_config.autocreate()
    distogram: distogram_head.DistogramHead.Config = base_config.autocreate()

  class Config(base_config.BaseConfig):
    evoformer: evoformer_network.Evoformer.Config = base_config.autocreate()
    global_config: model_config.GlobalConfig = base_config.autocreate()
    heads: 'Model.HeadsConfig' = base_config.autocreate()
    num_recycles: int = 10
    return_embeddings: bool = False
    return_distogram: bool = False
    # DESIGN: differentiate the LAST trunk pass only, AF2's recycle_mode='last'.
    # Off is AF3's own behaviour -- the recycle loop has no stop_gradient on
    # `prev`, so a gradient flows through every pass. Measured on the AF2 side,
    # where the same mistake cost 3x the interface quality (plddt 0.598->0.760,
    # iptm 0.167->0.522) AND 1.6x the runtime, so for design this wants to be
    # True. It was read by getattr and never declared here, which made it
    # invisible to anyone reading the config.
    recycle_last_only: bool = False
    # the confidence heads' logits per sample (several GiB for the pair ones on a large fold)
    return_confidence_logits: bool = False

  def __init__(self, config: Config, name: str = 'diffuser'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config
    self.diffusion_module = diffusion_head.DiffusionHead(
        self.config.heads.diffusion, self.global_config
    )
    if self.global_config.model == 'bioemu2':
      self.bioemu2 = bioemu2_network.BioEmu2()

  @hk.transparent
  def _sample_diffusion(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      *,
      sample_config: diffusion_head.SampleConfig,
      diffusion_state=None,
  ) -> dict[str, jnp.ndarray]:
    # ONCE, not once per sampling step per sample. The pair conditioning reads
    # only the trunk embeddings and the batch -- the noise level enters the
    # single half -- so building it inside the denoiser meant a LayerNorm, a
    # projection and two transition blocks over (num_tokens, num_tokens,
    # pair_channel), plus an (L, L, 139) relative encoding, were rebuilt for
    # every step of every sample: ~1000 times in a default fold.
    if diffusion_state is not None and len(diffusion_state) == 4:
      # The caller already has the conditioning from the diff_cond stage.
      # Recomputing it per step is what the comment above forbids: a LayerNorm,
      # a projection and two transition blocks over (num_tokens, num_tokens,
      # pair_channel) for every step of every sample.
      _, _, pair_cond, atom_cond = diffusion_state
      diffusion_state = diffusion_state[:2]
    else:
      pair_cond, atom_cond = self.diffusion_module(
          positions_noisy=None,        # unused on this path
          noise_level=jnp.zeros(()),   # unused by the pair half
          batch=batch,
          embeddings=embeddings,
          use_conditioning=True,
          conditioning_only=True,
      )

    override = diffusion_head.SAMPLER_OVERRIDE
    if batch.predicted_structure_info.atom_mask.shape[-1] != 24:
      override = None   # sampler kernels lay out 24 atom slots; boltz2's widened layout keeps the XLA sampler
    sampler = None if override is None or hk.running_init() else override(
        self.global_config.model, sample_config)
    if sampler is not None:
      return sampler(hk.experimental.get_params(), batch, embeddings, pair_cond,
                     atom_cond, hk.next_rng_key())

    denoising_step = functools.partial(
        self.diffusion_module,
        batch=batch,
        embeddings=embeddings,
        use_conditioning=True,
        pair_cond=pair_cond,
        atom_cond=atom_cond,
    )

    # LIVE PATH. `stepwise` hands back the conditioning and the initial carry
    # instead of running the loop, so a caller can apply one step at a time and
    # see every frame as it is produced -- which a scan cannot do, and which
    # `return_trajectory` cannot either, since that only yields the frames once
    # the whole loop has finished.
    if sample_config.stepwise and diffusion_state is None:
      init, noise_levels = diffusion_head.sample_init(
          batch, hk.next_rng_key(), sample_config, self.global_config)
      return {'pair_cond': pair_cond, 'atom_cond': atom_cond,
              'init': init, 'noise_levels': noise_levels}
    if diffusion_state is not None:
      # `noise_level` is the PAIR the scanned path feeds as xs: (sigma_i,
      # sigma_{i-1}). The previous level used to ride in the carry, which made
      # it per-sample and re-traced the whole noise/AdaLN half of the step at
      # num_samples * num_tokens; it is a scan input on both paths now, so the
      # driver has to hand over both.
      carry, noise_level = diffusion_state
      if diffusion_head.DENOISED_HOOK is not None:
        raise NotImplementedError('DENOISED_HOOK guides the fused sampler only, not a stepped (live) run')
      body = diffusion_head.make_denoising_body(
          denoising_step, batch.predicted_structure_info.atom_mask,
          sample_config, self.global_config,
          self.global_config.model == 'chai1',
          gamma_scale=diffusion_head.gamma_scale_for(batch, self.global_config))
      body = hk.vmap(body, in_axes=(0, None), split_rng=not hk.running_init())
      if jnp.ndim(noise_level[0]) == 0:
        carry, (positions, denoised) = body(carry, noise_level)
        return {'carry': carry, 'atom_positions': positions,
                'denoised': denoised}
      # A CHUNK of steps in one dispatch. One step per call left ~21 ms of
      # dispatch on top of the step's own ~12 ms, which is most of live mode's
      # remaining cost; scanning k steps divides that by k while `ys` still
      # returns every intermediate frame, so nothing is lost to the animation.
      # unroll=1 for the same reason sample() uses it: one body copy, so
      # compile does not track the chunk length.
      carry, (traj, denoised) = hk.scan(body, carry, noise_level, unroll=1)
      return {'carry': carry, 'atom_positions': traj, 'denoised': denoised}

    sample = diffusion_head.sample(
        denoising_step=denoising_step,
        batch=batch,
        key=hk.next_rng_key(),
        config=sample_config,
        global_config=self.global_config,
    )
    return sample

  @hk.transparent
  def _bioemu2(self, batch, key, structure=True, stage='all', recycle_carry=None,
               diffusion_state=None):
    """BioEmu2: its own trunk and denoiser (network/bioemu2.py) on its own
    features (pipeline/bioemu2_features.py, attached under `bioemu2/`), and
    OUR sampler on its flat atom layout, viewed as (atoms, 1 slot) so the
    sampler's (token, slot) reductions apply unchanged.

    Samples come back in the AF3 token-atom layout through `bioemu2/af3_index`,
    so the output writer is the shared one. There is no confidence head: the
    result carries no confidence keys and becomes the structure-only output.

    The staged entry points (staged.staged_fold, live mode) carry: `embed` ->
    {s_inputs, s_init, z_init, single, pair}; `trunk` one pass of it;
    `diff_cond` the static context and the initial sampler state; `denoise`
    steps on the flat layout, frames handed back in the AF3 layout; `score`
    the result. Propagation runs fused only.
    """
    f = {k[len('bioemu2/'):]: v for k, v in batch.items() if k.startswith('bioemu2/')}
    steer = {k[len('bioemu2steer/'):]: v for k, v in batch.items() if k.startswith('bioemu2steer/')}
    preset = next((k[len('preset_'):] for k in steer if k.startswith('preset_')), 'edm')
    steer = {k: v for k, v in steer.items() if not k.startswith('preset_')}
    for k in ('atom_pad_mask', 'token_pad_mask', 'atom_mask', 'is_dynamics_conditioned'):
      f[k] = jnp.asarray(f[k]).astype(bool)
    af3_index = f.pop('af3_index')
    af3_token = f.pop('af3_token')
    x0 = f.pop('x0', None)
    n_prop = f.pop('propagation_steps').shape[0] if 'propagation_steps' in f else 0

    def _centre(x, m):
      # bioemu2.sampling centres each sample on its real atoms (propagation,
      # superposed on the previous frame instead, does not)
      w = m[..., None].astype(jnp.float32)
      return (x - (x * w).sum(-3, keepdims=True) / w.sum()) * w
    b = self.bioemu2
    config = self.config.heads.diffusion.eval
    if config.steps < 2:   # edm_sample_schedule
      raise ValueError(f'BioEmu2 samples with at least 2 steps, got {config.steps}')
    if n_prop > 1 and preset in ('edm_fk', 'edm_fkc'):
      # Resampling reorders particles mid-step, so trajectory i's next frame
      # would come from another lineage; BioEmu2's propagation refuses it too.
      raise ValueError(f'BioEmu2 multi-step propagation does not support {preset}'
                       ' (it resamples); use edm or edm_gd, or one propagation step')
    mask = f['atom_mask'][:, None]

    class _Info:      # all the sampler reads of a batch
      class predicted_structure_info:   # noqa: N801
        atom_mask = mask

    idx = jnp.maximum(af3_index, 0)
    valid = af3_index >= 0

    def to_af3(flat):                     # (..., N, 1, 3) -> (..., T_af3, A, 3)
      return jnp.where(valid[..., None], flat[..., idx, 0, :], 0.0)

    def contact_probs(z):
      logits = b.distogram(z)
      # P(d < 8 A). BioEmu2's inference code never states its bins; these are
      # Boltz's (64 bins over 2-22 A), which its distogram head comes from.
      edges = np.linspace(2.0, 22.0, 63)
      contact = jax.nn.softmax(logits, -1)[..., :int((edges <= 8.0).sum())].sum(-1)
      tok = jnp.maximum(af3_token, 0)
      return logits, jnp.where((af3_token[:, None] >= 0) & (af3_token[None, :] >= 0),
                               contact[tok[:, None], tok[None, :]], 0.0)

    def denoising_step_for(ctx, s_inputs, s):
      def denoising_step(positions_noisy, noise_level, companion=None):
        fs = f if companion is None else {**f, 'coords_cond': companion[:, 0]}
        den = b.denoise(fs, ctx, s_inputs, s, positions_noisy[:, 0], noise_level)
        return den[:, None]
      return denoising_step

    def result(flat, s, z, s_inputs, extra=None):
      logits, contact = contact_probs(z)
      output = {'distogram': {'contact_probs': contact},
                'bioemu2': {'single': s, 'pair': z, 's_inputs': s_inputs, 'distogram': logits}}
      if flat is not None:
        positions = to_af3(flat)
        output['diffusion_samples'] = {
            'atom_positions': positions,
            'mask': jnp.tile(valid[None], (positions.shape[0], 1, 1)).astype(jnp.float32),
            'bioemu2_flat': flat[..., 0, :],      # BioEmu2's own atom order, for its parity harness
            **(extra or {})}
      return output

    # -- staged ---------------------------------------------------------------
    if stage != 'all':
      if n_prop or preset != 'edm':
        raise NotImplementedError('bioemu2 propagation and steering run fused only, not staged')
      if stage == 'embed':
        s_inputs = b.input_embedder(f)
        s_init, z_init = b.trunk_init(f, s_inputs)
        return {'s_inputs': s_inputs, 's_init': s_init, 'z_init': z_init,
                'single': jnp.zeros_like(s_init), 'pair': jnp.zeros_like(z_init)}, key
      c = recycle_carry
      if stage == 'trunk':
        sng, pr = b.trunk_pass(f, c['s_inputs'], c['s_init'], c['z_init'], c['single'], c['pair'])
        c = {**c, 'single': sng, 'pair': pr}
        return c, key, contact_probs(pr)[1]
      if stage == 'diff_cond':
        ctx = b.static_context(f, c['single'], c['pair'])
        init, levels = diffusion_head.sample_init(_Info, key, config, self.global_config)
        return {'init': init, 'noise_levels': levels, 'pair_cond': ctx['z'],
                'atom_cond': {'q0': ctx['q0'], 'p': ctx['p']}}
      if stage == 'denoise':
        dcarry, noise_level, pair_cond, atom_cond = diffusion_state
        ctx = {'z': pair_cond, 'q0': atom_cond['q0'], 'p': atom_cond['p']}
        body = diffusion_head.make_denoising_body(
            denoising_step_for(ctx, c['s_inputs'], c['single']), mask, config,
            self.global_config, False)
        body = hk.vmap(body, in_axes=(0, None), split_rng=not hk.running_init())
        if jnp.ndim(noise_level[0]) == 0:
          dcarry, (pos, den) = body(dcarry, noise_level)
        else:
          dcarry, (pos, den) = hk.scan(body, dcarry, noise_level, unroll=1)
        return {'carry': dcarry, 'atom_positions': to_af3(pos), 'denoised': to_af3(den)}
      if stage == 'score':
        return result(diffusion_state[0], c['single'], c['pair'], c['s_inputs'])
      raise ValueError(f'bioemu2 has no stage {stage!r}')

    # -- fused ----------------------------------------------------------------
    s_inputs = b.input_embedder(f)
    s_init, z_init = b.trunk_init(f, s_inputs)
    s, z = jnp.zeros_like(s_init), jnp.zeros_like(z_init)
    if hk.running_init():
      s, z = b.trunk_pass(f, s_inputs, s_init, z_init, s, z)
    else:
      n = num_trunk_passes(self.config.num_recycles, 'bioemu2')
      s, z = hk.fori_loop(0, n, lambda _, sz: b.trunk_pass(f, s_inputs, s_init, z_init, *sz),
                          (s, z))
    if not structure:
      return result(None, s, z, s_inputs)
    ctx = b.static_context(f, s, z)
    denoising_step = denoising_step_for(ctx, s_inputs, s)
    extra = {}
    if preset != 'edm':
      from .network import bioemu2_steering as steering
      flat_mask = f['atom_mask']

      def denoise_particles(x, sigma, comp):
        sig = jnp.broadcast_to(jnp.asarray(sigma, jnp.float32), x.shape[:1])

        def one(xp, sp, cp):
          fs = f if cp is None else {**f, 'coords_cond': cp}
          return b.denoise(fs, ctx, s_inputs, s, xp, sp)
        if comp is None:
          return jax.vmap(lambda xp, sp: one(xp, sp, None))(x, sig)
        return jax.vmap(one)(x, sig, comp)

      def sample_steered(k, comp):
        kw = dict(num_steps=config.steps, companion=comp,
                  injected=steering.INJECT, specs=steering.active_specs())
        if preset == 'edm_fkc':
          x, bad = steering.fkc_sample(denoise_particles, steer, flat_mask, config.num_samples, k, **kw)
          # the first step with a non-finite energy or log-weight (-1: none);
          # run_alphafold raises on it, as edm_fkc_sampler does
          extra['bioemu2_nonfinite_step'] = bad
          return x
        return steering.steered_sample(
            denoise_particles, steer, flat_mask, config.num_samples, k,
            step_scale=config.step_scale, noise_scale=config.noise_scale,
            resampling_interval=None if preset == 'edm_gd' else 3, **kw)
      if not n_prop:
        flat = _centre(sample_steered(key, None)[:, :, None], flat_mask[:, None])
      else:
        w = flat_mask[:, None, None].astype(jnp.float32)
        prev = jnp.broadcast_to(x0[None], (config.num_samples,) + x0.shape)
        frames = []
        for k in range(n_prop):
          centre = (prev * w[:, :, 0][None]).sum(1, keepdims=True) / w.sum()
          end = sample_steered(jax.random.fold_in(key, k), (prev - centre) * w[:, :, 0][None])
          end = steering._rigid_align(end, prev, flat_mask)   # pylint: disable=protected-access
          frames.append(end)
          prev = end
        flat = jnp.stack(frames, 1).reshape((-1,) + x0.shape)[:, :, None]
      return result(flat, s, z, s_inputs, extra)
    if not n_prop:
      sample = diffusion_head.sample(
          denoising_step=denoising_step, batch=_Info, key=key,
          config=config, global_config=self.global_config)
      flat = _centre(sample['atom_positions'], mask)        # (samples, N, 1, 3)
      if 'trajectory' in sample:
        extra['trajectory'] = sample['trajectory']
    else:
      # PROPAGATION (bioemu2.propagation): each sample a trajectory of n_prop
      # lag-time steps from x0. A step samples conditioned on the previous
      # frame (centred, carried through the sampler as its companion), then
      # superposes the endpoint on that frame, as propagate_with_embeddings
      # does. Sample i of the output is trajectory i // n_prop, step
      # i % n_prop + 1.
      w = mask[..., None].astype(jnp.float32)            # (N, 1, 1)
      prev = jnp.broadcast_to(x0[None, :, None], (config.num_samples,) + mask.shape + (3,))
      frames = []
      for k in range(n_prop):
        centre = (prev * w).sum(-3, keepdims=True) / w.sum()
        sample = diffusion_head.sample(
            denoising_step=denoising_step, batch=_Info, key=jax.random.fold_in(key, k),
            config=config, global_config=self.global_config,
            companion=(prev - centre) * w)
        end = jax.vmap(lambda p, r: diffusion_head._kabsch(p, r, mask))(   # pylint: disable=protected-access
            sample['atom_positions'], prev)
        frames.append(end)
        prev = end
      flat = jnp.stack(frames, 1).reshape((-1,) + mask.shape + (3,))
    return result(flat, s, z, s_inputs, extra)

  @hk.transparent
  def _structural_expand_refine(self, embeddings, residue_batch, struct_data,
                                struct_mask):
    """OpenDDE: residue trunk embeddings -> refined structural-token embeddings.

    Runs the StructuralTokenExpander on (target_feat, single, pair), then a 4-block
    PairformerStack refiner with the expander's structural pair attention bias. All
    gated by the opendde caller; returns (s_struct, z_struct, target_feat_struct).
    """
    gc = self.global_config
    c_s = self.config.evoformer.seq_channel
    c_z = self.config.evoformer.pair_channel
    c_s_inputs = embeddings['target_feat'].shape[-1]
    book = {k[len('structbook/'):]: v for k, v in struct_data.items()
            if k.startswith('structbook/')}
    batch_struct = {
        'parent_residue_idx': book['parent_residue_idx'],
        'subtoken_role_id': book['subtoken_role_id'],
        'prev_parent_residue_idx': book['prev_parent_residue_idx'],
        'next_parent_residue_idx': book['next_parent_residue_idx'],
        'residue_index': residue_batch.token_features.residue_index,
        'asym_id': residue_batch.token_features.asym_id,
    }
    tf_struct, s_struct, z_struct, attn_bias = (
        structural_tokens.StructuralTokenExpander(c_s, c_z, c_s_inputs, gc)(
            batch_struct, embeddings['target_feat'], embeddings['single'],
            embeddings['pair']))

    # 4-block refiner (identical block to the trunk pairformer). n_heads=8 single,
    # tri-att 12 heads via hidden_scale_up, pair transition x2 / single transition x4.
    ref_cfg = pairformer_modules.PairFormerIteration.Config(
        num_layer=1,
        pair_attention=pairformer_modules.GridSelfAttention.Config(num_head=c_z // 32),
        single_attention=diffusion_transformer.SelfAttentionConfig(num_head=8),
        pair_transition=pairformer_modules.TransitionBlock.Config(num_intermediate_factor=2),
        single_transition=pairformer_modules.TransitionBlock.Config(num_intermediate_factor=4))
    ref_cfg.shard_transition_blocks = False
    seq_mask = struct_mask.astype(jnp.float32)
    pair_mask = seq_mask[:, None] * seq_mask[None, :]

    def blk(carry):
      zz, ss = carry
      return pairformer_modules.PairFormerIteration(
          ref_cfg, gc, with_single=True, name='trunk_pairformer')(
              zz, pair_mask, single_act=ss, seq_mask=seq_mask,
              extra_pair_bias=attn_bias)

    z_struct, s_struct = hk.experimental.layer_stack(
        4, name='structural_token_refiner')(blk)((z_struct, s_struct))
    return s_struct, z_struct, tf_struct, attn_bias

  def __call__(
      self,
      batch: features.BatchDict,
      key: jax.Array | None = None,
      soft_seq=None,
      design_mask=None,
      structure=True,
      use_dropout=False,
      num_trunk_passes_override=None,
      recycle_carry=None,
      stage='all',
      diffusion_state=None,
  ) -> ModelResult:
    """ColabDesign2: soft_seq is the continuous relaxation of sequence.

    (num_tokens, 20) over the standard amino acids, or None for prediction.
    design_mask selects which tokens it replaces; everything else keeps its true
    identity, which is what a binder target, a scaffolded motif, and any DNA or
    ligand token require. See featurization.blend_soft.

    structure=False returns the trunk and the distogram only, leaving the
    diffusion sampler and the confidence head out of the graph entirely -- which
    is what a design loop wants and what makes it fit in memory.
    """
    if key is None:
      key = hk.next_rng_key()

    if self.global_config.model == 'bioemu2':
      return self._bioemu2(batch, key, structure=structure, stage=stage,
                           recycle_carry=recycle_carry, diffusion_state=diffusion_state)


    # OpenDDE structural-token expansion: featurise_spec attaches the structural
    # batch under 'struct/' + the expander bookkeeping under 'structbook/'. Pull
    # them aside; the trunk runs on the residue batch, the diffusion on the
    # structural one (see the opendde branch below). Gated by opendde so AF3/OF3/
    # IF2 -- which never carry these keys -- are unaffected.
    opendde = self.global_config.model == 'opendde'
    # Copy before the del below: the caller's dict is reused across seeds, and
    # deleting from it would leave the second seed without a structural batch.
    batch = dict(batch)
    _struct = {k: batch[k] for k in list(batch)
               if k.startswith('struct/') or k.startswith('structbook/')}
    for k in _struct:
      del batch[k]
    has_structural = opendde and any(
        k.startswith('structbook/') for k in _struct)
    # protenix's native randomness replayed per seed (pipeline.protenix_rng); read by the evoformer
    native_rng = {k[len('px_'):]: batch.pop(k) for k in list(batch) if k.startswith('px_')} or None
    # A vendor's random draws, injected by a parity harness (vendor_native_parity) where ours has no draw to replay.
    # Never present in a real batch.
    injected = {k[len('inject/'):]: batch.pop(k) for k in list(batch) if k.startswith('inject/')}

    batch = feat_batch.Batch.from_data_dict(batch)
    # opt-in: masked, ours folded bacteriorhodopsin's trimer on 1 of 10 seeds where native did on 4 (unmasked, ours 7 of
    # 10); the mask itself matches native's stack, the cause is open
    rate = model_config.MSA_COLUMN_MASK_RATE.get(self.global_config.model, 0.0) if os.environ.get(
        'AF3_ESM_COLUMN_MASK') else 0.0
    if rate:
      # drawn from the fold's key, so a seed reproduces it; every recycle sees the same columns
      col_keep = jax.random.uniform(jax.random.fold_in(key, 0xC0), batch.msa.mask.shape[1:]) >= rate
      keep = (jnp.arange(batch.msa.mask.shape[0]) == 0)[:, None] | col_keep[None, :]
      batch = dataclasses.replace(batch, msa=dataclasses.replace(
          batch.msa, mask=batch.msa.mask * keep.astype(batch.msa.mask.dtype)))

    # DENOISE RETURNS HERE, before the prologue. `create_target_feat_embedding`
    # below is an atom-level transformer over every atom, and entering
    # __call__ from the top ran it on EVERY denoise step -- 200 times in a
    # default fold, for a value the trunk carry already holds. Measured at 32
    # tokens: 0.17 s a step against the step's own ~0.012 s, i.e. the
    # conditioning prologue was 90% of live mode.
    #
    # Everything the score model needs is in the carry (single, pair,
    # target_feat) plus the conditioning the caller passes in, so nothing above
    # has to run again.
    if stage == 'denoise':
      if recycle_carry is None or diffusion_state is None:
        raise ValueError("stage='denoise' needs the trunk carry and a state")
      # 🔴 A STRUCTURAL-TOKEN MODEL DENOISES ON ITS OWN BATCH. opendde diffuses
      # in its `struct/` layout (160 tokens against 68 residues on a 68-residue
      # input), so the residue batch here handed its positions a residue-shaped
      # mask: "Incompatible shapes: mask_shape=(68, 24, 1), value_shape=(160,
      # 24, 3)" in random_augmentation, and live mode could not run opendde. The
      # batch is rebuilt from the struct/ keys - cheap - and the carry the
      # driver passes for such a model is diff_cond's `denoise_carry`, the
      # expander and refiner's embeddings, which are not rebuilt per step.
      if has_structural:
        batch = feat_batch.Batch.from_data_dict(
            {k[len('struct/'):]: v for k, v in _struct.items()
             if k.startswith('struct/')})
      return self._sample_diffusion(  # pyrefly: ignore[bad-return]
          batch, recycle_carry,
          sample_config=self.config.heads.diffusion.eval,
          diffusion_state=diffusion_state)

    embedding_module = evoformer_network.Evoformer(
        self.config.evoformer, self.global_config
    )
    # ColabDesign2: the continuous relaxation of sequence, or None for
    # prediction. See featurization.blend_soft.
    embedding_module.soft_seq = soft_seq
    embedding_module.design_mask = design_mask
    embedding_module.native_rng = native_rng
    # REUSED from the carry when the caller has it. This is an atom-level
    # transformer over every atom and it does not depend on the recycle state,
    # so the fused path computes it once before the loop. A stepwise caller
    # re-enters here for every trunk pass and for the conditioning and scoring
    # stages -- six times in a default fold -- and the value was then
    # overwritten by the carry's own copy anyway. Measured at 32 tokens: 7.1 s
    # of live mode against 2.4 s fused, almost all of it this.
    if recycle_carry is not None and 'target_feat' in recycle_carry:
      target_feat = recycle_carry['target_feat']
    else:
      target_feat = create_target_feat_embedding(
          batch=batch,  # pyrefly: ignore[bad-argument-type]
          config=embedding_module.config,
          global_config=self.global_config,
          soft_seq=soft_seq,
          design_mask=design_mask,
      )
    # chai projects the token embedding twice -- see
    # model_config.SEPARATE_STRUCTURE_TARGET_FEAT. The trunk gets the first and
    # the diffusion module the second, applied to `diff_emb` below.
    target_feat_structure = None
    if isinstance(target_feat, tuple):
      target_feat, target_feat_structure = target_feat

    def recycle_body(_, args):
      prev, key = args
      key, subkey = jax.random.split(key)
      embeddings = embedding_module(
          batch=batch,
          prev=prev,
          target_feat=target_feat,
          key=subkey,
          use_dropout=use_dropout,
      )
      embeddings['pair'] = embeddings['pair'].astype(jnp.float32)
      embeddings['single'] = embeddings['single'].astype(jnp.float32)
      return embeddings, key

    num_res = batch.num_res  # pyrefly: ignore[missing-attribute]

    embeddings = {
        'pair': jnp.zeros(
            [num_res, num_res, self.config.evoformer.pair_channel],
            dtype=jnp.float32,
        ),
        'single': jnp.zeros(
            [num_res, self.config.evoformer.seq_channel], dtype=jnp.float32
        ),
        'target_feat': target_feat,
    }
    if self.global_config.model in model_config.PAIR_ONLY_TRUNK:
      # ESMFold2's parcae recurrence carries the trunk stack's output, while the
      # `pair` the trunk RETURNS has been through parcae_readout + the coda. Two
      # different tensors, so two carry entries; seeded with zeros like `pair`.
      embeddings['pair_pre_coda'] = jnp.zeros(
          [num_res, num_res, self.config.evoformer.pair_channel],
          dtype=jnp.float32)
      if 'pair_state0' in injected:
        # native draws this state, trunc_normal(std=sqrt(2 / (5 c)), +-3 std)
        # (esm EsmFold2Model._init_pair_state); ours starts from its mean
        embeddings['pair_pre_coda'] = jnp.asarray(injected['pair_state0'], jnp.float32)
    if self.global_config.model == 'chai1':
      # chai's diffusion conditions on z_init as well as z_trunk. The recycle
      # loop carries `embeddings` as a scan carry, so this key has to exist in
      # the INITIAL dict too or the carry pytree changes shape on the first
      # iteration. Seeded with zeros; the evoformer overwrites it every pass.
      embeddings['pair_init'] = jnp.zeros(
          [num_res, num_res, self.config.evoformer.pair_channel],
          dtype=jnp.float32)
      # chai seeds the recycle carry with the initial reprs rather than zeros
      # (chai1.py: `token_pair_trunk_repr = token_pair_initial_repr` before the
      # loop). The evoformer cannot see the iteration index inside a fori_loop,
      # so flag the first pass in the carry and let it substitute z_init/s_init.
      embeddings['recycle_first'] = jnp.ones((), jnp.float32)
    # main's per-pass counter for the protenix RNG replay; part of the INITIAL
    # carry, so the staged path (stage='embed' then 'trunk') carries it too.
    if native_rng is not None:
      embeddings['px_pass'] = jnp.zeros((), jnp.int32)
    # The INITIAL carry: zeros for the recycled tensors plus target_feat, which
    # is the atom-level input embedding and does not depend on the recycle
    # state. Its own trace is cheap -- no trunk in it -- and it exists so every
    # trunk call can pass a FULL carry. Before this the first call passed None
    # and the rest passed a dict, which is two jit signatures for one
    # computation: measured 24.9 s for the first trace and 19.8 s for the
    # second, on a cold T4.
    if stage == 'embed':
      return embeddings, key  # pyrefly: ignore[bad-return]

    # ONE PASS, jitted once and driven from Python (stage='trunk'). The recycle
    # count then never reaches the graph at all -- one executable serves any
    # number of passes, and the caller sees the embeddings between them, which
    # is what lets a notebook show what each recycle changed. The AF2 runner in
    # ColabDesign2 drives its recycles the same way.
    #
    # `stage` is a Python string, so each stage is its own traced function; the
    # carry is a device pytree that never needs to come back to the host.
    if stage == 'trunk':
      if recycle_carry is not None:
        # keep the initialised keys (target_feat, and the per-model extras
        # seeded above) and overwrite only what a pass actually carries
        embeddings = {**embeddings, **recycle_carry}
      # The ADVANCED key comes back with the embeddings. The fused loop carries
      # its key in the scan carry, so a Python driver that split its own keys
      # instead produced a different sample -- measured at 1.098 A on a
      # 32-residue peptide, which is not noise. Threading the same key through
      # makes the two paths draw identically.
      embeddings, key = recycle_body(None, (embeddings, key))
      # The contact map comes back with every pass, so the caller can show the
      # trunk converging. The distogram head already computes contact_probs,
      # and it is one linear over the pair representation -- cheap enough to
      # run per pass, which is why the head lives with the trunk at all.
      dgram = distogram_head.DistogramHead(
          self.config.heads.distogram, self.global_config
      )(batch, embeddings, return_distogram=False)
      return embeddings, key, dgram['contact_probs']  # pyrefly: ignore[bad-return]
    # 'score' belongs here too. It fell through to the branch below, which
    # means it RAN THE WHOLE RECYCLE LOOP AGAIN -- four more trunk passes and a
    # second compile of the trunk -- before reaching the confidence head. That
    # is why scoring appeared to take 31.8 s on a cold T4 while the confidence
    # head itself is 81 ms warm: almost none of it was the head.
    if stage in ('heads', 'score'):
      if recycle_carry is None:
        raise ValueError(f"stage={stage!r} needs the trunk's embeddings")
      embeddings = {**embeddings, **recycle_carry}
    elif hk.running_init():
      embeddings, _ = recycle_body(None, (embeddings, key))
    else:
      num_iter = num_trunk_passes(self.config.num_recycles,
                                  self.global_config.model)
      # A TRACED trip count, when the caller supplies one: the recycle count is
      # otherwise a Python int baked into the HLO, so every value compiles its
      # own executable and a persistent compile cache built at one recycle
      # setting misses at every other (measured: 10/5/3/1 cached, 4 alone paid
      # 31 s). Passed in as an argument, one executable serves them all.
      #
      # PREDICTION ONLY. `lax.fori_loop` with a dynamic bound is not
      # reverse-differentiable, and the design path backprops through the trunk
      # -- see recycle_last_only below. Callers that need gradients leave this
      # None and keep the static count.
      if num_trunk_passes_override is not None:
        num_iter = num_trunk_passes_override
      # ColabDesign2: AF3's recycle loop has no stop_gradient on `prev`, so with
      # num_recycles>0 the gradient backprops through EVERY pass -- the exact
      # mistake that cost the AF2 design path 3x its interface quality. When
      # recycle_last_only is set (AF2's recycle_mode='last'), run all but the
      # final pass and DETACH their output, so only the last pass is
      # differentiated: the trunk still recycles, but the sequence gradient is
      # the clean single-pass one. Off by default -> exact original behaviour.
      # `num_iter > 1` is a Python bool only while the count is static, which is
      # exactly when this branch applies: recycle_last_only exists for design,
      # and design does not pass an override.
      if (self.config.recycle_last_only
          and num_trunk_passes_override is None and num_iter > 1):
        embeddings, key = hk.fori_loop(0, num_iter - 1, recycle_body,
                                       (embeddings, key))
        embeddings = jax.tree_util.tree_map(jax.lax.stop_gradient, embeddings)
        embeddings, _ = recycle_body(None, (embeddings, key))
      else:
        embeddings, _ = hk.fori_loop(0, num_iter, recycle_body,
                                     (embeddings, key))

    if self.global_config.model == 'chai1':
      # chai's diffusion conditions on [z_trunk | its STRUCTURE token-pair features], not on the trunk's
      # z_init -- see Chai1StructurePair
      embeddings['pair_init'] = Chai1StructurePair(self.config.evoformer.pair_channel)(
          batch.token_features, evoformer_network.chai1_restraints(batch))
      if batch.declared_bonds is not None:
        # the structure half of chai's bond projection (bond_loss_input_proj rows 256:512)
        from alphafold3.model.components import haiku_modules as hm
        embeddings['pair_init'] += hm.Linear(self.config.evoformer.pair_channel, use_bias=False,
                                             name='chai1_structure_bond')(
            batch.declared_bonds[..., None].astype(embeddings['pair_init'].dtype))

    # ColabDesign2: the distogram head stays with the trunk. It reads embeddings
    # only, it is what every design objective is built on, and it is cheap.
    distogram = distogram_head.DistogramHead(
        self.config.heads.distogram, self.global_config
    )(batch, embeddings, return_distogram=self.config.return_distogram)

    output = {'distogram': distogram}
    if self.config.return_embeddings:
      output['single_embeddings'] = embeddings['single']
      output['pair_embeddings'] = embeddings['pair']
      # the trunk's INPUTS as well as its outputs, so a port can tell a bad
      # trunk from a bad input embedder without re-running anything
      output['target_feat'] = embeddings['target_feat']
      if 'pair_init' in embeddings:
        output['pair_init'] = embeddings['pair_init']

    # ColabDesign2: everything below needs sampled coordinates, and a design
    # objective built on the distogram needs none of it. Skipping is not a
    # matter of what the sampler returns -- returning zeros still leaves the
    # confidence head in the graph, running on those zeros, producing output
    # af3_plddt refuses to score. That cost 16.39 GiB and made AF3 fixbb at
    # L=92 impossible on a 23 GiB card.
    # ColabDesign2: everything below needs sampled coordinates, and a design
    # objective built on the distogram needs none of it. Skipping is not a
    # matter of what the sampler returns -- returning zeros still leaves the
    # confidence head in the graph, running on those zeros, producing output
    # af3_plddt refuses to score. That cost 16.39 GiB and made AF3 fixbb at
    # L=92 impossible on a 23 GiB card.
    #
    # Inline rather than a method: haiku scopes parameters by the method they
    # are created in, so moving this into Model.structure_heads renamed every
    # weight to diffuser/~structure_heads/... and stopped matching the
    # checkpoint.
    if not structure:
      return output

    # OpenDDE: the diffusion + confidence run on the structural-token set. Expand
    # the residue trunk embeddings to structural tokens, refine them, and swap the
    # residue batch/embeddings for the structural ones. Everything else (AF3/OF3/
    # IF2) keeps running on the residue batch.
    if has_structural:
      diff_batch = feat_batch.Batch.from_data_dict(
          {k[len('struct/'):]: v for k, v in _struct.items()
           if k.startswith('struct/')})
      s_struct, z_struct, tf_struct, attn_bias = self._structural_expand_refine(
          embeddings, batch, _struct, diff_batch.token_features.mask)
      diff_emb = {'single': s_struct, 'pair': z_struct, 'target_feat': tf_struct,
                  'structural_pair_attn_bias': attn_bias}
    else:
      diff_batch = batch
      diff_emb = embeddings
      if target_feat_structure is not None:
        # The diffusion module's own s_inputs, and ONLY the diffusion module's:
        # the confidence head below reads `embeddings['target_feat']`, and chai
        # feeds its confidence head the TRUNK projection (measured; see
        # create_target_feat_embedding).
        diff_emb = dict(embeddings)
        diff_emb['target_feat'] = target_feat_structure

    if stage in ('diff_cond', 'denoise'):
      # Either hand back the conditioning, or apply one denoise step to the
      # state the caller passes in. Both return before the confidence heads.
      out = self._sample_diffusion(  # pyrefly: ignore[bad-return]
          diff_batch, diff_emb,
          sample_config=self.config.heads.diffusion.eval,
          diffusion_state=diffusion_state)
      # ...and the embeddings a structural-token model's steps must denoise
      # on, so the per-step path need not run the expander and refiner again.
      if has_structural and stage == 'diff_cond' and 'init' in out:
        out = {**out, 'denoise_carry': diff_emb}
      return out

    if diffusion_state is not None and stage == 'score':
      # Coordinates the caller produced by driving the steps itself; scored
      # through exactly the path a fused sample takes. The mask is derived here
      # rather than passed in -- the caller has no business reconstructing it.
      pos = diffusion_state[0]
      samples = {
          'atom_positions': pos,
          'mask': jnp.tile(
              diff_batch.predicted_structure_info.atom_mask[None],
              (pos.shape[0], 1, 1)),
      }
    else:
      samples = self._sample_diffusion(
          diff_batch,
          diff_emb,
          sample_config=self.config.heads.diffusion.eval,
      )

    if has_structural:
      # the confidence head reads the residue trunk (OpenDDE's pair_output_space='residue'), not the structural one
      book = {k[len('structbook/'):]: v for k, v in _struct.items()
              if k.startswith('structbook/')}
      samples = dict(samples)
      samples['atom_positions'], atom_valid = _structural_atoms_to_residue(
          samples['atom_positions'], book)
      n_res, max_atoms = atom_valid.shape
      # the pLDDT/resolved einsum runs over the dense (token, atom-slot) layout
      a2t = jnp.repeat(jnp.arange(n_res), max_atoms)
      a2ta = jnp.tile(jnp.arange(max_atoms), n_res)
      def _conf(dense_atom_positions):
        rep = atom_layout.convert(
            batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
            dense_atom_positions, layout_axes=(-3, -2))         # (n_res, 3)
        return opendde_confidence.OpenDDEConfidenceHead(
            self.config.evoformer.seq_channel, self.config.evoformer.pair_channel,
            embeddings['target_feat'].shape[-1], self.global_config)(
                embeddings['target_feat'], embeddings['single'], embeddings['pair'], rep,
                a2t, a2ta, batch.token_features.mask)
      logits = mapping.sharded_map(_conf, in_axes=0)(samples['atom_positions'])
      confidence_output = _opendde_confidence(
          logits, atom_valid,
          seq_mask=batch.token_features.mask,
          asym_id=batch.token_features.asym_id,
          num_plddt_bins=logits['predicted_lddt_logits'].shape[-1],
          num_pae_bins=logits['predicted_aligned_error_logits'].shape[-1],
          num_pde_bins=logits['predicted_distance_error_logits'].shape[-1],
          return_logits=self.config.return_confidence_logits)
    elif self.global_config.model in model_config.NO_CONFIDENCE_HEAD:
      # No head in the checkpoint, so none in the graph: this model predicts a
      # structure and nothing about it. The keys are simply absent, which is
      # honest and is what keeps the converter's coverage exact.
      confidence_output = {}
    else:
      head_config = self.global_config
      if (self.global_config.model in model_config.FP32_CONFIDENCE_HEAD
          and diff_batch.token_features.mask.shape[0]
          <= model_config.FP32_CONFIDENCE_HEAD[self.global_config.model]):
        # see model_config.FP32_CONFIDENCE_HEAD
        head_config = copy.copy(self.global_config)
        head_config.bfloat16 = 'none'
      # Compute dist_error_fn over all samples for distance error logging.
      confidence_output = mapping.sharded_map(
          lambda dense_atom_positions: confidence_head.ConfidenceHead(
              self.config.heads.confidence, head_config
          )(
              dense_atom_positions=dense_atom_positions,
              # the TRUNK projection of the token embedding, as chai's confidence head reads it; diff_emb carries the
              # diffusion module's own (target_feat_structure) and left pLDDT 3.7 low and pTM 0.38 against 0.54
              embeddings=embeddings,
              seq_mask=diff_batch.token_features.mask,
              token_atoms_to_pseudo_beta=diff_batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
              asym_id=diff_batch.token_features.asym_id,
              # boltz-2 rebuilds z inside the confidence head from relative position,
              # bonds and restraints, so it needs the token features and the bond
              # matrix -- unused by every other model's head.
              token_features=diff_batch.token_features,
              bond_matrix=evoformer_network.token_bond_matrix(
                  diff_batch, symmetrize=True),
              bond_type_matrix=evoformer_network.token_bond_type_matrix(
                  diff_batch, symmetrize=True),
              # chai gathers its pLDDT logits by ATOM37 slot, which is not our
              # dense slot order; the names are how we recover the permutation.
              atom_name_chars=diff_batch.ref_structure.atom_name_chars,
              use_dropout=use_dropout,
              return_logits=self.config.return_confidence_logits,
              # helixfold3 re-encodes the predicted atoms with an atom encoder
              batch=diff_batch,
          ),
          in_axes=0,
      )(samples['atom_positions'])

    output['diffusion_samples'] = samples
    output.update(confidence_output)
    return output

  @classmethod
  def get_inference_result(
      cls,
      batch: features.BatchDict,
      result: ModelResult,
      target_name: str = '',
      model_name: str | None = None,
  ) -> Iterable[InferenceResult]:
    """Get the predicted structure, scalars, and arrays for inference.

    `model_name` selects a vendor's own confidence summary where it is not AF3's
    (helixfold3: frame-weighted pTM/ipTM, ipTM 0 without an interface, its own
    clash test and ranking score).

    This function also computes any inference-time quantities, which are not a
    part of the forward-pass, e.g. additional confidence scores. Note that this
    function is not serialized, so it should be slim if possible.

    Args:
      batch: data batch used for model inference, incl. TPU invalid types.
      result: output dict from the model's forward pass.
      target_name: target name to be saved within structure.

    Yields:
      inference_result: dataclass object that contains a predicted structure,
      important inference-time scalars and arrays, as well as a slightly trimmed
      dictionary of raw model result from the forward pass (for debugging).
    """
    del target_name
    batch = feat_batch.Batch.from_data_dict(batch)  # pyrefly: ignore[bad-assignment]

    # Retrieve structure and construct a predicted structure.
    pred_structure = get_predicted_structure(result=result, batch=batch)  # pyrefly: ignore[bad-argument-type]

    num_tokens = batch.token_features.seq_length.item()  # pyrefly: ignore[missing-attribute]

    pae_single_mask = np.tile(
        batch.frames.mask[:, None],  # pyrefly: ignore[missing-attribute]
        [1, batch.frames.mask.shape[0]],  # pyrefly: ignore[missing-attribute]
    )
    hf3 = model_name == 'helixfold3'
    if hf3:
      # helixfold3 weights every pair by BOTH tokens' frame masks
      # (helixfold/common/confidence.py predicted_tm_score: w_i * w_j)
      fm = np.asarray(batch.frames.mask).astype(bool)  # pyrefly: ignore[missing-attribute]
      pae_single_mask = fm[:, None] & fm[None, :]
    asym_ids = batch.token_features.asym_id[:num_tokens]  # pyrefly: ignore[missing-attribute]
    # Map asym IDs back to chain IDs. Asym IDs are constructed from chain IDs by
    # iterating over the chain IDs, and for each unique chain ID incrementing
    # the asym ID by 1 and mapping it to the particular chain ID. Asym IDs are
    # 1-indexed, so subtract 1 to get back to the chain ID.
    chain_ids = [pred_structure.chains[asym_id - 1] for asym_id in asym_ids]
    res_ids = batch.token_features.residue_index[:num_tokens]  # pyrefly: ignore[missing-attribute]

    # A model with no confidence head predicts a structure and nothing about
    # it, so everything below this point has no basis. Rather than skip the
    # outputs -- which would break every consumer, from the ranking CSV to the
    # confidence JSONs -- the same keys are emitted filled with NaN. That is
    # what "not predicted" looks like in a float field, and it keeps a
    # structure-only model a first-class citizen of the same pipeline.
    #
    # Keyed on the RESULT, not on model_config.NO_CONFIDENCE_HEAD: this is a
    # classmethod with no global_config to consult, and the absence of the keys
    # is the actual fact being handled -- __call__ sets confidence_output = {}
    # for exactly these models.
    if 'tmscore_adjusted_pae_global' not in result:
      yield from _structure_only_results(
          result, pred_structure, num_tokens, chain_ids, res_ids)
      return

    ptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],  # pyrefly: ignore[missing-attribute]
        pae_single_mask=pae_single_mask,
        interface=False,
    )
    iptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],  # pyrefly: ignore[missing-attribute]
        pae_single_mask=pae_single_mask,
        interface=True,
    )
    if hf3:
      # no interface pair with weight (a monomer, or only frameless partners): 0, not NaN
      iptm = np.nan_to_num(iptm, nan=0.0)
    px = model_name in model_config.PROTENIX_FAMILY
    if px:
      # the Protenix family's own pTM / ipTM (sample_confidence.py): rows averaged over all
      # tokens (pTM) or the other chains' (ipTM), the max over framed rows; a monomer's ipTM is 0
      ptm, iptm = confidences.protenix_ptm_iptm(
          np.asarray(result['tmscore_adjusted_pae_global'])[:, :num_tokens, :num_tokens],
          np.asarray(batch.frames.mask)[:num_tokens],  # pyrefly: ignore[missing-attribute]
          np.asarray(asym_ids))
    ptm_iptm_average = 0.8 * iptm + 0.2 * ptm

    if len(np.unique(asym_ids[:num_tokens])) > 1:
      # There is more than one chain, hence interface pTM (i.e. ipTM) defined,
      # so use it.
      ranking_confidence = ptm_iptm_average
    else:
      # There is only one chain, hence ipTM=NaN, so use just pTM.
      ranking_confidence = ptm

    contact_probs = result['distogram']['contact_probs']
    # Compute PAE related summaries.
    _, chain_pair_pae_min, _ = confidences.chain_pair_pae(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,  # pyrefly: ignore[missing-attribute]
        full_pae=result['full_pae'],
        mask=pae_single_mask,
    )
    # A head in model_config.NO_PDE_HEAD emits no PDE at all (ESMFold2's
    # lm-tier pair have pae_head and no pde_head), so every PDE-derived metric
    # is absent rather than zero. Substituting zeros here would be worse than
    # omitting them: a chain-pair PDE of 0.0 reads as a PERFECT prediction, and
    # ranking scores are built from these.
    if 'full_pde' in result:
      chain_pair_pde_mean, chain_pair_pde_min = confidences.chain_pair_pde(
          num_tokens=num_tokens,
          asym_ids=batch.token_features.asym_id,  # pyrefly: ignore[missing-attribute]
          full_pde=result['full_pde'],
      )
      intra_chain_single_pde, cross_chain_single_pde, _ = confidences.pde_single(
          num_tokens,
          batch.token_features.asym_id,  # pyrefly: ignore[missing-attribute]
          result['full_pde'],
          contact_probs,
      )
    else:
      chain_pair_pde_mean = chain_pair_pde_min = None
      intra_chain_single_pde = cross_chain_single_pde = None
    pae_metrics = confidences.pae_metrics(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,  # pyrefly: ignore[missing-attribute]
        full_pae=result['full_pae'],
        mask=pae_single_mask,
        contact_probs=contact_probs,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    ranking_confidence_pae = confidences.rank_metric(
        result['full_pae'],
        contact_probs * batch.frames.mask[:, None].astype(float),  # pyrefly: ignore[missing-attribute]
    )
    chain_pair_iptm = _compute_chain_pair_iptm(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,  # pyrefly: ignore[missing-attribute]
        mask=pae_single_mask,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    # iptm_ichain is a vector of per-chain ptm values. iptm_ichain[0],
    # for example, is just the zeroth diagonal entry of the chain pair iptm
    # matrix:
    # [[x, , ],
    #  [ , , ],
    #  [ , , ]]]
    iptm_ichain = chain_pair_iptm.diagonal(axis1=-2, axis2=-1)
    # iptm_xchain is a vector of cross-chain interactions for each chain.
    # iptm_xchain[0], for example, is an average of chain 0's interactions with
    # other chains:
    # [[ ,x,x],
    #  [x, , ],
    #  [x, , ]]]
    iptm_xchain = confidences.get_iptm_xchain(chain_pair_iptm)

    predicted_distance_errors = result.get('average_pde')

    # Computing solvent accessible area with dssp can be slow for large
    # structures with lots of chains, so we parallelize the call.
    pred_structures = pred_structure.unstack()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(pred_structures), 32)
    ) as executor:
      # helixfold3 and the protenix family flag a clash the same way: polymer chain pairs only
      has_clash = list(executor.map(
          confidences.has_clash_helixfold3 if hf3 or px else confidences.has_clash, pred_structures))
      fraction_disordered = list(
          executor.map(confidences.fraction_disordered, pred_structures)
      )

    for idx, pred_structure in enumerate(pred_structures):
      ranking_score = (
          # helixfold3's ranking_confidence (helixfold/model/utils.py)
          0.8 * iptm[idx] + 0.2 * ptm[idx] - 1.0 * has_clash[idx] if hf3 else
          # the protenix family's: its disorder term is hard-coded 0
          0.8 * iptm[idx] + 0.2 * ptm[idx] - 100.0 * has_clash[idx] if px else
          confidences.get_ranking_score(
              ptm=ptm[idx],
              iptm=iptm[idx],
              fraction_disordered_=fraction_disordered[idx],
              has_clash_=has_clash[idx],
          ))
      if hf3:
        pred_structure = confidences.helixfold3_written_atoms(pred_structure)
      if model_name == 'mol1':
        pred_structure = confidences.mol1_rebuild_cterminal_oxygens(pred_structure)
      yield InferenceResult(
          predicted_structure=pred_structure,
          numerical_data={
              # omitted entirely for a NO_PDE_HEAD model -- see above
              **({} if 'full_pde' not in result else {
                  'full_pde': result['full_pde'][idx, :num_tokens, :num_tokens],
              }),
              'full_pae': result['full_pae'][idx, :num_tokens, :num_tokens],
              'contact_probs': contact_probs[:num_tokens, :num_tokens],
          },
          metadata={  # pyrefly: ignore[bad-argument-type]
              'predicted_distance_error': (
                  None if predicted_distance_errors is None
                  else predicted_distance_errors[idx]),
              'ranking_score': ranking_score,
              'fraction_disordered': fraction_disordered[idx],
              'has_clash': has_clash[idx],
              'predicted_tm_score': ptm[idx],
              'interface_predicted_tm_score': iptm[idx],
              'chain_pair_pde_mean': (None if chain_pair_pde_mean is None
                                   else chain_pair_pde_mean[idx]),
              'chain_pair_pde_min': (None if chain_pair_pde_min is None
                                   else chain_pair_pde_min[idx]),
              'chain_pair_pae_min': chain_pair_pae_min[idx],
              'ptm': ptm[idx],
              'iptm': iptm[idx],
              'ptm_iptm_average': ptm_iptm_average[idx],
              'intra_chain_single_pde': (None if intra_chain_single_pde is None
                                   else intra_chain_single_pde[idx]),
              'cross_chain_single_pde': (None if cross_chain_single_pde is None
                                   else cross_chain_single_pde[idx]),
              'pae_ichain': pae_metrics['pae_ichain'][idx],
              'pae_xchain': pae_metrics['pae_xchain'][idx],
              'ranking_confidence': ranking_confidence[idx],
              'ranking_confidence_pae': ranking_confidence_pae[idx],
              'chain_pair_iptm': chain_pair_iptm[idx],
              'iptm_ichain': iptm_ichain[idx],
              'iptm_xchain': iptm_xchain[idx],
              'token_chain_ids': chain_ids,
              'token_res_ids': res_ids,
          },
          model_id=result['__identifier__'],
          debug_outputs={},
      )
