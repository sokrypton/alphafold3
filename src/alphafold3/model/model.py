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
import dataclasses
import functools
from typing import Any, TypeAlias

from absl import logging
from alphafold3 import structure
from alphafold3.common import base_config
from alphafold3.model import confidences
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import mapping
from alphafold3.model.components import utils
from .network import atom_cross_attention
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
  model_output_coords = result['diffusion_samples']['atom_positions']

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

  predicted_lddt = result.get('predicted_lddt')

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
  """Create target feature embedding."""

  dtype = jnp.bfloat16 if global_config.bfloat16 == 'all' else jnp.float32

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
      profile = batch.msa.profile.astype(dtype)
      deletion = batch.msa.deletion_mean[..., None].astype(dtype)
      s_inputs = enc.token_act.astype(dtype)
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_res_type_encoding')(res_oh)
      s_inputs += hm.Linear(config.seq_channel, use_bias=False,
                            name='boltz2_msa_profile_encoding')(
          jnp.concatenate([profile, deletion], axis=-1))
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
              jnp.concatenate([batch.msa.profile,
                               batch.msa.deletion_mean[..., None]],
                              axis=-1).astype(dtype))
      esm = batch.token_features.esm_embeddings
      if esm is not None:
        token_feats += hm.Linear(
            config.seq_channel, use_bias=False,
            name='chai1_esm_embedding')(esm.astype(dtype))
      s_cat = jnp.concatenate([enc.token_act.astype(dtype), token_feats], axis=-1)
      return hm.Linear(config.seq_channel, use_bias=False,
                       name='chai1_single_proj_in_trunk')(s_cat).astype(dtype)

    target_feat = jnp.concatenate([target_feat, enc.token_act], axis=-1).astype(
        dtype
    )

  return target_feat


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



def _structural_to_residue(samples, conf, book, seq_mask, asym_id,
                           num_plddt_bins, num_pae_bins, num_pde_bins):
  """Bring an OpenDDE fold back from structural tokens to residue tokens.

  OpenDDE runs its diffusion and confidence on an EXPANDED token set -- several
  structural subtokens per residue -- so everything downstream (the mmCIF
  writer, the confidence JSON, pTM) would otherwise be handed arrays indexed the
  wrong way. Two mappings do it, and they are of different kinds:

    atoms   an exact reconstruction. residue_atom_gather names, for every
            residue atom slot, the structural (token, slot) holding it, so
            coordinates and per-atom pLDDT come back unchanged.
    tokens  a CHOICE. A residue has several structural subtokens and PAE is a
            token-PAIR quantity, so we read it at the residue's representative
            subtoken (its first). Nothing reconstructs a per-residue PAE from
            several subtoken rows; this at least picks the row belonging to the
            residue's own backbone.

  The PAE/PDE/pLDDT reductions run here rather than inside OpenDDE's head, so
  that they -- and the pTM built on them -- see the residue token count the rest
  of the pipeline reports rather than the expanded one.
  """
  gather = book['residue_atom_gather']            # (n_res, max_atoms), -1 = none
  rep = book['residue_rep_token']                 # (n_res,)
  n_res, max_atoms = gather.shape
  valid = gather >= 0
  flat_idx = jnp.where(valid, gather, 0).reshape(-1)

  def by_atom(arr):
    """(n_samples, n_struct * max_atoms, *rest) -> (n_samples, n_res, max_atoms, *rest)."""
    rest = arr.shape[2:]
    out = jnp.take(arr, flat_idx, axis=1)
    out = out.reshape((arr.shape[0], n_res, max_atoms) + rest)
    keep = valid.reshape((1, n_res, max_atoms) + (1,) * len(rest))
    return jnp.where(keep, out, 0.0)

  def by_token_pair(arr):
    """(n_samples, n_struct, n_struct, *rest) -> (n_samples, n_res, n_res, *rest)."""
    return jnp.take(jnp.take(arr, rep, axis=1), rep, axis=2)

  positions = samples['atom_positions']
  out_samples = dict(samples)
  out_samples['atom_positions'] = by_atom(
      positions.reshape((positions.shape[0], -1, positions.shape[-1])))

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

  pae_logits = by_token_pair(conf['predicted_aligned_error_logits'])
  pde_logits = by_token_pair(conf['predicted_distance_error_logits'])
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

  return out_samples, {
      'predicted_lddt': predicted_lddt,
      'full_pae': full_pae,
      'full_pde': full_pde,
      'average_pde': average_pde,
      'predicted_experimentally_resolved': jax.nn.softmax(
          resolved_logits, axis=-1)[..., 1],
      'tmscore_adjusted_pae_global': tm_global,
      'tmscore_adjusted_pae_interface': tm_interface,
      'predicted_lddt_logits': plddt_logits,
      'predicted_aligned_error_logits': pae_logits,
      'predicted_distance_error_logits': pde_logits,
      'experimentally_resolved_logits': resolved_logits,
  }


def num_trunk_passes(num_recycles: int, model: str) -> int:
  """How many times the trunk runs, from a `num_recycles` setting.

  AF3 counts ADDITIONAL passes, so num_recycles=3 runs the trunk four times.
  chai counts TOTAL passes: `for _ in range(num_trunk_recycles)` in chai1.py,
  whose default 3 runs it three times. Getting this wrong is invisible -- the
  recycle is nearly converged by then, so an extra pass only perturbs the
  output -- which is why it is a named function with a test rather than a `+ 1`.
  """
  if model == 'chai1':
    # clamped at 1: chai's own `range(0)` really would run the trunk zero times,
    # leaving the recycle carry at its seed, and callers that pass 0 mean "one
    # trunk pass, no recycling" -- which is what an injection harness wants.
    return max(1, num_recycles)
  return num_recycles + 1


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

  def __init__(self, config: Config, name: str = 'diffuser'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config
    self.diffusion_module = diffusion_head.DiffusionHead(
        self.config.heads.diffusion, self.global_config
    )

  @hk.transparent
  def _sample_diffusion(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      *,
      sample_config: diffusion_head.SampleConfig,
  ) -> dict[str, jnp.ndarray]:
    # ONCE, not once per sampling step per sample. The pair conditioning reads
    # only the trunk embeddings and the batch -- the noise level enters the
    # single half -- so building it inside the denoiser meant a LayerNorm, a
    # projection and two transition blocks over (num_tokens, num_tokens,
    # pair_channel), plus an (L, L, 139) relative encoding, were rebuilt for
    # every step of every sample: ~1000 times in a default fold.
    pair_cond = self.diffusion_module(
        positions_noisy=None,        # unused on this path
        noise_level=jnp.zeros(()),   # unused by the pair half
        batch=batch,
        embeddings=embeddings,
        use_conditioning=True,
        conditioning_only=True,
    )

    denoising_step = functools.partial(
        self.diffusion_module,
        batch=batch,
        embeddings=embeddings,
        use_conditioning=True,
        pair_cond=pair_cond,
    )

    sample = diffusion_head.sample(
        denoising_step=denoising_step,
        batch=batch,
        key=hk.next_rng_key(),
        config=sample_config,
        global_config=self.global_config,
    )
    return sample

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

    batch = feat_batch.Batch.from_data_dict(batch)

    embedding_module = evoformer_network.Evoformer(
        self.config.evoformer, self.global_config
    )
    # ColabDesign2: the continuous relaxation of sequence, or None for
    # prediction. See featurization.blend_soft.
    embedding_module.soft_seq = soft_seq
    embedding_module.design_mask = design_mask
    target_feat = create_target_feat_embedding(
        batch=batch,  # pyrefly: ignore[bad-argument-type]
        config=embedding_module.config,
        global_config=self.global_config,
        soft_seq=soft_seq,
        design_mask=design_mask,
    )

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
    if hk.running_init():
      embeddings, _ = recycle_body(None, (embeddings, key))
    else:
      num_iter = num_trunk_passes(self.config.num_recycles,
                                  self.global_config.model)
      # ColabDesign2: AF3's recycle loop has no stop_gradient on `prev`, so with
      # num_recycles>0 the gradient backprops through EVERY pass -- the exact
      # mistake that cost the AF2 design path 3x its interface quality. When
      # recycle_last_only is set (AF2's recycle_mode='last'), run all but the
      # final pass and DETACH their output, so only the last pass is
      # differentiated: the trunk still recycles, but the sequence gradient is
      # the clean single-pass one. Off by default -> exact original behaviour.
      if getattr(self.config, 'recycle_last_only', False) and num_iter > 1:
        embeddings, key = hk.fori_loop(0, num_iter - 1, recycle_body,
                                       (embeddings, key))
        embeddings = jax.tree_util.tree_map(jax.lax.stop_gradient, embeddings)
        embeddings, _ = recycle_body(None, (embeddings, key))
      else:
        embeddings, _ = hk.fori_loop(0, num_iter, recycle_body,
                                     (embeddings, key))

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

    samples = self._sample_diffusion(
        diff_batch,
        diff_emb,
        sample_config=self.config.heads.diffusion.eval,
    )

    if has_structural:
      # OpenDDE's own confidence head, on the structural-token set. Rep-atom coords
      # per token come from the pseudo-beta gather; the pLDDT/resolved einsum runs
      # over the dense (token, atom-slot) layout, so atom_to_token repeats the token
      # index and atom_to_tokatom tiles the slot index.
      n_struct = diff_batch.token_features.mask.shape[0]
      max_atoms = samples['atom_positions'].shape[-2]
      a2t = jnp.repeat(jnp.arange(n_struct), max_atoms)
      a2ta = jnp.tile(jnp.arange(max_atoms), n_struct)
      c_s_inputs = diff_emb['target_feat'].shape[-1]
      def _conf(dense_atom_positions):
        rep = atom_layout.convert(
            diff_batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
            dense_atom_positions, layout_axes=(-3, -2))         # (n_struct, 3)
        return opendde_confidence.OpenDDEConfidenceHead(
            self.config.evoformer.seq_channel, self.config.evoformer.pair_channel,
            c_s_inputs, self.global_config)(
                diff_emb['target_feat'], diff_emb['single'], diff_emb['pair'], rep,
                a2t, a2ta, diff_batch.token_features.mask,
                extra_pair_bias=diff_emb.get('structural_pair_attn_bias'))
      confidence_output = mapping.sharded_map(_conf, in_axes=0)(samples['atom_positions'])
      # Back to residue tokens, so every consumer below -- the mmCIF writer, the
      # confidence JSON, pTM -- sees the same layout it does for every other
      # model. Without this the structure writer is handed 160 structural tokens
      # where its gather expects 128 residues, which is where it stops.
      book = {k[len('structbook/'):]: v for k, v in _struct.items()
              if k.startswith('structbook/')}
      samples, confidence_output = _structural_to_residue(
          samples, confidence_output, book,
          seq_mask=batch.token_features.mask,
          asym_id=batch.token_features.asym_id,
          num_plddt_bins=confidence_output['predicted_lddt_logits'].shape[-1],
          num_pae_bins=confidence_output['predicted_aligned_error_logits'].shape[-1],
          num_pde_bins=confidence_output['predicted_distance_error_logits'].shape[-1])
    elif self.global_config.model in model_config.NO_CONFIDENCE_HEAD:
      # No head in the checkpoint, so none in the graph: this model predicts a
      # structure and nothing about it. The keys are simply absent, which is
      # honest and is what keeps the converter's coverage exact.
      confidence_output = {}
    else:
      # Compute dist_error_fn over all samples for distance error logging.
      confidence_output = mapping.sharded_map(
          lambda dense_atom_positions: confidence_head.ConfidenceHead(
              self.config.heads.confidence, self.global_config
          )(
              dense_atom_positions=dense_atom_positions,
              embeddings=diff_emb,
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
  ) -> Iterable[InferenceResult]:
    """Get the predicted structure, scalars, and arrays for inference.

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

    predicted_distance_errors = result['average_pde']

    # Computing solvent accessible area with dssp can be slow for large
    # structures with lots of chains, so we parallelize the call.
    pred_structures = pred_structure.unstack()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(pred_structures), 32)
    ) as executor:
      has_clash = list(executor.map(confidences.has_clash, pred_structures))
      fraction_disordered = list(
          executor.map(confidences.fraction_disordered, pred_structures)
      )

    for idx, pred_structure in enumerate(pred_structures):
      ranking_score = confidences.get_ranking_score(
          ptm=ptm[idx],
          iptm=iptm[idx],
          fraction_disordered_=fraction_disordered[idx],
          has_clash_=has_clash[idx],
      )
      yield InferenceResult(
          predicted_structure=pred_structure,
          numerical_data={
              'full_pde': result['full_pde'][idx, :num_tokens, :num_tokens],
              'full_pae': result['full_pae'][idx, :num_tokens, :num_tokens],
              'contact_probs': contact_probs[:num_tokens, :num_tokens],
          },
          metadata={  # pyrefly: ignore[bad-argument-type]
              'predicted_distance_error': predicted_distance_errors[idx],
              'ranking_score': ranking_score,
              'fraction_disordered': fraction_disordered[idx],
              'has_clash': has_clash[idx],
              'predicted_tm_score': ptm[idx],
              'interface_predicted_tm_score': iptm[idx],
              'chain_pair_pde_mean': chain_pair_pde_mean[idx],
              'chain_pair_pde_min': chain_pair_pde_min[idx],
              'chain_pair_pae_min': chain_pair_pae_min[idx],
              'ptm': ptm[idx],
              'iptm': iptm[idx],
              'ptm_iptm_average': ptm_iptm_average[idx],
              'intra_chain_single_pde': intra_chain_single_pde[idx],
              'cross_chain_single_pde': cross_chain_single_pde[idx],
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
