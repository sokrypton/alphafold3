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

"""Confidence Head."""

from alphafold3.common import base_config
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.constants import atom_types
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from . import modules
from . import template_modules
import haiku as hk
import jax
import jax.numpy as jnp


def _safe_norm(x, keepdims, axis, eps=1e-8):
  return jnp.sqrt(eps + jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims))


class ConfidenceHead(hk.Module):
  """Head to predict the distance errors in a prediction."""

  class PAEConfig(base_config.BaseConfig):
    max_error_bin: float = 31.0
    num_bins: int = 64

  class Config(base_config.BaseConfig):
    """Configuration for ConfidenceHead."""

    pairformer: modules.PairFormerIteration.Config = base_config.autocreate(
        single_attention=base_config.autocreate(),
        single_transition=base_config.autocreate(),
        num_layer=4,
    )
    max_error_bin: float = 31.0
    num_plddt_bins: int = 50
    num_bins: int = 64
    no_embedding_prob: float = 0.2
    pae: 'ConfidenceHead.PAEConfig' = base_config.autocreate()
    dgram_features: template_modules.DistogramFeaturesConfig = (
        base_config.autocreate()
    )

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='confidence_head',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  @hk.transparent
  def _head_norm(self, name, x):
    """LayerNorm before a logit head -- except for boltz2, which has none.

    Boltz-2's ConfidenceHeads call to_pae/pde/plddt/resolved_logits directly on z/s.
    This cannot be expressed as a mapped weight: a LayerNorm with scale=1/offset=0
    still normalises, so it has to be skipped in the graph, which is also why those
    6 params have no source in the checkpoint.

    @hk.transparent is REQUIRED: haiku scopes parameters by the method that creates
    them, so without it these LayerNorms move from `<head>/logits_ln` to
    `<head>/~_head_norm/logits_ln` and every already-converted model (of3/if2/rf3/
    protenix) silently loses 8 parameters to init. Caught by the structural gate.
    """
    skip = model_config.NO_HEAD_NORM.get(self.global_config.model, ())
    if '*' in skip or name in skip:
      return x
    return hm.LayerNorm(name=name)(x)

  @hk.transparent
  def _boltz2_rel_pos(self, tf, num_channels, dtype):
    """Boltz-2's RelativePositionEncoder, as instantiated in the confidence module.

    Verified bit-exact (max|d| 0.0) against the module's own output on all four oracle
    cases -- monomer, contact-constrained, homodimer and covalently bonded.

    Two flags of the checkpoint's instantiation are NOT the signature defaults, and
    reading them off the source signature rather than the loaded model is what made an
    earlier attempt at this term make the residual worse:
      * fix_sym_check=True  -- the chain-offset sentinel is applied where the two tokens
        are of DIFFERENT entities, not where they share a chain. Copying the `False`
        branch inverts the mask on every cross-chain pair.
      * cyclic_pos_enc=True -- but it is a no-op unless a token carries a nonzero
        cyclic_period, which our featuriser has no field for yet, so the residue offset
        is left unwrapped here. See memory `cyclic-period-feature`: exposing it is a
        cross-model feature (cyclic peptides, circular permutants), not a boltz2 patch.
    """
    r_max, s_max = 32, 2
    same_chain = tf.asym_id[:, None] == tf.asym_id[None]
    same_res = tf.residue_index[:, None] == tf.residue_index[None]
    same_entity = tf.entity_id[:, None] == tf.entity_id[None]

    d_res = jnp.clip(tf.residue_index[:, None] - tf.residue_index[None] + r_max,
                     0, 2 * r_max)
    d_res = jnp.where(same_chain, d_res, 2 * r_max + 1)
    d_tok = jnp.clip(tf.token_index[:, None] - tf.token_index[None] + r_max,
                     0, 2 * r_max)
    d_tok = jnp.where(same_chain & same_res, d_tok, 2 * r_max + 1)
    d_chain = jnp.clip(tf.sym_id[:, None] - tf.sym_id[None] + s_max, 0, 2 * s_max)
    d_chain = jnp.where(~same_entity, 2 * s_max + 1, d_chain)

    feat = jnp.concatenate([
        jax.nn.one_hot(d_res, 2 * r_max + 2),
        jax.nn.one_hot(d_tok, 2 * r_max + 2),
        same_entity[..., None].astype(jnp.float32),
        jax.nn.one_hot(d_chain, 2 * s_max + 2),
    ], axis=-1)
    return hm.Linear(num_channels, name='rel_pos_project')(feat.astype(dtype))

  @hk.transparent
  def _boltz2_contact_conditioning(self, contact, threshold, num_channels, dtype):
    """Boltz-2's ContactConditioning: user distance restraints, embedded into z.

    `contact` is a one-hot over (UNSPECIFIED, UNSELECTED, <3 restraint classes>) and
    `threshold` the restraint distance in Angstrom. The first two classes bypass the
    encoder entirely and contribute a learned constant, which is why an unconstrained
    input still gets a nonzero term: every pair is UNSPECIFIED, so every pair receives
    `encoding_unspecified`.

    The encoder runs even when nothing is selected (it is multiplied by zero). That is
    deliberate: it keeps the restraint weights in the parameter tree and converted, so
    wiring real restraints later is a featurisation change rather than a graph change.
    """
    cutoff_min, cutoff_max = 4.0, 20.0
    tn = (threshold - cutoff_min) / (cutoff_max - cutoff_min)
    # FourierEmbedding: cos(2*pi*Linear(1->c)(t)), with a FROZEN random projection --
    # a trained parameter in the checkpoint only in the sense that it was stored.
    fourier = jnp.cos(2 * jnp.pi * hm.Linear(
        num_channels, use_bias=True, name='contact_fourier')(
            tn[..., None].astype(dtype)))
    x = jnp.concatenate(
        [contact[..., 2:].astype(dtype), tn[..., None].astype(dtype), fourier], -1)
    enc = hm.Linear(num_channels, use_bias=True, name='contact_encoder')(x)
    unspecified = hk.get_parameter('contact_encoding_unspecified', [num_channels],
                                   dtype, init=jnp.zeros)
    unselected = hk.get_parameter('contact_encoding_unselected', [num_channels],
                                  dtype, init=jnp.zeros)
    selected = contact[..., 0:2].sum(-1, keepdims=True).astype(dtype)
    return (enc * (1 - selected)
            + unspecified * contact[..., 0:1].astype(dtype)
            + unselected * contact[..., 1:2].astype(dtype))

  @hk.transparent
  def _boltz2_split_heads(self, z, asym_id, num_bins, intra_name, inter_name):
    """boltz2's intra/inter-chain logit pair: `intra(z)*same + inter(z)*different`.

    Not a blend -- the two are disjoint hard masks, so on a monomer this reduces
    exactly to the intra head and the inter weights are dead. That is the whole reason
    the monomer gate could not see the missing head.
    """
    same_chain = (asym_id[:, None] == asym_id[None])[..., None].astype(z.dtype)
    intra = hm.Linear(num_bins, initializer=self.global_config.final_init,
                      name=intra_name)(z)
    inter = hm.Linear(num_bins, initializer=self.global_config.final_init,
                      name=inter_name)(z)
    return intra * same_chain + inter * (1 - same_chain)

  def _boltz2_reembed(self, pair_act, single_act, target_feat, positions, pair_mask,
                      token_features, bond_matrix, bond_type_matrix):
    """Boltz-2 rebuilds s and z from scratch before its confidence pairformer.

    AF3 adds a couple of projections to the trunk's z; boltz LayerNorms the trunk
    outputs and then adds NINE terms. The full sum was verified exactly (mean|d| 3e-8)
    against the module's own hooked activations, which is how the term that had been
    missing all along -- s_to_z_prod, `add_s_to_z_prod=True` in this checkpoint -- was
    finally found: it is uncorrelated with every other term, so fitting it out of an
    aggregate residual had repeatedly mis-attributed it to whichever term was being
    tested at the time.

    contact_conditioning runs on a placeholder rather than a real feature, because our
    featuriser has no distance-restraint field. That is exactly what boltz produces for
    an unconstrained input, so it is a coverage limit, not a silent divergence.
    """
    c = pair_act.shape[-1]
    dtype = pair_act.dtype
    n = pair_act.shape[0]

    s_inputs = hm.LayerNorm(name='s_inputs_norm')(target_feat)
    single_act = hm.LayerNorm(name='s_norm')(single_act)
    single_act = single_act + hm.Linear(
        single_act.shape[-1], name='s_input_to_s')(s_inputs)

    z = hm.LayerNorm(name='z_norm')(pair_act)
    z += self._boltz2_rel_pos(token_features, c, dtype)
    z += hm.Linear(c, name='token_bonds_project')(
        bond_matrix[..., None].astype(dtype))
    z += hm.Linear(c, name='token_bonds_type_embed')(
        jax.nn.one_hot(bond_type_matrix, 7).astype(dtype))
    z += self._boltz2_contact_conditioning(
        jax.nn.one_hot(jnp.zeros((n, n), jnp.int32), 5),
        jnp.zeros((n, n), jnp.float32), c, dtype)

    # Orientation is load-bearing and the opposite of what the names suggest: boltz adds
    # s_to_z(s)[:, :, None] -- broadcast along j, so indexed by i -- which is our
    # RIGHT projection, and s_to_z_transpose is the LEFT one.
    z += hm.Linear(c, name='right_target_feat_project')(s_inputs)[:, None]
    z += hm.Linear(c, name='left_target_feat_project')(s_inputs)[None]
    prod = (hm.Linear(c, name='s_to_z_prod_in1')(s_inputs)[:, None]
            * hm.Linear(c, name='s_to_z_prod_in2')(s_inputs)[None])
    z += hm.Linear(c, name='s_to_z_prod_out')(prod)

    # Boltz embeds representative-atom distance into 64 bins (63 edges, 2..22 A) with an
    # nn.Embedding, not AF3's 39-bin projection.
    d = jnp.sqrt(jnp.sum((positions[:, None] - positions[None]) ** 2, -1) + 1e-10)
    bnd = jnp.linspace(2.0, 22.0, 63)
    dgram = jax.nn.one_hot((d[..., None] > bnd).sum(-1), 64).astype(dtype)
    z += hm.Linear(c, name='distogram_feat_project')(dgram * pair_mask[..., None])
    return z, single_act

  def _embed_features(
      self,
      dense_atom_positions,
      token_atoms_to_pseudo_beta,
      pair_mask,
      pair_act,
      target_feat,
  ):
    out = hm.Linear(pair_act.shape[-1], name='left_target_feat_project')(
        target_feat
    ).astype(pair_act.dtype)
    out += hm.Linear(pair_act.shape[-1], name='right_target_feat_project')(
        target_feat
    ).astype(pair_act.dtype)[:, None]
    positions = atom_layout.convert(
        token_atoms_to_pseudo_beta,
        dense_atom_positions,
        layout_axes=(-3, -2),
    )
    dgram = template_modules.dgram_from_positions(
        positions, self.config.dgram_features
    )
    dgram *= pair_mask[..., None]

    if self.global_config.model == 'rosettafold3':
      # RF3 confidence embeds the predicted structure via CA-CA (representative-atom)
      # distances discretized to 40 bins (bucketize over 39 boundaries 3.25..50.75),
      # NOT AF3's CB-CB dgram. positions here is the CA (pseudo_beta gather returns CB;
      # RF3 uses the token-center CA = dense atom index 1).
      ca = dense_atom_positions[:, 1, :]
      d = jnp.sqrt(jnp.sum((ca[:, None] - ca[None]) ** 2, -1) + 1e-10)
      bnd = jnp.arange(39) * ((50.75 - 3.25) / 39.0) + 3.25
      dgram = jax.nn.one_hot((d[..., None] > bnd).sum(-1), 40).astype(pair_act.dtype)
      dgram = dgram * pair_mask[..., None]
    if self.global_config.model == 'chai1':
      # chai bins the REFERENCE-atom distances into 16 bins by searchsorted over
      # 15 learned-but-frozen boundaries (atom_distance_v_bins, an evenly spaced
      # 3.375..21.375), where AF3 uses 39 bins over a different range. Its
      # reference atom is the pseudo-beta we already gather: reading
      # token_reference_atom_index back through the 6MRR atom layout gives dense
      # slot 4 (CB) for every residue and slot 1 (CA) for the glycine.
      # chai does NOT mask this term, so neither do we.
      d = jnp.sqrt(
          jnp.sum((positions[:, None] - positions[None]) ** 2, -1) + 1e-10)
      bnd = jnp.linspace(3.375, 21.375, 15)
      dgram = jax.nn.one_hot((d[..., None] > bnd).sum(-1), 16).astype(
          pair_act.dtype)
    out += hm.Linear(pair_act.shape[-1], name='distogram_feat_project')(
        dgram.astype(pair_act.dtype)
    )
    if self.global_config.model in model_config.PROTENIX_FAMILY:
      # Protenix adds a SECOND distance term alongside the binned one: a linear on the
      # raw distance (`linear_no_bias_d_wo_onehot`, in_features=1). Unbinned, so it
      # carries the sub-bin resolution the one-hot throws away. Its binning is otherwise
      # ours exactly -- arange(3.25, 52.0, 1.25) is the same 39 bins as AF3's default.
      d = jnp.sqrt(jnp.sum((positions[:, None] - positions[None]) ** 2, -1) + 1e-10)
      out += hm.Linear(pair_act.shape[-1], name='distance_feat_project')(
          d[..., None].astype(pair_act.dtype))
    return out

  def __call__(
      self,
      dense_atom_positions: jnp.ndarray,
      embeddings: dict[str, jnp.ndarray],
      seq_mask: jnp.ndarray,
      token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
      asym_id: jnp.ndarray,
      token_features=None,
      bond_matrix: jnp.ndarray | None = None,
      bond_type_matrix: jnp.ndarray | None = None,
      atom_name_chars: jnp.ndarray | None = None,
  ) -> dict[str, jnp.ndarray]:
    """Builds ConfidenceHead module.

    Arguments:
      dense_atom_positions: [N_res, N_atom, 3] array of positions.
      embeddings: Dictionary of representations.
      seq_mask: Sequence mask.
      token_atoms_to_pseudo_beta: Pseudo beta info for atom tokens.
      asym_id: Asym ID token features.

    Returns:
      Dictionary of results.
    """
    dtype = (
        jnp.bfloat16 if self.global_config.bfloat16 == 'all' else jnp.float32
    )
    with utils.bfloat16_context():
      seq_mask_cast = seq_mask.astype(dtype)
      pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]
      pair_mask = pair_mask.astype(dtype)

      pair_act = embeddings['pair'].astype(dtype)
      single_act = embeddings['single'].astype(dtype)
      target_feat = embeddings['target_feat'].astype(dtype)

      if self.global_config.model in model_config.PAIR_ONLY_TRUNK:
        # No trunk single exists, so there is nothing to read: ESMFold2 builds
        # its single by ROW-ATTENTION POOLING the pair -- softmax over j of a
        # learned scalar per (i, j), then a projection. This is the one piece of
        # ESMFold2 with no AF3 analogue, and it is here rather than in the trunk
        # because the pair it pools is the confidence head's own re-embedded
        # pair, not the trunk's.
        scores = hm.Linear(1, name='row_pool_attn')(pair_act)[..., 0]
        scores = jnp.where(seq_mask_cast[None, :] > 0, scores, -1e9)
        single_act = hm.Linear(
            single_act.shape[-1], name='row_pool_out')(
                jnp.einsum('nm,nmd->nd', jax.nn.softmax(scores, axis=-1), pair_act))

      if self.global_config.model in model_config.PROTENIX_FAMILY:
        # Protenix LayerNorms (and clamps) the trunk single before ANY use -- the
        # confidence pairformer and every head see the normalised one
        # (`input_strunk_ln(clamp(s_trunk, -512, 512))`). AF3 uses it raw, and our
        # unnormalised single was entering the head at std 211, so this is the same
        # class of gate-invisible divergence RF3's global LayerNorm turned out to be.
        single_act = hm.LayerNorm(name='input_single_norm')(
            jnp.clip(single_act, -512.0, 512.0))

      if self.global_config.model == 'rosettafold3':
        # RF3 confidence applies a parameter-free LayerNorm over the WHOLE tensor
        # (layer_norm_along_feature_dimension=False) to each detached trunk input
        # before use; AF3 uses them raw. No trained params, so purely a forward branch.
        #
        # The statistics are over REAL tokens only. Everything else here is
        # per-element, so padding cannot reach it -- but a mean and a variance
        # taken over the whole tensor are shifted by however many padding tokens
        # the bucket happened to add, and the head's own inputs move with them.
        # Native RF3 never sees padding, so including it is simply wrong, and it
        # is not a small effect: 76 residues padded into a 128-token bucket gave
        # a PAE of ~28 A everywhere, diagonal included, and pTM 0.04 against 0.89
        # for the same fold run unpadded.
        seq_mask_bool = seq_mask.astype(bool)
        pair_act = masked_global_norm(
            pair_act, seq_mask_bool[:, None] & seq_mask_bool[None, :])
        single_act = masked_global_norm(single_act, seq_mask_bool)
        target_feat = masked_global_norm(target_feat, seq_mask_bool)

      num_residues = seq_mask.shape[0]
      num_pair_channels = pair_act.shape[2]

      if self.global_config.model in model_config.REEMBED_CONFIDENCE_PAIR:
        positions = atom_layout.convert(
            token_atoms_to_pseudo_beta, dense_atom_positions, layout_axes=(-3, -2))
        pair_act, single_act = self._boltz2_reembed(
            pair_act, single_act, target_feat, positions, pair_mask,
            token_features, bond_matrix, bond_type_matrix)
      else:
        pair_act += self._embed_features(
            dense_atom_positions,
            token_atoms_to_pseudo_beta,
            pair_mask,
            pair_act,
            target_feat,
        )

      def pairformer_fn(act):
        pair_act, single_act = act
        return modules.PairFormerIteration(
            self.config.pairformer,
            self.global_config,
            with_single=True,
            name='confidence_pairformer',
        )(
            act=pair_act,
            single_act=single_act,
            pair_mask=pair_mask,
            seq_mask=seq_mask,
        )

      pairformer_stack = hk.experimental.layer_stack(
          self.config.pairformer.num_layer
      )(pairformer_fn)

      pair_act, single_act = pairformer_stack((pair_act, single_act))
      pair_act = pair_act.astype(jnp.float32)
      assert pair_act.shape == (num_residues, num_residues, num_pair_channels)

      # Produce logits to predict a distogram of pairwise distance errors
      # between the input prediction and the ground truth.
      pred_distance_error = None
      average_pred_distance_error = None
      no_pde = self.global_config.model in model_config.NO_PDE_HEAD

      # Shape (num_res, num_res, num_bins)
      if no_pde:
        # ESMFold2's experimental line has no PDE head at all -- no pde_head and
        # no pde_ln. Same reasoning as the missing resolved head below: building
        # it would leave three parameters at random init and emit a `full_pde`
        # that reads like a prediction.
        pass
      elif self.global_config.model == 'boltz2':
        # boltz2 has use_separate_heads=True: SEPARATE intra- and inter-chain heads for
        # both PDE and PAE, each hard-masked to its half of the pair matrix. On a
        # monomer the inter head never fires, which is why the intra head alone was
        # exact on 6MRR and why this needs a complex to validate.
        distance_logits = self._boltz2_split_heads(
            pair_act + jnp.swapaxes(pair_act, -2, -3),   # boltz symmetrizes FIRST
            asym_id, self.config.num_bins,
            'left_half_distance_logits', 'inter_half_distance_logits')
      else:
        left_distance_logits = hm.Linear(
            self.config.num_bins,
            initializer=self.global_config.final_init,
            name='left_half_distance_logits',
        )(self._head_norm('logits_ln', pair_act))
        right_distance_logits = left_distance_logits
        distance_logits = left_distance_logits + jnp.swapaxes(  # Symmetrize.
            right_distance_logits, -2, -3
        )
      # Shape (num_bins,)
      distance_breaks = jnp.linspace(
          0.0, self.config.max_error_bin, self.config.num_bins - 1
      )

      step = distance_breaks[1] - distance_breaks[0]

      # Add half-step to get the center
      bin_centers = distance_breaks + step / 2
      # Add a catch-all bin at the end.
      bin_centers = jnp.concatenate(
          [bin_centers, bin_centers[-1:] + step], axis=0
      )

      if not no_pde:
        distance_probs = jax.nn.softmax(distance_logits, axis=-1)

        pred_distance_error = (
            jnp.sum(distance_probs * bin_centers, axis=-1) * pair_mask
        )
        average_pred_distance_error = jnp.sum(
            pred_distance_error, axis=[-2, -1]
        ) / jnp.sum(pair_mask, axis=[-2, -1])

      # Predicted aligned error
      pae_outputs = {}
      # Shape (num_res, num_res, num_bins)
      if self.global_config.model == 'boltz2':
        pae_logits = self._boltz2_split_heads(
            pair_act, asym_id, self.config.pae.num_bins,
            'pae_logits', 'pae_inter_logits')
      else:
        pae_logits = hm.Linear(
            self.config.pae.num_bins,
            initializer=self.global_config.final_init,
            name='pae_logits',
        )(self._head_norm('pae_logits_ln', pair_act))
      # Shape (num_bins,)
      pae_breaks = jnp.linspace(
          0.0, self.config.pae.max_error_bin, self.config.pae.num_bins - 1
      )
      step = pae_breaks[1] - pae_breaks[0]
      # Add half-step to get the center
      bin_centers = pae_breaks + step / 2
      # Add a catch-all bin at the end.
      bin_centers = jnp.concatenate(
          [bin_centers, bin_centers[-1:] + step], axis=0
      )
      pae_probs = jax.nn.softmax(pae_logits, axis=-1)

      seq_mask_bool = seq_mask.astype(bool)
      pair_mask_bool = seq_mask_bool[:, None] * seq_mask_bool[None, :]
      pae = jnp.sum(pae_probs * bin_centers, axis=-1) * pair_mask_bool
      pae_outputs.update({
          'full_pae': pae,
      })

    # The pTM is computed outside of bfloat16 context.
    tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = (
        self._get_tmscore_adjusted_pae(
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask_bool,
            bin_centers=bin_centers,
            pae_probs=pae_probs,
        )
    )
    pae_outputs.update({
        'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global,
        'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface,
    })
    single_act = single_act.astype('float32')

    # pLDDT
    # Shape (num_res, num_atom, num_bins)
    # Boltz predicts pLDDT PER TOKEN ((n,50)) where AF3 predicts per dense-atom
    # slot ((n,24,50)). Broadcasting one token-level logit across the slots is
    # the same function as a per-slot weight whose slots are all equal, so the
    # boltz2 converter TILES its weight and this needs no branch. Its prediction
    # is still token-level -- every atom of a token reports the same number.
    _n_atom = dense_atom_positions.shape[-2]
    if self.global_config.model == 'chai1' and atom_name_chars is not None:
      # chai predicts pLDDT over the 37 canonical ATOM37 slots and gathers per
      # atom by the atom's index in THAT table -- not by our dense slot. The two
      # orders differ (CB is ATOM37 slot 3 and dense slot 4; TRP's NE1 is 24),
      # and the permutation is residue-type dependent, so it cannot be folded
      # into the weights. Recover it by matching the atom NAME, which we already
      # carry per dense slot, against ATOM37.
      plddt_logits = hm.Linear(
          (len(atom_types.ATOM37), self.config.num_plddt_bins),
          initializer=self.global_config.final_init,
          name='plddt_logits',
      )(self._head_norm('plddt_logits_ln', single_act))
      table = jnp.asarray(
          [[ord(c) - 32 for c in n.ljust(4)] for n in atom_types.ATOM37],
          jnp.int32)                                        # (37, 4)
      hit = jnp.all(atom_name_chars[:, :, None, :] == table[None, None], axis=-1)
      idx = jnp.argmax(hit, axis=-1)                        # (n_token, n_atom)
      plddt_logits = jnp.take_along_axis(
          plddt_logits, idx[..., None], axis=1)
    else:
      plddt_logits = hm.Linear(
          (_n_atom, self.config.num_plddt_bins),
          initializer=self.global_config.final_init,
          name='plddt_logits',
      )(self._head_norm('plddt_logits_ln', single_act))

    bin_width = 1.0 / self.config.num_plddt_bins
    bin_centers = jnp.arange(0.5 * bin_width, 1.0, bin_width)
    predicted_lddt = jnp.sum(
        jax.nn.softmax(plddt_logits, axis=-1) * bin_centers, axis=-1
    )
    predicted_lddt = predicted_lddt * 100.0

    # Experimentally resolved
    # Shape (num_res, num_atom, 2)
    #
    # chai has NO such head, so building it left three parameters at random
    # init and produced a `predicted_experimentally_resolved` that looked like a
    # prediction and was noise. Don't create it; the key is simply absent for
    # chai1, which is honest and also what makes the converter's coverage exact.
    experimentally_resolved_logits = None
    if self.global_config.model in model_config.NO_RESOLVED_HEAD:
      pass          # no such head here; nothing to build
    else:
      experimentally_resolved_logits = hm.Linear(
          (_n_atom, 2),
          initializer=self.global_config.final_init,
          name='experimentally_resolved_logits',
      )(self._head_norm('experimentally_resolved_ln', single_act))

    # gate on the LOGITS, which every branch either assigns or leaves None --
    # gating on the derived value read an unassigned local for every non-chai
    # model, which the fast suite could not see because the tests that exercise
    # those models are all slow-marked
    predicted_experimentally_resolved = None
    if experimentally_resolved_logits is not None:
      predicted_experimentally_resolved = jax.nn.softmax(
          experimentally_resolved_logits, axis=-1
      )[..., 1]

    out = {
        'predicted_lddt': predicted_lddt,
        **({} if pred_distance_error is None else {
            'full_pde': pred_distance_error,
            'average_pde': average_pred_distance_error,
        }),
        **pae_outputs,
    }
    if predicted_experimentally_resolved is not None:
      out['predicted_experimentally_resolved'] = (
          predicted_experimentally_resolved)
    return out

  def _get_tmscore_adjusted_pae(
      self,
      asym_id: jnp.ndarray,
      seq_mask: jnp.ndarray,
      pair_mask: jnp.ndarray,
      bin_centers: jnp.ndarray,
      pae_probs: jnp.ndarray,
  ):
    return tmscore_adjusted_pae(
        asym_id=asym_id, seq_mask=seq_mask, pair_mask=pair_mask,
        bin_centers=bin_centers, pae_probs=pae_probs)


def tmscore_adjusted_pae(
    *,
    asym_id: jnp.ndarray,
    seq_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
    pae_probs: jnp.ndarray,
):
  """PAE reweighted by the TM-score term, globally and per interface.

  Module level because it is the head's arithmetic and not its parameters:
  OpenDDE has its own confidence head whose outputs still have to reach the same
  pTM/ipTM the rest of the pipeline reports.
  """

  def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
    # Clip to avoid negative/undefined d0.
    clipped_num_res = jnp.maximum(num_interface_tokens, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
    # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    # Yang & Skolnick "Scoring function for automated
    # assessment of protein structure template quality" 2004.
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Make compatible with [num_tokens, num_tokens, num_bins]
    d0 = d0[:, :, None]
    bin_centers = bin_centers[None, None, :]

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(pae_probs * tm_per_bin, axis=-1)
    return predicted_tm_term

  # Interface version
  x = asym_id[None, :] == asym_id[:, None]
  num_chain_tokens = jnp.sum(x * pair_mask, axis=-1)
  num_interface_tokens = num_chain_tokens[None, :] + num_chain_tokens[:, None]
  # Don't double-count within a single chain
  num_interface_tokens -= x * (num_interface_tokens // 2)
  num_interface_tokens = num_interface_tokens * pair_mask

  num_global_tokens = jnp.full(
      shape=pair_mask.shape, fill_value=seq_mask.sum()
  )

  assert num_global_tokens.dtype == 'int32'
  assert num_interface_tokens.dtype == 'int32'
  global_apae = get_tmscore_adjusted_pae(
      num_global_tokens, bin_centers, pae_probs
  )
  interface_apae = get_tmscore_adjusted_pae(
      num_interface_tokens, bin_centers, pae_probs
  )
  return global_apae, interface_apae


def masked_global_norm(x, mask):
  """Normalise x by the mean and variance over the REAL tokens only.

  RoseTTAFold3's confidence head layer-norms each detached trunk input over the
  whole tensor rather than along the feature axis
  (`layer_norm_along_feature_dimension=False`). A statistic that reduces over
  more than the feature axis is padding-sensitive in a way a per-feature one is
  not: native RF3 never pads, so including our padding tokens shifts every
  confidence output by however many the bucket happened to add.

  mask broadcasts over x's leading axes; the feature axis is always last.
  """
  xf = x.astype(jnp.float32)
  m = mask.astype(jnp.float32)[..., None]
  n = jnp.maximum(jnp.sum(m) * xf.shape[-1], 1.0)
  mean = jnp.sum(xf * m) / n
  var = jnp.sum(jnp.square(xf - mean) * m) / n
  return ((xf - mean) / jnp.sqrt(var + 1e-5)).astype(x.dtype)
