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

"""Evoformer network."""

import functools

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from . import atom_cross_attention
from . import featurization
from . import modules
from . import template_modules
import os

import haiku as hk
import jax
import jax.numpy as jnp


# per-block pair reps when CHAI_TAP_MSA_BLOCKS=1; diagnostic, default empty
MSA_BLOCK_TAPS = []


def _stack(num_layer, fn, remat):
  """ColabDesign2: a layer_stack, optionally gradient-checkpointed.

  block_remat is declared in this file's config (and in
  diffusion_transformer.py) and consumed nowhere -- a dead knob that looks like
  it turns on checkpointing and does not. So AF3's trunk stores every block's
  activations for the backward pass, and at L=92 that is 16.3 GiB, which does
  not fit on a 23 GiB card. AF2 has had `use_remat` for exactly this since 2021.

  Recomputing each block in the backward pass trades time for memory. It is the
  difference between AF3 design being capped around L=70 and reaching the sizes
  AF2 handles.
  """
  if remat:
    fn = hk.remat(fn)
  return hk.experimental.layer_stack(num_layer)(fn)


def token_bond_matrix(batch, symmetrize: bool = False) -> jnp.ndarray:
  """(num_tokens, num_tokens) 0/1 matrix of inter-token bonds.

  Factored out of Evoformer._embed_bonds so the confidence head can build the same
  matrix: boltz-2 re-embeds token bonds inside its confidence module, and rederiving
  the gather there would be a second place for the padding convention below to drift.
  """
  num_tokens = batch.token_features.token_index.shape[0]
  contact_matrix = jnp.zeros((num_tokens, num_tokens))

  tokens_to_polymer_ligand_bonds = (
      batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
  )
  gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
  gather_mask_polymer_ligand = (
      tokens_to_polymer_ligand_bonds.gather_mask.prod(axis=1).astype(
          gather_idxs_polymer_ligand.dtype
      )[:, None]
  )
  # If valid mask then it will be all 1's, so idxs should be unchanged.
  gather_idxs_polymer_ligand = (
      gather_idxs_polymer_ligand * gather_mask_polymer_ligand
  )

  tokens_to_ligand_ligand_bonds = (
      batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
  )
  gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
  gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
      axis=1
  ).astype(gather_idxs_ligand_ligand.dtype)[:, None]
  gather_idxs_ligand_ligand = (
      gather_idxs_ligand_ligand * gather_mask_ligand_ligand
  )

  gather_idxs = jnp.concatenate(
      [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
  )
  contact_matrix = contact_matrix.at[
      gather_idxs[:, 0], gather_idxs[:, 1]
  ].set(1.0)

  if symmetrize:
    # OF3 weights (and boltz-2) were trained with a symmetric bond matrix (both
    # i->j and j->i set for each bond). AF3's featurization only provides one
    # direction from the CCD bond table, so we symmetrize here.
    contact_matrix = contact_matrix.at[
        gather_idxs[:, 1], gather_idxs[:, 0]
    ].set(1.0)

  # Because all the padded index's are 0's.
  return contact_matrix.at[0, 0].set(0.0)


def token_bond_type_matrix(batch, symmetrize: bool = False) -> jnp.ndarray:
  """(num_tokens, num_tokens) int matrix of boltz-2 `type_bonds` codes.

  Same scatter as `token_bond_matrix`, but writing a bond ORDER instead of 1.  The
  values are boltz's own convention -- 0 means "no bond", so every real bond is its
  `features.BOND_ORDER_*` code plus one.  Polymer-ligand links are COVALENT by
  construction (`get_bond_layout` only admits mmCIF `covale`), so they need no
  per-bond feature; only the ligand-ligand rows, which mix a residue's own CCD bond
  graph with inter-residue links, carry one.
  """
  num_tokens = batch.token_features.token_index.shape[0]
  type_matrix = jnp.zeros((num_tokens, num_tokens), jnp.int32)

  polymer_ligand = batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
  ligand_ligand = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds

  parts = []
  for gather, orders in (
      (
          polymer_ligand,
          jnp.full(
              polymer_ligand.gather_idxs.shape[:1],
              features.BOND_ORDER_COVALENT,
              jnp.int32,
          ),
      ),
      (
          ligand_ligand,
          jnp.asarray(batch.ligand_ligand_bond_info.bond_order, jnp.int32),
      ),
  ):
    mask = gather.gather_mask.prod(axis=1).astype(gather.gather_idxs.dtype)
    parts.append((gather.gather_idxs * mask[:, None], (orders + 1) * mask))

  gather_idxs = jnp.concatenate([idxs for idxs, _ in parts])
  values = jnp.concatenate([vals for _, vals in parts])

  type_matrix = type_matrix.at[gather_idxs[:, 0], gather_idxs[:, 1]].set(values)
  if symmetrize:
    type_matrix = type_matrix.at[gather_idxs[:, 1], gather_idxs[:, 0]].set(
        values
    )
  # Because all the padded index's are 0's.
  return type_matrix.at[0, 0].set(0)


def _boltz2_contact_conditioning(contact, threshold, num_channels, dtype):
  """Boltz-2's ContactConditioning for the TRUNK z-init.

  Deliberately a separate copy of ConfidenceHead._boltz2_contact_conditioning rather
  than a shared call: these are different weights (model.contact_conditioning vs
  confidence_module.contact_conditioning), and the confidence head's version sits
  behind a passing injection gate that a refactor would move param scopes under.

  `contact` one-hots (UNSPECIFIED, UNSELECTED, <3 restraint classes>); the first two
  bypass the encoder and contribute a learned constant, which is why an unconstrained
  input still gets a nonzero term. The encoder runs multiplied by zero so its weights
  stay in the tree and converted.
  """
  cutoff_min, cutoff_max = 4.0, 20.0
  tn = (threshold - cutoff_min) / (cutoff_max - cutoff_min)
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


class Evoformer(hk.Module):
  """Creates 'single' and 'pair' embeddings."""

  class PairformerConfig(modules.PairFormerIteration.Config):  # pytype: disable=invalid-function-definition
    block_remat: bool = False
    remat_block_size: int = 8

  class Config(base_config.BaseConfig):
    """Configuration for Evoformer."""

    max_relative_chain: int = 2
    msa_channel: int = 64
    seq_channel: int = 384
    max_relative_idx: int = 32
    num_msa: int = 1024
    pair_channel: int = 128
    pairformer: 'Evoformer.PairformerConfig' = base_config.autocreate(
        single_transition=base_config.autocreate(),
        single_attention=base_config.autocreate(),
        num_layer=48,
    )
    per_atom_conditioning: atom_cross_attention.AtomCrossAttEncoderConfig = (
        base_config.autocreate(
            per_token_channels=384,
            per_atom_channels=128,
            atom_transformer=base_config.autocreate(
                num_intermediate_factor=2,
                num_blocks=3,
            ),
            per_atom_pair_channels=16,
        )
    )
    template: template_modules.TemplateEmbedding.Config = (
        base_config.autocreate()
    )
    msa_stack: modules.EvoformerIteration.Config = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='evoformer',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def _relative_encoding(
      self, batch: feat_batch.Batch, pair_activations: jnp.ndarray
  ) -> jnp.ndarray:
    """Add relative position encodings."""
    if self.global_config.model == 'chai1':
      # chai has no separate relative encoding: its two relative-separation
      # features are just members of the TOKEN_PAIR input stream. For a de novo
      # single-chain protein the rest of that stream is constant (Docking=5,
      # RelativeChain=2, RelativeEntity=1, and both restraints at their -1
      # sentinel), so it collapses the same way the TOKEN stream did -- two
      # Linears over one-hots plus a constant, folded into this Linear's bias.
      #
      # Both formulas are chai's own, and both differ from the obvious guess.
      # RelativeSequenceSeparation is a searchsorted over bins [-32..32] of
      # (rel + 1e-4), which collapses to clip(rel + 33, 0, 65) -- so +32 and
      # >+32 SHARE the top index -- with 66 for inter-chain. RelativeTokenSeparation
      # is over the TOKEN index, not the residue index, and is filled with 66
      # wherever the pair is not in the same residue AND chain, so for a protein
      # everything off the diagonal is 66.
      tf = batch.token_features
      ri = tf.residue_index.astype(jnp.int32)
      ti = tf.token_index.astype(jnp.int32)
      same_chain = tf.asym_id[:, None] == tf.asym_id[None, :]
      rss = jnp.where(same_chain,
                      jnp.clip(ri[:, None] - ri[None, :] + 33, 0, 65), 66)
      same_res = (ri[:, None] == ri[None, :]) & same_chain
      rts = jnp.where(same_res,
                      jnp.clip(ti[:, None] - ti[None, :] + 32, 0, 65), 66)
      rel_feat = jnp.concatenate(
          [jax.nn.one_hot(rss, 67), jax.nn.one_hot(rts, 67)], axis=-1
      ).astype(pair_activations.dtype)
      pair_activations += hm.Linear(
          self.config.pair_channel, use_bias=True, name='position_activations'
      )(rel_feat)
      return pair_activations

    rel_feat = featurization.create_relative_encoding(
        seq_features=batch.token_features,
        max_relative_idx=self.config.max_relative_idx,
        max_relative_chain=self.config.max_relative_chain,
    )
    rel_feat = rel_feat.astype(pair_activations.dtype)

    pair_activations += hm.Linear(
        self.config.pair_channel, name='position_activations'
    )(rel_feat)
    return pair_activations

  @hk.transparent
  def _seq_pair_embedding(
      self,
      token_features: features.TokenFeatures,
      target_feat: jnp.ndarray,
      single_act: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generated Pair embedding from sequence.

    OpenDDE builds the pair init from the single embedding s_init (single_act) rather
    than target_feat; when single_act is provided (opendde branch) left/right_single
    read it (so those linears are c_s->pair, not target_feat->pair). Otherwise the
    stock AF3 path reads target_feat.
    """
    src = single_act if single_act is not None else target_feat
    left_single = hm.Linear(self.config.pair_channel, name='left_single')(
        src
    )[:, None]
    right_single = hm.Linear(self.config.pair_channel, name='right_single')(
        src
    )[None]
    dtype = left_single.dtype
    pair_activations = left_single + right_single
    num_residues = pair_activations.shape[0]
    assert pair_activations.shape == (
        num_residues,
        num_residues,
        self.config.pair_channel,
    )
    mask = token_features.mask
    pair_mask = (mask[:, None] * mask[None, :]).astype(dtype)
    assert pair_mask.shape == (num_residues, num_residues)
    return pair_activations, pair_mask  # pytype: disable=bad-return-type  # jax-ndarray

  @hk.transparent
  def _embed_bonds(
      self,
      batch: feat_batch.Batch,
      pair_activations: jnp.ndarray,
  ) -> jnp.ndarray:
    """Embeds bond features and merges into pair activations."""
    contact_matrix = token_bond_matrix(
        batch,
        symmetrize=self.global_config.model in model_config.OPENFOLD3_LINEAGE,
    )
    bonds_act = hm.Linear(self.config.pair_channel, name='bond_embedding')(
        contact_matrix[:, :, None].astype(pair_activations.dtype)
    )
    pair_activations = pair_activations + bonds_act

    if self.global_config.model == 'boltz2':
      # Boltz's z-init carries two more terms that AF3 has no equivalent for, and
      # BOTH contribute even when nothing is annotated:
      #   token_bonds_type -- an nn.Embedding, so the UNSPECIFIED row (index 0) is a
      #     learned NONZERO vector added to every pair, not a no-op.
      #   contact_conditioning -- distance restraints; unconstrained inputs still get
      #     `encoding_unspecified` on every pair.
      # Omitting them left a constant offset on z for every input (a protein tolerates
      # it; measured 0.969 -> 0.974 pair corr on a lone ligand).
      #
      # contact_conditioning still runs on a placeholder, exactly as the confidence
      # head's copy does: our featuriser has no distance-restraint field, so this is
      # what boltz produces for an unconstrained input -- a coverage limit, not a
      # silent divergence.
      c = self.config.pair_channel
      dtype = pair_activations.dtype
      n = pair_activations.shape[0]
      # token_bonds_type is an nn.Embedding over bond ORDER (boltz: type_bonds =
      # bond_type_id + 1, so 0=no bond, 2=SINGLE, 3=DOUBLE...), and row 0 is a learned
      # constant, so it contributes on unbonded pairs too.
      #
      # Getting only row 0 right is not enough: applying it everywhere is exact for a
      # bond-free input (identical to native on all 4624 pairs of 6MRR) but wrong on
      # bonded pairs by up to 8.1, and doing that wrecked a lone ligand (0.20 -> 18 A)
      # because those are precisely the pairs that define a molecule's geometry. The
      # orders now come from the CCD bond table via `ligand_ligand_bond_order`.
      pair_activations += hm.Linear(c, name='token_bonds_type_embed')(
          jax.nn.one_hot(
              token_bond_type_matrix(
                  batch,
                  symmetrize=self.global_config.model
                  in model_config.OPENFOLD3_LINEAGE,
              ),
              7,
          ).astype(dtype))
      pair_activations += _boltz2_contact_conditioning(
          jax.nn.one_hot(jnp.zeros((n, n), jnp.int32), 5),
          jnp.zeros((n, n), jnp.float32), c, dtype)
    return pair_activations

  @hk.transparent
  def _embed_template_pair(
      self,
      batch: feat_batch.Batch,
      pair_activations: jnp.ndarray,
      pair_mask: jnp.ndarray,
      key: jnp.ndarray,
      use_dropout=False,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embeds Templates and merges into pair activations."""
    # A template stack configured with ZERO blocks means the model has no template
    # path at all, not a template path with an empty stack. Protenix's own
    # TemplateEmbedder.forward opens with
    #     if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
    #         # Compatible with the Protenix 0.5.0 model series
    #         return 0
    # so for that lineage (protenix05, mini, tiny) native contributes EXACTLY zero
    # even when templates are supplied -- the fused-embedder tensors those
    # checkpoints still carry are vestigial. Running the embedder anyway would add
    # a term native never computes: with no templates present a_tij is 0, but
    # u_proj(relu(v_norm(z_proj(z_norm(z))))) is NOT, so the divergence is real and
    # is not masked away by an empty template batch.
    if not self.config.template.template_stack.num_layer:
      return pair_activations, key
    dtype = pair_activations.dtype
    key, subkey = jax.random.split(key)
    # Boltz-2 uses a distinct template module (own featurisation + forward, validated
    # corr 1.0 vs Boltz's TemplateModule); see Boltz2TemplateEmbedding.
    if self.global_config.model == 'rosettafold3':
      # RF3 template = distance-distribution conditioning (66-d: 64-bin CA-CA
      # distogram + has_condition + noise_level), same fused forward; see
      # RoseTTAFold3TemplateEmbedding.
      template_module = template_modules.RoseTTAFold3TemplateEmbedding(
          self.config.template, self.global_config)
    elif self.global_config.model in model_config.PROTENIX_FAMILY:
      # Protenix rides the Boltz2 fused-template forward (same converter scopes) with
      # its own 108-d feature builder (39-bin dgram, normalized frame-at-i unit vector,
      # 32-restype); see Protenix2TemplateEmbedding.
      template_module = template_modules.Protenix2TemplateEmbedding(
          self.config.template, self.global_config)
    elif self.global_config.model == 'boltz2':
      template_module = template_modules.Boltz2TemplateEmbedding(
          self.config.template, self.global_config)
    else:
      template_module = template_modules.TemplateEmbedding(
          self.config.template, self.global_config
      )
    templates = batch.templates
    asym_id = batch.token_features.asym_id
    # Construct a mask such that only intra-chain template features are
    # computed, since all templates are for each chain individually.
    multichain_mask = (asym_id[:, None] == asym_id[None, :]).astype(dtype)

    template_fn = functools.partial(
        template_module, key=subkey, use_dropout=use_dropout)
    template_act = template_fn(
        query_embedding=pair_activations,
        templates=templates,
        multichain_mask_2d=multichain_mask,
        padding_mask_2d=pair_mask,
    )
    return pair_activations + template_act, key

  @hk.transparent
  def _embed_process_msa(
      self,
      msa_batch: features.MSA,
      pair_activations: jnp.ndarray,
      pair_mask: jnp.ndarray,
      key: jnp.ndarray,
      target_feat: jnp.ndarray,
      use_dropout=False,
      is_ligand: jnp.ndarray | None = None,
      asym_id: jnp.ndarray | None = None,
      single_post_recycle: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Processes MSA and returns updated pair activations."""
    dtype = pair_activations.dtype
    chai1 = self.global_config.model == 'chai1'
    # chai does NOT shuffle: it takes the MSA in the order given and keeps the
    # query first. Shuffling here would also break the query-row handling below,
    # since gumbel_argsort ranks EVERY row including row 0 -- the query would
    # land somewhere random and some other sequence would be labelled as it.
    if not chai1:
      msa_batch, key = featurization.shuffle_msa(key, msa_batch)
    msa_batch = featurization.truncate_msa_batch(msa_batch, self.config.num_msa)
    msa_feat = featurization.create_msa_feat(
        msa_batch, self.soft_seq, self.design_mask, chai1=chai1,
        is_ligand=is_ligand, asym_id=asym_id).astype(dtype)

    # Boltz-2 appends an is_paired MSA feature (Boltz msa features = [onehot(33), has_deletion,
    # deletion_value, is_paired]; ours are [onehot(32), has_deletion, deletion_value]). For a
    # single-chain / unpaired MSA Boltz marks the query row paired (=1) and the rest 0; that
    # matches here (row 0 = query). Verified msa_activations then reproduces Boltz exactly (1.0).
    if self.global_config.model in ('boltz2', 'rosettafold3'):
      # RF3 msa feat = onehot(32)+has_del+del_val+is_paired = 35 (same is_paired append; RF3
      # uses onehot(32) like us, not boltz's 33).
      #
      # The two disagree on what the flag MEANS. Boltz marks the query row paired (=1).
      # RF3's `add_residue_is_paired_feature` marks rows that were MSA-PAIRED ACROSS CHAINS,
      # so for a single chain (or any unpaired MSA, which is all this codebase builds) it is
      # 0 everywhere INCLUDING the query -- confirmed against the native featurised batch,
      # whose msa_stack[..., 34] is identically zero. Setting the query to 1 here added a
      # constant bias to every MSA embedding.
      num_msa_rows = msa_feat.shape[0]
      query_paired = 0.0 if self.global_config.model == 'rosettafold3' else 1.0
      is_paired = jnp.where(jnp.arange(num_msa_rows) == 0, query_paired, 0.0).astype(dtype)
      is_paired = jnp.broadcast_to(is_paired[:, None, None], msa_feat.shape[:2] + (1,))
      msa_feat = jnp.concatenate([msa_feat, is_paired], axis=-1)

    # chai's MSA feature embedding is feature_embedding's input_projs.MSA,
    # which unlike ours carries a bias.
    msa_activations = hm.Linear(
        self.config.msa_channel, use_bias=chai1, name='msa_activations'
    )(msa_feat)

    # chai DOES add a single-representation term, `msa_module.linear_s2m`. It
    # lives INSIDE msa_module, which is how it was first missed -- the search
    # that concluded otherwise excluded that module -- and skipping it left the
    # stack short an entire input.
    s_for_msa = target_feat
    if chai1 and single_post_recycle is not None:
      s_for_msa = single_post_recycle.astype(target_feat.dtype)
    msa_activations += hm.Linear(
        self.config.msa_channel, name='extra_msa_target_feat'
    )(s_for_msa)[None]
    msa_mask = msa_batch.mask.astype(dtype)

    # Evoformer MSA stack.
    evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
    masks = {'msa': msa_mask, 'pair': pair_mask}

    def evoformer_fn(x):
      return modules.EvoformerIteration(
          self.config.msa_stack, self.global_config, name='msa_stack'
      )(
          activations=x,
          masks=masks,
          use_dropout=use_dropout,
      )

    if os.environ.get('CHAI_TAP_MSA_BLOCKS') == '1':
      # Diagnostic only (default OFF, like CHAI_NOCHURN). A python loop would
      # SHARE one set of weights across all four blocks -- layer_stack gives
      # each block its own -- so the blocks still run under layer_stack and the
      # per-block pair reps ride out as its per-layer outputs, exactly as the
      # atom stack's taps do.
      MSA_BLOCK_TAPS.clear()

      def tapped_fn(x):
        out = evoformer_fn(x)
        return out, out['pair']

      evoformer_output, per_block = hk.experimental.layer_stack(
          self.config.msa_stack.num_layer, with_per_layer_inputs=True
      )(tapped_fn)(evoformer_input)
      MSA_BLOCK_TAPS.append(per_block)
    else:
      evoformer_stack = _stack(self.config.msa_stack.num_layer, evoformer_fn,
                               self.config.pairformer.block_remat)
      evoformer_output = evoformer_stack(evoformer_input)
    pair_out = evoformer_output['pair']

    if self.global_config.model in ('boltz2', 'chai1'):
      # chai does the SAME double count, and it was the first of the three bugs
      # the trunk replica turned up: its trace reads
      # `z20 = token_pair_repr + (pair_repr13 + tri_attn_output2)`, so the
      # pre-MSA representation is added twice. Proved by ablating every output
      # projection in the trunk to zero -- it then returned exactly 2x its input
      # pair rep (2.0003*z_init + 1.9988*recycle) while the single track stayed
      # at 1.0007. Fixed in the numpy replica long before this branch existed,
      # which is why the assembled graph folded to 19 A with a distogram at
      # corr 0.933 while the replica reached 0.9717 argmax agreement.
      #
      # Boltz's MSAModule RETURNS the updated z (each MSALayer residual-updates it:
      # `z = z + outer_product_mean(...)` then `z = pairformer_layer(z, ...)`), and its
      # caller then does `z = z + msa_module(z, ...)`. So boltz computes 2*z_in + delta,
      # not z_in + delta. Whether that double-add was intended upstream does not matter:
      # the trained weights were fitted with it, and AF3's `pair += msa_update` shape
      # drops one copy of z. Measured on the lone ligand -- our MSA contribution had
      # std 9.1 against native's 18.1 (corr 0.916), and z entering the pairformer was
      # 15.8 vs 25.9.
      pair_out = pair_out + pair_activations

    return pair_out, key

  # ColabDesign2: the soft sequence and its mask, set by Model before use.
  # Attributes rather than call arguments so the change stays at the two ends --
  # here and in Model -- instead of threading through every intermediate call.
  soft_seq = None
  design_mask = None

  def __call__(
      self,
      batch: feat_batch.Batch,
      prev: dict[str, jnp.ndarray],
      target_feat: jnp.ndarray,
      key: jnp.ndarray,
      use_dropout=False,
  ) -> dict[str, jnp.ndarray]:

    assert self.global_config.bfloat16 in {'all', 'none'}

    num_residues = target_feat.shape[0]
    assert batch.token_features.aatype.shape == (num_residues,)

    dtype = (
        jnp.bfloat16 if self.global_config.bfloat16 == 'all' else jnp.float32
    )

    with utils.bfloat16_context():
      # OpenDDE derives the pair init from s_init (single_activations) instead of
      # target_feat -- compute single_activations early so it can feed pair init;
      # the stock path leaves it None here and builds it after MSA (below).
      opendde = self.global_config.model == 'opendde'
      single_activations = None
      if opendde:
        single_activations = hm.Linear(
            self.config.seq_channel, name='single_activations'
        )(target_feat)

      pair_activations, pair_mask = self._seq_pair_embedding(
          batch.token_features, target_feat, single_activations
      )

      # chai's diffusion conditions on the token embedder's PURE z_init, with
      # no recycled term in it (its trunk takes z_init and the recycled rep as
      # separate arguments and adds them internally). Everything here is an
      # addition, so applying the relative encoding first and capturing before
      # the recycle add gives an identical activation and a clean z_init.
      chai = self.global_config.model == 'chai1'
      pair_init = None

      # chai seeds its recycle carry with the INITIAL representations, not with
      # zeros: `token_pair_trunk_repr = token_pair_initial_repr` before the
      # loop, so pass 1 already adds `recycle_proj(LN(z_init))`. AF3 starts from
      # zeros, which makes our pass 1 a different function of the inputs than
      # chai's. A fori_loop cannot branch on the iteration, so the first pass is
      # flagged in the carry and the flag selects which tensor to recycle.
      first = prev.get('recycle_first')

      def _add_prev(act, init_act):
        z_prev = prev['pair'].astype(act.dtype)
        if first is not None:
          z_prev = jnp.where(first > 0.5, init_act.astype(z_prev.dtype), z_prev)
        if self.global_config.model in model_config.SSM_RECYCLE:
          # ESMFold2 recycles through a discretised diagonal SSM ("parcae")
          # rather than an addition:
          #     z = a * z_prev + b(norm(z_inject))
          # against AF3's
          #     z = z_inject + prev_embedding(norm(z_prev))
          # Same two modules, applied to the OTHER operand, plus a per-channel
          # decay. a and b are input-independent in the checkpoint, so b folds
          # into prev_embedding and only `recycle_decay` is new; at a = 1 with
          # the operands swapped back this reduces to AF3's own recycling.
          decay = hk.get_parameter('recycle_decay', [act.shape[-1]],
                                   dtype=act.dtype, init=jnp.ones)
          return decay * z_prev + hm.Linear(
              act.shape[-1],
              name='prev_embedding',
              initializer=self.global_config.final_init,
          )(
              hm.LayerNorm(name='prev_embedding_layer_norm')(act)
          )
        return act + hm.Linear(
            act.shape[-1],
            name='prev_embedding',
            initializer=self.global_config.final_init,
        )(
            hm.LayerNorm(name='prev_embedding_layer_norm')(z_prev)
        )

      if chai:
        pair_activations = self._relative_encoding(batch, pair_activations)
        pair_init = pair_activations
        pair_activations = _add_prev(pair_activations, pair_init)
      else:
        pair_activations = _add_prev(pair_activations, None)
        pair_activations = self._relative_encoding(batch, pair_activations)

      # chai has NO bond feature: its 163-column token-pair stream is
      # docking/chain/entity/separation/restraints only (converters.chai1
      # PAIR_COLS). So bond_embedding has no source weight, and running it left
      # a RANDOM-init Linear adding a term chai does not have. Harmless for a
      # bare protein, where the contact matrix is all zero -- but a ligand has
      # intra-ligand bonds (BTN contributes 34 nonzero entries), so this was
      # live on exactly the inputs the port had just started supporting.
      if self.global_config.model != 'chai1':
        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )

      pair_activations, key = self._embed_template_pair(
          batch=batch,
          pair_activations=pair_activations,
          pair_mask=pair_mask,
          key=key,
          use_dropout=use_dropout,
      )
      if single_activations is None:   # stock path: not hoisted for opendde above
        single_activations = hm.Linear(
            self.config.seq_channel, name='single_activations'
        )(target_feat)

      s_prev = prev['single'].astype(single_activations.dtype)
      if first is not None:
        # `single_activations` here is still s_init -- chai's
        # token_single_trunk_initial_repr, which is exactly what it seeds the
        # carry with.
        s_prev = jnp.where(first > 0.5,
                           single_activations.astype(s_prev.dtype), s_prev)
      single_activations += hm.Linear(
          single_activations.shape[-1],
          name='prev_single_embedding',
          initializer=self.global_config.final_init,
      )(
          hm.LayerNorm(name='prev_single_embedding_layer_norm')(s_prev)
      )

      pair_activations, key = self._embed_process_msa(
          msa_batch=batch.msa,
          pair_activations=pair_activations,
          pair_mask=pair_mask,
          key=key,
          target_feat=target_feat,
          use_dropout=use_dropout,
          is_ligand=batch.token_features.is_ligand,
          asym_id=batch.token_features.asym_id,
          # chai's msa_module.linear_s2m reads the single AFTER its recycle
          # term is added (its argument traces to
          # add(token_single_trunk_initial_repr, recycle_proj(prev))), not
          # s_init. Feeding s_init put the block-0 outer product at corr 0.613;
          # this takes it to 0.999993.
          single_post_recycle=single_activations,
      )
      del key  # Unused after this point.


      # ESMFold2's trunk has NO single track at all -- 48 pair-only blocks, and
      # its structure head is handed s_trunk=None. Building the single track and
      # zeroing it would still carry the single through unchanged (it is a
      # residual) and would create parameters no checkpoint can fill, so the
      # track is not built.
      pair_only = self.global_config.model in model_config.PAIR_ONLY_TRUNK

      def pairformer_fn(x):
        pairformer_iteration = modules.PairFormerIteration(
            self.config.pairformer,
            self.global_config,
            with_single=not pair_only,
            name='trunk_pairformer',
        )
        pair_act, single_act = x
        if pair_only:
          return pairformer_iteration(
              act=pair_act,
              pair_mask=pair_mask,
              use_dropout=use_dropout,
          ), single_act
        return pairformer_iteration(
            act=pair_act,
            single_act=single_act,
            pair_mask=pair_mask,
            seq_mask=batch.token_features.mask.astype(dtype),
            use_dropout=use_dropout,   # traced; captured from the enclosing call
        )

      pairformer_stack = _stack(self.config.pairformer.num_layer,
                                pairformer_fn,
                                self.config.pairformer.block_remat)

      pair_activations, single_activations = pairformer_stack(
          (pair_activations, single_activations)
      )

      assert pair_activations.shape == (
          num_residues,
          num_residues,
          self.config.pair_channel,
      )
      assert single_activations.shape == (num_residues, self.config.seq_channel)
      assert len(target_feat.shape) == 2
      assert target_feat.shape[0] == num_residues
      output = {
          'single': single_activations,
          'pair': pair_activations,
          'target_feat': target_feat,
          # chai's diffusion conditions on z_init alongside z_trunk. Cast to
          # float32: this rides the recycle loop's scan carry, which is float32,
          # and the trunk computes in bfloat16 inside utils.bfloat16_context.
          **({'pair_init': pair_init.astype(jnp.float32)}
             if pair_init is not None else {}),
          # consumed above; every pass after the first recycles for real
          **({'recycle_first': jnp.zeros((), jnp.float32)}
             if first is not None else {}),
      }

    return output
