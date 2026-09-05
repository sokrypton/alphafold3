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

"""Per-atom cross attention."""
import dataclasses

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from . import diffusion_transformer
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp


class AtomCrossAttEncoderConfig(base_config.BaseConfig):
  per_token_channels: int = 768
  per_atom_channels: int = 128
  atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
      base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
  )
  per_atom_pair_channels: int = 16


def _per_atom_conditioning(
    config: AtomCrossAttEncoderConfig,
    batch: feat_batch.Batch,
    name: str,
    global_config: model_config.GlobalConfig | None = None,
    need_pair: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """computes single and pair conditioning for all atoms in each token.

  need_pair=False returns None for the pair half and skips the two Linears that
  only it uses. Its sole caller discards that half, so those parameters were
  being created, left unmapped, and computed into a thrown-away tensor.
  """

  c = config
  # Compute per-atom single conditioning
  # Shape (num_tokens, num_dense, channels)
  act = hm.Linear(
      c.per_atom_channels, precision='highest', name=f'{name}_embed_ref_pos'
  )(batch.ref_structure.positions)
  act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_mask')(
      batch.ref_structure.mask.astype(jnp.float32)[:, :, None]
  )
  # Element is encoded as atomic number if the periodic table, so
  # 128 should be fine.
  act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_element')(
      jax.nn.one_hot(batch.ref_structure.element, 128)
  )
  # chai feeds the RAW charge; AF3 (and everyone else) feeds arcsinh(charge).
  # Identical at charge 0, so a neutral-only test would never see it.
  charge = batch.ref_structure.charge
  if global_config is not None and global_config.model != 'chai1':
    charge = jnp.arcsinh(charge)
  act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_charge')(
      charge[:, :, None]
  )
  # Characters are encoded as ASCII code minus 32, so we need 64 classes,
  # to encode all standard ASCII characters between 32 and 96.
  atom_name_chars_1hot = jax.nn.one_hot(batch.ref_structure.atom_name_chars, 64)
  num_token, num_dense, _ = act.shape
  act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_atom_name')(
      atom_name_chars_1hot.reshape(num_token, num_dense, -1)
  )
  if global_config.model == 'boltz2':
    # Boltz builds this as ONE Linear over the concatenated 388-d atom feature vector,
    # and it is `Linear`, not `LinearNoBias` -- so it carries a bias that AF3's
    # per-feature bias-free Linears have no slot for. Dropping it removed a constant
    # 128-vector (std 0.134) from EVERY atom's embedding, about a quarter of `c`'s own
    # std: per-atom c corr was 0.912 against native with byte-identical inputs.
    # Everything downstream inherited it -- `a` 0.949, s_inputs 0.934, trunk z 0.969.
    act += hk.get_parameter(f'{name}_embed_atom_features_bias',
                            [c.per_atom_channels], act.dtype, init=jnp.zeros)
  if global_config.model == 'rosettafold3':
    # RF3 also adds process_atom_level_embedding(f['atom_level_embedding']) to the
    # atom single rep (mlff.ConformerEmbeddingWeightedAverage). Without conformer
    # embeddings that input is all zeros, but the MLP has biases and its tail has a
    # LayerNorm, so it emits a fixed NONZERO vector -- the same for every atom, and
    # two thirds the magnitude of the ref-feature embedding. Dropping it (a zero
    # feature is not a zero contribution) left every atom's embedding pointing the
    # wrong way. The converter collapses that subtree to this constant.
    act += hk.get_parameter(f'{name}_conformer_embedding_bias',
                            [c.per_atom_channels], dtype=act.dtype,
                            init=hk.initializers.Constant(0.0))
  if global_config.model in model_config.NORMED_ATOM_FEATURES:
    # ESMFold2 builds this as ONE Linear over the concatenated 389-d atom
    # features and then LAYER-NORMS the result (atom_linear -> atom_norm). AF3
    # sums bias-free per-feature Linears and does not normalise. The sum is the
    # same tensor -- a fused Linear over a concatenation IS the sum of its column
    # blocks, which is why the converter can split atom_linear five ways -- but
    # the LayerNorm is data-dependent and cannot be folded into any of them.
    act = hm.LayerNorm(name=f'{name}_atom_features_norm')(act)
  act *= batch.ref_structure.mask.astype(jnp.float32)[:, :, None]

  # Compute pair conditioning
  # shape (num_tokens, num_dense, num_dense, channels)
  # Embed single features
  row_act = hm.Linear(
      c.per_atom_pair_channels, name=f'{name}_single_to_pair_cond_row'
  )(jax.nn.relu(act))
  col_act = hm.Linear(
      c.per_atom_pair_channels, name=f'{name}_single_to_pair_cond_col'
  )(jax.nn.relu(act))
  pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
  if not need_pair:
    return act, None
  # Embed pairwise offsets
  pair_act += hm.Linear(
      c.per_atom_pair_channels,
      precision='highest',
      name=f'{name}_embed_pair_offsets',
  )(
      batch.ref_structure.positions[:, :, None, :]
      - batch.ref_structure.positions[:, None, :, :]
  )
  # Embed pairwise inverse squared distances
  sq_dists = jnp.sum(
      jnp.square(
          batch.ref_structure.positions[:, :, None, :]
          - batch.ref_structure.positions[:, None, :, :]
      ),
      axis=-1,
  )
  pair_act += hm.Linear(
      c.per_atom_pair_channels, name=f'{name}_embed_pair_distances'
  )(1.0 / (1 + sq_dists[:, :, :, None]))

  return act, pair_act


@dataclasses.dataclass(frozen=True)
class AtomCrossAttEncoderOutput:
  token_act: jnp.ndarray  # (num_tokens, ch)
  skip_connection: jnp.ndarray  # (num_subsets, num_queries, ch)
  queries_mask: jnp.ndarray  # (num_subsets, num_queries)
  queries_single_cond: jnp.ndarray  # (num_subsets, num_queries, ch)
  keys_mask: jnp.ndarray  # (num_subsets, num_keys)
  keys_single_cond: jnp.ndarray  # (num_subsets, num_keys, ch)
  pair_cond: jnp.ndarray  # (num_subsets, num_queries, num_keys, ch)
  # the query the atom stack consumes: chai's `a + prev_pos_embed(scaled)` --
  # per-atom features plus the noisy-coordinate term, but NOT s_trunk. Carried so
  # a harness can split the atom conditioning into its two terms -- the per-atom
  # feature projection and the token_to_atom_single(s_trunk) broadcast -- without
  # re-implementing either. Defaulted, so it must stay last in the dataclass.
  query_base: jnp.ndarray | None = None
  # chai-1 only: (num_subsets, num_queries, num_keys) bool, true where the query
  # and key atoms belong to the SAME token and both are real. Carried so the
  # decoder's atom stack reuses the encoder's mask instead of rebuilding it.
  # Defaulted, so it must stay last in the dataclass.
  same_token_mask: jnp.ndarray | None = None


jax.tree_util.register_dataclass(
    AtomCrossAttEncoderOutput,
    data_fields=[f.name for f in dataclasses.fields(AtomCrossAttEncoderOutput)],
    meta_fields=[],
)



def _chiral_position_grads(positions, chirals):
  """d/dx of sum_centres (improper_dihedral(x) - ideal)^2, RF3's chirality signal.

  RF3 (loss.calc_chiral_grads_flat_impl) hand-derives this closed form over ~150 lines
  of chain rule; jax.grad of the same scalar is exact and far easier to check. The loss
  is a SUM (not a mean) over centres -- RF3's dmse_ddih = 2*(dih - true_dih) with no 1/N.

  positions: (num_token, num_dense, 3), the SCALED noisy coords -- RF3 feeds the same
  R_L it gives process_r. Returns the same shape.
  """
  centers, angles = chirals.centers, chirals.angles
  if centers.shape[0] == 0:
    return jnp.zeros_like(positions)
  flat = positions.reshape(-1, positions.shape[-1])
  # angle == 0 marks a padding row; a real centre is always +-arcsin(1/sqrt(3)).
  valid = (angles != 0.0).astype(flat.dtype)

  def dihedral_loss(x):
    p = x[centers]                                        # (M, 4, 3)
    a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    eps = 1e-6                                            # RF3's eps
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / (jnp.linalg.norm(b1, axis=-1, keepdims=True) + eps)
    v = b0 - jnp.sum(b0 * b1n, axis=-1, keepdims=True) * b1n
    w = b2 - jnp.sum(b2 * b1n, axis=-1, keepdims=True) * b1n
    dih = jnp.arctan2(jnp.sum(jnp.cross(b1n, v) * w, -1) + eps,
                      jnp.sum(v * w, -1) + eps)
    return jnp.sum(valid * jnp.square(dih - angles))

  # RF3 detaches: the chirality gradient is an INPUT FEATURE, not a path for learning.
  grads = jax.grad(dihedral_loss)(jax.lax.stop_gradient(flat))
  return jnp.nan_to_num(grads).reshape(positions.shape)


def _build_atom_rope(ref_pos, ref_space_uid, head_dim, cfg):
  """3D rotary from the reference conformer, in whatever layout it is handed.

  ESMFold2 packs 3 spatial axes x n_spatial pairs plus n_uid pairs of
  ref_space_uid, which for its 4-head/32-channel atom attention is
  3*2 + 10 = 16 = head_dim/2 exactly. Queries and keys must be built separately:
  they are different subsets of the same flat atom list, so one rotation does
  not serve both.
  """
  inv = lambda n, base: (1.0 / (base ** (np.arange(n, dtype=np.float32) / n)))
  sp = jnp.asarray(inv(cfg['n_spatial'], cfg['spatial_base']))
  ui = jnp.asarray(inv(cfg['n_uid'], cfg['uid_base']))
  fs = (ref_pos[..., None] * sp).reshape(ref_pos.shape[:-1] + (-1,))
  fu = ref_space_uid[..., None] * ui
  fr = jnp.concatenate([fs, fu], -1)
  half = head_dim // 2
  if fr.shape[-1] < half:
    pad = jnp.zeros(fr.shape[:-1] + (half - fr.shape[-1],), fr.dtype)
    fr = jnp.concatenate([fr, pad], -1)
  return jnp.cos(fr), jnp.sin(fr)


def atom_cross_att_encoder(
    token_atoms_act: jnp.ndarray | None,  # (num_tokens, max_atoms_per_token, 3)
    trunk_single_cond: jnp.ndarray | None,  # (num_tokens, ch)
    trunk_pair_cond: jnp.ndarray | None,  # (num_tokens, num_tokens, ch)
    config: AtomCrossAttEncoderConfig,
    global_config: model_config.GlobalConfig,
    batch: feat_batch.Batch,
    name: str,
) -> AtomCrossAttEncoderOutput:
  """Cross-attention on flat atom subsets and mapping to per-token features."""
  c = config

  # Compute single conditioning from atom meta data and convert to queries
  # layout.
  # (num_subsets, num_queries, channels)
  # The pair half of _per_atom_conditioning is DISCARDED here -- its only call
  # site -- so for chai1 skip building it. That was four Linears' worth of
  # parameters left at random init and computed into a thrown-away tensor: no
  # effect on results (atom_pair gates at 1.000000 either way), pure waste.
  # Gated rather than removed outright because other families' converters map
  # those names.
  token_atoms_single_cond, _ = _per_atom_conditioning(
      config, batch, name, global_config,
      need_pair=global_config.model != 'chai1',
  )
  token_atoms_mask = batch.predicted_structure_info.atom_mask
  queries_single_cond = atom_layout.convert(
      batch.atom_cross_att.token_atoms_to_queries,
      token_atoms_single_cond,
      layout_axes=(-3, -2),
  )
  queries_mask = atom_layout.convert(
      batch.atom_cross_att.token_atoms_to_queries,
      token_atoms_mask,
      layout_axes=(-2, -1),
  )

  # Boltz-2 and RF3: the atom encoder's QUERY (q) is the per-atom ref features ONLY
  # (q=c before s_trunk), while the CONDITIONING (c, used by the transformer's adaln) is
  # ref + s_trunk. AF3 instead folds trunk_single_cond into queries_single_cond which feeds
  # BOTH -> the large s_trunk swamps the query and scrambles the atom attention. So keep the
  # pre-trunk per-atom features (q_base) for the query and add s_trunk only to the
  # conditioning. RF3 does this explicitly: `Q_L = C_L` is taken BEFORE
  # `C_L = C_L + process_s_trunk(S_trunk)` (af3_diffusion_transformer.py AtomAttentionEncoder).
  # chai belongs here too: its trace builds the query as
  # `a + prev_pos_embed(scaled_coords)` -- the per-atom features BEFORE s_trunk
  # -- while the conditioning is `LayerNorm(a + token_to_atom_single(s_trunk))`.
  # Exactly the q != c split boltz2 and RF3 need.
  pre_trunk_query = global_config.model in ('boltz2', 'rosettafold3', 'chai1')
  q_base = queries_single_cond

  # If provided, broadcast single conditioning from trunk to all queries
  if trunk_single_cond is not None:
    trunk_single_cond = hm.Linear(
        c.per_atom_channels,
        precision='highest',
        initializer=global_config.final_init,
        name=f'{name}_embed_trunk_single_cond',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            # chai's token_to_atom_single.0 is an AFFINE LayerNorm carrying both
            # weight and bias (bias absmax 0.22 -- not negligible). Without an
            # offset the bias is silently dropped, which is a pure directional
            # error: it survives the affine-free LN of (a + t2a) downstream, so
            # atom_cond reads rms 1.0000 and corr 0.996 and looks like rounding.
            create_offset=global_config.model in ('boltz2', 'chai1',
                                                  'rosettafold3'),
            name=f'{name}_lnorm_trunk_single_cond',
        )(trunk_single_cond)
    )
    queries_single_cond += atom_layout.convert(
        batch.atom_cross_att.tokens_to_queries,
        trunk_single_cond,
        layout_axes=(-2,),
    )

  if global_config.model == 'chai1':
    # chai normalises the conditioning SUM:
    # `cond = LayerNorm(a + token_to_atom_single(s_trunk))`, affine-free.
    # This is not cosmetic. Without it the conditioning comes out 3.6x too
    # large, and since every adaLN in the atom stacks scales by (s + 1) off
    # this vector, the atom representation runs away -- measured 75x chai's
    # magnitude and uncorrelated (corr 0.045) at the encoder output.
    queries_single_cond = hm.LayerNorm(
        use_fast_variance=False, create_scale=False, create_offset=False,
        name=f'{name}_atom_cond_norm')(queries_single_cond)

  queries_single_cond = queries_single_cond * queries_mask[..., None]
  q_base = q_base * queries_mask[..., None]
  # Boltz/RF3 query uses per-atom features without s_trunk; AF3 uses the full conditioning.
  query_base = q_base if pre_trunk_query else queries_single_cond

  if token_atoms_act is None:
    # if no token_atoms_act is given (e.g. begin of evoformer), we use the
    # static conditioning only
    queries_act = query_base
  else:
    # Convert token_atoms_act to queries layout and map to per_atom_channels
    # (num_subsets, num_queries, channels)
    queries_act = atom_layout.convert(
        batch.atom_cross_att.token_atoms_to_queries,
        token_atoms_act,
        layout_axes=(-3, -2),
    )
    queries_act = hm.Linear(
        c.per_atom_channels,
        precision='highest',
        name=f'{name}_atom_positions_to_features',
    )(queries_act)
    if global_config.model == 'rosettafold3':
      # RF3 adds a chirality term to the QUERY alongside process_r(R_L): the gradient
      # of the chiral-centre dihedral error w.r.t. the same scaled noisy coordinates
      # (af3_diffusion_transformer.py, `if self.use_chiral_features`). It is the only
      # reflection-asymmetric signal in the network. Diffusion encoder only -- the
      # input-embedder encoder passes token_atoms_act=None, matching RF3's
      # `if R_L is not None`. No-op when the batch carries no chiral centres.
      chiral_grads = _chiral_position_grads(token_atoms_act, batch.chirals)
      queries_act += hm.Linear(
          c.per_atom_channels,
          precision='highest',
          name=f'{name}_atom_chiral_to_features',
      )(atom_layout.convert(
          batch.atom_cross_att.token_atoms_to_queries,
          chiral_grads,
          layout_axes=(-3, -2),
      ))
    queries_act *= queries_mask[..., None]
    queries_act += query_base
    # chai's `q_repr = a + prev_pos_embed(scaled_coords)` -- the query the atom
    # stack actually consumes. Captured HERE, after the coordinate term:
    # query_base alone is only `a`, and comparing that against chai's q_repr
    # reads as a 30% error that is really just the missing term.
    query_base = queries_act

  # Gather the keys from the queries. RF3 attends over ALL atoms, not a window.
  queries_to_keys = batch.atom_cross_att.queries_to_keys
  keys_single_cond = atom_layout.convert(
      queries_to_keys,
      queries_single_cond,
      layout_axes=(-3, -2),
  )
  keys_mask = atom_layout.convert(
      queries_to_keys, queries_mask, layout_axes=(-2, -1)
  )

  # chai-1's atom_block_pair_mask is exactly (same token) AND (both atoms real)
  # -- verified against the 6MRR seam over every one of its 184x32x128 entries.
  # Build it in the same two gathers the layouts already use: a dense per-atom
  # token index into the queries layout, then into the keys layout.
  same_token_mask = None
  if global_config.model == 'chai1':
    tok_idx = jnp.broadcast_to(
        jnp.arange(token_atoms_mask.shape[-2], dtype=jnp.int32)[:, None],
        token_atoms_mask.shape[-2:],
    )
    queries_tok = atom_layout.convert(
        batch.atom_cross_att.token_atoms_to_queries, tok_idx,
        layout_axes=(-2, -1),
    )
    keys_tok = atom_layout.convert(
        queries_to_keys, queries_tok, layout_axes=(-2, -1)
    )
    same_token_mask = (
        (queries_tok[..., :, None] == keys_tok[..., None, :])
        & queries_mask[..., :, None].astype(jnp.bool_)
        & keys_mask[..., None, :].astype(jnp.bool_)
    )

  # Embed single features into the pair conditioning.
  # shape (num_subsets, num_queries, num_keys, ch)
  row_act = hm.Linear(
      c.per_atom_pair_channels, name=f'{name}_single_to_pair_cond_row'
  )(jax.nn.relu(queries_single_cond))
  pair_cond_keys_input = atom_layout.convert(
      queries_to_keys,
      queries_single_cond,
      layout_axes=(-3, -2),
  )
  col_act = hm.Linear(
      c.per_atom_pair_channels, name=f'{name}_single_to_pair_cond_col'
  )(jax.nn.relu(pair_cond_keys_input))
  pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

  if trunk_pair_cond is not None:
    # If provided, broadcast the pair conditioning for the trunk (evoformer
    # pairs) to the atom pair activations. This should boost ligands, but also
    # help for cross attention within proteins, because we always have atoms
    # from multiple residues in a subset.
    # Map trunk pair conditioning to per_atom_pair_channels
    # (num_tokens, num_tokens, per_atom_pair_channels)
    trunk_pair_cond = hm.Linear(
        c.per_atom_pair_channels,
        precision='highest',
        initializer=global_config.final_init,
        name=f'{name}_embed_trunk_pair_cond',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            # same for token_pair_to_atom_pair.0
            create_offset=global_config.model in ('boltz2', 'chai1',
                                                  'rosettafold3'),
            name=f'{name}_lnorm_trunk_pair_cond',
        )(trunk_pair_cond)
    )

    # Create the GatherInfo into a flattened trunk_pair_cond from the
    # queries and keys gather infos.
    num_tokens = trunk_pair_cond.shape[0]
    # (num_subsets, num_queries)
    tokens_to_queries = batch.atom_cross_att.tokens_to_queries
    # (num_subsets, num_keys)
    tokens_to_keys = batch.atom_cross_att.tokens_to_keys
    # (num_subsets, num_queries, num_keys)
    trunk_pair_to_atom_pair = atom_layout.GatherInfo(
        gather_idxs=(
            num_tokens * tokens_to_queries.gather_idxs[:, :, None]
            + tokens_to_keys.gather_idxs[:, None, :]
        ),
        gather_mask=(
            tokens_to_queries.gather_mask[:, :, None]
            & tokens_to_keys.gather_mask[:, None, :]
        ),
        input_shape=jnp.array((num_tokens, num_tokens)),
    )
    # Gather the conditioning and add it to the atom-pair activations.
    pair_act += atom_layout.convert(
        trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
    )

  # Embed pairwise offsets
  queries_ref_pos = atom_layout.convert(
      batch.atom_cross_att.token_atoms_to_queries,
      batch.ref_structure.positions,
      layout_axes=(-3, -2),
  )
  queries_ref_space_uid = atom_layout.convert(
      batch.atom_cross_att.token_atoms_to_queries,
      batch.ref_structure.ref_space_uid,
      layout_axes=(-2, -1),
  )
  keys_ref_pos = atom_layout.convert(
      queries_to_keys,
      queries_ref_pos,
      layout_axes=(-3, -2),
  )
  keys_ref_space_uid = atom_layout.convert(
      queries_to_keys,
      queries_ref_space_uid,
      layout_axes=(-2, -1),
  )

  swa_rope = global_config.model in model_config.SWA_ROPE_ATOM_ATTENTION
  rope_q = rope_k = None
  swa_mask = None
  if swa_rope:
    cfg = model_config.ATOM_ROPE[global_config.model]
    head_dim = (c.atom_transformer.attention.key_dim
                or c.per_atom_channels) // c.atom_transformer.attention.num_head
    rope_q = _build_atom_rope(queries_ref_pos, queries_ref_space_uid, head_dim, cfg)
    rope_k = _build_atom_rope(keys_ref_pos, keys_ref_space_uid, head_dim, cfg)
    # The exact +/-64 window by RANK among valid atoms. AF3's key subset is only
    # block-aligned, so this is what actually defines ESMFold2's window; the
    # subset is merely wide enough to contain it.
    q_rank = jnp.cumsum(queries_mask.reshape(-1)) - 1
    q_rank = q_rank.reshape(queries_mask.shape)
    k_rank = atom_layout.convert(queries_to_keys, q_rank, layout_axes=(-2, -1))
    half = model_config.ATOM_ROPE_HALF_WINDOW
    swa_mask = jnp.abs(q_rank[:, :, None] - k_rank[:, None, :]) <= half
    swa_mask = swa_mask & queries_mask[:, :, None].astype(jnp.bool_)
    swa_mask = swa_mask & keys_mask[:, None, :].astype(jnp.bool_)

  offsets_valid = (
      queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
  )
  if global_config.model in model_config.OPENFOLD3_LINEAGE:
    # OF3 was trained with padded keys correctly excluded from offsets_valid.
    # Padded key atoms have ref_space_uid=0 (zero-fill), colliding with token 0.
    offsets_valid = offsets_valid & keys_mask[:, None, :].astype(jnp.bool_)
  offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
  chai = global_config.model == 'chai1'
  if chai:
    # chai does not embed raw offsets at all. Its atom-pair input is a 12-class
    # DISTOGRAM one-hot plus an inverse-square distance and a validity mask, so
    # unlike the single conditioning this cannot be folded into AF3's weights --
    # a one-hot distogram is not a linear function of the offsets. It can still
    # be built here rather than carried through featurisation, because AF3
    # already computes the offsets this needs.
    #
    # The bin index is a LEFT searchsorted over [0,1,2,3,4,5,6,8,12,16], i.e. the
    # number of bins strictly below the distance, and invalid pairs take the last
    # class (11). Matches chai's own generator exactly (1.0000 agreement).
    # The inverse-square value is NOT masked -- chai computes it on every pair
    # and carries validity in its own column.
    bins = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0],
                     dtype=offsets.dtype)
    sq = jnp.sum(jnp.square(offsets), axis=-1)
    # Compare SQUARED distances against squared bin edges. The obvious spelling,
    # `sqrt(maximum(sq, 1e-12)) > bins`, is wrong on the diagonal: the epsilon
    # that guards the sqrt turns a self-pair's exact 0 into 1e-6, and
    # 1e-6 > 0.0 is TRUE, so every self-pair lands in bin 1 instead of bin 0.
    # Measured against chai's own BlockedAtomPairDistogram that cost 11.3% of the
    # valid pairs -- exactly the self-pair fraction (573 self-pairs out of ~4800
    # same-residue pairs), and chai's bin 1 is legitimately empty because no two
    # distinct atoms are within 1 A. Squaring needs no epsilon and is exact.
    idx = jnp.sum((sq[..., None] > jnp.square(bins)).astype(jnp.int32), axis=-1)
    idx = jnp.where(offsets_valid, idx, len(bins) + 1)
    feat = jnp.concatenate([
        jax.nn.one_hot(idx, len(bins) + 2).astype(pair_act.dtype),
        (1.0 / (1.0 + sq))[..., None].astype(pair_act.dtype),
        offsets_valid[..., None].astype(pair_act.dtype),
    ], axis=-1)
    # chai's ATOM_PAIR projection carries a bias, and unlike the single side it
    # cannot be folded into a mask column: it applies to invalid pairs too.
    pair_act += hm.Linear(
        c.per_atom_pair_channels, use_bias=True,
        name=f'{name}_embed_atom_pair_feat'
    )(feat)
  else:
    pair_act += (
        hm.Linear(
            c.per_atom_pair_channels,
            precision='highest',
            name=f'{name}_embed_pair_offsets',
        )(offsets)
        * offsets_valid[:, :, :, None]
    )

  if not chai:
    # Embed pairwise inverse squared distances (chai folded these into the
    # distogram feature above)
    sq_dists = jnp.sum(jnp.square(offsets), axis=-1)
    pair_act += (
        hm.Linear(c.per_atom_pair_channels, name=f'{name}_embed_pair_distances')(
            1.0 / (1 + sq_dists[:, :, :, None])
        )
        * offsets_valid[:, :, :, None]
    )
    # Embed offsets valid mask
    pair_act += hm.Linear(
        c.per_atom_pair_channels, name=f'{name}_embed_pair_offsets_valid'
    )(offsets_valid[:, :, :, None].astype(jnp.float32))

  if chai:
    # chai's atom_pair_mlp is Linear, ReLU, Linear with the residual added
    # afterwards -- two layers, and NO relu on the way in. AF3 has three layers
    # and relus the input first.
    pair_act += hm.Linear(
        c.per_atom_pair_channels, initializer='relu', name=f'{name}_pair_mlp_2'
    )(jax.nn.relu(hm.Linear(
        c.per_atom_pair_channels, initializer='relu', name=f'{name}_pair_mlp_1'
    )(pair_act)))
  else:
    # Run a small MLP on the pair acitvations
    pair_act2 = hm.Linear(
        c.per_atom_pair_channels, initializer='relu', name=f'{name}_pair_mlp_1'
    )(jax.nn.relu(pair_act))
    pair_act2 = hm.Linear(
        c.per_atom_pair_channels, initializer='relu', name=f'{name}_pair_mlp_2'
    )(jax.nn.relu(pair_act2))
    pair_act += hm.Linear(
        c.per_atom_pair_channels,
        initializer=global_config.final_init,
        name=f'{name}_pair_mlp_3',
    )(jax.nn.relu(pair_act2))

  # Run the atom cross attention transformer.
  queries_act = diffusion_transformer.CrossAttTransformer(
      c.atom_transformer, global_config, name=f'{name}_atom_transformer_encoder'
  )(
      queries_act=queries_act,
      queries_mask=queries_mask,
      queries_to_keys=queries_to_keys,
      keys_mask=keys_mask,
      queries_single_cond=queries_single_cond,
      keys_single_cond=keys_single_cond,
      pair_cond=pair_act,
      pair_mask=swa_mask if swa_rope else same_token_mask,
      rope_q=rope_q,
      rope_k=rope_k,
  )
  queries_act *= queries_mask[..., None]
  skip_connection = queries_act

  # Convert back to token-atom layout and aggregate to tokens
  queries_act = hm.Linear(
      c.per_token_channels, name=f'{name}_project_atom_features_for_aggr'
  )(queries_act)
  token_atoms_act = atom_layout.convert(  # pyrefly: ignore[bad-assignment]
      batch.atom_cross_att.queries_to_token_atoms,
      queries_act,
      layout_axes=(-3, -2),
  )
  token_act = utils.mask_mean(
      token_atoms_mask[..., None], jax.nn.relu(token_atoms_act), axis=-2  # pyrefly: ignore[bad-argument-type]
  )

  return AtomCrossAttEncoderOutput(
      token_act=token_act,
      skip_connection=skip_connection,
      queries_mask=queries_mask,  # pyrefly: ignore[bad-argument-type]
      queries_single_cond=queries_single_cond,  # pyrefly: ignore[bad-argument-type]
      keys_mask=keys_mask,  # pyrefly: ignore[bad-argument-type]
      keys_single_cond=keys_single_cond,  # pyrefly: ignore[bad-argument-type]
      pair_cond=pair_act,
      query_base=query_base,
      same_token_mask=same_token_mask,
  )


class AtomCrossAttDecoderConfig(base_config.BaseConfig):
  per_atom_channels: int = 128
  atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
      base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
  )


def atom_cross_att_decoder(
    token_act: jnp.ndarray,  # (num_tokens, ch)
    enc: AtomCrossAttEncoderOutput,
    config: AtomCrossAttDecoderConfig,
    global_config: model_config.GlobalConfig,
    batch: feat_batch.Batch,
    name: str,
):  # (num_tokens, max_atoms_per_token, 3)
  """Mapping to per-atom features and self-attention on subsets."""
  c = config
  # map per-token act down to per_atom channels
  token_act = hm.Linear(
      c.per_atom_channels, name=f'{name}_project_token_features_for_broadcast'
  )(token_act)
  # Broadcast to token-atoms layout and convert to queries layout.
  num_token, max_atoms_per_token = (
      batch.atom_cross_att.queries_to_token_atoms.shape
  )
  token_atom_act = jnp.broadcast_to(
      token_act[:, None, :],
      (num_token, max_atoms_per_token, c.per_atom_channels),
  )
  queries_act = atom_layout.convert(
      batch.atom_cross_att.token_atoms_to_queries,
      token_atom_act,
      layout_axes=(-3, -2),
  )
  queries_act += enc.skip_connection
  queries_act *= enc.queries_mask[..., None]

  # chai conditions its DECODER on post_atom_cond_layernorm(cond) -- a second,
  # AFFINE LayerNorm over the encoder's atom conditioning. AF3 reuses the
  # encoder's conditioning unchanged.
  q_cond, k_cond = enc.queries_single_cond, enc.keys_single_cond
  if global_config.model == 'chai1':
    post = hm.LayerNorm(use_fast_variance=False,
                        name=f'{name}_post_atom_cond_layer_norm')
    q_cond, k_cond = post(q_cond), post(k_cond)

  # Run the atom cross attention transformer.
  queries_act = diffusion_transformer.CrossAttTransformer(
      c.atom_transformer, global_config, name=f'{name}_atom_transformer_decoder'
  )(
      queries_act=queries_act,
      queries_mask=enc.queries_mask,
      queries_to_keys=batch.atom_cross_att.queries_to_keys,
      keys_mask=enc.keys_mask,
      queries_single_cond=q_cond,
      keys_single_cond=k_cond,
      pair_cond=enc.pair_cond,
      pair_mask=enc.same_token_mask,
  )
  queries_act *= enc.queries_mask[..., None]
  queries_act = hm.LayerNorm(
      use_fast_variance=False,
      # chai's to_pos_updates starts with an AFFINE LayerNorm
      create_offset=global_config.model in ('boltz2', 'chai1',
                                                  'rosettafold3'),
      name=f'{name}_atom_features_layer_norm',
  )(queries_act)
  queries_position_update = hm.Linear(
      3,
      initializer=global_config.final_init,
      precision='highest',
      name=f'{name}_atom_features_to_position_update',
  )(queries_act)
  position_update = atom_layout.convert(
      batch.atom_cross_att.queries_to_token_atoms,
      queries_position_update,
      layout_axes=(-3, -2),
  )
  return position_update
