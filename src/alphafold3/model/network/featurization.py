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

"""Model-side of the input features processing."""

import functools

from alphafold3.constants import residue_names
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model.components import utils
import jax
import jax.numpy as jnp


def _grid_keys(key, shape):
  """Generate a grid of rng keys that is consistent with different padding.

  Generate random keys such that the keys will be identical, regardless of
  how much padding is added to any dimension.

  Args:
    key: A PRNG key.
    shape: The shape of the output array of keys that will be generated.

  Returns:
    An array of shape `shape` consisting of random keys.
  """
  if not shape:
    return key
  new_keys = jax.vmap(functools.partial(jax.random.fold_in, key))(
      jnp.arange(shape[0])
  )
  return jax.vmap(functools.partial(_grid_keys, shape=shape[1:]))(new_keys)


def _padding_consistent_rng(f):
  """Modify any element-wise random function to be consistent with padding.

  Normally if you take a function like jax.random.normal and generate an array,
  say of size (10,10), you will get a different set of random numbers to if you
  add padding and take the first (10,10) sub-array.

  This function makes a random function that is consistent regardless of the
  amount of padding added.

  Note: The padding-consistent function is likely to be slower to compile and
  run than the function it is wrapping, but these slowdowns are likely to be
  negligible in a large network.

  Args:
    f: Any element-wise function that takes (PRNG key, shape) as the first 2
      arguments.

  Returns:
    An equivalent function to f, that is now consistent for different amounts of
    padding.
  """

  def inner(key, shape, **kwargs):
    keys = _grid_keys(key, shape)
    signature = (
        '()->()'
        if jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key)
        else '(2)->()'
    )
    return jnp.vectorize(
        functools.partial(f, shape=(), **kwargs), signature=signature
    )(keys)

  return inner


def gumbel_argsort_sample_idx(
    key: jnp.ndarray, logits: jnp.ndarray
) -> jnp.ndarray:
  """Samples with replacement from a distribution given by 'logits'.

  This uses Gumbel trick to implement the sampling an efficient manner. For a
  distribution over k items this samples k times without replacement, so this
  is effectively sampling a random permutation with probabilities over the
  permutations derived from the logprobs.

  Args:
    key: prng key
    logits: logarithm of probabilities to sample from, probabilities can be
      unnormalized.

  Returns:
    Sample from logprobs in one-hot form.
  """
  gumbel = _padding_consistent_rng(jax.random.gumbel)
  z = gumbel(key, logits.shape)
  # This construction is equivalent to jnp.argsort, but using a non stable sort,
  # since stable sort's aren't supported by jax2tf
  axis = len(logits.shape) - 1
  iota = jax.lax.broadcasted_iota(jnp.int64, logits.shape, axis)
  _, perm = jax.lax.sort_key_val(
      logits + z, iota, dimension=-1, is_stable=False
  )
  return perm[::-1]


def blend_soft(one_hot: jax.Array, soft_seq, design_mask=None) -> jax.Array:
  """substitute a soft amino-acid distribution for the leading one-hot block

  ColabDesign2 addition. AF3 takes sequence as an integer aatype, so design needs
  a continuous relaxation of it; this is where that enters. The soft
  distribution covers the 20 standard amino acids and is zero-padded to AF3's
  token alphabet, which says "definitely a standard amino acid" rather than
  leaving UNK, GAP and the nucleic acid classes free.

  design_mask selects which tokens it applies to. Everything else keeps its true
  identity, which is what a binder target, a scaffolded motif, and any DNA or
  ligand token need -- a distribution over amino acids is meaningless on a
  ligand atom.

  This used to be done by replacing this module's functions at trace time from
  soft.py. That worked, but the jit cache key then had to hand-encode which
  patches were live, and when it did not two diffusion modes silently compiled
  to one graph.
  """
  if soft_seq is None:
    return one_hot
  # both arrays are aligned from the RIGHT, because the token axis is the last
  # but one in every caller: target_feat is (tokens, classes) while msa_feat is
  # (rows, tokens, classes). Aligning by leading dimensions instead put the mask
  # on the MSA row axis and silently produced (tokens, tokens, classes).
  n = soft_seq.shape[-1]
  soft = jnp.pad(soft_seq, [(0, 0)] * (soft_seq.ndim - 1) +
                 [(0, one_hot.shape[-1] - n)]).astype(one_hot.dtype)
  if design_mask is None:
    return jnp.broadcast_to(soft, one_hot.shape)
  return jnp.where(design_mask[..., None], soft, one_hot)


def hard_seq(soft_seq):
  """straight-through one-hot(argmax) of a soft distribution, or None.

  ColabDesign2. The profile-hard mode (AF2's pssm_hard=True) feeds the profile a
  DISCRETE one-hot of the designed sequence rather than its soft distribution.
  This reconstructs seq['hard'] from soft_seq at the blend site so the profile
  path needs no extra threading: argmax(softmax(logits)) == argmax(logits), and
  the straight-through estimator keeps the gradient flowing into soft_seq.
  """
  if soft_seq is None:
    return None
  one_hot = jax.nn.one_hot(jnp.argmax(soft_seq, axis=-1), soft_seq.shape[-1],
                           dtype=soft_seq.dtype)
  return soft_seq + jax.lax.stop_gradient(one_hot - soft_seq)


# Profile alphabet indices (residue_names.POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP):
# 0..19 = the 20 standard amino acids (0 = ALA), 20 = UNK ("some amino acid,
# identity unknown"), 21 = GAP ("no residue here"). The current 'frozen'
# placeholder is a one-hot at 0 -- poly-alanine, a confidently WRONG profile.
_PROFILE_UNK_INDEX = 20
_PROFILE_GAP_INDEX = 21


def token_profile(profile, token_index, design_mask):
  """set the profile to a one-hot at token_index on the design positions.

  ColabDesign2. A non-committal 'frozen' profile: rather than asserting
  poly-alanine (index 0), assert UNK (20, "unknown amino acid") or GAP (21). Off
  the design_mask -- targets, motifs, ligands -- the real profile is kept.
  """
  onehot = jax.nn.one_hot(token_index, profile.shape[-1], dtype=profile.dtype)
  onehot = jnp.broadcast_to(onehot, profile.shape)
  if design_mask is None:
    return onehot
  return jnp.where(design_mask[..., None], onehot, profile)


# ColabDesign2: how create_target_feat derives the MSA profile during design.
# This is the profile-feature analogue of the SEQUENCE's own soft/hard relaxation
# (opt['soft']/opt['hard']), and it is the AF3 twin of AF2's opt['pssm_hard'].
# Unified vocabulary (same words the AF2 runner uses for the profile):
#   'soft'  (DEFAULT) profile follows the designed sequence as a soft
#           distribution   == AF2 pssm_hard=False. REQUIRED for design so the
#           design forward matches the prediction forward; a no-op for prediction
#           (blend_soft returns the profile unchanged when soft_seq is None).
#   'hard'  profile follows the designed sequence as a straight-through argmax
#           one-hot          == AF2 pssm_hard=True -- the mode AF2 MULTIMER design
#           REQUIRES (a soft profile was not enough there). Untested for AF3; the
#           reason this knob exists.
#   'frozen' profile ignores the sequence (stays at the poly-alanine placeholder)
#           -- AF3's ORIGINAL behaviour, which reintroduces the design/prediction
#           mismatch. No AF2 analogue; kept only for ablation.
#   'unk'   profile ignores the sequence but asserts UNK ("some amino acid,
#           identity unknown") on the design positions instead of poly-alanine --
#           the HONEST non-committal frozen signal. AF2 never tried this (its
#           profile always tracks the sequence via pssm_hard).
#   'gap'   like 'unk' but asserts GAP ("no residue here"). Semantically wrong for
#           a query that has residues everywhere; the AF2-tried variant.
#   'zero'  no MSA profile information.
# Read at trace time, so switching modes recompiles -- an experiment knob, not a
# per-step option. See colabdesign2/af2/runner.py update_seq (the pssm_hard site).
PROFILE_MODE = 'soft'


def create_msa_feat(msa: features.MSA, soft_seq=None,
                    design_mask=None, chai1=False, is_ligand=None,
                    asym_id=None) -> jax.Array:
  """Create and concatenate MSA features."""
  rows = msa.rows
  if chai1 and is_ligand is not None:
    # A non-polymer token has no MSA, and chai does NOT mark it as a gap the way
    # we do. Read off its own features: the QUERY row carries class 20, the
    # unknown residue, and every other row carries 32, the mask -- where we put
    # our gap (21, i.e. chai's 31) on all of them. In our vocabulary those are
    # 20 and 31 respectively. Getting this wrong left the MSA input projection
    # exact on protein tokens (corr 0.999997) and adrift on ligand ones (0.774).
    lig = is_ligand.astype(bool)[None, :]
    query = jnp.arange(rows.shape[0])[:, None] == 0
    unk = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP.index(
        residue_names.UNK)                                     # 20
    mask_cls = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP  # 31, mask
    rows = jnp.where(lig, jnp.where(query, unk, mask_cls), rows)
  msa_1hot = jax.nn.one_hot(
      rows, residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1
  )
  msa_1hot = blend_soft(msa_1hot, soft_seq, design_mask)
  deletion_matrix = msa.deletion_matrix
  has_deletion = jnp.clip(deletion_matrix, 0.0, 1.0)[..., None]
  deletion_value = (jnp.arctan(deletion_matrix / 3.0) * (2.0 / jnp.pi))[
      ..., None
  ]

  msa_feat = [
      msa_1hot,
      has_deletion,
      deletion_value,
  ]

  if chai1:
    # chai's MSA stream is 42 columns in ALPHABETICAL feature order --
    #   IsPairedMSA(1) | MSADataSource(one-hot 6) | MSADeletionValue(1)
    #   | MSAHasDeletion(1) | MSAOneHot(one-hot 33)
    # note deletion VALUE precedes HAS-deletion, the opposite of ours -- and
    # verified against chai's own input_projs.MSA output at corr 0.99999711
    # (tools/oracles/chai1/cmp_msa_stream.py).
    #
    # The one-hot stays in OUR 32-class vocabulary here and the converter
    # permutes chai's 33 weight columns onto it, so the vocab mapping lives in
    # one place (converters.chai1.CHAI1_RESTYPE_PERM) instead of two.
    #
    # Two features we cannot reproduce faithfully:
    #  * IsPairedMSA marks rows paired ACROSS CHAINS. Derived here rather than
    #    carried: a row is paired exactly when it covers tokens of more than one
    #    chain, which is what pairing means. chai's own single-chain run has it 0
    #    throughout, and so does this for a monomer -- but a COMPLEX with a
    #    paired alignment needs it, and hardcoding 0 silently threw that signal
    #    away.
    #  * MSADataSource labels which database a row came from. chai's run uses
    #    4 for the query row and mostly 2 for the rest, so that is what we
    #    emit. Rows from a third source (0) are not distinguishable to us.
    n_row = msa_1hot.shape[0]
    if asym_id is not None:
      n_ch = 16          # bound: more chains than any complex this runs on
      oh = jax.nn.one_hot(jnp.clip(asym_id, 0, n_ch - 1), n_ch)
      covers = jnp.einsum('rt,tc->rc', msa.mask.astype(jnp.float32), oh) > 0
      row_paired = (covers.sum(-1) > 1).astype(msa_1hot.dtype)
      is_paired = jnp.broadcast_to(row_paired[:, None, None],
                                  msa_1hot.shape[:2] + (1,))
    else:
      is_paired = jnp.zeros(msa_1hot.shape[:2] + (1,), msa_1hot.dtype)
    src = jnp.where(jnp.arange(n_row) == 0, 4, 2)
    src = jax.nn.one_hot(src, 6, dtype=msa_1hot.dtype)
    src = jnp.broadcast_to(src[:, None, :], msa_1hot.shape[:2] + (6,))
    return jnp.concatenate(
        [is_paired, src, deletion_value, has_deletion, msa_1hot], axis=-1)

  return jnp.concatenate(msa_feat, axis=-1)


def truncate_msa_batch(msa: features.MSA, num_msa: int) -> features.MSA:
  indices = jnp.arange(num_msa)
  return msa.index_msa_rows(indices)


def create_target_feat(
    batch: feat_batch.Batch,
    append_per_atom_features: bool,
    soft_seq=None,
    design_mask=None,
) -> jax.Array:
  """Make target feat."""
  token_features = batch.token_features
  target_features = []
  target_features.append(
      blend_soft(
          jax.nn.one_hot(
              token_features.aatype,
              residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
          ),
          soft_seq,
          design_mask,
      )
  )
  # ColabDesign2: AF3 builds the profile ONCE from the placeholder sequence, so
  # left alone ('frozen') it is a hard one-hot of poly-alanine while soft_seq
  # changes the aatype one-hot and msa_1hot underneath it -- a confidently WRONG
  # signal, the direct analogue of AF2's pssm_hard problem (a mismatched profile
  # broke multimer design outright). PROFILE_MODE (see its definition above) is
  # the profile-feature twin of AF2's pssm_hard: 'soft' == pssm_hard=False,
  # 'hard' == pssm_hard=True, 'frozen' = ignore the sequence, 'zero' = no info.
  profile = batch.msa.profile
  if PROFILE_MODE == 'soft':
    profile = blend_soft(profile, soft_seq, design_mask)
  elif PROFILE_MODE == 'hard':
    profile = blend_soft(profile, hard_seq(soft_seq), design_mask)
  elif PROFILE_MODE == 'zero':
    profile = jnp.zeros_like(profile)
  elif PROFILE_MODE in ('unk', 'gap') and soft_seq is not None:
    idx = _PROFILE_UNK_INDEX if PROFILE_MODE == 'unk' else _PROFILE_GAP_INDEX
    profile = token_profile(profile, idx, design_mask)
  elif PROFILE_MODE not in ('frozen', 'unk', 'gap'):
    raise ValueError("PROFILE_MODE must be 'soft'/'hard'/'frozen'/'unk'/'gap'/"
                     f"'zero', got {PROFILE_MODE!r}")
  target_features.append(profile)
  target_features.append(batch.msa.deletion_mean[..., None])

  # Reference structure features
  if append_per_atom_features:
    ref_mask = batch.ref_structure.mask
    element_feat = jax.nn.one_hot(batch.ref_structure.element, 128)
    element_feat = utils.mask_mean(
        mask=ref_mask[..., None], value=element_feat, axis=-2, eps=1e-6  # pyrefly: ignore[bad-argument-type]
    )
    target_features.append(element_feat)
    pos_feat = batch.ref_structure.positions
    pos_feat = pos_feat.reshape([pos_feat.shape[0], -1])
    target_features.append(pos_feat)
    target_features.append(ref_mask)

  return jnp.concatenate(target_features, axis=-1)


def create_relative_encoding(
    seq_features: features.TokenFeatures,
    max_relative_idx: int,
    max_relative_chain: int,
    chain_bucket_on_same_chain: bool = False,
) -> jax.Array:
  """Add relative position encodings.

  `chain_bucket_on_same_chain` selects ESMFold2's convention for the relative-
  chain block, which differs from AF3's in BOTH the predicate and the branch:
  AF3 keys it on same-ENTITY and sends the mismatch to the 2*c+1 bucket, while
  ESMFold2 keys it on same-CHAIN and sends the MATCH there. On a monomer that is
  not a subtle difference -- every pair takes bucket 2*c+1 instead of bucket c --
  so a single wrong column of the embedding is added everywhere.

  Honours seq_features.cyclic_period when present: the residue offset is wrapped
  modulo each chain's period before clipping, so a cyclic peptide's termini read
  as adjacent. Tokens with period 0 fall back to 10000, which is a no-op for any
  realistic length -- so this is exactly identity when nothing is cyclic.
  """
  rel_feats = []
  token_index = seq_features.token_index
  residue_index = seq_features.residue_index
  asym_id = seq_features.asym_id
  entity_id = seq_features.entity_id
  sym_id = seq_features.sym_id

  left_asym_id = asym_id[:, None]
  right_asym_id = asym_id[None, :]

  left_residue_index = residue_index[:, None]
  right_residue_index = residue_index[None, :]

  left_token_index = token_index[:, None]
  right_token_index = token_index[None, :]

  left_entity_id = entity_id[:, None]
  right_entity_id = entity_id[None, :]

  left_sym_id = sym_id[:, None]
  right_sym_id = sym_id[None, :]

  # Embed relative positions using a one-hot embedding of distance along chain
  offset = left_residue_index - right_residue_index
  if seq_features.cyclic_period is not None:
    # Boltz's RelativePositionEncoder: period broadcasts over the LAST axis, so
    # the pair (i, j) uses token j's period -- keep that indexing, it is what the
    # boltz2 checkpoint was trained with.
    period = jnp.where(seq_features.cyclic_period > 0,
                       seq_features.cyclic_period,
                       jnp.full_like(seq_features.cyclic_period, 10000))
    offset = offset - period[None, :] * jnp.round(offset / period[None, :])
    offset = offset.astype(left_residue_index.dtype)
  clipped_offset = jnp.clip(
      offset + max_relative_idx, min=0, max=2 * max_relative_idx
  )
  asym_id_same = left_asym_id == right_asym_id
  final_offset = jnp.where(
      asym_id_same,
      clipped_offset,
      (2 * max_relative_idx + 1) * jnp.ones_like(clipped_offset),
  )
  rel_pos = jax.nn.one_hot(final_offset, 2 * max_relative_idx + 2)
  rel_feats.append(rel_pos)

  # Embed relative token index as a one-hot embedding of distance along residue
  token_offset = left_token_index - right_token_index
  clipped_token_offset = jnp.clip(
      token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
  )
  residue_same = (left_asym_id == right_asym_id) & (
      left_residue_index == right_residue_index
  )
  final_token_offset = jnp.where(
      residue_same,
      clipped_token_offset,
      (2 * max_relative_idx + 1) * jnp.ones_like(clipped_token_offset),
  )
  rel_token = jax.nn.one_hot(final_token_offset, 2 * max_relative_idx + 2)
  rel_feats.append(rel_token)

  # Embed same entity ID
  entity_id_same = left_entity_id == right_entity_id
  rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])

  # Embed relative chain ID inside each symmetry class
  rel_sym_id = left_sym_id - right_sym_id

  max_rel_chain = max_relative_chain

  clipped_rel_chain = jnp.clip(
      rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
  )

  special = (2 * max_rel_chain + 1) * jnp.ones_like(clipped_rel_chain)
  if chain_bucket_on_same_chain:
    final_rel_chain = jnp.where(asym_id_same, special, clipped_rel_chain)
  else:
    final_rel_chain = jnp.where(entity_id_same, clipped_rel_chain, special)
  rel_chain = jax.nn.one_hot(final_rel_chain, 2 * max_relative_chain + 2)

  rel_feats.append(rel_chain)

  return jnp.concatenate(rel_feats, axis=-1)


def shuffle_msa(
    key: jax.Array, msa: features.MSA
) -> tuple[features.MSA, jax.Array]:
  """Shuffle MSA randomly, return batch with shuffled MSA.

  Args:
    key: rng key for random number generation.
    msa: MSA object to sample msa from.

  Returns:
    Protein with sampled msa.
  """
  key, sample_key = jax.random.split(key)
  # Sample uniformly among sequences with at least one non-masked position.
  logits = (jnp.clip(jnp.sum(msa.mask, axis=-1), 0.0, 1.0) - 1.0) * 1e6
  index_order = gumbel_argsort_sample_idx(sample_key, logits)

  return msa.index_msa_rows(index_order), key
