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

"""Diffusion Head."""

from collections.abc import Callable
import os

from alphafold3.common import base_config
from alphafold3.constants import residue_names
from alphafold3.model import feat_batch
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from . import atom_cross_attention
from . import diffusion_transformer
from . import featurization
from . import noise_level_embeddings
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


def random_rotation(key):
  # Create a random rotation (Gram-Schmidt orthogonalization of two
  # random normal vectors)
  v0, v1 = jax.random.normal(key, shape=(2, 3))
  e0 = v0 / jnp.maximum(1e-10, jnp.linalg.norm(v0))
  v1 = v1 - e0 * jnp.dot(v1, e0, precision=jax.lax.Precision.HIGHEST)
  e1 = v1 / jnp.maximum(1e-10, jnp.linalg.norm(v1))
  e2 = jnp.cross(e0, e1)
  return jnp.stack([e0, e1, e2])


def random_augmentation(
    rng_key: jnp.ndarray,
    positions: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """Apply random rigid augmentation.

  Args:
    rng_key: random key
    positions: atom positions of shape (<common_axes>, 3)
    mask: per-atom mask of shape (<common_axes>,)

  Returns:
    Transformed positions with the same shape as input positions.
  """
  rotation_key, translation_key = jax.random.split(rng_key)

  center = utils.mask_mean(
      mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6
  )
  rot = random_rotation(rotation_key)
  translation = jax.random.normal(translation_key, shape=(3,))

  augmented_positions = (
      jnp.einsum(
          '...i,ij->...j',
          positions - center,
          rot,
          precision=jax.lax.Precision.HIGHEST,
      )
      + translation
  )
  return augmented_positions * mask[..., None]


# chai's DiffusionConfig, in sigma_data units (chai1.py:263)
CHAI_S_CHURN = 80.0
# NOT scaled by sigma_data, and that is not a typo on our side. chai's own loop
# compares the SCALED sigmas (its schedule multiplies by sigma_data, so they run
# 1262 -> 0.007) against the RAW DiffusionConfig thresholds S_tmin=4e-4 and
# S_tmax=80.0. The mixed units look like a slip in chai, but the weights were
# sampled with it, so it is the specification: churn is OFF above sigma=80 --
# 121 of 200 steps, not all 200. Scaling these by sigma_data turns the window on
# for the entire trajectory and injects 81-1236 A of extra noise across the first
# 79 steps, which is what left the samples as partially-denoised blobs (N-CA
# 0.70 A against an ideal 1.46).
CHAI_S_TMIN = 4e-4
CHAI_S_TMAX = 80.0


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
  return (
      SIGMA_DATA
      * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
  )


class ConditioningConfig(base_config.BaseConfig):
  pair_channel: int
  seq_channel: int
  prob: float


class SampleConfig(base_config.BaseConfig):
  steps: int
  gamma_0: float = 0.8
  gamma_min: float = 1.0
  noise_scale: float = 1.003
  step_scale: float = 1.5
  num_samples: int = 1
  # EDM schedule shape. AF3 hardcoded these as noise_schedule()'s defaults, which
  # silently applied AF3's sampler to every ported family; they are config fields
  # so each model can carry the constants it was trained with (boltz2 wants rho 8,
  # not 7). See runner._SAMPLER_CONSTANTS.
  sigma_min: float = 0.0004
  sigma_max: float = 160.0
  rho: float = 7.0
  # ESMFold2 CLIPS the schedule: it drops every sigma above max_inference_sigma
  # and starts from that value instead. With sigma_data 16 and smax 160 the EDM
  # schedule opens at 2560, so a clip at 256 is a tenfold cut in initial noise
  # -- not a detail, and there is no field for it in stock AF3. 0 = no clip.
  max_sigma: float = 0.0


class DiffusionHead(hk.Module):
  """Denoising Diffusion Head."""

  class Config(
      atom_cross_attention.AtomCrossAttEncoderConfig,
      atom_cross_attention.AtomCrossAttDecoderConfig,
  ):
    """Configuration for DiffusionHead."""

    eval_batch_size: int = 5
    eval_batch_dim_shard_size: int = 5
    conditioning: ConditioningConfig = base_config.autocreate(
        prob=0.8, pair_channel=128, seq_channel=384
    )
    eval: SampleConfig = base_config.autocreate(
        num_samples=5,
        steps=200,
    )
    transformer: diffusion_transformer.Transformer.Config = (
        base_config.autocreate()
    )

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='diffusion_head',
  ):
    self.config = config
    self.global_config = global_config
    super().__init__(name=name)

  @hk.transparent
  def _conditioning(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      noise_level: jnp.ndarray,
      use_conditioning: bool,
      parts: str = 'both',
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`parts` selects which half to build; the other is returned as None.

    The PAIR half depends only on the trunk embeddings and the batch -- nothing
    noise- or sample-dependent reaches it, the noise embedding is added to the
    SINGLE half further down. It was nonetheless built inside the denoiser,
    which is vmapped over samples and scanned over steps, so a default fold
    rebuilt it ~1000 times: a LayerNorm, a projection and two transition blocks
    over (num_tokens, num_tokens, pair_channel), plus the (L, L, 139) relative
    encoding feeding them. `sample` now builds it once and passes it in.

    Kept as ONE method with a selector rather than split in two: this is
    @hk.transparent, so its parameters are named against the enclosing module,
    and calling the same method twice keeps every name exactly where it was.
    """
    want_pair = parts in ('both', 'pair')
    want_single = parts in ('both', 'single')
    single_embedding = use_conditioning * embeddings['single']
    pair_embedding = use_conditioning * embeddings['pair']

    if want_pair:
      rel_features = featurization.create_relative_encoding(
          seq_features=batch.token_features,
          max_relative_idx=32,
          max_relative_chain=2,
      ).astype(pair_embedding.dtype)
      pc = self.config.conditioning.pair_channel
      if self.global_config.model == 'opendde':
        # OpenDDE compresses the trunk pair and the rel-pos features SEPARATELY to
        # pair_channel, concatenates (2*pc), then LN+projects -- rather than one joint
        # projection over [pair_embedding, RAW rel_features]. The joint LN over the
        # widened concat couples them, so this is a distinct forward path (new params:
        # z_trunk_norm/projection + relpe_projection + the LN/projection at 2*pc).
        z_trunk = hm.Linear(pc, precision='highest', name='z_trunk_projection')(
            hm.LayerNorm(use_fast_variance=False, create_offset=model_config.affine_norm(self.global_config.model,
                                                               'z_trunk_norm'),
                         name='z_trunk_norm')(pair_embedding))
        relpe = hm.Linear(pc, precision='highest', name='relpe_projection')(rel_features)
        features_2d = jnp.concatenate([z_trunk, relpe], axis=-1)
      elif self.global_config.model in model_config.DIFFUSION_PROJECTED_RELPOS:
        # Boltz's pairwise_conditioner concats [RAW z_trunk(128), PROJECTED relpos(128)] -> 256
        # (vs AF3's [z_trunk, RAW rel_features(139)] -> 267). relpe_projection == the trunk's
        # rel_pos.linear_layer (139->128), the same weight used for z-init position_activations.
        # Protenix rides the SAME path at c_z=256 ([z_trunk(256), relpe(256)] -> 512), but with
        # its OWN relpe weight (DiffusionConditioning.relpe, distinct from the trunk's) and
        # create_offset=False on the norms (the boltz2-gated offsets below stay boltz2-only).
        relpe = hm.Linear(pc, precision='highest', name='relpe_projection')(rel_features)
        features_2d = jnp.concatenate([pair_embedding, relpe], axis=-1)
      elif self.global_config.model == 'chai1':
        # chai conditions on [z_trunk, z_init] -- the token embedder's pair output,
        # not a relative-position encoding at all (its relative features are
        # already inside z_init). Trunk first, matching AF3's order here; note the
        # SINGLE track concatenates the other way round, which is why the
        # converter swaps that weight's halves and not this one.
        features_2d = jnp.concatenate(
            [pair_embedding, embeddings['pair_init'].astype(pair_embedding.dtype)],
            axis=-1)
      else:
        features_2d = jnp.concatenate([pair_embedding, rel_features], axis=-1)
      pair_cond = hm.Linear(
          pc,
          precision='highest',
          name='pair_cond_initial_projection',
      )(
          hm.LayerNorm(
              use_fast_variance=False,
              # chai is AFFINE here (token_pair_proj.0 / token_in_proj.0 /
              # fourier_proj.0 all carry a bias). Found by enumerating every
              # affine LayerNorm in the checkpoint and diffing against the
              # converter's scale-only scopes: three showed up, all three real.
              create_offset=model_config.affine_norm(
                  self.global_config.model, 'pair_cond_initial_norm'),
              name='pair_cond_initial_norm',
          )(features_2d)
      )

      for idx in range(2):
        pair_cond += diffusion_transformer.transition_block(
            pair_cond, 2, self.global_config, name=f'pair_transition_{idx}'
        )
      if self.global_config.model == 'chai1':
        # chai closes each track with an affine LayerNorm; the single one is
        # applied with the single half below.
        pair_cond = hm.LayerNorm(use_fast_variance=False,
                                 name='pair_cond_final_norm')(pair_cond)
    else:
      pair_cond = None

    if not want_single:
      return None, pair_cond

    target_feat = embeddings['target_feat']
    if self.global_config.model in model_config.PAIR_ONLY_TRUNK:
      # No single track exists, so there is no single_embedding to concatenate:
      # ESMFold2 conditions on s_inputs alone (451 channels, not 384 + 451).
      features_1d = target_feat
    else:
      features_1d = jnp.concatenate([single_embedding, target_feat], axis=-1)
    if self.global_config.model in ('openfold3', 'openbind0'):
      # BOTH OpenFold3 releases: openbind changed the diffusion transformer's
      # pair LayerNorm, not the feature layout, so it keeps this 833-channel
      # convention. A new model in this lineage has to be added here explicitly;
      # the shape manifest is what catches the omission (833 vs 831), which is
      # exactly how openbind was caught.
      #
      # OF3's restype and profile blocks carry 32 classes; AF3's carry 31 (AF3
      # folds unknown DNA into the shared unknown-nucleic class). Everywhere
      # else the extra class can simply be dropped from the weights, because a
      # zero input column contributes nothing to a bare Linear. Here it cannot:
      # the LayerNorm below maps a zero input to -mean/std, so OF3 always adds a
      # trained contribution through those two columns and normalises over 833
      # channels instead of 831. Re-insert them as zeros to match exactly; the
      # weight converter emits its rows in this same order.
      num_af3_restypes = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
      pad = jnp.zeros_like(features_1d[..., :1])
      single_channels = single_embedding.shape[-1]
      aatype_end = single_channels + num_af3_restypes
      profile_end = aatype_end + num_af3_restypes
      features_1d = jnp.concatenate(
          [
              features_1d[..., :aatype_end],
              pad,  # OF3 aatype UNK_DNA
              features_1d[..., aatype_end:profile_end],
              pad,  # OF3 profile UNK_DNA
              features_1d[..., profile_end:],
          ],
          axis=-1,
      )
    if self.global_config.model in model_config.PAIR_ONLY_TRUNK:
      # ESMFold2's s_inputs carries 33-class restype and profile blocks where
      # AF3's carry 31 (ESM reserves two classes AF3 has no input for). Four
      # dead columns -- and everywhere else in the port they are simply dropped
      # from the weights, because a zero input column contributes nothing to a
      # bias-free Linear. Here they cannot be: the LayerNorm below divides by
      # the width and subtracts the mean over it, so normalising 447 channels
      # instead of 451 rescales the ENTIRE diffusion conditioning. Same trap as
      # OpenFold3's UNK_DNA columns above. The converter's weight rows are in
      # this padded order to match.
      n = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
      pad = jnp.zeros_like(features_1d[..., :2])
      features_1d = jnp.concatenate(
          [pad, features_1d[..., :n], pad, features_1d[..., n:]], axis=-1)
    single_cond = hm.LayerNorm(
        use_fast_variance=False,
        # chai is AFFINE here (token_pair_proj.0 / token_in_proj.0 /
        # fourier_proj.0 all carry a bias). Found by enumerating every
        # affine LayerNorm in the checkpoint and diffing against the
        # converter's scale-only scopes: three showed up, all three real.
        create_offset=model_config.affine_norm(
            self.global_config.model, 'single_cond_initial_norm'),
        name='single_cond_initial_norm',
    )(features_1d)
    single_cond = hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        # boltz2's single_conditioner.single_embed is a plain `nn.Linear`, so it
        # carries a bias where stock AF3's projection does not
        # (encodersv2.py:142, `s = self.single_embed(self.norm_single(s))`).
        # This one is NOT inert the way an attention-bias offset is: it is a
        # constant added to the single conditioning of every token, and it
        # reaches the whole diffusion module through adaLN.
        use_bias=self.global_config.model == 'boltz2',
        name='single_cond_initial_projection',
    )(single_cond)

    if getattr(self.global_config, 'trained_fourier', False):
      # Models with an independently-trained Fourier embedding (OpenFold3,
      # IntelliFold-v2) differ from AF3's hardcoded constants. Load them as proper
      # Haiku parameters so they travel with the params file for BOTH such models
      # -- the same mechanism, gated by trained_fourier (true for of3 and if2),
      # rather than a per-model runtime monkey-patch. Stock AF3 leaves this off
      # and uses the constants in noise_level_embeddings.
      _dim = len(noise_level_embeddings._WEIGHT)
      fourier_weight = hk.get_parameter(
          'fourier_embedding_weight',
          shape=[_dim],
          dtype=jnp.float32,
          init=hk.initializers.Constant(0.0),
      )
      fourier_bias = hk.get_parameter(
          'fourier_embedding_bias',
          shape=[_dim],
          dtype=jnp.float32,
          init=hk.initializers.Constant(0.0),
      )
      # Every model is fed the sigma_data-scaled noise level here. chai does not
      # divide by sigma_data itself, but the difference is a constant shift of
      # 0.25*log(16) inside the log, and the embedding is
      # cos(2pi(0.25*log(s)*w + b)) -- so its converter folds w*0.25*log(16)
      # into the BIAS and the two are the same function to float round-off.
      noise_embedding = noise_level_embeddings.noise_embeddings(
          sigma_scaled_noise_level=noise_level / SIGMA_DATA,
          weight=fourier_weight,
          bias=fourier_bias,
      )
    else:
      noise_embedding = noise_level_embeddings.noise_embeddings(
          sigma_scaled_noise_level=noise_level / SIGMA_DATA
      )
    single_cond += hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        name='noise_embedding_initial_projection',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            # chai is AFFINE here (token_pair_proj.0 / token_in_proj.0 /
            # fourier_proj.0 all carry a bias). Found by enumerating every
            # affine LayerNorm in the checkpoint and diffing against the
            # converter's scale-only scopes: three showed up, all three real.
            create_offset=model_config.affine_norm(
                self.global_config.model, 'noise_embedding_initial_norm'),
            name='noise_embedding_initial_norm',
        )(noise_embedding)
    )

    for idx in range(2):
      single_cond += diffusion_transformer.transition_block(
          single_cond, 2, self.global_config, name=f'single_transition_{idx}'
      )

    if self.global_config.model == 'chai1' and want_single:
      # chai closes its conditioning with an AFFINE LayerNorm on each track --
      # `single_ln` and `pair_ln` -- which AF3 does not have. Both feed every
      # adaLN downstream, and those scale by (s + 1), so leaving them out let
      # the 16-block token transformer run away: measured 9.2e7 against chai's
      # 160 at its output. Same failure the atom conditioning had.
      single_cond = hm.LayerNorm(use_fast_variance=False,
                                 name='single_cond_final_norm')(single_cond)

    return single_cond, pair_cond

  def __call__(
      self,
      # positions_noisy.shape: (num_token, max_atoms_per_token, 3)
      positions_noisy: jnp.ndarray,
      noise_level: jnp.ndarray,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      use_conditioning: bool,
      pair_cond: jnp.ndarray | None = None,
      atom_cond: tuple | None = None,
      conditioning_only: bool = False,
  ) -> jnp.ndarray:

    with utils.bfloat16_context():
      if conditioning_only:
        # Build and return the pair conditioning, nothing else. It has to be
        # reached through __call__ rather than by calling _conditioning from
        # outside: _conditioning is @hk.transparent, so its parameters attach to
        # whatever module is on the stack, and calling it from the enclosing
        # Model put them at `diffuser/pair_cond_initial_norm` instead of
        # `diffuser/~/diffusion_head/pair_cond_initial_norm`.
        _, only_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
            parts='pair',
        )
        # The atom encoder's conditioning is position-independent too, and it
        # is the larger tensor: the atom pair block is
        # (num_subsets, num_queries, num_keys, channels).
        atom_cond = atom_cross_attention.atom_cross_att_encoder(
            token_atoms_act=None,
            trunk_single_cond=embeddings['single'],
            trunk_pair_cond=only_pair_cond,
            config=self.config,
            global_config=self.global_config,
            batch=batch,
            name='diffusion',
            conditioning_only=True,
        )
        return only_pair_cond, atom_cond

      # Get conditioning. The pair half is noise- and sample-independent, so
      # `sample` builds it once and hands it in; only the single half, which
      # carries the noise embedding, is rebuilt per step.
      trunk_single_cond, built_pair_cond = self._conditioning(
          batch=batch,
          embeddings=embeddings,
          noise_level=noise_level,
          use_conditioning=use_conditioning,
          parts='single' if pair_cond is not None else 'both',
      )
      trunk_pair_cond = pair_cond if pair_cond is not None else built_pair_cond

      # Extract features
      sequence_mask = batch.token_features.mask
      atom_mask = batch.predicted_structure_info.atom_mask

      # Position features
      act = positions_noisy * atom_mask[..., None]
      act = act / jnp.sqrt(noise_level**2 + SIGMA_DATA**2)

      enc = atom_cross_attention.atom_cross_att_encoder(
          token_atoms_act=act,
          trunk_single_cond=embeddings['single'],
          trunk_pair_cond=trunk_pair_cond,
          config=self.config,
          global_config=self.global_config,
          batch=batch,
          name='diffusion',
          cond=atom_cond,
      )
      act = enc.token_act

      # Token-token attention
      act = jnp.asarray(act, dtype=jnp.float32)

      # chai adds structure_cond_to_token_structure_proj(s_cond) with NO
      # LayerNorm -- its s_cond has already been through `single_ln` at the end
      # of the conditioning. AF3 normalises again here, and an unmapped
      # LayerNorm is not a no-op even at scale=1: it still re-centres and
      # re-scales.
      _s_cond_in = trunk_single_cond
      if self.global_config.model != 'chai1':
        _s_cond_in = hm.LayerNorm(
            use_fast_variance=False,
            create_offset=model_config.affine_norm(
                self.global_config.model, 'single_cond_embedding_norm'),
            name='single_cond_embedding_norm',
        )(trunk_single_cond)
      act += hm.Linear(
          act.shape[-1],
          precision='highest',
          initializer=self.global_config.final_init,
          name='single_cond_embedding_projection',
      )(_s_cond_in)

      act = jnp.asarray(act, dtype=jnp.float32)
      trunk_single_cond = jnp.asarray(trunk_single_cond, dtype=jnp.float32)
      trunk_pair_cond = jnp.asarray(trunk_pair_cond, dtype=jnp.float32)
      sequence_mask = jnp.asarray(sequence_mask, dtype=jnp.float32)

      transformer = diffusion_transformer.Transformer(
          self.config.transformer, self.global_config
      )
      act = transformer(
          act=act,
          single_cond=trunk_single_cond,
          mask=sequence_mask,
          pair_cond=trunk_pair_cond,
          # OpenDDE threads the structural-token pair attention bias (from the
          # token expander) into the diffusion transformer too; None otherwise.
          extra_pair_bias=embeddings.get('structural_pair_attn_bias'),
      )
      act = hm.LayerNorm(
          use_fast_variance=False,
          create_offset=model_config.affine_norm(
              self.global_config.model, 'output_norm'),
          name='output_norm'
      )(act)
      # (n_tokens, per_token_channels)

      # (Possibly) atom-granularity decoder
      assert isinstance(enc, atom_cross_attention.AtomCrossAttEncoderOutput)
      position_update = atom_cross_attention.atom_cross_att_decoder(
          token_act=act,
          enc=enc,
          config=self.config,
          global_config=self.global_config,
          batch=batch,
          name='diffusion',
      )

      skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
      out_scaling = (
          noise_level * SIGMA_DATA / jnp.sqrt(noise_level**2 + SIGMA_DATA**2)
      )
    # End `with utils.bfloat16_context()`.

    return (
        skip_scaling * positions_noisy + out_scaling * position_update
    ) * atom_mask[..., None]


def _kabsch(mob, ref, w):
  """Weighted rigid align of `mob` onto `ref`, over the dense atom layout."""
  wm = w[..., None].astype(mob.dtype)
  mc = (mob * wm).sum((-3, -2), keepdims=True) / wm.sum((-3, -2), keepdims=True)
  rc = (ref * wm).sum((-3, -2), keepdims=True) / wm.sum((-3, -2), keepdims=True)
  a, b = (mob - mc).reshape(-1, 3), (ref - rc).reshape(-1, 3)
  u, _, vt = jnp.linalg.svd((a * wm.reshape(-1, 1)).T @ b)
  d = jnp.sign(jnp.linalg.det(u @ vt))
  rot = u @ jnp.diag(jnp.array([1.0, 1.0, d], dtype=mob.dtype)) @ vt
  return ((mob - mc).reshape(-1, 3) @ rot).reshape(mob.shape) + rc


def sample(
    denoising_step: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    batch: feat_batch.Batch,
    key: jnp.ndarray,
    config: SampleConfig,
    global_config: model_config.GlobalConfig | None = None,
) -> dict[str, jnp.ndarray]:
  """Sample using denoiser on batch.

  Args:
    denoising_step: the denoising function.
    batch: the batch
    key: random key
    config: config for the sampling process (e.g. number of denoising steps,
      etc.)

  Returns:
    a dict
      {
         'atom_positions': jnp.array(...)       # shape (<common_axes>, 3)
         'mask': jnp.array(...)                 # shape (<common_axes>,)
      }
    where the <common_axes> are
    (num_samples, num_tokens, max_atoms_per_token)
  """

  mask = batch.predicted_structure_info.atom_mask
  chai = global_config is not None and global_config.model == 'chai1'

  def apply_denoising_step(carry, noise_level):
    key, positions, noise_level_prev = carry
    key, key_noise, key_aug = jax.random.split(key, 3)

    positions = random_augmentation(
        rng_key=key_aug, positions=positions, mask=mask  # pyrefly: ignore[bad-argument-type]
    )

    # chai's churn is EDM's own parameterisation, not AF3's: a constant
    # min(S_churn / N, sqrt(2) - 1) applied wherever S_tmin <= sigma <= S_tmax,
    # keyed on the CURRENT sigma. AF3 instead switches a fixed gamma_0 on above
    # a threshold.
    if chai:
      gamma = jnp.where(
          (noise_level_prev >= CHAI_S_TMIN) & (noise_level_prev <= CHAI_S_TMAX),
          min(CHAI_S_CHURN / config.steps, 2.0 ** 0.5 - 1.0), 0.0)
    else:
      gamma = config.gamma_0 * (noise_level > config.gamma_min)
    t_hat = noise_level_prev * (1 + gamma)

    # Guard against -0.0 (IEEE negative zero) when gamma=0: t_hat ==
    # noise_level_prev, so t_hat**2 - noise_level_prev**2 == -0.0 and
    # XLA's sqrt(-0.0) returns NaN. chai clamps at 1e-6 rather than 0, so on a
    # no-churn step it still adds 1e-3 A of noise; match that exactly now that
    # no-churn steps are the majority of the trajectory rather than none of it.
    floor = 1e-6 if chai else 0.0
    noise_scale = config.noise_scale * jnp.sqrt(
        jnp.maximum(floor, t_hat**2 - noise_level_prev**2)
    )
    noise = noise_scale * jax.random.normal(key_noise, positions.shape)
    positions_noisy = positions + noise

    positions_denoised = denoising_step(positions_noisy, t_hat)
    if global_config is not None and global_config.model in model_config.REALIGN_SAMPLER:
      # ESMFold2 rigid-aligns the NOISY coords onto the denoised prediction
      # before taking the Euler step. It augments (rotates and translates) the
      # state every step, and its denoiser is not equivariant -- it reads a
      # fixed reference conformer through rotary position embeddings -- so the
      # prediction comes back in its own frame. Differencing across the two
      # frames makes `grad` a rotation as much as a gradient.
      positions_noisy = _kabsch(positions_noisy, positions_denoised, mask)
    grad = (positions_noisy - positions_denoised) / t_hat

    d_t = noise_level - t_hat
    if chai:
      # chai's second-order step, copied verbatim rather than tidied. Note it
      # ADDS the averaged correction to the already-Euler-updated position
      # instead of replacing the Euler term, so the total move is
      # d_t * grad + d_t * (grad + grad') / 2 -- NOT textbook Heun. The weights
      # were sampled with this, so the deviation is the specification.
      # No step_scale either: chai's is 1.0.
      positions_out = positions_noisy + d_t * grad
      grad2 = (positions_out - denoising_step(positions_out, noise_level)
               ) / noise_level
      positions_out = jnp.where(
          noise_level > 0,
          positions_out + d_t * (grad2 + grad) / 2.0,
          positions_out)
    else:
      positions_out = positions_noisy + config.step_scale * d_t * grad

    return (key, positions_out, noise_level), positions_out

  num_samples = config.num_samples

  if chai:
    # chai evaluates the schedule at MIDPOINTS -- linspace(0, 1, 2N+1)[1::2] --
    # where AF3 uses the N+1 endpoints. So it never samples sigma=0 exactly, and
    # every sigma sits half a step inside AF3's.
    times = np.linspace(0.0, 1.0, 2 * config.steps + 1, dtype=np.float32)[1::2]
  else:
    times = np.linspace(0, 1, config.steps + 1, dtype=np.float32)
  # numpy, not jnp: the schedule is a compile-time constant, and the clip below
  # is a boolean mask that a traced array cannot take. float32 explicitly --
  # numpy would default to float64 here and hand every model a schedule that
  # differs from the traced one in the last bits, which is a silent change to
  # the sampling trajectory of thirteen models that have nothing to do with the
  # clip this was added for.
  noise_levels = np.asarray(noise_schedule(
      times, smin=config.sigma_min, smax=config.sigma_max, p=config.rho),
      dtype=np.float32)
  if getattr(config, 'max_sigma', 0.0):
    noise_levels = np.concatenate(
        [[config.max_sigma], noise_levels[noise_levels <= config.max_sigma]])
  noise_levels = jnp.asarray(noise_levels, jnp.float32)

  key, noise_key = jax.random.split(key)
  positions = jax.random.normal(noise_key, (num_samples,) + mask.shape + (3,))
  positions *= noise_levels[0]

  init = (
      jax.random.split(key, num_samples),
      positions,
      jnp.tile(noise_levels[None, 0], (num_samples,)),
  )

  apply_denoising_step = hk.vmap(
      apply_denoising_step, in_axes=(0, None), split_rng=(not hk.running_init())
  )
  # unroll=1, NOT AF3's 4. Measured on this graph (steps -> total compile):
  #   unroll=4: 1->3.5s  2->4.0s  4->50.3s  8->53.4s  20->52.9s  50->64.1s
  #   unroll=1: 20->40.6s  50->41.9s   (flat -- one body copy, compiled once)
  # and the two are RUNTIME-identical (0.30s vs 0.31s per call at 20 steps), so the
  # unrolling buys no speed and costs ~13s of compile plus ~7s per remainder copy.
  # XLA sees `unroll + (steps % unroll)` copies of the denoiser body, which is why
  # compile tracked the step count in a way that looked inexplicable.
  # (Below `unroll` steps jax's _scan_impl emits no loop at all -- num_trips==1 and
  # remainder==0 -- which is the 3.5s case, not something to design around.)
  result, _ = hk.scan(apply_denoising_step, init, noise_levels[1:], unroll=1)
  _, positions_out, _ = result

  final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))

  return {'atom_positions': positions_out, 'mask': final_dense_atom_mask}
