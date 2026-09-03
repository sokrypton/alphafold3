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

"""Distogram head."""

from typing import Final

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
import haiku as hk
import jax
import jax.numpy as jnp


_CONTACT_THRESHOLD: Final[float] = 8.0
_CONTACT_EPSILON: Final[float] = 1e-3


class DistogramHead(hk.Module):
  """Distogram head."""

  class Config(base_config.BaseConfig):
    first_break: float = 2.3125
    last_break: float = 21.6875
    num_bins: int = 64

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='distogram_head',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      return_distogram: bool = False,
  ) -> dict[str, jnp.ndarray]:
    pair_act = embeddings['pair']
    seq_mask = batch.token_features.mask.astype(bool)
    pair_mask = seq_mask[:, None] * seq_mask[None, :]

    if self.global_config.model == 'chai1':
      # chai has no distogram head of its own -- its trunk returns only
      # (single, pair). This one was trained post-hoc on the frozen trunk
      # (sokrypton/chai-lab@dgram) and is an MLP, not AF3's single linear:
      # LayerNorm -> 2*c_z -> GELU -> num_bins. It also symmetrises with the
      # MEAN rather than AF3's sum, which is not a rescaling once the softmax
      # sees it. The GELU is torch's exact erf form, so approximate=False --
      # jax defaults to the tanh approximation and the difference is silent.
      hidden = jax.nn.gelu(
          hm.Linear(2 * pair_act.shape[-1], initializer='linear', use_bias=True,
                    name='hidden')(
                        hm.LayerNorm(name='input_layer_norm')(pair_act)),
          approximate=False)
      half_logits = hm.Linear(
          self.config.num_bins, initializer=self.global_config.final_init,
          use_bias=True, name='half_logits')(hidden)
      logits = (half_logits + jnp.swapaxes(half_logits, -2, -3)) / 2
    else:
      left_half_logits = hm.Linear(
          self.config.num_bins,
          initializer=self.global_config.final_init,
          # Four ported families train a bias here and stock AF3 does not; see
          # model_config.DISTOGRAM_BIAS, including which of them need the
          # converter to halve it because their native symmetrises first.
          use_bias=self.global_config.model in model_config.DISTOGRAM_BIAS,
          name='half_logits',
      )(pair_act)

      right_half_logits = left_half_logits
      logits = left_half_logits + jnp.swapaxes(right_half_logits, -2, -3)
    probs = jax.nn.softmax(logits, axis=-1)
    breaks = jnp.linspace(
        self.config.first_break,
        self.config.last_break,
        self.config.num_bins - 1,
    )

    bin_tops = jnp.append(breaks, breaks[-1] + (breaks[-1] - breaks[-2]))
    threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
    is_contact_bin = 1.0 * (bin_tops <= threshold)
    contact_probs = jnp.einsum(
        'ijk,k->ij', probs, is_contact_bin, precision=jax.lax.Precision.HIGHEST
    )
    contact_probs = pair_mask * contact_probs

    return_dict = {'bin_edges': breaks, 'contact_probs': contact_probs}
    if return_distogram:
      return_dict['distogram'] = logits

    return return_dict  # pyrefly: ignore[bad-return]
