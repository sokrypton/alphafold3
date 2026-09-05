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

"""Diffusion transformer model."""

import os

from alphafold3.common import base_config
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import haiku_modules as hm
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import tokamax

# Diagnostic tap for the chai1 port (default OFF, same pattern as the diffusion
# head's CHAI_NOCHURN). The atom stack's per-block outputs are the one thing the
# frozen chai graph can be tapped for that we could not otherwise see, so when
# CHAI_TAP_ATOM_BLOCKS=1 the blocks' activations ride out through layer_stack's
# per-layer outputs instead of being discarded.
_TAP_ATOM_BLOCKS = os.environ.get('CHAI_TAP_ATOM_BLOCKS') == '1'
ATOM_BLOCK_TAPS = []


def adaptive_layernorm(x, single_cond, name, global_config=None):
  """Adaptive LayerNorm."""
  # Adopted from Scalable Diffusion Models with Transformers
  # https://arxiv.org/abs/2212.09748
  # chai-1 differs in three ways, all measured off its traced graph:
  #   * it normalises the activation with eps=0.1, not 1e-5 (the same odd eps it
  #     uses in the OPM product norm);
  #   * it does NOT layer-norm the conditioning first, so there is no
  #     `single_cond_layer_norm` to map -- and keeping it with scale=1 would be
  #     wrong, since a LayerNorm still re-centres and re-scales;
  #   * the scale is identity-centred, `(s + 1) * x`, where AF3 uses
  #     `sigmoid(s) * x`. Both start near the identity at init, so this is
  #     invisible until the trained weights are loaded.
  chai = global_config is not None and global_config.model == 'chai1'
  if single_cond is None:
    x = hm.LayerNorm(name=f'{name}layer_norm', use_fast_variance=False)(x)
  else:
    x = hm.LayerNorm(
        name=f'{name}layer_norm',
        use_fast_variance=False,
        create_scale=False,
        create_offset=False,
        eps=0.1 if chai else 1e-5,
    )(x)
    if not chai:
      single_cond = hm.LayerNorm(
          name=f'{name}single_cond_layer_norm',
          use_fast_variance=False,
          create_offset=False,
      )(single_cond)
    single_scale = hm.Linear(
        x.shape[-1],
        initializer='zeros',
        use_bias=not chai,
        name=f'{name}single_cond_scale',
    )(single_cond)
    single_bias = hm.Linear(
        x.shape[-1], initializer='zeros', name=f'{name}single_cond_bias'
    )(single_cond)
    gate = (single_scale + 1.0) if chai else jax.nn.sigmoid(single_scale)
    x = gate * x + single_bias
  return x


def adaptive_zero_init(
    x, num_channels, single_cond, global_config: model_config.GlobalConfig, name,
    project=True,
):
  """Adaptive zero init, from AdaLN-zero.

  project=False drops the output projection and keeps only the conditioning
  gate. chai-1's ATOM transformers multiply the raw concatenated heads by
  sigmoid(out_proj(cond)) and never project them -- its token transformer does
  project (`to_out`), so this is a per-call-site difference, not a model-wide one.
  """
  if not project:
    assert single_cond is not None, 'project=False only makes sense with a gate'
    cond = hm.Linear(
        x.shape[-1],
        initializer='zeros',
        use_bias=True,
        bias_init=-2.0,
        name=f'{name}adaptive_zero_cond',
    )(single_cond)
    return jax.nn.sigmoid(cond) * x
  if single_cond is None:
    output = hm.Linear(
        num_channels,
        initializer=global_config.final_init,
        name=f'{name}transition2',
    )(x)
  else:
    output = hm.Linear(num_channels, name=f'{name}transition2')(x)
    # Init to a small gain, sigmoid(-2) ~ 0.1
    cond = hm.Linear(
        output.shape[-1],
        initializer='zeros',
        use_bias=True,
        bias_init=-2.0,
        name=f'{name}adaptive_zero_cond',
    )(single_cond)
    output = jax.nn.sigmoid(cond) * output
  return output


def transition_block(
    x: jnp.ndarray,
    num_intermediate_factor: int,
    global_config: model_config.GlobalConfig,
    single_cond: jnp.ndarray | None = None,
    use_glu_kernel: bool = True,
    name: str = '',
) -> jnp.ndarray:
  """Transition Block."""
  num_channels = x.shape[-1]
  num_intermediates = num_intermediate_factor * num_channels

  x = adaptive_layernorm(x, single_cond, name=f'{name}ffw_',
                         global_config=global_config)

  # Boltz-2's ConditionedTransitionBlock adds an extra multiplicative up-gate a_to_b(a)
  # on the SwiGLU output before the down-projection: b = SwiGLU(swish_gate(a)) * a_to_b(a).
  # Gated so AF3/OF3/IF2/OpenDDE (which have no a_to_b) are unchanged. Captured from the
  # normed input now because the non-kernel path below overwrites x.
  # Boltz's ConditionedTransitionBlock (adaln-conditioned, single_cond set) has the a_to_b
  # up-gate; its plain Transition (used in the diffusion single/pair conditioners, called
  # with single_cond=None) does NOT. So gate a_to_b on single_cond being present -- exactly
  # the ConditionedTransitionBlock-vs-Transition distinction -- else the conditioner
  # transitions would ask for an a_to_b param the checkpoint doesn't have.
  a_to_b = None
  if global_config.model == 'boltz2' and single_cond is not None:
    a_to_b = hm.Linear(
        num_intermediates, use_bias=False, name=f'{name}ffw_a_to_b'
    )(x)

  if use_glu_kernel:
    weights, _ = hm.haiku_linear_get_params(
        x,
        num_output=num_intermediates * 2,
        initializer='relu',
        name=f'{name}ffw_transition1',
    )
    weights = jnp.reshape(weights, (len(weights), 2, num_intermediates))
    c = tokamax.gated_linear_unit(x=x, weights=weights, activation=jax.nn.swish)
  else:
    x = hm.Linear(
        num_intermediates * 2, initializer='relu', name=f'{name}ffw_transition1'
    )(x)
    a, b = jnp.split(x, 2, axis=-1)
    c = jax.nn.swish(a) * b

  if a_to_b is not None:
    c = c * a_to_b

  output = adaptive_zero_init(
      c, num_channels, single_cond, global_config, f'{name}ffw_'
  )
  return output


def _rms_norm(x):
  """Affine-free RMSNorm. torch's F.rms_norm(eps=None) uses finfo(dtype).eps,
  not the 1e-5 every LayerNorm in this graph uses."""
  eps = float(np.finfo(np.float32).eps)
  return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def _apply_rope(x, cos, sin):
  """Rotary embedding with TILED cos/sin -- [c|c], not interleaved.

  The tiling pairs with rotate_half's split-into-halves. Interleaving instead
  reads corr 0.88 on the atom encoder: high enough to look like noise, low
  enough to ruin the fold.
  """
  ro = cos.shape[-1] * 2
  c = jnp.concatenate([cos, cos], -1)[..., None, :]
  s = jnp.concatenate([sin, sin], -1)[..., None, :]
  a, b = jnp.split(x[..., :ro], 2, axis=-1)
  rot = jnp.concatenate([-b, a], -1)
  return jnp.concatenate([x[..., :ro] * c + rot * s, x[..., ro:]], -1)


class SelfAttentionConfig(base_config.BaseConfig):
  num_head: int = 16
  key_dim: int | None = None
  value_dim: int | None = None


def self_attention(
    x: jnp.ndarray,  # (num_tokens, ch)
    mask: jnp.ndarray,  # (num_tokens,)
    pair_logits: jnp.ndarray | None,  # (num_heads, num_tokens, num_tokens)
    config: SelfAttentionConfig,
    global_config: model_config.GlobalConfig,
    single_cond: jnp.ndarray | None = None,  # (num_tokens, ch)
    name: str = '',
    kq_norm: bool = False,
    gate_bias: float = 0.0,
    use_gating_query: bool = True,
    project_output: bool = True,
    rope: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
  """Multihead self-attention.

  gate_bias is added to the gate logits before the sigmoid. chai-1's pairformer
  single attention gates with sigmoid(g + 1) -- the constant is baked into its
  traced graph, exactly 48 times, one per block. AF3 expresses the same intent
  through bias_init=1.0 on the gating Linear, but that Linear is bias-free, so
  the init never materialises. chai's triangle attentions and its whole
  diffusion module have NO such offset, hence the explicit argument rather than
  a global model branch.
  """
  assert len(mask.shape) == len(x.shape) - 1, f'{mask.shape}, {x.shape}'
  # bias: ... x heads (1) x query (1) x key
  bias = (1e9 * (mask - 1.0))[..., None, None, :]

  x = adaptive_layernorm(x, single_cond, name=name, global_config=global_config)

  num_channels = x.shape[-1]
  # Sensible default for when the config keys are missing
  key_dim = config.key_dim if config.key_dim is not None else num_channels
  value_dim = config.value_dim if config.value_dim is not None else num_channels
  num_head = config.num_head
  assert key_dim % num_head == 0, f'{key_dim=} % {num_head=} != 0'
  assert value_dim % num_head == 0, f'{value_dim=} % {num_head=} != 0'
  key_dim = key_dim // num_head
  value_dim = value_dim // num_head

  qk_shape = (num_head, key_dim)
  q = hm.Linear(qk_shape, use_bias=True, name=f'{name}q_projection')(x)
  k = hm.Linear(qk_shape, use_bias=False, name=f'{name}k_projection')(x)

  # RF3 kq_norm: LayerNorm on q and k over the FLATTENED (num_head*key_dim) axis
  # (not per-head), applied after projection and before the key_dim scaling. Only
  # the diffusion score-model transformers set kq_norm; the trunk/confidence
  # pairformer single-attention (which also calls this fn) leaves it off.
  if kq_norm:
    qk_flat = num_head * key_dim
    q = hm.LayerNorm(name=f'{name}query_layer_norm', use_fast_variance=False)(
        q.reshape(q.shape[:-2] + (qk_flat,))).reshape(q.shape)
    k = hm.LayerNorm(name=f'{name}key_layer_norm', use_fast_variance=False)(
        k.reshape(k.shape[:-2] + (qk_flat,))).reshape(k.shape)

  if rope is not None:
    # ESMFold2's atom attention carries NO pair bias and no learned positional
    # term at all: its entire positional signal is a 3D rotary embedding built
    # from the reference conformer, preceded by an affine-free RMSNorm on q and
    # k. Neither is expressible as a weight, so it enters here -- and only here;
    # the sliding window itself needs nothing new, because it is an additive
    # mask and `bias` already carries one. `rope` is None for every other model.
    q = _apply_rope(_rms_norm(q), *rope)
    k = _apply_rope(_rms_norm(k), *rope)

  # In some situations the gradient norms can blow up without running this
  # einsum in float32.
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  bias = bias.astype(jnp.float32)
  logits = jnp.einsum('...qhc,...khc->...hqk', q * key_dim ** (-0.5), k) + bias
  if pair_logits is not None:
    logits += pair_logits  # (num_heads, seq_len, seq_len)
  weights = jax.nn.softmax(logits, axis=-1)
  weights = jnp.asarray(weights, dtype=x.dtype)

  v_shape = (num_head, value_dim)
  v = hm.Linear(v_shape, use_bias=False, name=f'{name}v_projection')(x)
  weighted_avg = jnp.einsum('...hqk,...khc->...qhc', weights, v)
  weighted_avg = jnp.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))

  # chai-1's diffusion transformers have no gating_query at all -- their only
  # gate is the conditioning one below. Its trunk pairformer DOES gate here
  # (with the +1 offset), so this cannot be a model-wide branch.
  if use_gating_query:
    gate_logits = hm.Linear(
        num_head * value_dim,
        bias_init=1.0,
        initializer='zeros',
        name=f'{name}gating_query',
    )(x)
    weighted_avg *= jax.nn.sigmoid(gate_logits + gate_bias)

  output = adaptive_zero_init(
      weighted_avg, num_channels, single_cond, global_config, name,
      project=project_output,
  )
  return output


class Transformer(hk.Module):
  """Simple transformer stack."""

  class Config(base_config.BaseConfig):
    attention: SelfAttentionConfig = base_config.autocreate()
    num_blocks: int = 24
    block_remat: bool = False
    super_block_size: int = 4
    num_intermediate_factor: int = 2

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name: str = 'transformer',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(
      self,
      act: jnp.ndarray,
      mask: jnp.ndarray,
      single_cond: jnp.ndarray,
      pair_cond: jnp.ndarray | None,
      extra_pair_bias: jnp.ndarray | None = None,
  ) -> jnp.ndarray:
    assert self.config.num_blocks % self.config.super_block_size == 0
    num_super_blocks = self.config.num_blocks // self.config.super_block_size

    # chai-1's token transformer is per-block-pair too (its own affine
    # `pair_layer_norm` + `pair_linear` per block), so it takes this path even
    # though it is deliberately not in OPENFOLD3_LINEAGE -- it is not
    # OpenFold-derived, it just happens to share this one shape.
    chai = self.global_config.model == 'chai1'
    # Keyed on the CONVENTION, not the lineage: openbind is OpenFold3 by descent
    # but runs the pair LayerNorm once, the way AlphaFold 3 does, so it takes the
    # stock path below. See model_config.PER_BLOCK_PAIR_LAYER_NORM.
    per_block_pair = (
        self.global_config.model in model_config.PER_BLOCK_PAIR_LAYER_NORM
        or chai)
    if per_block_pair and pair_cond is not None:
      # OF3 mode: per-block pair LayerNorm + projection. pair_cond is shared
      # across all blocks; each block in the layer_stack gets its own LN/Linear
      # params stacked along axis 0.
      def block(act):  # pylint: disable=function-redefined
        pair_act = hm.LayerNorm(
            name='pair_input_layer_norm',
            use_fast_variance=False,
            # chai's pair_layer_norm is affine on BOTH scale and offset
            create_offset=chai,
        )(pair_cond)
        block_pair_logits = hm.Linear(
            self.config.attention.num_head,
            name='pair_logits_projection',
        )(pair_act)
        block_pair_logits = jnp.transpose(block_pair_logits, [2, 0, 1])
        if extra_pair_bias is not None:
          block_pair_logits = block_pair_logits + extra_pair_bias[None].astype(
              block_pair_logits.dtype)
        attn = self_attention(
            act, mask, block_pair_logits,
            self.config.attention, self.global_config, single_cond,
            name=self.name, kq_norm=self.global_config.model == 'rosettafold3',
            # chai's diffusion token transformer has no gating_query; its only
            # gate is the conditioning one after `to_out`
            use_gating_query=not chai,
        )
        if self.global_config.model in ('rosettafold3', 'chai1'):
          # RF3 no_residual_connection_between_attention_and_transition, and
          # chai's parallel block: both hand the transition the PRE-attention
          # act, and attn + transition share one residual add.
          act = act + attn + transition_block(
              act, self.config.num_intermediate_factor,
              self.global_config, single_cond, name=self.name)
        else:
          act += attn
          act += transition_block(
              act, self.config.num_intermediate_factor,
              self.global_config, single_cond, name=self.name,
          )
        return act

      def super_block(act):  # pylint: disable=function-redefined
        return hk.experimental.layer_stack(self.config.super_block_size)(block)(act)

      return hk.experimental.layer_stack(num_super_blocks)(super_block)(act)

    # Original AF3 mode: single shared pair LayerNorm precomputed before all blocks.
    def block(act, pair_logits):
      act += self_attention(
          act,
          mask,
          pair_logits,
          self.config.attention,
          self.global_config,
          single_cond,
          name=self.name,
      )
      act += transition_block(
          act,
          self.config.num_intermediate_factor,
          self.global_config,
          single_cond,
          name=self.name,
      )
      return act, None

    # Precompute pair logits for performance
    if pair_cond is None:
      pair_act = None
    else:
      pair_act = hm.LayerNorm(
          name='pair_input_layer_norm',
          use_fast_variance=False,
          create_offset=False,
      )(pair_cond)

    def super_block(act):
      if pair_act is None:
        pair_logits = None
      else:
        pair_logits = hm.Linear(
            (self.config.super_block_size, self.config.attention.num_head),
            name='pair_logits_projection',
        )(pair_act)
        pair_logits = jnp.transpose(pair_logits, [2, 3, 0, 1])
      return hk.experimental.layer_stack(
          self.config.super_block_size, with_per_layer_inputs=True
      )(block)(act, pair_logits)

    return hk.experimental.layer_stack(
        num_super_blocks, with_per_layer_inputs=True
    )(super_block)(act)[0]


class CrossAttentionConfig(base_config.BaseConfig):
  num_head: int = 4
  key_dim: int = 128
  value_dim: int = 128


def cross_attention(
    x_q: jnp.ndarray,  # (..., Q, C)
    x_k: jnp.ndarray,  # (..., K, C)
    mask_q: jnp.ndarray,  # (..., Q)
    mask_k: jnp.ndarray,  # (..., K)
    config: CrossAttentionConfig,
    global_config: model_config.GlobalConfig,
    pair_logits: jnp.ndarray | None = None,  # (..., Q, K)
    single_cond_q: jnp.ndarray | None = None,  # (..., Q, C)
    single_cond_k: jnp.ndarray | None = None,  # (..., K, C)
    name: str = '',
    kq_norm: bool = False,
    pair_mask: jnp.ndarray | None = None,  # (..., Q, K)
    keys_from_queries=None,
    rope_q: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    rope_k: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
  """Multihead self-attention."""
  # chai-1's atom transformers drop BOTH the gating_query and the output
  # projection: the raw concatenated heads are multiplied by the conditioning
  # gate and that is the whole output. See adaptive_zero_init(project=False).
  chai = global_config.model == 'chai1'
  assert len(mask_q.shape) == len(x_q.shape) - 1, f'{mask_q.shape}, {x_q.shape}'
  assert len(mask_k.shape) == len(x_k.shape) - 1, f'{mask_k.shape}, {x_k.shape}'
  # bias: ... x heads (1) x query x key
  if global_config.model in model_config.KEY_MASKED_ATOM_ATTENTION:
    # AF3 multiplies the two mask terms, so a pair is only penalised when the query
    # AND the key are invalid -- padded KEYS stay fully attendable from real queries.
    # RF3's atom attention adds them instead (`-1e9 * (maskQ + maskK)`), which is an
    # OR, so it never lets a real atom attend to a padded one. With a lone ligand the
    # 16 real atoms sit in a 32-query/128-key window, so under AF3's rule almost the
    # whole key window is padding the softmax still sees.
    #
    # protenix2 and opendde need exactly the same thing and used to miss it: both
    # PAD their key window instead of sliding it (`_padded_key_window`), and both
    # natives then write -inf into the padded columns for real queries. See
    # model_config.KEY_MASKED_ATOM_ATTENTION for the sources and the membership
    # rule. Note AF3's own token-level `self_attention` already masks keys alone
    # (`1e9 * (mask - 1)`); the AND form appears ONLY here, where the sliding
    # guarantee was what made it safe.
    bias = -1e9 * (
        (1.0 - mask_q)[..., None, :, None] + (1.0 - mask_k)[..., None, None, :]
    )
  else:
    bias = (
        1e9
        * (mask_q - 1.0)[..., None, :, None]
        * (mask_k - 1.0)[..., None, None, :]
    )

  if pair_mask is not None:
    # chai-1 restricts ATOM attention to atoms of the same token: its
    # atom_block_pair_mask is exactly (same token) AND (both atoms real), an
    # exact match over all 184x32x128 entries of the 6MRR seam. AF3 instead lets
    # every atom attend across the whole 32/128 window, which spreads the
    # softmax over ~80 keys where chai opens ~9 -- an attenuated, blurred
    # average whose damage is intra-residue geometry, not the fold.
    bias = jnp.where(pair_mask[..., None, :, :], bias, -1e9)

  x_q = adaptive_layernorm(x_q, single_cond_q, name=f'{name}q',
                           global_config=global_config)
  if keys_from_queries is not None:
    # opendde/protenix chain the two adaptive LayerNorms instead of running them
    # in parallel. Their AttentionPairBias in cross_attention_mode reads
    #
    #     a  = layernorm_a(a, s)      # queries
    #     kv = layernorm_kv(a, s)     # <- the ALREADY-NORMALISED a, not the input
    #
    # (opendde/model/modules/transformer.py, protenix/model/modules/transformer.py
    # -- `a` is reassigned by the first call). AF3 normalises the raw activation
    # twice, once per side. Two chained LayerNorms are not one: the second
    # re-centres and re-scales a tensor whose statistics the first already fixed,
    # and it sees the first one's learned scale.
    #
    # Worth the plumbing rather than normalising x_k in the caller, because the
    # keys' adaLN has to run on the q-normalised value in the KEYS layout, so the
    # gather has to happen between the two -- hence the callback.
    #
    # Measured on 1EHZ with every encoder input exact (c_skip corr 1.000000,
    # atom-pair p_skip corr 1.000000 inside native's own valid mask), this was
    # the only remaining difference in the atom encoder, which read a_token corr
    # 0.990195 / q_skip 0.991027 before it.
    x_k = keys_from_queries(x_q)
  x_k = adaptive_layernorm(x_k, single_cond_k, name=f'{name}k',
                           global_config=global_config)

  assert config.key_dim % config.num_head == 0
  assert config.value_dim % config.num_head == 0
  key_dim = config.key_dim // config.num_head
  value_dim = config.value_dim // config.num_head

  q = hm.Linear(
      (config.num_head, key_dim), use_bias=True, name=f'{name}q_projection'
  )(x_q)
  k = hm.Linear(
      (config.num_head, key_dim), use_bias=False, name=f'{name}k_projection'
  )(x_k)

  # RF3 kq_norm: LayerNorm on q and k over the flattened (num_head*key_dim) axis.
  if kq_norm:
    qk_flat = config.num_head * key_dim
    q = hm.LayerNorm(name=f'{name}query_layer_norm', use_fast_variance=False)(
        q.reshape(q.shape[:-2] + (qk_flat,))).reshape(q.shape)
    k = hm.LayerNorm(name=f'{name}key_layer_norm', use_fast_variance=False)(
        k.reshape(k.shape[:-2] + (qk_flat,))).reshape(k.shape)

  # In some situations the gradient norms can blow up without running this
  # einsum in float32.
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  bias = bias.astype(jnp.float32)

  if rope_q is not None:
    # ESMFold2's atom attention has no pair bias and no learned positional term:
    # the whole positional signal is a 3D rotary built from the reference
    # conformer, after an affine-free RMSNorm on q and k. Queries and keys carry
    # DIFFERENT rotations because they are different atom subsets of the same
    # flat list -- passing one for both silently rotates the keys as if they sat
    # at the query positions. None for every other model.
    q = _apply_rope(_rms_norm(q), *rope_q)
    k = _apply_rope(_rms_norm(k), *rope_k)

  logits = jnp.einsum('...qhc,...khc->...hqk', q * key_dim ** (-0.5), k) + bias
  if pair_logits is not None:
    logits += pair_logits
  weights = jax.nn.softmax(logits, axis=-1)
  weights = jnp.asarray(weights, dtype=x_q.dtype)

  v = hm.Linear(
      (config.num_head, value_dim), use_bias=False, name=f'{name}v_projection'
  )(x_k)
  weighted_avg = jnp.einsum('...hqk,...khc->...qhc', weights, v)
  weighted_avg = jnp.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))

  if not chai:
    gate_logits = hm.Linear(
        config.num_head * value_dim,
        bias_init=1.0,
        initializer='zeros',
        name=f'{name}gating_query',
    )(x_q)
    weighted_avg *= jax.nn.sigmoid(gate_logits)

  output = adaptive_zero_init(
      weighted_avg, x_q.shape[-1], single_cond_q, global_config, name,
      project=not chai,
  )
  return output


class CrossAttTransformer(hk.Module):
  """Transformer that applies cross attention between two sets of subsets."""

  class Config(base_config.BaseConfig):
    num_intermediate_factor: int
    num_blocks: int
    attention: CrossAttentionConfig = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name: str = 'transformer',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(
      self,
      queries_act: jnp.ndarray,  # (num_subsets, num_queries, ch)
      queries_mask: jnp.ndarray,  # (num_subsets, num_queries)
      queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys)
      keys_mask: jnp.ndarray,  # (num_subsets, num_keys)
      queries_single_cond: jnp.ndarray,  # (num_subsets, num_queries, ch)
      keys_single_cond: jnp.ndarray,  # (num_subsets, num_keys, ch)
      pair_cond: jnp.ndarray,  # (num_subsets, num_queries, num_keys, ch)
      pair_mask: jnp.ndarray | None = None,  # (num_subsets, num_queries, num_keys)
      rope_q: tuple[jnp.ndarray, jnp.ndarray] | None = None,
      rope_k: tuple[jnp.ndarray, jnp.ndarray] | None = None,
  ) -> jnp.ndarray:
    chai = self.global_config.model == 'chai1'

    def block(queries_act, pair_logits):
      # chai's atom stack is PARALLEL and re-masks at the top of every block:
      #     m = single * atom_mask[:, None]
      #     x  = ln(m, eps=0.1)*(scale+1)+shift     # attention reads m
      #     x2 = ln(m, eps=0.1)*(s2+1)+sh2          # transition ALSO reads m
      #     single = m + local + trans
      # AF3 threads the transition through the post-attention activation. Same
      # divergence as the trunk's MSA block, one level down -- and with every
      # input to this stack now exact (atom_query / atom_cond / atom_pair all
      # 1.000000) it is the only thing left that can move atom_repr off 0.9930.
      if chai:
        queries_act = queries_act * queries_mask[..., None].astype(
            queries_act.dtype)
      block_in = queries_act

      # copy the queries activations to the keys layout
      keys_act = atom_layout.convert(
          queries_to_keys, queries_act, layout_axes=(-3, -2)
      )
      # cross attention
      attn = cross_attention(
          x_q=queries_act,
          x_k=keys_act,  # pyrefly: ignore[bad-argument-type]
          mask_q=queries_mask,
          mask_k=keys_mask,
          config=self.config.attention,
          global_config=self.global_config,
          pair_logits=pair_logits,
          single_cond_q=queries_single_cond,
          single_cond_k=keys_single_cond,
          name=self.name,
          pair_mask=pair_mask,
          rope_q=rope_q,
          rope_k=rope_k,
      )
      trans_in = block_in if chai else (queries_act + attn)
      trans = transition_block(
          trans_in,
          self.config.num_intermediate_factor,
          self.global_config,
          queries_single_cond,
          name=self.name,
      )
      queries_act = block_in + attn + trans
      # the two branches separately: chai's block output is
      # 1929 = 1777(block_in) + 1927(trans) + 1907(attn), so tapping each says
      # WHICH branch is wrong rather than just that the block is.
      return queries_act, ((queries_act, attn, trans) if _TAP_ATOM_BLOCKS
                           else None)

    # OpenDDE applies the atom-pair LayerNorm PER BLOCK (its own layernorm_z per
    # block) rather than once shared. Gated on global_config.model so AF3/OF3/IF2
    # (shared LN, below) are byte-unchanged. pair_input_layer_norm + pair_logits_
    # projection are created inside the block -> stacked along the layer_stack axis.
    if self.global_config.model in model_config.PER_BLOCK_ATOM_PAIR_LAYER_NORM:
      # OpenDDE/Protenix/RF3 apply the atom-pair LayerNorm PER BLOCK (their
      # per-block layernorm_z weights genuinely differ). Same forward path.
      rosettafold3 = self.global_config.model == 'rosettafold3'
      def od_block(queries_act):
        pa = hm.LayerNorm(name='pair_input_layer_norm', use_fast_variance=False,
                          create_offset=False)(pair_cond)
        pl = hm.Linear(self.config.attention.num_head,
                       name='pair_logits_projection')(pa)
        pl = jnp.transpose(pl, [0, 3, 1, 2])   # (subsets, heads, queries, keys)
        keys_act = atom_layout.convert(queries_to_keys, queries_act,
                                       layout_axes=(-3, -2))
        # rf3 keeps AF3's parallel normalisation; opendde/protenix chain them,
        # so hand cross_attention the gather and let it run between the two.
        _kfq = None if rosettafold3 else (
            lambda xq: atom_layout.convert(queries_to_keys, xq,
                                           layout_axes=(-3, -2)))
        attn = cross_attention(
            x_q=queries_act, x_k=keys_act, mask_q=queries_mask, mask_k=keys_mask,
            config=self.config.attention, global_config=self.global_config,
            pair_logits=pl, single_cond_q=queries_single_cond,
            single_cond_k=keys_single_cond, name=self.name, kq_norm=rosettafold3,
            keys_from_queries=_kfq)
        if rosettafold3:
          # RF3 no_residual: transition reads the pre-attention act; one residual.
          queries_act = queries_act + attn + transition_block(
              queries_act, self.config.num_intermediate_factor, self.global_config,
              queries_single_cond, name=self.name)
        else:
          queries_act += attn
          queries_act += transition_block(
              queries_act, self.config.num_intermediate_factor, self.global_config,
              queries_single_cond, name=self.name)
        return queries_act
      return hk.experimental.layer_stack(self.config.num_blocks)(od_block)(queries_act)

    # Precompute pair logits for performance.
    # chai shares this LayerNorm across the stack too (one
    # blocked_pairs2blocked_bias.0, with a per-block slice in .1), so it takes
    # this path -- but its LayerNorm is affine on both scale and offset.
    pair_act = hm.LayerNorm(
        name='pair_input_layer_norm',
        use_fast_variance=False,
        create_offset=self.global_config.model == 'chai1',
    )(pair_cond)
    # (num_subsets, num_queries, num_keys, num_blocks, num_heads)
    pair_logits = hm.Linear(
        (self.config.num_blocks, self.config.attention.num_head),
        name='pair_logits_projection',
    )(pair_act)
    # (num_block, num_subsets, num_heads, num_queries, num_keys)
    pair_logits = jnp.transpose(pair_logits, [3, 0, 4, 1, 2])

    stack_in = queries_act
    stacked, per_block = hk.experimental.layer_stack(
        self.config.num_blocks, with_per_layer_inputs=True
    )(block)(queries_act, pair_logits)
    if _TAP_ATOM_BLOCKS:
      # the stack's INPUT rides out too: a per-block gap means nothing until the
      # input to block 0 is known to be exact.
      ATOM_BLOCK_TAPS.append((stack_in, per_block))
    return stacked
