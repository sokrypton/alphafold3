# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules and code used in the core part of AlphaFold.

The structure generation code is in 'folding.py'.
"""
import functools
from alphafold3.af2.common import residue_constants
from alphafold3.af2.model import common_modules
from alphafold3.af2.model import layer_stack
from alphafold3.af2.model import lddt
from alphafold3.af2.model import mapping
from alphafold3.af2.model import prng
from alphafold3.af2.model import quat_affine
from alphafold3.af2.model import utils
import numpy as np
from typing import Sequence
from alphafold3.af2.model import all_atom
from alphafold3.af2.model import folding
from alphafold3.af2.model import geometry
import haiku as hk
import jax
import jax.numpy as jnp

from alphafold3.af2.model.r3 import Rigids, Rots, Vecs

def apply_dropout(*, tensor, safe_key, rate, broadcast_dim=None):
  """Applies dropout to a tensor."""
  shape = list(tensor.shape)
  if broadcast_dim is not None:
    shape[broadcast_dim] = 1
  keep_rate = 1.0 - rate
  keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
  return keep * tensor / keep_rate

def dropout_wrapper(module,
                    input_act,
                    mask,
                    safe_key,
                    global_config,
                    use_dropout,
                    output_act=None,
                    **kwargs):
  """Applies module + dropout + residual update."""
  if output_act is None:
    output_act = input_act

  gc = global_config
  residual = module(input_act, mask, **kwargs)

  if module.config.shared_dropout:
    if module.config.orientation == 'per_row':
      broadcast_dim = 0
    else:
      broadcast_dim = 1
  else:
    broadcast_dim = None

  residual = apply_dropout(tensor=residual,
                           safe_key=safe_key,
                           rate=jnp.where(use_dropout, module.config.dropout_rate, 0),
                           broadcast_dim=broadcast_dim)

  new_act = output_act + residual

  return new_act

class TemplatePairStack(hk.Module):
  """Pair stack for the templates.

  Jumper et al. (2021) Suppl. Alg. 16 "TemplatePairStack"
  """

  def __init__(self, config, global_config, name='template_pair_stack'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, pair_mask, use_dropout, safe_key=None):
    """Builds TemplatePairStack module.

    Arguments:
      pair_act: Pair activations for single template, shape [N_res, N_res, c_t].
      pair_mask: Pair mask, shape [N_res, N_res].
      safe_key: Safe key object encapsulating the random number generation key.

    Returns:
      Updated pair_act, shape [N_res, N_res, c_t].
    """

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    gc = self.global_config
    c = self.config

    if not c.num_block:
      return pair_act

    def block(x):
      """One block of the template pair stack."""
      pair_act, safe_key = x

      dropout_wrapper_fn = functools.partial(
          dropout_wrapper, global_config=gc, use_dropout=use_dropout)

      safe_key, *sub_keys = safe_key.split(6)
      sub_keys = iter(sub_keys)

      pair_act = dropout_wrapper_fn(
          TriangleAttention(c.triangle_attention_starting_node, gc,
                            name='triangle_attention_starting_node'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          TriangleAttention(c.triangle_attention_ending_node, gc,
                            name='triangle_attention_ending_node'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                                 name='triangle_multiplication_outgoing'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                                 name='triangle_multiplication_incoming'),
          pair_act,
          pair_mask,
          next(sub_keys))
      pair_act = dropout_wrapper_fn(
          Transition(c.pair_transition, gc, name='pair_transition'),
          pair_act,
          pair_mask,
          next(sub_keys))

      return pair_act, safe_key

    if gc.use_remat:
      block = hk.remat(block)

    res_stack = layer_stack.layer_stack(c.num_block)(block)
    pair_act, safe_key = res_stack((pair_act, safe_key))
    return pair_act


class Transition(hk.Module):
  """Transition layer.

  Jumper et al. (2021) Suppl. Alg. 9 "MSATransition"
  Jumper et al. (2021) Suppl. Alg. 15 "PairTransition"
  """

  def __init__(self, config, global_config, name='transition_block'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, mask):
    """Builds Transition module.

    Arguments:
      act: A tensor of queries of size [batch_size, N_res, N_channel].
      mask: A tensor denoting the mask of size [batch_size, N_res].

    Returns:
      A float32 tensor of size [batch_size, N_res, N_channel].
    """
    _, _, nc = act.shape

    num_intermediate = int(nc * self.config.num_intermediate_factor)
    mask = jnp.expand_dims(mask, axis=-1)

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='input_layer_norm')(
            act)

    transition_module = hk.Sequential([
        common_modules.Linear(
            num_intermediate,
            initializer='relu',
            name='transition1'), jax.nn.relu,
        common_modules.Linear(
            nc,
            initializer=utils.final_init(self.global_config),
            name='transition2')
    ])

    act = mapping.inference_subbatch(
        transition_module,
        self.global_config.subbatch_size,
        batched_args=[act],
        nonbatched_args=[],
        low_memory=self.global_config.subbatch_size is not None)

    return act


def glorot_uniform():
  return hk.initializers.VarianceScaling(scale=1.0,
                                         mode='fan_avg',
                                         distribution='uniform')


class Attention(hk.Module):
  """Multihead attention."""

  def __init__(self, config, global_config, output_dim, name='attention'):
    super().__init__(name=name)

    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim

  def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
    """Builds Attention module.

    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].

    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    # Sensible default for when the config keys are missing
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head

    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], num_head, value_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())

    q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)

    # ColabDesign2: AF3's attention kernel, on AF2's tensors.
    #
    # q/k/v are already (batch, seq, heads, dim) and the bias is already
    # (batch, heads, q, k) -- exactly tokamax's layout -- so this is a drop-in,
    # not a rewrite. AF3 dispatches every attention through
    # tokamax.dot_product_attention (model/network/modules.py:182); AF2 has
    # always built the full (b, h, q, k) logits matrix, which is quadratic in
    # sequence length. The flash kernels are linear.
    #
    # scale=1.0 because AF2 pre-scales q by key_dim**-0.5 above, and doubling
    # that would silently change every prediction.
    # Dispatch by size. Flash kernels are linear in sequence length where the
    # einsum is quadratic, but they carry per-call overhead and hardware limits,
    # so they only pay above a threshold -- measured on an A10, xla beats cudnn
    # at L=256 and loses 4.7x by L=1024.
    #
    # The hard constraint is the short case, not the slow one: AF2's MSA COLUMN
    # attention runs over the MSA depth axis, and design uses a single sequence,
    # so that axis is length 1. cuDNN rejects it outright ("Unsupported sequence
    # length Q 1, KV 1"), and the failure arrives as "vjp not implemented" from
    # several frames away.
    impl = self.global_config.get('flash_attention', None)
    min_len = self.global_config.get('flash_attention_min_len', 256)
    if (impl not in (None, 'none')
        and q_data.shape[-2] >= min_len and m_data.shape[-2] >= min_len):
      total_bias = bias
      if nonbatched_bias is not None:
        total_bias = total_bias + jnp.expand_dims(nonbatched_bias, axis=0)
      from alphafold3.model.components.attention import dot_product_attention as _flash_dpa
      weighted_avg = _flash_dpa(
          q, k, v, bias=total_bias, scale=1.0, implementation=impl)
    else:
      logits = jnp.einsum('bqhc,bkhc->bhqk', q, k) + bias
      if nonbatched_bias is not None:
        logits += jnp.expand_dims(nonbatched_bias, axis=0)

      # patch for jax > 0.3.25
      logits = jnp.clip(logits,-1e8,1e8)

      weights = jax.nn.softmax(logits)
      weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(0.0))
      gating_bias = hk.get_parameter(
          'gating_b',
          shape=(num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(1.0))

      gate_values = jnp.einsum('bqc, chv->bqhv', q_data,
                               gating_weights) + gating_bias

      gate_values = jax.nn.sigmoid(gate_values)

      weighted_avg *= gate_values

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        dtype=q_data.dtype,
        init=init)
    o_bias = hk.get_parameter('output_b', shape=(self.output_dim,),
                              dtype=q_data.dtype,
                              init=hk.initializers.Constant(0.0))

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

    return output


class GlobalAttention(hk.Module):
  """Global attention.

  Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention" lines 2-7
  """

  def __init__(self, config, global_config, output_dim, name='attention'):
    super().__init__(name=name)

    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim

  def __call__(self, q_data, m_data, q_mask, bias):
    """Builds GlobalAttention module.

    Arguments:
      q_data: A tensor of queries with size [batch_size, N_queries,
        q_channels]
      m_data: A tensor of memories from which the keys and values
        projected. Size [batch_size, N_keys, m_channels]
      q_mask: A binary mask for q_data with zeros in the padded sequence
        elements and ones otherwise. Size [batch_size, N_queries, q_channels]
        (or broadcastable to this shape).
      bias: A bias for the attention.

    Returns:
      A float32 tensor of size [batch_size, N_queries, output_dim].
    """
    # Sensible default for when the config keys are missing
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head

    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], value_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())

    v = jnp.einsum('bka,ac->bkc', m_data, v_weights)

    q_avg = utils.mask_mean(q_mask, q_data, axis=1)

    q = jnp.einsum('ba,ahc->bhc', q_avg, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ac->bkc', m_data, k_weights)
    bias = (1e9 * (q_mask[:, None, :, 0] - 1.))
    logits = jnp.einsum('bhc,bkc->bhk', q, k) + bias
    weights = jax.nn.softmax(logits)
    weighted_avg = jnp.einsum('bhk,bkc->bhc', weights, v)

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        dtype=q_data.dtype,
        init=init)
    o_bias = hk.get_parameter('output_b', shape=(self.output_dim,),
                              dtype=q_data.dtype,
                              init=hk.initializers.Constant(0.0))

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(0.0))
      gating_bias = hk.get_parameter(
          'gating_b',
          shape=(num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(1.0))

      gate_values = jnp.einsum('bqc, chv->bqhv', q_data, gating_weights)
      gate_values = jax.nn.sigmoid(gate_values + gating_bias)
      weighted_avg = weighted_avg[:, None] * gate_values
      output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias
    else:
      output = jnp.einsum('bhc,hco->bo', weighted_avg, o_weights) + o_bias
      output = output[:, None]
    return output


class MSARowAttentionWithPairBias(hk.Module):
  """MSA per-row attention biased by the pair representation.

  Jumper et al. (2021) Suppl. Alg. 7 "MSARowAttentionWithPairBias"
  """

  def __init__(self, config, global_config,
               name='msa_row_attention_with_pair_bias'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               msa_act,
               msa_mask,
               pair_act,
               pair_mask):
    """Builds MSARowAttentionWithPairBias module.

    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      pair_act: [N_res, N_res, c_z] pair representation.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m].
    """
    c = self.config

    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert c.orientation == 'per_row'

    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    pair_act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='feat_2d_norm')(
            pair_act)

    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        dtype=msa_act.dtype,
        init=hk.initializers.RandomNormal(stddev=init_factor))
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights)
    nonbatched_bias = nonbatched_bias + (1e9 * (pair_mask - 1.0))
    
    attn_mod = Attention(
        c, self.global_config, msa_act.shape[-1])
    msa_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[msa_act, msa_act, bias],
        nonbatched_args=[nonbatched_bias],
        low_memory=self.global_config.subbatch_size is not None)

    return msa_act


class MSAColumnAttention(hk.Module):
  """MSA per-column attention.

  Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
  """

  def __init__(self, config, global_config, name='msa_column_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               msa_act,
               msa_mask):
    """Builds MSAColumnAttention module.

    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m]
    """
    c = self.config

    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert c.orientation == 'per_column'

    msa_act = jnp.swapaxes(msa_act, -2, -3)
    msa_mask = jnp.swapaxes(msa_mask, -1, -2)

    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    attn_mod = Attention(
        c, self.global_config, msa_act.shape[-1])
    msa_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[msa_act, msa_act, bias],
        nonbatched_args=[],
        low_memory=self.global_config.subbatch_size is not None)

    msa_act = jnp.swapaxes(msa_act, -2, -3)

    return msa_act


class MSAColumnGlobalAttention(hk.Module):
  """MSA per-column global attention.

  Jumper et al. (2021) Suppl. Alg. 19 "MSAColumnGlobalAttention"
  """

  def __init__(self, config, global_config, name='msa_column_global_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               msa_act,
               msa_mask):
    """Builds MSAColumnGlobalAttention module.

    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.

    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m].
    """
    c = self.config

    assert len(msa_act.shape) == 3
    assert len(msa_mask.shape) == 2
    assert c.orientation == 'per_column'

    msa_act = jnp.swapaxes(msa_act, -2, -3)
    msa_mask = jnp.swapaxes(msa_mask, -1, -2)

    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    msa_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            msa_act)

    attn_mod = GlobalAttention(
        c, self.global_config, msa_act.shape[-1],
        name='attention')
    # [N_seq, N_res, 1]
    msa_mask = jnp.expand_dims(msa_mask, axis=-1)
    msa_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[msa_act, msa_act, msa_mask, bias],
        nonbatched_args=[],
        low_memory=self.global_config.subbatch_size is not None)

    msa_act = jnp.swapaxes(msa_act, -2, -3)

    return msa_act


class TriangleAttention(hk.Module):
  """Triangle Attention.

  Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
  Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
  """

  def __init__(self, config, global_config, name='triangle_attention'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, pair_act, pair_mask):
    """Builds TriangleAttention module.

    Arguments:
      pair_act: [N_res, N_res, c_z] pair activations tensor
      pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.

    Returns:
      Update to pair_act, shape [N_res, N_res, c_z].
    """
    c = self.config

    assert len(pair_act.shape) == 3
    assert len(pair_mask.shape) == 2
    assert c.orientation in ['per_row', 'per_column']

    if c.orientation == 'per_column':
      pair_act = jnp.swapaxes(pair_act, -2, -3)
      pair_mask = jnp.swapaxes(pair_mask, -1, -2)

    bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
    assert len(bias.shape) == 4

    pair_act = common_modules.LayerNorm(
        axis=[-1], create_scale=True, create_offset=True, name='query_norm')(
            pair_act)

    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter(
        'feat_2d_weights',
        shape=(pair_act.shape[-1], c.num_head),
        dtype=pair_act.dtype,
        init=hk.initializers.RandomNormal(stddev=init_factor))
    nonbatched_bias = jnp.einsum('qkc,ch->hqk', pair_act, weights)

    attn_mod = Attention(
        c, self.global_config, pair_act.shape[-1])
    pair_act = mapping.inference_subbatch(
        attn_mod,
        self.global_config.subbatch_size,
        batched_args=[pair_act, pair_act, bias],
        nonbatched_args=[nonbatched_bias],
        low_memory=self.global_config.subbatch_size is not None)

    if c.orientation == 'per_column':
      pair_act = jnp.swapaxes(pair_act, -2, -3)

    return pair_act


class MaskedMsaHead(hk.Module):
  """Head to predict MSA at the masked locations.

  The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
  version of the full MSA, based on a linear projection of
  the MSA representation.
  Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
  """

  def __init__(self, config, global_config, name='masked_msa_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

    if global_config.multimer_mode:
      self.num_output = len(residue_constants.restypes_with_x_and_gap)
    else:
      self.num_output = config.num_output

  def __call__(self, representations, batch):
    """Builds MaskedMsaHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'msa': MSA representation, shape [N_seq, N_res, c_m].
      batch: Batch, unused.

    Returns:
      Dictionary containing:
        * 'logits': logits of shape [N_seq, N_res, N_aatype] with
            (unnormalized) log probabilies of predicted aatype at position.
    """
    del batch
    logits = common_modules.Linear(
        self.num_output,
        initializer=utils.final_init(self.global_config),
        name='logits')(
            representations['msa'])
    return dict(logits=logits)

class PredictedLDDTHead(hk.Module):
  """Head to predict the per-residue LDDT to be used as a confidence measure.

  Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
  Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
  """

  def __init__(self, config, global_config, name='predicted_lddt_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, representations, batch):
    """Builds ExperimentallyResolvedHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'structure_module': Single representation from the structure module,
             shape [N_res, c_s].
      batch: Batch, unused.

    Returns:
      Dictionary containing :
        * 'logits': logits of shape [N_res, N_bins] with
            (unnormalized) log probabilies of binned predicted lDDT.
    """
    act = representations['structure_module']

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='input_layer_norm')(
            act)

    act = common_modules.Linear(
        self.config.num_channels,
        initializer='relu',
        name='act_0')(
            act)
    act = jax.nn.relu(act)

    act = common_modules.Linear(
        self.config.num_channels,
        initializer='relu',
        name='act_1')(
            act)
    act = jax.nn.relu(act)

    logits = common_modules.Linear(
        self.config.num_bins,
        initializer=utils.final_init(self.global_config),
        name='logits')(
            act)
    # Shape (batch_size, num_res, num_bins)
    return dict(logits=logits)


class PredictedAlignedErrorHead(hk.Module):
  """Head to predict the distance errors in the backbone alignment frames.

  Can be used to compute predicted TM-Score.
  Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
  """

  def __init__(self, config, global_config,
               name='predicted_aligned_error_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, representations, batch):
    """Builds PredictedAlignedErrorHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.

    Returns:
      Dictionary containing:
        * logits: logits for aligned error, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1].
    """

    act = representations['pair']

    # Shape (num_res, num_res, num_bins)
    logits = common_modules.Linear(
        self.config.num_bins,
        initializer=utils.final_init(self.global_config),
        name='logits')(act)
    # Shape (num_bins,)
    breaks = jnp.linspace(
        0., self.config.max_error_bin, self.config.num_bins - 1)
    return dict(logits=logits, breaks=breaks)


class ExperimentallyResolvedHead(hk.Module):
  """Predicts if an atom is experimentally resolved in a high-res structure.

  Only trained on high-resolution X-ray crystals & cryo-EM.
  Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
  """

  def __init__(self, config, global_config,
               name='experimentally_resolved_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, representations, batch):
    """Builds ExperimentallyResolvedHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'single': Single representation, shape [N_res, c_s].
      batch: Batch, unused.

    Returns:
      Dictionary containing:
        * 'logits': logits of shape [N_res, 37],
            log probability that an atom is resolved in atom37 representation,
            can be converted to probability by applying sigmoid.
    """
    logits = common_modules.Linear(
        37,  # atom_exists.shape[-1]
        initializer=utils.final_init(self.global_config),
        name='logits')(representations['single'])
    return dict(logits=logits)


def _layer_norm(axis=-1, name='layer_norm'):
  return common_modules.LayerNorm(
      axis=axis,
      create_scale=True,
      create_offset=True,
      eps=1e-5,
      use_fast_variance=True,
      scale_init=hk.initializers.Constant(1.),
      offset_init=hk.initializers.Constant(0.),
      param_axis=axis,
      name=name)

class TriangleMultiplication(hk.Module):
  """Triangle multiplication layer ("outgoing" or "incoming").
  Jumper et al. (2021) Suppl. Alg. 11 "TriangleMultiplicationOutgoing"
  Jumper et al. (2021) Suppl. Alg. 12 "TriangleMultiplicationIncoming"
  """

  def __init__(self, config, global_config, name='triangle_multiplication'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, left_act, left_mask):
    """Builds TriangleMultiplication module.
    Arguments:
      left_act: Pair activations, shape [N_res, N_res, c_z]
      left_mask: Pair mask, shape [N_res, N_res].
    Returns:
      Outputs, same shape/type as left_act.
    """

    if self.config.fuse_projection_weights:
      return self._fused_triangle_multiplication(left_act, left_mask)
    else:
      return self._triangle_multiplication(left_act, left_mask)

  @hk.transparent
  def _triangle_multiplication(self, left_act, left_mask):
    """Implementation of TriangleMultiplication used in AF2 and AF-M<2.3."""
    c = self.config
    gc = self.global_config

    mask = left_mask[..., None]

    act = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True,
                       name='layer_norm_input')(left_act)
    input_act = act

    left_projection = common_modules.Linear(
        c.num_intermediate_channel,
        name='left_projection')
    left_proj_act = mask * left_projection(act)

    right_projection = common_modules.Linear(
        c.num_intermediate_channel,
        name='right_projection')
    right_proj_act = mask * right_projection(act)

    left_gate_values = jax.nn.sigmoid(common_modules.Linear(
        c.num_intermediate_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='left_gate')(act))

    right_gate_values = jax.nn.sigmoid(common_modules.Linear(
        c.num_intermediate_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='right_gate')(act))

    left_proj_act *= left_gate_values
    right_proj_act *= right_gate_values

    # "Outgoing" edges equation: 'ikc,jkc->ijc'
    # "Incoming" edges equation: 'kjc,kic->ijc'
    # Note on the Suppl. Alg. 11 & 12 notation:
    # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
    # For the "incoming" edges, it's swapped:
    #   b = left_proj_act and a = right_proj_act
    act = jnp.einsum(c.equation, left_proj_act, right_proj_act)

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='center_layer_norm')(
            act)

    output_channel = int(input_act.shape[-1])

    act = common_modules.Linear(
        output_channel,
        initializer=utils.final_init(gc),
        name='output_projection')(act)

    gate_values = jax.nn.sigmoid(common_modules.Linear(
        output_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='gating_linear')(input_act))
    act *= gate_values

    return act

  @hk.transparent
  def _fused_triangle_multiplication(self, left_act, left_mask):
    """TriangleMultiplication with fused projection weights."""
    mask = left_mask[..., None]
    c = self.config
    gc = self.global_config

    left_act = _layer_norm(axis=-1, name='left_norm_input')(left_act)

    # Both left and right projections are fused into projection.
    projection = common_modules.Linear(
        2*c.num_intermediate_channel, name='projection')
    proj_act = mask * projection(left_act)

    # Both left + right gate are fused into gate_values.
    gate_values = common_modules.Linear(
        2 * c.num_intermediate_channel,
        name='gate',
        bias_init=1.,
        initializer=utils.final_init(gc))(left_act)
    proj_act *= jax.nn.sigmoid(gate_values)

    left_proj_act = proj_act[:, :, :c.num_intermediate_channel]
    right_proj_act = proj_act[:, :, c.num_intermediate_channel:]
    act = jnp.einsum(c.equation, left_proj_act, right_proj_act)

    act = _layer_norm(axis=-1, name='center_norm')(act)

    output_channel = int(left_act.shape[-1])

    act = common_modules.Linear(
        output_channel,
        initializer=utils.final_init(gc),
        name='output_projection')(act)

    gate_values = common_modules.Linear(
        output_channel,
        bias_init=1.,
        initializer=utils.final_init(gc),
        name='gating_linear')(left_act)
    act *= jax.nn.sigmoid(gate_values)

    return act


class DistogramHead(hk.Module):
  """Head to predict a distogram.

  Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
  """

  def __init__(self, config, global_config, name='distogram_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, representations, batch):
    """Builds DistogramHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.

    Returns:
      Dictionary containing:
        * logits: logits for distogram, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
    """
    half_logits = common_modules.Linear(
        self.config.num_bins,
        initializer=utils.final_init(self.global_config),
        name='half_logits')(
            representations['pair'])

    logits = half_logits + jnp.swapaxes(half_logits, -2, -3)
    breaks = jnp.linspace(self.config.first_break, self.config.last_break,
                          self.config.num_bins - 1)

    return dict(logits=logits, bin_edges=breaks)


class OuterProductMean(hk.Module):
  """Computes mean outer product.

  Jumper et al. (2021) Suppl. Alg. 10 "OuterProductMean"
  """

  def __init__(self,
               config,
               global_config,
               num_output_channel,
               name='outer_product_mean'):
    super().__init__(name=name)
    self.global_config = global_config
    self.config = config
    self.num_output_channel = num_output_channel

  def __call__(self, act, mask):
      """Builds OuterProductMean module."""
      gc = self.global_config
      c = self.config
      epsilon = 1e-3

      # Extract masks from dict
      if isinstance(mask, dict):
        msa_mask = mask["msa"][..., None]
        cov_mask = mask.get("cov", None)
      else:
        msa_mask = mask[..., None]
        cov_mask = None

      act = common_modules.LayerNorm([-1], True, True, name='layer_norm_input')(act)

      left_act = msa_mask * common_modules.Linear(
          c.num_outer_channel,
          initializer='linear',
          name='left_projection')(act)

      right_act = msa_mask * common_modules.Linear(
          c.num_outer_channel,
          initializer='linear',
          name='right_projection')(act)

      if gc.zero_init:
        init_w = hk.initializers.Constant(0.0)
      else:
        init_w = hk.initializers.VarianceScaling(scale=2., mode='fan_in')

      output_w = hk.get_parameter(
          'output_w',
          shape=(c.num_outer_channel, c.num_outer_channel,
                 self.num_output_channel),
          dtype=act.dtype,
          init=init_w)
      output_b = hk.get_parameter(
          'output_b', shape=(self.num_output_channel,),
          dtype=act.dtype,
          init=hk.initializers.Constant(0.0))

      if cov_mask is not None:

        left_act_sum = left_act.sum(0)
        norm_seq = msa_mask.sum(0) + epsilon
        right_act_mean = right_act.sum(0) / norm_seq
        
        left_act_sum = left_act_sum[None, :, :]
        cov_mask = cov_mask.T[None, :, :]

      def compute_chunk(left_act_chunk, left_act_sum_chunk=None, cov_mask_chunk=None):
        left_act_chunk = jnp.transpose(left_act_chunk, [0, 2, 1])
        act = jnp.einsum('acb,ade->dceb', left_act_chunk, right_act)
        
        if cov_mask is not None:
          left_act_sum_chunk = jnp.transpose(left_act_sum_chunk, [0, 2, 1])[0]
          cov_mask_chunk = cov_mask_chunk[0].T[:, None, None, :]
          
          act_alt = jnp.einsum('cb,de->dceb', left_act_sum_chunk, right_act_mean)
          act = act * cov_mask_chunk + act_alt * (1 - cov_mask_chunk)
        
        act = jnp.einsum('dceb,cef->dbf', act, output_w) + output_b
        return jnp.transpose(act, [1, 0, 2])

      if cov_mask is not None:
        batched_args = [left_act, left_act_sum, cov_mask]
      else:
        batched_args = [left_act]

      act = mapping.inference_subbatch(
          compute_chunk,
          c.chunk_size,
          batched_args=batched_args,
          nonbatched_args=[],
          low_memory=True,
          input_subbatch_dim=1,
          output_subbatch_dim=0)

      norm = jnp.einsum('abc,adc->bdc', msa_mask, msa_mask)
      act /= epsilon + norm

      return act

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
  """Compute distogram from amino acid positions.
  Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
        everything larger than `max_bin`.
  Returns:
    Distogram with the specified number of bins.
  """
  def squared_difference(x, y):
    return jnp.square(x - y)

  lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
  lower_breaks = jnp.square(lower_breaks)
  upper_breaks = jnp.concatenate([lower_breaks[1:],jnp.array([1e8], dtype=jnp.float32)], axis=-1)
  dist2 = jnp.sum(
      squared_difference(
          jnp.expand_dims(positions, axis=-2),
          jnp.expand_dims(positions, axis=-3)),
      axis=-1, keepdims=True)

  return ((dist2 > lower_breaks).astype(jnp.float32) * (dist2 < upper_breaks).astype(jnp.float32))

def dgram_from_positions_soft(positions, num_bins, min_bin, max_bin, temp=2.0):
  '''soft positions to dgram converter'''
  lower_breaks = jnp.append(-1e8,jnp.linspace(min_bin, max_bin, num_bins))
  upper_breaks = jnp.append(lower_breaks[1:],1e8)
  dist = jnp.sqrt(jnp.square(positions[...,:,None,:] - positions[...,None,:,:]).sum(-1,keepdims=True) + 1e-8)
  o = jax.nn.sigmoid((dist - lower_breaks)/temp) * jax.nn.sigmoid((upper_breaks - dist)/temp)
  o = o/(o.sum(-1,keepdims=True) + 1e-8)
  return o[...,1:]

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask=None):
  """Create pseudo beta features."""
  
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']

  # set gap character to alanine
  aatype = jnp.where(jnp.equal(aatype, 21),0,aatype)

  is_gly = jnp.equal(aatype, residue_constants.restype_order['G'])
  is_gly_tile = jnp.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3])
  pseudo_beta = jnp.where(is_gly_tile, all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :])

  if all_atom_mask is None:
    return pseudo_beta
  else:
    pseudo_beta_mask = jnp.where(is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.astype(jnp.float32)
    return pseudo_beta, pseudo_beta_mask

class EvoformerIteration(hk.Module):
  """Single iteration (block) of Evoformer stack.
  Jumper et al. (2021) Suppl. Alg. 6 "EvoformerStack" lines 2-10
  """

  def __init__(self, config, global_config, is_extra_msa,
               name='evoformer_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.is_extra_msa = is_extra_msa

  def __call__(self, activations, masks, use_dropout, safe_key=None):
    """Builds EvoformerIteration module.

    Arguments:
      activations: Dictionary containing activations:
        * 'msa': MSA activations, shape [N_seq, N_res, c_m].
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    c = self.config
    gc = self.global_config

    msa_act, pair_act = activations['msa'], activations['pair']

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    pair_mask = masks['pair']
    msa_mask_dict = {
      "msa":masks['msa'], 
      "cov":masks.get("cov",None)
    }

    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        global_config=gc,
        use_dropout=use_dropout)

    safe_key, *sub_keys = safe_key.split(11)
    sub_keys = iter(sub_keys)

    outer_module = OuterProductMean(
        config=c.outer_product_mean,
        global_config=self.global_config,
        num_output_channel=int(pair_act.shape[-1]),
        name='outer_product_mean')

    # ColabDesign2: outer_product_mean.first as a COMPILE-TIME setting (the config
    # bool), i.e. gamma's plain path -- OPM before the MSA stack (first=True,
    # multimer) or after it (first=False, monomer). No traced blend / no double
    # OPM. This recompiles on a monomer<->multimer switch; the traced blend that
    # avoided that was proven value-equivalent (bypass test), so this is a safe
    # simplification. FUTURE: jax.lax.cond for no-recompile without the blend.
    opm_first = bool(c.outer_product_mean.first)

    if opm_first:
      pair_act = dropout_wrapper_fn(
          outer_module, msa_act, msa_mask_dict,
          safe_key=next(sub_keys), output_act=pair_act)
    pair_for_msa = pair_act

    msa_act = dropout_wrapper_fn(
        MSARowAttentionWithPairBias(
            c.msa_row_attention_with_pair_bias, gc,
            name='msa_row_attention_with_pair_bias'),
        msa_act,
        msa_mask_dict["msa"],
        safe_key=next(sub_keys),
        pair_act=pair_for_msa,
        pair_mask=pair_mask)

    if not self.is_extra_msa:
      attn_mod = MSAColumnAttention(
          c.msa_column_attention, gc, name='msa_column_attention')
    else:
      attn_mod = MSAColumnGlobalAttention(
          c.msa_column_attention, gc, name='msa_column_global_attention')
    msa_act = dropout_wrapper_fn(
        attn_mod,
        msa_act,
        msa_mask_dict["msa"],
        safe_key=next(sub_keys))

    msa_act = dropout_wrapper_fn(
        Transition(c.msa_transition, gc, name='msa_transition'),
        msa_act,
        msa_mask_dict["msa"],
        safe_key=next(sub_keys))

    if not opm_first:
      # OPM after the MSA stack (the "last" placement)
      pair_act = dropout_wrapper_fn(
          outer_module, msa_act, msa_mask_dict,
          safe_key=next(sub_keys), output_act=pair_act)

    pair_act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                               name='triangle_multiplication_outgoing'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    pair_act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                               name='triangle_multiplication_incoming'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))

    pair_act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_starting_node, gc,
                          name='triangle_attention_starting_node'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))
    pair_act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_ending_node, gc,
                          name='triangle_attention_ending_node'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))

    pair_act = dropout_wrapper_fn(
        Transition(c.pair_transition, gc, name='pair_transition'),
        pair_act,
        pair_mask,
        safe_key=next(sub_keys))

    return {'msa': msa_act, 'pair': pair_act}


####################################################################


# ===== multimer forward wrapper (merged from modules_multimer) =====
class AlphaFoldIteration(hk.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file.
  """

  def __init__(self, config, global_config, name='alphafold_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               batch,
               safe_key=None):


    # Compute representations for each MSA sample and average.
    embedding_module = EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)

    safe_key, safe_subkey = safe_key.split()
    representations = embedding_module(batch, safe_key=safe_subkey)

    self.representations = representations
    self.batch = batch
    self.heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
      if not head_config.weight:
        continue  # Do not instantiate zero-weight heads.

      head_factory = {
          'masked_msa':
              MaskedMsaHead,
          'distogram':
              DistogramHead,
          'structure_module':
              folding.StructureModule,
          'predicted_aligned_error':
              PredictedAlignedErrorHead,
          'predicted_lddt':
              PredictedLDDTHead,
          'experimentally_resolved':
              ExperimentallyResolvedHead,
      }[head_name]
      self.heads[head_name] = (head_config,
                               head_factory(head_config, self.global_config))

    # ColabDesign2: run the structure module only if its head is kept (nonzero
    # weight). A distogram-only objective (fixbb) drops it via heads=['distogram']
    # and saves the 8 IPA layers; the confidence heads that read its output are
    # already zero-weight and skipped below. (Also fixes a KeyError when the head
    # was dropped -- this line used to access it unconditionally.)
    if 'structure_module' in self.heads:
      _, fold_module = self.heads['structure_module']
      structure_module_output = fold_module(representations, batch)
    else:
      structure_module_output = None


    ret = {}
    ret['representations'] = representations

    for name, (head_config, module) in self.heads.items():
      if name == 'structure_module' and structure_module_output is not None:
        ret[name] = structure_module_output
        representations['structure_module'] = structure_module_output.pop('act')
      # Skip confidence heads until StructureModule is executed.
      elif name in {'predicted_lddt', 'predicted_aligned_error',
                    'experimentally_resolved'}:
        continue
      else:
        ret[name] = module(representations, batch)


    # Add confidence heads after StructureModule is executed.
    if self.config.heads.get('predicted_lddt.weight', 0.0):
      name = 'predicted_lddt'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch)

    if self.config.heads.experimentally_resolved.weight:
      name = 'experimentally_resolved'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch)

    if self.config.heads.get('predicted_aligned_error.weight', 0.0):
      name = 'predicted_aligned_error'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch)
      # Will be used for ipTM computation.
      ret[name]['asym_id'] = batch['asym_id']

    return ret

class AlphaFold(hk.Module):
  """AlphaFold-Multimer model with recycling.
  """

  def __init__(self, config, name='alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(
      self,
      batch,
      safe_key=None):

    c = self.config
    impl = AlphaFoldIteration(c, self.global_config)

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    elif isinstance(safe_key, jnp.ndarray):
      safe_key = prng.SafeKey(safe_key)

    assert isinstance(batch, dict)
    num_res = batch['aatype'].shape[0]
    
    def get_prev(ret, use_dgram=False):
      new_prev = {
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      # ColabDesign2: no structure module (distogram-only objective) -> no
      # coordinate-based recycling state. Only reachable at num_recycle=0
      # (make_config forbids dropping heads with recycling), so prev is unused.
      if 'structure_module' not in ret:
        return new_prev
      if use_dgram:
        if self.global_config.use_dgram_pred:
          dgram = jax.nn.softmax(ret["distogram"]["logits"])
          dgram_map = jax.nn.one_hot(jnp.repeat(jnp.append(0,jnp.arange(15)),4),15).at[:,0].set(0)
          new_prev['prev_dgram'] = dgram @ dgram_map
        else:
          pos = ret['structure_module']['final_atom_positions']
          prev_pseudo_beta = pseudo_beta_fn(batch['aatype'], pos)
          new_prev['prev_dgram'] = dgram_from_positions(prev_pseudo_beta, min_bin=3.25, max_bin=20.75, num_bins=15)
      else:
        new_prev['prev_pos'] = ret['structure_module']['final_atom_positions']

      return new_prev

    def apply_network(prev, safe_key):
      recycled_batch = {**batch, **prev}
      return impl(
          batch=recycled_batch,
          safe_key=safe_key)
    
    prev = batch.pop("prev")
    ret = apply_network(prev=prev, safe_key=safe_key)
    ret["prev"] = get_prev(ret, use_dgram="prev_dgram" in prev)
    
    return ret

class EmbeddingsAndEvoformer(hk.Module):
  """Embeds the input data and runs Evoformer.

  Produces the MSA, single and pair representations.
  """

  def __init__(self, config, global_config, name='evoformer'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def _relative_encoding(self, batch):
    """Add relative position encodings.

    For position (i, j), the value is (i-j) clipped to [-k, k] and one-hotted.

    Args:
      batch: batch.
    Returns:
      Feature embedding using the features as described before.
    """
    c = self.config
    gc = self.global_config
    rel_feats = []
    asym_id = batch['asym_id']
    asym_id_same = jnp.equal(asym_id[:, None], asym_id[None, :])

    if "offset" in batch:
      offset = batch['offset']
    else:
      pos = batch['residue_index']
      offset = pos[:, None] - pos[None, :]


    dtype = jnp.bfloat16 if gc.bfloat16 else jnp.float32

    # add residue index
    if c.pseudo_multimer:
      o = asym_id[:,None] - asym_id[None,:]
      offset_asym_id = jnp.where(o > 0, c.max_relative_idx, -c.max_relative_idx)
      offset = jnp.where(o == 0, offset, offset_asym_id)
      clipped_offset = jnp.clip(offset + c.max_relative_idx, 0, 2 * c.max_relative_idx)
      rel_pos = jax.nn.one_hot(clipped_offset, 2 * c.max_relative_idx + 2)
    else:
      clipped_offset = jnp.clip(offset + c.max_relative_idx, 0, 2 * c.max_relative_idx)
      final_offset = jnp.where(asym_id_same, clipped_offset,
                               (2 * c.max_relative_idx + 1) *
                               jnp.ones_like(clipped_offset))
      rel_pos = jax.nn.one_hot(final_offset, 2 * c.max_relative_idx + 2)
    
    rel_feats.append(rel_pos)

    # add entity info
    entity_id = jnp.zeros_like(batch['entity_id']) if c.pseudo_multimer else batch['entity_id']
    entity_id_same = jnp.equal(entity_id[:, None], entity_id[None, :])
    rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])

    # add symmetry info
    sym_id = jnp.zeros_like(batch['sym_id']) if c.pseudo_multimer else batch['sym_id']
    rel_sym_id = sym_id[:, None] - sym_id[None, :]
    max_rel_chain = c.max_relative_chain
    clipped_rel_chain = jnp.clip(rel_sym_id + max_rel_chain, 0, 2 * max_rel_chain)
    final_rel_chain = jnp.where(entity_id_same, clipped_rel_chain,
                                (2 * max_rel_chain + 1) *
                                jnp.ones_like(clipped_rel_chain))
    rel_chain = jax.nn.one_hot(final_rel_chain, 2 * c.max_relative_chain + 2)
    rel_feats.append(rel_chain)

    # combine features
    rel_feat = jnp.concatenate(rel_feats, axis=-1)
    rel_feat = rel_feat.astype(dtype)
    return common_modules.Linear(
        c.pair_channel,
        name='position_activations')(rel_feat)

  def __call__(self, batch, safe_key=None):

    c = self.config
    gc = self.global_config

    batch = dict(batch)
    dtype = jnp.bfloat16 if gc.bfloat16 else jnp.float32

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    output = {}
    with utils.bfloat16_context():
      
      msa_feat = batch['msa_feat'].astype(dtype)
      target_feat = jnp.pad(batch["target_feat"].astype(dtype),[[0,0],[0,1]])

      if c.query_bias:
        avg_target = target_feat
      else:
        print("query bias disabled")
        msa_mask = batch['msa_mask'].astype(dtype)
        avg_target = jnp.sum(msa_feat[:,:,:20],0) / (jnp.sum(msa_mask[:,:,None],0) + 1e-8)
        avg_target = jnp.pad(avg_target,[[0,0],[0,1]])        

      preprocess_1d = common_modules.Linear(c.msa_channel, name='preprocess_1d')(avg_target)
      preprocess_1d = jnp.where(target_feat.sum(-1,keepdims=True) == 0, 0, preprocess_1d)

      preprocess_msa = common_modules.Linear(c.msa_channel, name='preprocess_msa')(msa_feat)
      msa_activations = preprocess_1d[None] + preprocess_msa
      num_msa_sequences = msa_activations.shape[0]

      left_single = common_modules.Linear(c.pair_channel, name='left_single')(target_feat)
      right_single = common_modules.Linear(c.pair_channel, name='right_single')(target_feat)
      pair_activations = left_single[:, None] + right_single[None]
      mask_2d = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]

      # allow for custom mask_2d
      if "mask_2d" in batch:
        mask_2d = jnp.where(batch["mask_2d"], mask_2d, 0)

      mask_2d = mask_2d.astype(dtype)

      if c.recycle_pos:
        if "prev_dgram" in batch:
          dgram = batch["prev_dgram"]
        else:        
          prev_pseudo_beta = pseudo_beta_fn(batch['aatype'], batch['prev_pos'])
          dgram = dgram_from_positions(prev_pseudo_beta, **self.config.prev_pos)
        
        dgram = dgram.astype(dtype)
        pair_activations += common_modules.Linear(c.pair_channel, name='prev_pos_linear')(dgram)

      if c.recycle_features:
        prev_msa_first_row = common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name='prev_msa_first_row_norm')(batch['prev_msa_first_row']).astype(dtype)
        
        msa_activations = msa_activations.at[0].add(prev_msa_first_row)

        pair_activations += common_modules.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name='prev_pair_norm')(batch['prev_pair']).astype(dtype)

      if c.max_relative_idx:
        pair_activations += self._relative_encoding(batch)

      if c.template.enabled:
        template_module = TemplateEmbedding(c.template, gc)
        template_batch = {
            'template_aatype': batch['template_aatype'],
            'template_all_atom_positions': batch['template_all_atom_positions'],
            'template_all_atom_mask': batch['template_all_atom_mask']
        }
        if "template_dgram" in batch:
          template_batch["template_dgram"] = batch["template_dgram"]

        
        if "interchain_mask" in batch:
          multichain_mask = batch["interchain_mask"]
        
        else:
          # Construct a mask such that only intra-chain template features are
          # computed, since all templates are for each chain individually.
          multichain_mask = batch['asym_id'][:, None] == batch['asym_id'][None, :]

        if "mask_template_interchain" in batch:
          multichain_mask = jnp.where(batch["mask_template_interchain"], multichain_mask, True)
        

        safe_key, safe_subkey = safe_key.split()
        template_act = template_module(
            query_embedding=pair_activations,
            template_batch=template_batch,
            padding_mask_2d=mask_2d,
            multichain_mask_2d=multichain_mask,
            use_dropout=batch["use_dropout"],
            safe_key=safe_subkey)
        pair_activations += template_act

      # Extra MSA stack.
      extra_msa_feat = batch["extra_msa_feat"]
      extra_msa_mask = batch["extra_msa_mask"]
      extra_msa_activations = common_modules.Linear(c.extra_msa_channel,
                                                    name='extra_msa_activations')(extra_msa_feat).astype(dtype)
      extra_msa_mask = extra_msa_mask.astype(dtype)
      extra_evoformer_input = {'msa': extra_msa_activations, 'pair': pair_activations}
      extra_masks = {'msa': extra_msa_mask, 'pair': mask_2d,
                     'opm_first': batch.get('opm_first', jnp.float32(1.0))}
      if "cov_mask" in batch:
        extra_masks["cov"] = batch["cov_mask"].astype(dtype)

      extra_evoformer_iteration = EvoformerIteration(c.evoformer, gc, is_extra_msa=True, name='extra_msa_stack')

      def extra_evoformer_fn(x):
        act, safe_key = x
        safe_key, safe_subkey = safe_key.split()
        extra_evoformer_output = extra_evoformer_iteration(
            activations=act,
            masks=extra_masks,
            use_dropout=batch["use_dropout"],
            safe_key=safe_subkey)
        return (extra_evoformer_output, safe_key)

      if gc.use_remat:
        extra_evoformer_fn = hk.remat(extra_evoformer_fn)

      safe_key, safe_subkey = safe_key.split()
      extra_evoformer_stack = layer_stack.layer_stack(
          c.extra_msa_stack_num_block)(
              extra_evoformer_fn)
      extra_evoformer_output, safe_key = extra_evoformer_stack(
          (extra_evoformer_input, safe_subkey))

      pair_activations = extra_evoformer_output['pair']
      # Get the size of the MSA before potentially adding templates, so we
      # can crop out the templates later.
      num_msa_sequences = msa_activations.shape[0]
      evoformer_input = {
          'msa': msa_activations,
          'pair': pair_activations,
      }
      evoformer_masks = {'msa': batch['msa_mask'].astype(dtype), 'pair': mask_2d,
                         'opm_first': batch.get('opm_first', jnp.float32(1.0))}      
      if "cov_mask" in batch:
        evoformer_masks["cov"] = batch['cov_mask'].astype(dtype)

      if c.template.enabled:
        template_features, template_masks = (
            template_embedding_1d(batch=batch, num_channel=c.msa_channel, global_config=gc))

        evoformer_input['msa'] = jnp.concatenate([evoformer_input['msa'], template_features], axis=0)
        evoformer_masks['msa'] = jnp.concatenate([evoformer_masks['msa'], template_masks], axis=0)
        
      evoformer_iteration = EvoformerIteration(
          c.evoformer, gc, is_extra_msa=False, name='evoformer_iteration')
    
      def evoformer_fn(x):
        act, safe_key = x
        safe_key, safe_subkey = safe_key.split()
        evoformer_output = evoformer_iteration(
            activations=act,
            masks=evoformer_masks,
            use_dropout=batch["use_dropout"],
            safe_key=safe_subkey)
        return (evoformer_output, safe_key)

      if gc.use_remat:
        evoformer_fn = hk.remat(evoformer_fn)

      safe_key, safe_subkey = safe_key.split()
      evoformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(
          evoformer_fn)

      def run_evoformer(evoformer_input):
        evoformer_output, _ = evoformer_stack((evoformer_input, safe_subkey))
        return evoformer_output

      evoformer_output = run_evoformer(evoformer_input)

      msa_activations = evoformer_output['msa']
      pair_activations = evoformer_output['pair']
      
      single_activations = common_modules.Linear(
          c.seq_channel, name='single_activations')(msa_activations[0])
    output.update({
        'single':
            single_activations,
        'pair':
            pair_activations,
        # Crop away template rows such that they are not used in MaskedMsaHead.
        'msa':
            msa_activations[:num_msa_sequences, :, :],
        'msa_first_row':
            msa_activations[0],
    })

    # Convert back to float32 if we're not saving memory.
    if not gc.bfloat16_output:
      for k, v in output.items():
        if v.dtype == jnp.bfloat16:
          output[k] = v.astype(jnp.float32)

    return output


class TemplateEmbedding(hk.Module):
  """Embed a set of templates."""

  def __init__(self, config, global_config, name='template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, query_embedding, template_batch, padding_mask_2d,
               multichain_mask_2d, use_dropout, safe_key=None):
    """Generate an embedding for a set of templates.

    Args:
      query_embedding: [num_res, num_res, num_channel] a query tensor that will
        be used to attend over the templates to remove the num_templates
        dimension.
      template_batch: A dictionary containing:
        `template_aatype`: [num_templates, num_res] aatype for each template.
        `template_all_atom_positions`: [num_templates, num_res, 37, 3] atom
          positions for all templates.
        `template_all_atom_mask`: [num_templates, num_res, 37] mask for each
          template.
      padding_mask_2d: [num_res, num_res] Pair mask for attention operations.
      multichain_mask_2d: [num_res, num_res] Mask indicating which residue pairs
        are intra-chain, used to mask out residue distance based features
        between chains.
      safe_key: random key generator.

    Returns:
      An embedding of size [num_res, num_res, num_channels]
    """
    c = self.config
    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    num_templates = template_batch['template_aatype'].shape[0]
    num_res, _, query_num_channels = query_embedding.shape

    # Embed each template separately.
    template_embedder = SingleTemplateEmbedding(self.config, self.global_config)
    def partial_template_embedder(template_batch, unsafe_key):
      safe_key = prng.SafeKey(unsafe_key)
      return template_embedder(query_embedding,
                               template_batch,
                               padding_mask_2d,
                               multichain_mask_2d,
                               use_dropout,
                               safe_key)

    safe_key, unsafe_key = safe_key.split()
    unsafe_keys = jax.random.split(unsafe_key._key, num_templates)

    def scan_fn(carry, x):
      return carry + partial_template_embedder(*x), None

    scan_init = jnp.zeros((num_res, num_res, c.num_channels), dtype=query_embedding.dtype)
    summed_template_embeddings, _ = hk.scan(scan_fn, scan_init, (template_batch, unsafe_keys))

    embedding = summed_template_embeddings / num_templates
    embedding = jax.nn.relu(embedding)
    embedding = common_modules.Linear(
        query_num_channels,
        initializer='relu',
        name='output_linear')(embedding)

    return embedding


class SingleTemplateEmbedding(hk.Module):
  """Embed a single template."""

  def __init__(self, config, global_config, name='single_template_embedding'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, query_embedding, template_batch,               
               padding_mask_2d, multichain_mask_2d, use_dropout, safe_key):
    """Build the single template embedding graph.

    Args:
      query_embedding: (num_res, num_res, num_channels) - embedding of the
        query sequence/msa.
      template_aatype: [num_res] aatype for each template.
      template_all_atom_positions: [num_res, 37, 3] atom positions for all
        templates.
      template_all_atom_mask: [num_res, 37] mask for each template.
      padding_mask_2d: Padding mask (Note: this doesn't care if a template
        exists, unlike the template_pseudo_beta_mask).
      multichain_mask_2d: A mask indicating intra-chain residue pairs, used
        to mask out between chain distances/features when templates are for
        single chains.
      safe_key: Random key generator.

    Returns:
      A template embedding (num_res, num_res, num_channels).
    """
    gc = self.global_config
    c = self.config
    assert padding_mask_2d.dtype == query_embedding.dtype
    dtype = query_embedding.dtype
    num_channels = self.config.num_channels
    
    def construct_input(query_embedding, template_batch, multichain_mask_2d):

      if "template_dgram" in template_batch:
        template_dgram = template_batch["template_dgram"].astype(dtype)
        template_dgram *= multichain_mask_2d[...,None]
        pseudo_beta_mask_2d = template_dgram.sum(-1)
      
      else:
        # Compute distogram feature for the template.
        template_positions, pseudo_beta_mask = pseudo_beta_fn(
            template_batch["template_aatype"],
            template_batch["template_all_atom_positions"],
            template_batch["template_all_atom_mask"])
        pseudo_beta_mask_2d = (pseudo_beta_mask[:, None] *
                               pseudo_beta_mask[None, :])
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram = dgram_from_positions(
            template_positions, **self.config.dgram_features)
        template_dgram *= pseudo_beta_mask_2d[..., None]
        template_dgram = template_dgram.astype(dtype)    
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)

      to_concat = [(template_dgram, 1), (pseudo_beta_mask_2d, 0)]
      aatype = jax.nn.one_hot(template_batch["template_aatype"], 22, axis=-1, dtype=dtype)
      to_concat.append((aatype[None, :, :], 1))
      to_concat.append((aatype[:, None, :], 1))

      # Compute a feature representing the normalized vector between each
      # backbone affine - i.e. in each residues local frame, what direction are
      # each of the other residues.
      raw_atom_pos = template_batch["template_all_atom_positions"]
      if gc.bfloat16:
        raw_atom_pos = raw_atom_pos.astype(jnp.float32)
        
      atom_pos = geometry.Vec3Array.from_array(raw_atom_pos)
      rigid, backbone_mask = folding.make_backbone_affine(
          atom_pos,
          template_batch["template_all_atom_mask"],
          template_batch["template_aatype"])
      points = rigid.translation
      rigid_vec = rigid[:, None].inverse().apply_to_point(points)
      unit_vector = rigid_vec.normalized()
      unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

      if gc.bfloat16:
        unit_vector = [x.astype(jnp.bfloat16) for x in unit_vector]
        backbone_mask = backbone_mask.astype(jnp.bfloat16)

      backbone_mask_2d = jnp.sqrt(backbone_mask[:,None] * backbone_mask[None,:])
      backbone_mask_2d *= multichain_mask_2d
      unit_vector = [x*backbone_mask_2d for x in unit_vector]

      # Note that the backbone_mask takes into account C, CA and N (unlike
      # pseudo beta mask which just needs CB) so we add both masks as features.
      to_concat.extend([(x, 0) for x in unit_vector])
      to_concat.append((backbone_mask_2d, 0))          
      query_embedding = common_modules.LayerNorm(
          axis=[-1],
          create_scale=True,
          create_offset=True,
          name='query_embedding_norm')(query_embedding)
      # Allow the template embedder to see the query embedding.  Note this
      # contains the position relative feature, so this is how the network knows
      # which residues are next to each other.
      to_concat.append((query_embedding, 1))

      act = 0

      for i, (x, n_input_dims) in enumerate(to_concat):
        act += common_modules.Linear(
            num_channels,
            num_input_dims=n_input_dims,
            initializer='relu',
            name=f'template_pair_embedding_{i}')(x)
      return act

    act = construct_input(query_embedding, template_batch, multichain_mask_2d)

    template_iteration = TemplateEmbeddingIteration(
        c.template_pair_stack, gc, name='template_embedding_iteration')

    def template_iteration_fn(x):
      act, safe_key = x

      safe_key, safe_subkey = safe_key.split()
      act = template_iteration(
          act=act,
          pair_mask=padding_mask_2d,
          use_dropout=use_dropout,
          safe_key=safe_subkey)
      return (act, safe_key)

    if gc.use_remat:
      template_iteration_fn = hk.remat(template_iteration_fn)

    safe_key, safe_subkey = safe_key.split()
    template_stack = layer_stack.layer_stack(
        c.template_pair_stack.num_block)(
            template_iteration_fn)
    act, safe_key = template_stack((act, safe_subkey))

    act = common_modules.LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='output_layer_norm')(
            act)
    return act

class TemplateEmbeddingIteration(hk.Module):
  """Single Iteration of Template Embedding."""

  def __init__(self, config, global_config,
               name='template_embedding_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, pair_mask, use_dropout, safe_key=None):
    """Build a single iteration of the template embedder.

    Args:
      act: [num_res, num_res, num_channel] Input pairwise activations.
      pair_mask: [num_res, num_res] padding mask.
      safe_key: Safe pseudo-random generator key.

    Returns:
      [num_res, num_res, num_channel] tensor of activations.
    """
    c = self.config
    gc = self.global_config

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())

    dropout_wrapper_fn = functools.partial(
        dropout_wrapper,
        use_dropout=use_dropout,
        global_config=gc)

    safe_key, *sub_keys = safe_key.split(20)
    sub_keys = iter(sub_keys)

    act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_outgoing, gc,
                                       name='triangle_multiplication_outgoing'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        TriangleMultiplication(c.triangle_multiplication_incoming, gc,
                                       name='triangle_multiplication_incoming'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_starting_node, gc,
                                  name='triangle_attention_starting_node'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        TriangleAttention(c.triangle_attention_ending_node, gc,
                                  name='triangle_attention_ending_node'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    act = dropout_wrapper_fn(
        Transition(c.pair_transition, gc,
                           name='pair_transition'),
        act,
        pair_mask,
        safe_key=next(sub_keys))

    return act


def template_embedding_1d(batch, num_channel, global_config):
  """Embed templates into an (num_res, num_templates, num_channels) embedding.

  Args:
    batch: A batch containing:
      template_aatype, (num_templates, num_res) aatype for the templates.
      template_all_atom_positions, (num_templates, num_residues, 37, 3) atom
        positions for the templates.
      template_all_atom_mask, (num_templates, num_residues, 37) atom mask for
        each template.
    num_channel: The number of channels in the output.

  Returns:
    An embedding of shape (num_templates, num_res, num_channels) and a mask of
    shape (num_templates, num_res).
  """

  # Embed the templates aatypes.
  aatype_one_hot = jax.nn.one_hot(batch['template_aatype'], 22, axis=-1)

  num_templates = batch['template_aatype'].shape[0]
  all_chi_angles = []
  all_chi_masks = []
  for i in range(num_templates):
    atom_pos = geometry.Vec3Array.from_array(
        batch['template_all_atom_positions'][i, :, :, :])
    template_chi_angles, template_chi_mask = all_atom.compute_chi_angles(
        atom_pos,
        batch['template_all_atom_mask'][i, :, :],
        batch['template_aatype'][i, :])
    all_chi_angles.append(template_chi_angles)
    all_chi_masks.append(template_chi_mask)
  chi_angles = jnp.stack(all_chi_angles, axis=0)
  chi_mask = jnp.stack(all_chi_masks, axis=0)

  template_features = jnp.concatenate([
      aatype_one_hot,
      jnp.sin(chi_angles) * chi_mask,
      jnp.cos(chi_angles) * chi_mask,
      chi_mask], axis=-1)

  template_mask = chi_mask[:, :, 0]

  if global_config.bfloat16:
    template_features = template_features.astype(jnp.bfloat16)
    template_mask = template_mask.astype(jnp.bfloat16)

  template_activations = common_modules.Linear(
      num_channel,
      initializer='relu',
      name='template_single_embedding')(
          template_features)
  template_activations = jax.nn.relu(template_activations)
  template_activations = common_modules.Linear(
      num_channel,
      initializer='relu',
      name='template_projection')(
          template_activations)
  return template_activations, template_mask

