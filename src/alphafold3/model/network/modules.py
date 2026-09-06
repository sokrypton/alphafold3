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

"""Haiku modules for the Diffuser model."""

from collections.abc import Sequence
from typing import Literal

from alphafold3.common import base_config
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import mapping
from . import diffusion_transformer
import haiku as hk
import jax
import jax.numpy as jnp
import tokamax


def _pair_dropout(act, use_dropout, rate=0.25, columnwise=False):
  """ColabDesign2: shared row/column-wise dropout on a pair tensor (i, j, c).

  AF3 was TRAINED with dropout in the pair stack (SI Algorithms 8/16/17, sec 5.5;
  cross-checked against OpenFold3): DropoutRowwise 0.25 on TriangleMultiplication
  outgoing/incoming and TriangleAttentionStartingNode, DropoutColumnwise 0.25 on
  TriangleAttentionEndingNode. The released inference code stripped it. This adds
  it back so a design run can use it as a stochastic regulariser, the way the AF2
  design path does.

  No-recompile toggle, exactly as AF2's apply_dropout: rate is jnp.where(
  use_dropout, rate, 0), so the op is always in the graph and OFF is an exact
  identity -- bernoulli(keep_rate=1.0) is all-ones and act*1/1 == act -- meaning
  prediction stays bit-identical and toggling never triggers a recompile.

  The mask is SHARED along one axis of the pair tensor (a whole row's or column's
  worth of channels are dropped together): row-wise shares over axis 0 (i),
  column-wise over axis 1 (j). Uses hk.next_rng_key(), which hk layer_stack
  splits per block.
  """
  r = jnp.where(use_dropout, rate, 0.0)
  keep_rate = 1.0 - r
  shape = list(act.shape)
  shape[1 if columnwise else 0] = 1
  keep = jax.random.bernoulli(hk.next_rng_key(), keep_rate, shape=shape)
  return act * keep / keep_rate


def get_shard_size(
    num_residues: int, shard_spec: Sequence[tuple[int | None, int | None]]
) -> int | None:
  shard_size = shard_spec[0][-1]
  for num_residues_upper_bound, num_residues_shard_size in shard_spec:
    shard_size = num_residues_shard_size
    if (
        num_residues_upper_bound is None
        or num_residues <= num_residues_upper_bound
    ):
      break
  return shard_size


class TransitionBlock(hk.Module):
  """Transition block for transformer."""

  class Config(base_config.BaseConfig):
    num_intermediate_factor: int = 4
    use_glu_kernel: bool = True

  def __init__(
      self, config: Config, global_config: model_config.GlobalConfig, *, name
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, broadcast_dim=0):
    num_channels = act.shape[-1]

    num_intermediate = int(num_channels * self.config.num_intermediate_factor)

    act = hm.LayerNorm(name='input_layer_norm')(act)

    if self.config.use_glu_kernel:
      weights, _ = hm.haiku_linear_get_params(
          act,
          num_output=num_intermediate * 2,
          initializer='relu',
          name='transition1',
      )
      weights = jnp.reshape(weights, (len(weights), 2, num_intermediate))
      c = tokamax.gated_linear_unit(
          x=act, weights=weights, activation=jax.nn.swish
      )
    else:
      act = hm.Linear(
          num_intermediate * 2, initializer='relu', name='transition1'
      )(act)
      a, b = jnp.split(act, 2, axis=-1)
      c = jax.nn.swish(a) * b

    return hm.Linear(
        num_channels,
        initializer=self.global_config.final_init,
        name='transition2',
    )(c)


class MSAAttention(hk.Module):
  """MSA Attention."""

  class Config(base_config.BaseConfig):
    num_head: int = 8
    # per-head value width. None keeps AF3's coupling value_dim = c_m // num_head
    # (so this is a no-op for AF3/OF3/IF2). OpenDDE decouples it -- its
    # MSAPairWeightedAveraging uses c=8 with c_m=128, num_head=8 (hidden 64), which
    # AF3's coupling (=16) cannot express; set value_dim=8 to load those weights.
    value_dim: int | None = None

  def __init__(
      self, config: Config, global_config: model_config.GlobalConfig, *, name
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, mask, pair_act, pair_mask=None):
    chai1 = self.global_config.model == 'chai1'
    act = hm.LayerNorm(name='act_norm')(act)
    pair_act = hm.LayerNorm(name='pair_norm')(pair_act)
    logits = hm.Linear(
        self.config.num_head, use_bias=False, name='pair_logits'
    )(pair_act)
    logits = jnp.transpose(logits, [2, 0, 1])
    if chai1 and pair_mask is not None:
      # chai masks these logits with the TOKEN PAIR mask (fill -10000), where we
      # derive a per-token mask from the MSA (max over rows). Read off its graph
      # rather than inferred -- trunk forward_256 node 10767.
      logits = jnp.where(pair_mask[None].astype(jnp.bool_), logits, -10000.0)
    else:
      logits += 1e9 * (jnp.max(mask, axis=0) - 1.0)
    weights = jax.nn.softmax(logits, axis=-1)
    num_channels = act.shape[-1]
    value_dim = self.config.value_dim or num_channels // self.config.num_head
    v = hm.Linear(
        [self.config.num_head, value_dim], use_bias=False, name='v_projection'
    )(act)
    if chai1:
      # chai ZEROES the value wherever the MSA mask is false (node 10818) before
      # the weighted average; we averaged those positions in. The einsum
      # contracts over KEY TOKENS, so an uncovered position in an otherwise real
      # row still contributed -- which is most of a cropped MSA.
      v = v * mask[..., None, None].astype(v.dtype)
    v_avg = jnp.einsum('hqk, bkhc -> bqhc', weights, v)
    v_avg = jnp.reshape(v_avg, v_avg.shape[:-2] + (-1,))
    gate_values = hm.Linear(
        self.config.num_head * value_dim,
        bias_init=1.0,
        initializer='zeros',
        name='gating_query',
    )(act)
    v_avg *= jax.nn.sigmoid(gate_values)

    return hm.Linear(
        num_channels,
        initializer=self.global_config.final_init,
        name='output_projection',
    )(v_avg)


class GridSelfAttention(hk.Module):
  """Self attention that is either per-sequence or per-residue."""

  class Config(base_config.BaseConfig):
    num_head: int = 4
    # Per-head dim. None -> derived (max(channels//num_head, 16)), the AF3 default. Boltz's
    # template pairformer fixes it (pairwise_head_width=32) independent of channels (64), so a
    # boltz2 template branch sets qkv_dim=32; leaving it None keeps the trunk bit-identical.
    qkv_dim: int | None = None
    # chai-1's CONFIDENCE triangle attention only. Its linear_out is
    # (2*c_z, 2*c_z), not the trunk's (c_z, 2*c_z): each direction's output
    # reads BOTH directions' heads and the two halves combine as
    # `out0 + transpose(out1)`. So each direction needs TWO output projections
    # -- one kept, one transposed -- instead of one. The cross-direction blocks
    # are not negligible (Frobenius 24-30, same as the diagonal), so dropping
    # them is not an option: it costs corr 0.831 against chai's own output.
    dual_output: bool = False

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      transpose: bool,
      *,
      name: str,
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.transpose = transpose

  @hk.transparent
  def _attention(
      self,
      act,
      mask,
      bias,
  ):
    num_channels = act.shape[-1]
    assert num_channels % self.config.num_head == 0
    # Triton requires a minimum dimension of 16 for doing matmul.
    qkv_dim = self.config.qkv_dim or max(num_channels // self.config.num_head, 16)

    qkv_shape = (self.config.num_head, qkv_dim)
    q = hm.Linear(
        qkv_shape, use_bias=False, name='q_projection', transpose_weights=True
    )(act)
    k = hm.Linear(
        qkv_shape, use_bias=False, name='k_projection', transpose_weights=True
    )(act)
    v = hm.Linear(qkv_shape, use_bias=False, name='v_projection')(act)

    # Dot product attention requires the bias term to have a batch dimension.
    bias = jnp.expand_dims(bias, 0)

    from alphafold3.model.components.attention import dot_product_attention as _flash_dpa
    weighted_avg = _flash_dpa(
        q,
        k,
        v,
        mask=mask,
        bias=bias,
        implementation=self.global_config.flash_attention_implementation,
    )
    weighted_avg = jnp.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))

    # RF3's triangle-attention gate and output projections carry trained biases
    # (to_g.bias init 1.0 -> gate open; to_out.bias) that stock AF3 GridSelfAttention
    # lacks. Dropping them halves the gate (weight inits to zeros, so the gate is
    # bias-dominated) and compounds across the 96 tri-attentions. Gated on
    # rosettafold3 so other families stay bias-free / byte-identical.
    rosettafold3 = self.global_config.model == 'rosettafold3'
    gate_values = hm.Linear(
        self.config.num_head * qkv_dim,
        use_bias=rosettafold3,
        bias_init=1.0,
        initializer='zeros',
        transpose_weights=True,
        name='gating_query',
    )(act)
    weighted_avg *= jax.nn.sigmoid(gate_values)

    out = hm.Linear(
        num_channels,
        use_bias=rosettafold3,
        initializer=self.global_config.final_init,
        name='output_projection',
    )(weighted_avg)
    if self.config.dual_output:
      # concatenated rather than returned as a tuple: this runs under
      # mapping.inference_subbatch, which chunks a single array.
      out = jnp.concatenate([out, hm.Linear(
          num_channels,
          use_bias=rosettafold3,
          initializer=self.global_config.final_init,
          name='output_projection_transposed',
      )(weighted_avg)], axis=-1)
    return out

  def __call__(self, act, pair_mask):
    """Builds a module.

    Arguments:
      act: [num_seq, num_res, channels] activations tensor
      pair_mask: [num_seq, num_res] mask of non-padded regions in the tensor.
        Only used in inducing points attention currently.

    Returns:
      Result of the self-attention operation.
    """
    assert len(act.shape) == 3
    assert len(pair_mask.shape) == 2

    pair_mask = jnp.swapaxes(pair_mask, -1, -2)
    act = hm.LayerNorm(name='act_norm')(act)

    nonbatched_bias = hm.Linear(
        self.config.num_head, use_bias=False, name='pair_bias_projection'
    )(act)
    nonbatched_bias = jnp.transpose(nonbatched_bias, [2, 0, 1])
    # OpenFold3 computes the pair bias from transposed z for column-wise attention;
    # swap axes to match that convention for its lineage. RoseTTAFold3's
    # TriangleAttention (end-node) computes to_b from the NON-transposed pair and does
    # NOT transpose the bias (it transposes only the pair before attention), so it is
    # excluded from the list below.
    # 'openbind0' (OpenFold3 >= 0.5.0) is deliberately ABSENT, following AF3's own
    # Algorithm 15 -- and this is the one openbind convention still taken on
    # reading rather than measurement. Upstream made the end-node transposition
    # explicit in v0.5.0 (TriangularAttention gained `transpose_bias`, passed True
    # from tri_att_start_end), but their release notes do not mention it and
    # NEITHER SETTING CHANGES ANY WEIGHT -- linear_z is still per block and
    # identically shaped -- so no shape gate and no coverage audit can see it.
    #
    # FOLDING CANNOT DISCRIMINATE IT EITHER, measured rather than assumed:
    #
    #   6MRR, single sequence     1.702 A (absent)  vs 1.712 A (present)
    #   1STP + MSA, seed 1        0.548 A           vs 0.763 A
    #   1STP + MSA, seed 7        0.532 A           vs 0.460 A
    #
    # The two seeds disagree about which is better, so the gap is inside the
    # sampling spread and one seed would have "confirmed" either answer. What
    # would settle it is an activation-level comparison against native OpenFold3
    # running openbind -- the trunk pair bias, before it is averaged away by a
    # diffusion sample. Until then this is the documented default, not a result.
    if (self.transpose
        and self.global_config.model in model_config.TRANSPOSED_COLUMN_PAIR_BIAS):
      nonbatched_bias = jnp.swapaxes(nonbatched_bias, -1, -2)

    num_residues = act.shape[0]

    chunk_size = get_shard_size(
        num_residues, self.global_config.pair_attention_chunk_size
    )

    if self.transpose:
      act = jnp.swapaxes(act, -2, -3)

    pair_mask = pair_mask[:, None, None, :].astype(jnp.bool_)

    act = mapping.inference_subbatch(
        self._attention,
        chunk_size,  # pyrefly: ignore[bad-argument-type]
        batched_args=[act, pair_mask],
        nonbatched_args=[nonbatched_bias],
    )

    # chai-1 does NOT transpose the ending-node direction's output back. Its two
    # directions are one module: the outputs are concatenated as
    # [dir0(i,j), dir1(j,i)] and a single linear_out reads them in that mixed
    # orientation. We hold the two halves of that linear_out in the two
    # pair_attention scopes, so this one must hand back the transposed
    # orientation for the sum to reproduce chai's. Restoring it here caps the
    # op at corr 0.78 against the native reference; skipping it gives 0.9991.
    if self.transpose and self.global_config.model != 'chai1':
      act = jnp.swapaxes(act, -2, -3)

    if self.config.dual_output:
      # (kept, to-be-transposed); the caller sums each group before the one
      # transpose, exactly as chai does at its nodes 856/861.
      return jnp.split(act, 2, axis=-1)
    return act


class TriangleMultiplication(hk.Module):
  """Triangle Multiplication."""

  class Config(base_config.BaseConfig):
    equation: Literal['ikc,jkc->ijc', 'kjc,kic->ijc']
    use_glu_kernel: bool = True
    # Width of the a/b projections and the einsum, when it is NOT the input
    # channel count. AF3 always ties the two, and every model here agrees except
    # Protenix-v1's TEMPLATE stack, which runs a 128-wide triangle multiplication
    # on a 64-channel template pair (the two projections are (128, 64) where AF3
    # would build (64, 64)). None keeps AF3's behaviour exactly.
    hidden_dim: int | None = None

  def __init__(
      self, config: Config, global_config: model_config.GlobalConfig, *, name
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, act, mask):
    """Applies Module.

    Args:
      act: The activation.
      mask: The mask.

    Returns:
      Outputs, should have same shape/type as output_act
    """
    mask = mask[None, ...]
    num_channels = act.shape[-1]
    # the a/b projections, the einsum and center_norm all run at hidden_dim;
    # only output_projection and the gate come back to num_channels
    hidden_dim = (num_channels if self.config.hidden_dim is None
                  else self.config.hidden_dim)
    equation = {
        'ikc,jkc->ijc': 'cik,cjk->cij',
        'kjc,kic->ijc': 'ckj,cki->cij',
    }[self.config.equation]

    act = hm.LayerNorm(name='left_norm_input')(act)
    input_act = act

    if self.config.use_glu_kernel:
      weights_projection, _ = hm.haiku_linear_get_params(
          act, num_output=hidden_dim * 2, name='projection'
      )
      weights_gate, _ = hm.haiku_linear_get_params(
          act,
          num_output=hidden_dim * 2,
          initializer=self.global_config.final_init,
          name='gate',
      )
      weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)

      projection = tokamax.gated_linear_unit(
          act, weights_glu, activation=jax.nn.sigmoid
      )
      projection = jnp.transpose(projection, (2, 0, 1))
      projection *= mask
    else:
      projection = hm.Linear(hidden_dim * 2, name='projection')(act)
      projection = jnp.transpose(projection, (2, 0, 1))
      projection *= mask

      gate = hm.Linear(
          hidden_dim * 2,
          name='gate',
          bias_init=1.0,
          initializer=self.global_config.final_init,
      )(act)
      gate = jnp.transpose(gate, (2, 0, 1))
      projection *= jax.nn.sigmoid(gate)

    projection = projection.reshape(hidden_dim, 2, *projection.shape[1:])
    a, b = jnp.split(projection, 2, axis=1)
    a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)
    act = jnp.einsum(equation, a, b)
    act = hm.LayerNorm(name='center_norm', axis=0, param_axis=0)(act)

    act = jnp.transpose(act, (1, 2, 0))
    act = hm.Linear(
        num_channels,
        initializer=self.global_config.final_init,
        name='output_projection',
    )(act)

    gate_out = hm.Linear(
        num_channels,
        name='gating_linear',
        bias_init=1.0,
        initializer=self.global_config.final_init,
    )(input_act)
    act *= jax.nn.sigmoid(gate_out)

    return act


class OuterProductMean(hk.Module):
  """Computed mean outer product."""

  class Config(base_config.BaseConfig):
    chunk_size: int = 128
    num_outer_channel: int = 32

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      num_output_channel,
      *,
      name,
  ):
    super().__init__(name=name)
    self.global_config = global_config
    self.config = config
    self.num_output_channel = num_output_channel

  def __call__(self, act, mask):
    mask = mask[..., None]
    act = hm.LayerNorm(name='layer_norm_input')(act)

    if self.global_config.model == 'chai1':
      return self._chai_grouped(act, mask)

    # RF3's proj_left/proj_right are nn.Linear, i.e. WITH bias, and both are trained
    # away from their zero init (|b| 0.080 / 0.108). AF3's are bias-free and have no
    # slot for them. Dropping them is not a constant offset either: the outer product
    # is bilinear, so a dropped bias loses the two cross terms b_l (x) W_r x_m and
    # W_l x_l (x) b_r as well. It held the MSA module's outer product to corr 0.924
    # against native with otherwise near-exact inputs.
    lr_bias = self.global_config.model == 'rosettafold3'

    left_act = mask * hm.Linear(
        self.config.num_outer_channel,
        initializer='linear',
        use_bias=lr_bias,
        name='left_projection',
    )(act)

    right_act = mask * hm.Linear(
        self.config.num_outer_channel,
        initializer='linear',
        use_bias=lr_bias,
        name='right_projection',
    )(act)

    if self.global_config.final_init == 'zeros':
      w_init = hk.initializers.Constant(0.0)
    else:
      w_init = hk.initializers.VarianceScaling(scale=2.0, mode='fan_in')

    output_w = hk.get_parameter(
        'output_w',
        shape=(
            self.config.num_outer_channel,
            self.config.num_outer_channel,
            self.num_output_channel,
        ),
        dtype=act.dtype,
        init=w_init,
    )
    output_b = hk.get_parameter(
        'output_b',
        shape=(self.num_output_channel,),
        dtype=act.dtype,
        init=hk.initializers.Constant(0.0),
    )

    def compute_chunk(left_act):
      # Make sure that the 'b' dimension is the most minor batch like dimension
      # so it will be treated as the real batch by XLA (both during the forward
      # and the backward pass)
      out = jnp.einsum('abc,ade,cef->bdf', left_act, right_act, output_w)
      return out + output_b

    act = mapping.inference_subbatch(
        compute_chunk,
        self.config.chunk_size,
        batched_args=[left_act],
        nonbatched_args=[],
        input_subbatch_dim=1,
        output_subbatch_dim=0,
    )

    norm = jnp.einsum('abc,adc->bdc', mask, mask)
    if self.global_config.model in model_config.CLAMPED_OPM_NORM:
      # ESMFold2 clamps the pair count at 1 instead of adding AF3's 1e-3. The
      # two agree to 1/(1 + eps/n), which is 3.3e-04 at depth 3 and 1e-03 at
      # DEPTH 1 -- and depth 1 is where ESMFold2 normally runs, so this is a
      # systematic 0.1% scale on the entire OPM contribution rather than noise.
      return act / jnp.maximum(norm, 1.0)
    epsilon = 1e-3
    return act / (epsilon + norm)

  @hk.transparent
  def _chai_grouped(self, act, mask):
    """chai-1's grouped outer product mean.

    AF3 projects the MSA to `num_outer_channel` on each side and takes one
    C x C outer product. chai projects to G GROUPS of K channels and takes the
    outer product WITHIN each group: its `weight_ab (2, G, K, c_m)` einsums
    (`abc,defc->abdef` per side, then `abcde,afcdg->cegabf`) contract only the
    MSA-depth axis and BROADCAST the group index, giving G*K*K channels where
    AF3's single group gives K*K. AF3 is the G = 1 case. Here G == K ==
    num_outer_channel (chai uses 8 and 8).

    Two further conventions, both of which would pass any shape check:
      * there is NO mean. The depth axis is summed and never divided -- not by
        the row count, not by the pair count (AF3 divides by the pair count).
      * the LayerNorm over the product channels, which is what absorbs that
        missing scale, uses epsilon 0.1 rather than the 1e-5 the rest of the
        graph uses.
    Both were read off the trace.
    """
    g = k = self.config.num_outer_channel

    def project(name):
      # one Linear to g*k, then split the group axis -- keeps AF3's own param
      # names (a name_scope here would land the weights under `~_chai_grouped`,
      # since haiku prefixes anything created outside __call__)
      proj = hm.Linear(g * k, initializer='linear', use_bias=False, name=name)(act)
      return mask[..., None] * proj.reshape(proj.shape[:-1] + (g, k))

    left_act = project('left_projection')
    right_act = project('right_projection')

    if self.global_config.final_init == 'zeros':
      w_init = hk.initializers.Constant(0.0)
    else:
      w_init = hk.initializers.VarianceScaling(scale=2.0, mode='fan_in')
    output_w = hk.get_parameter(
        'output_w', shape=(g, k, k, self.num_output_channel), dtype=act.dtype,
        init=w_init)
    output_b = hk.get_parameter(
        'output_b', shape=(self.num_output_channel,), dtype=act.dtype,
        init=hk.initializers.Constant(0.0))

    def compute_chunk(left_act):
      # contract the MSA-depth axis only; the group axis is shared, not summed
      prod = jnp.einsum('sigk,sjgl->ijgkl', left_act, right_act)
      return prod.reshape(prod.shape[0], prod.shape[1], g * k * k)

    prod = mapping.inference_subbatch(
        compute_chunk,
        self.config.chunk_size,
        batched_args=[left_act],
        nonbatched_args=[],
        input_subbatch_dim=1,
        output_subbatch_dim=0,
    )
    prod = hm.LayerNorm(name='product_norm', eps=0.1)(prod)
    return jnp.einsum('ijn,nf->ijf', prod,
                      output_w.reshape(g * k * k, self.num_output_channel)) + output_b


class PairFormerIteration(hk.Module):
  """Single Iteration of Pair Former."""

  class Config(base_config.BaseConfig):
    """Config for PairFormerIteration."""

    num_layer: int
    pair_attention: GridSelfAttention.Config = base_config.autocreate()
    pair_transition: TransitionBlock.Config = base_config.autocreate()
    single_attention: diffusion_transformer.SelfAttentionConfig | None = None
    single_transition: TransitionBlock.Config | None = None
    triangle_multiplication_incoming: TriangleMultiplication.Config = (
        base_config.autocreate(equation='kjc,kic->ijc')
    )
    triangle_multiplication_outgoing: TriangleMultiplication.Config = (
        base_config.autocreate(equation='ikc,jkc->ijc')
    )
    shard_transition_blocks: bool = True

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      with_single=False,
      with_pair_attention=True,
      *,
      name,
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.with_single = with_single
    # ESMFold2's PairUpdateBlock is tri_mul_out, tri_mul_in, pair_transition and
    # nothing else -- no triangle attention at all. The port expressed that by
    # filling the attention weights with zeros, which reproduces the block
    # exactly (the act_norm scale is zero, so the whole branch evaluates to zero
    # and lands on a residual add) but pays for two full GridSelfAttentions per
    # block to compute it. That is 24 or 48 blocks times every recycle, and
    # triangle attention is the same order as the triangle multiplications that
    # are actually wanted -- so it roughly doubled the trunk.
    #
    # NOT inferable from with_single: the template stacks are pair-only too and
    # their attention is real.
    self.with_pair_attention = with_pair_attention

  def __call__(
      self,
      act,
      pair_mask,
      single_act=None,
      seq_mask=None,
      use_dropout=False,
      extra_pair_bias=None,
  ):
    """Build a single iteration of the pair former.

    Args:
      act: [num_res, num_res, num_channel] Input pairwise activations.
      pair_mask: [num_res, num_res] padding mask.
      single_act: [num_res, single_channel] Single Input activations, optional
      seq_mask: [num_res] Sequence Mask, optional.
      use_dropout: traced flag enabling the SI's pair-stack dropout (0.25). Used
        for DESIGN as a stochastic regulariser; default False = exact inference.

    Returns:
      [num_res, num_res, num_channel] tensor of activations.
    """

    num_residues = act.shape[0]

    # chai-1's PairformerBlock is PARALLEL on both tracks: every pair update reads
    # the pair rep ENTERING the block and their results are summed into it, and the
    # single attention and single transition both read the single entering the block
    # (verified in chai's traced graph -- z20/input0 are the operand of all five).
    # AF3 instead threads each update through the running activation. `parallel`
    # therefore holds the block input and the deltas accumulate against it.
    parallel = self.global_config.model == 'chai1'
    pair_in = act
    pair_delta = jnp.zeros_like(act) if parallel else None

    def add_pair(delta):
      nonlocal act, pair_delta
      if parallel:
        pair_delta = pair_delta + delta
      else:
        act = act + delta

    # SI Algorithm 17: rowwise 0.25 on the two triangle multiplications and the
    # starting-node attention, columnwise 0.25 on the ending-node attention (the
    # transpose=True one). No dropout on the transition. off => bit-identical.
    add_pair(_pair_dropout(TriangleMultiplication(
        self.config.triangle_multiplication_outgoing,
        self.global_config,
        name='triangle_multiplication_outgoing',
    )(pair_in, pair_mask), use_dropout))

    add_pair(_pair_dropout(TriangleMultiplication(
        self.config.triangle_multiplication_incoming,
        self.global_config,
        name='triangle_multiplication_incoming',
    )(act if not parallel else pair_in, pair_mask), use_dropout))

    _attn = lambda nm, tr: GridSelfAttention(
        self.config.pair_attention,
        self.global_config,
        name=nm,
        transpose=tr,
    )(act if not parallel else pair_in, pair_mask)

    if not self.with_pair_attention:
      pass
    elif self.config.pair_attention.dual_output:
      # chai's confidence triangle attention is ONE module over both directions
      # whose output projection mixes them, then splits into a half that is kept
      # and a half that is transposed. Each direction therefore contributes to
      # both halves; we sum within each half and transpose once, which is chai's
      # `out0 + permute(out1)` (nodes 856/861/862).
      kept1, xpose1 = _attn('pair_attention1', False)
      kept2, xpose2 = _attn('pair_attention2', True)
      add_pair(_pair_dropout(kept1 + kept2, use_dropout))
      # NOT transposed. chai's tail reads
      #     856 dropout(out0)
      #     858 permute(out1) -> 859 dropout -> 861 permute
      #     862 add(856, 861)
      # and those two permutes CANCEL -- they exist only so the dropout lands
      # columnwise on that half. At inference dropout is identity, so chai adds
      # out1 in its original orientation. Transposing it here cost pae 0.983
      # against 0.99994, while pde barely noticed (0.9974) because that head
      # symmetrises: the error was purely antisymmetric, which is what pointed
      # at an orientation term in the first place.
      add_pair(_pair_dropout(xpose1 + xpose2, use_dropout, columnwise=True))
    else:
      add_pair(_pair_dropout(_attn('pair_attention1', False), use_dropout))
      add_pair(_pair_dropout(
          _attn('pair_attention2', True), use_dropout, columnwise=True))

    transition_block = TransitionBlock(
        self.config.pair_transition, self.global_config, name='pair_transition'
    )
    if self.config.shard_transition_blocks:
      transition_block = mapping.sharded_apply(
          transition_block,
          get_shard_size(
              num_residues, self.global_config.pair_transition_shard_spec
          ),
      )
    add_pair(transition_block(act if not parallel else pair_in))
    if parallel:
      act = pair_in + pair_delta

    if self.with_single:
      assert self.config.single_attention is not None
      # chai reads the pair rep entering the block here, not the updated one
      pair_logits = hm.Linear(
          self.config.single_attention.num_head,
          name='single_pair_logits_projection',
      )(hm.LayerNorm(name='single_pair_logits_norm')(pair_in if parallel else act))

      pair_logits = jnp.transpose(pair_logits, [2, 0, 1])

      # OpenDDE's structural-token refiner adds a precomputed pair attention bias
      # (from the token-expander's structural relationships) to the single-attention
      # logits, broadcast over heads. None for every other pairformer use.
      if extra_pair_bias is not None:
        pair_logits = pair_logits + extra_pair_bias[None].astype(pair_logits.dtype)

      single_in = single_act
      attn = diffusion_transformer.self_attention(
          single_act,
          seq_mask,
          pair_logits=pair_logits,
          config=self.config.single_attention,
          global_config=self.global_config,
          name='single_attention_',
          gate_bias=1.0 if parallel else 0.0,
      )
      # chai masks the attention residual, so padded rows stay exactly zero
      if parallel and seq_mask is not None:
        attn = attn * seq_mask[:, None].astype(attn.dtype)
      single_act = single_act + attn

      single_act = single_act + TransitionBlock(
          self.config.single_transition,
          self.global_config,
          name='single_transition',
      )(single_in if parallel else single_act, broadcast_dim=None)

      return act, single_act
    else:
      return act


class EvoformerIteration(hk.Module):
  """Single Iteration of Evoformer Main Stack."""

  class Config(base_config.BaseConfig):
    """Configuration for EvoformerIteration."""

    num_layer: int = 4
    msa_attention: MSAAttention.Config = base_config.autocreate()
    outer_product_mean: OuterProductMean.Config = base_config.autocreate()
    msa_transition: TransitionBlock.Config = base_config.autocreate()
    pair_attention: GridSelfAttention.Config = base_config.autocreate()
    pair_transition: TransitionBlock.Config = base_config.autocreate()
    triangle_multiplication_incoming: TriangleMultiplication.Config = (
        base_config.autocreate(equation='kjc,kic->ijc')
    )
    triangle_multiplication_outgoing: TriangleMultiplication.Config = (
        base_config.autocreate(equation='ikc,jkc->ijc')
    )
    shard_transition_blocks: bool = True

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='evoformer_iteration',
      with_pair_attention=True,
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    # As in PairFormerIteration: ESMFold2's MSA encoder block has no triangle
    # attention, and the port expressed that with zero weights -- which still
    # costs two GridSelfAttentions per block to compute a guaranteed zero.
    self.with_pair_attention = with_pair_attention

  def __call__(self, activations, masks, use_dropout=False):

    msa_act, pair_act = activations['msa'], activations['pair']

    num_residues = pair_act.shape[0]

    msa_mask, pair_mask = masks['msa'], masks['pair']

    def _opm():
      return OuterProductMean(
          config=self.config.outer_product_mean,
          global_config=self.global_config,
          num_output_channel=int(pair_act.shape[-1]),
          name='outer_product_mean',
      )(msa_act, msa_mask)

    def _msa_update(m):
      # MSAPairWeightedAveraging: DropoutRowwise 0.15 over the MSA tensor (s,r,c),
      # sharing the mask across sequences (axis 0) -- AF3 SI Alg 8, OF3 msa_dropout.
      m = m + _pair_dropout(MSAAttention(
          self.config.msa_attention, self.global_config, name='msa_attention1'
      )(m, msa_mask, pair_act=pair_act, pair_mask=pair_mask),
                            use_dropout, rate=0.15)
      m = m + TransitionBlock(
          self.config.msa_transition, self.global_config, name='msa_transition'
      )(m)
      return m

    # OpenDDE's MSABlock updates the MSA FIRST, then feeds the *updated* MSA to the
    # OuterProductMean (opendde/model/modules/pairformer.py MSABlock.forward). AF3
    # runs the OPM on the pre-update MSA. Order matters and compounds over blocks;
    # gate on opendde so AF3/OF3/IF2 keep their original ordering byte-for-byte.
    # Boltz-2's MSALayer uses the same update-then-OPM order (boltz2_msa_order gate,
    # separate from opendde so it does NOT pull in the structural-token stage).
    # protenix mini/tiny have no msa_stack at all -- OPM into the pair, and that
    # is the entire MSA contribution. Building _msa_update for them creates 12
    # parameters per block that the checkpoint cannot fill.
    if self.global_config.model in model_config.NO_MSA_ROW_UPDATE:
      pair_act += _opm()
    elif self.global_config.model in ('opendde', 'boltz2'):
      msa_act = _msa_update(msa_act)
      pair_act += _opm()
    else:
      pair_act += _opm()
      msa_act = _msa_update(msa_act)

    def _tri_mul_out(z):
      return _pair_dropout(TriangleMultiplication(
          self.config.triangle_multiplication_outgoing,
          self.global_config,
          name='triangle_multiplication_outgoing',
      )(z, pair_mask), use_dropout)

    def _tri_mul_in(z):
      return _pair_dropout(TriangleMultiplication(
          self.config.triangle_multiplication_incoming,
          self.global_config,
          name='triangle_multiplication_incoming',
      )(z, pair_mask), use_dropout)

    def _attn1(z):
      return _pair_dropout(GridSelfAttention(
          self.config.pair_attention,
          self.global_config,
          name='pair_attention1',
          transpose=False,
      )(z, pair_mask), use_dropout)

    def _attn2(z):
      return _pair_dropout(GridSelfAttention(
          self.config.pair_attention,
          self.global_config,
          name='pair_attention2',
          transpose=True,
      )(z, pair_mask), use_dropout, columnwise=True)

    def _transition(z):
      transition_block = TransitionBlock(
          self.config.pair_transition, self.global_config, name='pair_transition'
      )
      if self.config.shard_transition_blocks:
        transition_block = mapping.sharded_apply(
            transition_block,
            get_shard_size(
                num_residues, self.global_config.pair_transition_shard_spec
            ),
        )
      return transition_block(z)

    if self.global_config.model == 'chai1':
      # chai's MSA block is PARALLEL in two stages, and its pair transition sits
      # in the FIRST stage rather than last: the two triangle multiplications and
      # the transition all read the same post-OPM z and are summed into it, then
      # both attention directions read that result and are summed in turn. AF3
      # instead threads all five updates sequentially, each reading the output of
      # the one before. Same modules, same weights, different function -- and it
      # only shows up in the assembled graph, because the trunk harness
      # (cmp_trunk.py) hand-wires this block the parallel way and so agrees with
      # chai while the graph does not. Matches the trunk trace verbatim
      # (`z20 = token_pair_repr + (pair_repr13 + tri_attn_output2)`).
      delta = _tri_mul_out(pair_act) + _tri_mul_in(pair_act) + _transition(pair_act)
      pair_act = pair_act + delta
      if self.with_pair_attention:
        pair_act = pair_act + (_attn1(pair_act) + _attn2(pair_act))
    else:
      pair_act += _tri_mul_out(pair_act)
      pair_act += _tri_mul_in(pair_act)
      if self.with_pair_attention:
        pair_act += _attn1(pair_act)
        pair_act += _attn2(pair_act)
      pair_act += _transition(pair_act)

    return {'msa': msa_act, 'pair': pair_act}
