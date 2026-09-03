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

"""Global config for the model."""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from alphafold3.common import base_config
import tokamax

_Shape2DType: TypeAlias = tuple[int | None, int | None]


# Every model family this fork can run, by full name + version. `opendde` carries
# no version because upstream publishes none.
MODELS = (
    'alphafold3',
    'openfold3',
    'intellifold2',
    'opendde',
    'boltz2',
    'protenix2',
    'rosettafold3',
    'chai1',
)

# Models whose forward graph follows OpenFold3's conventions rather than stock
# AlphaFold3's: the per-block pair LayerNorm in the diffusion transformer, the
# 1-indexed element shift, the symmetrised bond matrix, the swapped
# column-attention pair bias, and Fourier noise weights read from the checkpoint.
# IntelliFold-2 is deliberately absent -- its converter emits stock-AF3 names.
OPENFOLD3_LINEAGE = (
    'openfold3',
    'opendde',
    'boltz2',
    'protenix2',
    'rosettafold3',
)
# chai-1 is deliberately absent: it is not OpenFold-derived at all. Its primitives
# (merged bidirectional triangle multiplication, fused two-direction triangle
# attention, grouped outer product mean) are its own, so it shares none of the OF3
# forward conventions and gets its own branches throughout.


# Models whose ATOM cross-attention masks a padded KEY from every real query.
#
# AF3 biases with `1e9 * (mask_q - 1) * (mask_k - 1)` -- an AND, penalising a pair
# only when the query and the key are both invalid, so for a real query the bias
# is zero on every key including padding. It gets away with that because
# `AtomCrossAtt` SHIFTS an out-of-bounds atom window bodily back inside the real
# atom count, so its 128 keys are real whenever there are 128 atoms to find.
#
# These three do not rely on that. rosettafold3 adds the two mask terms
# (`-1e9 * (maskQ + maskK)`); protenix2 and opendde pad the key sequence and then
# write -inf into the padded columns FOR REAL QUERIES
# (`attn_bias[..., :n, 0:pad_left] = -inf`, protenix
# `model/modules/primitives.py:497` and opendde `primitives.py:533`). All three
# are an OR, and the single OR form below reproduces every one of them.
#
# The membership rule is "what the native does", NOT "whether we pad" -- which is
# why this is its own list rather than being derived from the `padded_keys`
# featurisation knob. It is a superset of that knob, and `model_registry_test`
# asserts the containment, so a future padded-window port cannot land here
# masking the wrong way.
KEY_MASKED_ATOM_ATTENTION = (
    'rosettafold3',
    'protenix2',
    'opendde',
)


# Models whose distogram head carries a trained BIAS. Stock AF3's `half_logits`
# is bias-free (`hm.Linear` defaults to use_bias=False), so for four families the
# converter simply never asked for the key and no gate could see it -- the
# distogram moves no coordinates. protenix2's spans -2.05..+1.82, and softmax
# over the bias alone spreads 0.50 to 2e-4 across bins, so it is not decorative.
#
# WHERE THE FACTOR OF TWO GOES. Our graph applies the linear and THEN symmetrises
# (`logits = half + swap(half)`), which passes the bias through twice. The natives
# split on this and the split does not follow the family lineage:
#
#   b as-is (native also symmetrises AFTER, so it doubles too)
#     protenix2  `logits = linear(z); logits + logits.transpose(-2, -3)`
#     opendde    the same, head.py:44
#   b HALVED by the converter (native symmetrises the pair features FIRST, so its
#   bias enters once and our doubling has to be undone)
#     rosettafold3  `predictor(Z + Z.transpose(-2, -3))`, RF3_structure.py:252
#     boltz2        `z = z + z.transpose(1, 2); distogram(z)`, trunkv2.py:826
#
# Reading the tensor name alone would have got two of the four wrong. The halving
# lives in the converters, as a weight transform, so there is no forward branch.
DISTOGRAM_BIAS = (
    'protenix2',
    'opendde',
    'rosettafold3',
    'boltz2',
)


class GlobalConfig(base_config.BaseConfig):
  """Global configuration for the AlphaFold3 model."""

  bfloat16: Literal['all', 'none', 'intermediate'] = 'all'
  final_init: Literal['zeros', 'linear'] = 'zeros'
  pair_attention_chunk_size: Sequence[_Shape2DType] = ((1536, 128), (None, 32))
  pair_transition_shard_spec: Sequence[_Shape2DType] = (
      (2048, None),
      (None, 1024),
  )
  # Note: flash_attention_implementation = 'xla' means no flash attention.
  flash_attention_implementation: tokamax.DotProductAttentionImplementation = (
      'triton'
  )
  # Which model's weights/forward conventions this graph runs. One of MODELS
  # (alphafold3.model.model_config.MODELS): the single switch
  # every ported-family forward branch keys on, e.g.
  #   if self.global_config.model in ('boltz2', 'rosettafold3'): ...
  # Names are always full-name + version so a future port of another version of
  # the same model (boltz1, protenix3, ...) gets its own unambiguous name.
  model: str = 'alphafold3'
