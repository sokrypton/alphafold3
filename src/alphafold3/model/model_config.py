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
    # OpenFold3's v0.5.0 release. Its own name because it diverges from
    # `openfold3` in a forward convention, not just in weights.
    'openbind',
    # Protenix's v1 release (368 M). Stock AlphaFold 3 everywhere except one
    # width -- see PROTENIX1_SETTINGS.
    'protenix1',
    # A later training run of the SAME graph as protenix1 -- derive_dims reports
    # byte-identical dimensions -- so it is a weights-only variant.
    'protenix1_20250630',
    # Protenix v0.5.0 base: the same graph as v1 but TEMPLATELESS (the checkpoint
    # carries zero template_embedder blocks), which is what made it the variant an
    # earlier port reached for when the template widths were still unsolved.
    'protenix05',
    # Protenix ships nine model types that differ only in counts and widths.
    # protenix2 is its flagship; these two are the small ones, and they are
    # genuinely small -- 16 and 8 pairformer blocks against 48, an 8-block
    # diffusion transformer against 24, and 5 sampling steps against 200.
    'protenix_mini',
    'protenix_tiny',
    'intellifold2',
    'opendde',
    'boltz2',
    'protenix2',
    'rosettafold3',
    'chai1',
)

# The Protenix family. Its model types differ from one another ONLY in counts
# and widths (converters/protenix2.derive_dims reads both off the checkpoint), so
# every FORWARD branch that protenix2 takes, mini and tiny take too. Keeping the
# membership in one place is what stops the next variant from being added to four
# lists and missed in a fifth -- which happened three times while mini was being
# ported, each time surfacing only as a shape error or an uncovered-parameter
# count, never as anything that named the cause.
PROTENIX_FAMILY = ('protenix05', 'protenix1', 'protenix1_20250630',
                   'protenix2', 'protenix_mini', 'protenix_tiny')


# Models whose forward graph follows OpenFold3's conventions rather than stock
# AlphaFold3's: the 1-indexed element shift, the symmetrised bond matrix, and
# Fourier noise weights read from the checkpoint.
# IntelliFold-2 is deliberately absent -- its converter emits stock-AF3 names.
#
# Two conventions that USED to be described here have moved out, because
# openbind keeps this lineage while reverting them to AlphaFold 3's:
# the per-block pair LayerNorm (PER_BLOCK_PAIR_LAYER_NORM, below) and the
# swapped column-attention pair bias (an explicit list at modules.py, where the
# open question about openbind's direction is recorded).
OPENFOLD3_LINEAGE = (
    'openfold3', 'openbind', 'opendde', 'boltz2', 'rosettafold3',
) + PROTENIX_FAMILY


# Models whose diffusion transformer LayerNorms the pair conditioning ONCE PER
# BLOCK, rather than once for the whole stack as AlphaFold 3 does.
#
# This is NOT the same question as OPENFOLD3_LINEAGE, and openbind is why. Lineage
# is provenance -- who derived their model from whom, which decides the bond
# symmetrisation, the element index shift and where the Fourier weights come from.
# This is a CONVENTION, and OpenFold3 changed it between releases: their v0.5.0
# notes say "Moved the pair layer norm in the diffusion transformer out of
# attention pair bias. The pair layer norm is run once to match the AlphaFold3
# SI." So openbind is OpenFold3 by lineage and AlphaFold 3 here, and deriving one
# list from the other would make that impossible to express.
#
# The checkpoint states which it is, so nothing has to be remembered: preview-2
# carries 24 `blocks.N.attention_pair_bias.layer_norm_z.weight`, openbind carries
# a single `diffusion_transformer.layer_norm_z.weight` (converters/openfold3.py
# `is_openbind_checkpoint`).
PER_BLOCK_PAIR_LAYER_NORM = (
    'openfold3', 'opendde', 'boltz2', 'rosettafold3',
) + PROTENIX_FAMILY
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
KEY_MASKED_ATOM_ATTENTION = ('rosettafold3', 'opendde') + PROTENIX_FAMILY


# Models that recycle through a linear STATE-SPACE step instead of an addition.
#
# ESMFold2's "parcae" trunk is a discretised diagonal SSM over the recycle axis,
# z = a * z_prev + b(norm(z_inject)), where a and b are input-independent and so
# fold to plain arrays at conversion time. Everything else here recycles with
# z = z_inject + prev_embedding(norm(z_prev)), which is the same expression at
# a = 1 with the operands the other way round.
SSM_RECYCLE = ('esmfold2',)


# Models whose MSA stack does NOT update the MSA representation at all: their
# msa_module block is OuterProductMean + a pair stack, and there is no MSA row
# attention and no MSA transition anywhere in the checkpoint.
#
# Protenix's mini and tiny distillations drop the whole `msa_stack` submodule
# that protenix2 carries (`msa_pair_weighted_averaging` + `transition_m`).
# Building it anyway is not free: `_msa_update` creates 12 parameters per block
# that no checkpoint can fill, so the conversion reports them missing and the
# graph REFUSES TO APPLY -- which is how protenix_tiny came to ship a blob that
# could not be loaded against a batch carrying templates.
NO_MSA_ROW_UPDATE = ('protenix_mini', 'protenix_tiny')


# Models whose diffusion conditioning concatenates a PROJECTED relative-position
# encoding with the trunk pair, rather than AF3's raw 139-channel features.
# Widths: AF3 folds [z_trunk(c_z), rel_features(139)] -> c_z, these fold
# [z_trunk(c_z), relpe(c_z)] -> 2*c_z, so getting the membership wrong is a shape
# error at load (267 vs 256) rather than a silent one -- which is why this list
# was the third and last of the protenix_mini omissions to surface.
DIFFUSION_PROJECTED_RELPOS = ('boltz2', 'rosettafold3') + PROTENIX_FAMILY


# Models that compute the column-attention pair bias from the TRANSPOSED pair
# representation, as OpenFold3 preview-2 does. openbind is deliberately absent;
# see the note at modules.py, where the open question about its direction lives.
TRANSPOSED_COLUMN_PAIR_BIAS = ('openfold3', 'opendde', 'boltz2') + PROTENIX_FAMILY


# Models whose ATOM cross-attention transformer LayerNorms the atom-pair
# conditioning per block, rather than once for the stack.
#
# A SEPARATE question from PER_BLOCK_PAIR_LAYER_NORM, which is about the TOKEN
# transformer, and the two memberships genuinely differ: openfold3 and boltz2 are
# per-block on the token transformer and shared on the atom one. Deriving either
# list from the other would be wrong for four of nine models.
#
# The cost of getting it wrong is not a crash: the block parameters land under
# `__layer_stack_no_per_layer` while the graph reads
# `__layer_stack_with_per_layer`, so the conversion "succeeds" and every atom
# block is left at its init value. The shape manifest is what catches it, which
# is how protenix_mini was caught (104 uncovered parameters).
PER_BLOCK_ATOM_PAIR_LAYER_NORM = ('opendde', 'rosettafold3') + PROTENIX_FAMILY


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
DISTOGRAM_BIAS = ('opendde', 'rosettafold3', 'boltz2') + PROTENIX_FAMILY


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
