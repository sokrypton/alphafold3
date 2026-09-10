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
    'openbind0',
    # Protenix's v1 release (368 M). Stock AlphaFold 3 everywhere except one
    # width -- see PROTENIX1_SETTINGS.
    'protenix1',
    'intellifold2',
    'opendde',
    'boltz2',
    'protenix2',
    'rosettafold3',
    'chai1',
    # ESMFold2: an ESM-C-conditioned all-atom diffusion predictor. Rides the
    # shared graph on zero-filled pair attention (its trunk is pair-only), an
    # SSM recycle, a clamped OPM norm and a rotary atom window -- see
    # SSM_RECYCLE, CLAMPED_OPM_NORM, SWA_ROPE_ATOM_ATTENTION.
    'esmfold2',
    # ...and its released "Fast" variant, which halves the folding trunk (24
    # blocks against 48) and keeps the same ESM-C 6B tower.
    #
    # The ESMFold2-Experimental and -Experimental-Cutoff2025 lines, and their
    # -Fast siblings, were ported and then DROPPED on 2026-09-09. They are gone
    # from the registry, the converters and the oracles; git history and
    # PARITY.md hold what was learned from them, including the two fixes they
    # were the only models to exercise (the experimental MSA block order and
    # its outer-product bias placement).
    'esmfold2_fast',
    # the smaller-ESM-C tiers: the experimental-fast architecture trained
    # against a smaller tower, which is the only thing that differs -- same
    # trunk depth, same sampler, same heads.
    'esmfold2_lm600m',
    'esmfold2_lm300m',
)

# AlphaFold 2, which is NOT in MODELS above and must not be: every name there is
# a set of weights for THIS graph, selected by `global_config.model`, and AF2
# does not ride this graph at all. Its trunk is MSA row/column attention where
# AF3 uses pair-weighted averaging, and its head is IPA over backbone frames and
# torsions where AF3 diffuses coordinates -- there is no weight transform
# between those. AF2 is a sibling network under `alphafold3.af2`, reached by its
# own runner, and these names exist so that one registry can offer every model
# this package can run.
#
# `af2_ptm` covers the monomer pTM models and `af2_multimer` the multimer_v3
# ones. Both run on ONE graph: a monomer checkpoint is converted onto the
# multimer network at load (alphafold3.af2.convert), which is the same trick the
# AF3-family ports use, one model generation down.
AF2_MODELS = (
    'af2_ptm',
    'af2_multimer',
)

# Every model this package can run, whichever graph it rides. Use this for a
# user-facing list; use MODELS for anything that indexes the AF3 graph.
ALL_MODELS = MODELS + AF2_MODELS


# The ESMFold2 family. Like PROTENIX_FAMILY, the members differ only in COUNTS
# -- 24 or 48 trunk blocks -- so every forward branch one takes, all take.
# Naming the membership once is the whole point: `esmfold2` alone appeared in
# thirteen places in this file and eight more in the graph, and a variant added
# to twelve of them would fail as a shape error that names nothing.
# Four releases, not the eight that were ported: the ESMFold2-Experimental and
# -Experimental-Cutoff2025 lines (and their -Fast siblings) were dropped on
# 2026-09-09 to keep the debugging surface honest. What these four still cover
# between them is both trunk lines, the released MSA encoder, and all three
# ESM-C tower sizes; what went with the dropped four is the EXPERIMENTAL MSA
# encoder, since they were the only msa=4 models on that line.
ESMFOLD2_FAMILY = ('esmfold2', 'esmfold2_fast',
                   'esmfold2_lm600m', 'esmfold2_lm300m')

# ...with one real architectural split inside it. The two RELEASED models recycle
# through the parcae SSM; the two language-model-tier ones carry
# `pair_loop_proj` instead -- a LayerNorm(256) and a Linear(256, 256), which is exactly AF3's own
# `prev_embedding_layer_norm` + `prev_embedding`. So the experimental line reverts
# to stock recycling, and drops the parcae readout and coda with it.
ESMFOLD2_SSM_RECYCLE = ('esmfold2', 'esmfold2_fast')

# The experimental-line members that remain: esmfold2_lm600m and
# esmfold2_lm300m. Both are structure-only (msa 0, no confidence head), so this
# now selects the RECYCLE difference and nothing else -- every exception that
# needed an experimental model WITH an MSA stack or a confidence head is dead
# and has been removed rather than left keyed on a list that cannot match.
ESMFOLD2_EXPERIMENTAL = tuple(m for m in ESMFOLD2_FAMILY
                              if m not in ESMFOLD2_SSM_RECYCLE)

# Confidence heads a model does not have. Building one anyway leaves its
# parameters at random init and emits a prediction that looks like a prediction
# and is noise -- which is what happened to chai1's experimentally-resolved.
#
# NO_PDE_HEAD is gone: its only members were the ESMFold2-Experimental releases
# (dropped 2026-09-09) plus the lm-tier pair, and the lm pair ship NO confidence
# head at all (NO_CONFIDENCE_HEAD), so nothing could reach it.
NO_RESOLVED_HEAD = ('chai1',)
# ...and the LayerNorms they do not have. boltz2 has none before ANY head;
# ESMFold2's experimental line keeps plddt_ln but not pae_ln, so this is keyed
# by head, not by model.
# boltz2 has none before ANY head. The ESMFold2 entry that used to live here
# (`pae_logits_ln` for the experimental line) went with those releases: the
# lm-tier survivors build no confidence head at all.
NO_HEAD_NORM = {'boltz2': ('*',)}

# Models whose confidence re-embedding bins the predicted distances with their
# OWN trained boundaries rather than boltz2's constant 2..22 A over 63 edges.
# ESMFold2 trains 38 edges over 3.25..50.75 and its experimental line trains 127
# -- both of which reached the legacy reference map and NEITHER of which reached
# the graph, so the embedding was being fed bins its weights never saw. The
# number of CLASSES is one more than the number of edges and has to be static,
# so it rides in ESMFOLD2_VARIANTS as `conf_bins`.
LEARNED_CONFIDENCE_BINS = ESMFOLD2_FAMILY

# Models that ship NO confidence head at all (`confidence_head.enabled: false`
# in their config, and zero confidence_head tensors in the checkpoint).
# ESMFold2's language-model-tier releases are structure-only. Building the head
# anyway would leave ~100 parameters at random init and emit a pLDDT that looks
# like a prediction and is noise -- the same trap as chai1's resolved head.
NO_CONFIDENCE_HEAD = ('esmfold2_lm600m', 'esmfold2_lm300m')


# Which LayerNorms carry a trained OFFSET, keyed by the norm's own name.
# AlphaFold 3's are scale-only; several ports made specific ones affine, and
# each entry here was found the same way -- enumerate the checkpoint's affine
# LayerNorms and diff against the converter's scale-only scopes.
#
# ONE table rather than the eleven inline model-name tuples this replaces. A
# literal list edited by pattern is exactly how an ESMFold2 width leaked into
# chai1's and protenix2's settings and broke both for two commits; a new port
# now fills in rows here instead of hunting for tuples across four files.
AFFINE_LAYER_NORMS = {
    # diffusion conditioning
    'z_trunk_norm': ('boltz2', 'rosettafold3'),
    'pair_cond_initial_norm': (('boltz2', 'rosettafold3', 'chai1')
                               + ESMFOLD2_FAMILY),
    'single_cond_initial_norm': (('boltz2', 'rosettafold3', 'chai1')
                                 + ESMFOLD2_FAMILY),
    'noise_embedding_initial_norm': (('boltz2', 'rosettafold3', 'chai1')
                                     + ESMFOLD2_FAMILY),
    # the diffusion head's own re-embedding and output
    'single_cond_embedding_norm': ('boltz2', 'rosettafold3') + ESMFOLD2_FAMILY,
    'output_norm': ('boltz2', 'rosettafold3', 'chai1') + ESMFOLD2_FAMILY,
    # atom cross-attention (the names are suffixes: `<name>_lnorm_...`)
    'lnorm_trunk_single_cond': ('boltz2', 'chai1', 'rosettafold3'),
    'lnorm_trunk_pair_cond': ('boltz2', 'chai1', 'rosettafold3'),
    'atom_features_layer_norm': (('boltz2', 'chai1', 'rosettafold3')
                                 + ESMFOLD2_FAMILY),
    # the diffusion transformer's shared pair bias norm (both call sites)
    'pair_input_layer_norm': ('chai1',),
}


def affine_norm(model, name):
  """Does `model` carry a trained offset on the LayerNorm called `name`?"""
  return model in AFFINE_LAYER_NORMS.get(name, ())

# Where the MSA encoder sits relative to the recycle. The released line
# OVERWRITES the injection before it (`msa_encoder_overwrite: true`); the
# experimental line runs it AFTER, as an addition:
#     z = z_init + pair_loop_proj(z)

# Models whose MSA subsampling KEEPS THE QUERY AT ROW 0 and preserves the
# alignment's own row order, rather than taking AF3's uniform gumbel shuffle
# over every row followed by a truncation. ESMFold2 says so in its own
# docstring ("keeping query row 0") and sorts the indices it draws;
# `featurization.subsample_msa_keep_query` is that function, and why the
# difference is not cosmetic.
#
# Membership is stated per FAMILY, not per model, for the reason the protenix
# padded_keys bug taught: a convention named on one release is a convention the
# next release silently loses.
MSA_KEEP_QUERY_ROW = ESMFOLD2_FAMILY

# Models whose MSA block updates the MSA FIRST and then runs the outer product
# on the UPDATED msa, against AF3's outer-product-then-update. The order
# compounds over blocks, so it is not a detail.
#
# ESMFold2 ships BOTH orders, one per line, in two files:
#   modeling_esmfold2.py              OPM first, and the LAST block skips the
#                                     msa update entirely (is_final_block),
#   modeling_esmfold2_experimental.py update first, in EVERY block.
# So membership here is the EXPERIMENTAL releases only, and the released ones
# must stay out of it. That is also why the L1b gate read 1.000000 while this
# was wrong: `esmfold2_msa_dump.py` imports MSAEncoder from modeling_esmfold2,
# the RELEASED class, for all three msa-carrying variants -- so for the
# experimental line it was comparing us against the wrong native module, and
# agreeing with it.
#
# Invisible without a real alignment: the experimental encoder multiplies its
# whole output by `msa_track_mask`, which is False when the MSA has no non-query
# rows, so at depth 1 both orders return exactly zero.
# The ESMFold2-Experimental half of this list went with those releases: the two
# lm-tier survivors set msa=0, so no ESMFold2 model reaches an MSA block now.
MSA_UPDATE_BEFORE_OPM = ('opendde', 'boltz2')

# The Protenix family. Its model types differ from one another ONLY in counts
# and widths (converters/protenix2.derive_dims reads both off the checkpoint), so
# every FORWARD branch that protenix2 takes, the others take too. Keeping the
# membership in one place is what stops the next variant from being added to four
# lists and missed in a fifth -- which happened three times while the small
# variants were being ported, each time surfacing only as a shape error or an
# uncovered-parameter count, never as anything that named the cause.
#
# Deliberately TWO members. This fork carried six protenix model types
# (protenix05, protenix1_20250630, protenix_mini, protenix_tiny alongside these);
# they were removed on 2026-09-08 to stop spreading parity work across variants
# that differ only in training run or size. Everything below is still written as
# a FAMILY rather than a pair, because that is what made adding a variant a
# one-liner, and re-adding one should stay that way.
PROTENIX_FAMILY = ('protenix1', 'protenix2')


# Models whose forward graph follows OpenFold3's conventions rather than stock
# AlphaFold3's: the 1-indexed element shift, the symmetrised bond matrix, and
# Fourier noise weights read from the checkpoint.
# IntelliFold-2 is deliberately absent -- its converter emits stock-AF3 names.
#
# Two conventions that USED to be described here have moved out, because
# openbind keeps this lineage while reverting them to AlphaFold 3's:
# the per-block pair LayerNorm (PER_BLOCK_PAIR_LAYER_NORM, below) and the
# swapped column-attention pair bias (an explicit list at modules.py, where the
# openbind's direction, once an open question, is now measured -- see modules.py).
# Whose atom encoder is fed the RAW formal charge, and whose gets arcsinh(charge).
#
# AlphaFold 3 embeds `arcsinh(charge)`. Three families do not, and the two facts
# are indistinguishable on a neutral molecule -- 6MRR has 17 charged atoms out of
# 574 -- so this can only be caught by a gate that isolates the feature. The
# `FEAT=charge` arm of `dev/oracles/atom_parity.py` is that gate: with every
# other reference feature zeroed on both sides, boltz2 read max|d|/rms 2.18e-02
# where positions, element and atom-name characters were all exact at ~1e-6.
#
# Read off each vendor's own code, not inferred:
#   chai1          feeds the raw charge.
#   esmfold2       feeds the raw charge.
#   boltz2         `AtomEncoder.forward` concatenates `feats["ref_charge"]`
#                  verbatim (encodersv2.py:321), and its featuriser stores
#                  `GetFormalCharge()` unmodified (schema.py:684).
#   rosettafold3   `ref_charge = atom_array.charge` verbatim
#                  (atomworks af3_reference_molecule.py:429), and neither rf3
#                  nor atomworks' ML transforms contain an arcsinh or asinh at
#                  all. Found the same way as boltz2's, a day later: the
#                  `FEAT=charge` arm read max|d|/rms 6.03e-02 while position,
#                  element, atom-name characters and the mask were each exact
#                  at ~1e-6.
#   intellifold2   applies asinh INSIDE the module, so arcsinh here is right.
#   protenix/of3/opendde  AF3's convention, and their charge arms are exact
#                  (3.79e-06 / 2.02e-06), which is what says so.
RAW_REF_CHARGE = ('chai1', 'boltz2', 'rosettafold3') + ESMFOLD2_FAMILY


# Whose PDE head reads the pair AS IT IS -- no symmetrisation anywhere.
#
# AlphaFold 3 projects the pair and symmetrises the LOGITS
# (`left + swap(left)`); protenix symmetrises the PAIR inside the LayerNorm
# (PRE_SYMMETRISED_PDE); boltz2 symmetrises first and then splits by chain.
# ESMFold2 does NONE of it: `pde_logits = pde_head(pde_ln(pair))`, exactly like
# its PAE head (modeling_esmfold2.py ConfidenceHead.forward).
#
# Worth full_pde rms 0.389 of native's at corr 0.9017 -- ours came out 2.6x
# small, which is what summing a logit with its transpose does to an
# expectation. PAE was already close (0.9947) because nothing symmetrises it on
# either side, and that contrast is what pointed here.
UNSYMMETRISED_PDE = ESMFOLD2_FAMILY


# Whose atom transformer sees ZEROS in the padded atom slots at the top of EVERY
# block, rather than whatever the previous block left there.
#
# IntelliFold-2 pads its flat atom axis INSIDE each attention call
# (`pad_at_dim(a_row, ..., value=0.)` in `forward_local`), so every block starts
# from a freshly zero-padded activation. Our stack pads once, carries the
# activation through the layer_stack, and a masked QUERY still produces an
# output -- which the next block then GATHERS as a key. chai-1 re-masks for its
# own reason (a parallel block that reads the masked activation twice) and lands
# in the same place.
#
# The signature is unmistakable and took a pair gate to see: with if2's window
# alignment fixed, its atom pair is EXACT on every window (3.98e-06) and block 1
# is 9.5e-03, while blocks 2 and 3 blow up on windows 16 and 17 ALONE -- the two
# whose key sets reach past the last real atom into our padding.
MASK_ATOM_ACT_PER_BLOCK = ('chai1', 'intellifold2')


# Whose relative-CHAIN bucket is keyed on same-CHAIN, sending the MATCH to the
# pad class, rather than on same-ENTITY sending the MISMATCH there.
#
# On a monomer this flips EVERY pair: same-chain is universally true, so one
# convention makes the whole a_rel_chain one-hot the pad class (column 5) and
# the other the zero-offset class (column 2). Read off each vendor:
#
#   boltz2     `torch.where(b_same_chain, 2*s_max+1, d_chain)`
#              (encodersv2.py RelativePositionEncoder.forward)
#   esmfold2   the same, which is where this was first found -- worth
#              1.522 -> 0.719 A on esmfold2_lm600m
#   opendde,   `d_chain * b_same_entity + (1 - b_same_entity) * (2*s_max+1)`,
#   protenix   i.e. AF3's convention. Checked, not assumed.
#
# TWO CALL SITES build these features -- the trunk (evoformer.py) and the
# diffusion conditioning (diffusion_head.py) -- and they DO NOT AGREE for
# boltz2, which is why there are two lists.
#
# For the diffusion conditioning, boltz2's convention is certified by its own
# module: with it, `pair_cond` is 1.000000 / max|d|/rms 1.95e-06 against
# boltz's `PairwiseConditioning` fed our features; without it, 0.949940 / 1.60.
# It is also free on the fold -- 15 paired samples of 6MRR, best 0.425 against
# 0.433, mean 0.535 against 0.539.
#
# For the TRUNK it is NOT settled, and the fold says the opposite. Same 15
# paired samples with the trunk ALSO flipped: mean 0.801 against 0.539, best
# 0.514 against 0.433, and three samples at ~1.5 where the other arm has
# nothing above 0.65. That is not sampling noise, and it contradicts the vendor
# using ONE `rel_pos` module for both -- so something else in our trunk's
# relative encoding differs and the two errors have been cancelling. Nothing
# gates the trunk's z-INIT (trunk_parity feeds the pairformer stack synthetic
# s/z, so it never sees relpos at all), which is the missing measurement and
# the next step; until it exists, the trunk keeps the convention that folds.
# boltz2 is in BOTH lists, and the second list exists only because it took two
# gates to establish that. `dev/oracles/trunk_init_parity.py` measures the trunk
# z-init against boltz2's own five terms, and with `PASSES=n` the whole recycled
# loop (`s/z recycle -> msa_module -> pairformer`) against boltz's own:
#
#   convention     z-init      loop, 1 pass        loop, 4 passes
#   same-chain     2.77e-06    s 1.9e-05 z 3.5e-05  s 1.5e-05 z 1.9e-05
#   same-entity    6.49e-01    s 1.49    z 9.71     s 1.02    z 3.42
#
# So our recycled trunk is bit-faithful to boltz2 with the same-chain
# convention and badly wrong with AF3's. That is what settles it. Everything
# else agrees: boltz2's own `RelativePositionEncoder` keys the bucket on
# same-CHAIN and sends the MATCH to the pad class, and esmfold2 -- the other
# model that does -- was worth 1.522 -> 0.719 A when this was first found.
#
# THE FOLD STILL PREFERS THE WRONG ONE on 6MRR, and that is recorded rather
# than obeyed. 8 seeds x 5 samples per arm at 10 recycles: same-entity mean
# 0.540 with nothing above 0.66, same-chain mean 0.700 with four samples at
# 0.84-1.5; at zero recycles the two are level (0.536 vs 0.570). A 0.16 A mean
# on one 68-residue target does not outweigh a trunk that is provably exact
# through four passes, so the certified convention ships and
# AF3_BOLTZ2_TRUNK_ENTITY_BUCKET restores the other for anyone comparing.
#
# On other targets the change is mixed and small, which is the other half of
# the reason it is not worth trading the trunk for:
#
#   1STP protein   0.276 -> 0.266 best      1STP BTN  0.458 -> 0.895
#   1EHZ RNA       1.191 -> 1.198 best      6MRR      0.540 -> 0.700 mean
#
# What that leaves open is a real question, and it is DOWNSTREAM of the trunk:
# with the trunk exact, something in the diffusion or the sampler turns a
# correct pair representation into a worse structure on this target ~10% of the
# time. The loop gate is the tool for the next model that shows this.
CHAIN_BUCKET_ON_SAME_CHAIN = (
    (() if __import__('os').environ.get('AF3_BOLTZ2_TRUNK_ENTITY_BUCKET')
     else ('boltz2',)) + ESMFOLD2_FAMILY)
CHAIN_BUCKET_ON_SAME_CHAIN_DIFFUSION = ('boltz2',) + ESMFOLD2_FAMILY


# boltz2's atom cross-attention builds its keys by gathering the ALREADY
# NORMALISED queries -- `k_in = to_keys(b)` where `b = self.adaln(a, s)`, with no
# key-side norm after it (transformersv2.py DiffusionTransformerLayer.forward),
# and its checkpoint carries a single `adaln` per layer where opendde carries
# `layernorm_a` + `layernorm_kv`.
#
# THAT NEEDS NO BRANCH, and the reason is worth keeping: adaptive LayerNorm is
# POINTWISE per atom, so gathering before or after it is the same computation.
# `adaln_k(gather(a))` with the k-side weights equal to the q-side -- which is
# what a converter must write when the vendor has one `adaln` -- IS
# `gather(adaln_q(a))`. Implementing the gather-first form changed the atom
# encoder by nothing at all (q_atom max|d|/rms 1.20e-02 at one block, before and
# after, to five decimals), so it was reverted.
#
# The opendde/protenix CHAINED form is different precisely because it applies a
# norm TWICE, and composition does not commute away.


OPENFOLD3_LINEAGE = (
    'openfold3', 'openbind0', 'opendde', 'boltz2', 'rosettafold3',
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
# `of3_release`).
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
# These four do not rely on that. rosettafold3 adds the two mask terms
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
# boltz2 masks the key side and only the key side:
# `attn = attn + (1 - mask[:, None, None]) * -inf` with `mask` gathered into the
# KEYS layout (`model/layers/attention.py:124`, and `to_keys(mask)` two lines
# above). It was found by `model_registry_test.test_padded_key_windows_imply_
# the_or_mask` the moment boltz2 gained the `padded_keys` knob, which is what
# that test is for.
# intellifold2 masks the key side too: `forward_local` pads `single_mask` with
# zeros, windows it with the same duplicated-edge construction as the keys, and
# `_prep_inputs_local` turns that into the attention bias -- so a padded key is
# excluded from EVERY query, not only from a padded one. It has no
# `padded_keys` knob (its window policy is a third one: the edge windows are
# DUPLICATED), which is why the registry test that caught boltz2 did not fire
# here. The mask convention is independent of the window policy.
KEY_MASKED_ATOM_ATTENTION = (
    ('rosettafold3', 'opendde', 'boltz2', 'intellifold2') + PROTENIX_FAMILY)


# Models that recycle through a linear STATE-SPACE step instead of an addition.
#
# ESMFold2's "parcae" trunk is a discretised diagonal SSM over the recycle axis,
# z = a * z_prev + b(norm(z_inject)), where a and b are input-independent and so
# fold to plain arrays at conversion time. Everything else here recycles with
# z = z_inject + prev_embedding(norm(z_prev)), which is the same expression at
# a = 1 with the operands the other way round.
SSM_RECYCLE = ESMFOLD2_SSM_RECYCLE


# Models whose OuterProductMean divides by max(pair_count, 1) rather than by
# AF3's 1e-3 + pair_count. A scale, not an offset, and worth 1e-03 at MSA depth
# 1 -- which is the depth ESMFold2 runs at by default.
CLAMPED_OPM_NORM = ESMFOLD2_FAMILY


# Models whose OuterProductMean divides BEFORE its output projection, so the
# output BIAS is not scaled by the mask count. (The name is a misnomer kept for
# findability: the first reading of boltz's source said "per-token row count",
# and the comment below records how the gate disproved that.)
#
# Boltz-2's OPM (boltz/model/layers/outer_product_mean.py) is
#     mask = mask[:, :, None, :] * mask[:, :, :, None]   # PAIRWISE first
#     num_mask = mask.sum(1).clamp(min=1)
#     z = einsum(a, b) / num_mask                        # divide FIRST
#     return self.proj_o(z)                              # ...then project
# where AF3 projects first (bias included) and divides after. The COUNT is the
# same as AF3's -- boltz builds the pairwise mask before summing, so its
# `mask.sum(1)` is a pairwise count, not a per-token row count -- and the only
# other difference is `clamp(min=1)` against AF3's `+ 1e-3`.
#
# The bias placement is worth exactly (1 - 1/n) * output_b, predicted and then
# confirmed to 8e-04 (residual -0.008120 against a prediction of -0.008121;
# msa_parity.py LAYER=1, 2026-09-08).
#
# The gate runs a NON-UNIFORM msa mask as well as an all-ones one, and that is
# what caught a wrong first fix: reading `mask.sum(1)` as a per-token row count
# left the uniform case exact while making the non-uniform case WORSE
# (0.996 -> 0.968). A uniform mask cannot tell the two normalisers apart, so it
# cannot catch that error.
#
# **This bug needs MSA DEPTH > 1 to bite.** At depth 1 the bias term is
# (1 - 1/1) * b = 0 and the two normalisers agree, which is why boltz2's
# single-sequence 6MRR fold was exact throughout while its MSA module was not.
OPM_ROW_COUNT_NORM = ('boltz2',)

# Models that subtract each conformer group's OWN mean from `ref_pos` before the
# model sees it. Ours is the CCD ideal in whatever frame the CCD ships, and the
# offset is the size of the signal: on 1STP+BTN our 122 groups are off-centre by
# 1.493 A on average and 4.091 A at worst, against a raw-value rms of 1.554 A.
#
# It reaches a Linear RAW (the first element of the atom feature concatenation),
# as well as through a translation-INVARIANT pairwise difference -- so only the
# raw channel is affected, and it is affected fully.
#
#   boltz2                  featurizerv2.py:1494, centering=True, per ref_space_uid
#   openfold3 / openbind0   conformer.py:143 -> centre_random_augmentation,
#                           whose `pos_centered = xl - mean_xl` is unconditional
#   protenix* / opendde     random_transform(centralize=True), where the mean
#                           subtraction PRECEDES the apply_augmentation early return
#
# intellifold2 is deliberately absent: it passes `centering=False` and applies
# the rotation GLOBALLY rather than per group, so uncentred CCD ideals are what
# it expects. alphafold3 is absent because it is the reference implementation.
CENTRE_REF_CONFORMERS = ('boltz2', 'openfold3', 'openbind0',
                         'opendde') + PROTENIX_FAMILY

# The minimum blob CONVENTION this library will read, per model. See
# converters/common.BLOB_CONVENTION for what a convention is and what version 1
# means; an absent `__meta__/__convention__` record counts as 0, which is every
# blob published before the record existed.
#
# EMPTY ON PURPOSE, for now. Adding a model here makes the loader REFUSE that
# model's currently published blob until it is reconverted and re-uploaded, so
# an entry and a republish have to land together or every user of that model
# breaks in between. The esmfold2 family's convention DID change today (the
# alphabet permutation) and its blobs were republished, but without the record
# -- so enforcement for them rides along with the next reconversion rather than
# forcing a second 6.5 GB upload into a window where a half-applied guard is
# worse than none.
MIN_BLOB_CONVENTION: dict[str, int] = {}

# Models whose outer product adds the OUTPUT BIAS AFTER the divide, i.e.
# `Wout(outer) / n` against `Wout(outer / n)`. Algebraically the two differ by
# exactly `output_b * (1 - 1/n)` -- a per-CHANNEL CONSTANT, which is why corr
# cannot see it ([[correlation-hides-bias]]) and why it showed up as
# max|d| == p99.9|d| (1.3847 vs 1.38): the error is the same everywhere.
#
# ESMFold2 makes this a per-CHECKPOINT choice and says so in its own docstring
# ("different ESMFold2 checkpoints were trained with different orderings"):
# `OuterProductMean.divide_outer_before_proj` is False by default, which the
# RELEASED line takes, and the EXPERIMENTAL block hardcodes True
# (modeling_esmfold2_experimental.py:366). So the released line wants the bias
# INSIDE the divide -- which is what CLAMPED_OPM_NORM already gives it -- and
# the experimental line wants it outside.
#
# Note what this one defeats: `OuterProductMean` lives in the SHARED
# modeling_esmfold2_common.py, so the two lines do not differ by CLASS at all.
# They differ by the ARGUMENT one instantiates it with, which an audit of
# duplicated class names cannot see ([[esmfold2-two-file-sweep]]).
# Same story: only boltz2 remains, for the same reason as
# MSA_UPDATE_BEFORE_OPM above.
OPM_BIAS_AFTER_NORM = OPM_ROW_COUNT_NORM


# Models whose ATOM attention is a sliding window with 3D rotary positions
# instead of AF3's windowed pair bias.
#
# ESMFold2 gives each atom a window of +/-`ATOM_ROPE_HALF_WINDOW` by RANK among
# valid atoms, and its entire positional signal is a rotary embedding built from
# ref_pos and ref_space_uid -- there is no pair bias at all. AF3's window is
# BLOCK-aligned (query i sees [32b-48, 32b+79]), so the two are different masks
# even at the same width; the exact window rides in as an additive pair_mask and
# the key subset is widened to cover it (see ATOM_KEYS_SUBSET_SIZE).
SWA_ROPE_ATOM_ATTENTION = ESMFOLD2_FAMILY


# Models whose TRUNK carries no single track: 48 pair-only blocks, and the
# structure head is handed s_trunk=None. Their diffusion single conditioning is
# built from s_inputs alone rather than from [single_embedding, target_feat].
PAIR_ONLY_TRUNK = ESMFOLD2_FAMILY

# ESMFold2 keeps dropout on the LM pair representation at INFERENCE, resampled
# every recycle pass (config.lm_encoder.per_loop_lm_dropout; the top-level config
# says 0.0 and is overridden). It is not optional polish -- disabling it costs
# ~18 A on 6MRR.
# ...and only on the RELEASED line. Its 25% lives in the lm_encoder
# (`lm_encoder.lm_dropout`, `per_loop_lm_dropout: true`), and the experimental
# line has no lm_encoder at all: it calls the shim once with
# `lm_dropout=config.lm_dropout`, which its config sets to 0.0. Applying 25%
# there drops a quarter of a signal the model was never trained to lose.
LM_PAIR_DROPOUT = {m: (0.25 if m in ESMFOLD2_SSM_RECYCLE else 0.0)
                   for m in ESMFOLD2_FAMILY}


# Models whose sampler rigid-aligns the noisy coordinates onto the denoised
# prediction before the Euler step. See diffusion_head._kabsch.
REALIGN_SAMPLER = ESMFOLD2_FAMILY


# Models that LayerNorm the summed per-atom reference features. AF3 sums
# bias-free per-feature Linears and leaves the result unnormalised.
NORMED_ATOM_FEATURES = ESMFOLD2_FAMILY


# Models whose confidence head RE-EMBEDS the pair from s_inputs rather than
# reading the trunk pair directly: z_norm(z) + relpos + bonds + a row, a column
# and an outer PRODUCT of s_inputs, plus a distance-bin embedding of the
# PREDICTED coordinates. boltz2 established the path; ESMFold2 builds the same
# thing, which is why it reuses it rather than adding a second one.
REEMBED_CONFIDENCE_PAIR = ('boltz2',) + ESMFOLD2_FAMILY
ATOM_ROPE_HALF_WINDOW = 64
# 32 queries + 2*64 of context needs 160; the next power-of-two multiple that
# AF3's gather machinery is happy with is 192.
ATOM_KEYS_SUBSET_SIZE = {m: 192 for m in ESMFOLD2_FAMILY}
ATOM_ROPE = {m: dict(n_spatial=2, n_uid=10,
                     spatial_base=20.0, uid_base=10000.0)
             for m in ESMFOLD2_FAMILY}


# Models whose MSA stack does NOT update the MSA representation at all: their
# msa_module block is OuterProductMean + a pair stack, and there is no MSA row
# attention and no MSA transition anywhere in the checkpoint.
#
# EMPTY, and kept rather than deleted. Its only members were protenix_mini and
# protenix_tiny, removed on 2026-09-08: those distillations drop the whole
# `msa_stack` submodule that protenix2 carries
# (`msa_pair_weighted_averaging` + `transition_m`). Building it anyway is not
# free -- `_msa_update` creates 12 parameters per block that no checkpoint can
# fill, so the conversion reports them missing and the graph REFUSES TO APPLY,
# which is how protenix_tiny came to ship a blob that could not be loaded
# against a batch carrying templates.
#
# The forward branch at modules.py stays too. An empty membership costs nothing
# at runtime (`in ()` is always False) and the pair -- convention plus the
# reason -- is what a future distilled checkpoint needs; deleting it would make
# that a rediscovery.
NO_MSA_ROW_UPDATE = ()


# Models whose diffusion conditioning concatenates a PROJECTED relative-position
# encoding with the trunk pair, rather than AF3's raw 139-channel features.
# Widths: AF3 folds [z_trunk(c_z), rel_features(139)] -> c_z, these fold
# [z_trunk(c_z), relpe(c_z)] -> 2*c_z, so getting the membership wrong is a shape
# error at load (267 vs 256) rather than a silent one -- which is why this list
# was the third and last of the protenix_mini omissions to surface (mini itself
# is gone, the lesson is not).
DIFFUSION_PROJECTED_RELPOS = (('boltz2', 'rosettafold3') + ESMFOLD2_FAMILY
                              + PROTENIX_FAMILY)


# Models that compute the column-attention pair bias from the TRANSPOSED pair
# representation, as OpenFold3 preview-2 does. openbind is deliberately absent;
# see the note at modules.py, where the measurement that settled openbind0 lives.
TRANSPOSED_COLUMN_PAIR_BIAS = ('openfold3', 'openbind0', 'opendde', 'boltz2') + PROTENIX_FAMILY


# Models whose TEMPLATE stack adds an OUTER residual around the whole pairformer
# (`v = v + stack(v)`) rather than replacing the activation (`v = stack(v)`).
#
# The blocks are internally residual on both sides, so the outer term adds the
# input a SECOND time -- it is a real difference, not a rearrangement. And the
# vendors genuinely disagree:
#   * boltz2 DOES it: `v = v + self.pairformer(v, pair_mask, ...)` in
#     boltz/model/modules/trunkv2.py.
#   * protenix does NOT: `_, v = self.pairformer_stack(s=None, z=v, ...)`.
#   * rosettafold3 does NOT either, by the same reading:
#     `for block in self.pairformer: _, v_II = block(None, v_II)` in
#     rf3/model/layers/pairformer_layers.py.
#
# Our three template classes share one fused forward, so protenix and rf3
# inherited boltz's convention. For protenix that was worth corr 0.999998 vs
# 1.000000 at L1 (template_parity.py, 2026-09-08) -- small because the stack's
# contribution is small next to the projections, which is also why it hid behind
# a larger bug until that one was fixed.
#
# rosettafold3 is deliberately ABSENT, and for a reason worth writing down: it
# never had this bug, because `RoseTTAFold3TemplateEmbedding` overrides
# `__call__` rather than only `_features`, and its own copy already does
# `v = stack(v)`. So this flag never reaches it -- listing it here was inert,
# which the gate proved: rf3 reads corr 1.000000 with the name in the tuple or
# out of it, bit for bit. Gated 2026-09-08 (template_parity.py).
#
# The moral is not that duplication saved us. protenix inherited the shared
# forward and got the wrong convention; rf3 escaped by not inheriting it. Either
# a per-vendor convention is named -- as it now is here -- or the next subclass
# gets whichever behaviour its parent happened to have.
TEMPLATE_STACK_OUTER_RESIDUAL = ('boltz2',)

# Models whose template FEATURE projection is one fused Linear WITH A BIAS.
#
# AF3 embeds template features as nine separate bias-free Linears whose outputs
# are summed. chai instead runs one `input_projs.TEMPLATES.0`, a Linear(76 -> 64,
# bias=True) over the fused feature stream, so `converters/chai1.py` splits its
# WEIGHT across AF3's slots -- and the bias had nowhere to go and was dropped.
#
# It does not cancel. The 64-d activation goes straight into the template
# pairformer, whose LayerNorms normalise per position across channels: a constant
# vector added at every (i, j) changes each position's mean and variance and so
# its normalised direction. That is unlike the pair-bias LayerNorm offsets in
# rosettafold3/esmfold2 DEAD_TENSORS, which land inside a softmax over j and
# genuinely vanish -- the distinction is WHERE the constant lands, not how big it
# is (chai's is absmax 0.30).
#
# Found by giving chai1 an L0 gate for the first time: its loader returns five
# state dicts rather than one, so `dev/audit_coverage.py` could not run on it at
# all, and 5 of its 1912 tensors were unaccounted the moment it could.
FUSED_TEMPLATE_FEATURE_BIAS = ('chai1',)

# Models that project the token embedding TWICE -- once for the trunk and once
# for the diffusion module -- where AF3 computes one `s_inputs` for both.
#
# chai's TokenInputEmbedding returns `(s_trunk, s_structure, z_init)`:
#     input13     = cat[pooled_atom_single, token_single_input_feats]   # 768
#     s_structure = token_single_proj_in_structure(input13)             # 384
#     s_trunk     = token_single_proj_in_trunk(input13)                 # 384
# read straight off the shipped token_embedder graph, whose return node is
# `TupleConstruct(%s_trunk, %s_structure, %z_init)`. Two independently trained
# Linears over the same input, and the diffusion conditioning was fitted against
# the second one.
#
# Also found by chai1's first L0 gate. `token_single_proj_in_structure.weight`
# read as unaccounted; unlike the four genuinely dead tensors beside it, it is
# called in every `forward_*` of that graph -- which is the check that separates
# "the converter skipped it" from "the vendor ships it and never uses it".
SEPARATE_STRUCTURE_TARGET_FEAT = ('chai1',)


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
# block is left at its init value. Conversion coverage is what catches it, which
# is how the since-removed protenix_mini was caught (104 uncovered parameters).
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
DISTOGRAM_BIAS = (('opendde', 'rosettafold3', 'boltz2') + ESMFOLD2_FAMILY
                  + PROTENIX_FAMILY)


# Models whose PDE head symmetrises the PAIR ACTIVATION before its LayerNorm,
# rather than symmetrising the LOGITS as AlphaFold 3 does.
#
# AF3: `l = Linear(LN(z)); pde = l + l^T`.
# protenix: `pde = Linear(pde_ln(z + z^T))` (confidence.py
# `memory_efficient_forward`). LayerNorm is not linear, so these are different
# functions of z -- unlike the distogram case above this CANNOT be absorbed into
# the weight, which is why it is a forward branch and not a converter transform.
#
# Every other vendor here really does symmetrise the logits, and reading the
# checkpoint cannot tell you which: openfold3 `prediction_heads.py`
# `logits = logits + logits.transpose(-2, -3)`, intellifold2 `pDEHead._forward`
# the same, rosettafold3 the same in its af3-style branch (its OTHER branch
# symmetrises first, but the released config takes the af3 one). boltz2 also
# symmetrises first and is handled by its own split-heads branch.
#
# Found by dev/oracles/confidence_parity.py: pde read corr 0.87 while pae/plddt/
# resolved were at parity, on protenix2. No fold caught it -- a symmetric,
# plausibly-scaled error metric stays symmetric and plausible.
PRE_SYMMETRISED_PDE = PROTENIX_FAMILY


# Models whose diffusion SINGLE conditioning normalises over the vendor's 833
# channels rather than our 831, so the two residue classes AF3 lacks have to be
# re-inserted as zero columns before `single_cond_initial_norm`.
#
# Everywhere else those columns are simply dropped from the weights, because a
# zero input contributes nothing to a bias-free Linear. Not here: a LayerNorm
# maps a zero input to -mean/std, so the vendor always contributes them through
# their trained scale and weight rows AND normalises over a wider vector.
#
# openfold3 has done this from the start. protenix and rosettafold3 were left on
# the 831 path deliberately ("this graph's single_cond_initial_norm spans the
# AF3-width block"), which the L2 conditioning gate priced: single_cond corr
# 0.999989, rms ours/native 0.9987, max|d|/rms 0.11 for protenix2. The pair half
# of the same conditioner is 1.000000 at 3e-06, so this was the whole of the
# difference. Adding a model here REQUIRES its converter to emit the padded
# (833-row) scale and projection -- `_reorder_features_1d(pad_unk_dna=True)` --
# and therefore a re-conversion; the shape mismatch is loud if you forget.
PADDED_SINGLE_COND = (('openfold3', 'openbind0', 'rosettafold3')
                      + PROTENIX_FAMILY)


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
