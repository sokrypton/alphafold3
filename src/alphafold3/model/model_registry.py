# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Per-model settings: what each AF3-family model needs beyond the graph itself.

`global_config.model` names which family's forward branches run; this module
carries everything else that name implies -- the config-shape divergences (each
family widened its own channels, head counts and bin counts), the trained
Fourier noise embedding, the EDM sampler constants, and where the converted
weights live.

Runtime only. Conversion is offline tooling in `converters/` at the repo root;
nothing here imports torch or knows what a checkpoint looks like.
"""

from __future__ import annotations

from alphafold3.model import model_config

# IntelliFold-v2's "full_fat" preset: the four channels it widens over stock AF3.
# (scope, attribute) pairs resolved against model.Model.Config(); everything not
# listed is byte-identical to public AF3. Values from IntelliFold's own
# widen_config_full_fat (patches.py) / convert.py V2_HEADS (c_z=512, c_m=256,
# c_t=256, diffusion-conditioning pair=512).
FULL_FAT_CHANNELS = (
    ('evoformer.pair_channel', 512),
    ('evoformer.msa_channel', 256),
    ('evoformer.template.num_channels', 256),
    ('heads.diffusion.conditioning.pair_channel', 512),
)


# IntelliFold-v2 also raises the pair/template/msa attention head counts to 8
# (its converter's V2_HEADS: pair=8, template=8, msa=8; stock AF3 GridSelfAttention
# defaults to 4). Every GridSelfAttention in the fork is a pair- or template-axis
# attention, and MSAAttention is the msa one -- so widening num_head on those two
# config types everywhere (trunk pairformer, template embedder, evoformer msa
# stack, AND the confidence head's pairformer) is exactly V2_HEADS, no path list
# to drift. single (16) / token (16) / atom (4) attentions use other config types
# and are unchanged. All four full_fat channels stay divisible by 8.
# IntelliFold-v2 also raises the pair/template/msa attention head counts to 8
# (its converter's V2_HEADS: pair=8, template=8, msa=8; stock AF3
# GridSelfAttention defaults to 4). Every GridSelfAttention in this graph is a
# pair- or template-axis attention and MSAAttention is the msa one, so widening
# num_head on those two config types everywhere -- trunk pairformer, template
# embedder, evoformer msa stack, AND the confidence head's pairformer -- is
# exactly V2_HEADS, with no path list to drift. single (16) / token (16) / atom
# (4) attentions use other config types and are unchanged. All four full_fat
# channels stay divisible by 8.
FULL_FAT_NUM_HEAD = 8


def _widen_full_fat(cfg):
  '''widen cfg in place to IntelliFold-v2 dims; raise if a channel path is missing
  so a config-layout drift is caught loudly rather than silently loading zeros.'''
  for dotted, value in FULL_FAT_CHANNELS:
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
      if not hasattr(node, p):
        raise AttributeError(
            f'full_fat: config path {dotted!r} not found (missing {p!r}); '
            'AF3 config layout drift -- re-check FULL_FAT_CHANNELS')
      node = getattr(node, p)
    setattr(node, parts[-1], value)

  # raise every pair/template/msa attention to 8 heads by walking the config tree
  from alphafold3.model.network import modules as _modules
  head_types = (_modules.GridSelfAttention.Config, _modules.MSAAttention.Config)
  seen = set()
  def _walk(node):
    if id(node) in seen:
      return
    seen.add(id(node))
    if isinstance(node, head_types):
      node.num_head = FULL_FAT_NUM_HEAD
      return
    for v in list(getattr(node, '__dict__', {}).values()):
      if isinstance(v, (list, tuple)):
        for x in v:
          _walk(x)
      else:
        _walk(v)
  _walk(cfg)


# OpenDDE's width/head/bin profile (its production config, model_base.py). Unlike
# full_fat's uniform widening, OpenDDE's heads are NON-uniform (trunk triangle
# attention 12, template 2), so these are set by explicit config path. Each entry
# (dotted-path, value); attention head dims stay 32 (c / num_head).
OPENDDE_SETTINGS = (
    ('evoformer.pair_channel', 384),                              # c_z
    ('evoformer.msa_channel', 128),                               # c_m
    ('evoformer.pairformer.pair_attention.num_head', 12),         # trunk tri-att, 384/32
    ('evoformer.msa_stack.pair_attention.num_head', 12),          # MSA-module pair stack tri-att
    ('evoformer.msa_stack.msa_attention.num_head', 8),            # MSA pair-weighted-averaging
    ('evoformer.msa_stack.msa_attention.value_dim', 8),           # decoupled per-head width (blocker)
    ('evoformer.template.template_stack.pair_attention.num_head', 2),  # template tri-att, 64/32
    ('heads.confidence.pairformer.pair_attention.num_head', 12),  # confidence pairformer (c_z=384)
    ('heads.distogram.num_bins', 96),                             # OpenDDE no_bins=96 (vs AF3 64)
)


def _widen_opendde(cfg):
  '''set cfg in place to OpenDDE's production dims/heads/bins; raise on a missing
  path so a config-layout drift is caught loudly rather than silently mis-shaping.'''
  for dotted, value in OPENDDE_SETTINGS:
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
      if not hasattr(node, p):
        raise AttributeError(
            f'opendde: config path {dotted!r} not found (missing {p!r}); '
            'AF3 config layout drift -- re-check OPENDDE_SETTINGS')
      node = getattr(node, p)
    setattr(node, parts[-1], value)


# Boltz-2's config divergence from stock AF3 is tiny: same channels (c_z=128, c_s=384),
# same diffusion nesting (24 blocks / super_block_size 4 = 6 super) and atom/template/msa
# counts -- only the pairformer (64 vs 48) and confidence pairformer (8 vs 4) are deeper.
# The rest of Boltz's differences are FORWARD branches (a_to_b conditioned transition, the
# summed input embedder, the concat single_conditioner, MSA update-then-OPM order) gated
# on instance flags, not config shape. (WIP: only boltz2_cond_transition is wired so far.)
BOLTZ2_SETTINGS = (
    ('evoformer.pairformer.num_layer', 64),
    ('heads.confidence.pairformer.num_layer', 8),
    # Boltz's diffusion single conditioning s = 2*token_s = 768 (AF3's is seq_channel=384).
    # The token transformer adaln takes this full 768-d s, so widen the diffusion single
    # conditioning channel. per_token_channels (the token act) is already 768.
    ('heads.diffusion.conditioning.seq_channel', 768),
    # Boltz's MSA pair-weighted-averaging hidden = 8 heads x 32 = 256 (proj_m/proj_g are
    # (256,64), proj_o (64,256)). msa_attention.value_dim is PER-HEAD here (default None ->
    # 8), so 32 gives 8x32 = 256 total, not 256.
    ('evoformer.msa_stack.msa_attention.value_dim', 32),
    # Boltz template pairformer: pairwise_head_width=32 (AF3 would derive 64//4=16) and a
    # 4x transition (template_stack defaults to 2x). Only used on the boltz2 template path.
    ('evoformer.template.template_stack.pair_attention.qkv_dim', 32),
    ('evoformer.template.template_stack.pair_transition.num_intermediate_factor', 4),
)


def _widen_boltz2(cfg):
  '''set cfg in place to Boltz-2's block counts; raise on a missing path.'''
  for dotted, value in BOLTZ2_SETTINGS:
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
      if not hasattr(node, p):
        raise AttributeError(f'boltz2: config path {dotted!r} not found (missing {p!r})')
      node = getattr(node, p)
    setattr(node, parts[-1], value)


# Protenix-v2 (ByteDance, Apache-2.0): AF3-lineage, OpenFold-derived like OF3. Its ONLY
# architectural divergence from stock AF3 is the "hidden_scale_up" widening of the trunk
# pair channel to c_z=256 (stock is 128). Everything derived from c_z follows: triangle
# multiplication hidden = c_z = 256, and triangle attention heads = c_z//32 = 8 (stock 4).
# The diffusion stack is UNWIDENED (token 768, single cond 384, same as stock/of3) -- only
# the trunk carries the wider pair rep. Block counts match AF3 defaults (pairformer 48,
# msa 4, template 2, confidence 4, distogram 64 bins). c_m stays 128. So the config delta
# is just the pair channel + the derived per-stack head counts (set explicitly so a config
# layout drift is caught loudly). Protenix-v2 is in OPENFOLD3_LINEAGE (it rides the OF3
# forward branches) + trained Fourier; its Protenix-specific bits are the CONVERTER's
# naming/feature conventions, not forward-graph shape. See converters/protenix2.py +
# memory protenix-v2-port.md.
PROTENIX2_SETTINGS = (
    ('evoformer.pair_channel', 256),                              # c_z (hidden_scale_up)
    ('evoformer.msa_channel', 128),                               # c_m (unchanged)
    ('evoformer.pairformer.pair_attention.num_head', 8),          # trunk tri-att, 256/32
    ('evoformer.msa_stack.pair_attention.num_head', 8),           # MSA-module pair stack tri-att
    ('evoformer.msa_stack.msa_attention.num_head', 8),            # MSA pair-weighted-averaging
    ('evoformer.msa_stack.msa_attention.value_dim', 8),           # per-head width (8*8=64 = proj_m/g)
    ('evoformer.template.template_stack.pair_attention.num_head', 2),  # template tri-att, 64/32
    ('heads.confidence.pairformer.pair_attention.num_head', 8),   # confidence pairformer (c_z=256)
    # Protenix's DiffusionConditioning pair path is also widened to c_z=256 (its relpe
    # projects to 256, cat([z_trunk(256), relpe(256)])->layernorm_z(512)->256). The
    # diffusion conditioning pair_channel defaults to 128, so widen it too or the
    # diffusion pair_cond/transition params mis-shape (structural gate caught this).
    ('heads.diffusion.conditioning.pair_channel', 256),
)


# Protenix's SMALL model types (mini, tiny). Everything they change from the
# v0.5.0 base is a COUNT: 16 / 8 pairformer blocks against 48, one MSA block
# against 4, an 8-block diffusion transformer against 24, and a single-block
# diffusion atom encoder/decoder against 3. They keep stock AF3 widths (c_z=128,
# 4 tri-attention heads), unlike protenix2, so there is no widening here at all.
#
# Two things worth stating because they are easy to get wrong:
#   * the INPUT-EMBEDDER atom encoder stays at 3 blocks while the DIFFUSION one
#     drops to 1, so the two need separate knobs -- one `n_atom` would be wrong
#     for one of them.
#   * these are v0.5.0-lineage and TEMPLATELESS (the checkpoints carry zero
#     template_embedder blocks), so the template stack is set to 0.
# All of it is verified against the checkpoints by converters/protenix2.derive_dims,
# which reads the same numbers off the artefact rather than trusting this table.
_PROTENIX_SMALL_COMMON = (
    ('evoformer.msa_stack.num_layer', 1),
    ('evoformer.template.template_stack.num_layer', 0),
    ('heads.diffusion.transformer.num_blocks', 8),
    ('heads.diffusion.transformer.super_block_size', 4),
    ('heads.diffusion.atom_transformer.num_blocks', 1),
    ('evoformer.per_atom_conditioning.atom_transformer.num_blocks', 3),
)
PROTENIX_MINI_SETTINGS = _PROTENIX_SMALL_COMMON + (
    ('evoformer.pairformer.num_layer', 16),
)
PROTENIX_TINY_SETTINGS = _PROTENIX_SMALL_COMMON + (
    ('evoformer.pairformer.num_layer', 8),
)


def _apply_settings(cfg, settings, who):
  for dotted, value in settings:
    node = cfg
    parts = dotted.split('.')
    for part in parts[:-1]:
      if not hasattr(node, part):
        raise AttributeError(f'{who}: config path {dotted!r} not found (missing {part!r})')
      node = getattr(node, part)
    setattr(node, parts[-1], value)


# Protenix-v1 (368 M). Its counts are IDENTICAL to protenix2's -- same 4174
# tensors, same 48/4/2/24/3 blocks -- and it sits at stock AlphaFold 3 widths
# throughout (c_z 128, c_m 64, 4 tri-attention heads, msa value_dim 8), so almost
# nothing needs setting.
#
# The one exception is real and is what stalled an earlier attempt at this port:
# v1's TEMPLATE stack runs a 128-wide triangle multiplication on a 64-channel
# template pair. AF3 ties the two, so its projections would be (64, 64) where the
# checkpoint carries (128, 64). That needed a hidden_dim knob on
# TriangleMultiplication, which defaults to None and leaves every other model
# byte-identical.
PROTENIX1_SETTINGS = (
    ('evoformer.template.template_stack.triangle_multiplication_outgoing.hidden_dim', 128),
    ('evoformer.template.template_stack.triangle_multiplication_incoming.hidden_dim', 128),
    # and its template attention is 4 heads x 32 = 128 hidden on the same
    # 64-channel pair, where AF3 derives max(64 // 4, 16) = 16. Same story as the
    # tri-mul above: v1's template stack is uniformly twice as wide as its
    # channel count would imply.
    ('evoformer.template.template_stack.pair_attention.qkv_dim', 32),
)


def _widen_protenix1(cfg):
  _apply_settings(cfg, PROTENIX1_SETTINGS, 'protenix1')


def _widen_protenix1_20250630(cfg):
  # same graph as protenix1, different training run
  _apply_settings(cfg, PROTENIX1_SETTINGS, 'protenix1_20250630')


# Protenix v0.5.0 base. Stock AlphaFold 3 throughout -- v1's template widths do
# not apply because there is no template stack to widen.
PROTENIX05_SETTINGS = (
    ('evoformer.template.template_stack.num_layer', 0),
)


def _widen_protenix05(cfg):
  _apply_settings(cfg, PROTENIX05_SETTINGS, 'protenix05')


def _widen_protenix_mini(cfg):
  _apply_settings(cfg, PROTENIX_MINI_SETTINGS, 'protenix_mini')


def _widen_protenix_tiny(cfg):
  _apply_settings(cfg, PROTENIX_TINY_SETTINGS, 'protenix_tiny')


def _widen_protenix2(cfg):
  '''set cfg in place to Protenix-v2's c_z=256 trunk widening; raise on a missing path.'''
  for dotted, value in PROTENIX2_SETTINGS:
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
      if not hasattr(node, p):
        raise AttributeError(f'protenix: config path {dotted!r} not found (missing {p!r})')
      node = getattr(node, p)
    setattr(node, parts[-1], value)


# RoseTTAFold3 (Baker lab / RosettaCommons foundry): AF3-family, adopted the AF3 architecture
# (pairformer + af3 diffusion + af3 losses). STOCK AF3 dims (c_z=128, tri-att 4 heads, tri-mul
# hidden 128 -- all AF3 defaults), so unlike protenix there is NO trunk widening. The only
# config-shape divergences are the distogram bin count (65 vs 64) and the MSA depth (a single
# weight-tied module, not a 4-block stack). Everything else is a FORWARD divergence gated on the
# rf3 flag: kq_norm (q/k LayerNorm in diffusion attention), no_residual (diffusion block wiring),
# single/tied MSA, fused template (=protenix branch). See converters/rosettafold3.py + memory
# rosettafold3-port.md. (WIP: converter trunk+heads+diffusion(cond+token) done; atom path + branches next.)
ROSETTAFOLD3_SETTINGS = (
    ('heads.distogram.num_bins', 65),          # RF3 distogram = 65 bins (vs AF3 64)
    # RF3's MSAModule holds ONE set of weights and runs them n_block=4 times
    # (rf3_net.yaml msa_module.n_block; pairformer_layers.MSAModule.forward loops
    # `for i in range(self.n_block)` over the same submodules). The single copy in
    # the state dict is weight TYING, not a single iteration -- running it once
    # leaves the trunk three MSA/pair updates short, which no structural gate can
    # see. The converter replicates the one block across all four layers.
    ('evoformer.msa_stack.num_layer', 4),
    ('evoformer.msa_stack.msa_attention.value_dim', 32),  # RF3 msa pwa hidden 8x32=256 (AF3 default 8)
    # RF3 template pairformer: per-head qkv=64 (4 heads x 64 = 256 hidden; AF3 derives
    # 64//4=16) and a 4x transition (template_stack defaults to 2x).
    ('evoformer.template.template_stack.pair_attention.qkv_dim', 64),
    ('evoformer.template.template_stack.pair_transition.num_intermediate_factor', 4),
)


# chai-1 (chaidiscovery). Read off the JIT state_dict shapes, not a config file --
# chai ships no model source, only frozen TorchScript, so every number here was
# derived from a weight shape (see memory chai1-port.md for the dump). The trunk is
# c_z=256 like protenix, but chai widens/narrows far more than the pair channel:
#   - transitions are 2x in the pairformer and confidence stacks (AF3 is 4x), while
#     the MSA module keeps 4x. Read off linear_out: pairformer pair (256,512) = 2x256,
#     MSA pair (256,1024) = 4x256.
#   - the MSA pair-weighted-averaging head is 8 heads x 32 (AF3 derives 64//8 = 8):
#     linear_msa2vg (512,64) = v(256) | g(256).
#   - the template pairformer runs 4 heads x 32 = 128 hidden at c=64 (AF3 derives
#     64//4 = 16): pair2qkvg1 (512,64) = 4 x 128.
#   - the diffusion transformer is 16 blocks, not AF3's 24, and its conditioning pair
#     path carries the full c_z=256.
# num_outer_channel is chai's OPM GROUP size: its outer product mean is a grouped
# rank-8 einsum (weight_ab (2,8,8,64) -> 8*8*8 = 512 channels -> 256), not AF3's
# left/right projection to 32, so the knob means something different under the chai
# branch and 8 is the group width it reads.
CHAI1_SETTINGS = (
    ('evoformer.pair_channel', 256),                                    # c_z
    ('evoformer.msa_channel', 64),                                      # c_m (unchanged)
    ('evoformer.pairformer.pair_transition.num_intermediate_factor', 2),
    ('evoformer.pairformer.single_transition.num_intermediate_factor', 2),
    ('evoformer.msa_stack.msa_attention.value_dim', 32),                # 8 heads x 32
    ('evoformer.msa_stack.outer_product_mean.num_outer_channel', 8),    # grouped OPM width
    ('evoformer.template.template_stack.pair_attention.qkv_dim', 32),   # 4 heads x 32 at c=64
    ('heads.diffusion.transformer.num_blocks', 16),                     # AF3 has 24
    ('heads.diffusion.conditioning.pair_channel', 256),
    ('heads.confidence.pairformer.pair_transition.num_intermediate_factor', 2),
    ('heads.confidence.pairformer.single_transition.num_intermediate_factor', 2),
    # CONFIDENCE only, not the trunk: chai's confidence triangle attention has a
    # (2*c_z, 2*c_z) output projection whose two halves are combined as
    # `kept + transpose(other)`, so each direction needs two output projections.
    # The trunk's is (c_z, 2*c_z), which is AF3's per-direction pair summed, and
    # must stay on the ordinary path.
    ('heads.confidence.pairformer.pair_attention.dual_output', True),
    # chai's EDM schedule. Its InferenceNoiseSchedule multiplies by sigma_data
    # exactly as ours does, but with S_tmax = 80 where AF3 uses 160 -- so AF3
    # starts sampling at 16 * 160 = 2560 and chai at 16 * 80 = 1280. The seam
    # confirms it: chai's step-0 noise_sigma is 1261.6. Starting at twice the
    # noise the denoiser was trained for is not a small error.
    ('heads.diffusion.eval.sigma_max', 80.0),
)


def _widen_chai1(cfg):
  '''set cfg in place to chai-1's dims; raise on a missing path.'''
  for dotted, value in CHAI1_SETTINGS:
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
      if not hasattr(node, p):
        raise AttributeError(f'chai1: config path {dotted!r} not found (missing {p!r})')
      node = getattr(node, p)
    setattr(node, parts[-1], value)


def _widen_rosettafold3(cfg):
  '''set cfg in place to RF3's (minimal) config divergences; raise on a missing path.'''
  for dotted, value in ROSETTAFOLD3_SETTINGS:
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
      if not hasattr(node, p):
        raise AttributeError(
            f'rosettafold3: config path {dotted!r} not found (missing {p!r})')
      node = getattr(node, p)
    setattr(node, parts[-1], value)


# EDM sampler constants per model family. AF3's values are NOT universal -- each
# family trained its own, and running boltz2's network under AF3's constants was a
# silent mismatch (nothing errors; the sampler just anneals on the wrong schedule).
# Only entries verified against the native implementation belong here; a model
# absent from this table keeps AF3's defaults, which is the honest state for one
# nobody has checked.
#   boltz2: boltz.main.Boltz2DiffusionParams (BoltzDesign1/boltz2/src/boltz/main.py)
_SAMPLER_CONSTANTS = {
    # Protenix's small models ship their OWN sampler settings and they are not
    # small differences: 5 steps against 200, no churn (gamma0 0 against 0.8) and
    # step_scale 1.0 against 1.5 (configs_model_type.py, "the default setting for
    # mini model"). Running them on AF3's constants would anneal on the wrong
    # schedule 40x too slowly -- silent, as ever; nothing errors.
    'protenix_mini': dict(gamma_0=0.0, step_scale=1.0, steps=5),
    'protenix_tiny': dict(gamma_0=0.0, step_scale=1.0, steps=5),
    'boltz2': dict(gamma_0=0.605, gamma_min=1.107, noise_scale=0.901,
                   step_scale=1.638, rho=8.0, sigma_min=0.0004, sigma_max=160.0),
    # ESMFold2 samples in FOURTEEN steps, not 200, and clips the schedule at
    # sigma 256 -- the EDM schedule opens at sigma_data * smax = 2560, so
    # running AF3's constants starts the trajectory with ten times the noise the
    # model was trained to undo, and then anneals it fourteen times too slowly.
    # Every ESMFold2 variant BUT ONE samples the same way.
    **{m: dict(steps=14, max_sigma=256.0)
       for m in model_config.ESMFOLD2_FAMILY},
    # The 600M tier does not: its config carries a different EDM schedule
    # entirely -- and it is Boltz-2's, constant for constant (gamma_0 0.605,
    # gamma_min 1.107, noise_scale 0.901, step_scale 1.638, rho 8). Found by
    # check_release_config refusing the conversion, which is exactly the failure
    # it exists for: nothing about the wrong schedule errors, it just anneals
    # differently and returns a plausible structure.
    **{m: dict(steps=15, gamma_0=0.605, gamma_min=1.107,
               noise_scale=0.901, step_scale=1.638, rho=8.0,
               sigma_min=0.0004, sigma_max=160.0, max_sigma=256.0)
       for m in ('esmfold2_lm600m', 'esmfold2_lm300m')},
}



# Config-shape divergences, one widener per family that has any. A family absent
# here runs at stock AF3 dimensions.
# The trunk depth is the ONLY thing that differs across the family: "Fast" is
# 24 pair-only blocks where the base model is 48. Read off each release's
# config.json, not guessed.
# One row per ESMFold2 release, read off its own config.json and confirmed
# against the tensor inventory -- NOT assumed from the name. "Fast" is not only
# a shallower trunk: it also turns the MSA encoder OFF (msa_encoder.enabled
# false, 0 msa_encoder tensors), so it is ESM-C-only. And the EXPERIMENTAL line
# is a different architecture again: no parcae, no coda, no lm_encoder stack,
# a wider MSA head and a 128-bin confidence distogram. "Fast" ALWAYS means no
# MSA encoder, on both lines -- convert_esmfold2_weights asserts every row here
# against the checkpoint, because this table was wrong twice before it did.
#
#   hub      the biohub repo it is converted from (all six share ESM-C 6B)
#   trunk    folding_trunk.n_layers
#   msa      msa_encoder.n_layers, or 0 where the encoder is disabled
#   coda     parcae.coda_n_layers, 0 without parcae
#   lm_enc   lm_encoder.n_layers, 0 where the release has no lm_encoder
#   msa_w    msa_encoder.msa_head_width
#   bins     distogram_head bins (the TRUNK head, not confidence)
#   conf_bins classes in the confidence re-embedding distance one-hot
#            (its trained `boundaries` has one fewer)
#   esmc     WHICH ESM-C tower this release was trained against. Not
#            cosmetic: the shim is per-model and so is the tower, and
#            pairing a variant with the wrong one is silent -- the base
#            shim on another variant's hidden states reads corr 0.026.
ESMFOLD2_VARIANTS = {
    'esmfold2': dict(hub='ESMFold2', trunk=48, msa=4, coda=2, lm_enc=4,
                     msa_w=16, bins=64, conf_bins=39, esmc='esmc'),
    'esmfold2_fast': dict(hub='ESMFold2-Fast', trunk=24, msa=0, coda=2, lm_enc=4,
                          msa_w=0, bins=64, conf_bins=39, esmc='esmc'),
    'esmfold2_exp': dict(hub='ESMFold2-Experimental', trunk=48, msa=4, coda=0,
                         lm_enc=0, msa_w=32, bins=128, conf_bins=128, esmc='esmc'),
    'esmfold2_exp_fast': dict(hub='ESMFold2-Experimental-Fast', trunk=24, msa=0,
                              coda=0, lm_enc=0, msa_w=0, bins=128, conf_bins=128, esmc='esmc'),
    'esmfold2_exp_cutoff2025': dict(hub='ESMFold2-Experimental-Cutoff2025',
                                    trunk=48, msa=4, coda=0, lm_enc=0, msa_w=32,
                                    bins=128, conf_bins=128, esmc='esmc'),
    'esmfold2_exp_fast_cutoff2025': dict(
        hub='ESMFold2-Experimental-Fast-Cutoff2025', trunk=24, msa=0, coda=0,
        lm_enc=0, msa_w=0, bins=128, conf_bins=128, esmc='esmc'),
    # The 600M-ESM-C tier. Architecturally identical to esmfold2_exp_fast --
    # 24 blocks, no MSA encoder, no lm_encoder, no parcae -- and trained
    # against a DIFFERENT tower, which is the only thing `esmc` records and the
    # only thing that makes it a separate model. step1500k is the last
    # checkpoint of upstream's scaling series, i.e. the best of that tier.
    'esmfold2_lm600m': dict(
        hub='ESMFold2-Experimental-Fast-base600M-step1500k', trunk=24, msa=0,
        coda=0, lm_enc=0, msa_w=0, bins=128, conf_bins=128, esmc='esmc_600m'),
    # Identical to the row above in every field but the tower: the two configs
    # differ only in esmc_id, lm_d_model (960 vs 1152) and lm_num_layers (30 vs
    # 36), all of which live in the LM, not the folding trunk.
    'esmfold2_lm300m': dict(
        hub='ESMFold2-Experimental-Fast-base300M-step1500k', trunk=24, msa=0,
        coda=0, lm_enc=0, msa_w=0, bins=128, conf_bins=128, esmc='esmc_300m'),
}

ESMFOLD2_HUB_IDS = {m: v['hub'] for m, v in ESMFOLD2_VARIANTS.items()}

ESMFOLD2_SETTINGS = (
    # trunk: PAIR-ONLY blocks at c_z 256 (AF3 is 128), no templates. The block
    # COUNT is per-variant, applied after this tuple.
    ('evoformer.pair_channel', 256),
    ('evoformer.msa_channel', 128),
    ('evoformer.template.template_stack.num_layer', 0),
    ('evoformer.per_atom_conditioning.atom_transformer.num_blocks', 3),
    # diffusion: 12 token blocks in (3, 4) supers, 16 heads, 3 atom blocks
    ('heads.diffusion.transformer.num_blocks', 12),
    ('heads.diffusion.transformer.super_block_size', 4),
    ('heads.diffusion.transformer.num_head', 16),
    ('heads.diffusion.conditioning.pair_channel', 256),
    # ESMFold2's diffusion single is 768 wide, not AF3's 384
    ('heads.diffusion.conditioning.seq_channel', 768),
    ('heads.diffusion.atom_transformer.num_blocks', 3),
)


def _widen_esmfold2(name):
  """-> a widener for one ESMFold2 variant. The family shares every setting
  except the trunk depth, so the depth rides in as a per-name row rather than as
  six near-identical tuples."""
  v = ESMFOLD2_VARIANTS[name]

  def widen(cfg):
    _apply_settings(cfg, ESMFOLD2_SETTINGS, name)
    _apply_settings(cfg, (
        ('evoformer.pairformer.num_layer', v['trunk']),
        ('evoformer.msa_stack.num_layer', v['msa']),
        ('evoformer.coda.num_layer', v['coda']),
        ('evoformer.lm_encoder.num_layer', v['lm_enc']),
        ('evoformer.msa_stack.msa_attention.value_dim', v['msa_w']),
        ('heads.distogram.num_bins', v['bins']),
        ('heads.confidence.reembed_dist_bins', v['conf_bins']),
    ), name)
  return widen


_WIDENERS = {
    **{m: _widen_esmfold2(m) for m in model_config.ESMFOLD2_FAMILY},
    'opendde': _widen_opendde,
    'boltz2': _widen_boltz2,
    'protenix2': _widen_protenix2,
    'protenix05': _widen_protenix05,
    'protenix1': _widen_protenix1,
    'protenix1_20250630': _widen_protenix1_20250630,
    'protenix_mini': _widen_protenix_mini,
    'protenix_tiny': _widen_protenix_tiny,
    'rosettafold3': _widen_rosettafold3,
    'chai1': _widen_chai1,
}


# Featurisation knobs each family REQUIRES: conventions its weights were trained
# under that the input pipeline has to reproduce. Not preferences -- getting one
# wrong is silent, and shows up as a fold that is merely mediocre.
_FEATURISE = {
    # ESMFold2 attends +/-64 atoms by rank, which needs 32 + 2*64 = 160 keys of
    # context around a query block; AF3's default 128 is too narrow, so widen the
    # key subset and let the exact window ride in as a mask.
    **{m: dict(atom_keys_subset_size=192, lm_pair=True)
       for m in model_config.ESMFOLD2_FAMILY},
    # boltz2 keeps a modified residue as ONE token holding all its atoms
    # (data/tokenize/boltz2.py: standard -> per residue, NONPOLYMER -> per atom,
    # else -> one token, all atoms). AF3 atomises instead, and handing boltz2 ten
    # single-atom tokens where it wants one ten-atom token inflates the residue
    # ~2.4x. Inert when nothing is modified, and ligands atomise either way.
    'boltz2': dict(modified_as_one_token=True),
    # opendde runs its diffusion on an expanded structural-token set, and pads
    # the atom key window rather than sliding it in bounds.
    # struct_num_tokens is deliberately absent: the structural-token count
    # depends on the input, so a fixed bucket only works for inputs small enough
    # to fit it ("Can't pad to a smaller shape" for anything larger). Left unset,
    # attach_structural_batch rounds the true count up to a multiple of 32, which
    # keeps shapes stable across similar inputs without capping them.
    'opendde': dict(opendde=True, padded_keys=True),
    'protenix2': dict(padded_keys=True),
    # rf3 (atomworks) renames atomised atoms to their ELEMENT symbol, carries
    # chirality features, aligns restypes to its own alphabet, and calls an
    # atomised polymer token UNKNOWN where AlphaFold 3 keeps the parent residue
    # type (which is what asserted the L enantiomer over 11 D-amino acids).
    'rosettafold3': dict(chirals=True, atomized_element_names=True,
                         restype_alignment=True,
                         atomized_unknown_restype=True,
                         atomized_backbone_bonds=True),
    # chai-1's four input conventions, every one of them silent when forgotten:
    # it takes the atom key window MODULO the atom count where AF3 slides it back
    # in bounds; it numbers its atoms without the C-terminal OXT; it carries its
    # own standard-residue conformers (ligands fall through to the CCD, as chai
    # does too); and with no alignment it feeds an all-gap MSA rather than a
    # depth-1 self-MSA. It also takes ESM2 embeddings, which are optional but not
    # minor -- without them a natural protein folds to 5.70 A where chai reaches
    # 0.642.
    # std_conformers is deliberately absent: see converters/publish.py. chai's
    # own standard-residue conformers are worth 0.08 A on 6MRR (1.698 vs 1.776)
    # and we do not have the licence to redistribute them, so the CCD ideal
    # values every other model uses are the default. Pass
    # std_conformers='std_conformers.npz' here, with the file beside the blob,
    # to restore them.
    'chai1': dict(circular_keys=True, drop_atoms=('OXT',),
                  zero_msa_without_alignment=True, esm=True),
}


# Where a converted model's weights are published. The file name is what
# `converters/convert.py` writes; the repo is where we upload it. AlphaFold3
# itself is absent: DeepMind's blob is not ours to redistribute, and its terms
# require you to request it (--model_dir points at your own copy).
_WEIGHTS_REPO = 'sokrypton/af3-any-model'


# One folder per family in the published repo. The families that already have a
# membership table are read from it rather than restated; the rest are their own
# folder. openbind0 rides with openfold3 because it IS an OpenFold3 release.
WEIGHTS_FOLDERS = {
    **{m: 'protenix' for m in model_config.PROTENIX_FAMILY},
    **{m: 'esmfold2' for m in model_config.ESMFOLD2_FAMILY},
    'openfold3': 'openfold3',
    'openbind0': 'openfold3',
}

# The protein language models. Not ModelSpecs -- separate graphs with their own
# loader -- so they carry their folder here.
TOWER_FOLDER = 'lm'


class ModelSpec:
  """Everything `global_config.model = name` implies, in one place."""

  __slots__ = ('name', 'full_fat', 'trained_fourier', 'featurise',
               'weights_repo', 'weights_file', 'weights_prefix', 'weights_licence',
               'weights_source')

  # Which network this spec's weights run on. Everything in MODELS rides the
  # AF3 graph; AF2 does not (see AF2Spec). Read this rather than testing the
  # name against a tuple -- a name test is the thing that silently admits a new
  # model to the wrong graph.
  engine = 'af3'

  def __init__(self, name, full_fat=False, trained_fourier=None,
               weights_repo=_WEIGHTS_REPO, weights_file=None,
               weights_licence=None, weights_source=None):
    from alphafold3.model import model_config

    if name not in model_config.MODELS:
      raise ValueError(
          f'unknown model {name!r}; known: {list(model_config.MODELS)}')
    self.name = name
    self.full_fat = full_fat
    # chai-1 also carries a trained Fourier embedding but is deliberately not in
    # OPENFOLD3_LINEAGE (it is not OpenFold-derived), so it has to be named.
    if trained_fourier is None:
      trained_fourier = (name in model_config.OPENFOLD3_LINEAGE
                         or name == 'chai1')
    self.trained_fourier = trained_fourier
    self.featurise = dict(_FEATURISE.get(name, {}))
    self.weights_repo = weights_repo
    self.weights_file = weights_file or f'{name}.bin.zst'
    # Where it lives IN the repo. A flat repo of forty-odd blobs is unreadable;
    # one folder per family is. This is the published path only -- locally a
    # model still keeps its own directory named after itself.
    self.weights_prefix = WEIGHTS_FOLDERS.get(name, name)
    # What the OUTPUT of a run may be used for is the weights' licence, not this
    # code's. None means we have not established it -- say so and point at the
    # source rather than guess, because guessing here would be an assurance we
    # cannot give.
    self.weights_licence = weights_licence
    self.weights_source = weights_source

  # Storage precisions published beside the float32 blob. float32 stays the
  # default and keeps its original filename, so every link already handed out
  # resolves to the same bytes; the smaller forms are additions, not
  # replacements. See converters/quantise.py for what each costs.
  PRECISIONS = ('fp32', 'fp16', 'int8')

  def weights_path_for(self, precision='fp32'):
    """The path INSIDE the weights repo, folder included."""
    return '%s/%s' % (self.weights_prefix, self.weights_file_for(precision))

  def companion_path(self, filename):
    """The repo path of a companion artifact (an ESMFold2 shim, say)."""
    return '%s/%s' % (self.weights_prefix, filename)

  def weights_file_for(self, precision='fp32'):
    """The published filename of this model's weights at `precision`."""
    if precision not in self.PRECISIONS:
      raise ValueError(f'unknown precision {precision!r}, want one of '
                       f'{self.PRECISIONS}')
    if precision == 'fp32':
      return self.weights_file
    stem = self.weights_file.split('.', 1)[0]
    return f'{stem}.{precision}.bin.zst'

  def output_terms(self):
    """The terms-of-use text to write beside a prediction made with this model."""
    if self.name == 'alphafold3':
      import os
      import alphafold3.cpp

      # Packaged, the terms sit beside the built extension; in a source checkout
      # they are at the repo root, three levels above this file.
      here = os.path.dirname(os.path.abspath(__file__))
      for path in (
          os.path.join(os.path.dirname(alphafold3.cpp.__file__),
                       'OUTPUT_TERMS_OF_USE.md'),
          os.path.join(here, '..', '..', '..', 'OUTPUT_TERMS_OF_USE.md'),
      ):
        if os.path.isfile(path):
          with open(path) as fh:
            return fh.read()
      raise FileNotFoundError('OUTPUT_TERMS_OF_USE.md not found')
    where = self.weights_source or 'the model\'s own distribution'
    if self.weights_licence is None:
      return (
          '# OUTPUT TERMS OF USE\n\n'
          f'These structure predictions were generated using {self.name} model\n'
          'weights, run through AlphaFold 3 code (copyright Google DeepMind,\n'
          'Apache 2.0: https://github.com/google-deepmind/alphafold3).\n\n'
          'The AlphaFold 3 Output Terms of Use do NOT apply to these outputs --\n'
          'they were not produced with AlphaFold 3 parameters. What DOES apply is\n'
          f'the licence those weights are distributed under, which we have not\n'
          f'established here. Check it before you rely on these outputs:\n'
          f'  {where}\n')
    return (
        '# OUTPUT TERMS OF USE\n\n'
        f'These structure predictions were generated using {self.name} model\n'
        f'weights, which are licensed under {self.weights_licence}.\n\n'
        'The AlphaFold 3 Output Terms of Use (which restrict commercial use) do\n'
        'NOT apply to these outputs -- they were not produced with AlphaFold 3\n'
        f'parameters. Use is subject to {self.weights_licence} alone.\n\n'
        f'Weights: {where}\n\n'
        'AlphaFold 3 code is copyright Google DeepMind, also Apache 2.0:\n'
        'https://github.com/google-deepmind/alphafold3\n')

  def without(self, knobs):
    """A copy of this spec with the named featurisation conventions removed.

    For ablation: every convention here is silent when wrong, so the only honest
    way to say what one buys is to run without it and measure.
    """
    import copy

    unknown = set(knobs) - set(self.featurise)
    if unknown:
      raise ValueError(f'{self.name} has no featurisation conventions named '
                       f'{sorted(unknown)}; it has {sorted(self.featurise)}')
    clone = copy.copy(self)
    clone.featurise = {k: v for k, v in self.featurise.items()
                       if k not in set(knobs)}
    return clone

  @property
  def sampler(self):
    """EDM constants for this family, or {} to keep AF3's."""
    return dict(_SAMPLER_CONSTANTS.get(self.name, {}))

  def configure(self, config):
    """Apply this model's shape and sampler settings to a Model.Config, in place."""
    config.global_config.model = self.name
    # trained_fourier is not a declared GlobalConfig field (unlike `model`); the
    # dataclass is unfrozen, so set it on the instance -- diffusion_head reads it
    # back defensively with getattr.
    config.global_config.trained_fourier = self.trained_fourier
    if self.full_fat:
      _widen_full_fat(config)
    widen = _WIDENERS.get(self.name)
    if widen is not None:
      widen(config)
    for field, value in self.sampler.items():
      for sample_config in (config.heads.diffusion.eval,):
        if hasattr(sample_config, field):
          setattr(sample_config, field, value)
    return config



class AF2Spec:
  """A model that runs on the vendored AlphaFold 2 network, not the AF3 graph.

  Deliberately NOT a subclass of ModelSpec. Every attribute ModelSpec carries --
  full_fat, trained_fourier, featurise, a converted `.bin.zst` blob -- describes
  the AF3 graph, and inheriting them would give an AF2 model a set of settings
  that look meaningful and are not. What the two share is what a caller actually
  needs: a name, an engine, and enough licence provenance to write output terms.

  AF2's parameters are not converted and not republished. They are DeepMind's
  own `params_model_*.npz` from the AlphaFold 2 release, read straight from
  `--model_dir`, which is why there is no weights_file here.
  """

  __slots__ = ('name', 'model_type', 'weights_licence', 'weights_source')

  engine = 'af2'

  def __init__(self, name, model_type, weights_licence=None,
               weights_source=None):
    from alphafold3.model import model_config

    if name not in model_config.AF2_MODELS:
      raise ValueError(
          f'unknown AF2 model {name!r}; known: {list(model_config.AF2_MODELS)}')
    self.name = name
    # What `alphafold3.af2.runner.AF2Runner(model_type=...)` wants. The runner
    # keeps ColabDesign's names because its own branches are keyed on them.
    self.model_type = model_type
    self.weights_licence = weights_licence
    self.weights_source = weights_source

  def default_model_names(self, use_templates=False):
    from alphafold3.af2 import runner as af2_runner
    return af2_runner.default_model_names(self.model_type, use_templates)

  def output_terms(self):
    return (
        '# OUTPUT TERMS OF USE\n\n'
        f'These structure predictions were generated with AlphaFold 2 '
        f'({self.name}) parameters,\n'
        'run through the AlphaFold 2 network vendored in this package.\n\n'
        'The AlphaFold 3 Output Terms of Use do NOT apply: no AlphaFold 3\n'
        'parameters were used. What applies is the licence the AlphaFold 2\n'
        f'parameters are distributed under -- {self.weights_licence} --\n'
        f'see {self.weights_source}.\n')


AF2_SPECS = {
    'af2_ptm': AF2Spec(
        'af2_ptm', 'alphafold2_ptm',
        weights_licence='CC BY 4.0',
        weights_source='https://github.com/google-deepmind/alphafold'),
    'af2_multimer': AF2Spec(
        'af2_multimer', 'alphafold2_multimer_v3',
        weights_licence='CC BY 4.0',
        weights_source='https://github.com/google-deepmind/alphafold'),
}

MODEL_SPECS = {
    'alphafold3': ModelSpec('alphafold3', trained_fourier=False,
                            weights_repo=None, weights_file='af3.bin.zst'),
    'openfold3': ModelSpec('openfold3', weights_licence='the Apache License, Version 2.0',
                           weights_source='https://github.com/aqlaboratory/openfold'),
    # OpenBind, OpenFold3's v0.5.0 release, which upstream says supersedes the
    # preview-2 weights `openfold3` runs ("the previously released preview 2
    # weights are deprecated and will not function correctly with
    # openfold3>=0.5"). A SEPARATE MODEL rather than a flag on openfold3: it
    # differs in a forward convention, and every other divergence in this file is
    # named by model, so a boolean here would be the one thing you had to
    # remember. The two share a converter, which picks the mapping off the
    # checkpoint (converters/openfold3.py `of3_release`).
    'protenix05': ModelSpec('protenix05',
                            weights_licence='the Apache License, Version 2.0',
                            weights_source='https://github.com/bytedance/Protenix'),
    'protenix1': ModelSpec('protenix1',
                           weights_licence='the Apache License, Version 2.0',
                           weights_source='https://github.com/bytedance/Protenix'),
    'protenix1_20250630': ModelSpec('protenix1_20250630',
                                    weights_licence='the Apache License, Version 2.0',
                                    weights_source='https://github.com/bytedance/Protenix'),
    'protenix_mini': ModelSpec('protenix_mini',
                               weights_licence='the Apache License, Version 2.0',
                               weights_source='https://github.com/bytedance/Protenix'),
    'protenix_tiny': ModelSpec('protenix_tiny',
                               weights_licence='the Apache License, Version 2.0',
                               weights_source='https://github.com/bytedance/Protenix'),
    'openbind0': ModelSpec('openbind0', weights_licence='the Apache License, Version 2.0',
                           weights_source='https://github.com/aqlaboratory/openfold-3'),
    # IntelliFold-v2: stock-AF3 module tree (deliberately NOT in
    # OPENFOLD3_LINEAGE -- its converter emits stock-AF3 haiku names) at the
    # "full_fat" widened channels.
    # trained_fourier has to be named here: IF2 carries one, but it is not in
    # OPENFOLD3_LINEAGE (stock-AF3 module tree), which is what the default reads.
    'intellifold2': ModelSpec('intellifold2', full_fat=True,
                              trained_fourier=True,
                              weights_licence='the Apache License, Version 2.0',
                              weights_source='https://huggingface.co/intelligenAI/intellifold'),
    'opendde': ModelSpec('opendde', weights_licence='the Apache License, Version 2.0',
                         weights_source='https://huggingface.co/aurekaresearch/OpenDDE'),
    'boltz2': ModelSpec('boltz2', weights_licence='the MIT License',
                        weights_source='https://github.com/jwohlwend/boltz'),
    'protenix2': ModelSpec('protenix2', weights_licence='the Apache License, Version 2.0',
                           weights_source='https://github.com/bytedance/Protenix'),
    'rosettafold3': ModelSpec('rosettafold3',
                              weights_licence='the BSD 3-Clause License',
                              weights_source='https://github.com/RosettaCommons/foundry'),
    'chai1': ModelSpec('chai1', weights_licence='the Apache License, Version 2.0',
                       weights_source='https://github.com/chaidiscovery/chai-lab'),
    # ESMFold2 carries a trained Fourier noise embedding like the OF3 lineage,
    # but is not OpenFold-derived, so it is named rather than inherited -- the
    # same reason chai1 is named.
    # MIT, stated on every card in the family (with a third-party notice
    # alongside it). We were emitting "we have not established" while actively
    # redistributing these weights, which understated what the cards say.
    **{m: ModelSpec(m, trained_fourier=True,
                    weights_licence='the MIT License',
                    weights_source='https://huggingface.co/biohub/%s' % hub)
       for m, hub in ESMFOLD2_HUB_IDS.items()},
}

# Historical / abbreviated spellings, kept working.
ALIASES = {
    'af3': 'alphafold3',
    'af2': 'af2_ptm', 'alphafold2': 'af2_ptm', 'af2_monomer': 'af2_ptm',
    'af2_multimer_v3': 'af2_multimer', 'alphafold2_multimer': 'af2_multimer',
    'of3': 'openfold3',
    'if2': 'intellifold2', 'intellifold': 'intellifold2',
    'protenix': 'protenix2',
    'rf3': 'rosettafold3',
    'chai': 'chai1',
    # OpenBind's own release is called "OpenBind 0"
    # (openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction),
    # so the model key carries the version and leaves room for the next one.
    # The unversioned name stays an alias: it is what the published weights file
    # is still called, and what earlier runs and notes refer to.
    'openbind': 'openbind0',
}


def get(name):
  """-> the ModelSpec (AF3 graph) or AF2Spec (AF2 network) for a name or alias.

  Check `spec.engine` before assuming which graph you have: 'af3' for everything
  in MODELS, 'af2' for the AlphaFold 2 models.
  """
  key = ALIASES.get(name, name)
  spec = MODEL_SPECS.get(key) or AF2_SPECS.get(key)
  if spec is None:
    known = sorted(list(MODEL_SPECS) + list(AF2_SPECS))
    raise ValueError(f'unknown model {name!r}; known: {known} '
                     f'(aliases {sorted(ALIASES)})')
  return spec
