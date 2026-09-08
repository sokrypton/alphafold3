# Parity testing: what is gated, at which level, for which model

A port can be wrong in places a fold never reveals. This is the map of what is
actually checked against each vendor's own implementation, organised by LEVEL of
the graph, plus what it would take to fill each gap.

Read `README.md` for the numbers. This file is the plan.

## The levels

| level | what it compares | needs a native? |
|---|---|---|
| **L0** conversion coverage | every checkpoint tensor read, every graph parameter filled — in BOTH directions | no, checkpoint + graph only |
| **L1** trunk pairformer | our single / pair out of the PAIRFORMER STACK against native's, on identical synthetic activations | yes |
| **L1b** MSA module | the MSA stack's contribution to the pair, against native's MSABlock | yes |
| **L2** diffusion conditioning | conditioning z/s, atom encoder, token transformer | yes |
| **L3** denoise step | `r_update` / `x_denoised` for one step, EDM undone | yes |
| **L4** confidence heads | pae / pde / plddt / resolved logits | yes |
| **L5** end-to-end fold | CA-RMSD on a known target | no |
| **L6** modality | ligand, complex, RNA, DNA — behaviour, not activations | no |

L0 and L5–L6 need no vendor code, which is why they cover every model. L1–L4
need the vendor's forward pass, so coverage tracks which natives are installed.

## Current coverage

`✓` gated, `~` partially gated (see the footnote), `·` not measured,
`n/a` no vendor to compare against.

| model | L0 | L1 pairformer | L2 diff-cond | L3 denoise | L4 conf | L5 fold | L6 modality |
|---|---|---|---|---|---|---|---|
| `alphafold3` | n/a | n/a | n/a | n/a | n/a | ✓ | · |
| `openfold3` | ✓ | ✓ | ~ | · | ✓ | ✓ | ✓ |
| `openbind0` | ✓ | ✓ | ~ | · | ✓ | ✓ | · |
| `intellifold2` | ✓ | ✓ | ~ | · | ✓ | ✓ | ✓ |
| `protenix2` | ✓ | ✓ | ~ | · | ✓ | ✓ | ✓ |
| `protenix05` | ✓ | ✓ | ~ | · | ✓ | ✓ | · |
| `protenix1` | ✓ | ✓ | ~ | · | ✓ | ✓ | · |
| `protenix1_20250630` | ✓ | ✓ | ~ | · | ✓ | ✓ | · |
| `protenix_mini` | ✓ | ✓ | ~ | · | ✓ | ✓ | · |
| `protenix_tiny` | ✓ | ✓ | ~ | · | ✓ | ✓ | · |
| `boltz2` | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ |
| `opendde` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rosettafold3` | ✓ | ✓ | ~ | · | ✓ | ✓ | ✓ |
| `chai1` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `esmfold2` family | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | n/a — protein only |
| `af2_ptm` / `af2_multimer` | n/a | n/a | n/a | n/a | n/a | ✓ | n/a — protein only |

L2 is now covered in all three of its parts for protenix: the **token
transformer** (ten models), the **diffusion conditioning** (protenix's six plus
rosettafold3) and the **atom encoder** (protenix2, corr 1.000000 on the
per-atom conditioning, the per-atom output and the per-token output). The atom
DECODER and the other families' atom paths remain unmeasured.

The token-transformer half: ten models run their own vendor's
`DiffusionTransformer` and ours side by side on identical synthetic
`a`/`s`/`z`, all at corr 1.000000 and `rms ours/native` 1.0000
(`dev/oracles/diffusion_parity.py`, table under plan item 2a). The conditioning
projections and the atom encoder/decoder in the same column are still
unmeasured, so this is a third of L2, not L2.

`alphafold3` and the AF2 pair are `n/a` at L0–L4 by construction: the first IS
the reference implementation, and the second runs DeepMind's own network
unmodified. AF2's equivalent gate is a known-answer test against ColabDesign on
the same weights (`dev/oracles/af2_fold_check.py --compare`), agreeing to
0.0006 Å inside a 0.16–0.39 Å autotuning floor.

## What a green L1 actually verifies -- and what it does not

`dev/oracles/trunk_parity.py` runs `layer_stack(PairFormerIteration)` on
synthetic s/z. So a ✓ in the L1 column means the PAIRFORMER STACK matches
native: its triangle multiplications, triangle attentions, attention-pair-bias
and transitions, together with every convention consumed inside them. It does
NOT mean "the trunk is verified", and reading it that way is how the next
convention bug hides.

Checked by grepping each `model_config` table to the class that consumes it:

**Covered by L1** (inside the pairformer block)
  * `TRANSPOSED_COLUMN_PAIR_BIAS` -- `GridSelfAttention`. This is the one L1
    caught wrong for openbind0.

**NOT covered by L1, though they are trunk conventions**
  * `CLAMPED_OPM_NORM` -- `OuterProductMean`, in the MSA module
  * `NO_MSA_ROW_UPDATE` -- `EvoformerIteration`, the MSA stack
  * `MSA_AFTER_RECYCLE`, `SSM_RECYCLE`, `PAIR_ONLY_TRUNK`, `LM_PAIR_DROPOUT` --
    evoformer level, outside the stack
  * the template embedder entirely

**NOT covered by any level, for the models lacking L2-L4.** Nine tables feed the
diffusion path alone -- `PER_BLOCK_PAIR_LAYER_NORM`,
`KEY_MASKED_ATOM_ATTENTION`, `SWA_ROPE_ATOM_ATTENTION`, `REALIGN_SAMPLER`,
`NORMED_ATOM_FEATURES`, `ATOM_ROPE`, `ATOM_ROPE_HALF_WINDOW`,
`DIFFUSION_PROJECTED_RELPOS`, `PER_BLOCK_ATOM_PAIR_LAYER_NORM` -- and six feed
the confidence head. The six confidence tables are now covered for ten models
(L4, plan items 2b/2c), and the token transformer for the same ten (L2, 2a).
**What is left with no activation-level check anywhere is the diffusion
CONDITIONING and the ATOM encoder/decoder** -- nine tables, four model families.
That is now the whole of the named exposure, down from fifteen tables.

**This is the real exposure, and openbind0 showed why it matters.** A membership
decision that changes no weight is invisible to L0 by construction, and folding
could not discriminate the one bug found (the correct setting was marginally
WORSE on 6MRR, and two seeds disagreed). There is no reason to think the
diffusion-path tables are safer than the trunk one was -- only that nothing has
looked.

## Native availability — the actual constraint on L1–L4

| vendor code | present | checkpoints present |
|---|---|---|
| `~/protenix` | yes | **all six**: `protenix-v2`, `protenix_base_default_v0.5.0`, `protenix_base_default_v1.0.0`, `protenix_base_20250630_v1.0.0`, `protenix_mini_default_v0.5.0`, `protenix_tiny_default_v0.5.0` |
| `~/openfold-3` | yes | **both**: `of3-p2-155k.pt` (openfold3), `of3-ob-174k.pt` (openbind0) |
| `~/OpenDDE` | yes | `opendde_weights/opendde.pt` |
| `~/BoltzDesign1/boltz2` | yes | `boltz2_weights/boltz2_conf.ckpt` |
| esmfold2 (HF transformers) | yes | `~/esmfold2_variants` |
| chai-lab | **no** | activation dumps survive in `~/chai_*` |
| `~/IntelliFold` | yes | `~/model_v2/intellifold_v2.pt` |
| RoseTTAFold3 | **yes**, `~/foundry_rf3` | `rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt` |

`~/IntelliFold` and `~/foundry_rf3` import as PYTHONPATH overlays rather
than installed packages; rf3 additionally needs its dependency overlay
`~/rf3_extra` prepended. The rf3 row read **no** until an L2 gate was
actually attempted against it.

## The plan, cheapest first

**1. DONE for protenix (2026-09-07): five models gained an L1 gate.** All six
now read s 1.000000 / z 1.000000 against `~/protenix`. It cost more than the
"two harness edits" predicted below, and for an instructive reason: the
ColabDesign2 harness resolves models through THAT repo's vendored 8-model
registry, so `MODEL=protenix1` died with `unknown weights 'protenix1'` -- 14 of
the 22 models are unreachable from there. The gate had to move into the library,
which is `dev/oracles/trunk_parity.py`. It is also stricter than what it
replaces: it ASSERTS no missing native tensors, no unmapped params, and
`bfloat16 == 'none'` after the spec configures, and it REFUSES to run without
`JAX_DEFAULT_MATMUL_PRECISION=highest` rather than quietly reporting a
tf32-degraded number. Native dims come from the checkpoint, because the family
shares one implementation but not its widths (protenix2 c_z 256, the rest 128;
8, 16 and 48 blocks), and hardcoding protenix2's shape is what left five with no
gate.

`max|d|` grows with DEPTH, not with fidelity -- 0.0087 at 8 blocks, 0.068 at 16,
0.88 at 48, all at corr 1.000000 -- so read it against models of the same depth
or not at all. protenix05 reads 1.741 at 48 blocks where its two same-depth,
same-width siblings read 0.875 and 0.879; at ONE block (`--blocks 1`, which
truncates BOTH stacks) it reads corr 1.000000 / max|d| 0.00211, so that 2x is
compounding of a slightly different activation scale and not a divergent block.

The `--blocks` diagnostic is only meaningful because it truncates the NATIVE
stack too. Omitting that compared our 1 block against native's 48 and read
corr -0.019 with max|d| 15062 -- which looks exactly like a catastrophic port
bug. The tell was that it got WORSE with fewer blocks, which less accumulation
cannot do.

**1b. Six models at L1 for the cost of two harness edits (original estimate).** Every `·` in the L1
column shares a native with a model already gated: the five other protenix
variants use `~/protenix`, and `openbind0` uses `~/openfold-3`. Both harnesses
currently hardcode the checkpoint as a module constant
(`protenix2/cmp_trunk_parity.py:23`, `openfold3/cmp_trunk_parity.py:22`), so the
work is to parameterise by model name and run each. This is the highest
coverage-per-effort item by a wide margin.

**1c. DONE for openbind0 (2026-09-07), and it found a real bug.** L1 read
s 0.999113 / z 0.471977 against `~/openfold-3`, with openfold3 itself at
1.000000 through the same adapter as a control -- so not the harness. The cause
was a transposed column pair bias: v0.5.0 transposes
(`TriangularAttention(transpose_bias=True)` from `base_blocks.py:397`) and
`TRANSPOSED_COLUMN_PAIR_BIAS` omitted openbind0. Fixed; L1 is now
1.000000 / 1.000000. Neither setting changes a weight, so L0 could never see it,
and 6MRR moved 1.637 -> 1.649 -- the CORRECT setting being marginally worse is
why L5 could not see it either. This is the concrete case for L1 existing.

**2a. L2 token transformer GATED for TEN models (2026-09-07).** `dev/oracles/diffusion_parity.py`. protenix's
`DiffusionTransformer` is standalone-constructible exactly like
`PairformerStack`, so an L2 gate follows the L1 pattern. Established, so the
next attempt starts here rather than exploring:

  * weights under `module.diffusion_module.diffusion_transformer.` -- 552
    tensors, `load_state_dict` reports 0 missing / 0 unexpected;
  * dims off the checkpoint: 24 blocks, `c_a` 768, `c_s` 384, `c_z` 256, 16
    heads. Keys: `blocks.0.attention_pair_bias.attention.linear_q.weight`,
    `...layernorm_a.layernorm_s.weight`, `...layernorm_z.weight`,
    `...linear_nobias_z.weight`. PRINT them -- four successive guesses at these
    names were wrong.
  * `DiffusionTransformer(c_a, c_s, c_z, n_blocks, n_heads)` with
    `forward(a, s, z)`, verified on synthetic input to `(1, 68, 768)`.
  * the fused-LayerNorm stub needs MORE entry points than the trunk's did --
    the diffusion path also calls `forward_none_affine`. Dispatch on ARITY
    rather than enumerating. Probe: `dev/bench/l2_native_probe_protenix.py`.

  * dims are per-model (`c_z` 256 on protenix2, 128 on the other five; 24
    blocks except 8 on mini/tiny), so derive them from the checkpoint. Ours are
    mapped by the LAST path segment under
    `diffuser/~/diffusion_head/transformer/`, which sidesteps the nested
    layer_stack (super-blocks inside blocks) that makes the L1 harness's flat
    `[:n_blocks]` slice inapplicable.

Results, identical synthetic `a`/`s`/`z`, `corr` and `rms ours/native` all
1.000000 / 1.0000. Read `max|d|/rms` not `max|d|`: the latter tracks the
reference's own scale (`rms(native)` spans 32 to 1074 across these models) and
the stack depth, not fidelity.

| model | native | blocks | `c_z` | `rms(native)` | `max|d|` | `max|d|/rms` |
|---|---|---|---|---|---|---|
| `protenix2` | `~/protenix` | 24 | 256 | 32.7 | 0.00242 | 7.4e-05 |
| `protenix1` | " | 24 | 128 | 45.8 | 0.00342 | 7.5e-05 |
| `protenix1_20250630` | " | 24 | 128 | 54.2 | 0.00472 | 8.7e-05 |
| `protenix05` | " | 24 | 128 | 24.0 | 0.00025 | 1.0e-05 |
| `protenix_mini` | " | 8 | 128 | 88.4 | 0.00410 | 4.6e-05 |
| `protenix_tiny` | " | 8 | 128 | 103.5 | 0.00371 | 3.6e-05 |
| `openfold3` | `~/openfold-3` | 24 | 128 | 1020.9 | 4.03400 | 4.0e-03 |
| `openbind0` | " | 24 | 128 | 1073.9 | 1.75635 | 1.6e-03 |
| `intellifold2` | `~/IntelliFold` | 24 | 512 | 60.7 | 0.00702 | 1.2e-04 |
| `rosettafold3` | `~/foundry_rf3` | 24 | 128 | 267.1 | 0.00708 | 2.7e-05 |

`max|d|` moves ~20% between processes on identical input (protenix1 read
0.00414 then 0.00342; XLA autotunes by timing, so two processes can run
different kernels). In-process reruns are bit-identical. Do not treat a single
`max|d|` as a threshold; `dev/oracles/l2_all.sh` reruns the whole table.

Conventions this DID settle at activation level, each previously supported only
by a fold number:

  * `PER_BLOCK_PAIR_LAYER_NORM` in both directions. openfold3 preview-2 carries
    24 `blocks.N.attention_pair_bias.layer_norm_z.weight`; openbind0 (v0.5.0)
    carries a single top-level one, having "moved the pair layer norm out of
    attention pair bias ... to match the AlphaFold3 SI". Both now measured.
  * rf3's `no_residual_connection_between_attention_and_transition` (the
    transition reads the PRE-attention act, one shared residual add) and its
    `kq_norm`, at 2.7e-05 -- the tightest number here.

**rf3 is NOT blocked for native comparison**, which PARITY.md previously said:
foundry imports as a PYTHONPATH overlay with its deps in `~/rf3_extra`. Two
rf3-only handles: `force_bfloat16 = True` on every
`AttentionPairBiasDiffusion` must be switched off (otherwise the gate measures
bf16 rounding, ~1e-2 relative), and `Beta_II` must be None or the module takes
its windowed atom-attention branch instead.

This is one of the nine diffusion-path convention tables, across every model
family that had no L2 at all. It does NOT close L2 for these models: the
conditioning projections and the atom encoder/decoder are untouched by it.

**2d. L2 CONDITIONING gated, and it found two more bugs (2026-09-07).**
`dev/oracles/conditioning_parity.py`. protenix's `DiffusionConditioning` and
rf3's are both standalone-constructible, and the relative-position features are
built by the VENDOR's own code from OUR batch's token features -- so a
disagreement in the relative encoding is inside the gate rather than assumed
away. That matters: `DIFFUSION_PROJECTED_RELPOS` had no activation check.

  * **pair conditioning is exact**: protenix2 corr 1.000000, `max|d|/rms`
    3.2e-06, over trunk pair + relative encoding + LN + projection + two
    transitions. That gates `DIFFUSION_PROJECTED_RELPOS` and the
    `pair_cond_initial_norm` offset convention.
  * **single conditioning was NOT**: corr 0.999989, `rms ours/native` 0.9987,
    `max|d|/rms` 0.11.

**BUG 1: the 833-vs-831 LayerNorm, on seven models.** `single_cond_initial_norm`
normalises `[s_trunk(384), s_inputs(449)]` = 833 channels in protenix and rf3,
where our target_feat is 447 wide = 831. Everywhere else those two residue
classes can be dropped, because a zero input contributes nothing to a bias-free
Linear -- but a LayerNorm maps a zero input to `-mean/std`, so the vendor always
contributes them through their trained rows AND normalises over a wider vector.
openfold3 had handled this from the start; protenix and rosettafold3 were left
on the 831 path deliberately. Fixed: `model_config.PADDED_SINGLE_COND` plus
`_reorder_features_1d(pad_unk_dna=True)` in both converters. single_cond
0.999989 -> **1.000000**, `max|d|/rms` 0.11 -> 1.1e-05. 6MRR unmoved (protenix2
0.706 against a 0.702 baseline).

**BUG 2: rf3's diffusion conditioning was remapped with OF3's alphabet.**
`_reorder_features_1d` hardcoded `_AF3_TO_OF3_AATYPE`, and rf3 transposes G/C
and DG/DC -- the same transposition that once folded 1EHZ to 16.8 A against
native's 0.94. Protein indices coincide, so no protein gate could see it. The
helper now takes the permutation and rosettafold3 passes its own.

**Both fixes CHANGE THE BLOBS**: the seven affected models
(six protenix + rosettafold3) must be re-converted, and the published weights
for them are stale until re-published. A stale blob fails loudly -- 831 rows
where the graph wants 833 -- rather than folding quietly.

**2b. L4 GATED for all six protenix models (2026-09-07), and it found a
bug.** `dev/oracles/confidence_parity.py`. protenix's `ConfidenceHead` is
standalone-constructible like its other modules, so the L1/L2 recipe carries
over -- with one difference that will apply to every L4: the head consumes an
ATOM LAYOUT, so it cannot run on purely synthetic input. The trunk activations
stay synthetic and seeded; the layout comes from a real featurised 6MRR batch,
and native's flat-atom indices are DERIVED from that batch
(`atom_to_token_idx` = repeated token index, `atom_to_tokatom_idx` = tiled slot
index, `distogram_rep_atom_mask` = the slots our pseudo-beta gather picks). The
harness asserts the two sides' rep-atom coordinates are bit-identical before it
compares anything; without that check a geometry mismatch would read as a
plausible correlation.

| model | `full_pae` | `full_pde` | `plddt` | `resolved` |
|---|---|---|---|---|
| `protenix2` | 0.999985 | 0.999989 | 1.000000 | 0.999999 |
| `protenix1` | 0.999999 | 1.000000 | 1.000000 | 1.000000 |
| `protenix1_20250630` | 0.999998 | 0.999996 | 0.999999 | 0.999998 |
| `protenix05` | 0.999999 | 1.000000 | 1.000000 | 1.000000 |
| `protenix_mini` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `protenix_tiny` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

**2c. L4 for the other five (same day).** openfold3, openbind0, intellifold2,
rosettafold3, opendde -- so with the six protenix models and the three gated by
injection (boltz2, chai1, esmfold2), **every port now has an L4 gate**.

`opendde` is the odd one: its confidence runs on the STRUCTURAL token set at
c_s = c_z = 384 through its own module (`network/opendde_confidence.py`), and
both sides take the atom layout as explicit arguments -- so that gate needs no
featurised batch at all, it synthesises the layout `model.py` builds (token
index repeated, slot index tiled, rep atom at slot 0) and hands the same one to
both. It reads pae 1.000000 / pde 0.999999 / plddt 1.000000 / resolved
1.000000.

| model | `full_pae` | `full_pde` | `plddt` | `resolved` |
|---|---|---|---|---|
| `openfold3` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `openbind0` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `intellifold2` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `rosettafold3` | 0.999999 | 0.999999 | 1.000000 | 1.000000 |

Getting there cost two harness faults that each looked exactly like a port bug,
and both are now asserted or documented in the harness:

  * **intellifold2 read plddt corr 0.614.** Its converter writes trunk-region
    weights as **bfloat16 on disk** deliberately
    (`converters/intellifold2.py` `_record_dtype`, "mirrors AF3's param dtype
    policy, which the published blob follows") while its `.pt` checkpoint is
    fp32 -- so no config flag can lift our side to fp32. Rounding NATIVE the
    same way (LayerNorms and the four logit heads excepted, which the blob keeps
    fp32) gives 1.000000 on all four. `intellifold2` is the ONLY port that ships
    bf16 weights; of3 and protenix blobs are all-fp32. **It costs nothing**:
    the same checkpoint converted at fp32 folds 6MRR to best 1.519 / mean 1.611
    A against 1.517 / 1.611 for the bf16 blob, at 3010 MB instead of 1683 --
    `global_config.bfloat16` is 'all' by default, so the trunk casts to bf16 at
    inference whatever the storage says. It does mean a run with
    `bfloat16='none'` (a gradient or design run) still gets bf16-rounded trunk
    weights out of this blob.
  * **rosettafold3 read pae/pde 0.983.** The harness built our 447-wide
    target_feat from native's 449 using OF3's alphabet, but rf3 transposes G/C
    and DG/DC and its converter correctly uses its own
    (`_AF3_TO_RF3_AATYPE`). Discriminator: zero every column a protein input
    never populates and watch it vanish -- the disagreement lived exactly in the
    classes the two alphabets order differently.

**One real fidelity gain, in the graph.** rf3's confidence head applies a
parameter-free LayerNorm over the WHOLE TENSOR, so the statistic depends on the
FEATURE WIDTH -- and native's s_inputs is 449 wide where our target_feat is 447.
Passing `width=449` to `masked_global_norm` (which now accounts for the two
dropped, always-zero columns) halves the residual: pae `max|d|` 0.047 -> 0.022,
corr 0.999998 -> 0.999999. `converters/opendde.py` flags the same 831-vs-833
mismatch on `single_cond_initial_norm` as "minor; confirm via e2e"; this is that
confirmation. rf3's 6MRR is unchanged (best 0.974 A).

Also fixed: `_truncate` now applies to BOTH sides. With only ours truncated the
1-block run read corr -0.19 and looked catastrophic -- the same mistake, and the
same tell (it got WORSE with fewer blocks), as the L1 harness's `--blocks`.

**THE BUG: protenix's PDE head symmetrises the PAIR, not the LOGITS.** Native
computes `Linear(pde_ln(z + z^T))`; AlphaFold 3 computes
`l = Linear(LN(z)); pde = l + l^T`. LayerNorm is not linear, so those are
different functions of z, and unlike the distogram-bias case it cannot be folded
into the weight -- it needs a forward branch
(`model_config.PRE_SYMMETRISED_PDE`). `full_pde` went **0.869929 -> 0.999989**
on protenix2; six models fixed, no weights changed, no reconversion. Every other
vendor really does symmetrise the logits (of3 `prediction_heads.py`, if2
`pDEHead._forward`, rf3's af3-style branch -- which its released config does
select, `use_af3_style_binning_and_final_layer_norms: True`); boltz2 and opendde
symmetrise first and already had their own branches.

**Why no fold caught it**: pde is reported, never fed back into the structure, so
L5 is blind to it by construction -- and a symmetric, plausibly-scaled error
metric stays symmetric and plausible. Same shape of argument as openbind0's
transposed bias, and the second time in one session that a level below L5 found
something no fold could.

**What the numbers do NOT gate**: the harness applies OUR bin centers to both
sides, so a bin-convention error cancels. Those were checked by reading both
definitions: protenix pae/pde are 64 bins over [0, 32], centers
`linspace(0, 32-w, 64) + w/2` = 0.25 ... 31.75, which is exactly what our
`max_error_bin = 31.0` plus the catch-all bin produces; plddt is 50 bins over
[0, 1] on both sides. Re-check this per vendor -- confidence bugs here have been
bin and layout bugs, not weight bugs.

**Depth, not fidelity, drives `max|d|` here too.** Truncating the confidence
pairformer to one block (`BLOCKS=1`) reads corr 1.000000 with p99.9 |d| of
2e-03 A; at two blocks 1.3e-01; at the full four 3.4e-01. corr stays >= 0.99998
throughout and `rms ours/native` stays 1.0000. These outputs are expectations
over softmaxed bins, so a 1e-6 logit difference is worth milli-angstroms.

**2. L3, and L2's other two thirds** — for `openfold3`, `intellifold2`,
`protenix2` and `rosettafold3`. Token transformers are gated (2a) and so are the
confidence heads (2b/2c); what remains is the diffusion CONDITIONING, the atom
encoder/decoder, and then the denoise step. `opendde` is the one model with no
L4 -- it has its own head. These
need new oracles, and they are the four whose ports predate the injection-ladder
method (dump native's own tensors, inject them, compare our module's output).
The recipe to copy is `esmfold2`'s, which is the most completely gated model
here.

**3. L6 for the protenix variants and `openbind0`** — no vendor code needed,
just runs of the existing modality screens.

**4. L0 for everything, continuously.** Cheapest level and the one that catches
silent drops: it found the missing distogram bias in four models. Both
directions matter — native tensors we never read, AND graph parameters we never
fill.

## L6 modality, in this repository at last (2026-09-07)

`dev/oracles/modality_check.py`. Until now the RNA / DNA / ligand / complex
screens lived in another repository, so this one could fold those inputs but
could not SCORE them. Numbers below are best-of-5 samples, seed 0, on the same
featurisation a real run uses.

Each modality is scored on the right atom, which is the part that is easy to get
silently wrong: protein CA, nucleic C1', a ligand on its own heavy atoms after
superposing on the protein, and a complex in ONE shared frame so that a
correct-but-misplaced chain fails.

| model | 1EHZ tRNA (C1') | 1STP protein (CA) | BTN ligand (in-frame) |
|---|---|---|---|
| `rosettafold3` | **1.047** | 0.322 | 0.450 |
| `boltz2` | 1.196 | **0.276** | 0.457 |
| `opendde` | 1.327 | 0.298 | 0.876 |
| `openfold3` | 1.334 | 0.494 | — |
| `protenix05` | 1.409 | 0.307 | **0.426** |
| `alphafold3` | 1.412 | 0.564 | — |
| `intellifold2` | 1.472 | 0.316 | 0.436 |
| `openbind0` | 1.496 | 0.499 | — |
| `protenix1` | 1.737 | 1.867 | 0.918 |
| `protenix2` | 1.754 | 2.090 | 1.253 |
| `protenix_tiny` | 1.774 | 0.387 | 0.871 |
| `protenix_mini` | 2.129 | 0.339 | 0.439 |

Twelve models, where seven had a modality number before and five had only a
6MRR fold. The three em-dashes ran before the ligand-pairing fix below.

**The protenix v1/v2 line is the weak row on 1STP** (protein 1.9-2.1 Å where
everything else is 0.28-0.56), and it is NOT something this session's changes
caused: the recorded number for protenix2 before today was 2.497 / 1.147 on the
same target, and native protenix is worse than our port there. Its own 0.5.0
lineage -- protenix05, mini, tiny -- is among the best on it, which is what
makes this a property of those two checkpoints rather than of the port.

**Pairing a predicted ligand to its reference is model-dependent.** Most models
featurise a CCD ligand with its real atom names (`C11`, `O11`, ...), which pair
directly; `rosettafold3` rewrites them to the ELEMENT (`C`, `O`), leaving only
ORDER to match on. The harness tries names, falls back to order, and in the
fallback requires the two element sequences to agree atom for atom -- which is
what makes pairing by order a check rather than an assumption.

**Modified residues have to resolve to their parent.** 1EHZ is tRNA-Phe: 14 of
its 76 residues are modified (PSU, 2MG, H2U, 1MA, 7MG, ...). Skipping them --
what a plain A/G/C/U table does -- yields a 62-residue "sequence" whose indices
no longer line up with the coordinates, i.e. a screen that folds the wrong
molecule and scores it against a shifted reference. The parent comes from the
CCD's own `mon_nstd_parent_comp_id`.

First DNA and PTM numbers (best of 5, seed 0):

| model | 1LMB duplex (C1', both strands in one frame) | phospho-ubiquitin 5K9P (CA) |
|---|---|---|
| `openfold3` | **1.914** | **1.502** |
| `opendde` | 1.992 | 1.812 |
| `alphafold3` | 2.025 | — |
| `protenix2` | 2.079 | — |
| `intellifold2` | — | 1.556 |
| `rosettafold3` | — | 1.801 |
| `chai1` | — | 1.934 |

The PTM case also proves the WRITER: `SEP` comes back in the emitted mmCIF with
all ten of its atoms, so the modified residue is not quietly written out as
serine.

**Two scoring traps this case exposed, both in the harness.** A modified residue
is ATOMISED -- one token per atom -- so token index stops equalling residue
index after it, and matching by position scored ubiquitin at 12.4 A on every
model (identical to three digits across models, which is the tell). Scoring now
matches on the batch's own `residue_index`. And for the protein-DNA COMPLEX,
1LMB's two protein chains are one protein: a model that builds a perfect dimer
with the copies swapped scored 17 A while every chain read ~1 A (rosettafold3
does exactly this -- protein 1.141/1.361, DNA 0.838/0.928, joint 17.084). The
joint score now tries the assignments identical sequences allow, after trimming
the copies to their common residues -- but only when that trim leaves something,
because the two DNA strands are complementary with disjoint numbering and the
first version silently trimmed the DNA out of the complex entirely.

**The protein-DNA complex case measures single-sequence FOLDING more than
assembly, and must be read that way.** Scored symmetry-aware (1LMB's two protein
chains are one protein, so a perfect dimer with the copies swapped otherwise
reads ~17 A):

| model | protein chains (CA) | DNA strands (C1') | whole complex |
|---|---|---|---|
| `boltz2` | 0.390 / 0.235 | 0.272 / 0.278 | **0.424** |
| `rosettafold3` | 1.141 / 1.361 | 0.838 / 0.928 | 1.293 |
| `openfold3` | 11.347 / 11.308 | 1.060 / 1.172 | 12.721 |
| `protenix2` | 11.763 / 11.669 | 1.819 / 1.801 | 17.454 |
| `opendde` | ~11.3 | 1.265 / 1.353 | 17.880 |

The split is exact: every model that folds the 87-residue lambda-repressor
domain from a single sequence docks the complex, and every model that does not,
fails it. The harness gives those chains a SELF-MSA only. So protenix2's 17 A is
its protein chains at 11.7 A -- its DNA in the same run is 1.8 A -- and not a
docking defect, still less a port defect: protenix2's whole diffusion module
reproduces native to 0.0000 A per atom at L3. Give this case a real MSA before
reading it as a docking number.

**DNA is where this screen paid for itself immediately.** The 1LMB duplex is
the first DNA fold this repository could run in-tree, and it died on every
model except stock `alphafold3` with `TypeError: 'method' object is not
iterable`. `DnaChain.modifications` is missing its `@property` decorator in
upstream AlphaFold 3's `folding_input.py` -- `ProteinChain.ptms` and
`RnaChain.modifications` both have it -- so it returns a bound METHOD, and our
`_modified_residue_positions` iterates it. Every model with a `featurise` spec
(every port) went through that path; stock alphafold3 skips `model_features`
entirely, which is exactly why it was the one model that worked and why nothing
had caught this. Two more call sites, `data/pipeline.py` and
`data/msa_server.py`, pass the method through as a value.

## The featurisation diff, and what it found (2026-09-08)

Every gate above compares MODULES on identical inputs, or whole FOLDS. Nothing
compared the INPUTS -- and for protenix2 every module is exact while the fold
disagrees with native, so the inputs were the only place left. Native protenix's
own featuriser output for the same JSON, dumped through its dataloader on the
GPU venv and diffed field by field against our batch (5K9P + SEP-20, 85 tokens,
606 atoms on both sides):

| field | result |
|---|---|
| token count, atom count | identical (85 / 606) |
| `restype`, `residue_index`, `ref_charge`, `ref_space_uid`, `ref_element`, MSA row 0 | **identical** |
| `asym_id`, `entity_id`, `sym_id`, `token_index` | off by exactly 1 -- ours 1-based, protenix 0-based |
| **`ref_pos`** | **every value differs, max 8.57 A** |

**The off-by-one is benign, and worth knowing rather than fixing.** Those four
features reach the model only through equality (`asym_i == asym_j`) and through
differences (`token_index_i - token_index_j`), and a constant offset cancels in
both.

**The reference conformers differ by their TORSIONS.** The correspondence is
proven, not assumed: `ref_atom_name_chars` decodes to the same atom names in the
same order on both sides (as do `ref_element`, `ref_charge` and `ref_space_uid`),
so atom k on our side is atom k on theirs. Aligning per residue then leaves:

| aligned set | ours vs native | ours vs CCD ideal | native vs CCD ideal |
|---|---|---|---|
| whole residue | **0.90 A** (max 1.76, n=76) | 0.90 | 0.79 |
| backbone only (N,CA,C,O,CB) | 0.31 | 0.34 | 0.26 |
| side chain only | 0.17 | 0.14 | 0.15 |

Each rigid fragment agrees to ~0.2-0.3 A; only their RELATIVE ORIENTATION does
not. That is a chi-torsion difference -- two different conformers of the same
molecule, which is also why NEITHER side reproduces the CCD ideal coordinates
(0.90 and 0.79): both are generated, not read off the ideal columns. So neither
is wrong; they are not the same draw. `ref_pos` feeds the atom encoder's per-atom
features and its windowed atom-pair distances, so this is a real input difference
in every protenix fold -- of the same class as the boltz2 atom-encoder conformer
gap, and equally not a port bug.

It does NOT obviously explain protenix2, and saying so is the point: protenix05
carries the same conformer difference through the same code and matches native
end to end (ours 1.569/1.465, native 1.552/1.332). So the conformer difference is
real, is worth closing, and is not on its own the cause.

**OPEN: protenix2 does not reproduce native on a MODIFIED residue.** The one
place in this whole session where our port and native end-to-end disagree.
Phospho-ubiquitin (5K9P, SEP-20), native seed 101, native settings (10 recycles
x 200 sampling steps), no MSA on either side:

| | ours | native protenix2 |
|---|---|---|
| 5K9P plain | 7.161 | 9.648 |
| 5K9P + SEP-20 | 7.584 | **1.080** |

Natively the modification transforms the fold (9.6 -> 1.1 A, all five samples);
for us it changes nothing (7.16 -> 7.58). On the PLAIN target we are better than
native, so this is specific to the modified-residue path. What has been excluded:

  * SEEDS -- ours reads 7.2-8.3 across seeds 1, 7 and 101, native's own seed 101
    gives 1.08. Not sampling noise.
  * THE MSA -- ours is 7.99 with a self-MSA and 7.99 with none.
  * TOKENISATION -- both sides atomise a modified residue (protenix's
    `add_centre_atom_mask` cites the same AF3 SI rule), 76 tokens -> 85.
  * RESTYPE -- both give the atomised tokens the PARENT type: protenix maps
    `SEP -> S -> SER` in `add_cano_seq_resname`, which is what we do.
  * The junction BONDS an atomised residue loses. `atomized_backbone_bonds` is
    rf3-only in our registry and protenix builds its bonds from the atom array,
    so it looked like the answer; enabling it moves nothing (7.584 -> 7.225,
    inside noise). Reverted rather than kept on a hunch.

RULED OUT SINCE, by the featurisation diff above: the token and atom counts,
restype, residue_index, ref_charge, ref_space_uid, ref_element and the MSA row
are all identical to native's own featuriser. Also ruled out: the template path
(native's featuriser emits `template_aatype` with an ALL-EMPTY mask and runs its
embedder on it exactly as we do -- forcing our contribution to zero changes
nothing, 6MRR 0.700 vs 0.699), and protenix2's own modules, every one of which
is now measured exact including the MSA stack at 1.00000000.

What is left: the reference conformers (0.90 A per residue), and the recycling /
sampler integration, which no gate covers -- L3 compares ONE denoise step at a
fixed noise level with given conditioning. Every protenix2 MODULE is exact against native (L1, L2
conditioning including 4-chain, L2 token transformer, L2 atom encoder, L3 a full
denoise step at 0.0000 A, L4), so whatever this is, it is an input difference.

**Modified residues are their own case.** `ptm_5k9p` is ubiquitin
phosphorylated at Ser20 (SEP): AF3 ATOMISES a modified residue, so it reaches a
path no plain-protein fold does, and `--write` then checks that SEP survives
into the emitted mmCIF rather than being quietly written back as serine.

**`--write` validates the OUTPUT too.** It writes the model's own mmCIF and
re-reads it: chains present, component types, finite coordinates, pLDDT within
[0, 100], and for a ligand case that the ligand actually survived into the file.
A model can place a ligand correctly and still emit a file that drops it, and no
RMSD above would notice.

## L0, run across the protenix family for the first time (2026-09-07)

`dev/audit_coverage.py` only knew `protenix2`; the other five raised KeyError,
so L0 had never run on them. With the family added to `_LOADERS`/`_MAPPERS`:

| model | checkpoint tensors | unaccounted | what they are |
|---|---|---|---|
| `protenix2` | 4174 | 2 | `confidence_head.lower/upper_bins` — distance-bin EDGES, which our graph computes from config |
| `protenix1` | 4174 | 2 | the same two |
| `protenix05` | 4092 | 7 | the bins + **5 template-embedder tensors** |
| `protenix_mini` | 1612 | 9 | the bins + template embedder + `layernorm_v.bias` |
| `protenix_tiny` | 1157 | 10 | the above + **`input_embedder.linear_esm`** (449, 2560) |
| `rosettafold3` | 4075 | 33 | every `attention_pair_bias.ln_0.bias` |

Three findings, in descending order of consequence:

**The 05/mini/tiny template tensors are VESTIGIAL, and dropping them is
correct.** Their checkpoints carry the five loose template tensors
(`layernorm_z`, `linear_no_bias_z`, `linear_no_bias_a` over 108 template
features, `linear_no_bias_u`, `layernorm_v`) and **zero** pairformer blocks
under `template_embedder.pairformer_stack` — protenix's own
`TemplateEmbedder.forward` returns early on `n_blocks < 1` ("Compatible with the
Protenix 0.5.0 model series"), so the vendor disabled templates in that lineage
and the loose weights are never used. protenix2 by contrast carries 89 template
tensors including a real stack, and all of them ARE read, which is why its count
is 2. So this row is a clean bill, not a gap — but it is the L0 audit that
distinguishes "unported" from "unused upstream", and only after opening the
checkpoint.

**protenix_tiny carries an ESM input projection we never feed.**
`input_embedder.linear_esm` is (449, 2560) — a 2560-wide language-model
embedding, i.e. ESM2-3B. The model folds without it (1.483 A on 6MRR), so it is
optional rather than required, but tiny is the distillation most likely to lean
on it.

**rosettafold3's 33 are provably inert, and the L2 gate is the proof.** They are
the OFFSET of the pair-bias LayerNorm in every diffusion and atom block, and
their values are not small (up to 10.9). They drop out anyway: `ln_0` feeds a
BIAS-FREE projection to per-head attention logits, so the offset contributes
`W · b`, a constant per head, identical for every (i, j) — and a constant added
to every logit of a softmax cancels. The token-transformer gate confirms it
empirically at corr 1.000000 against a native module that HAS those biases.

## L2's atom half, and a third harness fault of the same shape (2026-09-07)

`dev/oracles/atom_parity.py` runs our atom cross-attention encoder against
protenix's `AtomAttentionEncoder` on a REAL featurised batch -- the windows and
the atom ordering cannot be synthesised. Final numbers on protenix2:

| tensor | what it is | corr | max\|d\|/rms |
|---|---|---|---|
| `c_atom_cond` | the per-atom conditioning, before any attention | 1.000000 | 8.1e-07 |
| `q_atom` | the per-atom output of the 3-block atom stack | 1.000000 | 7.1e-05 |
| `a_token` | pooled to tokens, what the token transformer eats | 1.000000 | 1.3e-05 |

It did not start there. The first run read `a_token` 0.968, and the localisation
ladder -- zero the trunk conditioning, zero the noisy coordinates, then keep ONE
reference feature at a time -- put it precisely:

| feature kept | `c_atom_cond` |
|---|---|
| positions | 1.000000 |
| charge | 1.000000 |
| atom-name chars | 1.000000 |
| **element** | **0.832** |

**And it was the harness, not the port.** protenix featurises an element as
`GetAtomicNum() - 1`; our batch stores AF3's 1-indexed `GetAtomicNum()`, and
`converters/common.py::fold_element_index_shift` folds that -1 into the
embedding ROWS rather than shifting the input. So native has to be fed the
shifted index, and feeding it ours moved every atom's element embedding by one
row. That is the THIRD fault of this shape today, after intellifold2's bf16
storage and rosettafold3's alphabet: each time, the port compensates for a
vendor convention somewhere the harness did not know about, and the harness
looks like the bug.

`p_atom_pair` stays at 0.94 while the outputs it feeds are exact to 1e-5. Our
windows clamp an out-of-range key onto atom 0 -- a real atom, repeated -- where
protenix pads with zeros, and both sides mask those cells out of the attention.
The outputs agreeing to 1e-5 IS the evidence that those cells never reach the
result.

## L1b, the trunk's other half (2026-09-07)

`dev/oracles/prot_parity.py` gates the MSA MODULE as well as the trunk, which is
what closes `CLAMPED_OPM_NORM` and `NO_MSA_ROW_UPDATE` -- two of the four trunk
conventions L1 explicitly does NOT cover. All three protenix releases that carry
an MSA stack agree with native:

| model | pairformer blocks | MSA blocks | single | pair | msa -> pair |
|---|---|---|---|---|---|
| `protenix05` | 48 | 4 | 1.00000000 (4.3e-07) | 1.00000000 (9.3e-05) | 1.00000000 (2.6e-05) |
| `protenix_mini` | 16 | 1 | 1.00000000 (5.0e-06) | 1.00000000 (1.8e-05) | 1.00000000 (6.2e-07) |
| `protenix_tiny` | 8 | 1 | 1.00000000 (7.7e-06) | 1.00000000 (1.5e-05) | 1.00000000 (8.0e-07) |

(corr, with max relative error in brackets.) It needed the same
dims-from-constants fix as everything else here: `PairformerStack` builds
`c_hidden` 128 whatever `c_z` is, so protenix2 died in `load_state_dict` until
the harness passed `hidden_scale_up=True` -- which trunk_parity.py had been
passing all along.

## The gates, and where they live

`dev/` is gitignored (see README, "Where the harnesses live"), so these exist on
the machine that runs them and NOT in the repository. Each is listed with what
it covers, because the file itself is the only other record:

| gate | level | covers |
|---|---|---|
| `dev/oracles/trunk_parity.py` | L1 | pairformer stack vs the vendor's module — 7 models |
| `dev/oracles/prot_parity.py` | L1b | protenix mini/tiny/05 trunk AND MSA module |
| `dev/oracles/conditioning_parity.py` | L2 | diffusion pair + single conditioning — 6 protenix, rf3 |
| `dev/oracles/atom_parity.py` | L2 | atom cross-attention encoder, real batch, windowed |
| `dev/oracles/diffusion_parity.py`, `l2_all.sh` | L2 | token diffusion transformer — 10 models |
| `dev/oracles/confidence_parity.py`, `l4_all.sh` | L4 | confidence head — every port |
| `dev/oracles/fold_check.py` | L5 | one model, one target, CA-RMSD (`MODEL_DIR=` to compare blobs) |
| `dev/oracles/modality_check.py` | L6 | RNA / DNA / ligand / complex folds scored against a reference, and `--write` validates the mmCIF the model emits |
| `dev/oracles/grad_check.py` | — | sequence-differentiability, either engine |
| `dev/oracles/af2_fold_check.py` | — | AF2 against ColabDesign on the same weights |

## Two harness confounds, non-negotiable at L1–L4

Without both switched off, an L1–L4 number is meaningless. These cost six false
leads on `protenix2` alone, whose trunk read 0.9929/0.9376 with a port that
turned out to be exact.

```
FP32=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  PYTHONPATH=/path/to/protenix:/path/to/ColabDesign2 \
  python tools/oracles/protenix2/cmp_trunk_parity.py
```

`FP32=1` reaches `models.build(..., fp32=True)`; parameters otherwise come back
bfloat16-rounded, and flipping `global_config.bfloat16` after `build()` does
nothing because `jax.eval_shape` already fixed the dtype.
`JAX_DEFAULT_MATMUL_PRECISION=highest` disables tf32 (~5e-4 per matmul,
compounded over 48 blocks).

## Method notes that make these numbers trustworthy

  * **Validate the tap before comparing anything.** Recompute native's own
    output from native's own captured tensors and gate on THAT first. It is what
    separated weights (6e-7) from convention (3e-6) from our module (2.5e-3) on
    protenix2, pinning the residual on tf32.
  * **Compare the UPDATE, not the output**, when localising a trunk gap. An
    ESMFold2 block's output read corr 0.971, the worst of 24, while its own
    update read 0.9985, the best.
  * **Run ONE block on native's own input.** That is what settles whether a
    divergence is the block or its input.
  * **Correlation hides a constant.** A dropped bias reads ~1.0 by rank
    correlation. Check the standard deviation and the mean separately.
  * **A duplex cannot detect an alphabet transposition** — G↔C in both strands
    stays Watson–Crick. Check the nucleotide alphabet statically instead.
