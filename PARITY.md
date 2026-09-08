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

## Regression check after 2026-09-08's changes

Two forward-graph fixes landed today (both in the protenix template path) plus
one model removal, so every model was re-folded on 6MRR afterwards. All nine
sit on their recorded baselines:

| model | after | baseline |
|---|---|---|
| `protenix2` | 0.705 | 0.702 |
| `protenix1` | 1.695 | 1.694 |
| `openfold3` | 1.545 | 1.541 |
| `openbind0` | 1.648 | 1.649 |
| `intellifold2` | 1.517 | 1.514 |
| `boltz2` | 0.431 | 0.434 |
| `opendde` | 0.771 | 0.767 |
| `rosettafold3` | 0.977 | 0.986 |
| `chai1` | 1.718 | 1.719 |

The templated path is where the fixes bite, and there the change is large and
intended: protenix2 on a templated 5K9P goes 1.588 -> 0.227 A.

## Four protenix models were removed (2026-09-08)

`protenix05`, `protenix1_20250630`, `protenix_mini` and `protenix_tiny` are gone;
`protenix2` and `protenix1` remain. They were removed because they differed from
the two survivors only by training run or by size, so debugging them spread
parity effort across variants without adding architectural coverage --
`protenix2` is the most completely gated model in this document, and it is the
one with an open question worth the attention (the modified-residue divergence
below).

**The measurements below were not rewritten.** Sections dated before this say
things like "all six protenix models" and quote numbers for the removed
variants; those runs happened and the findings they produced -- the PDE
symmetrisation, the 833-vs-831 LayerNorm, the atom-block-count derivation --
are why the surviving models are gated as tightly as they are. Only the coverage
tables and the current-state claims were edited. Nothing about the removal
invalidates a number recorded here.

Re-adding a variant is still close to a one-liner: a `converters/sources.py`
entry, a converter alias, a `MODELS` line and (if its widths differ) a widener.
`PROTENIX_FAMILY` and every list keyed off it were left written as a family.

Verified after the removal, because `PROTENIX_FAMILY` feeds six shared
convention lists and a bad edit there would move models that were never touched:

| model | 6MRR best, after | recorded baseline |
|---|---|---|
| `protenix2` | 0.703 | 0.702 |
| `protenix1` | 1.701 | 1.694 |
| `openfold3` | 1.544 | 1.541 |
| `boltz2` | 0.430 | 0.434 |
| `rosettafold3` | 0.956 | 0.986 |

The published weights for the four removed models are still live at
`sokrypton/af3-any-model/protenix/` and were deliberately NOT unpublished:
someone may have pinned them, and deleting a published file is not a decision
the removal itself implies.

## Current coverage

`✓` gated, `~` partially gated (see the footnote), `·` not measured,
`n/a` no vendor to compare against.

| model | L0 | L1 pairformer | L2 diff-cond | L3 denoise | L4 conf | L5 fold | L6 modality |
|---|---|---|---|---|---|---|---|
| `alphafold3` | n/a | n/a | n/a | n/a | n/a | ✓ | · |
| `openfold3` | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ |
| `openbind0` | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ |
| `intellifold2` | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ |
| `protenix2` | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ |
| `protenix1` | ✓ | ✓ | ~ | ~ | ✓ | ✓ | ✓ |
| `boltz2` | ✓ | ✓ | ✓ | ~ | ✓ | ✓ | ✓ |
| `opendde` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rosettafold3` | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ |
| `chai1` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `esmfold2` family | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | n/a — protein only |
| `af2_ptm` / `af2_multimer` | n/a | n/a | n/a | n/a | n/a | ✓ | n/a — protein only |

**What `~` at L2 means, model by model.** L2 has three parts, and the column is
`~` wherever fewer than all three are measured:

| part | gate | covered |
|---|---|---|
| token diffusion transformer | `diffusion_parity.py` | ten models, all corr 1.000000, `rms ours/native` 1.0000 |
| diffusion conditioning | `conditioning_parity.py` | `protenix2`, `protenix1`, `rosettafold3` |
| atom cross-attention ENCODER | `atom_parity.py` | `protenix2`, `protenix1`, both of3 releases, `intellifold2`, `rosettafold3` |

So `boltz2`, `opendde`, `chai1` and the `esmfold2` family read `✓` because their
L2 was gated a different way when they were ported (whole-module injection),
while the AF3-family rows read `~` because this document's three-part L2 is the
stricter standard and their conditioning or atom encoder is not all measured.

**The atom DECODER has no gate of its own, on any model.** It is covered only in
composition, by the models whose whole denoise step (L3) is exact -- `protenix2`
at 0.0000 A per atom, `openfold3` at 0.0001, `openbind0` at 0.0021. That is real
evidence, but it cannot localise a decoder-only fault, and it is the largest
remaining structural hole here.

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
| `~/protenix` | yes | `protenix-v2`, `protenix_base_default_v1.0.0` (the four other protenix model types were removed on 2026-09-08; their checkpoints are still on disk and in `sources.py` history) |
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
the models are unreachable from there. The gate had to move into the library,
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
encoder/decoder, and then the denoise step. DONE since: the atom encoder for
both of3 releases, `protenix2` and now `intellifold2`; L3 for all six protenix
releases; the atom encoder for `rosettafold3`; **L3 for every remaining port**
(of3, openbind0, intellifold2, rosettafold3). LEFT: the atom DECODER has no gate
of its own -- it is covered only in composition, by the three models whose
denoise step is exact -- and the window-edge residual that if2 and rf3 both show
is the padded-key mask item. `opendde` is the one model with no
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
| `alphafold3` | 1.412 | 0.564 | — |
| `intellifold2` | 1.472 | 0.316 | 0.436 |
| `openbind0` | 1.496 | 0.499 | — |
| `protenix1` | 1.737 | 1.867 | 0.918 |
| `protenix2` | 1.754 | 2.090 | 1.253 |

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

It does NOT explain protenix2's 5K9P behaviour (see the retraction below), and
saying so is the point: the since-removed protenix05
carries the same conformer difference through the same code and matches native
end to end (ours 1.569/1.465, native 1.552/1.332). So the conformer difference is
real, is worth closing, and is not on its own the cause.

**RETRACTED, and replaced by what the evidence actually shows (2026-09-08).**
This section previously read "protenix2 does not reproduce native on a MODIFIED
residue", built on a single comparison: ours 7.584 A against native's 1.080 on
5K9P+SEP-20 at seed 101. Both halves of that framing were wrong.

**1. The plain target fails too, so it was never the modification.** Re-run
side by side at seed 101, no MSA, native's own settings:

| target | ours best / mean | native best / mean |
|---|---|---|
| 5K9P plain | 7.489 / 9.581 | 9.648 / 11.368 |
| 5K9P + SEP-20 | 7.978 / 9.651 | 1.080 / 1.767 |

Our two rows are the same number. Nothing about the modified-residue path is
implicated by them.

**2. Native's 1.080 was one lucky seed.** Native protenix2 on this target is
wildly seed-dependent -- and seed 303 INVERTS the story, folding the plain
target well (1.57 A mean) and the modified one badly (10.2):

| native seed | plain mean | +SEP mean |
|---|---|---|
| 101 | 11.37 | 1.77 |
| 202 | 4.92 | 1.87 |
| 303 | **1.57** | **10.21** |
| 404 | 6.40 | 1.25 |
| all 20 samples each | 6.06 (7/20 under 3 A) | 3.77 (15/20 under 3 A) |

Native does gain something real from the modification (15/20 under 3 A against
7/20), but a fifth of that table's spread is bigger than the effect, and the
single-seed pair this section was built on was mostly noise. **One seed is not a
measurement on a target a model folds unreliably.**

**3. What survives is a distributional gap, not a defect anyone has localised.**
Across 40 samples (4 seeds x 5 samples x 2 targets) native lands under 3 A
twenty-two times; across our own 40 we never go below 7.3 A. Native reaches the
right basin on this target and we do not. That is real and unexplained.

**The first REAL-INPUT trunk comparison (2026-09-08).** Every gate above feeds
the trunk RANDOM s/z, so nothing had ever compared the trunk's actual output on
a real input -- the one link between "every module is exact" and "the fold
disagrees". Native's own tensors, captured where it hands them to its sampler
(`Protenix.sample_diffusion`, plus `DiffusionConditioning.prepare_cache` for the
raw pair, since native passes `z_trunk=None` and caches a CONDITIONED pair --
comparing ours against that one instead reads corr 0.011 and means nothing):

| tensor | 5K9P (both fold badly) | 6MRR (both fold at 0.70 A) |
|---|---|---|
| `s_inputs` | **0.99999658** | — |
| `s_trunk` (single) | 0.99823 (rms 0.896) | 0.99915 (rms 1.014) |
| `z_trunk` (pair) | **0.867** | **0.960** |

**The 6MRR column is why this is not a smoking gun.** On the target where our
fold MATCHES native at 0.70 A, the trunk pair still only correlates 0.96 -- so a
pair correlation well below 1.0 is the normal amplification of float differences
through 48 blocks x 10 recycles, and coexists with a perfect fold. 5K9P is worse
(0.867) but the same in kind. The input embedder is exact on both.

**Hypotheses tested and REJECTED here, so they are not retested:**
  * *The duplicate MSA row.* AF3 emits the query twice when the unpaired and
    paired MSAs are both empty; native keeps one. Masking ours to native's depth
    makes the trunk pair WORSE (0.867 -> 0.720). The fold number moved too
    (best 7.978 -> 3.996) but that is inside the 4-11 A spread of this target.
  * *The sampler loop.* Read against `generator.py::sample_diffusion` line by
    line: churn (`gamma_0` above `gamma_min`), `t_hat`, the noise term, the
    Euler step with `step_scale_eta`, and centre-random-augmentation at the top
    of every step all match.
  * *The noise schedule.* Identical for 200 of 201 entries; ours ends at 0.0064
    where native forces the last level to exactly 0.
  * *Recycling.* 0, 1, 4 and 10 recycles give 9.47 / 9.46 / 8.78 / 9.59 mean --
    our fold is flat in recycles on this target.
  * *`modified_res_mask`.* Native's featuriser emits it; no native model module
    reads it.

Still open, and now stated correctly: **on single-sequence 5K9P neither
implementation is reliable, and native's failures and successes are further
apart than ours.** No module, feature, or sampler difference explains it, and
the trunk evidence says the divergence is amplification rather than a discrete
fault.

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

### intellifold2's atom encoder, and the layout the CONFIG chooses (2026-09-08)

if2's encoder now has the same gate, and it passes on the layout the checkpoint
actually runs:

| tensor | corr | max\|d\|/rms |
|---|---|---|
| `c_atom_cond` | 1.000000 | 2.3e-06 |
| `q_atom` | 0.999999 | 7.4e-02 |
| `a_token` | 0.999999 | 4.8e-02 |

Reaching it turned on a fact worth keeping: **if2 ships TWO atom->token
broadcasts and the config picks one.** The default `repeat_consecutive_with_lens`
ignores its `lens` argument and repeats each token 24 times, leaving the padding
slots interleaved on the atom axis; `_advanced` honours the lens and packs the
real atoms, which is AF3's layout and ours. Run against the default, the same
weights on the same inputs score `a_token` **0.965** and `q_atom` 0.199 --
because the 32-atom windows then hold different atoms on the two sides, so the
local attention has different neighbourhoods. `v2_inference_config.py` sets
`advanced_conversion = True` for the atom encoder, the decoder and the input
embedder, so PACKED is what this checkpoint runs, and the harness defaults to it
(`IF2_DENSE=1` selects the other, which is how the two were told apart).

### rosettafold3's atom encoder, and a term the HARNESS was missing (2026-09-08)

| tensor | corr | max\|d\|/rms |
|---|---|---|
| `c_atom_cond` | 0.999998 | 5.1e-02 |
| `q_atom` | 0.999582 | 8.4e-01 |
| `a_token` | 0.999870 | 5.7e-01 |

Looser than the others, and expected to be: rf3's atom attention sets
`force_bfloat16 = True` on its own path, so the native side is computing in
bf16 where ours is fp32.

The route there is worth recording because it inverts the usual fault. Built
from rf3's own yaml the module came up 11 tensors short --
`process_atom_level_embedding.*`, the conformer embedding the yaml does not
switch on but THESE weights carry (the same yaml says 389 fused atom features
where the checkpoint has 393, so the config in the repo is older than the
release). Omitting it is not neutral: the MLP has biases and a LayerNorm tail,
so on the all-zero conformer input it still emits a fixed nonzero vector, and
`converters/rosettafold3.py::_conformer_embedding_bias` already folds exactly
that constant into our embeddings. So the first run showed `c_atom_cond` 0.939
with OUR magnitude 2.44x native's -- **the harness missing a term the port has**,
where every earlier case of this shape was the reverse. Constructing the module
with `use_atom_level_embedding=True` and feeding zeros took it to 0.999998.

Two rf3 shape conventions cost a run each and are worth stating: the noisy
coordinates carry rf3's leading diffusion-batch dim and the trunk tensors do
NOT, because `atom_attention` normalises `A_I` but unsqueezes the pair tensor
unconditionally (`Z_II[None]`); and `use_chiral_features` is on in rf3's config
with `process_ch` in the checkpoint, which is the one rf3 term the port does not
implement, so this gate drops it (`RF3_CHIRAL=1` to size it later).

The diagnostic that made this legible was `c_atom_cond` = 1.000000 under BOTH
layouts once the comparison selected native's real slots: the per-atom
embedding, which has no attention in it, cannot see the window layout, so an
exact input embedding beside a 0.965 output localises the difference to the
windowing and nowhere else.

`p_atom_pair` stays at 0.94 while the outputs it feeds are exact to 1e-5. Our
windows clamp an out-of-range key onto atom 0 -- a real atom, repeated -- where
protenix pads with zeros, and both sides mask those cells out of the attention.
The outputs agreeing to 1e-5 IS the evidence that those cells never reach the
result.

### L3 for openfold3, openbind0, intellifold2 and rosettafold3 (2026-09-08)

Every remaining vendor's whole diffusion module now runs against ours on one
step, so L3 covers every port that has a native:

| model | native tensors | missing / unexpected | corr | per-atom mean | per-atom max | native rms |
|---|---|---|---|---|---|---|
| `openfold3` | 763 | 0 / 0 | 1.000000 | **0.0001 A** | 0.0013 | 15.8 |
| `openbind0` | 740 | 24 handled / 1 | 1.000000 | **0.0021 A** | 0.019 | 19.8 |
| `intellifold2` | 706 | 0 / 0 | 0.999950 | **0.075 A** | 1.59 | 19.3 |
| `rosettafold3` | 879 | 0 / 0 | 0.999948 | **0.401 A** | 14.7 | 130.2 |

**of3 joins protenix2 as exact**, on both releases -- and that matters beyond
of3, because a denoise step runs the atom DECODER, which no gate reaches on its
own. An exact step is the only evidence the decoder is right, and there are now
three models carrying it.

**rf3 read 2.79 A until the harness handed it the right alphabet.** The first
run's error was flat across window positions (edge/interior 1.02) but varied
13x across TOKENS, and held at ~2% of the coordinate scale across noise levels
1 / 4 / 16 / 64 -- a systematic, token-dependent term, not a numerical one.
`main()` builds our 447-wide view of `s_inputs` with OF3's permutation for every
model, and rf3's alphabet is not of3's: the native side must be handed OUR
vector scattered into RF3's positions, not the vendor array read at rf3's
positions. Those two differ in exactly the transposed G/C columns. Corrected,
the step goes to **0.401 A / 0.999948**, relative error 2.16% -> 0.31%, and what
is left concentrates at the window edges (edge/interior 1.74, next to if2's
1.54) -- the padded-key item, not the decoder.

This is the third time rf3's alphabet has produced a wrong number, and the first
time it was the HARNESS rather than the port. The rule it earns: any harness
that builds a vendor-layout tensor must use THAT vendor's permutation, and the
tell is a systematic per-token error that survives every scale.

openbind0's 24 "missing" are the per-block pair LayerNorms it does not have:
v0.5.0 runs that LayerNorm ONCE for the stack where preview-2 runs it inside
every block, which is what `model_config.PER_BLOCK_PAIR_LAYER_NORM` encodes.
`~/openfold-3` only implements the per-block form, so the harness makes them
Identity and applies the single LN on the way in -- the checkpoint decides,
nothing is remembered.

**if2 lands where the other exact ports do.** Two vendor facts had to be right
first: its module returns `r_update`, NOT `x_denoised` -- the EDM scaling lives
outside it in `model.py::diffusion_edm_forward`, so the harness applies it -- and
its `s_inputs` is 447 wide, not 449 (`layer_norm_s` is 831 = 384 + 447), so if2
is NOT one of `model_config.PADDED_SINGLE_COND`. Both are the kind of difference
that still correlates well while being wrong.

**rf3 is the loose one, and its own parts are tighter than the whole.** Its
conditioning is 1.000000, its token transformer is gated, its atom encoder is
0.999870 -- yet the composed step is 2.79 A per atom (2% of a 130 A coordinate
spread). The one piece under it that NO gate reaches is the atom DECODER, which
makes it the first suspect; the second is `process_ch`, the chiral term the port
does not implement, dropped on the native side here so that both sides omit it.
Recorded as measured, not explained.

`force_bfloat16 = True` on all 30 of rf3's attention blocks has to be switched
off for this to run at all (a hard dtype error on CPU), the same surgery
`diffusion_parity.py` does for the token stack alone.

### boltz2's L3, by injection -- the last missing LEVEL (2026-09-08)

boltz2 was the only model carrying a `.` for a whole level. It cannot be gated
the way the others are: its atom path wants boltz's own feature layout (flat
atoms plus `atom_to_token`, its own `ref_*` names), which is why the port
deferred this. `dev/oracles/boltz2_denoise_parity.py` uses the tensors captured
from a real boltz run instead (`~/boltz2_6mrr/diff_dump.npz`, `score.in.*` ->
`score.out.r_update`) and feeds OUR head the same inputs.

| | corr | per-atom mean | per-atom max | native rms |
|---|---|---|---|---|
| `boltz2` (sigma 4608) | 0.999900 | **0.203 A** | 0.425 | 16.9 |

The EDM algebra makes this comparable at all, and it is boltz's own -- identical
to AF3's, `sigma_data` 16 with the same c_skip/c_out/c_in and
`c_noise(sigma) = log(sigma/16) * 0.25`. So the dump inverts:
`sigma = 16 e^(4t)`, `noised = r_noisy / c_in`, and the reference is rebuilt as
`c_skip * noised + c_out * r_update`. **`score.out.r_update` is the RAW network
output, pre-EDM, where our head returns denoised coordinates** -- comparing
those two directly is a units mismatch that still correlates well, which is
exactly the sort of thing that passes for a gate and is not one.

**It also found a featurisation difference: we give the C-terminal residue an
OXT and boltz does not.** Token 67 (GLU) carries 10 atoms our side and 9 in
boltz, so we hand boltz's weights an atom boltz never sees. One atom in 574 --
small, and permanent for every boltz2 fold. The gate aligns the two lists per
token and compares the 573 that correspond, rather than assuming they match:
boltz's flat axis is 576 = 573 real + 3 PADDING rows, whose all-zero
`atom_to_token` rows make `argmax` attribute them to token 0, which is why a
naive per-token count reads 4-vs-7 there and means nothing.

### L3 across the whole protenix family (2026-09-08)

With the atom encoder/decoder block counts now READ off the checkpoint rather
than defaulting to 3 (the fault that left 476 tensors unmapped for the small
releases), `denoise_parity.py` covers all six:

| model | native tensors | unmapped | corr | per-atom mean | per-atom max |
|---|---|---|---|---|---|
| `protenix2` | — | 0 | 1.000000 | **0.0000 A** | 0.0000 |

Both small releases land in the same partial band as the other `c_z` 128
checkpoints, and NOT at protenix2's exact match -- consistent with the open
`c_z` 128 atom-stack item below rather than with anything specific to the
distilled models. 0 missing and 0 unmapped on both sides means the gap is
numerical, not a weight that never arrived.

## L1b, the trunk's other half (2026-09-07)

`dev/oracles/prot_parity.py` gates the MSA MODULE as well as the trunk, which is
what closes `CLAMPED_OPM_NORM` and `NO_MSA_ROW_UPDATE` -- two of the four trunk
conventions L1 explicitly does NOT cover. All three protenix releases that carry
an MSA stack agree with native:

| model | pairformer blocks | MSA blocks | single | pair | msa -> pair |
|---|---|---|---|---|---|

(corr, with max relative error in brackets.) It needed the same
dims-from-constants fix as everything else here: `PairformerStack` builds
`c_hidden` 128 whatever `c_z` is, so protenix2 died in `load_state_dict` until
the harness passed `hidden_scale_up=True` -- which trunk_parity.py had been
passing all along.

## The MSA module beyond protenix (2026-09-08)

`prot_parity.py` gates the MSA module for protenix only -- which is what closes
`CLAMPED_OPM_NORM` and `NO_MSA_ROW_UPDATE`, two of the four trunk conventions L1
does NOT cover. `dev/oracles/msa_parity.py` starts on the rest:

| model | msa -> pair | rms ours/native |
|---|---|---|
| `openfold3` | **1.000000** | 1.0000 |
| `openbind0` | **1.000000** | 1.0000 |
| `intellifold2` | **1.000000** | 0.9999 |
| `opendde` | **1.000000** | 1.0000 |
| `rosettafold3` | 0.999971 | 0.9976 |
| `boltz2` | **0.974331** | 1.0584 |

With `prot_parity.py`'s protenix2 and protenix1 that is **8 of the 10** models
carrying an MSA stack. Only chai1 (TorchScript, no callable submodule forward)
and the esmfold2 family are unmeasured.

**boltz2 is the one that is not exact, and it does NOT compound.** One block
reads 0.967594 and four read 0.974331 -- so the divergence is present in a
SINGLE block rather than accumulating, which puts it inside the layer body:
`pair_weighted_averaging`, `msa_transition`, `outer_product_mean`, or the
`pairformer_layer`. The last is the least likely, since boltz2's 48-block trunk
pairformer is gated at 1.000000/1.000000 on the same class.

Checked and not the cause: the update-then-OPM order (ours gates on it and
opendde, which shares that order, reads 1.000000), `use_paired_feature` (read off
`msa_proj`'s width, 33+3), `get_dropout_mask` (returns ones under `eval()`), the
OPM normalisation (`CLAMPED_OPM_NORM` is esmfold2-only and the difference there
is 1e-3 relative, not 3%), and anything applied to `m`/`z` before the loop
(nothing is).

**Recorded as OPEN, and it may still be the harness.** Three of this session's
new gates produced plausible degraded numbers that turned out to be harness
faults, and this one has not yet been cross-checked against an independent
gate the way those were. What would settle it: instantiate boltz's `MSALayer`
alone and compare the `m` output as well as `z`, which splits the four
sublayers.

It compares the PAIR output, which is the half that survives into the trunk;
comparing only the msa rows would miss a wrong outer-product normalisation
entirely.

**if2's LAST MSA block has no msa update, and the checkpoint says so.** Block 3
carries no `msa_pair_weighted_averaging` at all -- 12 tensors that simply do not
exist, because its msa output is never read. That is what if2's
`skip_unused_modules` flag is for, and `v2_inference_config` sets it True;
constructing the stack without it asks for those tensors and reports 12
missing. The same idea as protenix's last-block omission that
`NO_MSA_ROW_UPDATE` was written for, arrived at independently by a different
vendor.

**Two rf3 facts this pinned down.** Its MSA keys carry NO block index: the
forward loops `n_block` times over ONE set of submodules, so the checkpoint
holds a single copy and the depth cannot be derived from key names (it is 4, from
rf3's own yaml). `converters/rosettafold3.py` already mirrors that sharing --
`_rosettafold3_msa_block` ignores its block index and `_stack_blocks` replicates
the one block across our four layers. And its signature is
`forward(f, Z_II, S_inputs_I)`, pair BEFORE s_inputs; reversing them feeds the
128-wide pair into a 449-wide projection and dies in the matmul, which is the
good outcome.

**A harness error worth recording, because I nearly published it.** The first
run read corr 0.468. Our side had been fed a RANDOM c_m activation while native
embedded the raw MSA rows itself -- two different inputs, so the number measured
nothing. My own docstring had described that as "the comparison starts one
projection later", which was hand-waving over an invalid comparison. Taking the
embedded MSA from native's own `msa_subsampler` and feeding it to both sides
gives 0.999971. **If a gate's two sides do not provably see the same input, its
number is not a measurement.**

## The atom DECODER, gated directly for the first time (2026-09-08)

`DECODER=1 python dev/oracles/atom_parity.py protenix2` runs protenix's own
`AtomAttentionDecoder` beside ours on one shared token activation, each side
using its OWN encoder output -- legitimate precisely because the encoder is
already exact for this model (1.000000), so the two sets of skips agree to
~1e-5 and anything larger in the output is the decoder.

| | corr | per-atom \|dr\| mean | max |
|---|---|---|---|
| `protenix2` | **1.000000** | **0.000002 A** | 0.000013 |

Until now the decoder was covered only in COMPOSITION, by the models whose whole
denoise step is exact. That was real evidence but could not localise a
decoder-only fault; now it can.

**A harness fault worth recording, because the number looked like a finding.**
The first run read `r_update` 0.976. Our side had been built by splicing three
fields (skip, queries_single_cond, pair_cond) into an encoder output computed
from ZERO inputs -- so `keys_single_cond`, which the decoder's cross-attention
consumes, came from a different invocation than the rest. Running the encoder
and decoder in ONE transform on the real inputs gives 1.000000. The tell was
available before any debugging: **protenix2's whole denoise step is exact at
0.0000 A/atom, and L3 runs this decoder, so a genuinely 0.976 decoder was
arithmetically impossible.** When a new gate disagrees with an established one,
suspect the new gate.

## What has NO gate at all (2026-09-08)

The L0-L6 table answers "how far down does each model's coverage go" and hides
the more useful question: which MODULES has nothing ever compared? The levels
were organised around the diffusion path, so the answer is not visible there.
Enumerated against the graph's own module list rather than from memory:

| module | models carrying it | gated on | note |
|---|---|---|---|
| ~~template embedder~~ | 9 | **8** | gated 2026-09-08, found TWO bugs; every model but chai1 |
| **MSA module** | 10 | **8** | all but chai1 (TorchScript) and esmfold2; boltz2 measured at 0.974 and OPEN |
| ~~distogram head~~ | all | **8** | CLOSED 2026-09-08, `dgram_parity.py`; chai1 is n/a (no native head) |
| ~~input embedder~~ | all | **2** | CLOSED 2026-09-08, `real_trunk_parity.py` |
| ~~recycling loop~~ | all | **2** | same gate — it compares the trunk AFTER all recycles |
| ~~atom decoder~~ | all | **1** | CLOSED 2026-09-08, `atom_parity.py DECODER=1` — protenix2 exact |

### The template embedder: gated for the first time, and it found TWO bugs (2026-09-08)

Nine models carry a template stack and nothing had ever compared one against
its vendor. `dev/oracles/template_parity.py` now does, and the first run was
NOT clean -- 0.998468 for protenix2, where the 48-block trunk pairformer reads
1.000000 and this stack is two blocks. Two independent port bugs, the second
hidden behind the first:

| protenix2 template embedder | corr | rms ours/native |
|---|---|---|
| as shipped | 0.998468 | 1.0093 |
| + restype_i/j in native's order | 0.999998 | 1.0002 |
| + no outer residual | **1.000000** | 1.0000 |

**Bug 1: `restype_i` and `restype_j` were swapped.** protenix appends
`expand_at_dim(aatype, dim=-3)` then `expand_at_dim(aatype, dim=-2)`, and
`expand_at_dim` UNSQUEEZES at that dim -- so `dim=-3` inserts the new axis first
and leaves the block varying along **j**, `dim=-2` along **i**. Native's first
32-column block is therefore the j-indexed one; ours was the i-indexed one, and
`a_proj` is converted with no column permutation, so the two 32-column blocks
fed the wrong weights.

**Bug 2: an outer residual protenix does not have.** Our three template classes
share one fused forward which does `v = v + stack(v)`. boltz2's native really
does that (`v = v + self.pairformer(v, ...)`); **protenix does not**
(`_, v = self.pairformer_stack(s=None, z=v, ...)`), and neither does rf3. The
blocks are internally residual on both sides, so the outer term adds the input a
SECOND time. Now `model_config.TEMPLATE_STACK_OUTER_RESIDUAL`.

**At the fold level, on a templated 5K9P (self-template, seed 101):**

| | best | mean |
|---|---|---|
| before | 1.588 | 2.124 |
| after | **0.227** | **0.560** |

Seven times better best-RMSD. Untemplated folds are untouched (6MRR protenix2
0.700 against a 0.702 baseline, protenix1 1.695 against 1.694), and so are the
models sharing the code (boltz2 0.426 vs 0.434, rosettafold3 0.976 vs 0.986).
No weight changed; both fixes are forward-graph.

**Why folds never caught it.** A wrong-but-plausible template contribution still
points the fold in roughly the right direction -- 1.588 A looks like a working
template, and templates were only ever verified by folds. It took a module
comparison to see that 1.588 should have been 0.227.

**And the docstrings claimed this path was already validated** -- "VALIDATED
exact vs native geometry" for the features, "corr 1.0 vs Boltz's TemplateModule"
for the forward. The second was true for BOLTZ2 and was silently inherited by
protenix and rf3, which is exactly how a shared forward hides a per-vendor
convention. A docstring is not a gate.

**rosettafold3: gated, and it never had either bug.** corr **1.000000**
(max|d| 1.5e-04). Its features are the 66-channel distance conditioning
`[distogram_condition(64), has_condition(1), joint_noise_level(1)]`, all
i/j-SYMMETRIC, so bug 1 cannot apply -- there are no restype blocks. And it
escaped bug 2 because `RoseTTAFold3TemplateEmbedding` overrides `__call__`
rather than only `_features`, and its own copy already does `v = stack(v)`.

So listing it in `TEMPLATE_STACK_OUTER_RESIDUAL` was INERT, which the gate
settled rather than the source reading: rf3 reads 1.000000 bit for bit with the
name in the tuple or out of it. It is now correctly absent, and 6MRR is
unchanged (0.976 against a 0.986 baseline).

**boltz2: gated, exact, and it CONFIRMS the outer residual.** corr **1.000000**
(max|d| 3e-05) -- and this is the strongest of the three adapters, because boltz
takes the RAW geometry (`template_frame_rot`, `template_frame_t`, ca/cb and
their masks) and builds its own 109 channels, so the gate tests our feature
DERIVATION as well as the forward. The protenix adapter feeds native our own
108-d features and therefore tests only the forward.

It also settles the one reading nothing had checked: ours keeps
`v = v + stack(v)` for boltz2 and matches native exactly, so boltz2 really is
the sole member of `TEMPLATE_STACK_OUTER_RESIDUAL` -- verified from both sides
now, not inferred from one.

`compute_frame` is transcribed verbatim into the harness rather than imported:
`boltz.data.tokenize.boltz2` pulls `boltz.data.types`, which needs mashumaro,
which is not in this venv and must not be installed into it. The model module
itself imports fine.

**openfold3 and openbind0: gated, both 1.000000.** They matter because of3 is
on AF3's template design -- one Linear PER FEATURE rather than one fused
`a_proj` -- which is precisely where protenix's restype bug would have lived if
the converter had mapped positionally: of3's `aatype_linear_1` takes the
**i**-varying block and `aatype_linear_2` the j-varying, while AF3's `to_concat`
puts the **j**-varying block FIRST. `converters/openfold3.py` already crosses
them (`[(3, 'aatype_linear_1'), (2, 'aatype_linear_2')]`) with a comment naming
the trap, and this gate is what turns that comment into a measurement.

**intellifold2: gated at 0.999999.** Same AF3 per-feature design, and it names
the two projections outright: `linear_aatype_col` takes `unsqueeze(-3)` (the
**j**-varying block) and `linear_aatype_row` `unsqueeze(-2)`, with
`converters/intellifold2.py` mapping col->slot 2 and row->slot 3 to match AF3's
order. The residual is in line with if2's bf16 storage floor (its 48-block trunk
pair reads 0.998994 for the same reason). Its template stack is the widened
"full_fat" tree -- c_t 256, 8 heads, c_hidden 32/256, NOT AF3's 64/16/4 -- so
every width is read off the checkpoint; hardcoding AF3's numbers fails in
load_state_dict with eight size mismatches rather than comparing quietly.

**chai1 cannot be gated here, and the reason is structural.** chai ships
TorchScript (`models_v2/trunk.pt`), and while `template_embedder` IS reachable as
a child module with its parameters, it has **no callable `forward`** -- the
computation exists only inline in the trunk's traced graph
(`AttributeError: Method 'forward' is not defined`). So its weights are covered
at L0 and its behaviour end to end, but the module cannot be invoked in
isolation from the shipped artifacts. That is 8 of 9, with the ninth blocked by
packaging rather than by effort.

**opendde: gated at 1.000000, and its converter was the one that had this
right.** opendde is protenix-lineage, so its template embedder takes ONE fused
108-d `linear_no_bias_a` while our side runs AF3's per-feature
TemplateEmbedding -- meaning `converters/opendde.py` SPLITS that fused weight
into AF3's nine slots. That split encodes exactly the convention the protenix
fix established today: cols 40:72 are the **j**-varying restype block (AF3 slot
2), 72:104 the i-varying (slot 3). So the opendde converter and
`Protenix2TemplateEmbedding` disagreed about the same vendor's layout, and the
converter was correct. Reading one against the other would have found this
without a gate; nobody did.

Two harness notes from the of3 pair. of3's config subtree is
`architecture/template`, not `template_embedder` -- that is the CHECKPOINT
prefix, and searching the config for it finds nothing. And the `Templates` we
hand our own module must be WRITABLE copies: `construct_input` does
`dense_atom_positions *= dense_atom_mask[..., None]` in place, which numpy
refuses on a read-only view and jax tracing hides.

**The moral is not that duplication saved us.** protenix inherited the shared
forward and got boltz2's convention; rf3 escaped only by not inheriting. Either
the per-vendor convention is NAMED -- as it now is -- or the next subclass
silently gets whatever its parent happened to do.

One harness detail worth keeping: rf3 rebuilds the noise channel itself from a
PER-TOKEN scale as `f(sqrt(ns_i^2 + ns_j^2))`, where ours uses a scalar eps as
the JOINT level directly. The gate therefore passes `ns = eps/sqrt(2)`; getting
that wrong shifts one of 66 channels and would read as a port difference.

Templates do work end to end (boltz2 folds 5CAJ to 0.72 A with one,
rosettafold3 to 1.56 A), which bounds how bad this can be -- and the one earlier
template test here, zeroing our contribution on 6MRR and seeing nothing change,
proved only that the path is inert when NO template is supplied.

**The distogram gap is CLOSED (2026-09-08).** It mattered out of proportion to
its size -- it is the head design gradients flow through (`zero recycles,
backprop into the distogram and stop`), so a divergence would have been
invisible to every structural number here while corrupting exactly the use case
the design work depends on. `dev/oracles/dgram_parity.py` now gates it on six
models, and all six are EXACT:

| model | c_z | bins | bias | corr | max\|d\| |
|---|---|---|---|---|---|
| `protenix2` | 256 | 64 | yes | 1.000000 | 0.00000 |
| `protenix1` | 128 | 64 | yes | 1.000000 | 0.00000 |
| `openfold3` | 128 | 64 | no | 1.000000 | 0.00000 |
| `openbind0` | 128 | 64 | no | 1.000000 | 0.00000 |
| `intellifold2` | 512 | 64 | no | 1.000000 | 2.2e-07 |
| `rosettafold3` | 128 | **65** | yes | 1.000000 | 2.4e-07 |
| `boltz2` | 128 | 64 | yes | 1.000000 | 2.4e-07 |
| `opendde` | 384 | **96** | yes | 1.000000 | 0.00000 |

The head is one projection off the trunk pair, so synthetic z suffices -- no
real batch, unlike the atom gates. What it actually checks is the thing a
correlation would hide: **where each vendor symmetrises.** protenix, of3 and if2
compute `Linear(z) + Linear(z)^T` = W(z+z^T) + 2b; **rosettafold3 symmetrises
BEFORE the linear**, `predictor(z + z^T)` = W(z+z^T) + b. Identical for the
weight, off by a factor of two on the BIAS -- which is why
`converters/rosettafold3.py` stores b/2, and this gate is what verifies that
claim instead of trusting the comment beside it. Both sides come out exactly
symmetric on every model.

**boltz2 is on rf3's side of that split** (`z = z + z.transpose(1, 2)` then the
linear) and `converters/boltz2.py` already halves its bias; opendde is on
protenix's side and correctly does not. Both now verified rather than asserted.
Bin counts differ too and are read off the checkpoint, never assumed: 64 for
most, 65 for rf3, **96 for opendde**.

`chai1` is the one model with no native distogram to compare: it has no such
head of its own, and ours was trained post-hoc on its frozen trunk
(`sokrypton/chai-lab@dgram`), so there is nothing to be faithful TO. That is a
genuine n/a rather than a gap.

**The input embedder and the recycling loop are CLOSED (2026-09-08)**, by
promoting the ad-hoc comparison that came out of the 5K9P investigation into
`dev/oracles/real_trunk_parity.py` + `native_trunk_dump.sh`. It compares against
native's own tensors from a real inference job -- so native's FEATURISER too --
and therefore covers three things nothing else did: the input embedder, the
trunk's real output, and recycling (every other gate measures a single pass).

| tensor | 5K9P plain | 6MRR (the control) |
|---|---|---|
| `s_inputs` | **0.99999663** | **0.99999699** |
| `s_trunk` (single) | 0.99823 (rms 0.896) | 0.99915 (rms 1.014) |
| `z_trunk` (pair) | 0.867 | 0.960 |

**The control column is the point, and it is now part of the harness.** A trunk
correlation well below 1.0 is NORMAL here: 48 blocks x 10 recycles amplify float
differences, and 6MRR -- where our fold matches native at 0.70 A -- still reads
0.960 on the pair. Reading 0.867 as a defect without that control cost most of a
session, so the oracle's docstring says to run a well-folded target beside every
suspect one.

Two native-side traps are baked into the dump script, because each silently
produces a meaningless number:
  * `Protenix.sample_diffusion` must be patched on the CLASS. The module-level
    `generator.sample_diffusion` is imported by value into protenix.py, so
    patching that one lets the job run to completion with no dump at all.
  * The RAW pair must come from `DiffusionConditioning.prepare_cache`. protenix
    precomputes a CONDITIONED pair and passes `z_trunk=None`, so the tensor
    reaching the sampler as `pair_z` is not the trunk pair -- comparing ours
    against it reads **corr 0.011**, which looks like catastrophe and means
    nothing.

Left: the template embedder on the other 8 models, and the atom decoder.

## The gates, and where they live

`dev/` is gitignored (see README, "Where the harnesses live"), so these exist on
the machine that runs them and NOT in the repository. Each is listed with what
it covers, because the file itself is the only other record:

| gate | level | covers |
|---|---|---|
| `dev/oracles/trunk_parity.py` | L1 | pairformer stack vs the vendor's module — 7 models |
| `dev/oracles/prot_parity.py` | L1b | protenix trunk AND MSA module (protenix2, protenix1) |
| `dev/oracles/conditioning_parity.py` | L2 | diffusion pair + single conditioning — protenix2, protenix1, rf3 |
| `dev/oracles/atom_parity.py` | L2 | atom cross-attention encoder, real batch, windowed — protenix2/1, both of3, intellifold2, rosettafold3 |
| `dev/oracles/diffusion_parity.py`, `l2_all.sh` | L2 | token diffusion transformer — 10 models |
| `dev/oracles/boltz2_denoise_parity.py` | L3 | boltz2's denoise step by INJECTION from a captured boltz run |
| `dev/oracles/denoise_parity.py` | L3 | one denoise step, whole diffusion module — protenix2/1, both of3, intellifold2, rosettafold3 |
| `dev/oracles/template_parity.py` | L1 | template embedder vs the vendor's own module — 8 models (all but chai1); found 2 bugs |
| `dev/oracles/real_trunk_parity.py` + `native_trunk_dump.sh` | L1 real-input | input embedder, trunk output and recycling, against native's own featurised run — protenix2/1 |
| `dev/oracles/dgram_parity.py` | L4 | distogram head — protenix2/1, both of3, intellifold2, rosettafold3, boltz2, opendde |
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
