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
| `esmfold2` family | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ — ligands 0.645 Å, DNA 2.9 Å, RNA 25 Å |
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

**The atom DECODER now has a gate of its own** -- `DECODER=1
dev/oracles/atom_parity.py`, added 2026-09-08 and written up below. Before that
it was covered only in composition, by the models whose whole denoise step (L3)
is exact (`protenix2` 0.0000 A per atom, `openfold3` 0.0001, `openbind0`
0.0021): real evidence, but unable to localise a decoder-only fault. It was the
largest remaining structural hole in this document and it is closed.

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
  * `CLAMPED_OPM_NORM`, `OPM_ROW_COUNT_NORM` -- `OuterProductMean`, in the
    MSA module. Both are now covered by L1b (`msa_parity.py`), and the second
    exists BECAUSE L1b was written: it is a real port bug this list's own
    "NOT covered" line predicted.
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
CONDITIONING and the ATOM ENCODER** -- and only for the models `conditioning_parity.py`
and `atom_parity.py` do not yet reach (boltz2, opendde, chai1, esmfold2). The
atom DECODER left this list on 2026-09-08 (`DECODER=1 atom_parity.py`). Down
from fifteen tables to a per-model remainder rather than a whole-stage hole.

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
(of3, openbind0, intellifold2, rosettafold3); and the atom DECODER, gated
directly since 2026-09-08 (`DECODER=1 atom_parity.py`). LEFT: the window-edge
residual that if2 and rf3 both show, which is the padded-key mask item. `opendde` is the one model with no
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

**of3 joins protenix2 as exact**, on both releases -- and at the time that
mattered beyond of3, because a denoise step runs the atom DECODER and no gate
then reached it on its own. An exact step was the only evidence the decoder was
right, and there are now three models carrying it. (The decoder has since been
gated directly; the composition evidence is what made that gate's own first
number, 0.976, recognisable as a harness fault rather than a finding.)

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
spread). At the time the one piece under it that no gate reached was the atom
DECODER, which made it the first suspect; the second is `process_ch`, the chiral
term the port does not implement, dropped on the native side here so that both
sides omit it. Recorded as measured, not explained. **Resolved since**: the
harness was handing rf3 native OF3's alphabet permutation, and with our
447-vector scattered into rf3's own positions the step reads 0.401 A -- so
neither suspect was the cause.

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
| `boltz2` | **1.000000** | 1.0000 (was 0.974331 / 1.0584 -- a real port bug, below) |
| `esmfold2` | **1.000000** | 1.0000 |
| `esmfold2_exp` | **1.000000** | 1.0000 |
| `esmfold2_exp_cutoff2025` | **1.000000** | 1.0000 |

With `prot_parity.py`'s protenix2 and protenix1 that is **11 of the 12** models
carrying an MSA stack. The one left is chai1 (TorchScript, no callable submodule
`forward`), blocked by packaging rather than by effort.

**And five of the eight esmfold2 releases were never a gap.** `msa=0` in
`model_registry.ESMFOLD2_VARIANTS` for every `*_fast` and `lm*` row: those
configs set `msa_encoder.enabled` false and their checkpoints carry no such
weights, so they are **n/a**, not ungated. Only `esmfold2`, `esmfold2_exp` and
`esmfold2_exp_cutoff2025` have an MSA encoder, and all three now read 1.000000.
Counting the family as one unmeasured row overstated the hole by four models --
the same error the level table makes at the model granularity.

**The esmfold2 adapter is the only one here that does not run the vendor
in-process**, and the reason is worth recording: ESMFold2's implementation ships
inside `transformers` (`models/esmfold2/modeling_esmfold2.py`), which is
installed in `~/venv_esm` only, and it must not be installed beside JAX in the
GPU venv. So `dev/oracles/esmfold2_msa_dump.py` runs the native module there and
writes its INPUTS as well as its output to an npz that `msa_parity.py` reads
with numpy alone. Writing the inputs is the load-bearing part: two independently
seeded `default_rng(0)` streams in two processes are not the same tensors, and
comparing on them would have measured nothing.

**A divergence that looked real and is not: the DEAD final block.**
`MSAEncoder.__init__` hardcodes `is_final_block=(i == n_layers - 1)`, so
transformers' last block has no `msa_pair_weighted_averaging` / `msa_transition`
at all. On the released line that matches the checkpoint. On the EXPERIMENTAL
line the checkpoint DOES ship those 12 tensors and `load_state_dict` reports
them as *unexpected* -- native inference never runs weights it was shipped with,
while our port does (`converters/esmfold2._drops_msa_update` reads the
checkpoint rather than the index, deliberately).

It makes no difference, and the gate is what proves it rather than an argument
about intent. `MSAEncoderBlock` runs the OPM into the pair FIRST and updates `m`
after, and `MSAEncoder` returns only `x_pair` -- so the last block's `m` is
never consumed by anything. Dumped both ways (`--keep_final_update`), our side
compares at 1.000000 against EITHER, max|d| 0.055 vs 0.063 on an rms of 2168.
Those 12 tensors are dead weight in both implementations.

Run with a NON-UNIFORM mask too (`--nonuniform` / `NONUNIFORM=1`): 1.000000,
which is the case that caught the wrong boltz2 fix below.

**boltz2 was the one that was not exact, and it was a REAL PORT BUG.** Fixed;
the localisation is worth writing down because it is the cleanest example in
this file of a gate that only bites under the right input.

The number did NOT compound -- one block read 0.967594 and four read 0.974331 --
so the divergence lived inside a single layer body rather than accumulating.
`msa_parity.py LAYER=1` then drives boltz's `MSALayer` alone and compares BOTH
outputs: `m` came back exact at 1.000000 while `z` read 0.925, which isolates it
to the pair-producing sublayer, and OPM alone read 0.992598.

The bug is in `OuterProductMean` (`modules.py`), and it is an ORDERING one:

    AF3:    act = einsum(a, b) @ output_w + output_b ;  return act / norm
    boltz:  act = einsum(a, b) ;  return proj_o(act / num_mask)   # bias AFTER

so ours divided the bias by the row count and boltz does not. Predicted before
measuring: the residual should be exactly `(1 - 1/n) * output_b`, a per-channel
CONSTANT. Confirmed at -0.008120 against a prediction of -0.008121, spread
6.5e-04. Gated by `model_config.OPM_ROW_COUNT_NORM` (named for the wrong first
reading, kept so the comment explaining it stays findable).

**And the fix was wrong the first time, which the gate caught.** boltz's source
reads `num_mask = mask.sum(1).clamp(min=1)`, which looks like a per-token ROW
count where AF3 uses `einsum('abc,adc->bdc', mask, mask)`, the count of rows
covering BOTH i and j. It is not: boltz builds the PAIRWISE mask first
(`mask[:, :, None, :] * mask[:, :, :, None]`) and only then sums, so the counts
are the same and `clamp(min=1)` vs `+ 1e-3` is the only remainder.

Implementing the per-token reading left the gate **exact** on an all-ones msa
mask and made a NON-UNIFORM one WORSE (OPM 0.996 -> 0.968, rms 0.980 -> 0.886).
A uniform mask cannot tell the two normalisers apart -- every row covers every
token, so the two counts are equal -- which is why `msa_parity.py` runs
`NONUNIFORM=1` as well, and why the second effect showed up at all: masking part
of the MSA turns a per-channel constant residual (spread 6.5e-04) into one that
varies across (i, j) (spread 2.1e-01). With the bias placement alone, both the
uniform and the non-uniform case read **1.000000**, and the four-block stack
went 0.974331 -> **1.000000**.

**It moved a fold, which is the point of the level.** The bug needs MSA depth >
1 to exist at all, so 6MRR (single-sequence) could never show it -- 0.424 A
before and after. The MSA-bearing case did:

| boltz2 | protein A | BTN A |
|---|---|---|
| `ligand_1stp` before | 0.385 | 0.907 |
| `ligand_1stp` after | **0.277** | **0.458** |

`complex_1lmb` (no MSA) is unmoved: A 0.307 / B 0.230 / C 0.273 / D 0.283,
whole complex 0.391. `OPM_ROW_COUNT_NORM` is boltz2-only, so no other model's
numbers can move.

Two lessons, both already earned elsewhere in this file and earned again here:
a gate is only as good as the input variety it runs (the non-uniform mask is the
whole reason the wrong fix did not ship), and a difference that is a per-channel
constant is invisible to correlation -- see [[correlation-hides-bias]] -- so the
residual has to be looked at, not just `corr`.

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

## ESMFold2's diffusion conditioning read AF3's chain bucket (2026-09-08)

The trunk's relative-CHAIN convention was fixed at `fbac0fc` -- ESMFold2 keys
that bucket on same-CHAIN and sends the MATCH to `2c+1`, where AF3 keys it on
same-ENTITY and sends the MISMATCH there, so on a monomer EVERY pair takes a
different bucket. It was worth 1.522 -> 0.719 A on `esmfold2_lm600m`.

**It was applied to one of the two call sites.** `evoformer.py:343` passes
`chain_bucket_on_same_chain=(model in ESMFOLD2_FAMILY)`; `diffusion_head.py:201`
did not, so the diffusion conditioning read AF3's convention while the trunk
read ESMFold2's. Cost, measured by a new esmfold2 adapter in
`conditioning_parity.py` driven by the reference:

| | before | after |
|---|---|---|
| `pair_cond` | 0.998925, rms ours/native **1.0354** | **1.000000**, rms 1.0000 |
| `single_cond` | 1.000000 | 1.000000 |

`single_cond` being exact throughout is what pointed at the pair path, and from
there at its only non-parametric input. Graph-only: no blob changes, so no
republish. protenix2 re-checked at 1.000000/1.000000 -- the flag is inert
outside the family.

**Two call sites for one convention** is the same shape as protenix1's
`padded_keys` earlier the same day, and as the template outer residual before
that. The flag is spelled identically in both places now, so a grep finds them
together.

### THE RESIDUAL WAS A REAL BUG: the atom DECODER had no window and no rotary

ESMFold2's decoder is the SAME STACK as its encoder -- the reference calls
`atom_stack(qd, c0, ad, 'blocks/', n, cos, sin, mask)` with the encoder's own
cos/sin and the encoder's mask. Ours ran it with AF3's `same_token_mask` and
**no rotary at all**, so its atom attention was restricted to atoms of the SAME
RESIDUE and carried no positional signal.

Found by bisection rather than by reading. `ZERO_ATOM=enc|dec` makes one of the
two stacks an identity:

| | corr | rms |
|---|---|---|
| decoder made an identity (encoder active) | **1.000000** | **0.0000 A** |
| encoder made an identity (decoder active) | 0.991137 | 1.2753 A |

The whole residual was the decoder, and the encoder was already exact on its own
gate. With it fixed, the two RELEASED variants have real L1 and L3 cells against
the reference for the first time:

| | L1 trunk | L3 denoise |
|---|---|---|
| `esmfold2` | corr 0.999957, relerr 4.9e-03 | 0.2363 A, corr 0.999751 |
| `esmfold2_fast` | corr 0.999850, relerr 2.4e-02 | 0.2027 A, corr 0.999872 |

and both of those residuals are the two featurisation differences below, which
go to 0.0000 A when equalised. The six EXPERIMENTAL releases cannot use this
harness at all: the reference implements the parcae SSM recurrence and they
carry `pair_loop_proj` instead (`ESMFOLD2_SSM_RECYCLE` is the released line
only), so it raises `KeyError('parcae_log_delta')`. Their harness pair is
`esmfold2_oracle_exp_trunk.py` + `esmfold2_localise_exp.py`. Fixed by carrying `swa_mask` / `rope_q` / `rope_k` on
`AtomCrossAttEncoderOutput` and handing them to the decoder's transformer --
they are gathers of the same flat atom list, so rebuilding them in the decoder
is exactly where a subtle mismatch would go. `enc.swa_mask is None` for every
other model, which falls back to `same_token_mask` and no rope, so nothing else
moves (protenix2's denoise re-checked at 0.0000 A).

| esmfold2 | before | after |
|---|---|---|
| denoise, same inputs | 1.3941 A | **0.0000 A** (corr 1.000000, max 0.0001) |
| denoise, real featurisation | 1.3411 A | **0.2524 A** (corr 0.999720) |
| 6MRR fold | 1.493 best / 1.732 mean | **1.393 / 1.607** (native 1.739) |

Graph-only: no parameters change, so no blob and no republish.

That the model folded at all -- indeed better than native -- with its decoder
attending only within residues is worth pausing on. The encoder does the
long-range work and the decoder only has to turn atom features into a position
update, so a wrong mask there degrades rather than destroys. It is exactly the
kind of error a fold gate cannot find and an activation gate finds in one bisect.

### What the remaining 0.2524 A is: two featurisation differences

Continued after the conditioning fix. `atom_parity.py` now has an esmfold2
adapter driven by the reference, which closes L2's atom-encoder cell for the
family and is a much faster handle than the denoise gate:

| esmfold2 atom encoder | |
|---|---|
| `c_atom_cond` (the conditioning every adaLN reads) | **1.000000** |
| `a_token` | 0.999756, rms ours/native 0.9990, max\|d\| 2.19 on rms 4.52 |
| `q_atom` | 0.999805 |

**And the max is one token.** `DIAG=1` gives per-token max\|d\|: token 67 --
the LAST -- reads 2.189 where the next worst is 0.729 and the median is 0.244.
That token has TEN atoms on our side and NINE on ESMFold2's: our featuriser
emits the terminal **OXT** and its `PROTEIN_HEAVY_ATOMS` table does not. The
encoder pools atoms to tokens with `scatter_mean`, so ten atoms against nine
changes that token's mean outright. **That is an input difference, not a port
bug** -- and arguably ours is the more correct input, since OXT is a real atom.

It is also why the denoise gate's error is worst in the last 64 atoms (1.78
against 1.08 in the middle): with a +/-64 rank window, our extra atom is a KEY
for exactly that many.

`SAME_ATOM_SET=1` masks that atom so both sides see the same 573, and it
splits the residual cleanly:

| esmfold2 atom encoder | max\|d\| | max\|d\|/rms | worst token |
|---|---|---|---|
| as featurised (574 vs 573 atoms) | 2.189 | 0.484 | **67**, the last |
| same atom set | **0.643** | **0.142** | 39 |

So the terminal OXT is the single largest term, and it is an INPUT difference.

**And `NATIVE_REF_POS=1` closes the rest: the atom encoder is EXACT.**

| esmfold2 atom encoder | a_token | max\|d\| |
|---|---|---|
| as featurised | 0.999756 | 2.189 |
| same atom set | 0.999898 | 0.643 |
| same atom set + ESMFold2's own `ref_pos` | **1.000000** | **0.00003** |

Per-token median 0.0000. `q_atom` and `c_atom_cond` likewise 1.000000. So the
whole residual was TWO FEATURISATION DIFFERENCES and no port bug at all:

  1. the terminal **OXT**, which we emit and its `PROTEIN_HEAVY_ATOMS` table
     does not;
  2. **`ref_pos`** -- our CCD/RDKit ideal conformer against its
     `PROTEIN_REF_POS` table, differing by mean 3.31 A in local FRAME. It feeds
     the rotary embedding, which is why it reaches attention at all.

Note (2) sits oddly beside the earlier finding that the reference's DENOISE
moves only 0.0791 A when its conformer is swapped: both are true, because a 14%
difference at `a_token` (max 0.64 on rms 4.52) is worth little by the time the
whole score network has run. The encoder is where it is visible.

**Neither is a bug to fix blind.** ESMFold2's weights were trained against its
own conformer table, so feeding the CCD ideal is out-of-distribution positional
information -- but our folds MATCH OR BEAT native on 7 of its 8 releases, so the
practical cost is nil or negative. Adopting its table for esmfold2 would be a
featurisation gate; it is recorded here as a decision, not taken silently.

**A single-block A/B is the obvious next step and my first attempt at one was a
broken harness** -- assembling CrossAttTransformer's ten arguments by hand
(the reference's `enc_queries_in` and `c0` mapped into our windowed layout, its
own rope tables, our masks and gathers) read corr 0.107 with every parameter
loaded and none at init. That is a fifth harness fault, not a finding, and it is
recorded here so the next attempt starts from `atom_parity.py`'s working
assembly rather than a fresh one.

### Everything the residual is NOT

The denoise step did NOT close: 1.3607 -> 1.3411 A per atom, corr 0.990764,
against every other port's 0.0000-0.401 A. Recorded OPEN, with the exclusions,
because a list of what a gap is not is worth more than a guess at what it is:

  * **the trunk.** `INJECT=1` runs our denoiser on the REFERENCE's trunk output:
    1.3396 A, unchanged. (The trunk itself is corr 0.999957, relerr 4.9e-03.)
  * **the conditioning**, now exact both halves.
  * **the atom correspondence.** The gate matched atoms by POSITION while our
    featuriser emits 574 atoms and ESMFold2's 573 -- ours carries the terminal
    OXT. Now matched by name per token, all 573; the number did not move, so the
    old alignment was right by luck, and `ref_pos` differing by mean 3.31 A is a
    local-FRAME difference the reference is insensitive to (swapping our
    conformer in moves its `r_update` by corr 0.999984).
  * **every atom-block parameter**: q/k/v out of the fused qkv, the attention
    gate, both transitions, the 6-way split of the fused adaLN modulation, and
    the key-side modulation -- all bit-equal to the reference's tree. The q
    projection's bias (which the reference does not have) is zero. The atom
    pair-logits projection, which ESMFold2 has no weight for, is zero.
  * **the attention scale** (per-head 32 on both sides), **the QK RMSNorm**
    (present both sides, same `finfo(float32).eps`), and **the RoPE tables**
    (same formula and bases).
  * **the sliding window.** ESMFold2's is +/-64 by rank over the whole atom
    list; ours restricts it to AF3's query/key subsets, and the arithmetic looked
    fatal (32 queries x +/-64 spans 160 > 128 keys) -- but esmfold2's subset is
    192 keys and a direct count finds **0** in-window partners missing.
  * **the discrete atom features**: `ref_element`, `ref_charge` and
    `ref_space_uid` identical atom for atom, and the fused 389-column
    `atom_linear` splits into our per-feature slots in the reference's own order
    `[ref_pos 3 | charge 1 | mask 1 | element 128 | atom_name 256]`.
  * **the atom-level conditioning**: ours against the reference's
    `c0 = LN(atom_features @ atom_linear)` reads **1.000000**.
  * **the trunk-single term and the coords projection**: ESMFold2 conditions its
    atom blocks on `c0` ALONE, and our `embed_trunk_single_cond` is exactly zero;
    its `coords_linear` takes `[r_noisy | zeros]` and our converter maps only the
    first three columns.
  * **`ref_pos`, properly this time.** It DOES differ -- mean 3.31 A name-matched,
    and it moves the rope table hard (cos corr 0.837, sin 0.610, entirely in the
    6 spatial pairs; the 10 uid pairs are bit-identical). Yet the reference's
    denoise barely moves: 0.0791 A. The reason is that **rotary encodes the phase
    DIFFERENCE**, so what reaches attention is `pos_i - pos_j` -- local geometry,
    which our CCD ideal conformer shares with ESMFold2's table even though the
    frame does not. A large table difference and a tiny output difference are
    consistent, and the check that says so was right where the intuition was
    wrong.
  * **a frame**: the mean offset is 0.34 A and removing it leaves 1.295 A;
    rigid-body alignment leaves 1.287 A. Not a translation, not a rotation.

What it costs end to end is bounded and small: 6MRR 1.494 -> 1.493 best,
1.742 -> 1.732 mean, against native's 1.739. The error accumulates with atom
blocks (`NB=1/2/3` -> 0.588 / 0.836 / 1.341 A), which is the one positive clue:
it is inside the atom stack and it compounds.

## L5 for all 18, in one driver run, with the language models attached (2026-09-08)

`bash dev/oracles/run_all_parity.sh L5`. Every row 6MRR, 5 samples from seed 0,
CA-RMSD, and -- this is the part that had never been true of an in-repo sweep --
each model given the language-model input it actually needs.

| model | best | mean | native, same target |
|---|---|---|---|
| `boltz2` | **0.423** | 0.529 | |
| `alphafold3` | 0.628 | 0.688 | |
| `protenix2` | 0.691 | 1.330 | |
| `esmfold2_exp` | 0.727 | 1.051 | 0.736 |
| `opendde` | 0.772 | 0.859 | |
| `esmfold2_lm600m` | 0.788 | 1.428 | 0.794 |
| `rosettafold3` | 0.942 | 1.514 | |
| `esmfold2_fast` | 1.243 | 1.699 | 1.646 |
| `esmfold2_exp_fast` | 1.264 | 1.529 | 1.546 |
| `esmfold2_exp_fast_cutoff2025` | 1.424 | 1.638 | 1.629 |
| `esmfold2` | 1.494 | 1.742 | 1.739 |
| `intellifold2` | 1.513 | 1.610 | |
| `openfold3` | 1.541 | 1.718 | |
| `esmfold2_exp_cutoff2025` | 1.609 | 1.673 | 1.607 |
| `openbind0` | 1.648 | 1.825 | |
| `protenix1` | 1.694 | 1.836 | |
| `chai1` | 1.723 | 1.789 | |
| `esmfold2_lm300m` | 1.752 | 1.763 | 1.687 |

**Ours matches or beats native on 7 of the 8 ESMFold2 releases** (the exception
is lm300m, 1.752 against 1.687), which is the strongest end-to-end statement in
this file: eight releases, each against its own weights, its own shim and its
own ESM-C tower. The native column comes from
`dev/oracles/esmfold2_oracle_6mrr.py`, which now takes MODEL and dumps per
variant.

**The language-model inputs are the reason this sweep means anything.** chai-1's
token stream is mostly ESM2 and ESMFold2 has no MSA at all, so without them nine
of the eighteen rows are a different model: chai1 reads 3.9 A on 1STP+BTN
without embeddings against 0.456 with them. `dev/oracles/lm_inputs.py` generates
them per (tower, case) -- sequences taken from `modality_check.py --dump_seqs`,
so they come from the same chain construction the fold uses, in chain order --
and the driver records in each log WHICH file it used, or that none was found.
A number measured without one must not be mistakable for a number measured with
one, and that mistake had already cost an hour chasing a phantom chai1
regression.

Two calibration notes for reading any of this:

  * **6MRR is insensitive to chai-1's ESM2** (1.712 with, 1.704 without) while
    1STP is transformed by it (0.456 against 3.9). A designed helical bundle
    carries little evolutionary signal; a natural protein does. So the 6MRR
    column understates what the LM is worth.
  * the spread within one model across 5 samples is often larger than the
    spread between models -- `protenix2` runs 0.691 to 1.641 -- so `best` and
    `mean` are both given and neither alone should be quoted.

## L6 for all 18 models, in one driver run (2026-09-08)

126 cells, every one measured. `best` per case; the ligand column is BTN's
in-frame RMSD after aligning on the protein, the nucleic columns C1', the rest
CA.

| model | complex_1lmb | dna_1lmb | ligand_1stp | plain_5k9p | 6mrr | ptm_5k9p | rna_1ehz |
|---|---|---|---|---|---|---|---|
| `alphafold3` | 17.704 | 2.022 | 0.450 | 1.527 | 0.628 | 1.623 | 1.409 |
| `boltz2` | **0.387** | 1.534 | 0.458 | 1.714 | **0.423** | 1.815 | 1.197 |
| `chai1` | **0.509** | 1.863 | 0.535 | 1.526 | 1.723 | 1.804 | 1.506 |
| `intellifold2` | 12.155 | 1.585 | **0.441** | 1.669 | 1.512 | 1.554 | 1.469 |
| `openbind0` | 16.711 | 2.206 | **0.426** | 10.388 | 1.650 | 11.542 | 1.497 |
| `opendde` | 17.855 | 1.987 | 0.876 | 1.794 | 0.769 | 1.811 | 1.326 |
| `openfold3` | 12.697 | 1.916 | 0.456 | 1.388 | 1.540 | 1.499 | 1.331 |
| `protenix1` | 10.332 | 1.695 | 0.936 | 10.983 | 1.696 | 2.085 | 1.801 |
| `protenix2` | 17.446 | 2.080 | 1.199 | 7.458 | 0.685 | 7.921 | 1.759 |
| `rosettafold3` | 1.410 | 2.443 | 0.451 | 1.574 | 0.942 | 1.805 | **1.047** |

(the eight esmfold2 rows are being re-measured after the atom-decoder fix; the
pre-fix set is in the driver's own summary.tsv)

Four things this says that no single-model run could:

  * **the 4-chain complex is where the ports diverge most.** `boltz2` 0.387,
    `chai1` 0.509 and `rosettafold3` 1.410 get it; everything else lands at
    10-25 A, `alphafold3` itself included at 17.7. Scored in ONE frame, so a
    correct-but-misplaced chain fails -- which is the point of scoring it that
    way, and why the per-chain numbers in the older table read fine.
  * **three models fail on ubiquitin and their families do not.** `openbind0`
    10.388 against `openfold3`'s 1.388 on the same architecture, and
    `protenix1` 10.983 / `protenix2` 7.458. The PTM column tracks the plain one
    in every case, so it is the target and not the modification --
    [[protenix2-5k9p-retraction]] already covers protenix2's; openbind0's is
    open.
  * **ligands are uniformly good** -- eight of ten under 0.94, four under 0.46 --
    which is the strongest cross-model row here.
  * **RNA is uniformly good for the AF3 family** (1.05-1.80) and hopeless for
    esmfold2 (16-26 A), which is a competence limit rather than a port fault;
    see below.

## ESMFold2 is NOT protein-only, and this document said it was (2026-09-08)

Both tables here read `n/a -- protein only` for the esmfold2 family at L6. That
is wrong, and it was never measured -- it was inferred from `prepare_protein_features`,
the HF helper that takes a sequence, and from ESMFold2 having no MSA.

The model itself is not protein-only. Its `s_inputs` is 451 wide with a 33-class
restype block (nucleotides included), and `modeling_esmfold2` carries a
`_NONPOLYMER_ID = 4` branch on `mol_type`. What is absent from the shipped
`transformers` release is `ESMFold2InputBuilder`, the vendor's own featuriser for
"multi-chain / ligand / MSA inputs" -- its own docstring points at it. Our
featuriser supplies those atom features anyway, so the model can be fed them.

Measured, on the same cases every other port runs:

| esmfold2 | result |
|---|---|
| `ligand_1stp` BTN | **0.645 Å** |
| `dna_1lmb` duplex (both strands, one frame) | **2.908 Å** best / 3.221 mean |
| `plain_5k9p` protein | 1.233 best / 1.516 mean |
| `protein_6mrr` | 1.482 best |
| `rna_1ehz` (tRNA) | **25.008 Å** |

**Ligands are good** -- 0.645 Å on BTN beats several AF3-family ports -- and a
DNA duplex folds. **tRNA does not**, and the DNA result is what makes that
interpretable: the nucleic path works, so 25 Å is not a dead feature. A B-form
duplex is nearly a fixed local geometry; tRNA is a tertiary fold, and ESMFold2
has no MSA and a protein-trained language model, so it has no evolutionary
signal for RNA tertiary structure at all. Recorded as a competence limit rather
than a port bug -- and explicitly NOT settled, because settling it needs native
ESMFold2 on the same input and the vendor's non-protein featuriser is not in
this release.

**And it could not be given its language model on ANY modified residue.**
`_attach_lm_pair` keyed the pair rep on the count of protein TOKENS, but a
language model reads a sequence, so its rows are RESIDUES -- and AF3 atomises a
modified residue into one token per atom, all protein. `ptm_5k9p` raised
`lm_pair is (76, 76) but the batch has 85 tokens (85 of them protein)`. Now
mapped by residue, the same parent-residue convention AF3 uses for an atomised
residue's restype.

## The matrix, as the driver reports it (2026-09-08)

`bash dev/oracles/run_all_parity.sh` over all 18 models and the L0-L4 gates,
after everything below landed:

    OK    87        SKIP  132        WARN  0        FAIL  0        (219 cells)

**Zero WARN and zero FAIL is the claim worth checking, not the 87.** Every
non-OK cell is a SKIP -- a gate with no adapter for that model, which the gate
itself says. The interesting number is how many cells are EMPTY, and the answer
is that 132 of 219 have no oracle: chai1 has no callable native module for most
of them, `alphafold3` is the reference implementation, and the eight esmfold2
releases have no vendor to compare against for the diffusion path. The driver's
`summary.tsv` is the honest version of the coverage tables in this file, and
where they disagree it is the tables that are stale.

**Its first full run found four real port bugs**, every one in a (gate, model)
pair that existed but had never been run together -- protenix1's window
convention, chai1's template bias, chai1's diffusion s_inputs, boltz2's cyclic
conditioning. Three of the four moved no fold at all. That is what the L0-L6
table cannot tell you: it says how far down each MODEL goes, never which cells
were actually filled.

## L0 across all 18: two more real omissions, and the audit's own blind spots

Running L0 for every model (rather than the ones someone thought to check)
turned up 4 unaccounted tensors for opendde and 35 for boltz2. Most were dead,
but **one of boltz2's was a feature we had half-ported**:

**boltz2's cyclic conditioning.** boltz adds a fourth term to its single track
beside method / modified / mol_type (`trunkv2.py:202`):

    cyclic = feats['cyclic_period'].clamp(max=1.0).unsqueeze(-1)
    s = s + self.cyclic_conditioning_init(cyclic)

a trained `Linear(1 -> 384)` (absmax 0.446) over a 0/1 FLAG, not over the
period. Our cyclic support had gone into the relative-position WRAP alone --
which is AF3's own mechanism and shared by every model -- so a cyclic input
reached boltz2's relpos and never its single track. boltz2 carries both, and we
had one.

Verified both directions. With nothing cyclic the term is exactly zero and the
folds are bit-identical (6MRR best 0.424 / mean 0.537; `ligand_1stp` 0.277
protein / 0.458 BTN -- the same digits as before the change). With a cyclic
chain, `s_inputs` moves by **max|d| 0.44617**, which is the weight's own absmax
to five decimals -- exactly what a unit flag times W must give -- and its rms
goes 0.300 -> 0.342.

**The dead ones, each checked rather than waved through.** 31 of boltz2's 35 are
`*_proj_z.{i}.0.bias`: `Sequential(LayerNorm, Linear)` feeding a per-block pair
BIAS, so the offset is one constant per head on every logit and softmax is
invariant to it -- float64 softmax max|d| 1.1e-15. Plus the B-factor head (which
this port does not run) and PAE/PDE bin edges. opendde's are bin edges too.

**And two of opendde's four are a blind spot in the AUDIT, not a fact about the
conversion** -- worth writing down because the audit is now a gate:

`linear_no_bias_f` (128, 385) IS consumed: `atom_encoder` splits it by column
into `embed_ref_mask` (1), `embed_ref_element` (128) and `embed_ref_atom_name`
(256), folding the element index shift into the middle slice.
`dev/audit_coverage.py` cannot see that, for two independent reasons:

  * **a scanned dict.** The watcher records `sd[key]` and `key in sd`; this
    converter builds a stripped sub-dict, so `.items()` bypasses name tracking
    entirely. Only 4 of opendde's 4482 tensors report unaccounted because the
    VALUE fallback catches the rest.
  * **a fused tensor split across leaves.** The value fallback compares whole
    leaves of the same SIZE, which a 385-column tensor split into 1/128/256
    never matches -- doubly so where one slice is transformed.

Both are declared in `DEAD_TENSORS` with that reason, so the gate reads clean
and the limitation is recorded where the next reader will meet it. **All 18
models now either audit clean or say why they cannot** -- `alphafold3` is the
reference implementation and has no conversion to audit.

## The driver paid for itself on its first full run: protenix1 (2026-09-08)

`run_all_parity.sh` asks every gate about every model. That is how protenix1's
atom encoder came to be run for the first time -- the L2/L3 write-ups above
cover "openfold3, openbind0, intellifold2 and rosettafold3", and protenix1 was
simply never in the list. It was not clean:

| protenix1 gate | before | after |
|---|---|---|
| L1 trunk / L1b MSA / template / distogram | 1.000000 | unchanged |
| L2 token diffusion transformer | 1.000000 | unchanged |
| L2 diffusion conditioning | 1.000000 | unchanged |
| L2 atom encoder `c_atom_cond` | 1.000000 | unchanged |
| L2 atom encoder `p_pair_valid` | **0.867785** | **1.000000** |
| L2 atom encoder `a_token` | 0.997238 | **1.000000** |
| L2 atom decoder `r_update` | 0.996954 | **1.000000** |
| **L3 denoise step** | **1.1635 A/atom** | **0.0000 A** (max 0.0003) |

**The bug: a convention set per MODEL that belongs to the FAMILY.**
`model_registry`'s featurise knobs had `'protenix2': dict(padded_keys=True)` and
nothing for protenix1 -- so protenix1 SLID its atom key window where native
pads. But one `protenix/model/modules/primitives.py` serves every protenix
release: the padded window is a property of the implementation, not of a
checkpoint. Now `**{m: dict(padded_keys=True) for m in PROTENIX_FAMILY}`.

This is the same lesson as the template outer residual, in the same file, two
sections apart: **per-vendor conventions must be NAMED by family, or the next
release of that vendor silently gets AF3's default.** Adding protenix1 to
`KEY_MASKED_ATOM_ATTENTION` had already been done -- and was inert, because a
model that does not pad has no padded keys to mask. Half a convention is not
half a fix.

**The DIAG signature is what a window-convention bug looks like**, and it is
worth memorising because three gates read normal while this one did not:

    per-atom mean 1.1635 A, max 15.99
    window edge / interior           0.93     <- NOT the window edges
    ends [<128, >=446] / interior    3.64     <- the chain ENDS
    final partial window (>=544)     2.93 vs 1.07 elsewhere
    per-token spread                 0.100 / 0.516 / 7.528 (min/median/max)
    worst tokens                     61, 3, 66, 4, 62 of 68 -- both termini

Sliding and padding agree everywhere except where the window runs off the end,
so the error concentrates at the sequence ends and in the last partial window --
NOT at the 32-atom window edges, which is the diagnostic most likely to be
reached for. (And that edge/interior ratio is only trustworthy since the DIAG
fix earlier the same day; it used to compute the window position from the masked
atom list.)

**The folds do not move, and that is the point of having activation gates.**
6MRR 1.704 -> 1.705 best (mean 1.844 either way), `ligand_1stp` 1.867 protein /
0.943 BTN, `rna_1ehz` 1.824, `complex_1lmb` 10.198/10.161 protein and
1.192/1.182 DNA -- the last in line with protenix2's 11.763/11.669 and
openfold3's 11.347/11.308, so no regression. A 1.16 A/atom error in a single
denoise step left the end-to-end structure where it was; only the module
comparison could see it.

## chai1's first L0 gate, and the two bugs it found (2026-09-08)

`dev/audit_coverage.py` had never run on chai1. Not an oversight of priority --
a shape mismatch: every other converter takes one flat state dict, and chai
publishes five TorchScript archives, so `load_chai1` returns
`{component: state_dict}` and the audit's `_Watched` wrapper had nothing to wrap.
The audit now detects that shape (all values are dicts) and watches each
component, reporting keys as `component.key`.

**1912 tensors, 5 unaccounted for, and the check that sorted them was the
GRAPH.** A tensor a vendor ships is not necessarily a tensor a vendor uses, and
the only way to tell is to ask whether its `forward_*` methods reference it:

    torch.jit.load(...)._c._get_method('forward_256').graph

Three did not appear and are genuinely dead. Two did, and both were real.

### The template feature bias

chai embeds template features with ONE fused `Linear(76 -> 64, bias=True)`
(`input_projs.TEMPLATES.0`). AF3 uses nine bias-free Linears whose outputs are
summed, so `converters/chai1.py` splits the WEIGHT across AF3's slots -- and the
bias had nowhere to go.

It does not cancel. The 64-d activation goes straight into the template
pairformer, whose LayerNorms normalise per position across channels, so a
constant vector added at every (i, j) changes each position's normalised
direction. That is the distinction worth carrying: rosettafold3's and esmfold2's
dropped LayerNorm offsets DO vanish, because they land inside a softmax over j.
Where the constant lands decides it, not how big it is.

**How it was sized, with no native module to compare against.**
`template_parity.py` gained `ZERO=<scope>/<leaf>`, which reruns OUR module with
one parameter zeroed and reports the difference. On chai1's self-template 5K9P:

    |param| max 0.3005 -> output max|d| 0.91595   rms(out) 2.5759
    relative 0.3556                               corr 0.999648

**35.6% relative at corr 0.999648.** A 0.999 gate passes that, which is the
third instance in this file of [[correlation-hides-bias]] and the reason the
method exists: when there is no oracle, ablate the parameter and measure what it
was worth.

### The diffusion module's own s_inputs

chai's `TokenInputEmbedding` returns THREE tensors, and its graph says so
explicitly -- `TupleConstruct(%s_trunk, %s_structure, %z_init)`:

    input13     = cat[pooled_atom_single, token_single_input_feats]   # 768
    s_structure = token_single_proj_in_structure(input13)             # 384
    s_trunk     = token_single_proj_in_trunk(input13)                 # 384

Two independently trained projections of the same input. AF3 computes one
`s_inputs` and hands it to both consumers, so the port fed the diffusion
conditioning the trunk's vector. The two matrices are effectively ORTHOGONAL --
cosine -0.0024, rms 0.107 against 0.182 -- so this was not a near-miss, it was
an unrelated 384-d vector. Gated by
`model_config.SEPARATE_STRUCTURE_TARGET_FEAT`; `diff_emb['target_feat']` was
already a seam, since opendde's structural path replaces it.

**Which consumer gets which was MEASURED, not inferred, and the first attempt at
inferring it failed.** chai has three consumers of a token single and three
differently-named inputs -- the trunk's `token_single_trunk_initial_repr`, the
diffusion module's `token_single_initial_repr`, and the confidence head's
`token_single_input_repr` -- and on 6MRR those last two are mutually
uncorrelated (corr -0.009, rms 0.805 vs 1.512), so they are not the same
tensor. Elimination said "the diffusion module must take the structure one",
but the diffusion capture's per-channel profile tracked NEITHER weight's row
norms (-0.02 / +0.01), which is not what a plain projection looks like. Both
weights' row norms have a comparable coefficient of variation (0.12 / 0.15), so
that null was real rather than a dead statistic.

Settled by building our own `s_cat` for the same target (our input embedder
reproduces chai's s_init at corr 0.99999280) and projecting it both ways:

| chai consumer | vs trunk projection | vs structure projection |
|---|---|---|
| confidence head (per-channel std vs row norms) | **+0.744** | -0.075 |
| diffusion module (activation, 6MRR) | -0.013 | **+0.682** |

So: trunk <- trunk, confidence <- trunk, diffusion <- structure. The 0.682
rather than 0.999 is our `s_cat` not being chai's `input13` byte for byte
(different ESM2 source, our featurisation, 68 tokens against a padded 256); the
DISCRIMINATION is what the test needed, and -0.013 against +0.682 is not close.

**And its fold-level effect is nearly nil, which is worth stating plainly.**
6MRR 1.721 -> 1.704, 1STP+BTN (with ESM) 0.467/0.534 -> 0.456/0.535 -- and
ZEROING the structure projection outright moves 6MRR by 0.004 A. chai's
diffusion conditioning is almost insensitive to `s_inputs` because the trunk
single already carries that information. The fix is correctness; there is no
number behind it, and pretending otherwise would misprice the next one.

### A trap this turned up: every in-repo chai1 number is a NO-ESM number

chai1's token stream is mostly ESM2, and the harnesses supply it only when
`ESM_EMB` names an npz. Without it `fold_check.py` and `modality_check.py` fold
**a different model**: 1STP+BTN reads 3.9 A protein / 1.7 A ligand where the
recorded figure is 0.335 / 0.993, and with ESM2 embeddings the same code reads
0.456 / 0.535. The 6MRR 1.718 in the fold table is likewise a no-ESM number,
against chai's own 0.642.

That cost a wrong lead here -- the 3.9 A was read as a regression from converter
drift, and a blob built with both fixes NEUTRALISED (structure projection set to
the trunk's, template bias zeroed) is what showed it was not: 3.891 with the old
behaviour, 3.926 with the new. Rebuilding the old behaviour inside the new
converter is the cheap way to separate "my change" from "everything else that
changed since the recorded number".

## L0 on the other models: five benign findings, each verified not waved through

The same sweep ran L0 across all 18 models. Everything it turned up outside
chai1 was benign, but "benign" was established rather than assumed:

**OF3 checkpoints ship the diffusion module TWICE.** 740 tensors under
`diffusion_module.` and byte-identical twins under
`sample_diffusion.diffusion_module.`, verified key by key with
`np.array_equal`, in the UPSTREAM file -- 571.53M elements of which 203.24M are
duplication. We convert one copy: 368.39M parameters against 368.29M unique.
Recorded with a warning, because `map_diffusion_head`'s guard reads the
un-prefixed name and RETURNS EARLY if it misses -- a release that kept only the
prefixed copy would map no diffusion head at all, silently.

**rosettafold3 drops 33 `attention_pair_bias.ln_0.bias`, and that is correct.**
Verified rather than argued by analogy: the offset contributes `offset @ to_b`,
one constant per head on every logit, and softmax is invariant to a constant per
row. In float64 the logit shift has std 2e-16 across (i, j) -- it IS a single
constant -- and the softmax changes by 2.8e-16, with offsets up to 0.496. Worth
measuring precisely because the magnitude invites the opposite conclusion.

**esmfold2's shim is a SECOND artifact, and the audit only saw the first.** The
conversion writes the blob from `map_esmfold2_to_af3_graph` AND `<model>.lm.npz`
from `language_model_shim`; auditing the blob mapper alone called the shim's 10
`language_model.base_z_*` tensors unaccounted. Converters now declare extra
artifacts in `AUDIT_EXTRA`. This is the two-directions point again
([[converter-coverage-audit]]): the question is whether every tensor reaches SOME
artifact, not the one artifact the script happens to build.

**And L0 for the base `esmfold2` was a FileNotFoundError, not a result** -- its
weights are hub-only, never pulled to `~/esmfold2_variants`. The audit now reads
the hub cache off disk, by glob: `huggingface_hub` is not in the GPU venv and
must not be installed into it.

Plus `confidence_head.{lower,upper}_bins` (protenix) and
`atom_distance_v_bins` (chai) -- bin EDGES, registered buffers rather than
weights, derived on our side from the configured bin count.

**All 18 models now either audit clean or say why they cannot.** A model with no
converter entry -- `alphafold3`, which IS the reference implementation -- says so
instead of raising `KeyError`.

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
| ~~template embedder~~ | 9 | **8** | gated 2026-09-08, found THREE bugs (two protenix, one chai1); the 9th, chai1, has no callable native module, but its parameters now audit clean and a `ZERO=<param>` ablation sizes them |
| **MSA module** | 12 | **11** | every model but chai1 (TorchScript, no callable submodule); all 11 exact (boltz2 was 0.974 -- a real bug, now fixed). The five `msa=0` esmfold2 rows are n/a, not ungated |
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

## chai1's first in-repo module gate: L4 by injection (2026-09-09)

chai1 had L0, L5 and L6 and nothing in between -- every module cell a SKIP,
because its modules ship inside TorchScript archives with no callable submodule
`forward`. But its confidence head's verbatim I/O was captured during the port
(nine input tensors, three output LOGIT tensors), and a capture needs no torch,
no trunk, no diffusion and no featurisation agreement:

    dev/oracles/chai1_confidence_parity.py
      pae_logits  corr 0.999938  rms ours/native 1.0001
      pde_logits  corr 0.999916  rms ours/native 1.0001

**Logits, not the derived pLDDT/PAE**, so no assumption about chai's bin centres
enters the gate -- the trap in [[confidence-heads-status]], where a head emitting
logits under a score key reads plausible and is wrong. Only PAE and PDE: they
are per TOKEN PAIR and need no atom-layout agreement, where pLDDT is per atom
and would.

Two things it took to get right, both worth keeping:

  * **`atom_name_chars` is REQUIRED for chai1.** `confidence_head.py:521` keys
    its 37-slot pLDDT gather on that argument being present; omit it and the
    generic path builds a 24-slot projection the blob cannot fill
    (`plddt_logits/weights` (384, 37, 50) against (384, 24, 50)). chai's 37 is
    ATOM37 padding for the largest residue.
  * **the PDE half.** AF3 emits the LEFT half and symmetrises
    (`left + swapaxes(left)`); chai computes `pde_projection(LN(z) + LN(z)^T)`,
    the same function. Comparing our half against its full matrix read corr
    0.963 with **rms ours/native 0.5019** -- and a ratio of exactly one half is
    the tell, not a finding. Symmetrised, 0.999916.

The residual few percent is bf16: chai's head runs in bfloat16 natively and the
capture is stored that way.

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

`dev/` is TRACKED (only its `.npz` dumps, logs and bytecode are ignored), and
`bash dev/oracles/run_all_parity.sh` re-runs every gate below with the right
vendor overlay and precision, resumably and serially. Each is still listed with
what it covers, because the table is what says whether a green run is a green
MATRIX -- the driver's own summary should agree with it, and if the two
disagree, this table is the stale one:

| gate | level | covers |
|---|---|---|
| **`dev/oracles/run_all_parity.sh`** | **all** | **the driver: every gate below, every model, resumable, serial** |
| `dev/oracles/trunk_parity.py` | L1 | pairformer stack vs the vendor's module — 7 models |
| `dev/oracles/prot_parity.py` | L1b | protenix trunk AND MSA module (protenix2, protenix1) |
| `dev/oracles/msa_parity.py` + `esmfold2_msa_dump.py` | L1b | MSA module vs the vendor's own — rf3, both of3, intellifold2, opendde, boltz2, all three MSA-bearing esmfold2 releases. `LAYER=1` splits one boltz2 layer; `NONUNIFORM=1` runs a non-trivial msa mask |
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
