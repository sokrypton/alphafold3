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

`~` at L2 is the **token transformer only**: ten models run their own vendor's
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
