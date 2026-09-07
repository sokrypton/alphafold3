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

`✓` gated, `·` not measured, `n/a` no vendor to compare against.

| model | L0 | L1 pairformer | L2 diff-cond | L3 denoise | L4 conf | L5 fold | L6 modality |
|---|---|---|---|---|---|---|---|
| `alphafold3` | n/a | n/a | n/a | n/a | n/a | ✓ | · |
| `openfold3` | ✓ | ✓ | · | · | · | ✓ | ✓ |
| `openbind0` | ✓ | ✓ | · | · | · | ✓ | · |
| `intellifold2` | ✓ | ✓ | · | · | · | ✓ | ✓ |
| `protenix2` | ✓ | ✓ | · | · | · | ✓ | ✓ |
| `protenix05` | ✓ | ✓ | · | · | · | ✓ | · |
| `protenix1` | ✓ | ✓ | · | · | · | ✓ | · |
| `protenix1_20250630` | ✓ | ✓ | · | · | · | ✓ | · |
| `protenix_mini` | ✓ | ✓ | · | · | · | ✓ | · |
| `protenix_tiny` | ✓ | ✓ | · | · | · | ✓ | · |
| `boltz2` | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ |
| `opendde` | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ |
| `rosettafold3` | ✓ | ✓ | · | · | · | ✓ | ✓ |
| `chai1` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `esmfold2` family | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | n/a — protein only |
| `af2_ptm` / `af2_multimer` | n/a | n/a | n/a | n/a | n/a | ✓ | n/a — protein only |

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
the confidence head. For `openfold3`, `intellifold2`, `protenix2` and
`rosettafold3` none of those fifteen has an activation-level check.

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
| RoseTTAFold3 | **no** | — |

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

**2. L2–L4 for the four trunk-only models** — `openfold3`, `intellifold2`,
`protenix2`, `rosettafold3`. These need new oracles, and they are the four whose
ports predate the injection-ladder method (dump native's own tensors, inject
them, compare our module's output). `rosettafold3` is blocked: no native
installed. The recipe to copy is `esmfold2`'s, which is the most completely
gated model here.

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
