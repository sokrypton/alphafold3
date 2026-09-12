# What is left for 100% parity

TWO categories, and the second was invisible until `parity_audit.py` existed:

  1. **HOLES** -- cells that never ran. `gate_applies.py` says which of those
     are real (the model has that module and no other cell covers it).
  2. **cells that ran and DISAGREE.** `run_all_parity.sh`'s `classify()` only
     asks whether a matching line exists, never what it says, so a comparison
     at corr 0.9678 is reported OK. `parity_audit.py` grades on corr AND
     max|d|/rms.

# Category 1: HOLES -- none left

| gate | models | what it needs |
|---|---|---|
| -- | -- | NONE. Every gate now has an adapter for every model the cell applies to. |
| `L4.confidence` | esmfold2, esmfold2_fast | NO LONGER A HOLE -- the cell RUNS, dump-driven, and its numbers are a finding rather than a gap. See below. |

Everything else is covered: L0, L1.trunk, L1i.trunk_init, L1t.template,
L2.conditioning, L2.atom_encoder, L2.atom_decoder, L2.diffusion, L3.denoise all
have an adapter for every model the cell applies to. (`alphafold3` is the
reference implementation; `chai1` ships TorchScript archives with no callable
submodule forward -- those are n/a by construction, not gaps.)

# Category 2: cells that ran and disagreed -- all closed

Eight PORT bugs, found in cells that had no gate before 2026-09-09:

| model | bug | after |
|---|---|---|
| opendde | the diffusion atom pair ran on ZERO weights (4 of 5 terms): the converter asserted which of a `X`/`X_1` haiku pair was live, and the forward changed under it | denoise 0.464 -> 0.0054 A/atom |
| boltz2 | `arcsinh(charge)` where it takes the RAW formal charge | c_atom_cond exact |
| boltz2 | slid the atom key window where it CLIPS AND PADS | a_token -> 1.000000 |
| boltz2 | padded keys not masked from real queries -- caught by a registry TEST, not a number | edge window 1.78 -> 0.18 |
| boltz2 | relative-CHAIN bucket keyed on entity, not chain | trunk loop exact through 4 passes |
| boltz2 | atom-pair offset is KEYS minus QUERIES, uniquely in the panel | whole atom path -> 1.000000 |
| rosettafold3 | slid the key window where it CLAMPS AND MASKS | a_token 5.7e-01 -> 1.11e-01 |
| rosettafold3 | `arcsinh(charge)` again, second model in two days | c_atom_cond 5.11e-02 -> 2.50e-06 |

And EIGHT ORACLE bugs, which is the half of the work that is easy to
under-report. Each had a distinguishing signature, and that is what to reuse:

| harness fault | how it was told apart |
|---|---|
| rf3's chirality term disabled on the native side while the port implements it | `ZERO_POS=1` made the whole encoder exact: the chirality signal is a gradient w.r.t. the NOISY COORDINATES, the only term that vanishes with them |
| if2's atom pair returned as None ("it windows the dense axis" -- true of the dense layout, false of the packed one the checkpoint runs) | once returned, `p_pair_valid` per window was exact everywhere while q blew up on windows 16-17 alone |
| boltz2 fed the DENSE atom layout when its featuriser packs | `c_atom_cond`, which has no window, stayed at 0.999999 throughout |
| boltz2's `SingleConditioning` fed sigma instead of `c_noise(sigma)` | single_cond rms 0.4367 of native's with max\|d\| 6762 -- what a Fourier embedding does when its input is off by that much |
| if2's pairformer needs `v2=True` (`pre_norm_s`, not `attention.norm_s`) | 128 tensors missing in load_state_dict |
| if2's MSA one-hot built in OUR class order, not boltz's | s 0.9537 / z 0.9291 -- a permuted vocabulary where most columns still land somewhere plausible |
| `is_paired` zeroed when boltz2 marks the QUERY ROW | s 0.974 -> 1.000000 on its own |
| BLOCKS truncating only OUR side (rf3), and `_truncate_atom_blocks` slicing the wrong axis for the per-block-LN family | the 1-block run came back WORSE than the 3-block one; haiku's scan caught the axis |

One more that is neither: **intellifold2's z_init read 2.06e-02 because our blob
stores its trunk in bfloat16** -- deliberately, mirroring AF3's own param dtype
policy, and measured fold-neutral where that policy is set. Rounding native to
the blob's dtype gives 3.79e-06. Only `alphafold3` and `intellifold2` store bf16
at all, so nothing else in the panel can hit it.

# boltz2's confidence head: the re-embedding is right, the stack is not

The last hole. Its converter has been structurally complete for a while (66
unported -> 0) and its three graph branches landed with it, but the VALUES had
never been compared -- the state [[boltz2-confidence-port]] recorded as
"structural coverage COMPLETE, values UNVERIFIED". The adapter builds boltz's own
`ConfidenceModule` and derives every flag from the checkpoint.

    full_pae   0.9198 / 1.88      plddt     0.9964 / 6.50e-02
    full_pde   0.9286 / 1.47      resolved  0.9975 / 5.50e-01

`BLOCKS=n` truncates the confidence pairformer on both sides, and that localises
it exactly:

    0 blocks   pae 0.999943 / 2.66e-02     pde 0.999697 / 9.97e-02
    1 block    pae 0.994351 / 2.14e-01     pde 0.991677 / 4.23e-01
    8 blocks   pae 0.919759 / 1.88e+00     pde 0.928585 / 1.47e+00

So the RE-EMBEDDING is right -- rel_pos, both bond terms, contact conditioning,
the two target-feat projections, the product term and the 64-bin distogram all
land within 2.7e-02 -- and the 8-block stack contributes ~2e-01 per block.

What that is NOT: our pairformer BLOCK. The same block reads 3.85e-04 on z in
the trunk gate at one block for this model, and boltz forces `v2=True` for both
stacks, so they are the same layer. So the difference is in what the CONFIDENCE
stack is given or how its params are mapped, not in the block arithmetic --
which is the next thing to measure, and it wants an injected single-block
comparison against `confidence_pairformer`'s own scope.

One oracle bug on the way, with a lesson worth more than the fix:
`bond_type_feature` is NOT in the checkpoint's `confidence_model_args`, so the
module defaulted it False and silently skipped its `token_bonds_type` term --
an nn.Embedding row 0 on every pair, a learned constant, not a no-op. It read
pae 0.333 / pde 0.293. The tell was there and I was not asserting on it:
`load_state_dict` returned `token_bonds_type.weight` as UNEXPECTED, because a
module that does not build a tensor cannot be missing it. Both flags are now
derived from the tensors (`token_bonds_type` present, and `token_bonds`'s input
width is 1 + maximum_bond_distance) and the gate asserts on unexpected too.

# The esmfold2 confidence head: two port bugs, then native's own bf16

The only esmfold2 module `esmfold2_reference` does not implement, so the only
one needing a native run. The dump hooks make it possible: 11 inputs and 14
outputs, with z, s_inputs and x_pred injected so both sides run on the tensors
native used.

TWO REAL BUGS, and the pattern in the numbers is what found each:

  1. THE SINGLE WAS POOLED FROM THE WRONG PAIR. ESMFold2 has no trunk single;
     its confidence head builds one by ROW-ATTENTION POOLING the pair -- and it
     pools the pair its FOLDING TRUNK HAS ALREADY UPDATED
     (`pair.add_(folding_trunk(pair)); single = row_attention_pooling(pair)`).
     Ours pooled the pre-stack pair. Every PAIR-derived output was close
     (pae 0.9947) and every PER-ATOM one uncorrelated, because plddt and
     resolved are the only heads that read the single:
         plddt     0.1059 -> 0.9883    resolved  0.3748 -> 0.9862
  2. IT DOES NOT SYMMETRISE ITS PDE. AF3 symmetrises the logits
     (`left + swap(left)`), protenix the pair inside the LayerNorm, boltz2 first
     and then splits by chain; ESMFold2 does none of it -- `pde_head(pde_ln(pair))`,
     its PAE head with different weights (`model_config.UNSYMMETRISED_PDE`).
     Ours came out 2.6x small, which is what summing a logit with its transpose
     does to an expectation:
         full_pde  0.9017 / rms 0.389 -> 0.9935 / rms 1.008

WHAT IS LEFT is native's own precision, traced rather than assumed. Its trunk
runs under `autocast(bfloat16)`, so the `relative_position_encoding` the dump
carries is a bf16 computation: recomputing it in fp32 from the SAME feature and
the SAME weight -- our converted `rel_pos_project` is the checkpoint's
`rel_pos.embed.weight` transposed to max|d| 0.000000 -- differs by 1.74e-02,
which the folding trunk then amplifies. Running OUR head in bf16 (`BF16=all`,
new knob) changes nothing, because rounding our weights is not the same as
reproducing native's accumulation.

    esmfold2       pae 0.9947  pde 0.9935  plddt 0.9883  resolved 0.9862
    esmfold2_fast  pae 0.9948  pde 0.9864  plddt 0.9684  resolved 0.9269

Three things say the setup is sound, so those numbers are a precision floor and
not a hidden gap: 573 of 574 atoms match BY NAME (the odd one is the terminal
OXT, absent from ESMFold2's atom table); the representative-atom assert passes
BIT-EXACTLY, ESMFold2's explicit `distogram_atom_idx` landing on the same
coordinates as our pseudo-beta gather; and native's plddt_logits reduce to its
own plddt_per_atom at corr 1.000000 through the gate's own 50-bin expectation.

# The gate that measures an amplifier

`L1.trunk` at FULL depth on synthetic input is not a port measurement. rf3's z
reads 5.8e-04 at one block, 5.0e-04 at four and 1.2e-01 at 48, where the single
track has grown to rms 2.7e4 while the pair track has FALLEN to 24 --
non-monotone, i.e. saturated far outside the trained input distribution. Read
1-4 blocks for the port and the full depth as a smoke test. The driver should
run a low-depth cell too; until it does, three models will keep reading BAD
there for a reason that is not a bug.

# The one question that is open and is not a hole

With boltz2's trunk provably exact through four recycle passes, something
DOWNSTREAM turns a correct pair representation into a worse structure on 6MRR
about 10% of the time (mean 0.700 against 0.540 over 40 samples, four of them
at 0.84-1.5, while at zero recycles the two conventions are level). See
`model_config.CHAIN_BUCKET_ON_SAME_CHAIN` for the full numbers and the knob.

Read `dev/oracles/gate_applies.py` first -- a cell is only a hole if the model
HAS that module and no other cell covers it.

## 2026-09-10, L1 closed: one more port bug and one more oracle bug

    PORT    rosettafold3  triangle multiplication divides by float(L) before the
            centre LayerNorm, so the term is observable only through the norm's
            epsilon. L1.trunk z 1.24e-01 -> 5.25e-03, corr 0.999989 -> 1.000000.
            Fold-neutral on real input (6MRR 0.892/1.507 vs 0.882/1.507; 1STP
            3.725 vs 3.722) -- the synthetic-input gate is what drives z into the
            regime where eps stops being negligible.
            Gated by model_config.TRIANGLE_MUL_DIVIDE_BY_LENGTH.

    ORACLE  intellifold2  the trunk gate loaded if2's deliberately bf16 blob
            against native's fp32 checkpoint, and reported the STORAGE dtype as
            a disagreement. z 6.98e-02 -> 3.77e-05 (1 block), 9.60e+00 ->
            8.96e-04 (48). SIGNATURE: the only model in the panel whose blob
            holds any bf16 at all, and a gate that turns bf16 off in both FORWARD
            passes while saying nothing about the weights.
            Fixed by rounding native to match; IF2_FP32_BLOB=1 and
            IF2_NO_BF16_WEIGHTS=1 keep both directions measurable.

Three cells that looked identical to those two were neither, and now say so:
openfold3, boltz2 and protenix2 read BAD/LOOSE on z at 48 blocks while a 1e-6
perturbation of the gate's own input moves its output FURTHER than our port does.
They grade FLOOR. `trunk_parity FLOOR=<eps>` is how a cell earns that, and rf3
failing to earn it is how its /L was found -- so the exemption discriminates.

## A gate that reports OK while testing nothing (2026-09-10)

`L1b.msa_nonuniform` runs `NONUNIFORM=1`, and that variable is read INSIDE
boltz2's adapter only (`msa_parity.py` native_boltz2). For the other thirteen
models the cell is a byte-identical duplicate of `L1b.msa`: same number, status
OK, nothing tested. It is not a hole by `gate_applies.py` -- the cell RAN -- and
that is what makes it worse than a hole.

It exists to separate two OPM normalisers that an all-ones mask cannot tell
apart, and there is a live question waiting on it:

  * **rosettafold3's OuterProductMean takes no mask argument at all** and divides
    by `float(N)`, the RAW row count (outer_product.py:29-37). AF3 masks the msa
    and divides by the pairwise valid count. Under padding those disagree: a
    padded row contributes nothing to rf3's outer product but still moves its
    divisor. Whether rf3's subsampler zeroes padded rows before the OPM sees
    them decides only the numerator, never the denominator.

So `OPM_ROW_COUNT_NORM` may well need rosettafold3 -- but that list currently has
NO code effect (it documents a disproved reading of boltz's source), so answering
this means building the nonuniform path for rf3 first, then deciding what the
divisor should count. Until then rf3's OPM is verified only for a uniform mask,
which is what the L1b cells feed.

The general lesson is the one [[harness-rot]] describes, one level in: a knob
implemented in one adapter and read by a cell for all fourteen models produces
thirteen silent passes.

## An existing gate the driver never runs (2026-09-10)

`dev/oracles/real_trunk_parity.py` is written, documented, has two native dumps
beside it (`native_trunk_5k9p_plain.npz`, `native_trunk_6mrr.npz`) -- and
appears nowhere in `run_all_parity.sh`. Its own docstring says what it covers
that nothing else does:

  * the INPUT EMBEDDER (`create_target_feat_embedding` -> s_inputs), which five
    oracles build and none compares;
  * the trunk's output on a REAL input rather than one block on noise;
  * the RECYCLING loop, since L1-L4 all measure a single pass.

This is not academic. The synthetic trunk cell is resolution-limited for the OF3
family: after one block openfold3's single track has rms 16750 from an
N(0, 0.5) input (openbind0 1958, against 6.8 for opendde), and at that magnitude
a 1e-6 input perturbation moves z further than our port does. Both cells now
grade FLOOR, which is honest but is not the same as verified -- and a real-input
gate is exactly what would verify them.

What it would take: `native_trunk_dump.sh` dumps PROTENIX's trunk specifically
(its own runner and featuriser, two class-level monkeypatches). Wiring the gate
in for protenix2 is a driver line. Covering openfold3 needs an of3 equivalent of
that dump script, which is real work -- and until it exists, `gate_applies.py`
should call the of3 cells HOLES rather than let the level look complete.

Read the control in its docstring before believing any number it produces: 48
blocks x 10 recycles amplify float differences, and on 6MRR -- where our fold
matches native at 0.70 A -- the pair still only correlates 0.960. That mistake
already cost most of a session once.

## boltz2's denoise: every part at parity, the whole at 0.32 A/atom (2026-09-10)

L3 closed for seven models (7.65e-06 to 8.63e-04 on x_denoised) and all four
ESMFold2 releases (exact, 0.00e+00 against the reference). boltz2 is the
exception, and it is TWO problems:

  1. **`L3.denoise` is a genuine HOLE for boltz2** -- `denoise_parity.py` has no
     boltz2 adapter, so the model that needs this cell most is the one model it
     does not cover. `gate_applies.py` says so ("applies to this model, no
     adapter"), which is the honest report.
  2. `boltz2_denoise_parity.py` is the substitute, injecting native's own
     conditioning, and it reads **x_denoised corr 0.999767, max|d|/rms 6.03e-02,
     per-atom distance mean 0.3248 A**.

It is REAL and it is resolvable -- perturbing the injected conditioning by 1e-6
moves our own output by 4.22e-06, four orders below the gap. And every piece of
the same path is at parity:

    L2.conditioning    pair 1.95e-06   single 1.39e-05
    L2.atom_encoder    a_token 1.0e-04   q_atom 3.6e-04   p_pair_valid 4.5e-06
    L2.atom_decoder    a_token 1.1e-04   q_atom 3.5e-04   r_update 1.8e-04
    L2.diffusion       a 1.88e-03, below its own 3.98e-03 floor
    L1i / L1.trunk1    z_init 2.77e-06, trunk 3.92e-04

So the parts are right and the COMPOSITION is not, which is the shape of a
wiring or scaling difference rather than a weight one. Two things already ruled
out by measurement:

  * NOT the extra C-terminal atom. Ours carries one atom boltz never sees
    (`per-token count differs at [(67, 10, 9)]`, the OXT) and it is dropped from
    the comparison but not from the model, so it still sits in our atom windows
    -- the ESMFold2 OXT failure exactly. That one showed as error rising over
    the final tokens; this profile is FLAT (8 bins 0.24-0.50, last-32 mean
    0.309 against a 0.322 median).
  * NOT today's work. Every src change of 2026-09-10 is behind a model-name or
    env gate that excludes boltz2, and `boltz2` sits in OPM_BIAS_AFTER_NORM
    identically before and after. The gate itself has not changed since dev/
    became tracked.

**UNEXPLAINED, and recorded as such:** [[boltz2-port]] carries 0.0269 A/atom for
this cell after the earlier round of fixes, and it now reads 0.3248 A. Today's
changes provably cannot have moved it, so either that figure came from a
different measurement or something moved earlier and unnoticed. Do not treat
0.0269 as a baseline to regress against until it can be reproduced.

**The gate's one blind spot to fix first:** the dump holds a SINGLE sampler step,
`times 1.4157 -> sigma 4608`, the highest-noise step of the schedule, where
c_in is 2.17e-04 and c_out is 16.0. So this compares almost pure network output
at the one operating point where the coordinates barely enter. A re-dump at a
mid-schedule sigma is the next measurement, and it needs the boltz venv.

## confidence_parity's `ours()` is not idempotent (2026-09-10)

Called twice in one process it builds a DIFFERENT head for some models. For
opendde the first call builds 52 scopes and maps all of them; the second builds
51 and cannot fill 66, and the names it cannot fill are stock-AF3
(`confidence_head/~_embed_features/left_target_feat_project/weights`, ...)
rather than opendde's own module's -- so the second build is the AF3 head, not
opendde's.

Found by adding FLOOR to the cell, which needs a second call. Consequences:

  * opendde and any model with a bespoke head cannot have its confidence FLOOR
    measured, so `L4.confidence` for opendde is graded on its number with no
    resolution estimate. The gate now PRINTS "FLOOR UNAVAILABLE ... ours() is
    not idempotent for this model" rather than dying or skipping silently.
  * nothing else in the matrix calls `ours()` twice, so no existing number is
    affected -- but any future two-call gate hits this.

The fix is to build the head ONCE and apply it to two sets of inputs (the
inputs are currently closed over inside `fwd()`), not to call `ours()` again.
Not done here: it is a refactor of the gate's core and the level was mid-run.

## RETRACTION: esmfold2's confidence gap is NOT our-side bf16 (2026-09-10)

`L4.confidence` for the two ESMFold2 releases with a confidence head reads

    esmfold2        full_pae 5.68e-01  full_pde 2.75e-01  plddt 7.83e-02  resolved 4.58e-02
    esmfold2_fast   full_pae 5.23e-01  full_pde 3.71e-01  plddt 1.84e-01  resolved 1.01e-01

and PARITY.md carried these as a characterised residual: "the two esmfold2
confidence heads at native's bf16 floor". `confidence_parity.ours()` even has a
BF16 knob for this family, with the reasoning written out -- ESMFold2's own head
casts under `autocast(bfloat16)` on a GPU and the dump was produced on one, so
an fp32 comparison would be comparing two precisions.

MEASURED, and it is not that:

    esmfold2       BF16=none  pae 5.68e-01   BF16=all  pae 5.62e-01
    esmfold2_fast  BF16=none  pae 5.23e-01   BF16=all  pae 5.17e-01

Running OUR head in bfloat16 moves the third digit. So whatever these rows are,
they are not our side's precision, and the driver was right not to set the knob
(it would have changed nothing while looking like diligence).

What is still POSSIBLE and untested: that NATIVE's own output is that noisy, i.e.
the DUMP is one draw from a bf16-wide band. That is a claim about native's
reproducibility and it cannot be tested from our side at all --
[[esm-tower-numerical-modes]] measured exactly this shape of thing before
(bimodal ~6e-4 across processes, "hid_*.npz is one draw; measure the band
first"). The test is two dumps from ~/venv_esm and the spread between them.
Until that exists, these four rows are UNEXPLAINED, not characterised, and
PARITY.md should not call them a bf16 floor.

## protenix's L4 gap: an exact stack amplifying a 3e-03 embedding difference

`L4.confidence` leaves the protenix lineage not at parity -- protenix2 pae
2.30e-02 / pde 1.72e-02, protenix1 5.4e-03 / 6.8e-03, opendde pae 9.8e-03 / pde
1.58e-02 -- and these are REAL: protenix2's FLOOR is 1.9e-06 to 6.0e-06.

TWO WRONG INFERENCES were made and corrected by measurement, both recorded
because the reasoning looked sound each time:

  1. "pde starts in the re-embedding, because BLOCKS=0 reads 4.40e-02, WORSE
     than 1.72e-02 at four blocks." No: at BLOCKS=0 full_pde's rms(native)
     collapses to 1.218 from 18.831, so that was a small denominator. In
     ABSOLUTE terms max|d| grows 0.0536 -> 0.323 with depth. Its p99.9 also
     sits 65x below max|d| there -- a few bin-boundary entries.
  2. "the stack itself differs, because the heads all grow with depth and the
     floor implies a gain of only 2-6x." No: isolated on the same synthetic
     input, protenix2's confidence pairformer is EXACT -- z 2.42e-05, s
     2.06e-06, corr 1.00000000, 0 unmapped. The floor knob was measuring the
     wrong sensitivity: it perturbs s_inputs/s/z, UPSTREAM of the embedding,
     which is a different path from the stack's own input.

Measured properly, the stack's GAIN on a perturbation of its own input pair is
6.9x to 8.8x, stable over three decades of eps:

    input rel 1.26e-05 -> output rel 1.11e-04   gain 8.8x
    input rel 1.25e-03 -> output rel 8.63e-03   gain 6.9x
    input rel 5.01e-03 -> output rel 3.46e-02   gain 6.9x

and protenix2's pae at four blocks (2.30e-02) is 7.6x its pae at BLOCKS=0
(3.03e-03) -- the gain, to within the measurement.

**So the whole of it lives in the BLOCKS=0 path and is ~3e-03, not 2.3e-02.**
That path is: `_embed_features` (left/right target-feat projections plus the
CB-CB dgram) and the four LN+Linear heads. p99.9 there is 4.0e-04 relative, so
whatever it is, it is small and broad. That is the next measurement, and the
cheap version of it is a direct comparison of the EMBEDDED pair before the
stack -- which no cell makes today, and which `ours()` cannot currently be
asked for (it returns only the four head outputs).

Note for whoever adds that: the L4 FLOOR knob's number is not the resolution of
the embedding path. Perturbing the head's inputs and perturbing the stack's
input measure different gains, and this cell has both.

### ...and the protenix L4 residual is probably the reference's own float32

Continuing the above. Every term of the BLOCKS=0 embedding path was checked by
reading, and all of it matches:

  * `linear_no_bias_s1` -> `left_target_feat_project` and `s2` -> `right`, which
    is the right way round: native adds s1 on the COLUMN axis
    (`[..., None, :, :]`) and s2 on the ROW axis (`[..., None, :]`), and ours
    adds `left` then `right[:, None]`, the same orientation.
  * both distance terms are present, including the unbinned
    `linear_no_bias_d_wo_onehot` that carries the sub-bin resolution.
  * the bin edges are identical: protenix's `arange(3.25, 52.0, 1.25)` is our
    `linspace(3.25, 50.75, 39)`, 39 lower edges plus a catch-all. Ours compares
    SQUARED distances against squared breaks, which is equivalent for positive
    distances.
  * the representative-atom positions are asserted bit-identical by the gate.
  * protenix does NOT mask its distance terms where we do (`dgram *= pair_mask`),
    but the gate feeds an all-ones mask, so that is inert HERE. It would matter
    on a padded input, and no cell covers that.

What is MEASURED:

    the confidence stack, isolated      z 2.42e-05, s 2.06e-06  (exact)
    its gain on its own input           6.9x - 8.8x, stable over 3 decades
    pae at 4 blocks / pae at 0 blocks   2.30e-02 / 3.03e-03 = 7.6x  (the gain)
    native's OWN fp32-vs-fp64 noise
      through the embedding arithmetic  max 4.36e-03, p99.9 6.33e-05

So our whole BLOCKS=0 gap (3.03e-03 max) is the same order as the reference's
own float32 noise in that path (4.36e-03 max), and the 4-block number is that
times an exact stack's gain.

**CONSISTENT WITH, NOT ESTABLISHED.** Those two numbers are measured on
different tensors (native's on z_embed at rms 8.16, ours on the pae expectation
at rms 14.76), so the orders line up but the comparison is not apples to apples.
The test that would settle it is native's whole head in fp32 against itself in
fp64, compared on the same pae expectation -- attempted, and it dies in
protenix's `primitives.LinearNoBias` with "expected m1 and m2 to have the same
dtype" because the head has inputs it does not convert. Plumbing, not physics,
but not done.

**REVISION of what this file said an hour earlier.** "protenix2 -> REAL, four
orders above its own floor" was read off the FLOOR knob, which perturbs
s_inputs/s/z upstream of the embedding and therefore measures the wrong gain.
Read against the arithmetic floor of the path instead, these rows are at or near
it. Treat them as precision-limited unless the fp64 test says otherwise.

## boltz2's L4 is the pair RE-EMBEDDING, at 2.66e-02 (2026-09-10)

The BLOCKS sweep says where it is not:

    BLOCKS=0   pae 2.66e-02   pde 9.97e-02   plddt 2.66e-07  (plddt EXACT)
    BLOCKS=1   pae 2.14e-01   pde 4.23e-01   plddt 9.96e-02
    BLOCKS=4   pae 5.97e-01   pde 1.24e+00   plddt 1.70e-01

plddt is exact with no pairformer because it never reads the pair; one block
mixes the pair into the single track and it inherits the error. So the whole of
L4 for boltz2 follows from the pair re-embedding being wrong by 2.66e-02, and
[[boltz2-confidence-port]]'s "re-embedding right at 2.66e-02" is exactly this
number. The "~2e-01 per block" in the same note is the amplification, not a
per-block bug.

THE STACK IS EXACT. Isolated on the same synthetic input, built the way the gate
builds it (through ConfidenceModule, whose PairformerModule differs from the one
you get by importing it directly -- see below): all 8 blocks, 0 unmapped,
z 3.59e-05, s 1.80e-05, corr 1.00000000.

ELIMINATED by reading, term by term against confidencev2.py:

  * every term is present in our `_boltz2_reembed`: s_inputs_norm, s_norm +
    s_input_to_s, z_norm, rel_pos, token_bonds, token_bonds_type,
    contact_conditioning, s_to_z / s_to_z_transpose, s_to_z_prod, distogram.
  * the s_to_z ORIENTATION is already right and already commented: boltz's
    `s_to_z(s)[:, :, None, :]` is indexed by i, which is our RIGHT projection,
    and `s_to_z_transpose` is our LEFT. The prod term's axes match too
    (in1 -> row, in2 -> column).
  * the distance bins match: boltz `linspace(2, max_dist=22, 63)` with
    `(d > boundaries).sum(-1)` into an nn.Embedding; ours the same 63 edges,
    one-hot into a Linear. Ours masks by pair_mask and boltz does not, which is
    inert on the all-ones mask the gate feeds (and live on a padded input --
    same unchecked case as protenix's distance terms).
  * the `contact_conditioning` PLACEHOLDER is faithful, which was worth
    checking because our featuriser has no distance-restraint field: boltz
    initialises `contact_threshold` to ZEROS and only writes `max_distance`
    where a constraint exists (featurizerv2.py:714), and its module drops the
    first two conditioning classes (UNSPECIFIED, UNSELECTED) before use. So an
    unconstrained input contributes nothing through either, which is what the
    placeholder supplies.
  * the three `add_*` flags are all True in this checkpoint, so native runs
    every branch we run.

NEXT MEASUREMENT: a term-by-term numerical diff of the re-embedding -- compute
native's z after `contact_conditioning`/`s_to_z`/`distogram` and ours, and
subtract. Everything above was established by READING, and every finding today
that survived came from a number instead.

TRAP, for whoever writes that probe: `from boltz.model.layers.pairformer import
PairformerModule` and `ConfidenceModule(...).pairformer_stack` are NOT the same
module here. Constructing it directly with the checkpoint's own
`pairformer_args` asks for `attention.norm_s` (16 tensors the checkpoint does
not have) while ConfidenceModule's build asks for `pre_norm_s` and loads clean.
Build it through ConfidenceModule, as the gate does. [[three-code-copies]].

## boltz2's template module is V2, and we implement V1 (2026-09-10)

Found by generalising the relative-position finding: read the CHECKPOINT's
flags, not the class defaults. boltz2's hyper_parameters carry
`use_templates_v2: True`, and `models/boltz2.py:231` selects `TemplateV2Module`
over `TemplateModule` on it. `template_parity.py:263` builds `TemplateModule`.

The two classes take the SAME weights -- both load the checkpoint's 81
`template_module.*` tensors with 0 missing and 0 unexpected -- so only the
forward differs, in exactly one place:

    v1   asym_mask = (asym_id[:, :, None] == asym_id[:, None, :])
         a_tij = a_tij * asym_mask...
    v2   tmlp_pair_mask = (visibility_ids[..., :, None] == visibility_ids[..., None, :])
         a_tij = a_tij * tmlp_pair_mask...

v1 masks the template pair features by SAME CHAIN; v2 masks by matching
`visibility_ids`, a per-template field. Our `Boltz2TemplateEmbedding._features`
takes `asym_mask_2d` -- the v1 convention.

**Inert on everything currently gated, live on a complex.** With one chain and
one template both masks are all-ones, which is why `L1t.template` reads 4e-05
for boltz2 and why this survived. It diverges as soon as a template covers more
than one chain, or when visibility_ids partition a chain.

NOT FIXED. `visibility_ids` is a boltz feature our featuriser does not produce,
so this is a featurisation job and not a one-line branch. Two things follow:

  * `L1t.template boltz2` is verified for the single-chain case ONLY, and the
    gate should build TemplateV2Module so that stops being invisible.
  * [[multimer-parity-status]] records boltz2 templates on a complex as
    "partial" -- this is a candidate explanation, and it is testable by giving
    the gate a two-chain template.

## esmfold2's confidence: the third rel_pos call site, and what is left

`create_relative_encoding`'s chain-bucket gate has THREE call sites -- trunk,
diffusion conditioning, confidence. The trunk's was fixed first (worth
1.522 -> 0.719 A on esmfold2_lm600m), the diffusion's after it, and the
confidence one was never wired: `confidence_head.py:117` called it without the
flag, so ESMFold2's confidence head built its relative-position block with AF3's
ENTITY convention while ESMFold2 keys it on same-CHAIN. Fixed 2026-09-11.

    esmfold2       pae 5.68e-01 -> 4.31e-01   pde 2.75e-01 -> 2.67e-01
    esmfold2_fast  pae 5.23e-01 -> 3.50e-01   pde 3.71e-01 -> 2.26e-01

Right on principle and a real improvement, but NOT the whole gap. boltz2 shares
the method and must not take the branch -- its checkpoint puts it on the entity
convention -- so the branch reads CHAIN_BUCKET_ON_SAME_CHAIN rather than being
keyed to the family.

ELIMINATED for the remaining ~4e-01, all by measurement:

  * the BIN CONVENTION. The dump carries native's OWN reduced `out.pae`
    alongside `out.pae_logits`, so our bin centres can be checked against
    native's rather than assumed: they agree to 4.62e-07 (pae) and 3.16e-07
    (pde). The gate's usual "both sides use OUR bin centers so it cancels" is
    for once verifiable, and it holds.
  * the spurious boltz-only terms. Our shared `_boltz2_reembed` adds
    `token_bonds_type_embed`, `contact_encoder`/`contact_fourier` and
    `s_input_to_s`, which ESMFold2's head does not have -- and all four are
    ZERO in its blob, so they contribute nothing.
  * the learned distogram. `distogram_boundaries` IS converted (38 edges, rms
    30.4, at the `confidence_head` scope) and `reembed_dist_bins` is 39, so the
    binning is ESMFold2's own and not boltz2's.
  * a BLOCKS sweep says nothing here and must not be read: this cell is
    DUMP-DRIVEN, so native is fixed at full depth and BLOCKS truncates only our
    side -- which is why the error grows as blocks are removed (3.68e+00 at
    BLOCKS=0). The one-sided-truncation trap, in a gate where it cannot be
    avoided.

STILL OPEN, and still not "native's bf16 floor" -- that attribution was retracted
on 2026-09-10 after BF16=all moved only the third digit. The untested
possibility remains that native's own output is that noisy; it needs two dumps
from ~/venv_esm and the spread between them, which is the same method
[[esm-tower-numerical-modes]] used for the tower.

### esmfold2 confidence: "native is noisy" is REFUTED (2026-09-11)

The measurement that was named as next -- two dumps from ~/venv_esm and the
spread between them -- is done. Native is indeed NOT reproducible run to run:

    conf.in.z                       max|d| 12.0      (3.64e-01 relative)
    conf.in.s_inputs                max|d| 3.1e-02   (2.07e-01)
    conf.out.pae                    max|d| 4.95e-01  (7.08e-02)
    conf.out.plddt                  max|d| 2.3e-03   (3.20e-03)
    conf.in.relative_position_encoding   IDENTICAL

That is not numerical noise, it is BY DESIGN: ESMFold2 keeps 25% dropout on the
LM pair rep at INFERENCE, resampled every loop
(`config.lm_encoder.per_loop_lm_dropout`; disabling it costs ~18 A on 6MRR).
The relative-position encoding, which has no stochastic input, is bit-identical
across the two runs -- which is the control that says the rest is the dropout
and not the tower's numerical band [[esm-tower-numerical-modes]].

**It does not excuse our gap.** The confidence gate INJECTS `conf.in.*` from one
dump and compares against `conf.out.*` from that SAME dump, so the run-to-run
variation cancels exactly. Our head differs by 4.31e-01 on inputs it was handed
verbatim. The remaining possibility named on 2026-09-10 -- "that NATIVE's own
output is that noisy" -- is therefore refuted for this cell, and the four rows
are OUR head.

Worth keeping in view for OTHER esmfold2 cells though: any gate that re-runs
native rather than injecting a dump is comparing against one draw of a
stochastic trunk, and a 3.64e-01 spread on z is large enough to matter.

### ...and the relpos residual is real but too small to be the gap

With the chain-bucket flag now passed, our confidence rel_pos against NATIVE's
dumped `in.relative_position_encoding` -- the exact tensor native adds, so this
comparison needs no reimplementation:

    corr 0.99999768   rms ours/nat 1.0001   max|d| 0.02061   max|d|/rms 1.74e-02
    error largely ANTISYMMETRIC (||d - d.T|| / ||d|| = 1.20)

Real, and NOT the remaining 4.31e-01: 0.0206 absolute on a z of rms 33.0 is
6e-4 relative, and the scale factor from z to pae is known from the two dumps --
a 3.64e-01 relative change in z moved pae by 7.08e-02, so 6e-4 would move it by
~1e-04. Two orders too small. An antisymmetric residual points at an orientation
or offset term in the bucketing rather than the chain predicate, and it is worth
closing on its own.

So, for the esmfold2 confidence rows, ELIMINATED so far: the bin convention, the
boltz-only terms our shared re-embedding adds, the learned distogram, native's
own reproducibility, and now rel_pos. NEXT: build native's `z_base` from the
dump's inputs and the checkpoint's confidence weights and compare it against our
`_boltz2_reembed` output -- the term-by-term diff that found boltz2's bug in one
step. The dump has every input, so nothing needs to be re-run in ~/venv_esm.

### esmfold2 confidence after the swap fix: rel_pos is the last term

The s->z swap took L4 from 8 BAD rows to 2. What is left is ONE term, and the
same one for both models -- each model's remaining `z_base` error matches its
rel_pos residual almost exactly:

              rel_pos residual      z_base residual     L4 pae
    esmfold2      1.74e-02             1.69e-02         2.59e-02
    esmfold2_fast 2.09e-02             2.67e-02         1.06e-01

(esmfold2_fast reads worse downstream from a comparable z_base error; its head
amplifies more. Both residuals are partially ANTISYMMETRIC -- 1.20 and 0.88 on
||d - d.T||/||d||, against 2.00 for the pure transpose the swap produced -- so
this is not another swap, it is a few buckets.)

CHECKED BY READING and agreeing, so the next step must be numerical:

  * the concat ORDER is the same on both sides --
    `[rel_pos 66, rel_token 66, same_entity 1, rel_chain 6]` = 139
    (modeling_esmfold2_common.py ResIdxAsymIdSymIdEntityIdEncoding vs AF3's
    `create_relative_encoding`, whose docstring lists that order).
  * `dij_residue[i,j] = residue_index[i] - residue_index[j]`, clipped to
    [0, 2r] and sentinelled to 2r+1 off-chain -- AF3's convention.
  * the chain block's predicate is the one fixed today (same-CHAIN -> sentinel).
  * `dij_token` gates on `bij_same_chain & bij_same_residue`, where AF3 gates on
    same-residue alone. INERT on a monomer, where same-residue implies
    same-chain -- but a real difference on a complex, and nothing covers it.

NEXT: an element-wise diff of the 139-wide FEATURE vectors, not their
projections. Both sides project the same `embed.weight` -- the converter maps
`rel_pos.embed.weight` straight into `rel_pos_project` -- so the disagreement is
in the features and a per-bucket diff will name it in one measurement.

### CORRECTION: rel_pos is NOT the remaining esmfold2 term

The entry above inferred that rel_pos accounted for the rest, because each
model's z_base residual matched its rel_pos residual in MAGNITUDE. Both halves
of that were tested and it is wrong:

  * our 139-wide relative-position FEATURES are BIT-IDENTICAL to native's
    (max|d| 0.0 across all four blocks, captured by hooking native's
    `rel_pos.embed` input). The features are not the problem.
  * the projected relpos differs only because NATIVE COMPUTED IT IN BF16 --
    its dumped tensor is EXACTLY bf16-representable (round-trips at max|d|
    0.000e+00), which ESMFold2's `autocast(bfloat16)` trunk explains. Not a port
    bug.
  * and injecting native's own relpos into our head barely moves anything:
    z_base 1.69e-02 -> 1.53e-02, full_pae 2.59e-02 -> 2.57e-02. So rel_pos is
    about a tenth of the residual, not the whole of it.

Two magnitude coincidences in a row, and both looked convincing. The standing
lesson holds harder than it reads: a number that matches in magnitude is not a
cause until substituting it changes the answer.

STILL OPEN: ~1.5e-02 on z_base for esmfold2 (2.6e-02 on its pae; esmfold2_fast
2.7e-02 and 1.06e-01), from another term in the re-embedding. `rms ours/nat` is
0.9992 -- ours slightly SMALL -- with a small constant part and partial
antisymmetry.

NEXT: the term-by-term numerical diff that found boltz2's bug in one step. The
tooling now exists and the method is proven -- native's confidence head REPLAYS
bit-exactly on the dump's recorded inputs (max|d| 0.000e+00), so each of
z_norm / s_to_z / s_to_z_transpose / s_to_z_prod / distogram can be hooked
individually and subtracted. Do that instead of reading the source again: every
finding today that survived came from a subtraction, and both that did not came
from a resemblance.

### esmfold2's confidence s_inputs LayerNorm: the THIRD dropped-vocab site

Found by the term-by-term subtraction, which named it in one measurement after
two failed resemblance arguments. Against native's captured terms:

    z_norm                  2.00e-06   rms ours/nat 1.0000   <- exact
    s_to_z (row)            2.73e-02                0.9952
    s_to_z_transpose (col)  2.77e-02                0.9950
    s_to_z_prod_out         8.06e-02                0.9912

Three terms low by the SAME ~0.5%, one exact. They share exactly one input --
`s_inputs_norm` -- and that is a LayerNorm taken over 447 channels where native
takes it over 451. Dropping the four columns AF3's 31-class blocks lack is exact
for a bias-free Linear and NOT for a LayerNorm, which is
[[dropped-vocab-columns]] for the third time (openfold3 from the start,
protenix/rf3 2026-09-07, opendde and now this one 2026-09-10/11).

Fixed the way the DIFFUSION already fixed its own `single_cond_initial_norm`:
the graph widens 447 -> 451 with ESMFold2's permutation (its gap sits at class
1, so this is not a pad-with-zeros) and the converter emits the 451-wide
weights via the existing `permute_s_inputs` / `_permute_vec`. boltz2 shares the
method and must not widen -- gated on PAIR_ONLY_TRUNK.

    all four terms      -> 7.6e-07 .. 2.7e-06, rms ratio 1.0000  (exact)
    z_base   esmfold2      1.69e-02 -> 7.60e-03   rms ratio 0.9993 -> 1.0000
             esmfold2_fast 2.67e-02 -> 1.10e-02

and what is LEFT of z_base is native's bf16: the residual max|d| is 0.02061,
which is the rel_pos residual's max|d| to the digit.

**THE OUTPUT GAP IS DOWNSTREAM.** full_pae barely moved (2.59e-02 -> 2.62e-02),
so with the re-embedding now exact the remaining ~2.6e-02 is in the folding
trunk stack, the row-attention pooling, or the four heads. Those are all
hookable in the same bit-exact replay, so the next subtraction is mechanical.

### ...and what remains is the confidence PAIRFORMER

With the re-embedding exact, the next subtraction puts the rest in the stack.
Our confidence pairformer, run on NATIVE's exact tapped `z_base` and built the
way `confidence_head` builds it (with_single=True, pair attention off for
PAIR_ONLY_TRUNK; 26 scopes, 0 unmapped):

    trunk_out   corr 0.99999524   rms ours/nat 1.0003
                rms(d)/rms 3.10e-03   p99.9/rms 3.43e-02   max/rms 4.41e-01

So the BULK agrees to 0.3% and a thin tail does not -- and the tail is what the
audit grades. Three explanations tested and rejected:

  * AMPLIFICATION. The stack multiplies the pair's magnitude 123-fold (rms 2.7
    -> 333), but its GAIN on a perturbation of its own input is only 1.4x to
    2.0x, stable across three decades. Our z_base enters 7.60e-03 out; 1.4x of
    that is ~1e-02, not 4.4e-01.
  * BF16. Native's confidence tensors are NOT bf16-representable (z_base
    max|d| 6.2e-02, trunk_out 3.1e+01 when round-tripped), so this head runs
    fp32 -- unlike the rel_pos it receives, which IS exactly bf16 because the
    TRUNK computed it. The two live in different precisions and only the first
    is explained by autocast.
  * a wrong isolation. The first attempt built the stack with_single=False and
    got the same number as with_single=True, so the reading is not an artifact
    of how the probe assembles it.

NEXT: the sub-module diff inside one confidence pairformer block -- the two
triangle multiplications and the transition -- which is the ladder that found
rf3's `/L`. Native's head replays bit-exactly, so each can be hooked.

### ...and the last of it is native's PER-MODULE bf16, measured not asserted

The sub-module subtraction inside confidence block 0, on native's own captured
inputs (the block runs them SEQUENTIALLY, so each input is rebuilt from native's
preceding output):

    tri_mul_out       rms(d)/rms 4.99e-03   ratio 1.0003
    tri_mul_in        rms(d)/rms 4.87e-03   ratio 0.9999
    pair_transition   rms(d)/rms 2.90e-03   ratio 1.0001

No outlier: a uniform ~0.5% across three different modules whose weights are
BIT-IDENTICAL to native's (LayerNorm scale and offset, `transition1 == w12.T`,
`transition2 == w3.T`, and the SwiGLU halves are not swapped -- the swapped
comparison reads 3.6). Exact weights, exact input, 0.5% out: arithmetic.

And the arithmetic is native's PRECISION, by a test that does not depend on
reading anything:

    b0.in               bf16 round-trip 2.30e-02   -> fp32
    b0.tri_mul_out      bf16 round-trip 0.00e+00   -> BF16
    b0.tri_mul_in       bf16 round-trip 0.00e+00   -> BF16
    b0.pair_transition  bf16 round-trip 5.96e-02   -> fp32

Native's two triangle multiplications emit exactly-bf16 tensors while the block
input and the transition do not, so autocast covers those modules and not the
rest. We run all of it fp32.

TWO THINGS THIS IS NOT. It is not fixed by rounding our output to bf16
(4.99e-03 -> 5.25e-03): bf16 ACCUMULATION is not bf16 rounding of an fp32
result. And it is not fixed by the gate's global `BF16=all` (full_pae
2.62e-02 -> 2.63e-02), because that casts everything while native casts two
modules -- a global knob cannot express a per-module autocast.

So esmfold2's remaining L4 residual is a precision difference in the confidence
pairformer, compounding over four blocks to ~2.6e-02 on the pae expectation.
This is a much better-founded claim than the "native's bf16 floor" retracted on
2026-09-10: that one asserted a precision limit on OUR side without measuring,
and this one is native's own tensors testifying to their dtype.

Closing it properly would need our tri_mul to run in bf16 for this family
alone -- a per-module precision knob the graph does not have. Worth it only if
these two rows are worth a new global_config axis.

## L5 validates the day's work, and the driver was recompiling everything

### The folds
6MRR, 5 samples, one seed, against the recorded baselines. The UNTOUCHED models
are the control and they all land within +-0.01:

    openfold3   1.544 / 1.541     openbind0  1.650 / 1.649
    protenix1   1.684 / 1.694     protenix2  0.697 / 0.702
    chai1       1.723 / 1.719

and of the eight changed today:

    opendde        0.737 / 0.767   better (the 833-wide single_cond LayerNorm)
    boltz2         0.421 / 0.434   better (the entity chain bucket)
    esmfold2       1.339 / 1.352   better (the duplicated MSA query row)
    intellifold2   1.515 / 1.514   neutral, as the fp32 template blob was
    esmfold2_fast  1.181 / 1.181   esmfold2_lm300m 1.687 / 1.687  neutral
    esmfold2_lm600m 1.522 / 1.506
    rosettafold3   1.026 / 0.986   flagged, and NOT a regression:

rosettafold3's fold is NONDETERMINISTIC run to run on the SAME seed. Three runs
of seed 0 give best 0.968, 0.949 and 1.026 -- the fourth sample sits near a
boundary and moves ~0.08 while the others move ~0.02. So +0.040 is inside its
own band, and the recorded 0.986 is one draw. Single-number fold comparisons at
this precision do not mean anything for this model.

### The driver was recompiling the graph in every cell
Every gate runs in a FRESH PROCESS. Measured on the A10 for a 6MRR fold:

    import                0.5 s
    featurise + setup     8.0 s
    fold, cold          140.9 s
    fold, warm PROCESS   67.5 s    <- no recompile

so ~73 s of each cell was XLA and ~67 s was the five samples. `run_alphafold.py`
has enabled JAX's persistent compilation cache from the start via
`platform.enable_compilation_cache`; the gates never did and no cache existed on
disk. Wired into `gate` now:

    one L5 cell, cold cache   2m26s
    one L5 cell, warm cache   0m43s      3.4x, cache 4.0 MB

The min_compile_time / min_entry_size settings are load-bearing: JAX's defaults
skip entries that are small or quick to build, which is most of what the module
gates compile.

NOT for timing work -- [[jax-cache-override]] records benchmarks silently
measuring cache hits. The parity matrix compares NUMBERS, so a hit is free.

### RESOLVED: the esmfold2 confidence gap was a MISSING OUTER RESIDUAL

And my bf16 attribution above was over-claimed. Native does

    pair_delta = self.folding_trunk(pair, ...)      modeling_esmfold2.py:53
    pair.add_(pair_delta.float())                                       :54
    ...
    pae_logits = self.pae_head(self.pae_ln(pair))                       :108

and `FoldingTrunk.forward` returns the FULL updated pair, not a delta -- its
blocks are already residual, and the variable's name at the call site is
misleading. So the tensor the heads read carries z_base TWICE. Our graph
replaced the pair with the stack's output and dropped the outer add.

Fed native's own tapped trunk output, our heads read pae 2.53e-02 (esmfold2)
and 1.10e-01 (esmfold2_fast); fed z_base + that output they read 1.35e-06 and
1.11e-06, rms ratio 1.0000 both. The cell:

    esmfold2       pae 2.62e-02 -> 5.98e-03   pde 2.23e-02 -> 4.83e-03
                   plddt 3.62e-03 -> 9.73e-04  resolved 3.81e-03 -> 5.95e-04
    esmfold2_fast  pae 1.08e-01 -> 6.83e-03   pde 1.19e-01 -> 6.56e-03
                   plddt 1.41e-02 -> 5.45e-04  resolved 3.19e-02 -> 7.72e-04

**BAD = 0.** boltz2 shares the method and is not in PAIR_ONLY_TRUNK, so it does
not take the branch; re-checked at 4.99e-06.

WHAT I GOT WRONG, since the reasoning was recorded above as settled: the bf16
measurements are all correct -- native's tri_mul tensors DO round-trip through
bfloat16 at max|d| 0, autocast is on `pair.is_cuda`, and the per-sub-module cost
IS ~0.5%. What was wrong was concluding that this explained the CELL. The tell
was there and I passed it: esmfold2_fast read 4x worse than esmfold2 while their
per-sub-module errors were identical (5.04e-03 vs 4.99e-03). Identical parts
cannot produce a 4x different whole, and that asymmetry should have sent me
downstream immediately instead of being waved at as "its head amplifies more".

The residual ~6e-03 IS the bf16 -- that part of the earlier entry stands.

## The EMPTY-TEMPLATE sweep: opendde needed the gap too, intellifold2 does not (2026-09-11)

The other half of the protenix pair, swept the same way. Two questions per
vendor: what an empty template SLOT contains, and what the embedder DIVIDES by.

**First, a claim in our own code that is wrong for every model.** The generic
`TemplateEmbedding` comment said "the empty slots contribute exactly zero". They
do not: an empty slot still carries the Z-DEPENDENT half of the embedding
(`z_proj(z_norm(z))`, or AF3's equivalent), so the template term is LIVE with no
template at all. Measured as the rms of `z_after_template - z_init_generic` with
no template supplied:

    openfold3    24.39        intellifold2   9.53
    protenix1    17.32        opendde        6.66

That is what made protenix1's missing term worth 9 A, and it means a "no
template" batch is not a no-op path for anybody.

| model | empty slot | divisor | verdict |
|---|---|---|---|
| `protenix1/2` | slot 0 GAP, slots 1-3 zero | padded slot count | FIXED (worth 9 A) |
| `opendde` | **ALL FOUR slots GAP** -- `make_dummy_feature` does `torch.full(..., 31)` with its own `# gap` comment | padded slot count, and it takes the GENERIC module which already divides that way | FIXED: the gap restype only (`empty_template_gap_slots='all'`) |
| `intellifold2` | **ZERO, deliberately**: `[STD_RESIDUES_WITH_GAP["-"]] * num_res ... * 0` with the comment "use 0 to indicate empty template, instead of 'GAP'" | slot count | matches us, nothing to do |
| `boltz2` | ONE slot, `template_mask` all zero | mean over PRESENT templates, so zero | matches us |
| `openfold3`, `openbind0` | zero | slot count | matches us, and proven: our of3 trunk agrees with native at 0.99993 on a real input, which it could not if this term differed |
| `rosettafold3` | n/a -- its embedder has no template gating at all and always contributes | one pass | already gated at corr 1.000000 |
| `chai1` | its own path | `clamp_min(n_templates, 1)` | already recorded |

opendde's is from its CODE plus the divisor already matching, not from an
activation comparison -- there is no native opendde real-input trunk dump yet,
and that is the check that would close it. Its folds moved inside their bands
(6MRR 0.734 -> 0.808, plain 5K9P 1.591 -> 1.641), which after the bistability
lesson is not evidence either way.

`TEMPLATE_MEAN_OVER_ALL_SLOTS` stays protenix-only on purpose: opendde does not
run the fused module, so adding it there would have been an inert config entry
implying a branch that never fires. I added it and then took it out.

## The self-MSA DEPTH sweep: 6 of 10 vendors emit ONE row, and we fed two (2026-09-11)

protenix's depth-1 MSA turned out to be a family question, so it was swept
across every vendor rather than guessed. AF3 concatenates a paired and an
unpaired MSA, each beginning with the query, so a chain with NO alignments
arrives with the query TWICE. The outer product mean over duplicates is
unchanged; the pair-weighted averaging and the row transition are
depth-sensitive.

Read off each vendor's own code or its own batch -- not inferred:

| model | native depth | where it is decided |
|---|---|---|
| `protenix1`, `protenix2` | **1** | dump carries `msa` (1, 76) |
| `esmfold2` (all) | **1** | dump carries `feat.msa` (1, 1, 68) |
| `boltz2` | **1** | `dummy_msa` holds ONE sequence; the pairing takes the first row per chain and finds nothing to add. Its data module's batch: `msa` (1, 1, 68) |
| `intellifold2` | **1** | forks boltz's featuriser, `dummy_msa` and pairing loop included |
| `rosettafold3` | **1** | atomworks early-returns `full_encoded_msa = expand_dims(encoded["seq"], 0)` when no polymer MSA is present |
| `openfold3`, `openbind0` | 2 | native dump carries `msa` (1, 2, 76, 32) -- matches us |
| `opendde` | 2 | builds a paired AND an unpaired `RawMsa`, each falling back to `[query]` ("Make sure the MSA always has at least the query") |
| `chai1` | n/a | settled earlier: a depth-1 MSA never reaches its trunk ([[chai-msa-of-one]]) |
| `alphafold3` | 2 | it IS the AF3 convention |

So `dedupe_self_msa` now covers protenix1/2, boltz2, intellifold2 and
rosettafold3 as well as esmfold2, each behind its own env override for A/B.

**The one ACTIVATION check available says the fix is right.** boltz2's native
trunk dump (`~/boltz2_6mrr/trunk_dump.npz`) lets our own trunk be compared on
our own features:

    pairformer input   corr 0.99913 (dedupe) vs 0.99741 (duplicate row)
    rms ours/native    0.99867     vs 1.01137   -- the duplicate inflates z ~1.1%
    trunk output       0.99928     vs 0.99894

Folds move little, which is the expected shape for a 1% pair difference and is
NOT the evidence: boltz2 6MRR 0.431 -> 0.477, plain 5K9P 1.714 -> 1.690, 1STP
0.458, RNA 1.197 -> 1.199, DNA 1.534 -> 1.520; intellifold2 6MRR 1.517 ->
1.521, plain 5K9P 1.669 -> 1.661; rf3 6MRR 0.953 -> 0.942, plain 5K9P 1.573 ->
1.574, RNA 1.044. All inside their own per-process bands.

STILL UNCHECKED, and the reason this entry names them: what an EMPTY TEMPLATE
slot contains and what each embedder divides by. Measured for protenix (gap
restype, divides by the padded slot count -- the 9 A bug above), for of3 (our
trunk matches native at 0.99993 with our current behaviour, so whatever of3
does, we reproduce), and for boltz2 (ONE slot, `template_mask` all zero, so its
present-weighted mean contributes exactly zero -- ours too). `intellifold2`,
`opendde` and `rosettafold3` have NOT been checked.

## L0-L6 re-run clean, and what the two "movers" actually were (2026-09-12)

The matrix had gone ~40 commits stale. Re-run in full: **256 OK, 0 FAIL, 1 SKIP,
63 N/A**. The SKIP is the one real hole (boltz2 has no in-process L3 adapter; its
`L3.denoise_inject` cell covers the same module by injection).

Two L5 rows sat above the 2026-09-06 baseline and BOTH dissolved under the right
statistic rather than a fix:

* **protenix2 0.702 -> 1.016 best-of-5.** Not the input-convention knobs: with
  `AF3_NO_PX_TEMPLATE_GAP` and `AF3_NO_PX_DEDUPE_MSA` set, every combination
  lands at mean 1.386-1.394, i.e. the knobs move this target by ~0.005 A. Over
  **20 samples** (SEEDS=0,1,2,3) it reaches **best 0.589, mean 1.459** -- better
  than the baseline row. 0.702 was a best-of-5 tail draw.
* **esmfold2_lm600m 0.858 -> 1.522 best-of-5.** Not the hidden-state file
  either: `dev/bench/hid_new.npz` and `lm_inputs/esmc_600m.protein_6mrr.npz`
  give 1.503 and 1.527. Over 20 samples: **best 1.090, mean 1.508**, which
  reproduces the own-tower condition recorded in
  [[esmfold2-relpos-chain-bucket]] (1.044 best / 1.519 mean) to 0.05 / 0.01 A.
  The 0.858 baseline row was measured under a different condition, not lost.

**The lesson is the one the baseline memory already states and I still had to
re-learn: best-of-5 is a TAIL statistic.** Both "regressions" were a single
lucky or unlucky draw, and both took one 20-sample run to settle. Do not open an
investigation on a best-of-5 move again.

## The terminal-atom drop MASKED where the vendors REMOVE (2026-09-12, FIXED)

The of3-lineage cluster above is a REGRESSION, and the cause is this week's
terminal-atom drop. `parity_audit` over the same 274 comparisons:

    2026-09-10   PARITY=237  CLOSE=8   FLOOR=17  LOOSE=10  BAD=2
    2026-09-12   PARITY=228  CLOSE=10  FLOOR=17  LOOSE=5   BAD=14

and all 14 BADs are openfold3/openbind0 atom encoder / decoder / denoise, which
on 2026-09-10 read corr 1.000000 (a_token max|d| 0.0011, p_pair_valid max|d|
0.00012). The gate headers date it exactly: `574 real atoms` -> `573`.

**The drop itself is right** -- of3's own featuriser agrees (the ligand/dimer
diff above matched 917 and 1202 atoms exactly). **The implementation is not.**
`_drop_atoms_by_name` masks the atom in place, and the vendors REMOVE it before
tokenising. A mask leaves a HOLE in the flat atom layout, and two things are cut
on that axis:

1. **The 32-atom attention windows.** Every block after the hole has a different
   membership than the vendor's compacted axis. That is the whole of the of3
   cluster -- and why only of3 and openbind0 show it, though rf3, if2 and boltz2
   ALL went 574 -> 573 too: of3's native adapter builds its own compacted layout,
   so our hole misaligns against it, while the other three consume our flat
   features hole and all, so it cancels on both sides.
2. **The mmCIF gather -- and this one reaches users.** `flat_output_layout` and
   `empty_output_struc` are built at featurisation and the drop never touches
   them, so the dropped atom is still in the output while the model no longer
   predicts it. Measured, featurisation only:

        alphafold3   flat_output_layout 574 | model predicts 574   ok
        openfold3    flat_output_layout 574 | model predicts 573   1 ORIGIN ATOM
        boltz2       flat_output_layout 574 | model predicts 573   1 ORIGIN ATOM

   i.e. every mmCIF from the five drop models (openfold3, openbind0, boltz2,
   intellifold2, rosettafold3) carries a terminal atom at (0, 0, 0), announced
   only by a `logging.warning`. `output_parity.py` was written blind to this and
   catches it on its first run, which is the argument for the gate.

**The fold is unaffected** (native of3 on 6MRR: 1.637/1.714 preview-2 and
1.644/1.766 v0.5.0, against our 1.546/1.717 and 1.579/1.776) because our graph
masks the hole consistently everywhere inside itself. Only a module gate and the
output side can see it -- which is exactly the class of bug L0-L6 was extended
to catch.

**Fix**: remove the atoms from the layouts rather than masking a finished batch
-- blank them in `token_atoms_layout` (`token_atoms_mask` is just
`atom_name.astype(bool)`, so the flat axis compacts by itself), rebuild the
`AtomCrossAtt` gathers, and filter `flat_output_layout` + `empty_output_struc`
so nothing is written at the origin.

**FIXED** in `_remove_dropped_atoms_from_layouts`. All 14 BAD cells resolved,
with the drop still applied (573 atoms, so the convention is kept):

    openfold3  a_token       0.997932 -> 1.000000   (max|d| 231.8 -> 0.00101)
               q_atom        0.997916 -> 1.000000   (121.6  -> 0.00107)
               p_pair_valid  0.994137 -> 1.000000   (62.79  -> 0.00012)
    openbind0  all three     -> 1.000000
    of3        denoise per-atom mean  1.2147 A -> 0.0002 A
    openbind0  denoise per-atom mean  1.2130 A -> 0.0013 A

which reproduces the 2026-09-10 numbers exactly. Every model's layouts now
agree (output == structure == predicted == real queries), and 6MRR is unmoved:
of3 1.548/1.718, openbind0 1.578/1.773, boltz2 0.467/0.553, if2 1.511/1.626,
rf3 1.020/1.556. The SEP-20 PTM case still gives 605 atoms on both of3 and
boltz2 -- the OXT dropped and phosphoserine's O3P kept -- so the
atomised-residue distinction survives the layout rebuild.

**Closed end to end.** The post-fix matrix (`parity_runs/2026-09-12-postfix/`)
audits to **PARITY=242 CLOSE=10 FLOOR=17 LOOSE=5 BAD=0** -- one PARITY better
than before the regression landed. L6 re-run over the five drop models is 36/36
OK across RNA, DNA, ligand, complex, PTM and both protein targets, with the
modality numbers unmoved (of3 RNA 1.333 both, ligand 0.457 both, complex 9.178
-> 8.281). And `output_parity` now passes for every drop model on the case that
used to carry the origin atom:

    openfold3 / boltz2 / rosettafold3 / intellifold2   monomer  601 atoms  OK
    alphafold3 (keeps its OXT)                         monomer  602 atoms  OK

The filter is by the same BOOLEAN, never by atom name: a name filter would take
the O3P case with it. And a drop combined with `flat_atom_order` now raises
rather than silently discarding the permutation; opendde is the only model with
a structural atom order and it keeps its terminal atoms.

## The of3-lineage atom path: a module gap with no fold cost (2026-09-12, OPEN)

Three levels in a row single out openfold3 and openbind0 and nobody else:

    L2.atom_encoder  p_pair_valid  corr 0.994137 / 0.987120   (others 1.000000)
    L2.atom_decoder  r_update      corr 0.998142 / 0.999486   (others 1.000000)
    L3.denoise       per-atom      mean 1.2147 A / 1.2130 A   (others <= 0.0003)

All three are recorded OK because each gate scales by rms(native) (15-20 A at
the denoise step). What has been established:

* It is NOT the input. `denoise_parity`/`atom_parity` build OUR batch and feed
  the SAME features to both sides, so the conformer difference the featurisation
  diff found cannot reach this.
* It is NOT the per-atom reference features. `c_atom_cond` is bit-exact
  (corr 1.000000, max|d| 0), and the FEAT sweep (keep pos / charge / element /
  chars, zero the rest) leaves the pair gap at 0.9943-0.9947 every time --
  including with positions zeroed, which makes every offset zero.
* It is NOT fully the trunk-pair gather: zeroing z moves 0.9941 -> 0.9968, and
  zeroing z AND positions together still leaves 0.9977.
* The key-window edge policy is NOT the cause: `slide_qblock` (if2's convention,
  and the one that reproduces native's 576-atom bound) makes it WORSE,
  0.9941 -> 0.9835. Retracted.

**And it costs nothing end to end.** Native of3 was run on 6MRR for both
releases, 5 samples each, scored the same way:

    openfold3 (preview-2)  ours 1.546 / 1.717      native 1.637 / 1.714
    openbind0 (v0.5.0)     ours 1.579 / 1.776      native 1.644 / 1.766

Means agree to 0.003 and 0.010 A, and our best is better in both. So whatever
the module gap is, it does not reach the fold on this target.

The remaining suspicion is the COMPARISON itself: our `p_lm` is (51, 32, 128, 16)
against native's (18, 32, 128, 16), because we pad the flat atom axis to
`padded_tokens * average_num_atoms_per_token` (1632) where native compacts to
`ceil(573/32)*32` (576). The gate truncates ours to native's 18 blocks, which is
only like-for-like if the key windows agree -- and for the LAST block holding
real atoms they do not (ours 496..623, native 448..575). An attempt to test this
by lowering `average_num_atoms_per_token` did not take effect (still 51 blocks),
so it is untested, not disproved. Next step: compare blocks 0..16 only; if those
are exact, the whole disagreement is the edge block and the gate needs a bound
that matches each vendor's own padding.

## Cross-chain gates: two modules, and FOUR harness faults (2026-09-13, CLOSED)

Every module gate ran ONE protein chain, so no L1-L4 cell exercised a
cross-chain convention -- and two of this project's bugs (boltz2's relpos entity
bucket, esmfold2's chain bucket) lived exactly there. Two gates now take a
complex:

* `trunk_init_parity.py --chains_json <fold input>` -- z-init, where the
  relative-position encoding lives. **All nine models pass** on a 152-token
  homodimer: corr 1.000000, max|d| <= 5.5e-04.
* `template_parity.py --dimer` -- chain A carries the self-template, chain B is
  an unrelated sequence, so 76 of 144 tokens are covered. **All eight models
  pass**: corr 1.000000, max|d| <= 2.0e-03.

**Both gates ended green, and every one of the four faults on the way was in the
HARNESS.** That is the finding: a cross-chain gate is easy to write in a way
that tests nothing, and easy to make read like a port bug.

1. **Both sides were told "one chain."** Ours by `multichain = np.ones(...)`,
   native by `asym_id = torch.zeros(...)` in EVERY adapter. Correct on the
   monomer the gate ran, and it silently disables every cross-chain term on a
   complex. Both now take the batch's real `asym_id`.
2. **5K9P IS ubiquitin.** The first `--dimer` used "ubiquitin" as the foreign
   chain against a 5K9P template, got 152 of 152 tokens covered by same-entity
   propagation, and read a meaningless corr 1.000000. Chain B is now 6MRR's
   designed sequence.
3. **The wrong native CLASS, and the wrong restype rule** (boltz2, read
   corr 0.801). The adapter built `TemplateModule` where the checkpoint's
   `hyper_parameters` carry `use_templates_v2=True`; the two share a parameter
   set, so `load_state_dict` takes either -- the openbind0-into-of3-main trap
   again. With one templated chain the V1 `asym_id` mask and the V2
   `visibility_ids` mask COINCIDE, so the class was not the cause: the cause was
   the adapter handing native `aa + 2` for every token when boltz allocates
   `res_type = np.zeros(...)` and fills it only where the template covers
   (featurizerv2). Native saw residue types our port correctly withholds. Both
   are fixed, and `visibility_ids` is now built by boltz's own rule (templated
   chain -> the template's pdb_id, untemplated -> `-1 - asym_id`).
4. **An UNMASKED re-derivation** (openfold3 0.9998, openbind0 0.9956,
   intellifold2 0.9959). `_af3_template_features` rebuilds the features for the
   four models that ride AF3's `TemplateEmbedding` -- which has no `_features`
   method to borrow -- and it returned the distogram and unit vector RAW, where
   `SingleTemplateEmbedding.construct_input` multiplies them by
   `pseudo_beta_mask_2d` and `backbone_mask_2d`, both already multiplied by
   `multichain_mask_2d`. Native was handed geometry our module zeroes. Invisible
   on a covered monomer, where those masks are all ones.

**The floor test is what made these worth chasing.** `TMPL_FLOOR=1` reruns
NATIVE on `z * (1 + 1e-6)` and compares it to native's own answer: mean|d|
**1.2e-06** against disagreements of 0.31 (of3) and 0.48 (if2). Five orders of
resolution, so "an amplifier being measured" was ruled out before any code was
touched. `TMPL_SPLIT=1` reports the error per chain block (covered x covered,
uncovered x uncovered, cross), which is what showed boltz2's output was half
native's magnitude on uncovered pairs.

The monomer case is unmoved by all of this (of3 max|d| 0.00024, boltz2 0.00003).

**Both are now a LEVEL in the driver**, `L1x`, and in the default level list --
a gate the driver does not run is a gate that rots (three tracked gates had
already stopped running once). First driver run: **21 OK, 6 N/A, 0 FAIL**
(`parity_runs/2026-09-13-l1x/`). `gate_applies.py` learned that `L1x.template`
inherits `L1t.template`'s rules, or the four ESMFold2 variants -- which have no
template embedder at all -- reported SKIP, and a SKIP that is not a real hole is
what makes the summary stop meaning anything.

## The output side is gated over three input classes (2026-09-12)

`output_parity.py`, 14/14 after the drop fix:

    openfold3 / boltz2 / rosettafold3 / intellifold2   601 monomer, 917 ligand, 1202 dimer
    alphafold3 (keeps its OXT)                         602 monomer, 918 ligand, 1204 dimer

The dimer is the case that mattered -- two chain termini, two dropped atoms, two
holes in the flat atom axis -- and 1202 = 2 x 601 with no atom at the origin.

Each case is a fresh shape and so a full compile, ~10 min per model-case; use
`CASES=monomer` (or a comma list) rather than running all 14 models blind.

## We recycle 11 times; three vendors run 3 or 4 (2026-09-12, DATA, no change made)

The counts, read from each vendor's own loop rather than its README:

| model | native trunk passes | ours | ratio |
|---|---|---|---|
| alphafold3 | 11 | 11 | **1.00** |
| openfold3 / openbind0 | **4** (`num_recycles + 1`, default 3) | 11 | 2.75 |
| boltz2 | **4** (`range(recycling_steps + 1)`, default 3) | 11 | 2.75 |
| chai1 | **3** (`range(n)`, total) | 10 | 3.33 |
| protenix1/2 | 10 (`range(N_cycle)`) | 11 | 1.10 |
| rosettafold3 | 10 (`range(n_recycles)`) | 11 | 1.10 |
| intellifold2 | 10 | 11 | 1.10 |

So it is three models, not the whole family, and alphafold3 -- the reference --
already matches.

**What the extra passes buy on 6MRR: nothing.** Five samples at our count and at
the vendor's own:

    openfold3      11 passes  best 1.546  mean 1.715   |  4 passes  1.689 / 1.749
    openbind0      11 passes  best 1.574  mean 1.771   |  4 passes  1.562 / 1.670
    boltz2         11 passes  best 0.467  mean 0.554   |  4 passes  0.505 / 0.534
    protenix2      11 passes  best 1.005  mean 1.389   | 10 passes  0.672 / 1.327
    rosettafold3   11 passes  best 1.018  mean 1.557   | 10 passes  1.095 / 1.646

Four of five are equal or BETTER at the vendor's count, the fifth (rf3) worse by
0.089 -- all inside the sampling band. Native of3 measured here is 1.714 mean,
so at MATCHED passes our 1.749 is within 0.035 of it, which is the honest form
of the parity claim: the earlier "we match native end to end" was true while we
spent 2.75x the trunk compute.

**And matching costs no runtime at this size.** of3 over 20 samples, compile
amortised: **261 s at 11 passes, 264 s at 4**. At 68 tokens the diffusion
sampler dominates -- the trunk runs once per seed (4 times) against 20 samples x
~200 denoise steps -- so the trunk share, and any saving, only grows with
length. The recycle question is therefore about COMPARABILITY and semantics, not
speed, and it is still the user's call: the default is unchanged at 10.

## The sweep now covers a PTM, a LIGAND and a DIMER (2026-09-12)

The featurisation diff only ever ran on a plain protein monomer, which certifies
a plain protein monomer -- and the first extension (the PTM case) immediately
found a bug I had just introduced. The other two extensions are now run, against
of3 in its own schema:

| case | tokens | atoms ours/native | atom order | ref_charge / space_uid / element | token ids |
|---|---|---|---|---|---|
| SEP-20 ubiquitin (ATOMISED) | 85 | **605/605** | 605/605 | exact | exact |
| 1STP + BTN (LIGAND) | 137 | **917/917** | 917/917 | exact | exact, plus `is_ligand` and `is_protein` identical |
| ubiquitin homodimer (TWO CHAINS) | 152 | **1202/1202** | 1202/1202 | exact | `asym_id`, `entity_id`, `sym_id`, `token_index`, `residue_index` all identical |

The dimer's 1202 is 2 x 601, which also shows the terminal-atom drop applying
per chain rather than once. The ligand's 16 BTN atoms come out in native's own
order (C11 O11 O12 C10 C9 C8 C7 C2 S1 C6 C5 N1 C3 O3 N2 C4), which is the check
that matters for a CCD-built component: a different CCD read would reorder them.

`ref_pos` differs in all three, by the usual conformer draw, as it does
everywhere.

**All five vendors now run the ligand and the dimer, and all five are clean:**

| vendor | ligand atoms ours/native | dimer atoms ours/native | names | every other field |
|---|---|---|---|---|
| openfold3 | 917/917 | 1202/1202 | 100% | exact |
| opendde | **918/918** | **1204/1204** | 100% | exact |
| intellifold2 | 917/917 | 1202/1202 | 100% | exact |
| rosettafold3 | 917/917 | 1202/1202 | 100% | exact |
| boltz2 | 917/917 | 1202/1202 | 100% | exact |

opendde's counts are one higher PER CHAIN because opendde KEEPS the terminal
OXT where the other four drop it -- and our side matches each vendor, which is
the per-model terminal-atom convention checked on both termini of a two-chain
input rather than on one monomer.

Two dump scripts were under-specified and had to be fixed before they could be
believed:

* `native_rf3_featdump.py` built rf3's pipeline with
  `use_element_for_atom_names_of_atomized_tokens` at its **library default of
  False**, while rf3's own inference engine sets it True
  (`models/rf3/src/rf3/inference_engines/rf3.py:330`). The dump therefore gave
  BTN its CCD names (C11 O11 O12 ...) where rf3 only ever sees `C O O ...`, and
  the ligand case read as "16/917 atom names disagree" -- a phantom bug in our
  correct `atomized_element_names` branch. **A vendor featuriser has to be
  driven with the vendor's INFERENCE overrides, not its constructor defaults**,
  which is the same lesson as building a native from the wrong release.
* `featurisation_diff.py` only stripped the `batch_` dump prefix, so rf3's
  nested `feats.*` names missed every field lookup.


**So of3's featurisation now agrees with ours on four input classes** -- monomer,
modified residue, ligand complex, two chains -- to every field but the conformer.
The remaining featurisation gap is the other vendors on these same three cases:
each has been diffed on a monomer only, and the PTM case is exactly where the
one bug turned up.

## The terminal-atom drop was eating a PHOSPHOSERINE oxygen (2026-09-12)

Caught by extending the featurisation diff to the ATOMISED path -- every run of
it so far had been a plain protein monomer, and modified residues are where atom
layouts get interesting.

of3's own featuriser on SEP-20 ubiquitin: **605 atoms**. Ours: **604** with
yesterday's drop on, **606** with it off. Both wrong, in opposite directions, and
the 604 is the one I introduced: `drop_atoms=('OXT', 'OP3', 'O3P')` drops by NAME
alone, and **O3P is a sidechain atom of phosphoserine**. We removed the terminal
OXT correctly and then removed a phosphate oxygen that native keeps.

Every vendor that drops these drops them from STANDARD residues only, and says
so -- of3's function is literally `remove_std_residue_terminal_atoms` ("terminal
atoms can be kept for any non-standard residues, as they are tokenized per-atom")
and rf3's filter carries `& ~is_atomized`. `_drop_atoms_by_name` now skips any
token holding a single real atom, which is what an atomised residue is.

After: **605 / 605, and 605/605 by name.** Folds: openfold3 6MRR 1.544, ptm 5K9P
1.436; boltz2 6MRR 0.459, ptm 5K9P 2.020.

**The lesson is about the sweep, not the bug.** A featurisation diff run only on
a plain monomer certifies a plain monomer. The same argument that made the
protein-only gates blind to nucleic OP3, and the monomer-only gates blind to
boltz2's cross-chain template visibility, applies to the diff itself: it needs a
PTM case, a ligand case and a two-chain case before "our featurisation matches"
means anything general. One of those three is now run.

## WE RECYCLE 11 TIMES FOR EVERYONE; the vendors do not (2026-09-12, OPEN)

Found while asking what the sweep still misses. Our `num_recycles` is 10 for
every model -- AF3's default, i.e. 11 trunk passes -- and almost nobody else runs
that:

| model | the vendor's OWN default | its trunk passes | ours |
|---|---|---|---|
| `boltz2` | `--recycling_steps` **3** | 4 | **11** |
| `openfold3`, `openbind0` | `num_recycles` **3** (model_config.py:177) | 4 | **11** |
| `protenix1`, `protenix2` | `N_cycle` **4** | 4 | **11** |
| `rosettafold3` | `n_recycles` **5** | 6 | **11** |
| `opendde` | `N_cycle` **10** | 10 | 11 |
| `intellifold2` | `--recycling_iters` **10** (its config says 3; the CLI overrides) | 10 | 11 |
| `chai1` | 3 TOTAL passes | 3 | 10 |

Measured, not inferred, for of3: its own dump carries `num_cycles = 4`.

**This is not a correctness bug -- it is a comparability one, and it cuts three
ways.**

  1. **Fold numbers.** Every L5/L6 row in this repo gives the port ~2.75x the
     recycling the vendor gives itself. More recycling generally helps, so our
     table flatters the ports against the vendors' own behaviour, and a reader
     comparing our `boltz2` row to Boltz's published numbers is not comparing
     like with like.
  2. **Runtime.** [[af3-runtime-benchmarks]] quotes 1.9x at 68 tokens against
     native. If that fold ran 11 passes against native's 4, the comparison
     understates us -- we did nearly three times the trunk work and were still
     faster. Either way the number is not what it says it is, and this has to be
     settled before any comparison against BioIR
     (see the BioNeMo entry below), because that one is explicitly about speed.
  3. **Parity.** Unaffected: the trunk comparisons in this file match cycle
     counts on both sides explicitly (`PASSES=n` against `--model.N_cycle n`),
     which is why they read 1e-6 rather than drifting.

**NOT CHANGED, because it is a user-facing default across eight models.** The
options are (a) set `num_recycles` per model to the vendor's own, so "our boltz2"
means what Boltz means, with the knob to raise it; (b) keep 10 everywhere and
state the multiplier wherever a number is published. (a) is the more honest
default and (b) is the smaller change; either way the runtime benchmarks need
re-running at matched counts.

## The featurisation sweep is COMPLETE: five vendors, one finding, one correction (2026-09-12)

Every vendor whose featuriser can be driven here has now been diffed against
ours, field by field, with the atom axis matched BY NAME first
(`dev/oracles/native_{boltz,dde,if2,rf3}_featdump.py` + `native_of3_dump.py`,
all feeding `featurisation_diff.py`).

| vendor | atoms ours/native | atom order | ref_charge / ref_space_uid / ref_element | token ids, profile, deletion_mean | ref_pos |
|---|---|---|---|---|---|
| `boltz2` | 574/573 -> **573/573** | 573/573 | exact | exact / relabeling / exact | 0.990 A per residue |
| `openfold3`, `openbind0` | 602/601 -> **601/601** | 601/601 | exact | exact | conformer draw |
| `opendde` | **602/602** | 602/602 | exact | exact / relabeling / exact | conformer draw |
| `intellifold2` | **573/573** | 573/573 | exact | exact, and `profile` EXACT (same 31-class width as ours) | conformer draw |
| `rosettafold3` | **573/573** | 573/573 | exact | exact | conformer draw |

Two of those atom counts were ours being wrong (the terminal-atom convention,
above); `opendde`'s 602/602 is the independent confirmation that it KEEPS the
OXT, and `intellifold2`'s and `rosettafold3`'s 573/573 confirm they drop it --
the drop/keep split now rests on each vendor's own featuriser output, not on
reading its tables.

**What the sweep produced, end to end: one finding (the terminal atoms, which
generalised to five models), one correction of my own fix (opendde's empty
template is [31, 0, 0, 0], not all-31), and otherwise a clean bill.** Everything
else -- charges, space uids, element indices, atom ORDER, residue/token/asym/
entity/sym ids, deletion means, MSA profiles -- agrees exactly or differs by a
documented relabeling.

`ref_pos` differs for every vendor, always by the same amount (about 1 A per
residue after alignment) and always for the same reason: the conformer is a
different RDKit draw, not a different molecule. It is priced -- substituting
native's own `ref_pos` into a protenix fold moved it 0.01 A -- and it is the one
input difference that is not worth chasing.

**Three harness faults surfaced inside the tool itself**, each the same shape as
the findings it looks for: comparing padded native slots against our real atoms
(needs the vendor's pad mask), assuming of3's 0-indexed element one-hot for
boltz2's 1-indexed one, and dropping the first axis of every multi-dimensional
array -- which turned opendde's unbatched `ref_pos` (602, 3) into (3,) and read
as "native has 3 atoms". The tool now filters by the vendor's own masks, detects
the element base, trims a padded TOKEN axis, and handles both atom layouts (flat
with a pad mask; dense per token, which is what intellifold2 and we use).

## opendde's featurisation: clean, and it CORRECTS my own template fix (2026-09-12)

Second vendor through `featurisation_diff.py`, driven through opendde's own
`build_inference_config` + `InferenceDataset` (its CLI path, not a
reimplementation), on plain ubiquitin:

| field | result |
|---|---|
| atom count | **602 / 602** -- both keep the terminal OXT, which is the independent confirmation of the split recorded above |
| atom order by name | **602/602** |
| `ref_charge`, `ref_space_uid`, `ref_element` | **exact** |
| `residue_index`, `token_index`, `asym_id`, `entity_id`, `sym_id`, `deletion_mean` | **exact** |
| `profile` | ours 31 classes, native 32 -- a consistent RELABELING (free) |
| `ref_pos` | the conformer draw, as everywhere |

**AND IT OVERTURNS HALF OF YESTERDAY'S opendde FIX.** I had set its empty
template to GAP in ALL FOUR slots, read off `make_dummy_feature`
(`torch.full(..., 31)  # gap` across the whole (4, N) block). Its own featuriser
emits **[31, 0, 0, 0]** -- gap in slot 0, zero in the rest, exactly protenix's
pattern. Measured on BOTH `--use_template true` and `--use_template false`, so it
is not a flag artifact: the all-31 path only runs when the template featurizer
returns nothing at all, and it does not return nothing.

Corrected to `empty_template_gap_slots='first'`, verified by remapping our
`template_aatype` through `_AF3_TO_OF3`: ours [[31], [0], [0], [0]] against
native's [[31], [0], [0], [0]]. Folds after the correction: 6MRR 0.729 (from
0.808 under the wrong pattern, 0.734 before any of this), plain 5K9P 1.592
(1.641 wrong, 1.591 before).

**This is the argument for the feature diff in one line.** I flagged that
opendde's fix rested on a code reading and named the measurement that would
close it. The measurement closed it by disagreeing: reading the source got the
right QUESTION (the empty slot is not zero) and the wrong ANSWER (how many slots
carry the gap). Every other vendor's empty-template convention in the table above
was gated by `template_parity EMPTY=1`, which compares our embedder against
native's on OUR features -- it could not see this, because it feeds both sides
the same slots.

## CLOSED: protenix's bf16 autocast is not something to match (2026-09-12)

Recorded earlier as "an opportunity, not a bug" -- protenix INFERS under
`torch.autocast(bf16)`, so bf16 is arguably part of its convention, and our
`BF16=all` is a different thing (it casts parameters where autocast keeps fp32
masters and casts per-op). Measured now, on the MSA-conditioned 5K9P case, which
is deterministic where the single-sequence one is bistable:

| | vs native fp32 | vs native autocast-bf16 |
|---|---|---|
| native's other mode | -- | corr **0.99829228**, max\|d\| 178.7 |
| ours `BF16=none` | corr **0.99999968**, max\|d\| 3.3 | 0.99829162 |
| ours `BF16=all` | 0.99992155, max\|d\| 32.3 | 0.99824220 |
| ours fp32 arrays + matmul `bfloat16` | -- | 0.99829408 |
| ours fp32 arrays + matmul `tensorfloat32` | -- | 0.99829130 |

**Every one of our modes sits at 0.9983 from native-bf16 -- the same distance
native's OWN fp32 sits at.** Our precision knobs move the trunk by far less than
the bf16 gap, so they all read as "native-fp32-like", and none of them tracks
native's bf16 trajectory. That is not a defect to fix: reproducing a bf16
trajectory means reproducing the op order and the kernel rounding, which is not
achievable across frameworks and is not what parity means. `JAX_DEFAULT_MATMUL_PRECISION=bfloat16`
was the plausible route to autocast's semantics (fp32 arrays, bf16 matmul inputs,
fp32 accumulation) and it changes the sixth decimal.

**And it buys nothing.** The fold on this target, two processes each, bit-stable:
`BF16=all` 1.854 / 1.854, `BF16=none` 1.857. The precision that matters is the
one the deterministic comparison uses, and there our fp32 trunk matches native's
fp32 trunk to corr 0.99999968.

So the standing item is CLOSED rather than fixed, and the general statement is
worth keeping: **bf16 IS the noise floor here, on both sides.** A parity claim
about protenix has to be made in fp32, which is what
`native_protenix_dump.py --dtype fp32` already defaults to, and a fold number
quoted in bf16 is quoting a draw from that floor.

## TO WATCH: NVIDIA's BioNeMo Inference Runtime, for the runtime comparison (2026-09-12)

https://developer.nvidia.com/blog/high-throughput-structure-prediction-with-bionemo-inference-runtime

A PyTorch inference runtime ("BioIR") for **Boltz-2, OpenFold3 and OpenFold2** --
two of the three are vendors this file compares against. Three layers: kernel
selection (BioIR-custom / cuEquivariance / PyTorch fallback, chosen from model
config, GPU, dtype and tensor shape), CUDA Graph capture per module, and Ray
replicas across GPUs. Claimed 1.78x (Boltz-2), 1.55x (OpenFold3), 2.56x
(OpenFold2) on model-forward latency against a `torch.compile` baseline, and
2.90x residue-normalised throughput on 8xH100.

**Not usable here today, for reasons that will not change quickly:** it is
PyTorch (we are JAX/XLA), it is H100/H200 (this box is an A10, cc 8.6, and the
fleet's others are A100s), and BioIR itself is closed source. The one reachable
piece, cuEquivariance, has JAX bindings but is the library already measured in
this project at **1.09x on the A10 and datacenter-only** -- which matches
NVIDIA's own support line.

**Two things to carry forward.**

  1. **The runtime comparison we will want.** Our published numbers are against
     stock native implementations on an A10 ([[af3-runtime-benchmarks]],
     [[runtime-vs-native]]). A fair "how fast is the port" answer on modern
     hardware has to say WHICH native: stock PyTorch, `torch.compile`, or BioIR.
     Against BioIR the honest baseline is 1.5-2.6x faster than the one we have
     been comparing to, and it needs an H100 to run at all.
  2. **AND IT IS A DIFFERENT REFERENCE, NUMERICALLY.** Swapping kernels changes
     the arithmetic. Anyone who compares our port against a BioIR-served Boltz-2
     or OpenFold3 instead of the stock package is comparing against a different
     implementation of the same weights -- which is exactly the failure that cost
     four days on openbind0 (native at `main` over v0.5.0 weights) and that
     `native_of3` now asserts against. If BioIR ever becomes the reference here,
     the adapters need the same kind of guard.

## The featurisation diff, vendor by vendor: boltz2 has NO terminal OXT (2026-09-12)

Every port bug found on 2026-09-11/12 was an INPUT convention, and L0-L4 cannot
see one by construction -- they feed each module NATIVE's own features. So the
systematic answer is to diff the FEATURES, which had been done once (protenix,
2026-09-08) and for nobody else. `dev/oracles/featurisation_diff.py` now does it
against any vendor dump. boltz2 first, on 6MRR:

| field | result |
|---|---|
| atom COUNT | **574 ours, 573 native** -- the extra one is the C-terminal OXT |
| atom order (by name, after the fix) | **573/573 agree** |
| `ref_charge`, `ref_space_uid`, `ref_element` | **exact** |
| `residue_index`, `token_index`, `asym_id`, `entity_id`, `sym_id` | **exact** |
| `deletion_mean` | exact |
| `profile` | ours 31 classes, native 33 -- a consistent RELABELING (free: the converter permutes the consuming Linear) |
| `ref_pos` | differs, 0.990 A per residue after alignment (max 1.54) |

**BOLTZ HAS NO OXT, EVER, and this is not an artifact of the input.** Its
canonical atom table is fixed and does not list one --
`const.ref_atoms["GLU"] = [N, CA, C, O, CB, CG, CD, OE1, OE2]` -- and its own CCD
mol carries OXT flagged `leaving_atom: True`. So we were handing it one atom per
protein chain that the model never saw in training, occupying a slot in the atom
windows and shifting the flat atom axis of every chain after the first. Fixed
with the knob esmfold2 and chai-1 already use (`drop_atoms=('OXT',)`,
`AF3_NO_BOLTZ2_DROP_OXT=1` to A/B).

### The same convention, across five models -- and what it costs

Reading each vendor's own source rather than generalising from boltz2 (the tally
is the finding as much as the fix):

| DROPS terminal atoms | KEEPS them |
|---|---|
| `openfold3`, `openbind0` -- `remove_std_residue_terminal_atoms`, with `MOLECULE_TYPE_TO_LEAVING_ATOMS = {PROTEIN: [OXT], DNA/RNA: [OP3, O3P]}` | `protenix1/2` -- `constants.py` indexes OXT per residue (`"ALA": {... "OXT": 5}`) |
| `boltz2` -- fixed tables (`ref_atoms["GLU"]` ends at OE2, `"A"` starts at P) and a CCD flagging OXT `leaving_atom: True` | `opendde` -- appends it explicitly: `staying_atoms = np.append(staying_atoms, ["OXT"])` |
| `intellifold2` -- forks boltz's tables | |
| `rosettafold3` -- its predict path calls atomworks' `remove_protein_terminal_oxygen` and the OP3 filter | |
| (`chai1`, `esmfold2` already carried the knob) | |

**The nucleic half matters more than the protein half.** OXT is the LAST atom of
a protein residue, so it only displaces the tail; OP3 is the FIRST atom of
residue 1, so carrying it shifted the ENTIRE flat atom axis of every nucleic
chain by one against native's. Our RNA batches started `OP3 P OP1 OP2 O5'` where
every one of these vendors starts at `P`.

A/B on the five (old -> fixed, best of 5, one process each):

| model | 1EHZ RNA | 6MRR |
|---|---|---|
| `openfold3` | 1.328 -> 1.333 | 1.546 -> 1.548 |
| `openbind0` | 1.315 -> 1.318 | 1.645 -> **1.579** |
| `boltz2` | 1.197 -> 1.195 | 0.474 -> 0.468 |
| `intellifold2` | 1.477 -> 1.487 | 1.521 -> 1.512 |
| `rosettafold3` | 1.047 -> 1.049 | 0.985 -> 0.965 |

**The per-process band is about 0.003 A** (openfold3 / 1EHZ repeated under the
SAME setting: 1.335, 1.332 -- two repeats of one cell, so a weak estimate and
quoted as one). Every delta above is within two or three times that except
openbind0's 6MRR, -0.066 and reproducible on a second process (1.579 twice),
which is the only one worth calling real. Signs are mixed and the mean is
-0.006: there is no systematic effect, and an earlier reading of "consistently
slightly worse" from the first three cells was noise, retracted.

**So the case for this fix is not the RMSD.** It is that five vendors' own
featurisers emit 601 protein atoms and 1625 RNA atoms where we emitted 602 and
1626, and of3 states the reason in its own source: "Models like AF3 and AF2
expect all tokens with the same restype to map to the same number of atoms". We
were feeding five models an atom their training never contained. That it costs
nothing measurable on these two targets is a fact about the targets.

`AF3_NO_TERMINAL_DROP=1` restores the old behaviour for every model at once.

`ref_pos` is the conformer draw already documented for protenix (0.90 A per
residue there, 0.99 here) and already priced: substituting native's own
conformers into a protenix fold moved it 0.01 A. Same class, not a bug.

**Two harness faults of mine inside this diff, both of the same shape as the
findings:** comparing 576 native slots against our 574 real atoms without
applying the vendor's `atom_pad_mask` (reads as a missing atom), and assuming
of3/protenix's 0-indexed element one-hot for boltz2, whose base is 1 (reported
100% of elements wrong). The script now filters by the vendor's pad mask and
DETECTS the element base instead of assuming it, and treats a width-mismatched
categorical as a relabeling question rather than comparing the first k columns of
two different vocabularies.

## boltz2 runs TemplateV2, and V2 does not mask by chain (2026-09-12)

Found by auditing the native adapters themselves rather than our port -- the last
two faults were adapters lying to native, so the deep dive started there.

`template_parity.native_boltz2` imports **TemplateModule**. The checkpoint says
otherwise: `hyper_parameters['use_templates_v2'] = True` (and `use_templates` is
True, so the module runs on every forward). Diffing the two classes in
`trunkv2.py` shows they are the same modules, same shapes, same aggregation --
81 template tensors load into either -- and differ in exactly ONE input:

    V1   asym_id          asym_mask = (asym_id_i == asym_id_j)
    V2   visibility_ids   vis_mask  = (visibility_ids_i == visibility_ids_j)

and `visibility_ids` is **not** asym_id. `featurizerv2` sets it to the
TEMPLATE'S PDB ID for every chain that template covers, and to `-1 - asym_id`
for chains with no template. So under V2 two chains templated from the same
structure SEE EACH OTHER, which is the entire point of giving a complex a complex
template; an untemplated chain still sees only itself.

We masked by asym_id for every model, boltz2 included. **On a shared template
that is the whole cross-chain block: 11552 of 23104 pairs, 50% of the pair map,
zeroed.** Our features express the right thing directly -- handing the same
template to both chains produces ONE row covering both (measured: 152 tokens
across chains {1, 2} for a ubiquitin homodimer, and 164 tokens across chains
{1, 2} in the 1LMB screen) -- so the row's own coverage IS the visibility group,
and `TEMPLATE_VISIBILITY_BY_COVERAGE` builds the mask per row from it, falling
back to same-chain for uncovered tokens (which is what `-1 - asym_id` means).

**Fold effect on the one templated complex we can score: none.** 1LMB with a
shared single-chain template reads 0.303 with the old mask and 0.304 with the
new one -- boltz2 already folds that complex at 0.387 with no template at all, so
there is no room for the interface block to help. The fix is faithfulness to the
convention the checkpoint declares, and it is live on that input (the row covers
both protein chains), not dormant.

**Why no gate could see it.** `folding_input.Template` is per chain and
`template_parity` folds a single one, so the cross-chain block is identically
zero on both sides there -- the gate reads corr 1.000000 before and after this
change, which is correct and uninformative. Same blind spot, third time: the
padded-window work was verified on RNA, openbind0's pair bias against the wrong
release, and this against a monomer. A convention that only exists BETWEEN chains
cannot be gated on one chain.

`AF3_NO_BOLTZ2_TEMPLATE_VIS=1` restores the old mask, and `TEMPLATE=<cif>:<chain>`
in `modality_check` gives every protein chain the same template, which is how the
complex case above was run.

## The empty-template convention, now GATED for all five (2026-09-12)

The template half of the sweep was settled by reading each vendor's code. That
was the weakest evidence in the whole protenix episode -- opendde's fix rested on
it -- so `template_parity.py` grew an `EMPTY=1` mode and it is now measured. The
mode builds the batch with NO template at all (so each slot carries whatever the
vendor's convention puts there) instead of zeroing a self-template's coordinates,
which would leave the QUERY's restypes in slot 0 and test nothing.

| model | empty slot | native vs ours, no template supplied |
|---|---|---|
| `protenix1` | slot 0 GAP, 1-3 zero | **corr 1.000000**, term rms 17.10 |
| `opendde` | ALL FOUR gap | **corr 1.000000**, term rms 6.62 |
| `intellifold2` | zeros | **corr 1.000000**, term rms 9.46 |
| `rosettafold3` | zeros (single unconditional pass) | **corr 1.000000**, term rms 17.66 |
| `boltz2` | zeros, and MASKED | **both exactly zero** |

So opendde's gap-restype fix is no longer a code reading -- it is an activation
comparison, and so is the DIVISOR: aggregating four identical empty slots equals
the single-slot result to max|d| 6e-05, which is what `sum / slot count` means
and what would fail loudly if either side divided by the number of PRESENT
templates.

**boltz2 is the one vendor that contributes nothing, and it is worth being
precise about why.** Both its TemplateModule and TemplateV2Module do
`u = (v * template_mask).sum(dim=1) / num_templates.clamp(min=1)` -- the
per-slot output is MASKED before the average, so an absent template contributes
exactly zero even though the Z-dependent half was computed. Every other vendor
here divides by the slot count without masking, which is why their term survives.
Its checkpoint also declares `use_templates=True`, so the module runs on every
forward regardless; it just returns zero.

**One harness fault found on the way, and it read exactly like a port bug.** The
first EMPTY run reported boltz2 native at rms 2.180 against our 0 -- a
protenix-shaped finding. It was `native_boltz2` hardcoding
`'template_mask': torch.ones(1, 1, N)`, i.e. telling boltz an all-zero template
was PRESENT. The mask is now derived from the features
(`(atom_mask.sum(-1) > 0)`), which is all-ones with a real template, so the
existing gate is unchanged: boltz2, intellifold2 and opendde all still read corr
1.000000 there.

## An external report on our opendde path: 2 of 3 confirmed (2026-09-11)

From chlee19990109-cloud, whose own Protenix port ([[teammate-protenix-port]])
hit the first of these in their tree. Verified against ours rather than taken on
trust; one half of one claim does not survive.

### 1. CONFIRMED -- the structural-token atom axis is permuted at a non-glycine terminus

Their probe reproduces here exactly:

    dense, residue-major   N CA C O CB SG OXT
    our struct flat order  N CA C O OXT | CB SG

`PROTEIN_BACKBONE_ATOMS` includes OXT (structural_features.py:22, copied from
opendde's tokenizer.py:21), rows are emitted backbone-token-then-sidechain-token,
and the atom axis is the row-major flattening of those rows. There is no
reordering step.

**The mechanism, which their report states and I confirmed at the source:**
native opendde keeps every atom in the ORIGINAL atom array and has each token
carry `atom_indices` into it -- `_get_atom_to_token_idx` fills an array indexed
by atom i over `range(n_atoms)` (data/core/featurizer.py:414). Its atom axis is
therefore residue-major, and the tokens are gathers. Ours re-packs atoms into
(token, slot) rows, so our axis is token-major. At a non-glycine terminus the
backbone set is not contiguous in the residue's atom order and the two axes
differ by a 3-cycle.

**Scope, measured here (`dev/oracles/struct_atom_axis_probe.py`).** What matters
is not that atoms move but whether a 32-atom QUERY BLOCK straddles the terminal
residue: inside one block a permutation is invisible, because every per-atom
feature travels with its atom.

| input | atoms | moved | query blocks whose SET differs |
|---|---|---|---|
| ubiquitin (ends GLY) | 602 | **0** | 0/19 |
| 6MRR | 574 | 6 | 0/18 |
| 1STP (121 res) | 902 | 4 | 0/29 |
| two chains, TRP and CYS termini | 212 | 14 | 0/7 |
| poly-A(12, 18, 19, 24, 25) ending TRP | 70-135 | 11 | **2** each |

So: real, and INERT on every target we currently fold, LIVE for roughly a third
of chain lengths. Their note about the blind spot is right and worth repeating --
our padded-window work was verified against native `pad_info` on 1EHZ, which is
RNA, and the nucleic CCD order puts the backbone atoms first so no permutation
arises. Ubiquitin is blind for a second reason: it ends in GLY, which takes the
single-token path and is never split at all.

**FIXED (2026-09-11), by decoupling the atom axis from the token rows.** The
first read of this was that our dense (token, slot) layout makes the axis
token-major by construction, so matching native meant surgery. It is smaller
than that, because AF3's own machinery already does the work: every gather in
`AtomCrossAtt.compute_features` is computed by MATCHING LAYOUTS, not by
arithmetic on positions. So permuting ONE array -- the flat atom list, before the
queries and keys are cut from it -- moves the whole axis and every gather follows.

  * `build_structural_layout` now also returns `flat_atom_order`: for the flat
    atoms in ROW-MAJOR order, their rank in residue-major order. It has the
    information to do this and nothing else does -- `rows_src` records each
    structural atom's (parent residue, dense slot).
  * `AtomCrossAtt.compute_features` takes an optional `flat_atom_order` and
    applies it to `flat_layout`. Residue path unaffected (it passes None, and its
    axis is already residue-major because a token IS a residue there).
  * `AF3_NO_STRUCT_ATOM_AXIS=1` restores the old axis for an A/B.

Verified: the probe (which now reads the QUERY axis, reconstructed through
`token_atoms_to_queries` -- the (token, slot) layout stays token-major by design
and is not the thing that had to move) reports `moved 0` on every case including
the five that were live, and reports them live again with the toggle set. The
opendde gates stay exact -- L2.diffusion / conditioning / atom_encoder /
atom_decoder all corr 1.000000, L3.denoise 0.0000 A per atom -- and 6MRR folds to
0.809 with the fix and 0.809 without, which is what "inert on this target" is
supposed to mean.

### 2. HALF CONFIRMED -- the MSA depth is 1280, but our shuffle is already valid-first

**Depth: right, and fixed.** opendde's `MSAModule.forward` subsamples to
`num_msa=self.msa_depth`, and `msa_depth` is 1280 (config/data.py:31). We ran
AF3's `num_msa=1024` with no per-model override. Now set in OPENDDE_SETTINGS
(`AF3_DDE_NUM_MSA` overrides it for an A/B). It can only bite on an MSA deeper
than 1024 -- every gate here has run self-MSAs of depth 1-2 -- and on the one
deep-MSA target we have it changes nothing: 1STP (2144 rows) reads CA 0.306 at
1024 against 0.301 at 1280, BTN 0.886 both.

**"A uniform shuffle where opendde takes valid-first": NO.** Our `shuffle_msa`
sorts by `logits = (clip(sum(msa.mask, -1), 0, 1) - 1) * 1e6`, i.e. 0 for any row
with an unmasked position and -1e6 for a padded one, then gumbel-argsorts. Rows
with content sort ahead of padding by construction -- that IS valid-first, and
the function's own comment says "Sample uniformly among sequences with at least
one non-masked position". Same semantics as their
`subsample_msa_feature_dict_valid_first`, different implementation.

### 3. CONFIRMED and inert, for the reason they give

`offsets_valid` takes the `keys_mask` term only for `OPENFOLD3_LINEAGE`
(atom_cross_attention.py:571-574), so on the protenix/opendde path a padded key
(ref_space_uid 0) collides with token 0 in term 1 of the atom-pair conditioning.
Inert because `opendde` and `PROTENIX_FAMILY` are both in
`KEY_MASKED_ATOM_ATTENTION`, which zeroes those slots at the attention. Left as
is, now written down.

## YES, an MSA stabilises it -- and with one we MATCH native (2026-09-11)

The bistability above is a single-sequence effect on this target. Given a real
MSA it disappears, on both sides, and the port agrees with native:

| protenix1, plain 5K9P | ours | native |
|---|---|---|
| single sequence | 1.53 **or** 11.4 (flips per process) | 1.85 **or** 10.1 (flips per process) |
| + a 4-row MSA | **1.83** (1.83-2.35; stable over 3 seeds and 4 processes) | **1.87** (1.87-2.19; stable) |

The MSA is REAL, not synthesised: three distinct ubiquitin sequences pulled out
of the bundled mini-databases (`uniref90__subsampled_1000.fasta`'s polyubiquitin
repeats and `pdb_seqres`'s 1otr_B), 1-3 differences from the query each. Shallow
and low-diversity, so it adds little evolutionary signal -- and it still removes
the flip, which says the flip is about how weak the conditioning is, not about
MSA information as such.

Bistability is also NOT a property of single-sequence input in general: 1STP
folded from its sequence alone is stable to seven digits across processes, and
6MRR is stable for everyone. It is this target, for this lineage.

The trunk and the denoiser agree with native throughout, on the SAME MSA:

    trunk, 1 cycle     corr 0.99999906      trunk, 10 cycles  corr 0.99999968
    denoise step 1     0.0201 A/atom (sigma 4608)
    denoise step 100   0.0079 A/atom (sigma 106)
    denoise step 190   0.0007 A/atom (sigma ~0)

and every per-atom feature is identical except `ref_pos`, whose conformer
difference -- the one PARITY.md's featurisation diff has carried as an open
"real input difference" since 2026-09-08 -- prices at **0.01 A** on this fold
(substituting native's own ref_pos moves the result from 4.026 to 4.036).

### The 2.3 A "gap" I chased for an hour was MY SCORER

Before the numbers above, the with-MSA case read 4.02 for us against native's
1.87, stable on both sides, and I went down the whole ladder looking for it:
MSA features (identical), trunk (1e-6), atom features (identical), sampler
constants (identical), the denoise step at three sigmas (0.0007-0.02 A/atom),
protenix's per-forward random MSA subsampling (real -- `sample_indices` draws
randint(1, n) rows at INFERENCE, so its trunk is stochastic per pass; patched
off for the comparison), three seeds, four processes.

All of that evidence said there was no gap. The fold number was the outlier, and
it was wrong: my scoring script mapped our `residue_index` to the reference with
a **+1**. Our residue_index is already the reference's numbering, so every
residue was compared against its neighbour -- on a compact 76-mer that is ~2 A
after superposition, and it reads exactly like a bad fold (median per-residue
deviation 3.8 A, and a confident pLDDT of 90 next to it). With the offset
removed: 1.83.

**The tell was in the output from the first run: 75 CA matched where the native
scorer matched 76.** A count that does not match the reference is a mapping
error until proven otherwise -- this is the second time this project has been
bitten by exactly this class (PARITY.md records matching by POSITION scoring
ubiquitin at 12.4 A on every model), and the first time it cost an hour of
ladder-climbing instead of being caught by the count.

`dev/oracles/fold_from_json.py` now prints an OFFSET SWEEP (-1, 0, +1, +2) with
every score, so the mapping has to declare itself. The same +1 was in
`fold_with_native_trunk.py` and is fixed there too; its injection numbers from
earlier today were inflated by it (openbind0 INJECT=pair read 4.83 and is really
2.47, with INJECT=none at 2.409). Those were only ever used as a hedge ("a
foreign representation costs accuracy of its own"), and the conclusion they
supported was carried by the sub-module measurement instead -- but the numbers
were wrong and are corrected here.

## CORRECTION 2026-09-11: plain 5K9P is BISTABLE, and two of my attributions were draws

The fold RMSD on this target is a PER-PROCESS DRAW from two basins, for both
implementations, with the same seed and the same precision. Measured, protenix1,
identical command repeated:

    ours   11.362  11.362  1.555  11.362  ...      (11.362 reproduces to 3 decimals)
    native  1.854  10.139  1.854  1.848

The two outcomes are each reproducible WITHIN a process to three decimals and
flip BETWEEN processes, which is the signature of XLA autotuning choosing one of
two kernel plans per process (and something equivalent on native's side), not of
continuous noise. It is also why the driver's cache-warmed cells report one
value consistently: a cached executable is one fixed plan.

TWO THINGS I ATTRIBUTED WRONGLY BEFORE MEASURING THE REPEAT:

  1. **"protenix1 folds 1.583 at the shipped matmul default and 11.362 at
     `highest`."** Both numbers are real; the cause is not. Repeating either
     configuration gives both outcomes. I had started changing the DRIVER on
     this basis (per-level matmul precision) -- reverted, and the driver keeps
     `highest` everywhere.
  2. **"protenix2's plain 5K9P is decided by precision: native bf16 6.999,
     native fp32 12.342."** Also a draw: native protenix2 bf16, repeated, gives
     8.952 / 12.148 / 11.503, and ours gives 12.183 / 12.183 / 12.683. Both
     sides span 7-12.5 on this target. The structure-to-structure numbers in
     that entry stand as measurements (ours sits ~1 A from native's fp32 samples
     and 4-10 A from its bf16 ones) -- what was wrong is calling the mechanism
     precision rather than basin choice, and those two dumps were simply in
     different basins.

I also briefly read the vendor PYTHONPATH overlay as the lever (11.738 with,
1.530 without). Same coin. The probe that settled it took four runs and prints
a coordinate hash: two configurations x two repeats, and the two outcomes
appeared in BOTH configurations.

**WHAT STILL STANDS, and why.** The template + MSA-depth fixes are supported by
the TRUNK, which is deterministic and was measured against native's own
tensors: per-cycle 0.99973 / 0.98542 / 0.93123 before, and 0.99999891 /
0.99999545 after, with the MSA-depth term the seed of the compounding. The fold
evidence is that BEFORE the fix every run of several was 8.97-12.89 -- the good
basin never appeared -- and after it, it does. That is a frequency change, not
the deterministic 10.98 -> 1.53 jump the earlier entry reads as. Both basins'
VALUES agree between implementations (ours 1.53-1.60 and 11.4-12.7, native 1.85
and 10.1), which is the best available statement of agreement on a bistable
target.

**THE RULE, since this is the third time a fold number has misled here:** on a
target where the samples cluster far from the reference, repeat the WHOLE
PROCESS before attributing anything to a code or configuration change. A
5-sample spread inside one process is not the band; the band is across
processes. And prefer a deterministic activation comparison (trunk, module) as
the evidence -- fold RMSD is the thing being explained, not the measurement.

## SOLVED 2026-09-11: protenix's empty template and its depth-1 self MSA

Two INPUT conventions, worth 9.4 A on plain ubiquitin, and neither moved a
single module gate: every protenix1 cell from L0 to L4 reads corr 1.000000,
because each one is fed NATIVE's own features. L1-L4 gate module MATH; they
cannot see what we hand the modules in a real fold.

Native protenix1 folds plain 5K9P to 1.855 A. Ours read 10.983.

**1. THE EMPTY TEMPLATE IS NOT A ZERO TEMPLATE.** Both pipelines pad the
template axis to 4 slots and mask every atom, so the distogram, the unit vector
and both masks come out zero either way. The RESTYPE one-hot does not:
protenix's featuriser fills its one empty template with the GAP class and
zero-pads the other three -- measured, `template_aatype` slot 0 all 31, slots
1-3 all 0 -- and `TemplateEmbedder.forward` divides by `num_templates`, the
PADDED slot count, not by the templates present. So its term is never zero:

    v = z_proj(z_norm(z)) + a_proj(a_tij)

keeps a Z-dependent half even with every template feature masked out, and that
half reaches the trunk on every recycle. We had Boltz's convention (mean over
PRESENT templates, hence exactly zero with none) for the whole family.

THE CONFIRMATION, which leaves no room: disabling the same term in NATIVE
(`TemplateEmbedder.forward = lambda *a, **kw: 0`) takes native to 8.40 A, onto
our number. Changing the reference and nothing else is the strongest form this
kind of claim can take.

**2. PROTENIX'S SELF-MSA IS ONE ROW, NOT TWO.** Its dump carries `msa` at
(1, 76). AF3 concatenates a paired and an unpaired MSA, so a chain with no
alignments arrives with the query TWICE. The outer product mean over duplicates
is unchanged, but the pair-weighted averaging and the row transition are
depth-sensitive -- the fault esmfold2 already had, so the same knob
(`dedupe_self_msa`). This was the SEED of the compounding divergence: at cycle 1
it showed up as rms ours/native 0.991 at the MSA stage and nowhere else.

Trunk against native per cycle, on the real input:

| cycle | before | after the template fix | after both |
|---|---|---|---|
| 1 | 0.99973 | 0.99973 | **0.99999891** |
| 2 | 0.98542 | — | — |
| 3 | 0.93123 | — | **0.99999545** |

**THE REFERENCE HAS TO BE fp32.** protenix autocasts to BF16 by default
(`configs_base` "dtype": "bf16", `runner/inference.py:214`), and its own
bf16-vs-fp32 trunk differs at corr 0.789 on this target with max|d| 676 -- 20x
the disagreement we were trying to explain. Against the bf16 dump our
pairformer read 0.9996 on native's real input; against fp32 it reads 0.99999982
and is exact. `native_dump.py` therefore passes `--dtype fp32` by default.
([[oracle-parity-confounds]] says the same thing about protenix2; it cost six
false leads then and would have cost more here.)

Folds after the fix, native in brackets:

| case | before | after | native protenix1 |
|---|---|---|---|
| plain 5K9P | 10.983 | **1.532** | 1.855 |
| 1STP BTN | 0.917 | **0.465** | |
| 1LMB DNA | 1.692 | 1.543 | |
| 6MRR | 1.684 | 1.565 | |
| 1EHZ RNA | 1.707 | 2.290 | **2.440** |
| ptm 5K9P | 2.697 | 11.326 | **11.771** |

**The last two rows are not regressions, and both had to be measured to say so.**

  * 1EHZ: we were accidentally BETTER than native while wrong, and are now
    faithful (2.29 against native's 2.44). Same shape as esmfold2's 1QYS.
  * ptm 5K9P: native protenix1 fails phospho-ubiquitin too, at 11.771 against
    our 11.326. The old "2.697" was one luckier sample set of five -- exactly
    the trap [[protenix2-5k9p-retraction]] records. Ruled out as ours twice:
    once by the A/B (AF3_NO_PX_TEMPLATE_GAP / AF3_NO_PX_DEDUPE_MSA move it by
    0.3 A in either direction) and once by native.

WHAT FOUND IT, since the module gates could not: native's real trunk dumped
per cycle (`px1/native_dump.py`, hooked at `get_pairformer_output` plus taps on
the template embedder, the MSA module and the pairformer stack), our stack fed
native's real input (exact -> the fault is upstream), and then NEW GENERIC-PATH
TAPS in `evoformer.py` (`z_init_generic`, `z_after_template`, `z_after_msa`,
`z_before_prev`, `z_after_prev`). The old taps only covered the pair-only
ESMFold2 branch. With them, one run says: z_init exact (0.99999681), z after
template corr 0.876 -- the stage named in a single reading.

TWO HARNESS FAULTS OF MINE ON THE WAY, both of which briefly read as findings:

  * comparing our FIRST pass against native's LAST cycle (`taps[name][0]`
    instead of `[-1]`), which made the cycle-2 recycle term look 37% too small.
  * carrying `pair_pre_coda` in the harness's `prev` dict for a STOCK model.
    `Evoformer` reads `prev.get('pair_pre_coda', prev['pair'])`, so the recycle
    read a tensor nothing ever updated -- zeros -- and the pair rms was
    identical to 8 digits across passes. That one read as "our recycling is
    inert", which would have been a serious bug had it been true.

## protenix2 under the same two fixes: five cases better, and ubiquitin CLOSED

Both conventions are CONFIRMED for protenix2 by its own native dump -- 4
template slots with slot 0 all gap (31) and slots 1-3 all 0, `msa` at (1, 76),
identical to protenix1 -- and with both knobs on, its trunk against native is
the most exact it has ever been:

    z_init      corr 0.99999993      (one cycle, real input, fp32 reference)
    template    corr 0.99999998
    msa         corr 0.99999997
    trunk out   corr 0.99999953   max|d| 0.93 on rms 27.5

Turning EITHER knob off makes every row worse, which is the check that the
conventions are protenix2's too and not protenix1's alone.

L6 before -> after (2026-09-10 driver row -> re-measured):

| case | before | after | native protenix2 |
|---|---|---|---|
| ligand_1stp BTN | 1.190 | **0.471** | |
| dna_1lmb | 2.068 | **1.815** | |
| rna_1ehz | 1.758 | **1.374** | |
| complex_1lmb | 17.572 | **15.700** | |
| ptm_5k9p | 7.933 | **2.312** | |
| protein_6mrr | 0.691 | 0.983 | |
| **plain_5k9p** | 7.476 | **12.684** | **6.999** |

**plain_5k9p is CLOSED, and it is not a port bug: the target is BISTABLE for
protenix2 -- on both sides.** (Read the CORRECTION entry above first: the
mechanism below is stated as precision and it is not. Repeating either
configuration produces either outcome; native bf16 alone gives 6.999 / 8.952 /
12.148 / 11.503 across processes. The numbers are measurements; "fp32 vs bf16"
as the CAUSE is retracted.)

    native protenix2, bf16 autocast (its default)   6.999 A
    native protenix2, fp32                         12.342 A
    ours (BF16=none 12.304, BF16=all 12.028)       12.3-12.7 A

And the structure-to-structure comparison, which is the port claim rather than a
score against the crystal:

    ours vs native-fp32   0.711 / 1.053 / 1.337 / 1.366 / 1.756 A
    ours vs native-bf16   4.175 / 7.946 / 7.950 / 8.088 / 9.967 A

We reproduce native-in-fp32 to about 1 A and are nowhere near native-in-bf16. So
the port is faithful and the 6.999 is native's bf16 arithmetic landing in a
different basin on a chaotic target -- protenix2's OWN bf16-vs-fp32 trunk differs
at corr 0.854 with max|d| 434 after 10 cycles, while our trunk sits at corr
0.994 / max|d| 74 against its fp32 reference, i.e. well INSIDE its own
precision band.

**What this leaves as a real opportunity, not a bug.** protenix INFERS under
torch autocast bf16, so bf16 is arguably part of the convention rather than a
degradation of it -- worth matching on principle, though NOT worth 5 A here:
that figure came from comparing two draws. Our `BF16=all` is a
different thing -- it casts parameters, where autocast keeps LayerNorm, softmax
and accumulation in fp32 -- and it does not reproduce the basin (12.028).
Matching torch's autocast semantics is an open question for the whole family,
and this is the first target where it demonstrably matters.

The old 7.476 we used to print was, as suspected, a different wrong answer: with
the template term missing the trajectory differed and happened to land near
native's bf16 number. Two wrongs pointing the same way is exactly what
[[protenix2-5k9p-retraction]] warns about.

What was established on the way:

  * it is not the conventions -- the trunk is exact to 1e-7 at one cycle WITH
    them, and worse without either.
  * native protenix2 is itself bad here (6.999 A), so this is a target the
    checkpoint does not solve, not a fold we are failing to reproduce well. Our
    7.476 before the fix was not a better port, it was a different wrong answer
    on a weak-signal target -- the precise trap [[protenix2-5k9p-retraction]]
    records, where one lucky seed passed for a finding.
  * protenix1, same conventions, same code, lands at 1.532 against its native's
    1.855. So nothing family-wide is left broken.

Both of the "next measurements" were then made: the 10-cycle trunk comparison
(corr 0.994 against a floor of 0.854, so inside the band) and the fold in both
precisions on both sides. They are one command each -- `native_protenix_dump.py`
with CYCLES=10 and DTYPE, `native_protenix_fold.py` with DTYPE, and
`score_native_cif.py` pointed at two PREDICTIONS rather than at the crystal,
which is what turned a 5 A "gap" into a 1 A agreement.

## SOLVED 2026-09-11: openbind0's end-node pair bias was transposed. 10.4 -> 2.4 A

One axis swap, and an oracle that certified the wrong side of it.

**The convention.** Both OF3 releases hand the end-node triangle attention an
ALREADY-TRANSPOSED pair, then take the bias from a permutation of `linear_z`
over that transposed tensor:

    main / preview-2   permute(lz(zT), (2, 0, 1)) -> b[h,i,j] = lz(z)[j,i,h]
    v0.5.0 (openbind)  permute(lz(zT), (2, 1, 0)) -> b[h,i,j] = lz(z)[i,j,h]

We build the bias from the NON-transposed pair, so preview-2 needs our swap and
v0.5.0 needs none. `openbind0` is out of `TRANSPOSED_COLUMN_PAIR_BIAS`.

**The oracle bug.** This was settled the other way on 2026-09-07, and the
activation comparison that settled it ran against the WRONG RELEASE.
`~/openfold-3` sits at main, and main's `PairFormerStack` loads openbind0's
pairformer tensors without complaint -- the releases differ by name only in the
diffusion LayerNorms (4890 vs 4936 tensors) -- so `trunk_parity` ran main's
convention over v0.5.0's weights and reported z 1.000000 for the wrong setting.
**A native reference has to be the right RELEASE, not just the right
repository.** The driver now routes openbind0 to `~/openfold-3-v050` and
`native_of3` asserts on whether `TriangleAttention.forward` takes
`transpose_bias`, so the wrong tree is an error rather than a green gate.

**The ladder that found it**, in order, because no step of it was optional:

  1. A native openbind0 FOLD (2.383 A) against ours (10.388), with native
     openfold3 (1.308) and our openfold3 (1.391) as the control that proves the
     harness. Until this existed every conclusion here was an inference.
  2. Native's real trunk tensors, dumped from that fold at `run_trunk` and at
     every sub-module (`native_of3_dump.py`).
  3. Our pairformer stack on native's real `pf_in` -- corr 0.99682 for
     openbind0, 0.99999976 for openfold3.
  4. **Native's own floor on a FIXED input** (`native_of3_pf_floor.py`): tf32
     vs tf32-off, max|d| 2.10 over 48 blocks and 0.09 over one. This is the step
     that made the rest mean anything -- measured against the RECYCLED trunk's
     floor instead, our error looked like noise, because openbind0's recycling
     is genuinely chaotic (a tf32 flip moves its z by max|d| 200 on rms 149, and
     still folds to 1.693 A, so that chaos is not what was wrong).
  5. Per sub-module, each on native's own input: four of five pair modules exact
     to 2e-5 for both releases, and `pair_attention2` corr 0.93 for openbind0 --
     then corr 0.93 for openfold3 with the setting flipped. Exactly
     complementary, which is what a per-release convention looks like.

**Why nothing cheaper could have found it.** Neither setting changes any weight,
so no shape gate and no coverage audit can see it (`audit_coverage` reports 0
unaccounted for either way). The random-input trunk cell grades FLOOR. Folding
6MRR and 1STP could not discriminate it -- two seeds disagreed about which
setting was better. And `fold_with_native_trunk.py` showed the fold recovering
only to 4.8 A on an injected native pair, which is a reminder that injecting a
foreign representation costs accuracy of its own and is not a clean bound.

Measured after the fix, against the numbers this entry was written about:

| case | before | after | native openbind0 |
|---|---|---|---|
| plain 5K9P | 10.388 | **2.407** | 2.383 |
| ptm 5K9P | 11.537 | **1.776** | |
| 6MRR | 1.651 | 1.641 | |
| 1EHZ RNA | 1.495 | 1.315 | |
| 1LMB DNA | 2.209 | 2.352 | |
| 1LMB complex | 16.895 | 14.309 | |
| 1STP BTN | 0.426 | 0.445 | |
| L1 trunk | FLOOR | s corr 1.000000 / z corr 1.000000 | |

openfold3 is unchanged throughout (fold 1.392, trunk s/z corr 1.000000).

What in the entry below still stands: every exclusion in it was correct, and the
distogram was the right suspect (contact precision 0.303 -- the pair track WAS
the broken one). What did not: "consistent with the checkpoint's own behaviour",
and "the trunk cell cannot resolve this". The cell could not resolve it on
RANDOM inputs. On native's real input, with native's own floor measured, it
resolved it to a single sub-module.

## openbind0 fails ubiquitin at 10.4 A, and the matrix CANNOT say why

L6 turned this up: openbind0 folds 5K9P (plain ubiquitin, 76 res) to 10.4-12.7 A
across all five samples, where openfold3 on the identical input reaches 1.4 A.
The two share every line of our code and differ only by checkpoint
(of3-ob-174k vs of3-p2-155k).

RULED OUT, each by measurement:

  * not the MSA -- with NO MSA at all it reads 10.387, against 10.388 with the
    self-MSA the L6 cell uses.
  * not systematic -- openbind0 folds 1QYS 1.230, 6MRR 1.651 and 1STP 0.498
    (ligand BTN 0.426), all comparable to openfold3.
  * not a mirror -- allowing reflection in the superposition changes nothing
    (10.382 against 10.392), so it is not a handedness or chirality sign bug.
  * not collapsed or exploded -- Rg 11.91 A against the reference's 11.18 A, so
    the structure is compact and correctly sized. It is simply a different fold.
  * not the DIFFUSION SAMPLER -- the trunk's own distogram is already wrong:
    top-L contact precision 0.303 and P(contact | true contact) 0.169, against
    openfold3's 0.868 and 0.839 on the same sequence.
  * not numerically degenerate -- its contact_probs are statistically ordinary
    (max 1.000, mean 0.112, 8.9% above 0.5); they just point elsewhere.

AND THE MODEL KNOWS. Mean pLDDT 50.5 with mean PAE 11.12, against 74.0 / 5.08
for the same model on 6MRR and 71.6 / 5.08 for openfold3 here. A broken input
path usually yields a CONFIDENTLY wrong structure
([[postcutoff-generalisation]]); this one reports its own failure.

**WHAT IS ACTUALLY BROKEN IS THE GATE.** openbind0's trunk is the one module the
failure points at, and it is exactly the module the matrix cannot verify: both
its trunk cells grade FLOOR. Worse than previously recorded -- the cell cannot
be rescued by scaling the input, because the OF3 block's output magnitude is
INPUT-INDEPENDENT: rms 447 at INPUT_SCALE 0.5, 445 at 0.05, 447 at 0.005. The
block is bias-dominated, so no synthetic input reaches a regime where the cell
resolves. `INPUT_SCALE` was added to test this and the answer is that it does
not help.

So the honest state: the failure is real, reproducible, localised to the trunk,
and CONSISTENT WITH being the checkpoint's own behaviour -- but our openbind0
trunk has never been verified against native's on a real input, and cannot be by
any cell that exists.

NEXT, and it is the same unrun gate as before: `real_trunk_parity.py` compares
the trunk on a REAL input, which is the only regime where the OF3 family's cell
has resolution. It needs an of3 equivalent of `native_trunk_dump.sh` (that one
dumps protenix). Both checkpoints are on disk and `openfold3/run_openfold.py`
has a CLI. (The "needs GPU torch, which ~/venv does not have and must not get"
that stood here was half right: ~/venv must not get it, and did not -- see the
ANSWERED entry below, which ran the fold from a separate venv.)

### ANSWERED 2026-09-11: native openbind0 folds ubiquitin to 2.38 A. OURS IS A BUG.

The measurement this entry kept asking for, finally made. Native PyTorch
openbind0, its own featuriser, its own runner, single sequence, five samples:

| | plain 5K9P (CA-RMSD, best of 5) |
|---|---|
| **native openbind0** (`of3-ob-174k.pt`, v0.5.0 tree) | **2.383** (2.383-2.531) |
| **our openbind0** | **10.388** (10.4-12.7) |
| native openfold3 (`of3-p2-155k.pt`, main tree) | 1.308 |
| our openfold3 | 1.391 |

The control is what makes it decisive: our openfold3 reproduces native openfold3
(1.391 vs 1.308, inside sampling noise) on the SAME target, through the SAME
code. So the harness, the scoring and the featurisation are all fine, and the
4.4x gap on openbind0 is OURS. Everything this entry ruled out was ruled out
correctly -- and the conclusion those exclusions pointed at, "consistent with
the checkpoint's own behaviour", was WRONG. Six ruled-out alternatives are not a
measurement, which is why that was written down as an inference and not a
finding.

Two notes, both load-bearing for the next person:

  * **`~/openfold-3` at main CANNOT load openbind0.** `load_state_dict(strict=True)`
    fails with 24 missing `blocks.N.attention_pair_bias.layer_norm_z.weight` and
    one unexpected `diffusion_transformer.layer_norm_z.weight`, twice over (the
    file carries two copies). That is exactly the convention split
    `model_config.PER_BLOCK_PAIR_LAYER_NORM` already records -- so no patch is
    needed, only the right tree: `git worktree add ~/openfold-3-v050 v0.5.0`,
    whose tag IS the openbind release (`Merge pull request #375 from
    aqlaboratory/release/openbind`). It loads strict and runs.
  * **of3 native inference needs a GPU, full stop.** Forcing
    `pl_trainer_args.accelerator: cpu` gets through weight loading and then dies
    in the sampler with `0 active drivers ([]). There should only be one.`. On
    the GPU the whole 5-sample job takes **15 s**, so it never needed the CPU --
    it was run alongside a live L6 cell holding 17.2 of 23 GB and fit anyway.

The environment, since the symlink dir was not enough (`pdbeccdutils` is real
work, not a shim, and `lmdb` / `func_timeout` / `memory_profiler` / `kalign` are
in none of our venvs):

    ~/of3_venv            a venv layered on boltz_gpu_venv by ONE .pth line:
                          site-packages/_boltz_gpu.pth ->
                            /home/ubuntu/boltz_gpu_venv/lib/python3.12/site-packages
                          (`venv --system-site-packages` inherits the BASE
                           interpreter, not the venv it was called from, so the
                           .pth is the part that does the work)
                          + pip install pdbeccdutils lmdb func-timeout
                            memory_profiler pydantic lightning awscrt wandb
                            ml_collections biotite absl-py boto3
    ~/of3_stub/kalign.py  the ONE genuine stub. Not on PyPI (of3 gets it from
                          pixi), template-realignment only, and it RAISES --
                          `--use_templates false` must never reach it.

    PYTHONPATH=/home/ubuntu/openfold-3-v050:$HOME/of3_stub ~/of3_venv/bin/python \
      openfold3/run_openfold.py predict --query_json <q>.json \
      --inference_ckpt_path ~/of3-ob-174k.pt --num_diffusion_samples 5 \
      --num_model_seeds 1 --use_msa_server false --use_templates false \
      --output_dir <out>

`~/venv` and `~/boltz_venv` are untouched; nothing was installed into either.

WHERE TO LOOK NEXT. The two models share every line of our code and every
converter path but one, and the checkpoints differ by name in exactly those 25
LayerNorm tensors (4890 vs 4936 tensors, `converters/openfold3.py`
`has_shared_pair_norm`). That is a DIFFUSION-side convention, and the failure is
TRUNK-side (distogram contact precision 0.303 vs openfold3's 0.868), so the two
facts do not yet meet. Which makes the next measurement a real-input trunk
comparison against a native openbind0 fold -- now buildable, because that fold
runs.

### NOT ANSWERED: native PyTorch openbind0 has NOT been run on ubiquitin

Stated plainly because everything above is our JAX port compared against native
MODULES in-process, never against a native FOLD. So "consistent with the
checkpoint's own behaviour" is an inference from six ruled-out alternatives plus
the model's own low confidence -- it is not the measurement.

What the measurement needs, and the two routes:

  1. **A native of3 fold. THE GPU ROUTE EXISTS -- corrected 2026-09-11.** I
     wrote that this was blocked on GPU torch; that was wrong, and it was
     repeating a comment in `native_trunk_dump.sh` instead of checking. Measured:

         ~/venv           torch 2.13.0+cpu   cuda False
         ~/venv_esm       torch 2.13.0+cu130 cuda True
         ~/boltz_gpu_venv torch 2.13.0+cu130 cuda True

     of3 was missing four packages there, resolved the way this repo already
     resolves such things (`~/boltz2_extra`, `px_deps`) -- a directory of
     SYMLINKS, nothing installed:

         ~/of3_deps -> gemmi, ml_collections, absl, biotite  (from ~/venv)

         PYTHONPATH=/home/ubuntu/openfold-3:$HOME/of3_deps \
           ~/boltz_gpu_venv/bin/python ...

     Verified: `PairFormerStack` imports, cuda True, and of3-ob-174k.pt loads
     (4890 tensors). Both checkpoints are on disk. A full native fold still
     needs a query JSON and runner YAML for `openfold3/run_openfold.py`.
  2. **A real-input TRUNK comparison**, which is cheaper and enough: our trunk
     is the module the failure points at, and of3's 48-block stack already runs
     on CPU inside `trunk_parity` in ~2 min. Feed native's stack the REAL (s, z)
     our graph builds for ubiquitin and compare the pair output.

Route 2 was attempted and hit a trap worth recording: tapping the generic trunk
path from a full `fold_check.fold` run captures JAX TRACERS, not arrays --
recycling runs inside `model.py`'s `fori_loop`, so anything appended from there
is traced (`TracerArrayConversionError: ... traced array with shape
bfloat16[76,384]`). The ESMFold2 taps avoid this only because
`esmfold2_localise_trunk.py` calls `ev.Evoformer` directly in its own
`hk.transform` with no recycle loop. That is the pattern to copy: build the
Evoformer directly, not through `Model`.

`evoformer.py` now carries `trunk_in_pair` / `trunk_in_single` /
`trunk_out_pair` taps on the generic path alongside the ESMFold2 ones, gated off
behind AF3_ESM_TRUNK_TAPS, so route 2 needs only the direct-Evoformer harness.
