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
