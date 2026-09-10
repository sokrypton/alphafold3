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
