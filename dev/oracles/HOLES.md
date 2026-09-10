# What is left for 100% parity

TWO categories, and the second was invisible until `parity_audit.py` existed:

  1. **HOLES** -- cells that never ran. `gate_applies.py` says which of those
     are real (the model has that module and no other cell covers it).
  2. **cells that ran and DISAGREE.** `run_all_parity.sh`'s `classify()` only
     asks whether a matching line exists, never what it says, so a comparison
     at corr 0.9678 is reported OK. `parity_audit.py` grades on corr AND
     max|d|/rms.

# Category 1: HOLES -- four left

| gate | models | what it needs |
|---|---|---|
| `L4.confidence` | boltz2 | the converter is done (66 -> 11); the rest is forward branches, recipe in [[boltz2-confidence-port]]. |
| `L4.confidence` | esmfold2, esmfold2_fast | `esmfold2_reference` does NOT implement the confidence head, so this is the one esmfold2 cell that needs the DUMP. `esmfold2_oracle_6mrr.py` now hooks it (`conf.in.*`, one `conf.out.<name>` per output, `with_kwargs=True` because it is called with keywords only), and `esmfold2_dumps.module_io(model, 'conf')` hands them over. Our side then has to run on the RECORDED inputs, which is a different shape from every gate here -- they hand both sides a synthetic activation. |
| `L1b.msa` | esmfold2 | the only release with `msa=4`; the other three are n/a. |

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
