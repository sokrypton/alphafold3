# Parity

Every model this fork runs is checked against its **own vendor's stock code**, not just for plausible structures. This file holds the current results. The dated notebook of how each finding was made is in [docs/parity_log.md](docs/parity_log.md), and open module-level items are in [dev/oracles/HOLES.md](dev/oracles/HOLES.md).

**Status (2026-10-20):** every model matches its stock vendor on every input that vendor can take. That covers the trunk, one denoise step, confidence, padded inputs and the full diffusion trajectory. There is one caveat, protenix1's trajectory, explained [below](#full-trajectory).

- [How it is measured](#how-it-is-measured)
- [Results by model](#results-by-model)
- [Full trajectory](#full-trajectory)
- [Inputs a vendor cannot take (N/A)](#inputs-a-vendor-cannot-take-na)
- [Vendor bugs the harness patches](#vendor-bugs-the-harness-patches)
- [Vendor conventions the port reproduces](#vendor-conventions-the-port-reproduces)
- [BioEmu2](#bioemu2)
- [HelixFold3](#helixfold3)
- [Mol-1](#mol-1)
- [AlphaFold 2](#alphafold-2)
- [Module-level gates (L0 to L7)](#module-level-gates-l0-to-l7)
- [Fold accuracy](#fold-accuracy)
- [Re-running](#re-running)

## How it is measured

Two harnesses compare the whole model against stock code:

- `dev/oracles/vendor_native_parity.py` covers every ported model.
- `dev/oracles/af3_native_parity.py` covers `alphafold3` against DeepMind's own repository.

Both sides run their **shipped entry points** on the same input JSON. The vendor side runs in the vendor's own environment, with its own featuriser, model loader and forward pass. Our side runs our featurisation and `Model.__call__`. The harness hands neither side anything except **random draws**:

- the reference-conformer draw;
- the samples that the confidence head scores;
- one noised input for the denoise step.

No flag, constructor argument or feature is chosen by the harness. Atoms are matched by (token, atom name), so an atom that only one side has is reported, never silently dropped.

**Rows graded per case:**

- `s_inputs`;
- the trunk single and pair after recycling;
- distogram logits;
- one denoise step;
- pLDDT, PAE and PDE.

Each row is graded on correlation **and** `max|d|/rms`:

| grade | corr | max\|d\|/rms |
|---|---|---|
| PARITY | ≥ 0.99999 | ≤ 1e-3 |
| CLOSE | ≥ 0.9999 | ≤ 1e-2 |
| LOOSE | ≥ 0.999 | ≤ 1e-1 |
| BAD | otherwise | |

A case PASSes when every row is PARITY or CLOSE. `af3_native_parity.py` is stricter and requires PARITY.

**The 23 cases** come from `af3_native_parity.cases()`. They are deliberately small, so that each one fits a float32 fold on a 23 GB card:

| group | cases |
|---|---|
| chains | `protein` (6MRR), `homodimer`, `heterodimer`, `long_chain` |
| MSA | `msa`, `deep_msa` (more than 1024 rows, so subsampling and crops apply), `paired_msa` (paired across a heterodimer, with species keys) |
| nucleic acids | `rna`, `rna_mod` (PSU), `dna`, `dna_mod` (5CM), `protein_dna` |
| ligands | `ligand_ccd` (ATP), `ligand_smiles`, `ion` (MG), `multi_ligand` |
| modified / unknown | `ptm` (SEP), `unknown` (X) |
| bonds | `covalent` (NAG on Asn), `glycan` (NAG–NAG, two declared bonds) |
| templates | `template` (5K9P, carries SEP), `template_std` (the same, standard residues only), `multi_template` (two hits, one a sub-range) |
| cyclic | `cyclic`: only for vendors whose input can express it |

**Two more checks run on top of the 23 cases:**

- **Padding** (`OURS_BUCKET=shipped`) pads our side exactly as `run_alphafold.py` does. Real tokens must not move.
- **Full trajectory** (`--trajectory`) runs each side's complete sampler at its default settings. Every random draw is injected by meaning: the initial positions, plus each step's augmentation rotation, translation and noise. A different step count, schedule, noise scale or augmentation convention then shows up as a different final structure.

## Results by model

The vendor versions are the ones installed for the harness. All runs used float32 with `JAX_DEFAULT_MATMUL_PRECISION=highest`, on an A10 (24 GB) and an A100 (40 GB).

| `--model` | stock reference | 23 cases | padded | trajectory |
|---|---|---|---|---|
| `alphafold3` | DeepMind alphafold3 v3.0.4-49 | 23/23 (PARITY-only rule) | — | — |
| `boltz2` | boltz 2.2.1 | 23/23, plus `cyclic` | PASS | CLOSE, 0.025 Å |
| `chai1` | chai_lab 0.6.1 | 20/20, 3 N/A | PASS | PARITY, 0.001 Å |
| `protenix2` | Protenix v2.0.0 | 23/23 | PASS | CLOSE, 0.012 Å |
| `protenix1` | Protenix v2.0.0 (v1 checkpoint) | 23/23 | PASS | LOOSE; superposed 0.007 Å, [see below](#full-trajectory) |
| `opendde` | OpenDDE v1.1.0 | 23/23 | PASS | PARITY, 0.000 Å |
| `rosettafold3` | foundry aad357b | 21/21, 2 N/A | PASS | CLOSE, 0.007 Å |
| `openfold3` | OpenFold3 0.4.5 (preview-2) | 21/21, 2 N/A | PASS | PARITY, 0.000 Å |
| `openbind0` | OpenFold3 v0.5.0 | 21/21, 2 N/A | PASS | PARITY, 0.000 Å |
| `intellifold2` | IntelliFold v2.0.4 (native PyTorch) | 23/23 | PASS | PARITY, 0.005 Å |
| `esmfold2` | esm 3.4.1 | 23/23 | PASS | PARITY, 0.000 Å |
| `esmfold2_fast` | esm 3.4.1 | 23/23 | — | PARITY, 0.000 Å |
| `esmfold2_lm600m` | Synthyra FastPLMs | 20/20, 3 N/A | — | PARITY, 0.000 Å |
| `esmfold2_lm300m` | Synthyra FastPLMs | 20/20, 3 N/A | — | PARITY, 0.000 Å |
| `bioemu2` | BioEmu2 @7846769 | 20/20, 3 N/A | PASS | PARITY, 0.000 Å ([modes](#bioemu2)) |
| `mol1` | Mol-1 @bfc01fa (torch 2.12) | 24/24, 1 N/A, at its own 20 recycles, no floor grading | PASS | CLOSE, 0.066 Å; 8/8 ([see](#mol-1)) |
| `helixfold3` | PaddleHelix @8e3991a (paddle 3.1) | 20/20, 4 N/A, graded [against its own spread](#helixfold3); one multi_ligand row at the edge | — | PARITY, 0.000 Å; 8/8 within its own spread ([see](#helixfold3)) |
| `af2_ptm`, `af2_multimer` | DeepMind alphafold (AF2) | [18 gates, all green](#alphafold-2) | | |

The padded runs cover `protein`, `ligand_ccd` and `heterodimer`.

## Full trajectory

The trajectory distance is the maximum per-atom distance between the final coordinates. "Superposed" is the RMS after removing the rigid-body difference.

| model | steps | max \|d\| | superposed rms | rigid rotation | grade |
|---|---|---|---|---|---|
| `boltz2` | 200 | 0.025 Å | 0.0001 Å | 0.003° | CLOSE |
| `chai1` | 199 | 0.001 Å | 0.0000 Å | 0.000° | PARITY |
| `protenix2` | 200 | 0.012 Å | 0.0008 Å | 0.047° | CLOSE |
| `protenix1` | 200 | 0.221 Å | 0.0066 Å | 0.908° | LOOSE |
| `opendde` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `rosettafold3` | 49 | 0.007 Å | 0.0006 Å | 0.005° | CLOSE |
| `openfold3` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `openbind0` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `intellifold2` | 201 | 0.005 Å | 0.0002 Å | 0.019° | PARITY |
| `esmfold2` | 10 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `esmfold2_fast` | 10 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `esmfold2_lm600m` | 11 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `esmfold2_lm300m` | 11 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `bioemu2` | 200 | 0.000 Å | 0.0000 Å | 0.000° | PARITY |
| `mol1` | 200 | 0.066 Å | 0.0050 Å | 0.254° | CLOSE |

**protenix1 is the one LOOSE row.** Its difference is almost entirely a 0.9° rigid rotation of the whole structure; after superposition the remaining difference is 0.0066 Å. To tell model sensitivity from a port bug, the harness nudges the vendor's *own* trunk outputs by 1e-5 (`TRAJ_FLOOR=trunk`):

- protenix1 then rotates itself by **0.38°** (superposed 0.0027 Å);
- protenix2 rotates itself by only 0.025°.

So protenix1's sampler amplifies trunk differences at the 1e-5 level into a global rotation. The single-step rows for the same input are all PARITY or CLOSE. The raw grade is left as LOOSE rather than relaxed.

The step counts are each vendor's own default. Each schedule ends the way its vendor's does (`schedule_tail` in `diffusion_head.py`):

- protenix and opendde replace the last level with 0;
- intellifold2 appends a 0 to N+1 points;
- the ESMFold2 family spaces N points and appends a 0;
- chai evaluates at midpoints.

## Inputs a vendor cannot take (N/A)

A case is N/A only when the **stock** vendor has no way to express the input. Its runner exits with code 3, and that is reported as N/A, never as PASS. A crash is always a FAIL. Our port runs every one of these inputs; there is simply nothing to compare against.

| model | N/A cases | why |
|---|---|---|
| `chai1` | `ligand_ccd`, `ion`, `multi_ligand` | chai takes ligands as SMILES only (`ligand_smiles` PASSes) |
| `openfold3`, `openbind0` | `covalent`, `glycan` | stock OpenFold3 never reads `Query.covalent_bonds` |
| `rosettafold3` | `multi_template` | RF3 templates come from its input structure, and only one templated chain maps |
| `rosettafold3` | `glycan` | the RF3 JSON takes one `ccd_code` per chain |
| `esmfold2_lm600m`, `esmfold2_lm300m` | `msa`, `deep_msa`, `paired_msa` | trained without MSA conditioning; the vendor rejects explicit MSAs |
| `intellifold2` | `cyclic` | IntelliFold parses `cyclic: true`, but its `RelativePositionEncoding` never reads `cyclic_period` |
| `bioemu2` | `template`, `template_std`, `multi_template` | BioEmu2 has no template input |
| all but `boltz2` | `cyclic` | no cyclic input at all |

`--cyclic` is a feature of this fork: every model here honours it, but only boltz2 was trained with it.

## Vendor bugs the harness patches

Stock OpenFold3 0.4.5 has three MSA bugs that would make a correct port read as a failure. All three are fixed in OpenFold3 v0.5.0. 0.4.5 is still the reference because it is the newest release that loads the preview-2 checkpoint. The runner (`openfold3_upstream_runner.py`) patches each bug behind its own flag, and the sweeps set all three. Our port follows the fixed behaviour.

| flag | stock behaviour |
|---|---|
| `OF3_DIAG_FIX_PROFILE` | `calculate_profile` repeats column indices against a row-major ravel, so `profile[c]` counts the wrong cells. It is exact only for a one-row MSA. Fixed upstream in v0.5.0 (f896db32). |
| `OF3_DIAG_FIX_DELETION` | Operator precedence scales the deletion value by 8/π where AF3 (and v0.5.0) uses 2/π: four times too large. |
| `OF3_DIAG_FIX_PAIRED` | `create_paired_from_precomputed` never calls `set_row_counts`, so a precomputed paired MSA is silently dropped. Fixed in v0.5.0. |

Some vendors draw random MSA rows. `MSA_ALL_ROWS=1`, the harness default, lifts each vendor's draw so that both sides see the same rows:

- OF3 `subsample_all_msa`;
- RF3 `n_msa`;
- protenix `sample_indices`;
- OpenDDE `msa_depth` (1280 rows);
- IntelliFold `msa_depth` (1024 rows).

## Vendor conventions the port reproduces

These are the places where a vendor deliberately differs from AlphaFold 3, and where the sweep found that the port had not yet followed it. Each fix is in the model's spec or policy, not in the harness.

- **Templates:** `pipeline/template_policy.py` has one policy per model (`TEMPLATE_POLICIES` in `model_registry`):
  - RF3 takes the first altloc.
  - chai drops any hit that has a modified residue.
  - OF3 and OpenBind read a modified residue as UNK, realign the whole chain, use the parent's one-letter code, and rank hits.
- **OpenFold3's paired MSA:**
  - Rows are stacked as the query block, then the paired block as given (whose row 0 is the query again), then the main MSA minus the paired rows. The query therefore sits in rows 0 and 1.
  - 0.4.5 does not dedupe the main MSA.
- **RF3:**
  - Rows are paired by TaxID and flagged with `residue_is_paired`.
  - Triangle multiplication divides by the *real* token count, not the padded one.
  - Trailing insertions in an a3m are ignored, as AF3's parser does.
- **chai:**
  - The MSA query row is marked paired.
  - Its pair mask comes from the reference mask, so a reference-only leaving atom (NAG O1) is still an attended key.
  - Its rows are ordered by numpy 1.26's unstable quicksort, which is ported exactly.
  - Since 2026-10-28 both sides fold with ESM2, as chai-1 is run: stock embeds each protein with its own ESM2 3B
    trace, and ours folds from those rows (`chai1_upstream_runner.py` writes them; `CHAI_NO_ESM=1` turns both off;
    `CHAI_ESM_TOWER=1` grades our tower against them, corr 0.99993 on the heterodimer). Run one case per process: in
    one process ours keeps up to 14 GiB between cases and stock's forward then runs out of memory on an A10. With ESM:
    20 cases PASS (`pocket` and `contact` with the restraint-carrying blob). `paired_msa` reads BAD at the trunk pair
    (corr 0.9993; 0.9974 without ESM, so not ESM's) and is open; `long_chain` does not fit ours at float32 on an A10,
    with or without ESM.
- **IntelliFold-2:** uses the boltz-fork MSA build, with a per-chain profile over `msa_valid_depths`.
- **protenix:** the seed applies MC dropout on the recycled pair. Replaying it needs torch's fused-dropout vector width: 4 for float32, 8 for bfloat16.
- **Padded atoms:** every non-AF3 model masks padded atom queries.

## HelixFold3

HelixFold3 runs on PaddlePaddle, so its runner (`dev/oracles/helixfold3_upstream_runner.py`) drives stock `inference.py`'s own featuriser and model in `~/venv_helixfold3`, with TF32 off. Every array of its `s_inputs` equals ours on every case, and with one trunk pass every stage (input embedding, template embedder, MSA module) matches to 1e-5. Getting there meant reproducing these stock conventions:

- **Doubled residuals.** Paddle's `+=` rebinds rather than mutating, so `pair_act += msa_module(...)` and the template's `v_t += stack(v_t)` add each module's *full* output back to its input: z + MSA(z) and v + Stack(v) (`MSA_STACK_OUTER_RESIDUAL`, `TEMPLATE_STACK_OUTER_RESIDUAL`).
- **A bf16 triangle bias.** Stock's attention does `logits += nonbatched_bias.cast("bfloat16")` even in fp32, so the triangle-attention pair bias is always bf16-rounded (`BF16_TRI_ATT_PAIR_BIAS`).
- **Biased trunk Linears** in the triangle multiplication, triangle attention and outer product mean; the relative-position block gated on an equal residue index; a per-axis inverse squared distance; sequence-index RoPE in the diffusion transformers.
- **Featurisation:** chains in stock order; every CCD heavy atom kept (OXT, OP3, leaving atoms); blank atom names on atom tokens; the query row once and duplicate rows dropped; the profile over the merged MSA; a GAP profile and UNK ligand row; DN, its separate unknown-DNA class, carried as two extra `target_feat` columns; empty templates GAP-typed only in its no-MSA mode; AF2's template prefilter and parent residue letters.

**Graded against stock's own spread.** Stock HelixFold3 is chaotic under recycling. Nudging its own initial pair by 1e-6 moves its trunk pair by up to max|d|/rms 3.3 after two passes, and that holds with the bf16 cast removed too (0.7). Its rows are therefore graded with `CASE_FLOOR=1e-6`: stock is rerun against itself with that nudge (8–16 draws), and a row that misses PARITY but is no further from stock than stock's own worst draw (both corr and max|d|/rms) reads FLOOR. Every stock self-difference is printed beside the row it grades.

- At one trunk pass with the bf16 cast removed on both sides (`HF3_NO_BF16_BIAS`), the configuration with no rounding boundary in it, the cases read PARITY or CLOSE on s_inputs, trunk, PAE, PDE, the denoise step and pLDDT; deep_msa reads 1e-6 everywhere.
- At stock's settings (two passes, the cast on), every case passes, the recycled rows reading FLOOR, with one row over the line: multi_ligand's trunk pair reads max|d|/rms 2.75 against stock's worst 2.47 over 16 draws at a 1e-6 nudge, and 2.64 over 8 at 5e-5, the size of our input-level difference. Its correlation (0.99964) is better than stock's worst at either nudge (0.99943, 0.99879), the maximum sits on two protein tokens (32-33) rather than on the ligand, and the same case with the cast removed passes well inside its floor. It is graded BAD and left so; it is the bf16 rounding landing on the other side of a tie in a chaotic stack.

N/A: `covalent` and `glycan` (no bond input), `rna_mod` (stock's featuriser asserts on a modified RNA residue), `cyclic`.

**Double-check (2026-10-21).** An independent review against stock on 26 further featurisation probes and a line-by-line network/converter read found and fixed:
- **Chain order.** Stock orders chains protein and RNA (identical sequences regrouped together, in order of first appearance), then DNA, then ligands, and its token order, flat atom windows and diffusion RoPE positions follow. Ours now keeps that order (`hf3_chain_order`: chain ids kept, the pipeline's sort by id skipped). The new harness case `reorder` (ATP, DNA, protein, RNA) failed every row before and now reads PARITY/CLOSE, padded too; the runner reports stock's arrays in its own order (`HF3_CASE_ORDER=1` for the case's).
- **MSA-mode templates** are GAP on RNA, DNA and ligand tokens; a nucleic chain's deletion mean is over its own rows; **MSE** is one token per atom (`atomize_mse`); copies of one **SMILES ligand** are one entity (`shared_smiles_ligand_names`; two separate entries of the same SMILES also merge, where stock keeps them apart).
- **Confidence summary** is stock's: pTM/ipTM weight each pair by both tokens' frame masks, ipTM is 0 without an interface, the clash test and `ranking_score = 0.8 ipTM + 0.2 pTM - has_clash` are its own (checked exact against `helixfold/common/confidence.py`). Written structures keep OXT only at a chain's last residue and OP3 only at its first, as stock writes them; the model still predicts every atom.
- Under the fused kernels (`PAD_KEEP`) the template stack's same-chain mask no longer gates the outputs of cross-chain template pairs, which stock updates.

Of the 21 runnable cases at one pass with the cast off, 17 read PARITY/CLOSE on every row; msa, homodimer, template and template_std miss only on the trunk pair (max|d|/rms 1.1e-2, 7.2e-2, 4.3e-2, 4.3e-2), and that is inside stock's own one-pass spread: with `CASE_FLOOR=1e-6` (4 draws) stock moves itself by 5.9e-2, 0.85 and 9.6e-2, so all read FLOOR, every later row PARITY/CLOSE.

**Full trajectory**, graded on the sampler alone (one trunk pass, cast removed; `TRAJ_FLOOR=trunk TRAJ_FLOOR_EPS=1e-4` reruns stock with its trunk nudged by 1e-4, the size of our trunk difference). 200 steps and the sigma schedule match on every case:

| case | final coordinates | max \|d\| | superposed | stock vs itself |
|---|---|---|---|---|
| protein | PARITY | 0.000 Å | 0.0000 Å | 0.002 Å |
| msa | PARITY | 0.002 Å | 0.0003 Å | 0.004 Å |
| heterodimer | PARITY | 0.001 Å | 0.0002 Å | 0.004 Å |
| rna | PARITY | 0.000 Å | 0.0000 Å | 0.003 Å |
| dna | PARITY | 0.000 Å | 0.0000 Å | 0.004 Å |
| ptm | PARITY | 0.006 Å | 0.0005 Å | 0.003 Å |
| ion | PARITY | 0.000 Å | 0.0001 Å | 0.005 Å |
| ligand_ccd | FLOOR | 0.244 Å | 0.0254 Å | 0.754 Å (0.0754 Å superposed) |

## BioEmu2

BioEmu2 shares no network stage with AF3, so its harness branch compares its own feature dict and outputs: `bioemu2_case`, `bioemu2_trajectory` and `bioemu2_steer` in `vendor_native_parity.py`, run against `dev/oracles/bioemu2_upstream_runner.py`. Stock BioEmu2 runs in float32 at `highest` matmul precision; its default is a bf16 trunk with a TF32 denoiser.

**Features.** Every array of stock BioEmu2's feature dict equals ours on all 23 inputs, and on 28 of 30 further edge-case inputs (PTMs at chain ends, unknown nucleotides, MSA caps, partial pairing, glycans split across entries, deuterium, tricky SMILES). The other two use HEM, which RCSB revised after AF3's CCD snapshot; stock fetches its components live. That covers CCD-ideal reference positions, OXT and OP3 dropped at every polymer position, atom tokens for ligands, modified residues and glycans, and Boltz-1 MSA pairing with `deletion_value = π/2·atan(d/3)`.

**Network.** Every case reads PARITY on `s_inputs`, the trunk single and pair, the distogram, the pair conditioning and one denoise step, at max\|d\|/rms ≤ 1.1e-4. The padded check (`OURS_BUCKET=shipped`) and contact conditioning (residue–residue, residue–ligand atom) pass too.

**Its other modes**, final coordinates against stock on the same draws:

| run | result |
|---|---|
| structure mode, protein | PARITY, max \|d\| 0.000 Å |
| structure mode, ATP ligand | PARITY, 0.001 Å |
| equilibrium mode, heterodimer | PARITY, 0.001 Å |
| propagation, 1 ns from a stock equilibrium sample | PARITY, 0.000 Å |
| `edm_gd`, 4 particles | PARITY / CLOSE, ≤ 0.009 Å (superposed ≤ 0.0009 Å) |
| `edm_fk` (with FK resampling), 4 particles | PARITY, ≤ 0.001 Å |
| `edm_fkc` (gradient through the denoiser), 4 particles | PARITY, 0.000 Å |
| `edm_fk` / `edm_fkc` under a custom `--potentials` YAML | PARITY, ≤ 0.001 Å |
| `edm_fk` steering a propagation | PARITY, ≤ 0.001 Å |
| structure mode, homodimer | PARITY, 0.000 Å |
| `edm_gd` / `edm_fkc`, homodimer | PARITY / CLOSE, ≤ 0.022 Å |
| `edm_fk`, homodimer | LOOSE: a near-tie resample (below) |

The ten steering potentials match stock directly as well. Energies agree to ≤ 1e-7 relative and gradients to ≤ 4.4e-5, at random coordinates and at a real ligand geometry. Every topology table is equal.

**Steering a ligand is chaotic.** On the ATP case `edm_gd` reads LOOSE: 0.005–0.05 Å superposed against stock. `STEER_FLOOR` nudges stock's own x0 by 1e-5, and stock then moves 0.004–0.03 Å against itself. The flat-bottom restraints have step-function gradients, so a float-level difference that switches one restraint on or off changes a whole guidance step. This is the same class as protenix1's trajectory, and the raw grade is left as it is.

**FK resampling on the homodimer meets a near-tie.** A trace of every resampling step (`BIOEMU2_FK_TRACE`) shows ours and stock drawing identical indices for 45 resamples. At step 135 the weights are ours [0, 0.216, 0.706, 0.078] against stock's [0, 0.207, 0.723, 0.070], and the systematic draw falls between 0.207 and 0.216, so one side keeps particle 1 and the other particle 2. The two lineages end 2.8 Å apart. Stock does the same to itself: with its x0 nudged by 1e-4 Å (`STEER_FLOOR_EPS`) it switches lineage, and in that run ours matched stock to 0.003 Å. Which side of the tie a run lands on is float noise. Each multi-chain potential alone (`SymmetricChainCOM`, `InterChainVDW`, `IntraChainVDW`) reads PARITY, and so does the default set without its four ligand potentials. Those potentials are empty here, but dropping them reorders the energy sum, which is enough to move the tie.

**Physicality report.** Every stat, pass and reason equals stock's `check_physicality`. The check covered six structures (protein, homodimer, ligand, PTM, glycan, protein–DNA), each under 5–7 noise levels, in all three presets.

Two inputs come from outside, and the harness hands over the vendor's copy rather than grading them:

- A SMILES ligand's conformer is an unseeded RDKit draw in stock BioEmu2.
- The ligand restraint bounds are RDKit's. Releases differ (2025.09 here, 2026.03 in the vendor environment); under the vendor's release, ours are exact.

Not ported, and not reachable from stock's own entry points either:

- `ContactPotential` and `TemplateReferencePotential`: no `prepare()` fills their tables.
- Context-parallel sampling.

## Mol-1

Molence Bio's Mol-1 forks Protenix at upstream bd54a05, which is v2.0.0's model code, so it reuses the Protenix-v2 port. Its trunk is split in two:

- 8 "PairMixer" blocks (pairformer blocks with no triangle attention) at a pair width of 448;
- 40 ordinary blocks at 320, between two bias-free width adapters.

Its runner (`dev/oracles/mol1_upstream_runner.py`) drives stock `mol` in `~/venv_mol1` at Mol-1's own inference defaults: TF32 off, no MC dropout, and 20 recycles. LayerNorm and the triangle kernels run in plain PyTorch, as for protenix.

- **At one trunk pass**, all 24 runnable cases pass: 190 rows PARITY, 2 CLOSE (the denoise step on `rna_mod` and `dna_mod`, 2e-3).
- **At stock settings** (20 recycles, stock's per-pass MSA rows replayed), all 24 pass again: 186 rows PARITY, 6 CLOSE (1-2e-3). Unlike HelixFold3, Mol-1's recycling does not amplify float-level differences, so no row needed grading against its own spread.
- **`long_chain`** needs smaller stock attention chunks to fit a 22 GB card (`MOL1_CHUNK=64`, memory only).
- **Full trajectory, 8 cases.** `ion`, `heterodimer` and `ligand_ccd` are PARITY (≤0.003 Å). `dna`, `protein`, `rna` and `msa` are CLOSE (0.013-0.066 Å). `ptm` reads 0.084 Å, FLOOR: nudged by 1e-4 at its own trunk, stock moves itself by up to 7.9 Å on that input.
- **bf16** (`--bf16-graded all`, 4 cases) passes: our bf16 error averages 0.31× stock's own on the trunk pair, 0.27× on PAE, 0.35× on the denoise step and 0.12× on pLDDT.
  - Two things stay float32 in our bf16 mode, as `model_config.FP32_TRUNK_SINGLE` and `FP32_CONFIDENCE_HEAD` record. The trunk single carries six channels at ~3×10⁵ on every token, on top of a per-token signal of ~44 rms. A bf16 residual stream (step 256 at that size) rounds every token to the same vector after the first recycle. The confidence head runs in fp32 up to 2560 tokens, as stock's inference runner does.
  - **Stock's own default `--dtype bf16` hits exactly that.** Its pLDDT comes out a flat ~89 for any input: poly-G, a random 80-mer and ubiquitin all read 84–88, with every residue's CA at 89.2. Stock at `--dtype fp32` reads 78 / 75 / 92 with real per-residue variation, and ours matches it in both precisions (fp32 78.5 / 75 / 92.4; bf16 78.3 / 75.3 / 92.4). Run stock Mol-1 at `--dtype fp32` if its pLDDT matters.
  - Stock computes its row-chunked diffusion-conditioning cache in bf16, where Protenix computes it in fp32.
  - Measured with the chunking on and off, that changes stock's own bf16 error by less than its case-to-case spread, in both directions. The port does not imitate it.

Two output conventions follow stock rather than AlphaFold 3:

- **The summary confidences.** pTM and ipTM average each framed token's row over all tokens (pTM) or over the other chains' tokens (ipTM), then take the maximum. A monomer's ipTM is 0. The ranking score is `0.8 ipTM + 0.2 pTM - 100 has_clash`, with the clash test over polymer chains only. This is `sample_confidence.py`, unchanged from Protenix, so all of `PROTENIX_FAMILY` now uses it. It equals stock's functions on identical logits to 6e-8.
- **Each chain's C-terminal O and OXT are rebuilt** in the written structure, as `fix_cterminal_carboxyl_oxygens` does after stock's confidences. Ours is identical to stock's on the same coordinates.

N/A: `cyclic` (no cyclic input).

## Pocket and contact conditioning

The fold input's `constraints` (docs/input.md) reach the three models trained to read them. Each is graded against its stock vendor, fed the same constraints in its own format: Boltz YAML `pocket`/`contact`, chai's restraints CSV, BioEmu2's `Contact`.

There are four harness cases:

- `pocket`: a pocket of five residues for the SMILES ligand.
- `pocket_contact`: the same pocket on ATP, plus a contact to its atom PG.
- `contact_ligand`: that contact alone.
- `contact`: two residue contacts across the heterodimer.

| model | cases | result |
|---|---|---|
| `boltz2` | all four | PARITY on every row (s_inputs, trunk, distogram, PAE, PDE, denoise, pLDDT), at most 7e-5 |
| `chai1` | `pocket`, `contact` (it takes no CCD ligand and no atom endpoint) | PARITY on every row, at most 2e-5, with `CHAI_FP32=1` |
| `bioemu2` | `contact`, `contact_ligand` (it was trained on contacts only) | all 42 stock feature arrays equal; every model row PARITY |

**Negative control.** `OURS_NO_CONSTRAINTS=1` drops the constraints on our side only, so stock is conditioned and ours is not. Every constraint case must then FAIL, which shows that stock reads the constraints and that ours matches only because it reads them too.

- boltz2: the trunk pair falls to corr 0.97-0.98 (max|d|/rms 6-8) on all three cases run.
- chai1: the trunk pair falls to corr 0.96-0.98 (max|d|/rms 3-7), and pLDDT to 0.89 on `contact`.

**chai1 needs a new blob.** Its restraint columns enter the trunk through `token_pair_proj_in_trunk`, which the converter now folds into `chai1_restraint_trunk`. A blob from before this change loads with that key zeroed and logs a warning: the constraints then reach only the diffusion's pair input. On unconstrained inputs the new blob is unchanged (`ligand_smiles` and `heterodimer` PARITY).

## AlphaFold 2

None of the AF3 module levels apply to AF2, a sibling network with no converter, so AF2 is graded separately. **L5af2 is 18 gates, all green.**

Thirteen of them compare against DeepMind's own repository:

- the trunk end to end;
- the structure module as eight fold iterations;
- extra-MSA and IPA;
- all five heads;
- all three template paths;
- the MSA pipeline.

They cover both the monomer and multimer paths. That oracle found three real divergences:

- a masked template distogram;
- BERT masking that was implemented but never called;
- an AF3 confidence pairformer that never received the dropout flag.

There are two deliberate differences:

- MSA sizes are 512/1024 on both paths, where stock uses 512/5120 and 508/2048.
- The monomer's TensorFlow row selection is replaced by multimer's gumbel argsort, because there is no TensorFlow here.

## Module-level gates (L0 to L7)

These came before the whole-model harness. They compare one module at a time on bounded inputs, and they localise a failure that the whole-model rows only detect. Last full run: 2026-09-14 (`bash dev/oracles/run_all_parity.sh all`, graded by `dev/oracles/parity_audit.py`).

| level | what it compares |
|---|---|
| L0 | conversion coverage: every checkpoint tensor accounted for, in both directions |
| L1 | the trunk: z-init, pairformer, MSA module, template embedder, distogram |
| L1r | the trunk on a real input: the input embedder and recycling |
| L1x | the same on a complex, where the cross-chain terms come into play |
| L2 | diffusion conditioning, token transformer, atom encoder and decoder |
| L3 | one full denoise step |
| L4 | the confidence head: PAE, PDE, pLDDT, resolved |
| L5 | an end-to-end fold, scored against an experimental structure |
| L6 | modality: RNA, DNA, ligands, complexes, modified residues |
| L7 | the output side: fold, write the mmCIF, parse it back, compare |
| Lg | differentiability in the sequence (the design path): 16/16 models |

```
369 cells    295 OK   1 SKIP   73 N/A   0 FAIL
346 comparisons: PARITY=304  CLOSE=14  FLOOR=17  DIAG=6  LOOSE=5  BAD=0
```

How to read these labels:

- **N/A** means the model has no such module. `dev/oracles/gate_applies.py` decides this from the registry, so an empty cell can never quietly mean "not run".
- **FLOOR** means the cell has no resolution left: nudging the gate's own input by 1e-6 moves its output further than the port differs. All 17 FLOOR cells are the trunk pairformer on synthetic input.
- **DIAG** is L1r: the whole trunk with ten recycles, read against a control case rather than a fixed bar.
- **The SKIP** is boltz2's L3, which is gated by injection instead.
- **The 5 LOOSE** are confidence tails at corr ≥ 0.99994:
  - in the protenix lineage, `torch.cdist`'s float32 expansion on the vendor side. Given an exact distance, they collapse to about 1e-4;
  - in chai1, a bfloat16 head.

  The whole-model harness grades these heads PARITY or CLOSE on every case.

## Fold accuracy

These numbers measure how well a model folds, not how faithfully it was ported. The table is 6MRR best-of-5 CA-RMSD, de novo from a single sequence (68 aa), 2026-09-14:

| model | Å | | model | Å |
|---|---|---|---|---|
| `boltz2` | 0.459 | | `intellifold2` | 1.512 |
| `alphafold3` | 0.628 | | `esmfold2_lm600m` | 1.522 |
| `opendde` | 0.729 | | `openfold3` | 1.547 |
| `rosettafold3` | 0.967 | | `protenix1` | 1.555 |
| `protenix2` | 1.016 | | `openbind0` | 1.578 |
| `esmfold2_fast` | 1.182 | | `esmfold2_lm300m` | 1.687 |
| `esmfold2` | 1.330 | | `chai1` | 1.713 |

AF2 runs single-seed at float32 with zero recycles: `af2_ptm` 1.712, `af2_multimer` 1.788.

Read these numbers as a band, not a ranking. Best-of-5 is a tail statistic, and some models vary run to run on the same seed: rosettafold3 seed 0 gave 0.968, 0.949 and 1.026. For a real comparison, use means over 20 or more samples.

Modality, from the L6 run:

| model | protein+ligand (Å / LDDT) | complex | RNA | DNA | confidence (mean / r) |
|---|---|---|---|---|---|
| `intellifold2` | 0.301 / 0.881 | 0.92 | 1.621 | 1.95 | 90.4 / .776 |
| `boltz2` | 0.385 / 0.907 | 0.92 | 1.419 | 1.59 | 96.9 / .830 |
| `protenix2` | 2.497 / 1.147 | 1.02 | 2.062 | 2.21 | 92.7 / .807 |
| `rosettafold3` | 0.464 / 0.889 | 1.12 | 1.143 | 2.67 | 85.2 / .708 |
| `opendde` | 1.269 / 0.870 | 1.49 | 1.372 | 1.99 | 89.9 / .635 |
| `chai1` | 0.335 / 0.993 | 1.57 | 1.648 | 1.89 | 78.9 / .543 |
| `openfold3` | 0.348 / 0.894 | 1.60 | 1.441 | 2.22 | 83.2 / .678 |

`r` is the correlation between predicted and actual error. A constant head would read r ≈ 0, whatever its mean.

## Re-running

```bash
# one model, all cases (the vendor's environment is looked up in VENDORS)
~/venv/bin/python dev/oracles/vendor_native_parity.py boltz2
# chosen cases; the full sampler; padded like a real run
~/venv/bin/python dev/oracles/vendor_native_parity.py rosettafold3 paired_msa template
~/venv/bin/python dev/oracles/vendor_native_parity.py protenix2 protein --trajectory
OURS_BUCKET=shipped ~/venv/bin/python dev/oracles/vendor_native_parity.py chai1 protein heterodimer
# alphafold3 against DeepMind's repository (~/af3_upstream, ~/venv_af3)
~/venv/bin/python dev/oracles/af3_native_parity.py
```

Each case runs in its own process, so that a case which runs out of memory does not starve the next. Useful environment variables:

| variable | effect |
|---|---|
| `VENDOR_NO_TAPS=1` | skip the vendor's localisation taps (large files) |
| `OPENDDE_CHUNK=64` | chunk OpenDDE's trunk, needed for `long_chain` |
| `TRAJ_FLOOR=trunk` | measure the vendor against itself with its trunk nudged 1e-5 |
| `MODEL_DIR` | use an alternative blob for our side |

chai1 also needs `CHAI_FP32=1` and the stock-chai interpreter in `CHAI_VENDOR_PY`.

Reference conformers are RDKit embeddings, so both sides must use the same RDKit version; `af3_native_parity.py` refuses to compare when they differ.
