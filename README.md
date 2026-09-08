![header](docs/header.jpg)

# AlphaFold 3

This package provides an implementation of the inference pipeline of
AlphaFold 3. See below for how to access the model parameters. You may only use
AlphaFold 3 model parameters if received directly from Google. Use is subject to
these
[terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Any publication that discloses findings arising from using this source code, the
model parameters or outputs produced by those should [cite](#citing-this-work)
the
[Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
paper.

Please also refer to the Supplementary Information for a detailed description of
the method.

AlphaFold 3 is also available at
[alphafoldserver.com](https://alphafoldserver.com) for non-commercial use,
though with a more limited set of ligands and covalent modifications.

If you have any questions, please contact the AlphaFold team at
[alphafold@google.com](mailto:alphafold@google.com).

## Obtaining Model Parameters

This repository contains all necessary code for AlphaFold 3 inference. You can
download the AlphaFold 3 model parameters from
https://storage.googleapis.com/alphafold3/af3.bin.zst. Use is subject to these
[terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

## The models this fork runs

**18 model types**, and one `--model` flag decides which forward branches,
config shapes, sampler constants and input conventions are used. The same input
JSON drives all of them.

Two ENGINES live here. Everything in the first four groups rides the shared
AlphaFold 3 graph — a pair+single trunk into a diffusion sampler — so adding one
is a weight remap plus a branch. AlphaFold 2 shares neither end (MSA row/column
attention, and an IPA head over backbone frames) and runs as a sibling network
under `alphafold3.af2`, reached through the same CLI and the same output writer.
Check `model_registry.get(name).engine` rather than testing the name.

`6MRR` is best-of-5 CA-RMSD, de novo from a single sequence, 68 aa — the shared
regression gate (`dev/bench/sweep22.sh`). It says how well a model folds, not
how faithfully it was ported; for that see [Parity status](#parity-status).

### AlphaFold 3 lineage

| `--model` | model | weights | 6MRR Å |
|---|---|---|---|
| `alphafold3` | AlphaFold 3 (Google DeepMind) | request from DeepMind | 0.632 |
| `openfold3` | [OpenFold3 preview-2](https://github.com/aqlaboratory/openfold3) (AlQuraishi Lab) | Apache 2.0 | 1.541 |
| `openbind0` | [OpenFold3 v0.5.0 "OpenBind"](https://github.com/aqlaboratory/openfold-3/releases/tag/v0.5.0) | Apache 2.0 | 1.649 |
| `intellifold2` | [IntelliFold-v2](https://huggingface.co/intelligenAI/intellifold) (IntelligenAI) | see upstream | 1.514 |

### Protenix family

Protenix publishes nine model types that differ only in counts and widths, so
`derive_dims` + `PROTENIX_FAMILY` make each one close to a one-liner. This fork
carried six of them and now runs **two**: `protenix2`, the flagship and the most
completely gated model here, and `protenix1`. `protenix05`,
`protenix1_20250630`, `protenix_mini` and `protenix_tiny` were removed on
2026-09-08 — they differed from these only by training run or by size, and
keeping them spread parity work across variants without adding coverage. The
machinery that made them one-liners is untouched, so re-adding one is still a
`sources.py` entry, a converter alias and a `MODELS` line.

| `--model` | model | weights | 6MRR Å |
|---|---|---|---|
| `protenix2` | [Protenix-v2](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 | 0.702 |
| `protenix1` | Protenix-v1 | Apache 2.0 | 1.694 |

### Other AF3-architecture models

| `--model` | model | weights | 6MRR Å |
|---|---|---|---|
| `boltz2` | [Boltz-2](https://github.com/jwohlwend/boltz) | MIT | 0.434 |
| `opendde` | [OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) (Aureka Research) | see upstream | 0.767 |
| `rosettafold3` | [RoseTTAFold3](https://files.ipd.uw.edu/pub/rf3/) (RosettaCommons) | see upstream | 0.986 |
| `chai1` | [chai-1](https://github.com/chaidiscovery/chai-lab) (Chai Discovery) | Apache 2.0 | 1.719 |

### ESMFold2 family — folds from ESM-C, not an MSA

Pair-only trunk conditioned on an ESM-C language model.

| `--model` | model | weights | 6MRR Å |
|---|---|---|---|
| `esmfold2` | [ESMFold2](https://huggingface.co/biohub/ESMFold2) (Arc / CZ Biohub) | MIT | 1.483 |
| `esmfold2_fast` | [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast) — half the trunk | MIT | 1.245 |
| `esmfold2_exp` | [ESMFold2-Experimental](https://huggingface.co/biohub/ESMFold2-Experimental) | MIT | 0.722 |
| `esmfold2_exp_fast` | [ESMFold2-Experimental-Fast](https://huggingface.co/biohub/ESMFold2-Experimental-Fast) | MIT | 1.267 |
| `esmfold2_exp_cutoff2025` | […-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Cutoff2025) | MIT | 1.611 |
| `esmfold2_exp_fast_cutoff2025` | […-Fast-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-Cutoff2025) | MIT | 1.423 |
| `esmfold2_lm600m` | […-base600M-step1500k](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-base600M-step1500k) — **ESM-C 600M** | MIT | 0.858 |
| `esmfold2_lm300m` | […-base300M-step1500k](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-base300M-step1500k) — **ESM-C 300M** | MIT | 1.753 |

All but the last two condition on ESM-C 6B; those use the 600M and 300M towers
and cost 0.5 GB and 0.3 GB against 5.1. Upstream ships the "Experimental" line
for paper reproducibility rather than research use — its own card says to prefer
plain `esmfold2`. The language-model input is built outside the fold by
`alphafold3.model.esm` (tower, published int8 at 5.5 GB) plus a per-model shim
that ships with the weights. **Passing one variant another's tower reads corr
0.026 against native** — take the pairing from
`model_registry.ESMFOLD2_VARIANTS[name]['esmc']`. Given an MSA instead they fold
from that: 5CAJ reads 17.5 Å from a single sequence and 1.24 Å at MSA depth 256.
Supplying neither is the one broken configuration.

### AlphaFold 2 — a sibling network, not the AF3 graph

| `--model` | model | weights | 6MRR Å |
|---|---|---|---|
| `af2_ptm` | AlphaFold 2 monomer pTM (`params_model_*_ptm.npz`) | CC BY 4.0 | 1.712 |
| `af2_multimer` | AlphaFold 2 multimer v3 (`params_model_*_multimer_v3.npz`) | CC BY 4.0 | 1.788 |

DeepMind's own AlphaFold 2 parameters, read from `--model_dir` as
`params/params_model_*.npz` — nothing is converted or republished. Monomer and
multimer run on ONE graph: a monomer checkpoint is converted onto the multimer
network at load. **Protein only** — a fold input carrying a ligand, a nucleotide
or an inter-chain bond raises rather than silently folding the protein subset.
MSAs come from the AF3 data pipeline, chain pairing included. Templates are not
wired up yet.

The two AF2 rows are single-seed at float32 with zero recycles, not best-of-5 at
config defaults like the rows above, so read them against each other rather than
against the AF3 lineage.

## Parity status

This fork runs **18 model types**, and the question that matters for every one
of them is whether it reproduces its own vendor's implementation rather than
merely producing a plausible structure. `PARITY.md` is the full record; this is
the summary.

Parity is measured at seven levels, from the weights inwards to the fold:

| level | what it compares |
|---|---|
| **L0** | conversion coverage — every checkpoint tensor accounted for, both directions |
| **L1** | the trunk: pairformer single + pair, against the vendor's own module |
| **L2** | diffusion conditioning, token transformer, atom encoder (three parts) |
| **L3** | one full denoise step — conditioning, atom encoder, transformer, decoder, EDM |
| **L4** | the confidence head: PAE, PDE, pLDDT, resolved |
| **L5** | an end-to-end fold, scored against an experimental structure |
| **L6** | modality: RNA, DNA, ligands, complexes, modified residues, and the mmCIF written out |

### Coverage, every model

`✓` gated, `~` partially gated, `·` not measured, `n/a` no vendor to compare
against.

| model | L0 | L1 trunk | L2 diff-cond | L3 denoise | L4 conf | L5 fold | L6 modality |
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
| `esmfold2` family (8) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | n/a — protein only |
| `af2_ptm` / `af2_multimer` | n/a | n/a | n/a | n/a | n/a | ✓ | n/a — protein only |

`alphafold3` and the AF2 pair are `n/a` by construction: the first IS the
reference implementation, the second runs DeepMind's own network unmodified
(its gate is a known-answer test against ColabDesign on the same weights,
agreeing to 0.0006 Å).

**What the gaps are, stated plainly rather than left as symbols:**

* **`~` at L2 means fewer than three of its parts are measured.** The token
  transformer is gated on ten models at corr 1.000000; the diffusion
  conditioning on `protenix2`, `protenix1` and `rosettafold3`; the atom encoder
  on those plus both OpenFold3 releases and `intellifold2`. The `✓` rows
  (`boltz2`, `opendde`, `chai1`, `esmfold2`) were gated by whole-module
  injection when they were ported, which is a different and coarser standard.
* **The atom DECODER is gated on `protenix2`** (exact, 0.000002 Å/atom) and
  covered in composition elsewhere by the exact denoise steps.
* **`chai1`'s template embedder cannot be gated** from its shipped TorchScript:
  the submodule's parameters are reachable but it has no callable `forward`.

And the table hides four modules that nothing has ever compared, because the
levels were organised around the diffusion path:

| module | models carrying it | gated on |
|---|---|---|
| **template embedder** | 9 | **8** — gated 2026-09-08, found two bugs |
| **MSA module** | 12 | **11** — all exact; only chai1 unreachable. boltz2's 0.974 was a real OPM bug, fixed |
| ~~distogram head~~ | all | **8** — closed 2026-09-08 |
| ~~input embedder~~ | all | **2** — closed 2026-09-08 |
| ~~recycling loop~~ | all | **2** — same gate |

Templates worked end to end — `boltz2` folds 5CAJ to 0.72 Å with one — but that
was evidence from folds, and folds were not enough. The first module-level
comparison read **0.998468** for `protenix2` and turned up **two port bugs**:
`restype_i`/`restype_j` were concatenated in the wrong order (protenix's
`expand_at_dim(dim=-3)` makes its first block the *j*-varying one, ours was
*i*), and our shared template forward applied boltz2's outer residual
(`v = v + stack(v)`) to protenix, which does not have it. Fixed, protenix2 and
protenix1 read **1.000000**.

At the fold level, a templated 5K9P went from **1.588 Å to 0.227 Å** best. That
is why folds never caught it: a wrong-but-plausible template contribution still
points a fold roughly the right way, and 1.588 Å looks like a working template
until something compares the module. Untemplated folds and the other models
sharing that code are unchanged.

`rosettafold3` is gated too and had neither bug — its features are i/j-symmetric
distance conditioning, and it overrides the shared forward rather than
inheriting it. Which is the real lesson: protenix inherited boltz2's convention
and was wrong, rf3 escaped only by not inheriting, so the convention is now
named in `model_config` instead of left to inheritance. See `PARITY.md`.

The distogram head **was** on that list and is now gated on six models, all
exact. It was worth doing first despite being one projection: it is the head
design gradients flow through, so a divergence would have been invisible to
every structural number here. The gate checks what a correlation hides — where
each vendor symmetrises. protenix, of3 and if2 do it after the projection
(`W(z+zᵀ) + 2b`); `rosettafold3` does it before (`W(z+zᵀ) + b`), which is why
its bias is halved on conversion. See `PARITY.md`.

### L1 — the trunk, against each model's own native module

Correlation of our single (`s`) and pair (`z`) trunk representations against the
vendor's torch module on identical inputs (`dev/oracles/trunk_parity.py`).

| model | single | pair | notes |
|---|---|---|---|
| `boltz2` | 1.000000 | 1.000000 | |
| `protenix2` | 1.000000 | 1.000000 | was 0.9929/0.9376 — harness confounds, never a port defect |
| `opendde` | 1.000000 | 1.000000 | |
| `openfold3` | 1.000000 | 1.000000 | |
| `protenix1` | 1.000000 | 1.000000 | 48 blocks, c_z 128 |
| `openbind0` | 1.000000 | 1.000000 | v0.5.0 weights; found and fixed a transposed column pair bias |
| `rosettafold3` | 0.999997 | 0.999996 | |
| `intellifold2` | 1.000000 | 0.998994 | over 48 blocks; ONE block is 1.000000/1.000000 and every op bisects to ≥0.999997, so the residual is fp compounding |
| `chai1` | 0.999945 | 0.999918 | at the floor — native chai's TorchScript is bf16 |
| `esmfold2` family | — | — | whole trunk corr **0.99961** from raw features (not split s/z); the eight variants share this graph |
| `alphafold3` | n/a | n/a | this IS the reference implementation |

### L3 — one full denoise step

The level that subsumes the diffusion side: conditioning, atom encoder, token
transformer, atom decoder and the EDM scaling all run, so a match here means the
whole score model agrees for one step. It is also **the only evidence the atom
decoder is right**, since nothing gates it directly.

| model | corr | per-atom mean | native tensors, missing/unexpected |
|---|---|---|---|
| `protenix2` | 1.000000 | **0.0000 Å** | 0 / 0 |
| `openfold3` | 1.000000 | **0.0001 Å** | 763, 0 / 0 |
| `openbind0` | 1.000000 | **0.0021 Å** | 740, 24 handled / 1 |
| `intellifold2` | 0.999950 | 0.075 Å | 706, 0 / 0 |
| `rosettafold3` | 0.999948 | 0.401 Å | 879, 0 / 0 |
| `protenix1` | 0.999428 | 0.207 Å | 0 / 0 |
| `chai1` | 1.000000 | 0.012 Å | injected |
| `boltz2` | 0.999900 | 0.203 Å | injected from a captured boltz run |
| `opendde` | — | at parity across every sigma from 4608 down to 1 | |
| `esmfold2` family | 0.99999765 | — | `r_update` / `x_denoised` |

`openbind0`'s 24 "missing" tensors are the per-block pair LayerNorms it does not
have: OpenFold3 v0.5.0 runs that LayerNorm once for the stack where preview-2
runs it inside every block. The checkpoint decides which, not a remembered flag.

### L4 — the confidence head

**Every port has one**, and every one agrees with its vendor:

| model | pae / pde / plddt / resolved |
|---|---|
| `esmfold2` family | **≥ 0.99999981** |
| `openfold3`, `openbind0`, `intellifold2` | **1.000000** (if2 once native is rounded to bf16 — which is how its blob stores trunk weights) |
| `opendde` | pae/plddt/resolved **1.000000**, pde 0.999999 — its own structural-token head |
| `protenix2`, `protenix1` | **≥ 0.999985** |
| `rosettafold3` | pae/pde **0.999999**, plddt/resolved **1.000000** |
| `chai1` | 0.999905–0.999968 — bf16 floor |
| `boltz2` | pairformer ×8: s 1.000000 / z 0.999998; z re-embedding 0.9999996 |

This gate earned its keep the first time it ran: it found that protenix's PDE
head symmetrises the pair activation *before* its LayerNorm where AlphaFold 3
symmetrises the logits *after* the projection. LayerNorm is not linear, so those
differ; `full_pde` read 0.870 and now reads 0.999989. Six models were affected
and no weight changed — and no fold could have caught it, because PDE is
reported and never fed back into the structure.

Note that this confidence column and the one in the modality screens measure
different things. This asks whether our module reproduces NATIVE's numbers; that
asks whether the head PREDICTS ERROR at all. A head can pass either and fail the
other, and a faithful port of a badly calibrated head passes this and fails that.

### A caution about single-seed comparisons

The longest-standing "open bug" in this document — protenix2 failing on a
modified residue — was **retracted** on 2026-09-08. It rested on one seed: ours
7.584 Å against native's 1.080 on phospho-ubiquitin. Re-run across four seeds,
native ranges 0.87–12.70 Å on that target and one seed inverts the story
entirely, while our own plain-target number turned out identical to our modified
one. There was no modified-residue bug.

The habit that produced it is worth naming: a fold number on a target the model
folds *unreliably* is not a measurement, and comparing two of them is not a
gate. What remains genuinely open there is a distributional difference — native
reaches a good basin on that target and we do not — and it is recorded as
unexplained rather than attributed. See `PARITY.md`.

### Where the harnesses live, and what ships

Worth saying plainly, because the numbers above cite scripts a reader will not
find: **the verification harnesses are not in this repository.** `dev/` is
gitignored (`.gitignore:21`). What ships here is the model code, the
`converters/`, and `run_alphafold.py`; `PARITY.md` is the tracked record of what
those harnesses measured, and is the only such record.

| gate | level | covers |
|---|---|---|
| `dev/oracles/trunk_parity.py` | L1 | pairformer stack vs the vendor's module — 7 models |
| `dev/oracles/prot_parity.py` | L1b | protenix trunk AND MSA module |
| `dev/oracles/msa_parity.py` + `esmfold2_msa_dump.py` | L1b | MSA module vs the vendor's own — rf3, both of3, intellifold2, opendde, boltz2, all three MSA-bearing esmfold2 releases. `LAYER=1` splits one boltz2 layer; `NONUNIFORM=1` runs a non-trivial msa mask |
| `dev/oracles/conditioning_parity.py` | L2 | diffusion pair + single conditioning — protenix2, protenix1, rf3 |
| `dev/oracles/atom_parity.py` | L2 | atom cross-attention encoder, real batch, windowed — protenix2/1, both of3, intellifold2, rosettafold3 |
| `dev/oracles/diffusion_parity.py`, `l2_all.sh` | L2 | token diffusion transformer — 10 models |
| `dev/oracles/denoise_parity.py` | L3 | one denoise step, whole diffusion module — protenix2/1, both of3, intellifold2, rosettafold3 |
| `dev/oracles/confidence_parity.py`, `l4_all.sh` | L4 | confidence head — every port |
| `dev/oracles/fold_check.py` | L5 | one model, one target, CA-RMSD |
| `dev/oracles/modality_check.py` | L6 | RNA / DNA / ligand / complex folds scored against a reference; `--write` validates the mmCIF the model emits |
| `dev/oracles/grad_check.py` | — | sequence-differentiability, either engine |
| `dev/oracles/af2_fold_check.py` | — | AF2 against ColabDesign on the same weights |
| `dev/bench/sweep22.sh` | — | the 6MRR regression sweep (22 models when last run; 18 now) |

Each needs its vendor's source on `PYTHONPATH` — the whole point is to run the
vendor's own module beside ours — so they are only runnable on a machine that
has those checkouts.

**Re-running any parity number requires two switches**, or the result is
meaningless. Both cost six false leads on protenix2 alone:

```
JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:/path/to/protenix \
  python dev/oracles/trunk_parity.py protenix2
```

`JAX_DEFAULT_MATMUL_PRECISION=highest` disables tf32, which is ~5e-4 per matmul
and compounds over 48 blocks. And parameters must be fp32: they otherwise come
back bfloat16-rounded, and flipping `global_config.bfloat16` after the model is
built does nothing, because the dtype was already fixed by `jax.eval_shape`.

Two more confounds are documented in `PARITY.md` because each impersonated a
port bug for a while: **the blob's own storage dtype** (intellifold2 stores its
trunk weights bf16 on disk, on purpose) and **the vendor's own alphabet**
(rosettafold3 transposes G/C against OpenFold3's, which broke RNA while every
protein and ligand gate passed — and later broke a harness the same way).

### Modality screens

Folding 6MRR says nothing about ligands, nucleic acids or complexes. Four
screens cover those, and **they were run on seven models, not all
twenty-four** — the protenix variants, `openbind0` and the ESMFold2 family have
a 6MRR number and nothing else here. That is a gap in the testing, not a
statement about those models.

| model | protein+ligand | complex | RNA | DNA | confidence |
|---|---|---|---|---|---|
| `intellifold2` | 0.301 / 0.881 | 0.92 | 1.621 | 1.95 | 90.4 / r .776 |
| `boltz2` | 0.385 / 0.907 | 0.92 | 1.419 | 1.59 | 96.9 / r .830 |
| `protenix2` | 2.497 / 1.147 | 1.02 | 2.062 | 2.21 | 92.7 / r .807 |
| `rosettafold3` | 0.464 / 0.889 | 1.12 | 1.143 | 2.67 | 85.2 / r .708 |
| `opendde` | 1.269 / 0.870 | 1.49 | 1.372 | 1.99 | 89.9 / r .635 |
| `chai1` | 0.335 / 0.993 | 1.57 | 1.648 | 1.89 | 78.9 / r .543 |
| `openfold3` | 0.348 / 0.894 | 1.60 | 1.441 | 2.22 | 83.2 / r .678 |

**These screens now also run IN THIS REPOSITORY** (`dev/oracles/modality_check.py`,
added 2026-09-07), which is what closes the "seven models, not twenty-four" gap
above for RNA and ligands. Different measurement from the table above — default
recycles, best of 5 samples, seed 0 — so read it as its own column, not as a
correction:

| model | 1EHZ tRNA (C1') | 1STP protein (CA) | BTN ligand (in-frame) |
|---|---|---|---|
| `alphafold3` | 1.412 | 0.564 | — |
| `openfold3` | 1.334 | 0.494 | — |
| `openbind0` | 1.496 | 0.499 | — |
| `intellifold2` | 1.472 | 0.316 | 0.436 |
| `protenix2` | 1.754 | 2.090 | 1.253 |
| `protenix1` | 1.737 | 1.867 | 0.918 |
| `boltz2` | 1.196 | 0.276 | 0.457 |
| `opendde` | 1.327 | 0.298 | 0.876 |
| `rosettafold3` | 1.047 | 0.322 | 0.450 |

RNA is single-sequence here; the ligand case takes its MSA from the same JSON
the older screen used, and streptavidin from a single sequence lands at 3-5 Å,
which measures the missing MSA rather than the ligand. The harness also covers
DNA, a protein-DNA complex scored in one shared frame, and a modified residue
(ubiquitin phospho-Ser20), and `--write` re-reads the mmCIF the model emits to
check the ligand or the modified residue survived into it.

  * **protein+ligand** — 1STP chain A + biotin, 10 recycles, protein-superposed
    so a ligand in the wrong pocket cannot hide. Two numbers: protein Å / BTN Å.
    All seven handle it.
  * **complex** — a 146+74 heterodimer, chain B's RMSD in chain A's frame (the
    docking). Subunits are 0.3–1.2 Å and fnat 0.79–0.95 throughout. Do not
    over-read the ordering: the reference used templates and the screen feeds
    none, and seed spread is ~0.2 Å.
  * **RNA / DNA** — 1EHZ tRNA-Phe (76 nt) and the 1LMB operator duplex (20 bp),
    single sequence, 10 recycles. Note a duplex CANNOT detect a G↔C alphabet
    transposition, because the swap stays Watson-Crick complementary; that is
    checked statically instead.
  * **confidence** — mean pLDDT / Pearson r of per-residue pLDDT against minus
    the per-residue CA deviation. `r` is the column that matters: a constant
    head reads r ≈ 0 however plausible its mean looks. All seven predict error.
  * `esmfold2_lm600m` and `esmfold2_lm300m` ship no confidence head at all
    (`NO_CONFIDENCE_HEAD`), and are structure-only by design.

Every model is also differentiable in the sequence (24/24, verified with
`dev/oracles/grad_check.py`), which is what the design path needs; the
magnitudes vary widely between models and are recorded in
`dev/bench/grad_sweep_2026-09-07.txt` rather than here.

### Getting the weights

You do not have to do anything. The first run of a model downloads its converted
weights from
[huggingface.co/sokrypton/af3-any-model](https://huggingface.co/sokrypton/af3-any-model)
and caches them; every run after that is offline.

```bash
python run_alphafold.py \
  --model=openfold3 \
  --json_path=fold_input.json \
  --output_dir=./output/
```

They land in `~/.cache/alphafold3/weights/<model>/`, or under `$AF3_WEIGHTS_DIR`
if you set it. `--model_dir` overrides both and skips the download entirely, which
is what you want for weights you converted or staged yourself. AlphaFold 3's own
parameters are the exception: they are not ours to redistribute, so `--model
alphafold3` needs `--model_dir` pointing at your own copy.

**Smaller downloads.** `--weights_precision int8` fetches the same weights stored
as 8-bit with a per-channel scale, expanded back when the model loads. This is a
storage format, not a compute one — inference is unchanged. Measured cost on
rosettafold3: within sampling noise on protein, ligand, RNA and a D/L peptide,
with stereochemistry unchanged.

| `--model` | fp32 | int8 |
|---|---|---|
| `chai1` | 1.20 GB | 0.27 GB |
| `protenix2` | 1.33 GB | 0.19 GB |
| `rosettafold3` | 1.36 GB | 0.27 GB |
| `openfold3` | 1.37 GB | 0.26 GB |
| `openbind0` | 1.31 GB | 0.27 GB |
| `intellifold2` | 1.77 GB | 0.63 GB |
| `boltz2` | 1.88 GB | 0.38 GB |
| `opendde` | 2.47 GB | 0.35 GB |

Each precision caches to its own directory (`<model>-int8/`), so asking for one
never silently gets you the other, and switching back to a form you already have
is instant.

### Downloading the weights yourself

The repo is grouped by family and served over plain HTTPS, so nothing more than
`wget` is needed — useful for pre-staging a shared filesystem or an air-gapped
machine. A model's folder is its family, not its own name: the Protenix
releases share `protenix/`, the eight ESMFold2 releases share `esmfold2/`,
openbind0 sits with `openfold3/`, and the ESM2 and ESM-C towers are under `lm/`.
The four Protenix model types this fork dropped on 2026-09-08 (`protenix05`,
`protenix1_20250630`, `protenix_mini`, `protenix_tiny`) are still published
under `protenix/`; they are simply no longer reachable through `--model`.

```bash
BASE=https://huggingface.co/sokrypton/af3-any-model/resolve/main
mkdir -p params/openfold3
wget -P params/openfold3 $BASE/openfold3/openfold3.bin.zst   # or .int8.bin.zst

python run_alphafold.py --model=openfold3 --model_dir=params/openfold3 ...
```

Locally a model keeps a flat directory named after itself; only the published
layout is grouped. `--model_dir` therefore points at a directory holding
`<model>.bin.zst`, exactly as before.

Or with the Hugging Face CLI:

```bash
pip install huggingface_hub
hf download sokrypton/af3-any-model openfold3/openfold3.bin.zst \
  --local-dir params
```

### Converting the weights yourself

The published blobs are produced by `converters/`, and you can run it yourself
against a checkpoint you already trust. Conversion needs PyTorch; a run never
does.

```bash
# fetches the published checkpoint and converts it
python -m converters.convert --model openfold3 --out ./params/openfold3
```

Then run any model on the same input file:

```bash
python run_alphafold.py \
  --model=openfold3 \
  --model_dir=./params/openfold3 \
  --json_path=fold_input.json \
  --output_dir=./output/
```

### chai-1 needs ESM2 embeddings

Two of the models fold from a protein language model, and are a different model
without it: chai-1 from ESM2 3B, whose token embeddings are most of its token
feature stream (a natural protein folds to 5.70 Å without them where chai
reaches 0.642), and ESMFold2 from ESM-C, which is its alternative to an MSA
rather than an extra on top of one (a variant with no MSA encoder folds to
~14 Å without it).

Both towers are converted and run here, in jax, so one flag covers both. Which
tower is right is read from the model, not chosen on the command line —
ESMFold2's variants are trained against step-matched ESM-C snapshots:

```bash
python run_alphafold.py --model=chai1         --use_esm_embeddings ...
python run_alphafold.py --model=esmfold2_fast --use_esm_embeddings ...
```

The tower runs in-process and is downloaded on demand (2.4 GB for ESM2, 5.1 GB
for ESM-C 6B), which is why it is opt-in rather than the default. Running one of
these models without the flag warns, because the result is a different model
rather than a slightly worse one.

`python -m alphafold3.model.esm` also runs either tower standalone, if you want the hidden
states themselves rather than a fold.

### Cyclic chains

`--cyclic=A,B` (or `--cyclic=all`) makes those chains' relative-position
encoding wrap, so they have no N- or C-terminus. This is not an AlphaFold 3
feature — the input JSON has no way to say it — but the encoding is shared, so
**every model here honours it**. A chain left out is byte-identical to before.

The terms of use written beside a prediction follow the weights that made it,
not AlphaFold 3's. See [converters/README.md](converters/README.md) for what
each conversion involves and [OF3_AF3_PORTING_NOTES.md](OF3_AF3_PORTING_NOTES.md)
for the conventions that differ between these codebases and why.

## Installation and Running Your First Prediction

See the [installation documentation](docs/installation.md).

Once you have installed AlphaFold 3, you can test your setup using e.g. the
following input JSON file named `fold_input.json`:

```json
{
  "name": "2PV7",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

You can then run AlphaFold 3 using the following command:

```
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume <DATABASES_DIR>:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

There are various flags that you can pass to the `run_alphafold.py` command, to
list them all run `python run_alphafold.py --help`. Two fundamental flags that
control which parts AlphaFold 3 will run are:

*   `--run_data_pipeline` (defaults to `true`): whether to run the data
    pipeline, i.e. genetic and template search. This part is CPU-only, time
    consuming and could be run on a machine without a GPU.
*   `--run_inference` (defaults to `true`): whether to run the inference. This
    part requires a GPU.

## AlphaFold 3 Input

See the [input documentation](docs/input.md).

## AlphaFold 3 Output

See the [output documentation](docs/output.md).

## Performance

See the [performance documentation](docs/performance.md).

## Known Issues

Known issues are documented in the
[known issues documentation](docs/known_issues.md).

Please
[create an issue](https://github.com/google-deepmind/alphafold3/issues/new/choose)
if it is not already listed in [Known Issues](docs/known_issues.md) or in the
[issues tracker](https://github.com/google-deepmind/alphafold3/issues).

## Citing This Work

Any publication that discloses findings arising from using this source code, the
model parameters or outputs produced by those should cite:

```bibtex
@article{Abramson2024,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and Bodenstein, Sebastian W. and Evans, David A. and Hung, Chia-Chun and O’Neill, Michael and Reiman, David and Tunyasuvunakool, Kathryn and Wu, Zachary and Žemgulytė, Akvilė and Arvaniti, Eirini and Beattie, Charles and Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and Congreve, Miles and Cowen-Rivers, Alexander I. and Cowie, Andrew and Figurnov, Michael and Fuchs, Fabian B. and Gladman, Hannah and Jain, Rishub and Khan, Yousuf A. and Low, Caroline M. R. and Perlin, Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine and Yakneen, Sergei and Zhong, Ellen D. and Zielinski, Michal and Žídek, Augustin and Bapst, Victor and Kohli, Pushmeet and Jaderberg, Max and Hassabis, Demis and Jumper, John M.},
  journal = {Nature},
  title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493–-500},
  doi     = {10.1038/s41586-024-07487-w}
}
```

## Acknowledgements

AlphaFold 3's release was made possible by the invaluable contributions of the
following people:

Andrew Cowie, Bella Hansen, Charlie Beattie, Chris Jones, Grace Margand,
Jacob Kelly, James Spencer, Josh Abramson, Kathryn Tunyasuvunakool, Kuba Perlin,
Lindsay Willmore, Max Bileschi, Molly Beck, Oleg Kovalevskiy,
Sebastian Bodenstein, Sukhdeep Singh, Tim Green, Toby Sargeant, Uchechi Okereke,
Yotam Doron, and Augustin Žídek (engineering lead).

We also extend our gratitude to our collaborators at Google and Isomorphic Labs.

AlphaFold 3 uses the following separate libraries and packages:

*   [abseil-cpp](https://github.com/abseil/abseil-cpp) and
    [abseil-py](https://github.com/abseil/abseil-py)
*   [Docker](https://www.docker.com)
*   [DSSP](https://github.com/PDB-REDO/dssp)
*   [HMMER Suite](https://github.com/EddyRivasLab/hmmer)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [JAX](https://github.com/jax-ml/jax/)
*   [libcifpp](https://github.com/pdb-redo/libcifpp)
*   [NumPy](https://github.com/numpy/numpy)
*   [pybind11](https://github.com/pybind/pybind11) and
    [pybind11_abseil](https://github.com/pybind/pybind11_abseil)
*   [RDKit](https://github.com/rdkit/rdkit)
*   [Tokamax](https://github.com/openxla/tokamax)
*   [tqdm](https://github.com/tqdm/tqdm)

We thank all their contributors and maintainers!

### Running other models through this code

Everything under [The models this fork runs](#the-models-this-fork-runs) rests on work
by other people. First and most obviously the model authors, who trained and
released the weights — each is linked in that table, and each should be cited
alongside AlphaFold 3 if you use their model.

Beyond them, three whose code this borrows from directly:

*   **[Marielle Russo](https://github.com/maraxen)** —
    [plegadx](https://github.com/maraxen/plegadx), an Equinox/JAX
    re-implementation of RoseTTAFold3, Boltz, chai-1, IntelliFold and AlphaFold 3
    on a shared substrate, each validated against its vendor implementation. Its
    chai-1 diffusion modules were the op-level reference while ours was being
    debugged, and its sampler — replayed to 0.00015 Å under identical noise —
    independently confirmed two details we had derived separately: that the
    vendor loop runs `len(sigmas) - 1` iterations, and that it augments with a
    rotation *and* a translation each step. Its chai-1 trunk diverged in the same
    place ours did, which is its own kind of signpost.

*   **[ChoongHwanLee](https://github.com/chlee19990109-cloud)** — an independent
    Protenix → AlphaFold 3 port
    ([ColabFold, `colabfold2-protenix-proof`](https://github.com/chlee19990109-cloud/ColabFold/tree/colabfold2-protenix-proof)),
    built on this same Haiku graph. Two things here come straight from it. One is
    the padded-key attention mask, which protenix2 and opendde were missing. The
    other is the discipline of asserting that **every** checkpoint tensor is
    consumed exactly once — `dev/audit_coverage.py` is that check (dev-only, out of tree), and it
    found four dropped distogram biases, three LayerNorm offsets and a
    single-conditioner bias that eight ports' worth of correlation gates had all
    scored as passing. Independent agreement on the rest of the Protenix config
    is the strongest evidence either port has.

*   **[juliabuhmann](https://github.com/juliabuhmann)** — the OpenBind port
    ([PR #6](https://github.com/sokrypton/alphafold3/pull/6)). She found that
    OpenFold3's v0.5.0 weights revert two of the deviations this code compensates
    for, worked out which, and built the one-key checkpoint detection that tells
    the two releases apart without a version string. `--model openbind0` is her
    work, re-expressed for this branch's per-model registry.

*   **[Milot Mirdita](https://github.com/milot-mirdita)** —
    [ColabFold](https://github.com/sokrypton/ColabFold) and the public MMseqs2
    API. `src/alphafold3/data/msa_server.py` is adapted from ColabFold's
    `run_mmseqs2`, which is what lets this run with no local sequence database at
    all. The device-portability matrix in `run_alphafold.py` — which attention
    implementation and which XLA flags each GPU generation needs — also comes
    from ColabFold, where every row of it was paid for by a real failure.

## Get in Touch

If you have any questions not covered in this overview, please contact the
AlphaFold team at alphafold@google.com.

We would love to hear your feedback and understand how AlphaFold 3 has been
useful in your research. Share your stories with us at
[alphafold@google.com](mailto:alphafold@google.com).

## Licence and Disclaimer

This is not an officially supported Google product.

Copyright 2024 DeepMind Technologies Limited.

### AlphaFold 3 Source Code and Model Parameters

AlphaFold 3 source code is licensed under the Apache License, Version 2.0 (the
"License"); you may not use its source code except in compliance with the
License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

The AlphaFold 3 model parameters are made available under the
[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
(the "Terms"); you may not use these except in compliance with the Terms. You
may obtain a copy of the Terms at
[https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Unless required by applicable law, AlphaFold 3 and its output are distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. You are solely responsible for determining the appropriateness of
using AlphaFold 3, or using or distributing its source code or output, and
assume any and all risks associated with such use or distribution and your
exercise of rights and obligations under the relevant terms. Output are
predictions with varying levels of confidence and should be interpreted
carefully. Use discretion before relying on, publishing, downloading or
otherwise using the AlphaFold 3 Assets.

AlphaFold 3 and its output are for theoretical modeling only. They are not
intended, validated, or approved for clinical use. You should not use the
AlphaFold 3 or its output for clinical purposes or rely on them for medical or
other professional advice. Any content regarding those topics is provided for
informational purposes only and is not a substitute for advice from a qualified
professional. See the relevant terms for the specific language governing
permissions and limitations under the terms.

### Third-party Software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.

### Mirrored and Reference Databases

The following databases have been: (1) mirrored by Google DeepMind; and (2) in
part, included with the inference code package for testing purposes, and are
available with reference to the following:

*   [BFD](https://bfd.mmseqs.com/) (modified), by Steinegger M. and Söding J.,
    modified by Google DeepMind, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
    See the Methods section of the
    [AlphaFold proteome paper](https://www.nature.com/articles/s41586-021-03828-1)
    for details.
*   [PDB](https://wwpdb.org) (unmodified), by H.M. Berman et al., available free
    of all copyright restrictions and made fully and freely available for both
    non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
*   [MGnify: v2022\_05](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2022_05/README.txt)
    (unmodified), by Mitchell AL et al., available free of all copyright
    restrictions and made fully and freely available for both non-commercial and
    commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
*   [UniProt: 2021\_04](https://www.uniprot.org/) (unmodified), by The UniProt
    Consortium, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
*   [UniRef90: 2022\_05](https://www.uniprot.org/) (unmodified) by The UniProt
    Consortium, available under a
    [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/deed.en).
*   [NT: 2023\_02\_23](https://www.ncbi.nlm.nih.gov/nucleotide/) (modified) See
    the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
*   [RFam: 14\_4](https://rfam.org/) (modified), by I. Kalvari et al., available
    free of all copyright restrictions and made fully and freely available for
    both non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
    See the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
*   [RNACentral: 21\_0](https://rnacentral.org/) (modified), by The RNAcentral
    Consortium available free of all copyright restrictions and made fully and
    freely available for both non-commercial and commercial use under
    [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
    See the Supplementary Information of the
    [AlphaFold 3 paper](https://nature.com/articles/s41586-024-07487-w) for
    details.
