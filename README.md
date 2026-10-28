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

This fork, published on PyPI as `alphafold3-colabfold`, runs **19 models**:
17 on AlphaFold 3's graph (`model_config.MODELS`) and two AlphaFold 2 models.
One `--model` flag decides which forward branches, config shapes, sampler
constants and input conventions are used, and the same input JSON drives all of
them. The registry is `src/alphafold3/model/model_registry.py`.

Two ENGINES live here. Everything except AlphaFold 2 rides the shared AlphaFold 3
graph — a pair+single trunk into a diffusion sampler — so adding one is a weight
remap plus a branch; BioEmu2 is the partial exception, sharing only the sampler
and output path. AlphaFold 2 shares neither end (MSA row/column attention, and
an IPA head over backbone frames) and runs as a sibling network under
`alphafold3.af2`, reached through the same CLI and the same output writer.
Check `model_registry.get(name).engine` (`af3` or `af2`) rather than testing the
name. What each port needed, and the input conventions each family requires, are
in [docs/ported_models.md](docs/ported_models.md).

### AlphaFold 3 lineage

| `--model` | model | weights |
|---|---|---|
| `alphafold3` | AlphaFold 3 (Google DeepMind) | request from DeepMind |
| `openfold3` | [OpenFold3 preview-2](https://github.com/aqlaboratory/openfold3) (AlQuraishi Lab) | Apache 2.0 |
| `openbind0` | [OpenFold3 v0.5.0 "OpenBind"](https://github.com/aqlaboratory/openfold-3/releases/tag/v0.5.0) | Apache 2.0 |
| `intellifold2` | [IntelliFold-v2](https://huggingface.co/intelligenAI/intellifold) (IntelligenAI) | Apache 2.0 |

### Protenix family

Protenix publishes nine model types that differ only in counts and widths;
`derive_dims` + `PROTENIX_FAMILY` make each one close to a one-liner. Two of
ByteDance's are wired up, plus Mol-1, a fork. The others (`protenix05`, `protenix1_20250630`, `protenix_mini`,
`protenix_tiny`) were dropped on 2026-09-08 to keep parity work focused; re-adding
one is a `sources.py` entry, a converter alias and a `MODELS` line.

| `--model` | model | weights |
|---|---|---|
| `protenix2` | [Protenix-v2](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 |
| `protenix1` | Protenix-v1 | Apache 2.0 |
| `mol1` | [Mol-1](https://github.com/molence/Mol-1) (Molence Bio), a Protenix-v2 fork | Apache 2.0 |

Mol-1 runs Protenix-v2's model code at a wider pair representation (c_z 448),
with its 48-block trunk split in two: 8 "PairMixer" blocks (no triangle
attention) at 448, then 40 pairformer blocks at 320 between two bias-free width
adapters. Its own converter (`converters/mol1.py`) maps that split; everything
else reuses the Protenix mappers. It recycles 20 times by default, against
Protenix's 10. Its trunk single and confidence head stay float32 in bf16 mode:
stock Mol-1's default bf16 run flattens pLDDT to ~89 for any input (see
[PARITY.md](PARITY.md#mol-1)).

### Other AF3-architecture models

| `--model` | model | weights |
|---|---|---|
| `boltz2` | [Boltz-2](https://github.com/jwohlwend/boltz) | MIT |
| `opendde` | [OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) (Aureka Research) | Apache 2.0 |
| `rosettafold3` | [RoseTTAFold3](https://github.com/RosettaCommons/foundry) (RosettaCommons) | BSD 3-Clause |
| `chai1` | [chai-1](https://github.com/chaidiscovery/chai-lab) (Chai Discovery) | Apache 2.0 |
| `helixfold3` | [HelixFold3](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold3) (Baidu PaddleHelix, 2025-07-14 weights) | **non-commercial** (NC-SA) |

HelixFold3's code and weights are licensed for non-commercial use only, under
Attribution-NonCommercial-ShareAlike terms that also forbid incorporating it into
unauthorised automated systems (screening or design workflows). Its checkpoint is a
PaddlePaddle `.pdparams`, converted without paddle (`converters/helixfold3.py`).

### ESMFold2 family — folds from ESM-C, not an MSA

Pair-only trunk conditioned on an ESM-C language model.

| `--model` | model | weights |
|---|---|---|
| `esmfold2` | [ESMFold2](https://huggingface.co/biohub/ESMFold2) (Arc / CZ Biohub) | MIT |
| `esmfold2_fast` | [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast) — half the trunk | MIT |
| `esmfold2_lm600m` | […-base600M-step1500k](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-base600M-step1500k) — **ESM-C 600M** | MIT |
| `esmfold2_lm300m` | […-base300M-step1500k](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-base300M-step1500k) — **ESM-C 300M** | MIT |

All but the last two condition on ESM-C 6B; those use the 600M and 300M towers
and cost 0.5 GB and 0.3 GB against 5.5. Upstream ships the "Experimental" line
for paper reproducibility rather than research use — its own card says to prefer
plain `esmfold2`. The language-model input is built outside the fold by
`alphafold3.model.esm` (tower, published int8 at 5.5 GB) plus a per-model shim
that ships with the weights. **Passing one variant another's tower reads corr
0.026 against native** — take the pairing from
`model_registry.ESMFOLD2_VARIANTS[name]['esmc']`. Given an MSA instead they fold
from that: 5CAJ reads 17.5 Å from a single sequence and 1.24 Å at MSA depth 256.
Supplying neither is the one broken configuration.

### BioEmu2 — its own trunk and denoiser

| `--model` | model | weights |
|---|---|---|
| `bioemu2` | [BioEmu2](https://github.com/microsoft/bioemu2) `bioemu2-preview` (Microsoft Research) | MIT |

Boltz-derived, but neither of its stages is AF3's: a PairMixer trunk (Boltz's
MSA module without triangle attention, then 48 triangle-multiplication blocks)
and a dual-stream denoiser that runs atom and token transformers side by side.
The weights download on first use like the others' (int8 by default: 0.02-0.05 A
CA RMSD from float32 on the same seed, against 0.9 A between two samples).
It runs as its own network (`network/bioemu2.py`) on its own features
(`pipeline/bioemu2_features.py`, built from the same input JSON) and shares
the sampler and the output path. There is no confidence head, so confidence
fields are written as NaN. Every sample also gets BioEmu2's protein physicality
report (`<name>_physicality.json`).

BioEmu2 is more than a structure predictor, and its other modes are here too:

| flag | what it does |
|---|---|
| `--bioemu2_mode=equilibrium` | sample its equilibrium ensemble instead of predicting a structure |
| `--bioemu2_propagate_from=<cif>` | propagate from a structure: each sample a trajectory of `--bioemu2_propagation_steps` steps of `--bioemu2_lag_ns` |
| `--bioemu2_sampler=edm_gd\|edm_fk\|edm_fkc` | steer sampling with its physicality potentials (gradient guidance, Feynman-Kac resampling, the FK corrector); the samples are the particles |
| `--bioemu2_potentials=<yaml>` | steer with a BioEmu2 `--potentials` file instead of the default potentials |
| `--bioemu2_contacts=<json>` | condition on contacts, in BioEmu2's `Contact` form |
| `--bioemu2_physicality=<preset>` | the physicality report's thresholds (`high_quality` by default); nothing is removed |

Steering fetches two pieces of reference data on first use rather than
shipping them: the bond-geometry tables BioEmu2 bundles (CCP4 MON_LIB, LGPL)
and Boltz-1's CCD molecules for ligand restraints. A potential written in
Python is out of reach (the YAML can name BioEmu2's own potentials only), and
`--cyclic` is refused, as BioEmu2 has no cyclic input.

Where it differs from the BioEmu2 CLI:

- Chemical components come from AF3's bundled CCD (or the input's `userCCD`),
  not fetched live from RCSB. A component revised since AF3's CCD snapshot
  (HEM, for one) gets the older reference geometry and charges.
- The MSA is the input JSON's, not fetched from ColabFold.
- Samples default to `--num_diffusion_samples` (5), not BioEmu2's 8. For the
  steered samplers that is also the particle count.
- Inputs BioEmu2 rejects (chain ids over 4 characters; B, Z, U, O, J in a
  protein sequence) run here, the letters as unknown residues.
- Multi-step propagation is refused under `edm_fk`/`edm_fkc`, as BioEmu2
  refuses it; an empty potentials YAML is refused rather than run unsteered.

### AlphaFold 2 — a sibling network, not the AF3 graph

| `--model` | model | weights |
|---|---|---|
| `af2_ptm` | AlphaFold 2 monomer pTM (`params_model_*_ptm.npz`) | CC BY 4.0 |
| `af2_multimer` | AlphaFold 2 multimer v3 (`params_model_*_multimer_v3.npz`) | CC BY 4.0 |

DeepMind's own AlphaFold 2 parameters, read from `--model_dir` as
`params_model_*.npz` (directly or under `params/`); if the directory holds none,
they are downloaded there from DeepMind's own release
(`weights.AF2_PARAMS_URL`). Nothing is converted or republished. Monomer and
multimer run on ONE graph: a monomer checkpoint is converted onto the multimer
network at load. **Protein only** — a fold input carrying a ligand, a nucleotide
or an inter-chain bond raises rather than silently folding the protein subset.
MSAs come from the AF3 data pipeline, chain pairing included, and the MSA
pipeline is the model's own: subsampling, the cluster/extra split, BERT masking
and clustering all run inside JAX under `--seed`, as they do upstream.

**Templates work on both paths.** A fold input carrying a template switches on
the template embedder and the template-trained parameter sets — no flag, the
input decides. Monomer templates exist only in `model_1` and `model_2`
upstream, and only those two are used when templates are requested. Template
residue types use AF2's `restype` alphabet, not the HHBLITS order the search
tools emit; feeding the wrong one is silent and costs about 1.2 Å.

`--num_recycles` and `--dropout` reach AF2 exactly as they reach AF3.

AF2 is measured single-seed at float32 with zero recycles, not best-of-5 like
the AF3 lineage, so read its numbers in
[PARITY.md](PARITY.md#fold-accuracy) against each other rather than against the rest.

## Parity

Each model is checked against **its own vendor's stock code**: both sides run their shipped entry points on the same input with the same random draws. Every model matches its vendor on every input that vendor can take, across 23 input types (HelixFold3 is graded against its own run-to-run spread and has one row at the edge of it; see PARITY.md). The comparison covers the trunk, a denoise step, the confidence heads, padded inputs and the full diffusion trajectory. AlphaFold 3 is compared against DeepMind's own repository, and AlphaFold 2 passes 18 module gates against DeepMind's original code.

| `--model` | stock reference | input types | full trajectory, final coordinates |
|---|---|---|---|
| `alphafold3` | DeepMind alphafold3 v3.0.4 | 23/23 | — |
| `boltz2` | boltz 2.2.1 | 23/23 + cyclic | 0.025 Å |
| `chai1` | chai_lab 0.6.1 | 20/20 (SMILES ligands only) | 0.001 Å |
| `protenix2` / `protenix1` | Protenix v2.0.0 | 23/23 | 0.012 Å / 0.007 Å superposed* |
| `opendde` | OpenDDE v1.1.0 | 23/23 | 0.000 Å |
| `rosettafold3` | foundry (aad357b) | 21/21 | 0.007 Å |
| `openfold3` / `openbind0` | OpenFold3 0.4.5 / v0.5.0 | 21/21 (no bond input) | 0.000 Å |
| `intellifold2` | IntelliFold v2.0.4 | 23/23 | 0.005 Å |
| `esmfold2` / `esmfold2_fast` | esm 3.4.1 | 23/23 | exact |
| `esmfold2_lm600m` / `lm300m` | Synthyra FastPLMs | 20/20 (no MSA input) | exact |
| `bioemu2` | BioEmu2 (7846769) | 20/20 (no template input) | 0.000 Å |
| `mol1` | Mol-1 (bfc01fa) | 24/24 (no cyclic input) | 0.066 Å |
| `helixfold3` | PaddleHelix (8e3991a) | 20/20 against its own spread; one row at the edge | 0.000 Å |
| `af2_ptm` / `af2_multimer` | DeepMind alphafold | 18 module gates | — |

\* protenix1's final structure comes out rotated by 0.9° as a whole. Stock protenix1 rotates itself by 0.38° when its own trunk outputs are nudged by 1e-5, so this is the model's sensitivity, not a port difference.

BioEmu2's other modes are graded the same way: equilibrium sampling,
propagation, all three steering presets with default and custom potentials,
contact conditioning and the physicality report (see PARITY.md).

[PARITY.md](PARITY.md) has the method, the per-model results, which inputs each vendor cannot take, and the three stock OpenFold3 0.4.5 bugs the harness patches. It also has the module-level gates and fold accuracy. [docs/parity_log.md](docs/parity_log.md) is the dated notebook of how each finding was made.

## Running the other models

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

**Download precision.** `--weights_precision` picks which published form to
fetch, and **int8 is the default**: the same weights stored as 8-bit, expanded
back when the model loads. This is a storage format, not a compute one —
inference is unchanged. `fp32` is the exact bytes the converters wrote. Measured cost on rosettafold3: within
sampling noise on protein, ligand, RNA and a D/L peptide, with stereochemistry
unchanged; on 6MRR the int8 blobs fold at float32's accuracy (boltz2 0.557 vs
0.558, opendde 1.194 vs 1.197, openfold3 1.642 vs 1.639 CA-RMSD).

Scales are per output channel and per block of 128 rows down the reduction axis,
halved until the tensor is within 1e-2 relative rms, so one outlying row cannot
set the step for a whole column. Blobs written under the older single-scale
scheme (before 2026-09-18) still load. `fp16` is the same idea at half precision.

| `--model` | fp32 | fp16 | int8 |
|---|---|---|---|
| `chai1` | 1.20 GB | 0.59 GB | 0.31 GB |
| `protenix1` | 0.99 GB | 0.47 GB | 0.23 GB |
| `protenix2` | 1.33 GB | 0.61 GB | 0.29 GB |
| `rosettafold3` | 1.36 GB | 0.68 GB | 0.35 GB |
| `openfold3` | 1.37 GB | 0.68 GB | 0.35 GB |
| `openbind0` | 1.37 GB | 0.68 GB | 0.35 GB |
| `intellifold2` | 1.77 GB | 1.37 GB | 0.80 GB |
| `boltz2` | 1.88 GB | 0.94 GB | 0.49 GB |
| `opendde` | 2.47 GB | 0.94 GB | 0.54 GB |
| `esmfold2` | 0.87 GB | 0.43 GB | 0.23 GB |
| `esmfold2_fast` | 0.70 GB | 0.35 GB | 0.18 GB |
| `esmfold2_lm600m` | 0.66 GB | 0.33 GB | 0.17 GB |
| `esmfold2_lm300m` | 0.66 GB | 0.33 GB | 0.17 GB |
| `bioemu2` | 1.90 GB | 0.95 GB | 0.49 GB |
| `helixfold3` | 1.37 GB | 0.68 GB | 0.36 GB |
| `mol1` | 2.15 GB | 0.73 GB | 0.50 GB |

HelixFold3's trunk is chaotic under recycling (stock's own trunk moves visibly under a 1e-6 nudge, see
PARITY.md), so a storage rounding can carry a sample to a different nearby structure: on one test protein
one of two samples landed 1.6 A (fp16) / 2.5 A (int8) CA RMSD from fp32's and the other 0.1 A, against
0.3 A between fp32's own two samples, at the same ranking score. Use `fp32` where that matters.

Mol-1 is not chaotic under recycling, but on an input with several distinct conformations a rounding can
still tip a sample into a neighbouring one. On 7r6r (a protein-DNA complex whose stock samples span 2-11 A
CA RMSD of each other), three of five samples stayed within 0.2-0.6 A of fp32's and two moved 1.5-3.6 A
(fp16 and int8 alike), with pTM, ipTM and ranking score unchanged.

The ESMFold2 rows exclude the ESM-C tower, which is fetched separately (below).

Each precision caches to its own directory (`<model>-int8/`, `<model>-fp16/`;
fp32 uses the bare `<model>/`), so asking for one
never silently gets you the other, and switching back to a form you already have
is instant.

### Downloading the weights yourself

The repo is grouped by family and served over plain HTTPS, so nothing more than
`wget` is needed — useful for pre-staging a shared filesystem or an air-gapped
machine. A model's folder is its family, not its own name
(`model_registry.WEIGHTS_FOLDERS`): the Protenix releases share `protenix/`
(Mol-1 has its own `mol1/`), the four ESMFold2 models share `esmfold2/`,
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

### chai-1 and ESMFold2 fold with their language model

Two of the models fold from a protein language model, and are a different model
without it: chai-1 from ESM2 3B, whose token embeddings are most of its token
feature stream (a natural protein folds to 5.70 Å without them where chai
reaches 0.642), and ESMFold2 from ESM-C, which is its alternative to an MSA
rather than an extra on top of one (a variant with no MSA encoder folds to
~14 Å without it).

Both towers are converted and run here, in jax, and both models run with theirs
by default. Which tower is right is read from the model, not chosen on the
command line — ESMFold2's variants are trained against step-matched ESM-C
snapshots. The tower runs in-process and is downloaded on first use (2.5 GB for
ESM2, 5.5 GB for ESM-C 6B). `--nouse_esm_embeddings` folds without it, and warns,
because the result is a different model rather than a slightly worse one:

```bash
python run_alphafold.py --model=chai1 ...                          # with ESM2
python run_alphafold.py --model=chai1 --nouse_esm_embeddings ...   # without
```

`--use_esm_embeddings` on any other model is an error.

`python -m alphafold3.model.esm` also runs either tower standalone, if you want the hidden
states themselves rather than a fold.

### Cyclic chains

`--cyclic=A,B` (or `--cyclic=all`) makes those chains' relative-position
encoding wrap, so they have no N- or C-terminus. This is not an AlphaFold 3
feature (the input JSON has no way to say it), but the encoding is shared, so
**every model on the AF3 graph honours it**, except BioEmu2, which has no
cyclic input and refuses it; AlphaFold 2 does not read it. Only Boltz-2 was trained with it, so for the others it is an
extrapolation; stock IntelliFold accepts the flag and ignores it. A chain left out is byte-identical to before.

### Crosslinks as explicit ligands

A crosslinking mass-spec result can be given the way
[af3x](https://github.com/KosinskiLab/af3x) takes it
([Obarska-Kosinska & Kosinski, bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.12.03.626671v2)):
as the crosslinker itself, covalently bonded to the two residues it joined,
rather than as a distance restraint. The input is af3x's:

```json
"crosslinks": [
  {"name": "DSSO", "residue_pairs": [[["A", 2], ["B", 3]], [["A", 8], ["B", 11]]]},
  {"name": "sulfo-SDA", "residue_pairs": [[["A", 16], ["B", 20]]]}
]
```

Each pair becomes one ligand chain (on the chain ids after the job's last) and two
`bondedAtomPairs`: nothing is model-specific and no weights are involved, so
each model treats a crosslinker as it treats any covalently bonded ligand. An NHS-ester end
acylates LYS NZ, SER OG, THR OG1, TYR OH or residue 1's N; a diazirine end (the
SDA family; a pair's second residue) bonds to the residue's last side-chain
atom, CA for glycine. Known: DSS, BS3, DSG, BS2G, DSSO, DSBU, CDI, PhoX,
BS(PEG)5, BS(PEG)9, azide-A-DSBSO, SDA, LC-SDA, SDAD (sulfo- forms take the
same bridge).

This is an independent implementation of af3x's input, not its code: each
crosslinker is written as its reacted bridge from the public structure
([src/alphafold3/common/crosslinks.py](src/alphafold3/common/crosslinks.py),
checked against each supplier's formula) and generated into
`crosslinkers.cif`, so geometry and atom names are not af3x's. af3x's options
are here too: `--remove_overlapping_crosslinks` (default true: a pair touching
an already crosslinked residue is dropped and reported) and
`--sample_crosslink_combinations=k` (every k-subset as its own job). af3x's
experimental dynamic-length and disulfide links are not.

A declared bond is a feature, not a hard constraint, so how firmly it holds is the
model's: on two DSSO crosslinks the structure can satisfy (localfold's page, with
an MSA), AlphaFold 3 closed all four residue-bridge bonds and Boltz-2 two. Pairs the
model's confident structure cannot satisfy get broken, as af3x notes.

### Licences and porting notes

The terms of use written beside a prediction follow the weights that made it,
not AlphaFold 3's. See [converters/README.md](converters/README.md) for what
each conversion involves and [docs/of3_af3_porting_notes.md](docs/of3_af3_porting_notes.md)
for the conventions that differ between these codebases and why, and
[docs/ported_models.md](docs/ported_models.md) for what each later port needed.

## Installation and Running Your First Prediction

See the [installation documentation](docs/installation.md). To run in Google
Colab instead, with every model above, use ColabFold's
[ColabFold2_preview.ipynb](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ColabFold2_preview.ipynb)
(it installs this package from PyPI).

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

```bash
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

See the [performance documentation](docs/performance.md). Runtime and peak
memory of the ported models by sequence length are in
[docs/benchmarks/README.md](docs/benchmarks/README.md).

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
    consumed exactly once — `dev/audit_coverage.py` is that check (dev-only, not in the package), and it
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
