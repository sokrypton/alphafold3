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

This fork runs **18 model types**, and the question for every one is whether it
reproduces its own vendor's implementation rather than merely producing a
plausible structure. `PARITY.md` is the full record and `dev/oracles/HOLES.md`
tracks what is still open; this is the current state.

Parity is measured at seven levels, from the weights inwards to the fold:

| level | what it compares |
|---|---|
| **L0** | conversion coverage — every checkpoint tensor accounted for, both directions |
| **L1** | the trunk: z-init, pairformer, MSA module, template embedder, distogram |
| **L2** | diffusion conditioning, token transformer, atom encoder and decoder |
| **L3** | one full denoise step |
| **L4** | the confidence head: PAE, PDE, pLDDT, resolved |
| **L5** | an end-to-end fold, scored against an experimental structure |
| **L6** | modality: RNA, DNA, ligands, complexes, modified residues |

### Current numbers (2026-09-11)

L0 through L5 have all been driven to completion across every model they apply
to. Grading each comparison on correlation **and** `max|d|/rms`:

```
274 comparisons in 131 logs
PARITY=237   CLOSE=8   FLOOR=17   LOOSE=10   BAD=2
```

`bash dev/oracles/run_all_parity.sh all` re-runs everything;
`dev/oracles/parity_audit.py <logdir>` grades it and
`dev/oracles/gate_applies.py <gate> <model>` answers whether an empty cell is a
hole or genuinely not applicable.

**FLOOR** is not a pass. It means the cell has no resolution left: perturbing the
gate's own input by 1e-6 moves its output further than our port differs, so the
number cannot be read. All 17 are the trunk pairformer on synthetic input, where
the stack is driven far outside its trained distribution — the shallower
`L1.trunk1` is the measurement that counts. A cell only earns FLOOR by measuring
it, so this cannot quietly excuse anything.

**The 2 BAD** are `esmfold2_fast`'s PAE and PDE, and they are understood rather
than open: native runs its confidence pairformer's triangle multiplications
under `autocast(bfloat16)` — its tensors round-trip through bf16 at max|d| 0 —
while we run fp32, which costs ~0.5% per sub-module with bit-identical weights
on identical inputs.

### Folds

6MRR, best of 5 samples, against each model's recorded baseline:

| model | best Å | | model | best Å |
|---|---|---|---|---|
| `boltz2` | 0.421 | | `intellifold2` | 1.515 |
| `alphafold3` | 0.626 | | `openfold3` | 1.544 |
| `protenix2` | 0.697 | | `esmfold2_lm600m` | 1.522 |
| `opendde` | 0.737 | | `openbind0` | 1.650 |
| `rosettafold3` | 1.026 | | `protenix1` | 1.684 |
| `esmfold2_fast` | 1.181 | | `esmfold2_lm300m` | 1.687 |
| `esmfold2` | 1.339 | | `chai1` | 1.723 |

Read these as a band, not a ranking: several models are **nondeterministic run
to run on the same seed**. Three runs of `rosettafold3` seed 0 give 0.968, 0.949
and 1.026, because one sample sits near a decision boundary. A single fold
number to three decimals is not a gate.

### Modality

| model | protein+ligand | complex | RNA | DNA | confidence |
|---|---|---|---|---|---|
| `intellifold2` | 0.301 / 0.881 | 0.92 | 1.621 | 1.95 | 90.4 / r .776 |
| `boltz2` | 0.385 / 0.907 | 0.92 | 1.419 | 1.59 | 96.9 / r .830 |
| `protenix2` | 2.497 / 1.147 | 1.02 | 2.062 | 2.21 | 92.7 / r .807 |
| `rosettafold3` | 0.464 / 0.889 | 1.12 | 1.143 | 2.67 | 85.2 / r .708 |
| `opendde` | 1.269 / 0.870 | 1.49 | 1.372 | 1.99 | 89.9 / r .635 |
| `chai1` | 0.335 / 0.993 | 1.57 | 1.648 | 1.89 | 78.9 / r .543 |
| `openfold3` | 0.348 / 0.894 | 1.60 | 1.441 | 2.22 | 83.2 / r .678 |

`r` is the column that matters for confidence: a constant head reads r ≈ 0
however plausible its mean looks. All seven predict error.
`esmfold2_lm600m` and `esmfold2_lm300m` ship no confidence head by design.

Every model is differentiable in the sequence (24/24, `dev/oracles/grad_check.py`),
which is what the design path needs.

### What is still open

Tracked in `dev/oracles/HOLES.md` with the next measurement named for each:

* **`boltz2`'s denoise step**, 0.207 Å/atom against a 4e-06 noise floor, so
  real. Its gate's dump holds a single sampler step at the highest-noise end of
  the schedule.
* **`chai1`'s confidence**, two LOOSE rows, reachable only by injection because
  its modules ship as TorchScript with no callable `forward`.
* **the protenix lineage's confidence**, 6 CLOSE + 3 LOOSE, most likely native's
  own float32 — its stack is exact and amplifies 7×, and native's fp32-vs-fp64
  noise through the embedding is the same order as the whole gap. Consistent
  with, not established.
* **three things no cell covers**: `L1b.msa_nonuniform` is a silent duplicate of
  the uniform cell for 13 of 14 models; `real_trunk_parity.py` exists and the
  driver never runs it; `boltz2`'s template module is V2 upstream and we
  implement V1 (inert on one chain, live on a complex).

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
