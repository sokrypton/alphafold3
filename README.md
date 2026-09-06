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

## Other AF3-family models

Several groups have published AlphaFold3-architecture models with freely
available weights. This fork runs any of them through this same code and the
same input JSON — one `--model` flag decides which forward branches, config
shapes, sampler constants and input conventions are used.

| `--model` | model | weights |
|---|---|---|
| `alphafold3` | AlphaFold 3 (Google DeepMind) | request from DeepMind |
| `openfold3` | [OpenFold3 preview-2](https://github.com/aqlaboratory/openfold3) (AlQuraishi Lab) | Apache 2.0 |
| `openbind0` | [OpenFold3 v0.5.0 "OpenBind"](https://github.com/aqlaboratory/openfold-3/releases/tag/v0.5.0) (AlQuraishi Lab) | Apache 2.0 |
| `intellifold2` | [IntelliFold-v2](https://huggingface.co/intelligenAI/intellifold) (IntelligenAI) | see upstream |
| `opendde` | [OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) (Aureka Research) | see upstream |
| `boltz2` | [Boltz-2](https://github.com/jwohlwend/boltz) | MIT |
| `protenix2` | [Protenix-v2](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 |
| `rosettafold3` | [RoseTTAFold3](https://files.ipd.uw.edu/pub/rf3/) (RosettaCommons) | see upstream |
| `chai1` | [chai-1](https://github.com/chaidiscovery/chai-lab) (Chai Discovery) | Apache 2.0 |
| `esmfold2` | [ESMFold2](https://huggingface.co/biohub/ESMFold2) (Arc Institute / CZ Biohub) | see upstream |
| `esmfold2_fast` | [ESMFold2-Fast](https://huggingface.co/biohub/ESMFold2-Fast) — half the trunk, no MSA encoder | see upstream |
| `esmfold2_exp` | [ESMFold2-Experimental](https://huggingface.co/biohub/ESMFold2-Experimental) | see upstream |
| `esmfold2_exp_fast` | [ESMFold2-Experimental-Fast](https://huggingface.co/biohub/ESMFold2-Experimental-Fast) | see upstream |
| `esmfold2_exp_cutoff2025` | [ESMFold2-Experimental-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Cutoff2025) | see upstream |
| `esmfold2_exp_fast_cutoff2025` | [ESMFold2-Experimental-Fast-Cutoff2025](https://huggingface.co/biohub/ESMFold2-Experimental-Fast-Cutoff2025) | see upstream |
| `protenix05` | [Protenix v0.5.0](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 |
| `protenix1` | [Protenix-v1](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 |
| `protenix1_20250630` | [Protenix-v1 2025-06-30](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 |
| `protenix_mini` | [Protenix mini](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 |
| `protenix_tiny` | [Protenix tiny](https://github.com/bytedance/Protenix) (ByteDance) | Apache 2.0 |

The six ESMFold2 variants fold from **ESM-C**, not an MSA. Their language-model
input is built outside the fold by `converters.esmc_embed` (the tower, published
int8 at 5.5 GB) and `converters.esmfold2_lm` (the per-model shim, which rides
along with the weights). Given an MSA instead they fold from that: 5CAJ reads
17.5 A from a single sequence and 1.24 A at MSA depth 256. Supplying neither is
the one broken configuration.

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

The repo is flat and served over plain HTTPS, so nothing more than `wget` is
needed — useful for pre-staging a shared filesystem or an air-gapped machine:

```bash
BASE=https://huggingface.co/sokrypton/af3-any-model/resolve/main
mkdir -p params/openfold3
wget -P params/openfold3 $BASE/openfold3.bin.zst        # or openfold3.int8.bin.zst
wget -P params/openfold3 $BASE/openfold3.shapes.json

python run_alphafold.py --model=openfold3 --model_dir=params/openfold3 ...
```

Or with the Hugging Face CLI:

```bash
pip install huggingface_hub
hf download sokrypton/af3-any-model openfold3.bin.zst openfold3.shapes.json \
  --local-dir params/openfold3
```

Take the `.shapes.json` too. It is small, and it is what reports a gap in the
conversion at load time instead of letting it surface as an opaque failure
mid-forward.

### Converting the weights yourself

The published blobs are produced by `converters/`, and you can run it yourself
against a checkpoint you already trust. Conversion needs PyTorch; a run never
does.

```bash
# fetches the published checkpoint, converts it, writes the shape manifest
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

chai-1 folds with ESM2 token embeddings, which are most of its token feature
stream — without them it is a different model (a natural protein folds to 5.70 Å
where chai reaches 0.642). Precompute them once, in whatever environment has
chai's traced ESM archive, and pass them in:

```bash
python converters/esm_embed.py --sequence MQIFVKT... --out esm.npz
python run_alphafold.py --model=chai1 --esm_embeddings=esm.npz ...
```

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

Everything under [Other AF3-family models](#other-af3-family-models) rests on work
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
    consumed exactly once — `converters/audit_coverage.py` is that check, and it
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
