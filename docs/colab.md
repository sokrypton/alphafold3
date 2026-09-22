# The ColabFold2 preview notebook

Reference for
[ColabFold2_preview.ipynb](https://colab.research.google.com/github/sokrypton/alphafold3/blob/colab/ColabFold2_preview.ipynb).
The notebook itself keeps a quick start and links here, because a wall of
tables in front of a form is what a new reader meets first and it should not
be. Nothing here is notebook-only: the models, the input grammar and the
output files are the same from the CLI.

## Options

The run cell's fields. A field for the other family is ignored, not an error,
and **0 means the model's own default**.

| field | family | what it does |
|---|---|---|
| `num_recycles` | either | Refinement passes. 3 here, which is AlphaFold 2's own number and ColabFold's habit; AlphaFold 3's own is 10, so this trades a little accuracy on hard targets for seven fewer trunk passes. Raise it for anything unconverged. |
| `num_msa` | either | MSA rows the trunk reads, 512 here. AlphaFold 2's second, "extra" stack follows at twice this. Barely a speed dial — 1 against 1024 measured 0.4% of runtime at 512 tokens — but it is the knob for an out-of-memory. |
| `live_view` | either | Fold in the notebook's own process and stream it: AlphaFold 3 draws every denoising step and a contact map per recycle, AlphaFold 2 draws every recycle, being the only intermediate structure it has. A frame is drawn one step behind the maths, so the fold never waits for the picture. It writes the same output folder as a normal run. **Not bit-identical**: each stage restarts haiku's rng, so a structure moves about as much as a different seed would (1.52 Å, against 0.47–1.35 Å between seeds). Untick it for a bit-identical run. |
| `num_diffusion_samples` | AF3 | Structures per seed, so the total is seeds × samples. Both modes fold all of them; the live view animates the first. |
| `diffusion_steps` | AF3 | Denoising steps, 32 here against AlphaFold 3's own 200. Below about 20 the sampler does not land — ten gives 5.91 Å on 6MRR against a Cα–Cα of 8.40. |
| `af2_num_models` | AF2 | AlphaFold 2 ships five separately trained parameter sets — five models, not five seeds — and they appear as the samples of a seed. Running several and letting the ranking sort them is what ColabFold's `num_models` does. |

## Models

Ported weights come from [sokrypton/af3-any-model](https://huggingface.co/sokrypton/af3-any-model).
"Tower" is a protein language model fetched separately on first use.

| model | weights | licence | notes |
|---|---|---|---|
| `openbind0` | [OpenFold3 v0.5.0 "OpenBind"](https://github.com/aqlaboratory/openfold-3/releases/tag/v0.5.0) (AlQuraishi Lab) | Apache-2.0 | The current release, and the default here. |
| `openfold3` | [OpenFold3 preview-2](https://github.com/aqlaboratory/openfold) (AlQuraishi Lab) | Apache-2.0 | The earlier preview, kept because earlier results used it. |
| `boltz2` | [Boltz-2](https://github.com/jwohlwend/boltz) (Wohlwend et al.) | MIT | Strong on ligands; keeps a modified residue as one token. |
| `protenix2` | [Protenix-v2](https://github.com/bytedance/Protenix) (ByteDance) | Apache-2.0 | The widest trunk here (pair 256), so the slowest. |
| `rosettafold3` | [RoseTTAFold3](https://github.com/RosettaCommons/foundry) (RosettaCommons) | BSD-3-Clause | Carries chirality features; handles D-amino acids. |
| `chai1` | [chai-1](https://github.com/chaidiscovery/chai-lab) (Chai Discovery) | Apache-2.0 | Folds from ESM2 3B, fetched and run automatically. |
| `intellifold2` | [IntelliFold-v2](https://huggingface.co/intelligenAI/intellifold) (IntelliGen-AI) | Apache-2.0 | Widened channels (pair 512), largest ported download. |
| `opendde` | [OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) (Aureka Research) | Apache-2.0 | Runs its diffusion on an expanded structural-token set. |
| `esmfold2` | [ESMFold2](https://huggingface.co/biohub/ESMFold2) (Chan Zuckerberg Biohub) | MIT | Folds from ESM-C instead of an MSA — single sequence, no search. |
| `esmfold2_lm600m` | ESMFold2, 600M tower | MIT | No confidence head. |
| `esmfold2_lm300m` | ESMFold2, 300M tower | MIT | No confidence head. |
| `af2_ptm` | AlphaFold 2 monomer pTM (DeepMind) | CC BY 4.0 | **Protein only** — a ligand or nucleotide raises rather than quietly folding the rest. Templates use the model_1/model_2 parameter sets. |
| `af2_multimer` | AlphaFold 2 multimer v3 (DeepMind) | CC BY 4.0 | Protein only, as above. |
| `alphafold3` | Google DeepMind's own parameters | [AF3 terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) | DeepMind's public release, downloaded on first use (~1 GB). Its terms govern the weights and the outputs; run_alphafold prints them at startup. |
---

## Input

Each molecule type has its own box; within a box, separate chains with `:`.

| box | contents | example |
|---|---|---|
| **protein** | amino-acid sequence(s) | `MKTAY...` or `SEQ1:SEQ2` |
| **dna** | DNA sequence(s) | `CGCGAATTCGCG` |
| **rna** | RNA sequence(s) | `GCGGAUUUA` |
| **ligand_ccd** | ligand(s) by PDB CCD code | `ATP:MG:HEM` |
| **ligand_smiles** | ligand(s) by SMILES | `CC(=O)Oc1ccccc1C(=O)O` |

Mix boxes freely to build a complex. Chain IDs A, B, C… follow AlphaFold 3's canonical
order (protein → RNA → DNA → ligand). Identical protein sequences are merged, so
`SEQ:SEQ` is a homodimer. Sequences and CCD codes are upper-cased; **SMILES are left
as typed**. Whitespace and extra colons are forgiven (`SEQ1::::SEQ2` = `SEQ1:SEQ2`) —
which is also why an atom-mapped SMILES containing `:` needs a raw AF3 JSON instead.

**seeds**: comma-separated, one prediction each (`1,2,3`). Junk and duplicates are
dropped. **msa_mode**: `mmseqs2_server` queries the public
[ColabFold](https://colabfold.mmseqs.com/) API (protein only — RNA/DNA always run
MSA-free); `single_sequence` skips it, faster and less accurate.

## Output

| file | contents |
|---|---|
| `*.cif` | Best-ranked structure. B-factor = pLDDT (0–100). |
| `*_confidences.json` | Per-residue pLDDT, PAE matrix, contact probabilities. |
| `*_summary_confidences.json` | Mean pLDDT, pTM, ipTM, ranking score. |
| `*_ranking_scores.csv` | Every seed × sample combination. |
| `seed-N_sample-M/` | One directory per prediction. |
| `TERMS_OF_USE.md` | The licence for whichever weights you ran. |

pLDDT above 90 is very high, 70–90 reliable backbone, 50–70 doubtful, below 50 likely
disordered or wrong. Lower PAE means two residues are confidently placed *relative to
each other*, which is what to read for an interface. ipTM above 0.8 is a well-defined
complex interface, and is `n/a` for a single chain — there is no interface to score.

## Troubleshooting

**OOM**: shorter sequence, or a larger GPU (`Runtime → Change runtime type`).
**MSA server timeout**: the public server is rate-limited — retry, or use
`single_sequence`. **Download popup blocked**: disable your ad blocker.

## Licence

The AlphaFold 3 **source code** is [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0);
the **weights** are each their own, as listed in the model table. Outputs from the seven
Apache/MIT/BSD-licensed ported models are **not** subject to DeepMind's AlphaFold 3 Output
Terms of Use and may be used freely, including commercially. `alphafold3` is the exception:
its parameters and outputs carry DeepMind's own terms. Every run writes a
`TERMS_OF_USE.md` naming the licence that actually applies to it.

## Bugs / feedback

https://github.com/sokrypton/alphafold3/issues
