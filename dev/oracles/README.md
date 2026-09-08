# oracles

Harnesses that compare a port against something authoritative. Nothing here is
imported by the served model; everything is runnable on its own.

Two standing rules, both learned the hard way:

* **Run on the GPU.** Three scripts here used to force `jax_platform_name='cpu'`
  because the float32 ESM-C tower did not fit a 23 GB card. It does at int8, and
  those scripts are gone. CPU hides the flash-attention path.
* **Serialise.** One fold at a time; two concurrent jobs OOM the card.

## Cross-model

| file | what it answers |
|---|---|
| `fold_check.py` | Does model X fold a target? The pipeline gate every port must pass. `LM_PAIR=`/`ESMC_HIDDEN=` supply ESMFold2's language model; `BF16=none` forces float32. |
| `prot_parity.py` | Do our converted BLOCKS compute what native's do, on synthetic activations? Says nothing about featurisation -- that is what `fold_check` is for. Covers the protenix trunk AND its MSA module. |
| `trunk_parity.py` | L1: the pairformer stack against the vendor's own module. 7 models; `--blocks N` truncates BOTH stacks. |
| `conditioning_parity.py` | L2: diffusion pair + single conditioning. The relative-position features come from the VENDOR's code on OUR batch, so the relative encoding is gated too. `OF3_ALPHABET=1` reproduces the rf3 alphabet bug this found. |
| `atom_parity.py` | L2: the atom cross-attention encoder, on a real batch because the windows cannot be synthesised. |
| `diffusion_parity.py` + `l2_all.sh` | L2: the token diffusion transformer. 10 models. |
| `denoise_parity.py` | L3: one whole denoise step -- conditioning, atom encoder, token transformer, atom decoder and the EDM scaling at once. |
| `confidence_parity.py` + `l4_all.sh` + `l4_table.py` | L4: the confidence head. Every port. `BLOCKS=`, `ZERO_SINPUTS=`, `POS_SCALE=` and `PROTEIN_ONLY_SINPUTS=` are the localisation knobs. |
| `modality_check.py` | L6: RNA / DNA / ligand / complex folds scored against a reference on the RIGHT atom, and `--write DIR` validates the mmCIF the model emits. |
| `grad_check.py`, `af2_grad_check.py` | Is the model differentiable at its sequence input, on either engine? |
| `af2_fold_check.py` | AF2 against ColabDesign on the same weights (`--compare`). |

Three rules the gates above enforce on themselves, each of which produced a
wrong number first:

* **Truncate BOTH sides.** Cutting only ours made a 1-block run read corr -0.19
  against native's four, twice, in two different harnesses. The tell is that the
  number gets WORSE with fewer blocks.
* **Match the storage dtype.** `intellifold2`'s blob stores trunk weights in
  bfloat16 on purpose; comparing it to an fp32 native reads plddt corr 0.614 and
  looks like a port bug.
* **Use the VENDOR's alphabet.** rf3 transposes G/C against of3, so a harness
  that builds our 447-wide target_feat with of3's permutation disagrees exactly
  in the nucleic classes. `PROTEIN_ONLY_SINPUTS=1` is the one-run discriminator.

## ESMFold2

The port is [documented in memory](../../../README.md); these are its gates.

| file | what it answers |
|---|---|
| `esmfold2_reference.py` | **The spec.** A complete self-contained JAX ESMFold2. Every other harness compares against this or against native. Read it before changing the port. |
| `esmfold2_localise_trunk.py` | Graph trunk vs reference trunk, with per-stage taps (`ESM_TRUNK_TAP=1`). |
| `esmfold2_localise_diffusion.py` | Trunk or diffusion? Folds with the GRAPH's trunk through the REFERENCE's sampler. |
| `esmfold2_localise_denoise.py` | One denoise step, graph vs reference, with atom-level taps (`ESM_DIFF_TAP=1`). |
| `esmfold2_localise_exp.py` | Our trunk vs NATIVE's for one variant, with native's `lm_z` injected so ESM-C is out of the comparison. |
| `esmfold2_fold_from_native_z.py` | Folds with NATIVE's trunk through OUR diffusion -- the call that says which half is wrong. |
| `esmfold2_audit_biases.py` | Which trained LayerNorm biases never reach the graph? Asks the CHECKPOINT, the direction a scope diff cannot look. |
| `esmfold2_native_variants.py` | Native folds, one release at a time. Picks the model class from the checkpoint's own `architectures`. |
| `esmfold2_oracle_exp_trunk.py` | Native PER-PASS trunk activations for an experimental release -- `lm_z`, `trunk_in{i}`, `trunk_pass{i}`. The generator `esmfold2_localise_exp.py` reads; without it that harness has nothing to compare against. |
| `esmfold2_msa_gate.py` / `esmfold2_msa_vs_lm.py` | Does an MSA substitute for ESM-C? Ours and native's answer to the same 2x2. |
| `esmfold2_oracle*.py`, `esmfold2_targets.py` | Dump native activations and survey native's accuracy. These GENERATE the npz files the gates consume. |
| `esmfold2_bench_*.py`, `esmfold2_ablate_lm.py`, `esmfold2_native_*.py` | Cost and ablation runs. Their conclusions are recorded in commits; the scripts regenerate them. |

## ESM-C

| file | what it answers |
|---|---|
| `esmfold2_gate_esmc_int8.py` | The 6B tower against native's hidden states, on GPU. corr 0.99973. |
| `esmc_gate_small.py` | The same for the 300M/600M towers. 0.99991 and 0.99988. |

## Deleted, and why

Nine `esmfold2_gate_*.py` (atom, block, conditioning, confidence, diffusion,
end_to_end, full, msa, trunk) gated a hand-written jnp replica block by block
against native during the port. That replica became `esmfold2_reference.py`,
which is gated end to end and is what every harness above compares against, so
the component gates were scaffolding for a building that now stands. Plus
`esmfold2_full_jax.py` and `esmfold2_gate_ubiquitin.py` (obsolete AND forcing
CPU) and `esmfold2_run_reference.py` (a thin driver the localise harnesses
replaced). All recoverable from git.
