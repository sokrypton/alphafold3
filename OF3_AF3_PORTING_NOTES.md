# OpenFold3 → AlphaFold3 Weight Porting: Code Changes and Implications

This document records every code change made to the AlphaFold3 (JAX/Haiku) codebase
to run OpenFold3 (PyTorch) weights, and what each change implies for anyone trying to
reproduce or re-implement AlphaFold3 from the released source.

**Repositories**
- AF3 source: [sokrypton/alphafold3](https://github.com/sokrypton/alphafold3) (fork of google-deepmind/alphafold3)
- OF3 source: [aqlaboratory/openfold3](https://github.com/aqlaboratory/openfold)
- OF3 weights: `s3://openfold/staging/of3-p2-155k.pt` (public, no sign required)

---

## Background

Google DeepMind released the AlphaFold3 source code but not the model weights
(commercial use restricted). AlQuraishi Lab released OpenFold3 with public weights
for the same architecture. Since both implement the same algorithm, the weights are
portable — but not without changes.

All OF3-specific branches in AF3 are gated on `global_config.of3_weights = True`,
set automatically via `--of3_weights` at the command line. Standard AF3 behavior
is unchanged when the flag is absent.

---

## Model Architecture Changes

These changes live in `src/alphafold3/model/`. They cannot be handled in the weight
converter alone because they reflect genuine differences in how OF3 and AF3 implement
or apply the same algorithm.

| File | Change | Why |
|---|---|---|
| `model_config.py` | Added `GlobalConfig.of3_weights: bool = False` | Master switch for all OF3 branches |
| `model.py` | Element one-hot index shifted by −1 | OF3 featurizes elements as `GetAtomicNumber() - 1` (0-indexed); AF3 uses `GetAtomicNum()` (1-indexed) |
| `network/atom_cross_attention.py` | `queries_single_cond *= queries_mask` | Masks padded query atoms from single conditioning, consistent with OF3's `atom_pair_mask` behavior |
| `network/atom_cross_attention.py` | `keys_ref_space_uid` sourced from `queries_ref_space_uid` | Matches OF3's uid lookup convention |
| `network/atom_cross_attention.py` | `offsets_valid &= keys_mask` | OF3 multiplies uid-match by `atom_pair_mask`; AF3 did not mask padded key atoms — `ref_space_uid=0` of padded atoms collided with token 0, distorting N-terminal backbone (CA–C was ~0.85 Å instead of 1.52 Å) |
| `network/evoformer.py` | Bond matrix symmetrized in `_embed_bonds` | OF3's `create_token_bonds` sets both `[i,j]` and `[j,i]` per bond; AF3 featurizes only one direction — ring bonds broken without this (ATP ribose C–C ~2.0 Å instead of ~1.5 Å) |
| `network/modules.py` | Pair bias axes swapped for column attention (`transpose=True`) | OF3 computes column attention by transposing the sequence input before `linear_z`, producing `Linear(z[k,q])` as the bias; AF3 computes `Linear(z[q,k])`. The weights were trained with OF3's convention so the bias must be swapped |
| `network/diffusion_transformer.py` | Per-block pair LayerNorm + linear projection branch | OF3's `AttentionPairBias` contains its own `layer_norm_z` + `linear_z` per block; AF3 originally used a single shared LN before the entire block stack |
| `network/diffusion_head.py` | Fourier `w`/`b` loaded as Haiku params instead of AF3's hardcoded constants | OF3 stores these as `register_buffer` (saved to `state_dict`); we convert them directly from the checkpoint so AF3's hardcoded JAX constants are never used |
| `network/noise_level_embeddings.py` | Optional `weight`/`bias` args added to `noise_embeddings()` | Allows passing the converted Fourier values from `diffusion_head` instead of falling back to AF3's hardcoded values |

---

## Weight Converter

`convert_of3_weights.py` + `src/alphafold3/model/of3_weight_converter.py`

Handles the systematic PyTorch → JAX/Haiku translation:

- **Parameter renames**: `weight` → `weights`, `LayerNorm.weight` → `scale`, `LayerNorm.bias` → `offset`
- **Linear transpositions**: PyTorch stores `(out, in)`; AF3 stores `(in, out)` for most projections
- **Attention head reshapes**: Q/K stored as `(H, D, in)` (transpose_weights=True); V as `(in, H, D)`
- **SwiGLU concatenation**: OF3 has separate `linear_a` (gate) and `linear_b`; AF3 concatenates them as `[gate | linear]` for a single fused projection
- **Layer stack aggregation**: OF3 has per-block parameter dicts (`blocks.0.*, blocks.1.*`); AF3's `hk.experimental.layer_stack` expects a leading stacked axis

**Critical bug fixed during conversion**: TriangularMultiplication incoming `a`/`b` projection weights were swapped.

---

## Data Pipeline

### MSA Server (`src/alphafold3/data/msa_server.py`)

New file implementing a ColabFold/MMseqs2 server client, enabled via `--use_msa_server`.

**Two A3M parsing bugs fixed:**

1. **Null-byte block separator**: ColabFold separates per-query blocks with `\x00`. The original code stripped these before checking, so all hits from different query chains accumulated in the first query's block. Fix: split on `\x00` first, then parse each block independently.

2. **Bare query sequence leak**: When merging `bfd.mgnify30.metaeuk30.smag30.a3m` into `uniref.a3m` blocks, skipping only the `>M` header but not the query sequence line caused the query sequence to appear as a bare continuation of the previous hit. AF3's parser concatenates consecutive sequences under the same header, doubling the alignment width. Fix: skip both `lines[0]` (header) and `lines[1]` (query sequence) when extending from secondary files.

3. **Missing trailing newline**: `raw_block.strip()` removes the trailing `\n` from the last hit in each block. When extended with hits from another file, the first new header was concatenated directly onto the last sequence with no separator — inserting `>` and digits into what the parser treats as a sequence. Fix: `(raw_block + '\n').splitlines(keepends=True)` ensures every line terminates cleanly.

---

## Build System

### `build_data` (`src/alphafold3/build_data.py`)

Removed the `chemical_component_sets_gen` step. This file (three `frozenset`s of glycan and ion CCD codes) ships pre-built in the wheel and does not need to be regenerated.

**Root cause of the original failure**: `build_data` runs `ccd_pickle_gen` (parses the full CCD, ~1–2 GB in memory) then immediately `chemical_component_sets_gen` (loads `ccd.pickle` again). When running alongside weight download + conversion (`convert_of3_weights.py` peaks at ~4 GB), the combined memory spike OOM-killed `build_data` between the two steps — leaving `ccd.pickle` on disk but `chemical_component_sets.pickle` missing.

### `ccd_pickle_gen` (`src/alphafold3/constants/converters/ccd_pickle_gen.py`)

Fixed fragile assertion:
```python
# Before (broken):
assert len(result) == whole_file.count(b'data_')

# After:
n_headers = sum(1 for line in whole_file.split(b'\n') if line.startswith(b'data_'))
if len(result) != n_headers:
    raise ValueError(...)
```

`whole_file.count(b'data_')` counts all byte occurrences of `data_` — including those inside CIF quoted string values (compound names, synonyms). CCD versions that include any compound with `"data_"` in a field would trigger a spurious `AssertionError` before `pickle.dump`, preventing `ccd.pickle` from being written.

---

## Implications for AlphaFold3 Reproduction

### 1. Atom cross-attention padding (N-terminal backbone distortion)

The `offsets_valid` bug in `atom_cross_attention.py` affects anyone using AF3 with padded atom sequences. AF3 pads atom arrays to a fixed size; padded slots are zero-filled, giving them `ref_space_uid=0`. Since the first real residue also has `ref_space_uid=0`, every padded key atom was treated as a valid neighbor of residue 0. This produced severe backbone geometry errors at the N-terminus in all our test cases (CA–C ~0.85 Å, C–O ~0.95 Å vs. ideals of 1.52 Å and 1.23 Å). **This is a bug in the released AF3 source**, not specific to OF3 weights.

### 2. Bond matrix asymmetry

AF3's featurizer provides bonds in one direction only (lower-index → higher-index from the CCD bond table). OF3 trained with a symmetric bond matrix. Whether this matters for AF3's own weights is unknown — but ring-topology ligands (saccharides, ATP ribose) are likely affected. Anyone training AF3 from scratch should verify which convention is used in their featurizer and ensure consistency with the bond embedding weights.

### 3. Column attention pair bias convention (OF3 bug)

OF3's `TriangleAttention` (`starting=False`) transposes the pair representation **before** applying `linear_z`:

```python
# OF3 — column attention
x = x.transpose(-2, -3)           # z → z.T
triangle_bias[h, q, k] = Linear(z.T[q, k]) = Linear(z[k, q])
```

AF3's Algorithm 15 specifies the pair bias between query q and key k as `Linear(z[q, k])`. OF3 computes `Linear(z[k, q])` — the **transposed** pair bias. This is a deviation from the paper. The model trained with this convention and learned to compensate, so predictions are good in practice. Our AF3 fix (`jnp.swapaxes(nonbatched_bias, -1, -2)` when `of3_weights=True`) reproduces the OF3 convention.

Anyone reimplementing triangle attention ending node should use `Linear(z[q, k])` (the paper-correct version) if training from scratch, not OF3's transposed version.

### 4. Diffusion transformer pair conditioning (per-block vs. shared)

OF3 applies a separate `LayerNorm + Linear` to the pair representation inside each diffusion transformer block. AF3's original code applied a single shared LayerNorm before all blocks. This is a genuine architectural difference. If training AF3 from scratch, either approach could work, but they produce incompatible parameter layouts.

### 5. Fourier noise embeddings

Both AF3 and OF3 initialize the Fourier embedding weights with seed 42, but PyTorch's `torch.Generator().manual_seed(42)` + `normal_` + `uniform_` produces different values than JAX's equivalent. AF3 hardcodes its values as `_WEIGHT` and `_BIAS` constants; OF3 saves them as `register_buffer` in the checkpoint. Anyone converting OF3 weights must use the checkpoint values, not AF3's constants.

---

## Summary: What the Released AF3 Code Gets Wrong

| Issue | Severity | Affects AF3 weights too? |
|---|---|---|
| `offsets_valid` missing `keys_mask` gate (N-terminal distortion) | High — ~0.85 Å CA–C at residue 0 | Likely yes |
| Bond matrix one-directional (ring ligand geometry) | Medium — ring bonds broken | Depends on training data convention |
| Column attention pair bias transposed (OF3 bug) | Low — model compensates | No — only relevant when using OF3 weights |
| `ccd_pickle_gen` assertion overcounts `data_` | Build-time only | Yes |
