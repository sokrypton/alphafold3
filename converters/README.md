# Weight converters

Offline tooling that turns a published AF3-family PyTorch checkpoint into the
haiku parameter blob (`<model>.bin.zst`) that `run_alphafold.py` loads.

**This package is deliberately outside `src/alphafold3`.** The served model path
never imports torch, never downloads a 2 GB checkpoint, and never runs a
converter: it loads already-converted weights (from a local `--model_dir`, or
pulled from Hugging Face). Conversion happens once, here, and the product is
published.

```bash
# fetches the original checkpoint into --out, converts, writes <model>.bin.zst
PYTHONPATH=src python -m converters.convert --model boltz2 --out ~/ported/boltz2

# or convert a checkpoint you already have
PYTHONPATH=src python -m converters.convert --model openfold3 \
    --checkpoint ./of3-p2-155k.pt --out ~/ported/openfold3
```

| module | model | source |
|---|---|---|
| `openfold3.py` | OpenFold3 (AlQuraishi Lab) | `s3://openfold/staging/of3-p2-155k.pt` |
| `intellifold2.py` | IntelliFold-v2 (IntelligenAI) | HF `intelligenAI/intellifold` |
| `opendde.py` | OpenDDE (Aureka Research) | HF `aurekaresearch/OpenDDE` |
| `boltz2.py` | Boltz-2 (jwohlwend/boltz) | HF `boltz-community/boltz-2` |
| `protenix2.py` | Protenix-v2 (ByteDance) | HF mirror `TMF001/protenix-v2-weights` |
| `rosettafold3.py` | RoseTTAFold3 (RosettaCommons) | `files.ipd.uw.edu/pub/rf3` |
| `chai1.py` | chai-1 (Chai Discovery) | `chaiassets.com` (5 TorchScript archives) |

Each conversion also writes `<model>.shapes.json`, the parameter tree derived
from the graph itself (`shapes.py`). It is what a run loads instead of asking
`jax.eval_shape`, and it is the record of what the conversion covers: any
parameter the graph wants and the converter did not produce is listed there,
reported at load, and filled with zeros rather than random values. All eight
current conversions report 0 missing.

`esm_lm.py` is the odd one out: it carries two whole models rather than a map.
Two of the ports fold from a protein language model — chai-1 from ESM2 3B,
ESMFold2 from ESM-C — and neither runs at full fidelity without one, so the
towers are converted and run here, in jax, like everything else. They are one
file because they are one architecture: pre-LayerNorm, rotary, head_dim 64,
differing in five documented places (fused vs split qkv, q/k norm, SwiGLU vs
gelu, residual scaling, and how many hidden states the consumer reads). The
ESMFold2 shim that turns ESM-C's states into a pair representation is not part
of the tower and lives in `esmfold2.py`.

Both towers convert at int8 and run under a `lax.scan`. For ESM-C that is not an
optimisation: float32 cannot be written at all (a record header packs its length
as a signed 32-bit int, and the fused fc1 stack is one 5.27 GiB tensor), and a
python loop over the 80 blocks leaves the whole 25 GB tower resident on a 23 GB
card.

`common.py` holds the primitives every family shares -- the transpose/reshape
math is the same everywhere, only the torch leaf names and the fusion mode
differ, so each family is a `Dialect` plus a map. `sources.py` records where the
originals come from. AlphaFold3's own weights are not converted: DeepMind
publishes a haiku blob already, and its terms require you to request it.

## Coverage runs BOTH ways

`<model>.shapes.json` audits the **graph** side: a parameter the graph wants and
the converter did not produce is reported at load and filled with zeros rather
than random values. All eight conversions report 0 missing.

That check is blind to the opposite leak -- **a trained tensor in the checkpoint
that the graph has no slot for**. Stock AF3 creates its norms with
`create_offset=False` and its Linears with `use_bias=False`, so wherever a ported
family added an offset or a bias, the converter simply never asks for the key and
nothing anywhere fires. `audit_coverage.py` is that second direction:

```bash
PYTHONPATH=src:. python dev/audit_coverage.py protenix2
```

It wraps the state dict to record every key addressed by name, then resolves the
rest by VALUE against the converted tree -- two converters scan with `sd.items()`
instead of indexing, and a scan cannot be attributed key by key. The value index
knows the three fusions this codebase uses (SwiGLU `concat_ab` halves, triangle
`interleave_ab` even/odd columns, and `layer_stack`/`stack_super` slices to two
levels), because a source tensor hidden inside a fused leaf is not missing.

**Every hit still needs triage** -- the tool proves a tensor does not appear
verbatim, not that it matters. Three outcomes recur:

* *duplicate* -- openfold3's checkpoint carries the whole diffusion module twice,
  under `diffusion_module.*` and `sample_diffusion.diffusion_module.*`,
  bit-identical. The converter reads one. Nothing is lost.
* *inert* -- an offset on a LayerNorm whose output feeds only an attention-logit
  projection adds the same constant to every key in a head, and softmax is
  shift-invariant. rosettafold3's 30 `ln_0.bias` and boltz2's 30 `*_proj_z.N.0.bias`
  are all of this kind. Leave them, and say so, or someone re-derives it.
* *sliced* -- a fused source projection the converter cuts into pieces by COLUMN.
  opendde's `linear_no_bias_f` (128, 385) becomes ref_mask(1) + ref_element(128) +
  atom-name-chars(256), and an arbitrary column range is not something the value
  index can enumerate, so both instances report as missing. This is the tool's
  known false positive: check the converter before believing a fused-projection
  hit.
* *real* -- see the table below.

### What the audit found (2026-09-04) -- all `real` rows now FIXED

| model | tensor | verdict |
|---|---|---|
| protenix2 | `distogram_head.linear.bias` (64) | **real** |
| rosettafold3 | `distogram_head.predictor.bias` (65) | **real** |
| boltz2 | `distogram_module.distogram.bias` (64) | **real** |
| opendde | `distogram_head.linear.bias` (96) | **real** |
| rosettafold3 | `atom_attention_encoder.process_s_trunk.0.bias` (384) | **real** |
| rosettafold3 | `atom_attention_encoder.process_z.0.bias` (128) | **real** |
| boltz2 | `single_conditioner.single_embed.bias` (768) | **real** |
| rosettafold3 | `to_r_update.0.bias` (128) | cosmetic: a constant added to every atom's position update is a rigid translation |
| rosettafold3 | 30 x `ln_0.bias` | inert (softmax shift) |
| boltz2 | 30 x `*_proj_z.N.0.bias` | inert (softmax shift) |
| protenix2 | `confidence_head.{lower,upper}_bins` | not trained: bin edges, and they match AF3's 39 defaults |
| boltz2 | `bfactor_module.*` | out of scope: no B-factor head |
| openfold3 | 8 under `sample_diffusion.*` + `version_tensor` | duplicate |
| opendde | 2 x `linear_no_bias_f` (128, 385) | sliced (see below) |
| intellifold2 | -- | **0 unmapped** |

The distogram bias is one shared fix: `half_logits` is built by
`distogram_head.py`'s stock path with no bias, so four families lose a trained
one. It is not small -- protenix2's spans -2.05..+1.82, and native adds
`b + b^T`, so `softmax(2b)` alone spreads 0.50 to 2e-4 across bins. Blast radius
is the distogram outputs and `contact_probs` only; diffusion never reads these
logits, which is why no fold gate could catch it.

Re-running the audit after the fixes: protenix2 3 -> 2, rosettafold3 37 -> 33,
boltz2 37 -> 35, opendde 5 -> 4.

**Fix all THREE copies, and measure in the one your gates import.** The package
converters here are what builds a published blob, but every ColabDesign2 oracle
converts in-process through `colabdesign2/af3/converters/` (its `CHECKPOINTS`
point at the raw torch files, so it never reads `~/ported`). Patching only this
directory left every gate running the old conversion and reporting a pass; the
isolation that caught it read `distogram bias absmax 0.000`. The trees have
really diverged, so patch each -- do not copy one over the other. What remains in each is exactly the inert,
duplicate, sliced and not-trained rows above.

**Two of the fixes needed the native FORWARD read, not just the tensor name.** Our
graph applies the distogram linear and then symmetrises, passing the bias through
twice. protenix2 and opendde symmetrise after the linear too, so their bias maps
straight across; rosettafold3 and boltz2 symmetrise the pair features FIRST, so
theirs enters once and the converter halves it. Name-matching alone would have got
two of the four wrong by a factor of two.

Run both directions on every new converter. The idea is borrowed: an independent
Protenix port by ChoongHwanLee (`chlee19990109-cloud/ColabFold`, branch
`colabfold2-protenix-proof`)
asserts all 4174 of its source tensors are consumed exactly once, and that
discipline is what surfaced this whole table.

**A stronger check we do not have yet:** they also round-trip the state dict
through the *native* model (`load_state_dict` -> 0 missing / 0 unexpected), which
distinguishes a tensor we drop from one the native code ignores too. Without it,
a dead tensor left in a checkpoint by an older training run reads as a gap.

`openfold3_test.py` covers the conventions that are silent when wrong -- the
residue-alphabet permutations, the two features_1d classes AF3 lacks, and the
i/j crossing between AF3's two pair-embedding sites (see
`../OF3_AF3_PORTING_NOTES.md`).

```bash
PYTHONPATH=src:. python -m pytest dev/openfold3_test.py -q
```

## Publishing

`run_alphafold.py` fetches a ported model's weights on first use from the
Hugging Face repo named in `alphafold3.model.model_registry` (`_WEIGHTS_REPO`),
by exact filename, over plain HTTPS. So the repo is flat:

```
<repo>/
  openfold3.bin.zst        openfold3.shapes.json
  intellifold2.bin.zst     intellifold2.shapes.json
  opendde.bin.zst          opendde.shapes.json
  boltz2.bin.zst           boltz2.shapes.json
  protenix2.bin.zst        protenix2.shapes.json
  rosettafold3.bin.zst     rosettafold3.shapes.json
  chai1.bin.zst            chai1.shapes.json   std_conformers.npz
```

Both files per model: without the manifest a run works but cannot tell you what
a conversion missed. chai-1 additionally needs `std_conformers.npz`, which is
part of its weights in the same sense the blob is — the model was trained on
those conformers.

AlphaFold 3's own parameters are not published here and never will be: DeepMind
requires you to request them.
