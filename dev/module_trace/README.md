# module_trace

Run one input through one model and record **every haiku module's output**, so two
runs can be diffed module by module. Built to answer one question fast: *which
module is the first to differ?*

```
python -m tools.module_trace list
python -m tools.module_trace run --model openfold3 --case 6mrr_single_seq --label baseline
python -m tools.module_trace compare ~/module_traces/openfold3/6mrr_single_seq/{baseline,after_change}
python -m tools.module_trace show ~/module_traces/openfold3/6mrr_single_seq/baseline
```

## How it captures

`hk.intercept_methods` wraps every module call, so the trace *is* the module tree
-- nothing is instrumented by hand and nothing goes stale when the graph changes.
The trunk recycles run in a `fori_loop` and the pairformer/diffusion stacks in
scans, so at trace time every activation is a tracer; each output leaf is
therefore handed to a `jax.debug.callback`, which fires with concrete values at
run time, inside loops and scans included. A module that runs 48 times (one per
pairformer block) records 48 entries under the same key, in execution order.

Each array is fingerprinted inside the callback and dropped: shape, dtype,
mean/std/min/max/absmax, fraction of zeros, count of non-finite values, and a
fixed-seed random projection of the flattened array. The projection is what makes
"identical" mean identical -- a transpose, a permutation or one changed element
moves it, where mean/std easily agree. Tracing a 48-block trunk costs kilobytes.

`--max-calls-per-site N` (default 8) bounds the cost of repetitive loops: all
calls are **counted**, the first N are fingerprinted. `--save-arrays` additionally
writes `modules.npz` with the arrays themselves (<= 8 MB each) for hands-on
debugging.

## What a trace holds

`manifest.json`:

| field | meaning |
| --- | --- |
| `modules` | call site -> `{calls, fp[]}`, in execution order (`call_order`) |
| `outputs` | the model's own returned leaves, fingerprinted the same way |
| `inputs` | per batch key: shape, dtype, fraction non-zero, distinct values |
| `channels_live` / `channels_unmet` | which optional inputs (msa, template, ligand, multimer) are actually populated, vs what the case declared it exercises -- this is how "the extra input silently did nothing" gets caught |
| `metrics` | CA-RMSD to the case's native structure, CA-CA spacing, pLDDT/PAE |
| `params_left_at_init` | weights the converter did not supply (unported heads) |

## Comparing

`compare` walks A's execution order and reports `MATCH` / `CLOSE` / `DIFFER` /
`ONLY_A` / `ONLY_B` per call site, plus **`FIRST DIVERGENCE`** -- the earliest
differing module, which is the bug site; everything after it is downstream
contamination. A changed call count is itself a difference (the graph changed).

## Cases and models

Cases live in `cases.py`. Active now: `6mrr_single_seq` (68-residue protein,
single sequence, no template/MSA) and `6mrr_self_template`. Registered for later,
and deliberately taken from AlphaFold3's own canonical inputs rather than
invented: `af3_featurised_example` (DeepMind's featurised test batch -- real
16384-row MSA, 4 templates, polymer-ligand bonds) and the 13 `alphafold3/examples`
JSONs covering homodimer, protein-protein, CCD ligand + covalent bond, SMILES
ligand, ions, DNA duplex, methylated DNA, RNA hairpin, modified RNA,
glycosylation and phosphorylation.

Models come from `models.py` (one entry per `global_config.model` name), loaded by
the init+merge path so a partially ported model still traces, with the unported
scopes listed in `params_left_at_init`.
