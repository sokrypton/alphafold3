# Runtime benchmarks

Plots and the script that makes them (`make_plots.py`). Regenerate with:

    python docs/benchmarks/make_plots.py <dir holding the sweep TSVs>

## What is measured

Steady-state seconds per prediction: 3 recycles, 200 diffusion steps, 1 sample,
`num_msa=1024`, single sequence, no template, confidence head on. "Steady state"
is the median of the LAST THIRD of the calls in one process -- see the warm-up
trap below for why not "everything after the first call".

Sequence length is swept with a random sequence; only the shape matters for
timing, and the native harnesses are fed the SAME sequences (same seed, same
alphabet) so both sides fold the same input.

## Provenance of these numbers (2026-09-04)

The A10 sweep (`sweep_a10.tsv`) was re-measured after the parameter fixes of
2026-09-04 and now includes `openbind`. Every model landed within +/-0.025 s of
its previous 64-token figure, so those fixes cost no measurable speed.

**The A100 curves are OLDER.** `sweep_a100.tsv` predates that re-measurement and
has no `openbind` row, so `runtime_all_A100.png` shows eight ports where the A10
plot shows nine. It is a different day's data on a different machine; do not
read an A10-vs-A100 difference off these two plots as if it were controlled.

`opendde` has no 512 point on the A10: it OOMs, asking for 19.53 GiB on a 23 GB
card. That is a result, not a gap -- it is the widest trunk here (pair_channel
384) and runs its diffusion on an expanded structural-token set. The previous
sweep recorded the same failure at the same length.

Three models have no native torch ratio. `alphafold3` has no torch native at all
-- it IS the JAX one, so that comparison is port fidelity, not speed.
`intellifold2`'s two harnesses (torch and jax) are written but have never been
run. `openbind` needs openfold3>=0.5, which the `~/of3_extra` install predates.

## Results

`runtime_all_A10.png` -- all nine ports, linear and log-log.
`runtime_per_model.png` -- one panel per port, per GPU, native where available.
`boltz2_vs_native.png` -- ours vs native boltz-2, and the ratio by size.

Two things the numbers say that are easy to get wrong:

* **Scaling is NOT linear.** Below ~256 tokens the local exponent is ~1.0; from
  256 to 512 it is ~1.8 (alphafold3: 9.16 -> 18.27 -> 31.00 s at 256/384/512).
  A curve fitted below 384 will badly underestimate long-sequence cost. An
  earlier version of this file claimed "five ports scale linearly" -- that was
  an artefact of stopping at 384.
* **The advantage over native torch shrinks with size, then flattens.** boltz-2:
  2.05x at 64 tokens, 1.53x at 128, ~1.45x at 192-256, 1.59x at 384. It is
  mostly PyTorch eager dispatch overhead, which amortises as arithmetic grows.
  Do not quote the small-size number on its own.

### Native coverage (as of this commit)

| port | native timing | note |
|---|---|---|
| boltz2 | 64-384, forward vs forward | the clean comparison |
| rosettafold3 | 64-384, featurisation split out | see below |
| chai1, protenix2, opendde, intellifold2 | none | runners exist, no timing harness |
| openfold3 | none | no native install on this box |
| alphafold3 | 68 tokens (older note) | native IS jax/haiku -- same graph, so this measures port fidelity, not a framework advantage |

`rosettafold3` needed care: its `run()` bundles atomworks featurisation, cif
parsing and writing outputs, while our number is forward-only, so the raw ratio
is an UPPER BOUND. Timing the pipeline separately (same wrapper point
dump_native_batch.py uses) splits it:

    tokens  ours   native total  featurise  native fwd  ratio_total  ratio_fwd
      64    3.12      14.38        2.66       11.72        4.61x      3.76x
     192    7.18      22.19        3.81       18.38        3.09x      2.56x
     384   20.28      61.53        5.34       56.19        3.03x      2.77x

So rosettafold3 is ~2.6-2.8x rather than the ~3.0-4.6x the unsplit number
suggests. Incidentally native's featuriser costs 2.7-5.3 s against our 0.9 s.

## Long lengths

The 512 row (A10) puts every port's local exponent in 1.76-2.09, including the
three that read ~1.0 below 256 -- so the exponent is a property of the ARCHITECTURE
at that size, not of the port. What differs between ports is the constant factor.
`opendde` cannot run 512 on a 23 GB card at all (`RESOURCE_EXHAUSTED: Out of
memory while trying to allocate 19.53GiB`), which fits its expanded
structural-token diffusion needing more memory per token than the rest.

The spread also widens sharply with length: at 64 tokens the slowest port is
1.4x the fastest; at 512 intellifold2 is 3.0x alphafold3 and opendde is out of
memory. Port choice matters far more on long inputs than short ones.

## Traps, all of which produced wrong numbers here first

* **Concurrent GPU processes.** Four orphaned benchmark processes (~6 GB each)
  made native boltz-2 read 25 s at 128 tokens where the truth is 8.67 s -- and
  the contaminated numbers looked *better* for us, which is how they survived a
  sanity check. Background loops here outlive the turn that starts them and
  stack up. Verify the GPU is EMPTY before every measurement:

      nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l

* **Warm-up is not always one call.** On an A100 the model retraced on call 2
  (tokamax's lazily-created JAX user context -- see the fix in
  `ModelRunner._preinit_tokamax_context`), so call 2 cost ~45 s. Taking "the
  median after the first call" reported 27.8 s where steady state was 1.9 s.
  Always print every call, not just the summary.

* **Both caches or neither.** `run_alphafold.py` defaults
  `--cache_dir=/tmp/alphafold_cache` and wires the JAX compilation cache and the
  tokamax autotune cache. A harness that builds the model itself gets neither
  and pays a full cold compile every process (56.3 s vs 8.1 s warm, from a
  3.9 MB cache). A one-pass SWEEP cannot benefit either way -- each length is a
  new executable -- but anything repeating a shape can.

* **`Autotuning cache miss` costs nothing.** tokamax logs it and falls back to a
  heuristics config (`ops/op.py:514-523`). Do not "fix" it by calling
  `tokamax.autotune()` per shape; that adds real search time.

* **MSA depth is not a runtime axis.** Featurisation pads `msa` to 16384 rows
  whatever the input carries and the model truncates to `config.evoformer.num_msa`,
  so input depth changes a scalar (`num_alignments`) and no tensor shape. The
  config knob does cost time -- 10% of the trunk at 64 tokens, 3.9% at 256 --
  but under 1% of a full fold, because ~93% of that is diffusion sampling.

* **Feed native the same input you feed us.** Handing native rf3 `6MRR.pdb`
  makes atomworks split the waters into a second chain, so it folds more than
  our 68 tokens. Strip to protein-only first.

* **Not every comparison is like-for-like.** Native boltz-2 is timed on a
  pre-built batch, so it is forward-vs-forward with ours. Native rosettafold3's
  `run()` includes atomworks featurisation, cif parsing and writing outputs, so
  its number is an UPPER BOUND on the ratio, not a measurement of it. The A100
  series also runs tokamax Triton kernels an A10 cannot, so A10-vs-A100 differs
  in kernel path as well as hardware.

## Memory ceilings

An A10 (23 GB) reaches 512 tokens at 17.2 GB and cannot do 768: pair memory
grows ~quadratically. 768 and beyond need the A100's 40 GB.
