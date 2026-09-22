# Native oracle setup and runtime benchmarks

Every native implementation we compare against, how it is installed, and how to
time it. Written because working this out consumed most of a session and none of
it is discoverable from the code.

## Which natives exist, and what a comparison against them MEANS

| port | native framework | comparison is | entry point |
|---|---|---|---|
| boltz2 | torch | framework speed | `boltz.model.models.boltz2.Boltz2` (module API) |
| rosettafold3 | torch | framework speed | `rf3.inference_engines.rf3.RF3InferenceEngine.run` |
| chai1 | torch | framework speed | `chai_lab.chai1.run_inference` |
| protenix2 | torch | framework speed | `runner.inference.InferenceRunner.predict` |
| opendde | torch | framework speed | `runner.inference.InferenceRunner.predict` |
| openfold3 | torch | framework speed | `openfold-3/openfold3/run_openfold.py` |
| intellifold2 | **BOTH** torch and jax | see below | torch: `run_intellifold.py`; jax: `intellifold/run_jax_inference.py` |
| alphafold3 | jax/haiku | PORT FIDELITY, not speed | the same graph we run |

Two of these are not speed comparisons and must not be quoted as such:

* **alphafold3.** Native AF3 is jax/haiku running the same computation as our
  port. Parity is the expected -- and reassuring -- answer; a large gap either
  way would mean we broke something.
* **intellifold2's jax path.** IntelliGen ported their own model to jax on top of
  a vendored, patched AlphaFold 3 (`intellifold/cli.py` calls itself "a thin
  wrapper around the vendored, patched AlphaFold 3 run_jax_inference.py"). So
  intellifold2 gives THREE numbers worth having: our jax port, their jax port,
  and their torch original. Their torch path is the framework comparison; their
  jax path is the more interesting one -- two independent jax ports of the same
  weights.

## Install recipe per native

Standing constraint: **never install into `~/venv`** (the jax test env), and
`~/boltz_venv` must stay CPU-only (it is the CPU oracle). Native deps therefore
go into a side directory loaded with PYTHONPATH -- the pattern `~/rf3_extra`
already established.

    boltz2        ~/boltz_gpu_venv (CUDA torch); weights ~/boltz2_weights/boltz2_conf.ckpt
                  source tree ~/BoltzDesign1/boltz2/src
    rosettafold3  ~/venv + PYTHONPATH=~/rf3_extra:~/foundry_rf3/src:~/foundry_rf3/models/rf3/src
                  weights ~/rf3_weights/rf3_foundry_01_24_latest_remapped.ckpt
    chai1         ~/chai_venv, CHAI_DOWNLOADS_DIR=~/chai1_weights
    protenix2     ~/venv, repo ~/protenix, checkpoint ~/checkpoint/protenix-v2.pt
                  (symlink to ~/protenix_weights); needs the esm + fast-layernorm
                  stubs in tools/oracles/protenix2/run_native.py
    opendde       ~/venv, repo ~/OpenDDE, checkpoint ~/opendde_weights/opendde.pt
                  same two stubs
    openfold3     repo ~/openfold-3, checkpoint ~/of3-p2-155k.pt
    intellifold2  repo ~/IntelliFold, torch checkpoint ~/model_v2/intellifold_v2.pt
      torch path: ~/boltz_gpu_venv/bin/python with
                  PYTHONPATH=~/if2_extra:~/IntelliFold
                  --cache ~/model_v2   (the runner looks for
                  `<cache>/intellifold_v2.pt`, which is what that file is called)
      jax path:   `intellifold predict` (installed in ~/venv), model dir ./model_v2

### The `~/if2_extra` side-load, and one trap in it

    ~/boltz_gpu_venv/bin/pip install --target ~/if2_extra \
        accelerate ml_collections biopython modelcif

Then **delete numpy from the side-load**:

    rm -rf ~/if2_extra/numpy ~/if2_extra/numpy-*.dist-info ~/if2_extra/numpy.libs

pip pulls numpy 2.5.2 into the target dir, and on PYTHONPATH it SHADOWS the
venv's numpy. numba requires <2.3, so IntelliFold dies with a numba import error
that says nothing about numpy. Removing it lets the venv's compatible numpy win.
(`~/venv` itself has torch 2.13.0+**cpu**, so the torch path cannot run there at
all -- that is why this uses boltz_gpu_venv.)

## protenix2 / opendde: the CUDA venv and the side-load traps

`~/venv` has torch 2.13.0+**cpu**, so native protenix2 there never touches the
GPU -- it appeared to "hang" for minutes and was really doing `kaiming_uniform_`
over 464M parameters on CPU before loading the checkpoint (faulthandler stack:
`nn.Linear.__init__` <- `pairformer.__init__` <- `init_model`). Run it under
`~/boltz_gpu_venv` with `PYTHONPATH=~/if2_extra`, plus
`PYTHONPATH` must also reach `/home/ubuntu/protenix` for `configs.configs_base`
(python puts the SCRIPT's directory on sys.path, not the cwd, so `cd` is not
enough).

**`pip install --target` shadows the venv.** Anything in the target directory
wins on PYTHONPATH, so pip's dependency resolution silently replaces the venv's
packages. This bit four times in one session:

| shadowed | symptom |
|---|---|
| torch 2.14 over the venv's 2.13 | none -- would have benchmarked a DIFFERENT torch build than documented |
| numpy 2.5.2 over 1.26.4 | numba import error that never mentions numpy |
| scipy built for numpy 2 | `AttributeError: module 'numpy' has no attribute 'long'` inside scipy.sparse |
| biotite 1.7.1 instead of 1.4.0 | `BondList object has no attribute '_bonds'` |

Rules: install with `--no-deps`; delete anything the venv already provides
(torch, numpy, scipy, pandas); and PIN to the version the working environment
uses -- `~/venv` is the reference for what protenix2 needs even though it cannot
run there. Purging the accidental extras took `~/if2_extra` from 5.4 GB to
~350 MB. The torch one is the dangerous entry: it fails silently.

**predict() CONSUMES its batch.** Re-running it on the same dict raises
`KeyError: 'profile'`, which inference.py swallows as "L64 failed" -- leaving one
timing that looks like a steady state but is a COLD call. Deep-copy per call.

**Do not leave native deoptimised.** protenix's own default is
`cuequivariance` for both triangle ops (configs_base.py:129-130); the oracle
forces `torch` only because those kernels are absent, which measures native with
its optimisations OFF and flatters us. Install
`cuequivariance-torch cuequivariance cuequivariance-ops-torch-cu12` and pass
`PX_KERNELS=cuequivariance`. Measured: it changes nothing at 64 tokens on an A10
(26.74 s cuEq vs 26.1 s torch), which confirms rather than assumes the note that
cuEq tri-mul is ~1.09x and effectively datacentre-only.

Native protenix2 also runs `torch.autocast(cuda, bf16)` by default
(`configs.dtype = "bf16"`), so precision matches ours -- it is not a confound.

RESULT: 64 tokens, warm, cuEq kernels, forward-vs-forward: native 26.74 s
against our 3.31 s = **8.1x**, the one genuine outlier against the 1.4-2.8x the
other ports show.

## openfold3: its own schema, and an argv trap that kills DataLoader workers

Environment: `~/boltz_gpu_venv` + `PYTHONPATH=~/of3_extra:~/openfold-3`, weights
`~/of3-p2-155k.pt`. `of3_extra` is a SEPARATE side-load from `if2_extra` on
purpose: openfold3's closure is large and installing it with deps would have
re-upgraded the pins protenix2/opendde need (pydantic 2.13.4 <-> pydantic_core
2.46.4, biotite 1.4.0) and broken two working measurements to fix a third.
Install with deps to catch the transitive tail in one pass, then purge
torch/numpy/scipy/pandas/triton/nvidia so the venv's versions stay authoritative.

**It does not take AF3-format JSON.** openfold3 has its own `InferenceQuerySet`:

    {"seeds": [1],
     "queries": {"L64": {"chains": [{"molecule_type": "protein",
                                     "chain_ids": ["A"], "sequence": "..."}]}}}

See `examples/example_inference_inputs/`. Every other port here takes the AF3
fold-input list. Its `predict` subcommand takes --query-json /
--inference-ckpt-path / --output-dir / --num-diffusion-samples /
--num-model-seeds; `--seed` belongs to `train`, not `predict`.

**Do not read benchmark parameters from sys.argv.** These harnesses rewrite
`sys.argv` to drive the target's CLI, and openfold3's DataLoader spawns workers
that RE-IMPORT the harness as `__main__` -- under the rewritten argv. So
`int(sys.argv[1])` ran on the string `'predict'` and every worker died. The
symptom is only `RuntimeError: DataLoader worker (pid(s) ...) exited
unexpectedly`; the real cause is in worker stderr. Read parameters from the
ENVIRONMENT (OF3_L / OF3_REPS) so re-import is harmless. Any harness that
rewrites argv is exposed to this the moment its target uses multiple workers.

A stale `/tmp/of3-of-ubuntu/colabfold_msas/raw` from a failed run also aborts the
next one with FileExistsError -- clear it between attempts.

RESULT: 64 tokens, warm, forward-only: native 13.56 s vs our 2.99 s = 4.5x.

## A harness that measures nothing must SAY so

`runpy.run_path()` re-executes the module as `__main__` and builds a FRESH
`InferenceRunner` class, so a patch applied to the previously imported class is
discarded and `predict` is never timed. The run then completes normally. Import
the module and patch THAT namespace, then call its own `run()`. The only reason
this was caught is that the harness prints `NO_TIMINGS (predict never called)`
instead of a plausible zero -- keep that guard in every harness.

## Timing harnesses (this directory)

    bench_sweep.py       OURS: one ported model, one length. Uses the jax
                         compilation cache AND pre-inits tokamax's user context
                         (both matter -- see below).
    bench_boltz.py       native boltz-2, on a pre-built batch (forward vs forward)
    bench_rf3_native.py  native rf3; also times the PIPELINE so featurisation can
                         be subtracted
    bench_chai_native.py native chai-1; records the BUCKET it actually folded
    bench_px_native.py   native protenix2 / opendde, via InferenceRunner.predict
                         (already-featurised batch, so forward-only)

Settings are matched across all of them: 3 recycles, 200 diffusion steps, 1
sample, num_msa=1024, single sequence, no template, confidence on. Sequences are
generated from `random.Random(length)` over the 20 amino acids so every harness
folds the SAME sequence at a given length.

## Traps that produced wrong numbers

* **chai-1 BUCKETS.** `AVAILABLE_MODEL_SIZES = [256, 384, 512, 768, 1024, 1536,
  2048]` (`chai_lab/data/collate/utils.py`); every input pads up to the next one.
  64, 128 and 192 tokens all fold as 256, so timing them against our unpadded
  runs compared different problems and read as 9.9x / 6.2x / 3.9x. Only bucket
  BOUNDARIES are fair. bench_chai_native.py now logs `crop_size` so this cannot
  hide. protenix2 and opendde do NOT bucket (protenix logs `N_token 64`, and
  neither data pipeline has pad_size/bucket logic).
* **Not every seam includes the same work.** boltz2 and protenix2/opendde are
  forward-only. chai-1's `run_inference` and rf3's `run()` also featurise and
  write files; subtract that before quoting a ratio (rf3: 4.61x unsplit vs 2.56x
  forward-only at 192 tokens; chai-1 at its 256 boundary: 3.84x unsplit vs 2.03x
  model-only).
* **HOST CPU load inflates everything, and the GPU checks miss it.** Three
  numbers in one session were wrong this way -- protenix2 read 26.74 s where the
  truth is 10.40, opendde 16.22 where it is 8.43, rosettafold3 19.15 where it is
  14.70. EVERY error flattered us, which is why none of them looked suspicious.
  Two guards that do NOT work: "wait for zero GPU processes" (idle
  multiprocessing forkserver workers from a previous run linger for over an hour
  holding 228 MiB each and never clear, so the wait never ends) and GPU
  utilisation alone (it says nothing about host load).

  **Use featurisation time as a load canary.** It is pure CPU work of known cost
  and it moves with system load: in the bad rosettafold3 run it read 3.869 s
  against 2.836 s clean, ~36% inflated, exactly tracking the GPU number. If the
  featurise time is high, throw the measurement away.

  With a quiet machine the within-process spread is ~1% over six calls
  (rosettafold3: 14.55, 14.53, 14.78, 14.83, 14.70). Use >=6 calls, discard the
  first, and take the median of the tail. Kill stray forkservers first:
  `pkill -9 -f multiprocessing.forkserver`.

* **Kernel choice barely matters on an A10.** protenix2 torch 10.40 vs cuEq
  11.61; opendde torch 8.425 vs cuEq 8.478. cuEq warns
  "NOT using the fast SM100f kernel" -- its fast path needs Blackwell. This
  MATCHES the pre-existing ~1.09x/datacentre-only note; an earlier claim in this
  file that measurement had "confirmed" a large cuEq penalty was itself taken
  under load and was wrong.

  Note also that installing `cuequivariance-torch` for protenix2 SILENTLY changed
  opendde, whose kernel default is `auto` and resolves to cuEq whenever the
  package is importable (opendde/config/inference.py: TRIANGLE_KERNELS).

* **openfold3 needs its scratch cleared between runs**: a stale
  `/tmp/of3-of-ubuntu/colabfold_msas/raw` or output dir aborts the next run
  before the model loads.

* **Concurrent GPU work.** Four orphaned benchmark processes made native boltz-2
  read 25 s at 128 tokens where the truth is 8.67 s -- and the contaminated
  numbers FLATTERED us, so they passed a sanity check. Verify the GPU is empty
  before every measurement:
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l`
  Background loops here outlive the turn that started them; run one measurement
  per invocation.
* **Warm-up is not always one call.** On an A100 our model retraced on call 2
  (tokamax's lazily-created jax user context; fixed by
  `ModelRunner._preinit_tokamax_context`). Print every call, not a summary.
* **MSA depth is not a runtime axis.** Featurisation pads `msa` to 16384 rows and
  the model truncates to `config.evoformer.num_msa`, so the input depth changes
  no tensor shape. The config knob costs 10% of the trunk at 64 tokens, 3.9% at
  256, under 1% of a full fold.
* **Feed native the same input.** Native rf3 given `6MRR.pdb` splits the waters
  into a second chain and folds more tokens than we do. Strip to protein-only.
