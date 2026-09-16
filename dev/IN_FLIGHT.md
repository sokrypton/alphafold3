# IN FLIGHT as of 2026-09-16 ~23:30 -- two tasks, both mid-run

Written before a context compaction. Neither task is finished; both have
processes running and artefacts in EPHEMERAL locations. Read this first.

SCRATCHPAD = `/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad`
It is in /tmp and does NOT survive a reboot. Copies of what mattered are in
`dev/bench/results/`.

---

## TASK 1 -- runtime benchmarks, native vs jax, across length and MSA rows

### What is running
`bash $SCRATCHPAD/chain2.sh` (one instance only -- verify by PID, not by grep
count; duplicates were started twice tonight) does, in order:

    matrix -> esmfold2 re-run -> OURS sweep -> NATIVE sweep

Stages 1 and 2 are DONE (`$SCRATCHPAD/chain_bench.txt` records MATRIX_DONE
20:57, ESMFIX_DONE 21:01). The OURS sweep is running now.

### Where the numbers go
    live (ephemeral):  $SCRATCHPAD/ours_sweep.tsv        <- the sweep APPENDS here
    live (ephemeral):  $SCRATCHPAD/native_sweep.tsv      <- starts after
    snapshot (repo):   dev/bench/results/ours_sweep_2026-09-16_partial.tsv

**FIRST THING TO DO WHEN THE SWEEP FINISHES: copy both TSVs into
`dev/bench/results/` and commit.** Do not move them while it runs -- the script
appends to that exact path and flocks beside it.

### Progress
79 of 252 cells (14 models x 6 lengths x 3 num_msa). The `num_msa 1` plane is
complete. Render with:

    ~/venv/bin/python ~/ColabDesign2/tools/benchmarks/bench_table.py <ours.tsv> [native.tsv]

`num_msa 1` steady-state seconds, A10: esmfold2 family 0.17-4.29 (AN ORDER OF
MAGNITUDE FASTER, but see the caveat), alphafold3/openbind0 2.8-27, openfold3/
protenix1/rf3 2.85-31, protenix2 3.08-52, boltz2 3.55-37, chai1 3.61-22.85,
intellifold2 3.29-72, opendde 3.80-54.87 and **OOM at 512** -- the only OOM, and
a RESULT rather than a failure (where an implementation stops on this card).

### What was fixed to make this run at all
* `tools/benchmarks/bench_sweep.py` was DEAD CODE: it called
  `params.read_shape_manifest` / `fill_from_manifest`, both deleted with the
  shape manifests, so it raised AttributeError before measuring anything. Same
  root cause as `dev/audit_published.py`. Fixed.
* `run_ours_sweep.sh` listed 9 models including `openbind` (renamed
  `openbind0`, so that cell had been failing silently) and predated protenix1
  and the esmfold2 family. Now 14.
* Added the `num_msa` axis (MSAS=1 256 1024) AND extended the resume key to
  include it -- the old skip test was `^model\tlength\t`, which would have made
  every msa value after the first look already-done and collapsed the axis.

### Caveats that must travel with the numbers
* **esmfold2 rows are NO-LM numbers.** The harness passes `esm=None` and
  ESMFold2 folds from ESM-C's hidden states, so those cells time the trunk
  without the tower. Do NOT print them beside a native esmfold2 figure.
* **alphafold3 native is jax/haiku running the same graph** -- that column is
  port fidelity, not speed.
* **intellifold2 has BOTH** a torch and a jax native; its jax port is the more
  interesting comparison (two independent jax ports of one set of weights).
* No native baseline exists for the esmfold2 family or AF2. AF2 is dropped from
  this benchmark on purpose: DeepMind's AF2 is jax, so there is no native
  counterpart to compare against.
* Column 8 of the TSV is PEAK GPU MEMORY in GiB.
* **NEVER run two sweeps at once.** `NATIVE_SETUP.md` records 17 contaminated
  cells, every error inflating our own numbers. Both scripts flock; do not
  rely on it.
* `bench_table.py`'s speedup section maps our model names onto the native TSV's
  labels and only `rosettafold3 -> rf3` is handled. CHECK the mapping against
  real native output -- a silently empty speedup table is the failure mode.

---

## TASK 2 -- make the Colab notebook fast, and correct, on Colab

### The notebook
`ColabFold2_preview.ipynb` (renamed twice today: `af3-any-model.ipynb` ->
`ColabFold2-Preview.ipynb` -> this; both old paths now 404). Title "ColabFold2
preview". Badge points at `main/ColabFold2_preview.ipynb`, HTTP 200.

### THE IMMEDIATE NEXT STEP
**v3.1.8 was tagged and its wheel build was IN PROGRESS at compaction.** It
carries the real fix for the import failure (see below). When it publishes:

1. Verify the **SLIM** wheel (`pip install alphafold3-colabfold==3.1.8`, ~9 MB
   from PyPI) IMPORTS AND FOLDS with no `components.cif` anywhere. A clean
   Python 3.13 venv exists for this: `$SCRATCHPAD/t313`, and
   `dev/bench/results/slim.ipynb` is the Colab fold test.
2. If it does: point the notebook's install at the slim PyPI wheel, delete the
   `_whl` release-asset URL and the `WHEEL_DL_DONE` prefetch, and RETIRE the fat
   `+data` wheel machinery (the `bundle_data` dispatch input and the
   `github-release` job in `.github/workflows/release_wheel.yaml`).
3. Then `ccd_fetch.py` is IN the wheel, so stop fetching it from `main` (see
   the drift note below) and pin both halves to the tag again.

Do not assume the C++ fix works in a built wheel because it compiles. It
compiled in CI (`Install Python dependencies` and `Build data` both green); that
is not the same as the published artefact importing without the dictionary.

### The import failure, and the real fix
`pip install alphafold3-colabfold` died with

    ImportError: Could not find the libcifpp components.cif file.

The error was OURS, not libcifpp's: `src/alphafold3/model/mkdssp_pybind.cc`
threw from `RegisterModuleMkdssp` -- module REGISTRATION -- so it failed the
import of the whole `alphafold3.cpp` module, including `cif_dict` which
featurisation needs, for the sake of DSSP. DSSP is used in exactly one place
(`confidences.predicted_disorder`, the AlphaFold-RSA metric) and no fold touches
it. The check is now deferred to the `get_dssp` call (commit 7f79c77).

Two workarounds were tried and REVERTED in favour of that: a 130 MB fat wheel
bundling the full 518 MB dictionary, and a 35-component dictionary bundled in
the package with an `__init__` hook setting `LIBCIFPP_DATA_DIR`. The fat wheel
is still what the notebook installs TODAY and still works.

Wheel composition, which is why this mattered (130 MB fat wheel, zipped):
libcifpp DATA 120.27 MB (92%), test_data 5.05, libcifpp lib+headers 3.01,
cpp .so 1.31, all the Python 0.69. The library costs 3 MB; the dictionary
costs 120.

### Measured, on FRESH Colab T4 sessions
    deps install            ~minutes (jax pin) -> 8.5 s
    build_data -> rcsb CCD  48.0 s -> 0.6 s (40 components)
    ccd.pickle              505 MB -> 0.30 MB
    setup end to end        ~180 s label -> 91 s (before the CCD change landed)
    fold                    80 s, PASS, 2 cifs, 165 atoms
    slim wheel + minimal components.cif: IMPORTS (verified locally in t313)

### What changed in the notebook
* **No jax pin.** Colab ships 0.11.1; the pin forced a downgrade and re-pulled
  the CUDA stack. jax 0.11.1 imports, traces and folds.
  **CAVEAT: verified on T4 ONLY.** A T4 takes the XLA attention path; tokamax's
  Triton kernels are restricted to datacenter GPUs, so an A100/H100 exercises
  code a T4 never touches. Test one A100 run before this is anyone's default;
  if a datacenter GPU misbehaves, restore the pin first.
* Added `ml_collections` (declared dep, skipped by `--no-deps`, imported only by
  the af2 path -- which is why every af3 model worked and `af2_ptm` died).
* Dropped `awscli` and `py3Dmol`: installed, never used.
* `py2Dmol` now from `git+https://github.com/sokrypton/py2Dmol.git`.
* CCD from `files.rcsb.org/ligands/download/<CODE>.cif` per component
  (`src/alphafold3/constants/ccd_fetch.py`) instead of `build_data`.
* The 130 MB wheel download now overlaps the pip installs (they were serial at
  8.5 + 9.6 + 12.7 s). Only the DOWNLOAD is backgrounded -- two pips writing one
  site-packages would race.
* `aria2c -x 16` for AF2's 5.3 GB parameter tar
  (`weights._download_parallel`), apt-installed for af2 only.

### KNOWN DRIFT, deliberate, fix it in step 3 above
The notebook fetches `ccd_fetch.py` from **`main`**, not from the pinned tag,
because the module is not in the published 3.1.7 wheel. This is the exact drift
removed earlier the same evening (the old notebook mixed a fixed v3.1.5 binary
with a moving branch head). It is a stopgap.

### Why the rcsb CCD trick is safe
Both pickles are built from the SAME fetched set, so they are self-consistent:
NAG lands in `GLYCAN_LINKING_LIGANDS` exactly as with the full dictionary, and
all fields are byte-identical to libcifpp's for ALA SER GOL ATP SEP NAG DA U
(0 differing). A component the input names and we did not fetch raises KeyError
-- loud. What must never happen is fetching a SUBSET and treating it as the
dictionary: a glycan missing from that set is silently a generic ligand. Hence
`codes_for_input` is generous, and `build_data` refuses to build a pickle from a
dictionary holding under 1000 components (fires on the 35-component file, passes
on the full 50,433).

### Colab CLI, now working and reusable
    ~/colabcli_venv/bin/colab --auth adc new --gpu T4
    ~/colabcli_venv/bin/colab --auth adc exec -f <notebook> --timeout 1800
    ~/colabcli_venv/bin/colab --auth adc stop        # or it burns compute units

* `--auth adc` is REQUIRED on every call; plain `colab` retries OAuth and
  prompts, which has no TTY under Claude Code and aborts.
* ADC is already configured (`~/.config/gcloud/application_default_credentials.json`).
  It needed one interactive `gcloud auth application-default login
  --no-launch-browser` in tmux -- the authorization code is bound to a PKCE
  verifier held by the process that printed the URL, so piping it or handing it
  to someone else CANNOT work.
* Pin `jupyter-kernel-client==0.15.0`. 1.0.2 exports `JupyterKernelClient` where
  colab-cli 0.6.0 calls `KernelClient`.
* `%%time` is a CELL magic and must be line 1. Colab strips `#@title` form
  comments; colab-cli's kernel does not, so the magic raises UsageError and
  ABORTS THE WHOLE CELL -- which looked like "No module named alphafold3", i.e.
  a false negative that pointed at the install rather than the harness.
* A T4 (compute capability 7.5) REFUSES to run without
  `XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter`.
* Test harnesses are in `dev/bench/results/`: `colab_smoke.ipynb` (install +
  import chain), `colab_af2_smoke.ipynb` (the af2 path), `e2e.ipynb` (real
  install cell, timed, plus a fold), `slim.ipynb` (the slim-wheel experiment),
  `probe.py` (what Colab preinstalls -- RUN IT ON A FRESH SESSION; run on a
  used one it reports everything as preinstalled, which would justify deleting
  installs we need).

---

## Open decisions, all the user's
1. **CI goldens.** `test_config`, `test_featurisation`,
   `test_all_examples_are_valid` fail on main because the fork legitimately adds
   fields (`coda`, `ligand_ligand_bond_order`). Regenerate so CI can catch real
   drift, or keep red as an upstream-divergence signal? A permanently red CI is
   the same failure mode as the five dead guards found today.
2. **AF2 MSA sizes**: ours 512/1024 both paths; stock 512/5120 (ptm) and
   508/2048 (multimer).
3. **Recycle counts**: we run 11 trunk passes; of3/boltz2 run 4, chai 3.
4. **`--weights_precision` default is `int8`** -- the path users hit without
   choosing it, and the one that was broken for six days.
5. `dump_af3_batch.py` is untracked in the repo root, is not ours, and
   references `tools/gpu/fold.js` and `oracle-dumps/` which do not exist here.
6. `~/publish_stage` holds 11 GB of staged wheels; the uploads are verified, so
   it is safe to delete.
