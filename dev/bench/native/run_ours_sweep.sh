#!/bin/bash
# OUR ports across LENGTH, resumable, one cell at a time.
#
#   bash tools/benchmarks/run_ours_sweep.sh [out.tsv]
#
# Serialised on purpose: two of these at once OOM the GPU and, worse, inflate
# each other's numbers without failing (see NATIVE_SETUP.md, "HOST CPU load").
# Every cell is appended as it completes, so a kill loses one cell, not the run.
OUT=${1:-/tmp/ours_sweep.tsv}
S=/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad
# ALL FOURTEEN on the AF3 graph. The old list had nine and carried `openbind`,
# which was renamed `openbind0`, so that cell had been silently failing; it also
# predated protenix1 and the esmfold2 family.
#
# af2_ptm / af2_multimer are NOT here: they are a sibling network, not the AF3
# graph this harness builds (`model.Model(cfg)`), so they need their own runner.
#
# The esmfold2 rows are NO-LM numbers. This harness passes `esm=None`, and
# ESMFold2 folds from ESM-C's hidden states -- so those cells time the trunk
# without the tower, which is a real configuration but NOT what a native
# esmfold2 run does. Do not put them beside a native esmfold2 number.
MODELS=${MODELS:-alphafold3 openfold3 openbind0 protenix1 protenix2 intellifold2 opendde boltz2 rosettafold3 chai1 esmfold2 esmfold2_fast esmfold2_lm600m esmfold2_lm300m}
LENGTHS=${LENGTHS:-64 128 192 256 384 512}
# NUM_MSA is the CONFIG's row count, not the input alignment's depth -- see
# bench_sweep.py's docstring. Featurisation pads the msa to 16384 rows whatever
# the input carries and the model truncates to config.evoformer.num_msa, so
# sweeping input depth gives a flat line and this is the axis that costs time.
MSAS=${MSAS:-1 256 1024}
touch "$OUT"

exec 9>"${OUT}.lock"
if ! flock -n 9; then echo "another sweep holds ${OUT}.lock; refusing"; exit 1; fi
# NEVER `rm` this lock file: flock locks an INODE, so deleting it lets the next
# launch lock a fresh one and run concurrently -- which is how two sweeps once
# corrupted 17 cells.

for N in $MSAS; do
 for L in $LENGTHS; do
  for M in $MODELS; do
    # The skip key MUST include num_msa. Testing only model+length made every
    # msa value after the first look already-done, which would have collapsed
    # the whole second axis into a copy of the first.
    if grep -qP "^${M}\t${L}\t${N}\t" "$OUT"; then echo "skip $M $L $N (done)"; continue; fi
    # wait for a quiet GPU before each cell
    for i in $(seq 1 120); do
      busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
      [ "$busy" -eq 0 ] && break
      sleep 5
    done
    echo "=== $M $L msa=$N ==="
    # OOM is a RESULT (where an implementation stops on this card), a crash is a
    # bug, and a plot that shows them the same way hides one of them. Keep each
    # cell's stderr so the distinction is made from the message, not guessed.
    cell_err="${OUT}.${M}.${L}.${N}.err"
    if SWEEP_CACHE_DIR=$S/sweep_cache timeout 7200 ~/venv/bin/python \
         tools/benchmarks/bench_sweep.py "$M" "$HOME/ported/$M" "$L" "$N" 4 \
         >> "$OUT" 2>"$cell_err"; then
      rm -f "$cell_err"
    elif grep -qiE 'RESOURCE_EXHAUSTED|Out of memory|OOM when allocating' "$cell_err"; then
      echo -e "${M}\t${L}\t${N}\tOOM" >> "$OUT"
    else
      echo -e "${M}\t${L}\t${N}\tFAILED" >> "$OUT"
    fi
    cat "$cell_err" >> "${OUT}.err" 2>/dev/null
  done
 done
done
echo "OURS SWEEP DONE" >> "$OUT"
