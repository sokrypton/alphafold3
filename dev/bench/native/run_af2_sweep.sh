#!/bin/bash
# AlphaFold 2 across LENGTH x MSA cluster count, resumable, one cell at a time.
#
#   bash tools/benchmarks/run_af2_sweep.sh [out.tsv]
#
# Same shape and same guards as run_ours_sweep.sh: serialised, flock-protected,
# waits for a quiet GPU, appends each cell as it lands. AF2 needs its own driver
# because it is a sibling network rather than a model on the AF3 graph.
OUT=${1:-/tmp/af2_sweep.tsv}
S=/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad
MODELS=${MODELS:-af2_ptm af2_multimer}
LENGTHS=${LENGTHS:-64 128 192 256 384 512}
# AF2's MSA sizes are real tensor shapes, not a config truncation, so these cost
# real time. 512 is AF2's own default cluster count.
MSAS=${MSAS:-1 128 512}
PARAMS=${PARAMS:-$HOME/params}
touch "$OUT"
exec 9>"${OUT}.lock"
if ! flock -n 9; then echo "another af2 sweep holds ${OUT}.lock; refusing"; exit 1; fi

for N in $MSAS; do
 for L in $LENGTHS; do
  for M in $MODELS; do
    if grep -qP "^${M}\t${L}\t${N}\t" "$OUT"; then echo "skip $M $L $N (done)"; continue; fi
    for i in $(seq 1 120); do
      busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
      [ "$busy" -eq 0 ] && break
      sleep 5
    done
    echo "=== $M $L msa=$N ==="
    cell_err="${OUT}.${M}.${L}.${N}.err"
    if SWEEP_CACHE_DIR=$S/sweep_cache AF3_SRC=$HOME/alphafold3 timeout 7200 \
         ~/venv/bin/python tools/benchmarks/bench_sweep_af2.py "$M" "$PARAMS" "$L" "$N" 4 \
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
echo "AF2 SWEEP DONE" >> "$OUT"
