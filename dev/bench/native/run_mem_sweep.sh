#!/bin/bash
# PEAK DEVICE MEMORY per (model, length). Separate from the timing sweep because
# the two want different things: peak memory is reached on the FIRST call, so one
# call is enough, while a timing number needs several and costs 4x as much.
#
#   bash tools/benchmarks/run_mem_sweep.sh [out.tsv]
#
# THE TIMINGS IN THIS FILE ARE NOT USABLE -- one call each, so every row carries
# the compile. Read column 8 (peak GiB) and nothing else. It shares the timing
# sweep's compile cache, so most cells skip the compile entirely.
OUT=${1:-/tmp/mem_sweep.tsv}
S=/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad
MODELS=${MODELS:-alphafold3 openfold3 openbind intellifold2 opendde boltz2 protenix2 rosettafold3 chai1}
LENGTHS=${LENGTHS:-64 128 192 256 384 512 768 1024}
touch "$OUT"
exec 9>"${OUT}.lock"
if ! flock -n 9; then echo "another mem sweep holds ${OUT}.lock; refusing"; exit 1; fi

for L in $LENGTHS; do
  for M in $MODELS; do
    if grep -qP "^${M}\t${L}\t" "$OUT"; then echo "skip $M $L"; continue; fi
    for i in $(seq 1 120); do
      [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
      sleep 5
    done
    echo "=== $M $L ==="
    cell_err="${OUT}.${M}.${L}.err"
    if SWEEP_CACHE_DIR=$S/sweep_cache timeout 7200 ~/venv/bin/python \
         tools/benchmarks/bench_sweep.py "$M" "$HOME/ported/$M" "$L" 1024 1 \
         >> "$OUT" 2>"$cell_err"; then
      rm -f "$cell_err"
    elif grep -qiE 'RESOURCE_EXHAUSTED|Out of memory|OOM when allocating' "$cell_err"; then
      echo -e "${M}\t${L}\tOOM" >> "$OUT"
    else
      echo -e "${M}\t${L}\tFAILED" >> "$OUT"
    fi
  done
done
echo "MEM SWEEP DONE" >> "$OUT"
