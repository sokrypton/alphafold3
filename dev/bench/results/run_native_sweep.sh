#!/bin/bash
# Native length sweep: every torch baseline, every length, resumable.
#
#   bash tools/benchmarks/run_native_sweep.sh [out.tsv]
#
# Each cell is measured on a QUIET machine. That matters more than anything else
# here: three numbers in this project were wrong because of host load, every one
# inflated, every one flattering us (protenix2 read 26.74 s against a true 10.40).
#
# Guards, in the order they were learned:
#   * kill orphaned multiprocessing forkservers FIRST. openfold3 leaves them for
#     over an hour holding 228 MiB each; they never exit, so "wait for zero GPU
#     processes" waits forever.
#   * then require an idle GPU *and* a low host load average -- GPU checks alone
#     are blind to CPU contention, which is what actually corrupted the numbers.
#   * every call is recorded, so a run whose spread is wide can be spotted and
#     discarded. Quiet-machine spread is ~1% over 6 calls.
OUT=${1:-/tmp/native_sweep.tsv}
# The bench_*.py harnesses live BESIDE this script; they used to live in a
# session scratchpad and the path was never updated when they were moved into
# the repo, so every cell of the 2026-09-17 run died with "can't open file".
# Same root cause as bench_sweep.py's calls into deleted helpers.
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Generated INPUTS still live outside the repo (rf3's PDBs, boltz's processed
# dirs); DATA overrides where to find them.
S=${DATA:-/tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad}
REPS=${REPS:-3}
touch "$OUT"

# SINGLE INSTANCE ONLY. Two copies of this script once ran together for 17 cells
# and every one was contaminated -- opendde@128 read 58.86 s against a true
# ~10 s. The per-cell quiesce cannot prevent it: each copy waits for an idle GPU
# and sees one between the other's cells. A launch that "failed" can still have
# started its nohup, so this must be enforced here, not by being careful.
# NEVER `rm` this lock file. flock locks an INODE, not a path: deleting it while
# a sweep holds it lets the next launch create a fresh inode, lock that instead,
# and run concurrently -- which is exactly how two sweeps got past this guard and
# contaminated a cell (rosettafold3@128 featurise 21.2 s against a clean 3.5 s).
exec 9>"${OUT}.lock"
flock -n 9 || { echo "another sweep is running (${OUT}.lock) -- refusing"; exit 1; }

quiesce () {
  # Wait for an actually-idle machine, then stop waiting. An earlier version
  # required the 1-minute LOAD AVERAGE below 1.5 and deadlocked: load is a
  # decaying average, so after a burst of contention it sits near 2 for minutes
  # with the GPU at 0% and nothing running -- the sweep blocked forever on a
  # machine that was already idle. Load average is a lagging indicator; what
  # matters is whether anything is RUNNING now.
  pkill -9 -f "multiprocessing.forkserver" 2>/dev/null
  sleep 3
  local waited=0
  while [ "$waited" -lt 600 ]; do
    local g b
    g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    b=$(ps -eo args | grep -c "[b]ench_" || true)
    [ "$g" -eq 0 ] && [ "$b" -eq 0 ] && break
    sleep 20; waited=$((waited + 20))
  done
  # Every harness also prints its featurisation time, which is pure CPU work of
  # known cost -- the reliable post-hoc canary for load (a contaminated run read
  # 3.87 s against 2.84 s clean). Check it when reading results, not here.
}

run_cell () {                     # model  length  command...
  local m=$1 L=$2; shift 2
  # match a COMPLETED cell only: a FAILED/OOM row should be retried, and a
  # duplicate row from the concurrent-sweep incident must not mask a real gap.
  grep -qE "^$m	$L	NATIVE[ -]" "$OUT" 2>/dev/null && { echo "skip $m $L"; return; }
  quiesce
  local err=/tmp/_cell.$$ line
  line=$("$@" 2>"$err" | grep -aE "^NATIVE[ -]" | tail -1)
  if [ -n "$line" ]; then
    printf '%s\t%s\t%s\n' "$m" "$L" "$line" >> "$OUT"; echo "ok   $m $L"
  elif grep -qiE "out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED" "$err"; then
    printf '%s\t%s\tOOM\n' "$m" "$L" >> "$OUT"; echo "OOM  $m $L"
  else
    printf '%s\t%s\tFAILED\n' "$m" "$L" >> "$OUT"
    echo "FAIL $m $L: $(tail -1 "$err" | cut -c1-80)"
  fi
  rm -f "$err"
}

PX=/home/ubuntu/boltz_gpu_venv/bin/python
# 64 INCLUDED. The loop started at 128, so every native curve was missing its
# first point -- and since a fresh curve replaces the 2026-09-03 fallback
# wholesale in the plot, the old 64-token values disappeared with it.
for L in 64 128 192 256 384 512 768; do
  run_cell boltz2 $L /home/ubuntu/boltz_gpu_venv/bin/python $BENCH/bench_boltz.py \
      200 off 3 "$S/nb/out$L/boltz_results_L$L/processed"
  run_cell rosettafold3 $L env \
      PYTHONPATH=/home/ubuntu/rf3_extra:/home/ubuntu/foundry_rf3/src:/home/ubuntu/foundry_rf3/models/rf3/src \
      /home/ubuntu/venv/bin/python $BENCH/bench_rf3_native.py "$S/nr/L$L.pdb" $REPS
  run_cell protenix2 $L env PX_KERNELS=torch PYTHONPATH=/home/ubuntu/if2_extra \
      $PX $BENCH/bench_px_native.py protenix2 $L $REPS
  run_cell opendde $L env PX_KERNELS=torch PYTHONPATH=/home/ubuntu/if2_extra \
      $PX $BENCH/bench_px_native.py opendde $L $REPS
  rm -rf /tmp/of3-of-ubuntu "/tmp/_of3_out_$L" 2>/dev/null   # of3 refuses stale scratch
  run_cell openfold3 $L env OF3_L=$L OF3_REPS=$REPS \
      PYTHONPATH=/home/ubuntu/of3_extra:/home/ubuntu/openfold-3 \
      $PX $BENCH/bench_of3_native.py $L $REPS
done
# chai-1 ONLY on bucket boundaries: it pads to the next AVAILABLE_MODEL_SIZE, so
# off-boundary lengths fold a different problem than we do.
for L in 384 512 768; do
  run_cell chai1 $L env CHAI_DOWNLOADS_DIR=/home/ubuntu/chai1_weights TQDM_DISABLE=1 \
      /home/ubuntu/chai_venv/bin/python $BENCH/bench_chai_native.py $L $REPS
done
echo "NATIVE SWEEP DONE"
