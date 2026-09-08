#!/bin/bash
# The ESMFold2 family, each with ITS OWN ESM-C shim input. A shared one reads
# corr 0.026 and folds 8.8 A (see memory esmfold2-port), so the wrong hidden
# file measures the gradient of a different model.
#
# Note what this does NOT change: the LM input is computed outside the graph
# from the TRUE sequence, so it is a constant w.r.t. soft_seq either way. This
# fixes the OPERATING POINT, not the gradient path.
cd /home/ubuntu/alphafold3
B=dev/bench
OUT=$1; : > $OUT
run() {  # model, hidden-file
  local log=/home/ubuntu/alphafold3/dev/bench/out/gradlm.$1.log
  ESMC_HIDDEN=$B/$2 PYTHONPATH=src:. timeout 1800 ~/venv/bin/python \
      dev/oracles/grad_check.py "$1" > "$log" 2>&1
  local g=$(grep -E "^  \|grad\|" "$log" | tail -1 | sed 's/^  //')
  local l=$(grep -E "GATE|ERROR" "$log" | tail -1 | sed 's/^  //')
  printf '%-32s %s\n' "$1" "${l:-NO OUTPUT}" >> $OUT
  [ -n "$g" ] && printf '%-32s   %s\n' "" "$g" >> $OUT
}
for m in esmfold2 esmfold2_fast esmfold2_exp esmfold2_exp_fast \
         esmfold2_exp_cutoff2025 esmfold2_exp_fast_cutoff2025; do
  run $m hid_6b.npz
done
run esmfold2_lm600m hid_new.npz
run esmfold2_lm300m hid_300m.npz
echo DONE >> $OUT
