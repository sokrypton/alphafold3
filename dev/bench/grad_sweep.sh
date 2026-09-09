#!/bin/bash
# One model per process: a backward pass through a 48-block trunk does not
# reliably free between models in one process, and each check is self-contained
# (its epsilon sweep only has to be internally consistent).
cd /home/ubuntu/alphafold3
OUT=$1; : > $OUT
for m in alphafold3 openfold3 openbind0 protenix05 protenix1 protenix1_20250630 \
         protenix_mini protenix_tiny protenix2 intellifold2 opendde boltz2 \
         rosettafold3 chai1 esmfold2 esmfold2_fast esmfold2_lm600m \
         esmfold2_lm300m af2_ptm af2_multimer; do
  log=/home/ubuntu/alphafold3/dev/bench/out/grad.$m.log
  PYTHONPATH=src:. timeout 1800 ~/venv/bin/python dev/oracles/grad_check.py "$m" > "$log" 2>&1
  # the summary line, or the reason there isn't one
  prec=$(grep -E "^  precision" "$log" | tail -1 | awk '{print $2}')
  g=$(grep -E "^  loss" "$log" | tail -1 | sed 's/^  //')
  n=$(grep -E "^  \|grad\|" "$log" | tail -1 | sed 's/^  //')
  line=$(grep -E "GATE|ERROR" "$log" | tail -1 | sed 's/^  //')
  printf '%-30s %-9s %s\n' "$m" "${prec:-?}" "${line:-NO OUTPUT}" >> $OUT
  [ -n "$g" ] && printf '%-30s   %s\n' "" "$g" >> $OUT
  [ -n "$n" ] && printf '%-30s   %s\n' "" "$n" >> $OUT
done
echo DONE >> $OUT
