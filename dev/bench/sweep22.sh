#!/bin/bash
cd /home/ubuntu/alphafold3
S=${SWEEP_AUX:-$HOME/alphafold3/dev/bench}
out=$S/sweep20.txt; : > $out
plain="alphafold3 boltz2 chai1 intellifold2 openbind0 opendde openfold3 protenix05 protenix1 protenix1_20250630 protenix2 protenix_mini protenix_tiny rosettafold3"
for m in $plain; do
  printf "%-30s " "$m" >> $out
  PYTHONPATH=src:. ~/venv/bin/python dev/oracles/fold_check.py $m ~/6MRR.pdb 2>&1 | tail -1 >> $out
done
for m in esmfold2 esmfold2_fast esmfold2_lm600m esmfold2_lm300m; do
  printf "%-30s " "$m" >> $out
  ESMC_HIDDEN=$S/hid_6b.npz PYTHONPATH=src:. ~/venv/bin/python dev/oracles/fold_check.py $m ~/6MRR.pdb 2>&1 | tail -1 >> $out
done
printf "%-30s " "esmfold2_lm600m" >> $out
ESMC_HIDDEN=$S/hid_new.npz PYTHONPATH=src:. ~/venv/bin/python dev/oracles/fold_check.py esmfold2_lm600m ~/6MRR.pdb 2>&1 | tail -1 >> $out
printf "%-30s " "esmfold2_lm300m" >> $out
ESMC_HIDDEN=$S/hid_300m.npz PYTHONPATH=src:. ~/venv/bin/python dev/oracles/fold_check.py esmfold2_lm300m ~/6MRR.pdb 2>&1 | tail -1 >> $out
echo "SWEEP20 DONE" >> $out
