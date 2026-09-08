#!/bin/bash
# L4: every model with a confidence-head native adapter, serially.
#   bash dev/oracles/l4_all.sh 2>&1 | tee /tmp/l4.txt
cd "$(dirname "$0")/../.."
V_protenix=/home/ubuntu/protenix
V_of3=/home/ubuntu/openfold-3
V_if2=/home/ubuntu/IntelliFold
V_rf3=/home/ubuntu/rf3_extra:/home/ubuntu/foundry_rf3/src:/home/ubuntu/foundry_rf3/models/rf3/src

run () {  # run <model> <vendor overlay>
  echo "== $1"
  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=src:.:$2 \
    ~/venv/bin/python dev/oracles/confidence_parity.py "$1" 2>&1 \
    | grep -E 'checkpoint:|native:|ours:|rep-atom|NOTE|rounded|cleared|^  (full|plddt|resolved)'
}

for m in protenix2 protenix1; do run $m $V_protenix; done
run openfold3    $V_of3
run openbind0    $V_of3
run intellifold2 $V_if2
run rosettafold3 $V_rf3
run opendde      /home/ubuntu/OpenDDE
echo ALL_L4_DONE
