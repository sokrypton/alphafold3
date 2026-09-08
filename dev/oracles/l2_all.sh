#!/bin/bash
# Every model with an L2 (token diffusion transformer) native adapter, serially.
# One at a time: two concurrent jobs OOM a 23 GB card.
#
#   bash dev/oracles/l2_all.sh 2>&1 | tee /tmp/l2.txt
#
# max|d| moves ~20% between processes on the same input (XLA autotunes by
# timing); max|d|/rms is the number to read, and in-process reruns are
# bit-identical.
cd "$(dirname "$0")/../.."
P=src:.
V_protenix=/home/ubuntu/protenix
V_of3=/home/ubuntu/openfold-3
V_if2=/home/ubuntu/IntelliFold
V_rf3=/home/ubuntu/rf3_extra:/home/ubuntu/foundry_rf3/src:/home/ubuntu/foundry_rf3/models/rf3/src

run () {  # run <model> <vendor overlay>
  echo "== $1"
  JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=$P:$2 \
    ~/venv/bin/python dev/oracles/diffusion_parity.py "$1" 2>&1 \
    | grep -E 'checkpoint:|native:|ours:|^  a '
}

for m in protenix2 protenix1; do run $m $V_protenix; done
run openfold3    $V_of3
run openbind0    $V_of3
run intellifold2 $V_if2
run rosettafold3 $V_rf3
echo ALL_L2_DONE
