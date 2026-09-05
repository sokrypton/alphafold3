#!/bin/bash
cd /tmp/claude-1000/-home-ubuntu-ColabDesign2/77aa66c7-a908-4cb6-bf0e-1ff700d68150/scratchpad
set -e
for spec in "fp32 32 0" "int8-g128 8 128" "int4-g128 4 128" "int3-g64 3 64" "int2-g64 2 64" "int2-g32 2 32"; do
  set -- $spec
  ~/venv/bin/python q_hidden.py "$1" "$2" "$3"
  ~/venv/bin/python q_fold.py "$1"
  rm -f "hidden_$1.npz"
done
echo "SWEEP COMPLETE"
