#!/bin/bash
# WORKDIR must hold q_hidden.py and the hidden-state npz this sweeps over.
# It used to cd into a session scratchpad, which is why this could not be
# re-run: pass WORKDIR explicitly.
cd "${WORKDIR:?set WORKDIR to the directory holding q_hidden.py}"
set -e
for spec in "fp32 32 0" "int8-g128 8 128" "int4-g128 4 128" "int3-g64 3 64" "int2-g64 2 64" "int2-g32 2 32"; do
  set -- $spec
  ~/venv/bin/python q_hidden.py "$1" "$2" "$3"
  ~/venv/bin/python q_fold.py "$1"
  rm -f "hidden_$1.npz"
done
echo "SWEEP COMPLETE"
