#!/bin/bash
# Diagnose the twelve BAD comparisons parity_audit.py found. Launch AFTER the
# full matrix -- it wants the GPU to itself.
#
# The order is deliberate: the p_atom_pair tension first, because if that turns
# out to be a comparison artifact then four of the twelve evaporate and the
# remaining work is much smaller than it looks.
cd /home/ubuntu/alphafold3
export JAX_DEFAULT_MATMUL_PRECISION=highest
V_protenix=/home/ubuntu/protenix
V_of3=/home/ubuntu/openfold-3
V_if2=/home/ubuntu/IntelliFold
V_rf3=/home/ubuntu/rf3_extra:/home/ubuntu/foundry_rf3/src:/home/ubuntu/foundry_rf3/models/rf3/src
overlay () { case "$1" in
  protenix1|protenix2) echo "$V_protenix" ;;
  openfold3|openbind0) echo "$V_of3" ;;
  intellifold2)        echo "$V_if2" ;;
  rosettafold3)        echo "$V_rf3" ;;
  *)                   echo "" ;; esac; }

echo "########## 1. p_atom_pair: artifact or bug? (protenix2 is the worst at 9.64)"
for m in protenix2 protenix1 openfold3 openbind0; do
  o=$(overlay $m)
  echo "===== $m"
  PYTHONPATH=src:.${o:+:$o} DIAG=1 ~/venv/bin/python dev/oracles/atom_parity.py $m 2>&1 \
    | grep -E "p_atom_pair|DIAG p|p_lm ours|Error"
done

echo "########## 2. rosettafold3 owns 4 of the 12 -- is it an atom-set mismatch, like esmfold2's OXT?"
o=$(overlay rosettafold3)
PYTHONPATH=src:.${o:+:$o} DIAG=1 ~/venv/bin/python dev/oracles/atom_parity.py rosettafold3 2>&1 \
  | grep -E "a_token|q_atom|c_atom_cond|DIAG|atoms|Error"
# NOT an atom-set mismatch: rf3's gate builds native's side from OUR features,
# so both sides see the same atoms by construction. And NOT the known chiral
# gap either -- the gate DROPS process_ch by default precisely so it measures
# everything else. So c_atom_cond at 5.11e-02 is a real, unknown difference in
# rf3's atom conditioning, and q_atom 0.84 / a_token 0.57 are what it grows
# into. RF3_CHIRAL=1 sizes the known missing term separately, which is worth
# having as a number even though it is not the cause.
echo "----- RF3_CHIRAL=1, to size the one term the port does not implement"
PYTHONPATH=src:.${o:+:$o} RF3_CHIRAL=1 ~/venv/bin/python dev/oracles/atom_parity.py rosettafold3 2>&1 \
  | grep -E "a_token|q_atom|c_atom_cond|chiral|Error"

echo "########## 3. intellifold2's template_embed (0.157) and x_denoised (0.141)"
o=$(overlay intellifold2)
PYTHONPATH=src:.${o:+:$o} ~/venv/bin/python dev/oracles/template_parity.py intellifold2 2>&1 \
  | grep -E "corr|Error" | head -4
PYTHONPATH=src:.${o:+:$o} DIAG=1 ~/venv/bin/python dev/oracles/denoise_parity.py intellifold2 2>&1 \
  | grep -E "corr|DIAG|per-atom|Error" | head -6

echo "########## 4. rosettafold3 single_cond (0.124) and x_denoised (0.153)"
o=$(overlay rosettafold3)
PYTHONPATH=src:.${o:+:$o} ~/venv/bin/python dev/oracles/conditioning_parity.py rosettafold3 2>&1 \
  | grep -E "corr|Error" | head -4
PYTHONPATH=src:.${o:+:$o} DIAG=1 ~/venv/bin/python dev/oracles/denoise_parity.py rosettafold3 2>&1 \
  | grep -E "corr|DIAG|per-atom|Error" | head -6

echo "########## 5. the two new UNTESTED conditioning adapters"
for m in boltz2 opendde; do
  o=$(overlay $m); [ "$m" = boltz2 ] && o=/home/ubuntu/BoltzDesign1/boltz2/src
  [ "$m" = opendde ] && o=/home/ubuntu/OpenDDE
  echo "===== $m L2.conditioning (first run ever)"
  PYTHONPATH=src:.${o:+:$o} ~/venv/bin/python dev/oracles/conditioning_parity.py $m 2>&1 \
    | grep -E "corr|checkpoint|native|Error|Traceback" | head -8
done
echo BADDIAGDONE

echo "########## 6. the released line's 5.6e-03 MSA-path residual"
# The three msa=0 variants are exact and the three msa=4 ones are not, so the
# residual is the MSA path. Ours runs the stack on num_msa rows (1 real +
# padding) where the reference runs on the 1 real row, so NUM_MSA=1 says whether
# the padded rows are the difference.
for nm in 1 4 1024; do
  echo "===== esmfold2 NUM_MSA=$nm"
  PYTHONPATH=src:. NUM_MSA=$nm ~/venv/bin/python dev/oracles/esmfold2_localise_trunk.py esmfold2 2>&1 \
    | grep -E "corr|relerr|stage|Error" | head -5
done
echo BADDIAG6DONE
