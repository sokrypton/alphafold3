#!/bin/bash
# EVERY parity gate in this directory, over every model, in one resumable run.
#
#   bash dev/oracles/run_all_parity.sh                 # L0-L4 (the module gates)
#   bash dev/oracles/run_all_parity.sh L5 L6           # the fold-level screens
#   bash dev/oracles/run_all_parity.sh all             # everything
#   FORCE=1 bash dev/oracles/run_all_parity.sh L1b     # ignore cached logs
#   MODELS="boltz2 protenix2" bash dev/oracles/run_all_parity.sh
#
# WHY A DRIVER AND NOT A LIST IN A DOCUMENT. Every number in PARITY.md came out
# of one of these scripts with a specific vendor overlay on PYTHONPATH and
# JAX_DEFAULT_MATMUL_PRECISION=highest, and getting either wrong produces a
# plausible degraded number rather than an error -- bf16 params and tf32 matmuls
# faked six protenix2 "bugs" once. A refactor needs to re-run all of it, and
# reconstructing thirty command lines by hand is where that goes wrong.
#
# HOW IT DECIDES WHAT TO RUN. It asks each gate about each model rather than
# carrying a table: a gate with no adapter for a model exits non-zero with
# "no native adapter for ...", which is recorded as SKIP, not FAIL. So adding an
# adapter is picked up here with no change to this file, and the summary doubles
# as the coverage matrix -- if it disagrees with PARITY.md, PARITY.md is stale.
#
# SERIALLY, ALWAYS. Two concurrent jobs OOM a 23 GB card. And note max|d| moves
# ~20% between processes on the same input (XLA autotunes by timing), so read
# corr and max|d|/rms; in-process reruns are bit-identical.
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
PY=${PY:-~/venv/bin/python}
PY_ESM=${PY_ESM:-~/venv_esm/bin/python}

# --- vendor source overlays. One implementation usually serves a whole family,
#     which is the leverage: ~/protenix covers both protenix releases and
#     ~/openfold-3 covers openfold3 + openbind0.
V_protenix=/home/ubuntu/protenix
V_of3=/home/ubuntu/openfold-3
V_if2=/home/ubuntu/IntelliFold
V_dde=/home/ubuntu/OpenDDE
V_rf3=/home/ubuntu/rf3_extra:/home/ubuntu/foundry_rf3/src:/home/ubuntu/foundry_rf3/models/rf3/src
V_boltz2=/home/ubuntu/BoltzDesign1/boltz2/src
# The overlay is per MODEL, and a wrong one is a silent wrong answer, not an
# ImportError -- rf3's MSA gate ran green against openfold-3 on PYTHONPATH.
vendor () {
  case "$1" in
    protenix1|protenix2)                echo "$V_protenix" ;;
    openfold3|openbind0)                echo "$V_of3" ;;
    intellifold2)                       echo "$V_if2" ;;
    opendde)                            echo "$V_dde" ;;
    rosettafold3)                       echo "$V_rf3" ;;
    boltz2)                             echo "$V_boltz2" ;;
    # Everything else runs with no overlay: chai1's native is a TorchScript
    # artifact loaded by path, esmfold2's lives in ~/venv_esm (see l1b_esm),
    # and alphafold3 IS the reference implementation.
    *)                                  echo "" ;;
  esac
}

# Per DATE by default, which means a run spanning midnight writes into two
# directories -- set LOGDIR explicitly to keep one run together.
LOGDIR=${LOGDIR:-$ROOT/dev/oracles/parity_runs/$(date +%Y-%m-%d)}
mkdir -p "$LOGDIR"
SUMMARY=$LOGDIR/summary.tsv
[ -f "$SUMMARY" ] || printf 'gate\tmodel\tstatus\theadline\n' > "$SUMMARY"

MODELS=${MODELS:-$($PY -c 'import sys; sys.path.insert(0,"src")
from alphafold3.model import model_config as c; print(" ".join(c.MODELS))')}

# gate <name> <script+args...> -- runs one gate for one model.
#   $1 log tag   $2 model   $3 grep pattern for the headline   rest: argv
# A non-zero exit is not automatically a failure here, which is the whole reason
# classification is its own function: a gate with no adapter for a model SAYS so
# and exits 1 (SKIP), and dev/audit_coverage.py exits 1 to REPORT unaccounted
# checkpoint tensors, which is a finding to read rather than a crash (WARN).
# Calling those FAIL would bury the real failures in noise. Applied to cached
# logs too, so a re-run re-classifies instead of keeping an old verdict.
classify () {  # classify <log> -> status on stdout
  local log=$1
  local rc; rc=$(sed -n 's/^__GATE_EXIT //p' "$log" | tail -1)
  if [ "$rc" = 0 ]; then echo OK; return; fi
  if [ "$rc" = 124 ]; then echo TIMEOUT; return; fi
  if grep -qi 'no native adapter\|no converter registered\|nothing to audit\|has no msa_encoder\|has no final-block\|no weights for\|no native dump at\|run first:\|No module named\|KeyError' "$log"; then
    echo SKIP; return
  fi
  # A RESOURCE failure is not a result. Two gates sharing a 23 GB card is the
  # usual cause and it says nothing about the port, so it gets its own status
  # rather than being counted as a failure.
  if grep -qi 'RESOURCE_EXHAUSTED\|CUDA_ERROR_OUT_OF_MEMORY\|Out of memory\|solver_handle_pool\|gpusolverDnCreate\|CUBLAS_STATUS_\|cuSolver internal error' "$log"; then
    echo OOM; return
  fi
  # NONZERO counts only. `[1-9][0-9]*` matters: every trunk gate prints
  # "0 unmapped" on success, and matching a bare 'unmapped' made opendde's L1
  # read WARN off a success line.
  if grep -qEi '[1-9][0-9]* (unaccounted for|unmapped)' "$log"; then
    echo WARN; return
  fi
  echo FAIL
}

gate () {
  local tag=$1 model=$2 pat=$3; shift 3
  local log=$LOGDIR/$tag.$model.log
  if [ -n "${FORCE:-}" ] || [ ! -f "$log" ] || ! grep -q '^__GATE_EXIT ' "$log"; then
    local overlay; overlay=$(vendor "$model")
    local pp=src:.
    [ -n "$overlay" ] && pp=$pp:$overlay
    # LM_CASE is set by the L5/L6 loops; the module gates feed their own inputs
    # and want none of this.
    local lm=; [ -n "${LM_CASE:-}" ] && lm=$(lm_env "$model" "$LM_CASE")
    # Only a real VAR=path is passed to env. `${lm#__LM_MISSING=*}` was wrong
    # here: `#` strips the SHORTEST matching prefix, so a missing-file marker
    # became the bare filename and env ran it as a command (exit 127).
    local lmset=
    case "$lm" in *=*) [ "${lm#__LM_MISSING}" = "$lm" ] && lmset=$lm ;; esac
    ( echo "__LM ${lm:-none}"
      JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONPATH=$pp \
        env ${lmset:+"$lmset"} \
        timeout "${GATE_TIMEOUT:-3600}" $PY "$@" 2>&1
      echo "__GATE_EXIT $?" ) > "$log"
  fi
  local status; status=$(classify "$log")
  local head; head=$(grep -E "$pat" "$log" | tail -1)
  printf '  %-24s %-30s %-8s %s\n' "$tag" "$model" "$status" "${head:0:88}"
  printf '%s\t%s\t%s\t%s\n' "$tag" "$model" "$status" "$head" >> "$SUMMARY"
}

# --- language-model inputs, per (model family, case) ---------------------
# chai-1 folds a DIFFERENT MODEL without its ESM2 embeddings (5.70 A where chai
# reaches 0.642), and ESMFold2 has no MSA at all -- it folds from ESM-C hidden
# states, and each release is trained against its own tower AND its own shim
# (crossing them reads corr 0.026). A fold-level sweep that omits these is not
# measuring these two families; it is measuring nine other models.
#
# dev/oracles/lm_inputs.py writes lm_inputs/<tower>.<case>.npz. This wires them
# in, and the log records which file was used -- or that none was found, because
# a no-LM number must not be mistakable for a real one.
LM_DIR=$ROOT/dev/oracles/lm_inputs

lm_env () {  # lm_env <model> <case> -> prints VAR=path, or a note, or nothing
  local model=$1 case=$2 tower= var=
  case "$model" in
    chai1)                      tower=esm2;      var=ESM_EMB ;;
    esmfold2_lm600m)            tower=esmc_600m; var=ESMC_HIDDEN ;;
    esmfold2_lm300m)            tower=esmc_300m; var=ESMC_HIDDEN ;;
    esmfold2*)                  tower=esmc;      var=ESMC_HIDDEN ;;
    *)                          return 0 ;;
  esac
  local f=$LM_DIR/$tower.$case.npz
  if [ -f "$f" ]; then
    echo "$var=$f"
  else
    echo "__LM_MISSING=$tower.$case.npz"
  fi
}

want () {  # is this level selected?
  local lvl=$1
  case " $LEVELS " in *" all "*|*" $lvl "*) return 0 ;; *) return 1 ;; esac
}

LEVELS=${*:-L0 L1 L1b L1t L1d L2 L3 L4}
echo "levels: $LEVELS"
echo "models: $MODELS"
echo "logs:   $LOGDIR"

# --- L0: does every checkpoint tensor reach a parameter, and back ----------
if want L0; then
  echo "== L0 conversion coverage (both directions -- see converter-coverage-audit)"
  for m in $MODELS; do
    gate L0.audit "$m" 'coverage|missing|unexpected|clean|CLEAN' dev/audit_coverage.py "$m"
  done
fi

# --- L1 / L1b: the trunk, and the MSA stack inside it ---------------------
if want L1; then
  echo "== L1 trunk pairformer"
  for m in $MODELS; do gate L1.trunk "$m" '^  (single|pair) ' dev/oracles/trunk_parity.py "$m"; done
  # ESMFold2 has no vendor MODULE to import -- its implementation is inside
  # `transformers`, which lives in ~/venv_esm. What it has instead is
  # `esmfold2_reference.py`, a complete self-contained JAX reimplementation that
  # the port was built against and that is itself validated on native's dumps.
  # So the reference IS the oracle here, and these two harnesses ARE the gates;
  # they were simply never wired in. Family-scoped on purpose: they import
  # esmfold2's converter and reference directly, so running them for another
  # model would compare the wrong things rather than say it cannot.
  for m in $MODELS; do
    case $m in esmfold2*)
      MODEL=$m gate L1.trunk_ref "$m" 'corr .*relerr' \
        dev/oracles/esmfold2_localise_trunk.py ;;
    esac
  done
fi
if want L1b; then
  # protenix's MSA module (and its trunk) go through prot_parity, which takes
  # its models positionally and KeyErrors on a name it does not know -- so this
  # one gate carries an explicit list rather than probing every model.
  echo "== L1b MSA module (protenix goes through prot_parity)"
  for m in protenix2 protenix1; do
    case " $MODELS " in *" $m "*)
      FP32=1 gate L1b.prot "$m" 'corr' dev/oracles/prot_parity.py "$m" ;;
    esac
  done
  # ESMFold2's native module ships inside `transformers`, which is installed in
  # ~/venv_esm ONLY and must not be installed beside JAX. So its native side
  # runs there first and writes an npz (inputs included, so both sides compare
  # on identical tensors) that msa_parity.py reads with numpy alone.
  for m in $MODELS; do
    case $m in esmfold2|esmfold2_exp|esmfold2_exp_cutoff2025)
      $PY_ESM dev/oracles/esmfold2_msa_dump.py "$m" > "$LOGDIR/L1b.esmdump.$m.log" 2>&1
      $PY_ESM dev/oracles/esmfold2_msa_dump.py "$m" --nonuniform \
        >> "$LOGDIR/L1b.esmdump.$m.log" 2>&1 ;;
    esac
  done
  for m in $MODELS; do gate L1b.msa "$m" 'msa -> pair' dev/oracles/msa_parity.py "$m"; done
  # An all-ones msa mask cannot distinguish two OPM normalisers -- that is how a
  # wrong boltz2 fix passed once. The non-uniform case is not optional polish.
  for m in $MODELS; do
    NONUNIFORM=1 gate L1b.msa_nonuniform "$m" 'msa -> pair' dev/oracles/msa_parity.py "$m"
  done
fi

# --- L1t / L1d: the two heads L1 does not reach --------------------------
if want L1t; then
  echo "== L1 template embedder"
  for m in $MODELS; do gate L1t.template "$m" 'template|corr' dev/oracles/template_parity.py "$m"; done
fi
if want L1d; then
  echo "== L1 distogram head"
  for m in $MODELS; do gate L1d.dgram "$m" 'dgram|corr' dev/oracles/dgram_parity.py "$m"; done
fi

# --- L2: the three parts, which is why the level table reads `~` ----------
if want L2; then
  echo "== L2 token diffusion transformer"
  for m in $MODELS; do gate L2.diffusion "$m" '^  a ' dev/oracles/diffusion_parity.py "$m"; done
  echo "== L2 diffusion conditioning"
  for m in $MODELS; do gate L2.conditioning "$m" 'corr' dev/oracles/conditioning_parity.py "$m"; done
  echo "== L2 atom cross-attention encoder"
  for m in $MODELS; do gate L2.atom_encoder "$m" 'corr' dev/oracles/atom_parity.py "$m"; done
  echo "== L2 atom cross-attention DECODER"
  for m in $MODELS; do
    DECODER=1 gate L2.atom_decoder "$m" 'corr' dev/oracles/atom_parity.py "$m"
  done
fi

# --- L3: one denoise step, the whole diffusion module ---------------------
if want L3; then
  echo "== L3 denoise step"
  for m in $MODELS; do
    DIAG=1 gate L3.denoise "$m" 'per-atom distance' dev/oracles/denoise_parity.py "$m"
  done
  # boltz2 has no standalone-constructible diffusion module; its L3 is by
  # injection from a captured native run (~/boltz2_6mrr/diff_dump.npz).
  gate L3.denoise_inject boltz2 'corr|per-atom' dev/oracles/boltz2_denoise_parity.py
  # ESMFold2's denoise step against the reference -- see the L1 note above.
  for m in $MODELS; do
    case $m in esmfold2*)
      # 'rms diff' rather than 'corr': the LAST corr line in that harness is
      # its conformer-substitution diagnostic, not its headline.
      MODEL=$m gate L3.denoise_ref "$m" 'rms diff' \
        dev/oracles/esmfold2_localise_denoise.py ;;
    esac
  done
fi

# --- L4: the confidence head --------------------------------------------
if want L4; then
  echo "== L4 confidence head"
  for m in $MODELS; do
    gate L4.confidence "$m" '^  (full_pae|plddt)' dev/oracles/confidence_parity.py "$m"
  done
  # chai1's head has no standalone-constructible vendor module, but its verbatim
  # I/O was captured during the port -- nine inputs and three LOGIT tensors -- so
  # its L4 is an injection gate. Logits, not the derived pLDDT/PAE, so no
  # assumption about chai's bin centres enters it.
  gate L4.confidence_inject chai1 '_logits' \
    dev/oracles/chai1_confidence_parity.py
fi

# --- L5 / L6: folds. Much slower, and opt-in for that reason -------------
if want L5; then
  echo "== L5 fold (6MRR)"
  for m in $MODELS; do
    LM_CASE=protein_6mrr gate L5.fold "$m" 'CA-RMSD' \
      dev/oracles/fold_check.py "$m"
  done
fi
if want L6; then
  echo "== L6 modality screens"
  for m in $MODELS; do
    for case in rna_1ehz dna_1lmb complex_1lmb ligand_1stp ptm_5k9p plain_5k9p protein_6mrr; do
      # ptm_5k9p reuses plain_5k9p's LM input: the SEQUENCE is identical (SEP
      # modifies a residue already there), and a language model reads sequences.
      # This only became valid once `_attach_lm_pair` learned to map its rows by
      # RESIDUE -- AF3 atomises a modified residue, so the batch has 85 protein
      # tokens against the LM's 76 rows, and keying on the token count raised
      # `lm_pair is (76, 76) but the batch has 85 tokens`.
      lmcase=$case; [ "$case" = ptm_5k9p ] && lmcase=plain_5k9p
      LM_CASE=$lmcase gate "L6.$case" "$m" 'RMSD|best' \
        dev/oracles/modality_check.py "$m" "$case"
    done
  done
fi

echo
echo "summary: $SUMMARY"
awk -F'\t' 'NR>1 {n[$3]++} END {for (k in n) printf "  %-8s %d\n", k, n[k]}' "$SUMMARY"
echo RUN_ALL_PARITY_DONE
