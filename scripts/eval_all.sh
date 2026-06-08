#!/bin/bash
# Run evaluation for all 5 ABM models sequentially on a single machine.
# Uses the latest checkpoint found in each model's results directory.
#
# Usage
# -----
#   bash scripts/eval_all.sh
#   DATA=/my/processed RESULTS_ROOT=/my/results bash scripts/eval_all.sh
#   SPLIT=val MODELS="mlp mmimreg" bash scripts/eval_all.sh
#   NO_PLOTS=1 bash scripts/eval_all.sh     # headless / no matplotlib
#
# Outputs are written to:
#   $EVAL_ROOT/<model>/metrics.json
#   $EVAL_ROOT/<model>/predictions.csv   (numerical models)
#   $EVAL_ROOT/<model>/scatter.png       (numerical models)
#   $EVAL_ROOT/<model>/image_grid.png    (image models)
#   $EVAL_ROOT/<model>/eval.log

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATA="${DATA:-/path/to/processed}"    # directory produced by pipeline.py
RESULTS_ROOT="${RESULTS_ROOT:-results}"
EVAL_ROOT="${EVAL_ROOT:-eval}"
SPLIT="${SPLIT:-test}"                # sub-directory to evaluate: train | val | test
CONDA_ENV="${CONDA_ENV:-abm}"
MODELS="${MODELS:-mlp imreg cgan mmimreg mmcgan}"
NO_PLOTS="${NO_PLOTS:-0}"             # set to 1 for headless servers
N_IMAGES="${N_IMAGES:-8}"            # number of image pairs in grid

# ── Activate environment ──────────────────────────────────────────────────────
if [ -n "${CONDA_ENV}" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || echo "[WARN] Could not activate '${CONDA_ENV}'"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# ── Resolve data split path ───────────────────────────────────────────────────
if [ -d "${DATA}/${SPLIT}" ]; then
    DATA_PATH="${DATA}/${SPLIT}"
else
    DATA_PATH="${DATA}"
fi

echo "========================================================"
echo "  ABM — evaluate all models"
echo "  DATA     : ${DATA_PATH}"
echo "  RESULTS  : ${RESULTS_ROOT}"
echo "  OUTPUT   : ${EVAL_ROOT}"
echo "  MODELS   : ${MODELS}"
echo "  Host     : $(hostname)"
echo "  Date     : $(date)"
echo "========================================================"

# ── Helper: find latest checkpoint ───────────────────────────────────────────
latest_ckpt() {
    local ckpt_dir="${RESULTS_ROOT}/${1}/checkpoints"
    [ -d "${ckpt_dir}" ] || { echo ""; return; }
    ls -1 "${ckpt_dir}"/*.pt 2>/dev/null | sort | tail -1
}

# ── Helper: run evaluation for one model ──────────────────────────────────────
run_eval() {
    local model="$1"
    local ckpt
    ckpt="$(latest_ckpt "${model}")"

    if [ -z "${ckpt}" ]; then
        echo "[SKIP] ${model} — no checkpoint found in ${RESULTS_ROOT}/${model}/checkpoints/"
        return
    fi

    local out="${EVAL_ROOT}/${model}"
    mkdir -p "${out}"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Evaluate : ${model}"
    echo "  Checkpoint: ${ckpt}"
    echo "  Output    : ${out}"
    echo "  Start     : $(date)"
    echo "──────────────────────────────────────────────────────"

    local extra_flags=""
    [ "${NO_PLOTS}" = "1" ] && extra_flags="--no_plots"

    python evaluation.py \
        --ckpt       "${ckpt}"      \
        --data       "${DATA_PATH}" \
        --out        "${out}"       \
        --batch_size 64             \
        --image_size 256            \
        --n_images   "${N_IMAGES}"  \
        ${extra_flags}              \
        2>&1 | tee "${out}/eval.log"

    echo "  Done: $(date)"
}

# ── Run ───────────────────────────────────────────────────────────────────────
for model in ${MODELS}; do
    run_eval "${model}"
done

echo ""
echo "========================================================"
echo "  Evaluation finished: $(date)"
echo "========================================================"
