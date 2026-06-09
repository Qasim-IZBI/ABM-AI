#!/bin/bash
# Run inference for all 5 ABM models sequentially on a single machine.
# Uses the latest checkpoint found in each model's results directory.
#
# Usage
# -----
#   bash scripts/infer_all.sh
#   DATA=/my/processed RESULTS_ROOT=/my/results bash scripts/infer_all.sh
#   SPLIT=val MODELS="mlp mmimreg" bash scripts/infer_all.sh
#   SAVE_IMAGES=1 bash scripts/infer_all.sh          # also write per-sample PNGs
#
# Outputs are written to:
#   $INFER_ROOT/<model>/predictions.npy   (numerical models)
#   $INFER_ROOT/<model>/images.npy        (image models)
#   $INFER_ROOT/<model>/images/           (if SAVE_IMAGES=1)
#   $INFER_ROOT/<model>/infer.log

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATA="${DATA:-/path/to/processed}"    # directory produced by pipeline.py
RESULTS_ROOT="${RESULTS_ROOT:-results}"
INFER_ROOT="${INFER_ROOT:-inference}"
SPLIT="${SPLIT:-test}"                # sub-directory to run on: train | val | test
CONDA_ENV="${CONDA_ENV:-abm}"
MODELS="${MODELS:-mlp imreg cgan mmimreg mmcgan}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"       # set to 1 to write individual PNGs

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
    DATA_PATH="${DATA}"     # flat layout — no pre-split subdirectory
fi

echo "========================================================"
echo "  ABM — inference all models"
echo "  DATA     : ${DATA_PATH}"
echo "  RESULTS  : ${RESULTS_ROOT}"
echo "  OUTPUT   : ${INFER_ROOT}"
echo "  MODELS   : ${MODELS}"
echo "  Host     : $(hostname)"
echo "  Date     : $(date)"
echo "========================================================"

# ── Helper: find latest checkpoint for a model ────────────────────────────────
latest_ckpt() {
    local ckpt_dir="${RESULTS_ROOT}/${1}/checkpoints"
    if [ ! -d "${ckpt_dir}" ]; then
        echo ""
        return
    fi
    ls -1 "${ckpt_dir}"/*.pt 2>/dev/null | sort | tail -1
}

# ── Helper: run inference for one model ───────────────────────────────────────
run_infer() {
    local model="$1"
    local ckpt
    ckpt="$(latest_ckpt "${model}")"

    if [ -z "${ckpt}" ]; then
        echo "[SKIP] ${model} — no checkpoint found in ${RESULTS_ROOT}/${model}/checkpoints/"
        return
    fi

    local out="${INFER_ROOT}/${model}"
    mkdir -p "${out}"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Inference: ${model}"
    echo "  Checkpoint: ${ckpt}"
    echo "  Output    : ${out}"
    echo "  Start     : $(date)"
    echo "──────────────────────────────────────────────────────"

    local extra_flags=""
    [ "${SAVE_IMAGES}" = "1" ] && extra_flags="--save_images"

    # Batch size and image args vary by model to avoid GPU OOM at 1024×1024
    local batch_args
    case "${model}" in
        mlp)
            batch_args="--batch_size 128"
            ;;
        imreg|mmimreg)
            batch_args="--batch_size 16 --image_size 1024 --pad_images"
            ;;
        cgan|mmcgan)
            batch_args="--batch_size 8 --image_size 1024 --pad_images"
            ;;
    esac

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python inference.py \
        --ckpt       "${ckpt}"      \
        --data       "${DATA_PATH}" \
        --out        "${out}"       \
        ${batch_args}               \
        ${extra_flags}              \
        2>&1 | tee "${out}/infer.log"

    echo "  Done: $(date)"
}

# ── Run ───────────────────────────────────────────────────────────────────────
for model in ${MODELS}; do
    run_infer "${model}"
done

echo ""
echo "========================================================"
echo "  Inference finished: $(date)"
echo "========================================================"
