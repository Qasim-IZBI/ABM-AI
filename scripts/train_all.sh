#!/bin/bash
# Train all 5 ABM models sequentially on a single machine.
#
# Usage
# -----
#   bash scripts/train_all.sh                        # use defaults below
#   DATA=/my/data OUT_ROOT=/my/results bash scripts/train_all.sh
#   MODELS="mlp imreg" bash scripts/train_all.sh     # subset of models
#
# Each model writes checkpoints to:
#   $OUT_ROOT/<model>/checkpoints/epoch_NNNN.pt
# and stdout/stderr to:
#   $OUT_ROOT/<model>/train.log

set -euo pipefail

# ── Configuration (override with env vars) ────────────────────────────────────
DATA="${DATA:-/path/to/processed}"          # directory produced by pipeline.py
OUT_ROOT="${OUT_ROOT:-results}"
CONDA_ENV="${CONDA_ENV:-abm}"               # set to "" to skip conda activation
MODELS="${MODELS:-mlp imreg cgan mmimreg mmcgan}"

# ── Activate environment ──────────────────────────────────────────────────────
if [ -n "${CONDA_ENV}" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || echo "[WARN] Could not activate conda env '${CONDA_ENV}' — continuing."
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "========================================================"
echo "  ABM — train all models"
echo "  DATA     : ${DATA}"
echo "  OUT_ROOT : ${OUT_ROOT}"
echo "  MODELS   : ${MODELS}"
echo "  Host     : $(hostname)"
echo "  Date     : $(date)"
echo "========================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
run_model() {
    local model="$1"; shift
    local out="${OUT_ROOT}/${model}"
    mkdir -p "${out}"
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Training: ${model}"
    echo "  Output  : ${out}"
    echo "  Start   : $(date)"
    echo "──────────────────────────────────────────────────────"
    python train.py --data "${DATA}" --model "${model}" --out "${out}" "$@" \
        2>&1 | tee "${out}/train.log"
    echo "  Done    : $(date)"
}

# ── Per-model hyperparameters ─────────────────────────────────────────────────
for model in ${MODELS}; do
    case "${model}" in

        mlp)
            run_model mlp \
                --epochs      200    \
                --batch_size  64     \
                --lr          1e-3   \
                --hidden_dims 128 64 \
                --dropout     0.2    \
                --activation  relu   \
                --save_every  20     \
                --log_every   10
            ;;

        imreg)
            run_model imreg \
                --epochs      300    \
                --batch_size  32     \
                --lr          2e-4   \
                --image_size  256    \
                --ngf         32     \
                --lambda_l1   1.0    \
                --lambda_mse  1.0    \
                --save_every  25     \
                --log_every   10
            ;;

        cgan)
            run_model cgan \
                --epochs      300    \
                --batch_size  16     \
                --lr          2e-4   \
                --image_size  256    \
                --ngf         32     \
                --ndf         32     \
                --noise_dim   64     \
                --lambda_l1   10.0   \
                --save_every  25     \
                --log_every   10
            ;;

        mmimreg)
            run_model mmimreg \
                --epochs      300    \
                --batch_size  32     \
                --lr          2e-4   \
                --image_size  256    \
                --ngf         32     \
                --hidden_dims 64 64  \
                --lambda_l1   1.0    \
                --lambda_mse  1.0    \
                --lambda_reg  1.0    \
                --save_every  25     \
                --log_every   10
            ;;

        mmcgan)
            run_model mmcgan \
                --epochs      400    \
                --batch_size  16     \
                --lr          2e-4   \
                --image_size  256    \
                --ngf         32     \
                --ndf         32     \
                --noise_dim   64     \
                --hidden_dims 64 64  \
                --lambda_l1   10.0   \
                --lambda_reg  1.0    \
                --save_every  25     \
                --log_every   10
            ;;

        *)
            echo "[ERROR] Unknown model: ${model}"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================================"
echo "  All models finished: $(date)"
echo "========================================================"
