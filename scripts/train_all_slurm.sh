#!/bin/bash
#SBATCH --job-name=abm_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4        # 5 jobs — one per model

# ── Usage ─────────────────────────────────────────────────────────────────────
# Submit all 5 models in parallel:
#   sbatch scripts/train_all_slurm.sh
#
# Submit a single model by index:
#   sbatch --array=0 scripts/train_all_slurm.sh   # mlp only
#   sbatch --array=2 scripts/train_all_slurm.sh   # cgan only
#
# Model index mapping:
#   0 → mlp
#   1 → imreg
#   2 → cgan
#   3 → mmimreg
#   4 → mmcgan

set -euo pipefail

# ── Paths (edit before submitting) ───────────────────────────────────────────
DATA="/path/to/processed"          # directory produced by pipeline.py
OUT_ROOT="results"
CONDA_ENV="abm"

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Anaconda3 2>/dev/null || true

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# SLURM copies the script to its spool dir, so $0 resolves there, not here.
# SLURM_SUBMIT_DIR is set to the directory where sbatch was called from.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
mkdir -p logs

echo "Host : $(hostname)"
echo "Date : $(date)"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID}"
nvidia-smi 2>/dev/null || echo "[INFO] No GPU detected"

# ── Model registry ────────────────────────────────────────────────────────────
MODELS=(mlp imreg cgan mmimreg mmcgan)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
OUT="${OUT_ROOT}/${MODEL}"
mkdir -p "${OUT}"

echo "Model : ${MODEL}"
echo "Output: ${OUT}"

# ── Per-model hyperparameters ─────────────────────────────────────────────────
case "${MODEL}" in

    mlp)
        python train.py                 \
            --data        "${DATA}"     \
            --model       mlp           \
            --out         "${OUT}"      \
            --epochs      200           \
            --batch_size  64            \
            --lr          1e-3          \
            --hidden_dims 128 64        \
            --dropout     0.2           \
            --activation  relu          \
            --save_every  20            \
            --log_every   10
        ;;

    imreg)
        python train.py                 \
            --data        "${DATA}"     \
            --model       imreg         \
            --out         "${OUT}"      \
            --epochs      300           \
            --batch_size  8             \
            --lr          2e-4          \
            --image_size  1024          \
            --pad_images            \
            --ngf         32            \
            --lambda_l1   1.0           \
            --lambda_mse  1.0           \
            --save_every  25            \
            --log_every   10
        ;;

    cgan)
        python train.py                 \
            --data        "${DATA}"     \
            --model       cgan          \
            --out         "${OUT}"      \
            --epochs      300           \
            --batch_size  4             \
            --lr          2e-4          \
            --image_size  1024          \
            --pad_images            \
            --ngf         32            \
            --ndf         32            \
            --noise_dim   64            \
            --lambda_l1   10.0          \
            --save_every  25            \
            --log_every   10
        ;;

    mmimreg)
        python train.py                 \
            --data        "${DATA}"     \
            --model       mmimreg       \
            --out         "${OUT}"      \
            --epochs      300           \
            --batch_size  8             \
            --lr          2e-4          \
            --image_size  1024          \
            --pad_images            \
            --ngf         32            \
            --hidden_dims 64 64         \
            --lambda_l1   1.0           \
            --lambda_mse  1.0           \
            --lambda_reg  1.0           \
            --save_every  25            \
            --log_every   10
        ;;

    mmcgan)
        python train.py                 \
            --data        "${DATA}"     \
            --model       mmcgan        \
            --out         "${OUT}"      \
            --epochs      400           \
            --batch_size  4             \
            --lr          2e-4          \
            --image_size  1024          \
            --pad_images            \
            --ngf         32            \
            --ndf         32            \
            --noise_dim   64            \
            --hidden_dims 64 64         \
            --lambda_l1   10.0          \
            --lambda_reg  1.0           \
            --save_every  25            \
            --log_every   10
        ;;

    *)
        echo "[ERROR] Unknown model: ${MODEL}"
        exit 1
        ;;
esac

echo "Finished: $(date)"
