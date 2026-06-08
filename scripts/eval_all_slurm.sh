#!/bin/bash
#SBATCH --job-name=abm_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4        # 5 jobs — one per model

# ── Usage ─────────────────────────────────────────────────────────────────────
# Submit all 5 models in parallel:
#   sbatch scripts/eval_all_slurm.sh
#
# Submit a single model by index:
#   sbatch --array=0 scripts/eval_all_slurm.sh   # mlp only
#
# Model index mapping:
#   0 → mlp
#   1 → imreg
#   2 → cgan
#   3 → mmimreg
#   4 → mmcgan

set -euo pipefail

# ── Paths (edit before submitting) ────────────────────────────────────────────
DATA="/path/to/processed"
RESULTS_ROOT="results"
EVAL_ROOT="eval"
SPLIT="test"             # sub-directory to evaluate: train | val | test
CONDA_ENV="abm"
NO_PLOTS=1               # 1 = headless (no matplotlib), 0 = save plots
N_IMAGES=8               # number of image pairs in grid

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

echo "Host : $(hostname)"
echo "Date : $(date)"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID}"
nvidia-smi 2>/dev/null || true

# ── Model registry ────────────────────────────────────────────────────────────
MODELS=(mlp imreg cgan mmimreg mmcgan)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

# ── Resolve paths ─────────────────────────────────────────────────────────────
CKPT_DIR="${RESULTS_ROOT}/${MODEL}/checkpoints"
CKPT="$(ls -1 "${CKPT_DIR}"/*.pt 2>/dev/null | sort | tail -1)"

if [ -z "${CKPT}" ]; then
    echo "[ERROR] No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

if [ -d "${DATA}/${SPLIT}" ]; then
    DATA_PATH="${DATA}/${SPLIT}"
else
    DATA_PATH="${DATA}"
fi

OUT="${EVAL_ROOT}/${MODEL}"
mkdir -p "${OUT}"

echo "Model      : ${MODEL}"
echo "Checkpoint : ${CKPT}"
echo "Data       : ${DATA_PATH}"
echo "Output     : ${OUT}"

# ── Run evaluation ────────────────────────────────────────────────────────────
EXTRA_FLAGS=""
[ "${NO_PLOTS}" = "1" ] && EXTRA_FLAGS="--no_plots"

python evaluation.py    \
    --ckpt       "${CKPT}"      \
    --data       "${DATA_PATH}" \
    --out        "${OUT}"       \
    --batch_size 64             \
    --image_size 256            \
    --n_images   "${N_IMAGES}"  \
    ${EXTRA_FLAGS}

echo "Finished: $(date)"
