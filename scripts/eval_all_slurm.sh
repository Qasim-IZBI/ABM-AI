#!/bin/bash
#SBATCH --job-name=abm_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu        # no GPU needed — model is not reloaded
#SBATCH --array=0-4            # 5 jobs — one per model

# ── Usage ─────────────────────────────────────────────────────────────────────
# Must be run after infer_all_slurm.sh.
#
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
INFER_ROOT="inference"       # where infer_all_slurm.sh wrote its outputs
EVAL_ROOT="eval"
SPLIT="test"                 # sub-directory to evaluate: train | val | test
CONDA_ENV="abm"
NO_PLOTS=1                   # 1 = headless (no matplotlib), 0 = save plots
N_IMAGES=8                   # number of image pairs in grid

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
mkdir -p logs

echo "Host : $(hostname)"
echo "Date : $(date)"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID}"

# ── Model registry ────────────────────────────────────────────────────────────
MODELS=(mlp imreg cgan mmimreg mmcgan)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

# ── Resolve paths ─────────────────────────────────────────────────────────────
INFER_DIR="${INFER_ROOT}/${MODEL}"

if [ ! -d "${INFER_DIR}" ]; then
    echo "[ERROR] Inference directory not found: ${INFER_DIR}"
    echo "        Run infer_all_slurm.sh before eval_all_slurm.sh."
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
echo "Inference  : ${INFER_DIR}"
echo "Data       : ${DATA_PATH}"
echo "Output     : ${OUT}"

# ── Run evaluation ────────────────────────────────────────────────────────────
EXTRA_FLAGS=""
[ "${NO_PLOTS}" = "1" ] && EXTRA_FLAGS="--no_plots"

python evaluation.py        \
    --inference_dir "${INFER_DIR}"  \
    --data          "${DATA_PATH}"  \
    --out           "${OUT}"        \
    --image_size    1024            \
    --pad_images                    \
    --n_images      "${N_IMAGES}"   \
    ${EXTRA_FLAGS}

echo "Finished: $(date)"
