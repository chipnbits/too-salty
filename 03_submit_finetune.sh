#!/bin/bash
#SBATCH --job-name=finetune-branches
#SBATCH --account=aip-evanesce
#SBATCH --partition=gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --array=0-25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune-%A_%a.out
#SBATCH --error=logs/finetune-%A_%a.err

set -euo pipefail

# -- Paths -----------------------------------------------------------------
PROJECT_DIR="/scratch/sghyseli/too-salty"
VENV_DIR="${PROJECT_DIR}/venv"
BASELINE_DIR="${PROJECT_DIR}/models/cifar100-resnet50/baseline-resnet50"

# -- Modules ----------------------------------------------------------------
module load python/3.12.4 cuda/12.6

# -- Virtual environment ----------------------------------------------------
source "${VENV_DIR}/bin/activate"

# -- Make local salty package importable ------------------------------------
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# -- Use node-local storage for data (fast NVMe) --------------------------
export DATA_DIR="${SLURM_TMPDIR}/data"
export MODEL_DIR="${PROJECT_DIR}/models"

# -- Copy dataset from scratch to node-local NVMe --------------------------
cd "${PROJECT_DIR}"
mkdir -p logs models

echo "=== Copying CIFAR-100 to ${DATA_DIR} ==="
cp -r "${PROJECT_DIR}/data" "${DATA_DIR}"

# -- Map array task ID to checkpoint epoch (50, 60, ..., 300) ---------------
EPOCH=$(( 50 + SLURM_ARRAY_TASK_ID * 10 ))
CHECKPOINT="${BASELINE_DIR}/baseline-resnet50-epoch_${EPOCH}.pt"

echo "=== Finetuning 4 branches from epoch ${EPOCH} checkpoint ==="
python scripts/03_finetune_branches.py \
    --checkpoint "${CHECKPOINT}" \
    --variants-config configs/resnet50_finetune.yaml \
    --seed 42
