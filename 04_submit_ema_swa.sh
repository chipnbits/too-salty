#!/bin/bash
#SBATCH --job-name=ema-swa
#SBATCH --account=aip-evanesce
#SBATCH --partition=gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# -- Paths -----------------------------------------------------------------
PROJECT_DIR="/scratch/sghyseli/too-salty"
VENV_DIR="${PROJECT_DIR}/venv"

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

# -- Run single trial (SLURM_ARRAY_TASK_ID = run index = 0-11, seed = 42-53) --
echo "=== Running EMA/SWA experiment: run ${SLURM_ARRAY_TASK_ID} ==="
python scripts/04_train_ema_swa.py \
    --config configs/resnet50_baseline.yaml \
    --runs $((SLURM_ARRAY_TASK_ID + 1)) \
    --start-run ${SLURM_ARRAY_TASK_ID} \
    --swa-start 220 \
    --swa-lr 0.03 \
    --ema-decay 0.999
