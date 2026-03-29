#!/bin/bash
#SBATCH --job-name=eval-soups
#SBATCH --account=aip-evanesce
#SBATCH --partition=gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

NUM_WORKERS=4

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

# -- Copy datasets to node-local NVMe -------------------------------------
cd "${PROJECT_DIR}"
mkdir -p logs analysis

echo "=== Copying data to ${DATA_DIR} ==="
cp -r "${PROJECT_DIR}/data" "${DATA_DIR}"

# -- Run soup evaluation ---------------------------------------------------
echo "=== Worker ${SLURM_ARRAY_TASK_ID}/${NUM_WORKERS}: Evaluating pairwise model soups ==="
python scripts/05_evaluate_soups.py \
    --variants-dir "${PROJECT_DIR}/models/cifar100-resnet50" \
    --output-dir "${PROJECT_DIR}/analysis" \
    --batch-size 1024 \
    --seed 42 \
    --severities 3 \
    --num-workers ${NUM_WORKERS} \
    --worker-id ${SLURM_ARRAY_TASK_ID}
