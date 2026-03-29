#!/bin/bash
#SBATCH --job-name=ternary-eval
#SBATCH --account=aip-evanesce
#SBATCH --partition=gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# -- Triplets (one per array task) -----------------------------------------
TRIPLETS=(
    "240_3,170_2,250_4"
    "250_4,230_4,240_3"
    "240_3,190_2,250_4"
    "240_3,210_2,250_4"
    "240_3,250_4,200_2"
    "270_4,220_2,250_4"
    "260_4,180_4,280_1"
    "240_3,250_4,160_4"
)

TRIPLET="${TRIPLETS[$SLURM_ARRAY_TASK_ID]}"

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

# -- Copy datasets to node-local NVMe -------------------------------------
cd "${PROJECT_DIR}"
mkdir -p logs analysis/ternary

echo "=== Copying data to ${DATA_DIR} ==="
cp -r "${PROJECT_DIR}/data" "${DATA_DIR}"

# -- Run ternary simplex evaluation ----------------------------------------
echo "=== Task ${SLURM_ARRAY_TASK_ID}: Evaluating triplet ${TRIPLET} ==="
python scripts/ternary_simplex_eval.py \
    --triplet "${TRIPLET}" \
    --resolution 20 \
    --variants-dir "${PROJECT_DIR}/models/cifar100-resnet50" \
    --output-dir "${PROJECT_DIR}/analysis/ternary" \
    --batch-size 1024 \
    --num-workers 4
