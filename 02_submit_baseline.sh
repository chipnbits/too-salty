#!/bin/bash
#SBATCH --job-name=resnet50-baseline-10ep
#SBATCH --account=aip-evanesce
#SBATCH --partition=gpubase_l40s_b2,gpubase_l40s_b3,gpubase_l40s_b4,gpubase_l40s_b5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

# Step 2: Train baseline ResNet-50
echo "=== Training baseline ResNet-50 ==="
python scripts/02_train_baseline.py --config configs/resnet50_baseline.yaml
