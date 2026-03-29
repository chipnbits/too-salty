#!/bin/bash
# setup_env.sh — Create a virtual environment for too-salty on the Alliance cluster
# Run this ONCE from the project root:  bash setup_env.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

echo "=== Loading modules ==="
module load python/3.12.4 cuda/12.6

echo "=== Creating virtualenv at ${VENV_DIR} ==="
virtualenv --no-download "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "=== Installing CC-wheeled packages (--no-index) ==="
pip install --no-index --upgrade pip
pip install --no-index \
    torch \
    torchvision \
    wandb \
    scikit-learn \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    pillow

echo "=== Installing packages from PyPI ==="
pip install python-dotenv einops rebasin fastparquet

echo "=== Installing local salty package ==="
# hatchling build backend should work here
pip install -e .

echo "=== Downloading CIFAR-100 dataset to scratch ==="
export DATA_DIR="${PROJECT_DIR}/data"
mkdir -p "${DATA_DIR}"
python scripts/01_build_dataset.py --skip-cifar100c

echo ""
echo "=== Done! ==="
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "The SLURM job script handles module loads + activation automatically."
