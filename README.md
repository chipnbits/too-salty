# too-salty

## Description

A Model Souping project focused on alignment techniques.

**Background**
- [Model Soups](https://arxiv.org/abs/2203.05482) is a technique for combining multiple neural network models by averaging their weights to improve performance.
- [Model Similarity and Alignment](https://arxiv.org/abs/1905.00414) explores the relationship between model similarity and alignment in large language models.
- [Model Permutations](https://arxiv.org/abs/2209.04836) 

## Project Setup

This project uses **uv** for Python environments and dependency management.

### Installation

#### 1. Install **uv**:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
iwr -useb https://astral.sh/uv/install.ps1 | iex
```

#### 2. Clone repository
```bash
git clone https://github.com/chipnbits/too-salty.git
cd too-salty
```

#### 3. Create environment and install dependencies:

`uv` does not interact very well with the intricacies of PyTorch environment handling. To sync, an extra flag for if CPU or CUDA versions are desired is needed. The extras are defined in `pyproject.toml` and are for `cpu` and `cu130` (CUDA 13.0). Modify the command below as needed.

```bash
uv sync --extra cpu
# or for CUDA 13.0
# uv sync --extra cu130
```
This creates the `.venv` and installs all dependencies listed in `pyproject.toml` and `uv.lock`.

#### 4. Setup a `.env` file for local environment variables:
```bash
cp .env.example .env
```
Modify `.env` as needed if you want to change any default settings.

#### 5. Download the datasets (CIFAR-100 and CIFAR-100-C):
```bash
uv run python build_dataset.py
```
This will download:
- CIFAR-100 train and test sets (~170 MB)
- CIFAR-100-C with all 19 corruption types (~2.7 GB)

The datasets will be saved to the directory specified in `DATA_DIR` (default: `./data`) in the `.env` file.

Optional flags:
```bash
uv run python build_dataset.py --skip-cifar100c  # Skip CIFAR-100-C download
uv run python build_dataset.py --skip-cifar100   # Skip CIFAR-100 download
```

#### 6. Setup precommit hooks for code quality:
```bash
uv run pre-commit install
```
Precommit hooks are added for `black` formatter and `isort` import ordering. They will run automatically on `git commit`.

### Development

```bash
uv add package-name # To add new dependencies
uv remove package-name # To remove dependencies
uv sync # To update the environment after modifying dependencies
```

### Using the Package

The project is installed as the `salty` package. You can import modules from the `src/salty` directory from anywhere within the `uv` environment.

```python
import salty.datasets
import salty.models
import salty.resnet

# Or import specific items
from salty.datasets import CIFAR100Dataset
from salty.models import ResNet
```

### Running the Project
The project components are meant to be run as scripts from the root directory.
For VSCode integration, select the Python interpreter from the `.venv` folder.

```bash
# 00 - Check CUDA availability
uv run python scripts/00_check_cuda.py

# 01 - Download CIFAR-100 and CIFAR-100-C datasets
uv run python scripts/01_build_dataset.py

# 02 - Train baseline ResNet-50 on CIFAR-100 (300 epochs, checkpoints every 10 epochs)
uv run python scripts/02_train_baseline.py --config configs/resnet50_baseline.yaml

# 03 - Finetune 4 variant branches from each baseline checkpoint (epochs 50-300)
#      From all checkpoints in the baseline directory:
uv run python scripts/03_finetune_branches.py \
    --checkpoint-dir models/cifar100-resnet50/baseline-resnet50/ \
    --variants-config configs/resnet50_finetune.yaml \
    --seed 42

# 04 - Train EMA and SWA experiments (12 runs with different seeds)
uv run python scripts/04_train_ema_swa.py \
    --config configs/resnet50_baseline.yaml \
    --runs 12 \
    --swa-start 220 \
    --swa-lr 0.03 \
    --ema-decay 0.999

# 05 - Evaluate model soups (pairwise weight averaging)
uv run python scripts/05_evaluate_soups.py

# 06 - Evaluate EMA/SWA models
uv run python scripts/06_evaluate_ema_swa.py
```
