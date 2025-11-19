# too-salty

## Description

A Model Souping project focused on alignment techniques. Work completed as part of CPSC 532X at the University of British Columbia.

**Background**
- [Model Soups](https://arxiv.org/abs/2203.05482) is a technique for combining multiple neural network models by averaging their weights to improve performance.
- [Model Similarity and Alignment](https://arxiv.org/abs/1905.00414) explores the relationship between model similarity and alignment in large language models.
- [Model Permutations](https://arxiv.org/abs/2209.04836) 

## Project Setup

This project uses **uv** for Python environments and dependency management.

### Installation

1. Install **uv**:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
iwr -useb https://astral.sh/uv/install.ps1 | iex
```

2. Clone repository
```bash
git clone https://github.com/chipnbits/too-salty.git
cd too-salty
```

3. Create environment and install dependencies:
```bash
uv sync
```
This creates the `.venv` and installs all dependencies listed in `pyproject.toml` and `uv.lock`.

### Development

```bash
uv add package-name # To add new dependencies
uv remove package-name # To remove dependencies
uv sync # To update the environment after modifying dependencies
```

### Running the Project
```bash
uv run python main.py

# For VSCode integration, select the Python interpreter from the `.venv` folder.
```
