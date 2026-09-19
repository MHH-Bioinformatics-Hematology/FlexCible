#!/usr/bin/env bash
# One-time environment setup on the HPC login node.
#   bash scripts/hpc/setup_env.sh [env-name]
# Creates a conda env with everything hicmultiome needs (CUDA torch via pip).
set -euo pipefail
ENV="${1:-hicmultiome}"

conda create -y -n "$ENV" python=3.12
conda run -n "$ENV" pip install \
    "torch --index-url https://download.pytorch.org/whl/cu121" || \
    conda run -n "$ENV" pip install torch
conda run -n "$ENV" pip install \
    numpy scipy pyyaml tqdm matplotlib cooler pyBigWig pyfaidx pandas hicrep
echo "env $ENV ready. Install the package with:"
echo "  conda run -n $ENV pip install -e ."
