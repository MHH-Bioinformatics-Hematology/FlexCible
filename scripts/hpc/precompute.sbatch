#!/usr/bin/env bash
#SBATCH --job-name=hicmu-pre
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
# CPU stage: window cache + motif/sequence scans. No GPU needed.
#   sbatch scripts/hpc/precompute.sbatch configs/multiome_25kb.yaml
# Env overrides: HICMU_ENV (conda env), HICMU_ROOT (repo dir).
set -euo pipefail
CFG="${1:?usage: sbatch precompute.sbatch <config.yaml>}"
ENV="${HICMU_ENV:-hicmultiome}"
cd "${HICMU_ROOT:-$SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
python -m hicmultiome.precompute --config "$CFG"
