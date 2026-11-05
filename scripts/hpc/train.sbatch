#!/usr/bin/env bash
#SBATCH --job-name=hicmu-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
# GPU stage: DDP training over however many GPUs the job got. Reads the cache
# written by precompute.sbatch (same repo/output folder).
#   sbatch scripts/hpc/train.sbatch configs/multiome_25kb.yaml
# For a full-length HPC run, override the local time-box, e.g.:
#   HICMU_MAX_MINUTES=1200 sbatch scripts/hpc/train.sbatch configs/multiome_25kb.yaml
set -euo pipefail
CFG="${1:?usage: sbatch train.sbatch <config.yaml>}"
ENV="${FLEXCIBLE_ENV:-flexcible}"
cd "${HICMU_ROOT:-$SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

RUN_CFG="$CFG"
if [ -n "${HICMU_MAX_MINUTES:-}" ]; then
    # write a derived config with the requested training budget
    RUN_CFG="${CFG%.yaml}.hpc$$.yaml"
    python - "$CFG" "$RUN_CFG" "$HICMU_MAX_MINUTES" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["train"]["max_minutes"] = float(sys.argv[3])
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
EOF
    trap 'rm -f "$RUN_CFG"' EXIT
fi

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "training on $NGPU GPU(s), config $RUN_CFG"
if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" -m flexcible.train --config "$RUN_CFG"
else
    python -m flexcible.train --config "$RUN_CFG"
fi
