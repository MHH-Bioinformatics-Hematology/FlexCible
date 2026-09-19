#!/usr/bin/env bash
# Submit the full pipeline for one config with Slurm dependency chaining:
#   precompute (CPU)  ->  train (GPU)  ->  evaluate (GPU)
#
#   bash scripts/hpc/submit_pipeline.sh configs/multiome_25kb.yaml
#   HICMU_MAX_MINUTES=1200 bash scripts/hpc/submit_pipeline.sh configs/multiome_2kb.yaml
#
# Partitions/resources live in the .sbatch headers; override per cluster with
# sbatch's own flags, e.g.:
#   SBATCH_EXTRA="--partition=gpu_a100" bash scripts/hpc/submit_pipeline.sh cfg.yaml
set -euo pipefail
CFG="${1:?usage: submit_pipeline.sh <config.yaml> [sample ...]}"
shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"
EXTRA=(${SBATCH_EXTRA:-})

pre=$(sbatch --parsable "${EXTRA[@]}" "$HERE/precompute.sbatch" "$CFG")
echo "precompute: $pre"
tr=$(sbatch --parsable --dependency=afterok:"$pre" "${EXTRA[@]}" \
     --export=ALL "$HERE/train.sbatch" "$CFG")
echo "train:      $tr (after $pre)"
if [ "$#" -gt 0 ]; then
    for smp in "$@"; do
        ev=$(sbatch --parsable --dependency=afterok:"$tr" "${EXTRA[@]}" \
             "$HERE/evaluate.sbatch" "$CFG" "$smp")
        echo "evaluate:   $ev sample=$smp (after $tr)"
    done
else
    ev=$(sbatch --parsable --dependency=afterok:"$tr" "${EXTRA[@]}" \
         "$HERE/evaluate.sbatch" "$CFG")
    echo "evaluate:   $ev (after $tr)"
fi
