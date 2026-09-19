#!/usr/bin/env bash
#SBATCH --job-name=hicmu-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
# Post-training stage: predict held-out chromosomes, SCC, loop/TAD metrics.
#   sbatch scripts/hpc/evaluate.sbatch configs/multiome_25kb.yaml [sample]
set -euo pipefail
CFG="${1:?usage: sbatch evaluate.sbatch <config.yaml> [sample]}"
SAMPLE="${2:-}"
ENV="${HICMU_ENV:-hicmultiome}"
cd "${HICMU_ROOT:-$SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

RUN_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['train']['output_dir'])")
SFLAG=(); TAG=""
if [ -n "$SAMPLE" ]; then SFLAG=(--sample "$SAMPLE"); TAG="_$SAMPLE"; fi

python -m hicmultiome.predict  --config "$CFG" "${SFLAG[@]}" --chroms test \
    --out "$RUN_DIR/pred_even$TAG.cool" \
    --pchic-out "$RUN_DIR/pred_even_capture$TAG.cool" || \
python -m hicmultiome.predict  --config "$CFG" "${SFLAG[@]}" --chroms test \
    --out "$RUN_DIR/pred_even$TAG.cool"
python -m hicmultiome.evaluate --config "$CFG" "${SFLAG[@]}" \
    --pred "$RUN_DIR/pred_even$TAG.cool" --chroms test --out "$RUN_DIR/scc$TAG.json"
python -m hicmultiome.loops    --config "$CFG" "${SFLAG[@]}" \
    --pred "$RUN_DIR/pred_even$TAG.cool" --chroms test \
    --out "$RUN_DIR/loops_tads$TAG.json" || true
python -m hicmultiome.diagnose --config "$CFG" "${SFLAG[@]}" --chrom 2 \
    --pred "$RUN_DIR/pred_even$TAG.cool" --out-dir "$RUN_DIR" --max-bins 20000 || true
