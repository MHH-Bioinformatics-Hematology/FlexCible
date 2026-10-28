#!/usr/bin/env bash
# Stage 3: resolution ladder -- full multiome GM12878 at 10 kb, 5 kb, 2 kb
# (150-min DDP training each; evaluation degrades gracefully to
# hicrep-package-only where dense matrices would not fit).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/hicdiff/bin/python
TR=~/miniconda3/envs/hicdiff/bin/torchrun
LOG=runs/overnight; mkdir -p "$LOG"

for name in multiome_10kb multiome_5kb multiome_2kb; do
    if [ "$name" = "multiome_2kb" ] && [ ! -s /home/mh-hannover.local/wolffjoa/data/hic-multiome/GM12878_2kb_coarsen.cool ]; then
        echo "SKIP $name (2kb cool not ready)"
        continue
    fi
    echo "=== $name precompute $(date +%H:%M:%S) ==="
    $PY -m flexcible.precompute --config "configs/$name.yaml" \
        > "$LOG/$name.precompute.log" 2>&1 || { echo "PRECOMPUTE FAILED $name"; continue; }
    echo "=== $name train $(date +%H:%M:%S) ==="
    $TR --nproc_per_node=2 -m flexcible.train --config "configs/$name.yaml" \
        > "$LOG/$name.train.log" 2>&1 || { echo "TRAIN FAILED $name"; continue; }
    echo "=== $name predict $(date +%H:%M:%S) ==="
    $PY -m flexcible.predict --config "configs/$name.yaml" --chroms test \
        --out "runs/$name/pred_even.cool" > "$LOG/$name.predict.log" 2>&1 \
        || { echo "PREDICT FAILED $name"; continue; }
    $PY -m flexcible.evaluate --config "configs/$name.yaml" \
        --pred "runs/$name/pred_even.cool" --chroms test \
        --out "runs/$name/scc.json" > "$LOG/$name.evaluate.log" 2>&1 \
        || echo "EVAL FAILED $name"
    $PY -m flexcible.diagnose --config "configs/$name.yaml" --chrom 2 \
        --pred "runs/$name/pred_even.cool" --out-dir "runs/$name" \
        --max-bins 20000 > "$LOG/$name.diagnose.log" 2>&1 || true
done
echo "STAGE3_DONE $(date +%H:%M:%S)"
