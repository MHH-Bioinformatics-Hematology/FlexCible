#!/usr/bin/env bash
# Local improvement battery: reversal aug / obs-exp target / combined, each a
# 30-min DDP run, evaluated with and without test-time augmentation.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/hicdiff/bin/python
TR=~/miniconda3/envs/hicdiff/bin/torchrun
LOG=runs/overnight; mkdir -p "$LOG"

for name in flip_25kb oe_25kb best_25kb; do
    echo "=== $name $(date +%H:%M:%S) ==="
    if [ ! -s "cache/$name/manifest.json" ] && [ "$name" != "flip_25kb" ]; then
        $PY -m hicmultiome.precompute --config "configs/$name.yaml" \
            > "$LOG/$name.precompute.log" 2>&1 || { echo "PRECOMPUTE FAILED $name"; continue; }
    fi
    $TR --nproc_per_node=2 -m hicmultiome.train --config "configs/$name.yaml" \
        > "$LOG/$name.train.log" 2>&1 || { echo "TRAIN FAILED $name"; continue; }
    for mode in plain tta; do
        FLAG=(); [ $mode = tta ] && FLAG=(--tta)
        $PY -m hicmultiome.predict --config "configs/$name.yaml" --chroms test \
            "${FLAG[@]}" --out "runs/$name/pred_even_$mode.cool" \
            > "$LOG/$name.predict.$mode.log" 2>&1 || { echo "PREDICT FAILED $name $mode"; continue; }
        $PY -m hicmultiome.evaluate --config "configs/$name.yaml" \
            --pred "runs/$name/pred_even_$mode.cool" --chroms test \
            --out "runs/$name/scc_$mode.json" \
            > "$LOG/$name.evaluate.$mode.log" 2>&1 || echo "EVAL FAILED $name $mode"
    done
    $PY -m hicmultiome.loops --config "configs/$name.yaml" \
        --pred "runs/$name/pred_even_tta.cool" --chroms test \
        --out "runs/$name/loops_tads.json" > "$LOG/$name.loops.log" 2>&1 || true
done
echo "IMPROVE_DONE $(date +%H:%M:%S)"
