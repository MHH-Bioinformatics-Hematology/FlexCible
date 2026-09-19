#!/usr/bin/env bash
# Stage 1: feature/protein ablations at 25 kb (6 x 30-min DDP runs) +
# leave-one-protein-out zeroing on the existing chip_only checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/hicdiff/bin/python
TR=~/miniconda3/envs/hicdiff/bin/torchrun
LOG=runs/overnight; mkdir -p "$LOG"

for name in abl_ctcf abl_h3k27ac abl_ctcf_h3k27ac abl_all14_seq abl_all14_motif abl_all14_capture; do
    echo "=== $name $(date +%H:%M:%S) ==="
    $PY -m hicmultiome.precompute --config "configs/$name.yaml" \
        > "$LOG/$name.precompute.log" 2>&1 || { echo "PRECOMPUTE FAILED $name"; continue; }
    $TR --nproc_per_node=2 -m hicmultiome.train --config "configs/$name.yaml" \
        > "$LOG/$name.train.log" 2>&1 || { echo "TRAIN FAILED $name"; continue; }
    $PY -m hicmultiome.predict --config "configs/$name.yaml" --chroms test \
        --out "runs/$name/pred_even.cool" > "$LOG/$name.predict.log" 2>&1 \
        || { echo "PREDICT FAILED $name"; continue; }
    $PY -m hicmultiome.evaluate --config "configs/$name.yaml" \
        --pred "runs/$name/pred_even.cool" --chroms test \
        --out "runs/$name/scc.json" > "$LOG/$name.evaluate.log" 2>&1 \
        || echo "EVAL FAILED $name"
done

# capture-ablation loop metrics (its checkpoint predicts a capture channel)
$PY -m hicmultiome.predict --config configs/abl_all14_capture.yaml --chroms test \
    --out runs/abl_all14_capture/pred_even2.cool \
    --pchic-out runs/abl_all14_capture/pred_even_capture.cool \
    > "$LOG/abl_all14_capture.predict2.log" 2>&1 || true
$PY -m hicmultiome.loops --config configs/abl_all14_capture.yaml \
    --pred runs/abl_all14_capture/pred_even.cool \
    --pred-capture runs/abl_all14_capture/pred_even_capture.cool \
    --chroms test --out runs/abl_all14_capture/loops_tads.json \
    > "$LOG/abl_all14_capture.loops.log" 2>&1 || true

# leave-one-protein-out on the trained chip-only model (chr2 only, fast)
echo "=== leave-one-out $(date +%H:%M:%S) ==="
$PY scripts/leave_one_out.py > "$LOG/leave_one_out.log" 2>&1 || echo "LOO FAILED"
echo "STAGE1_DONE $(date +%H:%M:%S)"
