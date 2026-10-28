#!/usr/bin/env bash
# Stage: methylation value + minimal-input search.
#   meth_25kb      14 assays + DNA methylation
#   methonly_25kb  methylation as the ONLY input (no other assay, no sequence)
#   methseq_25kb   methylation + free DNA features
#   dropout_25kb   track-dropout model -> greedy minimal assay set (hic)
# Waits for any running stage_improve.sh so the time-boxed runs never share GPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/hicdiff/bin/python
TR=~/miniconda3/envs/hicdiff/bin/torchrun
LOG=runs/overnight; mkdir -p "$LOG"

while pgrep -f "stage_improve.sh" > /dev/null; do sleep 60; done
echo "GPUs free, starting $(date +%H:%M:%S)"

run_one() {
    local name=$1 mins=${2:-}
    echo "=== $name $(date +%H:%M:%S) ==="
    [ -s "cache/$name/manifest.json" ] || \
        $PY -m flexcible.precompute --config "configs/$name.yaml" \
            > "$LOG/$name.precompute.log" 2>&1 || { echo "PRECOMPUTE FAILED $name"; return 1; }
    $TR --nproc_per_node=2 -m flexcible.train --config "configs/$name.yaml" \
        > "$LOG/$name.train.log" 2>&1 || { echo "TRAIN FAILED $name"; return 1; }
    $PY -m flexcible.predict --config "configs/$name.yaml" --chroms test --tta \
        --out "runs/$name/pred_even.cool" > "$LOG/$name.predict.log" 2>&1 \
        || { echo "PREDICT FAILED $name"; return 1; }
    $PY -m flexcible.evaluate --config "configs/$name.yaml" \
        --pred "runs/$name/pred_even.cool" --chroms test \
        --out "runs/$name/scc.json" > "$LOG/$name.evaluate.log" 2>&1 \
        || echo "EVAL FAILED $name"
}

for name in meth_25kb methonly_25kb methseq_25kb dropout_25kb; do
    run_one "$name"
done

# greedy minimal assay set on the dropout-trained model (inference only)
echo "=== minimal-set search (hic) $(date +%H:%M:%S) ==="
$PY scripts/minimal_set.py --config configs/dropout_25kb.yaml --mode hic \
    --chrom 2 --out runs/dropout_25kb/minimal_set_hic.json \
    > "$LOG/minimal_set_hic.log" 2>&1 || echo "MINSET FAILED"

# capture-Hi-C project: architecture zoo + its own minimal-set search
if [ -d ../pchic-unet ]; then
    echo "=== pchic zoo $(date +%H:%M:%S) ==="
    (cd ../pchic-unet && bash scripts/run_zoo.sh) > "$LOG/pchic_zoo.log" 2>&1 \
        || echo "PCHIC ZOO FAILED"
fi
echo "METH_MINSET_DONE $(date +%H:%M:%S)"
