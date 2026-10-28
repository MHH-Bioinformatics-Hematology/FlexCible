#!/usr/bin/env bash
# Stage 2: multi-cell + cohesin-degron model (GM12878, K562, HCT116 wt,
# HCT116 RAD21-degron), 60-min DDP training, per-sample evaluation, and the
# in-silico perturbation demos.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/hicdiff/bin/python
TR=~/miniconda3/envs/hicdiff/bin/torchrun
LOG=runs/overnight; mkdir -p "$LOG"
CFG=configs/multicell_25kb.yaml

echo "=== multicell precompute $(date +%H:%M:%S) ==="
$PY -m flexcible.precompute --config $CFG > "$LOG/multicell.precompute.log" 2>&1 \
    || { echo "PRECOMPUTE FAILED"; exit 1; }
echo "=== multicell train $(date +%H:%M:%S) ==="
$TR --nproc_per_node=2 -m flexcible.train --config $CFG \
    > "$LOG/multicell.train.log" 2>&1 || { echo "TRAIN FAILED"; exit 1; }

for smp in gm12878 k562 hct116 hct116_cohesin_degron; do
    echo "=== eval $smp $(date +%H:%M:%S) ==="
    $PY -m flexcible.predict --config $CFG --sample "$smp" --chroms test \
        --out "runs/multicell_25kb/pred_even_$smp.cool" \
        > "$LOG/multicell.predict.$smp.log" 2>&1 || { echo "PREDICT FAILED $smp"; continue; }
    $PY -m flexcible.evaluate --config $CFG --sample "$smp" \
        --pred "runs/multicell_25kb/pred_even_$smp.cool" --chroms test \
        --out "runs/multicell_25kb/scc_$smp.json" \
        > "$LOG/multicell.evaluate.$smp.log" 2>&1 || echo "EVAL FAILED $smp"
done

# Does zeroing cohesin move the WT prediction toward degron reality?
# Predict hct116 wt tracks with RAD21 zeroed, score against the DEGRON cool.
echo "=== degron cross-check $(date +%H:%M:%S) ==="
$PY -m flexcible.predict --config $CFG --sample hct116 --chroms test \
    --zero-tracks RAD21 --out runs/multicell_25kb/pred_even_hct116_zeroRAD21.cool \
    > "$LOG/multicell.predict.zeroRAD21.log" 2>&1 || true
$PY -m flexcible.evaluate --config $CFG --sample hct116_cohesin_degron \
    --pred runs/multicell_25kb/pred_even_hct116_zeroRAD21.cool --chroms test \
    --out runs/multicell_25kb/scc_zeroRAD21_vs_degron.json \
    > "$LOG/multicell.evaluate.zeroRAD21_vs_degron.log" 2>&1 || true
$PY -m flexcible.evaluate --config $CFG --sample hct116_cohesin_degron \
    --pred runs/multicell_25kb/pred_even_hct116.cool --chroms test \
    --out runs/multicell_25kb/scc_wtpred_vs_degron.json \
    > "$LOG/multicell.evaluate.wtpred_vs_degron.log" 2>&1 || true

# Perturbation demos (goal b): delete CTCF / cohesin in a 4 Mb window
echo "=== perturbation demos $(date +%H:%M:%S) ==="
$PY -m flexcible.perturb --config $CFG --sample gm12878 \
    --region chr2:20000000-24000000 --tracks CTCF,RAD21,SMC3 \
    --out runs/multicell_25kb/perturb_chr2_ctcf_cohesin.png \
    > "$LOG/multicell.perturb1.log" 2>&1 || echo "PERTURB1 FAILED"
$PY -m flexcible.perturb --config $CFG --sample gm12878 \
    --region chr2:20000000-24000000 --tracks CTCF --zero-motifs \
    --out runs/multicell_25kb/perturb_chr2_ctcf_sites.png \
    > "$LOG/multicell.perturb2.log" 2>&1 || echo "PERTURB2 FAILED"
echo "STAGE2_DONE $(date +%H:%M:%S)"
