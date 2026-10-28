#!/bin/bash
set -u
cd ~/src/flexcible
export CUDA_VISIBLE_DEVICES=1
PY=~/miniconda3/envs/hicdiff/bin/python
for n in sharp_grad sharp_hf sharp_ssim; do
  echo "[$(date +%T)] $n train"
  $PY -m flexcible.train --config configs/$n.yaml || exit 1
  echo "[$(date +%T)] $n score (with TTA)"
  $PY scripts/score_like_hicgan.py --config configs/$n.yaml --tta --even-only \
      --out runs/$n/score_tta.csv || exit 1
  echo "[$(date +%T)] $n score (NO TTA - averaging is itself a blur)"
  $PY scripts/score_like_hicgan.py --config configs/$n.yaml --even-only \
      --out runs/$n/score_nott a.csv 2>/dev/null || \
  $PY scripts/score_like_hicgan.py --config configs/$n.yaml --even-only \
      --out runs/$n/score_notta.csv || exit 1
done
echo "[$(date +%T)] SHARP_DONE"
