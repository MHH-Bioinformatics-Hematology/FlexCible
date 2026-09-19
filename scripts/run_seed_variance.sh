#!/bin/bash
set -u
cd ~/src/hic-multiome-unet
export CUDA_VISIBLE_DEVICES=1
PY=~/miniconda3/envs/hicdiff/bin/python
for s in 1 2; do
  echo "[$(date +%T)] seed $s train"
  $PY -m hicmultiome.train --config configs/akita_native_s$s.yaml || exit 1
  echo "[$(date +%T)] seed $s score"
  $PY scripts/score_like_hicgan.py --config configs/akita_native_s$s.yaml --tta --even-only || exit 1
done
echo "[$(date +%T)] SEEDS_DONE"
