#!/bin/bash
set -u
cd ~/src/hic-multiome-unet
PY=~/miniconda3/envs/hicdiff/bin/python
run () {   # gpu arm
  export CUDA_VISIBLE_DEVICES=$1
  echo "[$(date +%T)] $2 train (gpu $1)"
  $PY -m hicmultiome.train --config configs/$2.yaml || return 1
  echo "[$(date +%T)] $2 score"
  $PY scripts/score_like_hicgan.py --config configs/$2.yaml --even-only \
      --out runs/$2/score.csv || return 1
  echo "[$(date +%T)] $2 DONE"
}
( run 0 struct_insul; run 0 struct_both ) &
( run 1 struct_donut ) &
wait
echo "[$(date +%T)] STRUCT_DONE"
