#!/bin/bash
set -u
cd /home/mh-hannover.local/wolffjoa/src/hic-multiome-unet
PY=~/miniconda3/envs/hicdiff/bin/python
export CUDA_VISIBLE_DEVICES=0
for v in struct_donut struct_insul struct_both; do
  echo "[$(date +%T)] $v predict"
  $PY scripts/bioscore_predict.py --config configs/$v.yaml \
      --checkpoint runs/$v/checkpoint-best --out runs/bioscore/preds_$v.npz || exit 1
  echo "[$(date +%T)] $v bioscore"
  $PY scripts/bioscore_akita.py --preds runs/bioscore/preds_$v.npz \
      --out runs/bioscore/bs_$v || exit 1
done
echo "[$(date +%T)] STRUCT_BIOSCORE_DONE"
