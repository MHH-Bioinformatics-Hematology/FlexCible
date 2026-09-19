#!/bin/bash
# Does the feature-level score agree with the eye? Pixel Pearson calls +hf the
# worst arm; the picture calls it the best. Score every arm on TADs and loops.
set -u
cd /home/mh-hannover.local/wolffjoa/src/hic-multiome-unet
PY=~/miniconda3/envs/hicdiff/bin/python
export CUDA_VISIBLE_DEVICES=0
for v in akita_native sharp_hf sharp_grad sharp_ssim; do
  echo "[$(date +%T)] $v predict"
  $PY scripts/bioscore_predict.py --config configs/$v.yaml \
      --checkpoint runs/$v/checkpoint-best \
      --out runs/bioscore/preds_$v.npz || exit 1
  echo "[$(date +%T)] $v bioscore"
  $PY scripts/bioscore_akita.py --preds runs/bioscore/preds_$v.npz \
      --out runs/bioscore/bs_$v || exit 1
done
echo "[$(date +%T)] BIOSCORE_VARIANTS_DONE"
