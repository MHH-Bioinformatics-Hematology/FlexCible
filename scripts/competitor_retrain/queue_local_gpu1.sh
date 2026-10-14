#!/bin/bash
# Registry 77b: retraining queue of the local GPU 1 (one training at a time; claims competitor_retrain_*).
# Order: Epiphany adaptations (25, 5, 2 kb; the 10 kb run is the native one of section 77, 55 epochs), then C.Origami
# adaptations to the exact 10, 25, 5, 2 kb bins (80 epochs cap, patience 12, as the original).
RT=/mnt/processed/wolffjoa/hic-multiome/competitor_retrain
PY=$HOME/miniconda3/envs/hicdiff/bin/python
cd $HOME/src/hic-multiome-unet/scripts/competitor_retrain
export CUDA_VISIBLE_DEVICES=1 COMPRT_LOCAL=$HOME/scratch/competitor_retrain OMP_NUM_THREADS=2
CLAIM="python3 $HOME/src/hic-multiome-unet/scripts/claim_job.py"
epi() {  # res; the 30-minute pilot run directory is continued (latest.pt) by the full run
  k=$(($1/1000))
  $CLAIM claim competitor_retrain_epiphany_${k}kb_full >/dev/null || return
  while pgrep -f "train_epiphany.py.*max-minutes" >/dev/null; do sleep 30; done
  COMPRT_EPI_RES=$1 nice -n 5 $PY train_epiphany.py --out $RT/runs/epiphany_${k}kb_pilot --epochs 55 --patience 12 --workers 4 >> $RT/logs/epiphany_${k}kb_full.log 2>&1
}
co() {   # bin
  name=corigami_$(($1/1000))kb_exact80
  $CLAIM claim competitor_retrain_$name >/dev/null || return
  while pgrep -f "rsync.*co_feats" >/dev/null; do sleep 60; done
  micro=4; [ $1 -ge 25000 ] && micro=2
  COMPRT_BIN=$1 nice -n 5 $PY train_corigami.py --out $RT/runs/$name --epochs 80 --patience 12 --micro $micro --workers 8 >> $RT/logs/$name.log 2>&1
}
epi 25000
epi 5000
epi 2000
co 10000
co 25000
co 5000
co 2000
