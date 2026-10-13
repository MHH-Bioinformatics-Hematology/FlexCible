#!/bin/bash
# 4 workers share the block queue (locks per block); 25 kb blocks first, then 10 kb
RT=/mnt/processed/wolffjoa/hic-multiome/competitor_retrain
PY=$HOME/miniconda3/envs/hicdiff/bin/python
cd $HOME/src/hic-multiome-unet
w() {
  OMP_NUM_THREADS=1 nice -n 10 $PY scripts/hic_measures.py --stage chrho_score --chrho-competitor-retrain --jobs 2 --threads 2 --only '_25kb_' >> $RT/logs/score_w$1.log 2>&1
  OMP_NUM_THREADS=1 nice -n 10 $PY scripts/hic_measures.py --stage chrho_score --chrho-competitor-retrain --jobs 2 --threads 2 --only '_10kb_' >> $RT/logs/score_w$1.log 2>&1
}
for i in 1 2 3 4; do w $i & sleep 20; done
wait
OMP_NUM_THREADS=2 nice -n 10 $PY scripts/hic_measures.py --stage chrho_report --chrho-competitor-retrain >> $RT/logs/score_report.log 2>&1
echo finished >> $RT/logs/score_report.log
