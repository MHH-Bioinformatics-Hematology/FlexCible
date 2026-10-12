#!/bin/bash
cd ~/hic-multiome-unet
export HICMULTI_GPUS=0
D=/mnt/processed/wolffjoa/hic-multiome/runs
L=$D/diff_marks/meth_resid.log
for F in 10 11 12; do
  echo "$(date +%T) eval fold $F" >> $L
  ~/hicenv/bin/python scripts/aml_train_eval.py --fold $F --variant meth_resid >> $D/diff_marks/meth_resid_eval_f$F.out 2>&1
  echo "$(date +%T) rc=$? fold $F" >> $L
  mkdir -p $D/aml_train_remote/fold$F
  for p in results/aml_train/eval_f${F}_meth_resid_*.csv; do cp $p $D/aml_train_remote/fold$F/; done
done
echo "$(date +%T) loop detection predict/score" >> $L
~/hicenv/bin/python scripts/aml_loop_detection.py --recipe meth_resid --stage predict score --gpu 0 >> $D/diff_marks/meth_resid_loop.out 2>&1
echo "$(date +%T) rc=$? loop predict/score" >> $L
~/hicenv/bin/python scripts/aml_loop_detection.py --recipe meth_resid --stage stats inc criteria >> $D/diff_marks/meth_resid_loop.out 2>&1
~/hicenv/bin/python scripts/aml_loop_detection.py --stage criteria2 --scores results/aml_loop_detection_meth_resid/object_scores.csv.gz --tag meth_resid --nboot 1000 >> $D/diff_marks/meth_resid_loop.out 2>&1
echo "$(date +%T) rc=$? criteria2" >> $L
cp results/aml_loop_detection_criteria2/meth_resid_*.csv $D/diff_marks/ 2>/dev/null
mkdir -p $D/diff_marks/aml_loop_detection_meth_resid; cp results/aml_loop_detection_meth_resid/*.csv* $D/diff_marks/aml_loop_detection_meth_resid/ 2>/dev/null
echo "$(date +%T) METH_RESID_DONE" >> $L
