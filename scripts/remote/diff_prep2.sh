#!/bin/bash
# HSPC-reference caches (channels only, 10 kb) after diff_prep.sh has finished
cd ~/hic-multiome-unet
export CUDA_VISIBLE_DEVICES=
while ! grep -q DIFF_PREP_DONE diff_prep.log; do sleep 30; done
for F in 10 11 12; do
  echo "$(date +%T) start all12diffhspc_refres_drop_f$F" >> diff_prep.log
  nice -n 10 ~/hicenv/bin/python scripts/multicell_expanded.py --runs all12diffhspc_refres_drop_f$F --res 10000 --stage precompute --workers 16 >> diff_prep.out 2>&1
  echo "$(date +%T) rc=$? all12diffhspc_refres_drop_f$F" >> diff_prep.log
done
echo "$(date +%T) DIFFHSPC_PREP_DONE" >> diff_prep.log
