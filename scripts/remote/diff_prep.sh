#!/bin/bash
cd ~/hic-multiome-unet
export CUDA_VISIBLE_DEVICES=
for RUN in all12diffcells_refres_drop all12diff_loopfocus_refres_drop_f10 all12diff_loopfocus_refres_drop_f11 all12diff_loopfocus_refres_drop_f12; do
  echo "$(date +%T) start $RUN" >> diff_prep.log
  nice -n 10 ~/hicenv/bin/python scripts/multicell_expanded.py --runs $RUN --res 10000 --stage precompute --workers 16 >> diff_prep.out 2>&1
  echo "$(date +%T) rc=$? $RUN" >> diff_prep.log
done
echo "$(date +%T) DIFF_PREP_DONE" >> diff_prep.log
