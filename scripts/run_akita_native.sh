#!/bin/bash
# Akita's own published matrices. Train file already has the test/valid
# regions cut out, so nothing here normalises or splits anything.
set -u
cd ~/src/flexcible
PY=~/miniconda3/envs/hicdiff/bin/python
TR=~/miniconda3/envs/hicdiff/bin/torchrun
CFG=configs/akita_native.yaml
echo "[$(date +%T)] cache"
$PY -m flexcible.precompute --config $CFG || exit 1
echo "[$(date +%T)] train 30 min, 2 GPUs"
$TR --nproc_per_node=2 --master_port=29571 -m flexcible.train --config $CFG || exit 1
echo "[$(date +%T)] score vs Akita's published TEST matrix"
$PY scripts/score_like_hicgan.py --config $CFG --tta --even-only || exit 1
echo "[$(date +%T)] NATIVE_DONE"
