#!/bin/bash
# Corrected-target GM12878, Akita-style area holdout, 30 min on both GPUs,
# then scored with the exact Hi-cGAN-vs-Akita measurement.
set -u
cd ~/src/flexcible
PY=~/miniconda3/envs/hicdiff/bin/python
TR=~/miniconda3/envs/hicdiff/bin/torchrun
CFG=configs/gm_corr_region_2kb.yaml
COOL=/mnt/processed/wolffjoa/hic-multiome/gm12878_2kb_akita_corrected.cool

echo "[$(date +%T)] waiting for the corrected target to finish"
while pgrep -f "make_akita_normalised_matrix.py" > /dev/null; do sleep 30; done
$PY -c "import cooler;c=cooler.Cooler('$COOL');print('target ready:',c.binsize,'bp',c.info['nnz'],'pixels')" || exit 1

echo "[$(date +%T)] building cache"
$PY -m flexcible.precompute --config $CFG || exit 1

echo "[$(date +%T)] training 30 min on 2 GPUs"
$TR --nproc_per_node=2 --master_port=29561 -m flexcible.train --config $CFG || exit 1

echo "[$(date +%T)] scoring against Akita's own target, their metric"
$PY scripts/score_like_hicgan.py --config $CFG --tta --even-only || exit 1
echo "[$(date +%T)] PIPELINE_DONE"
