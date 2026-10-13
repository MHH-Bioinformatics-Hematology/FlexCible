#!/bin/bash
# One Akita job on a LOCAL GPU (only when gpu_scheduler.py and its queue are idle).
#   run_local.sh <gpu index> <run> <split A|B> <cells all|gm12878> <max-minutes> [train.py args]
# At most 6 CPU threads, nice 10. Env: ~/miniconda3/envs/basenji (TensorFlow 2.15).
set -u
GPU=$1; RUN=$2; SPLIT=$3; CELLS=$4; MINUTES=$5; shift 5
W=/mnt/processed/wolffjoa/hic-multiome/akita_retrain
export PYTHONPATH=$W/pylib:$W/basenji_src
export CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=4 AKITA_TF_THREADS=4
export AKITA_DATA_LOCAL=${AKITA_DATA_LOCAL:-$HOME/scratch/akita_data}
cd $W
nice -n 10 $HOME/miniconda3/envs/basenji/bin/python scripts/train.py --split $SPLIT --cells $CELLS --run $RUN \
    --max-minutes $MINUTES "$@" > logs/$RUN.train.log 2>&1
