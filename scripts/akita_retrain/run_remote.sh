#!/bin/bash
# One Akita job on the remote RTX 4090: claim, train, predict held-out chromosomes.
#   run_remote.sh <run> <split A|B> <cells all|gm12878> <max-minutes> [extra train.py args]
# Run on the remote machine (env ~/miniconda3/envs/hicgan, TensorFlow 2.15). The claim name is
# akita_retrain_<run>; a finished run (DONE file) is skipped, a failed one is released.
set -u
RUN=$1; SPLIT=$2; CELLS=$3; MINUTES=$4; shift 4
W=/mnt/processed/wolffjoa/hic-multiome/akita_retrain
PY=$HOME/miniconda3/envs/hicgan/bin/python
export PYTHONPATH=$W/pylib:$W/basenji_src
export TF_CPP_MIN_LOG_LEVEL=2
export TF_CUDNN_USE_AUTOTUNE=${TF_CUDNN_USE_AUTOTUNE:-0}
export AKITA_DATA_LOCAL=${AKITA_DATA_LOCAL:-$HOME/scratch/akita_data}
cd $W
python3 $HOME/hic-multiome-unet/scripts/claim_job.py claim akita_retrain_$RUN \
    --done-path $W/runs/$RUN/DONE --cmd "akita_retrain $RUN split $SPLIT cells $CELLS" || exit 0
$PY scripts/train.py --split $SPLIT --cells $CELLS --run $RUN --max-minutes $MINUTES "$@" \
    > logs/$RUN.train.log 2>&1 \
    || { python3 $HOME/hic-multiome-unet/scripts/claim_job.py release akita_retrain_$RUN; exit 1; }
if [ "$SPLIT" = "A" ]; then CH="chr2 chr3 chr11 chr14 chr17"; else CH="chr2 chr4 chr6 chr8 chr10 chr12 chr14 chr16 chr18 chr20 chr22"; fi
if [ "${NO_PREDICT:-0}" != "1" ]; then
  $PY scripts/predict.py --kind retrained --run $RUN --chroms $CH > logs/$RUN.predict.log 2>&1
fi
