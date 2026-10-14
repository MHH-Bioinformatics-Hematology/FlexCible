#!/bin/bash
# Full-schedule Akita run (registry 79b): original stopping rule, no wall-clock budget.
#   run_orig.sh <py> <gpu|""> <run> <split> <params> [extra train.py args]
# claim akita_orig_<run>; train (patience 12 epochs on validation loss, cap 200 epochs,
# resumable); predict the held-out chromosomes.
PY=$1; GPU=$2; RUN=$3; SPLIT=$4; PARAMS=$5; shift 5
W=/mnt/processed/wolffjoa/hic-multiome/akita_retrain
CJ=$HOME/src/hic-multiome-unet/scripts/claim_job.py; [ -f $CJ ] || CJ=$HOME/hic-multiome-unet/scripts/claim_job.py
export PYTHONPATH=$W/pylib:$W/basenji_src TF_CPP_MIN_LOG_LEVEL=2 TF_CUDNN_USE_AUTOTUNE=0
export AKITA_DATA_LOCAL=${AKITA_DATA_LOCAL:-$HOME/scratch/akita_data}
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU
cd $W
python3 $CJ claim akita_orig_$RUN --done-path $W/runs/$RUN/DONE --cmd "akita_orig $RUN split $SPLIT" || exit 0
$PY scripts/train.py --split $SPLIT --cells all --run $RUN --params $PARAMS --patience 12 \
    --max-epochs 200 --select loss --resume "$@" >> logs/$RUN.train.log 2>&1 \
  || { python3 $CJ release akita_orig_$RUN; exit 1; }
if [ "$SPLIT" = "A" ]; then CH="chr2 chr3 chr11 chr14 chr17"; else CH="chr2 chr4 chr6 chr8 chr10 chr12 chr14 chr16 chr18 chr20 chr22"; fi
$PY scripts/predict.py --kind retrained --run $RUN --chroms $CH --batch ${PRED_BATCH:-4} > logs/$RUN.predict.log 2>&1
