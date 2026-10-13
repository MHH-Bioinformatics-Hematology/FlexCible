#!/bin/bash
# Window prior (registry 76): P trainings, then the cooler predictions. One GPU job at a time, claims
# windowprior_* by name (scripts/claim_job.py), logs in $HM/window_prior/queue.log.
# usage: bash scripts/window_prior_queue.sh <res...>     e.g. 25000 10000
cd "$(dirname "$0")/.." || exit 1
source ~/hicenv/bin/activate
export HICMULTI_GPUS=0 CUDA_VISIBLE_DEVICES=0
HM=/mnt/processed/wolffjoa/hic-multiome
WP=$HM/window_prior
mkdir -p $WP
LOG=$WP/queue.log
say() { echo "$(date +%F_%T) $*" | tee -a $LOG; }
job() {   # job <claim name> <done path> <logfile> <command...>
  local name=$1 done=$2 lf=$3; shift 3
  python scripts/claim_job.py claim "$name" --done-path "$done" --cmd "$*" >/dev/null || { say "skip $name"; return 0; }
  say "start $name"
  if nice -n 5 "$@" >> "$lf" 2>&1; then say "done $name"; else say "FAILED $name"; python scripts/claim_job.py release "$name"; return 1; fi
}
for RES in "$@"; do
  K=$((RES/1000))
  for MODE in ref own; do
    for FOLD in odd even full; do
      D=$WP/${K}kb/$MODE/P_$FOLD
      mkdir -p $D
      job windowprior_P_${MODE}_${FOLD}_${K}kb $D/DONE $D/train.log python scripts/window_prior.py train --res $RES --mode $MODE --fold $FOLD || exit 1
    done
    for SRC in oof; do
      D=$WP/${K}kb/$MODE
      job windowprior_cools_${MODE}_${K}kb $D/cools/caco2.cool $D/predict.log python scripts/window_prior.py predict --res $RES --mode $MODE --source $SRC \
        --cells gm12878 imr90 hct116 hff h1 hepg2 a549 nhek gm23248 mcf7 cd14mono pc9 k562 hmec huvec hmec_deep huvec_deep pc3 panc1 nk caco2 || exit 1
    done
  done
done
say "WINDOW_PRIOR_QUEUE_DONE $*"
