#!/bin/bash
# Window prior (registry 76): the main models all12wp_{ref,self,both}_drop. Waits for the cooler of the P
# they read, precomputes the cache (CPU, nice 10), trains (one GPU, run_single_gpu.py doubles the per-rank
# batch) and evaluates (reference_summary format), claims windowprior_main_<run>.
# usage: bash scripts/window_prior_main.sh <res> <run...>
cd "$(dirname "$0")/.." || exit 1
source ~/hicenv/bin/activate
export HICMULTI_GPUS=0 CUDA_VISIBLE_DEVICES=0
HM=/mnt/processed/wolffjoa/hic-multiome
WP=$HM/window_prior
LOG=$WP/queue_main.log
say() { echo "$(date +%F_%T) $*" | tee -a $LOG; }
RES=$1; shift
K=$((RES/1000))
for R in "$@"; do
  NAME=${R}_${K}kb
  case $R in
    *self*) TAG=own ;;
    *refm*) TAG=ref_motif ;;
    *) TAG=ref ;;
  esac
  say "wait for $WP/${K}kb/$TAG/cools/caco2.cool"
  until [ -f $WP/${K}kb/$TAG/cools/caco2.cool ]; do sleep 30; done
  if [ ! -f $HM/cache_expanded/mc_$NAME/manifest.json ]; then
    say "precompute $NAME"
    nice -n 10 python scripts/multicell_expanded.py --runs $R --res $RES --stage precompute --workers 8 \
      >> $HM/runs/multicell_reference/$NAME.precompute.out 2>&1 || { say "PRECOMPUTE FAILED $NAME"; exit 1; }
  fi
  python scripts/claim_job.py claim windowprior_main_$NAME --done-path $HM/runs/multicell_reference/$NAME/best \
      --cmd "run_single_gpu $R $RES" > /dev/null
  rc=$?
  if [ $rc -eq 0 ]; then
    say "train $NAME"
    python scripts/run_single_gpu.py multicell_expanded --runs $R --res $RES --stage train --gpus 0 \
      > $HM/runs/multicell_reference/$NAME.remote.out 2>&1 || { say "TRAIN FAILED $NAME"; python scripts/claim_job.py release windowprior_main_$NAME; exit 1; }
  else
    say "train skipped $NAME rc=$rc"
  fi
  say "eval $NAME"
  python scripts/run_single_gpu.py multicell_expanded --runs $R --res $RES --stage eval --gpus 0 \
      > $HM/runs/multicell_reference/$NAME.eval.out 2>&1 || say "EVAL FAILED $NAME"
  say "done $NAME"
done
say "WINDOW_PRIOR_MAIN_DONE $RES $*"
