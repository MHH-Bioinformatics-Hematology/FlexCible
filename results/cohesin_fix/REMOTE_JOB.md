# Remote job spec: cohesin-fraction input and cohesin-loss augmentation (registry section 55)

Status: NOT STARTED. Needs the GPU time to be released by the user (after the AML folds).
Nothing here touches gpu_scheduler.py, gpu_queue_extra.py or the remote_queue*.py runners.

## Jobs (single 4090, scripts/run_single_gpu.py recipe: batch size 6, same optimizer steps as the 2-GPU runs)

| order | claim name | what | estimate (measured 12-cell speeds: 25 kb about 10 min, 10 kb about 23 min) |
|---|---|---|---|
| 1 | all12cohfix_refres_drop_25kb | main run, 25 kb | 0.2 h |
| 2 | all12cohfix_refres_drop_10kb | main run, 10 kb | 0.4 h |
| 3 (optional) | all12cohfixna_refres_drop_25kb | ablation without augmentation (row and shared contact scale only) | 0.2 h |
| 4 (optional) | all12cohfixna_refres_drop_10kb | the same, 10 kb | 0.4 h |

Total 0.6 GPU hours for 1 and 2, 1.2 GPU hours with the ablation. The augmentation adds one 5-layer
convolution (24 channels) on 25 percent of the windows of a batch, under 3 percent of a step. The
number of epochs to the patience stop is the uncertainty: the degron run all12deg_refres_drop took 78
epochs at 10 kb (121 min on the busy local pair, 169 windows/s) and 100 epochs at 25 kb; a cohfix run
that needs twice as many epochs costs 1.2 GPU hours (jobs 1 and 2). GPU evaluation of the intact cells
(`--stage eval`) adds about 5 min per run; everything else (perturbation, washout, STAG2 knockout) runs on
the CPU.

## Prerequisites (all CPU, done on the local machine before the GPU is needed)

1. Remote repo copy synced from the local working tree (new files: hicmultiome/cohesin.py,
   scripts/cohesin_operator.py, scripts/cohesin_fix_diagnose.py, scripts/cohesin_fix_measured.py,
   tests/test_cohesin_fix.py; changed: hicmultiome/config.py, precompute.py, train.py,
   scripts/multicell_expanded.py, perturb_validate.py, washout.py, aml_eval.py, hard_split.py,
   crosscell_predict.py). `python -m pytest tests/test_cohesin_fix.py -q` passes there on the CPU.
2. Caches `$HM/cache_expanded/mc_all12cohfix_refres_drop_{25,10}kb/manifest.json` exist (one cache is
   shared by the main run and the ablation). Built on the local CPU at nice 19:
   `nice -n 19 python scripts/multicell_expanded.py --runs all12cohfix_refres_drop --res 25000 10000 --stage precompute --workers 4`
   (25 kb about 10 min, 10 kb about 35 min with 8 workers).
3. Operators on the share: `$HM/cohesin_fix/operator_{10,25}kb.pt` (copies of
   results/cohesin_fix/operator_*.pt, fitted by `scripts/cohesin_operator.py --res 10000|25000` on the
   odd chromosomes of the HCT116 pair).

## Run (per job; remote shell as in scripts/remote/REMOTE.md)

    source ~/hicenv/bin/activate ; cd ~/hic-multiome-unet
    HM=/mnt/processed/wolffjoa/hic-multiome
    RUN=all12cohfix_refres_drop ; RES=25000 ; NAME=${RUN}_25kb      # then RES=10000 ; NAME=${RUN}_10kb
    ls $HM/cache_expanded/mc_all12cohfix_refres_drop_$((RES/1000))kb/manifest.json $HM/cohesin_fix/operator_$((RES/1000))kb.pt
    python scripts/claim_job.py claim $NAME --done-path $HM/runs/multicell_reference/$NAME/best \
        --cmd "run_single_gpu $RUN $RES" || exit 0
    python scripts/run_single_gpu.py multicell_expanded --runs $RUN --res $RES --stage train --gpus 0 \
        > $HM/runs/multicell_reference/$NAME.remote.out 2>&1 \
        || python scripts/claim_job.py release $NAME
    python scripts/run_single_gpu.py multicell_expanded --runs $RUN --res $RES --stage eval --gpus 0 \
        >> $HM/runs/multicell_reference/$NAME.remote.out 2>&1

For the ablation use RUN=all12cohfixna_refres_drop. Start nothing while ~/seqcnn or another job holds the
GPU (`nvidia-smi`, `pgrep -f seq_cnn_chip`).

## Evaluation afterwards (CPU, same numbers as for the degron models)

    cd ~/src/hic-multiome-unet ; export CUDA_VISIBLE_DEVICES= ; PY=~/miniconda3/envs/hicdiff/bin/python
    nice -n 10 $PY scripts/perturb_validate.py --stage predict predict_chip score
    nice -n 10 $PY scripts/washout.py --stage predict score
    nice -n 10 $PY scripts/aml_eval.py --stage predict --models all12cohfix_refres_drop_10kb all12cohfix_refres_drop_25kb \
        all12cohfixna_refres_drop_10kb all12cohfixna_refres_drop_25kb
    nice -n 10 $PY scripts/aml_eval.py --stage perturb --res 10000 25000
    nice -n 10 $PY scripts/hard_split.py ...      # optional: hard-region summary as in section 25 (same call as for the cohunk runs)

Outputs: results/perturb_validate/perturb_summary.csv (rows `minus_RAD21`, `minus_RAD21_f0`,
`minus_RAD21_f0.5`, `minus_CTCF`, `measured_chip`), results/washout/washout_summary.csv,
results/aml/perturb_summary.csv (variants u937_ko_absent, _f0, _f0.5, u937_ko_f0.5, u937_ko_f0.75).
Intact performance: `--stage summary` of multicell_expanded and the 413-region medians of crosscell_predict
(compare with all12deg_refres_drop at the same resolution; the augmentation must not lower them).
Diagnostic of the trained model (same script as the diagnosis, 184 even-chromosome windows):
`$PY scripts/cohesin_fix_diagnose.py --run all12cohfix_refres_drop_10kb`.
