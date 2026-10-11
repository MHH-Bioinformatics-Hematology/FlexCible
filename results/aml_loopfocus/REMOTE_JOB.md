# Remote GPU job: loop-centric AML recipe (registry section 63)

Status: PENDING (implemented and CPU-tested; no GPU run started). Read scripts/remote/REMOTE.md first
(environment `source ~/hicenv/bin/activate`, `cd ~/hic-multiome-unet`, claims, the GPU shared with
the sequence CNN, batch size wrapper).

## What runs

Six trainings, one recipe: the residual-reference AML model (`all12aml_refres_drop`, model.reference
residual, track dropout uniform_count, AML x3 oversampling, patient hard validation) plus
`train.aml_object_*` (hicmultiome/aml_objects.py):

| run name (= claim name) | resolution | cache (exists, shared with all12aml_refres_drop) |
|---|---|---|
| all12aml_loopfocus_refres_drop_f10_10kb | 10 kb | cache_expanded/mc_all12aml_ref_10kb_f10 |
| all12aml_loopfocus_refres_drop_f11_10kb | 10 kb | cache_expanded/mc_all12aml_ref_10kb_f11 |
| all12aml_loopfocus_refres_drop_f12_10kb | 10 kb | cache_expanded/mc_all12aml_ref_10kb_f12 |
| all12aml_loopfocus_refres_drop_f10_25kb | 25 kb | cache_expanded/mc_all12aml_ref_25kb_f10 |
| all12aml_loopfocus_refres_drop_f11_25kb | 25 kb | cache_expanded/mc_all12aml_ref_25kb_f11 |
| all12aml_loopfocus_refres_drop_f12_25kb | 25 kb | cache_expanded/mc_all12aml_ref_25kb_f12 |

Priority: the three 10 kb runs first (the evaluation of section 62 is at 10 kb), the 25 kb runs after.
Folds 10, 11, 12 are those of results/aml_train/split_v2.json (test patients 018 027 270 | 168 546 629
1360 | 472 773 1916); same-donor pairs (168/496, 260/424) stay on one side.

Recipe flags (written by `scripts/multicell_expanded.py` run_config, `loopfocus_train`):

    data.aml_objects_table  /mnt/processed/wolffjoa/hic-multiome/aml/objects/objects_f<N>.csv.gz
    train.aml_object_oversample 5        # matched (AML-specific, control) window pairs x5 on top of AML x3
    train.aml_object_controls [A_hspc]   # control set of the classifier
    train.aml_object_pair_batches true   # batches of half AML-specific, half control windows
    train.aml_object_weight 1.0          # neighbourhood L1 + object-enrichment match, residual space
    train.aml_object_rank 0.5            # pairwise logistic ranking, AML-specific vs control
    train.aml_object_rank_temp 0.05, aml_object_rank_consistent true, aml_object_rank_space residual
    train.aml_object_donut [1, 4] (10 kb), [1, 3] (25 kb)

## Preparation (CPU, nothing to precompute)

1. Caches: the six caches above exist (manifest.json checked 2026-10-11). No new window cache is
   needed: the loop-centric runs read the AML reference caches of the refres / refin runs. If a
   cache is missing, `python scripts/multicell_expanded.py --runs all12aml_loopfocus_refres_drop_f10 --res 10000 --stage precompute --workers 8`
   builds the same cache name.
2. Objects tables (done, on the share, identical on both machines):
   `/mnt/processed/wolffjoa/hic-multiome/aml/objects/objects_f{10,11,12}.csv.gz`
   (regenerate: `python scripts/aml_objects_table.py`; it asserts that no test patient or same-donor
   partner is in a table). The generated configs point at these paths, not at the repo.
3. Code sync: rsync the working tree of ~/src/hic-multiome-unet to the remote repo copy
   (new: hicmultiome/aml_objects.py, scripts/aml_objects_table.py, edits in hicmultiome/train.py,
   hicmultiome/config.py, scripts/multicell_expanded.py, scripts/aml_train_eval.py,
   scripts/aml_loop_detection.py). Running jobs import the old copy from memory; sync before the first
   loop-centric job starts, not while another job of the remote queue is about to start.
4. `ls results/aml_train/hard_f1{0,1,2}_{10,25}kb.json` in the remote repo copy (written there by
   aml_prep.py; the local copy does not hold them; the config references them).
5. 60-step smoke (GPU idle first: `nvidia-smi`, `pgrep -f seq_cnn_chip`). The reviewed configs are
   written by `python scripts/aml_loopfocus_configs.py` (results/aml_loopfocus/configs/, paths of the repo
   copy that runs it, hard-split path included):

       python scripts/aml_loopfocus_configs.py
       python - <<'PY'
       import yaml, os
       n = "all12aml_loopfocus_refres_drop_f10_25kb"
       c = yaml.safe_load(open(f"results/aml_loopfocus/configs/{n}.yaml"))
       c["train"].update(max_steps=60, eval_every=30, max_epochs=0, patience_epochs=0, batch_size=6,
                         output_dir=os.path.expanduser("~/scratch/loopfocus_smoke"), save_checkpoints=False)
       yaml.safe_dump(c, open(os.path.expanduser("~/scratch/loopfocus_smoke.yaml"), "w"), sort_keys=False)
       PY
       torchrun --standalone --nproc_per_node=1 -m hicmultiome.train --config ~/scratch/loopfocus_smoke.yaml

   Expected log lines: `aml objects ...: N objects of 7 samples`, `oversample x5 adds ...`, and per epoch
   `obj_pix= obj_enr= obj_rank= obj_n=`.

## Train (one job per claim, as in REMOTE.md)

    source ~/hicenv/bin/activate ; cd ~/hic-multiome-unet
    HM=/mnt/processed/wolffjoa/hic-multiome
    for RES in 10000 25000 ; do for F in 10 11 12 ; do
      RUN=all12aml_loopfocus_refres_drop_f$F ; NAME=${RUN}_$((RES/1000))kb
      python scripts/claim_job.py claim $NAME --done-path $HM/runs/multicell_reference/$NAME/best \
          --cmd "run_single_gpu $RUN $RES" || continue          # exit 1 or 2: claimed elsewhere or finished
      python scripts/run_single_gpu.py multicell_expanded --runs $RUN --res $RES --stage train --gpus 0 \
          > $HM/runs/multicell_reference/$NAME.remote.out 2>&1 \
          || python scripts/claim_job.py release $NAME
    done ; done

`run_single_gpu.py` doubles the per-rank batch size (3 to 6), so a batch of the pair-batch composer is
3 AML-specific plus 3 control windows. The output folder is runs/multicell_reference/<NAME>, log
<NAME>.log, early stopping as the other AML runs (patience 10 epochs on the honest patient-hard validation).

## Evaluate

Per fold after its 10 kb (and 25 kb) run (inference only, same GPU, `HICMULTI_GPUS=0`):

    python scripts/aml_train_eval.py --fold 10 --variant extra     # also 11, 12
    # tables results/aml_train/eval_f<N>_extra_{summary,patients,...}.csv; copy to
    # $HM/runs/aml_train_remote/fold<N>/ as the other extra evaluations (model label aml_loopfocus, 10 kb and 25 kb)

The variant evaluates every ready model of the extra list (Meth, residual reference, loop-centric);
predictions already on disk are reused. Success criterion 3 (no loss of Pearson): the pooled 10 kb
Pearson of aml_loopfocus against aml_refin (0.813), from scripts/aml_folds_v2_summary.py with the
aml_loopfocus rows.

Loop and stripe test (10 kb only, ten test patients, needs all three folds trained; the prediction
stage needs the GPU, scoring and statistics are CPU):

    python scripts/aml_loop_detection.py --recipe loopfocus --stage predict --gpu 0      # remote, ~25 min
    python scripts/aml_loop_detection.py --recipe loopfocus --stage score stats inc criteria   # any machine
    # tables results/aml_loop_detection_loopfocus/{criteria,auroc,levels_by_class,gain_correlation,increments}.csv

The section 62 predictions (aml_refin, no-marks, HSPC reference, base model) are reused from
runs/aml_loop_detection/f<N>/ on the share; only aml_loopfocus (marks and no-marks) is predicted.
`criteria.csv` holds the success criteria with donor-bootstrap intervals:

| criterion | target | section 62 reference model |
|---|---|---|
| AUROC of the gain over HSPC, private loops against the matched shared loops (A_hspc) | >= 0.70 | 0.59 |
| amplitude (median model gain / median measured gain), private loops | >= 0.30 | 0.11 |
| pooled 10 kb Pearson on the Akita test regions | no loss against 0.813 (aml_refin) | 0.813 |

## Estimated GPU time (remote 4090, measured speeds: 10 kb 330 windows/s, 25 kb 630 windows/s)

AML 10 kb refin_drop took 1.0 to 1.7 h (f10 100 min / 49 epochs, f11 57 min / 30 epochs, f12 66 min /
33 epochs; 40.5k, 38.1k, 39.9k windows per epoch). The recipe adds about 7.5k pair windows per epoch
(934 matched pairs x 2 x 4 extra copies at fold 10; 18 percent more windows) and the object loss
(window geometry on the CPU, two small convolutions) about 5 to 10 percent per step, so 25 to 30
percent per epoch; the epoch count is assumed unchanged to 1.5x (the ranking term adds gradient noise).

| job | estimate |
|---|---|
| 10 kb, per fold | 1.3 to 2.6 h (f10 2.2 to 2.6, f11 1.3 to 1.5, f12 1.4 to 1.7 at 1.0x epochs; up to 1.5x epochs) |
| 10 kb, three folds | about 5 h (range 4 to 7.5 h) |
| 25 kb, per fold | 0.6 to 1.4 h (refin_drop took 28 to 55 min) |
| 25 kb, three folds | about 2.5 h (range 1.8 to 4 h) |
| aml_train_eval --variant extra, per fold | about 0.3 h inference |
| aml_loop_detection predict (aml_loopfocus, ten patients, 10 kb) | about 0.4 h |
| total, 10 kb only | about 6.5 h; with the 25 kb runs about 9.5 h |

Do not start while ~/seqcnn holds the GPU. Run the 10 kb folds sequentially (one 4090).

## Stop rules and risks

- Early epochs: obj_rank near 0.69 x weight (softplus at zero) is expected; if obj_pix does not fall
  within the first 5 epochs, stop and report (the log prints the three terms per epoch).
- If the honest validation Pearson of a fold ends more than 0.01 below the aml_refres_drop run of the
  same fold, report before continuing with the other folds (success criterion 3).
- Seed variation was not measured for the AML models; compare the criteria against the section 62
  intervals (donor bootstrap), not point values alone.
