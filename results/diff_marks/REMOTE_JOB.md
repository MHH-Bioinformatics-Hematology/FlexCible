# Remote GPU job: differential mark channels and all-cell objects (registry section 66)

Status: PENDING (implemented and CPU-tested; no cache built and no GPU run started). Read
scripts/remote/REMOTE.md first (environment `source ~/hicenv/bin/activate`, `cd ~/hic-multiome-unet`,
claims, the GPU shared with the sequence CNN, batch size wrapper).

## The idea in one paragraph

The rule "this chromatin-mark / methylation pattern leads to a loop or a stripe" is cell-type independent;
AML is only where it shows most clearly. The model gets, for every sample S, the marks of the reference
whose Hi-C it also reads and the difference of S to that reference, and one function
residual Hi-C ~ f(delta marks) is learned for the 12 training cell lines, the held-out cells and the AML
patients alike. The loop-centric recipe of section 63 (window oversampling, object neighbourhood loss,
ranking) is applied to the cell-specific loops and stripes of every training cell, not only of the AML
patients.

## Components (all flags default off; a run without them is unchanged, checked byte-identical)

| flag | value in the primary run | role |
|---|---|---|
| `data.diff_marks` | true | reference (`dref:<mark>`) and difference (`diff:<mark>:s<r>`) rows per mark, hicmultiome/diffmarks.py |
| `data.diff_scales` | [0, 20000, 100000] | radii in bp: the 10 kb bin, +-20 kb, +-100 kb box smoothing (25 kb: 0, +-1, +-4 bins) |
| `data.diff_reference` | mean_aml | reference of AML samples: `mean_aml` (training patients of the fold, leave-one-out), `hspc` (HSPC training donors), `consensus` (the 12-cell consensus) |
| `data.diff_reference_sets` | consensus: 12 cells; mean_aml: fold training patients; hspc: donors 050, 213 | reference pools; a sample reads its pool without itself (and without its cell identity) |
| `samples[].diff_set` | cell lines consensus; patients, MOLM-13 mean_aml; HSPC donors hspc | pool of the sample |
| `data.cell_objects_table` | share `cell_objects/cell_objects.csv.gz` | cell-specific loops / stripes of the 12 training cells and their matched shared controls |
| `train.object_scope` | all | objects of `data.aml_objects_table` and `data.cell_objects_table` |
| `train.aml_object_*` | as section 63; controls [A_hspc, A_ref] | oversample x5, pair batches, neighbourhood weight 1, ranking 0.5 |

Channel definition (per mark, per sample, no Hi-C read): z = (log1p(raw) - q50) / (q99 - q50) over the
autosomal bins of the sample (per-sample quantile anchoring); reference = mean of z over the pool members
that have the mark; difference = z(S) - reference, box-smoothed per scale, then standardised per sample,
mark and scale with the median and 1.4826 MAD, clipped to +-8. The rows enter after the other
sequence-feature rows and bypass the track normaliser (`TrackNormalizer.passthrough`). A mark the sample
lacks, or that no pool member has, gives 0 in all its rows; track dropout of a mark zeros its rows
(`apply_diff_drop`; at inference `blank_diff_rows`). The rows of a mark that `train.select_tracks` leaves
out are dropped too. 13 ChIP marks (AML patients: CTCF, H3K27ac, H3K27me3; some without CTCF) and Meth in
the `diffmeth` family give 13 x 4 = 52 (14 x 4 = 56) rows.

Objects of a cell (scripts/cell_objects_table.py): the cell-agnostic definitions of scripts/crosscell_rule_transfer.py
(stage `objects`, `$HM/crosscell_rules/cell_objects.csv.gz`, same operational definitions as the AML objects: loops are
local maxima of the log2 donut enrichment (1 to 4 bins, 10 kb, 6 to 150 bins) with both anchors on peak bins of H3K27ac or
CTCF, specific = enrichment >= 0.5 and gain over the consensus >= 0.30, shared = >= 0.6 in the cell and the consensus;
stripes 4 anchor bins x 40 extent bins at an offset of 50 bins, specific = score >= 0.55 and gain >= 0.20; controls `A_ref`
matched on distance, orientation and the consensus' CTCF and H3K27ac) converted into the table of section 63
(`--source rule_transfer`, the committed tables; `from_rule_transfer`). Per training cell the 400 loops and 200 stripes
that have a matched control, strongest by gain first (the AML patients have about 230 private loops each; the cap keeps
the oversampled window count at the section 63 level); the held-out cells K562, HMEC (deep), HUVEC (deep) go to the
evaluation table with 1000 loops and 1000 stripes each, a seeded random draw. `n_train_other` counts the other training
cells with a specific object within one bin (private among the training cells: loops 77 percent, stripes 47 percent).
`--source local` (the script's own Hi-C calls: donut peaks, stripe rectangle 4 x 20 bins adjacent to the anchor, 400 / 200
strongest per cell, controls matched on distance, CTCF and H3K27ac of the cell) is the self-contained fallback;
`--source auto` (default) picks `rule_transfer` when its file exists. `results/cell_objects/*.source` says which source wrote a table.

## Runs (run name = claim name)

Joint training = the all12aml_* composition (12 cell lines + the fold's training patients + HSPC donors
050, 213 + MOLM-13; AML x3, lab row, patient-hard validation, residual reference, track dropout
uniform_count 0.5), fold ids 10, 11, 12 of results/aml_train/split_v2.json (test patients 018 027 270 |
168 546 629 1360 | 472 773 1916; same-donor pairs stay together).

| tier | run name | what | cache (new unless noted) |
|---|---|---|---|
| 0 | all12diffcells_refres_drop_25kb, _10kb | 12 cell lines alone, diff channels only | mc_all12ref_diff_<k>kb |
| 0 | all12diffcells_loopfocus_refres_drop_25kb, _10kb | 12 cell lines alone, diff channels + cell objects | same cache |
| 1 | all12diff_loopfocus_refres_drop_f10_10kb, f11, f12 | PRIMARY: joint, diff + all-cell objects | mc_all12aml_ref_10kb_f<N>_diff |
| 1 | all12diff_loopfocus_refres_drop_f10_25kb, f11, f12 | the same at 25 kb | mc_all12aml_ref_25kb_f<N>_diff |
| 2 | all12diff_refres_drop_f<N>_10kb | ablation: diff channels only (compare with all12aml_refres_drop_f<N>) | same _diff cache |
| 2 | all12allobj_refres_drop_f<N>_10kb | ablation: all-cell objects only (compare with all12aml_loopfocus_refres_drop_f<N>) | existing mc_all12aml_ref_10kb_f<N> |
| 3 | all12diffmeth_loopfocus_refres_drop_f<N>_10kb | primary + the 14th track Meth (v2 cell set, Xu WGBS where it exists) | mc_all12aml_meth_ref_10kb_f<N>_diff |
| 3 | all12diffhspc_loopfocus_refres_drop_f<N>_10kb | primary with the HSPC donors as the reference of AML samples | mc_all12aml_ref_10kb_f<N>_diffhspc |

Tier 0 first (cheap, cell lines only: tells whether the channels or the cell objects cost or gain anything on
the cell-line evaluation before the AML runs start), then tier 1 at 10 kb, tier 2, the 25 kb runs, tier 3.

## Preparation (CPU, nothing on the GPU)

1. Code: rsync the working tree of ~/src/hic-multiome-unet to the remote repo copy. New:
   hicmultiome/diffmarks.py, scripts/cell_objects_table.py, scripts/cell_objects_eval.py,
   scripts/diff_marks_configs.py; edited: hicmultiome/{config,precompute,train,tracks,predict}.py,
   scripts/{multicell_expanded,crosscell_predict,aml_train_eval,aml_loop_detection}.py. Running jobs import
   the old copy from memory; sync before the first job of this section starts.
2. Cell objects tables (already written to `$HM/cell_objects/` and `results/cell_objects/` from the rule-transfer objects; to regenerate after that file changes, any machine; the local source needs about 2 min per cell for the Hi-C calls):

       python scripts/cell_objects_table.py --stage calls build               # 12 training cells -> $HM/cell_objects/cell_objects.csv.gz
       python scripts/cell_objects_table.py --heldout --stage calls build     # K562, HMEC deep, HUVEC deep (evaluation only)
       # --source rule_transfer / local to force one; with the default the crosscell_rule_transfer objects are taken once they exist

   The first asserts that only training cells are in the table (`assert_no_leak`) and prints the counts per
   cell, kind and set (results/cell_objects/summary.csv).
3. Window caches (CPU, `--workers 4`; each sample's bigWigs are read once, about 2.5 min per sample on the
   share, into `<cache>/diffmarks/` and reused; then the usual window pass). Local machine, niced:

       cd ~/src/hic-multiome-unet ; export CUDA_VISIBLE_DEVICES=
       for RUN in all12diffcells_refres_drop ; do for RES in 25000 10000 ; do
         nice -n 10 python scripts/multicell_expanded.py --runs $RUN --res $RES --stage precompute --workers 4 ; done ; done
       for F in 10 11 12 ; do for RES in 10000 25000 ; do
         nice -n 10 python scripts/multicell_expanded.py --runs all12diff_loopfocus_refres_drop_f$F --res $RES --stage precompute --workers 4 ; done ; done
       # tier 3, after tier 1: all12diffmeth_loopfocus_refres_drop_f$F and all12diffhspc_loopfocus_refres_drop_f$F at 10000

   (`all12diff_refres_drop_f<N>` shares the cache of the primary run; `all12allobj_refres_drop_f<N>` the existing AML cache.)
   `python scripts/multicell_expanded.py --runs <run> --res 10000 --stage ready` checks the inputs.
4. Reviewed configs: `python scripts/diff_marks_configs.py` (`--all` for the ablations) writes
   results/diff_marks/configs/<run>.yaml (paths of the machine that runs it; the hard-split JSONs of folds
   10 to 12 exist only in the remote repo copy).
5. 60-step smoke on the GPU before the first long job (GPU idle: `nvidia-smi`, `pgrep -f seq_cnn_chip`):

       python scripts/diff_marks_configs.py
       python - <<'PY'
       import yaml, os
       n = "all12diff_loopfocus_refres_drop_f10_25kb"
       c = yaml.safe_load(open(f"results/diff_marks/configs/{n}.yaml"))
       c["train"].update(max_steps=60, eval_every=30, max_epochs=0, patience_epochs=0, batch_size=6,
                         output_dir=os.path.expanduser("~/scratch/diff_smoke"), save_checkpoints=False)
       yaml.safe_dump(c, open(os.path.expanduser("~/scratch/diff_smoke.yaml"), "w"), sort_keys=False)
       PY
       torchrun --standalone --nproc_per_node=1 -m hicmultiome.train --config ~/scratch/diff_smoke.yaml

   Expected log lines: `diff marks: dropout of a track blanks its 52 reference/difference rows`,
   `aml objects <aml table>,<cell table>: N objects of M samples` (cells and patients), `obj_pix= obj_enr= obj_rank=`
   per epoch.

## Train (one job per claim)

    source ~/hicenv/bin/activate ; cd ~/hic-multiome-unet
    HM=/mnt/processed/wolffjoa/hic-multiome
    for RES in 25000 10000 ; do for RUN in all12diffcells_refres_drop all12diffcells_loopfocus_refres_drop ; do
      NAME=${RUN}_$((RES/1000))kb
      python scripts/claim_job.py claim $NAME --done-path $HM/runs/multicell_reference/$NAME/best \
          --cmd "run_single_gpu $RUN $RES" || continue
      python scripts/run_single_gpu.py multicell_expanded --runs $RUN --res $RES --stage train --gpus 0 \
          > $HM/runs/multicell_reference/$NAME.remote.out 2>&1 || python scripts/claim_job.py release $NAME
    done ; done
    # tier 1 (then the 25 kb runs, then tiers 2 and 3): the same loop with
    #   RUN=all12diff_loopfocus_refres_drop_f$F  for F in 10 11 12, RES 10000 then 25000
    #   RUN=all12diff_refres_drop_f$F, all12allobj_refres_drop_f$F (10000)
    #   RUN=all12diffmeth_loopfocus_refres_drop_f$F, all12diffhspc_loopfocus_refres_drop_f$F (10000)

Output folders runs/multicell_reference/<NAME>, logs <NAME>.log, early stopping on the honest patient-hard
validation (patience 10 epochs) as the other AML runs. `run_single_gpu.py` doubles the per-rank batch (3 to 6).

## Evaluate

1. Cell lines, standard evaluation (17 cells, 413 Akita test regions, disjoint test windows, every run; per
   cell the leave-one-out consensus / consensus12 alone as the baseline). Inference only, same GPU:

       python scripts/multicell_expanded.py --runs <run> --res 10000 --stage eval --gpus 0
       python scripts/multicell_expanded.py --stage refsummary      # rows of every run in $HM/runs/multicell_reference/reference_summary.csv

   (the diff channels of a held-out cell are built from the 12 training
   cells, its own track never enters its reference.)
2. AML, ten held-out patients, per fold (needs the run finished; model labels aml_diff_loopfocus, aml_diff,
   aml_allobj, aml_diffmeth_loopfocus):

       python scripts/aml_train_eval.py --fold 10 --variant extra     # also 11, 12; ~0.3 h inference per fold
       python scripts/aml_loop_detection.py --recipe diff_loopfocus --stage predict --gpu 0       # remote, ~0.5 h
       python scripts/aml_loop_detection.py --recipe diff_loopfocus --stage score stats inc criteria   # CPU
       # tables results/aml_loop_detection_diff_loopfocus/{criteria,auroc,levels_by_class,gain_correlation,increments}.csv

   The section 62 / 63 predictions (aml_refin, no-marks, HSPC reference, base model, aml_loopfocus) are reused from
   runs/aml_loop_detection/f<N>/ on the share when they exist; models whose run is not finished are skipped.
3. Cell-specific loops and stripes of the held-out cells (cell-agnostic object scoring; K562, HMEC deep, HUVEC deep;
   the evaluation table of step 2 of the preparation):

       python scripts/cell_objects_eval.py --stage windows                                                  # CPU
       python scripts/cell_objects_eval.py --stage predict --models base_refres diff_loopfocus_f10 diff_loopfocus_f11 diff_loopfocus_f12 --gpu 0
       python scripts/cell_objects_eval.py --stage score --models base_refres diff_loopfocus_f10 diff_loopfocus_f11 diff_loopfocus_f12
       # results/cell_objects_eval/auroc_by_model.csv: AUROC of the gain over consensus12, specific objects against matched
       # shared controls, per model, cell, kind, and class (all / private), bootstrap interval; amplitude; Spearman of the gain

## Success criteria (written before any run; 10 kb, ten held-out AML patients, donor bootstrap)

| # | criterion | target | reference value |
|---|---|---|---|
| 1 | AUROC of the gain over HSPC, AML-private loops against matched shared loops (A_hspc), model `aml_diff_loopfocus` | >= 0.70 (criteria.csv, class private, kind loop) | aml_refin 0.59 (section 62) |
| 2 | enhancer-like stripes (pair_type enh): Spearman of the model's gain over HSPC with the measured gain | above that of the reference map alone (criteria.csv, `enh_gain_spearman`, `pass_enh_gain_above_reference`) | reference map alone |
| 3 | median Pearson on the 413 Akita regions, 12 training and 5 held-out cells (reference_summary.csv) | no loss against the matching baseline beyond the honest seed spread: difference >= -0.006 (section 51: two-seed spreads of consensus-input models 0.0002 to 0.0062); use the seed replicates all12refres_drop_s1 / _s2 once they are in reference_summary.csv | all12refres_drop_10kb 0.9270 / 0.8972 (12 / 5 cells); the joint baseline all12aml_refres_drop_f<N> |
| 4 | cell-specific loops and stripes of the held-out cells (K562, HMEC, HUVEC) | AUROC of the gain with its interval reported per cell, kind and class, against `base_refres` | base_refres |

Further reported, no threshold: amplitude (median predicted / measured gain, private loops; section 63 target
0.30), the ablations (diff only, objects only) against the primary run, the Meth and HSPC variants.

## Estimated GPU time (remote 4090; measured speeds: 12 cells 25 kb about 10 min, 10 kb about 23 min on an idle GPU; AML 10 kb refin_drop 1.0 to 1.7 h, f10 100 min / f11 57 min / f12 66 min)

The channels add 52 input rows to the first convolution (a few percent per step) and 3 to 8 percent to the
resident window tensor; the all-cell objects add to the 7.5k pair windows of section 63 about 4k more (12 cells x
(400 loops + 200 stripes), about 40 percent of them inside cached training windows, x5 oversampling): per epoch
about 1.5 x an AML refin_drop epoch; epochs 1.0 to 1.5 x.

| job | estimate |
|---|---|
| tier 0, 12 cells alone: 25 kb 0.2 h and 0.25 h, 10 kb 0.4 h and 0.5 h | about 1.4 h in all |
| tier 1, primary 10 kb, per fold | f10 2.6 to 3.9 h, f11 1.5 to 2.3 h, f12 1.7 to 2.6 h; three folds about 6 h (range 5 to 9 h) |
| tier 1, primary 25 kb, per fold | 0.75 to 1.45 h; three folds about 3 h (2.2 to 4.5 h) |
| tier 2, diff only, 10 kb, three folds | about 4 h (3 to 6 h) |
| tier 2, objects only, 10 kb, three folds | about 5.5 h (4.5 to 8 h) |
| tier 3, diffmeth_loopfocus 10 kb / diffhspc_loopfocus 10 kb, three folds each | about 6.3 h / 6 h |
| standard cell-line evaluation per run (17 cells, inference) | about 0.3 h |
| aml_train_eval --variant extra, per fold, all models of the fold | about 0.3 h per model |
| aml_loop_detection predict (ten patients, 10 kb), all diff models | about 0.5 h per model |
| cell_objects_eval predict (3 cells, windows around 1000 loops and 1000 stripes each plus controls), per model | about 0.15 h |
| minimum useful set: tier 0 + tier 1 at 10 kb + their evaluations | about 9 h |
| tier 1 at 10 and 25 kb + tier 2 + evaluations | about 22 h |
| everything above | about 34 h |

Do not start while ~/seqcnn holds the GPU. Run jobs sequentially on the one 4090.

## Stop rules and risks

- Early epochs: obj_rank near 0.69 x weight is expected; if obj_pix does not fall within 5 epochs, stop and report.
- If the honest validation Pearson of a joint fold ends more than 0.01 below the all12aml_refres_drop run of the same
  fold, report before continuing with the other folds (criterion 3).
- Tier 0 decides how much to invest: if the cell-line-only run with the channels loses more than 0.006 median Pearson on the
  12 or 5 cells against all12refres_drop at the same resolution, report before the joint runs.
- One seed per run; the AML numbers are compared through the donor bootstrap intervals, not point values.
- The cell objects are defined from the cell's own map against its reference; maps of different depth give different numbers of
  candidates, hence the cap of 400 / 200 per cell. A shallow map's strongest "specific" loops can still be noise peaks.
- The channels move the same information the reference already carries for a cell whose marks equal the pool mean (difference
  near 0): a model that ignores them reproduces the section 45 residual model. The ablation runs separate the two components.
- Regional in-silico knockouts (`predict --zero-region`) act on the raw track only; the reference / difference rows keep the
  unperturbed values (warning in the log). Whole-track knockouts and keep sets blank them.
