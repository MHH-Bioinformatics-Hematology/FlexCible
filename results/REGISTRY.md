# Results registry

Every experiment behind the manuscript numbers, with the command that produced
it and the tables in this folder. All inputs, options and scripts are
committed; each approach stays selectable by config even when the final model
does not use it. Refresh the tables with `python scripts/collect_results.py`.

Metric throughout, unless stated: median over regions of the per-region
Pearson correlation between predicted and measured Hi-C, on the upper triangle
(offset >= 2 bins), on distance-normalized log(observed/expected) (O/E) at
2 kb. The 413 regions are Akita's test regions (`sequences_test_list.tsv`).
Status: FINAL (cite), SUPERSEDED (kept for the record, do not cite as a
result), PENDING (still running).

Validation protocol (methods note, 2026-09-28). Checkpoint selection and
early stopping use held-out validation windows of the training chromosomes.
Since 2026-09-28 `train.val_exclude_overlap` defaults to true: validation
windows are drawn by genomic position (every training cell's window at a
drawn position is a validation window) and every training window of any
cell sharing a bin with a validation position is removed. Before that date
the default was false, and validation windows overlapped training windows by
up to half a window (the stride), in multi-cell runs also through the other
cells' windows at the same position. Runs with overlapping validation:
sections 1, 2 (the elimination path itself is ranked by this validation
score), 3 and 4 (the section 1 and 4 models), 6, 8 (all runs except the
`__hv` runs), 9, 10, 11 and 13. Sections 14 and 15 use honest validation
(single-cell runs there have one sample, where the old same-sample
exclusion and the position rule coincide). The test scores of every section
come from windows and regions disjoint from all training windows (held-out
Akita valid and test regions, `data.holdout_bed`); section 15 verifies this
per cell. Configs of the earlier runs reproduce with an explicit
`val_exclude_overlap: false`, which every `configs/*.yaml` and every script
regenerating those runs now sets.

## 1. Comparison with Akita (FINAL)

Model `configs/akita_native.yaml` (13 ChIP tracks, CTCF motif, 2048 bp, Akita
O/E target), scored with `scripts/score_like_hicgan.py` against Akita's
published GM12878 target on the paired regions.

- `akita_comparison/score_like_hicgan.csv`: per region, ours vs Akita vs Hi-cGAN.
- 222 paired regions: ours 0.652, Akita 0.556; ours higher in 68% of regions.
  This model used overlapping validation (methods note above).

Honest validation and a reference map (2026-09-28). `scripts/akita_targets.py`
rebuilt Akita's published targets of all five cells on the 2048 bp grid from
the public 1m records (gs://basenji_hic/1m/data); the rebuilt GM12878 map
equals the section 1 test cool on all 413 test windows (maximum difference
0.0, `akita_targets_check.json`). Reference = consensus4, the per-window mean
of the HFF, H1, IMR90 and HCT116 targets (never GM12878). Models:
`configs/akita_native_honest.yaml` (the section 1 model with honest
validation, same cache and 30-minute schedule) and
`configs/akita_native_refres.yaml` (plus consensus4, residual). The
reference model trains on a cache built from the rebuilt train-fold target,
which holds 3762 of the 4424 section 1 training windows with identical
targets; the other 662 windows are more than 95% empty in the rebuilt map and
are dropped by the empty-window filter (`akita_cache_check.json`). Scored by
`scripts/score_like_hicgan.py --tta` against Akita's target, with the measured
maps of the other cells scored on the same pixels (tables
`akita_comparison/*.score.csv`).

| model | 222 paired regions | 413 regions | above Akita (222) | above consensus4 (222) |
|---|---|---|---|---|
| section 1 model, overlapping validation | 0.652 | 0.642 | 68% | 60% |
| section 1 model, honest validation | 0.640 | 0.636 | 66% | 55% |
| residual reference model (consensus4), honest | 0.744 | 0.752 | 96% | 99% |
| Akita (recorded) | 0.556 | 0.548 | | |
| consensus4 alone | 0.616 | 0.618 | | |
| best single other cell (HFF) | 0.519 | 0.525 | | |

With honest validation the ChIP model scores 0.640 on the 222 regions, 0.012
below the overlapping-validation checkpoint and 0.084 above Akita. The
four-cell consensus alone reaches 0.616, above Akita's 0.556, and the
residual reference model reaches 0.744 and exceeds Akita in 96% and
consensus4 in 99% of the regions.

## 2. Backward elimination of ChIP tracks (FINAL)

`scripts/backward_tracks.py --config configs/akita_native.yaml --cache
cache/akita_native --val-regions 100 --patience 10 --max-epochs 100
--full-path --gpus 0,1`. Validation: 100 held-out windows, median per-region
Pearson; every candidate trained to convergence (patience 10 epochs).

- `backward_elimination/backward.csv` (path), `backward_rankings.csv` (every
  candidate of every round, rank and value).
- Path (tracks kept: val): 13: 0.627, 12 (-H3K36me3): 0.636, 11 (-CTCF):
  0.642 (optimum), 10 (-H3K9ac): 0.640, 9: 0.619, 8: 0.628, 7: 0.629,
  6 (-SMC3): 0.616, 5: 0.608, 4: 0.611, 3: 0.585, 2: 0.590, 1 (RAD21 only):
  0.501. RAD21 is the last track kept; removing it in round 1 gives the
  lowest score of all 13 drops (0.607).

Proxy check (SUPERSEDED as a method, cite as a negative result): the same
search on chr2, chr3 and chr5 only (`--chroms chr2,chr3,chr5`), table
`backward_elimination/proxy_chr2_3_5_round1.csv`, compared with
`scripts/compare_search.py`. Round-1 ranking agreement with the full-data
search over 11 common drops: Spearman -0.14, Kendall -0.13. RAD21 ranks 4th
of 12 on the proxy and last on full data.

## 3. Redundancy between ChIP tracks (FINAL)

- `track_redundancy/track_correlation.csv` (`scripts/track_correlation.py`):
  2 kb Pearson between tracks. Clusters at |r| >= 0.7: RAD21 with SMC3 (0.74);
  H3K27ac, H3K4me1, H3K4me2, H3K4me3 and H3K9ac.
- `track_redundancy/summary.csv` (`scripts/track_substitution.py`, all-13
  model, 413 regions): swapping two tracks of one group changes the median by
  0.001 to 0.007; swapping across groups by 0.034 (CTCF with H3K27me3) and 0.151
  (RAD21 with H3K27ac). Omitting SMC3 costs 0.080, CTCF 0.041.
- `track_redundancy/combos.csv` (`scripts/queue_combos.py`): correlation-picked
  subsets score below backward-elimination subsets of the same size.

## 4. Prediction from fewer tracks (FINAL)

Model `configs/akita_native_trackdrop.yaml` (track dropout, uniform keep
count), scored with `scripts/subset_curve.py`, 413 regions.

- `track_dropout_subsets/summary.csv`: 13 tracks 0.636, 8: 0.621, 5: 0.607,
  3: 0.601, 2: 0.557, 1: 0.443. Training with dropout left the all-13
  validation score unchanged (0.628 vs 0.627).

## 5. Distance normalization of the target (FINAL, methods)

`scripts/oe_proper.py` computes expected(d) over all mappable pairs and
removes the per-diagonal log mean per chromosome. On its output a
distance-only predictor scores 0.28; on the older `akita_normalize.py` output
it scores 0.76, and on Akita's published target 0.155. Section 6 and later
use `oe_proper.py` targets.

## 6. Cross-cell prediction (FINAL)

Model `configs/full2k_proper.yaml` (2000 bp, track dropout, GM12878 trained,
validation 0.643), `scripts/crosscell_predict.py` with `--tta`. Each table
holds, per region, the model prediction from the cell's own ChIP and the
measured maps of the other cells scored against the same target.

| target cell | model (own ChIP) | best other measured map |
|---|---|---|
| GM12878 (training cell) | 0.683 | 0.528 (IMR90) |
| IMR90 (held out) | 0.598 | 0.528 (GM12878) |
| K562 (held out, aneuploid) | 0.410 | 0.517 (IMR90) |
| HMEC rep1, no RAD21/SMC3 ChIP | 0.208 | 0.294 (IMR90); HMEC rep2 0.415 |

The HMEC row uses a single-lane HMEC map from our own pipeline (171 M
contacts) as the target, so all its values are lower; rep2 shares that
processing. `predict_gm12878_2048check.csv`: the 2048 bp model scored against
its native target, 0.642, the reference for the resampling check.

SUPERSEDED: `crosscell_superseded_inflated/` holds the first cross-cell
matrix (0.76 to 0.82) computed on `akita_normalize.py` targets. The
distance-only predictor reaches 0.76 there, so these values measure the shared
distance decay. Do not cite.

## 7. Biological replicates, 5 cell lines, 2 to 25 kb (FINAL)

`scripts/replicates_all_cells.py` then `scripts/replicate_corr.py`: ENCODE
MboI in situ Hi-C, one lane-pair per replicate, minibwa, HiCExplorer v4
`hicBuildMatrix` at 2, 5, 10 and 25 kb, cooler ICE balancing, `oe_proper.py`,
rep1 vs rep2 on the 413 regions. Table `replicates/replicate_corr.csv`
(contacts per replicate from the 25 kb level).

| cell | contacts rep1 / rep2 (M) | 2 kb | 5 kb | 10 kb | 25 kb |
|---|---|---|---|---|---|
| GM12878 | 138 / 230 | 0.386 | 0.564 | 0.779 | 0.933 |
| IMR90 | 166 / 199 | 0.501 | 0.689 | 0.867 | 0.956 |
| K562 | 144 / 210 | 0.359 | 0.500 | 0.764 | 0.932 |
| HMEC | 171 / 167 | 0.317 | 0.526 | 0.780 | 0.936 |
| HUVEC | 112 / 117 | 0.332 | 0.429 | 0.697 | 0.888 |

At one sequencing lane per replicate (110 to 230 M contacts), replicate
agreement at 2 kb is 0.32 to 0.50 and rises to 0.89 to 0.96 at 25 kb. The maps
the model is scored against in sections 6 and 8 hold 0.9 to 4.0 billion
contacts, so these values bound replicate agreement at single-lane depth and
sit below the agreement expected for those deep maps. The first single-cell run
(`scripts/raw_hic_ceiling.sh`, HMEC, unbalanced) gave 0.415 at 2 kb.

## 8. Sequence channels x ChIP (FINAL)

`scripts/motif_grid.py` and `scripts/annot_grid.py` (proxy chromosomes chr1,
4, 7, 10, 13, 16, 19), `scripts/full_confirm.py` (all chromosomes, two seeds
for the reference), each checkpoint scored by `scripts/eval_val_windows.py` on
the validation windows and on the disjoint Akita test windows (211 proxy, 730
full). Configs `configs/full2k_motifs.yaml`, `configs/full2k_annot.yaml`.
Table `sequence_chip_grid/results_all.csv` (val and test, O/E and log-obs).

Full data, test (730 windows):

| model | test |
|---|---|
| all 13 ChIP + CTCF motif (2 seeds) | 0.674, 0.687 |
| all 13 ChIP + CTCF, YY1, ZNF143, MAZ motifs (2 seeds) | 0.695, 0.694 |
| no cohesin ChIP + CTCF motif | 0.657 |
| no cohesin ChIP + 4 motifs | 0.663 |
| no cohesin ChIP + 4 motifs + annotation | 0.649 |
| CTCF and cohesin ChIP only + 4 motifs | 0.591 |
| CTCF and cohesin ChIP only + 4 motifs + annotation | 0.626 |

Proxy test (211 windows): without ChIP, GC/CpG only 0.181, CTCF motif 0.377,
4 motifs 0.371. The learned 1 Mb DNA encoder raises validation scores and
lowers test scores in every ChIP set (e.g. no ChIP: validation 0.566, test
0.204), because validation windows overlap training windows by half a window
and the encoder memorizes position; with overlapping training windows removed
(`__hv` runs) it still lowers test scores. Seed-to-seed spread of one
configuration: 0.013 on full data.

Distance bias (`model.dist_bias`, proxy, all 13 + CTCF motif): test 0.634
with, 0.646 and 0.652 without. In decay-inflated log-obs space all models
score about 0.972 and a distance-only predictor 0.953
(`*_logobs_*` columns).

## 9. Post-hoc sharpening (FINAL)

`scripts/sharpen_cache.py`, `scripts/sharpen_eval.py` (tuned on chr1, 4, 7,
10, 13, 16; reported on the other 272 regions), model
`configs/full2k_proper.yaml`. Table `sharpening/sharpen_median.csv`.
No filter raises pixel Pearson (GM12878 0.701, IMR90 0.615 unfiltered). A
per-distance contrast gain learned on the tuning split raises the APA ratio
from 0.51 to 0.73 and fixed-threshold boundary F1 by 0.02 (GM12878) and 0.08
(IMR90), at a cost of 0.004 to 0.008 pixel Pearson.

## 10. Promoter capture Hi-C in training (FINAL)

Question: does GM12878 promoter capture Hi-C (Mifsud et al. 2015,
E-MTAB-2323) improve the 2 kb O/E prediction, as an auxiliary training
target (variant A) or as an optional input with modality dropout (variant B,
one checkpoint scored with and without capture input)?

Data. The published significant promoter-other interactions (1,618,000,
raw count and log(observed/expected) per HindIII fragment pair) and the
22,076-fragment HindIII bait design were lifted from hg19 to hg38 fragment by
fragment with the UCSC hg19ToHg38 chain (`scripts/lift_capture_hg38.py`,
output `~/data/hic-multiome/pchic/*.hg38.*`). 1,616,240 interactions (99.9%)
and 22,055 baits kept; 99.99% of 40,000 checked lifted fragment ends lie on an
hg38 HindIII site (AAGCTT), so the lifted intervals are the hg38 fragments and
the published scores carry over unchanged. The raw reads (six GM12878 runs,
884 M read pairs, 165 GB) were not reprocessed. The capture code bins the
scores by fragment midpoint onto the 2 kb grid (maximum per pixel), encodes
them as score / 99.9th percentile (12.39) mapped to [-1, 1] with -1 for no
significant interaction, and masks the capture loss to pixels whose row or
column holds a designed bait (`data.samples[].capture_baits`, 3.3% of the
upper-triangle pixels of a test window). `capture_lift/lift_capture_hg38.json`.

Runs. `python scripts/capture_experiment.py --variants A B --seeds 0 1`
(config `configs/full2k_capture.yaml`, cache `cache/full2k_capture`, whose Hi-C
windows and conditioning are identical to `cache/full2k_motifs`): the
reference model of section 8 (13 ChIP tracks, 4 motifs, no encoder, no
distance bias, track dropout 0, patience 10 epochs) plus a capture output
channel with loss weight 0.5 (A), or additionally the capture map and bait mask
as input planes, blanked in 50% of training windows, with the capture loss on
the blanked windows (B). Early stopping uses validation without capture input.
Scoring: `scripts/eval_capture_windows.py` on the 730 test windows, median
per-window Pearson on O/E over all upper-triangle pixels, over pixels touching
a bait and over the rest; the reference checkpoints are rescored on the same
windows (0.695 and 0.694, as in section 8).

| model | capture at test | all | bait pixels | non-bait pixels |
|---|---|---|---|---|
| reference, seeds 0, 1 | no | 0.695, 0.694 | 0.710, 0.711 | 0.696, 0.694 |
| A: auxiliary target, seeds 0, 1 | no | 0.694, 0.701 | 0.713, 0.718 | 0.697, 0.700 |
| B: input + auxiliary target, seeds 0, 1 | no | 0.698, 0.690 | 0.714, 0.711 | 0.697, 0.688 |
| B: input + auxiliary target, seeds 0, 1 | yes | 0.696, 0.688 | 0.717, 0.711 | 0.696, 0.686 |

All differences lie within the seed spread of 0.013. Paired over windows, the
same B checkpoint with capture input scores +0.0003 and +0.0002 higher per
window than without (bait pixels +0.0004 and +0.0011). Capture input changes
the predicted map by 0.005 on average (encoded scale, target standard
deviation 0.22), equally at bait and at other pixels. The capture map carries
little of the 2 kb O/E target: at bait pixels the per-window Pearson between
capture plane and Hi-C O/E is 0.11, and pixels with a significant capture
interaction exceed the other bait pixels by 0.07 encoded units, a third of a
target standard deviation. The auxiliary capture channel itself reaches a
per-window Pearson of 0.29 to 0.30 against the capture target on bait pixels.
Capture Hi-C is a contact measurement of the same cell, so at bait pixels
capture input partly supplies the answer; variant A and the non-bait pixels
are the comparisons free of that, and they show no gain either. Capture
supervision and capture input leave the GM12878 prediction unchanged at 2 kb.
Tables `capture/capture_summary.csv`, `capture/capture_input_diagnostics.json`
(`scripts/capture_input_diagnostics.py`).

## 11. Multi-cell training on Akita's cell types (FINAL)

`scripts/multicell_akita.py` (runs gm1, hic3, all5, all5assay), best
configuration (13 ChIP, 4 motifs, 2 kb, `oe_proper.py` targets), training
windows restricted to Akita's training sequences (`data.train_regions_bed`,
263 merged regions, 2009.5 Mb; removes 1327 of 4536 GM12878 windows), stopped
by patience (all5 continued past the 100-epoch cap, stopped at 193; all5assay
at 169). Tables `multicell_akita/test730.csv` (730 disjoint test windows),
`crosscell.csv` (413 regions, with every other cell's measured map on the same
regions), `training_stop.csv`, `train_windows.csv`.

Test windows (730), cells seen in training:

| cell | GM12878 only | 3 Hi-C cells | 5 cells | 5 cells + assay indicator |
|---|---|---|---|---|
| GM12878 | 0.685 | 0.696 | 0.678 | 0.680 |
| IMR90 | | 0.710 | 0.697 | 0.698 |
| HCT116 | | 0.650 | 0.630 | 0.626 |
| H1 (Micro-C) | | | 0.667 | 0.668 |
| HFF (Micro-C) | | | 0.760 | 0.757 |

Cells never seen in training (413 regions; best measured map of another cell
in brackets): K562 0.362, 0.463, 0.494, 0.448 (0.517, IMR90); HMEC 0.148,
0.219, 0.248, 0.229 (0.302, IMR90); HUVEC 0.183, 0.206, 0.229, 0.196 (0.239,
IMR90), for GM12878 only, 3 cells, 5 cells and 5 cells with the assay
indicator. HMEC and HUVEC targets are single-lane maps (section 7). With 3
cells, H1 (unseen) scores 0.609 against 0.594 for the best measured map (HFF).
Adding the two Micro-C cells lowers the Hi-C cells by 0.013 to 0.020; the
assay indicator (`data.assay_indicator`) does not recover it.

The GM12878-only model trained on all training windows (section 8) scores
IMR90 at 0.489 and HCT116 at 0.491, and the HFF map predicts IMR90 at 0.662,
so the IMR90 comparison in section 6 holds only against GM12878 and K562 as
measured baselines.

## 12. Deep replicate agreement, intact Hi-C (FINAL)

`scripts/replicate_corr_intact.py`: ENCODE per-replicate intact Hi-C .hic
converted with the in-house C++ hic2cool, ICE, `oe_proper.py`, rep A vs rep B
on the 413 regions. Table `replicates/replicate_corr_intact_deep.csv`. K562
ENCSR676BEP (0.97 and 1.45 billion contacts): 0.650 (2 kb), 0.865 (5 kb),
0.954 (10 kb), 0.990 (25 kb). The three GM12878 pairs (0.25 to 0.47 billion
contacts per replicate) give 0.46 to 0.52 at 2 kb and 0.90 to 0.97 at 25 kb.
The expected score of an error-free prediction against one replicate is about
the square root of the replicate agreement (0.81 at 2 kb for the K562 pair).

## 13. Expanded multi-cell training, 12 cell types, 2 to 25 kb (FINAL)

`scripts/encode_expand.py`, `scripts/multicell_expanded.py`,
`runs/multicell_expanded/driver.sh`: 7 further ENCODE cells (HepG2, A549,
MCF-7, NHEK, GM23248, CD14+ monocytes, PC-9) from ENCODE .hic via the C++
hic2cool, three-class assay indicator (in situ Hi-C, intact Hi-C, Micro-C),
models all12 at 2, 5, 10 and 25 kb and insitu7 at 2 kb.
Stopped by patience at epoch 106 (all12, 2 kb), 114 (5 kb), 107 (10 kb), 127
(25 kb) and 148 (insitu7). Tables `multicell_expanded/test730.csv`,
`crosscell.csv`, `scaling.csv`, `resolution_vs_replicates.csv`,
`training_stop.csv`; `resolution_inputs/resolution_compare_summary.csv`,
`averaging_ceiling.csv`.

All 12 training cells, 413 Akita test regions, median over cells:

| resolution | model | best measured map of another cell | model beats map | test windows | 2 kb model averaged |
|---|---|---|---|---|---|
| 2 kb | 0.656 | 0.581 | 11/12 | 0.649 | |
| 5 kb | 0.777 | 0.769 | 6/12 | 0.735 | 0.692 |
| 10 kb | 0.823 | 0.860 | 2/12 | 0.786 | 0.706 |
| 25 kb | 0.815 | 0.905 | 1/12 | 0.804 | 0.701 |

Held-out cells (413 regions, model / best measured map): K562 0.544 / 0.517,
0.653 / 0.723, 0.752 / 0.827, 0.767 / 0.866 at 2, 5, 10 and 25 kb; HMEC deep
0.461 / 0.588 and HUVEC deep 0.553 / 0.571 at 2 kb. The measured 2 kb GM12878
map block-averaged to 5, 10 and 25 kb scores 0.806, 0.812 and 0.815 against
the coarse targets, which bounds the averaged 2 kb model.

Unseen cells versus number of training cells (413 regions): K562 0.362,
0.463, 0.494, 0.551, 0.544 for 1, 3, 5, 7 (insitu7) and 12 cells; HMEC deep
0.355, 0.408, 0.504, 0.451, 0.461; HUVEC deep 0.234, 0.380, 0.494, 0.526,
0.553. insitu7 (the 7 in situ Hi-C cells) scores its training cells 0.006 to
0.016 above all12 on the test windows (GM12878 0.684 vs 0.670, IMR90 0.697 vs
0.691).

## 14. Input groups and window size per resolution (FINAL)

`scripts/resolution_inputs.py` (stages groups, combo, windows; driver
`runs/resolution_inputs/gpu_chain.sh`), GM12878 only, `cache_expanded`
targets, 2 seeds each, validation windows disjoint from training windows
(`train.val_exclude_overlap`). Groups: CTCF; cohesin (RAD21, SMC3); active
(H3K27ac, H3K4me1/2/3, H3K9ac); gene body (H3K36me3, H3K79me2, H4K20me1);
repressive (H3K27me3, H3K9me3); motifs (4). Windows of 384, 192 and 80 bins
at 5, 10 and 25 kb (1.92 Mb, 1.92 Mb, 2.0 Mb). Tables
`resolution_inputs/summary.csv` (seed means and per-seed values),
`runs.csv`.

Mean of 2 seeds, validation / 730 test windows / 413 Akita test regions:

| inputs | 5 kb | 10 kb | 25 kb |
|---|---|---|---|
| all | 0.802 / 0.784 / 0.802 | 0.842 / 0.824 / 0.838 | 0.823 / 0.825 / 0.826 |
| without CTCF | 0.797 / 0.781 / 0.800 | 0.818 / 0.800 / 0.821 | 0.798 / 0.811 / 0.810 |
| without cohesin | 0.787 / 0.772 / 0.781 | 0.812 / 0.805 / 0.798 | 0.801 / 0.823 / 0.817 |
| without active | 0.790 / 0.779 / 0.799 | 0.823 / 0.811 / 0.817 | 0.808 / 0.816 / 0.809 |
| without gene body | 0.804 / 0.788 / 0.801 | 0.837 / 0.824 / 0.827 | 0.815 / 0.835 / 0.829 |
| without repressive | 0.790 / 0.789 / 0.802 | 0.834 / 0.815 / 0.818 | 0.830 / 0.827 / 0.822 |
| without motifs | 0.798 / 0.782 / 0.793 | 0.833 / 0.816 / 0.818 | 0.823 / 0.817 / 0.822 |
| repressive + gene body | 0.650 / 0.640 / 0.620 | 0.704 / 0.686 / 0.660 | 0.676 / 0.686 / 0.655 |
| sequence only | 0.446 / 0.423 / 0.501 | 0.514 / 0.469 / 0.513 | 0.452 / 0.471 / 0.530 |

Removing CTCF or cohesin lowers every resolution, by up to 0.040 on the
Akita test regions at 10 kb. Repressive and gene-body marks alone reach 0.62
to 0.70 at every resolution. The single-group removals of gene body,
repressive marks and motifs change the scores by at most 0.02, within the
seed spread (up to 0.048 between the two seeds at 25 kb). The selection
rule (best seed-mean validation) picked without gene body at 5 kb, all at
10 kb and without repressive at 25 kb; the 25 kb combination without both
scores 0.813 / 0.811 / 0.811.

Window size, chosen inputs, same metrics: 5 kb 256 bins 0.788 / 0.790 /
0.804 (384 bins 0.804 / 0.788 / 0.801); 10 kb 256 bins 0.821 / 0.825 / 0.822
(192 bins 0.842 / 0.824 / 0.838); 25 kb 160 bins 0.831 / 0.833 / 0.804 and
256 bins 0.826 / 0.823 / 0.765 (80 bins 0.830 / 0.827 / 0.822). The Akita
test regions span about 1 Mb, so wider windows place them in a smaller share
of the context.

## 15. Closing the coarse-resolution gap (GM12878 FINAL, 12 cells PENDING)

Question: at 10 and 25 kb the GM12878 model of section 14 (0.838 and 0.826
on the 413 Akita test regions) scores about as well as the measured map of
another cell (best single map 0.799 and 0.844, K562), while at 2 kb the
model wins (section 13). Does a reference map of other cells as input close
the gap?

Reference map. `scripts/consensus_oe.py`: pixel-wise mean of the proper O/E
maps of IMR90, HFF, H1, HCT116, HepG2, A549, NHEK, GM23248, MCF-7, CD14+
monocytes and PC-9 (consensus11; cell list in `<cool>.json`). It excludes
GM12878 and the held-out K562, HMEC and HUVEC. Options `data.reference_cool`
(or per sample `samples[].reference_cool`) and `model.reference`: input (one
extra 2D conditioning plane, A1 "refin") or residual (input plane, and the
output is reference plus network output, A2 "refres"); default none.
`hicmultiome/reference.py` rejects any reference whose recorded cells include
the trained or scored cell (precompute, training, eval_val_windows,
crosscell_predict, consensus baseline columns).

GM12878 runs. `scripts/coarse_gap.py` (driver `runs/coarse_gap/driver.sh`,
then `driver_gm_ref.sh`), protocol of section 14 (13 ChIP tracks, 4 motifs,
80 and 192 bin windows, 100 honest validation windows, patience 10, seeds 0
and 1). Tables `coarse_gap/summary.csv` (seed means and seed values),
`runs.csv`, `heldout.csv`, `maps_gm12878.csv`. Baseline = the section 14
"all" runs. Reference alone = consensus11 scored as the prediction on the
identical windows, regions and pixels.

Validation / 311 or 295 test windows / 413 Akita test regions, mean of 2
seeds (seed values in brackets for the test regions):

| model | 25 kb | 10 kb |
|---|---|---|
| baseline (section 14) | 0.823 / 0.825 / 0.826 (0.840, 0.813) | 0.842 / 0.824 / 0.838 (0.836, 0.840) |
| reference alone | 0.857 / 0.848 / 0.882 | 0.842 / 0.819 / 0.858 |
| A1 reference input | 0.935 / 0.930 / 0.931 (0.931, 0.931) | 0.929 / 0.924 / 0.933 (0.933, 0.933) |
| A2 reference residual | 0.939 / 0.937 / 0.935 (0.937, 0.933) | 0.930 / 0.924 / 0.932 (0.932, 0.933) |
| best single measured map (K562) | 0.844 | 0.799 |

At 25 kb the 295 test windows are the disjoint windows of that cache (311 at
10 kb). With the reference map as input the model reaches 0.93 at both
resolutions, 0.09 to 0.11 above the baseline and 0.05 to 0.08 above the
reference alone on the same regions. Paired over the 413 regions the
residual model exceeds consensus11 in 82% of the regions at 25 kb (median
difference 0.045) and in 94 to 95% at 10 kb (0.065). Input and residual
variants differ by at most 0.007. GM12878 single-lane replicates agree at
0.779 (10 kb) and 0.933 (25 kb), deep intact replicate pairs at 0.90 to 0.97
(25 kb, section 12).

Held-out cells, GM12878-trained models, 413 regions, reference consensus11
(seeds 0, 1; GM12878-only models without track dropout, HMEC and HUVEC lack
RAD21 and SMC3 ChIP):

| cell | res | A2 residual | A1 input | baseline | consensus11 alone | best single map |
|---|---|---|---|---|---|---|
| K562 | 25 kb | 0.896, 0.886 | 0.887, 0.884 | 0.692, 0.687 | 0.893 | 0.866 (HCT116) |
| HMEC deep | 25 kb | 0.916, 0.902 | 0.904, 0.893 | 0.523, 0.563 | 0.930 | 0.921 (NHEK) |
| HUVEC deep | 25 kb | 0.906, 0.900 | 0.897, 0.892 | 0.623, 0.713 | 0.911 | 0.882 (IMR90) |
| K562 | 10 kb | 0.879, 0.877 | 0.871, 0.871 | 0.689, 0.686 | 0.854 | 0.827 (HCT116) |
| HMEC deep | 10 kb | 0.898, 0.901 | 0.814, 0.825 | 0.570, 0.473 | 0.905 | 0.865 (NHEK) |
| HUVEC deep | 10 kb | 0.892, 0.893 | 0.800, 0.835 | 0.644, 0.596 | 0.883 | 0.835 (IMR90) |

On held-out cells the residual model scores above the best single measured
map in every case except HMEC at 25 kb (0.916 and 0.902 against 0.921 for
NHEK), and within 0.03 of consensus11: above it for K562 and HUVEC at 10 kb
and for K562 with seed 0 at 25 kb, below it for HMEC at both resolutions and
HUVEC at 25 kb. Without the reference the GM12878 model scores 0.47 to 0.71
on these cells.

Not run (plan change 2026-09-28): the 5 kb reference runs (consensus11 at
5 kb was built, `crosscell_oe/consensus11_proper_5kb.cool`), idea B
(quarter-stride windows `window.stride = size / 4` and a narrow U-Net
`model.base_channels` 24, variants q4, narrow, q4narrow in
`scripts/coarse_gap.py`) and idea C (coarse targets block-averaged from the
2 kb map, `scripts/coarsen_oe.py`, targets
`crosscell_oe/gm12878_avg2kb_{10,25}kb.cool` built, variant avg2kb). The code
of both stays selectable; no model was trained.

12-cell training with the reference map (PENDING).
`scripts/multicell_expanded.py` runs all12refres (residual) and all12honest
(the all12 configuration without reference), both with honest validation, at
25 and 10 kb, DDP on two GPUs, driver `runs/multicell_reference/driver.sh`,
tables `multicell_reference/reference_compare.csv`,
`reference_summary.csv`, `training_stop.csv`, `split_disjoint.csv`. The
all12 configuration of section 13 at these resolutions has no assay
indicator (`assay_decision.json` chose all12), and the new runs keep it.
Each training cell reads an 11-cell leave-one-out consensus (the other
training cells, `consensus_loo_<cell>_proper_<k>kb.cool`); the held-out
cells K562, HMEC and HUVEC read the consensus of all 12 training cells
(`consensus12_proper_<k>kb.cool`). Validation: 100 positions, 1200
windows (25 kb; 101 positions, 1211 windows at 10 kb), 100 per cell, 1194
(1269) training windows per cell after removing the 1872 (1955) windows at
validation positions. `scripts/check_split_disjoint.py`: no training window
of any cell overlaps one of the 413 Akita test regions or one of the test
windows, in the 12-cell caches at 25 and 10 kb and the GM12878 caches of
this section; as a control, 148 of the 295 GM12878 test windows at 25 kb
overlap an Akita test region under the same check. A first all12refres run
at 25 kb with overlapping validation was stopped at epoch 52
(`all12refres_25kb.overlapval*`) and is not reported.

Results at 25 kb (all12refres stopped by patience at epoch 38, honest
validation 0.9445; all12honest at epoch 27, 0.8349). 413 Akita test regions
per cell, reference alone = the cell's leave-one-out consensus (training
cells) or consensus12 (held-out cells) scored on identical pixels; last
column: disjoint test windows, model / reference alone / all12honest:

| cell | all12refres | reference alone | best other map | all12honest | all12 (section 13, overlapping validation) | test windows |
|---|---|---|---|---|---|---|
| gm12878 | 0.938 | 0.882 | 0.844 (k562) | 0.822 | 0.815 | 0.940 / 0.848 / 0.828 |
| imr90 | 0.955 | 0.928 | 0.931 (gm23248) | 0.881 | 0.872 | 0.960 / 0.924 / 0.855 |
| hct116 | 0.952 | 0.932 | 0.910 (a549) | 0.839 | 0.823 | 0.959 / 0.923 / 0.833 |
| hff | 0.961 | 0.932 | 0.940 (gm23248) | 0.890 | 0.879 | 0.967 / 0.934 / 0.867 |
| h1 | 0.937 | 0.912 | 0.899 (hct116) | 0.819 | 0.825 | 0.944 / 0.908 / 0.819 |
| hepg2 | 0.938 | 0.906 | 0.871 (a549) | 0.833 | 0.816 | 0.945 / 0.895 / 0.835 |
| a549 | 0.949 | 0.946 | 0.910 (hct116) | 0.864 | 0.844 | 0.962 / 0.941 / 0.855 |
| nhek | 0.939 | 0.927 | 0.921 (hmec_deep) | 0.809 | 0.782 | 0.945 / 0.927 / 0.768 |
| gm23248 | 0.948 | 0.939 | 0.940 (hff) | 0.786 | 0.777 | 0.954 / 0.938 / 0.733 |
| mcf7 | 0.930 | 0.918 | 0.893 (nhek) | 0.835 | 0.809 | 0.934 / 0.916 / 0.793 |
| cd14mono | 0.879 | 0.668 | 0.712 (gm12878) | 0.807 | 0.806 | 0.898 / 0.677 / 0.823 |
| pc9 | 0.916 | 0.907 | 0.889 (a549) | 0.821 | 0.813 | 0.926 / 0.902 / 0.810 |
| k562 | 0.906 | 0.897 | 0.866 (hct116) | 0.771 | 0.767 |  /  /  |
| hmec | 0.873 |  |  | 0.691 | 0.677 |  /  /  |
| huvec | 0.859 |  |  | 0.763 | 0.738 |  /  /  |
| hmec_deep | 0.910 | 0.932 | 0.921 (nhek) | 0.735 | 0.698 |  /  /  |
| huvec_deep | 0.894 | 0.916 | 0.882 (imr90) | 0.805 | 0.781 |  /  /  |

Across the 12 training cells the residual reference model scores a median
of 0.939, against 0.922 for the leave-one-out consensus alone, 0.905 for the
best single other map and 0.828 for all12honest; it exceeds the consensus
alone in all 12 cells (paired over regions in 56% to 95% of the regions,
most for CD14+ monocytes, 0.879 against 0.668) and all12honest in all 12.
On the test windows the medians are 0.945, 0.919 and 0.826. Of the held-out
deep maps the model exceeds consensus12 for K562 (0.906 against 0.897) and
stays below it for HMEC (0.910 against 0.932) and HUVEC (0.894 against
0.916); all12honest scores 0.771, 0.735 and 0.805 on them. With honest
validation the no-reference 12-cell model scores 0.828 on the training
cells, 0.013 above the section 13 all12 run (0.815).

Results at 10 kb (all12refres stopped by patience at epoch 67, honest
validation 0.9181; all12honest at epoch 31, 0.8146), same columns; for the
held-out cells the reference is consensus12 (the single-lane HMEC and HUVEC
maps included):

| cell | all12refres | reference alone | best other map | all12honest | all12 (section 13, overlapping validation) | test windows |
|---|---|---|---|---|---|---|
| gm12878 | 0.938 | 0.858 | 0.799 (k562) | 0.835 | 0.825 | 0.931 / 0.819 / 0.822 |
| imr90 | 0.948 | 0.903 | 0.903 (gm23248) | 0.881 | 0.880 | 0.947 / 0.909 / 0.845 |
| hct116 | 0.950 | 0.910 | 0.869 (a549) | 0.835 | 0.841 | 0.942 / 0.899 / 0.821 |
| hff | 0.959 | 0.917 | 0.913 (gm23248) | 0.884 | 0.884 | 0.955 / 0.920 / 0.854 |
| h1 | 0.932 | 0.886 | 0.856 (hct116) | 0.835 | 0.830 | 0.926 / 0.879 / 0.798 |
| hepg2 | 0.926 | 0.874 | 0.831 (hct116) | 0.825 | 0.822 | 0.920 / 0.871 / 0.802 |
| a549 | 0.938 | 0.913 | 0.869 (hct116) | 0.838 | 0.833 | 0.930 / 0.907 / 0.814 |
| nhek | 0.918 | 0.888 | 0.865 (hmec_deep) | 0.779 | 0.773 | 0.906 / 0.884 / 0.728 |
| gm23248 | 0.943 | 0.925 | 0.913 (hff) | 0.787 | 0.771 | 0.940 / 0.917 / 0.741 |
| mcf7 | 0.909 | 0.879 | 0.832 (h1) | 0.814 | 0.807 | 0.893 / 0.868 / 0.764 |
| cd14mono | 0.858 | 0.626 | 0.662 (gm12878) | 0.775 | 0.772 | 0.856 / 0.638 / 0.768 |
| pc9 | 0.894 | 0.852 | 0.820 (a549) | 0.788 | 0.787 | 0.889 / 0.859 / 0.756 |
| k562 | 0.885 | 0.859 | 0.827 (hct116) | 0.750 | 0.752 |  /  /  |
| hmec | 0.806 | 0.809 | 0.768 (nhek) | 0.613 | 0.621 |  /  /  |
| huvec | 0.768 | 0.769 | 0.728 (imr90) | 0.659 | 0.648 |  /  /  |
| hmec_deep | 0.900 | 0.906 | 0.865 (nhek) | 0.703 | 0.713 |  /  /  |
| huvec_deep | 0.881 | 0.884 | 0.835 (imr90) | 0.791 | 0.770 |  /  /  |

Across the 12 training cells the residual reference model scores a median
of 0.935, against 0.887 for the leave-one-out consensus alone, 0.860 for the
best single other map and 0.830 for all12honest (test windows 0.928, 0.882
and 0.800); it exceeds the consensus alone in all 12 cells, in 77% to 98% of
the regions per cell. On held-out cells it exceeds consensus12 for K562
(0.885 against 0.859) and matches it within 0.007 for HMEC and HUVEC
(deep maps 0.900 against 0.907 and 0.881 against 0.884, single-lane maps
0.806 against 0.809 and 0.769 against 0.769); all12honest
scores 0.61 to 0.79 on them. With honest validation the no-reference model
scores 0.830 on the training cells against 0.823 for the section 13 run.

## 16. Worst-predicted regions (FINAL)

`scripts/worst_regions.py` (CPU only; stages predict, collect, reps,
features, model, metrics, fixes, chipref, tables, figures; outputs
`runs/worst_regions/`, tables `worst_regions/`). Models: the 12-cell
residual reference model at 25 kb (section 15, all12refres_25kb) and the
all12 2 kb model (section 13), re-predicted on CPU with `--save-pred` (the
GM12878 per-region scores reproduce the crosscell table to 1e-4). Region
score as in `crosscell_predict.py`; bottom 10% = 41 of 413 regions, ranked
by the median over the 12 training cells.

Worst regions at 25 kb (median over all regions 0.940): chr5:101449728
0.697, chr4:32845824 0.706, chr12:37810176 0.748, chr2:240488448 0.795,
chr5:61294592 0.805, chr17:48642048 0.820 (`worst_25kb.csv`). Worst at 2 kb
(median 0.656): chr4:31272960 0.257, chr1:188526592 0.261, chr1:190623744
0.279, clustered in chr4:28.6 to 33.9 Mb, chr1:188 to 192 Mb and chr7:52 to
54.6 Mb (`worst_2kb.csv`). The two bad sets are unrelated (Spearman 0.21
between the all12 2 and 25 kb scores, 7% of the bottom 10% shared). Across
cells the 25 kb scores agree with Spearman 0.42 and two cells share 26% of
their bottom 10%; the 25 kb model and the consensus it reads share 46%
(`consistency.csv`).

Target-side and model-side failures, from independent intact Hi-C
(GM12878 ENCSR965OVH, K562 ENCSR676BEP) scored against the in situ target
(`fix_eval_rules.csv`). At 2 kb the independent replicate agrees with the
target at 0.098 (GM12878) and 0.093 (K562) in the model's bottom 10%,
against 0.414 and 0.545 over all regions, and 85% and 90% of these regions
lie in the lowest replicate quartile: the 2 kb failures are target noise.
These are gene-poor B-compartment stretches with 65 raw contacts per 25 kb
pixel against 117 elsewhere and H3K27ac 1.2 log2 units below the genome
mean (region-level random forest CV R2 0.78; target SD Spearman 0.75). At
25 kb the replicate agrees at 0.885 (GM12878) and 0.936 (K562) in the
bottom 10% (0.955 and 0.965 overall) while the model scores 0.763 and
0.675; 39% and 15% of these regions have replicate agreement below 0.85.
The 25 kb failures are mostly model-side, on flat and cell-specific maps:
the share of pixel variance between the 12 cells (Spearman -0.60; 0.48 in
the bottom 10% against 0.30) and the target SD (0.54; 0.278 against 0.380)
explain most of the score (region-level random forest CV R2 0.50, ridge
0.60; cell specificity alone 0.40, target-map features alone 0.36, genome
annotation 0.03, ChIP and compartment below 0, window position 0). Cancer
lines score as well as the others (median 0.934 against 0.932) and coverage
steps do not predict the score; pericentromeric (chr12:37.8 Mb, 28.5%
segmental duplication) and subtelomeric (chr2:240.5 Mb) regions are single
cases (`feature_groups.csv`, `feature_univariate.csv`).

Metric sensitivity (`metric_variants_summary.csv`), 25 kb, 15 cells,
overall / bottom 10%: crosscell score 0.933 / 0.855, upper triangle k >= 2
0.933 / 0.855, k >= 10 0.932 / 0.842, unmappable bins and a 2-bin halo
masked 0.935 / 0.859, copy-number segments masked 0.934 / 0.859, Spearman
0.926 / 0.842, stratum-adjusted correlation 0.893 / 0.804. Pearson variants
keep 80 to 100% of the bottom 10%, SCC keeps 29%
(`metric_badset_stability.csv`). At 2 kb masking changes the bottom 10% by
at most 0.021.

Fixes tested on the saved predictions. Blending the prediction with its
consensus input, (1 - a) P + a C, raises the training cells from 0.9387 to
0.9417 (a = 0.3; bottom 10% 0.859 to 0.872) and the held-out cells K562,
HMEC deep and HUVEC deep from 0.9056 to 0.9166 (a = 0.5; bottom 10% 0.823
to 0.847); a was read on the test regions and is not a result until chosen
on validation windows (`fix_blend_summary.csv`). A reference map weighting
the other cells by ChIP profile similarity (softmax, tau 0.1) scores 0.9249
against 0.9223 for their plain mean (bottom 10% 0.846 against 0.833); the
single most ChIP-similar cell scores 0.884 (`chipref_summary.csv`).
Excluding the 41 regions with median target SD below 0.25 moves the
12-cell median from 0.9387 to 0.9411.

## 16. Consensus carried by the model (embedded FINAL, learned PENDING)

Question: can the only cell-specific input at prediction time be the cell's
ChIP (with sequence and motif channels), with the consensus Hi-C carried by
the model? Script `scripts/learned_reference.py`, tables
`learned_reference/learned_reference.csv` and the JSON files beside it.

Embedded consensus. `model.embed_reference` (default off) and
`scripts/embed_reference.py` store a reference map in the checkpoint
(`embedded_reference.cool` with a .json of its cells, source, bin size and
chromosome sizes); `predict.py` and `crosscell_predict.py` look up each
window in it by chromosome name and bin, so the bigWigs must be on the
map's assembly (hg38 for every map here; `open_reference` checks chromosome
sizes against the prediction grid). Training is unchanged (per-cell
leave-one-out maps); a training cell keeps its own leave-one-out map at
prediction, and the leakage guard refuses the embedded map for any cell it
contains. consensus12 embedded into all12refres_25kb: `predict.py` on
chr2 of K562 gives identical maps with the embedded and the explicit
consensus12 (maximum absolute difference 0.0 over 1,159,616 pixels); on the
413 regions `crosscell_predict.py` on the GPU differs by at most
1.6e-5 (K562, HMEC, HUVEC), and on the CPU by 0.0 (K562), so the GPU
difference is floating-point nondeterminism.

Learned consensus from sequence, stage 1. A model with GC/CpG and the four
motif planes only, trained on consensus12 (all12 configuration, honest
validation), 2-fold by chromosome for out-of-fold maps (odd chromosomes;
even chromosomes and chrX), plus an all-chromosome model for comparison.
Against consensus12 on the 413 Akita test regions and the disjoint test
windows: out-of-fold map 0.552 / 0.557 at 25 kb and 0.559 / 0.547 at 10
kb; all-chromosome model 0.649 / 0.577 and 0.616 / 0.579. Sequence alone
recovers the consensus at 0.55 to 0.65, well below the measured consensus
it approximates. Stage 2 (all12reflearn, the residual model with the
out-of-fold learned map as reference) is queued after the 2 kb pair.

## 17. Multi-resolution training (PENDING)

A. Resolution-conditioned shared U-Net, run all12multires
(`hicmultiome/train_multires.py`, `scripts/multires.py`): one residual
reference model (12 training cells, per-cell leave-one-out consensus,
consensus12 for held-out cells, honest validation per resolution) trained on
2, 5, 10 and 25 kb windows together. `model.resolutions` gives every
resolution a learned 16-dimensional embedding that modulates every U-Net
block by FiLM (scale and shift per channel, zero-initialised). Windows keep
the sizes of the dedicated models (512, 384, 192 and 80 bins); the U-Net is
fully convolutional, so each step draws one resolution uniformly (the same
draw on both DDP ranks) and a batch of its windows, which gives every
resolution the same number of gradient steps; an epoch is as many steps as
one pass over all training windows, and the mean of the four validation
scores drives patience. No torch.compile (four window shapes). Per
resolution the model is exposed as a run folder
(`multicell_reference/all12multires[_<k>kb]`) and evaluated exactly like the
dedicated runs (`reference_summary.csv`).

C. Scale-dependent input importance (`scripts/scale_importance.py`) on
model A: per resolution, the score loss when one input group is set to its
absent value (CTCF, cohesin, active, gene body, repressive, motifs, GC/CpG,
reference plane) on the disjoint test windows, and gradient x input shares
per group and distance band (<100 kb, 100 to 500 kb, >500 kb); figure
`scale_importance/importance.png`.

B. Coarse-to-fine cascade 25, 10, 5, 2 kb (`scripts/cascade.py`, run
all12casc, `model.prior`): the coarser level's prediction, resampled onto
the finer window, as one more input plane of the residual reference model.
Priors are out of fold, 2-fold by chromosome: two fold models per coarse
level (odd chromosomes; even chromosomes and chrX; the cascade setup below
25 kb) each predict the other fold's chromosomes for every cell, and the
merged maps (`crosscell_oe/cascade/prior_<cell>_<k>kb.cool`) feed both the
next level's fold models and its final model trained on all training
chromosomes.

All three run under `scripts/gpu_scheduler.py`; results follow here.

Model A training: stopped by patience at epoch 50 (665,050 steps), best
mean validation 0.8729 at epoch 40, with honest validation per resolution
of 0.758 (2 kb), 0.860 (5 kb), 0.923 (10 kb) and 0.951 (25 kb). The
validation windows are those of the dedicated runs (same caches, seed and
split): all12refres reached 0.9445 at 25 kb and 0.9181 at 10 kb, so on
validation sharing raises 25 kb by 0.006 and 10 kb by 0.005; the dedicated
5 and 2 kb reference models for the same comparison are queued. Test
scores follow from the evaluation now running. The training windows were
staged once as float16 memory maps on local disk
(`hicmultiome.train_multires --stage-only`); the first start, in which
each rank decompressed its stripe from the network share, was stopped
after 45 minutes of loading.

Analysis C on model A (`results/scale_importance/`): median over cells of
the per-window Pearson on the disjoint test windows with all inputs, and
the median per-cell loss when one group is set to its absent value
(`importance_drop.csv`; gradient x input shares per distance band in
`importance_attr.csv`, figure `importance.png`):

| dropped group | 2 kb | 5 kb | 10 kb | 25 kb |
|---|---|---|---|---|
| none (score) | 0.764 | 0.864 | 0.924 | 0.946 |
| CTCF | 0.038 | 0.041 | 0.039 | 0.033 |
| cohesin | 0.100 | 0.052 | 0.042 | 0.033 |
| active | 0.114 | 0.100 | 0.115 | 0.121 |
| gene body | 0.033 | 0.028 | 0.022 | 0.023 |
| repressive | 0.034 | 0.037 | 0.031 | 0.032 |
| motifs | 0.000 | 0.000 | 0.000 | 0.004 |
| sequence | 0.102 | 0.108 | 0.090 | 0.066 |
| reference map | 0.276 | 0.301 | 0.313 | 0.350 |

The reference plane carries most of the score at every resolution (loss
0.28 at 2 kb to 0.35 at 25 kb); among the inputs the active marks cost most
(0.10 to 0.12), then GC/CpG (0.07 to 0.11) and cohesin, which matters most
at 2 kb (0.10) and least at 25 kb (0.03); CTCF, gene-body and repressive
marks cost 0.02 to 0.04 each. The motif planes cost 0.000 to 0.004. This is
no wiring error: the dropped rows are the model's motif rows (the fwd/rev
rows at the motif offset that also build the directed pair planes), all of
their values change, and the prediction changes (25 kb: maximum 0.35, mean
0.016 absolute, against a prediction standard deviation of 0.24), but the
change is almost uncorrelated with the target. The reference map makes the
motifs redundant: on 1500 test windows the motif-drop loss (median paired
per window) is 0.029 at 25 kb and 0.014 at 10 kb for all12honest (no
reference), against 0.002 and 0.0002 for all12refres and 0.001 for model A
at 25 kb. CTCF sites are largely cell-invariant, so the consensus already
holds the loops and boundaries they anchor; section 8 measured the motif
gain on GM12878 models without a reference.

## 18. Head-to-head benchmark against published predictors (FINAL)

Question: how do our models compare with Akita, Hi-cGAN, Epiphany, HiC-Reg
and C.Origami on identical regions, pixels and targets?

Predictions. Every competitor is scored from predictions that already exist
(none was rerun): Akita's released GM12878 and HFF predictions and the Hi-cGAN
(submitted, under review) 2 kb held-out prediction, both in the Akita-space
raster `akita_testset_rasters.npz`; Hi-cGAN GM12878 whole-genome predictions
of the Hi-cGAN Zenodo record 02 (10 kb CTCF+H3K4me2 window 512, 25 kb
H3K27ac+CTCF window 256, hg19, trained on the odd chromosomes) and its
cross-cell-type predictions (records 05 and 06: the GM12878 model on K562,
HMEC, HUVEC and IMR-90 inputs, the IMR-90 model on IMR-90); Epiphany with its
released GM12878 weights (hg38, 10 kb, chr2, 3, 11, 17; chr3, 11 and 17 held
out); C.Origami with its released IMR-90 model (hg38, 8192 bp; chr2 test,
chr3 validation); HiC-Reg's published GM12878 predictions (hg19, 25 kb,
1 Mb reach; training split undocumented, and HiC-Reg's own protocol splits
bin pairs within a chromosome). Ours: akita_native_honest and
akita_native_refres (section 1) at 2 kb, all12 (section 13) for HFF at 2 kb,
all12refres and all12honest (section 15) at 10 and 25 kb, all predicted on
the CPU with `--save-pred` and test-time reverse averaging.

Targets and normalization. Block A uses Akita's own target space (Akita's
GM12878 and HFF targets, 448-bin crop of the 2048 bp grid, offset 2), the
only space in which Akita's and the Hi-cGAN 2 kb predictions exist; our 2000
bp all12 prediction is resampled bilinearly onto Akita's bin centres. Blocks B
to E score against our proper O/E targets (`oe_proper.py`, 4DN GM12878,
IMR-90 and K562 maps, ENCODE HMEC and HUVEC deep maps). Each competitor's
predicted matrix goes through the same `oe_proper.py` (expected over mappable
pairs, log O/E, per-diagonal centring, clip, Gaussian sigma 2 bins), limited
to its own distance reach (C.Origami log(count+1) back to counts first), and
is sampled bilinearly at the target's bin centres; hg19 predictions are
sampled at the target bin centres lifted with the UCSC hg38ToHg19 chain, and
a region is dropped when its lifted centres are not collinear. A region
enters a block only when every model and competitor covers it; metrics use
the pixels finite in the target and in every method (upper triangle, at least
2 bins off the diagonal, within the shortest reach in the block). Regions:
the 413 Akita test regions, restricted per block to chromosomes held out by
the competitors (block A all; B Epiphany: chr3, 11, 17; B, C and E even
chromosomes and chrX, held out by Hi-cGAN; D: chr2, chr3).

Metrics (`scripts/benchmark_metrics.py`, `scripts/benchmark_blocks.py`):
median per-region Pearson with its interquartile range; HiCRep SCC (per
diagonal, rank-variance weights, `scripts/scc_akita_regions.py` is the
stand-alone Akita-space version); insulation-profile Pearson; boundary
recall, the share of reference boundaries (HiCExplorer v4 C++ hicFindTADs on
the measured count map, 10 kb, depth 30 to 100 kb, FDR 0.05; 25 kb calls for
25 kb blocks) matched within 1 bin (5 bins at 2048 bp) by the K most
prominent insulation minima of a map, K the number of reference boundaries in
the region; loop recall, the share of reference loops (hicDetectLoops of
HiCExplorer 3.7.6 on the measured 10 kb map, peak interaction threshold 0,
other defaults) matched by the K strongest donut-enrichment peaks of a map
(bioscore, 2048 bp maps coarsened 5-fold), the APA ratio (prediction peak
enrichment over the measured one at the same anchors) and the anchor AUROC
against distance-matched control pixels; per-distance Pearson; paired
Wilcoxon tests and win rates per region. The measured map of every other
cell and their leave-one-out consensus (consensus12 for held-out cells,
consensus4 in block A) are scored the same way. The v4 C++ hicDetectLoops
aborts with a double free in `neighborhood_merge` on the 4DN GM12878 map, so
the loop calls come from the released Python tool.

| block | method | regions | Pearson median (IQR) | SCC | insulation r | boundary recall | loop recall | APA ratio | anchor AUROC |
|---|---|---|---|---|---|---|---|---|---|
| A GM12878 2 kb | Akita | 411 | 0.551 (0.356 to 0.696) | 0.512 | 0.686 | 0.494 | 0.314 | 0.26 | 0.917 |
| A GM12878 2 kb | akita_native_honest | 411 | 0.640 (0.455 to 0.741) | 0.611 | 0.819 | 0.547 | 0.394 | 0.38 | 0.933 |
| A GM12878 2 kb | akita_native_refres | 411 | 0.746 (0.594 to 0.837) | 0.698 | 0.920 | 0.648 | 0.574 | 0.85 | 0.977 |
| A GM12878 2 kb | consensus4 | 411 | 0.612 (0.420 to 0.748) | 0.559 | 0.781 | 0.590 | 0.513 | 1.03 | 0.967 |
| A GM12878 2 kb | Hi-cGAN (submitted) | 411 | 0.238 (0.049 to 0.400) | 0.208 | 0.380 | 0.321 | 0.138 | 0.18 | 0.775 |
| A GM12878 2 kb | best other map (h1) | 411 | 0.516 (0.311 to 0.635) | 0.457 | 0.722 | 0.541 | 0.442 | 1.22 | 0.957 |
| A HFF 2 kb | Akita | 413 | 0.626 (0.404 to 0.742) | 0.568 | 0.771 | 0.538 | 0.321 | 0.39 | 0.911 |
| A HFF 2 kb | all12 2 kb | 413 | 0.657 (0.495 to 0.749) | 0.664 | 0.856 | 0.672 | 0.347 | 0.56 | 0.944 |
| A HFF 2 kb | best other map (imr90) | 413 | 0.625 (0.471 to 0.728) | 0.576 | 0.864 | 0.575 | 0.320 | 0.53 | 0.934 |
| B GM12878 10 kb chr3/11/17 | all12honest | 42 | 0.856 (0.720 to 0.926) | 0.787 | 0.801 | 0.498 | 0.331 | 0.38 | 0.912 |
| B GM12878 10 kb chr3/11/17 | all12refres | 42 | 0.948 (0.910 to 0.962) | 0.912 | 0.926 | 0.575 | 0.612 | 0.78 | 0.961 |
| B GM12878 10 kb chr3/11/17 | consensus of other cells | 42 | 0.876 (0.806 to 0.913) | 0.793 | 0.851 | 0.595 | 0.636 | 0.98 | 0.959 |
| B GM12878 10 kb chr3/11/17 | Epiphany | 42 | 0.648 (0.523 to 0.808) | 0.552 | 0.442 | 0.328 | 0.058 | 0.22 | 0.723 |
| B GM12878 10 kb chr3/11/17 | Hi-cGAN (submitted) | 42 | 0.831 (0.667 to 0.892) | 0.666 | 0.551 | 0.344 | 0.207 | 0.51 | 0.793 |
| B GM12878 10 kb chr3/11/17 | best other map (hct116) | 42 | 0.829 (0.737 to 0.887) | 0.716 | 0.762 | 0.595 | 0.570 | 0.69 | 0.947 |
| B GM12878 10 kb even | all12honest | 212 | 0.843 (0.739 to 0.911) | 0.777 | 0.794 | 0.455 | 0.336 | 0.42 | 0.938 |
| B GM12878 10 kb even | all12refres | 212 | 0.944 (0.896 to 0.964) | 0.902 | 0.911 | 0.486 | 0.551 | 0.78 | 0.979 |
| B GM12878 10 kb even | consensus of other cells | 212 | 0.856 (0.732 to 0.914) | 0.789 | 0.810 | 0.551 | 0.541 | 0.86 | 0.972 |
| B GM12878 10 kb even | Hi-cGAN (submitted) | 212 | 0.507 (0.328 to 0.646) | 0.380 | 0.203 | 0.181 | 0.048 | 0.18 | 0.692 |
| B GM12878 10 kb even | best other map (a549) | 212 | 0.801 (0.621 to 0.878) | 0.727 | 0.732 | 0.516 | 0.424 | 0.89 | 0.955 |
| C GM12878 25 kb even | all12honest | 218 | 0.852 (0.739 to 0.921) | 0.798 | 0.759 | 0.553 | 0.271 | 0.41 | 0.841 |
| C GM12878 25 kb even | all12refres | 218 | 0.945 (0.897 to 0.965) | 0.910 | 0.907 | 0.660 | 0.437 | 0.71 | 0.898 |
| C GM12878 25 kb even | consensus of other cells | 218 | 0.880 (0.756 to 0.932) | 0.824 | 0.854 | 0.710 | 0.485 | 0.72 | 0.898 |
| C GM12878 25 kb even | Hi-cGAN (submitted) | 218 | 0.486 (0.191 to 0.653) | 0.315 | 0.184 | 0.238 | 0.097 | 0.18 | 0.644 |
| C GM12878 25 kb even | HiC-Reg | 218 | 0.943 (0.908 to 0.961) | 0.910 | 0.805 | 0.667 | 0.326 | 0.31 | 0.861 |
| C GM12878 25 kb even | best other map (k562) | 218 | 0.838 (0.663 to 0.909) | 0.759 | 0.815 | 0.661 | 0.394 | 0.70 | 0.887 |
| D IMR-90 10 kb chr2/3 | all12honest | 45 | 0.868 (0.790 to 0.918) | 0.817 | 0.831 | 0.537 | 0.314 | 0.38 | 0.922 |
| D IMR-90 10 kb chr2/3 | all12refres | 45 | 0.951 (0.927 to 0.967) | 0.908 | 0.948 | 0.570 | 0.524 | 0.76 | 0.969 |
| D IMR-90 10 kb chr2/3 | consensus of other cells | 45 | 0.916 (0.872 to 0.925) | 0.850 | 0.902 | 0.686 | 0.562 | 0.86 | 0.981 |
| D IMR-90 10 kb chr2/3 | C.Origami | 45 | 0.908 (0.874 to 0.938) | 0.842 | 0.846 | 0.574 | 0.229 | 0.62 | 0.928 |
| D IMR-90 10 kb chr2/3 | Hi-cGAN (submitted), GM12878 model | 45 | 0.472 (0.339 to 0.640) | 0.405 | 0.283 | 0.223 | 0.067 | 0.12 | 0.684 |
| D IMR-90 10 kb chr2/3 | Hi-cGAN (submitted), IMR-90 model | 45 | 0.719 (0.440 to 0.834) | 0.581 | 0.405 | 0.281 | 0.133 | 0.32 | 0.686 |
| D IMR-90 10 kb chr2/3 | best other map (gm23248) | 45 | 0.919 (0.889 to 0.950) | 0.861 | 0.895 | 0.686 | 0.495 | 0.78 | 0.972 |
| E K562 10 kb | all12honest | 212 | 0.782 (0.581 to 0.870) | 0.671 | 0.712 | 0.445 | 0.263 | 0.32 | 0.898 |
| E K562 10 kb | all12refres | 212 | 0.883 (0.800 to 0.922) | 0.812 | 0.860 | 0.462 | 0.510 | 0.75 | 0.960 |
| E K562 10 kb | consensus of other cells | 212 | 0.851 (0.734 to 0.907) | 0.787 | 0.860 | 0.572 | 0.539 | 0.85 | 0.964 |
| E K562 10 kb | Hi-cGAN (submitted), GM12878 model | 212 | 0.304 (0.061 to 0.529) | 0.166 | 0.084 | 0.165 | 0.026 | 0.15 | 0.612 |
| E K562 10 kb | best other map (hct116) | 212 | 0.821 (0.673 to 0.882) | 0.717 | 0.693 | 0.501 | 0.477 | 0.61 | 0.946 |
| E K562 25 kb | all12honest | 218 | 0.803 (0.609 to 0.904) | 0.720 | 0.720 | 0.521 | 0.283 | 0.27 | 0.795 |
| E K562 25 kb | all12refres | 218 | 0.906 (0.846 to 0.941) | 0.850 | 0.854 | 0.629 | 0.409 | 0.70 | 0.869 |
| E K562 25 kb | consensus of other cells | 218 | 0.889 (0.774 to 0.934) | 0.834 | 0.874 | 0.750 | 0.474 | 0.86 | 0.891 |
| E K562 25 kb | Hi-cGAN (submitted), GM12878 model | 218 | 0.456 (0.167 to 0.651) | 0.274 | 0.108 | 0.233 | 0.089 | 0.40 | 0.656 |
| E K562 25 kb | best other map (hct116) | 218 | 0.859 (0.726 to 0.928) | 0.773 | 0.779 | 0.677 | 0.415 | 0.72 | 0.884 |
| E HMEC 10 kb | all12honest | 212 | 0.720 (0.568 to 0.823) | 0.684 | 0.687 | 0.393 | 0.224 | 0.42 | 0.891 |
| E HMEC 10 kb | all12refres | 212 | 0.904 (0.868 to 0.939) | 0.854 | 0.853 | 0.497 | 0.516 | 0.92 | 0.972 |
| E HMEC 10 kb | consensus of other cells | 212 | 0.908 (0.847 to 0.939) | 0.857 | 0.901 | 0.594 | 0.562 | 0.98 | 0.973 |
| E HMEC 10 kb | Hi-cGAN (submitted), GM12878 model | 212 | 0.372 (0.134 to 0.558) | 0.232 | 0.177 | 0.198 | 0.023 | 0.16 | 0.640 |
| E HMEC 10 kb | best other map (nhek) | 212 | 0.870 (0.824 to 0.913) | 0.779 | 0.821 | 0.569 | 0.266 | 1.02 | 0.963 |
| E HMEC 25 kb | all12honest | 218 | 0.759 (0.579 to 0.871) | 0.702 | 0.687 | 0.466 | 0.241 | 0.64 | 0.815 |
| E HMEC 25 kb | all12refres | 218 | 0.916 (0.864 to 0.953) | 0.873 | 0.867 | 0.637 | 0.397 | 0.95 | 0.893 |
| E HMEC 25 kb | consensus of other cells | 218 | 0.933 (0.877 to 0.957) | 0.902 | 0.922 | 0.802 | 0.465 | 1.00 | 0.904 |
| E HMEC 25 kb | Hi-cGAN (submitted), GM12878 model | 218 | 0.397 (0.103 to 0.606) | 0.301 | 0.164 | 0.252 | 0.085 | 0.33 | 0.645 |
| E HMEC 25 kb | best other map (nhek) | 218 | 0.928 (0.875 to 0.954) | 0.866 | 0.872 | 0.744 | 0.411 | 1.22 | 0.905 |
| E HUVEC 10 kb | all12honest | 212 | 0.812 (0.691 to 0.875) | 0.720 | 0.666 | 0.430 | 0.258 | 0.32 | 0.928 |
| E HUVEC 10 kb | all12refres | 212 | 0.879 (0.792 to 0.930) | 0.796 | 0.762 | 0.465 | 0.471 | 0.63 | 0.965 |
| E HUVEC 10 kb | consensus of other cells | 212 | 0.877 (0.792 to 0.930) | 0.802 | 0.838 | 0.564 | 0.519 | 0.68 | 0.962 |
| E HUVEC 10 kb | Hi-cGAN (submitted), GM12878 model | 212 | 0.392 (0.189 to 0.571) | 0.279 | 0.250 | 0.204 | 0.036 | 0.12 | 0.645 |
| E HUVEC 10 kb | best other map (imr90) | 212 | 0.833 (0.732 to 0.906) | 0.777 | 0.839 | 0.510 | 0.344 | 0.64 | 0.920 |
| E HUVEC 25 kb | all12honest | 218 | 0.821 (0.674 to 0.890) | 0.743 | 0.684 | 0.512 | 0.232 | 0.35 | 0.815 |
| E HUVEC 25 kb | all12refres | 218 | 0.892 (0.802 to 0.943) | 0.831 | 0.802 | 0.563 | 0.325 | 0.69 | 0.855 |
| E HUVEC 25 kb | consensus of other cells | 218 | 0.912 (0.823 to 0.950) | 0.852 | 0.872 | 0.725 | 0.384 | 0.77 | 0.871 |
| E HUVEC 25 kb | Hi-cGAN (submitted), GM12878 model | 218 | 0.481 (0.200 to 0.669) | 0.382 | 0.279 | 0.293 | 0.082 | 0.23 | 0.685 |
| E HUVEC 25 kb | best other map (imr90) | 218 | 0.876 (0.781 to 0.936) | 0.825 | 0.872 | 0.669 | 0.332 | 0.77 | 0.836 |

In Akita's own space (block A, 411 GM12878 regions) the honest akita_native
model exceeds Akita by a median 0.040 Pearson per region (66% of the
regions, Wilcoxon p = 2e-17) and 0.061 SCC, and the reference residual model
by 0.135 (97% of the regions); Hi-cGAN (submitted) reaches 0.238. On HFF,
Akita's second head, the 12-cell 2 kb model scores 0.657 against 0.626
(56% of the regions, p = 3e-4) and 0.664 against 0.568 SCC; the HFF block
converts our proper O/E prediction into Akita's grid only, so SCC, which
ignores per-diagonal offsets between the two O/E definitions, is the fairer
measure there.

At 10 kb on Epiphany's held-out chromosomes (42 regions) Epiphany scores
0.648 and 0.552 SCC; all12honest exceeds it by 0.181 per region (88%,
p = 2e-6) and all12refres by 0.258 (all regions). Hi-cGAN (submitted) scores
0.831 on these chromosomes, which are its training chromosomes, and 0.507 on
its held-out even chromosomes (212 regions), where all12honest reaches 0.843
and all12refres 0.944. At 25 kb (218 regions on even chromosomes) HiC-Reg
scores 0.943 and 0.910 SCC, level with all12refres (0.945 and 0.910; median
difference 0.002, 53% of the regions, p = 0.80) and above all12honest
(0.852); HiC-Reg's per-distance Pearson is 0.26 at 50 kb against 0.82 for
all12refres and approaches it only beyond 500 kb, its APA ratio is 0.31
against 0.71 and its loop recall 0.33 against 0.44, and its training split
may include these bin pairs. On C.Origami's held-out chromosomes (45 IMR-90
regions) C.Origami scores 0.908 and 0.842 SCC, all12refres 0.951 and 0.908
(91% of the regions, p = 6e-10) and all12honest 0.868 (below C.Origami in
78% of the regions); the best single other map (GM23248) scores 0.919.

On cells never seen in training (block E) the GM12878 Hi-cGAN model applied
to K562, HMEC and HUVEC inputs scores 0.30 to 0.48 and all12honest 0.72 to
0.82. all12refres scores 0.88 to 0.92 and exceeds the consensus of the 12
training cells for K562 (by 0.025 at 10 kb and 0.017 at 25 kb, 78% and 68%
of the regions), equals it for HUVEC at 10 kb and HMEC at 10 kb, and stays
below it for HMEC and HUVEC at 25 kb (0.916 against 0.933, 0.892 against
0.912). The consensus of the other cells' measured maps reaches a higher
boundary recall than every model in blocks B to E (consensus 0.55 to 0.80,
all12refres 0.46 to 0.66, HiC-Reg 0.67); the measured target's own
insulation minima recover 0.63 to 0.88 of the hicFindTADs boundaries. Loop
recall of all12refres (0.33 to 0.61) lies within 0.07 of the consensus in
blocks B to E (above it only for GM12878 on the even chromosomes, 0.551
against 0.541), and at 2 kb akita_native_refres recovers 0.574 of the loops
against 0.513 for consensus4 and 0.314 for Akita.

Tables `benchmark/benchmark_summary.csv` (every method, every block: median,
mean, quartiles, SD and n of Pearson, SCC, Spearman and insulation Pearson;
boundary and loop recall; APA; AUROC), `benchmark_paired.csv` (paired
Wilcoxon and win rates of each of our models against every other method),
`<block>_per_region.csv.gz`, `<block>_per_distance.csv`,
`scc_akita_regions.csv`; figures `runs/benchmark/figures/<block>_per_distance.png`
and `<block>_regions.png`.

## 19. AML cell lines never seen in training (FINAL)

`scripts/aml_fetch.py` (ChIP inputs), `scripts/aml_eval.py` (stages targets,
perturb_inputs, predict, score, perturb, figures), tests
`tests/test_aml_eval.py`. Data under `/mnt/processed/wolffjoa/hic-multiome/aml/`
(`STATUS.md` there). Tables `aml/summary.csv` (every method, median Pearson
and median SCC per target and resolution), `grid.csv`, `per_distance.csv`,
`replicates.csv`, `replicates_per_region.csv`, `perturb_summary.csv`,
`targets/*.json` (contacts, ICE masked fraction, agreement with the GEO
weights); figures `per_distance.png`, `regions_<target>_<k>kb.png`,
`perturb_<k>kb_<model>_<input>.png`.

Measured maps. MOLM-13: Arima in situ Hi-C GSE285123 / GSM9769670 (hg38
mcool, 434 M contacts, levels from 5 kb) as the target; MboI in situ Hi-C of
Xu et al. 2022, GSE147123 / GSM4417606 (hg38 .hic through the C++ hic2cool,
124 M contacts) as the second, independent map. Neither holds a 2 kb level,
so MOLM-13 is scored at 5, 10 and 25 kb; the 2 x 474 M read pairs of
GSE224931 exist only as raw reads and were not mapped. U-937: Micro-C
GSE341752, wild-type clones 1 to 3 (GSM9916468-70; 3.26, 1.72 and 2.18 B
contacts) and STAG2 knockout clones 1 and 2 (GSM9916471-72; 1.22 and 0.98 B).
Targets are wild-type clone 1 and knockout clone 1; the other clones are the
replicates. Every map was balanced with cooler ICE (`--ignore-diags 2
--max-iters 500`) and converted with `scripts/oe_proper.py`. For the U-937
clones the GEO weights were used; at 25 kb they equal a fresh ICE (Pearson
1.000), as do the MOLM-13 Arima weights at 5 and 10 kb.

ChIP inputs. One ChIP-Atlas hg38 experiment per mark (untreated or control
arm, most reads and peaks; picks in `aml/picks_<cell>.tsv`). The ChIP-Atlas
bigWigs are RPM coverage that omits uncovered runs; they are written on a
200 bp grid with uncovered bases as 0 and scaled so the genome-wide mean
equals the mean of the training cells' ENCODE fold-change bigWigs of the same
mark. MOLM-13: CTCF SRX3927299, RAD21 SRX9891934, H3K27ac SRX1735804,
H3K27me3 SRX7480103, H3K36me3 SRX13440075, H3K4me1 SRX1735805, H3K4me3
SRX1735806, H3K79me2 SRX5444521, H3K9ac SRX1735807, H3K9me3 SRX9522726;
absent SMC3 (also absent in 8 of the 12 training cells; STAG1/STAG2 ChIP
exists but no substitution was used, since section 3 tested only RAD21 with
SMC3), H3K4me2 and H4K20me1. U-937: CTCF ERX704042, RAD21 ERX626810, SMC3
ERX626812, H3K27ac SRX11712771, H3K4me3 SRX685532, H3K9ac SRX18235716,
H3K9me3 SRX9407972; absent H3K27me3 (GSE285122 deposits peaks only),
H3K36me3, H3K4me1, H3K4me2, H3K79me2 and H4K20me1. The U-937 CTCF, RAD21 and
SMC3 experiments carry 10443, 1044 and 780 peaks. No ChIP of the U-937 STAG2
knockout is public (ChIP-Atlas; GEO GSE165038, GSE198518, GSE341414).

Input-processing control. K562 (held out from every model) predicted from
ChIP-Atlas processing of the same ENCODE experiments as `hg38_k562` (11 of
13 marks; RAD21 and H3K4me3 from other ENCODE K562 experiments) against the
ENCODE fold-change inputs on the same pixels: all12honest 0.808 / 0.775 (25
kb) and 0.770 / 0.749 (10 kb), all12refres 0.932 / 0.906 and 0.905 / 0.885
(ChIP-Atlas / ENCODE, median Pearson). all12 at 5 and 2 kb from ChIP-Atlas
inputs scores 0.693 and 0.579 on K562.

Models: all12 (section 13, 2 to 25 kb), all12honest and all12refres
(section 15, 10 and 25 kb; reference consensus12, which contains no AML map;
the reference guard passed), GM12878-only models (section 6 model at 2 kb,
section 14 "all" seed 0 at 5 to 25 kb, section 15 A2 residual seed 0 with
consensus11 at 10 and 25 kb). `scripts/crosscell_predict.py --tta` on the
CPU, U-937 with `--assay microc`. Scoring: the 413 Akita test regions, one
pixel mask per target and resolution shared by every method (upper triangle,
offset >= 2 bins, measured in the target and defined in every map and
prediction), median over regions of Pearson and of SCC
(`scripts/benchmark_metrics.py`).

Median Pearson / median SCC, 413 regions; best map = the best of the 12
training cells' maps:

| target | res | all12refres | GM12878 refres | all12honest | all12 | GM12878 only | consensus12 | best map | K562 map | replicate |
|---|---|---|---|---|---|---|---|---|---|---|
| MOLM-13 | 25 kb | 0.904 / 0.829 | 0.906 / 0.828 | 0.782 / 0.645 | 0.782 / 0.659 | 0.677 / 0.567 | 0.909 / 0.844 | 0.878 / 0.811 (GM12878) | 0.866 / 0.783 | 0.913 / 0.859 |
| MOLM-13 | 10 kb | 0.865 / 0.769 | 0.872 / 0.798 | 0.725 / 0.595 | 0.730 / 0.600 | 0.693 / 0.586 | 0.864 / 0.780 | 0.826 / 0.744 (GM12878) | 0.792 / 0.704 | 0.802 / 0.717 |
| MOLM-13 | 5 kb | | | | 0.621 / 0.539 | 0.542 / 0.477 | 0.790 / 0.686 | 0.727 / 0.642 (GM12878) | 0.658 / 0.578 | 0.629 / 0.540 |
| MOLM-13 (Xu map) | 25 kb | 0.890 / 0.843 | 0.891 / 0.831 | 0.751 / 0.657 | 0.752 / 0.648 | 0.622 / 0.535 | 0.908 / 0.859 | 0.866 / 0.802 (A549) | 0.846 / 0.794 | 0.913 / 0.859 |
| MOLM-13 (Xu map) | 10 kb | 0.809 / 0.735 | 0.803 / 0.725 | 0.600 / 0.550 | 0.609 / 0.540 | 0.594 / 0.526 | 0.822 / 0.749 | 0.759 / 0.683 (A549) | 0.741 / 0.669 | 0.802 / 0.717 |
| MOLM-13 (Xu map) | 5 kb | | | | 0.464 / 0.416 | 0.352 / 0.333 | 0.676 / 0.586 | 0.605 / 0.518 (HFF) | 0.561 / 0.502 | 0.629 / 0.540 |
| U-937 WT | 25 kb | 0.837 / 0.759 | 0.851 / 0.768 | 0.652 / 0.535 | 0.530 / 0.490 | 0.234 / 0.304 | 0.866 / 0.787 | 0.825 / 0.755 (GM12878) | 0.815 / 0.726 | 0.959, 0.982 / 0.928, 0.960 |
| U-937 WT | 10 kb | 0.832 / 0.735 | 0.805 / 0.703 | 0.328 / 0.360 | 0.483 / 0.443 | 0.128 / 0.165 | 0.818 / 0.722 | 0.783 / 0.696 (GM12878) | 0.743 / 0.644 | 0.896, 0.937 / 0.834, 0.883 |
| U-937 WT | 5 kb | | | | 0.389 / 0.347 | 0.037 / 0.053 | 0.749 / 0.648 | 0.698 / 0.610 (GM12878) | 0.624 / 0.542 | 0.765, 0.832 / 0.699, 0.753 |
| U-937 WT | 2 kb | | | | 0.335 / 0.284 | 0.423 / 0.390 | 0.587 / 0.494 | 0.507 / 0.423 (HFF) | 0.420 / 0.329 | 0.523, 0.587 / 0.437, 0.486 |
| U-937 STAG2 KO | 25 kb | 0.795 / 0.736 | 0.796 / 0.718 | 0.583 / 0.499 | 0.509 / 0.489 | 0.189 / 0.276 | 0.801 / 0.745 | 0.774 / 0.707 (GM12878) | 0.776 / 0.711 | 0.948 / 0.906 |
| U-937 STAG2 KO | 10 kb | 0.752 / 0.677 | 0.722 / 0.650 | 0.281 / 0.357 | 0.440 / 0.434 | 0.077 / 0.148 | 0.727 / 0.672 | 0.696 / 0.628 (GM12878) | 0.684 / 0.613 | 0.843 / 0.768 |
| U-937 STAG2 KO | 5 kb | | | | 0.291 / 0.336 | -0.028 / 0.049 | 0.617 / 0.558 | 0.586 / 0.525 (GM12878) | 0.545 / 0.479 | 0.649 / 0.559 |
| U-937 STAG2 KO | 2 kb | | | | 0.306 / 0.234 | 0.301 / 0.298 | 0.416 / 0.373 | 0.350 / 0.312 (HFF) | 0.303 / 0.248 | 0.396 / 0.306 |

Replicate column (on the shared mask of the row): MOLM-13, the Arima map
against the Xu map; U-937 WT, clone 1 against clone 2 and against clone 3;
KO, clone 1 against clone 2.
The U-937 KO rows score the wild-type ChIP input; the measured wild-type
clone 1 map scores 0.905 / 0.885, 0.838 / 0.788, 0.705 / 0.636 and 0.441 /
0.378 against KO clone 1 at 25, 10, 5 and 2 kb. Wild-type clone 2 against
clone 3 on their own pixels (`replicates.csv`): 0.952, 0.885, 0.749 and
0.504. The 5 and 2 kb honest and
reference 12-cell runs do not exist yet.

On MOLM-13 the residual reference models (12-cell and GM12878) score within
0.01 of consensus12 at 10 and 25 kb against the Arima map (0.904 and 0.906
against 0.909 at 25 kb; 0.865 and 0.872 against 0.864 at 10 kb) and above
the best single training map by 0.03 to 0.05; against the Xu map they score
0.013 to 0.019 below consensus12. The two independent MOLM-13 maps agree at
0.913, 0.802 and 0.629 (25, 10, 5 kb); at 10 kb consensus12 (0.864) and the
reference models exceed this replicate agreement, since the Xu map holds 124
M contacts. Without the reference the 12-cell models score 0.72 to 0.78 on
MOLM-13 at 10 and 25 kb, below 11 of the 12 training maps at 25 kb (0.782
against 0.817 to 0.878; CD14+ monocytes 0.722) and below the median training
map at 10 kb (0.725 and 0.730 against 0.783). On U-937, with 7 of 13 marks
and cohesin ChIP of about 1000 peaks, the models without reference score 0.04
to 0.65 at 5 to 25 kb;
all12refres scores 0.832 (WT, 10 kb), 0.014 above consensus12, and 0.837 at
25 kb, 0.029 below it. At 2 kb the GM12878-only model with track dropout
(0.423) exceeds all12 (0.335); both stay below consensus12 (0.587) and the
best single training map (0.507, HFF).

STAG2 knockout. The knockout was predicted from the wild-type ChIP with the
cohesin tracks changed (`aml_eval.py`, block STAG2-KO inputs): ko_absent
(RAD21 and SMC3 declared absent), ko_ctcfonly (cohesin signal kept within 1
kb of a U-937 CTCF peak and set to the track mean elsewhere, the retention of
STAG1-cohesin at CTCF sites after STAG2 loss) and ko_half (enrichment above
the mean halved). Change = KO map minus WT map (log O/E), predicted change =
prediction from the KO input minus prediction from the WT input, Pearson per
region on the shared mask (median over 413 regions). Loops: donut calls on
the measured WT and KO maps (union, `hicmultiome.bioscore.call_loops` on a 10
kb grid, 60 to 600 kb), strength = log2 donut enrichment; boundaries: minima
of the WT insulation profile (diamond mean of log O/E, 100 kb). Ceiling: the
change between independent clone pairs, (KO c1 - WT c1) against (KO c2 - WT
c2), (KO c2 - WT c3), and (KO c1 - WT c2) against (KO c2 - WT c3).

| res | model:input | change map Pearson / SCC | loop change Pearson (Spearman) | loop change sign agreement | predicted mean loop change (short / long) | boundary change Pearson |
|---|---|---|---|---|---|---|
| 10 kb | all12refres ko_absent | 0.183 / 0.118 | 0.351 (0.281) | 0.565 | -0.144 / -0.109 | 0.082 |
| 10 kb | all12refres ko_ctcfonly | 0.041 / 0.082 | 0.354 (0.405) | 0.586 | -0.056 / -0.037 | 0.121 |
| 10 kb | all12refres ko_half | 0.071 / 0.104 | 0.369 (0.427) | 0.588 | -0.035 / -0.022 | 0.111 |
| 10 kb | all12 ko_absent | 0.070 / 0.038 | 0.156 (0.156) | 0.532 | -0.044 / -0.034 | -0.169 |
| 10 kb | all12 ko_ctcfonly | -0.019 / 0.046 | 0.175 (0.254) | 0.573 | -0.038 / -0.020 | -0.071 |
| 10 kb | all12honest ko_ctcfonly | -0.114 / 0.001 | 0.066 (0.070) | 0.514 | -0.009 / -0.006 | -0.121 |
| 10 kb | GM12878 refres ko_absent | -0.145 / -0.102 | -0.372 (-0.342) | 0.415 | 0.064 / 0.043 | -0.109 |
| 25 kb | all12honest ko_ctcfonly | 0.164 / 0.073 | 0.103 (0.180) | 0.649 | -0.044 / -0.040 | -0.028 |
| 25 kb | all12refres ko_absent | 0.115 / 0.102 | 0.183 (0.229) | 0.588 | 0.000 / -0.041 | -0.158 |
| 2 kb | all12 ko_absent | 0.108 / 0.086 | -0.003 (-0.016) | 0.550 | -0.018 / -0.061 | -0.098 |

Measured at 10 kb (626 loops, 221 shorter than 200 kb, 832 boundaries):
loop strength changes by -0.105 (short) and +0.181 (long) log2 units, WT
boundaries weaken (insulation +0.087). Clone-pair ceiling of the change map:
0.254, 0.339 and 0.122 at 10 kb; 0.547, 0.615 and 0.293 at 25 kb; 0.118,
0.183 and 0.042 at 2 kb. The 12-cell models predict weaker short and long
loops alike (all12refres at 10 kb: -0.022 to -0.144); the measured gain of
long loops (+0.181) appears in no variant, and the measured weakening of WT
boundaries (+0.087) appears in all12refres at 0.008 to 0.018. The best change map
(all12refres, ko_absent, 10 kb, 0.183) lies within the clone-pair range at
that resolution; loop-level changes correlate at 0.35 to 0.37 for
all12refres, where the sign agreement stays at 0.57 to 0.59. Full grid in
`perturb_summary.csv` (every model, input and resolution, loop and boundary
counts).

## 20. Minimal input sets (absent-at-inference FINAL, dedicated trainings PENDING)

Question: how well do practical small input sets predict, and can the
existing models be run with the other tracks absent? Sets (`scripts/minimal_inputs.py`;
every set keeps all motif channels of its model and GC/CpG): (a) CTCF and
H3K27ac, the C.Origami-like analogue (accessibility is not a model track);
(b) CTCF and RAD21; (c) CTCF, RAD21, H3K27ac, H3K4me3 and H3K27me3, one mark
per group of section 14; (d) motifs only.

Absent at inference. The existing models predict with every track outside
the set filled with its absent value (`crosscell_predict.py --keep-sets`,
one pass per model and cell, test-time reverse averaging, CPU): the 12-cell
models at 25 and 10 kb for GM12878, IMR-90 (training cells) and K562 (held
out) on the 413 Akita test regions against the proper O/E targets, and the
GM12878 2048 bp models against Akita's target (section 1 metric). Only
akita_native_trackdrop (section 4) was trained with track dropout; the
12-cell models and akita_native_honest/refres were trained with dropout 0,
and in the 12-cell training an absent track occurs only where a cell lacks
the ChIP (cohesin for HFF, SMC3 for HCT116, several marks for HFF, HMEC and
HUVEC). Median per-region Pearson (full inputs first):

| model | cell | full | (a) CTCF + H3K27ac | (b) CTCF + RAD21 | (c) one per group | (d) motifs only |
|---|---|---|---|---|---|---|
| akita_native_trackdrop | gm12878 | 0.642 | 0.564 | 0.488 | 0.609 | 0.301 |
| akita_native_honest | gm12878 | 0.636 | 0.002 | -0.001 | 0.028 | -0.005 |
| akita_native_refres | gm12878 | 0.751 | 0.656 | 0.630 | 0.685 | 0.636 |
| all12honest_25kb | gm12878 | 0.822 | 0.574 | 0.508 | 0.655 | 0.269 |
| all12honest_25kb | imr90 | 0.881 | 0.597 | 0.490 | 0.712 | 0.271 |
| all12honest_25kb | k562 | 0.771 | 0.495 | 0.416 | 0.601 | 0.252 |
| all12refres_25kb | gm12878 | 0.938 | 0.872 | 0.841 | 0.854 | 0.854 |
| all12refres_25kb | imr90 | 0.955 | 0.880 | 0.888 | 0.909 | 0.896 |
| all12refres_25kb | k562 | 0.906 | 0.853 | 0.857 | 0.846 | 0.864 |
| all12honest_10kb | gm12878 | 0.835 | 0.571 | 0.346 | 0.584 | 0.081 |
| all12honest_10kb | imr90 | 0.881 | 0.560 | 0.302 | 0.597 | 0.044 |
| all12honest_10kb | k562 | 0.750 | 0.461 | 0.256 | 0.483 | 0.053 |
| all12refres_10kb | gm12878 | 0.938 | 0.879 | 0.792 | 0.877 | 0.826 |
| all12refres_10kb | imr90 | 0.948 | 0.889 | 0.826 | 0.880 | 0.862 |
| all12refres_10kb | k562 | 0.885 | 0.846 | 0.772 | 0.855 | 0.813 |

The track-dropout model degrades gradually: 0.642 with all 13 tracks, 0.609
with one mark per group, 0.564 with CTCF and H3K27ac (Akita scores 0.548 on
the same 413 regions, section 1), 0.488 with CTCF and RAD21 and 0.301 with
the CTCF motif alone. The same architecture trained without dropout
(akita_native_honest) collapses to 0.00 to 0.03 with any track absent, so
absent-at-inference results need a dropout-trained model. The 12-cell
no-reference models keep 0.42 to 0.71 with sets (a) to (c) at 25 kb and 0.26
to 0.60 at 10 kb and fall to 0.04 to 0.27 with motifs only. The reference
residual models keep 0.77 to 0.91 with every set, but with the tracks absent
they score below the leave-one-out consensus they read in 22 of the 24
set, cell and resolution combinations (consensus alone 0.882, 0.928, 0.897
for GM12878, IMR-90, K562 at 25 kb; 0.858, 0.903, 0.859 at 10 kb); only (a)
and (c) at 10 kb for GM12878 exceed it (0.879 and 0.877). At 2 kb the residual model
with motifs only scores 0.636, level with consensus4 alone (0.618 on the 413
regions, section 1), and 0.656 to 0.685 with sets (a) and (c).

Dedicated trainings (PENDING, GPU). Configs `configs/minimal/min_<set>_<setup>.yaml`
for the 12-cell residual setup at 25 and 10 kb (base all12refres, existing
caches, `train.select_tracks` / `select_motifs`) and for GM12878 at 2 kb
(base akita_native_honest), honest validation, track dropout 0. The running
scheduler (`scripts/gpu_scheduler.py`) holds its job list in memory, so the
request is in `runs/benchmark/QUEUE_REQUEST.md` with a driver
`runs/benchmark/minimal_queue.sh` and a scheduler snippet; evaluation by
`python scripts/minimal_inputs.py --stage eval --run <run>` and the table by
`--stage summary`. Table `minimal_inputs/minimal_inputs_summary.csv`
(median, mean, quartiles and n per model, set and cell).

## 21. In-silico perturbation against degron experiments (FINAL)

Question: does removing a protein from the input reproduce the measured
change of the contact map when that protein is degraded?

Experiments (hg38, `runs/perturb_validate`, data provenance in
`/mnt/processed/wolffjoa/hic-multiome/perturb/README.md`): HCT116 RAD21-mAC,
untreated against 6 h auxin (Rao et al. 2017, 4DN 4DNFIFLDVASC and
4DNFILP99QJS, 2.69 and 2.55 billion contacts); HCT116 CTCF-mAID2, untreated
against 6 h 5-Ph-IAA (ENCODE intact Hi-C ENCSR086RCC and ENCSR669VZH, .hic
converted with the C++ hic2cool, cooler ICE). Every map goes through
`oe_proper.py` at 10 and 25 kb; the measured change is treated minus
untreated log O/E. The untreated RAD21-mAC map is the HCT116 training target
of the 12-cell models; the 413 Akita test regions are held out from their
training.

Predictions (`scripts/perturb_validate.py`, CPU): all12honest and
all12refres at 10 and 25 kb predict HCT116 from its ENCODE ChIP (no SMC3
track exists for HCT116) with all tracks and with one track set to its
absent value (`crosscell_predict.py --keep-sets`): RAD21, CTCF, and H3K36me3
as a control; motif channels unchanged. A second RAD21 prediction reads
measured ChIP of the degron cells (4DN GRCh38 fold change; RAD21 and CTCF of
Kim et al. 2024, eight histone marks of Rao et al. 2017, untreated and
6 h auxin; H3K4me2, H3K9ac and H4K20me1 from ENCODE in both), predicted change
= auxin-ChIP prediction minus untreated-ChIP prediction. The residual
models read the leave-one-out consensus in both conditions. Metrics per
region (upper triangle, at least 2 bins off the diagonal): Pearson and SCC
between predicted and measured change maps; at the 536 (10 kb) and 549
(25 kb) loops that hicDetectLoops calls in the untreated RAD21-mAC map
inside the regions, the change of the log O/E maximum within 1 bin, the share
of loops that weaken and APA (centre over lower-left 3 x 3 corner, 7 x 7
window); at the 10 kb hicFindTADs boundaries the change of the insulation
score.

Measured effect. After RAD21 degradation 99.4% of the loops weaken (median
log O/E change -0.75 at 10 kb), APA falls from 1.45 to 1.05 and boundary
insulation rises by 0.063 (weaker boundaries). After CTCF degradation 99.1%
weaken (median -0.49), APA falls from 1.70 to 1.35.

| experiment | model | input change | change Pearson median (IQR) | change SCC | loops weaker, predicted (measured) | loop change Spearman | APA full / perturbed (measured untreated / treated) |
|---|---|---|---|---|---|---|---|
| RAD21 degron | all12honest_10kb | RAD21 absent | 0.041 (-0.184 to 0.302) | -0.031 | 0.24 (0.99) | -0.06 | 1.19 / 1.22 (1.45 / 1.05) |
| RAD21 degron | all12honest_10kb | measured ChIP, auxin vs untreated | 0.032 (-0.226 to 0.206) | 0.110 | 0.62 (0.99) | -0.07 | 1.29 / 1.26 (1.45 / 1.05) |
| RAD21 degron | all12honest_10kb | CTCF absent | 0.151 (-0.102 to 0.403) | 0.195 | 0.78 (0.99) | 0.33 | 1.19 / 1.09 (1.45 / 1.05) |
| RAD21 degron | all12honest_10kb | H3K36me3 absent (control) | 0.116 (-0.137 to 0.357) | 0.033 | 0.36 (0.99) | -0.12 | 1.19 / 1.19 (1.45 / 1.05) |
| RAD21 degron | all12honest_25kb | RAD21 absent | -0.067 (-0.318 to 0.249) | -0.133 | 0.13 (0.99) | -0.15 | 1.14 / 1.18 (1.33 / 1.02) |
| RAD21 degron | all12honest_25kb | measured ChIP, auxin vs untreated | -0.098 (-0.374 to 0.184) | 0.000 | 0.56 (0.99) | 0.03 | 1.18 / 1.17 (1.33 / 1.02) |
| RAD21 degron | all12honest_25kb | CTCF absent | 0.207 (-0.104 to 0.523) | 0.317 | 0.75 (0.99) | 0.38 | 1.14 / 1.07 (1.33 / 1.02) |
| RAD21 degron | all12honest_25kb | H3K36me3 absent (control) | 0.076 (-0.265 to 0.427) | -0.038 | 0.13 (0.99) | -0.31 | 1.14 / 1.19 (1.33 / 1.02) |
| RAD21 degron | all12refres_10kb | RAD21 absent | -0.342 (-0.556 to 0.085) | -0.388 | 0.02 (0.99) | -0.32 | 1.38 / 1.52 (1.45 / 1.05) |
| RAD21 degron | all12refres_10kb | measured ChIP, auxin vs untreated | 0.168 (-0.098 to 0.409) | 0.195 | 0.79 (0.99) | 0.10 | 1.43 / 1.41 (1.45 / 1.05) |
| RAD21 degron | all12refres_10kb | CTCF absent | -0.421 (-0.610 to -0.190) | -0.421 | 0.08 (0.99) | -0.19 | 1.38 / 1.37 (1.45 / 1.05) |
| RAD21 degron | all12refres_10kb | H3K36me3 absent (control) | 0.131 (-0.106 to 0.339) | 0.049 | 0.18 (0.99) | 0.13 | 1.38 / 1.36 (1.45 / 1.05) |
| RAD21 degron | all12refres_25kb | RAD21 absent | -0.228 (-0.526 to 0.141) | -0.352 | 0.16 (0.99) | -0.27 | 1.26 / 1.31 (1.33 / 1.02) |
| RAD21 degron | all12refres_25kb | measured ChIP, auxin vs untreated | -0.091 (-0.384 to 0.242) | 0.016 | 0.40 (0.99) | 0.07 | 1.27 / 1.30 (1.33 / 1.02) |
| RAD21 degron | all12refres_25kb | CTCF absent | -0.577 (-0.738 to -0.336) | -0.532 | 0.10 (0.99) | -0.42 | 1.26 / 1.27 (1.33 / 1.02) |
| RAD21 degron | all12refres_25kb | H3K36me3 absent (control) | 0.305 (0.010 to 0.510) | 0.166 | 0.24 (0.99) | 0.02 | 1.26 / 1.25 (1.33 / 1.02) |
| CTCF degron | all12honest_10kb | RAD21 absent | 0.057 (-0.101 to 0.271) | 0.005 | 0.24 (0.99) | 0.24 | 1.19 / 1.22 (1.70 / 1.35) |
| CTCF degron | all12honest_10kb | CTCF absent | 0.093 (-0.055 to 0.291) | 0.191 | 0.78 (0.99) | 0.05 | 1.19 / 1.09 (1.70 / 1.35) |
| CTCF degron | all12honest_10kb | H3K36me3 absent (control) | 0.117 (-0.051 to 0.325) | 0.033 | 0.36 (0.99) | 0.15 | 1.19 / 1.19 (1.70 / 1.35) |
| CTCF degron | all12honest_25kb | RAD21 absent | -0.063 (-0.285 to 0.229) | -0.107 | 0.13 (0.96) | 0.21 | 1.14 / 1.18 (1.43 / 1.26) |
| CTCF degron | all12honest_25kb | CTCF absent | 0.271 (0.003 to 0.469) | 0.315 | 0.75 (0.96) | 0.30 | 1.14 / 1.07 (1.43 / 1.26) |
| CTCF degron | all12honest_25kb | H3K36me3 absent (control) | 0.219 (-0.065 to 0.450) | 0.047 | 0.13 (0.96) | 0.10 | 1.14 / 1.19 (1.43 / 1.26) |
| CTCF degron | all12refres_10kb | RAD21 absent | -0.311 (-0.492 to 0.017) | -0.300 | 0.02 (0.99) | -0.06 | 1.38 / 1.53 (1.70 / 1.35) |
| CTCF degron | all12refres_10kb | CTCF absent | -0.371 (-0.520 to -0.161) | -0.317 | 0.08 (0.99) | -0.17 | 1.38 / 1.38 (1.70 / 1.35) |
| CTCF degron | all12refres_10kb | H3K36me3 absent (control) | 0.039 (-0.141 to 0.224) | 0.024 | 0.18 (0.99) | 0.06 | 1.38 / 1.36 (1.70 / 1.35) |
| CTCF degron | all12refres_25kb | RAD21 absent | -0.149 (-0.423 to 0.195) | -0.236 | 0.16 (0.96) | 0.10 | 1.26 / 1.31 (1.43 / 1.26) |
| CTCF degron | all12refres_25kb | CTCF absent | -0.519 (-0.653 to -0.286) | -0.409 | 0.10 (0.96) | -0.28 | 1.26 / 1.27 (1.43 / 1.26) |
| CTCF degron | all12refres_25kb | H3K36me3 absent (control) | 0.248 (0.005 to 0.463) | 0.183 | 0.24 (0.96) | 0.13 | 1.26 / 1.25 (1.43 / 1.26) |

Setting CTCF absent in the no-reference model weakens 75 to 78% of the
loops in the prediction (both experiments, both resolutions; median change
-0.09 to -0.13 against -0.26 to -0.75 measured), lowers the predicted APA
from 1.19 to 1.09 (10 kb), and the predicted loop changes rank-correlate with
the measured ones after CTCF degradation (Spearman 0.05 at 10 kb and 0.30 at
25 kb) and after RAD21 degradation (0.33 and 0.38). Setting RAD21 absent
predicts the opposite of the RAD21 degron: 2 to 24% of the loops weaken, the
predicted APA rises (1.19 to 1.22 at 10 kb, 1.38 to 1.52 for the residual
model), and the change maps anticorrelate with the measured ones (residual
model median Pearson -0.34 at 10 kb, -0.23 at 25 kb). The H3K36me3 control
changes the prediction by a similar amount (mean absolute change 0.05 to 0.17
against 0.09 to 0.12 for CTCF) and weakens 13 to 36% of the loops, yet its
change maps correlate with
the measured change about as well as the CTCF perturbation (median Pearson
0.04 to 0.31), so the pixel-level change correlation is not specific to the
perturbed protein; the loop-level measures are. With measured degron ChIP as
input the predicted change is small (mean absolute change 0.05 to 0.09
against 0.20 to 0.22 measured): 40 to 79% of the loops weaken, by a median
between -0.06 and +0.01 log O/E, and the change maps correlate at 0.17 (residual, 10 kb) or
below. In the residual models every perturbation of CTCF or RAD21 moves the
prediction against the measured change (median Pearson -0.15 to -0.58); the
leave-one-out consensus they read carries the loops in both conditions.

The 12-cell models were trained without track dropout (section 15). In
training, an absent cohesin track occurs only for HFF, a Micro-C map with
the sharpest loops of the 12 cells (no RAD21 or SMC3 ChIP), and SMC3 is
absent for HCT116; an absent RAD21 input therefore marks a cell identity in
training rather than the loss of cohesin, and the model responds with the
loop strength of that identity. CTCF is present in all 12 training cells.
Tables `perturb_validate/perturb_summary.csv` (all metrics, full grid),
`change_per_region.csv.gz`, `change_per_distance.csv`,
`loops_per_anchor.csv.gz`, `boundaries_per_bin.csv.gz`; figures
`runs/perturb_validate/figures/` (per-distance change correlation, APA
panels, the three regions with the strongest measured change per model).
Not analysed: HeLa CTCF-AID (Wutz et al. 2017) and the HAP1 CTCF-site
deletions (Sanborn et al. 2015, 20 maps, 4DN), both downloaded to
`perturb/`; neither cell has a ChIP set among the model inputs.

## 22. Track dropout in the 12-cell models (PENDING)

Models trained without track dropout collapse when tracks are missing
(section 20), read a missing RAD21 as HFF-like sharp loops (section 21) and
add little beyond the consensus for unseen cells (section 19). Runs
all12refres_drop (residual reference, per-cell leave-one-out consensus) and
all12honest_drop (no reference) at 25 and 10 kb repeat all12refres and
all12honest (section 15) with track dropout 0.5 in `uniform_count` mode (per
window a uniform number of kept tracks, the rest set to their absent value,
as `configs/full2k_proper.yaml`), honest validation, everything else equal.
all12refres_refdrop additionally withholds the reference plane in 30% of the
training windows (`train.reference_dropout`, default 0; the plane is set to
0, the expected level). Evaluated as in section 15 (413 regions, test
windows, held-out cells against consensus12) and re-scored by
`scripts/minimal_inputs.py --stage absent`, `scripts/perturb_validate.py`
and `scripts/aml_eval.py`; queued in `scripts/gpu_scheduler.py` right after
the multi-resolution model.

## 23. Cell-type-specific structure (all12refres, all12honest FINAL; dropout models PENDING)

Question: are differences between cell types (loops, domain boundaries,
compartment flips present in one cell and absent in another) predicted, or
does the model return a shared structure? `scripts/cellspecific.py` (stages
split, predict, pairref, wide, calls, e1, load, diffmap, features, showcase,
figures, summary; CPU and GPU inference workers share one queue through lock
files), tests `tests/test_cellspecific.py`, tables `cellspecific/*.csv`,
figures and per-figure npz in `runs/cellspecific/figures`.

Setup. Only the 413 Akita test regions; no training window of any cell
overlaps one of them in the 10 and 25 kb 12-cell caches
(`cellspecific/split_check.csv`, from `split_disjoint.csv`). Cells: the 12
training cells (each read with its leave-one-out consensus), K562, HMEC deep
and HUVEC deep (consensus12), MOLM-13, U-937 wild type and U-937 STAG2
knockout (consensus12; the knockout predicted from the wild-type ChIP with
RAD21 and SMC3 declared absent, section 19). Models all12refres and
all12honest at 10 and 25 kb (section 15), predicted with `--tta`. For each
of the 153 cell pairs and each region, the difference map pred_A - pred_B
is correlated with meas_A - meas_B (upper triangle, offset >= 2, pixels
defined in both measured maps and every prediction). Noise floor: two
differences from maps that share nothing (single-lane in situ rep1 and rep2
of GM12878, IMR-90, K562, HMEC, HUVEC; deep intact GM12878 and K562; MOLM-13
Arima with U-937 clone 2 against the Xu map with clone 3; U-937 WT clone 2 -
KO clone 1 against WT clone 3 - KO clone 2). Features: loops called by
hicDetectLoops of the HiCExplorer 3.7.6 release at 10 kb on every measured
map (settings of section 18), strength = log O/E of the 3 x 3 loop pixels
minus the donut ring 4 to 7 bins away; a loop is differential for a pair
when it is called in cell A, not in cell B, and its strength is at least 0.3
in A and 0.3 above B. Boundaries: minima of the insulation profile (100 kb
diamond of log O/E) with a dip of at least 0.3 in A, 0.25 above B and no
minimum of B within 2 bins. Compartment flips: genome-wide cis E1 at 100 kb
(cooltools, phased by GC) per measured map, z-scored; a flip is a run of at
least four 25 kb bins with E1 > 0.5 in A and < -0.5 in B, scored by the
contact of the flipping bins with the bins that are A in both cells minus
those that are B in both (pairs at least 75 kb apart); the measured contact
score agrees with the E1 call in 92% of the flips. Overlapping test regions
are merged, so each genomic feature counts once per pair. Direction accuracy
is the share of features whose predicted difference has the measured sign;
since one cell of a pair often has stronger loops or boundaries throughout,
a predictor that assigns every feature of a pair the same sign reaches 0.67
to 0.86 for loops and boundaries, so the balanced accuracy (features stronger in A and in B as two
classes, mean of the two class accuracies, averaged over pairs) is the
reported measure.

Difference maps, median over regions, then over pairs (all12refres /
all12honest; the consensus the model reads in the last column):

| pairs | n | 10 kb | 25 kb | consensus read (25 kb) |
|---|---|---|---|---|
| training / training | 66 | 0.654 / 0.585 | 0.703 / 0.607 | -1.000 |
| training / held-out | 36 | 0.513 / 0.446 | 0.552 / 0.466 | -0.702 |
| held-out / held-out | 3 | 0.421 / 0.307 | 0.474 / 0.358 | none (identical) |
| training / AML | 36 | 0.490 / 0.369 | 0.548 / 0.456 | -0.686 |
| held-out / AML | 9 | 0.356 / 0.292 | 0.430 / 0.337 | none |
| MOLM-13 / U-937 | 1 | 0.349 / 0.266 | 0.481 / 0.382 | none |
| U-937 WT / STAG2 KO | 1 | 0.183 / -0.027 | 0.115 / 0.089 | none |

The SD of the predicted difference is 0.38 to 0.96 of the measured one for
all12refres (training pairs 0.61 at 10 kb, 0.68 at 25 kb) and 0.77 to 1.45
for all12honest (0.78, 0.84). Among training cells the best predicted
contrasts are IMR-90 / HCT116, IMR-90 / CD14+ monocytes and HFF / CD14+
monocytes (0.83 to 0.84 at 25 kb) and the worst NHEK / MCF-7, NHEK /
GM23248 and MCF-7 / PC-9 (0.42 to 0.50). Per cell (median over its pairs, 25 kb) CD14+ monocytes lead (0.783)
and the held-out HMEC (0.462), HUVEC (0.446) and MOLM-13 (0.485) trail.
Noise floor at 25 kb: single-lane replicate differences agree at 0.56 to
0.81, deep intact GM12878 / K562 at 0.899, MOLM-13 / U-937 at 0.526 and the
U-937 knockout contrast at 0.293. all12refres reaches the replicate level
for GM12878 / IMR-90 (0.831 against 0.806); the pairs with a held-out cell
stay 0.05 to 0.32 below it (IMR-90 / K562 0.761 against 0.810, GM12878 /
K562 0.582 against 0.899, K562 / HUVEC 0.474 against 0.632, HMEC / HUVEC
0.432 against 0.622). At 10 kb the single-lane replicate differences agree
at 0.30 to 0.54, below the all12refres difference maps for every pair; one
independent replicate difference against the deep target difference agrees
at 0.38 to 0.78.

Leave-one-out references and the pair-common control. For two training
cells the references differ by (map_B - map_A) / 11, the inverted true
difference (consensus read, -1.000). Both cells of 24 pairs (all 15 among
GM12878, IMR-90, HCT116, H1, HepG2 and CD14+ monocytes, the training pairs
of the showcase and review sets, 4 training / held-out pairs) were
predicted again with one common reference, the consensus of the training
cells other than both (10 cells; for a training / held-out pair the
training cell's leave-one-out map). Median over pairs, same pixels, leave-
one-out reference / common reference / all12honest: 0.730 / 0.760 / 0.664
(10 kb, training pairs), 0.550 / 0.574 / 0.551 (10 kb, training / held-out),
0.773 / 0.796 / 0.699 and 0.567 / 0.590 / 0.572 (25 kb). The common
reference scores higher in all 48 pair x resolution cases, by 0.017 to
0.056:
the inverted difference carried by the leave-one-out references lowers the
agreement of the predicted difference with the measured one
(`cellspecific/diffmap_pairref*.csv`, `features_direction_pairref.csv`;
feature direction on these pairs, leave-one-out / common reference,
balanced: loops 0.861 / 0.895, boundaries 0.938 / 0.951, compartment flips
0.917 / 0.920).

Feature direction, all12refres (all12honest), balanced accuracy;
replicate = share of the same features whose direction an independent
replicate pair reproduces (`cellspecific/features_direction.csv`):

| feature | pairs | n | balanced | plain | pair-centred r | replicate |
|---|---|---|---|---|---|---|
| loops | training / training | 11712 | 0.818 (0.841) | 0.924 (0.888) | 0.340 (0.321) | 0.978 |
| loops | training / held-out | 5211 | 0.809 (0.811) | 0.877 (0.845) | 0.285 (0.270) | 0.87 to 0.91 |
| loops | held-out / held-out | 401 | 0.732 (0.738) | 0.793 (0.733) | 0.249 (0.240) | 0.81 to 0.83 |
| loops | with AML | 8042 | 0.655 to 0.665 (0.655 to 0.667) | 0.71 to 0.83 | 0.05 to 0.20 | 0.88 to 0.97 |
| loops | U-937 WT / KO | 34 | 0.642 (0.538) | 0.735 (0.529) | -0.137 | 0.706 |
| boundaries | training / training | 5067 | 0.919 (0.926) | 0.959 (0.927) | 0.543 (0.438) | 1.000 |
| boundaries | training / held-out | 2203 | 0.890 (0.886) | 0.916 (0.900) | 0.487 (0.400) | 0.96 to 0.99 |
| boundaries | held-out / held-out | 174 | 0.844 (0.850) | 0.862 (0.862) | 0.492 (0.352) | 0.95 to 0.98 |
| boundaries | with AML | 3796 | 0.689 to 0.752 (0.713 to 0.748) | 0.83 to 0.89 | 0.36 to 0.42 | 0.96 to 1.00 |
| compartment flips | training / training | 1829 | 0.880 (0.861) | 0.884 (0.861) | 0.612 (0.537) | 0.92 to 0.95 |
| compartment flips | training / held-out | 961 | 0.820 (0.827) | 0.839 (0.828) | 0.552 (0.488) | 0.93 |
| compartment flips | held-out / held-out | 96 | 0.792 (0.845) | 0.719 (0.771) | 0.457 (0.426) | 0.81 to 0.89 |
| compartment flips | with AML | 1648 | 0.662 to 0.749 (0.678 to 0.726) | 0.57 to 0.69 | 0.42 to 0.51 | 0.53 to 0.90 |

The consensus read gets 0.00 on training pairs (it predicts every
difference inverted) and has no direction on pairs of cells that both read
consensus12. Boundaries and compartment flips are learned for training,
held-out and, less, AML pairs; loops are learned at the level of the cell
(which cell of a pair has stronger loops) more than loop by loop: the
pair-centred correlation of loop-strength differences is 0.25 to 0.34 for
non-AML pairs and 0.05 to 0.20 with an AML cell, while replicate pairs
reproduce 0.87 to 0.98 of the loop directions. The U-937 STAG2 knockout
contrast is not learned (difference map 0.18 at 10 kb against a clone-pair
floor of 0.12 and a target-versus-replicate agreement of 0.34; 34 loops,
balanced 0.64), as in section 19. The predicted feature differences are
0.08 to 0.59 of the measured ones (median ratio per pair kind).

Showcase (`cellspecific/showcase.csv`), selected on the measured maps only
(largest measured feature difference not contradicted by an independent
replicate pair, one per genomic region): S01 loop HFF > MCF-7
chr12:114.1 Mb; S02 loop CD14+ monocytes > NHEK chr12:39.4 Mb; S03 loop HFF
> HUVEC chr16:51.1 Mb; S04 loop PC-9 > MOLM-13 chr17:49.7 Mb; S05 loop U-937
KO > WT chr5:62.9 Mb; S06 boundary IMR-90 > CD14+ monocytes chr4:7.7 Mb;
S07 boundary A549 > CD14+ monocytes chr16:77.1 Mb; S08 boundary CD14+
monocytes > HUVEC chr18:47.0 Mb; S09 boundary PC-9 > U-937 chr4:30.2 Mb;
S10, S11 compartment flip CD14+ monocytes A / HFF B chr10:18.9 and
chr5:102.0 Mb; S12 flip GM23248 / HUVEC chr15:61.7 Mb; S13 flip MOLM-13 /
IMR-90 chr21:14.5 Mb; S14, S15 largest between-cell variance share
(HFF / HepG2 chr4:32.8 Mb, IMR-90 / CD14+ monocytes chr9:134.5 Mb). Figures
`<id>_<feature>_<region>.png`: rows A, B and the cells at both extremes of
the measured feature, columns measured, all12refres, all12honest,
all12refres_drop (25 kb), consensus read, pair-common reference, per-
distance Pearson on the test region, and a last row A - B for every source
on one scale; the review set of 10 regions as `showcase80_*.png` (rows =
sources, columns GM12878, IMR-90, K562, HMEC, HUVEC, MOLM-13, U-937, A - B).
Every panel holds 80 bins: at 10 kb an 800 kb crop inside the test region,
at 25 kb the model's 2 Mb window centred on the test region with the 1 Mb
test region framed (pixels outside the frame may overlap training windows;
every metric uses the framed region only).

Dropout models (PENDING). all12refres_drop_25kb (section 22) is included
at 25 kb: difference maps 0.664 (training pairs), 0.574 (training /
held-out), 0.548 (held-out pairs), 0.569 (training / AML), 0.498 (held-out
/ AML); compartment flips balanced 0.867, 0.835, 0.792, 0.846 (training /
AML) and 0.843 (held-out / AML), above all12refres for the AML pairs (0.749
and 0.662). all12honest_drop_25kb and the 10 kb dropout runs enter when
their predictions exist (the script uses a model once its log reports the
early stop).

## 24. Hard split: windows where a cell deviates most from its consensus (PENDING)

`scripts/hard_split.py`, tables in `results/hard_split/`. Deviation score of a
window of cell c against the consensus C that c reads (leave-one-out for the
12 training cells, consensus12 for K562, HMEC and HUVEC), over the upper
triangle (offset >= 2 bins) of pixels measured in both maps:
0.5 * (1 - Pearson(T, C)) + 0.5 * var(T - C) / var(T). Candidates: the
windows of the 12-cell cache (80 bins at 25 kb, 192 at 10 kb, half-window
stride, inside Akita's training regions, disjoint from the 413 test
regions); for the held-out cells the union of these positions.

Noise-aware filters, in this order, over 44739 candidate windows (15 cells,
2 resolutions): 15 without enough measured pixels,
72 with more than 20% of bins unbalanced, 6708 in
each cell's bottom 15% of raw contacts per pixel (the cell's own unbalanced
matrix), 415 whose deviation does not reproduce between two
replicate maps (Pearson(R1 - C, R2 - C) < 0.3; gm12878 10 kb 94; gm12878 25 kb 5; hmec_deep 10 kb 128; hmec_deep 25 kb 2; huvec_deep 10 kb 161; huvec_deep 25 kb 2; imr90 10 kb 22; imr90 25 kb 1; replicate pairs
exist for GM12878 and K562, deep intact, and IMR90, HMEC and HUVEC,
single lane); 37529 kept (`filters.csv`).

Selection. The requested 3% of the genome per cell, with every hard window
of any cell removed from training for all cells, removes 62% (25 kb) and 59%
(10 kb) of all training windows, because the 15 cells' windows barely
overlap. Preferring loci another cell already claimed (a bonus added to the
score) and a smaller fraction trade removed training windows against
hardness (`tradeoff.csv`: median score of the chosen windows relative to the
pure top windows). The split uses 1.5% of the genome per cell (23 windows at
25 kb, 24 at 10 kb) with a bonus of 0.2: it removes 33.2% and
29.2% of training windows (5782 of 17397 and 5363 of
18393) at 0.99 and 0.95 of the pure top windows' median score. At most
twice a chromosome's length share of a cell's windows lie on one
chromosome. Training cells: odd chromosomes hard-validation, even
chromosomes and chrX hard-test; held-out cells: every hard window is
hard-test (121 and 140 hard-validation windows over the 12 training
cells). Median score of the hard windows against all candidate windows,
per cell, in `selection.csv` (e.g. CD14+ monocytes 2.26 against 0.51,
A549 0.49 against 0.11 at 25 kb). Files: `<k>kb/<cell>.val.bed`,
`<k>kb/<cell>.test.bed` (score in column 4), `<k>kb/split.json`.

Training (`train.hard_split`, default off): every training window of any
cell overlapping any hard window is removed; checkpoints are selected on
each cell's hard-validation windows (median per-window Pearson), and the
honest validation of the remaining windows is logged alongside. Runs
all12refres_drop_hard and all12honest_drop_hard at 25 and 10 kb
(all12refres_refdrop_hard only if reference dropout wins in section 22),
queued in `scripts/gpu_scheduler.py` after all12refres_refdrop 10 kb.
At 25 kb the hard runs train on 8664 windows (11,615 left by the union,
then 1211 honest-validation windows and 1740 overlapping them removed),
against 14,325 for all12refres_drop.
Hard-test metrics per cell: Pearson and SCC on the window, the residual
correlation corr(P - C, T - C) (0: the model adds nothing beyond the
consensus), the consensus alone, the best single other measured map and,
where replicates exist, the replicate ceiling (Pearson of the two replicate
maps and corr(R1 - C, R2 - C)).

Quick answer without retraining (`hard_split.py --stage quick`, tables
`results/hard_split/quick_*.csv`). The existing models on the 413 Akita test
regions (never trained on by any cell), per cell ranked by the same score
and filters (quick_filters.csv: of 6195 cell-regions per resolution, 941
(25 kb) and 1164 (10 kb) removed, most by the coverage rule and at 10 kb
230 by replicate reproducibility: HUVEC 128, HMEC 57, GM12878 29, IMR90
16); the hardest 20% of each cell's passing regions
(about 70) against all passing regions. The best single other map is the
one other cell whose map has the highest median over the subset; its
per-region values come from the section 15 crosscell tables (same regions,
target and pixel rule). all12refres_refdrop at 10 kb was still training
and is not included.

Hardest 20% of the passing 413 regions per cell, 25 kb, median over cells (training cells / held-out cells); Pearson, SCC, residual correlation:

| model | Pearson | SCC | residual correlation |
|---|---|---|---|
| residual reference (all12refres) | 0.880 / 0.809 | 0.857 / 0.770 | 0.759 / 0.579 |
| residual reference, track dropout (all12refres_drop) | 0.880 / 0.851 | 0.853 / 0.825 | 0.723 / 0.641 |
| residual, track and reference dropout (all12refres_refdrop) | 0.869 / 0.830 | 0.842 / 0.788 | 0.681 / 0.557 |
| multi-resolution residual (all12multires) | 0.902 / 0.824 | 0.877 / 0.797 | 0.800 / 0.582 |
| no reference (all12honest) | 0.762 / 0.629 | 0.719 / 0.618 | 0.647 / 0.512 |
| no reference, track dropout (all12honest_drop) | 0.719 / 0.686 | 0.677 / 0.633 | 0.563 / 0.674 |
| consensus alone | 0.784 / 0.751 | 0.752 / 0.707 | 0 by definition |
| best single other map | 0.811 / 0.740 | | |
| replicate maps (where they exist) | 0.929 / 0.988 | | 0.882 / 0.984 |

Hardest 20% of the passing 413 regions per cell, 10 kb, median over cells (training cells / held-out cells); Pearson, SCC, residual correlation:

| model | Pearson | SCC | residual correlation |
|---|---|---|---|
| residual reference (all12refres) | 0.888 / 0.801 | 0.839 / 0.742 | 0.773 / 0.585 |
| residual reference, track dropout (all12refres_drop) | 0.868 / 0.828 | 0.816 / 0.781 | 0.716 / 0.639 |
| multi-resolution residual (all12multires) | 0.897 / 0.813 | 0.851 / 0.763 | 0.785 / 0.597 |
| no reference (all12honest) | 0.747 / 0.668 | 0.683 / 0.618 | 0.625 / 0.497 |
| no reference, track dropout (all12honest_drop) | 0.727 / 0.680 | 0.655 / 0.638 | 0.572 / 0.585 |
| consensus alone | 0.768 / 0.695 | 0.716 / 0.648 | 0 by definition |
| best single other map | 0.755 / 0.667 | | |
| replicate maps (where they exist) | 0.805 / 0.696 | | 0.642 / 0.558 |

On the hardest regions every model loses Pearson against all passing
regions, and the consensus alone loses more (25 kb training cells 0.924 to
0.784): the regions are those where the cell departs from the consensus.
The residual reference models keep 0.880 (25 kb) and 0.888 (10 kb) on the
training cells, and their residual correlation, the part of the departure
they predict, is 0.76 and 0.77 (multi-resolution model 0.80 and 0.79),
against a replicate ceiling of 0.88 (25 kb) and 0.64 (10 kb) where
replicate pairs exist (GM12878, IMR90, K562, HMEC, HUVEC; at 10 kb the
single-lane pairs are noisier than the models' target). On the held-out
cells track dropout raises the residual correlation from 0.58 to 0.64
(25 kb) and 0.58 to 0.64 (10 kb) and Pearson from 0.809 to 0.851 and 0.801
to 0.828, while it lowers the residual correlation on the training cells
(0.76 to 0.72, 0.77 to 0.72). Reference dropout in addition lowers both at
25 kb (0.68 and 0.56), so the gated all12refres_refdrop_hard runs are not
queued. Without a reference the residual correlation is 0.50 to 0.67, so
the residual reference models add the most cell-specific signal beyond the
consensus.
