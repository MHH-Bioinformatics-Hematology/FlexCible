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
