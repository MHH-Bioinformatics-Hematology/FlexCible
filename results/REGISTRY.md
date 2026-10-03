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
| GM12878 | 138 / 230 | 0.386 | 0.589 | 0.779 | 0.938 |
| IMR90 | 166 / 199 | 0.501 | 0.712 | 0.867 | 0.959 |
| K562 | 144 / 210 | 0.359 | 0.525 | 0.764 | 0.937 |
| HMEC | 171 / 167 | 0.317 | 0.552 | 0.780 | 0.941 |
| HUVEC | 112 / 117 | 0.332 | 0.455 | 0.697 | 0.896 |

Correction (2026-10-02): the mcool levels named 5000 and 25000 that
hicBuildMatrix wrote from the 2 kb base hold 4 kb and 24 kb bins (the base
bin times floor(res / 2000)). The 5 and 25 kb columns first reported here
(GM12878 0.564 and 0.933, IMR90 0.689 and 0.956, K562 0.500 and 0.932, HMEC
0.526 and 0.936, HUVEC 0.429 and 0.888) were rep1 against rep2 at 4 and
24 kb. The maps now sit on the 5 and 25 kb grid by base-pair-overlap
averaging of the 4 and 24 kb pixels (`scripts/regrid_replicates.py`; the
originals are kept as `*_ice_4kb.cool` and `*_ice_24kb.cool`), which raises
the agreement by 0.023 to 0.026 at 5 kb and 0.003 to 0.008 at 25 kb; part of
that rise is the smoothing of the averaging.

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
0.779 (10 kb) and 0.938 (25 kb; 0.933 before the 24 kb grid correction of
section 7), deep intact replicate pairs at 0.90 to 0.97
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
| hmec | 0.880 | 0.898 | 0.883 (nhek) | 0.695 | 0.673 |  /  /  |
| huvec | 0.864 | 0.887 | 0.872 (imr90) | 0.769 | 0.744 |  /  /  |
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

The single-lane HMEC and HUVEC rows are corrected (2026-10-02): their 25 kb
targets were 24 kb maps (section 7), so the measured-map columns were empty
(grid mismatch) and the model columns, resampled onto the 24 kb grid, read
0.873, 0.691, 0.677 (HMEC) and 0.859, 0.763, 0.738 (HUVEC). On the 25 kb
grid the model stays below consensus12 for both single-lane maps, as for
the deep maps. The same re-scoring of every 25 kb and 5 kb model on these
two targets changes the model medians by 0.000 to 0.012 at 25 kb and by
0.012 to 0.016 at 5 kb
(`*/crosscell/predict_{hmec,huvec}.grid24kb.csv` keep the old values).

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
5 and 2 kb reference models for the same comparison are queued. The first
2 kb all12refres and all12casc starts (2026-09-30 11:35 and 14:55) were
stopped by earlyoom (SIGTERM to both ranks during loading): each rank held
the whole 2 kb train split with a 512 x 512 target and a reference (and
prior) plane in float16 in RAM, about 40 GB per rank, next to other jobs;
all12honest (no reference plane) fitted. These runs now read their windows
from memory maps staged once on local disk (`train.stage_dir`,
`python -m hicmultiome.train --stage-only`), one batch at a time. Test
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
1 Mb reach; cross-validation within each chromosome, so the scored pixels'
neighbours are training pixels: see "HiC-Reg training split" below). Ours: akita_native_honest and
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
against 0.71 and its loop recall 0.33 against 0.44. These HiC-Reg maps are
cross-validation predictions within the chromosome (next subsection); its
leak-free cross-chromosome predictions score 0.66. On C.Origami's held-out chromosomes (45 IMR-90
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

HiC-Reg training split (`scripts/hicreg_audit.py`, tables
`benchmark/hicreg_*.csv`). The section 18 HiC-Reg maps are the authors'
published GM12878 predictions, files `Gm12878_chr<N>_RF_tree20_prediction`
(5 kb, log(count + 1), maximum distance 1 Mb), which is the file name of
HiC-Reg's cross-validation (CV) mode; the HiC-Reg Zenodo release (records
3525432, 3525510, 3525514) holds per cell the CV predictions, the five CV
models, cross-chromosome predictions (model of chr17 applied to chr14 and
vice versa) and cross-cell predictions for chr14 and chr17. The released
models carry their training pair lists (`trainset_error.txt`), from which
the folds follow (`hicreg_split.csv`; GM12878 and K562, WINDOW features,
chr14 and chr17): each chromosome's 5 kb regions are assigned to five folds
at random (adjacent regions fall in different folds in 79.8 to 79.9% of the
cases, 80% expected for random assignment), each model trains on the bin
pairs with both regions outside its fold (64.0% of the chromosome's pairs)
and predicts the rest, so every pixel is predicted by a model trained on the
same chromosome of the same cell, on the same Hi-C map. Of the 8 neighbouring
5 kb pixels of a predicted pixel, 19% are training pixels of every model
that can have predicted it and 69% of at least one (GM12878; K562 13 to 15%
and 49 to 57%); 72 to 98% of the predicted pixels have a training pixel
among their 8 neighbours, and each 25 kb pixel of section 18 sums 25 such
5 kb predictions. The published chr1-22 maps match the released CV
predictions and not the cross-chromosome ones: on chr14 their
distance-centred log counts correlate at 0.970 with the CV predictions and
at 0.665 to 0.678 with the cross-chromosome predictions
(`hicreg_published_vs_released.csv`; the Zenodo data archive, which would
document the chr1-22 run, ended early in transfer with an unexpected end
of stream and was not completed). The section 18
HiC-Reg score is therefore a within-chromosome interpolation score: the
model saw neighbouring pixels of the same map.

Leak-free HiC-Reg. The released cross-chromosome predictions (model trained
on all pairs of the other chromosome, chr17 for chr14 and chr14 for chr17,
same cell, same features) are HiC-Reg's own leak-free setting, available
for chr14 and chr17 only (feature files for other chromosomes are not
released, so a genome-wide odd-to-even retraining is not cheap and was not
run). All released maps go through the section 18 conversion
(exp(value) - 1, summed into 10 or 25 kb bins, `oe_proper.py`, 1 Mb reach);
the published set is also rescored in this count space, which raises it
from 0.943 to 0.959 (block C2; section 18 summed the log values). Blocks
F: the 23 Akita test regions on chr14 and chr17 (13 and 10), identical
pixels, median per-region Pearson / SCC:

| block | all12refres | all12honest | HiC-Reg CV, WINDOW / MULTICELL | HiC-Reg cross-chromosome, WINDOW / MULTICELL | HiC-Reg cross-cell (other cell's model) | consensus | best other map |
|---|---|---|---|---|---|---|---|
| GM12878 25 kb | 0.957 / 0.898 | 0.813 / 0.683 | 0.961 / 0.976 (SCC 0.929 / 0.953) | 0.660 / 0.712 (SCC 0.434 / 0.477) | 0.668 | 0.923 | 0.919 |
| GM12878 10 kb | 0.935 / 0.889 | 0.781 / 0.730 | 0.937 / 0.961 | 0.609 / 0.634 | 0.597 | 0.898 | 0.870 |
| K562 25 kb | 0.909 / 0.830 | 0.792 / 0.604 | 0.949 / 0.965 | 0.845 / 0.858 | 0.815 | 0.914 | 0.889 |
| K562 10 kb | 0.904 / 0.823 | 0.741 / 0.565 | 0.907 / 0.928 | 0.715 / 0.760 | 0.774 | 0.897 | 0.852 |
| C2 GM12878 25 kb even (218) | 0.945 / 0.910 | 0.852 / 0.798 | published CV: 0.943 (log sum), 0.959 (counts) | | | 0.880 | 0.838 |

Without the within-chromosome leak HiC-Reg falls by 0.30 (GM12878, 25 kb,
WINDOW) and by 0.10 (K562): all12honest exceeds the GM12878
cross-chromosome predictions by a median 0.17 at 25 kb and 0.14 at 10 kb
(87% of the regions, p = 1e-4 and 3e-4) and all12refres by 0.27 and 0.32
(all regions); for K562 all12honest is level with the cross-chromosome
and cross-cell predictions (median differences -0.02 to 0.04, p = 0.20 to
0.99) and all12refres exceeds them by 0.05 to 0.13 (78 to 87% of the
regions, p = 3e-4 to 0.004). With the leak, the CV predictions score level
with all12refres for GM12878 (25 kb 0.961 against 0.957, p = 0.16; 10 kb
0.937 against 0.935, p = 0.89) and for K562 at 10 kb (0.907 against 0.904,
p = 0.15), above it for K562 at 25 kb (0.949 against 0.909, p = 0.008) and
with the MULTICELL features (0.976 against 0.957 for GM12878 at 25 kb,
p = 0.03),
and the published set in count space exceeds all12refres on the 218 even-
chromosome regions (0.959 against 0.945, all12refres higher in 33% of the
regions, p < 1e-4). The 25 kb tie of section 18 is a comparison with
within-chromosome interpolation; on held-out chromosomes HiC-Reg ranks below
both 12-cell models for GM12878 and below all12refres for K562.

Structure on identical pixels (blocks C2 and F; `<block>_summary.csv`,
`hicreg_distance_bands.csv`, `hicreg_local_e1.csv`). Distance bands
(median over regions of the per-region Pearson per 25 kb diagonal, up to
950 kb; the regions span 1 Mb and HiC-Reg reaches 1 Mb, so no band beyond
1 Mb exists for it): on the even chromosomes the published CV maps score
0.59 at 50 kb, 0.80 at 100 kb, 0.96 at 250 kb and 0.99 at 500 kb, against
0.86, 0.90, 0.94 and 0.95 for all12refres and 0.74, 0.76, 0.83 and 0.88 for
all12honest; the cross-chromosome HiC-Reg (GM12878, chr14 and chr17)
scores 0.28, 0.40, 0.58 and 0.58 at the same distances. Loops and
boundaries (25 kb, GM12878): loop recall CV 0.30 to 0.34, cross-chromosome
0.06 to 0.09, all12refres 0.36, all12honest 0.21; APA ratio CV 0.66 to
0.80, cross-chromosome -0.05 to 0.08, all12refres 0.65; boundary recall CV
0.64 to 0.71, cross-chromosome 0.37 to 0.40, all12refres 0.60, consensus
0.71; insulation-profile Pearson CV 0.82 to 0.91, cross-chromosome 0.43 to
0.44, all12refres 0.85. Local compartment structure (absolute Pearson of
the first eigenvector of the region's O/E row correlation, target against
prediction; a 1 Mb region carries local A/B structure only, and HiC-Reg's
1 Mb reach excludes chromosome-wide E1): published CV 0.986, all12refres
0.975, all12honest 0.950, consensus 0.939 (block C2); cross-chromosome
HiC-Reg 0.81 (WINDOW) and 0.88 (MULTICELL) against 0.975 for all12refres
(GM12878, block F).

Cell-specific difference (`hicreg_cellspecific.csv`): the measured GM12878
minus K562 O/E difference on the 23 chr14/chr17 regions against each
method's GM12878 minus K562 prediction, median per-region Pearson (25 kb /
10 kb): HiC-Reg CV 0.92 / 0.81 (WINDOW) and 0.94 / 0.83 (MULTICELL);
HiC-Reg cross-chromosome 0.33 / 0.38 and 0.37 / 0.41; HiC-Reg cross-cell
(each cell predicted by the other cell's model from its own features) 0.42 /
0.38; all12refres 0.53 / 0.51; all12honest 0.46 / 0.36. The CV maps
reproduce the cell difference because each cell's CV model is trained on
that cell's map of the same chromosome; held out by chromosome, HiC-Reg
distinguishes the two cells less than all12refres at 25 kb and 10 kb and
about as well as all12honest. The predicted difference of all12refres has
a standard deviation of 0.12 against 0.17 to 0.22 measured (HiC-Reg 0.12 to
0.29).

Noise-corrected 2 kb scores (`noise_corrected_2kb.csv`). Per region r /
sqrt(r_replicate), with r_replicate the GM12878 replicate agreement at 2 kb
on the same region (`runs/worst_regions/replicates_per_region.csv`, proper
O/E; regions with r_replicate < 0.1 left out, 411 regions kept), block A
GM12878, median over regions. The replicate maps are shallower than the
4DN target (single-lane in situ pair, median r_replicate 0.386; deepest
intact pair ENCSR965OVH, 0.520), so the corrected values are upper bounds
and exceed 1 where the target is less noisy than the replicates; HFF has
no replicate pair and no corrected score.

| method | raw Pearson | corrected, in situ lane pair | corrected, intact ENCSR965OVH | raw SCC | corrected SCC, in situ / intact |
|---|---|---|---|---|---|
| akita_native_refres | 0.746 | 1.184 | 1.035 | 0.698 | 1.114 / 0.974 |
| akita_native_honest | 0.640 | 1.001 | 0.884 | 0.611 | 0.960 / 0.844 |
| consensus4 | 0.612 | 0.986 | 0.872 | 0.559 | 0.890 / 0.779 |
| Akita | 0.552 | 0.872 | 0.768 | 0.512 | 0.816 / 0.708 |
| best other map (H1) | 0.516 | 0.827 | 0.725 | 0.457 | 0.755 / 0.645 |
| Hi-cGAN (submitted) | 0.238 | 0.369 | 0.335 | 0.208 | 0.340 / 0.304 |

The correction keeps the ranking of section 18; it compresses the gap
between models where replicate agreement is low.

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
Noise floor at 25 kb: single-lane replicate differences agree at 0.58 to
0.82, deep intact GM12878 / K562 at 0.899, MOLM-13 / U-937 at 0.526 and the
U-937 knockout contrast at 0.293. all12refres reaches the replicate level
for GM12878 / IMR-90 (0.831 against 0.813); the pairs with a held-out cell
stay 0.06 to 0.32 below it (IMR-90 / K562 0.761 against 0.819, GM12878 /
K562 0.582 against 0.899, K562 / HUVEC 0.474 against 0.645, HMEC / HUVEC
0.432 against 0.635). Corrected 2026-10-02: the single-lane replicate maps
were first read on their 24 kb grid (section 7), which gave 0.56 to 0.81
and 0.806, 0.810, 0.632, 0.622 for the four pairs above. At 10 kb the single-lane replicate differences agree
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
| compartment flips | training / training | 1829 | 0.880 (0.861) | 0.884 (0.861) | 0.612 (0.537) | 0.95 |
| compartment flips | training / held-out | 961 | 0.820 (0.827) | 0.839 (0.828) | 0.552 (0.488) | 0.93 to 0.95 |
| compartment flips | held-out / held-out | 96 | 0.792 (0.845) | 0.719 (0.771) | 0.457 (0.426) | 0.83 to 0.89 |
| compartment flips | with AML | 1648 | 0.662 to 0.749 (0.678 to 0.726) | 0.57 to 0.69 | 0.42 to 0.51 | 0.53 to 0.90 |

The replicate column of the compartment flips (25 kb) is corrected for the
24 kb grid of the single-lane replicate maps (section 7); before, it read
0.92 to 0.95, 0.93 and 0.81 to 0.89 for the first three rows; the
compartment flip masks are on the 25 kb grid, so the old values compared
misaligned bins.

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

Hard-split trainings (stopped by patience; best hard-validation median
Pearson): all12refres_drop_hard 0.789 at 25 kb (epoch 55) and 0.748 at 10
kb (epoch 37), all12honest_drop_hard 0.701 (epoch 20) and 0.661 (epoch 32).
Hard-test evaluation (`hard_split.py --stage hardeval`, tables
`results/hard_split/hardtest_*.csv`): 12 hard-test windows per training cell
on average and 23 to 24 per held-out cell. Only the hard-split models never
saw these loci: for the training cells the hard windows come from the
training grid, so the other models were trained on them for the same cell
and their rows there are in-sample (listed for reference); for the held-out
cells those models were trained on the loci for other cells.

Hard-test windows, 25 kb, median over cells, training cells / held-out cells (Pearson; SCC; residual correlation):

| model | Pearson | SCC | residual correlation |
|---|---|---|---|
| reference, track dropout, hard split | 0.828 / 0.710 | 0.824 / 0.747 | 0.787 / 0.656 |
| no reference, track dropout, hard split | 0.661 / 0.685 | 0.648 / 0.657 | 0.761 / 0.713 |
| reference (trained on these loci) | 0.860 / 0.745 | 0.839 / 0.692 | 0.869 / 0.635 |
| reference, track dropout (trained on these loci) | 0.846 / 0.747 | 0.828 / 0.783 | 0.837 / 0.692 |
| multi-resolution reference (trained on these loci) | 0.909 / 0.709 | 0.865 / 0.749 | 0.917 / 0.689 |
| no reference, track dropout (trained on these loci) | 0.760 / 0.682 | 0.708 / 0.687 | 0.797 / 0.766 |
| consensus alone | 0.538 / 0.543 | 0.567 / 0.560 | 0 |
| best single other map | 0.668 / 0.565 | | |
| replicate maps (GM12878, IMR90; K562, HMEC, HUVEC) | 0.884 / 0.861 | | 0.865 / 0.842 |

Hard-test windows, 10 kb, median over cells, training cells / held-out cells (Pearson; SCC; residual correlation):

| model | Pearson | SCC | residual correlation |
|---|---|---|---|
| reference, track dropout, hard split | 0.755 / 0.701 | 0.724 / 0.659 | 0.759 / 0.652 |
| no reference, track dropout, hard split | 0.633 / 0.662 | 0.593 / 0.558 | 0.685 / 0.599 |
| reference (trained on these loci) | 0.840 / 0.673 | 0.803 / 0.580 | 0.843 / 0.560 |
| reference, track dropout (trained on these loci) | 0.808 / 0.755 | 0.788 / 0.687 | 0.791 / 0.663 |
| multi-resolution reference (trained on these loci) | 0.854 / 0.696 | 0.818 / 0.636 | 0.859 / 0.640 |
| no reference, track dropout (trained on these loci) | 0.693 / 0.630 | 0.650 / 0.566 | 0.704 / 0.571 |
| consensus alone | 0.559 / 0.541 | 0.598 / 0.539 | 0 |
| best single other map | 0.604 / 0.496 | | |
| replicate maps (GM12878, IMR90; K562, HMEC, HUVEC) | 0.730 / 0.621 | | 0.683 / 0.567 |

On windows where the cell departs most from its consensus the consensus
alone scores 0.54 to 0.56 and the best single other map 0.60 to 0.67 (0.50
to 0.56 for held-out cells). The hard-split residual reference model scores
0.828 at 25 kb and 0.755 at 10 kb on the training cells and predicts the
departure from the consensus with a residual correlation of 0.79 and 0.76,
against a replicate ceiling of 0.87 at 25 kb (at 10 kb the single-lane
replicate pairs, 0.68, are noisier than the target). Without a reference
the hard-split model scores 0.661 and 0.633 with residual correlations of
0.76 and 0.69. On the held-out cells the hard-split reference model scores
0.710 and 0.701 (residual correlation 0.66 and 0.65), 0.04 to 0.05 below the
dropout model trained on these loci for other cells (0.747 and 0.755),
which quantifies what seeing a locus in other cells contributes there.

Correction (2026-10-02, single-lane replicate maps on a 24 kb grid, section
7): at 25 kb the region-level replicate columns skipped IMR-90, HMEC and
HUVEC (grid mismatch), so the replicate row of the 25 kb hard-test table
stood for GM12878 (training) and K562 (held out) alone (0.874 / 0.978,
residual 0.868 / 0.982); with the regridded maps
(`scripts/hard_split.py --stage repfill`) it is 0.884 / 0.861 (residual
0.865 / 0.842). The window-level replicate filter of the selection
(rep_dev_r of IMR-90, HMEC deep and HUVEC deep at 25 kb) compared the 24 kb
replicate band with the 25 kb target band bin by bin, which inflated the
agreement (median 0.96, 0.95 and 0.92; 0.79, 0.68 and 0.57 on the regridded
maps). Reselected with the corrected values
(`results/hard_split_regrid/25kb/split.json`), 4 training cells (IMR-90,
HFF, NHEK, MCF-7) change one of their 23 hard windows each and the union
holds 228 instead of 229 windows, 226 shared. The 25 kb hard-split models
keep the original split, which differs in 3 of 229 union windows. The 10 kb
split is unchanged.

## 25. Cohesin response (FINAL)

Problem (sections 19 and 21): removing RAD21 at inference made the models
predict stronger loops, opposite to the measured RAD21 degron; HFF, the only
training cell without cohesin ChIP, is Micro-C with the sharpest loops, so
"no cohesin input" reads as "HFF-like".

1b, the RAD21 degron as training data (runs all12deg_refres_drop and
all12deg_honest_drop, 25 and 10 kb, track dropout 0.5, honest validation).
Rao et al. 2017 HCT116 RAD21-mAC: the existing hct116 training target is
already this study's untreated Hi-C (4DN 4DNES3QAGOZZ), so the untreated
condition is not added a second time; in these runs hct116 reads the
matched untreated degron ChIP (Kim et al. 2024 RAD21 and CTCF, Rao 2017
histone marks, H3K4me2/H3K9ac/H4K20me1 from ENCODE HCT116; no SMC3) instead
of the ENCODE HCT116 set, and a 13th sample hct116_auxin holds the 6 h auxin
Hi-C (4DNESJV9TH8Q, `oe_proper.py`) with the auxin ChIP, on the odd autosomes
only (`samples[].train_chroms`). Its reference is the HCT116 leave-one-out
consensus. The RAD21 degron is scored on the even chromosomes (never seen
under auxin); the HCT116 CTCF degron (ENCODE intact) and the U-937 STAG2
knockout stay fully held out.

1a, HFF cohesin ChIP. No RAD21, SMC3, STAG1/2 or NIPBL ChIP of HFF or HFFc6
exists on ENCODE; ChIP-Atlas holds two HFFc6 SMC1A experiments (SRX9995801,
30.6 M reads, 44,030 peaks; SRX9995802, 19.9 M reads, 32,989 peaks), the
cohesin ring subunit that co-occupies RAD21 and SMC3 sites, and RAD21/SMC3 of
BJ, a different foreskin fibroblast line (not used). Run
all12hffcoh_refres_drop (25 and 10 kb) gives HFF the two SMC1A replicates in
the RAD21 and SMC3 slots (`aml_fetch.py --cells hffcoh`, rescaled to the
training cells' genome-wide mean), so no training cell lacks cohesin.

1c (cohesin-unknown indicator) is run after 1a and 1b left the absent-RAD21
direction wrong; design and results in section 32.

Washout (held-out test, `scripts/washout.py`): Rao 2017 measured Hi-C 20, 40,
60 and 180 min after auxin withdrawal (4DN 4DNESCOJ3ADI, 4DNES7RDRS69,
4DNES4KDIQNN, 4DNESOGIV7NG, GRCh38 mcools). Kim 2024 ChIP exists only at 9 h
and 12 h of washout, so no time point has matching ChIP; the maps are never
trained on and are predicted approximately with the RAD21 track interpolated
between the auxin and the untreated track. Nora et al. 2017 (CTCF degron with
washout) is mouse ES cells and is not used by this hg38 model.

Washout and auxin baseline, models without degron training
(`results/washout/washout_summary.csv`): HCT116 predicted from the untreated
degron ChIP with only the RAD21 track replaced by (1 - f) * auxin + f *
untreated, at the loops called in the untreated map (536 at 10 kb, 549 at
25 kb, on the 413 test regions; even chromosomes alone in brackets). Measured:
99.4% of these loops are weaker after 6 h auxin (median log O/E change -0.75
at 10 kb, -0.48 at 25 kb), and the loop strength recovers to 0.18, 0.48,
0.66 and 1.12 (10 kb) and 0.15, 0.41, 0.60 and 1.01 (25 kb) of the untreated
level at 20, 40, 60 and 180 min of washout.

| model | res | loops weaker with auxin RAD21 | median predicted change | f matching 20 / 40 / 60 / 180 min |
|---|---|---|---|---|
| all12refres_drop | 10 kb | 89.7% (89.4%) | -0.050 | 0.11 / 0.30 / 0.45 / 1.0 |
| all12refres_drop | 25 kb | 78.0% (81.5%) | -0.033 | 0.10 / 0.29 / 0.48 / 1.0 |
| all12refres | 10 kb | 88.3% (85.1%) | -0.062 | 0.13 / 0.37 / 0.53 / 1.0 |
| all12refres | 25 kb | 57.0% (57.3%) | -0.008 | 0.07 / 0.20 / 0.34 / 1.0 |
| all12honest | 10 kb | 76.9% (77.3%) | -0.061 | 0.08 / 0.22 / 0.35 / 1.0 |
| all12honest | 25 kb | 67.6% (70.8%) | -0.024 | 0.08 / 0.21 / 0.36 / 1.0 |
| all12honest_drop | 10 kb | 71.1% (72.2%) | -0.043 | not monotone |
| all12honest_drop | 25 kb | 55.6% (58.9%) | -0.006 | not monotone |

With the measured auxin RAD21 track (reduced, not absent) the models point
in the measured direction at most loops, unlike the absent-RAD21 test of
section 21, but the predicted weakening is 12 to 80 times smaller than the
measured one, and 10 to 44% of the loops still strengthen. The predicted
loop strength rises with the RAD21 fraction f, so recovery is ordered in
time: the measured 20, 40 and 60 min states match RAD21 at about 10, 30
and 45% of its untreated level for the 10 kb dropout model (an
approximation: the actual RAD21 occupancy at these times is not measured).

Degron-trained and HFF-cohesin models (`washout_summary.csv`, all 536 / 549
loops; the auxin condition was trained on odd chromosomes only, the even
chromosomes alone give the same picture):

| model | res | loops weaker with auxin RAD21 | median predicted change (measured -0.75 / -0.48) | f matching 20 / 40 / 60 / 180 min |
|---|---|---|---|---|
| all12deg_refres_drop | 10 kb | 99.1% (98.4%) | -0.381 | 0.09 / 0.24 / 0.39 / 1.0 |
| all12deg_refres_drop | 25 kb | 96.0% (98.9%) | -0.105 | 0.09 / 0.24 / 0.44 / 1.0 |
| all12deg_honest_drop | 10 kb | 98.7% (98.8%) | -0.172 | 0.17 / 0.46 / 0.64 / 1.0 |
| all12deg_honest_drop | 25 kb | 90.7% (90.0%) | -0.074 | 0.14 / 0.38 / 0.57 / 1.0 |
| all12hffcoh_refres_drop | 10 kb | 84.7% (83.5%) | -0.054 | 0.09 / 0.25 / 0.40 / 1.0 |
| all12hffcoh_refres_drop | 25 kb | 92.4% (90.8%) | -0.054 | 0.10 / 0.28 / 0.47 / 1.0 |

Training on the auxin condition brings the predicted weakening to half of
the measured one at 10 kb (-0.381 against -0.754 for the reference model,
against -0.050 without degron training) and weakens 96 to 99% of the loops.
For the no-reference degron model the predicted recovery is close to linear
in the RAD21 fraction: the measured 20, 40 and 60 min states match f = 0.17,
0.46 and 0.64, against measured recoveries of 0.18, 0.48 and 0.66. SMC1A as
HFF's cohesin input moves the predictions much less (median change -0.054).

Perturbation tests (`perturb_validate/perturb_summary.csv`, even
chromosomes, never trained under auxin; change Pearson = correlation of the
predicted with the measured auxin-minus-untreated difference per region,
median over regions):

| model | res | RAD21 degron, measured auxin ChIP: change Pearson / loops weaker / loop-change Spearman | RAD21 set absent: loops weaker | CTCF degron (held out, CTCF absent): change Pearson / loops weaker |
|---|---|---|---|---|
| all12refres_drop | 10 kb | 0.417 / 89.4% / 0.04 | 21.2% | 0.212 / 89.0% |
| all12refres_drop | 25 kb | 0.351 / 85.0% / 0.21 | 10.0% | 0.010 / 60.5% |
| all12deg_refres_drop | 10 kb | 0.803 / 98.4% / 0.57 | 20.8% | 0.396 / 94.6% |
| all12deg_refres_drop | 25 kb | 0.842 / 95.8% / 0.62 | 33.1% | 0.517 / 82.9% |
| all12deg_honest_drop | 10 kb | 0.475 / 95.3% / 0.36 | 34.1% | 0.351 / 91.2% |
| all12deg_honest_drop | 25 kb | 0.419 / 83.5% / 0.46 | 30.0% | 0.403 / 77.6% |
| all12hffcoh_refres_drop | 10 kb | 0.456 / 93.3% / -0.01 | 36.9% | 0.116 / 80.7% |
| all12hffcoh_refres_drop | 25 kb | 0.503 / 90.0% / 0.24 | 8.9% | -0.176 / 18.8% |

With the measured auxin ChIP the degron-trained reference model predicts the
degron difference map at 0.80 to 0.84 on chromosomes it never saw under
auxin, against 0.35 to 0.42 for the same model without degron training, and
it generalises to the held-out CTCF degron (0.40 and 0.52 against 0.21 and
0.01). Declaring RAD21 absent still strengthens most loops in every model
(21 to 37% weaker at 10 kb), so the training condition (RAD21 reduced, ChIP
measured) does not transfer to an absent track; on the U-937 STAG2 knockout
with cohesin declared absent (`aml/perturb_summary.csv`) the loop-change
Spearman stays negative at 10 kb (all12deg_refres_drop -0.44,
all12deg_honest_drop -0.20, all12hffcoh_refres_drop -0.05, all12refres_drop
-0.34). HFF with SMC1A in the cohesin slots leaves the absent-RAD21
direction wrong (8.9 to 36.9% of loops weaker) and lowers the CTCF degron
agreement at 25 kb (-0.176), so 1a does not fix the cohesin response; 1b
fixes it for measured cohesin ChIP. 1c (cohesin-unknown indicator) remains
the open option for samples without cohesin ChIP.

## 26. Gain beyond the consensus on new cells (PENDING)

(a) all12refin_drop: input-only reference (model.reference input) with track
dropout, and all12refin_drop_ctcfac, the same reading CTCF and H3K27ac only.
(b) all12refres_drop_varw: residual reference with track dropout and
`train.loss_weighting` between_cell_var (per pixel 1 + v / mean(v), v the
variance across the training cells' maps at that locus and pixel, from
training windows only; mean 1 per batch). (c) residual shrinkage for unseen
cells, C + lambda * (P - C), lambda chosen per model and resolution on the
held-out cells' Akita validation regions (419, disjoint from the 413 test
regions), then applied to their test regions (`hard_split.py --shrink`,
`shrink_summary.csv`). Key numbers: model above the consensus alone on the
held-out cells (K562, HMEC deep, HUVEC deep; AML MOLM-13 and U-937 through
`aml_eval.py`) and the residual correlation on the hardest regions.

(c) Residual shrinkage, existing models (`results/hard_split/shrink_beats.csv`,
`shrink_summary.csv`). lambda in 0 to 1.5 (steps of 0.25) maximises the
median over K562, HMEC deep and HUVEC deep of their median Pearson on Akita's
419 validation regions; the test numbers are the 413 test regions, never
used for the choice. Median Pearson per held-out cell, cells above
consensus12 alone:

| res | model | lambda | model above consensus | shrunk above consensus | median model / shrunk / consensus |
|---|---|---|---|---|---|
| 25 kb | all12refres | 0.25 | 1/3 | 2/3 | 0.906 / 0.916 / 0.916 |
| 25 kb | all12refres_drop | 0.5 | 1/3 | 3/3 | 0.918 / 0.920 / 0.916 |
| 25 kb | all12refin | 0.25 | 1/3 | 2/3 | 0.901 / 0.913 / 0.916 |
| 25 kb | all12multires | 0.25 | 1/3 | 3/3 | 0.904 / 0.918 / 0.916 |
| 25 kb | all12refres_refdrop | 0.5 | 1/3 | 3/3 | 0.918 / 0.918 / 0.916 |
| 10 kb | all12refres | 0.5 | 1/3 | 3/3 | 0.885 / 0.890 / 0.884 |
| 10 kb | all12refres_drop | 0.5 | 3/3 | 3/3 | 0.905 / 0.896 / 0.884 |
| 10 kb | all12refin | 0.75 | 2/3 | 3/3 | 0.896 / 0.894 / 0.884 |
| 10 kb | all12multires | 0.5 | 1/3 | 3/3 | 0.892 / 0.891 / 0.884 |
| 10 kb | all12refres_refdrop | 0.5 | 2/3 | 3/3 | 0.901 / 0.893 / 0.884 |

The chosen lambda is 0.25 to 0.75 for every model: on unseen cells the
predicted deviation from the consensus is too large, and halving it raises
HMEC and HUVEC (deep maps) by 0.01 to 0.03 at 25 kb while K562, the one
held-out cell the models already predict above the consensus, loses up to
0.013. With shrinkage every model except all12refres and all12refin at 25 kb
is above the consensus on all three held-out cells, but by at most 0.005 at
25 kb (HMEC 0.934 to 0.938 against 0.933, HUVEC 0.916 to 0.918 against
0.916) and 0.003 to 0.033 at 10 kb. Without shrinkage, all12refres_drop at
10 kb already exceeds the consensus on all three (0.911, 0.897, 0.905
against 0.908, 0.884, 0.859) with the highest median (0.905); shrinkage
lowers its median to 0.897.

## 27. Mean DNA methylation as an input (PENDING)

Model all12refres_drop_meth (10 kb): all12refres_drop with a 14th track,
Meth, the mean CpG methylation per 1 kb bin from one ENCODE4 GRCh38 WGBS
file per cell (scripts/methylation.py; bin value 1 + coverage-weighted
methylated fraction, bins under 5 CpG reads at 1 + the genome mean, so a raw
zero stays "no data"). Seven of the 12 training cells have WGBS (GM12878,
IMR-90, H1, HepG2, A549, GM23248, CD14+ monocytes); HCT116, MCF-7 and HFF
(hg19 RRBS only), NHEK and PC-9 carry the track as missing. Held-out cells
with WGBS: K562 and HMEC. Mean bin values range from 1.22 (K562) to 1.85
(H1). Methylation stands in for marks primary samples lack: partially
methylated domains coincide with H3K9me3 heterochromatin and gene-body
methylation with H3K36me3. Evaluation (scripts/meth_eval.py) against
all12refres_drop_10kb with full inputs and with only CTCF, H3K27ac and
H3K27me3 present, the marks of the Xu 2022 primary AML samples, each with
and without Meth; cells K562, HMEC deep, GM12878 and IMR-90.

First training excluded (run renamed all12refres_drop_meth_selbug_10kb,
`results/methylation/meth_eval_summary_selbug.csv`): the base configuration
carries train.select_tracks with the 13 ChIP tracks, which dropped the Meth
row from the model input although the cache holds it (7 cells with WGBS,
normaliser mean 0.957, sd 0.105 on log1p values). The run log shows 23 sym
rows, the same as without Meth, and predictions with Meth present and set
absent are identical (maximum absolute difference 0.0 over 9 K562 test
regions), hence identical medians for full and full_nometh (K562 0.9045)
and for patient and patient_nometh (K562 0.8912). That run is in effect a
second training of all12refres_drop_10kb. Fixed in
scripts/multicell_expanded.py (tracks added to the 13 are appended to
select_tracks, also for all12refres_drop_cnv, which had not started);
all12refres_drop_meth_10kb is retrained.

## 28. Copy number as an input and a structural-variant loss mask (PENDING)

Model all12refres_drop_cnv (10 kb): all12refres_drop with two more tracks,
CNVgain and CNVloss (positive and negative part of the called log2 copy
ratio, raw zero = copy neutral), and train.sv_mask: in a training window
holding a breakpoint, every pixel whose two bins lie on opposite sides of it
gets loss weight 0. Validation and test are unmasked.

Copy number (scripts/cnv_sv.py) comes from Hi-C coverage in 100 kb bins:
log2 ratio to the median of the 15 maps, corrected for GC content (median
per GC class; uncorrected, coverage follows GC with |r| up to 0.79 in
copy-neutral cells), binary segmentation (segments of at least 1 Mb, t above
8), segments with |log2| of 0.3 or more called, values clipped at 3. GC
correction lowers the 2 Mb-scale spread of GM12878 from 0.32 to 0.15 and of
HFF from 0.31 to 0.16. Called fraction of the autosomes
(`results/cnv_sv/cnv_sv_summary.csv`): MCF-7 0.52, A549 0.47, PC-9 0.46,
K562 0.45, HepG2 0.27, HCT116 0.12; GM12878 0.05, CD14+ monocytes 0.05,
IMR-90 0.04, NHEK 0.03, H1 0.02, GM23248 and HMEC 0.004 to 0.005. HFF (Micro-C)
reaches 0.08 and the held-out HUVEC deep map 0.28, which we read as residual
protocol bias of those maps. The calls recover the known K562 lesions: the
homozygous 9p21 deletion around CDKN2A (chr9:20.8-25.3 Mb, log2 -8), the
BCR amplicon on chr22 (up to 23.3 Mb, log2 2.0 to 2.6) and the ABL1 region
on 9q34 (130.4-131.4 Mb, log2 1.6); HCT116 shows gains on chr8, 10, 16 and
17, as in its karyotype. The coverage of a map is a one-dimensional summary
of the measured Hi-C, for held-out cells too; the channel stands for a
copy-number profile from WGS or arrays.

Structural variants: EagleC2 2.1.1 predictSV (pretrained models) on the
ICE-balanced 10 and 25 kb maps of HepG2, A549, MCF-7, PC-9, HCT116 and K562;
both ends of each call are breakpoints. The karyotypically normal cells
carry no mask. Comparison (`results/cnv_sv/cnv_compare.csv`) against
all12refres_drop_10kb per cell, cancer lines apart.

## 29. Primary AML patients (FINAL)

`scripts/aml_primary.py` (stages fetch, fetch_pairs, targets, inputs,
predict, amlref, score, specific, fischer, figures),
`scripts/aml_primary_explain.py` (cnv, annotate, scores, learn,
features_predict, features, figures), `scripts/aml_sv_predict.py`; tests
`tests/test_aml_primary.py`. Data under
`/mnt/processed/wolffjoa/hic-multiome/aml/primary/` (`../STATUS_primary.md`).
Tables in `results/aml_primary/`.

Data. Xu et al., Nature 2022, GSE152136: in situ MboI Hi-C of 28 AML
patients and 4 CD34+ HSPC donors (GSE152135; pairs files, since the
.iced.mcool start at 10 kb in steps of 2), CUT&Tag of 12 patients (GSE152132:
H3K27ac and H3K27me3 for all 12, CTCF for 9; only MACS2 -log10 p-value
bigWigs are public), and the supplementary tables of the paper (Nature
MOESM3: AML-specific loops, stripes, SVs from WGS plus hic_breakfinder,
neo-loop hijacking; as CSV in `xu_tables/`). The 3 PBMC controls of the
paper have no public Hi-C. Fischer et al., Cell Rep 2024, GSE268035:
per-patient CTCF, RAD21 and H3K27ac ChIP (GSE268031, CNV-normalised
bigWigs; cohesin wild type 17 to 21 patients, RAD21-mutant 8 to 9,
STAG2-mutant 10 to 15) and pooled group Hi-C (GSE267780; named .hic, the
files are FAN-C HDF5 objects with 10 kb bins, converted to cooler). Every
map: 5 kb cooler from the pairs (cooler cload pairs), 10 and 25 kb by
coarsening, ICE (`--ignore-diags 2 --max-iters 500`), `scripts/oe_proper.py`.
Contacts per sample 0.11 to 0.41 B (`targets/*.json`); 2 kb maps were not
built.

Inputs. Each track is quantile-matched per mark to the pooled 200 bp
fold-change values of the 12 training cells (rank intervals of tied source
values get the mean of the training quantile function over the interval);
monotone, so the ordering of the source is kept. Control on K562 (held out
from every model), only CTCF, H3K27ac and H3K27me3 given, median Pearson at
25 / 10 / 5 kb (`k562_control.csv`): all12refres_drop with ENCODE fold
change 0.913 / 0.890, ENCODE fold change quantile-matched 0.921 / 0.895,
ENCODE p-value quantile-matched 0.921 / 0.895, K562 CUT&Tag (ChIP-Atlas
SRX5242662, SRX19742232, SRX5545339) quantile-matched 0.913 / 0.889, CUT&Tag
with the section 19 mean scaling 0.910 / 0.886; consensus12 0.897 / 0.859.

Models: all12refres_drop and all12honest_drop (section 22) and the dedicated
min_ctcf_h3k27ac_refres (section 20) at 25 and 10 kb, all12refres at 25, 10
and 5 kb and all12honest at 5 kb; reference consensus12 (no AML map in it).
Each reference model was also run with a leave-one-patient-out AML
consensus as reference (pixel-wise mean of the other 27 AML patients' maps,
HSPC excluded; its cell list `amlxu<patient>` lets the reference guard reject
a map containing the scored patient), without retraining. Scoring on the 413
Akita test regions, one pixel mask per patient and resolution shared by
every method.

Median over the 12 patients of the per-patient median Pearson (SCC in
`per_patient.csv`):

| method | 25 kb | 10 kb | 5 kb |
|---|---|---|---|
| all12refres_drop | 0.822 | 0.688 | |
| min_ctcf_h3k27ac_refres | 0.828 | 0.702 | |
| all12refres | 0.795 | 0.687 | 0.518 |
| all12honest_drop (5 kb: all12honest) | 0.530 | 0.414 | 0.196 |
| all12refres_drop, AML-consensus reference | 0.886 | 0.773 | |
| min_ctcf_h3k27ac_refres, AML-consensus reference | 0.882 | 0.770 | |
| all12refres, AML-consensus reference | 0.846 | 0.757 | 0.596 |
| AML consensus (other 27 patients) alone | 0.906 | 0.784 | 0.682 |
| consensus12 | 0.816 | 0.669 | 0.513 |
| HSPC consensus (4 donors) | 0.868 | 0.717 | 0.578 |
| best single HSPC map | 0.856 | 0.668 | 0.493 |
| best other AML patient (maximum of 27) | 0.900 | 0.722 | 0.549 |
| median other AML patient | 0.826 | 0.639 | 0.479 |
| best training-cell map | 0.840 | 0.701 | 0.495 |
| GM12878 / CD14+ monocytes / K562 | 0.794 / 0.786 / 0.751 | 0.665 / 0.656 / 0.595 | 0.474 / 0.473 / 0.430 |
| MOLM-13 / U-937 (section 19) | 0.792 / 0.775 | 0.650 / 0.636 | 0.488 / 0.462 |

With the AML consensus as reference the reference models exceed their
consensus12 version in 11 or 12 of 12 patients and consensus12 alone in 10
to 12 of 12 patients at every resolution, and stay
below the AML consensus alone in all 12 patients at every resolution
(0.886 against 0.906 at 25 kb). For MOLM-13 and U-937 (maps from other labs)
the all-patient AML consensus scores below consensus12 (0.876 / 0.852
against 0.909 / 0.866 at 25 kb), and the reference models drop with it
(all12refres_drop 0.874 / 0.865 against 0.907 / 0.892;
`lines_aml_reference.csv`).

Structural variants and copy number. SVs of Suppl. Table 5 for 10 of the 12
patients (none listed for 103 and 673); copy number from Hi-C coverage per
500 kb against the mean of the four HSPC maps (autosomes, segments of at
least 3 Mb with |log2 ratio| >= max(0.3, 4 robust SD)): 54 segments in 18
samples, among them trisomy 8 in 1916, monosomy 7 in 936 and del(5q) in 168
and 496, which share identical copy-number breakpoints
(`explain_cnv_segments.csv`). A test region is flagged when it lies within
1 Mb of an SV breakpoint, inside an intra-chromosomal SV or on a copy-number
segment: 514 of 13216 sample-regions. Without them every median above
changes by at most 0.015 (`explain_scores.csv`, `explain_lineage.csv`); on
the flagged regions alone the models and maps score within 0.06 of the
unflagged regions, in both directions.

Lineage and protocol (`explain_lineage.csv`, 25 / 10 / 5 kb, all regions):
HSPC maps agree with each other at 0.915 / 0.711 / 0.492, AML patients with
HSPC maps at 0.836 / 0.639 / 0.449 and AML patients with each other at
0.813 / 0.612 / 0.434. A single healthy HSPC map of the same study scores a
patient as well as the median other patient (0.82 to 0.86 against 0.826 at
25 kb), while blood maps from other laboratories (GM12878, CD14+
monocytes, K562, MOLM-13, U-937) score below consensus12. Averaging lifts
every group: HSPC consensus of 4 maps 0.868, AML consensus of 27 maps 0.906.

Patient-specific structure. Deviation score of the patient map T from
consensus12 C (section 24 score) per region, coverage and masked-bin
filters of the hard split (no replicate maps exist per patient), hardest
20% per patient (`specific_summary.csv`, 25 / 10 kb): the mean of the other
AML patients' maps A reproduces T - C at 0.818 / 0.681 (residual
correlation), the models with consensus12 at 0.43 to 0.52 / 0.31 to 0.47 and
with the AML reference at 0.76 to 0.79 / 0.65 to 0.68. Against A, the
patient's own deviation T - A is reproduced at 0.00 to 0.18 by every model.
Regions deviating from both C and the HSPC consensus H (top 20% of both
scores, SV/CNV excluded, 1589 patient-regions; `explain_learn.csv`): the AML
consensus reproduces T - H at 0.48 / 0.43, single HSPC maps at -0.02 to 0.02,
the reference models with consensus12 at 0.25 to 0.30 / 0.23 to 0.29 (the
CD14+ monocyte map 0.29 / 0.23), with the AML reference at 0.43 to 0.46 /
0.38 to 0.42. The mean absolute change of the patient's CUT&Tag tracks in
these regions against the training cells' tracks does not track the residual
correlation (Spearman between -0.16 and 0.16 per mark and model,
`explain_learn_regions.csv`).

Xu et al.'s features (10 kb, `explain_features.csv`; windows of 192 bins
centred on each feature, up to 150 loops and 150 stripes per patient, 10
patients with listed features; loops typed by the patient's CUT&Tag at the
anchors). AML-specific loops are stronger in the patient maps than in the
HSPC consensus by 0.39 (P-E-like, 176 loops), 0.45 (P-S-like, 156) and 0.40
(other, 1147) log2 donut enrichment, and stronger than in consensus12 by
0.29 to 0.31. all12refres_drop places 0.14 above the HSPC consensus (above
it at 78 to 83% of the loops; correlation with the measured gain 0.27 to
0.32) and 0.02 to 0.04 above consensus12 (correlation with the measured gain
over consensus12 -0.03 to 0.06); all12honest_drop and the minimal model show
correlations of 0.07 to 0.34 against consensus12. Stripes (1067): measured
+0.22 over the HSPC consensus, predicted +0.03 to +0.04 (correlation 0.39 to
0.43). Compartment switches are not assessable on the 1 to 2 Mb prediction
windows, and the CTCF sites lost through hypermethylation are listed only
for patient 424, who has no CUT&Tag.

SV-aware prediction (`sv_junctions.csv`, `sv_summary.csv`, 10 kb, 158
junctions of 10 patients): the derivative allele of each junction (pieces of
the patient's input rows joined in the called orientation, reversed pieces
with swapped motif strands, reference blocks per piece and the expected
level between pieces), predicted and mixed 0.5 / 0.5 with the normal allele
in contact space (the tracks cannot be phased and are used as measured on
both alleles). In the cis window around the breakpoint the SV-aware
prediction differs from the naive one by less than 0.01 in Pearson for
all12refres_drop (median change 0.000). In the junction block (rows of the
left piece, columns of the right piece; 84 junctions with measured
contacts) the predicted log O/E correlates with the measured block at -0.24
to 0.12 (median per SV type and model), below the distance-only junction model log E(d) (0.21
to 0.33); measured junction blocks lie 0.21 to 0.41 above the cis expected
of their derivative distance (deletions, inversions, translocations), the
predictions -0.10 to -0.12 below it. The neo-loops of Suppl. Table 6 that
fall inside a window (15 junctions, among them HSF4 and MYH11 at the inv(16)
of 629) are not enriched in the derivative predictions (log2 donut
enrichment -0.24 to 0.21, measured -0.47 to 0.28).

Cohesin-mutant groups (Fischer, 10 kb, `fischer_groups.csv`,
`fischer_change.csv`): each group map is predicted from the group's mean
quantile-matched tracks at 0.86 to 0.91 by all12refres_drop (consensus12
0.84 to 0.88; the other groups' measured maps 0.92 to 0.94). The predicted
mutant - wild-type change correlates with the measured change at 0.180
(RAD21-mutant) and 0.197 (STAG2-mutant) per region (all12refres_drop; the
two measured changes, which share the wild-type map, correlate at 0.50);
loop-strength changes at 0.07 / 0.19 and insulation changes at 0.22 / 0.19.

Figures: `per_patient.png`, `per_distance.png` (pooled per-distance Pearson),
`showcase_{25,10,5}kb.png` (hardest regions, 80 bins per panel),
`explain_features_loops.png`, `sv_junctions.png` (80 bins per panel).

### 29b. Methylation in place of ChIP for the patients

`scripts/aml_primary_meth.py` (stages build, predict, score, checks); tables
`meth_patients.csv`, `meth_summary.csv`, `meth_compartment.csv`,
`meth_features.csv` in `results/aml_primary/`. Xu et al. deposited WGBS of 20
AML patients (GSE152099; hg38, bismark and methylpy; processed bigWigs of the
CG methylation level per 10 bp bin, no read coverage; no raw reads; the 2
PBMC are not deposited). Ten patients have WGBS and CUT&Tag (103, 673, 629,
270, 773, 018, 546, 472, 168, 027; 673 without CTCF), ten have WGBS only
(566, 260, 1071, 1324, 424, 496, 504, 798, 838, 933), 1916 and 1360 have
CUT&Tag without WGBS. The channel follows `scripts/methylation.py` (1 kb
bins, 1 + methylated fraction, thin bins at 1 + the patient's genome mean,
raw zero = no data); since the files carry no coverage, a bin is the
unweighted mean of its covered 10 bp bins and needs 3 of them (the ENCODE
build weights by reads and needs 5). Genome means 0.69 to 0.86 (ENCODE cells
0.22 to 0.85), 78% to 79% of the 1 kb bins with data. Model: the refreshed
all12refres_drop_meth_10kb (retrained after the select_tracks fix; no later
version existed), keep sets of the patient's tracks, TTA, 413 regions, two
references: consensus12 and the patient's leave-one-patient-out AML consensus
(whose mean over the other patients is the other-patient mean, identical
scores).

Median Pearson over patients (SCC in `meth_patients.csv`), 10 kb, patients
with both data types (10):

| input | consensus12 reference | AML-consensus reference |
|---|---|---|
| CTCF, H3K27ac, H3K27me3 (the patient's own marks) | 0.690 | 0.769 |
| the same plus Meth | 0.696 | 0.768 |
| CTCF, H3K27ac, Meth (Meth for H3K27me3) | 0.701 | 0.771 |
| H3K27ac, H3K27me3, Meth (Meth for CTCF) | 0.716 | 0.771 |
| H3K27ac, Meth | 0.717 | 0.757 |
| Meth only | 0.660 | 0.760 |
| all12refres_drop_10kb without the channel, same marks | 0.687 | 0.771 |
| reference alone (consensus12 / AML consensus) | 0.669 | 0.784 |

Meth added to the three marks raises the score in 8 of 10 patients with
consensus12 (median +0.004) and in 3 of 10 with the AML reference (median
-0.005). Meth in place of H3K27me3 or CTCF costs nothing (+0.006 and +0.008
over the three marks, and CTCF-less patient 673 0.569 against 0.561); Meth
alone scores below consensus12 (0.660 against 0.669) and, with the AML
reference, below the reference alone (0.760 against 0.784). Patients with
WGBS only (10; no marks, 14 tracks absent): with consensus12 the model with
no input beyond sequence and motifs scores 0.679 and with Meth 0.690
(consensus12 0.688; higher with Meth in 10 of 10 patients, median +0.005, at most +0.016),
with the AML reference 0.788 without and 0.782 with Meth (AML consensus
0.796). Methylation therefore replaces H3K27me3 or CTCF in the patient
set to within the seed noise of the other inputs and gives a small gain over
a bare sequence model, and no gain over a disease-matched reference. On the
ENCODE evaluation of section 27 the same channel raises GM12878 from 0.898 to
0.922 with the three marks (a training cell with WGBS) and K562 from 0.897 to
0.898.

AML-specific checks. Local compartment signal (first eigenvector of the
correlation matrix of each 1 to 2 Mb region, sign fixed on consensus12; the
paper's switch calls hold no coordinates and chromosome-wide A/B compartments
are not accessible from the test windows; 82 candidate regions per patient,
the lowest-agreement 20% against the HSPC consensus): the eigenvector change
from HSPC is reproduced at 0.60 (consensus12 reference) and 0.60 (AML
reference) without methylation and at 0.58 and 0.60 with it
(`meth_compartment.csv`). Xu et al.'s AML-specific loops and stripes
(`meth_features.csv`, 8 patients with CUT&Tag and WGBS and features,
10 kb): the correlation of the predicted gain over the HSPC consensus with
the measured gain is 0.35 / 0.31 / 0.27 (P-E-like, P-S-like, other loops)
and 0.39 (stripes) without methylation, 0.41 / 0.31 / 0.27 and 0.40 with it;
over consensus12 0.14 / 0.10 / 0.05 and 0.27 without, 0.30 / 0.09 / 0.10 and
0.22 with it. The paper's DNA-methylation-linked compartment changes and the
CTCF sites lost to hypermethylation are therefore not recovered through the
channel.

## 30. Structure-aware learning: TAD, loop and compartment terms (PENDING)

Base: all12refin_drop (input reference, track dropout), the best model on
new cells at 10 kb. Added to its L1+MSE loss (hicmultiome/losses.py), one at
a time: insul (half L1, half 1 - correlation of the insulation profile,
10-bin square; TADs), donut (L1 of the pixel minus its 2-to-5-bin ring;
loops) and compart (1 - Pearson between the row-correlation matrices of
prediction and target, pixels at least 2 bins off the diagonal). Windows
are 1.9 to 2 Mb, so compart sees the compartment checkerboard inside 2 Mb
only; it is a within-window proxy for A/B compartments, which span tens of
Mb, and it matches the whole correlation matrix rather than its first
eigenvector to avoid the sign ambiguity and the eigen-decomposition.
Every term is logged per epoch.

Weights: grid 0.1, 0.3 and 1.0 per term at 25 kb; per term the weight
whose selected checkpoint has the highest combined validation score on the
honest validation windows (mean over Pearson, stratified correlation,
insulation correlation and APA agreement of the rank among the three grid
runs) is used at 10 kb and in struct3 (all three terms, 25 and 10 kb)
(scripts/structure_loss.py --stage choose, `results/structure_loss/
weight_grid_val.csv`, `chosen_weights.json`).

Checkpoint selection (train.select_metric): besides the median validation
Pearson, "combined" keeps the checkpoint whose mean over Pearson, stratified
correlation, insulation correlation and APA agreement min(r, 1/r) of the
rank among all evaluations of the run so far is the highest (run
all12refin_drop_selcomb, 25 and 10 kb, L1+MSE); "hic_measures" uses
scripts/hic_measures.py combined_score, the window-level proxy of the
multi-measure evaluation (pixel, TAD, loop and within-window compartment
levels, each the mean of its terms, averaged over the levels; run
all12refin_drop_selhm, 25 and 10 kb).

Evaluation: the standard tables (crosscell eval of every run) and
crosscell_predict --structure on K562, HMEC deep, HUVEC deep (held out) and
GM12878, IMR-90, HepG2 (test regions) with the consensus reference scored
alike: Pearson, stratified correlation, insulation correlation, boundary F1,
loop enrichment correlation, APA ratio on the target's top 1 % enrichment
pixels at least 50 kb off the diagonal, correlation-matrix agreement and
first-eigenvector agreement (hicmultiome/structure.py;
`results/structure_loss/structure_eval.csv`).

## 31. Multi-measure evaluation (FINAL)

Question: does the model ranking by the median per-region Pearson hold when
A/B compartments, TADs and loops are scored, and with the combined measures
of the hyperparameter-scoring work (J. Wolff, "A novel metric for
hyperparameter optimization in Hi-C matrix prediction", Zenodo
10.5281/zenodo.17296001), whose trained regressors imitate an expert ranking
from ENT3C, HiCRep, GenomeDISCO, HiC-Spector, QuASAR-Rep, Pearson AUC, TAD
score MSE, TAD fraction and TAD fraction exact match (FEM)?

Script `scripts/hic_measures.py` (stages predict, expected, score, ceilings,
combine, report, tables; tests `tests/test_hic_measures.py`), outputs
`runs/hic_measures/`, tables `results/hic_measures/`.

Whole-chromosome maps. `hicmultiome.predict` tiles each model's windows over
whole chromosomes (half-window stride, Hann-weighted averaging, test-time
reverse averaging; 360 maps, 12 models x 15 cells x 2 resolutions). Every map
reaches only as far as the model window: 2.0 Mb at 25 kb, 1.92 Mb at 10 kb
(the proper O/E targets are banded to 2 Mb as well). Evaluation sets:

- heldout: K562, HMEC deep and HUVEC deep, cells no model was trained on;
  whole chr2, chr12 and chr18. The loci lie in training windows of other
  cells, as the consensus reference does.
- trainstretch: the 12 training cells, chr2 and chr18, pixels restricted to
  the Akita valid+test stretches (`data.holdout_bed`; no training window of
  any cell overlaps them, registry section 15; 23% of chr2, 27% of chr18,
  stretches of 1 to 11 Mb): the clean evaluation of the main models on
  training cells.
- trainchrom: the 12 training cells, whole chr2 and chr18. Clean entry: the
  cascade fold models (`cascfold_odd/even_<k>kb`, the all12refres setup
  trained on the odd or on the even chromosomes; their out-of-fold maps are
  `crosscell_oe/cascade/prior_<cell>_<k>kb.cool`; chr2 and chr18 come from the
  model trained on the odd chromosomes). all12refres, which trained on
  windows covering about 75% of these chromosomes, is listed as its
  in-training counterpart.
- hicreg and epiphany: GM12878 with the band cut to the competitor's reach
  (1 Mb on chr2 and chr18 at 25 kb; 990 kb on chr17 at 10 kb); our models
  trained on these chromosomes there.

Sources per block: every model, the consensus the model reads (leave-one-out
for training cells, consensus12 for held-out cells), the best single other
measured map (section 15), an independent replicate map where one exists
(intact GM12878 ENCSR965OVH r75, intact K562 ENCSR676BEP r5, single-lane in
situ IMR-90, HMEC and HUVEC rep1; for HMEC and HUVEC the independence of the
single-lane map from the ENCODE deep map is not established), and the
competitor predictions of section 18 (reused, none rerun): Hi-cGAN
(GM12878, even chromosomes held out; its GM12878 model on the held-out
cells), C.Origami (IMR-90, chr2 = its test chromosome, 10 kb), HiC-Reg
(within-chromosome CV maps), Epiphany (chr17 held out). Akita exists only as
1 Mb Akita-space predictions and has no whole-chromosome map. Every measure of
a block uses the pixels finite in the target and in every source of the block.
The single-lane replicate maps at 25 kb are stored on a 24 kb grid
(`*_rep1_ice_25kb.cool`, bin size 24000); they are sampled bilinearly at the
25 kb bin centres (`resample_band`). `multicell_expanded.target("hmec" |
"huvec", 25000)` points to these 24 kb files as well.

Measures. Count space, with the tools and settings of the hyperparameter
work: ENT3C (`ENT3C run_all` per pair, SUB_M_SIZE_FIX 500, phi 1, PHI_MAX
1e3, no weights), HiCRep (python hicrep, h 5, dBPMax 1 Mb, no
down-sampling), GenomeDISCO, HiC-Spector and QuASAR-Rep
(3DChromatin_ReplicateQC with that work's parameter file), Pearson AUC
(Hi-cGAN computePearsonCorrelation reimplemented, identical to the pandas
original in the unit test), TAD score MSE, TAD fraction and TAD FEM
(hicFindTADs 3.7.6, depth 3 to 10 bins, step 1 bin, minimum boundary
distance 100 kb, FDR, threshold 0.5). Each map, the target included, is
converted back to counts with the target's own balanced expected and
per-diagonal offset (exactly the curves of `oe_proper.py`) and scaled to the
raw contact total of the chromosome band; the measured map itself (balanced
counts, same scale) scored against the converted target is the cost of the
round trip through the clipped, smoothed target. ENT3C and the 3DChromatin
measures take 1 to 5 min per map and chromosome (HiC-Spector is the slow
part) and run on every held-out-cell block at 25 kb, on chr18 for the other
25 kb blocks and on chr18 of the held-out cells at 10 kb (`slow_tools`).
Structure, on log O/E: Pearson over all band pixels and its median over 1 Mb
tiles, per-distance Pearson and its AUC, SCC, insulation-profile Pearson
(diamond 75 kb at 25 kb, 50 kb at 10 kb), boundary recall and precision
(hicFindTADs boundaries of the prediction against those of the target,
1 bin), loops against the hicDetectLoops 3.7.6 calls of the measured 10 kb
map (rank-matched donut recall, APA ratio and anchor AUROC as section 18; at
10 kb outside trainstretch also hicDetectLoops 3.7.6 run on the converted
prediction: recall, precision, F1, 1 bin tolerance), compartments (see
below), the window-level combined score `combined_score` (next paragraph) on
the chromosome cut into model windows, and the rank score: per block the
rank of each model among the models (0 worst, 1 best) per measure, averaged
within four levels (maps: tile Pearson, SCC, HiCRep, GenomeDISCO,
HiC-Spector, QuASAR-Rep, ENT3C, Pearson AUC; TADs: insulation r, boundary
recall and precision, TAD score MSE, |TAD fraction - 1|, TAD FEM; loops:
loop recall, APA agreement min(r, 1/r), anchor AUROC, called-loop F1;
compartments: compartment preference r, A/B sign agreement, saddle agreement
min(r, 1/r), local E1 r) and then over the levels.

Compartments on whole chromosomes. A chromosome-wide E1 needs contacts over
tens of Mb; the maps hold 2 Mb. Applied to the band of the measured target,
the leading eigenvectors of the banded matrix localize (correlation with the
genome-wide 100 kb E1 of the full measured map, `runs/cellspecific/e1`,
0.07 to 0.35 for K562, GM12878 and HMEC), and an E1 of 2 Mb windows phased
by GC and averaged ("local E1") correlates with it at 0.26 to 0.32 (target,
median per set). The measures therefore use the measured E1 as the
compartment label of each bin and ask whether a map reproduces the
compartment-specific contacts within 2 Mb: compartment preference per bin
(mean log O/E with A partners minus with B partners, partners 200 kb to 2 Mb
away), its correlation with the target's preference at 100 kb (compartment
preference r), the share of 100 kb bins whose preference has the sign of the
measured E1 (A/B sign agreement; target 0.80 to 0.85), and the saddle
strength (AA + BB) / 2 AB over the top and bottom 20% of the E1 within the
band (saddle ratio = map / target; target strength 3.0 to 3.8). The local E1
of a map against the target's local E1 is reported as well (local E1 r).

Window-level combined score for checkpoint selection
(`train.select_metric: hic_measures`, section 30). `combined_score(maps,
res)` takes the validation windows (prediction, target) and returns the
mean of four levels, each the mean of its terms: pixels (median window
Pearson, median stratified correlation, Pearson AUC of the per-distance
Pearson pooled over windows), TADs (median insulation correlation, median
boundary F1, agreement min(r, 1/r) of the pooled boundary counts), loops
(median donut-enrichment correlation, median APA agreement) and
compartments (median correlation of the row-correlation matrices, median
|r| of their first eigenvectors); hicmultiome.structure supplies the window
measures. It costs 0.011 s per 80-bin and 0.020 s per 192-bin window (13
and 25 s for 1200 validation windows). The nine count-space measures and
their regressors are not used there: they need whole chromosomes and
external tools.

Regressors and their training domain (stage combine; joblib pipelines of
`Revision_1/regression_model_results_revision_no_gpu`, env hyperparameter_gpu):
TabPFN with 8 measures (lowest error of the original selection) and with
the 6 measures the nested cross-validation chose most often, gradient
boosting with 7 and 4 measures, random forest with 4 measures. On their own
361 training rows they reproduce the ELO ranking at r = 0.85 to 0.94. The
training rows are Hi-cGAN hyperparameter-search predictions of GM12878 chr2
at 25 kb in count space. Our maps lie outside that domain on seven of the
nine measures: Pearson AUC 0.49 to 1.00 (training maximum 0.41), HiCRep
median 0.92 (maximum 0.52), GenomeDISCO 0.76 to 0.98 (maximum 0.68),
QuASAR-Rep 0.89 to 1.00 (maximum 0.79), TAD score MSE median 0.12 (minimum
0.77) and TAD FEM median 0.55 (maximum 0.14); only TAD fraction (100% of
the scored maps), ENT3C (80%) and HiC-Spector (71%) fall inside the
training range, and no scored map has all inputs of any pipeline in range
(`regressor_domain.csv`). The regressor scores are reported for completeness
and are not valid for these maps; the 10 kb rows are additionally outside
the training resolution.

Results, held-out cells (median over K562, HMEC deep and HUVEC deep of the
per-cell mean over chr2, chr12, chr18) and the clean training-cell sets;
full tables below and in `summary.csv`, `measures_per_cell.csv`,
`measures_per_chromosome.csv`, `combined_scores.csv`, heat maps
`heatmap_<set>_<k>kb.png`.

At 25 kb on the held-out cells the dropout models lead every structure
level: all12refin_drop has the highest 413-region Pearson (0.924) and tile
Pearson (0.939), all12refin_drop_ctcfac the highest compartment preference
correlation (0.939), boundary recall is 0.78 to 0.82 for every reference
model, and all12refres_drop_varw has the highest rank score (0.708, then
all12refin_drop 0.679 and all12refres_refdrop 0.658). The models without a
reference trail on every level (rank score 0.11). The consensus alone is level
with the models on insulation (0.846, highest of all non-replicate sources)
and below them on compartment preference (0.885 against 0.90 to 0.94) and
SCC; the Hi-cGAN GM12878 model reaches 0.39 tile Pearson and 0.25 boundary
recall. At 10 kb all12deg_refres_drop has the highest 413-region Pearson
(0.908), tile Pearson, window score and rank score (0.760), followed by
all12refres_drop_varw (0.715). Loop recall (rank-matched) is 0.29 to 0.34 for
every reference model against 0.30 to 0.41 for the target's own donut peaks
(`loop_recall_ceiling.csv`); hicDetectLoops on the reference-model predictions recovers
0.31 to 0.38 of the measured loops at precision 0.30 to 0.35 (F1 0.32 to
0.35), the consensus 0.35 at 0.29 (F1 0.33), the replicate map 0.36 at 0.13
(F1 0.19), and the measured map itself in count space 0.69 at 0.86 (the
caller's own agreement after the round trip).

On the clean training-cell stretches all12multires leads at both
resolutions (413-region Pearson 0.944 and 0.938, rank score 0.806 and
0.841), then all12refin and all12refres. The chromosome-held-out cascade
fold model scores 0.948 against 0.953 tile Pearson for all12refres on whole
chr2 and chr18 at 25 kb (0.919 against 0.929 at 10 kb), with equal boundary
recall (0.816 and 0.815) and loop recall (0.333 and 0.339): the in-training
advantage on whole chromosomes is 0.005 to 0.010. HiC-Reg's within-chromosome
CV maps reach a tile Pearson of 0.946 in the 1 Mb band (all12multires 0.951)
but an insulation correlation of 0.562 against 0.86 to 0.93 for the
reference models; C.Origami (IMR-90 chr2, 10 kb) reaches 0.904 tile Pearson
and 0.466 boundary recall against 0.633 for the cascade fold model; Epiphany
0.511 and 0.226.

Model ranking: Kendall tau of each measure's model ranking (median over
cells) with the ranking by the 413-region median Pearson
(`ranking.csv`, figure `ranking_agreement.png`; models only, 12 per set):

| set | best by median Pearson | tile Pearson | window combined | rank score | TabPFN 8m | TabPFN 6m | GB 7m | GB 4m | RF 4m | lowest-agreement measures |
|---|---|---|---|---|---|---|---|---|---|---|
| held-out, 25 kb | all12refin_drop | 0.79 (all12refin_drop) | 0.15 (all12refres_drop) | 0.76 (all12refres_drop_varw) | -0.42 (all12honest_drop) | -0.48 (all12honest) | -0.27 (all12multires) | -0.26 (all12refin) | -0.18 (all12multires) | ENT3C -0.12, TAD fraction -0.12, HiC-Spector -0.03, APA ratio 0.15, anchor AUROC 0.15 |
| held-out, 10 kb | all12deg_refres_drop | 0.82 (all12deg_refres_drop) | 0.82 (all12deg_refres_drop) | 0.79 (all12deg_refres_drop) | -0.45 (all12honest) | -0.12 | -0.03 | 0.11 | -0.03 | HiC-Spector 0.03, ENT3C 0.21, anchor AUROC 0.21, TAD fraction 0.30, loop recall 0.30 |
| training stretches, 25 kb | all12multires | 0.82 (all12multires) | 0.79 (all12multires) | 0.91 (all12multires) | -0.70 (all12honest_drop) | -0.73 (all12honest_drop) | 0.08 | -0.15 | 0.50 | A/B sign agreement 0.36, saddle ratio 0.36, boundary recall 0.39, anchor AUROC 0.39, TAD fraction 0.42, ENT3C 0.42 |
| training stretches, 10 kb | all12multires | 0.76 (all12refres) | 0.61 (all12multires) | 0.88 (all12multires) | | | | | | TAD fraction 0.27, APA ratio 0.36, saddle ratio 0.39, loop recall 0.55 |

Ranking by the multi-measure rank score agrees with ranking by median
Pearson (tau 0.76 to 0.91) and picks the same best model in three of four
cases; on held-out cells at 25 kb it puts all12refres_drop_varw (0.708)
ahead of all12refin_drop (0.679), which leads on median Pearson (0.924
against 0.922) and on the compartment and map levels, because
all12refres_drop_varw ranks higher on the TAD and loop levels. The measures
that order the models differently from Pearson are the loop measures (APA
ratio, anchor AUROC, rank-matched recall: tau 0.15 to 0.55), TAD fraction,
ENT3C and HiC-Spector (tau -0.12 to 0.45) and the compartment sign and
saddle measures on training cells (0.36); the compartment preference
correlation, SCC, HiCRep, GenomeDISCO, QuASAR-Rep, Pearson AUC and TAD score
MSE follow the Pearson ranking (tau 0.6 to 0.9). The trained regressors
reverse it: both TabPFN pipelines rank a model without a reference first on
every set (tau -0.42 to -0.73) and score the GM12878 Hi-cGAN model on the
held-out cells (3.7) above every one of our models (1.6 to 2.3) and the
measured map itself lowest (0.8), and the tree pipelines assign tied values
to up to 5 of 12 models (tau -0.27 to 0.50). This follows from the domain
check: these maps are better than every map the experts ranked on seven
measures, so the regressors extrapolate; TabPFN's flattening near the
optimum and the trees' constant leaves beyond their last split (both
described in that manuscript) decide the order.

Loop and compartment ceilings (median target value per set): rank-matched
loop recall of the target's own donut peaks 0.41 / 0.30 (held-out 25 / 10
kb) and 0.40 / 0.31 (training stretches); A/B sign agreement of the
target's compartment preference 0.81 / 0.80 and 0.85 / 0.85; correlation of
the target's preference with the measured E1 0.74 / 0.71 and 0.78 / 0.78,
against 0.62 to 0.76 for all12refres and all12refin_drop and 0.53 to 0.63
for the consensus. Local E1 of the 2 Mb band against the genome-wide E1: 0.26 to
0.32 for the target and every map alike.

Full tables (`runs/hic_measures/registry_tables.md`, generated by `--stage
tables`): median over cells of the per-cell mean over chromosomes. Pearson
413 regions = section 15/22/26 crosscell tables (the registry metric);
rank score among the models of the block; regressor columns are outside
their training domain (see above).

held-out cells K562, HMEC deep, HUVEC deep; chr2, chr12, chr18, 25 kb, log O/E structure measures (median over 3 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined | rank score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres | 0.906 | 0.919 | 0.878 | 0.809 | 0.796 | 0.780 | 0.287 | 0.767 | 0.912 | 0.745 | 0.827 | 0.843 | 0.900 | 0.406 |
| all12honest | 0.771 | 0.794 | 0.752 | 0.656 | 0.603 | 0.634 | 0.181 | 0.384 | 0.836 | 0.762 | 0.713 | 0.741 | 0.737 | 0.112 |
| all12refin | 0.901 | 0.916 | 0.871 | 0.779 | 0.783 | 0.778 | 0.301 | 0.801 | 0.902 | 0.734 | 0.694 | 0.832 | 0.890 | 0.357 |
| all12refres_drop | 0.915 | 0.933 | 0.892 | 0.787 | 0.791 | 0.808 | 0.291 | 0.798 | 0.928 | 0.778 | 0.926 | 0.846 | 0.908 | 0.602 |
| all12honest_drop | 0.799 | 0.826 | 0.767 | 0.679 | 0.614 | 0.641 | 0.157 | 0.261 | 0.866 | 0.784 | 0.656 | 0.749 | 0.734 | 0.114 |
| all12refres_refdrop | 0.919 | 0.934 | 0.891 | 0.832 | 0.816 | 0.817 | 0.293 | 0.786 | 0.930 | 0.771 | 0.921 | 0.849 | 0.907 | 0.658 |
| all12refres_drop_hard | 0.916 | 0.931 | 0.889 | 0.797 | 0.785 | 0.806 | 0.293 | 0.773 | 0.927 | 0.775 | 0.826 | 0.846 | 0.907 | 0.500 |
| all12refres_drop_varw | 0.922 | 0.932 | 0.897 | 0.838 | 0.791 | 0.803 | 0.288 | 0.817 | 0.929 | 0.776 | 0.841 | 0.848 | 0.906 | 0.708 |
| all12refin_drop | 0.924 | 0.939 | 0.897 | 0.840 | 0.795 | 0.826 | 0.310 | 0.744 | 0.928 | 0.769 | 0.772 | 0.865 | 0.900 | 0.679 |
| all12refin_drop_ctcfac | 0.920 | 0.931 | 0.902 | 0.843 | 0.806 | 0.830 | 0.297 | 0.834 | 0.939 | 0.763 | 0.705 | 0.859 | 0.906 | 0.646 |
| all12deg_refres_drop | 0.922 | 0.938 | 0.892 | 0.826 | 0.795 | 0.825 | 0.299 | 0.775 | 0.923 | 0.776 | 0.842 | 0.855 | 0.901 | 0.608 |
| all12multires | 0.902 | 0.919 | 0.880 | 0.832 | 0.793 | 0.788 | 0.292 | 0.827 | 0.911 | 0.757 | 0.760 | 0.833 | 0.907 | 0.501 |
| consensus alone | 0.916 | 0.917 | 0.874 | 0.846 | 0.775 | 0.784 | 0.289 | 0.825 | 0.885 | 0.724 | 0.736 | 0.798 | 0.871 |  |
| best other measured map | 0.882 | 0.893 | 0.832 | 0.798 | 0.733 | 0.769 | 0.269 | 0.805 | 0.845 | 0.702 | 0.862 | 0.790 | 0.855 |  |
| replicate map |  | 0.961 | 0.927 | 0.916 | 0.827 | 0.803 | 0.275 | 1.121 | 0.969 | 0.774 | 0.943 | 0.926 | 0.937 |  |
| measured map, balanced counts |  |  |  |  | 0.957 | 0.869 |  |  |  |  |  |  |  |  |
| Hi-cGAN, GM12878 model |  | 0.392 | 0.184 | 0.276 | 0.249 | 0.338 | 0.049 | 0.153 | 0.603 | 0.684 | 0.686 | 0.593 | 0.389 |  |

held-out cells K562, HMEC deep, HUVEC deep; chr2, chr12, chr18, 25 kb, count-space measures and regressor scores (median over 3 cells of the per-cell mean over chromosomes):

| source | ENT3C | HiCRep | GenomeDISCO | HiC-Spector | QuASAR-Rep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM | TabPFN 8m | TabPFN 6m | GB 7m | GB 4m | RF 4m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres | 0.444 | 0.864 | 0.916 | 0.554 | 0.983 | 0.951 | 0.155 | 1.021 | 0.481 | 1.64 | -1.64 | 9.53 | 0.32 | 8.79 |
| all12honest | 0.567 | 0.736 | 0.869 | 0.634 | 0.966 | 0.893 | 0.278 | 0.979 | 0.291 | 2.24 | -0.97 | 8.74 | 0.04 | 7.85 |
| all12refin | 0.600 | 0.857 | 0.914 | 0.627 | 0.983 | 0.949 | 0.190 | 1.007 | 0.473 | 1.74 | -1.31 | 9.57 | 0.62 | 8.78 |
| all12refres_drop | 0.495 | 0.864 | 0.927 | 0.493 | 0.985 | 0.955 | 0.183 | 0.978 | 0.494 | 1.71 | -1.94 | 7.83 | 0.24 | 7.80 |
| all12honest_drop | 0.602 | 0.750 | 0.873 | 0.651 | 0.965 | 0.902 | 0.271 | 0.931 | 0.320 | 2.28 | -1.26 | 7.49 | -0.70 | 7.14 |
| all12refres_refdrop | 0.446 | 0.878 | 0.922 | 0.487 | 0.986 | 0.952 | 0.143 | 0.998 | 0.503 | 1.62 | -2.04 | 8.32 | 0.32 | 8.78 |
| all12refres_drop_hard | 0.508 | 0.870 | 0.925 | 0.517 | 0.986 | 0.956 | 0.172 | 0.974 | 0.488 | 1.72 | -1.94 | 8.61 | 0.46 | 8.48 |
| all12refres_drop_varw | 0.564 | 0.885 | 0.928 | 0.671 | 0.987 | 0.958 | 0.132 | 0.986 | 0.486 | 1.57 | -1.78 | 8.70 | 0.31 | 8.76 |
| all12refin_drop | 0.544 | 0.886 | 0.927 | 0.703 | 0.987 | 0.960 | 0.139 | 0.962 | 0.489 | 1.63 | -1.75 | 7.87 | -0.12 | 7.51 |
| all12refin_drop_ctcfac | 0.547 | 0.907 | 0.921 | 0.660 | 0.985 | 0.955 | 0.127 | 0.985 | 0.510 | 1.63 | -1.80 | 8.42 | 0.10 | 7.86 |
| all12deg_refres_drop | 0.498 | 0.875 | 0.928 | 0.511 | 0.986 | 0.960 | 0.142 | 0.964 | 0.486 | 1.70 | -2.30 | 7.95 | -0.11 | 7.56 |
| all12multires | 0.484 | 0.889 | 0.921 | 0.625 | 0.984 | 0.958 | 0.135 | 1.007 | 0.485 | 1.57 | -1.86 | 9.73 | 0.43 | 8.81 |
| consensus alone | 0.477 | 0.879 | 0.902 | 0.472 | 0.979 | 0.940 | 0.158 | 1.011 | 0.467 | 1.69 | -2.01 | 8.64 | 0.21 | 7.86 |
| best other measured map | 0.529 | 0.846 | 0.888 | 0.644 | 0.971 | 0.917 | 0.191 | 0.965 | 0.452 | 1.87 | -1.70 | 7.86 | 0.13 | 7.12 |
| replicate map | 0.694 | 0.960 | 0.943 | 0.771 | 0.991 | 0.968 | 0.075 | 1.030 | 0.614 | 1.30 | -1.95 | 9.42 | -0.31 | 8.83 |
| measured map, balanced counts | 0.564 | 0.988 | 0.960 | 0.632 | 0.992 | 0.945 | 0.026 | 1.103 | 0.663 | 0.77 | -2.45 | 6.32 | -0.56 | 8.52 |
| Hi-cGAN, GM12878 model | 0.128 | 0.275 | 0.821 | 0.448 | 0.920 | 0.594 | 0.821 | 0.739 | 0.080 | 3.69 | -0.38 | 3.99 | -4.15 | 5.15 |

held-out cells K562, HMEC deep, HUVEC deep; chr2, chr12, chr18, 10 kb, log O/E structure measures (median over 3 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | loop F1 (called) | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined | rank score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres | 0.885 | 0.888 | 0.850 | 0.857 | 0.627 | 0.634 | 0.335 | 0.764 | 0.319 | 0.915 | 0.750 | 0.795 | 0.803 | 0.810 | 0.468 |
| all12honest | 0.750 | 0.745 | 0.707 | 0.668 | 0.442 | 0.496 | 0.168 | 0.348 | 0.152 | 0.820 | 0.757 | 0.816 | 0.731 | 0.658 | 0.089 |
| all12refin | 0.895 | 0.900 | 0.862 | 0.835 | 0.603 | 0.615 | 0.334 | 0.694 | 0.340 | 0.930 | 0.765 | 0.743 | 0.840 | 0.823 | 0.479 |
| all12refres_drop | 0.905 | 0.907 | 0.873 | 0.833 | 0.639 | 0.643 | 0.334 | 0.838 | 0.344 | 0.934 | 0.790 | 0.880 | 0.826 | 0.839 | 0.686 |
| all12honest_drop | 0.765 | 0.772 | 0.729 | 0.672 | 0.465 | 0.479 | 0.167 | 0.225 | 0.118 | 0.866 | 0.763 | 0.775 | 0.720 | 0.667 | 0.095 |
| all12refres_refdrop | 0.900 | 0.905 | 0.865 | 0.805 | 0.625 | 0.636 | 0.331 | 0.685 | 0.326 | 0.927 | 0.792 | 0.902 | 0.835 | 0.819 | 0.489 |
| all12refres_drop_hard | 0.894 | 0.901 | 0.863 | 0.796 | 0.611 | 0.617 | 0.333 | 0.737 | 0.330 | 0.929 | 0.784 | 0.792 | 0.822 | 0.825 | 0.463 |
| all12refres_drop_varw | 0.899 | 0.910 | 0.875 | 0.839 | 0.646 | 0.652 | 0.336 | 0.831 | 0.348 | 0.938 | 0.789 | 0.878 | 0.835 | 0.832 | 0.715 |
| all12refin_drop | 0.905 | 0.911 | 0.875 | 0.835 | 0.639 | 0.649 | 0.327 | 0.660 | 0.330 | 0.937 | 0.792 | 0.916 | 0.841 | 0.838 | 0.689 |
| all12refin_drop_ctcfac | 0.903 | 0.910 | 0.876 | 0.825 | 0.638 | 0.648 | 0.329 | 0.855 | 0.332 | 0.946 | 0.780 | 0.794 | 0.830 | 0.843 | 0.535 |
| all12deg_refres_drop | 0.908 | 0.917 | 0.879 | 0.850 | 0.654 | 0.669 | 0.335 | 0.809 | 0.339 | 0.933 | 0.791 | 0.894 | 0.838 | 0.844 | 0.760 |
| all12multires | 0.888 | 0.890 | 0.853 | 0.825 | 0.624 | 0.625 | 0.326 | 0.779 | 0.336 | 0.919 | 0.771 | 0.803 | 0.804 | 0.818 | 0.415 |
| consensus alone | 0.884 | 0.878 | 0.842 | 0.827 | 0.634 | 0.638 | 0.331 | 0.801 | 0.330 | 0.897 | 0.734 | 0.782 | 0.800 | 0.792 |  |
| best other measured map | 0.835 | 0.836 | 0.794 | 0.778 | 0.571 | 0.565 | 0.209 | 0.655 | 0.235 | 0.858 | 0.718 | 0.976 | 0.758 | 0.748 |  |
| replicate map |  | 0.886 | 0.848 | 0.830 | 0.609 | 0.601 | 0.025 | 0.961 | 0.185 | 0.954 | 0.789 | 0.705 | 0.903 | 0.852 |  |
| measured map, balanced counts |  |  |  |  | 0.848 | 0.821 |  |  | 0.768 |  |  |  |  |  |  |
| Hi-cGAN, GM12878 model |  | 0.385 | 0.266 | 0.357 | 0.109 | 0.146 | 0.019 | 0.106 | 0.027 | 0.602 | 0.611 | 0.570 | 0.526 | 0.326 |  |

held-out cells K562, HMEC deep, HUVEC deep; chr2, chr12, chr18, 10 kb, count-space measures and regressor scores (median over 3 cells of the per-cell mean over chromosomes):

| source | ENT3C | HiCRep | GenomeDISCO | HiC-Spector | QuASAR-Rep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM | TabPFN 8m | TabPFN 6m | GB 7m | GB 4m | RF 4m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres | 0.765 | 0.879 | 0.915 | 0.762 | 0.925 | 0.940 | 0.116 | 0.974 | 0.330 | 2.21 | -1.81 | 8.61 | -0.23 | 8.68 |
| all12honest | 0.698 | 0.737 | 0.862 | 0.575 | 0.845 | 0.873 | 0.244 | 0.924 | 0.210 | 4.18 | -1.94 | 7.24 | -0.90 | 6.22 |
| all12refin | 0.805 | 0.880 | 0.919 | 0.710 | 0.929 | 0.943 | 0.128 | 0.958 | 0.313 | 2.44 | -1.86 | 8.40 | -0.59 | 7.79 |
| all12refres_drop | 0.734 | 0.892 | 0.919 | 0.688 | 0.938 | 0.945 | 0.124 | 0.992 | 0.335 | 2.28 | -1.94 | 8.61 | 0.14 | 7.85 |
| all12honest_drop | 0.716 | 0.732 | 0.877 | 0.519 | 0.876 | 0.878 | 0.243 | 0.959 | 0.202 | 4.06 | -1.82 | 7.24 | 0.16 | 6.08 |
| all12refres_refdrop | 0.743 | 0.877 | 0.918 | 0.720 | 0.935 | 0.942 | 0.164 | 0.981 | 0.318 | 2.15 | -1.75 | 7.29 | -0.59 | 8.74 |
| all12refres_drop_hard | 0.764 | 0.872 | 0.919 | 0.725 | 0.935 | 0.941 | 0.157 | 0.992 | 0.321 | 2.15 | -1.73 | 8.55 | 0.58 | 8.75 |
| all12refres_drop_varw | 0.745 | 0.893 | 0.923 | 0.681 | 0.938 | 0.947 | 0.128 | 0.990 | 0.320 | 2.32 | -1.87 | 9.25 | -0.19 | 8.79 |
| all12refin_drop | 0.756 | 0.887 | 0.920 | 0.709 | 0.937 | 0.945 | 0.129 | 0.984 | 0.341 | 2.21 | -1.84 | 7.29 | -0.24 | 8.74 |
| all12refin_drop_ctcfac | 0.790 | 0.904 | 0.909 | 0.694 | 0.934 | 0.944 | 0.148 | 0.956 | 0.332 | 2.12 | -1.76 | 6.61 | -0.59 | 6.08 |
| all12deg_refres_drop | 0.747 | 0.896 | 0.920 | 0.743 | 0.936 | 0.947 | 0.114 | 0.977 | 0.349 | 2.07 | -1.95 | 6.61 | 0.14 | 6.05 |
| all12multires | 0.724 | 0.885 | 0.917 | 0.733 | 0.927 | 0.941 | 0.132 | 0.984 | 0.326 | 2.27 | -1.91 | 7.29 | -0.19 | 8.74 |
| consensus alone | 0.811 | 0.868 | 0.891 | 0.696 | 0.913 | 0.921 | 0.153 | 0.994 | 0.350 | 2.26 | -1.82 | 8.61 | 0.58 | 8.82 |
| best other measured map | 0.688 | 0.840 | 0.859 | 0.621 | 0.894 | 0.897 | 0.188 | 1.012 | 0.297 | 2.48 | -1.61 | 9.65 | 0.36 | 8.74 |
| replicate map | 0.789 | 0.905 | 0.932 | 0.591 | 0.951 | 0.934 | 0.151 | 1.043 | 0.357 | 2.04 | -1.58 | 9.06 | 0.34 | 8.82 |
| measured map, balanced counts | 0.797 | 0.984 | 0.923 | 0.816 | 0.904 | 0.826 | 0.037 | 1.033 | 0.529 | 1.29 | -2.01 | 10.05 | -0.09 | 8.78 |
| Hi-cGAN, GM12878 model | -0.172 | 0.344 | 0.772 | 0.276 | 0.731 | 0.613 | 0.795 | 0.751 | 0.040 | 4.27 | -4.94 | 2.43 | -5.14 | 1.70 |

12 training cells, Akita valid+test stretches of chr2 and chr18, 25 kb, log O/E structure measures (median over 12 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined | rank score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres | 0.939 | 0.953 | 0.941 | 0.927 | 0.847 | 0.847 | 0.336 | 0.821 | 0.960 | 0.833 | 0.858 | 0.835 | 0.933 | 0.678 |
| all12honest | 0.828 | 0.842 | 0.820 | 0.788 | 0.698 | 0.713 | 0.215 | 0.323 | 0.881 | 0.821 | 0.808 | 0.770 | 0.773 | 0.162 |
| all12refin | 0.940 | 0.957 | 0.945 | 0.929 | 0.866 | 0.859 | 0.353 | 0.779 | 0.965 | 0.829 | 0.837 | 0.844 | 0.923 | 0.700 |
| all12refres_drop | 0.936 | 0.952 | 0.935 | 0.907 | 0.849 | 0.839 | 0.321 | 0.794 | 0.961 | 0.829 | 0.945 | 0.823 | 0.928 | 0.564 |
| all12honest_drop | 0.809 | 0.819 | 0.785 | 0.734 | 0.649 | 0.685 | 0.204 | 0.246 | 0.867 | 0.826 | 0.716 | 0.744 | 0.734 | 0.067 |
| all12refres_refdrop | 0.930 | 0.947 | 0.926 | 0.897 | 0.864 | 0.834 | 0.358 | 0.752 | 0.953 | 0.834 | 0.981 | 0.831 | 0.918 | 0.471 |
| all12refres_drop_hard | 0.929 | 0.945 | 0.928 | 0.896 | 0.848 | 0.835 | 0.329 | 0.735 | 0.956 | 0.820 | 0.860 | 0.826 | 0.918 | 0.440 |
| all12refres_drop_varw | 0.934 | 0.946 | 0.931 | 0.908 | 0.858 | 0.836 | 0.332 | 0.813 | 0.961 | 0.832 | 0.846 | 0.849 | 0.923 | 0.490 |
| all12refin_drop | 0.936 | 0.952 | 0.936 | 0.913 | 0.857 | 0.854 | 0.347 | 0.786 | 0.959 | 0.828 | 0.831 | 0.825 | 0.922 | 0.583 |
| all12refin_drop_ctcfac | 0.929 | 0.946 | 0.929 | 0.909 | 0.857 | 0.847 | 0.336 | 0.842 | 0.955 | 0.819 | 0.829 | 0.831 | 0.915 | 0.453 |
| all12deg_refres_drop | 0.936 | 0.955 | 0.939 | 0.922 | 0.847 | 0.852 | 0.339 | 0.780 | 0.964 | 0.819 | 0.866 | 0.831 | 0.928 | 0.604 |
| all12multires | 0.944 | 0.961 | 0.949 | 0.939 | 0.861 | 0.864 | 0.351 | 0.833 | 0.972 | 0.838 | 0.872 | 0.856 | 0.939 | 0.806 |
| consensus alone | 0.922 | 0.913 | 0.868 | 0.836 | 0.837 | 0.795 | 0.334 | 0.728 | 0.896 | 0.748 | 0.624 | 0.791 | 0.869 |  |
| best other measured map | 0.905 | 0.900 | 0.834 | 0.791 | 0.788 | 0.804 | 0.341 | 0.776 | 0.891 | 0.756 | 0.728 | 0.758 | 0.858 |  |
| replicate map |  | 0.972 | 0.961 | 0.951 | 0.918 | 0.893 | 0.367 | 1.078 | 0.982 | 0.831 | 0.980 | 0.935 | 0.953 |  |
| measured map, balanced counts |  |  |  |  | 0.973 | 0.884 |  |  |  |  |  |  |  |  |
| cascade fold model (all12refres setup, chromosome held out) |  | 0.950 | 0.935 | 0.919 | 0.855 | 0.843 | 0.339 | 0.749 | 0.960 | 0.831 | 0.822 | 0.849 | 0.915 |  |

12 training cells, Akita valid+test stretches of chr2 and chr18, 25 kb, count-space measures and regressor scores (median over 12 cells of the per-cell mean over chromosomes):

| source | ENT3C | HiCRep | GenomeDISCO | HiC-Spector | QuASAR-Rep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM | TabPFN 8m | TabPFN 6m | GB 7m | GB 4m | RF 4m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres | 0.919 | 0.957 | 0.932 | 0.629 | 0.991 | 0.988 | 0.092 | 1.016 | 0.609 | 1.26 | -1.79 | 9.74 | -0.19 | 8.78 |
| all12honest | 0.826 | 0.847 | 0.888 | 0.472 | 0.979 | 0.955 | 0.219 | 0.959 | 0.382 | 1.89 | -1.43 | 9.18 | 0.52 | 8.46 |
| all12refin | 0.946 | 0.959 | 0.933 | 0.575 | 0.991 | 0.988 | 0.084 | 0.996 | 0.613 | 1.26 | -1.86 | 9.18 | -0.02 | 8.76 |
| all12refres_drop | 0.906 | 0.950 | 0.927 | 0.557 | 0.991 | 0.986 | 0.111 | 1.015 | 0.596 | 1.30 | -1.81 | 8.91 | 0.36 | 8.76 |
| all12honest_drop | 0.856 | 0.826 | 0.874 | 0.534 | 0.976 | 0.950 | 0.294 | 0.963 | 0.332 | 2.25 | -1.08 | 7.65 | -0.31 | 6.39 |
| all12refres_refdrop | 0.895 | 0.940 | 0.924 | 0.492 | 0.989 | 0.983 | 0.132 | 1.006 | 0.593 | 1.40 | -1.68 | 9.74 | 0.36 | 8.75 |
| all12refres_drop_hard | 0.927 | 0.943 | 0.926 | 0.559 | 0.990 | 0.984 | 0.124 | 1.018 | 0.578 | 1.40 | -1.67 | 9.70 | 0.58 | 8.74 |
| all12refres_drop_varw | 0.933 | 0.944 | 0.929 | 0.573 | 0.990 | 0.985 | 0.113 | 1.025 | 0.609 | 1.18 | -1.74 | 9.32 | 0.22 | 8.70 |
| all12refin_drop | 0.924 | 0.951 | 0.929 | 0.552 | 0.991 | 0.986 | 0.102 | 1.015 | 0.608 | 1.33 | -1.82 | 9.41 | -0.44 | 8.70 |
| all12refin_drop_ctcfac | 0.930 | 0.945 | 0.927 | 0.595 | 0.990 | 0.984 | 0.115 | 1.013 | 0.584 | 1.35 | -1.75 | 9.79 | -0.24 | 8.79 |
| all12deg_refres_drop | 0.908 | 0.953 | 0.933 | 0.602 | 0.991 | 0.986 | 0.096 | 1.004 | 0.616 | 1.30 | -1.88 | 8.97 | -0.24 | 8.75 |
| all12multires | 0.941 | 0.964 | 0.940 | 0.668 | 0.993 | 0.989 | 0.070 | 1.004 | 0.624 | 1.15 | -1.90 | 9.59 | -0.19 | 8.79 |
| consensus alone | 0.914 | 0.888 | 0.890 | 0.486 | 0.980 | 0.970 | 0.221 | 1.060 | 0.550 | 1.45 | -0.93 | 7.65 | 0.52 | 8.79 |
| best other measured map | 0.902 | 0.851 | 0.890 | 0.484 | 0.978 | 0.964 | 0.260 | 0.997 | 0.513 | 1.71 | -1.11 | 8.84 | -0.32 | 8.40 |
| replicate map | 0.888 | 0.978 | 0.952 | 0.770 | 0.994 | 0.992 | 0.056 | 1.028 | 0.703 | 1.07 | -1.95 | 10.05 | -1.27 | 8.83 |
| measured map, balanced counts | 0.901 | 0.995 | 0.967 | 0.794 | 0.993 | 0.975 | 0.031 | 1.088 | 0.671 | 0.82 | -2.03 | 9.43 | -0.02 | 8.77 |
| cascade fold model (all12refres setup, chromosome held out) | 0.954 | 0.954 | 0.931 | 0.575 | 0.990 | 0.985 | 0.092 | 1.013 | 0.605 | 1.33 | -1.83 | 10.12 | -0.02 | 8.78 |

12 training cells, Akita valid+test stretches of chr2 and chr18, 10 kb, log O/E structure measures (median over 12 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined | rank score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres | 0.935 | 0.941 | 0.923 | 0.919 | 0.713 | 0.718 | 0.393 | 0.771 | 0.971 | 0.832 | 0.864 | 0.869 | 0.865 | 0.740 |
| all12honest | 0.830 | 0.820 | 0.802 | 0.792 | 0.539 | 0.574 | 0.217 | 0.381 | 0.898 | 0.799 | 0.764 | 0.753 | 0.708 | 0.111 |
| all12refin | 0.934 | 0.935 | 0.924 | 0.919 | 0.691 | 0.705 | 0.424 | 0.715 | 0.971 | 0.827 | 0.843 | 0.864 | 0.864 | 0.693 |
| all12refres_drop | 0.927 | 0.927 | 0.911 | 0.897 | 0.691 | 0.693 | 0.388 | 0.845 | 0.963 | 0.828 | 0.866 | 0.862 | 0.868 | 0.576 |
| all12honest_drop | 0.819 | 0.813 | 0.787 | 0.748 | 0.530 | 0.544 | 0.218 | 0.271 | 0.887 | 0.810 | 0.780 | 0.753 | 0.683 | 0.065 |
| all12refres_refdrop | 0.924 | 0.922 | 0.907 | 0.885 | 0.678 | 0.685 | 0.408 | 0.714 | 0.961 | 0.822 | 0.867 | 0.842 | 0.859 | 0.446 |
| all12refres_drop_hard | 0.915 | 0.912 | 0.896 | 0.871 | 0.680 | 0.673 | 0.371 | 0.749 | 0.951 | 0.821 | 0.809 | 0.819 | 0.845 | 0.357 |
| all12refres_drop_varw | 0.927 | 0.928 | 0.917 | 0.901 | 0.691 | 0.709 | 0.410 | 0.827 | 0.968 | 0.820 | 0.827 | 0.851 | 0.865 | 0.617 |
| all12refin_drop | 0.927 | 0.930 | 0.915 | 0.901 | 0.684 | 0.711 | 0.369 | 0.730 | 0.965 | 0.818 | 0.903 | 0.856 | 0.866 | 0.527 |
| all12refin_drop_ctcfac | 0.922 | 0.920 | 0.909 | 0.895 | 0.676 | 0.692 | 0.387 | 0.826 | 0.962 | 0.826 | 0.884 | 0.866 | 0.865 | 0.481 |
| all12deg_refres_drop | 0.924 | 0.931 | 0.913 | 0.901 | 0.686 | 0.700 | 0.369 | 0.826 | 0.964 | 0.830 | 0.856 | 0.841 | 0.859 | 0.568 |
| all12multires | 0.938 | 0.940 | 0.930 | 0.929 | 0.717 | 0.716 | 0.392 | 0.795 | 0.973 | 0.836 | 0.902 | 0.873 | 0.879 | 0.841 |
| consensus alone | 0.887 | 0.886 | 0.850 | 0.810 | 0.656 | 0.645 | 0.385 | 0.739 | 0.903 | 0.745 | 0.641 | 0.790 | 0.793 |  |
| best other measured map | 0.860 | 0.836 | 0.776 | 0.738 | 0.568 | 0.599 | 0.275 | 0.601 | 0.880 | 0.757 | 0.778 | 0.742 | 0.744 |  |
| replicate map |  | 0.919 | 0.895 | 0.930 | 0.660 | 0.626 | 0.123 | 1.225 | 0.976 | 0.825 | 0.790 | 0.921 | 0.878 |  |
| measured map, balanced counts |  |  |  |  | 0.879 | 0.862 |  |  |  |  |  |  |  |  |
| cascade fold model (all12refres setup, chromosome held out) |  | 0.923 | 0.910 | 0.900 | 0.697 | 0.703 | 0.405 | 0.791 | 0.964 | 0.823 | 0.825 | 0.855 | 0.868 |  |

12 training cells, Akita valid+test stretches of chr2 and chr18, 10 kb, count-space measures and regressor scores (median over 12 cells of the per-cell mean over chromosomes):

| source | HiCRep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM |
|---|---|---|---|---|---|
| all12refres | 0.948 | 0.984 | 0.120 | 0.967 | 0.426 |
| all12honest | 0.813 | 0.946 | 0.274 | 0.922 | 0.258 |
| all12refin | 0.947 | 0.983 | 0.125 | 0.965 | 0.421 |
| all12refres_drop | 0.933 | 0.981 | 0.168 | 0.965 | 0.424 |
| all12honest_drop | 0.798 | 0.940 | 0.351 | 0.954 | 0.263 |
| all12refres_refdrop | 0.931 | 0.980 | 0.184 | 0.973 | 0.408 |
| all12refres_drop_hard | 0.911 | 0.977 | 0.205 | 0.986 | 0.402 |
| all12refres_drop_varw | 0.934 | 0.981 | 0.160 | 0.976 | 0.407 |
| all12refin_drop | 0.938 | 0.981 | 0.164 | 0.963 | 0.395 |
| all12refin_drop_ctcfac | 0.932 | 0.980 | 0.169 | 0.965 | 0.396 |
| all12deg_refres_drop | 0.934 | 0.981 | 0.159 | 0.974 | 0.378 |
| all12multires | 0.953 | 0.985 | 0.107 | 0.978 | 0.419 |
| consensus alone | 0.870 | 0.959 | 0.307 | 1.008 | 0.372 |
| best other measured map | 0.805 | 0.948 | 0.429 | 0.997 | 0.311 |
| replicate map | 0.961 | 0.975 | 0.112 | 1.056 | 0.412 |
| measured map, balanced counts | 0.991 | 0.937 | 0.043 | 1.015 | 0.550 |
| cascade fold model (all12refres setup, chromosome held out) | 0.933 | 0.980 | 0.149 | 0.976 | 0.397 |

12 training cells, whole chr2 and chr18, 25 kb, log O/E structure measures (median over 12 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.939 | 0.953 | 0.939 | 0.922 | 0.815 | 0.815 | 0.339 | 0.855 | 0.959 | 0.822 | 0.853 | 0.870 | 0.940 |
| cascade fold model (all12refres setup, chromosome held out) |  | 0.948 | 0.931 | 0.921 | 0.816 | 0.811 | 0.333 | 0.806 | 0.951 | 0.820 | 0.803 | 0.861 | 0.928 |
| consensus alone | 0.922 | 0.917 | 0.863 | 0.829 | 0.796 | 0.792 | 0.336 | 0.762 | 0.888 | 0.765 | 0.671 | 0.794 | 0.875 |
| best other measured map | 0.905 | 0.884 | 0.817 | 0.778 | 0.733 | 0.747 | 0.307 | 0.798 | 0.867 | 0.760 | 0.755 | 0.757 | 0.865 |
| replicate map |  | 0.970 | 0.954 | 0.946 | 0.878 | 0.829 | 0.351 | 1.114 | 0.977 | 0.838 | 0.976 | 0.953 | 0.951 |
| measured map, balanced counts |  |  |  |  | 0.951 | 0.869 |  |  |  |  |  |  |  |
| Hi-cGAN (GM12878, held-out chromosome) |  | 0.452 | 0.215 | 0.161 | 0.209 | 0.268 | 0.045 | -0.047 | 0.608 | 0.700 | 0.751 | 0.611 | 0.420 |

12 training cells, whole chr2 and chr18, 25 kb, count-space measures and regressor scores (median over 12 cells of the per-cell mean over chromosomes):

| source | ENT3C | HiCRep | GenomeDISCO | HiC-Spector | QuASAR-Rep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM | TabPFN 8m | TabPFN 6m | GB 7m | GB 4m | RF 4m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.818 | 0.949 | 0.932 | 0.833 | 0.990 | 0.959 | 0.104 | 1.000 | 0.563 | 1.31 | -1.78 | 9.90 | -0.44 | 8.78 |
| cascade fold model (all12refres setup, chromosome held out) | 0.858 | 0.942 | 0.926 | 0.833 | 0.988 | 0.954 | 0.110 | 1.003 | 0.558 | 1.33 | -1.74 | 8.68 | 0.14 | 8.79 |
| consensus alone | 0.759 | 0.862 | 0.891 | 0.766 | 0.977 | 0.907 | 0.213 | 1.039 | 0.502 | 1.27 | -1.04 | 8.58 | 0.21 | 8.74 |
| best other measured map | 0.784 | 0.820 | 0.880 | 0.738 | 0.972 | 0.882 | 0.300 | 1.000 | 0.456 | 1.70 | -0.80 | 9.00 | -0.22 | 8.69 |
| replicate map | 0.902 | 0.968 | 0.949 | 0.743 | 0.993 | 0.980 | 0.045 | 1.061 | 0.652 | 0.85 | -1.94 | 7.53 | -1.11 | 8.37 |
| measured map, balanced counts | 0.878 | 0.983 | 0.949 | 0.865 | 0.993 | 0.901 | 0.045 | 1.093 | 0.639 | 0.74 | -1.96 | 8.26 | 0.05 | 8.56 |
| Hi-cGAN (GM12878, held-out chromosome) | 0.392 | 0.281 | 0.834 | 0.455 | 0.935 | 0.646 | 0.972 | 0.779 | 0.071 | 3.13 | 4.06 | 5.23 | -2.90 | 7.16 |

12 training cells, whole chr2 and chr18, 10 kb, log O/E structure measures (median over 12 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | loop F1 (called) | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.935 | 0.929 | 0.925 | 0.917 | 0.649 | 0.682 | 0.375 | 0.802 | 0.396 | 0.969 | 0.825 | 0.860 | 0.893 | 0.870 |
| cascade fold model (all12refres setup, chromosome held out) |  | 0.919 | 0.906 | 0.900 | 0.633 | 0.667 | 0.373 | 0.818 | 0.406 | 0.958 | 0.821 | 0.811 | 0.872 | 0.858 |
| consensus alone | 0.887 | 0.879 | 0.843 | 0.801 | 0.624 | 0.622 | 0.345 | 0.744 | 0.331 | 0.888 | 0.775 | 0.684 | 0.793 | 0.804 |
| best other measured map | 0.860 | 0.823 | 0.750 | 0.720 | 0.542 | 0.550 | 0.236 | 0.608 | 0.264 | 0.837 | 0.770 | 0.775 | 0.725 | 0.728 |
| replicate map |  | 0.910 | 0.879 | 0.925 | 0.650 | 0.608 | 0.084 | 1.160 | 0.275 | 0.969 | 0.841 | 0.776 | 0.908 | 0.877 |
| measured map, balanced counts |  |  |  |  | 0.866 | 0.838 |  |  | 0.904 |  |  |  |  |  |
| C.Origami (IMR-90, chr2) |  | 0.904 | 0.914 | 0.870 | 0.466 | 0.519 | 0.145 | 0.733 | 0.196 | 0.971 | 0.802 | 1.155 | 0.931 | 0.797 |
| Hi-cGAN (GM12878, held-out chromosome) |  | 0.500 | 0.300 | 0.184 | 0.117 | 0.168 | 0.028 | 0.139 | 0.027 | 0.664 | 0.718 | 0.832 | 0.594 | 0.427 |
| Hi-cGAN, GM12878 model on IMR-90 (chr2) |  | 0.493 | 0.314 | 0.564 | 0.137 | 0.174 | 0.040 | 0.202 | 0.050 | 0.717 | 0.655 | 0.583 | 0.498 | 0.393 |
| Hi-cGAN, IMR-90 model (chr2) |  | 0.590 | 0.567 | 0.584 | 0.105 | 0.120 | 0.014 | 0.120 | 0.036 | 0.734 | 0.757 | 0.661 | 0.639 | 0.480 |

12 training cells, whole chr2 and chr18, 10 kb, count-space measures and regressor scores (median over 12 cells of the per-cell mean over chromosomes):

| source | HiCRep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM |
|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.946 | 0.949 | 0.131 | 0.957 | 0.403 |
| cascade fold model (all12refres setup, chromosome held out) | 0.929 | 0.939 | 0.157 | 0.972 | 0.379 |
| consensus alone | 0.862 | 0.891 | 0.272 | 1.014 | 0.353 |
| best other measured map | 0.800 | 0.836 | 0.433 | 1.003 | 0.295 |
| replicate map | 0.952 | 0.949 | 0.071 | 1.071 | 0.395 |
| measured map, balanced counts | 0.985 | 0.832 | 0.058 | 1.022 | 0.552 |
| C.Origami (IMR-90, chr2) | 0.935 | 0.944 | 0.152 | 0.897 | 0.198 |
| Hi-cGAN (GM12878, held-out chromosome) | 0.372 | 0.672 | 1.031 | 0.714 | 0.044 |
| Hi-cGAN, GM12878 model on IMR-90 (chr2) | 0.463 | 0.636 | 0.554 | 0.786 | 0.064 |
| Hi-cGAN, IMR-90 model (chr2) | 0.646 | 0.744 | 0.519 | 0.873 | 0.044 |

GM12878, whole chr2 and chr18, 1 Mb band (HiC-Reg reach), 25 kb, log O/E structure measures (median over 1 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined | rank score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.938 | 0.943 | 0.927 | 0.912 | 0.821 | 0.822 | 0.398 | 0.789 | 0.947 | 0.802 | 0.893 | 0.830 | 0.935 | 0.657 |
| all12honest (chromosome in training) | 0.822 | 0.822 | 0.803 | 0.787 | 0.670 | 0.656 | 0.223 | 0.443 | 0.874 | 0.833 | 0.876 | 0.724 | 0.791 | 0.194 |
| all12refin (chromosome in training) | 0.939 | 0.944 | 0.926 | 0.910 | 0.858 | 0.841 | 0.423 | 0.797 | 0.949 | 0.799 | 0.858 | 0.850 | 0.928 | 0.748 |
| all12refres_drop (chromosome in training) | 0.931 | 0.935 | 0.912 | 0.887 | 0.847 | 0.822 | 0.399 | 0.817 | 0.937 | 0.810 | 0.958 | 0.832 | 0.929 | 0.556 |
| all12honest_drop (chromosome in training) | 0.810 | 0.794 | 0.765 | 0.706 | 0.659 | 0.644 | 0.225 | 0.271 | 0.846 | 0.800 | 0.785 | 0.694 | 0.748 | 0.056 |
| all12refres_refdrop (chromosome in training) | 0.926 | 0.928 | 0.902 | 0.857 | 0.818 | 0.807 | 0.409 | 0.787 | 0.935 | 0.801 | 0.956 | 0.817 | 0.922 | 0.452 |
| all12refres_drop_hard (chromosome in training) | 0.924 | 0.922 | 0.898 | 0.857 | 0.799 | 0.801 | 0.402 | 0.744 | 0.928 | 0.813 | 0.904 | 0.820 | 0.922 | 0.375 |
| all12refres_drop_varw (chromosome in training) | 0.927 | 0.929 | 0.909 | 0.876 | 0.831 | 0.820 | 0.401 | 0.862 | 0.935 | 0.790 | 0.919 | 0.828 | 0.931 | 0.504 |
| all12refin_drop (chromosome in training) | 0.936 | 0.936 | 0.916 | 0.893 | 0.845 | 0.831 | 0.420 | 0.753 | 0.942 | 0.798 | 0.831 | 0.822 | 0.929 | 0.601 |
| all12refin_drop_ctcfac (chromosome in training) | 0.927 | 0.923 | 0.901 | 0.879 | 0.815 | 0.821 | 0.407 | 0.853 | 0.926 | 0.775 | 0.828 | 0.825 | 0.917 | 0.384 |
| all12deg_refres_drop (chromosome in training) | 0.934 | 0.937 | 0.924 | 0.903 | 0.829 | 0.832 | 0.402 | 0.819 | 0.943 | 0.785 | 0.854 | 0.845 | 0.931 | 0.618 |
| all12multires (chromosome in training) | 0.942 | 0.951 | 0.945 | 0.931 | 0.842 | 0.859 | 0.434 | 0.829 | 0.962 | 0.799 | 0.876 | 0.862 | 0.948 | 0.856 |
| consensus alone | 0.882 | 0.865 | 0.796 | 0.753 | 0.804 | 0.762 | 0.400 | 0.758 | 0.831 | 0.727 | 0.756 | 0.765 | 0.856 |  |
| best other measured map | 0.844 | 0.812 | 0.734 | 0.659 | 0.718 | 0.686 | 0.348 | 0.718 | 0.796 | 0.726 | 0.805 | 0.723 | 0.800 |  |
| replicate map |  | 0.954 | 0.933 | 0.901 | 0.879 | 0.814 | 0.397 | 1.122 | 0.971 | 0.826 | 1.204 | 0.886 | 0.930 |  |
| measured map, balanced counts |  |  |  |  | 0.961 | 0.859 |  |  |  |  |  |  |  |  |
| cascade fold model (all12refres setup, chromosome held out) |  | 0.928 | 0.912 | 0.894 | 0.826 | 0.817 | 0.384 | 0.771 | 0.934 | 0.799 | 0.894 | 0.817 | 0.921 |  |
| HiC-Reg (within-chromosome CV) |  | 0.946 | 0.843 | 0.562 | 0.689 | 0.782 | 0.259 | 0.346 | 0.980 | 0.796 | 0.613 | 0.890 | 0.836 |  |

GM12878, whole chr2 and chr18, 1 Mb band (HiC-Reg reach), 25 kb, count-space measures and regressor scores (median over 1 cells of the per-cell mean over chromosomes):

| source | ENT3C | HiCRep | GenomeDISCO | HiC-Spector | QuASAR-Rep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM | TabPFN 8m | TabPFN 6m | GB 7m | GB 4m | RF 4m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.584 | 0.940 | 0.944 | 0.862 | 0.994 | 0.969 | 0.068 | 0.998 | 0.549 | 1.23 | -1.93 | 10.36 | 0.14 | 8.79 |
| all12honest (chromosome in training) | 0.250 | 0.836 | 0.909 | 0.907 | 0.985 | 0.911 | 0.162 | 1.021 | 0.305 | 1.38 | -1.51 | 9.74 | 0.36 | 8.78 |
| all12refin (chromosome in training) | 0.781 | 0.937 | 0.944 | 0.848 | 0.994 | 0.968 | 0.072 | 1.020 | 0.547 | 1.07 | -1.88 | 10.05 | 0.14 | 8.78 |
| all12refres_drop (chromosome in training) | 0.628 | 0.922 | 0.938 | 0.810 | 0.993 | 0.963 | 0.089 | 1.030 | 0.506 | 1.21 | -1.81 | 9.74 | -0.19 | 8.78 |
| all12honest_drop (chromosome in training) | 0.206 | 0.799 | 0.903 | 0.842 | 0.983 | 0.897 | 0.225 | 1.024 | 0.283 | 1.53 | -1.21 | 10.05 | 1.03 | 8.85 |
| all12refres_refdrop (chromosome in training) | 0.538 | 0.910 | 0.934 | 0.872 | 0.992 | 0.959 | 0.118 | 1.013 | 0.496 | 1.38 | -1.78 | 10.36 | -0.59 | 8.79 |
| all12refres_drop_hard (chromosome in training) | 0.509 | 0.907 | 0.936 | 0.880 | 0.993 | 0.958 | 0.114 | 0.997 | 0.497 | 1.37 | -1.80 | 10.36 | -0.28 | 8.79 |
| all12refres_drop_varw (chromosome in training) | 0.489 | 0.920 | 0.937 | 0.840 | 0.993 | 0.962 | 0.097 | 1.013 | 0.536 | 1.28 | -1.84 | 10.36 | -0.19 | 9.03 |
| all12refin_drop (chromosome in training) | 0.660 | 0.929 | 0.941 | 0.859 | 0.994 | 0.964 | 0.081 | 1.017 | 0.548 | 1.19 | -1.86 | 10.36 | -0.19 | 9.03 |
| all12refin_drop_ctcfac (chromosome in training) | 0.667 | 0.913 | 0.936 | 0.821 | 0.992 | 0.958 | 0.098 | 0.994 | 0.513 | 1.44 | -1.89 | 8.61 | -0.19 | 8.46 |
| all12deg_refres_drop (chromosome in training) | 0.513 | 0.936 | 0.943 | 0.819 | 0.994 | 0.968 | 0.074 | 0.997 | 0.554 | 1.29 | -1.90 | 10.36 | 0.14 | 8.79 |
| all12multires (chromosome in training) | 0.687 | 0.955 | 0.952 | 0.843 | 0.996 | 0.976 | 0.053 | 0.981 | 0.563 | 1.29 | -2.00 | 8.61 | 0.14 | 8.82 |
| consensus alone | 0.758 | 0.808 | 0.890 | 0.765 | 0.982 | 0.910 | 0.221 | 1.055 | 0.423 | 1.11 | -1.01 | 9.43 | 0.98 | 9.08 |
| best other measured map | 0.778 | 0.748 | 0.896 | 0.735 | 0.982 | 0.887 | 0.300 | 1.046 | 0.349 | 1.60 | -0.28 | 10.05 | -0.22 | 8.85 |
| replicate map | 0.675 | 0.941 | 0.933 | 0.793 | 0.995 | 0.967 | 0.091 | 1.079 | 0.623 | 0.94 | -1.68 | 9.43 | -0.59 | 9.23 |
| measured map, balanced counts | 0.818 | 0.995 | 0.977 | 0.901 | 0.998 | 0.966 | 0.024 | 1.120 | 0.619 | 0.62 | -2.07 | 5.32 | -0.18 | 7.96 |
| cascade fold model (all12refres setup, chromosome held out) | 0.852 | 0.925 | 0.938 | 0.857 | 0.993 | 0.963 | 0.083 | 1.010 | 0.516 | 1.18 | -1.85 | 10.36 | -0.19 | 9.03 |
| HiC-Reg (within-chromosome CV) | 0.379 | 0.870 | 0.906 | 0.790 | 0.986 | 0.908 | 0.275 | 0.881 | 0.381 | 2.04 | -1.63 | 6.61 | -0.96 | 6.50 |

GM12878, whole chr17, 990 kb band (Epiphany reach), 10 kb, log O/E structure measures (median over 1 cells of the per-cell mean over chromosomes):

| source | Pearson 413 regions | Pearson 1 Mb tiles | SCC | insulation r | boundary recall | boundary precision | loop recall | APA ratio | loop F1 (called) | compartment preference r | A/B sign agreement | saddle ratio | local E1 r | window combined | rank score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.938 | 0.934 | 0.928 | 0.916 | 0.748 | 0.757 | 0.475 | 0.863 | 0.483 | 0.966 | 0.733 | 0.830 | 0.874 | 0.903 | 0.778 |
| all12honest (chromosome in training) | 0.835 | 0.796 | 0.788 | 0.785 | 0.544 | 0.561 | 0.239 | 0.382 | 0.141 | 0.873 | 0.780 | 0.605 | 0.699 | 0.700 | 0.140 |
| all12refin (chromosome in training) | 0.941 | 0.936 | 0.926 | 0.929 | 0.728 | 0.747 | 0.471 | 0.779 | 0.476 | 0.956 | 0.724 | 0.750 | 0.876 | 0.901 | 0.657 |
| all12refres_drop (chromosome in training) | 0.930 | 0.923 | 0.916 | 0.904 | 0.738 | 0.765 | 0.496 | 0.922 | 0.500 | 0.949 | 0.743 | 0.778 | 0.839 | 0.895 | 0.574 |
| all12honest_drop (chromosome in training) | 0.815 | 0.761 | 0.760 | 0.715 | 0.495 | 0.479 | 0.257 | 0.275 | 0.127 | 0.857 | 0.758 | 0.532 | 0.717 | 0.663 | 0.078 |
| all12refres_refdrop (chromosome in training) | 0.928 | 0.926 | 0.912 | 0.896 | 0.682 | 0.698 | 0.486 | 0.807 | 0.469 | 0.948 | 0.748 | 0.790 | 0.841 | 0.884 | 0.464 |
| all12refres_drop_hard (chromosome in training) | 0.920 | 0.919 | 0.901 | 0.856 | 0.711 | 0.711 | 0.467 | 0.835 | 0.452 | 0.935 | 0.719 | 0.697 | 0.833 | 0.885 | 0.326 |
| all12refres_drop_varw (chromosome in training) | 0.936 | 0.927 | 0.921 | 0.916 | 0.695 | 0.709 | 0.478 | 0.860 | 0.486 | 0.952 | 0.743 | 0.801 | 0.831 | 0.893 | 0.632 |
| all12refin_drop (chromosome in training) | 0.933 | 0.930 | 0.918 | 0.914 | 0.682 | 0.705 | 0.431 | 0.709 | 0.425 | 0.955 | 0.738 | 0.782 | 0.865 | 0.877 | 0.432 |
| all12refin_drop_ctcfac (chromosome in training) | 0.923 | 0.915 | 0.896 | 0.873 | 0.675 | 0.696 | 0.504 | 0.985 | 0.520 | 0.928 | 0.726 | 0.829 | 0.844 | 0.885 | 0.456 |
| all12deg_refres_drop (chromosome in training) | 0.934 | 0.929 | 0.920 | 0.910 | 0.675 | 0.694 | 0.478 | 0.906 | 0.497 | 0.950 | 0.751 | 0.752 | 0.872 | 0.898 | 0.594 |
| all12multires (chromosome in training) | 0.943 | 0.942 | 0.933 | 0.927 | 0.721 | 0.717 | 0.514 | 0.864 | 0.483 | 0.965 | 0.743 | 0.809 | 0.876 | 0.905 | 0.869 |
| consensus alone | 0.858 | 0.877 | 0.832 | 0.803 | 0.725 | 0.674 | 0.453 | 0.900 | 0.499 | 0.863 | 0.689 | 0.907 | 0.786 | 0.846 |  |
| best other measured map | 0.799 | 0.824 | 0.782 | 0.808 | 0.689 | 0.650 | 0.297 | 0.906 | 0.445 | 0.828 | 0.682 | 0.802 | 0.702 | 0.801 |  |
| replicate map |  | 0.869 | 0.855 | 0.896 | 0.751 | 0.682 | 0.109 | 1.366 | 0.491 | 0.951 | 0.746 | 0.846 | 0.868 | 0.811 |  |
| measured map, balanced counts |  |  |  |  | 0.859 | 0.853 |  |  | 0.944 |  |  |  |  |  |  |
| cascade fold model (all12refres setup, chromosome held out) |  | 0.921 | 0.915 | 0.915 | 0.711 | 0.759 | 0.453 | 0.862 | 0.476 | 0.954 | 0.748 | 0.863 | 0.870 | 0.889 |  |
| Epiphany (held-out chr17) |  | 0.511 | 0.341 | 0.248 | 0.226 | 0.267 | 0.047 | 0.220 | 0.058 | 0.603 | 0.782 | 0.432 | 0.652 | 0.480 |  |

GM12878, whole chr17, 990 kb band (Epiphany reach), 10 kb, count-space measures and regressor scores (median over 1 cells of the per-cell mean over chromosomes):

| source | HiCRep | Pearson AUC | TAD score MSE | TAD fraction | TAD FEM |
|---|---|---|---|---|---|
| all12refres (chromosome in training) | 0.946 | 0.957 | 0.112 | 0.987 | 0.462 |
| all12honest (chromosome in training) | 0.817 | 0.858 | 0.225 | 0.970 | 0.252 |
| all12refin (chromosome in training) | 0.947 | 0.955 | 0.091 | 0.974 | 0.472 |
| all12refres_drop (chromosome in training) | 0.936 | 0.948 | 0.124 | 0.964 | 0.469 |
| all12honest_drop (chromosome in training) | 0.793 | 0.845 | 0.303 | 1.033 | 0.249 |
| all12refres_refdrop (chromosome in training) | 0.937 | 0.949 | 0.136 | 0.977 | 0.443 |
| all12refres_drop_hard (chromosome in training) | 0.926 | 0.942 | 0.182 | 1.000 | 0.443 |
| all12refres_drop_varw (chromosome in training) | 0.943 | 0.953 | 0.108 | 0.980 | 0.452 |
| all12refin_drop (chromosome in training) | 0.937 | 0.947 | 0.114 | 0.967 | 0.436 |
| all12refin_drop_ctcfac (chromosome in training) | 0.920 | 0.937 | 0.156 | 0.970 | 0.413 |
| all12deg_refres_drop (chromosome in training) | 0.942 | 0.952 | 0.116 | 0.974 | 0.452 |
| all12multires (chromosome in training) | 0.950 | 0.959 | 0.097 | 1.007 | 0.469 |
| consensus alone | 0.870 | 0.900 | 0.250 | 1.075 | 0.420 |
| best other measured map | 0.846 | 0.874 | 0.222 | 1.059 | 0.377 |
| replicate map | 0.935 | 0.909 | 0.125 | 1.102 | 0.485 |
| measured map, balanced counts | 0.995 | 0.935 | 0.017 | 1.007 | 0.538 |
| cascade fold model (all12refres setup, chromosome held out) | 0.935 | 0.946 | 0.113 | 0.938 | 0.456 |
| Epiphany (held-out chr17) | 0.518 | 0.634 | 0.743 | 0.846 | 0.111 |

## 32. Cohesin-unknown indicator (PENDING)

Problem (section 25): a RAD21 track declared absent still strengthens most
loops (9 to 37% of the loops weaker at 10 and 25 kb in every model), and the
U-937 STAG2 knockout without knockout ChIP stays wrong. An absent track is
encoded by its raw-zero value, which the models read as a low signal and not
as missing information; five of the twelve training cells (HFF, NHEK,
GM23248, PC-9, CD14+ monocyte) have no RAD21 and no SMC3 ChIP, so "no
cohesin input" reads as the average of these cells.

Implementation (`data.cohesin_unknown`, default off). One more constant
conditioning row after the assay rows (`cohesin_unknown`, cond row
`cohesin:unknown`): 1 where neither RAD21 nor SMC3 is measured (missing or
zeroed for the sample), 0 where cohesin is measured; a sample with SMC3 and
no RAD21 does not occur. During training with track dropout the SMC3 row
follows the RAD21 dropout draw, so the cohesin group is dropped jointly, and
the indicator row is set to 1 in every window that loses it (the RAD21 draw
keeps its marginal; SMC3 is absent anyway in HCT116, H1 and MCF-7). At inference crosscell_predict sets the indicator
to 1 for a cell without cohesin ChIP (HMEC, HUVEC, U-937 knockout input with
RAD21 and SMC3 declared absent) and for keep sets that remove both tracks;
`--cohesin-unknown 0` and keep sets named `*_ind0` force 0, which tests the
same model with absent RAD21 and the indicator withheld. The degron recipe is
the base (hct116 reads the untreated degron ChIP, hct116_auxin the 6 h auxin
Hi-C and ChIP on odd chromosomes, uniform-count track dropout 0.5, honest
validation): runs all12cohunk_refres_drop (residual on the leave-one-out
consensus) and all12cohunk_honest_drop (no reference), 25 and 10 kb. Tests: tests/test_pipeline.py (cohesin_unknown row, training).

Evaluation as sections 15 and 25: 413 regions, held-out K562, HMEC deep,
HUVEC deep against the consensus; perturb_validate.py (RAD21 and CTCF
degrons, even chromosomes), aml_eval.py (MOLM-13, U-937 and the knockout
variants ko_ctcfonly, ko_half, ko_absent and ko_absent_ind0), the washout
test, and the hard-region summary.

Results: pending.

## 33. Recipes at 5 kb (PENDING)

The 25 and 10 kb recipes of sections 22, 25, 26, 30 and 32 at 5 kb (window
384 bins, 1.92 Mb), in the order a, c, b, d, e, all with honest validation
(val_exclude_overlap) and no training window overlapping a validation
position: (a) track dropout, all12refres_drop_5kb and all12honest_drop_5kb;
(c) the RAD21 degron in training, all12deg_refres_drop_5kb (Rao et al. 2017
auxin Hi-C at 5 kb built with oe_proper.py from the 4DN mcool level 5000,
with the auxin ChIP bigWigs of section 25, which are base-resolution fold
changes); (b) all12refin_drop_5kb (input-only reference with track
dropout); (d) all12cohunk_refres_drop_5kb (section 32); (e)
all12refin_drop_struct3_5kb (insulation, donut and compartment loss terms at
the weights chosen on the 25 kb grid, section 30; the terms keep their window
sizes in bins, 10-bin insulation square and 2-to-5-bin ring, which cover
half of the physical size at 5 kb that they cover at 10 kb). Evaluation: the
standard 413-region table, AML (MOLM-13, U-937 and the knockout variants), the
perturbation test for the dropout, degron and cohesin-unknown models, and
the structure measures for (e). The hard split exists at 25 and 10 kb only,
so hard-region metrics do not apply at 5 kb.

Results: pending.

## 34. Recipes at 2 kb (PENDING)

The same recipes at 2 kb (window 512 bins, 1.02 Mb), after the 5 kb block:
all12refres_drop, all12honest_drop, all12deg_refres_drop, all12refin_drop
and all12refin_drop_struct3 (names without suffix), the cohesin-unknown model
as the last, optional job. Training windows are read from memory-mapped
files on local disk (train.stage_dir, ~/scratch/hicmx_stage) because
the resident 2 kb data exceed the RAM of both ranks (earlyoom killed
all12refres and all12casc on 2026-09-30). MOLM-13 and U-937 maps start at
5 kb, so AML does not apply; the perturbation test covers the dropout and
degron models. The 4DN mcool of the Rao 2017 auxin Hi-C has a 2 kb level, built
with oe_proper.py.

Results: pending.

## 35. How far DNA methylation replaces ChIP-seq signal (A, B FINAL; C PENDING)

Model of parts A: all12refres_drop_meth_10kb (13 ChIP tracks + Meth, residual
on the leave-one-out consensus, track dropout 0.5, honest validation, retrained
after the select_tracks fix so that Meth reaches the network; section 27), next
to all12refres_drop_10kb (same recipe without the channel). Meth is the ENCODE
WGBS methylated fraction per 1 kb (value - 1), available for seven of the 12
training cells (GM12878, IMR-90, H1, HepG2, A549, GM23248, CD14+ monocyte)
and for the held-out K562 and HMEC.

**Part A, inference only** (`scripts/meth_replace.py`, 413 Akita test regions,
TTA, `crosscell_predict --structure`; tables `results/methylation/
meth_replace_scores.csv`, `meth_replacement_fraction_*.csv`). ChIP groups:
CTCF; cohesin = RAD21 + SMC3; active = H3K27ac, H3K4me1, H3K4me2, H3K4me3,
H3K9ac; gene body = H3K36me3, H3K79me2, H4K20me1; repressive = H3K27me3,
H3K9me3. Sets: each single group removed; cumulative removal in the order
cohesin (cum1), + repressive (cum2), + gene body (cum3), + active (CTCF only),
+ CTCF (sequence and motifs only); the ChIP-light sets CTCF + H3K27ac,
CTCF + H3K27ac + H3K27me3. Every set was predicted without and with Meth, the
other tracks at their absent value (track dropout makes this a trained state).
Replacement fraction RF = (score with Meth - score without) / (score with all
13 ChIP tracks - score without), per cell and metric, and pooled over cells
(mean gain / mean loss, percentile bootstrap over cells for the interval). A
fraction is given only where the loss without the group exceeds the seed noise
of the median Pearson (0.013, section 15); smaller losses are marked n/d.
The seed noise was measured for Pearson only, so the other metrics carry the
same threshold without a separate noise estimate. Seven cells: the five
training cells with WGBS in the pooled "training" rows and K562 (held out)
and HMEC deep (held out) separately. The model saw Meth for the training
cells during training; K562 and HMEC are unseen.

Median Pearson with all 13 ChIP tracks, and the consensus the model reads:

| cell | full | consensus alone |
|---|---|---|
| GM12878 | 0.930 | 0.858 |
| IMR-90 | 0.945 | 0.903 |
| HepG2 | 0.917 | 0.874 |
| A549 | 0.928 | 0.913 |
| H1 | 0.920 | 0.886 |
| K562 (held out) | 0.904 | 0.859 |
| HMEC deep (held out) | 0.902 | 0.906 |

Removing ChIP groups costs little because the reference map carries most of
the signal: removing any single group changes the median Pearson by at most
0.04 (GM12878, cohesin), and removing all ChIP (sequence and motifs only)
costs 0.045 on average over the five training cells (0.076 GM12878, 0.049
K562, 0.002 HMEC, which lacks RAD21 and SMC3 and sits at the consensus level).
Pooled over the five training cells, mean loss without the group / mean gain
from Meth / RF:

| set removed or kept | Pearson | SCC | insulation corr. | loop enrichment corr. | compartment corr. | first eigenvector |
|---|---|---|---|---|---|---|
| minus cohesin | 0.017 / 0.016 / 0.93 | 0.021 / 0.019 / 0.93 | 0.009 / 0.008 / n/d | 0.014 / 0.009 / 0.65 | 0.013 / 0.012 / n/d | 0.007 / 0.007 / n/d |
| minus active | 0.012 / 0.003 / n/d | 0.012 / 0.004 / n/d | 0.006 / 0.001 / n/d | 0.021 / 0.003 / 0.13 | 0.008 / 0.001 / n/d | 0.003 / 0.001 / n/d |
| cum3 (CTCF + active kept) | 0.018 / 0.014 / 0.76 | 0.021 / 0.016 / 0.77 | 0.010 / 0.008 / n/d | 0.016 / 0.009 / 0.56 | 0.014 / 0.011 / 0.79 | 0.007 / 0.006 / n/d |
| CTCF + H3K27ac + H3K27me3 | 0.018 / 0.009 / 0.51 | 0.019 / 0.011 / 0.55 | 0.012 / 0.008 / n/d | 0.016 / 0.005 / 0.35 | 0.013 / 0.007 / n/d | 0.006 / 0.002 / n/d |
| CTCF + H3K27ac | 0.019 / 0.009 / 0.51 | 0.021 / 0.011 / 0.51 | 0.013 / 0.007 / n/d | 0.017 / 0.006 / 0.33 | 0.014 / 0.008 / 0.56 | 0.007 / 0.003 / n/d |
| CTCF only | 0.044 / 0.017 / 0.39 | 0.048 / 0.018 / 0.38 | 0.026 / 0.010 / 0.39 | 0.044 / 0.007 / 0.15 | 0.039 / 0.015 / 0.39 | 0.022 / 0.008 / 0.37 |
| sequence and motifs only | 0.045 / 0.015 / 0.32 | 0.047 / 0.015 / 0.31 | 0.031 / 0.012 / 0.39 | 0.044 / 0.007 / 0.16 | 0.039 / 0.014 / 0.35 | 0.024 / 0.011 / 0.45 |

The single groups CTCF, gene body and repressive cost at most 0.002 (Pearson)
and have no defined fraction. Bootstrap intervals over cells for the Pearson
fraction: minus cohesin 0.85 to 1.06, cum3 0.59 to 0.91, CTCF + H3K27ac 0.29
to 0.68, CTCF only 0.19 to 0.51, sequence only 0.13 to 0.46. Per cell, the
Pearson fraction without CTCF and cohesin and active marks (CTCF only) is
0.55 (GM12878), 0.49 (HepG2), 0.31 (IMR-90), 0.17 (A549), 0.04 (H1) and 0.32
on the held-out K562 (loss 0.038, gain 0.012); with sequence and motifs only
0.43, 0.48, 0.20, 0.44, -0.07 and 0.30 (K562, loss 0.049, gain 0.015). On
HMEC deep the loss is 0.002 or less in every set and Meth changes the score
by -0.005 to +0.004. The loop APA ratio and the boundary F1 are not
reported as fractions: the APA ratio changes sign with the set (mean loss
-0.043 to +0.046) and the boundary F1 is a median of per-region values on a
coarse grid (0, 0.33, 0.67, 1). The model without the channel scores
0.003 higher on average than the methylation model run without Meth
(mean difference over 91 cell-set pairs 0.0025, range -0.012 to +0.023), the
size of the seed noise.

Reading of part A. With the consensus reference in the input, methylation
gives back about 0.9 of what is lost when the cohesin ChIP is removed (loss
0.017 Pearson, 0.021 SCC) and about 0.5 of what is lost when only CTCF,
H3K27ac and H3K27me3 remain; it gives back about 0.3 to 0.4 of the loss when
CTCF alone or no ChIP at all remains (Pearson, SCC, insulation, compartment,
eigenvector agreement). For loop enrichment the fractions are lower (0.65
for cohesin removal, 0.15 for CTCF only), so loop strength is the structure
measure methylation replaces least. The active marks are not replaced where
the loss is measurable only in loop enrichment (RF 0.13). The replacement is
largest in training cells that the model saw with Meth and does not show on
the held-out cells at the noise level except for K562 with CTCF or sequence
only (RF 0.32 and 0.30, one cell), and the gains stay below 0.02 in every
set, so the effect is real but small in absolute terms. A 25 kb methylation
model was not trained: no such model exists and a training with its
evaluation costs about two GPU hours for a table that would show the same
small absolute effects.

**Part B, signal level** (`scripts/meth_track_prediction.py`, CPU; tables
`meth_track_prediction.csv`, `meth_track_residual.csv`,
`meth_track_importance.csv`). Targets: each ChIP track at 10 kb, log1p and
z-scored as the model sees it. Features from the 1 kb Meth track and the
genome: mean, minimum, maximum and standard deviation of the ten 1 kb values
in the bin (the minimum is the methylation dip of promoters, islands and
enhancers), mean of +-50 kb and +-500 kb, standard deviation of the 1 kb
values in +-50 kb, fraction of low-methylated (< 0.3) 1 kb bins in +-50 kb
and in the bin, fraction of high-methylated (> 0.8) bins in +-50 kb (the
partially methylated domain proxies), GC, CpG density and CTCF motif
strength; histogram gradient boosting, 500,000 training rows. Training on
the odd chromosomes only, testing on the even chromosomes of nine cells (the
seven training cells, K562, HMEC). Two schemes: seen_cell (trained on the
odd chromosomes of all seven training cells) and unseen_cell (leave one cell
out, K562 and HMEC never in a training set). The sequence-only model (GC,
CpG, motif) is the baseline. Median over the nine cells, unseen_cell:

| mark | R2 sequence only | R2 + methylation | Pearson sequence only | Pearson + methylation | R2 seen_cell + methylation |
|---|---|---|---|---|---|
| H3K9ac | 0.19 | 0.54 | 0.48 | 0.76 | 0.66 |
| H3K4me2 | 0.22 | 0.53 | 0.48 | 0.76 | 0.60 |
| H3K4me3 | 0.19 | 0.47 | 0.49 | 0.71 | 0.58 |
| H3K79me2 | 0.07 | 0.43 | 0.31 | 0.67 | 0.45 |
| H3K36me3 | 0.00 | 0.40 | 0.25 | 0.68 | 0.53 |
| H3K27ac | 0.03 | 0.27 | 0.33 | 0.65 | 0.46 |
| RAD21 | 0.15 | 0.19 | 0.43 | 0.48 | 0.40 |
| CTCF | 0.15 | 0.18 | 0.42 | 0.52 | 0.35 |
| SMC3 | 0.16 | 0.18 | 0.45 | 0.55 | 0.31 |
| H3K27me3 | 0.12 | 0.20 | 0.39 | 0.52 | 0.43 |
| H3K4me1 | 0.05 | 0.16 | 0.34 | 0.59 | 0.44 |
| H4K20me1 | 0.07 | 0.15 | 0.37 | 0.44 | 0.29 |
| H3K9me3 | 0.02 | 0.01 | 0.30 | 0.33 | 0.24 |

R2 on a new cell is lower than on a seen cell and varies between cells (H1 and
IMR-90 reach negative R2 for H3K27ac and H3K4me1, which reflects cell-specific
track scales and not a loss of rank agreement, since the Pearson correlation
stays at 0.27 to 0.61); the Pearson column is the scale-free measure.
Held-out K562 and HMEC (median of the two): R2 0.19 (H3K27ac), 0.53 (H3K36me3),
0.59 (H3K4me2), 0.49 (H3K9ac), 0.38 (H3K79me2), 0.15 (CTCF), -0.42 (RAD21),
-0.25 (H3K4me3), -0.35 (H3K4me1).

Permutation importance of the feature groups on the unseen fits (median drop
of R2 when the group is shuffled; local = bin mean, minimum, maximum and
standard deviation; context = +-50 kb and +-500 kb statistics and low/high
fractions): H3K4me2 local 0.75, context 0.41; H3K4me3 local 0.69, context
0.36; H3K9ac local 0.60, context 0.43; H3K36me3 local 0.54, context 0.15;
H3K27ac local 0.46, context 0.33; H3K79me2 context 0.43, local 0.17; H3K27me3
local 0.31, sequence 0.29, context 0.11; CTCF sequence 0.24, context 0.23,
local 0.14; RAD21 context 0.28, sequence 0.21, local 0.13; SMC3 context 0.38,
sequence 0.19; H3K9me3 sequence 0.13, local 0.05, context -0.02.

Residual: the out-of-fold residual of the unseen fit regressed on the other
12 tracks of the same cell (ridge, five folds over the even chromosomes)
has R2 of 0.30 (H3K9me3), 0.30 (H3K27me3), 0.37 (H3K9ac), 0.39 (H3K4me3 and
H3K36me3), 0.46 to 0.55 for CTCF, RAD21, SMC3, H3K27ac, H3K4me1, H3K79me2,
H3K4me2 and H4K20me1; adding the other tracks raises the median R2 of the
tracks that methylation predicts worst (CTCF 0.18 to 0.62, RAD21 0.19 to
0.64, H3K4me1 0.16 to 0.66, H4K20me1 0.15 to 0.61, H3K27me3 0.20 to 0.53),
whereas H3K9me3 stays at 0.27.

Expectations tested. (1) Promoter and enhancer methylation dips track
H3K27ac, H3K4me3, H3K4me2 and H3K9ac: confirmed, Pearson 0.65 to 0.76 and
R2 0.27 to 0.54, the bin minimum and mean being the dominant features.
(2) Partially methylated domains track H3K9me3 and H3K27me3: not confirmed
for H3K9me3 (R2 0.01, Pearson 0.33, the context group carries no
importance) and weak for H3K27me3 (R2 0.20, Pearson 0.52, the local group
and sequence dominate). (3) Gene-body methylation tracks H3K36me3 and
H3K79me2: confirmed (R2 0.40 and 0.43, Pearson 0.68 and 0.67; H3K36me3
depends on the local group, H3K79me2 on the +-50 kb to +-500 kb context).
(4) CTCF and cohesin are recoverable only through methylation dips at motifs:
partly; the methylation model improves Pearson from 0.42 to 0.52 (CTCF), 0.43
to 0.48 (RAD21) and 0.45 to 0.55 (SMC3), with R2 gains of 0.04 or less
beyond the sequence baseline, so the cohesin and CTCF ChIP signal is not
recovered from methylation at the signal level. At the Hi-C level part A
shows nevertheless that Meth returns most of the loss from removing cohesin
(RF 0.93); the information is about where open chromatin (dips and
context) lies and not about the cohesin signal itself.

**Part C, dedicated 10 kb trainings with Meth** (PENDING). Methylation
coverage was extended to the cells with hg19 RRBS only: ENCODE RRBS of
HCT116 (ENCSR000DFS), MCF-7 (ENCSR939RXT) and HFF (ENCSR000DFG, newborn
foreskin fibroblast, a different sample of the tissue HFFc6 comes from),
replicates 1 and 2 summed, lifted to hg38 CpG by CpG with the UCSC
hg19ToHg38 chain (2.27 to 2.44 million CpG records per cell, 99.99 percent
lifted; lifted HFF values correlate with IMR-90 WGBS at 0.82 on the 4.6
percent of 1 kb bins the RRBS covers, HCT116 0.66, MCF-7 0.65), binned as
the WGBS cells. RRBS covers CpG-dense regions only (4 to 5 percent of the 1 kb
bins reach 5 reads; the other bins hold the cell mean), so the channel of
these three cells is sparser than that of the WGBS cells. With them 10 of
the 12 training cells carry Meth (7 before); NHEK and PC-9 have no ENCODE
methylation data. The v2 cache (mc_all12refres_drop_meth_v2_10kb) and the
runs min_meth_only (Meth + motifs and sequence), min_ctcf_meth,
min_ctcf_h3k27ac_meth (10 kb, residual on the 12-cell consensus, track
dropout 0.5 over the kept tracks, honest validation) and
all12refres_drop_meth_v2_10kb are queued after the structure-loss jobs and
before the cohesin-unknown block (about 7 GPU hours). They are scored
(`scripts/meth_replace.py --stage dedicated`, GM12878, IMR-90, K562, HMEC
deep, with the structure measures) next to min_ctcf_h3k27ac_refres_10kb,
min_ctcf_rad21_refres_10kb, min_one_per_group_refres_10kb and
min_motifs_only_refres_10kb. The new runs use track dropout 0.5 and the
older ones none; the comparison therefore also contains that difference.

Results of part C: pending.

