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

### 29c. Controls: noise ceiling, non-AML cells with the patient marks, marks at loop anchors

Question: do histone marks and CTCF/cohesin behave differently in AML, or
is the patient-specific deviation just noisy, or is it the information in
three marks? `scripts/aml_primary_noise.py`, `scripts/aml_primary_controls.py`;
tables `noise_ceiling.csv`, `noise_ceiling_patients.csv`,
`control_nonaml_cells.csv`, `control_nonaml_summary.csv`,
`control_loops_marks.csv` in `results/aml_primary/`.

Control 1, noise ceiling. Each patient's pairs file (12 with CUT&Tag) was
split into two halves by the parity of the last character of the read name
(a deterministic function of the read name; headers to both), each half
built like the full map (5 kb cooler, 10 and 25 kb, ICE, oe_proper; half
depth 0.05 to 0.2 B contacts). On the patient-specific regions of section
29 (top 20% of the deviation score against both consensus12 and the HSPC
consensus, noise filters, SV/CNV excluded; 38 to 46 regions per patient) the
deviation of half 1 from the reference is correlated with that of half 2
(medians over patients; Spearman-Brown r_full = 2 r_half/(1 + r_half); a
predictor that knows the true signal correlates at most at sqrt(r_full)):

| deviation from | res | r_half | r_full | ceiling of a predictor | model (all12refres_drop) | other-patient mean |
|---|---|---|---|---|---|---|
| consensus12 | 25 kb | 0.810 | 0.895 | 0.946 | 0.379 | 0.732 |
| consensus12 | 10 kb | 0.596 | 0.746 | 0.864 | 0.270 | 0.648 |
| HSPC consensus | 25 kb | 0.674 | 0.805 | 0.897 | 0.251 | 0.483 |
| HSPC consensus | 10 kb | 0.419 | 0.590 | 0.768 | 0.227 | 0.433 |

On each patient's hardest 20% against consensus12 alone r_full is 0.910 (25
kb) and 0.776 (10 kb). Per patient r_full against HSPC at 25 kb ranges from
0.67 (546) to 0.91 (168). The ceiling is therefore far above the 0.25 to 0.38
of the model; data depth does not explain the shortfall. The halves share the
library, so the reliability also contains any library-specific systematic
component (restriction-fragment and GC biases), and the ceiling is an upper
bound for the biological part.

Control 2, the same task on non-AML cells with only the patient marks.
all12refres_drop (10 and 25 kb) predicts the 413 test regions of K562, HMEC
deep and HUVEC deep (held out; reference consensus12) and GM12878, IMR-90,
HepG2, A549 and H1 (training cells; leave-one-out consensus) from CTCF +
H3K27ac + H3K27me3 (p3) and H3K27ac + H3K27me3 (p2, the patients without
CTCF), the tracks quantile-matched as for the patients, and from all tracks
of the cell. Residual correlation corr(P - C, T - C) on the hardest 20% of
the passing regions by the deviation score from the reference (hard-split
filters; no HSPC analogue exists, so only the consensus-based definition is
used, and the patients are compared on that definition: all12refres_drop,
patients' hardest 20% against consensus12):

| set | res | held-out cells (3) | training cells (5) | patients (12) |
|---|---|---|---|---|
| CTCF, H3K27ac, H3K27me3 | 25 kb | 0.671 | 0.622 | 0.428 |
| CTCF, H3K27ac, H3K27me3 | 10 kb | 0.688 | 0.651 | 0.308 |
| H3K27ac, H3K27me3 | 25 kb | 0.693 | 0.597 | |
| H3K27ac, H3K27me3 | 10 kb | 0.635 | 0.599 | |
| all tracks of the cell | 25 kb | 0.641 | 0.762 | |
| all tracks of the cell | 10 kb | 0.585 | 0.766 | |

(per cell in `control_nonaml_cells.csv`; the dedicated three-mark model
reaches 0.486 / 0.467 on the patients.) Non-AML cells recover 0.6 to 0.7 of
their deviation from three marks, 1.5 to 2 times the patients, and the held-out
cells gain nothing from the remaining tracks (0.64 / 0.59 with all tracks
against 0.67 / 0.69 with three), as expected for a model with a reference
map. The median deviation score of these hardest regions is 0.36 against 0.66
(25 kb) and 0.87 (10 kb) for the patients: the patients deviate more from
consensus12, and on the patient-specific regions the HSPC consensus alone
reproduces 0.55 to 0.65 of the deviation from consensus12 (section 29),
i.e. the deviation is mostly shared by healthy cells of the same lab and
protocol.

Control 3, marks at the AML-specific loop anchors (Suppl. Table 1 loops of
the 10 patients with CUT&Tag, 1479 loops, upper against lower tercile (498 each) of the
anchor signal within patients; AUROC):

| mark | measured gain over HSPC separates high from low anchor signal | predicted gain (all12refres_drop) | anchor signal separates high from low measured gain |
|---|---|---|---|
| H3K27ac | 0.413 | 0.543 | 0.415 |
| H3K27me3 | 0.572 | 0.513 | 0.575 |
| CTCF | 0.536 | 0.575 | 0.523 |

At the loops Xu et al. call AML-specific, the patient's own anchor H3K27ac,
H3K27me3 and CTCF are not related to the measured gain (0.41 to 0.58; H3K27ac
is on the wrong side of 0.5); the model's predicted gain moves with H3K27ac
and CTCF at 0.54 to 0.58. The marks are not informative about the loop gain
in these data, and the model uses them only weakly.

Reading. The three numbers side by side at 25 kb: patient-specific
deviation from HSPC is reproducible to 0.81 (ceiling of a predictor 0.90),
non-AML cells with the same three marks recover 0.62 to 0.67, patients
recover 0.25 (HSPC definition) to 0.43 (consensus12 definition). Neither
data depth nor the information in three marks plus a reference limits the
patients: it is the part of the patient maps that lab, protocol and lineage
share (reproduced by the HSPC and the other patients' maps at 0.5 to 0.8) and
a loop gain that does not follow the patient's own marks. The behaviour of
the marks learned on the 12 cells does not carry over to the Xu et al.
patients at the AML-specific loops, and the evidence cannot separate
domain shift of the CUT&Tag p-value tracks from a real difference of AML
chromatin, since no patient has a replicate CUT&Tag.

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

## 35. How far DNA methylation replaces ChIP-seq signal (FINAL)

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

**Part C, dedicated 10 kb trainings with Meth.** Methylation
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
before the cohesin-unknown block (7 GPU hours, 2026-10-04). They are scored
(`scripts/meth_replace.py --stage dedicated`, GM12878, IMR-90, K562, HMEC
deep, with the structure measures) next to min_ctcf_h3k27ac_refres_10kb,
min_ctcf_rad21_refres_10kb, min_one_per_group_refres_10kb and
min_motifs_only_refres_10kb. The new runs use track dropout 0.5 and the
older ones none; the comparison therefore also contains that difference.

Results of part C (`results/methylation/meth_dedicated.csv`; median over 413
regions, mean of GM12878, IMR-90, K562 and HMEC deep; K562 and HMEC are
held out):

| run | inputs | Pearson mean | GM12878 | IMR-90 | K562 | HMEC deep | SCC mean | insulation | loop enrichment | compartment |
|---|---|---|---|---|---|---|---|---|---|---|
| min_motifs_only | sequence and motifs | 0.879 | 0.854 | 0.902 | 0.858 | 0.902 | 0.864 | 0.938 | 0.703 | 0.906 |
| min_meth_only (dropout) | Meth | 0.898 | 0.898 | 0.921 | 0.876 | 0.898 | 0.880 | 0.952 | 0.709 | 0.924 |
| min_ctcf_rad21 | CTCF, RAD21 | 0.886 | 0.880 | 0.924 | 0.858 | 0.881 | 0.871 | 0.944 | 0.704 | 0.912 |
| min_ctcf_meth (dropout) | CTCF, Meth | 0.901 | 0.901 | 0.924 | 0.879 | 0.900 | 0.886 | 0.956 | 0.713 | 0.930 |
| min_ctcf_h3k27ac | CTCF, H3K27ac | 0.909 | 0.924 | 0.940 | 0.869 | 0.902 | 0.892 | 0.959 | 0.734 | 0.936 |
| min_ctcf_h3k27ac_meth (dropout) | CTCF, H3K27ac, Meth | 0.917 | 0.923 | 0.941 | 0.893 | 0.913 | 0.902 | 0.964 | 0.740 | 0.943 |
| min_one_per_group | CTCF, RAD21, H3K27ac, H3K4me3, H3K27me3 | 0.908 | 0.935 | 0.947 | 0.875 | 0.876 | 0.896 | 0.958 | 0.737 | 0.931 |

Meth alone lifts the sequence and motif model by 0.019 Pearson (0.021 SCC,
0.014 insulation, 0.006 loop enrichment, 0.018 compartment) and is better
than CTCF + RAD21 ChIP on the mean of the four cells (0.898 against 0.886,
better in 3 of 4 cells; K562 0.876 against 0.858, HMEC 0.898 against 0.881).
CTCF + Meth is better than CTCF + RAD21 by 0.015. Adding Meth to CTCF +
H3K27ac raises the mean by 0.008 (K562 +0.024, HMEC +0.011, GM12878 -0.001,
IMR-90 +0.001) and makes the three-input model better than the five-track
set on the mean (0.917 against 0.908; the five tracks win on GM12878 0.935
against 0.923 and IMR-90 0.947 against 0.941 and lose on K562 and HMEC, where
the mean of the two held-out cells is 0.903 against 0.876). Differences
under 0.013 (seed noise of the median) are the GM12878 and IMR-90 entries;
the held-out gains of the Meth models are 0.011 to 0.024 and the loop
enrichment gains are within noise. The new runs were trained with track
dropout 0.5 and the older ones without, so the comparison contains that
difference; the 0.013 noise level was measured on runs of one recipe.

The refreshed model all12refres_drop_meth_v2_10kb (10 of 12 cells with Meth,
trained 07:11 to 08:31) scores like v1 with all 13 ChIP tracks (Pearson 0.900
to 0.953 over ten cells; K562 0.905, HMEC 0.905, GM12878 0.932) and gives
the same replacement fractions on the same cells: pooled over eight cells
with Meth (the five WGBS cells and the three RRBS cells HCT116, MCF-7, HFF),
Pearson fractions 0.93 (cohesin removed), 0.38 (CTCF + H3K27ac), 0.28
(CTCF only), 0.22 (sequence only), lower than for the seven WGBS cells
(0.39 and 0.32) because the sparse RRBS channel adds nothing: Meth changes
the CTCF-only score by -0.003 (HCT116), -0.007 (MCF-7) and -0.001 (HFF).
Negative result: lifted RRBS does not stand in for ChIP signal at 10 kb.

Overall statement. With the consensus map in the input, whole-genome
bisulfite methylation recovers about 0.9 of the Hi-C accuracy lost when the
cohesin ChIP is missing, about 0.4 to 0.5 of the loss when only CTCF with or
without H3K27ac and H3K27me3 remain, and 0.3 to 0.4 when no ChIP remains,
for Pearson, SCC and the compartment agreement; loop enrichment is
recovered least (0.15 to 0.65). Methylation stands in at the signal level
for the active-promoter and gene-body marks (H3K4me2, H3K4me3, H3K9ac,
H3K36me3, H3K79me2, H3K27ac; Pearson 0.65 to 0.76), not for CTCF and cohesin
signal (Pearson 0.48 to 0.55, R2 0.18 to 0.19), H3K9me3 or H3K4me1 beyond a
Pearson of 0.33 to 0.59. The practical result is that Meth with two ChIP
marks (CTCF, H3K27ac) outperforms five ChIP marks on held-out cells at 10 kb.

## 36. AML-inclusive training (FINAL for fold 0; folds 1 and 2 not run)

Question: does a model that learns from AML Hi-C reproduce what is specific
to a patient, which the models trained on the 12 non-AML cells do not
(section 29: other-patient mean 0.906 against 0.82 to 0.89 for the models
at 25 kb; AML-specific loop gain beyond the consensus about zero), and is
what a patient map deviates by mostly the lab, protocol and lineage
component that healthy CD34+ cells of the same lab share (section 29c:
noise-ceiling reliability of the patient deviation 0.81 at 25 kb, 0.59 at
10 kb; non-AML cells recover 0.67 against 0.43 of it from the same three
marks)?

Design (`scripts/aml_train.py`, `scripts/aml_train_eval.py`; runs
all12aml_refin_drop and all12aml_honest_drop at 25 and 10 kb, input reference
with track dropout 0.5 and the no-reference counterpart, honest validation).
Training cells: the 12 non-AML cells, MOLM-13 (all its public tracks), the
Xu et al. 2022 patients of the training folds and two of the four CD34+
HSPC donors of the same study. Patient split (`results/aml_train/split.json`):
the 12 patients with CUT&Tag form three folds of four; stratified by CTCF
CUT&Tag (9 of 12) and by the presence of a copy-number segment called from
Hi-C coverage, ordered within a stratum by depth; fold 0 (test 027, 270,
773, 1916; train 018, 103, 168, 472, 546, 629, 673, 1360) is run first, folds
1 (103, 168, 472, 673) and 2 (018, 546, 629, 1360) wait for the flag file
runs/multicell_reference/AML_CV. Subtype labels are not assigned to patients
in the paper's tables, so no stratification by subtype is possible.
U-937 WT is a further held-out AML line; HSPC donors 341 and 822 are held
out, 050 and 213 train. Inputs: the marks each sample has (patients H3K27ac
and H3K27me3, CTCF for 9; quantile-matched p-value tracks), the others absent
under track dropout. HSPC donors have Hi-C only; their marks are one public
CD34+ ChIP-Atlas experiment per mark (H3K27ac SRX1761289, H3K27me3
SRX1761288, CTCF SRX037935, `aml/picks_hspc_cd34.tsv`), quantile-matched like
the patients' tracks; the ChIP donor is not a Hi-C donor of the study.
Reference: the AML reference of a sample is the mean proper O/E of the
training patients (leave-one-out for a training patient, all eight for
MOLM-13 and for the test patients and U-937); the reference of an HSPC donor
is the mean of the training donors (the other donor for a training donor);
the non-AML cells keep consensus_loo. The cell list in each reference json
(`amlxu<patient>`, `amlxuhspc<donor>`) is checked by the leakage guard, so a
test patient or donor never is in its own reference. AML windows (patients,
HSPC donors, MOLM-13) enter every epoch three times (`train.oversample_*`);
the 413 Akita test regions stay out of every cell's training windows. Lab
indicator (`data.lab_classes: [xu]`, `samples[].lab`): one constant input row,
1 for the Xu et al. Hi-C (patients and HSPC), 0 for everything else, so the
healthy HSPC separate the lab and protocol component from the AML component;
the ablation all12aml_refin_drop_nolab (25 kb) has the same data without the
row. Validation: honest (no training window of any cell at a validation
position) and checkpoint selection on patient-specific hard-validation
windows (25 per training patient where the patient deviates most from both
consensus12 and the HSPC consensus, the deviation score of `hard_split.py`,
both scores in the top 20% of the windows that pass the mask and copy-number
filters), held out from training for every sample (200 windows).

Evaluation on the 413 Akita test regions of the test patients, the held-out
HSPC donors and U-937: Pearson, SCC and structure measures; baselines on the
same pixels (AML reference, mean of the other eleven patients, HSPC
consensus, consensus12); residual correlation corr(P - C, T - C) against the
AML reference, consensus12 and the HSPC-matched baseline, on all regions and
on the patient-specific regions of section 29 (explain_learn_regions); at
10 kb Xu et al.'s AML-specific loops and stripes (Suppl. Tables 1 and 4);
the gain of the patient's own ChIP beyond the reference from the pair with
and without ChIP; the same for the models trained without AML data.

Incident. At 2026-10-04 15:55 a scheduler restart with an incomplete list of
running jobs started a second all12aml_refin_drop_25kb training and two
duplicate evaluations next to the originals (started 15:15, 14:58 and
15:05); the duplicates were stopped within two minutes by PID. The
duplicate truncated the original's log file (the first 40 minutes of its
log are lost; the training itself was not affected: same configuration,
same output folder, no checkpoint written by the duplicate). The scheduler
now writes a pid file per started job and finds running jobs on restart
(`.run_<job>.pid`).

Results, fold 0, 25 kb (`results/aml_train/eval_f0_*.csv`).

Training. all12aml_refin_drop_25kb stopped at epoch 33 (best hard-validation
Pearson 0.733; honest-validation Pearson 0.916 at epoch 14), all12aml_honest_
drop_25kb at epoch 12 (best hard-validation 0.408, reached at epoch 2: the
patient-specific windows select an early checkpoint of the model without a
reference). Normal accuracy on the 413 Akita test regions (median over the
cells; section 15 evaluation): all12aml_refin_drop 0.929 on the 12 training
cells and 0.906 on the five held-out cells, against 0.936 and 0.914 for
all12refin_drop (-0.007 and -0.008, about half the seed noise); all12aml_honest_
drop 0.781 and 0.736 against 0.809 and 0.775 for all12honest_drop (-0.028
and -0.039, the early checkpoint).

Held-out patients (027, 270, 773, 1916; median over patients of the per-
patient median Pearson on the 413 regions, identical pixels):

| source | with the patient's marks | no ChIP |
|---|---|---|
| AML reference alone (mean of the 8 training patients) | 0.912 | |
| mean of the 11 other patients | 0.922 | |
| HSPC consensus (4 donors) / consensus12 | 0.887 / 0.844 | |
| all12aml_refin_drop (AML reference) | 0.900 | 0.898 |
| all12aml_refin_drop (HSPC reference) | 0.873 | 0.865 |
| all12refin_drop (AML reference, no AML data in training) | 0.886 | 0.896 |
| all12refin_drop (consensus12 reference) | 0.842 | 0.816 |
| all12aml_honest_drop (no reference) | 0.623 | 0.456 |
| all12honest_drop | 0.607 | 0.469 |

No test patient is above the AML reference (model against reference: 027
0.917 / 0.930, 1916 0.907 / 0.926, 270 0.876 / 0.881, 773 0.893 / 0.897), and
the mean of all eleven other patients, the section 29 baseline, stays 0.022
above the model. Training on AML data raises the input-reference model by
0.014 with marks (0.886 to 0.900) and 0.002 without ChIP (0.896 to 0.898),
and the reference-free model by 0.016 and -0.013 (0.607 to 0.623, 0.469 to
0.456). The patient's CUT&Tag adds 0.002 to the AML-trained model (0.900 and
0.898), 0.008 with the HSPC reference, and -0.010 to the model trained
without AML data (0.886 against 0.896). On the patient-specific regions of
section 29 (30 to 43 per patient) the AML-trained model with marks reaches
0.755 against 0.751 for the AML reference and 0.730 without ChIP: the
ChIP adds 0.025 there, and the model does not exceed its reference. Residual
correlation corr(P - AML reference, T - AML reference): 0.167 with marks and
0.023 without ChIP on all regions, 0.274 and 0.104 on the patient-specific
regions (non-AML-trained model with the AML reference: 0.152 / 0.049 and 0.218 /
0.074). Against the HSPC consensus the AML-trained model has residual
correlation 0.464 (marks) and 0.434 (no ChIP), the non-AML-trained model with
the AML reference 0.423 and 0.350, with consensus12 as reference 0.345 and
0.202; against consensus12 the AML-trained model reaches 0.711, mostly
because its reference already differs from consensus12 by the AML component.

Held-out healthy HSPC donors (341, 822; Pearson, reference the training-
donor mean): all12aml_refin_drop 0.926 with the CD34+ marks and 0.929 without,
all12refin_drop with the HSPC reference 0.907 and 0.919, consensus12 as
reference 0.845 and 0.820; the reference alone scores 0.938 and the mean of
the other three donors 0.949, so the model does not exceed its reference for
healthy cells either, and AML training changes the HSPC score by +0.019 with
marks and +0.010 without. U-937 WT (other lab and protocol, Micro-C):
all12aml_refin_drop 0.857 with marks and 0.863 without, the AML reference
alone 0.834, consensus12 0.866, all12refin_drop with consensus12 0.886 and
0.850: the AML reference lifts the model over the reference but the
non-AML consensus reference remains better on marks (0.886), as expected for
a cell from another lab.

Xu et al.'s AML-specific loops and stripes (10 kb, four test patients, 592
loops and 481 stripes), models trained without AML data (the AML models
follow below): the measured loop enrichment lies 0.31 log2 units
above the AML reference at the loop pixel (stripes 0.21), the predictions
0.01 to 0.02 above it for the input-reference models (stripes 0.03 to 0.07;
the reference-free model lies below it) and correlate with the measured gain
at 0.10 to 0.24 (loops) and 0.25 to 0.42 (stripes).

Results, fold 0, 10 kb and the lab-indicator ablation.

Training. all12aml_refin_drop_10kb stopped at epoch 34 (best hard-
validation 0.524), all12aml_honest_drop_10kb at epoch 12 (0.217). Normal
accuracy, 12 training cells / five held-out cells: 0.923 / 0.889 for
all12aml_refin_drop_10kb against 0.927 / 0.903 for all12refin_drop_10kb;
0.760 / 0.720 for the AML no-reference model against 0.819 / 0.761. The
25 kb ablation without the lab row (all12aml_refin_drop_nolab_25kb, same data)
stopped at epoch 36 (hard-validation 0.735) with normal accuracy 0.928 /
0.915.

Held-out patients, 10 kb (median over the four patients; Pearson, 413 regions):
the AML reference alone 0.802, the mean of the 11 other patients 0.814, HSPC
consensus 0.754, consensus12 0.717. Models, with the patient's marks / without
ChIP: all12aml_refin_drop 0.818 / 0.805; with the HSPC reference 0.772 /
0.759; all12refin_drop with the AML reference 0.801 / 0.803, with consensus12
0.736 / 0.708; all12aml_honest_drop 0.525 / 0.366; all12honest_drop 0.510 /
0.369. All four test patients are above the AML reference (027 0.845 against
0.831, 1916 0.855 / 0.851, 270 0.791 / 0.773, 773 0.771 / 0.754) and two are
above the mean of the other eleven patients (270, 773). On the patient-
specific regions the AML-trained model reaches 0.592 with marks and 0.560
without ChIP against 0.541 for the reference (all12refin_drop with the AML
reference: 0.585 / 0.574). Residual correlation with the AML reference
(all regions / patient-specific): 0.234 / 0.268 with marks, 0.164 / 0.223
without ChIP (all12refin_drop with the AML reference 0.217 / 0.258 and 0.162 /
0.156); against the HSPC consensus 0.446 with marks and 0.422 without (0.437
and 0.410 for the model without AML data; with consensus12 as reference
0.303 and 0.236); against consensus12 0.616. Healthy HSPC donors 341 and 822:
all12aml_refin_drop 0.827 / 0.828 against 0.778 for the training-donor mean
it reads and 0.808 for the mean of the other three donors (all12refin_drop
with that reference 0.799 / 0.814). U-937: 0.852 / 0.826 against 0.781 for the
AML reference and 0.818 for consensus12 (all12refin_drop with consensus12 0.856 /
0.815).

Xu et al.'s AML-specific loops and stripes (10 kb, test patients, 592 loops,
481 stripes): the measured loop enrichment lies 0.31 log2 units above the AML
reference at the loop pixel (stripes 0.21); the AML-trained model with marks
predicts -0.02 (stripes -0.03), 41% of the loops (37% of the stripes) above
the reference and correlates with the measured gain at 0.14 (stripes 0.34);
without ChIP -0.03 (-0.08), 29% (6%), 0.16 (-0.07). The model without AML data
and the AML reference: +0.01 (+0.03), 57% (57%), 0.10 (0.34). The loop gain is
not reproduced by either model.

Lab indicator (25 kb, with marks / without ChIP): all12aml_refin_drop_nolab
0.904 / 0.895 against 0.900 / 0.898 with the row; patient-specific 0.768 /
0.744 against 0.755 / 0.730; residual correlation with the AML reference
0.231 / -0.010 against 0.167 / 0.023, with the HSPC consensus 0.488 / 0.420
against 0.464 / 0.434. The row does not help; every difference is within
the 0.013 noise of the median Pearson or in the direction of the model
without it.

Reading. Training on AML data and HSPC does not make the 25 kb model exceed
the AML reference or the other-patient mean, and the patient's own ChIP adds
0.002 there. At 10 kb the model exceeds the AML reference in all four test
patients (+0.016 median) and the mean of the other patients in two, the
patient-specific regions gain 0.051 over the reference with marks (0.019
without ChIP), and the ChIP, neutral for the model without AML data (0.801
with, 0.803 without), now adds 0.013 Pearson overall and 0.032 on the
patient-specific regions: the first sign that the ChIP carries patient
information once the model has seen AML maps. The gain is modest (about the
size of the seed noise overall), it is not reproduced at the loops and
stripes of the paper, healthy donors show the same pattern (model above its
reference by 0.049 at 10 kb, below at 25 kb), and the lab and protocol row
changes nothing, so the improvement is not a lab effect that the row could
absorb. The three-fold cross-validation (flag file AML_CV) was not run.

### 36b. Eight AML-specific sites against GM12878, measured and predicted

`scripts/aml_sites_figures.py` (stages select, predict, figures); figures
and per-locus npz in `/mnt/processed/wolffjoa/hic-multiome/runs/aml_sites/figures/`
(copied to `results/aml_primary/figures/`; the repository ignores `*.png`,
so the copies are untracked), numbers in `results/aml_primary/sites_loci.csv`
and `sites_sv_locus.csv`, candidate table in `runs/aml_sites/candidates.csv`.
Loci come from Xu et al.'s Supplementary Tables 1 and 4 (10 kb) for the 10
patients with CUT&Tag (2148 loops, 484 stripes with intra-chromosomal
anchors); a locus qualifies when the patient's strength (log2 donut
enrichment at the loop pixel; mean log O/E in the stripe band) exceeds that
of the HSPC consensus and of measured GM12878 by at least 0.8, both half-depth
maps of the patient (section 29c) exceed the HSPC level by at least 0.4, and
no anchor lies within 1 Mb of an SV breakpoint or on a copy-number segment:
77 of 2632 qualify. Types by the patient's own CUT&Tag at the anchors
(promoter-enhancer: H3K27ac at both anchors; promoter-silencer: H3K27me3 at
one; stripes by the 40 kb anchor at the 75th percentile of the training
distribution). Genes are the refGene TSS inside an anchor. Of the named loci
IKZF2 (6 loops), RTTN (4) and RUNX1 (1) exist in the tables but none passes
the criteria (gains of 0.0 to 0.6), KLF4 (patient 629) passes with weaker
evidence on GM12878 (gain 0.65); MYCN, MEIS1, ERG and WT1 do not occur among
the loops of these patients. The enhancer stripe passes only a looser tier
(gains of at least 0.5; marked in the table). Panels per locus: GM12878
measured, HSPC consensus, patient measured, patient predicted by
all12aml_refin_drop_10kb (the patient's marks, lab indicator xu, AML
reference of fold 0), by all12refres_drop_10kb (no AML data, same reference
map), the AML reference both read, GM12878 predicted by the AML-trained
model from its ChIP (a training cell: its windows may be in training), and
the patient minus GM12878 difference, measured and predicted; 80 bins per
panel (800 kb), one colour scale per locus from the 98th percentile of
|patient measured|, r (upper triangle, offset >= 2 bins) under each panel,
CUT&Tag and GM12878 track strips, a per-distance Pearson strip. The Meth
variant (all12aml_meth_refin_drop_10kb) had no checkpoint when the figures
were made, so its column is absent.

| locus | patient | held out | measured gain over HSPC | predicted gain, AML-trained | predicted gain, no AML data | AML-trained minus reference | r patient AML-trained / no AML data / reference |
|---|---|---|---|---|---|---|---|
| P-E loop CIP2A | 773 | yes | 1.03 | 0.56 | 0.58 | 0.11 | 0.88 / 0.86 / 0.87 |
| P-E loop CLEC4O | 773 | yes | 0.80 | 0.02 | -0.05 | 0.02 | 0.79 / 0.79 / 0.77 |
| P-E loop CLECL1 | 168 | no | 1.21 | 0.29 | 0.34 | 0.05 | 0.76 / 0.80 / 0.76 |
| P-S loop KLF4 | 629 | no | 0.90 | 0.35 | 0.38 | 0.09 | 0.94 / 0.92 / 0.89 |
| P-S loop LNX2 | 546 | no | 1.02 | 0.17 | 0.16 | 0.00 | 0.91 / 0.90 / 0.89 |
| repressive stripe chr13 31.44 Mb | 270 | yes | 0.89 | 0.17 | 0.19 | 0.10 | 0.87 / 0.87 / 0.86 |
| enhancer stripe chr11 75.88 Mb (looser tier) | 270 | yes | 0.61 | 0.09 | 0.07 | -0.04 | 0.85 / 0.84 / 0.84 |
| SV, inv(16) HSF4 | 629 | no | neo-loop enrichment -0.16 | -0.24 (no AML data) | | | derivative allele r 0.61, junction block -0.06 |

Reading. All seven sites are absent from GM12878 and HSPC in the measured
maps (strength -0.34 to 0.83 against 0.68 to 1.44 in the patient) and the
patient-minus-GM12878 difference is predicted at r 0.25 to 0.74. The
predicted gain is 0.02 to 0.58 against a measured 0.61 to 1.21: median
fraction 0.19 (AML-trained) and 0.21 (no AML data); the correlation of
predicted with measured gain over the seven sites is 0.55 and 0.61. Most of
the predicted gain is already in the AML reference both models read (the
AML-trained model adds 0.00 to 0.11 beyond it, the model without AML data
adds -0.06 to 0.12): sites are partly predicted when the other patients
share them (CIP2A, KLF4, CLECL1: gains 0.3 to 0.6, loops with H3K27ac at
both anchors or H3K27me3 at the silencer anchor) and not when only the
patient has them (CLEC4O, the enhancer stripe, stripes in general: 0.1 to
0.2). Training the model on AML Hi-C makes no difference at these sites.
The SV locus fails as expected: the derivative-allele prediction reproduces
the cis blocks on both sides of the inversion (r 0.61 over the window) but
not the junction block (r -0.06) and the HSF4 enhancer-promoter contact
(predicted -0.24, measured -0.16, log2 donut enrichment over the 60 listed
neo-loop pairs; the AML-trained model was not run on derivative alleles).
Three of the seven sites are in held-out patients (773, 270) and four in
training patients (168, 629, 546); GM12878 predictions come from a training
cell.

#### 36b-v2. The AML+Meth model as a further predicted column (2026-10-10)

`python scripts/aml_sites_figures.py --stage predict --v2` (inference of all12aml_meth_refin_drop_10kb
of section 38 on the 80-bin windows of the 5 site patients and GM12878; run on the remote RTX 4090,
under 1 GB of GPU memory) and `--stage figures --v2` (figures in
`/mnt/processed/wolffjoa/hic-multiome/runs/aml_sites/figures_v2/`, same file names as the published
ones; the published figures are unchanged; numbers in `results/aml_primary/sites_loci_v2.csv`,
`sites_sv_locus_v2.csv`, `sites_gain_corr_v2.csv`). The old columns reproduce the published values
exactly (maximum difference 0.0). All five site patients (773, 168, 629, 546, 270) have the Xu WGBS Meth
bigWig, so the Meth channel is present for every site; a control with Meth withheld (same model, the
channel absent as in training under track dropout; column gain_methabs) separates the channel from
retraining. GM12878 is predicted with its own Meth. The SV locus is not rerun (no derivative-allele
prediction with this model).

Gain over HSPC (log2 donut enrichment, or mean log O/E for stripes):

| site | measured | AML-trained | no AML | AML+Meth | AML+Meth, Meth withheld |
|---|---|---|---|---|---|
| CIP2A (773, held-out) | 1.03 | 0.56 | 0.58 | 0.44 | 0.40 |
| CLEC4O (773, held-out) | 0.80 | 0.02 | -0.05 | -0.01 | -0.02 |
| CLECL1 (168) | 1.21 | 0.29 | 0.34 | 0.42 | 0.27 |
| KLF4 (629) | 0.90 | 0.35 | 0.38 | 0.36 | 0.34 |
| LNX2 (546) | 1.02 | 0.17 | 0.16 | 0.17 | 0.15 |
| stripe chr13 31.44-31.76 Mb (270, held-out) | 0.89 | 0.17 | 0.19 | 0.49 | 0.39 |
| stripe chr11 75.88-76.16 Mb (270, held-out) | 0.61 | 0.09 | 0.07 | 0.10 | 0.07 |
| mean over the 7 | 0.92 | 0.24 | 0.24 | 0.28 | 0.23 |

Pearson r of measured against predicted gain over the 7 sites: AML-trained 0.555, no AML 0.606, AML+Meth
0.578, Meth withheld 0.511 (Spearman 0.71, 0.71, 0.54); RMSE 0.71, 0.70, 0.66. The methylation variant
moves the predicted gain by more than 0.1 at three sites (stripe chr13 +0.32, CLECL1 +0.13, CIP2A -0.12 against
the AML-trained model) and by 0.02 or less at the other four; it leaves the under-prediction of the
gains unchanged (all models predict about a quarter of the measured gain). GM12878 prediction at the
loop or stripe differs by at most 0.12 between the AML-trained and the AML+Meth model. Limits: 7 sites
(r differences of 0.03 to 0.1 are far inside the noise), 4 sites are in the held-out patients 773 and 270,
3 sites are in training patients (168, 629, 546), GM12878 is a training cell, the sites were chosen
for AML specificity and strong gain, the stripe gain is a band mean and not a pixel, and the SV
case (inv(16), HSF4) is expected to fail and is not rerun. The v2 figures were viewed (stripe 6 and
the summary).

## 37. Chromosome-held-out final models (FINAL for refin_drop and honest_drop; refres_drop rescoring running)

Question: how do the headline recipes compare with the published predictors
when no training window of any cell lies on the chromosomes the competitors
held out? The earlier comparisons (section 18, 31) used chromosomes our models
trained on, or the Akita test regions only.

Held-out chromosomes, from the competitors' own splits: C.Origami chr2 (test)
and chr3 (validation) (section 18 and its released IMR-90 model), Epiphany
chr3, chr11, chr17 (its released GM12878 weights), HiC-Reg chr14 and chr17
(the released cross-chromosome predictions, section 18 "Leak-free HiC-Reg");
Akita's test regions were excluded from every run already. Union: chr2,
chr3, chr11, chr14, chr17 (26% of the autosomal genome). Hi-cGAN held out the
even chromosomes; the comparison with it uses chr2 and chr14, the two held
out by both.

Runs (`data.chroms` without the five chromosomes, so neither a training
window nor a normaliser statistic of any cell sees them; separate caches
mc_all12ref_chrho_<k>kb and mc_all12chrho_<k>kb): all12refin_drop_chrho,
all12honest_drop_chrho (the ChIP-only model without the consensus, the fair
comparison with competitors that get none) at 25 and 10 kb, and
all12refres_drop_chrho. Scoring (`scripts/hic_measures.py --stage
chrho_predict chrho_score chrho_report`, `results/hic_measures_chrho/`):
whole-chromosome tiling, identical pixels (pixels finite in the target and in
every source of the block, band cut to the shortest reach), against
C.Origami (IMR-90 chr2, chr3), Epiphany (GM12878 chr3, 11, 17, 990 kb
band), HiC-Reg cross-chromosome WINDOW and MULTICELL and the leaky CV maps
for comparison (GM12878 and K562, chr14, chr17, 1 Mb band), Hi-cGAN
(GM12878 and its K562 predictions, chr2, chr14), the consensus the model reads,
the best other measured map and the replicate map; measures as section 31.

Results, 25 kb models (all12refin_drop_chrho and all12honest_drop_chrho;
`results/hic_measures_chrho/chrho_summary.csv`). The training stopped at epoch
72 (best honest-validation 0.939) and epoch 25 (0.799); normal accuracy on
the 413 Akita regions, 12 training cells / five held-out cells: 0.933 / 0.910
for refin_drop_chrho (0.936 / 0.914 without the held-out chromosomes) and
the ChIP-only honest model is in `reference_summary.csv`. Whole held-out
chromosomes, mean over the chromosomes of a block, identical pixels (Pearson
over all band pixels; median Pearson over 1 Mb tiles; SCC; insulation r;
boundary recall; rank-matched loop recall; HiCRep):

| block | source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | HiCRep |
|---|---|---|---|---|---|---|---|---|
| HiC-Reg, GM12878 chr14, chr17 (1 Mb band) | all12refin_drop_chrho | 0.931 | 0.950 | 0.926 | 0.905 | 0.833 | 0.396 | 0.935 |
| | all12honest_drop_chrho (ChIP only) | 0.782 | 0.798 | 0.763 | 0.674 | 0.608 | 0.191 | 0.792 |
| | consensus of the other cells | 0.868 | 0.901 | 0.853 | 0.804 | 0.805 | 0.396 | 0.857 |
| | HiC-Reg cross-chromosome WINDOW | 0.579 | 0.620 | 0.538 | 0.447 | 0.322 | 0.032 | 0.668 |
| | HiC-Reg cross-chromosome MULTICELL | 0.615 | 0.634 | 0.567 | 0.455 | 0.346 | 0.038 | 0.692 |
| | HiC-Reg CV (published, within-chromosome) | 0.908 | 0.964 | 0.870 | 0.678 | 0.723 | 0.323 | 0.915 |
| HiC-Reg, K562 chr14, chr17 | all12refin_drop_chrho | 0.897 | 0.928 | 0.889 | 0.856 | 0.738 | 0.347 | 0.904 |
| | all12honest_drop_chrho | 0.730 | 0.777 | 0.711 | 0.622 | 0.568 | 0.125 | 0.748 |
| | consensus | 0.857 | 0.905 | 0.853 | 0.839 | 0.769 | 0.332 | 0.868 |
| | HiC-Reg cross-chromosome WINDOW | 0.736 | 0.820 | 0.696 | 0.484 | 0.523 | 0.038 | 0.797 |
| | HiC-Reg cross-chromosome MULTICELL | 0.740 | 0.819 | 0.700 | 0.500 | 0.519 | 0.030 | 0.800 |
| | HiC-Reg CV | 0.888 | 0.950 | 0.864 | 0.670 | 0.688 | 0.178 | 0.923 |
| Hi-cGAN, GM12878 chr2, chr14 (6.4 Mb reach, our 2 Mb band) | all12refin_drop_chrho | 0.934 | 0.949 | 0.929 | 0.901 | 0.808 | 0.405 | 0.933 |
| | all12honest_drop_chrho | 0.796 | 0.824 | 0.778 | 0.698 | 0.616 | 0.200 | 0.773 |
| | consensus | 0.860 | 0.899 | 0.848 | 0.784 | 0.777 | 0.389 | 0.835 |
| | Hi-cGAN (submitted) | 0.221 | 0.485 | 0.225 | 0.238 | 0.218 | 0.048 | 0.322 |
| Hi-cGAN, K562 chr2, chr14 (GM12878 model) | all12refin_drop_chrho | 0.904 | 0.935 | 0.896 | 0.842 | 0.726 | 0.356 | 0.892 |
| | all12honest_drop_chrho | 0.765 | 0.826 | 0.742 | 0.657 | 0.550 | 0.141 | 0.721 |
| | consensus | 0.853 | 0.909 | 0.849 | 0.816 | 0.718 | 0.350 | 0.850 |
| | Hi-cGAN GM12878 model | 0.143 | 0.479 | 0.147 | 0.218 | 0.213 | 0.050 | 0.220 |

On held-out chromosomes the reference-map model exceeds every cross-chromosome
or cross-cell competitor prediction on every listed measure, and the
consensus it reads on all of them except boundary recall on K562 (0.738
against 0.769) and loop recall in the GM12878 HiC-Reg block (tie, 0.396).
The published HiC-Reg CV maps exceed it on the 1 Mb tile median (0.964 and
0.950 against 0.950 and 0.928) and on K562 HiCRep (0.923 against 0.904): they
interpolate within the chromosome (section 18, "HiC-Reg training split") and
are not a held-out score. The ChIP-only model,
which gets no consensus, scores 0.782 and 0.730 Pearson against 0.579 to
0.615 and 0.736 to 0.740 for HiC-Reg's cross-chromosome predictions
(GM12878, K562): ahead by 0.17 to 0.20 on GM12878 and level on K562, where
the HiC-Reg tile median (0.820) exceeds the ChIP-only model's (0.777); its
SCC is 0.763 and 0.711 against 0.538 to 0.567 and 0.696 to 0.700, its
insulation correlation 0.674 and 0.622 against 0.447 to 0.455 and 0.484 to
0.500. Against Hi-cGAN (even chromosomes held out) the ChIP-only model
leads by 0.57 to 0.62 Pearson.

10 kb (all12refin_drop_chrho stopped at epoch 55, best honest-validation
0.906; all12honest_drop_chrho at epoch 29, 0.767; all12refres_drop_chrho at
epoch 85, 0.907; normal accuracy on the 413 regions, 12 training / five
held-out cells: 0.923 / 0.887 for refin_drop_chrho, 0.924 / 0.893 for
refres_drop_chrho, 0.927 / 0.903 for the model that saw the five
chromosomes). Whole held-out chromosomes, mean over the chromosomes of a
block (Pearson over band pixels / tile median / SCC / insulation r / boundary
recall / loop recall):

| block | source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall |
|---|---|---|---|---|---|---|---|
| C.Origami, IMR-90 chr2, chr3 (2 Mb band) | all12refin_drop_chrho | 0.926 | 0.944 | 0.927 | 0.924 | 0.654 | 0.355 |
| | all12honest_drop_chrho | 0.808 | 0.855 | 0.812 | 0.822 | 0.497 | 0.192 |
| | consensus of the other cells | 0.866 | 0.906 | 0.862 | 0.796 | 0.641 | 0.383 |
| | C.Origami | 0.920 | 0.905 | 0.913 | 0.866 | 0.474 | 0.162 |
| | Hi-cGAN IMR-90 model (chr2) | 0.567 | 0.590 | 0.567 | 0.584 | 0.105 | 0.014 |
| Epiphany, GM12878 chr3, 11, 17 (990 kb band) | all12refin_drop_chrho | 0.914 | 0.934 | 0.908 | 0.875 | 0.655 | 0.495 |
| | all12honest_drop_chrho | 0.760 | 0.775 | 0.751 | 0.715 | 0.497 | 0.220 |
| | consensus | 0.826 | 0.868 | 0.818 | 0.778 | 0.660 | 0.452 |
| | Epiphany | 0.512 | 0.642 | 0.501 | 0.318 | 0.208 | 0.052 |
| Hi-cGAN, GM12878 chr2, chr14 | all12refin_drop_chrho | 0.921 | 0.934 | 0.917 | 0.875 | 0.620 | 0.400 |
| | all12honest_drop_chrho | 0.797 | 0.810 | 0.787 | 0.726 | 0.497 | 0.164 |
| | Hi-cGAN | 0.328 | 0.530 | 0.330 | 0.260 | 0.113 | 0.031 |
| Hi-cGAN, K562 chr2, chr14 | all12refin_drop_chrho | 0.844 | 0.884 | 0.844 | 0.825 | 0.559 | 0.364 |
| | all12honest_drop_chrho | 0.731 | 0.746 | 0.728 | 0.732 | 0.438 | 0.160 |
| | Hi-cGAN, GM12878 model | 0.125 | 0.259 | 0.141 | 0.339 | 0.108 | 0.008 |

The reference-map model is level with C.Origami on Pearson over the band
(0.926 against 0.920) and above it on the 1 Mb tile median (0.944 against
0.905), SCC, insulation correlation (0.924 against 0.866), boundary recall and
loop recall (0.355 against 0.162); it exceeds Epiphany by 0.40 Pearson. The
ChIP-only model, which has no consensus, is below C.Origami in Pearson
(0.808 against 0.920), SCC (0.812 against 0.913), tile median (0.855
against 0.905) and insulation correlation (0.822 against 0.866); it exceeds Epiphany by 0.25 (0.760 against 0.512), the
Hi-cGAN IMR-90 model by 0.24 and Hi-cGAN on GM12878 and K562 by 0.47 and 0.61.
C.Origami's boundary recall (0.474) and loop recall (0.162) lie below the
ChIP-only model's (0.497, 0.192). The Hi-cGAN IMR-90 model scores on chr2
only; C.Origami and the other predictions on chr2 and chr3. Rescoring with
all12refres_drop_chrho in every block runs (`runs/hic_measures_chrho`,
earlier blocks kept in `blocks_norefres`); its rows follow when it finishes.

## 38. AML-inclusive training with the methylation channel (FINAL for fold 0; folds 1 and 2 not run)

Recipe of section 36 (patients, HSPC donors, MOLM-13, AML windows x3, lab
indicator, AML reference of the training patients, input-reference model,
track dropout 0.5, honest validation, selection on the patient-specific hard
windows, fold 0) plus the 14th track Meth: run all12aml_meth_refin_drop at
10 kb and 25 kb (cache mc_all12aml_meth_ref_<k>kb). Meth per sample: the ENCODE
WGBS of the non-AML cells and the RRBS of HCT116, MCF-7 and HFF (the v2 cell
set of section 35, 10 of the 12 cells; NHEK and PC-9 absent), the Xu et al.
WGBS (`aml/primary/meth/xu_<patient>`, `aml_primary_meth.py`, 1 kb bins,
value 1 + methylated fraction) for the seven training patients that have it
(018, 103, 168, 472, 546, 629, 673); absent for training patient 1360, the two
HSPC donors and MOLM-13. `select_tracks` receives Meth through the added-track
rule of `run_config` (checked: the run's select_tracks ends with
'H4K20me1', 'Meth'). Test patients 027, 270 and 773 have WGBS; 1916 has CUT&Tag
and no WGBS and is scored with Meth absent. Inference conditions
(`aml_train_eval.py`, label aml_meth_refin): the patient's marks, marks plus Meth,
Meth alone and no input; for U-937 and the HSPC donors Meth is absent.
Evaluation as section 36 (Pearson, SCC, patient-specific regions, residual
correlation against the AML reference, the HSPC consensus, consensus12 and
the mean of the other patients; Xu loops and stripes at 10 kb; the standard
413-region evaluation of the non-AML cells), the test patients with WGBS
additionally summarised alone.

Results, fold 0 (`results/aml_train/eval_f0_summary.csv`, `eval_f0_patients.csv`,
`eval_f0_features.csv`; file mtimes used 2026-10-06 02:44 for summary and
patients, 02:49 for features; the AML residual run of section 45 may rewrite
these files later).

Training. all12aml_meth_refin_drop_25kb stopped at epoch 60 (best hard-
validation Pearson 0.7493) and _10kb at epoch 60 (0.5332), against epoch 33
(0.7330) and epoch 34 (0.5240) for all12aml_refin_drop without the channel.
Normal accuracy on the 413 Akita test regions (median over cells;
`reference_summary.csv`, 12 training cells / 5 held-out cells): 25 kb
0.9295 / 0.9115 against 0.9288 / 0.9060 without the channel; 10 kb 0.9230 /
0.8952 against 0.9227 / 0.8887.

Held-out patients 027, 270, 773, 1916 (median over the four patients of the
per-patient median on the 413 regions; Pearson, SCC and structure
correlations are Pearson-type scores of `crosscell_predict --structure`; the
last column is the Pearson on the 30 to 53 patient-specific regions per
patient of section 29, where both deviation scores of the patient are in its
top 20%). Patient 1916 has no WGBS, so its Meth conditions equal the
conditions without Meth.

| 25 kb, model | input | Pearson | SCC | insulation | loop enrichment | compartment | patient-specific |
|---|---|---|---|---|---|---|---|
| AML reference alone | | 0.912 | | | | | 0.751 |
| mean of the 11 other patients | | 0.922 | | | | | |
| all12aml_refin_drop | marks | 0.900 | 0.895 | 0.958 | 0.803 | 0.920 | 0.755 |
| all12aml_refin_drop | no input | 0.898 | 0.894 | 0.956 | 0.796 | 0.915 | 0.730 |
| all12aml_meth_refin_drop | marks | 0.904 | 0.894 | 0.959 | 0.801 | 0.922 | 0.752 |
| all12aml_meth_refin_drop | marks + Meth | 0.905 | 0.893 | 0.960 | 0.802 | 0.920 | 0.751 |
| all12aml_meth_refin_drop | Meth only | 0.898 | 0.892 | 0.958 | 0.796 | 0.919 | 0.736 |
| all12aml_meth_refin_drop | no input | 0.897 | 0.893 | 0.957 | 0.793 | 0.916 | 0.738 |

| 10 kb, model | input | Pearson | SCC | insulation | loop enrichment | compartment | patient-specific |
|---|---|---|---|---|---|---|---|
| AML reference alone | | 0.802 | | | | | 0.541 |
| mean of the 11 other patients | | 0.814 | | | | | |
| all12aml_refin_drop | marks | 0.818 | 0.806 | 0.962 | 0.440 | 0.874 | 0.592 |
| all12aml_refin_drop | no input | 0.805 | 0.793 | 0.960 | 0.432 | 0.867 | 0.560 |
| all12aml_meth_refin_drop | marks | 0.815 | 0.801 | 0.966 | 0.441 | 0.874 | 0.576 |
| all12aml_meth_refin_drop | marks + Meth | 0.816 | 0.804 | 0.965 | 0.445 | 0.871 | 0.586 |
| all12aml_meth_refin_drop | Meth only | 0.807 | 0.796 | 0.963 | 0.433 | 0.872 | 0.563 |
| all12aml_meth_refin_drop | no input | 0.809 | 0.793 | 0.963 | 0.433 | 0.868 | 0.575 |

The three test patients with WGBS (027, 270, 773; median):

| model, input | Pearson 25 kb | patient-specific 25 kb | Pearson 10 kb | patient-specific 10 kb |
|---|---|---|---|---|
| AML reference alone | 0.897 | 0.692 | 0.773 | 0.453 |
| all12aml_refin_drop, marks | 0.893 | 0.702 | 0.791 | 0.528 |
| all12aml_refin_drop, no input | 0.888 | 0.667 | 0.771 | 0.479 |
| all12aml_meth_refin_drop, marks | 0.892 | 0.692 | 0.788 | 0.506 |
| all12aml_meth_refin_drop, marks + Meth | 0.894 | 0.697 | 0.790 | 0.526 |
| all12aml_meth_refin_drop, Meth only | 0.886 | 0.678 | 0.772 | 0.476 |
| all12aml_meth_refin_drop, no input | 0.884 | 0.678 | 0.776 | 0.490 |

Residual correlation corr(P - AML reference, T - AML reference), all regions /
patient-specific regions (four patients):

| model, input | 25 kb | 10 kb |
|---|---|---|
| all12aml_refin_drop, marks | 0.167 / 0.274 | 0.234 / 0.268 |
| all12aml_refin_drop, no input | 0.023 / 0.105 | 0.164 / 0.223 |
| all12aml_meth_refin_drop, marks | 0.106 / 0.219 | 0.269 / 0.274 |
| all12aml_meth_refin_drop, marks + Meth | 0.134 / 0.279 | 0.271 / 0.289 |
| all12aml_meth_refin_drop, Meth only | -0.002 / 0.170 | 0.235 / 0.246 |
| all12aml_meth_refin_drop, no input | -0.034 / 0.107 | 0.218 / 0.237 |

Xu et al.'s AML-specific loops (592) and stripes (481), 10 kb, four test
patients: the measured loop enrichment lies 0.307 log2 units above the AML
reference at the loop pixel (stripes 0.210). Prediction minus AML reference,
fraction of the loops (stripes) above the reference, and correlation with the
measured gain:

| model, input | loops: difference | above reference | correlation | stripes: difference | above reference | correlation |
|---|---|---|---|---|---|---|
| all12aml_refin_drop, marks | -0.015 | 0.41 | 0.14 | -0.033 | 0.37 | 0.34 |
| all12aml_refin_drop, no input | -0.034 | 0.29 | 0.16 | -0.079 | 0.06 | -0.07 |
| all12aml_meth_refin_drop, marks | -0.008 | 0.46 | 0.17 | -0.015 | 0.46 | 0.41 |
| all12aml_meth_refin_drop, marks + Meth | -0.004 | 0.47 | 0.18 | 0.014 | 0.53 | 0.42 |
| all12aml_meth_refin_drop, Meth only | -0.009 | 0.43 | 0.13 | 0.032 | 0.66 | 0.23 |
| all12aml_meth_refin_drop, no input | -0.019 | 0.39 | 0.14 | 0.015 | 0.61 | 0.11 |

Held-out HSPC donors and U-937 WT (Meth absent for both; Pearson with marks /
no input): HSPC 25 kb 0.926 / 0.928 (all12aml_refin_drop 0.926 / 0.929), 10 kb
0.828 / 0.829 (0.827 / 0.829); U-937 25 kb 0.858 / 0.862 (0.857 / 0.863), 10 kb
0.859 / 0.825 (0.852 / 0.826).

Reading. The channel does not change what the AML-trained model predicts for
the held-out patients at the level that the data resolve. With the patient's
marks the Pearson differs from the model without the channel by +0.004 (25 kb,
0.904 against 0.900) and -0.003 (10 kb, 0.815 against 0.818), and without ChIP
by -0.001 and +0.005, all inside the 0.013 seed noise of the median Pearson;
per patient the marks difference is +0.003, +0.008, -0.003, -0.001 at 25 kb
(027, 1916, 270, 773) and -0.004, +0.004, -0.002, -0.004 at 10 kb. Giving the
meth model the patient's Meth on top of the marks changes the Pearson by
+0.001 at both resolutions; per patient with WGBS between -0.003 and +0.002.
Meth alone equals no input (0.898 against 0.897 at 25 kb, 0.807 against 0.809
at 10 kb). At 25 kb the models stay below the AML reference in all four
patients (0.904 against 0.912), at 10 kb all four are above it with either
model (0.815 against 0.802), and (meth model, marks) above the mean of the other eleven patients
only for 270. The patient-specific regions show the one visible effect: at 10
kb adding Meth to the marks raises the patient-specific Pearson from 0.576 to
0.586 over the four patients and from 0.506 to 0.526 over the three with WGBS
(per patient +0.001, +0.020, +0.024 for 027, 270, 773), but the meth model with
marks (0.576) stays below the model without the channel (0.592), and at 25 kb
the per-patient changes are -0.008, +0.013, -0.018. The residual correlation
with the AML reference rises at 10 kb with the channel in the model (0.269
against 0.234 with marks, 0.218 against 0.164 without input), but at 25 kb it
falls with marks (0.106 against 0.167). At the loops and stripes the meth
model with marks correlates with the measured gain at 0.17 and 0.41 against
0.14 and 0.34, but the loop enrichment stays 0.008 below the AML reference
where the measured loops lie 0.307 above it, so the gain at the paper's
loops is not reproduced. Methylation as an input therefore does not give the AML-trained
model information about the patient that the marks and the AML reference do
not already carry; a patient-specific effect of a few thousandths to 0.02 in
Pearson on 30 to 53 regions per patient is within what the checkpoint choice
and the seed move (below).

Limitations. Fold 0 only (folds 1 and 2 not run), four test patients of which
three have WGBS; the medians over three or four patients move with a single
patient. One training run per model: the seed noise of the median Pearson
(0.013, section 15) was measured on the non-AML models and is the only noise
estimate, and no noise estimate exists for the patient-specific, residual,
loop and stripe columns. The two models differ in training length (60 epochs
against 33 and 34; the early stopping with patience 10 fired later), so the
comparison mixes the channel with a longer training; the hard-validation
Pearson is higher with the channel (0.7493 against 0.7330 at 25 kb, 0.5332
against 0.5240 at 10 kb) while the test scores are not. Meth is absent in
training for patient 1360, MOLM-13, the HSPC donors and, among the non-AML
cells, NHEK and PC-9, so the model sees the absent state often; the test
patients' WGBS comes from the same lab and protocol as the training patients'
WGBS. The HSPC and U-937 rows test only the absent state. The CSVs have been
rewritten by later jobs before; the numbers above are from the files with the
mtimes given.


## 39. Across-cell variability as a predictor of the consensus error (FINAL, 25 and 10 kb)

Question: does the across-cell variability of the other cells' Hi-C predict
where the leave-one-out consensus is wrong for a target cell, so that std and
presence channels would be worth adding to the model input? CPU only, no model.
Script `scripts/consensus_variance_diag.py`, tables and figures in
`results/consensus_variance/` (`corr_by_cell_<k>kb.csv`, `corr_pooled_<k>kb.csv`,
`shrinkage_by_cell_<k>kb.csv`, `shrinkage_summary_<k>kb.csv`,
`noise_floor_<k>kb.csv`, `binned_curve_<k>kb.csv`,
`consensus_variance_<k>kb.png`). 5 kb was not run.

Setup. Per-pixel statistics of the other cells' proper O/E maps
(`crosscell_oe/<cell>_proper_<k>kb.cool`, the maps that `consensus_oe.py` averages;
the recomputed mean equals the stored `consensus_loo_gm12878` to 2e-7). Targets: the
5 held-out cells (K562, HMEC, HUVEC, HMEC deep, HUVEC deep; others = the 12
training cells, i.e. consensus12) and the 12 training cells (others = the 11
remaining training cells, i.e. consensus_loo). A held-out cell never enters a
mean, std, threshold or fitted parameter. Per pixel (upper triangle, k >= 2,
target and consensus nonzero): mean_others, std_others (ddof 1), presence (share
of other cells above their own 90th percentile of that diagonal, taken on 300
random training windows), range, residual = measured - mean_others. Two window
sets: the 413 Akita test regions (the convention of the reference_summary
tables) and the disjoint test windows of cache mc_all12ref (295 at 25 kb, 311 at
10 kb). The median-of-cell-medians of the consensus alone reproduces the
reference_summary values on the 413 regions (held-out 0.8977 at 25 kb, 0.8586 at
10 kb; training cells 0.9223 and 0.8867). Spearman on diagonal ranks = Pearson of
the percentile ranks within each diagonal, with a 95% interval from a bootstrap
over windows.

(a) |residual| against std_others, median over cells (range over cells), 413
regions:

| res | cells | pixel Spearman | diagonal-rank Spearman | pooled over cells (95% CI) | R2 of res^2 on std^2 per diagonal | R2 with std^2, presence, range^2, (mean dev)^2 |
|---|---|---|---|---|---|---|
| 25 kb | 5 held-out | 0.27 | 0.12 (0.09 to 0.16) | 0.125 (0.112 to 0.135) | 0.027 | 0.041 |
| 25 kb | K562, HMEC deep, HUVEC deep | 0.28 | 0.145 (0.11 to 0.16) | 0.139 (0.123 to 0.155) | 0.031 | 0.064 |
| 25 kb | 12 training LOO | 0.27 | 0.143 (0.12 to 0.22) | 0.149 (0.141 to 0.157) | 0.024 | 0.039 |
| 10 kb | 5 held-out | 0.25 | 0.10 (0.04 to 0.13) | 0.086 (0.078 to 0.093) | 0.015 | 0.024 |
| 10 kb | K562, HMEC deep, HUVEC deep | 0.29 | 0.123 (0.10 to 0.13) | 0.117 (0.107 to 0.128) | 0.018 | 0.032 |
| 10 kb | 12 training LOO | 0.28 | 0.119 (0.10 to 0.20) | 0.133 (0.127 to 0.139) | 0.015 | 0.024 |

The disjoint test windows give the same picture (diagonal-rank Spearman 0.115,
0.136, 0.146 at 25 kb and 0.091, 0.113, 0.116 at 10 kb for the three groups).
The pixel-level Spearman of 0.25 to 0.28 is mostly the shared distance trend
(both quantities fall with distance). Presence and range add nothing beyond
std (diagonal-rank Spearman with |residual|: presence -0.05 to 0.03, range
0.09 to 0.13). The best single training cell is gm23248 (0.22 at 25 kb); no
held-out cell exceeds 0.165.

(b) Signed residual, diagonal ranks, 413 regions, median over cells. Held-out
cells at 25 kb: against std 0.20, against presence -0.19, against the
consensus deviation from its diagonal mean -0.36; training cells: 0.01, 0.00 and
-0.04 (per-cell values range from -0.09 to 0.13 for std and from -0.37 to 0.23
for presence, in both signs). The consensus overshoots away from the diagonal
mean for K562 and HMEC (the signed residual falls with the consensus deviation)
but not for HUVEC, and the direction differs between training cells, so no sign
rule transfers. The consensus overshoot (consensus minus measured, signed by the
consensus deviation) against std is 0.03 to 0.09 for the held-out cells. At 10 kb
the held-out values are 0.14, -0.14 and similarly cell-dependent.

(d) Median per-region Pearson, 413 regions, median over cells of the cell
medians (held-out 5 / training 12). Shrinkage parameters fit on the 12 training
cells' training windows, leave-the-target-out (the target's own residual never
fits its own parameter; the other cells' consensus maps still contain it):
pred = mu_d + w (consensus - mu_d), mu_d the per-diagonal mean of training-cell
maps, w = 1 / (1 + lambda (std / diagonal-mean std)^2), lambda in 0 to 8 chosen
by median Pearson (1 parameter); or a slope per bin of std, of presence or of
distance (5 to 7 parameters), plus a constant w as a variance-free control.

| res | consensus alone | std shrinkage (lambda) | std-bin slopes | presence-bin slopes | distance-bin slopes (control) | constant w |
|---|---|---|---|---|---|---|
| 25 kb held-out | 0.8977 | 0.8977 (0) | 0.8982 | 0.8978 | 0.8979 | 0.8978 |
| 25 kb training | 0.9223 | 0.9223 (0) | 0.9222 | 0.9222 | 0.9222 | 0.9223 |
| 10 kb held-out | 0.8586 | 0.8586 (0) | 0.8567 | 0.8584 | 0.8587 | 0.8586 |
| 10 kb training | 0.8867 | 0.8867 (0) | 0.8872 | 0.8866 | 0.8866 | 0.8866 |

Training selects lambda = 0 at both resolutions, so the one-parameter
variant is the consensus. The per-cell medians move by at most 0.0004 (25 kb)
and 0.002 (10 kb), in both directions, and the std-aware variants are no better
than the distance-only control. Variance information alone does not move the
0.898 (25 kb) and 0.859 (10 kb) consensus-alone medians. Figure: mean |residual|
rises with std_others in every distance band (at 10 kb, 50 to 100 kb: 0.10 to
0.16 over the std range 0.08 to 0.28), so std tracks the error magnitude, but
the per-pixel scatter around that trend is large, which is the R2 of 0.015 to
0.03.

Noise floor (`noise_floor_<k>kb.csv`). Replicate noise per pixel as
(rep1 - rep2)^2 / 2: single-lane replicate pairs of GM12878, IMR90, K562, HMEC and
HUVEC (110 to 230 M contacts per replicate) and, as a lower noise estimate, four
deep intact pairs (K562 ENCSR676BEP 1.0 and 1.45 billion contacts; three
GM12878 pairs). Share of mean std_others^2 that one replicate's noise accounts
for: 23% (single lane) and 21% (deep pairs) at 25 kb (range over distance bands
0.19 to 0.38), and 65% and 49% at 10 kb. The diagonal-rank Spearman between
std_others and the pixel noise level is 0.10 (single lane) and 0.13 (deep) at
25 kb, and -0.03 and 0.04 at 10 kb; the pixel-level Spearman is 0.35 and 0.33
(a shared distance trend). Against the target's own replicate disagreement,
|residual| has diagonal-rank Spearman -0.00 to 0.10, and the partial Spearman of
|residual| with std_others given the target's own replicate-difference rank is
0.11 to 0.17 at 25 kb and 0.10 to 0.15 at 10 kb, the same as without the
adjustment. Replicate noise accounts for 4.7% (deep K562 pair) to 25%
(single-lane K562) of the mean squared residual at 25 kb and 16% to 83% at
10 kb for the K562 map; the HUVEC maps reach 42% (HUVEC) and 44% (HUVEC deep) at 25 kb.

Decision. At 25 kb the diagonal-rank Spearman between |residual| and std_others
is 0.09 to 0.16 for the held-out cells (pooled 0.125) and at 10 kb 0.04 to 0.13
(pooled 0.086); both are between the 0.1 (not informative) and 0.2
(informative) thresholds and below 0.2 in every held-out cell. Presence and
range carry no extra signal, and every variance-aware shrinkage leaves the
consensus-alone medians unchanged. Recommendation: do not train a model with
std and presence input channels on the basis of this analysis. The extra
channels would give the network a feature that explains 2 to 3% of the squared
residual at 25 kb and 1.5 to 2% at 10 kb, and at 10 kb the std is dominated
by replicate noise (49 to 65% of std^2).

Limitations. The std rests on 11 or 12 other cells, so each pixel's std is
itself a noisy estimate, which attenuates every correlation here; a larger
reference panel could raise them somewhat, and the size of that attenuation was
not measured. The consensus cells include 2 Micro-C and 3 intact Hi-C maps next
to 7 in situ Hi-C maps, so std partly measures assay differences. The replicate noise is that
of a single replicate and of 0.25 to 1.45 billion contacts per replicate (deep pairs), higher than that
of the merged deep maps entering the consensus (upper bound), and replicate
pairs exist for 5 cells only (the pixel noise map is their mean for every
target); HMEC and HUVEC single-lane targets are replicate 1 itself. The parameter
fit leaves the target's own map out of its own parameters but not out of the
other cells' consensus maps during fitting. A model that learns where to deviate
from the consensus may use variance in ways a pixel-wise shrinkage cannot, so
this analysis bounds only the variance-alone route. Windows in the test-window
set overlap by half, so its intervals are optimistic. 5 kb not run.

## 40. Similarity-weighted consensus against the plain mean (FINAL, 25 and 10 kb)

Question: does weighting the other cells' Hi-C by how similar their ChIP profiles
(or DNA methylation) are to the target cell's predict the target's Hi-C better
than the plain unweighted mean of the other cells' maps (the leave-one-out
consensus)? CPU only, no model. Script `scripts/consensus_weighted.py`, tables and
figures in `results/consensus_weighted/` (`summary_<k>kb.csv`, `per_cell_<k>kb.csv`,
`grid_by_cell_<k>kb.csv`, `selected_params_<k>kb.csv`, `fit_selection_<k>kb.csv`,
`rank_agreement_<k>kb.csv`, `similarity_<k>kb.csv`, `similarity_by_track_<k>kb.csv`,
`consensus_weighted_<k>kb.png`). 5 kb was not run.

Setup. Maps, regions and evaluation as in section 39 (proper O/E maps,
`consensus_variance_diag.py` loading; median per-region Pearson on the upper
triangle, k >= 2; the 413 Akita test regions and the disjoint test windows of
cache mc_all12ref, 295 at 25 kb and 311 at 10 kb). The plain consensus alone
reproduces 0.8977 (25 kb) and 0.8586 (10 kb) for the 5 held-out cells and 0.9223
and 0.8867 for the training cells. Targets: the 5 held-out cells (references =
the 12 training cells) and the 12 training cells (strict leave-one-out, the 11
other training cells). The similarity S(T, c) uses only input tracks (13 ChIP
bigWigs as the model reads them, log1p of the binned mean signal; the Meth track
of section 27), never any Hi-C of T. HMEC and HMEC deep share one set of inputs, as
do HUVEC and HUVEC deep.

Similarity variants. chip_gw: Pearson across all genome bins per track, mean over
the tracks both cells have (a cell with missing tracks is compared on the common
tracks; GM23248 has no CTCF, HFF 7 of 13 tracks); chip_gw_sp: the same with
Spearman; ctcfk27_gw: CTCF and H3K27ac only; chip_loc and ctcfk27_loc: the same
Pearson computed inside each evaluation window (about 42 bins at 25 kb, 105 at
10 kb for the 413 regions), so the weights vary by locus; meth_gw, meth_loc: Meth
only; comb_gw, comb_loc: mean of the available ChIP and Meth similarities.
Methylation exists for 10 of the 12 training cells (WGBS for 7, lifted hg19 RRBS
for HCT116, MCF-7 and HFF; none for NHEK and PC-9) and for K562 and HMEC (HMEC
deep), none for HUVEC. A reference without Meth takes the mean of the defined
similarities of its row; a target without Meth gets equal weights (meth variants
are informative for 3 of 5 held-out cells and 10 of 12 training cells; comb
variants fall back to ChIP only).

Weights. softmax(S / tau) over the references, tau in {inf, 1, 0.5, 0.3, 0.2, 0.15,
0.1, 0.075, 0.05, 0.03, 0.02, 0.01, 0.005}, and top-k with equal weights (k = 5, 3,
2, 1, all); inf and "all" are the plain mean, k = 1 is the limit tau -> 0. tau and k
are chosen per variant on 300 random training-chromosome windows (the "fit" set of
section 39, 0 overlaps with the 413 regions and with the test windows, checked in
the script) by the pooled median per-window Pearson with the 12 training cells as
targets, each with its own leave-one-out references, leaving the target out of its
own selection (a training target never sees its own Hi-C in the choice; Meth
variants use only training cells with Meth). No held-out cell, test region or test
window selects anything. Selected: chip_loc tau 0.075 (K562 and the held-out set)
or 0.1 (training cells) with k = 5 at 25 kb, tau 0.1 and k = 5 at 10 kb; chip_gw
tau 0.05 (25 kb) and 0.075 (10 kb); the Meth-only variants select tau = inf or 1
(plain mean or nearly so). On the fit windows the pooled median Pearson of the 12
training cells is 0.8925 (plain) and 0.9078 (chip_loc, tau 0.1) at 25 kb, 0.8598 and
0.8745 at 10 kb.

Reference rows (ORACLE, use the target's own Hi-C, not achievable): best single
other cell; non-negative least-squares weights fitted on the evaluation pixels
(in-sample) and 2-fold cross-fitted over regions; softmax of the true Hi-C
similarity (median per-region Pearson of T with c) with the best tau chosen on the
evaluation regions. These oracles give one weight vector per target, constant over
loci, so they bound genome-wide weighting only, not locus-varying weights.

(a) Median per-region Pearson on the 413 Akita regions, 5 held-out cells. Column
"delta" is the median over the 5 cells of the per-cell difference (variant minus
plain, difference of the two region-medians), "won" the cells with delta > 0.0001,
95% interval from a bootstrap over regions (shared resample across cells) of that
median over cells:

| variant (selected) | 25 kb median over cells | 25 kb delta (95% CI) | 25 kb won | 10 kb median over cells | 10 kb delta (95% CI) | 10 kb won |
|---|---|---|---|---|---|---|
| plain mean | 0.8977 | 0 | | 0.8586 | 0 | |
| chip_gw softmax | 0.9051 | 0.0024 (-0.0018 to 0.0071) | 3 | 0.8659 | 0.0006 (-0.0024 to 0.0064) | 3 |
| chip_gw top-k | 0.8972 | -0.0031 (-0.0081 to -0.0003) | 0 | 0.8598 | -0.0091 (-0.0127 to -0.0035) | 1 |
| chip_gw Spearman softmax | 0.9035 | 0.0006 (-0.0040 to 0.0058) | 3 | 0.8608 | 0.0010 (-0.0029 to 0.0061) | 3 |
| ctcfk27_gw softmax | 0.9052 | 0.0012 (-0.0018 to 0.0050) | 3 | 0.8660 | -0.0023 (-0.0042 to 0.0031) | 2 |
| **chip_loc softmax** | **0.9131** | **0.0080 (0.0052 to 0.0126)** | **5** | **0.8731** | **0.0060 (0.0024 to 0.0103)** | **5** |
| chip_loc top-k | 0.9039 | 0.0021 (-0.0009 to 0.0069) | 5 | 0.8617 | -0.0030 (-0.0070 to 0.0032) | 1 |
| ctcfk27_loc softmax | 0.9061 | 0.0048 (0.0020 to 0.0094) | 5 | 0.8709 | 0.0040 (-0.0008 to 0.0096) | 4 |
| meth_gw softmax | 0.8977 | 0.0000 | 0 of 3 informative | 0.8586 | 0.0000 | 0 of 3 |
| meth_loc softmax | 0.8990 | 0.0000 (0.0000 to 0.0014) | 2 of 3 | 0.8613 | 0.0000 (0.0000 to 0.0013) | 2 of 3 |
| comb_gw softmax | 0.9013 | -0.0002 (-0.0007 to 0.0025) | 2 | 0.8588 | 0.0002 (-0.0004 to 0.0016) | 3 |
| comb_loc softmax | 0.9038 | 0.0049 (0.0023 to 0.0078) | 5 | 0.8620 | 0.0034 (0.0011 to 0.0057) | 5 |
| ORACLE best single other cell | 0.8821 | -0.0153 | 0 | 0.8267 | -0.0417 | 0 |
| ORACLE NNLS in-sample | 0.9069 | 0.0052 | 5 | 0.8692 | 0.0040 | 3 |
| ORACLE NNLS cross-fitted | 0.9057 | 0.0044 | 5 | 0.8671 | 0.0037 | 3 |
| ORACLE softmax of Hi-C similarity | 0.9047 | 0.0069 | 4 | 0.8698 | 0.0028 | 5 |

The median over cells of the cell medians (column "median over cells") differs
from plain by more than the median of the per-cell deltas (chip_loc: 0.9131 against
0.8977, a difference of 0.0154) because the cells change rank order; the decision
rule below uses the smaller, paired median of deltas. Weighted tau = 0.005, k = 1
(the single most similar cell) loses 0.047 (25 kb, chip_gw) to 0.053 (meth_gw): the
best-looking single neighbour is worse than the 12-cell mean in every held-out cell.
Per cell, chip_loc softmax (413 regions, delta with bootstrap interval): 25 kb K562
+0.0159 (0.0084 to 0.0217), HMEC +0.0080 (0.0015 to 0.0128), HUVEC +0.0058
(-0.0005 to 0.0162), HMEC deep +0.0090 (0.0045 to 0.0137), HUVEC deep +0.0079
(0.0016 to 0.0136); 10 kb +0.0145, +0.0018 (-0.0020 to 0.0108), +0.0081, +0.0052,
+0.0060. Share of regions above the plain mean 0.61 to 0.68 (25 kb) and 0.61 to
0.69 (10 kb).

(b) Same on the disjoint test windows (windows overlap by half, so the intervals are
optimistic), 5 held-out cells, median of per-cell deltas (won): chip_loc softmax
+0.0183 (5) at 25 kb and +0.0079 (5) at 10 kb; ctcfk27_loc +0.0087 (5) and +0.0097
(5); comb_loc +0.0088 (5) and +0.0032 (5); chip_gw +0.0042 (3) and +0.0040 (4);
ctcfk27_gw +0.0027 (4) and +0.0048 (5); meth_gw 0 and meth_loc +0.0016 (3 of 3, 25
kb) and 0.0000 (10 kb); ORACLE NNLS cross-fitted +0.0115 and +0.0045; ORACLE
Hi-C-similarity softmax +0.0064 and +0.0052.

(c) 12 training cells, strict leave-one-out, 413 regions, delta (cells won of 12):
chip_loc softmax +0.0068 (11) at 25 kb and +0.0086 (11) at 10 kb; chip_gw softmax
+0.0076 (9) and +0.0050 (12); ctcfk27_gw +0.0042 (10) and +0.0068 (11); ctcfk27_loc
+0.0044 (11) and +0.0072 (11); comb_loc +0.0014 (7) and +0.0011 (8); meth and comb_gw
0.0000 to +0.0001; top-k variants chip_gw +0.0064 (9) and +0.0049 (9), chip_loc +0.0052
(9) and +0.0034 (8), ctcfk27_loc +0.0006 (6) and +0.0036 (8); chip_gw Spearman softmax
+0.0023 (8) and +0.0038 (9). ORACLE
NNLS cross-fitted +0.0131 (12) and +0.0136 (12); ORACLE Hi-C-similarity softmax +0.0112
and +0.0079 (11 won, 1 tie); best single cell -0.0195 (4 won) and -0.0316 (1 won).
Per training cell chip_loc 25 kb: gm12878 +0.0164, imr90 +0.0101, hct116 +0.0040, hff
+0.0201, h1 +0.0053, hepg2 +0.0036, a549 +0.0037, nhek -0.0016, gm23248 +0.0150, mcf7
+0.0069, cd14mono +0.0233, pc9 +0.0067.

(d) Does S rank the other cells like the true Hi-C similarity? Spearman between S
and the median per-region Pearson of the Hi-C maps of T and c, over the 11 or 12
references, 413 regions, median (range) over targets, held-out / training:

| S | 25 kb held-out | 25 kb training | 10 kb held-out | 10 kb training |
|---|---|---|---|---|
| chip_gw | 0.69 (0.55 to 0.76) | 0.62 (0.32 to 0.86) | 0.69 (0.39 to 0.76) | 0.50 (-0.02 to 0.83) |
| chip_loc | 0.69 (0.55 to 0.82) | 0.57 (0.15 to 0.90) | 0.64 (0.24 to 0.75) | 0.38 (-0.16 to 0.87) |
| ctcfk27_gw | 0.41 (0.33 to 0.55) | 0.49 (-0.12 to 0.93) | 0.36 (0.22 to 0.38) | 0.45 (-0.08 to 0.91) |
| ctcfk27_loc | 0.43 (0.29 to 0.52) | 0.68 (-0.17 to 0.87) | 0.34 (0.06 to 0.40) | 0.55 (-0.13 to 0.90) |
| chip_gw Spearman | 0.59 (0.14 to 0.64) | 0.37 (-0.34 to 0.74) | 0.54 (0.19 to 0.70) | 0.40 (-0.11 to 0.78) |
| meth_gw | 0.27 (0.21 to 0.28; 3 cells) | 0.15 (-0.83 to 0.37; 10 cells) | 0.39 (0.36 to 0.41) | 0.16 (-0.68 to 0.60) |
| comb_gw | 0.38 (0.11 to 0.69) | 0.27 (-0.40 to 0.55) | 0.39 (0.16 to 0.69) | 0.25 (-0.55 to 0.52) |

ChIP similarity over the 13 tracks ranks the references moderately well (0.6 to
0.7 held-out), CTCF plus H3K27ac alone and methylation alone rank them poorly, and
the methylation ranking is negative for some training cells. The per-window
Spearman between the local S and the per-window Hi-C similarity (median over
windows) is only 0.33 (chip_loc, 25 kb), 0.25 (ctcfk27_loc and comb_loc) and 0.09
(meth_loc), 0.36, 0.28 and 0.10 at 10 kb: the local ChIP similarity of one 1 Mb
window identifies the similar cell at that window only weakly.

Decision rule (median over the 5 held-out cells of the paired delta above 0.005
and wins in at least 4 of 5, 413 regions, softmax variants with tau chosen on
training cells). Met by chip_loc softmax at both resolutions (0.0080 with 5 of 5 at
25 kb, 0.0060 with 5 of 5 at 10 kb; also on the disjoint test windows and as 11 of 12
on the training cells). Not met by: ctcfk27_loc (0.0048 at 25 kb, 0.0040 at 10 kb),
comb_loc (0.0049, 0.0034), every genome-wide variant (chip_gw 0.0024 and 0.0006),
every top-k variant, and the methylation variants (0.0000; no more than 3 of 5
held-out cells have Meth, so wins in 4 of 5 are impossible). A weighting by
methylation alone or by CTCF and H3K27ac alone gives no usable gain, and the
methylation similarity chosen on training cells is the plain mean.

Verdict. The idea works only in the locus-varying ChIP form, and then the gain is
small: +0.006 to +0.008 median Pearson on the 413 regions (+0.008 to +0.018 on the
disjoint windows) against a seed noise of 0.013 (section 15). The genome-wide ORACLE
weights (one vector per target, fitted on the target's own Hi-C) reach +0.0044 to
+0.0069 at 25 kb, so chip_loc (+0.0080) exceeds them because its weights vary by
locus. 60 to 70% of regions improve. Methylation adds nothing here. Recommendation:
not worth a training campaign as such. The target's own ChIP tracks enter the model
as input and the model learns the residual against the reference, so the part of the
0.006 to 0.008 that survives as reference-input gain is expected to be smaller than
on the consensus alone; this was not measured. If a retraining happens anyway, one
25 kb ablation with the chip_loc softmax consensus (tau 0.075 to 0.1, computed per
window from the target's own tracks and the 12 or 11 reference tracks) as the
reference input and residual base would settle it. It needs a reference cache
that depends on the target's inputs per window, and overlapping windows
(stride half the window) give window-dependent references for the same pixel,
which the present cache design does not allow.

Limitations. (1) Only 11 or 12 reference cells; a softmax over so few maps has few
degrees of freedom (the selected top-5 of 11 or 12 already drops half the cells).
(2) The reference panel mixes assays (7 in situ Hi-C, 3 intact Hi-C, 2 Micro-C) and
the similarity of inputs may track assay-linked differences in the maps; the effect
on the weights was not separated. (3) Methylation is missing for NHEK, PC-9 and HUVEC,
and HCT116, MCF-7 and HFF are lifted RRBS next to WGBS; the Meth results rest on 3
informative held-out cells (K562, HMEC, HMEC deep) and 10 training cells and say
little about methylation beyond this panel. (4) tau and k were chosen on 300
training windows of the training chromosomes with the 12 training cells (11 for a
training target), so every selected value is fitted to training-cell behaviour;
the selected parameters are nearly flat around the optimum (the fit-window median
changes by under 0.001 between tau 0.1 and 0.075 for chip_loc) but 12 cells make
the choice itself uncertain. (5) HMEC and HMEC deep, and HUVEC and HUVEC deep, share
inputs and tau, so the 5 held-out cells are 3 independent input sets; "5 of 5" is
effectively 3 of 3. (6) Nine similarity variants, two weightings and two resolutions
were evaluated; one variant passes the rule at both resolutions, which is
unlikely to be a selection artefact of one window set, but the gain was not tested on
a second independent panel. (7) The local similarity correlates the cells' tracks in
one 1 Mb to 2 Mb window of 42 to 192 bins and is noisy; per-window rank agreement
with the true Hi-C similarity is 0.1 to 0.4. (8) The oracles use one global weight
vector; a locus-wise oracle was not computed, so the table gives no upper bound for
locus-varying weights. (9) Seed noise 0.013 is the model-training noise of
section 15 and a loose scale for a deterministic reference; paired intervals over
regions are the relevant uncertainty, and the half-overlapping test windows make
those on the test-window set optimistic. (10) cd14mono has a consensus of 0.668
(25 kb), well below the other cells, and its large gain (+0.023) moves the training-cell
mean; the medians over cells are less affected. (11) The weighted consensus was
evaluated as a map, not inside a model; a model with the target's ChIP may profit
less or differently.

## 41. ChIP-seq peak height against DNA sequence and DNA methylation: shared and independent parts (FINAL)

`scripts/chip_seq_meth_relationships.py` (CPU), tables and figures in
`results/chip_seq_meth_relationships/`. Targets, cells and the honest split
are those of section 35 part B: each ChIP track at 10 kb, log1p and z-scored
as the model sees it, training on the odd chromosomes, scoring on the even
chromosomes only, seen_cell (odd chromosomes of the seven training cells) and
unseen_cell (leave one training cell out; K562 and HMEC in no training set and
scored with the model trained on all seven, so their seen_cell and unseen_cell
results are one fit, reported as unseen_cell). Independent cells: 9 (GM12878,
IMR-90, H1, HepG2, A549, GM23248, CD14+ monocyte, K562, HMEC; HMEC deep and
HMEC share inputs and count once). Cells that lack a track reduce the count:
CTCF 8, RAD21 6, SMC3 5, the other ten marks 9 (seen_cell: 6, 5, 4 and 7).
The lifted hg19 RRBS cells (HCT116, MCF-7, HFF) are excluded because their
methylation channel is the cell mean in over 95 percent of the 1 kb bins
(section 35 part C). Intervals are percentile bootstrap intervals of the
median over cells (5000 resamples of cells); with 5 to 9 cells they are wide.

**Feature sets.** Methylation (M): the 10 section 35 features (bin mean,
minimum, maximum, standard deviation of the 1 kb values, +-50 kb and +-500 kb
means, local standard deviation, low and high methylated fractions). Sequence
(S): the 3 section 35 features (GC, CpG density, CTCF motif strength;
`seq_base`) or the extended set (`seq`, 16 features: those 3 plus GC and CpG
density over +-50 kb and +-500 kb, forward and reverse CTCF motif counts, CTCF
motif count in +-50 kb, and bin and +-50 kb counts of the YY1, ZNF143 and MAZ
motifs the model also reads). Other marks (O): the other 12 tracks of the
same cell (missing tracks are NaN, handled by the boosting trees); W: the
other marks of the same group; X: the marks of the other groups. Groups:
CTCF and cohesin (CTCF, RAD21, SMC3), active (H3K27ac, H3K4me1, H3K4me2,
H3K4me3, H3K9ac), gene body (H3K36me3, H3K79me2, H4K20me1), repressive
(H3K27me3, H3K9me3). Histogram gradient boosting as in section 35, 500,000
training rows, 143 fit sets (13 marks x 11 sets) each with the all-training fit
and the leave-one-cell-out fits.

**Reproduction check.** The set `both_base` (3 sequence features + methylation)
gives the section 35 medians (unseen_cell R2: H3K9ac 0.54, H3K4me2 0.54,
H3K4me3 0.47, H3K79me2 0.44, H3K36me3 0.40, H3K27ac 0.27, CTCF 0.18).

### 1. Variance partition (sequence against methylation)

Unique sequence = R2(both) - R2(M), unique methylation = R2(both) - R2(S),
shared = R2(S) + R2(M) - R2(both). unseen_cell, extended sequence set, median
over cells (R2; in brackets the signed squared Pearson correlation, which
removes the cell-specific scale error, section 4). Tables
`variance_partition_summary.csv`, `variance_partition_per_cell.csv`; figure
`fig_variance_partition.png`.

| mark | cells | R2 S | R2 M | R2 both (95% interval) | unique S | shared | unique M |
|---|---|---|---|---|---|---|---|
| H3K9ac | 9 | 0.30 (0.32) | 0.49 (0.54) | 0.55 (0.29 to 0.65) | 0.04 | 0.26 | 0.29 (0.29) |
| H3K4me2 | 9 | 0.32 (0.33) | 0.48 (0.56) | 0.51 (0.36 to 0.63) | 0.04 | 0.27 | 0.25 (0.26) |
| H3K4me3 | 9 | 0.33 (0.36) | 0.36 (0.43) | 0.47 (-0.79 to 0.54) | 0.07 | 0.22 | 0.21 (0.21) |
| H3K79me2 | 9 | 0.20 (0.21) | 0.42 (0.43) | 0.43 (0.07 to 0.46) | 0.04 | 0.14 | 0.23 (0.24) |
| H3K36me3 | 9 | 0.14 (0.22) | 0.39 (0.46) | 0.41 (-0.21 to 0.57) | 0.01 | 0.14 | 0.22 (0.27) |
| H3K27ac | 9 | 0.11 (0.18) | 0.27 (0.41) | 0.26 (-0.30 to 0.43) | 0.01 | 0.09 | 0.09 (0.27) |
| H3K27me3 | 9 | 0.19 (0.23) | 0.10 (0.21) | 0.26 (0.07 to 0.40) | 0.15 | 0.02 | 0.11 (0.19) |
| RAD21 | 6 | 0.24 (0.29) | 0.02 (0.12) | 0.24 (-0.16 to 0.32) | 0.17 | 0.07 | -0.01 (0.02) |
| SMC3 | 5 | 0.25 (0.30) | 0.06 (0.20) | 0.22 (-0.13 to 0.40) | 0.15 | 0.13 | 0.12 (0.08) |
| CTCF | 8 | 0.24 (0.27) | 0.07 (0.17) | 0.19 (0.08 to 0.27) | 0.13 | 0.10 | 0.04 (0.04) |
| H3K4me1 | 9 | 0.13 (0.18) | 0.21 (0.35) | 0.18 (-0.68 to 0.43) | 0.03 | 0.09 | 0.22 (0.24) |
| H4K20me1 | 9 | 0.19 (0.28) | 0.01 (0.14) | 0.16 (-0.43 to 0.28) | 0.08 | 0.07 | -0.03 (-0.01) |
| H3K9me3 | 9 | 0.07 (0.13) | -0.06 (0.04) | 0.05 (-0.42 to 0.18) | 0.11 | 0.02 | -0.07 (-0.02) |

The parts are medians per part and do not add to R2(both). Bootstrap
intervals of the unique methylation part exclude 0 only for H3K9ac (0.21 to
0.35), H3K4me2 (0.01 to 0.35), H3K4me3 (0.04 to 0.51) and, in the squared Pearson partition, for
H3K9ac, H3K4me2, H3K4me3, H3K36me3, H3K79me2, H3K27ac, H3K4me1 and H3K27me3;
for CTCF, RAD21, SMC3, H4K20me1 and H3K9me3 the unique methylation part is
indistinguishable from 0 and the unique sequence part carries the explained
variance (CTCF 0.13, RAD21 0.17, SMC3 0.15, H3K9me3 0.11, H3K27me3 0.15).
seen_cell (7 training cells): R2 both 0.67 (H3K9ac), 0.62 (H3K4me3), 0.61
(H3K4me2), 0.55 (H3K36me3), 0.50 (H3K79me2), 0.47 (H3K27ac), 0.46
(H3K27me3), 0.45 (H3K4me1), 0.42 (RAD21), 0.39 (SMC3), 0.38 (CTCF), 0.35
(H4K20me1), 0.29 (H3K9me3), with unique methylation 0.11 to 0.39 for every
mark (intervals above 0 for all 13, e.g. CTCF 0.11, interval 0.05 to 0.28)
and unique sequence 0.02 to 0.14. A seen cell therefore gives methylation a
unique share also for CTCF and cohesin that an unseen cell does not.

Sequence set matters. With the 3 section 35 sequence features the unique
methylation part of unseen_cell R2 is larger (H3K9ac 0.42, H3K36me3 0.34,
H3K79me2 0.33, H3K27ac 0.15, H3K4me1 0.28) than with the extended set (0.29,
0.22, 0.23, 0.09, 0.22): a part of the methylation signal in section 35 is
GC and CpG context, which is DNA sequence. The extended set raises R2 of
sequence alone, e.g. H3K27ac 0.03 to 0.11, H3K36me3 0.00 to 0.14, H3K79me2
0.07 to 0.20.

Grouped permutation importance of the unseen_cell combined model (extended
sequence + methylation; median drop of R2 when the group is shuffled; `
permutation_importance_summary.csv`; correlated groups share their drop, so
the values underestimate each group): sequence 0.36 (CTCF), 0.36 (RAD21),
0.33 (SMC3), 0.42 (H3K27me3), 0.23 (H3K9me3, H4K20me1, H3K4me3); methylation
local + context: H3K4me2 0.66 + 0.44, H3K4me3 0.64 + 0.31, H3K9ac 0.57 +
0.43, H3K27ac 0.42 + 0.36, H3K36me3 0.49 + 0.13, H3K79me2 0.20 + 0.36,
H3K4me1 0.34 + 0.36; CTCF 0.10 + 0.19; H3K9me3 0.05 + 0.03.

### 2. CTCF motif sites: ChIP height, motif strength and CpG methylation

Sites: 72,626 non-overlapping hits of the CTCF motif (JASPAR MA0139.2, PWM
score at least 0.6 of the maximum; the model channel uses 0.8, which gives
about 6,000 sites) genome-wide. Per cell: log1p of the mean bigWig signal of
CTCF, RAD21 and SMC3 within +-100 bp of the motif centre, of H3K27ac and
H3K4me3 within +-500 bp (activity covariates), and the methylated fraction of
the CpGs wholly inside the 15 bp motif (ENCODE WGBS CpG beds, both strands of
a CpG summed; at least 10 reads, otherwise the site has no value) and of the
CpGs in the flanks (+-250 bp). 82 percent of the sites have no CpG in the
motif; 4,365 (GM12878) down to 291 (IMR-90) even-chromosome sites per cell
have a covered motif CpG (IMR-90 291, HepG2 1,401, CD14+ 1,804, HMEC 1,867,
K562 2,883, A549 3,283, H1 3,964: WGBS depth differs between cells). Only even
chromosomes enter the statistics; intervals resample whole chromosomes.
Tables `motif_statistics.csv`, `motif_models.csv`; figure
`fig_ctcf_height_vs_motif_strength_by_methylation.png`.

Height against strength (Spearman, all even sites): 0.25 to 0.32 for CTCF
over the eight cells (0.24 to 0.28 RAD21, 0.23 to 0.25 SMC3). Among sites
with a covered motif CpG, CTCF height falls with motif methylation (Spearman
-0.36 K562 to -0.74 HMEC, median -0.65); the median log1p height is 2.7 to
3.8 at methylation below 0.2 and 0.3 to 1.0 above 0.8 (K562 3.13 and 0.65).
Partial Spearman correlation of height with motif methylation given motif
strength (cubic in the rank of the strength):

| cell | n sites | given strength | + activity covariates | + flank methylation | flank methylation given strength |
|---|---|---|---|---|---|
| GM12878 | 4365 | -0.46 | -0.44 | -0.33 | -0.38 |
| IMR-90 | 291 | -0.67 | -0.55 | -0.42 | -0.58 |
| H1 | 3964 | -0.64 | -0.60 | -0.43 | -0.53 |
| HepG2 | 1401 | -0.48 | -0.44 | -0.38 | -0.31 |
| A549 | 3283 | -0.60 | -0.55 | -0.44 | -0.46 |
| CD14+ monocyte | 1804 | -0.61 | -0.52 | -0.40 | -0.51 |
| K562 (held out) | 2883 | -0.28 | -0.30 | -0.36 | 0.04 |
| HMEC (held out) | 1867 | -0.68 | -0.64 | -0.48 | -0.57 |

Chromosome-bootstrap intervals of the first column are at most +-0.04 wide
except IMR-90 (-0.77 to -0.55). Methylation at the motif lowers CTCF height
beyond motif strength in every cell, also in the two held-out cells, and the
association remains at -0.33 to -0.48 after the flank methylation and the
H3K27ac and H3K4me3 height of the site are held fixed (the flank column
partials out strength and flank methylation: motif methylation at the site
carries information beyond the surrounding methylation). RAD21 (median
partial -0.53 given strength, -0.39 with flank) and SMC3 (-0.47, -0.40)
behave alike. The association is correlational: bound CTCF protects its
CpGs from methylation, so low methylation can be a consequence of binding;
the data cannot separate the two directions.

Odd-to-even models (boosting, trained on the odd-chromosome sites of the
other training cells with height z-scored within cell, scored on the
even-chromosome sites with a covered motif CpG; Spearman, median over cells;
K562 and HMEC separately in brackets, mean of the two): CTCF strength alone
0.44 (0.44); + motif methylation 0.70 (0.63); + motif and flank methylation
0.69 (0.65); + activity 0.56 (0.53); activity + motif + flank methylation 0.72
(0.67). Motif methylation raises the Spearman correlation by 0.25 over
strength (median over cells) and by 0.14 over strength + activity (minimum
over cells 0.06); the gain on K562 is the smallest (0.43 to 0.52 with motif methylation).
RAD21: 0.43, 0.64, 0.64 (K562 alone 0.55 for the last); SMC3: 0.44, 0.62, 0.62
(K562 0.56).

### 3. Cross-mark relationships

unseen_cell, signed squared Pearson (R2 in the table `metrics_summary.csv`),
median over cells; figure `fig_crossmark_predictability.png`.

| mark | group | S | M | own group (W) | other groups (X) | all 12 (O) | O + S | O + M | O + S + M |
|---|---|---|---|---|---|---|---|---|---|
| CTCF | ctcf_cohesin | 0.27 | 0.17 | 0.60 | 0.35 | 0.68 | 0.70 | 0.68 | 0.69 |
| RAD21 | ctcf_cohesin | 0.29 | 0.12 | 0.75 | 0.47 | 0.75 | 0.75 | 0.74 | 0.74 |
| SMC3 | ctcf_cohesin | 0.30 | 0.20 | 0.76 | 0.57 | 0.78 | 0.78 | 0.77 | 0.77 |
| H3K27ac | active | 0.18 | 0.41 | 0.81 | 0.57 | 0.81 | 0.80 | 0.82 | 0.81 |
| H3K4me1 | active | 0.18 | 0.35 | 0.82 | 0.56 | 0.83 | 0.83 | 0.83 | 0.84 |
| H3K4me2 | active | 0.33 | 0.56 | 0.89 | 0.54 | 0.88 | 0.89 | 0.89 | 0.89 |
| H3K4me3 | active | 0.36 | 0.43 | 0.78 | 0.39 | 0.77 | 0.80 | 0.79 | 0.79 |
| H3K9ac | active | 0.32 | 0.54 | 0.88 | 0.62 | 0.89 | 0.89 | 0.89 | 0.90 |
| H3K36me3 | gene body | 0.22 | 0.46 | 0.57 | 0.45 | 0.64 | 0.67 | 0.67 | 0.68 |
| H3K79me2 | gene body | 0.21 | 0.43 | 0.60 | 0.57 | 0.79 | 0.81 | 0.81 | 0.82 |
| H4K20me1 | gene body | 0.28 | 0.14 | 0.57 | 0.42 | 0.63 | 0.66 | 0.64 | 0.67 |
| H3K27me3 | repressive | 0.23 | 0.21 | 0.11 | 0.54 | 0.53 | 0.58 | 0.64 | 0.67 |
| H3K9me3 | repressive | 0.13 | 0.04 | 0.09 | 0.38 | 0.40 | 0.36 | 0.37 | 0.35 |

The other twelve tracks explain 0.40 (H3K9me3) to 0.89 (H3K9ac, H3K4me2) of
the height of a mark on an unseen cell, against 0.10 to 0.60 for sequence
plus methylation. Adding the other marks to sequence + methylation raises
the squared Pearson by 0.14 (H3K9me3) to 0.45 (H3K4me1). In the other
direction, once the other marks are present neither sequence nor methylation
adds more than a median of 0.02 (sequence) or 0.014 (methylation) for most
marks (`redundancy_gains.csv`; the intervals of 10 of the 13 marks include 0).
Exceptions where methylation adds to the other marks and sequence: H3K27me3
+0.044 (interval 0.008 to 0.089) and H3K36me3 +0.039 (0.008 to 0.081),
H3K9ac +0.003 (0.001 to 0.007, negligible in size); sequence adds to the
other marks for H3K27me3 (+0.07, interval 0.05 to 0.12) and CTCF (+0.017,
0.005 to 0.05), and to the other marks + methylation for H3K27me3 (+0.04),
H3K4me3 (+0.010) and H3K4me1 (+0.004) with intervals above 0 and for H3K27ac
(-0.007) below 0 (sizes under 0.02 except H3K27me3). Methylation therefore carries information on
H3K27me3 and H3K36me3 that the other 12 tracks lack, and none that is
measurable on the cohesin, active and CTCF marks.

Within group against cross group. Active marks are best explained by their
group (W 0.78 to 0.89) and clearly less by the other groups (X 0.39 to 0.62;
adding the own group to X raises the squared Pearson by 0.22 to 0.35); the
CTCF and cohesin group likewise (W 0.60 to 0.76 against X 0.35 to 0.57). Gene
body marks lean on the other groups nearly as much as on their own (W 0.57 to
0.60, X 0.42 to 0.57; H3K79me2 gains 0.17 from the other groups). The two
repressive marks are not predictable from each other (W 0.11 and 0.09) and are
predicted by the other groups (X 0.54 and 0.38; H3K9me3 +0.29, H3K27me3
+0.46). In the combined model with the other marks, sequence and methylation
the permutation drop of R2 when a whole group is shuffled is 1.1 to 1.8 for
the active group on the active marks (H3K4me2 1.80, H3K27ac 1.49, H3K4me3
1.45, H3K9ac 1.17, H3K4me1 1.09), 0.85 to 1.29 for the CTCF and cohesin
group on CTCF, RAD21 and SMC3, 1.39 for the gene body group on H4K20me1, 0.57
for H3K79me2 and 0.44 for H3K36me3, whereas on the active and CTCF and cohesin marks every other
group, sequence and methylation cost under 0.05. The gene body marks also
lean on the active group (H4K20me1 0.22, H3K79me2 0.49, H3K36me3 0.15) and
H3K27me3 on gene body marks (0.20), local methylation (0.25) and
sequence (0.15). The model reads the group, and within a
group the individual marks are replaceable in the prediction of another mark,
which fits the Hi-C finding of section 3 (swaps within a group cost 0.001 to
0.007, across groups up to 0.15). The repressive group is the exception: its
two marks are not interchangeable, which section 3 did not test.

### 4. Generalisation limits

seen_cell to unseen_cell on the same seven training cells, sequence +
methylation model (`generalisation.csv`; median over cells; change of R2 with
interval over cells): CTCF 0.38 to 0.21 (-0.12, -0.34 to -0.06), H3K27ac
0.47 to 0.26 (-0.20), H3K27me3 0.46 to 0.26, H3K36me3 0.55 to 0.22 (-0.33),
H3K4me1 0.45 to 0.18, H3K4me2 0.61 to 0.47, H3K4me3 0.62 to 0.49, H3K79me2
0.50 to 0.45 (-0.07), H3K9ac 0.67 to 0.60 (-0.07), H3K9me3 0.29 to 0.05
(-0.34), H4K20me1 0.35 to 0.20, RAD21 0.42 to 0.26, SMC3 0.39 to 0.30. The
decrease is negative for all 13 marks with intervals below 0. Pearson falls
less (CTCF 0.64 to 0.56, H3K9ac 0.82 to 0.81, H3K36me3 0.76 to 0.61, H3K9me3
0.56 to 0.32) and Spearman by 0.05 to 0.18.

Cell-specific scale. The unseen_cell R2 of the combined model is negative
in 23 of 109 mark-cell pairs, and in every one of them the Pearson
correlation is positive (0.10 to 0.80). Examples: H3K4me3 in HMEC (R2 -1.00,
Pearson 0.60), H3K4me1 in K562 (-1.13, 0.59), H3K27ac in H1 (-1.15, 0.26),
H3K9me3 in K562 (-0.89, 0.28), RAD21 in K562 (-0.44, 0.49), H3K4me3 in
IMR-90 (-0.79, 0.80). R2 is exactly the squared Pearson minus a bias term
(difference of the means) and a scale term (difference of the spreads) in
units of the target variance; the median bias term of the combined model is
0.02 to 0.13 over marks and the median share of bias in the gap between
squared Pearson and R2 is 0.75. A cell whose z-scored track has another mean
level than the training cells (the z-scoring uses fixed training statistics)
therefore loses R2 without losing rank agreement. Fraction of cells with
negative R2: 0.44 (H3K9me3), 0.40 (SMC3), 0.33 (H3K27ac, H3K4me1,
H4K20me1), 0.22 (H3K36me3, H3K4me3), 0.17 (RAD21), 0.11 (H3K27me3, H3K4me2,
H3K9ac), 0 (CTCF, H3K79me2). On K562 and HMEC (R2 / Pearson of the combined
model): H3K9ac 0.51 / 0.77 (K562), 0.44 / 0.73 (HMEC); H3K36me3 0.57 / 0.83
and 0.51 / 0.72; CTCF 0.08 / 0.53 and 0.17 / 0.44. Spearman is below Pearson
for H3K4me3 (0.22 against 0.72) and the other zero-inflated tracks, so the
Pearson values of the active marks lean on a few strong peaks.

### Novel against section 35

(a) The variance is partitioned into unique and shared parts with intervals
over cells; the extended sequence set shows that a part of what section 35
attributed to methylation is shared with GC and CpG context (unique
methylation for H3K27ac 0.15 to 0.09, H3K9ac 0.42 to 0.29) and that for
CTCF, RAD21 and SMC3 the sequence part is the unique one on unseen cells.
(b) The motif-level analysis (new data: CpG-level methylation inside the
motif) shows a strong negative association of motif methylation with CTCF,
RAD21 and SMC3 height beyond motif strength, activity marks and flank
methylation, which the 1 kb Meth track of section 35 cannot resolve and which
is plausibly diluted in the 10 kb bin (section 35 found Pearson 0.42 to
0.52 for CTCF); this dilution was not tested. (c) The
other twelve marks make methylation redundant for most marks, with H3K27me3
and H3K36me3 as exceptions, and the group structure of the predictability
matches the Hi-C group result. (d) The seen_cell to unseen_cell loss and the
bias term of the cell-specific scale are quantified.

### Limitations

1. Five to nine cells; intervals over cells are wide (the interval of the
unseen R2 of H3K4me3 reaches -0.79 because of one cell). RAD21 and SMC3 rest
on 6 and 5 cells (4 and 5 training cells), CTCF on 8; GM23248, CD14+
monocyte and HMEC lack some of the three. K562 and HMEC are two held-out
cells; conclusions about held-out cells rest on two cells.
2. The 10 kb table models are boosting on aggregated features; no sequence
network (planned for a GPU) is included, so "sequence" means composition,
motif counts and context, and its unique part is a lower bound.
3. Permutation importance of correlated groups underestimates each group;
unique parts are differences of R2 values of separate fits and can be
negative (suppression) or biased by the 500,000-row subsample.
4. The z-scoring uses the training statistics of the checkpoint, so R2 on
a new cell contains the bias term; the squared Pearson partition is given
next to the R2 partition for that reason. Squared Pearson is not a held-out
R2 (it assumes a known per-cell affine recalibration) and is a diagnostic.
5. Other marks as predictors contain NaN for missing tracks; models trained
mostly on cells with all 13 tracks meet other missingness patterns in the test cell
(HMEC lacks RAD21 and SMC3, GM23248 lacks CTCF, RAD21 and SMC3).
6. Motif analysis: the PWM threshold 0.6 admits many unbound sites, and only
sites with a CpG wholly inside the 15 bp motif and 10 or more reads enter the
methylation statistics (9 to 13 percent of the sites in the deeper cells, 18 percent of motifs
contain a CpG, a subset with stronger motif scores, so the height-against-
strength correlation of this subset, 0.44, is higher than that of all sites,
0.28). IMR-90 has 291 and HepG2 1,401 such sites. The motif signal is the
bigWig mean in a 200 bp window, not a peak call. The statistics are descriptive; the
odd-to-even models use the cell's z-scored height and the other training
cells; the partial correlations use the even chromosomes of the cell alone.
Methylation at the motif and CTCF binding are not separable causally.
7. The RRBS cells (HCT116, MCF-7, HFF) are excluded; HUVEC has no
methylation.
8. Figures were written but not visually inspected (the permission to view
images was declined during this analysis): `fig_variance_partition.png`,
`fig_crossmark_predictability.png`, `fig_scale_problem.png`,
`fig_ctcf_height_vs_motif_strength_by_methylation.png`. Every number above
was read from the CSV tables.

## 42. How much of ChIP height DNA sequence and methylation explain (FINAL for v1; a per-cell methylation normalisation run is in progress)

Question. How much of the ChIP-seq peak height of the 13 marks (CTCF, RAD21,
SMC3, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K79me2, H3K9ac,
H3K9me3, H4K20me1) at 10 kb bins does DNA sequence explain, how much the
cell's methylation, and how much do the two share. The gradient-boosting
analysis of section 35 part B uses hand-made sequence features (GC, CpG density,
CTCF motif strength; sequence-only Pearson 0.30 to 0.49,
`results/methylation/meth_track_prediction.csv`); this section replaces them by
a sequence CNN.

Script `scripts/seq_cnn_chip.py` (self-contained; needs torch, numpy, scipy,
pandas, pyyaml, pyBigWig, pyfaidx), launcher `scripts/run_seq_cnn_remote.sh`,
defaults `configs/seq_cnn.yaml`. Model: dilated residual 1D CNN, 9.3 M
parameters, one-hot input window of 131072 bp, trunk stride 128 bp, 12 dilated
residual blocks (dilations 1 to 128), exact pooling of the trunk to 8 output
bins of 10 kb (80 kb centred, 25.5 kb context each side), 13-mark multitask
head. Training with bf16 autocast, AdamW, cosine schedule, 20000 steps at batch
32. Target: log1p of the bin mean, z-scored per mark with training statistics;
loss MSE averaged over marks (`--loss poisson` available). Augmentation:
reverse complement (Meth channel and bin order reversed). Bins with an N or
overlapping the ENCODE blacklist v2 are excluded.

Modes (each trained multitask):
- `seq`: sequence only. The prediction is identical for all cells; the training
  loss uses the tracks of all seven training cells, so the optimum is the
  across-cell mean. Scored against each cell's own track, next to the ceiling of
  a cell-agnostic prediction (`ceiling_incl`: mean of the training cells
  including the scored one; `ceiling_loo`: mean of the other training cells).
- `seq_meth`: sequence + two channels at base-pair resolution, the 1 kb Meth
  track of the cell (value - 1 = methylated fraction, bins without data at the
  cell mean) and the genome-wide mean methylation of the cell.
- `meth`: the two methylation channels only (control).
- `--cell-embedding`: learned cell embedding for seen cells. It cannot
  generalise to unseen cells (k562 and hmec are skipped in such a run); it is
  a ceiling for cell-specific information, not a model for new cells.
- `--holdout-cell X`: leave one training cell out; X is scored as unseen.
- `--res 1000`: 1 kb output bins (80 bins per window).

Splits (by chromosome only). Training: odd chromosomes chr1 to chr21 of the
seven training cells (gm12878, imr90, h1, hepg2, a549, gm23248, cd14mono).
Validation: chr2 and chr4 of the training cells, used for model selection.
Test: chr6, 8, 10, 12, 14, 16, 18, 20, 22 for all nine cells, i.e. the training
cells (seen) and k562 and hmec (never trained on). The sets are disjoint and
asserted at start; input windows are clipped to their own chromosome, so no
validation or test base is in a training window. The z-score statistics come
from training chromosomes of training cells only. The number of independent
cells is nine; hmec_deep shares ChIP and Meth with hmec and is not scored
separately. This differs from section 35 part B, which tests on all even
chromosomes.

Commands.
    # 1. smoke test on CPU (minutes)
    python scripts/seq_cnn_chip.py --smoke --mode seq_meth --data-root $DATA_ROOT
    # 2. full run on one GPU (cache, three modes, summary)
    DATA_ROOT=/path/to/hic-multiome OUT_ROOT=./runs/seq_cnn bash scripts/run_seq_cnn_remote.sh
    # 3. optional variants
    EXTRA="--cell-embedding" MODES="seq_meth" bash scripts/run_seq_cnn_remote.sh
    EXTRA="--holdout-cell imr90" MODES="seq_meth" bash scripts/run_seq_cnn_remote.sh
    EXTRA="--res 1000 --steps 40000" MODES="seq seq_meth" bash scripts/run_seq_cnn_remote.sh

Outputs: `results/seq_cnn/seq_cnn_results.csv` (mode, variant, mark, cell,
scheme seen_cell or unseen_cell, split, n_bins, r2, pearson, spearman),
`seq_cnn_summary.csv` (median over cells per mode, scheme and mark, with the
number of cells and of independent cells), `seq_cnn_partition.csv` (R2 of seq,
meth and seq_meth per mark and cell with shared = R2_seq + R2_meth - R2_seq_meth,
unique_seq = R2_seq_meth - R2_meth, unique_meth = R2_seq_meth - R2_seq).

What answers what.
- R2_seq and Pearson of `seq` against `ceiling_incl`: how much of ChIP height
  sequence explains, and how much of the cell-agnostic part it reaches.
- `seq_meth` against `seq` on k562 and hmec (unseen_cell): the gain from
  methylation for cells never trained on; against `meth`: the gain from sequence.
- `shared`, `unique_seq`, `unique_meth` of the partition: the overlap of the two
  information sources per mark.
- `seq` and `seq_meth` against the gradient-boosting values of section 35 part
  B: whether a sequence CNN improves on hand-made features.
- seen_cell against unseen_cell with `--holdout-cell`: the cost of an unseen
  cell.
Results of the v1 run (`results/seq_cnn/seq_cnn_results.csv`,
`seq_cnn_summary.csv`, `seq_cnn_partition.csv`, copied from the GPU machine, run
log `run_20261009_165303.log` there; RTX 4090, 2026-10-09 16:53 to 18:49; 20000
steps per mode, 9.27 M parameters, 93,428 test bins per cell and mark). Modes
`seq`, `seq_meth` and `meth` were trained once each; best validation Pearson
(chr2 and chr4 of the training cells) 0.5511 (seq), 0.7879 (seq_meth), 0.7237
(meth). The cell-embedding, hold-out-cell and 1 kb variants were not run.

Seen cells (odd-chromosome training, test chr6 to chr22 even, seven training
cells; R2, median over cells; ceilings are the cell-agnostic across-cell mean of
the training cells including (incl) or excluding (loo) the scored cell; GBM =
gradient boosting of section 41 with the extended sequence set, seen_cell,
median over cells of all even chromosomes):

| mark | cells | CNN seq | CNN meth | CNN seq+meth | ceiling incl | ceiling loo | GBM seq | GBM meth | GBM seq+meth |
|---|---|---|---|---|---|---|---|---|---|
| CTCF | 6 | 0.28 | 0.34 | 0.44 | 0.77 | 0.67 | 0.26 | 0.24 | 0.38 |
| RAD21 | 5 | 0.34 | 0.40 | 0.53 | 0.82 | 0.72 | 0.26 | 0.28 | 0.42 |
| SMC3 | 4 | 0.39 | 0.51 | 0.62 | 0.76 | 0.58 | 0.28 | 0.28 | 0.39 |
| H3K27ac | 7 | 0.28 | 0.60 | 0.69 | 0.54 | 0.37 | 0.13 | 0.46 | 0.47 |
| H3K27me3 | 7 | 0.35 | 0.55 | 0.65 | 0.59 | 0.45 | 0.20 | 0.36 | 0.46 |
| H3K36me3 | 7 | 0.39 | 0.62 | 0.73 | 0.71 | 0.60 | 0.16 | 0.51 | 0.55 |
| H3K4me1 | 7 | 0.25 | 0.62 | 0.70 | 0.54 | 0.37 | 0.15 | 0.40 | 0.45 |
| H3K4me2 | 7 | 0.48 | 0.74 | 0.82 | 0.71 | 0.60 | 0.32 | 0.55 | 0.61 |
| H3K4me3 | 7 | 0.57 | 0.77 | 0.87 | 0.77 | 0.68 | 0.35 | 0.52 | 0.62 |
| H3K79me2 | 7 | 0.43 | 0.61 | 0.70 | 0.76 | 0.68 | 0.20 | 0.45 | 0.50 |
| H3K9ac | 7 | 0.51 | 0.75 | 0.82 | 0.75 | 0.65 | 0.32 | 0.64 | 0.67 |
| H3K9me3 | 7 | 0.19 | 0.44 | 0.52 | 0.55 | 0.39 | 0.09 | 0.15 | 0.29 |
| H4K20me1 | 7 | 0.34 | 0.43 | 0.61 | 0.66 | 0.54 | 0.22 | 0.25 | 0.35 |
| median over marks | | 0.35 | 0.60 | 0.69 | 0.71 | 0.60 | 0.22 | 0.40 | 0.46 |

Pearson, median over marks: seq 0.63 (0.50 H3K9me3 to 0.80 H3K4me3), meth 0.78,
seq_meth 0.83 (0.67 CTCF to 0.93 H3K4me3), ceiling incl 0.89.

Partition into shared, unique sequence and unique methylation (seen cells,
median over cells; shared = R2 seq + R2 meth - R2 seq+meth, unique seq = R2
seq+meth - R2 meth, unique meth = R2 seq+meth - R2 seq):

| mark | R2 seq | R2 meth | R2 seq+meth | shared | unique seq | unique meth | unexplained |
|---|---|---|---|---|---|---|---|
| CTCF | 0.28 | 0.34 | 0.44 | 0.16 | 0.10 | 0.17 | 0.56 |
| RAD21 | 0.34 | 0.40 | 0.53 | 0.22 | 0.12 | 0.22 | 0.47 |
| SMC3 | 0.39 | 0.51 | 0.62 | 0.29 | 0.11 | 0.23 | 0.38 |
| H3K27ac | 0.28 | 0.60 | 0.69 | 0.24 | 0.06 | 0.46 | 0.31 |
| H3K27me3 | 0.35 | 0.55 | 0.65 | 0.16 | 0.11 | 0.40 | 0.35 |
| H3K36me3 | 0.39 | 0.62 | 0.73 | 0.28 | 0.10 | 0.31 | 0.27 |
| H3K4me1 | 0.25 | 0.62 | 0.70 | 0.16 | 0.09 | 0.45 | 0.30 |
| H3K4me2 | 0.48 | 0.74 | 0.82 | 0.39 | 0.07 | 0.33 | 0.18 |
| H3K4me3 | 0.57 | 0.77 | 0.87 | 0.54 | 0.07 | 0.25 | 0.13 |
| H3K79me2 | 0.43 | 0.61 | 0.70 | 0.30 | 0.08 | 0.25 | 0.30 |
| H3K9ac | 0.51 | 0.75 | 0.82 | 0.46 | 0.06 | 0.31 | 0.18 |
| H3K9me3 | 0.19 | 0.44 | 0.52 | 0.10 | 0.09 | 0.39 | 0.48 |
| H4K20me1 | 0.34 | 0.43 | 0.61 | 0.22 | 0.14 | 0.21 | 0.39 |
| median over marks | 0.35 | 0.60 | 0.69 | 0.24 | 0.09 | 0.31 | 0.31 |

Unseen cells K562 and HMEC (never in training; R2 and Pearson r, median over the
cells that have the mark, i.e. the mean of K562 and HMEC; the GBM columns are the
section 41 mean of the two held-out cells):

| mark | cells | seq R2 | seq r | seq+meth R2 | seq+meth r | meth R2 | meth r | ceiling R2 | ceiling r | GBM seq R2 | GBM seq+meth R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CTCF | 2 | 0.19 | 0.46 | -0.19 | 0.44 | -4.87 | 0.07 | 0.68 | 0.83 | 0.16 | 0.12 |
| RAD21 | 1 | 0.30 | 0.57 | -1.38 | 0.43 | -84.14 | -0.19 | 0.61 | 0.82 | 0.24 | -0.44 |
| SMC3 | 1 | 0.44 | 0.67 | -3.69 | 0.34 | -11.63 | -0.19 | 0.62 | 0.79 | 0.36 | -0.04 |
| H3K27ac | 2 | 0.23 | 0.56 | -7.07 | 0.37 | -7.93 | 0.31 | 0.39 | 0.71 | 0.08 | 0.15 |
| H3K27me3 | 2 | 0.24 | 0.53 | -1.75 | 0.24 | -18.66 | 0.45 | 0.47 | 0.71 | 0.17 | 0.28 |
| H3K36me3 | 2 | 0.38 | 0.62 | 0.16 | 0.68 | -7.34 | 0.73 | 0.69 | 0.84 | 0.17 | 0.54 |
| H3K4me1 | 2 | 0.21 | 0.57 | -5.19 | 0.70 | -47.91 | 0.19 | 0.38 | 0.74 | 0.07 | -0.33 |
| H3K4me2 | 2 | 0.53 | 0.74 | -18.02 | 0.44 | -94.32 | 0.28 | 0.69 | 0.83 | 0.33 | 0.59 |
| H3K4me3 | 2 | 0.24 | 0.72 | -23.92 | 0.40 | -85.62 | 0.28 | 0.38 | 0.84 | -0.02 | -0.26 |
| H3K79me2 | 2 | 0.43 | 0.68 | -0.67 | 0.60 | -33.74 | 0.12 | 0.73 | 0.87 | 0.20 | 0.39 |
| H3K9ac | 2 | 0.45 | 0.70 | -15.80 | 0.35 | -39.22 | 0.23 | 0.64 | 0.82 | 0.23 | 0.48 |
| H3K9me3 | 2 | 0.12 | 0.43 | -8.96 | 0.33 | -30.19 | 0.20 | 0.28 | 0.66 | 0.15 | -0.34 |
| H4K20me1 | 2 | 0.13 | 0.56 | -0.46 | 0.43 | -31.41 | 0.02 | 0.32 | 0.70 | 0.12 | -0.09 |

Per cell, median over marks (Pearson): seq 0.66 (K562) and 0.58 (HMEC), seq_meth
0.34 and 0.74, meth -0.20 and 0.67; the seven seen cells reach 0.57 to 0.65 (seq),
0.80 to 0.86 (seq_meth) and 0.68 to 0.81 (meth). R2 per mark and held-out cell,
seq_meth: K562 -0.20 (H4K20me1) to -45.7 (H3K4me3), HMEC 0.16 (CTCF) to 0.58
(H3K79me2) for nine of eleven marks, H3K4me3 -2.18 and H4K20me1 -0.71; meth
alone: K562 -9.7 to -189, HMEC -3.90 (H3K4me3) and -2.21 (H4K20me1) for two
marks and -0.10 to 0.57 for the other nine.

Reading. Sequence alone: the CNN explains more than the hand-made features in
every mark on the seen cells (R2 median over marks 0.35 against 0.22; 13 of 13
marks) and in 12 of 13 on K562 and HMEC (all but H3K9me3; median 0.24 against
0.17). It reaches 0.34 (H3K9me3) to 0.75 (H3K4me3) of the R2 of the
cell-agnostic ceiling on seen cells, so about half of the cell-agnostic part of
ChIP height is a function of the sequence in a 131 kb window, and the
sequence-only prediction of a new cell reaches R2 0.12 to 0.53 and Pearson 0.43
to 0.74. Methylation: with both inputs the seen-cell R2 is 0.44 (CTCF) to 0.87
(H3K4me3), median 0.69 against 0.46 for boosting, and it exceeds the
cell-agnostic ceiling for seven marks (H3K27ac, H3K27me3, H3K36me3, H3K4me1,
H3K4me2, H3K4me3, H3K9ac), i.e. the methylation channel carries cell-specific
height beyond the across-cell mean there. The partition on seen cells gives
unique methylation 0.17 to 0.46 (median 0.31), unique sequence 0.06 to 0.14
(median 0.09) and shared 0.10 to 0.54 (median 0.24); section 41 found unique
methylation 0.11 to 0.39 and unique sequence 0.02 to 0.14, so the CNN raises the
shared and the unique methylation parts and leaves the unique sequence part at
about the boosting size. CTCF, H4K20me1, RAD21 and SMC3 have the smallest unique
methylation part (0.17 to 0.23); the unexplained part is largest for CTCF
(0.56), H3K9me3 (0.48) and RAD21 (0.47) and smallest for H3K4me3, H3K4me2 and
H3K9ac (0.13 to 0.18).

On the unseen cells the methylation modes fail. seq_meth reaches a positive R2
only for H3K36me3 (0.16), meth alone for none (-4.9 to -94), against boosting
with 7 of 13 positive for seq+meth (e.g. H3K4me2 0.59, H3K36me3 0.54, H3K9ac
0.48); the CNN's seq_meth is below the boosting value in 13 of 13 marks on the
mean of the two cells, and below the sequence-only CNN in 13 of 13 marks. The
Pearson shows where: on HMEC seq_meth keeps 0.74 median over marks (above seq
0.58; R2 above seq in 8 of 11 marks, tied in 1, below in 2) and on K562 it
collapses to 0.34 (meth alone -0.20; seq_meth correlates negatively for 3 of
13 marks and between 0 and 0.1 for 3 more, meth alone negatively for 11 of 13),
so the summary over two cells is dominated by K562. A cell-agnostic
sequence model therefore transfers to new cells, and the methylation inputs of
this v1 model do not transfer to K562; the cause was not diagnosed in this run
(the two cells are the only evidence; K562 is a leukaemia line and its
methylation distribution may lie outside the seven training cells, which was not
tested). A follow-up run is in progress that normalises the methylation inputs
per cell.

Limitations. (1) The number of independent cells is 2 for the unseen scheme
(K562, HMEC), 1 for RAD21 and SMC3 (HMEC lacks them), and 7 for the seen scheme
(CTCF 6, RAD21 5, SMC3 4, where GM23248, CD14+ monocyte or H1 lack the track);
no interval over cells can be given. The `n_cells` and `n_independent_cells`
columns of `seq_cnn_summary.csv` show 7 and 2 for every mark because they count
the empty rows of cells without the track (n_bins 0); the medians skip those
rows, the counts above are from the non-empty rows. (2) No ENCODE blacklist was
used (the run logged a warning that no blacklist file was present; only bins with
an N were excluded), against the method text above that excludes blacklisted
bins; blacklisted regions with extreme signal remain in training and test. (3)
One training run per mode, no seed repeat and no interval; the validation
Pearson of the seq mode swings between 0.41 and 0.55 in the first 9000 steps and
stays at 0.535 to 0.549 from step 10000 on, and the checkpoint is chosen on it
(best 0.5511 at step 8000), so single-run differences of a few hundredths in R2
are not resolved.
(4) The CNN is scored on chr6, 8, 10, 12, 14, 16, 18, 20, 22, section 41 on all
even chromosomes (chr2 and chr4 are the CNN's validation chromosomes), so the
CNN and boosting columns are not on identical bins. (5) The seen-cell scores
use cells whose other chromosomes were trained on; the scheme tests new
positions, not new cells. (6) R2 on unseen cells contains the cell-specific
scale and bias (section 41, part 4); the Pearson columns are given for that
reason. (7) Only the v1 model was run: no cell embedding, no hold-out of a
training cell, no 1 kb resolution, and no Poisson loss.

## 43. Methylation-aware directed CTCF motif planes (FINAL for all-track input; sparse input in section 44)

Question. Section 41 found that CpG methylation inside the CTCF motif lowers the
CTCF ChIP height beyond the motif score (partial Spearman -0.46 to -0.68 with the
motif strength held fixed), while the 1 kb Meth track averaged to the 10 kb bin
dilutes that signal. Does giving the Hi-C model the methylation of each motif site
improve the prediction over the Meth track alone.

Input option `motifs.motif_meth` (default off; `motif_meth_id` MA0139.2). Code:
`hicmultiome/motifmeth.py`, `precompute.assemble_chrom`, `config.py`, `train.py`
(row selection and dropout coupling), `scripts/crosscell_predict.py` (absent-Meth
sets), `scripts/motif_meth.py`. Per cell and chromosome, every CTCF hit of the
model's motif scan (PWM >= 0.8 of the maximum, the channel threshold; start
positions from `motifs.cached_motif_positions`) gets the weight

    w = 1 - (methylated reads / reads) over the CpGs wholly inside the motif,
        reads pooled over the CpGs, at least 10 reads (MIN_SITE_COV as in section 41);
    w = 1 for a site without covered motif CpG (no CpG in the motif or coverage
        below 10 reads), the unmodified hit.

A bin holds the sum of w over the forward (reverse) hits starting in it, so a cell
without CpG data has planes equal to the unmodified hit counts. The planes enter as
one more directed motif pair, pseudo motif `MA0139.2:meth`, next to the unmodified
pair, so the fwd_i * rev_j conditioning planes, the reverse augmentation and the
row selection of `select_tracks` / `select_motifs` apply unchanged. Because w = 1
means both "unmethylated" and "no information", a conditioning row
`motifmeth_covered` (sequence-feature block) carries the fraction of the bin's hits
with measured motif methylation (0 without CpG data and where Meth is absent).
Indicator rather than mean imputation: the indicator keeps the unmodified strength
exact and lets the network learn what an uncovered site means. CpG tables are the
per-chromosome arrays of `scripts/chip_seq_meth_relationships.py --stage cpg`
(`<chrom>_pos`, `_cov`, `_meth`), copied by `scripts/motif_meth.py --stage cpg` to
`meth/<cell>/CpG.npz` of the nine WGBS cells; every other cell (NHEK, PC-9,
HUVEC, the RRBS cells HCT116, MCF-7, HFF) gets the unmodified planes and covered = 0.

Honesty. The planes are inputs derived from the cell's own WGBS and its genome; no
Hi-C or reference enters them. Normalisation statistics are fit on training
chromosomes of the training cells like every other row; held-out cells (K562,
HMEC, HUVEC) are never trained on. Track dropout removes the methylation of the
planes together with the Meth track: in a window that drops Meth the weighted pair
is replaced by the unmodified pair (rescaled to its own z-score statistics, the
reverse-augmented windows with the swapped plane's statistics) and the covered row
by its absent value, which is exactly the input of a cell without CpG data.
`crosscell_predict --keep-sets` without Meth does the same, so absent-Meth scoring
cannot leak methylation through the planes. Edited-sequence prediction
(`variants.predict_edited`) raises for such a model.

Meth reaching the network (regression of the 2026-10-02 select_tracks bug,
`tests/test_motif_meth.py`, CPU, 8 tests, plus the 177 tests of the suite): a
select_tracks list with Meth keeps the Meth row, the weighted pair and the covered
row, one without it drops the Meth row (asserted); a random-weight model gives a
different output when the CpG-derived rows change, when the Meth row changes, and
the input channel count exceeds that of the option-off model; the per-site weights
equal an independent loop on a synthetic genome with planted sites; a normalised
CpG cell after `drop_methylation` equals the same cell without CpG file; the torch
dropout path equals it for plain and reverse-augmented windows; an end-to-end
precompute plus train with track dropout 0.5 over three cells (CpG, Meth without
CpG, no Meth) runs and logs the coupling.

Runs (scheduler-style names, `scripts/multicell_expanded.py`):
`all12refres_drop_motifmeth_25kb` and `_10kb` (residual on the leave-one-out
consensus, track dropout 0.5, the 12 training cells, the nine WGBS cells carry
Meth and CpG as in all12refres_drop_meth), caches `mc_all12refres_drop_motifmeth_<k>kb`,
queued in `scripts/gpu_queue_extra.py` (block a) with the standard evaluation, so
the rows land in `reference_summary.csv`.

Results (`/mnt/processed/wolffjoa/hic-multiome/runs/multicell_reference/
reference_summary.csv` and `reference_compare.csv`, read 2026-10-10; the copies
in `results/multicell_reference/` predate these runs). Training stopped at
epoch 49 (best validation Pearson 0.9380) at 25 kb against epoch 62 (0.9401) for
all12refres_drop_25kb, and at epoch 72 (0.9126) at 10 kb against epoch 69
(0.9123) for all12refres_drop_10kb and 69 (0.9125) for all12refres_drop_meth_10kb.
Median Pearson on the 413 Akita test regions over the 12 training cells and
over the five held-out cells (K562, HMEC, HUVEC, and the deeper maps HMEC deep
and HUVEC deep, scored as separate cells):

| res | model | 12 training cells | 5 held-out cells | held-out cells above the consensus |
|---|---|---|---|---|
| 25 kb | consensus alone | 0.9223 | 0.8977 | |
| 25 kb | all12refin_drop | 0.9362 | 0.9137 | 2/5 |
| 25 kb | all12refres_drop | 0.9358 | 0.9072 | 1/5 |
| 25 kb | all12refres_drop_motifmeth | 0.9322 | 0.8949 | 1/5 |
| 10 kb | consensus alone | 0.8867 | 0.8586 | |
| 10 kb | all12refin_drop | 0.9266 | 0.9028 | 4/5 |
| 10 kb | all12refres_drop | 0.9270 | 0.8972 | 4/5 |
| 10 kb | all12refres_drop_meth (Meth track only) | 0.9270 | 0.8834 | 2/5 |
| 10 kb | all12refres_drop_motifmeth | 0.9307 | 0.8870 | 4/5 |

No 25 kb model with the Meth track alone exists (section 35), so at 25 kb the
comparison is with the model without any methylation input.

Held-out cells, median Pearson per cell (25 kb: consensus / all12refres_drop /
motifmeth; 10 kb: consensus / all12refres_drop / meth / motifmeth):

| cell | 25 kb | 10 kb |
|---|---|---|
| K562 | 0.8972 / 0.9213 / 0.9246 | 0.8586 / 0.9053 / 0.9064 / 0.9071 |
| HMEC | 0.8977 / 0.8836 / 0.8853 | 0.8094 / 0.8089 / 0.8019 / 0.8082 |
| HMEC deep | 0.9324 / 0.9149 / 0.9192 | 0.9065 / 0.9081 / 0.9048 / 0.9113 |
| HUVEC | 0.8869 / 0.8802 / 0.8651 | 0.7689 / 0.7840 / 0.7765 / 0.7755 |
| HUVEC deep | 0.9159 / 0.9072 / 0.8949 | 0.8836 / 0.8972 / 0.8834 / 0.8870 |

Reading. With all 13 ChIP tracks present the planes do not improve the
prediction. Against the model without them the held-out median is lower by
0.0123 at 25 kb and 0.0102 at 10 kb, the training-cell median lower by 0.0036
at 25 kb and higher by 0.0037 at 10 kb, all about the size of the seed noise of
the median Pearson (0.013, section 15) or smaller. Against the model with the
Meth track alone (10 kb) the planes add 0.0036 on the held-out median and
0.0037 on the training cells. Split by whether the cell has a CpG table (the
nine WGBS cells: seven training cells, K562, HMEC and HMEC deep), the change
against the model without planes is median +0.0033 (range +0.0017 to +0.0043)
at 25 kb and +0.0018 (-0.0007 to +0.0032) at 10 kb on the three held-out cells
with CpG data, and -0.0137 and -0.0094 on HUVEC and HUVEC deep, which have no
CpG table and see the unmodified planes with the covered row at 0; on the
training cells the change is -0.0010 (CpG, 7 cells) and -0.0037 (no CpG, 5
cells) at 25 kb and +0.0024 and -0.0001 at 10 kb. The held-out median is
therefore lowered by the two HUVEC maps, in a cell that cannot use the new
input, while the cells that can use it move by +0.002 to +0.003 at the median.
Against the Meth track alone (10 kb) the planes change the cells with CpG data
by +0.0063 (held-out, range +0.0007 to +0.0065) and +0.0028 (training, +0.0004
to +0.0056). The gains on the CpG cells are below the 0.013 seed noise; the
HUVEC loss (-0.0151 and -0.0123 at 25 kb, -0.0085 and -0.0102 at 10 kb) is
above it at 25 kb only. The test of the planes where the
ChIP tracks are absent is section 44 (pending); the dropout coupling is built
for that case.

Limitations. One training run per model and resolution, so no estimate of the
seed noise of these specific differences. The standard evaluation reports the
Pearson only; no structure measure (insulation, loops, compartments) was
computed for these two models, so a gain in structure at equal Pearson is not
excluded. The five held-out cells are three biological cells (K562, HMEC,
HUVEC), HMEC and HUVEC each counted twice through their deeper maps; the
CpG-cell subsets have two or three entries per resolution. Only the
presence of the CpG table separates the HUVEC loss from the planes' effect on
the other cells in this comparison, not an ablation: no run tests the trained
model with the planes replaced by the unmodified pair at inference, so the
data do not show how much the network uses the methylation weighting. The
10 kb comparison with the Meth track alone is the only one that holds the
rest of the recipe fixed; the 25 kb baseline has no Meth track, so that
difference combines the Meth track and the planes.

## 44. Sparse input sets without the consensus, with and without Meth (PENDING, queued)

Question. For a user who has only a few tracks, which input sets give a usable
prediction without any reference map, and what does the methylation channel add to
each. Sets: meth only; CTCF + Meth; CTCF + H3K27ac; CTCF + H3K27ac + H3K4me3; the
five marks CTCF, H3K27ac, H3K4me3, H3K27me3, H3K36me3; the last three with and
without Meth. Reference rows: all 13 ChIP tracks, 13 + Meth, sequence only.

Design (`scripts/scenario_bench.py`). No reference map enters any model. Base
model `all12honest_drop_meth` (12 training cells, honest validation, track dropout
0.5, 14th track Meth with the section 35 v2 cell set), 25 and 10 kb, seeds 0 and 1
(`_s1`, same cache, different training seed and validation draw). Two ways to
score a set on identical regions and metric: (infer) the base model with the other
tracks absent (`crosscell_predict --keep-sets`, as section 35 part A), and
(dedicated) a model trained on the set alone (`select_tracks` = the set, motifs
kept, track dropout 0.5, honest validation) as the dedicated runs of sections 20
and 35 part C but without the consensus input. Evaluation: the 413 Akita test
regions and the 295 (25 kb) and 311 (10 kb) disjoint test windows of the 12-cell
cache, for K562, HMEC (deep) and HUVEC (deep) held out and seven training cells with
WGBS; HUVEC has no methylation, so only the sets without Meth are scored there.
Median per-region Pearson of the O/E maps with test-time augmentation; the measured
consensus is scored alongside as a floor and is never an input.

Crossing with the motif planes (added 2026-10-09). Methylation inside a CTCF motif
marks where CTCF cannot bind (necessary, not sufficient), so with the CTCF track
missing the planes of section 43 are the only route by which CTCF site information
still enters. A second honest base model, `all12honest_drop_motifmeth` (the base
model plus `motifs.motif_meth`; 25 kb seeds 0 and 1, 10 kb seed 0), is scored
inference-only on three sets with the planes on and with their methylation removed
(`crosscell_predict` pseudo track NOMM, same model and tracks, set suffix `_nomm`):
H3K27ac + H3K4me3 + Meth (no CTCF track), Meth only, CTCF + H3K27ac + Meth. Because
track dropout trains the model on random track subsets, these sets are in
distribution; the rows are also set against the model trained without planes
(`scenario_motifmeth_vs_off.csv`). No dedicated training is made for these rows.

Budget (GPU hours, in `runs/coarse_gap/QUEUE.md`): base models 4.0; motif_meth base
models 3.0; dedicated at 25 kb all 8 sets with 2 seeds 17.6 (25 kb total 21.6, under
30); dedicated at 10 kb for the key sets meth_only, ctcf_meth, ctcf_h3k27ac_meth with
seed 0 5.4; total 30.0. The 10 kb dedicated five_marks_meth row (1.8 h) was dropped
to absorb the motif_meth base models. Dedicated trainings exist only where inference-only scoring
needs a check: the table `scenario_infer_vs_dedicated.csv` gives the difference
per set. Outputs: `results/scenario_bench/scenario_cells.csv`,
`scenario_summary.csv`, `scenario_infer_vs_dedicated.csv`. Results: pending.

## 45. AML-inclusive training with the reference as residual (PENDING, queued)

Question. In the AML-inclusive recipe of section 36 the AML reference map (mean
proper O/E of the training patients) enters as an input plane (`model.reference`
input). With `residual` the network predicts patient minus AML reference and the
reference is also added to the output, so the target of the backbone is the
deviation from the AML reference alone.

Implementation. Run `all12aml_refres_drop` (fold 0, 10 kb) in
`scripts/multicell_expanded.py`: the config of `all12aml_refin_drop` with
`model.reference: residual`; it reads the same cache (`mc_all12aml_ref_<k>kb`: same
windows, AML reference, lab indicator, oversampling x3, patient-specific hard
validation), so no window is recomputed and the held-out patients and HSPC donors
stay held out. `scripts/aml_train_eval.py` scores it as `aml_refres` next to
`aml_refin` and the other models of fold 0 (queue block c). The 25 kb run and
folds 1 and 2 are not queued. Results: pending.

## 46. CTCF binding, motif CpG methylation and insulation in the Xu et al. AML patients (FINAL; sections 42 to 45 were taken by another agent, this is the next free number)

`scripts/aml_ctcf_methylation.py` (CPU), tables and figures in
`results/aml_ctcf_methylation/`. Question: does the section 41 finding (CpG
methylation inside the CTCF motif lowers CTCF height beyond motif strength) hold in
the primary AML patients of Xu et al. 2022, and does it have a Hi-C consequence.

**Data.** Sites are the 72,626 motif hits of section 41 (hg38). Patients with both
CTCF CUT&Tag and WGBS: 018, 027, 103, 168, 270, 472, 546, 629, 773 (9; 673 has WGBS
but no CTCF CUT&Tag, 1916 and 1360 have CUT&Tag but no WGBS). All 9 have Hi-C
(10 kb O/E maps `aml/oe/xu_*_proper_10kb.cool`, the maps of the AML scoring scripts),
and 4 HSPC donors serve as the Hi-C baseline. H3K27ac CUT&Tag exists for all 9;
H3K4me3 was not measured in the patients, so the activity covariate is H3K27ac alone.
Patient WGBS is the GEO processed bigWig (CG methylation per 10 bp bin, no read
counts): motif methylation is the unweighted mean of the 10 bp bins that hold a CpG
of the 15 bp motif (the bin of the C and of the G; a bin can also hold a CpG outside
the motif), flank methylation the mean of the other covered bins within +-250 bp.
The reads >= 10 rule of section 41 cannot be applied to the patients. Of the 72,626
sites 13,729 hold a CpG in the motif; 13,142 to 13,324 of these have a covered
bin in each patient (6.1 to 6.2 thousand on the even chromosomes, against 291 to
4,365 per non-AML cell in section 41). CTCF height is log1p of the mean MACS2
-log10 p-value signal within +-100 bp of the motif centre, H3K27ac within +-500 bp.
Healthy references: no HSPC or PBMC WGBS exists in the repo or in the GEO deposit, so
the healthy methylation reference is ENCODE CD14+ monocyte WGBS (CpG level, reads >=
10, section 41 table); the HSPC CTCF reference is the CD34+ ChIP-seq SRX037935
(ChIP-Atlas, 9.8 M reads), a different assay and laboratory than the CUT&Tag, so
CTCF is compared by rank, never by value. Sample 773 (extreme CpG-island
methylation in the paper) is kept and every pooled number is given with and without it.

### 1. Methylation at the motif CpGs against CTCF height (even chromosomes)

Partial Spearman correlation of CTCF height with motif methylation given a cubic in
the rank of motif strength, exactly the section 41 statistic (the code is imported
from `chip_seq_meth_relationships.py`; the recomputation on the 8 non-AML cells
reproduces the section 41 column: GM12878 -0.46, IMR-90 -0.67, H1 -0.64, HepG2 -0.48,
A549 -0.60, CD14+ -0.61, K562 -0.28, HMEC -0.68). Chromosome-bootstrap intervals
(300 resamples) of the first column in brackets. Table `q1_partial_spearman.csv`,
`q1_per_patient_even.csv`.

| patient | n sites | Spearman height vs strength (all sites) | given strength | + H3K27ac | + flank methylation | + flank + H3K27ac | flank given strength |
|---|---|---|---|---|---|---|---|
| 018 | 6122 | 0.14 | -0.18 (-0.20 to -0.15) | -0.15 | -0.06 | -0.07 | -0.18 |
| 027 | 6138 | 0.28 | -0.58 (-0.61 to -0.52) | -0.55 | -0.29 | -0.31 | -0.54 |
| 103 | 6205 | 0.22 | -0.39 (-0.42 to -0.36) | -0.34 | -0.16 | -0.17 | -0.38 |
| 168 | 6131 | 0.18 | -0.27 (-0.31 to -0.23) | -0.20 | -0.12 | -0.12 | -0.26 |
| 270 | 6138 | 0.13 | -0.18 (-0.21 to -0.16) | -0.15 | -0.08 | -0.08 | -0.18 |
| 472 | 6162 | 0.13 | -0.25 (-0.28 to -0.22) | -0.19 | -0.11 | -0.10 | -0.24 |
| 546 | 6133 | 0.26 | -0.51 (-0.55 to -0.45) | -0.47 | -0.24 | -0.25 | -0.49 |
| 629 | 6199 | 0.21 | -0.37 (-0.40 to -0.34) | -0.30 | -0.14 | -0.15 | -0.38 |
| 773 | 6165 | 0.25 | -0.52 (-0.56 to -0.47) | -0.50 | -0.22 | -0.24 | -0.52 |

Pooled (median over samples, the same chromosomes resampled in every sample, 300
resamples; `q1_pooled.csv`; all-chromosome values differ by at most 0.03):

| group | given strength | + H3K27ac | + flank | + flank + H3K27ac | flank given strength |
|---|---|---|---|---|---|
| 9 AML patients | -0.37 (-0.40 to -0.33) | -0.30 (-0.33 to -0.25) | -0.14 (-0.18 to -0.11) | -0.15 (-0.19 to -0.11) | -0.38 |
| 8 AML patients without 773 | -0.32 (-0.35 to -0.30) | -0.25 | -0.13 | -0.13 | -0.32 |
| 8 non-AML cells of section 41 (H3K27ac only as covariate) | -0.60 (-0.62 to -0.57) | -0.58 | -0.41 | -0.42 | -0.48 |

Methylation at the motif lowers CTCF height beyond motif strength in every patient
(all nine intervals exclude 0), and the association persists with flank methylation
and H3K27ac fixed (-0.06 to -0.31), but it is weaker than in the non-AML cells
(median -0.37 against -0.60). Six of the nine patients are weaker than every
non-AML cell except K562 (-0.28); 027 (-0.58), 773 (-0.52) and 546 (-0.51) lie
within the non-AML range. Median log1p CTCF height at motif methylation below 0.2
and above 0.8: 027 1.87 and 0.31, 773 1.68 and 0.34, 546 1.31 and 0.38, 103 0.87 and
0.33, 629 0.72 and 0.24, 018 0.47 and 0.42 (non-AML: GM12878 3.60 and 0.30, K562 3.12 and 0.65; table
`q1_height_by_methylation_level.csv`). The patients differ by CUT&Tag signal
quality: height against motif strength alone is 0.13 to 0.28 (018, 270, 472 0.13 to
0.14) against 0.25 to 0.32 in the non-AML cells, and across the 9 patients the partial
correlation follows this value (Spearman -0.95). Part of the weaker AML association
is therefore measurement (CUT&Tag depth, unweighted 10 bp bin methylation without
read counts, a bin that can mix motif and flank CpGs) and the data do not say how
much; for the patients with good CUT&Tag (027, 546, 773) the partial is -0.51 to -0.58.

### 2. Sites that gain methylation against healthy and lose CTCF

Gain: motif methylation in the patient at least 0.3 above CD14+ monocytes (reads >= 10
there). HSPC-bound: the site is in the top 20 percent of all 72,626 sites by HSPC CTCF
height (percentile >= 0.8). Lost, definition A (pre-set): the patient percentile of
CTCF height is at least 0.3 below the HSPC percentile. Definition B (added because the
number of A-lost sites differs between patients with CUT&Tag depth, 1,868 for 027 to
6,226 for 018): the patient height is in the lowest quarter among the HSPC-bound
sites (same prevalence in every patient). Testable: HSPC-bound with a covered motif CpG
in the patient and in CD14+. Tables `q2_*.csv`; 18 definitions (A: drop 0.2, 0.3, 0.4,
B: lowest 0.15, 0.25, 0.35, each with bound 0.7, 0.8, 0.9) in `q2_2x2_grid.csv`.

Pooled over the 9 patients (chromosome-bootstrap intervals, 200 resamples of the same
chromosomes in all patients; `q2_2x2_pooled.csv`):

| definition, patients | gain and lost | gain, kept | no gain, lost | no gain, kept | precision P(lost given gain) | P(lost given no gain) | relative risk | sensitivity P(gain given lost) |
|---|---|---|---|---|---|---|---|---|
| A, 9 | 208 | 274 | 3091 | 12583 | 0.43 (0.35 to 0.49) | 0.20 | 2.19 (1.80 to 2.52) | 0.063 (0.056 to 0.071) |
| A, 8 without 773 | 136 | 172 | 2991 | 11061 | 0.44 (0.33 to 0.54) | 0.21 | 2.07 (1.53 to 2.52) | 0.044 (0.036 to 0.052) |
| B, 9 | 226 | 256 | 2523 | 13151 | 0.47 (0.37 to 0.55) | 0.16 | 2.91 (2.39 to 3.44) | 0.082 (0.070 to 0.094) |
| B, 8 without 773 | 133 | 175 | 2364 | 11688 | 0.43 (0.31 to 0.53) | 0.17 | 2.57 (1.79 to 3.11) | 0.053 (0.045 to 0.064) |

Over the 18 definitions the pooled relative risk lies between 1.92 and 4.08 and the
sensitivity between 0.045 and 0.11. Per patient (A, `q2_2x2_primary.csv`; relative
risk, gain-and-lost sites): 027 9.9 (15), 773 6.7 (72), 546 5.7 (33), 629 3.3 (13),
103 2.2 (4), 472 1.7 (16), 018 1.7 (19), 168 1.5 (28), 270 1.2 (8); the interval of
103 and 270 includes 1. The AUROC of the methylation change for separating lost from
kept sites among the testable ones is 0.48 to 0.54 for 018, 103, 168, 270, 472, 629,
and 0.67 (027), 0.62 (546), 0.77 (773). The explanatory power is therefore carried by
the patients with a large methylation change.

How many lost sites are explained (A, pooled over patients, `q2_explained_lost_sites.csv`):
of 38,441 patient lost sites 26,809 (70 percent) have no CpG in the motif and cannot be
explained by motif methylation; 3,299 (8.6 percent) are testable (a covered CpG in the
patient and in CD14+; for another 19 percent the CpG is not covered in CD14+ at 10
reads, and 2.4 percent lack patient coverage); 208 of the testable ones (6.3 percent) and 0.54 percent of all lost sites
gain methylation. By patient the testable lost sites with a gain: 773 42 percent, 027 22
percent, 546 21 percent, the other six 1.6 to 5.4 percent.

Coverage (`q2_coverage_dependence_pooled.csv`; pooled A, 9 patients): restricting the
reference to at least 10, 20, 40 reads gives relative risk 2.19 (482 gain sites), 2.60
(84), 4.35 (6): the association does not weaken with more reference depth, but the
number of gain sites falls below 100 beyond 20 reads. Patient read depth is not
available. As proxies, sites in the upper half of the number of covered bins in the
+-250 bp window (a local depth surrogate) give 2.24 (454 gain sites), the lower half
1.35 (28 gain sites); sites whose motif bin value is neither 0 nor 1 (these carry
information on at least two reads) give 2.04 with sensitivity 0.19, a selection that
favours intermediate values and is only a robustness check.

Same-assay check (`q2_same_assay_vs_other_patients.csv`): each patient against the
median of the other 8 patients (CUT&Tag percentile and WGBS, sites covered in at least 4
others; no healthy sample, so this is differential within AML): pooled 200 gain-and-lost,
676 gain-kept, 6,411 no-gain-lost, 30,835 no-gain-kept; precision 0.23, P(lost given
no gain) 0.17, relative risk 1.33, sensitivity 0.030. Per patient relative risk 12.3
(027), 5.6 (773), 4.2 (546), 2.4 (103), 2.0 (629), 1.5 (472), 1.4 (270), 1.0 (168),
1.4 (018): the same three patients carry the effect.

### 3. Hi-C consequence: insulation at gain-and-lost sites

Insulation is `scripts/hic_measures.py:insulation_band` on the patient O/E maps (10 kb,
mean log O/E of the w x w square across the site bin, w = 10 bins = 100 kb as the main
setting, w = 5 as a check), taken at the 10 kb bin of the motif centre; a higher value
means more contact across the site, i.e. weaker insulation. Per site d = insulation
(patient) - insulation (mean of the 4 HSPC donors). Test sites: gain-and-lost (A or B),
one per 10 kb bin, CNV segments of `explain_cnv_segments.csv` removed. Controls: sites
covered for methylation with no gain (change < 0.1), HSPC-bound, matched on motif
strength (within 0.02 of the fraction of the maximum PWM score) and HSPC CTCF height
(within 0.25 log1p units), up to 5 per test site, caliper-limited (test sites without a
control are dropped). Paired difference = d(site) minus mean d(controls), mean over
sites, 95 percent interval from a chromosome bootstrap (2000 resamples; per patient
1000). This is descriptive on all 22 autosomes; no model was fitted. Tables
`q3_insulation_summary.csv`, `q3_paired_sites.csv.gz`.

Pooled over patients (103 has only 4 gain-and-lost sites and has no per-patient row; its
sites are included in the pooled rows):

| test sites against controls | def. | w | test sites | mean d, test | mean d, controls | paired difference | 95% interval |
|---|---|---|---|---|---|---|---|
| gain-and-lost, no-gain controls | A | 10 | 195 | +0.010 | -0.018 | +0.028 | -0.003 to +0.057 |
| same, controls also matched on HSPC insulation | A | 10 | 190 | +0.014 | -0.014 | +0.028 | +0.002 to +0.054 |
| gain-and-lost, no-gain controls | B | 10 | 216 | +0.008 | -0.018 | +0.026 | -0.003 to +0.051 |
| gain-and-lost, no-gain controls | A | 5 | 195 | +0.010 | -0.022 | +0.032 | -0.0005 to +0.064 |
| same, controls also matched on HSPC insulation | A | 5 | 192 | +0.009 | -0.026 | +0.035 | +0.010 to +0.061 |
| gain-and-lost, no-gain controls | B | 5 | 216 | +0.008 | -0.025 | +0.033 | +0.004 to +0.061 |
| gain-and-lost against gain-kept controls (same methylation gain, CTCF kept) | A | 10 | 168 | +0.018 | -0.005 | +0.023 | -0.010 to +0.057 |
| same | B | 10 | 195 | +0.014 | -0.009 | +0.023 | -0.001 to +0.047 |
| same | A | 5 | 168 | +0.014 | +0.013 | +0.001 | -0.028 to +0.031 |
| lost without gain, no-gain-kept controls | A | 10 | 2688 | -0.015 | -0.010 | -0.005 | -0.015 to +0.006 |
| gain with CTCF kept, no-gain controls | A | 10 | 1572 | -0.029 | -0.002 | -0.028 | -0.042 to -0.012 |

Sites that gain methylation and lose CTCF are about 0.03 weaker in insulation than
matched controls (57 percent of the paired differences are positive), about 0.2 of the
standard deviation of d among these sites (0.147). The interval includes 0 for the
pre-set matching (A, w = 10) and for the unmatched A at w = 5 (upper end -0.0005); it
excludes 0 when the controls are also matched on the HSPC insulation (w = 10 and 5) and
for definition B at w = 5. Against controls that gain methylation but keep CTCF
the difference is +0.023 at w = 10 and +0.001 at w = 5. Lost sites without a
methylation gain show no difference (-0.005), so loss alone does not weaken the
insulation in this comparison. Sites that gain methylation with CTCF kept have
stronger insulation than no-gain controls (-0.028), a site-class offset unrelated to
the CTCF loss that is the reason for the gain-kept control. Per patient (A, w = 10;
paired difference, sites): 546 +0.076 (32; interval +0.015 to +0.135), 018 +0.065 (16;
+0.006 to +0.131), 773 +0.032 (70; -0.003 to +0.068), 027 +0.019 (15), 168 +0.017 (23),
629 -0.024 (13), 472 -0.035 (15), 270 -0.057 (7); the patients with a positive,
interval-excluding-0 difference are 546 and 018 only, and 018 has one of the weakest CTCF
associations of part 1.

### 4. What this shows

Direction of causality is not resolved. Methylation at the motif CpGs and CTCF height
are negatively associated in all nine patients beyond motif strength, flank
methylation and H3K27ac, and sites that gain methylation against a healthy myeloid
sample lose CTCF about twice as often as those that do not (relative risk 2.1 to
2.9, interval above 1 in each of the four pooled rows, carried by 027, 546 and 773).
The same pattern is expected if methylation blocks CTCF binding and if CTCF loss lets
the CpGs become methylated (bound CTCF protects its motif from methylation, section
41); a cross-sectional comparison cannot separate the two, and the Hi-C difference
(about 0.03 weaker insulation at gain-and-lost sites, borderline) is compatible with
either order. The 5-azacytidine and knockdown experiments of the paper address the
direction; they are not used here. The share of lost sites that motif methylation can
explain is small: 70 percent of the lost sites have no CpG in the motif and 6 percent
of the testable lost sites gain methylation (sensitivity 0.04 to 0.08 pooled; 0.2 to
0.4 in the three strong patients). Motif hypermethylation accounts for a small part
of the lost sites; the remainder has no motif CpG, no covered comparison or no gain.

Sample sizes. Patients: 9 with CTCF CUT&Tag and WGBS (8 without 773); Hi-C for all 9
(103 has 4 gain-and-lost sites and no per-patient insulation row). Sites: 72,626 motif
hits, 13,729 with a motif CpG, 13,142 to 13,324 covered per patient; 1,786 to 1,803
testable HSPC-bound sites per patient (16,156 pooled in the 2x2 table); 482 gain sites
and 208 (A) or 226 (B) gain-and-lost sites pooled; 195 (A) or 216 (B) gain-and-lost
sites enter the insulation comparison after the one-site-per-bin, CNV and
control-match filters.

Limits. (i) Patient WGBS has no read counts: the 10 reads rule is not applied to the
patients, and the motif value is an unweighted bin mean that can include flank CpGs and
attenuates the correlation. (ii) The healthy references are not HSPC: the methylation
reference is differentiated CD14+ monocytes of other donors, the CTCF reference a
CD34+ ChIP-seq from another laboratory and assay; CUT&Tag against ChIP-seq is compared
by rank only and a rank loss includes assay and depth differences (the per-patient
A-lost counts range from 1,868 to 6,226). A site methylated in HSPC but not in
monocytes would be called a gain that is not one, a site unmethylated in HSPC and
methylated in monocytes a gain that is missed; the size of this is not measured.
(iii) Gain and loss thresholds are arbitrary; the 18-definition grid gives a relative
risk between 1.92 and 4.08 with 80 to 511 gain-and-lost sites. (iv) Sites are motif
hits with PWM at least 0.6 of the maximum, not called peaks; the bound set is the top
20 percent by HSPC height. (v) H3K4me3 is not available for the patients. (vi)
Intervals resample chromosomes; the patients share chromosomes and sites, so pooled
intervals treat the patients as fixed, and the 10 kb insulation resolution cannot place
a site inside a bin. (vii) No multiple-testing correction; per-patient rows rest on 4
to 72 sites. (viii) The statistics of part 1 use the even chromosomes (as section 41);
the all-chromosome versions are in the same tables; the Hi-C comparison is descriptive
and uses all autosomes.

Figures `fig_q1_partial_spearman.png`, `fig_q2_precision_sensitivity.png` and
`fig_q3_insulation.png` were written but not visually inspected (viewing images was
not permitted for this task). The figures are not tracked by git (`*.png` is ignored);
`python scripts/aml_ctcf_methylation.py --stage figures` rebuilds them.


## 47. Do the patients' own marks and methylation separate AML-specific loops from shared loops? (FINAL; sections 42 to 46 were taken by other agents, this is the next free number)

`scripts/aml_loop_classifier.py` (CPU only, hicdiff env), tables and figures in `results/aml_loop_classifier/`.
Question: with held-out patients, does the patient's own CUT&Tag (H3K27ac, H3K27me3, CTCF), WGBS methylation or
CTCF motif content distinguish the AML-specific loops and stripes of Xu et al. 2022 (Suppl. Tables 1 and 4) from
loops that are shared with HSPC?

### Data and sample sizes
- Positives: AML-specific loops of Suppl. Table 1 (10 kb anchors, 60 kb to 1.5 Mb), stripes of Suppl. Table 4
  (anchor 40 kb, extent block; at most 150 sampled per patient, geometry limited to 3.2 Mb).
- Patients: 8 with all four data types (018, 027, 168, 270, 472, 546, 629, 773; CTCF, H3K27ac, H3K27me3, WGBS),
  10 with H3K27ac and H3K27me3 (adds 1360, 1916, no CTCF, no WGBS), 18 with WGBS (adds 1071, 1324, 260, 424, 496,
  504, 566, 798, 838, 933 without CUT&Tag). Patients 103 and 673 have CUT&Tag but no AML-specific loops in Suppl.
  Table 1. Loops: 1.86 to 1.89k positives in the 8-patient set, 2.2k in the 10-patient set, 3.8 to 4.1k in the
  18-patient set; stripes 0.95 to 1.1k positives (stripe experiments use the 8 patients with all data types).
- Features use only the patient's data and one public CD34+ ChIP experiment per mark (ChIP-Atlas, other lab, one
  donor) as HSPC reference: anchor mean, max and +-20 kb flank of the quantile-matched CUT&Tag/ChIP tracks (the AML
  model inputs), WGBS methylation (1 kb bins: anchor mean, minimum, 1 kb bin at the strongest CTCF motif), CTCF
  motif (MA0139.2, hits at >= 70 percent of the maximal log-odds, strength, orientation class), distance, promoter
  (refGene TSS in anchor, distance to nearest TSS), HSPC values and patient minus HSPC. Per pair every anchor
  feature enters as min and max. No Hi-C-derived feature of the target pair. No RNA-seq exists in the repo, so no
  expression feature. Methylation of HSPC does not exist (no public WGBS), so methylation has no difference feature.
- Shared loops are called here, not by Peakachu (the Peakachu calls are not public): local maximum of the 10 kb donut
  enrichment (hicmultiome.bioscore, same parameters as the AML scripts), log2 enrichment >= 0.6 in the patient map
  and >= 0.6 (3x3 maximum) in the HSPC consensus, both anchors mappable, not within 2 bins of an AML-specific loop
  of the patient. Median log2 enrichment: positives 0.69 (patient) and 0.18 (HSPC consensus; for the 55 percent of
  positives with a local maximum within 1 bin); negatives 0.83 to 0.88 and 0.71 to 0.73. Negatives are stronger
  loops than positives; strength is not matched and the enrichment is never a feature.
- Negative sets (1:1, within the patient, nearest neighbour without replacement, exact orientation class, distance
  caliper max(2 bins, 15 percent)): A_full = shared loops matched on distance, motif orientation (convergent,
  divergent, tandem, none) and the anchors' mean H3K27ac (caliper 0.25 SD; CUT&Tag patients); A_do = distance and
  orientation only; A_hspc = distance, orientation and the anchors' HSPC CTCF and HSPC H3K27ac (own marks left free);
  B_rand = random anchor pairs on mappable bins of the same chromosome at the same bin distance. Stripes: the
  stripe rectangle (same gap and extent) moved along the chromosome; shared = band mean log O/E >= the 25th
  percentile of the patient's positives in both the patient and the HSPC consensus; matching on orientation
  (anchor motif facing the extent) and anchor K27ac (A_full) or HSPC marks (A_hspc).
- Matching quality (`matching_quality.csv`): matched positives 4500/4520 (A_do), 2232/2276 (A_full, 8+2 patients
  with CUT&Tag), 4160/4520 (A_hspc), 4520/4520 (B_rand) for loops; stripes 1317, 1283, 1109, 1317 of 1394. Mean
  log-distance SMD < 0.001; orientation equal in 100 percent. A_full anchor-K27ac SMD within +-0.033 per patient.
  A_hspc residual SMD: HSPC CTCF up to 0.21, HSPC K27ac up to 0.11 per patient. In A_do and A_hspc the own anchor
  K27ac is higher in positives (mean SMD 0.10 and 0.32, per patient up to 0.36 and 0.71): a gain signal.

### Validation
Leave-one-patient-out (LOPO) and leave-subtype-out (LSO; subtype from the SV junctions of Suppl. Table 5: BCR-ABL1
{027, 168, 496}, inv(16) {629, 504, 798}, KMT2A-r {270}, t(8;21) {933}; every other patient is its own group; mutation
subtypes are not assigned to patients in the paper's tables). Never a random split. Recurrent loops share anchors
across patients, and the HSPC features identify the locus, so the primary folds are locus-disjoint: training rows
with an anchor within 2 bins of a test anchor are dropped. Plain LOPO is kept (`cv = lopo_plain`) and is higher
(A_do, gradient boosting 0.912 against 0.880): that part was locus memorisation. Models: logistic regression (C 0.05) and
histogram gradient boosting (depth 3, 150 iterations). Metrics: mean over patients of the per-patient AUROC
(macro) and pooled AUROC and AUPRC, patient-bootstrap 95 percent intervals (1000 resamples of patients). Controls:
distance only (logistic regression), labels shuffled within each patient (20 repetitions logistic regression, 10
gradient boosting; 97.5th percentile of the macro AUROC reported). Prevalence 0.5.

### Results (gradient boosting, LOPO locus-disjoint, macro AUROC [95 percent patient bootstrap])
Distance only 0.50 and shuffled labels 0.50 (97.5th percentile 0.51 to 0.53) in every row. LSO equals LOPO within 0.004.

| experiment | patients | pos | negatives | AUROC | pooled AUPRC |
|---|---|---|---|---|---|
| loops, all data | 8 | 1861 | A_full | 0.874 [0.841, 0.903] | 0.881 |
| loops, all data | 8 | 1885 | A_do | 0.880 [0.853, 0.902] | 0.893 |
| loops, all data | 8 | 1851 | A_hspc | 0.837 [0.805, 0.865] | 0.832 |
| loops, all data | 8 | 1886 | B_rand | 0.792 [0.760, 0.826] | 0.775 |
| loops, K27ac + K27me3 | 10 | 2184 | A_hspc | 0.833 [0.805, 0.860] | 0.824 |
| loops, meth + motif + gene + HSPC | 18 | 3827 | A_hspc | 0.820 [0.802, 0.840] | 0.806 |
| stripes, all data | 8 | 1099 | A_full | 0.674 [0.652, 0.696] | 0.651 |
| stripes, all data | 8 | 945 | A_hspc | 0.608 [0.581, 0.634] | 0.612 |
| stripes, all data | 8 | 1128 | B_rand | 0.810 [0.794, 0.829] | 0.800 |

Logistic regression is 0.01 to 0.04 lower everywhere. All rows are far above the controls; the question is what
carries it.

Which groups carry it (`group_importance.csv`, `incremental.csv`, `univariate.csv`; 8 patients, A_do unless noted):
- The HSPC state of the locus alone gives 0.873 (A_do), 0.813 (A_hspc), mostly HSPC CTCF (0.847 and 0.789 alone).
  This is largely built into the labels (a shared loop exists in HSPC Hi-C and sits on HSPC CTCF) and is not a
  patient signal. Positives have lower CTCF in the flanks (univariate AUROC 0.26 to 0.36 for CTCF flank features, 0 of 8
  patients above 0.5) and lower H3K27me3 (0.37 to 0.40).
- Patient marks without any HSPC input: all four data types 0.752 [0.704, 0.801] (A_full) and 0.730 (A_hspc);
  own CTCF 0.717, own H3K27me3 0.606, own H3K27ac 0.538 (A_do); with HSPC matching own CTCF 0.669, H3K27me3
  0.628, H3K27ac 0.596. Own marks plus methylation, motif and gene: 0.787 (A_full), 0.746 (A_hspc), 0.637 [0.619, 0.657] in the
  18 patients (A_hspc, methylation instead of marks). The patient-minus-HSPC differences alone give 0.80 (A_do).
- Gain of H3K27ac is the clearest directional patient signal: patient-minus-HSPC H3K27ac at the anchors has a
  univariate AUROC of 0.62 and is above 0.5 in 8 of 8 patients (A_hspc), also own H3K27ac max 0.61 (7 of 8).
- Added to the HSPC state (paired patient bootstrap of the AUROC difference, A_hspc, 8 patients): + own marks
  +0.023 [0.018, 0.028], + differences +0.017 [0.005, 0.024], + methylation +0.002 [-0.002, 0.005], + motif and
  gene +0.007 [0.001, 0.013], everything +0.023 [0.014, 0.031]. 10 patients with K27ac and K27me3: +0.015 [0.010, 0.020].
  A_full: +0.015 [0.011, 0.021]. A_do: +0.006 [-0.001, 0.012] (not different from zero). 18 patients, methylation
  and motif without own marks: +0.007 [0.004, 0.009].
- Methylation alone separates shared from specific loops (0.641, A_do) but adds 0.002 to 0.004 on top of the HSPC
  state; in A_hspc alone it is 0.548. Distance is 0.50 by construction.

Pair types (`by_type.csv`, A_hspc, pooled AUROC [bootstrap]): P-E-like 0.864 [0.802, 0.913] (241 positives), P-S-like
0.741 [0.607, 0.827] (83), other 0.838 (1527); A_do: 0.854, 0.753, 0.889. Types follow the patient's own anchors (promoter in
one anchor, the partner above the 70th percentile of the training H3K27ac scale for P-E-like, of H3K27me3 for P-S-like, P-S
first); few P-S loops carry the numbers, the intervals overlap. Stripes (A_hspc): repressive-anchor 0.677 (397), enhancer-anchor 0.495
(278), other 0.619; against shared stripes the own data add nothing over the HSPC state (+0.004 [-0.013, 0.018]),
and with methylation +0.025 [0.008, 0.038] only in A_full.

### Answer
Yes, a held-out-patient signal exists far above the controls (0.84 to 0.88 AUROC for loops, controls 0.50), but most
of it is the HSPC state of the locus, which the definition of a shared loop imposes. The patient's own data add a small, reproducible amount over
the HSPC state: +0.015 to +0.026 AUROC when the negatives are matched on HSPC state or anchor K27ac, carried by CTCF,
H3K27me3 and the patient-minus-HSPC H3K27ac gain, not by methylation. Against negatives matched on distance and orientation
only, the increment is not distinguishable from zero. The signal is real but modest and is weaker than the full
classifier numbers suggest. Stripes: only 0.61 to 0.67 against shared stripes, with no increment of the own marks beyond the HSPC state.

### Limits
- Few patients with CUT&Tag (8 with all data types, 10 with the two histone marks); one lab and one protocol, so batch, depth
  and CUT&Tag quality are shared by train and test and a patient-level quality effect is not excluded.
- HSPC marks come from one donor of another lab, constant over all loops; patient-minus-HSPC differences contain lab and assay
  differences. HSPC methylation does not exist.
- AML-specific loops are Peakachu calls (probabilities, thresholded) against 4 HSPC; shared loops are our donut calls at
  10 kb and are stronger than the positives. The 55 percent local-maximum recovery of the positives shows that the two
  call sets are not the same operation.
- CTCF motif hits are sparse at 10 kb (about 12 percent of pairs have a hit in both anchors); the orientation class is
  mostly "none". Methylation is at 1 kb resolution (motif CpGs not resolved individually).
- Locus-disjoint folds remove anchors within 2 bins only; related loci (same TAD or gene) remain. Mutation subtypes are not
  known per patient, so LSO is nearly LOPO. Bootstrap intervals over 8 to 18 patients are narrow because per-patient AUROCs
  are similar; they do not cover the batch limits above.
- Stripe negatives are geometry-matched shifts of the rectangle, a weaker notion of "shared stripe" than for loops.
- Figures `fig_auroc_summary.png`, `fig_group_importance.png`, `fig_per_patient.png`, `fig_matching_balance.png` were
  not visually checked. Intermediate Hi-C label tables and tracks are cached under aml/primary/work/loopclf/ (not in git).

Reproduce: `python scripts/aml_loop_classifier.py --stage hic tracks motifs stripes_hic build model figures`
(hicdiff env; `--jobs 8 --null-reps-lr 20 --null-reps-gb 10` were used).


## 48. Can motif CpG methylation exclude unbound CTCF motif hits? Site-level bound versus unbound classification (FINAL; section 47 was the last number when this was written)

`scripts/ctcf_site_exclusion.py` (CPU, 4 threads), tables and figures in
`results/ctcf_site_exclusion/`. Sections 41 and 46 measured a height regression. The question here is the
site-level one that matters for a model without a CTCF track: is a motif hit bound or unbound, and does
methylation of the CpGs inside the motif identify the unbound hits.

**Data.** The 72,626 motif hits of section 41 (JASPAR MA0139.2, PWM >= 0.6 of the maximum, hg38), the per-cell
tables of section 41 (CTCF height = log1p of the mean ChIP signal within +-100 bp of the motif centre; motif
methylation = methylated fraction over the CpGs wholly inside the motif, reads pooled, >= 10 reads, called
"covered"; flank = +-250 bp) and, for patients, the tables of section 46 (9 patients with CTCF CUT&Tag and
WGBS; 10 bp bin means, no read counts). GM23248 has no CTCF track and is not used, so the cells are the
6 training cells (GM12878, IMR-90, H1, HepG2, A549, CD14+ monocyte) and K562 and HMEC (in no training set). No
ENCODE peak calls exist in the repo, so bound and unbound are defined from the signal. The per-site maximum over
the motif CpGs (CpGs with >= 5 reads) was added from the CpG tables (`runs/chip_meth_rel/cpg`).

**Labels** (per cell and patient; thresholds in `labels_thresholds.csv`). Background = 100,000 random 200 bp
windows >= 500 bp from every motif hit, scored like a site (log1p of the mean signal).
Definition A: bound = height above the 95th percentile of the background; unbound = height at or below the
median of the cell's 72,626 hits. Definition B: bound = top 20 percent of the hits by height; unbound = at or
below the median. Sites between the two thresholds are ambiguous and are left out of every main number. In H1
and HepG2 the motif median lies above the background 95th percentile, so definition A has no ambiguous sites
there and unbound there means at or below that percentile. Sites per cell (A, bound/unbound; B is 14,525/36,313 everywhere): GM12878 28,066/36,313, IMR-90 30,885/36,313,
H1 39,642/32,984, HepG2 37,338/35,288, A549 32,378/36,313, CD14+ 26,754/36,313, K562 35,430/36,313, HMEC
27,866/36,313. The share bound among labelled sites is 0.42 to 0.55 in the cells and 0.14 to 0.34 in the patients
(sparser CUT&Tag signal).

**Models.** Features: (a) motif strength (PWM fraction) and orientation; (a_cpg) a plus the number of motif
CpGs (control for CpG-containing sequence without methylation values); (b) a_cpg plus covered flag, motif
methylation mean and maximum; (c) b plus flank methylation and its covered flag; (d) methylation alone (b and
c without strength and orientation). Logistic regression and histogram gradient boosting, odd chromosomes of the
training cells to even chromosomes of the scored cell. `seen_cell`: the cell is in the training set, other
chromosomes. `unseen_cell`: leave-one-cell-out for the 6 training cells, the model of all 6 for K562 and HMEC.
Chromosome-block bootstrap (300 resamples of the 11 even chromosomes) for AUROC, AUPRC and the paired
difference to the strength-only model. Tables `site_classification_metrics.csv` (all rows, both scopes, both
models, both schemes), `summary_covered_sites.csv`, `patient_transfer_metrics.csv`.

### 1. Coverage (`coverage.csv`)

A motif CpG exists in 18.9 percent of the 72,626 hits (30.0 percent of the 14,131 strong hits, PWM >= 0.75 of
the maximum). The share with a covered motif CpG (>= 10 reads) is 12.9 (GM12878), 0.9 (IMR-90), 11.8 (H1),
3.2 (HepG2), 9.6 (A549), 5.5 (CD14+), 9.1 (K562) and 5.5 (HMEC) percent of all hits, i.e. 5 to 68 percent of the hits that
hold a CpG. The patient bin means cover 95.7 to 97.0 percent of the hits that hold a CpG (18.1 to 18.4 percent of all hits). Of the unbound strong hits (label A), 94.5 to 99.6 percent carry no covered motif CpG in the cells, so a methylation rule can
only act on a small minority of them.

### 2. Classification on sites with a covered motif CpG (gradient boosting, label A, unseen cell)

AUROC, 95 percent chromosome-bootstrap interval in brackets. n = covered labelled sites of the even chromosomes,
prev = share bound. d is methylation alone. Definition B and the seen_cell scheme give the same ordering
(`summary_covered_sites.csv`).

| cell | n | prev | a strength + orientation | b + motif methylation | c + flank methylation | d methylation alone |
|---|---|---|---|---|---|---|
| GM12878 | 4,048 | 0.73 | 0.767 [0.757, 0.779] | 0.850 [0.834, 0.862] | 0.834 [0.818, 0.846] | 0.747 |
| IMR-90 | 283 | 0.80 | 0.659 [0.562, 0.746] | 0.886 [0.833, 0.937] | 0.894 [0.838, 0.947] | 0.872 |
| H1 | 4,031 | 0.81 | 0.766 [0.752, 0.783] | 0.893 [0.884, 0.902] | 0.893 [0.884, 0.903] | 0.871 |
| HepG2 | 1,457 | 0.81 | 0.768 [0.745, 0.786] | 0.862 [0.844, 0.885] | 0.863 [0.847, 0.880] | 0.797 |
| A549 | 3,201 | 0.74 | 0.770 [0.752, 0.784] | 0.889 [0.877, 0.901] | 0.888 [0.876, 0.899] | 0.845 |
| CD14+ | 1,685 | 0.73 | 0.743 [0.725, 0.757] | 0.897 [0.883, 0.910] | 0.896 [0.879, 0.913] | 0.850 |
| K562 (held out) | 2,928 | 0.81 | 0.768 [0.752, 0.780] | 0.814 [0.797, 0.830] | 0.816 [0.799, 0.834] | 0.738 |
| HMEC (held out) | 1,733 | 0.72 | 0.766 [0.743, 0.786] | 0.912 [0.902, 0.921] | 0.912 [0.899, 0.923] | 0.892 |

The gain of b over a is +0.046 (K562) to +0.227 (IMR-90) AUROC, every paired bootstrap interval excludes zero,
and the AUPRC gain is +0.013 to +0.088 (the baseline AUPRC is already 0.88 to 0.94 because 72 to 81 percent of
the covered sites are bound). The number-of-CpGs control a_cpg adds only +0.006 to +0.031 AUROC, so the gain
comes from the methylation values. Flank methylation adds nothing beyond the motif (c versus b within +-0.02,
lower in GM12878 by 0.016). Methylation alone (d) reaches 0.74 to 0.89, above the strength-only model in 6 of 8
cells (GM12878 and K562 are the exceptions). Logistic regression gives the same picture (b 0.81 to 0.91 for the 8 cells). On all labelled sites
(covered or not, `scope = all_labelled_sites`) the strength-only AUROC is 0.66 to 0.77 and b is 0.82 to 0.91; the model
with methylation there uses the covered flag as well. Definition B gives AUROC 0.89 to 0.98
for b (a: 0.72 to 0.83). Patients (models of the 6 training cells applied to the patients' even chromosomes, no
CpG maximum because the bins give none): a 0.63 to 0.73, b_nm 0.70 to 0.88, c_nm 0.70 to 0.88 AUROC;
the gain over a is +0.070 to +0.165, all intervals above zero (`patient_transfer_metrics.csv`; the patient
methylation is a bin mean on a different scale than the read-pooled value the models were trained on).

### 3. Exclusion view: strong motifs (PWM >= 0.75), covered, label A

Fraction bound by motif methylation (`exclusion_bound_fraction_by_methylation.csv`, GM12878 as example, n in
brackets): 0.97 (2,402) at 0-0.1, 0.89 at 0.1-0.2, 0.74 at 0.2-0.3, 0.76 at 0.4-0.5, 0.72 at 0.6-0.7, 0.47 to
0.67 at 0.8-1.0 (19 to 92 sites per bin). In the other cells the fraction bound at 0.9-1.0 is 0.33 (IMR-90, 9
sites), 0.59 (CD14+), 0.58 (HMEC), 0.60 (A549), 0.74 (K562), 0.76 (H1), 0.80 (HepG2). Fully methylated strong hits are
therefore still bound in the majority of cases, and the fraction bound is 0.97 to 1.00 at 0-0.1 and 0.89 to 1.00 at 0.1-0.2.

Rule "motif methylation > t implies unbound" (`exclusion_rules_cells.csv`; 95 percent chromosome-bootstrap
intervals in the file). precision = share of excluded sites that are unbound; FER = bound covered sites that the rule
excludes, as share of all bound covered strong sites; removed = share of ALL unbound strong hits (covered and
uncovered) that the rule removes. The base rate of unbound among covered strong sites is 0.02 to 0.06.

| cell | covered strong bound/unbound | unbound strong total | t=0.5 excl. (bound), precision, FER | t=0.7 | t=0.9 | removed of all unbound at 0.5/0.7/0.9 |
|---|---|---|---|---|---|---|
| GM12878 | 2605/146 | 2667 | 167 (111), 0.34, 0.043 | 120 (77), 0.36, 0.030 | 82 (57), 0.30, 0.022 | 0.021/0.016/0.009 |
| IMR-90 | 155/9 | 2564 | 16 (9), 0.44, 0.058 | 12 (6), 0.50, 0.039 | 8 (3), 0.62, 0.019 | 0.003/0.002/0.002 |
| H1 | 2594/56 | 1797 | 292 (245), 0.16, 0.094 | 216 (174), 0.19, 0.067 | 99 (74), 0.25, 0.029 | 0.026/0.023/0.014 |
| HepG2 | 654/15 | 1969 | 33 (27), 0.18, 0.041 | 21 (17), 0.19, 0.026 | 6 (4), 0.33, 0.006 | 0.003/0.002/0.001 |
| A549 | 1895/92 | 2409 | 167 (114), 0.32, 0.060 | 130 (81), 0.38, 0.043 | 65 (40), 0.38, 0.021 | 0.022/0.020/0.010 |
| CD14+ | 1120/72 | 2671 | 166 (113), 0.32, 0.101 | 135 (86), 0.36, 0.077 | 80 (47), 0.41, 0.042 | 0.020/0.018/0.012 |
| K562 (held out) | 1991/55 | 2312 | 42 (36), 0.14, 0.018 | 31 (25), 0.19, 0.013 | 16 (12), 0.25, 0.006 | 0.003/0.003/0.002 |
| HMEC (held out) | 1031/57 | 2606 | 122 (79), 0.35, 0.077 | 94 (58), 0.38, 0.056 | 60 (35), 0.42, 0.034 | 0.017/0.014/0.010 |

With the strict bound definition B (bound = top 20 percent) the precision is 0.55 to 1.00 (GM12878 0.65/0.75/0.69,
A549 0.90/0.96/1.00, HMEC 0.74/0.84/0.89, K562 0.60/0.67/0.80 at t = 0.5/0.7/0.9) and the FER 0 to 0.037,
so the exclusion precision depends on how weak a binding is counted as bound. The removed share of all unbound strong hits is 0.1 to 2.7
percent under both definitions: at least 97 percent of them stay, because they are below the threshold or have no covered motif CpG.

AML patients (`exclusion_rules_patients*.csv`, pooled over 9 patients, label A): precision of the rule 0.76, 0.77,
0.80 at t = 0.5, 0.7, 0.9 against a base unbound rate of 0.39 among covered strong sites (per patient 0.59 to 0.91;
base rate 0.17 to 0.62); FER 0.076, 0.060, 0.036; removed share of all unbound strong hits 0.077, 0.066, 0.046.
Patients are different from the ENCODE cells in two ways: the CUT&Tag signal makes 17 to 62 percent of the covered
strong sites unbound under A, and the 10 bp bin means give no read-count filter, so coverage is high (96 percent of
hits with a CpG) but each value has no depth. Patient 773 and 027 have the lowest precision (0.62).

### 4. Hi-C relevance (`hic_regions_excluded.csv`)

The 413 Akita test regions (1,048,576 bp windows; scored central window = 917,504 bp) hold 1,281 strong hits in
the scored window. The sites that the rule excludes inside the scored windows (label A, t = 0.5/0.7/0.9; unique
sites, of which unbound): GM12878 18/15/11 (8/6/4), IMR-90 2/1/1, H1 22/13/6 (4/4/0), HepG2 5/2/1, A549 16/13/4,
CD14+ 8/7/5, K562 1/1/1 (1/1/1), HMEC 16/13/10 (6/5/4). Against 120 to 200 unbound strong hits in the regions per cell, the
rule removes 1 to 8 of them at t = 0.5 (0.6 to 4 percent). The nine patients exclude 39 to 68 sites per patient at t = 0.5 (full table
in the file).

### What the result supports

Methylation of the motif CpGs is informative about binding of a CTCF motif hit that has a covered CpG: among
such hits it separates bound from unbound with AUROC 0.81 to 0.91 (unseen cells, including the two never-trained
cells) against 0.66 to 0.77 for motif strength and orientation, and the gain persists in AML patients. The
information is mostly "unmethylated hits are bound" (0.97 to 1.00 bound at 0-0.1); the converse is weak, since
fully methylated strong hits are bound in 33 to 80 percent of cases. Methylation of the motif does not decide binding in either direction: at least 97 percent of the unbound strong hits survive the rule, so an
unmethylated motif can still be unbound (an unmethylated motif is permissive for binding and far from sufficient; a methylated motif is bound in 33 to 80 percent of the fully methylated cases, so methylation is not necessary for exclusion either), and 58 to 86 percent of the sites that a hard rule excludes are bound
(precision 0.14 to 0.42 under A at t = 0.5 to 0.9 in the cells; 0.55 to 1.00 under B). Coverage limit: 1 to 13 percent
of all hits (5 to 68 percent of hits with a CpG) have a covered motif CpG in the cells, 81 percent have no CpG in the
motif at all; a rule therefore removes at most 2.7 percent of the unbound strong hits.

For a model that gets no CTCF track, motif methylation can reduce the probability assigned to a methylated
CpG-containing hit and so improve the ranking of bound over unbound hits where it is measured, which is
the effect behind the methylation-weighted motif planes (section 43). It cannot supply the information about where CTCF is
absent for the bulk of the genome, because the unbound hits are overwhelmingly unmethylated or without CpG. The remaining separation of
unbound hits has to come from other inputs (accessibility, histone marks, the sequence context of the hit).

Limits. No peak calls exist for the ENCODE cells, so the labels are signal quantiles; the
results depend on the definition (A and B bracket them) and signal scale is cell specific (thresholds are per cell).
The strong motif cut-off (0.75) is a choice. IMR-90 has 9 and HepG2 15 unbound covered strong sites, so their rows
carry wide intervals. Methylation correlates with CpG islands and promoters, which are bound for reasons other than
the motif, and the models do not separate this from a direct effect of the methylated base; the
number-of-CpGs control addresses sequence content only. The patient transfer uses models trained on read-pooled
methylation and 10 bp bin means differ in scale. The figures `fig_auroc_by_feature_set.png`,
`fig_bound_fraction_by_methylation.png` and `fig_exclusion_rule.png` were written by the script and not viewed.

Reproduce: `python scripts/ctcf_site_exclusion.py --stage bg labels coverage models exclusion patients hic figures`
(python env with scikit-learn, pyBigWig, matplotlib; CPU).

## 49. Model against the replicate noise ceiling, per cell and resolution (FINAL; section 48 was the last number when this was written)

`scripts/replicate_ceiling_table.py` (CPU, reads existing tables only). Output in `results/replicate_ceiling/`:
`replicate_ceiling_table.csv` (full table, 17 cells x 4 resolutions), `replicate_ceiling_summary.csv`,
`replicate_ceiling_table.md` (compact), `replicate_ceiling.png`.

Metric: median per-region Pearson on the upper triangle (k >= 2) of the proper O/E map, the 413 Akita GM12878
test regions applied to every cell. Model: all12refin_drop (all12refres_drop in the CSV; within 0.01 of it),
per-region values of `runs/multicell_reference/<run>/crosscell/predict_<cell>.csv`; the medians reproduce
`reference_summary.csv` (25 kb: 0.9362 training, 0.9137 held-out). Consensus alone: leave-one-out consensus
(training cells) and consensus12 (held-out cells). No consensus: all12honest_drop. Replicates: rep1 vs rep2 of
`replicates/replicate_corr.csv` (section 7: ENCODE in situ Hi-C, two independent biological replicates, one
sequencing lane each, 110 to 230 M contacts, regridded to the nominal 5 and 25 kb grid) for GM12878, IMR-90,
K562, HMEC and HUVEC; the other 10 cells have no replicate pair (ceiling NA, no imputation). HMEC deep and
HUVEC deep have no pair of their own and use the HMEC and HUVEC single-lane pair as a proxy (different
experiment from the deep target; HUVEC deep is intact Hi-C). HMEC / HMEC deep and HUVEC / HUVEC deep share
their inputs and their predictions are the same model on the same tracks; they differ in the target only
(HMEC and HUVEC are scored against the single-lane rep1 map, the deep cells against the deep merged map).

Ceiling formula. A predictor that knows the true signal correlates with a noisy target at sqrt(reliability of
the target) (sections 12, 18, 29c). The reliability is the replicate agreement r_rep. Target a single replicate
(HMEC, HUVEC): ceiling = sqrt(r_rep). Target a deep merged map: Spearman-Brown for the mean of two replicates,
r_full = 2 r_rep / (1 + r_rep), ceiling = sqrt(r_full); the deep targets hold 5 to 40 times the contacts of a
single-lane replicate, so this is a LOWER BOUND of their ceiling. ratio = model / ceiling, gap = ceiling -
model. For GM12878 (ENCSR965OVH) and K562 (ENCSR676BEP) the deepest intact pair (section 12) gives a second,
stricter ceiling (`ceiling_deep_pair_sb`).

Medians over cells (refin_drop; ceiling statistics over the cells with a replicate pair: 2 training cells,
GM12878 and IMR-90; 5 held-out cells):

| res | set | median model | median consensus | median no consensus | median ceiling | median model / ceiling | cells >= 95% | cells within 0.01 |
|---|---|---|---|---|---|---|---|---|
| 25 kb | training (12) | 0.936 | 0.922 | 0.809 | 0.987 | 0.955 | 2/2 | 0/2 |
| 25 kb | held-out (5) | 0.914 | 0.898 | 0.775 | 0.972 | 0.938 | 0/5 | 0/5 |
| 10 kb | training | 0.927 | 0.887 | 0.819 | 0.950 | 0.990 | 2/2 | 1/2 |
| 10 kb | held-out | 0.903 | 0.859 | 0.761 | 0.906 | 0.970 | 3/5 | 1/5 |
| 5 kb | training | 0.885 | 0.828 | 0.764 | 0.887 | 1.027 | 2/2 | 2/2 |
| 5 kb | held-out | 0.830 | 0.785 | 0.684 | 0.791 | 1.004 | 3/5 | 3/5 |
| 2 kb | training | 0.761 | 0.686 | 0.641 | 0.782 | 1.019 | 2/2 | 1/2 |
| 2 kb | held-out | 0.677 | 0.629 | 0.544 | 0.694 | 0.932 | 2/5 | 2/5 |

Per cell with a ceiling (model / ceiling; 25, 10, 5, 2 kb):

| cell | target | 25 kb | 10 kb | 5 kb | 2 kb |
|---|---|---|---|---|---|
| GM12878 (training) | deep | 0.936 / 0.984 | 0.933 / 0.936 | 0.911 / 0.861 | 0.791 / 0.746 |
| IMR-90 (training) | deep | 0.948 / 0.989 | 0.946 / 0.964 | 0.909 / 0.912 | 0.800 / 0.817 |
| K562 (held out) | deep | 0.926 / 0.984 | 0.905 / 0.931 | 0.834 / 0.830 | 0.677 / 0.727 |
| HMEC deep (held out) | deep | 0.924 / 0.985 | 0.908 / 0.936 | 0.858 / 0.843 | 0.710 / 0.694 |
| HUVEC deep (held out) | deep | 0.914 / 0.972 | 0.903 / 0.906 | 0.830 / 0.791 | 0.704 / 0.706 |
| HMEC (held out) | rep1 | 0.890 / 0.970 | 0.803 / 0.883 | 0.601 / 0.743 | 0.336 / 0.563 |
| HUVEC (held out) | rep1 | 0.888 / 0.947 | 0.785 / 0.835 | 0.491 / 0.675 | 0.263 / 0.576 |

Stricter ceiling from the deepest intact pair: GM12878 model / ceiling 0.936 / 0.992, 0.933 / 0.966, 0.911 /
0.924, 0.791 / 0.828 (ratios 0.94, 0.97, 0.99, 0.96); K562 0.926 / 0.997, 0.905 / 0.988, 0.834 / 0.963, 0.677 /
0.888 (ratios 0.93, 0.92, 0.87, 0.76).

Reading.
- 25 kb: the model sits 0.04 to 0.08 below the ceiling in all 7 cells with a pair (GM12878 and IMR-90 at
  0.95 and 0.96 of it, held-out 0.92 to 0.94); no cell is within 0.01. The model is above the consensus alone in
  all training cells; it is above the raw rep1-vs-rep2 agreement in none of the training cells (0.936 against
  0.938, 0.948 against 0.959) and in 1 of 5 held-out cells (HUVEC deep).
- 10 kb: at replicate level for the training cells (0.98 and 1.00 of the ceiling) and for K562, HMEC deep and HUVEC
  deep (0.97, 0.97, 1.00); not for the single-lane-target HMEC and HUVEC (0.91 and 0.94).
- 5 and 2 kb: ratios of 1.00 to 1.06 for the deep targets reflect the lower-bound ceiling (single-lane pairs of
  0.32 to 0.59 agreement against a target 5 to 40 times deeper), not super-replicate accuracy; the stricter
  deep-pair ceiling gives 0.99 and 0.96 for GM12878 but 0.87 and 0.76 for K562. The correctly specified rows are
  those whose target is the single replicate itself (HMEC, HUVEC): there the model reaches 0.81 and 0.73 of the
  ceiling at 5 kb and 0.60 and 0.46 at 2 kb. The same predictions score 0.86 and 0.83 (5 kb) and 0.71 and 0.70
  (2 kb) against the deep maps, so the shallow-target score is dominated by target noise and the shortfall to
  its ceiling is larger than the deep-target ratios suggest.
- At 25 kb the consensus alone is 0.01 below to 0.05 below the model (CD14 mono 0.21 below, consensus 0.668; HMEC,
  HMEC deep and HUVEC deep 0.001 to 0.009 above it); the no-consensus model is 0.08 to 0.18 below the model.

Verdict for the sentence "accuracy basically at biological replicate level": supported at 10 kb for the training
cells and for the held-out cells scored on deep maps (K562, HMEC deep, HUVEC deep), and at 5 kb for GM12878 and
IMR-90 (K562 at 5 kb only against the lower-bound ceiling: 0.87 against the deep-pair ceiling); at 25 kb the model is 0.04 to 0.08 below the replicate ceiling even on the training cells (ratio 0.95 to
0.96, gap 0.04) and 0.92 to 0.94 on the held-out cells, and the claim fails at 2 kb for K562 and for the
single-lane-target HMEC and HUVEC. Without a replicate pair 10 of 12 training cells cannot be placed against a
ceiling; the claim is a statement about the 2 training and 5 held-out cells that have one. The better wording is
"at or near replicate agreement at 5 to 10 kb on cells with a replicate pair, 0.04 to 0.08 below the ceiling at 25 kb,
and below it on held-out cells at 2 kb".

Limits. (1) Replicate pairs are single-lane (110 to 230 M contacts per replicate) while the targets hold 0.9 to
4.0 billion; a half-depth pair underestimates the ceiling of the merged map, so ratios above 1 occur (GM12878
5 and 2 kb, 1.06) and mean a too low ceiling; the deep-pair ceilings (GM12878, K562 only) are 0.01 to 0.16
higher. (2) The Spearman-Brown step assumes two exchangeable replicates of equal depth and treats the
per-region Pearson of O/E maps as a reliability; k = 2 is used for every deep target. (3) The replicate maps went
through a regridding from 4 and 24 kb to 5 and 25 kb (section 7; averaging smooths and raised the agreement by
0.023 to 0.026 at 5 kb and 0.003 to 0.008 at 25 kb), so the 5 and 25 kb replicate agreement carries a small
smoothing bonus. (4) HMEC deep and HUVEC deep use the single-lane HMEC and HUVEC pair as a proxy, and HMEC /
HMEC deep and HUVEC / HUVEC deep are one set of inputs, not four independent cells; the held-out n is effectively
3 cell types. (5) The 413 regions are Akita's GM12878 test regions applied to all cells; the per-region
replicate agreement differs by region (section 18) and the medians are over regions, not over the cell's
genome. (6) Ceilings exist for only 2 of 12 training cells; the training median of ratios is a median of two.
(7) The consensus-alone column for training cells is the leave-one-out consensus; the no-consensus column is
all12honest_drop (not all12honest of `reference_summary.csv`, which has 0.828 at 25 kb).

Reproduce: `python scripts/replicate_ceiling_table.py` (CPU, seconds).

## 50. Mutation subtype of the Xu et al. patients as model input or stratifier (FINAL feasibility study, CPU only; section 49 was the last number when this was written)

Question: can the AML subtype or driver status of each patient be a model input or an evaluation stratifier, and is a test
statistically meaningful? Files: `results/aml_subtype/patient_subtype_table.csv` (25 AML Hi-C patients, source column),
`results/aml_subtype/FEASIBILITY.md`, `scripts/aml_subtype_table.py`.

Data. No per-patient mutation list exists in GEO (GSE152135 source name only), the text or the Suppl. Tables; the mutation
labels come from Fig. 1c of the paper (hand-read; fusions cross-checked against the SV junctions of Suppl. Table 5: all
agree). Consensus groups of the 25 AML Hi-C patients: TET2 6, KMT2B 6, CEBPA 5, RUNX1 4 (3 with t(9;22)), STAG2 3, FLT3-ITD
only 1; FLT3-ITD 6, NPM1 4, KRAS/NRAS 4. Suppl. Table 3 numbers (3, 4, 3, 4) are loop-harbouring sample counts, not group sizes.

Finding that changes earlier sections: patients 103 and 673 (and 576) are healthy PBMC donors (GEO source name, Fig. 1c "Normal
PBMC"), not AML. The 12 CUT&Tag "patients" of sections 36 and 38 are 10 AML plus 2 healthy donors; both are in the training
AML reference of the fold 0 test patients, and fold 1 tests 103 and 673. Section 29 states that the PBMC controls have no public
Hi-C; GSM4604271 to 73 are their maps.

Fold 0 (test 027 RUNX1, 270 KMT2B, 773 CEBPA, 1916 TET2) has one test patient per group; AML training patients 018 and 546 (TET2),
168 (RUNX1), 629 (KMT2B), 472 (STAG2), 1360 (FLT3-ITD only), no CEBPA. Same-group Pearson minus different-group Pearson over
the 413 test regions, 25 patients: 0.024 at 25 kb (permutation p 0.052), 0.021 at 10 kb (0.16), 0.017 at 5 kb (0.20); carried by
the TET2 group, confounded with the GEO batch.

Verdict: an indicator input is not learnable (1 to 2 training patients per group, collinear with patient identity, CEBPA
test group without a training patient, section 36 shows no headroom over the reference). Usable as evaluation stratifier and
as a group-matched reference (Hi-C only, leave-one-patient-out over 24 patients, 0 GPU h); MYCN-loop presence (TET2/FLT3-ITD,
CEBPA, STAG2/KRAS/NRAS versus RUNX1, KMT2B) is a 5 against 5 test among the CUT&Tag AML with minimal exact p 0.004 for one
locus (0.17 in fold 0 alone). Details in FEASIBILITY.md.

## 51. Number audit: which cited numbers come from honest validation (FINAL; section 50 was the last number when this was written)

`results/number_audit/` (CPU only, no training): `audit_table.csv` (89 claims of the progress page and of the chat
summaries with source, status and replacement), `AUDIT.md` (summary and ordered problem list),
`run_honesty_scan.csv` (524 training logs: yaml `val_exclude_overlap`, honest-split log line, verdict; script
`scan_runs.py`), `split_disjoint_5kb_2kb.csv` (no training window of the 5 and 2 kb 12-cell caches overlaps an
Akita test region or test window; the 25 and 10 kb caches are in `runs/multicell_reference/split_disjoint.csv`).

Counts: 41 HONEST, 23 LEAKY, 18 AFFECTED (depend on the healthy donors xu_103 and xu_673 counted as AML patients,
section 50), 7 UNVERIFIABLE. Main findings: (1) the page headline U-Net numbers (0.652, 68%, SCC row, scatter, seven
example panels, summary row 1) come from the overlapping-validation section 1 model; honest values 0.640, 66%,
Pearson 0.647 and SCC 0.611 on the 448-bin grid. (2) The resolution chart model series, the K562 cell-count curve,
three held-out "without consensus" dots and the four-motif row come from sections 8, 11 and 13 (overlapping
validation); honest all12honest values 0.6576 / 0.7857 / 0.8299 / 0.8277 and 0.771 / 0.735 / 0.805. (3) The seed
noise 0.013 is a section 8 number; honest two-seed spreads are 0.0002 to 0.0062 for consensus-input models and 0.004
to 0.028 without consensus. (4) "Beats Akita in 69%" is the overlapping model (honest 66%); "Hi-cGAN 0.196" has no
source (0.238 on 411 regions, 0.220 on 222). (5) Stale values: replicate agreement at 5 and 25 kb before the grid
correction, "0.822" in summary row 5 (12-cell median 0.828). The consensus-input claims 0.914 / 0.903 / 0.830 / 0.677,
consensus alone 0.898 / 0.859 / 0.785 / 0.629, honest_drop 0.775 / 0.761 / 0.684 / 0.544, 0.828 to 0.939, the Akita
region numbers (n = 411), the chromosome-held-out comparisons and sections 39, 40, 41, 48 are honest. Sections 29, 36,
38, 46 and 47 AML numbers need recomputing without xu_103 and xu_673 (not done here).

## 52. AML-inclusive cross-validation without the healthy donors, folds 10 to 12 (PENDING until the trainings and the evaluation finish; section 51 was the last number when this was written)

Reason: xu_103, xu_673 and xu_576 are healthy PBMC donors (GEO GSE152135 source name "Primary Healthy donor", Fig. 1c
"Normal PBMC", absent from Suppl. Table 2; section 50). The AML-inclusive folds of section 36 counted 103 and 673 as
training or test patients and put them in the AML reference. Fold 0 of section 36 (suffix-free runs, hard_f0, eval_f0,
mc_all12aml_*) is kept unchanged for the record; the folds 1 and 2 of section 36 were never trained (flag file
`AML_CV` absent). The partial fold-1 preparation made on 2026-10-10 on the remote machine (reference cools
`amltrain_f1_*`, cache `mc_all12aml_ref_25kb_f1`) was deleted.

Split (`results/aml_train/split_v2.json`, `scripts/aml_train.py`, fold ids 10, 11, 12 so that no name collides with the
old suffixes): the ten AML patients with Hi-C and CUT&Tag (018, 027, 168, 270, 472, 546, 629, 773, 1360, 1916), healthy
donors excluded (`HEALTHY`), patient-disjoint, groups from `results/aml_subtype/patient_subtype_table.csv`:

| fold | test patients (group) | training AML patients |
|---|---|---|
| 10 | 018 (TET2), 027 (RUNX1), 270 (KMT2B) | 168, 472, 546, 629, 773, 1360, 1916 (7) |
| 11 | 168 (RUNX1), 546 (TET2), 629 (KMT2B), 1360 (FLT3-ITD only) | 018, 027, 270, 472, 773, 1916 (6) |
| 12 | 472 (STAG2), 773 (CEBPA), 1916 (TET2) | 018, 027, 168, 270, 546, 629, 1360 (7) |

Every fold tests one TET2 patient; RUNX1 and KMT2B are tested in two folds, STAG2, CEBPA and FLT3-ITD only once; the two
patients without CTCF CUT&Tag (1360, 1916) are in different folds. 3 to 4 test patients per fold. Same donors (section 54): 168 and 496 are
one donor, 260 and 424 probably one (`SAME_DONOR` in aml_train.py). 496, 260 and 424 have no CUT&Tag and are in no fold
or training reference; `assert_same_donor_together` checks that no pair is split across train and test (no fold
reassignment was needed). The evaluation baselines (`amlclean_lopo_xu_<p>`, `amlclean_all`, from `baseline_labels`)
exclude the test patient's same-donor partner (the baseline of test patient 168 has no 496) and count every other
donor once (the second sample of each pair, 496 and 424, is dropped); 22 maps per leave-one-out baseline.
HSPC donors 050 and 213 and MOLM-13 train in every fold, U-937 WT and HSPC donors 341 and 822 are held out as before.

References: the AML reference of a fold is the mean proper O/E of the fold's training AML patients (leave-one-out for a
training patient, mean of all for MOLM-13 and the test patients), no test patient and no healthy donor. New for these
folds: the evaluation baselines "mean of all other AML patients" (`amlclean_lopo_xu_<p>`, `amlclean_all`) are built from
the AML patients only (24 others), where the section 36 baselines `amlcons_*` included the healthy donors.

Code: `aml_train.py --stage split [--exclude-healthy | --include-healthy]` (default new split; `--include-healthy`
reproduces and verifies the old split.json), `fold_of` accepts two-digit folds, `multicell_expanded` RUNS has
`all12aml_{refin_drop,honest_drop}_f{10,11,12}`. Jobs per fold: all12aml_refin_drop and all12aml_honest_drop at 25 and
10 kb (single GPU on the remote 4090, doubled per-rank batch, scripts/run_single_gpu.py), then `aml_train_eval.py --fold N`
with the results copied to `runs/aml_train_remote/fold<N>/`. Expected about 13 min per 25 kb and 30 min per 10 kb run
(measured 10 and 23 min for the 12-cell runs times 1.3), 40 to 80 min evaluation per fold. Results: PENDING.

## 53. Independence of the replicate pairs from each other and from the deep target maps (FINAL, metadata check; section 52 was the last number when this was written)

Question: for each cell of section 49, are rep1 and rep2 independent biological replicates, technical replicates or
half-splits, and are the replicates parts of the deep target map they are compared with. Method: ENCODE REST
(`/files/<ACC>/`, `/experiments/<ACC>/`, frame=embedded, retrieved 2026-10-10), 4DN REST (`/files-processed/<ACC>/`,
`/experiment-set-replicates/<ACC>/`, `/experiments-hi-c/<ACC>/`) and the repo's `scripts/replicates_all_cells.py` (`REPS`),
`scripts/encode_expand.py` (`HIC`, `PAIRS`, `LOCAL`). All accessions and fields: `results/replicate_independence/replicate_metadata.csv`
(one row per file, one per target; source URL per row). Nothing was recomputed; section 49 is unchanged.

Replicate pairs (section 7 / 49; fastq accession pairs R1 and R2, the lane pair that went through minibwa and hicBuildMatrix):

| cell | experiment | rep1: files, ENCODE rep, library, biosample | rep2: files, ENCODE rep, library(ies), biosample | relation of rep1 and rep2 |
|---|---|---|---|---|
| GM12878 | ENCSR410MDC, in situ Hi-C MboI | ENCFF991GHV + ENCFF075YWD, 1_8, ENCLB239DIW (HIC008), ENCBS386THJ (culture 2012-10-30, harvest 2012-11-02) | ENCFF416CVP + ENCFF997OUB, 2_1, ENCLB420YBU (HIC019), ENCBS538MCC (harvest 2013-01-05) | independent biological replicates: separate cultures of one cell line (donor record ENCDO000AAK), different libraries, no shared reads |
| IMR-90 | ENCSR852KQC, in situ Hi-C MboI | ENCFF346EOS + ENCFF983FKT, 1_5, ENCLB995LGB (HIC054), ENCBS532ZGJ (culture 2012-11-02, harvest 2012-12-03, passage 4) | ENCFF110XAL + ENCFF490HFV, 2_2, ENCLB126OSS (HIC056), ENCBS361XFK (culture 2013-07-17, harvest 2013-07-22, passage 4) | independent biological replicates (cultures 8 months apart, one donor record ENCDO000AAX) |
| K562 | ENCSR545YBD, in situ Hi-C MboI | ENCFF245VNR + ENCFF887BPM, 1_1, ENCLB941YRT (HIC069), ENCBS122OOP (culture 2013-03-20) | 8 lane pairs ENCFF017YRC/ENCFF740JAL, ENCFF077DJW/ENCFF942LOO, ENCFF299EZU/ENCFF272RRK, ENCFF307HYX/ENCFF570ZJK, ENCFF402DLV/ENCFF403MXC, ENCFF416MGD/ENCFF749NNL, ENCFF681AQK/ENCFF375MOP, ENCFF825VCF/ENCFF991GET; ENCODE 2_1 to 2_4, 4 libraries ENCLB243GST, ENCLB598XMB, ENCLB127KCJ, ENCLB463ZSY (HIC071 to 074), one biosample ENCBS037RBR (culture 2014-05-15) | independent biological replicates; rep2 is 4 technical libraries (8 lanes, 317.6 M read pairs) of one culture, rep1 one library (262.9 M read pairs) |
| HMEC | ENCSR711AVS, in situ Hi-C MboI | ENCFF055UHA + ENCFF416YDZ, 1_5, ENCLB179MJQ (iHIC6018), ENCBS240JYJ (culture 2017-01-10 to 2017-01-19) | ENCFF126WCO + ENCFF244GNV, 2_4, ENCLB145ZQA (iHIC6019), ENCBS544EML (culture 2017-01-19 to 2017-01-25) | independent cultures (two biosample accessions, 9 days apart) of one donor lot (ENCDO935FXX, Lonza lot 0000525584), different libraries |
| HUVEC | ENCSR855CCB, in situ Hi-C | ENCFF678PIY + ENCFF411KFA, 1_1, ENCLB655WVT (iHIC4925), ENCBS257NSY (culture 2017-03-06 to 2017-03-10) | ENCFF765YLB + ENCFF544HRQ, 2_1, ENCLB292XDZ (HIC080), ENCBS718VOW (culture 2012-10-31 to 2012-11-09) | independent cultures 4.4 years apart of one donor lot (ENCDO080ZCD, Lonza lot 7F3239) |

No pair is a technical replicate, a same-library pair or a half-split of one map. The ENCODE replicate numbers differ
(1 against 2) and so do the biosample accessions in all five pairs. The biological variation covered is culture to culture
(cell lines) and culture to culture within one donor lot (HMEC, HUVEC); donor-to-donor variation is not covered. Section 49
and 7 say "one sequencing lane each"; this holds for four pairs. K562 rep2 pools 8 lanes of 4 libraries (section 7 script docstring),
so the K562 pair is unequal in depth (144 M against 210 M contacts) and in composition, and the Spearman-Brown step
(equal-depth replicates) is approximate there.

Deep targets and what they contain:

| cell | deep target | what it merges | replicate libraries inside | replicate share of the target |
|---|---|---|---|---|
| GM12878 | 4DNFIXP4QG5B (mcool, 4.010e9 contacts at the 10 Mb level), set 4DNES3JX38V5 | 27 4DN experiments: HIC003 to 018, 021 to 029, 048, 049 (Rao 2014 in situ Hi-C) | rep1 library HIC008 yes; rep2 library HIC019 no | rep1: 138 M / 4010 M = 3.4% of contacts (4.6% of raw fastq bytes); rep2: 0% |
| IMR-90 | 4DNFIJTOIGOI (identified by the exact file size of `hyperparameter_publication/imr90.mcool`, 8,307,947,968 B; 1.083e9 contacts), set 4DNES1ZEJNRU | 7 experiments HIC050 to 056 (ENCSR852KQC bio 1 libraries 1_1 to 1_5 and bio 2 libraries 2_1, 2_2) | rep1 HIC054 and rep2 HIC056 both yes | rep1 165.5 M / 1082.5 M = 15.3%, rep2 18.3%, together 33.6% (fastq bytes 37.1%) |
| K562 | 4DNFI18UHVRO (0.907e9 contacts), set 4DNESI7DEJTM | 6 experiments HIC069 to 074 (ENCSR545YBD bio 1 and bio 2 completely; the 4DN records carry ENCODE:ENCLB and the rep2 fastq accessions ENCFF077DJW etc.) | rep1 and all 4 rep2 libraries yes | rep1 143.6 M / 907.1 M = 15.8%, rep2 23.2%, together 39.0% |
| HMEC deep | ENCFF943JRY (ENCSR711AVS, merged_all, biological_replicates [1, 2], 9 technical replicates 1_1 to 1_5 and 2_1 to 2_4; 1.260e9 contacts, 1.931e9 read pairs) | the whole experiment | rep1 (1_5) and rep2 (2_4) both yes | read pairs 243.5 M + 232.9 M = 476.4 M of 1931.2 M = 24.7% (12.6% and 12.1%); contacts 170.5 M + 166.7 M of 1260.3 M = 26.8% |
| HUVEC deep | ENCFF783KQI (ENCSR788FBI, intact Hi-C MNase, unreplicated, libraries 1_1 to 1_4; 1.107e9 contacts) | another experiment | no | 0%: other experiment, other donor record (ENCDO628LXZ, Lonza C2517A_0000300102, against ENCDO080ZCD of the pair), other assay |

The contact ratios divide our own hicBuildMatrix 25 kb totals (section 7) by the target totals of the 4DN or ENCODE
pipelines; the filters differ, so the ratios are accurate to a few percentage points (the fastq-byte shares bracket them for
GM12878 and IMR-90; for K562 bytes per read differ 4-fold between libraries and give 36% and 18%, contacts are used).
The HMEC read-pair ratio uses ENCODE `read_count` values of the fastqs and is the cleanest of the five.
In GM12878 the target also holds 16 of its 27 libraries (55% of the raw bytes) from rep1's biosample ENCBS386THJ (HIC003 to
018); rep2's biosample contributes nothing, so rep1 versus target shares a culture and rep2 versus target does not.

The deep intact pairs of section 12 and the `ceiling_deep_pair_sb` column: GM12878 ENCSR965OVH (ENCFF159BIG, rep 75_3,
ENCLB853FDT, ENCBS150RYT; ENCFF389TAA, rep 76_1, ENCLB562BJE, ENCBS649DHT) and K562 ENCSR676BEP (ENCFF740BHI, 5_3, ENCLB844LOP,
ENCBS846SIL; ENCFF891MHD, 7_4, ENCLB957WME, ENCBS418ZPY; DNaseI + MNase intact Hi-C) are independent biological replicates
(different biosamples, different libraries) of intact Hi-C experiments. They are not sources of the in situ 4DN targets.
The other three GM12878/K562 intact pairs in `PAIRS` (ENCSR653JLK, ENCSR946AKI, ENCSR178ENS) are likewise distinct
biosamples and libraries (rows in the CSV).

Consequences for section 49.

1. All five single-lane pairs are valid independent biological-replicate estimates of reliability. HMEC and HUVEC scored
   against their own rep1 (target = one member of the pair, the other member independent of it): ceiling sqrt(r_rep) is valid.
2. GM12878: the target holds 3.4% of rep1 and none of rep2. The ceiling is the cleanest in the table. Its deep-pair
   ceiling (intact, ENCSR965OVH) is independent of the target.
3. IMR-90, K562 and HMEC deep: both replicates are parts of the target (27% to 39% of its contacts or reads). The ceiling
   formula uses only rep1 versus rep2, so its value does not rise from the overlap, and the Spearman-Brown projection
   from the parts to a whole 2.6 to 3.7 times deeper than the pair stays a lower bound. The ceiling is then a split-half
   estimate of the target's reliability, not an independent-replicate bound; model-versus-target and rep-versus-rep agreement
   (`model_minus_replicate_r`) are compared at different depth and should be read as such. Quantities that correlate a
   replicate with the deep target are inflated by the shared reads and need the same caveat: the noise-floor fractions of the
   residual (section 39, "(rep1 - rep2)^2 / 2" against the target residual, K562 and HMEC deep) and `rep_dev_r`
   of `hard_split.py` (section 24; Pearson of R1 - C and R2 - C for IMR-90 and HMEC deep, where C holds the replicates; shared reads
   push the two deviations apart, so the filter drops windows more often for these cells). Neither was recomputed.
4. HMEC deep uses its own constituents as the pair (same experiment, donor, lot): the proxy is exact in biosample terms.
   HUVEC deep uses a pair from another donor (ENCDO080ZCD) and another assay (in situ MboI) than the target (ENCDO628LXZ,
   intact MNase): its ceiling is a proxy for a different measurement and carries that mismatch.
5. Independent samples. HMEC and HMEC deep are one biological sample: same donor lot, same experiment, the deep target
   contains the HMEC target (rep1) and rep2. HUVEC and HUVEC deep share inputs (same ChIP tracks and model predictions) but their
   targets come from two donor records and two experiments, so they are two independent target measurements on one set of
   inputs. For held-out cells the effective n is 3 cell types (K562, HMEC, HUVEC) with 4 independent targets (K562, HMEC,
   HUVEC donor 1, HUVEC donor 2); the 5 rows of section 49 are not 5 independent samples.
6. For K562 the "one lane each" description of the pair should read "rep1 one library, rep2 four libraries pooled".

Also found: `crosscell_hic/hmec_rep1.hic` and `hmec_rep2.hic` have the identical size (19,932,447,443 B, the size of
ENCFF943JRY) and identical first 300 MB and last 100 MB; both are copies of the merged deep file, not replicates. The
section 49 replicate maps are the minibwa/hicBuildMatrix maps of `replicates_all_cells.py`, not these files.

Replicate files outside section 49 (`scripts/all_replicate_scc.sh`): 4DN `track_and_facet_info.replicate_info` gives
HFFc6 DpnII in situ Hi-C 4DNFINU26HEQ (Biorep 1, Techrep 1) and 4DNFIVTQ72WM (Biorep 2, Techrep 1): biological; GM12878
HindIII dilution Hi-C (Selvaraj 2013) 4DNFII7498B3 and 4DNFIV6HB4VT (Biorep 1 and 2): biological; H1 DpnII in situ Hi-C
4DNFI29JL6ZO (Biorep 1, Techrep 1) and 4DNFIKTHE2OT (Biorep 1, Techrep 2): technical replicates of one biological replicate.
GSE155971 (HCT116 MCM2-AID, G1 DMSO and AUX rep1, rep2): the series lists lane-level samples G1_DMSO_1 to 4 and G1_DMSO_r2_1 to 4
without stating how the repo's rep1 and rep2 files map to them.

Not retrieved or not verified.
(a) The identity of the on-disk replicate cools with the accessions rests on the `REPS` dictionary and the download log; the
cools were not re-derived from their contents. (b) The ENCSR855CCB restriction enzyme (MboI) is taken from the script docstring;
the portal field was not read. (c) IMR-90 deep = 4DNFIJTOIGOI follows from the exact byte size and the 4DN date
(2018-03-05) of the file; the 4DN file record itself does not name the cell in the fields read. (d) The HMEC deep file on
disk (`hmec_rep1.hic` as ENCFF943JRY) relies on the md5 check noted in `encode_expand.py`; not repeated. (e) Contact ratios
mix pipelines (see above); the 4DN portal gives no contact counts. (f) `hmec_rep1.hic` versus `hmec_rep2.hic` was compared on
the head and tail only. (g) The size of the bias in item 3 (noise-floor fractions, `rep_dev_r`) was not quantified.
Source: ENCODE REST and 4DN REST responses of 2026-10-10 (cached JSON in the session scratchpad, not in the repo).


## 54. Are Xu et al. patients 168 and 496 the same donor? (FINAL; section 53 was the last number when this was written)

`scripts/aml_duplicate_check.py` (CPU), tables in `results/aml_duplicate_check/`, figure `duplicate_check.png`. Unit of
comparison is the sample pair: 231 pairs of the 22 patients with SVs in Suppl. Table 5, 378 pairs of the 28 patient Hi-C
samples (3 of them healthy donors; with the 4 HSPC donors 496 pairs in `hic_scc_chrom.csv`), 190 pairs of the 20 WGBS samples.

**Verdict: the same donor (two samples of one patient). Strong evidence from structural variants and copy number; Hi-C and
methylation similarity is consistent with it but not decisive alone; a germline genotype comparison is out of reach.**

| Evidence | 168 vs 496 | all other pairs |
|---|---|---|
| SV endpoints equal to the base pair | 20 | median 2, max 8.5 |
| SVs matched at both ends within 100 kb (of 30 and 36 SVs) | 11.5 | median 1.75, max 10 (504 vs 798, both inv(16)) |
| SV endpoints within 100 kb | 27 (45% of the smaller call set) | median 4, max 20.5 |
| CNV segment edges within 500 kb | 10 of 10 segments | max 2.5 |
| CNV 500 kb altered-bin state Jaccard / log2 profile r | 0.85 / 0.944 | Jaccard max 0.75 (673 vs 1324, one segment each), r max 0.853, median 0.14 |
| Hi-C SCC (100 kb, up to 10 Mb, mean over chr1 to 22) | 0.810 (rank 2 of 378; first on 10 of 22 chromosomes) | median 0.56; 260 vs 424 0.814, 103 vs 576 0.800 |
| WGBS, 100 kb windows, cohort-centred r (raw r) | 0.783 (0.958) | median -0.02, max 0.569 (103 vs 673); raw max 0.950 |
| Patient-specific loops, Suppl. Table 1 (exact 10 kb anchors) | 16 shared, Jaccard 0.038 | median 0.002, max 0.018 |
| Stripe anchors, Suppl. Table 4 | 333 shared, Jaccard 0.219 | median 0.049, max 0.183 (424 vs 933) |
| Sex | both female: chrX cis density 1.146 and 1.129 of autosomal (males 0.49 to 0.60); chrY-end pairs 232 and 204 per million (male controls 027 3594, 270 3526) | consistent, uninformative |

**Breakpoints** (`sv_breakpoints_168_496.csv`, `cnv_segments_168_496.csv`). 30 of the 60 endpoints of 168 and 24 of the 72 of
496 lie within 100 kb of an endpoint of the partner; apart from the BCR-ABL1 translocation none lies within 100 kb of any of the
20 other patients (private to the pair). Base-pair identical coordinates in both: dup chr1:31,002,710 to 63,329,421 and t(1;5)
chr1:57,739,716 to chr5:107,483,785; the same 10 kb coordinates in both: del chr1:56.09 to 195.22 Mb, del chr13 from 86.89 Mb,
inv chr3:125.73 to 130.04 Mb, del chr4:29.95 to 32.96 Mb, del chr11 from 96.49 Mb, del chr1 from 68.62 Mb. Recurrent across
patients, not evidence: t(9;22) BCR-ABL1 (168 and 496 coincide at 10 kb, 027 is 60 kb away), chr16:34.57 to 46.39 Mb dup/inv
(also 027 and 424; the pericentromeric hg38 region) and chr8:8.14 to 12.5 Mb inv (also 027; the 8p23 inversion polymorphism).
CNV: del(5q) chr5:107.5 to 181.5 Mb, chr1:70 to 121.5 Mb (102 bins in both), chr1:63.5 to 68.5 Mb, chr4:30 to 33 Mb and the
chr1q gain block (195.5 to 246.5 Mb) have the same edges within 500 kb; 496 additionally has chr1:0.5 to 3.5 Mb and chr7:56.5
to 64 Mb losses, 168 additionally a chr1:38.5 to 47 Mb gain. 48 of the 72 endpoints of 496 and 30 of the 60 of 168 are not in the
partner (mostly translocations). The differences are those of two samples of one genome with clonal changes, not of two
leukaemias. MOESM12 b gives CN > 2 / CN < 2 bin counts of 9483 / 13630 (168) and 9733 / 13618 (496).

**Metadata.** GEO titles `168_Hi-C` (GSM4604288, submitted 2020-06-09, BioSample SAMN15192043) and `496_Hi-C` (GSM5688518,
submitted 2021-11-12, SAMN23102357), both series GSE152135 and GSE152136, both source "Primary AML", cell type "Selected PBMC or BM
cells", NovaSeq 6000. Source: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5688518&targ=self&form=text&view=full`
and the same URL with acc=GSM4604288; the HTML page returned an NCBI bot check, the text record was used and the titles quoted
from the raw text (saved by WebFetch). 496 belongs to the 2021 addition of 8 patients (1071, 1324, 424, 496, 504, 798, 838,
933), which have Hi-C, RNA-seq and WGBS but no CUT&Tag. The supplementary tables of the paper (MOESM10 to 13) hold no timepoint,
pairing or clinical annotation.

**Out of reach.** Germline SNP concordance: the repository and the downloaded GEO files hold no BAM, VCF or read sequence
(pairs files carry coordinates only, the WGBS is a 10 bp processed bigWig, the mcool files lack chrY). It needs the SRA reads of
the two Hi-C runs and would be the decisive test. CUT&Tag cannot be compared (496 has none).

**Second pair found by the same screen: 260 and 424.** Base-pair identical private translocations t(2;3) chr2:159,292,423 to
chr3:58,786,722 and t(4;20) chr4:13,783,536 to chr20:30,377,207 (no other patient within 100 kb); t(19;22) within 9 bp (also
018 and 629 near it), Hi-C SCC 0.814 (the highest of all pairs), WGBS cohort-centred r 0.535 (third). 260 lists 6 SVs and 424
11, so the evidence is smaller than for 168/496 but the two private events are the same kind. Probably one donor. Both are
WGBS-only (no CUT&Tag) and are in no AML training fold.

**Affected analyses and fix.**
1. AML-inclusive training (section 36, `results/aml_train/split.json`, 12 patients): 496 has no CUT&Tag and is in no fold, 168 is
   a test patient of fold 1, so no copy of 168 trains; the split is clean. `aml_train.SAME_DONOR` asserts 168 and 496 on one side
   for the v2 folds 10 to 12 (section 52); add ("260", "424"). The AML consensus reference is contaminated: `amlcons_lopo_xu_168_*`
   leaves out 168 only and still averages 496, so the consensus-based references and scores of 168 are optimistic (the 496 map
   is 0.81 SCC to 168); same for 260 and 424. Fix: rebuild the leave-out consensus of 168 without both 168 and 496 (and 260 / 424).
2. Loop classifier (section 47, 18 patients incl. 168 and 496): LOPO and plain LOPO test 168 while 496 (same loop anchors, 16 exact
   loop and 333 stripe anchors shared) trains, which inflates the AUROC for those two and 260 / 424; the locus-disjoint filter
   (training anchors within 2 bins of a test anchor dropped) removes the shared loci only. Leave-subtype-out is unaffected
   (BCR-ABL1 {027, 168, 496} is one group). Fix: one fold unit for {168, 496} and {260, 424} in `make_folds`, or drop 496 and 424
   (the WGBS-only copies) and report 16 patients.
3. CTCF methylation (section 46, 9 patients with CTCF CUT&Tag and WGBS): 496, 260 and 424 are absent, no duplicate inside the 9;
   unaffected.
4. AML-sites figures (`aml_sites_figures.py`): use patients with CUT&Tag; 496, 260 and 424 are not among them; unaffected, but any
   "n patients" statement for Hi-C or WGBS counts samples, which overstate the number of donors by 2.

## 55. Cohesin-fraction input and cohesin-loss augmentation (FINAL; section 54 was the last number when this was written)

Problem (sections 21, 25, 32): the RAD21 degron is predicted with a change-map correlation of 0.80
(10 kb) when the measured auxin ChIP is given, but a RAD21 track declared absent moves the loops by
+0.03 log O/E (measured -0.73), 20.8% of the loops come out weaker, and the U-937 STAG2 knockout stays
wrong; the cohesin-unknown indicator (section 32) did not help. Diagnosis in
`results/cohesin_fix/DIAGNOSIS.md` (`scripts/cohesin_fix_diagnose.py`, 184 even-chromosome Akita test
windows of HCT116, all12deg_refres_drop_10kb): the change predicted from an absent RAD21 has a spread of
0.034 against the measured 0.319 (11%), from the measured auxin track 0.202 (63%); loops, borders and the
distance profile all move by 0 to 5% of the measured amount. The model reads the peak-to-background
contrast of the RAD21 track (contrast scaled to 0.26 around the window median: Pearson 0.70, spread
0.116; every value scaled to 0.26: Pearson 0.08, spread 0.016; absent: spread 0.034), not the level, not the
consensus, not the CTCF or histone tracks (auxin CTCF only: -0.10). Cause: with track dropout an absent
RAD21 means "not measured" in every cell with an intact map (five of the twelve have no cohesin ChIP), the
only low-cohesin example is one sample, and the cohesin-unknown indicator is 1 exactly where loops are
intact (response to an absent RAD21 0.042 with the indicator, 0.070 without). The degron sample was also
encoded with its own contact scale (1.316 against 1.914), which stretches the weaker map (predicted spread
0.254 decoded with its own scale against 0.202). The STAG2 knockout is a different perturbation: loops
+0.08 (short -0.105, long +0.181), clone-pair ceiling of the change map 0.25, change map correlated with
the RAD21 degron change at 0.14, opposite sign of the mean change at 100 to 250 kb (+0.018 against
-0.021); a total-cohesin scalar cannot span it and RAD21/SMC3 ChIP of the wild type cannot encode the loss
of STAG2-specific cohesin.

Fix (all default off). (1) Cohesin-fraction row (`data.cohesin_fraction`, `samples[].cohesin_fraction`):
one constant input row, log1p(f)/ln 2 - 1 (1 intact = 0, none = -1), a statement about the biology and not
about which ChIP was measured (1 for every sample, also those without RAD21/SMC3 ChIP; 0.26 for the 6 h
auxin sample: RAD21 enrichment above the median in the top 1% untreated bins, auxin over untreated, odd
chromosomes). (2) Cohesin-loss augmentation (`train.cohesin_aug` p = 0.25, `hicmultiome/cohesin.py`): a
training window of an intact sample is replaced by a synthetic depleted one with f drawn from
{0, 0.1, 0.26, 0.5, 0.75}: RAD21/SMC3 contrast scaled to f around the window median, or (share 0.5) set to
the absent code, the fraction row set to f, and the target moved by (1 - f)/(1 - 0.26) times the change of
the degron operator in log O/E units. The operator (`scripts/cohesin_operator.py`; 5 dilated 3x3
convolutions, 24 channels, symmetric, zero within 2 bins of the diagonal) is fitted on the odd
chromosomes of the HCT116 untreated/auxin pair only and predicts the change of the even-chromosome Akita
test windows from the untreated map alone at median window Pearson 0.834 (10 kb; high-pass part 0.516,
spread 0.257 against 0.338) and 0.877 (25 kb; 0.636; 0.244 against 0.309); the per-diagonal mean change
scores -0.087 / -0.119 and a single linear high-pass factor 0.288 / 0.432. (3) `samples[].norm_like`:
hct116_auxin takes the contact scale of hct116. Runs all12cohfix_refres_drop (all of it) and
all12cohfixna_refres_drop (row and shared scale only, no augmentation; ablation), 25 and 10 kb, otherwise
the degron recipe of section 25 (hct116 reads the untreated degron ChIP, hct116_auxin the 6 h auxin Hi-C
and ChIP on the odd autosomes, uniform-count track dropout 0.5, residual on the leave-one-out consensus,
honest validation). One cache (`mc_all12cohfix_refres_drop_<k>kb`) serves both; the operators sit on the
share (`cohesin_fix/operator_<k>kb.pt`).

Leakage safeguards. The operator sees HCT116 pairs of the odd chromosomes only; its even-chromosome
numbers come from the Akita test windows and no choice (architecture, 30 epochs) was made on them. The
auxin sample trains on odd chromosomes only; the even chromosomes are the evaluation. Synthetic windows are
made on the fly from training windows only, after the honest validation split (no training window overlaps
a validation position in any sample); validation windows are not augmented. The consensus of each
sample excludes it (leave-one-out; hct116_auxin uses the hct116 one). The CTCF degron, the washout maps and
the U-937 knockout are never trained on. The track normaliser statistics are fitted on all training
chromosomes of every sample, the auxin sample's even chromosomes included (ChIP signal only, as in
section 25).

Evaluation (same scripts and tables as for the degron models; new rows): `perturb_validate.py`
(`minus_RAD21` fraction 1, `minus_RAD21_f0`, `minus_RAD21_f0.5`, `measured_chip` with fraction 0.26,
`minus_CTCF`), `washout.py` (the fraction row follows 0.26 + f (1 - 0.26)), `aml_eval.py` perturb (variants
u937_ko_absent_f0, _f0.5, u937_ko_f0.5 and _f0.75), intact performance (`multicell_expanded.py --stage eval`
and summary; the augmentation must not lower the 413-region medians of all12deg_refres_drop), and
`cohesin_fix_diagnose.py` on the trained model. Acceptance: change Pearson on the even chromosomes and loops
weaker for `minus_RAD21_f0` at least those of `measured_chip` of all12deg_refres_drop (0.80, 98.4%),
CTCF degron (held out) not below 0.40, washout monotone, intact medians within 0.005. A fraction row that
the model ignores (spread of `wt_f0` below 0.1 of the measured) counts as a failure.

Results (FINAL; `results/cohesin_fix/RESULTS.md`, `results_*.csv`, `scripts/cohesin_fix_results.py`; single seed;
10 kb / 25 kb; deg = all12deg_refres_drop, cohfix = all12cohfix_refres_drop, cohfixna = row and shared contact
scale only). All four runs trained (cohfix 25 kb and both cohfixna on the local pair, cohfix 10 kb on the remote
machine; perturbation, washout and AML predictions on the remote machine). Intact accuracy: cohfix within
-0.0022 of deg in all comparisons (12 training cells 0.9279 / 0.9336 against 0.9237 / 0.9358, 5 held-out cells
0.9060 / 0.9133 against 0.9039 / 0.9037); cohfixna loses 0.0264 on the 10 kb held-out cells and 0.0050 on the
25 kb training cells. RAD21 degron with the fraction stated: spread of the predicted change at fraction 0 is
97% / 84% of the measured (deg, RAD21 absent: 11% / 14%; threshold 10%), loops weaker 100% / 98.5% (deg absent
20.8% / 33.1%), loop change -1.14 / -0.49 at fraction 0 (measured -0.73 / -0.48) and -0.65 / -0.34 with the
measured auxin track at fraction 0.26 (deg -0.49 / -0.20), change-map Pearson 0.784 / 0.852 at fraction 0 (the
strict "at least 0.80" misses by 0.019 at 10 kb) and 0.839 / 0.888 at fraction 0.26 (deg 0.803 / 0.842).
Washout correct and closer to the measured: median change -0.67 / -0.32 (deg -0.36 / -0.12), monotone recovery;
CTCF degron held out 0.403 / 0.402 (floor 0.40 met; 25 kb 0.115 below deg 0.517). RAD21 absent as ChIP only
(row 1) is unchanged (16.9% / 26.2% loops weaker): the response runs through the row, the RAD21 track shape
alone gives Pearson -0.27 (deg 0.71). STAG2 knockout stays wrong in size and sign (ko_f0.5 loop change -0.23 /
-0.32 against +0.08 / -0.03 measured; the change-map Pearson 0.20 / 0.26 approaches the 0.25 ceiling at 10 kb);
ko_half unchanged. AML rows: MOLM-13 +0.034 / +0.012 over deg, U-937 wild type +0.007 / +0.011. Ablation:
row and scale alone give the change map at the measured state (Pearson 0.85 / 0.89, spread 70% / 61%) but do not
extrapolate (spread at fraction 0 is 58% / 49%, below that at 0.26; washout recovery at 0.25 is 0.13 at 10 kb);
the augmentation supplies the graded response and the held-out intact accuracy at a cost in CTCF-degron
generalisation at 25 kb (0.402 against 0.602 without it). Limitations: one seed, one degron cell line (RAD21,
6 h), the operator from one cell, synthetic augmentation, HFF cohesin not re-tested. Verdicts: (a) pass,
(b) pass with the 10 kb f0 Pearson miss, (c) pass, (d) not solved (as expected), (e) pass, (f) see above.
The queue's diagnose of cohfixna failed on a cache-name fallback and was rerun with
`--cache mc_all12cohfix_refres_drop_<k>kb`; `aml_eval.py --stage score` was run for 10 and 25 kb (28 rows added
to `results/aml/summary.csv`, existing values unchanged).

## 56. CTCF binding, motif methylation and insulation in the Xu et al. AML patients without the healthy donor (FINAL; recomputation of section 46; section 55 was the last number when this was written)

`scripts/aml_ctcf_methylation.py` with the new flag `--exclude-healthy` (default) and
`--include-healthy` (original section 46 behaviour), tables and figures in
`results/aml_ctcf_methylation_aml_only/` (CPU only; `results/aml_ctcf_methylation/` is untouched).
Reason: section 50 found that xu_103 is a healthy PBMC donor (also xu_576 and xu_673; 673 has no
CUT&Tag), so section 46 pooled 8 AML patients plus one healthy donor. All stages (q1, q2, q3) were
rerun with every definition and sensitivity setting of section 46 (same thresholds, same 18-definition
grid, same coverage filters, same control matching, same bootstrap sizes) over the 8 AML patients
018, 027, 168, 270, 472, 546, 629, 773. Sample 103 is a separate healthy-donor row in the per-sample
tables and enters no pooled number. `--include-healthy --out <dir>` reproduces every number of
section 46 exactly (all q1, q2 and q3 tables compared, maximum difference 0.0, including the bootstrap
intervals).

### 1. Before and after (even chromosomes for q1; all autosomes for q2 and q3)

Before = section 46 tables (9 samples incl. 103), after = this section (8 AML patients). Intervals are
chromosome bootstrap 95 percent intervals. The pre-set insulation interval is quoted from
`q3_insulation_summary.csv` (-0.0016), section 46 text rounded it to -0.003.

| quantity | before (9 incl. 103) | after (8 AML) |
|---|---|---|
| pooled median partial Spearman, given strength | -0.37 (-0.40 to -0.33) | -0.32 (-0.35 to -0.29) |
| same, mean over samples | -0.36 | -0.36 |
| + H3K27ac / + flank | -0.30 / -0.14 | -0.25 / -0.13 |
| without 773, given strength | -0.32 (-0.35 to -0.30) | -0.27 (-0.31 to -0.24), 7 patients |
| relative risk, definition A | 2.19 (1.80 to 2.52) | 2.13 (1.75 to 2.45) |
| relative risk, definition B | 2.91 (2.39 to 3.44) | 2.89 (2.38 to 3.41) |
| precision A / B | 0.43 / 0.47 | 0.44 / 0.48 |
| sensitivity A / B | 0.063 / 0.082 | 0.067 / 0.089 |
| relative risk A without 773 / B without 773 | 2.07 / 2.57 | 2.00 / 2.54 |
| gain-and-lost sites A / B (pooled) | 208 / 226 | 204 / 223 |
| gain-kept A (pooled) | 274 | 265 |
| 18-definition grid: relative risk range; sensitivity range; gain-and-lost range | 1.92 to 4.08; 0.045 to 0.11; 80 to 511 | 1.90 to 4.03; 0.047 to 0.12; 78 to 497 |
| reference depth >= 20 / >= 40 reads, relative risk (gain sites) | 2.60 (84) / 4.35 (6) | 2.52 (83) / 4.34 (6) |
| lost sites with no motif CpG; testable; testable with gain (A) | 26,809 of 38,441; 3,299; 208 (6.3 %) | 23,890 of 34,476; 3,043; 204 (6.7 %) |
| same-assay check (vs other patients), pooled relative risk | 1.33 | 1.29 |
| insulation, A, w = 10: test sites; paired difference | 195; +0.028 (-0.002 to +0.058) | 191; +0.027 (-0.003 to +0.057) |
| same, controls also matched on baseline insulation | 190; +0.028 (+0.003 to +0.053) | 186; +0.026 (+0.001 to +0.052) |
| insulation, B, w = 10 | 216; +0.026 (-0.002 to +0.053) | 213; +0.025 (-0.003 to +0.052) |
| insulation, A, w = 5; matched; B | +0.032 (-0.001 to +0.062); +0.035 (+0.010 to +0.059); +0.033 (+0.004 to +0.063) | +0.030 (-0.004 to +0.061); +0.032 (+0.006 to +0.057); +0.032 (+0.003 to +0.061) |
| against gain-kept controls, A, w = 10 / w = 5 | +0.023 / +0.001 | +0.023 (-0.011 to +0.057) / -0.001 |
| lost without gain (A, w = 10) | -0.005 (-0.016 to +0.006), 2,688 sites | -0.003 (-0.014 to +0.007), 2,470 sites |
| gain with CTCF kept (A, w = 10) | -0.028 (-0.042 to -0.012) | -0.028 (-0.042 to -0.012) |

Per patient values of section 46 for the 8 AML patients are unchanged (q1 given strength 018 -0.18, 027
-0.58, 168 -0.27, 270 -0.18, 472 -0.25, 546 -0.51, 629 -0.37, 773 -0.52; relative risk A 027 9.9, 773
6.7, 546 5.7, 629 3.3, 472 1.7, 018 1.7, 168 1.5, 270 1.2; insulation A w = 10 546 +0.076, 018 +0.065,
773 +0.032, 027 +0.019, 168 +0.017, 629 -0.024, 472 -0.035, 270 -0.057). The pooled median drops by 0.05
because 103 (-0.39) was above the middle of the nine values; the median of eight is the mean of the
fourth and fifth value. All eight intervals exclude 0 (formerly nine). Against the non-AML cells
(-0.60) the gap widens from 0.23 to 0.28.

### 2. Healthy donor 103 as a separate row (same definitions, CD14+ / CD34+ references)

| quantity | 103 | AML pooled |
|---|---|---|
| partial Spearman given strength (even chr) | -0.395 (-0.42 to -0.36) | median -0.32 |
| median log1p CTCF height, motif methylation < 0.2 / > 0.8 | 0.87 / 0.33 | 027 1.87 / 0.31, 018 0.47 / 0.42 |
| gain-and-lost sites A / B | 4 / 3 | 204 / 223 |
| relative risk A / B | 2.19 (0.0 to 4.09) / 1.75 (0.0 to 3.63) | 2.13 / 2.89 |
| sensitivity A / B | 0.02 / 0.01 | 0.067 / 0.089 |
| testable lost sites with a methylation gain (A) | 4 of 256 (1.6 %) | 204 of 3,043 (6.7 %) |
| same-assay relative risk vs median of the 8 AML | 2.40 (10 gain-and-lost) | 1.29 |
| insulation gain-and-lost vs no-gain, A, w = 10 | +0.082 (4 sites, no interval) | +0.027 |

In the normal cell the methylation-CTCF association is as strong as in the AML patients with moderate
CUT&Tag quality (-0.39), but the number of sites that gain methylation relative to monocytes and lose
CTCF is 4, so the 2x2 and insulation numbers of 103 carry no information.

### 3. Is the healthy reference (CD14+ WGBS, CD34+ CTCF ChIP-seq) still appropriate, and is 103 better matched?

Rank agreement between each AML patient and each candidate reference (even chromosomes,
`refcheck_*.csv`, descriptive).

| comparison | result |
|---|---|
| CTCF height of patient vs CD34+ ChIP-seq (Spearman, 35,205 sites) | 0.15 to 0.52, median 0.27 |
| CTCF height of patient vs 103 CUT&Tag | 0.18 to 0.34, median 0.25 |
| CTCF height of patient vs median of the other AML patients | 0.25 to 0.57 |
| 103 CUT&Tag vs CD34+ ChIP-seq | 0.33 |
| motif methylation of patient vs CD14+ WGBS (1,810 to 1,824 sites) | 0.82 to 0.87 |
| motif methylation of patient vs 103 WGBS (6.1 thousand sites) | 0.83 to 0.86 |
| 103 WGBS vs CD14+ WGBS (1,828 shared sites) | 0.87; mean difference -0.003; 3.9 % differ by >= 0.3 |
| insulation profile Pearson r (w = 10) patient vs HSPC mean / PBMC mean (576, 673, 103) / 576 alone | 0.77 to 0.92 / 0.77 to 0.86 / 0.76 to 0.84 |
| HSPC mean vs PBMC mean insulation profile | r = 0.84 (w = 10), 0.80 (w = 5) |

For methylation the two references agree (103 and CD14+ are interchangeable at the motif CpGs); 103
covers 3.4 times more motif sites because the CD14+ table is restricted to >= 10 reads, but 103 has no read
counts either. For CTCF, the CD34+ ChIP-seq predicts the patient CTCF rank at least as well as 103 and
clearly better for the three patients with good CUT&Tag (027 0.52 vs 0.34, 546 0.46 vs 0.33, 773 0.47
vs 0.32); the other five patients are equal within 0.05. 103 is a single donor of a lymphocyte-dominated
mixture, so site-specific donor and lineage differences cannot be separated from noise. The same
laboratory and assay does not make 103 the better CTCF reference, the CTCF height of 103 has the same
low signal-to-noise as the weak patients (height vs strength 0.22). Conclusion: CD14+ / CD34+ stays the
primary reference; 103 is reported as a labelled variant. For insulation the HSPC maps are closer to 6 of
the 8 patients at both window sizes (PBMC closer only for 270 and 629), so HSPC stays the Hi-C baseline;
the healthy PBMC Hi-C maps (GSM4604271 = xu_576, GSM4604272/73 = xu_103 and xu_673) exist in
`aml/oe/` and are used as extra baseline variants (for patient 103 itself the baseline is 576 and 673).

Variants (all labelled `variant` in the tables; 8 AML patients pooled; `q2_variant_d103_*.csv`,
`q3_insulation_summary.csv`):

| variant | gain-and-lost A / B | relative risk A | relative risk B | sensitivity A / B | grid relative risk range |
|---|---|---|---|---|---|
| primary: CD14+ methylation, CD34+ CTCF | 204 / 223 | 2.13 (1.75 to 2.45) | 2.89 (2.38 to 3.41) | 0.067 / 0.089 | 1.90 to 4.03 |
| d103: 103 WGBS and 103 CUT&Tag | 757 / 573 | 1.63 (1.42 to 1.84) | 1.92 (1.63 to 2.17) | 0.075 / 0.087 | 1.50 to 2.29 |
| d103ctcf_cd14meth: 103 CUT&Tag, CD14+ WGBS | 215 / 170 | 2.01 (1.70 to 2.32) | 2.43 (1.98 to 3.08) | 0.067 / 0.080 | 1.73 to 2.73 |

Per patient relative risk A with d103: 027 2.94, 546 2.73, 773 2.69, 629 2.29, 270 1.64, 472 1.62, 168
1.39, 018 1.25 (interval of 018 reaches 0.98); the same three patients lead, the spread is compressed.
The d103 variant calls 3.7 times more gain-and-lost sites (the methylation reference covers more motif
sites and 103 is a noisier CTCF reference), and the association stays above 1 in every definition.

Insulation, pooled paired difference, gain-and-lost vs no-gain controls (A, w = 10; matched on
baseline insulation; w = 5):

| sites | Hi-C baseline | test sites | A w = 10 | A w = 10 matched | A w = 5 matched |
|---|---|---|---|---|---|
| primary | HSPC (4 donors) | 191 | +0.027 (-0.003 to +0.057) | +0.026 (+0.001 to +0.052) | +0.032 (+0.006 to +0.057) |
| primary | PBMC (576, 673, 103) | 191 | +0.025 (-0.011 to +0.062) | +0.031 (-0.002 to +0.068) | +0.043 (+0.008 to +0.081) |
| primary | 576 (GSM4604271) alone | 190 | +0.009 (-0.023 to +0.045) | +0.011 (-0.021 to +0.042) | +0.019 (-0.017 to +0.056) |
| d103 | HSPC | 709 | +0.001 (-0.026 to +0.021) | +0.004 (-0.026 to +0.024) | +0.010 (-0.005 to +0.024) |
| d103 | PBMC | 709 | +0.007 (-0.015 to +0.027) | +0.018 (-0.001 to +0.037) | +0.019 (+0.001 to +0.039) |

### 4. Change in conclusions

Directions and the qualitative conclusions of section 46 hold with the 8 AML patients: methylation at the
motif is negatively associated with CTCF height beyond motif strength in every patient (all 8 intervals
exclude 0, carried most by 027, 546, 773), sites that gain methylation lose CTCF about twice as often as
others (relative risk 2.1 for A, 2.9 for B, interval above 1 in all four pooled rows), motif
hypermethylation accounts for a small share of the lost sites (6.7 percent of the testable lost sites,
0.6 percent of all lost sites), and the insulation difference is about 0.03 weaker at gain-and-lost
sites with an interval that touches 0 for the pre-set matching. Changes: the pooled median partial
correlation is -0.32 instead of -0.37 and the AML-to-non-AML gap is larger; relative risks fall by 0.06
(A) and 0.02 (B); counts of gain-and-lost sites fall by 4 and 3; the insulation estimates move by at most
0.003. Added qualifications: the insulation effect depends on the reference (+0.001 with the 103
reference, +0.009 against the single PBMC map 576, +0.025 against the PBMC mean, +0.027 against HSPC), all
five estimates are positive (+0.001 to +0.027) but the size is not established and only the HSPC and PBMC
baselines with the primary sites give intervals that exclude 0 after insulation matching; the methylation-CTCF relative risk is
1.6 to 2.9 in every reference variant.

Limits added: 103 is one donor with unknown WGBS coverage (no read counts); the PBMC baselines have only
three maps (576, 673, 103) of which two have no CUT&Tag; the insulation profile similarity compares each
patient with the baselines, not the baselines among themselves for noise (HSPC donor to donor was not
computed). Limits (i) to (viii) of section 46 apply unchanged.

Figures `fig_q1_partial_spearman.png` (103 in blue), `fig_q2_precision_sensitivity.png` (103 last) and
`fig_q3_insulation.png` (primary variant) in `results/aml_ctcf_methylation_aml_only/`.


## 57. Loop classifier of section 47 without healthy donors and with leave-one-donor-out folds (FINAL; recomputation of section 47; section 56 was the last number when this was written)

`scripts/aml_loop_classifier.py` got two flags, `--exclude-healthy` (drops xu_103, xu_673, xu_576 from every patient list)
and `--group-donors` (folds, bootstrap and per-patient AUROC run over donor units; {168, 496} and {260, 424} are one
unit; leave-subtype-out unchanged, the subtype of a unit is that of its samples). Both are on by default, the opposite
flags (`--no-exclude-healthy --no-group-donors`) reproduce section 47 (checked: loops_meth_Ahspc gives 0.820454 and
plain 0.848212 as in section 47). Output goes to `results/aml_loop_classifier_v2/` (default; `_legacy` suffix when a
flag is off; the section 47 folder is not touched). All 14 experiments of section 47 were re-run (all stages: the Hi-C and
track caches are per patient and unchanged, features rebuilt; gradient boosting and logistic regression, locus-disjoint LOPO,
plain LOPO, LSO, shuffle controls (20 and 10 repetitions), group importance, increments, univariate, pair types, figures).

### Findings about the data
- The healthy donors never entered section 47's loop and stripe patient lists (103, 673, 576 have no rows in Suppl.
  Tables 1 and 4; the lists are the 20 AML patients). Rebuilt `features.csv.gz` is identical to section 47's (26,352 rows,
  `DataFrame.equals` True). The healthy flag is therefore a guard (the model stage asserts that no healthy donor is in the
  features) and changes no number.
- 168 and 496 enter the same experiment only in the 18-patient (WGBS) experiments; the 8- and 10-patient experiments contain
  168 without 496 (496 has no CUT&Tag), 260 and 424 are only in the 18-patient set. The donor grouping therefore changes
  only `loops_meth_*` (18 samples, 16 donor units, held-out unit = both samples). All other numbers are identical to
  section 47 to the last digit.

### Before / after (gradient boosting, LOPO locus-disjoint, macro AUROC; patient bootstrap over donor units now)
| quantity | section 47 | this section |
|---|---|---|
| loops all data, A_do, 8 patients (8 donors) | 0.880 | 0.880 [0.853, 0.902] |
| loops all data, A_full | 0.874 | 0.874 [0.841, 0.903] |
| loops all data, A_hspc | 0.837 | 0.837 [0.805, 0.865] |
| loops all data, B_rand | 0.792 | 0.792 [0.760, 0.826] |
| 10 patients K27ac+K27me3, A_hspc | 0.833 | 0.833 |
| 18 samples methylation/motif/gene, A_hspc (16 donors) | 0.820 [0.802, 0.840] | 0.822 [0.802, 0.844] |
| 18 samples, A_do | 0.875 | 0.875 [0.857, 0.891] |
| 18 samples, B_rand | 0.703 | 0.701 |
| increment over HSPC state, A_hspc, + own marks (8) | +0.023 | +0.023 [0.018, 0.028] |
| increment, A_full | +0.015 | +0.015 |
| increment, A_do | +0.006 | +0.006 [-0.001, 0.012] |
| increment, 10 patients, A_hspc | +0.015 | +0.015 |
| increment, 18 samples, methylation + motif, no own marks, A_hspc | +0.007 [0.004, 0.009] | +0.006 [0.003, 0.009] |
| H3K27ac patient-minus-HSPC, univariate AUROC (A_hspc) | 0.617, 8/8 patients above 0.5 | 0.617, 8/8 donors above 0.5 |
| pair types A_hspc (pooled): P-E-like / P-S-like | 0.864 / 0.741 | 0.864 / 0.741 |
| stripes A_hspc / A_full / B_rand | 0.608 / 0.674 / 0.810 | 0.608 / 0.674 / 0.810 |
| plain LOPO, 18 samples A_hspc (inflated by the partner in training) | 0.848 | 0.850 (still plain patient-wise folds, over donor units) |
| held-out 168 (A_hspc, 18 samples): sample alone / donor unit | 0.857 (496: 0.787) | 0.822 (unit 168+496) |
| held-out 260 / 424 (A_hspc): alone / donor unit | 0.755 / 0.798 | 0.766 (unit 260+424) |
| units remaining | 18 patients | 18 AML samples = 16 donor units (8 and 10 in the CUT&Tag sets, unchanged); 0 healthy donors |

Controls unchanged (distance only 0.50, shuffled labels 0.50). The plain-LOPO row differs from the locus-disjoint row for the
reason in section 47 (locus memorisation); with the donor unit held out the partner leak is removed in the locus-disjoint
and in the plain folds alike (plain folds now also hold out both samples; they still allow shared loci of other donors).

### Verdict
No conclusion of section 47 changes. The headline 8-patient numbers, the increments over the HSPC state (+0.023, +0.015,
+0.006), the gain-of-H3K27ac signal (0.617, 8 of 8) and the pair-type and stripe numbers were never affected: the healthy
donors were not in the data, and the duplicate pairs were never together in an 8- or 10-sample experiment. Only the
18-sample methylation experiments were exposed to the partner leak, and they move by at most 0.003 (A_hspc 0.820 to 0.822, A_do
0.875 to 0.875, B_rand 0.703 to 0.701; plain folds 0.848 to 0.850, 0.905 to 0.903, 0.712 to 0.709). The held-out 168/496 unit scores 0.822
where 168 alone had 0.857 and 496 0.787, so the single-sample numbers were higher than the unit number. Pooled over units the
change is negligible because the locus-disjoint folds already removed most shared-anchor information the partner carried. The 18-sample
experiments now rest on 16 donors. The remaining limit is the "probably" for 260 and 424 (section 54).

Reproduce: `python scripts/aml_loop_classifier.py --stage hic tracks motifs stripes_hic build subtypes model figures --jobs 4
--workers 4 --null-reps-lr 20 --null-reps-gb 10` (hicdiff env).

## 58. Chromosome-held-out comparison at 25 kb, competitor resolutions, and the 5 kb plan (FINAL for 25 kb and 5 kb; section 57 was the last number when this was written)

Question: how do the chromosome-held-out final models of section 37 compare
with each published predictor at 25 kb, which competitor supports which
resolution, and what is needed to repeat the comparison at 5 kb?

Inventory of the competitor predictions on disk (none was rerun; checked from
the cooler headers and file lists on 2026-10-10):

| competitor | native grid and reach | predictions on disk | held out by the competitor | 25 kb | 10 kb | 5 kb |
|---|---|---|---|---|---|---|
| C.Origami | 8192 bp bins, 2 097 152 bp window, hg38, released IMR-90 model | `C.Origami/testNovember/IMR90/chr2.cool`, `chr3.cool` | chr2 (test), chr3 (validation) | sampled from 8192 bp | sampled (section 37) | sampled (up-sampling) |
| Epiphany | 10 kb bins, 990 kb reach, hg38, released GM12878 weights | `epiphany_pretrained/assemble_matrix_chr{2,3,11,17}.cool` | chr3, 11, 17 | sampled from 10 kb | native (section 37) | sampled (up-sampling) |
| HiC-Reg | 5 kb bins, 1 Mb reach, hg19 | published GM12878 chr1 to 22 (`Gm12878_chr<N>_RF_tree20_prediction_predicted_coords.cool`, 5 kb, and a `_25kb.cool` sum); released text predictions for GM12878 and K562, chr14 and chr17: CV, cross-chromosome (WINDOW, MULTICELL), cross-cell (`home_offload/HiC-Reg`) | cross-chromosome: chr14 (model of chr17) and chr17 (model of chr14) | native by summing 5 kb counts | native by summing 5 kb counts | native, not yet converted (`hicreg_audit.variants()` covers 25 and 10 kb) |
| Hi-cGAN | per-resolution Zenodo records: 2 kb (Akita regions), 5 kb (GM12878 only, best_plus_two RAD21, H3K4me3, SMC3), 10 kb and 25 kb (GM12878 whole genome and 25 cross-cell-type sets: the GM12878, HMEC, HUVEC, IMR-90 and K562 models on each cell's inputs) | `zenodo_records/02_best_models`, `04_sweep_5kb`, `05_sweep_10kb`, `06_sweep_25kb` | trained on odd chromosomes, even held out | native | native | native, GM12878 only, no cross-cell set |
| Akita | 2048 bp bins, 448-bin (917 kb) crop of a 1 Mb window, 413 test regions | released predictions in `akita_testset_rasters.npz` (GM12878, HFF) | all 413 regions are excluded from our training | none | none | none |

C.Origami and Epiphany have no native 25 kb output. Their O/E maps are read
at the 25 kb bin centres by the code path of the 10 kb blocks
(`benchmark_blocks.sample_competitor`, bilinear in (bin, bin) space, a pixel
kept when all four neighbours exist), so the 8192 bp and 10 kb values are read
at points and keep structure finer than a 25 kb bin. HiC-Reg's 25 kb maps are
sums of its 5 kb counts. Akita has no whole-chromosome output and no 25 kb
score exists (the 2 kb comparison is section 18, block A). Hi-cGAN has no
IMR-90 prediction in the 25 kb chrho blocks of section 37; the record 06 sets
`pred_imr90_on_imr90` and `pred_gm12878_on_imr90` were converted with
`oe_proper.py` for this section (`BENCH_HICGAN_IMR90_25KB=1`).

Scoring is the section 37 procedure at 25 kb (`scripts/hic_measures.py --stage
chrho_score chrho_report --chrho-25kb-comps`): chr2, 3, 11, 14, 17 absent
from the data of every cell, whole-chromosome tiling, identical pixels within
a block (finite in the target and in every source), band cut to the shortest
reach (C.Origami 2 Mb, Epiphany 39 bins = 975 kb, HiC-Reg 1 Mb, Hi-cGAN 2 Mb).
The HiC-Reg and Hi-cGAN blocks are the 25 kb rows of section 37 (identical
files); the C.Origami blocks (chr2, chr3), the Epiphany blocks (chr3, 11, 17)
and the Hi-cGAN IMR-90 rows (chr2 only: chr3 is one of its training
chromosomes) are new. Mean over the chromosomes of a block; compartment r is
the compartment preference correlation of section 31; the APA ratio equals 1
for the measured map.

| block | source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|---|
| C.Origami, IMR-90 chr2, chr3 | all12refin_drop_chrho | 0.940 | 0.961 | 0.941 | 0.943 | 0.817 | 0.349 | 1.038 | 0.958 | 0.947 |
|  | all12honest_drop_chrho (ChIP only) | 0.804 | 0.864 | 0.800 | 0.775 | 0.647 | 0.206 | 0.541 | 0.862 | 0.795 |
|  | consensus of the other cells | 0.877 | 0.932 | 0.872 | 0.817 | 0.788 | 0.374 | 0.855 | 0.885 | 0.844 |
|  | C.Origami | 0.942 | 0.922 | 0.933 | 0.898 | 0.752 | 0.313 | 2.930 | 0.980 | 0.950 |
| C.Origami block, chr2 only | all12refin_drop_chrho | 0.945 | 0.962 | 0.946 | 0.941 | 0.803 | 0.336 | 1.030 | 0.960 | 0.952 |
|  | all12honest_drop_chrho (ChIP only) | 0.806 | 0.855 | 0.799 | 0.767 | 0.616 | 0.197 | 0.556 | 0.856 | 0.798 |
|  | C.Origami | 0.944 | 0.921 | 0.935 | 0.899 | 0.751 | 0.283 | 2.924 | 0.980 | 0.952 |
|  | Hi-cGAN IMR-90 model | 0.808 | 0.924 | 0.792 | 0.630 | 0.618 | 0.258 | 0.714 | 0.945 | 0.845 |
|  | Hi-cGAN GM12878 model | 0.335 | 0.592 | 0.349 | 0.443 | 0.309 | 0.075 | 0.217 | 0.739 | 0.483 |
| Epiphany, GM12878 chr3, 11, 17 | all12refin_drop_chrho | 0.917 | 0.944 | 0.913 | 0.900 | 0.827 | 0.453 | 0.786 | 0.948 | 0.928 |
|  | all12honest_drop_chrho (ChIP only) | 0.754 | 0.788 | 0.739 | 0.668 | 0.608 | 0.241 | 0.353 | 0.846 | 0.773 |
|  | consensus of the other cells | 0.841 | 0.887 | 0.832 | 0.804 | 0.805 | 0.456 | 0.765 | 0.878 | 0.843 |
|  | Epiphany | 0.518 | 0.672 | 0.501 | 0.377 | 0.437 | 0.254 | 1.513 | 0.775 | 0.640 |
| HiC-Reg, GM12878 chr14, chr17 | all12refin_drop_chrho | 0.931 | 0.950 | 0.925 | 0.905 | 0.833 | 0.396 | 0.792 | 0.953 | 0.935 |
|  | all12honest_drop_chrho (ChIP only) | 0.782 | 0.798 | 0.763 | 0.674 | 0.608 | 0.191 | 0.319 | 0.867 | 0.791 |
|  | consensus of the other cells | 0.868 | 0.901 | 0.853 | 0.804 | 0.805 | 0.396 | 0.818 | 0.894 | 0.857 |
|  | HiC-Reg cross-chromosome WINDOW | 0.579 | 0.620 | 0.538 | 0.447 | 0.322 | 0.032 | -0.002 | 0.846 | 0.668 |
|  | HiC-Reg cross-chromosome MULTICELL | 0.615 | 0.634 | 0.567 | 0.455 | 0.345 | 0.038 | -0.032 | 0.856 | 0.692 |
|  | HiC-Reg CV (within-chromosome) | 0.908 | 0.964 | 0.870 | 0.677 | 0.723 | 0.323 | 0.764 | 0.987 | 0.915 |
| HiC-Reg, K562 chr14, chr17 | all12refin_drop_chrho | 0.897 | 0.928 | 0.889 | 0.857 | 0.738 | 0.347 | 0.843 | 0.927 | 0.904 |
|  | all12honest_drop_chrho (ChIP only) | 0.730 | 0.777 | 0.711 | 0.622 | 0.568 | 0.125 | 0.187 | 0.834 | 0.748 |
|  | consensus of the other cells | 0.857 | 0.905 | 0.853 | 0.839 | 0.769 | 0.333 | 0.876 | 0.887 | 0.868 |
|  | HiC-Reg cross-chromosome WINDOW | 0.736 | 0.820 | 0.696 | 0.484 | 0.523 | 0.038 | 0.292 | 0.927 | 0.797 |
|  | HiC-Reg cross-chromosome MULTICELL | 0.740 | 0.819 | 0.700 | 0.500 | 0.519 | 0.030 | 0.260 | 0.927 | 0.800 |
|  | HiC-Reg CV (within-chromosome) | 0.888 | 0.950 | 0.864 | 0.670 | 0.688 | 0.178 | 0.698 | 0.973 | 0.923 |
| Hi-cGAN, GM12878 chr2, chr14 | all12refin_drop_chrho | 0.934 | 0.949 | 0.929 | 0.901 | 0.808 | 0.405 | 0.810 | 0.959 | 0.933 |
|  | all12honest_drop_chrho (ChIP only) | 0.796 | 0.824 | 0.778 | 0.698 | 0.616 | 0.200 | 0.323 | 0.883 | 0.773 |
|  | consensus of the other cells | 0.860 | 0.899 | 0.848 | 0.784 | 0.777 | 0.389 | 0.841 | 0.897 | 0.835 |
|  | Hi-cGAN GM12878 model | 0.221 | 0.485 | 0.225 | 0.237 | 0.218 | 0.048 | 0.051 | 0.644 | 0.322 |
| Hi-cGAN, K562 chr2, chr14 | all12refin_drop_chrho | 0.904 | 0.935 | 0.896 | 0.843 | 0.727 | 0.356 | 0.851 | 0.941 | 0.891 |
|  | all12honest_drop_chrho (ChIP only) | 0.765 | 0.826 | 0.743 | 0.657 | 0.550 | 0.141 | 0.193 | 0.854 | 0.721 |
|  | consensus of the other cells | 0.853 | 0.909 | 0.849 | 0.816 | 0.718 | 0.350 | 0.840 | 0.895 | 0.850 |
|  | Hi-cGAN GM12878 model | 0.143 | 0.479 | 0.147 | 0.218 | 0.213 | 0.050 | 0.156 | 0.655 | 0.220 |


Replicate ceilings at 25 kb (same blocks): IMR-90 Pearson 0.976, SCC 0.979;
GM12878 0.941 to 0.947; K562 0.936 (`chrho_25kb_competitors.csv`).

Verdict per competitor at 25 kb, ours = all12refin_drop_chrho (reference
map as input) and all12honest_drop_chrho (ChIP only, no consensus); "level"
is a difference within 0.01 of the mean over the block's chromosomes
(`chrho_25kb_vs_competitor.csv`, with per-chromosome win counts):

- C.Origami (IMR-90, chr2 and chr3, resampled from 8192 bp). The reference-map
  model is level on Pearson (0.940 against 0.942; higher on 1 of 2
  chromosomes), SCC (0.941 against 0.933) and HiCRep (0.947 against 0.950),
  ahead on the 1 Mb tile median (0.961 against 0.922), insulation r (0.943
  against 0.898), boundary recall (0.817 against 0.752) and loop recall (0.349
  against 0.313), each on both chromosomes, and behind on compartment r (0.958
  against 0.980, 0 of 2). C.Origami's APA ratio is 2.93, the model's 1.04.
  The ChIP-only model is behind on every measure (Pearson 0.804, SCC 0.800,
  tile median 0.864, insulation 0.775, boundary recall 0.647, loop recall
  0.206). At 10 kb (section 37) the corresponding values are 0.926 against
  0.920 Pearson and 0.162 loop recall for C.Origami.
- Epiphany (GM12878, chr3, 11, 17, resampled from 10 kb). The reference-map
  model is ahead on all measures on all three chromosomes (Pearson 0.917
  against 0.518, SCC 0.913 against 0.501, insulation 0.900 against 0.377,
  loop recall 0.453 against 0.254); the ChIP-only model leads by 0.24
  Pearson (0.754) and 0.24 SCC, with loop recall 0.241 against 0.254. The
  Epiphany APA ratio is 1.51 and the model's 0.79 (ChIP only 0.35).
- Hi-cGAN (GM12878 and K562 chr2, chr14; native 25 kb). Ahead on every
  measure on every chromosome: Pearson 0.934 (ChIP only 0.796) against
  0.221 on GM12878 and 0.904 (0.765) against 0.143 on K562. On IMR-90 chr2,
  Hi-cGAN's IMR-90 model (trained on IMR-90, odd chromosomes) scores 0.808
  Pearson, 0.792 SCC, 0.924 tile median, 0.630 insulation r, 0.618 boundary
  recall, 0.258 loop recall: the reference-map model is ahead on every
  measure (Pearson 0.945, difference 0.136); the ChIP-only model (0.806) is
  level on Pearson and SCC, ahead on insulation (0.767 against 0.630) and
  behind on the tile median (0.855), loop recall (0.197) and compartment r
  (0.856 against 0.945). The GM12878 Hi-cGAN model applied to IMR-90 scores
  0.335.
- HiC-Reg, cross-chromosome (leak-free; native 25 kb sum; chr14, chr17).
  GM12878: the reference-map model is ahead on every measure (Pearson 0.931
  against 0.579 WINDOW and 0.615 MULTICELL) and the ChIP-only model too
  (0.783). K562: reference-map model ahead on every measure (0.897 against
  0.736 and 0.740); the ChIP-only model is level on Pearson (0.730, -0.006
  and -0.010), ahead on SCC (0.711 against 0.696 and 0.700), insulation
  (0.622 against 0.484 and 0.500), boundary recall and loop recall (0.125
  against 0.038 and 0.030), behind on the tile median (0.777 against 0.820
  and 0.819), compartment r (0.834 against 0.927) and HiCRep (0.748 against
  0.797 and 0.800).
- HiC-Reg, published within-chromosome CV maps (not a held-out score). The
  reference-map model is ahead on Pearson on GM12878 (0.931 against 0.908)
  and level on K562 (0.897 against 0.888), ahead on SCC, insulation r,
  boundary and loop recall in both cells, ahead on HiCRep on GM12878 (0.935
  against 0.915) and behind on K562 (0.904 against 0.923), and behind on the
  tile median (0.950 against 0.964; 0.928 against 0.950) and compartment r
  (0.953 against 0.987; 0.927 against 0.973); the ChIP-only model is behind
  on every measure except insulation r on GM12878 (level, 0.674 against
  0.677); Pearson 0.783 and 0.730.
- Akita: no 25 kb score exists.

Limits. (i) The competitors differ in training data and inputs from our
12-cell models; the comparison states what each released prediction scores on
chromosomes it did not train on, not a comparison of architectures
. (ii) Leakage status: C.Origami chr2 (test) and chr3 (validation) are
unseen by C.Origami and by our chrho models, and both saw IMR-90 Hi-C on
other chromosomes (ours: 17 chromosomes of IMR-90 and 11 further cells;
C.Origami: IMR-90 only); Epiphany chr3, 11, 17 unseen by both; HiC-Reg
cross-chromosome files are leak-free (model trained on the other of chr14 and
chr17, both absent from our data), HiC-Reg CV files are within-chromosome
interpolation (section 18); Hi-cGAN GM12878 and K562 on the even chromosomes
chr2 and chr14 (K562 was never a Hi-cGAN training cell; IMR-90 chr2 for the
IMR-90 model). The reference-map model reads the leave-one-out consensus of
the other cells' measured maps on the held-out chromosomes as input (its
intended use; the ChIP-only model reads none). (iii) The resampled
C.Origami and Epiphany maps are read at points on the 25 kb grid, and the
insulation, boundary and loop measures are defined in bins of the grid; the
competitor scores differ between 10 and 25 kb (C.Origami: loop recall 0.313
at 25 kb, 0.162 at 10 kb; boundary recall 0.752 and 0.474), so the margins
are specific to the grid. (iv) Chromosome counts are small (2 to 3 per
block); the win counts in `chrho_25kb_vs_competitor.csv` are per chromosome,
not per region. (v) The C.Origami APA ratio of 2.9 is an over-enrichment of
loop signal against the measured map, which the loop recall does not show.
(vi) The consensus of the other cells has a higher boundary recall than the
reference-map model in the K562 HiC-Reg block (0.769 against 0.738). The
refres model is in the csv files and equals refin within 0.01 on Pearson in
every block.

5 kb (FINAL). The three chromosome-held-out 5 kb models (`all12refin_drop_chrho_5kb`,
`all12honest_drop_chrho_5kb` (ChIP only, no consensus), `all12refres_drop_chrho_5kb`;
chr2, 3, 11, 14, 17 absent from the data of every cell) were trained on the
remote 4090 and predicted on the five chromosomes on a local GPU (inference only).
Scoring is the 25 kb procedure on the 5 kb grid (`scripts/hic_measures.py --stage
chrho_score chrho_report --chrho-res 5000`; 11 blocks), identical pixels within a
block, same metric set.

5 kb parameters (all behind `--chrho-res 5000`; the 25 and 10 kb code paths are
unchanged): model window 384 bins (1.92 Mb), insulation square 10 bins (50 kb,
as 5 bins at 10 kb), loop and APA windows of the 10 kb row doubled in bins
(donut 2 to 6, loop distance 6 to 200 bins, APA half-width 6, corner 2, NMS 2,
matching tolerance 2 bins = 10 kb; `benchmark_blocks.params(5000)`), boundary
matching tolerance 2 bins (10 kb, as the 10 kb row), hicFindTADs depths 3 to 10
bins as at every resolution, HiCRep h = 5 bins. The reference loops are the 10 kb
hicDetectLoops calls of the measured map. Bands: C.Origami 383 bins (1.9 Mb),
Epiphany 198 bins (990 kb), HiC-Reg 200 bins (1 Mb), Hi-cGAN 383 bins (1.9 Mb).
The count-space Pearson AUC uses min(1 Mb / res, band) distances (the Epiphany band
is 199 diagonals), a no-op for the existing blocks. Expected-count curves of a band
wider than 200 diagonals get their own file (`_D383`). Regression check: the 10 kb
Hi-cGAN GM12878 chr14 block re-scored with the edited code into a scratch folder
equals the committed `measures.csv` in all 48 columns (exact) except the HiCRep
value of all12refres_drop_chrho, which is empty in the committed file (that
tool run failed there) and 0.937 in the re-run.

Competitor conversions (nothing re-run; saved predictions only):
- Hi-cGAN: the GM12878 5 kb record (`02_best_models/.../5kb/best_plus_two`,
  RAD21, H3K4me3, SMC3; trained on odd chromosomes) through `oe_proper.py`
  (chr2, chr14; 2 Mb distance cap). No K562 or IMR-90 5 kb prediction exists
  (cross-cell sets exist at 10 and 25 kb only), so there is no K562 Hi-cGAN
  block at 5 kb.
- HiC-Reg: released 5 kb predictions of GM12878 and K562, chr14 and chr17,
  WINDOW and MULTICELL, cross-chromosome and CV, native 5 kb (no summing), through
  `oe_proper.py` (`hicreg_audit.variants()` with `BENCH_5KB=1`).
- C.Origami (8192 bp, IMR-90 chr2, chr3) and Epiphany (10 kb, GM12878 chr3, 11,
  17): no native 5 kb output. Their O/E maps (the native-resolution files already
  made for the 10 and 25 kb blocks) are read at the 5 kb bin centres with the
  same convention as at 25 kb (bilinear in (bin, bin) space,
  `benchmark_blocks.sample_competitor`, a pixel kept when all four neighbours
  exist), an up-sampling: the 5 kb scores of these two carry no information
  finer than 8192 bp and 10 kb.

Mean over the chromosomes of a block (`chrho_5kb_table.md`, csv files
`chrho_5kb_competitors.csv`, `chrho_5kb_vs_competitor.csv`):

| block | source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|---|
| C.Origami: IMR-90, chr2, chr3 | all12refin_drop_chrho | 0.882 | 0.904 | 0.884 | 0.936 | 0.547 | 0.302 | 0.806 | 0.964 | 0.909 |
| C.Origami: IMR-90, chr2, chr3 | all12honest_drop_chrho | 0.756 | 0.788 | 0.760 | 0.834 | 0.462 | 0.177 | 0.558 | 0.893 | 0.769 |
| C.Origami: IMR-90, chr2, chr3 | all12refres_drop_chrho | 0.878 | 0.899 | 0.880 | 0.928 | 0.555 | 0.319 | 0.954 | 0.961 | 0.906 |
| C.Origami: IMR-90, chr2, chr3 | consensus | 0.819 | 0.853 | 0.815 | 0.803 | 0.581 | 0.339 | 0.964 | 0.902 | 0.815 |
| C.Origami: IMR-90, chr2, chr3 | corigami_imr90_8kb | 0.849 | 0.846 | 0.854 | 0.852 | 0.348 | 0.164 | 0.466 | 0.956 | 0.872 |
| Epiphany: GM12878, chr3, 11, 17 | all12refin_drop_chrho | 0.894 | 0.908 | 0.891 | 0.889 | 0.577 | 0.430 | 0.653 | 0.953 | 0.919 |
| Epiphany: GM12878, chr3, 11, 17 | all12honest_drop_chrho | 0.737 | 0.752 | 0.734 | 0.726 | 0.472 | 0.222 | 0.403 | 0.844 | 0.762 |
| Epiphany: GM12878, chr3, 11, 17 | all12refres_drop_chrho | 0.887 | 0.896 | 0.881 | 0.864 | 0.566 | 0.424 | 0.762 | 0.947 | 0.912 |
| Epiphany: GM12878, chr3, 11, 17 | consensus | 0.793 | 0.820 | 0.790 | 0.784 | 0.560 | 0.416 | 0.936 | 0.876 | 0.831 |
| Epiphany: GM12878, chr3, 11, 17 | epiphany_gm12878_10kb | 0.523 | 0.612 | 0.504 | 0.310 | 0.133 | 0.043 | 0.142 | 0.770 | 0.634 |
| Hi-cGAN: GM12878, chr2, chr14 | all12refin_drop_chrho | 0.882 | 0.902 | 0.882 | 0.886 | 0.540 | 0.337 | 0.668 | 0.954 | 0.918 |
| Hi-cGAN: GM12878, chr2, chr14 | all12honest_drop_chrho | 0.734 | 0.762 | 0.740 | 0.737 | 0.459 | 0.161 | 0.418 | 0.881 | 0.768 |
| Hi-cGAN: GM12878, chr2, chr14 | all12refres_drop_chrho | 0.873 | 0.893 | 0.873 | 0.851 | 0.547 | 0.318 | 0.751 | 0.953 | 0.907 |
| Hi-cGAN: GM12878, chr2, chr14 | consensus | 0.791 | 0.819 | 0.792 | 0.759 | 0.538 | 0.319 | 0.930 | 0.882 | 0.831 |
| Hi-cGAN: GM12878, chr2, chr14 | hicgan_gm12878_5kb | 0.346 | 0.448 | 0.353 | 0.397 | 0.124 | 0.068 | 0.235 | 0.638 | 0.440 |
| HiC-Reg: GM12878, chr14, chr17 | all12refin_drop_chrho | 0.906 | 0.911 | 0.902 | 0.898 | 0.562 | 0.363 | 0.675 | 0.957 | 0.923 |
| HiC-Reg: GM12878, chr14, chr17 | all12honest_drop_chrho | 0.749 | 0.758 | 0.749 | 0.741 | 0.455 | 0.188 | 0.385 | 0.871 | 0.770 |
| HiC-Reg: GM12878, chr14, chr17 | all12refres_drop_chrho | 0.899 | 0.899 | 0.894 | 0.873 | 0.568 | 0.349 | 0.772 | 0.954 | 0.919 |
| HiC-Reg: GM12878, chr14, chr17 | consensus | 0.821 | 0.842 | 0.817 | 0.798 | 0.556 | 0.354 | 0.946 | 0.885 | 0.853 |
| HiC-Reg: GM12878, chr14, chr17 | hicregmulticellcrosschrom_gm12878_5kb | 0.626 | 0.519 | 0.571 | 0.525 | 0.195 | 0.000 | 0.035 | 0.890 | 0.611 |
| HiC-Reg: GM12878, chr14, chr17 | hicregwindowcrosschrom_gm12878_5kb | 0.603 | 0.520 | 0.554 | 0.515 | 0.161 | 0.000 | 0.045 | 0.891 | 0.593 |
| HiC-Reg: GM12878, chr14, chr17 | hicregwindowcv_gm12878_5kb | 0.893 | 0.898 | 0.859 | 0.647 | 0.252 | 0.020 | 0.398 | 0.992 | 0.922 |
| HiC-Reg: K562, chr14, chr17 | all12refin_drop_chrho | 0.771 | 0.817 | 0.776 | 0.842 | 0.510 | 0.338 | 0.718 | 0.925 | 0.814 |
| HiC-Reg: K562, chr14, chr17 | all12honest_drop_chrho | 0.641 | 0.641 | 0.653 | 0.730 | 0.415 | 0.169 | 0.283 | 0.831 | 0.664 |
| HiC-Reg: K562, chr14, chr17 | all12refres_drop_chrho | 0.784 | 0.812 | 0.789 | 0.869 | 0.509 | 0.338 | 0.689 | 0.922 | 0.829 |
| HiC-Reg: K562, chr14, chr17 | consensus | 0.673 | 0.777 | 0.685 | 0.810 | 0.505 | 0.323 | 0.818 | 0.875 | 0.727 |
| HiC-Reg: K562, chr14, chr17 | hicregmulticellcrosschrom_k562_5kb | 0.783 | 0.716 | 0.768 | 0.640 | 0.271 | 0.003 | 0.242 | 0.962 | 0.810 |
| HiC-Reg: K562, chr14, chr17 | hicregwindowcrosschrom_k562_5kb | 0.774 | 0.703 | 0.755 | 0.599 | 0.226 | 0.003 | 0.261 | 0.959 | 0.790 |
| HiC-Reg: K562, chr14, chr17 | hicregwindowcv_k562_5kb | 0.845 | 0.814 | 0.840 | 0.728 | 0.282 | 0.005 | 0.441 | 0.978 | 0.899 |

Replicate ceilings at 5 kb are low (IMR-90 Pearson 0.670, GM12878 0.69, K562
0.706; the single-lane replicate maps are sparse at 5 kb) and are not
ceilings of what a model can reach against the deep target.

Verdict per competitor at 5 kb (ours = all12refin_drop_chrho, ChIP-only =
all12honest_drop_chrho; "level" within 0.01; win counts per chromosome in
`chrho_5kb_vs_competitor.csv`):

- C.Origami (IMR-90, chr2, chr3, up-sampled from 8192 bp). The reference-map
  model is ahead on Pearson (0.882 against 0.849), tile median (0.904 against
  0.846), SCC (0.884 against 0.854), insulation r (0.936 against 0.852),
  boundary recall (0.547 against 0.348), loop recall (0.302 against 0.164) and
  HiCRep (0.909 against 0.872), on both chromosomes each, and level on
  compartment r (0.964 against 0.956). The ChIP-only model is behind on
  Pearson (0.756), tile median, SCC, insulation r, compartment r and HiCRep,
  ahead on boundary recall (0.462) and, on 1 of 2 chromosomes, loop recall
  (0.177 against 0.164). At 25 kb the reference-map model was level on
  Pearson against C.Origami (0.940 against 0.942); at 5 kb it is ahead (0.882
  against 0.849), C.Origami's score falling by 0.09 from 25 to 5 kb and ours by 0.06.
- Epiphany (GM12878, chr3, 11, 17, up-sampled from 10 kb). Ahead on every
  measure on all three chromosomes (Pearson 0.894 against 0.523, SCC 0.891
  against 0.504, insulation 0.889 against 0.310, loop recall 0.430 against
  0.043); the ChIP-only model is ahead on all measures too (Pearson 0.737).
- Hi-cGAN (GM12878, chr2, chr14; native 5 kb). Ahead on every measure on both
  chromosomes: Pearson 0.882 (ChIP only 0.734) against 0.346, insulation r
  0.886 against 0.397, loop recall 0.337 against 0.068.
- HiC-Reg cross-chromosome (leak-free; native 5 kb). GM12878: the
  reference-map model is ahead on every measure (Pearson 0.906 against 0.603
  WINDOW and 0.626 MULTICELL); the ChIP-only model too, except compartment r
  (0.871 against 0.891). K562: Pearson 0.771 against 0.774 WINDOW (level) and
  0.783 MULTICELL (behind by 0.012); ahead on tile median (0.817 against 0.703
  and 0.716), insulation, boundary and loop recall, and behind on compartment
  r (0.925 against 0.959 and 0.962); SCC and HiCRep are level or ahead on
  1 of 2 chromosomes. The ChIP-only model is behind on Pearson (0.641), SCC,
  tile median, compartment r and HiCRep, ahead on boundary and loop recall.
- HiC-Reg within-chromosome CV maps (not a held-out score). GM12878: ahead of
  CV on Pearson (0.906 against 0.893), SCC, tile median, insulation r,
  boundary and loop recall, level on HiCRep, behind on compartment r (0.957
  against 0.992). K562: behind on Pearson (0.771 against 0.845), SCC and HiCRep,
  level on tile median, ahead on insulation r (0.842 against 0.728), boundary
  and loop recall, behind on compartment r (0.925 against 0.978).
- The refres model is within 0.01 of refin on Pearson in 4 of 5 blocks (K562
  HiC-Reg block: 0.784 against 0.771).

Limits (5 kb). (i) C.Origami and Epiphany have no 5 kb output: the maps are
up-sampled, the competitors' 5 kb insulation, boundary and loop scores reflect
a resolution they never produced, and their margins to our native 5 kb output
are partly a resolution effect. (ii) Few chromosomes per block (2 to 3), 11
chromosome-blocks in total; win counts are per chromosome. (iii) Loop recall,
APA and compartment r are noisier at 5 kb: the 5 kb maps are sparser, the
reference loops are the 10 kb calls (a 5 kb call set does not exist), the loop
matching tolerance is 10 kb in bp and the C.Origami APA ratio (0.47) and
the HiC-Reg APA ratios (0.04 to 0.44) are far from 1 while the reference-map
model gives 0.65 to 0.81. (iv) Leakage status is that of the 25 kb section:
C.Origami chr2 and chr3, Epiphany chr3, 11, 17, HiC-Reg cross-chromosome
chr14 and chr17 are unseen by the competitor and by our models; HiC-Reg CV is
within-chromosome interpolation; Hi-cGAN chr2, chr14 are even chromosomes (held
out by Hi-cGAN). Hi-cGAN has no 5 kb K562 output. (v) The reference-map
model reads the leave-one-out consensus of the other cells' 5 kb maps as
input; the ChIP-only model reads none. (vi) Competitors differ in training
data and inputs from the 12-cell models (not an architecture comparison).
(vii) The "replicate" and "best_other_map" rows are weak baselines at 5 kb
(sparse single-lane maps); the consensus of the other cells has a higher boundary
and loop recall than the reference-map model in the C.Origami block (0.581 and
0.339 against 0.547 and 0.302).

Prediction used a local GPU for inference only; the GPU recipe and the
estimates of `results/benchmark/REMOTE_JOB_chrho_5kb.md` are unchanged.

Reproduce (hicdiff env, `BENCH_5KB=1` registers the 5 kb competitors):

    BENCH_5KB=1 python scripts/benchmark_competitors.py --stage oe --names hicgan_gm12878_5kb
    BENCH_5KB=1 python scripts/hicreg_audit.py --stage cools
    python scripts/hic_measures.py --stage chrho_predict --chrho-res 5000 --gpu 1
    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-res 5000 --jobs 2 --threads 4
    python scripts/chrho_5kb_table.py

Files: `results/benchmark/chrho_25kb_competitors.csv` (every source per block,
every measure), `chrho_25kb_vs_competitor.csv` (differences, win counts,
verdicts), `chrho_25kb_table.md`; `results/hic_measures_chrho/chrho_25kb_summary.csv`
and `chrho_25kb_per_block.csv` (the scorer's output, 22 blocks).
Reproduce (hicdiff env):

    BENCH_HICGAN_IMR90_25KB=1 python scripts/benchmark_competitors.py --stage oe --names hicgan_imr90_on_imr90_25kb hicgan_gm12878_on_imr90_25kb
    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-25kb-comps --jobs 1 --threads 4
    python scripts/chrho_25kb_table.py


## 59. Why the sequence + methylation CNN fails on K562 and what fixes it (FINAL; section 58 was the last number when this was written)

Question. In section 42 the sequence + methylation CNN (`seq_meth`) beats gradient
boosting on the seven training cells but fails on K562 and HMEC, which are in no
training set (unseen_cell R2 -0.2 to -24 for most marks; K562 median Pearson 0.34
for `seq_meth` and -0.20 for `meth`; HMEC 0.74 and 0.67), while sequence alone
generalises (R2 0.12 to 0.53) and the gradient-boosting analysis of section 41
reaches R2 up to 0.49. Which input property causes this and does a change of the
Meth input fix it.

Setup. Section 42 model, split, targets and budget unchanged (9.3 M parameters,
20000 steps, batch 32, one seed, one RTX 4090, about 33 minutes of training and 4
minutes of evaluation per variant). Test: chr6, 8, ..., 22 of the nine cells.
Independent cells on unseen_cell: 2 (K562, HMEC); on seen_cell: 7. Median
"unseen" values below are the median of the two cells (for RAD21 and SMC3 only
K562 has the track, so that median is one cell). Gradient boosting (`GB_both`) is
the `both` set of section 41 (extended sequence features + methylation features,
`partition_fits.csv`, unseen_cell), scored on all even chromosomes (including
chr2 and chr4, 500,000 training rows); `ceiling_loo` is the across-cell mean of
the seven training cells scored against the cell (a cell-agnostic prediction, the
best `seq` can reach).

### Diagnosis (CPU for the distributions, GPU only to dump predictions of the saved section 42 checkpoints)

Meth input (value as the script uses it: methylated fraction of 1 kb bins inside
valid 10 kb bins, all chromosomes):

| cell | mean | sd | q05 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|---|
| gm12878 | 0.446 | 0.299 | 0.01 | 0.20 | 0.41 | 0.68 | 0.96 |
| imr90 | 0.691 | 0.216 | 0.25 | 0.65 | 0.66 | 0.86 | 1.00 |
| h1 | 0.855 | 0.129 | 0.65 | 0.83 | 0.89 | 0.92 | 0.96 |
| hepg2 | 0.397 | 0.341 | 0.00 | 0.08 | 0.32 | 0.70 | 0.97 |
| a549 | 0.512 | 0.311 | 0.02 | 0.23 | 0.56 | 0.81 | 0.93 |
| gm23248 | 0.600 | 0.284 | 0.10 | 0.36 | 0.63 | 0.86 | 0.97 |
| cd14mono | 0.833 | 0.177 | 0.53 | 0.73 | 0.88 | 0.96 | 1.00 |
| k562 (unseen) | 0.219 | 0.307 | 0.00 | 0.00 | 0.03 | 0.35 | 0.89 |
| hmec (unseen) | 0.725 | 0.221 | 0.27 | 0.61 | 0.77 | 0.90 | 0.98 |

The training cells span a genome-wide mean methylation of 0.40 (HepG2) to 0.86
(H1). K562 has 0.22 and half of its bins below 0.03, outside the training range
and in the corner of the input space the model never saw (the constant mean
channel is 0.22 against a training minimum of 0.40). HMEC (0.73) lies inside the
range. The z-scored targets (training-cell statistics, valid bins, all
chromosomes) have a per-cell, per-mark mean within +-0.55 of 0 and a standard
deviation of 0.5 to 1.6 in all cells including K562 (e.g. K562 H3K27ac +0.20 /
0.67, H3K4me3 +0.09 / 0.86, H3K9ac -0.05 / 0.98); the only larger offsets are
seen cells (IMR-90 H3K4me3 -0.89). The target scale of K562 and HMEC is thus
not unusual.

Predictions against targets on the test chromosomes (checkpoints of section 42;
median over marks; `results/seq_cnn/v2/diag_prediction_bias.csv`, columns y_mean,
p_mean, slope of y on the prediction, R2 with the offset removed):

| model, cells | abs. offset p_mean - y_mean | sd of prediction (sd of target 0.9) | slope | R2 | R2 after removing the offset | Pearson |
|---|---|---|---|---|---|---|
| seq_meth, 7 seen | 0.01 | 0.74 | 0.99 | 0.68 | 0.68 | 0.83 |
| seq_meth, HMEC | 0.27 | 0.81 | 0.95 | 0.41 | 0.55 | 0.74 |
| seq_meth, K562 | 1.68 | 0.90 | 0.25 | -3.93 | -1.64 | 0.34 |
| meth, 7 seen | 0.02 | 0.70 | 0.97 | 0.58 | 0.59 | 0.77 |
| meth, HMEC | 0.29 | 0.85 | 0.70 | 0.27 | 0.40 | 0.67 |
| meth, K562 | 5.42 | 2.85 | -0.06 | -60.65 | -16.73 | -0.20 |

Per mark in K562 `seq_meth` predicts offsets up to +5.6 z (H3K4me2, H3K4me3,
H3K9ac) with prediction sd up to 2.3; `meth` predicts +12 z for H3K4me2. Removing
the offset still leaves R2 below zero, so K562 has both a level error and a
ranking error (Pearson -0.2 to 0.6 across marks), and the prediction slope of 0.25
means the predictions are not a rescaled copy of the truth.

Verdict. Hypothesis 1 (Meth input of the unseen cell outside the training
distribution) is supported for K562: the failure grows with the distance of the
input from the training range (HMEC inside the range works, Pearson 0.74; K562
outside fails; the methylation-only control, which has no sequence to fall back
on, explodes to offsets of 5 to 12 z). Hypothesis 3 (target scale per cell) is not
supported: the target means and sds of K562 and HMEC are ordinary; the offset is
produced by the model, so `--target-norm cell` was implemented but not trained.
Hypothesis 2 (memorisation of training cells by their methylation fingerprint) is
weak: removing the cell-mean channel and the per-cell scale (`cellnorm`) leaves
the seen-cell results unchanged (Pearson 0.82 against 0.83) and does not by itself
fix K562.

### Fix: flags of `scripts/seq_cnn_chip.py` (defaults keep section 42 reproducible; variant names carry a suffix)

- `--meth-norm cell`: the Meth channel is z-scored with the cell's own mean and sd
  over its valid bins (statistics from the cell's Meth track only, no ChIP, no
  label); the constant mean-methylation channel is set to 0. `--meth-norm rank`
  does the same by quantile normalisation to a standard normal (implemented,
  smoke-tested, not trained). Variant `*_cellnorm`.
- `--meth-aug`: training only, random per-window scale U(0.5, 1.5) and shift
  N(0, 0.5) of the two Meth channels (`--meth-aug-scale`, `--meth-aug-shift`).
  Variant `*_aug`.
- `--meth-dropout P`: training only, a whole window gets zeros in both Meth channels
  with probability P, so the sequence path stays usable. Variant `*_mdropP`.
- `--target-norm cell`: per-cell, per-mark z-scoring of the targets; test metrics are
  computed after standardising prediction and target per mark and cell, so R2 is
  then scale free and Pearson and Spearman stay the main metrics. Variant `*_tcell`,
  ceilings `ceiling_*_tcell`. Smoke-tested only.
- `stage_summarize` keeps every variant (old rows unchanged); the partition table
  still uses `seq`, `seq_meth`, `meth`. CPU smoke tests of all new options passed
  (`--smoke`, chr21 train, chr22 test).

Commands (remote GPU; `OUT_ROOT` = the section 42 run folder with its cache):

    P="python scripts/seq_cnn_chip.py --config configs/seq_cnn.yaml --stage train eval"
    $P --mode seq_meth --meth-norm cell
    $P --mode meth     --meth-norm cell
    $P --mode seq_meth --meth-aug
    $P --mode seq_meth --meth-norm cell --meth-aug --meth-dropout 0.2
    python scripts/seq_cnn_chip.py --config configs/seq_cnn.yaml --stage summarize --results-dir results/seq_cnn/v2

Wall time on the RTX 4090: 4 variants x (33 min training + 4 min evaluation) = 2 h 12 min
(08:22 to 10:34 on 2026-10-10), plus 5 min per prediction dump.

### Results

Summary (median over marks of the per-mark medians over cells; full per-mark
values below and in `results/seq_cnn/v2/comparison_by_mark.csv`):

| cells | metric | seq | meth | seq_meth | meth_cellnorm | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|---|---|---|
| unseen (K562, HMEC) | R2 | 0.24 | -31.41 | -3.69 | -0.64 | -0.70 | 0.34 | -0.24 | 0.12 | 0.61 |
| unseen | Pearson | 0.57 | 0.20 | 0.43 | 0.51 | 0.58 | 0.73 | 0.70 | 0.64 | 0.82 |
| unseen | Spearman | 0.50 | 0.11 | 0.31 | 0.47 | 0.41 | 0.60 | 0.49 | 0.51 | 0.64 |
| K562 | R2 | 0.40 | -60.65 | -3.93 | -0.90 | -1.45 | 0.41 | -0.53 | 0.16 | 0.62 |
| K562 | Pearson | 0.66 | -0.20 | 0.34 | 0.43 | 0.41 | 0.73 | 0.62 | 0.60 | 0.81 |
| K562 | Spearman | 0.54 | -0.27 | 0.00 | 0.43 | 0.35 | 0.63 | 0.53 | 0.47 | 0.64 |
| HMEC | R2 | 0.23 | 0.27 | 0.41 | 0.45 | 0.53 | 0.52 | 0.51 | 0.36 | 0.59 |
| HMEC | Pearson | 0.58 | 0.67 | 0.74 | 0.69 | 0.75 | 0.72 | 0.74 | 0.64 | 0.79 |
| HMEC | Spearman | 0.44 | 0.61 | 0.63 | 0.57 | 0.61 | 0.64 | 0.64 | 0.51 | 0.67 |
| seen (7 cells) | R2 | 0.35 | 0.60 | 0.69 | 0.57 | 0.67 | 0.64 | 0.62 | 0.46 | 0.60 |
| seen | Pearson | 0.63 | 0.78 | 0.83 | 0.76 | 0.82 | 0.81 | 0.79 | 0.69 | 0.82 |
| seen | Spearman | 0.57 | 0.69 | 0.75 | 0.67 | 0.74 | 0.72 | 0.72 | 0.61 | 0.69 |

Prediction offset and slope after the fix (same diagnostic, median over marks):

| model | cells | abs. offset | slope | R2 | R2 after removing the offset | Pearson |
|---|---|---|---|---|---|---|
| seq_meth_aug | K562 | 0.20 | 0.98 | 0.41 | 0.47 | 0.73 |
| seq_meth_aug | HMEC | 0.05 | 0.93 | 0.52 | 0.52 | 0.72 |
| seq_meth_cellnorm | K562 | 0.88 | 0.52 | -1.45 | 0.03 | 0.41 |
| seq_meth_cellnorm | HMEC | 0.09 | 0.84 | 0.53 | 0.54 | 0.75 |
| meth_cellnorm | K562 | 0.74 | 0.60 | -0.90 | 0.09 | 0.43 |
| seq_meth_cellnorm_aug_mdrop0.2 | K562 | 0.60 | 0.51 | -0.53 | 0.02 | 0.62 |

Per-mark tables (median over cells; blank = mark absent in the cell set):

**unseen_median2, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.19 | -0.19 | -0.97 | 0.28 | -0.24 | 0.12 | 0.68 |
| H3K27ac | 0.23 | -7.07 | -0.14 | 0.36 | 0.07 | 0.15 | 0.39 |
| H3K27me3 | 0.24 | -1.75 | -1.07 | 0.39 | -0.29 | 0.28 | 0.47 |
| H3K36me3 | 0.38 | 0.16 | 0.36 | 0.60 | 0.57 | 0.54 | 0.69 |
| H3K4me1 | 0.21 | -5.19 | -0.70 | 0.09 | -0.30 | -0.33 | 0.38 |
| H3K4me2 | 0.53 | -18.02 | 0.52 | 0.74 | 0.67 | 0.59 | 0.69 |
| H3K4me3 | 0.24 | -23.92 | -0.56 | 0.22 | -0.04 | -0.26 | 0.38 |
| H3K79me2 | 0.43 | -0.67 | 0.53 | 0.58 | 0.55 | 0.39 | 0.73 |
| H3K9ac | 0.45 | -15.80 | 0.34 | 0.69 | 0.58 | 0.48 | 0.64 |
| H3K9me3 | 0.12 | -8.96 | -1.95 | 0.02 | -0.82 | -0.34 | 0.28 |
| H4K20me1 | 0.13 | -0.46 | -0.97 | 0.07 | -0.38 | -0.09 | 0.32 |
| RAD21 | 0.30 | -1.38 | -3.89 | 0.15 | -1.58 | -0.44 | 0.61 |
| SMC3 | 0.44 | -3.69 | -2.94 | 0.34 | -1.02 | -0.04 | 0.62 |
| median over marks | 0.24 | -3.69 | -0.70 | 0.34 | -0.24 | 0.12 | 0.61 |

**unseen_median2, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.46 | 0.44 | 0.40 | 0.56 | 0.48 | 0.48 | 0.83 |
| H3K27ac | 0.56 | 0.37 | 0.73 | 0.77 | 0.75 | 0.67 | 0.71 |
| H3K27me3 | 0.53 | 0.24 | 0.49 | 0.66 | 0.49 | 0.57 | 0.71 |
| H3K36me3 | 0.62 | 0.68 | 0.74 | 0.80 | 0.79 | 0.78 | 0.84 |
| H3K4me1 | 0.57 | 0.70 | 0.71 | 0.73 | 0.73 | 0.64 | 0.74 |
| H3K4me2 | 0.74 | 0.44 | 0.79 | 0.87 | 0.86 | 0.79 | 0.83 |
| H3K4me3 | 0.72 | 0.40 | 0.58 | 0.80 | 0.70 | 0.65 | 0.84 |
| H3K79me2 | 0.68 | 0.60 | 0.77 | 0.80 | 0.80 | 0.66 | 0.87 |
| H3K9ac | 0.70 | 0.35 | 0.78 | 0.85 | 0.83 | 0.75 | 0.82 |
| H3K9me3 | 0.43 | 0.33 | 0.35 | 0.51 | 0.41 | 0.37 | 0.66 |
| H4K20me1 | 0.56 | 0.43 | 0.41 | 0.58 | 0.46 | 0.38 | 0.70 |
| RAD21 | 0.57 | 0.43 | 0.35 | 0.64 | 0.49 | 0.49 | 0.82 |
| SMC3 | 0.67 | 0.34 | 0.29 | 0.66 | 0.45 | 0.44 | 0.79 |
| median over marks | 0.57 | 0.43 | 0.58 | 0.73 | 0.70 | 0.64 | 0.82 |

**unseen_median2, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.28 | 0.31 | 0.27 | 0.42 | 0.36 | 0.30 | 0.56 |
| H3K27ac | 0.48 | 0.22 | 0.67 | 0.69 | 0.68 | 0.55 | 0.64 |
| H3K27me3 | 0.50 | 0.19 | 0.41 | 0.60 | 0.39 | 0.51 | 0.69 |
| H3K36me3 | 0.49 | 0.51 | 0.68 | 0.66 | 0.69 | 0.59 | 0.68 |
| H3K4me1 | 0.56 | 0.65 | 0.68 | 0.64 | 0.68 | 0.63 | 0.72 |
| H3K4me2 | 0.54 | 0.34 | 0.59 | 0.67 | 0.61 | 0.52 | 0.65 |
| H3K4me3 | 0.37 | 0.23 | 0.31 | 0.50 | 0.43 | 0.20 | 0.52 |
| H3K79me2 | 0.65 | 0.40 | 0.70 | 0.69 | 0.69 | 0.54 | 0.73 |
| H3K9ac | 0.54 | 0.16 | 0.62 | 0.68 | 0.66 | 0.55 | 0.66 |
| H3K9me3 | 0.44 | 0.30 | 0.30 | 0.45 | 0.38 | 0.40 | 0.64 |
| H4K20me1 | 0.51 | 0.35 | 0.30 | 0.47 | 0.35 | 0.25 | 0.63 |
| RAD21 | 0.46 | 0.35 | 0.35 | 0.53 | 0.49 | 0.43 | 0.64 |
| SMC3 | 0.63 | 0.23 | 0.24 | 0.57 | 0.44 | 0.34 | 0.63 |
| median over marks | 0.50 | 0.31 | 0.41 | 0.60 | 0.49 | 0.51 | 0.64 |

**seen_median7, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.28 | 0.44 | 0.43 | 0.43 | 0.43 | 0.38 | 0.67 |
| H3K27ac | 0.28 | 0.69 | 0.67 | 0.66 | 0.61 | 0.47 | 0.37 |
| H3K27me3 | 0.35 | 0.65 | 0.65 | 0.62 | 0.62 | 0.46 | 0.45 |
| H3K36me3 | 0.39 | 0.73 | 0.73 | 0.68 | 0.68 | 0.55 | 0.60 |
| H3K4me1 | 0.25 | 0.70 | 0.69 | 0.66 | 0.66 | 0.45 | 0.37 |
| H3K4me2 | 0.48 | 0.82 | 0.80 | 0.79 | 0.78 | 0.61 | 0.60 |
| H3K4me3 | 0.57 | 0.87 | 0.84 | 0.82 | 0.83 | 0.62 | 0.68 |
| H3K79me2 | 0.43 | 0.70 | 0.69 | 0.64 | 0.65 | 0.50 | 0.68 |
| H3K9ac | 0.51 | 0.82 | 0.78 | 0.77 | 0.76 | 0.67 | 0.65 |
| H3K9me3 | 0.19 | 0.52 | 0.50 | 0.45 | 0.45 | 0.29 | 0.39 |
| H4K20me1 | 0.34 | 0.61 | 0.58 | 0.53 | 0.53 | 0.35 | 0.54 |
| RAD21 | 0.34 | 0.53 | 0.53 | 0.53 | 0.52 | 0.42 | 0.72 |
| SMC3 | 0.39 | 0.62 | 0.62 | 0.57 | 0.57 | 0.39 | 0.58 |
| median over marks | 0.35 | 0.69 | 0.67 | 0.64 | 0.62 | 0.46 | 0.60 |

**seen_median7, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.54 | 0.67 | 0.66 | 0.67 | 0.66 | 0.64 | 0.86 |
| H3K27ac | 0.59 | 0.83 | 0.82 | 0.81 | 0.78 | 0.69 | 0.70 |
| H3K27me3 | 0.60 | 0.81 | 0.80 | 0.79 | 0.79 | 0.71 | 0.72 |
| H3K36me3 | 0.69 | 0.86 | 0.85 | 0.83 | 0.83 | 0.76 | 0.83 |
| H3K4me1 | 0.56 | 0.84 | 0.83 | 0.82 | 0.81 | 0.69 | 0.66 |
| H3K4me2 | 0.70 | 0.90 | 0.90 | 0.89 | 0.89 | 0.79 | 0.80 |
| H3K4me3 | 0.80 | 0.93 | 0.92 | 0.91 | 0.91 | 0.80 | 0.87 |
| H3K79me2 | 0.66 | 0.84 | 0.84 | 0.81 | 0.81 | 0.71 | 0.85 |
| H3K9ac | 0.74 | 0.90 | 0.89 | 0.88 | 0.88 | 0.82 | 0.85 |
| H3K9me3 | 0.50 | 0.72 | 0.71 | 0.68 | 0.68 | 0.56 | 0.66 |
| H4K20me1 | 0.64 | 0.78 | 0.77 | 0.74 | 0.74 | 0.60 | 0.77 |
| RAD21 | 0.60 | 0.73 | 0.73 | 0.73 | 0.72 | 0.67 | 0.86 |
| SMC3 | 0.63 | 0.79 | 0.79 | 0.78 | 0.78 | 0.68 | 0.82 |
| median over marks | 0.63 | 0.83 | 0.82 | 0.81 | 0.79 | 0.69 | 0.82 |

**seen_median7, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.43 | 0.57 | 0.58 | 0.58 | 0.58 | 0.47 | 0.62 |
| H3K27ac | 0.63 | 0.78 | 0.77 | 0.73 | 0.73 | 0.61 | 0.63 |
| H3K27me3 | 0.60 | 0.78 | 0.77 | 0.75 | 0.75 | 0.68 | 0.69 |
| H3K36me3 | 0.57 | 0.80 | 0.76 | 0.73 | 0.73 | 0.60 | 0.68 |
| H3K4me1 | 0.60 | 0.82 | 0.81 | 0.80 | 0.79 | 0.68 | 0.71 |
| H3K4me2 | 0.57 | 0.77 | 0.76 | 0.75 | 0.73 | 0.61 | 0.64 |
| H3K4me3 | 0.39 | 0.59 | 0.61 | 0.61 | 0.60 | 0.40 | 0.51 |
| H3K79me2 | 0.62 | 0.75 | 0.74 | 0.72 | 0.72 | 0.62 | 0.70 |
| H3K9ac | 0.68 | 0.82 | 0.81 | 0.82 | 0.81 | 0.66 | 0.77 |
| H3K9me3 | 0.56 | 0.68 | 0.68 | 0.65 | 0.66 | 0.54 | 0.71 |
| H4K20me1 | 0.64 | 0.67 | 0.69 | 0.64 | 0.64 | 0.51 | 0.69 |
| RAD21 | 0.54 | 0.70 | 0.70 | 0.69 | 0.68 | 0.61 | 0.71 |
| SMC3 | 0.55 | 0.69 | 0.69 | 0.65 | 0.66 | 0.53 | 0.65 |
| median over marks | 0.57 | 0.75 | 0.74 | 0.72 | 0.72 | 0.61 | 0.69 |

**k562, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.32 | -0.54 | -2.12 | 0.35 | -0.63 | 0.08 | 0.72 |
| H3K27ac | 0.23 | -14.54 | -0.80 | 0.18 | -0.36 | -0.13 | 0.30 |
| H3K27me3 | 0.28 | -3.93 | -2.50 | 0.41 | -0.99 | 0.31 | 0.42 |
| H3K36me3 | 0.43 | -0.21 | 0.18 | 0.68 | 0.56 | 0.57 | 0.72 |
| H3K4me1 | 0.08 | -10.79 | -1.93 | -0.35 | -1.13 | -1.13 | 0.17 |
| H3K4me2 | 0.54 | -36.60 | 0.35 | 0.78 | 0.66 | 0.51 | 0.65 |
| H3K4me3 | 0.63 | -45.66 | -0.91 | 0.62 | 0.04 | 0.47 | 0.77 |
| H3K79me2 | 0.40 | -1.92 | 0.47 | 0.63 | 0.56 | 0.43 | 0.69 |
| H3K9ac | 0.53 | -31.97 | 0.07 | 0.73 | 0.50 | 0.51 | 0.68 |
| H3K9me3 | 0.08 | -18.36 | -4.20 | -0.25 | -1.97 | -0.89 | 0.07 |
| H4K20me1 | 0.45 | -0.20 | -1.45 | 0.42 | -0.53 | 0.16 | 0.52 |
| RAD21 | 0.30 | -1.38 | -3.89 | 0.15 | -1.58 | -0.44 | 0.61 |
| SMC3 | 0.44 | -3.69 | -2.94 | 0.34 | -1.02 | -0.04 | 0.62 |
| median over marks | 0.40 | -3.93 | -1.45 | 0.41 | -0.53 | 0.16 | 0.62 |

**k562, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.59 | 0.43 | 0.35 | 0.64 | 0.50 | 0.53 | 0.86 |
| H3K27ac | 0.62 | 0.00 | 0.73 | 0.81 | 0.78 | 0.69 | 0.71 |
| H3K27me3 | 0.57 | -0.20 | 0.37 | 0.70 | 0.33 | 0.60 | 0.70 |
| H3K36me3 | 0.67 | 0.59 | 0.72 | 0.87 | 0.81 | 0.83 | 0.86 |
| H3K4me1 | 0.54 | 0.61 | 0.66 | 0.73 | 0.72 | 0.59 | 0.70 |
| H3K4me2 | 0.74 | 0.02 | 0.73 | 0.90 | 0.87 | 0.76 | 0.81 |
| H3K4me3 | 0.79 | 0.08 | 0.41 | 0.82 | 0.62 | 0.69 | 0.88 |
| H3K79me2 | 0.66 | 0.42 | 0.74 | 0.83 | 0.80 | 0.68 | 0.84 |
| H3K9ac | 0.73 | -0.09 | 0.77 | 0.89 | 0.85 | 0.77 | 0.82 |
| H3K9me3 | 0.46 | -0.04 | 0.03 | 0.47 | 0.22 | 0.28 | 0.61 |
| H4K20me1 | 0.67 | 0.44 | 0.36 | 0.67 | 0.41 | 0.45 | 0.73 |
| RAD21 | 0.57 | 0.43 | 0.35 | 0.64 | 0.49 | 0.49 | 0.82 |
| SMC3 | 0.67 | 0.34 | 0.29 | 0.66 | 0.45 | 0.44 | 0.79 |
| median over marks | 0.66 | 0.34 | 0.41 | 0.73 | 0.62 | 0.60 | 0.81 |

**k562, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.54 | 0.41 | 0.32 | 0.59 | 0.53 | 0.47 | 0.66 |
| H3K27ac | 0.48 | -0.29 | 0.63 | 0.67 | 0.63 | 0.48 | 0.60 |
| H3K27me3 | 0.60 | -0.24 | 0.33 | 0.64 | 0.23 | 0.56 | 0.71 |
| H3K36me3 | 0.50 | 0.26 | 0.61 | 0.63 | 0.63 | 0.47 | 0.63 |
| H3K4me1 | 0.50 | 0.51 | 0.61 | 0.62 | 0.64 | 0.59 | 0.65 |
| H3K4me2 | 0.65 | -0.02 | 0.51 | 0.68 | 0.54 | 0.50 | 0.64 |
| H3K4me3 | 0.41 | -0.01 | 0.12 | 0.37 | 0.25 | 0.22 | 0.49 |
| H3K79me2 | 0.57 | -0.00 | 0.66 | 0.66 | 0.62 | 0.37 | 0.68 |
| H3K9ac | 0.61 | -0.29 | 0.63 | 0.73 | 0.67 | 0.59 | 0.70 |
| H3K9me3 | 0.43 | 0.00 | 0.02 | 0.37 | 0.18 | 0.34 | 0.53 |
| H4K20me1 | 0.74 | 0.46 | 0.31 | 0.64 | 0.37 | 0.39 | 0.74 |
| RAD21 | 0.46 | 0.35 | 0.35 | 0.53 | 0.49 | 0.43 | 0.64 |
| SMC3 | 0.63 | 0.23 | 0.24 | 0.57 | 0.44 | 0.34 | 0.63 |
| median over marks | 0.54 | 0.00 | 0.35 | 0.63 | 0.53 | 0.47 | 0.64 |

**hmec, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.05 | 0.16 | 0.18 | 0.21 | 0.15 | 0.17 | 0.63 |
| H3K27ac | 0.23 | 0.40 | 0.53 | 0.54 | 0.51 | 0.43 | 0.49 |
| H3K27me3 | 0.20 | 0.42 | 0.36 | 0.37 | 0.40 | 0.24 | 0.52 |
| H3K36me3 | 0.33 | 0.52 | 0.54 | 0.52 | 0.57 | 0.51 | 0.66 |
| H3K4me1 | 0.34 | 0.41 | 0.53 | 0.52 | 0.54 | 0.47 | 0.59 |
| H3K4me2 | 0.53 | 0.57 | 0.70 | 0.70 | 0.69 | 0.66 | 0.72 |
| H3K4me3 | -0.14 | -2.18 | -0.22 | -0.18 | -0.13 | -1.00 | -0.01 |
| H3K79me2 | 0.46 | 0.58 | 0.60 | 0.54 | 0.54 | 0.36 | 0.78 |
| H3K9ac | 0.38 | 0.38 | 0.61 | 0.66 | 0.66 | 0.44 | 0.60 |
| H3K9me3 | 0.15 | 0.43 | 0.30 | 0.29 | 0.33 | 0.21 | 0.48 |
| H4K20me1 | -0.20 | -0.71 | -0.49 | -0.29 | -0.24 | -0.35 | 0.12 |
| RAD21 |  |  |  |  |  |  |  |
| SMC3 |  |  |  |  |  |  |  |
| median over marks | 0.23 | 0.41 | 0.53 | 0.52 | 0.51 | 0.36 | 0.59 |

**hmec, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.32 | 0.45 | 0.46 | 0.49 | 0.45 | 0.44 | 0.81 |
| H3K27ac | 0.50 | 0.74 | 0.73 | 0.73 | 0.72 | 0.66 | 0.70 |
| H3K27me3 | 0.49 | 0.68 | 0.62 | 0.63 | 0.64 | 0.54 | 0.73 |
| H3K36me3 | 0.58 | 0.76 | 0.75 | 0.72 | 0.76 | 0.72 | 0.81 |
| H3K4me1 | 0.60 | 0.79 | 0.76 | 0.72 | 0.74 | 0.69 | 0.79 |
| H3K4me2 | 0.73 | 0.85 | 0.84 | 0.84 | 0.84 | 0.81 | 0.86 |
| H3K4me3 | 0.65 | 0.72 | 0.76 | 0.79 | 0.79 | 0.60 | 0.79 |
| H3K79me2 | 0.70 | 0.79 | 0.80 | 0.78 | 0.79 | 0.64 | 0.91 |
| H3K9ac | 0.67 | 0.80 | 0.80 | 0.82 | 0.81 | 0.73 | 0.82 |
| H3K9me3 | 0.40 | 0.70 | 0.67 | 0.54 | 0.61 | 0.46 | 0.70 |
| H4K20me1 | 0.45 | 0.43 | 0.46 | 0.48 | 0.50 | 0.31 | 0.67 |
| RAD21 |  |  |  |  |  |  |  |
| SMC3 |  |  |  |  |  |  |  |
| median over marks | 0.58 | 0.74 | 0.75 | 0.72 | 0.74 | 0.64 | 0.79 |

**hmec, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.03 | 0.21 | 0.22 | 0.25 | 0.20 | 0.13 | 0.45 |
| H3K27ac | 0.49 | 0.73 | 0.70 | 0.71 | 0.72 | 0.62 | 0.68 |
| H3K27me3 | 0.41 | 0.63 | 0.49 | 0.56 | 0.56 | 0.47 | 0.66 |
| H3K36me3 | 0.49 | 0.76 | 0.74 | 0.69 | 0.75 | 0.71 | 0.72 |
| H3K4me1 | 0.62 | 0.79 | 0.74 | 0.66 | 0.72 | 0.67 | 0.79 |
| H3K4me2 | 0.42 | 0.70 | 0.67 | 0.66 | 0.67 | 0.53 | 0.67 |
| H3K4me3 | 0.34 | 0.47 | 0.51 | 0.64 | 0.62 | 0.18 | 0.54 |
| H3K79me2 | 0.72 | 0.80 | 0.74 | 0.73 | 0.75 | 0.72 | 0.79 |
| H3K9ac | 0.47 | 0.61 | 0.61 | 0.64 | 0.64 | 0.51 | 0.62 |
| H3K9me3 | 0.44 | 0.61 | 0.59 | 0.54 | 0.58 | 0.46 | 0.76 |
| H4K20me1 | 0.28 | 0.24 | 0.29 | 0.31 | 0.33 | 0.11 | 0.52 |
| RAD21 |  |  |  |  |  |  |  |
| SMC3 |  |  |  |  |  |  |  |
| median over marks | 0.44 | 0.63 | 0.61 | 0.64 | 0.64 | 0.51 | 0.67 |


### Reading

- The unseen-cell failure is fixed for `seq_meth_aug` (legacy Meth scaling plus
  random per-window scale and shift): median unseen R2 0.34 (from -3.69),
  Pearson 0.73 (from 0.43), Spearman 0.60 (from 0.31); K562 Pearson 0.73 (from
  0.34) and R2 0.41 (from -3.93), prediction offset 0.20 z (from 1.68) and slope
  0.98 (from 0.25). Marks with a negative R2 remain in single cells (K562
  H3K4me1 -0.35 and H3K9me3 -0.25; HMEC H3K4me3 -0.18 and H4K20me1 -0.29), the
  same marks for which `seq` and the ceiling are weak.
- Per-cell normalisation alone (`seq_meth_cellnorm`, `meth_cellnorm`) improves the
  ranking on K562 (Pearson 0.34 to 0.41 and -0.20 to 0.43 for meth-only) and HMEC
  (R2 0.41 to 0.53) but leaves a level error on K562 (offset 0.88 z, R2 -1.45,
  R2 after removing the offset 0.03), so it does not fix K562 by itself. Adding
  augmentation and window dropout to it (`seq_meth_cellnorm_aug_mdrop0.2`) gives
  Pearson 0.70 and R2 -0.24; the augmentation on the legacy scaling is the better
  variant here. Why the augmented legacy scaling beats the normalised input on
  K562 was not tested.
- Against gradient boosting (`GB_both`, section 41) on the two unseen cells the
  best CNN has R2 0.34 against 0.12, Pearson 0.73 against 0.64, Spearman 0.60
  against 0.51 (K562: 0.41 / 0.73 / 0.63 against 0.16 / 0.60 / 0.47; HMEC: 0.52 /
  0.72 / 0.64 against 0.36 / 0.64 / 0.51). The unaugmented `seq_meth` is below
  gradient boosting on unseen cells. On the seven seen cells all CNN variants
  except `meth_cellnorm` are above gradient boosting (R2 0.62 to 0.69 against
  0.46); the augmentation costs 0.05 R2 and 0.02 Pearson there.
- Against sequence alone (`seq`) the best methylation-aware CNN adds on unseen
  cells 0.10 R2 (0.34 against 0.24), 0.16 Pearson (0.73 against 0.57) and 0.10
  Spearman. The gain comes mostly from HMEC (R2 0.52 against 0.23); on K562 R2 is
  equal (0.41 against 0.40) and only Pearson rises (0.73 against 0.66). The
  cell-agnostic ceiling (`ceiling_loo`, the mean of the seven measured training
  tracks, which needs ChIP data of other cells as input) is still above every
  model on unseen cells (R2 0.61, Pearson 0.82).

### Limitations

- Two independent unseen cells (K562, HMEC); medians of two cells have no
  interval, and RAD21 and SMC3 exist only for K562. One training run per variant,
  one seed.
- The four variants were compared on the same two unseen cells that report the
  result; there is no validation on an unseen cell, so the choice of
  `seq_meth_aug` as the best variant is made on the evaluation cells and the
  size of its advantage is an optimistic estimate. The diagnosis (K562 input
  outside the training range) rests on the correlation of failure with input
  distance across two cells and the meth-only control, not on an intervention
  on K562 (for example shifting its methylation into the training range).
- Gradient boosting uses all even chromosomes and 500,000 training rows, the CNN the
  even chromosomes without chr2 and chr4 (validation); the two columns are close
  but not identical test sets.
- `--target-norm cell` and `--meth-norm rank` are implemented and smoke-tested only.
  The remote repository copy of the script differs from the committed one only in the
  header comment.

### Files

`scripts/seq_cnn_chip.py`; `results/seq_cnn/v2/seq_cnn_results.csv`,
`seq_cnn_summary.csv`, `seq_cnn_partition.csv` (all variants, section 42 rows
included; the partition still uses `seq`, `seq_meth`, `meth`),
`comparison_by_mark.csv` (per mark medians for seen, unseen, K562 and HMEC with
`GB_both` and `ceiling_loo`), `diag_prediction_bias.csv` (y_mean, p_mean, sds,
slope, R2, R2 after removing the offset, Pearson per model, cell and mark). The
section 42 files stay in `results/seq_cnn/`.

## 60. Do CNN-predicted ChIP tracks help the track-dropout Hi-C model when measured tracks are missing? (FINAL; section 59 was the last number when this was written)

Question: when ChIP tracks are absent, does replacing them by tracks that the sequence
+ methylation CNN predicts (sections 42 and 59, variant `seq_meth_aug`) improve the
existing track-dropout Hi-C model over leaving them absent? Inference only; no Hi-C
training.

### Method

- Hi-C model: `all12refres_drop_meth_10kb` (13 ChIP tracks + Meth input, residual on the
  consensus, track dropout 0.5; the model with Meth; `all12refin_drop_10kb` has no Meth
  input and was not run). 413 Akita test regions at 10 kb, `crosscell_predict.py` with
  TTA, median per-region Pearson on the upper triangle k >= 2 (column `model`), plus the
  structure metrics of section 35. Consensus: leave-one-out for the training cells,
  consensus12 for K562 and HMEC (HMEC deep, same ChIP and Meth inputs as HMEC).
- Stage 1: `seq_meth_aug` checkpoint (`ckpt_best.pt`, copied to
  `/mnt/processed/wolffjoa/hic-multiome/runs/seq_cnn_twostage/`) predicts the 13 marks at
  10 kb for the bins of every 192-bin Hi-C window (and the bins 1 Mb downstream for the
  control), with reverse-complement averaging. K562 and HMEC are unseen by the CNN; the
  seven training cells gm12878, imr90, h1, hepg2, a549, gm23248, cd14mono were fitted by
  it, on the odd chromosomes. Of the 413 regions, 169 lie on the CNN test chromosomes
  (chr6, 8, 10, ..., 22; held out of CNN training and validation) and the rest on odd
  chromosomes (CNN training sequence) or chr2 and chr4 (CNN validation). Both the 413 and
  the 169 are reported; for the training cells only the 169 are free of in-sample
  chromosomes.
- Scale: the CNN output z is converted to log1p signal units with the pooled training
  statistics (z*sd + mu, `D.mu`, `D.sd`, global target normalisation), then to the Hi-C
  input by the Hi-C model's own `norm_tracks.json` (log1p, per-track z-score, no
  smoothing): both models use the same log1p of the 10 kb bin mean. No per-cell scaling,
  no measured ChIP of the evaluated cell. Check on cells with measured tracks
  (`scale_check.csv`, CNN test chromosomes, windows of the evaluation regions, Hi-C z
  units, mean over marks): imputed mean minus measured mean is -0.26 for K562 (K562
  CTCF -0.25, H3K27ac -0.49, RAD21 -0.52), +0.01 for HMEC and between -0.08 and +0.07 for
  the seven training cells; the SD of the imputed track is 0.74 to 0.91 times the
  measured SD (K562 0.83, HMEC 0.84; per mark 0.56 to 1.47) as expected from a
  regression predictor; Pearson of imputed with measured 0.68 (K562), 0.64 (HMEC),
  0.70 to 0.77 (training cells).
- Scenarios (set kept; every other track absent or imputed): `all13` (every mark the cell
  has; HMEC lacks RAD21 and SMC3, which stay absent), `meth` (Meth only), `none` (no ChIP,
  no Meth), `ctcf`, `ctcf_h3k27ac`, `five` (CTCF, H3K27ac, H3K4me3, H3K27me3, H3K36me3),
  each with Meth given to the Hi-C model (`+meth`, `meth`) and without. Variants:
  absent, `@imp` (CNN prediction for every mark not kept), `@shuf` (the CNN prediction
  of the bin 100 positions (1 Mb) downstream on the same chromosome: right cell and
  scale, wrong locus). Paired differences use a region bootstrap (2000 resamples) of the
  median.
- Code: `scripts/twostage_cnn_imputation.py` (stages `cnn distribution predict summary`),
  `scripts/crosscell_predict.py` gets `--impute name=npz` and keep sets named
  `<set>@<name>`.

### Results

Median per-region Pearson, 413 regions (set with Meth given; `none` has no Meth):

| cell | scenario | absent | CNN-imputed | shuffled | consensus |
|---|---|---|---|---|---|
| K562 | all13 (upper bound) | 0.906 | 0.906 | 0.906 | 0.859 |
| K562 | meth only | 0.869 | 0.878 | 0.772 | 0.859 |
| K562 | none (no Meth) | 0.855 | 0.869 | 0.716 | 0.859 |
| K562 | CTCF + Meth | 0.878 | 0.878 | 0.775 | 0.859 |
| K562 | CTCF + H3K27ac + Meth | 0.896 | 0.886 | 0.819 | 0.859 |
| K562 | 5 marks + Meth | 0.892 | 0.888 | 0.839 | 0.859 |
| HMEC | all13 (upper bound) | 0.905 | 0.905 | 0.905 | 0.907 |
| HMEC | meth only | 0.903 | 0.898 | 0.851 | 0.907 |
| HMEC | none (no Meth) | 0.904 | 0.896 | 0.833 | 0.907 |
| HMEC | CTCF + Meth | 0.900 | 0.900 | 0.856 | 0.907 |
| HMEC | CTCF + H3K27ac + Meth | 0.903 | 0.908 | 0.889 | 0.907 |
| HMEC | 5 marks + Meth | 0.906 | 0.907 | 0.894 | 0.907 |

Paired difference imputed minus absent (median per region, 95% region bootstrap
interval, regions improved of 413; `paired_differences.csv`):

| cell | scenario | difference | interval | improved |
|---|---|---|---|---|
| K562 | meth only | +0.005 | 0.003 to 0.009 | 246 |
| K562 | none | +0.012 | 0.007 to 0.018 | 256 |
| K562 | CTCF + Meth | +0.000 | -0.003 to 0.003 | 210 |
| K562 | CTCF + H3K27ac + Meth | -0.006 | -0.009 to -0.005 | 138 |
| K562 | 5 marks + Meth | -0.005 | -0.006 to -0.004 | 162 |
| HMEC | meth only | -0.003 | -0.005 to 0.000 | 183 |
| HMEC | none | -0.003 | -0.007 to 0.000 | 180 |
| HMEC | CTCF + Meth | -0.001 | -0.004 to 0.002 | 201 |
| HMEC | CTCF + H3K27ac + Meth | +0.002 | -0.001 to 0.004 | 219 |
| HMEC | 5 marks + Meth | +0.000 | -0.002 to 0.002 | 205 |

On the 169 CNN-test-chromosome regions the picture is the same (K562 meth only +0.006,
none +0.014, CTCF + H3K27ac + Meth -0.007; HMEC meth only -0.005, none -0.006).
Median over the seven training cells (169 regions, in-distribution cells; CNN-fitted):
meth only +0.012, none +0.029, CTCF + Meth +0.011, CTCF only +0.023, CTCF + H3K27ac +
Meth +0.000, 5 marks + Meth -0.002; one cell dominates (cd14mono, whose consensus is
0.626: meth only +0.090, CTCF + Meth +0.053, but CTCF + H3K27ac + Meth -0.022 and 5 marks
+ Meth -0.027).
Structure metrics (`scenario_medians.csv`): small and mixed. K562 meth only:
compartment 0.903 to 0.914, eigenvector 0.953 to 0.960, SCC 0.860 to 0.865, loop APA
0.629 to 0.681, insulation 0.946 to 0.940; the median boundary F1 falls from 1.000 to
0.667 for K562 CTCF + Meth and 5 marks + Meth. HMEC is unchanged within 0.01 except loop
APA (+-0.02 to 0.03). The all13 rows are identical in the three variants because no mark
is left to impute (HMEC keeps its two missing cohesin marks absent).

Shuffled control: replacing the CNN tracks by their 1 Mb shifted copies lowers every
scenario in which a mark is replaced below absent (K562 meth only 0.772, none 0.716;
HMEC meth only 0.851, none 0.833; K562 -0.05 to -0.14, HMEC -0.01 to -0.07 against
absent) and below imputed in all of them. The Hi-C model reads the imputed tracks locus
by locus; wrong-locus tracks are actively harmful, so the locus information in the CNN
prediction is what carries the small gain, and the neutral absent value is the safer
fallback when the CNN is unsure.

### Verdict

Imputation helps only where almost nothing is measured, and by little. On K562
the paired median difference is +0.005 (meth only; +0.009 in medians, a quarter of the
0.037 gap from meth only to all marks) and +0.012 (neither Meth nor ChIP) over absent,
against a 0.047 gap of the full model over the consensus; on HMEC, where the model equals the
consensus within 0.002 at full input, it changes nothing within +-0.005. Once CTCF and
H3K27ac or five marks are measured, imputing the rest is neutral to harmful on K562
(-0.005 to -0.006, intervals exclude 0). The CNN route to Meth is also redundant: the
model with the imputed tracks and no Meth input (K562 none 0.869) equals the model
with Meth and absent tracks (0.869). The gains stay well below the 0.016 to 0.05
model-vs-consensus gaps; imputing all 13 marks gives about what one measured CTCF track
gives (K562 absent: CTCF + Meth over meth only +0.009, CTCF over none +0.011).

### Caveats

- Two independent unseen cells (K562, HMEC); medians of two cells have no interval, and
  the bootstrap resamples regions of one cell, not cells.
- The CNN variant `seq_meth_aug` was selected on K562 and HMEC (section 59), so the
  imputed tracks of these two cells are optimistic; the Hi-C scenario gain is bounded by
  that optimism and is still small.
- The Hi-C model also receives the consensus, so the gains are on top of it; where the
  consensus is already near the model (HMEC) there is nothing to gain.
- The imputed tracks of the seven training cells are in-sample for the CNN (cell and
  odd chromosomes; even test chromosomes are held-out in position, not in cell); they
  serve as contrast, not evidence for unseen cells. For unseen cells, 244 of the 413
  regions lie on CNN training or validation chromosomes (sequence seen in other cells);
  the 169-region subset agrees.
- K562 imputed tracks are shifted by -0.26 z (mean over marks; -0.5 for H3K27ac and
  RAD21), the known K562 failure of section 59; no per-cell rescaling was tried, since a
  correction would need measured K562 ChIP.
- One Hi-C model, one seed. The seed noise of the median Pearson is 0.013 (section 15);
  the paired differences share the model and regions, so the intervals above exclude
  seed variation, and most differences are smaller than that noise.

### Files

`scripts/twostage_cnn_imputation.py`; `scripts/crosscell_predict.py` (`--impute`);
`results/twostage_cnn/scenario_medians.csv` (all metrics, subsets all413 and clean169),
`paired_differences.csv`, `scenario_table_pearson.csv`, `scale_check.csv`,
`per_region_scores.csv.gz`, `twostage_cnn.png`. CNN checkpoint, caches, track dumps
(`pred/`) and per-set Hi-C predictions under
`/mnt/processed/wolffjoa/hic-multiome/runs/seq_cnn_twostage/`.

## 61. AML-inclusive cross-validation without healthy donors: pooled ten test patients and the section 36 claims re-tested (FINAL; section 60 was the last number when this was written)

Question: with the rebuilt folds (sections 50, 52, 54; healthy PBMC donors 103,
673, 576 removed, same-donor pairs 168/496 and 260/424 kept together, three
patient-disjoint folds over the ten AML patients with Hi-C and CUT&Tag; test
patients fold 10: 018, 027, 270; fold 11: 168, 546, 629, 1360; fold 12: 472,
773, 1916), do the section 36 findings (other-patient mean beats every model at
25 kb; model beats it at 10 kb; patient-specific gain; marks +0.013 / +0.032;
AML-specific loops and stripes not reproduced; lab indicator no help) hold?

Method. `scripts/aml_folds_v2_summary.py` pools `eval_f10/11/12_*.csv` from
`/mnt/processed/wolffjoa/hic-multiome/runs/aml_train_remote/fold10..12`
(scored by `scripts/aml_train_eval.py --fold N`, 413 Akita test regions, 25 and
10 kb). The unit is the donor (10 donors; 168 stands for 168+496, the 260/424
pair is in no test set). Pooled value = median over the ten patients, interval
= 95% percentile bootstrap over donors (10000 resamples, seed 0). Paired
differences are patient-wise. Two other-patient references exist: `ref_other_patients`
= mean of the fold's training patients (7 or 6 patients; the map the models
receive as input) and `ref_lopo_all` = mean of all other patients except the
same donor (9 patients). Their Pearson is the only metric scored for baselines.
Patient-specific = windows in the patient's top 20% on both deviation scores
(35 to 40 windows per patient). Models: `aml_refin` (AML-inclusive, reference
input), `aml_honest` (no reference input), `base_honest` and `base_refin@aml`
(trained without AML data), `aml_refin@hspc` (same model fed the HSPC
reference). Conditions: own marks, none.

### Pooled medians over ten test patients (donor bootstrap 95% interval)

| 25 kb | marks | no marks |
|---|---|---|
| Pearson aml_refin | 0.899 [0.882, 0.918] | 0.894 [0.872, 0.916] |
| SCC aml_refin | 0.896 [0.872, 0.913] | 0.887 [0.867, 0.912] |
| insulation | 0.956 [0.947, 0.966] | 0.958 [0.947, 0.964] |
| loop enrichment | 0.803 [0.745, 0.832] | 0.791 [0.748, 0.826] |
| compartment | 0.919 [0.904, 0.933] | 0.914 [0.894, 0.929] |
| patient-specific windows (Pearson) | 0.770 [0.686, 0.814] | 0.732 [0.692, 0.806] |
| Pearson aml_honest | 0.595 [0.534, 0.625] | 0.434 [0.422, 0.467] |
| Pearson base_honest | 0.533 [0.503, 0.603] | 0.448 [0.416, 0.490] |
| Pearson base_refin@aml | 0.885 [0.866, 0.907] | 0.888 [0.866, 0.918] |
| Pearson aml_refin@hspc | 0.870 [0.847, 0.895] | 0.868 [0.838, 0.886] |

| 10 kb | marks | no marks |
|---|---|---|
| Pearson aml_refin | 0.813 [0.769, 0.847] | 0.809 [0.764, 0.845] |
| SCC aml_refin | 0.805 [0.745, 0.837] | 0.796 [0.736, 0.837] |
| insulation | 0.961 [0.951, 0.966] | 0.959 [0.948, 0.968] |
| loop enrichment | 0.429 [0.353, 0.497] | 0.420 [0.356, 0.495] |
| compartment | 0.879 [0.842, 0.906] | 0.873 [0.824, 0.905] |
| patient-specific windows (Pearson) | 0.550 [0.526, 0.673] | 0.558 [0.502, 0.660] |
| Pearson aml_honest | 0.494 [0.459, 0.542] | 0.357 [0.336, 0.404] |
| Pearson base_honest | 0.429 [0.385, 0.500] | 0.345 [0.310, 0.426] |
| Pearson base_refin@aml | 0.795 [0.745, 0.839] | 0.805 [0.760, 0.843] |
| Pearson aml_refin@hspc | 0.760 [0.717, 0.806] | 0.760 [0.707, 0.804] |

Baselines (Pearson, then patient-specific Pearson), 25 kb: other patients
(training mean) 0.908 [0.885, 0.935] and 0.750 [0.687, 0.814]; other patients
(all nine) 0.911 and 0.754; HSPC consensus (4 donors) 0.884 and 0.697; HSPC
training mean (2 donors) 0.866 and 0.678; consensus12 0.822 and 0.558.
10 kb: 0.803 [0.754, 0.834] and 0.527 [0.488, 0.642]; all nine 0.814 and 0.559;
HSPC 0.748 and 0.498; HSPC training mean 0.711 and 0.442; consensus12 0.694 and
0.394. Per-patient values in `per_patient.csv`.

### Claims re-tested (paired, patient-wise; difference with donor bootstrap interval; patients where the first term is larger)

| claim | 25 kb | 10 kb |
|---|---|---|
| aml_refin(marks) minus training-mean reference, Pearson | -0.012 [-0.015, -0.002], 1/10 | +0.011 [+0.007, +0.019], 10/10 |
| same against the nine-patient mean | -0.016 [-0.020, -0.009], 0/10 | -0.004 [-0.008, +0.005], 4/10 |
| best of all models per patient minus training mean | -0.011 [-0.013, -0.002], 1/10 | +0.013 [+0.009, +0.019], 10/10 |
| aml_honest, base_honest minus training mean | -0.32, -0.37, 0/10 each | -0.28, -0.34, 0/10 each |
| base_refin@aml(marks) minus training mean | -0.025 [-0.031, -0.012], 0/10 | +0.000 [-0.012, +0.002], 5/10 |
| patient-specific windows, aml_refin(marks) minus training mean | -0.002 [-0.014, +0.027], 5/10 | +0.035 [+0.026, +0.056], 10/10 |
| patient-specific, same against the nine-patient mean | -0.012 [-0.028, +0.021], 3/10 | +0.017 [-0.003, +0.056], 8/10 |
| marks minus no marks, aml_refin, Pearson | +0.004 [-0.000, +0.009], 8/10 | +0.000 [-0.005, +0.011], 5/10 |
| marks minus no marks, aml_refin, patient-specific | +0.007 [-0.006, +0.036], 7/10 | +0.013 [-0.001, +0.026], 7/10 |
| marks minus no marks, aml_honest, Pearson | +0.135 [+0.103, +0.187], 10/10 | +0.122 [+0.100, +0.176], 10/10 |
| AML training (aml_refin minus base_refin@aml, marks), Pearson | +0.014 [+0.007, +0.016], 10/10 | +0.014 [+0.006, +0.022], 9/10 |
| AML training, patient-specific | +0.005 [-0.012, +0.017], 6/10 | +0.014 [+0.002, +0.028], 9/10 |

Residual correlation corr(P - R, T - R), R = training-mean reference, all
windows (patient-specific windows in brackets): aml_refin marks 25 kb 0.203
[0.256], none 0.111 [0.150]; 10 kb marks 0.278 [0.282], none 0.238 [0.265];
base_refin@aml marks 0.191 [0.261] and 0.213 [0.238]; aml_honest marks 0.080
and 0.115. The prediction deviates from the reference in the direction of the
measured deviation, with a small share of its variance.

Loops and stripes (10 kb, Xu et al. AML-specific loops and stripes, windows
from the evaluation output, per-patient medians then pooled over ten donors;
log2 donut enrichment and stripe band minus the reference): measured loops
+0.292 [+0.239, +0.354] against aml_refin(marks) -0.012 [-0.019, +0.009] and
no marks -0.029 [-0.033, -0.018]; measured stripes +0.184 [+0.084, +0.205]
against +0.006 [-0.022, +0.026]. Across windows corr(P - R, T - R) is 0.196
for loops and 0.357 for stripes (aml_refin marks); base_refin@aml (no AML
training) gives 0.211 and 0.325. The enrichment above the reference is not
produced by either model. The evaluation output already contains this stage,
so no perturbation run was needed.

### Subtype stratification (descriptive, `subtype_groups.csv`)

Groups from `results/aml_subtype/patient_subtype_table.csv` (consensus_group):
TET2 (018, 546, 1916), RUNX1 (027, 168), KMT2B (270, 629), CEBPA (773),
STAG2 (472), FLT3-ITD only (1360). Median aml_refin(marks) minus training
mean at 10 kb, Pearson (patient-specific): KMT2B +0.015 (+0.083), RUNX1 +0.013
(+0.059), CEBPA +0.019 (+0.037), FLT3-ITD only +0.010 (+0.026), STAG2 +0.007
(+0.021), TET2 +0.006 (+0.030). At 25 kb every group lies between -0.016 and
-0.002 in Pearson. No fold removes a whole group that has several patients;
the three singleton groups (CEBPA, STAG2, FLT3-ITD only) are the leave-one-
group-out cases (no patient of the same group in training): median patient-
specific gain +0.026 against +0.040 for the seven patients with a same-group
patient in training, Pearson +0.010 against +0.012. The patient-specific gain is
carried by KMT2B patient 270 (+0.126). Groups have one to three patients; the
comparison supports no conclusion about subtype transfer.

### Old fold 0 against the rebuilt folds

The four old test patients are 027 and 270 (now fold 10) and 773 and 1916 (now
fold 12), so the comparison is patient-wise (`old_vs_new_same_patients.csv`).
Median over these four patients, old (contaminated) / rebuilt:

| quantity | 25 kb | 10 kb |
|---|---|---|
| aml_refin Pearson | 0.900 / 0.908 | 0.818 / 0.812 |
| training-mean reference | 0.912 / 0.916 | 0.802 / 0.800 |
| nine-patient mean (old: all eleven other) | 0.922 / 0.921 | 0.814 / 0.810 |
| aml_refin patient-specific | 0.755 / 0.763 | 0.592 / 0.600 |
| training-mean reference, patient-specific | 0.751 / 0.774 | 0.541 / 0.559 |
| aml_honest Pearson | 0.623 / 0.625 | 0.525 / 0.521 |
| aml_refin marks minus none, Pearson | +0.006 / +0.004 | +0.008 / +0.003 |
| aml_refin marks minus none, patient-specific | +0.031 / +0.004 | +0.018 / +0.019 |
| corr(P - R, T - R), marks | 0.167 / 0.209 | 0.234 / 0.278 |

Individual patients move by at most 0.025 in Pearson (up to 0.05 in
patient-specific windows, e.g. 773 at 25 kb patient-specific reference 0.692
against 0.697, 270 0.623 against 0.605). Removing the healthy donors from
training and from the reference, and separating the duplicate pairs, changed
the four-patient medians by 0.01 or less in Pearson; the old section 36 numbers
were not inflated by the contamination. The main change is the sample: ten
patients with intervals instead of four. The 25 kb marks effect in
patient-specific windows (+0.031 old) falls to +0.004 on the same four
patients and +0.007 on all ten, a difference within the interval. The old
claim that the reference wins in 12/12 patients at 25 kb becomes: aml_refin is
below the training mean in 9/10 patients and below the nine-patient mean in
10/10; the old 4/4 model wins at 10 kb becomes 10/10 against the training mean
and 4/10 against the nine-patient mean.

### Variants not re-run

AML+Meth (`aml_meth_refin`, section 38) was not trained for folds 10 to 12.
The residual variant `aml_refres` (section 45) exists for old fold 0 only, at
10 kb, and is contaminated by the old split. Old fold-0 medians over four
patients, 10 kb: aml_refres marks 0.812 Pearson, 0.588 patient-specific (none
0.803, 0.566); aml_refin marks 0.818, 0.592; aml_meth_refin marks+Meth 0.816,
0.586; training mean 0.802, 0.541 (`old_fold0_refres_meth.csv`). The residual
form does not improve on the reference-input form. The lab indicator ablation
(`aml_refin_nolab`) was fold 0 and 25 kb only and is not re-tested.

### Verdict

At 25 kb the other-patient mean is at least as good as every model in
Pearson (aml_refin 1/10 patients above it, median -0.012; -0.016 against the
nine-patient mean, 0/10); in patient-specific windows the model and the
reference tie (-0.002 [-0.014, +0.027]). At 10 kb aml_refin is above the
training-mean reference in 10/10 patients (+0.011 Pearson, +0.035 in
patient-specific windows, intervals above zero) and level with the
nine-patient mean (-0.004, 4/10; patient-specific +0.017, 8/10, interval
including zero), so part of the 10 kb gain reflects the model's reference
being built from fewer patients. The AML training adds +0.014 Pearson to the
same architecture trained without AML (10/10 at 25 kb). The deviation of the
prediction from the reference correlates 0.20 to 0.28 with the measured
deviation, and the model contains no AML-specific loop or stripe enrichment
(measured +0.29 and +0.18 log2, model about 0). Marks add nothing detectable
to the reference-input model (+0.000 to +0.013, intervals include zero), while
the no-reference model needs them (+0.12 to +0.14). With AML patients in
training the model remains a refined version of the other-patient reference,
about 0.01 Pearson above it at 10 kb in this design and not above it at 25 kb;
it does not predict AML-specific structure.

Caveats. Ten donors, folds of three or four, one seed per model and fold; the
bootstrap covers patient sampling and excludes training variance. Pooled
intervals treat the three folds as exchangeable although the reference of each
fold is built from different patients. Seed variation was not measured here, so differences of about 0.01 cannot be separated from it. Subtype groups have one to three patients.
Baselines exist for Pearson only; insulation, loop and compartment scores are
available for models alone.

### Files

`scripts/aml_folds_v2_summary.py`; `results/aml_train_v2/`: `per_patient.csv`,
`pooled_medians.csv`, `paired_claims.csv`, `loop_stripe_per_patient.csv`,
`loop_stripe_pooled.csv`, `subtype_per_patient.csv`, `subtype_groups.csv`,
`old_vs_new_same_patients.csv`, `old_fold0_refres_meth.csv`, `aml_folds_v2.png`.
Evaluation inputs under `/mnt/processed/wolffjoa/hic-multiome/runs/aml_train_remote/`.

## 62. Loop-level test: does the model predict AML-specific loops and stripes in held-out patients? (FINAL; section 61 was the last number when this was written)

Question: the eight site figures of section 36b-v2 show a predicted but weaker gain over HSPC, the fold analysis of section 61 shows no gain beyond the AML reference. This section tests all AML-specific objects of the ten test donors of folds 10, 11, 12 (018, 027, 270 | 168, 546, 629, 1360 | 472, 773, 1916) at 10 kb.

### Design
Objects and matched controls are those of the loop classifier (sections 47, 57; `results/aml_loop_classifier/features.csv.gz`, identical sampling): AML-specific loops of Xu et al. Suppl. Table 1 (2276 loops, 193 to 284 per donor) and stripes of Suppl. Table 4 (1075 stripes, at most 150 per donor, span up to 1.5 Mb; 319 wider ones are dropped because they do not fit the 1.92 Mb window); controls: A_hspc, A_full, A_do (shared loops or stripes matched 1:1) and B_rand (random pair or shifted rectangle at the same distance). Every object and control gets a 1.92 Mb window centred on it; the fold's models predict it (own marks, the fold's AML reference): aml_refin with marks and without any ChIP (`none`), aml_refin with the HSPC training reference (`aml_refin@hspc`), and the model trained without AML data (`base_refin@aml`). The reference baseline is the fold's AML reference map itself; measured HSPC is the four-donor HSPC consensus. Loop score: log2 donut enrichment (inner 1, outer 4, as the labels of section 47), maximum over the 3x3 pixels around the annotated pixel; stripe score: mean log O/E of the stripe rectangle. Gain = score minus measured HSPC score. Recurrence: other donors among the 25 AML samples of Suppl. Table 1 and 4 with the same object (loops within one bin in both anchors; stripes overlapping anchor and extent), 168/496 and 260/424 counted once, the donor's own partner excluded: recurrent >= 2 other donors, shared 1, private 0. Intervals: bootstrap over the ten donors (500 to 2000 resamples). 16,466 objects scored. Inference on one local GPU.

Of the loops 1805 are private (79 percent), 278 shared by one other donor, 193 recurrent (8 percent); of the stripes 285 private, 195 shared, 595 recurrent (55 percent). Only 87 loops are present in two or more training donors of the fold (the reference), 195 in one.

### Results, loops (median over objects, log2 enrichment)
| class | n | measured gain | model gain | reference alone | no marks | no AML training | model minus reference | model minus no marks |
|---|---|---|---|---|---|---|---|---|
| all | 2276 | 0.404 [0.355, 0.476] | 0.060 [0.046, 0.070] | 0.096 | 0.037 | 0.079 | -0.034 [-0.043, -0.026] | +0.014 [0.006, 0.020] |
| recurrent | 193 | 0.450 | 0.163 | 0.219 | 0.146 | 0.204 | -0.045 [-0.063, -0.034] | +0.010 [0.003, 0.028] |
| shared by 1 | 278 | 0.388 | 0.094 | 0.137 | 0.064 | 0.117 | -0.039 | +0.021 |
| private | 1805 | 0.401 | 0.045 | 0.081 | 0.025 | 0.063 | -0.032 [-0.040, -0.024] | +0.013 [0.006, 0.019] |

Measured gain over the reference is +0.30 (all), +0.24 (recurrent), +0.32 (private); the model is below the reference in 10/10 donors in every class. Amplitude ratio (model gain / measured gain): 0.15 all, 0.36 recurrent, 0.11 private; the reference alone 0.24, 0.49, 0.20. Predicted gain over the predicted HSPC-input model (`aml_refin@hspc`, which itself sits 0.04 below measured HSPC): 0.083 all, 0.248 recurrent, 0.069 private. Promoter-enhancer-like loops (n=301): measured 0.368, model 0.080, reference 0.100; promoter-repressive-like (n=103): 0.375, 0.046, 0.098.

Predicted against measured gain (Spearman, pooled, donor bootstrap): model 0.28 [0.19, 0.35] (Pearson 0.26 [0.17, 0.36]; median per donor 0.31, positive in 10/10), reference alone 0.29 [0.22, 0.36], no AML training 0.30, no marks 0.25. Recurrent loops: model 0.34, reference 0.46 [0.34, 0.56]. Private: model 0.25, reference 0.25. Against the deviation from the reference (what the model adds): model 0.18 [0.09, 0.26] (recurrent 0.10, includes zero), no marks 0.12.

Separation of AML-specific loops from matched shared loops (AUROC, A_hspc controls):
| score | measured | model | no marks | no AML | reference |
|---|---|---|---|---|---|
| gain over HSPC, all | 0.834 [0.820, 0.860] | 0.621 [0.580, 0.663] | 0.625 | 0.661 | 0.585 [0.541, 0.618] |
| gain, recurrent | 0.848 | 0.771 [0.718, 0.819] | 0.775 | 0.806 | 0.771 |
| gain, private | 0.835 | 0.594 [0.549, 0.636] | 0.599 | 0.635 | 0.551 |
| enrichment, all (A_hspc) | 0.199 | 0.086 | 0.075 | 0.098 | 0.062 |
| enrichment, all (B_rand) | 0.890 | 0.807 | 0.791 | 0.795 | 0.758 |
Fraction of AML-specific loops above the 90th percentile of the matched controls (gain; chance 0.10): measured 0.50, model 0.153 [0.120, 0.194], reference 0.170, no AML 0.215; recurrent 0.575 / 0.446 / 0.492; private 0.500 / 0.111 [0.094, 0.143] / 0.122. A_full and A_do give the same picture (model gain AUROC 0.568 with A_full). The shared controls are selected for enrichment >= 0.6 in both patient and HSPC, so their enrichment exceeds that of AML-specific loops by construction (AUROC of raw enrichment far below 0.5 in measured data too); the gain is the discriminating score, and against random pairs the raw enrichment separates the loops (measured 0.89, model 0.81, recurrent 0.95).

### Results, stripes
Measured gain 0.221 (mean log O/E), model 0.044 [0.017, 0.064], reference 0.044, no marks 0.044, no AML 0.027; model minus reference +0.003 [-0.014, 0.022], model minus no marks +0.003. Amplitude ratio 0.20 (recurrent 0.22, private 0.22). Spearman of predicted and measured gain 0.50 [0.42, 0.58] (reference 0.46, no marks 0.44); against the deviation from the reference 0.28 (no marks 0.10). AUROC against matched shared stripes (A_hspc, gain): measured 0.63, model 0.52 [0.49, 0.56], reference 0.46, no AML 0.56; enhancer-like stripes (n=326) model 0.61 (measured 0.63, reference 0.53), repressive-like (n=442) 0.49 (measured 0.64), other 0.48. Enhancer-like stripes are the one class where the own-marks model is above the reference (+0.018 [0.001, 0.039], above the no-marks model by +0.021 [0.004, 0.041]); private stripes +0.026 [0.009, 0.041] over the reference. Fraction above the control p90: measured 0.25, model 0.16, reference 0.12.

### Verdict
(a) Against healthy cells the model predicts a gain at AML-specific loops, with the right sign in 10/10 donors, 15 percent of the measured amplitude (median +0.06 against +0.40 log2) and a rank correlation of 0.28 with the measured gain per loop. It separates AML-specific from matched shared loops with AUROC 0.62 (measured 0.83), and 15 percent of AML-specific loops exceed the 90th percentile of the controls (measured 50 percent). Stripes: gain 20 percent of the measured amplitude, AUROC 0.52.
(b) Recurrent loops (present in at least two other donors) carry the signal: gain 0.16, 36 percent of the measured amplitude, AUROC 0.77, 45 percent above control p90. Private loops: gain 0.045, 11 percent amplitude, AUROC 0.59, 11 percent above control p90 (chance 10 percent).
(c) The signal comes from the reference. The reference map alone has the same correlation (0.29), the same recurrent AUROC (0.77), and a larger gain (0.096) than the model; the model lies 0.034 log2 below it (10/10 donors), the model trained without AML data is level with it (0.079). Own marks add +0.014 log2 over the no-marks condition (8/10 donors; +0.010 recurrent, +0.021 promoter-enhancer-like and promoter-repressive-like) and +0.02 for enhancer-like stripes; they do not lift AUROC (0.621 against 0.625). Section 61 and the eight sites are consistent: the correlation of 0.55 to 0.61 over seven hand-picked sites is higher than the 0.28 over all loops (the sites were chosen for a clear gain, and most are recurrent), and the model contains no gain beyond the reference (measured +0.30 over the reference, model -0.034).
(d) Suggested wording: "In the ten held-out patients the model predicts a weaker gain at AML-specific loops than in healthy cells (median +0.06 against +0.40 log2 measured; rank correlation 0.28), almost entirely at loops that recur in other patients and are therefore present in the AML reference (AUROC against matched shared loops 0.77); at loops private to the patient the gain is 0.045 and the separation 0.59. The AML reference map alone gives the same result, and the patient's own marks add 0.014 log2. The model reproduces AML-specific structure that other patients share and does not predict the patient-private loops and stripes."

Caveats. Ten donors, one seed per model; 10 kb only (25 kb not analysed; the loop tables are 10 kb); the matched shared controls are selected for enrichment in both patient and HSPC; recurrence uses a one-bin tolerance; stripe controls are shifted rectangles; the donut maximum over 3x3 pixels favours the sharper measured map slightly (exact-pixel values are in `object_scores.csv.gz`, columns `#px`).

### Files
`scripts/aml_loop_detection.py`; `results/aml_loop_detection/`: `items.csv.gz` (objects, windows, recurrence), `object_scores.csv.gz`, `levels_by_class.csv`, `increments.csv`, `gain_correlation.csv`, `auroc.csv`, `auroc_by_pair_type.csv`, `per_patient.csv`, `class_sizes.csv`, `levels_by_train_recurrence.csv`, `aml_loop_detection.png`. Predictions under `/mnt/processed/wolffjoa/hic-multiome/runs/aml_loop_detection/`.

## 63. Loop-centric training of the AML-inclusive model: can it predict AML-specific loops and stripes of a new patient? (PENDING; section 62 was the last number when this was written)

Question: section 62 shows that the AML-inclusive model predicts a weak gain at AML-specific loops (median +0.06 against +0.40 log2 measured, Spearman 0.28, AUROC 0.62 against matched shared loops, private loops 0.59), carried by the AML reference. The pixel loss is dominated by background; AML-specific loops (about 230 per patient, 7 training patients per fold) and stripes are rare, small-amplitude features, and the patient's own marks add little under that loss (sections 47, 57, 61, 62). The residual-reference model alone (section 45) did not change this. This section adds a loop-centric recipe and tests it on the same ten held-out patients.

### Implementation (status: implemented, CPU-tested, no GPU run yet)
Run family `all12aml_loopfocus_refres_drop_f<N>` (N = 10, 11, 12) at 10 kb and 25 kb: the residual-reference AML model (`all12aml_refres_drop`: model.reference residual, so the network output is the patient-specific deviation from the AML reference; same cache `mc_all12aml_ref_<k>kb_f<N>`, AML x3 oversampling, patient-hard validation) plus `train.aml_object_*` (`hicmultiome/aml_objects.py`, hooks in `hicmultiome/train.py`, all defaults off, existing runs unchanged):

| flag | value | role |
|---|---|---|
| `data.aml_objects_table` | share `aml/objects/objects_f<N>.csv.gz` | objects of the fold's training patients |
| `train.aml_object_oversample` | 5 | windows holding an AML-specific object and an equal number of one-to-one matched control windows (same patient and object kind; first choice the classifier's matched control of the object, else an unused control window of the patient) enter every epoch 5 times, on top of the AML x3 |
| `train.aml_object_controls` | `[A_hspc]` | control set (shared loops / stripes matched on distance, orientation, HSPC CTCF and H3K27ac) |
| `train.aml_object_pair_batches` | true | batches of half AML-specific and half control windows, so each batch holds ranking pairs |
| `train.aml_object_weight` | 1.0 | extra loss on the donut neighbourhood of loops (box of outer+1 bins) and the stripe rectangle with one bin of flank, both objects and controls: L1 between predicted and measured residual (prediction minus reference, target minus reference) plus L1 between the object-level enrichments (loop: maximum of the donut enrichment over the 3 x 3 pixels around the annotated pixel, as the evaluation; stripe: mean over the rectangle) |
| `train.aml_object_rank` | 0.5 | pairwise logistic ranking softplus(-(s_AML - s_control)/0.05) over every AML-specific and control object of the same kind in the batch, scores on the residual; `aml_object_rank_consistent` keeps pairs whose measured scores are ordered the same way; `aml_object_rank_space` residual or full |
| `train.aml_object_donut` | [1, 4] at 10 kb, [1, 3] at 25 kb | enrichment ring (inner, outer half-width in bins; the donut of `hicmultiome/losses.py` with the evaluation's 10 kb setting) |

Reverse-complement augmentation flips the object coordinates with the window (tested against the tensor flip). The loss terms are logged per epoch (`obj_pix`, `obj_enr`, `obj_rank`, `obj_n`). The loss keys are `train.*` because `train.loss` is one spec string.

Objects (`scripts/aml_objects_table.py`, `results/aml_objects/summary.csv`): from the loop classifier table (Suppl. Table 1 loops, Suppl. Table 4 stripes and their matched controls, 10 kb, span up to 150 bins) restricted to the training patients of the fold. Fold 10 (7 patients): 1591 AML-specific loops, 729 stripes, 1510 / 647 A_hspc controls; fold 11 (6 patients): 1320 / 700, 1296 / 603; fold 12 (7 patients): 1641 / 721, 1562 / 628. The column `n_train_other` counts the other training donors carrying the object (private among the training patients: 1438 of 1591 loops, 375 of 729 stripes in fold 10).

On the real fold 10 10 kb cache the recipe finds 1114 training windows with AML-specific objects and 922 with controls only (1891 of 4477 objects lie wholly inside a cached training window with margin: validation and training-region filters), 934 matched pairs; oversampling x5 adds 7472 windows to the 40.5k of the refin run (+18 percent).

### Leakage safeguards
- A table holds only the training patients of its fold (`aml_train.train_patients`), after removing every patient of the same donor as a test patient (`aml_train.SAME_DONOR`, `donor_group`); `assert_no_leak` runs before the file is written, `tests/test_aml_objects.py` repeats it on the written tables of folds 10, 11, 12 and on a synthetic table that includes test patients and the partners 496 and 424.
- `n_train_other` counts training donors only; no test patient, partner or non-training sample enters the table.
- The trainer rejects a table with a cell that is not a training sample of the run (`load_objects`).
- Oversampling and loss terms act on the rows of the training split `ti`, selected after the honest validation split (`val_exclude_overlap`: no training window overlaps a validation position in any sample), so no validation window is a training window; HSPC donors (050, 213 train; 341, 822 held out) and MOLM-13 are as in the other AML runs, U-937 and the test patients are never trained on.
- Recurrent test-patient loops that a training patient also carries are learnable from that training patient (as from the reference); the private criterion (objects of the test patient seen in no other donor) is not reachable that way.

### Tests
`tests/test_aml_objects.py` (22): table leakage (synthetic and written tables), foreign-cell rejection, window membership with margin, one-to-one matched pairs, batch composition, flip equivalence, loss values and gradient direction, ranking direction, consistent-pair filtering, config defaults and validation, run configs of folds 10, 11, 12 at 10 and 25 kb (identical to the refres run except the recipe keys), and a 60-step CPU training smoke with all flags (synthetic genome; the three object terms appear in the log). Full suite 209 passed, 1 skipped. Configs: `results/aml_loopfocus/configs/` (`scripts/aml_loopfocus_configs.py`).

### GPU job and evaluation
`results/aml_loopfocus/REMOTE_JOB.md`: six runs (3 folds x 10 / 25 kb), claims by run name, evaluation by `aml_train_eval.py --fold N --variant extra` (model label `aml_loopfocus`) and `aml_loop_detection.py --recipe loopfocus` (tables in `results/aml_loop_detection_loopfocus/`, stage `criteria`). Estimated remote 4090 time: about 5 h for the three 10 kb runs (range 4 to 7.5 h; refin_drop took 1.0 to 1.7 h each), about 2.5 h for the three 25 kb runs, about 1 h for the evaluations; about 6.5 h at 10 kb alone, about 9.5 h in total.

**Superseded (section 69, 70):** the pilot passed criterion 1 through the HSPC-level shortcut; this section's recipe is replaced by `all12aml_loopfocus2_*` (section 70). The criteria below are kept as registered.

### Success criteria (10 kb, ten held-out patients, donor bootstrap)
1. AUROC of the gain over HSPC, private loops against matched shared loops (A_hspc): >= 0.70 (section 62: 0.59).
2. Amplitude (median model gain / median measured gain), private loops: >= 0.30 (section 62: 0.11).
3. Pooled Pearson on the Akita test regions: no loss against the aml_refin models (0.813).
Results: not yet available.

### Risks
The ranking and enrichment terms push enrichment at annotated positions up; a model that raises enrichment at every annotated-looking site would also raise the false-positive rate (the matched shared controls and the ranking term are the counterweight; B_rand and A_do controls in `criteria`/`auroc.csv` test this). Only 42 percent of the table objects lie in cached training windows. Weights (1.0, 0.5, temperature 0.05, x5) were not tuned; one seed per fold. The 25 kb variant uses the 10 kb object coordinates mapped to 25 kb bins and a narrower ring, and its loops are only 2 to 60 bins long. The hard-split JSONs of folds 10 to 12 exist only in the remote repo copy.

### Files
`hicmultiome/aml_objects.py`, `scripts/aml_objects_table.py`, `scripts/aml_loopfocus_configs.py`, edits in `hicmultiome/train.py`, `hicmultiome/config.py`, `scripts/multicell_expanded.py`, `scripts/aml_train_eval.py`, `scripts/aml_loop_detection.py`; `tests/test_aml_objects.py`; `results/aml_objects/summary.csv`, `results/aml_loopfocus/REMOTE_JOB.md`, `results/aml_loopfocus/configs/`.


## 64. Which marks and methylation features go with AML-specific loops and stripes, and can they be boosted? (FINAL; section 63 was the last number when this was written)

Script `scripts/aml_loop_mark_correlations.py` (stages inventory, tracks, features, corr, boost, maps, figures; hicdiff env, CPU only), results in `results/aml_loop_marks/`. Objects: the 2276 AML-specific loops and 1075 stripes of the ten held-out donors of section 62 (168/496 and 260/424 count once; healthy donors excluded) with their matched shared controls; target = measured gain over the HSPC consensus (`gain|measured`). CUT&Tag CTCF and WGBS exist for 8 of the 10 donors (1360 and 1916 have H3K27ac and H3K27me3 only).

### Data (inventory.csv, inventory.md, inventory_data_types.md)
Everything held locally is processed; no raw reads exist for any assay. Hi-C: nodups pairs for 32 Xu samples (25 AML, 3 healthy PBMC, 4 HSPC; 169 GB) plus iced mcools (10 kb and coarser, 25 GB), 0.02 to 0.49 B contacts at 5 kb per sample. CUT&Tag: 33 MACS2 -log10 p-value bigWigs (12 H3K27ac, 12 H3K27me3, 9 CTCF incl. healthy 103), one sample per mark and donor; GEO also lists MACS2 peak files, not downloaded (command line not in the files we hold). WGBS: 20 `.CGN.10bp.bw` (fraction per 10 bp bin, no read counts or coverage; 10 with CUT&Tag, 10 WGBS only; healthy 103 and 673, no 576, no CD34+ WGBS; CD14+ monocyte ENCODE as the other healthy reference). SV (595 junctions, 22 samples), CNV (54 segments, 18 samples), fusion / mutation labels (hand-read from Xu Fig. 1c), Suppl. Tables 1 and 4 (loops, stripes). Fischer 2024: 160 ChIP bigWigs (CTCF, RAD21, STAG2, H3K27ac; 45 patient folders) and 3 pooled Hi-C maps; cell lines MOLM-13 and U-937 (Hi-C + ChIP), K562. Used now: Hi-C targets and references, CUT&Tag as quantile-matched model inputs, WGBS as 1 kb `Meth` channel, SV table in the SV-aware runs, Fischer at group level. Not used before this section: MACS2 peaks (not downloaded), WGBS at 10 bp, Fischer per-patient RAD21 / STAG2, the 7 extra CD34+ ChIP experiments fetched here (3 H3K27ac, 3 H3K27me3, 1 CTCF; `work/loopmarks/hspc_extra`), cell-line loops. ATAC (33 files) and RNA (22 files) were downloaded by the private-loop agent (`results/aml_private_loop_info/`, in progress; its first 8-donor logistic AUROC increments of ATAC / RNA features over marks are about -0.002 [-0.005, 0.000], work not finished) and are not analysed here.

### Correlations (corr; correlations_all_features.csv.gz, auroc_all_features.csv.gz, top_features.md, expected_relationships.csv, fig_mark_heatmap.png)
1093 features: mark (H3K27ac, H3K27me3, CTCF) x representation (own, HSPC, patient minus HSPC; also z-scored / percentile difference, minus healthy 103/673, minus other AML donors, HSPC mean) x position (anchor A, B, min, max, sum, product, promoter, partner, interior; stripes anchor, extent mean/max) x scale (bin, +-20 kb, +-100 kb); WGBS mean, minimum, low-methylated fraction (1 kb bins < 0.5) with own / healthy / CD14+ / minus-reference; CpG islands; methylation and CTCF signal at the CTCF motif hits (JASPAR MA0139.2 >= 70 percent, anchor +-1 bin; methylation from the 10 bp bigWigs; data for about one third of the anchors). Pooled Spearman with donor bootstrap (1000), per-donor Spearman, partial Spearman controlling log distance and the HSPC enrichment of the object, AUROC against matched shared controls (pair-type classes against controls of the same own-mark type). Family-wise null (gain permuted within donor, max |rho| over about 475 features, 95th percentile): 0.12 (all loops), 0.15 (private loops), 0.17 (all stripes), 0.28 (enhancer-like stripes).
- Stripes: the one clear signal. Patient-minus-HSPC H3K27ac at the anchor (minimum over anchor and extent, +-100 kb) rho 0.30 [0.23, 0.35] all stripes, 0.45 [0.36, 0.51] enhancer-like (10/10 donors), 0.36 [0.27, 0.45] private, 0.26 recurrent; partial identical. HSPC H3K27ac and H3K27me3 correlate negatively (-0.23 to -0.36; low HSPC state, large gain).
- Loops: no patient-specific mark correlates with the gain magnitude beyond the family-wise null of 0.12 to 0.15. The strongest are locus properties: healthy WGBS mean at +-100 kb rho -0.26 [-0.30, -0.21] private loops (all 8 donors negative; patient-minus-healthy -0.1 to -0.2 only for some positions), own H3K27ac +-100 kb -0.22 [-0.30, -0.14] (10/10 negative): the more active / hypomethylated the region, the smaller the gain over HSPC. Recurrent loops: own CTCF and own H3K27ac negative (-0.28 to -0.32).
- Expected relationships. Delta H3K27ac at both anchors for promoter-enhancer loops: rho with the gain -0.04 [-0.19, 0.07] (not supported for magnitude); as a separator against own-mark-matched PE controls AUROC 0.69 [0.64, 0.73] (min over anchors), 0.73 [0.69, 0.77] (max), 10/10 donors (supported); own H3K27ac min 0.76. H3K27me3 for promoter-silencer loops: gain rho -0.05 to -0.16 (CI includes 0, n=103), AUROC of delta H3K27me3 0.63 to 0.74 [0.70, 0.80] (4 donors evaluable; supported, small n). Repressive stripes: delta H3K27me3 at the anchor rho 0.01 [-0.08, 0.09] (not supported for magnitude), AUROC 0.74 [0.70, 0.76] (9/9 donors; own H3K27me3 0.81); enhancer-like stripes delta H3K27ac rho 0.27 [0.16, 0.35] and AUROC 0.67 (supported). CTCF: delta and own CTCF rho -0.05 to 0.02 (CI around 0) and AUROC 0.45 to 0.53 for loops and stripes, all classes (not supported); HSPC CTCF +-20 kb separates AML-specific loops from shared controls (AUROC 0.24, i.e. controls are CTCF-richer) but that is HSPC state. Methylation at CTCF motifs: rho -0.03 [-0.09, 0.04] (bound hit), patient-minus-healthy -0.10 [-0.19, 0.01]; recurrent loops +0.34 [0.22, 0.48] (n=43, 0 evaluable donors); AUROC of bound-motif methylation 0.58 [0.56, 0.62] for loops (AML-specific loops sit at less-methylated motifs than controls, partly the HSPC matching), none for stripes; patient-minus-healthy AUROC 0.51 (not supported). Patient-minus-CD14 WGBS mean AUROC 0.56 [0.54, 0.58] for loops.

### Boosting (boost; boost_metrics.csv, boost_increments.csv, boost_increments.md, boost_oof_*.csv.gz, fig_boost_increments.png)
Leave-one-donor-out, locus-disjoint (classifier folds), histogram gradient boosting as in the loop classifier (BASE reproduces its AUROC 0.837); AUROC against matched controls and Spearman of the predicted gain on positives; bootstrap over donors of paired differences. Private loops, increment over BASE (AUROC 0.850 pooled): all engineered sets together (ALL, 685 features) +0.013 pooled [-0.011, 0.033], +0.014 per donor [0.007, 0.021]; multiscale alone +0.005 [-0.008, 0.017]; pairwise products / convergent-CTCF interactions -0.001 [-0.003, 0.001]; per-patient z / percentile differences +0.002; HSPC mean of 4 experiments +0.004 [-0.011, 0.016]; healthy-donor or other-AML-donor reference -0.001, -0.002; extended methylation / CpG / motif sets +0.002; monotone constraints +0.008; per-donor feature centring +0.020 [-0.010, 0.044]. Gain Spearman, private loops (BASE 0.197): multiscale +0.045 [0.013, 0.069] (per donor +0.051 [0.028, 0.076]), ALL +0.038 [-0.017, 0.094], centring +0.068 [-0.040, 0.180] (per donor +0.064 [0.012, 0.115]). Patient data over locus-only (HSPC + sequence covariates, AUROC 0.834): BASE +0.016 [0.001, 0.029], ALL +0.029 [-0.008, 0.058]. Private stripes: no increment is reliable (AUROC BASE 0.588, locus-only 0.605, ALL 0.628, ALL over BASE +0.040 [-0.001, 0.090]; gain Spearman +0.01 to +0.08, all intervals include 0); the logistic model on ALL +0.062 [0.018, 0.108] is one of 19 variants and not corrected for that. Of the 19 variants x 2 objects x 2 tasks, few intervals exclude 0.
2D maps (maps; maps2d_summary.csv, fig_maps2d.png): outer-product maps of the 1D deltas (H3K27ac, H3K27me3, CTCF, WGBS minus healthy, own H3K27ac, H3K27ac x H3K27me3) in the 1.92 Mb windows of all 3351 objects, diagonal-detrended pixel Spearman with measured minus AML reference, measured minus HSPC and the model residual (measured minus aml_refin): local rho (loop +-15 bins, stripe rectangle +-5) between -0.03 and +0.09 for every channel (model residual +0.001 to +0.021; delta H3K27ac outer product +0.018 loops, +0.007 stripes; additive and minimum channels the same size), partial correlation given the row and column marginals 0.000 to 0.008. CTCF channels 0.00 / negative.

### Recommendation
Add to the U-Net inputs (and to a loop-level re-scorer): (1) multiscale patient-minus-HSPC H3K27ac, smoothed at +-20 kb and +-100 kb (expected: the stripe signal, rho 0.30 to 0.45, is a +-100 kb delta; +0.045 gain Spearman for private loops), (2) per-patient normalisation of the difference channels (genome-wide z or percentile of the patient track before differencing; +0.02 to +0.03), (3) healthy-donor / HSPC-mean H3K27ac and methylation as locus context at +-100 kb (regional activity predicts a smaller gain, rho -0.2 to -0.26); keep the HSPC mean of several experiments as reference instead of one experiment (no measurable gain, cheap). Not worth it: outer-product / pairwise channels (maps: partial rho <= 0.008, tree interactions -0.001), convergent-CTCF x delta CTCF (CTCF carries no gain signal at 10 kb), motif-level methylation channels (data for a third of the anchors, no signal), patient-minus-other-AML reference (no gain), additional CpG-island / low-methylation fraction channels (+0.002). The ceiling is low: the best private-loop AUROC is 0.86 to 0.87 against matched shared loops, of which 0.83 is HSPC / sequence state; the patient's own marks add 0.016 and all engineering a further 0.01 to 0.015.

### Caveats
Ten donors (eight for CTCF and WGBS, 168 and 496 counted once), one HSPC experiment per mark in all reference-dependent analyses, three marks only, one CUT&Tag sample per donor (depth differs 4-fold across donors; the quantile matching removes scale, not noise). Pair-type classes are defined from the patient's own marks, which restricts the range of own-mark features inside the class (hence AUROC against same-type controls). Wide-window features are regional chromatin-state proxies. Many features (family-wise nulls above) and 19 boosting variants; intervals are not corrected. Methylation at motifs: data for about one third of the anchors (CpG-poor 19 bp windows have no 10 bp bin).

### Files
`scripts/aml_loop_mark_correlations.py`; `results/aml_loop_marks/`: `inventory.csv`, `inventory.md`, `inventory_data_types.{csv,md}`, `correlations_all_features.csv.gz`, `auroc_all_features.csv.gz`, `top_features_{correlation,auroc}.csv`, `top_features.md`, `expected_relationships.csv`, `correlation_familywise_null.csv`, `boost_metrics.csv`, `boost_increments.{csv,md}`, `boost_oof_{loop,stripe}.csv.gz`, `maps2d_summary.csv`, `fig_mark_heatmap.png`, `fig_boost_increments.png`, `fig_maps2d.png`. Caches (feature matrix, tracks) under `/mnt/processed/wolffjoa/hic-multiome/aml/primary/work/loopmarks/`.

## 65. Does patient ATAC-seq and RNA-seq predict private AML-specific loops and stripes? (FINAL; section 64 was the last number when this was written)

Question: sections 47, 57, 61 and 62 leave private loops (79 percent of AML-specific loops) unpredicted. Is there data we can obtain that carries information beyond the HSPC state of the locus, the patient's own CUT&Tag marks and WGBS?

### Inventory (Xu et al. 2022, superseries GSE152136; PBMC donors 103, 673, 576 are healthy)
Sub-series: GSE152096 `[RNA-Seq]`, GSE152099 `[WGBS]`, GSE152132 `[CUT&TAG]`, GSE152134 `[ATAC-Seq]`, GSE152135 `[Hi-C]`. Sample titles are of the form `018_ATAC-seq`, `018_RNA-seq`, `018_WGBS`, `018_CUT&TAG_H3K27ac`, `018_Hi-C` (table in `results/aml_private_loop_info/geo_samples.csv`, per-patient overlap in `inventory_per_patient.csv`).
- ATAC-seq (GSE152134, 16 samples, 2.5 GB): per sample `GSM46042xx_<id>_ATAC-seq.10bp.RPGC.bigwig` (median 152 MB) and `.MACS2.filt.peaks.txt.gz` (0.8 MB). Patients 018, 027, 168, 260, 270, 472, 546, 566, 629, 773, 990, 1360, 1916; PBMC 103, 576, 673. No ATAC for 1071, 1324, 424, 496, 504, 798, 838, 933.
- RNA-seq (GSE152096, 22 samples, 72 MB): `GSM..._<id>_RNA-seq.RSEM.quant.genes.txt.gz` (RSEM counts, TPM, FPKM, GENCODE v24, hg38). Patients 018, 027, 168, 260, 270, 472, 546, 566, 629, 773, 868, 1071, 1324, 424, 496, 504, 798, 838, 933; PBMC 103, 576, 673. No RNA for 1360, 1916, 990, 1021, 936, 1953. Raw reads are withheld by the submitters (patient privacy).
- No ATAC or RNA for the four HSPC donors (HSPC-050, -213, -341, -822; Hi-C only). No H3K4me1, H3K4me3 or other marks in the cohort (CUT&Tag: H3K27ac 12, H3K27me3 12, CTCF 9 samples).
- Overlap with the loop classifier (20 non-healthy samples): all 8 patients with CTCF + K27ac + K27me3 + WGBS have ATAC and RNA (018, 027, 168, 270, 472, 546, 629, 773); 1360 and 1916 have ATAC and K27ac/K27me3 only; 260 and 566 have ATAC, RNA, WGBS; 1071, 1324, 424, 496, 504, 798, 838, 933 have RNA and WGBS only.
- Other AML sources (`aml_data_survey.md`): Hi-C plus marks exist for THP-1, MOLM-13, HL-60, U-937, OCI-AML3, MV4-11, NB-4 (cell lines; each one genome, so at most one more "patient" each, no private-versus-recurrent split); Kasumi-1 has no genome-wide Hi-C; Fischer et al. 2024 (GSE268035) has per-patient CTCF/RAD21/STAG2/H3K27ac ChIP for 7 patients but pooled Hi-C only. None adds per-patient ATAC or RNA with Hi-C.

### Downloaded (`/mnt/processed/wolffjoa/hic-multiome/aml/primary/{atac,rna}`, 3.5 GB, `scripts/aml_private_loop_download.sh`)
Xu ATAC bigWigs and peaks, Xu RSEM gene files, GENCODE v24 GTF; HSPC reference: five healthy CD34+ ATAC experiments (ChIP-Atlas hg38 bigWig SRX19490150-52, SRX10942129-30) and ENCODE ENCSR830HIN (CD34+ common myeloid progenitor, polyA RNA, ENCFF690QPA, GENCODE V24; one sample). Pearson of 10 kb bins between HSPC replicates 0.61 to 0.80; patient against mean of other patients 0.71 to 0.90, patient against HSPC 0.24 to 0.66 (`atac_sample_corr.csv`), so patient-minus-HSPC features carry cross-laboratory batch. RNA Spearman patient-patient 0.73, patient-HSPC 0.75.

### Design
`scripts/aml_private_loop_info.py`. ATAC: 10 kb bin mean and max of log2(1 + RPGC), quantile-matched to the mean AML profile. RNA: log2(TPM + 1), quantile-matched; genes (protein coding and lincRNA, 27,346) placed on refGene TSS by symbol (80 percent of TSS), GENCODE TSS otherwise. New groups, each per pair as min and max over the two regions: atacown, atachspc, atacdiff (mean, max, +-20 kb flank), rnaown, rnahspc, rnadiff (promoter-anchor expression, genes within +-20 kb, strongest up and down gene against HSPC), acxatac (product of genome-wide percentiles of H3K27ac and ATAC; patient and patient minus HSPC). Rows, controls (A_hspc matched shared loops or stripes, B_rand), locus-disjoint leave-one-donor-out folds, donor units ({168, 496}, {260, 424}) and models (logistic regression, gradient boosting) are those of `results/aml_loop_classifier_v2`. Private = no other donor carries the object (one-bin tolerance, Suppl. Tables 1 and 4): 3684 of 4520 loops (81 percent), 405 of 1394 sampled stripes (29 percent). Datasets: D1, 8 patients with all data, baseline HSPC state + own marks + WGBS + motif + gene + distance; D2, 10 patients with marks and ATAC (no WGBS in 1360, 1916), baseline without methylation; D3, 16 donors with RNA and WGBS, no own marks, baseline HSPC + WGBS; D4, 10 patients with ATAC, RNA and WGBS, no own marks. Increments: paired donor bootstrap (500) of macro AUROC.

### Results (gradient boosting; logistic regression is the same within 0.01)
| set | target vs controls | donors | baseline pooled AUROC [95%] | + all new groups (macro increment [95%]) | best single group |
|---|---|---|---|---|---|
| D1 loops | private vs A_hspc | 8 | 0.856 [0.824, 0.886] | -0.000 [-0.006, 0.004] | rnadiff +0.001 |
| D1 loops | private vs B_rand | 8 | 0.772 [0.723, 0.823] | -0.002 [-0.011, 0.008] | atacown +0.004 |
| D1 loops | all vs A_hspc | 8 | 0.838 [0.804, 0.868] | -0.001 [-0.007, 0.004] | rnaown +0.001 |
| D1 loops | non-private vs A_hspc | 8 | 0.743 [0.713, 0.778] | +0.001 [-0.017, 0.017] | rnaown +0.006 |
| D2 loops | private vs A_hspc | 10 | 0.850 [0.822, 0.877] | -0.001 [-0.004, 0.002] | atachspc +0.001 |
| D3 loops | private vs A_hspc | 16 | 0.831 [0.810, 0.852] | +0.002 [0.000, 0.004] | rna_all +0.002 |
| D3 loops | private vs B_rand | 16 | 0.678 [0.638, 0.709] | +0.009 [0.000, 0.018] | rnadiff +0.008 |
| D4 loops | private vs A_hspc | 10 | 0.837 [0.809, 0.863] | +0.006 [0.001, 0.011] | rnadiff +0.001 |
| D4 loops | private vs B_rand | 10 | 0.687 [0.642, 0.731] | +0.010 [-0.004, 0.024] | rna_all +0.006 |
| D1 stripes | private vs A_hspc | 8 | 0.602 [0.544, 0.654] | +0.002 [-0.034, 0.035] | acxatac +0.012 |
| D1 stripes | private vs B_rand | 8 | 0.678 [0.649, 0.712] | +0.017 [-0.009, 0.040] | acxatac +0.022 |
| D1 stripes | all vs A_hspc | 8 | 0.600 [0.578, 0.628] | -0.003 [-0.022, 0.014] | acxatac +0.004 |

All 48 dataset x target x negative x model rows are in `headline_table.csv`, the 12 feature sets per row in `increments.csv` and `summary.csv` (AUROC and AUPRC with intervals). Largest point estimate of any increment 0.026 (logistic regression, non-private loops, D1), largest upper interval limit 0.056 in that one row; for private loops the largest point estimate is 0.014 (D4 vs B_rand, logistic regression). By pair type (D1, `by_type.csv`): promoter-enhancer loops baseline 0.88 and with new groups 0.88 (private, gradient boosting), promoter-silencer loops 0.77 against 0.76; stripes enhancer-type +0.04 [0.00, 0.08] and repressive +0.01 on 91 and 101 private stripes. The new groups alone (distance + ATAC + RNA, no HSPC marks or own marks) reach pooled AUROC 0.72 for private loops against A_hspc, so they carry the signal but all of it is contained in the baseline. Strongest single features (mean donor AUROC, private vs A_hspc): H3K27ac x ATAC interaction patient-minus-HSPC 0.60, H3K27ac itself 0.62, down-regulation of nearby genes 0.58, patient-minus-HSPC ATAC 0.56 to 0.57; HSPC ATAC at the anchor goes the other way (0.35 to 0.43; AML-specific anchors are closed in HSPC).

### Gate verdict
Not passed. ATAC and RNA raise the private-loop AUROC by at most +0.006 (point estimates -0.002 to +0.014 over all configurations), against the threshold of +0.05; the intervals exclude +0.03 everywhere for loops. The baseline (HSPC state, own marks, WGBS) already separates private loops from matched shared loops at 0.83 to 0.86 AUROC (random pairs 0.69 to 0.77), and ATAC, RNA and the H3K27ac x ATAC interaction are redundant with it: accessibility and expression of the patient say where regulatory elements are, which H3K27ac and the HSPC state already say. Private loops are no better and no worse separated than recurrent ones (0.86 against 0.74 in D1), so the earlier 0.59 of the Hi-C model (section 62) is a limit of the predictor on private loops, not of the information in these inputs: a classifier on tabular features reaches 0.85 with matched controls. What the data cannot show is which of the many accessible, H3K27ac-marked anchor pairs a given patient actually joins by a loop; no feature of one anchor carries the partner choice, and the classifier increments for pair-level features (HSPC contact, motif orientation, distance) are held constant by the matching.

### Further data and availability
- H3K4me1, H3K4me3, H3K27ac for the RNA-only and ATAC-only patients: not released for Xu; the cohort has no H3K4me marks. Would need new CUT&Tag on patient material.
- CTCF for 496-like patients (WGBS and RNA only; 1071, 1324, 424, 496, 504, 798, 838, 933) and for 1360, 1916: not released; CTCF is available for 9 samples only. Fischer et al. 2024 releases CTCF, RAD21, STAG2, H3K27ac per patient (7 WT patients) with pooled Hi-C (GSE268031/GSE267780), usable for marks but not for private-loop labels.
- Mutation subtype and SV: subtype inferred from Suppl. Table 3 is in `aml_subtype`; SVs in Suppl. Table 5/6. Raw WGS is withheld.
- TF expression: contained in the RNA used here (promoter and nearby genes only); a patient-level TF-expression vector (RUNX1, CEBPA, GATA2, MECOM, cohesin) was not tested and has 18 patients as the sample size, too few for a locus-level model.
- Single-cell multiome or HiChIP on the same patient: GSE302267 (scHi-C of AML bone marrow HSPCs, no marks); no public per-patient H3K27ac HiChIP or capture Hi-C in AML cohorts was found.
Recommendation: private AML-specific loops cannot be pushed by more input tracks of this kind; claims for new patients should be made for recurrent loops and stripes (reference-carried, section 62), with private loops stated as not predicted. A route that could add information is deeper own-patient 3D data (capture Hi-C or HiChIP of the same patient), which is a measurement of the target, not an input.

Caveats. Eight to sixteen donors; one HSPC RNA sample from another laboratory (CD34+ CMP) and ATAC of healthy donors from other laboratories, so diff groups carry batch; private is defined by absence from the other donors' call sets (detection limit, not proof of absence); stripes are subsampled to 150 per patient and only 29 percent are private; no seed variation measured.

### Files
`scripts/aml_private_loop_info.py`, `scripts/aml_private_loop_download.sh`; `results/aml_private_loop_info/`: `headline_table.csv`, `summary.csv`, `increments.csv`, `by_type.csv`, `univariate.csv`, `private_counts.csv`, `inventory_per_patient.csv`, `inventory_files.csv`, `geo_samples.csv`, `atac_sample_corr.csv`, `atac_hspc_replicate_corr.csv`, `rna_sample_spearman.csv`, `oof_predictions.csv.gz`, `fig_private_loop_info.png`. Caches under `/mnt/processed/wolffjoa/hic-multiome/aml/primary/work/privinfo/`.


## 66. One rule for all cell types: differential mark channels and cell-specific objects of every training cell (PENDING; section 65 was the last number when this was written)

Premise: "this chromatin-mark / methylation pattern leads to a loop or a stripe" is cell-type independent biology and AML is only where it shows most clearly. The AML-specific recipe of section 63 gives the model the patient's marks and the AML reference and rewards AML objects only; whatever it learns can be an AML-specific rule. This section implements one rule for the 12 training cell lines, the 5 held-out cells and the AML patients: the network gets the reference's marks and the sample-minus-reference difference of the marks beside the reference Hi-C it already reads, so that residual Hi-C ~ f(delta marks) with one shared f, and the loop-centric sampling and loss act on the cell-specific loops and stripes of every training cell.

### Implementation (status: implemented and CPU-tested; no window cache built, no GPU run)
Differential mark channels (`hicmultiome/diffmarks.py`; hooks in `precompute.py`, `train.py`, `predict.py`, `scripts/crosscell_predict.py`; all off by default, existing caches, run configs and 60-step training logs identical to the previous commit):

| flag | default | role |
|---|---|---|
| `data.diff_marks` | false | per mark (all tracks but annotation and copy number: 13 ChIP marks, Meth in the `diffmeth` runs) the reference mark `dref:<mark>` and the difference `diff:<mark>:s<r>` rows, appended after the other sequence-feature rows |
| `data.diff_scales` | [0, 20000, 100000] | radii in bp: the 10 kb bin, +-20 kb, +-100 kb box smoothing (25 kb: 0, +-1, +-4 bins) |
| `data.diff_reference` | consensus | reference of the AML samples: `mean_aml` (training patients of the fold, leave-one-out), `hspc` (HSPC training donors), `consensus` (the 12 cell lines) |
| `data.diff_reference_sets`, `samples[].diff_set`, `data.diff_default_set` | empty | named pools of sample names; a sample reads its pool without itself and without any sample of its cell identity |
| `data.diff_ref_channels`, `data.diff_tracks` | true, all | drop the `dref` rows; restrict the marks |
| `train.object_scope`, `data.cell_objects_table` | aml, none | `all`: the objects of `data.aml_objects_table` and of the cell-line table feed the section 63 oversampling, pair batches, neighbourhood loss and ranking |

Channel definition, no Hi-C read: per sample and mark z = (log1p(raw) - q50) / (q99 - q50) over the autosomal bins (per-sample quantile anchoring, so a deep and a shallow track sit on one scale; the model's own log1p / global z-score is a different, shared normalisation); reference = mean of z over the pool members that have the mark; difference = z(S) - reference, box-smoothed per scale, standardised per sample, mark and scale ((d - median) / (1.4826 MAD), clipped to +-8). The rows bypass the track normaliser (`TrackNormalizer.passthrough`, absent from old files). A mark the sample lacks, or that no pool member has, gives 0 in all its rows ("no information", the cached absent value); track dropout of a mark zeros its rows (`apply_diff_drop`, in the same draw as the ChIP row), inference keep sets and imputation do the same (`blank_diff_rows`); `train.select_tracks` drops the rows of the marks it leaves out (the select_tracks lesson of section 27: the rows must reach the network, regression test below). Reference sets of the generator (`scripts/multicell_expanded.py`): cell lines the 12 cells (leave-one-out = the leave-one-out consensus of the Hi-C reference; a held-out cell reads all 12 = consensus12), AML patients and MOLM-13 the fold's training patients (= the AML reference), HSPC donors the other training donor. Unseen cells at inference (`crosscell_predict --diff-group cell|aml|hspc`, `predict --diff-group`) read the pool of their group and leave their own identity out.

Objects of every training cell (`scripts/cell_objects_table.py`, table columns of section 63, `n_train_other` = other training cells carrying the object): the cell-agnostic definitions of `scripts/crosscell_rule_transfer.py`, written in parallel, which builds the cell-specific loops and stripes of all 15 cells with the operational definitions of the AML objects (loops: local maxima of the log2 donut enrichment, 1 to 4 bins, 10 kb, 6 to 150 bins, both anchors on peak bins; specific = enrichment >= 0.5 and gain over the reference >= 0.30; stripes: 4 anchor bins x 40 extent bins at an offset of 50 bins, specific = score >= 0.55 and gain >= 0.20; shared controls `A_ref` matched on distance, orientation and the reference's CTCF and H3K27ac). `from_rule_transfer` converts its `$HM/crosscell_rules/cell_objects.csv.gz` (sets pos and A_ref) into the table: the 12 training cells to the training table, the strongest 400 loops and 200 stripes per cell that have a matched control (the AML patients have about 230 private loops each; the cap keeps the oversampled window count at the section 63 level), K562, HMEC (deep) and HUVEC (deep) with consensus12 to a separate evaluation table (1000 loops and 1000 stripes per cell, seeded draw) that the trainer rejects. The reference of a cell is its leave-one-out consensus. The script also has its own Hi-C calls (`--source local`: donut peaks, a 4 x 20-bin stripe rectangle next to the anchor, controls matched on distance and the cell's CTCF and H3K27ac) as a self-contained fallback; `--source auto` takes the rule-transfer file when it exists. The generated configs read the same output path either way. Tables written (`results/cell_objects/`, copies on the share `cell_objects/`, source file next to each): training table 12 cells x (400 loops + 200 stripes), 4800 / 2400 specific loops / stripes with 4070 / 2400 matched controls, private among the other training cells 77 percent of the loops and 47 percent of the stripes; evaluation table K562, HMEC (deep), HUVEC (deep): 3000 loops and 2563 stripes with 628 / 2211 controls. The local fallback gave 4800 / 2251 specific objects on the same cells.

Run families (`scripts/multicell_expanded.py`; the joint composition of `all12aml_*` is kept: 12 cells + the fold's training patients + HSPC donors 050, 213 + MOLM-13, AML x3, lab row, patient-hard validation, residual reference, track dropout): `all12diff_loopfocus_refres_drop_f10..12` at 10 and 25 kb (PRIMARY: channels + all-cell objects, caches `mc_all12aml_ref_<k>kb_f<N>_diff`), ablations `all12diff_refres_drop_f<N>` (channels only), `all12allobj_refres_drop_f<N>` (objects only, existing AML cache), `all12diffmeth_loopfocus_refres_drop_f<N>` (+ Meth), `all12diffhspc_loopfocus_refres_drop_f<N>` (AML samples read the HSPC donors), and the pilot on the 12 cell lines alone `all12diffcells[_loopfocus]_refres_drop_{10,25}kb`. Configs: `results/diff_marks/configs/` (`scripts/diff_marks_configs.py`).

### Leakage safeguards
- A sample never sees its own track in its reference (`reference_members`; test: replacing s0's bigWigs leaves s0's reference rows unchanged and moves the others'); a copy standing for an unseen cell is left out by identity (`diff_identity`).
- The per-sample statistics use marks only; no label enters. The reference sets of the generated configs contain no test patient, no same-donor partner (asserted in `tests/test_diffmarks.py::test_run_config_of_diff_loopfocus`).
- The AML table keeps the section 63 safeguards (training patients of the fold only, same-donor partners removed, `assert_no_leak`); the cell table holds the 12 training cells only (`cell_objects_table.assert_no_leak`; the trainer's `load_objects` rejects any cell that is not a training sample of the run, tested with K562, HMEC, HUVEC); held-out cells exist only in the evaluation table.
- Oversampling and losses act on the rows of the training split chosen after the honest validation split (`val_exclude_overlap`: no training window overlaps a validation position in any sample), unchanged.

### Tests
`tests/test_diffmarks.py` (38): reference = mean of the others, own track never in its reference, shapes / scales / per-sample standardisation (including a 4x deeper sample), missing and zeroed marks, empty references, unseen-cell identity, config validation and defaults, normaliser pass-through and file compatibility, cache layout (same targets, same ChIP rows as the cache without the channels), `select_tracks`, dropout coupling, a trained model whose output changes when any reference or difference row it reads is changed (select_tracks and track dropout in force), loop / stripe definitions on synthetic maps, control matching, objects-table leakage for training and held-out cells, conversion of the rule-transfer objects, run configs of all families at 10 and 25 kb, existing run configs without new keys, and a 60-step CPU smoke with every flag (both tables, ranking, dropout). Full suite 247 passed, 1 skipped (209 passed, 1 skipped before). Existing run configs, precompute caches (arrays and normaliser files) and a 60-step training log are byte-identical to the previous commit.

### GPU job and evaluation
`results/diff_marks/REMOTE_JOB.md`: run names = claim names, CPU preparation (object tables, caches), commands via `scripts/run_single_gpu.py`, evaluation commands: cell lines by the standard stage (`multicell_expanded.py --stage eval`, rows in `reference_summary.csv`: 17 cells, 413 Akita test regions), AML by `aml_train_eval.py --fold N --variant extra` and `aml_loop_detection.py --recipe diff_loopfocus` (`criteria.csv`, new `enh_gain_spearman` rows), held-out cell objects by `scripts/cell_objects_eval.py`. Estimated remote 4090 time: pilot 1.4 h, primary 10 kb about 6 h (5 to 9 h), 25 kb about 3 h, ablations about 9.5 h, extras about 12 h; minimum useful set about 9 h, everything about 34 h.

### Success criteria (written before any run; 10 kb, ten held-out patients, donor bootstrap)
1. AUROC of the gain over HSPC, AML-private loops against matched shared loops (A_hspc): >= 0.70 (aml_refin 0.59, section 62).
2. Enhancer-like stripes: Spearman of the model's gain over HSPC with the measured gain above that of the reference map alone.
3. Median Pearson on the 413 Akita regions, 12 training and 5 held-out cells: no loss against the matching baseline beyond the honest seed spread (>= -0.006; section 51 two-seed spreads of consensus-input models 0.0002 to 0.0062; use the seed replicates in `reference_summary.csv` once they are there); all12refres_drop_10kb 0.9270 / 0.8972.
4. Cell-specific loops and stripes of K562, HMEC, HUVEC: AUROC of the gain over consensus12 with its interval, reported against the residual model without the recipe.
Results: not yet available.

### Risks
The channels carry no new information for a cell whose marks equal the pool mean, and a network that ignores them returns the section 45 residual model: the ablations separate the channels from the objects. The quantile anchoring is a choice (q50 to 0, q99 to 1); sparse tracks (CUT&Tag) have q50 near 0. The cell objects depend on map depth (cap of 400 / 200 per cell) and on the definitions of `crosscell_rule_transfer.py`, which was still being developed when its objects file (2026-10-11 12:24) was converted: the tables must be regenerated (`python scripts/cell_objects_table.py`, then the 10 kb caches are unaffected, only the run configs read the table) if that stage is rerun with other thresholds; unmatched specific objects keep no control (the fallback of section 63 then takes a control of the same cell and kind). The AML patients have three marks, so most difference rows of the 13-mark layout are 0 for them; the cell lines carry 13. One seed per run; regional in-silico knockouts do not touch the difference rows. Another section may already use number 66 when this lands.

### Files
`hicmultiome/diffmarks.py`, edits in `hicmultiome/{config,precompute,train,tracks,predict}.py`, `scripts/{cell_objects_table,cell_objects_eval,diff_marks_configs}.py`, edits in `scripts/{multicell_expanded,crosscell_predict,aml_train_eval,aml_loop_detection}.py`; `tests/test_diffmarks.py`; `results/diff_marks/REMOTE_JOB.md`, `results/diff_marks/configs/`, `results/cell_objects/` (tables and counts).

## 67. HiC-Reg retrained leak-free on the chromosome split of our models (FINAL; section 66 was the last number when this was written)

Question: the released HiC-Reg numbers (sections 18, 37, 58) are suspected of leakage. How does HiC-Reg score when
it is rebuilt for our cells and tracks and trained on exactly our setting (train on all chromosomes except chr2, 3,
11, 14, 17; predict those five), and how large is the within-chromosome leak?

Audit (full text with file and line references: `results/benchmark/HICREG_LEAK_AUDIT.md`). Routes found: (1) the
released CV predictions are region-wise 5-fold CV inside the chromosome (`Framework.C:199-300`; 64% of the pairs train
each model, 89% of the test pairs have one region among the training regions, 19% (GM12878) and 13-15% (K562) of the
8 neighbouring pixels are training pixels of every model that could have predicted them); (2) the feature files hold no
Hi-C-derived column, the SQRTVC normalisation enters the target only (not a route); (3) new: the prediction pair list
is the list of non-zero pixels of the held-out chromosome's own Hi-C (GM12878 74-85% of the 1 Mb pair slots, K562
42-60%; CV and cross-chromosome files have the same pairs), and `oe_proper.py` turns an absent mappable pixel into the
clipped depletion that a zero-count target pixel has, so the zero pixels of the held-out map are delivered for free
(K562 has 24-39% zero pixels at 5 kb); (4) the released cross-cell maps use models trained on the same chromosome of the
other cell; (5) the MULTICELL fold script cuts contiguous blocks of the pair-sorted table, but the released MULTICELL
files have the pair lists of WINDOW, so the script was not the one used, and the models ship without training pair
lists; (6) no tuner is shipped, `-k1 -l10 -n20` are fixed defaults; the local `hyperopt_hicreg.py` scores against the
held-out matrix; (7) hg19 maps lifted to hg38 centres, 5 kb counts summed into 10 and 25 kb; (8) the released
cross-chromosome files are trained on one chromosome (81 to 107 Mb) with DNase and TBP.

Rebuild (`scripts/hicreg_retrain.py`): our 13 tracks per 5 kb bin as E, P, W and Distance (DNase and TBP absent for our
cells, dropped; SMC3 added), HiC-Reg's native target ln(1 + SQRTVC count) from the VC_SQRT vectors of the 4DN mcools
(GM12878, K562; the vectors divide), every pair within 1 Mb predicted, scored through the section 18 conversion
(expm1, sum into 10 and 25 kb, `oe_proper.py`) against our O/E targets on the identical pixels of the chrho blocks;
random forest as in `Potential.C` (20 trees, bootstrap, one third of the features per split) in scikit-learn
(agreement with regForest 0.983 on 60k pairs, r with the truth 0.830 against 0.826); training on all 75M mappable pairs
of the 17 training chromosomes, no sub-sampling; leaf size 100 and training size chosen on chr10 (`hicreg_retrain_tune_*.csv`);
MULTICELL (HiC-Reg's five cells, 196 features) on 12M pairs and chr14 and chr17 only. Mean of chr14 and chr17 (the blocks
that also hold the released maps), `results/benchmark/hicreg_retrained_*.csv`, `hicreg_retrained_table.md` (all
measures, five-chromosome means):

| cell, grid | source | Pearson | tile median | SCC | insulation r | loop recall | HiCRep |
|---|---|---|---|---|---|---|---|
| GM12878 25 kb | all12refin_drop_chrho | 0.931 | 0.950 | 0.925 | 0.905 | 0.396 | 0.935 |
| GM12878 25 kb | all12honest_drop_chrho (ChIP only) | 0.783 | 0.798 | 0.763 | 0.674 | 0.191 | 0.791 |
| GM12878 25 kb | HiC-Reg retrained, WINDOW | 0.430 | 0.523 | 0.482 | 0.502 | 0.050 | 0.599 |
| GM12878 25 kb | HiC-Reg retrained, MULTICELL | 0.383 | 0.569 | 0.473 | 0.498 | 0.039 | 0.601 |
| GM12878 25 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.579 | 0.620 | 0.538 | 0.447 | 0.032 | 0.668 |
| GM12878 25 kb | HiC-Reg CV, our pipeline | 0.879 | 0.886 | 0.855 | 0.615 | 0.216 | 0.885 |
| GM12878 25 kb | HiC-Reg CV (released) | 0.908 | 0.965 | 0.870 | 0.677 | 0.323 | 0.915 |
| GM12878 10 kb | all12refin_drop_chrho | 0.925 | 0.937 | 0.919 | 0.890 | 0.408 | 0.927 |
| GM12878 10 kb | all12honest_drop_chrho (ChIP only) | 0.788 | 0.810 | 0.777 | 0.735 | 0.182 | 0.796 |
| GM12878 10 kb | HiC-Reg retrained, WINDOW | 0.446 | 0.467 | 0.473 | 0.523 | 0.014 | 0.549 |
| GM12878 10 kb | HiC-Reg retrained, MULTICELL | 0.428 | 0.540 | 0.470 | 0.515 | 0.011 | 0.555 |
| GM12878 10 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.590 | 0.561 | 0.542 | 0.491 | 0.004 | 0.601 |
| GM12878 10 kb | HiC-Reg CV, our pipeline | 0.879 | 0.846 | 0.843 | 0.556 | 0.057 | 0.882 |
| GM12878 10 kb | HiC-Reg CV (released) | 0.920 | 0.939 | 0.881 | 0.626 | 0.148 | 0.925 |
| GM12878 5 kb | all12refin_drop_chrho | 0.906 | 0.911 | 0.902 | 0.898 | 0.363 | 0.923 |
| GM12878 5 kb | all12honest_drop_chrho (ChIP only) | 0.749 | 0.758 | 0.749 | 0.741 | 0.188 | 0.770 |
| GM12878 5 kb | HiC-Reg retrained, WINDOW | 0.454 | 0.433 | 0.475 | 0.537 | 0.005 | 0.525 |
| GM12878 5 kb | HiC-Reg retrained, MULTICELL | 0.448 | 0.503 | 0.474 | 0.534 | 0.034 | 0.534 |
| GM12878 5 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.603 | 0.520 | 0.554 | 0.515 | 0.000 | 0.593 |
| GM12878 5 kb | HiC-Reg CV, our pipeline | 0.872 | 0.814 | 0.836 | 0.565 | 0.030 | 0.874 |
| GM12878 5 kb | HiC-Reg CV (released) | 0.893 | 0.898 | 0.859 | 0.647 | 0.020 | 0.922 |
| K562 25 kb | all12refin_drop_chrho | 0.897 | 0.928 | 0.889 | 0.857 | 0.347 | 0.904 |
| K562 25 kb | all12honest_drop_chrho (ChIP only) | 0.730 | 0.777 | 0.711 | 0.622 | 0.125 | 0.748 |
| K562 25 kb | HiC-Reg retrained, WINDOW | 0.286 | 0.436 | 0.347 | 0.497 | 0.049 | 0.465 |
| K562 25 kb | HiC-Reg retrained, MULTICELL | 0.297 | 0.486 | 0.392 | 0.528 | 0.029 | 0.518 |
| K562 25 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.736 | 0.820 | 0.696 | 0.484 | 0.038 | 0.797 |
| K562 25 kb | HiC-Reg CV, our pipeline | 0.801 | 0.844 | 0.799 | 0.603 | 0.133 | 0.841 |
| K562 25 kb | HiC-Reg CV (released) | 0.888 | 0.950 | 0.864 | 0.670 | 0.178 | 0.923 |
| K562 10 kb | all12refin_drop_chrho | 0.858 | 0.900 | 0.857 | 0.848 | 0.366 | 0.887 |
| K562 10 kb | all12honest_drop_chrho (ChIP only) | 0.715 | 0.724 | 0.714 | 0.721 | 0.139 | 0.745 |
| K562 10 kb | HiC-Reg retrained, WINDOW | 0.336 | 0.417 | 0.378 | 0.554 | 0.028 | 0.452 |
| K562 10 kb | HiC-Reg retrained, MULTICELL | 0.361 | 0.420 | 0.426 | 0.584 | 0.016 | 0.500 |
| K562 10 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.747 | 0.760 | 0.708 | 0.496 | 0.002 | 0.782 |
| K562 10 kb | HiC-Reg CV, our pipeline | 0.817 | 0.794 | 0.808 | 0.618 | 0.064 | 0.858 |
| K562 10 kb | HiC-Reg CV (released) | 0.878 | 0.899 | 0.858 | 0.650 | 0.009 | 0.925 |
| K562 5 kb | all12refin_drop_chrho | 0.771 | 0.817 | 0.776 | 0.842 | 0.338 | 0.814 |
| K562 5 kb | all12honest_drop_chrho (ChIP only) | 0.641 | 0.641 | 0.653 | 0.730 | 0.169 | 0.664 |
| K562 5 kb | HiC-Reg retrained, WINDOW | 0.414 | 0.324 | 0.440 | 0.625 | 0.030 | 0.428 |
| K562 5 kb | HiC-Reg retrained, MULTICELL | 0.439 | 0.333 | 0.479 | 0.654 | 0.026 | 0.453 |
| K562 5 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.774 | 0.703 | 0.755 | 0.599 | 0.003 | 0.790 |
| K562 5 kb | HiC-Reg CV, our pipeline | 0.781 | 0.697 | 0.783 | 0.692 | 0.039 | 0.758 |
| K562 5 kb | HiC-Reg CV (released) | 0.845 | 0.814 | 0.840 | 0.728 | 0.005 | 0.899 |

Within-chromosome leak (CV protocol on the retrained pipeline minus the retrained cross-chromosome model, Pearson): GM12878
+0.449 (25 kb), +0.433 (10 kb), +0.418 (5 kb); K562 +0.515, +0.482, +0.367; tile median +0.36 to +0.41, SCC +0.34 to +0.45
(`hicreg_retrained_leak.csv`). Released CV minus retrained cross-chromosome: +0.478 (GM12878 25 kb), +0.601 (K562 25 kb).
Support effect (retrained model, pixels with zero target count removed as in the release, Pearson, chr14+chr17):
K562 0.419 / 0.544 / 0.758 at 25 / 10 / 5 kb against 0.227 / 0.297 / 0.394 with every pair predicted; GM12878 0.394 / 0.456 /
0.512 against 0.358 / 0.401 / 0.420 (`hicreg_leak_support_effect.csv`).

Verdict. The leak-free retrained HiC-Reg is worse than the released single-chromosome cross-chromosome version, not better:
Pearson 0.430 against 0.579 (GM12878 25 kb) and 0.286 against 0.736 (K562 25 kb), 0.454 against 0.603 and 0.414 against
0.774 at 5 kb; training on 17 chromosomes instead of one does not help (4M-pair control 0.432 and 0.257). The released K562
cross-chromosome score is mostly the zero-pixel support (retrained with that support: 0.758 against 0.774 at 5 kb). Our
margin grows: all12honest_drop_chrho (ChIP only) leads retrained HiC-Reg by 0.35 (GM12878) and 0.44 (K562) Pearson at
25 kb and by 0.30 and 0.23 at 5 kb, all12refin_drop_chrho by 0.50 and 0.61 at 25 kb; both are ahead on all nine measures in
both cells and all grids; K562, for which our models never see the cell's Hi-C while HiC-Reg trains on K562 Hi-C, is not
closer. MULTICELL equals WINDOW (25 kb Pearson 0.383 and 0.297). The CV protocol on the same pipeline reaches 0.879 and
0.801 at 25 kb (released CV 0.908, 0.888), so the pipeline reproduces the interpolation level.

Caveats. The rebuild lacks DNase and TBP and differs in Hi-C source, signal source and forest implementation; the gap to
the released cross-chromosome maps combines the support effect with these differences, only the support is separated;
one seed; two chromosomes per block; MULTICELL on 12M pairs; released maps hg19 lifted, retrained maps hg38 native.
The ceiling of the native target in this scoring (true SQRTVC maps through the conversion) is 0.78 to 0.94 Pearson
(`hicreg_leak_oracle.csv`).

Files: `scripts/hicreg_retrain.py`, `scripts/hicreg_leak_audit.py`, `scripts/hicreg_retrained_table.py`,
`--chrho-hicreg-retrained` in `scripts/hic_measures.py`; `results/benchmark/HICREG_LEAK_AUDIT.md`,
`hicreg_retrained_{per_chromosome,summary,leak}.csv`, `hicreg_retrained_table.md`, `hicreg_leak_*.csv`,
`hicreg_retrain_tune_*.csv`, `hicreg_retrain_check_regforest_gm12878.json`;
`results/hic_measures_chrho/chrho_hicregre_{per_block,summary}.csv`; predictions and cools under
`/mnt/processed/wolffjoa/hic-multiome/runs/benchmark/hicreg_retrain/`. Reproduce:

    python scripts/hicreg_retrain.py --stage data
    python scripts/hicreg_retrain.py --stage tune --cell gm12878 --grid-sizes 4000000 16000000 --grid-leaf 10 30 100 300 1000
    python scripts/hicreg_retrain.py --stage fit --cell gm12878 --leaf 100 && ... --feat multicell --n-train 12000000
    python scripts/hicreg_retrain.py --stage cv --cell gm12878 --leaf 100 --n-train 4000000 --chroms chr14 chr17
    python scripts/hicreg_retrain.py --stage fit --cell gm12878 --suffix small --leaf 100 --n-train 4000000 --chroms chr14 chr17
    python scripts/hicreg_retrain.py --stage cools
    python scripts/hicreg_leak_audit.py --stage support mcfolds oracle supportleak
    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-hicreg-retrained
    python scripts/hicreg_retrained_table.py


## 68. Are the mark-to-structure rules the same in every cell type? Cell-specific loops and stripes of 15 cells against AML patients (FINAL; section 67 was the last number when this was written)

Script `scripts/crosscell_rule_transfer.py` (stages tracks, hic, objects, features, counts, rules, transfer, figures; hicdiff env, CPU only, nice 10), results in `results/crosscell_rules/`, caches in `/mnt/processed/wolffjoa/hic-multiome/crosscell_rules/`. Cells: the 12 training cells (reference = leave-one-out consensus of the other 11, `consensus_loo_*`) and K562, HMEC (deep map), HUVEC (deep map) (reference = consensus12). Cancer: K562, HCT116, MCF-7, A549, HepG2, PC-9; normal: the other nine. AML: the ten test donors of section 62 with the objects of `aml_loop_classifier_v2/features.csv.gz` and `aml_loop_detection/object_scores.csv.gz` (gain|measured over HSPC), delta features recomputed here from the 10 kb track caches. Section 66 (differential channels, model-level, PENDING) is a different analysis; this one is a feature-level test.

### Definitions (as the AML objects, 10 kb)
Loops: donut enrichment of the cell map at local maxima (> 0.4 log2, 60 kb to 1.5 Mb; MIN_BINS 6 as in the AML pool because the donut outer radius is 4 bins, not 20 kb), consensus value = 3x3 maximum, gain = cell minus consensus; candidates = both anchors on peak bins (top 8 percent H3K27ac or CTCF in the cell or its consensus track). Specific: enrichment >= 0.5 and gain >= 0.30 (about the 30th percentile of AML positives); shared: >= 0.6 in cell and consensus, not within 2 bins of a specific loop; controls matched 1:1 on distance, CTCF motif orientation and the anchors' mean H3K27ac (A_full) or the anchors' consensus CTCF and H3K27ac (A_ref, the analogue of A_hspc). Stripes: rectangle mean log O/E, 4 anchor bins x 40 extent bins starting 50 bins to the right or left (the Xu extents start 16 to 268 bins away), anchor with a peak bin; specific >= 0.55 and gain >= 0.20, shared both >= 0.55; matched on direction, orientation and anchor H3K27ac. Positives capped at 2500 per cell and kind before matching. Types as in `pair_feats` (TSS in one anchor, partner H3K27ac or H3K27me3 above the 70th percentile of the pooled training scale: PE, PS; stripes enh, rep). Tracks of all cells and donors are quantile-normalised at 10 kb to the mean training-cell distribution; features: delta (own minus reference), reference level, own, at 10 kb, +-20 kb, +-100 kb, min and max over the two anchors (stripes: anchor and extent block), delta standardised per sample (median / robust SD of the genome-wide delta, `delta_standardisation.csv`), methylation delta at +-100 kb for 9 cells (GM12878, IMR-90, H1, HepG2, A549, GM23248, CD14+, K562, HMEC; HCT116, HFF, MCF-7 tracks flat, NHEK, PC-9, HUVEC none) and 8 donors (reference mean of healthy 103 and 673), distance, motif orientation / score, promoter flags. GM23248 has no CTCF track.

### Counts (object_counts*.csv)
Cell-specific loops 125 779 in 15 cells (PE 39 475, PS 18 746, other 67 558; per cell 426 to 27 079), stripes 36 222 (enh 10 766, rep 16 523, other 8 933); after the cap and A_full matching 10 110 loop pairs (20 220 objects, 1348 per cell) and 25 310 stripe pairs (50 620 objects). AML: 2276 loops (PE 304, PS 127, other 1845) and 1075 stripes (enh 340, rep 494); 2232 loop and 1059 stripe pairs (4464 / 2118 objects, 446 / 212 per donor); per fold 1254 to 1858 loop and 686 to 734 stripe objects (the "1.5k / 0.7k" of the task). Cells give 4.5 times as many matched loop objects as the ten donors and 24 times as many stripe objects, but the shared-loop pool limits matching (25 percent of the specific loops matched).

### Within-cell rules (rules_key.csv, rules_consistency_*.csv, fig_effect_heatmap.png)
Controls A_ref; effect = Spearman with the gain, or AUROC against matched shared controls; mean over cells [bootstrap CI over cells] / AML donors. Two class conventions: classes from the own marks of positives and controls (own), and controls inheriting the class of their matched positive (inherit, the convention of section 64; circular for PE / PS / enh / rep because the class is defined by the same own marks). AML reproduces section 64 under inherit (PE delta H3K27ac max 0.74, min 0.70; PS delta H3K27me3 0.79; repressive stripes 0.72; stripes rho 0.28, enh 0.37).
- PE loops, delta H3K27ac max(A,B) AUROC: own 0.56 [0.54, 0.58] in 15 cells (13 above 0.5; cancer 0.56, normal 0.56), AML 0.58 (pooled 0.59); inherit 0.65 [0.63, 0.68] cells (15/15), AML 0.74. Same sign and about the same size in cells and donors with class defined from the marks; larger in AML only under the circular convention.
- All loops, delta H3K27ac max AUROC 0.55 [0.53, 0.58] cells (13/15), AML 0.61 (pooled): supported, small. Gain magnitude: delta H3K27ac +-100 kb rho -0.03 [-0.05, 0.01] cells, -0.13 AML (negative in 8/10 donors and 12/15 cells): no positive relation anywhere.
- PS loops delta H3K27me3: own 0.50 cells (4/13 above 0.5), AML 0.54 (2 evaluable donors); inherit 0.63 cells (12/13), AML 0.79: only under the circular convention. Repressive stripes delta H3K27me3 anchor: own 0.50 cells, 0.49 AML; inherit 0.64 cells (15/15), 0.72 AML. Enhancer-like stripes delta H3K27ac: own 0.53 cells, 0.49 AML; inherit 0.62 cells, 0.60 AML.
- Stripe gain vs delta H3K27ac +-100 kb (the clear AML signal): AML rho 0.28 [0.24, 0.33] all, 0.37 enhancer-like, 10/10 donors; cells -0.06 [-0.13, 0.00] (6/15 positive; cancer -0.11, normal -0.03), enhancer-like 0.00. AML minus cells 0.35 [0.27, 0.42]. With the other AML donors as reference the AML value falls to 0.06 (0.19 enhancer-like): the signal depends on the HSPC reference (differentiation axis).
- CTCF delta: AUROC 0.50 [0.46, 0.54] cells, 0.45 AML loops (below 0.5), 0.50 stripes: no support anywhere. Methylation delta (min over anchors, +-100 kb): AUROC 0.50 [0.47, 0.53] cells (5/9 above 0.5; cancer 0.54, normal 0.48), 0.57 AML loops, gain rho 0.01 / -0.02. Consensus H3K27ac level +-100 kb vs gain: loops 0.03 cells, -0.11 AML (AML minus cells -0.17 [-0.25, -0.09]); stripes -0.05 cells, -0.23 AML (-0.17 [-0.31, -0.03]): the locus-state effect has the opposite sign in AML.
- Cancer minus normal cells: no key rule differs (all CIs include 0 except loop all delta H3K27ac +-100 kb rho -0.05 [-0.10, 0.00]); PS delta H3K27me3 AUROC -0.07 [-0.13, -0.02] (own, n = 5 + 8 cells).
- Between-cell heterogeneity of the loop rules is small (I2 0.0 to 0.6 for PE and PS AUROC with SE from loci bootstrap), larger for all stripe and methylation / CTCF rules (0.8 to 1.0; effects near zero with spread above sampling error).

### Transfer (transfer_results.csv, transfer_summary.csv, transfer_paired_differences.csv, fig_transfer_matrix.png (A_full), fig_transfer_matrix_Aref.png)
Boosted trees (the classifier settings of section 57) on standardised delta (H3K27ac, H3K27me3; 12 features), consensus-state (12) and locus (10) features, trained on whole groups, locus-disjoint (training rows within 2 bins of a test anchor dropped), per-group gain percentile as regression target; mean over test groups [bootstrap CI over groups]. Controls A_full (A_ref in the CSV and `fig_transfer_matrix_Aref.png`).
- (a) leave-one-cell-out, 14 other cells: loops AUROC 0.69 [0.66, 0.71], gain rho 0.31 [0.25, 0.36] (range 0.49 (PC-9) to 0.79 (HMEC); 14 of 15 cells above 0.5); stripes 0.59 [0.57, 0.60], rho 0.14. Adding CTCF (14 cells): loops 0.82 [0.80, 0.84]. Without consensus-state and locus features (delta + distance + orientation only, RULE): loops 0.62, stripes 0.54; state alone (STATE): 0.67 and 0.58. Block permutation importance (AUROC drop): loops delta 0.054, reference 0.089, locus 0.083; stripes 0.018, 0.065, 0.007.
- (b) all cells to AML donors: loops 0.70 [0.67, 0.73] (private 0.71; with CTCF, 8 donors, 0.81), gain rho 0.09 [0.04, 0.14]; stripes 0.58 [0.55, 0.61], rho 0.10. RULE only: 0.57 and 0.52 (rho -0.06 loops); STATE alone 0.69 and 0.57. AML-trained within AML (leave-one-donor-out, d_aml_only): loops 0.77 [0.73, 0.80] (private 0.78), rho 0.30; stripes 0.62, rho 0.25.
- (c) AML donors to cells: loops 0.54 [0.51, 0.57], rho -0.15 [-0.18, -0.11] (with CTCF 0.70 on 14 cells); stripes 0.52, rho -0.01. The gain ranking learned in AML is inverted in cells.
- (d) held-out donor, AML-only vs AML + all cells: loops AUROC 0.766 vs 0.738 (difference -0.028 [-0.040, -0.016], 0/10 donors better), private loops -0.030 [-0.042, -0.020], gain rho 0.30 vs 0.19 (-0.11 [-0.15, -0.07]); with the AML rows weighted to equal total weight the difference is -0.012 [-0.017, -0.007] (rho -0.04 [-0.07, -0.01]); stripes -0.027 [-0.057, 0.000] and -0.009 [-0.022, 0.005]. Cells added to the AML training do not raise the private-loop AUROC or the gain Spearman. (e) held-out cell, other cells vs other cells + AML: -0.008 [-0.013, -0.004] AUROC (loops), 0.000 for RULE; stripes -0.001.
- Locus-disjoint caveat: A_full controls are matched on own H3K27ac but not on reference state, so reference-level and locus features carry most of the cross-cell AUROC (STATE-only 0.67 of 0.69 in (a), 0.69 of 0.70 in (b)).
- Sign / size consistency (lr_coefficients.csv, lr_sign_consistency.csv; per-group logistic regression on the 12 standardised delta features): for loops 5 of 12 features have the same mean sign in cells and AML (e.g. H3K27ac +-100 kb max: positive in 80 percent of cells and of donors). The minimum-over-anchors H3K27ac at 10 kb has coefficient -0.13 in cells (negative in 15/15) and +0.11 in AML (positive in 9/10), and the +-20 kb maximum is +0.15 in cells (14/15) and -0.10 in AML (2/10): opposite signs. For stripes 7 of 12 agree in sign, with small coefficients (|mean| <= 0.15).

### Verdict
No conclusion of section 47 changes. The headline 8-patient numbers, the increments over the HSPC state (+0.023, +0.015,
+0.006), the gain-of-H3K27ac signal (0.617, 8 of 8) and the pair-type and stripe numbers were never affected: the healthy
donors were not in the data, and the duplicate pairs were never together in an 8- or 10-sample experiment. Only the
18-sample methylation experiments were exposed to the partner leak, and they move by at most 0.003 (A_hspc 0.820 to 0.822, A_do
0.875 to 0.875, B_rand 0.703 to 0.701; plain folds 0.848 to 0.850, 0.905 to 0.903, 0.712 to 0.709). The held-out 168/496 unit scores 0.822
where 168 alone had 0.857 and 496 0.787, so the single-sample numbers were higher than the unit number. Pooled over units the
change is negligible because the locus-disjoint folds already removed most shared-anchor information the partner carried. The 18-sample
experiments now rest on 16 donors. The remaining limit is the "probably" for 260 and 424 (section 54).

Reproduce: `python scripts/aml_loop_classifier.py --stage hic tracks motifs stripes_hic build subtypes model figures --jobs 4
--workers 4 --null-reps-lr 20 --null-reps-gb 10` (hicdiff env).

## 58. Chromosome-held-out comparison at 25 kb, competitor resolutions, and the 5 kb plan (FINAL for 25 kb and 5 kb; section 57 was the last number when this was written)

Question: how do the chromosome-held-out final models of section 37 compare
with each published predictor at 25 kb, which competitor supports which
resolution, and what is needed to repeat the comparison at 5 kb?

Inventory of the competitor predictions on disk (none was rerun; checked from
the cooler headers and file lists on 2026-10-10):

| competitor | native grid and reach | predictions on disk | held out by the competitor | 25 kb | 10 kb | 5 kb |
|---|---|---|---|---|---|---|
| C.Origami | 8192 bp bins, 2 097 152 bp window, hg38, released IMR-90 model | `C.Origami/testNovember/IMR90/chr2.cool`, `chr3.cool` | chr2 (test), chr3 (validation) | sampled from 8192 bp | sampled (section 37) | sampled (up-sampling) |
| Epiphany | 10 kb bins, 990 kb reach, hg38, released GM12878 weights | `epiphany_pretrained/assemble_matrix_chr{2,3,11,17}.cool` | chr3, 11, 17 | sampled from 10 kb | native (section 37) | sampled (up-sampling) |
| HiC-Reg | 5 kb bins, 1 Mb reach, hg19 | published GM12878 chr1 to 22 (`Gm12878_chr<N>_RF_tree20_prediction_predicted_coords.cool`, 5 kb, and a `_25kb.cool` sum); released text predictions for GM12878 and K562, chr14 and chr17: CV, cross-chromosome (WINDOW, MULTICELL), cross-cell (`home_offload/HiC-Reg`) | cross-chromosome: chr14 (model of chr17) and chr17 (model of chr14) | native by summing 5 kb counts | native by summing 5 kb counts | native, not yet converted (`hicreg_audit.variants()` covers 25 and 10 kb) |
| Hi-cGAN | per-resolution Zenodo records: 2 kb (Akita regions), 5 kb (GM12878 only, best_plus_two RAD21, H3K4me3, SMC3), 10 kb and 25 kb (GM12878 whole genome and 25 cross-cell-type sets: the GM12878, HMEC, HUVEC, IMR-90 and K562 models on each cell's inputs) | `zenodo_records/02_best_models`, `04_sweep_5kb`, `05_sweep_10kb`, `06_sweep_25kb` | trained on odd chromosomes, even held out | native | native | native, GM12878 only, no cross-cell set |
| Akita | 2048 bp bins, 448-bin (917 kb) crop of a 1 Mb window, 413 test regions | released predictions in `akita_testset_rasters.npz` (GM12878, HFF) | all 413 regions are excluded from our training | none | none | none |

C.Origami and Epiphany have no native 25 kb output. Their O/E maps are read
at the 25 kb bin centres by the code path of the 10 kb blocks
(`benchmark_blocks.sample_competitor`, bilinear in (bin, bin) space, a pixel
kept when all four neighbours exist), so the 8192 bp and 10 kb values are read
at points and keep structure finer than a 25 kb bin. HiC-Reg's 25 kb maps are
sums of its 5 kb counts. Akita has no whole-chromosome output and no 25 kb
score exists (the 2 kb comparison is section 18, block A). Hi-cGAN has no
IMR-90 prediction in the 25 kb chrho blocks of section 37; the record 06 sets
`pred_imr90_on_imr90` and `pred_gm12878_on_imr90` were converted with
`oe_proper.py` for this section (`BENCH_HICGAN_IMR90_25KB=1`).

Scoring is the section 37 procedure at 25 kb (`scripts/hic_measures.py --stage
chrho_score chrho_report --chrho-25kb-comps`): chr2, 3, 11, 14, 17 absent
from the data of every cell, whole-chromosome tiling, identical pixels within
a block (finite in the target and in every source), band cut to the shortest
reach (C.Origami 2 Mb, Epiphany 39 bins = 975 kb, HiC-Reg 1 Mb, Hi-cGAN 2 Mb).
The HiC-Reg and Hi-cGAN blocks are the 25 kb rows of section 37 (identical
files); the C.Origami blocks (chr2, chr3), the Epiphany blocks (chr3, 11, 17)
and the Hi-cGAN IMR-90 rows (chr2 only: chr3 is one of its training
chromosomes) are new. Mean over the chromosomes of a block; compartment r is
the compartment preference correlation of section 31; the APA ratio equals 1
for the measured map.

| block | source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|---|
| C.Origami, IMR-90 chr2, chr3 | all12refin_drop_chrho | 0.940 | 0.961 | 0.941 | 0.943 | 0.817 | 0.349 | 1.038 | 0.958 | 0.947 |
|  | all12honest_drop_chrho (ChIP only) | 0.804 | 0.864 | 0.800 | 0.775 | 0.647 | 0.206 | 0.541 | 0.862 | 0.795 |
|  | consensus of the other cells | 0.877 | 0.932 | 0.872 | 0.817 | 0.788 | 0.374 | 0.855 | 0.885 | 0.844 |
|  | C.Origami | 0.942 | 0.922 | 0.933 | 0.898 | 0.752 | 0.313 | 2.930 | 0.980 | 0.950 |
| C.Origami block, chr2 only | all12refin_drop_chrho | 0.945 | 0.962 | 0.946 | 0.941 | 0.803 | 0.336 | 1.030 | 0.960 | 0.952 |
|  | all12honest_drop_chrho (ChIP only) | 0.806 | 0.855 | 0.799 | 0.767 | 0.616 | 0.197 | 0.556 | 0.856 | 0.798 |
|  | C.Origami | 0.944 | 0.921 | 0.935 | 0.899 | 0.751 | 0.283 | 2.924 | 0.980 | 0.952 |
|  | Hi-cGAN IMR-90 model | 0.808 | 0.924 | 0.792 | 0.630 | 0.618 | 0.258 | 0.714 | 0.945 | 0.845 |
|  | Hi-cGAN GM12878 model | 0.335 | 0.592 | 0.349 | 0.443 | 0.309 | 0.075 | 0.217 | 0.739 | 0.483 |
| Epiphany, GM12878 chr3, 11, 17 | all12refin_drop_chrho | 0.917 | 0.944 | 0.913 | 0.900 | 0.827 | 0.453 | 0.786 | 0.948 | 0.928 |
|  | all12honest_drop_chrho (ChIP only) | 0.754 | 0.788 | 0.739 | 0.668 | 0.608 | 0.241 | 0.353 | 0.846 | 0.773 |
|  | consensus of the other cells | 0.841 | 0.887 | 0.832 | 0.804 | 0.805 | 0.456 | 0.765 | 0.878 | 0.843 |
|  | Epiphany | 0.518 | 0.672 | 0.501 | 0.377 | 0.437 | 0.254 | 1.513 | 0.775 | 0.640 |
| HiC-Reg, GM12878 chr14, chr17 | all12refin_drop_chrho | 0.931 | 0.950 | 0.925 | 0.905 | 0.833 | 0.396 | 0.792 | 0.953 | 0.935 |
|  | all12honest_drop_chrho (ChIP only) | 0.782 | 0.798 | 0.763 | 0.674 | 0.608 | 0.191 | 0.319 | 0.867 | 0.791 |
|  | consensus of the other cells | 0.868 | 0.901 | 0.853 | 0.804 | 0.805 | 0.396 | 0.818 | 0.894 | 0.857 |
|  | HiC-Reg cross-chromosome WINDOW | 0.579 | 0.620 | 0.538 | 0.447 | 0.322 | 0.032 | -0.002 | 0.846 | 0.668 |
|  | HiC-Reg cross-chromosome MULTICELL | 0.615 | 0.634 | 0.567 | 0.455 | 0.345 | 0.038 | -0.032 | 0.856 | 0.692 |
|  | HiC-Reg CV (within-chromosome) | 0.908 | 0.964 | 0.870 | 0.677 | 0.723 | 0.323 | 0.764 | 0.987 | 0.915 |
| HiC-Reg, K562 chr14, chr17 | all12refin_drop_chrho | 0.897 | 0.928 | 0.889 | 0.857 | 0.738 | 0.347 | 0.843 | 0.927 | 0.904 |
|  | all12honest_drop_chrho (ChIP only) | 0.730 | 0.777 | 0.711 | 0.622 | 0.568 | 0.125 | 0.187 | 0.834 | 0.748 |
|  | consensus of the other cells | 0.857 | 0.905 | 0.853 | 0.839 | 0.769 | 0.333 | 0.876 | 0.887 | 0.868 |
|  | HiC-Reg cross-chromosome WINDOW | 0.736 | 0.820 | 0.696 | 0.484 | 0.523 | 0.038 | 0.292 | 0.927 | 0.797 |
|  | HiC-Reg cross-chromosome MULTICELL | 0.740 | 0.819 | 0.700 | 0.500 | 0.519 | 0.030 | 0.260 | 0.927 | 0.800 |
|  | HiC-Reg CV (within-chromosome) | 0.888 | 0.950 | 0.864 | 0.670 | 0.688 | 0.178 | 0.698 | 0.973 | 0.923 |
| Hi-cGAN, GM12878 chr2, chr14 | all12refin_drop_chrho | 0.934 | 0.949 | 0.929 | 0.901 | 0.808 | 0.405 | 0.810 | 0.959 | 0.933 |
|  | all12honest_drop_chrho (ChIP only) | 0.796 | 0.824 | 0.778 | 0.698 | 0.616 | 0.200 | 0.323 | 0.883 | 0.773 |
|  | consensus of the other cells | 0.860 | 0.899 | 0.848 | 0.784 | 0.777 | 0.389 | 0.841 | 0.897 | 0.835 |
|  | Hi-cGAN GM12878 model | 0.221 | 0.485 | 0.225 | 0.237 | 0.218 | 0.048 | 0.051 | 0.644 | 0.322 |
| Hi-cGAN, K562 chr2, chr14 | all12refin_drop_chrho | 0.904 | 0.935 | 0.896 | 0.843 | 0.727 | 0.356 | 0.851 | 0.941 | 0.891 |
|  | all12honest_drop_chrho (ChIP only) | 0.765 | 0.826 | 0.743 | 0.657 | 0.550 | 0.141 | 0.193 | 0.854 | 0.721 |
|  | consensus of the other cells | 0.853 | 0.909 | 0.849 | 0.816 | 0.718 | 0.350 | 0.840 | 0.895 | 0.850 |
|  | Hi-cGAN GM12878 model | 0.143 | 0.479 | 0.147 | 0.218 | 0.213 | 0.050 | 0.156 | 0.655 | 0.220 |


Replicate ceilings at 25 kb (same blocks): IMR-90 Pearson 0.976, SCC 0.979;
GM12878 0.941 to 0.947; K562 0.936 (`chrho_25kb_competitors.csv`).

Verdict per competitor at 25 kb, ours = all12refin_drop_chrho (reference
map as input) and all12honest_drop_chrho (ChIP only, no consensus); "level"
is a difference within 0.01 of the mean over the block's chromosomes
(`chrho_25kb_vs_competitor.csv`, with per-chromosome win counts):

- C.Origami (IMR-90, chr2 and chr3, resampled from 8192 bp). The reference-map
  model is level on Pearson (0.940 against 0.942; higher on 1 of 2
  chromosomes), SCC (0.941 against 0.933) and HiCRep (0.947 against 0.950),
  ahead on the 1 Mb tile median (0.961 against 0.922), insulation r (0.943
  against 0.898), boundary recall (0.817 against 0.752) and loop recall (0.349
  against 0.313), each on both chromosomes, and behind on compartment r (0.958
  against 0.980, 0 of 2). C.Origami's APA ratio is 2.93, the model's 1.04.
  The ChIP-only model is behind on every measure (Pearson 0.804, SCC 0.800,
  tile median 0.864, insulation 0.775, boundary recall 0.647, loop recall
  0.206). At 10 kb (section 37) the corresponding values are 0.926 against
  0.920 Pearson and 0.162 loop recall for C.Origami.
- Epiphany (GM12878, chr3, 11, 17, resampled from 10 kb). The reference-map
  model is ahead on all measures on all three chromosomes (Pearson 0.917
  against 0.518, SCC 0.913 against 0.501, insulation 0.900 against 0.377,
  loop recall 0.453 against 0.254); the ChIP-only model leads by 0.24
  Pearson (0.754) and 0.24 SCC, with loop recall 0.241 against 0.254. The
  Epiphany APA ratio is 1.51 and the model's 0.79 (ChIP only 0.35).
- Hi-cGAN (GM12878 and K562 chr2, chr14; native 25 kb). Ahead on every
  measure on every chromosome: Pearson 0.934 (ChIP only 0.796) against
  0.221 on GM12878 and 0.904 (0.765) against 0.143 on K562. On IMR-90 chr2,
  Hi-cGAN's IMR-90 model (trained on IMR-90, odd chromosomes) scores 0.808
  Pearson, 0.792 SCC, 0.924 tile median, 0.630 insulation r, 0.618 boundary
  recall, 0.258 loop recall: the reference-map model is ahead on every
  measure (Pearson 0.945, difference 0.136); the ChIP-only model (0.806) is
  level on Pearson and SCC, ahead on insulation (0.767 against 0.630) and
  behind on the tile median (0.855), loop recall (0.197) and compartment r
  (0.856 against 0.945). The GM12878 Hi-cGAN model applied to IMR-90 scores
  0.335.
- HiC-Reg, cross-chromosome (leak-free; native 25 kb sum; chr14, chr17).
  GM12878: the reference-map model is ahead on every measure (Pearson 0.931
  against 0.579 WINDOW and 0.615 MULTICELL) and the ChIP-only model too
  (0.783). K562: reference-map model ahead on every measure (0.897 against
  0.736 and 0.740); the ChIP-only model is level on Pearson (0.730, -0.006
  and -0.010), ahead on SCC (0.711 against 0.696 and 0.700), insulation
  (0.622 against 0.484 and 0.500), boundary recall and loop recall (0.125
  against 0.038 and 0.030), behind on the tile median (0.777 against 0.820
  and 0.819), compartment r (0.834 against 0.927) and HiCRep (0.748 against
  0.797 and 0.800).
- HiC-Reg, published within-chromosome CV maps (not a held-out score). The
  reference-map model is ahead on Pearson on GM12878 (0.931 against 0.908)
  and level on K562 (0.897 against 0.888), ahead on SCC, insulation r,
  boundary and loop recall in both cells, ahead on HiCRep on GM12878 (0.935
  against 0.915) and behind on K562 (0.904 against 0.923), and behind on the
  tile median (0.950 against 0.964; 0.928 against 0.950) and compartment r
  (0.953 against 0.987; 0.927 against 0.973); the ChIP-only model is behind
  on every measure except insulation r on GM12878 (level, 0.674 against
  0.677); Pearson 0.783 and 0.730.
- Akita: no 25 kb score exists.

Limits. (i) The competitors differ in training data and inputs from our
12-cell models; the comparison states what each released prediction scores on
chromosomes it did not train on, not a comparison of architectures
. (ii) Leakage status: C.Origami chr2 (test) and chr3 (validation) are
unseen by C.Origami and by our chrho models, and both saw IMR-90 Hi-C on
other chromosomes (ours: 17 chromosomes of IMR-90 and 11 further cells;
C.Origami: IMR-90 only); Epiphany chr3, 11, 17 unseen by both; HiC-Reg
cross-chromosome files are leak-free (model trained on the other of chr14 and
chr17, both absent from our data), HiC-Reg CV files are within-chromosome
interpolation (section 18); Hi-cGAN GM12878 and K562 on the even chromosomes
chr2 and chr14 (K562 was never a Hi-cGAN training cell; IMR-90 chr2 for the
IMR-90 model). The reference-map model reads the leave-one-out consensus of
the other cells' measured maps on the held-out chromosomes as input (its
intended use; the ChIP-only model reads none). (iii) The resampled
C.Origami and Epiphany maps are read at points on the 25 kb grid, and the
insulation, boundary and loop measures are defined in bins of the grid; the
competitor scores differ between 10 and 25 kb (C.Origami: loop recall 0.313
at 25 kb, 0.162 at 10 kb; boundary recall 0.752 and 0.474), so the margins
are specific to the grid. (iv) Chromosome counts are small (2 to 3 per
block); the win counts in `chrho_25kb_vs_competitor.csv` are per chromosome,
not per region. (v) The C.Origami APA ratio of 2.9 is an over-enrichment of
loop signal against the measured map, which the loop recall does not show.
(vi) The consensus of the other cells has a higher boundary recall than the
reference-map model in the K562 HiC-Reg block (0.769 against 0.738). The
refres model is in the csv files and equals refin within 0.01 on Pearson in
every block.

5 kb (FINAL). The three chromosome-held-out 5 kb models (`all12refin_drop_chrho_5kb`,
`all12honest_drop_chrho_5kb` (ChIP only, no consensus), `all12refres_drop_chrho_5kb`;
chr2, 3, 11, 14, 17 absent from the data of every cell) were trained on the
remote 4090 and predicted on the five chromosomes on a local GPU (inference only).
Scoring is the 25 kb procedure on the 5 kb grid (`scripts/hic_measures.py --stage
chrho_score chrho_report --chrho-res 5000`; 11 blocks), identical pixels within a
block, same metric set.

5 kb parameters (all behind `--chrho-res 5000`; the 25 and 10 kb code paths are
unchanged): model window 384 bins (1.92 Mb), insulation square 10 bins (50 kb,
as 5 bins at 10 kb), loop and APA windows of the 10 kb row doubled in bins
(donut 2 to 6, loop distance 6 to 200 bins, APA half-width 6, corner 2, NMS 2,
matching tolerance 2 bins = 10 kb; `benchmark_blocks.params(5000)`), boundary
matching tolerance 2 bins (10 kb, as the 10 kb row), hicFindTADs depths 3 to 10
bins as at every resolution, HiCRep h = 5 bins. The reference loops are the 10 kb
hicDetectLoops calls of the measured map. Bands: C.Origami 383 bins (1.9 Mb),
Epiphany 198 bins (990 kb), HiC-Reg 200 bins (1 Mb), Hi-cGAN 383 bins (1.9 Mb).
The count-space Pearson AUC uses min(1 Mb / res, band) distances (the Epiphany band
is 199 diagonals), a no-op for the existing blocks. Expected-count curves of a band
wider than 200 diagonals get their own file (`_D383`). Regression check: the 10 kb
Hi-cGAN GM12878 chr14 block re-scored with the edited code into a scratch folder
equals the committed `measures.csv` in all 48 columns (exact) except the HiCRep
value of all12refres_drop_chrho, which is empty in the committed file (that
tool run failed there) and 0.937 in the re-run.

Competitor conversions (nothing re-run; saved predictions only):
- Hi-cGAN: the GM12878 5 kb record (`02_best_models/.../5kb/best_plus_two`,
  RAD21, H3K4me3, SMC3; trained on odd chromosomes) through `oe_proper.py`
  (chr2, chr14; 2 Mb distance cap). No K562 or IMR-90 5 kb prediction exists
  (cross-cell sets exist at 10 and 25 kb only), so there is no K562 Hi-cGAN
  block at 5 kb.
- HiC-Reg: released 5 kb predictions of GM12878 and K562, chr14 and chr17,
  WINDOW and MULTICELL, cross-chromosome and CV, native 5 kb (no summing), through
  `oe_proper.py` (`hicreg_audit.variants()` with `BENCH_5KB=1`).
- C.Origami (8192 bp, IMR-90 chr2, chr3) and Epiphany (10 kb, GM12878 chr3, 11,
  17): no native 5 kb output. Their O/E maps (the native-resolution files already
  made for the 10 and 25 kb blocks) are read at the 5 kb bin centres with the
  same convention as at 25 kb (bilinear in (bin, bin) space,
  `benchmark_blocks.sample_competitor`, a pixel kept when all four neighbours
  exist), an up-sampling: the 5 kb scores of these two carry no information
  finer than 8192 bp and 10 kb.

Mean over the chromosomes of a block (`chrho_5kb_table.md`, csv files
`chrho_5kb_competitors.csv`, `chrho_5kb_vs_competitor.csv`):

| block | source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|---|
| C.Origami: IMR-90, chr2, chr3 | all12refin_drop_chrho | 0.882 | 0.904 | 0.884 | 0.936 | 0.547 | 0.302 | 0.806 | 0.964 | 0.909 |
| C.Origami: IMR-90, chr2, chr3 | all12honest_drop_chrho | 0.756 | 0.788 | 0.760 | 0.834 | 0.462 | 0.177 | 0.558 | 0.893 | 0.769 |
| C.Origami: IMR-90, chr2, chr3 | all12refres_drop_chrho | 0.878 | 0.899 | 0.880 | 0.928 | 0.555 | 0.319 | 0.954 | 0.961 | 0.906 |
| C.Origami: IMR-90, chr2, chr3 | consensus | 0.819 | 0.853 | 0.815 | 0.803 | 0.581 | 0.339 | 0.964 | 0.902 | 0.815 |
| C.Origami: IMR-90, chr2, chr3 | corigami_imr90_8kb | 0.849 | 0.846 | 0.854 | 0.852 | 0.348 | 0.164 | 0.466 | 0.956 | 0.872 |
| Epiphany: GM12878, chr3, 11, 17 | all12refin_drop_chrho | 0.894 | 0.908 | 0.891 | 0.889 | 0.577 | 0.430 | 0.653 | 0.953 | 0.919 |
| Epiphany: GM12878, chr3, 11, 17 | all12honest_drop_chrho | 0.737 | 0.752 | 0.734 | 0.726 | 0.472 | 0.222 | 0.403 | 0.844 | 0.762 |
| Epiphany: GM12878, chr3, 11, 17 | all12refres_drop_chrho | 0.887 | 0.896 | 0.881 | 0.864 | 0.566 | 0.424 | 0.762 | 0.947 | 0.912 |
| Epiphany: GM12878, chr3, 11, 17 | consensus | 0.793 | 0.820 | 0.790 | 0.784 | 0.560 | 0.416 | 0.936 | 0.876 | 0.831 |
| Epiphany: GM12878, chr3, 11, 17 | epiphany_gm12878_10kb | 0.523 | 0.612 | 0.504 | 0.310 | 0.133 | 0.043 | 0.142 | 0.770 | 0.634 |
| Hi-cGAN: GM12878, chr2, chr14 | all12refin_drop_chrho | 0.882 | 0.902 | 0.882 | 0.886 | 0.540 | 0.337 | 0.668 | 0.954 | 0.918 |
| Hi-cGAN: GM12878, chr2, chr14 | all12honest_drop_chrho | 0.734 | 0.762 | 0.740 | 0.737 | 0.459 | 0.161 | 0.418 | 0.881 | 0.768 |
| Hi-cGAN: GM12878, chr2, chr14 | all12refres_drop_chrho | 0.873 | 0.893 | 0.873 | 0.851 | 0.547 | 0.318 | 0.751 | 0.953 | 0.907 |
| Hi-cGAN: GM12878, chr2, chr14 | consensus | 0.791 | 0.819 | 0.792 | 0.759 | 0.538 | 0.319 | 0.930 | 0.882 | 0.831 |
| Hi-cGAN: GM12878, chr2, chr14 | hicgan_gm12878_5kb | 0.346 | 0.448 | 0.353 | 0.397 | 0.124 | 0.068 | 0.235 | 0.638 | 0.440 |
| HiC-Reg: GM12878, chr14, chr17 | all12refin_drop_chrho | 0.906 | 0.911 | 0.902 | 0.898 | 0.562 | 0.363 | 0.675 | 0.957 | 0.923 |
| HiC-Reg: GM12878, chr14, chr17 | all12honest_drop_chrho | 0.749 | 0.758 | 0.749 | 0.741 | 0.455 | 0.188 | 0.385 | 0.871 | 0.770 |
| HiC-Reg: GM12878, chr14, chr17 | all12refres_drop_chrho | 0.899 | 0.899 | 0.894 | 0.873 | 0.568 | 0.349 | 0.772 | 0.954 | 0.919 |
| HiC-Reg: GM12878, chr14, chr17 | consensus | 0.821 | 0.842 | 0.817 | 0.798 | 0.556 | 0.354 | 0.946 | 0.885 | 0.853 |
| HiC-Reg: GM12878, chr14, chr17 | hicregmulticellcrosschrom_gm12878_5kb | 0.626 | 0.519 | 0.571 | 0.525 | 0.195 | 0.000 | 0.035 | 0.890 | 0.611 |
| HiC-Reg: GM12878, chr14, chr17 | hicregwindowcrosschrom_gm12878_5kb | 0.603 | 0.520 | 0.554 | 0.515 | 0.161 | 0.000 | 0.045 | 0.891 | 0.593 |
| HiC-Reg: GM12878, chr14, chr17 | hicregwindowcv_gm12878_5kb | 0.893 | 0.898 | 0.859 | 0.647 | 0.252 | 0.020 | 0.398 | 0.992 | 0.922 |
| HiC-Reg: K562, chr14, chr17 | all12refin_drop_chrho | 0.771 | 0.817 | 0.776 | 0.842 | 0.510 | 0.338 | 0.718 | 0.925 | 0.814 |
| HiC-Reg: K562, chr14, chr17 | all12honest_drop_chrho | 0.641 | 0.641 | 0.653 | 0.730 | 0.415 | 0.169 | 0.283 | 0.831 | 0.664 |
| HiC-Reg: K562, chr14, chr17 | all12refres_drop_chrho | 0.784 | 0.812 | 0.789 | 0.869 | 0.509 | 0.338 | 0.689 | 0.922 | 0.829 |
| HiC-Reg: K562, chr14, chr17 | consensus | 0.673 | 0.777 | 0.685 | 0.810 | 0.505 | 0.323 | 0.818 | 0.875 | 0.727 |
| HiC-Reg: K562, chr14, chr17 | hicregmulticellcrosschrom_k562_5kb | 0.783 | 0.716 | 0.768 | 0.640 | 0.271 | 0.003 | 0.242 | 0.962 | 0.810 |
| HiC-Reg: K562, chr14, chr17 | hicregwindowcrosschrom_k562_5kb | 0.774 | 0.703 | 0.755 | 0.599 | 0.226 | 0.003 | 0.261 | 0.959 | 0.790 |
| HiC-Reg: K562, chr14, chr17 | hicregwindowcv_k562_5kb | 0.845 | 0.814 | 0.840 | 0.728 | 0.282 | 0.005 | 0.441 | 0.978 | 0.899 |

Replicate ceilings at 5 kb are low (IMR-90 Pearson 0.670, GM12878 0.69, K562
0.706; the single-lane replicate maps are sparse at 5 kb) and are not
ceilings of what a model can reach against the deep target.

Verdict per competitor at 5 kb (ours = all12refin_drop_chrho, ChIP-only =
all12honest_drop_chrho; "level" within 0.01; win counts per chromosome in
`chrho_5kb_vs_competitor.csv`):

- C.Origami (IMR-90, chr2, chr3, up-sampled from 8192 bp). The reference-map
  model is ahead on Pearson (0.882 against 0.849), tile median (0.904 against
  0.846), SCC (0.884 against 0.854), insulation r (0.936 against 0.852),
  boundary recall (0.547 against 0.348), loop recall (0.302 against 0.164) and
  HiCRep (0.909 against 0.872), on both chromosomes each, and level on
  compartment r (0.964 against 0.956). The ChIP-only model is behind on
  Pearson (0.756), tile median, SCC, insulation r, compartment r and HiCRep,
  ahead on boundary recall (0.462) and, on 1 of 2 chromosomes, loop recall
  (0.177 against 0.164). At 25 kb the reference-map model was level on
  Pearson against C.Origami (0.940 against 0.942); at 5 kb it is ahead (0.882
  against 0.849), C.Origami's score falling by 0.09 from 25 to 5 kb and ours by 0.06.
- Epiphany (GM12878, chr3, 11, 17, up-sampled from 10 kb). Ahead on every
  measure on all three chromosomes (Pearson 0.894 against 0.523, SCC 0.891
  against 0.504, insulation 0.889 against 0.310, loop recall 0.430 against
  0.043); the ChIP-only model is ahead on all measures too (Pearson 0.737).
- Hi-cGAN (GM12878, chr2, chr14; native 5 kb). Ahead on every measure on both
  chromosomes: Pearson 0.882 (ChIP only 0.734) against 0.346, insulation r
  0.886 against 0.397, loop recall 0.337 against 0.068.
- HiC-Reg cross-chromosome (leak-free; native 5 kb). GM12878: the
  reference-map model is ahead on every measure (Pearson 0.906 against 0.603
  WINDOW and 0.626 MULTICELL); the ChIP-only model too, except compartment r
  (0.871 against 0.891). K562: Pearson 0.771 against 0.774 WINDOW (level) and
  0.783 MULTICELL (behind by 0.012); ahead on tile median (0.817 against 0.703
  and 0.716), insulation, boundary and loop recall, and behind on compartment
  r (0.925 against 0.959 and 0.962); SCC and HiCRep are level or ahead on
  1 of 2 chromosomes. The ChIP-only model is behind on Pearson (0.641), SCC,
  tile median, compartment r and HiCRep, ahead on boundary and loop recall.
- HiC-Reg within-chromosome CV maps (not a held-out score). GM12878: ahead of
  CV on Pearson (0.906 against 0.893), SCC, tile median, insulation r,
  boundary and loop recall, level on HiCRep, behind on compartment r (0.957
  against 0.992). K562: behind on Pearson (0.771 against 0.845), SCC and HiCRep,
  level on tile median, ahead on insulation r (0.842 against 0.728), boundary
  and loop recall, behind on compartment r (0.925 against 0.978).
- The refres model is within 0.01 of refin on Pearson in 4 of 5 blocks (K562
  HiC-Reg block: 0.784 against 0.771).

Limits (5 kb). (i) C.Origami and Epiphany have no 5 kb output: the maps are
up-sampled, the competitors' 5 kb insulation, boundary and loop scores reflect
a resolution they never produced, and their margins to our native 5 kb output
are partly a resolution effect. (ii) Few chromosomes per block (2 to 3), 11
chromosome-blocks in total; win counts are per chromosome. (iii) Loop recall,
APA and compartment r are noisier at 5 kb: the 5 kb maps are sparser, the
reference loops are the 10 kb calls (a 5 kb call set does not exist), the loop
matching tolerance is 10 kb in bp and the C.Origami APA ratio (0.47) and
the HiC-Reg APA ratios (0.04 to 0.44) are far from 1 while the reference-map
model gives 0.65 to 0.81. (iv) Leakage status is that of the 25 kb section:
C.Origami chr2 and chr3, Epiphany chr3, 11, 17, HiC-Reg cross-chromosome
chr14 and chr17 are unseen by the competitor and by our models; HiC-Reg CV is
within-chromosome interpolation; Hi-cGAN chr2, chr14 are even chromosomes (held
out by Hi-cGAN). Hi-cGAN has no 5 kb K562 output. (v) The reference-map
model reads the leave-one-out consensus of the other cells' 5 kb maps as
input; the ChIP-only model reads none. (vi) Competitors differ in training
data and inputs from the 12-cell models (not an architecture comparison).
(vii) The "replicate" and "best_other_map" rows are weak baselines at 5 kb
(sparse single-lane maps); the consensus of the other cells has a higher boundary
and loop recall than the reference-map model in the C.Origami block (0.581 and
0.339 against 0.547 and 0.302).

Prediction used a local GPU for inference only; the GPU recipe and the
estimates of `results/benchmark/REMOTE_JOB_chrho_5kb.md` are unchanged.

Reproduce (hicdiff env, `BENCH_5KB=1` registers the 5 kb competitors):

    BENCH_5KB=1 python scripts/benchmark_competitors.py --stage oe --names hicgan_gm12878_5kb
    BENCH_5KB=1 python scripts/hicreg_audit.py --stage cools
    python scripts/hic_measures.py --stage chrho_predict --chrho-res 5000 --gpu 1
    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-res 5000 --jobs 2 --threads 4
    python scripts/chrho_5kb_table.py

Files: `results/benchmark/chrho_25kb_competitors.csv` (every source per block,
every measure), `chrho_25kb_vs_competitor.csv` (differences, win counts,
verdicts), `chrho_25kb_table.md`; `results/hic_measures_chrho/chrho_25kb_summary.csv`
and `chrho_25kb_per_block.csv` (the scorer's output, 22 blocks).
Reproduce (hicdiff env):

    BENCH_HICGAN_IMR90_25KB=1 python scripts/benchmark_competitors.py --stage oe --names hicgan_imr90_on_imr90_25kb hicgan_gm12878_on_imr90_25kb
    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-25kb-comps --jobs 1 --threads 4
    python scripts/chrho_25kb_table.py


## 59. Why the sequence + methylation CNN fails on K562 and what fixes it (FINAL; section 58 was the last number when this was written)

Question. In section 42 the sequence + methylation CNN (`seq_meth`) beats gradient
boosting on the seven training cells but fails on K562 and HMEC, which are in no
training set (unseen_cell R2 -0.2 to -24 for most marks; K562 median Pearson 0.34
for `seq_meth` and -0.20 for `meth`; HMEC 0.74 and 0.67), while sequence alone
generalises (R2 0.12 to 0.53) and the gradient-boosting analysis of section 41
reaches R2 up to 0.49. Which input property causes this and does a change of the
Meth input fix it.

Setup. Section 42 model, split, targets and budget unchanged (9.3 M parameters,
20000 steps, batch 32, one seed, one RTX 4090, about 33 minutes of training and 4
minutes of evaluation per variant). Test: chr6, 8, ..., 22 of the nine cells.
Independent cells on unseen_cell: 2 (K562, HMEC); on seen_cell: 7. Median
"unseen" values below are the median of the two cells (for RAD21 and SMC3 only
K562 has the track, so that median is one cell). Gradient boosting (`GB_both`) is
the `both` set of section 41 (extended sequence features + methylation features,
`partition_fits.csv`, unseen_cell), scored on all even chromosomes (including
chr2 and chr4, 500,000 training rows); `ceiling_loo` is the across-cell mean of
the seven training cells scored against the cell (a cell-agnostic prediction, the
best `seq` can reach).

### Diagnosis (CPU for the distributions, GPU only to dump predictions of the saved section 42 checkpoints)

Meth input (value as the script uses it: methylated fraction of 1 kb bins inside
valid 10 kb bins, all chromosomes):

| cell | mean | sd | q05 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|---|
| gm12878 | 0.446 | 0.299 | 0.01 | 0.20 | 0.41 | 0.68 | 0.96 |
| imr90 | 0.691 | 0.216 | 0.25 | 0.65 | 0.66 | 0.86 | 1.00 |
| h1 | 0.855 | 0.129 | 0.65 | 0.83 | 0.89 | 0.92 | 0.96 |
| hepg2 | 0.397 | 0.341 | 0.00 | 0.08 | 0.32 | 0.70 | 0.97 |
| a549 | 0.512 | 0.311 | 0.02 | 0.23 | 0.56 | 0.81 | 0.93 |
| gm23248 | 0.600 | 0.284 | 0.10 | 0.36 | 0.63 | 0.86 | 0.97 |
| cd14mono | 0.833 | 0.177 | 0.53 | 0.73 | 0.88 | 0.96 | 1.00 |
| k562 (unseen) | 0.219 | 0.307 | 0.00 | 0.00 | 0.03 | 0.35 | 0.89 |
| hmec (unseen) | 0.725 | 0.221 | 0.27 | 0.61 | 0.77 | 0.90 | 0.98 |

The training cells span a genome-wide mean methylation of 0.40 (HepG2) to 0.86
(H1). K562 has 0.22 and half of its bins below 0.03, outside the training range
and in the corner of the input space the model never saw (the constant mean
channel is 0.22 against a training minimum of 0.40). HMEC (0.73) lies inside the
range. The z-scored targets (training-cell statistics, valid bins, all
chromosomes) have a per-cell, per-mark mean within +-0.55 of 0 and a standard
deviation of 0.5 to 1.6 in all cells including K562 (e.g. K562 H3K27ac +0.20 /
0.67, H3K4me3 +0.09 / 0.86, H3K9ac -0.05 / 0.98); the only larger offsets are
seen cells (IMR-90 H3K4me3 -0.89). The target scale of K562 and HMEC is thus
not unusual.

Predictions against targets on the test chromosomes (checkpoints of section 42;
median over marks; `results/seq_cnn/v2/diag_prediction_bias.csv`, columns y_mean,
p_mean, slope of y on the prediction, R2 with the offset removed):

| model, cells | abs. offset p_mean - y_mean | sd of prediction (sd of target 0.9) | slope | R2 | R2 after removing the offset | Pearson |
|---|---|---|---|---|---|---|
| seq_meth, 7 seen | 0.01 | 0.74 | 0.99 | 0.68 | 0.68 | 0.83 |
| seq_meth, HMEC | 0.27 | 0.81 | 0.95 | 0.41 | 0.55 | 0.74 |
| seq_meth, K562 | 1.68 | 0.90 | 0.25 | -3.93 | -1.64 | 0.34 |
| meth, 7 seen | 0.02 | 0.70 | 0.97 | 0.58 | 0.59 | 0.77 |
| meth, HMEC | 0.29 | 0.85 | 0.70 | 0.27 | 0.40 | 0.67 |
| meth, K562 | 5.42 | 2.85 | -0.06 | -60.65 | -16.73 | -0.20 |

Per mark in K562 `seq_meth` predicts offsets up to +5.6 z (H3K4me2, H3K4me3,
H3K9ac) with prediction sd up to 2.3; `meth` predicts +12 z for H3K4me2. Removing
the offset still leaves R2 below zero, so K562 has both a level error and a
ranking error (Pearson -0.2 to 0.6 across marks), and the prediction slope of 0.25
means the predictions are not a rescaled copy of the truth.

Verdict. Hypothesis 1 (Meth input of the unseen cell outside the training
distribution) is supported for K562: the failure grows with the distance of the
input from the training range (HMEC inside the range works, Pearson 0.74; K562
outside fails; the methylation-only control, which has no sequence to fall back
on, explodes to offsets of 5 to 12 z). Hypothesis 3 (target scale per cell) is not
supported: the target means and sds of K562 and HMEC are ordinary; the offset is
produced by the model, so `--target-norm cell` was implemented but not trained.
Hypothesis 2 (memorisation of training cells by their methylation fingerprint) is
weak: removing the cell-mean channel and the per-cell scale (`cellnorm`) leaves
the seen-cell results unchanged (Pearson 0.82 against 0.83) and does not by itself
fix K562.

### Fix: flags of `scripts/seq_cnn_chip.py` (defaults keep section 42 reproducible; variant names carry a suffix)

- `--meth-norm cell`: the Meth channel is z-scored with the cell's own mean and sd
  over its valid bins (statistics from the cell's Meth track only, no ChIP, no
  label); the constant mean-methylation channel is set to 0. `--meth-norm rank`
  does the same by quantile normalisation to a standard normal (implemented,
  smoke-tested, not trained). Variant `*_cellnorm`.
- `--meth-aug`: training only, random per-window scale U(0.5, 1.5) and shift
  N(0, 0.5) of the two Meth channels (`--meth-aug-scale`, `--meth-aug-shift`).
  Variant `*_aug`.
- `--meth-dropout P`: training only, a whole window gets zeros in both Meth channels
  with probability P, so the sequence path stays usable. Variant `*_mdropP`.
- `--target-norm cell`: per-cell, per-mark z-scoring of the targets; test metrics are
  computed after standardising prediction and target per mark and cell, so R2 is
  then scale free and Pearson and Spearman stay the main metrics. Variant `*_tcell`,
  ceilings `ceiling_*_tcell`. Smoke-tested only.
- `stage_summarize` keeps every variant (old rows unchanged); the partition table
  still uses `seq`, `seq_meth`, `meth`. CPU smoke tests of all new options passed
  (`--smoke`, chr21 train, chr22 test).

Commands (remote GPU; `OUT_ROOT` = the section 42 run folder with its cache):

    P="python scripts/seq_cnn_chip.py --config configs/seq_cnn.yaml --stage train eval"
    $P --mode seq_meth --meth-norm cell
    $P --mode meth     --meth-norm cell
    $P --mode seq_meth --meth-aug
    $P --mode seq_meth --meth-norm cell --meth-aug --meth-dropout 0.2
    python scripts/seq_cnn_chip.py --config configs/seq_cnn.yaml --stage summarize --results-dir results/seq_cnn/v2

Wall time on the RTX 4090: 4 variants x (33 min training + 4 min evaluation) = 2 h 12 min
(08:22 to 10:34 on 2026-10-10), plus 5 min per prediction dump.

### Results

Summary (median over marks of the per-mark medians over cells; full per-mark
values below and in `results/seq_cnn/v2/comparison_by_mark.csv`):

| cells | metric | seq | meth | seq_meth | meth_cellnorm | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|---|---|---|
| unseen (K562, HMEC) | R2 | 0.24 | -31.41 | -3.69 | -0.64 | -0.70 | 0.34 | -0.24 | 0.12 | 0.61 |
| unseen | Pearson | 0.57 | 0.20 | 0.43 | 0.51 | 0.58 | 0.73 | 0.70 | 0.64 | 0.82 |
| unseen | Spearman | 0.50 | 0.11 | 0.31 | 0.47 | 0.41 | 0.60 | 0.49 | 0.51 | 0.64 |
| K562 | R2 | 0.40 | -60.65 | -3.93 | -0.90 | -1.45 | 0.41 | -0.53 | 0.16 | 0.62 |
| K562 | Pearson | 0.66 | -0.20 | 0.34 | 0.43 | 0.41 | 0.73 | 0.62 | 0.60 | 0.81 |
| K562 | Spearman | 0.54 | -0.27 | 0.00 | 0.43 | 0.35 | 0.63 | 0.53 | 0.47 | 0.64 |
| HMEC | R2 | 0.23 | 0.27 | 0.41 | 0.45 | 0.53 | 0.52 | 0.51 | 0.36 | 0.59 |
| HMEC | Pearson | 0.58 | 0.67 | 0.74 | 0.69 | 0.75 | 0.72 | 0.74 | 0.64 | 0.79 |
| HMEC | Spearman | 0.44 | 0.61 | 0.63 | 0.57 | 0.61 | 0.64 | 0.64 | 0.51 | 0.67 |
| seen (7 cells) | R2 | 0.35 | 0.60 | 0.69 | 0.57 | 0.67 | 0.64 | 0.62 | 0.46 | 0.60 |
| seen | Pearson | 0.63 | 0.78 | 0.83 | 0.76 | 0.82 | 0.81 | 0.79 | 0.69 | 0.82 |
| seen | Spearman | 0.57 | 0.69 | 0.75 | 0.67 | 0.74 | 0.72 | 0.72 | 0.61 | 0.69 |

Prediction offset and slope after the fix (same diagnostic, median over marks):

| model | cells | abs. offset | slope | R2 | R2 after removing the offset | Pearson |
|---|---|---|---|---|---|---|
| seq_meth_aug | K562 | 0.20 | 0.98 | 0.41 | 0.47 | 0.73 |
| seq_meth_aug | HMEC | 0.05 | 0.93 | 0.52 | 0.52 | 0.72 |
| seq_meth_cellnorm | K562 | 0.88 | 0.52 | -1.45 | 0.03 | 0.41 |
| seq_meth_cellnorm | HMEC | 0.09 | 0.84 | 0.53 | 0.54 | 0.75 |
| meth_cellnorm | K562 | 0.74 | 0.60 | -0.90 | 0.09 | 0.43 |
| seq_meth_cellnorm_aug_mdrop0.2 | K562 | 0.60 | 0.51 | -0.53 | 0.02 | 0.62 |

Per-mark tables (median over cells; blank = mark absent in the cell set):

**unseen_median2, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.19 | -0.19 | -0.97 | 0.28 | -0.24 | 0.12 | 0.68 |
| H3K27ac | 0.23 | -7.07 | -0.14 | 0.36 | 0.07 | 0.15 | 0.39 |
| H3K27me3 | 0.24 | -1.75 | -1.07 | 0.39 | -0.29 | 0.28 | 0.47 |
| H3K36me3 | 0.38 | 0.16 | 0.36 | 0.60 | 0.57 | 0.54 | 0.69 |
| H3K4me1 | 0.21 | -5.19 | -0.70 | 0.09 | -0.30 | -0.33 | 0.38 |
| H3K4me2 | 0.53 | -18.02 | 0.52 | 0.74 | 0.67 | 0.59 | 0.69 |
| H3K4me3 | 0.24 | -23.92 | -0.56 | 0.22 | -0.04 | -0.26 | 0.38 |
| H3K79me2 | 0.43 | -0.67 | 0.53 | 0.58 | 0.55 | 0.39 | 0.73 |
| H3K9ac | 0.45 | -15.80 | 0.34 | 0.69 | 0.58 | 0.48 | 0.64 |
| H3K9me3 | 0.12 | -8.96 | -1.95 | 0.02 | -0.82 | -0.34 | 0.28 |
| H4K20me1 | 0.13 | -0.46 | -0.97 | 0.07 | -0.38 | -0.09 | 0.32 |
| RAD21 | 0.30 | -1.38 | -3.89 | 0.15 | -1.58 | -0.44 | 0.61 |
| SMC3 | 0.44 | -3.69 | -2.94 | 0.34 | -1.02 | -0.04 | 0.62 |
| median over marks | 0.24 | -3.69 | -0.70 | 0.34 | -0.24 | 0.12 | 0.61 |

**unseen_median2, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.46 | 0.44 | 0.40 | 0.56 | 0.48 | 0.48 | 0.83 |
| H3K27ac | 0.56 | 0.37 | 0.73 | 0.77 | 0.75 | 0.67 | 0.71 |
| H3K27me3 | 0.53 | 0.24 | 0.49 | 0.66 | 0.49 | 0.57 | 0.71 |
| H3K36me3 | 0.62 | 0.68 | 0.74 | 0.80 | 0.79 | 0.78 | 0.84 |
| H3K4me1 | 0.57 | 0.70 | 0.71 | 0.73 | 0.73 | 0.64 | 0.74 |
| H3K4me2 | 0.74 | 0.44 | 0.79 | 0.87 | 0.86 | 0.79 | 0.83 |
| H3K4me3 | 0.72 | 0.40 | 0.58 | 0.80 | 0.70 | 0.65 | 0.84 |
| H3K79me2 | 0.68 | 0.60 | 0.77 | 0.80 | 0.80 | 0.66 | 0.87 |
| H3K9ac | 0.70 | 0.35 | 0.78 | 0.85 | 0.83 | 0.75 | 0.82 |
| H3K9me3 | 0.43 | 0.33 | 0.35 | 0.51 | 0.41 | 0.37 | 0.66 |
| H4K20me1 | 0.56 | 0.43 | 0.41 | 0.58 | 0.46 | 0.38 | 0.70 |
| RAD21 | 0.57 | 0.43 | 0.35 | 0.64 | 0.49 | 0.49 | 0.82 |
| SMC3 | 0.67 | 0.34 | 0.29 | 0.66 | 0.45 | 0.44 | 0.79 |
| median over marks | 0.57 | 0.43 | 0.58 | 0.73 | 0.70 | 0.64 | 0.82 |

**unseen_median2, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.28 | 0.31 | 0.27 | 0.42 | 0.36 | 0.30 | 0.56 |
| H3K27ac | 0.48 | 0.22 | 0.67 | 0.69 | 0.68 | 0.55 | 0.64 |
| H3K27me3 | 0.50 | 0.19 | 0.41 | 0.60 | 0.39 | 0.51 | 0.69 |
| H3K36me3 | 0.49 | 0.51 | 0.68 | 0.66 | 0.69 | 0.59 | 0.68 |
| H3K4me1 | 0.56 | 0.65 | 0.68 | 0.64 | 0.68 | 0.63 | 0.72 |
| H3K4me2 | 0.54 | 0.34 | 0.59 | 0.67 | 0.61 | 0.52 | 0.65 |
| H3K4me3 | 0.37 | 0.23 | 0.31 | 0.50 | 0.43 | 0.20 | 0.52 |
| H3K79me2 | 0.65 | 0.40 | 0.70 | 0.69 | 0.69 | 0.54 | 0.73 |
| H3K9ac | 0.54 | 0.16 | 0.62 | 0.68 | 0.66 | 0.55 | 0.66 |
| H3K9me3 | 0.44 | 0.30 | 0.30 | 0.45 | 0.38 | 0.40 | 0.64 |
| H4K20me1 | 0.51 | 0.35 | 0.30 | 0.47 | 0.35 | 0.25 | 0.63 |
| RAD21 | 0.46 | 0.35 | 0.35 | 0.53 | 0.49 | 0.43 | 0.64 |
| SMC3 | 0.63 | 0.23 | 0.24 | 0.57 | 0.44 | 0.34 | 0.63 |
| median over marks | 0.50 | 0.31 | 0.41 | 0.60 | 0.49 | 0.51 | 0.64 |

**seen_median7, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.28 | 0.44 | 0.43 | 0.43 | 0.43 | 0.38 | 0.67 |
| H3K27ac | 0.28 | 0.69 | 0.67 | 0.66 | 0.61 | 0.47 | 0.37 |
| H3K27me3 | 0.35 | 0.65 | 0.65 | 0.62 | 0.62 | 0.46 | 0.45 |
| H3K36me3 | 0.39 | 0.73 | 0.73 | 0.68 | 0.68 | 0.55 | 0.60 |
| H3K4me1 | 0.25 | 0.70 | 0.69 | 0.66 | 0.66 | 0.45 | 0.37 |
| H3K4me2 | 0.48 | 0.82 | 0.80 | 0.79 | 0.78 | 0.61 | 0.60 |
| H3K4me3 | 0.57 | 0.87 | 0.84 | 0.82 | 0.83 | 0.62 | 0.68 |
| H3K79me2 | 0.43 | 0.70 | 0.69 | 0.64 | 0.65 | 0.50 | 0.68 |
| H3K9ac | 0.51 | 0.82 | 0.78 | 0.77 | 0.76 | 0.67 | 0.65 |
| H3K9me3 | 0.19 | 0.52 | 0.50 | 0.45 | 0.45 | 0.29 | 0.39 |
| H4K20me1 | 0.34 | 0.61 | 0.58 | 0.53 | 0.53 | 0.35 | 0.54 |
| RAD21 | 0.34 | 0.53 | 0.53 | 0.53 | 0.52 | 0.42 | 0.72 |
| SMC3 | 0.39 | 0.62 | 0.62 | 0.57 | 0.57 | 0.39 | 0.58 |
| median over marks | 0.35 | 0.69 | 0.67 | 0.64 | 0.62 | 0.46 | 0.60 |

**seen_median7, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.54 | 0.67 | 0.66 | 0.67 | 0.66 | 0.64 | 0.86 |
| H3K27ac | 0.59 | 0.83 | 0.82 | 0.81 | 0.78 | 0.69 | 0.70 |
| H3K27me3 | 0.60 | 0.81 | 0.80 | 0.79 | 0.79 | 0.71 | 0.72 |
| H3K36me3 | 0.69 | 0.86 | 0.85 | 0.83 | 0.83 | 0.76 | 0.83 |
| H3K4me1 | 0.56 | 0.84 | 0.83 | 0.82 | 0.81 | 0.69 | 0.66 |
| H3K4me2 | 0.70 | 0.90 | 0.90 | 0.89 | 0.89 | 0.79 | 0.80 |
| H3K4me3 | 0.80 | 0.93 | 0.92 | 0.91 | 0.91 | 0.80 | 0.87 |
| H3K79me2 | 0.66 | 0.84 | 0.84 | 0.81 | 0.81 | 0.71 | 0.85 |
| H3K9ac | 0.74 | 0.90 | 0.89 | 0.88 | 0.88 | 0.82 | 0.85 |
| H3K9me3 | 0.50 | 0.72 | 0.71 | 0.68 | 0.68 | 0.56 | 0.66 |
| H4K20me1 | 0.64 | 0.78 | 0.77 | 0.74 | 0.74 | 0.60 | 0.77 |
| RAD21 | 0.60 | 0.73 | 0.73 | 0.73 | 0.72 | 0.67 | 0.86 |
| SMC3 | 0.63 | 0.79 | 0.79 | 0.78 | 0.78 | 0.68 | 0.82 |
| median over marks | 0.63 | 0.83 | 0.82 | 0.81 | 0.79 | 0.69 | 0.82 |

**seen_median7, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.43 | 0.57 | 0.58 | 0.58 | 0.58 | 0.47 | 0.62 |
| H3K27ac | 0.63 | 0.78 | 0.77 | 0.73 | 0.73 | 0.61 | 0.63 |
| H3K27me3 | 0.60 | 0.78 | 0.77 | 0.75 | 0.75 | 0.68 | 0.69 |
| H3K36me3 | 0.57 | 0.80 | 0.76 | 0.73 | 0.73 | 0.60 | 0.68 |
| H3K4me1 | 0.60 | 0.82 | 0.81 | 0.80 | 0.79 | 0.68 | 0.71 |
| H3K4me2 | 0.57 | 0.77 | 0.76 | 0.75 | 0.73 | 0.61 | 0.64 |
| H3K4me3 | 0.39 | 0.59 | 0.61 | 0.61 | 0.60 | 0.40 | 0.51 |
| H3K79me2 | 0.62 | 0.75 | 0.74 | 0.72 | 0.72 | 0.62 | 0.70 |
| H3K9ac | 0.68 | 0.82 | 0.81 | 0.82 | 0.81 | 0.66 | 0.77 |
| H3K9me3 | 0.56 | 0.68 | 0.68 | 0.65 | 0.66 | 0.54 | 0.71 |
| H4K20me1 | 0.64 | 0.67 | 0.69 | 0.64 | 0.64 | 0.51 | 0.69 |
| RAD21 | 0.54 | 0.70 | 0.70 | 0.69 | 0.68 | 0.61 | 0.71 |
| SMC3 | 0.55 | 0.69 | 0.69 | 0.65 | 0.66 | 0.53 | 0.65 |
| median over marks | 0.57 | 0.75 | 0.74 | 0.72 | 0.72 | 0.61 | 0.69 |

**k562, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.32 | -0.54 | -2.12 | 0.35 | -0.63 | 0.08 | 0.72 |
| H3K27ac | 0.23 | -14.54 | -0.80 | 0.18 | -0.36 | -0.13 | 0.30 |
| H3K27me3 | 0.28 | -3.93 | -2.50 | 0.41 | -0.99 | 0.31 | 0.42 |
| H3K36me3 | 0.43 | -0.21 | 0.18 | 0.68 | 0.56 | 0.57 | 0.72 |
| H3K4me1 | 0.08 | -10.79 | -1.93 | -0.35 | -1.13 | -1.13 | 0.17 |
| H3K4me2 | 0.54 | -36.60 | 0.35 | 0.78 | 0.66 | 0.51 | 0.65 |
| H3K4me3 | 0.63 | -45.66 | -0.91 | 0.62 | 0.04 | 0.47 | 0.77 |
| H3K79me2 | 0.40 | -1.92 | 0.47 | 0.63 | 0.56 | 0.43 | 0.69 |
| H3K9ac | 0.53 | -31.97 | 0.07 | 0.73 | 0.50 | 0.51 | 0.68 |
| H3K9me3 | 0.08 | -18.36 | -4.20 | -0.25 | -1.97 | -0.89 | 0.07 |
| H4K20me1 | 0.45 | -0.20 | -1.45 | 0.42 | -0.53 | 0.16 | 0.52 |
| RAD21 | 0.30 | -1.38 | -3.89 | 0.15 | -1.58 | -0.44 | 0.61 |
| SMC3 | 0.44 | -3.69 | -2.94 | 0.34 | -1.02 | -0.04 | 0.62 |
| median over marks | 0.40 | -3.93 | -1.45 | 0.41 | -0.53 | 0.16 | 0.62 |

**k562, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.59 | 0.43 | 0.35 | 0.64 | 0.50 | 0.53 | 0.86 |
| H3K27ac | 0.62 | 0.00 | 0.73 | 0.81 | 0.78 | 0.69 | 0.71 |
| H3K27me3 | 0.57 | -0.20 | 0.37 | 0.70 | 0.33 | 0.60 | 0.70 |
| H3K36me3 | 0.67 | 0.59 | 0.72 | 0.87 | 0.81 | 0.83 | 0.86 |
| H3K4me1 | 0.54 | 0.61 | 0.66 | 0.73 | 0.72 | 0.59 | 0.70 |
| H3K4me2 | 0.74 | 0.02 | 0.73 | 0.90 | 0.87 | 0.76 | 0.81 |
| H3K4me3 | 0.79 | 0.08 | 0.41 | 0.82 | 0.62 | 0.69 | 0.88 |
| H3K79me2 | 0.66 | 0.42 | 0.74 | 0.83 | 0.80 | 0.68 | 0.84 |
| H3K9ac | 0.73 | -0.09 | 0.77 | 0.89 | 0.85 | 0.77 | 0.82 |
| H3K9me3 | 0.46 | -0.04 | 0.03 | 0.47 | 0.22 | 0.28 | 0.61 |
| H4K20me1 | 0.67 | 0.44 | 0.36 | 0.67 | 0.41 | 0.45 | 0.73 |
| RAD21 | 0.57 | 0.43 | 0.35 | 0.64 | 0.49 | 0.49 | 0.82 |
| SMC3 | 0.67 | 0.34 | 0.29 | 0.66 | 0.45 | 0.44 | 0.79 |
| median over marks | 0.66 | 0.34 | 0.41 | 0.73 | 0.62 | 0.60 | 0.81 |

**k562, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.54 | 0.41 | 0.32 | 0.59 | 0.53 | 0.47 | 0.66 |
| H3K27ac | 0.48 | -0.29 | 0.63 | 0.67 | 0.63 | 0.48 | 0.60 |
| H3K27me3 | 0.60 | -0.24 | 0.33 | 0.64 | 0.23 | 0.56 | 0.71 |
| H3K36me3 | 0.50 | 0.26 | 0.61 | 0.63 | 0.63 | 0.47 | 0.63 |
| H3K4me1 | 0.50 | 0.51 | 0.61 | 0.62 | 0.64 | 0.59 | 0.65 |
| H3K4me2 | 0.65 | -0.02 | 0.51 | 0.68 | 0.54 | 0.50 | 0.64 |
| H3K4me3 | 0.41 | -0.01 | 0.12 | 0.37 | 0.25 | 0.22 | 0.49 |
| H3K79me2 | 0.57 | -0.00 | 0.66 | 0.66 | 0.62 | 0.37 | 0.68 |
| H3K9ac | 0.61 | -0.29 | 0.63 | 0.73 | 0.67 | 0.59 | 0.70 |
| H3K9me3 | 0.43 | 0.00 | 0.02 | 0.37 | 0.18 | 0.34 | 0.53 |
| H4K20me1 | 0.74 | 0.46 | 0.31 | 0.64 | 0.37 | 0.39 | 0.74 |
| RAD21 | 0.46 | 0.35 | 0.35 | 0.53 | 0.49 | 0.43 | 0.64 |
| SMC3 | 0.63 | 0.23 | 0.24 | 0.57 | 0.44 | 0.34 | 0.63 |
| median over marks | 0.54 | 0.00 | 0.35 | 0.63 | 0.53 | 0.47 | 0.64 |

**hmec, r2**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.05 | 0.16 | 0.18 | 0.21 | 0.15 | 0.17 | 0.63 |
| H3K27ac | 0.23 | 0.40 | 0.53 | 0.54 | 0.51 | 0.43 | 0.49 |
| H3K27me3 | 0.20 | 0.42 | 0.36 | 0.37 | 0.40 | 0.24 | 0.52 |
| H3K36me3 | 0.33 | 0.52 | 0.54 | 0.52 | 0.57 | 0.51 | 0.66 |
| H3K4me1 | 0.34 | 0.41 | 0.53 | 0.52 | 0.54 | 0.47 | 0.59 |
| H3K4me2 | 0.53 | 0.57 | 0.70 | 0.70 | 0.69 | 0.66 | 0.72 |
| H3K4me3 | -0.14 | -2.18 | -0.22 | -0.18 | -0.13 | -1.00 | -0.01 |
| H3K79me2 | 0.46 | 0.58 | 0.60 | 0.54 | 0.54 | 0.36 | 0.78 |
| H3K9ac | 0.38 | 0.38 | 0.61 | 0.66 | 0.66 | 0.44 | 0.60 |
| H3K9me3 | 0.15 | 0.43 | 0.30 | 0.29 | 0.33 | 0.21 | 0.48 |
| H4K20me1 | -0.20 | -0.71 | -0.49 | -0.29 | -0.24 | -0.35 | 0.12 |
| RAD21 |  |  |  |  |  |  |  |
| SMC3 |  |  |  |  |  |  |  |
| median over marks | 0.23 | 0.41 | 0.53 | 0.52 | 0.51 | 0.36 | 0.59 |

**hmec, pearson**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.32 | 0.45 | 0.46 | 0.49 | 0.45 | 0.44 | 0.81 |
| H3K27ac | 0.50 | 0.74 | 0.73 | 0.73 | 0.72 | 0.66 | 0.70 |
| H3K27me3 | 0.49 | 0.68 | 0.62 | 0.63 | 0.64 | 0.54 | 0.73 |
| H3K36me3 | 0.58 | 0.76 | 0.75 | 0.72 | 0.76 | 0.72 | 0.81 |
| H3K4me1 | 0.60 | 0.79 | 0.76 | 0.72 | 0.74 | 0.69 | 0.79 |
| H3K4me2 | 0.73 | 0.85 | 0.84 | 0.84 | 0.84 | 0.81 | 0.86 |
| H3K4me3 | 0.65 | 0.72 | 0.76 | 0.79 | 0.79 | 0.60 | 0.79 |
| H3K79me2 | 0.70 | 0.79 | 0.80 | 0.78 | 0.79 | 0.64 | 0.91 |
| H3K9ac | 0.67 | 0.80 | 0.80 | 0.82 | 0.81 | 0.73 | 0.82 |
| H3K9me3 | 0.40 | 0.70 | 0.67 | 0.54 | 0.61 | 0.46 | 0.70 |
| H4K20me1 | 0.45 | 0.43 | 0.46 | 0.48 | 0.50 | 0.31 | 0.67 |
| RAD21 |  |  |  |  |  |  |  |
| SMC3 |  |  |  |  |  |  |  |
| median over marks | 0.58 | 0.74 | 0.75 | 0.72 | 0.74 | 0.64 | 0.79 |

**hmec, spearman**

| mark | seq | seq_meth | seq_meth_cellnorm | seq_meth_aug | seq_meth_cellnorm_aug_mdrop0.2 | GB_both | ceiling_loo |
|---|---|---|---|---|---|---|---|
| CTCF | 0.03 | 0.21 | 0.22 | 0.25 | 0.20 | 0.13 | 0.45 |
| H3K27ac | 0.49 | 0.73 | 0.70 | 0.71 | 0.72 | 0.62 | 0.68 |
| H3K27me3 | 0.41 | 0.63 | 0.49 | 0.56 | 0.56 | 0.47 | 0.66 |
| H3K36me3 | 0.49 | 0.76 | 0.74 | 0.69 | 0.75 | 0.71 | 0.72 |
| H3K4me1 | 0.62 | 0.79 | 0.74 | 0.66 | 0.72 | 0.67 | 0.79 |
| H3K4me2 | 0.42 | 0.70 | 0.67 | 0.66 | 0.67 | 0.53 | 0.67 |
| H3K4me3 | 0.34 | 0.47 | 0.51 | 0.64 | 0.62 | 0.18 | 0.54 |
| H3K79me2 | 0.72 | 0.80 | 0.74 | 0.73 | 0.75 | 0.72 | 0.79 |
| H3K9ac | 0.47 | 0.61 | 0.61 | 0.64 | 0.64 | 0.51 | 0.62 |
| H3K9me3 | 0.44 | 0.61 | 0.59 | 0.54 | 0.58 | 0.46 | 0.76 |
| H4K20me1 | 0.28 | 0.24 | 0.29 | 0.31 | 0.33 | 0.11 | 0.52 |
| RAD21 |  |  |  |  |  |  |  |
| SMC3 |  |  |  |  |  |  |  |
| median over marks | 0.44 | 0.63 | 0.61 | 0.64 | 0.64 | 0.51 | 0.67 |


### Reading

- The unseen-cell failure is fixed for `seq_meth_aug` (legacy Meth scaling plus
  random per-window scale and shift): median unseen R2 0.34 (from -3.69),
  Pearson 0.73 (from 0.43), Spearman 0.60 (from 0.31); K562 Pearson 0.73 (from
  0.34) and R2 0.41 (from -3.93), prediction offset 0.20 z (from 1.68) and slope
  0.98 (from 0.25). Marks with a negative R2 remain in single cells (K562
  H3K4me1 -0.35 and H3K9me3 -0.25; HMEC H3K4me3 -0.18 and H4K20me1 -0.29), the
  same marks for which `seq` and the ceiling are weak.
- Per-cell normalisation alone (`seq_meth_cellnorm`, `meth_cellnorm`) improves the
  ranking on K562 (Pearson 0.34 to 0.41 and -0.20 to 0.43 for meth-only) and HMEC
  (R2 0.41 to 0.53) but leaves a level error on K562 (offset 0.88 z, R2 -1.45,
  R2 after removing the offset 0.03), so it does not fix K562 by itself. Adding
  augmentation and window dropout to it (`seq_meth_cellnorm_aug_mdrop0.2`) gives
  Pearson 0.70 and R2 -0.24; the augmentation on the legacy scaling is the better
  variant here. Why the augmented legacy scaling beats the normalised input on
  K562 was not tested.
- Against gradient boosting (`GB_both`, section 41) on the two unseen cells the
  best CNN has R2 0.34 against 0.12, Pearson 0.73 against 0.64, Spearman 0.60
  against 0.51 (K562: 0.41 / 0.73 / 0.63 against 0.16 / 0.60 / 0.47; HMEC: 0.52 /
  0.72 / 0.64 against 0.36 / 0.64 / 0.51). The unaugmented `seq_meth` is below
  gradient boosting on unseen cells. On the seven seen cells all CNN variants
  except `meth_cellnorm` are above gradient boosting (R2 0.62 to 0.69 against
  0.46); the augmentation costs 0.05 R2 and 0.02 Pearson there.
- Against sequence alone (`seq`) the best methylation-aware CNN adds on unseen
  cells 0.10 R2 (0.34 against 0.24), 0.16 Pearson (0.73 against 0.57) and 0.10
  Spearman. The gain comes mostly from HMEC (R2 0.52 against 0.23); on K562 R2 is
  equal (0.41 against 0.40) and only Pearson rises (0.73 against 0.66). The
  cell-agnostic ceiling (`ceiling_loo`, the mean of the seven measured training
  tracks, which needs ChIP data of other cells as input) is still above every
  model on unseen cells (R2 0.61, Pearson 0.82).

### Limitations

- Two independent unseen cells (K562, HMEC); medians of two cells have no
  interval, and RAD21 and SMC3 exist only for K562. One training run per variant,
  one seed.
- The four variants were compared on the same two unseen cells that report the
  result; there is no validation on an unseen cell, so the choice of
  `seq_meth_aug` as the best variant is made on the evaluation cells and the
  size of its advantage is an optimistic estimate. The diagnosis (K562 input
  outside the training range) rests on the correlation of failure with input
  distance across two cells and the meth-only control, not on an intervention
  on K562 (for example shifting its methylation into the training range).
- Gradient boosting uses all even chromosomes and 500,000 training rows, the CNN the
  even chromosomes without chr2 and chr4 (validation); the two columns are close
  but not identical test sets.
- `--target-norm cell` and `--meth-norm rank` are implemented and smoke-tested only.
  The remote repository copy of the script differs from the committed one only in the
  header comment.

### Files

`scripts/seq_cnn_chip.py`; `results/seq_cnn/v2/seq_cnn_results.csv`,
`seq_cnn_summary.csv`, `seq_cnn_partition.csv` (all variants, section 42 rows
included; the partition still uses `seq`, `seq_meth`, `meth`),
`comparison_by_mark.csv` (per mark medians for seen, unseen, K562 and HMEC with
`GB_both` and `ceiling_loo`), `diag_prediction_bias.csv` (y_mean, p_mean, sds,
slope, R2, R2 after removing the offset, Pearson per model, cell and mark). The
section 42 files stay in `results/seq_cnn/`.

## 60. Do CNN-predicted ChIP tracks help the track-dropout Hi-C model when measured tracks are missing? (FINAL; section 59 was the last number when this was written)

Question: when ChIP tracks are absent, does replacing them by tracks that the sequence
+ methylation CNN predicts (sections 42 and 59, variant `seq_meth_aug`) improve the
existing track-dropout Hi-C model over leaving them absent? Inference only; no Hi-C
training.

### Method

- Hi-C model: `all12refres_drop_meth_10kb` (13 ChIP tracks + Meth input, residual on the
  consensus, track dropout 0.5; the model with Meth; `all12refin_drop_10kb` has no Meth
  input and was not run). 413 Akita test regions at 10 kb, `crosscell_predict.py` with
  TTA, median per-region Pearson on the upper triangle k >= 2 (column `model`), plus the
  structure metrics of section 35. Consensus: leave-one-out for the training cells,
  consensus12 for K562 and HMEC (HMEC deep, same ChIP and Meth inputs as HMEC).
- Stage 1: `seq_meth_aug` checkpoint (`ckpt_best.pt`, copied to
  `/mnt/processed/wolffjoa/hic-multiome/runs/seq_cnn_twostage/`) predicts the 13 marks at
  10 kb for the bins of every 192-bin Hi-C window (and the bins 1 Mb downstream for the
  control), with reverse-complement averaging. K562 and HMEC are unseen by the CNN; the
  seven training cells gm12878, imr90, h1, hepg2, a549, gm23248, cd14mono were fitted by
  it, on the odd chromosomes. Of the 413 regions, 169 lie on the CNN test chromosomes
  (chr6, 8, 10, ..., 22; held out of CNN training and validation) and the rest on odd
  chromosomes (CNN training sequence) or chr2 and chr4 (CNN validation). Both the 413 and
  the 169 are reported; for the training cells only the 169 are free of in-sample
  chromosomes.
- Scale: the CNN output z is converted to log1p signal units with the pooled training
  statistics (z*sd + mu, `D.mu`, `D.sd`, global target normalisation), then to the Hi-C
  input by the Hi-C model's own `norm_tracks.json` (log1p, per-track z-score, no
  smoothing): both models use the same log1p of the 10 kb bin mean. No per-cell scaling,
  no measured ChIP of the evaluated cell. Check on cells with measured tracks
  (`scale_check.csv`, CNN test chromosomes, windows of the evaluation regions, Hi-C z
  units, mean over marks): imputed mean minus measured mean is -0.26 for K562 (K562
  CTCF -0.25, H3K27ac -0.49, RAD21 -0.52), +0.01 for HMEC and between -0.08 and +0.07 for
  the seven training cells; the SD of the imputed track is 0.74 to 0.91 times the
  measured SD (K562 0.83, HMEC 0.84; per mark 0.56 to 1.47) as expected from a
  regression predictor; Pearson of imputed with measured 0.68 (K562), 0.64 (HMEC),
  0.70 to 0.77 (training cells).
- Scenarios (set kept; every other track absent or imputed): `all13` (every mark the cell
  has; HMEC lacks RAD21 and SMC3, which stay absent), `meth` (Meth only), `none` (no ChIP,
  no Meth), `ctcf`, `ctcf_h3k27ac`, `five` (CTCF, H3K27ac, H3K4me3, H3K27me3, H3K36me3),
  each with Meth given to the Hi-C model (`+meth`, `meth`) and without. Variants:
  absent, `@imp` (CNN prediction for every mark not kept), `@shuf` (the CNN prediction
  of the bin 100 positions (1 Mb) downstream on the same chromosome: right cell and
  scale, wrong locus). Paired differences use a region bootstrap (2000 resamples) of the
  median.
- Code: `scripts/twostage_cnn_imputation.py` (stages `cnn distribution predict summary`),
  `scripts/crosscell_predict.py` gets `--impute name=npz` and keep sets named
  `<set>@<name>`.

### Results

Median per-region Pearson, 413 regions (set with Meth given; `none` has no Meth):

| cell | scenario | absent | CNN-imputed | shuffled | consensus |
|---|---|---|---|---|---|
| K562 | all13 (upper bound) | 0.906 | 0.906 | 0.906 | 0.859 |
| K562 | meth only | 0.869 | 0.878 | 0.772 | 0.859 |
| K562 | none (no Meth) | 0.855 | 0.869 | 0.716 | 0.859 |
| K562 | CTCF + Meth | 0.878 | 0.878 | 0.775 | 0.859 |
| K562 | CTCF + H3K27ac + Meth | 0.896 | 0.886 | 0.819 | 0.859 |
| K562 | 5 marks + Meth | 0.892 | 0.888 | 0.839 | 0.859 |
| HMEC | all13 (upper bound) | 0.905 | 0.905 | 0.905 | 0.907 |
| HMEC | meth only | 0.903 | 0.898 | 0.851 | 0.907 |
| HMEC | none (no Meth) | 0.904 | 0.896 | 0.833 | 0.907 |
| HMEC | CTCF + Meth | 0.900 | 0.900 | 0.856 | 0.907 |
| HMEC | CTCF + H3K27ac + Meth | 0.903 | 0.908 | 0.889 | 0.907 |
| HMEC | 5 marks + Meth | 0.906 | 0.907 | 0.894 | 0.907 |

Paired difference imputed minus absent (median per region, 95% region bootstrap
interval, regions improved of 413; `paired_differences.csv`):

| cell | scenario | difference | interval | improved |
|---|---|---|---|---|
| K562 | meth only | +0.005 | 0.003 to 0.009 | 246 |
| K562 | none | +0.012 | 0.007 to 0.018 | 256 |
| K562 | CTCF + Meth | +0.000 | -0.003 to 0.003 | 210 |
| K562 | CTCF + H3K27ac + Meth | -0.006 | -0.009 to -0.005 | 138 |
| K562 | 5 marks + Meth | -0.005 | -0.006 to -0.004 | 162 |
| HMEC | meth only | -0.003 | -0.005 to 0.000 | 183 |
| HMEC | none | -0.003 | -0.007 to 0.000 | 180 |
| HMEC | CTCF + Meth | -0.001 | -0.004 to 0.002 | 201 |
| HMEC | CTCF + H3K27ac + Meth | +0.002 | -0.001 to 0.004 | 219 |
| HMEC | 5 marks + Meth | +0.000 | -0.002 to 0.002 | 205 |

On the 169 CNN-test-chromosome regions the picture is the same (K562 meth only +0.006,
none +0.014, CTCF + H3K27ac + Meth -0.007; HMEC meth only -0.005, none -0.006).
Median over the seven training cells (169 regions, in-distribution cells; CNN-fitted):
meth only +0.012, none +0.029, CTCF + Meth +0.011, CTCF only +0.023, CTCF + H3K27ac +
Meth +0.000, 5 marks + Meth -0.002; one cell dominates (cd14mono, whose consensus is
0.626: meth only +0.090, CTCF + Meth +0.053, but CTCF + H3K27ac + Meth -0.022 and 5 marks
+ Meth -0.027).
Structure metrics (`scenario_medians.csv`): small and mixed. K562 meth only:
compartment 0.903 to 0.914, eigenvector 0.953 to 0.960, SCC 0.860 to 0.865, loop APA
0.629 to 0.681, insulation 0.946 to 0.940; the median boundary F1 falls from 1.000 to
0.667 for K562 CTCF + Meth and 5 marks + Meth. HMEC is unchanged within 0.01 except loop
APA (+-0.02 to 0.03). The all13 rows are identical in the three variants because no mark
is left to impute (HMEC keeps its two missing cohesin marks absent).

Shuffled control: replacing the CNN tracks by their 1 Mb shifted copies lowers every
scenario in which a mark is replaced below absent (K562 meth only 0.772, none 0.716;
HMEC meth only 0.851, none 0.833; K562 -0.05 to -0.14, HMEC -0.01 to -0.07 against
absent) and below imputed in all of them. The Hi-C model reads the imputed tracks locus
by locus; wrong-locus tracks are actively harmful, so the locus information in the CNN
prediction is what carries the small gain, and the neutral absent value is the safer
fallback when the CNN is unsure.

### Verdict

Imputation helps only where almost nothing is measured, and by little. On K562
the paired median difference is +0.005 (meth only; +0.009 in medians, a quarter of the
0.037 gap from meth only to all marks) and +0.012 (neither Meth nor ChIP) over absent,
against a 0.047 gap of the full model over the consensus; on HMEC, where the model equals the
consensus within 0.002 at full input, it changes nothing within +-0.005. Once CTCF and
H3K27ac or five marks are measured, imputing the rest is neutral to harmful on K562
(-0.005 to -0.006, intervals exclude 0). The CNN route to Meth is also redundant: the
model with the imputed tracks and no Meth input (K562 none 0.869) equals the model
with Meth and absent tracks (0.869). The gains stay well below the 0.016 to 0.05
model-vs-consensus gaps; imputing all 13 marks gives about what one measured CTCF track
gives (K562 absent: CTCF + Meth over meth only +0.009, CTCF over none +0.011).

### Caveats

- Two independent unseen cells (K562, HMEC); medians of two cells have no interval, and
  the bootstrap resamples regions of one cell, not cells.
- The CNN variant `seq_meth_aug` was selected on K562 and HMEC (section 59), so the
  imputed tracks of these two cells are optimistic; the Hi-C scenario gain is bounded by
  that optimism and is still small.
- The Hi-C model also receives the consensus, so the gains are on top of it; where the
  consensus is already near the model (HMEC) there is nothing to gain.
- The imputed tracks of the seven training cells are in-sample for the CNN (cell and
  odd chromosomes; even test chromosomes are held-out in position, not in cell); they
  serve as contrast, not evidence for unseen cells. For unseen cells, 244 of the 413
  regions lie on CNN training or validation chromosomes (sequence seen in other cells);
  the 169-region subset agrees.
- K562 imputed tracks are shifted by -0.26 z (mean over marks; -0.5 for H3K27ac and
  RAD21), the known K562 failure of section 59; no per-cell rescaling was tried, since a
  correction would need measured K562 ChIP.
- One Hi-C model, one seed. The seed noise of the median Pearson is 0.013 (section 15);
  the paired differences share the model and regions, so the intervals above exclude
  seed variation, and most differences are smaller than that noise.

### Files

`scripts/twostage_cnn_imputation.py`; `scripts/crosscell_predict.py` (`--impute`);
`results/twostage_cnn/scenario_medians.csv` (all metrics, subsets all413 and clean169),
`paired_differences.csv`, `scenario_table_pearson.csv`, `scale_check.csv`,
`per_region_scores.csv.gz`, `twostage_cnn.png`. CNN checkpoint, caches, track dumps
(`pred/`) and per-set Hi-C predictions under
`/mnt/processed/wolffjoa/hic-multiome/runs/seq_cnn_twostage/`.

## 61. AML-inclusive cross-validation without healthy donors: pooled ten test patients and the section 36 claims re-tested (FINAL; section 60 was the last number when this was written)

Question: with the rebuilt folds (sections 50, 52, 54; healthy PBMC donors 103,
673, 576 removed, same-donor pairs 168/496 and 260/424 kept together, three
patient-disjoint folds over the ten AML patients with Hi-C and CUT&Tag; test
patients fold 10: 018, 027, 270; fold 11: 168, 546, 629, 1360; fold 12: 472,
773, 1916), do the section 36 findings (other-patient mean beats every model at
25 kb; model beats it at 10 kb; patient-specific gain; marks +0.013 / +0.032;
AML-specific loops and stripes not reproduced; lab indicator no help) hold?

Method. `scripts/aml_folds_v2_summary.py` pools `eval_f10/11/12_*.csv` from
`/mnt/processed/wolffjoa/hic-multiome/runs/aml_train_remote/fold10..12`
(scored by `scripts/aml_train_eval.py --fold N`, 413 Akita test regions, 25 and
10 kb). The unit is the donor (10 donors; 168 stands for 168+496, the 260/424
pair is in no test set). Pooled value = median over the ten patients, interval
= 95% percentile bootstrap over donors (10000 resamples, seed 0). Paired
differences are patient-wise. Two other-patient references exist: `ref_other_patients`
= mean of the fold's training patients (7 or 6 patients; the map the models
receive as input) and `ref_lopo_all` = mean of all other patients except the
same donor (9 patients). Their Pearson is the only metric scored for baselines.
Patient-specific = windows in the patient's top 20% on both deviation scores
(35 to 40 windows per patient). Models: `aml_refin` (AML-inclusive, reference
input), `aml_honest` (no reference input), `base_honest` and `base_refin@aml`
(trained without AML data), `aml_refin@hspc` (same model fed the HSPC
reference). Conditions: own marks, none.

### Pooled medians over ten test patients (donor bootstrap 95% interval)

| 25 kb | marks | no marks |
|---|---|---|
| Pearson aml_refin | 0.899 [0.882, 0.918] | 0.894 [0.872, 0.916] |
| SCC aml_refin | 0.896 [0.872, 0.913] | 0.887 [0.867, 0.912] |
| insulation | 0.956 [0.947, 0.966] | 0.958 [0.947, 0.964] |
| loop enrichment | 0.803 [0.745, 0.832] | 0.791 [0.748, 0.826] |
| compartment | 0.919 [0.904, 0.933] | 0.914 [0.894, 0.929] |
| patient-specific windows (Pearson) | 0.770 [0.686, 0.814] | 0.732 [0.692, 0.806] |
| Pearson aml_honest | 0.595 [0.534, 0.625] | 0.434 [0.422, 0.467] |
| Pearson base_honest | 0.533 [0.503, 0.603] | 0.448 [0.416, 0.490] |
| Pearson base_refin@aml | 0.885 [0.866, 0.907] | 0.888 [0.866, 0.918] |
| Pearson aml_refin@hspc | 0.870 [0.847, 0.895] | 0.868 [0.838, 0.886] |

| 10 kb | marks | no marks |
|---|---|---|
| Pearson aml_refin | 0.813 [0.769, 0.847] | 0.809 [0.764, 0.845] |
| SCC aml_refin | 0.805 [0.745, 0.837] | 0.796 [0.736, 0.837] |
| insulation | 0.961 [0.951, 0.966] | 0.959 [0.948, 0.968] |
| loop enrichment | 0.429 [0.353, 0.497] | 0.420 [0.356, 0.495] |
| compartment | 0.879 [0.842, 0.906] | 0.873 [0.824, 0.905] |
| patient-specific windows (Pearson) | 0.550 [0.526, 0.673] | 0.558 [0.502, 0.660] |
| Pearson aml_honest | 0.494 [0.459, 0.542] | 0.357 [0.336, 0.404] |
| Pearson base_honest | 0.429 [0.385, 0.500] | 0.345 [0.310, 0.426] |
| Pearson base_refin@aml | 0.795 [0.745, 0.839] | 0.805 [0.760, 0.843] |
| Pearson aml_refin@hspc | 0.760 [0.717, 0.806] | 0.760 [0.707, 0.804] |

Baselines (Pearson, then patient-specific Pearson), 25 kb: other patients
(training mean) 0.908 [0.885, 0.935] and 0.750 [0.687, 0.814]; other patients
(all nine) 0.911 and 0.754; HSPC consensus (4 donors) 0.884 and 0.697; HSPC
training mean (2 donors) 0.866 and 0.678; consensus12 0.822 and 0.558.
10 kb: 0.803 [0.754, 0.834] and 0.527 [0.488, 0.642]; all nine 0.814 and 0.559;
HSPC 0.748 and 0.498; HSPC training mean 0.711 and 0.442; consensus12 0.694 and
0.394. Per-patient values in `per_patient.csv`.

### Claims re-tested (paired, patient-wise; difference with donor bootstrap interval; patients where the first term is larger)

| claim | 25 kb | 10 kb |
|---|---|---|
| aml_refin(marks) minus training-mean reference, Pearson | -0.012 [-0.015, -0.002], 1/10 | +0.011 [+0.007, +0.019], 10/10 |
| same against the nine-patient mean | -0.016 [-0.020, -0.009], 0/10 | -0.004 [-0.008, +0.005], 4/10 |
| best of all models per patient minus training mean | -0.011 [-0.013, -0.002], 1/10 | +0.013 [+0.009, +0.019], 10/10 |
| aml_honest, base_honest minus training mean | -0.32, -0.37, 0/10 each | -0.28, -0.34, 0/10 each |
| base_refin@aml(marks) minus training mean | -0.025 [-0.031, -0.012], 0/10 | +0.000 [-0.012, +0.002], 5/10 |
| patient-specific windows, aml_refin(marks) minus training mean | -0.002 [-0.014, +0.027], 5/10 | +0.035 [+0.026, +0.056], 10/10 |
| patient-specific, same against the nine-patient mean | -0.012 [-0.028, +0.021], 3/10 | +0.017 [-0.003, +0.056], 8/10 |
| marks minus no marks, aml_refin, Pearson | +0.004 [-0.000, +0.009], 8/10 | +0.000 [-0.005, +0.011], 5/10 |
| marks minus no marks, aml_refin, patient-specific | +0.007 [-0.006, +0.036], 7/10 | +0.013 [-0.001, +0.026], 7/10 |
| marks minus no marks, aml_honest, Pearson | +0.135 [+0.103, +0.187], 10/10 | +0.122 [+0.100, +0.176], 10/10 |
| AML training (aml_refin minus base_refin@aml, marks), Pearson | +0.014 [+0.007, +0.016], 10/10 | +0.014 [+0.006, +0.022], 9/10 |
| AML training, patient-specific | +0.005 [-0.012, +0.017], 6/10 | +0.014 [+0.002, +0.028], 9/10 |

Residual correlation corr(P - R, T - R), R = training-mean reference, all
windows (patient-specific windows in brackets): aml_refin marks 25 kb 0.203
[0.256], none 0.111 [0.150]; 10 kb marks 0.278 [0.282], none 0.238 [0.265];
base_refin@aml marks 0.191 [0.261] and 0.213 [0.238]; aml_honest marks 0.080
and 0.115. The prediction deviates from the reference in the direction of the
measured deviation, with a small share of its variance.

Loops and stripes (10 kb, Xu et al. AML-specific loops and stripes, windows
from the evaluation output, per-patient medians then pooled over ten donors;
log2 donut enrichment and stripe band minus the reference): measured loops
+0.292 [+0.239, +0.354] against aml_refin(marks) -0.012 [-0.019, +0.009] and
no marks -0.029 [-0.033, -0.018]; measured stripes +0.184 [+0.084, +0.205]
against +0.006 [-0.022, +0.026]. Across windows corr(P - R, T - R) is 0.196
for loops and 0.357 for stripes (aml_refin marks); base_refin@aml (no AML
training) gives 0.211 and 0.325. The enrichment above the reference is not
produced by either model. The evaluation output already contains this stage,
so no perturbation run was needed.

### Subtype stratification (descriptive, `subtype_groups.csv`)

Groups from `results/aml_subtype/patient_subtype_table.csv` (consensus_group):
TET2 (018, 546, 1916), RUNX1 (027, 168), KMT2B (270, 629), CEBPA (773),
STAG2 (472), FLT3-ITD only (1360). Median aml_refin(marks) minus training
mean at 10 kb, Pearson (patient-specific): KMT2B +0.015 (+0.083), RUNX1 +0.013
(+0.059), CEBPA +0.019 (+0.037), FLT3-ITD only +0.010 (+0.026), STAG2 +0.007
(+0.021), TET2 +0.006 (+0.030). At 25 kb every group lies between -0.016 and
-0.002 in Pearson. No fold removes a whole group that has several patients;
the three singleton groups (CEBPA, STAG2, FLT3-ITD only) are the leave-one-
group-out cases (no patient of the same group in training): median patient-
specific gain +0.026 against +0.040 for the seven patients with a same-group
patient in training, Pearson +0.010 against +0.012. The patient-specific gain is
carried by KMT2B patient 270 (+0.126). Groups have one to three patients; the
comparison supports no conclusion about subtype transfer.

### Old fold 0 against the rebuilt folds

The four old test patients are 027 and 270 (now fold 10) and 773 and 1916 (now
fold 12), so the comparison is patient-wise (`old_vs_new_same_patients.csv`).
Median over these four patients, old (contaminated) / rebuilt:

| quantity | 25 kb | 10 kb |
|---|---|---|
| aml_refin Pearson | 0.900 / 0.908 | 0.818 / 0.812 |
| training-mean reference | 0.912 / 0.916 | 0.802 / 0.800 |
| nine-patient mean (old: all eleven other) | 0.922 / 0.921 | 0.814 / 0.810 |
| aml_refin patient-specific | 0.755 / 0.763 | 0.592 / 0.600 |
| training-mean reference, patient-specific | 0.751 / 0.774 | 0.541 / 0.559 |
| aml_honest Pearson | 0.623 / 0.625 | 0.525 / 0.521 |
| aml_refin marks minus none, Pearson | +0.006 / +0.004 | +0.008 / +0.003 |
| aml_refin marks minus none, patient-specific | +0.031 / +0.004 | +0.018 / +0.019 |
| corr(P - R, T - R), marks | 0.167 / 0.209 | 0.234 / 0.278 |

Individual patients move by at most 0.025 in Pearson (up to 0.05 in
patient-specific windows, e.g. 773 at 25 kb patient-specific reference 0.692
against 0.697, 270 0.623 against 0.605). Removing the healthy donors from
training and from the reference, and separating the duplicate pairs, changed
the four-patient medians by 0.01 or less in Pearson; the old section 36 numbers
were not inflated by the contamination. The main change is the sample: ten
patients with intervals instead of four. The 25 kb marks effect in
patient-specific windows (+0.031 old) falls to +0.004 on the same four
patients and +0.007 on all ten, a difference within the interval. The old
claim that the reference wins in 12/12 patients at 25 kb becomes: aml_refin is
below the training mean in 9/10 patients and below the nine-patient mean in
10/10; the old 4/4 model wins at 10 kb becomes 10/10 against the training mean
and 4/10 against the nine-patient mean.

### Variants not re-run

AML+Meth (`aml_meth_refin`, section 38) was not trained for folds 10 to 12.
The residual variant `aml_refres` (section 45) exists for old fold 0 only, at
10 kb, and is contaminated by the old split. Old fold-0 medians over four
patients, 10 kb: aml_refres marks 0.812 Pearson, 0.588 patient-specific (none
0.803, 0.566); aml_refin marks 0.818, 0.592; aml_meth_refin marks+Meth 0.816,
0.586; training mean 0.802, 0.541 (`old_fold0_refres_meth.csv`). The residual
form does not improve on the reference-input form. The lab indicator ablation
(`aml_refin_nolab`) was fold 0 and 25 kb only and is not re-tested.

### Verdict

At 25 kb the other-patient mean is at least as good as every model in
Pearson (aml_refin 1/10 patients above it, median -0.012; -0.016 against the
nine-patient mean, 0/10); in patient-specific windows the model and the
reference tie (-0.002 [-0.014, +0.027]). At 10 kb aml_refin is above the
training-mean reference in 10/10 patients (+0.011 Pearson, +0.035 in
patient-specific windows, intervals above zero) and level with the
nine-patient mean (-0.004, 4/10; patient-specific +0.017, 8/10, interval
including zero), so part of the 10 kb gain reflects the model's reference
being built from fewer patients. The AML training adds +0.014 Pearson to the
same architecture trained without AML (10/10 at 25 kb). The deviation of the
prediction from the reference correlates 0.20 to 0.28 with the measured
deviation, and the model contains no AML-specific loop or stripe enrichment
(measured +0.29 and +0.18 log2, model about 0). Marks add nothing detectable
to the reference-input model (+0.000 to +0.013, intervals include zero), while
the no-reference model needs them (+0.12 to +0.14). With AML patients in
training the model remains a refined version of the other-patient reference,
about 0.01 Pearson above it at 10 kb in this design and not above it at 25 kb;
it does not predict AML-specific structure.

Caveats. Ten donors, folds of three or four, one seed per model and fold; the
bootstrap covers patient sampling and excludes training variance. Pooled
intervals treat the three folds as exchangeable although the reference of each
fold is built from different patients. Seed variation was not measured here, so differences of about 0.01 cannot be separated from it. Subtype groups have one to three patients.
Baselines exist for Pearson only; insulation, loop and compartment scores are
available for models alone.

### Files

`scripts/aml_folds_v2_summary.py`; `results/aml_train_v2/`: `per_patient.csv`,
`pooled_medians.csv`, `paired_claims.csv`, `loop_stripe_per_patient.csv`,
`loop_stripe_pooled.csv`, `subtype_per_patient.csv`, `subtype_groups.csv`,
`old_vs_new_same_patients.csv`, `old_fold0_refres_meth.csv`, `aml_folds_v2.png`.
Evaluation inputs under `/mnt/processed/wolffjoa/hic-multiome/runs/aml_train_remote/`.

## 62. Loop-level test: does the model predict AML-specific loops and stripes in held-out patients? (FINAL; section 61 was the last number when this was written)

Question: the eight site figures of section 36b-v2 show a predicted but weaker gain over HSPC, the fold analysis of section 61 shows no gain beyond the AML reference. This section tests all AML-specific objects of the ten test donors of folds 10, 11, 12 (018, 027, 270 | 168, 546, 629, 1360 | 472, 773, 1916) at 10 kb.

### Design
Objects and matched controls are those of the loop classifier (sections 47, 57; `results/aml_loop_classifier/features.csv.gz`, identical sampling): AML-specific loops of Xu et al. Suppl. Table 1 (2276 loops, 193 to 284 per donor) and stripes of Suppl. Table 4 (1075 stripes, at most 150 per donor, span up to 1.5 Mb; 319 wider ones are dropped because they do not fit the 1.92 Mb window); controls: A_hspc, A_full, A_do (shared loops or stripes matched 1:1) and B_rand (random pair or shifted rectangle at the same distance). Every object and control gets a 1.92 Mb window centred on it; the fold's models predict it (own marks, the fold's AML reference): aml_refin with marks and without any ChIP (`none`), aml_refin with the HSPC training reference (`aml_refin@hspc`), and the model trained without AML data (`base_refin@aml`). The reference baseline is the fold's AML reference map itself; measured HSPC is the four-donor HSPC consensus. Loop score: log2 donut enrichment (inner 1, outer 4, as the labels of section 47), maximum over the 3x3 pixels around the annotated pixel; stripe score: mean log O/E of the stripe rectangle. Gain = score minus measured HSPC score. Recurrence: other donors among the 25 AML samples of Suppl. Table 1 and 4 with the same object (loops within one bin in both anchors; stripes overlapping anchor and extent), 168/496 and 260/424 counted once, the donor's own partner excluded: recurrent >= 2 other donors, shared 1, private 0. Intervals: bootstrap over the ten donors (500 to 2000 resamples). 16,466 objects scored. Inference on one local GPU.

Of the loops 1805 are private (79 percent), 278 shared by one other donor, 193 recurrent (8 percent); of the stripes 285 private, 195 shared, 595 recurrent (55 percent). Only 87 loops are present in two or more training donors of the fold (the reference), 195 in one.

### Results, loops (median over objects, log2 enrichment)
| class | n | measured gain | model gain | reference alone | no marks | no AML training | model minus reference | model minus no marks |
|---|---|---|---|---|---|---|---|---|
| all | 2276 | 0.404 [0.355, 0.476] | 0.060 [0.046, 0.070] | 0.096 | 0.037 | 0.079 | -0.034 [-0.043, -0.026] | +0.014 [0.006, 0.020] |
| recurrent | 193 | 0.450 | 0.163 | 0.219 | 0.146 | 0.204 | -0.045 [-0.063, -0.034] | +0.010 [0.003, 0.028] |
| shared by 1 | 278 | 0.388 | 0.094 | 0.137 | 0.064 | 0.117 | -0.039 | +0.021 |
| private | 1805 | 0.401 | 0.045 | 0.081 | 0.025 | 0.063 | -0.032 [-0.040, -0.024] | +0.013 [0.006, 0.019] |

Measured gain over the reference is +0.30 (all), +0.24 (recurrent), +0.32 (private); the model is below the reference in 10/10 donors in every class. Amplitude ratio (model gain / measured gain): 0.15 all, 0.36 recurrent, 0.11 private; the reference alone 0.24, 0.49, 0.20. Predicted gain over the predicted HSPC-input model (`aml_refin@hspc`, which itself sits 0.04 below measured HSPC): 0.083 all, 0.248 recurrent, 0.069 private. Promoter-enhancer-like loops (n=301): measured 0.368, model 0.080, reference 0.100; promoter-repressive-like (n=103): 0.375, 0.046, 0.098.

Predicted against measured gain (Spearman, pooled, donor bootstrap): model 0.28 [0.19, 0.35] (Pearson 0.26 [0.17, 0.36]; median per donor 0.31, positive in 10/10), reference alone 0.29 [0.22, 0.36], no AML training 0.30, no marks 0.25. Recurrent loops: model 0.34, reference 0.46 [0.34, 0.56]. Private: model 0.25, reference 0.25. Against the deviation from the reference (what the model adds): model 0.18 [0.09, 0.26] (recurrent 0.10, includes zero), no marks 0.12.

Separation of AML-specific loops from matched shared loops (AUROC, A_hspc controls):
| score | measured | model | no marks | no AML | reference |
|---|---|---|---|---|---|
| gain over HSPC, all | 0.834 [0.820, 0.860] | 0.621 [0.580, 0.663] | 0.625 | 0.661 | 0.585 [0.541, 0.618] |
| gain, recurrent | 0.848 | 0.771 [0.718, 0.819] | 0.775 | 0.806 | 0.771 |
| gain, private | 0.835 | 0.594 [0.549, 0.636] | 0.599 | 0.635 | 0.551 |
| enrichment, all (A_hspc) | 0.199 | 0.086 | 0.075 | 0.098 | 0.062 |
| enrichment, all (B_rand) | 0.890 | 0.807 | 0.791 | 0.795 | 0.758 |
Fraction of AML-specific loops above the 90th percentile of the matched controls (gain; chance 0.10): measured 0.50, model 0.153 [0.120, 0.194], reference 0.170, no AML 0.215; recurrent 0.575 / 0.446 / 0.492; private 0.500 / 0.111 [0.094, 0.143] / 0.122. A_full and A_do give the same picture (model gain AUROC 0.568 with A_full). The shared controls are selected for enrichment >= 0.6 in both patient and HSPC, so their enrichment exceeds that of AML-specific loops by construction (AUROC of raw enrichment far below 0.5 in measured data too); the gain is the discriminating score, and against random pairs the raw enrichment separates the loops (measured 0.89, model 0.81, recurrent 0.95).

### Results, stripes
Measured gain 0.221 (mean log O/E), model 0.044 [0.017, 0.064], reference 0.044, no marks 0.044, no AML 0.027; model minus reference +0.003 [-0.014, 0.022], model minus no marks +0.003. Amplitude ratio 0.20 (recurrent 0.22, private 0.22). Spearman of predicted and measured gain 0.50 [0.42, 0.58] (reference 0.46, no marks 0.44); against the deviation from the reference 0.28 (no marks 0.10). AUROC against matched shared stripes (A_hspc, gain): measured 0.63, model 0.52 [0.49, 0.56], reference 0.46, no AML 0.56; enhancer-like stripes (n=326) model 0.61 (measured 0.63, reference 0.53), repressive-like (n=442) 0.49 (measured 0.64), other 0.48. Enhancer-like stripes are the one class where the own-marks model is above the reference (+0.018 [0.001, 0.039], above the no-marks model by +0.021 [0.004, 0.041]); private stripes +0.026 [0.009, 0.041] over the reference. Fraction above the control p90: measured 0.25, model 0.16, reference 0.12.

### Verdict
(a) Against healthy cells the model predicts a gain at AML-specific loops, with the right sign in 10/10 donors, 15 percent of the measured amplitude (median +0.06 against +0.40 log2) and a rank correlation of 0.28 with the measured gain per loop. It separates AML-specific from matched shared loops with AUROC 0.62 (measured 0.83), and 15 percent of AML-specific loops exceed the 90th percentile of the controls (measured 50 percent). Stripes: gain 20 percent of the measured amplitude, AUROC 0.52.
(b) Recurrent loops (present in at least two other donors) carry the signal: gain 0.16, 36 percent of the measured amplitude, AUROC 0.77, 45 percent above control p90. Private loops: gain 0.045, 11 percent amplitude, AUROC 0.59, 11 percent above control p90 (chance 10 percent).
(c) The signal comes from the reference. The reference map alone has the same correlation (0.29), the same recurrent AUROC (0.77), and a larger gain (0.096) than the model; the model lies 0.034 log2 below it (10/10 donors), the model trained without AML data is level with it (0.079). Own marks add +0.014 log2 over the no-marks condition (8/10 donors; +0.010 recurrent, +0.021 promoter-enhancer-like and promoter-repressive-like) and +0.02 for enhancer-like stripes; they do not lift AUROC (0.621 against 0.625). Section 61 and the eight sites are consistent: the correlation of 0.55 to 0.61 over seven hand-picked sites is higher than the 0.28 over all loops (the sites were chosen for a clear gain, and most are recurrent), and the model contains no gain beyond the reference (measured +0.30 over the reference, model -0.034).
(d) Suggested wording: "In the ten held-out patients the model predicts a weaker gain at AML-specific loops than in healthy cells (median +0.06 against +0.40 log2 measured; rank correlation 0.28), almost entirely at loops that recur in other patients and are therefore present in the AML reference (AUROC against matched shared loops 0.77); at loops private to the patient the gain is 0.045 and the separation 0.59. The AML reference map alone gives the same result, and the patient's own marks add 0.014 log2. The model reproduces AML-specific structure that other patients share and does not predict the patient-private loops and stripes."

Caveats. Ten donors, one seed per model; 10 kb only (25 kb not analysed; the loop tables are 10 kb); the matched shared controls are selected for enrichment in both patient and HSPC; recurrence uses a one-bin tolerance; stripe controls are shifted rectangles; the donut maximum over 3x3 pixels favours the sharper measured map slightly (exact-pixel values are in `object_scores.csv.gz`, columns `#px`).

### Files
`scripts/aml_loop_detection.py`; `results/aml_loop_detection/`: `items.csv.gz` (objects, windows, recurrence), `object_scores.csv.gz`, `levels_by_class.csv`, `increments.csv`, `gain_correlation.csv`, `auroc.csv`, `auroc_by_pair_type.csv`, `per_patient.csv`, `class_sizes.csv`, `levels_by_train_recurrence.csv`, `aml_loop_detection.png`. Predictions under `/mnt/processed/wolffjoa/hic-multiome/runs/aml_loop_detection/`.

## 63. Loop-centric training of the AML-inclusive model: can it predict AML-specific loops and stripes of a new patient? (PENDING; section 62 was the last number when this was written)

Question: section 62 shows that the AML-inclusive model predicts a weak gain at AML-specific loops (median +0.06 against +0.40 log2 measured, Spearman 0.28, AUROC 0.62 against matched shared loops, private loops 0.59), carried by the AML reference. The pixel loss is dominated by background; AML-specific loops (about 230 per patient, 7 training patients per fold) and stripes are rare, small-amplitude features, and the patient's own marks add little under that loss (sections 47, 57, 61, 62). The residual-reference model alone (section 45) did not change this. This section adds a loop-centric recipe and tests it on the same ten held-out patients.

### Implementation (status: implemented, CPU-tested, no GPU run yet)
Run family `all12aml_loopfocus_refres_drop_f<N>` (N = 10, 11, 12) at 10 kb and 25 kb: the residual-reference AML model (`all12aml_refres_drop`: model.reference residual, so the network output is the patient-specific deviation from the AML reference; same cache `mc_all12aml_ref_<k>kb_f<N>`, AML x3 oversampling, patient-hard validation) plus `train.aml_object_*` (`hicmultiome/aml_objects.py`, hooks in `hicmultiome/train.py`, all defaults off, existing runs unchanged):

| flag | value | role |
|---|---|---|
| `data.aml_objects_table` | share `aml/objects/objects_f<N>.csv.gz` | objects of the fold's training patients |
| `train.aml_object_oversample` | 5 | windows holding an AML-specific object and an equal number of one-to-one matched control windows (same patient and object kind; first choice the classifier's matched control of the object, else an unused control window of the patient) enter every epoch 5 times, on top of the AML x3 |
| `train.aml_object_controls` | `[A_hspc]` | control set (shared loops / stripes matched on distance, orientation, HSPC CTCF and H3K27ac) |
| `train.aml_object_pair_batches` | true | batches of half AML-specific and half control windows, so each batch holds ranking pairs |
| `train.aml_object_weight` | 1.0 | extra loss on the donut neighbourhood of loops (box of outer+1 bins) and the stripe rectangle with one bin of flank, both objects and controls: L1 between predicted and measured residual (prediction minus reference, target minus reference) plus L1 between the object-level enrichments (loop: maximum of the donut enrichment over the 3 x 3 pixels around the annotated pixel, as the evaluation; stripe: mean over the rectangle) |
| `train.aml_object_rank` | 0.5 | pairwise logistic ranking softplus(-(s_AML - s_control)/0.05) over every AML-specific and control object of the same kind in the batch, scores on the residual; `aml_object_rank_consistent` keeps pairs whose measured scores are ordered the same way; `aml_object_rank_space` residual or full |
| `train.aml_object_donut` | [1, 4] at 10 kb, [1, 3] at 25 kb | enrichment ring (inner, outer half-width in bins; the donut of `hicmultiome/losses.py` with the evaluation's 10 kb setting) |

Reverse-complement augmentation flips the object coordinates with the window (tested against the tensor flip). The loss terms are logged per epoch (`obj_pix`, `obj_enr`, `obj_rank`, `obj_n`). The loss keys are `train.*` because `train.loss` is one spec string.

Objects (`scripts/aml_objects_table.py`, `results/aml_objects/summary.csv`): from the loop classifier table (Suppl. Table 1 loops, Suppl. Table 4 stripes and their matched controls, 10 kb, span up to 150 bins) restricted to the training patients of the fold. Fold 10 (7 patients): 1591 AML-specific loops, 729 stripes, 1510 / 647 A_hspc controls; fold 11 (6 patients): 1320 / 700, 1296 / 603; fold 12 (7 patients): 1641 / 721, 1562 / 628. The column `n_train_other` counts the other training donors carrying the object (private among the training patients: 1438 of 1591 loops, 375 of 729 stripes in fold 10).

On the real fold 10 10 kb cache the recipe finds 1114 training windows with AML-specific objects and 922 with controls only (1891 of 4477 objects lie wholly inside a cached training window with margin: validation and training-region filters), 934 matched pairs; oversampling x5 adds 7472 windows to the 40.5k of the refin run (+18 percent).

### Leakage safeguards
- A table holds only the training patients of its fold (`aml_train.train_patients`), after removing every patient of the same donor as a test patient (`aml_train.SAME_DONOR`, `donor_group`); `assert_no_leak` runs before the file is written, `tests/test_aml_objects.py` repeats it on the written tables of folds 10, 11, 12 and on a synthetic table that includes test patients and the partners 496 and 424.
- `n_train_other` counts training donors only; no test patient, partner or non-training sample enters the table.
- The trainer rejects a table with a cell that is not a training sample of the run (`load_objects`).
- Oversampling and loss terms act on the rows of the training split `ti`, selected after the honest validation split (`val_exclude_overlap`: no training window overlaps a validation position in any sample), so no validation window is a training window; HSPC donors (050, 213 train; 341, 822 held out) and MOLM-13 are as in the other AML runs, U-937 and the test patients are never trained on.
- Recurrent test-patient loops that a training patient also carries are learnable from that training patient (as from the reference); the private criterion (objects of the test patient seen in no other donor) is not reachable that way.

### Tests
`tests/test_aml_objects.py` (22): table leakage (synthetic and written tables), foreign-cell rejection, window membership with margin, one-to-one matched pairs, batch composition, flip equivalence, loss values and gradient direction, ranking direction, consistent-pair filtering, config defaults and validation, run configs of folds 10, 11, 12 at 10 and 25 kb (identical to the refres run except the recipe keys), and a 60-step CPU training smoke with all flags (synthetic genome; the three object terms appear in the log). Full suite 209 passed, 1 skipped. Configs: `results/aml_loopfocus/configs/` (`scripts/aml_loopfocus_configs.py`).

### GPU job and evaluation
`results/aml_loopfocus/REMOTE_JOB.md`: six runs (3 folds x 10 / 25 kb), claims by run name, evaluation by `aml_train_eval.py --fold N --variant extra` (model label `aml_loopfocus`) and `aml_loop_detection.py --recipe loopfocus` (tables in `results/aml_loop_detection_loopfocus/`, stage `criteria`). Estimated remote 4090 time: about 5 h for the three 10 kb runs (range 4 to 7.5 h; refin_drop took 1.0 to 1.7 h each), about 2.5 h for the three 25 kb runs, about 1 h for the evaluations; about 6.5 h at 10 kb alone, about 9.5 h in total.

### Success criteria (10 kb, ten held-out patients, donor bootstrap)
1. AUROC of the gain over HSPC, private loops against matched shared loops (A_hspc): >= 0.70 (section 62: 0.59).
2. Amplitude (median model gain / median measured gain), private loops: >= 0.30 (section 62: 0.11).
3. Pooled Pearson on the Akita test regions: no loss against the aml_refin models (0.813).
Results: not yet available.

### Risks
The ranking and enrichment terms push enrichment at annotated positions up; a model that raises enrichment at every annotated-looking site would also raise the false-positive rate (the matched shared controls and the ranking term are the counterweight; B_rand and A_do controls in `criteria`/`auroc.csv` test this). Only 42 percent of the table objects lie in cached training windows. Weights (1.0, 0.5, temperature 0.05, x5) were not tuned; one seed per fold. The 25 kb variant uses the 10 kb object coordinates mapped to 25 kb bins and a narrower ring, and its loops are only 2 to 60 bins long. The hard-split JSONs of folds 10 to 12 exist only in the remote repo copy.

### Files
`hicmultiome/aml_objects.py`, `scripts/aml_objects_table.py`, `scripts/aml_loopfocus_configs.py`, edits in `hicmultiome/train.py`, `hicmultiome/config.py`, `scripts/multicell_expanded.py`, `scripts/aml_train_eval.py`, `scripts/aml_loop_detection.py`; `tests/test_aml_objects.py`; `results/aml_objects/summary.csv`, `results/aml_loopfocus/REMOTE_JOB.md`, `results/aml_loopfocus/configs/`.


## 64. Which marks and methylation features go with AML-specific loops and stripes, and can they be boosted? (FINAL; section 63 was the last number when this was written)

Script `scripts/aml_loop_mark_correlations.py` (stages inventory, tracks, features, corr, boost, maps, figures; hicdiff env, CPU only), results in `results/aml_loop_marks/`. Objects: the 2276 AML-specific loops and 1075 stripes of the ten held-out donors of section 62 (168/496 and 260/424 count once; healthy donors excluded) with their matched shared controls; target = measured gain over the HSPC consensus (`gain|measured`). CUT&Tag CTCF and WGBS exist for 8 of the 10 donors (1360 and 1916 have H3K27ac and H3K27me3 only).

### Data (inventory.csv, inventory.md, inventory_data_types.md)
Everything held locally is processed; no raw reads exist for any assay. Hi-C: nodups pairs for 32 Xu samples (25 AML, 3 healthy PBMC, 4 HSPC; 169 GB) plus iced mcools (10 kb and coarser, 25 GB), 0.02 to 0.49 B contacts at 5 kb per sample. CUT&Tag: 33 MACS2 -log10 p-value bigWigs (12 H3K27ac, 12 H3K27me3, 9 CTCF incl. healthy 103), one sample per mark and donor; GEO also lists MACS2 peak files, not downloaded (command line not in the files we hold). WGBS: 20 `.CGN.10bp.bw` (fraction per 10 bp bin, no read counts or coverage; 10 with CUT&Tag, 10 WGBS only; healthy 103 and 673, no 576, no CD34+ WGBS; CD14+ monocyte ENCODE as the other healthy reference). SV (595 junctions, 22 samples), CNV (54 segments, 18 samples), fusion / mutation labels (hand-read from Xu Fig. 1c), Suppl. Tables 1 and 4 (loops, stripes). Fischer 2024: 160 ChIP bigWigs (CTCF, RAD21, STAG2, H3K27ac; 45 patient folders) and 3 pooled Hi-C maps; cell lines MOLM-13 and U-937 (Hi-C + ChIP), K562. Used now: Hi-C targets and references, CUT&Tag as quantile-matched model inputs, WGBS as 1 kb `Meth` channel, SV table in the SV-aware runs, Fischer at group level. Not used before this section: MACS2 peaks (not downloaded), WGBS at 10 bp, Fischer per-patient RAD21 / STAG2, the 7 extra CD34+ ChIP experiments fetched here (3 H3K27ac, 3 H3K27me3, 1 CTCF; `work/loopmarks/hspc_extra`), cell-line loops. ATAC (33 files) and RNA (22 files) were downloaded by the private-loop agent (`results/aml_private_loop_info/`, in progress; its first 8-donor logistic AUROC increments of ATAC / RNA features over marks are about -0.002 [-0.005, 0.000], work not finished) and are not analysed here.

### Correlations (corr; correlations_all_features.csv.gz, auroc_all_features.csv.gz, top_features.md, expected_relationships.csv, fig_mark_heatmap.png)
1093 features: mark (H3K27ac, H3K27me3, CTCF) x representation (own, HSPC, patient minus HSPC; also z-scored / percentile difference, minus healthy 103/673, minus other AML donors, HSPC mean) x position (anchor A, B, min, max, sum, product, promoter, partner, interior; stripes anchor, extent mean/max) x scale (bin, +-20 kb, +-100 kb); WGBS mean, minimum, low-methylated fraction (1 kb bins < 0.5) with own / healthy / CD14+ / minus-reference; CpG islands; methylation and CTCF signal at the CTCF motif hits (JASPAR MA0139.2 >= 70 percent, anchor +-1 bin; methylation from the 10 bp bigWigs; data for about one third of the anchors). Pooled Spearman with donor bootstrap (1000), per-donor Spearman, partial Spearman controlling log distance and the HSPC enrichment of the object, AUROC against matched shared controls (pair-type classes against controls of the same own-mark type). Family-wise null (gain permuted within donor, max |rho| over about 475 features, 95th percentile): 0.12 (all loops), 0.15 (private loops), 0.17 (all stripes), 0.28 (enhancer-like stripes).
- Stripes: the one clear signal. Patient-minus-HSPC H3K27ac at the anchor (minimum over anchor and extent, +-100 kb) rho 0.30 [0.23, 0.35] all stripes, 0.45 [0.36, 0.51] enhancer-like (10/10 donors), 0.36 [0.27, 0.45] private, 0.26 recurrent; partial identical. HSPC H3K27ac and H3K27me3 correlate negatively (-0.23 to -0.36; low HSPC state, large gain).
- Loops: no patient-specific mark correlates with the gain magnitude beyond the family-wise null of 0.12 to 0.15. The strongest are locus properties: healthy WGBS mean at +-100 kb rho -0.26 [-0.30, -0.21] private loops (all 8 donors negative; patient-minus-healthy -0.1 to -0.2 only for some positions), own H3K27ac +-100 kb -0.22 [-0.30, -0.14] (10/10 negative): the more active / hypomethylated the region, the smaller the gain over HSPC. Recurrent loops: own CTCF and own H3K27ac negative (-0.28 to -0.32).
- Expected relationships. Delta H3K27ac at both anchors for promoter-enhancer loops: rho with the gain -0.04 [-0.19, 0.07] (not supported for magnitude); as a separator against own-mark-matched PE controls AUROC 0.69 [0.64, 0.73] (min over anchors), 0.73 [0.69, 0.77] (max), 10/10 donors (supported); own H3K27ac min 0.76. H3K27me3 for promoter-silencer loops: gain rho -0.05 to -0.16 (CI includes 0, n=103), AUROC of delta H3K27me3 0.63 to 0.74 [0.70, 0.80] (4 donors evaluable; supported, small n). Repressive stripes: delta H3K27me3 at the anchor rho 0.01 [-0.08, 0.09] (not supported for magnitude), AUROC 0.74 [0.70, 0.76] (9/9 donors; own H3K27me3 0.81); enhancer-like stripes delta H3K27ac rho 0.27 [0.16, 0.35] and AUROC 0.67 (supported). CTCF: delta and own CTCF rho -0.05 to 0.02 (CI around 0) and AUROC 0.45 to 0.53 for loops and stripes, all classes (not supported); HSPC CTCF +-20 kb separates AML-specific loops from shared controls (AUROC 0.24, i.e. controls are CTCF-richer) but that is HSPC state. Methylation at CTCF motifs: rho -0.03 [-0.09, 0.04] (bound hit), patient-minus-healthy -0.10 [-0.19, 0.01]; recurrent loops +0.34 [0.22, 0.48] (n=43, 0 evaluable donors); AUROC of bound-motif methylation 0.58 [0.56, 0.62] for loops (AML-specific loops sit at less-methylated motifs than controls, partly the HSPC matching), none for stripes; patient-minus-healthy AUROC 0.51 (not supported). Patient-minus-CD14 WGBS mean AUROC 0.56 [0.54, 0.58] for loops.

### Boosting (boost; boost_metrics.csv, boost_increments.csv, boost_increments.md, boost_oof_*.csv.gz, fig_boost_increments.png)
Leave-one-donor-out, locus-disjoint (classifier folds), histogram gradient boosting as in the loop classifier (BASE reproduces its AUROC 0.837); AUROC against matched controls and Spearman of the predicted gain on positives; bootstrap over donors of paired differences. Private loops, increment over BASE (AUROC 0.850 pooled): all engineered sets together (ALL, 685 features) +0.013 pooled [-0.011, 0.033], +0.014 per donor [0.007, 0.021]; multiscale alone +0.005 [-0.008, 0.017]; pairwise products / convergent-CTCF interactions -0.001 [-0.003, 0.001]; per-patient z / percentile differences +0.002; HSPC mean of 4 experiments +0.004 [-0.011, 0.016]; healthy-donor or other-AML-donor reference -0.001, -0.002; extended methylation / CpG / motif sets +0.002; monotone constraints +0.008; per-donor feature centring +0.020 [-0.010, 0.044]. Gain Spearman, private loops (BASE 0.197): multiscale +0.045 [0.013, 0.069] (per donor +0.051 [0.028, 0.076]), ALL +0.038 [-0.017, 0.094], centring +0.068 [-0.040, 0.180] (per donor +0.064 [0.012, 0.115]). Patient data over locus-only (HSPC + sequence covariates, AUROC 0.834): BASE +0.016 [0.001, 0.029], ALL +0.029 [-0.008, 0.058]. Private stripes: no increment is reliable (AUROC BASE 0.588, locus-only 0.605, ALL 0.628, ALL over BASE +0.040 [-0.001, 0.090]; gain Spearman +0.01 to +0.08, all intervals include 0); the logistic model on ALL +0.062 [0.018, 0.108] is one of 19 variants and not corrected for that. Of the 19 variants x 2 objects x 2 tasks, few intervals exclude 0.
2D maps (maps; maps2d_summary.csv, fig_maps2d.png): outer-product maps of the 1D deltas (H3K27ac, H3K27me3, CTCF, WGBS minus healthy, own H3K27ac, H3K27ac x H3K27me3) in the 1.92 Mb windows of all 3351 objects, diagonal-detrended pixel Spearman with measured minus AML reference, measured minus HSPC and the model residual (measured minus aml_refin): local rho (loop +-15 bins, stripe rectangle +-5) between -0.03 and +0.09 for every channel (model residual +0.001 to +0.021; delta H3K27ac outer product +0.018 loops, +0.007 stripes; additive and minimum channels the same size), partial correlation given the row and column marginals 0.000 to 0.008. CTCF channels 0.00 / negative.

### Recommendation
Add to the U-Net inputs (and to a loop-level re-scorer): (1) multiscale patient-minus-HSPC H3K27ac, smoothed at +-20 kb and +-100 kb (expected: the stripe signal, rho 0.30 to 0.45, is a +-100 kb delta; +0.045 gain Spearman for private loops), (2) per-patient normalisation of the difference channels (genome-wide z or percentile of the patient track before differencing; +0.02 to +0.03), (3) healthy-donor / HSPC-mean H3K27ac and methylation as locus context at +-100 kb (regional activity predicts a smaller gain, rho -0.2 to -0.26); keep the HSPC mean of several experiments as reference instead of one experiment (no measurable gain, cheap). Not worth it: outer-product / pairwise channels (maps: partial rho <= 0.008, tree interactions -0.001), convergent-CTCF x delta CTCF (CTCF carries no gain signal at 10 kb), motif-level methylation channels (data for a third of the anchors, no signal), patient-minus-other-AML reference (no gain), additional CpG-island / low-methylation fraction channels (+0.002). The ceiling is low: the best private-loop AUROC is 0.86 to 0.87 against matched shared loops, of which 0.83 is HSPC / sequence state; the patient's own marks add 0.016 and all engineering a further 0.01 to 0.015.

### Caveats
Ten donors (eight for CTCF and WGBS, 168 and 496 counted once), one HSPC experiment per mark in all reference-dependent analyses, three marks only, one CUT&Tag sample per donor (depth differs 4-fold across donors; the quantile matching removes scale, not noise). Pair-type classes are defined from the patient's own marks, which restricts the range of own-mark features inside the class (hence AUROC against same-type controls). Wide-window features are regional chromatin-state proxies. Many features (family-wise nulls above) and 19 boosting variants; intervals are not corrected. Methylation at motifs: data for about one third of the anchors (CpG-poor 19 bp windows have no 10 bp bin).

### Files
`scripts/aml_loop_mark_correlations.py`; `results/aml_loop_marks/`: `inventory.csv`, `inventory.md`, `inventory_data_types.{csv,md}`, `correlations_all_features.csv.gz`, `auroc_all_features.csv.gz`, `top_features_{correlation,auroc}.csv`, `top_features.md`, `expected_relationships.csv`, `correlation_familywise_null.csv`, `boost_metrics.csv`, `boost_increments.{csv,md}`, `boost_oof_{loop,stripe}.csv.gz`, `maps2d_summary.csv`, `fig_mark_heatmap.png`, `fig_boost_increments.png`, `fig_maps2d.png`. Caches (feature matrix, tracks) under `/mnt/processed/wolffjoa/hic-multiome/aml/primary/work/loopmarks/`.

## 65. Does patient ATAC-seq and RNA-seq predict private AML-specific loops and stripes? (FINAL; section 64 was the last number when this was written)

Question: sections 47, 57, 61 and 62 leave private loops (79 percent of AML-specific loops) unpredicted. Is there data we can obtain that carries information beyond the HSPC state of the locus, the patient's own CUT&Tag marks and WGBS?

### Inventory (Xu et al. 2022, superseries GSE152136; PBMC donors 103, 673, 576 are healthy)
Sub-series: GSE152096 `[RNA-Seq]`, GSE152099 `[WGBS]`, GSE152132 `[CUT&TAG]`, GSE152134 `[ATAC-Seq]`, GSE152135 `[Hi-C]`. Sample titles are of the form `018_ATAC-seq`, `018_RNA-seq`, `018_WGBS`, `018_CUT&TAG_H3K27ac`, `018_Hi-C` (table in `results/aml_private_loop_info/geo_samples.csv`, per-patient overlap in `inventory_per_patient.csv`).
- ATAC-seq (GSE152134, 16 samples, 2.5 GB): per sample `GSM46042xx_<id>_ATAC-seq.10bp.RPGC.bigwig` (median 152 MB) and `.MACS2.filt.peaks.txt.gz` (0.8 MB). Patients 018, 027, 168, 260, 270, 472, 546, 566, 629, 773, 990, 1360, 1916; PBMC 103, 576, 673. No ATAC for 1071, 1324, 424, 496, 504, 798, 838, 933.
- RNA-seq (GSE152096, 22 samples, 72 MB): `GSM..._<id>_RNA-seq.RSEM.quant.genes.txt.gz` (RSEM counts, TPM, FPKM, GENCODE v24, hg38). Patients 018, 027, 168, 260, 270, 472, 546, 566, 629, 773, 868, 1071, 1324, 424, 496, 504, 798, 838, 933; PBMC 103, 576, 673. No RNA for 1360, 1916, 990, 1021, 936, 1953. Raw reads are withheld by the submitters (patient privacy).
- No ATAC or RNA for the four HSPC donors (HSPC-050, -213, -341, -822; Hi-C only). No H3K4me1, H3K4me3 or other marks in the cohort (CUT&Tag: H3K27ac 12, H3K27me3 12, CTCF 9 samples).
- Overlap with the loop classifier (20 non-healthy samples): all 8 patients with CTCF + K27ac + K27me3 + WGBS have ATAC and RNA (018, 027, 168, 270, 472, 546, 629, 773); 1360 and 1916 have ATAC and K27ac/K27me3 only; 260 and 566 have ATAC, RNA, WGBS; 1071, 1324, 424, 496, 504, 798, 838, 933 have RNA and WGBS only.
- Other AML sources (`aml_data_survey.md`): Hi-C plus marks exist for THP-1, MOLM-13, HL-60, U-937, OCI-AML3, MV4-11, NB-4 (cell lines; each one genome, so at most one more "patient" each, no private-versus-recurrent split); Kasumi-1 has no genome-wide Hi-C; Fischer et al. 2024 (GSE268035) has per-patient CTCF/RAD21/STAG2/H3K27ac ChIP for 7 patients but pooled Hi-C only. None adds per-patient ATAC or RNA with Hi-C.

### Downloaded (`/mnt/processed/wolffjoa/hic-multiome/aml/primary/{atac,rna}`, 3.5 GB, `scripts/aml_private_loop_download.sh`)
Xu ATAC bigWigs and peaks, Xu RSEM gene files, GENCODE v24 GTF; HSPC reference: five healthy CD34+ ATAC experiments (ChIP-Atlas hg38 bigWig SRX19490150-52, SRX10942129-30) and ENCODE ENCSR830HIN (CD34+ common myeloid progenitor, polyA RNA, ENCFF690QPA, GENCODE V24; one sample). Pearson of 10 kb bins between HSPC replicates 0.61 to 0.80; patient against mean of other patients 0.71 to 0.90, patient against HSPC 0.24 to 0.66 (`atac_sample_corr.csv`), so patient-minus-HSPC features carry cross-laboratory batch. RNA Spearman patient-patient 0.73, patient-HSPC 0.75.

### Design
`scripts/aml_private_loop_info.py`. ATAC: 10 kb bin mean and max of log2(1 + RPGC), quantile-matched to the mean AML profile. RNA: log2(TPM + 1), quantile-matched; genes (protein coding and lincRNA, 27,346) placed on refGene TSS by symbol (80 percent of TSS), GENCODE TSS otherwise. New groups, each per pair as min and max over the two regions: atacown, atachspc, atacdiff (mean, max, +-20 kb flank), rnaown, rnahspc, rnadiff (promoter-anchor expression, genes within +-20 kb, strongest up and down gene against HSPC), acxatac (product of genome-wide percentiles of H3K27ac and ATAC; patient and patient minus HSPC). Rows, controls (A_hspc matched shared loops or stripes, B_rand), locus-disjoint leave-one-donor-out folds, donor units ({168, 496}, {260, 424}) and models (logistic regression, gradient boosting) are those of `results/aml_loop_classifier_v2`. Private = no other donor carries the object (one-bin tolerance, Suppl. Tables 1 and 4): 3684 of 4520 loops (81 percent), 405 of 1394 sampled stripes (29 percent). Datasets: D1, 8 patients with all data, baseline HSPC state + own marks + WGBS + motif + gene + distance; D2, 10 patients with marks and ATAC (no WGBS in 1360, 1916), baseline without methylation; D3, 16 donors with RNA and WGBS, no own marks, baseline HSPC + WGBS; D4, 10 patients with ATAC, RNA and WGBS, no own marks. Increments: paired donor bootstrap (500) of macro AUROC.

### Results (gradient boosting; logistic regression is the same within 0.01)
| set | target vs controls | donors | baseline pooled AUROC [95%] | + all new groups (macro increment [95%]) | best single group |
|---|---|---|---|---|---|
| D1 loops | private vs A_hspc | 8 | 0.856 [0.824, 0.886] | -0.000 [-0.006, 0.004] | rnadiff +0.001 |
| D1 loops | private vs B_rand | 8 | 0.772 [0.723, 0.823] | -0.002 [-0.011, 0.008] | atacown +0.004 |
| D1 loops | all vs A_hspc | 8 | 0.838 [0.804, 0.868] | -0.001 [-0.007, 0.004] | rnaown +0.001 |
| D1 loops | non-private vs A_hspc | 8 | 0.743 [0.713, 0.778] | +0.001 [-0.017, 0.017] | rnaown +0.006 |
| D2 loops | private vs A_hspc | 10 | 0.850 [0.822, 0.877] | -0.001 [-0.004, 0.002] | atachspc +0.001 |
| D3 loops | private vs A_hspc | 16 | 0.831 [0.810, 0.852] | +0.002 [0.000, 0.004] | rna_all +0.002 |
| D3 loops | private vs B_rand | 16 | 0.678 [0.638, 0.709] | +0.009 [0.000, 0.018] | rnadiff +0.008 |
| D4 loops | private vs A_hspc | 10 | 0.837 [0.809, 0.863] | +0.006 [0.001, 0.011] | rnadiff +0.001 |
| D4 loops | private vs B_rand | 10 | 0.687 [0.642, 0.731] | +0.010 [-0.004, 0.024] | rna_all +0.006 |
| D1 stripes | private vs A_hspc | 8 | 0.602 [0.544, 0.654] | +0.002 [-0.034, 0.035] | acxatac +0.012 |
| D1 stripes | private vs B_rand | 8 | 0.678 [0.649, 0.712] | +0.017 [-0.009, 0.040] | acxatac +0.022 |
| D1 stripes | all vs A_hspc | 8 | 0.600 [0.578, 0.628] | -0.003 [-0.022, 0.014] | acxatac +0.004 |

All 48 dataset x target x negative x model rows are in `headline_table.csv`, the 12 feature sets per row in `increments.csv` and `summary.csv` (AUROC and AUPRC with intervals). Largest point estimate of any increment 0.026 (logistic regression, non-private loops, D1), largest upper interval limit 0.056 in that one row; for private loops the largest point estimate is 0.014 (D4 vs B_rand, logistic regression). By pair type (D1, `by_type.csv`): promoter-enhancer loops baseline 0.88 and with new groups 0.88 (private, gradient boosting), promoter-silencer loops 0.77 against 0.76; stripes enhancer-type +0.04 [0.00, 0.08] and repressive +0.01 on 91 and 101 private stripes. The new groups alone (distance + ATAC + RNA, no HSPC marks or own marks) reach pooled AUROC 0.72 for private loops against A_hspc, so they carry the signal but all of it is contained in the baseline. Strongest single features (mean donor AUROC, private vs A_hspc): H3K27ac x ATAC interaction patient-minus-HSPC 0.60, H3K27ac itself 0.62, down-regulation of nearby genes 0.58, patient-minus-HSPC ATAC 0.56 to 0.57; HSPC ATAC at the anchor goes the other way (0.35 to 0.43; AML-specific anchors are closed in HSPC).

### Gate verdict
Not passed. ATAC and RNA raise the private-loop AUROC by at most +0.006 (point estimates -0.002 to +0.014 over all configurations), against the threshold of +0.05; the intervals exclude +0.03 everywhere for loops. The baseline (HSPC state, own marks, WGBS) already separates private loops from matched shared loops at 0.83 to 0.86 AUROC (random pairs 0.69 to 0.77), and ATAC, RNA and the H3K27ac x ATAC interaction are redundant with it: accessibility and expression of the patient say where regulatory elements are, which H3K27ac and the HSPC state already say. Private loops are no better and no worse separated than recurrent ones (0.86 against 0.74 in D1), so the earlier 0.59 of the Hi-C model (section 62) is a limit of the predictor on private loops, not of the information in these inputs: a classifier on tabular features reaches 0.85 with matched controls. What the data cannot show is which of the many accessible, H3K27ac-marked anchor pairs a given patient actually joins by a loop; no feature of one anchor carries the partner choice, and the classifier increments for pair-level features (HSPC contact, motif orientation, distance) are held constant by the matching.

### Further data and availability
- H3K4me1, H3K4me3, H3K27ac for the RNA-only and ATAC-only patients: not released for Xu; the cohort has no H3K4me marks. Would need new CUT&Tag on patient material.
- CTCF for 496-like patients (WGBS and RNA only; 1071, 1324, 424, 496, 504, 798, 838, 933) and for 1360, 1916: not released; CTCF is available for 9 samples only. Fischer et al. 2024 releases CTCF, RAD21, STAG2, H3K27ac per patient (7 WT patients) with pooled Hi-C (GSE268031/GSE267780), usable for marks but not for private-loop labels.
- Mutation subtype and SV: subtype inferred from Suppl. Table 3 is in `aml_subtype`; SVs in Suppl. Table 5/6. Raw WGS is withheld.
- TF expression: contained in the RNA used here (promoter and nearby genes only); a patient-level TF-expression vector (RUNX1, CEBPA, GATA2, MECOM, cohesin) was not tested and has 18 patients as the sample size, too few for a locus-level model.
- Single-cell multiome or HiChIP on the same patient: GSE302267 (scHi-C of AML bone marrow HSPCs, no marks); no public per-patient H3K27ac HiChIP or capture Hi-C in AML cohorts was found.
Recommendation: private AML-specific loops cannot be pushed by more input tracks of this kind; claims for new patients should be made for recurrent loops and stripes (reference-carried, section 62), with private loops stated as not predicted. A route that could add information is deeper own-patient 3D data (capture Hi-C or HiChIP of the same patient), which is a measurement of the target, not an input.

Caveats. Eight to sixteen donors; one HSPC RNA sample from another laboratory (CD34+ CMP) and ATAC of healthy donors from other laboratories, so diff groups carry batch; private is defined by absence from the other donors' call sets (detection limit, not proof of absence); stripes are subsampled to 150 per patient and only 29 percent are private; no seed variation measured.

### Files
`scripts/aml_private_loop_info.py`, `scripts/aml_private_loop_download.sh`; `results/aml_private_loop_info/`: `headline_table.csv`, `summary.csv`, `increments.csv`, `by_type.csv`, `univariate.csv`, `private_counts.csv`, `inventory_per_patient.csv`, `inventory_files.csv`, `geo_samples.csv`, `atac_sample_corr.csv`, `atac_hspc_replicate_corr.csv`, `rna_sample_spearman.csv`, `oof_predictions.csv.gz`, `fig_private_loop_info.png`. Caches under `/mnt/processed/wolffjoa/hic-multiome/aml/primary/work/privinfo/`.


## 66. One rule for all cell types: differential mark channels and cell-specific objects of every training cell (PENDING; section 65 was the last number when this was written)

Premise: "this chromatin-mark / methylation pattern leads to a loop or a stripe" is cell-type independent biology and AML is only where it shows most clearly. The AML-specific recipe of section 63 gives the model the patient's marks and the AML reference and rewards AML objects only; whatever it learns can be an AML-specific rule. This section implements one rule for the 12 training cell lines, the 5 held-out cells and the AML patients: the network gets the reference's marks and the sample-minus-reference difference of the marks beside the reference Hi-C it already reads, so that residual Hi-C ~ f(delta marks) with one shared f, and the loop-centric sampling and loss act on the cell-specific loops and stripes of every training cell.

### Implementation (status: implemented and CPU-tested; no window cache built, no GPU run)
Differential mark channels (`hicmultiome/diffmarks.py`; hooks in `precompute.py`, `train.py`, `predict.py`, `scripts/crosscell_predict.py`; all off by default, existing caches, run configs and 60-step training logs identical to the previous commit):

| flag | default | role |
|---|---|---|
| `data.diff_marks` | false | per mark (all tracks but annotation and copy number: 13 ChIP marks, Meth in the `diffmeth` runs) the reference mark `dref:<mark>` and the difference `diff:<mark>:s<r>` rows, appended after the other sequence-feature rows |
| `data.diff_scales` | [0, 20000, 100000] | radii in bp: the 10 kb bin, +-20 kb, +-100 kb box smoothing (25 kb: 0, +-1, +-4 bins) |
| `data.diff_reference` | consensus | reference of the AML samples: `mean_aml` (training patients of the fold, leave-one-out), `hspc` (HSPC training donors), `consensus` (the 12 cell lines) |
| `data.diff_reference_sets`, `samples[].diff_set`, `data.diff_default_set` | empty | named pools of sample names; a sample reads its pool without itself and without any sample of its cell identity |
| `data.diff_ref_channels`, `data.diff_tracks` | true, all | drop the `dref` rows; restrict the marks |
| `train.object_scope`, `data.cell_objects_table` | aml, none | `all`: the objects of `data.aml_objects_table` and of the cell-line table feed the section 63 oversampling, pair batches, neighbourhood loss and ranking |

Channel definition, no Hi-C read: per sample and mark z = (log1p(raw) - q50) / (q99 - q50) over the autosomal bins (per-sample quantile anchoring, so a deep and a shallow track sit on one scale; the model's own log1p / global z-score is a different, shared normalisation); reference = mean of z over the pool members that have the mark; difference = z(S) - reference, box-smoothed per scale, standardised per sample, mark and scale ((d - median) / (1.4826 MAD), clipped to +-8). The rows bypass the track normaliser (`TrackNormalizer.passthrough`, absent from old files). A mark the sample lacks, or that no pool member has, gives 0 in all its rows ("no information", the cached absent value); track dropout of a mark zeros its rows (`apply_diff_drop`, in the same draw as the ChIP row), inference keep sets and imputation do the same (`blank_diff_rows`); `train.select_tracks` drops the rows of the marks it leaves out (the select_tracks lesson of section 27: the rows must reach the network, regression test below). Reference sets of the generator (`scripts/multicell_expanded.py`): cell lines the 12 cells (leave-one-out = the leave-one-out consensus of the Hi-C reference; a held-out cell reads all 12 = consensus12), AML patients and MOLM-13 the fold's training patients (= the AML reference), HSPC donors the other training donor. Unseen cells at inference (`crosscell_predict --diff-group cell|aml|hspc`, `predict --diff-group`) read the pool of their group and leave their own identity out.

Objects of every training cell (`scripts/cell_objects_table.py`, table columns of section 63, `n_train_other` = other training cells carrying the object): the cell-agnostic definitions of `scripts/crosscell_rule_transfer.py`, written in parallel, which builds the cell-specific loops and stripes of all 15 cells with the operational definitions of the AML objects (loops: local maxima of the log2 donut enrichment, 1 to 4 bins, 10 kb, 6 to 150 bins, both anchors on peak bins; specific = enrichment >= 0.5 and gain over the reference >= 0.30; stripes: 4 anchor bins x 40 extent bins at an offset of 50 bins, specific = score >= 0.55 and gain >= 0.20; shared controls `A_ref` matched on distance, orientation and the reference's CTCF and H3K27ac). `from_rule_transfer` converts its `$HM/crosscell_rules/cell_objects.csv.gz` (sets pos and A_ref) into the table: the 12 training cells to the training table, the strongest 400 loops and 200 stripes per cell that have a matched control (the AML patients have about 230 private loops each; the cap keeps the oversampled window count at the section 63 level), K562, HMEC (deep) and HUVEC (deep) with consensus12 to a separate evaluation table (1000 loops and 1000 stripes per cell, seeded draw) that the trainer rejects. The reference of a cell is its leave-one-out consensus. The script also has its own Hi-C calls (`--source local`: donut peaks, a 4 x 20-bin stripe rectangle next to the anchor, controls matched on distance and the cell's CTCF and H3K27ac) as a self-contained fallback; `--source auto` takes the rule-transfer file when it exists. The generated configs read the same output path either way. Tables written (`results/cell_objects/`, copies on the share `cell_objects/`, source file next to each): training table 12 cells x (400 loops + 200 stripes), 4800 / 2400 specific loops / stripes with 4070 / 2400 matched controls, private among the other training cells 77 percent of the loops and 47 percent of the stripes; evaluation table K562, HMEC (deep), HUVEC (deep): 3000 loops and 2563 stripes with 628 / 2211 controls. The local fallback gave 4800 / 2251 specific objects on the same cells.

Run families (`scripts/multicell_expanded.py`; the joint composition of `all12aml_*` is kept: 12 cells + the fold's training patients + HSPC donors 050, 213 + MOLM-13, AML x3, lab row, patient-hard validation, residual reference, track dropout): `all12diff_loopfocus_refres_drop_f10..12` at 10 and 25 kb (PRIMARY: channels + all-cell objects, caches `mc_all12aml_ref_<k>kb_f<N>_diff`), ablations `all12diff_refres_drop_f<N>` (channels only), `all12allobj_refres_drop_f<N>` (objects only, existing AML cache), `all12diffmeth_loopfocus_refres_drop_f<N>` (+ Meth), `all12diffhspc_loopfocus_refres_drop_f<N>` (AML samples read the HSPC donors), and the pilot on the 12 cell lines alone `all12diffcells[_loopfocus]_refres_drop_{10,25}kb`. Configs: `results/diff_marks/configs/` (`scripts/diff_marks_configs.py`).

### Leakage safeguards
- A sample never sees its own track in its reference (`reference_members`; test: replacing s0's bigWigs leaves s0's reference rows unchanged and moves the others'); a copy standing for an unseen cell is left out by identity (`diff_identity`).
- The per-sample statistics use marks only; no label enters. The reference sets of the generated configs contain no test patient, no same-donor partner (asserted in `tests/test_diffmarks.py::test_run_config_of_diff_loopfocus`).
- The AML table keeps the section 63 safeguards (training patients of the fold only, same-donor partners removed, `assert_no_leak`); the cell table holds the 12 training cells only (`cell_objects_table.assert_no_leak`; the trainer's `load_objects` rejects any cell that is not a training sample of the run, tested with K562, HMEC, HUVEC); held-out cells exist only in the evaluation table.
- Oversampling and losses act on the rows of the training split chosen after the honest validation split (`val_exclude_overlap`: no training window overlaps a validation position in any sample), unchanged.

### Tests
`tests/test_diffmarks.py` (38): reference = mean of the others, own track never in its reference, shapes / scales / per-sample standardisation (including a 4x deeper sample), missing and zeroed marks, empty references, unseen-cell identity, config validation and defaults, normaliser pass-through and file compatibility, cache layout (same targets, same ChIP rows as the cache without the channels), `select_tracks`, dropout coupling, a trained model whose output changes when any reference or difference row it reads is changed (select_tracks and track dropout in force), loop / stripe definitions on synthetic maps, control matching, objects-table leakage for training and held-out cells, conversion of the rule-transfer objects, run configs of all families at 10 and 25 kb, existing run configs without new keys, and a 60-step CPU smoke with every flag (both tables, ranking, dropout). Full suite 247 passed, 1 skipped (209 passed, 1 skipped before). Existing run configs, precompute caches (arrays and normaliser files) and a 60-step training log are byte-identical to the previous commit.

### GPU job and evaluation
`results/diff_marks/REMOTE_JOB.md`: run names = claim names, CPU preparation (object tables, caches), commands via `scripts/run_single_gpu.py`, evaluation commands: cell lines by the standard stage (`multicell_expanded.py --stage eval`, rows in `reference_summary.csv`: 17 cells, 413 Akita test regions), AML by `aml_train_eval.py --fold N --variant extra` and `aml_loop_detection.py --recipe diff_loopfocus` (`criteria.csv`, new `enh_gain_spearman` rows), held-out cell objects by `scripts/cell_objects_eval.py`. Estimated remote 4090 time: pilot 1.4 h, primary 10 kb about 6 h (5 to 9 h), 25 kb about 3 h, ablations about 9.5 h, extras about 12 h; minimum useful set about 9 h, everything about 34 h.

### Success criteria (written before any run; 10 kb, ten held-out patients, donor bootstrap)
1. AUROC of the gain over HSPC, AML-private loops against matched shared loops (A_hspc): >= 0.70 (aml_refin 0.59, section 62).
2. Enhancer-like stripes: Spearman of the model's gain over HSPC with the measured gain above that of the reference map alone.
3. Median Pearson on the 413 Akita regions, 12 training and 5 held-out cells: no loss against the matching baseline beyond the honest seed spread (>= -0.006; section 51 two-seed spreads of consensus-input models 0.0002 to 0.0062; use the seed replicates in `reference_summary.csv` once they are there); all12refres_drop_10kb 0.9270 / 0.8972.
4. Cell-specific loops and stripes of K562, HMEC, HUVEC: AUROC of the gain over consensus12 with its interval, reported against the residual model without the recipe.
Results: not yet available.

### Risks
The channels carry no new information for a cell whose marks equal the pool mean, and a network that ignores them returns the section 45 residual model: the ablations separate the channels from the objects. The quantile anchoring is a choice (q50 to 0, q99 to 1); sparse tracks (CUT&Tag) have q50 near 0. The cell objects depend on map depth (cap of 400 / 200 per cell) and on the definitions of `crosscell_rule_transfer.py`, which was still being developed when its objects file (2026-10-11 12:24) was converted: the tables must be regenerated (`python scripts/cell_objects_table.py`, then the 10 kb caches are unaffected, only the run configs read the table) if that stage is rerun with other thresholds; unmatched specific objects keep no control (the fallback of section 63 then takes a control of the same cell and kind). The AML patients have three marks, so most difference rows of the 13-mark layout are 0 for them; the cell lines carry 13. One seed per run; regional in-silico knockouts do not touch the difference rows. Another section may already use number 66 when this lands.

### Files
`hicmultiome/diffmarks.py`, edits in `hicmultiome/{config,precompute,train,tracks,predict}.py`, `scripts/{cell_objects_table,cell_objects_eval,diff_marks_configs}.py`, edits in `scripts/{multicell_expanded,crosscell_predict,aml_train_eval,aml_loop_detection}.py`; `tests/test_diffmarks.py`; `results/diff_marks/REMOTE_JOB.md`, `results/diff_marks/configs/`, `results/cell_objects/` (tables and counts).

## 67. HiC-Reg retrained leak-free on the chromosome split of our models (FINAL; section 66 was the last number when this was written)

Question: the released HiC-Reg numbers (sections 18, 37, 58) are suspected of leakage. How does HiC-Reg score when
it is rebuilt for our cells and tracks and trained on exactly our setting (train on all chromosomes except chr2, 3,
11, 14, 17; predict those five), and how large is the within-chromosome leak?

Audit (full text with file and line references: `results/benchmark/HICREG_LEAK_AUDIT.md`). Routes found: (1) the
released CV predictions are region-wise 5-fold CV inside the chromosome (`Framework.C:199-300`; 64% of the pairs train
each model, 89% of the test pairs have one region among the training regions, 19% (GM12878) and 13-15% (K562) of the
8 neighbouring pixels are training pixels of every model that could have predicted them); (2) the feature files hold no
Hi-C-derived column, the SQRTVC normalisation enters the target only (not a route); (3) new: the prediction pair list
is the list of non-zero pixels of the held-out chromosome's own Hi-C (GM12878 74-85% of the 1 Mb pair slots, K562
42-60%; CV and cross-chromosome files have the same pairs), and `oe_proper.py` turns an absent mappable pixel into the
clipped depletion that a zero-count target pixel has, so the zero pixels of the held-out map are delivered for free
(K562 has 24-39% zero pixels at 5 kb); (4) the released cross-cell maps use models trained on the same chromosome of the
other cell; (5) the MULTICELL fold script cuts contiguous blocks of the pair-sorted table, but the released MULTICELL
files have the pair lists of WINDOW, so the script was not the one used, and the models ship without training pair
lists; (6) no tuner is shipped, `-k1 -l10 -n20` are fixed defaults; the local `hyperopt_hicreg.py` scores against the
held-out matrix; (7) hg19 maps lifted to hg38 centres, 5 kb counts summed into 10 and 25 kb; (8) the released
cross-chromosome files are trained on one chromosome (81 to 107 Mb) with DNase and TBP.

Rebuild (`scripts/hicreg_retrain.py`): our 13 tracks per 5 kb bin as E, P, W and Distance (DNase and TBP absent for our
cells, dropped; SMC3 added), HiC-Reg's native target ln(1 + SQRTVC count) from the VC_SQRT vectors of the 4DN mcools
(GM12878, K562; the vectors divide), every pair within 1 Mb predicted, scored through the section 18 conversion
(expm1, sum into 10 and 25 kb, `oe_proper.py`) against our O/E targets on the identical pixels of the chrho blocks;
random forest as in `Potential.C` (20 trees, bootstrap, one third of the features per split) in scikit-learn
(agreement with regForest 0.983 on 60k pairs, r with the truth 0.830 against 0.826); training on all 75M mappable pairs
of the 17 training chromosomes, no sub-sampling; leaf size 100 and training size chosen on chr10 (`hicreg_retrain_tune_*.csv`);
MULTICELL (HiC-Reg's five cells, 196 features) on 12M pairs and chr14 and chr17 only. Mean of chr14 and chr17 (the blocks
that also hold the released maps), `results/benchmark/hicreg_retrained_*.csv`, `hicreg_retrained_table.md` (all
measures, five-chromosome means):

| cell, grid | source | Pearson | tile median | SCC | insulation r | loop recall | HiCRep |
|---|---|---|---|---|---|---|---|
| GM12878 25 kb | all12refin_drop_chrho | 0.931 | 0.950 | 0.925 | 0.905 | 0.396 | 0.935 |
| GM12878 25 kb | all12honest_drop_chrho (ChIP only) | 0.783 | 0.798 | 0.763 | 0.674 | 0.191 | 0.791 |
| GM12878 25 kb | HiC-Reg retrained, WINDOW | 0.430 | 0.523 | 0.482 | 0.502 | 0.050 | 0.599 |
| GM12878 25 kb | HiC-Reg retrained, MULTICELL | 0.383 | 0.569 | 0.473 | 0.498 | 0.039 | 0.601 |
| GM12878 25 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.579 | 0.620 | 0.538 | 0.447 | 0.032 | 0.668 |
| GM12878 25 kb | HiC-Reg CV, our pipeline | 0.879 | 0.886 | 0.855 | 0.615 | 0.216 | 0.885 |
| GM12878 25 kb | HiC-Reg CV (released) | 0.908 | 0.965 | 0.870 | 0.677 | 0.323 | 0.915 |
| GM12878 10 kb | all12refin_drop_chrho | 0.925 | 0.937 | 0.919 | 0.890 | 0.408 | 0.927 |
| GM12878 10 kb | all12honest_drop_chrho (ChIP only) | 0.788 | 0.810 | 0.777 | 0.735 | 0.182 | 0.796 |
| GM12878 10 kb | HiC-Reg retrained, WINDOW | 0.446 | 0.467 | 0.473 | 0.523 | 0.014 | 0.549 |
| GM12878 10 kb | HiC-Reg retrained, MULTICELL | 0.428 | 0.540 | 0.470 | 0.515 | 0.011 | 0.555 |
| GM12878 10 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.590 | 0.561 | 0.542 | 0.491 | 0.004 | 0.601 |
| GM12878 10 kb | HiC-Reg CV, our pipeline | 0.879 | 0.846 | 0.843 | 0.556 | 0.057 | 0.882 |
| GM12878 10 kb | HiC-Reg CV (released) | 0.920 | 0.939 | 0.881 | 0.626 | 0.148 | 0.925 |
| GM12878 5 kb | all12refin_drop_chrho | 0.906 | 0.911 | 0.902 | 0.898 | 0.363 | 0.923 |
| GM12878 5 kb | all12honest_drop_chrho (ChIP only) | 0.749 | 0.758 | 0.749 | 0.741 | 0.188 | 0.770 |
| GM12878 5 kb | HiC-Reg retrained, WINDOW | 0.454 | 0.433 | 0.475 | 0.537 | 0.005 | 0.525 |
| GM12878 5 kb | HiC-Reg retrained, MULTICELL | 0.448 | 0.503 | 0.474 | 0.534 | 0.034 | 0.534 |
| GM12878 5 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.603 | 0.520 | 0.554 | 0.515 | 0.000 | 0.593 |
| GM12878 5 kb | HiC-Reg CV, our pipeline | 0.872 | 0.814 | 0.836 | 0.565 | 0.030 | 0.874 |
| GM12878 5 kb | HiC-Reg CV (released) | 0.893 | 0.898 | 0.859 | 0.647 | 0.020 | 0.922 |
| K562 25 kb | all12refin_drop_chrho | 0.897 | 0.928 | 0.889 | 0.857 | 0.347 | 0.904 |
| K562 25 kb | all12honest_drop_chrho (ChIP only) | 0.730 | 0.777 | 0.711 | 0.622 | 0.125 | 0.748 |
| K562 25 kb | HiC-Reg retrained, WINDOW | 0.286 | 0.436 | 0.347 | 0.497 | 0.049 | 0.465 |
| K562 25 kb | HiC-Reg retrained, MULTICELL | 0.297 | 0.486 | 0.392 | 0.528 | 0.029 | 0.518 |
| K562 25 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.736 | 0.820 | 0.696 | 0.484 | 0.038 | 0.797 |
| K562 25 kb | HiC-Reg CV, our pipeline | 0.801 | 0.844 | 0.799 | 0.603 | 0.133 | 0.841 |
| K562 25 kb | HiC-Reg CV (released) | 0.888 | 0.950 | 0.864 | 0.670 | 0.178 | 0.923 |
| K562 10 kb | all12refin_drop_chrho | 0.858 | 0.900 | 0.857 | 0.848 | 0.366 | 0.887 |
| K562 10 kb | all12honest_drop_chrho (ChIP only) | 0.715 | 0.724 | 0.714 | 0.721 | 0.139 | 0.745 |
| K562 10 kb | HiC-Reg retrained, WINDOW | 0.336 | 0.417 | 0.378 | 0.554 | 0.028 | 0.452 |
| K562 10 kb | HiC-Reg retrained, MULTICELL | 0.361 | 0.420 | 0.426 | 0.584 | 0.016 | 0.500 |
| K562 10 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.747 | 0.760 | 0.708 | 0.496 | 0.002 | 0.782 |
| K562 10 kb | HiC-Reg CV, our pipeline | 0.817 | 0.794 | 0.808 | 0.618 | 0.064 | 0.858 |
| K562 10 kb | HiC-Reg CV (released) | 0.878 | 0.899 | 0.858 | 0.650 | 0.009 | 0.925 |
| K562 5 kb | all12refin_drop_chrho | 0.771 | 0.817 | 0.776 | 0.842 | 0.338 | 0.814 |
| K562 5 kb | all12honest_drop_chrho (ChIP only) | 0.641 | 0.641 | 0.653 | 0.730 | 0.169 | 0.664 |
| K562 5 kb | HiC-Reg retrained, WINDOW | 0.414 | 0.324 | 0.440 | 0.625 | 0.030 | 0.428 |
| K562 5 kb | HiC-Reg retrained, MULTICELL | 0.439 | 0.333 | 0.479 | 0.654 | 0.026 | 0.453 |
| K562 5 kb | HiC-Reg cross-chromosome (released), WINDOW | 0.774 | 0.703 | 0.755 | 0.599 | 0.003 | 0.790 |
| K562 5 kb | HiC-Reg CV, our pipeline | 0.781 | 0.697 | 0.783 | 0.692 | 0.039 | 0.758 |
| K562 5 kb | HiC-Reg CV (released) | 0.845 | 0.814 | 0.840 | 0.728 | 0.005 | 0.899 |

Within-chromosome leak (CV protocol on the retrained pipeline minus the retrained cross-chromosome model, Pearson): GM12878
+0.449 (25 kb), +0.433 (10 kb), +0.418 (5 kb); K562 +0.515, +0.482, +0.367; tile median +0.36 to +0.41, SCC +0.34 to +0.45
(`hicreg_retrained_leak.csv`). Released CV minus retrained cross-chromosome: +0.478 (GM12878 25 kb), +0.601 (K562 25 kb).
Support effect (retrained model, pixels with zero target count removed as in the release, Pearson, chr14+chr17):
K562 0.419 / 0.544 / 0.758 at 25 / 10 / 5 kb against 0.227 / 0.297 / 0.394 with every pair predicted; GM12878 0.394 / 0.456 /
0.512 against 0.358 / 0.401 / 0.420 (`hicreg_leak_support_effect.csv`).

Verdict. The leak-free retrained HiC-Reg is worse than the released single-chromosome cross-chromosome version, not better:
Pearson 0.430 against 0.579 (GM12878 25 kb) and 0.286 against 0.736 (K562 25 kb), 0.454 against 0.603 and 0.414 against
0.774 at 5 kb; training on 17 chromosomes instead of one does not help (4M-pair control 0.432 and 0.257). The released K562
cross-chromosome score is mostly the zero-pixel support (retrained with that support: 0.758 against 0.774 at 5 kb). Our
margin grows: all12honest_drop_chrho (ChIP only) leads retrained HiC-Reg by 0.35 (GM12878) and 0.44 (K562) Pearson at
25 kb and by 0.30 and 0.23 at 5 kb, all12refin_drop_chrho by 0.50 and 0.61 at 25 kb; both are ahead on all nine measures in
both cells and all grids; K562, for which our models never see the cell's Hi-C while HiC-Reg trains on K562 Hi-C, is not
closer. MULTICELL equals WINDOW (25 kb Pearson 0.383 and 0.297). The CV protocol on the same pipeline reaches 0.879 and
0.801 at 25 kb (released CV 0.908, 0.888), so the pipeline reproduces the interpolation level.

Caveats. The rebuild lacks DNase and TBP and differs in Hi-C source, signal source and forest implementation; the gap to
the released cross-chromosome maps combines the support effect with these differences, only the support is separated;
one seed; two chromosomes per block; MULTICELL on 12M pairs; released maps hg19 lifted, retrained maps hg38 native.
The ceiling of the native target in this scoring (true SQRTVC maps through the conversion) is 0.78 to 0.94 Pearson
(`hicreg_leak_oracle.csv`).

Files: `scripts/hicreg_retrain.py`, `scripts/hicreg_leak_audit.py`, `scripts/hicreg_retrained_table.py`,
`--chrho-hicreg-retrained` in `scripts/hic_measures.py`; `results/benchmark/HICREG_LEAK_AUDIT.md`,
`hicreg_retrained_{per_chromosome,summary,leak}.csv`, `hicreg_retrained_table.md`, `hicreg_leak_*.csv`,
`hicreg_retrain_tune_*.csv`, `hicreg_retrain_check_regforest_gm12878.json`;
`results/hic_measures_chrho/chrho_hicregre_{per_block,summary}.csv`; predictions and cools under
`/mnt/processed/wolffjoa/hic-multiome/runs/benchmark/hicreg_retrain/`. Reproduce:

    python scripts/hicreg_retrain.py --stage data
    python scripts/hicreg_retrain.py --stage tune --cell gm12878 --grid-sizes 4000000 16000000 --grid-leaf 10 30 100 300 1000
    python scripts/hicreg_retrain.py --stage fit --cell gm12878 --leaf 100 && ... --feat multicell --n-train 12000000
    python scripts/hicreg_retrain.py --stage cv --cell gm12878 --leaf 100 --n-train 4000000 --chroms chr14 chr17
    python scripts/hicreg_retrain.py --stage fit --cell gm12878 --suffix small --leaf 100 --n-train 4000000 --chroms chr14 chr17
    python scripts/hicreg_retrain.py --stage cools
    python scripts/hicreg_leak_audit.py --stage support mcfolds oracle supportleak
    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-hicreg-retrained
    python scripts/hicreg_retrained_table.py


## 68. Are the mark-to-structure rules the same in every cell type? Cell-specific loops and stripes of 15 cells against AML patients (FINAL; section 67 was the last number when this was written)

Script `scripts/crosscell_rule_transfer.py` (stages tracks, hic, objects, features, counts, rules, transfer, figures; hicdiff env, CPU only, nice 10), results in `results/crosscell_rules/`, caches in `/mnt/processed/wolffjoa/hic-multiome/crosscell_rules/`. Cells: the 12 training cells (reference = leave-one-out consensus of the other 11, `consensus_loo_*`) and K562, HMEC (deep map), HUVEC (deep map) (reference = consensus12). Cancer: K562, HCT116, MCF-7, A549, HepG2, PC-9; normal: the other nine. AML: the ten test donors of section 62 with the objects of `aml_loop_classifier_v2/features.csv.gz` and `aml_loop_detection/object_scores.csv.gz` (gain|measured over HSPC), delta features recomputed here from the 10 kb track caches. Section 66 (differential channels, model-level, PENDING) is a different analysis; this one is a feature-level test.

### Definitions (as the AML objects, 10 kb)
Loops: donut enrichment of the cell map at local maxima (> 0.4 log2, 60 kb to 1.5 Mb; MIN_BINS 6 as in the AML pool because the donut outer radius is 4 bins, not 20 kb), consensus value = 3x3 maximum, gain = cell minus consensus; candidates = both anchors on peak bins (top 8 percent H3K27ac or CTCF in the cell or its consensus track). Specific: enrichment >= 0.5 and gain >= 0.30 (about the 30th percentile of AML positives); shared: >= 0.6 in cell and consensus, not within 2 bins of a specific loop; controls matched 1:1 on distance, CTCF motif orientation and the anchors' mean H3K27ac (A_full) or the anchors' consensus CTCF and H3K27ac (A_ref, the analogue of A_hspc). Stripes: rectangle mean log O/E, 4 anchor bins x 40 extent bins starting 50 bins to the right or left (the Xu extents start 16 to 268 bins away), anchor with a peak bin; specific >= 0.55 and gain >= 0.20, shared both >= 0.55; matched on direction, orientation and anchor H3K27ac. Positives capped at 2500 per cell and kind before matching. Types as in `pair_feats` (TSS in one anchor, partner H3K27ac or H3K27me3 above the 70th percentile of the pooled training scale: PE, PS; stripes enh, rep). Tracks of all cells and donors are quantile-normalised at 10 kb to the mean training-cell distribution; features: delta (own minus reference), reference level, own, at 10 kb, +-20 kb, +-100 kb, min and max over the two anchors (stripes: anchor and extent block), delta standardised per sample (median / robust SD of the genome-wide delta, `delta_standardisation.csv`), methylation delta at +-100 kb for 9 cells (GM12878, IMR-90, H1, HepG2, A549, GM23248, CD14+, K562, HMEC; HCT116, HFF, MCF-7 tracks flat, NHEK, PC-9, HUVEC none) and 8 donors (reference mean of healthy 103 and 673), distance, motif orientation / score, promoter flags. GM23248 has no CTCF track.

### Counts (object_counts*.csv)
Cell-specific loops 125 779 in 15 cells (PE 39 475, PS 18 746, other 67 558; per cell 426 to 27 079), stripes 36 222 (enh 10 766, rep 16 523, other 8 933); after the cap and A_full matching 10 110 loop pairs (20 220 objects, 1348 per cell) and 25 310 stripe pairs (50 620 objects). AML: 2276 loops (PE 304, PS 127, other 1845) and 1075 stripes (enh 340, rep 494); 2232 loop and 1059 stripe pairs (4464 / 2118 objects, 446 / 212 per donor); per fold 1254 to 1858 loop and 686 to 734 stripe objects (the "1.5k / 0.7k" of the task). Cells give 4.5 times as many matched loop objects as the ten donors and 24 times as many stripe objects, but the shared-loop pool limits matching (25 percent of the specific loops matched).

### Within-cell rules (rules_key.csv, rules_consistency_*.csv, fig_effect_heatmap.png)
Controls A_ref; effect = Spearman with the gain, or AUROC against matched shared controls; mean over cells [bootstrap CI over cells] / AML donors. Two class conventions: classes from the own marks of positives and controls (own), and controls inheriting the class of their matched positive (inherit, the convention of section 64; circular for PE / PS / enh / rep because the class is defined by the same own marks). AML reproduces section 64 under inherit (PE delta H3K27ac max 0.74, min 0.70; PS delta H3K27me3 0.79; repressive stripes 0.72; stripes rho 0.28, enh 0.37).
- PE loops, delta H3K27ac max(A,B) AUROC: own 0.56 [0.54, 0.58] in 15 cells (13 above 0.5; cancer 0.56, normal 0.56), AML 0.58 (pooled 0.59); inherit 0.65 [0.63, 0.68] cells (15/15), AML 0.74. Same sign and about the same size in cells and donors with class defined from the marks; larger in AML only under the circular convention.
- All loops, delta H3K27ac max AUROC 0.55 [0.53, 0.58] cells (13/15), AML 0.61 (pooled): supported, small. Gain magnitude: delta H3K27ac +-100 kb rho -0.03 [-0.05, 0.01] cells, -0.13 AML (negative in 8/10 donors and 12/15 cells): no positive relation anywhere.
- PS loops delta H3K27me3: own 0.50 cells (4/13 above 0.5), AML 0.54 (2 evaluable donors); inherit 0.63 cells (12/13), AML 0.79: only under the circular convention. Repressive stripes delta H3K27me3 anchor: own 0.50 cells, 0.49 AML; inherit 0.64 cells (15/15), 0.72 AML. Enhancer-like stripes delta H3K27ac: own 0.53 cells, 0.49 AML; inherit 0.62 cells, 0.60 AML.
- Stripe gain vs delta H3K27ac +-100 kb (the clear AML signal): AML rho 0.28 [0.24, 0.33] all, 0.37 enhancer-like, 10/10 donors; cells -0.06 [-0.13, 0.00] (6/15 positive; cancer -0.11, normal -0.03), enhancer-like 0.00. AML minus cells 0.35 [0.27, 0.42]. With the other AML donors as reference the AML value falls to 0.06 (0.19 enhancer-like): the signal depends on the HSPC reference (differentiation axis).
- CTCF delta: AUROC 0.50 [0.46, 0.54] cells, 0.45 AML loops (below 0.5), 0.50 stripes: no support anywhere. Methylation delta (min over anchors, +-100 kb): AUROC 0.50 [0.47, 0.53] cells (5/9 above 0.5; cancer 0.54, normal 0.48), 0.57 AML loops, gain rho 0.01 / -0.02. Consensus H3K27ac level +-100 kb vs gain: loops 0.03 cells, -0.11 AML (AML minus cells -0.17 [-0.25, -0.09]); stripes -0.05 cells, -0.23 AML (-0.17 [-0.31, -0.03]): the locus-state effect has the opposite sign in AML.
- Cancer minus normal cells: no key rule differs (all CIs include 0 except loop all delta H3K27ac +-100 kb rho -0.05 [-0.10, 0.00]); PS delta H3K27me3 AUROC -0.07 [-0.13, -0.02] (own, n = 5 + 8 cells).
- Between-cell heterogeneity of the loop rules is small (I2 0.0 to 0.6 for PE and PS AUROC with SE from loci bootstrap), larger for all stripe and methylation / CTCF rules (0.8 to 1.0; effects near zero with spread above sampling error).

### Transfer (transfer_results.csv, transfer_summary.csv, transfer_paired_differences.csv, fig_transfer_matrix.png (A_full), fig_transfer_matrix_Aref.png)
Boosted trees (the classifier settings of section 57) on standardised delta (H3K27ac, H3K27me3; 12 features), consensus-state (12) and locus (10) features, trained on whole groups, locus-disjoint (training rows within 2 bins of a test anchor dropped), per-group gain percentile as regression target; mean over test groups [bootstrap CI over groups]. Controls A_full (A_ref in the CSV and `fig_transfer_matrix_Aref.png`).
- (a) leave-one-cell-out, 14 other cells: loops AUROC 0.69 [0.66, 0.71], gain rho 0.31 [0.25, 0.36] (range 0.49 (PC-9) to 0.79 (HMEC); 14 of 15 cells above 0.5); stripes 0.59 [0.57, 0.60], rho 0.14. Adding CTCF (14 cells): loops 0.82 [0.80, 0.84]. Without consensus-state and locus features (delta + distance + orientation only, RULE): loops 0.62, stripes 0.54; state alone (STATE): 0.67 and 0.58. Block permutation importance (AUROC drop): loops delta 0.054, reference 0.089, locus 0.083; stripes 0.018, 0.065, 0.007.
- (b) all cells to AML donors: loops 0.70 [0.67, 0.73] (private 0.71; with CTCF, 8 donors, 0.81), gain rho 0.09 [0.04, 0.14]; stripes 0.58 [0.55, 0.61], rho 0.10. RULE only: 0.57 and 0.52 (rho -0.06 loops); STATE alone 0.69 and 0.57. AML-trained within AML (leave-one-donor-out, d_aml_only): loops 0.77 [0.73, 0.80] (private 0.78), rho 0.30; stripes 0.62, rho 0.25.
- (c) AML donors to cells: loops 0.54 [0.51, 0.57], rho -0.15 [-0.18, -0.11] (with CTCF 0.70 on 14 cells); stripes 0.52, rho -0.01. The gain ranking learned in AML is inverted in cells.
- (d) held-out donor, AML-only vs AML + all cells: loops AUROC 0.766 vs 0.738 (difference -0.028 [-0.040, -0.016], 0/10 donors better), private loops -0.030 [-0.042, -0.020], gain rho 0.30 vs 0.19 (-0.11 [-0.15, -0.07]); with the AML rows weighted to equal total weight the difference is -0.012 [-0.017, -0.007] (rho -0.04 [-0.07, -0.01]); stripes -0.027 [-0.057, 0.000] and -0.009 [-0.022, 0.005]. Cells added to the AML training do not raise the private-loop AUROC or the gain Spearman. (e) held-out cell, other cells vs other cells + AML: -0.008 [-0.013, -0.004] AUROC (loops), 0.000 for RULE; stripes -0.001.
- Locus-disjoint caveat: A_full controls are matched on own H3K27ac but not on reference state, so reference-level and locus features carry most of the cross-cell AUROC (STATE-only 0.67 of 0.69 in (a), 0.69 of 0.70 in (b)).
- Sign / size consistency (lr_coefficients.csv, lr_sign_consistency.csv; per-group logistic regression on the standardised deltas): among 12 loop delta features, 4 have the same sign in cells and AML in the mean coefficient with at least 70 percent of groups agreeing in neither domain consistently (z H3K27ac 100 kb max positive in 80 percent of cells and AML); the 10 kb and +-20 kb minimum H3K27ac coefficients are negative in 15/15 cells (-0.13, 0.00 positive) and positive in 9/10 donors (+0.11): opposite sign. Stripes: only 4 of 12 same sign.

### Verdict
Universal (same sign, comparable size, cells and AML): the weak separation of cell-specific loops from shared loops by delta H3K27ac at the anchors (PE AUROC 0.56 cells, 0.58 AML; all loops 0.55 vs 0.61), and the absence of any CTCF effect and of any positive relation of delta marks with the gain magnitude of loops. Cell-type (or reference) specific: the stripe gain vs delta H3K27ac (rho 0.28 AML, -0.06 cells), the sign of the consensus H3K27ac level effect on the gain (-0.11 AML, +0.03 cells), the classifier ranking of the gain (AML to cells rho -0.15), the PS / repressive-stripe / enhancer-stripe separations, which exist in every cell only when the class is defined from the same own marks as the feature (circular). Methylation: no support in cells (0.50) and weak in AML (0.57). Cancer vs normal: no key rule differs. The data do not support one rule for all cell types for stripes or for the gain magnitude; for loops a shared weak delta H3K27ac component exists, and the transfer AUROC (0.70 cells to AML) is carried by consensus-state and locus features. More cell types (17 groups) do not help AML (d), so the AML-specific part is not the shared part.

### Inventory
`dataset_inventory.md` (web search, nothing downloaded). Usable: GSE133928 (colon tumour/normal Hi-C 40 kb hg19 + H3K27ac/H3K27me3/CTCF + bisulfite, with effort), ENCODE4 cancer lines with Hi-C + marks (A673, Caco-2, HeLa-S3, Panc1, PC-3, SJCRH30, HAP-1; hg38), GSE231486 HSC/CMP/GMP/MEP (Hi-C + all marks, hg38), GSE137188 CRC cohort (Hi-C only), GSE66733 MCF-10A vs MCF-7 (Hi-C only). T-ALL (GSE115896: no Hi-C matrices), CLL (EGA controlled), glioma (no Hi-C) not realistic.

### Caveats
Specific-object thresholds (0.5 / 0.30, 0.55 / 0.20) and the stripe geometry (fixed extent 50 to 90 bins) are my choices; cell stripes are fixed-geometry rectangles, not Xu calls, which may explain the missing stripe-gain signal; the LOO consensus is a mean of O/E maps, so gains are inflated for any loop not present in most cells, and GM12878 / GM23248 (same lineage) sit in each other's consensus; the HSPC reference of AML is a single differentiation-distant cell type, the consensus of cells is not; positives capped at 2500; CTCF unavailable for GM23248, methylation for 6 cells; the gain regression and AUROC use matched positives only in transfer, all positives in the rules; many rules, intervals not corrected; the sign-consistency sentence is a count of table rows (see lr_sign_consistency.csv for exact values).

### Files
`scripts/crosscell_rule_transfer.py`; `results/crosscell_rules/`: `rules_key.csv`, `rules_all_features.csv.gz`, `rules_consistency_{A_ref,A_full}_{own,inherit,v2inherit}.csv`, `transfer_{results,summary,paired_differences,block_importance}.csv`, `lr_coefficients.csv`, `lr_sign_consistency.csv`, `object_counts*.csv`, `matching_quality_cells.csv`, `delta_standardisation.csv`, `dataset_inventory.md`, `fig_effect_heatmap.png`, `fig_transfer_matrix.png`, `fig_transfer_matrix_Aref.png`.

## 69. Is the loop-focus pilot of section 63 patient-specific? Tests of the gain-over-HSPC shortcut and confound-free criteria (criteria2) (FINAL; section 68 was the last number when this was written)

Script `scripts/aml_loop_detection.py --stage criteria2` (numpy / pandas / scipy only, CPU, no retraining; existing stages unchanged), results in `results/aml_loop_detection_criteria2/` (`<tag>_match_quality.csv`, `_stats.csv`, `_per_donor_marks.csv`, `_calibration.csv`, `_adjusted.csv`, `_criteria2.csv`). Inputs: the pilot of section 63 (`/mnt/processed/wolffjoa/hic-multiome/runs/aml_loopfocus/object_scores_f10_pilot.csv.gz`, model `all12aml_loopfocus_refres_drop_f10_10kb`, test donors 018, 027, 270; 541 private loops, controls A_hspc 674) and the rebuilt-fold object scores of section 62 (`results/aml_loop_detection/object_scores.csv.gz`, 10 donors, 1805 private loops). Pilot tag `f10_pilot`, section 62 tag `aml_refin_10donors`. Intervals: resampling of objects within donor for 3 donors (pilot), donor bootstrap for 10 donors; 1000 replicates.

### Hypotheses and results (pilot, private loops, 3 donors)
- Gain over HSPC is a proxy of the HSPC level. The measured HSPC enrichment alone separates private loops from the A_hspc controls with AUROC 0.991 (reference enrichment alone 0.966); positives have median reference 0.24 and HSPC 0.15 (95th percentile of HSPC 0.57), the A_hspc controls 0.77 and 0.71 (5th percentile of HSPC 0.61): the two classes do not overlap in these covariates. The level of every model (aml_refin, aml_loopfocus, reference) is lower at the positives (level AUROC 0.05 to 0.08, i.e. 0.92 to 0.95 inverted), and the gain AUROC is the level difference minus a HSPC difference: a model that tracks the strong shared loops less closely scores higher. Hypothesis (2): confirmed. The gain AUROC of aml_loopfocus (0.901 [0.883, 0.917]) is below the trivial score minus-HSPC.
- Marks do not enter the gain AUROC. Marks minus no marks, gain AUROC vs A_hspc: aml_loopfocus -0.002 [-0.007, 0.002], aml_refin -0.002 [-0.016, 0.013]; per donor -0.002, -0.008, -0.002 (aml_loopfocus). Level and residual AUROC vs A_hspc likewise within 0.012 of zero (residual aml_loopfocus -0.007 [-0.015, 0.000]). Hypothesis (1): confirmed.
- Controls matched on reference and HSPC enrichment, distance (0.10 log10) and CTCF orientation (calipers 0.15 log2). A_ref from the shared-loop pool (A_hspc, A_full, A_do) retains 27 of 541 positives (5 percent; 100 of 1805 in the 10 donors), HSPC SMD -0.49 after matching (-0.62): the shared loops cannot be matched to AML-specific loops on these covariates; reported (aml_loopfocus gain 0.667 [0.512, 0.803], aml_refin 0.613 [0.462, 0.756]) but not used for criteria. A_refrand (random distance-matched pairs B_rand, same matching) retains 398 (74 percent; 1302 of 1805), SMD reference 0.17 and HSPC -0.02, remaining AUROC of minus-HSPC 0.50 and minus-reference 0.44. On A_refrand the gain AUROC is aml_loopfocus 0.620 [0.582, 0.658] against aml_refin 0.678 [0.641, 0.714]: the advantage of 0.36 on A_hspc becomes -0.06; the whole advantage is explained by the level difference of the controls. Residual AUROC (model minus reference) on A_refrand: aml_loopfocus 0.581 [0.540, 0.621], aml_refin 0.664 [0.627, 0.701], aml_refin 10 donors 0.663 [0.641, 0.681], reference 0.50 by construction. Adjusting the scores linearly (quadratic) for reference and HSPC reduces the A_hspc gain AUROC of aml_loopfocus to 0.576 and of aml_refin to 0.467 (extrapolation, descriptive).
- Hypothesis (3), unmatched random pairs: level AUROC vs B_rand aml_loopfocus 0.769, aml_refin 0.794, reference 0.740; gain 0.722, 0.738, 0.698. aml_loopfocus is not better than aml_refin or the reference.
- Marks effect where it can be seen (A_refrand, residual AUROC, marks minus no marks): aml_refin 0.062 [0.044, 0.083] (per donor 0.026, 0.055, 0.088; 10 donors 0.052 [0.031, 0.069]), aml_loopfocus 0.013 [0.000, 0.026] (per donor 0.013, 0.014, 0.012). The refinement model uses the patient's marks; the loop-focus model uses them much less.
- Residual correlation among private loops (predicted minus reference vs measured minus reference): aml_loopfocus 0.210 [0.124, 0.294] (no marks 0.201), partial on reference and HSPC ranks 0.138 [0.049, 0.222]; aml_refin 0.137 [0.051, 0.217] (no marks 0.068), partial 0.053 [-0.029, 0.130]; base_refin@aml 0.215, partial 0.188; aml_refin 10 donors 0.182 [0.101, 0.266], partial 0.070. The aml_loopfocus value is the same without marks (0.201): not patient-specific.
- Calibration (median level, log2 enrichment): random pairs measured 0.167, aml_loopfocus 0.155, aml_refin 0.013; A_hspc controls measured 0.887, aml_loopfocus 0.670, aml_refin 0.760; private loops measured 0.554, aml_loopfocus 0.319, aml_refin 0.181. Relative to the reference map aml_loopfocus adds +0.093 at private loops, +0.054 at random pairs and -0.107 at A_hspc controls (aml_refin -0.026, -0.043, -0.011): a flatter dynamic range (strong loops lowered, weak raised), closer to the measured level at random pairs, no excess over the measured map (median excess -0.012, 90th percentile -0.125). Hypothesis (d), inflation: refuted for the level; the range compression is the mechanism of the high gain AUROC.
- Stripes, enhancer-like (n = 101): gain Spearman aml_loopfocus 0.619, reference 0.615 (difference 0.004 [-0.050, 0.067]), aml_refin 0.671 (0.055 [-0.028, 0.139]; 10 donors 0.558, 0.095 [-0.004, 0.194]). No model exceeds the reference with a bound above zero. Residual Spearman of the enhancer stripes: aml_loopfocus 0.173 [-0.009, 0.356] (no marks 0.061), aml_refin 0.331 [0.157, 0.500] (no marks 0.093).
- Model run with the HSPC reference (aml_refin@hspc exists for aml_refin only): the model-own gain (aml_refin minus aml_refin@hspc) gives AUROC 0.297 vs A_hspc, 0.636 vs A_refrand.

### Verdict
The pilot improvement of section 63 is an artifact of the score, not a patient-specific gain. The loop-focus model carries no extra patient information (marks effect 0.013 against 0.062 for aml_refin; the gain AUROC identical with and without marks), is worse than aml_refin against reference-matched controls (residual AUROC 0.58 vs 0.66), and its stripe gain does not exceed the reference. Real but small: the residual Spearman among private loops (0.21, partial 0.14) exists, yet it is equally present without marks, so it reflects sequence / reference rather than the patient. The registered pass of section 63 stands as computed (0.901, 43 percent) and is superseded by criteria2.

Follow-up (section 70): the recipe was redesigned so that the shortcut is unavailable (A_refrand controls for sampling, object loss and ranking; marks-dependence margin) and is to be judged by criteria2 on the fold-10 donors before any other run.

### criteria2 (stage `criteria2`)
Per model with a '|none' partner, private loops. C1 marks effect: residual AUROC vs A_refrand with marks minus without >= 0.03 and CI lower bound >= 0. C2: residual AUROC vs A_refrand >= 0.70. C3: Spearman of predicted and measured residual among private loops >= 0.20 (partial Spearman reported). C4: no inflation at random pairs, median(model) - median(measured) <= 0.05 and 90th percentile <= 0.10 (log2). C0 (section 63, gain AUROC vs A_hspc >= 0.70) is reported only. Verdict = C1 to C4 all true. Results: aml_loopfocus pilot C1 0.013 (false), C2 0.581 (false), C3 0.210 (true), C4 true: FAIL. aml_refin pilot C1 true (0.062), C2 0.664 false, C3 0.137 false, C4 true: FAIL; aml_refin 10 donors C1 true (0.052 [0.031, 0.069]), C2 0.663, C3 0.182, C4 true: FAIL. C1 is passed by the model that uses marks (aml_refin), and C2 and C3 are not reached by any model yet.

Run for a new object_scores table (hicdiff env or any env with numpy, pandas, scipy; CPU, about 30 s):
`nice -n 10 python scripts/aml_loop_detection.py --stage criteria2 --scores <object_scores.csv.gz> --tag <name> [--nboot 1000]` (tables to `results/aml_loop_detection_criteria2/<name>_*.csv`; the table must contain the base model column and its `|none` column, e.g. `aml_diff_loopfocus` and `aml_diff_loopfocus|none`, produced by `--recipe diff_loopfocus --stage score`; with `--recipe` set and no `--scores`, `object_scores.csv.gz` of the recipe folder is read).

### Caveats
Three pilot donors: object-level intervals ignore donor heterogeneity (per-donor values in `_per_donor_marks.csv`). A_ref (shared loops) cannot be matched; A_refrand is made of random pairs, so C1 and C2 test loop-ness at equal reference / HSPC level, not shared-loop discrimination. Thresholds (0.03, 0.70, 0.20, 0.05, 0.10) and calibers (0.15 log2, 0.10 log10) are my choices; C2 (0.70) has not been reached by any model. Matching is greedy, one-to-one, fixed across bootstrap replicates. The residual AUROC of the reference is 0.5 by construction. Section numbering: the registry holds two sections headed 63 (lines 7246 and 8730).

### Files
`scripts/aml_loop_detection.py` (stage criteria2), `results/aml_loop_detection_criteria2/{f10_pilot,aml_refin_10donors}_{match_quality,stats,per_donor_marks,calibration,adjusted,criteria2}.csv`.

## 70. Loop-centric recipe v2: controls that the reference / HSPC level cannot separate, and a marks-dependence term (PENDING; section 69 was the last number when this was written)

Question: section 69 shows that the section 63 pilot (private-loop gain AUROC 0.90) learned the shortcut "residual positive where the reference / HSPC enrichment is weak" (the A_hspc shared-loop controls are separable by HSPC enrichment alone, AUROC 0.991, reference 0.966), lost the use of the patient's marks (criteria2 C1 0.013 against 0.062 for aml_refin) and reached residual AUROC 0.58 against reference/HSPC-matched random pairs. The recipe is redesigned so that the shortcut is unavailable. No GPU run has been made.

### Changes
1. **Controls.** New control set `A_refrand` (`scripts/refrand_controls.py`, called by `scripts/aml_objects_table.py --refrand`): for every AML-specific loop / stripe of a TRAINING patient a random object of the same kind on the same chromosome, matched on the AML reference enrichment (the leave-one-out reference the model reads), the measured HSPC consensus enrichment (both within 0.15; loops log2 donut enrichment, 3 x 3 maximum; stripes mean log O/E of the rectangle), log10 distance within 0.10 (stripes: identical geometry) and CTCF orientation class (exact); never within 2 bins of an AML-specific loop (stripes: 8 bins of an anchor), on bins measured in the patient, one control per object, none reused, drawn at random among the 8 closest free candidates; the same calipers as the `A_refrand` pool of criteria2. Tables `objects2_f<N>.csv.gz` (share `aml/objects/`, repo `results/aml_objects/`) add the sets `A_refrand` and the columns `ref_val`, `hspc_val`; training patients only, leakage rules and tests as section 63. `train.aml_object_controls = [A_refrand]` drives window pairing / oversampling (x5), the pair batches, the object loss and the ranking.
2. **Shared loops as a weak term.** `train.aml_object_weak_controls = [A_hspc, A_ref]` (role 3): neighbourhood and enrichment terms at `aml_object_weak_weight` 0.2; never ranked, never sampled.
3. **Marks dependence.** `train.aml_object_marks_pass`: object batches that hold a ranking pair see all of the sample's marks (no random track dropout) and are run a second time with the marks replaced by the absent encoding (no gradient, also through the differential-channel rows). `aml_object_marks_margin` 0.5 x relu(d_without - d_with + 0.02), d = score of the AML-specific object minus score of the control (residual space, same consistent pairs as the ranking); `obj_gap` (mean d_with - d_without) and `obj_margin` are logged each epoch (weight 0 = diagnostic only). Needs `track_dropout` > 0.
4. **Evaluation.** criteria2 on the held-out donors (C1 marks effect on the residual AUROC against A_refrand >= 0.03 with CI lower bound >= 0, C2 residual AUROC >= 0.70, C3 residual Spearman among private loops >= 0.20, C4 no inflation at random pairs); `aml_loop_detection.py --recipe loopfocus2` / `diff_loopfocus2`, labels `aml_loopfocus2`, `aml_diff_loopfocus2`, `aml_diff_loopfocus2aml`, `aml_diffmeth_loopfocus2`, `aml_diffhspc_loopfocus2` (also in `aml_train_eval.py --variant extra`).
5. **Run families.** `all12aml_loopfocus2_refres_drop_f{10,11,12}` (10 and 25 kb; cache of the refres runs) and, for the differential-channel models, `all12{diff,diffmeth,diffhspc}_loopfocus2_refres_drop_f<N>` (AML table v2 + cell-line table v2, scope all) and `..._loopfocus2aml_...` (AML objects only), `all12diffcells_loopfocus2_refres_drop` (12 cells). The old `*_loopfocus*` names keep the old tables. The cell-line table gets the same control set (`scripts/cell_objects_refrand.py`, reference = leave-one-out consensus, no HSPC term, `cell_objects_refrand.csv.gz` on the share): 12 cell lines, 400 specific loops and 200 specific stripes each; 4799 of 4800 loops and all 2400 stripes have a control, the mean reference enrichment of objects and controls agrees to 0.002 (`results/cell_objects/refrand_match_quality.csv`).

### Tables
A_refrand per fold (objects with a control / objects): loops 1591/1591, 1320/1320, 1641/1641; stripes 717/729, 686/700, 707/721 (folds 10, 11, 12). After matching, the AUROC of the reference and of the HSPC enrichment between AML-specific objects and their controls is 0.497 to 0.501 (before, against A_hspc: 0.97 / 0.99); maximum differences 0.15 (the caliper); `results/aml_objects/refrand_match_quality.csv`. On the real fold-10 10 kb cache 1114 training windows hold AML-specific objects and 1118 hold controls of the training set; 1025 one-to-one window pairs (490 with the object's own matched control, the rest an unused A_refrand window of the same patient and kind), so x5 oversampling adds 8200 windows (+20 percent).

### Tests
`tests/test_aml_objects2.py` (19): calipers, orientation, avoid distance, uniqueness and absence of the reference / HSPC shortcut in the matched controls on synthetic maps (loops and stripes), weak roles in sampling, geometry and ranking, marks margin (penalises an identical margin without marks, passes a marks gain, gradient direction, diagnostic-only mode), config validation, run configs of folds 10 to 12 at 10 and 25 kb and of all diff variants (tables, controls, cache identical to the old runs), written tables (leakage, one control per object, calipers), and a 60-step CPU smoke with all flags. Full suite passes.

### Job and estimates
`results/aml_loopfocus/REMOTE_JOB.md` (version 2, first part): pilot = `all12aml_loopfocus2_refres_drop_f10_10kb` alone, evaluated with criteria2 on donors 018, 027, 270; stop if C1 or C2 fails. Estimated remote 4090 time: pilot 2.2 to 3.3 h (+0.5 h evaluation); folds 11, 12 at 10 kb 2.8 to 4.2 h; 25 kb 1.8 to 3.5 h; diff_loopfocus2 10 kb about 6.5 h (5 to 9.5 h).

### Risks
The marks term can only push the with-marks pass up; if the marks carry little information beyond the reference (aml_refin C1 is +0.06) the margin cannot be met and the term is a constant penalty (watch `obj_gap`). Object batches train without track dropout (all marks) which shifts their input distribution towards complete marks. Matching uses the leave-one-out reference of the training patient; the reference of a test patient is the mean of all training patients, a slightly different map. Weights (1.0, 0.5, 0.5, 0.02, x5, weak 0.2) are untuned; one seed. Whether random matched controls are also the right negatives for stripes (rectangle means) is untested.

### Files
`scripts/refrand_controls.py`, `scripts/cell_objects_refrand.py`, `scripts/aml_objects_table.py`, `scripts/multicell_expanded.py`, `scripts/aml_loop_detection.py`, `scripts/aml_train_eval.py`, `scripts/cell_objects_eval.py`, `hicmultiome/{aml_objects,config,train}.py`, `tests/test_aml_objects2.py`, `results/aml_objects/{refrand_match_quality,summary2}.csv`, `results/aml_loopfocus/{REMOTE_JOB.md,configs2/}`.

## 71. Input-information ceiling of the criteria2 loop test (FINAL; section 70 was the last number when this was written)

Question: how well does a flexible feature-based classifier separate the 1805 AML-specific private loops of the ten test donors from A_refrand controls (random pairs matched on reference and HSPC enrichment, log10 distance and orientation; the 1302 matched pairs of section 69, reproduced exactly) when it may use every patient-level input that exists? If it is far above the Hi-C models (residual AUROC 0.66 to 0.69) the models leave information unused.

### Design
`scripts/aml_input_ceiling.py`. Features per object (no feature uses the Hi-C of the target): a level (AML reference, HSPC consensus enrichment; 2), b sequence / orientation / distance / gene annotation / CpG islands (44), x HSPC context (HSPC marks, healthy methylation, HSPC ATAC and RNA; 318), c own H3K27ac / H3K27me3 / CTCF at bin, +-20 kb, +-100 kb with A, B, min, max, sum, product, promoter / partner, mid (91), d patient minus HSPC marks in six reference variants (436), e WGBS (254), f ATAC multiscale and ATAC x H3K27ac (54), g anchor-gene RNA multiscale (69); sources `results/aml_loop_marks` features.pkl, `aml_loop_classifier_v2`, `aml_private_loop_info` new_features plus multiscale ATAC / RNA computed here. HGB (depth 3, 150 iterations) and L2 logistic regression; leave-one-donor-out, training rows within 2 bins of a test anchor dropped; pooled AUROC with donor bootstrap (1000), paired contrasts on the same resamples. Regression (HGB, ridge) of the measured residual (patient minus reference) among private loops for the Spearman ceiling. Datasets: private loops (1302 pairs), all AML-specific loops (1407 pairs), enhancer-like stripes (only 91 matched pairs; underpowered).

### Results, private loops vs A_refrand (AUROC, donor bootstrap 95 percent interval)
| features | HGB | logistic |
|---|---|---|
| a level only (check) | 0.559 [0.544, 0.572] | 0.567 [0.558, 0.576] |
| b + sequence / orientation / distance | 0.598 | 0.583 |
| x + HSPC context | 0.708 [0.649, 0.766] | 0.690 [0.641, 0.734] |
| c + own marks | 0.768 | 0.737 |
| d + patient-minus-HSPC marks | 0.768 | 0.733 |
| e + methylation | 0.767 | 0.720 |
| f + ATAC | 0.769 | 0.726 |
| g + RNA (= everything) | 0.771 [0.692, 0.845] | 0.728 [0.664, 0.782] |
| everything without own-patient features (a, b, x) | 0.708 | 0.690 |
| labels permuted within donor | 0.493 | |
| Hi-C models | aml_refin 0.663 [0.640, 0.683], aml_diff 0.689 [0.653, 0.713], aml_diffhspc 0.667 [0.642, 0.689] | |

All variants of the same row set; all-AML loops give 0.779 / 0.734 (Hi-C 0.666 / 0.686 / 0.668), enhancer-like stripes 0.836 / 0.756 (Hi-C 0.66; n = 91 pairs, intervals 0.78 to 0.88 / 0.66 to 0.85). Marks effect of the classifier (everything minus without own-patient features): HGB +0.063 [0.028, 0.096], logistic +0.038 [0.003, 0.073]; Hi-C models +0.052 / +0.068 / +0.050. Classifier minus aml_diff: HGB +0.083 [0.027, 0.141], logistic +0.039 [-0.003, 0.076]; minus aml_refin +0.108 [0.040, 0.178] / +0.064 [0.011, 0.114]. Increments (HGB): x +0.110 [0.059, 0.156], c +0.059 [0.028, 0.086], b +0.039 [0.003, 0.074], d / e / f / g within +-0.003 and intervals containing 0; drop-one differences of every group are below 0.01. Single groups with the level (HGB): c 0.732, d 0.739, x 0.694 (HSPC marks 0.688, healthy methylation 0.651, HSPC ATAC / RNA 0.632), f 0.716, g 0.655, e 0.634, b 0.598. Learning curve (HGB, everything): 0.696 / 0.701 / 0.734 / 0.754 / 0.764 / 0.774 with 1 / 2 / 3 / 5 / 7 / 9 training donors, still rising; without own-patient features 0.643 to 0.713.

Residual ranking among the 1805 private loops (Spearman with measured minus reference; partial = given reference and HSPC ranks): regression HGB on everything 0.395 [0.289, 0.490], partial 0.326 [0.220, 0.424]; ridge 0.313, partial 0.248; regression without own-patient features 0.360, partial 0.280. Hi-C models: aml_refin 0.182 (partial 0.070), aml_diff 0.121 (0.038), aml_diffhspc 0.120 (0.031). The classifier score itself correlates negatively with the residual (-0.21 HGB, -0.18 logistic; partial -0.21 / -0.17).

### Verdict
The Hi-C models are below the input ceiling of this test: gradient boosting reaches 0.771 (logistic 0.728), 0.08 (0.04) above aml_diff and 0.11 (0.06) above aml_refin, and the regression ceiling for the residual Spearman (partial 0.33) exceeds C3 (0.20) while the Hi-C models reach 0.03 to 0.07 partial. The gap is not patient-specific information: patient-independent features alone (a, b, x) give 0.708 (0.690) and the classifier's marks effect (+0.04 to +0.06) equals that of the Hi-C models (+0.05 to +0.07). The AUROC is carried by HSPC context (+0.11; HSPC marks, then healthy methylation and HSPC ATAC / RNA; the HSPC marks mark the loop anchors as regulatory elements against random pairs), own marks (+0.06) and sequence / distance (+0.04); patient-minus-HSPC marks, methylation, ATAC and RNA add nothing beyond the own marks (each +0.003 or less, drop-one below 0.01). C2 (0.70) is reachable by a classifier on the existing inputs (0.77 HGB, interval 0.69 to 0.85; the logistic 0.73 interval includes 0.70), so the inputs do not cap the test at 0.69.

### Caveats
Matching leaves a residual reference difference (AUROC of the reference 0.545 between positives and controls inside the 0.15 caliper), so the level-only check is 0.56 and not 0.50; this is included in every classifier value but is not in the Hi-C residual scores. The classifier is trained directly on this contrast (pos against random pairs) while the Hi-C models learn a map and are scored on it afterwards; its 0.77 is a ceiling for features, not a model that predicts Hi-C. Nine training donors only (the Hi-C models see more training patients), one seed, fixed hyper-parameters, no monotone constraints, many correlated features (1268) on 2600 rows; the partial Spearman residualises ranks linearly and keeps the residualisation fixed across bootstrap resamples. Patient-minus-AML-mean references (diffA) use the marks of the other nine donors (no labels). aml_loopfocus has missing scores for most matched rows in the T1b table and is left out. Donors 1360 and 1916 have no CTCF, WGBS or RNA (NaN columns).

### Recommendations
Feed the Hi-C model the HSPC marks (it reads the patient's own marks and the HSPC reference Hi-C only) and the CTCF motif / orientation and gene annotation tracks as inputs, and train with the ranking of A_refrand controls (section 70) so that the marks-based locus-activity signal enters the residual; the data do not support extra patient-minus-HSPC, methylation, ATAC or RNA channels for this test (no increment). More training donors would help (learning curve not saturated). Quantity prediction (C3) has the larger gap (regression partial 0.33 against 0.03 to 0.07), and about 0.28 of it comes from patient-independent features.

### Files
`scripts/aml_input_ceiling.py`, `results/aml_input_ceiling/{table_main,auroc_all_variants,auroc_per_donor,paired_contrasts,spearman,learning_curve,dataset_summary,feature_groups}.csv`, `oof_scores.csv.gz`, `fig_input_ceiling.png`.


## 72. Do published Hi-C predictors detect the AML-specific loops and stripes? (FINAL; section 71 was the last number when this was written)

Question: on the objects and the rule of section 62 / 66 (10 held-out donors, 2276 AML-specific loops, 1075 stripes; share of objects whose predicted enrichment exceeds the 90th percentile of the same predictor at the B_rand pairs of the same donors), do C.Origami, Epiphany, Akita and (not run) HiC-Reg detect them, and how do they compare with the consensus and with our models?

### Design
`scripts/aml_competitors_loops.py` (stages ours, corigami, epiphany, akita, score, merge, stats, summary, figure). Objects, windows (192 bins of 10 kb centred on the object) and the existing scores (measured, consensus = AML reference map, HSPC, aml_refin, aml_diff, aml_diffhspc, base_refin@aml) are those of `runs/diff_marks/aml_loop_detection_diff_loopfocus_after_T1b/object_scores.csv.gz`; the enrichment estimator is the one of `aml_loop_detection.stage_score` (donut inner 1 / outer 4 on exp(log O/E) as log2, maximum over the 3 x 3 pixels around the annotated pixel for loops; mean log O/E over the anchor x extent rectangle for stripes). The detection rates reproduce the quoted values (truth 61 / 76 percent, consensus 44 / 68, aml_refin 49 / 70, aml_diff 51 / 69). Intervals: 95 percent, donors resampled with replacement (500), the threshold recomputed in every resample.

Models without any AML sample in training (the rows to compare with the competitors): (a) base_refin@aml, the 12-cell model all12refin_drop with the AML reference map as Hi-C input (AML data as input only); (b) all12refin_drop_10kb and all12refres_drop_10kb with consensus12 (12 cell lines) as reference; (c) all12diffcells_refres_drop_10kb (marks relative to the mean of the 12 cell lines, `--diff-group cell`) with consensus12. Predicted with `crosscell_predict.py --tta` on the same windows, patient marks CTCF / H3K27ac / H3K27me3 (1360 and 1916 lack CTCF, handled as absent marks as in section 62), one GPU, inference only. Reference maps read: consensus = AML reference of the fold's training patients (contains AML data); consensus12 = mean of the 12 training cell lines (no AML); base_refin@aml reads the AML reference; the three c12 models read consensus12; aml_* models read the AML reference.

Competitors (released weights, `/mnt/processed/wolffjoa/galaxy-hic-models`, nothing retrained), each converted by the section 18 / 58 recipe applied per window: counts (C.Origami expm1; Epiphany 2^x - 1) -> expected(d) = mean over the pairs of each diagonal -> log(obs / exp), zero count = -2, clip +-2 -> per-diagonal centring -> clip -> Gaussian sigma 2 native bins -> band to the model's reach (C.Origami 256 bins, Epiphany 99 bins) -> bilinear sampling (in bin, bin space) at the 10 kb bin centres of the window, a pixel kept only when no interpolation weight touches a missing pixel. Akita: log2 O/E of its target x ln 2, then the same centring / smoothing / sampling (no count step). The expected and the centring offsets come from the 256 / 200 / 448 bin window of the prediction, not from the whole chromosome (the chromosome-wide expected of `oe_proper.py` needs genome-wide predictions per patient); donut and rectangle estimators are local, so only the level per diagonal differs.
- C.Origami (`corigami_base.ckpt`, IMR-90, hg38, 2 097 152 bp window, 8192 bp bins): window centred on the object, hg38 sequence of the C.Origami folder, CTCF = `inputs/xu_<p>/CTCF.bigwig` (CUT&Tag), ATAC = `atac/GSM*_<p>_ATAC-seq.10bp.RPGC.bigwig`; code normalisation kept (ATAC log(x + 1), CTCF as is). C.Origami expects CTCF ChIP-seq log2 fold change over input and an RPKM ATAC track (IMR-90 tracks: CTCF mean -0.37, ATAC mean 31); the patient tracks (CUT&Tag p-value signal, RPGC mean 1) are on other scales. Two variants: `q` (monotone quantile map of each track to the IMR-90 training track, 600 random 25 kb segments, then the code normalisation) and `raw` (tracks as given). Patients: 018, 027, 168, 270, 472, 546, 629, 773 (8; 1360 and 1916 have no CTCF).
- Epiphany (`pretrained_10kb.pt_model`, GM12878, 10 kb, 990 kb reach; checked against the released `assemble_matrix_chr3.cool`, r = 0.98 on chr3:70-90 Mb): channels DNaseI <- ATAC (substitute), H3K27ac, H3K4me3, H3K27me3, CTCF, 100 bp means of the bigWigs scaled to a genome mean of 1; H3K4me3 is absent for every patient: variants `zero` (0) and `mean` (1.0). The model has a fixed 5-channel input, so a 4-input run is not possible. Same 8 patients. One forward pass of 200 rows per window (3.4 Mb input).
- Akita (`model_best.h5`, hg38, 1 048 576 bp, 448 central bins of 2048 bp, head 2 = GM12878; forward and reverse-complement averaged): sequence only, identical for every patient (a sequence baseline). 917 kb of coverage: objects (and stripes) wider than about 800 kb are not scored (loops 2228 of 2276, stripes 605 of 1075).
- HiC-Reg: not run. The retrained forests of `hicreg_retrain.py` were not persisted (`stage_fit` writes the predictions and a json, no model file; no joblib / pickle exists under `runs/benchmark/hicreg_retrain/`), so applying them to AML inputs needs a refit (about 30 min for the window model), which the task excluded; the released HiC-Reg forests (5 kb, HMEC to GM12878 chr14 and CV models) need DNase and 10 further tracks of which patients have two. Hi-cGAN: not run, the 10 kb model needs CTCF + H3K4me2 and the 5 kb model RAD21, H3K4me3 and SMC3 (none exist for the patients; only the 25 kb H3K27ac + CTCF model could be formed, coarser than the donut of 10 to 40 kb).
- Healthy contrast: each predictor on the HSPC CD34+ inputs of the same windows (`inputs/hspc_cd34` CTCF / H3K27ac / H3K27me3; ATAC = mean of the five HSPC ATAC replicates, `atac/hspc`); gain = prediction minus the same predictor on HSPC (truth, consensus, aml_* models: minus the measured HSPC consensus). Akita has no contrast (patient-independent).

### Results
Detection rate, share of AML-specific objects above the 90th percentile of the predictor's own B_rand values (false-positive rate 10 percent by construction; each predictor on the objects it covers; 95 percent interval over donors):

| predictor | loops all | loops private | loops recurrent | stripes all | stripes private | stripes recurrent |
|---|---|---|---|---|---|---|
| truth (measured) | 0.61 (0.55-0.68) | 0.58 (0.52-0.66) | 0.85 (0.81-0.90) | 0.76 (0.72-0.81) | 0.64 (0.57-0.71) | 0.83 (0.77-0.86) |
| consensus (AML reference) | 0.44 (0.35-0.52) | 0.37 (0.29-0.47) | 0.84 (0.81-0.89) | 0.68 (0.63-0.76) | 0.48 (0.41-0.59) | 0.80 (0.75-0.86) |
| consensus12 (no AML) | 0.38 (0.29-0.48) | 0.34 (0.25-0.43) | 0.65 (0.54-0.73) | 0.55 (0.47-0.60) | 0.41 (0.32-0.49) | 0.62 (0.53-0.68) |
| 12-cell, AML reference input | 0.48 (0.39-0.56) | 0.42 (0.34-0.49) | 0.85 (0.80-0.92) | 0.67 (0.63-0.73) | 0.49 (0.43-0.58) | 0.77 (0.73-0.82) |
| 12-cell refin, consensus12 | 0.42 (0.34-0.50) | 0.38 (0.31-0.46) | 0.64 (0.58-0.74) | 0.56 (0.49-0.63) | 0.44 (0.35-0.52) | 0.62 (0.54-0.70) |
| 12-cell refres, consensus12 | 0.42 (0.34-0.52) | 0.38 (0.29-0.47) | 0.65 (0.58-0.74) | 0.57 (0.50-0.64) | 0.44 (0.37-0.52) | 0.63 (0.55-0.72) |
| 12-cell differential, consensus12 | 0.40 (0.32-0.50) | 0.37 (0.29-0.47) | 0.62 (0.51-0.72) | 0.52 (0.46-0.58) | 0.41 (0.33-0.49) | 0.58 (0.51-0.65) |
| C.Origami, quantile-matched inputs | 0.21 (0.19-0.24) | 0.19 (0.16-0.22) | 0.36 (0.30-0.41) | 0.19 (0.15-0.22) | 0.14 (0.10-0.19) | 0.22 (0.18-0.24) |
| C.Origami, inputs as given | 0.13 (0.11-0.15) | 0.13 (0.11-0.14) | 0.12 (0.06-0.18) | 0.17 (0.14-0.20) | 0.16 (0.10-0.20) | 0.19 (0.16-0.22) |
| Epiphany, H3K4me3 = 0 | 0.19 (0.14-0.23) | 0.19 (0.15-0.22) | 0.18 (0.08-0.29) | 0.21 (0.15-0.27) | 0.20 (0.15-0.26) | 0.26 (0.17-0.35) |
| Epiphany, H3K4me3 = mean | 0.18 (0.14-0.22) | 0.18 (0.14-0.22) | 0.19 (0.10-0.30) | 0.22 (0.17-0.28) | 0.20 (0.15-0.25) | 0.28 (0.18-0.38) |
| Akita, sequence only | 0.34 (0.26-0.43) | 0.31 (0.23-0.39) | 0.56 (0.49-0.60) | 0.53 (0.46-0.57) | 0.35 (0.28-0.44) | 0.65 (0.56-0.70) |
| *trained with AML patients (not comparable)* aml_refin | 0.49 (0.40-0.61) | 0.42 (0.34-0.53) | 0.89 (0.84-0.93) | 0.70 (0.67-0.74) | 0.53 (0.47-0.59) | 0.80 (0.77-0.84) |
| aml_diff | 0.51 (0.40-0.62) | 0.44 (0.35-0.54) | 0.88 (0.84-0.92) | 0.69 (0.66-0.73) | 0.51 (0.46-0.57) | 0.79 (0.76-0.84) |
| aml_diffhspc | 0.48 (0.38-0.59) | 0.42 (0.33-0.53) | 0.89 (0.83-0.93) | 0.71 (0.65-0.74) | 0.52 (0.42-0.58) | 0.81 (0.76-0.84) |

Coverage (objects scored, loops / stripes): C.Origami 1886 / 921 (8 patients), Epiphany 1880 / 889 (8 patients), Akita 2228 / 605 (10 patients), all other predictors 2276 / 1073 to 1075 (10 patients). On the objects every competitor scores (`detection_rates.csv`, subset "objects scored by all competitors") the order is unchanged. Further classes (shared1, PE, PS, enh, rep, other) are in `detection_rates.csv`.

AUROC of the predicted level, AML-specific objects against B_rand (all) and against the A_refrand controls matched on AML reference and HSPC enrichment (0.15 each), log10 distance (0.10) and orientation (criteria2 `match_controls`, applied per class to B_rand candidates; for stripes too), donor-bootstrap 95 percent intervals:

| predictor | loops vs B_rand | loops vs A_refrand (all) | loops vs A_refrand (private) | stripes vs B_rand | stripes vs A_refrand (all) |
|---|---|---|---|---|---|
| truth | 0.89 (0.87-0.91) | 0.85 (0.83-0.87) | 0.85 (0.84-0.88) | 0.92 (0.90-0.93) | 0.67 (0.64-0.70) |
| consensus (AML reference) | 0.76 (0.71-0.80) | 0.55 (0.54-0.56) | 0.55 (0.53-0.56) | 0.88 (0.86-0.90) | 0.51 (0.50-0.52) |
| consensus12 | 0.76 (0.70-0.81) | 0.63 (0.60-0.66) | 0.63 (0.60-0.66) | 0.84 (0.81-0.86) | 0.57 (0.52-0.62) |
| 12-cell, AML reference input | 0.80 (0.75-0.84) | 0.64 (0.61-0.66) | 0.63 (0.61-0.65) | 0.89 (0.86-0.90) | 0.55 (0.52-0.58) |
| 12-cell refin, consensus12 | 0.78 (0.74-0.83) | 0.67 (0.64-0.69) | 0.67 (0.63-0.70) | 0.84 (0.81-0.87) | 0.59 (0.54-0.64) |
| 12-cell refres, consensus12 | 0.78 (0.73-0.82) | 0.66 (0.63-0.69) | 0.66 (0.63-0.69) | 0.85 (0.82-0.87) | 0.60 (0.55-0.65) |
| 12-cell differential, consensus12 | 0.78 (0.72-0.82) | 0.66 (0.62-0.68) | 0.66 (0.62-0.69) | 0.83 (0.81-0.85) | 0.56 (0.51-0.61) |
| C.Origami, quantile-matched | 0.60 (0.58-0.62) | 0.56 (0.54-0.58) | 0.56 (0.53-0.59) | 0.60 (0.59-0.62) | 0.53 (0.48-0.59) |
| C.Origami, as given | 0.53 (0.51-0.55) | 0.51 (0.49-0.54) | 0.51 (0.48-0.54) | 0.60 (0.56-0.63) | 0.52 (0.45-0.59) |
| Epiphany, H3K4me3 = 0 | 0.56 (0.52-0.59) | 0.53 (0.50-0.55) | 0.53 (0.50-0.56) | 0.62 (0.56-0.68) | 0.49 (0.46-0.53) |
| Epiphany, H3K4me3 = mean | 0.55 (0.51-0.59) | 0.52 (0.49-0.55) | 0.53 (0.50-0.56) | 0.61 (0.56-0.68) | 0.50 (0.45-0.54) |
| Akita, sequence only | 0.74 (0.69-0.78) | 0.63 (0.60-0.67) | 0.63 (0.59-0.67) | 0.84 (0.81-0.86) | 0.68 (0.64-0.72) |
| aml_refin / aml_diff / aml_diffhspc (AML-trained) | 0.81 / 0.81 / 0.80 | 0.67 / 0.67 / 0.66 | 0.66 / 0.66 / 0.65 | 0.89 / 0.89 / 0.89 | 0.55 / 0.56 / 0.54 |

Spearman of the predicted with the measured enrichment among the AML-specific objects (level; loops all / private, stripes all / private): consensus 0.56 / 0.51, 0.79 / 0.76; consensus12 0.51 / 0.49, 0.70 / 0.71; 12-cell AML reference input 0.56 / 0.52, 0.81 / 0.77; 12-cell consensus12 models 0.49 to 0.51 / 0.48 to 0.49, 0.70 to 0.71 / 0.70 to 0.71; C.Origami 0.10 / 0.09, 0.14 / 0.20 (as given 0.05 / 0.05, 0.09 / 0.07); Epiphany 0.11 to 0.12 / 0.10, 0.12 to 0.13 / 0.09 to 0.10; Akita 0.41 / 0.40, 0.33 / 0.41; aml_refin / aml_diff 0.55 / 0.50, 0.80 / 0.78. Spearman of the contrast with HSPC (predicted gain against measured minus HSPC; `spearman_gain.csv`, loops all): consensus 0.29, 12-cell AML reference input 0.30, consensus12 models 0.09 to 0.10, aml_refin 0.28, aml_diff 0.26, C.Origami 0.01, Epiphany 0.05. Detection rate of the contrast (loops all / stripes all): truth 0.54 / 0.46, consensus 0.32 / 0.26, 12-cell AML reference input 0.42 / 0.23, consensus12 models 0.27 to 0.33 / 0.14 to 0.21, C.Origami 0.19 / 0.15 (as given 0.09 / 0.01), Epiphany 0.12 / 0.09 to 0.11, aml_refin 0.40 / 0.25, aml_diff 0.43 / 0.28; AUROC of the contrast against A_refrand (loops all): truth 0.87, consensus 0.58, consensus12 models 0.59 to 0.63, C.Origami 0.54, Epiphany 0.51, aml_refin / aml_diff 0.69 / 0.70.

Number of patients each predictor was run on: our models, consensus12 and Akita 10; C.Origami 8; Epiphany 8 (1360 and 1916 have no CTCF); HiC-Reg 0; Hi-cGAN 0.

### Verdict
The published predictors detect AML-specific loops and stripes at 1.3 to 2.2 times the false-positive rate of 10 percent (C.Origami 0.13 to 0.21 loops, 0.17 to 0.19 stripes; Epiphany 0.18 to 0.19 and 0.21 to 0.22), with AUROC 0.53 to 0.62 against random pairs and 0.49 to 0.56 against matched controls: a weak detection that exists for recurrent loops with C.Origami (0.36) but is far below the consensus (0.44 / 0.68) and below every no-AML model of ours (0.40 to 0.48 / 0.52 to 0.67). Akita, which sees only sequence, detects 0.34 of the loops and 0.53 of the stripes (of the 605 it covers), above C.Origami and Epiphany with the patient's own marks, and at the level of the consensus12 on loops (0.38) and slightly below it on stripes (0.55); the patient marks, as these two models read them with substituted inputs (see caveats), add nothing that a sequence baseline does not already give. None of the competitors exceeds the consensus on any class, and none carries a patient-specific signal: their Spearman with the measured enrichment is 0.05 to 0.14 (Akita 0.33 to 0.41, which is the sequence-only distance and anchor structure), and the contrast with HSPC correlates 0.01 to 0.05 with the measured contrast against 0.09 to 0.10 for our consensus12 models and 0.26 to 0.30 for the models that read the AML reference. Relative to our models trained without AML (the fair comparison), the 12-cell models that read consensus12 equal the consensus12 itself on loops (0.40 to 0.42 against 0.38) and stripes (0.52 to 0.57 against 0.55), with AUROC against matched loop controls 0.66 to 0.67 against 0.63; the model with the AML reference map input reaches 0.48 / 0.67 and the AML-trained models 0.49 to 0.51 / 0.69 to 0.71, so AML data used as input already provides the whole gain over the consensus12 that training on AML adds. The AML-specific objects that any no-AML predictor recovers beyond its own B_rand threshold are those the reference map already contains (recurrent: 0.62 to 0.85 loops); the private objects reach 0.34 to 0.42 loops and 0.41 to 0.49 stripes against 0.58 / 0.64 for the truth.

### Caveats
(i) Inputs differ from the native ones: C.Origami reads CUT&Tag CTCF (p-value signal) in place of ChIP log2 fold change and an RPGC ATAC track in place of RPKM; the quantile mapping to the IMR-90 distribution is a monotone substitute, not the C.Origami pipeline; a diagnostic on 30 windows of patient 018 (not used to choose a variant) gave a median per-window Pearson of 0.16 to 0.25 against the measured map for the patient inputs and 0.33 to 0.35 for HSPC CTCF with either ATAC, so the CUT&Tag CTCF is the weak input. Epiphany reads ATAC for DNase and has no H3K4me3 (zeros or a constant; no 4-input run exists for the fixed 5-channel model). (ii) Different training data: C.Origami IMR-90, Epiphany GM12878 (trained on chromosomes that include some of the windows' chromosomes; chr3, 11 and 17 are its held-out ones), Akita five cells; our models 12 cell lines. Both competitors are therefore tested out of cell type as ours are. (iii) Per-window expected and centring instead of chromosome-wide; the bilinear resampling from 8192 bp (C.Origami) and 2048 bp (Akita) to 10 kb smooths the donut; Epiphany covers 990 kb and Akita 917 kb (objects beyond are excluded for that predictor only, the subset tables give the others on the same objects). (iv) HiC-Reg and Hi-cGAN were not run (reasons above). (v) The HSPC contrast uses CD34+ inputs of a different assay mix than the patients' (the HSPC CTCF and ATAC differ in depth and processing), so the competitors' contrast also carries a processing difference; our models have the same limitation. (vi) The B_rand percentile is the predictor's own, which removes level offsets; AUROC and Spearman are the offset-free measures. (vii) 10 donors, donor-bootstrap intervals are wide for private stripes (285).

### Files
`scripts/aml_competitors_loops.py`, `results/aml_competitors_loops/{scores.csv.gz, detection_rates, detection_rates_gain, auroc, auroc_gain, spearman, spearman_gain, coverage, match_quality}*.csv`, `summary_tables.md`, `summary_tables_gain.md`, `summary_*.csv`, `fig_aml_competitors_loops.png`; predictions under `/mnt/processed/wolffjoa/hic-multiome/runs/aml_competitors_loops/` (ours, corigami, epiphany, akita; windows as float16 / float32 npz).

## 73. Structure scores of our models against Akita and the other published predictors, one definition on identical pixels (FINAL; section 72 was the last number when this was written)

Question: on TAD, domain, loop, stripe-related and compartment measures, how do our models compare with Akita (main comparison, GM12878, the 413 Akita test regions) and with C.Origami, Epiphany and Hi-cGAN (whole held-out chromosomes and AML windows), computed with one definition per table? HiC-Reg is not part of this table.

### Inventory (what existed before)
- Block A of section 18 (`results/benchmark/A_gm12878_2kb_*`): 411 regions (Hi-cGAN has no prediction in 2 of the 413), Akita, Hi-cGAN, akita_native_honest, akita_native_refres, consensus4 and measured maps of other cells; columns Pearson, SCC, Spearman, insulation Pearson (51 kb), boundary recall of the hicFindTADs boundaries (K-matched), loop recall, APA ratio, anchor AUROC. No boundary precision or F1, no multi-window insulation, no domain-level or stripe-related measure, no compartment measure, no per-region win counts for the structure columns. The 222-region subset of section 1 holds Pearson only.
- Whole chromosomes: `results/hic_measures_chrho/*` and `results/benchmark/chrho_*_table.md` (10, 25 and 5 kb; insulation r, boundary recall and precision of hicFindTADs boundaries, TAD fraction, TAD score MSE, loop recall, APA ratio, compartment preference). Akita has no whole-chromosome map (1 Mb predictions in its own space), so it is absent from the chromosome table.
- AML: `insulation_corr`, `boundary_f1`, `loop_enrich_corr`, `compartment_corr` of `runs/aml_train_remote/fold10..12/eval_f*_patients.csv` for our models only (413 regions, 10 donors); reference maps and competitors have Pearson only (the base_* columns).

### Design
`scripts/structure_scores_vs_akita.py` (stages akita, aml, chrho, report; the exact definitions are in its docstring and in `results/structure_vs_akita/structure_scores_vs_akita.md`). Nothing was rerun: Akita and Hi-cGAN come from `akita_testset_rasters.npz`, our two models from `runs/benchmark/pred/akita_space/*.npz`, consensus4 and the HFF map from block A (2 regions fetched from the 2 kb cools), the AML predictions from `runs/aml_competitors_loops`. One pixel mask per region (upper triangle, at least 2 bins off the diagonal, finite in the target and in every method; Hi-cGAN is scored where it is fully defined, 408 regions). Metrics: Pearson, SCC; insulation correlation at 10, 25, 51, 102 and 250 kb; boundary recall, precision and F1 of insulation minima (51 kb square, prominence 0.5 SD, tolerance 10 kb, strict 4 kb) against the boundaries of the measured target, per region and pooled; boundary-strength correlation; measure of concordance of the domain partitions; K-matched recall of the hicFindTADs boundaries and of the hicDetectLoops loops of the measured map; APA ratio; anchor profile correlation at 20 to 200 kb (stripe proxy); within-window compartment and eigenvector correlations. Paired statistics: ours minus Akita per region, 2000 region bootstrap resamples (seed 0), regions ahead and fraction ahead.

### Results, GM12878, 413 Akita test regions (median over regions; pooled values with 95% interval)

| predictor | regions | Pearson | SCC | ins r 51 kb | ins r 250 kb | bnd F1 | bnd strength r | MoC | stripe r | compart. r | pooled bnd F1 | hFT bnd recall | loop recall | APA ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Akita (GM12878 head) | 413 | 0.548 | 0.510 | 0.688 | 0.847 | 0.500 | 0.608 | 0.625 | 0.697 | 0.710 | 0.436 [0.411, 0.462] | 0.494 [0.471, 0.519] | 0.314 [0.280, 0.346] | 0.264 [0.243, 0.286] |
| ours, ChIP only | 413 | 0.638 | 0.610 | 0.819 | 0.919 | 0.571 | 0.730 | 0.674 | 0.791 | 0.776 | 0.487 [0.457, 0.518] | 0.548 [0.522, 0.573] | 0.383 [0.345, 0.420] | 0.381 [0.356, 0.408] |
| ours, with consensus | 413 | 0.746 | 0.694 | 0.920 | 0.971 | 0.667 | 0.871 | 0.718 | 0.894 | 0.885 | 0.575 [0.545, 0.609] | 0.647 [0.621, 0.672] | 0.571 [0.537, 0.603] | 0.845 [0.814, 0.878] |
| consensus of 4 measured cells | 413 | 0.611 | 0.559 | 0.777 | 0.916 | 0.598 | 0.724 | 0.656 | 0.759 | 0.797 | 0.533 [0.511, 0.556] | 0.589 [0.564, 0.615] | 0.514 [0.479, 0.549] | 1.025 [0.972, 1.076] |
| HFF measured map | 413 | 0.514 | 0.469 | 0.705 | 0.893 | 0.500 | 0.639 | 0.612 | 0.669 | 0.716 | 0.488 [0.469, 0.509] | 0.522 [0.497, 0.548] | 0.418 [0.384, 0.451] | 1.552 [1.453, 1.654] |
| Hi-cGAN (submitted) | 408 | 0.236 | 0.207 | 0.376 | 0.567 | 0.222 | 0.227 | 0.516 | 0.309 | 0.546 | 0.243 [0.225, 0.263] | 0.321 [0.298, 0.344] | 0.137 [0.111, 0.162] | 0.179 [0.146, 0.215] |

Paired against Akita (ours minus Akita, median over regions [95% interval]; regions ahead of 413):

| metric | ChIP only | ahead | with consensus | ahead |
|---|---|---|---|---|
| Pearson | 0.040 [0.029, 0.050] | 274 (0.66) | 0.136 [0.124, 0.149] | 401 (0.97) |
| SCC | 0.061 [0.051, 0.074] | 309 (0.75) | 0.140 [0.128, 0.153] | 405 (0.98) |
| ins r 10 kb | 0.272 [0.254, 0.281] | 397 (0.96) | 0.324 [0.314, 0.343] | 408 (0.99) |
| ins r 51 kb | 0.096 [0.075, 0.116] | 323 (0.78) | 0.190 [0.175, 0.213] | 398 (0.96) |
| ins r 250 kb | 0.026 [0.013, 0.037] | 248 (0.60) | 0.072 [0.055, 0.097] | 331 (0.80) |
| bnd F1 | 0.048 [0.000, 0.083] | 223 (0.61) | 0.127 [0.100, 0.159] | 281 (0.78) |
| bnd strength r | 0.075 [0.054, 0.095] | 266 (0.64) | 0.181 [0.157, 0.199] | 365 (0.88) |
| MoC | 0.026 [0.012, 0.035] | 249 (0.60) | 0.063 [0.048, 0.081] | 307 (0.75) |
| stripe r | 0.077 [0.057, 0.101] | 296 (0.72) | 0.173 [0.146, 0.188] | 389 (0.94) |
| compart. r | 0.029 [0.016, 0.042] | 263 (0.64) | 0.099 [0.081, 0.111] | 376 (0.91) |
| pooled bnd F1 | 0.051 [0.028, 0.075] | | 0.140 [0.118, 0.162] | |
| pooled hFT bnd recall | 0.054 [0.029, 0.080] | | 0.153 [0.129, 0.176] | |
| pooled loop recall | 0.069 [0.025, 0.115] | | 0.257 [0.221, 0.296] | |
| APA ratio | 0.117 [0.086, 0.148] | | 0.581 [0.551, 0.612] | |

Reading. Both models are ahead of Akita on every measure; the with-consensus model in 75 to 99 percent of regions, the ChIP-only model in 58 to 96 percent (ahead fraction 0.58 for the eigenvector, 0.60 for insulation at 250 kb, MoC and strict boundary F1; the per-region boundary F1 interval includes 0, the pooled one does not). The ChIP-only model is level with the consensus of four measured cells on insulation at 102 and 250 kb (difference -0.002 and -0.004), and behind it on pooled boundary recall (0.386 against 0.503), hicFindTADs boundary recall (0.548 against 0.589), loop recall (0.383 against 0.514), APA ratio (0.38 against 1.03) and compartment correlation (0.776 against 0.797). The HFF measured map, a different cell, is ahead of Akita on pooled boundary recall (0.522 against 0.494 hFT), loop recall (0.418 against 0.314) and APA ratio (1.55 against 0.26, an over-enriched map) and behind it on Pearson, SCC and insulation at 10 and 25 kb. Akita has the lowest APA ratio of the non-Hi-cGAN predictors (0.26), so its loop signal is attenuated. Hi-cGAN is the lowest on every measure.

### Whole held-out chromosomes (mean over chromosomes; hicFindTADs boundaries on the prediction against the measured map, 1 bin; K-matched loop recall)

| 10 kb block | predictor | Pearson | ins r | bnd F1 | loop recall | APA ratio | compart. r |
|---|---|---|---|---|---|---|---|
| C.Origami IMR-90, chr2, chr3 | ours, refres | 0.928 | 0.943 | 0.666 | 0.364 | 0.943 | 0.964 |
| | ours, refin | 0.926 | 0.924 | 0.667 | 0.355 | 0.830 | 0.960 |
| | ours, ChIP only | 0.808 | 0.823 | 0.518 | 0.192 | 0.409 | 0.885 |
| | C.Origami | 0.920 | 0.866 | 0.498 | 0.162 | 0.729 | 0.972 |
| Epiphany GM12878, chr3, 11, 17 | ours, refres | 0.919 | 0.892 | 0.683 | 0.474 | 0.949 | 0.956 |
| | ours, ChIP only | 0.760 | 0.715 | 0.502 | 0.220 | 0.340 | 0.844 |
| | Epiphany | 0.512 | 0.318 | 0.226 | 0.052 | 0.265 | 0.776 |
| Hi-cGAN GM12878, chr2, chr14 | ours, refres | 0.924 | 0.890 | 0.640 | 0.407 | 0.945 | 0.960 |
| | ours, ChIP only | 0.797 | 0.726 | 0.492 | 0.164 | 0.295 | 0.876 |
| | Hi-cGAN | 0.328 | 0.260 | 0.125 | 0.031 | 0.135 | 0.700 |
| Hi-cGAN K562, chr2, chr14 | ours, refres | 0.849 | 0.830 | 0.593 | 0.382 | 0.794 | 0.940 |
| | ours, ChIP only | 0.731 | 0.732 | 0.466 | 0.160 | 0.207 | 0.872 |
| | Hi-cGAN (GM12878 model) | 0.125 | 0.339 | 0.127 | 0.008 | 0.088 | 0.519 |

The 25 kb table, refin rows, TAD fraction, TAD score MSE and per-chromosome ahead counts are in `results/structure_vs_akita/structure_scores_vs_akita.md` and `chrho_*_structure.csv`. Against C.Origami, the with-consensus models are ahead on insulation, boundary F1 and loop recall on both chromosomes (2/2) at 10 and 25 kb and level on Pearson (25 kb: 0.940 against 0.942); C.Origami is ahead on compartment preference (0.972 against 0.964 at 10 kb, 0.980 against 0.960 at 25 kb; 0/2 chromosomes for ours) and its APA ratio at 25 kb is 2.93. The ChIP-only model is behind C.Origami on Pearson and insulation at 10 and 25 kb, ahead on boundary F1 and loop recall at 10 kb (2/2), and behind on all four at 25 kb (0/2). Against Epiphany and Hi-cGAN every one of our three models is ahead on all four counted measures on every chromosome, except the ChIP-only model against Epiphany (compartment preference 2/3 chromosomes at 10 kb; loop recall and compartment preference 2/3 at 25 kb).

### AML windows and AML patients
- 10 kb object windows of 8 donors (13 064 windows, identical pixels; saved predictions, no AML in the training of our models): Pearson 0.761 (ours, refres), 0.746 (consensus12), 0.591 (Akita), 0.248 (Epiphany), 0.188 (C.Origami); insulation r at 50 kb 0.735, 0.738, 0.641, 0.342, 0.190; boundary F1 0.500, 0.500, 0.400, 0.000, 0.222; compartment r 0.821, 0.810, 0.653, 0.578, 0.575. Ours is ahead of Akita, C.Origami and Epiphany in 8/8 donors on each of the nine measures of the table (`aml_windows_paired.csv`). Against the measured consensus12 reference map our model is level on the structure measures (boundary F1 difference 0.000, ahead in 0/8 donors; insulation r at 50 kb +0.002, 5/8; Pearson +0.016, 7/8), so on these windows the TAD-level gain over the reference map is nil.
- Models trained with AML patients (folds 10 to 12, 10 donors, 413 regions, median over donors): insulation r 0.961 (10 kb) and 0.956 (25 kb) for aml_refin, 0.954 and 0.950 for base_refin@aml, 0.901 and 0.906 for base_refin@c12, 0.675 and 0.667 for aml_honest. The structure.py boundary F1 is 0 or 1 for most donors (250 kb square, 4 kb tolerance) and is not informative here. Competitors were not run on the 413 regions of these tables; the competitor values are the object-window values above.

### Caveats
(i) Akita is sequence only with one head per training cell type (GM12878 head) and is restricted to its cell types; the ChIP-only model gets cell-specific inputs and the with-consensus model also gets the Hi-C of other cells, so the comparison with Akita is a comparison of input sets as much as of models. (ii) The chromosome table uses hicFindTADs boundaries and 50 or 75 kb insulation squares; the Akita table uses insulation minima of the target and a 51 kb square; values are not comparable between the two tables. (iii) At 2 kb the donut of the structure.py loop enrichment (4 to 10 kb) is smaller than a loop, the compartment measures see 917 kb, and structure.py's boundary F1 (250 kb square) is 0 in the median for every predictor on the Akita regions; the table does not use it. (iv) Stripe r is a proxy, not a stripe caller. (v) Chromosome blocks have 2 or 3 chromosomes and AML windows 8 donors; counts are descriptive. (vi) Akita and the competitors are scored on pixels that all methods share, which restricts the AML windows to the central 917 kb (Akita) and to 99 bins from the diagonal (Epiphany) of the 1.92 Mb window.

### Files
`scripts/structure_scores_vs_akita.py`, `results/structure_vs_akita/{structure_scores_vs_akita.md, akita413_per_region.csv.gz, akita413_per_region_summary.csv, akita413_pooled.csv, akita413_paired.csv, chrho_summary_structure.csv, chrho_vs_competitor_structure.csv, aml_windows_*.csv, aml_eval_*.csv, run.log}`; the per-window AML table `aml_windows_per_window.csv.gz` (4.5 MB) is git-ignored and regenerated by the aml stage.


## 74. Independent validation of the unseen-cell-type result on four cell types untouched by the pipeline (FINAL; section 73 was the last number when this was written)

Question: do the 12-cell models (consensus12 as Hi-C reference input, marks of the new cell as ChIP input) keep their advantage over consensus12 alone and over the model without a reference on cell types that were in no training, validation, tuning, threshold choice or earlier table. Models, checkpoints and thresholds are unchanged; nothing is retrained or tuned on the new samples.

### Pre-registration (the file `results/independent_validation/PREREGISTRATION.md`, committed in 8782adc before any score of a new sample existed; reproduced here before the result tables were assembled)
Question. Does the unseen-cell-type result (12-cell models with consensus12 as Hi-C reference input, marks of the new cell as ChIP input) hold on cell types that no part of the pipeline has touched?

Samples (fixed before scoring; chosen by mark completeness, depth >= 300 M contacts where available, lineage not in the 17 samples, hg38). PC-3 (prostate adenocarcinoma, ENCODE intact Hi-C ENCSR038TZA), Panc1 (pancreatic carcinoma, ENCODE in situ Hi-C ENCSR440CTR, 200 M contacts, below the 300 M threshold but equal to A549 in training), natural killer cell (primary, ENCSR971CJS, ChIP of the same donor), Caco-2 (colorectal adenocarcinoma, ENCSR820DYU; colon lineage overlaps HCT116). HeLa-S3 (planned, 13 marks) was dropped before scoring: its only processed Hi-C is a 4DN mcool that the portal serves to logged-in users only. Inputs: the seven marks CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3 for every sample (the intersection of the marks all four have; RAD21, SMC3 and further marks are absent and given to the models as absent values).

Models, unchanged: all12refin_drop, all12refres_drop, all12honest_drop (no consensus), all12diffcells_refres_drop at 10 kb and 25 kb; chromosome-held-out variants (*_chrho) on chromosomes 2, 3, 11, 14, 17. Checkpoint-best, test-time reverse averaging, consensus12 (the 12 training cells; none of the new samples is in it) as Hi-C reference input. No retraining, no parameter or threshold change after seeing a score.

Unit of analysis. Windows: the 413 Akita test sequences, central 917 504 bp, whole 10 kb (25 kb) bins inside it, pixels >= 2 bins off the diagonal finite in the target and in every compared predictor (identical pixels), windows with >= 100 such pixels. Score: per-window Pearson of log O/E; the sample score is the median over windows.

Primary endpoint (10 kb, all seven marks, test windows). A: median Pearson of all12refin_drop_10kb minus that of consensus12 alone; B: all12refin_drop_10kb minus all12honest_drop_10kb. "Above" = difference of medians > 0 with the 95 percent paired bootstrap interval (2000 resamples of windows) excluding 0. Success = A above in >= 3 of the 4 samples (the 2/3 rule; 2 of 3 had the planned number of samples been 3) and B above in all 4.

Secondary (reported whether or not the primary succeeds). 25 kb; the other three models; keep sets CTCF + H3K27ac, CTCF, none; SCC, insulation r, boundary F1, loop-enrichment r, APA ratio, compartment r; whole chromosomes (all autosomes, 1 Mb tiles) and chromosomes 2, 3, 11, 14, 17 with the chrho models; versus the best single other cell's measured map (chosen per sample on the test windows, an oracle choice), versus Akita (GM12878 head, sequence only); split-half reproducibility ceiling of each target. C.Origami is omitted (no ATAC or DNase signal downloaded).

Negative outcomes (model not above consensus12 or not above the honest model in a sample) are reported as such.

Timeline note. The prediction and per-window scoring jobs of PC-3, Panc1 and Caco-2 ran in the interrupted session after the pre-registration file was written; their per-window score files were not read for this section before the text above was entered into the registry. Two prediction jobs of NK (25 kb refin, 10 kb whole-genome diff) had failed with CUDA out of memory on the shared local GPU and were rerun on the remote 4090 under 8 GB.

### Inventory and selection (`inventory.csv`, `samples.csv`)
Candidates were queried on the ENCODE portal (released GRCh38 fold-change bigWigs, unperturbed, untreated; in situ or intact Hi-C of the same biosample term). HeLa-S3 (13 marks) was dropped before scoring: its only processed Hi-C is a 4DN mcool served to logged-in users only. Selected, with all seven marks (CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3; RAD21 and SMC3 absent in all four):

| sample | Hi-C (ENCODE) | assay | contacts (M, cis M) | ChIP experiments (labs) |
|---|---|---|---|---|
| PC-3 | ENCSR038TZA / ENCFF322IBI | intact MNase | 662 (486) | Bernstein (Broad), 7 |
| Panc1 | ENCSR440CTR / ENCFF168OXT | in situ HindIII, 2 replicates | 212 (143) | Farnham, Snyder, Stamatoyannopoulos |
| NK cell (primary, 33 y) | ENCSR971CJS / ENCFF197OWW | intact MNase | 1404 (897) | Bernstein (Broad) |
| Caco-2 | ENCSR820DYU / ENCFF234MDO | intact MNase | 992 (803) | Stamatoyannopoulos, Bernstein, Snyder |

Download: Hi-C .hic files 26.2 + 4.7 + 48.8 + 31.6 = 111 GB, ChIP bigWigs 35 GB (28 files, 0.23 to 2.5 GB each); the whole `validation_new/` folder (including balanced and O/E cools 5 GB, predictions, scores) is 145 GB, below the 150 GB limit; nothing larger was fetched. Inputs were built as for the training cells: hicxchange hic2cool, cooler balance (ignore-diags 2, 500 iterations), `oe_proper.py`, bigWigs as `hg38_<cell>/<MARK>.bigwig`.

### Results (10 kb, 413 Akita test windows, identical pixels; `tables.md`, `paired_contrasts.csv`, `fig_independent_validation.png`)

| median per-window Pearson | PC-3 | Panc1 | NK | Caco-2 |
|---|---|---|---|---|
| split-half ceiling (Spearman-Brown) | 0.952 | 0.738 | 0.935 | 0.949 |
| consensus12 alone | 0.883 | 0.613 | 0.658 | 0.870 |
| best other measured cell (oracle) | 0.830 | 0.577 | 0.694 | 0.835 |
| Akita (GM12878 head) | 0.673 | 0.434 | 0.420 | 0.688 |
| honest (no consensus), 7 marks | 0.772 | 0.526 | 0.676 | 0.737 |
| refin, 7 marks | 0.895 | 0.615 | 0.766 | 0.894 |
| refres, 7 marks | 0.896 | 0.617 | 0.782 | 0.897 |
| diff channels, 7 marks | 0.902 | 0.623 | 0.802 | 0.898 |
| A: refin - consensus12 [95% CI] | +0.012 [0.007, 0.021] | +0.002 [-0.009, 0.012] | +0.108 [0.090, 0.145] | +0.024 [0.016, 0.033] |
| B: refin - honest [95% CI] | +0.123 [0.111, 0.140] | +0.088 [0.064, 0.105] | +0.090 [0.068, 0.123] | +0.157 [0.142, 0.173] |

Primary endpoint: A above in 3 of 4 samples (PC-3, NK, Caco-2; Panc1 not: interval includes 0), B above in 4 of 4. By the pre-registered rule (A >= 3 of 4, B in all) the endpoint is met (`primary_endpoint_verdict.json`).

### Secondary
- 25 kb: same pattern (A: +0.010, +0.003 [-0.004, 0.014], +0.115, +0.019; B +0.125, +0.132, +0.089, +0.175).
- Whole genome, 1 Mb tiles chr1-22 (about 2900 tiles per sample, 10 kb): refin minus consensus12 +0.016 (PC-3), +0.002 (Panc1, interval [-0.002, 0.005]), +0.126 (NK), +0.020 (Caco-2); refin minus honest +0.131, +0.094, +0.119, +0.175. Median Pearson refin 0.893, 0.591, 0.763, 0.888; consensus12 0.878, 0.590, 0.637, 0.867.
- Chromosome-held-out models on chr2, 3, 11, 14, 17 (tiles): refin_chrho minus consensus12 +0.008, +0.010, +0.135, +0.017 (all intervals above 0); refres_chrho +0.012, +0.007 (interval includes 0), +0.100, +0.025. On the Akita test windows of these chromosomes (86 windows per sample) refin_chrho minus consensus12 is +0.004 [-0.005, 0.013] (PC-3), +0.017 [-0.0001, 0.033] (Panc1), +0.111 [0.075, 0.165] (NK), +0.019 [0.0005, 0.037] (Caco-2): above 0 in NK and Caco-2 only.
- Marks subsets (refin): CTCF + H3K27ac is within 0.006 of all seven marks in PC-3 and Caco-2 and 0.015 above them in NK (0.781 vs 0.766) and below them in Panc1 (-0.017); CTCF only and no marks equal consensus12 within 0.01 in PC-3, Panc1, Caco-2 and fall to 0.677 and 0.644 in NK (below consensus12 0.658 for no marks). Consensus12 carries the Hi-C level; the marks add structure only where consensus12 is poor.
- Structure measures (10 kb, test windows; medians): insulation r refin 0.974, 0.929, 0.890, 0.974 against consensus12 0.971, 0.936, 0.773, 0.960 (Panc1 lower); loop enrichment r refin 0.747, 0.206, 0.546, 0.737 against 0.725, 0.204, 0.469, 0.716; compartment r 0.933, 0.769, 0.819, 0.926 against 0.922, 0.762, 0.737, 0.906. Akita is below all trained models in every sample and measure (Pearson 0.42 to 0.69). Boundary F1 of single windows is 0 or 1 (median uninformative); the mean is in `tables.md` and Panel C of the figure.
- C.Origami omitted (no ATAC/DNase downloaded).

### Verdict and caveats
The pre-registered endpoint is met, but unevenly. The gain over consensus12 is large in NK only (+0.11, from 0.658 to 0.766, ceiling 0.935), the one sample whose Hi-C differs from the 12-cell consensus. In the three cancer lines consensus12 alone already reaches 0.87 to 0.88 (PC-3, Caco-2) of a 0.95 ceiling and the models add +0.01 to +0.02; in Panc1 (212 M contacts, ceiling 0.738, in situ HindIII) there is no gain over consensus12 (+0.002, not above) and the model is level with it. All models are far above the model without consensus (+0.09 to +0.16) and above Akita (+0.18 to +0.35), and above the best single measured other cell (+0.04 to +0.07, oracle choice). Caveats: (i) four samples; the 2/3 rule at n = 4 is 3; the Panc1 failure is a negative result for A; (ii) lineage overlap with training cells (Caco-2 with HCT116; PC-3 and Panc1 are epithelial like MCF-7, A549), which makes consensus12 strong; NK shares its hematopoietic lineage with GM12878, K562 and CD14+ but is the only sample whose Hi-C is poorly matched by consensus12; (iii) Hi-C assays differ (intact MNase vs in situ), as in training; (iv) the Panc1 and NK ChIP experiments come from several labs and donors; (v) PC-3, Panc1 and Caco-2 scores were computed in the interrupted session after the pre-registration file and before this section; the NK scores were computed in this session; (vi) two NK prediction jobs failed on the shared GPU and were rerun on CPU (identical code, `--device cpu`).

### Files
`scripts/independent_validation_fetch.py`, `scripts/independent_validation_predict.py`, `scripts/independent_validation.py`, `results/independent_validation/{PREREGISTRATION.md, inventory.csv, samples.csv, ceilings.csv, summary_*.csv, paired_contrasts.csv, primary_endpoint_verdict.json, tables.md, fig_independent_validation.png}`; per-window scores, predictions and inputs in `/mnt/processed/wolffjoa/hic-multiome/validation_new/`.


## 75. Transfer of the human-trained models to mouse (mm10), unchanged models, three mouse cell types (FINAL; section 74 was the last number when this was written)

### Pre-registration (the file `results/mouse_transfer/PREREGISTRATION.md`, committed in 7e8e290 before any score of a mouse window existed; reproduced here)
Question. Do the human-trained models, applied unchanged to mouse chromatin marks, predict mouse Hi-C? A positive
answer says the learned mapping from marks (and local sequence composition) to 3D structure is shared between
human and mouse; a negative answer says that part of what the models learned is human-specific.

Design. No retraining, no tuning, no threshold or parameter choice after seeing a score. Models: the final
checkpoints (checkpoint-best) of all12honest_drop, all12refin_drop, all12refres_drop and
all12diffcells_refres_drop at 10 kb and 25 kb, trained on 12 human cell lines only.

Genome build. mm10 (GRCm38) throughout. All ENCODE Hi-C and ChIP files are mm10 natively and the Akita mouse
heads are mm10; every input of one sample shares the build, nothing is lifted to mm39 (a chain lift would
re-bin contacts and move bins; staying in mm10 loses nothing). The model is build-agnostic (bins, not
coordinates).

Samples (fixed before scoring; three distinct lineages; Hi-C and ChIP of the same cell type or tissue):
- mesc: mouse ES cell E14. Hi-C ENCSR051JZL (ENCFF092WXU, in situ, MAPQ >= 30). ChIP: ES-E14 (H3K27ac, H3K4me1,
  H3K4me3, H3K36me3, H3K9me3) and E14TG2a.4 (CTCF, H3K27me3; the E14 subclone). Replicate experiment of the
  same line: ENCSR819XFA (ENCFF079MKO), used for a replicate ceiling.
- ch12: CH12.LX B-cell lymphoma. Hi-C ENCSR005UON (ENCFF076GYR, in situ, 3 replicates pooled). ChIP: CH12.LX,
  seven marks, plus RAD21 and SMC3.
- heart: adult heart (2 months). Hi-C ENCSR997TTD (ENCFF651SGY, intact, B6CASTF1/J). ChIP: heart, adult 2 months
  (CTCF, H3K27ac, H3K4me3 of B6CASTF1/J female; H3K27me3, H3K36me3, H3K4me1 of B6NCrl male; H3K9me3 of the
  same B6NCrl series).
- Not used: left cerebral cortex (Hi-C ENCSR639NRL, 52.8 GB, disk budget; only CTCF, H3K27ac, H3K4me3 exist for
  the same tissue and strain).
- Control variant: mesc_b4 = the mESC Hi-C with the ES-Bruce4 ChIP of all seven marks (another line, one lab).

Inputs (as for the human cells, scripts/independent_validation_fetch.py conventions): MAPQ30 .hic -> hicxchange
hic2cool -> cooler balance (ignore-diags 2, 500 iterations) -> scripts/oe_proper.py (10 and 25 kb); ChIP as the
released mm10 fold-change-over-control bigWig, through the models' own track normaliser (log1p, global z-score
of training) and the models' own matrix normaliser (that of the first training sample, as for unseen human
cells). What the model reads: seven ChIP tracks (CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3,
H3K9me3), GC and CpG fraction per bin and the four motif planes (CTCF MA0139.2, YY1 MA0095.3, ZNF143 MA0088.2,
MAZ MA1522.1) scanned with the human JASPAR PWMs on mm10 sequence (the motifs are conserved). RAD21, SMC3,
H3K4me2, H3K79me2, H3K9ac, H4K20me1 are given as absent (CH12 has RAD21 and SMC3: used only in the extra set
"avail"). The Hi-C reference of refin / refres and the mark reference of the differential-channel model are
species-matched: the leave-one-out mean of the other mouse samples (O/E maps of oe_proper.py; marks of the
other samples), so three samples give a two-sample consensus; the human consensus12 is not used (syntenic
regions differ).

Windows. The 6 038 tiles of Akita v2 mm10 (sequences.bed, 1 310 720 bp, 8 folds); scored region the central
1 048 576 bp, whole 10 kb (25 kb) bins inside it; a tile is used when the model input span (192 bins at 10 kb,
80 bins at 25 kb, centred) lies inside the chromosome and contains no assembly gap (N run >= 1 kb).
Autosomes chr1 to chr19 are the headline set; chrX is reported separately (the differential-channel model has
no chrX rows: statistics of autosomes only). Pixels: >= 2 bins off the diagonal, finite in the target and in
every compared predictor (identical pixels), outside bins overlapping the ENCODE mm10 blacklist v2; windows
with >= 100 such pixels. Score: per-window Pearson of log O/E; sample score = median over windows.

Predictors. (1) honest: all12honest_drop_{10,25}kb, seven marks, no consensus. (2) refin, refres, diff:
all12refin_drop, all12refres_drop, all12diffcells_refres_drop with the mouse leave-one-out reference. (3)
consensus alone: the mouse leave-one-out map. (4) distance-decay-only: the chromosome-wide mean O/E per
distance, which is zero by construction of oe_proper.py; a constant has no covariance, its Pearson is set to 0.
Also reported, not used for the endpoint: the in-sample per-window per-diagonal mean of the target (upper bound
of any function of distance). (5) Akita v2 mouse heads (mm10, gs://basenji_hic/3-2021, sequence only): for
every tile the model of the fold that holds the tile out as test; heads Hsieh mESC Micro-C and Bonev ES for
mesc, Bonev cortical neuron / NPC as off-lineage heads; only mesc has a matching head; evaluated on a random
1 500-window subset (seed 0) because of CPU time. Controls: marks shuffled (every ChIP track of a chromosome
rolled by one common offset of 25 to 75 percent of its length; sequence, GC, motif planes untouched), marks of
a different mouse sample (mesc <- heart, ch12 <- mesc, heart <- ch12), mark subsets (CTCF + H3K27ac, CTCF,
none), the untrained network (random initialisation, same normalisers; human-pretrained vs nothing), mesc_b4.

Ceilings. Split-half: raw counts binomially split 1:1, both halves balanced and transformed as the registry
replicate ceiling, per-window Pearson, Spearman-Brown to full depth, median over a random 1 000-window subset
(seed 0). Replicate experiment (mesc only): O/E of ENCSR819XFA against ENCSR051JZL on the same pixels
(lower depth, not corrected).

PRIMARY ENDPOINT. Median per-window Pearson, 10 kb, autosome windows, of all12honest_drop_10kb with the seven
marks of the sample (human model, mouse marks, no consensus) is above (i) the distance-decay baseline and (ii)
the shuffled-marks control in every one of the three mouse samples; "above" = difference of medians > 0 with
the 95 percent paired bootstrap interval (2000 resamples of windows) excluding 0. Its fraction of the split-half
ceiling (median / median) is reported per sample. The endpoint is met only if both comparisons hold in all
three samples; any sample that fails is reported as such and the endpoint is then not met.

Secondary (reported whether or not the primary succeeds). (S1) refin, refres, diff with the mouse leave-one-out
reference minus the consensus alone (paired contrast, 10 kb; 25 kb in the tables), and refin minus honest.
(S2) honest and refin versus Akita v2 mouse heads on mesc. (S3) mark subsets and CH12 "avail" (with cohesin).
(S4) structure classes, median per window on identical pixels: insulation r, boundary F1 (mean), loop enrichment r
and APA ratio, compartment r (window-level correlation-matrix correlation), stripe-profile r (per-bin mean
O/E over 3 to 30 bins downstream and upstream, exploratory). (S5) 25 kb. (S6) chrX.

Interpretation rules, fixed now. "Conserved biology" is supported by the endpoint only in the sense that a
human-trained function from marks to structure carries information in mouse above the controls; the size of
the effect is read from the fraction of the ceiling, not from the sign alone. If honest does not exceed the
shuffled marks in a sample, that sample contributes no support. A structure class transfers when its
median measure of honest exceeds that of the shuffled-marks control with the interval above 0 in all three
samples; classes with weaker or inconsistent results are reported as not transferring. Few samples, assay
differences (in situ vs intact), strain mixing of the heart ChIP and the lack of species-specific retraining are
stated as limits, not corrected.


### Results (FINAL; scores in results/mouse_transfer/, tables.md, fig_mouse_transfer.png)
Run exactly as pre-registered: no model, parameter or threshold changed. 5 629 autosome windows at 10 kb in every
sample (5 579 at 25 kb); chrX (S6) was not run. Hi-C: mesc ENCFF092WXU (ENCSR051JZL, 14.4 GB .hic, 1 087 M cis
contacts at MAPQ >= 30), ch12 ENCFF076GYR (ENCSR005UON, 8.7 GB, 533 M), heart ENCFF651SGY (ENCSR997TTD, 52.2 GB,
749 M), mesc replicate ENCFF079MKO (ENCSR819XFA, 7.4 GB, 415 M); mm10 throughout; ChIP accessions in
validation_mouse/inventory.csv. Mouse leave-one-out consensus = mean of the two other samples (3 samples).

Primary endpoint (honest, 10 kb, median per-window Pearson; interval = 95 percent paired bootstrap): MET.
| sample | honest | minus distance decay | minus shuffled marks | split-half ceiling | fraction of ceiling |
|---|---|---|---|---|---|
| mesc | 0.733 | +0.733 [+0.727, +0.738] | +0.432 [+0.421, +0.444] | 0.944 | 0.78 |
| ch12 | 0.635 | +0.635 [+0.624, +0.643] | +0.473 [+0.459, +0.485] | 0.934 | 0.68 |
| heart | 0.722 | +0.722 [+0.715, +0.731] | +0.571 [+0.559, +0.584] | 0.963 | 0.75 |
Windows with honest above shuffled marks: 0.93, 0.91, 0.94. The same holds at 25 kb (0.759, 0.748, 0.775; above
shuffled marks by +0.42 to +0.57).

Medians, 10 kb (mesc / ch12 / heart): distance decay 0 / 0 / 0; in-sample per-diagonal mean 0.425 / 0.376 / 0.380;
untrained network 0.031 / 0.007 / 0.026; shuffled marks 0.300 / 0.162 / 0.150; no marks (sequence only) 0.536 /
0.386 / 0.452; CTCF 0.660 / 0.500 / 0.620; CTCF + H3K27ac 0.683 / 0.542 / 0.674; marks of another sample 0.722 /
0.533 / 0.608; honest 0.733 / 0.635 / 0.722; mouse consensus alone 0.766 / 0.746 / 0.765; refin 0.833 / 0.803 /
0.840; refres 0.829 / 0.808 / 0.835; diff 0.834 / 0.812 / 0.836.
Secondary. S1: refin, refres, diff minus consensus alone are +0.057 to +0.076 (all intervals above 0, 10 and 25 kb);
refin minus honest +0.100 / +0.168 / +0.119. Honest minus consensus alone is -0.033 / -0.111 / -0.043 (honest does
not reach the consensus). S2 (mesc, 1 500-window subset, 10 kb): honest minus Akita v2 Bonev ES head +0.014
[-0.001, +0.028], Hsieh mESC +0.023 [+0.008, +0.038], Bonev CN +0.067, NPC +0.036; refin minus Bonev ES +0.106.
S3: honest minus CTCF + H3K27ac +0.050 / +0.094 / +0.047; CH12 with RAD21 and SMC3 +0.053 (honest), +0.014
(refin); mesc with ES-Bruce4 marks: difference 0.000 [-0.005, +0.005]. Replicate experiment of mesc 0.921
(honest -0.188 below it). Marks of another sample: honest minus it +0.011 [+0.006, +0.017] (mesc), +0.103 (ch12),
+0.114 (heart); at 25 kb mesc +0.006 [-0.001, +0.013].
S4 structure classes (honest minus shuffled marks; every interval above 0 in all three samples, 10 kb): insulation
r +0.46 to +0.60; boundary F1 +0.19 to +0.20; loop enrichment r +0.32 to +0.46; APA ratio +0.13 to +0.32;
compartment r +0.18 to +0.21; stripe profile r +0.43 to +0.56. By the pre-registered rule all classes transfer.
Honest values (mesc / ch12 / heart): insulation 0.79 / 0.75 / 0.81, boundary F1 0.29 / 0.27 / 0.26, loop
enrichment 0.45 / 0.38 / 0.53, APA ratio 0.30 / 0.15 / 0.35, compartment 0.82 / 0.75 / 0.80. Against the marks of
another sample the margin is near zero or negative for several classes in mesc (insulation -0.003, boundary
-0.004, stripe -0.025, APA ratio -0.186), positive in ch12 and heart.

Caveats. The shuffled-marks control is below the sequence-only model (0.15 to 0.30 against 0.39 to 0.54): a
mismatched mark track is worse than none, so the margin over it overstates the information carried by marks; the
margin over no marks (+0.20 to +0.27) and over the marks of another sample (+0.01 to +0.11) is the conservative
reading. In mesc the marks of heart predict almost as well as its own. The mouse consensus alone beats the human
model with mouse marks in every sample, so most of the mouse structure is shared across mouse tissues and the
unchanged model recovers less of it than a map of another mouse tissue. Three samples; in situ vs intact Hi-C;
heart ChIP of mixed strains; the human PWMs on mouse sequence; no species-specific retraining; chrX not run;
Akita on a 1 500-window subset; the primary distance-decay comparator is a constant (Pearson 0 by construction).
Interpretation under the pre-registered rule: the human-trained mapping from marks and local sequence to structure
carries information in mouse above every control, at 68 to 78 percent of the replicate ceiling for the honest
model, and each structure class transfers; the evidence does not separate conserved marks-to-structure biology
from conserved sequence and chromatin-state composition, and cell-type-specific marks contribute least in mESC.

## 76. Window prior: the consensus as a function of window content instead of genomic position (FINAL; section 75 was the last number when this was written)

### Pre-registration (the file `results/window_prior/PREREGISTRATION.md`, committed before any score of a prior-based model existed; reproduced here)
Question. The consensus input of the reference models is positional: the mean of the other cells' Hi-C at the same genomic coordinates. It cannot be built for another organism, whose syntenic regions differ. Can the consensus be replaced by a function learned from windows across the genome that maps the content of a window (chromatin marks, no coordinates) to the typical Hi-C of such a window, so that the main model learns from it and the same function transfers to an organism without reference Hi-C?

Window prior network P (hicmultiome/windowprior.py, scripts/window_prior.py). Same U-Net family as the main models, 24 base channels, depth 4, conditioning "full" (mean, product and absolute difference of the mark rows per pixel plus the |i-j|/W plane), no reference or prior plane, no distance bias, no sequence encoder. Input: per mark the reference mark of the window, built as the dref channels of registry 66 (mean over the reference set, without the sample and its cell identity, of the per-sample quantile-anchored track; leave-one-out set of the 11 other cells for a training cell, the 12 training cells for a held-out cell or a new sample; 0 for a mark the sample or the whole reference lacks), 13 marks. Strictly position-free: no coordinate, chromosome id, positional embedding or per-locus table enters; the CTCF motif planes (fwd / rev of the four motifs) are an ablation switch, off in the primary models. Target: the leave-one-out consensus window in log O/E (the map the positional consensus input holds), divided by 2 and clipped to +-1. Loss L1 + MSE, AdamW 1e-3, batch 16, mark dropout (uniform count, probability 0.5), genomic reversal augmentation, honest validation windows (100 per cell, no training window overlapping a validation position in any cell), early stopping on the validation loss (patience 8). Training windows: the train split of the cache of all12diffcells_refres_drop (Akita training regions), all 12 cells.

Cross-fitting. A chromosome is predicted by a P that was trained without it (P_odd on odd, P_even on even autosomes, half the windows each), so that no training window receives a prior fitted on its own region and no validation or test window sees a P that was fitted on its chromosome. P_full (all training chromosomes) serves the controls and organisms for which chromosome numbers carry no meaning. Chromosome-held-out variant (chr2, 3, 11, 14, 17 excluded from every P) is implemented (--chrho) and run only for the headline model.

Models (10 kb primary, 25 kb afterwards; configuration of the existing run with another reference or prior map; the output of P is written as a cooler and enters through samples[].reference_cool or prior_cool, no change of training code).
- all12wp_ref_drop: all12diffcells_refres_drop with P(reference marks) in place of the positional consensus; the residual base of the output is the same map as the input plane.
- all12wp_self_drop: all12refres_drop (sample marks, no difference channels) with P(the sample's own marks; P trained on own marks, same architecture and target) in place of the positional consensus; needs no other cell.
- all12wp_both_drop: all12diffcells_refres_drop with the positional consensus kept (reference plane and residual base) and P(reference marks) as an additional prior plane.
- References, existing results (seeds in reference_summary.csv): all12honest_drop (no consensus), all12refres_drop, all12diffcells_refres_drop (positional consensus).

Evaluation. Standard evaluation of scripts/multicell_expanded.py (413 Akita regions, Pearson per region, SCC, insulation and the other measures, 12 training cells and 5 held-out cells K562, HMEC, HUVEC, HMEC deep, HUVEC deep; reference_summary.csv); the four independent samples PC-3, Panc1, NK, Caco-2 (scripts/independent_validation.py windows and pixels, reference pool of the marks = the 12 cell lines, seven marks); the mouse samples of registry 75 when their inputs exist (the human-trained P unchanged; reference marks = the other mouse samples for wp_ref, the sample itself for wp_self). Per sample: no consensus (honest), positional consensus, wp_ref, wp_self, wp_both, the consensus alone, the split-half ceiling.

Primary endpoint (10 kb, all seven or thirteen marks as available, 413 test windows, paired by window and cell, 2000 bootstrap resamples of windows within cell, seed 0, 95 percent interval of the difference of medians). For X in wp_ref and wp_self: (E1) median Pearson of X minus that of all12honest_drop on the five held-out cells, pooled; (E2) the same on the four new samples, pooled. "Above" = lower bound of the interval > 0. The primary endpoint is met only if X is above in E1 and E2 for both wp_ref and wp_self; each of the four comparisons is reported on its own. Per-cell differences are reported with the same interval.

Secondary (reported whether or not the primary is met). (S1) Fraction of the gap closed: (X - honest) / (positional consensus model - honest), per cell and pooled, with the bootstrap interval of the ratio of the pooled median differences; positional consensus model = all12refres_drop for wp_self, all12diffcells_refres_drop for wp_ref and wp_both. (S2) Additive value: wp_both minus all12diffcells_refres_drop, paired. (S3) wp_ref and wp_self against the positional consensus models (expected below). (S4) P alone as predictor on the 413 regions: Pearson with the true leave-one-out consensus and with the cell's own Hi-C, for P(reference marks) and P(own marks); the consensus alone for comparison. (S5) The motif-plane ablation of P (P alone, and all12wp_refm_drop when GPU time allows). (S6) 25 kb. (S7) SCC, insulation r, boundary F1, loop enrichment, compartment r on identical pixels. (S8) chrho variant.

Controls. (C1) Position-free control: the mark signal of a chromosome rolled by a random whole number of windows (grid aligned) and the chromosome labels permuted; P is run through the production path; maximum absolute difference of the log O/E maps after undoing the roll (a function of the content gives 0 up to float rounding), reported with the same shift applied to the positional consensus as positive control. (C2) P-only baseline (S4). Leakage rules unchanged: no target-cell Hi-C in any input (the reference of a training cell excludes the cell and its identity); P is trained on Hi-C of the 12 training cells on training windows only and applied to a chromosome only by a P trained without that chromosome; validation windows never overlap training windows.

Expectation, stated before the score exists. The prior is a function of the marks of a window and the marks of the sample are already given to the main model, so the part of the positional consensus that is not a function of the marks of the window (the locus-specific structure shared by the cells) is out of reach of P. We expect wp_ref and wp_self to lie between all12honest_drop and the positional consensus models and to close only part of the gap; wp_both to add little or nothing to all12diffcells_refres_drop. A result of wp_self not above all12honest_drop is a possible outcome and is reported as such.

Fixed in advance: P architecture and hyperparameters as above (no tuning on a score; a change after seeing a score is a new run that is reported next to this one), one seed (0) for every model, no model selection on the held-out cells or new samples, 10 kb checkpoint-best of each main run chosen by its own honest validation.

Limits (to be stated with the results): one seed; reference marks pool of 12 cell lines, assay differences between the cells (in situ, intact, Micro-C); the new samples carry seven marks; the models are human-trained and the mouse samples test transfer only.

### Results (FINAL, 2026-10-14; the pre-registration above is unchanged; all numbers from results/window_prior/, scripts/window_prior_summary.py, window_prior_indep.py, window_prior_mouse.py)

Training and evaluation: P (ref and own, odd / even / full folds) and the three main models trained and evaluated at 10 kb and 25 kb, one seed (0), no run repeated or tuned. All five held-out cells and four independent samples scored on identical windows (413 Akita test regions).

Median Pearson per region, median of the per-cell medians (10 kb; tables_10kb.md, cells_10kb.csv)

| group | no consensus | positional (refres) | positional (diff) | wp_ref | wp_self | wp_both | consensus alone | ceiling |
|---|---|---|---|---|---|---|---|---|
| held-out cells (5) | 0.761 | 0.897 | 0.905 | 0.778 | 0.754 | 0.901 | 0.859 | 0.906 |
| independent samples (4) | 0.706 | 0.839 | 0.850 | 0.719 | 0.719 | 0.855 | 0.764 | 0.942 |

25 kb: held-out 0.775 / 0.907 / 0.924 / 0.798 / 0.757 / 0.922 / 0.898; independent 0.747 / 0.897 / 0.898 / 0.779 / 0.777 / 0.898 / 0.877. Ceiling = Spearman-Brown corrected split-half replicate (replicate_ceiling and independent_validation/ceilings.csv). Positional (refres) is the consensus model that wp_self is compared with, positional (diff) the one for wp_ref and wp_both.

Primary endpoint (10 kb, paired bootstrap, 2000 resamples, difference of pooled medians vs all12honest_drop): E1 held-out wp_ref +0.0229 [0.0141, 0.0317] above; E1 wp_self -0.0054 [-0.0121, 0.0023] not above; E2 independent wp_ref +0.0111 [0.0020, 0.0213] above; E2 wp_self +0.0001 [-0.0100, 0.0087] not above. The primary endpoint is NOT met (wp_self fails both). 25 kb: wp_ref +0.0279 [0.0209, 0.0395] and +0.0298 [0.0189, 0.0374]; wp_self -0.0197 [-0.0270, -0.0113] and +0.0092 [-0.0008, 0.0166].

Fraction of the gap closed (S1, 10 kb pooled): wp_ref 0.156 [0.100, 0.207] held-out, 0.074 [0.015, 0.138] independent; wp_self -0.039 [-0.093, 0.016] and 0.000 [-0.071, 0.059]. 25 kb: wp_ref 0.202 [0.156, 0.269] and 0.203 [0.135, 0.250]; wp_self -0.152 and 0.064. Of the 12 training cells at 10 kb wp_ref closes 0.036 and wp_self 0.020.

Additive value (S2): wp_both minus all12diffcells_refres_drop is -0.0081 [-0.0105, -0.0054] on the held-out cells and +0.0046 [-0.0024, 0.0080] on the independent samples at 10 kb (25 kb: -0.0005 and -0.0009, intervals include 0). No gain from adding P to the positional consensus.

P alone (S4, 10 kb, held-out cells, median Pearson): P(reference marks) 0.723 with the cell's own Hi-C and 0.850 with the leave-one-out consensus; P(own marks) 0.732 and 0.787; the consensus alone 0.859 with the own Hi-C. Medians over the 12 training cells: 0.757 / 0.772 / consensus 0.887.

Controls. C1 position-free: maximum absolute difference of the log O/E maps after undoing a random whole-window roll and permuting chromosome labels, over 17 cells and 6 chromosome pairs each: P at most 1.4e-4 (float16 rounding), positional consensus 1.8 to 2.8. The prior is a function of window content only. C2 P-only baseline: as above; P alone is 0.14 below the consensus alone on the own Hi-C.

Mouse (10 kb, human-trained models and P unchanged, 5629 windows per sample, mm10; mouse_contrasts_10kb.csv; windows overlap, intervals treat windows as independent and are optimistic): median Pearson no consensus / wp_ref / wp_self / wp_both / positional diff / consensus alone: mESC 0.733 / 0.736 / 0.747 / 0.827 / 0.834 / 0.766; CH12 0.635 / 0.637 / 0.675 / 0.811 / 0.812 / 0.746; heart 0.722 / 0.714 / 0.719 / 0.818 / 0.836 / 0.765. wp_self minus honest: +0.014 [0.010, 0.018], +0.040 [0.035, 0.048], -0.002 [-0.008, 0.002]; wp_ref minus honest: +0.004 [-0.002, 0.008], +0.002 [-0.005, 0.011], -0.008 [-0.014, -0.001]. The transfer of the human-trained P closes under 10 percent of the gap to the positional model.

Verdict. The window prior does not replace the positional consensus. wp_ref (P of the reference marks) closes 16 percent (held-out) and 7 percent (independent) of the gap at 10 kb and 20 percent at 25 kb, a small but reproducible gain over no consensus; wp_self does not exceed the model without consensus; wp_both adds nothing to the positional model. The locus-specific structure shared by cells is not recoverable from the marks of the window, as stated in the expectation. On mouse the prior gives small gains only for CH12 with own marks.

Caveats. One seed for every model; the held-out cells HMEC / HMEC deep and HUVEC / HUVEC deep share inputs; the independent samples carry seven marks (P and the main models see missing marks as zero); 25 kb and mouse intervals are windows within sample, not independent samples; mouse positional models use the mouse pool consensus at identical mm10 coordinates; the ceiling is the replicate ceiling, not a bound for a function of marks. Motif-plane ablation (S5) and the chromosome-held-out variant (S8) were not run (GPU time); wp_refm is not reported.


## 77. C.Origami and Epiphany retrained on our cell lines and our chromosome split (FINAL; section 76 was the last number when this was written)

Question: does the gap between our chromosome-held-out models and C.Origami / Epiphany remain when the two competitors are trained on the same cells, the same chromosome split and the same O/E targets as ours, each with its own architecture and its own native input set?

### Design
- Split (identical for both): training chromosomes 1, 4-10, 12, 13, 15, 16, 19, 20, 22; validation chromosomes 18 and 21; test chromosomes 2, 3, 11, 14, 17 (the chromosomes held out of every chrho model). X is not used.
- Cells: 11 training cells whose native inputs exist (GM12878, IMR-90, HCT116, HFF, H1, HepG2, A549, NHEK, MCF-7, CD14+ monocytes, PC-9); held-out cell types K562, HMEC, HUVEC (inputs exist; never used in any training of the retrained models or of our chrho models). GM23248 is excluded (no CTCF track).
- Inputs. C.Origami: hg38 sequence + CTCF + chromatin accessibility. ATAC bigWigs exist for 7 of our 15 cells only, ENCODE DNase-seq signal for all, so DNase-seq was downloaded for the 14 cells (14 files, 8.9 GB, size and md5 checked, `results/competitor_retrain/dnase_picks.tsv`, `scripts/competitor_retrain/fetch_dnase.py`) and used as the accessibility channel; CTCF and DNase of every cell are mapped onto the distributions of the released IMR-90 tracks by a quantile map (the "q" variant of section 72) so that the released and the retrained model see the same input scale. Epiphany: DNase, CTCF, H3K27ac, H3K27me3, H3K4me3 at 100 bp.
- Targets: our O/E maps (oe_proper.py) at 10 kb; Epiphany is trained on the 10 kb pixels (990 kb reach), C.Origami on the exact area average of the 10 kb pixels over its 8192 bp bins (2 Mb window, 256 x 256). Scoring by the identical-pixel pipeline of sections 58 and 67 (hic_measures.py `--chrho-competitor-retrain`; blocks per cell, grid, test chromosome and family; C.Origami block 2 Mb band, Epiphany block 990 kb band; 280 blocks, 9 to 10 sources each).
- Architectures and recipes unchanged (copies in `scripts/competitor_retrain/vendor`, the working trees of C.Origami and Epiphany untouched). C.Origami: ConvTransModel, Adam 2e-4, clipping 1, MSE masked to measured pixels, batch 8, 24 000 steps (about 4.9 passes over the 39 369 training intervals; the original 80 epochs did not fit the budget), best checkpoint by validation MSE (step 22 000). Epiphany: generator + discriminator, lam 0.95, 69 000 steps (the 55 epochs of the original), best checkpoint by validation MSE.
- GPU time (RTX 4090, shared with other jobs at times): C.Origami pilot 0.45 h (1 500 steps, 10 cells, validation Pearson 0.47), full 7.2 h (0.9 h interrupted first segment plus 5.9 h resumed from step 2 000); Epiphany pilot 0.53 h (3 672 steps, validation Pearson 0.64), full 5.5 h resumed from the pilot checkpoint (6.0 h in total); inference of the released and the retrained models on 14 cells x 5 chromosomes 3.0 h. Total 16.2 GPU hours; each full training is below the 12 h limit. Validation Pearson at the end: C.Origami 0.58, Epiphany 0.73 (`history_*.json`).

### Results (mean over cells of the mean over chr2, 3, 11, 14, 17; Pearson on O/E; full measure set in `results/competitor_retrain/table.md`)
| family, grid, cell group | ours (refin) | ours (honest, no Hi-C input) | competitor retrained | competitor released weights on the same inputs |
|---|---|---|---|---|
| C.Origami, 25 kb, 11 training cells | 0.915 | 0.785 | 0.724 | 0.449 |
| C.Origami, 25 kb, K562/HMEC/HUVEC | 0.893 | 0.752 | 0.730 | 0.393 |
| C.Origami, 10 kb, 11 training cells | 0.895 | 0.774 | 0.703 | 0.422 |
| C.Origami, 10 kb, K562/HMEC/HUVEC | 0.853 | 0.723 | 0.690 | 0.369 |
| Epiphany, 25 kb, 11 training cells | 0.918 | 0.775 | 0.727 | 0.378 |
| Epiphany, 25 kb, K562/HMEC/HUVEC | 0.893 | 0.730 | 0.697 | 0.454 |
| Epiphany, 10 kb, 11 training cells | 0.905 | 0.774 | 0.715 | 0.373 |
| Epiphany, 10 kb, K562/HMEC/HUVEC | 0.861 | 0.725 | 0.667 | 0.447 |

Retraining raises the Pearson of C.Origami from 0.45 to 0.72 and of Epiphany from 0.38 to 0.73 on the training cells at 25 kb, and from 0.39 to 0.73 and from 0.45 to 0.70 on the held-out cell types; the retrained models stay below our models. Per cell, all12refin_drop_chrho is above the retrained competitor in 11 of 11 training cells and 3 of 3 held-out cell types for both families and both grids (mean Pearson difference 0.16 to 0.20). all12honest_drop_chrho (marks only, no Hi-C reference input) is above in 11 of 11 training cells except Epiphany at 25 kb (10 of 11; CD14+ monocytes), and in 3 of 3 held-out cell types except C.Origami at 25 kb (2 of 3; K562 equal at 0.752); its mean difference is 0.02 to 0.07 (\`vs_retrained.csv\`). Boundary recall, loop recall and compartment r of the retrained models are below those of our models at both grids (10 kb, training cells: loop recall 0.136 and 0.044 against 0.384 for refin; compartment r 0.81 and 0.84 against 0.95). In the same table the retrained models are below the two Hi-C-based baselines of section 58 in Pearson, consensus12 (0.84 at 25 kb, training cells) and the best other cell's map (0.82), and in all three held-out cell types they are below both as well.

### Caveats
(i) The retrained models see one cell type per sample and no Hi-C of other cells, as published; our models receive the consensus12 or the cell-type Hi-C reference as input (refin) or only marks (honest); the honest rows are the fair architecture-to-architecture comparison. (ii) C.Origami was trained for about 5 passes over the intervals, not 80 epochs (budget); its validation Pearson was still rising slowly (0.57 at 18 000, 0.58 at 22 000 steps), so a longer run would be somewhat higher. (iii) DNase-seq replaces ATAC-seq (ATAC bigWigs for 7 of 15 cells only); the original DNase vs ATAC difference was not measured. (iv) Released-weight rows use the quantile-mapped inputs of the "q" variant; on the authors' own IMR-90 tracks the released C.Origami reaches 0.94 (chr2, chr3; n = 1) and the released Epiphany 0.51 on GM12878 chr3, 11, 17, which bounds what input mismatch costs the released rows. (v) Targets are our O/E, not the HiC-DC+ z-score of Epiphany or the log(count+1) of C.Origami, for the retrained models by design. (vi) One training seed per family; the cell-level pairing (11 and 3 cells) is the only variance estimate. (vii) 8 of 280 blocks failed once with a zip read race on the expected-file cache (concurrent writers) and were re-scored; no block is partial (9 or 10 sources everywhere).

### Files
`scripts/competitor_retrain/` (fetch_dnase, prep, train_corigami, train_epiphany, predict_*, finalize, blocks, table, score_workers.sh, vendor), `results/competitor_retrain/{table.md, per_cell.csv, summary.csv, vs_retrained.csv, dnase_picks.tsv, history_*.json}`, `results/hic_measures_chrho/chrho_competitor_retrain_{per_block,summary}.csv`; checkpoints and predictions on the share under `competitor_retrain/` and `runs/benchmark/competitor_oe/`. Hooks: `scripts/hic_measures.py` flags `--chrho-competitor-retrain`, `--chrho-cells`.

## 77b. Equal-data retraining of C.Origami and Epiphany at 2, 5, 10 and 25 kb with the original stopping rules (PENDING; follow-up to section 77, GPU capacity exhausted)

Question (author): is the gap of section 77 an artefact of resolution (C.Origami at 8 kb, Epiphany at 10 kb only) or of the shortened C.Origami schedule? Rules: same split, cells, targets and per-method input sets as section 77; original stopping rule of each method; no wall-clock budget; claims only at resolutions at which the retrained model was trained.

### Original stopping rules (read from the code)
- C.Origami (`src/corigami/training/main.py`): max_epochs 80, EarlyStopping(monitor val_loss, patience 80), ModelCheckpoint on val_loss; warm-up 10 epochs then cosine to epoch 80. The patience equals the epoch cap, so the effective rule is 80 epochs with the best-val-loss checkpoint.
- Epiphany (`epiphany/adversarial.py`): 55 epochs of 1 250 chunks, learning rate halved every 16 epochs, no early stopping, one checkpoint per epoch.
- Since neither has an active early-stopping criterion, the follow-up applies the author's rule on top of the original epoch schedule: validation loss every epoch, patience 12 epochs, best-loss weights kept, cap = the original epoch count (80 and 55; hit-the-cap is reported). One epoch = one pass over the training intervals of all 11 cells (C.Origami: 39 369 intervals = 4 921 steps of batch 8 at 8 kb) or 1 250 chunks (Epiphany).
- Consequence for section 77: the Epiphany 10 kb run (69 000 steps = 55 epochs, validation every 2 epochs, best validation MSE at step 30 000, epoch 24) already satisfies this rule (patience 12 would stop at epoch 36 with the same weights). The C.Origami 8 kb run of section 77 (4.9 passes) does not; it is being retrained to its stop.

### Resolution adaptation (implemented, `scripts/competitor_retrain/`)
- C.Origami keeps 256 bins, the 12 residual blocks, the transformer and the decoder unchanged. The encoder downsampling is the product of the input convolution (stride 2) and the block strides, and ConvBlock takes the stride as a parameter, so exact bins are possible: 8 192 bp = 2 x 2^12 (native); 2 000 = 2 x (2,2,2,5,5,5); 5 000 = 2 x (2,2,5,5,5,5); 10 000 = 2 x (2,2,2,5,5,5,5); 25 000 = 2 x (2,2,5,5,5,5,5), remaining blocks stride 1 (`BIN_STRIDES` in `vendor/corigami_models.py`). Windows: 0.512, 1.28, 2.56 and 6.4 Mb (256 bins each); no resampling, the output cools are on our grids. Deviations: stride-5 convolutions with kernel 5 (no overlap) and fewer blocks with receptive-field growth. Targets: our O/E at that grid; diagonals beyond those stored in the O/E cools (200) are masked. Interval span 5 Mb (2.4 x window above 2.2 Mb). Built and forward-tested at all four bins (12.84 M parameters, output 256 x 256).
- Epiphany keeps 200 rows per chunk and 100 diagonals, the Net of model_10kb.py, with input bins of 100 bp and the row stride and window scaled with the row grid (same number of bins, reach scales: 2 kb 200 kb, 5 kb 500 kb, 10 kb 1 Mb, 25 kb 2.5 Mb; windows 0.28, 0.7, 1.4, 3.5 Mb). The separate 5 kb architecture of the original (model_5kb.py, 30 input channels, 200 diagonals) is not used because it needs the original 30-track input set; Epiphany is therefore native only at 10 kb.
- Native resolutions: C.Origami 8 192 bp, Epiphany 10 kb. 10 and 25 kb of C.Origami and 2, 5 and 25 kb of Epiphany are adaptations trained natively at that grid; the 8 kb C.Origami output scored at 10 or 25 kb (section 77) is a resampled output and carries no claim.

### Our models at the four grids
chrho models exist at 25, 10 and 5 kb (`all12{refin,honest,refres}_drop_chrho`; 5 kb predicted for GM12878, IMR-90 and K562 only). No chrho model exists at 2 kb: `all12honest_drop_chrho_2kb` and `all12refin_drop_chrho_2kb` must be trained from the 5 kb chrho configs; the non-chrho 2 kb model saw the test chromosomes and is not scored on them.

### Status when this entry was written
- Running: C.Origami 8 192 bp to its stop, remote 4090, claim `competitor_retrain_corigami_8kb_native_epochs80`, 393 680 steps at the 80-epoch cap (9.5 to 12.8 samples/s depending on the other jobs on that GPU: 70 to 95 GPU hours).
- Epiphany 25 kb: two pilot starts failed (wrong target grid: the band reader ignored the row grid; then a tmp-file race of the dataloader workers on the target band files), both fixed in `common.py`; the third start was killed by an out-of-memory error when the Akita retraining occupied the second local GPU (17 GB used on each local GPU, 22.5 GB on the remote). No adapted model has been trained yet; no score exists.
- Queue ready, `scripts/competitor_retrain/queue_local_gpu1.sh`: Epiphany 25, 5, 2 kb (each about 6.4 h at 3.3 steps/s), then C.Origami 10, 25, 5, 2 kb (80-epoch cap each, estimated 70 to 100 GPU hours each). Estimated total 330 to 450 GPU hours without the 2 kb chrho models and the scoring.
- Not yet built: prediction names and block lists for 5 and 2 kb in `blocks.py`, the 2 kb scoring grid in `hic_measures.py`, the chrho 2 kb models.

### One-epoch smoke test of the 5 and 2 kb adaptations (2026-10-15; `results/competitor_retrain/test_5kb_2kb.md`)
- Both adapted C.Origami models train without error at the settings of the 8 kb run (4 921 steps per epoch; 66 min at 5 kb, 104 min at 2 kb under CPU contention; 5.4 and 4.7 GB; data-loader bound; 80 epochs about 88 h and 85 to 140 h).
- Validation after epoch 1 (chr18, chr21): Pearson 0.224 (5 kb) and 0.073 (2 kb) against 0.420 at 8 kb; MSE 6.6 % and 10 % below the zero prediction (8 kb: 15 %). Held-out block GM12878 chr14: Pearson 0.29 (5 kb), 0.04 (2 kb) against consensus 0.82 and 0.71 and our chrho models 0.77 to 0.90 (5 kb; none at 2 kb). One epoch decides no ranking; a rise needs the continuation of the two test directories.
- Bug: transient EAGAIN (`BlockingIOError`) of the band-file mmap on the share killed one attempt at step 4 400; `common.load_band` now retries. No error in the target grid.
- Added behind `COMPRT_BIN=5000|2000`: test rows and specs in `blocks.py`, 2 kb grid in `hic_measures.py` and `benchmark_blocks.params`, `score_test.py`, `--run-steps`. This closes the "not yet built" item for block lists and the 2 kb grid; the 2 kb chrho models are still missing.

## 78. Methylation as a proxy that replaces missing tracks (FINAL; section 77 was the last number when this was written)

Question: when CTCF, cohesin or histone tracks are missing, how much of the Hi-C accuracy lost is returned by the methylation track (Meth), as a replacement and not as an addition to a complete set.

### Design
- Replacement fraction RF = (score with Meth - score without) / (score with all tracks - score without), median Pearson over the 413 Akita test regions per cell, mean over cells, in the same trained model with Meth present or absent (track dropout makes both states in-distribution). Group sets as in section 35 (CTCF; cohesin RAD21+SMC3; active H3K27ac/H3K4me1/me2/me3/H3K9ac; gene body H3K36me3/H3K79me2/H4K20me1; repressive H3K27me3/H3K9me3; cum1 to cum3; CTCF only; CTCF+H3K27ac; CTCF+H3K27ac+H3K27me3; sequence only).
- Models (10 and 25 kb): all12refres_drop_meth (10 kb, + v2), all12refres_drop_motifmeth, all12honest_drop_meth (+ s1), all12honest_drop_motifmeth (+ s1 at 25 kb). Cells: seven WGBS training cells (honest meth), two for motifmeth models, K562 and HMEC held out.
- Loss threshold: median over cells of the absolute difference of the median Pearson of the two seeds of the honest family without Meth: 0.0062 (10 kb), 0.0072 (25 kb), 0.0139 (25 kb motifmeth); 0.0062 for consensus models (section 51). `reference_summary.csv` holds no seed replicates, so the s1 models of this section supply the spread. RF is reported only where mean loss >= threshold and the bootstrap lower bound of the loss is above 0. Bootstrap: 500 draws of regions (paired across sets), cells resampled when >= 3 cells.
- AML: aml_meth_refin folds 10 to 12, 8 test donors with WGBS (018, 027, 270, 168, 546, 629, 472, 773), own marks H3K27ac, H3K27me3, CTCF removed singly and together, with and without the patient's Meth; 10 and 25 kb; patient-specific windows; 10 kb loop and stripe objects of scripts/aml_loop_detection.py (1886 loops, 921 stripes against matched random pairs).

### Results (10 kb, median Pearson; loss and gain in Pearson units; RF with 95 percent interval; - = loss below threshold)
| set | honest held-out loss / gain / RF | honest training loss / gain / RF | consensus training loss / gain / RF |
|---|---|---|---|
| minus CTCF | 0.037 / 0.001 / 0.02 (-0.12 to 0.25) | 0.015 / 0.007 / 0.48 (-0.24 to 1.47) | 0.001 / 0.002 / - |
| minus cohesin | 0.008 / 0.009 / 1.19 (-0.13 to 8.9) | 0.013 / 0.008 / 0.59 (0.15 to 2.08) | 0.017 / 0.016 / 0.93 (0.72 to 1.13) |
| minus active | 0.024 / 0.011 / 0.48 (0.13 to 0.80) | 0.055 / 0.008 / 0.14 (-0.00 to 0.36) | 0.012 / 0.003 / 0.25 (-0.02 to 0.47) |
| minus gene body | -0.003 / 0.010 / - | 0.016 / 0.007 / 0.44 (-0.02 to 1.21) | 0.000 / 0.001 / - |
| minus repressive | 0.001 / 0.009 / - | 0.007 / 0.008 / 1.13 (-0.03 to 11) | 0.002 / 0.002 / - |
| cum2 | 0.007 / 0.009 / - | 0.020 / 0.008 / 0.39 (0.08 to 0.86) | 0.017 / 0.015 / 0.86 (0.65 to 1.04) |
| cum3 | 0.014 / 0.011 / 0.79 (0.09 to 2.7) | 0.039 / 0.007 / 0.19 (-0.02 to 0.42) | 0.018 / 0.014 / 0.76 (0.57 to 1.02) |
| CTCF only | 0.092 / 0.033 / 0.36 (0.25 to 0.48) | 0.174 / 0.033 / 0.19 (0.06 to 0.34) | 0.044 / 0.017 / 0.39 (0.18 to 0.54) |
| sequence only | 0.223 / 0.074 / 0.34 (0.24 to 0.38) | 0.254 / 0.048 / 0.19 (0.06 to 0.30) | 0.045 / 0.015 / 0.32 (0.12 to 0.47) |

25 kb honest, held-out: minus CTCF loss 0.055, gain 0.009, RF 0.17 (0.03 to 0.34); CTCF only 0.084 / 0.043 / 0.51 (0.35 to 0.72); sequence only 0.199 / 0.066 / 0.33 (0.23 to 0.45). Every other single group is below the threshold. Seed s1 and motif-plane models agree within the intervals (minus CTCF RF 0.11 to 0.31; CTCF only 0.35 to 0.47; sequence only 0.26 to 0.34), motif planes at 25 kb training cells raise cohesin and active RF (1.17, 0.53; two cells only). Consensus-input models on held-out cells lose at most 0.005 for any single group, so no RF exists there.

AML test donors (pooled over 8 donors): global Pearson loss from removing any of the three marks is at most 0.005 at 10 and 25 kb and the patient's Meth adds at most 0.001, so RF is undefined. In patient-specific windows removing H3K27ac costs 0.019 (10 kb) and 0.027 (25 kb) Pearson and Meth returns 0.014 (RF 0.71, -1.65 to 1.55) and 0.003; residual correlation with the AML reference in these windows: loss 0.041 and 0.155, gain 0.022 and 0.067, RF 0.52 (-0.17 to 1.32) and 0.43 (0.07 to 1.09). Removing H3K27me3 or CTCF alone costs nothing measurable. All three removed: 0.014 (10 kb) and 0.021 (25 kb) specific-window Pearson lost, 0.006 and 0.004 returned. Loops and stripes: removal of the three marks lowers the residual AUROC from 0.555 to 0.482 (loops) and from 0.542 to 0.451 (stripes); Meth gives +0.004 (RF 0.05, -0.39 to 0.39) and +0.028 (RF 0.31, -0.37 to 0.66). Share of AML-specific loops above the 90th percentile of random pairs: 0.482 full, 0.459 without the three marks, 0.466 with Meth; all RF intervals include 0.

### Mark summary (results/meth_proxy/mark_summary.csv; unseen-cell signal Pearson sequence -> sequence + Meth, gradient boosting; CNN gain in brackets)
| mark | signal Pearson seq -> seq+Meth | Hi-C loss when group removed (honest 10 kb, held-out / training) | Hi-C gain from Meth |
|---|---|---|---|
| CTCF | 0.42 -> 0.52 (+0.11) | 0.037 / 0.015 | 0.001 / 0.007 |
| RAD21, SMC3 | 0.43 -> 0.48, 0.45 -> 0.55 (+0.05, -0.01) | 0.008 / 0.013 | 0.009 / 0.008 |
| H3K27ac | 0.33 -> 0.65 (+0.22) | 0.024 / 0.055 | 0.011 / 0.008 |
| H3K4me1/me2/me3, H3K9ac | 0.34-0.49 -> 0.59-0.76 (+0.08 to +0.16) | same group | same group |
| H3K36me3, H3K79me2, H4K20me1 | 0.25-0.37 -> 0.44-0.68 (+0.02 to +0.17) | -0.003 / 0.016 | 0.010 / 0.007 |
| H3K27me3 | 0.39 -> 0.52 (+0.13) | 0.000 / 0.007 | 0.009 / 0.008 |
| H3K9me3 | 0.30 -> 0.33 (+0.08) | same group | same group |

### Findings (paragraph-ready)
Methylation returns a fraction of the Pearson lost after tracks are removed that depends on how much is removed and on whether the consensus is an input. Without the consensus the gain is 0.007 to 0.011 whichever single group is removed, including groups whose removal costs nothing, and grows to 0.033 (CTCF only) and 0.048 to 0.074 (sequence only), which is 0.19 to 0.36 of the loss at 10 kb and 0.33 to 0.51 at 25 kb. With the consensus as input the fractions are 0.32 to 0.39 for CTCF only and sequence only and 0.76 to 0.93 for cohesin, cum2 and cum3 on training cells, where the losses are 0.017 to 0.018. On the held-out cell types K562 and HMEC the consensus models lose at most 0.005 for any single group and cohesin, active and repressive losses are inside the seed spread; CTCF removal is not replaced (RF 0.02 at 10 kb, 0.17 at 25 kb). In the AML donors the three own marks carry at most 0.005 of the global Pearson given the AML reference, the patient's Meth adds at most 0.001, and in the patient-specific windows it returns about half of the H3K27ac loss at 25 kb (residual RF 0.43, 0.07 to 1.09); the loop and stripe recovery is not distinguishable from zero. Signal-level predictability from methylation is highest for H3K4me2, H3K9ac, H3K36me3 and H3K27ac and lowest for H3K9me3, H4K20me1, RAD21, and does not rank the Hi-C gain, which is flat across groups.

### Verdict and rule
Partial proxy, not a replacement. Never a full substitute for CTCF. Practical rule: with the reference consensus present, drop cohesin or active/repressive/gene-body marks and add Meth (loss stays <= 0.02); with a CTCF track missing or few marks available keep Meth as an input for roughly one third to one half of the lost Pearson; do not expect Meth to recover patient-specific loops or stripes.

### Caveats
(i) Only two held-out cells (K562, HMEC): the interval over regions does not cover the cell-to-cell variation, no cell bootstrap. (ii) Seed spread from two seeds of the honest family (0.006 to 0.007, per-set maxima up to 0.018); most single-group losses are below it, so their RF is undefined or unstable (RF above 1 with wide intervals reflects a Meth gain of 0.007 to 0.011 that is present without any removal and is additive). (iii) Group-level sets give the same gain to marks of one group, so the mark summary ranks groups. (iv) Motif-plane models: two training cells. (v) AML: eight donors, fold-wise references, the three marks are the whole AML panel.

### Files
`scripts/meth_proxy_analysis.py` (stages predict, aml, aml_loops, summary; run with the hicdiff env), `results/meth_proxy/{replacement_pooled.csv, replacement_by_cell.csv, seed_noise.csv, signal_level.csv, mark_summary.csv, aml_replacement_pooled.csv, aml_replacement_by_donor.csv, aml_loops_replacement.csv, aml_regions.csv.gz, aml_object_scores.csv.gz, meth_proxy.png}`; predictions under `/mnt/processed/wolffjoa/hic-multiome/runs/meth_proxy`.


## 79. Akita retrained on our cell lines and our chromosome splits (FINAL; section 78 was the last number when this was written)

Question: does the original Akita (architecture of manuscripts/akita/params.json of ~/src/basenji, one output head per cell, hg38 sequence only), trained on the same cells and the same O/E targets as our models, approach our chromosome-held-out models, and how does it compare with the released Akita on identical pixels?

### Design
- Data: 12 training cells (GM12878, IMR-90, HCT116, HFF, H1, HepG2, A549, NHEK, GM23248, MCF-7, CD14+ monocytes, PC-9), our O/E maps resampled onto Akita's 2048 bp grid (1 048 576 bp input, 448 x 448 target bins, 99 681 upper-triangle values per head, NaN pixels masked out of the loss); 12 heads, one per cell. Training windows lie inside the merged Akita training regions and do not overlap the Akita validation/test stretches (the regions and holdout BED of our chrho models), stride 229 376 bp.
- Split A: train 1, 5-10, 13, 15, 16, 18-22 (4 306 windows); validation chr4 and chr12 (691 windows, 150 per epoch used); test chr2, 3, 11, 14, 17 (the chromosomes held out of both chrho models). Split B: train odd chromosomes except 5 and 13 (2 725 windows); validation chr5 and chr13 (589); test even chromosomes. No window of any split overlaps another role.
- Scoring: scripts/akita_retrain/evaluate.py plugs the predictions into the blocks of scripts/hic_measures.py (10 kb grid, band 900 kb, pixels finite in the target and in every source, whole chromosome; hicFindTADs / hicDetectLoops of HiCExplorer 3.7.6, HiCRep and the other measures as in sections 37, 58 and 77). Sources per block: ours (all12refin_drop_chrho, all12honest_drop_chrho), Akita retrained A (and B on chr2 and chr14, where all models are held out), the released Akita for the five cells it has heads for (hff, h1, gm12878, imr90, hct116), and the consensus, best-other-map and replicate baselines. 173 blocks (12 cells x 5 chromosomes of A, 12 cells x 9 further even chromosomes of B, K562 x 5 for ours).
- Cells median over the 12 cells of the mean over chromosomes; per-cell values in `per_cell.csv`.

### Deviations from the original recipe (all forced by the data or the hardware)
- Optimizer: the original SGD (learning rate 0.0065, momentum 0.99575) did not train on our targets in the 12 min screening runs (validation Pearson 0.06 after 2 630 steps in fp32; bf16 and micro-batch variants ran out of memory or were released); Adam 3e-4 with batch size 2, gradient clipping 10, and a cosine decay over the wall-clock budget to 5 percent (lr-cosine 0.05) was used instead. Every layer of the trunk and the heads is unchanged except the number of output units (12 instead of 5).
- Epoch = one pass over the windows of build_data.py instead of the original TFRecord epochs; the loss is the MSE over finite pixels; patience 100 (effectively the wall-clock budget); the model of the best validation Pearson (mean over heads) is used for scoring.
- Schedule shortened: split A 6.5 h (25 epochs, 52 480 steps, best epoch 24, validation Pearson 0.451), split B 4.5 h (33 epochs, 44 240 steps, best epoch 15, validation Pearson 0.345). Pilot GM12878 single head: 1.5 h for 12 201 steps (2.7 to 4.4 steps/s on one RTX 4090), validation Pearson 0.29 at epoch 3, unstable afterwards (0.12 at epoch 5), which gave the 12-head budget estimate of about 6.5 h for 25 epochs. GPU time of the full runs 11.0 h in total (limit 14 h per full run); pilot and screening runs 3 h in addition.

### Results (median over cells, Pearson per 1 Mb region on O/E; the full measure set in `results/akita_retrain/summary_tables.md`)
| set | ours refin | ours honest (ChIP only) | Akita retrained A | Akita retrained B | Akita released | consensus of other cells |
|---|---|---|---|---|---|---|
| chr2, 3, 11, 14, 17, 12 cells | 0.928 | 0.789 | 0.461 | | | 0.893 |
| chr2, 3, 11, 14, 17, 5 cells of the released Akita | 0.942 | 0.824 | 0.474 | | 0.682 | 0.911 |
| chr2, 14, 12 cells (held out by all) | 0.931 | 0.812 | 0.437 | 0.217 | | 0.904 |
| chr2, 14, 5 cells | 0.945 | 0.849 | 0.437 | 0.264 | 0.671 | 0.911 |
| even chromosomes 4-22 (split B test), 12 cells | not scored (trained on these) | | | 0.225 | | 0.889 |

Other measures on chr2, 3, 11, 14, 17 (12 cells; SCC / insulation r / boundary recall / loop recall): ours refin 0.908 / 0.869 / 0.670 / 0.412; ours honest 0.745 / 0.702 / 0.504 / 0.208; Akita retrained A 0.446 / 0.509 / 0.339 / 0.077.

Verdict. Retrained on our cells with sequence as the only input, Akita reaches Pearson 0.46 on the held-out chromosomes of split A, below our ChIP-only model (0.79), below our model with Hi-C reference (0.93), and below the released Akita on its five cells (0.47 retrained against 0.68 released, same pixels). With the odd/even split the score is lower (0.22 to 0.26) because fewer training windows are available and the training chromosomes are only 2 725 windows. The retrained model also scores below the consensus of the other cells (0.89), which needs no model.

### Caveats
(i) Akita has one head per training cell and cannot predict an unseen cell type; K562 (no head) is scored for our models only, rows in `summary_unseen_cell_k562.csv`. Our models predict K562 with Pearson 0.889 (refin) and 0.738 (honest). (ii) Optimizer and schedule differ from the original (see Deviations); the released Akita was trained on 5 cells for far longer on the Akita authors' own target preparation (5 cells, different normalization and 2048 bp bins, the same genome), so the retrained scores are an upper bound of what our short schedule delivered, not of what Akita can reach with its original training budget; the SGD recipe did not train on our targets within the screening time and was not tuned further. (iii) Validation Pearson of the retrained models fluctuates by 0.01 to 0.03 between epochs, one seed, no variance estimate. (iv) The retrained Akita is fitted on a 2048 bp grid and sampled bilinearly onto the 10 kb pixels as for the released Akita; band 900 kb. (v) Our models receive the consensus12 or the cell-type Hi-C reference as input (refin) or the ChIP marks (honest) and the retrained Akita receives sequence only; the honest rows are the closest in information but still carry ChIP marks.

### Files
`scripts/akita_retrain/` (build_data, common, train, predict, evaluate, run_local.sh, run_remote.sh), `results/akita_retrain/{summary_tables.md, summary_*.csv, per_cell.csv, per_block.csv}`; runs, predictions and logs in /mnt/processed/wolffjoa/hic-multiome/akita_retrain/{runs,preds,logs}; blocks in the chrho root of hic_measures as chrhoakitaA_* and chrhoakitaB_*.

## 80. Contact structure shared between cell types: pairwise SCC of 15 measured maps (FINAL)

Question: how similar are the measured contact maps of different cell types, the basis of the consensus
reference? Stratum-adjusted correlation (HiCRep definition, smoothing h 5, 1 Mb band) between the O/E target maps
`crosscell_oe/<cell>_proper_<res>kb.cool` of the 12 training cells, K562, HMEC (deep) and HUVEC (deep), all 105
pairs, on the held-out chromosomes 2, 3, 11, 14 and 17, at 25 and 10 kb; per pair the mean over the five
chromosomes. `hicmultiome.evaluate.scc_streaming` per chromosome (the hicrep package refuses the pairs because the
4DN and ENCODE maps differ in their chromosome sets). No training, CPU only.

`scripts/pairwise_celltype_scc.py` (hicdiff env, 16 workers, about 15 min); `results/pairwise_celltype_scc/pairs_{25,10}kb.csv`.

| | 25 kb | 10 kb |
|---|---|---|
| median over 105 pairs | 0.777 | 0.778 |
| interquartile range | 0.746 to 0.818 | 0.748 to 0.812 |
| share of pairs above 0.7 | 0.85 | 0.87 |
| minimum (H1 vs CD14+ monocytes) | 0.330 | 0.367 |
| maximum (IMR-90 vs GM23248) | 0.910 | 0.908 |
| minimum without CD14+ monocytes and H1 (78 pairs) | 0.688 | 0.704 |

For scale (sections 37, 73, HiCRep column): replicate pairs 0.93 to 0.99 in the three benchmark blocks, the
consensus of the other cells against the left-out cell 0.83 to 0.86. Published predictors against the consensus
and the best single other cell on the held-out chromosomes (same tables): Epiphany and Hi-cGAN below the consensus
at 25, 10 and 5 kb in HiCRep SCC and loop recall, and below the best single other cell except Hi-cGAN loop recall at
5 kb (0.068 against 0.020); C.Origami (IMR-90, its training cell type) above the consensus in HiCRep SCC at every
resolution (0.950, 0.933, 0.872 against 0.844, 0.860, 0.815) but below it in boundary recall (0.752, 0.473, 0.348
against 0.788, 0.641, 0.581) and loop recall (0.313, 0.162, 0.164 against 0.374, 0.383, 0.339). On Akita's 413 regions
(section 73) the four-cell consensus is above Akita in SCC, loop and boundary recall. Manuscript Fig. 1.

## 81. C.Origami on a cell type it was not trained on: GM12878 de novo (FINAL)

Question: does the advantage of C.Origami over the consensus on IMR-90 (section 80) hold on a cell type outside its
training set? The released `corigami_base.ckpt` (trained on IMR-90) was applied to GM12878 with the C.Origami authors'
own processed GM12878 inputs (`corigami_data/data/hg38/gm12878/genomic_features/{ctcf_log2fc,atac}.bw`), whole
chromosomes 2 and 3 (its held-out test and validation chromosomes) through the whole-chromosome cool writer on the
hic-prediction-workbench branch (`--full-chr`, 2 Mb windows every 32 kb, 8192 bp bins). Reproducibility: chr21 of
IMR-90 rerun with the same code is identical to the earlier prediction. Scored with the chrho pipeline (O/E stage
`benchmark_competitors.run_oe`, then `hic_measures.py --chrho-corigami-denovo`, opt-in competitor
`corigami_gm12878_denovo_8kb` behind `BENCH_CORIGAMI_DENOVO=1`) at 25, 10 and 5 kb, blocks `chrho_codenovo`.

| | 25 kb | 10 kb | 5 kb |
|---|---|---|---|
| HiCRep SCC, C.Origami | 0.661 | 0.654 | 0.640 |
| HiCRep SCC, consensus | 0.823 | 0.822 | 0.812 |
| HiCRep SCC, best single other cell (K562) | 0.748 | 0.749 | 0.708 |
| HiCRep SCC, ours (all12refin_drop) | 0.931 | 0.920 | 0.917 |
| boundary recall, C.Origami / consensus | 0.586 / 0.782 | 0.379 / 0.613 | 0.319 / 0.527 |
| loop recall, C.Origami / consensus | 0.202 / 0.405 | 0.089 / 0.386 | 0.078 / 0.350 |

Verdict. On GM12878 C.Origami falls below the consensus and the best single other cell in SCC, boundary and loop
recall at every resolution; its advantage in SCC on IMR-90 is specific to its training cell type. Caveat: two
chromosomes, one cell type, inputs as processed by the C.Origami authors.

`results/hic_measures_chrho/chrho_corigami_denovo_{summary,per_block}.csv`, log `results/chrho_corigami_denovo.log`;
predictions `~/src/C.Origami/denovo/gm12878/chr{2,3}.cool`. Manuscript Fig. 1c, d.

## 82. Prediction time and memory, whole chromosome into a cool file (FINAL)

Question: what does it cost to predict a whole chromosome, and on how little hardware can each method run? Every
method predicts all of GM12878 chromosome 2 (242 Mb) into one cool file, one job at a time, on one RTX 4090 or on 8
threads of a Ryzen 9 7950X without GPU. Wall time from process start to exit (model loading and cool writing
included); peak GPU memory from nvidia-smi and peak resident memory from psutil, both over the process tree (all
tools run in one process, so no shared pages are counted twice). "Default" is each tool's default batch, "lowest"
batch 1 (C.Origami: 452 data batches instead of 16). Ours: reference-input model with test-time reverse complement
(two forward passes per window), stride half a window. Released competitors cannot write whole chromosomes; each run
uses the whole-chromosome cool writer on our forks: C.Origami 7e6f197 (2 Mb window every 32 kb, pixels collected in
Python lists, first window kept per pixel), Epiphany 9be0ab4 (2 Mb chunks every 2 Mb, Hann taper), Akita/basenji
9c1fcd9 (head 2, GM12878). Epiphany has no batch setting (one chunk at a time; `--batch` is the progress step), so one
run. Akita first measured 22.5 GB at both batches because TensorFlow reserves the whole card; rerun with
`TF_FORCE_GPU_ALLOW_GROWTH=true` (allocator high-water mark), first rows discarded.

`scripts/inference_benchmark.py`; `results/inference_benchmark/runs.csv`; logs and coolers in
/mnt/processed/wolffjoa/hic-multiome/runs/inference_benchmark/.

| method | grid | device | setting | time (s) | peak GPU (MiB) | peak RAM (MiB) | cool (MB) |
|---|---|---|---|---|---|---|---|
| ours | 25 kb | GPU | default / lowest | 11.7 / 11.3 | 836 / 722 | 1164 / 1146 | 3.0 |
| ours | 10 kb | GPU | default / lowest | 21.4 / 22.3 | 1382 / 724 | 1643 / 1361 | 16.0 |
| ours | 5 kb | GPU | default / lowest | 45.4 / 47.5 | 3226 / 948 | 3026 / 2972 | 62.2 |
| ours | 2 kb | GPU | default / lowest | 118.3 / 113.3 | 4976 / 1190 | 6616 / 6556 | 208.9 |
| ours | 25 kb | CPU | lowest | 20.7 | 0 | 816 | 3.0 |
| ours | 10 kb | CPU | lowest | 49.6 | 0 | 1210 | 16.0 |
| Epiphany | 10 kb | GPU | only | 17.9 | 2682 | 1424 | 7.9 |
| Akita | 2048 bp | GPU | default / lowest | 124.0 / 119.3 | 8736 / 6592 | 4288 / 4050 | 116.3 |
| C.Origami | 8192 bp | GPU | default / lowest | 3970 / 3693 | 3624 / 3624 | 58547 / 45504 | 24.9 |

Verdict. At its lowest setting our model needs less than 1 GB of GPU memory from 25 to 5 kb and 1.2 GB at 2 kb, less
than every competitor at its lowest (Epiphany 2.6 GB, C.Origami 3.5 GB, Akita 6.4 GB), and runs without GPU at 25 and
10 kb within a minute per chromosome and 1.2 GB RAM. Time is comparable to Epiphany at 10 kb and to Akita at 2 kb
(both with fewer forward passes: Epiphany non-overlapping, no reverse complement); C.Origami takes an hour per
chromosome and 45 GB RAM, a cost of the 32 kb window stride and pixel lists of its cool writer rather than of its
model. Caveats: one chromosome, one run per job, warm file cache; competitor writers are ours, not the authors'.
