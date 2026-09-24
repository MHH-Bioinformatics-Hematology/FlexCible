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

## 1. Comparison with Akita (FINAL)

Model `configs/akita_native.yaml` (13 ChIP tracks, CTCF motif, 2048 bp, Akita
O/E target), scored with `scripts/score_like_hicgan.py` against Akita's
published GM12878 target on the paired regions.

- `akita_comparison/score_like_hicgan.csv`: per region, ours vs Akita vs Hi-cGAN.
- 222 paired regions: ours 0.652, Akita 0.556; ours higher in 68% of regions.

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
