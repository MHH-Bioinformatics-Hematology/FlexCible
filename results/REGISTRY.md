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

## 7. Biological replicates, 5 cell lines, 2 to 25 kb (PENDING)

`scripts/replicates_all_cells.py` then `scripts/replicate_corr.py`: ENCODE
MboI in situ Hi-C, one lane-pair per replicate, minibwa, HiCExplorer v4
`hicBuildMatrix`, ICE, `oe_proper.py`, rep1 vs rep2 on the 413 regions.
Output `replicates/replicate_corr.csv` (appears after the GM12878 rep2 rebuild
finishes). First single-cell run (`scripts/raw_hic_ceiling.sh`, HMEC,
unbalanced): 0.415 at 2 kb with 171 M contacts, 5 to 23 times fewer contacts
than the maps in section 6, so replicate agreement at 2 kb is limited by
depth.

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
