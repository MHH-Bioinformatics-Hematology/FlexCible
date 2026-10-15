# C.Origami adapted to 5 kb and 2 kb: one-epoch smoke test (registry 77b)

Purpose: check that the stride-adapted architecture and the data pipeline train at 5 kb and 2 kb. One epoch each
(one full pass over the 39 369 training intervals of the 11 training cells, 4 921 steps of batch 8), settings of
`common.py` / `train_corigami.py` unchanged (Adam 2e-4, schedule of an 80-epoch run: warm-up 10 epochs, so the
learning rate reaches only 2e-5 at the end of epoch 1; bf16; micro-batch 4). Added only `--run-steps` (stop and
validate after N steps; the 80-epoch schedule is kept). The epoch took less than 2.5 h on both, so the 1 500-step
fallback was not used. Validation on chr18 and chr21 (132 windows, 12 per cell), test blocks on chr14 and chr2
(never in training or validation). Local GPUs 0 (5 kb) and 1 (2 kb), claims `competitor_retrain_corigami_test_5kb`
and `_2kb`. Run directories `competitor_retrain/runs/corigami_test_{5,2}kb`, logs `competitor_retrain/logs/`,
validation histories `history_corigami_test_{5,2}kb.json` in this folder.

## Training cost

| bin | window | steps/epoch | min/epoch | samples/s | GPU memory (peak allocated) | projected 80 epochs |
|---|---|---|---|---|---|---|
| 8 192 bp (running reference, remote 4090) | 2.10 Mb | 4 921 | 72 (80 for later epochs) | 8.3 | 6.6 GB | 95 to 105 h |
| 5 kb | 1.28 Mb | 4 921 | 66 (attempt 2) | 9.3 to 11.5 | 5.4 GB | 88 h |
| 2 kb | 0.512 Mb | 4 921 | 104 (CPU contention, load 25 of 32 cores) | 5.4 to 8.3 (6.3 mean) | 4.7 GB | 139 h (about 85 h at 10 samples/s) |

Both runs are data-loader bound (GPU utilisation near 0 in the snapshots: target area averaging and reading of the
memory-mapped feature files on 8 workers), not GPU bound. The 80-epoch figures are upper bounds unless the
early-stopping rule (patience 12) ends the run sooner.

## Validation after epoch 1 (11 training cells, chr18 and chr21)

| bin | train loss (last 200 steps) | validation MSE | zero-prediction MSE (same windows) | MSE reduction | validation Pearson per window |
|---|---|---|---|---|---|
| 8 192 bp (reference) | 0.245 | 0.267 | 0.314 | 15 % | 0.420 |
| 5 kb | 0.237 | 0.189 | 0.202 | 6.6 % | 0.224 |
| 2 kb | 0.171 | 0.149 | 0.166 | 10 % | 0.073 |

The 8 kb run reaches 0.587 (epoch 2), 0.624 (epoch 3) and 0.617 (epoch 4). The 5 and 2 kb runs have one point
each, so a rise cannot be shown; the training loss fell from about 1.0 to 0.24 (5 kb) and 0.17 (2 kb) within the
epoch.

## Held-out blocks (identical pixels, `chrho_rtco` set of `hic_measures.py`)

Tile-median Pearson is `oe_pearson_tile_median` (1 Mb tiles); SCC is `oe_scc`; all on the pixels finite in the
target and in every listed source (C.Origami reach: 256 bins). Our models are the chrho models.

GM12878 chr14, 5 kb (4.13 M pixels, band 383 bins):

| source | O/E Pearson | tile-median Pearson | SCC | insulation r | HiCRep |
|---|---|---|---|---|---|
| all12refin_drop_chrho | 0.900 | 0.907 | 0.898 | 0.895 | 0.919 |
| all12refres_drop_chrho | 0.891 | 0.903 | 0.890 | 0.876 | 0.916 |
| all12honest_drop_chrho | 0.766 | 0.768 | 0.765 | 0.743 | 0.781 |
| consensus | 0.819 | 0.840 | 0.817 | 0.795 | 0.847 |
| best other measured map | 0.560 | 0.675 | 0.574 | 0.691 | 0.722 |
| replicate | 0.684 | 0.741 | 0.733 | 0.903 | 0.900 |
| C.Origami 5 kb, 1 epoch | 0.287 | 0.210 | 0.246 | -0.132 | 0.155 |

GM12878 chr14, 2 kb (11.0 M pixels, band 255 bins; no chrho model exists at 2 kb):

| source | O/E Pearson | tile-median Pearson | SCC | insulation r | HiCRep |
|---|---|---|---|---|---|
| consensus | 0.712 | 0.692 | 0.713 | 0.810 | 0.728 |
| best other measured map | 0.531 | 0.458 | 0.547 | 0.700 | 0.668 |
| replicate | 0.456 | 0.417 | 0.513 | 0.806 | 0.685 |
| C.Origami 2 kb, 1 epoch | 0.043 | 0.036 | 0.036 | -0.024 | 0.056 |

IMR-90 chr2, 2 kb (29.1 M pixels): consensus 0.740 / 0.720 / 0.731 (Pearson / tile median / SCC), best other map
0.637 / 0.631 / 0.633, replicate 0.661 / 0.568 / 0.731, C.Origami 2 kb 0.052 / 0.053 / 0.014. IMR-90 chr2 at
5 kb was not scored (about 2 h of CPU for one block; GM12878 chr14 is the smoke test at that bin).

The one-epoch C.Origami values are far below every baseline, as expected after one epoch at a learning rate of
at most 2e-5; they are the numbers of a model at the start of its schedule and carry no claim.

## Bugs found

- Transient `BlockingIOError` (EAGAIN) in `np.load(..., mmap_mode="r")` of a target band file on the share,
  raised in a dataloader worker; it killed the first 5 kb attempt at step 4 400 of 4 921 (107 min, no checkpoint
  before the epoch end). Fix: `common.load_band` retries up to 6 times. The log of the first attempt is kept as
  `corigami_test_5kb_attempt1_crashed_BlockingIOError.log`. The running 8 kb job uses the unpatched code and is
  exposed to the same error; it resumes from `latest.pt` (written every epoch).
- No error in the target grid or the band files: the 2 kb target built by `co_target` equals the cool pixels
  (r = 1.000, max abs difference 4.6e-4, all pixels valid), and shifting the prediction by 1 to 3 bins does not
  improve its correlation with the target (no off-by-one in placement).
- Missing definitions (added behind the existing keys and `COMPRT_BIN=5000|2000` only, defaults unchanged, 52 unit
  tests pass): `blocks.py` test rows and specs (`corigami_t5kb_<cell>`, `corigami_t2kb_<cell>`), `hic_measures.py`
  2 kb window (256), insulation square (25 bins), boundary tolerance (5 bins) and best-other table,
  `benchmark_blocks.params(2000)` (the 10 kb row with every window multiplied by 5), `score_test.py` driver.
- Scoring cost: 35 min (GM12878 chr14, 2 kb) and 75 min (IMR-90 chr2, 2 kb) per block, almost all in the native
  measures (HiCRep, ENT3C).

## Verdict

5 kb: trains and learns, but is clearly behind the 8 kb run at the same point (validation Pearson 0.22 against
0.42; MSE 6.6 % below the zero prediction against 15 %). 2 kb: trains (loss falls, no OOM, 4.7 GB) but is barely
above chance after one epoch (Pearson 0.07, MSE 10 % below the zero prediction, block Pearson 0.04 to 0.05).
Whether either rises as the 8 kb run did (0.42, 0.59, 0.62) is not decided by one epoch; continuing the two test
directories (`latest.pt` is kept) for 3 to 5 epochs would decide it at 3 to 9 GPU hours. The 80-epoch cost is
about 88 h (5 kb) and 85 to 140 h (2 kb), limited by data loading.
