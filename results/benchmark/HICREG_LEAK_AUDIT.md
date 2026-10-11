# HiC-Reg: routes by which its evaluation leaks, and the leak-free retraining on our split

Code references are to `~/src/HiC-Reg` (original, unchanged) and to this repository. Numbers
are in `results/benchmark/hicreg_leak_*.csv`, `hicreg_retrained_*.csv` and registry section 62.

## 1. Within-chromosome cross-validation (the released CV predictions)

- `Scripts/genPairFeatures/Framework.C:199-300` (`splitPairs`, `splitRegionsGenPairs`) and the
  dispatch at `:1511-1518`: with `splittype=regionwise` the 5 kb regions of one chromosome are
  randomly assigned to `foldCnt` folds (`populateRandIntegers`, `:251`); the test file of fold k
  holds every pair with at least one region in the fold, the training file the pairs with both
  regions outside it. `README.md` (Step 1.1) and `genPairFeatures.sh` use this for 5 folds,
  `prerandomize_pairs=yes`. The `pairwise` option of the same program splits pairs at random and
  would leak more.
- The released models carry their training pair lists; `scripts/hicreg_audit.py --stage split`
  (`hicreg_split.csv`) recovers: 64.0% of the chromosome's pairs train each model, adjacent
  regions fall in different folds in 79.8-79.9% of cases (random), and of the 8 neighbouring 5 kb
  pixels of a predicted pixel 19% (GM12878) and 13-15% (K562) were training pixels of every model
  that could have predicted it, 69-73% and 49-65% of at least one; 72-98% of the predicted
  pixels have a training pixel among their 8 neighbours. With random folds a test pair has one
  region among the training regions in 0.32/0.36 = 89% of the cases (both regions in the test
  fold: 11%). The released score is a within-map interpolation.
- Quantified on our own pipeline (same features, target, forest; CV protocol of
  `hicreg_retrain.py --stage cv`, five region-wise folds, 4M-pair cap) against the model trained
  on the 17 other chromosomes: section 4 below.

## 2. Features that contain the target map or its normalisation

- Not a leak route. The released feature files (header of
  `Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt`) hold 43 columns: 14 signals (Ctcf, Dnase,
  H3k27ac, H3k27me3, H3k36me3, H3k4me1, H3k4me2, H3k4me3, H3k79me2, H3k9ac, H3k9me3, H4k20me1,
  RAD21, TBP) as `_E` (region 1), `_P` (region 2) and `_W` (mean of the regions between, 0.01
  when none; checked on the rows of region `chr17_0_5000`), `Distance` (gap in bp) and `Count`.
  No column is derived from Hi-C. The optional `correlation` flag (`Framework.C:790, 898, 1039`)
  correlates the two regions' signals, not Hi-C, and is `no` in `genPairFeatures.sh`.
- The Hi-C map enters as the target ln(1 + SQRTVC count) (README example: 212.50 -> 5.3636);
  the SQRTVC vector is computed on the whole matrix including the held-out chromosome's own
  contacts, which affects the target only.

## 3. Prediction support taken from the held-out target (found in this audit)

- The pair list that is featurised and predicted is the list of non-zero pixels of the sparse
  Hi-C matrix of the chromosome (`Framework.C:148-196` (`readPairs`; the target is `log(paircnt+1)`, `:191`)). The
  released CV and cross-chromosome files of a chromosome hold exactly the same pairs
  (`hicreg_leak_support_released.csv`): GM12878 chr14 74.3% and chr17 85.1% of the 1 Mb pair
  slots, K562 chr14 42.4% and chr17 59.9%; in our hg38 4DN maps 5.8% and 3.9% (GM12878) and 38.7%
  and 24.4% (K562) of the mappable slots of chr14 and chr17 are zero-count pixels at 5 kb
  (`hicreg_leak_zero_share.csv`; 55% and 40% beyond 500 kb for K562).
- `scripts/oe_proper.py:118-119, 143` turns a mappable pixel without a value into the clipped
  depletion (-2), which is the value of a zero-count pixel of the target. A prediction file that
  only exists for non-zero pixels therefore delivers the zero pixels of the held-out map for
  free, in both the released CV and the released cross-chromosome files. K562 5 kb: the retrained,
  leak-free predictions score 0.758 (5 kb), 0.544 (10 kb) and 0.419 (25 kb) Pearson when the pixels whose target count is
  zero are removed from them (as in the release), against 0.394, 0.297 and 0.227 with every pair predicted
  (`hicreg_leak_support_effect.csv`, mean of chr14 and chr17, same model and pixels, only the support differs;
  GM12878, with 4-6% zero pixels: 0.512, 0.456, 0.394 against 0.420, 0.401, 0.358). The released K562
  cross-chromosome score of 0.774 (5 kb) is what the retrained model reaches with the released support (0.758).

## 4. Training and test pairs from the same chromosome; anchors across the split

Same features, target, forest and 4M-pair cap (CV protocol, pairs with both regions outside the
fold train, five random region folds) against training on the 17 other chromosomes; mean of
chr14 and chr17, WINDOW, CV minus cross-chromosome (`hicreg_retrained_leak.csv`):

| cell, grid | Pearson | tile median | SCC | HiCRep | insulation r | loop recall |
|---|---|---|---|---|---|---|
| GM12878 25 kb | +0.449 (0.879 - 0.430) | +0.362 | +0.372 | +0.286 | +0.113 | +0.165 |
| GM12878 10 kb | +0.433 | +0.379 | +0.370 | +0.333 | +0.033 | +0.043 |
| GM12878 5 kb | +0.418 | +0.381 | +0.361 | +0.349 | +0.029 | +0.025 |
| K562 25 kb | +0.515 (0.801 - 0.286) | +0.408 | +0.452 | +0.376 | +0.106 | +0.083 |
| K562 10 kb | +0.482 | +0.377 | +0.430 | +0.406 | +0.064 | +0.036 |
| K562 5 kb | +0.367 | +0.374 | +0.343 | +0.330 | +0.067 | +0.010 |

The released CV maps minus the retrained cross-chromosome maps: +0.478 (GM12878 25 kb) and +0.601
(K562 25 kb) Pearson; the released CV maps minus the released cross-chromosome maps: +0.330 and
+0.152, the K562 difference being small because the released cross-chromosome K562 score carries
the support leak of section 3. A size control (WINDOW trained on 4M pairs of the 17 chromosomes)
scores like the 75M-pair model (chr14+chr17 Pearson 0.432 against 0.430 GM12878 25 kb, 0.257 against
0.286 K562), so the training size does not explain the gap. The CV protocol on our pipeline (0.879,
0.801) reproduces the released CV level (0.908, 0.888) within 0.03 to 0.09 Pearson.

## 5. Cross-cell predictions trained on the same chromosome

- `~/data/HiC-Reg/Gm12878/README_Gm12878.txt:5, 53`: the cross-cell predictions use the models
  and feature data of the CV mode of the same chromosome: a model trained (5 folds, 64% of the
  pairs) on the other cell's map of chr14 or chr17 is applied to this cell's chr14 or chr17.
  The other cell's Hi-C of the very chromosome that is scored is in the training data.
- Retrained counterpart (model of the other cell trained on the other 17 chromosomes, applied to
  chr14 and chr17 features of this cell): Pearson 0.327 (GM12878 25 kb), 0.168 (K562 25 kb)
  against the released cross-cell maps at 0.668 and 0.815 (registry section 18, block F, same
  25 kb); the released maps differ by the same-chromosome training and by the support of section 3
  (their pair list is the scored cell's own), so the two effects are not separated.

## 6. MULTICELL features

- `Scripts/genMULTICELLfeats/gen-MULTICELL-features.py:18-22` merges the five cells on the pair
  name (outer) and fills every gap with 0.01 including the target column; `:62-71` cuts the merged
  table into contiguous row blocks of the pair-sorted table. Applied to the pair list of the
  released CV file, these folds are clustered (a test fold is a set of leading-digit groups of the
  region name): the neighbouring pixels of a test pair are training pairs in 0.02-0.11% of the
  cases, and 6-15% of the test regions occur in training pairs (`hicreg_leak_folds.csv`).
- The released MULTICELL predictions have exactly the pair counts of the WINDOW files (chr17:
  2,748,296 lines GM12878, 1,933,968 K562), not the union table, so the released run did not use
  this script as shipped. The MULTICELL models ship without `trainset_error.txt` (22 files, trees
  only), so their folds cannot be reconstructed; the CV score (0.976 against 0.961 for WINDOW,
  GM12878 25 kb, registry section 18) is that of a leaky protocol, not of a clustered hold-out.
- `README_Gm12878.txt:4` states that MULTICELL cross-chromosome predictions use "the same feature
  data and models in CV mode"; the cross-chromosome file scores at cross-chromosome level (0.62,
  not 0.9+), so it was not scored with CV models.

## 7. Hyper-parameter selection on the test set

- No tuner is shipped. README, `Scripts/LearnModelsPred/HiCReg_modeltrain_pred.sh` and the
  model file names fix `-k1 -l10 -n20` for every chromosome and cell; `Potential.C:145-186`
  shows the forest (bootstrap sample per tree, one third of the features per split, leaf size).
  There is no evidence of selection on the scored pairs in the released files; the values are
  documented defaults ("1 should be OK", "e.g. 10"). The only tuner on disk is the local
  `hyperopt_hicreg.py` (Ray Tune; objective = distance-stratified Pearson AUC of the prediction
  against `--comparisonMatrix`, i.e. against the held-out matrix), which selects on the test
  chromosome by construction; none of the released or section 18 numbers come from it.
- In the retraining the hyper-parameters (minimum leaf size, training-set size) were chosen on
  chr10, a training chromosome, never on chr2, 3, 11, 14 or 17 (`hicreg_retrain_tune_*.csv`).

## 8. Resolution and build conversions

- HiC-Reg is hg19, 5 kb. Section 18 samples the maps at hg38 bin centres lifted with the UCSC
  chain (`benchmark_blocks.py:136-150`), a small handicap (blur, dropped regions) and no leak.
  The retrained maps are hg38 native.
- 10 and 25 kb are sums of the 5 kb counts, expm1 of the predicted ln(1 + count)
  (`hicreg_audit.py:159-170`; the retrained maps use the same conversion). The first
  section 18 table summed the log values (0.943), the count-space sum scores 0.959. Pairs of
  one bin (d = 0) do not exist in HiC-Reg, so a diagonal 25 kb bin sums 15 of its 25 5 kb pairs.
  The 1 Mb reach (200 5 kb bins) cuts the 25 kb sums at the band edge.

## 9. Released cross-chromosome files are handicapped as well

- Each is trained on the single other chromosome (chr14 -> chr17 and back), 81 to 107 Mb of one
  cell, with DNase and TBP; the retraining below uses 17 chromosomes (75 M pairs) but no DNase
  and no TBP (not available for our cells; SMC3 added).

## 10. HiC-Reg rebuilt for our cells and retrained on our split

Script `scripts/hicreg_retrain.py` (stages data, tune, fit, cv, check, cools), scoring
`scripts/hic_measures.py --stage chrho_score chrho_report --chrho-hicreg-retrained`, tables
`scripts/hicreg_retrained_table.py`.

- Features: mean bigWig signal per 5 kb bin of our 13 tracks (CTCF, RAD21, SMC3, H3K27ac, H3K27me3,
  H3K36me3, H3K4me1, H3K4me2, H3K4me3, H3K79me2, H3K9ac, H3K9me3, H4K20me1) as E, P, W and Distance
  (40 features, layout of the released feature files checked on their rows). HiC-Reg's DNase and
  TBP do not exist for our cells and are dropped; SMC3 is added; RAD21 and SMC3 are absent for
  HMEC, HUVEC and NHEK (constant 0.01) in the MULTICELL variant, whose five cells are HiC-Reg's
  (GM12878, K562, HUVEC, HMEC, NHEK; 196 features).
- Target: HiC-Reg's native one, ln(1 + SQRTVC count), from the VC_SQRT vectors of the 4DN mcools
  (GM12878 4DNFIXP4QG5B, K562 4DNFI18UHVRO; the vectors divide: checked against the ICE weights),
  zero-count pairs of mappable bins included and every pair of the chromosome predicted, so the
  predicted pair list does not depend on the target. Scoring converts exactly as for the released
  maps (expm1, sum into 10 and 25 kb, `oe_proper.py`, 1 Mb reach) and compares with our ICE-based
  proper O/E targets on the identical pixels of the chrho blocks. The same true SQRTVC maps pushed
  through this conversion score 0.78-0.87 (25 kb), 0.87-0.92 (10 kb) and 0.90-0.94 (5 kb) Pearson,
  tile median 0.97-0.98 (`hicreg_leak_oracle.csv`): the ceiling of HiC-Reg's native target in this
  scoring, below 1 because SQRTVC and ICE are different normalisations.
- Forest: scikit-learn RandomForestRegressor with the regForest settings (20 trees, bootstrap per
  tree, one third of the features per split, leaf size), because regForest takes 20 s per tree on
  60k pairs and is single-threaded. Check on the same 60k training and 20k validation-chromosome
  pairs (`hicreg_retrain_check_regforest_gm12878.json`): r with the truth 0.826 (regForest) and
  0.830 (scikit-learn), r between the two predictions 0.983.
- Training: chr1, 4-10, 12, 13, 15, 16, 18-22, all 75.3M (GM12878) and 75.4M (K562) mappable pairs
  within 1 Mb (no sub-sampling for WINDOW; MULTICELL 12M random pairs). Leaf size and training size
  were chosen on chr10 (distance-centred Pearson of ln(1 + count); `hicreg_retrain_tune_*.csv`):
  leaf 100 (HiC-Reg default 10) is best for both cells at 16M pairs (0.360 GM12878, 0.254 K562;
  leaf 10: 0.343, 0.238; leaf 1000: 0.346, 0.244); 4M against 16M pairs changes it by 0.004. The
  same values are used for MULTICELL. Test chromosomes chr2, 3, 11, 14, 17 enter no training and no
  choice. One seed.
- K562 note: our models never see K562 Hi-C (cell held out); retrained HiC-Reg trains on the K562
  Hi-C of the other 17 chromosomes.

Scoring (`results/hic_measures_chrho/chrho_hicregre_*.csv`, 42 blocks; chrom-wise identical pixels;
mean of chr14 and chr17, where the released predictions exist; Pearson over the band, 1 Mb tile
median, SCC, insulation r, boundary recall, rank-matched loop recall, APA ratio, compartment r,
HiCRep). "CV, our pipeline" is the released protocol (region-wise 5-fold CV inside the chromosome)
on the retrained pipeline.

GM12878 25 kb, mean of chr14 and chr17

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| all12refin_drop_chrho | 0.931 | 0.950 | 0.925 | 0.905 | 0.833 | 0.396 | 0.792 | 0.954 | 0.935 |
| all12honest_drop_chrho (ChIP only) | 0.783 | 0.798 | 0.763 | 0.674 | 0.608 | 0.191 | 0.319 | 0.867 | 0.791 |
| consensus of the other cells | 0.868 | 0.901 | 0.853 | 0.804 | 0.805 | 0.396 | 0.818 | 0.894 | 0.857 |
| HiC-Reg retrained (our split), WINDOW | 0.430 | 0.523 | 0.482 | 0.502 | 0.331 | 0.050 | -0.110 | 0.755 | 0.599 |
| HiC-Reg retrained (our split), MULTICELL | 0.383 | 0.569 | 0.473 | 0.498 | 0.342 | 0.039 | -0.124 | 0.741 | 0.601 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.579 | 0.620 | 0.538 | 0.447 | 0.322 | 0.032 | -0.002 | 0.846 | 0.668 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.615 | 0.634 | 0.567 | 0.455 | 0.346 | 0.038 | -0.032 | 0.856 | 0.692 |
| HiC-Reg CV, our pipeline | 0.879 | 0.886 | 0.855 | 0.615 | 0.554 | 0.216 | 0.357 | 0.979 | 0.885 |
| HiC-Reg CV (released) | 0.908 | 0.965 | 0.870 | 0.677 | 0.723 | 0.323 | 0.765 | 0.987 | 0.915 |
| (chr2, 3, 11, 14, 17) all12refin_drop_chrho / all12honest_drop_chrho / HiC-Reg retrained WINDOW, Pearson | 0.927 / 0.766 / 0.364 | | | | | | | | |

GM12878 10 kb, mean of chr14 and chr17

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| all12refin_drop_chrho | 0.925 | 0.937 | 0.919 | 0.890 | 0.665 | 0.408 | 0.728 | 0.952 | 0.927 |
| all12honest_drop_chrho (ChIP only) | 0.788 | 0.810 | 0.777 | 0.735 | 0.495 | 0.182 | 0.301 | 0.854 | 0.796 |
| consensus of the other cells | 0.853 | 0.882 | 0.842 | 0.793 | 0.674 | 0.397 | 0.902 | 0.880 | 0.861 |
| HiC-Reg retrained (our split), WINDOW | 0.446 | 0.467 | 0.473 | 0.523 | 0.215 | 0.014 | -0.009 | 0.776 | 0.549 |
| HiC-Reg retrained (our split), MULTICELL | 0.428 | 0.540 | 0.470 | 0.515 | 0.224 | 0.011 | -0.038 | 0.761 | 0.555 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.590 | 0.561 | 0.542 | 0.491 | 0.208 | 0.004 | 0.016 | 0.857 | 0.601 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.623 | 0.574 | 0.569 | 0.501 | 0.229 | 0.000 | 0.006 | 0.858 | 0.622 |
| HiC-Reg CV, our pipeline | 0.879 | 0.846 | 0.843 | 0.556 | 0.254 | 0.057 | 0.140 | 0.981 | 0.882 |
| HiC-Reg CV (released) | 0.920 | 0.939 | 0.881 | 0.626 | 0.393 | 0.148 | 0.494 | 0.991 | 0.925 |
| (chr2, 3, 11, 14, 17) all12refin_drop_chrho / all12honest_drop_chrho / HiC-Reg retrained WINDOW, Pearson | 0.920 / 0.774 / 0.398 | | | | | | | | |

GM12878 5 kb, mean of chr14 and chr17

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| all12refin_drop_chrho | 0.906 | 0.911 | 0.902 | 0.898 | 0.562 | 0.363 | 0.675 | 0.957 | 0.923 |
| all12honest_drop_chrho (ChIP only) | 0.749 | 0.758 | 0.749 | 0.741 | 0.456 | 0.188 | 0.385 | 0.871 | 0.770 |
| consensus of the other cells | 0.821 | 0.842 | 0.817 | 0.798 | 0.556 | 0.354 | 0.946 | 0.885 | 0.853 |
| HiC-Reg retrained (our split), WINDOW | 0.454 | 0.433 | 0.475 | 0.537 | 0.146 | 0.005 | 0.026 | 0.773 | 0.525 |
| HiC-Reg retrained (our split), MULTICELL | 0.448 | 0.503 | 0.474 | 0.534 | 0.163 | 0.034 | 0.005 | 0.761 | 0.534 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.603 | 0.520 | 0.554 | 0.515 | 0.161 | 0.000 | 0.045 | 0.891 | 0.593 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.626 | 0.519 | 0.571 | 0.525 | 0.195 | 0.000 | 0.035 | 0.890 | 0.611 |
| HiC-Reg CV, our pipeline | 0.872 | 0.814 | 0.836 | 0.565 | 0.166 | 0.030 | 0.101 | 0.980 | 0.874 |
| HiC-Reg CV (released) | 0.893 | 0.898 | 0.859 | 0.647 | 0.252 | 0.020 | 0.398 | 0.992 | 0.922 |
| (chr2, 3, 11, 14, 17) all12refin_drop_chrho / all12honest_drop_chrho / HiC-Reg retrained WINDOW, Pearson | 0.898 / 0.745 / 0.419 | | | | | | | | |

K562 25 kb, mean of chr14 and chr17

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| all12refin_drop_chrho | 0.897 | 0.928 | 0.889 | 0.857 | 0.738 | 0.347 | 0.843 | 0.927 | 0.904 |
| all12honest_drop_chrho (ChIP only) | 0.730 | 0.777 | 0.711 | 0.622 | 0.568 | 0.125 | 0.187 | 0.834 | 0.748 |
| consensus of the other cells | 0.857 | 0.905 | 0.853 | 0.839 | 0.769 | 0.333 | 0.876 | 0.887 | 0.868 |
| HiC-Reg retrained (our split), WINDOW | 0.286 | 0.436 | 0.347 | 0.497 | 0.255 | 0.049 | -0.097 | 0.667 | 0.465 |
| HiC-Reg retrained (our split), MULTICELL | 0.297 | 0.486 | 0.392 | 0.528 | 0.291 | 0.029 | -0.250 | 0.680 | 0.518 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.736 | 0.820 | 0.696 | 0.484 | 0.523 | 0.038 | 0.292 | 0.927 | 0.797 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.740 | 0.819 | 0.700 | 0.500 | 0.519 | 0.030 | 0.260 | 0.927 | 0.800 |
| HiC-Reg CV, our pipeline | 0.801 | 0.844 | 0.799 | 0.603 | 0.493 | 0.133 | 0.232 | 0.960 | 0.841 |
| HiC-Reg CV (released) | 0.888 | 0.950 | 0.864 | 0.670 | 0.688 | 0.178 | 0.698 | 0.973 | 0.923 |
| (chr2, 3, 11, 14, 17) all12refin_drop_chrho / all12honest_drop_chrho / HiC-Reg retrained WINDOW, Pearson | 0.892 / 0.717 / 0.271 | | | | | | | | |

K562 10 kb, mean of chr14 and chr17

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| all12refin_drop_chrho | 0.858 | 0.900 | 0.857 | 0.848 | 0.627 | 0.366 | 0.568 | 0.924 | 0.887 |
| all12honest_drop_chrho (ChIP only) | 0.715 | 0.724 | 0.714 | 0.721 | 0.484 | 0.139 | 0.220 | 0.848 | 0.745 |
| consensus of the other cells | 0.811 | 0.866 | 0.814 | 0.836 | 0.660 | 0.386 | 0.811 | 0.879 | 0.847 |
| HiC-Reg retrained (our split), WINDOW | 0.336 | 0.417 | 0.378 | 0.554 | 0.117 | 0.028 | 0.071 | 0.664 | 0.452 |
| HiC-Reg retrained (our split), MULTICELL | 0.361 | 0.420 | 0.426 | 0.584 | 0.228 | 0.016 | -0.027 | 0.685 | 0.500 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.747 | 0.760 | 0.708 | 0.496 | 0.267 | 0.002 | 0.210 | 0.954 | 0.782 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.756 | 0.763 | 0.719 | 0.541 | 0.326 | 0.000 | 0.186 | 0.954 | 0.793 |
| HiC-Reg CV, our pipeline | 0.817 | 0.794 | 0.808 | 0.618 | 0.259 | 0.064 | 0.126 | 0.967 | 0.858 |
| HiC-Reg CV (released) | 0.878 | 0.899 | 0.858 | 0.650 | 0.413 | 0.009 | 0.456 | 0.985 | 0.925 |
| (chr2, 3, 11, 14, 17) all12refin_drop_chrho / all12honest_drop_chrho / HiC-Reg retrained WINDOW, Pearson | 0.857 / 0.718 / 0.336 | | | | | | | | |

K562 5 kb, mean of chr14 and chr17

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| all12refin_drop_chrho | 0.771 | 0.817 | 0.776 | 0.842 | 0.510 | 0.338 | 0.718 | 0.925 | 0.814 |
| all12honest_drop_chrho (ChIP only) | 0.641 | 0.641 | 0.653 | 0.730 | 0.415 | 0.169 | 0.283 | 0.831 | 0.664 |
| consensus of the other cells | 0.673 | 0.777 | 0.684 | 0.810 | 0.505 | 0.323 | 0.819 | 0.875 | 0.727 |
| HiC-Reg retrained (our split), WINDOW | 0.414 | 0.324 | 0.440 | 0.625 | 0.089 | 0.030 | 0.108 | 0.636 | 0.428 |
| HiC-Reg retrained (our split), MULTICELL | 0.439 | 0.333 | 0.479 | 0.654 | 0.150 | 0.026 | 0.038 | 0.677 | 0.453 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.774 | 0.703 | 0.755 | 0.599 | 0.226 | 0.003 | 0.261 | 0.959 | 0.790 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.783 | 0.716 | 0.768 | 0.640 | 0.272 | 0.003 | 0.242 | 0.962 | 0.810 |
| HiC-Reg CV, our pipeline | 0.781 | 0.697 | 0.783 | 0.692 | 0.178 | 0.039 | 0.114 | 0.959 | 0.758 |
| HiC-Reg CV (released) | 0.845 | 0.814 | 0.840 | 0.728 | 0.282 | 0.005 | 0.441 | 0.978 | 0.899 |
| (chr2, 3, 11, 14, 17) all12refin_drop_chrho / all12honest_drop_chrho / HiC-Reg retrained WINDOW, Pearson | 0.768 / 0.638 / 0.430 | | | | | | | | |

All five chromosomes, mean Pearson (all12refin_drop_chrho / all12honest_drop_chrho / HiC-Reg
retrained WINDOW): the last row of each table; the per-chromosome values, all measures, are in
`hicreg_retrained_per_chromosome.csv`, group means in `hicreg_retrained_summary.csv` and
`hicreg_retrained_table.md`.

### Verdict

- Retrained leak-free on our split, HiC-Reg is not better than the released cross-chromosome
  predictions; it is clearly worse (WINDOW Pearson 0.430 against 0.579 for GM12878 25 kb, 0.286
  against 0.736 for K562 25 kb; 0.454 against 0.603 and 0.414 against 0.774 at 5 kb). The 17
  training chromosomes do not help (size control above); what the released files have that the
  retrained maps do not is the support of section 3 (K562) and DNase and TBP among the features
  (not separable here; with the released support the retrained K562 5 kb maps score 0.758 against
  0.774 released). The margin of our models grows rather than narrows: all12honest_drop_chrho
  (ChIP only, 13 tracks) leads retrained HiC-Reg by 0.35 (GM12878 25 kb) and 0.44 (K562 25 kb)
  Pearson and by 0.29 and 0.23 at 5 kb; all12refin_drop_chrho by 0.50 and 0.61 (25 kb).
  Both models are ahead of retrained HiC-Reg on all nine measures in both cells and all three
  grids (chr14+chr17 means).
- The MULTICELL variant is level with WINDOW (25 kb Pearson 0.383, 0.297; 5 kb 0.448, 0.439).
- Within-chromosome leak (CV minus cross-chromosome, same pipeline): +0.42 to +0.45 Pearson
  (GM12878) and +0.37 to +0.52 (K562), +0.36 to +0.41 tile median, +0.34 to +0.45 SCC.
- Loop recall and APA ratio of retrained HiC-Reg stay at 0.005 to 0.05 and -0.11 to 0.11 at every
  grid; its compartment r (0.64 to 0.78) is below the released maps (0.85 to 0.96).

### Caveats

- The retrained model differs from the original in the features (no DNase and TBP, SMC3 added),
  in the Hi-C source (4DN hg38 maps, VC_SQRT vectors of the mcool, instead of the Rao hg19 SQRTVC
  matrices), in the forest implementation (scikit-learn, agreement 0.983 with regForest on a
  60k-pair check) and in the bigWig signal source (our tracks, not sequencing-depth-normalised read
  counts). The gap to the released cross-chromosome maps combines the support effect of section 3
  with these differences; the audit separates only the support (0.758 against 0.774 for K562 5 kb,
  smaller shares elsewhere). A retrained model with the original inputs is not possible: the
  released feature files exist for chr14 and chr17 only.
- One seed per model; chr14 and chr17 are two chromosomes per block (win counts are per
  chromosome); MULTICELL was trained on 12M pairs and scored on chr14 and chr17 only; the CV
  protocol on our pipeline caps training at 4M pairs (64% of the pairs of the chromosome is
  about 2M, so the cap does not bind).
- The released predictions are hg19, lifted to the hg38 bin centres by the scorer; the retrained
  maps are hg38 native.
- The 5 kb tile median and boundary recall of the sparse K562 map are noisy (39% zero pixels).

### Compute

Remote (64 cores, 251 GB, nice 10, at most 48 cores): hyper-parameter grid 3 rounds x 2 cells
(24 threads each, about 70 min), final WINDOW fits 2 x 75M pairs (20 threads each, about 28 min),
MULTICELL fits 2 x 12M pairs (20 threads each, about 20 min). Local (nice 10): features and
targets (5 min), CV protocol and size control (about 1 h at 8 threads), regForest check (9 min),
conversion to cools and O/E (about 20 min), scoring of 42 blocks (4 processes of 2 threads,
about 1.9 h).

