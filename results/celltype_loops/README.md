# Cell-type-specific loops, IMR-90 versus GM12878 (10 kb, held-out chromosomes)

Script: `scripts/celltype_specific_loops.py`. Chromosomes: 2, 3, 11, 14, 17.

## Inputs
- loop calls (hicDetectLoops of the measured maps): `/mnt/processed/wolffjoa/hic-multiome/runs/benchmark/calls/{imr90,gm12878}_10kb_loops.bedgraph` (columns chrom1 start1 end1 chrom2 start2 end2 score); held-out calls: {'imr90': 2201, 'gm12878': 2233}
- maps (10 kb): measured `crosscell_oe/<cell>_proper_10kb.cool`; consensus alone `crosscell_oe/consensus_loo_<cell>_proper_10kb.cool`; ours `runs/hic_measures_chrho/pred/10kb/all12refin_drop_chrho/<cell>.cool` (with consensus) and `all12honest_drop_chrho` (ChIP only)

## Definitions
- Value at a loop: maximum over the 3x3 pixel neighbourhood of the loop pixel; difference d = IMR-90 minus GM12878, for the measured and for every predicted map.
- Shared: called in both maps (a call of the other map within 1 bin in both coordinates); 957 IMR-90 calls match GM12878 calls before the finite filter.
- IMR-90-specific: called in IMR-90, not shared, measured d >= threshold. GM12878-specific: called in GM12878, not shared, measured d <= -threshold. Thresholds 1 (primary) and 0.5 (natural-log units of O/E). Loops called in one map and neither shared nor specific are unclassified.
- Union of calls 3477; loops with a non-finite 3x3 value in any predictor or measured map are dropped for all predictors (228 dropped, 3249 scored; `loops_scored.csv`), so every predictor is scored on the identical loops.
- AUROC: positives = specific class, negatives = shared or the opposite specific class; score = d (IMR-90-specific positives) or -d (GM12878-specific positives); sklearn `roc_auc_score`.

## Class counts (scored loops; `class_counts.csv`)
| threshold | class | n_loops |
|---|---|---|
| 1.0 | unclassified | 2171 |
| 1.0 | shared | 936 |
| 1.0 | imr90_specific | 121 |
| 1.0 | gm12878_specific | 21 |
| 0.5 | unclassified | 1585 |
| 0.5 | shared | 936 |
| 0.5 | imr90_specific | 582 |
| 0.5 | gm12878_specific | 146 |

## AUROC (`auroc.csv`)
| threshold | predictor | positives | negatives | auroc | n_pos | n_neg |
|---|---|---|---|---|---|---|
| 1.0 | measured (ceiling) | imr90_specific | shared | 1.0 | 121 | 936 |
| 1.0 | measured (ceiling) | imr90_specific | gm12878_specific | 1.0 | 121 | 21 |
| 1.0 | measured (ceiling) | gm12878_specific | shared | 1.0 | 21 | 936 |
| 1.0 | measured (ceiling) | gm12878_specific | imr90_specific | 1.0 | 21 | 121 |
| 1.0 | ours with consensus | imr90_specific | shared | 0.794 | 121 | 936 |
| 1.0 | ours with consensus | imr90_specific | gm12878_specific | 0.985 | 121 | 21 |
| 1.0 | ours with consensus | gm12878_specific | shared | 0.938 | 21 | 936 |
| 1.0 | ours with consensus | gm12878_specific | imr90_specific | 0.985 | 21 | 121 |
| 1.0 | ours ChIP only | imr90_specific | shared | 0.824 | 121 | 936 |
| 1.0 | ours ChIP only | imr90_specific | gm12878_specific | 0.964 | 121 | 21 |
| 1.0 | ours ChIP only | gm12878_specific | shared | 0.813 | 21 | 936 |
| 1.0 | ours ChIP only | gm12878_specific | imr90_specific | 0.964 | 21 | 121 |
| 1.0 | consensus alone | imr90_specific | shared | 0.0 | 121 | 936 |
| 1.0 | consensus alone | imr90_specific | gm12878_specific | 0.0 | 121 | 21 |
| 1.0 | consensus alone | gm12878_specific | shared | 0.0 | 21 | 936 |
| 1.0 | consensus alone | gm12878_specific | imr90_specific | 0.0 | 21 | 121 |
| 0.5 | measured (ceiling) | imr90_specific | shared | 0.952 | 582 | 936 |
| 0.5 | measured (ceiling) | imr90_specific | gm12878_specific | 1.0 | 582 | 146 |
| 0.5 | measured (ceiling) | gm12878_specific | shared | 0.998 | 146 | 936 |
| 0.5 | measured (ceiling) | gm12878_specific | imr90_specific | 1.0 | 146 | 582 |
| 0.5 | ours with consensus | imr90_specific | shared | 0.681 | 582 | 936 |
| 0.5 | ours with consensus | imr90_specific | gm12878_specific | 0.961 | 582 | 146 |
| 0.5 | ours with consensus | gm12878_specific | shared | 0.923 | 146 | 936 |
| 0.5 | ours with consensus | gm12878_specific | imr90_specific | 0.961 | 146 | 582 |
| 0.5 | ours ChIP only | imr90_specific | shared | 0.706 | 582 | 936 |
| 0.5 | ours ChIP only | imr90_specific | gm12878_specific | 0.941 | 582 | 146 |
| 0.5 | ours ChIP only | gm12878_specific | shared | 0.848 | 146 | 936 |
| 0.5 | ours ChIP only | gm12878_specific | imr90_specific | 0.941 | 146 | 582 |
| 0.5 | consensus alone | imr90_specific | shared | 0.049 | 582 | 936 |
| 0.5 | consensus alone | imr90_specific | gm12878_specific | 0.0 | 582 | 146 |
| 0.5 | consensus alone | gm12878_specific | shared | 0.002 | 146 | 936 |
| 0.5 | consensus alone | gm12878_specific | imr90_specific | 0.0 | 146 | 582 |

## Notes
- Competitors (C.Origami IMR-90 only, Epiphany and Hi-cGAN GM12878 only) cover one cell each, so they cannot form a difference and are not scored.
- Measured is a ceiling by construction: the classes are defined from the measured difference.
- The consensus alone is a leave-one-out mean of the other cells. The consensus for IMR-90 contains GM12878 and the consensus for GM12878 contains IMR-90, so its difference equals (GM12878 minus IMR-90) divided by the number of averaged cells (up to the 3x3 maximum): it carries the measured contrast with inverted sign, and its AUROC near 0 reflects this identity, not a predictor of the contrast. The model with consensus receives the same two inputs, so its contrast scores are obtained with an opposite-signed contrast in its reference input.
