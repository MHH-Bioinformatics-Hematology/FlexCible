# Flexible input on the one model all12one

Script `scripts/flexible_inputs.py`; registry section 87. Model: the 10, 25, 5 and 2 kb views of the one shared checkpoint all12one (reference = input, track dropout uniform_count). Metric: per-region Pearson correlation on the 413 Akita test regions with test-time augmentation; every number below is the median over regions (quartiles per set are in `summary.csv`, per-region values in `per_region.csv` and `per_region_random.csv.gz`). The consensus alone, scored on the same regions, is the baseline. "With consensus" gives the reference plane the leave-one-out consensus (gm12878, imr90) or the 12-cell consensus (k562, hmec, huvec); "without consensus" sets that plane to 0, the expected level. HMEC and HUVEC have no RAD21 and SMC3 track, so sets that ask for them are scored with what exists (`marks_present` in `summary.csv`). Random subsets (sizes 1 to 12, 20 draws each, numpy seed 0, `subsets.json`) are identical for every cell and resolution. Parenthesised ranges are the smallest and largest per-draw median.

## Main findings

With the consensus, the model with no ChIP mark at all (sequence, motifs and the consensus) reaches the level of the consensus alone at 10 kb (K562 0.854 against 0.859, HMEC 0.804 against 0.809, HUVEC 0.760 against 0.769, GM12878 0.846 against 0.858, IMR-90 0.898 against 0.903). One mark lifts K562 to 0.876 (draw median), and all 13 marks to 0.906. The gain from marks is therefore the distance between the consensus level and the all-13 level: 0.052 in K562, 0.095 in GM12878 and 0.049 in IMR-90 at 10 kb. In HMEC and HUVEC the gain is 0.005 and 0.021 at 10 kb and 0.007 and 0.013 at 25 kb, so in these two cells the marks add little to the consensus whatever their number.

The curve rises steeply over the first three to four marks and flattens after that. In K562 at 10 kb with the consensus, four random marks give 0.893, three quarters of the gain from none to all 13 (0.854 to 0.906), and eight give 0.900. In GM12878 three marks give 0.913 (70 per cent of the gain), six give 0.930 (88 per cent). The seven-mark set of the pre-registered cells (CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3) is within 0.015 of all 13 in every cell at 10 and 25 kb (K562 0.897 against 0.906, GM12878 0.935 against 0.941). CTCF with H3K27ac recovers most of it (K562 0.888, GM12878 0.925, IMR-90 0.931), CTCF with RAD21 less (K562 0.873, GM12878 0.863).

Without the consensus the accuracy depends strongly on the marks. No marks give 0.08 to 0.18 (sequence and motifs alone), one mark 0.25 to 0.47 (draw medians at 10 kb), four marks 0.33 to 0.62, all 13 marks 0.46 (HMEC) to 0.73 (IMR-90). The curve still rises at 12 marks (K562 0.660 at 12, 0.662 at 13, GM12878 0.613 and 0.622), so every mark contributes. CTCF is the single most informative mark without consensus in the held-out cells (mean 0.388 over K562, HMEC, HUVEC against 0.31 for the next, H3K27ac), and the mark whose removal costs most (mean 0.435 against 0.52 to 0.55 for any other removal). With the consensus the ranking of single marks is flat: at 10 kb the held-out mean runs from 0.825 (H3K9ac, H3K27ac) to 0.805 (H3K9me3); removing one mark of 13 changes the held-out mean by at most 0.004 (consensus present), CTCF and H3K4me1 being the most costly.

Resolution. At 25 kb the picture is the same as at 10 kb. At 5 and 2 kb (K562, GM12878) the consensus alone is weaker (K562 0.785 and 0.629, GM12878 0.821 and 0.661) and the marks add more: all 13 marks give 0.833 and 0.666 in K562, 0.916 and 0.789 in GM12878. Without consensus all 13 marks give 0.58 at 5 kb and 0.41 to 0.45 at 2 kb, so flexible input without consensus loses accuracy at the finest resolution, while the model with consensus keeps the benefit of any mark that is given.

## Accuracy against the number of marks (median over 20 draws, range of draws)

Marks 0 and 13 are single sets (no range).

**10 kb, with consensus** (consensus alone: k562 0.859, hmec 0.809, huvec 0.769, gm12878 0.858, imr90 0.903)

| n_marks | k562 | hmec | huvec | gm12878 | imr90 |
|---|---|---|---|---|---|
| 0 | 0.855 | 0.804 | 0.760 | 0.846 | 0.898 |
| 1 | 0.876 (0.86-0.89) | 0.803 (0.79-0.81) | 0.765 (0.76-0.78) | 0.876 (0.86-0.91) | 0.901 (0.88-0.93) |
| 2 | 0.882 (0.87-0.89) | 0.803 (0.79-0.81) | 0.774 (0.76-0.78) | 0.902 (0.87-0.92) | 0.918 (0.84-0.94) |
| 3 | 0.885 (0.87-0.90) | 0.805 (0.78-0.81) | 0.775 (0.76-0.79) | 0.913 (0.89-0.93) | 0.930 (0.86-0.94) |
| 4 | 0.893 (0.88-0.90) | 0.806 (0.78-0.81) | 0.776 (0.76-0.79) | 0.920 (0.89-0.94) | 0.934 (0.91-0.94) |
| 6 | 0.895 (0.89-0.91) | 0.807 (0.80-0.82) | 0.779 (0.77-0.79) | 0.930 (0.92-0.94) | 0.944 (0.94-0.95) |
| 8 | 0.900 (0.89-0.91) | 0.808 (0.80-0.81) | 0.777 (0.76-0.79) | 0.935 (0.93-0.94) | 0.945 (0.94-0.95) |
| 10 | 0.901 (0.89-0.91) | 0.809 (0.80-0.81) | 0.781 (0.77-0.79) | 0.938 (0.94-0.94) | 0.947 (0.94-0.95) |
| 13 | 0.906 | 0.808 | 0.781 | 0.941 | 0.947 |

**10 kb, without consensus** (consensus alone: k562 0.859, hmec 0.809, huvec 0.769, gm12878 0.858, imr90 0.903)

| n_marks | k562 | hmec | huvec | gm12878 | imr90 |
|---|---|---|---|---|---|
| 0 | 0.082 | 0.128 | 0.114 | 0.116 | 0.175 |
| 1 | 0.281 (0.17-0.54) | 0.247 (0.13-0.29) | 0.298 (0.08-0.34) | 0.345 (0.18-0.42) | 0.470 (0.32-0.51) |
| 2 | 0.394 (0.19-0.59) | 0.283 (0.21-0.46) | 0.317 (0.23-0.53) | 0.395 (0.32-0.62) | 0.516 (0.34-0.68) |
| 3 | 0.409 (0.18-0.61) | 0.289 (0.23-0.46) | 0.316 (0.19-0.53) | 0.418 (0.31-0.62) | 0.535 (0.42-0.69) |
| 4 | 0.566 (0.30-0.64) | 0.333 (0.23-0.47) | 0.408 (0.22-0.53) | 0.529 (0.32-0.62) | 0.616 (0.51-0.68) |
| 6 | 0.609 (0.45-0.68) | 0.361 (0.28-0.50) | 0.410 (0.24-0.59) | 0.527 (0.47-0.62) | 0.643 (0.58-0.72) |
| 8 | 0.611 (0.48-0.68) | 0.388 (0.28-0.48) | 0.391 (0.26-0.56) | 0.548 (0.50-0.62) | 0.678 (0.61-0.72) |
| 10 | 0.631 (0.54-0.68) | 0.450 (0.32-0.49) | 0.501 (0.29-0.55) | 0.603 (0.51-0.63) | 0.707 (0.64-0.73) |
| 13 | 0.662 | 0.455 | 0.492 | 0.622 | 0.726 |

**25 kb, with consensus** (consensus alone: k562 0.897, hmec 0.898, huvec 0.887, gm12878 0.882, imr90 0.928)

| n_marks | k562 | hmec | huvec | gm12878 | imr90 |
|---|---|---|---|---|---|
| 0 | 0.876 | 0.886 | 0.867 | 0.857 | 0.909 |
| 1 | 0.888 (0.88-0.91) | 0.887 (0.88-0.89) | 0.868 (0.86-0.89) | 0.882 (0.86-0.91) | 0.914 (0.89-0.94) |
| 2 | 0.903 (0.89-0.91) | 0.885 (0.88-0.89) | 0.877 (0.85-0.89) | 0.900 (0.87-0.92) | 0.922 (0.86-0.95) |
| 3 | 0.911 (0.90-0.92) | 0.886 (0.88-0.89) | 0.880 (0.87-0.90) | 0.912 (0.89-0.92) | 0.934 (0.87-0.95) |
| 4 | 0.919 (0.90-0.93) | 0.885 (0.86-0.89) | 0.880 (0.87-0.89) | 0.917 (0.89-0.93) | 0.943 (0.91-0.95) |
| 6 | 0.921 (0.91-0.93) | 0.889 (0.88-0.90) | 0.882 (0.87-0.89) | 0.930 (0.92-0.94) | 0.951 (0.94-0.95) |
| 8 | 0.926 (0.92-0.93) | 0.890 (0.88-0.90) | 0.879 (0.87-0.89) | 0.933 (0.93-0.94) | 0.952 (0.95-0.95) |
| 10 | 0.928 (0.92-0.93) | 0.891 (0.88-0.89) | 0.877 (0.87-0.89) | 0.936 (0.93-0.94) | 0.953 (0.95-0.95) |
| 13 | 0.933 | 0.893 | 0.880 | 0.938 | 0.955 |

**25 kb, without consensus** (consensus alone: k562 0.897, hmec 0.898, huvec 0.887, gm12878 0.882, imr90 0.928)

| n_marks | k562 | hmec | huvec | gm12878 | imr90 |
|---|---|---|---|---|---|
| 0 | 0.116 | 0.178 | 0.159 | 0.093 | 0.202 |
| 1 | 0.333 (0.23-0.54) | 0.280 (0.18-0.35) | 0.384 (0.16-0.46) | 0.359 (0.19-0.45) | 0.481 (0.32-0.57) |
| 2 | 0.401 (0.22-0.61) | 0.328 (0.28-0.54) | 0.431 (0.30-0.63) | 0.387 (0.25-0.63) | 0.540 (0.36-0.71) |
| 3 | 0.429 (0.19-0.63) | 0.324 (0.28-0.54) | 0.423 (0.26-0.63) | 0.400 (0.26-0.58) | 0.565 (0.42-0.69) |
| 4 | 0.550 (0.29-0.65) | 0.395 (0.24-0.46) | 0.501 (0.29-0.62) | 0.480 (0.25-0.61) | 0.609 (0.53-0.68) |
| 6 | 0.596 (0.42-0.68) | 0.399 (0.30-0.53) | 0.507 (0.36-0.66) | 0.490 (0.43-0.61) | 0.647 (0.56-0.72) |
| 8 | 0.599 (0.49-0.68) | 0.405 (0.27-0.51) | 0.490 (0.31-0.66) | 0.505 (0.45-0.60) | 0.668 (0.60-0.72) |
| 10 | 0.623 (0.53-0.69) | 0.457 (0.34-0.52) | 0.578 (0.37-0.63) | 0.569 (0.46-0.62) | 0.697 (0.61-0.73) |
| 13 | 0.682 | 0.471 | 0.592 | 0.574 | 0.725 |

**5 kb, with consensus** (consensus alone: k562 0.785, gm12878 0.821)

| n_marks | gm12878 | k562 |
|---|---|---|
| 0 | 0.817 | 0.780 |
| 1 | 0.851 (0.83-0.88) | 0.797 (0.78-0.82) |
| 3 | 0.888 (0.86-0.90) | 0.810 (0.79-0.83) |
| 6 | 0.906 (0.89-0.91) | 0.819 (0.81-0.83) |
| 13 | 0.916 | 0.833 |

**5 kb, without consensus** (consensus alone: k562 0.785, gm12878 0.821)

| n_marks | gm12878 | k562 |
|---|---|---|
| 0 | 0.093 | 0.075 |
| 1 | 0.300 (0.16-0.41) | 0.268 (0.14-0.42) |
| 3 | 0.373 (0.30-0.61) | 0.334 (0.18-0.54) |
| 6 | 0.520 (0.41-0.58) | 0.520 (0.36-0.59) |
| 13 | 0.579 | 0.576 |

**2 kb, with consensus** (consensus alone: k562 0.629, gm12878 0.661)

| n_marks | gm12878 | k562 |
|---|---|---|
| 0 | 0.672 | 0.626 |
| 1 | 0.701 (0.64-0.73) | 0.637 (0.63-0.66) |
| 3 | 0.747 (0.66-0.77) | 0.647 (0.63-0.66) |
| 6 | 0.780 (0.76-0.79) | 0.659 (0.64-0.67) |
| 13 | 0.789 | 0.666 |

**2 kb, without consensus** (consensus alone: k562 0.629, gm12878 0.661)

| n_marks | gm12878 | k562 |
|---|---|---|
| 0 | 0.099 | 0.098 |
| 1 | 0.202 (0.10-0.31) | 0.187 (0.11-0.31) |
| 3 | 0.289 (0.18-0.45) | 0.268 (0.20-0.39) |
| 6 | 0.373 (0.31-0.44) | 0.400 (0.31-0.46) |
| 13 | 0.408 | 0.447 |

**Named sets, 10 kb**

| cons | set_name | k562 | hmec | huvec | gm12878 | imr90 |
|---|---|---|---|---|---|---|
| without | all13 | 0.662 | 0.455 | 0.492 | 0.622 | 0.726 |
| without | ctcf_h3k27ac | 0.624 | 0.460 | 0.518 | 0.604 | 0.639 |
| without | ctcf_rad21 | 0.534 | 0.289 | 0.338 | 0.542 | 0.674 |
| without | none | 0.082 | 0.128 | 0.114 | 0.116 | 0.175 |
| without | seven | 0.675 | 0.461 | 0.559 | 0.609 | 0.686 |
| with | all13 | 0.906 | 0.809 | 0.781 | 0.941 | 0.947 |
| with | ctcf_h3k27ac | 0.888 | 0.813 | 0.780 | 0.925 | 0.931 |
| with | ctcf_rad21 | 0.873 | 0.803 | 0.762 | 0.863 | 0.913 |
| with | none | 0.854 | 0.804 | 0.760 | 0.846 | 0.898 |
| with | seven | 0.897 | 0.814 | 0.784 | 0.935 | 0.933 |

**Named sets, 25 kb**

| cons | set_name | k562 | hmec | huvec | gm12878 | imr90 |
|---|---|---|---|---|---|---|
| without | all13 | 0.682 | 0.471 | 0.592 | 0.574 | 0.725 |
| without | ctcf_h3k27ac | 0.627 | 0.478 | 0.619 | 0.616 | 0.671 |
| without | ctcf_rad21 | 0.551 | 0.267 | 0.457 | 0.545 | 0.634 |
| without | none | 0.116 | 0.178 | 0.159 | 0.093 | 0.202 |
| without | seven | 0.678 | 0.472 | 0.645 | 0.575 | 0.695 |
| with | all13 | 0.933 | 0.893 | 0.880 | 0.938 | 0.955 |
| with | ctcf_h3k27ac | 0.909 | 0.887 | 0.883 | 0.919 | 0.936 |
| with | ctcf_rad21 | 0.893 | 0.881 | 0.868 | 0.869 | 0.927 |
| with | none | 0.876 | 0.886 | 0.867 | 0.857 | 0.909 |
| with | seven | 0.919 | 0.890 | 0.886 | 0.935 | 0.941 |

## Single marks and removed marks

The complete tables (every mark, every cell, 25 and 10 kb, with and without consensus) are in `tables.md`; `summary.csv` holds the quartiles. Held-out mean over K562, HMEC and HUVEC, 10 kb (a single RAD21 or SMC3 in HMEC and HUVEC equals no marks):

| rank | single mark alone, with consensus | single mark alone, without consensus | removed mark, without consensus (lowest first) |
|---|---|---|---|
| 1 | H3K9ac 0.825 | CTCF 0.388 | CTCF 0.435 |
| 2 | H3K27ac 0.825 | H3K27ac 0.312 | H3K4me1 0.515 |
| 3 | H3K4me2 0.823 | H3K9ac 0.300 | RAD21 0.528 |
| 4 | H3K4me1 0.822 | H3K4me2 0.298 | H3K36me3 0.532 |
| last | H3K9me3 0.805 | H3K4me3 0.159 | H3K4me3 0.554 |

All 13 marks: 0.832 (with consensus) and 0.536 (without) as held-out means at 10 kb. In GM12878 and IMR-90 RAD21 is the strongest single mark without consensus (0.442 and 0.587), in K562 CTCF is (0.536, RAD21 0.442).

## Files

`summary.csv` (one row per cell, resolution, set, with or without consensus; columns cell, res, set_kind, set_name, marks_present, n_marks, n_present, with_reference, n_regions, median_r, q25, q75, mean_r, consensus_median_r), `per_region.csv` and `per_region_random.csv.gz` (r per region), `curve.csv` (median, minimum, maximum over draws per number of marks), `tables.md` (all tables), `subsets.json`. Figure panels: `~/src/manuscripts/hic-multiome-natcomms/figures/scripts/fig_flexible.py` (flex_a to flex_d).
