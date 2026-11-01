# Methylation as a proxy for single missing marks, release candidate all12one_fullrd (registry section 97)

Question: for which individual mark can the methylation track (Meth) act as a proxy when that mark is missing? Model: `all12one_fullrd` (views at 10 and 25 kb, section 92), 413 Akita test regions, two uses (consensus plane as input; plane at 0 = marks-only use), cells as in section 95: seven WGBS training cells (GM12878, IMR-90, HepG2, A549, H1, GM23248, CD14 monocytes), three RRBS training cells (in the CSV, not in the tables below) and the held-out K562 and HMEC (deep map). Script `scripts/meth_proxy_release.py --sets minus alone` (stages predict, summary_single), tables by `scripts/meth_proxy_per_mark_report.py`; one pass of 104 keep sets per view and cell with `--set-batch 8`, agreement with the unbatched group pass of section 95 checked on minus CTCF (m1_CTCF against minus_CTCF, GM12878, 10 kb, all four states): maximal absolute difference of the per-region Pearson 9e-5.

Definitions (section 78): loss = median Pearson of the full set (13 marks, no Meth) minus that of the set; gain = median Pearson with Meth minus without, for the same set; RF = gain / loss. Median over the 413 regions per cell, mean over the cells of the group; 95 percent intervals from 500 bootstrap draws of regions (paired across sets) and, with at least three cells, of cells. Views: "mark removed" = all 13 marks except that one; "mark alone" = only that mark (loss against the full set). RF is n/d where the mean loss is below the seed noise (0.0062 for the consensus plane; 0.0062 at 10 kb and 0.0072 at 25 kb for the plane at 0) or the lower bound of the loss is not above 0. Held-out has two cells, so its intervals cover regions only plus two cells (no real cell-level coverage).

Files: `METH_PROXY_PER_MARK.csv` (columns res_kb, use, cell_group, mark, loss, loss_lo, loss_hi, gain, gain_lo, gain_hi, rf, rf_lo, rf_hi, n_cells, view, noise, n_cells_rf_defined; Pearson; cell groups held-out, training WGBS, training RRBS, training all; view = minus or alone), `METH_PROXY_PER_MARK_all_metrics.csv` (the same with insulation, boundary F1, loop enrichment, compartment, SCC), `METH_PROXY_PER_MARK_by_cell.csv` (per cell). Raw predictions: `/mnt/processed/wolffjoa/hic-multiome/runs/meth_proxy/all12one_fullrd_{10,25}kb/{m1_<mark>,o1_<mark>}[+meth][_noref]`.

## Reading

1. Removing a single mark costs almost nothing in this model, so there is almost nothing to replace. With the consensus plane no mark has a loss above the noise in any cell group at either size (largest 0.004, RRBS cells); RF is n/d for all 13 marks. With the plane at 0 only CTCF and, at 10 kb, RAD21 lose more than the noise. Ordered by RF where defined (10 kb, plane at 0): RAD21 RF 0.79 (0.15 to 2.52) held-out, 0.62 (0.17 to 1.68) training WGBS, from a loss of 0.009 to 0.010 that Meth returns by 0.006 to 0.007; CTCF RF 0.10 (-0.16 to 0.24) held-out, loss 0.043, and 0.25 (0.01 to 0.91) training WGBS, loss 0.017. At 25 kb: CTCF 0.18 (-0.05 to 0.31) held-out (loss 0.056) and 0.25 (0.05 to 0.92) training; in the training cells also H3K4me1 1.02 (0.10 to 3.54) and RAD21 1.08 (0.21 to 5.86) from losses of 0.007 and 0.009, which are within 0.002 of the noise and have intervals that reach 0.1 and 0.2. The gain from Meth is 0.001 to 0.011 for every mark whether its removal costs anything or not, so the fraction of an almost-zero loss is not stable.
2. Methylation is therefore a partial proxy for RAD21 (cohesin) when the consensus is absent and a weak one for CTCF (a quarter of a larger loss in training cells, about a tenth to a fifth in the held-out cells); histone marks removed singly are not missed by the model and need no proxy. This agrees with the group result of section 95 (cohesin pair RF 0.84 to 0.89 with the consensus, CTCF not replaced in held-out cells).
3. The "mark alone" view gives the informative ranking, because there the loss is large enough for a fraction. Consensus plane, 10 kb, held-out cells: RF 0.53 (0.36 to 0.69) for H4K20me1, 0.52 (0.36 to 0.63) H3K9me3, 0.50 (0.28 to 0.71) H3K4me1, 0.47 H3K4me2, 0.46 RAD21, 0.44 H3K4me3, 0.41 SMC3, 0.38 H3K9ac, 0.37 H3K27me3, 0.25 CTCF, 0.22 H3K27ac, 0.07 H3K79me2, 0.04 H3K36me3; at 25 kb H3K4me1 and H4K20me1 0.75, RAD21 0.64, H3K9me3 0.61, and again H3K36me3 (0.21) and H3K79me2 (0.16) last. In the training WGBS cells (losses of 0.03 to 0.07, gains 0.010 to 0.020) the order differs: H3K4me1, H3K4me2, H3K27ac (0.41 to 0.52 at 10 and 25 kb) and H3K36me3 (0.33 to 0.34) lead, the cohesin and repressive marks are lowest (0.13 to 0.18). The two cell groups agree on the leading position of H3K4me1 and on H3K4me2, and disagree on H3K36me3 and H3K79me2 (lowest in the held-out cells, mid-range in training cells), so these two rankings should not be merged; with two held-out cells the held-out order is also uncertain (intervals of 0.3 to 0.5 width).
4. Plane at 0 (the marks-only use), mark alone, 10 kb, held-out: the best single mark is CTCF (loss 0.084, gain 0.030, RF 0.35, 0.14 to 0.47), then RAD21 and SMC3 (0.30 each), H4K20me1, H3K4me3, H3K9me3 (0.26), H3K79me2 0.18, H3K4me2 0.17, H3K27me3 0.13, H3K27ac 0.10, H3K4me1 0.06, H3K36me3 and H3K9ac 0.03 (intervals include 0). Training WGBS cells: all RF between 0.13 and 0.27, flat (H4K20me1 0.27, CTCF and H3K9me3 0.25, H3K36me3 0.23, lowest H3K27me3 0.13), with gains of 0.023 to 0.059. At 25 kb held-out CTCF, RAD21 and H4K20me1 0.37, H3K9me3 0.34, down to H3K4me1 0.14. Methylation helps most when it is added to the architectural marks and least to the active promoter and enhancer marks, whose loss against the full set is the smallest (0.11 to 0.14) because they already carry the correlation 0.25 to 0.47 that section 87 found; the gain is largest for the marks with the largest loss.
5. Summary ordered by evidence: (i) no single missing mark needs a proxy when the consensus is present; (ii) without the consensus only RAD21 (RF 0.6 to 0.8 at 10 kb, 1.1 at 25 kb in training cells) and CTCF (0.1 to 0.25) lose enough to define a fraction, and Meth returns at most about one fifth to four fifths of that loss; (iii) when a single mark is all that is available, Meth returns a third to a half of the loss with the consensus (held-out: H4K20me1, H3K9me3, H3K4me1) and a fifth to a third without it (CTCF, cohesin, H4K20me1); (iv) histone marks that Meth predicts best at the signal level in section 78 (H3K4me2, H3K9ac, H3K36me3, H3K27ac) do not have the highest Hi-C RF, as in section 78, with the exception of H3K4me1/me2 with the consensus.

Limits: one checkpoint (no seed spread beyond the family noise of section 78), two held-out cells, RF of an unbalanced pair of small numbers near the noise (the CIs of the removal view reach beyond 1), three RRBS cells are in the CSV only and gain nothing measurable (section 95).

## Tables

### mark removed from the full set of 13, 10 kb, consensus plane

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| CTCF | 0.003 [0.001, 0.006] | 0.001 [-0.001, 0.004] | n/d | 0.002 [-0.000, 0.004] | 0.001 [0.000, 0.004] | n/d |
| H3K27ac | 0.001 [-0.000, 0.004] | 0.002 [-0.000, 0.004] | n/d | 0.002 [0.001, 0.005] | 0.001 [-0.000, 0.003] | n/d |
| SMC3 | 0.001 [-0.001, 0.002] | 0.004 [0.001, 0.006] | n/d | 0.002 [0.000, 0.004] | 0.002 [0.000, 0.004] | n/d |
| H3K79me2 | 0.000 [-0.001, 0.002] | 0.002 [0.000, 0.004] | n/d | -0.000 [-0.001, 0.001] | 0.001 [-0.000, 0.003] | n/d |
| H3K9ac | 0.000 [-0.001, 0.002] | 0.002 [0.000, 0.005] | n/d | 0.001 [-0.000, 0.003] | 0.002 [-0.000, 0.004] | n/d |
| H3K9me3 | -0.000 [-0.002, 0.003] | 0.002 [-0.000, 0.004] | n/d | 0.001 [-0.000, 0.003] | 0.002 [0.000, 0.003] | n/d |
| H3K4me2 | -0.000 [-0.001, 0.002] | 0.001 [-0.000, 0.004] | n/d | 0.001 [-0.001, 0.003] | 0.001 [0.000, 0.003] | n/d |
| H3K27me3 | -0.000 [-0.001, 0.002] | 0.002 [-0.000, 0.004] | n/d | 0.001 [-0.000, 0.002] | 0.002 [-0.000, 0.003] | n/d |
| H4K20me1 | -0.000 [-0.001, 0.002] | 0.000 [-0.002, 0.002] | n/d | 0.000 [-0.000, 0.003] | 0.002 [-0.000, 0.003] | n/d |
| RAD21 | -0.001 [-0.002, 0.001] | 0.002 [-0.000, 0.004] | n/d | 0.002 [0.000, 0.006] | 0.003 [0.000, 0.006] | n/d |
| H3K4me3 | -0.001 [-0.003, 0.000] | 0.001 [-0.001, 0.003] | n/d | 0.000 [-0.001, 0.002] | 0.001 [-0.000, 0.003] | n/d |
| H3K36me3 | -0.001 [-0.002, 0.001] | 0.002 [0.000, 0.005] | n/d | 0.000 [-0.001, 0.002] | 0.001 [-0.000, 0.003] | n/d |
| H3K4me1 | -0.002 [-0.003, 0.002] | 0.002 [-0.001, 0.004] | n/d | 0.001 [-0.001, 0.004] | 0.002 [-0.000, 0.003] | n/d |

### mark removed from the full set of 13, 10 kb, plane at 0

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| RAD21 | 0.009 [0.002, 0.013] | 0.007 [0.001, 0.013] | 0.79 (0.15 to 2.52) | 0.010 [0.003, 0.018] | 0.006 [0.001, 0.012] | 0.62 (0.17 to 1.68) |
| CTCF | 0.043 [0.028, 0.051] | 0.004 [-0.006, 0.010] | 0.10 (-0.16 to 0.24) | 0.017 [0.007, 0.031] | 0.004 [0.000, 0.010] | 0.25 (0.01 to 0.91) |
| H3K9me3 | 0.006 [-0.002, 0.010] | 0.004 [-0.001, 0.010] | n/d | 0.004 [-0.000, 0.008] | 0.004 [0.000, 0.008] | n/d |
| H3K79me2 | 0.003 [-0.002, 0.007] | 0.005 [-0.001, 0.010] | n/d | 0.002 [-0.002, 0.005] | 0.005 [0.000, 0.009] | n/d |
| H3K4me2 | 0.003 [-0.003, 0.007] | 0.003 [-0.002, 0.009] | n/d | 0.002 [-0.002, 0.006] | 0.004 [-0.000, 0.009] | n/d |
| H3K4me1 | 0.002 [-0.006, 0.008] | 0.001 [-0.004, 0.008] | n/d | 0.005 [-0.000, 0.011] | 0.004 [-0.000, 0.009] | n/d |
| H3K27ac | 0.001 [-0.005, 0.007] | 0.003 [-0.004, 0.009] | n/d | 0.004 [-0.001, 0.009] | 0.005 [-0.000, 0.009] | n/d |
| H4K20me1 | 0.001 [-0.006, 0.006] | 0.006 [-0.001, 0.010] | n/d | 0.002 [-0.002, 0.005] | 0.003 [-0.000, 0.007] | n/d |
| H3K4me3 | -0.001 [-0.007, 0.003] | 0.002 [-0.002, 0.009] | n/d | 0.001 [-0.003, 0.005] | 0.004 [0.000, 0.009] | n/d |
| H3K9ac | -0.001 [-0.007, 0.003] | 0.006 [-0.001, 0.011] | n/d | 0.004 [-0.000, 0.009] | 0.002 [-0.001, 0.008] | n/d |
| SMC3 | -0.001 [-0.005, 0.004] | 0.004 [-0.001, 0.011] | n/d | 0.002 [-0.002, 0.006] | 0.005 [-0.001, 0.009] | n/d |
| H3K36me3 | -0.001 [-0.009, 0.004] | 0.008 [0.002, 0.014] | n/d | 0.005 [0.001, 0.012] | 0.004 [-0.001, 0.008] | n/d |
| H3K27me3 | -0.002 [-0.006, 0.004] | 0.004 [-0.001, 0.011] | n/d | 0.003 [-0.001, 0.006] | 0.005 [0.001, 0.010] | n/d |

### mark removed from the full set of 13, 25 kb, consensus plane

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| H3K79me2 | 0.001 [-0.000, 0.003] | 0.004 [0.001, 0.007] | n/d | 0.000 [-0.001, 0.002] | 0.003 [0.001, 0.005] | n/d |
| H4K20me1 | 0.001 [-0.001, 0.003] | 0.003 [0.000, 0.005] | n/d | -0.000 [-0.001, 0.002] | 0.001 [-0.000, 0.003] | n/d |
| H3K27me3 | 0.001 [-0.001, 0.002] | 0.003 [0.001, 0.005] | n/d | 0.001 [-0.001, 0.003] | 0.003 [0.000, 0.005] | n/d |
| H3K9me3 | 0.001 [-0.001, 0.003] | 0.002 [-0.000, 0.004] | n/d | 0.001 [-0.001, 0.002] | 0.002 [-0.000, 0.004] | n/d |
| CTCF | 0.001 [-0.002, 0.004] | 0.001 [-0.001, 0.003] | n/d | 0.002 [-0.001, 0.005] | 0.002 [0.000, 0.004] | n/d |
| SMC3 | 0.000 [-0.002, 0.002] | 0.004 [0.001, 0.006] | n/d | 0.002 [0.000, 0.005] | 0.003 [0.001, 0.005] | n/d |
| H3K27ac | 0.000 [-0.002, 0.002] | 0.002 [-0.000, 0.004] | n/d | 0.002 [-0.000, 0.005] | 0.003 [-0.000, 0.005] | n/d |
| H3K9ac | 0.000 [-0.002, 0.001] | 0.003 [0.000, 0.005] | n/d | 0.001 [-0.001, 0.003] | 0.002 [0.000, 0.004] | n/d |
| H3K4me2 | -0.000 [-0.002, 0.002] | 0.001 [0.000, 0.004] | n/d | -0.001 [-0.003, 0.002] | 0.002 [0.000, 0.004] | n/d |
| H3K36me3 | -0.000 [-0.002, 0.002] | 0.003 [0.001, 0.005] | n/d | 0.000 [-0.002, 0.002] | 0.002 [0.000, 0.004] | n/d |
| RAD21 | -0.001 [-0.002, 0.001] | 0.002 [0.000, 0.005] | n/d | 0.001 [-0.000, 0.004] | 0.003 [0.000, 0.005] | n/d |
| H3K4me1 | -0.001 [-0.003, 0.001] | 0.002 [-0.001, 0.004] | n/d | 0.001 [-0.001, 0.005] | 0.002 [-0.000, 0.004] | n/d |
| H3K4me3 | -0.002 [-0.004, -0.000] | 0.001 [-0.001, 0.004] | n/d | 0.000 [-0.001, 0.002] | 0.003 [0.000, 0.004] | n/d |

### mark removed from the full set of 13, 25 kb, plane at 0

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| CTCF | 0.056 [0.039, 0.067] | 0.010 [-0.002, 0.018] | 0.18 (-0.05 to 0.31) | 0.027 [0.013, 0.042] | 0.007 [0.001, 0.015] | 0.25 (0.05 to 0.92) |
| H3K9me3 | 0.010 [-0.000, 0.016] | 0.011 [-0.002, 0.016] | n/d | 0.003 [-0.001, 0.009] | 0.005 [-0.000, 0.013] | n/d |
| H3K4me1 | 0.005 [-0.005, 0.013] | 0.006 [-0.003, 0.015] | n/d | 0.007 [0.001, 0.012] | 0.008 [0.001, 0.015] | 1.02 (0.10 to 3.54) |
| H3K4me2 | 0.004 [-0.005, 0.007] | 0.008 [-0.002, 0.015] | n/d | -0.001 [-0.004, 0.005] | 0.005 [0.001, 0.014] | n/d |
| H3K27me3 | 0.003 [-0.002, 0.010] | 0.006 [-0.001, 0.017] | n/d | 0.004 [-0.003, 0.007] | 0.009 [0.002, 0.016] | n/d |
| H3K9ac | 0.003 [-0.006, 0.007] | 0.006 [-0.001, 0.015] | n/d | 0.003 [-0.003, 0.009] | 0.007 [-0.000, 0.014] | n/d |
| RAD21 | 0.002 [-0.004, 0.010] | 0.003 [-0.004, 0.014] | n/d | 0.009 [0.001, 0.018] | 0.010 [0.002, 0.016] | 1.08 (0.21 to 5.86) |
| H3K27ac | 0.001 [-0.007, 0.005] | 0.007 [-0.003, 0.014] | n/d | 0.003 [-0.003, 0.008] | 0.009 [0.001, 0.016] | n/d |
| H4K20me1 | 0.001 [-0.008, 0.007] | 0.006 [-0.003, 0.013] | n/d | 0.002 [-0.004, 0.007] | 0.007 [0.001, 0.013] | n/d |
| H3K79me2 | 0.001 [-0.005, 0.007] | 0.006 [-0.004, 0.014] | n/d | 0.001 [-0.004, 0.005] | 0.006 [0.001, 0.014] | n/d |
| H3K4me3 | 0.000 [-0.007, 0.007] | 0.005 [-0.004, 0.014] | n/d | 0.002 [-0.002, 0.006] | 0.007 [0.000, 0.012] | n/d |
| SMC3 | -0.000 [-0.008, 0.005] | 0.003 [-0.005, 0.012] | n/d | -0.000 [-0.005, 0.003] | 0.007 [0.002, 0.014] | n/d |
| H3K36me3 | -0.003 [-0.013, 0.006] | -0.001 [-0.007, 0.010] | n/d | 0.006 [-0.000, 0.013] | 0.006 [0.000, 0.013] | n/d |

### mark alone, 10 kb, consensus plane

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| H4K20me1 | 0.021 [0.016, 0.029] | 0.011 [0.007, 0.018] | 0.53 (0.36 to 0.69) | 0.062 [0.030, 0.116] | 0.018 [0.009, 0.026] | 0.28 (0.15 to 0.49) |
| H3K9me3 | 0.026 [0.021, 0.031] | 0.014 [0.008, 0.017] | 0.52 (0.36 to 0.63) | 0.069 [0.029, 0.132] | 0.015 [0.007, 0.025] | 0.22 (0.10 to 0.47) |
| H3K4me1 | 0.014 [0.010, 0.017] | 0.007 [0.004, 0.010] | 0.50 (0.28 to 0.71) | 0.033 [0.016, 0.061] | 0.016 [0.005, 0.035] | 0.47 (0.30 to 0.60) |
| H3K4me2 | 0.008 [0.005, 0.013] | 0.004 [0.000, 0.008] | 0.47 (0.04 to 0.82) | 0.040 [0.020, 0.075] | 0.019 [0.006, 0.040] | 0.47 (0.28 to 0.58) |
| RAD21 | 0.023 [0.019, 0.029] | 0.011 [0.006, 0.015] | 0.46 (0.28 to 0.59) | 0.064 [0.023, 0.131] | 0.011 [0.004, 0.021] | 0.18 (0.04 to 0.50) |
| H3K4me3 | 0.025 [0.019, 0.030] | 0.011 [0.006, 0.016] | 0.44 (0.28 to 0.60) | 0.065 [0.029, 0.122] | 0.018 [0.006, 0.028] | 0.28 (0.14 to 0.49) |
| SMC3 | 0.025 [0.020, 0.032] | 0.010 [0.005, 0.015] | 0.41 (0.23 to 0.52) | 0.065 [0.026, 0.132] | 0.010 [0.002, 0.018] | 0.15 (0.02 to 0.44) |
| H3K9ac | 0.009 [0.006, 0.014] | 0.003 [-0.001, 0.007] | 0.38 (-0.12 to 0.64) | 0.052 [0.020, 0.104] | 0.011 [0.004, 0.019] | 0.21 (0.14 to 0.38) |
| H3K27me3 | 0.020 [0.014, 0.024] | 0.007 [0.001, 0.010] | 0.37 (0.05 to 0.49) | 0.068 [0.027, 0.133] | 0.009 [0.003, 0.015] | 0.13 (0.04 to 0.31) |
| CTCF | 0.021 [0.016, 0.026] | 0.005 [0.000, 0.009] | 0.25 (0.02 to 0.40) | 0.068 [0.028, 0.129] | 0.015 [0.006, 0.025] | 0.22 (0.11 to 0.43) |
| H3K27ac | 0.007 [0.004, 0.011] | 0.002 [-0.001, 0.005] | 0.22 (-0.27 to 0.68) | 0.037 [0.014, 0.078] | 0.015 [0.004, 0.035] | 0.41 (0.24 to 0.49) |
| H3K79me2 | 0.020 [0.016, 0.025] | 0.001 [-0.003, 0.006] | 0.07 (-0.14 to 0.26) | 0.063 [0.029, 0.123] | 0.014 [0.005, 0.025] | 0.23 (0.13 to 0.34) |
| H3K36me3 | 0.017 [0.014, 0.024] | 0.001 [-0.003, 0.006] | 0.04 (-0.16 to 0.28) | 0.060 [0.030, 0.104] | 0.020 [0.007, 0.036] | 0.34 (0.22 to 0.40) |

### mark alone, 10 kb, plane at 0

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| CTCF | 0.084 [0.064, 0.098] | 0.030 [0.010, 0.041] | 0.35 (0.14 to 0.47) | 0.173 [0.120, 0.245] | 0.044 [0.020, 0.065] | 0.25 (0.12 to 0.40) |
| RAD21 | 0.149 [0.132, 0.168] | 0.045 [0.026, 0.063] | 0.30 (0.18 to 0.41) | 0.181 [0.111, 0.272] | 0.038 [0.018, 0.058] | 0.21 (0.09 to 0.38) |
| SMC3 | 0.173 [0.153, 0.190] | 0.051 [0.033, 0.068] | 0.30 (0.20 to 0.39) | 0.200 [0.145, 0.302] | 0.035 [0.017, 0.059] | 0.18 (0.07 to 0.35) |
| H4K20me1 | 0.165 [0.141, 0.188] | 0.044 [0.032, 0.066] | 0.26 (0.21 to 0.39) | 0.215 [0.162, 0.278] | 0.059 [0.034, 0.079] | 0.27 (0.16 to 0.38) |
| H3K4me3 | 0.185 [0.166, 0.203] | 0.048 [0.034, 0.066] | 0.26 (0.19 to 0.36) | 0.220 [0.155, 0.297] | 0.047 [0.023, 0.072] | 0.21 (0.11 to 0.33) |
| H3K9me3 | 0.167 [0.148, 0.182] | 0.043 [0.027, 0.060] | 0.26 (0.17 to 0.36) | 0.225 [0.174, 0.296] | 0.057 [0.037, 0.081] | 0.25 (0.15 to 0.39) |
| H3K79me2 | 0.162 [0.143, 0.188] | 0.029 [0.013, 0.049] | 0.18 (0.08 to 0.28) | 0.190 [0.136, 0.257] | 0.037 [0.016, 0.054] | 0.20 (0.10 to 0.28) |
| H3K4me2 | 0.132 [0.117, 0.146] | 0.023 [0.012, 0.040] | 0.17 (0.09 to 0.29) | 0.154 [0.118, 0.193] | 0.034 [0.016, 0.060] | 0.22 (0.12 to 0.33) |
| H3K27me3 | 0.151 [0.134, 0.176] | 0.020 [0.008, 0.038] | 0.13 (0.06 to 0.24) | 0.214 [0.150, 0.294] | 0.028 [0.013, 0.047] | 0.13 (0.05 to 0.26) |
| H3K27ac | 0.128 [0.110, 0.141] | 0.013 [0.000, 0.022] | 0.10 (0.00 to 0.17) | 0.148 [0.112, 0.194] | 0.023 [0.007, 0.042] | 0.15 (0.06 to 0.23) |
| H3K4me1 | 0.124 [0.105, 0.145] | 0.007 [-0.007, 0.023] | 0.06 (-0.07 to 0.17) | 0.141 [0.109, 0.174] | 0.027 [0.012, 0.048] | 0.19 (0.09 to 0.31) |
| H3K36me3 | 0.139 [0.121, 0.159] | 0.004 [-0.007, 0.021] | 0.03 (-0.05 to 0.14) | 0.192 [0.153, 0.234] | 0.044 [0.028, 0.059] | 0.23 (0.16 to 0.32) |
| H3K9ac | 0.130 [0.114, 0.150] | 0.004 [-0.008, 0.021] | 0.03 (-0.07 to 0.16) | 0.176 [0.130, 0.224] | 0.030 [0.014, 0.050] | 0.17 (0.09 to 0.29) |

### mark alone, 25 kb, consensus plane

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| H3K4me1 | 0.010 [0.005, 0.015] | 0.008 [0.002, 0.011] | 0.75 (0.28 to 1.03) | 0.032 [0.012, 0.065] | 0.017 [0.006, 0.037] | 0.52 (0.33 to 0.64) |
| H4K20me1 | 0.018 [0.013, 0.024] | 0.013 [0.009, 0.019] | 0.75 (0.51 to 0.96) | 0.058 [0.024, 0.113] | 0.017 [0.009, 0.026] | 0.29 (0.16 to 0.55) |
| RAD21 | 0.019 [0.013, 0.024] | 0.012 [0.006, 0.016] | 0.64 (0.39 to 0.86) | 0.062 [0.020, 0.134] | 0.010 [0.002, 0.020] | 0.17 (0.02 to 0.56) |
| H3K9me3 | 0.021 [0.015, 0.028] | 0.013 [0.008, 0.020] | 0.61 (0.42 to 0.86) | 0.066 [0.027, 0.130] | 0.016 [0.008, 0.026] | 0.25 (0.12 to 0.55) |
| H3K4me3 | 0.018 [0.013, 0.026] | 0.010 [0.006, 0.017] | 0.56 (0.35 to 0.77) | 0.061 [0.023, 0.123] | 0.016 [0.007, 0.027] | 0.27 (0.16 to 0.51) |
| CTCF | 0.018 [0.013, 0.022] | 0.010 [0.005, 0.014] | 0.54 (0.28 to 0.74) | 0.065 [0.024, 0.132] | 0.014 [0.006, 0.025] | 0.22 (0.08 to 0.49) |
| H3K4me2 | 0.007 [0.003, 0.011] | 0.004 [0.000, 0.007] | 0.53 (0.01 to 1.01) | 0.038 [0.014, 0.079] | 0.019 [0.005, 0.044] | 0.49 (0.26 to 0.60) |
| SMC3 | 0.021 [0.015, 0.025] | 0.011 [0.006, 0.014] | 0.52 (0.33 to 0.72) | 0.062 [0.021, 0.135] | 0.009 [0.002, 0.017] | 0.15 (0.02 to 0.52) |
| H3K27me3 | 0.017 [0.010, 0.023] | 0.008 [0.002, 0.013] | 0.49 (0.17 to 0.66) | 0.067 [0.025, 0.137] | 0.010 [0.002, 0.015] | 0.14 (0.02 to 0.37) |
| H3K27ac | 0.009 [0.005, 0.013] | 0.004 [-0.000, 0.008] | 0.47 (-0.02 to 0.88) | 0.037 [0.012, 0.086] | 0.015 [0.004, 0.038] | 0.41 (0.25 to 0.51) |
| H3K9ac | 0.008 [0.003, 0.013] | 0.002 [-0.002, 0.007] | 0.28 (-0.38 to 0.79) | 0.048 [0.017, 0.100] | 0.011 [0.003, 0.023] | 0.23 (0.14 to 0.44) |
| H3K36me3 | 0.015 [0.010, 0.019] | 0.003 [-0.001, 0.007] | 0.21 (-0.05 to 0.46) | 0.055 [0.024, 0.106] | 0.018 [0.008, 0.033] | 0.33 (0.24 to 0.43) |
| H3K79me2 | 0.016 [0.011, 0.022] | 0.003 [-0.002, 0.007] | 0.16 (-0.16 to 0.35) | 0.060 [0.022, 0.130] | 0.015 [0.004, 0.030] | 0.25 (0.13 to 0.43) |

### mark alone, 25 kb, plane at 0

| mark | held-out (2 cells) loss [CI] | gain [CI] | RF (CI) | training WGBS (7 cells) loss [CI] | gain [CI] | RF (CI) |
|---|---|---|---|---|---|---|
| CTCF | 0.086 [0.067, 0.100] | 0.032 [0.018, 0.052] | 0.37 (0.24 to 0.57) | 0.175 [0.123, 0.256] | 0.051 [0.029, 0.076] | 0.29 (0.16 to 0.47) |
| RAD21 | 0.179 [0.152, 0.204] | 0.066 [0.043, 0.085] | 0.37 (0.27 to 0.45) | 0.199 [0.127, 0.304] | 0.053 [0.028, 0.078] | 0.27 (0.13 to 0.48) |
| H4K20me1 | 0.169 [0.132, 0.188] | 0.062 [0.035, 0.075] | 0.37 (0.23 to 0.44) | 0.216 [0.163, 0.282] | 0.060 [0.036, 0.082] | 0.28 (0.15 to 0.41) |
| H3K9me3 | 0.180 [0.152, 0.208] | 0.062 [0.041, 0.084] | 0.34 (0.26 to 0.43) | 0.224 [0.169, 0.300] | 0.058 [0.028, 0.091] | 0.26 (0.11 to 0.43) |
| SMC3 | 0.188 [0.158, 0.209] | 0.064 [0.041, 0.082] | 0.34 (0.24 to 0.42) | 0.228 [0.164, 0.327] | 0.057 [0.033, 0.081] | 0.25 (0.12 to 0.43) |
| H3K4me3 | 0.195 [0.162, 0.220] | 0.061 [0.036, 0.081] | 0.32 (0.21 to 0.40) | 0.213 [0.153, 0.294] | 0.050 [0.025, 0.075] | 0.24 (0.12 to 0.38) |
| H3K79me2 | 0.173 [0.143, 0.191] | 0.045 [0.017, 0.059] | 0.26 (0.11 to 0.34) | 0.199 [0.152, 0.270] | 0.051 [0.031, 0.067] | 0.26 (0.15 to 0.38) |
| H3K4me2 | 0.137 [0.108, 0.153] | 0.029 [0.005, 0.040] | 0.21 (0.05 to 0.29) | 0.153 [0.116, 0.192] | 0.036 [0.020, 0.057] | 0.24 (0.13 to 0.36) |
| H3K27ac | 0.140 [0.114, 0.154] | 0.029 [0.006, 0.043] | 0.21 (0.05 to 0.30) | 0.145 [0.107, 0.186] | 0.031 [0.015, 0.049] | 0.21 (0.11 to 0.31) |
| H3K27me3 | 0.159 [0.128, 0.182] | 0.032 [0.012, 0.054] | 0.20 (0.08 to 0.33) | 0.213 [0.157, 0.302] | 0.041 [0.021, 0.060] | 0.19 (0.08 to 0.34) |
| H3K9ac | 0.144 [0.125, 0.170] | 0.024 [0.007, 0.045] | 0.17 (0.05 to 0.30) | 0.177 [0.131, 0.236] | 0.037 [0.017, 0.059] | 0.21 (0.10 to 0.33) |
| H3K4me1 | 0.110 [0.092, 0.133] | 0.016 [-0.002, 0.030] | 0.14 (-0.02 to 0.27) | 0.134 [0.097, 0.170] | 0.034 [0.019, 0.050] | 0.26 (0.15 to 0.36) |
| H3K36me3 | 0.140 [0.119, 0.159] | 0.009 [-0.003, 0.029] | 0.07 (-0.02 to 0.20) | 0.185 [0.136, 0.236] | 0.044 [0.023, 0.062] | 0.24 (0.14 to 0.37) |

