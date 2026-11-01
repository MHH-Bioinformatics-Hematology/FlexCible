# Count-space back-transform with the consensus distance decay

2026-11-01. Release twin all12one_fullrd_chrho, chr2 (held out), predicted O/E of GM12878, IMR-90 and K562 turned into
balanced counts in two ways. Scripts: `scripts/oe_to_counts_consdecay.py` (new variant), `scripts/counts_consensus_decay_eval.py`
(table). Table: `results/one_model/counts_consensus_decay.csv`.

Source of the decay. The consensus O/E cools have the per-diagonal offset removed and carry no expected(d), so the decay was
rebuilt from the measured balanced 10 kb chr2 maps of exactly the cells that form the reference of the cell
(`consensus_loo_<cell>_proper_10kb.cool.json`: the 11 other cells for GM12878 and IMR-90, consensus12 for K562; the predicted cell is
never among them): expected(d) is the geometric mean over those cells of expected_c(d), offset(d) the arithmetic mean of offset_c(d),
both from `oe_proper.stream_stats` (chromosome-wide chr2, band 200 bins), the same definitions as in `oe_to_counts.py`. The model
sees the same cells as input, so the variant uses nothing about the target cell beyond its prediction. The counts are not rescaled to
the target. Sources of the measured maps: those of `scripts/cellspecific.py` MEASURED (4DN mcools, ENCODE .hic conversions).

Region chr2:18-22 Mb, 10 kb, up to 1.5 Mb from the diagonal, pixels as in `fig_counts.py` (mappable, 2 bins or more from the diagonal,
upper triangle, measured count above 0).

| Cell | pixels | (a) measured vs prediction, measured decay: diag. removed / raw | (b) measured vs prediction, consensus decay: diag. removed / raw | (c) the two reconstructions: diag. removed / raw | log10 shift (b minus a), median (range) |
|---|---|---|---|---|---|
| GM12878 | 47,979 | 0.805 / 0.952 | 0.805 / 0.950 | 1.000 / 0.9996 | -0.144 (-0.195 to -0.035) |
| IMR-90 | 42,558 | 0.820 / 0.924 | 0.820 / 0.925 | 1.000 / 0.9998 | -0.128 (-0.173 to -0.103) |
| K562 | 42,557 | 0.681 / 0.888 | 0.681 / 0.888 | 1.000 / 0.9996 | -0.172 (-0.188 to -0.075) |

Pearson of log10 counts; "diag. removed" subtracts the per-diagonal mean of each map before the correlation. The two
reconstructions differ by a factor that depends on the diagonal only, so the correlation after removing the per-diagonal mean is
identical for (a) and (b) (differences below 1e-9) and 1 for (c); this is an identity and not a finding.

Reading. Replacing the decay of the target cell by the decay of the other cells leaves the count-space agreement after removal of the
distance decay unchanged (0.81, 0.82 and 0.68), so that figure of the manuscript does not depend on the target's own Hi-C. The raw
log10 correlation, which still contains the decay, moves by at most 0.013 (GM12878 0.952 to 0.950, IMR-90 0.924 to 0.925, K562 0.888 to
0.888) and the two reconstructions correlate at 0.9996 or higher, so the consensus decay is as close to the measured decay as the
decay of one cell is to the decay of another. What does change is the level: the consensus decay gives counts that are on average
0.13 to 0.17 in log10 (about 25 to 33 percent) lower than the measured decay, with a shift that grows with distance for GM12878
(-0.03 at 20 kb to -0.20 at 1.5 Mb) and is nearly flat for IMR-90 and K562 after 20 kb; a count-space panel with the consensus decay
therefore sits below the measured map by that factor, and its P(d) curve follows the typical decay of the other cells. The
variant removes the possible reading of leakage from the target's own decay, at the price of an absolute scale that is borrowed from
the other cells.

Cools: `/mnt/processed/wolffjoa/hic-multiome/runs/dense_region_fullrd_chrho/consensus_decay/{gm12878,imr90,k562}_counts_chr2_consdecay.cool`
(chr2, 4,355,622 pixels each, same pixels as `<cell>_counts_chr2.cool`), logs `<cell>_counts_consdecay.log`, per-cell decay caches
`stats/<cell>_chr2_stats.npz` (expected and offset of each of the 12 source cells).
