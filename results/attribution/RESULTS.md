# Integrated-gradient attribution for all12one at 10 kb (K562, held out)

Script `scripts/attribute.py`; registry section 90. Model: `all12one_10kb/checkpoint-best`, cell K562 with all 13 marks, 192-bin windows (1.92 Mb) centred on the Akita test regions, no test-time augmentation. Integrated gradients (Sundararajan et al. 2017), 50 midpoint steps, of a scalar of the predicted log O/E map with respect to the 23 per-bin input rows (13 marks, GC, CpG, fwd and rev tracks of four motifs, z-scored) and the 2D consensus plane. The pairing into mean, product, |diff| and directed motif planes is inside the model (`build_spatial_cond`), so the gradients are taken with respect to the 1D rows before pairing (no 2D back-summation); the motif strand pairs are a view of the same rows. Baselines: marks and motif rows = absent-track encoding (`TrackNormalizer.absent_values()`, the z-score of a raw zero), GC and CpG = 0 in z units (genome average; a raw-zero base content is not a state the model has seen), consensus plane = 0. "Without consensus" has the plane at 0 at input and baseline, so it is not attributed. Targets: loop = sum of the prediction over the 3x3 box at the strongest measured loop pixel; border = minus the mean of [b-5,b) x (b,b+5]; global = sum over the upper triangle (k = 2). Loop pixel: maximum of (3x3 mean minus donut mean) of the measured map at 100 to 600 kb; border: maximum of 0.5 * (two intra-domain 5x5 squares) minus the cross square. Detailed tables: `tables.md`.

## Chosen regions (K562, all 13 marks with consensus, r > 0.9 in `results/flexible_inputs/per_region.csv`, picked by inspection of the top 24 by measured loop strength, `candidates.csv`)

Selection by eye: a loop dot at the corner of a domain with a sharp border and little other structure. Window bins are counted from the window start (`window_start_bp` in the npz); coordinates in hg38.

| region | r | loop anchors (bp) | span | border (bp) |
|---|---|---|---|---|
| chr3:136605696-137654272 (figure window) | 0.973 | 136,760,000 / 137,010,000 | 250 kb | coincides with the right anchor |
| chr14:24182784-25231360 | 0.963 | 24,320,000 / 24,430,000 | 110 kb | 24,440,000 |
| chr10:16289792-17338368 | 0.942 | 16,830,000 / 17,250,000 | 420 kb | 16,810,000 |

In all three windows the strongest border lies within two bins of a loop anchor (the domain corner dot), so a loop and a border share an anchor and the two targets overlap in position.

## Completeness (sum of attributions against f(x) - f(baseline))

With consensus the 50-step sum is within 5 per cent for all six targets: loop 2.1, 1.7, 1.9 per cent; border 0.7, 3.4, 2.8 per cent (chr10, chr14, chr3 order). Without consensus the sums lag more: loop 6.1, 5.0, 5.5 per cent; border 7.9, 61.9, 14.3 per cent (the border target without consensus is a small difference, f(x) - f(b) = 0.03 to 0.09, so a small absolute error is a large relative one). At 300 steps all twelve are within 2.5 per cent (`results/attribution_s300/`). The global target (whole map) at 50 steps: median relative error 8.4 per cent with and 22 per cent without consensus (30 and 12.5 per cent of the 40 regions within 5 per cent, cases with a near-zero f(x) - f(b) dominate the tail); at 300 steps median 0.3 and 0.7 per cent, 97.5 and 92.5 per cent of regions within 5 per cent. The mean shares of the 50-step and 300-step global runs correlate at r = 0.99998 and the ranking is identical, so the shares and tracks reported are insensitive to step count. With consensus 50 steps meet the 5 per cent criterion at the loci; without consensus only the 300-step run does.

## Global ranking (40 random K562 test regions, seed 0; share of the summed absolute attribution of the whole map; `ig_global.csv`)

| rank | with consensus | share | without consensus | share |
|---|---|---|---|---|
| 1 | consensus plane | 0.331 | SMC3 | 0.150 |
| 2 | SMC3 | 0.098 | RAD21 | 0.130 |
| 3 | RAD21 | 0.086 | H3K27ac | 0.113 |
| 4 | H3K27ac | 0.074 | H3K4me1 | 0.097 |
| 5 | H3K4me1 | 0.068 | CTCF | 0.076 |

Next: H3K9ac (0.052 / 0.073). GC, CpG and the motif tracks together carry about 3 per cent (CTCF motif tracks below 0.1 per cent). Ranks of an attribution are not ranks of importance by removal: by ablation CTCF is the most costly mark to remove without consensus (registry 87), but its integrated-gradient share is below SMC3, RAD21, H3K27ac and H3K4me1, since the baseline is the absence of every mark at once and attribution divides the joint effect, with the cohesin subunits and CTCF sharing the same anchor bins.

## Interpretation

Where the attribution sits. For the loop target the absolute attribution of CTCF, RAD21 and SMC3 together falls to 20 to 31 per cent within two bins of the two anchors, which are 5 per cent of the window (4 to 6 times uniform; table in `tables.md`). The active marks (H3K27ac, H3K4me1/2/3, H3K9ac) concentrate equally strongly at the anchors (17 to 28 per cent) and the repressive marks as well (17 to 22 per cent), so the anchor enrichment is a property of the pixel, not of the cohesin group alone. In the figure window the CTCF, RAD21 and SMC3 attribution forms peaks exactly at both anchor bins with the same sign, the minus-strand CTCF motif track contributes at the anchors, and the consensus attribution is a broad bump centred on the anchors. The sign of the cohesin tracks is positive at the anchors and mixed in the span between them.

Inside the domain. The active marks keep a large share between the anchors (24 to 41 per cent against 3 to 19 per cent of bins), more than the repressive marks (6 to 22 per cent) in two of the three windows; in the figure window H3K27ac and H3K4me1 add at the anchors and subtract near the left anchor, H3K27me3 adds on the right flank. For the border target the picture is a signed, two-sided pattern around the border: the cohesin and CTCF tracks and the active marks carry 10 to 23 per cent of their attribution within two bins of the border (against 2.6 per cent of bins), with H3K27ac and H3K4me1 positive on the left (loop-domain) side of the border and negative or small on the right side in the figure window.

Without the consensus. The loop attribution barely moves in its distribution (anchor shares of the cohesin group 19 to 27 per cent against 20 to 31 with consensus) but its size falls to about a third (loop f(x) - f(b) of 4.7, 2.5 and 3.3 against 14.6, 8.9 and 10.1), because the consensus plane supplies most of the predicted loop. At the border the cohesin and CTCF share near the border doubles (19 to 21 per cent against 10 to 11 per cent with consensus), the consensus taking over part of what the ChIP tracks must carry otherwise. Globally the consensus plane carries a third of all attribution and removing it moves its share onto SMC3, RAD21 and the active marks (SMC3 0.098 to 0.150, RAD21 0.086 to 0.130, H3K27ac 0.074 to 0.113).

Caveats. Three windows is a case study; the global ranking over 40 regions is the only region-level statistic. Attributions depend on the baseline (all marks absent at once), so they are not marginal effects. The 5 per cent completeness criterion is met at 50 steps by the six with-consensus loci targets and the 12 loci targets at 300 steps; the 50-step arrays in `ig_locus_*.npz` without consensus (border in particular) are less complete, the 300-step arrays are in `results/attribution_s300/`. The figure uses the 50-step arrays.
