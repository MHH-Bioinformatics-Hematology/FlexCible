# CTCF zero-signal degron on the release checkpoint, no consensus (2026-11-01)

`scripts/perturb_degron_zero.py --set release --stage predict score`: view all12one_fullrd_10kb, sample hct116 (ChIP of the
untreated cells, HCT116 methylation, fraction 1), `flexcible predict --no-reference --tta`, CTCF (or H3K36me3) raw signal set
to zero on chr1-chr21 with `--zero-region`, motifs stay. Predictions: `/mnt/processed/wolffjoa/hic-multiome/perturb/degron_zero/cool/all12one_fullrd_noref_10kb/{full,zero_CTCF,zero_H3K36me3}.cool`.
Scoring as for the dedicated models (same regions, loops, boundaries). `perturb_summary.csv`: CTCF degron, 411 regions,
536 loops.

CTCF set to zero: change-map Pearson median 0.141 [-0.056, 0.325]; loops weaker 57.5% (predicted) against 99.1% (measured),
median loop change -0.020 against -0.489. Dedicated all12honest_drop_10kb with the same input: 0.347 and 89.3%. The release
checkpoint without the consensus gives a clearly weaker response to zero signal than to a declared-absent CTCF track
(section 99: Pearson 0.380, 94.2% of the loops weaker, median -0.221). H3K36me3 zero (control): Pearson 0.032, loop change -0.005.
