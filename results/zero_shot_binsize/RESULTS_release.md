# Untrained bin sizes: the release candidate all12one_fullrd

Registry section 102. `summary.csv` in this folder has the layout of `../summary.csv` (section 89), so a figure reads either by changing one path. Produced by `scripts/zero_shot_binsize.py --stage config precompute score summary --model all12one_fullrd`.

Protocol. As in `../RESULTS.md`: GM12878 (leave-one-out consensus) and K562 (12-cell consensus), 80 Akita test regions on chr2, chr3 and chr10 (62 at 1 kb, chr2 and chr10), maps, consensus cools, normaliser variants (`nearest`, `refit_tracks`, `refit_all`) and windows (1024, 512, 96, 48 bins at 1, 4, 20, 50 kb) are the files of section 89, scored with `scripts/crosscell_predict.py --tta`. The checkpoint is the release candidate `all12one_fullrd` (13 marks plus Meth, cohesin-fraction row, reference dropout 0.3, trained on all chromosomes) with the consensus as reference. The Meth track is given to both cells (`meth/gm12878`, `meth/k562`; 14 of 14 marks read). The trained-size rows are the 80 regions of the 413-region tables of `all12one_fullrd_<k>kb/crosscell`. The consensus baseline column is identical to section 89 (same cools). The 1 kb consensus remains reduced (4 and 5 cells, see `../RESULTS.md`).

## Median r, nearest-size normalisers, against the consensus alone

| bin size (kb) | GM12878 all12one_fullrd | GM12878 all12one (89) | GM12878 consensus | K562 all12one_fullrd | K562 all12one (89) | K562 consensus |
|---|---|---|---|---|---|---|
| 1 | 0.586 | 0.587 | 0.500 | 0.451 | 0.462 | 0.416 |
| 2 (trained) | 0.767 | 0.772 | 0.652 | 0.649 | 0.638 | 0.600 |
| 4 | 0.878 | 0.889 | 0.774 | 0.789 | 0.798 | 0.740 |
| 5 (trained) | 0.908 | 0.916 | 0.794 | 0.831 | 0.826 | 0.777 |
| 10 (trained) | 0.934 | 0.935 | 0.833 | 0.902 | 0.907 | 0.858 |
| 20 | 0.935 | 0.932 | 0.850 | 0.922 | 0.933 | 0.894 |
| 25 (trained) | 0.937 | 0.935 | 0.851 | 0.927 | 0.940 | 0.899 |
| 50 | 0.913 | 0.896 | 0.881 | 0.921 | 0.923 | 0.910 |

## Reading

Distance to the consensus alone. At every untrained size and for both cells all12one_fullrd is above the consensus: GM12878 +0.086 (1 kb), +0.104 (4 kb), +0.085 (20 kb), +0.032 (50 kb); K562 +0.035, +0.049, +0.029, +0.012. At the trained sizes the gain is +0.115 to +0.113 (GM12878, 2 and 5 kb) and +0.049 to +0.053 (K562), +0.101 and +0.044 at 10 kb, +0.086 and +0.028 at 25 kb. The untrained sizes inside the trained range (4 and 20 kb) keep 0.9 to 1.0 of the gain of the neighbouring trained size for GM12878 (0.104 against 0.113 at 5 kb, 0.085 against 0.086 at 25 kb) and for K562 (0.049 against 0.053, 0.029 against 0.028); at 50 kb the gain shrinks to +0.032 and +0.012, because the consensus alone is already close to the measured map (0.881, 0.910). Section 89 gave the same ordering for all12one (gains +0.015 and +0.013 at 50 kb).

Distance to the trained level (same model, nearest trained size, same regions). 4 kb is 0.030 (GM12878) and 0.041 (K562) below 5 kb, 20 kb is 0.002 and 0.005 below 25 kb, 50 kb is 0.024 and 0.006 below 25 kb, 1 kb is 0.171 and 0.171 below 2 kb (section 89 for all12one: 0.026 / 0.028, 0.003 / 0.006, 0.039 / 0.017, 0.180 / 0.155).

Against all12one (section 89). On the untrained sizes the median r of all12one_fullrd differs from that of all12one by -0.011 to +0.017 with the nearest-size normalisers (GM12878 -0.001, -0.011, +0.003, +0.017 at 1, 4, 20, 50 kb; K562 -0.011, -0.009, -0.011, -0.001), the largest gain being GM12878 at 50 kb. At the trained sizes the differences are -0.012 to +0.011 (GM12878 -0.005, -0.008, 0.000, +0.002 at 2, 5, 10, 25 kb; K562 +0.011, +0.005, -0.006, -0.012). The 80 regions carry about +-0.01 sampling noise (section 89), so the release candidate and all12one are level at untrained sizes as they are at trained sizes; all12one_fullrd is lower by 0.011 to 0.012 only for K562 at 1, 20 and 25 kb.

Refitted normalisers. Refitting the track statistics changes the median r by -0.019 to +0.011 against the nearest-size normalisers (largest change K562 1 kb -0.019; GM12878 1 kb +0.011, 20 kb -0.014, 50 kb -0.015), refitting also the contact normaliser by -0.009 to +0.011. As in section 89 there is no systematic gain, so the normalisers of the nearest trained size are used.

Limits. As in section 89: 80 regions on three chromosomes, GM12878 is a training cell, the 1 kb consensus is reduced, the maps at new bin sizes are balanced on the band-limited cis matrix. all12one_fullrd was trained on all chromosomes, so the regions on chr2, chr3 and chr10 are in its training chromosomes at the trained sizes, as for all12one in section 89; the windows at the untrained sizes were never seen. One checkpoint, one run. The `flexcible predict --bin-size` option itself was not run on real data here either; the scoring uses `crosscell_predict.py`.
