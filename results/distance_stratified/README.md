# Distance-stratified Pearson correlation

Script: `scripts/distance_stratified.py`. Figure: `~/src/manuscripts/hic-multiome-natcomms/figures/scripts/figS_distance.py`.

## Definition
Pearson r between a source map and the measured map (distance-normalised log O/E) per bin distance (2 bins up to 2 Mb), pooled over all upper-triangle pixels of the chromosomes of the set at that distance. A pixel enters a set only when it is stored and finite in the measured map and in every source of the set (identical pixels). Distances where a source has no pixels (ours stores up to 1.91 Mb at 10 kb, Epiphany up to 0.99 Mb) are absent. Non-native grids (C.Origami 8192 bp) are read at the target bin centres (nearest native bin); Hi-cGAN (hg19) after lifting the bin centres with hg38ToHg19 (bins whose lift is not locally monotone are dropped).

## Inputs
- measured: `/mnt/processed/wolffjoa/hic-multiome/crosscell_oe/<cell>_proper_<k>kb.cool`
- consensus: `crosscell_oe/consensus_loo_<cell>_proper_<k>kb.cool` (k562: `consensus12_proper_<k>kb.cool`)
- ours: `runs/hic_measures_chrho/pred/<k>kb/all12refin_drop_chrho/<cell>.cool` (with consensus), `.../all12honest_drop_chrho/<cell>.cool` (ChIP only)
- competitors: `runs/benchmark/competitor_oe/corigami_imr90_8kb_chr{2,3}.cool`, `epiphany_gm12878_10kb_chr{2,3,11,17}.cool`, `hicgan_gm12878_10kb.cool`

## Sets (column `set` of the CSVs)
- `core`: ours, ChIP only, consensus; held-out chromosomes 2, 3, 11, 14, 17 (all cells; 10 kb and 25 kb).
- `all_sources`: core sources plus the competitors of the cell on the chromosomes the task fixed: IMR-90 chr2, chr3 (C.Origami); GM12878 chr2, 3, 11, 17 (Epiphany and Hi-cGAN, identical pixels). Distances are limited by Epiphany to <= 0.99 Mb, hence n/a at 1 Mb.
- `strict_epiphany` (GM12878 chr3, 11, 17) and `strict_hicgan` (GM12878 chr2, chr14): the same comparison restricted to chromosomes outside the competitor's own training chromosomes according to the registry (Epiphany held-out chr3, 11, 17; Hi-cGAN trained on odd chromosomes, held-out even). In `all_sources` chr2 is a training chromosome of Epiphany and chr3, 11, 17 are training chromosomes of Hi-cGAN.

## Identical pixels (`pixel_counts.csv`)
| cell | res_bp | set | chromosomes | identical_pixels |
|---|---|---|---|---|
| imr90 | 10000 | core | chr2,chr3,chr11,chr14,chr17 | 10873079 |
| imr90 | 10000 | all_sources | chr2,chr3 | 6260781 |
| imr90 | 25000 | core | chr2,chr3,chr11,chr14,chr17 | 1784835 |
| gm12878 | 10000 | core | chr2,chr3,chr11,chr14,chr17 | 10873079 |
| gm12878 | 10000 | all_sources | chr2,chr3,chr11,chr17 | 6206078 |
| gm12878 | 10000 | strict_epiphany | chr3,chr11,chr17 | 4065767 |
| gm12878 | 10000 | strict_hicgan | chr2,chr14 | 4596967 |
| gm12878 | 25000 | core | chr2,chr3,chr11,chr14,chr17 | 1784835 |
| k562 | 10000 | core | chr2,chr3,chr11,chr14,chr17 | 10873079 |
| k562 | 25000 | core | chr2,chr3,chr11,chr14,chr17 | 1784835 |

## Pearson r at 100 kb, 500 kb, 1 Mb (10 kb; `headline_10kb.csv`, full curves in `<cell>_<k>kb.csv`)
| cell | set | source | 100kb | 500kb | 1Mb |
|---|---|---|---|---|---|
| gm12878 | all_sources | chip | 0.722 | 0.74 | n/a |
| gm12878 | all_sources | consensus | 0.81 | 0.823 | n/a |
| gm12878 | all_sources | epiphany | 0.369 | 0.596 | n/a |
| gm12878 | all_sources | hicgan | 0.283 | 0.326 | n/a |
| gm12878 | all_sources | ours | 0.892 | 0.912 | n/a |
| gm12878 | core | chip | 0.713 | 0.729 | 0.728 |
| gm12878 | core | consensus | 0.811 | 0.827 | 0.817 |
| gm12878 | core | ours | 0.891 | 0.91 | 0.901 |
| gm12878 | strict_epiphany | chip | 0.719 | 0.732 | n/a |
| gm12878 | strict_epiphany | consensus | 0.828 | 0.823 | n/a |
| gm12878 | strict_epiphany | epiphany | 0.256 | 0.458 | n/a |
| gm12878 | strict_epiphany | ours | 0.899 | 0.911 | n/a |
| gm12878 | strict_hicgan | chip | 0.717 | 0.742 | 0.77 |
| gm12878 | strict_hicgan | consensus | 0.788 | 0.832 | 0.83 |
| gm12878 | strict_hicgan | hicgan | 0.223 | 0.207 | 0.221 |
| gm12878 | strict_hicgan | ours | 0.883 | 0.912 | 0.913 |
| imr90 | all_sources | chip | 0.804 | 0.79 | 0.789 |
| imr90 | all_sources | consensus | 0.847 | 0.88 | 0.866 |
| imr90 | all_sources | corigami | 0.765 | 0.863 | 0.898 |
| imr90 | all_sources | ours | 0.926 | 0.932 | 0.921 |
| imr90 | core | chip | 0.792 | 0.787 | 0.775 |
| imr90 | core | consensus | 0.861 | 0.882 | 0.868 |
| imr90 | core | ours | 0.926 | 0.93 | 0.917 |
| k562 | core | chip | 0.691 | 0.672 | 0.707 |
| k562 | core | consensus | 0.822 | 0.794 | 0.788 |
| k562 | core | ours | 0.845 | 0.846 | 0.845 |

## Release twin (registry 101)
Rows with `source` = `all12one_fullrd_chrho` (consensus input) and `all12one_fullrd_chrho_noref` (ChIP only) were appended to every `<cell>_<k>kb.csv` and `headline_10kb.csv` by `scripts/one_model_chrho_release_followup.py --stage distance`: every set was rerun with the twin as extra source, the old sources reproduce their stored rows (largest |dr| 1.1e-16) on the same pixel counts, the old rows are unchanged. Model: `runs/hic_measures_chrho/pred/<k>kb/<model>/<cell>.cool`.
