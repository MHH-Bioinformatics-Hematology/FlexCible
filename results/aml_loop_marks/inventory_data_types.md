| data type | level | format | files | GB | used now | not used yet |
|---|---|---|---|---|---|---|
| Hi-C read pairs (Xu 2022) | processed (aligned, de-duplicated pairs; no FASTQ) | nodups.pairs.gz | 32 | 168.8 | all 32: 5 kb cooler, 10 / 25 kb ICE and oe_proper targets | 2 kb maps not built |
| Hi-C iced matrices (Xu 2022) | processed | .iced.mcool (10 kb and coarser) | 32 | 24.8 | targets rebuilt from the pairs | 5 kb and finer are not in the mcools |
| Hi-C targets and consensus maps | processed | cooler 5/10/25 kb, oe_proper | 148 | 24.1 | reference maps of the models and the loop detection |  |
| CUT&Tag signal (H3K27ac, H3K27me3, CTCF) | processed (MACS2 -log10 p-value signal, nodup_x_ctl) | bigWig | 33 | 5.6 | quantile-matched to the training scale (inputs/xu_*) = model inputs; 10 kb bin means in the loop analyses |  |
| CUT&Tag peaks (MACS2 filt.peaks.txt.gz) | processed | txt.gz |  |  | no | peak calls: not used, not downloaded (MACS2 command line is not in the files we hold) |
| CUT&Tag raw reads | raw | FASTQ |  |  | no | not available locally |
| WGBS methylation (Xu 2022) | processed (CpG methylation fraction per 10 bp bin, no read counts) | .CGN.10bp.bw | 20 | 3.9 | 1 kb Meth.bigwig (unweighted mean of covered 10 bp bins, >= 3 per kb) in the Meth model variants and the loop classifier; 10 bp values at CTCF motifs in this analysis | per-CpG read counts / coverage (not in the bigWigs) |
| ATAC-seq (Xu 2022) | processed (RPGC bigWig + MACS2 peaks) | bigWig, peaks.txt.gz | 33 | 2.4 | not here | in progress (other agent) |
| RNA-seq (Xu 2022) | processed (RSEM gene quantification) | txt.gz | 22 | 0.1 | not here | in progress (other agent) |
| Fischer 2024 ChIP (RAD21, STAG2, CTCF, H3K27ac per AML patient) | processed (CNV-normalised bigWig) | bigWig | 160 | 29.5 | group-level inputs (figroup_*) for the Fischer cohesin predictions | per-patient RAD21 / STAG2 ChIP not used in any loop analysis |
| Fischer 2024 pooled Hi-C | processed (FAN-C HDF5, 10 kb) | .fanc.h5 | 3 | 34.3 | cooler conversion and group predictions |  |
| AML cell-line Hi-C and ChIP | processed | mcool / hic, ChIP-Atlas bigWig | 3 | 1.9 | model test cells (aml_primary sections) | THP-1 / HL-60 / OCI-AML3 / MV4-11 listed in the survey only; cell-line loops / stripes not analysed |
| SV junctions | processed (published table) | csv | 1 | 0.0 | SV-aware prediction (aml_sv_predict) |  |
| CNV segments | derived from Hi-C coverage vs HSPC | csv | 1 | 0.0 | region flags |  |
| Mutation / fusion labels | published figure, hand-read | csv | 1 | 0.0 | subtype folds of the loop classifier | no per-mutation effect analysis on loop gain yet |
| AML-specific loops and stripes | processed (published tables) | csv | 1 | 0.0 | loop classifier, loop detection, this analysis |  |
| HSPC ChIP reference | processed (ChIP-Atlas bigWig, 1 experiment per mark) | bigWig | 3 | 0.1 | the HSPC reference of all loop analyses | 7 further CD34+/HSC experiments fetched in this analysis (WORK/hspc_extra) |
