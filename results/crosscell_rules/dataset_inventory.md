# Public datasets for validating mark-to-structure rules beyond the 15 cells and the AML set

Compiled 2026-10-11 from GEO FTP SOFT files / directory listings, the ENCODE and Europe PMC APIs (GEO and PMC web pages were behind a captcha). No data files downloaded. UNVERIFIED = not read from a metadata source. Sizes marked (est.) are estimates.

| Accession | Tissue / disease | Cancer vs normal pairing | Assays (same samples) | n | Processed size | Build | Verdict | 
|---|---|---|---|---|---|---|---|
| GSE133928 (Johnstone 2020 Cell) | Colon tumours, matched normal colon, HCT116, SW480, RKO, LS174T, FHC, WI-38 | Patient tumour/normal pairs (BRD3179, BRD3187, BRD3162, BRD3462 have Hi-C on both sides) | 42 Hi-C, 54 ChIP (CTCF, H3K27ac, H3K27me3, H3K9me3, H3K36me3; patchy per patient), 48 bisulfite-seq, 22 RNA-seq, 9 HiChIP | 175 GSM | HiC-Pro iced matrices at 100 and 40 kb (0.17-2.2 GB each at 40 kb), bigWigs 49-934 MB; pair subset about 10-20 GB (est.); raw tar 327-351 GB | hg19 | Usable with effort (matrix conversion, 40 kb too coarse for stripes; patient raw reads in dbGaP phs002288) |
| GSE137188 (Kim 2023) | Colorectal cancer | 40 CRC vs 10 paired normals | Hi-C only | 50 | raw tar 19 GB; processed at junglab.kaist.ac.kr (matrices vs fastq UNVERIFIED) | UNVERIFIED | Usable with effort; no marks, pair with GSE133928 / ENCODE colon |
| GSE231486 | Human CD34+ HSC, CMP, GMP, MEP | Normal differentiation | Hi-C (2 reps), H3K27ac HiChIP, H3K27ac, H3K4me3, H3K27me3 (4 subsets), CTCF, TFs | 156 | Hi-C subseries sizes UNVERIFIED (HiC-Pro matrix/bed) | hg38 | Usable now probably; cell-type-specific test, no methylation |
| ENCODE4 cell-line panel | A673, Caco-2, HeLa-S3, HAP-1, Jurkat, LNCaP, MCF 10A, T47D, PC-3, Panc1, SJCRH30, SK-MEL-5, THP-1, others | Pairs MCF-7/T47D vs MCF 10A, PC-3/LNCaP vs prostate tissue | Hi-C (296 cell-line experiments); H3K27ac+H3K27me3 for A673, Caco-2, HAP-1, HeLa-S3, Panc1, PC-3, PC-9, SJCRH30 (term-level match); WGBS for A549, HepG2, HeLa-S3, K562, OCI-LY7, SK-N-SH, others | many | 1-30 GB per .hic; 8-10 line subset 20-100 GB (est.) | hg38 | Usable now for extra cancer lines (same pipeline as ours); normal partners mostly lack marks |
| ENCODE tissues Hi-C | 92 Hi-C tissue experiments, 46 tissues | normal only | Hi-C; ChIP per tissue not checked | 92 | GB per .hic | hg38 | Usable with effort; tissue specificity, not cancer |
| GSE118392 | Primary gastric adenocarcinoma | H3K27ac T/N pairs; Hi-C on 3 samples without normal | H3K27ac, WGS, RNA, Hi-C, 4C | 54 | small | UNVERIFIED | Not realistic for the loop rule |
| GSE66733 (Barutcu 2015) | MCF-10A vs MCF-7 | normal-like vs cancer line | HindIII Hi-C only | 2 | 283 MB | hg19 | Usable now (small, low resolution); marks from ENCODE |
| GSE115896 (Kloetgen 2020) | T-ALL, ETP-ALL, healthy T, Jurkat, CUTLL1 | 8 patients vs 3 healthy T | CTCF ChIP, H3K27ac in CUTLL1 only, RNA | 59 | Hi-C maps not on GEO (TAD beds only) | hg19 | Not realistic for matrices |
| GSE87112 (Schmitt 2016) | 19-21 human tissues and cell lines | normal only | Hi-C | 19 listed | 36 GB tar | hg19 | Usable with effort (>20 GB); marks from Roadmap |
| GSE63525 (Rao 2014) | GM12878, K562, IMR90, KBM7, NHEK, HMEC, HUVEC, HeLa, CH12 | cell lines | Hi-C | 9 | 0.16-51 GB per .hic | hg19 | Usable for KBM7 and HeLa only (rest overlaps ours) |
| GSE73785 (Taberlay 2016) | PrEC vs LNCaP, PC3 | normal vs cancer lines | Hi-C (fastq only), LNCaP H3K27ac/H3K4me1 peaks | 3 sub-series | fastq via SRA | hg19 | Usable with effort (mapping needed) |
| GSE87585 (Wu 2017) | Multiple myeloma RPMI-8226, U266 | no normal | Hi-C, WGS, RNA | 2 | 1.6 GB | hg19 | Thin; no marks |
| GSE70991 (Flavahan 2016) | IDH-mutant glioma | IDH-mut vs wt | CTCF, H3K27ac only (tdf); no Hi-C | about 40 | tdf | hg19 | Not realistic |
| GSE128463 | 25 neuroblastoma lines (H3K27ac), HiChIP in 2 lines | no normal | H3K27ac | 25 | bigWigs | hg19 | Usable with effort, K27ac only |
| 4DN portal | HeLa-S3, HCT116, GM12878, K562, 120 primary-cell sets, 22 tissues | no systematic pairs | Hi-C, Micro-C, some ChIP | 497+ | 5-50 GB each | hg38 | Usable with effort |
| GSE52457 (Dixon 2015) | hESC differentiation | not cancer | Hi-C and others | UNVERIFIED | 7.3 GB BED archive | UNVERIFIED | Usable with effort |

Not realistic or unverifiable: CLL (Beekman 2018 / BLUEPRINT, EGAD00001004046, controlled access), ESCC H3K27ac (accession UNVERIFIED, no matched Hi-C found), HCC (ChIP-chip only), pancreatic PANC1/BxPC3 vs HPDE6C7 Hi-C (PMC9218654 cites GSE97003, not opened), Leung 2015, Bonev 2017 (mouse), Wu 2021, 3D-GNOME not checked; osteosarcoma / Ewing / medulloblastoma / Wilms / melanoma: only ENCODE lines (A673, SJCRH30, SJSA1, MG63, G401, SK-MEL-5), no primary tumour sets found; AML other than Xu: none found.

Best candidates: (1) GSE133928 colon tumour vs normal (the only verified set with matched patient Hi-C matrices, H3K27ac, H3K27me3, CTCF and methylation; K27ac pair complete only for few patients, so HCT116 vs FHC plus BRD3179/BRD3187; hg19, 40 kb); (2) ENCODE4 cancer lines (A673, Caco-2, HeLa-S3, Panc1, PC-3, PC-9, SJCRH30, HAP-1; hg38, same pipeline); (3) GSE231486 HSC/CMP/GMP/MEP (hg38, all marks in the same cell types); (4) GSE137188 CRC cohort (loop gain only, needs marks elsewhere); (5) GSE66733 MCF-10A vs MCF-7.

Not verified: GSE137188 matrices and build; GSE231486 Hi-C sizes; whether ENCODE Hi-C and ChIP of a shared term share a biosample; 4DN counts from a summarised JSON; GEO accessions for Beekman, Javierre, Rhie, Wu 2021, Leung 2015.
