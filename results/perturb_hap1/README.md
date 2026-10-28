# In silico deletion of CTCF motifs in HAP1 (Sanborn et al. 2015 capture Hi-C)

Model: ChIP-only, chromosome-independent `all12honest_drop_10kb` (and `all12honest_drop_5kb`), new-cell mode, no consensus, `--tta`, batch size 4, GPU 1.
HAP1 is not a training cell. Predictions are log O/E (model units, within 2 Mb of the diagonal). Nothing was retrained.

## 1. Coordinates (GRCh38, no liftover needed)

The deleted motifs come from the 4DN modification records of the experiment sets (`target_of_mod` -> genomic region, assembly GRCh38,
1-based inclusive, 20 bp; raw dump in `sites_4dn_raw.tsv`). Region identity follows from the paper (Fig. 7: A/B/C on chr8, D/E/F on chr1, chr5 locus),
the 4DN names and the cis-coverage of the WT map.

| region | captured (GRCh38, 2 Mb) | deleted motif | coordinates | strand |
|---|---|---|---|---|
| 1 | chr8:132,000,000-134,000,000 | A forward | chr8:132,875,646-132,875,665 | + |
| 1 | | B reverse | chr8:133,203,069-133,203,088 | - |
| 1 | | B forward | chr8:133,209,431-133,209,450 | + |
| 2 | chr1:179,800,000-181,800,000 | E reverse | chr1:180,863,366-180,863,385 | - |
| 2 | | E forward | chr1:180,882,097-180,882,116 | + |
| 3 | chr5:31,000,000-33,000,000 | H forward | chr5:31,825,672-31,825,691 | + |

(The README of the share lists a fourth captured locus, chr4:140-142 Mb, which belongs to none of the 20 files used here, and gave region 2 as 180.1-181.8 Mb; the coverage profile starts at 179.8 Mb.)
Every clone captured only its own region (counts of the other three loci are about 1% of the own region), so no clone is "WT-like" at another region.

Verification (all agree, none contradicts):
* orientation: the JASPAR MA0139.2 best hit overlapping each 20 bp window lies on the strand named by 4DN for all 6 sites (`site_motif_check.csv`). Motif score as a fraction of the maximum: A 0.75, B-rev 0.87, B-fwd 0.74, E-rev 0.68, E-fwd 0.78, H 0.59. These are below the 0.80 threshold of the model's own scan (flexcible hits at score_frac 0.8), so at 4 of 6 sites the model's motif channel would not even have a hit; the knockout of the motif channel therefore changes little and the effect rests on the CTCF ChIP track (see caveat 3).
* HAP1 CTCF ChIP: every site lies in a MACS3 peak (fold enrichment 9.8-34.7, -log10 q 27-115; max fold change within +-50 bp 14.8-56.7), `site_chip_check.csv`.
* hg19 cross-check: liftOver GRCh38 -> hg19 of A, B, E, H gives chr8:133.888 Mb (paper: locus A at 133.9 Mb), chr8:134.215/134.222 (B, 134.2 Mb), chr1:180.833/180.851, chr5:31.826. The Sanborn HiCCUPS loops (GSE74072, hg19, lifted to GRCh38 with the UCSC chain, `loops_hap1_hiccups_grch38.tsv`) have anchors at every deleted motif and connect to the partner motifs (A-B, A-C, B-C, D-E, E-F), orientation convergent.
* Uncertainty: the 4DN record is the only source of the 20 bp coordinates; Sanborn SI Table S7 (guide RNAs) could not be downloaded (PMC and PNAS block scripted download), so the exact deleted base pairs (a 17-159 bp deletion in the paper) were not compared with it. A 20 bp window within a CTCF peak with matching orientation and loop anchor is sufficient for a +-1 kb knockout. The paper's chr5 insertion (hg19 chr5:31,581,788) is the G site, not the H deletion used here. Region 3 anchor: the H-motif lies 0.8 kb outside the 5 kb loop anchor bin of the HiCCUPS list (offset of the grid), handled by a 5 kb tolerance.

## 2. Inputs

ENCODE has **no CTCF, RAD21, SMC3, H3K4me2, H3K79me2, H3K9ac or H4K20me1 ChIP-seq for HAP-1** (the released HAP-1 experiments are 6 histone ChIP, 1 control, 2 DNase, 1 in situ Hi-C; searched with biosample_ontology.term_name=HAP-1, also with searchTerm). Downloaded (md5 verified; `track_provenance_encode.tsv`, candidates in `tracks/PROVENANCE_candidates.tsv`): H3K4me3 ENCFF487XII, H3K27ac ENCFF730PAV, H3K36me3 ENCFF490WCQ, H3K9me3 ENCFF396PFA, H3K27me3 ENCFF375LTF, H3K4me1 ENCFF884TTU (all pooled replicates, fold change over control, GRCh38; none of the FC files carries preferred_default).

**Deviation:** without CTCF the experiment is void, so CTCF.bigwig was built from public HAP1 reads: Haarhuis et al. 2017 (GSE94992), WT HAP1 CTCF ChIP SRX2570608 (SRR5266526, 21.5 M single-end reads) against WT HAP1 IgG SRX2570604 (SRR5266522, 27.5 M reads); bowtie2 to hg38 (MAPQ >= 30), MACS3 3.1.0 `callpeak --nomodel --extsize 200 -p 1e-2 -B`, `bdgcmp -m FE` (fold enrichment over the local lambda), bigWig (`scripts/perturb_hap1/make_hap1_ctcf_track.sh`; md5 of CTCF.bigwig in `ctcf_track.md5`, build log `ctcf_track_build.log`). This is not an ENCODE file, its scale may differ from the ENCODE fold-change tracks the model was trained on, and the same-lab SMC1 ChIP (SRX2570612, not SMC3) was not used. Zeroed absent tracks: RAD21, SMC3, H3K4me2, H3K79me2, H3K9ac, H4K20me1. The ChIP tracks are in `/mnt/processed/wolffjoa/hic-multiome/perturb/hap1_insilico/tracks/`.

## 3. Method

Measured (`scripts/perturb_hap1/lib.py`, `measure.py`): raw counts of the region from the mcool (no balancing), scaled to 1e6 counts in the upper triangle, E(d) = mean of diagonal d inside the region (zeros included), log2((O+1)/(E+1)), uncovered bins NaN. Change = deletion minus WT; each clone separately, the mean of the clone changes, and a pooled-counts map. 10 kb and 5 kb.
Predicted: change = prediction with `--zero-region CTCF:...` and `--zero-region motif:MA0139.2:...` over motif +-1 kb (one 10 kb bin, two where it crosses a boundary) minus the WT prediction. E_both removes both motifs. Commands: `commands_predict.sh`; cool files in `.../hap1_insilico/cools/{10kb,5kb}/` (wt_<region>, del_<deletion>, ctrl_<deletion>).
Scores (`analyze.py`, per-region CSVs `region{1,2,3}_{map,loop,insulation}_scores.csv`, summary `summary.csv`):
1. Pearson over all pixels >= 2 bins off the diagonal (also <= 1 Mb, Spearman).
2. Loops: Sanborn HiCCUPS loops with an anchor (widened by 5 kb) at the deleted motif; pixel = centres of both anchors; value = max over 3x3 of log O/E in WT and in deletion; change = difference; sign agreement of measured and predicted change. `convergent_anchor` marks loops whose anchor orientation fits the motif (x anchor = forward, y anchor = reverse); only these enter the summary.
3. Insulation: mean log O/E over the 100 kb (and 200 kb) square upstream x downstream of the deleted motif; change = deletion minus WT (positive = weaker boundary).
Controls: (a) the same knockout at a CTCF motif (JASPAR >= 0.70, strongest HAP1 CTCF fold change, >= 50 kb from region edge, not a deleted-site loop anchor) 1.0 Mb from the deleted site, `control_sites.tsv`, scored on the deleted site's loops/insulation and on the map correlation; (b) specificity: measured change of one deletion against the predicted change of the other deletions of the same region; (c) noise floor: no WT-like clone pair exists (one WT map; each clone covers one region), so the agreement of measured change maps between independent clones of the same deletion is given as the ceiling (`clone_vs_clone`).
`hicDetectLoops` on a region-only ICE-balanced WT cooler detected no loops with default and relaxed parameters (script `detect_loops.py`), so the paper's loops are used.

## 4. Results (summary.csv; unit = mean over clones)

| model | region | deletion | clones | Pearson pred vs meas (clone range) | Pearson of control-site prediction | clone-vs-clone Pearson (ceiling) | loops sign agree | measured / predicted / control change at loops (log2 / model units) | insulation 100 kb sign agree | measured / predicted / control change |
|---|---|---|---|---|---|---|---|---|---|---|
| 10kb | 1 | A fwd | 1 | 0.21 | 0.01 | n/a | 2/2 | -1.27 / -0.24 / 0.00 | 1/1 | 0.41 / 0.23 / 0.00 |
| 10kb | 1 | B rev | 1 | 0.05 | 0.00 | n/a | 1/1 | -1.15 / -0.21 / 0.01 | 1/1 | 0.54 / 0.17 / 0.00 |
| 10kb | 1 | B fwd | 2 | 0.06 (0.05-0.06) | 0.03 | 0.53 | 1/1 | -1.06 / -0.17 / 0.00 | 1/1 | 0.30 / 0.17 / 0.00 |
| 10kb | 2 | E fwd | 4 | -0.05 (-0.05 to -0.02) | 0.01 | 0.36 | 1/1 | -0.89 / -0.07 / 0.00 | 0/1 | 0.22 / -0.00 / 0.00 |
| 10kb | 2 | E rev | 3 | 0.17 (0.11-0.15) | 0.04 | 0.38 | 1/1 | -1.46 / -0.13 / 0.00 | 1/1 | 0.27 / 0.08 / 0.00 |
| 10kb | 2 | E both | 6 | 0.23 (0.11-0.18) | 0.03 | 0.26 | 2/2 | -1.60 / -0.18 / 0.00 | 2/2 | 0.82 / 0.08 / 0.00 |
| 10kb | 3 | H fwd | 2 | 0.07 (0.05-0.06) | -0.15 | 0.34 | 2/2 | -0.38 / -0.13 / 0.00 | 1/1 | 0.49 / 0.15 / 0.00 |
| 5kb | 1 | A fwd | 1 | 0.11 | -0.01 | n/a | 2/2 | -1.55 / -0.56 / 0.01 | 1/1 | 0.32 / 0.18 / 0.00 |
| 5kb | 1 | B rev | 1 | 0.06 | 0.00 | n/a | 1/1 | -0.62 / -0.26 / 0.01 | 1/1 | 0.46 / 0.16 / 0.00 |
| 5kb | 1 | B fwd | 2 | 0.03 (0.01-0.03) | 0.01 | 0.54 | 1/1 | -0.93 / -0.23 / 0.00 | 1/1 | 0.31 / 0.04 / 0.00 |
| 5kb | 2 | E fwd | 4 | 0.01 (0.00-0.01) | 0.00 | 0.41 | 1/1 | -1.09 / -0.13 / 0.00 | 1/1 | 0.03 / 0.01 / 0.00 |
| 5kb | 2 | E rev | 3 | 0.11 (0.06-0.10) | 0.01 | 0.42 | 1/1 | -1.52 / -0.32 / 0.00 | 1/1 | 0.10 / 0.12 / 0.00 |
| 5kb | 2 | E both | 6 | 0.16 (0.08-0.13) | -0.01 | 0.29 | 2/2 | -1.93 / -0.30 / 0.00 | 2/2 | 0.45 / 0.12 / 0.00 |
| 5kb | 3 | H fwd | 2 | 0.07 (0.06-0.06) | -0.05 | 0.35 | 2/2 | -0.68 / -0.32 / 0.00 | 1/1 | 0.35 / 0.25 / 0.00 |

(The Pearson of the mean over clones exceeds the range of single clones because averaging removes noise; per-clone values are in the CSVs.)
Loop pixels: the predicted change at the loop is negative in every one of the 12 deleted-site loops (convergent anchor) at both resolutions, as measured, and the same knockout 1 Mb away changes them by <= 0.011. Two loops with a non-convergent anchor (B_rev on the B-C loop, B_fwd on the A-B loop) lose less in the measured map (-0.33, -0.29 at 10 kb) and the prediction is also small there.
Insulation: 15 of 16 site-windows at 10 kb have the measured and predicted sign (E forward at 100 and 200 kb is the exception at 10 kb); the control changes are |<=0.004|.
Map-wide agreement is low: Pearson 0.0-0.23 against a clone-vs-clone ceiling of 0.26-0.54, because the measured change maps are dominated by pixel noise of the low-depth capture maps (SD of the measured change about 1 log2 unit against a predicted SD of 0.01-0.09) and the model predicts a change that is local to the site (stripe and the loop/domain block), larger in B/A/E-both than in E forward and H. Specificity: the measured change of A_fwd correlates negatively with the predictions of B_rev and B_fwd (-0.17), E_both's measured change with the E_rev prediction at 0.23 (E_rev and E_both share the D-E loop).
Region 3 control: the only motif available 1 Mb away has a weak peak (fold change 3.3); the control prediction correlates -0.15 with the measured change at 10 kb, which is the largest control value.

## 5. Caveats and things that did not work
1. CTCF track is self-processed from GSE94992 (see 2); ENCODE has none for HAP-1. RAD21/SMC3 absent (zeroed), so the loop-extrusion partner channels of the model are absent for this cell.
2. Sanborn SI Table S7 not retrieved; coordinates from 4DN only.
3. Motif scores of 4 of 6 sites are below the model's 0.80 threshold, so the motif knockout is partly moot and the CTCF-track knockout carries the effect.
4. hicDetectLoops found no loops on the capture region; HiCCUPS loops of the paper were used.
5. No WT-like clone pair exists (one WT map, one region per clone), hence the clone-vs-clone ceiling instead of a WT-vs-WT noise floor; 5 kb maps have about 5 counts per pixel.
6. The loop-anchor assignment uses a 5 kb tolerance; the figure colour scale of the predicted change is separate from the measured one. The CTCF build ran bowtie2 -p 8 plus samtools -@4 threads (slightly above 8 threads in total, for about 40 minutes).
7. A control for each deletion shares a site across deletions of one region (B_rev, B_fwd; E_*).

## 6. Files
Scripts `scripts/perturb_hap1/` (common.py, lib.py, measure.py, prep_loops.py, verify_sites.py, make_hap1_ctcf_track.sh, pick_controls.py, run_predictions.py, analyze.py, summarize.py, detect_loops.py). Figure script (manuscripts repo, uncommitted): `~/src/manuscripts/hic-multiome-natcomms/figures/scripts/fig_perturb_hap1.py` -> `fig_perturb_hap1_<region>.pdf` (one row per deletion; orange = increase, purple = decrease in the change maps; ticks mark the deleted motif; 80 bins of 10 kb centred on the motif) and one PDF per panel `fig_perturb_hap1_<region>_<deletion>_<panel>.pdf`.
Run order: measure.py, prep_loops.py <looplist.gz>, verify_sites.py, make_hap1_ctcf_track.sh, pick_controls.py, run_predictions.py, analyze.py, summarize.py.
