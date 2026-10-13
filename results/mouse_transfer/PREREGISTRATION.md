# Pre-registration of the mouse transfer (written 2026-10-13, before any score of a mouse window was computed)

Question. Do the human-trained models, applied unchanged to mouse chromatin marks, predict mouse Hi-C? A positive
answer says the learned mapping from marks (and local sequence composition) to 3D structure is shared between
human and mouse; a negative answer says that part of what the models learned is human-specific.

Design. No retraining, no tuning, no threshold or parameter choice after seeing a score. Models: the final
checkpoints (checkpoint-best) of all12honest_drop, all12refin_drop, all12refres_drop and
all12diffcells_refres_drop at 10 kb and 25 kb, trained on 12 human cell lines only.

Genome build. mm10 (GRCm38) throughout. All ENCODE Hi-C and ChIP files are mm10 natively and the Akita mouse
heads are mm10; every input of one sample shares the build, nothing is lifted to mm39 (a chain lift would
re-bin contacts and move bins; staying in mm10 loses nothing). The model is build-agnostic (bins, not
coordinates).

Samples (fixed before scoring; three distinct lineages; Hi-C and ChIP of the same cell type or tissue):
- mesc: mouse ES cell E14. Hi-C ENCSR051JZL (ENCFF092WXU, in situ, MAPQ >= 30). ChIP: ES-E14 (H3K27ac, H3K4me1,
  H3K4me3, H3K36me3, H3K9me3) and E14TG2a.4 (CTCF, H3K27me3; the E14 subclone). Replicate experiment of the
  same line: ENCSR819XFA (ENCFF079MKO), used for a replicate ceiling.
- ch12: CH12.LX B-cell lymphoma. Hi-C ENCSR005UON (ENCFF076GYR, in situ, 3 replicates pooled). ChIP: CH12.LX,
  seven marks, plus RAD21 and SMC3.
- heart: adult heart (2 months). Hi-C ENCSR997TTD (ENCFF651SGY, intact, B6CASTF1/J). ChIP: heart, adult 2 months
  (CTCF, H3K27ac, H3K4me3 of B6CASTF1/J female; H3K27me3, H3K36me3, H3K4me1 of B6NCrl male; H3K9me3 of the
  same B6NCrl series).
- Not used: left cerebral cortex (Hi-C ENCSR639NRL, 52.8 GB, disk budget; only CTCF, H3K27ac, H3K4me3 exist for
  the same tissue and strain).
- Control variant: mesc_b4 = the mESC Hi-C with the ES-Bruce4 ChIP of all seven marks (another line, one lab).

Inputs (as for the human cells, scripts/independent_validation_fetch.py conventions): MAPQ30 .hic -> hicxchange
hic2cool -> cooler balance (ignore-diags 2, 500 iterations) -> scripts/oe_proper.py (10 and 25 kb); ChIP as the
released mm10 fold-change-over-control bigWig, through the models' own track normaliser (log1p, global z-score
of training) and the models' own matrix normaliser (that of the first training sample, as for unseen human
cells). What the model reads: seven ChIP tracks (CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3,
H3K9me3), GC and CpG fraction per bin and the four motif planes (CTCF MA0139.2, YY1 MA0095.3, ZNF143 MA0088.2,
MAZ MA1522.1) scanned with the human JASPAR PWMs on mm10 sequence (the motifs are conserved). RAD21, SMC3,
H3K4me2, H3K79me2, H3K9ac, H4K20me1 are given as absent (CH12 has RAD21 and SMC3: used only in the extra set
"avail"). The Hi-C reference of refin / refres and the mark reference of the differential-channel model are
species-matched: the leave-one-out mean of the other mouse samples (O/E maps of oe_proper.py; marks of the
other samples), so three samples give a two-sample consensus; the human consensus12 is not used (syntenic
regions differ).

Windows. The 6 038 tiles of Akita v2 mm10 (sequences.bed, 1 310 720 bp, 8 folds); scored region the central
1 048 576 bp, whole 10 kb (25 kb) bins inside it; a tile is used when the model input span (192 bins at 10 kb,
80 bins at 25 kb, centred) lies inside the chromosome and contains no assembly gap (N run >= 1 kb).
Autosomes chr1 to chr19 are the headline set; chrX is reported separately (the differential-channel model has
no chrX rows: statistics of autosomes only). Pixels: >= 2 bins off the diagonal, finite in the target and in
every compared predictor (identical pixels), outside bins overlapping the ENCODE mm10 blacklist v2; windows
with >= 100 such pixels. Score: per-window Pearson of log O/E; sample score = median over windows.

Predictors. (1) honest: all12honest_drop_{10,25}kb, seven marks, no consensus. (2) refin, refres, diff:
all12refin_drop, all12refres_drop, all12diffcells_refres_drop with the mouse leave-one-out reference. (3)
consensus alone: the mouse leave-one-out map. (4) distance-decay-only: the chromosome-wide mean O/E per
distance, which is zero by construction of oe_proper.py; a constant has no covariance, its Pearson is set to 0.
Also reported, not used for the endpoint: the in-sample per-window per-diagonal mean of the target (upper bound
of any function of distance). (5) Akita v2 mouse heads (mm10, gs://basenji_hic/3-2021, sequence only): for
every tile the model of the fold that holds the tile out as test; heads Hsieh mESC Micro-C and Bonev ES for
mesc, Bonev cortical neuron / NPC as off-lineage heads; only mesc has a matching head; evaluated on a random
1 500-window subset (seed 0) because of CPU time. Controls: marks shuffled (every ChIP track of a chromosome
rolled by one common offset of 25 to 75 percent of its length; sequence, GC, motif planes untouched), marks of
a different mouse sample (mesc <- heart, ch12 <- mesc, heart <- ch12), mark subsets (CTCF + H3K27ac, CTCF,
none), the untrained network (random initialisation, same normalisers; human-pretrained vs nothing), mesc_b4.

Ceilings. Split-half: raw counts binomially split 1:1, both halves balanced and transformed as the registry
replicate ceiling, per-window Pearson, Spearman-Brown to full depth, median over a random 1 000-window subset
(seed 0). Replicate experiment (mesc only): O/E of ENCSR819XFA against ENCSR051JZL on the same pixels
(lower depth, not corrected).

PRIMARY ENDPOINT. Median per-window Pearson, 10 kb, autosome windows, of all12honest_drop_10kb with the seven
marks of the sample (human model, mouse marks, no consensus) is above (i) the distance-decay baseline and (ii)
the shuffled-marks control in every one of the three mouse samples; "above" = difference of medians > 0 with
the 95 percent paired bootstrap interval (2000 resamples of windows) excluding 0. Its fraction of the split-half
ceiling (median / median) is reported per sample. The endpoint is met only if both comparisons hold in all
three samples; any sample that fails is reported as such and the endpoint is then not met.

Secondary (reported whether or not the primary succeeds). (S1) refin, refres, diff with the mouse leave-one-out
reference minus the consensus alone (paired contrast, 10 kb; 25 kb in the tables), and refin minus honest.
(S2) honest and refin versus Akita v2 mouse heads on mesc. (S3) mark subsets and CH12 "avail" (with cohesin).
(S4) structure classes, median per window on identical pixels: insulation r, boundary F1 (mean), loop enrichment r
and APA ratio, compartment r (window-level correlation-matrix correlation), stripe-profile r (per-bin mean
O/E over 3 to 30 bins downstream and upstream, exploratory). (S5) 25 kb. (S6) chrX.

Interpretation rules, fixed now. "Conserved biology" is supported by the endpoint only in the sense that a
human-trained function from marks to structure carries information in mouse above the controls; the size of
the effect is read from the fraction of the ceiling, not from the sign alone. If honest does not exceed the
shuffled marks in a sample, that sample contributes no support. A structure class transfers when its
median measure of honest exceeds that of the shuffled-marks control with the interval above 0 in all three
samples; classes with weaker or inconsistent results are reported as not transferring. Few samples, assay
differences (in situ vs intact), strain mixing of the heart ChIP and the lack of species-specific retraining are
stated as limits, not corrected.
