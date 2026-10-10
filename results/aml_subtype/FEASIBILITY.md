# Can the AML mutation subtype of the Xu et al. 2022 patients be used as a model input or stratifier?

Table: `patient_subtype_table.csv` (script `scripts/aml_subtype_table.py`, CPU only, no training).

## Sources
- GEO GSE152135 sample list (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152135): source name only ("Primary AML" or "Primary Healthy donor"), no mutation field. GSM4604271-73 (576, 103, 673) are healthy donors (PBMC).
- Paper text and Suppl. Tables (MOESM3): no per-patient mutation list. Suppl. Table 3 tags loops with Mutation:number (RUNX1, TET2, CEBPA, KMT2B; STAG2 excluded) where the number is the count of samples that harbour the loop (maximum seen 3, 4, 3, 4), not the group size. Suppl. Table 5 holds the WGS SV junctions of 22 patients.
- Fig. 1c of the paper (image, https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-022-05365-x/MediaObjects/41586_2022_5365_Fig1_HTML.png): per-sample "Consensus mutations" and "Other mutations" in the row order of the x-axis. Read by hand into the table. Cross-checks agree with the independent SV calls: t(9;22) in 027, 168, 496; inv(16) in 504, 798, 629; t(8;21) in 933; t(6;9) in 1021. Handwritten labels should be proofread against the figure once (small print).

## Findings
1. Two of the twelve "patients" with CUT&Tag (103 and 673) are healthy PBMC donors (GEO source name, Fig. 1c "Normal PBMC", absent from Suppl. Table 2). The AML-inclusive training of registry section 36 treats them as patients (training patients 103 and 673 are in the AML reference of the fold 0 test patients; fold 1 test = 103, 168, 472, 673). The CUT&Tag AML set is 10 patients; registry section 29 also states that the PBMC controls have no public Hi-C, but GSM4604271-73 are Hi-C maps.
2. Consensus group sizes over the 25 AML Hi-C patients: TET2 6 (018, 260, 424, 546, 1021, 1916), KMT2B 6 (270, 504, 629, 798, 838, 868), CEBPA 5 (566, 773, 933, 1324, 1953), RUNX1 4 (027, 168, 496, 990; three with t(9;22)), STAG2 3 (472, 936, 1071), FLT3-ITD only 1 (1360). FLT3-ITD 6 (018, 546, 933, 1021, 1360, 1916), NPM1 4 (018, 260, 424, 838), KRAS/NRAS 4 (424, 472, 838, 936).
3. Among the 10 AML with CUT&Tag: TET2 3 (018, 546, 1916), RUNX1 2 (027, 168), KMT2B 2 (270, 629), CEBPA 1 (773), STAG2 1 (472), FLT3-ITD only 1 (1360).
4. Fold 0 (test 027, 270, 773, 1916): one test patient per group (RUNX1, KMT2B, CEBPA, TET2). Training AML patients: TET2 2 (018, 546), RUNX1 1 (168), KMT2B 1 (629), STAG2 1 (472), FLT3-ITD only 1 (1360), CEBPA 0, plus the 2 healthy donors. Test patient 270 is KMT2B in Fig. 1c but carries a KMT2A-rearrangement call in the SV table (chr11 118.4-118.6 Mb); both are listed.
5. Patients with a fusion class from SVs: 22 of 25 (868, 990, 1953 have no WGS SV rows; 868 is t(9;22) in Fig. 1c).
6. Pairwise similarity at the 413 test regions (`results/aml_primary/patient_pairs.csv`, 25 patients): mean Pearson of same-group pairs minus different-group pairs 0.024 (25 kb, permutation p 0.052, 5000 label permutations), 0.021 (10 kb, p 0.16), 0.017 (5 kb, p 0.20). The effect is carried by the TET2 group (same-minus-other +0.09 to +0.11 per patient); CEBPA and KMT2B patients are not closer to their group than to others (-0.07 to +0.03). TET2 patients 424 and 260 etc. are from the same GEO batch (GSM5688xxx), so lab batch is confounded.

## What is testable with leave-patient-out
- Group-matched reference (no new training): the reference of a test patient is the mean O/E of same-group patients (Hi-C only, all 24 other patients usable, groups of 2 to 5 donors) against the mean of all others (section 29 baseline, 0.922 at 25 kb in fold 0). Testable for TET2 (n=6), KMT2B (6), CEBPA (5), RUNX1 (4), STAG2 (3); not for 1360.
- MYCN loop (present with TET2/FLT3-ITD, CEBPA or STAG2/KRAS/NRAS; absent in RUNX1 and KMT2B patients by the paper's statement): among CUT&Tag AML the positives are 018, 546, 1916, 773, 472 and the negatives 027, 168, 629, 270, 1360 (5 versus 5). Leave-patient-out predicts the loop for each patient. With 10 patients and one locus the exact permutation p cannot go below 1/252 = 0.004 (a perfect classification); fold 0 alone (2 versus 2) cannot go below 1/6 = 0.17. Which patients carry the MYCN loop was not read from the paper (Fig. 2d image not read). Loops from Suppl. Table 3 (3436 P-E rows, group counts) give a loop-level set (RUNX1 2303 rows, TET2 1633, CEBPA 990, KMT2B 90) but cannot be tied to named patients, so they can be scored per group, not per patient.
- Fold-wise subtype strata of the existing evaluation: descriptive only (1 test patient per group in fold 0; at most 2 to 3 per group over all three folds).

## Not testable
- Learning a subtype effect from an indicator input: 5 to 6 AML training patients per fold in 5 groups, 1 to 2 per group; an indicator is perfectly collinear with patient identity and can only learn a per-group mean shift of the output from 1 to 2 maps. The fold 0 CEBPA test patient has no CEBPA training patient (indicator out of distribution). Lab/batch is not separable from group. Section 36 shows no headroom either: the AML-trained model stays 0.022 below the mean of the other patients and does not exceed its reference.
- Per-mutation effects (FLT3-ITD vs TET2 vs NPM1 are nested within the TET2 group: 018 has all three).
- Any claim of statistical significance for a single group effect with 1 to 2 test patients.

## Recommendation
Not as a model input. Yes as an evaluation stratifier and as a subtype-matched reference (CPU, 0 GPU hours):
1. Inputs: Hi-C O/E maps of all 24 other patients with Fig. 1c group labels (table), 413 Akita regions, 25 and 10 kb.
2. Folds: leave-one-patient-out over the 25 AML patients (patients with CUT&Tag: the 10 AML, excluding 103 and 673).
3. Metric: Pearson/SCC per patient of group-matched reference minus all-others reference; paired over patients (n=24, Wilcoxon, groups with >=3 other members), plus on the patient-specific regions of section 29; MYCN-locus pair as a case study.
4. If the matched reference is better, one GPU job could replace the AML reference by the group-matched one (a reference choice, not a new input; about 2 to 4 GPU h per resolution for the existing refin_drop recipe, estimate from section 36, not measured). Before any further AML-inclusive run, re-define the 12 CUT&Tag "patients" as 10 AML plus 2 healthy donors and rebuild the folds and reference without 103 and 673.
