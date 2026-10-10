# Number audit, 2026-10-10

Scope: every quantitative claim on the progress page (`template.html`, built `hic-progress.html`, figure data
`panels.json`, `scatter.json`, `scc_rows.json`, `make_figs.py`) and the claims listed in the chat summaries.
Table: `audit_table.csv` (89 rows: id, claim, where cited, value, source, status, evidence, replacement).
Supporting files: `run_honesty_scan.csv` (524 training logs: yaml `val_exclude_overlap`, honest-split log line,
end of training, verdict; made by `scan_runs.py`), `split_disjoint_5kb_2kb.csv` (training windows against the
413 Akita test regions at 5 and 2 kb, 0 overlaps; the 25 and 10 kb caches were already checked in
`runs/multicell_reference/split_disjoint.csv`).

Status rule. HONEST: run with `val_exclude_overlap: true` (or an honest-split log line) or an analysis without
a validation split, with a consensus that excludes the target. LEAKY: overlapping validation. UNVERIFIABLE: no
current registered source, stale, or contradicted by the tables. AFFECTED: depends on the healthy PBMC donors
xu_103 and xu_673 counted as AML patients (coordinator finding, registry section 50; not recomputed here).

## Counts

| status | rows |
|---|---|
| HONEST | 41 |
| LEAKY | 23 |
| AFFECTED | 18 |
| UNVERIFIABLE | 7 |

## Problems, by importance

1. Headline U-Net numbers of the page come from the overlapping-validation section 1 model
   (`configs/akita_native.yaml`, `val_exclude_overlap: false`). Rows P03, P04, P06, P07, P09 to P12, P14, P15, P50:
   0.652, 68% (151 of 222), SCC table row 0.659 / 0.623, 181 and 202 SCC wins, the 222-dot scatter, the seven
   example panels with scores, and the summary row 1. Fix: replace by `akita_native_honest` (222 regions: Pearson
   0.640, 66% = 147, 448-bin Pearson 0.647, SCC 0.611, SCC wins 169 and 200; per-region scores in
   `results/akita_comparison/akita_native_honest.score.csv`, `results/benchmark/scc_akita_regions.csv`) and
   regenerate the panels and scatter. The note under the SCC table already gives the honest 0.640 and 0.746.
2. Section 11 and 13 runs (`multicell`, `multicell_expanded`, trained 2026-09-24 to 27, no honest-split line):
   resolution chart "Model" series 0.656 / 0.777 / 0.823 / 0.815 (P23), K562 cell-count curve 0.362 / 0.463 /
   0.494 / 0.551 / 0.544 (P21, P22, P53), the three held-out "without consensus" dots 0.767 / 0.698 / 0.781
   (P29), the 11 of 12 and 1 of 12 counts (P27), the four-motif row 0.681 to 0.694 from section 8 (P51).
   Fix: all12honest 0.6576 / 0.7857 / 0.8299 / 0.8277 (P23); held-out dots 0.771 / 0.735 / 0.805 (P29, label the
   rows HMEC deep and HUVEC deep); 12-cell K562 0.544 from all12honest_drop; no honest 1, 3, 5, 7 cell series and
   no honest 1-versus-4 motif comparison exist, so remove those curve points and the motif row.
3. Healthy donors counted as AML patients (N01 to N05, rows 46 to 48, 34, 35, 49, C08 to C11, C18, C22, C23).
   Section 29 (12 patients, "AML consensus of the other 27 patients", PBMC sentence), sections 36 and 38 (AML
   reference of the 8 training patients contains 103 and 673), section 46 ("nine AML patients" = 8 AML plus 103:
   pooled partial Spearman -0.37, relative risk 2.19 and 2.91, insulation 0.028, per-patient rows), section 36b
   site columns, page text in #seqmeth, #amlctcf, #amlloops. Fix: recompute over the AML-only sets and references.
4. Seed noise 0.013 (P44, also registry sections 35 and 38, Figure 2b band, the n/d threshold of section 35).
   It is the difference of two section 8 models (0.674 vs 0.687, overlapping validation) and is cited as
   "section 15". Honest two-seed spreads (`coarse_gap/summary.csv`): 0.0002 to 0.0062 for consensus-input models,
   0.004 to 0.028 without consensus; section 14 median 0.0075. The claim "every gain lies inside the seed noise"
   is unsupported (a deterministic weighted consensus has no seed noise; the gains 0.006 to 0.008 exceed the
   honest spread of consensus-input models). Fix: delete the sentence and the band.
5. Chat claims: "beats Akita in 69% of regions" is the overlapping model (0.6925); honest 66% (66.2%), 97% with
   consensus (C04). "Hi-cGAN 0.196" has no source; 0.238 on 411 regions, 0.220 on 222 (C03). Region count: 411
   for the identical-region block (two regions lack Hi-cGAN predictions), Akita 0.5515; 413 regions give 0.636,
   0.752 and 0.548 (C02).
6. Stale or wrong values. Resolution chart replicate series 0.6265 and 0.9445 are pre-correction 4 kb and 24 kb
   values (P25; correct 0.6505 and 0.9485). Summary row 5 "0.822" is GM12878 alone, the 12-cell median is 0.828
   (P54). "The same order holds at 5 and 25 kb" and the title "CTCF and cohesin carry the prediction" are not
   supported by `resolution_inputs/summary.csv` (P18). Page "Queued next" and header date are stale (P56, P58).
7. Section 16 (worst-predicted regions) ranks the 2 kb regions with the section 13 all12 model (P55); re-run with
   an honest 2 kb model before citing the 2 kb half. Section 20 row akita_native_trackdrop is leaky (C13); sections
   1 to 13 in general must not be cited (C25).
8. Figures. The 25 kb panels show 2 Mb with the 1 Mb test region framed, with the warning printed on the image
   (checked on the compartment image and the 10 kb image: 10 kb crops are inside the held-out region). Exception:
   the AML-site figures (800 kb crops at Xu loci, outside the Akita regions) show the GM12878 prediction (a
   training cell, stated in each image title) and rows for training patients 168, 629, 546 (stated in captions);
   the section lead does not say so (P35).
9. Wording check: Hi-cGAN is "submitted for peer review" twice and never "published" in visible text; the CSS
   comment `/* published figure */` in the source refers to the Hi-cGAN figure and should be renamed (P57).

## Honest and verified (cite)

Consensus-input held-out medians 0.914 / 0.903 / 0.830 / 0.677 and consensus alone 0.898 / 0.859 / 0.785 / 0.629
and honest_drop 0.775 / 0.761 / 0.684 / 0.544 (P36 to P41); lift 0.828 to 0.939 (C01); Akita-region 0.640, 0.746
vs 0.551 (n = 411, C02); whole-chromosome 10 kb 0.926 vs C.Origami 0.920 and ChIP-only 0.808 (C05, no interval,
consensus input labelled); HiC-Reg 0.660 vs 0.957 (C07, but ours trained on chr14 and chr17; use section 37:
0.931 vs 0.579); minimal-input sets (C12: dedicated CTCF plus H3K27ac 0.909 vs full 0.920, K562 gap 0.035);
replacement fractions 0.93 / 0.51 (v2 0.38) / 0.39 and 0.32 (training cells only, C14 to C16); noise ceilings
0.806 / 0.590 (C18, after the AML recompute); hard-split replicate ceiling 0.861 (C19); degron change maps
0.803 / 0.842 on even chromosomes only (C20); sections 39, 40, 41, 48 and the loop-classifier numbers.

## Could not verify

Per-panel r values of the example-region images (taken from the score tables, images not regenerated); the
cell-specific figure panels other than two inspected images; that all AML-site images match `sites_loci.csv`
beyond the captions; Hi-cGAN 0.196 (no source found); the chrho comparison intervals (none exist); whether
reference_summary.csv changes after 2026-10-10 07:24; the AML recomputation (not done, see section 3).
