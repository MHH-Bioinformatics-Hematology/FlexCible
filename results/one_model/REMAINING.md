# Remaining analyses on the release checkpoint all12one_fullrd (registry section 98)

Written 2026-11-01. The manuscript (`~/src/manuscripts/hic-multiome-natcomms/main.tex`, read only) switches to ONE released
checkpoint, `all12one_fullrd` (13 marks + Meth, cohesin-fraction row, reference dropout 0.3, one network for 2, 5, 10 and 25 kb;
views `runs/multicell_reference/all12one_fullrd_<k>kb`). This file reruns the six analyses that still rest on the older dedicated
models, with the protocol, regions and metrics of the paper numbers, and gives for each the paper's current number, the release
number, all statistics, and the sentences of main.tex that change. Existing results were not modified; everything new is under
`results/one_model/remaining/<analysis>/` (scripts `scripts/one_model_remaining_*.py`, predictions on the share under
`runs/aml_one_model`, `validation_new/pred_rel`, `validation_mouse/pred_rel`, `runs/akita_one_model`). Predictions ran on the
remote RTX 4090 (data staged under `~/scratch`, only files created for this work were removed afterwards), scoring on this workstation.

Two uses of the checkpoint appear everywhere: **with a reference** (the Hi-C reference input as in the paper's reference-input models:
consensus12, the leave-one-out consensus, the AML reference or the mouse consensus, depending on the analysis) and **marks only**
(`flexcible predict --no-reference`, the reference plane at 0), which is the counterpart of the paper's "model without consensus"
(all12honest_drop). Rows are labelled PAPER (the dedicated models behind the current numbers, recomputed in the same pass or read from
the existing files) and RELEASE. In every analysis the PAPER rows reproduce the registry values (checked: sections 72, 74, 75, 73, 49).

Not run / not available: the whole-genome 1 Mb tiles of the new cell types (section 74 secondary), the untrained-network control and the CH12.LX
RAD21/SMC3 set of the mouse analysis, the chromosome-held-out twin `all12one_fullrd_chrho` (still training when this was written; section 94).
The release checkpoint is one run (no seed spread). The timing was measured on the second computer, not on the workstation of section 82.


## 1. AML-specific loops and stripes in the ten held-out patients

Protocol of registry sections 62 / 72 (`scripts/aml_competitors_loops.py`, functions called unchanged by `scripts/one_model_remaining_aml.py`): 2276 AML-specific loops and 1075 stripes of 10 donors (Xu et al.), controls B_rand (distance-matched random pairs) and A_refrand (matched on AML reference and HSPC enrichment, distance, orientation), 192-bin windows of 10 kb, patient marks CTCF / H3K27ac / H3K27me3 (1360 and 1916 lack CTCF), detection = share of the objects above the 90th percentile of the same predictor at the B_rand pairs of the same donors (false-positive rate 10 percent by construction), 95 percent intervals over donors (500 resamples, the paper tables used 500 too). The old rows (PAPER) are recomputed from the same score table and reproduce the registry-72 values; the intervals differ in the third decimal because the resampling stream differs. The release checkpoint (`all12one_fullrd_10kb` view) has seen no AML sample; the lines of the table correspond to the paper numbers as follows:

- RELEASE, AML reference input <-> `base_refin@aml` (paper: 48 % of the loops, 67 % of the stripes): the model without AML in training that reads the AML reference map of the fold's training patients as Hi-C input (AML data used as input only).
- RELEASE, consensus12 <-> `c12_refin`, `c12_refres`, `c12_diffcells` (paper: 40 to 42 % of the loops, 52 to 57 % of the stripes): reference = mean map of the 12 cell lines, no AML data anywhere.
- RELEASE, marks only <-> the paper sentence "from the marks alone, without any reference, the model detected 28 % of the loops and 10 % of the stripes". I could not find the source of 28 % / 10 % in the result files (the `aml_refin|none` columns give 0.48 / 0.68 above their own B_rand threshold and 0.34 / 0.58 as the share above the reference in `aml_train_v2`); the like-for-like comparison is therefore the dedicated no-reference model all12honest_drop_10kb, run here on the same windows and scored in the same table (row "PAPER-model family, no reference").
- The AML-trained rows are listed for completeness; the release checkpoint was not trained on AML data, so there is no release counterpart. The paper's AML-inclusive result (Supplementary Fig. S4, Tables S7, S16) is not affected.

**Detection rate** (loops / stripes; all, private, recurrent; level of the predicted enrichment)

| predictor | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|
| truth (measured patient map) | 0.61 (0.55-0.68) | 0.58 (0.52-0.66) | 0.85 (0.81-0.90) | 0.76 (0.72-0.81) | 0.64 (0.56-0.71) | 0.83 (0.77-0.86) |
| AML reference map of the fold (other patients) | 0.44 (0.35-0.52) | 0.37 (0.29-0.47) | 0.84 (0.80-0.89) | 0.68 (0.63-0.76) | 0.48 (0.41-0.61) | 0.80 (0.74-0.86) |
| consensus12 (12 cell lines) | 0.38 (0.29-0.48) | 0.34 (0.26-0.44) | 0.65 (0.54-0.72) | 0.55 (0.47-0.61) | 0.41 (0.33-0.48) | 0.62 (0.53-0.69) |
| PAPER 12-cell model, AML reference input (all12refin_drop) | 0.48 (0.39-0.56) | 0.42 (0.34-0.50) | 0.85 (0.80-0.91) | 0.67 (0.63-0.73) | 0.49 (0.43-0.59) | 0.77 (0.74-0.83) |
| RELEASE, AML reference input | 0.49 (0.38-0.60) | 0.43 (0.35-0.52) | 0.85 (0.79-0.91) | 0.69 (0.65-0.73) | 0.52 (0.46-0.56) | 0.79 (0.74-0.83) |
| PAPER 12-cell refin, consensus12 | 0.42 (0.34-0.50) | 0.38 (0.31-0.46) | 0.64 (0.57-0.74) | 0.56 (0.49-0.62) | 0.44 (0.35-0.51) | 0.62 (0.54-0.69) |
| PAPER 12-cell refres, consensus12 | 0.42 (0.34-0.52) | 0.38 (0.29-0.47) | 0.65 (0.58-0.73) | 0.57 (0.49-0.65) | 0.44 (0.36-0.53) | 0.63 (0.54-0.72) |
| PAPER 12-cell differential, consensus12 | 0.40 (0.32-0.50) | 0.37 (0.29-0.47) | 0.62 (0.52-0.71) | 0.52 (0.46-0.59) | 0.41 (0.33-0.51) | 0.58 (0.50-0.64) |
| RELEASE, consensus12 | 0.42 (0.33-0.51) | 0.38 (0.30-0.46) | 0.63 (0.53-0.73) | 0.54 (0.46-0.61) | 0.43 (0.33-0.52) | 0.59 (0.49-0.67) |
| PAPER-model family, no reference (all12honest_drop, run here) | 0.29 (0.25-0.33) | 0.27 (0.23-0.31) | 0.35 (0.28-0.43) | 0.25 (0.19-0.32) | 0.23 (0.17-0.31) | 0.25 (0.18-0.33) |
| RELEASE, marks only (reference plane 0) | 0.30 (0.24-0.35) | 0.28 (0.23-0.33) | 0.37 (0.29-0.44) | 0.29 (0.24-0.35) | 0.27 (0.22-0.32) | 0.30 (0.23-0.37) |
| Akita (sequence only) | 0.34 (0.26-0.43) | 0.31 (0.23-0.40) | 0.56 (0.49-0.60) | 0.53 (0.47-0.57) | 0.35 (0.27-0.45) | 0.65 (0.55-0.70) |
| C.Origami, quantile-matched | 0.21 (0.19-0.24) | 0.19 (0.16-0.22) | 0.36 (0.30-0.41) | 0.19 (0.14-0.21) | 0.14 (0.10-0.19) | 0.22 (0.18-0.24) |
| C.Origami, as given | 0.13 (0.11-0.14) | 0.13 (0.11-0.14) | 0.12 (0.06-0.17) | 0.17 (0.14-0.20) | 0.16 (0.10-0.20) | 0.19 (0.15-0.22) |
| Epiphany, H3K4me3 = 0 | 0.19 (0.14-0.23) | 0.19 (0.15-0.22) | 0.18 (0.07-0.29) | 0.21 (0.16-0.27) | 0.20 (0.14-0.27) | 0.26 (0.16-0.34) |
| Epiphany, H3K4me3 = mean | 0.18 (0.14-0.22) | 0.18 (0.15-0.22) | 0.19 (0.10-0.30) | 0.22 (0.16-0.29) | 0.20 (0.14-0.26) | 0.28 (0.16-0.38) |
| (trained with AML patients, not comparable) aml_refin | 0.49 (0.40-0.61) | 0.42 (0.33-0.54) | 0.89 (0.84-0.93) | 0.70 (0.67-0.74) | 0.53 (0.47-0.59) | 0.80 (0.76-0.83) |
| (trained with AML) aml_diff | 0.51 (0.40-0.61) | 0.44 (0.34-0.55) | 0.88 (0.85-0.92) | 0.69 (0.66-0.74) | 0.51 (0.46-0.57) | 0.79 (0.76-0.84) |

**AUROC of the predicted level, AML-specific objects against B_rand**

| predictor | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|
| truth (measured patient map) | 0.89 (0.87-0.91) | 0.88 (0.87-0.90) | 0.95 (0.94-0.96) | 0.92 (0.90-0.93) | 0.88 (0.85-0.90) | 0.94 (0.92-0.95) |
| AML reference map of the fold (other patients) | 0.76 (0.72-0.81) | 0.73 (0.68-0.78) | 0.94 (0.92-0.96) | 0.88 (0.86-0.90) | 0.80 (0.76-0.83) | 0.93 (0.92-0.94) |
| consensus12 (12 cell lines) | 0.76 (0.70-0.81) | 0.74 (0.68-0.78) | 0.89 (0.87-0.91) | 0.84 (0.81-0.86) | 0.75 (0.71-0.79) | 0.88 (0.85-0.90) |
| PAPER 12-cell model, AML reference input (all12refin_drop) | 0.80 (0.75-0.84) | 0.77 (0.72-0.81) | 0.95 (0.93-0.96) | 0.89 (0.87-0.90) | 0.81 (0.77-0.85) | 0.92 (0.91-0.94) |
| RELEASE, AML reference input | 0.80 (0.75-0.84) | 0.77 (0.72-0.82) | 0.94 (0.93-0.96) | 0.89 (0.87-0.90) | 0.82 (0.77-0.85) | 0.92 (0.91-0.94) |
| PAPER 12-cell refin, consensus12 | 0.78 (0.73-0.83) | 0.76 (0.71-0.81) | 0.90 (0.88-0.92) | 0.84 (0.81-0.87) | 0.77 (0.72-0.82) | 0.88 (0.85-0.91) |
| PAPER 12-cell refres, consensus12 | 0.78 (0.73-0.82) | 0.76 (0.71-0.81) | 0.90 (0.88-0.93) | 0.85 (0.82-0.87) | 0.78 (0.72-0.82) | 0.89 (0.86-0.91) |
| PAPER 12-cell differential, consensus12 | 0.78 (0.72-0.82) | 0.75 (0.70-0.80) | 0.89 (0.87-0.92) | 0.83 (0.80-0.85) | 0.76 (0.71-0.80) | 0.87 (0.84-0.89) |
| RELEASE, consensus12 | 0.78 (0.73-0.83) | 0.76 (0.71-0.80) | 0.90 (0.88-0.92) | 0.84 (0.81-0.86) | 0.77 (0.73-0.81) | 0.87 (0.85-0.90) |
| PAPER-model family, no reference (all12honest_drop, run here) | 0.66 (0.63-0.69) | 0.65 (0.61-0.68) | 0.70 (0.66-0.73) | 0.66 (0.63-0.70) | 0.65 (0.60-0.70) | 0.66 (0.61-0.71) |
| RELEASE, marks only (reference plane 0) | 0.67 (0.64-0.70) | 0.66 (0.63-0.68) | 0.70 (0.67-0.73) | 0.68 (0.65-0.72) | 0.67 (0.62-0.71) | 0.68 (0.64-0.72) |
| Akita (sequence only) | 0.74 (0.68-0.79) | 0.72 (0.67-0.77) | 0.82 (0.79-0.85) | 0.84 (0.80-0.86) | 0.76 (0.70-0.82) | 0.88 (0.86-0.90) |
| C.Origami, quantile-matched | 0.60 (0.58-0.62) | 0.59 (0.57-0.61) | 0.67 (0.62-0.72) | 0.60 (0.58-0.62) | 0.57 (0.54-0.61) | 0.63 (0.62-0.64) |
| C.Origami, as given | 0.53 (0.51-0.55) | 0.53 (0.51-0.55) | 0.51 (0.47-0.53) | 0.60 (0.56-0.63) | 0.60 (0.55-0.63) | 0.61 (0.56-0.65) |
| Epiphany, H3K4me3 = 0 | 0.56 (0.51-0.59) | 0.56 (0.52-0.60) | 0.57 (0.51-0.64) | 0.62 (0.57-0.68) | 0.63 (0.59-0.68) | 0.63 (0.54-0.71) |
| Epiphany, H3K4me3 = mean | 0.55 (0.51-0.59) | 0.55 (0.52-0.59) | 0.57 (0.50-0.63) | 0.61 (0.56-0.67) | 0.63 (0.60-0.68) | 0.63 (0.55-0.71) |
| (trained with AML patients, not comparable) aml_refin | 0.81 (0.76-0.85) | 0.78 (0.74-0.82) | 0.95 (0.93-0.96) | 0.89 (0.87-0.90) | 0.81 (0.78-0.84) | 0.93 (0.91-0.94) |
| (trained with AML) aml_diff | 0.81 (0.76-0.85) | 0.78 (0.73-0.83) | 0.95 (0.93-0.96) | 0.89 (0.87-0.90) | 0.81 (0.78-0.83) | 0.92 (0.91-0.93) |

**AUROC against the A_refrand controls** (matched on AML reference and HSPC enrichment, distance, orientation)

| predictor | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|
| truth (measured patient map) | 0.85 (0.83-0.87) | 0.85 (0.83-0.87) | 0.80 (0.76-0.83) | 0.67 (0.64-0.70) | 0.71 (0.66-0.75) | 0.63 (0.58-0.67) |
| AML reference map of the fold (other patients) | 0.55 (0.54-0.56) | 0.55 (0.53-0.56) | 0.57 (0.55-0.59) | 0.51 (0.50-0.53) | 0.50 (0.49-0.51) | 0.53 (0.52-0.54) |
| consensus12 (12 cell lines) | 0.63 (0.60-0.66) | 0.63 (0.60-0.66) | 0.56 (0.50-0.62) | 0.57 (0.52-0.62) | 0.56 (0.47-0.63) | 0.57 (0.49-0.65) |
| PAPER 12-cell model, AML reference input (all12refin_drop) | 0.64 (0.61-0.66) | 0.63 (0.60-0.65) | 0.60 (0.56-0.65) | 0.55 (0.52-0.58) | 0.57 (0.53-0.60) | 0.54 (0.51-0.57) |
| RELEASE, AML reference input | 0.65 (0.62-0.67) | 0.64 (0.61-0.66) | 0.60 (0.55-0.65) | 0.56 (0.53-0.59) | 0.58 (0.54-0.62) | 0.56 (0.52-0.60) |
| PAPER 12-cell refin, consensus12 | 0.67 (0.64-0.70) | 0.67 (0.63-0.70) | 0.58 (0.50-0.66) | 0.59 (0.54-0.63) | 0.60 (0.52-0.67) | 0.57 (0.48-0.66) |
| PAPER 12-cell refres, consensus12 | 0.66 (0.64-0.69) | 0.66 (0.63-0.69) | 0.57 (0.52-0.65) | 0.60 (0.55-0.65) | 0.61 (0.52-0.68) | 0.59 (0.51-0.68) |
| PAPER 12-cell differential, consensus12 | 0.66 (0.62-0.68) | 0.66 (0.62-0.68) | 0.57 (0.49-0.66) | 0.56 (0.51-0.61) | 0.59 (0.51-0.66) | 0.53 (0.47-0.61) |
| RELEASE, consensus12 | 0.66 (0.63-0.69) | 0.66 (0.62-0.69) | 0.56 (0.48-0.64) | 0.59 (0.53-0.64) | 0.60 (0.53-0.67) | 0.57 (0.49-0.66) |
| PAPER-model family, no reference (all12honest_drop, run here) | 0.59 (0.56-0.62) | 0.59 (0.56-0.62) | 0.57 (0.52-0.65) | 0.51 (0.46-0.56) | 0.55 (0.48-0.61) | 0.45 (0.37-0.56) |
| RELEASE, marks only (reference plane 0) | 0.60 (0.57-0.63) | 0.60 (0.56-0.63) | 0.53 (0.47-0.61) | 0.51 (0.45-0.57) | 0.57 (0.49-0.64) | 0.45 (0.37-0.56) |
| Akita (sequence only) | 0.63 (0.60-0.67) | 0.63 (0.59-0.67) | 0.56 (0.50-0.63) | 0.68 (0.64-0.72) | 0.65 (0.58-0.74) | 0.74 (0.68-0.81) |
| C.Origami, quantile-matched | 0.56 (0.54-0.58) | 0.56 (0.53-0.59) | 0.59 (0.52-0.66) | 0.53 (0.48-0.58) | 0.54 (0.48-0.63) | 0.54 (0.44-0.64) |
| C.Origami, as given | 0.51 (0.49-0.54) | 0.51 (0.48-0.54) | 0.42 (0.33-0.52) | 0.52 (0.46-0.59) | 0.57 (0.51-0.64) | 0.48 (0.39-0.58) |
| Epiphany, H3K4me3 = 0 | 0.53 (0.49-0.56) | 0.53 (0.49-0.56) | 0.45 (0.39-0.52) | 0.49 (0.46-0.53) | 0.55 (0.47-0.63) | 0.49 (0.41-0.56) |
| Epiphany, H3K4me3 = mean | 0.52 (0.49-0.55) | 0.53 (0.49-0.56) | 0.44 (0.39-0.51) | 0.50 (0.45-0.54) | 0.56 (0.48-0.63) | 0.50 (0.40-0.61) |
| (trained with AML patients, not comparable) aml_refin | 0.67 (0.64-0.68) | 0.66 (0.63-0.68) | 0.64 (0.60-0.69) | 0.55 (0.53-0.57) | 0.55 (0.53-0.58) | 0.55 (0.53-0.58) |
| (trained with AML) aml_diff | 0.67 (0.64-0.69) | 0.66 (0.63-0.68) | 0.62 (0.58-0.67) | 0.56 (0.53-0.59) | 0.56 (0.53-0.59) | 0.57 (0.53-0.61) |

**Spearman of the predicted with the measured enrichment among the AML-specific objects** (level)

| predictor | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|
| AML reference map of the fold (other patients) | 0.56 (0.49-0.65) | 0.51 (0.44-0.60) | 0.70 (0.61-0.77) | 0.79 (0.73-0.85) | 0.76 (0.68-0.85) | 0.80 (0.75-0.84) |
| consensus12 (12 cell lines) | 0.51 (0.41-0.57) | 0.49 (0.40-0.59) | 0.40 (0.31-0.52) | 0.70 (0.61-0.77) | 0.71 (0.64-0.77) | 0.66 (0.55-0.76) |
| PAPER 12-cell model, AML reference input (all12refin_drop) | 0.56 (0.47-0.64) | 0.52 (0.43-0.59) | 0.64 (0.55-0.71) | 0.81 (0.75-0.85) | 0.77 (0.71-0.84) | 0.81 (0.76-0.86) |
| RELEASE, AML reference input | 0.55 (0.45-0.63) | 0.50 (0.42-0.59) | 0.63 (0.50-0.72) | 0.81 (0.76-0.87) | 0.77 (0.72-0.84) | 0.82 (0.75-0.87) |
| PAPER 12-cell refin, consensus12 | 0.49 (0.41-0.57) | 0.48 (0.39-0.57) | 0.36 (0.22-0.51) | 0.71 (0.62-0.78) | 0.71 (0.65-0.78) | 0.68 (0.57-0.78) |
| PAPER 12-cell refres, consensus12 | 0.51 (0.43-0.58) | 0.49 (0.41-0.58) | 0.39 (0.27-0.50) | 0.70 (0.62-0.78) | 0.70 (0.64-0.78) | 0.67 (0.56-0.76) |
| PAPER 12-cell differential, consensus12 | 0.51 (0.43-0.59) | 0.49 (0.40-0.60) | 0.36 (0.24-0.49) | 0.70 (0.62-0.77) | 0.70 (0.64-0.77) | 0.69 (0.58-0.79) |
| RELEASE, consensus12 | 0.49 (0.40-0.56) | 0.48 (0.38-0.57) | 0.36 (0.23-0.49) | 0.70 (0.60-0.77) | 0.70 (0.63-0.78) | 0.68 (0.57-0.79) |
| PAPER-model family, no reference (all12honest_drop, run here) | 0.21 (0.15-0.27) | 0.23 (0.16-0.27) | 0.00 (-0.17-0.15) | 0.39 (0.31-0.48) | 0.40 (0.30-0.53) | 0.43 (0.31-0.53) |
| RELEASE, marks only (reference plane 0) | 0.20 (0.13-0.25) | 0.20 (0.14-0.25) | -0.01 (-0.16-0.12) | 0.42 (0.34-0.50) | 0.45 (0.34-0.55) | 0.44 (0.33-0.53) |
| Akita (sequence only) | 0.41 (0.34-0.49) | 0.40 (0.31-0.48) | 0.23 (0.06-0.39) | 0.33 (0.22-0.45) | 0.41 (0.26-0.53) | 0.22 (0.10-0.35) |
| C.Origami, quantile-matched | 0.10 (0.05-0.14) | 0.09 (0.03-0.15) | 0.02 (-0.07-0.11) | 0.14 (0.09-0.19) | 0.20 (0.15-0.28) | 0.09 (0.01-0.18) |
| C.Origami, as given | 0.05 (-0.02-0.12) | 0.05 (-0.03-0.12) | 0.10 (-0.07-0.26) | 0.09 (0.03-0.15) | 0.07 (-0.02-0.18) | 0.08 (0.03-0.14) |
| Epiphany, H3K4me3 = 0 | 0.12 (0.08-0.16) | 0.10 (0.05-0.17) | 0.13 (-0.13-0.35) | 0.13 (0.03-0.24) | 0.10 (-0.08-0.27) | 0.17 (0.09-0.26) |
| Epiphany, H3K4me3 = mean | 0.11 (0.07-0.16) | 0.10 (0.05-0.17) | 0.13 (-0.13-0.32) | 0.12 (0.02-0.26) | 0.09 (-0.07-0.27) | 0.16 (0.07-0.26) |
| (trained with AML patients, not comparable) aml_refin | 0.55 (0.46-0.62) | 0.50 (0.42-0.57) | 0.61 (0.51-0.69) | 0.80 (0.76-0.84) | 0.78 (0.73-0.83) | 0.80 (0.74-0.85) |
| (trained with AML) aml_diff | 0.55 (0.46-0.63) | 0.50 (0.42-0.58) | 0.64 (0.54-0.71) | 0.79 (0.74-0.85) | 0.77 (0.72-0.84) | 0.79 (0.72-0.85) |

**Contrast with HSPC** (score = prediction minus the same predictor on HSPC CD34+ inputs; truth, reference, consensus12 and AML-trained rows minus the measured HSPC consensus): detection rate

| predictor | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|
| truth (measured patient map) | 0.54 (0.51-0.60) | 0.54 (0.49-0.59) | 0.64 (0.57-0.70) | 0.46 (0.39-0.52) | 0.49 (0.39-0.59) | 0.43 (0.35-0.49) |
| AML reference map of the fold (other patients) | 0.32 (0.30-0.35) | 0.27 (0.25-0.30) | 0.66 (0.61-0.74) | 0.26 (0.23-0.30) | 0.21 (0.15-0.28) | 0.29 (0.25-0.32) |
| consensus12 (12 cell lines) | 0.29 (0.25-0.33) | 0.28 (0.23-0.31) | 0.33 (0.25-0.43) | 0.09 (0.07-0.11) | 0.11 (0.07-0.15) | 0.08 (0.06-0.09) |
| PAPER 12-cell model, AML reference input (all12refin_drop) | 0.42 (0.38-0.45) | 0.36 (0.33-0.39) | 0.77 (0.71-0.82) | 0.23 (0.17-0.30) | 0.24 (0.18-0.31) | 0.21 (0.16-0.28) |
| RELEASE, AML reference input | 0.38 (0.34-0.41) | 0.34 (0.31-0.37) | 0.61 (0.55-0.68) | 0.17 (0.13-0.21) | 0.21 (0.14-0.26) | 0.14 (0.11-0.18) |
| PAPER 12-cell refin, consensus12 | 0.32 (0.26-0.38) | 0.31 (0.25-0.37) | 0.33 (0.28-0.37) | 0.14 (0.11-0.17) | 0.19 (0.13-0.24) | 0.11 (0.07-0.15) |
| PAPER 12-cell refres, consensus12 | 0.33 (0.27-0.38) | 0.32 (0.26-0.38) | 0.34 (0.26-0.43) | 0.21 (0.15-0.26) | 0.22 (0.14-0.28) | 0.19 (0.13-0.24) |
| PAPER 12-cell differential, consensus12 | 0.27 (0.21-0.33) | 0.26 (0.20-0.32) | 0.32 (0.22-0.42) | 0.15 (0.12-0.20) | 0.17 (0.11-0.27) | 0.12 (0.10-0.15) |
| RELEASE, consensus12 | 0.32 (0.26-0.38) | 0.31 (0.24-0.37) | 0.34 (0.27-0.40) | 0.15 (0.12-0.18) | 0.19 (0.13-0.24) | 0.11 (0.08-0.15) |
| PAPER-model family, no reference (all12honest_drop, run here) | 0.16 (0.11-0.21) | 0.16 (0.12-0.20) | 0.13 (0.07-0.20) | 0.09 (0.07-0.13) | 0.12 (0.07-0.20) | 0.08 (0.05-0.12) |
| RELEASE, marks only (reference plane 0) | 0.18 (0.14-0.23) | 0.18 (0.13-0.22) | 0.15 (0.10-0.20) | 0.11 (0.09-0.13) | 0.13 (0.07-0.19) | 0.09 (0.06-0.13) |
| C.Origami, quantile-matched | 0.19 (0.15-0.23) | 0.17 (0.13-0.21) | 0.34 (0.27-0.38) | 0.15 (0.11-0.18) | 0.13 (0.09-0.17) | 0.17 (0.13-0.20) |
| C.Origami, as given | 0.09 (0.07-0.12) | 0.09 (0.07-0.12) | 0.07 (0.04-0.12) | 0.01 (0.01-0.02) | 0.03 (0.01-0.05) | 0.01 (0.00-0.01) |
| Epiphany, H3K4me3 = 0 | 0.12 (0.09-0.15) | 0.13 (0.10-0.15) | 0.12 (0.06-0.17) | 0.11 (0.07-0.16) | 0.11 (0.08-0.15) | 0.12 (0.07-0.18) |
| Epiphany, H3K4me3 = mean | 0.12 (0.09-0.15) | 0.12 (0.10-0.15) | 0.11 (0.06-0.19) | 0.09 (0.05-0.13) | 0.08 (0.06-0.13) | 0.10 (0.05-0.15) |
| (trained with AML patients, not comparable) aml_refin | 0.40 (0.36-0.43) | 0.34 (0.31-0.38) | 0.71 (0.64-0.80) | 0.25 (0.20-0.30) | 0.26 (0.21-0.32) | 0.25 (0.20-0.29) |
| (trained with AML) aml_diff | 0.43 (0.39-0.48) | 0.38 (0.34-0.42) | 0.71 (0.66-0.76) | 0.28 (0.22-0.33) | 0.28 (0.22-0.35) | 0.26 (0.20-0.33) |

AUROC of the contrast against A_refrand

| predictor | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|
| truth (measured patient map) | 0.87 (0.86-0.89) | 0.88 (0.86-0.90) | 0.88 (0.84-0.91) | 0.74 (0.70-0.77) | 0.77 (0.71-0.82) | 0.65 (0.60-0.70) |
| AML reference map of the fold (other patients) | 0.58 (0.57-0.59) | 0.57 (0.56-0.58) | 0.65 (0.62-0.68) | 0.51 (0.49-0.53) | 0.47 (0.45-0.49) | 0.52 (0.48-0.56) |
| consensus12 (12 cell lines) | 0.66 (0.64-0.69) | 0.66 (0.63-0.69) | 0.61 (0.54-0.68) | 0.59 (0.53-0.65) | 0.57 (0.46-0.66) | 0.57 (0.49-0.65) |
| PAPER 12-cell model, AML reference input (all12refin_drop) | 0.67 (0.64-0.68) | 0.66 (0.64-0.68) | 0.70 (0.64-0.76) | 0.57 (0.52-0.62) | 0.59 (0.52-0.64) | 0.53 (0.47-0.58) |
| RELEASE, AML reference input | 0.67 (0.65-0.69) | 0.67 (0.64-0.68) | 0.68 (0.61-0.74) | 0.59 (0.53-0.64) | 0.59 (0.53-0.65) | 0.55 (0.50-0.62) |
| PAPER 12-cell refin, consensus12 | 0.62 (0.57-0.66) | 0.61 (0.57-0.65) | 0.57 (0.49-0.66) | 0.57 (0.52-0.62) | 0.62 (0.53-0.71) | 0.51 (0.48-0.53) |
| PAPER 12-cell refres, consensus12 | 0.63 (0.59-0.67) | 0.63 (0.57-0.66) | 0.59 (0.50-0.68) | 0.61 (0.56-0.64) | 0.66 (0.57-0.75) | 0.54 (0.49-0.59) |
| PAPER 12-cell differential, consensus12 | 0.59 (0.55-0.63) | 0.59 (0.54-0.63) | 0.60 (0.47-0.73) | 0.54 (0.48-0.59) | 0.60 (0.51-0.71) | 0.45 (0.40-0.50) |
| RELEASE, consensus12 | 0.62 (0.57-0.66) | 0.61 (0.56-0.65) | 0.54 (0.47-0.65) | 0.56 (0.52-0.61) | 0.61 (0.54-0.68) | 0.49 (0.42-0.54) |
| PAPER-model family, no reference (all12honest_drop, run here) | 0.49 (0.46-0.51) | 0.48 (0.45-0.50) | 0.47 (0.38-0.56) | 0.47 (0.42-0.51) | 0.49 (0.41-0.56) | 0.46 (0.39-0.50) |
| RELEASE, marks only (reference plane 0) | 0.52 (0.49-0.55) | 0.52 (0.48-0.54) | 0.47 (0.37-0.59) | 0.48 (0.42-0.52) | 0.53 (0.44-0.63) | 0.44 (0.40-0.48) |
| C.Origami, quantile-matched | 0.54 (0.51-0.58) | 0.54 (0.50-0.57) | 0.58 (0.51-0.66) | 0.55 (0.49-0.60) | 0.56 (0.46-0.67) | 0.59 (0.50-0.68) |
| C.Origami, as given | 0.52 (0.50-0.53) | 0.52 (0.49-0.54) | 0.43 (0.36-0.50) | 0.52 (0.47-0.56) | 0.54 (0.49-0.59) | 0.48 (0.40-0.56) |
| Epiphany, H3K4me3 = 0 | 0.51 (0.49-0.54) | 0.51 (0.48-0.55) | 0.46 (0.39-0.56) | 0.46 (0.41-0.49) | 0.51 (0.46-0.57) | 0.47 (0.42-0.52) |
| Epiphany, H3K4me3 = mean | 0.51 (0.48-0.53) | 0.51 (0.48-0.54) | 0.46 (0.40-0.54) | 0.44 (0.41-0.47) | 0.51 (0.46-0.58) | 0.46 (0.41-0.51) |
| (trained with AML patients, not comparable) aml_refin | 0.69 (0.67-0.70) | 0.68 (0.67-0.69) | 0.75 (0.69-0.80) | 0.58 (0.56-0.60) | 0.58 (0.53-0.62) | 0.57 (0.53-0.59) |
| (trained with AML) aml_diff | 0.70 (0.68-0.71) | 0.69 (0.67-0.71) | 0.72 (0.66-0.78) | 0.58 (0.53-0.62) | 0.57 (0.52-0.61) | 0.56 (0.50-0.62) |

Spearman of the predicted gain with the measured gain (all loops / all stripes)

| predictor | loops all | stripes all |
|---|---|---|
| AML reference map of the fold (other patients) | 0.29 (0.23-0.36) | 0.46 (0.36-0.56) |
| consensus12 (12 cell lines) | 0.22 (0.15-0.30) | 0.32 (0.18-0.44) |
| PAPER 12-cell model, AML reference input (all12refin_drop) | 0.30 (0.22-0.38) | 0.49 (0.43-0.55) |
| RELEASE, AML reference input | 0.29 (0.20-0.37) | 0.51 (0.43-0.57) |
| PAPER 12-cell refin, consensus12 | 0.09 (0.03-0.14) | 0.23 (0.17-0.28) |
| PAPER 12-cell refres, consensus12 | 0.10 (0.05-0.15) | 0.23 (0.14-0.35) |
| PAPER 12-cell differential, consensus12 | 0.09 (0.05-0.13) | 0.23 (0.17-0.27) |
| RELEASE, consensus12 | 0.07 (0.02-0.12) | 0.27 (0.17-0.34) |
| PAPER-model family, no reference (all12honest_drop, run here) | 0.04 (-0.01-0.07) | 0.02 (-0.03-0.08) |
| RELEASE, marks only (reference plane 0) | 0.04 (0.01-0.07) | 0.06 (-0.01-0.13) |
| C.Origami, quantile-matched | 0.01 (-0.06-0.07) | 0.02 (-0.04-0.10) |
| C.Origami, as given | 0.09 (0.04-0.13) | 0.10 (0.02-0.19) |
| Epiphany, H3K4me3 = 0 | 0.05 (-0.01-0.09) | 0.00 (-0.08-0.10) |
| Epiphany, H3K4me3 = mean | 0.05 (-0.01-0.09) | -0.01 (-0.08-0.11) |
| (trained with AML patients, not comparable) aml_refin | 0.28 (0.20-0.35) | 0.50 (0.43-0.59) |
| (trained with AML) aml_diff | 0.26 (0.17-0.35) | 0.49 (0.37-0.60) |

**Structure measures in the object windows of the eight patients with CTCF** (Supplementary Table S13; median over patients, range in brackets; identical pixels for all methods)

| method | pearson | ins_r_50k | bnd_f1 | compartment_corr | moc | stripe_r | loop_enrich_corr |
|---|---|---|---|---|---|---|---|
| consensus12 | 0.746 (0.703-0.842) | 0.738 (0.658-0.845) | 0.500 (0.444-0.667) | 0.810 (0.760-0.898) | 0.653 (0.624-0.721) | 0.718 (0.644-0.843) | 0.424 (0.348-0.514) |
| PAPER 12-cell refin | 0.762 (0.712-0.839) | 0.741 (0.665-0.841) | 0.500 (0.444-0.667) | 0.823 (0.768-0.898) | 0.649 (0.628-0.721) | 0.726 (0.649-0.842) | 0.443 (0.367-0.547) |
| PAPER 12-cell refres | 0.761 (0.711-0.839) | 0.735 (0.666-0.841) | 0.500 (0.444-0.667) | 0.821 (0.766-0.895) | 0.651 (0.628-0.722) | 0.724 (0.654-0.843) | 0.441 (0.366-0.545) |
| RELEASE, consensus12 | 0.758 (0.706-0.835) | 0.729 (0.655-0.832) | 0.500 (0.444-0.667) | 0.818 (0.762-0.892) | 0.649 (0.624-0.725) | 0.710 (0.645-0.835) | 0.445 (0.363-0.548) |
| RELEASE, marks only | 0.488 (0.412-0.619) | 0.413 (0.322-0.564) | 0.321 (0.250-0.444) | 0.664 (0.608-0.727) | 0.574 (0.557-0.631) | 0.394 (0.290-0.508) | 0.239 (0.177-0.329) |
| RELEASE, AML reference | 0.841 (0.773-0.881) | 0.856 (0.803-0.904) | 0.545 (0.500-0.667) | 0.890 (0.818-0.917) | 0.669 (0.629-0.732) | 0.883 (0.836-0.919) | 0.493 (0.408-0.578) |
| PAPER-model family, no reference | 0.461 (0.382-0.598) | 0.446 (0.328-0.587) | 0.308 (0.250-0.400) | 0.660 (0.605-0.721) | 0.578 (0.558-0.634) | 0.433 (0.320-0.550) | 0.227 (0.165-0.310) |
| Akita | 0.591 (0.547-0.646) | 0.641 (0.557-0.747) | 0.400 (0.400-0.545) | 0.653 (0.637-0.703) | 0.621 (0.603-0.683) | 0.635 (0.536-0.759) | 0.334 (0.288-0.423) |
| C.Origami (q) | 0.188 (0.136-0.240) | 0.190 (0.120-0.261) | 0.222 (0.200-0.250) | 0.575 (0.559-0.602) | 0.541 (0.524-0.560) | 0.144 (0.035-0.174) | 0.077 (0.056-0.098) |
| Epiphany (mean) | 0.248 (0.090-0.379) | 0.342 (0.077-0.423) | 0.000 (0.000-0.250) | 0.578 (0.548-0.616) | 0.468 (0.441-0.510) | 0.385 (0.083-0.474) | 0.068 (0.035-0.127) |
| Epiphany (zero) | 0.233 (0.099-0.368) | 0.357 (0.077-0.412) | 0.000 (0.000-0.250) | 0.577 (0.546-0.616) | 0.485 (0.449-0.535) | 0.375 (0.073-0.443) | 0.067 (0.034-0.129) |

Patients (of 8) in which the release row is ahead, per measure

| release row | against | pearson | scc | ins_r_50k | ins_r_100k | ins_r_250k | bnd_f1 | moc | stripe_r | loop_enrich_corr | compartment_corr | eigenvector_abs_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RELEASE, consensus12 | Akita | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| RELEASE, consensus12 | C.Origami (q) | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| RELEASE, consensus12 | Epiphany (mean) | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| RELEASE, consensus12 | Epiphany (zero) | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| RELEASE, consensus12 | consensus12 | 7/8 | 7/8 | 2/8 | 5/8 | 6/8 | 1/8 | 3/8 | 3/8 | 8/8 | 6/8 | 5/8 |
| RELEASE, marks only | Akita | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 3/8 | 8/8 |
| RELEASE, marks only | C.Origami (q) | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| RELEASE, marks only | Epiphany (mean) | 8/8 | 8/8 | 6/8 | 8/8 | 8/8 | 8/8 | 8/8 | 5/8 | 8/8 | 8/8 | 8/8 |
| RELEASE, marks only | Epiphany (zero) | 8/8 | 8/8 | 7/8 | 8/8 | 8/8 | 8/8 | 8/8 | 5/8 | 8/8 | 8/8 | 8/8 |
| RELEASE, marks only | consensus12 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

### Sentences of main.tex that change (Results "In held-out AML patients, the model detects disease-specific loops and stripes", Fig. 6, Supplementary Table S13; Discussion)

Setting correspondence (see the section text): "reading the mean map of the other patients" = RELEASE, AML reference input; "when the consensus of the 12 cell lines replaced the patient reference" = RELEASE, consensus12; "from the marks alone, without any reference" = RELEASE, marks only. The release checkpoint has seen no AML sample in any of them.

1. "Our 12-cell model was trained without any AML data." becomes "The release checkpoint was trained without any AML data (12 cell lines)." The sentence about the three marks stays.
2. "Given the three marks of a patient together with the mean map of the other patients, it detected 48% of the 2276 AML-specific loops and 67% of the 1075 stripes" becomes "49% ... and 69%" (loops 0.490 [0.38, 0.60], stripes 0.693 [0.65, 0.73]; paper model 0.478 and 0.674; recurrent loops 0.855, private loops 0.433 against 0.418).
3. "The published predictors detected fewer: Akita 34% of the loops and 53% of the stripes, C.Origami at most 21% and Epiphany at most 22% of the loops." and "The measured patient maps detected 61% of the loops and 76% of the stripes" are unchanged (competitor and truth rows are the same).
4. "When the consensus of the 12 cell lines replaced the patient reference, the model still detected 40% to 42% of the loops and 52% to 57% of the stripes" becomes "42% of the loops and 54% of the stripes" (0.417 and 0.537; the three dedicated models gave 0.403 to 0.420 and 0.523 to 0.566). A single checkpoint replaces the range.
5. "Against these controls, our models reached an AUROC of 0.64 to 0.67, and the measured maps 0.85" becomes "0.65 to 0.66" (release 0.645 with the AML reference, 0.660 with consensus12; dedicated 0.636 to 0.668); measured maps 0.851.
6. "Around the AML-specific loops and stripes, the model was also more accurate than Akita, C.Origami and Epiphany in every structure measure for all eight patients with CTCF data (Supplementary Table S13)" stays true for the release checkpoint with consensus12: it is ahead in 8 of 8 patients on 13 of 14 measures including Pearson, SCC, insulation at 50, 100, 250 kb, boundary F1, boundary strength, MoC, stripe, loop enrichment, compartment and eigenvector correlation (the 14th, `bnd_f1_struct`, is 0 or 1 for most donors and is 5 of 8, as in the paper). Table S13 values (median over patients): Pearson 0.758 (paper model refres 0.762), insulation 50 kb 0.729 (0.735), boundary F1 0.500 (0.500), compartment 0.818 (0.821). Against consensus12 the release model is ahead in 7 of 8 patients on SCC and 3 of 8 on the stripe correlation (the paper model refres: 8 of 8 and 7 of 8), and in 1 of 8 on boundary F1 (paper model 0 of 8); the statement "no gain over the reference map at TAD level" of registry 73 therefore stays.
7. "From the marks alone, without any reference, the model detected 28% of the loops and 10% of the stripes, fewer than Akita." The release checkpoint with the reference plane at 0 detects 30% of the loops and 29% of the stripes (0.302 [0.24, 0.35], 0.293 [0.24, 0.35]); the dedicated no-reference model on the same objects detects 29% and 25%. I could not reproduce or locate the paper's 28% / 10% (the stripe value in particular); both no-reference models are above it for stripes, and "fewer than Akita" (34% / 53%) holds for both. The paper sentence has to be checked against its source before the release numbers (30%, 29%) replace it. In the window structure measures the marks-only release model is behind Akita in 0 of 8 patients on most measures (Pearson 0.488 against 0.591), unlike the model with the consensus, so the Supplementary Table S13 sentence must stay restricted to the models that read a reference.
8. "The mean map of the other patients alone detected 44% of the loops and 68% of the stripes. The model therefore detects AML structure that recurs between patients." unchanged; the release model with the AML reference adds 5 points for loops and 1 for stripes (0.490 against 0.438; 0.693 against 0.683), recurrent loops 0.855 against 0.845.
9. "A model trained with the AML patients of the other folds (AML-inclusive model) improved on private loops only slightly (43% against 42%)" becomes "did not improve on the release model for private loops (43% for both: aml_refin 0.425, aml_diff 0.44; release 0.433)". The sentence "It predicted 36% of the measured gain over HSPC at recurrent loops and 11% at private loops" and the Supplementary Fig. S4 / Tables S7 and S16 concern the AML-trained models and are unchanged.
10. Fig. 6a/b: the dots "Our model, AML reference input" and "Our model, cell-line consensus" are replaced by the release rows (`rel_aml`, `rel_c12`) from `results/one_model/remaining/aml/summary_detection_rate.csv` and `summary_auroc_vs_A_refrand.csv` (same layout as `results/aml_competitors_loops/`, so `fig3_aml.py` needs only the path and the two keys `rel_aml` and `rel_c12` in `PRED`). Fig. 6c (patient 018 maps) reads the saved windows `runs/aml_one_model/ours/rel_c12@patient__xu_018/marks.npz`.
11. Discussion: "In AML, the model detected disease-specific loops and stripes in held-out patients more often than Akita, C.Origami and Epiphany. This detection relies on the reference map, and a reference from other patients detected more than a reference from cell lines" holds for the release checkpoint (0.490 / 0.693 against 0.417 / 0.537 for loops / stripes; marks only 0.30 / 0.29).

## 2. Pre-registered new cell types PC-3, Panc1, NK, Caco-2

Protocol of registry section 74 and its pre-registration (`scripts/independent_validation.py` functions, `scripts/one_model_remaining_iv.py`): 413 Akita test windows (central 917 504 bp), seven marks, identical pixels, per-window Pearson, medians over windows, paired bootstrap over windows (2000 resamples, seed 0), consensus12 as Hi-C reference input. The release checkpoint is used unchanged (one network for 10 and 25 kb; Meth absent, cohesin fraction 1); "noref" is the same checkpoint with the reference plane at 0 (`--no-reference`) and corresponds to the paper's model without consensus (honest). The PAPER rows are recomputed in the same pass (same pixels) and equal the values of section 74. Endpoint of the pre-registration applied to the release model: A = model with consensus above consensus12 in >= 3 of 4 cells, B = above the model without consensus in all 4.

**Median per-window Pearson, 10 kb**

| predictor | PC-3 | Panc1 | NK | Caco-2 |
|---|---|---|---|---|
| split-half ceiling (Spearman-Brown) | 0.952 | 0.738 | 0.935 | 0.949 |
| consensus12 alone | 0.883 | 0.613 | 0.658 | 0.870 |
| best other measured cell (oracle) | 0.830 | 0.577 | 0.694 | 0.835 |
| Akita (GM12878 head) | 0.673 | 0.434 | 0.420 | 0.688 |
| PAPER no consensus, 7 marks (honest) | 0.772 | 0.526 | 0.676 | 0.737 |
| PAPER reference input, 7 marks (refin) | 0.895 | 0.615 | 0.766 | 0.894 |
| PAPER reference residual (refres) | 0.896 | 0.617 | 0.782 | 0.897 |
| PAPER difference channels (diff) | 0.902 | 0.623 | 0.802 | 0.898 |
| RELEASE with consensus12, 7 marks | 0.897 | 0.604 | 0.783 | 0.897 |
| RELEASE with consensus12, CTCF + H3K27ac | 0.888 | 0.604 | 0.804 | 0.893 |
| RELEASE with consensus12, CTCF | 0.876 | 0.619 | 0.669 | 0.869 |
| RELEASE with consensus12, no marks | 0.878 | 0.622 | 0.641 | 0.874 |
| RELEASE no consensus, 7 marks | 0.779 | 0.528 | 0.655 | 0.756 |
| RELEASE no consensus, CTCF + H3K27ac | 0.729 | 0.487 | 0.654 | 0.729 |
| RELEASE no consensus, CTCF | 0.695 | 0.478 | 0.424 | 0.641 |
| RELEASE no consensus, no marks | 0.534 | 0.453 | 0.332 | 0.526 |

**Median per-window Pearson, 25 kb**

| predictor | PC-3 | Panc1 | NK | Caco-2 |
|---|---|---|---|---|
| split-half ceiling (Spearman-Brown) | 0.978 | 0.857 | 0.977 | 0.979 |
| consensus12 alone | 0.932 | 0.839 | 0.707 | 0.915 |
| best other measured cell (oracle) | 0.897 | 0.811 | 0.790 | 0.899 |
| Akita (GM12878 head) | 0.663 | 0.549 | 0.404 | 0.688 |
| PAPER no consensus, 7 marks (honest) | 0.817 | 0.709 | 0.734 | 0.759 |
| PAPER reference input, 7 marks (refin) | 0.942 | 0.841 | 0.823 | 0.934 |
| PAPER reference residual (refres) | 0.941 | 0.841 | 0.863 | 0.930 |
| PAPER difference channels (diff) | 0.944 | 0.841 | 0.858 | 0.939 |
| RELEASE with consensus12, 7 marks | 0.936 | 0.824 | 0.830 | 0.933 |
| RELEASE with consensus12, CTCF + H3K27ac | 0.931 | 0.825 | 0.857 | 0.929 |
| RELEASE with consensus12, CTCF | 0.925 | 0.844 | 0.717 | 0.907 |
| RELEASE with consensus12, no marks | 0.926 | 0.843 | 0.683 | 0.913 |
| RELEASE no consensus, 7 marks | 0.827 | 0.718 | 0.690 | 0.785 |
| RELEASE no consensus, CTCF + H3K27ac | 0.780 | 0.670 | 0.696 | 0.754 |
| RELEASE no consensus, CTCF | 0.733 | 0.661 | 0.452 | 0.658 |
| RELEASE no consensus, no marks | 0.573 | 0.605 | 0.400 | 0.525 |

**Paired contrasts (median difference of per-window Pearson, 95 % bootstrap interval over the 413 windows; share of windows in which the first is better)**

| cell | kb | contrast | difference | interval excludes 0 | share better |
|---|---|---|---|---|---|
| PC-3 | 10 kb | A: rel:all - consensus12 | +0.014 [+0.009, +0.022] | yes | 0.65 |
| PC-3 | 10 kb | B: rel:all - rel:all_noref | +0.118 [+0.106, +0.134] | yes | 0.94 |
| PC-3 | 10 kb | rel:all - refin:all | +0.002 [-0.003, +0.005] | no | 0.51 |
| PC-3 | 10 kb | rel:all_noref - honest:all | +0.007 [-0.005, +0.019] | no | 0.58 |
| PC-3 | 10 kb | rel:all_noref - consensus12 | -0.104 [-0.120, -0.090] | no | 0.14 |
| PC-3 | 10 kb | rel:all - akita | +0.224 [+0.200, +0.256] | yes | 0.98 |
| PC-3 | 10 kb | rel:all_noref - akita | +0.106 [+0.078, +0.137] | yes | 0.73 |
| PC-3 | 10 kb | rel:all - best_cell | +0.067 [+0.053, +0.078] | yes | 0.89 |
| PC-3 | 10 kb | refin:all - consensus12 (section 74 A) | +0.012 [+0.007, +0.021] | yes | 0.71 |
| PC-3 | 10 kb | refin:all - honest:all (section 74 B) | +0.123 [+0.111, +0.140] | yes | 0.95 |
| Panc1 | 10 kb | A: rel:all - consensus12 | -0.009 [-0.021, +0.002] | no | 0.44 |
| Panc1 | 10 kb | B: rel:all - rel:all_noref | +0.075 [+0.049, +0.093] | yes | 0.82 |
| Panc1 | 10 kb | rel:all - refin:all | -0.011 [-0.020, -0.001] | no | 0.36 |
| Panc1 | 10 kb | rel:all_noref - honest:all | +0.002 [-0.014, +0.020] | no | 0.53 |
| Panc1 | 10 kb | rel:all_noref - consensus12 | -0.084 [-0.105, -0.057] | no | 0.19 |
| Panc1 | 10 kb | rel:all - akita | +0.170 [+0.145, +0.205] | yes | 0.88 |
| Panc1 | 10 kb | rel:all_noref - akita | +0.094 [+0.067, +0.141] | yes | 0.63 |
| Panc1 | 10 kb | rel:all - best_cell | +0.027 [+0.010, +0.043] | yes | 0.71 |
| Panc1 | 10 kb | refin:all - consensus12 (section 74 A) | +0.002 [-0.009, +0.012] | no | 0.51 |
| Panc1 | 10 kb | refin:all - honest:all (section 74 B) | +0.088 [+0.064, +0.105] | yes | 0.84 |
| NK | 10 kb | A: rel:all - consensus12 | +0.126 [+0.108, +0.161] | yes | 0.86 |
| NK | 10 kb | B: rel:all - rel:all_noref | +0.128 [+0.105, +0.155] | yes | 0.85 |
| NK | 10 kb | rel:all - refin:all | +0.017 [+0.003, +0.029] | yes | 0.66 |
| NK | 10 kb | rel:all_noref - honest:all | -0.021 [-0.042, +0.004] | no | 0.48 |
| NK | 10 kb | rel:all_noref - consensus12 | -0.003 [-0.029, +0.040] | no | 0.53 |
| NK | 10 kb | rel:all - akita | +0.363 [+0.328, +0.394] | yes | 0.93 |
| NK | 10 kb | rel:all_noref - akita | +0.235 [+0.190, +0.273] | yes | 0.77 |
| NK | 10 kb | rel:all - best_cell | +0.089 [+0.067, +0.123] | yes | 0.70 |
| NK | 10 kb | refin:all - consensus12 (section 74 A) | +0.108 [+0.090, +0.145] | yes | 0.84 |
| NK | 10 kb | refin:all - honest:all (section 74 B) | +0.090 [+0.068, +0.123] | yes | 0.72 |
| Caco-2 | 10 kb | A: rel:all - consensus12 | +0.027 [+0.018, +0.035] | yes | 0.79 |
| Caco-2 | 10 kb | B: rel:all - rel:all_noref | +0.141 [+0.124, +0.158] | yes | 0.95 |
| Caco-2 | 10 kb | rel:all - refin:all | +0.003 [-0.003, +0.007] | no | 0.62 |
| Caco-2 | 10 kb | rel:all_noref - honest:all | +0.019 [+0.004, +0.034] | yes | 0.60 |
| Caco-2 | 10 kb | rel:all_noref - consensus12 | -0.114 [-0.133, -0.096] | no | 0.18 |
| Caco-2 | 10 kb | rel:all - akita | +0.209 [+0.184, +0.231] | yes | 0.99 |
| Caco-2 | 10 kb | rel:all_noref - akita | +0.068 [+0.040, +0.092] | yes | 0.63 |
| Caco-2 | 10 kb | rel:all - best_cell | +0.062 [+0.052, +0.079] | yes | 0.91 |
| Caco-2 | 10 kb | refin:all - consensus12 (section 74 A) | +0.024 [+0.015, +0.033] | yes | 0.79 |
| Caco-2 | 10 kb | refin:all - honest:all (section 74 B) | +0.157 [+0.142, +0.173] | yes | 0.94 |
| PC-3 | 25 kb | A: rel:all - consensus12 | +0.004 [-0.003, +0.009] | no | 0.54 |
| PC-3 | 25 kb | B: rel:all - rel:all_noref | +0.108 [+0.098, +0.125] | yes | 0.92 |
| PC-3 | 25 kb | rel:all - refin:all | -0.006 [-0.009, -0.003] | no | 0.41 |
| PC-3 | 25 kb | rel:all_noref - honest:all | +0.011 [-0.004, +0.028] | no | 0.51 |
| PC-3 | 25 kb | rel:all_noref - consensus12 | -0.104 [-0.122, -0.094] | no | 0.15 |
| PC-3 | 25 kb | rel:all - akita | +0.272 [+0.232, +0.306] | yes | 0.98 |
| PC-3 | 25 kb | rel:all_noref - akita | +0.164 [+0.121, +0.202] | yes | 0.72 |
| PC-3 | 25 kb | rel:all - best_cell | +0.038 [+0.028, +0.045] | yes | 0.79 |
| PC-3 | 25 kb | refin:all - consensus12 (section 74 A) | +0.010 [+0.004, +0.015] | yes | 0.62 |
| PC-3 | 25 kb | refin:all - honest:all (section 74 B) | +0.125 [+0.107, +0.149] | yes | 0.91 |
| Panc1 | 25 kb | A: rel:all - consensus12 | -0.015 [-0.022, -0.001] | no | 0.39 |
| Panc1 | 25 kb | B: rel:all - rel:all_noref | +0.106 [+0.085, +0.128] | yes | 0.83 |
| Panc1 | 25 kb | rel:all - refin:all | -0.017 [-0.024, -0.009] | no | 0.27 |
| Panc1 | 25 kb | rel:all_noref - honest:all | +0.009 [-0.007, +0.028] | no | 0.54 |
| Panc1 | 25 kb | rel:all_noref - consensus12 | -0.120 [-0.142, -0.096] | no | 0.18 |
| Panc1 | 25 kb | rel:all - akita | +0.276 [+0.233, +0.332] | yes | 0.93 |
| Panc1 | 25 kb | rel:all_noref - akita | +0.170 [+0.120, +0.230] | yes | 0.69 |
| Panc1 | 25 kb | rel:all - best_cell | +0.013 [+0.002, +0.025] | yes | 0.59 |
| Panc1 | 25 kb | refin:all - consensus12 (section 74 A) | +0.003 [-0.004, +0.014] | no | 0.52 |
| Panc1 | 25 kb | refin:all - honest:all (section 74 B) | +0.132 [+0.115, +0.151] | yes | 0.86 |
| NK | 25 kb | A: rel:all - consensus12 | +0.122 [+0.101, +0.160] | yes | 0.86 |
| NK | 25 kb | B: rel:all - rel:all_noref | +0.139 [+0.106, +0.168] | yes | 0.79 |
| NK | 25 kb | rel:all - refin:all | +0.007 [-0.005, +0.020] | no | 0.57 |
| NK | 25 kb | rel:all_noref - honest:all | -0.044 [-0.066, -0.011] | no | 0.41 |
| NK | 25 kb | rel:all_noref - consensus12 | -0.017 [-0.051, +0.029] | no | 0.53 |
| NK | 25 kb | rel:all - akita | +0.425 [+0.375, +0.466] | yes | 0.94 |
| NK | 25 kb | rel:all_noref - akita | +0.286 [+0.226, +0.342] | yes | 0.76 |
| NK | 25 kb | rel:all - best_cell | +0.040 [+0.016, +0.076] | yes | 0.60 |
| NK | 25 kb | refin:all - consensus12 (section 74 A) | +0.115 [+0.095, +0.153] | yes | 0.86 |
| NK | 25 kb | refin:all - honest:all (section 74 B) | +0.089 [+0.062, +0.124] | yes | 0.70 |
| Caco-2 | 25 kb | A: rel:all - consensus12 | +0.018 [+0.011, +0.028] | yes | 0.73 |
| Caco-2 | 25 kb | B: rel:all - rel:all_noref | +0.148 [+0.129, +0.167] | yes | 0.92 |
| Caco-2 | 25 kb | rel:all - refin:all | -0.001 [-0.004, +0.005] | no | 0.50 |
| Caco-2 | 25 kb | rel:all_noref - honest:all | +0.025 [+0.009, +0.041] | yes | 0.61 |
| Caco-2 | 25 kb | rel:all_noref - consensus12 | -0.130 [-0.148, -0.110] | no | 0.17 |
| Caco-2 | 25 kb | rel:all - akita | +0.245 [+0.219, +0.268] | yes | 0.96 |
| Caco-2 | 25 kb | rel:all_noref - akita | +0.096 [+0.066, +0.124] | yes | 0.64 |
| Caco-2 | 25 kb | rel:all - best_cell | +0.034 [+0.024, +0.045] | yes | 0.74 |
| Caco-2 | 25 kb | refin:all - consensus12 (section 74 A) | +0.019 [+0.012, +0.028] | yes | 0.78 |
| Caco-2 | 25 kb | refin:all - honest:all (section 74 B) | +0.175 [+0.154, +0.187] | yes | 0.94 |

Pre-registered endpoint with the release model at 10 kb: A (above consensus12) in 3 of 4 cells (pc3, nk, caco2), success True; B (above the model without consensus) in 4 of 4 cells, success True.

**Structure measures at 10 kb** (median over windows, all eight measures)

| predictor | cell | pearson | scc | insulation_corr | boundary_f1 | loop_enrich_corr | loop_apa_ratio | compartment_corr | eigenvector_abs_corr |
|---|---|---|---|---|---|---|---|---|---|
| consensus12 | PC-3 | 0.883 | 0.885 | 0.971 | 1.000 | 0.725 | 0.520 | 0.922 | 0.963 |
| consensus12 | Panc1 | 0.613 | 0.517 | 0.936 | 0.000 | 0.204 | 0.114 | 0.762 | 0.899 |
| consensus12 | NK | 0.658 | 0.633 | 0.773 | 0.000 | 0.469 | 0.581 | 0.737 | 0.823 |
| consensus12 | Caco-2 | 0.870 | 0.876 | 0.960 | 1.000 | 0.716 | 0.650 | 0.906 | 0.949 |
| best other cell | PC-3 | 0.830 | 0.834 | 0.940 | 0.000 | 0.657 | 0.744 | 0.880 | 0.939 |
| best other cell | Panc1 | 0.577 | 0.485 | 0.911 | 0.000 | 0.167 | 0.157 | 0.760 | 0.888 |
| best other cell | NK | 0.694 | 0.674 | 0.866 | 0.000 | 0.361 | 0.549 | 0.746 | 0.836 |
| best other cell | Caco-2 | 0.835 | 0.844 | 0.956 | 0.667 | 0.610 | 0.904 | 0.891 | 0.943 |
| Akita | PC-3 | 0.673 | 0.757 | 0.883 | 0.000 | 0.574 | 0.197 | 0.739 | 0.772 |
| Akita | Panc1 | 0.434 | 0.461 | 0.863 | 0.000 | 0.181 | 0.048 | 0.591 | 0.659 |
| Akita | NK | 0.420 | 0.497 | 0.634 | 0.000 | 0.344 | 0.187 | 0.625 | 0.684 |
| Akita | Caco-2 | 0.688 | 0.765 | 0.900 | 0.000 | 0.593 | 0.245 | 0.744 | 0.803 |
| PAPER refin | PC-3 | 0.895 | 0.895 | 0.975 | 1.000 | 0.747 | 0.589 | 0.933 | 0.970 |
| PAPER refin | Panc1 | 0.615 | 0.519 | 0.929 | 0.000 | 0.206 | 0.120 | 0.769 | 0.905 |
| PAPER refin | NK | 0.766 | 0.750 | 0.890 | 0.000 | 0.546 | 0.591 | 0.819 | 0.898 |
| PAPER refin | Caco-2 | 0.894 | 0.890 | 0.974 | 1.000 | 0.737 | 0.699 | 0.926 | 0.966 |
| PAPER honest | PC-3 | 0.772 | 0.758 | 0.894 | 0.000 | 0.579 | 0.256 | 0.843 | 0.916 |
| PAPER honest | Panc1 | 0.526 | 0.400 | 0.801 | 0.000 | 0.155 | 0.045 | 0.693 | 0.843 |
| PAPER honest | NK | 0.676 | 0.645 | 0.821 | 0.000 | 0.376 | 0.172 | 0.745 | 0.852 |
| PAPER honest | Caco-2 | 0.737 | 0.725 | 0.878 | 0.000 | 0.528 | 0.252 | 0.820 | 0.894 |
| RELEASE with consensus12 | PC-3 | 0.897 | 0.894 | 0.973 | 1.000 | 0.744 | 0.598 | 0.932 | 0.969 |
| RELEASE with consensus12 | Panc1 | 0.604 | 0.509 | 0.927 | 0.000 | 0.210 | 0.114 | 0.763 | 0.895 |
| RELEASE with consensus12 | NK | 0.783 | 0.768 | 0.910 | 0.000 | 0.546 | 0.594 | 0.827 | 0.904 |
| RELEASE with consensus12 | Caco-2 | 0.897 | 0.891 | 0.975 | 1.000 | 0.741 | 0.756 | 0.926 | 0.963 |
| RELEASE no consensus | PC-3 | 0.779 | 0.757 | 0.891 | 0.000 | 0.595 | 0.318 | 0.842 | 0.918 |
| RELEASE no consensus | Panc1 | 0.528 | 0.400 | 0.804 | 0.000 | 0.163 | 0.049 | 0.687 | 0.853 |
| RELEASE no consensus | NK | 0.655 | 0.643 | 0.813 | 0.000 | 0.410 | 0.254 | 0.721 | 0.832 |
| RELEASE no consensus | Caco-2 | 0.756 | 0.745 | 0.896 | 0.000 | 0.570 | 0.361 | 0.824 | 0.894 |

### Sentences of main.tex that change (Results "The model improves on the consensus in new cell types", Fig. 10a, Supplementary Table S14, Methods "New cell types")

The pre-registration fixed the dedicated models; the release checkpoint was evaluated on the same samples after the pre-registered analysis, so it has to be described as an additional evaluation of an unchanged protocol, not as part of the pre-registration (Methods sentence "Models and checkpoints were used unchanged" needs "the release checkpoint was evaluated afterwards under the same protocol").

1. "The pre-registered endpoint ... was met (Fig. 10a; Supplementary Table S14)" stays true for the release checkpoint: above the consensus of the 12 training cells in 3 of 4 cell types (PC-3 +0.014 [0.009, 0.022], NK +0.126 [0.108, 0.161], Caco-2 +0.027 [0.018, 0.036]; Panc1 -0.009 [-0.021, +0.002]) and above the model without consensus in 4 of 4 (+0.118, +0.076, +0.128, +0.141).
2. "In all four cell types, the model was also more accurate than Akita and than the model without consensus." stays (release minus Akita +0.224, +0.170, +0.363, +0.209; minus the release without consensus as above).
3. "It was largest in NK cells, whose map differs most from the consensus (0.77 against 0.66)" becomes "(0.78 against 0.66)" (0.783 against 0.658).
4. "In PC-3 and Caco-2 it was small, and in Panc1, the map with the lowest sequencing depth, it was absent." stays, with the sign: in Panc1 the release model is 0.009 below the consensus (0.604 against 0.613, interval including 0); with CTCF alone or no marks the release model with the consensus is at the consensus (0.619, 0.622).
5. Fig. 10a numbers (the labels "reference-input model minus the consensus with its 95% bootstrap interval") change to the values of item 1; the bars of "FlexCible without consensus" are the release without consensus (PC-3 0.779, Panc1 0.528, NK 0.655, Caco-2 0.756; paper model 0.772, 0.526, 0.676, 0.737). The release without consensus is below consensus12 in PC-3, Panc1 and Caco-2 (differences -0.104, -0.084, -0.114) and level with it in NK (-0.002, interval -0.029 to +0.040); against the paper's model without consensus (honest) it differs by +0.007, +0.002, -0.021, +0.019. The bars "reference-input" and "difference-track" models are replaced by the single release model.
6. Fig. 10a / S14 source: `results/one_model/remaining/new_cell_types/summary_test_windows.csv` (columns as `results/independent_validation/summary_test_windows.csv`; methods `rel:all`, `rel:all_noref`).
7. The numbers of the K562, HMEC, HUVEC paragraph (0.90 at 25 kb ... improvement by 0.016 to 0.048) are registry 93 / Supplementary Table S2 and are not part of this analysis.

## 3. Transfer to mouse

Protocol of registry section 75 and its pre-registration (`scripts/mouse_transfer.py` functions, `scripts/one_model_remaining_mouse.py`): the 5629 autosome windows of the Akita-v2 tiles at 10 kb (ENCODE mm10 blacklist and assembly gaps excluded), mESC E14, CH12.LX, adult heart, the seven marks (heart has six: its H3K9me3 file does not exist, the mark is absent), human motifs and normalisers, the leave-one-out mean of the other mouse samples as reference, identical pixels, per-window Pearson and insulation correlation, medians over windows. The release checkpoint is used unchanged (13 marks + Meth; the six marks and Meth that mouse lacks are absent). noref = reference plane 0, the use that corresponds to the paper's model without consensus (honest). Shuffled marks and the marks of another sample as in section 75. PAPER rows are re-scored in the same pass; the untrained network (an architecture control) and the CH12 RAD21/SMC3 set were not rerun. The Akita-v2 mouse heads (1500-window random subset) and the replicate-experiment map are those of section 75.

**Median per-window pearson, 10 kb**

| predictor | mESC | CH12.LX | heart |
|---|---|---|---|
| PAPER shuffled marks (no consensus) | 0.300 | 0.162 | 0.150 |
| RELEASE shuffled marks (no consensus) | 0.280 | 0.160 | 0.159 |
| PAPER sequence only (no consensus) | 0.536 | 0.386 | 0.452 |
| RELEASE sequence only (no consensus) | 0.545 | 0.386 | 0.467 |
| PAPER CTCF | 0.660 | 0.500 | 0.620 |
| RELEASE CTCF | 0.661 | 0.486 | 0.645 |
| PAPER CTCF + H3K27ac | 0.683 | 0.542 | 0.674 |
| RELEASE CTCF + H3K27ac | 0.700 | 0.529 | 0.708 |
| PAPER marks of another sample | 0.722 | 0.533 | 0.608 |
| RELEASE marks of another sample | 0.710 | 0.535 | 0.621 |
| PAPER own seven marks (no consensus) | 0.733 | 0.635 | 0.722 |
| RELEASE own marks (no consensus) | 0.748 | 0.638 | 0.743 |
| mouse consensus alone | 0.766 | 0.746 | 0.765 |
| distance decay only | 0.000 | 0.000 | 0.000 |
| per-window diagonal means (in-sample bound) | 0.425 | 0.376 | 0.380 |
| PAPER reference input | 0.833 | 0.803 | 0.840 |
| RELEASE with the mouse consensus | 0.832 | 0.810 | 0.848 |
| RELEASE with consensus, shuffled marks | 0.771 | 0.680 | 0.725 |
| RELEASE with consensus, marks of another sample | 0.782 | 0.740 | 0.762 |
| RELEASE with consensus, no marks | 0.802 | 0.761 | 0.797 |
| Akita v2 mouse head Hsieh mESC (mESC only, 1500 windows) | 0.715 |  |  |
| Akita v2 mouse head Bonev ES (mESC only) | 0.724 |  |  |

**Median per-window insulation corr, 10 kb**

| predictor | mESC | CH12.LX | heart |
|---|---|---|---|
| PAPER shuffled marks (no consensus) | 0.333 | 0.235 | 0.212 |
| RELEASE shuffled marks (no consensus) | 0.300 | 0.242 | 0.224 |
| PAPER sequence only (no consensus) | 0.557 | 0.485 | 0.495 |
| RELEASE sequence only (no consensus) | 0.572 | 0.488 | 0.520 |
| PAPER CTCF | 0.713 | 0.615 | 0.699 |
| RELEASE CTCF | 0.721 | 0.607 | 0.720 |
| PAPER CTCF + H3K27ac | 0.748 | 0.662 | 0.758 |
| RELEASE CTCF + H3K27ac | 0.768 | 0.654 | 0.777 |
| PAPER marks of another sample | 0.797 | 0.659 | 0.681 |
| RELEASE marks of another sample | 0.794 | 0.662 | 0.697 |
| PAPER own seven marks (no consensus) | 0.794 | 0.754 | 0.808 |
| RELEASE own marks (no consensus) | 0.817 | 0.753 | 0.820 |
| mouse consensus alone | 0.875 | 0.877 | 0.880 |
| per-window diagonal means (in-sample bound) | 0.104 | 0.092 | 0.110 |
| PAPER reference input | 0.906 | 0.907 | 0.915 |
| RELEASE with the mouse consensus | 0.910 | 0.917 | 0.920 |
| RELEASE with consensus, shuffled marks | 0.867 | 0.821 | 0.834 |
| RELEASE with consensus, marks of another sample | 0.873 | 0.870 | 0.864 |
| RELEASE with consensus, no marks | 0.887 | 0.875 | 0.884 |
| Akita v2 mouse head Hsieh mESC (mESC only, 1500 windows) | 0.921 |  |  |
| Akita v2 mouse head Bonev ES (mESC only) | 0.921 |  |  |

**Median per-window pearson, 25 kb**

| predictor | mESC | CH12.LX | heart |
|---|---|---|---|
| PAPER shuffled marks (no consensus) | 0.341 | 0.235 | 0.204 |
| RELEASE shuffled marks (no consensus) | 0.309 | 0.232 | 0.200 |
| PAPER sequence only (no consensus) | 0.603 | 0.558 | 0.556 |
| RELEASE sequence only (no consensus) | 0.579 | 0.521 | 0.527 |
| PAPER CTCF | 0.679 | 0.623 | 0.654 |
| RELEASE CTCF | 0.663 | 0.595 | 0.682 |
| PAPER CTCF + H3K27ac | 0.715 | 0.661 | 0.726 |
| RELEASE CTCF + H3K27ac | 0.705 | 0.655 | 0.747 |
| PAPER marks of another sample | 0.753 | 0.658 | 0.671 |
| RELEASE marks of another sample | 0.744 | 0.649 | 0.669 |
| PAPER own seven marks (no consensus) | 0.759 | 0.748 | 0.775 |
| RELEASE own marks (no consensus) | 0.762 | 0.749 | 0.782 |
| mouse consensus alone | 0.838 | 0.831 | 0.833 |
| distance decay only | 0.000 | 0.000 | 0.000 |
| per-window diagonal means (in-sample bound) | 0.473 | 0.511 | 0.448 |
| PAPER reference input | 0.878 | 0.874 | 0.882 |
| RELEASE with the mouse consensus | 0.882 | 0.875 | 0.882 |
| RELEASE with consensus, shuffled marks | 0.821 | 0.763 | 0.775 |
| RELEASE with consensus, marks of another sample | 0.841 | 0.815 | 0.818 |
| RELEASE with consensus, no marks | 0.860 | 0.836 | 0.847 |
| Akita v2 mouse head Hsieh mESC (mESC only, 1500 windows) | 0.697 |  |  |
| Akita v2 mouse head Bonev ES (mESC only) | 0.712 |  |  |

**Median per-window insulation corr, 25 kb**

| predictor | mESC | CH12.LX | heart |
|---|---|---|---|
| PAPER shuffled marks (no consensus) | 0.367 | 0.272 | 0.257 |
| RELEASE shuffled marks (no consensus) | 0.321 | 0.255 | 0.244 |
| PAPER sequence only (no consensus) | 0.593 | 0.515 | 0.532 |
| RELEASE sequence only (no consensus) | 0.570 | 0.491 | 0.527 |
| PAPER CTCF | 0.708 | 0.612 | 0.668 |
| RELEASE CTCF | 0.699 | 0.606 | 0.705 |
| PAPER CTCF + H3K27ac | 0.741 | 0.650 | 0.755 |
| RELEASE CTCF + H3K27ac | 0.732 | 0.657 | 0.772 |
| PAPER marks of another sample | 0.790 | 0.658 | 0.673 |
| RELEASE marks of another sample | 0.789 | 0.637 | 0.675 |
| PAPER own seven marks (no consensus) | 0.790 | 0.764 | 0.809 |
| RELEASE own marks (no consensus) | 0.788 | 0.765 | 0.814 |
| mouse consensus alone | 0.888 | 0.891 | 0.888 |
| per-window diagonal means (in-sample bound) | 0.160 | 0.168 | 0.170 |
| PAPER reference input | 0.916 | 0.925 | 0.923 |
| RELEASE with the mouse consensus | 0.918 | 0.928 | 0.921 |
| RELEASE with consensus, shuffled marks | 0.882 | 0.850 | 0.848 |
| RELEASE with consensus, marks of another sample | 0.888 | 0.882 | 0.873 |
| RELEASE with consensus, no marks | 0.900 | 0.893 | 0.893 |
| Akita v2 mouse head Hsieh mESC (mESC only, 1500 windows) | 0.923 |  |  |
| Akita v2 mouse head Bonev ES (mESC only) | 0.921 |  |  |

**Fraction of the split-half ceiling at 10 kb** (median Pearson / median full-depth split-half ceiling of section 75)

| sample | ceiling (full depth) | split half | PAPER honest | PAPER fraction | RELEASE noref | RELEASE fraction |
|---|---|---|---|---|---|---|
| mESC | 0.944 | 0.895 | 0.733 | 0.776 | 0.748 | 0.792 |
| CH12.LX | 0.934 | 0.877 | 0.635 | 0.680 | 0.638 | 0.683 |
| heart | 0.963 | 0.929 | 0.722 | 0.749 | 0.743 | 0.772 |

**Paired contrasts of the per-window Pearson** (median difference, 95 % bootstrap interval over windows; share of windows in which the first is better)

| sample | kb | contrast | windows | difference | share better |
|---|---|---|---|---|---|
| mESC | 10 kb | rel:all7_noref - distance_decay | 5629 | +0.748 [+0.742, +0.752] | 0.99 |
| mESC | 10 kb | rel:all7_noref - rel_shuf:all7_noref | 5629 | +0.468 [+0.452, +0.481] | 0.95 |
| mESC | 10 kb | rel:all7_noref - rel_xm:all7_noref | 5629 | +0.038 [+0.033, +0.043] | 0.61 |
| mESC | 10 kb | rel:all7_noref - rel:none_noref | 5629 | +0.203 [+0.194, +0.214] | 0.86 |
| mESC | 10 kb | rel:all7_noref - rel:ctcf_noref | 5629 | +0.086 [+0.080, +0.092] | 0.76 |
| mESC | 10 kb | rel:all7_noref - rel:ctcf_k27ac_noref | 5629 | +0.048 [+0.043, +0.052] | 0.72 |
| mESC | 10 kb | rel:all7_noref - consensus | 5629 | -0.018 [-0.024, -0.012] | 0.48 |
| mESC | 10 kb | rel:all7_noref - distance_oracle | 5629 | +0.323 [+0.313, +0.333] | 0.81 |
| mESC | 10 kb | rel:all7 - consensus | 5629 | +0.066 [+0.062, +0.070] | 0.92 |
| mESC | 10 kb | rel:all7 - rel:all7_noref | 5629 | +0.084 [+0.079, +0.089] | 0.72 |
| mESC | 10 kb | rel:all7 - rel_shuf:all7 | 5629 | +0.061 [+0.057, +0.065] | 0.83 |
| mESC | 10 kb | rel:all7 - rel_xm:all7 | 5629 | +0.050 [+0.046, +0.054] | 0.81 |
| mESC | 10 kb | rel:all7 - rel:none | 5629 | +0.030 [+0.027, +0.033] | 0.75 |
| mESC | 10 kb | rel:all7_noref - honest:all7 | 5629 | +0.015 [+0.011, +0.019] | 0.59 |
| mESC | 10 kb | rel:all7 - refin:all7 | 5629 | -0.001 [-0.003, +0.001] | 0.52 |
| mESC | 10 kb | honest:all7 - distance_decay | 5629 | +0.733 [+0.727, +0.738] | 0.99 |
| mESC | 10 kb | honest:all7 - honest_shuf:all7 | 5629 | +0.432 [+0.421, +0.445] | 0.93 |
| mESC | 10 kb | honest:all7 - honest_xm:all7 | 5629 | +0.011 [+0.006, +0.017] | 0.53 |
| mESC | 10 kb | honest:all7 - consensus | 5629 | -0.033 [-0.039, -0.026] | 0.44 |
| mESC | 10 kb | refin:all7 - consensus | 5629 | +0.067 [+0.063, +0.071] | 0.90 |
| mESC | 10 kb | refin:all7 - honest:all7 | 5629 | +0.100 [+0.094, +0.105] | 0.75 |
| mESC | 10 kb | rel:all7_noref - akita:Hsieh_mESC_uC | 1506 | +0.040 [+0.025, +0.054] | 0.54 |
| mESC | 10 kb | rel:all7_noref - akita:Bonev_ES | 1506 | +0.031 [+0.017, +0.044] | 0.53 |
| mESC | 10 kb | rel:all7 - akita:Bonev_ES | 1506 | +0.106 [+0.095, +0.118] | 0.76 |
| mESC | 10 kb | honest:all7 - akita:Hsieh_mESC_uC | 1506 | +0.023 [+0.009, +0.038] | 0.51 |
| mESC | 10 kb | honest:all7 - akita:Bonev_ES | 1506 | +0.014 [-0.000, +0.029] | 0.51 |
| mESC | 10 kb | rel:all7_noref - replicate_experiment | 5629 | -0.173 [-0.178, -0.168] | 0.07 |
| mESC | 10 kb | honest:all7 - replicate_experiment | 5629 | -0.188 [-0.193, -0.183] | 0.06 |
| CH12.LX | 10 kb | rel:all7_noref - distance_decay | 5629 | +0.638 [+0.628, +0.646] | 0.96 |
| CH12.LX | 10 kb | rel:all7_noref - rel_shuf:all7_noref | 5629 | +0.479 [+0.464, +0.488] | 0.91 |
| CH12.LX | 10 kb | rel:all7_noref - rel_xm:all7_noref | 5629 | +0.104 [+0.094, +0.111] | 0.69 |
| CH12.LX | 10 kb | rel:all7_noref - rel:none_noref | 5629 | +0.252 [+0.241, +0.262] | 0.86 |
| CH12.LX | 10 kb | rel:all7_noref - rel:ctcf_noref | 5629 | +0.153 [+0.143, +0.162] | 0.76 |
| CH12.LX | 10 kb | rel:all7_noref - rel:ctcf_k27ac_noref | 5629 | +0.109 [+0.099, +0.116] | 0.72 |
| CH12.LX | 10 kb | rel:all7_noref - consensus | 5629 | -0.107 [-0.116, -0.100] | 0.32 |
| CH12.LX | 10 kb | rel:all7_noref - distance_oracle | 5629 | +0.263 [+0.253, +0.272] | 0.68 |
| CH12.LX | 10 kb | rel:all7 - consensus | 5629 | +0.065 [+0.059, +0.070] | 0.82 |
| CH12.LX | 10 kb | rel:all7 - rel:all7_noref | 5629 | +0.172 [+0.166, +0.180] | 0.88 |
| CH12.LX | 10 kb | rel:all7 - rel_shuf:all7 | 5629 | +0.130 [+0.123, +0.136] | 0.88 |
| CH12.LX | 10 kb | rel:all7 - rel_xm:all7 | 5629 | +0.071 [+0.066, +0.076] | 0.80 |
| CH12.LX | 10 kb | rel:all7 - rel:none | 5629 | +0.050 [+0.045, +0.055] | 0.76 |
| CH12.LX | 10 kb | rel:all7_noref - honest:all7 | 5629 | +0.003 [-0.002, +0.010] | 0.54 |
| CH12.LX | 10 kb | rel:all7 - refin:all7 | 5629 | +0.007 [+0.004, +0.010] | 0.61 |
| CH12.LX | 10 kb | honest:all7 - distance_decay | 5629 | +0.635 [+0.624, +0.644] | 0.96 |
| CH12.LX | 10 kb | honest:all7 - honest_shuf:all7 | 5629 | +0.473 [+0.460, +0.486] | 0.90 |
| CH12.LX | 10 kb | honest:all7 - honest_xm:all7 | 5629 | +0.103 [+0.092, +0.110] | 0.71 |
| CH12.LX | 10 kb | honest:all7 - consensus | 5629 | -0.111 [-0.121, -0.102] | 0.29 |
| CH12.LX | 10 kb | refin:all7 - consensus | 5629 | +0.058 [+0.052, +0.063] | 0.77 |
| CH12.LX | 10 kb | refin:all7 - honest:all7 | 5629 | +0.168 [+0.161, +0.177] | 0.87 |
| heart | 10 kb | rel:all7_noref - distance_decay | 5629 | +0.743 [+0.738, +0.750] | 0.97 |
| heart | 10 kb | rel:all7_noref - rel_shuf:all7_noref | 5629 | +0.585 [+0.573, +0.597] | 0.95 |
| heart | 10 kb | rel:all7_noref - rel_xm:all7_noref | 5629 | +0.122 [+0.114, +0.131] | 0.76 |
| heart | 10 kb | rel:all7_noref - rel:none_noref | 5629 | +0.276 [+0.266, +0.286] | 0.92 |
| heart | 10 kb | rel:all7_noref - rel:ctcf_noref | 5629 | +0.099 [+0.093, +0.105] | 0.85 |
| heart | 10 kb | rel:all7_noref - rel:ctcf_k27ac_noref | 5629 | +0.035 [+0.031, +0.039] | 0.75 |
| heart | 10 kb | rel:all7_noref - consensus | 5629 | -0.021 [-0.028, -0.015] | 0.48 |
| heart | 10 kb | rel:all7_noref - distance_oracle | 5629 | +0.363 [+0.354, +0.372] | 0.79 |
| heart | 10 kb | rel:all7 - consensus | 5629 | +0.083 [+0.079, +0.086] | 0.93 |
| heart | 10 kb | rel:all7 - rel:all7_noref | 5629 | +0.104 [+0.098, +0.109] | 0.81 |
| heart | 10 kb | rel:all7 - rel_shuf:all7 | 5629 | +0.122 [+0.117, +0.129] | 0.93 |
| heart | 10 kb | rel:all7 - rel_xm:all7 | 5629 | +0.085 [+0.080, +0.090] | 0.87 |
| heart | 10 kb | rel:all7 - rel:none | 5629 | +0.050 [+0.047, +0.053] | 0.87 |
| heart | 10 kb | rel:all7_noref - honest:all7 | 5629 | +0.022 [+0.016, +0.026] | 0.60 |
| heart | 10 kb | rel:all7 - refin:all7 | 5629 | +0.007 [+0.005, +0.009] | 0.61 |
| heart | 10 kb | honest:all7 - distance_decay | 5629 | +0.722 [+0.715, +0.730] | 0.97 |
| heart | 10 kb | honest:all7 - honest_shuf:all7 | 5629 | +0.571 [+0.559, +0.585] | 0.94 |
| heart | 10 kb | honest:all7 - honest_xm:all7 | 5629 | +0.114 [+0.106, +0.124] | 0.73 |
| heart | 10 kb | honest:all7 - consensus | 5629 | -0.043 [-0.050, -0.035] | 0.43 |
| heart | 10 kb | refin:all7 - consensus | 5629 | +0.075 [+0.071, +0.079] | 0.90 |
| heart | 10 kb | refin:all7 - honest:all7 | 5629 | +0.119 [+0.111, +0.125] | 0.79 |
| mESC | 25 kb | rel:all7_noref - distance_decay | 5579 | +0.762 [+0.755, +0.769] | 0.97 |
| mESC | 25 kb | rel:all7_noref - rel_shuf:all7_noref | 5579 | +0.453 [+0.438, +0.468] | 0.90 |
| mESC | 25 kb | rel:all7_noref - rel_xm:all7_noref | 5579 | +0.018 [+0.011, +0.025] | 0.52 |
| mESC | 25 kb | rel:all7_noref - rel:none_noref | 5579 | +0.183 [+0.173, +0.193] | 0.80 |
| mESC | 25 kb | rel:all7_noref - rel:ctcf_noref | 5579 | +0.099 [+0.092, +0.106] | 0.74 |
| mESC | 25 kb | rel:all7_noref - rel:ctcf_k27ac_noref | 5579 | +0.057 [+0.051, +0.064] | 0.70 |
| mESC | 25 kb | rel:all7_noref - consensus | 5579 | -0.076 [-0.084, -0.069] | 0.34 |
| mESC | 25 kb | rel:all7_noref - distance_oracle | 5579 | +0.289 [+0.277, +0.299] | 0.74 |
| mESC | 25 kb | rel:all7 - consensus | 5579 | +0.043 [+0.040, +0.048] | 0.82 |
| mESC | 25 kb | rel:all7 - rel:all7_noref | 5579 | +0.120 [+0.113, +0.127] | 0.79 |
| mESC | 25 kb | rel:all7 - rel_shuf:all7 | 5579 | +0.061 [+0.057, +0.064] | 0.81 |
| mESC | 25 kb | rel:all7 - rel_xm:all7 | 5579 | +0.040 [+0.037, +0.043] | 0.79 |
| mESC | 25 kb | rel:all7 - rel:none | 5579 | +0.022 [+0.020, +0.025] | 0.71 |
| mESC | 25 kb | rel:all7_noref - honest:all7 | 5579 | +0.003 [-0.003, +0.009] | 0.52 |
| mESC | 25 kb | rel:all7 - refin:all7 | 5579 | +0.004 [+0.002, +0.006] | 0.55 |
| mESC | 25 kb | honest:all7 - distance_decay | 5579 | +0.759 [+0.752, +0.767] | 0.97 |
| mESC | 25 kb | honest:all7 - honest_shuf:all7 | 5579 | +0.418 [+0.407, +0.434] | 0.90 |
| mESC | 25 kb | honest:all7 - honest_xm:all7 | 5579 | +0.006 [-0.001, +0.013] | 0.52 |
| mESC | 25 kb | honest:all7 - consensus | 5579 | -0.079 [-0.087, -0.071] | 0.34 |
| mESC | 25 kb | refin:all7 - consensus | 5579 | +0.040 [+0.037, +0.043] | 0.82 |
| mESC | 25 kb | refin:all7 - honest:all7 | 5579 | +0.119 [+0.112, +0.126] | 0.77 |
| mESC | 25 kb | rel:all7_noref - akita:Hsieh_mESC_uC | 1490 | +0.073 [+0.056, +0.090] | 0.56 |
| mESC | 25 kb | rel:all7_noref - akita:Bonev_ES | 1490 | +0.058 [+0.041, +0.077] | 0.55 |
| mESC | 25 kb | rel:all7 - akita:Bonev_ES | 1490 | +0.170 [+0.157, +0.186] | 0.84 |
| mESC | 25 kb | honest:all7 - akita:Hsieh_mESC_uC | 1490 | +0.075 [+0.057, +0.096] | 0.57 |
| mESC | 25 kb | honest:all7 - akita:Bonev_ES | 1490 | +0.060 [+0.043, +0.082] | 0.55 |
| mESC | 25 kb | rel:all7_noref - replicate_experiment | 5579 | -0.217 [-0.225, -0.210] | 0.00 |
| mESC | 25 kb | honest:all7 - replicate_experiment | 5579 | -0.220 [-0.227, -0.212] | 0.00 |
| CH12.LX | 25 kb | rel:all7_noref - distance_decay | 5579 | +0.749 [+0.742, +0.755] | 0.97 |
| CH12.LX | 25 kb | rel:all7_noref - rel_shuf:all7_noref | 5579 | +0.517 [+0.501, +0.533] | 0.89 |
| CH12.LX | 25 kb | rel:all7_noref - rel_xm:all7_noref | 5579 | +0.100 [+0.092, +0.109] | 0.72 |
| CH12.LX | 25 kb | rel:all7_noref - rel:none_noref | 5579 | +0.228 [+0.214, +0.241] | 0.84 |
| CH12.LX | 25 kb | rel:all7_noref - rel:ctcf_noref | 5579 | +0.154 [+0.145, +0.163] | 0.78 |
| CH12.LX | 25 kb | rel:all7_noref - rel:ctcf_k27ac_noref | 5579 | +0.094 [+0.087, +0.101] | 0.72 |
| CH12.LX | 25 kb | rel:all7_noref - consensus | 5579 | -0.082 [-0.091, -0.075] | 0.35 |
| CH12.LX | 25 kb | rel:all7_noref - distance_oracle | 5579 | +0.238 [+0.226, +0.247] | 0.64 |
| CH12.LX | 25 kb | rel:all7 - consensus | 5579 | +0.044 [+0.040, +0.047] | 0.78 |
| CH12.LX | 25 kb | rel:all7 - rel:all7_noref | 5579 | +0.126 [+0.120, +0.134] | 0.82 |
| CH12.LX | 25 kb | rel:all7 - rel_shuf:all7 | 5579 | +0.113 [+0.107, +0.117] | 0.87 |
| CH12.LX | 25 kb | rel:all7 - rel_xm:all7 | 5579 | +0.061 [+0.057, +0.064] | 0.84 |
| CH12.LX | 25 kb | rel:all7 - rel:none | 5579 | +0.039 [+0.035, +0.043] | 0.76 |
| CH12.LX | 25 kb | rel:all7_noref - honest:all7 | 5579 | +0.001 [-0.004, +0.006] | 0.50 |
| CH12.LX | 25 kb | rel:all7 - refin:all7 | 5579 | +0.001 [-0.000, +0.003] | 0.53 |
| CH12.LX | 25 kb | honest:all7 - distance_decay | 5579 | +0.748 [+0.742, +0.754] | 0.98 |
| CH12.LX | 25 kb | honest:all7 - honest_shuf:all7 | 5579 | +0.513 [+0.497, +0.529] | 0.89 |
| CH12.LX | 25 kb | honest:all7 - honest_xm:all7 | 5579 | +0.089 [+0.083, +0.097] | 0.71 |
| CH12.LX | 25 kb | honest:all7 - consensus | 5579 | -0.084 [-0.091, -0.076] | 0.34 |
| CH12.LX | 25 kb | refin:all7 - consensus | 5579 | +0.043 [+0.038, +0.046] | 0.77 |
| CH12.LX | 25 kb | refin:all7 - honest:all7 | 5579 | +0.126 [+0.120, +0.132] | 0.81 |
| heart | 25 kb | rel:all7_noref - distance_decay | 5579 | +0.782 [+0.775, +0.787] | 0.98 |
| heart | 25 kb | rel:all7_noref - rel_shuf:all7_noref | 5579 | +0.582 [+0.567, +0.597] | 0.92 |
| heart | 25 kb | rel:all7_noref - rel_xm:all7_noref | 5579 | +0.112 [+0.104, +0.120] | 0.71 |
| heart | 25 kb | rel:all7_noref - rel:none_noref | 5579 | +0.255 [+0.244, +0.265] | 0.88 |
| heart | 25 kb | rel:all7_noref - rel:ctcf_noref | 5579 | +0.100 [+0.091, +0.106] | 0.80 |
| heart | 25 kb | rel:all7_noref - rel:ctcf_k27ac_noref | 5579 | +0.035 [+0.030, +0.039] | 0.70 |
| heart | 25 kb | rel:all7_noref - consensus | 5579 | -0.051 [-0.058, -0.046] | 0.40 |
| heart | 25 kb | rel:all7_noref - distance_oracle | 5579 | +0.334 [+0.323, +0.345] | 0.75 |
| heart | 25 kb | rel:all7 - consensus | 5579 | +0.049 [+0.046, +0.052] | 0.80 |
| heart | 25 kb | rel:all7 - rel:all7_noref | 5579 | +0.101 [+0.095, +0.107] | 0.80 |
| heart | 25 kb | rel:all7 - rel_shuf:all7 | 5579 | +0.108 [+0.102, +0.113] | 0.88 |
| heart | 25 kb | rel:all7 - rel_xm:all7 | 5579 | +0.064 [+0.061, +0.068] | 0.80 |
| heart | 25 kb | rel:all7 - rel:none | 5579 | +0.036 [+0.033, +0.038] | 0.77 |
| heart | 25 kb | rel:all7_noref - honest:all7 | 5579 | +0.007 [+0.002, +0.011] | 0.52 |
| heart | 25 kb | rel:all7 - refin:all7 | 5579 | +0.000 [-0.002, +0.002] | 0.50 |
| heart | 25 kb | honest:all7 - distance_decay | 5579 | +0.775 [+0.769, +0.781] | 0.98 |
| heart | 25 kb | honest:all7 - honest_shuf:all7 | 5579 | +0.571 [+0.556, +0.588] | 0.92 |
| heart | 25 kb | honest:all7 - honest_xm:all7 | 5579 | +0.104 [+0.094, +0.114] | 0.69 |
| heart | 25 kb | honest:all7 - consensus | 5579 | -0.058 [-0.064, -0.051] | 0.39 |
| heart | 25 kb | refin:all7 - consensus | 5579 | +0.049 [+0.046, +0.052] | 0.85 |
| heart | 25 kb | refin:all7 - honest:all7 | 5579 | +0.107 [+0.101, +0.113] | 0.78 |

**Structure classes at 10 kb: own marks minus shuffled marks / minus no marks / minus marks of another sample** (median over windows; boundary F1 mean)

| sample | own marks | minus | measure | difference |
|---|---|---|---|---|
| mESC | rel:all7_noref | rel_shuf:all7_noref | pearson | +0.468 [+0.452, +0.481] |
| mESC | rel:all7_noref | rel:none_noref | pearson | +0.203 [+0.194, +0.214] |
| mESC | rel:all7_noref | rel_xm:all7_noref | pearson | +0.038 [+0.033, +0.043] |
| mESC | honest:all7 | honest_shuf:all7 | pearson | +0.432 [+0.421, +0.445] |
| mESC | honest:all7 | honest:none | pearson | +0.197 [+0.189, +0.206] |
| mESC | honest:all7 | honest_xm:all7 | pearson | +0.011 [+0.006, +0.017] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | scc | +0.471 [+0.460, +0.486] |
| mESC | rel:all7_noref | rel:none_noref | scc | +0.236 [+0.225, +0.245] |
| mESC | rel:all7_noref | rel_xm:all7_noref | scc | +0.016 [+0.010, +0.022] |
| mESC | honest:all7 | honest_shuf:all7 | scc | +0.427 [+0.412, +0.439] |
| mESC | honest:all7 | honest:none | scc | +0.228 [+0.217, +0.237] |
| mESC | honest:all7 | honest_xm:all7 | scc | -0.013 [-0.021, -0.006] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | insulation_corr | +0.517 [+0.486, +0.544] |
| mESC | rel:all7_noref | rel:none_noref | insulation_corr | +0.245 [+0.225, +0.262] |
| mESC | rel:all7_noref | rel_xm:all7_noref | insulation_corr | +0.023 [+0.011, +0.032] |
| mESC | honest:all7 | honest_shuf:all7 | insulation_corr | +0.461 [+0.434, +0.487] |
| mESC | honest:all7 | honest:none | insulation_corr | +0.236 [+0.218, +0.255] |
| mESC | honest:all7 | honest_xm:all7 | insulation_corr | -0.003 [-0.014, +0.009] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | boundary_f1 | +0.226 [+0.209, +0.242] |
| mESC | rel:all7_noref | rel:none_noref | boundary_f1 | +0.155 [+0.140, +0.170] |
| mESC | rel:all7_noref | rel_xm:all7_noref | boundary_f1 | +0.021 [+0.006, +0.038] |
| mESC | honest:all7 | honest_shuf:all7 | boundary_f1 | +0.188 [+0.172, +0.203] |
| mESC | honest:all7 | honest:none | boundary_f1 | +0.102 [+0.088, +0.117] |
| mESC | honest:all7 | honest_xm:all7 | boundary_f1 | -0.004 [-0.018, +0.011] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | loop_enrich_corr | +0.382 [+0.376, +0.389] |
| mESC | rel:all7_noref | rel:none_noref | loop_enrich_corr | +0.214 [+0.208, +0.221] |
| mESC | rel:all7_noref | rel_xm:all7_noref | loop_enrich_corr | +0.036 [+0.031, +0.042] |
| mESC | honest:all7 | honest_shuf:all7 | loop_enrich_corr | +0.337 [+0.331, +0.344] |
| mESC | honest:all7 | honest:none | loop_enrich_corr | +0.205 [+0.199, +0.212] |
| mESC | honest:all7 | honest_xm:all7 | loop_enrich_corr | +0.010 [+0.004, +0.015] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | loop_apa_ratio | +0.300 [+0.288, +0.307] |
| mESC | rel:all7_noref | rel:none_noref | loop_apa_ratio | +0.243 [+0.233, +0.249] |
| mESC | rel:all7_noref | rel_xm:all7_noref | loop_apa_ratio | -0.144 [-0.160, -0.135] |
| mESC | honest:all7 | honest_shuf:all7 | loop_apa_ratio | +0.227 [+0.219, +0.237] |
| mESC | honest:all7 | honest:none | loop_apa_ratio | +0.159 [+0.151, +0.167] |
| mESC | honest:all7 | honest_xm:all7 | loop_apa_ratio | -0.186 [-0.194, -0.176] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | compartment_corr | +0.194 [+0.184, +0.202] |
| mESC | rel:all7_noref | rel:none_noref | compartment_corr | +0.102 [+0.094, +0.109] |
| mESC | rel:all7_noref | rel_xm:all7_noref | compartment_corr | +0.038 [+0.030, +0.044] |
| mESC | honest:all7 | honest_shuf:all7 | compartment_corr | +0.177 [+0.171, +0.185] |
| mESC | honest:all7 | honest:none | compartment_corr | +0.100 [+0.095, +0.108] |
| mESC | honest:all7 | honest_xm:all7 | compartment_corr | +0.013 [+0.007, +0.021] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | eigenvector_abs_corr | +0.182 [+0.170, +0.191] |
| mESC | rel:all7_noref | rel:none_noref | eigenvector_abs_corr | +0.089 [+0.080, +0.097] |
| mESC | rel:all7_noref | rel_xm:all7_noref | eigenvector_abs_corr | +0.032 [+0.025, +0.039] |
| mESC | honest:all7 | honest_shuf:all7 | eigenvector_abs_corr | +0.153 [+0.142, +0.164] |
| mESC | honest:all7 | honest:none | eigenvector_abs_corr | +0.080 [+0.072, +0.091] |
| mESC | honest:all7 | honest_xm:all7 | eigenvector_abs_corr | +0.009 [+0.003, +0.015] |
| mESC | rel:all7_noref | rel_shuf:all7_noref | stripe_corr | +0.493 [+0.479, +0.506] |
| mESC | rel:all7_noref | rel:none_noref | stripe_corr | +0.228 [+0.217, +0.239] |
| mESC | rel:all7_noref | rel_xm:all7_noref | stripe_corr | +0.000 [-0.007, +0.007] |
| mESC | honest:all7 | honest_shuf:all7 | stripe_corr | +0.433 [+0.418, +0.447] |
| mESC | honest:all7 | honest:none | stripe_corr | +0.211 [+0.203, +0.220] |
| mESC | honest:all7 | honest_xm:all7 | stripe_corr | -0.025 [-0.033, -0.018] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | pearson | +0.479 [+0.464, +0.488] |
| CH12.LX | rel:all7_noref | rel:none_noref | pearson | +0.252 [+0.241, +0.262] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | pearson | +0.104 [+0.094, +0.111] |
| CH12.LX | honest:all7 | honest_shuf:all7 | pearson | +0.473 [+0.460, +0.486] |
| CH12.LX | honest:all7 | honest:none | pearson | +0.249 [+0.237, +0.259] |
| CH12.LX | honest:all7 | honest_xm:all7 | pearson | +0.103 [+0.092, +0.110] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | scc | +0.454 [+0.443, +0.465] |
| CH12.LX | rel:all7_noref | rel:none_noref | scc | +0.265 [+0.254, +0.277] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | scc | +0.098 [+0.088, +0.106] |
| CH12.LX | honest:all7 | honest_shuf:all7 | scc | +0.443 [+0.431, +0.455] |
| CH12.LX | honest:all7 | honest:none | scc | +0.238 [+0.226, +0.251] |
| CH12.LX | honest:all7 | honest_xm:all7 | scc | +0.084 [+0.074, +0.091] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | insulation_corr | +0.511 [+0.484, +0.540] |
| CH12.LX | rel:all7_noref | rel:none_noref | insulation_corr | +0.265 [+0.244, +0.282] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | insulation_corr | +0.091 [+0.075, +0.104] |
| CH12.LX | honest:all7 | honest_shuf:all7 | insulation_corr | +0.518 [+0.489, +0.546] |
| CH12.LX | honest:all7 | honest:none | insulation_corr | +0.269 [+0.246, +0.288] |
| CH12.LX | honest:all7 | honest_xm:all7 | insulation_corr | +0.094 [+0.078, +0.109] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | boundary_f1 | +0.181 [+0.166, +0.195] |
| CH12.LX | rel:all7_noref | rel:none_noref | boundary_f1 | +0.116 [+0.102, +0.129] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | boundary_f1 | +0.029 [+0.016, +0.043] |
| CH12.LX | honest:all7 | honest_shuf:all7 | boundary_f1 | +0.188 [+0.173, +0.202] |
| CH12.LX | honest:all7 | honest:none | boundary_f1 | +0.106 [+0.092, +0.119] |
| CH12.LX | honest:all7 | honest_xm:all7 | boundary_f1 | +0.048 [+0.035, +0.061] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | loop_enrich_corr | +0.331 [+0.321, +0.341] |
| CH12.LX | rel:all7_noref | rel:none_noref | loop_enrich_corr | +0.211 [+0.202, +0.218] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | loop_enrich_corr | +0.062 [+0.056, +0.069] |
| CH12.LX | honest:all7 | honest_shuf:all7 | loop_enrich_corr | +0.315 [+0.308, +0.324] |
| CH12.LX | honest:all7 | honest:none | loop_enrich_corr | +0.196 [+0.188, +0.203] |
| CH12.LX | honest:all7 | honest_xm:all7 | loop_enrich_corr | +0.051 [+0.045, +0.057] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | loop_apa_ratio | +0.164 [+0.156, +0.174] |
| CH12.LX | rel:all7_noref | rel:none_noref | loop_apa_ratio | +0.137 [+0.130, +0.145] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | loop_apa_ratio | +0.024 [+0.018, +0.031] |
| CH12.LX | honest:all7 | honest_shuf:all7 | loop_apa_ratio | +0.127 [+0.120, +0.133] |
| CH12.LX | honest:all7 | honest:none | loop_apa_ratio | +0.088 [+0.082, +0.093] |
| CH12.LX | honest:all7 | honest_xm:all7 | loop_apa_ratio | +0.019 [+0.014, +0.023] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | compartment_corr | +0.192 [+0.182, +0.202] |
| CH12.LX | rel:all7_noref | rel:none_noref | compartment_corr | +0.120 [+0.112, +0.128] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | compartment_corr | +0.071 [+0.063, +0.078] |
| CH12.LX | honest:all7 | honest_shuf:all7 | compartment_corr | +0.198 [+0.188, +0.206] |
| CH12.LX | honest:all7 | honest:none | compartment_corr | +0.129 [+0.120, +0.137] |
| CH12.LX | honest:all7 | honest_xm:all7 | compartment_corr | +0.077 [+0.067, +0.083] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | eigenvector_abs_corr | +0.196 [+0.183, +0.208] |
| CH12.LX | rel:all7_noref | rel:none_noref | eigenvector_abs_corr | +0.124 [+0.112, +0.135] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | eigenvector_abs_corr | +0.076 [+0.065, +0.086] |
| CH12.LX | honest:all7 | honest_shuf:all7 | eigenvector_abs_corr | +0.191 [+0.178, +0.200] |
| CH12.LX | honest:all7 | honest:none | eigenvector_abs_corr | +0.124 [+0.114, +0.137] |
| CH12.LX | honest:all7 | honest_xm:all7 | eigenvector_abs_corr | +0.073 [+0.063, +0.084] |
| CH12.LX | rel:all7_noref | rel_shuf:all7_noref | stripe_corr | +0.453 [+0.439, +0.465] |
| CH12.LX | rel:all7_noref | rel:none_noref | stripe_corr | +0.259 [+0.247, +0.270] |
| CH12.LX | rel:all7_noref | rel_xm:all7_noref | stripe_corr | +0.079 [+0.069, +0.089] |
| CH12.LX | honest:all7 | honest_shuf:all7 | stripe_corr | +0.441 [+0.427, +0.458] |
| CH12.LX | honest:all7 | honest:none | stripe_corr | +0.227 [+0.215, +0.242] |
| CH12.LX | honest:all7 | honest_xm:all7 | stripe_corr | +0.072 [+0.062, +0.084] |
| heart | rel:all7_noref | rel_shuf:all7_noref | pearson | +0.585 [+0.573, +0.597] |
| heart | rel:all7_noref | rel:none_noref | pearson | +0.276 [+0.266, +0.286] |
| heart | rel:all7_noref | rel_xm:all7_noref | pearson | +0.122 [+0.114, +0.131] |
| heart | honest:all7 | honest_shuf:all7 | pearson | +0.571 [+0.559, +0.585] |
| heart | honest:all7 | honest:none | pearson | +0.270 [+0.259, +0.279] |
| heart | honest:all7 | honest_xm:all7 | pearson | +0.114 [+0.106, +0.124] |
| heart | rel:all7_noref | rel_shuf:all7_noref | scc | +0.573 [+0.560, +0.585] |
| heart | rel:all7_noref | rel:none_noref | scc | +0.318 [+0.306, +0.329] |
| heart | rel:all7_noref | rel_xm:all7_noref | scc | +0.144 [+0.133, +0.154] |
| heart | honest:all7 | honest_shuf:all7 | scc | +0.560 [+0.547, +0.572] |
| heart | honest:all7 | honest:none | scc | +0.320 [+0.310, +0.331] |
| heart | honest:all7 | honest_xm:all7 | scc | +0.147 [+0.136, +0.158] |
| heart | rel:all7_noref | rel_shuf:all7_noref | insulation_corr | +0.596 [+0.571, +0.626] |
| heart | rel:all7_noref | rel:none_noref | insulation_corr | +0.299 [+0.277, +0.321] |
| heart | rel:all7_noref | rel_xm:all7_noref | insulation_corr | +0.123 [+0.109, +0.142] |
| heart | honest:all7 | honest_shuf:all7 | insulation_corr | +0.596 [+0.568, +0.626] |
| heart | honest:all7 | honest:none | insulation_corr | +0.313 [+0.294, +0.332] |
| heart | honest:all7 | honest_xm:all7 | insulation_corr | +0.127 [+0.112, +0.142] |
| heart | rel:all7_noref | rel_shuf:all7_noref | boundary_f1 | +0.224 [+0.208, +0.239] |
| heart | rel:all7_noref | rel:none_noref | boundary_f1 | +0.159 [+0.145, +0.174] |
| heart | rel:all7_noref | rel_xm:all7_noref | boundary_f1 | +0.095 [+0.080, +0.110] |
| heart | honest:all7 | honest_shuf:all7 | boundary_f1 | +0.197 [+0.181, +0.212] |
| heart | honest:all7 | honest:none | boundary_f1 | +0.117 [+0.103, +0.131] |
| heart | honest:all7 | honest_xm:all7 | boundary_f1 | +0.069 [+0.053, +0.084] |
| heart | rel:all7_noref | rel_shuf:all7_noref | loop_enrich_corr | +0.500 [+0.493, +0.507] |
| heart | rel:all7_noref | rel:none_noref | loop_enrich_corr | +0.339 [+0.333, +0.347] |
| heart | rel:all7_noref | rel_xm:all7_noref | loop_enrich_corr | +0.178 [+0.171, +0.185] |
| heart | honest:all7 | honest_shuf:all7 | loop_enrich_corr | +0.461 [+0.454, +0.468] |
| heart | honest:all7 | honest:none | loop_enrich_corr | +0.311 [+0.303, +0.318] |
| heart | honest:all7 | honest_xm:all7 | loop_enrich_corr | +0.160 [+0.153, +0.169] |
| heart | rel:all7_noref | rel_shuf:all7_noref | loop_apa_ratio | +0.358 [+0.349, +0.367] |
| heart | rel:all7_noref | rel:none_noref | loop_apa_ratio | +0.325 [+0.317, +0.333] |
| heart | rel:all7_noref | rel_xm:all7_noref | loop_apa_ratio | +0.207 [+0.200, +0.214] |
| heart | honest:all7 | honest_shuf:all7 | loop_apa_ratio | +0.318 [+0.311, +0.323] |
| heart | honest:all7 | honest:none | loop_apa_ratio | +0.272 [+0.267, +0.277] |
| heart | honest:all7 | honest_xm:all7 | loop_apa_ratio | +0.203 [+0.198, +0.208] |
| heart | rel:all7_noref | rel_shuf:all7_noref | compartment_corr | +0.232 [+0.224, +0.240] |
| heart | rel:all7_noref | rel:none_noref | compartment_corr | +0.127 [+0.119, +0.134] |
| heart | rel:all7_noref | rel_xm:all7_noref | compartment_corr | +0.069 [+0.063, +0.077] |
| heart | honest:all7 | honest_shuf:all7 | compartment_corr | +0.205 [+0.194, +0.212] |
| heart | honest:all7 | honest:none | compartment_corr | +0.132 [+0.123, +0.138] |
| heart | honest:all7 | honest_xm:all7 | compartment_corr | +0.067 [+0.059, +0.074] |
| heart | rel:all7_noref | rel_shuf:all7_noref | eigenvector_abs_corr | +0.220 [+0.206, +0.230] |
| heart | rel:all7_noref | rel:none_noref | eigenvector_abs_corr | +0.105 [+0.096, +0.114] |
| heart | rel:all7_noref | rel_xm:all7_noref | eigenvector_abs_corr | +0.053 [+0.044, +0.061] |
| heart | honest:all7 | honest_shuf:all7 | eigenvector_abs_corr | +0.190 [+0.178, +0.200] |
| heart | honest:all7 | honest:none | eigenvector_abs_corr | +0.111 [+0.102, +0.121] |
| heart | honest:all7 | honest_xm:all7 | eigenvector_abs_corr | +0.050 [+0.043, +0.059] |
| heart | rel:all7_noref | rel_shuf:all7_noref | stripe_corr | +0.565 [+0.551, +0.578] |
| heart | rel:all7_noref | rel:none_noref | stripe_corr | +0.300 [+0.289, +0.312] |
| heart | rel:all7_noref | rel_xm:all7_noref | stripe_corr | +0.120 [+0.110, +0.129] |
| heart | honest:all7 | honest_shuf:all7 | stripe_corr | +0.559 [+0.545, +0.573] |
| heart | honest:all7 | honest:none | stripe_corr | +0.305 [+0.293, +0.316] |
| heart | honest:all7 | honest_xm:all7 | stripe_corr | +0.132 [+0.123, +0.142] |

Pre-registered endpoint with the release model (no consensus, 10 kb; above distance decay and above shuffled marks in every sample): met = True.

### Sentences of main.tex that change (Results "The human-trained model transfers to mouse", Fig. 7, Supplementary Table S15 / S16, Methods "Mouse")

"The model without consensus" is the release checkpoint with the reference plane at 0 (`flexcible predict --no-reference`); "the models" reading the mouse consensus are the release checkpoint with the leave-one-out mean of the other mouse samples. The pre-registration of section 75 fixed the dedicated models; the release checkpoint is an additional evaluation under the same protocol and has to be described as such. Heart has six marks, not seven (its H3K9me3 file does not exist), in the paper and here; "each with seven marks" in the Results sentence and the Methods are therefore not exact for heart.

1. "It reached 68% to 78% of the agreement between replicates" becomes "68% to 79%" (CH12.LX 0.638 / 0.934 = 0.683, heart 0.743 / 0.963 = 0.772, mESC 0.748 / 0.944 = 0.792; paper model 0.680, 0.750, 0.776).
2. "and was clearly more accurate than with shuffled marks in every sample (Fig. 7a)": holds. Differences of the median Pearson, own marks minus shuffled marks: +0.468 [0.452, 0.481] mESC, +0.479 [0.464, 0.488] CH12.LX, +0.585 [0.573, 0.597] heart (paper model: 0.432, 0.473, 0.571). Pre-registered endpoint (above distance decay and above shuffled marks in all three samples): met.
3. "It was also more accurate than with the sequence alone, and every structure class, from insulation to loops and compartments, was predicted better than with shuffled marks": holds. Sequence only (release, no consensus) 0.545, 0.386, 0.467; all nine measures have intervals above 0 against shuffled marks and against no marks in all three samples (the contrast with the marks of another sample is above 0 for Pearson in all samples, not for the APA ratio of mESC, -0.144, as in the paper).
4. "In mESC, it was as accurate as the mouse model of Akita, which was trained on mouse Hi-C data (Supplementary Table S15)" becomes "slightly more accurate than" (median per-window difference to the Akita-v2 mouse heads on its 1500-window subset: +0.040 [0.025, 0.054] against the Hsieh mESC head, +0.031 [0.017, 0.044] against the Bonev ES head; the paper model +0.023 [0.009, 0.038] and +0.014 [-0.000, 0.028]).
5. "With the mean map of the other mouse samples as reference, the models exceeded this consensus by 0.06 to 0.08, as in human." stays ("by 0.06 to 0.08") with the release values +0.066 [0.062, 0.070] mESC, +0.065 [0.059, 0.070] CH12.LX, +0.083 [0.079, 0.086] heart (paper reference-input model 0.067, 0.058, 0.076): the range 0.06 to 0.08 holds. At 25 kb: +0.044, +0.044, +0.049.
6. Fig. 7a bars and Table S16 rows (kb, predictor, r, insulation): see the tables below (`results/one_model/remaining/mouse/summary_autosomes.csv`, methods `rel:all7_noref`, `rel_shuf:all7_noref`, `rel:none_noref`, `rel:ctcf_noref`, `rel:ctcf_k27ac_noref`, `rel_xm:all7_noref`, `rel:all7`); the "untrained network" row is the architecture control of section 75 and was not rerun for the release architecture. The "reference residual" row has no release counterpart (one reference-input design).
7. Release without consensus against the paper's model without consensus on the same windows: +0.015 [0.011, 0.019] mESC, +0.003 [-0.002, 0.010] CH12.LX, +0.022 [0.016, 0.026] heart; with the consensus the release and the paper reference-input model are equal (-0.001, +0.007, +0.007).
8. Discussion: "In mouse, the consensus of a few other mouse samples was already as useful as in human" is unchanged (gain over the consensus 0.065 to 0.083 in mouse; human 0.016 to 0.048, registry 93).

## 4. Replicate ceiling

Statistic: median per-region Pearson of the reference-input model on the 413 Akita regions divided by the ceiling sqrt(r) (single-replicate target) or sqrt(2r/(1+r)) (merged target) of the agreement r of two biological replicates; deepest pair for GM12878 and K562. paper = all12refin_drop (dedicated per size), release = all12one_fullrd with the consensus (leave-one-out for training cells, consensus12 for held-out cells), noref = all12one_fullrd with the reference plane at 0. The ceilings are the existing ones; nothing was predicted again (the release 413-region tables of registry section 93 were used). Script `scripts/one_model_remaining_ceiling.py`, files `results/one_model/remaining/replicate_ceiling/`.

**25 kb** (cells with a replicate pair)

| cell | set | paper | release | noref | consensus | replicate r | ceiling | paper/ceiling | release/ceiling | noref/ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| GM12878 | training | 0.936 | 0.937 | 0.830 | 0.882 | 0.938 | 0.984 | 0.951 | 0.952 | 0.843 |
| IMR-90 | training | 0.948 | 0.955 | 0.871 | 0.928 | 0.959 | 0.990 | 0.958 | 0.965 | 0.881 |
| K562 | held-out | 0.926 | 0.925 | 0.804 | 0.897 | 0.937 | 0.984 | 0.942 | 0.940 | 0.817 |
| HMEC | held-out | 0.890 | 0.892 | 0.741 | 0.898 | 0.941 | 0.970 | 0.918 | 0.920 | 0.764 |
| HMEC deep | held-out | 0.924 | 0.923 | 0.771 | 0.932 | 0.941 | 0.985 | 0.938 | 0.937 | 0.783 |
| HUVEC | held-out | 0.888 | 0.879 | 0.791 | 0.887 | 0.896 | 0.947 | 0.938 | 0.929 | 0.836 |
| HUVEC deep | held-out | 0.914 | 0.903 | 0.815 | 0.916 | 0.896 | 0.972 | 0.940 | 0.929 | 0.838 |

**10 kb** (cells with a replicate pair)

| cell | set | paper | release | noref | consensus | replicate r | ceiling | paper/ceiling | release/ceiling | noref/ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| GM12878 | training | 0.933 | 0.936 | 0.841 | 0.858 | 0.779 | 0.936 | 0.997 | 1.000 | 0.898 |
| IMR-90 | training | 0.947 | 0.945 | 0.883 | 0.903 | 0.867 | 0.964 | 0.982 | 0.980 | 0.916 |
| K562 | held-out | 0.905 | 0.902 | 0.783 | 0.859 | 0.764 | 0.931 | 0.972 | 0.970 | 0.841 |
| HMEC | held-out | 0.803 | 0.808 | 0.666 | 0.809 | 0.780 | 0.883 | 0.909 | 0.915 | 0.754 |
| HMEC deep | held-out | 0.908 | 0.909 | 0.773 | 0.906 | 0.780 | 0.936 | 0.970 | 0.971 | 0.826 |
| HUVEC | held-out | 0.785 | 0.775 | 0.671 | 0.769 | 0.697 | 0.835 | 0.940 | 0.928 | 0.804 |
| HUVEC deep | held-out | 0.903 | 0.886 | 0.793 | 0.884 | 0.697 | 0.906 | 0.996 | 0.978 | 0.875 |

**5 kb** (cells with a replicate pair)

| cell | set | paper | release | noref | consensus | replicate r | ceiling | paper/ceiling | release/ceiling | noref/ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| GM12878 | training | 0.911 | 0.912 | 0.820 | 0.821 | 0.589 | 0.861 | 1.058 | 1.060 | 0.952 |
| IMR-90 | training | 0.909 | 0.907 | 0.825 | 0.857 | 0.712 | 0.912 | 0.997 | 0.994 | 0.905 |
| K562 | held-out | 0.834 | 0.831 | 0.690 | 0.785 | 0.525 | 0.830 | 1.004 | 1.002 | 0.832 |
| HMEC | held-out | 0.601 | 0.605 | 0.498 | 0.614 | 0.552 | 0.743 | 0.809 | 0.814 | 0.671 |
| HMEC deep | held-out | 0.858 | 0.864 | 0.718 | 0.851 | 0.552 | 0.843 | 1.017 | 1.025 | 0.851 |
| HUVEC | held-out | 0.490 | 0.492 | 0.440 | 0.495 | 0.455 | 0.674 | 0.727 | 0.730 | 0.652 |
| HUVEC deep | held-out | 0.830 | 0.829 | 0.712 | 0.821 | 0.455 | 0.791 | 1.050 | 1.048 | 0.900 |

**2 kb** (cells with a replicate pair)

| cell | set | paper | release | noref | consensus | replicate r | ceiling | paper/ceiling | release/ceiling | noref/ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| GM12878 | training | 0.790 | 0.784 | 0.659 | 0.661 | 0.386 | 0.746 | 1.059 | 1.050 | 0.883 |
| IMR-90 | training | 0.800 | 0.791 | 0.686 | 0.741 | 0.501 | 0.817 | 0.979 | 0.968 | 0.839 |
| K562 | held-out | 0.677 | 0.668 | 0.550 | 0.629 | 0.359 | 0.727 | 0.932 | 0.920 | 0.756 |
| HMEC | held-out | 0.337 | 0.347 | 0.269 | 0.355 | 0.317 | 0.563 | 0.598 | 0.616 | 0.478 |
| HMEC deep | held-out | 0.710 | 0.709 | 0.555 | 0.687 | 0.317 | 0.694 | 1.023 | 1.021 | 0.800 |
| HUVEC | held-out | 0.263 | 0.249 | 0.178 | 0.271 | 0.332 | 0.576 | 0.457 | 0.433 | 0.309 |
| HUVEC deep | held-out | 0.704 | 0.676 | 0.572 | 0.659 | 0.332 | 0.706 | 0.997 | 0.958 | 0.809 |

**Summary over the cells with a ceiling** (training: GM12878, IMR-90; held-out: K562, HMEC, HMEC deep, HUVEC, HUVEC deep)

| kb | set | model | cells | median model | median ceiling | median fraction | min | max | >= 95% |
|---|---|---|---|---|---|---|---|---|---|
| 25 | training | paper | 2 | 0.942 | 0.987 | 0.955 | 0.951 | 0.958 | 2/2 |
| 25 | training | release | 2 | 0.946 | 0.987 | 0.959 | 0.952 | 0.965 | 2/2 |
| 25 | training | noref | 2 | 0.851 | 0.987 | 0.862 | 0.843 | 0.881 | 0/2 |
| 25 | held-out | paper | 5 | 0.914 | 0.972 | 0.938 | 0.918 | 0.942 | 0/5 |
| 25 | held-out | release | 5 | 0.903 | 0.972 | 0.929 | 0.920 | 0.940 | 0/5 |
| 25 | held-out | noref | 5 | 0.791 | 0.972 | 0.817 | 0.764 | 0.838 | 0/5 |
| 10 | training | paper | 2 | 0.940 | 0.950 | 0.990 | 0.982 | 0.997 | 2/2 |
| 10 | training | release | 2 | 0.940 | 0.950 | 0.990 | 0.980 | 1.000 | 2/2 |
| 10 | training | noref | 2 | 0.862 | 0.950 | 0.907 | 0.898 | 0.916 | 0/2 |
| 10 | held-out | paper | 5 | 0.903 | 0.906 | 0.970 | 0.909 | 0.996 | 3/5 |
| 10 | held-out | release | 5 | 0.886 | 0.906 | 0.970 | 0.915 | 0.978 | 3/5 |
| 10 | held-out | noref | 5 | 0.773 | 0.906 | 0.826 | 0.754 | 0.875 | 0/5 |
| 5 | training | paper | 2 | 0.910 | 0.886 | 1.027 | 0.997 | 1.058 | 2/2 |
| 5 | training | release | 2 | 0.909 | 0.886 | 1.027 | 0.994 | 1.060 | 2/2 |
| 5 | training | noref | 2 | 0.822 | 0.886 | 0.928 | 0.905 | 0.952 | 1/2 |
| 5 | held-out | paper | 5 | 0.830 | 0.791 | 1.004 | 0.727 | 1.050 | 3/5 |
| 5 | held-out | release | 5 | 0.829 | 0.791 | 1.002 | 0.730 | 1.048 | 3/5 |
| 5 | held-out | noref | 5 | 0.690 | 0.791 | 0.832 | 0.652 | 0.900 | 0/5 |
| 2 | training | paper | 2 | 0.795 | 0.782 | 1.019 | 0.979 | 1.059 | 2/2 |
| 2 | training | release | 2 | 0.787 | 0.782 | 1.009 | 0.968 | 1.050 | 2/2 |
| 2 | training | noref | 2 | 0.672 | 0.782 | 0.861 | 0.839 | 0.883 | 0/2 |
| 2 | held-out | paper | 5 | 0.677 | 0.694 | 0.932 | 0.457 | 1.023 | 2/5 |
| 2 | held-out | release | 5 | 0.668 | 0.694 | 0.920 | 0.433 | 1.021 | 2/5 |
| 2 | held-out | noref | 5 | 0.550 | 0.694 | 0.756 | 0.309 | 0.809 | 0/5 |

**Deepest intact replicate pair as the ceiling** (GM12878 ENCSR965OVH, K562 ENCSR676BEP)

| cell | kb | paper | release | noref | deep pair r | ceiling | paper/ceiling | release/ceiling | noref/ceiling |
|---|---|---|---|---|---|---|---|---|---|
| GM12878 | 25 | 0.936 | 0.937 | 0.830 | 0.968 | 0.992 | 0.944 | 0.945 | 0.836 |
| GM12878 | 10 | 0.933 | 0.936 | 0.841 | 0.874 | 0.966 | 0.967 | 0.969 | 0.871 |
| GM12878 | 5 | 0.911 | 0.912 | 0.820 | 0.744 | 0.924 | 0.986 | 0.988 | 0.887 |
| GM12878 | 2 | 0.790 | 0.784 | 0.659 | 0.521 | 0.828 | 0.955 | 0.947 | 0.796 |
| K562 | 25 | 0.926 | 0.925 | 0.804 | 0.990 | 0.998 | 0.929 | 0.927 | 0.806 |
| K562 | 10 | 0.905 | 0.902 | 0.783 | 0.954 | 0.988 | 0.916 | 0.913 | 0.792 |
| K562 | 5 | 0.834 | 0.831 | 0.690 | 0.865 | 0.963 | 0.865 | 0.863 | 0.716 |
| K562 | 2 | 0.677 | 0.668 | 0.550 | 0.650 | 0.888 | 0.763 | 0.753 | 0.619 |

### Sentences of main.tex that change (Results "Accuracy approaches the agreement of biological replicates", Fig. 11a, Supplementary Table S7)

1. "At 10 kb, the reference-input model reached 97% to 100% of this limit on the deeply sequenced held-out cells" becomes "97% to 98%" (K562 0.970, HMEC deep 0.971, HUVEC deep 0.978 of the limit; paper model 0.972, 0.970, 0.996). The release model does not reach 100% on HUVEC deep (0.886 against 0.903 for the dedicated 10 kb model); GM12878 (a training cell) is at 1.000 (paper 0.997).
2. "and at 25 kb 92% to 96% on training and held-out cells" stays (release 0.920 for HMEC to 0.965 for IMR-90; paper 0.918 to 0.958).
3. "Against the deepest pair of intact Hi-C maps of K562, the model reached 87% of the limit at 5 kb and 76% at 2 kb" becomes "86% ... and 75%" (0.863 and 0.753; paper 0.865 and 0.763).
4. "The gap grows at higher resolution." stays (release fractions of the limit over the held-out cells with a ceiling: median 0.929, 0.969, 1.001, 0.920 at 25, 10, 5, 2 kb; 5 kb is at the limit because the 5 kb ceilings of the deep merged maps are low).
5. "The results were reproducible between training runs. Three seeds gave medians between 0.907 and 0.914 at 25 kb on the held-out cells, and every seed exceeded the consensus (Supplementary Fig. S7a,b; Supplementary Table S9)." The three seeds belong to the dedicated 25 kb models. The release checkpoint is one run (no seed spread exists, registry 93), so the sentence has to say that the seeds refer to the dedicated models, or be dropped.
6. Fig. 11a and Supplementary Table S7 are regenerated from `results/one_model/remaining/replicate_ceiling/replicate_ceiling_table.csv` (column `release_model`, `ratio_release`; the figure script `fig4a_6_consensus_ceiling.py` reads `results/replicate_ceiling/replicate_ceiling_table.csv` with the columns of the old table, so the release table needs a column rename: model_refin_drop <- release_model, ratio_refin_drop <- ratio_release).
7. Discussion: "At 10 kb, the predictions reached the agreement of biological replicates on the held-out cells, and the largest remaining gap is at 2 kb" stays (2 kb held-out: HMEC 0.62, HUVEC 0.43, K562 0.92 of the limit; deep maps HMEC 1.02, HUVEC deep 0.96).

## 5. Comparison with Akita on its 413 GM12878 test regions

Protocol of registry section 73 (`scripts/structure_scores_vs_akita.py` stage akita, called unchanged by `scripts/one_model_remaining_akita.py`): Akita's 2048 bp grid and targets, the 448-bin crop, one pixel mask per region, median over regions and pooled values with 2000 region bootstrap resamples. PAPER rows: `akita_native_honest` (ChIP only) and `akita_native_refres` (with the consensus of four other cells; models trained on Akita's own normalised targets). RELEASE rows: the 2 kb view of the release checkpoint predicts the 2000 bp grid of every sequence (GM12878 marks, leave-one-out consensus of the other 11 training cells as reference) and is resampled to Akita's bin centres by bilinear interpolation, the procedure of `score_like_hicgan.py`. Differences from the paper rows: the release checkpoint is trained on `proper` O/E maps and not on Akita's normalisation (the scoring target is Akita's own); the reference is the 11-cell leave-one-out consensus, not the four-cell consensus; GM12878 is a training cell (the 413 regions are excluded from the training windows); the Meth track of GM12878 is part of the "+ Meth" rows.

| predictor | regions | pearson | scc | ins_r_10k | ins_r_50k | ins_r_250k | bnd_f1 | bnd_strength_r | moc | stripe_r | compartment_corr | pooled bnd F1 | hFT bnd recall | loop recall | APA ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Akita (GM12878 head) | 413 | 0.548 | 0.510 | 0.391 | 0.688 | 0.847 | 0.500 | 0.608 | 0.625 | 0.697 | 0.710 | 0.436 [0.411, 0.462] | 0.494 [0.471, 0.519] | 0.314 [0.280, 0.346] | 0.264 [0.243, 0.286] |
| Hi-cGAN (submitted) | 408 | 0.236 | 0.207 | 0.209 | 0.376 | 0.567 | 0.222 | 0.227 | 0.516 | 0.309 | 0.546 | 0.243 [0.225, 0.263] | 0.321 [0.298, 0.344] | 0.137 [0.111, 0.162] | 0.179 [0.146, 0.215] |
| ours, ChIP only | 413 | 0.638 | 0.610 | 0.717 | 0.819 | 0.919 | 0.571 | 0.730 | 0.674 | 0.791 | 0.776 | 0.487 [0.457, 0.518] | 0.548 [0.522, 0.573] | 0.383 [0.345, 0.420] | 0.381 [0.356, 0.408] |
| ours, with consensus | 413 | 0.746 | 0.694 | 0.779 | 0.920 | 0.971 | 0.667 | 0.871 | 0.718 | 0.894 | 0.885 | 0.575 [0.545, 0.609] | 0.647 [0.621, 0.672] | 0.571 [0.537, 0.603] | 0.845 [0.814, 0.878] |
| release, marks + Meth only | 413 | 0.595 | 0.598 | 0.713 | 0.820 | 0.903 | 0.571 | 0.754 | 0.676 | 0.764 | 0.729 | 0.504 [0.476, 0.536] | 0.550 [0.525, 0.576] | 0.432 [0.399, 0.466] | 0.624 [0.584, 0.670] |
| release, 13 marks only | 413 | 0.578 | 0.587 | 0.702 | 0.809 | 0.900 | 0.571 | 0.742 | 0.676 | 0.755 | 0.717 | 0.493 [0.465, 0.525] | 0.551 [0.524, 0.578] | 0.428 [0.394, 0.461] | 0.610 [0.572, 0.655] |
| release, with consensus (+ Meth) | 413 | 0.676 | 0.673 | 0.747 | 0.912 | 0.965 | 0.667 | 0.868 | 0.715 | 0.869 | 0.795 | 0.569 [0.541, 0.601] | 0.625 [0.602, 0.648] | 0.530 [0.494, 0.565] | 0.938 [0.890, 0.991] |
| release, with consensus, 13 marks | 413 | 0.674 | 0.672 | 0.735 | 0.907 | 0.963 | 0.667 | 0.864 | 0.704 | 0.869 | 0.784 | 0.572 [0.545, 0.603] | 0.624 [0.601, 0.648] | 0.530 [0.497, 0.564] | 0.922 [0.875, 0.973] |
| control: dedicated 2 kb model (proper targets), with consensus | 413 | 0.670 | 0.684 | 0.779 | 0.919 | 0.968 | 0.667 | 0.883 | 0.715 | 0.862 | 0.802 | 0.606 [0.581, 0.633] | 0.623 [0.596, 0.647] | 0.561 [0.526, 0.595] | 0.890 [0.848, 0.934] |
| control: dedicated 2 kb model (proper targets), ChIP only | 413 | 0.603 | 0.596 | 0.706 | 0.826 | 0.915 | 0.600 | 0.751 | 0.681 | 0.783 | 0.749 | 0.509 [0.480, 0.539] | 0.561 [0.535, 0.587] | 0.414 [0.380, 0.449] | 0.615 [0.578, 0.656] |
| consensus of 4 measured cells | 413 | 0.611 | 0.559 | 0.474 | 0.777 | 0.916 | 0.598 | 0.724 | 0.656 | 0.759 | 0.797 | 0.533 [0.511, 0.556] | 0.589 [0.564, 0.615] | 0.514 [0.479, 0.549] | 1.025 [0.972, 1.076] |
| HFF measured map | 413 | 0.514 | 0.469 | 0.380 | 0.705 | 0.893 | 0.500 | 0.639 | 0.612 | 0.669 | 0.716 | 0.488 [0.469, 0.509] | 0.522 [0.497, 0.548] | 0.418 [0.384, 0.451] | 1.552 [1.453, 1.654] |

**Paired against Akita** (ours minus Akita, median over regions [95 % interval]; regions ahead)

| predictor | measure | difference to Akita | regions ahead |
|---|---|---|---|
| ours, ChIP only | pearson | 0.040 [0.029, 0.050] | 274 of 413 |
| ours, ChIP only | scc | 0.061 [0.051, 0.074] | 309 of 413 |
| ours, ChIP only | ins_r_10k | 0.272 [0.254, 0.281] | 397 of 413 |
| ours, ChIP only | ins_r_25k | 0.176 [0.159, 0.190] | 373 of 413 |
| ours, ChIP only | ins_r_50k | 0.096 [0.075, 0.116] | 323 of 413 |
| ours, ChIP only | ins_r_100k | 0.058 [0.040, 0.080] | 279 of 413 |
| ours, ChIP only | ins_r_250k | 0.026 [0.013, 0.037] | 248 of 413 |
| ours, ChIP only | bnd_f1 | 0.048 [0.000, 0.083] | 223 of 413 |
| ours, ChIP only | bnd_f1_strict | 0.029 [0.000, 0.067] | 214 of 413 |
| ours, ChIP only | bnd_recall | 0.000 [0.000, 0.000] | 184 of 413 |
| ours, ChIP only | bnd_precision | 0.000 [0.000, 0.067] | 199 of 410 |
| ours, ChIP only | bnd_strength_r | 0.075 [0.054, 0.095] | 266 of 413 |
| ours, ChIP only | moc | 0.026 [0.012, 0.035] | 249 of 413 |
| ours, ChIP only | stripe_r | 0.077 [0.057, 0.101] | 296 of 413 |
| ours, ChIP only | loop_enrich_corr | 0.024 [0.021, 0.026] | 360 of 413 |
| ours, ChIP only | compartment_corr | 0.029 [0.016, 0.042] | 263 of 413 |
| ours, ChIP only | eigenvector_abs_corr | 0.017 [0.006, 0.030] | 238 of 413 |
| ours, ChIP only | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| ours, ChIP only | hft_bnd_recall_per_region | 0.000 [nan, nan] | 141 of 404 |
| ours, ChIP only | loop_recall_per_region | 0.000 [nan, nan] | 89 of 287 |
| ours, ChIP only | bnd_f1_pooled (pooled) | 0.051 [0.028, 0.075] |  |
| ours, ChIP only | bnd_recall_pooled (pooled) | 0.038 [0.017, 0.061] |  |
| ours, ChIP only | bnd_precision_pooled (pooled) | 0.075 [0.044, 0.105] |  |
| ours, ChIP only | hft_bnd_recall (pooled) | 0.054 [0.029, 0.080] |  |
| ours, ChIP only | loop_recall (pooled) | 0.069 [0.025, 0.115] |  |
| ours, ChIP only | apa_ratio (pooled) | 0.117 [0.086, 0.148] |  |
| ours, with consensus | pearson | 0.136 [0.124, 0.149] | 401 of 413 |
| ours, with consensus | scc | 0.140 [0.128, 0.154] | 405 of 413 |
| ours, with consensus | ins_r_10k | 0.324 [0.315, 0.343] | 408 of 413 |
| ours, with consensus | ins_r_25k | 0.255 [0.235, 0.267] | 409 of 413 |
| ours, with consensus | ins_r_50k | 0.190 [0.175, 0.213] | 398 of 413 |
| ours, with consensus | ins_r_100k | 0.140 [0.121, 0.165] | 386 of 413 |
| ours, with consensus | ins_r_250k | 0.072 [0.055, 0.094] | 331 of 413 |
| ours, with consensus | bnd_f1 | 0.127 [0.100, 0.159] | 281 of 413 |
| ours, with consensus | bnd_f1_strict | 0.167 [0.123, 0.197] | 280 of 413 |
| ours, with consensus | bnd_recall | 0.125 [0.000, 0.167] | 225 of 413 |
| ours, with consensus | bnd_precision | 0.150 [0.083, 0.167] | 245 of 409 |
| ours, with consensus | bnd_strength_r | 0.181 [0.157, 0.202] | 365 of 413 |
| ours, with consensus | moc | 0.063 [0.048, 0.081] | 307 of 413 |
| ours, with consensus | stripe_r | 0.173 [0.148, 0.188] | 389 of 413 |
| ours, with consensus | loop_enrich_corr | 0.039 [0.036, 0.042] | 384 of 413 |
| ours, with consensus | compartment_corr | 0.099 [0.081, 0.111] | 376 of 413 |
| ours, with consensus | eigenvector_abs_corr | 0.062 [0.045, 0.075] | 327 of 413 |
| ours, with consensus | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| ours, with consensus | hft_bnd_recall_per_region | 0.000 [nan, nan] | 196 of 404 |
| ours, with consensus | loop_recall_per_region | 0.000 [nan, nan] | 140 of 287 |
| ours, with consensus | bnd_f1_pooled (pooled) | 0.140 [0.118, 0.162] |  |
| ours, with consensus | bnd_recall_pooled (pooled) | 0.122 [0.101, 0.145] |  |
| ours, with consensus | bnd_precision_pooled (pooled) | 0.158 [0.132, 0.184] |  |
| ours, with consensus | hft_bnd_recall (pooled) | 0.153 [0.129, 0.176] |  |
| ours, with consensus | loop_recall (pooled) | 0.257 [0.221, 0.296] |  |
| ours, with consensus | apa_ratio (pooled) | 0.581 [0.551, 0.612] |  |
| release, marks + Meth only | pearson | 0.006 [-0.006, 0.017] | 216 of 413 |
| release, marks + Meth only | scc | 0.056 [0.038, 0.072] | 286 of 413 |
| release, marks + Meth only | ins_r_10k | 0.264 [0.252, 0.279] | 389 of 413 |
| release, marks + Meth only | ins_r_25k | 0.170 [0.155, 0.189] | 356 of 413 |
| release, marks + Meth only | ins_r_50k | 0.098 [0.076, 0.117] | 306 of 413 |
| release, marks + Meth only | ins_r_100k | 0.042 [0.028, 0.065] | 258 of 413 |
| release, marks + Meth only | ins_r_250k | 0.019 [0.006, 0.043] | 237 of 413 |
| release, marks + Meth only | bnd_f1 | 0.067 [0.016, 0.100] | 228 of 413 |
| release, marks + Meth only | bnd_f1_strict | 0.022 [0.000, 0.061] | 213 of 413 |
| release, marks + Meth only | bnd_recall | 0.000 [0.000, 0.000] | 184 of 413 |
| release, marks + Meth only | bnd_precision | 0.045 [0.000, 0.083] | 209 of 410 |
| release, marks + Meth only | bnd_strength_r | 0.082 [0.063, 0.102] | 268 of 413 |
| release, marks + Meth only | moc | 0.032 [0.017, 0.044] | 252 of 413 |
| release, marks + Meth only | stripe_r | 0.044 [0.027, 0.069] | 251 of 413 |
| release, marks + Meth only | loop_enrich_corr | 0.011 [0.006, 0.015] | 250 of 413 |
| release, marks + Meth only | compartment_corr | -0.001 [-0.016, 0.010] | 204 of 413 |
| release, marks + Meth only | eigenvector_abs_corr | -0.012 [-0.027, 0.003] | 195 of 413 |
| release, marks + Meth only | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 7 |
| release, marks + Meth only | hft_bnd_recall_per_region | 0.000 [nan, nan] | 147 of 404 |
| release, marks + Meth only | loop_recall_per_region | 0.000 [nan, nan] | 104 of 287 |
| release, marks + Meth only | bnd_f1_pooled (pooled) | 0.068 [0.045, 0.094] |  |
| release, marks + Meth only | bnd_recall_pooled (pooled) | 0.063 [0.041, 0.087] |  |
| release, marks + Meth only | bnd_precision_pooled (pooled) | 0.069 [0.040, 0.100] |  |
| release, marks + Meth only | hft_bnd_recall (pooled) | 0.057 [0.029, 0.083] |  |
| release, marks + Meth only | loop_recall (pooled) | 0.119 [0.076, 0.163] |  |
| release, marks + Meth only | apa_ratio (pooled) | 0.360 [0.320, 0.406] |  |
| release, 13 marks only | pearson | -0.003 [-0.016, 0.014] | 201 of 413 |
| release, 13 marks only | scc | 0.049 [0.036, 0.065] | 277 of 413 |
| release, 13 marks only | ins_r_10k | 0.256 [0.239, 0.266] | 386 of 413 |
| release, 13 marks only | ins_r_25k | 0.162 [0.148, 0.183] | 354 of 413 |
| release, 13 marks only | ins_r_50k | 0.082 [0.067, 0.108] | 293 of 413 |
| release, 13 marks only | ins_r_100k | 0.037 [0.021, 0.057] | 251 of 413 |
| release, 13 marks only | ins_r_250k | 0.019 [0.005, 0.038] | 233 of 413 |
| release, 13 marks only | bnd_f1 | 0.063 [0.000, 0.100] | 223 of 413 |
| release, 13 marks only | bnd_f1_strict | 0.015 [0.000, 0.071] | 209 of 413 |
| release, 13 marks only | bnd_recall | 0.000 [0.000, 0.000] | 181 of 413 |
| release, 13 marks only | bnd_precision | 0.000 [0.000, 0.071] | 201 of 409 |
| release, 13 marks only | bnd_strength_r | 0.080 [0.053, 0.109] | 256 of 413 |
| release, 13 marks only | moc | 0.026 [0.012, 0.040] | 241 of 413 |
| release, 13 marks only | stripe_r | 0.037 [0.019, 0.059] | 242 of 413 |
| release, 13 marks only | loop_enrich_corr | 0.008 [0.003, 0.011] | 241 of 413 |
| release, 13 marks only | compartment_corr | -0.009 [-0.029, 0.005] | 196 of 413 |
| release, 13 marks only | eigenvector_abs_corr | -0.022 [-0.047, -0.007] | 180 of 413 |
| release, 13 marks only | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 7 |
| release, 13 marks only | hft_bnd_recall_per_region | 0.000 [nan, nan] | 146 of 404 |
| release, 13 marks only | loop_recall_per_region | 0.000 [nan, nan] | 103 of 287 |
| release, 13 marks only | bnd_f1_pooled (pooled) | 0.058 [0.034, 0.083] |  |
| release, 13 marks only | bnd_recall_pooled (pooled) | 0.056 [0.034, 0.080] |  |
| release, 13 marks only | bnd_precision_pooled (pooled) | 0.052 [0.021, 0.082] |  |
| release, 13 marks only | hft_bnd_recall (pooled) | 0.058 [0.029, 0.085] |  |
| release, 13 marks only | loop_recall (pooled) | 0.114 [0.071, 0.158] |  |
| release, 13 marks only | apa_ratio (pooled) | 0.346 [0.307, 0.391] |  |
| release, with consensus (+ Meth) | pearson | 0.067 [0.050, 0.080] | 308 of 413 |
| release, with consensus (+ Meth) | scc | 0.118 [0.106, 0.129] | 376 of 413 |
| release, with consensus (+ Meth) | ins_r_10k | 0.302 [0.281, 0.315] | 399 of 413 |
| release, with consensus (+ Meth) | ins_r_25k | 0.232 [0.213, 0.243] | 393 of 413 |
| release, with consensus (+ Meth) | ins_r_50k | 0.164 [0.152, 0.180] | 382 of 413 |
| release, with consensus (+ Meth) | ins_r_100k | 0.120 [0.104, 0.146] | 369 of 413 |
| release, with consensus (+ Meth) | ins_r_250k | 0.067 [0.047, 0.090] | 319 of 413 |
| release, with consensus (+ Meth) | bnd_f1 | 0.133 [0.111, 0.167] | 287 of 413 |
| release, with consensus (+ Meth) | bnd_f1_strict | 0.143 [0.102, 0.182] | 278 of 413 |
| release, with consensus (+ Meth) | bnd_recall | 0.091 [0.000, 0.143] | 223 of 413 |
| release, with consensus (+ Meth) | bnd_precision | 0.167 [0.150, 0.200] | 260 of 411 |
| release, with consensus (+ Meth) | bnd_strength_r | 0.170 [0.152, 0.196] | 358 of 413 |
| release, with consensus (+ Meth) | moc | 0.058 [0.044, 0.077] | 305 of 413 |
| release, with consensus (+ Meth) | stripe_r | 0.124 [0.106, 0.141] | 346 of 413 |
| release, with consensus (+ Meth) | loop_enrich_corr | 0.007 [0.002, 0.011] | 239 of 413 |
| release, with consensus (+ Meth) | compartment_corr | 0.031 [0.021, 0.047] | 270 of 413 |
| release, with consensus (+ Meth) | eigenvector_abs_corr | 0.005 [0.000, 0.016] | 227 of 413 |
| release, with consensus (+ Meth) | bnd_f1_struct | 0.000 [0.000, 0.500] | 1 of 6 |
| release, with consensus (+ Meth) | hft_bnd_recall_per_region | 0.000 [nan, nan] | 185 of 404 |
| release, with consensus (+ Meth) | loop_recall_per_region | 0.000 [nan, nan] | 127 of 287 |
| release, with consensus (+ Meth) | bnd_f1_pooled (pooled) | 0.133 [0.113, 0.154] |  |
| release, with consensus (+ Meth) | bnd_recall_pooled (pooled) | 0.116 [0.096, 0.138] |  |
| release, with consensus (+ Meth) | bnd_precision_pooled (pooled) | 0.152 [0.124, 0.183] |  |
| release, with consensus (+ Meth) | hft_bnd_recall (pooled) | 0.131 [0.106, 0.155] |  |
| release, with consensus (+ Meth) | loop_recall (pooled) | 0.216 [0.176, 0.259] |  |
| release, with consensus (+ Meth) | apa_ratio (pooled) | 0.673 [0.624, 0.724] |  |
| release, with consensus, 13 marks | pearson | 0.061 [0.050, 0.075] | 305 of 413 |
| release, with consensus, 13 marks | scc | 0.114 [0.100, 0.123] | 375 of 413 |
| release, with consensus, 13 marks | ins_r_10k | 0.288 [0.273, 0.304] | 396 of 413 |
| release, with consensus, 13 marks | ins_r_25k | 0.221 [0.208, 0.236] | 389 of 413 |
| release, with consensus, 13 marks | ins_r_50k | 0.160 [0.143, 0.171] | 377 of 413 |
| release, with consensus, 13 marks | ins_r_100k | 0.118 [0.100, 0.142] | 372 of 413 |
| release, with consensus, 13 marks | ins_r_250k | 0.059 [0.046, 0.084] | 319 of 413 |
| release, with consensus, 13 marks | bnd_f1 | 0.133 [0.109, 0.167] | 289 of 413 |
| release, with consensus, 13 marks | bnd_f1_strict | 0.143 [0.100, 0.182] | 271 of 413 |
| release, with consensus, 13 marks | bnd_recall | 0.100 [0.000, 0.143] | 224 of 413 |
| release, with consensus, 13 marks | bnd_precision | 0.167 [0.100, 0.200] | 253 of 411 |
| release, with consensus, 13 marks | bnd_strength_r | 0.167 [0.148, 0.188] | 359 of 413 |
| release, with consensus, 13 marks | moc | 0.058 [0.045, 0.079] | 295 of 413 |
| release, with consensus, 13 marks | stripe_r | 0.123 [0.101, 0.140] | 344 of 413 |
| release, with consensus, 13 marks | loop_enrich_corr | 0.005 [0.001, 0.010] | 235 of 413 |
| release, with consensus, 13 marks | compartment_corr | 0.031 [0.020, 0.044] | 267 of 413 |
| release, with consensus, 13 marks | eigenvector_abs_corr | 0.006 [-0.002, 0.015] | 224 of 413 |
| release, with consensus, 13 marks | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| release, with consensus, 13 marks | hft_bnd_recall_per_region | 0.000 [nan, nan] | 184 of 404 |
| release, with consensus, 13 marks | loop_recall_per_region | 0.000 [nan, nan] | 127 of 287 |
| release, with consensus, 13 marks | bnd_f1_pooled (pooled) | 0.137 [0.116, 0.158] |  |
| release, with consensus, 13 marks | bnd_recall_pooled (pooled) | 0.122 [0.103, 0.144] |  |
| release, with consensus, 13 marks | bnd_precision_pooled (pooled) | 0.148 [0.119, 0.178] |  |
| release, with consensus, 13 marks | hft_bnd_recall (pooled) | 0.130 [0.104, 0.153] |  |
| release, with consensus, 13 marks | loop_recall (pooled) | 0.216 [0.176, 0.258] |  |
| release, with consensus, 13 marks | apa_ratio (pooled) | 0.658 [0.610, 0.706] |  |
| control: dedicated 2 kb model (proper targets), ChIP only | pearson | 0.013 [0.002, 0.024] | 231 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | scc | 0.063 [0.046, 0.076] | 304 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | ins_r_10k | 0.260 [0.240, 0.276] | 384 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | ins_r_25k | 0.167 [0.149, 0.189] | 362 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | ins_r_50k | 0.092 [0.075, 0.112] | 304 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | ins_r_100k | 0.043 [0.035, 0.060] | 269 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | ins_r_250k | 0.020 [0.009, 0.040] | 244 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_f1 | 0.083 [0.048, 0.111] | 235 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_f1_strict | 0.022 [0.000, 0.081] | 213 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_recall | 0.000 [0.000, 0.077] | 190 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_precision | 0.057 [0.000, 0.100] | 213 of 409 |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_strength_r | 0.073 [0.053, 0.105] | 258 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | moc | 0.034 [0.023, 0.048] | 261 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | stripe_r | 0.055 [0.036, 0.070] | 261 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | loop_enrich_corr | 0.007 [0.004, 0.011] | 248 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | compartment_corr | 0.009 [-0.004, 0.022] | 219 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | eigenvector_abs_corr | 0.004 [-0.004, 0.015] | 216 of 413 |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| control: dedicated 2 kb model (proper targets), ChIP only | hft_bnd_recall_per_region | 0.000 [nan, nan] | 155 of 404 |
| control: dedicated 2 kb model (proper targets), ChIP only | loop_recall_per_region | 0.000 [nan, nan] | 107 of 287 |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_f1_pooled (pooled) | 0.073 [0.050, 0.097] |  |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_recall_pooled (pooled) | 0.066 [0.045, 0.088] |  |
| control: dedicated 2 kb model (proper targets), ChIP only | bnd_precision_pooled (pooled) | 0.077 [0.045, 0.105] |  |
| control: dedicated 2 kb model (proper targets), ChIP only | hft_bnd_recall (pooled) | 0.067 [0.040, 0.094] |  |
| control: dedicated 2 kb model (proper targets), ChIP only | loop_recall (pooled) | 0.100 [0.057, 0.144] |  |
| control: dedicated 2 kb model (proper targets), ChIP only | apa_ratio (pooled) | 0.351 [0.314, 0.392] |  |
| control: dedicated 2 kb model (proper targets), with consensus | pearson | 0.068 [0.047, 0.086] | 298 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | scc | 0.129 [0.109, 0.138] | 373 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | ins_r_10k | 0.331 [0.318, 0.345] | 404 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | ins_r_25k | 0.247 [0.232, 0.269] | 403 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | ins_r_50k | 0.182 [0.166, 0.200] | 385 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | ins_r_100k | 0.127 [0.114, 0.147] | 369 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | ins_r_250k | 0.064 [0.047, 0.089] | 327 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_f1 | 0.179 [0.143, 0.200] | 309 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_f1_strict | 0.182 [0.143, 0.222] | 289 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_recall | 0.167 [0.125, 0.200] | 242 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_precision | 0.167 [0.150, 0.200] | 270 of 411 |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_strength_r | 0.188 [0.166, 0.209] | 360 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | moc | 0.080 [0.068, 0.097] | 305 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | stripe_r | 0.130 [0.114, 0.149] | 342 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | loop_enrich_corr | 0.007 [0.001, 0.011] | 231 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | compartment_corr | 0.032 [0.022, 0.044] | 274 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | eigenvector_abs_corr | 0.005 [-0.000, 0.018] | 225 of 413 |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| control: dedicated 2 kb model (proper targets), with consensus | hft_bnd_recall_per_region | 0.000 [nan, nan] | 180 of 404 |
| control: dedicated 2 kb model (proper targets), with consensus | loop_recall_per_region | 0.167 [nan, nan] | 144 of 287 |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_f1_pooled (pooled) | 0.170 [0.148, 0.192] |  |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_recall_pooled (pooled) | 0.169 [0.146, 0.191] |  |
| control: dedicated 2 kb model (proper targets), with consensus | bnd_precision_pooled (pooled) | 0.149 [0.117, 0.184] |  |
| control: dedicated 2 kb model (proper targets), with consensus | hft_bnd_recall (pooled) | 0.129 [0.102, 0.154] |  |
| control: dedicated 2 kb model (proper targets), with consensus | loop_recall (pooled) | 0.247 [0.207, 0.285] |  |
| control: dedicated 2 kb model (proper targets), with consensus | apa_ratio (pooled) | 0.626 [0.583, 0.671] |  |

**Against the consensus of four measured cells**

| predictor | measure | difference to the consensus of 4 cells | regions ahead |
|---|---|---|---|
| ours, with consensus | pearson | 0.098 [0.090, 0.106] | 409 of 413 |
| ours, with consensus | scc | 0.114 [0.107, 0.122] | 411 of 413 |
| ours, with consensus | ins_r_10k | 0.237 [0.216, 0.251] | 397 of 413 |
| ours, with consensus | ins_r_25k | 0.180 [0.161, 0.200] | 393 of 413 |
| ours, with consensus | ins_r_50k | 0.123 [0.110, 0.135] | 395 of 413 |
| ours, with consensus | ins_r_100k | 0.082 [0.070, 0.100] | 375 of 413 |
| ours, with consensus | ins_r_250k | 0.032 [0.023, 0.046] | 318 of 413 |
| ours, with consensus | bnd_f1 | 0.050 [0.000, 0.068] | 224 of 413 |
| ours, with consensus | bnd_f1_strict | 0.081 [0.050, 0.105] | 255 of 413 |
| ours, with consensus | bnd_recall | 0.000 [0.000, 0.000] | 103 of 413 |
| ours, with consensus | bnd_precision | 0.143 [0.114, 0.167] | 258 of 410 |
| ours, with consensus | bnd_strength_r | 0.095 [0.085, 0.111] | 360 of 413 |
| ours, with consensus | moc | 0.028 [0.019, 0.045] | 270 of 413 |
| ours, with consensus | stripe_r | 0.110 [0.094, 0.122] | 372 of 413 |
| ours, with consensus | loop_enrich_corr | 0.066 [0.062, 0.070] | 398 of 413 |
| ours, with consensus | compartment_corr | 0.043 [0.032, 0.054] | 334 of 413 |
| ours, with consensus | eigenvector_abs_corr | 0.020 [0.012, 0.031] | 292 of 413 |
| ours, with consensus | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 8 |
| ours, with consensus | hft_bnd_recall_per_region | 0.000 [nan, nan] | 116 of 404 |
| ours, with consensus | loop_recall_per_region | 0.000 [nan, nan] | 62 of 287 |
| ours, with consensus | bnd_f1_pooled (pooled) | 0.042 [0.023, 0.063] |  |
| ours, with consensus | bnd_recall_pooled (pooled) | -0.032 [-0.052, -0.010] |  |
| ours, with consensus | bnd_precision_pooled (pooled) | 0.173 [0.152, 0.196] |  |
| ours, with consensus | hft_bnd_recall (pooled) | 0.058 [0.039, 0.078] |  |
| ours, with consensus | loop_recall (pooled) | 0.057 [0.028, 0.085] |  |
| ours, with consensus | apa_ratio (pooled) | -0.181 [-0.213, -0.147] |  |
| release, with consensus (+ Meth) | pearson | 0.033 [0.025, 0.040] | 293 of 413 |
| release, with consensus (+ Meth) | scc | 0.090 [0.081, 0.098] | 393 of 413 |
| release, with consensus (+ Meth) | ins_r_10k | 0.214 [0.185, 0.232] | 387 of 413 |
| release, with consensus (+ Meth) | ins_r_25k | 0.151 [0.137, 0.171] | 367 of 413 |
| release, with consensus (+ Meth) | ins_r_50k | 0.106 [0.094, 0.116] | 352 of 413 |
| release, with consensus (+ Meth) | ins_r_100k | 0.067 [0.054, 0.079] | 337 of 413 |
| release, with consensus (+ Meth) | ins_r_250k | 0.027 [0.018, 0.037] | 300 of 413 |
| release, with consensus (+ Meth) | bnd_f1 | 0.048 [0.000, 0.071] | 227 of 413 |
| release, with consensus (+ Meth) | bnd_f1_strict | 0.061 [0.038, 0.084] | 249 of 413 |
| release, with consensus (+ Meth) | bnd_recall | 0.000 [0.000, 0.000] | 100 of 413 |
| release, with consensus (+ Meth) | bnd_precision | 0.150 [0.122, 0.167] | 265 of 412 |
| release, with consensus (+ Meth) | bnd_strength_r | 0.087 [0.073, 0.103] | 350 of 413 |
| release, with consensus (+ Meth) | moc | 0.034 [0.023, 0.047] | 267 of 413 |
| release, with consensus (+ Meth) | stripe_r | 0.072 [0.059, 0.085] | 310 of 413 |
| release, with consensus (+ Meth) | loop_enrich_corr | 0.032 [0.027, 0.036] | 332 of 413 |
| release, with consensus (+ Meth) | compartment_corr | -0.019 [-0.022, -0.006] | 177 of 413 |
| release, with consensus (+ Meth) | eigenvector_abs_corr | -0.008 [-0.020, -0.002] | 179 of 413 |
| release, with consensus (+ Meth) | bnd_f1_struct | 0.000 [0.000, 0.500] | 1 of 6 |
| release, with consensus (+ Meth) | hft_bnd_recall_per_region | 0.000 [nan, nan] | 120 of 404 |
| release, with consensus (+ Meth) | loop_recall_per_region | 0.000 [nan, nan] | 69 of 287 |
| release, with consensus (+ Meth) | bnd_f1_pooled (pooled) | 0.036 [0.017, 0.056] |  |
| release, with consensus (+ Meth) | bnd_recall_pooled (pooled) | -0.038 [-0.058, -0.018] |  |
| release, with consensus (+ Meth) | bnd_precision_pooled (pooled) | 0.168 [0.144, 0.194] |  |
| release, with consensus (+ Meth) | hft_bnd_recall (pooled) | 0.036 [0.017, 0.055] |  |
| release, with consensus (+ Meth) | loop_recall (pooled) | 0.016 [-0.020, 0.049] |  |
| release, with consensus (+ Meth) | apa_ratio (pooled) | -0.088 [-0.137, -0.036] |  |
| release, marks + Meth only | pearson | -0.029 [-0.037, -0.018] | 157 of 413 |
| release, marks + Meth only | scc | 0.026 [0.016, 0.035] | 261 of 413 |
| release, marks + Meth only | ins_r_10k | 0.175 [0.150, 0.194] | 353 of 413 |
| release, marks + Meth only | ins_r_25k | 0.114 [0.099, 0.131] | 306 of 413 |
| release, marks + Meth only | ins_r_50k | 0.031 [0.016, 0.044] | 246 of 413 |
| release, marks + Meth only | ins_r_100k | -0.011 [-0.032, 0.009] | 197 of 413 |
| release, marks + Meth only | ins_r_250k | -0.005 [-0.012, 0.003] | 193 of 413 |
| release, marks + Meth only | bnd_f1 | 0.000 [-0.013, 0.000] | 173 of 413 |
| release, marks + Meth only | bnd_f1_strict | 0.000 [-0.029, 0.000] | 162 of 413 |
| release, marks + Meth only | bnd_recall | 0.000 [0.000, 0.000] | 88 of 413 |
| release, marks + Meth only | bnd_precision | 0.024 [0.000, 0.083] | 206 of 411 |
| release, marks + Meth only | bnd_strength_r | -0.003 [-0.020, 0.017] | 204 of 413 |
| release, marks + Meth only | moc | 0.001 [-0.012, 0.011] | 208 of 413 |
| release, marks + Meth only | stripe_r | -0.003 [-0.029, 0.016] | 206 of 413 |
| release, marks + Meth only | loop_enrich_corr | 0.034 [0.030, 0.039] | 335 of 413 |
| release, marks + Meth only | compartment_corr | -0.062 [-0.076, -0.042] | 123 of 413 |
| release, marks + Meth only | eigenvector_abs_corr | -0.040 [-0.058, -0.022] | 150 of 413 |
| release, marks + Meth only | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 7 |
| release, marks + Meth only | hft_bnd_recall_per_region | 0.000 [nan, nan] | 103 of 404 |
| release, marks + Meth only | loop_recall_per_region | 0.000 [nan, nan] | 54 of 287 |
| release, marks + Meth only | bnd_f1_pooled (pooled) | -0.029 [-0.052, -0.005] |  |
| release, marks + Meth only | bnd_recall_pooled (pooled) | -0.092 [-0.115, -0.069] |  |
| release, marks + Meth only | bnd_precision_pooled (pooled) | 0.085 [0.057, 0.114] |  |
| release, marks + Meth only | hft_bnd_recall (pooled) | -0.038 [-0.066, -0.013] |  |
| release, marks + Meth only | loop_recall (pooled) | -0.082 [-0.125, -0.039] |  |
| release, marks + Meth only | apa_ratio (pooled) | -0.401 [-0.462, -0.341] |  |

**Gain over the consensus** (regions ahead of both Akita and the consensus; the region at the 90th percentile of the smaller of both gains and the correlation of the two difference maps after averaging to 80 bins of 10 240 bp, the strip of the manuscript)

| model | regions | ahead of Akita | share ahead of Akita and consensus | median gain over consensus (Pearson) | p90 region | r(pred, measured) | r(consensus, measured) | r(measured - cons, pred - cons) |
|---|---|---|---|---|---|---|---|---|
| akita_native_refres | 413 | 401 | 0.966 | +0.098 | chr20:47613952-48662528 | 0.80 | 0.60 | 0.69 |
| all12one_fullrd_cons | 413 | 308 | 0.613 | +0.033 | chr5:76333056-77381632 | 0.66 | 0.53 | 0.65 |
| all12one_fullrd_cons_nometh | 413 | 305 | 0.596 | +0.029 | chr6:69451776-70500352 | 0.89 | 0.79 | 0.66 |

### Sentences of main.tex that change (Results "Chromatin marks and the consensus predict contacts and insulation more accurately than DNA sequence", Fig. 1 strip and panels a-d, Supplementary Table S11)

Read this first. The paper rows of this comparison (`akita_native_honest`, `akita_native_refres`) are models trained for this one comparison on Akita's own normalised targets on Akita's 2048 bp grid. The release checkpoint predicts the 2000 bp grid of the `proper` O/E maps, is resampled to Akita's bin centres, and is scored against Akita's targets. The control rows show what this costs: the dedicated 2 kb models of the paper family (all12honest_drop, all12refin_drop; same recipe as the release, trained on `proper` maps) go through the identical pipeline and give 0.603 (ChIP only) and 0.670 (with consensus) median Pearson, against 0.638 and 0.746 for the Akita-space models and 0.595 and 0.676 for the release checkpoint. In its own target space the release checkpoint equals the dedicated models (GM12878, 413 regions, 2 kb: 0.784 against 0.790 with the consensus, 0.659 against 0.669 without; replicate-ceiling table). The loss against the paper rows is therefore a property of the target space and grid of this comparison, not of the release checkpoint. The reference of the release rows is the leave-one-out consensus of the 11 other training cells, that of the paper row the consensus of four cells (the baseline "consensus of 4 measured cells" is the same in both). A fair single-model statement needs either the paper rows kept for this one figure (they are dedicated models and cannot be the released checkpoint) or the release rows below with this caveat.

1. "From the marks alone, without any Hi-C input, the model was more accurate than Akita in two of three regions (median Pearson correlation 0.64 against 0.55; Fig. 1a)" becomes, for the release checkpoint (13 marks and methylation of GM12878, reference plane 0): "more accurate than Akita in 216 of 413 regions (median Pearson correlation 0.60 against 0.55)"; paired difference +0.006 [-0.006, 0.017], SCC +0.056 [0.038, 0.072] (286 of 413 regions). Without Meth: 0.578. The statement "two of three" does not hold; the dedicated control gives 231 of 413.
2. "The largest gain was in insulation, where the predicted profile was closer to the measured one in 96% of regions" becomes "94%" (insulation correlation at 10 kb, 389 of 413 regions; median difference +0.264 [0.252, 0.279]; paper 397 of 413, +0.272). Other measures (difference to Akita, ChIP only, regions ahead): insulation 51 kb +0.098 (306), 250 kb +0.019 (237), boundary F1 +0.067 (228), stripe profile +0.044 (251), compartment -0.001 (204).
3. "Loop recall rose from 0.31 to 0.38 (Fig. 1b,c)" becomes "from 0.31 to 0.43" (pooled loop recall 0.432 [0.399, 0.466] against 0.314 [0.280, 0.346]; APA ratio 0.62 against 0.26; pooled boundary F1 0.504 against 0.436; hicFindTADs boundary recall 0.550 against 0.494).
4. "With the consensus of four other cell types as base, the model was more accurate than Akita in 401 of 413 regions (median 0.75), and loop recall rose to 0.57 (Supplementary Table S11)" becomes, with the leave-one-out consensus of 11 other cells as input: "in 308 of 413 regions (median 0.68), loop recall 0.53" (Pearson +0.067 [0.050, 0.080], SCC +0.118 [0.106, 0.129] in 376 regions, insulation 10 kb +0.302 in 399 regions, APA ratio 0.94 against 0.26; the dedicated control with the same input: 298 of 413 regions, median 0.67, loop recall 0.56).
5. "The consensus of four cells alone recovered more loops than our model without it, and the measured map of HFF alone recovered more loops than Akita." stays true (consensus of four cells 0.514, HFF 0.418; release without consensus 0.432; Akita 0.314). "The model adds to the consensus where the target cell differs from it. It was ahead of both Akita and the consensus in 97% of the regions." becomes "in 61% of the regions" (share of regions with the release model above Akita and above the consensus of four cells: 0.613; paper model 0.966); median gain over the consensus of four cells +0.033 (paper model +0.098).
6. "In a region at the 90th percentile of this gain, the prediction reproduced where the measured map differs from the consensus, with a correlation of 0.69 between both differences (Fig. 1d)" becomes 0.65 for the release model (region chr5:76,333,056-77,381,632, after averaging to 80 bins of 10 kb; the paper's region chr20:47,613,952-48,662,528 is a different region because the ranking of the regions changes). Fig. 1d is redrawn from `results/one_model/remaining/akita/akita413_gain_over_consensus.csv` and the saved predictions `runs/akita_one_model/all12one_fullrd_cons.npz`.
7. Panels b and c (medians over regions of nine measures, pooled F1 / hFT / loop / APA with bootstrap intervals): `akita413_per_region_summary.csv` and `akita413_pooled.csv` in `results/one_model/remaining/akita/` have the same columns as `results/structure_vs_akita/`; the rows `all12one_fullrd_chip` and `all12one_fullrd_cons` replace `akita_native_honest` and `akita_native_refres`.

## 6. Prediction time and memory

Chromosome 2 of GM12878 (242 Mb) into one cool file, one job at a time, wall time from process start to exit (model and inputs loading and cool writing included), peak GPU memory (nvidia-smi over the process tree) and peak resident memory (psutil), "default" batch 8, "lowest" batch 1, test-time reverse-complement averaging, reference input; CPU rows on 8 threads, batch 1. Script `scripts/one_model_remaining_timing.py`, `results/one_model/remaining/timing/runs.csv`. **Hardware.** The paper rows (registry section 82, columns "paper, section 82") were measured on the Ryzen 9 7950X workstation with an RTX 4090 (CUDA device 1); the local GPUs were busy during this work, so both models were measured in one session on the second computer (AMD Ryzen Threadripper 7970X, one RTX 4090, host PC3301501): columns "paper model, this host" and "release, this host". Comparisons between models use the two columns of this host. The release rows read the 13 marks and methylation of GM12878 (14 tracks) and the cohesin fraction, the paper model the 13 marks.

| grid | device | setting | paper, section 82: s / GPU MiB / RAM MiB | paper model, this host: s / GPU MiB / RAM MiB | release, this host: s / GPU MiB / RAM MiB | time release / paper model | cool (MB) |
|---|---|---|---|---|---|---|---|
| 25 kb | GPU | default | 11.7 / 836 / 1164 | 8.9 / 836 / 1101 | 9.5 / 846 / 1138 | 1.07 | 3.0 |
| 25 kb | GPU | lowest | 11.3 / 722 / 1146 | 9.2 / 722 / 1046 | 10.3 / 730 / 1029 | 1.12 | 3.0 |
| 10 kb | GPU | default | 21.4 / 1382 / 1643 | 19.5 / 1382 / 1528 | 21.5 / 1356 / 1386 | 1.10 | 16.0 |
| 10 kb | GPU | lowest | 22.3 / 724 / 1361 | 20.3 / 724 / 1296 | 22.0 / 732 / 1535 | 1.08 | 16.0 |
| 5 kb | GPU | default | 45.4 / 3226 / 3026 | 39.6 / 3226 / 2967 | 43.5 / 3410 / 2801 | 1.10 | 62.4 |
| 5 kb | GPU | lowest | 47.5 / 948 / 2972 | 40.1 / 948 / 2912 | 43.5 / 986 / 2932 | 1.08 | 62.4 |
| 2 kb | GPU | default | 118.3 / 4976 / 6616 | 95.1 / 4976 / 6547 | 105.1 / 4902 / 6588 | 1.11 | 207.5 |
| 2 kb | GPU | lowest | 113.3 / 1190 / 6556 | 96.2 / 1190 / 6496 | 105.3 / 1262 / 6519 | 1.09 | 207.5 |
| 25 kb | CPU | lowest | 20.7 / 0 / 816 | 18.1 / 0 / 765 | 19.8 / 0 / 772 | 1.09 | 3.0 |
| 10 kb | CPU | lowest | 49.6 / 0 / 1210 | 44.9 / 0 / 1086 | 47.9 / 0 / 1121 | 1.07 | 16.0 |

### Sentences of main.tex that change (Results "Whole chromosomes are predicted within minutes on standard hardware", Methods "Prediction time and memory", Supplementary Table S17)

The release checkpoint was measured on the second computer (Threadripper 7970X, RTX 4090), not on the workstation of the paper table, because the local GPUs were busy. On that host the dedicated paper model needs 8.9, 19.5, 39.6 and 95.1 s at 25, 10, 5 and 2 kb (default batch; the workstation needed 11.7, 21.4, 45.4 and 118.3 s), and the release checkpoint needs 7 to 12 % longer than the dedicated model at every size (9.5, 21.5, 43.5, 105.1 s). Applied to the workstation values the release times would be about 12.5, 23.5, 50 and 130 s. These are estimates, not measurements; `python scripts/one_model_remaining_timing.py --models release` on the workstation (GPU idle, `HM_ROOT` unset) gives the table rows directly.

1. "On one graphics card, our model needed 11 s at 25 kb, 22 s at 10 kb and 2 min at 2 kb" holds in order of magnitude; with the measured ratio the numbers become about 12 s, 24 s and 2 min 10 s on the workstation (this host: 9.5 s, 21.5 s, 1 min 45 s). Update after the remeasurement.
2. "Our model used less than 1 GB of graphics memory from 25 to 5 kb and 1.2 GB at 2 kb" holds for the release checkpoint at the lowest batch size: 730, 732 and 986 MiB at 25, 10 and 5 kb and 1262 MiB (1.2 GB) at 2 kb (paper model: 722, 724, 948, 1190 MiB). At the default batch size 8: 846, 1356, 3410, 4902 MiB (paper model 836, 1382, 3226, 4976).
3. "Without a graphics card, on eight processor threads, the 25 and 10 kb models predicted the chromosome in less than one minute with 1.2 GB of main memory" holds: 19.8 s and 47.9 s with 772 and 1121 MiB (paper model on this host 18.1 s, 44.9 s, 765 and 1086 MiB; the paper's 1.2 GB is the 10 kb workstation value 1210 MiB, the release value would be "1.1 GB").
4. "Epiphany at 10 kb and Akita at 2 kb needed a similar time, and C.Origami about one hour. ... Epiphany needed 2.6 GB, C.Origami 3.5 GB and Akita 6.4 GB" are competitor numbers and stay unchanged.
5. Methods: the sentence "Our reference-input models averaged both strand orientations with a stride of half a window" now concerns one network at four bin sizes (`--bin-size` is not needed, the four views are used). The release rows read 14 tracks (13 marks and methylation) and the cohesin fraction; the cool files are the same size (2 kb 207.5 MB against 208.9 MB).
6. Discussion: "With less than 1 GB of graphics memory from 25 to 5 kb" stays (986 MiB at 5 kb is the tightest).


## Problems met

- The mouse predictions were produced twice: the first pull script excluded all sub-directories, copied nothing and deleted the
  remote copies. The remote job was rerun in full (1 h) and the pull script now verifies the file counts before deleting anything
  (`pull` logic described in registry section 98). No result was affected.
- The wall clock of the shell jumped by about seven hours once (02:40 to 08:45) while background jobs were waiting; no job was affected.
- The Akita comparison cannot use the paper protocol one to one (see its section): the paper rows are dedicated models trained in Akita's
  target space; a control run of the dedicated proper-target models separates the effect of the space from that of the checkpoint.
- The paper sentence "28% of the loops and 10% of the stripes" (marks only, AML) could not be traced to a result file.
