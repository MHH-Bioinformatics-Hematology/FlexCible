# Degron perturbation with zero-signal semantics (registry section 21 redone)

Section 21 set one protein to its absent value, the encoding of a mark that was not measured. Here the raw
signal of the protein is set to zero (the state of a region without binding) with `flexcible predict
--zero-region`, on every chromosome of the 413 Akita test regions (chr1-chr21), and the prediction change is
scored against the measured degron change with the section-21 scorer.

## Provenance

- Models: `all12honest_drop_10kb`, `all12honest_drop_25kb` (ChIP only, track dropout, no consensus) and
  `all12refin_drop_10kb` (with consensus; CTCF zero only). Configs and checkpoints in
  `/mnt/processed/wolffjoa/hic-multiome/runs/multicell_reference/<model>.yaml` and `<model>/checkpoint-best`. No retraining.
- Sample hct116 (training cell, ChIP `hg38_hct116`, SMC3 absent for the cell); the 413 test regions are held out of training.
- Measured maps and O/E conversions: section 21 (`runs/perturb_validate/oe`, `oe_proper.py`), untreated vs auxin for
  RAD21-mAC (Rao 2017) and CTCF-AID (ENCODE intact Hi-C). Loop calls (`runs/benchmark/calls`), boundaries, regions
  and metrics are imported from `scripts/perturb_validate.py`. Predicted change = perturbed minus unperturbed log O/E.
- Predictions (GPU 1, batch size 4, 8 threads, `--tta`), one command per model and perturbation, written by
  `scripts/perturb_degron_zero.py --stage predict`; example (10 kb, CTCF zero + site removal):

      CUDA_VISIBLE_DEVICES=1 python -m flexcible.cli predict --config runs/multicell_reference/all12honest_drop_10kb.yaml \
        --checkpoint runs/multicell_reference/all12honest_drop_10kb/checkpoint-best --sample hct116 \
        --chroms chr1,chr2,...,chr21 --tta --batch-size 4 \
        --zero-region CTCF:chr1:0-<length chr1> ... --zero-region CTCF:chr21:0-<length chr21> \
        --zero-region motif:MA0139.2:chr1:0-<length chr1> ... --out cool/all12honest_drop_10kb/zero_CTCF_motif.cool

  Whole-chromosome regions are accepted; several `--zero-region` flags per call work. Perturbations: `full`,
  `zero_CTCF`, `zero_RAD21`, `zero_H3K36me3` (control), `zero_CTCF_motif` (CTCF zero plus motif MA0139.2 removed:
  site-deletion semantics, whereas in the degron the motif stays). Full command lines are in the first line of
  each log, `/mnt/processed/wolffjoa/hic-multiome/perturb/degron_zero/logs/<model>_<pert>.log`.
- Scoring: `python scripts/perturb_degron_zero.py --stage score compare` (CPU).
- Cool files (predicted log O/E on the HCT116 grid): `/mnt/processed/wolffjoa/hic-multiome/perturb/degron_zero/cool/<model>/<pert>.cool`.
- Figure: `~/src/manuscripts/hic-multiome-natcomms/figures/scripts/fig_perturb_degron.py` (manuscripts repo, uncommitted).
- Code fix made on the way: `flexcible predict` split every `--zero-region` into three fields and failed on
  `motif:ID:chrom:start-end`; the parser now matches the one of `flexcible.variants` (`flexcible/predict.py`).
- Check of the pipeline: the unperturbed prediction of this run against the section-21 `full.npz` of the same model
  (per-region predictions with Hann merging, first 60 regions): median Pearson 0.986, so the baseline differs slightly
  (APA of the unperturbed prediction 1.15 here, 1.13 there) because chromosomes are tiled whole here.
- HeLa CTCF-AID (Wutz 2017): not run. No HeLa ChIP tracks exist on the share or in `~/data/hic-multiome`
  (only `perturb/hela_ctcf_aid_wutz2017` Hi-C; `validation_new/logs/chip_hela.log` shows an unfinished download of 0 GB).
  Nothing was downloaded.

## Result table

All rows are the 413 test regions (loops: 534-549 called in the untreated map inside them; boundaries at 10 kb only).
Rows with encoding "absent track" are section-21 numbers recomputed by the same scorer, for the section-21 models
(`all12honest_*`, no track dropout, as in the registry table) and for the dropout models used here.

| experiment | model | encoding | input change | change Pearson median (IQR) | change SCC | loops weaker, predicted (measured) | loop change Spearman | APA unperturbed / perturbed (measured untreated / treated) | boundary insulation change, predicted (measured) |
|---|---|---|---|---|---|---|---|---|---|
| CTCF degron | all12honest_10kb | absent track | CTCF absent | 0.093 (-0.055 to 0.291) | 0.191 | 0.78 (0.99) | 0.05 | 1.19 / 1.09 (1.70 / 1.35) | 0.016 (0.117) |
| CTCF degron | all12honest_10kb | absent track | H3K36me3 absent (control) | 0.117 (-0.051 to 0.325) | 0.033 | 0.36 (0.99) | 0.15 | 1.19 / 1.19 (1.70 / 1.35) | 0.000 (0.117) |
| CTCF degron | all12honest_10kb | absent track | RAD21 absent | 0.057 (-0.101 to 0.271) | 0.005 | 0.24 (0.99) | 0.24 | 1.19 / 1.22 (1.70 / 1.35) | -0.006 (0.117) |
| CTCF degron | all12honest_drop_10kb | absent track | CTCF absent | 0.347 (0.168 to 0.475) | 0.309 | 0.89 (0.99) | 0.19 | 1.13 / 1.08 (1.70 / 1.35) | 0.013 (0.117) |
| CTCF degron | all12honest_drop_10kb | zero signal | CTCF zero | 0.335 (0.171 to 0.480) | 0.316 | 0.91 (0.99) | 0.20 | 1.15 / 1.08 (1.70 / 1.35) | 0.016 (0.117) |
| CTCF degron | all12honest_drop_10kb | zero signal | CTCF zero + MA0139.2 sites removed | 0.360 (0.179 to 0.502) | 0.333 | 0.91 (0.99) | 0.17 | 1.15 / 1.07 (1.70 / 1.35) | 0.020 (0.117) |
| CTCF degron | all12honest_drop_10kb | absent track | H3K36me3 absent (control) | 0.002 (-0.215 to 0.183) | 0.030 | 0.57 (0.99) | -0.08 | 1.13 / 1.12 (1.70 / 1.35) | -0.001 (0.117) |
| CTCF degron | all12honest_drop_10kb | zero signal | H3K36me3 zero (control) | -0.011 (-0.203 to 0.190) | 0.031 | 0.57 (0.99) | -0.09 | 1.15 / 1.14 (1.70 / 1.35) | -0.002 (0.117) |
| CTCF degron | all12honest_drop_10kb | absent track | RAD21 absent | -0.058 (-0.223 to 0.171) | -0.049 | 0.41 (0.99) | 0.16 | 1.13 / 1.13 (1.70 / 1.35) | -0.007 (0.117) |
| CTCF degron | all12honest_drop_10kb | zero signal | RAD21 zero | -0.062 (-0.236 to 0.172) | -0.055 | 0.40 (0.99) | 0.16 | 1.15 / 1.15 (1.70 / 1.35) | -0.007 (0.117) |
| CTCF degron | all12refin_drop_10kb | zero signal | CTCF zero | -0.130 (-0.360 to 0.070) | -0.071 | 0.64 (0.99) | 0.03 | 1.38 / 1.36 (1.70 / 1.35) | -0.002 (0.117) |
| CTCF degron | all12honest_25kb | absent track | CTCF absent | 0.271 (0.003 to 0.469) | 0.315 | 0.75 (0.96) | 0.30 | 1.14 / 1.07 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_25kb | absent track | H3K36me3 absent (control) | 0.219 (-0.065 to 0.450) | 0.047 | 0.13 (0.96) | 0.10 | 1.14 / 1.19 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_25kb | absent track | RAD21 absent | -0.063 (-0.285 to 0.229) | -0.107 | 0.13 (0.96) | 0.21 | 1.14 / 1.18 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_drop_25kb | absent track | CTCF absent | 0.401 (0.159 to 0.589) | 0.387 | 0.77 (0.96) | 0.29 | 1.12 / 1.08 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_drop_25kb | zero signal | CTCF zero | 0.404 (0.190 to 0.596) | 0.392 | 0.78 (0.96) | 0.29 | 1.14 / 1.09 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_drop_25kb | zero signal | CTCF zero + MA0139.2 sites removed | 0.434 (0.213 to 0.637) | 0.432 | 0.81 (0.96) | 0.32 | 1.14 / 1.09 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_drop_25kb | absent track | H3K36me3 absent (control) | 0.001 (-0.240 to 0.270) | -0.005 | 0.57 (0.96) | -0.02 | 1.12 / 1.11 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_drop_25kb | zero signal | H3K36me3 zero (control) | -0.020 (-0.268 to 0.260) | -0.029 | 0.57 (0.96) | -0.04 | 1.14 / 1.13 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_drop_25kb | absent track | RAD21 absent | 0.133 (-0.113 to 0.378) | 0.116 | 0.53 (0.96) | 0.25 | 1.12 / 1.12 (1.43 / 1.26) |  () |
| CTCF degron | all12honest_drop_25kb | zero signal | RAD21 zero | 0.150 (-0.114 to 0.396) | 0.116 | 0.53 (0.96) | 0.26 | 1.14 / 1.14 (1.43 / 1.26) |  () |
| RAD21 degron | all12honest_10kb | absent track | CTCF absent | 0.151 (-0.102 to 0.403) | 0.195 | 0.78 (0.99) | 0.33 | 1.19 / 1.09 (1.45 / 1.05) | 0.016 (0.063) |
| RAD21 degron | all12honest_10kb | absent track | H3K36me3 absent (control) | 0.116 (-0.137 to 0.357) | 0.033 | 0.36 (0.99) | -0.12 | 1.19 / 1.19 (1.45 / 1.05) | 0.000 (0.063) |
| RAD21 degron | all12honest_10kb | absent track | RAD21 absent | 0.041 (-0.184 to 0.302) | -0.031 | 0.24 (0.99) | -0.06 | 1.19 / 1.22 (1.45 / 1.05) | -0.006 (0.063) |
| RAD21 degron | all12honest_drop_10kb | absent track | CTCF absent | 0.418 (0.174 to 0.565) | 0.364 | 0.89 (0.99) | 0.37 | 1.13 / 1.08 (1.45 / 1.05) | 0.013 (0.063) |
| RAD21 degron | all12honest_drop_10kb | zero signal | CTCF zero | 0.403 (0.171 to 0.565) | 0.370 | 0.90 (0.99) | 0.39 | 1.14 / 1.09 (1.45 / 1.05) | 0.016 (0.063) |
| RAD21 degron | all12honest_drop_10kb | zero signal | CTCF zero + MA0139.2 sites removed | 0.431 (0.179 to 0.609) | 0.408 | 0.90 (0.99) | 0.35 | 1.14 / 1.07 (1.45 / 1.05) | 0.020 (0.063) |
| RAD21 degron | all12honest_drop_10kb | absent track | H3K36me3 absent (control) | 0.042 (-0.222 to 0.289) | 0.080 | 0.56 (0.99) | 0.13 | 1.13 / 1.12 (1.45 / 1.05) | -0.001 (0.063) |
| RAD21 degron | all12honest_drop_10kb | zero signal | H3K36me3 zero (control) | 0.029 (-0.206 to 0.302) | 0.066 | 0.57 (0.99) | 0.11 | 1.14 / 1.14 (1.45 / 1.05) | -0.002 (0.063) |
| RAD21 degron | all12honest_drop_10kb | absent track | RAD21 absent | -0.003 (-0.263 to 0.275) | -0.046 | 0.41 (0.99) | 0.02 | 1.13 / 1.13 (1.45 / 1.05) | -0.007 (0.063) |
| RAD21 degron | all12honest_drop_10kb | zero signal | RAD21 zero | -0.009 (-0.259 to 0.283) | -0.056 | 0.40 (0.99) | 0.02 | 1.14 / 1.15 (1.45 / 1.05) | -0.007 (0.063) |
| RAD21 degron | all12refin_drop_10kb | zero signal | CTCF zero | -0.146 (-0.381 to 0.101) | -0.075 | 0.64 (0.99) | -0.03 | 1.38 / 1.36 (1.45 / 1.05) | -0.002 (0.063) |
| RAD21 degron | all12honest_25kb | absent track | CTCF absent | 0.207 (-0.104 to 0.523) | 0.317 | 0.75 (0.99) | 0.38 | 1.14 / 1.07 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_25kb | absent track | H3K36me3 absent (control) | 0.076 (-0.265 to 0.427) | -0.038 | 0.13 (0.99) | -0.31 | 1.14 / 1.19 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_25kb | absent track | RAD21 absent | -0.067 (-0.318 to 0.249) | -0.133 | 0.13 (0.99) | -0.15 | 1.14 / 1.18 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_drop_25kb | absent track | CTCF absent | 0.436 (0.171 to 0.635) | 0.411 | 0.77 (0.99) | 0.49 | 1.12 / 1.08 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_drop_25kb | zero signal | CTCF zero | 0.442 (0.179 to 0.662) | 0.424 | 0.78 (0.99) | 0.47 | 1.14 / 1.09 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_drop_25kb | zero signal | CTCF zero + MA0139.2 sites removed | 0.483 (0.210 to 0.676) | 0.469 | 0.81 (0.99) | 0.47 | 1.14 / 1.09 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_drop_25kb | absent track | H3K36me3 absent (control) | 0.091 (-0.156 to 0.349) | 0.086 | 0.57 (0.99) | 0.14 | 1.12 / 1.11 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_drop_25kb | zero signal | H3K36me3 zero (control) | 0.089 (-0.187 to 0.333) | 0.073 | 0.57 (0.99) | 0.14 | 1.14 / 1.13 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_drop_25kb | absent track | RAD21 absent | 0.169 (-0.159 to 0.437) | 0.131 | 0.53 (0.99) | 0.15 | 1.12 / 1.12 (1.33 / 1.02) |  () |
| RAD21 degron | all12honest_drop_25kb | zero signal | RAD21 zero | 0.187 (-0.155 to 0.419) | 0.118 | 0.53 (0.99) | 0.14 | 1.14 / 1.14 (1.33 / 1.02) |  () |

Files: `perturb_summary.csv` (full metric grid of the zero-signal runs), `comparison.csv`/`comparison.md` (table above),
`change_per_region.csv.gz`, `change_per_distance.csv`, `loops_per_anchor.csv.gz`, `boundaries_per_bin.csv.gz`.

## Reading

- CTCF zero reproduces the CTCF degron better than any section-21 row of a model without track dropout: change Pearson
  median 0.34 (10 kb) and 0.40 (25 kb), 91 and 78% of the loops weaken (99 and 96% measured), predicted APA 1.15 to 1.08
  (measured 1.70 to 1.35), and the loop changes rank-correlate with the measured ones (Spearman 0.20 and 0.29). The same
  perturbation correlates equally with the RAD21 degron change (0.40, 0.44) because both degrons remove loops.
  The control (H3K36me3 zero) gives Pearson -0.01 to 0.09 and 57% weaker loops, and RAD21 zero -0.06 to 0.19, so the
  correlation is specific to CTCF (the control correlated at 0.12 to 0.22 with the no-dropout models in section 21).
- The gain over section 21 comes from the track-dropout training, not from the zero encoding: the dropout models with
  the absent value give the same numbers (CTCF absent 0.35 and 0.40, loops weaker 0.89 and 0.77). Zero and absent
  differ by at most 0.02 in median Pearson; the change is the same within the spread of the regions.
- Removing the motif sites in addition to the signal raises the correlation (0.36 and 0.43 against CTCF-AID) and the
  predicted change magnitude (mean absolute change 0.10 against 0.075 at 10 kb), as expected for a deletion; the degron
  semantics (motif stays) is the one that matches the experiment, and the site deletion is the stronger perturbation.
- RAD21 zero does not reproduce the RAD21 degron: Pearson -0.01 (10 kb) and 0.19 (25 kb), 40 and 53% of loops weaken
  (99% measured), median loop change +0.013 (measured -0.75), APA unchanged (1.14 to 1.15), insulation change -0.007
  (measured +0.063). The model ChIP-only has no cohesin subunit left to read (HCT116 has no SMC3 ChIP, and RAD21 is the
  only cohesin input), and the zero state of RAD21 behaves like a low-signal state that cohesin-poor training cells
  also show; the model does not link it to loop loss. Predicted amplitudes of every perturbation are
  2 to 6 times smaller than the measured change (mean absolute 0.03-0.12 against 0.16-0.22).
- Consensus: with the leave-one-out consensus as input (`all12refin_drop_10kb`) CTCF zero changes the prediction by
  0.017 on average against 0.075 without it, 64% of the loops weaken (91%), and the change correlation turns negative
  (-0.13 against CTCF-AID, -0.15 against RAD21-mAC). The consensus damps the perturbation, as expected.
