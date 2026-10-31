# Release candidate all12one_fullrd: cohesin perturbations and methylation proxy

Registry section 95, 2026-10-31. The evaluations of section 55 (RAD21 degron, washout time course, CTCF degron, U-937 STAG2 knockout) and of section 78 (methylation as a proxy for missing tracks) were repeated on the per-resolution views `all12one_fullrd_{10,25}kb` of the release candidate (one checkpoint for all bin sizes, 13 marks plus Meth, cohesin-fraction row, cohesin-loss augmentation, reference dropout 0.3), with the commands, evaluation sets and settings of those sections. Columns of the perturbation tables: deg = all12deg_refres_drop (the base model of section 55: RAD21 degron pair in training, no fraction row), cohfix = all12cohfix_refres_drop (fraction row and augmentation, dedicated per-resolution model of section 55), release = all12one_fullrd. All tables below are generated from the committed result files by `scripts/perturb_meth_report.py` (also written to `perturb_meth_tables.md`).

## Summary

RAD21 degron (HCT116 even chromosomes, never trained under auxin; 222 regions, 255 loops at 10 kb, 260 at 25 kb). With the measured auxin ChIP and the fraction row at 0.26 the release model reaches a change-map Pearson (median over regions) of 0.854 at 10 kb and 0.901 at 25 kb, against 0.839 / 0.888 for cohfix and 0.803 / 0.842 for deg. The predicted median loop change is -0.667 / -0.428 log O/E (measured -0.726 / -0.477; cohfix -0.652 / -0.341; deg -0.488 / -0.201), and 100% / 99.2% of the loops come out weaker (measured 98.8% / 98.9%). With RAD21 declared absent and the fraction at 0 the Pearson is 0.794 / 0.857 (cohfix 0.784 / 0.852), the loop change -1.138 / -0.533 (cohfix -1.135 / -0.489), and 100% / 99.6% of the loops weaker; at fraction 0.5 the loop change is -0.629 / -0.298 and the Pearson 0.801 / 0.863. The graded response of the fraction row is retained in the one model. RAD21 declared absent without the fraction row (a missing ChIP, fraction 1) stays without effect, as designed: 9.4% / 10.0% of the loops weaker (deg 20.8% / 33.1%, cohfix 16.9% / 26.2%), change-map Pearson -0.283 / -0.470 (deg -0.249 / 0.245, cohfix -0.299 / -0.066). The mean absolute predicted change relative to the measured one is 0.83 / 0.84 at the measured state, 1.21 / 1.05 at fraction 0 (deg 0.64 / 0.62 at the measured state).

Washout (all loops of the untreated map, 536 at 10 kb, 549 at 25 kb). The predicted median loop change at the 6 h auxin RAD21 track is -0.682 / -0.419 (measured -0.754 / -0.481; cohfix -0.693 / -0.346; deg -0.381 / -0.105); on the even chromosomes -0.675 / -0.437 (measured -0.726 / -0.477). The predicted recovery is monotone in the RAD21 interpolation f (10 kb: 0.28, 0.64, 0.89 at f = 0.25, 0.5, 0.75; 25 kb: 0.24, 0.57, 0.83; cohfix 0.32, 0.65, 0.86 and 0.29, 0.61, 0.83). The f that matches the measured recovery rises with the washout time (10 kb: 0.16, 0.39, 0.51, 1.00 at 20, 40, 60, 180 min; 25 kb: 0.16, 0.38, 0.53, 1.00; cohfix 0.14, 0.37, 0.51, 1.00 at 10 kb), so the ordering of the time points is reproduced.

CTCF degron (all chromosomes, never trained; 411 / 410 regions). Change-map Pearson of the release model 0.444 / 0.520 (cohfix 0.403 / 0.402; deg 0.396 / 0.517), above the floor of 0.40 of section 55 at both sizes; the 25 kb value that cohfix lost against deg (0.115) is recovered. The loop change is -0.111 / -0.072 (measured -0.489 / -0.255; cohfix -0.030 / -0.017) with 98.7% / 96.0% of the loops weaker (measured 99.1% / 96.2%; cohfix 83.3% / 66.5%).

U-937 STAG2 knockout. The knockout stays unexplained. For the wild-type ChIP with the fraction at 0.5 the predicted mean loop change is -0.249 (10 kb) and -0.287 (25 kb) log2, against +0.080 and -0.033 measured (cohfix -0.231 / -0.315); at 0.75 it is -0.084 / -0.102. The change-map Pearson is 0.190 / 0.232 (cohfix 0.200 / 0.261; ceiling between independent clone pairs 0.254, 0.339, 0.122 at 10 kb). RAD21 and SMC3 halved or kept at CTCF peaks only give no response (change-map Pearson 0.03 to 0.05 at both sizes; cohfix 0.07 to 0.27, deg 0.08 to 0.18; loop change +0.011 to -0.053 against -0.105 to -0.020 measured in short and long loops). Intact predictions of the AML patient lines are unchanged by the one model (MOLM-13 median Pearson 0.887 / 0.916 against 0.882 / 0.912 for cohfix; U-937 wild type 0.860 / 0.891 against 0.868 / 0.891; U-937 knockout clone 1 from the wild-type input 0.769 / 0.830 against 0.781 / 0.832).

Methylation proxy (413 Akita test regions; the group sets and the replacement fraction RF of section 78; 10 training cells with WGBS or lifted RRBS and the held-out K562 and HMEC (deep map); the bootstrap and the noise threshold of section 78: 0.0062 at 10 kb, 0.0072 at 25 kb for the plane at 0, 0.0062 for the consensus plane). With the consensus plane the one model behaves like the dedicated consensus model of section 78. Methylation added to the full ChIP set gains 0.001 to 0.003 (10 kb, 25 kb, training WGBS and held-out cells), removal of one group costs at most 0.025 (active; cohesin 0.009) on the WGBS training cells at 10 kb and at most 0.009 on the held-out cells. On the seven WGBS training cells Meth returns 0.84 (0.47 to 1.22) of the cohesin loss at 10 kb (dedicated 0.93, 0.72 to 1.13) and 0.89 (0.54 to 1.75) at 25 kb (dedicated 0.89, 0.76 to 1.05), 0.70 and 0.79 of the cum2 loss, and 0.22 (0.11 to 0.42) / 0.22 (0.07 to 0.48) of the CTCF-only loss and 0.17 (0.05 to 0.45 at 10 kb, 0.03 to 0.45 at 25 kb) of the sequence-only loss (dedicated at 10 kb 0.39 and 0.32); the losses of the release model without ChIP are larger (0.068 and 0.073 against 0.044 and 0.045) while the gains are the same (0.013 to 0.015). On the held-out cells no single group costs more than 0.009 with the consensus plane, so RF is defined for minus active at 10 kb (0.21, -0.03 to 0.51), CTCF only (0.25, 0.04 to 0.43, at 10 kb; 0.54, 0.27 to 0.74, at 25 kb) and sequence only (0.47, 0.30 to 0.58; 0.62, 0.40 to 0.77). With the reference plane at 0 (the ChIP-only use of the checkpoint) the release model is close to the dedicated ChIP-only (honest) model of section 78 on the held-out cells: CTCF removal costs 0.043 / 0.056 (dedicated 0.037 / 0.055) and Meth returns 0.10 (-0.16 to 0.25) / 0.18 (-0.08 to 0.31) of it (dedicated 0.02 / 0.17); CTCF only 0.35 (0.15 to 0.46) / 0.37 (0.25 to 0.57) (dedicated 0.36 / 0.51); sequence only 0.28 (0.18 to 0.39) / 0.37 (0.27 to 0.44) (dedicated 0.33 / 0.33), with gains of 0.057 / 0.082 Pearson (dedicated 0.074 / 0.066). The cohesin loss is replaced almost fully (0.89, 0.25 to 1.71, at 10 kb) from a small loss (0.013). On the three RRBS training cells (HCT116, MCF-7, HFF; lifted hg19 CpGs) Meth returns nothing measurable (sequence-only gain 0.005 / 0.006 with the plane at 0, 0.001 / -0.000 with the consensus plane). Structure measures (insulation, loop enrichment, compartment, eigenvector, boundary F1) follow the Pearson pattern; gains on the held-out cells (consensus plane, 10 kb) are 0.004 to 0.013 for the sequence-only set and 0.001 to 0.007 for CTCF only, and the boundary F1 is 0 on the held-out cells for every set.

## Deviations from the section 55 and 78 protocols

* Predictions of the perturbation, washout and AML steps ran on one GPU (cuda:1) instead of the CPU (`PERT_GPU`, `AML_EVAL_GPU`); the section 55 predictions of the dedicated models ran on the CPU and on the remote machine.
* The release model reads the leave-one-out consensus of HCT116 as an input plane (`reference: input`); the dedicated models of section 55 read it as a residual base. The reference plane was present in every perturbation call.
* The HCT116 methylation track (lifted RRBS, `meth/hct116`) was given to every call of the release model, with the Meth track kept in every keep set; the auxin sample of the training used the untreated methylation as well. U-937 and MOLM-13 have no methylation (Meth declared absent).
* Fractions as in section 55: 1 for the untreated ChIP, 0.26 for the measured 6 h auxin ChIP (`--cohesin-fraction 0.26`), `_f0`, `_f0.5` suffixes of the keep sets, washout fraction 0.26 + f (1 - 0.26), U-937 variants as in `aml_eval.py`.
* Not rerun: `cohesin_fix_diagnose.py` (the cached windows of the shared multi-resolution cache carry 23 condition rows and the script's row selection raises an IndexError; the per-region change statistics of `perturb_validate.py` replace it), the hard-region summary, and the AML-donor part of section 78 (its models are fold-wise trained on AML donors and have no counterpart in the release checkpoint).
* Single run of one checkpoint; the tables of earlier models are the committed ones of sections 55 and 78 (their result files were not rewritten: `--tag all12one_fullrd`).

## Files

Scripts: `scripts/perturb_validate.py`, `scripts/washout.py`, `scripts/aml_eval.py` (view models, `--models`, `--families`, `--tag`), `scripts/meth_proxy_release.py` (stages predict, summary), `scripts/perturb_meth_report.py`, test `tests/test_release_perturbation.py`. Results: `results/perturb_validate/all12one_fullrd_*`, `results/washout/all12one_fullrd_washout_summary.csv`, `results/aml/perturb_summary_all12one_fullrd.csv` (and added rows in `results/aml/summary.csv`), `results/meth_proxy/all12one_fullrd_replacement_{pooled,by_cell}.csv`, `results/one_model/perturb_meth_tables.md`. Predictions: `/mnt/processed/wolffjoa/hic-multiome/runs/{perturb_validate,washout,meth_proxy}` and `aml/pred`.

## Tables

### RAD21 degron (HCT116, even chromosomes)

#### RAD21 degron, 10 kb

| perturbation | model | Pearson median [q25, q75] | Pearson mean | Pearson sd | regions | SCC median [q25, q75] | Spearman median | mean abs change pred / meas |
|---|---|---|---|---|---|---|---|---|
| RAD21 absent, fraction 0 | cohfix | 0.784 [0.675, 0.862] | 0.751 | 0.146 | 222 | 0.719 [0.585, 0.801] | 0.760 | 0.280 / 0.222 |
| RAD21 absent, fraction 0 | release | 0.794 [0.667, 0.857] | 0.749 | 0.147 | 222 | 0.710 [0.606, 0.799] | 0.768 | 0.268 / 0.222 |
| RAD21 absent, fraction 0.5 | cohfix | 0.792 [0.681, 0.861] | 0.755 | 0.144 | 222 | 0.722 [0.599, 0.802] | 0.772 | 0.146 / 0.222 |
| RAD21 absent, fraction 0.5 | release | 0.801 [0.696, 0.863] | 0.762 | 0.140 | 222 | 0.726 [0.611, 0.801] | 0.779 | 0.147 / 0.222 |
| measured auxin ChIP, fraction 0.26 | deg | 0.803 [0.668, 0.868] | 0.730 | 0.208 | 222 | 0.739 [0.602, 0.807] | 0.785 | 0.143 / 0.222 |
| measured auxin ChIP, fraction 0.26 | cohfix | 0.839 [0.749, 0.898] | 0.808 | 0.127 | 222 | 0.766 [0.668, 0.830] | 0.821 | 0.180 / 0.222 |
| measured auxin ChIP, fraction 0.26 | release | 0.854 [0.772, 0.907] | 0.816 | 0.124 | 222 | 0.778 [0.673, 0.836] | 0.835 | 0.184 / 0.222 |
| RAD21 absent, fraction 1 (ChIP only) | deg | -0.249 [-0.568, 0.188] | -0.172 | 0.433 | 222 | -0.311 [-0.552, 0.024] | -0.238 | 0.027 / 0.222 |
| RAD21 absent, fraction 1 (ChIP only) | cohfix | -0.299 [-0.539, 0.165] | -0.193 | 0.426 | 222 | -0.287 [-0.544, 0.093] | -0.266 | 0.020 / 0.222 |
| RAD21 absent, fraction 1 (ChIP only) | release | -0.283 [-0.469, -0.015] | -0.238 | 0.301 | 222 | -0.319 [-0.475, -0.056] | -0.261 | 0.021 / 0.222 |
| CTCF absent | deg | 0.411 [0.196, 0.587] | 0.374 | 0.284 | 222 | 0.429 [0.257, 0.558] | 0.394 | 0.029 / 0.222 |
| CTCF absent | cohfix | 0.399 [0.093, 0.587] | 0.317 | 0.356 | 222 | 0.392 [0.112, 0.576] | 0.354 | 0.025 / 0.222 |
| CTCF absent | release | 0.469 [0.246, 0.629] | 0.428 | 0.261 | 222 | 0.520 [0.364, 0.624] | 0.425 | 0.041 / 0.222 |

| perturbation | model | loops | loop change measured (median log O/E) | predicted | weaker measured | weaker predicted | Spearman of change | APA untreated | APA treated meas / pred full / pred pert | boundaries | insulation change measured | predicted | Spearman of change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RAD21 absent, fraction 0 | cohfix | 255 | -0.726 | -1.135 | 98.8% | 100.0% | 0.685 | n/a | n/a | 955 | 0.062 | 0.124 | 0.729 |
| RAD21 absent, fraction 0 | release | 255 | -0.726 | -1.138 | 98.8% | 100.0% | 0.681 | n/a | n/a | 955 | 0.062 | 0.131 | 0.720 |
| RAD21 absent, fraction 0.5 | cohfix | 255 | -0.726 | -0.611 | 98.8% | 100.0% | 0.686 | n/a | n/a | 955 | 0.062 | 0.066 | 0.725 |
| RAD21 absent, fraction 0.5 | release | 255 | -0.726 | -0.629 | 98.8% | 100.0% | 0.632 | n/a | n/a | 955 | 0.062 | 0.065 | 0.714 |
| measured auxin ChIP, fraction 0.26 | deg | 255 | -0.726 | -0.488 | 98.8% | 98.4% | 0.572 | n/a | n/a | 955 | 0.062 | 0.042 | 0.730 |
| measured auxin ChIP, fraction 0.26 | cohfix | 255 | -0.726 | -0.652 | 98.8% | 100.0% | 0.754 | n/a | n/a | 955 | 0.062 | 0.056 | 0.755 |
| measured auxin ChIP, fraction 0.26 | release | 255 | -0.726 | -0.667 | 98.8% | 100.0% | 0.778 | n/a | n/a | 955 | 0.062 | 0.069 | 0.777 |
| RAD21 absent, fraction 1 (ChIP only) | deg | 255 | -0.726 | 0.026 | 98.8% | 20.8% | -0.153 | n/a | n/a | 955 | 0.062 | -0.011 | -0.457 |
| RAD21 absent, fraction 1 (ChIP only) | cohfix | 255 | -0.726 | 0.043 | 98.8% | 16.9% | -0.166 | n/a | n/a | 955 | 0.062 | -0.005 | -0.326 |
| RAD21 absent, fraction 1 (ChIP only) | release | 255 | -0.726 | 0.037 | 98.8% | 9.4% | -0.005 | n/a | n/a | 955 | 0.062 | -0.009 | -0.453 |
| CTCF absent | deg | 255 | -0.726 | -0.052 | 98.8% | 96.9% | 0.069 | n/a | n/a | 955 | 0.062 | 0.008 | 0.496 |
| CTCF absent | cohfix | 255 | -0.726 | -0.032 | 98.8% | 86.3% | 0.069 | n/a | n/a | 955 | 0.062 | 0.008 | 0.467 |
| CTCF absent | release | 255 | -0.726 | -0.120 | 98.8% | 99.2% | 0.102 | n/a | n/a | 955 | 0.062 | 0.015 | 0.549 |

#### RAD21 degron, 25 kb

| perturbation | model | Pearson median [q25, q75] | Pearson mean | Pearson sd | regions | SCC median [q25, q75] | Spearman median | mean abs change pred / meas |
|---|---|---|---|---|---|---|---|---|
| RAD21 absent, fraction 0 | cohfix | 0.852 [0.737, 0.906] | 0.797 | 0.160 | 222 | 0.792 [0.629, 0.870] | 0.824 | 0.214 / 0.204 |
| RAD21 absent, fraction 0 | release | 0.857 [0.756, 0.911] | 0.813 | 0.140 | 222 | 0.802 [0.675, 0.869] | 0.843 | 0.215 / 0.204 |
| RAD21 absent, fraction 0.5 | cohfix | 0.846 [0.740, 0.910] | 0.799 | 0.158 | 222 | 0.790 [0.637, 0.874] | 0.827 | 0.120 / 0.204 |
| RAD21 absent, fraction 0.5 | release | 0.863 [0.769, 0.915] | 0.819 | 0.137 | 222 | 0.813 [0.689, 0.874] | 0.845 | 0.120 / 0.204 |
| measured auxin ChIP, fraction 0.26 | deg | 0.842 [0.699, 0.907] | 0.743 | 0.270 | 222 | 0.797 [0.634, 0.879] | 0.831 | 0.126 / 0.204 |
| measured auxin ChIP, fraction 0.26 | cohfix | 0.888 [0.798, 0.930] | 0.843 | 0.137 | 222 | 0.807 [0.699, 0.883] | 0.867 | 0.144 / 0.204 |
| measured auxin ChIP, fraction 0.26 | release | 0.901 [0.832, 0.938] | 0.862 | 0.122 | 222 | 0.847 [0.739, 0.903] | 0.884 | 0.171 / 0.204 |
| RAD21 absent, fraction 1 (ChIP only) | deg | 0.245 [-0.147, 0.543] | 0.193 | 0.445 | 222 | 0.077 [-0.269, 0.453] | 0.223 | 0.030 / 0.204 |
| RAD21 absent, fraction 1 (ChIP only) | cohfix | -0.066 [-0.522, 0.430] | -0.051 | 0.491 | 222 | -0.168 [-0.475, 0.285] | -0.035 | 0.018 / 0.204 |
| RAD21 absent, fraction 1 (ChIP only) | release | -0.470 [-0.644, -0.072] | -0.336 | 0.400 | 222 | -0.462 [-0.638, -0.105] | -0.446 | 0.021 / 0.204 |
| CTCF absent | deg | 0.572 [0.318, 0.704] | 0.477 | 0.327 | 222 | 0.527 [0.250, 0.690] | 0.540 | 0.039 / 0.204 |
| CTCF absent | cohfix | 0.380 [0.005, 0.591] | 0.281 | 0.395 | 222 | 0.356 [0.109, 0.562] | 0.368 | 0.033 / 0.204 |
| CTCF absent | release | 0.588 [0.343, 0.739] | 0.509 | 0.306 | 222 | 0.583 [0.419, 0.715] | 0.580 | 0.040 / 0.204 |

| perturbation | model | loops | loop change measured (median log O/E) | predicted | weaker measured | weaker predicted | Spearman of change | APA untreated | APA treated meas / pred full / pred pert | boundaries | insulation change measured | predicted | Spearman of change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RAD21 absent, fraction 0 | cohfix | 260 | -0.477 | -0.489 | 98.9% | 98.5% | 0.780 | n/a | n/a | n/a | n/a | n/a | n/a |
| RAD21 absent, fraction 0 | release | 260 | -0.477 | -0.533 | 98.9% | 99.6% | 0.791 | n/a | n/a | n/a | n/a | n/a | n/a |
| RAD21 absent, fraction 0.5 | cohfix | 260 | -0.477 | -0.281 | 98.9% | 99.6% | 0.788 | n/a | n/a | n/a | n/a | n/a | n/a |
| RAD21 absent, fraction 0.5 | release | 260 | -0.477 | -0.298 | 98.9% | 99.6% | 0.770 | n/a | n/a | n/a | n/a | n/a | n/a |
| measured auxin ChIP, fraction 0.26 | deg | 260 | -0.477 | -0.201 | 98.9% | 95.8% | 0.623 | n/a | n/a | n/a | n/a | n/a | n/a |
| measured auxin ChIP, fraction 0.26 | cohfix | 260 | -0.477 | -0.341 | 98.9% | 98.9% | 0.804 | n/a | n/a | n/a | n/a | n/a | n/a |
| measured auxin ChIP, fraction 0.26 | release | 260 | -0.477 | -0.428 | 98.9% | 99.2% | 0.833 | n/a | n/a | n/a | n/a | n/a | n/a |
| RAD21 absent, fraction 1 (ChIP only) | deg | 260 | -0.477 | 0.010 | 98.9% | 33.1% | -0.035 | n/a | n/a | n/a | n/a | n/a | n/a |
| RAD21 absent, fraction 1 (ChIP only) | cohfix | 260 | -0.477 | 0.014 | 98.9% | 26.2% | -0.282 | n/a | n/a | n/a | n/a | n/a | n/a |
| RAD21 absent, fraction 1 (ChIP only) | release | 260 | -0.477 | 0.030 | 98.9% | 10.0% | -0.274 | n/a | n/a | n/a | n/a | n/a | n/a |
| CTCF absent | deg | 260 | -0.477 | -0.036 | 98.9% | 85.0% | 0.326 | n/a | n/a | n/a | n/a | n/a | n/a |
| CTCF absent | cohfix | 260 | -0.477 | -0.018 | 98.9% | 70.4% | 0.111 | n/a | n/a | n/a | n/a | n/a | n/a |
| CTCF absent | release | 260 | -0.477 | -0.085 | 98.9% | 98.5% | 0.379 | n/a | n/a | n/a | n/a | n/a | n/a |

### CTCF degron (HCT116, all chromosomes, never trained)

#### CTCF degron, 10 kb

| perturbation | model | Pearson median [q25, q75] | Pearson mean | Pearson sd | regions | SCC median [q25, q75] | Spearman median | mean abs change pred / meas |
|---|---|---|---|---|---|---|---|---|
| CTCF absent | deg | 0.396 [0.217, 0.538] | 0.369 | 0.227 | 411 | 0.394 [0.248, 0.517] | 0.374 | 0.028 / 0.199 |
| CTCF absent | cohfix | 0.403 [0.166, 0.566] | 0.355 | 0.267 | 411 | 0.358 [0.140, 0.515] | 0.383 | 0.025 / 0.199 |
| CTCF absent | release | 0.444 [0.296, 0.574] | 0.401 | 0.238 | 411 | 0.427 [0.277, 0.543] | 0.413 | 0.039 / 0.199 |

| perturbation | model | loops | loop change measured (median log O/E) | predicted | weaker measured | weaker predicted | Spearman of change | APA untreated | APA treated meas / pred full / pred pert | boundaries | insulation change measured | predicted | Spearman of change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTCF absent | deg | 534 | -0.489 | -0.048 | 99.1% | 94.6% | 0.281 | 1.70 | 1.35 / 1.49 / 1.46 | 1811 | 0.117 | 0.009 | 0.451 |
| CTCF absent | cohfix | 534 | -0.489 | -0.030 | 99.1% | 83.3% | 0.437 | 1.70 | 1.35 / 1.44 / 1.42 | 1811 | 0.117 | 0.008 | 0.463 |
| CTCF absent | release | 534 | -0.489 | -0.111 | 99.1% | 98.7% | 0.193 | 1.70 | 1.35 / 1.42 / 1.35 | 1811 | 0.117 | 0.014 | 0.423 |

#### CTCF degron, 25 kb

| perturbation | model | Pearson median [q25, q75] | Pearson mean | Pearson sd | regions | SCC median [q25, q75] | Spearman median | mean abs change pred / meas |
|---|---|---|---|---|---|---|---|---|
| CTCF absent | deg | 0.517 [0.270, 0.674] | 0.444 | 0.322 | 410 | 0.457 [0.219, 0.596] | 0.504 | 0.037 / 0.158 |
| CTCF absent | cohfix | 0.402 [0.151, 0.616] | 0.337 | 0.349 | 410 | 0.309 [0.048, 0.501] | 0.393 | 0.034 / 0.158 |
| CTCF absent | release | 0.520 [0.331, 0.680] | 0.472 | 0.276 | 410 | 0.485 [0.267, 0.629] | 0.506 | 0.038 / 0.158 |

| perturbation | model | loops | loop change measured (median log O/E) | predicted | weaker measured | weaker predicted | Spearman of change | APA untreated | APA treated meas / pred full / pred pert | boundaries | insulation change measured | predicted | Spearman of change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTCF absent | deg | 549 | -0.255 | -0.031 | 96.2% | 82.9% | 0.260 | 1.43 | 1.26 / 1.31 / 1.28 | n/a | n/a | n/a | n/a |
| CTCF absent | cohfix | 549 | -0.255 | -0.017 | 96.2% | 66.5% | 0.165 | 1.43 | 1.26 / 1.26 / 1.26 | n/a | n/a | n/a | n/a |
| CTCF absent | release | 549 | -0.255 | -0.072 | 96.2% | 96.0% | 0.336 | 1.43 | 1.26 / 1.26 / 1.23 | n/a | n/a | n/a | n/a |

### Washout time course

#### Washout, 10 kb

| loops from | model | loops | weaker measured (auxin) | weaker predicted | median change measured | median change predicted (RAD21 auxin) | predicted recovery at f 0.25 / 0.5 / 0.75 | measured recovery 20 / 40 / 60 / 180 min | matching f 20 / 40 / 60 / 180 min |
|---|---|---|---|---|---|---|---|---|---|
| all | deg | 536 | 99.4% | 99.1% | -0.754 | -0.381 | 0.49 / 0.79 / 0.93 | 0.18 / 0.48 / 0.66 / 1.12 | 0.09 / 0.24 / 0.39 / 1.00 |
| all | cohfix | 536 | 99.4% | 100.0% | -0.754 | -0.693 | 0.32 / 0.65 / 0.86 | 0.18 / 0.48 / 0.66 / 1.12 | 0.14 / 0.37 / 0.51 / 1.00 |
| all | release | 536 | 99.4% | 100.0% | -0.754 | -0.682 | 0.28 / 0.64 / 0.89 | 0.18 / 0.48 / 0.66 / 1.12 | 0.16 / 0.39 / 0.51 / 1.00 |
| even_chromosomes | deg | 255 | 98.8% | 98.4% | -0.726 | -0.360 | 0.43 / 0.77 / 0.93 | 0.18 / 0.50 / 0.66 / 1.12 | 0.11 / 0.30 / 0.41 / 1.00 |
| even_chromosomes | cohfix | 255 | 98.8% | 100.0% | -0.726 | -0.674 | 0.32 / 0.66 / 0.87 | 0.18 / 0.50 / 0.66 / 1.12 | 0.15 / 0.38 / 0.50 / 1.00 |
| even_chromosomes | release | 255 | 98.8% | 100.0% | -0.726 | -0.675 | 0.28 / 0.65 / 0.90 | 0.18 / 0.50 / 0.66 / 1.12 | 0.16 / 0.40 / 0.51 / 1.00 |

#### Washout, 25 kb

| loops from | model | loops | weaker measured (auxin) | weaker predicted | median change measured | median change predicted (RAD21 auxin) | predicted recovery at f 0.25 / 0.5 / 0.75 | measured recovery 20 / 40 / 60 / 180 min | matching f 20 / 40 / 60 / 180 min |
|---|---|---|---|---|---|---|---|---|---|
| all | deg | 549 | 99.5% | 96.0% | -0.481 | -0.105 | 0.42 / 0.67 / 0.82 | 0.15 / 0.41 / 0.60 / 1.01 | 0.09 / 0.24 / 0.44 / 1.00 |
| all | cohfix | 549 | 99.5% | 98.7% | -0.481 | -0.346 | 0.29 / 0.61 / 0.83 | 0.15 / 0.41 / 0.60 / 1.01 | 0.13 / 0.34 / 0.50 / 1.00 |
| all | release | 549 | 99.5% | 100.0% | -0.481 | -0.419 | 0.24 / 0.57 / 0.83 | 0.15 / 0.41 / 0.60 / 1.01 | 0.16 / 0.38 / 0.53 / 1.00 |
| even_chromosomes | deg | 260 | 98.9% | 98.9% | -0.477 | -0.119 | 0.46 / 0.70 / 0.86 | 0.15 / 0.42 / 0.61 / 1.00 | 0.08 / 0.23 / 0.41 / 1.00 |
| even_chromosomes | cohfix | 260 | 98.9% | 98.5% | -0.477 | -0.323 | 0.27 / 0.58 / 0.84 | 0.15 / 0.42 / 0.61 / 1.00 | 0.14 / 0.37 / 0.52 / 1.00 |
| even_chromosomes | release | 260 | 98.9% | 100.0% | -0.477 | -0.437 | 0.25 / 0.59 / 0.85 | 0.15 / 0.42 / 0.61 / 1.00 | 0.15 / 0.37 / 0.52 / 1.00 |

### U-937 STAG2 knockout

#### U-937 STAG2 knockout, 10 kb

| input | model | regions | change-map Pearson median | SCC median | loops | loop change measured (mean log2) | predicted | Pearson of loop change | Spearman | sign agreement | short loops (<200 kb) meas / pred | long loops meas / pred | boundaries | Pearson of insulation change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RAD21 and SMC3 absent, fraction 1 | deg | 413 | -0.025 | -0.068 | 626 | 0.080 | 0.070 | -0.436 | -0.437 | 40.4% | -0.105 / 0.107 | 0.181 / 0.050 | 832 | -0.271 |
| RAD21 and SMC3 absent, fraction 1 | cohfix | 413 | 0.059 | -0.026 | 626 | 0.080 | 0.065 | -0.305 | -0.333 | 42.8% | -0.105 / 0.096 | 0.181 / 0.049 | 832 | -0.142 |
| RAD21 and SMC3 absent, fraction 1 | release | 413 | -0.020 | -0.057 | 626 | 0.080 | 0.085 | -0.406 | -0.386 | 42.0% | -0.105 / 0.116 | 0.181 / 0.068 | 832 | -0.197 |
| RAD21 and SMC3 absent, fraction 0 | cohfix | 413 | 0.214 | 0.147 | 626 | 0.080 | -0.425 | 0.614 | 0.448 | 55.6% | -0.105 / -0.537 | 0.181 / -0.363 | 832 | 0.036 |
| RAD21 and SMC3 absent, fraction 0 | release | 413 | 0.205 | 0.151 | 626 | 0.080 | -0.465 | 0.573 | 0.411 | 55.3% | -0.105 / -0.549 | 0.181 / -0.418 | 832 | 0.011 |
| RAD21 and SMC3 absent, fraction 0.5 | cohfix | 413 | 0.195 | 0.141 | 626 | 0.080 | -0.226 | 0.560 | 0.433 | 56.1% | -0.105 / -0.275 | 0.181 / -0.199 | 832 | 0.021 |
| RAD21 and SMC3 absent, fraction 0.5 | release | 413 | 0.185 | 0.146 | 626 | 0.080 | -0.197 | 0.517 | 0.404 | 55.6% | -0.105 / -0.234 | 0.181 / -0.176 | 832 | -0.059 |
| wild-type ChIP, fraction 0.5 | cohfix | 413 | 0.200 | 0.149 | 626 | 0.080 | -0.231 | 0.599 | 0.455 | 55.3% | -0.105 / -0.294 | 0.181 / -0.197 | 832 | 0.072 |
| wild-type ChIP, fraction 0.5 | release | 413 | 0.190 | 0.149 | 626 | 0.080 | -0.249 | 0.581 | 0.435 | 56.2% | -0.105 / -0.310 | 0.181 / -0.215 | 832 | 0.050 |
| wild-type ChIP, fraction 0.75 | cohfix | 413 | 0.204 | 0.136 | 626 | 0.080 | -0.109 | 0.556 | 0.443 | 55.4% | -0.105 / -0.139 | 0.181 / -0.092 | 832 | 0.081 |
| wild-type ChIP, fraction 0.75 | release | 413 | 0.197 | 0.151 | 626 | 0.080 | -0.084 | 0.527 | 0.440 | 56.9% | -0.105 / -0.113 | 0.181 / -0.069 | 832 | 0.034 |
| cohesin only at CTCF peaks | deg | 413 | 0.102 | 0.061 | 626 | 0.080 | -0.005 | 0.069 | 0.035 | 52.1% | -0.105 / -0.007 | 0.181 / -0.003 | 832 | 0.087 |
| cohesin only at CTCF peaks | cohfix | 413 | 0.085 | 0.076 | 626 | 0.080 | -0.024 | 0.276 | 0.291 | 57.8% | -0.105 / -0.034 | 0.181 / -0.019 | 832 | 0.137 |
| cohesin only at CTCF peaks | release | 413 | 0.038 | 0.006 | 626 | 0.080 | 0.011 | -0.158 | -0.223 | 43.5% | -0.105 / 0.015 | 0.181 / 0.008 | 832 | -0.016 |
| cohesin enrichment halved | deg | 413 | 0.083 | 0.058 | 626 | 0.080 | -0.004 | 0.091 | 0.074 | 52.9% | -0.105 / -0.006 | 0.181 / -0.003 | 832 | 0.069 |
| cohesin enrichment halved | cohfix | 413 | 0.070 | 0.069 | 626 | 0.080 | -0.013 | 0.272 | 0.304 | 58.6% | -0.105 / -0.019 | 0.181 / -0.010 | 832 | 0.097 |
| cohesin enrichment halved | release | 413 | 0.034 | 0.010 | 626 | 0.080 | 0.004 | -0.133 | -0.177 | 47.3% | -0.105 / 0.005 | 0.181 / 0.004 | 832 | -0.010 |

Clone-pair ceiling of the change map (Pearson median): 0.254, 0.339, 0.122.

#### U-937 STAG2 knockout, 25 kb

| input | model | regions | change-map Pearson median | SCC median | loops | loop change measured (mean log2) | predicted | Pearson of loop change | Spearman | sign agreement | short loops (<200 kb) meas / pred | long loops meas / pred | boundaries | Pearson of insulation change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RAD21 and SMC3 absent, fraction 1 | deg | 413 | -0.025 | -0.033 | 211 | -0.033 | 0.098 | 0.055 | -0.000 | 36.5% | -0.117 / 0.117 | -0.020 / 0.094 | 782 | -0.160 |
| RAD21 and SMC3 absent, fraction 1 | cohfix | 413 | 0.130 | 0.008 | 211 | -0.033 | 0.034 | -0.224 | -0.179 | 39.3% | -0.117 / 0.032 | -0.020 / 0.035 | 782 | -0.009 |
| RAD21 and SMC3 absent, fraction 1 | release | 413 | 0.036 | -0.035 | 211 | -0.033 | 0.061 | -0.154 | -0.056 | 38.4% | -0.117 / 0.055 | -0.020 / 0.062 | 782 | -0.150 |
| RAD21 and SMC3 absent, fraction 0 | cohfix | 413 | 0.260 | 0.230 | 211 | -0.033 | -0.502 | 0.128 | 0.081 | 65.4% | -0.117 / -0.553 | -0.020 / -0.494 | 782 | 0.056 |
| RAD21 and SMC3 absent, fraction 0 | release | 413 | 0.246 | 0.200 | 211 | -0.033 | -0.536 | 0.138 | 0.095 | 65.4% | -0.117 / -0.558 | -0.020 / -0.532 | 782 | 0.013 |
| RAD21 and SMC3 absent, fraction 0.5 | cohfix | 413 | 0.266 | 0.217 | 211 | -0.033 | -0.241 | 0.038 | 0.016 | 65.4% | -0.117 / -0.240 | -0.020 / -0.242 | 782 | 0.058 |
| RAD21 and SMC3 absent, fraction 0.5 | release | 413 | 0.232 | 0.203 | 211 | -0.033 | -0.245 | 0.054 | 0.049 | 64.9% | -0.117 / -0.258 | -0.020 / -0.243 | 782 | -0.060 |
| wild-type ChIP, fraction 0.5 | cohfix | 413 | 0.261 | 0.210 | 211 | -0.033 | -0.315 | 0.124 | 0.060 | 65.9% | -0.117 / -0.320 | -0.020 / -0.314 | 782 | 0.054 |
| wild-type ChIP, fraction 0.5 | release | 413 | 0.232 | 0.188 | 211 | -0.033 | -0.287 | 0.122 | 0.056 | 65.9% | -0.117 / -0.306 | -0.020 / -0.283 | 782 | 0.056 |
| wild-type ChIP, fraction 0.75 | cohfix | 413 | 0.262 | 0.203 | 211 | -0.033 | -0.115 | 0.112 | 0.068 | 65.4% | -0.117 / -0.108 | -0.020 / -0.116 | 782 | 0.029 |
| wild-type ChIP, fraction 0.75 | release | 413 | 0.235 | 0.191 | 211 | -0.033 | -0.102 | 0.096 | 0.019 | 65.9% | -0.117 / -0.115 | -0.020 / -0.100 | 782 | 0.032 |
| cohesin only at CTCF peaks | deg | 413 | 0.176 | 0.126 | 211 | -0.033 | -0.017 | 0.067 | 0.129 | 55.5% | -0.117 / -0.005 | -0.020 / -0.019 | 782 | 0.100 |
| cohesin only at CTCF peaks | cohfix | 413 | 0.269 | 0.165 | 211 | -0.033 | -0.056 | 0.072 | 0.061 | 59.2% | -0.117 / -0.047 | -0.020 / -0.058 | 782 | 0.106 |
| cohesin only at CTCF peaks | release | 413 | 0.039 | 0.011 | 211 | -0.033 | 0.007 | -0.053 | -0.049 | 42.7% | -0.117 / 0.001 | -0.020 / 0.007 | 782 | -0.025 |
| cohesin enrichment halved | deg | 413 | 0.170 | 0.118 | 211 | -0.033 | -0.003 | 0.107 | 0.128 | 54.0% | -0.117 / 0.001 | -0.020 / -0.004 | 782 | 0.043 |
| cohesin enrichment halved | cohfix | 413 | 0.242 | 0.150 | 211 | -0.033 | -0.026 | 0.101 | 0.077 | 61.1% | -0.117 / -0.024 | -0.020 / -0.027 | 782 | 0.035 |
| cohesin enrichment halved | release | 413 | 0.054 | 0.013 | 211 | -0.033 | 0.002 | -0.028 | -0.013 | 44.5% | -0.117 / -0.001 | -0.020 / 0.003 | 782 | -0.020 |

Clone-pair ceiling of the change map (Pearson median): 0.547, 0.615, 0.293.

### AML patient-line maps, intact prediction (413 regions)

| kb | map | model | Pearson median | SCC median | Pearson mean | regions |
|---|---|---|---|---|---|---|
| 10 | MOLM-13 | deg | 0.848 | 0.748 | 0.773 | 413 |
| 10 | MOLM-13 | cohfix | 0.882 | 0.795 | 0.869 | 413 |
| 10 | MOLM-13 | release | 0.887 | 0.792 | 0.866 | 413 |
| 10 | U-937 wild type, clone 1 | deg | 0.861 | 0.777 | 0.818 | 413 |
| 10 | U-937 wild type, clone 1 | cohfix | 0.868 | 0.781 | 0.821 | 413 |
| 10 | U-937 wild type, clone 1 | release | 0.860 | 0.768 | 0.816 | 413 |
| 10 | U-937 STAG2 KO clone 1, wild-type input | deg | 0.762 | 0.714 | 0.693 | 413 |
| 10 | U-937 STAG2 KO clone 1, wild-type input | cohfix | 0.781 | 0.723 | 0.695 | 413 |
| 10 | U-937 STAG2 KO clone 1, wild-type input | release | 0.769 | 0.714 | 0.693 | 413 |
| 25 | MOLM-13 | deg | 0.900 | 0.819 | 0.871 | 413 |
| 25 | MOLM-13 | cohfix | 0.912 | 0.822 | 0.891 | 413 |
| 25 | MOLM-13 | release | 0.916 | 0.835 | 0.896 | 413 |
| 25 | U-937 wild type, clone 1 | deg | 0.879 | 0.813 | 0.840 | 413 |
| 25 | U-937 wild type, clone 1 | cohfix | 0.891 | 0.825 | 0.855 | 413 |
| 25 | U-937 wild type, clone 1 | release | 0.891 | 0.824 | 0.851 | 413 |
| 25 | U-937 STAG2 KO clone 1, wild-type input | deg | 0.835 | 0.776 | 0.762 | 413 |
| 25 | U-937 STAG2 KO clone 1, wild-type input | cohfix | 0.832 | 0.785 | 0.765 | 413 |
| 25 | U-937 STAG2 KO clone 1, wild-type input | release | 0.830 | 0.782 | 0.759 | 413 |

### Methylation proxy

#### Methylation proxy, 10 kb, held-out cells (Pearson; loss [95% CI] / gain [95% CI] / replacement fraction (95% CI))

| set | dedicated, no consensus | dedicated, consensus | release, consensus plane | release, plane at 0 |
|---|---|---|---|---|
| minus CTCF | 0.037 [0.028, 0.053] / 0.001 [-0.004, 0.010] / 0.02 (-0.12 to 0.25) | -0.001 [-0.002, 0.002] / 0.002 [-0.001, 0.005] / - | 0.003 [0.001, 0.006] / 0.001 [-0.001, 0.004] / - | 0.043 [0.029, 0.052] / 0.004 [-0.006, 0.010] / 0.10 (-0.16 to 0.25) |
| minus cohesin | 0.008 [0.000, 0.015] / 0.009 [0.000, 0.015] / 1.19 (-0.13 to 8.94) | 0.002 [0.001, 0.006] / 0.003 [0.000, 0.007] / - | 0.002 [0.000, 0.004] / 0.005 [0.002, 0.008] / - | 0.013 [0.005, 0.020] / 0.011 [0.003, 0.017] / 0.89 (0.25 to 1.71) |
| minus active | 0.024 [0.016, 0.035] / 0.011 [0.003, 0.018] / 0.48 (0.13 to 0.80) | 0.005 [0.003, 0.010] / 0.001 [-0.002, 0.003] / - | 0.009 [0.006, 0.013] / 0.002 [-0.000, 0.005] / 0.21 (-0.03 to 0.51) | 0.021 [0.012, 0.030] / 0.007 [-0.000, 0.015] / 0.31 (-0.01 to 0.68) |
| minus gene body | -0.003 [-0.012, 0.006] / 0.010 [0.000, 0.014] / - | -0.001 [-0.003, 0.002] / 0.004 [0.001, 0.006] / - | 0.001 [-0.001, 0.004] / 0.003 [0.000, 0.005] / - | 0.001 [-0.009, 0.010] / 0.010 [0.003, 0.015] / - |
| minus repressive | 0.001 [-0.004, 0.010] / 0.009 [0.003, 0.017] / - | 0.000 [-0.002, 0.004] / -0.000 [-0.003, 0.003] / - | 0.001 [-0.001, 0.004] / 0.002 [0.000, 0.005] / - | 0.004 [-0.003, 0.011] / 0.002 [-0.002, 0.011] / - |
| cum2 | 0.007 [-0.002, 0.017] / 0.009 [0.000, 0.016] / - | 0.003 [-0.000, 0.006] / 0.001 [-0.002, 0.004] / - | 0.001 [-0.001, 0.005] / 0.004 [0.000, 0.006] / - | 0.013 [0.004, 0.022] / 0.004 [-0.002, 0.013] / 0.33 (-0.19 to 1.39) |
| cum3 | 0.014 [0.004, 0.025] / 0.011 [0.001, 0.020] / 0.79 (0.09 to 2.68) | 0.003 [-0.001, 0.007] / 0.005 [0.001, 0.008] / - | 0.001 [-0.001, 0.006] / 0.003 [0.001, 0.006] / - | 0.018 [0.006, 0.028] / 0.010 [0.003, 0.016] / 0.56 (0.16 to 1.54) |
| CTCF only | 0.092 [0.078, 0.112] / 0.033 [0.023, 0.047] / 0.36 (0.25 to 0.48) | 0.019 [0.014, 0.024] / 0.005 [-0.001, 0.009] / 0.25 (-0.06 to 0.47) | 0.021 [0.016, 0.026] / 0.005 [0.001, 0.010] / 0.25 (0.04 to 0.43) | 0.084 [0.065, 0.098] / 0.030 [0.011, 0.041] / 0.35 (0.15 to 0.46) |
| sequence only | 0.223 [0.196, 0.240] / 0.074 [0.052, 0.084] / 0.33 (0.24 to 0.38) | 0.024 [0.019, 0.032] / 0.007 [0.002, 0.013] / 0.29 (0.07 to 0.49) | 0.027 [0.023, 0.033] / 0.013 [0.008, 0.017] / 0.47 (0.30 to 0.58) | 0.201 [0.179, 0.228] / 0.057 [0.035, 0.081] / 0.28 (0.18 to 0.39) |
| cells, seed noise | 2 cells, noise 0.0062 | 2 cells, noise 0.0062 | 2 cells, noise 0.0062 | 2 cells, noise 0.0062 |
| Meth added to all 13 tracks | gain on the full set 0.007 [0.001, 0.015] | gain on the full set 0.002 [-0.001, 0.004] | gain on the full set 0.002 [-0.000, 0.004] | gain on the full set 0.004 [-0.002, 0.010] |

#### Methylation proxy, 10 kb, training WGBS cells (Pearson; loss [95% CI] / gain [95% CI] / replacement fraction (95% CI))

| set | dedicated, no consensus | dedicated, consensus | release, consensus plane | release, plane at 0 |
|---|---|---|---|---|
| minus CTCF | 0.015 [0.002, 0.031] / 0.007 [-0.001, 0.017] / 0.48 (-0.24 to 1.47) | 0.001 [-0.001, 0.002] / 0.002 [0.001, 0.004] / - | 0.002 [-0.000, 0.004] / 0.001 [0.000, 0.004] / - | 0.017 [0.007, 0.032] / 0.004 [-0.000, 0.009] / 0.25 (-0.02 to 0.81) |
| minus cohesin | 0.013 [0.003, 0.023] / 0.008 [0.002, 0.014] / 0.59 (0.15 to 2.08) | 0.017 [0.009, 0.027] / 0.016 [0.008, 0.025] / 0.93 (0.72 to 1.13) | 0.009 [0.003, 0.016] / 0.007 [0.002, 0.014] / 0.84 (0.47 to 1.22) | 0.021 [0.007, 0.035] / 0.012 [0.003, 0.019] / 0.58 (0.26 to 0.95) |
| minus active | 0.055 [0.026, 0.097] / 0.008 [-0.000, 0.014] / 0.14 (-0.00 to 0.36) | 0.012 [0.008, 0.015] / 0.003 [-0.000, 0.005] / 0.25 (-0.02 to 0.47) | 0.025 [0.010, 0.053] / 0.003 [0.001, 0.007] / 0.12 (0.03 to 0.44) | 0.048 [0.023, 0.089] / 0.009 [0.001, 0.020] / 0.18 (0.02 to 0.45) |
| minus gene body | 0.016 [0.006, 0.023] / 0.007 [-0.000, 0.015] / 0.44 (-0.02 to 1.21) | 0.000 [-0.001, 0.002] / 0.001 [-0.001, 0.004] / - | 0.003 [0.001, 0.006] / 0.002 [0.000, 0.004] / - | 0.013 [0.005, 0.021] / 0.005 [0.000, 0.011] / 0.35 (-0.00 to 0.90) |
| minus repressive | 0.007 [0.000, 0.016] / 0.008 [0.000, 0.015] / 1.13 (-0.02 to 11.16) | 0.002 [0.000, 0.004] / 0.002 [0.000, 0.004] / - | 0.002 [0.000, 0.004] / 0.002 [0.000, 0.004] / - | 0.005 [0.001, 0.011] / 0.004 [0.001, 0.010] / - |
| cum2 | 0.020 [0.010, 0.032] / 0.008 [0.002, 0.014] / 0.39 (0.08 to 0.86) | 0.017 [0.009, 0.030] / 0.015 [0.007, 0.027] / 0.86 (0.65 to 1.04) | 0.009 [0.004, 0.016] / 0.006 [0.002, 0.012] / 0.70 (0.39 to 0.97) | 0.022 [0.010, 0.035] / 0.008 [0.002, 0.017] / 0.37 (0.14 to 0.65) |
| cum3 | 0.039 [0.028, 0.050] / 0.007 [-0.001, 0.016] / 0.19 (-0.02 to 0.42) | 0.018 [0.009, 0.029] / 0.014 [0.006, 0.024] / 0.76 (0.57 to 1.02) | 0.012 [0.006, 0.018] / 0.006 [0.002, 0.011] / 0.51 (0.28 to 0.67) | 0.036 [0.024, 0.046] / 0.009 [0.002, 0.016] / 0.24 (0.05 to 0.41) |
| CTCF only | 0.174 [0.130, 0.236] / 0.033 [0.012, 0.056] / 0.19 (0.06 to 0.34) | 0.044 [0.027, 0.066] / 0.017 [0.005, 0.033] / 0.39 (0.18 to 0.54) | 0.068 [0.030, 0.129] / 0.015 [0.007, 0.026] / 0.22 (0.11 to 0.42) | 0.173 [0.124, 0.244] / 0.044 [0.024, 0.064] / 0.25 (0.13 to 0.41) |
| sequence only | 0.254 [0.199, 0.321] / 0.048 [0.016, 0.071] / 0.19 (0.06 to 0.30) | 0.045 [0.027, 0.064] / 0.015 [0.004, 0.028] / 0.32 (0.12 to 0.47) | 0.073 [0.031, 0.131] / 0.013 [0.004, 0.023] / 0.17 (0.05 to 0.45) | 0.253 [0.190, 0.327] / 0.058 [0.031, 0.085] / 0.23 (0.12 to 0.37) |
| cells, seed noise | 7 cells, noise 0.0062 | 5 cells, noise 0.0062 | 7 cells, noise 0.0062 | 7 cells, noise 0.0062 |
| Meth added to all 13 tracks | gain on the full set 0.007 [0.001, 0.014] | gain on the full set 0.002 [0.000, 0.005] | gain on the full set 0.001 [-0.000, 0.003] | gain on the full set 0.003 [-0.001, 0.008] |

#### Methylation proxy, 10 kb, training RRBS cells (Pearson; loss [95% CI] / gain [95% CI] / replacement fraction (95% CI))

| set | dedicated, no consensus | dedicated, consensus | release, consensus plane | release, plane at 0 |
|---|---|---|---|---|
| minus CTCF | n/a | n/a | -0.001 [-0.008, 0.005] / 0.005 [0.001, 0.008] / - | 0.051 [0.034, 0.074] / 0.003 [-0.007, 0.010] / 0.05 (-0.15 to 0.21) |
| minus cohesin | n/a | n/a | 0.004 [0.000, 0.008] / 0.002 [-0.000, 0.006] / - | 0.009 [0.000, 0.019] / -0.001 [-0.005, 0.007] / - |
| minus active | n/a | n/a | 0.009 [0.002, 0.019] / 0.005 [0.002, 0.007] / 0.53 (0.14 to 2.32) | 0.035 [0.018, 0.054] / 0.004 [-0.003, 0.009] / 0.11 (-0.09 to 0.34) |
| minus gene body | n/a | n/a | 0.001 [-0.001, 0.003] / 0.002 [-0.001, 0.003] / - | 0.008 [-0.000, 0.013] / 0.000 [-0.005, 0.006] / - |
| minus repressive | n/a | n/a | 0.002 [-0.000, 0.004] / 0.001 [-0.002, 0.003] / - | 0.006 [-0.003, 0.014] / 0.001 [-0.005, 0.006] / - |
| cum2 | n/a | n/a | 0.005 [0.002, 0.009] / 0.002 [-0.000, 0.005] / - | 0.012 [0.004, 0.020] / 0.002 [-0.005, 0.006] / 0.15 (-0.67 to 0.66) |
| cum3 | n/a | n/a | 0.006 [0.003, 0.010] / 0.001 [-0.001, 0.005] / - | 0.021 [0.009, 0.032] / -0.000 [-0.006, 0.007] / -0.01 (-0.33 to 0.32) |
| CTCF only | n/a | n/a | 0.022 [0.007, 0.035] / -0.000 [-0.004, 0.005] / -0.00 (-0.41 to 0.16) | 0.095 [0.069, 0.115] / 0.001 [-0.009, 0.008] / 0.01 (-0.10 to 0.08) |
| sequence only | n/a | n/a | 0.027 [0.011, 0.042] / 0.001 [-0.001, 0.003] / 0.03 (-0.09 to 0.13) | 0.230 [0.194, 0.275] / 0.005 [-0.010, 0.017] / 0.02 (-0.04 to 0.07) |
| cells, seed noise | n/a | n/a | 3 cells, noise 0.0062 | 3 cells, noise 0.0062 |
| Meth added to all 13 tracks | n/a | n/a | gain on the full set 0.002 [-0.000, 0.004] | gain on the full set -0.004 [-0.007, 0.002] |

#### Methylation proxy, 25 kb, held-out cells (Pearson; loss [95% CI] / gain [95% CI] / replacement fraction (95% CI))

| set | dedicated, no consensus | dedicated, consensus (motif planes) | release, consensus plane | release, plane at 0 |
|---|---|---|---|---|
| minus CTCF | 0.055 [0.036, 0.069] / 0.009 [0.001, 0.018] / 0.17 (0.03 to 0.34) | 0.001 [-0.002, 0.003] / 0.002 [-0.002, 0.005] / - | 0.001 [-0.002, 0.003] / 0.001 [-0.001, 0.003] / - | 0.056 [0.039, 0.067] / 0.010 [-0.003, 0.018] / 0.18 (-0.08 to 0.31) |
| minus cohesin | 0.002 [-0.006, 0.010] / 0.001 [-0.005, 0.010] / - | 0.009 [0.005, 0.012] / 0.006 [0.002, 0.010] / 0.73 (0.31 to 1.07) | 0.002 [0.000, 0.005] / 0.005 [0.002, 0.008] / - | 0.004 [-0.005, 0.012] / 0.006 [-0.001, 0.016] / - |
| minus active | 0.013 [-0.003, 0.022] / 0.012 [0.001, 0.019] / - | 0.004 [0.001, 0.008] / 0.004 [0.000, 0.007] / - | 0.003 [0.000, 0.007] / 0.001 [-0.002, 0.004] / - | 0.024 [0.009, 0.032] / 0.007 [-0.002, 0.015] / 0.28 (-0.13 to 0.78) |
| minus gene body | 0.002 [-0.013, 0.011] / 0.018 [0.006, 0.024] / - | 0.000 [-0.002, 0.003] / 0.000 [-0.003, 0.003] / - | 0.002 [0.000, 0.005] / 0.004 [0.002, 0.006] / - | -0.005 [-0.016, 0.008] / 0.005 [-0.003, 0.015] / - |
| minus repressive | -0.002 [-0.012, 0.008] / 0.001 [-0.005, 0.015] / - | 0.001 [-0.001, 0.003] / -0.000 [-0.004, 0.003] / - | 0.001 [-0.001, 0.004] / 0.002 [0.000, 0.005] / - | 0.012 [0.001, 0.018] / 0.013 [-0.000, 0.019] / 1.08 (-0.15 to 3.93) |
| cum2 | 0.002 [-0.009, 0.014] / -0.002 [-0.008, 0.012] / - | 0.007 [0.004, 0.012] / 0.004 [0.001, 0.009] / 0.60 (0.11 to 1.00) | 0.001 [-0.001, 0.005] / 0.003 [0.001, 0.006] / - | 0.015 [0.002, 0.023] / 0.011 [0.001, 0.020] / 0.71 (0.05 to 2.68) |
| cum3 | 0.009 [-0.007, 0.022] / 0.004 [-0.001, 0.014] / - | 0.005 [0.002, 0.010] / 0.001 [-0.003, 0.006] / - | 0.005 [0.002, 0.008] / 0.005 [0.002, 0.007] / - | 0.020 [0.002, 0.030] / 0.012 [-0.001, 0.021] / 0.60 (-0.15 to 2.56) |
| CTCF only | 0.084 [0.063, 0.105] / 0.043 [0.028, 0.062] / 0.51 (0.35 to 0.72) | 0.018 [0.014, 0.024] / 0.013 [0.009, 0.019] / 0.72 (0.49 to 0.91) | 0.018 [0.013, 0.022] / 0.010 [0.004, 0.013] / 0.54 (0.27 to 0.74) | 0.086 [0.068, 0.104] / 0.032 [0.020, 0.051] / 0.37 (0.25 to 0.57) |
| sequence only | 0.199 [0.170, 0.231] / 0.066 [0.042, 0.091] / 0.33 (0.23 to 0.45) | 0.022 [0.017, 0.027] / 0.014 [0.011, 0.019] / 0.65 (0.50 to 0.88) | 0.025 [0.018, 0.029] / 0.015 [0.009, 0.019] / 0.62 (0.40 to 0.77) | 0.223 [0.193, 0.247] / 0.082 [0.056, 0.100] / 0.37 (0.27 to 0.44) |
| cells, seed noise | 2 cells, noise 0.0072 | 2 cells, noise 0.0062 | 2 cells, noise 0.0062 | 2 cells, noise 0.0072 |
| Meth added to all 13 tracks | gain on the full set 0.002 [-0.004, 0.014] | gain on the full set 0.001 [-0.003, 0.004] | gain on the full set 0.003 [0.001, 0.005] | gain on the full set 0.003 [-0.004, 0.012] |

#### Methylation proxy, 25 kb, training WGBS cells (Pearson; loss [95% CI] / gain [95% CI] / replacement fraction (95% CI))

| set | dedicated, no consensus | dedicated, consensus (motif planes) | release, consensus plane | release, plane at 0 |
|---|---|---|---|---|
| minus CTCF | 0.035 [0.019, 0.052] / 0.004 [-0.003, 0.010] / 0.12 (-0.08 to 0.36) | 0.001 [-0.002, 0.003] / 0.005 [0.001, 0.008] / - | 0.002 [-0.001, 0.005] / 0.002 [0.000, 0.004] / - | 0.027 [0.013, 0.042] / 0.007 [-0.000, 0.013] / 0.26 (-0.00 to 0.84) |
| minus cohesin | 0.014 [0.005, 0.024] / 0.004 [-0.000, 0.012] / 0.30 (-0.04 to 0.95) | 0.029 [0.022, 0.033] / 0.026 [0.020, 0.029] / 0.89 (0.76 to 1.05) | 0.008 [0.002, 0.016] / 0.007 [0.002, 0.013] / 0.89 (0.54 to 1.75) | 0.013 [0.004, 0.025] / 0.013 [0.005, 0.020] / 1.02 (0.49 to 2.75) |
| minus active | 0.053 [0.028, 0.089] / 0.006 [-0.004, 0.017] / 0.12 (-0.06 to 0.42) | 0.010 [0.005, 0.015] / 0.006 [0.003, 0.009] / 0.63 (0.32 to 1.13) | 0.022 [0.007, 0.047] / 0.004 [0.001, 0.007] / 0.19 (0.05 to 0.60) | 0.045 [0.023, 0.079] / 0.011 [0.001, 0.021] / 0.24 (0.03 to 0.50) |
| minus gene body | 0.007 [0.000, 0.016] / 0.002 [-0.001, 0.010] / - | 0.000 [-0.003, 0.003] / 0.003 [0.001, 0.007] / - | 0.001 [-0.001, 0.005] / 0.002 [0.001, 0.004] / - | 0.010 [0.002, 0.020] / 0.005 [0.001, 0.012] / 0.56 (0.05 to 2.42) |
| minus repressive | 0.006 [-0.004, 0.018] / 0.005 [-0.001, 0.013] / - | 0.001 [-0.002, 0.005] / 0.005 [0.002, 0.008] / - | 0.002 [-0.000, 0.004] / 0.002 [0.000, 0.005] / - | 0.006 [0.001, 0.014] / 0.010 [0.003, 0.017] / - |
| cum2 | 0.021 [0.011, 0.031] / 0.007 [0.001, 0.016] / 0.34 (0.07 to 0.67) | 0.029 [0.023, 0.034] / 0.026 [0.021, 0.030] / 0.89 (0.75 to 1.04) | 0.007 [0.002, 0.015] / 0.005 [0.001, 0.011] / 0.79 (0.28 to 1.50) | 0.016 [0.006, 0.031] / 0.013 [0.005, 0.022] / 0.80 (0.43 to 1.52) |
| cum3 | 0.034 [0.023, 0.043] / 0.007 [0.001, 0.014] / 0.21 (0.04 to 0.40) | 0.023 [0.018, 0.028] / 0.018 [0.014, 0.021] / 0.78 (0.64 to 0.98) | 0.010 [0.005, 0.016] / 0.006 [0.002, 0.012] / 0.66 (0.36 to 0.93) | 0.028 [0.014, 0.040] / 0.011 [0.003, 0.019] / 0.38 (0.13 to 0.72) |
| CTCF only | 0.179 [0.138, 0.238] / 0.042 [0.020, 0.063] / 0.23 (0.10 to 0.38) | 0.048 [0.041, 0.057] / 0.026 [0.019, 0.034] / 0.53 (0.44 to 0.64) | 0.065 [0.024, 0.142] / 0.014 [0.005, 0.026] / 0.22 (0.07 to 0.48) | 0.175 [0.123, 0.251] / 0.051 [0.032, 0.075] / 0.29 (0.17 to 0.46) |
| sequence only | 0.261 [0.201, 0.331] / 0.059 [0.025, 0.090] / 0.23 (0.09 to 0.38) | 0.050 [0.041, 0.057] / 0.024 [0.017, 0.032] / 0.49 (0.39 to 0.60) | 0.069 [0.025, 0.132] / 0.012 [0.003, 0.021] / 0.17 (0.03 to 0.45) | 0.265 [0.200, 0.363] / 0.075 [0.041, 0.111] / 0.28 (0.14 to 0.45) |
| cells, seed noise | 7 cells, noise 0.0072 | 2 cells, noise 0.0062 | 7 cells, noise 0.0062 | 7 cells, noise 0.0072 |
| Meth added to all 13 tracks | gain on the full set 0.004 [-0.002, 0.010] | gain on the full set 0.005 [0.001, 0.009] | gain on the full set 0.002 [-0.000, 0.004] | gain on the full set 0.006 [0.000, 0.012] |

#### Methylation proxy, 25 kb, training RRBS cells (Pearson; loss [95% CI] / gain [95% CI] / replacement fraction (95% CI))

| set | dedicated, no consensus | dedicated, consensus (motif planes) | release, consensus plane | release, plane at 0 |
|---|---|---|---|---|
| minus CTCF | n/a | n/a | -0.001 [-0.009, 0.004] / 0.005 [0.001, 0.007] / - | 0.049 [0.030, 0.070] / 0.002 [-0.008, 0.009] / 0.05 (-0.20 to 0.19) |
| minus cohesin | n/a | n/a | 0.003 [-0.002, 0.009] / 0.003 [0.000, 0.007] / - | -0.003 [-0.009, 0.008] / -0.004 [-0.009, 0.004] / - |
| minus active | n/a | n/a | 0.008 [-0.003, 0.018] / 0.005 [0.001, 0.007] / - | 0.034 [0.011, 0.061] / 0.005 [-0.006, 0.012] / 0.13 (-0.27 to 0.38) |
| minus gene body | n/a | n/a | 0.001 [-0.002, 0.003] / 0.002 [0.000, 0.004] / - | -0.000 [-0.010, 0.011] / -0.000 [-0.008, 0.006] / - |
| minus repressive | n/a | n/a | 0.002 [-0.001, 0.004] / 0.001 [-0.000, 0.004] / - | 0.002 [-0.004, 0.012] / -0.001 [-0.007, 0.008] / - |
| cum2 | n/a | n/a | 0.003 [-0.002, 0.008] / 0.002 [-0.001, 0.006] / - | 0.002 [-0.006, 0.013] / -0.001 [-0.007, 0.008] / - |
| cum3 | n/a | n/a | 0.005 [0.001, 0.009] / 0.002 [-0.001, 0.006] / - | 0.015 [0.003, 0.029] / 0.004 [-0.005, 0.012] / 0.24 (-0.40 to 0.93) |
| CTCF only | n/a | n/a | 0.022 [0.007, 0.037] / 0.002 [-0.002, 0.005] / 0.10 (-0.17 to 0.17) | 0.098 [0.076, 0.122] / 0.010 [0.001, 0.020] / 0.11 (0.01 to 0.20) |
| sequence only | n/a | n/a | 0.024 [0.008, 0.044] / -0.000 [-0.003, 0.002] / -0.01 (-0.27 to 0.13) | 0.247 [0.211, 0.293] / 0.006 [-0.006, 0.020] / 0.03 (-0.02 to 0.08) |
| cells, seed noise | n/a | n/a | 3 cells, noise 0.0062 | 3 cells, noise 0.0072 |
| Meth added to all 13 tracks | n/a | n/a | gain on the full set 0.003 [0.000, 0.006] | gain on the full set 0.001 [-0.006, 0.007] |

#### Other measures, 10 kb, release with consensus plane (loss without the tracks / gain from Meth)

| set | cells | Pearson | SCC | insulation | boundary F1 | loop enrichment | compartment | eigenvector |
|---|---|---|---|---|---|---|---|---|
| minus CTCF | held-out | 0.003 / 0.001 | 0.006 / 0.004 | 0.004 / 0.004 | 0.000 / 0.000 | 0.001 / 0.001 | 0.001 / 0.001 | 0.003 / 0.003 |
| minus CTCF | training all | 0.001 / 0.002 | 0.002 / 0.003 | 0.002 / 0.002 | 0.000 / 0.000 | 0.003 / 0.002 | 0.002 / 0.002 | 0.001 / 0.001 |
| minus cohesin | held-out | 0.002 / 0.005 | 0.003 / 0.007 | 0.001 / 0.003 | 0.000 / 0.000 | 0.002 / 0.003 | -0.001 / 0.000 | -0.001 / 0.002 |
| minus cohesin | training all | 0.007 / 0.006 | 0.008 / 0.007 | 0.005 / 0.004 | 0.000 / 0.000 | 0.006 / 0.003 | 0.006 / 0.005 | 0.003 / 0.003 |
| CTCF only | held-out | 0.021 / 0.005 | 0.018 / 0.007 | 0.010 / 0.005 | 0.000 / 0.000 | 0.028 / 0.003 | 0.019 / 0.004 | 0.008 / 0.001 |
| CTCF only | training all | 0.054 / 0.011 | 0.054 / 0.011 | 0.037 / 0.008 | 0.117 / 0.050 | 0.055 / 0.004 | 0.046 / 0.009 | 0.032 / 0.006 |
| sequence only | held-out | 0.027 / 0.013 | 0.025 / 0.011 | 0.018 / 0.012 | 0.000 / 0.000 | 0.027 / 0.007 | 0.026 / 0.010 | 0.012 / 0.004 |
| sequence only | training all | 0.059 / 0.009 | 0.060 / 0.010 | 0.045 / 0.010 | 0.100 / 0.033 | 0.055 / 0.005 | 0.049 / 0.007 | 0.039 / 0.006 |

#### Other measures, 10 kb, release with plane at 0 (loss without the tracks / gain from Meth)

| set | cells | Pearson | SCC | insulation | boundary F1 | loop enrichment | compartment | eigenvector |
|---|---|---|---|---|---|---|---|---|
| minus CTCF | held-out | 0.043 / 0.004 | 0.073 / 0.012 | 0.059 / 0.004 | 0.000 / 0.000 | 0.058 / -0.001 | 0.030 / 0.003 | 0.026 / 0.008 |
| minus CTCF | training all | 0.028 / 0.004 | 0.036 / 0.005 | 0.031 / 0.001 | 0.145 / 0.000 | 0.040 / 0.003 | 0.020 / 0.004 | 0.014 / 0.004 |
| minus cohesin | held-out | 0.013 / 0.011 | 0.014 / 0.010 | 0.005 / -0.011 | 0.000 / 0.000 | 0.009 / 0.008 | 0.006 / 0.003 | 0.002 / 0.000 |
| minus cohesin | training all | 0.018 / 0.008 | 0.021 / 0.009 | 0.016 / 0.006 | -0.005 / 0.017 | 0.020 / 0.004 | 0.009 / 0.005 | 0.006 / 0.003 |
| CTCF only | held-out | 0.084 / 0.030 | 0.103 / 0.036 | 0.103 / 0.032 | 0.000 / 0.000 | 0.110 / 0.033 | 0.043 / 0.001 | 0.031 / 0.002 |
| CTCF only | training all | 0.150 / 0.031 | 0.158 / 0.030 | 0.126 / 0.023 | 0.145 / 0.000 | 0.158 / 0.024 | 0.099 / 0.017 | 0.078 / 0.016 |
| sequence only | held-out | 0.201 / 0.057 | 0.277 / 0.081 | 0.248 / 0.068 | 0.000 / 0.000 | 0.289 / 0.072 | 0.096 / 0.013 | 0.081 / 0.006 |
| sequence only | training all | 0.246 / 0.042 | 0.303 / 0.044 | 0.264 / 0.039 | 0.145 / 0.000 | 0.315 / 0.039 | 0.145 / 0.023 | 0.110 / 0.017 |

#### Other measures, 25 kb, release with consensus plane (loss without the tracks / gain from Meth)

| set | cells | Pearson | SCC | insulation | boundary F1 | loop enrichment | compartment | eigenvector |
|---|---|---|---|---|---|---|---|---|
| minus CTCF | held-out | 0.001 / 0.001 | 0.003 / 0.006 | 0.004 / 0.002 | 0.000 / 0.000 | -0.003 / 0.000 | 0.002 / 0.003 | 0.002 / 0.000 |
| minus CTCF | training all | 0.001 / 0.003 | -0.000 / 0.002 | 0.002 / 0.001 | 0.000 / 0.000 | 0.002 / 0.002 | 0.002 / 0.003 | 0.001 / 0.001 |
| minus cohesin | held-out | 0.002 / 0.005 | 0.004 / 0.007 | 0.003 / 0.004 | 0.000 / 0.000 | 0.002 / 0.006 | -0.000 / 0.002 | 0.001 / 0.001 |
| minus cohesin | training all | 0.006 / 0.006 | 0.007 / 0.007 | 0.005 / 0.005 | 0.000 / 0.000 | 0.005 / 0.004 | 0.005 / 0.005 | 0.002 / 0.003 |
| CTCF only | held-out | 0.018 / 0.010 | 0.016 / 0.011 | 0.009 / 0.005 | 0.000 / 0.000 | 0.022 / 0.006 | 0.011 / 0.003 | 0.007 / 0.003 |
| CTCF only | training all | 0.052 / 0.011 | 0.048 / 0.008 | 0.033 / 0.006 | 0.033 / -0.067 | 0.050 / 0.005 | 0.042 / 0.010 | 0.027 / 0.005 |
| sequence only | held-out | 0.025 / 0.015 | 0.020 / 0.014 | 0.014 / 0.006 | 0.000 / 0.000 | 0.021 / 0.006 | 0.016 / 0.009 | 0.011 / 0.007 |
| sequence only | training all | 0.056 / 0.008 | 0.053 / 0.009 | 0.041 / 0.009 | 0.033 / -0.017 | 0.052 / 0.007 | 0.046 / 0.008 | 0.030 / 0.005 |

#### Other measures, 25 kb, release with plane at 0 (loss without the tracks / gain from Meth)

| set | cells | Pearson | SCC | insulation | boundary F1 | loop enrichment | compartment | eigenvector |
|---|---|---|---|---|---|---|---|---|
| minus CTCF | held-out | 0.056 / 0.010 | 0.061 / 0.005 | 0.048 / 0.002 | 0.375 / 0.000 | 0.074 / 0.003 | 0.032 / 0.000 | 0.022 / 0.004 |
| minus CTCF | training all | 0.033 / 0.005 | 0.043 / 0.006 | 0.034 / 0.004 | 0.083 / -0.050 | 0.046 / 0.005 | 0.029 / 0.005 | 0.017 / 0.002 |
| minus cohesin | held-out | 0.004 / 0.006 | 0.004 / 0.006 | -0.006 / -0.003 | -0.333 / 0.000 | 0.006 / 0.005 | 0.005 / 0.002 | -0.003 / -0.001 |
| minus cohesin | training all | 0.008 / 0.008 | 0.012 / 0.007 | 0.009 / 0.010 | 0.000 / -0.067 | 0.015 / 0.006 | 0.004 / 0.006 | 0.003 / 0.003 |
| CTCF only | held-out | 0.086 / 0.032 | 0.091 / 0.036 | 0.075 / 0.045 | 0.167 / 0.333 | 0.126 / 0.047 | 0.043 / 0.006 | 0.040 / 0.015 |
| CTCF only | training all | 0.152 / 0.038 | 0.168 / 0.041 | 0.135 / 0.037 | 0.567 / 0.233 | 0.188 / 0.047 | 0.095 / 0.018 | 0.065 / 0.016 |
| sequence only | held-out | 0.223 / 0.082 | 0.255 / 0.091 | 0.237 / 0.079 | 0.500 / 0.000 | 0.307 / 0.098 | 0.107 / 0.024 | 0.078 / 0.017 |
| sequence only | training all | 0.259 / 0.055 | 0.303 / 0.059 | 0.284 / 0.062 | 0.967 / 0.000 | 0.350 / 0.066 | 0.149 / 0.026 | 0.112 / 0.026 |

#### Median Pearson of the 413 regions per cell, 10 kb (without Meth / with Meth)

| cell | 13 tracks, consensus plane | no ChIP, consensus plane | 13 tracks, plane at 0 | no ChIP, plane at 0 |
|---|---|---|---|---|
| gm12878 | 0.935 / 0.936 | 0.851 / 0.885 | 0.834 / 0.841 | 0.567 / 0.653 |
| imr90 | 0.943 / 0.945 | 0.894 / 0.907 | 0.882 / 0.883 | 0.626 / 0.684 |
| hepg2 | 0.917 / 0.918 | 0.864 / 0.893 | 0.826 / 0.830 | 0.590 / 0.683 |
| a549 | 0.935 / 0.935 | 0.910 / 0.920 | 0.846 / 0.848 | 0.595 / 0.682 |
| h1 | 0.924 / 0.926 | 0.895 / 0.896 | 0.829 / 0.830 | 0.675 / 0.672 |
| gm23248 | 0.941 / 0.943 | 0.922 / 0.927 | 0.779 / 0.784 | 0.635 / 0.702 |
| cd14mono | 0.847 / 0.847 | 0.595 / 0.591 | 0.780 / 0.782 | 0.321 / 0.336 |
| hct116 | 0.921 / 0.924 | 0.912 / 0.912 | 0.829 / 0.825 | 0.622 / 0.628 |
| mcf7 | 0.902 / 0.904 | 0.876 / 0.877 | 0.812 / 0.810 | 0.589 / 0.602 |
| hff | 0.951 / 0.950 | 0.906 / 0.908 | 0.887 / 0.881 | 0.626 / 0.621 |
| k562 | 0.901 / 0.902 | 0.854 / 0.879 | 0.783 / 0.783 | 0.557 / 0.635 |
| hmec_deep | 0.907 / 0.909 | 0.899 / 0.900 | 0.766 / 0.773 | 0.591 / 0.627 |

#### Median Pearson of the 413 regions per cell, 25 kb (without Meth / with Meth)

| cell | 13 tracks, consensus plane | no ChIP, consensus plane | 13 tracks, plane at 0 | no ChIP, plane at 0 |
|---|---|---|---|---|
| gm12878 | 0.934 / 0.937 | 0.860 / 0.890 | 0.829 / 0.830 | 0.532 / 0.650 |
| imr90 | 0.953 / 0.955 | 0.906 / 0.916 | 0.869 / 0.871 | 0.622 / 0.685 |
| hepg2 | 0.927 / 0.932 | 0.882 / 0.910 | 0.815 / 0.827 | 0.586 / 0.689 |
| a549 | 0.947 / 0.948 | 0.927 / 0.938 | 0.849 / 0.854 | 0.603 / 0.714 |
| h1 | 0.929 / 0.935 | 0.905 / 0.905 | 0.817 / 0.820 | 0.624 / 0.637 |
| gm23248 | 0.943 / 0.945 | 0.931 / 0.935 | 0.753 / 0.771 | 0.607 / 0.698 |
| cd14mono | 0.886 / 0.883 | 0.622 / 0.622 | 0.817 / 0.817 | 0.322 / 0.349 |
| hct116 | 0.925 / 0.930 | 0.918 / 0.916 | 0.813 / 0.818 | 0.582 / 0.591 |
| mcf7 | 0.926 / 0.929 | 0.907 / 0.908 | 0.812 / 0.812 | 0.592 / 0.600 |
| hff | 0.958 / 0.958 | 0.912 / 0.912 | 0.882 / 0.882 | 0.594 / 0.595 |
| k562 | 0.921 / 0.925 | 0.875 / 0.904 | 0.798 / 0.804 | 0.561 / 0.654 |
| hmec_deep | 0.921 / 0.923 | 0.917 / 0.919 | 0.770 / 0.771 | 0.562 / 0.632 |
