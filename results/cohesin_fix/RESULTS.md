# Cohesin-fraction input and cohesin-loss augmentation: results (registry section 55)

Models (single seed, 25 and 10 kb, same recipe as all12deg_refres_drop):

* deg = all12deg_refres_drop (no fix)
* cohfix = all12cohfix_refres_drop (cohesin-fraction input row, perturbation augmentation, hct116_auxin on the contact scale of hct116)
* cohfixna = all12cohfixna_refres_drop (ablation: row and shared contact scale, no augmentation)

Hosts (claims): cohfix 25 kb and both cohfixna runs trained on this machine, cohfix 10 kb trained on the
remote machine (1.30 h); the perturbation, washout and AML predict steps ran on the remote machine and wrote
to the share; all scoring, the diagnosis (cohfixna and deg 25 kb were rerun here, because the queue's
diagnose of cohfixna failed on a cache-name fallback; `--cache mc_all12cohfix_refres_drop_<k>kb` fixes it) and
`aml_eval.py --stage score` ran here on the CPU. Tables: `results_*.csv` in this folder
(`scripts/cohesin_fix_results.py`), the underlying `results/perturb_validate/perturb_summary.csv`,
`results/washout/washout_summary.csv`, `results/aml/*.csv`, `diagnose_<run>.csv`.

Evaluation sets: RAD21 degron on the even chromosomes of HCT116 (never trained under auxin; 255 loops at
10 kb, 260 at 25 kb); CTCF degron fully held out (all chromosomes); washout and AML on the full 413 regions.
"Fraction" is the value of the cohesin-fraction row (1 intact, 0.26 the measured 6 h auxin state, 0 none).

## Summary table (10 kb / 25 kb)

| criterion | measured | deg | cohfix | cohfixna | verdict (cohfix) |
|---|---|---|---|---|---|
| (a) intact, 12 training cells, median Pearson (delta vs deg) | | 0.9237 / 0.9358 | 0.9279 / 0.9336 (+0.0042 / -0.0022) | 0.9221 / 0.9308 (-0.0016 / -0.0050) | pass (limit -0.005) |
| (a) intact, 5 held-out cells (delta vs deg) | | 0.9039 / 0.9037 | 0.9060 / 0.9133 (+0.0021 / +0.0096) | 0.8775 / 0.9060 (-0.0264 / +0.0023) | pass |
| (a) 413-region test-window median (delta vs deg) | | 0.9237 / 0.9457 | 0.9234 / 0.9435 (-0.0003 / -0.0022) | 0.9218 / 0.9441 (-0.0019 / -0.0016) | pass |
| (b) spread of the predicted change, wt at fraction 0, 184 / 174 windows (% of measured spread) | 0.302 / 0.260 | absent RAD21: 0.034 / 0.037 (11% / 14%) | 0.292 / 0.220 (97% / 84%) | 0.176 / 0.128 (58% / 49%) | pass (threshold 10%; moved to the measured 0.2 to 0.3) |
| (b) same at fraction 0.26 (measured state) | | | 0.217 / 0.170 (72% / 65%) | 0.212 / 0.158 (70% / 61%) | |
| (b) minus_RAD21_f0, loop change (median log O/E, even chr) | -0.726 / -0.477 | +0.027 / +0.010 (absent) | -1.135 / -0.489 | -0.838 / -0.528 | direction fixed; 10 kb overshoots 1.6x |
| (b) minus_RAD21_f0.5, loop change | -0.726 / -0.477 | | -0.611 / -0.281 | -0.698 / -0.360 | |
| (b) measured_chip (fraction 0.26), loop change | -0.726 / -0.477 | -0.488 / -0.201 | -0.652 / -0.341 | -0.628 / -0.397 | pass (closer to measured than deg) |
| (b) loops weaker, f0 / f0.5 / measured_chip (measured 98.8% / 98.9%) | | 20.8% absent; 98.4% / 95.8% measured_chip | 100% / 100% / 100%; 98.5% / 99.6% / 98.9% | 99.6% / 100% / 100%; 98.5% / 98.1% / 99.6% | pass |
| (b) change-map Pearson, minus_RAD21_f0 | | -0.249 / 0.245 (absent) | 0.784 / 0.852 | 0.821 / 0.864 | 10 kb marginal fail against 0.803 (strict), 25 kb pass |
| (b) change-map Pearson, minus_RAD21_f0.5 | | | 0.792 / 0.846 | 0.783 / 0.806 | |
| (b) change-map Pearson, measured_chip (deg 0.803 / 0.842) | | 0.803 / 0.842 | 0.839 / 0.888 | 0.853 / 0.891 | pass |
| (b) change-map Pearson, diagnosis windows, wt_f0 / absent_f0 | | 0.505 / 0.547 (absent) | 0.822 / 0.886; 0.807 / 0.866 | 0.803 / 0.843; 0.807 / 0.852 | |
| (b) minus_RAD21 (ChIP absent, fraction row 1), Pearson / loops weaker | | -0.249 / 20.8%; 0.245 / 33.1% | -0.299 / 16.9%; -0.066 / 26.2% | 0.166 / 55.3%; 0.058 / 21.2% | unchanged by design (see below) |
| (c) washout, loops weaker with auxin track (measured 98.8% / 98.5%) | | 98.4% / 98.8% | 100% / 98.5% | 98.8% / 100% | pass |
| (c) washout, median change at auxin (measured -0.726 / -0.477) | | -0.360 / -0.119 | -0.674 / -0.323 | -0.614 / -0.403 | pass (monotone, closer to measured) |
| (c) washout recovery at f 0.25 / 0.5 / 0.75 | | 0.43 / 0.77 / 0.93 | 0.32 / 0.66 / 0.87 | 0.13 / 0.58 / 0.86 (10 kb) | monotone |
| (c) CTCF degron held out, change Pearson (floor 0.40) | | 0.396 / 0.517 | 0.403 / 0.402 | 0.433 / 0.602 | pass at the floor; 25 kb lost 0.115 against deg |
| (c) CTCF degron, loops weaker | | 94.6% / 82.9% | 83.3% / 66.5% | 91.2% / 92.7% | |
| (d) U-937 STAG2 KO, loop change (measured +0.080 / -0.033), ko_f0.5 | | absent/half: +0.070 / +0.098; -0.004 / -0.003 | -0.231 / -0.315 | -0.139 / -0.175 | wrong (sign and size) |
| (d) STAG2 KO, ko_f0.75 loop change | | | -0.109 / -0.115 | -0.026 / -0.028 | wrong |
| (d) STAG2 KO, ko_absent_f0 loop change | | | -0.425 / -0.502 | -0.325 / -0.513 | wrong |
| (d) STAG2 KO, change-map Pearson (clone-pair ceiling 0.25 at 10 kb), ko_f0.5 | | 0.083 (half) / 0.170 | 0.200 / 0.261 | 0.214 / 0.310 | map pattern up, size wrong |
| (d) STAG2 KO, ko_half (RAD21/SMC3 halved, fraction untouched) | | -0.004 / -0.003 | -0.013 / -0.026 | -0.006 / -0.010 | no response |
| (e) AML patient-line maps, MOLM-13 median Pearson | | 0.848 / 0.900 | 0.882 / 0.912 | 0.880 / 0.914 | pass |
| (e) U-937 wild type clone 1, median Pearson | | 0.861 / 0.879 | 0.868 / 0.891 | 0.865 / 0.890 | pass |
| (e) U-937 KO clone 1 with the wild-type input | | 0.762 / 0.835 | 0.781 / 0.833 | 0.766 / 0.834 | within 0.02 |

Deg values for the "RAD21 absent" rows are the registry-section-25 absent-RAD21 input (no fraction row exists).
HFF cohesin (all12hffcoh): not part of this fix. cohfix and cohfixna set the fraction to 1 for HFF and receive no
HFF cohesin ChIP, so the SMC1A-substitution question of section 25 is not re-tested by these runs.

## Verdict per criterion

* (a) Intact accuracy: pass for cohfix in all eight comparisons (largest loss -0.0022, at 25 kb on the 12 training
  cells; held-out cells improve by +0.002 and +0.010). The ablation loses 0.0264 on the 5 held-out cells at 10 kb
  and exactly 0.0050 at 25 kb on the training cells.
* (b) RAD21 change maps: pass. A fraction 0 given through the row yields a predicted change with 97% (10 kb) and 84%
  (25 kb) of the measured spread (deg: 11% and 14% for an absent track), weakens 100% and 98.5% of the loops
  (deg absent: 20.8%), and the loop change moves from +0.03 to -1.14 (10 kb) and -0.49 (25 kb) against the measured
  -0.73 and -0.48. At the matched fraction 0.26 the loop change is -0.65 (deg with the real auxin track: -0.49) and the
  change-map Pearson 0.84 and 0.89 (deg 0.80 and 0.84). One strict miss: the pre-registered "minus_RAD21_f0 at least
  0.80" is 0.784 at 10 kb (0.852 at 25 kb); fraction 0 is a deeper depletion than the measured 74% loss, so the matched
  comparison (measured_chip, 0.839) is the like-for-like one, but the number is below the threshold as written.
* (b) RAD21 absent as ChIP only (`minus_RAD21`, fraction row 1): still wrong, by construction. With the row at 1 the
  model reads "cohesin intact" and the change is +0.04 at loops (16.9% weaker). The measured auxin RAD21 track
  alone (row 1) gives Pearson -0.27 against 0.71 for deg (diagnosis `aux_RAD21_only`): the response now runs through
  the row, and the RAD21 ChIP shape carries almost none of it. A user has to state the cohesin fraction.
* (c) Washout: pass. 100% / 98.5% of loops weaker with the auxin track, median change -0.67 / -0.32 (deg -0.36 / -0.12,
  measured -0.73 / -0.48), recovery monotone in the fraction (0.32, 0.66, 0.87 at 0.25, 0.5, 0.75 at 10 kb);
  the fractions matching the measured 20 / 40 / 60 min states are 0.15 / 0.38 / 0.50 (deg 0.11 / 0.30 / 0.42).
  CTCF degron: 0.403 / 0.402 (floor 0.40), passes at the floor, with 25 kb 0.115 below deg (0.517) and loops weaker
  66.5% against 82.9%. The ablation keeps (0.433 / 0.602).
* (d) STAG2 knockout: not solved. Measured loops change +0.080 at 10 kb (short -0.105, long +0.181), the
  fraction-conditioned predictions are -0.23 (fraction 0.5), -0.11 (0.75), -0.43 (0 with absent track) at 10 kb and
  -0.32, -0.12, -0.50 at 25 kb against -0.033: a RAD21-degron-like weakening of the wrong size and sign. The
  region-level change-map Pearson rises from 0.08 to 0.10 (deg) to 0.20 (10 kb, ceiling 0.25) and 0.26 (25 kb), and the
  per-loop change correlation is 0.60 at 10 kb for ko_f0.5; the pattern of the change is partly captured (a global
  low-pass loss pattern shared with the degron) while the loop gain of the measured knockout is not. Halving
  RAD21/SMC3 in the ChIP without the row changes nothing (-0.01).
* (e) AML evaluation rows: pass. MOLM-13 +0.034 (10 kb) and +0.012 (25 kb) over deg, U-937 wild type +0.007 / +0.011;
  U-937 KO clone 1 with the wild-type input within 0.02 of deg (0.78 / 0.83).
* (f) Ablation. Row plus shared contact scale (cohfixna) already gives the change map at the measured state (Pearson
  0.85 / 0.89 with fraction 0.26; spread 70% / 61%), because the row separates the 6 h auxin sample from every other
  sample (two values: 1 and 0.26). It does not extrapolate: the spread at fraction 0 (58%, 49%) is below that at
  0.26 at 10 kb and 25 kb (non-monotone), the washout recovery at fraction 0.25 is 0.13 at 10 kb against the linear 0.26,
  and the held-out intact accuracy at 10 kb falls by 0.026. The augmentation supplies the graded response over
  fractions 0 to 0.75 (spread 97 / 72 / 42 / 17% at 0 / 0.26 / 0.5 / 0.75 at 10 kb) and the intact held-out
  accuracy; it costs CTCF-degron generalisation at 25 kb (0.402 against 0.602). The change-map Pearson of the
  ablation at fraction 0 and 0.5 (0.821, 0.783 at 10 kb) matches the full model's, so the correlation comes mostly
  from the row and the scale and the amplitude from the augmentation.

## Honest verdict

The fix solves the RAD21 degron when the cohesin fraction is stated: the amplitude (spread 11% to 97% at 10 kb),
the loop direction (20.8% to 100% weaker) and the washout all follow the row and agree with the measured values,
without lowering the intact accuracy. It does not solve "absent RAD21" as a ChIP statement: a RAD21/SMC3 track
removed or shrunk with the row at 1 still gives no response, and the measured auxin ChIP alone no longer drives it.
STAG2 knockout stays wrong in the size and sign of the loop change (predicted weakening, measured slight gain at
10 kb), although the correlation of its change map (0.20) approaches the clone-pair ceiling (0.25) at 10 kb; a total-cohesin
fraction describes the degron, not the loss of one cohesin variant.

## Limitations

* One seed per model; differences of 0.005 in intact accuracy and of a few percentage points in loop counts are
  inside the unmeasured seed variation (the held-out cell medians rest on 5 cells).
* One degron cell line (HCT116, RAD21-AID, 6 h): the operator, the fraction 0.26 and every RAD21 score come from the
  same cell line; the auxin sample trains on the odd chromosomes and is scored on the even ones.
* The depletion operator is fitted on one cell's pair of maps and applied to the other training cells' windows
  (cell-type dependence of the change is assumed shared), and the augmented windows are synthetic: the target
  change is the operator's, scaled linearly in (1 - fraction); fractions between 0 and 0.26 have no measured anchor
  (fraction 0 overshoots the measured loop change at 10 kb by 1.6x).
* The U-937 knockout, CTCF degron and washout have no training data in this fix; their agreement is a test of the
  row, and the knockout has one clone pair, so its ceiling (0.25) is low.
* Washout predictions use an interpolated fraction (0.26 + f (1 - 0.26)); no matched ChIP exists.
