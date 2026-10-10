# Why the cohesin-loss response stays wrong (diagnosis, 10 kb, models without the fix)

Data: perturb_validate.py / washout.py / aml_eval.py tables of sections 21, 25 and 32, plus
`scripts/cohesin_fix_diagnose.py` (184 Akita test windows of the even chromosomes of HCT116, never
trained under auxin; the cached untreated ChIP is fed to the trained model with the RAD21 row changed;
medians over windows; tables `diagnose_all12deg_refres_drop_10kb.csv`,
`diagnose_all12cohunk_refres_drop_10kb.csv`) and `scripts/cohesin_fix_measured.py` (413 regions).

## 1. What is wrong, by feature (all12deg_refres_drop, 10 kb, even chromosomes)

| quantity | measured RAD21 degron | model, measured auxin ChIP | model, RAD21 absent |
|---|---|---|---|
| change map Pearson (median region) | | 0.80 | -0.25 |
| loop strength change at 536 loops (median log O/E) | -0.73 | -0.49 | +0.03 |
| loops weaker | 99.4% | 98.4% | 20.8% |
| loop change by distance (<=100 kb / >500 kb) | -0.52 / -0.97 | -0.35 / -0.68 | +0.03 / +0.02 |
| boundary insulation change | +0.062 (weaker) | +0.042 | -0.012 |
| per-distance Pearson of the change map | | 0.64 to 0.76 | -0.07 to -0.28 |
| spread (sd) of the change map, 184 windows | 0.319 | 0.202 | 0.034 |

The failure is the amplitude of the whole change, not one feature: with an absent RAD21 the predicted
change has 11% of the measured spread; loops, borders and the distance profile all move by 0 to 5% of
the measured amount, and the small residual has the wrong sign at loops (+0.03). The window-level
Pearson of an absent RAD21 is positive (0.51) only because its tiny change has the right low-pass
shape. The measured change is mostly low-pass block structure (about 80% of the variance), the distance
profile carries 13% and the loop/stripe scale 5%.

## 2. What the model reads

Same windows, RAD21 changed alone (Pearson / sd of the predicted change; measured sd 0.319):

| RAD21 given as | Pearson | sd |
|---|---|---|
| measured auxin track | 0.71 | 0.158 (all auxin tracks: 0.202) |
| contrast around the window median x0.26 (peaks shrink, background stays) | 0.70 | 0.116 |
| contrast x0 (flat) | 0.71 | 0.237 |
| every value x0.26 (peaks and background shrink) | 0.08 | 0.016 |
| every value x0 | 0.51 | 0.034 |
| absent code (the dropout code) | 0.51 | 0.034 |
| auxin CTCF only / auxin histones only | -0.10 / 0.15 | 0.004 / 0.017 |

1. The model reads the peak-to-background contrast of the RAD21 track (the shape of the 6 h auxin
   track: peaks 2.5 against 7.2, median 0.73 against 0.52), not its overall level, and not the
   consensus (the same leave-one-out consensus enters both conditions) and not the CTCF or histone tracks.
2. An absent or scaled-down track is outside what the auxin sample teaches. With track dropout (uniform
   count of kept tracks, 50% of the windows lose RAD21) an absent RAD21 occurs in every cell with an
   intact map and means "not measured"; five of the 12 cells have no cohesin ChIP at all. The only example
   where low cohesin goes with weak loops is one sample (hct116_auxin, odd chromosomes). The model learns
   "absent = unknown, ignore", and learns the auxin response only for that track shape.
3. The cohesin-unknown indicator (cohunk) cannot change this: indicator = 1 is paired with intact loops
   in every cell but hct116_auxin, so the response to an absent RAD21 stays at 12% of the measured spread
   (0.042; indicator forced to 0: 0.070).
4. The degron sample is encoded with its own contact scale (1.316 against 1.914 of the untreated
   map, fitted on the clip quantile of |log O/E|), which stretches the weaker auxin map back; decoded with
   its own scale the predicted spread is 0.254 instead of 0.202 (+26%). A smaller effect.

## 3. Two different failing cases

* RAD21 degron / absent RAD21 input: a ChIP input can in principle encode it. Total cohesin falls to about
  26% (RAD21 enrichment above the median of the top 1% untreated bins, auxin over untreated, odd
  chromosomes; 0.34 as a genome-wide sum), CTCF stays (0.96). The map change is a function of the
  control map: an operator fitted on the odd chromosomes of the HCT116 pair predicts the even-chromosome
  change from the untreated map alone at Pearson 0.83 (10 kb) and 0.88 (25 kb) (operator_*.json).
* U-937 STAG2 knockout: partial cohesin loss with STAG1 compensation. Measured loop change +0.08
  (short -0.105, long +0.181), boundaries +0.087, clone-pair ceiling of the change map 0.25 (10 kb). Its
  change map correlates with the RAD21-degron change at 0.14 only (region median), its spread (0.22) is
  mostly noise, and the mean change by distance has the opposite sign (+0.018 against -0.021 at 100 to
  250 kb). A scalar for total cohesin spans the degron direction, not this one; RAD21/SMC3 ChIP of the
  wild type cannot encode the loss of STAG2-specific cohesin (it needs STAG1 and STAG2 ChIP, or RAD21 of the
  knockout). The fix is not expected to solve it; it is scored with fractions 0.5 and 0.75.

## 4. Consequence for the fix

The response has to be taught across cell contexts with the cohesin state given explicitly, and "absent
ChIP" has to be separated from "less cohesin". See registry section 55: a cohesin-fraction input row
(biology, 1 for every sample incl. those without cohesin ChIP) plus augmentation with synthetic depleted
windows built by a degron operator, and the degron pair on one contact scale.
