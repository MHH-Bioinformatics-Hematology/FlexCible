# Longer range and sharper boundaries: a design

Two limits of the current model, and the ideas raised to get past them.

The model predicts a Hi-C window from 1D marks along the diagonal. Its span is
the window (1024 bins), so at 1 kb that is a ~1 Mb receptive field: anything
longer-range than 1 Mb is invisible to it, and the fine 1 kb map is sparse and
noisy, so TAD boundaries are hard to read off it directly. Inter-chromosomal
contacts are never modelled at all: every window sits on one chromosome's
diagonal.

Three tracks, cheapest first.

## 1. An insulation / boundary head (near-term)

Rather than reading boundaries off a noisy predicted map, predict the boundary
structure explicitly, as the user put it: predict a smooth gradient, then let it
form the boundaries.

- **Target.** From each target window compute the insulation score (the standard
  sliding-diamond mean of contacts across each bin), a 1D profile the length of
  the window. Its local minima are TAD boundaries; the sign change of its
  derivative (the gradient) marks them. Optionally also a 0/1 boundary label per
  bin from the minima.
- **Head.** Add a 1D output head to the U-Net: project the decoder bottleneck (or
  a row-reduction of the 2D output) to a length-`W` vector. One head for the
  insulation profile (regression), optionally a second for boundary probability
  (per-bin BCE).
- **Loss.** `L = L_hic + a * MSE(insulation) + b * BCE(boundary)`. The insulation
  term is smooth and dense, so it trains stably even where the 2 kb/1 kb map is
  sparse, and it directly sharpens where the model places domain edges.
- **Where.** Target computed in `precompute.py` (or on the fly from the cached
  target window); head added in `model.py`; loss assembled in `train.py` next to
  the existing capture aux loss (same `present`/masking pattern).
- **Payoff.** Boundaries come out of the model directly and are usable without
  post-hoc calling; the insulation gradient is a cleaner training signal than the
  raw pixels. Small, self-contained change; no new data.

## 2. Coarse-to-fine cascade (longer range)

To see past the 1 Mb window, give the fine model a coarse view of a much larger
span as context. This is the multi-scale idea (cf. Orca): a coarse field carries
the long-range/compartment structure the fine window cannot reach.

- **Stage A (coarse).** Train a coarse model at 25 kb (the existing
  `multiome_25kb.yaml`) but over a large window, e.g. 512 bins = ~13 Mb,
  predicting the smooth long-range field (log1p or O/E). Compartments and
  long-range contacts live here.
- **Stage B (fine).** For a 1 kb window, run Stage A over the enclosing ~13 Mb
  region, crop the coarse prediction to the fine window's span, upsample it to
  1 kb, and feed it to the fine model as an extra input plane (concatenated with
  the marks/sequence/motif planes). The fine model then predicts detail
  *conditioned on* long-range context.
- **Implementation.** Add one `context` input channel to the fine model. Feed it
  either offline (a precompute pass writes the coarse prediction as a context
  array/cool per region) or online (call the frozen Stage A inside the fine
  window loader). Train two-stage: fit coarse, freeze, then fine; later optionally
  fine-tune jointly.
- **Payoff.** Effective range becomes the coarse span (tens of Mb), not the fine
  window, while keeping 1 kb detail. Bigger change than the boundary head; needs
  a coarse checkpoint and a context channel, but reuses existing configs.

## 3. Inter-chromosomal, compartment scale (separate model)

Trans contacts have no distance-decay backbone, are ~100x sparser than cis, and
are governed by A/B compartments and nuclear positioning rather than local loops.
Predicting them at 1 kb is not realistic; the honest target is compartment-scale
trans.

- **Target.** Balanced trans O/E between chromosome pairs at 100 kb - 1 Mb.
- **Model.** Predict a per-bin compartment embedding from compartment-associated
  marks (H3K27ac, H3K9me3, lamina/LAD signal if available); approximate the trans
  matrix between two chromosomes as the outer product / bilinear form of their
  embeddings. This captures "like associates with like" (A-A, B-B) without trying
  to place individual trans pixels.
- **Scope.** A distinct coarse model and target, not a plane bolted onto the fine
  cis model. Do it after the cis work.

## Suggested order

1. Boundary/insulation head on the existing 1 kb (and 2 kb) cis model - smallest,
   no new data, directly answers "gradient then boundaries".
2. Coarse-to-fine context channel - the real fix for long range.
3. Compartment-scale trans model - separate effort, coarse resolution only.
