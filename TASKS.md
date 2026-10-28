# Task board

I supervise and report; each workstream below runs as its own agent so they do
not interleave. GPUs are assigned explicitly so two tasks never contend.

## State of the record (verified, 2026-09-01)

| measurement | value |
|---|---|
| ours vs AkitaV1, Akita's own test target, their metric | **0.656** median (68% of 222 windows) |
| AkitaV1 (recorded) | 0.556 |
| Hi-cGAN (recorded) | 0.220 |
| distance-only floor on that target | 0.244 |
| 25 kb SCC, `l1+mse+scc`, 3 seeds | 0.9553 vs 0.9520 baseline |

Trained 30 min, 2 GPUs, bf16 + torch.compile, on Akita's published TRAIN
matrix. Cache at `cache/akita_native` is built and reusable.

## Settled, do not redo

- Use Akita's published matrices verbatim (`/mnt/project/wolffjoa/
  hicgan_experimental/data/4Dnucleosome/akita_published_GM12878_{train,test}_2048bp.cool`).
  No local renormalisation. Every attempt at one made things worse.
- bf16 + torch.compile + DDP: 2.8x, accuracy-neutral. Adopted.
- Train on the metric ALONGSIDE l1+mse in count space (`l1+mse+scc`, +0.0033).
  `dcorr` LOSES in obs/exp space. `scc` alone and `ssim` both lose.
- Absolute position is not a feature anywhere: it breaks transfer.

## Open tasks

| # | task | owner | GPU |
|---|---|---|---|
| 1 | pairwise (no-window) approach: fix, test, ablate | agent | GPU1 |
| 2 | longer Akita-native training, does 0.656 improve | agent | GPU0 |
| 3 | Akita's other 4 cell types from their published data | agent | none |
| 4 | manuscript figure refinements | me, on request | none |

## Known traps (cost hours already)

- `predict.py` clamped predictions at 0, deleting the negative half of
  Akita-space output. Fixed; regression test in `tests/test_predict_space.py`.
- Scoring against the whole-genome target instead of Akita's test-span target
  inflates everything: distance-only scores 0.74 there, 0.24 on Akita's.
- `max_zero_fraction` does not filter Akita-space matrices (nothing is exactly
  zero after smoothing). Use `min_mappable_fraction`.
- Installing packages can drag numpy to 2.x and break torch/pyBigWig. Pin
  `numpy<2`.

## Feature-level score (`flexcible/bioscore.py`)

Pixel Pearson is not the score. `bioscore` reports insulation-profile
correlation, boundary F1, loop F1 and APA separately, plus a combined number,
and every run carries the distance-only and region-swap baselines. Pipeline:

    scripts/bioscore_predict.py     one cached forward pass per test region
    scripts/bioscore_tune_loops.py  loop-caller parameters, chosen on CTCF
    scripts/bioscore_akita.py       Parts 1-2: three models + baselines, strata
    scripts/bioscore_depth.py       Part 3: down-sampling, prediction vs depth
    scripts/bioscore_report.py      joins them into the reported tables

More traps found while building it:

- The loop caller must run on COARSENED bins. On native 2 kb its calls carry
  no CTCF signal at all (anchor AUROC 0.50, i.e. Poisson noise); at 8x
  coarsening it reaches 0.71 at the loop density Rao et al. report.
- A fixed loop threshold measures how SHARP a model is, not where it puts its
  loops. Match the prediction's top-k peaks to the measured set, and report
  the fixed-threshold count separately as the sharpness diagnostic it is.
- In a down-sampling experiment the shallow map must be DISJOINT from the
  reference (S ~ Binom(C,f), reference C-S) and balanced independently, or it
  gets credit for reproducing its own noise and its own bias vector.
- Compare against a SMOOTHED shallow map too. Raw is a strawman, and the
  smoothing changes the crossover depth by about an order of magnitude.
