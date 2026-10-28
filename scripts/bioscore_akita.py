"""Parts 1 and 2: score three models on identical pixels, then stratify.

Part 1 -- the feature-level score
    Every candidate is scored on Akita's own 448 x 448, 2048 bp test rasters,
    against Akita's own published GM12878 target, so the comparison is on
    identical pixels and on the target the published numbers were computed
    against. Akita's and Hi-cGAN's maps are read from
    akita_testset_rasters.npz; nothing of theirs is re-run. Ours comes from
    one cached forward pass per region (scripts/bioscore_predict.py) and is
    the central 448 bins of the model's own 512-bin, 2048 bp window -- the
    same grid, no resampling.

    Three trivial baselines are scored on exactly the same pixels, every time:

      distance  the mean of the measured data at each genomic separation,
                averaged over the whole test set. Knows decay, knows nothing
                else. Any metric it does not lose on is broken.
      swap      the measured target of a DIFFERENT region. A map that looks
                exactly like real Hi-C, in the wrong place. Catches a metric
                that rewards "looks plausible".
      blur2     the measured target itself, Gaussian-blurred (sigma 2 bins).
                Not a baseline to beat but a ceiling to read against: it is
                the score of a perfect model that is merely out of focus.

Part 2 -- stratification that cannot be circular
    Regions are classified from the INPUT TRACKS ONLY: the raw per-region mean
    of the 13 ChIP tracks, cached by bioscore_predict.py, z-scored across
    regions and averaged into one "how much signal is here at all" index. No
    model score, and no property of the target, enters the classification --
    selecting regions by their score would manufacture any result wanted.
    Tertiles of that index give signal-poor / mid / signal-rich, and both the
    all-region and the stratified numbers are always printed side by side
    with the region count of each class.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible import bioscore as bs                              # noqa: E402
from flexcible.utils import get_logger                            # noqa: E402

log = get_logger("bioscore-akita")

REPORT = ["n_regions", "pearson_mean", "pearson_median", "insulation_pearson",
          "boundary_precision", "boundary_recall", "boundary_f1",
          "loop_n_true", "loop_f1", "loop_n_invented",
          "loopthr_n_pred", "loopthr_precision", "loopthr_n_invented",
          "apa_true", "apa_pred", "apa_ratio",
          "anchor_spearman", "anchor_auroc",
          "tad_score", "loop_score", "bioscore"]


def _score_one(args):
    k, true_v, cand_v, p, side, off = args
    T = bs.dense_from_triu(true_v, side, off)
    P = cand_v if cand_v.ndim == 2 else bs.dense_from_triu(cand_v, side, off)
    return k, bs.region_metrics(T, P, p, seed_offset=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/bioscore")
    ap.add_argument("--preds", default="runs/bioscore/our_preds.npz")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default="runs/bioscore/akita_bioscore")
    a = ap.parse_args()

    D = Path(a.dir)
    target = np.load(D / "rast_target.npy")
    akita = np.load(D / "rast_akita.npy")
    hicgan = np.load(D / "rast_hicgan.npy")
    regions = [str(r) for r in np.load(D / "rast_regions.npy", allow_pickle=True)]
    side, off = (int(x) for x in np.load(D / "rast_meta.npy"))

    P = np.load(a.preds, allow_pickle=True)
    our_regions = [str(r) for r in P["regions"]]
    our_idx = {r: i for i, r in enumerate(our_regions)}
    ours_native = P["ours_native"]
    crop = (int(P["window"]) - side) // 2
    track_names = [str(t) for t in P["track_names"]]
    track_means = P["track_means"]

    idx = [k for k in range(0, len(regions), a.stride) if regions[k] in our_idx]
    if a.limit:
        idx = idx[: a.limit]
    log.info("%d regions scored on identical %dx%d pixels", len(idx), side, side)

    p = bs.BioParams()

    # --- candidates, all on the same 448-bin grid -------------------------- #
    dense_targets = np.stack([bs.dense_from_triu(target[k], side, off)
                              for k in idx])
    dist = bs.distance_baseline(dense_targets, side)
    from scipy.ndimage import gaussian_filter

    cands: dict[str, list] = {"ours": [], "akita": [], "hicgan": [],
                              "distance": [], "swap": [], "blur2": []}
    for n, k in enumerate(idx):
        cands["ours"].append(
            ours_native[our_idx[regions[k]]][crop:crop + side,
                                             crop:crop + side].astype(np.float64))
        cands["akita"].append(bs.dense_from_triu(akita[k], side, off))
        cands["hicgan"].append(bs.dense_from_triu(hicgan[k], side, off))
        cands["distance"].append(dist)
        cands["swap"].append(dense_targets[(n + len(idx) // 2) % len(idx)])
        cands["blur2"].append(gaussian_filter(dense_targets[n], 2.0))

    # --- score ------------------------------------------------------------- #
    from concurrent.futures import ProcessPoolExecutor

    per_region: dict[str, dict[int, bs.RegionResult]] = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for lab, maps in cands.items():
            jobs = [(k, target[k], maps[n], p, side, off)
                    for n, k in enumerate(idx)]
            per_region[lab] = dict(ex.map(_score_one, jobs, chunksize=4))
            log.info("scored %s", lab)

    # --- Part 1 table ------------------------------------------------------ #
    table = {lab: bs.pool([per_region[lab][k] for k in idx], p)
             for lab in cands}
    T1 = pd.DataFrame({lab: {m: table[lab].get(m) for m in REPORT}
                       for lab in ["ours", "akita", "hicgan",
                                   "distance", "swap", "blur2"]})
    print("\n=== PART 1: feature-level score, Akita test rasters "
          f"({len(idx)} regions, identical pixels) ===")
    print(T1.round(4).to_string())

    # --- Part 2 stratification, from input tracks only --------------------- #
    tm = np.array([track_means[our_idx[regions[k]]] for k in idx], float)
    lt = np.log1p(np.maximum(tm, 0))
    z = (lt - np.nanmean(lt, 0)) / np.maximum(np.nanstd(lt, 0), 1e-9)
    signal_index = np.nanmean(z, axis=1)
    q = np.nanquantile(signal_index, [1 / 3, 2 / 3])
    klass = np.where(signal_index <= q[0], "signal-poor",
                     np.where(signal_index <= q[1], "mid", "signal-rich"))

    strat = []
    for cls in ["signal-poor", "mid", "signal-rich"]:
        sel = [k for k, c in zip(idx, klass) if c == cls]
        for lab in ["ours", "akita", "hicgan", "distance", "swap"]:
            row = bs.pool([per_region[lab][k] for k in sel], p)
            strat.append({"class": cls, "model": lab,
                          **{m: row.get(m) for m in REPORT}})
    S = pd.DataFrame(strat)
    print("\n=== PART 2: stratified by INPUT-TRACK signal only ===")
    print("class sizes: " + ", ".join(
        f"{c} n={int((klass == c).sum())}"
        for c in ["signal-poor", "mid", "signal-rich"]))
    show = ["class", "model", "pearson_mean", "insulation_pearson",
            "boundary_f1", "loop_f1", "apa_ratio", "anchor_auroc", "bioscore"]
    print(S[show].round(4).to_string(index=False))

    # how steeply does each model depend on the input signal?
    print("\ncorrelation of the per-region score with the input-signal index "
          "(the asymmetry: our advantage comes from reading the tracks, so we "
          "have the most to lose where they are uninformative)")
    slopes = []
    for lab in ["ours", "akita", "hicgan"]:
        pr = np.array([per_region[lab][k].pearson for k in idx], float)
        ib = np.array([per_region[lab][k].insulation_pearson for k in idx], float)
        ok = np.isfinite(pr) & np.isfinite(signal_index)
        ok2 = np.isfinite(ib) & np.isfinite(signal_index)
        slopes.append({"model": lab,
                       "r(pearson, signal)": float(np.corrcoef(
                           pr[ok], signal_index[ok])[0, 1]),
                       "r(insulation, signal)": float(np.corrcoef(
                           ib[ok2], signal_index[ok2])[0, 1])})
    print(pd.DataFrame(slopes).round(3).to_string(index=False))

    # --- write everything out ---------------------------------------------- #
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    T1.to_csv(out.with_name(out.name + "_table.csv"))
    S.to_csv(out.with_name(out.name + "_stratified.csv"), index=False)
    pd.DataFrame(slopes).to_csv(out.with_name(out.name + "_slopes.csv"),
                                index=False)
    rows = []
    for n, k in enumerate(idx):
        r = {"region": regions[k], "signal_index": float(signal_index[n]),
             "class": klass[n]}
        for t, name in enumerate(track_names):
            r[name] = float(tm[n, t])
        for lab in cands:
            rr = per_region[lab][k]
            r[f"{lab}_pearson"] = rr.pearson
            r[f"{lab}_insulation"] = rr.insulation_pearson
            r[f"{lab}_loop_tp"] = rr.l_tp_true
            r[f"{lab}_loop_n"] = rr.l_n_true
        rows.append(r)
    pd.DataFrame(rows).to_csv(out.with_name(out.name + "_per_region.csv"),
                              index=False)
    (out.with_name(out.name + "_params.json")).write_text(
        json.dumps(p.as_dict(), indent=2))

    # --- the APA panels, the one picture a loop metric owes the reader ----- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = ["akita_target (measured)", "ours", "akita", "hicgan",
            "distance", "swap"]
    stacks = []
    n_anch = sum(per_region["ours"][k].apa_n for k in idx)
    stacks.append(sum(per_region["ours"][k].apa_true_sum for k in idx) / n_anch)
    for lab in ["ours", "akita", "hicgan", "distance", "swap"]:
        stacks.append(sum(per_region[lab][k].apa_pred_sum for k in idx)
                      / max(1, sum(per_region[lab][k].apa_n for k in idx)))
    fig, axes = plt.subplots(1, len(labs), figsize=(2.5 * len(labs), 3.0))
    vmax = float(np.nanmax(stacks[0]))
    for ax, lab, A in zip(axes, labs, stacks):
        ax.imshow(A, cmap="RdYlBu_r", vmin=0.6, vmax=vmax,
                  interpolation="nearest")
        c = A.shape[0] // 2
        ax.set_title(f"{lab}\nP2LL {A[c, c] / A[-p.apa_corner:, :p.apa_corner].mean():.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Aggregate peak analysis at {n_anch} loop anchors called in the "
        f"MEASURED map ({len(idx)} Akita test regions).\n"
        f"Anchors come from the measurement alone, so no model influences the "
        f"reference set.", fontsize=10)
    figpath = out.with_name(out.name + "_apa.png")
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    fig.savefig(figpath.with_suffix(".pdf"), bbox_inches="tight")
    log.info("wrote %s", figpath)
    log.info("wrote %s_*.csv", out)


if __name__ == "__main__":
    main()
