"""Same region panels, but one row per OUR-model variant.

Upper triangle measured, lower triangle that variant's prediction, identical
regions and identical colour scale down every column, so the only thing
changing between rows is the training objective.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config                        # noqa: E402
from scripts.plot_regions_three_models import (RAST, GM, dense,
                                               our_prediction)  # noqa: E402


def scc_region(true: np.ndarray, pred: np.ndarray, max_d: int | None = None,
               weighted: bool = True) -> float:
    """HiCRep's stratum-adjusted correlation for one region.

    Correlate within each diagonal separately, then average with HiCRep's
    n*sx*sy weights. NaN pixels (the band the target does not define) are
    dropped per stratum rather than poisoning the whole map.
    """
    n = true.shape[0]
    dmax = min(max_d or n, n)
    num = den = 0.0
    for d in range(1, dmax):
        x = np.diagonal(true, d).astype(float)
        y = np.diagonal(pred, d).astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        x, y = x[ok], y[ok]
        sx, sy = x.std(), y.std()
        if sx < 1e-9 or sy < 1e-9:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        w = x.size * sx * sy if weighted else 1.0
        num += w * r
        den += w
    return num / den if den > 0 else float("nan")


def variant_score(run: str, region: str) -> float:
    """Whatever score file this run has, for annotation."""
    for f in ("score.csv", "score_notta.csv", "score_tta.csv",
              "score_like_hicgan.csv"):
        p = Path("runs") / run / f
        if p.exists():
            d = pd.read_csv(p)
            hit = d[d.region == region]
            if len(hit) and np.isfinite(hit.ours.iloc[0]):
                return float(hit.ours.iloc[0])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True,
                    help="run names under runs/, e.g. akita_native sharp_grad")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cmap", default="RdYlBu_r")
    ap.add_argument("--vmax", type=float, default=2.0)
    ap.add_argument("--metric", default="pearson",
                    choices=["pearson", "scc", "both"])
    ap.add_argument("--out", default="runs/variants_regions.png")
    a = ap.parse_args()
    labels = a.labels or a.variants

    z = np.load(RAST, allow_pickle=True)
    side, off = int(z["side"]), int(z["offset"])
    regions = list(z["regions"])

    ref = pd.read_csv("runs/akita_native/score_like_hicgan.csv").dropna(
        subset=["ours"])
    ref = ref[ref.region.isin(regions)].sort_values("ours")
    picks = ref.iloc[np.linspace(0, len(ref) - 1, a.n).astype(int)]

    nrow = len(a.variants)
    fig, axes = plt.subplots(nrow, a.n, figsize=(3.1 * a.n, 3.2 * nrow))
    axes = np.atleast_2d(axes)
    if nrow == 1:
        axes = axes.reshape(1, -1)

    for col, (_, r) in enumerate(picks.iterrows()):
        k = regions.index(r.region)
        chrom, span = r.region.split(":")
        s, e = (int(x) for x in span.split("-"))
        truth = dense(z["akita_target"][k, :, GM], side, off)
        keep = np.triu(np.ones((side, side), bool), 1)
        for row, run in enumerate(a.variants):
            cfg = Config.load(f"configs/{run}.yaml")
            P = our_prediction(cfg, Path("runs") / run / "checkpoint-best",
                               chrom, s, e, side)
            ax = axes[row, col]
            ax.imshow(np.where(keep, truth, P), cmap=a.cmap,
                      vmin=-a.vmax, vmax=a.vmax, interpolation="nearest")
            ax.plot([0, side - 1], [0, side - 1], color="#444", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if a.metric in ("scc", "both"):
                sv = scc_region(truth, P)
            if a.metric == "scc":
                lab = f"SCC = {sv:.3f}"
            elif a.metric == "both":
                lab = f"r = {variant_score(run, r.region):.3f}\nSCC = {sv:.3f}"
            else:
                lab = f"r = {variant_score(run, r.region):.3f}"
            sc = 0.0
            if True:
                ax.text(0.03, 0.97, lab, transform=ax.transAxes,
                        va="top", ha="left", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="none", boxstyle="round,pad=0.25"))
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=10)
            if row == 0:
                ax.set_title(f"{chrom}:{s/1e6:.2f}-{e/1e6:.2f} Mb", fontsize=9)
        print(f"  column {col+1}/{a.n} done", flush=True)

    cb = fig.colorbar(axes[0, 0].images[0], ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("log observed/expected")
    fig.suptitle("Above the diagonal: measured Hi-C.  Below: prediction.  "
                 "One row per training objective; identical regions and scale.",
                 fontsize=10)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
