"""Extend the manuscript's Akita-vs-Hi-cGAN figure with this model.

READ-ONLY with respect to the manuscript. It reads
Hi-cGAN/manuscript_gigascience/data/akita_vs_hicgan_heldout.csv and writes
nothing back: no column is added to that table, the manuscript's own plotting
code is not imported (its savefig writes into the manuscript figure directory),
and every output goes to this project's runs/ directory.

Panels:
  A  our per-region Pearson against Akita's, one point per test window, with
     the line of equality. Hi-cGAN's points are drawn against Akita too, so
     all three models appear in one frame rather than in separate figures.
  B  ECDF per model: compares them at every quantile instead of only at the
     median, which matters because the medians hide very different spreads.
  C  paired difference against Akita, sorted, so it is visible how much of the
     win is a few regions and how much is a broad shift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

MAN_TABLE = (Path.home() / "src/Hi-cGAN/manuscript_gigascience/data"
             / "akita_vs_hicgan_heldout.csv")
INK, MUTED, GREY = "#22252a", "#7b8090", "#c8ccd4"
COL = {"akita": "#5aa469", "hicgan": "#b05aa4", "ours": "#cf5b56"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="runs/akita_native/score_like_hicgan.csv")
    ap.add_argument("--label", default="this model (multi-omic U-Net)")
    ap.add_argument("--out", default="runs/akita_hicgan_ours.png")
    a = ap.parse_args()

    man = pd.read_csv(MAN_TABLE)              # read-only
    ours = pd.read_csv(a.ours)[["region", "ours"]]
    df = man.merge(ours, on="region", how="inner").dropna(
        subset=["akita_gm12878", "ours"])
    hg = df.dropna(subset=["hicgan_fixed"])
    print(f"{len(df)} shared test windows ({len(hg)} with Hi-cGAN)")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "grid.color": GREY, "figure.dpi": 150})
    fig = plt.figure(figsize=(13.6, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # ---- A: against Akita, both challengers in one frame
    ax = fig.add_subplot(gs[0, 0])
    lo = min(df.akita_gm12878.min(), df.ours.min(), hg.hicgan_fixed.min()) - 0.05
    hi = max(df.akita_gm12878.max(), df.ours.max(), hg.hicgan_fixed.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.0, zorder=1)
    ax.scatter(hg.akita_gm12878, hg.hicgan_fixed, s=12, color=COL["hicgan"],
               alpha=0.45, edgecolor="none", zorder=2, label="Hi-cGAN")
    ax.scatter(df.akita_gm12878, df.ours, s=14, color=COL["ours"],
               alpha=0.6, edgecolor="none", zorder=3, label=a.label)
    won = int((df.ours > df.akita_gm12878).sum())
    ax.text(0.03, 0.97,
            f"n = {len(df)} test windows\n"
            f"Akita   median {df.akita_gm12878.median():.3f}\n"
            f"ours    median {df.ours.median():.3f}\n"
            f"ours ahead in {won} ({won/len(df):.0%})",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                      boxstyle="round,pad=0.3"))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Akita, Pearson per test window")
    ax.set_ylabel("challenger, Pearson per test window")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("A  per-region, against Akita", loc="left", fontsize=10)

    # ---- B: ECDF
    ax = fig.add_subplot(gs[0, 1])
    for key, col, lab in (("akita_gm12878", COL["akita"], "Akita (recorded)"),
                          ("hicgan_fixed", COL["hicgan"], "Hi-cGAN (recorded)"),
                          ("ours", COL["ours"], a.label)):
        v = np.sort(df[key].dropna().values)
        if not v.size:
            continue
        y = np.arange(1, v.size + 1) / v.size
        ax.step(v, y, where="post", color=col, lw=2.0,
                label=f"{lab}   median {np.median(v):.3f}")
    ax.axvline(0.0, color=GREY, lw=0.9)
    ax.set_xlabel("Pearson per Akita test window")
    ax.set_ylabel("fraction of windows at or below")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("B  distribution over windows", loc="left", fontsize=10)

    # ---- C: sorted paired difference
    ax = fig.add_subplot(gs[0, 2])
    d_ours = np.sort((df.ours - df.akita_gm12878).values)
    ax.fill_between(np.arange(d_ours.size), 0, d_ours,
                    where=d_ours >= 0, color=COL["ours"], alpha=0.55,
                    step="mid", label="ours better")
    ax.fill_between(np.arange(d_ours.size), 0, d_ours,
                    where=d_ours < 0, color=MUTED, alpha=0.45,
                    step="mid", label="Akita better")
    if len(hg):
        d_hg = np.sort((hg.hicgan_fixed - hg.akita_gm12878).values)
        ax.plot(np.linspace(0, d_ours.size - 1, d_hg.size), d_hg,
                color=COL["hicgan"], lw=1.6, label="Hi-cGAN - Akita")
    ax.axhline(0, color=GREY, lw=1.0)
    ax.set_xlabel("test windows, sorted by difference")
    ax.set_ylabel("Pearson difference vs Akita")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("C  paired difference", loc="left", fontsize=10)

    fig.suptitle(
        "Akita's 412 held-out windows, GM12878, scored on Akita's own published "
        "target with the manuscript's metric.  Akita and Hi-cGAN values are "
        "reused as recorded, never recomputed.", fontsize=9, y=1.02)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    print(f"manuscript table untouched: {MAN_TABLE}")


if __name__ == "__main__":
    main()
