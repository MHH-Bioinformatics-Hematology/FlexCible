"""How much of the measured map is reproducible signal, and how close are we?

Scoring against a single measurement treats its sampling noise as truth. A
model that is smoother than the noise is punished for it, even where the
smoothing is right -- which the depth experiment already showed from the other
side: below about 3% of depth our prediction recovers TAD boundaries better
than an actual experiment of that depth.

The ceiling is measured, not assumed. Split the deep contact counts into two
DISJOINT halves (S ~ Binom(C, 0.5), and C - S), balance each independently, and
correlate them. Neither half sees the other's noise, so their agreement is the
share of the map that is reproducible at that depth. Spearman-Brown then
extrapolates the two half-depth replicates to the full-depth ceiling.

A model at the ceiling is as good as repeating the experiment. Above it would
mean predicting the noise, which is neither possible nor desirable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_regions_three_models import RAST, GM, dense   # noqa: E402


def akita_transform(counts: np.ndarray, max_d: int) -> np.ndarray:
    """counts -> log(obs/exp) clipped, the space everything else is scored in."""
    n = counts.shape[0]
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    exp = np.array([counts[d == k].mean() if (d == k).any() else 0.0
                    for k in range(n)])
    e = np.maximum(exp[d], 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        oe = np.log(np.maximum(counts, 1e-9) / e)
    oe = np.clip(oe, -2, 2)
    oe[d > max_d] = np.nan
    return oe


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100 or a[ok].std() < 1e-9 or b[ok].std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def spearman_brown(r_half: float) -> float:
    """Two half-depth replicates -> the full-depth reliability."""
    if not np.isfinite(r_half) or r_half <= 0:
        return np.nan
    return 2 * r_half / (1 + r_half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", default="/mnt/processed/wolffjoa/"
                    "restart_of_prediction_project/originalData/4Dnucleosome/"
                    "GM12878_4DNFIXP4QG5B.mcool::/resolutions/2000")
    ap.add_argument("--n-regions", type=int, default=60)
    ap.add_argument("--max-dist-bins", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/noise_ceiling.csv")
    a = ap.parse_args()

    z = np.load(RAST, allow_pickle=True)
    side, off = int(z["side"]), int(z["offset"])
    regions = list(z["regions"])
    clr = cooler.Cooler(a.deep)
    rng = np.random.default_rng(a.seed)

    ours = pd.read_csv("runs/akita_native/score_like_hicgan.csv")
    rows = []
    idx = np.linspace(0, len(regions) - 1, a.n_regions).astype(int)
    for c, k in enumerate(idx):
        reg = regions[k]
        chrom, span = reg.split(":")
        s, e = (int(x) for x in span.split("-"))
        name = chrom if chrom in clr.chromnames else chrom.replace("chr", "")
        if name not in clr.chromnames:
            continue
        try:
            C = np.asarray(clr.matrix(balance=False).fetch(
                f"{name}:{s}-{e}"), dtype=np.float64)
        except Exception:
            continue
        n = min(C.shape[0], side)
        C = C[:n, :n]
        C = np.nan_to_num(C)
        # two DISJOINT halves: neither carries the other's sampling noise
        S = rng.binomial(C.astype(np.int64), 0.5).astype(np.float64)
        T = C - S
        oe_s = akita_transform(S, a.max_dist_bins)
        oe_t = akita_transform(T, a.max_dist_bins)
        r_half = pearson(oe_s, oe_t)

        # the published target and our prediction, on the same pixels
        tgt = dense(z["akita_target"][k, :, GM], side, off)[:n, :n]
        aki = dense(z["akita_pred"][k, :, GM], side, off)[:n, :n]
        hg = dense(z["hicgan"][k], side, off)[:n, :n]
        hit = ours[ours.region == reg]
        r_ours = float(hit.ours.iloc[0]) if len(hit) else np.nan

        rows.append({"region": reg, "n": n,
                     "r_replicate_half": r_half,
                     "ceiling_full": spearman_brown(r_half),
                     "akita": pearson(tgt, aki),
                     "hicgan": pearson(tgt, hg),
                     "ours": r_ours})
        if (c + 1) % 10 == 0:
            print(f"  {c+1}/{len(idx)}", flush=True)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    d = df.dropna(subset=["ceiling_full"])
    print(f"\n{len(d)} regions")
    print(f"  half-depth replicate agreement : {d.r_replicate_half.median():.3f}")
    print(f"  NOISE CEILING (full depth)     : {d.ceiling_full.median():.3f}")
    print("\n  model scores, raw and as a fraction of the ceiling:")
    for m in ("ours", "akita", "hicgan"):
        v = d[m].dropna()
        frac = (d[m] / d.ceiling_full).dropna()
        print(f"    {m:7s} {v.median():.3f}   "
              f"{100*frac.median():5.1f}% of achievable")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
