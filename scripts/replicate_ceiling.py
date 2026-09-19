"""Reproducibility ceiling from real biological replicates.

Two independent experiments on the same cells cannot agree perfectly: what
they disagree about is noise, and no predictor can be expected to beat that.
Their agreement is therefore the honest denominator for every model score.

This is measured, not assumed, and it uses BIOLOGICAL replicates where they
exist. A binomial split of one experiment only captures sampling noise; two
biological replicates additionally capture library, protocol and cell-state
variation, which is what a model actually has to contend with.

Both replicates go through exactly the pipeline the scored targets went
through -- obs/exp, natural log, clipped, Gaussian-smoothed -- because a
ceiling computed on unsmoothed data is not comparable with scores computed
against a smoothed target. Getting that wrong produced an impossible "138% of
achievable" earlier today.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

CLIP = 2.0
SIGMA = 2.0


def akita_space(counts: np.ndarray, max_d: int, smooth: bool = True):
    """counts -> log(obs/exp), clipped, optionally smoothed. NaN beyond max_d."""
    n = counts.shape[0]
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    exp = np.array([np.nanmean(counts[d == k]) if (d == k).any() else np.nan
                    for k in range(n)])
    e = np.where(np.isfinite(exp) & (exp > 0), exp, np.nan)[d]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log(np.where(counts > 0, counts, np.nan) / e)
    v = np.clip(v, -CLIP, CLIP)
    if smooth:
        filled = np.where(np.isfinite(v), v, 0.0)
        v = gaussian_filter(filled, SIGMA, truncate=1.0, mode="nearest")
    v[d > max_d] = np.nan
    return v


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 200 or a[ok].std() < 1e-9 or b[ok].std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def hicrep_scc(A, B, max_d):
    """HiCRep's SCC on the BALANCED COUNTS, which is what it is defined for.

    Applying SCC to obs/exp is close to meaningless -- the decay it stratifies
    away has already been divided out, and on our data SCC and plain Pearson
    correlate at 0.9976. On raw contact matrices, which is the replicate
    comparison HiCRep was written for, it is the right measure.
    """
    num = den = 0.0
    for k in range(1, max_d):
        x, y = np.diagonal(A, k).astype(float), np.diagonal(B, k).astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20:
            continue
        x, y = x[ok], y[ok]
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        w = x.size * sx * sy
        num += w * r
        den += w
    return num / den if den > 0 else np.nan


def stratified(a, b, max_d):
    rs = []
    for k in range(1, max_d):
        x, y = np.diagonal(a, k), np.diagonal(b, k)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20:
            continue
        x, y = x[ok], y[ok]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        rs.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def open_at(path, res):
    """Cooler at (or nearest above) `res`.

    Older mcools name their levels /0../12 rather than by resolution, so the
    binsize has to be read from each level instead of parsed from its name.
    """
    avail = cooler.fileops.list_coolers(path)
    sizes = {}
    for u in avail:
        try:
            sizes[u] = cooler.Cooler(f"{path}::{u}").binsize
        except Exception:
            continue
    exact = [u for u, b in sizes.items() if b == res]
    if exact:
        return cooler.Cooler(f"{path}::{exact[0]}")
    # nearest level at or coarser than requested, so both replicates can match
    ge = sorted(((b, u) for u, b in sizes.items() if b and b >= res))
    if not ge:
        raise SystemExit(f"{path}: finest is {min(b for b in sizes.values() if b)}")
    b, u = ge[0]
    print(f"    {Path(path).name}: using {b} bp (asked {res})")
    return cooler.Cooler(f"{path}::{u}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="label=fileA,fileB")
    ap.add_argument("--res", type=int, default=8000)
    ap.add_argument("--span", type=int, default=1_048_576)
    ap.add_argument("--n-regions", type=int, default=40)
    ap.add_argument("--max-dist-bins", type=int, default=100)
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--out", default="runs/replicate_ceiling.csv")
    a = ap.parse_args()

    rows = []
    for spec in a.pairs:
        label, files = spec.split("=", 1)
        fa, fb = files.split(",")
        ca, cb = open_at(fa, a.res), open_at(fb, a.res)
        chroms = [c for c in ca.chromnames
                  if c.replace("chr", "").isdigit() and c in cb.chromnames]
        rng = np.random.default_rng(0)
        per = max(1, a.n_regions // max(len(chroms), 1))
        n_done = 0
        for ch in chroms:
            L = int(min(ca.chromsizes[ch], cb.chromsizes[ch]))
            if L < 4 * a.span:
                continue
            for s in rng.integers(a.span, L - 2 * a.span, per):
                reg = f"{ch}:{int(s)}-{int(s) + a.span}"
                try:
                    A = np.nan_to_num(np.asarray(
                        ca.matrix(balance=True).fetch(reg), float))
                    B = np.nan_to_num(np.asarray(
                        cb.matrix(balance=True).fetch(reg), float))
                except Exception:
                    continue
                n = min(A.shape[0], B.shape[0])
                if n < 40 or A.sum() <= 0 or B.sum() <= 0:
                    continue
                oa = akita_space(A[:n, :n], a.max_dist_bins, not a.no_smooth)
                ob = akita_space(B[:n, :n], a.max_dist_bins, not a.no_smooth)
                rows.append({"pair": label, "region": reg, "res": a.res,
                             "pearson_oe": pearson(oa, ob),
                             "stratified_oe": stratified(oa, ob, a.max_dist_bins),
                             "scc_counts": hicrep_scc(A[:n, :n], B[:n, :n],
                                                      a.max_dist_bins)})
                n_done += 1
        print(f"  {label}: {n_done} regions", flush=True)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nreplicate agreement at {a.res} bp, "
          f"{'smoothed' if not a.no_smooth else 'UNsmoothed'} "
          f"(this is the ceiling any model is measured against):\n")
    print(f"{'pair':12s} {'n':>4s} {'pearson_oe':>11s} {'strat_oe':>9s} "
          f"{'SCC_counts':>11s}")
    for label, g in df.groupby("pair"):
        print(f"{label:12s} {len(g):4d} {g.pearson_oe.median():11.3f} "
              f"{g.stratified_oe.median():9.3f} {g.scc_counts.median():11.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
