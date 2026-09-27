"""Proper distance-normalized O/E (distance-only correlation ~0), genome-wide.

Fixes the distance leakage in akita_normalize.py so the target is a real
structural signal, not the distance-decay curve. Per chromosome:

  balanced counts (cooler weights)
  -> expected(d) = mean over ALL MAPPABLE pixel pairs at distance d
     (not just non-zero pairs -- that is what left ~0.76 distance-only)
  -> log(obs/exp), observed-absent mappable pixels = -2 (real depletion),
     unmappable = missing
  -> subtract the per-diagonal mean in log space (chromosome-wide), so every
     diagonal is centred: a distance-only predictor then scores ~0
  -> re-clip +/-2, Gaussian smooth (sigma 2), band to 2 Mb

Validated: GM12878 distance-only median r = 0.015 (vs 0.76 for the old script).
Two streaming passes over the pixel table gather expected(d) and the per-diagonal
log-offset(d); a third block pass writes the smoothed, banded, recentred cool.

    python scripts/oe_proper.py --input <mcool|cool> --output <cool> --binsize 2000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

MAX_DIST = 2_000_000
SIGMA = 2.0
CLIP = 2.0


def open_at(path: str, binsize: int) -> cooler.Cooler:
    if path.endswith(".mcool") or "::" in path:
        if "::" in path:
            return cooler.Cooler(path)
        avail = cooler.fileops.list_coolers(path)
        want = [u for u in avail if u.rstrip("/").endswith(f"/{binsize}")]
        if not want:
            res = sorted(int(u.rsplit("/", 1)[-1]) for u in avail)
            raise SystemExit(f"{path}: no {binsize} bp level; has {res}")
        return cooler.Cooler(f"{path}::{want[0]}")
    c = cooler.Cooler(path)
    if c.binsize != binsize:
        raise SystemExit(f"{path}: binsize {c.binsize}, wanted {binsize}")
    return c


def mappable_pairs_per_d(mp: np.ndarray, band: int) -> np.ndarray:
    """n mappable (i, i+d) pairs for d in [0, band], via direct shift-and-dot."""
    m = mp.astype(np.float64)
    out = np.zeros(band + 1)
    for d in range(band + 1):
        if d == 0:
            out[d] = m.sum()
        else:
            out[d] = float(np.dot(m[:-d], m[d:]))
    return out


def stream_stats(clr, chrom, band, weights, mp):
    """expected(d) over all mappable pairs, and per-diagonal log-offset(d)."""
    lo, hi = clr.extent(chrom)
    off = clr.offset(chrom)
    npair = mappable_pairs_per_d(mp, band)          # denominator per diagonal
    sum_v = np.zeros(band + 1)                       # sum balanced value (nonzero)
    n_nz = np.zeros(band + 1)                        # nonzero mappable pairs
    sel = clr.pixels()
    n = clr.info["nnz"]

    def first_at_least(target):
        a, b = 0, n
        while a < b:
            m = (a + b) // 2
            if int(sel[m:m + 1]["bin1_id"].values[0]) < target:
                a = m + 1
            else:
                b = m
        return a

    p0, p1 = first_at_least(lo), first_at_least(hi)
    step = 20_000_000
    # pass 1: expected(d)
    for s in range(p0, p1, step):
        px = sel[s:min(s + step, p1)]
        b1 = px["bin1_id"].values; b2 = px["bin2_id"].values
        v = px["count"].values.astype(np.float64)
        d = b2 - b1
        keep = (d <= band) & (b2 < hi) & (b1 >= lo)
        b1, b2, d, v = b1[keep], b2[keep], d[keep], v[keep]
        if weights is not None:
            v = v * weights[b1] * weights[b2]
        okm = np.isfinite(v) & (v > 0)
        np.add.at(sum_v, d[okm], v[okm])
        np.add.at(n_nz, d[okm], 1.0)
    exp = np.divide(sum_v, npair, out=np.full(band + 1, np.nan), where=npair > 0)
    exp[~np.isfinite(exp) | (exp <= 0)] = np.nan
    # pass 2: per-diagonal mean of clip(log(obs/exp)) over mappable pairs
    sum_lc = np.zeros(band + 1)
    for s in range(p0, p1, step):
        px = sel[s:min(s + step, p1)]
        b1 = px["bin1_id"].values; b2 = px["bin2_id"].values
        v = px["count"].values.astype(np.float64)
        d = b2 - b1
        keep = (d <= band) & (b2 < hi) & (b1 >= lo)
        b1, b2, d, v = b1[keep], b2[keep], d[keep], v[keep]
        if weights is not None:
            v = v * weights[b1] * weights[b2]
        # only mappable pairs (finite balanced value), nonzero handled here
        mm = np.isfinite(v) & (v > 0) & (exp[d] > 0)
        lc = np.clip(np.log(v[mm] / exp[d[mm]]), -CLIP, CLIP)
        np.add.at(sum_lc, d[mm], lc)
    # absent mappable pairs contribute -CLIP each: (npair - n_nz) * (-CLIP)
    n_absent = np.clip(npair - n_nz, 0, None)
    offset = np.divide(sum_lc + n_absent * (-CLIP), npair,
                       out=np.zeros(band + 1), where=npair > 0)
    return exp, offset


def emit_chrom(clr, chrom, band, weight_name, exp, offset, mp, block=2500):
    off = clr.offset(chrom)
    n = clr.extent(chrom)[1] - off
    halo = band
    for s in range(0, n, block):
        e = min(s + block, n)
        hs, he = max(0, s - halo), min(n, e + halo)
        reg = (f"{chrom}:{hs * clr.binsize}-"
               f"{min(he * clr.binsize, int(clr.chromsizes[chrom]))}")
        bal = weight_name if weight_name else False
        m = np.asarray(clr.matrix(balance=bal, sparse=False).fetch(reg), np.float64)
        k = m.shape[0]
        idx = np.arange(k)
        d = np.abs(idx[:, None] - idx[None, :])
        mappable = mp[hs:hs + k].astype(bool)
        pair_ok = mappable[:, None] & mappable[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            e_map = np.where(d <= band, exp[np.clip(d, 0, band)], np.nan)
            val = np.log(m / e_map)
        val = np.where(pair_ok & (m == 0), -CLIP, val)
        val = np.clip(val, -CLIP, CLIP)
        # subtract per-diagonal offset (the distance-normalization step)
        val = val - np.where(d <= band, offset[np.clip(d, 0, band)], 0.0)
        val = np.clip(val, -CLIP, CLIP)
        val[~np.isfinite(val)] = np.nan
        bad = ~np.isfinite(val)
        if bad.any():
            val = np.where(bad, 0.0, val)
        sm = gaussian_filter(val, sigma=SIGMA, truncate=1.0, mode="nearest")
        r0 = s - hs
        rows = np.arange(s, e)
        sub = sm[r0:r0 + (e - s)]
        ii, jj = np.nonzero(np.triu(np.ones_like(sub, dtype=bool), k=0))
        abs_i = rows[ii]; abs_j = jj + hs
        keep = (abs_j >= abs_i) & ((abs_j - abs_i) <= band)
        abs_i, abs_j = abs_i[keep], abs_j[keep]
        vals = sub[ii[keep], jj[keep]]
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        yield pd.DataFrame({"bin1_id": abs_i[ok] + off, "bin2_id": abs_j[ok] + off,
                            "count": vals[ok].astype(np.float32)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--binsize", type=int, default=2000)
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--max-dist", type=int, default=MAX_DIST,
                    help="band in bp (default 2 Mb); wider for windows spanning "
                         "more than 2 Mb, so no window pixel is an unmeasured zero")
    a = ap.parse_args()

    clr = open_at(a.input, a.binsize)
    band = a.max_dist // a.binsize
    chroms = a.chroms or [c for c in clr.chromnames
                          if c.replace("chr", "") in
                          [str(i) for i in range(1, 23)] + ["X"]]
    has_w = "weight" in clr.bins().columns
    weight_name = "weight" if has_w else None
    w_all = clr.bins()[:]["weight"].values if has_w else None
    print(f"{a.input}\n  {len(chroms)} chroms, band {band}, weights "
          + ("stored" if has_w else "NONE"), flush=True)

    bins = clr.bins()[:][["chrom", "start", "end"]]
    t0 = time.time()
    st = {"n": 0, "sum": 0.0, "neg": 0, "lo": np.inf, "hi": -np.inf}

    def stream():
        for ci, ch in enumerate(chroms, 1):
            o = clr.offset(ch); n = clr.extent(ch)[1] - o
            mp = np.isfinite(w_all[o:o + n]) if w_all is not None else np.ones(n, bool)
            exp, offset = stream_stats(clr, ch, band, w_all, mp)
            got = 0
            for fr in emit_chrom(clr, ch, band, weight_name, exp, offset, mp):
                v = fr["count"].values
                st["n"] += v.size; st["sum"] += float(v.sum())
                st["neg"] += int((v < 0).sum())
                st["lo"] = min(st["lo"], float(v.min())); st["hi"] = max(st["hi"], float(v.max()))
                got += len(fr); yield fr
            print(f"  [{ci}/{len(chroms)}] {ch}: {got:,} px ({(time.time()-t0)/60:.1f} min)", flush=True)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(a.output, bins=bins, pixels=stream(),
                         dtypes={"count": np.float32}, ordered=True)
    n = max(st["n"], 1)
    print(f"  -> {a.output}\n  range {st['lo']:.3f}..{st['hi']:.3f} "
          f"mean {st['sum']/n:.3f} neg {100*st['neg']/n:.1f}%", flush=True)


if __name__ == "__main__":
    main()
