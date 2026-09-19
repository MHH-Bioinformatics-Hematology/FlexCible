"""Put any Hi-C / Micro-C matrix into Akita's target space, identically.

Why this exists: the two matrices already on disk are NOT in the same space.
GM12878 holds log(obs/exp) (95.6% negative, clipped at -2) while H1 holds
log1p(obs/exp) (never negative). They came from different code paths. Training
one model on both would make H1 look systematically unlike GM12878 for purely
arithmetic reasons, and the model would learn a "cell type" difference that is
a normalisation artifact -- which would silently corrupt exactly the
cell-type-independence claim these matrices are meant to test.

So every cell type is renormalised from source through this one path:

  balanced counts            (cooler weights; KR/ICE as shipped)
  -> obs / expected(distance)   expected = mean over the NON-ZERO pixels at
                                that distance (the convention the reference
                                matrices use); averaging over all possible
                                pixels instead makes expected ~3.4x smaller
                                and shifts every value by about +1.2
  -> log, clipped to +/-2       Akita's transform
  -> Gaussian smoothing         sigma=2 (5x5), as in the original notebook
  -> banded to 2 Mb             beyond that Akita predicts nothing anyway

Validation: run with --check against the existing GM12878 file. If the value
distribution does not match, the convention is wrong and nothing downstream
should be trusted.
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
    """Cooler at the requested resolution, from a .cool or an .mcool."""
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


def _pixel_range(clr: cooler.Cooler, lo: int, hi: int) -> tuple[int, int]:
    """Index range of a chromosome's rows in the pixel table (sorted by bin1)."""
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

    return first_at_least(lo), first_at_least(hi)


def expected_by_distance(clr: cooler.Cooler, chrom: str, band: int,
                         weights: np.ndarray | None) -> np.ndarray:
    """Mean balanced value per diagonal, averaged over NON-ZERO pixels.

    This is the convention the reference matrices use (hence their
    'obs_exp_non_zero' name), and it is not cosmetic: averaging over all
    possible pixels instead makes the expected ~3.4x smaller here, shifting
    every log(obs/exp) by about +1.2 and putting the targets in a completely
    different space. Streaming the pixel table gives the non-zero counts for
    free, since absent pixels simply are not stored.
    """
    lo, hi = clr.extent(chrom)
    p0, p1 = _pixel_range(clr, lo, hi)
    tot = np.zeros(band + 1)
    cnt = np.zeros(band + 1)
    sel = clr.pixels()
    step = 20_000_000
    for s in range(p0, p1, step):
        px = sel[s:min(s + step, p1)]
        b1 = px["bin1_id"].values
        b2 = px["bin2_id"].values
        v = px["count"].values.astype(np.float64)
        d = b2 - b1
        m = (d <= band) & (b2 < hi) & (b1 >= lo)
        b1, b2, d, v = b1[m], b2[m], d[m], v[m]
        if weights is not None:
            v = v * weights[b1] * weights[b2]
        ok = np.isfinite(v) & (v > 0)
        np.add.at(tot, d[ok], v[ok])
        np.add.at(cnt, d[ok], 1.0)
    exp = np.divide(tot, cnt, out=np.full(band + 1, np.nan), where=cnt > 0)
    exp[~np.isfinite(exp) | (exp <= 0)] = np.nan
    return exp


def normalise_chrom(clr, chrom, band, weight_name, exp, block=2500):
    """Yield normalised pixel frames for one chromosome, block by block."""
    off = clr.offset(chrom)
    n = clr.extent(chrom)[1] - off
    halo = band
    for s in range(0, n, block):
        e = min(s + block, n)
        # halo on both sides so smoothing near block edges sees real data
        hs, he = max(0, s - halo), min(n, e + halo)
        # the last bin of a chromosome is partial, so bins*binsize overruns the
        # chromosome length and cooler rejects the region
        reg = (f"{chrom}:{hs * clr.binsize}-"
               f"{min(he * clr.binsize, int(clr.chromsizes[chrom]))}")
        bal = weight_name if weight_name else False
        m = clr.matrix(balance=bal, sparse=False).fetch(reg)
        m = np.asarray(m, dtype=np.float64)
        k = m.shape[0]
        idx = np.arange(k)
        d = np.abs(idx[:, None] - idx[None, :])
        # Two very different reasons a pixel has no value, and they must not be
        # conflated. A mappable pixel with zero contacts is real signal --
        # log(0) = -inf, which clips to -2, the strongest "less than expected".
        # An unmappable bin carries no information at all and has to be filled
        # neutrally instead. Treating the first like the second sets 70% of the
        # band to 0 ("exactly as expected") and wrecks the distribution.
        try:
            wcol = clr.bins()[hs + off:he + off]["weight"].values
            mappable = np.isfinite(wcol)
        except Exception:
            mappable = np.ones(k, dtype=bool)
        pair_ok = mappable[:, None] & mappable[None, :]

        with np.errstate(invalid="ignore", divide="ignore"):
            e_map = np.where(d <= band, exp[np.clip(d, 0, band)], np.nan)
            oe = m / e_map
            val = np.log(oe)
        val = np.where(pair_ok & (m == 0), -CLIP, val)     # observed absence
        val = np.clip(val, -CLIP, CLIP)
        val[~np.isfinite(val)] = np.nan
        bad = ~np.isfinite(val)                            # unmappable only
        if bad.any():
            val = np.where(bad, 0.0, val)
        sm = gaussian_filter(val, sigma=SIGMA, truncate=1.0, mode="nearest")
        # keep only the rows this block owns, and only the 2 Mb band
        r0 = s - hs
        rows = np.arange(s, e)
        sub = sm[r0:r0 + (e - s)]
        ii, jj = np.nonzero(np.triu(np.ones_like(sub, dtype=bool), k=0))
        # translate to absolute bins
        abs_i = rows[ii]
        abs_j = jj + hs
        keep = (abs_j >= abs_i) & ((abs_j - abs_i) <= band)
        abs_i, abs_j = abs_i[keep], abs_j[keep]
        vals = sub[ii[keep], jj[keep]]
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        yield pd.DataFrame({"bin1_id": abs_i[ok] + off,
                            "bin2_id": abs_j[ok] + off,
                            "count": vals[ok].astype(np.float32)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--binsize", type=int, default=2000)
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--check", default=None,
                    help="reference cool to compare the value distribution to")
    a = ap.parse_args()

    clr = open_at(a.input, a.binsize)
    band = MAX_DIST // a.binsize
    chroms = a.chroms or [c for c in clr.chromnames
                          if c.replace("chr", "") in
                          [str(i) for i in range(1, 23)] + ["X"]]
    print(f"{a.input}\n  {len(chroms)} chromosomes, band {band} bins "
          f"({MAX_DIST/1e6:.0f} Mb)")

    has_w = "weight" in clr.bins().columns
    weight_name = "weight" if has_w else None
    w_all = clr.bins()[:]["weight"].values if has_w else None
    print("  balancing weights: " + ("stored 'weight'" if has_w
                                     else "NONE -> raw counts"))

    bins = clr.bins()[:][["chrom", "start", "end"]]
    t0 = time.time()
    stats = {"n": 0, "sum": 0.0, "neg": 0, "lo": np.inf, "hi": -np.inf}

    def stream():
        """Yield pixel frames in globally sorted order.

        Holding them all first and concatenating cost ~27 GB for GM12878's
        1.37 billion pixels, doubled again by the sort -- earlyoom killed every
        run at exactly that step, after an hour of correct per-chromosome work.
        Chromosomes are visited in bin order and blocks in ascending rows, so
        the stream is already sorted and cooler can consume it directly.
        """
        for ci, ch in enumerate(chroms, 1):
            exp = expected_by_distance(clr, ch, band, w_all)
            got = 0
            for fr in normalise_chrom(clr, ch, band, weight_name, exp):
                v = fr["count"].values
                stats["n"] += v.size
                stats["sum"] += float(v.sum())
                stats["neg"] += int((v < 0).sum())
                stats["lo"] = min(stats["lo"], float(v.min()))
                stats["hi"] = max(stats["hi"], float(v.max()))
                got += len(fr)
                yield fr
            print(f"  [{ci}/{len(chroms)}] {ch}: {got:,} pixels "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(a.output, bins=bins, pixels=stream(),
                         dtypes={"count": np.float32}, ordered=True)

    n = max(stats["n"], 1)
    print(f"  total {stats['n']:,} pixels -> {a.output}")
    print(f"  result: range {stats['lo']:.3f}..{stats['hi']:.3f}  "
          f"mean {stats['sum']/n:.3f}  neg {100*stats['neg']/n:.1f}%")
    if a.check:
        c2 = cooler.Cooler(a.check)
        r = c2.pixels()[:2_000_000]["count"].values.astype(float)
        print(f"  reference: range {r.min():.3f}..{r.max():.3f}  "
              f"mean {r.mean():.3f}  neg {100*(r<0).mean():.1f}%")


if __name__ == "__main__":
    main()
