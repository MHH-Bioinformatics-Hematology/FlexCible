"""Back-transform proper O/E (scripts/oe_proper.py) to balanced contact counts.

The model (and the targets) live in the O/E space of oe_proper.py. Per chromosome,
for the measured map of the SAME cell:

  expected(d) = mean balanced count over all mappable pairs at distance d
  offset(d)   = chromosome-wide per-diagonal mean of clip(log(obs/exp), +/-2),
                absent mappable pixels counted as -2
  O/E pixel   = gaussian_sigma2( clip( clip(log(obs/exp), +/-2) - offset(d), +/-2 ) )

This script inverts the log-offset-exp part for every pixel of an O/E cool:

  count(i, j) = exp( oe(i, j) + offset(d) ) * expected(d),    d = |j - i|

so the output is a cool in BALANCED-COUNT scale (cooler weights applied,
i.e. the quantity oe_proper.py divides by expected), restricted to the pixels
present in the input O/E cool (within the 2 Mb band of oe_proper.py).

What the output is, and is not
  * expected(d) and offset(d) are computed from the MEASURED balanced map
    (--measured), with the same two streaming passes as oe_proper.py
    (oe_proper.stream_stats is imported, not copied). The distance decay of the
    output is therefore the measured one and is NOT predicted; only the
    deviation from it comes from the O/E input.
  * Clipping at +/-2 and the Gaussian smoothing (sigma 2) of oe_proper.py cannot
    be undone. Applying this script to the measured O/E cool of the cell gives
    the "smoothing floor": the counts a perfect prediction in O/E space would at
    best reproduce. Compare a prediction with the measured counts and with that
    floor, not with the measured counts alone. (--oe-is-target only changes the
    log line; the arithmetic is the same.)
  * Pixels with expected(d) undefined (no mappable pair at that distance) are
    dropped.

    python scripts/oe_to_counts.py --oe pred.cool --measured cell.mcool \
        --output counts.cool --binsize 10000 [--chroms chr2] \
        [--region chr2:18000000-22000000] [--stats-cache stats.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe_proper as OP  # noqa: E402


def chrom_stats(clr, chrom, band, cache=None):
    """expected(d), offset(d) of the measured balanced map for one chromosome."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        if f"{chrom}_exp" in z.files and len(z[f"{chrom}_exp"]) == band + 1:
            return z[f"{chrom}_exp"], z[f"{chrom}_off"]
    has_w = "weight" in clr.bins().columns
    o = clr.offset(chrom)
    n = clr.extent(chrom)[1] - o
    w_all = clr.bins()[:]["weight"].values if has_w else None
    mp = np.isfinite(w_all[o:o + n]) if has_w else np.ones(n, bool)
    exp, off = OP.stream_stats(clr, chrom, band, w_all, mp)
    if cache is not None:
        old = dict(np.load(cache)) if Path(cache).exists() else {}
        old.update({f"{chrom}_exp": exp, f"{chrom}_off": off})
        np.savez(cache, **old)
    return exp, off


def to_counts(oe_clr, chrom, band, exp, offset, region=None):
    """Pixel table (absolute bin ids) of exp(oe + offset(d)) * expected(d)."""
    q = region if region else chrom
    px = oe_clr.matrix(balance=False, as_pixels=True, join=False).fetch(q)
    b1 = px["bin1_id"].values
    b2 = px["bin2_id"].values
    v = px["count"].values.astype(np.float64)
    d = b2 - b1
    keep = (d >= 0) & (d <= band) & np.isfinite(v)
    b1, b2, d, v = b1[keep], b2[keep], d[keep], v[keep]
    e = exp[d]
    c = np.exp(v + offset[d]) * e
    ok = np.isfinite(c) & (e > 0)
    return pd.DataFrame({"bin1_id": b1[ok], "bin2_id": b2[ok],
                         "count": c[ok].astype(np.float32)})


def convert(oe_path, measured, output, binsize, chroms=None, region=None,
            max_dist=OP.MAX_DIST, stats_cache=None):
    oe = cooler.Cooler(oe_path)
    if oe.binsize != binsize:
        raise SystemExit(f"{oe_path}: binsize {oe.binsize}, wanted {binsize}")
    meas = OP.open_at(measured, binsize)
    band = max_dist // binsize
    if region:
        chroms = [region.split(":")[0]]
    chroms = chroms or [c for c in oe.chromnames if c in meas.chromnames]
    bins = oe.bins()[:][["chrom", "start", "end"]]

    def stream():
        for ch in chroms:
            exp, off = chrom_stats(meas, ch, band, stats_cache)
            df = to_counts(oe, ch, band, exp, off, region)
            print(f"  {ch}: {len(df):,} px", flush=True)
            if len(df):
                yield df

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(str(output), bins=bins, pixels=stream(),
                         dtypes={"count": np.float32}, ordered=True)
    print(f"  -> {output}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oe", required=True,
                    help="O/E cool: model output, or the measured O/E target (smoothing floor)")
    ap.add_argument("--measured", required=True,
                    help="balanced mcool/cool of the same cell (source of expected and offset)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--binsize", type=int, default=10000)
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--region", default=None,
                    help="chrom:start-end; only pixels with both bins inside (statistics stay chromosome-wide)")
    ap.add_argument("--max-dist", type=int, default=OP.MAX_DIST)
    ap.add_argument("--stats-cache", default=None, help="npz cache of expected/offset per chromosome")
    ap.add_argument("--oe-is-target", action="store_true",
                    help="label only: the input is the measured O/E target, output = smoothing floor")
    a = ap.parse_args()
    print(("smoothing floor from " if a.oe_is_target else "back-transform of ") + a.oe, flush=True)
    convert(a.oe, a.measured, a.output, a.binsize, a.chroms, a.region, a.max_dist, a.stats_cache)


if __name__ == "__main__":
    main()
