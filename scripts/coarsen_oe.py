"""Coarse O/E targets by block-averaging a fine proper O/E map.

Reads a fine oe_proper.py map (e.g. GM12878 at 2 kb) and writes, per coarse
resolution, the mean of the fine values over each coarse pixel's block of the
symmetric matrix, weighted by base-pair overlap (25 kb bins cut every second
2 kb bin in half; such a bin contributes half its weight to each side). The
alternative coarse target to running oe_proper.py on the coarse matrix: the
fine map is distance-normalized and smoothed at its own resolution (sigma 2
fine bins), so the averaged map keeps structure the coarse smoothing removes.

Coverage. The fine map stores only its distance band (2 Mb for the default
oe_proper.py output), so a coarse pixel is written only when its whole block
lies inside that band (offset D <= (band + fine) / coarse - 1: 79 bins at
25 kb, 199 at 10 kb, which covers the 80- and 192-bin windows), and when the
fine pixels cover at least --min-coverage of the block (chromosome ends).
The per-diagonal mean is not recentred.

    python scripts/coarsen_oe.py --fine crosscell_oe/gm12878_proper_2kb.cool \
        --res 10000 25000 --template crosscell_oe/gm12878_proper_{k}kb.cool \
        --out crosscell_oe/gm12878_avg2kb_{k}kb.cool

Each output is written to <out>.part and renamed when complete.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cooler
import h5py
import numpy as np
import pandas as pd

CHUNK = 20_000_000


def bin_map(nf: int, fres: int, cres: int, chromlen: int):
    """Per fine bin: first/second coarse bin and bp-overlap weights (w2 = 0 when
    the fine bin lies in one coarse bin); weights in units of a full fine bin."""
    st = np.arange(nf, dtype=np.int64) * fres
    en = np.minimum(st + fres, chromlen)
    c1 = st // cres
    c2 = (en - 1) // cres
    w1 = (np.minimum(en, (c1 + 1) * cres) - st) / fres
    w2 = np.where(c2 > c1, (en - (c1 + 1) * cres) / fres, 0.0)
    return c1, c2, w1, w2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fine", required=True)
    ap.add_argument("--res", nargs="+", type=int, required=True)
    ap.add_argument("--template", required=True,
                    help="bin table source per resolution; {k} = res in kb, {res} = res in bp")
    ap.add_argument("--out", required=True, help="output per resolution; {k} = res in kb, {res} = res in bp")
    ap.add_argument("--band", type=int, default=2_000_000,
                    help="distance band of the fine map in bp (oe_proper --max-dist)")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--chroms", nargs="*", default=None)
    a = ap.parse_args()

    fine = cooler.Cooler(a.fine)
    fres = fine.binsize
    band_f = a.band // fres
    tmpl = {r: cooler.Cooler(a.template.format(k=r // 1000, res=r)) for r in a.res}
    for r, t in tmpl.items():
        if t.binsize != r:
            raise SystemExit(f"template for {r}: binsize {t.binsize}")
    dmax = {r: int((band_f * fres + fres) // r - 1) for r in a.res}
    print(f"fine {a.fine} ({fres} bp, band {band_f} bins); coarse max offset "
          + ", ".join(f"{r // 1000} kb: {d}" for r, d in dmax.items()), flush=True)
    with h5py.File(fine.filename, "r") as h:
        grp = h[fine.root] if fine.root != "/" else h
        b1off = grp["indexes/bin1_offset"][:]
    chroms = a.chroms or [c for c in fine.chromnames
                          if fine.info["nnz"] and c not in ("chrM", "chrY", "MT", "Y", "M")]
    out_px = {r: [] for r in a.res}
    stats = {r: {"pixels": 0, "dropped_low_coverage": 0} for r in a.res}
    sel = fine.pixels()
    t0 = time.time()
    for chrom in chroms:
        fo = int(fine.offset(chrom))
        nf = int(fine.extent(chrom)[1]) - fo
        p0, p1 = int(b1off[fo]), int(b1off[fo + nf])
        if p1 <= p0:
            continue
        clen = int(fine.chromsizes[chrom])
        acc = {}
        for r in a.res:
            nc = int(tmpl[r].extent(chrom)[1] - tmpl[r].offset(chrom))
            acc[r] = (nc, np.zeros(nc * (dmax[r] + 1)), np.zeros(nc * (dmax[r] + 1)),
                      bin_map(nf, fres, r, clen))
        for s in range(p0, p1, CHUNK):
            px = sel[s:min(s + CHUNK, p1)]
            i = px["bin1_id"].values - fo
            j = px["bin2_id"].values - fo
            v = px["count"].values.astype(np.float64)
            ok = (j < nf) & (j >= i) & np.isfinite(v)
            i, j, v = i[ok], j[ok], v[ok]
            offdiag = i < j
            for r in a.res:
                nc, S, Wt, (c1, c2, w1, w2) = acc[r]
                for ci_, wi_ in ((c1, w1), (c2, w2)):
                    ci, wi = ci_[i], wi_[i]
                    for cj_, wj_ in ((c1, w1), (c2, w2)):
                        cj, wj = cj_[j], wj_[j]
                        w = wi * wj
                        # the full symmetric block: an off-diagonal fine pixel
                        # appears twice in a diagonal coarse block, once elsewhere
                        w = np.where((ci == cj) & offdiag, 2.0 * w, w)
                        d = cj - ci
                        k = (w > 0) & (d >= 0) & (d <= dmax[r]) & (cj < nc)
                        flat = ci[k] * (dmax[r] + 1) + d[k]
                        S += np.bincount(flat, weights=w[k] * v[k], minlength=S.size)
                        Wt += np.bincount(flat, weights=w[k], minlength=Wt.size)
        for r in a.res:
            nc, S, Wt, _ = acc[r]
            full = (r / fres) ** 2
            cov = Wt / full
            keep = cov >= a.min_coverage
            stats[r]["dropped_low_coverage"] += int(((Wt > 0) & ~keep).sum())
            flat = np.nonzero(keep)[0]
            ci, d = flat // (dmax[r] + 1), flat % (dmax[r] + 1)
            co = int(tmpl[r].offset(chrom))
            out_px[r].append(pd.DataFrame({
                "bin1_id": ci + co, "bin2_id": ci + d + co,
                "count": (S[flat] / Wt[flat]).astype(np.float32)}))
            stats[r]["pixels"] += len(flat)
        print(f"  {chrom}: {p1 - p0:,} fine pixels ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    for r in a.res:
        out = Path(a.out.format(k=r // 1000, res=r))
        part = out.with_name(out.name + ".part")
        part.unlink(missing_ok=True)
        bins = tmpl[r].bins()[:][["chrom", "start", "end"]]
        px = pd.concat(out_px[r], ignore_index=True)
        cooler.create_cooler(str(part), bins=bins, pixels=px,
                             dtypes={"count": np.float32}, ordered=True)
        part.rename(out)
        meta = {"fine": a.fine, "res": r, "max_offset_bins": int(dmax[r]),
                "min_coverage": a.min_coverage, **stats[r],
                "method": "bp-overlap weighted block mean of the fine O/E map"}
        Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
        print(f"-> {out}: {stats[r]}", flush=True)


if __name__ == "__main__":
    main()
