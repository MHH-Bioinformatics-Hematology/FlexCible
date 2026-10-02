"""Put the single-replicate O/E maps on their nominal grid.

The replicate mcools of scripts/replicates_all_cells.py (hicBuildMatrix,
base 2 kb) hold their "5000" and "25000" levels at 4 kb and 24 kb bins (the
base bin times floor(res / 2000)), so <cell>_<rep>_ice_5kb.cool and
_25kb.cool are 4 and 24 kb maps. This script keeps them under their true
names (_ice_4kb.cool, _ice_24kb.cool) and writes 5 and 25 kb maps by
base-pair-overlap averaging: a 25 kb pixel is the overlap-weighted mean of
the measured 24 kb pixels it covers, written when they cover at least half
of its area (0 = unmeasured, as in the source).

    python scripts/regrid_replicates.py [--jobs 4]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import scipy.sparse as sp

OE = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_oe")
CELLS = ["gm12878", "imr90", "k562", "hmec", "huvec"]
REPS = ["rep1", "rep2"]
NOMINAL = {5000: 4000, 25000: 24000}       # nominal -> actual bin of the source
MIN_COVER = 0.5


def overlap_matrix(n_t: int, res_t: int, n_s: int, res_s: int, chromlen: int):
    """(n_t, n_s) fraction of each target bin covered by each source bin."""
    rows, cols, vals = [], [], []
    for j in range(n_t):
        a, b = j * res_t, min((j + 1) * res_t, chromlen)
        for i in range(a // res_s, min(n_s, (b - 1) // res_s + 1)):
            ov = min(b, (i + 1) * res_s) - max(a, i * res_s)
            if ov > 0:
                rows.append(j); cols.append(i); vals.append(ov / (b - a))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_t, n_s))


def regrid(src: Path, dst: Path, res_t: int, chromsizes: pd.Series):
    c = cooler.Cooler(str(src))
    res_s = c.binsize
    bins_t = cooler.binnify(chromsizes, res_t)
    offs = np.concatenate([[0], np.cumsum([int(np.ceil(chromsizes[ch] / res_t))
                                           for ch in chromsizes.index])])
    parts = []
    for k, ch in enumerate(chromsizes.index):
        if ch not in c.chromnames:
            continue
        L = int(chromsizes[ch])
        n_t = int(np.ceil(L / res_t))
        ext = c.extent(ch)
        n_s = ext[1] - ext[0]
        px = c.matrix(balance=False, as_pixels=True).fetch(ch)
        px = px[(px["count"] != 0) & np.isfinite(px["count"])]
        i = px["bin1_id"].values - ext[0]
        j = px["bin2_id"].values - ext[0]
        v = px["count"].values.astype(float)
        off = i != j
        r = np.concatenate([i, j[off]]); s = np.concatenate([j, i[off]])
        M = sp.csr_matrix((np.concatenate([v, v[off]]), (r, s)), shape=(n_s, n_s))
        I = sp.csr_matrix((np.ones(r.size), (r, s)), shape=(n_s, n_s))
        A = overlap_matrix(n_t, res_t, n_s, res_s, L)
        N = (A @ M @ A.T).tocsr()
        D = (A @ I @ A.T).tocsr()
        N.sort_indices(); D.sort_indices()
        if not (np.array_equal(N.indptr, D.indptr) and np.array_equal(N.indices, D.indices)):
            raise RuntimeError(f"{src} {ch}: value and coverage patterns differ")
        N, D = N.tocoo(), D.tocoo()
        up = N.row <= N.col
        rr, cc, nv, dv = N.row[up], N.col[up], N.data[up], D.data[up]
        keep = dv >= MIN_COVER
        parts.append(pd.DataFrame({"bin1_id": rr[keep] + offs[k], "bin2_id": cc[keep] + offs[k],
                                   "count": nv[keep] / dv[keep]}))
    pix = pd.concat(parts).sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    tmp = dst.with_suffix(".part.cool")
    cooler.create_cooler(str(tmp), bins_t, pix, dtypes={"count": np.float64},
                         ordered=True, assembly="hg38",
                         metadata={"regridded_from": str(src), "source_binsize": int(res_s)})
    tmp.rename(dst)
    return f"{dst.name}: {len(pix)} pixels from {src.name} ({res_s} bp)"


def job(cell, rep, nominal):
    actual = NOMINAL[nominal]
    nom = OE / f"{cell}_{rep}_ice_{nominal // 1000}kb.cool"
    true = OE / f"{cell}_{rep}_ice_{actual // 1000}kb.cool"
    if not true.exists():
        if cooler.Cooler(str(nom)).binsize != actual:
            return f"{nom.name}: binsize {cooler.Cooler(str(nom)).binsize}, nothing to do"
        nom.rename(true)                    # keep the source under its true name
    if nom.exists() and cooler.Cooler(str(nom)).binsize == nominal:
        return f"{nom.name}: done"
    two = cooler.Cooler(str(OE / f"{cell}_{rep}_ice_2kb.cool"))
    return regrid(true, nom, nominal, two.chromsizes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    todo = [(c, r, n) for n in (25000, 5000) for c in CELLS for r in REPS]
    with ProcessPoolExecutor(a.jobs) as ex:
        for msg in ex.map(job, *zip(*todo)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
