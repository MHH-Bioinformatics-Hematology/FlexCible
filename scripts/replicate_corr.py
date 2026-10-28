"""rep1-vs-rep2 per-region Pearson on proper O/E, per cell line and resolution.

Reads <cell>_<rep>_ice_<res>kb.cool from crosscell_oe (see replicates_all_cells.py)
and the raw mcools for depth. Regions: the 413 Akita test regions; a region needs
>= 20 bins, so 1 Mb regions still count at 25 kb (40 bins).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_like_hicgan as sc                       # noqa: E402
from flexcible.genome import match_name           # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_oe")
W = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_raw")
CELLS = ["gm12878", "imr90", "k562", "hmec", "huvec"]
RES = [2, 5, 10, 25]


def contacts(cell, rep):
    p = W / f"{cell}_{rep}.mcool"
    if not p.exists():
        return float("nan")
    c = cooler.Cooler(f"{p}::/resolutions/25000")
    px = c.pixels(); s = 0.0
    for i in range(0, c.info["nnz"], 20_000_000):
        s += float(px[i:i + 20_000_000]["count"].sum())
    return s


def region_rs(a, b, regions):
    rs = []
    for _, r in regions.iterrows():
        na, nb = match_name(r.chrom, list(a.chromnames)), match_name(r.chrom, list(b.chromnames))
        if not na or not nb:
            continue
        try:
            A = np.asarray(a.matrix(balance=False).fetch(f"{na}:{int(r.start)}-{int(r.end)}"), float)
            B = np.asarray(b.matrix(balance=False).fetch(f"{nb}:{int(r.start)}-{int(r.end)}"), float)
        except Exception:
            continue
        if A.shape != B.shape or A.shape[0] < 20:
            continue
        iu = np.triu_indices(A.shape[0], 2)
        x, y = A[iu], B[iu]
        ok = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
        if ok.sum() > 100 and x[ok].std() > 0 and y[ok].std() > 0:
            rs.append(pearsonr(x[ok], y[ok]).statistic)
    return np.array(rs)


def main():
    regions = pd.read_csv(sc.REGIONS, sep="\t", names=["chrom", "start", "end"])
    rows = []
    for cell in CELLS:
        d1, d2 = contacts(cell, "rep1"), contacts(cell, "rep2")
        rec = {"cell": cell, "rep1_Mcontacts": round(d1 / 1e6, 1), "rep2_Mcontacts": round(d2 / 1e6, 1)}
        for res in RES:
            p1, p2 = OUT / f"{cell}_rep1_ice_{res}kb.cool", OUT / f"{cell}_rep2_ice_{res}kb.cool"
            if p1.exists() and p2.exists():
                rs = region_rs(cooler.Cooler(str(p1)), cooler.Cooler(str(p2)), regions)
                rec[f"{res}kb"] = round(float(np.median(rs)), 3) if len(rs) else float("nan")
                rec[f"{res}kb_n"] = len(rs)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(W / "replicate_corr.csv", index=False)
    print("rep1 vs rep2, median per-region Pearson on ICE-balanced proper O/E "
          "(413 Akita regions)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
