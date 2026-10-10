"""What the measured cohesin perturbations look like (registry section 55), CPU only.

Per region of the 413 Akita test regions (10 kb): the change map (perturbed minus
control, log O/E) of the HCT116 RAD21 degron (6 h auxin) and of the U-937 STAG2
knockout (clone 1, against wild type clone 1): spread, mean change by genomic
distance, and the per-region Pearson correlation between the two change maps
(HCT116 and U-937 share hg38 coordinates).

    CUDA_VISIBLE_DEVICES= nice -n 10 python scripts/cohesin_fix_measured.py [--res 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import aml_eval as ae                                                # noqa: E402
import benchmark_metrics as bm                                      # noqa: E402
import perturb_validate as pv                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=10000)
    ap.add_argument("--out", default=str(REPO / "results/cohesin_fix/measured_changes.csv"))
    a = ap.parse_args()
    res = a.res
    R = pv.regions()
    cw = cooler.Cooler(str(ae.map_path("u937_wt_c1", res)))
    ck = cooler.Cooler(str(ae.map_path("u937_ko_c1", res)))
    cache: dict = {}
    rows = []
    bins = [(2, 10), (10, 25), (25, 50), (50, 10 ** 6)]          # in bins of 10 kb: <=100k ...
    for r in R.itertuples():
        reg = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
        W, K = ae.fetch_region(cw, reg, cache), ae.fetch_region(ck, reg, cache)
        b0, b1 = r.start // res, -(-r.end // res)
        U = pv._fetch(pv.oe_path("rad21", "untreated", res), r.chrom, b0, b1)
        T = pv._fetch(pv.oe_path("rad21", "treated", res), r.chrom, b0, b1)
        if W is None or K is None or U is None or T is None or W.shape != K.shape:
            continue
        n = min(W.shape[0], U.shape[0])
        W, K, U, T = W[:n, :n], K[:n, :n], U[:n, :n], T[:n, :n]
        dK, dD = K - W, T - U
        mk = bm.triu_mask(n, 2) & np.isfinite(dK) & np.isfinite(dD)
        if mk.sum() < 200:
            continue
        dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        rec = {"region": reg, "sd_ko": dK[mk].std(), "sd_degron": dD[mk].std(),
               "pearson_ko_vs_degron": bm.pearson(dK, dD, mk)}
        for lo, hi in bins:
            m2 = mk & (dist >= lo) & (dist < hi)
            rec[f"mean_ko_{lo}_{hi}"] = dK[m2].mean() if m2.any() else np.nan
            rec[f"mean_degron_{lo}_{hi}"] = dD[m2].mean() if m2.any() else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(len(df), "regions")
    print(df.drop(columns="region").median().round(3).to_string())


if __name__ == "__main__":
    main()
