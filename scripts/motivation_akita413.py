"""Akita's 413 test regions: the measured maps of the other Akita cell types and their mean as the
prediction for the target cell, against Akita (and Hi-cGAN), on the same pixels (registry section 103).

Setting of registry 73 (stage akita of scripts/structure_scores_vs_akita.py, definitions and worker
unchanged): 413 regions of 448 bins of 2048 bp, target GM12878 (head 1 of akita_testset_rasters.npz,
Akita and Hi-cGAN predictions as stored) and HFF (head 0; Akita prediction and target from block
A_hff_2kb). The measured O/E maps of the other cells of the Akita data set (block arrays
map_<cell> of A_gm12878_2kb / A_hff_2kb: for GM12878 HFF, H1, IMR-90, HCT116; for HFF GM12878, H1,
IMR-90, HCT116) are scored as predictions, with their mean ("consensus of the other four"). One pixel
mask per region: upper triangle, >= 2 bins off the diagonal, finite in the target and in every
method except Hi-cGAN (as in section 73).

    python scripts/motivation_akita413.py --threads 6
Output: results/motivation/akita413_per_region.csv.gz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import structure_scores_vs_akita as S  # noqa: E402

OUT = REPO / "results/motivation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import benchmark_blocks as bb
    from flexcible import bioscore as bs
    Z = np.load(S.RASTER, allow_pickle=True)
    side, off = int(Z["side"]), int(Z["offset"])
    regs = [str(r) for r in Z["regions"]]
    cells = [str(c) for c in Z["cells"]]
    blk = {"gm12878": np.load(S.BENCH / "blocks/A_gm12878_2kb.npz"),
           "hff": np.load(S.BENCH / "blocks/A_hff_2kb.npz")}
    others = {"gm12878": ["hff", "h1", "imr90", "hct116"], "hff": ["gm12878", "h1", "imr90", "hct116"]}
    rows = []
    for target in ("gm12878", "hff"):
        B = blk[target]
        k = [c.lower() for c in cells].index(target)
        jobs = []
        for i, reg in enumerate(regs):
            if a.limit and i >= a.limit:
                break
            if f"{reg}|target" not in B.files:
                continue
            dense = lambda v: bs.dense_from_triu(v, side, off).astype(float)        # noqa: E731
            T = dense(Z["akita_target"][i, :, k])
            T[T == 0] = np.nan
            maps = {"akita": dense(Z["akita_pred"][i, :, k])}
            if target == "gm12878":
                h = dense(Z["hicgan"][i])
                maps["hicgan"] = h if np.isfinite(h).any() else None
            ok = True
            om = {}
            for c in others[target]:
                key = f"{reg}|map_{c}"
                if key not in B.files:
                    ok = False
                    break
                om[c] = B[key].astype(float)
            if not ok:
                continue
            maps["consensus_other4"] = np.nanmean(np.stack(list(om.values())), 0)
            for c, M in om.items():
                maps[f"other_{c}"] = M
            chrom, se = reg.split(":")
            s = int(se.split("-")[0])
            mask = np.triu(np.ones((side, side), bool), 2) & np.isfinite(T)
            for m, P in maps.items():
                if m != "hicgan" and P is not None:
                    mask &= np.isfinite(P)
            if mask.sum() < 500:
                continue
            b0 = s // 2048 + (512 - side) // 2
            bnd, loops = bb.ref_sets({"cell": target, "res": 2048}, reg, b0, side)
            jobs.append((reg, b0, T, maps, mask, bnd, loops))
        S.log(f"{target}: {len(jobs)} of {len(regs)} regions assembled")
        S._JOBS[:] = jobs
        with mp.get_context("fork").Pool(a.threads) as pool:
            res = pool.map(S._akita_worker, range(len(jobs)), chunksize=4)
        for rs, _ in res:
            for r in rs:
                r["target"] = target
                rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "akita413_per_region.csv.gz", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
