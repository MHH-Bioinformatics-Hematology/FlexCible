"""HiCRep SCC and Pearson per Akita test region, in Akita's target space.

Scores Akita's released predictions, Hi-cGAN (submitted) and the measured HFF
map against Akita's GM12878 target (akita_testset_rasters.npz: 413 regions,
448-bin crop of the 2048 bp grid, first two diagonals excluded), plus any of
our predictions given as a region-grid npz (score_like_hicgan.py --save-pred,
keyed chrom:start-end, 512 bins; cropped to the same 448 bins).

SCC: per-diagonal Pearson weighted by N_d * sqrt(var(rank_x / N_d) *
var(rank_y / N_d)) (Yang et al. 2017), all strata up to 917 kb, no extra
smoothing (the maps are smoothed O/E). Pearson: Akita's metric, all finite
pixels of the upper triangle where the target is non-zero.

The full benchmark (more metrics, other resolutions and competitors) is
scripts/benchmark_competitors.py; this script is the stand-alone version of
the first SCC comparison.

    python scripts/scc_akita_regions.py --pred ours=runs/.../akita_native_honest.npz \
        --out results/benchmark/scc_akita_regions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_metrics as bm                                       # noqa: E402
from benchmark_competitors import RASTER                             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="*", default=[], help="name=npz of region-grid predictions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = np.load(RASTER, allow_pickle=True)
    n, off = int(R["side"]), int(R["offset"])
    crop = (512 - n) // 2
    iu = np.triu_indices(n, off)
    preds = {k: np.load(v) for k, v in (x.split("=", 1) for x in a.pred)}
    mask = bm.triu_mask(n, off)

    def dense(v):
        M = np.full((n, n), np.nan)
        M[iu] = v
        return np.where(np.isnan(M), M.T, M)

    rows = []
    for i, reg in enumerate(R["regions"]):
        reg = str(reg)
        T = dense(R["akita_target"][i, :, 1])
        T[T == 0] = np.nan
        maps = {"akita": dense(R["akita_pred"][i, :, 1]), "hicgan": dense(R["hicgan"][i]),
                "hff_map": dense(R["akita_target"][i, :, 0])}
        for k, z in preds.items():
            if reg in z.files:
                maps[k] = np.asarray(z[reg], float)[crop:crop + n, crop:crop + n]
        rec = {"region": reg}
        for k, P in maps.items():
            if not np.isfinite(P).any():
                continue
            rec[f"scc_{k}"] = bm.scc(P, T, mask)
            rec[f"r_{k}"] = bm.pearson(P, T, mask)
        rows.append(rec)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.4f")
    keys = [c[4:] for c in df.columns if c.startswith("scc_")]
    paired = df.dropna(subset=[f"scc_{k}" for k in keys])
    print(f"{len(paired)} regions scored by all of {keys}")
    for m in ("r", "scc"):
        print(m, {k: round(float(paired[f"{m}_{k}"].median()), 3) for k in keys})


if __name__ == "__main__":
    main()
