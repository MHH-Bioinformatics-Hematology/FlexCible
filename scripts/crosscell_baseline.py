"""Measured-map baseline: how well does one cell's real Hi-C predict another's?

For a common set of test regions, load each cell's measured O/E map (2 kb hg38,
same normalization) and compute the per-region Pearson between every pair of cells
over the finite upper-triangle pixels. The off-diagonal entries are exactly "use
cell A's real matrix as the prediction for cell B" -- the bar any model prediction
must beat. Aggregates to an N x N median-per-region correlation matrix.

    python scripts/crosscell_baseline.py \
        --cool gm12878=/.../gm12878_2kb_oe.cool imr90=/.../imr90_2kb_oe.cool \
               k562=/.../k562_2kb_oe.cool \
        --out-dir /mnt/processed/wolffjoa/hic-multiome/runs/crosscell
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_like_hicgan as sc          # REGIONS, match_name
from hicmultiome.genome import match_name


def region_mat(clr, chrom, start, end):
    name = match_name(chrom, list(clr.chromnames))
    if name is None:
        return None
    try:
        M = np.asarray(clr.matrix(balance=False).fetch(f"{name}:{start}-{end}"), float)
    except Exception:
        return None
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cool", nargs="+", required=True,
                    help="cell=path entries, O/E cools at the same res/build")
    ap.add_argument("--regions", default=None,
                    help="BED-like tsv chrom start end; default = Akita test list")
    ap.add_argument("--out-dir", default="/mnt/processed/wolffjoa/hic-multiome/runs/crosscell")
    a = ap.parse_args()

    cells = {}
    for kv in a.cool:
        k, v = kv.split("=", 1)
        cells[k] = cooler.Cooler(v)
    names = list(cells)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    if a.regions:
        reg = pd.read_csv(a.regions, sep="\t", names=["chrom", "start", "end"])
    else:
        reg = pd.read_csv(sc.REGIONS, sep="\t", names=["chrom", "start", "end"])
    print(f"{len(reg)} regions, {len(names)} cells: {names}", flush=True)

    # per-region cross-cell Pearson for every ordered/unordered pair
    rows = []
    for _, r in reg.iterrows():
        mats = {}
        for c, clr in cells.items():
            M = region_mat(clr, r.chrom, int(r.start), int(r.end))
            if M is not None and M.shape[0] == M.shape[1] and M.shape[0] > 50:
                mats[c] = M
        if len(mats) < 2:
            continue
        n = next(iter(mats.values())).shape[0]
        iu = np.triu_indices(n, k=2)
        rec = {"region": f"{r.chrom}:{int(r.start)}-{int(r.end)}"}
        for c1, c2 in itertools.combinations(names, 2):
            if c1 in mats and c2 in mats and mats[c1].shape == mats[c2].shape:
                x, y = mats[c1][iu], mats[c2][iu]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 100 and x[ok].std() > 0 and y[ok].std() > 0:
                    rec[f"{c1}~{c2}"] = float(pearsonr(x[ok], y[ok]).statistic)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "crosscell_per_region.csv", index=False)

    # N x N median matrix
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for c1, c2 in itertools.combinations(names, 2):
        col = f"{c1}~{c2}"
        if col in df:
            m = df[col].median()
            mat.loc[c1, c2] = mat.loc[c2, c1] = round(m, 4)
    mat.to_csv(out / "crosscell_matrix.csv")
    print("\n=== measured-map baseline: median per-region Pearson ===", flush=True)
    print(mat.to_string(), flush=True)
    print("\n(off-diagonal = using row-cell's real map as prediction for column-cell)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 1.4 * len(names) + 1))
        im = ax.imshow(mat.values.astype(float), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat.values[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat.values[i, j] < 0.6 else "black", fontsize=9)
        fig.colorbar(im, label="median per-region Pearson")
        ax.set_title("Measured-map cross-cell baseline (2 kb O/E)")
        fig.tight_layout(); fig.savefig(out / "crosscell_matrix.png", dpi=150)
        print("wrote", out / "crosscell_matrix.png")
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    main()
