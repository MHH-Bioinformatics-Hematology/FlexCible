"""Figure: target | raw prediction | sharpened, for 3 held-out regions per cell.

    python scripts/sharpen_figure.py --variant "unsharp s=2 a=0.5"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sharpen_eval as se


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/mnt/processed/wolffjoa/hic-multiome/runs/sharpen")
    ap.add_argument("--variant", nargs="+", required=True)
    ap.add_argument("--regions", nargs="*", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    df = pd.read_csv(d / "sharpen_per_region.csv")
    z = {c: np.load(d / f"cache_{c}.npz", allow_pickle=True) for c in ("gm12878", "imr90")}
    names = list(z["gm12878"]["names"])
    tune_chr = set(df[(df.split == "tune")].region.str.split(":").str[0])
    stt = se.DiagStats([p for p, n in zip(z["gm12878"]["P"], names) if n.split(":")[0] in tune_chr],
                       [t for t, n in zip(z["gm12878"]["T"], names) if n.split(":")[0] in tune_chr])
    V = se.build_variants(stt)
    if a.regions:
        regs = a.regions
    else:                                       # held-out GM12878 regions at 25/50/75 % of raw r
        s = df[(df.cell == "gm12878") & (df.split == "test") & (df.variant == "identity")].sort_values("pixel_r")
        regs = [s.region.iloc[int(q * (len(s) - 1))] for q in (0.25, 0.5, 0.75)]
    nc = 2 + len(a.variant)
    fig, ax = plt.subplots(2 * len(regs), nc, figsize=(3.4 * nc, 7.0 * len(regs)))
    for row, (cell, reg) in enumerate([(c, r) for r in regs for c in ("gm12878", "imr90")]):
        nm = list(z[cell]["names"])
        if reg not in nm:
            continue
        i = nm.index(reg)
        P = z[cell]["P"][i].astype(float); T = z[cell]["T"][i].astype(float)
        live = np.abs(P).sum(1) > 0
        Qs = [np.where(live[:, None] & live[None, :], V[v](P), np.nan) for v in a.variant]
        sub = df[(df.cell == cell) & (df.region == reg)].set_index("variant")
        for j, (M, title) in enumerate([
                (np.where(T == 0, np.nan, T), f"{cell} target\n{reg}"),
                (np.where(live[:, None] & live[None, :], P, np.nan),
                 f"raw prediction  r={sub.loc['identity', 'pixel_r']:.3f}")]
                + [(Q, f"{v}  r={sub.loc[v, 'pixel_r']:.3f}") for Q, v in zip(Qs, a.variant)]):
            ax[row, j].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax[row, j].set_title(title, fontsize=9)
            ax[row, j].set_xticks([]); ax[row, j].set_yticks([])
    fig.tight_layout()
    out = d / "sharpen_examples.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
