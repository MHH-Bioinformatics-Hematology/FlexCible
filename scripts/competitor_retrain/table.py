"""Tables of the retrained C.Origami and Epiphany (registry section 77).

Reads results/hic_measures_chrho/chrho_competitor_retrain_{per_block,summary}.csv (written by
`python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-competitor-retrain`) and writes
to results/competitor_retrain/:

  per_cell.csv         one row per family, grid, cell, source: mean over the test chromosomes of the block
  summary.csv          one row per family, grid, cell group (seen / held-out), source: mean over the cells
                       (each cell has the weight of one, its chromosomes are averaged first), n cells
  vs_retrained.csv     our models minus the retrained competitor, per family, grid and cell group,
                       with the number of cells on which ours is higher
  table.md             the tables in the format of results/benchmark/chrho_25kb_table.md

    python scripts/competitor_retrain/table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IN = REPO / "results/hic_measures_chrho/chrho_competitor_retrain_per_block.csv"
OUT = REPO / "results/competitor_retrain"
MEASURES = ["oe_pearson", "oe_pearson_tile_median", "oe_scc", "insulation_r", "bnd_recall",
            "loop_recall_rank", "apa_ratio", "comp_pref_r", "HiCRep"]
SHORT = {"oe_pearson": "Pearson", "oe_pearson_tile_median": "tile median", "oe_scc": "SCC",
         "insulation_r": "insulation r", "bnd_recall": "boundary recall",
         "loop_recall_rank": "loop recall", "apa_ratio": "APA ratio",
         "comp_pref_r": "compartment r", "HiCRep": "HiCRep"}
TRAIN = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "mcf7", "cd14mono", "pc9"]
TEST = ["k562", "hmec_deep", "huvec_deep"]
GROUP = {"seen cell types, unseen chromosomes": TRAIN, "held-out cell types (K562, HMEC, HUVEC)": TEST}
OURS = ["all12refin_drop_chrho", "all12honest_drop_chrho", "all12refres_drop_chrho"]
BASE = ["consensus", "best_other_map", "replicate", "cascfold"]
FAM = {"chrho_rtco": ("C.Origami", "corigami", ["chr2", "chr3"], "2 Mb band"),
       "chrho_rtep": ("Epiphany", "epiphany", ["chr3", "chr11", "chr17"], "990 kb band")}
LEVEL = 0.01


def label(src: str) -> str:
    for fam, (name, key, _, _) in FAM.items():
        if src.startswith(f"{key}_re_"):
            return f"{name} retrained (our split)"
        if src.startswith(f"{key}_pre_"):
            return f"{name} pretrained (released weights, same inputs)"
    return {"corigami_imr90_8kb": "C.Origami released predictions (authors' IMR-90 tracks)",
            "epiphany_gm12878_10kb": "Epiphany released predictions (authors' GM12878 tracks)"}.get(src, src)


def main() -> None:
    D = pd.read_csv(IN)
    D = D[~D.source.isin(["measured_balanced", "target"])]
    cols = [m for m in MEASURES if m in D.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    D = D.assign(src=D.source.map(label))
    # pretrained rows restricted to the chromosomes the released model left out of its training
    rows = []
    for (set_, res), g in D.groupby(["set", "res"]):
        held = FAM[set_][2]
        sub = g[g.source.str.contains("_pre_") | g.source.isin(["corigami_imr90_8kb", "epiphany_gm12878_10kb"])].copy()
        sub = sub[sub.groupby("source").chrom.transform(lambda c: not set(c) <= set(held))]   # released rows already on own held-out chromosomes only: no second row
        sub["src"] = sub.src + f", own held-out chromosomes ({', '.join(c.replace('chr', '') for c in held)})"
        rows.append(sub[sub.chrom.isin(held)])
    D = pd.concat([D] + rows, ignore_index=True)
    per_cell = (D.groupby(["set", "res", "cell", "src"], dropna=False)[cols].mean()
                .join(D.groupby(["set", "res", "cell", "src"]).chrom.nunique().rename("n_chromosomes"))
                .reset_index())
    per_cell.to_csv(OUT / "per_cell.csv", index=False, float_format="%.4f")

    out = []
    for gname, cells in GROUP.items():
        g = per_cell[per_cell.cell.isin(cells)]
        # sources per cell differ in name (corigami_pre_<cell> -> one label); the mean is over cells
        s = g.groupby(["set", "res", "src"], dropna=False)[cols].mean().reset_index()
        n = g.groupby(["set", "res", "src"]).cell.nunique().rename("n_cells").reset_index()
        s = s.merge(n, on=["set", "res", "src"])
        s["group"] = gname
        out.append(s)
    S = pd.concat(out, ignore_index=True)
    S.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")

    # ours minus the retrained competitor
    cmp_rows = []
    for gname, cells in GROUP.items():
        for (set_, res), g in per_cell[per_cell.cell.isin(cells)].groupby(["set", "res"]):
            rt = g[g.src == f"{FAM[set_][0]} retrained (our split)"].set_index("cell")
            for o in OURS:
                a = g[g.src == o].set_index("cell")
                common = sorted(set(a.index) & set(rt.index))
                if not common:
                    continue
                row = dict(family=FAM[set_][0], res=res, group=gname, ours=o, n_cells=len(common))
                for m in cols:
                    d = (a.loc[common, m] - rt.loc[common, m]).dropna()
                    row[f"{m}_diff"] = d.mean() if len(d) else np.nan
                    row[f"{m}_wins"] = f"{int((d > 0).sum())}/{len(d)}"
                cmp_rows.append(row)
    pd.DataFrame(cmp_rows).to_csv(OUT / "vs_retrained.csv", index=False, float_format="%.4f")

    order_key = lambda s: (0 if s in OURS else 1 if s in BASE else 2 if "retrained" in s else 3, s)   # noqa: E731
    with open(OUT / "table.md", "w") as fh:
        for set_, (name, key, held, band) in FAM.items():
            for res in sorted(S.res.unique()):
                for gname in GROUP:
                    g = S[(S.set == set_) & (S.res == res) & (S.group == gname)]
                    if g.empty:
                        continue
                    ch = "chr2, 3, 11, 14, 17"
                    fh.write(f"\n{name} block ({band}), {res // 1000} kb: {gname}; {int(g.n_cells.max())} cells, "
                             f"mean over cells of the mean over {ch}\n\n")
                    fh.write("| source | " + " | ".join(SHORT[m] for m in cols) + " |\n")
                    fh.write("|---|" + "---|" * len(cols) + "\n")
                    for src in sorted(g.src.unique(), key=order_key):
                        r = g[g.src == src].iloc[0]
                        nc = "" if r.n_cells == g.n_cells.max() else f" (n={int(r.n_cells)})"
                        vals = " | ".join("" if pd.isna(r[m]) else f"{r[m]:.3f}" for m in cols)
                        fh.write(f"| {src}{nc} | {vals} |\n")
    print(f"wrote {OUT}/per_cell.csv, summary.csv, vs_retrained.csv, table.md")


if __name__ == "__main__":
    sys.exit(main())
