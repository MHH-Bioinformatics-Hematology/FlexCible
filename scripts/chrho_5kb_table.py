"""5 kb table of the chromosome-held-out comparison (registry section 58).

Copy of chrho_25kb_table.py with the file tag and resolution changed.

Reads the files written by

    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-res 5000

(results/hic_measures_chrho/chrho_5kb_summary.csv and chrho_5kb_per_block.csv)
and writes, to results/benchmark/:

  chrho_5kb_competitors.csv   one row per block and source: the mean over the
                               held-out chromosomes of the block of every measure
                               (identical pixels within a block)
  chrho_5kb_vs_competitor.csv one row per block and competitor: the difference
                               of each of our models to the competitor, in the
                               mean over the block's chromosomes, the number of
                               chromosomes on which our model is higher, and a
                               verdict (ahead / level / behind; level = within
                               +-0.01 of the measure)
  chrho_5kb_table.md          the same as a markdown table

Nothing is recomputed: every number is read from the two input csv files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "results/hic_measures_chrho"
OUT = REPO / "results/benchmark"
MEASURES = ["oe_pearson", "oe_pearson_tile_median", "oe_scc", "insulation_r", "bnd_recall",
            "loop_recall_rank", "apa_ratio", "comp_pref_r", "HiCRep"]
SHORT = {"oe_pearson": "Pearson", "oe_pearson_tile_median": "tile median", "oe_scc": "SCC",
         "insulation_r": "insulation r", "bnd_recall": "boundary recall",
         "loop_recall_rank": "loop recall", "apa_ratio": "APA ratio",
         "comp_pref_r": "compartment r", "HiCRep": "HiCRep"}
OURS = ["all12refin_drop_chrho", "all12honest_drop_chrho", "all12refres_drop_chrho",
        "all12one_chrho", "all12one_chrho_noref"]
BASE = ["consensus", "best_other_map", "replicate", "cascfold"]
IGNORE = {"measured_balanced"}
LEVEL = 0.01
BLOCK_LABEL = {
    ("chrho_corigami", "imr90"): "C.Origami block: IMR-90, chr2, chr3 (1.9 Mb band, resampled from 8192 bp)",
    ("chrho_epiphany", "gm12878"): "Epiphany block: GM12878, chr3, 11, 17 (990 kb band, resampled from 10 kb)",
    ("chrho_hicreg", "gm12878"): "HiC-Reg block: GM12878, chr14, chr17 (1 Mb band)",
    ("chrho_hicreg", "k562"): "HiC-Reg block: K562, chr14, chr17 (1 Mb band)",
    ("chrho_hicgan", "gm12878"): "Hi-cGAN block: GM12878, chr2, chr14 (1.9 Mb band)",
}


def main() -> None:
    S = pd.read_csv(IN / "chrho_5kb_summary.csv")
    P = pd.read_csv(IN / "chrho_5kb_per_block.csv")
    S = S[(S.res == 5000) & ~S.source.isin(IGNORE)]
    P = P[(P.res == 5000) & ~P.source.isin(IGNORE)]
    cols = [m for m in MEASURES if m in S.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    S = S.copy()
    S["block"] = [BLOCK_LABEL.get((a, b), f"{a} {b}") for a, b in zip(S.set, S.cell)]
    S[["block", "set", "cell", "res", "source", "flag", "n_chromosomes"] + cols] \
        .sort_values(["block", "source"]).to_csv(OUT / "chrho_5kb_competitors.csv",
                                                 index=False, float_format="%.4f")

    comp_rows = []
    for (set_, cell), g in P.groupby(["set", "cell"]):
        comps = sorted(g[g.kind == "competitor"].source.unique())
        for comp in comps:
            for ours in OURS:
                a = g[g.source == ours].set_index("chrom")
                b = g[g.source == comp].set_index("chrom")
                chroms = sorted(set(a.index) & set(b.index))
                if not chroms:
                    continue
                row = {"block": BLOCK_LABEL.get((set_, cell), f"{set_} {cell}"), "set": set_,
                       "cell": cell, "competitor": comp, "ours": ours,
                       "chromosomes": ",".join(chroms)}
                for m in cols:
                    d = (a.loc[chroms, m] - b.loc[chroms, m]).dropna()
                    row[f"{m}_diff"] = d.mean() if len(d) else float("nan")
                    row[f"{m}_wins"] = f"{int((d > 0).sum())}/{len(d)}"
                    if len(d) and m == "apa_ratio":          # 1 is ideal, not larger
                        row[f"{m}_verdict"] = "see values"
                    elif len(d):
                        mean = d.mean()
                        row[f"{m}_verdict"] = ("ahead" if mean > LEVEL else
                                               "behind" if mean < -LEVEL else "level")
                comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(OUT / "chrho_5kb_vs_competitor.csv", index=False,
                                   float_format="%.4f")

    with open(OUT / "chrho_5kb_table.md", "w") as fh:
        hdr = "| source | " + " | ".join(SHORT[m] for m in cols) + " |\n"
        hdr += "|---|" + "---|" * len(cols) + "\n"
        for block, g in S.groupby("block", sort=False):
            fh.write(f"\n{block}; mean over {int(g.n_chromosomes.max())} chromosomes\n\n{hdr}")
            order = {s: i for i, s in enumerate(OURS + BASE)}
            g = g.assign(o=g.source.map(order).fillna(99)).sort_values(["o", "source"])
            for _, r in g.iterrows():
                vals = " | ".join("" if pd.isna(r[m]) else f"{r[m]:.3f}" for m in cols)
                fh.write(f"| {r.source} | {vals} |\n")
    print(f"wrote {OUT}/chrho_5kb_competitors.csv, chrho_5kb_vs_competitor.csv, chrho_5kb_table.md")


if __name__ == "__main__":
    sys.exit(main())
