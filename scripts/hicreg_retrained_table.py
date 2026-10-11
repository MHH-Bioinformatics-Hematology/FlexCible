"""Tables of HiC-Reg retrained on our chromosome split (registry section 63).

Reads results/hic_measures_chrho/chrho_hicregre_per_block.csv, written by

    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-hicreg-retrained

and writes to results/benchmark/

  hicreg_retrained_per_chromosome.csv  one row per cell, grid, chromosome and source, all measures
  hicreg_retrained_summary.csv         mean over the chromosomes of a group: "chr14+chr17" (blocks that
                                       also hold the released predictions), "chr2+chr3+chr11" (retrained
                                       and ours only) and "all five"
  hicreg_retrained_leak.csv            within-chromosome leak: CV score minus the cross-chromosome score
                                       (released CV against released cross-chromosome, CV protocol on
                                       our pipeline against retrained), per measure
  hicreg_retrained_table.md            the same as markdown, one table per cell and grid

Nothing is recomputed: every number is read from the per-block csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "results/hic_measures_chrho/chrho_hicregre_per_block.csv"
OUT = REPO / "results/benchmark"
MEASURES = ["oe_pearson", "oe_pearson_tile_median", "oe_scc", "insulation_r", "bnd_recall",
            "loop_recall_rank", "apa_ratio", "comp_pref_r", "HiCRep"]
SHORT = {"oe_pearson": "Pearson", "oe_pearson_tile_median": "tile median", "oe_scc": "SCC",
         "insulation_r": "insulation r", "bnd_recall": "boundary recall",
         "loop_recall_rank": "loop recall", "apa_ratio": "APA ratio",
         "comp_pref_r": "compartment r", "HiCRep": "HiCRep"}
OURS = {"all12refin_drop_chrho": "ours: all12refin_drop_chrho",
        "all12honest_drop_chrho": "ours: all12honest_drop_chrho (ChIP only)",
        "all12refres_drop_chrho": "ours: all12refres_drop_chrho"}
BASE = {"consensus": "consensus of the other cells", "best_other_map": "best other measured map"}


def label(src: str) -> str | None:
    if src in OURS:
        return OURS[src]
    if src in BASE:
        return BASE[src]
    k = src.split("_")[0]
    return {"hicregrewindow": "HiC-Reg retrained (our split), WINDOW",
            "hicregremulticell": "HiC-Reg retrained (our split), MULTICELL",
            "hicregrewindowcvours": "HiC-Reg CV protocol on our pipeline (within-chromosome)",
            "hicregrewindowsmall": "HiC-Reg retrained, WINDOW, 4M-pair training (size control)",
            "hicregrecrosscell": "HiC-Reg retrained, cross-cell (model of the other cell)",
            "hicregwindowcrosschrom": "HiC-Reg cross-chromosome (released), WINDOW",
            "hicregmulticellcrosschrom": "HiC-Reg cross-chromosome (released), MULTICELL",
            "hicregwindowcv": "HiC-Reg CV (released, within-chromosome)"}.get(k)


ORDER = list(OURS.values()) + list(BASE.values()) + [
    "HiC-Reg retrained (our split), WINDOW", "HiC-Reg retrained (our split), MULTICELL",
    "HiC-Reg retrained, cross-cell (model of the other cell)",
    "HiC-Reg retrained, WINDOW, 4M-pair training (size control)",
    "HiC-Reg cross-chromosome (released), WINDOW", "HiC-Reg cross-chromosome (released), MULTICELL",
    "HiC-Reg CV protocol on our pipeline (within-chromosome)", "HiC-Reg CV (released, within-chromosome)"]


def main() -> None:
    P = pd.read_csv(IN)
    P = P[P.kind.isin(["model", "cool", "competitor"])].copy()
    P["label"] = P.source.map(label)
    P = P[P.label.notna()]
    # the MULTICELL rows come from the blocks of set chrho_hicregmc, every other source from
    # chrho_hicregre (the models and baselines are scored in both; keep the first)
    is_mc = P.source.str.startswith("hicregremulticell")
    P = pd.concat([P[is_mc & (P.set == "chrho_hicregmc")], P[~is_mc & (P.set == "chrho_hicregre")]])
    cols = [m for m in MEASURES if m in P.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    P[["cell", "res", "chrom", "source", "label"] + cols].sort_values(["cell", "res", "chrom", "label"]) \
        .to_csv(OUT / "hicreg_retrained_per_chromosome.csv", index=False, float_format="%.4f")

    groups = {"chr14+chr17": ["chr14", "chr17"], "chr2+chr3+chr11": ["chr2", "chr3", "chr11"],
              "all five": ["chr2", "chr3", "chr11", "chr14", "chr17"]}
    rows = []
    for gname, chs in groups.items():
        Q = P[P.chrom.isin(chs)]
        for (cell, res, lab), g in Q.groupby(["cell", "res", "label"]):
            if set(g.chrom) != set(chs):
                continue
            rows.append(dict(group=gname, cell=cell, res=res, label=lab, n_chromosomes=len(chs),
                             **{m: g[m].mean() for m in cols}))
    S = pd.DataFrame(rows)
    S["o"] = S.label.map({l: i for i, l in enumerate(ORDER)})
    S = S.sort_values(["group", "cell", "res", "o"]).drop(columns="o")
    S.to_csv(OUT / "hicreg_retrained_summary.csv", index=False, float_format="%.4f")

    leak = []
    pairs = [("HiC-Reg CV (released, within-chromosome)", "HiC-Reg cross-chromosome (released), WINDOW",
              "released CV minus released cross-chromosome (WINDOW)"),
             ("HiC-Reg CV protocol on our pipeline (within-chromosome)",
              "HiC-Reg retrained (our split), WINDOW",
              "CV protocol minus retrained cross-chromosome (our pipeline, WINDOW)"),
             ("HiC-Reg CV (released, within-chromosome)", "HiC-Reg retrained (our split), WINDOW",
              "released CV minus retrained cross-chromosome"),
             ("HiC-Reg retrained (our split), WINDOW", "HiC-Reg cross-chromosome (released), WINDOW",
              "retrained minus released cross-chromosome (WINDOW)")]
    G = S[S.group == "chr14+chr17"]
    for (cell, res), g in G.groupby(["cell", "res"]):
        g = g.set_index("label")
        for a, b, name in pairs:
            if a in g.index and b in g.index:
                leak.append(dict(cell=cell, res=res, comparison=name,
                                 **{m: g.loc[a, m] - g.loc[b, m] for m in cols}))
    pd.DataFrame(leak).to_csv(OUT / "hicreg_retrained_leak.csv", index=False, float_format="%.4f")

    with open(OUT / "hicreg_retrained_table.md", "w") as fh:
        hdr = "| source | " + " | ".join(SHORT[m] for m in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        for gname in groups:
            for cell in ("gm12878", "k562"):
                for res in (25000, 10000, 5000):
                    g = S[(S.group == gname) & (S.cell == cell) & (S.res == res)]
                    if g.empty:
                        continue
                    fh.write(f"\n{cell.upper()} {res // 1000} kb, {gname} (mean over the chromosomes; "
                             f"identical pixels within a chromosome block)\n\n{hdr}")
                    for _, r in g.iterrows():
                        vals = " | ".join("" if pd.isna(r[m]) else f"{r[m]:.3f}" for m in cols)
                        fh.write(f"| {r.label} | {vals} |\n")
    print(f"wrote {OUT}/hicreg_retrained_*.csv and hicreg_retrained_table.md")


if __name__ == "__main__":
    sys.exit(main())
