"""The one model with held-out chromosomes on the held-out-chromosome competitor benchmark (registry 91).

Reads the tables of

    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-models \\
        all12refin_drop_chrho all12honest_drop_chrho all12refres_drop_chrho all12one_chrho all12one_chrho_noref
    (the same with --chrho-25kb-comps and with --chrho-res 5000)

(results/hic_measures_chrho/chrho[_25kb|_5kb]_onemodel_{per_block,summary}.csv; the blocks hold the same
pixels for all models they list) and

  1. checks that the older models score as before in the new blocks (same pixels, same numbers);
  2. appends the rows of the two new models (all12one_chrho, all12one_chrho_noref) to the tables
     results/hic_measures_chrho/chrho[_25kb|_5kb]_{per_block,summary}.csv, the old rows stay;
  3. writes results/one_model/chrho_benchmark.csv (one row per block, chromosome and source) and
     chrho_benchmark.md (per block the mean over the held-out chromosomes of the block, as the
     manuscript's benchmark figure reads them, and the difference of the one model to the
     per-resolution models).

Nothing is recomputed: every number is read from the csv files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "results/hic_measures_chrho"
OUT = REPO / "results/one_model"
NEW = ["all12one_chrho", "all12one_chrho_noref"]
OLD = ["all12refin_drop_chrho", "all12honest_drop_chrho"]
OLD_ALL = OLD + ["all12refres_drop_chrho"]
TAGS = ["chrho", "chrho_25kb", "chrho_5kb"]
MEAS = ["oe_pearson", "oe_scc", "insulation_r", "bnd_recall", "loop_recall_rank", "comp_pref_r"]
SHORT = {"oe_pearson": "Pearson", "oe_scc": "SCC", "insulation_r": "insulation r",
         "bnd_recall": "boundary recall", "loop_recall_rank": "loop recall", "comp_pref_r": "compartment r"}
LABEL = {"all12one_chrho": "one model, consensus input", "all12one_chrho_noref": "one model, ChIP only",
         "all12refin_drop_chrho": "per-resolution, consensus input",
         "all12honest_drop_chrho": "per-resolution, ChIP only"}
BLOCK = {"chrho_corigami": "C.Origami block", "chrho_epiphany": "Epiphany block",
         "chrho_hicgan": "Hi-cGAN block", "chrho_hicreg": "HiC-Reg block"}
KEY = ["set", "cell", "res", "chrom", "source"]
KEYS = ["set", "cell", "res", "source", "flag"]
IGNORE = {"measured_balanced"}
FIG_BLOCKS = [("chrho_corigami", "imr90"), ("chrho_epiphany", "gm12878"),
              ("chrho_hicgan", "gm12878"), ("chrho_hicgan", "k562")]   # the blocks of the benchmark figure


def check_old(tag: str, new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Largest absolute difference per measure between the stored rows of the older models and their
    rows in the new blocks (same models, same pixels when the common mask is unchanged)."""
    a = new[new.source.isin(OLD_ALL)].set_index(KEY)
    b = old[old.source.isin(OLD_ALL)].set_index(KEY)
    idx = a.index.intersection(b.index)
    rows = []
    for m in MEAS:
        d = (a.loc[idx, m] - b.loc[idx, m]).abs()
        rows.append(dict(tag=tag, measure=m, rows=int(d.notna().sum()), max_abs_diff=float(d.max())))
    return pd.DataFrame(rows)


def merge(tag: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nb = pd.read_csv(IN / f"{tag}_onemodel_per_block.csv")
    ns = pd.read_csv(IN / f"{tag}_onemodel_summary.csv")
    ob = pd.read_csv(IN / f"{tag}_per_block.csv")
    os_ = pd.read_csv(IN / f"{tag}_summary.csv")
    chk = check_old(tag, nb, ob)
    ob = ob[~ob.source.isin(NEW)]
    os_ = os_[~os_.source.isin(NEW)]
    add_b = nb[nb.source.isin(NEW)].reindex(columns=ob.columns)
    add_s = ns[ns.source.isin(NEW)].reindex(columns=os_.columns)
    mb = pd.concat([ob, add_b], ignore_index=True)
    ms = pd.concat([os_, add_s], ignore_index=True)
    mb.to_csv(IN / f"{tag}_per_block.csv", index=False, float_format="%.4f")
    ms.to_csv(IN / f"{tag}_summary.csv", index=False, float_format="%.4f")
    return nb, ns, chk


def block_name(set_: str, cell: str, res: int) -> str:
    return f"{BLOCK.get(set_, set_)}, {cell}, {res // 1000} kb"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per, summ, chks = [], [], []
    for tag in TAGS:
        if not (IN / f"{tag}_onemodel_per_block.csv").exists():
            print(f"missing {tag}_onemodel_per_block.csv", file=sys.stderr)
            continue
        nb, ns, chk = merge(tag)
        per.append(nb.assign(table=tag))
        summ.append(ns.assign(table=tag))
        chks.append(chk)
    chk = pd.concat(chks, ignore_index=True)
    chk.to_csv(OUT / "chrho_benchmark_oldmodel_check.csv", index=False, float_format="%.6f")
    P = pd.concat(per, ignore_index=True)
    S = pd.concat(summ, ignore_index=True)
    P = P[~P.source.isin(IGNORE)]
    S = S[~S.source.isin(IGNORE)]
    # a block of one grid occurs in several tables (the 25 kb blocks in chrho and chrho_25kb): keep one
    P = P.drop_duplicates(KEY)
    S = S.drop_duplicates(KEYS)
    cols = [m for m in MEAS if m in P.columns]
    P[["set", "cell", "res", "chrom", "source", "kind", "flag"] + cols] \
        .sort_values(["res", "set", "cell", "chrom", "source"], ascending=[False, True, True, True, True]) \
        .to_csv(OUT / "chrho_benchmark.csv", index=False, float_format="%.4f")

    lines = ["# One model on the held-out-chromosome benchmark", "",
             "Mean over the held-out chromosomes of each block (the value the benchmark figure reads); "
             "one model = `all12one_chrho` (one network for 2, 5, 10 and 25 kb, trained without chr2, "
             "3, 11, 14, 17 of every cell), consensus input and, from the same checkpoint, ChIP only "
             "(reference plane 0); per-resolution = `all12refin_drop_chrho` and `all12honest_drop_chrho`. "
             "Per-chromosome values: `chrho_benchmark.csv`.", ""]
    hdr = "| source | " + " | ".join(SHORT[m] for m in cols) + " | chromosomes |\n|---|" + "---|" * (len(cols) + 1)
    for res in (10000, 25000, 5000):
        for (set_, cell), g in S[S.res == res].groupby(["set", "cell"], sort=True):
            lines += [f"## {block_name(set_, cell, res)}", "", hdr]
            order = {s: i for i, s in enumerate(NEW + OLD + ["all12refres_drop_chrho", "consensus", "cascfold"])}
            g = g.assign(o=g.source.map(order).fillna(99)).sort_values(["o", "source"])
            chroms = sorted(P[(P.set == set_) & (P.cell == cell) & (P.res == res)].chrom.unique())
            for _, r in g.iterrows():
                nm = LABEL.get(r.source, r.source)
                lines.append(f"| {nm} | " + " | ".join("" if pd.isna(r[m]) else f"{r[m]:.3f}" for m in cols)
                             + f" | {int(r.n_chromosomes)} |")
            lines.append("")
            lines.append(f"Chromosomes: {', '.join(chroms)}.")
            lines.append("")
    # differences of the one model to the per-resolution models, per block and over the figure's blocks
    lines += ["## Difference, one model minus per-resolution model", "",
              "Mean over the held-out chromosomes of the block; positive = the one model is higher.", ""]
    for pair in (("all12one_chrho", "all12refin_drop_chrho"), ("all12one_chrho_noref", "all12honest_drop_chrho")):
        lines += [f"### {LABEL[pair[0]]} minus {LABEL[pair[1]]}", "",
                  "| block | " + " | ".join(SHORT[m] for m in cols) + " |", "|---|" + "---|" * len(cols)]
        rows = []
        for res in (10000, 25000, 5000):
            for (set_, cell), g in S[S.res == res].groupby(["set", "cell"], sort=True):
                a = g[g.source == pair[0]]
                b = g[g.source == pair[1]]
                if a.empty or b.empty:
                    continue
                d = {m: float(a[m].iloc[0] - b[m].iloc[0]) for m in cols}
                rows.append((res, set_, cell, d))
                lines.append(f"| {block_name(set_, cell, res)} | "
                             + " | ".join(f"{d[m]:+.3f}" for m in cols) + " |")
        for res in (10000, 25000, 5000):
            sel = [d for r_, s_, c_, d in rows if r_ == res and (s_, c_) in FIG_BLOCKS]
            if sel:
                lines.append(f"| mean of the {len(sel)} figure blocks, {res // 1000} kb | "
                             + " | ".join(f"{np.nanmean([d[m] for d in sel]):+.3f}" for m in cols) + " |")
        lines.append("")
    lines += ["## Check of the older models", "",
              "Largest absolute difference between the stored rows of the per-resolution models and "
              "their rows in the blocks scored together with the one model (`chrho_benchmark_oldmodel_check.csv`).",
              "", "| table | measure | rows | max abs difference |", "|---|---|---|---|"]
    for _, r in chk.iterrows():
        lines.append(f"| {r.tag} | {r.measure} | {r.rows} | {r.max_abs_diff:.2g} |")
    (OUT / "chrho_benchmark.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "chrho_benchmark.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
