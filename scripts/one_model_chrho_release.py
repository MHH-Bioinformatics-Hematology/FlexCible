"""The release twin all12one_fullrd_chrho on the held-out-chromosome competitor benchmark (registry 101).

Reads the tables of

    python scripts/hic_measures.py --stage chrho_score chrho_report --chrho-tag release --chrho-models \\
        all12refin_drop_chrho all12honest_drop_chrho all12refres_drop_chrho all12one_chrho \\
        all12one_chrho_noref all12one_fullrd_chrho all12one_fullrd_chrho_noref
    (the same with --chrho-25kb-comps and with --chrho-res 5000)

(results/hic_measures_chrho/chrho[_25kb|_5kb]_release_{per_block,summary}.csv; the blocks hold the same
pixels for all models they list) and

  1. checks that the stored rows of the older models (per-resolution models and the all12one_chrho pair)
     are reproduced in the new blocks (same pixels, same numbers);
  2. appends the rows of the two new models to results/hic_measures_chrho/chrho[_25kb|_5kb]_{per_block,summary}.csv
     (rows added only, old rows unchanged);
  3. writes results/one_model/chrho_benchmark_release.csv (one row per block, chromosome and source) and
     chrho_benchmark_release.md: per block the mean over the held-out chromosomes of Pearson, SCC,
     insulation r, boundary recall, loop recall and compartment r, differences of the release twin to the
     other models, per-chromosome win counts and the competitors that exceed the twin in any measure of any
     block.

Nothing is recomputed: every number is read from the csv files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import one_model_chrho as om  # noqa: E402

IN, OUT = om.IN, om.OUT
REL = ["all12one_fullrd_chrho", "all12one_fullrd_chrho_noref"]
ONE = ["all12one_chrho", "all12one_chrho_noref"]
PERRES = ["all12refin_drop_chrho", "all12honest_drop_chrho", "all12refres_drop_chrho"]
STORED = PERRES + ONE                     # rows that exist in the stored tables and must be reproduced
BASELINES = ["consensus", "best_other_map", "replicate", "cascfold"]
MEAS, SHORT, KEY, KEYS, TAGS = om.MEAS, om.SHORT, om.KEY, om.KEYS, om.TAGS
LABEL = {"all12one_fullrd_chrho": "release twin, consensus input",
         "all12one_fullrd_chrho_noref": "release twin, ChIP only",
         "all12one_chrho": "one model (no dropout), consensus input",
         "all12one_chrho_noref": "one model (no dropout), ChIP only",
         "all12refin_drop_chrho": "per-resolution, consensus input",
         "all12honest_drop_chrho": "per-resolution, ChIP only",
         "all12refres_drop_chrho": "per-resolution, consensus residual"}
FIG = om.FIG_BLOCKS


def check_old(tag, nb, ob):
    a = nb[nb.source.isin(STORED)].set_index(KEY)
    b = ob[ob.source.isin(STORED)].set_index(KEY)
    idx = a.index.intersection(b.index)
    rows = []
    for m in MEAS:
        d = (a.loc[idx, m] - b.loc[idx, m]).abs()
        rows.append(dict(tag=tag, measure=m, rows=int(d.notna().sum()), max_abs_diff=float(d.max())))
    return pd.DataFrame(rows)


def merge(tag):
    nb = pd.read_csv(IN / f"{tag}_release_per_block.csv")
    ns = pd.read_csv(IN / f"{tag}_release_summary.csv")
    ob = pd.read_csv(IN / f"{tag}_per_block.csv")
    os_ = pd.read_csv(IN / f"{tag}_summary.csv")
    chk = check_old(tag, nb, ob)
    ob = ob[~ob.source.isin(REL)]
    os_ = os_[~os_.source.isin(REL)]
    n_old_b, n_old_s = len(ob), len(os_)
    mb = pd.concat([ob, nb[nb.source.isin(REL)].reindex(columns=ob.columns)], ignore_index=True)
    ms = pd.concat([os_, ns[ns.source.isin(REL)].reindex(columns=os_.columns)], ignore_index=True)
    mb.to_csv(IN / f"{tag}_per_block.csv", index=False, float_format="%.4f")
    ms.to_csv(IN / f"{tag}_summary.csv", index=False, float_format="%.4f")
    print(tag, "rows added:", len(mb) - n_old_b, len(ms) - n_old_s)
    return nb, ns, chk


def bname(set_, cell, res):
    return om.block_name(set_, cell, res)


def main() -> int:
    per, summ, chks = [], [], []
    for tag in TAGS:
        f = IN / f"{tag}_release_per_block.csv"
        if not f.exists():
            print("missing", f, file=sys.stderr)
            continue
        nb, ns, chk = merge(tag)
        per.append(nb.assign(table=tag))
        summ.append(ns.assign(table=tag))
        chks.append(chk)
    chk = pd.concat(chks, ignore_index=True)
    chk.to_csv(OUT / "chrho_benchmark_release_oldmodel_check.csv", index=False, float_format="%.6f")
    P = pd.concat(per, ignore_index=True)
    S = pd.concat(summ, ignore_index=True)
    P = P[~P.source.isin(om.IGNORE)].drop_duplicates(KEY)
    S = S[~S.source.isin(om.IGNORE)].drop_duplicates(KEYS)
    cols = [m for m in MEAS if m in P.columns]
    P[["set", "cell", "res", "chrom", "source", "kind", "flag"] + cols] \
        .sort_values(["res", "set", "cell", "chrom", "source"], ascending=[False, True, True, True, True]) \
        .to_csv(OUT / "chrho_benchmark_release.csv", index=False, float_format="%.4f")

    L = ["# The release twin on the held-out-chromosome benchmark", "",
         "Mean over the held-out chromosomes of each block (the value the benchmark figure reads). Release twin = "
         "`all12one_fullrd_chrho` (one network for 2, 5, 10 and 25 kb; 13 marks and methylation, cohesin-fraction "
         "row, depletion augmentation, reference dropout 0.3; trained without chr2, 3, 11, 14, 17 of every cell), "
         "with the consensus as reference input and, from the same checkpoint, ChIP only (reference plane 0, "
         "methylation given). All sources of a block are scored on the same pixels. Per-chromosome values: "
         "`chrho_benchmark_release.csv`.", ""]
    hdr = "| source | " + " | ".join(SHORT[m] for m in cols) + " | chromosomes |\n|---|" + "---|" * (len(cols) + 1)
    order = {s: i for i, s in enumerate(REL + ONE + PERRES + BASELINES)}
    blocks = []
    for res in (10000, 25000, 5000):
        for (set_, cell), g in S[S.res == res].groupby(["set", "cell"], sort=True):
            blocks.append((res, set_, cell, g))
            chroms = sorted(P[(P.set == set_) & (P.cell == cell) & (P.res == res)].chrom.unique())
            L += [f"## {bname(set_, cell, res)}", "", hdr]
            g = g.assign(o=g.source.map(order).fillna(99)).sort_values(["o", "source"])
            for _, r in g.iterrows():
                L.append(f"| {LABEL.get(r.source, r.source)} | "
                         + " | ".join("" if pd.isna(r[m]) else f"{r[m]:.3f}" for m in cols)
                         + f" | {int(r.n_chromosomes)} |")
            L += ["", f"Chromosomes: {', '.join(chroms)}.", ""]

    # differences and per-chromosome win counts
    pairs = [(REL[0], PERRES[0]), (REL[1], PERRES[1]), (REL[0], ONE[0]), (REL[1], ONE[1]), (REL[1], REL[0])]
    L += ["## Differences of the release twin", "",
          "Mean over the held-out chromosomes of the block; positive = the first model is higher. "
          "Wins = held-out chromosomes (of all blocks of the table family) on which the first model is higher.", ""]
    Pm = P.set_index(["set", "cell", "res", "chrom", "source"])
    for a_, b_ in pairs:
        L += [f"### {LABEL[a_]} minus {LABEL[b_]}", "",
              "| block | " + " | ".join(SHORT[m] for m in cols) + " |", "|---|" + "---|" * len(cols)]
        rows = []
        for res, set_, cell, g in blocks:
            a = g[g.source == a_]
            b = g[g.source == b_]
            if a.empty or b.empty:
                continue
            d = {m: float(a[m].iloc[0] - b[m].iloc[0]) for m in cols}
            rows.append((res, set_, cell, d))
            L.append(f"| {bname(set_, cell, res)} | " + " | ".join(f"{d[m]:+.3f}" for m in cols) + " |")
        for res in (10000, 25000, 5000):
            sel = [d for r_, s_, c_, d in rows if r_ == res and (s_, c_) in FIG]
            if sel:
                L.append(f"| mean of the {len(sel)} figure blocks, {res // 1000} kb | "
                         + " | ".join(f"{np.nanmean([d[m] for d in sel]):+.3f}" for m in cols) + " |")
        # win counts per resolution over chromosomes of the figure blocks
        for res in (10000, 25000, 5000):
            wins = {m: [0, 0] for m in cols}
            for (set_, cell) in FIG:
                sub = P[(P.set == set_) & (P.cell == cell) & (P.res == res)]
                for ch in sub.chrom.unique():
                    try:
                        va = Pm.loc[(set_, cell, res, ch, a_)]
                        vb = Pm.loc[(set_, cell, res, ch, b_)]
                    except KeyError:
                        continue
                    for m in cols:
                        if np.isfinite(va[m]) and np.isfinite(vb[m]):
                            wins[m][1] += 1
                            wins[m][0] += int(va[m] > vb[m])
            if wins[cols[0]][1]:
                L.append(f"| chromosome wins, figure blocks, {res // 1000} kb | "
                         + " | ".join(f"{w[0]} of {w[1]}" for w in wins.values()) + " |")
        L.append("")

    # competitors above the twin
    COMP = set(P[P.kind == "competitor"].source)
    L += ["## Competitors that exceed the release twin", "",
          "Competitor block mean above the twin's block mean in a measure (the twin with the consensus and "
          "ChIP only listed separately); the per-resolution counterpart is given for comparison.", ""]
    for tw in REL:
        L += [f"### {LABEL[tw]}", "", "| block | competitor | measure | competitor | twin | difference | per-resolution model |",
              "|---|---|---|---|---|---|---|"]
        pr = PERRES[0] if tw == REL[0] else PERRES[1]
        n = 0
        for res, set_, cell, g in blocks:
            t = g[g.source == tw]
            if t.empty:
                continue
            for _, c in g[g.source.isin(COMP)].iterrows():
                for m in cols:
                    if np.isfinite(c[m]) and np.isfinite(t[m].iloc[0]) and c[m] > t[m].iloc[0]:
                        p_ = g[g.source == pr]
                        pv = f"{p_[m].iloc[0]:.3f}" if not p_.empty else ""
                        L.append(f"| {bname(set_, cell, res)} | {c.source} | {SHORT[m]} | {c[m]:.3f} | "
                                 f"{t[m].iloc[0]:.3f} | {c[m] - t[m].iloc[0]:+.3f} | {pv} |")
                        n += 1
        if not n:
            L.append("| none | | | | | | |")
        L.append("")
    L += ["## Check of the older models", "",
          "Largest absolute difference between the stored rows of the per-resolution models and the all12one_chrho "
          "pair and their rows in the blocks scored together with the release twin "
          "(`chrho_benchmark_release_oldmodel_check.csv`).", "",
          "| table | measure | rows | max abs difference |", "|---|---|---|---|"]
    for _, r in chk.iterrows():
        L.append(f"| {r.tag} | {r.measure} | {r.rows} | {r.max_abs_diff:.2g} |")
    (OUT / "chrho_benchmark_release.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "chrho_benchmark_release.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
