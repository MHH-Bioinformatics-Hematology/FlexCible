"""Replicate ceiling for the release checkpoint all12one_fullrd (registry section 98).

The statistic of `replicate_ceiling_table.py` (registry section 49): median per-region Pearson on the 413 Akita
regions of the reference-input model divided by the ceiling sqrt(r) (single-replicate target) or
sqrt(2r / (1 + r)) (merged target, Spearman-Brown), r = agreement of the two biological replicates, and the
deepest intact pair for GM12878 and K562. The ceilings and the replicate agreements are the existing ones
(`results/replicates/`, unchanged); only the model column changes:

  paper    all12refin_drop_<k>kb, `crosscell/predict_<cell>.csv` (consensus input, dedicated per size)
  release  all12one_fullrd_<k>kb, `crosscell/predict_<cell>.csv` (consensus input; leave-one-out for the training
           cells, consensus12 for the held-out cells; section 93)
  noref    all12one_fullrd_<k>kb, `crosscell_noref/predict_<cell>.csv` (reference plane at 0)

    python scripts/one_model_remaining_ceiling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replicate_ceiling_table as rt                                    # noqa: E402

RUNS = rt.RUNS
OUT = ROOT / "results/one_model/remaining/replicate_ceiling"
RELEASE = "all12one_fullrd"


def csv(base_dir, sub, cell):
    p = base_dir / sub / f"predict_{cell}.csv"
    return pd.read_csv(p) if p.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = pd.read_csv(ROOT / "results/replicates/replicate_corr.csv").set_index("cell")
    deep = pd.read_csv(ROOT / "results/replicates/replicate_corr_intact_deep.csv").set_index(["cell", "experiment"])
    rows = []
    for cell in rt.TRAIN + rt.HELD:
        for res in rt.RES:
            r = dict(cell=cell, name=rt.NAMES[cell], group="held-out" if cell in rt.HELD else "training",
                     res_kb=res)
            old = csv(rt.run_dir("all12refin_drop", res), "crosscell", cell)
            rel = csv(RUNS / f"{RELEASE}_{res}kb", "crosscell", cell)
            nor = csv(RUNS / f"{RELEASE}_{res}kb", "crosscell_noref", cell)
            r["paper_model"] = old.model.median() if old is not None else np.nan
            r["release_model"] = rel.model.median() if rel is not None else np.nan
            r["noref_model"] = nor.model.median() if nor is not None else np.nan
            r["n_regions_release"] = int(rel.model.notna().sum()) if rel is not None else 0
            cc = rt.cons_col(rel) if rel is not None else None
            r["consensus_alone"] = rel[cc].median() if cc else np.nan
            rc = rt.REP_CELL.get(cell)
            if rc in rep.index:
                rr = float(rep.loc[rc, f"{res}kb"])
                r["replicate_r"] = rr
                r["ceiling"] = np.sqrt(rr) if cell in rt.SINGLE_TARGET else np.sqrt(rt.sb(rr))
            else:
                r.update(replicate_r=np.nan, ceiling=np.nan)
            if cell in rt.DEEP_PAIR:
                rd = float(deep.loc[(cell, rt.DEEP_PAIR[cell]), f"{res}kb"])
                r["deep_replicate_r"] = rd
                r["ceiling_deep_pair"] = np.sqrt(rt.sb(rd))
            else:
                r.update(deep_replicate_r=np.nan, ceiling_deep_pair=np.nan)
            for k in ("paper_model", "release_model", "noref_model"):
                r["ratio_" + k.replace("_model", "")] = r[k] / r["ceiling"]
                r["ratio_deep_" + k.replace("_model", "")] = r[k] / r["ceiling_deep_pair"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "replicate_ceiling_table.csv", index=False, float_format="%.4f")

    # summary over the cells with a ceiling, as in replicate_ceiling_summary.csv
    srows = []
    for res in rt.RES:
        for g in ("training", "held-out"):
            s = df[(df.res_kb == res) & (df.group == g)]
            sc = s[s.ceiling.notna()]
            for tag in ("paper", "release", "noref"):
                col = {"paper": "paper_model", "release": "release_model", "noref": "noref_model"}[tag]
                ratio = sc["ratio_" + tag]
                srows.append(dict(res_kb=res, group=g, model=tag, n_cells=len(s), median_model=s[col].median(),
                                  n_with_ceiling=len(sc), median_ceiling=sc.ceiling.median(),
                                  median_model_where_ceiling=sc[col].median(), median_ratio=ratio.median(),
                                  min_ratio=ratio.min(), max_ratio=ratio.max(),
                                  median_gap=(sc.ceiling - sc[col]).median(),
                                  n_ge_95pct=int((ratio >= 0.95).sum()),
                                  n_model_ge_replicate_r=int((sc[col] >= sc.replicate_r).sum())))
    sm = pd.DataFrame(srows)
    sm.to_csv(OUT / "replicate_ceiling_summary.csv", index=False, float_format="%.4f")

    f = lambda x: "NA" if pd.isna(x) else f"{x:.3f}"                      # noqa: E731
    L = ["Model accuracy (median per-region Pearson, 413 Akita regions), ceiling and fraction of the ceiling. "
         "paper = all12refin_drop (dedicated per size), release = all12one_fullrd with the consensus, "
         "noref = all12one_fullrd with the reference plane at 0.", ""]
    for res in rt.RES:
        L += [f"{res} kb", "", "| cell | set | paper | release | noref | consensus | replicate r | ceiling | "
              "paper/ceiling | release/ceiling | noref/ceiling |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for cell in rt.TRAIN + rt.HELD:
            x = df[(df.cell == cell) & (df.res_kb == res)].iloc[0]
            L.append(f"| {x['name']} | {'H' if cell in rt.HELD else 'train'} | {f(x.paper_model)} | "
                     f"{f(x.release_model)} | {f(x.noref_model)} | {f(x.consensus_alone)} | {f(x.replicate_r)} | "
                     f"{f(x.ceiling)} | {f(x.ratio_paper)} | {f(x.ratio_release)} | {f(x.ratio_noref)} |")
        L.append("")
    L += ["Summary over the cells with a ceiling (training: GM12878, IMR-90; held-out: K562, HMEC, HMEC deep, HUVEC, "
          "HUVEC deep)", "", "| res | set | model | cells with ceiling | median model | median ceiling | median fraction | "
          "min | max | >= 95% | model >= replicate r |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, s in sm.iterrows():
        L.append(f"| {s.res_kb} | {s.group} | {s.model} | {s.n_with_ceiling} | {f(s.median_model_where_ceiling)} | "
                 f"{f(s.median_ceiling)} | {f(s.median_ratio)} | {f(s.min_ratio)} | {f(s.max_ratio)} | "
                 f"{s.n_ge_95pct}/{s.n_with_ceiling} | {s.n_model_ge_replicate_r}/{s.n_with_ceiling} |")
    L += ["", "Deepest intact replicate pair as ceiling (GM12878 ENCSR965OVH, K562 ENCSR676BEP)", "",
          "| cell | res | paper | release | noref | deep pair r | ceiling | paper/ceiling | release/ceiling | "
          "noref/ceiling |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in df.dropna(subset=["ceiling_deep_pair"]).iterrows():
        L.append(f"| {x['name']} | {x.res_kb} | {f(x.paper_model)} | {f(x.release_model)} | {f(x.noref_model)} | "
                 f"{f(x.deep_replicate_r)} | {f(x.ceiling_deep_pair)} | {f(x.ratio_deep_paper)} | "
                 f"{f(x.ratio_deep_release)} | {f(x.ratio_deep_noref)} |")
    (OUT / "replicate_ceiling_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
