"""Release candidates against all12one and the dedicated ChIP-only models (registry section 93).

Reads the evaluation written by scripts/one_model_release_eval.py (views all12one_full_<k>kb and
all12one_fullrd_<k>kb) and the earlier evaluation of all12one_<k>kb (section 88) and of the dedicated
no-reference track-dropout models all12honest_drop (2 kb) and all12honest_drop_<k>kb (5, 10, 25 kb),
and writes results/one_model/release.csv (one row per bin size, target set, cell and mode),
release_summary.csv (mean and median over cells) and RELEASE.md (the tables, preceded by the
hand-written results/one_model/RELEASE_narrative.md when it exists).

Modes (columns):
  all12one, all12one_full, all12one_fullrd   consensus map as input (the training protocol)
  full_noref, fullrd_noref                   the same checkpoints with the reference plane at 0
  consensus                                  the consensus map alone as the prediction
  dedicated_chip                             all12honest_drop[_<k>kb], ChIP tracks only

Target sets: test_windows (12 training cells, chromosome-held-out test windows of the cell's own map,
median window Pearson r of log O/E; eval_val_windows.py), akita_413 (the 413 Akita test regions;
crosscell_predict.py; the 12 training cells with their leave-one-out consensus, K562, HMEC and HUVEC
held out with the 12-cell consensus). The dedicated ChIP-only model is a leave-nothing-out model of the
same 12 cells; its scores are on the same windows and regions.

    python scripts/one_model_release.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REF = Path("/mnt/processed/wolffjoa/hic-multiome/runs/multicell_reference")
OUT = REPO / "results/one_model"
RES = [2, 5, 10, 25]
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248", "mcf7",
           "cd14mono", "pc9"]
HELD = ["k562", "hmec", "huvec"]
MODES = ["all12one", "all12one_full", "all12one_fullrd", "full_noref", "fullrd_noref", "consensus",
         "dedicated_chip"]


def view(base, k):
    return REF / (f"{base}_{k}kb")


def dedicated(k):
    return REF / ("all12honest_drop" if k == 2 else f"all12honest_drop_{k}kb")


def j(path, key="oe_median_r"):
    return json.loads(path.read_text())[key] if path.exists() else np.nan


def csv(run, cell, sub="crosscell"):
    f = run / sub / f"predict_{cell}.csv"
    return pd.read_csv(f) if f.exists() else None


def med(df, col):
    return float(df[col].median()) if df is not None and col in df else np.nan


def build():
    rows = []
    for k in RES:
        one, full, frd, ded = view("all12one", k), view("all12one_full", k), view("all12one_fullrd", k), dedicated(k)
        for cell in TRAIN12:
            vals = {"all12one": j(one / f"test_eval_{cell}.json"),
                    "all12one_full": j(full / f"test_eval_{cell}.json"),
                    "all12one_fullrd": j(frd / f"test_eval_{cell}.json"),
                    "full_noref": j(full / f"noref_test_eval_{cell}.json"),
                    "fullrd_noref": j(frd / f"noref_test_eval_{cell}.json"),
                    "consensus": next((v for v in (j(r / f"refalone_test_eval_{cell}.json") for r in (one, frd, full))
                                      if v == v), np.nan),
                    "dedicated_chip": j(ded / f"test_eval_{cell}.json")}
            rows.append(dict(res_kb=k, target_set="test_windows", group="training", cell=cell,
                             n=int(j(frd / f"test_eval_{cell}.json", "n")) if frd.exists() and
                             (frd / f"test_eval_{cell}.json").exists() else 0, **vals))
        for cell in TRAIN12 + HELD:
            grp = "training" if cell in TRAIN12 else "held-out"
            ccol = "base_consensus_loo" if cell in TRAIN12 else "base_consensus12"
            d = {m: csv(r, cell, s) for m, r, s in (
                ("all12one", one, "crosscell"), ("all12one_full", full, "crosscell"),
                ("all12one_fullrd", frd, "crosscell"), ("full_noref", full, "crosscell_noref"),
                ("fullrd_noref", frd, "crosscell_noref"), ("dedicated_chip", ded, "crosscell"))}
            vals = {m: med(x, "model") for m, x in d.items()}
            cons = next((med(x, ccol) for x in (d["all12one_fullrd"], d["all12one"], d["dedicated_chip"])
                         if x is not None and ccol in x), np.nan)
            vals["consensus"] = cons
            n = next((int(x["model"].notna().sum()) for x in d.values() if x is not None), 0)
            rows.append(dict(res_kb=k, target_set="akita_413", group=grp, cell=cell, n=n, **vals))
    return pd.DataFrame(rows).round(4)


def summary(df):
    s = []
    for (k, ts, g), x in df.groupby(["res_kb", "target_set", "group"], sort=False):
        r = dict(res_kb=k, target_set=ts, group=g, n_cells=len(x))
        for stat in ("mean", "median"):
            for m in MODES:
                r[f"{stat}_{m}"] = getattr(x[m], stat)()
        for a, b in (("all12one_fullrd", "all12one"), ("all12one_full", "all12one"),
                     ("all12one_fullrd", "all12one_full"), ("fullrd_noref", "dedicated_chip"),
                     ("full_noref", "dedicated_chip"), ("fullrd_noref", "full_noref")):
            r[f"mean_diff_{a}_minus_{b}"] = (x[a] - x[b]).mean()
            r[f"wins_{a}_over_{b}"] = f"{int((x[a] > x[b]).sum())}/{int(x[[a, b]].notna().all(axis=1).sum())}"
        s.append(r)
    return pd.DataFrame(s).round(4)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(("" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in r) + " |")
    return "\n".join(out)


def tables(sm):
    names = {"all12one": "all12one", "all12one_full": "all12one_full", "all12one_fullrd": "all12one_fullrd",
             "full_noref": "full, plane 0", "fullrd_noref": "fullrd, plane 0", "consensus": "consensus alone",
             "dedicated_chip": "dedicated ChIP-only"}
    out = []
    for stat in ("mean", "median"):
        for ts, g, title in (("test_windows", "training", "12 training cells, test windows"),
                             ("akita_413", "training", "12 training cells, 413 Akita regions"),
                             ("akita_413", "held-out", "held-out K562, HMEC, HUVEC, 413 Akita regions")):
            x = sm[(sm.target_set == ts) & (sm.group == g)]
            t = pd.DataFrame({"bin size (kb)": x.res_kb, "cells": x.n_cells,
                              **{names[m]: x[f"{stat}_{m}"] for m in MODES}})
            out += [f"### {stat.capitalize()} over cells: {title}", "", md(t), ""]
    dif = ["all12one_fullrd_minus_all12one", "all12one_full_minus_all12one", "all12one_fullrd_minus_all12one_full",
           "fullrd_noref_minus_dedicated_chip", "full_noref_minus_dedicated_chip", "fullrd_noref_minus_full_noref"]
    for ts, g, title in (("test_windows", "training", "12 training cells, test windows"),
                         ("akita_413", "held-out", "held-out cells, 413 Akita regions")):
        x = sm[(sm.target_set == ts) & (sm.group == g)]
        t = pd.DataFrame({"bin size (kb)": x.res_kb, **{
            c.replace("mean_diff_", "") + " (wins)": [f"{a:+.3f} ({w})" for a, w in
                                                        zip(x[c], x[c.replace("mean_diff_", "wins_").replace("_minus_", "_over_")])]
            for c in [f"mean_diff_{d}" for d in dif]}})
        out += [f"### Mean difference over cells (cells won): {title}", "", md(t), ""]
    return "\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT / "release.csv", index=False)
    sm = summary(df)
    sm.to_csv(OUT / "release_summary.csv", index=False)
    head = ["# Release candidates all12one_full and all12one_fullrd", "",
            "Generated by `scripts/one_model_release.py` (registry section 93) from the evaluation of "
            "`scripts/one_model_release_eval.py`. `release.csv` holds one row per bin size, target set and cell, "
            "`release_summary.csv` the means and medians over cells and the paired differences.", ""]
    nar = OUT / "RELEASE_narrative.md"
    if nar.exists():
        head += [nar.read_text().rstrip(), ""]
    head += ["Columns: all12one, all12one_full and all12one_fullrd read the consensus map as input; \"full, plane 0\" "
             "and \"fullrd, plane 0\" are the same two checkpoints with the reference plane set to 0 (ChIP tracks "
             "alone); \"consensus alone\" scores the consensus map as the prediction; \"dedicated ChIP-only\" is "
             "all12honest_drop (2 kb) and all12honest_drop_<k>kb (5, 10, 25 kb), trained without any reference. "
             "Pearson r of log O/E; test windows: median over the windows of a cell, then the statistic over cells; "
             "Akita 413: median over the 413 regions.", ""]
    (OUT / "RELEASE.md").write_text("\n".join(head) + "\n" + tables(sm))
    print(tables(sm))


if __name__ == "__main__":
    main()
