#!/usr/bin/env python
"""Collect the cohesin-fix evaluation (registry section 55) into CSVs and markdown tables.

Reads the shared result tables (reference_summary.csv on the share, results/perturb_validate,
results/washout, results/aml, results/cohesin_fix/diagnose_*.csv) and writes
results/cohesin_fix/results_*.csv plus results/cohesin_fix/results_tables.md (the tables pasted
into RESULTS.md). deg = all12deg_refres_drop, cohfix = all12cohfix_refres_drop (row + augmentation +
shared contact scale), cohfixna = all12cohfixna_refres_drop (row + shared scale, no augmentation).
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OUT = REPO / "results/cohesin_fix"
FAM = {"deg": "all12deg_refres_drop", "cohfix": "all12cohfix_refres_drop",
       "cohfixna": "all12cohfixna_refres_drop"}
RES = (10, 25)


def run(f, k):
    return f"{FAM[f]}_{k}kb"


def intact():
    d = pd.read_csv(HM / "runs/multicell_reference/reference_summary.csv")
    rows = []
    for k in RES:
        for cells in ("12 training cells", "held-out cells"):
            ref = None
            for f in FAM:
                r = d[(d.run == run(f, k)) & (d.cells == cells)].iloc[0]
                if f == "deg":
                    ref = r
                rows.append({"res_kb": k, "cells": cells, "model": f,
                             "median_model": r.median_model,
                             "delta_vs_deg": r.median_model - ref.median_model,
                             "median_test_windows": r.median_test_windows,
                             "delta_test_windows_vs_deg": (r.median_test_windows - ref.median_test_windows
                                                           if pd.notna(r.median_test_windows) else None),
                             "median_reference_alone": r.median_reference_alone,
                             "beats_reference": r.beats_reference})
    return pd.DataFrame(rows)


def perturb():
    d = pd.read_csv(REPO / "results/perturb_validate/perturb_summary.csv")
    keep = [run(f, k) for f in FAM for k in RES]
    d = d[d.model.isin(keep)]
    cols = ["subset", "experiment", "model", "perturbation", "change_pearson_median", "change_pearson_n",
            "pred_abs_change_mean", "meas_abs_change_mean", "loops_n", "loops_meas_median_change",
            "loops_pred_median_change", "loops_meas_frac_weaker", "loops_pred_frac_weaker",
            "loops_change_spearman", "boundary_ins_meas_median_change", "boundary_ins_pred_median_change"]
    return d[cols].sort_values(["experiment", "subset", "perturbation", "model"])


def washout():
    d = pd.read_csv(REPO / "results/washout/washout_summary.csv")
    keep = list(FAM.values())
    return d[d.model.isin(keep)].sort_values(["res_kb", "subset", "model"])


def aml():
    d = pd.read_csv(REPO / "results/aml/perturb_summary.csv")
    d["model"] = d.method.str.split(":").str[0]
    d["variant"] = d.method.str.split(":").str[1]
    d = d[d.model.isin([run(f, k) for f in FAM for k in RES])]
    return d.drop(columns="method").sort_values(["res_kb", "variant", "model"])


def aml_score():
    d = pd.read_csv(REPO / "results/aml/summary.csv")
    d = d[d.method.str.contains("|".join(run(f, k) for f in FAM for k in RES))]
    return d.sort_values(["target", "res_kb", "method"])


def diagnose():
    rows = []
    for f in FAM:
        for k in RES:
            p = OUT / f"diagnose_{run(f, k)}.csv"
            if not p.exists():
                continue
            g = pd.read_csv(p).groupby("scenario", sort=False).median(numeric_only=True)
            g.insert(0, "res_kb", k)
            g.insert(0, "model", f)
            g["spread_pct_of_measured"] = 100 * g.pred_sd / g.meas_sd
            rows.append(g.reset_index())
    return pd.concat(rows)


def main():
    I, P, W, A, S, G = intact(), perturb(), washout(), aml(), aml_score(), diagnose()
    I.to_csv(OUT / "results_intact.csv", index=False)
    P.to_csv(OUT / "results_perturb.csv", index=False)
    W.to_csv(OUT / "results_washout.csv", index=False)
    A.to_csv(OUT / "results_aml_perturb.csv", index=False)
    S.to_csv(OUT / "results_aml_score.csv", index=False)
    G.to_csv(OUT / "results_diagnose.csv", index=False)
    print(I.round(4).to_string())
    print(G[G.scenario.isin(["aux_chip", "absent", "wt_f0", "wt_f0.26", "wt_f0.5", "wt_f0.75", "absent_f0",
                             "absent_f0.5", "aux_RAD21_only"])][
        ["model", "res_kb", "scenario", "pearson", "pred_sd", "meas_sd", "spread_pct_of_measured"]]
        .round(3).to_string())


if __name__ == "__main__":
    main()
