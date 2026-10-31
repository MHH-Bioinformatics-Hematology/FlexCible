#!/usr/bin/env python
"""Tables of results/one_model/PERTURB_METH.md (registry section 95): the cohesin perturbations
(RAD21 degron, washout, CTCF degron, U-937 STAG2 knockout; section 55) and the methylation proxy
(section 78) of the release candidate all12one_fullrd next to the models of those sections.

deg = all12deg_refres_drop (RAD21 degron pair in training, no fraction row: the base model of section 55),
cohfix = all12cohfix_refres_drop (fraction row + depletion augmentation, section 55),
release = all12one_fullrd (one checkpoint, 13 marks + Meth, fraction row, augmentation, reference
dropout 0.3). Reads the committed tables of results/ and writes the markdown tables to stdout
(and to results/one_model/perturb_meth_tables.md).

    python scripts/perturb_meth_report.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
MODELS = {"deg": "all12deg_refres_drop", "cohfix": "all12cohfix_refres_drop", "release": "all12one_fullrd"}
RES = (10, 25)


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def n(x):
    return "n/a" if x is None or pd.isna(x) else str(int(x))


def pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def perturb_table() -> pd.DataFrame:
    old = pd.read_csv(R / "perturb_validate/perturb_summary.csv")
    new = pd.read_csv(R / "perturb_validate/all12one_fullrd_perturb_summary.csv")
    return pd.concat([old, new], ignore_index=True)


def row(D, subset, exp, pert, key, k):
    q = D[(D.subset == subset) & (D.experiment == exp) & (D.perturbation == pert) &
          (D.model == f"{MODELS[key]}_{k}kb")]
    return q.iloc[0] if len(q) else None


PERT_LABEL = {"minus_RAD21_f0": "RAD21 absent, fraction 0", "minus_RAD21_f0.5": "RAD21 absent, fraction 0.5",
              "measured_chip": "measured auxin ChIP, fraction 0.26", "minus_RAD21": "RAD21 absent, fraction 1 (ChIP only)",
              "minus_CTCF": "CTCF absent", "minus_H3K36me3": "H3K36me3 absent (control)"}


def degron_tables(D, exp, subset, perts, title):
    """One table per resolution: rows perturbation, columns the three models; change-map statistics
    (median [q25, q75], mean, sd, n regions; Pearson, SCC and Spearman) and the loop and boundary measures."""
    out = []
    for k in RES:
        out.append(f"#### {title}, {k} kb\n")
        rows = []
        for p in perts:
            for key in MODELS:
                r = row(D, subset, exp, p, key, k)
                if r is None:
                    continue
                rows.append([PERT_LABEL.get(p, p), key,
                             f"{f(r.change_pearson_median)} [{f(r.change_pearson_q25)}, {f(r.change_pearson_q75)}]",
                             f(r.change_pearson_mean), f(r.change_pearson_sd), n(r.change_pearson_n),
                             f"{f(r.change_scc_median)} [{f(r.change_scc_q25)}, {f(r.change_scc_q75)}]",
                             f(r.change_spearman_median),
                             f"{f(r.pred_abs_change_mean)} / {f(r.meas_abs_change_mean)}"])
        out.append(md(rows, ["perturbation", "model", "Pearson median [q25, q75]", "Pearson mean", "Pearson sd",
                             "regions", "SCC median [q25, q75]", "Spearman median",
                             "mean abs change pred / meas"]))
        out.append("")
        rows = []
        for p in perts:
            for key in MODELS:
                r = row(D, subset, exp, p, key, k)
                if r is None:
                    continue
                rows.append([PERT_LABEL.get(p, p), key, n(r.loops_n), f(r.loops_meas_median_change),
                             f(r.loops_pred_median_change), pct(r.loops_meas_frac_weaker),
                             pct(r.loops_pred_frac_weaker), f(r.loops_change_spearman),
                             f(r.apa_U, 2) if pd.notna(r.apa_U) else "n/a",
                             f"{f(r.apa_T, 2)} / {f(r.apa_P0, 2)} / {f(r.apa_P1, 2)}" if pd.notna(r.apa_T) else "n/a",
                             n(r.boundaries_n), f(r.boundary_ins_meas_median_change),
                             f(r.boundary_ins_pred_median_change), f(r.boundary_change_spearman)])
        out.append(md(rows, ["perturbation", "model", "loops", "loop change measured (median log O/E)",
                             "predicted", "weaker measured", "weaker predicted", "Spearman of change",
                             "APA untreated", "APA treated meas / pred full / pred pert", "boundaries",
                             "insulation change measured", "predicted", "Spearman of change"]))
        out.append("")
    return "\n".join(out)


def washout_table() -> str:
    old = pd.read_csv(R / "washout/washout_summary.csv")
    new = pd.read_csv(R / "washout/all12one_fullrd_washout_summary.csv")
    W = pd.concat([old, new], ignore_index=True)
    out = []
    for k in RES:
        rows = []
        for subset in ("all", "even_chromosomes"):
            for key, fam in MODELS.items():
                q = W[(W.model == fam) & (W.res_kb == k) & (W.subset == subset)]
                if not len(q):
                    continue
                r = q.iloc[0]
                rows.append([subset, key, n(r.loops), pct(r.meas_frac_weaker_auxin),
                             pct(r.pred_frac_weaker_rad21_auxin), f(r.meas_median_change_auxin),
                             f(r.pred_median_change_rad21_auxin),
                             " / ".join(f(r[f"pred_recovery_f{x}"], 2) for x in (0.25, 0.5, 0.75)),
                             " / ".join(f(r[f"meas_recovery_{t}min"], 2) for t in (20, 40, 60, 180)),
                             " / ".join(f(r[f"matching_f_{t}min"], 2) for t in (20, 40, 60, 180))])
        out.append(f"#### Washout, {k} kb\n")
        out.append(md(rows, ["loops from", "model", "loops", "weaker measured (auxin)", "weaker predicted",
                             "median change measured", "median change predicted (RAD21 auxin)",
                             "predicted recovery at f 0.25 / 0.5 / 0.75",
                             "measured recovery 20 / 40 / 60 / 180 min",
                             "matching f 20 / 40 / 60 / 180 min"]))
        out.append("")
    return "\n".join(out)


KO_LABEL = {"u937_ko_absent": "RAD21 and SMC3 absent, fraction 1", "u937_ko_absent_f0": "RAD21 and SMC3 absent, fraction 0",
            "u937_ko_absent_f0.5": "RAD21 and SMC3 absent, fraction 0.5", "u937_ko_f0.5": "wild-type ChIP, fraction 0.5",
            "u937_ko_f0.75": "wild-type ChIP, fraction 0.75", "u937_ko_ctcfonly": "cohesin only at CTCF peaks",
            "u937_ko_half": "cohesin enrichment halved"}


def ko_table() -> str:
    old = pd.read_csv(R / "aml/perturb_summary.csv")
    p = R / "aml/perturb_summary_all12one_fullrd.csv"
    new = pd.read_csv(p) if p.exists() else old.iloc[0:0]
    A = pd.concat([old, new[~new.method.str.startswith("clones_")]], ignore_index=True)
    A["model"] = A.method.str.split(":").str[0]
    A["variant"] = A.method.str.split(":").str[1]
    out = []
    for k in RES:
        rows = []
        for v in KO_LABEL:
            for key, fam in MODELS.items():
                q = A[(A.res_kb == k) & (A.model == f"{fam}_{k}kb") & (A.variant == v)]
                if not len(q):
                    continue
                r = q.iloc[0]
                rows.append([KO_LABEL[v], key, n(r.n_regions), f(r.delta_pearson_median), f(r.delta_scc_median),
                             n(r.loops_n), f(r.loops_meas_mean_change), f(r.loops_pred_mean_change),
                             f(r.loops_change_pearson), f(r.loops_change_spearman), pct(r.loops_sign_agreement),
                             f"{f(r.loops_short_meas_mean_change)} / {f(r.loops_short_pred_mean_change)}",
                             f"{f(r.loops_long_meas_mean_change)} / {f(r.loops_long_pred_mean_change)}",
                             n(r.boundaries_n),
                             f(r.boundaries_change_pearson)])
        clones = A[(A.res_kb == k) & A.method.str.startswith("clones_")]
        out.append(f"#### U-937 STAG2 knockout, {k} kb\n")
        out.append(md(rows, ["input", "model", "regions", "change-map Pearson median", "SCC median", "loops",
                             "loop change measured (mean log2)", "predicted", "Pearson of loop change",
                             "Spearman", "sign agreement", "short loops (<200 kb) meas / pred",
                             "long loops meas / pred", "boundaries", "Pearson of insulation change"]))
        if len(clones):
            out.append("\nClone-pair ceiling of the change map (Pearson median): " +
                       ", ".join(f(x) for x in clones.delta_pearson_median) + ".")
        out.append("")
    return "\n".join(out)


def aml_intact_table() -> str:
    S = pd.read_csv(R / "aml/summary.csv")
    rows = []
    for k in RES:
        for tgt, inp, lab in (("molm13", "molm13", "MOLM-13"), ("u937_wt_c1", "u937", "U-937 wild type, clone 1"),
                              ("u937_ko_c1", "u937", "U-937 STAG2 KO clone 1, wild-type input")):
            for key, fam in MODELS.items():
                q = S[(S.target == tgt) & (S.res_kb == k) & (S.method == f"model:{fam}_{k}kb:{inp}")]
                if len(q):
                    r = q.iloc[0]
                    rows.append([k, lab, key, f(r.pearson_median), f(r.scc_median), f(r.pearson_mean), n(r.n_regions)])
    return md(rows, ["kb", "map", "model", "Pearson median", "SCC median", "Pearson mean", "regions"])


# ------------------------------------------------------------------------- methylation
SETS = ["minus_CTCF", "minus_cohesin", "minus_active", "minus_genebody", "minus_repressive", "cum2", "cum3",
        "ctcf_only", "seq_only"]
SET_LABEL = {"minus_CTCF": "minus CTCF", "minus_cohesin": "minus cohesin", "minus_active": "minus active",
             "minus_genebody": "minus gene body", "minus_repressive": "minus repressive", "cum2": "cum2",
             "cum3": "cum3", "ctcf_only": "CTCF only", "seq_only": "sequence only"}


def cell(q):
    if q is None or not len(q):
        return "n/a"
    r = q.iloc[0]
    rf = "-" if pd.isna(r.rf) else f"{r.rf:.2f} ({r.rf_lo:.2f} to {r.rf_hi:.2f})"
    return f"{f(r.mean_loss)} [{f(r.loss_lo)}, {f(r.loss_hi)}] / {f(r.mean_gain)} [{f(r.gain_lo)}, {f(r.gain_hi)}] / {rf}"


def meth_tables() -> str:
    old = pd.read_csv(R / "meth_proxy/replacement_pooled.csv")
    p = R / "meth_proxy/all12one_fullrd_replacement_pooled.csv"
    if not p.exists():
        return "(methylation predictions not finished)\n"
    new = pd.read_csv(p)
    P = pd.concat([old, new], ignore_index=True)
    P = P[P.metric == "pearson"]
    out = []
    for k in RES:
        cons = {10: "all12refres_drop_meth_10kb", 25: "all12refres_drop_motifmeth_25kb"}[k]
        cols = [("honest", f"all12honest_drop_meth_{k}kb", "dedicated, no consensus"),
                ("consensus", cons, "dedicated, consensus" + (" (motif planes)" if k == 25 else "")),
                ("release", f"all12one_fullrd_{k}kb", "release, consensus plane"),
                ("release_noref", f"all12one_fullrd_noref_{k}kb", "release, plane at 0")]
        for grp in ("held-out", "training WGBS", "training RRBS"):
            out.append(f"#### Methylation proxy, {k} kb, {grp} cells (Pearson; loss [95% CI] / gain [95% CI] / replacement fraction (95% CI))\n")
            rows = []
            for st in SETS:
                r = [SET_LABEL[st]]
                for _, m, _ in cols:
                    q = P[(P.model == m) & (P.cell_group == grp) & (P.set == st)]
                    r.append(cell(q))
                rows.append(r)
            ncell = []
            for _, m, _ in cols:
                q = P[(P.model == m) & (P.cell_group == grp)]
                ncell.append(f"{int(q.n_cells.iloc[0])} cells, noise {f(q.noise.iloc[0], 4)}" if len(q) else "n/a")
            rows.append(["cells, seed noise"] + ncell)
            fq = []
            for _, m, _ in cols:
                q = P[(P.model == m) & (P.cell_group == grp) & (P.set == "full")]
                fq.append(f"gain on the full set {cell(q).split(' / ')[1]}" if len(q) else "n/a")
            rows.append(["Meth added to all 13 tracks"] + fq)
            out.append(md(rows, ["set"] + [c[2] for c in cols]))
            out.append("")
    return "\n".join(out)


def meth_other_metrics() -> str:
    p = R / "meth_proxy/all12one_fullrd_replacement_pooled.csv"
    if not p.exists():
        return ""
    P = pd.read_csv(p)
    out = []
    for k in RES:
        for m, lab in ((f"all12one_fullrd_{k}kb", "consensus plane"), (f"all12one_fullrd_noref_{k}kb", "plane at 0")):
            rows = []
            for st in ["minus_CTCF", "minus_cohesin", "ctcf_only", "seq_only"]:
                for grp in ("held-out", "training all"):
                    r = [SET_LABEL[st], grp]
                    for met in ("pearson", "scc", "insulation", "boundary_f1", "loop_enrichment", "compartment",
                                "eigenvector"):
                        q = P[(P.model == m) & (P.cell_group == grp) & (P.set == st) & (P.metric == met)]
                        r.append(f"{f(q.mean_loss.iloc[0])} / {f(q.mean_gain.iloc[0])}" if len(q) else "n/a")
                    rows.append(r)
            out.append(f"#### Other measures, {k} kb, release with {lab} (loss without the tracks / gain from Meth)\n")
            out.append(md(rows, ["set", "cells", "Pearson", "SCC", "insulation", "boundary F1", "loop enrichment",
                                 "compartment", "eigenvector"]))
            out.append("")
    return "\n".join(out)


def meth_scores() -> str:
    p = R / "meth_proxy/all12one_fullrd_replacement_by_cell.csv"
    if not p.exists():
        return ""
    C = pd.read_csv(p)
    C = C[C.metric == "pearson"]
    out = []
    for k in RES:
        rows = []
        for cellname in C.cell.unique():
            r = [cellname]
            for m in (f"all12one_fullrd_{k}kb", f"all12one_fullrd_noref_{k}kb"):
                for st in ("full", "seq_only"):
                    q = C[(C.model == m) & (C.cell == cellname) & (C.set == st)]
                    r.append(f"{f(q.score_without_meth.iloc[0])} / {f(q.score_with_meth.iloc[0])}" if len(q) else "n/a")
            rows.append(r)
        out.append(f"#### Median Pearson of the 413 regions per cell, {k} kb (without Meth / with Meth)\n")
        out.append(md(rows, ["cell", "13 tracks, consensus plane", "no ChIP, consensus plane", "13 tracks, plane at 0",
                             "no ChIP, plane at 0"]))
        out.append("")
    return "\n".join(out)


def main():
    D = perturb_table()
    parts = [
        "### RAD21 degron (HCT116, even chromosomes)\n",
        degron_tables(D, "rad21", "even_chromosomes",
                      ["minus_RAD21_f0", "minus_RAD21_f0.5", "measured_chip", "minus_RAD21", "minus_CTCF"],
                      "RAD21 degron"),
        "### CTCF degron (HCT116, all chromosomes, never trained)\n",
        degron_tables(D, "ctcf", "all", ["minus_CTCF"], "CTCF degron"),
        "### Washout time course\n", washout_table(),
        "### U-937 STAG2 knockout\n", ko_table(),
        "### AML patient-line maps, intact prediction (413 regions)\n", aml_intact_table(), "",
        "### Methylation proxy\n", meth_tables(), meth_other_metrics(), meth_scores()]
    text = "\n".join(parts)
    (R / "one_model/perturb_meth_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
