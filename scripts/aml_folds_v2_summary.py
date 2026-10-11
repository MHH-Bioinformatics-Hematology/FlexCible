"""Pooled summary of the rebuilt AML-inclusive cross-validation (registry section 61).

Folds 10, 11, 12 (patient-disjoint over the ten AML patients with Hi-C and
CUT&Tag, healthy donors removed, same-donor pairs kept together) are scored by
scripts/aml_train_eval.py --fold N; this script pools the ten test patients and
tests the section 36 claims. Intervals are donor-level percentile bootstraps
(patients resampled with replacement, median of the resample).

    python scripts/aml_folds_v2_summary.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REMOTE = Path("/mnt/processed/wolffjoa/hic-multiome/runs/aml_train_remote")
METRICS = ["pearson", "scc", "insulation_corr", "loop_enrich_corr", "compartment_corr",
           "pearson_specific", "resid_vs_aml_train_mean", "resid_vs_aml_train_mean_specific"]
MODELS = ["aml_refin", "aml_honest", "base_honest", "base_refin@aml", "base_refin@c12",
          "aml_refin@hspc"]
BASES = {"ref_other_patients": "base_aml_train_mean", "ref_lopo_all": "base_aml_lopo_all",
         "ref_hspc4": "base_hspc", "ref_hspc_train2": "base_hspc_train_mean",
         "ref_consensus12": "base_consensus12"}
FEAT = ["aml_refin|marks", "aml_refin|none", "aml_honest|marks", "base_refin@aml|marks",
        "base_honest|marks"]


def boot(x, rng, nb):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), (nb, len(x)))
    m = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_new(folds):
    ps, ws, fs = [], [], []
    for f in folds:
        d = REMOTE / f"fold{f}"
        ps.append(pd.read_csv(d / f"eval_f{f}_patients.csv"))
        ws.append(pd.read_csv(d / f"eval_f{f}_features_windows.csv").assign(fold=f))
    p = pd.concat(ps, ignore_index=True)
    p = p[p.group == "patient"].copy()
    p["patient"] = p.cell.str.replace("amlxu", "", regex=False)
    return p, pd.concat(ws, ignore_index=True)


def wide(p, res):
    """one row per patient: columns metric__model__condition, plus the baselines"""
    q = p[p.res_kb == res]
    rows = []
    for pat, g in q.groupby("patient"):
        r = {"patient": pat, "fold": int(g.fold.iloc[0]), "res_kb": res, "n_specific": int(g.n_specific.iloc[0])}
        for _, x in g.iterrows():
            for m in METRICS:
                r[f"{m}__{x.model}__{x.condition}"] = x[m]
        x0 = g.iloc[0]
        for k, c in BASES.items():
            r[f"pearson__{k}"] = x0[c]
            r[f"pearson_specific__{k}"] = x0[c + "_specific"]
        rows.append(r)
    return pd.DataFrame(rows)


def pooled_tables(w, rng, nb):
    pooled = []
    for res, d in w.items():
        for c in d.columns:
            if "__" not in c:
                continue
            pooled.append(dict(res_kb=res, column=c, **dict(zip(("median", "lo", "hi"), boot(d[c], rng, nb))),
                               n=int(d[c].notna().sum())))
    return pd.DataFrame(pooled)


def paired(w, rng, nb):
    out = []

    def add(res, name, a, b, metric):
        d = w[res][a] - w[res][b]
        med, lo, hi = boot(d, rng, nb)
        out.append(dict(res_kb=res, comparison=name, metric=metric, median_diff=med, lo=lo, hi=hi,
                        n_pos=int((d > 0).sum()), n=int(d.notna().sum())))
    for res in w:
        for metric in ("pearson", "pearson_specific"):
            for ref in ("ref_other_patients", "ref_lopo_all", "ref_hspc4"):
                for m in MODELS:
                    for cond in ("marks", "none"):
                        col = f"{metric}__{m}__{cond}"
                        if col in w[res]:
                            add(res, f"{m}|{cond} minus {ref}", col, f"{metric}__{ref}", metric)
            for m in ("aml_refin", "aml_honest", "base_honest", "base_refin@aml", "base_refin@c12",
                      "aml_refin@hspc"):
                add(res, f"{m}: marks minus none", f"{metric}__{m}__marks", f"{metric}__{m}__none", metric)
            add(res, "aml_refin|marks minus base_refin@aml|marks (effect of AML training)",
                f"{metric}__aml_refin__marks", f"{metric}__base_refin@aml__marks", metric)
            add(res, "aml_honest|marks minus base_honest|marks (effect of AML training)",
                f"{metric}__aml_honest__marks", f"{metric}__base_honest__marks", metric)
        for m in ("scc", "insulation_corr", "loop_enrich_corr", "compartment_corr"):
            for mod in ("aml_refin", "aml_honest", "base_honest"):
                add(res, f"{mod}: marks minus none", f"{m}__{mod}__marks", f"{m}__{mod}__none", m)
        for mod in ("aml_refin", "aml_honest", "base_refin@aml"):
            for cond in ("marks", "none"):
                for m in ("resid_vs_aml_train_mean", "resid_vs_aml_train_mean_specific"):
                    pass
        # best of all models per patient against the other-patient reference
        for metric in ("pearson", "pearson_specific"):
            cols = [f"{metric}__{m}__{c}" for m in MODELS for c in ("marks", "none") if f"{metric}__{m}__{c}" in w[res]]
            best = w[res][cols].max(axis=1)
            d = best - w[res][f"{metric}__ref_other_patients"]
            med, lo, hi = boot(d, rng, nb)
            out.append(dict(res_kb=res, comparison="best model per patient minus ref_other_patients",
                            metric=metric, median_diff=med, lo=lo, hi=hi, n_pos=int((d > 0).sum()), n=len(d)))
    return pd.DataFrame(out)


def loop_stripe(ww, rng, nb):
    """Per patient and kind: median(model - ref), median(measured - ref), fraction above ref,
    corr(model - ref, measured - ref) over the feature windows; pooled by donor bootstrap."""
    rows = []
    for (pat, kind), g in ww.groupby(["patient", "kind"]):
        ref = g.aml_reference.values
        mr_all = g.measured.values - ref
        r = {"patient": pat.replace("xu_", ""), "kind": kind, "n_windows": len(g),
             "measured_minus_ref": float(np.nanmedian(mr_all))}
        for c in FEAT:
            dm_all = g[c].values - ref
            ok = ~(np.isnan(dm_all) | np.isnan(mr_all))
            dm, mr = dm_all[ok], mr_all[ok]
            r[f"{c}__minus_ref"] = float(np.median(dm)) if ok.any() else np.nan
            r[f"{c}__frac_above"] = float((dm > 0).mean()) if ok.any() else np.nan
            r[f"{c}__corr"] = float(np.corrcoef(dm, mr)[0, 1]) if ok.sum() > 2 and np.std(dm) > 0 else np.nan
        rows.append(r)
    per = pd.DataFrame(rows)
    pooled = []
    for kind, g in per.groupby("kind"):
        for c in g.columns:
            if c in ("patient", "kind", "n_windows"):
                continue
            pooled.append(dict(kind=kind, column=c, **dict(zip(("median", "lo", "hi"), boot(g[c], rng, nb))), n=len(g)))
    return per, pd.DataFrame(pooled)


def subtype(w, table, split, rng):
    grp = {str(r.patient): r.consensus_group for r in table.itertuples()}
    folds = split["folds"]
    of = {p: f for f, ps in folds.items() for p in ps}
    rows = []
    for res, d in w.items():
        for r in d.itertuples():
            pat = r.patient
            g = grp.get(pat)
            n_same = sum(1 for p2, f2 in of.items() if f2 != str(r.fold) and (grp.get(p2) == g))
            rows.append(dict(res_kb=res, patient=pat, fold=r.fold, group=g, n_same_group_in_training=n_same,
                             refin_minus_ref=r.pearson__aml_refin__marks - r.pearson__ref_other_patients,
                             refin_minus_ref_specific=r.pearson_specific__aml_refin__marks - r.pearson_specific__ref_other_patients,
                             marks_minus_none_specific=r.pearson_specific__aml_refin__marks - r.pearson_specific__aml_refin__none,
                             ref_pearson=r.pearson__ref_other_patients))
    per = pd.DataFrame(rows)
    gm = per.groupby(["res_kb", "group"]).agg(n=("patient", "size"), patients=("patient", lambda s: ",".join(s)),
                                              refin_minus_ref=("refin_minus_ref", "median"),
                                              refin_minus_ref_specific=("refin_minus_ref_specific", "median"),
                                              marks_minus_none_specific=("marks_minus_none_specific", "median"),
                                              ref_pearson=("ref_pearson", "median"),
                                              mean_n_same_group_in_training=("n_same_group_in_training", "mean")).reset_index()
    return per, gm


def old_vs_new(w_new, p_old, new_ids):
    """Old (contaminated) fold 0 against the new folds: the four old test patients are
    027, 270 (new fold 10) and 773, 1916 (new fold 12)."""
    po = p_old[p_old.group == "patient"].copy()
    po["patient"] = po.cell.str.replace("amlxu", "", regex=False)
    rows = []
    for res in (25, 10):
        q = po[po.res_kb == res]
        for pat in sorted(q.patient.unique()):
            g = q[q.patient == pat]
            n = w_new[res].set_index("patient").loc[pat]
            for m in ("aml_refin", "aml_honest", "base_honest", "base_refin@aml"):
                for c in ("marks", "none"):
                    for met in ("pearson", "pearson_specific"):
                        x = g[(g.model == m) & (g.condition == c)]
                        if len(x):
                            rows.append(dict(res_kb=res, patient=pat, model=m, condition=c, metric=met,
                                             old=float(x[met].iloc[0]), new=float(n[f"{met}__{m}__{c}"])))
            for met, oc, k in (("pearson", "base_aml_lopo_all", "ref_lopo_all"),
                               ("pearson", "base_aml_train_mean", "ref_other_patients"),
                               ("pearson_specific", "base_aml_lopo_all_specific", "ref_lopo_all"),
                               ("pearson_specific", "base_aml_train_mean_specific", "ref_other_patients")):
                rows.append(dict(res_kb=res, patient=pat, model=k, condition="-", metric=met,
                                 old=float(g[oc].iloc[0]), new=float(n[f"{met}__{k}"])))
    return pd.DataFrame(rows)


def figure(w, paired_df, ls_per, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c_m, c_n, c_r = "#0072B2", "#E69F00", "#555555"
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    pats = sorted(w[25].patient, key=lambda x: int(x))
    for a, (res, metric, title) in zip(ax.ravel()[:3], [(25, "pearson", "Pearson, 25 kb"), (10, "pearson", "Pearson, 10 kb"),
                                                         (10, "pearson_specific", "Patient-specific windows, 10 kb")]):
        d = w[res].set_index("patient").loc[pats]
        x = np.arange(len(pats))
        a.axhline(0, color=c_r, lw=1)
        for col, lab, cc, dx in [("aml_refin__marks", "aml_refin, marks", c_m, -0.15), ("aml_refin__none", "aml_refin, no marks", c_n, 0.0),
                                  ("base_refin@aml__marks", "base_refin@aml, marks (no AML training)", "#009E73", 0.15)]:
            if col.startswith("base_honest") and metric == "pearson_specific":
                pass
            y = d[f"{metric}__{col}"] - d[f"{metric}__ref_other_patients"]
            a.scatter(x + dx, y, s=22, color=cc, label=lab)
        a.set_xticks(x)
        a.set_xticklabels(pats, rotation=45)
        a.set_ylabel("model minus mean of other patients")
        a.set_title(title)
    ax[0, 0].legend(fontsize=8, frameon=False)
    a = ax[1, 1]
    s = ls_per[ls_per.kind == "loop"]
    a.axhline(0, color=c_r, lw=1)
    x = np.arange(len(s))
    a.scatter(x - 0.15, s["measured_minus_ref"], color=c_r, s=22, label="measured")
    a.scatter(x + 0.0, s["aml_refin|marks__minus_ref"], color=c_m, s=22, label="aml_refin, marks")
    a.scatter(x + 0.15, s["aml_refin|none__minus_ref"], color=c_n, s=22, label="aml_refin, no marks")
    a.set_xticks(x)
    a.set_xticklabels(s.patient, rotation=45)
    a.set_ylabel("loop enrichment minus reference (log2)")
    a.set_title("Xu AML loops, 10 kb")
    a.legend(fontsize=8, frameon=False)
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--out", default=str(REPO / "results/aml_train_v2"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    p, ww = load_new(a.folds)
    w = {res: wide(p, res) for res in (25, 10)}
    pd.concat(w.values()).to_csv(out / "per_patient.csv", index=False)
    pooled_tables(w, rng, a.boot).to_csv(out / "pooled_medians.csv", index=False)
    pr = paired(w, rng, a.boot)
    pr.to_csv(out / "paired_claims.csv", index=False)
    per, pool = loop_stripe(ww, rng, a.boot)
    per.to_csv(out / "loop_stripe_per_patient.csv", index=False)
    pool.to_csv(out / "loop_stripe_pooled.csv", index=False)
    table = pd.read_csv(REPO / "results/aml_subtype/patient_subtype_table.csv", dtype={"patient": str})
    split = json.load(open(REPO / "results/aml_train/split_v2.json"))
    sp, sg = subtype(w, table, split, rng)
    sp.to_csv(out / "subtype_per_patient.csv", index=False)
    sg.to_csv(out / "subtype_groups.csv", index=False)
    p_old = pd.read_csv(REPO / "results/aml_train/eval_f0_patients.csv")
    ov = old_vs_new(w, p_old, None)
    ov.to_csv(out / "old_vs_new_same_patients.csv", index=False)
    # old fold 0 pooled over its four patients, all models incl. aml_refres
    po = p_old[(p_old.group == "patient") & p_old.model.isin(["aml_refres", "aml_refin", "aml_meth_refin"])]
    po.groupby(["res_kb", "model", "condition"])[["pearson", "scc", "pearson_specific", "base_aml_train_mean",
                                                   "base_aml_lopo_all", "pearson_specific"]].median().reset_index() \
        .to_csv(out / "old_fold0_refres_meth.csv", index=False)
    figure(w, pr, per[per.kind == "loop"].reset_index(drop=True) if False else per, out / "aml_folds_v2.png")
    print("written", out)


if __name__ == "__main__":
    main()
