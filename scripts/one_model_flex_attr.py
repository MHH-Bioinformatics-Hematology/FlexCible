"""Release candidate all12one_fullrd against all12one: flexible input, Shapley values, attribution, group importance.

Reads (read only) the results of scripts/flexible_inputs.py, scripts/attribute.py and scripts/scale_importance.py for
the plain one model (results/flexible_inputs, results/attribution, results/scale_importance_all12one) and for the
release candidate (results/<analysis>/all12one_fullrd[/nometh]) and writes results/one_model/FLEX_ATTR.md and the
csv files next to it (flex_attr_*.csv). Registry section 96.

    python scripts/one_model_flex_attr.py [--boot 200]

Definitions
  value of a mark set  median over the 413 Akita test regions of the per-region Pearson r (summary.csv)
  curve                median over the 20 random draws of the set median; fraction of gain = (v(n) - v(none)) / (v(all13) - v(none))
  Shapley values       KernelSHAP fit as in the manuscript figure script fig_shapley.py (Shapley kernel weights, efficiency
                       constraint) on all evaluated mark sets of one cell, bin size and consensus setting; the 13 marks are
                       the players, Meth (release candidate) is constant context in the "meth" variant
  Meth marginal        for the same mark set, v(set + Meth) - v(set) from the two release sweeps (paired over all sets)
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
OUT = R / "one_model"
FI = R / "flexible_inputs"
VARIANTS = {"all12one": FI, "fullrd_meth": FI / "all12one_fullrd", "fullrd_nometh": FI / "all12one_fullrd/nometh"}
CELLS = ["k562", "hmec", "huvec", "gm12878", "imr90"]
HELD = ("k562", "hmec", "huvec")


def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                     else ("" if isinstance(v, (float, np.floating)) else str(v)) for v in r) + " |")
    return "\n".join(out)


def load(name):
    return pd.read_csv(VARIANTS[name] / "summary.csv")


def per_region(name):
    d = VARIANTS[name]
    return pd.concat([pd.read_csv(d / "per_region.csv"), pd.read_csv(d / "per_region_random.csv.gz")])


# ----------------------------------------------------------------------------------------------- headline tables
def v(s, cell, res, wr, kind=None, name=None):
    d = s[(s.cell == cell) & (s.res == res) & (s.with_reference == wr)]
    if kind:
        d = d[d.set_kind == kind]
    if name:
        d = d[d.set_name == name]
    return d


def curve_value(s, cell, res, wr, n):
    """median over the draws of the set median r at n marks (n = 0 none, 13 all13; single marks stand in for n = 1 at 5, 2 kb)"""
    if n == 0:
        return float(v(s, cell, res, wr, "none").median_r.iloc[0])
    if n == 13:
        return float(v(s, cell, res, wr, "all13").median_r.iloc[0])
    d = v(s, cell, res, wr)
    d = d[((d.set_kind == "random") & (d.n_marks == n)) | ((d.set_kind == "single") & (n == 1))]
    return float(d.median_r.median()) if len(d) else np.nan


def headline(res):
    rows = []
    for name in VARIANTS:
        s = load(name)
        for cell in CELLS:
            for wr in (1, 0):
                if v(s, cell, res, wr).empty:
                    continue
                none, full = curve_value(s, cell, res, wr, 0), curve_value(s, cell, res, wr, 13)
                seven = float(v(s, cell, res, wr, "seven").median_r.iloc[0])
                cons = float(v(s, cell, res, wr).consensus_median_r.iloc[0])
                row = {"model": name, "cell": cell, "consensus": "with" if wr else "without", "consensus_alone": cons,
                       "none": none, "one": curve_value(s, cell, res, wr, 1), "three": curve_value(s, cell, res, wr, 3),
                       "four": curve_value(s, cell, res, wr, 4), "six": curve_value(s, cell, res, wr, 6),
                       "seven": seven, "all13": full, "gap_all13_minus_seven": full - seven,
                       "ctcf_h3k27ac": float(v(s, cell, res, wr, "ctcf_h3k27ac").median_r.iloc[0]),
                       "ctcf_rad21": (float(v(s, cell, res, wr, "ctcf_rad21").median_r.iloc[0])
                                      if not v(s, cell, res, wr, "ctcf_rad21").empty else np.nan)}
                for n, k in ((3, "three"), (4, "four"), (6, "six")):
                    row[f"frac_gain_{k}"] = (row[k] - none) / (full - none) if abs(full - none) > 1e-9 else np.nan
                row["gain_none_to_all13"] = full - none
                row["none_minus_consensus"] = none - cons
                rows.append(row)
    return pd.DataFrame(rows)


def draw_stats(res):
    """spread over the 20 draws at 1, 3, 4, 6 marks: median, q25, q75, min, max, per model, cell and setting"""
    rows = []
    for name in VARIANTS:
        s = load(name)
        for cell in CELLS:
            for wr in (1, 0):
                for n in (1, 3, 4, 6, 12):
                    d = v(s, cell, res, wr, "random")
                    d = d[d.n_marks == n].median_r
                    if len(d):
                        rows.append({"model": name, "cell": cell, "consensus": "with" if wr else "without", "n_marks": n,
                                     "n_draws": len(d), "median": d.median(), "q25": d.quantile(.25), "q75": d.quantile(.75),
                                     "min": d.min(), "max": d.max(), "mean": d.mean()})
    return pd.DataFrame(rows)


def paired_regions(res):
    """fullrd_nometh and fullrd_meth against all12one on identical sets: paired per-region differences"""
    base = per_region("all12one")
    rows = []
    for name in ("fullrd_nometh", "fullrd_meth"):
        new = per_region(name)
        for cell in CELLS:
            for wr in (1, 0):
                for setname in ("all13", "none", "seven", "ctcf_h3k27ac", "single_CTCF", "minus_CTCF"):
                    a = base[(base.cell == cell) & (base.res == res) & (base.with_reference == wr) & (base.set_name == setname)]
                    b = new[(new.cell == cell) & (new.res == res) & (new.with_reference == wr) & (new.set_name == setname)]
                    m = a.merge(b, on="region", suffixes=("_old", "_new"))
                    if len(m) < 20:
                        continue
                    d = m.r_new - m.r_old
                    try:
                        p = wilcoxon(d).pvalue
                    except ValueError:
                        p = np.nan
                    rows.append({"variant": name, "cell": cell, "consensus": "with" if wr else "without", "set": setname,
                                 "n_regions": len(m), "median_old": m.r_old.median(), "median_new": m.r_new.median(),
                                 "mean_diff": d.mean(), "median_diff": d.median(), "q25_diff": d.quantile(.25),
                                 "q75_diff": d.quantile(.75), "frac_new_higher": float((d > 0).mean()), "wilcoxon_p": p})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------- Shapley
def kernel_shap(members, values, marks):
    M = len(marks)
    idx = {m: k for k, m in enumerate(marks)}
    full = frozenset(marks)
    agg = pd.Series(list(values), index=pd.Index(members)).groupby(level=0).mean()
    members, values = list(agg.index), agg.values
    v0, vf = agg[frozenset()], agg[full]
    rows, y, w = [], [], []
    for s, val in zip(members, values):
        n = len(s)
        if n == 0 or n == M:
            continue
        x = np.zeros(M)
        x[[idx[m] for m in s]] = 1
        rows.append(x)
        y.append(val - v0)
        w.append((M - 1) / (comb(M, n) * n * (M - n)))
    X, y, w = np.array(rows), np.array(y), np.array(w)
    Xr = X[:, :-1] - X[:, -1:]
    yr = y - X[:, -1] * (vf - v0)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xr * sw[:, None], yr * sw, rcond=None)
    return pd.Series(np.append(beta, (vf - v0) - beta.sum()), index=marks)


def shap_for(name, s, pr, cell, res, wr, boot, rng):
    d = s[(s.cell == cell) & (s.res == res) & (s.with_reference == wr)]
    if d.empty:
        return None
    present = d[d.set_kind == "all13"].marks_present.iloc[0].split("+")
    members = [frozenset(x.split("+")) if isinstance(x, str) and x else frozenset() for x in d.marks_present]
    phi = kernel_shap(members, d.median_r.values, present)
    out = {"phi": phi}
    if boot:
        P = pr[(pr.cell == cell) & (pr.res == res) & (pr.with_reference == wr)].pivot_table(
            index="region", columns="set_name", values="r")[d.set_name.values].values
        bs = []
        for _ in range(boot):
            take = rng.integers(0, P.shape[0], P.shape[0])
            bs.append(kernel_shap(members, np.nanmedian(P[take], axis=0), present))
        out["boot"] = pd.DataFrame(bs)
    return out


def shapley(boot):
    rng = np.random.default_rng(0)
    rows = []
    for name in VARIANTS:
        s, pr = load(name), per_region(name)
        for res in (10, 25):
            for cell in CELLS:
                for wr in (1, 0):
                    f = shap_for(name, s, pr, cell, res, wr, boot, rng)
                    if f is None:
                        continue
                    for mk, val in f["phi"].items():
                        row = {"model": name, "res": res, "cell": cell, "consensus": "with" if wr else "without",
                               "mark": mk, "phi": val}
                        if boot:
                            row["ci_lo"], row["ci_hi"] = f["boot"][mk].quantile(.025), f["boot"][mk].quantile(.975)
                        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- Meth marginal
def meth_marginal():
    """paired v(set + Meth) - v(set) over all sets, from the two release sweeps"""
    a, b = load("fullrd_nometh"), load("fullrd_meth")
    k = ["cell", "res", "set_name", "with_reference"]
    m = a.merge(b, on=k, suffixes=("_no", "_me"))
    m["d"] = m.median_r_me - m.median_r_no
    m["consensus"] = np.where(m.with_reference == 1, "with", "without")
    rows = []
    for (cell, res, cons), g in m.groupby(["cell", "res", "consensus"]):
        rows.append({"cell": cell, "res": res, "consensus": cons, "n_sets": len(g), "mean": g.d.mean(), "median": g.d.median(),
                     "q25": g.d.quantile(.25), "q75": g.d.quantile(.75), "min": g.d.min(), "max": g.d.max(),
                     "frac_sets_higher": float((g.d > 0).mean()),
                     "none_plus_meth_minus_none": float(g[g.set_name == "none"].d.iloc[0]) if (g.set_name == "none").any() else np.nan,
                     "all13_plus_meth_minus_all13": float(g[g.set_name == "all13"].d.iloc[0]) if (g.set_name == "all13").any() else np.nan})
    return pd.DataFrame(rows), m


# ------------------------------------------------------------------------------------------------ attribution
def attribution():
    out = {}
    base = {"all12one": R / "attribution", "fullrd_meth": R / "attribution/all12one_fullrd",
            "fullrd_nometh": R / "attribution/all12one_fullrd/nometh"}
    rows = []
    for name, d in base.items():
        f = d / "ig_global.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f)
        sh = [c for c in g.columns if c.startswith("share:")]
        for setting in ("cons", "nocons"):
            x = g[g.setting == setting].set_index("track")
            tot = x["mean_share"]
            for rank, (tr, val) in enumerate(tot.sort_values(ascending=False).items(), 1):
                rows.append({"model": name, "setting": setting, "rank": rank, "track": tr, "mean_share": val,
                             "sd_share": x.loc[tr, "sd_share"],
                             "median_share": float(np.median(x.loc[tr, sh].astype(float))),
                             "min_share": float(x.loc[tr, sh].astype(float).min()),
                             "max_share": float(x.loc[tr, sh].astype(float).max()),
                             "mean_abs_ig": x.loc[tr, "mean_abs_ig"]})
    return pd.DataFrame(rows)


def completeness():
    rows = []
    base = {"all12one": R / "attribution", "fullrd_meth": R / "attribution/all12one_fullrd",
            "fullrd_nometh": R / "attribution/all12one_fullrd/nometh"}
    for name, d in base.items():
        for sub, steps in (("", 50), ("s300", 300)):
            f = d / sub / "ig_global_completeness.csv" if sub else d / "ig_global_completeness.csv"
            if not f.exists() and name == "all12one" and sub:
                f = R / "attribution_s300/ig_global_completeness.csv"
            if f.exists():
                c = pd.read_csv(f)
                for setting, g in c.groupby("setting"):
                    rows.append({"model": name, "steps": steps, "target": "global", "setting": setting, "n": len(g),
                                 "median_rel_err": g.rel_err.median(), "max_rel_err": g.rel_err.max(),
                                 "frac_within_5pct": float((g.rel_err < .05).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- group importance
def group_importance():
    srcs = {"all12multires (S9, model A)": R / "scale_importance",
            "all12one": R / "scale_importance_all12one",
            "all12one_fullrd": R / "scale_importance_all12one_fullrd"}
    rows, cells = [], []
    for name, d in srcs.items():
        if (d / "importance_drop.csv").exists():
            x = pd.read_csv(d / "importance_drop.csv")
            x.insert(0, "model", name)
            rows.append(x)
        if (d / "importance_drop_cells.csv").exists():
            x = pd.read_csv(d / "importance_drop_cells.csv")
            x.insert(0, "model", name)
            cells.append(x)
    return (pd.concat(rows) if rows else pd.DataFrame()), (pd.concat(cells) if cells else pd.DataFrame())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ["# Release candidate all12one_fullrd against all12one: flexible input, Shapley values, attribution, group importance\n",
             "Generated by `scripts/one_model_flex_attr.py` (registry section 96). Models: `all12one` (13 marks, section 87, 90), "
             "`fullrd_meth` = all12one_fullrd with the cell's methylation track given in every set (14 tracks), `fullrd_nometh` = "
             "all12one_fullrd with Meth absent (the 13-mark design of all12one; same sets, same draws, same regions). Metric: "
             "median over the 413 Akita test regions of the per-region Pearson r of log O/E.\n"]
    for res in (10, 25, 5, 2):
        h = headline(res)
        if h.empty:
            continue
        h.round(4).to_csv(OUT / f"flex_attr_headline_{res}kb.csv", index=False)
        parts.append(f"## Headline sets at {res} kb\n")
        for cons in ("with", "without"):
            x = h[h.consensus == cons]
            cols = ["model", "cell", "consensus_alone", "none", "one", "three", "four", "six", "seven", "all13",
                    "gap_all13_minus_seven", "ctcf_h3k27ac"]
            parts.append(f"**{res} kb, {cons} consensus** (none = no mark; one, three, four, six = median over the draws; "
                         "gap = all13 minus seven-mark set)\n")
            parts.append(md(x[cols]) + "\n")
            cols = ["model", "cell", "gain_none_to_all13", "none_minus_consensus", "frac_gain_three", "frac_gain_four",
                    "frac_gain_six"]
            parts.append(f"Fraction of the gain from no mark to all 13 reached with 3, 4, 6 marks ({res} kb, {cons} consensus)\n")
            parts.append(md(x[cols]) + "\n")
        ds = draw_stats(res)
        ds.round(4).to_csv(OUT / f"flex_attr_draws_{res}kb.csv", index=False)
        parts.append(f"Spread over the 20 random draws at {res} kb (median r per draw; median, quartiles, range, mean)\n")
        parts.append(md(ds[ds.n_marks.isin([1, 4])]) + "\n")
        if res in (10, 25):
            pr = paired_regions(res)
            pr.round(4).to_csv(OUT / f"flex_attr_paired_regions_{res}kb.csv", index=False)
            parts.append(f"Paired per-region differences, release candidate minus all12one, same set and region ({res} kb)\n")
            parts.append(md(pr, "{:.4f}") + "\n")
    sh = shapley(a.boot)
    sh.round(5).to_csv(OUT / "flex_attr_shapley.csv", index=False)
    parts.append("## Shapley values (KernelSHAP on the sweep; r units)\n")
    for res in (10, 25):
        for cons in ("with", "without"):
            x = sh[(sh.res == res) & (sh.consensus == cons)]
            if x.empty:
                continue
            t = x.pivot_table(index="mark", columns=["model", "cell"], values="phi")
            t.columns = [f"{m}:{c}" for m, c in t.columns]
            t = t.assign(**{"mean_held_all12one": x[(x.model == "all12one") & x.cell.isin(HELD)].groupby("mark").phi.mean(),
                            "mean_held_nometh": x[(x.model == "fullrd_nometh") & x.cell.isin(HELD)].groupby("mark").phi.mean(),
                            "mean_held_meth": x[(x.model == "fullrd_meth") & x.cell.isin(HELD)].groupby("mark").phi.mean()})
            t = t.sort_values("mean_held_all12one", ascending=False).reset_index()
            parts.append(f"**{res} kb, {cons} consensus** (per model and cell; last three columns: mean over the held-out cells with the mark)\n")
            parts.append(md(t, "{:.4f}") + "\n")
        # rank agreement
        for cons in ("with", "without"):
            x = sh[(sh.res == res) & (sh.consensus == cons)]
            rs = []
            for cell in CELLS:
                p = x[(x.cell == cell)].pivot(index="mark", columns="model", values="phi").dropna()
                if {"all12one", "fullrd_nometh"} <= set(p.columns) and len(p) > 3:
                    rs.append(f"{cell} {spearmanr(p.all12one, p.fullrd_nometh)[0]:.3f}")
            parts.append(f"Spearman correlation of the 13 Shapley values, all12one against fullrd_nometh ({res} kb, {cons} consensus): "
                         + ", ".join(rs) + "\n")
    mm, _ = meth_marginal()
    if not mm.empty:
        mm.round(4).to_csv(OUT / "flex_attr_meth_marginal.csv", index=False)
        parts.append("## Marginal value of the methylation track (release candidate; paired over all mark sets)\n")
        parts.append(md(mm, "{:.4f}") + "\n")
    at = attribution()
    if not at.empty:
        at.round(5).to_csv(OUT / "flex_attr_attribution_global.csv", index=False)
        parts.append("## Integrated gradients, global target (40 random K562 test regions, seed 0, 50 steps): top tracks by share of the absolute attribution\n")
        for setting in ("cons", "nocons"):
            x = at[(at.setting == setting) & (at["rank"] <= 8)]
            t = x.assign(v=lambda d: d.apply(lambda r: f"{r.track} {r.mean_share:.3f}", axis=1)).pivot(
                index="rank", columns="model", values="v").reset_index()
            parts.append(f"**{'with' if setting == 'cons' else 'without'} consensus**\n")
            parts.append(md(t) + "\n")
        parts.append("Full table (mean, sd, median, min, max over the regions) in `flex_attr_attribution_global.csv`.\n")
    cp = completeness()
    if not cp.empty:
        cp.round(4).to_csv(OUT / "flex_attr_attribution_completeness.csv", index=False)
        parts.append("Completeness of the global target\n")
        parts.append(md(cp, "{:.4f}") + "\n")
    gi, gc = group_importance()
    if not gi.empty:
        gi.to_csv(OUT / "flex_attr_group_importance.csv", index=False)
        parts.append("## Group importance (loss of the median per-window Pearson r when a group is set absent; test windows)\n")
        for kb in (2, 5, 10, 25):
            x = gi[(gi.res_kb == kb) & (gi.dropped != "all inputs")]
            if x.empty:
                continue
            t = x.pivot(index="dropped", columns="model", values="loss_vs_all_inputs")
            sc = gi[(gi.res_kb == kb) & (gi.dropped == "all inputs")].set_index("model").median_over_cells
            t.loc["(score, all inputs)"] = sc
            t = t.reset_index()
            parts.append(f"**{kb} kb**\n")
            parts.append(md(t, "{:.4f}") + "\n")
    (OUT / "FLEX_ATTR_tables.md").write_text("\n".join(parts) + "\n")
    print("wrote", OUT / "FLEX_ATTR_tables.md")


if __name__ == "__main__":
    main()
