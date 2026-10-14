"""Tables, paired contrasts, endpoint verdict and figure of the mouse transfer (registry section 75).

Called by scripts/mouse_transfer.py summary | figure. Reads $HM/validation_mouse/scores/*.csv.gz and
ceiling_*.csv; writes results/mouse_transfer/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import mouse_transfer as mt

OUT = mt.RESULTS
VM = mt.VM
CELLS = mt.POOL
LAB = {"mesc": "mESC (E14)", "ch12": "CH12.LX (B lymphoma)", "heart": "heart (adult)"}
NBOOT = 2000
STRUCT = ["pearson", "scc", "insulation_corr", "boundary_f1", "loop_enrich_corr", "loop_apa_ratio",
          "compartment_corr", "eigenvector_abs_corr", "stripe_corr"]


def load_scores(sets=("auto",)):
    fr = []
    for f in sorted((VM / "scores").glob("*_*kb.csv.gz")):
        if "limited" in f.name:
            continue
        fr.append(pd.read_csv(f))
    D = pd.concat(fr, ignore_index=True)
    return D[D.set.isin(sets)]


def window_vec(D, cell, res, method, metric="pearson"):
    s = D[(D.cell == cell) & (D.res == res) & (D.method == method)]
    return s.set_index("key")[metric]


def boot_median_diff(x, y, rng, n=NBOOT):
    idx = rng.integers(0, len(x), (n, len(x)))
    return np.median(x[idx], axis=1) - np.median(y[idx], axis=1)


def paired(D, cell, res, m1, m2, metric="pearson", seed=0, fn=np.median):
    a, b = window_vec(D, cell, res, m1, metric), window_vec(D, cell, res, m2, metric)
    j = a.index.intersection(b.index)
    x, y = a.loc[j].values, b.loc[j].values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 30:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (NBOOT, len(x)))
    d = fn(x[idx], axis=1) - fn(y[idx], axis=1)
    return {"cell": cell, "res": res, "m1": m1, "m2": m2, "metric": metric, "n": len(x),
            "stat_m1": float(fn(x)), "stat_m2": float(fn(y)), "diff": float(fn(x) - fn(y)),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "frac_windows_m1_above": float((x > y).mean())}


def ceilings():
    rows = []
    for c in CELLS:
        for res in (10000, 25000):
            f = VM / "scores" / f"ceiling_{c}_{mt.kb(res)}.csv"
            if f.exists():
                d = pd.read_csv(f)
                rows.append({"cell": c, "res": res, "n_windows": len(d), "half_depth_median": d.r_half.median(),
                             "full_depth_median": d.r_full.median()})
    return pd.DataFrame(rows)


def stats_table():
    rows = []
    for c in CELLS + ["mesc_rep"]:
        for res in (10000, 25000):
            f = VM / "stats" / f"{c}_{mt.kb(res)}.json"
            if f.exists():
                s = json.loads(f.read_text())
                rows.append({"sample": c, "res": res, "experiment": s.get("experiment"), "assay": s.get("assay"),
                             "contacts_M": round(s.get("contacts", np.nan) / 1e6, 1) if "contacts" in s else np.nan,
                             "cis_contacts_M": round(s.get("cis_contacts", np.nan) / 1e6, 1) if "cis_contacts" in s else np.nan,
                             "ice_masked_frac": round(s.get("ice_masked_frac", np.nan), 4)})
    return pd.DataFrame(rows)


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                     else ("" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def run(stage):
    OUT.mkdir(parents=True, exist_ok=True)
    if stage == "summary":
        summary()
    else:
        figure()


# ------------------------------------------------------------------ summary
def summary():
    D = load_scores(("auto",))
    X = load_scores(("x",))
    # medians per cell, resolution, method
    g = D.groupby(["cell", "res", "method"])
    S = g[mt.METRICS].median().add_suffix("_median").join(g[["boundary_f1"]].mean().add_suffix("_mean")) \
        .join(g.size().rename("n_windows")).reset_index()
    S.to_csv(OUT / "summary_autosomes.csv", index=False, float_format="%.4f")
    if len(X):
        gx = X.groupby(["cell", "res", "method"])
        SX = gx[mt.METRICS].median().add_suffix("_median").join(gx.size().rename("n_windows")).reset_index()
        SX.to_csv(OUT / "summary_chrX.csv", index=False, float_format="%.4f")
    C = ceilings()
    C.to_csv(OUT / "ceilings.csv", index=False, float_format="%.4f")
    stats_table().to_csv(OUT / "samples.csv", index=False)
    # paired contrasts of the Pearson
    pairs = []
    for res in (10000, 25000):
        for c in CELLS:
            def add(m1, m2, metric="pearson"):
                r = paired(D, c, res, m1, m2, metric)
                if r:
                    pairs.append(r)
            for a, b in (("honest:all7", "distance_decay"), ("honest:all7", "honest_shuf:all7"),
                         ("honest:all7", "honest_xm:all7"), ("honest:all7", "untrained:all7"),
                         ("honest:all7", "honest:none"), ("honest:all7", "honest:ctcf"),
                         ("honest:all7", "honest:ctcf_k27ac"), ("honest:all7", "honest_b4:all7"),
                         ("honest:all7", "consensus"), ("honest:all7", "distance_oracle"),
                         ("refin:all7", "consensus"), ("refres:all7", "consensus"), ("diff:all7", "consensus"),
                         ("refin:all7", "honest:all7"), ("refin:all7", "refin_shuf:all7"),
                         ("refin:all7", "refin_xm:all7"), ("refin:all7", "refin:none"),
                         ("honest:avail", "honest:all7"), ("refin:avail", "refin:all7"),
                         ("honest:all7", "akita:Hsieh_mESC_uC"), ("honest:all7", "akita:Bonev_ES"),
                         ("refin:all7", "akita:Bonev_ES"), ("honest:all7", "akita:Bonev_CN"),
                         ("honest:all7", "akita:Bonev_NPC"), ("akita:Bonev_ES", "consensus"),
                         ("akita:Hsieh_mESC_uC", "honest_shuf:all7"),
                         ("honest:all7", "replicate_experiment"), ("consensus", "replicate_experiment")):
                add(a, b)
    P = pd.DataFrame(pairs)
    P.to_csv(OUT / "paired_contrasts.csv", index=False, float_format="%.4f")

    # structure classes: honest vs shuffled marks vs none vs consensus, 10 kb
    rows = []
    for c in CELLS:
        for metric in STRUCT:
            fn = np.mean if metric == "boundary_f1" else np.median
            for m in ("honest:all7", "honest_shuf:all7", "honest:none", "honest_xm:all7", "refin:all7",
                      "consensus", "distance_oracle", "akita:Bonev_ES"):
                v = window_vec(D, c, 10000, m, metric).dropna()
                if len(v):
                    rows.append({"cell": c, "metric": metric, "method": m, "stat": float(fn(v.values)),
                                 "n": len(v)})
    SC = pd.DataFrame(rows)
    SC.to_csv(OUT / "structure_classes.csv", index=False, float_format="%.4f")
    cls = []
    for c in CELLS:
        for metric in STRUCT:
            fn = np.mean if metric == "boundary_f1" else np.median
            for ref in ("honest_shuf:all7", "honest:none", "honest_xm:all7"):
                r = paired(D, c, 10000, "honest:all7", ref, metric, fn=fn)
                if r:
                    r["ref"] = ref
                    cls.append(r)
    CL = pd.DataFrame(cls)
    CL.to_csv(OUT / "structure_class_contrasts.csv", index=False, float_format="%.4f")

    # per chromosome (honest 10 kb)
    h = D[(D.res == 10000) & (D.method == "honest:all7")]
    h.groupby(["cell", "chrom"]).pearson.agg(["median", "size"]).reset_index().to_csv(
        OUT / "per_chromosome_honest_10kb.csv", index=False, float_format="%.4f")

    # primary endpoint
    verdict = {"definition": "honest:all7 10 kb autosome windows above distance_decay and above honest_shuf:all7 "
                             "in every sample (difference of medians, 95% paired bootstrap interval excludes 0)",
               "samples": {}}
    ok_all = True
    for c in CELLS:
        e = {}
        for ref in ("distance_decay", "honest_shuf:all7"):
            r = paired(D, c, 10000, "honest:all7", ref)
            e[ref] = r
            if not (r and r["lo"] > 0):
                ok_all = False
        cc = C[(C.cell == c) & (C.res == 10000)]
        med = float(window_vec(D, c, 10000, "honest:all7").median())
        e["honest_median"] = med
        if len(cc):
            e["ceiling_full_depth_median"] = float(cc.full_depth_median.iloc[0])
            e["fraction_of_ceiling"] = med / float(cc.full_depth_median.iloc[0])
        verdict["samples"][c] = e
    verdict["endpoint_met"] = bool(ok_all and len(verdict["samples"]) == len(CELLS))
    (OUT / "primary_endpoint_verdict.json").write_text(json.dumps(verdict, indent=1))

    # tables.md
    L = ["# Mouse transfer: tables (10 kb unless stated; autosome windows; identical pixels)\n"]
    L.append("## Samples\n")
    L.append(md(stats_table()))
    L.append("\n## Median per-window Pearson of log O/E\n")
    order = ["distance_decay", "distance_oracle", "untrained:all7", "honest_shuf:all7", "honest:none", "honest:ctcf",
             "honest:ctcf_k27ac", "honest_xm:all7", "honest_b4:all7", "honest:all7", "honest:avail", "consensus",
             "refin_shuf:all7", "refin_xm:all7", "refin:none", "refin:all7", "refres:all7", "diff:all7",
             "akita:Hsieh_mESC_uC", "akita:Bonev_ES", "akita:Bonev_CN", "akita:Bonev_NPC", "replicate_experiment"]
    for res in (10000, 25000):
        t = S[(S.res == res)].pivot(index="method", columns="cell", values="pearson_median")
        t = t.reindex([m for m in order if m in t.index])
        if res == 10000 and len(C):
            cr = C[C.res == res].set_index("cell").full_depth_median
            t.loc["split-half ceiling (Spearman-Brown)"] = [cr.get(c, np.nan) for c in t.columns]
        t = t.reset_index().rename(columns={"method": f"method ({mt.kb(res)})"})
        L.append(f"\n### {mt.kb(res)}\n")
        L.append(md(t))
        nw = S[(S.res == res) & (S.method == "honest:all7")].set_index("cell").n_windows
        L.append("\nwindows: " + ", ".join(f"{c} {int(nw.get(c, 0))}" for c in CELLS))
    L.append("\n## Paired contrasts of the median Pearson (95 percent bootstrap interval)\n")
    if len(P):
        q = P.assign(contrast=lambda d: d.m1 + " - " + d.m2,
                     ci=lambda d: d.apply(lambda r: f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]", axis=1))
        for res in (10000, 25000):
            t = q[q.res == res].pivot(index="contrast", columns="cell", values="ci").reset_index()
            L.append(f"\n### {mt.kb(res)}\n")
            L.append(md(t))
    L.append("\n## Structure classes, honest (7 marks) against controls, 10 kb\n")
    if len(CL):
        q = CL.assign(ci=lambda d: d.apply(lambda r: f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]", axis=1))
        for ref in ("honest_shuf:all7", "honest:none"):
            t = q[q.ref == ref].pivot(index="metric", columns="cell", values="ci").reset_index()
            L.append(f"\n### honest minus {ref}\n")
            L.append(md(t))
        t = SC[SC.method == "honest:all7"].pivot(index="metric", columns="cell", values="stat").reset_index()
        L.append("\n### honest (7 marks), median per window (boundary F1: mean)\n")
        L.append(md(t))
        t = SC[SC.method == "consensus"].pivot(index="metric", columns="cell", values="stat").reset_index()
        L.append("\n### consensus alone (mouse leave-one-out)\n")
        L.append(md(t))
    L.append("\n## Primary endpoint\n")
    L.append("```\n" + json.dumps(verdict, indent=1) + "\n```")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ figure
def figure():
    import matplotlib
    matplotlib.use("Agg")
    import cooler
    import matplotlib.pyplot as plt
    D = load_scores(("auto",))
    C = ceilings()
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.35], hspace=0.38)
    # A: median Pearson by method
    ax = fig.add_subplot(gs[0])
    meths = [("distance_decay", "distance\ndecay"), ("untrained:all7", "untrained"), ("honest_shuf:all7", "marks\nshuffled"),
             ("honest:none", "no marks"), ("honest_xm:all7", "marks of\nother sample"), ("honest:all7", "honest\n7 marks"),
             ("consensus", "mouse\nconsensus"), ("refin:all7", "refin\n+ consensus"), ("refres:all7", "refres\n+ consensus"),
             ("diff:all7", "diff\n+ consensus"), ("akita:Bonev_ES", "Akita v2\nES head")]
    w = 0.26
    cols = {"mesc": "#4C78A8", "ch12": "#F58518", "heart": "#54A24B"}
    for i, c in enumerate(CELLS):
        vals, lo, hi = [], [], []
        for m, _ in meths:
            v = window_vec(D, c, 10000, m).dropna().values
            if len(v) == 0:
                vals.append(np.nan); lo.append(0); hi.append(0)
                continue
            rng = np.random.default_rng(0)
            bm = np.median(v[rng.integers(0, len(v), (500, len(v)))], axis=1)
            vals.append(np.median(v)); lo.append(np.median(v) - np.percentile(bm, 2.5)); hi.append(np.percentile(bm, 97.5) - np.median(v))
        x = np.arange(len(meths)) + (i - 1) * w
        ax.bar(x, vals, w, yerr=[lo, hi], color=cols[c], label=LAB[c], capsize=1.5, linewidth=0)
        cr = C[(C.cell == c) & (C.res == 10000)]
        if len(cr):
            ax.axhline(cr.full_depth_median.iloc[0], color=cols[c], ls=":", lw=1)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(np.arange(len(meths)))
    ax.set_xticklabels([l for _, l in meths], fontsize=8)
    ax.set_ylabel("median per-window Pearson, log O/E")
    ax.set_title("A. Human-trained models on mouse marks, 10 kb, autosome windows (dotted: split-half ceiling)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    # B: structure classes, honest minus shuffled
    ax = fig.add_subplot(gs[1])
    cl = pd.read_csv(OUT / "structure_class_contrasts.csv")
    names = [("pearson", "Pearson"), ("scc", "SCC"), ("insulation_corr", "insulation r"), ("boundary_f1", "boundary F1\n(mean)"),
             ("loop_enrich_corr", "loop enrichment r"), ("loop_apa_ratio", "APA ratio"), ("compartment_corr", "compartment r"),
             ("stripe_corr", "stripe profile r")]
    for i, c in enumerate(CELLS):
        for j, (mname, _) in enumerate(names):
            r = cl[(cl.cell == c) & (cl.metric == mname) & (cl.ref == "honest_shuf:all7")]
            if not len(r):
                continue
            r = r.iloc[0]
            x = j + (i - 1) * w
            ax.bar(x, r["stat_m1"], w * 0.9, color=cols[c], alpha=0.95, linewidth=0)
            ax.plot([x, x], [r["stat_m2"]] * 2, marker="_", color="k", ms=7, lw=0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n for _, n in names], fontsize=8)
    ax.set_ylabel("honest, 7 marks (bars)\nshuffled marks (black ticks)")
    ax.set_title("B. Structure measures of the honest model on identical pixels; shuffled marks of the same chromosome as control", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # C: example regions, 80 bins per panel
    sub = gs[2].subgridspec(3, 5, wspace=0.08, hspace=0.35)
    res = 10000
    NB = 80
    for r_i, c in enumerate(CELLS):
        hv = window_vec(D, c, res, "honest:all7").dropna()
        key = (hv - hv.median()).abs().idxmin()
        chrom, rng_ = key.split(":")
        a0, a1 = [int(v) for v in rng_.split("-")]
        lo_b, hi_b = a0 // res, a1 // res
        T = np.asarray(cooler.Cooler(str(mt.oe_cool(c, res))).matrix(balance=False).fetch((chrom, a0, a1)), float)
        T = np.where(T == 0, np.nan, T)
        mp = {"measured": T}
        for m, nm in (("honest", "honest, 7 marks"), ("honest_shuf", "marks shuffled")):
            f = VM / "pred" / mt.kb(res) / m / c / "all7.npz" if m == "honest" else \
                VM / "pred" / mt.kb(res) / m / c / "all7.npz"
            if f.exists():
                z = np.load(f)
                if key in z.files:
                    mp[nm] = np.asarray(z[key], float)
        cc = np.asarray(cooler.Cooler(str(mt.loo_cool(c, res))).matrix(balance=False).fetch((chrom, a0, a1)), float)
        mp["mouse consensus"] = np.where(cc == 0, np.nan, cc)
        f = VM / "pred" / mt.kb(res) / "refin" / c / "all7.npz"
        if f.exists():
            z = np.load(f)
            if key in z.files:
                mp["refin + consensus"] = np.asarray(z[key], float)
        n = T.shape[0]
        s0 = (n - NB) // 2
        vmax = np.nanpercentile(np.abs(T[s0:s0 + NB, s0:s0 + NB]), 99)
        order = ["measured", "honest, 7 marks", "marks shuffled", "mouse consensus", "refin + consensus"]
        for j, nm in enumerate(order):
            axp = fig.add_subplot(sub[r_i, j])
            M = mp.get(nm)
            if M is None:
                axp.axis("off")
                continue
            blk = M[s0:s0 + NB, s0:s0 + NB]
            axp.imshow(blk, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            axp.set_xticks([]); axp.set_yticks([])
            if r_i == 0:
                axp.set_title(nm, fontsize=8)
            if j == 0:
                axp.set_ylabel(f"{LAB[c].split(' (')[0]}\n{chrom}\n{(lo_b + s0) * res // 10**6}.{((lo_b + s0) * res // 10**4) % 100:02d} Mb\n"
                               f"r = {hv.loc[key]:.2f}", fontsize=7)
    fig.text(0.01, 0.375, "C. Example windows (median Pearson of honest in the sample), 80 bins of 10 kb per panel, one colour scale per row (log O/E)", fontsize=10)
    fig.savefig(OUT / "fig_mouse_transfer.png", dpi=170, bbox_inches="tight")
    print("figure written")
