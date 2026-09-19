"""Join Parts 1-3 into the tables that get reported.

Nothing new is measured here; this reads the three scoring scripts' CSVs and
answers the questions that need more than one of them at once.

The last table separates two things that are easy to conflate and that the
whole Part 3 claim depends on telling apart:

    "the MODEL fails here"        -- the input tracks are uninformative
    "the EXPERIMENT fails here"   -- mappability and coverage are bad

They are correlated with each other, and a region where both fail proves
nothing at all. The case that would support "we can predict where Hi-C cannot
measure" is the cell where the tracks are informative and the experiment is
not, so that cell is reported on its own.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _shallow_cols(cols) -> list:
    return [c for c in cols if str(c).startswith("shallow")]


def crossover(dep: pd.DataFrame, metric: str, total: float) -> pd.DataFrame:
    """At each depth: our score, the best the measurement can do, and who wins.

    The measurement's score is the best over the raw shallow map and all its
    smoothed versions, chosen per depth on the very data it is scored on. That
    is deliberately generous -- smoothing a shallow map is the first thing any
    analyst does, and picking the best smoothing in hindsight can only flatter
    the measurement, which is the conservative direction for the claim being
    tested.
    """
    rows = []
    for f, g in dep.groupby("fraction"):
        m = g.groupby("model")[metric].mean()
        sh = _shallow_cols(m.index)
        best = m[sh].idxmax()
        wide = g.pivot_table(index="region", columns="model", values=metric)
        won = np.nan
        if "ours" in wide and best in wide:
            d = (wide["ours"] - wide[_shallow_cols(wide.columns)].max(axis=1))
            won = float((d > 0).mean())
        rows.append({
            "fraction": f, "genome_contacts": total * f,
            "contacts_per_region":
                g[g.model == "shallow"].shallow_contacts.mean(),
            "measurement_raw": m.get("shallow", np.nan),
            "measurement_best": m[best], "best_variant": best,
            "ours": m.get("ours", np.nan), "akita": m.get("akita", np.nan),
            "hicgan": m.get("hicgan", np.nan),
            "ours_minus_measurement": m.get("ours", np.nan) - m[best],
            "frac_regions_ours_wins": won,
        })
    return pd.DataFrame(rows).sort_values("fraction", ascending=False)


def micro_boundary_f1(dep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (f, lab), g in dep.groupby(["fraction", "model"]):
        prec = g.b_tp_pred.sum() / max(1, g.b_n_pred.sum())
        rec = g.b_tp_true.sum() / max(1, g.b_n_true.sum())
        rows.append({"fraction": f, "model": lab,
                     "boundary_f1": 2 * prec * rec / (prec + rec)
                     if prec + rec else np.nan})
    B = pd.DataFrame(rows).pivot(index="fraction", columns="model",
                                 values="boundary_f1")
    B["measurement_best"] = B[_shallow_cols(B.columns)].max(axis=1)
    B["ours_minus_measurement"] = B["ours"] - B["measurement_best"]
    return B.sort_index(ascending=False)


def _plot(tables: dict, dep: pd.DataFrame, path: Path, total: float) -> None:
    """The crossover figure: at what depth is a prediction worth an experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("pearson", "distance-stratified pixel Pearson"),
              ("insulation_pearson", "insulation-profile Pearson"),
              ("boundary_f1", "boundary F1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, (key, label) in zip(axes, panels):
        if key == "boundary_f1":
            B = tables["boundary_f1"]
            x = B.index.values * total
            shal_raw, shal_best = B["shallow"].values, B["measurement_best"].values
            ours = B["ours"].values
        else:
            T = tables[key]
            x = T.genome_contacts.values
            shal_raw = T.measurement_raw.values
            shal_best = T.measurement_best.values
            ours = T.ours.values
        ax.plot(x, shal_raw, "o--", color="#888", label="measurement, raw")
        ax.plot(x, shal_best, "o-", color="#1f6feb",
                label="measurement, best smoothing")
        ax.plot(x, ours, "s-", color="#d1495b", label="our prediction")
        ax.set_xscale("log")
        ax.set_xlabel("sequencing depth of the shallow experiment\n"
                      "(genome-equivalent contacts)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "How deep an experiment is the prediction worth? Every curve is scored "
        "against the SAME reference:\nthe disjoint remainder of the deep map. "
        "Where the red line is above the blue one, the prediction recovers the "
        "reference\nbetter than a real experiment of that depth does.",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="runs/bioscore/depth4kb.csv")
    ap.add_argument("--per-region",
                    default="runs/bioscore/akita_bioscore_per_region.csv")
    ap.add_argument("--total-contacts", type=float, default=4.010478943e9)
    ap.add_argument("--target-fraction", type=float, default=0.03)
    ap.add_argument("--out", default="runs/bioscore/report")
    a = ap.parse_args()

    dep = pd.read_csv(a.depth)
    reg = pd.read_csv(a.per_region)
    tables: dict[str, pd.DataFrame] = {}

    print("=" * 78)
    print("PART 3a  down-sampling: prediction vs a real experiment of that depth")
    print("=" * 78)
    print(f"reference: the disjoint remainder of the deep map "
          f"({a.total_contacts:.3g} contacts genome-wide).\n"
          f"S ~ Binomial(C, f) is the shallow experiment, R = C - S is the "
          f"reference; they share\nno reads, so the measurement gets no credit "
          f"for reproducing its own noise.\n")
    for metric, label in [("pearson", "distance-stratified pixel Pearson"),
                          ("insulation_pearson", "insulation-profile Pearson")]:
        T = crossover(dep, metric, a.total_contacts)
        tables[metric] = T
        print(f"--- {label} ---")
        print(T.round(4).to_string(index=False))
        print()

    B = micro_boundary_f1(dep)
    tables["boundary_f1"] = B
    print("--- boundary F1 (micro-averaged over regions) ---")
    print(B.round(4).to_string())
    print()

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("PART 3b  does disagreement concentrate where the experiment is bad?")
    print("=" * 78)
    print("Low-confidence pixels are declared a priori from the deep cooler "
          "alone: a bin whose\nbalancing weight is NaN, or in the lowest "
          "coverage decile. The measurement's own\nreproducibility is the "
          "yardstick -- at the deepest split, S and R are two half-depth\n"
          "replicates of the same experiment. If reproducibility collapses at "
          "those pixels while\nthe predictions' agreement does not, the "
          "disagreement is the experiment's, not the model's.\n")
    deepest = dep.fraction.max()
    g = dep[dep.fraction == deepest]
    # Keep only regions where BOTH pixel classes exist. Otherwise the two
    # columns average over different sets of regions -- with a global
    # coverage threshold many regions have no low-confidence bins at all --
    # and their ratio measures which regions have bad bins rather than what
    # happens at bad bins.
    ok = g.dropna(subset=["pearson_highconf", "pearson_lowconf"])
    counts = ok.region.value_counts()
    g = ok[ok.region.isin(counts[counts == g.model.nunique()].index)]
    rows = []
    for lab, gg in g.groupby("model"):
        hi, lo = gg.pearson_highconf.mean(), gg.pearson_lowconf.mean()
        rows.append({"model": lab, "n_regions": len(gg),
                     "pearson_highconf": hi, "pearson_lowconf": lo,
                     "drop": hi - lo, "retained": lo / hi if hi else np.nan})
    C = pd.DataFrame(rows).sort_values("model")
    tables["lowconf"] = C
    print(f"at f = {deepest} (S and R are two independent half-depth replicates)")
    print(C.round(4).to_string(index=False))

    w = g.pivot_table(index="region", columns="model",
                      values=["pearson_highconf", "pearson_lowconf"])
    if ("pearson_highconf", "ours") in w and ("pearson_highconf", "shallow") in w:
        ret_o = (w[("pearson_lowconf", "ours")]
                 / w[("pearson_highconf", "ours")])
        ret_s = (w[("pearson_lowconf", "shallow")]
                 / w[("pearson_highconf", "shallow")])
        ok = np.isfinite(ret_o) & np.isfinite(ret_s)
        st = wilcoxon(ret_o[ok], ret_s[ok])
        print(f"\npaired over {int(ok.sum())} regions, retained fraction "
              f"(low-conf / high-conf):\n  ours   median "
              f"{np.median(ret_o[ok]):.3f}\n  shallow median "
              f"{np.median(ret_s[ok]):.3f}\n  Wilcoxon signed-rank "
              f"p = {st.pvalue:.3g}; ours retains more in "
              f"{100 * (ret_o[ok] > ret_s[ok]).mean():.0f}% of regions")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("PART 3c  'the model fails here' vs 'the experiment fails here'")
    print("=" * 78)
    rel = (dep.groupby("region").frac_low_conf_bins.first().reset_index())
    j = reg.merge(rel, on="region", how="inner")
    j["tracks"] = np.where(j["class"] == "signal-poor",
                           "tracks poor", "tracks informative")
    med = j.frac_low_conf_bins.median()
    j["experiment"] = np.where(j.frac_low_conf_bins > med,
                               "experiment poor", "experiment clean")
    tf = float(min(dep.fraction.unique(), key=lambda x: abs(x - a.target_fraction)))
    sub = dep[dep.fraction == tf]
    wide = sub.pivot_table(index="region", columns="model", values="pearson")
    wide["measurement_best"] = wide[_shallow_cols(wide.columns)].max(axis=1)
    j2 = j.merge(wide.reset_index()[["region", "ours", "measurement_best"]],
                 on="region", how="inner")
    j2["advantage"] = j2["ours"] - j2["measurement_best"]
    cell = (j2.groupby(["tracks", "experiment"])
            .agg(n_regions=("region", "size"),
                 low_conf_bin_frac=("frac_low_conf_bins", "mean"),
                 ours_vs_akita_target=("ours_pearson", "mean"),
                 ours_vs_ref=("ours", "mean"),
                 measurement_vs_ref=("measurement_best", "mean"),
                 advantage=("advantage", "mean"),
                 frac_regions_ours_wins=("advantage", lambda s: (s > 0).mean()))
            .reset_index())
    tables["cells"] = cell
    print(f"split at the median low-confidence-bin fraction ({med:.3f}); the "
          f"tracks class comes\nfrom the input tracks only. Scored at f = {tf} "
          f"({a.total_contacts * tf:.3g} genome-equivalent\ncontacts), pixel "
          f"Pearson against the disjoint reference.\n")
    print(cell.round(4).to_string(index=False))
    print("\nThe cell that would support the claim is 'tracks informative' + "
          "'experiment poor'.\n'tracks poor' + 'experiment poor' proves "
          "nothing either way and is reported for that reason.")

    # ------------------------------------------------------------------ #
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _plot(tables, dep, out.with_name(out.name + "_depth_crossover.png"),
          a.total_contacts)
    for name, T in tables.items():
        T.to_csv(out.with_name(out.name + f"_{name}.csv"),
                 index=name == "boundary_f1")
    j2.to_csv(out.with_name(out.name + "_region_cells.csv"), index=False)
    print(f"\nwrote {out}_*.csv")


if __name__ == "__main__":
    main()
