"""Per-cell table: model next to the replicate-agreement noise ceiling.

Per cell (12 training + 5 held-out) and resolution (25, 10, 5, 2 kb), on the
413 Akita test regions, median per-region Pearson on the upper triangle (k>=2)
of the proper O/E map:

  model          all12refin_drop (and all12refres_drop), crosscell predict CSVs
  consensus      consensus alone (leave-one-out consensus for training cells,
                 consensus12 for held-out cells), same CSV
  honest         all12honest_drop, the same model trained without consensus
  rep_r          rep1 vs rep2 (results/replicates/replicate_corr.csv, single
                 lane, regridded to the nominal 5/25 kb grid) and, for
                 GM12878 and K562, the deepest intact pair
                 (replicate_corr_intact_deep.csv)

Ceiling. A prediction that knows the true signal correlates with a target of
reliability r_T at sqrt(r_T) (registry sections 12, 18, 29c). The reliability
of the target is taken from the replicate pair:
  - target is a single replicate (HMEC and HUVEC, whose target map is rep1):
        ceiling = sqrt(r_rep)
  - target is a deep merged map (all other cells): Spearman-Brown for the mean
    of two replicates, r_full = 2 r_rep / (1 + r_rep), ceiling = sqrt(r_full).
    The deep targets hold 5 to 40 times the contacts of a single-lane
    replicate, so this is a lower bound of the ceiling.
Cells without a replicate pair get NA (no imputation). HMEC deep and HUVEC deep
use the HMEC / HUVEC single-lane pair as a proxy (other experiment than the
deep target). ratio = model / ceiling, gap = ceiling - model.

    python scripts/replicate_ceiling_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("/mnt/processed/wolffjoa/hic-multiome/runs/multicell_reference")
TRAIN = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
         "gm23248", "mcf7", "cd14mono", "pc9"]
HELD = ["k562", "hmec", "hmec_deep", "huvec", "huvec_deep"]
SHARED = {"hmec": "hmec_deep", "hmec_deep": "hmec", "huvec": "huvec_deep",
          "huvec_deep": "huvec"}
REP_CELL = {"gm12878": "gm12878", "imr90": "imr90", "k562": "k562",
            "hmec": "hmec", "hmec_deep": "hmec", "huvec": "huvec",
            "huvec_deep": "huvec"}
SINGLE_TARGET = {"hmec", "huvec"}          # target map is rep1 itself
DEEP_PAIR = {"gm12878": "ENCSR965OVH", "k562": "ENCSR676BEP"}
RES = [25, 10, 5, 2]
NAMES = {"gm12878": "GM12878", "imr90": "IMR-90", "hct116": "HCT116", "hff": "HFF",
         "h1": "H1", "hepg2": "HepG2", "a549": "A549", "nhek": "NHEK",
         "gm23248": "GM23248", "mcf7": "MCF-7", "cd14mono": "CD14 mono", "pc9": "PC9",
         "k562": "K562", "hmec": "HMEC", "hmec_deep": "HMEC deep",
         "huvec": "HUVEC", "huvec_deep": "HUVEC deep"}


def run_dir(base: str, res: int) -> Path:
    return RUNS / (base if res == 2 else f"{base}_{res}kb")


def per_cell(base: str, res: int, cell: str):
    p = run_dir(base, res) / "crosscell" / f"predict_{cell}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def cons_col(d: pd.DataFrame):
    for c in ("base_consensus_loo", "base_consensus12"):
        if c in d:
            return c
    return None


def sb(r):
    return 2 * r / (1 + r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results/replicate_ceiling"))
    ap.add_argument("--thresh", type=float, default=0.95)
    ap.add_argument("--near", type=float, default=0.01)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rep = pd.read_csv(ROOT / "results/replicates/replicate_corr.csv").set_index("cell")
    deep = pd.read_csv(ROOT / "results/replicates/replicate_corr_intact_deep.csv")
    deep = deep.set_index(["cell", "experiment"])

    rows = []
    for cell in TRAIN + HELD:
        for res in RES:
            r = dict(cell=cell, name=NAMES[cell], group="held-out" if cell in HELD else "training",
                     shared_inputs_with=SHARED.get(cell, ""), res_kb=res)
            d = per_cell("all12refin_drop", res, cell)
            r["model_refin_drop"] = d.model.median() if d is not None else np.nan
            r["n_regions"] = int(d.model.notna().sum()) if d is not None else 0
            cc = cons_col(d) if d is not None else None
            r["consensus_alone"] = d[cc].median() if cc else np.nan
            r["consensus_kind"] = {"base_consensus_loo": "leave-one-out",
                                   "base_consensus12": "consensus12"}.get(cc, "")
            d2 = per_cell("all12refres_drop", res, cell)
            r["model_refres_drop"] = d2.model.median() if d2 is not None else np.nan
            d3 = per_cell("all12honest_drop", res, cell)
            r["honest_drop_no_consensus"] = d3.model.median() if d3 is not None else np.nan

            rc = REP_CELL.get(cell)
            if rc in rep.index:
                rr = float(rep.loc[rc, f"{res}kb"])
                r["replicate_r"] = rr
                r["rep_contacts_M"] = f"{rep.loc[rc, 'rep1_Mcontacts']:.0f}/{rep.loc[rc, 'rep2_Mcontacts']:.0f}"
                r["rep_kind"] = ("independent ENCODE biological replicates, one lane each (single-lane)")
                r["target_kind"] = ("single replicate (rep1)" if cell in SINGLE_TARGET
                                    else "deep merged map")
                r["rep_proxy"] = cell in ("hmec_deep", "huvec_deep")
                r["ceiling_sqrt_single"] = np.sqrt(rr)
                r["ceiling_sb_merged"] = np.sqrt(sb(rr))
                r["ceiling"] = r["ceiling_sqrt_single"] if cell in SINGLE_TARGET else r["ceiling_sb_merged"]
            else:
                r.update(replicate_r=np.nan, rep_contacts_M="", rep_kind="none",
                         target_kind="deep merged map", rep_proxy=False,
                         ceiling_sqrt_single=np.nan, ceiling_sb_merged=np.nan, ceiling=np.nan)
            if cell in DEEP_PAIR:
                rd = float(deep.loc[(cell, DEEP_PAIR[cell]), f"{res}kb"])
                r["deep_pair"] = DEEP_PAIR[cell]
                r["deep_replicate_r"] = rd
                r["ceiling_deep_pair_sb"] = np.sqrt(sb(rd))
            else:
                r.update(deep_pair="", deep_replicate_r=np.nan, ceiling_deep_pair_sb=np.nan)
            for k in ("model_refin_drop", "model_refres_drop"):
                r[k.replace("model_", "ratio_")] = r[k] / r["ceiling"]
            r["gap_refin_drop"] = r["ceiling"] - r["model_refin_drop"]
            r["gap_refres_drop"] = r["ceiling"] - r["model_refres_drop"]
            r["ratio_refin_drop_deep_pair"] = r["model_refin_drop"] / r["ceiling_deep_pair_sb"]
            r["gap_refin_drop_deep_pair"] = r["ceiling_deep_pair_sb"] - r["model_refin_drop"]
            r["model_minus_replicate_r"] = r["model_refin_drop"] - r["replicate_r"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "replicate_ceiling_table.csv", index=False, float_format="%.4f")

    # summary over groups
    srows = []
    for res in RES:
        for g in ("training", "held-out"):
            s = df[(df.res_kb == res) & (df.group == g)]
            sc = s[s.ceiling.notna()]
            for model in ("refin_drop", "refres_drop"):
                m = s[f"model_{model}"]
                srows.append(dict(
                    res_kb=res, group=g, model=model, n_cells=len(s),
                    median_model=m.median(), median_consensus=s.consensus_alone.median(),
                    median_honest_drop=s.honest_drop_no_consensus.median(),
                    n_with_ceiling=len(sc),
                    median_ceiling=sc.ceiling.median() if len(sc) else np.nan,
                    median_model_where_ceiling=sc[f"model_{model}"].median() if len(sc) else np.nan,
                    median_ratio=sc[f"ratio_{model}"].median() if len(sc) else np.nan,
                    median_gap=(sc.ceiling - sc[f"model_{model}"]).median() if len(sc) else np.nan,
                    n_ge_95pct=int((sc[f"ratio_{model}"] >= a.thresh).sum()),
                    n_within_0p01=int(((sc.ceiling - sc[f"model_{model}"]) <= a.near).sum()),
                    n_model_ge_replicate_r=int((sc[f"model_{model}"] >= sc.replicate_r).sum()),
                ))
    sm = pd.DataFrame(srows)
    sm.to_csv(out / "replicate_ceiling_summary.csv", index=False, float_format="%.4f")

    # compact markdown
    f = lambda x: "NA" if pd.isna(x) else f"{x:.3f}"
    L = ["Per cell: model (all12refin_drop) / consensus alone / honest_drop (no consensus) / "
         "ceiling / model:ceiling, median per-region Pearson, 413 Akita regions. "
         "H = held-out; * = shares inputs with the other HMEC / HUVEC map; "
         "ceiling NA = no replicate pair.", ""]
    for res in RES:
        L += [f"{res} kb", "",
              "| cell | set | model | consensus | no consensus | replicate r | ceiling | model/ceiling | gap |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cell in TRAIN + HELD:
            r = df[(df.cell == cell) & (df.res_kb == res)].iloc[0]
            nm = NAMES[cell] + ("*" if cell in SHARED else "")
            L.append(f"| {nm} | {'H' if cell in HELD else 'train'} | {f(r.model_refin_drop)} | "
                     f"{f(r.consensus_alone)} | {f(r.honest_drop_no_consensus)} | {f(r.replicate_r)} | "
                     f"{f(r.ceiling)} | {f(r.ratio_refin_drop)} | {f(r.gap_refin_drop)} |")
        L.append("")
    L += ["Summary (all12refin_drop; ceiling cells only for ratio and gap)", "",
          "| res | set | cells | median model | median consensus | median no consensus | cells with ceiling | "
          "median ceiling | median model/ceiling | >= 95% | within 0.01 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, s in sm[sm.model == "refin_drop"].iterrows():
        L.append(f"| {s.res_kb} | {s.group} | {s.n_cells} | {f(s.median_model)} | {f(s.median_consensus)} | "
                 f"{f(s.median_honest_drop)} | {s.n_with_ceiling} | {f(s.median_ceiling)} | "
                 f"{f(s.median_ratio)} | {s.n_ge_95pct}/{s.n_with_ceiling} | {s.n_within_0p01}/{s.n_with_ceiling} |")
    L += ["", "Deepest intact replicate pair as ceiling (GM12878 ENCSR965OVH, K562 ENCSR676BEP; "
          "Spearman-Brown, sqrt)", "",
          "| cell | res | model | deep pair r | ceiling | model/ceiling | gap |", "|---|---|---|---|---|---|---|"]
    for _, r in df[df.deep_pair != ""].dropna(subset=["ceiling_deep_pair_sb"]).iterrows():
        L.append(f"| {r['name']} | {r.res_kb} | {f(r.model_refin_drop)} | {f(r.deep_replicate_r)} | "
                 f"{f(r.ceiling_deep_pair_sb)} | {f(r.ratio_refin_drop_deep_pair)} | {f(r.gap_refin_drop_deep_pair)} |")
    (out / "replicate_ceiling_table.md").write_text("\n".join(L) + "\n")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = TRAIN + HELD
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, res in zip(axs.ravel(), RES):
        s = df[df.res_kb == res].set_index("cell").loc[order]
        x = np.arange(len(order))
        ax.axvspan(len(TRAIN) - 0.5, len(order) - 0.5, color="#eeeeee", zorder=0)
        ax.plot(x, s.consensus_alone, "s", color="#888888", ms=5, label="consensus alone")
        ax.plot(x, s.honest_drop_no_consensus, "^", color="#d98c1f", ms=5, label="no-consensus model")
        ax.plot(x, s.model_refin_drop, "o", color="#1f5fbf", ms=7, label="model (refin_drop)")
        ax.plot(x, s.ceiling, "_", color="#c0392b", ms=16, mew=2.5, label="replicate ceiling")
        ax.plot(x, s.replicate_r, "x", color="#c0392b", ms=6, label="rep1 vs rep2")
        for xi, (m, c) in enumerate(zip(s.model_refin_drop, s.ceiling)):
            if not np.isnan(c):
                ax.plot([xi, xi], [m, c], color="#c0392b", lw=1, alpha=0.6, zorder=1)
        ax.set_title(f"{res} kb")
        ax.set_ylabel("median Pearson, 413 regions")
        ax.grid(axis="y", alpha=0.3)
    for ax in axs[1]:
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([NAMES[c] + ("*" if c in SHARED else "") for c in order],
                           rotation=60, ha="right")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False)
    fig.text(0.5, 0.005, "Shaded: held-out cells. * HMEC / HMEC deep and HUVEC / HUVEC deep share inputs. "
             "Ceiling shown only where a replicate pair exists.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out / "replicate_ceiling.png", dpi=150)
    print(sm[sm.model == "refin_drop"].to_string())


if __name__ == "__main__":
    main()
