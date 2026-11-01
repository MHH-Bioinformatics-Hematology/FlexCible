"""Tables of registry section 103: measured maps of other cells and their consensus against the
published predictors. Reads the blocks of scripts/motivation_other_cells.py, the AML table of
scripts/motivation_aml.py and the Akita table of scripts/motivation_akita413.py.

Outputs (results/motivation/):
  per_row.csv          one row per (setting, target, bin size, block, source): the scores of the
                       competitor, of the consensus and of every other cell on the same pixels
  summary.csv          one row per (setting, competitor, target, bin size, measure): competitor score,
                       consensus score, number and share of other cells above the competitor, median
                       and range of the other-cell scores, best other cell
  MOTIVATION.md        tables and reading (reading.md is inserted verbatim when it exists)
  check_vs_stored.csv  the competitor rows of the new blocks against the stored benchmark rows
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/motivation"
BLOCKS = Path("/mnt/processed/wolffjoa/hic-multiome/runs/motivation/blocks")

MEAS = {"oe_pearson": "pearson", "oe_pearson_tile_median": "pearson_tile", "oe_scc": "scc", "HiCRep": "hicrep",
        "insulation_r": "insulation_r", "bnd_recall": "boundary_recall", "loop_recall_rank": "loop_recall",
        "comp_pref_r": "compartment_r"}
HEAD = ["pearson", "scc", "insulation_r", "boundary_recall", "loop_recall"]      # measures of the tables below
EXTRA = ["loop_enrich_r", "compartment_r"]          # AML (loop enrichment r in the windows) and compartment r
ALL_M = list(MEAS.values())


def method_of(name: str) -> tuple[str, str]:
    """(method, variant) of a competitor source name."""
    if name.startswith("corigami_gm12878_denovo"):
        return "C.Origami", "IMR-90 model on GM12878 (authors' tracks)"
    if name.startswith("corigami"):
        return "C.Origami", "IMR-90 model, released"
    if name.startswith("epiphany"):
        return "Epiphany", "GM12878 model, released"
    if name.startswith("hicgan_gm12878_on_k562"):
        return "Hi-cGAN", "GM12878 model on K562"
    if name.startswith("hicgan_gm12878_on_imr90"):
        return "Hi-cGAN", "GM12878 model on IMR-90"
    if name.startswith("hicgan_imr90"):
        return "Hi-cGAN", "IMR-90 model"
    if name.startswith("hicgan"):
        return "Hi-cGAN", "GM12878 model"
    if name.startswith("hicreg"):
        v = {"hicregwindowcrosschrom": "cross-chromosome", "hicregmulticellcrosschrom": "multi-cell cross-chromosome",
             "hicregwindowcv": "cross-validated"}
        return "HiC-Reg", v.get(name.split("_")[0], name.split("_")[0])
    return name, ""


def load_chrom_blocks() -> pd.DataFrame:
    fs = sorted(BLOCKS.glob("*__motiv/measures.csv"))
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    D = D[D.source != "target"].copy()
    D["source_type"] = np.where(D.kind == "competitor", "competitor",
                                np.where(D.source == "consensus", "consensus",
                                         np.where(D.source.str.startswith("other_"), "other_cell", "drop")))
    D = D[D.source_type != "drop"].copy()
    D["other_cell"] = np.where(D.source_type == "other_cell", D.source.str.replace("other_", "", regex=False), "")
    D["setting"] = np.where(D.set == "chrho_codenovo", "C.Origami de novo (section 81)", "held-out chromosomes")
    D["res_kb"] = D.res // 1000
    D = D.rename(columns=MEAS)
    return D


def summarise(per_unit: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    """per_unit: one row per (keys..., source_type, source) with the scores. Rows of the summary: per
    competitor source and measure."""
    rows = []
    for k, g in per_unit.groupby(keys, dropna=False):
        comp = g[g.source_type == "competitor"]
        cons = g[g.source_type == "consensus"]
        oth = g[g.source_type == "other_cell"]
        for _, c in comp.iterrows():
            for m in value_cols:
                if not np.isfinite(c[m]):
                    continue
                o = oth[m].dropna()
                if o.empty:
                    continue
                cv = float(cons[m].iloc[0]) if len(cons) and np.isfinite(cons[m].iloc[0]) else np.nan
                oc = oth.loc[o.index, "other_cell"]
                r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
                r.update(competitor=c.source, measure=m, competitor_score=float(c[m]), consensus_score=cv,
                         consensus_above=bool(cv > c[m]) if np.isfinite(cv) else np.nan,
                         n_other=int(o.size), n_other_above=int((o > c[m]).sum()),
                         share_other_above=float((o > c[m]).mean()), other_median=float(o.median()),
                         other_min=float(o.min()), other_max=float(o.max()), best_other=oc[o.idxmax()],
                         lowest_other=oc[o.idxmin()])
                rows.append(r)
    return pd.DataFrame(rows)


def chrom_part() -> tuple[pd.DataFrame, pd.DataFrame]:
    D = load_chrom_blocks()
    keep = ["setting", "set", "cell", "res_kb", "chrom", "block", "source_type", "source", "other_cell", "n_pixels"] + ALL_M
    per_row = D[keep].rename(columns={"cell": "target"})
    # mean over the held-out chromosomes of a block family, as the benchmark tables read it
    key = ["setting", "set", "cell", "res_kb"]
    U = (D.groupby(key + ["source_type", "source", "other_cell"], dropna=False)[ALL_M].mean().reset_index())
    nch = D.groupby(key).chrom.nunique().rename("n_chromosomes").reset_index()
    S = summarise(U, key, ALL_M).merge(nch, on=key)
    S = S.rename(columns={"cell": "target"})
    S["method"], S["variant"] = zip(*S.competitor.map(method_of))
    return per_row, S


def check_stored() -> pd.DataFrame:
    D = load_chrom_blocks()
    D = D[D.source_type == "competitor"]
    st = []
    for f in ("chrho_per_block.csv", "chrho_25kb_per_block.csv", "chrho_5kb_per_block.csv", "chrho_corigami_denovo_per_block.csv"):
        st.append(pd.read_csv(REPO / "results/hic_measures_chrho" / f))
    st = pd.concat(st).drop_duplicates(["block", "source"]).rename(columns=MEAS)
    m = D.merge(st, on=["set", "cell", "res", "chrom", "source"], suffixes=("", "_stored"))
    rows = []
    for c in ["pearson", "scc", "hicrep", "insulation_r", "boundary_recall", "loop_recall"]:
        d = (m[c] - m[c + "_stored"]).abs().dropna()
        rows.append(dict(measure=c, n=int(d.size), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean())))
    return pd.DataFrame(rows)


def aml_part() -> tuple[pd.DataFrame, pd.DataFrame]:
    A = pd.read_csv(OUT / "aml_per_patient.csv", dtype={"patient": str})
    comp = {"corigami_q": "C.Origami (quantile-matched inputs)", "epiphany_mean": "Epiphany (H3K4me3 = mean)",
            "epiphany_zero": "Epiphany (H3K4me3 = 0)", "akita": "Akita (sequence only)"}
    A["source_type"] = np.where(A.method.isin(comp), "competitor", np.where(A.method == "mean_other_patients", "consensus",
                                np.where(A.method.str.startswith("patient_"), "other_cell", "extra")))
    A["other_cell"] = np.where(A.source_type == "other_cell", A.method.str.replace("patient_", "", regex=False), "")
    ren = {"pearson": "pearson", "scc": "scc", "ins_r_50k": "insulation_r", "bnd_recall": "boundary_recall",
           "loop_enrich_corr": "loop_enrich_r", "compartment_corr": "compartment_r"}
    A = A.rename(columns=ren).rename(columns={"method": "source", "patient": "target"})
    A["setting"], A["res_kb"], A["set"] = "AML patients", 10, "aml_windows"
    per_row = A[["setting", "set", "target", "res_kb", "source_type", "source", "other_cell", "n_windows"] + list(ren.values())]
    U = A[A.source_type != "extra"].rename(columns={"target": "cell"})
    S = summarise(U, ["setting", "set", "cell", "res_kb"], list(ren.values())).rename(columns={"cell": "target"})
    S["method"] = S.competitor.map(comp)
    S["variant"] = ""
    return per_row, S


def akita_part() -> tuple[pd.DataFrame, pd.DataFrame]:
    R = pd.read_csv(OUT / "akita413_per_region.csv.gz")
    R = R[R.pearson.notna()].copy()
    R["source_type"] = np.where(R.method.isin(["akita", "hicgan"]), "competitor",
                                np.where(R.method == "consensus_other4", "consensus", "other_cell"))
    R["other_cell"] = np.where(R.source_type == "other_cell", R.method.str.replace("other_", "", regex=False), "")
    # pooled recall over the regions (the definition of the Akita table of section 73)
    g = R.groupby(["target", "method"])
    P = pd.DataFrame({"pearson": g.pearson.median(), "scc": g.scc.median(), "insulation_r": g.ins_r_50k.median(),
                      "compartment_r": g.compartment_corr.median(),
                      "boundary_recall": g.hft_hits.sum() / g.hft_n.sum(),
                      "loop_recall": g.loop_hits.sum() / g.loop_n.sum(),
                      "n_regions": g.pearson.size()}).reset_index()
    P["source_type"] = np.where(P.method.isin(["akita", "hicgan"]), "competitor",
                                np.where(P.method == "consensus_other4", "consensus", "other_cell"))
    P["other_cell"] = np.where(P.source_type == "other_cell", P.method.str.replace("other_", "", regex=False), "")
    P = P.rename(columns={"method": "source"})
    P["setting"], P["res_kb"], P["set"] = "Akita 413 regions (2 kb grid)", 2, "akita413"
    mm = ["pearson", "scc", "insulation_r", "boundary_recall", "loop_recall", "compartment_r"]
    U = P.rename(columns={"target": "cell"})
    S = summarise(U, ["setting", "set", "cell", "res_kb"], mm).rename(columns={"cell": "target"})
    S["method"] = S.competitor.map({"akita": "Akita", "hicgan": "Hi-cGAN"})
    S["variant"] = "released predictions"
    # per-region share of regions in which a given other cell / the consensus beats Akita (Pearson)
    W = R.pivot_table(index=["target", "region"], columns="method", values="pearson")
    reg = []
    for t, w in W.groupby(level=0):
        for m in w.columns:
            if m == "akita":
                continue
            ok = w[["akita", m]].dropna()
            reg.append(dict(target=t, source=m, regions=len(ok), share_regions_above_akita=float((ok[m] > ok["akita"]).mean())))
    pd.DataFrame(reg).to_csv(OUT / "akita413_region_wins.csv", index=False, float_format="%.4f")
    return P[["setting", "set", "target", "res_kb", "source_type", "source", "other_cell", "n_regions"] + mm], S


# ---------------------------------------------------------------- markdown
def f3(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def table(S: pd.DataFrame, measure: str, label_cols=("method", "variant", "target", "res_kb")) -> str:
    d = S[S.measure == measure].copy()
    d = d.sort_values(["method", "variant", "res_kb", "target"], ascending=[True, True, False, True])
    h = "| method | target | bin (kb) | competitor | consensus | other cells above | share | other median [min, max] | best other |\n"
    h += "|---|---|---|---|---|---|---|---|---|\n"
    rows = ""
    for _, r in d.iterrows():
        v = f" ({r.variant})" if r.variant else ""
        rows += (f"| {r.method}{v} | {r.target} | {r.res_kb} | {f3(r.competitor_score)} | {f3(r.consensus_score)}"
                 f"{' *' if r.consensus_above else ''} | {r.n_other_above}/{r.n_other} | {r.share_other_above:.2f} | "
                 f"{f3(r.other_median)} [{f3(r.other_min)}, {f3(r.other_max)}] | {r.best_other} |\n")
    return h + rows


def headline(S: pd.DataFrame, measures=None) -> str:
    """Per method, bin size and measure: pooled over targets and blocks, share of (target, other cell)
    pairs above the competitor, and the number of targets in which the consensus is above."""
    measures = measures or [m for m in HEAD + EXTRA if m in set(S.measure) and m != "compartment_r"]
    rows = []
    for (mth, res, m), g in S[S.measure.isin(measures)].assign(method=lambda x: x.method + np.where(x.variant != "", " (" + x.variant + ")", "")).groupby(["method", "res_kb", "measure"]):
        rows.append(dict(method=mth, res_kb=res, measure=m, n_cases=len(g),
                         share_pairs=g.n_other_above.sum() / g.n_other.sum(),
                         median_case_share=g.share_other_above.median(),
                         consensus_above=int(g.consensus_above.sum()), n=len(g)))
    H = pd.DataFrame(rows)
    out = "| method | bin (kb) | " + " | ".join(measures) + " |\n|---|---|" + "---|" * len(measures) + "\n"
    for (mth, res), g in H.sort_values(["method", "res_kb"], ascending=[True, False]).groupby(["method", "res_kb"], sort=False):
        cells = []
        for m in measures:
            x = g[g.measure == m]
            cells.append("" if x.empty else f"{x.share_pairs.iloc[0]:.2f} ({x.consensus_above.iloc[0]}/{x.n.iloc[0]})")
        out += f"| {mth} | {res} | " + " | ".join(cells) + " |\n"
    return out, H


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pr_c, S_c = chrom_part()
    parts_r, parts_s = [pr_c], [S_c]
    have_aml = (OUT / "aml_per_patient.csv").exists()
    have_ak = (OUT / "akita413_per_region.csv.gz").exists()
    if have_aml:
        a, b = aml_part()
        parts_r.append(a)
        parts_s.append(b)
    if have_ak:
        a, b = akita_part()
        parts_r.append(a)
        parts_s.append(b)
    per_row = pd.concat(parts_r, ignore_index=True)
    S = pd.concat(parts_s, ignore_index=True)
    per_row.to_csv(OUT / "per_row.csv", index=False, float_format="%.4f")
    cols = ["setting", "set", "method", "variant", "competitor", "target", "res_kb", "measure", "competitor_score",
            "consensus_score", "consensus_above", "n_other", "n_other_above", "share_other_above", "other_median",
            "other_min", "other_max", "best_other", "lowest_other", "n_chromosomes"]
    S = S[[c for c in cols if c in S.columns]]
    S.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    chk = check_stored()
    chk.to_csv(OUT / "check_vs_stored.csv", index=False, float_format="%.4f")

    md = ["# Measured maps of other cells and the consensus against published predictors\n"]
    md.append("Generated by `scripts/motivation_report.py` from `summary.csv` and `per_row.csv`; registry section 103.\n")
    rd = OUT / "reading.md"
    if rd.exists():
        md.append(rd.read_text())
    hs = {}
    for setting in S.setting.unique():
        d = S[S.setting == setting]
        t, H = headline(d)
        hs[setting] = H
        md.append(f"## {setting}\n")
        md.append("Share of (target, other cell) pairs whose measured map scores above the published predictor "
                  "on the same pixels; in brackets the number of target blocks in which the consensus scores "
                  "above the predictor / number of blocks.\n")
        md.append(t)
        for m in HEAD + EXTRA:
            if m in set(d.measure):
                md.append(f"\n### {setting}: {m}\n")
                md.append(table(d, m))
    md.append("\n## Pixel check\n\nCompetitor rows of the new blocks against the stored rows of the benchmark tables "
              "(same block, same chromosome; the new blocks exclude our models from the pixel set):\n\n")
    md.append("| measure | rows | max abs diff | mean abs diff |\n|---|---|---|---|\n" + "".join(
        f"| {r.measure} | {r.n} | {r.max_abs_diff:.4f} | {r.mean_abs_diff:.4f} |\n" for r in chk.itertuples()))
    (OUT / "MOTIVATION.md").write_text("\n".join(md))
    print((OUT / "MOTIVATION.md").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
