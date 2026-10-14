"""Tables, paired bootstrap contrasts and figure of the window prior (registry section 76).

Reads the crosscell predict_<cell>.csv of the runs in runs/multicell_reference (12 training and 5 held-out
cells, 413 Akita regions) and the scores of scripts/window_prior_indep.py (four independent samples), writes
results/window_prior/*.csv, tables.md and fig_window_prior.png.

    python scripts/window_prior_summary.py --res 10000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
MR = HM / "runs/multicell_reference"
VN = HM / "validation_new"
OUT = REPO / "results/window_prior"
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248", "mcf7", "cd14mono", "pc9"]
HELD = ["k562", "hmec", "huvec", "hmec_deep", "huvec_deep"]
NEW = ["pc3", "panc1", "nk", "caco2"]
RUNS = {"honest": "all12honest_drop", "refres": "all12refres_drop", "diff": "all12diffcells_refres_drop",
        "wp_ref": "all12wp_ref_drop", "wp_self": "all12wp_self_drop", "wp_both": "all12wp_both_drop",
        "wp_refm": "all12wp_refm_drop"}
ORDER = ["honest", "refres", "diff", "wp_ref", "wp_self", "wp_both", "wp_refm"]
NB = 2000


def load_cell(cell: str, k: int) -> pd.DataFrame | None:
    """region x predictor Pearson of a cell from the crosscell tables of the runs."""
    cols, base = {}, {}
    for key, run in RUNS.items():
        f = MR / f"{run}_{k}kb" / "crosscell" / f"predict_{cell}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("region")
        cols[key] = d["model"]
        for c in d:
            if c.startswith("base_consensus") and "consensus" not in base:
                base["consensus"] = d[c]
            if c == "base_learned":
                base["P_" + key] = d[c]
    if not cols:
        return None
    df = pd.DataFrame({**cols, **base})
    df.index.name = "key"
    return df


def load_new(cell: str, k: int) -> pd.DataFrame | None:
    f = OUT / f"indep_scores_{cell}_{k}kb.csv.gz"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    p = d.pivot_table(index="key", columns="method", values="pearson")
    p.columns = [("consensus" if c == "consensus12" else c.split(":")[0]) for c in p.columns]
    return p


def boot(frames: dict, a: str, b: str, ref: str | None = None, seed: int = 0):
    """Pooled difference of medians a - b (and the fraction of the gap closed (a-b)/(ref-b)); windows
    resampled within cell, NB resamples. Returns dict or None."""
    cols = [a, b] + ([ref] if ref else [])
    parts = {}
    for c, df in frames.items():
        if all(x in df for x in cols):
            d = df[cols].dropna()
            if len(d) >= 10:
                parts[c] = d.values
    if not parts:
        return None
    rng = np.random.default_rng(seed)
    allv = np.concatenate(list(parts.values()))
    est = np.median(allv, 0)
    ds, fr = [], []
    for _ in range(NB):
        s = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in parts.values()])
        m = np.median(s, 0)
        ds.append(m[0] - m[1])
        if ref:
            fr.append((m[0] - m[1]) / (m[2] - m[1]) if abs(m[2] - m[1]) > 1e-6 else np.nan)
    out = {"a": a, "b": b, "n_cells": len(parts), "n_windows": len(allv), "median_a": est[0], "median_b": est[1],
           "diff": est[0] - est[1], "lo": np.percentile(ds, 2.5), "hi": np.percentile(ds, 97.5)}
    out["above"] = bool(out["lo"] > 0)
    if ref:
        out["median_ref"] = est[2]
        out["gap_closed"] = (est[0] - est[1]) / (est[2] - est[1]) if abs(est[2] - est[1]) > 1e-6 else np.nan
        out["gap_lo"], out["gap_hi"] = np.nanpercentile(fr, 2.5), np.nanpercentile(fr, 97.5)
    return out


def _md(df, index):
    """Pipe table without the tabulate dependency."""
    d = df.reset_index() if index else df
    rows = [list(map(str, d.columns))] + [[str(v) for v in r] for r in d.itertuples(index=False)]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=10000)
    a = ap.parse_args()
    k = a.res // 1000
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {"training_cells": {c: load_cell(c, k) for c in TRAIN12},
              "held_out_cells": {c: load_cell(c, k) for c in HELD},
              "new_samples": {c: load_new(c, k) for c in NEW}}
    groups = {g: {c: d for c, d in v.items() if d is not None} for g, v in groups.items()}
    # per cell medians
    rows = []
    for g, v in groups.items():
        for c, d in v.items():
            r = {"group": g, "cell": c, "res_kb": k, "n_regions": len(d)}
            r.update({m: round(float(d[m].median()), 4) for m in d.columns})
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / f"cells_{k}kb.csv", index=False)
    # contrasts
    con = []
    for g, v in groups.items():
        for x in ("wp_ref", "wp_self", "wp_both", "wp_refm"):
            pos = "refres" if x == "wp_self" else "diff"
            for b, ref in (("honest", pos),):
                r = boot(v, x, b, ref)
                if r:
                    con.append({"group": g, "contrast": f"{x} - {b}", "gap_reference": pos, **r})
            r = boot(v, x, pos)
            if r:
                con.append({"group": g, "contrast": f"{x} - {pos}", **r})
        r = boot(v, "wp_both", "diff")
        if r:
            con.append({"group": g, "contrast": "S2 wp_both - diff", **r})
        for pos in ("refres", "diff"):
            r = boot(v, pos, "honest")
            if r:
                con.append({"group": g, "contrast": f"{pos} - honest (the gap)", **r})
        # per cell
        for c, d in v.items():
            for x in ("wp_ref", "wp_self"):
                r = boot({c: d}, x, "honest")
                if r:
                    con.append({"group": f"{g}:{c}", "contrast": f"{x} - honest", **r})
    C = pd.DataFrame(con)
    C.to_csv(OUT / f"contrasts_{k}kb.csv", index=False, float_format="%.4f")
    # primary endpoint
    prim = {}
    for x in ("wp_ref", "wp_self"):
        for g, lab in (("held_out_cells", "E1"), ("new_samples", "E2")):
            s = C[(C.group == g) & (C.contrast == f"{x} - honest")]
            prim[f"{lab}_{x}"] = (None if s.empty else {"diff": float(s["diff"].iloc[0]), "lo": float(s["lo"].iloc[0]),
                                                      "hi": float(s["hi"].iloc[0]), "above": bool(s["above"].iloc[0])})
    prim["met"] = bool(all(v is not None and v["above"] for v in prim.values()))
    (OUT / f"primary_endpoint_{k}kb.json").write_text(json.dumps(prim, indent=1))
    # markdown
    md = [f"## {k} kb: median Pearson per cell (413 Akita test regions)\n"]
    T = pd.DataFrame(rows)
    cols = [c for c in ["honest", "refres", "diff", "wp_ref", "wp_self", "wp_both", "wp_refm", "consensus",
                        "P_wp_ref", "P_wp_self", "n_regions"] if c in T]
    md.append(_md(T.set_index("cell")[["group"] + cols], True))
    md.append("\n## Pooled paired contrasts (difference of medians, 95 percent bootstrap interval)\n")
    show = [c for c in ["group", "contrast", "n_cells", "n_windows", "median_a", "median_b", "diff", "lo", "hi",
                        "above", "gap_closed", "gap_lo", "gap_hi"] if c in C]
    md.append(_md(C[show].round(4), False))
    md.append("\n## Primary endpoint\n\n```\n" + json.dumps(prim, indent=1) + "\n```\n")
    (OUT / f"tables_{k}kb.md").write_text("\n".join(md))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [(g, c) for g in ("held_out_cells", "new_samples") for c in groups[g]]
    if cells:
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(cells) + 2), 4.2))
        style = {"honest": ("#555555", "o"), "refres": ("#0072B2", "s"), "diff": ("#56B4E9", "D"),
                 "wp_ref": ("#D55E00", "^"), "wp_self": ("#009E73", "v"), "wp_both": ("#CC79A7", "P"),
                 "consensus": ("#999999", "x")}
        for m, (col, mk) in style.items():
            ys = [groups[g][c][m].median() if m in groups[g][c] else np.nan for g, c in cells]
            if np.isfinite(ys).any():
                ax.plot(range(len(cells)), ys, mk, color=col, label=m, ms=7, ls="none")
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([c for _, c in cells], rotation=45, ha="right")
        ax.set_ylabel("median Pearson per region")
        ax.legend(frameon=False, ncol=4, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_window_prior_{k}kb.png", dpi=170)
    print((OUT / f"tables_{k}kb.md").read_text())


if __name__ == "__main__":
    main()
