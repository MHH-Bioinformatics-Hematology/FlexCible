"""Markdown tables of results/one_model/METH_PROXY_PER_MARK.csv (registry section 97).

One table per view (mark removed from the full set / mark alone), bin size and use; rows are the 13 marks
ordered by the replacement fraction (RF) of the held-out cells where defined, then by loss. Columns:
loss [95% CI], gain [95% CI], RF (95% CI) for held-out and for WGBS training cells. n/d = loss below the
seed noise of the family or lower bootstrap bound of the loss not above 0 (section 78).

    python scripts/meth_proxy_per_mark_report.py > /tmp/tables.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GROUPS = ["held-out", "training WGBS"]
TITLE = {"minus": "mark removed from the full set of 13", "alone": "mark alone"}


def cell(r) -> str:
    f = lambda x: f"{x:.3f}"
    rf = "n/d" if not np.isfinite(r.rf) else f"{r.rf:.2f} ({r.rf_lo:.2f} to {r.rf_hi:.2f})"
    return (f"{f(r.loss)} [{f(r.loss_lo)}, {f(r.loss_hi)}] | {f(r.gain)} [{f(r.gain_lo)}, {f(r.gain_hi)}] | {rf}")


def main() -> None:
    d = pd.read_csv(REPO / "results/one_model/METH_PROXY_PER_MARK.csv")
    out = []
    for view in [v for v in ("minus", "alone") if (d.view == v).any()]:
        for res in sorted(d.res_kb.unique()):
            for use in ("consensus plane", "plane at 0"):
                t = d[(d.view == view) & (d.res_kb == res) & (d.use == use)]
                if t.empty:
                    continue
                ho = t[t.cell_group == "held-out"].set_index("mark")
                tr = t[t.cell_group == "training WGBS"].set_index("mark")
                order = sorted(ho.index, key=lambda m: (not np.isfinite(ho.rf[m]),
                                                        -ho.rf[m] if np.isfinite(ho.rf[m]) else -ho.loss[m]))
                out.append(f"### {TITLE[view]}, {res} kb, {use}\n")
                out.append(f"| mark | held-out ({int(ho.n_cells.iloc[0])} cells) loss [CI] | gain [CI] | RF (CI) | "
                           f"training WGBS ({int(tr.n_cells.iloc[0])} cells) loss [CI] | gain [CI] | RF (CI) |")
                out.append("|---|---|---|---|---|---|---|")
                for m in order:
                    out.append(f"| {m} | {cell(ho.loc[m])} | {cell(tr.loc[m])} |")
                out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
