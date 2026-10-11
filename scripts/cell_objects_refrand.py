"""A_refrand controls for the cell-line objects table (registry section 70).

The cell-specific objects of scripts/cell_objects_table.py are defined by a weak reference enrichment
(loops: 3 x 3 maximum of the reference enrichment <= 0.2) and their shared controls A_ref by a strong one
(>= 0.6), so a ranking loss between them is solved by the reference alone, as for the AML tables. This
script adds, per cell line, random objects of the same kind matched on the cell's reference enrichment (the
leave-one-out consensus the model reads), distance and orientation (scripts/refrand_controls.py; no HSPC term)
and writes the table `cell_objects_refrand.csv.gz` = the original rows plus the set A_refrand, with the
matching values in ref_val.

    python scripts/cell_objects_refrand.py [--cells gm12878 ...] [--workers 3]
Output: share HM/cell_objects/cell_objects_refrand.csv.gz, results/cell_objects/refrand_match_quality.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
SHARE = HM / "cell_objects"
RES = 10000


def table_path() -> Path:
    return SHARE / "cell_objects_refrand.csv.gz"


def _job(job):
    import pickle

    import refrand_controls as rc
    cell, pos, ref = job
    dest = SHARE / "work" / f"refrand_{cell}.pkl"
    if dest.exists():
        return cell, pickle.load(open(dest, "rb"))
    al = pos[pos.kind == "loop"][["chrom", "a_lo", "b_lo"]].rename(columns={"a_lo": "i", "b_lo": "j"})
    anchors = {c: sorted(set(g.a_lo.astype(int))) for c, g in pos[pos.kind == "stripe"].groupby("chrom")}
    out = rc.refrand_controls(pos, str(ref), None, None, al, anchors, seed=sum(map(ord, cell)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(dest, "wb"))
    return cell, out


def build(d: pd.DataFrame, cells: list[str], ref_of, workers: int = 3) -> pd.DataFrame:
    """d: the original table; ref_of(cell) -> reference cool path. Returns d plus the A_refrand rows."""
    from concurrent.futures import ProcessPoolExecutor
    pos = d[(d["set"] == "pos") & d.cell.isin(cells)].copy()
    for c in ("a", "b"):
        pos[c + "_lo"] = (pos[c + "_start"] // RES).astype(int)
        pos[c + "_hi"] = (pos[c + "_end"] // RES).astype(int)
    jobs = [(c, g[["kind", "pair_id", "chrom", "a_lo", "a_hi", "b_lo", "b_hi"]].reset_index(drop=True), ref_of(c))
            for c, g in pos.groupby("cell")]
    new, vals = [], []
    with ProcessPoolExecutor(workers) as ex:
        for cell, (C, V) in ex.map(_job, jobs):
            if len(C):
                tp = pos[pos.cell == cell].set_index(["kind", "pair_id"])
                C = C.copy()
                C["cell"], C["patient"], C["fold"] = cell, cell, 0
                C["role"], C["set"] = "ctrl", "A_refrand"
                C["pair_type"] = [tp.pair_type.get(k, "other") for k in zip(C.kind, C.pair_id)]
                C["a_start"], C["a_end"] = C.a_lo * RES, C.a_hi * RES
                C["b_start"], C["b_end"] = C.b_lo * RES, C.b_hi * RES
                C["n_train_other"] = np.nan
                new.append(C)
            vals.append(V.assign(cell=cell))
    V = pd.concat(vals, ignore_index=True).drop_duplicates(["cell", "kind", "pair_id"]).set_index(["cell", "kind", "pair_id"])
    d = d.copy()
    d["ref_val"] = np.nan
    d["hspc_val"] = np.nan
    m = d["set"] == "pos"
    key = pd.MultiIndex.from_arrays([d.cell[m], d.kind[m], d.pair_id[m]])
    d.loc[m, "ref_val"] = V.ref_val.reindex(key).to_numpy()
    cols = list(d.columns)
    out = pd.concat([d] + [n[cols] for n in new], ignore_index=True)
    return out.sort_values(["cell", "kind", "set", "pair_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import multicell_expanded as mx
    import cell_objects_table as cot
    d = pd.read_csv(cot.table_path())
    cells = a.cells or sorted(d.cell.unique())
    cot.assert_no_leak(d)
    out = build(d, cells, lambda c: mx.consensus_loo(c, RES), a.workers)
    out.to_csv(table_path(), index=False)
    rows = []
    for (c, kind), g in out.groupby(["cell", "kind"]):
        P, C = g[g["set"] == "pos"], g[g["set"] == "A_refrand"]
        m = P.merge(C, on="pair_id", suffixes=("_p", "_c"))
        rows.append({"cell": c, "kind": kind, "n_pos": len(P), "n_matched": len(m),
                     "frac_matched": len(m) / max(1, len(P)),
                     "mean_pos_ref": m.ref_val_p.mean(), "mean_ctl_ref": m.ref_val_c.mean(),
                     "max_abs_diff_ref": (m.ref_val_p - m.ref_val_c).abs().max()})
    Q = pd.DataFrame(rows)
    (REPO / "results/cell_objects").mkdir(parents=True, exist_ok=True)
    Q.to_csv(REPO / "results/cell_objects/refrand_match_quality.csv", index=False, float_format="%.4f")
    print(Q.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
