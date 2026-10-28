"""Cell-type-specific loops, IMR-90 versus GM12878, 10 kb, held-out chromosomes (2, 3, 11, 14, 17).

Loops: hicDetectLoops calls of the measured maps (runs/benchmark/calls/<cell>_10kb_loops.bedgraph; columns
chrom1 start1 end1 chrom2 start2 end2 score). Every value at a loop is the maximum over the 3x3 pixel
neighbourhood. Difference d = IMR-90 minus GM12878 (measured or predicted). Classes (disjoint):
  shared        called in both maps (a call of the other map within 1 bin in both directions)
  IMR-90-spec.  called in IMR-90, not shared, measured d >= thr
  GM12878-spec. called in GM12878, not shared, measured d <= -thr
Only loops whose 3x3 value is finite in every predictor and in both measured maps are scored (identical set).
AUROC: positives = specific class, negatives = shared or the opposite specific class; score = predicted d
(IMR-90-specific positives) or -d (GM12878-specific positives).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distance_stratified import HELD, OE, PRED, SH, Table  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results/celltype_loops"
CALLS = SH / "runs/benchmark/calls"
RES = 10000
TOL = 1
THRS = (1.0, 0.5)
CELLS = ("imr90", "gm12878")


def src_path(kind, cell):
    k = "10kb"
    return {"measured": OE / f"{cell}_proper_{k}.cool",
            "ours": PRED / k / "all12refin_drop_chrho" / f"{cell}.cool",
            "chip": PRED / k / "all12honest_drop_chrho" / f"{cell}.cool",
            "consensus": OE / f"consensus_loo_{cell}_proper_{k}.cool"}[kind]


def read_calls(cell):
    d = pd.read_csv(CALLS / f"{cell}_10kb_loops.bedgraph", sep="\t", header=None,
                    names=["c1", "s1", "e1", "c2", "s2", "e2", "score"])
    d = d[(d.c1 == d.c2) & d.c1.isin(HELD)].copy()
    d["i"], d["j"] = d.s1 // RES, d.s2 // RES
    lo, hi = np.minimum(d.i, d.j), np.maximum(d.i, d.j)
    d["i"], d["j"] = lo, hi
    return d[["c1", "i", "j"]].rename(columns={"c1": "chrom"}).drop_duplicates().reset_index(drop=True)


def near(a, b):
    """Boolean per row of a: some call of b on the same chromosome within TOL bins in both coordinates."""
    out = np.zeros(len(a), bool)
    for ch, g in a.groupby("chrom"):
        bb = b[b.chrom == ch]
        S = set(zip(bb.i, bb.j))
        for ix, i, j in zip(g.index, g.i, g.j):
            out[ix] = any((i + a_, j + b_) in S for a_ in range(-TOL, TOL + 1) for b_ in range(-TOL, TOL + 1))
    return out


def val3(tabs, ch, i, j):
    """Max over the 3x3 neighbourhood of the pixel (i, j), bins; NaN when no neighbour is finite."""
    t = tabs[ch]
    st = np.full((9, len(i)), np.nan)
    k = 0
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            st[k] = t.at((i + a + 0.5) * RES, (j + b + 0.5) * RES)
            k += 1
    with np.errstate(all="ignore"):
        m = np.nanmax(st, axis=0)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = ["measured", "ours", "chip", "consensus"]
    calls = {c: read_calls(c) for c in CELLS}
    ncalls = {c: len(calls[c]) for c in CELLS}
    sh_i = near(calls["imr90"], calls["gm12878"])
    sh_g = near(calls["gm12878"], calls["imr90"])
    # loop table: every call of either map (shared ones taken from the IMR-90 call positions)
    A = calls["imr90"].assign(origin="imr90", shared=sh_i)
    B = calls["gm12878"].assign(origin="gm12878", shared=sh_g)
    loops = pd.concat([A, B[~B.shared]], ignore_index=True)  # shared GM12878 calls are already in A
    tabs = {(k, c): {} for k in kinds for c in CELLS}
    for k in kinds:
        for c in CELLS:
            for ch in HELD:
                tabs[(k, c)][ch] = Table(src_path(k, c), ch)
    for k in kinds:
        for c in CELLS:
            loops[f"{k}_{c}"] = np.nan
    for ch in HELD:
        m = (loops.chrom == ch).values
        i, j = loops.i.values[m], loops.j.values[m]
        for k in kinds:
            for c in CELLS:
                loops.loc[m, f"{k}_{c}"] = val3(tabs[(k, c)], ch, i, j)
    fin = np.ones(len(loops), bool)
    for k in kinds:
        for c in CELLS:
            fin &= np.isfinite(loops[f"{k}_{c}"].values)
    n_drop = int((~fin).sum())
    L = loops[fin].copy()
    for k in kinds:
        L[f"d_{k}"] = L[f"{k}_imr90"] - L[f"{k}_gm12878"]
    L.to_csv(OUT / "loops_scored.csv", index=False)
    rows, counts = [], []
    for thr in THRS:
        cls = np.where(L.shared, "shared",
                       np.where(L.d_measured >= thr, "imr90_specific",
                                np.where(L.d_measured <= -thr, "gm12878_specific", "unclassified")))
        cls = np.where((L.origin == "gm12878") & ~L.shared & (L.d_measured >= thr), "unclassified", cls)
        cls = np.where((L.origin == "imr90") & ~L.shared & (L.d_measured <= -thr), "unclassified", cls)
        L["cls"] = cls
        for q, n in L.cls.value_counts().items():
            counts.append((thr, q, int(n)))
        for k in kinds:
            for pos, negs in (("imr90_specific", ["shared", "gm12878_specific"]),
                              ("gm12878_specific", ["shared", "imr90_specific"])):
                sign = 1 if pos == "imr90_specific" else -1
                for neg in negs:
                    s = L[L.cls.isin([pos, neg])]
                    y = (s.cls == pos).astype(int).values
                    auc = roc_auc_score(y, sign * s[f"d_{k}"].values) if 0 < y.sum() < len(y) else np.nan
                    rows.append((thr, k, pos, neg, auc, int(y.sum()), int((1 - y).sum())))
    au = pd.DataFrame(rows, columns=["threshold", "predictor", "positives", "negatives", "auroc",
                                     "n_pos", "n_neg"])
    au["predictor"] = au.predictor.replace({"measured": "measured (ceiling)", "ours": "ours with consensus",
                                            "chip": "ours ChIP only", "consensus": "consensus alone"})
    au.to_csv(OUT / "auroc.csv", index=False)
    cn = pd.DataFrame(counts, columns=["threshold", "class", "n_loops"])
    cn.to_csv(OUT / "class_counts.csv", index=False)
    print(ncalls, "loops total", len(loops), "dropped non-finite", n_drop)
    print(cn.to_string())
    print(au.to_string())
    (OUT / "_stats.txt").write_text(f"calls_held_out={ncalls}\nunion_loops={len(loops)}\ndropped={n_drop}\n"
                                    f"scored={len(L)}\nshared_imr_calls={int(sh_i.sum())}\nshared_gm_calls={int(sh_g.sum())}\n")


if __name__ == "__main__":
    main()
