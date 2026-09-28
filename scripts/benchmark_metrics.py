"""Region-level metrics shared by the competitor benchmark, the minimal-input
evaluation and the perturbation analysis.

Every function takes square maps in log(observed/expected) space on one bin
grid (NaN = not measured / not predicted) and a boolean pixel mask that is the
same for every method scored in a block, so all methods are compared on
identical pixels.

pearson        Pearson over the masked upper-triangle pixels of a region.
scc            HiCRep stratum-adjusted correlation (Yang et al. 2017): per
               diagonal Pearson, weighted by N_d * sqrt(var(rank_x / N_d) *
               var(rank_y / N_d)); no extra smoothing (all maps here are
               already smoothed O/E). Per-diagonal centring makes it
               insensitive to a per-distance offset between two O/E
               definitions, which is why it is the fair metric wherever a
               competitor's value space differs from the target's.
per_distance   Pearson per diagonal, pooled over regions by the caller.
insulation     diamond insulation profile (hicmultiome.loops) of exp(map).
boundary_recall
               recall of reference boundaries (hicFindTADs on the measured
               count map) by the K most prominent insulation minima of a map,
               K = number of reference boundaries in the region; a rank cut,
               so smooth and sharp predictions are compared on placement.
loop metrics   hicmultiome.bioscore: donut enrichment, rank-matched loop calls
               (K = number of reference loops, hicDetectLoops on the measured
               count map), APA around the reference anchors, and the anchor
               AUROC against distance-matched control pixels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome import bioscore as bs                                # noqa: E402
from hicmultiome.loops import insulation_profile                      # noqa: E402


def triu_mask(n: int, k: int = 2, max_d: int | None = None) -> np.ndarray:
    i, j = np.indices((n, n))
    m = (j - i) >= k
    if max_d is not None:
        m &= (j - i) <= max_d
    return m


def pearson(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    a, b = x[mask], y[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    a, b = x[mask], y[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    return float(np.corrcoef(rankdata(a[ok]), rankdata(b[ok]))[0, 1])


def scc(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """HiCRep SCC over the diagonals present in the mask."""
    n = x.shape[0]
    num = den = 0.0
    for d in range(1, n):
        idx = np.arange(n - d)
        mk = mask[idx, idx + d]
        if mk.sum() < 3:
            continue
        a = x[idx, idx + d][mk].astype(np.float64)
        b = y[idx, idx + d][mk].astype(np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        m = int(ok.sum())
        if m < 3:
            continue
        a, b = a[ok], b[ok]
        if a.std() == 0 or b.std() == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(r):
            continue
        w = m * np.sqrt(np.var(rankdata(a) / m) * np.var(rankdata(b) / m))
        num += w * r
        den += w
    return num / den if den > 0 else float("nan")


def per_distance(x: np.ndarray, y: np.ndarray, mask: np.ndarray
                 ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Masked pixel pairs per diagonal, for pooling across regions."""
    n = x.shape[0]
    out = {}
    for d in range(1, n):
        idx = np.arange(n - d)
        mk = mask[idx, idx + d]
        a, b = x[idx, idx + d][mk], y[idx, idx + d][mk]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            out[d] = (a[ok].astype(np.float32), b[ok].astype(np.float32))
    return out


def insulation(logmap: np.ndarray, window: int) -> np.ndarray:
    oe = bs.oe_from_log(np.nan_to_num(logmap, nan=0.0))
    return insulation_profile(oe, window)


def minima_ranked(prof: np.ndarray, flank: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Local minima of an insulation profile, ordered by prominence."""
    n = prof.size
    pos, prom = [], []
    for b in range(1, n - 1):
        v = prof[b]
        if not np.isfinite(v) or not (v <= prof[b - 1] and v <= prof[b + 1]):
            continue
        left = prof[max(0, b - flank):b]
        right = prof[b + 1:b + 1 + flank]
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        if left.size == 0 or right.size == 0:
            continue
        pos.append(b)
        prom.append(min(left.max(), right.max()) - v)
    order = np.argsort(-np.asarray(prom)) if prom else np.zeros(0, int)
    return np.asarray(pos, int)[order], np.asarray(prom, float)[order]


def boundary_recall(logmap: np.ndarray, ref: np.ndarray, window: int, tol: int,
                    flank: int = 10) -> tuple[int, int]:
    """(hits, n_ref): reference boundaries recovered by the top-K minima."""
    if ref.size == 0:
        return 0, 0
    prof = insulation(logmap, window)
    pos, _ = minima_ranked(prof, flank)
    # a flat-bottomed minimum yields neighbouring tied minima; keep the most
    # prominent of any group closer than tol + 1 bins, so it counts once
    kept: list[int] = []
    for b in pos:
        if all(abs(b - k) > tol + 1 for k in kept):
            kept.append(int(b))
        if len(kept) == ref.size:
            break
    top = np.asarray(kept, int)
    if top.size == 0:
        return 0, int(ref.size)
    hits = int(sum(np.abs(top - b).min() <= tol for b in ref))
    return hits, int(ref.size)


def insulation_pearson(t: np.ndarray, p: np.ndarray, window: int) -> float:
    a, b = insulation(t, window), insulation(p, window)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or a[ok].std() == 0:
        return float("nan")
    if b[ok].std() == 0:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def loop_counts(t: np.ndarray, p: np.ndarray, ref: np.ndarray, prm: bs.BioParams,
                seed: int = 0) -> dict:
    """Rank-matched loop recall, APA stacks and anchor/control strengths.

    ``ref`` are reference loop anchors (i < j) on the grid of the maps after
    coarsening by prm.loop_coarsen. Returns additive quantities.
    """
    to, po = bs.oe_from_log(t), bs.oe_from_log(p)
    to[~np.isfinite(t)] = np.nan
    po[~np.isfinite(p)] = np.nan
    tc, pc = bs.coarsen(to, prm.loop_coarsen), bs.coarsen(po, prm.loop_coarsen)
    ep = bs.local_enrichment(pc, prm)
    out = {"loop_hits": 0, "loop_n": int(len(ref)), "apa_t": None, "apa_p": None,
           "apa_n": 0, "anchor": np.zeros(0), "control": np.zeros(0)}
    if len(ref) == 0:
        return out
    pa = bs.call_loops(pc, prm, ep, top_k=len(ref))[0]
    m = bs.match_loops(ref, pa, prm.loop_match_tol)
    out["loop_hits"] = m["tp_true"]
    st, sp, n = bs.apa_pair(tc, pc, ref, prm.apa_half)
    out["apa_t"], out["apa_p"], out["apa_n"] = st, sp, n
    out["anchor"] = bs._peak_strength(ep, ref[:, 0], ref[:, 1], prm.loop_match_tol)
    ca = bs._controls(ref, pc.shape[0], prm, np.random.default_rng(prm.seed + seed))
    if ca.size:
        out["control"] = bs._peak_strength(ep, ca[:, 0], ca[:, 1], prm.loop_match_tol)
    return out


def pool_loops(items: list[dict], prm: bs.BioParams) -> dict:
    hits = sum(i["loop_hits"] for i in items)
    n = sum(i["loop_n"] for i in items)
    w = 2 * prm.apa_half + 1
    st, sp, na = np.zeros((w, w)), np.zeros((w, w)), 0
    for i in items:
        if i["apa_n"]:
            st += i["apa_t"]; sp += i["apa_p"]; na += i["apa_n"]
    at = bs.apa_score(st, na, prm.apa_corner)
    ap = bs.apa_score(sp, na, prm.apa_corner)
    anc = np.concatenate([i["anchor"] for i in items]) if items else np.zeros(0)
    con = np.concatenate([i["control"] for i in items]) if items else np.zeros(0)
    anc, con = anc[np.isfinite(anc)], con[np.isfinite(con)]
    auroc = bs._auroc(anc, con) if anc.size > 5 and con.size > 5 else float("nan")
    return {"loop_recall": hits / n if n else float("nan"), "loop_n_ref": n,
            "apa_pred": ap, "apa_target": at, "apa_n": na,
            "apa_ratio": ((ap - 1) / (at - 1)) if np.isfinite(at) and at > 1 else float("nan"),
            "anchor_auroc": auroc}


def summarize(vals, prefix: str) -> dict:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_n": 0}
    return {f"{prefix}_median": float(np.median(v)), f"{prefix}_mean": float(v.mean()),
            f"{prefix}_q25": float(np.quantile(v, 0.25)),
            f"{prefix}_q75": float(np.quantile(v, 0.75)),
            f"{prefix}_sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            f"{prefix}_n": int(v.size)}
