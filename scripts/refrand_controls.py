"""Random controls matched on the reference and HSPC enrichment ('A_refrand', registry section 69/70).

The shared-loop controls of the section 63 tables (A_hspc: shared loops chosen for a strong signal in the
patient AND the HSPC consensus) are separable from AML-specific loops by the reference / HSPC enrichment
alone (AUROC 0.97 / 0.99), so a ranking loss on them is solved by the shortcut 'residual positive where the
reference is weak'. The controls built here make the shortcut unavailable: for every AML-specific object of
a TRAINING patient a random object of the same kind on the same chromosome, matched on

    reference enrichment   |difference| <= 0.15  (loops: log2 donut enrichment, 3 x 3 maximum, of the map the
                                                   model reads as reference; stripes: mean log O/E of the rectangle)
    HSPC enrichment        |difference| <= 0.15  (the measured HSPC consensus; omitted for cell lines)
    distance               loops |log10 d' - log10 d| <= 0.10 (stripes: the same geometry, exactly)
    orientation class      exact (CTCF motif orientation, as the classifier)

one control per object, no control reused, never within 2 bins of an AML-specific object of the patient
(stripes: +-8 bins of an anchor), on bins that are measured in the patient. This is the construction of the
'A_refrand' pool of scripts/aml_loop_detection.py stage criteria2 (random pairs, same calipers), with all
candidates on the chromosome instead of one random pair per object.

The functions take arrays; the drivers read the cools (reference per training patient: the leave-one-out
reference the model sees, HSPC consensus) and call them. Run with the hicdiff env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CAL_ENR = 0.15            # caliper on the reference and HSPC enrichment
CAL_DIST = 0.10           # caliper on log10 distance (loops)
NEAR = 2                  # bins: no control within this Chebyshev distance of an AML-specific loop
NEAR_ANCHOR = 8           # bins: no stripe control with its anchor this close to an AML-specific anchor
MIN_BINS, MAX_BINS = 6, 150
TOPK = 8                  # a control is drawn at random among the TOPK closest free candidates


# --------------------------------------------------------------------------- arrays
def orient_codes(strand, score, i, j) -> np.ndarray:
    """0 none, 1 convergent, 2 divergent, 3 tandem (aml_loop_classifier.orient_class, vectorised)."""
    sa, sb, ca, cb = strand[i], strand[j], score[i], score[j]
    none = (ca <= 0) | (cb <= 0) | (sa == 0) | (sb == 0)
    return np.where(none, 0, np.where((sa == 1) & (sb == -1), 1, np.where((sa == -1) & (sb == 1), 2, 3)))


def anchor_codes(strand, score, a, ra: int, dirn: int) -> np.ndarray:
    """Stripe orientation: 0 no motif in the anchor, 1 the strongest motif faces the extent, 2 faces away."""
    a = np.asarray(a)
    out = np.zeros(len(a), dtype=int)
    for k, s in enumerate(a):
        seg = score[s:s + ra]
        if seg.size == 0 or seg.max() <= 0:
            continue
        st = strand[s + int(np.argmax(seg))]
        out[k] = 1 if st == dirn else 2
    return out


def match_loops(pos: pd.DataFrame, Eref, Eh, strand, score, valid, avoid_ij, rng,
                cal: float = CAL_ENR, cal_d: float = CAL_DIST):
    """Controls for the loops `pos` (columns i, j, bins; one chromosome).

    Eref, Eh: (n, MAX_BINS + 1) enrichment of pixel (i, i + d), NaN where unmeasured (Eh None: no HSPC
    matching). strand / score: motif arrays of the chromosome. valid: bool (n,) measured bins of the patient or
    None. avoid_ij: (m, 2) pixels of AML-specific loops. Returns (list of (row, i', j')), reference value and
    HSPC value of every positive (NaN where undefined)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(avoid_ij, float)) if len(avoid_ij) else None
    n = Eref.shape[0]
    used: set = set()
    out = []
    rv = np.full(len(pos), np.nan)
    hv = np.full(len(pos), np.nan)
    ii0, jj0 = pos.i.to_numpy(), pos.j.to_numpy()
    oc0 = orient_codes(strand, score, ii0, jj0)
    for r in rng.permutation(len(pos)):
        i0, j0 = int(ii0[r]), int(jj0[r])
        d = j0 - i0
        rv[r] = Eref[i0, d]
        hv[r] = Eh[i0, d] if Eh is not None else np.nan
        if not np.isfinite(rv[r]) or (Eh is not None and not np.isfinite(hv[r])):
            continue
        dmin = max(MIN_BINS, int(np.ceil(d * 10 ** -cal_d)))
        dmax = min(MAX_BINS, int(np.floor(d * 10 ** cal_d)))
        m = np.abs(Eref[:, dmin:dmax + 1] - rv[r]) <= cal
        if Eh is not None:
            m &= np.abs(Eh[:, dmin:dmax + 1] - hv[r]) <= cal
        ci, cd = np.nonzero(m)
        cd = cd + dmin
        cj = ci + cd
        ok = cj < n
        if valid is not None:
            ok &= valid[ci] & valid[np.minimum(cj, n - 1)]
        ci, cd, cj = ci[ok], cd[ok], cj[ok]
        if len(ci) == 0:
            continue
        keep = orient_codes(strand, score, ci, cj) == oc0[r]
        ci, cj = ci[keep], cj[keep]
        cost = np.abs(Eref[ci, cj - ci] - rv[r]) / cal
        if Eh is not None:
            cost = cost + np.abs(Eh[ci, cj - ci] - hv[r]) / cal
        pool = []                                     # the TOPK closest free candidates, one drawn at random
        for k in np.argsort(cost + 1e-6 * rng.random(len(ci))):
            q = (int(ci[k]), int(cj[k]))
            if q in used or (tree is not None and tree.query([q[0], q[1]], p=np.inf)[0] <= NEAR):
                continue
            pool.append(q)
            if len(pool) >= TOPK:
                break
        if pool:
            q = pool[int(rng.integers(len(pool)))]
            used.add(q)
            out.append((int(r), q[0], q[1]))
    return out, rv, hv


def match_stripes(pos: pd.DataFrame, Dref, Dh, strand, score, valid, avoid_anchors, rng, shift_scores,
                  cal: float = CAL_ENR):
    """Controls for the stripes `pos` (columns a_lo, ra, eb, o; one chromosome): the same rectangle geometry
    shifted along the chromosome. Dref, Dh: banded log O/E maps (aml_loop_classifier.banded); shift_scores the
    classifier's rectangle-mean function. Returns (list of (row, a')), reference and HSPC value per positive."""
    n = Dref.shape[0]
    near = np.zeros(n, bool)
    for a in avoid_anchors:
        near[max(0, a - NEAR_ANCHOR):a + NEAR_ANCHOR + 1] = True
    used: set = set()
    out = []
    rv = np.full(len(pos), np.nan)
    hv = np.full(len(pos), np.nan)
    for r in rng.permutation(len(pos)):
        row = pos.iloc[r]
        a0, ra, eb, o = int(row.a_lo), int(row.ra), int(row.eb), int(row.o)
        sr = shift_scores(Dref, ra, eb, o)
        sh = shift_scores(Dh, ra, eb, o) if Dh is not None else None
        rv[r] = sr[a0]
        hv[r] = sh[a0] if sh is not None else np.nan
        if not np.isfinite(rv[r]) or (sh is not None and not np.isfinite(hv[r])):
            continue
        base = np.arange(n)
        ok = np.isfinite(sr) & (np.abs(sr - rv[r]) <= cal) & ~near
        if sh is not None:
            ok &= np.isfinite(sh) & (np.abs(sh - hv[r]) <= cal)
        lo, hi = np.minimum(base, base + o), np.maximum(base + ra, base + o + eb)
        ok &= (lo >= 0) & (hi < n)
        if valid is not None:
            ok &= valid[np.clip(base, 0, n - 1)] & valid[np.clip(base + o, 0, n - 1)]
        cand = np.nonzero(ok)[0]
        if len(cand) == 0:
            continue
        dirn = 1 if o > 0 else -1
        c0 = anchor_codes(strand, score, [a0], ra, dirn)[0]
        cand = cand[np.argsort(np.abs(sr[cand] - rv[r]) + (np.abs(sh[cand] - hv[r]) if sh is not None else 0.0))]
        pool = []
        for q in cand:                                # closest candidates first; the orientation test is costly
            q = int(q)
            if q in used or q == a0 or anchor_codes(strand, score, [q], ra, dirn)[0] != c0:
                continue
            pool.append(q)
            if len(pool) >= TOPK:
                break
        if pool:
            q = pool[int(rng.integers(len(pool)))]
            used.add(q)
            out.append((int(r), q))
    return out, rv, hv


# --------------------------------------------------------------------------- maps
def band_enrichment(clr, chrom: str, dmax: int = MAX_BINS) -> np.ndarray:
    """E[i, d]: log2 donut enrichment (inner 1, outer 4, 3 x 3 maximum, as the evaluation) of pixel
    (i, i + d) of a log O/E cool, NaN where unmeasured; windows as aml_loop_classifier.hic_one."""
    import aml_loop_classifier as lc
    from scipy.ndimage import maximum_filter
    from flexcible.bioscore import BioParams, local_enrichment
    prm = BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4)
    n = int(clr.chromsizes[chrom]) // lc.RES + 1
    B = np.full((n, dmax + 1), np.nan, np.float32)
    dd = np.arange(dmax + 1)
    for s, e in lc._windows(n):
        M = lc._fetch(clr, chrom, s, e)
        with np.errstate(invalid="ignore", divide="ignore"):
            L = np.log2(local_enrichment(np.exp(M), prm))
        m3 = maximum_filter(np.nan_to_num(L, nan=-9.0), size=3)
        m3 = np.where(m3 < -8, np.nan, m3)
        r0 = lc.MARGIN if s > 0 else 0
        own = np.arange(r0, lc.MARGIN + lc.STRIDE)
        own = own[s + own < n]
        col = own[:, None] + dd[None, :]
        vals = np.where(col < lc.WIN, m3[own[:, None], np.minimum(col, lc.WIN - 1)], np.nan)
        B[s + own] = vals
    return B


# --------------------------------------------------------------------------- driver
def refrand_controls(pos: pd.DataFrame, ref_cool: str, hspc_cool: str | None, valid_npz: str | None,
                     avoid_loops: pd.DataFrame, avoid_anchors: dict, seed: int, motif=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Controls for one cell / patient. `pos`: its AML-specific (or cell-specific) objects with columns kind,
    pair_id, chrom, a_lo, a_hi, b_lo, b_hi (10 kb bins). Returns (control rows with the pair_id of their
    positive, reference / HSPC values of every positive)."""
    import aml_loop_classifier as lc
    import cooler
    rng = np.random.default_rng(seed)
    if motif is None:
        motif = lc.load_motif_bins(lc.chrom_sizes())
    cref = cooler.Cooler(ref_cool)
    chsp = cooler.Cooler(hspc_cool) if hspc_cool else None
    valid = dict(np.load(valid_npz)) if valid_npz else {}
    rows, vals = [], []
    for chrom, g in pos.groupby("chrom"):
        if chrom not in motif:
            continue
        st, sc = motif[chrom]["strand"], motif[chrom]["score"]
        v = valid.get(chrom)
        loops = g[g.kind == "loop"]
        if len(loops):
            Er = band_enrichment(cref, chrom)
            Eh = band_enrichment(chsp, chrom) if chsp is not None else None
            P = pd.DataFrame({"i": loops.a_lo.to_numpy(), "j": loops.b_lo.to_numpy()})
            al = avoid_loops[avoid_loops.chrom == chrom][["i", "j"]].to_numpy()
            res, rv, hv = match_loops(P, Er, Eh, st, sc, v, al, rng)
            pid = loops.pair_id.to_numpy()
            for r, i, j in res:
                rows.append({"kind": "loop", "pair_id": pid[r], "chrom": chrom, "a_lo": i, "a_hi": i + 1,
                             "b_lo": j, "b_hi": j + 1, "ref_val": float(Er[i, j - i]),
                             "hspc_val": float(Eh[i, j - i]) if Eh is not None else np.nan})
            vals += [{"pair_id": pid[r], "kind": "loop", "ref_val": rv[r], "hspc_val": hv[r]} for r in range(len(P))]
        stripes = g[g.kind == "stripe"]
        if len(stripes):
            Dr = lc.banded(cref, chrom)
            Dh = lc.banded(chsp, chrom) if chsp is not None else None
            P = pd.DataFrame({"a_lo": stripes.a_lo.to_numpy(), "ra": (stripes.a_hi - stripes.a_lo).to_numpy(),
                              "eb": (stripes.b_hi - stripes.b_lo).to_numpy(),
                              "o": (stripes.b_lo - stripes.a_lo).to_numpy()})
            aa = avoid_anchors.get(chrom, [])
            res, rv, hv = match_stripes(P, Dr, Dh, st, sc, v, aa, rng, lc.shift_scores)
            pid = stripes.pair_id.to_numpy()
            for r, a in res:
                q = P.iloc[r]
                sr = lc.shift_scores(Dr, int(q.ra), int(q.eb), int(q.o))
                sh = lc.shift_scores(Dh, int(q.ra), int(q.eb), int(q.o)) if Dh is not None else None
                rows.append({"kind": "stripe", "pair_id": pid[r], "chrom": chrom, "a_lo": a, "a_hi": a + int(q.ra),
                             "b_lo": a + int(q.o), "b_hi": a + int(q.o) + int(q.eb), "ref_val": float(sr[a]),
                             "hspc_val": float(sh[a]) if sh is not None else np.nan})
            vals += [{"pair_id": pid[r], "kind": "stripe", "ref_val": rv[r], "hspc_val": hv[r]} for r in range(len(P))]
    return pd.DataFrame(rows), pd.DataFrame(vals)
