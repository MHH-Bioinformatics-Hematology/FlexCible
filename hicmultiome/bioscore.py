"""A feature-level score for predicted Hi-C: TADs and loops, not pixels.

Why not Pearson over the map
----------------------------
The number everybody reports is a correlation over all pixels of a region.
It has three problems that this module is written to avoid:

* it is dominated by whatever correlates, and most of a Hi-C map is smooth
  distance decay plus large low-contact areas that carry no biology;
* it says nothing about the two features people actually use a contact map
  for -- domain boundaries and point loops;
* it is a single blended number, so a model that gets the domains right and
  the loops wrong is indistinguishable from the reverse.

So this module scores the *features*, keeps the components separate, and only
then offers a combined number as a convenience.

Input space
-----------
Everything here takes maps in Akita space: ``log(observed / expected)``, i.e.
already distance-normalised, symmetric, NaN where unmeasured. Internally the
maps are exponentiated back to an observed/expected ratio, because that is the
quantity a donut enrichment is defined on and the quantity ``loops.insulation_
profile`` expects (a non-negative, counts-like signal whose local mean means
something). Distance normalisation is what makes the loop caller usable: the
donut background and the peak threshold are then comparable at every genomic
separation, which is the "stratify or distance decay eats your metric" rule
applied at the level of the metric definition rather than afterwards.

Components
----------
TAD structure
    ``insulation_pearson``  correlation of the two insulation profiles.
                            Answers "is the domain structure in the right
                            places", and is blind to the overall contact
                            level, unlike a pixel correlation.
    ``boundary_f1``         F1 of boundary calls with a tolerance. Answers
                            the discrete question -- would you have drawn the
                            same TADs -- which a profile correlation can pass
                            while still misplacing every boundary by 20 kb.
    Both come from :mod:`hicmultiome.loops`, unchanged.

Loops
    Loops are called in the MEASURED map (local maximum of a donut/local
    background enrichment, on bins coarsened to ~16 kb), never in the
    prediction first, so the reference set never depends on the model being
    scored.
    ``loop_f1``             did the prediction put its loops in the same
                            places.  The prediction is called by a rank cut at
                            the number of measured loops, not by an absolute
                            threshold, because the models differ enormously in
                            how sharp their output is and a fixed threshold
                            would grade sharpness rather than placement.
    ``loopthr_*``           the same matching with the absolute threshold
                            instead: how many loops the model actually
                            asserts, how many of those the measurement has,
                            and how many it invented.  This is the
                            false-positive side and the sharpness diagnostic.
    ``apa_ratio``           aggregate peak analysis around the MEASURED
                            anchors, evaluated on the prediction, expressed
                            as the fraction of the measured peak enrichment
                            that the prediction reproduces.  A model can miss
                            every discrete call and still show real, weaker
                            enrichment at the right anchors; APA sees that,
                            an F1 cannot.
    ``anchor_spearman``     measured vs predicted strength at measured
                            anchors -- are the strong loops the strong ones.
    ``anchor_auroc``        predicted enrichment at measured anchors versus
                            distance-matched non-anchor pixels.  This is the
                            control that stops "predict high everywhere" from
                            scoring well.

Combined
    ``bioscore = 0.5 * tad_score + 0.5 * loop_score`` with
    ``tad_score  = 0.5*clip(insulation_pearson,0,1) + 0.5*boundary_f1`` and
    ``loop_score = 0.5*loop_f1 + 0.5*clip(apa_ratio,0,1)``.
    Equal weights, stated rather than tuned. It is reported *next to* its
    parts, never instead of them.

Pooling
-------
Per-region F1 and APA are noisy (a 448-bin region holds a handful of loops),
so every count this module produces is additive: boundary and loop TP/FP/FN,
and the APA stack as a sum plus an anchor count. Any subset of regions -- a
chromatin stratum, a depth bin -- is then scored by summing its regions and
computing the ratios once, which is what :func:`pool` does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loops import boundaries, boundary_f1, insulation_profile

__all__ = [
    "BioParams", "oe_from_log", "donut_kernel", "local_enrichment",
    "call_loops", "match_loops", "apa_stack", "apa_pair", "region_metrics", "pool",
    "resample_square", "distance_baseline", "dense_from_triu",
]


# --------------------------------------------------------------------------- #
@dataclass
class BioParams:
    """Every threshold in one place, so a run can be reproduced from it."""

    # --- insulation / TADs ---
    insulation_window: int = 25      # diamond half-width, bins (51 kb at 2048)
    boundary_delta: float = 0.10     # minimum prominence of a boundary
    boundary_tol: int = 5            # bins; 10 kb at 2048 bp

    # --- loops ---
    # Loops are called on a coarsened map. The parameters below are not
    # guesses: scripts/bioscore_tune_loops.py sweeps them and scores each
    # setting by how strongly the resulting MEASURED loop anchors are
    # occupied by CTCF -- an external, model-independent criterion. Called
    # natively on 2 kb bins the anchors carry no CTCF signal at all (AUROC
    # 0.50, i.e. the caller finds Poisson noise); at 8x coarsening (16.4 kb)
    # with these thresholds they reach AUROC 0.71 at ~3 loops per 917 kb
    # region, which is the loop density Rao et al. report for GM12878.
    loop_coarsen: int = 8            # average f x f blocks before loop calling
    donut_inner: int = 1             # excluded plus-shaped core, coarse bins
    donut_outer: int = 4             # donut half-width, coarse bins
    loop_min_dist: int = 4           # coarse bins; ~65 kb
    loop_max_dist: int = 37          # coarse bins; ~600 kb
    loop_min_oe: float = 1.15        # peak must be above expected at all
    loop_min_enrich: float = 1.80    # ... and above its own local background
    loop_nms: int = 2                # Chebyshev radius for peak suppression
    loop_max_peaks: int = 25         # cap per region, keeps counts comparable
    loop_match_tol: int = 1          # coarse bins; ~16 kb, HiCCUPS-like
    apa_half: int = 5                # APA window is (2*half+1) square
    apa_corner: int = 2              # corner block used as the APA denominator
    n_controls: int = 8              # distance-matched controls per anchor
    seed: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def dense_from_triu(vals: np.ndarray, side: int, offset: int) -> np.ndarray:
    """Rebuild a symmetric dense map from a stored upper-triangle vector."""
    r, c = np.triu_indices(side, offset)
    M = np.full((side, side), np.nan, np.float32)
    M[r, c] = vals
    return np.where(np.isnan(M), M.T, M)


def oe_from_log(log_map: np.ndarray) -> np.ndarray:
    """Akita space -> observed/expected ratio, clipped to a sane range."""
    return np.exp(np.clip(np.asarray(log_map, dtype=np.float64), -8.0, 8.0))


def resample_square(P: np.ndarray, src_start_bp: float, src_res: float,
                    dst_start_bp: float, n_dst: int,
                    dst_res: float) -> np.ndarray:
    """Bilinear resample of a square map onto another bin grid.

    Legitimate for a predicted or normalised *field*; never do this to raw
    counts.
    """
    from scipy.ndimage import map_coordinates

    centres = dst_start_bp + (np.arange(n_dst) + 0.5) * dst_res
    coord = (centres - src_start_bp) / src_res - 0.5
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    return map_coordinates(np.nan_to_num(P), [yy.ravel(), xx.ravel()],
                           order=1, mode="nearest").reshape(n_dst, n_dst)


def distance_baseline(targets: np.ndarray, side: int) -> np.ndarray:
    """The trivial prediction: the mean of the data at each genomic distance.

    ``targets`` is (N, side, side) of measured log(o/e) maps. Averaging per
    diagonal over the whole test set (not per region -- that would peek at the
    region being scored) gives a map that knows distance decay and nothing
    else. Any metric that cannot separate a model from this is broken.
    """
    prof = np.full(side, 0.0)
    for d in range(side):
        v = np.diagonal(targets, offset=d, axis1=1, axis2=2)
        v = v[np.isfinite(v)]
        prof[d] = float(v.mean()) if v.size else 0.0
    idx = np.abs(np.subtract.outer(np.arange(side), np.arange(side)))
    return prof[idx]


# --------------------------------------------------------------------------- #
# loop calling: donut / local-background enrichment
# --------------------------------------------------------------------------- #
def donut_kernel(inner: int, outer: int) -> np.ndarray:
    """Rao-style donut: a square annulus with the central cross removed.

    The cross has to go, otherwise a loop's own stripe (the CTCF anchor's
    contacts with everything else) leaks into its background and a real loop
    suppresses its own enrichment.
    """
    d = np.arange(-outer, outer + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    cheb = np.maximum(np.abs(dy), np.abs(dx))
    k = (cheb <= outer) & (cheb > inner)
    k &= ~((np.abs(dy) <= inner) | (np.abs(dx) <= inner))
    return k.astype(np.float64)


def local_enrichment(oe: np.ndarray, p: BioParams) -> np.ndarray:
    """observed/expected divided by its donut background. NaN where undefined.

    ``oe`` is already distance-normalised, so the donut -- which straddles
    several diagonals -- is not biased by decay.
    """
    from scipy.ndimage import convolve

    k = donut_kernel(p.donut_inner, p.donut_outer)
    valid = np.isfinite(oe)
    filled = np.where(valid, oe, 0.0)
    num = convolve(filled, k, mode="reflect")
    cnt = convolve(valid.astype(np.float64), k, mode="reflect")
    bg = np.divide(num, cnt, out=np.full_like(num, np.nan), where=cnt >= 8)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = oe / bg
    e[~valid] = np.nan
    return e


def coarsen(M: np.ndarray, f: int) -> np.ndarray:
    """Average f x f blocks. Loops are called on a coarsened map.

    A 2 kb bin is far finer than the scale of a CTCF loop, and at that bin
    size a single Poisson-noisy pixel outranks a real, broad peak: a donut
    caller run natively on 2 kb picks up noise (measured: its calls are not
    enriched for CTCF at all, AUROC 0.50). Averaging to ~8-16 kb first is what
    HiCCUPS-style callers do, and is chosen here on an external criterion --
    CTCF occupancy at the called anchors -- not on any model's score.
    """
    if f <= 1:
        return M
    n = M.shape[0] // f * f
    return M[:n, :n].reshape(n // f, f, n // f, f).mean(axis=(1, 3))


def _distance_mask(n: int, lo: int, hi: int) -> np.ndarray:
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return (d >= lo) & (d <= hi)


def call_loops(oe: np.ndarray, p: BioParams,
               enrich: np.ndarray | None = None,
               top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Peaks of the donut enrichment. Returns (anchors (n,2) i<j, enrichment).

    A peak must be (a) inside the loop distance range, (b) the maximum of the
    enrichment in a small neighbourhood, and -- unless ``top_k`` is given --
    (c) above expected and above its own local background. Overlapping calls
    are suppressed greedily so one broad blob counts once.

    ``top_k`` replaces the absolute thresholds by a rank cut: return the k
    strongest peaks whatever their absolute enrichment. That is how a
    prediction is called when it is being compared to a measured loop set,
    because the models differ enormously in how sharp their output is (a
    smooth model can place every loop correctly and still never cross a fixed
    threshold, a noisy one crosses it everywhere) and a fixed threshold would
    measure sharpness rather than placement. The fixed-threshold call is still
    computed separately and reported, as the sharpness diagnostic it is.
    """
    from scipy.ndimage import maximum_filter

    n = oe.shape[0]
    e = local_enrichment(oe, p) if enrich is None else enrich
    ok = (_distance_mask(n, p.loop_min_dist, p.loop_max_dist)
          & np.isfinite(e) & np.isfinite(oe))
    if top_k is None:
        ok &= (oe >= p.loop_min_oe) & (e >= p.loop_min_enrich)
    ok &= np.triu(np.ones((n, n), bool), 1)
    if not ok.any():
        return np.zeros((0, 2), np.int64), np.zeros(0)
    size = 2 * p.loop_nms + 1
    filled = np.where(np.isfinite(e), e, -np.inf)
    ok &= filled >= maximum_filter(filled, size=size, mode="nearest")

    limit = p.loop_max_peaks if top_k is None else int(top_k)
    if limit <= 0:
        return np.zeros((0, 2), np.int64), np.zeros(0)

    ii, jj = np.nonzero(ok)
    vals = e[ii, jj]
    order = np.argsort(-vals)
    ii, jj, vals = ii[order], jj[order], vals[order]

    keep_i, keep_j, keep_v = [], [], []
    for a, b, v in zip(ii, jj, vals):
        if any(max(abs(a - x), abs(b - y)) <= p.loop_nms
               for x, y in zip(keep_i, keep_j)):
            continue
        keep_i.append(int(a)); keep_j.append(int(b)); keep_v.append(float(v))
        if len(keep_i) >= limit:
            break
    return (np.array(list(zip(keep_i, keep_j)), np.int64).reshape(-1, 2),
            np.asarray(keep_v))


def match_loops(true_a: np.ndarray, pred_a: np.ndarray, tol: int) -> dict:
    """TP/FP/FN counts for two anchor sets, Chebyshev distance tolerance."""
    if true_a.size == 0 and pred_a.size == 0:
        return {"tp_true": 0, "tp_pred": 0, "n_true": 0, "n_pred": 0}
    if true_a.size == 0:
        return {"tp_true": 0, "tp_pred": 0, "n_true": 0,
                "n_pred": int(pred_a.shape[0])}
    if pred_a.size == 0:
        return {"tp_true": 0, "tp_pred": 0, "n_true": int(true_a.shape[0]),
                "n_pred": 0}
    d = np.maximum(np.abs(true_a[:, None, 0] - pred_a[None, :, 0]),
                   np.abs(true_a[:, None, 1] - pred_a[None, :, 1]))
    hit = d <= tol
    return {"tp_true": int(hit.any(axis=1).sum()),
            "tp_pred": int(hit.any(axis=0).sum()),
            "n_true": int(true_a.shape[0]), "n_pred": int(pred_a.shape[0])}


def apa_stack(oe: np.ndarray, anchors: np.ndarray, half: int
              ) -> tuple[np.ndarray, int]:
    """Sum of (2*half+1)^2 windows of ``oe`` centred on ``anchors``.

    Returned as a sum plus a count so stacks from different regions add.
    """
    w = 2 * half + 1
    acc = np.zeros((w, w))
    n = 0
    if anchors.size == 0:
        return acc, 0
    m = oe.shape[0]
    for i, j in anchors:
        if i - half < 0 or j - half < 0 or i + half >= m or j + half >= m:
            continue
        sub = oe[i - half:i + half + 1, j - half:j + half + 1]
        if not np.isfinite(sub).all():
            continue
        acc += sub
        n += 1
    return acc, n


def apa_pair(true_oe: np.ndarray, pred_oe: np.ndarray, anchors: np.ndarray,
             half: int) -> tuple[np.ndarray, np.ndarray, int]:
    """APA stacks for measurement and prediction over the SAME anchor set.

    An anchor is used only if both windows are complete and finite, otherwise
    the two stacks would average different loops and their ratio would be
    meaningless.
    """
    w = 2 * half + 1
    st, sp, n = np.zeros((w, w)), np.zeros((w, w)), 0
    if anchors.size == 0:
        return st, sp, 0
    m = true_oe.shape[0]
    for i, j in anchors:
        if i - half < 0 or j - half < 0 or i + half >= m or j + half >= m:
            continue
        a = true_oe[i - half:i + half + 1, j - half:j + half + 1]
        b = pred_oe[i - half:i + half + 1, j - half:j + half + 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        st += a
        sp += b
        n += 1
    return st, sp, n


def apa_score(stack: np.ndarray, count: int, corner: int) -> float:
    """Centre pixel over the lower-left corner block (Rao's P2LL)."""
    if count == 0:
        return float("nan")
    A = stack / count
    c = A.shape[0] // 2
    denom = A[-corner:, :corner].mean()
    return float(A[c, c] / denom) if denom > 0 else float("nan")


def _peak_strength(e: np.ndarray, i: np.ndarray, j: np.ndarray,
                   tol: int) -> np.ndarray:
    """Max enrichment within +/- tol bins of each (i, j)."""
    n = e.shape[0]
    out = np.full(i.size, np.nan)
    for k in range(i.size):
        a0, a1 = max(0, i[k] - tol), min(n, i[k] + tol + 1)
        b0, b1 = max(0, j[k] - tol), min(n, j[k] + tol + 1)
        sub = e[a0:a1, b0:b1]
        sub = sub[np.isfinite(sub)]
        if sub.size:
            out[k] = sub.max()
    return out


def _controls(anchors: np.ndarray, n: int, p: BioParams,
              rng: np.random.Generator) -> np.ndarray:
    """Distance-matched non-anchor pixels: same separation, shifted anchor."""
    if anchors.size == 0:
        return np.zeros((0, 2), np.int64)
    taken = {(int(a), int(b)) for a, b in anchors}
    out = []
    for a, b in anchors:
        d = int(b - a)
        for _ in range(p.n_controls):
            for _try in range(20):
                s = int(rng.integers(0, max(1, n - d)))
                if all(max(abs(s - x), abs(s + d - y)) > p.loop_match_tol
                       for x, y in taken):
                    out.append((s, s + d))
                    break
    return np.asarray(out, np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# per-region metrics
# --------------------------------------------------------------------------- #
@dataclass
class RegionResult:
    """Everything additive, so subsets can be pooled without re-scoring."""

    pearson: float = float("nan")
    insulation_pearson: float = float("nan")
    b_tp_true: int = 0
    b_tp_pred: int = 0
    b_n_true: int = 0
    b_n_pred: int = 0
    l_tp_true: int = 0
    l_tp_pred: int = 0
    l_n_true: int = 0
    l_n_pred: int = 0
    lt_tp_true: int = 0
    lt_tp_pred: int = 0
    lt_n_pred: int = 0
    apa_true_sum: np.ndarray = field(default_factory=lambda: np.zeros((21, 21)))
    apa_pred_sum: np.ndarray = field(default_factory=lambda: np.zeros((21, 21)))
    apa_n: int = 0
    anchor_true: np.ndarray = field(default_factory=lambda: np.zeros(0))
    anchor_pred: np.ndarray = field(default_factory=lambda: np.zeros(0))
    control_pred: np.ndarray = field(default_factory=lambda: np.zeros(0))


def region_metrics(true_log: np.ndarray, pred_log: np.ndarray,
                   p: BioParams | None = None,
                   true_loops: np.ndarray | None = None,
                   seed_offset: int = 0) -> RegionResult:
    """Score one region. ``true_loops`` lets a caller reuse measured calls."""
    p = p or BioParams()
    n = min(true_log.shape[0], pred_log.shape[0])
    T, P = np.asarray(true_log[:n, :n], float), np.asarray(pred_log[:n, :n], float)
    res = RegionResult(apa_true_sum=np.zeros((2 * p.apa_half + 1,) * 2),
                       apa_pred_sum=np.zeros((2 * p.apa_half + 1,) * 2))

    ok = np.isfinite(T) & np.isfinite(P)
    if ok.sum() > 100 and T[ok].std() > 0 and P[ok].std() > 0:
        res.pearson = float(np.corrcoef(T[ok], P[ok])[0, 1])

    to, po = oe_from_log(T), oe_from_log(P)
    to[~np.isfinite(T)] = np.nan
    po[~np.isfinite(P)] = np.nan

    # --- TADs (functions reused from hicmultiome.loops, unmodified) ---
    pt = insulation_profile(np.nan_to_num(to), p.insulation_window)
    pp = insulation_profile(np.nan_to_num(po), p.insulation_window)
    good = np.isfinite(pt) & np.isfinite(pp)
    if good.sum() > 10 and pt[good].std() > 0:
        if pp[good].std() > 0:
            r = float(np.corrcoef(pt[good], pp[good])[0, 1])
            res.insulation_pearson = r if np.isfinite(r) else 0.0
        else:
            # a candidate with a flat insulation profile predicts no domain
            # structure at all. That earns zero, not "undefined" -- the
            # distance-only baseline is exactly this case and must not be
            # allowed to drop out of the table it exists to anchor.
            res.insulation_pearson = 0.0
    bt = boundaries(pt, p.boundary_delta)
    bp = boundaries(pp, p.boundary_delta)
    bf = boundary_f1(bt, bp, p.boundary_tol)
    res.b_n_true, res.b_n_pred = bf["n_true"], bf["n_pred"]
    res.b_tp_true = int(round(bf["recall"] * max(1, bf["n_true"])))
    res.b_tp_pred = int(round(bf["precision"] * max(1, bf["n_pred"])))

    # --- loops: reference set from the MEASUREMENT ---
    # NaN survives the block mean, so the unmeasured near-diagonal band stays
    # unmeasured after coarsening; it sits below loop_min_dist regardless.
    tc = coarsen(to, p.loop_coarsen)
    pc = coarsen(po, p.loop_coarsen)
    et = local_enrichment(tc, p)
    ep = local_enrichment(pc, p)
    ta = call_loops(tc, p, et)[0] if true_loops is None else true_loops
    # rank-matched call: the prediction's |ta| strongest peaks (placement)
    pa = call_loops(pc, p, ep, top_k=len(ta))[0]
    m = match_loops(ta, pa, p.loop_match_tol)
    res.l_tp_true, res.l_tp_pred = m["tp_true"], m["tp_pred"]
    res.l_n_true, res.l_n_pred = m["n_true"], m["n_pred"]
    # fixed-threshold call: how many loops the model actually asserts
    pat = call_loops(pc, p, ep)[0]
    mt = match_loops(ta, pat, p.loop_match_tol)
    res.lt_tp_true, res.lt_tp_pred = mt["tp_true"], mt["tp_pred"]
    res.lt_n_pred = mt["n_pred"]

    res.apa_true_sum, res.apa_pred_sum, res.apa_n = apa_pair(
        tc, pc, ta, p.apa_half)

    if ta.size:
        nc = tc.shape[0]
        i, j = ta[:, 0], ta[:, 1]
        # strength is read with the same tolerance the matching uses: a loop
        # placed one bin off is still that loop, and a single-pixel readout at
        # the exact anchor would be dominated by noise rather than by whether
        # the model put a peak there.
        res.anchor_true = _peak_strength(et, i, j, p.loop_match_tol)
        res.anchor_pred = _peak_strength(ep, i, j, p.loop_match_tol)
        rng = np.random.default_rng(p.seed + seed_offset)
        ca = _controls(ta, nc, p, rng)
        if ca.size:
            res.control_pred = _peak_strength(ep, ca[:, 0], ca[:, 1],
                                              p.loop_match_tol)
    return res


# --------------------------------------------------------------------------- #
def _f1(tp_t: int, tp_p: int, n_t: int, n_p: int) -> dict:
    rec = tp_t / n_t if n_t else float("nan")
    prec = tp_p / n_p if n_p else float("nan")
    if not np.isfinite(rec) or not np.isfinite(prec) or prec + rec == 0:
        f1 = 0.0 if (n_t or n_p) else float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def pool(results: list[RegionResult], p: BioParams | None = None) -> dict:
    """Micro-average a set of regions into the reported score.

    Counts are summed before any ratio is taken, and the APA stacks are summed
    before the peak-to-corner ratio is taken, so a region with two loops does
    not carry the same weight as one with forty.
    """
    p = p or BioParams()
    if not results:
        return {}
    out: dict = {"n_regions": len(results)}
    def _nanmean(vals, fn=np.nanmean):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        return float(fn(v)) if v.size else float("nan")

    pear = [r.pearson for r in results]
    out["pearson_mean"] = _nanmean(pear)
    out["pearson_median"] = _nanmean(pear, np.median)
    out["insulation_pearson"] = _nanmean([r.insulation_pearson
                                          for r in results])

    b = _f1(sum(r.b_tp_true for r in results), sum(r.b_tp_pred for r in results),
            sum(r.b_n_true for r in results), sum(r.b_n_pred for r in results))
    out.update({f"boundary_{k}": v for k, v in b.items()})
    out["boundary_n_true"] = sum(r.b_n_true for r in results)
    out["boundary_n_pred"] = sum(r.b_n_pred for r in results)

    l = _f1(sum(r.l_tp_true for r in results), sum(r.l_tp_pred for r in results),
            sum(r.l_n_true for r in results), sum(r.l_n_pred for r in results))
    out.update({f"loop_{k}": v for k, v in l.items()})
    out["loop_n_true"] = sum(r.l_n_true for r in results)
    out["loop_n_pred"] = sum(r.l_n_pred for r in results)
    out["loop_n_invented"] = out["loop_n_pred"] - sum(r.l_tp_pred for r in results)
    lt = _f1(sum(r.lt_tp_true for r in results), sum(r.lt_tp_pred for r in results),
             sum(r.l_n_true for r in results), sum(r.lt_n_pred for r in results))
    out.update({f"loopthr_{k}": v for k, v in lt.items()})
    out["loopthr_n_pred"] = sum(r.lt_n_pred for r in results)
    out["loopthr_n_invented"] = (out["loopthr_n_pred"]
                                 - sum(r.lt_tp_pred for r in results))

    n_apa = sum(r.apa_n for r in results)
    st = sum((r.apa_true_sum for r in results), np.zeros_like(results[0].apa_true_sum))
    sp = sum((r.apa_pred_sum for r in results), np.zeros_like(results[0].apa_pred_sum))
    out["apa_n_anchors"] = int(n_apa)
    out["apa_true"] = apa_score(st, n_apa, p.apa_corner)
    out["apa_pred"] = apa_score(sp, n_apa, p.apa_corner)
    if np.isfinite(out["apa_true"]) and out["apa_true"] > 1:
        out["apa_ratio"] = float((out["apa_pred"] - 1.0) /
                                 (out["apa_true"] - 1.0))
    else:
        out["apa_ratio"] = float("nan")

    at = np.concatenate([r.anchor_true for r in results]) if results else np.zeros(0)
    ap = np.concatenate([r.anchor_pred for r in results]) if results else np.zeros(0)
    cp = np.concatenate([r.control_pred for r in results]) if results else np.zeros(0)
    good = np.isfinite(at) & np.isfinite(ap)
    if good.sum() > 10 and at[good].std() > 0 and ap[good].std() > 0:
        from scipy.stats import rankdata
        out["anchor_pearson"] = float(np.corrcoef(at[good], ap[good])[0, 1])
        out["anchor_spearman"] = float(np.corrcoef(
            rankdata(at[good]), rankdata(ap[good]))[0, 1])
    else:
        out["anchor_pearson"] = out["anchor_spearman"] = float("nan")
    apg, cpg = ap[np.isfinite(ap)], cp[np.isfinite(cp)]
    out["anchor_auroc"] = (_auroc(apg, cpg) if apg.size > 5 and cpg.size > 5
                           else float("nan"))
    out["n_anchor_controls"] = int(cpg.size)

    tad = 0.5 * float(np.clip(out["insulation_pearson"], 0, 1)) + \
        0.5 * float(np.nan_to_num(out["boundary_f1"]))
    loop = 0.5 * float(np.nan_to_num(out["loop_f1"])) + \
        0.5 * float(np.clip(np.nan_to_num(out["apa_ratio"]), 0, 1))
    out["tad_score"] = tad
    out["loop_score"] = loop
    out["bioscore"] = 0.5 * tad + 0.5 * loop
    return out


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    rp = r[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
