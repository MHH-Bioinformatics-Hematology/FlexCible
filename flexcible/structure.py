"""Structure measures of one predicted window against its target (registry
section 30): TADs (insulation, boundaries), loops (local enrichment, APA) and
compartments (within-window correlation-matrix similarity, first
eigenvector), next to the pixel Pearson and the distance-stratified
correlation.

Maps are log observed/expected (n, n), NaN where unmeasured; every measure is
affine-invariant in the map values except the APA ratio, which is a ratio of
enrichments of the same map. Windows are 1.9 to 2 Mb, so the compartment
measures see only the checkerboard inside 2 Mb; they are a within-window
proxy for A/B compartments, which span tens of Mb.
"""
from __future__ import annotations

import numpy as np

INSUL_BP = 250_000          # insulation square side
LOOP_MIN_BP = 50_000        # loop pixels at least this far from the diagonal
LOOP_TOP = 0.01             # loop pixels: top 1 % of the target's enrichment
DONUT = (2, 5)              # inner, outer half-widths in bins


def _corr(x, y) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    x, y = x[m], y[m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def stratified_corr(P, T, min_d: int = 1, max_frac: float = 0.5) -> float:
    """Mean over diagonals d of the Pearson along that diagonal (each
    distance weighted equally; the O/E analogue of SCC)."""
    n = P.shape[0]
    rs = []
    for d in range(min_d, max(min_d + 1, int(n * max_frac))):
        r = _corr(np.diagonal(P, d), np.diagonal(T, d))
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def insulation(M, w: int) -> np.ndarray:
    """Mean log O/E in the square [i-w, i) x (i, i+w] per bin; NaN at edges."""
    n = M.shape[0]
    out = np.full(n, np.nan)
    if n <= 2 * w:
        return out
    fin = np.isfinite(M)
    S = np.zeros((n + 1, n + 1))
    C = np.zeros((n + 1, n + 1))
    S[1:, 1:] = np.where(fin, M, 0.0).cumsum(0).cumsum(1)
    C[1:, 1:] = fin.cumsum(0).cumsum(1)
    b = np.arange(w, n - w)
    r0, r1, c0, c1 = b - w, b, b + 1, b + w + 1

    def box(A):
        return A[r1, c1] - A[r0, c1] - A[r1, c0] + A[r0, c0]

    cnt = box(C)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[b] = np.where(cnt > 0, box(S) / cnt, np.nan)
    return out


def boundaries(prof, prominence: float) -> np.ndarray:
    out = []
    for b in range(1, prof.size - 1):
        v = prof[b]
        if not np.isfinite(v) or not (v <= prof[b - 1] and v <= prof[b + 1]):
            continue
        lo, hi = prof[max(0, b - 10):b], prof[b + 1:b + 11]
        lo, hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
        if lo.size and hi.size and min(lo.max(), hi.max()) - v >= prominence:
            out.append(b)
    return np.asarray(out, dtype=np.int64)


def boundary_f1(tb, pb, tol: int = 2) -> float:
    if tb.size == 0 and pb.size == 0:
        return float("nan")
    if tb.size == 0 or pb.size == 0:
        return 0.0
    rec = np.mean([np.abs(pb - x).min() <= tol for x in tb])
    prec = np.mean([np.abs(tb - x).min() <= tol for x in pb])
    return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0


def enrichment(M) -> np.ndarray:
    """Pixel minus the mean of its donut ring (what loop callers threshold)."""
    inner, outer = DONUT
    n = M.shape[0]
    X = np.where(np.isfinite(M), M, 0.0)
    W = np.isfinite(M).astype(float)
    k = 2 * outer + 1
    ring = np.ones((k, k))
    ring[outer - inner:outer + inner + 1, outer - inner:outer + inner + 1] = 0
    from scipy.signal import fftconvolve
    s = fftconvolve(X, ring, mode="same")
    c = fftconvolve(W, ring, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        bg = s / np.where(c > 0.5, c, np.nan)
    e = M - bg
    e[:outer, :] = np.nan
    e[n - outer:, :] = np.nan
    e[:, :outer] = np.nan
    e[:, n - outer:] = np.nan
    return e


def corr_matrix(M) -> np.ndarray:
    """Pearson correlation between the rows of a window (NaN -> row mean)."""
    X = np.where(np.isfinite(M), M, np.nan)
    mu = np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isfinite(X), X, mu)
    X = X - X.mean(1, keepdims=True)
    s = np.sqrt((X * X).mean(1, keepdims=True))
    X = X / np.where(s > 1e-12, s, np.nan)
    C = (X @ X.T) / X.shape[1]
    return C


def structure_metrics(P, T, res: int) -> dict:
    """All measures of prediction P against target T (both (n, n) log O/E)."""
    n = T.shape[0]
    P = np.where(np.isfinite(P), P, np.nan)
    T = np.where(np.isfinite(T), T, np.nan)
    out = {}
    iu = np.triu_indices(n, 1)
    out["pearson"] = _corr(P[iu], T[iu])
    out["scc"] = stratified_corr(P, T)
    w = max(3, int(round(INSUL_BP / res)))
    ip, it = insulation(P, w), insulation(T, w)
    out["insulation_corr"] = _corr(ip, it)
    sd = np.nanstd(it)
    out["boundary_f1"] = (boundary_f1(boundaries(it, 0.5 * sd), boundaries(ip, 0.5 * np.nanstd(ip)))
                          if np.isfinite(sd) and sd > 0 else float("nan"))
    ep, et = enrichment(P), enrichment(T)
    dmin = max(DONUT[1] + 1, int(round(LOOP_MIN_BP / res)))
    ii, jj = np.triu_indices(n, dmin)
    jj_ok = jj - ii <= n // 2
    ii, jj = ii[jj_ok], jj[jj_ok]
    a, b = ep[ii, jj], et[ii, jj]
    out["loop_enrich_corr"] = _corr(a, b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() > 50:
        thr = np.quantile(b[ok], 1 - LOOP_TOP)
        sel = ok & (b >= thr)
        te, pe = float(np.mean(b[sel])), float(np.mean(a[sel]))
        out["loop_apa_ratio"] = pe / te if te > 0 else float("nan")
    else:
        out["loop_apa_ratio"] = float("nan")
    Cp, Ct = corr_matrix(P), corr_matrix(T)
    iu2 = np.triu_indices(n, 2)
    out["compartment_corr"] = _corr(Cp[iu2], Ct[iu2])
    try:
        vt = np.linalg.eigh(np.nan_to_num(Ct))[1][:, -1]
        vp = np.linalg.eigh(np.nan_to_num(Cp))[1][:, -1]
        out["eigenvector_abs_corr"] = abs(_corr(vp, vt))
    except np.linalg.LinAlgError:
        out["eigenvector_abs_corr"] = float("nan")
    return out


SELECT_KEYS = ("pearson", "scc", "insulation_corr", "loop_apa")


def selection_values(m: dict) -> dict:
    """The four measures of the combined checkpoint selection, each higher =
    better: Pearson, stratified correlation, insulation correlation and the
    APA agreement min(r, 1/r) of the APA ratio r."""
    r = m.get("loop_apa_ratio", float("nan"))
    apa = min(r, 1.0 / r) if np.isfinite(r) and r > 0 else 0.0
    return {"pearson": m["pearson"], "scc": m["scc"],
            "insulation_corr": m["insulation_corr"], "loop_apa": apa}


def rank_combined(history: list[dict]) -> list[float]:
    """Combined score of every evaluation in `history` (dicts of the
    SELECT_KEYS medians): per measure the rank among all evaluations so far,
    scaled to [0, 1] (ties averaged), then the mean over the measures."""
    if not history:
        return []
    from scipy.stats import rankdata
    n = len(history)
    tot = np.zeros(n)
    for k in SELECT_KEYS:
        v = np.array([h.get(k, np.nan) for h in history], float)
        v = np.where(np.isfinite(v), v, -np.inf)
        r = rankdata(v, method="average")
        tot += (r - 1) / max(1, n - 1)
    return list(tot / len(SELECT_KEYS))
