"""Measured and predicted maps of the captured regions: O/E construction, scores."""
import numpy as np, cooler
from common import *

EPS = 1.0          # pseudo-count after scaling the region to TOTAL counts
TOTAL = 1.0e6


def region_counts(path, res, chrom, start, end):
    """Dense symmetric raw-count matrix of the region from an mcool (no balancing)."""
    c = cooler.Cooler(f"{path}::resolutions/{res}")
    m = c.matrix(balance=False, sparse=False).fetch(f"{chrom}:{start}-{end}").astype(np.float64)
    m = np.triu(m) + np.triu(m, 1).T            # make sure it is symmetric
    return m


def region_logoe(counts, eps=EPS, total=TOTAL):
    """Raw counts -> log2 O/E inside the region.
    Counts are scaled so the upper triangle (incl. diagonal) sums to `total`; E(d) is the mean over the
    covered diagonal d (zeros included); bins without any reads are NaN."""
    n = counts.shape[0]
    cov = counts.sum(1) > 0
    m = counts * (total / np.triu(counts).sum())
    E = np.zeros(n)
    for d in range(n):
        v = np.diagonal(m, d)
        ok = cov[: n - d] & cov[d:]
        E[d] = v[ok].mean() if ok.any() else 0.0
    D = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    out = np.log2((m + eps) / (E[D] + eps))
    out[~cov, :] = np.nan
    out[:, ~cov] = np.nan
    return out


def pixel_mask(n, min_off=2, max_off=None):
    i, j = np.triu_indices(n, k=min_off)
    if max_off is not None:
        k = (j - i) <= max_off
        i, j = i[k], j[k]
    return i, j


def pearson(a, b, i, j):
    x, y = a[i, j], b[i, j]
    k = np.isfinite(x) & np.isfinite(y)
    if k.sum() < 10: return np.nan
    return float(np.corrcoef(x[k], y[k])[0, 1])


def spearman(a, b, i, j):
    from scipy.stats import spearmanr
    x, y = a[i, j], b[i, j]
    k = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[k], y[k])[0]) if k.sum() > 10 else np.nan


def win_max(m, i, j, h=1):
    n = m.shape[0]
    sub = m[max(0, i - h): i + h + 1, max(0, j - h): j + h + 1]
    return float(np.nanmax(sub)) if np.isfinite(sub).any() else np.nan


def insulation(m, b, w):
    """Mean log O/E in the square of w bins upstream x w bins downstream of bin b (contacts across b)."""
    n = m.shape[0]
    a0, a1, b0, b1 = max(0, b - w), b, b + 1, min(n, b + 1 + w)
    sq = m[a0:a1, b0:b1]
    return float(np.nanmean(sq)) if np.isfinite(sq).any() else np.nan


def pred_region(cool_path, chrom, start, end, res):
    """Predicted log O/E (model units; off-band and unpredicted pixels NaN) as a dense matrix."""
    c = cooler.Cooler(str(cool_path))
    sp = c.matrix(balance=False, sparse=True).fetch(f"{chrom}:{start}-{end}").tocoo()
    n = (end - start) // res
    m = np.full((n, n), np.nan)
    m[sp.row, sp.col] = sp.data
    m = np.where(np.isfinite(m), m, m.T)
    return m


def pooled_counts_logoe(count_list):
    return region_logoe(sum(c.astype(np.float64) for c in count_list))
