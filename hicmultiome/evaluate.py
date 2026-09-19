"""Score predicted Hi-C matrices against ground truth.

Primary metric is HiCRep's SCC (stratum-adjusted correlation coefficient), the
standard for Hi-C reproducibility. If the ``hicrep`` package is installed we use
it; otherwise we fall back to a self-contained SCC implementation so evaluation
always runs. We also report distance-stratified Pearson and a log-MSE.

CLI:  python -m hicmultiome.evaluate --config configs/gm12878_25kb.yaml \
          --pred predicted_25kb.cool --chroms even
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .config import Config
from .genome import select_by_parity
from .matrix import cool_info, fetch_chrom_matrix
from .utils import get_logger

log = get_logger()


def scc_streaming(true_cool: str, pred_cool: str, chrom: str,
                  max_dist_bins: int, h: int = 1,
                  core_bins: int = 8000) -> float:
    """Stratum-adjusted correlation without a dense chromosome matrix.

    The chromosome is walked in overlapping slices (overlap = the scored band
    plus the smoothing halo), each dense but bounded, and the per-stratum
    Pearson sums are accumulated exactly. This makes 5 kb / 2 kb / 1 kb
    evaluation possible, where a full matrix would be tens to hundreds of GiB.
    """
    import cooler

    from .genome import match_name

    ct, cp = cooler.Cooler(true_cool), cooler.Cooler(pred_cool)
    binsize = ct.binsize
    name_t = match_name(chrom, list(ct.chromnames))
    name_p = match_name(chrom, list(cp.chromnames))
    n_bins = int(ct.extent(name_t)[1] - ct.extent(name_t)[0])
    chromlen = int(ct.chromsizes[name_t])      # last bin may be partial
    halo = max_dist_bins + 2 * h + 1
    acc = {d: np.zeros(5) for d in range(max_dist_bins)}   # n, sx, sy, sxx, syy
    sxy = np.zeros(max_dist_bins)

    for s0 in range(0, n_bins, core_bins):
        e0 = min(s0 + core_bins, n_bins)
        # halo on BOTH sides: smoothing needs left context too, otherwise the
        # first rows of every slice are smoothed against missing data
        s1, e1 = max(0, s0 - halo), min(e0 + halo, n_bins)
        b0, b1 = s1 * binsize, min(e1 * binsize, chromlen)
        reg_t = f"{name_t}:{b0}-{b1}"
        reg_p = f"{name_p}:{b0}-{b1}"
        a = np.nan_to_num(np.asarray(
            ct.matrix(balance=False).fetch(reg_t), dtype=np.float64))
        b = np.nan_to_num(np.asarray(
            cp.matrix(balance=False).fetch(reg_p), dtype=np.float64))
        m = min(a.shape[0], b.shape[0])
        a, b = _smooth2d(a[:m, :m], h), _smooth2d(b[:m, :m], h)
        off = s0 - s1                      # where the core starts in the block
        core = e0 - s0
        for d in range(max_dist_bins):
            if d >= m:
                break
            x = np.diagonal(a, d)[off : off + core]
            y = np.diagonal(b, d)[off : off + core]
            k = min(x.size, y.size)
            if k < 1:
                continue
            x, y = x[:k], y[:k]
            acc[d] += np.array([k, x.sum(), y.sum(), (x * x).sum(), (y * y).sum()])
            sxy[d] += float((x * y).sum())
        del a, b
    num = den = 0.0
    for d in range(max_dist_bins):
        n, sx, sy, sxx, syy = acc[d]
        if n < 2:
            continue
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        if vx <= 1e-12 or vy <= 1e-12:
            continue
        cov = sxy[d] / n - (sx / n) * (sy / n)
        r = cov / np.sqrt(vx * vy)
        if not np.isfinite(r):
            continue
        w = n * np.sqrt(vx) * np.sqrt(vy)
        num += w * r
        den += w
    return float(num / den) if den > 0 else float("nan")


def _smooth2d(m: np.ndarray, h: int) -> np.ndarray:
    """(2h+1)x(2h+1) box filter, as used by HiCRep before scoring."""
    if h <= 0:
        return m
    from scipy.ndimage import uniform_filter

    return uniform_filter(m, size=2 * h + 1, mode="constant")


def scc(
    mat1: np.ndarray, mat2: np.ndarray, max_dist_bins: int, h: int = 1
) -> float:
    """Stratum-adjusted correlation coefficient between two dense matrices."""
    a = _smooth2d(mat1.astype(np.float64), h)
    b = _smooth2d(mat2.astype(np.float64), h)
    num, den = 0.0, 0.0
    for d in range(0, max_dist_bins):
        x = np.diagonal(a, d)
        y = np.diagonal(b, d)
        if x.size < 2:
            continue
        sx, sy = x.std(), y.std()
        if sx < 1e-8 or sy < 1e-8:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if not np.isfinite(r):
            continue
        w = x.size * sx * sy        # HiCRep stratum weight
        num += w * r
        den += w
    return float(num / den) if den > 0 else float("nan")


def _try_hicrep(pred_path, true_path, chrom, binsize, max_dist_bins, h):
    """Use the hicrep package if available; return None on any failure."""
    try:
        import cooler
        from hicrep import hicrepSCC

        c1 = cooler.Cooler(pred_path)
        c2 = cooler.Cooler(true_path)
        vals = hicrepSCC(c1, c2, h=h, dBPMax=max_dist_bins * binsize,
                         bDownSample=False, chrNames=[chrom])
        return float(np.asarray(vals).mean())
    except Exception as e:  # noqa: BLE001
        log.debug("hicrep package unavailable/failed (%s); using builtin SCC", e)
        return None


def evaluate(cfg: Config, pred_path: str, chroms_spec: str, h: int = 1,
             max_dist_mb: float = 5.0, sample: str | None = None,
             max_dense_bins: int = 40000) -> dict:
    truth_cool = cfg.get_sample(sample).cool_path
    info = cool_info(truth_cool)
    binsize = info["binsize"]
    max_dist_bins = min(cfg.window.size, int(max_dist_mb * 1e6 / binsize))

    if chroms_spec in ("odd", "even", "all"):
        chroms = select_by_parity(info["chromnames"], chroms_spec)
    elif chroms_spec in ("test", "train"):
        par = cfg.data.test_parity if chroms_spec == "test" else cfg.data.train_parity
        chroms = select_by_parity(info["chromnames"], par)
    else:
        chroms = [c.strip() for c in chroms_spec.split(",") if c.strip()]

    results = {}
    for chrom in chroms:
        nbins_c = info["chromsizes"][chrom] // binsize + 1
        if nbins_c > max_dense_bins:
            # high resolution: score by streaming slices, never a dense chrom
            s_str = scc_streaming(truth_cool, pred_path, chrom,
                                  max_dist_bins, h)
            results[chrom] = {"scc": s_str, "scc_streaming": s_str,
                              "nbins": int(nbins_c), "dense_skipped": True}
            log.info("%s  SCC=%.4f (streamed; %d bins)", chrom, s_str, nbins_c)
            continue
        true = fetch_chrom_matrix(truth_cool, chrom)
        pred = fetch_chrom_matrix(pred_path, chrom)
        n = min(true.shape[0], pred.shape[0])
        true, pred = true[:n, :n], pred[:n, :n]

        s_pkg = _try_hicrep(pred_path, truth_cool, chrom, binsize, max_dist_bins, h)
        s_builtin = scc(pred, true, max_dist_bins, h)

        # distance-stratified Pearson on log1p, and log-MSE within the band
        lt, lp = np.log1p(np.maximum(true, 0)), np.log1p(np.maximum(pred, 0))
        band_mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) < max_dist_bins
        log_mse = float(((lt - lp) ** 2)[band_mask].mean())
        pear = float(np.corrcoef(lt[band_mask], lp[band_mask])[0, 1])

        # prediction-vs-truth value statistics within the scored band
        tb, pb = true[band_mask], pred[band_mask]
        spear = float(_spearman(tb, pb))
        results[chrom] = {
            "scc": s_pkg if s_pkg is not None else s_builtin,
            "scc_builtin": s_builtin,
            "scc_hicrep_pkg": s_pkg,
            "pearson_log": pear,
            "spearman": spear,
            "log_mse": log_mse,
            "nbins": int(n),
            "true_mean": float(tb.mean()),
            "pred_mean": float(pb.mean()),
            "true_nnz_frac": float((tb > 0).mean()),
            "pred_nnz_frac": float((pb > 0).mean()),
            "true_max": float(tb.max()),
            "pred_max": float(pb.max()),
        }
        log.info("%s  SCC=%.4f  pearson_log=%.4f  spearman=%.4f  log_mse=%.4f  "
                 "pred/true mean=%.2f/%.2f",
                 chrom, results[chrom]["scc"], pear, spear, log_mse,
                 pb.mean(), tb.mean())

    sccs = [r["scc"] for r in results.values()
            if r.get("scc") is not None and np.isfinite(r["scc"])]
    pears = [r["pearson_log"] for r in results.values()
             if r.get("pearson_log") is not None and np.isfinite(r["pearson_log"])]
    summary = {
        "per_chrom": results,
        "n_chroms": len(sccs),
        "mean_scc": _stat(sccs, "mean"),
        "median_scc": _stat(sccs, "median"),
        "std_scc": _stat(sccs, "std"),
        "min_scc": _stat(sccs, "min"),
        "max_scc": _stat(sccs, "max"),
        "mean_pearson_log": _stat(pears, "mean"),
    }
    log.info("SCC over %d chroms: mean=%.4f median=%.4f std=%.4f [min=%.4f max=%.4f]",
             len(sccs), summary["mean_scc"], summary["median_scc"],
             summary["std_scc"], summary["min_scc"], summary["max_scc"])
    return summary


def _stat(xs, kind):
    if not xs:
        return float("nan")
    return float(getattr(np, kind)(xs))


def _spearman(a, b):
    if a.size < 2:
        return float("nan")
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return np.corrcoef(ra, rb)[0, 1]


def main():
    ap = argparse.ArgumentParser(description="Evaluate predicted Hi-C vs ground truth.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred", required=True, help="predicted .cool")
    ap.add_argument("--chroms", default="test")
    ap.add_argument("--h", type=int, default=1, help="HiCRep smoothing radius")
    ap.add_argument("--max-dist-mb", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="write JSON summary here")
    ap.add_argument("--sample", default=None, help="sample name (default: first)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    summary = evaluate(cfg, args.pred, args.chroms, args.h, args.max_dist_mb,
                       args.sample)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
