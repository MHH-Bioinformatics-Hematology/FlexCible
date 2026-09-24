"""Enhancer-promoter loop and TAD evaluation.

Two questions the plain SCC metric does not answer:

1. Loops: does the model rank true capture-Hi-C enhancer-promoter contacts
   highly? Scored on the bait-masked band of the test chromosomes: Pearson /
   Spearman of the predicted capture channel (or the Hi-C channel) against the
   binned capture scores, plus average precision for detecting significant
   interactions.

2. TADs: insulation profiles. Computed directly on log1p counts with a diamond
   window (self-contained -- needs no matrix balancing), then Pearson between
   the true and predicted profiles and a boundary-overlap F1 (minima of the
   insulation profile).

CLI:  python -m hicmultiome.loops --config ... --pred pred.cool \
          [--pred-capture pred_pchic.cool] --chroms test --out metrics.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .capture import CaptureData
from .config import Config
from .genome import select_by_parity
from .matrix import chrom_nbins, cool_info, fetch_chrom_matrix
from .utils import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# loops vs capture Hi-C
# --------------------------------------------------------------------------- #
def capture_metrics(pred: np.ndarray, cap_scores: np.ndarray,
                    bait: np.ndarray, max_dist_bins: int) -> dict:
    n = min(pred.shape[0], cap_scores.shape[0])
    i, j = np.triu_indices(n, k=1)
    keep = (j - i < max_dist_bins) & ((bait[i] > 0) | (bait[j] > 0))
    p = np.log1p(np.maximum(pred[i, j][keep], 0)).astype(np.float64)
    t = cap_scores[i, j][keep].astype(np.float64)
    if p.size < 10 or p.std() < 1e-9 or t.std() < 1e-9:
        return {"n_pairs": int(p.size)}
    from scipy.stats import rankdata

    pear = float(np.corrcoef(p, t)[0, 1])
    spear = float(np.corrcoef(rankdata(p), rankdata(t))[0, 1])
    pos = t > 0
    ap = _average_precision(p, pos) if 0 < pos.sum() < pos.size else float("nan")
    return {
        "n_pairs": int(p.size),
        "n_positive": int(pos.sum()),
        "pearson": pear,
        "spearman": spear,
        "average_precision": ap,
        "positive_rate": float(pos.mean()),
    }


def interaction_strength_metrics(
    pred: np.ndarray, capture: "CaptureData", chrom: str, n_bins: int,
    binsize: int, max_dist_bins: int, seed: int = 0,
) -> dict:
    """Score the INTERACTIONS, not the matrix.

    For every significant capture interaction on this chromosome we take its
    measured strength and the predicted strength at that bin pair; an
    interaction the model did not predict contributes a predicted strength of
    0. Reported: correlation of measured vs predicted strength, and how many
    interactions were recovered at all.

    Because a strength correlation over positives alone cannot tell whether
    the model simply predicts high everywhere, the same number of
    DISTANCE-MATCHED non-interacting bait pairs is drawn as a control, and
    the separation between the two groups is reported as AUROC.
    """
    from .genome import match_name

    name = match_name(chrom, list(capture.by_chrom.keys()))
    if name is None:
        return {"n_interactions": 0}
    arr = np.asarray(capture.by_chrom[name], dtype=np.float64)
    i = (arr[:, 0] // binsize).astype(np.int64)
    j = (arr[:, 1] // binsize).astype(np.int64)
    s = arr[:, 2].astype(np.float64)
    d = np.abs(j - i)
    keep = (i < n_bins) & (j < n_bins) & (d > 0) & (d < max_dist_bins)
    i, j, s, d = i[keep], j[keep], s[keep], d[keep]
    if i.size == 0:
        return {"n_interactions": 0}

    # several fragment-level calls can fall on one bin pair: keep the strongest
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    key = lo * n_bins + hi
    order = np.lexsort((-s, key))
    key, lo, hi, s = key[order], lo[order], hi[order], s[order]
    first = np.ones(key.size, dtype=bool)
    first[1:] = key[1:] != key[:-1]
    lo, hi, s = lo[first], hi[first], s[first]
    d = hi - lo

    n = min(pred.shape[0], n_bins)
    inside = (lo < n) & (hi < n)
    lo, hi, s, d = lo[inside], hi[inside], s[inside], d[inside]
    p = np.maximum(pred[lo, hi], 0.0)              # not predicted -> 0

    # distance-matched negatives: same offsets, from the same bait bins
    rng = np.random.default_rng(seed)
    truth_set = set(zip(lo.tolist(), hi.tolist()))
    neg = []
    baits = np.unique(lo)
    for off in d:
        for _ in range(8):
            b = int(rng.choice(baits))
            o = b + int(off) * (1 if rng.random() < 0.5 else -1)
            a, c = (b, o) if b < o else (o, b)
            if 0 <= a and c < n and a != c and (a, c) not in truth_set:
                neg.append(pred[a, c])
                break
    negp = np.maximum(np.asarray(neg, dtype=np.float64), 0.0)

    out = {
        "n_interactions": int(s.size),
        "n_recovered": int((p > 0).sum()),
        "recovery_rate": float((p > 0).mean()),
        "mean_true_strength": float(s.mean()),
        "mean_pred_strength": float(p.mean()),
        "median_pred_strength": float(np.median(p)),
    }
    if s.std() > 1e-9 and p.std() > 1e-9:
        from scipy.stats import rankdata

        out["strength_pearson"] = float(np.corrcoef(s, p)[0, 1])
        out["strength_spearman"] = float(
            np.corrcoef(rankdata(s), rankdata(p))[0, 1])
    if negp.size >= 3:
        out["mean_pred_strength_negatives"] = float(negp.mean())
        out["auroc_vs_distance_matched"] = _auroc(p, negp)
        out["strength_ratio_pos_neg"] = float(
            p.mean() / max(negp.mean(), 1e-9))
    return out


def interaction_metrics_from_cool(
    pred_cool: str, capture: "CaptureData", chrom: str, n_bins: int,
    binsize: int, max_dist_bins: int, slice_bins: int = 4000, seed: int = 0,
) -> dict:
    """interaction_strength_metrics without ever holding a whole chromosome.

    The predicted cool is read in overlapping diagonal slices (each slice is
    dense but small), so this works at 1 kb where a full chromosome matrix
    would be hundreds of GiB. Interactions are looked up in the slice that
    contains both of their ends; the overlap is the maximum scored distance,
    so every pair is seen exactly once.
    """
    import cooler

    from .genome import match_name

    name = match_name(chrom, list(capture.by_chrom.keys()))
    if name is None:
        return {"n_interactions": 0}
    arr = np.asarray(capture.by_chrom[name], dtype=np.float64)
    lo = np.minimum(arr[:, 0], arr[:, 1]).astype(np.int64) // binsize
    hi = np.maximum(arr[:, 0], arr[:, 1]).astype(np.int64) // binsize
    s = arr[:, 2].astype(np.float64)
    keep = (hi < n_bins) & (hi - lo > 0) & (hi - lo < max_dist_bins)
    lo, hi, s = lo[keep], hi[keep], s[keep]
    if lo.size == 0:
        return {"n_interactions": 0}
    # dedup bin pairs, keeping the strongest
    key = lo * n_bins + hi
    order = np.lexsort((-s, key))
    key, lo, hi, s = key[order], lo[order], hi[order], s[order]
    first = np.ones(key.size, bool)
    first[1:] = key[1:] != key[:-1]
    lo, hi, s = lo[first], hi[first], s[first]

    clr = cooler.Cooler(pred_cool)
    cname = match_name(chrom, list(clr.chromnames))
    step = max(slice_bins, 2 * max_dist_bins)
    p = np.zeros(lo.size)
    rng = np.random.default_rng(seed)
    negs = []
    for start in range(0, n_bins, step - max_dist_bins):
        end = min(start + step, n_bins)
        sel = (lo >= start) & (hi < end)
        if not sel.any() and start + step >= n_bins:
            break
        block = clr.matrix(balance=False).fetch(
            f"{cname}:{start * binsize}-{end * binsize}")
        block = np.nan_to_num(np.asarray(block, dtype=np.float32))
        if sel.any():
            p[sel] = np.maximum(block[lo[sel] - start, hi[sel] - start], 0.0)
            # distance-matched negatives inside the same slice
            d = (hi[sel] - lo[sel])
            b = lo[sel]
            o = b + d * np.where(rng.random(d.size) < 0.5, 1, -1)
            ok = (o >= start) & (o < end) & (o != b)
            a1 = np.minimum(b[ok], o[ok]) - start
            a2 = np.maximum(b[ok], o[ok]) - start
            if a1.size:
                negs.append(np.maximum(block[a1, a2], 0.0))
        del block
    negp = np.concatenate(negs) if negs else np.zeros(0)

    out = {
        "n_interactions": int(s.size),
        "n_recovered": int((p > 0).sum()),
        "recovery_rate": float((p > 0).mean()),
        "mean_true_strength": float(s.mean()),
        "mean_pred_strength": float(p.mean()),
    }
    if s.std() > 1e-9 and p.std() > 1e-9:
        from scipy.stats import rankdata

        out["strength_pearson"] = float(np.corrcoef(s, p)[0, 1])
        out["strength_spearman"] = float(np.corrcoef(rankdata(s), rankdata(p))[0, 1])
    if negp.size >= 3:
        out["mean_pred_strength_negatives"] = float(negp.mean())
        out["auroc_vs_distance_matched"] = _auroc(p, negp)
        out["strength_ratio_pos_neg"] = float(p.mean() / max(negp.mean(), 1e-9))
    return out


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC separating positives from negatives (ties = 0.5)."""
    from scipy.stats import rankdata

    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    rp = r[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _average_precision(score: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(-score)
    pos = positive[order]
    tp = np.cumsum(pos)
    prec = tp / (np.arange(pos.size) + 1)
    denom = max(1, int(positive.sum()))
    return float((prec * pos).sum() / denom)


# --------------------------------------------------------------------------- #
# TADs via insulation profiles
# --------------------------------------------------------------------------- #
def insulation_profile(mat: np.ndarray, window: int) -> np.ndarray:
    """log2(diamond mean / genome-wide diamond mean) per bin; NaN at edges."""
    lm = np.log1p(np.maximum(mat, 0))
    n = lm.shape[0]
    prof = np.full(n, np.nan)
    for b in range(window, n - window):
        prof[b] = lm[b - window : b, b + 1 : b + window + 1].mean()
    med = np.nanmedian(prof)
    if med > 0:
        prof = np.log2(np.maximum(prof, 1e-9) / med)
    return prof


def boundaries(prof: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """Local minima of the insulation profile with prominence >= delta."""
    n = prof.size
    out = []
    for b in range(2, n - 2):
        v = prof[b]
        if not np.isfinite(v):
            continue
        left = prof[max(0, b - 10) : b]
        right = prof[b + 1 : b + 11]
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        if left.size == 0 or right.size == 0:
            continue
        if v <= prof[b - 1] and v <= prof[b + 1] and \
                min(left.max(), right.max()) - v >= delta:
            out.append(b)
    return np.asarray(out, dtype=np.int64)


def boundary_f1(true_b: np.ndarray, pred_b: np.ndarray, tol: int = 2) -> dict:
    def matched(a, b):
        if a.size == 0 or b.size == 0:
            return 0
        return int(sum((np.abs(b - x).min() <= tol) for x in a))

    tp_t = matched(true_b, pred_b)
    tp_p = matched(pred_b, true_b)
    rec = tp_t / max(1, true_b.size)
    prec = tp_p / max(1, pred_b.size)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"n_true": int(true_b.size), "n_pred": int(pred_b.size),
            "precision": prec, "recall": rec, "f1": f1}


def tad_metrics(true: np.ndarray, pred: np.ndarray, window: int) -> dict:
    n = min(true.shape[0], pred.shape[0])
    pt = insulation_profile(true[:n, :n], window)
    pp = insulation_profile(pred[:n, :n], window)
    ok = np.isfinite(pt) & np.isfinite(pp)
    pear = float(np.corrcoef(pt[ok], pp[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    bf = boundary_f1(boundaries(pt), boundaries(pp))
    return {"insulation_pearson": pear, **{f"boundary_{k}": v for k, v in bf.items()}}


# --------------------------------------------------------------------------- #
def evaluate(cfg: Config, pred_path: str, pred_capture_path: str | None,
             chroms_spec: str, insulation_window_bins: int = 10,
             max_dist_mb: float = 5.0, sample: str | None = None) -> dict:
    smp = cfg.get_sample(sample)
    truth_cool = smp.cool_path
    info = cool_info(truth_cool)
    if chroms_spec in ("odd", "even", "all"):
        chroms = select_by_parity(info["chromnames"], chroms_spec)
    elif chroms_spec in ("test", "train"):
        par = cfg.data.test_parity if chroms_spec == "test" else cfg.data.train_parity
        chroms = select_by_parity(info["chromnames"], par)
    else:
        chroms = [c.strip() for c in chroms_spec.split(",") if c.strip()]

    capture = None
    if smp.capture_interactions:
        capture = CaptureData(smp.capture_interactions, cfg.capture.format,
                              cfg.capture.min_score, baits=smp.capture_baits)
    max_dist_bins = min(cfg.window.size, int(max_dist_mb * 1e6 / info["binsize"]))

    results: dict = {"per_chrom": {}}
    for chrom in chroms:
        true = fetch_chrom_matrix(truth_cool, chrom)
        pred = fetch_chrom_matrix(pred_path, chrom)
        r = {"tads": tad_metrics(true, pred, insulation_window_bins)}
        if capture is not None:
            nb = chrom_nbins(truth_cool, chrom)
            cc = capture.chrom(chrom, nb, cfg.data.resolution)
            r["loops_hic_channel"] = capture_metrics(
                pred, cc.scores, cc.bait_bins, max_dist_bins)
            r["interactions_hic_channel"] = interaction_strength_metrics(
                pred, capture, chrom, nb, cfg.data.resolution, max_dist_bins)
            if pred_capture_path:
                pcap = fetch_chrom_matrix(pred_capture_path, chrom)
                r["loops_capture_channel"] = capture_metrics(
                    pcap, cc.scores, cc.bait_bins, max_dist_bins)
                r["interactions_capture_channel"] = interaction_strength_metrics(
                    pcap, capture, chrom, nb, cfg.data.resolution, max_dist_bins)
        results["per_chrom"][chrom] = r
        log.info("%s  insul-pearson=%.4f boundary-F1=%.3f%s", chrom,
                 r["tads"]["insulation_pearson"], r["tads"]["boundary_f1"],
                 "" if "loops_capture_channel" not in r else
                 "  EP-spearman(cap)=%.4f AP=%.4f" % (
                     r["loops_capture_channel"].get("spearman", float("nan")),
                     r["loops_capture_channel"].get("average_precision", float("nan"))))

    def _mean(path):
        vals = []
        for r in results["per_chrom"].values():
            v = r
            for k in path:
                v = v.get(k, {}) if isinstance(v, dict) else {}
            if isinstance(v, float) and np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    results["mean_insulation_pearson"] = _mean(["tads", "insulation_pearson"])
    results["mean_boundary_f1"] = _mean(["tads", "boundary_f1"])
    for ch in ("loops_hic_channel", "loops_capture_channel"):
        for m in ("pearson", "spearman", "average_precision"):
            results[f"mean_{ch}_{m}"] = _mean([ch, m])
    for ch in ("interactions_hic_channel", "interactions_capture_channel"):
        for m in ("strength_pearson", "strength_spearman", "recovery_rate",
                  "auroc_vs_distance_matched"):
            results[f"mean_{ch}_{m}"] = _mean([ch, m])
    log.info("MEAN  insulation-pearson=%.4f  boundary-F1=%.3f  "
             "EP hic-ch spearman=%.4f  EP cap-ch spearman=%.4f",
             results["mean_insulation_pearson"], results["mean_boundary_f1"],
             results["mean_loops_hic_channel_spearman"],
             results["mean_loops_capture_channel_spearman"])
    return results


def main():
    ap = argparse.ArgumentParser(description="Loop (capture) and TAD evaluation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred", required=True, help="predicted Hi-C .cool")
    ap.add_argument("--pred-capture", default=None,
                    help="predicted capture-channel .cool (from predict --pchic-out)")
    ap.add_argument("--chroms", default="test")
    ap.add_argument("--insulation-window", type=int, default=10,
                    help="diamond half-width in bins")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", default=None)
    args = ap.parse_args()
    res = evaluate(Config.load(args.config), args.pred, args.pred_capture,
                   args.chroms, args.insulation_window, sample=args.sample)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
