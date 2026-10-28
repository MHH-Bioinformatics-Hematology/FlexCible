"""Pipeline diagnostics: is the windowing/VAE/merge able to reproduce real Hi-C?

Reconstructs a ground-truth chromosome through two model-free paths and reports
**Pearson correlation per genomic distance** (the right metric for Hi-C, which is
dominated by distance decay):

  merge-only   : windows -> normalize(encode) -> decode -> Hann-merge -> cool
                 (NO VAE, NO diffusion model)  -> tests the windowing + merge.
  vae-roundtrip: same, but the window image is also pushed through the SDXL VAE
                 encode+decode  -> tests whether the VAE can even represent Hi-C.

If --pred is given, the trained model's prediction is scored on the same axis so
all three curves are directly comparable. A PNG of Pearson-vs-distance is written.

CLI:  python -m flexcible.diagnose --config configs/ctcf_256.yaml --chrom 2 \
          --pred runs/ctcf_256/pred_even.cool --out-dir runs/diag_256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import Config
from .matrix import (
    BandedAccumulator,
    MatrixNormalizer,
    chrom_nbins,
    extract_diagonal_window,
    fetch_chrom_matrix,
    hann2d,
    plan_windows,
)
from .utils import get_logger

log = get_logger()


def _pearson_logs(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(np.log1p(np.clip(x, 0, None)),
                             np.log1p(np.clip(y, 0, None)))[0, 1])


def dist_pearson_full(true: np.ndarray, pred: np.ndarray, maxd: int) -> np.ndarray:
    """Per-distance Pearson(log1p) between two dense matrices."""
    out = np.full(maxd, np.nan)
    for d in range(maxd):
        out[d] = _pearson_logs(np.diagonal(true, d), np.diagonal(pred, d))
    return out


def dist_pearson_banded(true: np.ndarray, banded: np.ndarray, maxd: int) -> np.ndarray:
    """Per-distance Pearson where `pred` is the banded form: column d is diagonal d."""
    n = true.shape[0]
    out = np.full(maxd, np.nan)
    for d in range(min(maxd, banded.shape[1])):
        out[d] = _pearson_logs(np.diagonal(true, d), banded[: n - d, d])
    return out


def merge_blocks(full: np.ndarray, wins, size: int, mat_norm: MatrixNormalizer) -> np.ndarray:
    """Reconstruct a dense banded matrix from per-window encode/decode, exactly
    mirroring the predict-time merge (no model -- an upper bound on fidelity)."""
    n = full.shape[0]
    acc = BandedAccumulator(n, band=size)
    w2d = hann2d(size)
    for w in wins:
        block = extract_diagonal_window(full, w.start, size)
        rec = np.maximum(mat_norm.decode(mat_norm.encode(block)), 0.0)
        acc.add_block(w.start, 0.5 * (rec + rec.T), w2d)
    # banded reconstruction: column d holds diagonal d (entries (i, i+d))
    return np.where(acc.den > 0, acc.num / np.maximum(acc.den, 1e-9), 0.0)


def diagnose(cfg: Config, chrom: str, pred_cool: str | None, out_dir: str,
             sample: str | None = None, max_bins: int | None = None):
    truth_cool = cfg.get_sample(sample).cool_path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    size = cfg.window.size
    maxd = size

    # normaliser fitted during precompute (same one the model used)
    norm_path = Path(cfg.train.cache_dir) / "norm_matrix.json"
    if norm_path.exists():
        mat_norm = MatrixNormalizer.load(norm_path)
        log.info("loaded normaliser (scale=%.3f) from %s", mat_norm.scale, norm_path)
    else:
        full0 = fetch_chrom_matrix(truth_cool, chrom)
        band = np.concatenate([np.diagonal(full0, o) for o in range(size)])
        mat_norm = MatrixNormalizer(transform=cfg.window.transform,
                                    clip_quantile=cfg.window.clip_quantile).fit(band[band > 0])
        log.info("fit normaliser on the fly (scale=%.3f)", mat_norm.scale)

    full = fetch_chrom_matrix(truth_cool, chrom)
    if max_bins and full.shape[0] > max_bins:
        log.info("slicing chrom to first %d bins for memory", max_bins)
        full = full[:max_bins, :max_bins]
    n = full.shape[0]
    wins = plan_windows(n, size, cfg.window.stride, chrom)
    log.info("chrom %s: %d bins, %d windows, max distance %d bins (%.1f Mb)",
             chrom, n, len(wins), maxd, maxd * cfg.data.resolution / 1e6)

    curves = {}

    # 1) merge-only (no model): upper bound that the windowing+merge can reach
    recon_merge = merge_blocks(full, wins, size, mat_norm)
    curves["merge-only (no model)"] = dist_pearson_banded(full, recon_merge, maxd)

    # 2) trained model prediction
    if pred_cool and Path(pred_cool).exists():
        pred = fetch_chrom_matrix(pred_cool, chrom)
        m = min(pred.shape[0], n)
        curves["model prediction"] = dist_pearson_full(full[:m, :m], pred[:m, :m], maxd)

    # report
    log.info("=== Pearson per genomic distance (log1p), chrom %s ===", chrom)
    sample_d = [1, 2, 4, 8, 16, 32, 64, min(128, maxd - 1)]
    header = "dist(bins)  " + "  ".join(f"d={d:<4d}" for d in sample_d) + "   mean"
    log.info(header)
    for name, c in curves.items():
        vals = "  ".join(f"{c[d]:6.3f}" if np.isfinite(c[d]) else "   nan" for d in sample_d)
        log.info("%-28s %s   %6.3f", name, vals, np.nanmean(c))

    _plot_curves(curves, cfg.data.resolution, chrom, out / f"pearson_vs_distance_chr{chrom}.png")
    np.savez(out / f"curves_chr{chrom}.npz",
             **{k.split()[0]: v for k, v in curves.items()})
    return curves


def _plot_curves(curves, resolution, chrom, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, c in curves.items():
        d_mb = np.arange(len(c)) * resolution / 1e6
        ax.plot(d_mb, c, label=name, lw=2)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("genomic distance (Mb)")
    ax.set_ylabel("Pearson (log1p) per distance")
    ax.set_title(f"Hi-C reconstruction fidelity by distance — chr{chrom}")
    ax.set_ylim(-0.2, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("wrote %s", out_path)


def main():
    ap = argparse.ArgumentParser(description="Diagnose the Hi-C pipeline (per-distance Pearson).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--chrom", required=True, help="chromosome name as in the cool (e.g. 2)")
    ap.add_argument("--pred", default=None, help="optional model prediction .cool to also score")
    ap.add_argument("--out-dir", default="runs/diagnose")
    ap.add_argument("--sample", default=None)
    ap.add_argument("--max-bins", type=int, default=None,
                    help="slice the chromosome to its first N bins (high-res memory guard)")
    args = ap.parse_args()
    diagnose(Config.load(args.config), args.chrom, args.pred, args.out_dir,
             args.sample, args.max_bins)


if __name__ == "__main__":
    main()
