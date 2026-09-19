"""In-silico perturbation: delete a protein somewhere and see how structure
changes.

Zeroes the chosen tracks (raw space, i.e. genuine "protein absent" -- the
same encoding the degron training samples use) within a genomic interval,
re-predicts the surrounding windows, and renders original prediction vs
perturbed prediction vs difference. Optionally also zeroes the protein's
motif channels (--zero-motifs simulates deleting the binding sites from the
DNA rather than removing the protein).

CLI:
  python -m hicmultiome.perturb --config configs/multicell_25kb.yaml \
      --sample gm12878 --region chr2:20000000-22000000 \
      --tracks CTCF,RAD21,SMC3 --out runs/perturb_chr2.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .matrix import BandedAccumulator, MatrixNormalizer, hann2d, open_cool, plan_windows
from .model import build_spatial_cond
from .plot import parse_region
from .precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from .predict import _resolve_ckpt, build_model, load_norm
from .tracks import TrackNormalizer
from .utils import get_logger, set_seed

log = get_logger()


def _predict_banded(model, channels, sym_full, wins, size, stride, nb, mat_norm,
                    device, batch=8, cond_mode="full"):
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    acc = BandedAccumulator(nb, band=size)
    w2d = hann2d(size)
    for i in range(0, len(wins), batch):
        bw = wins[i : i + batch]
        conds = []
        for w in bw:
            ct = sym_full[:, w.start : w.start + size]
            if ct.shape[1] < size:
                ct = np.concatenate(
                    [ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
            conds.append(ct)
        cond = torch.from_numpy(np.stack(conds)).to(device)
        pairs = None
        if n_motifs:
            pairs = cond[:, offset : offset + 2 * n_motifs].reshape(
                cond.shape[0], n_motifs, 2, size)
        planes = None
        if channels.get("capture_input"):
            # perturbation runs tracks-only (capture never required)
            planes = torch.full((cond.shape[0], 2, size, size), -1.0,
                                device=device)
            planes[:, 1] = 0.0
        with torch.no_grad():
            imgs = model(build_spatial_cond(cond, pairs, planes,
                                            cond_mode)).float().cpu().numpy()
        for w, img in zip(bw, imgs):
            block = np.maximum(mat_norm.decode(img[0]), 0.0)
            acc.add_block(w.start, 0.5 * (block + block.T), w2d)
    return np.where(acc.den > 0, acc.num / np.maximum(acc.den, 1e-9), 0.0)


def _banded_to_dense(banded: np.ndarray, b0: int, b1: int) -> np.ndarray:
    n = b1 - b0
    out = np.zeros((n, n), dtype=np.float32)
    band = banded.shape[1]
    for d in range(min(band, n)):
        vals = banded[b0 : b1 - d, d]
        idx = np.arange(vals.shape[0])
        out[idx, idx + d] = vals
        out[idx + d, idx] = vals
    return out


def perturb(cfg: Config, region: str, tracks: list[str], out_png: str,
            sample_name: str | None = None, zero_motifs: bool = False,
            checkpoint: str | None = None, flank_windows: int = 1):
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_ckpt(cfg, checkpoint)
    sample = cfg.get_sample(sample_name)
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    mat_norm = load_norm(ckpt, "norm_matrix", sample.name, MatrixNormalizer)
    model, channels = build_model(cfg, ckpt, device)

    for t in tracks:
        if t not in channels["tracks"]:
            raise ValueError(f"unknown track {t!r} (have {channels['tracks']})")

    chrom, start, end = parse_region(region)
    clr = open_cool(sample.cool_path)
    from .genome import match_name

    chrom = match_name(chrom, list(clr.chromnames))
    res = cfg.data.resolution
    nb = chrom_nbins_of(clr, chrom)
    rb0, rb1 = start // res, min((end + res - 1) // res, nb)
    size, stride = cfg.window.size, cfg.window.stride

    track_files = sample_track_files(cfg, sample)
    raw = assemble_chrom(cfg, sample, track_files, chrom, nb)

    raw_pert = raw.copy()
    rows = channels["sym_rows"]
    zero_rows = [rows.index(t) for t in tracks]
    if zero_motifs:
        for t in cfg.data.tracks:
            if t.name in tracks and t.motif:
                for strand in ("fwd", "rev"):
                    rn = f"motif:{t.motif}:{strand}"
                    if rn in rows:
                        zero_rows.append(rows.index(rn))
    zero_rows = sorted(set(zero_rows))
    raw_pert[zero_rows, rb0:rb1] = 0.0            # raw zero = absent
    log.info("zeroing rows %s in %s bins %d-%d",
             [rows[i] for i in zero_rows], chrom, rb0, rb1)

    sym = track_norm.apply(raw)
    sym_pert = track_norm.apply(raw_pert)

    # windows overlapping the region, plus flanks for clean merging
    all_wins = plan_windows(nb, size, stride, chrom)
    wins = [w for w in all_wins
            if w.start < rb1 + flank_windows * size
            and w.start + size > rb0 - flank_windows * size]

    orig = _predict_banded(model, channels, sym, wins, size, stride, nb,
                           mat_norm, device, cond_mode=cfg.model.cond_mode)
    pert = _predict_banded(model, channels, sym_pert, wins, size, stride, nb,
                           mat_norm, device, cond_mode=cfg.model.cond_mode)

    p0 = max(rb0 - size // 2, 0)
    p1 = min(rb1 + size // 2, nb)
    d_orig = np.log1p(_banded_to_dense(orig, p0, p1))
    d_pert = np.log1p(_banded_to_dense(pert, p0, p1))
    diff = d_pert - d_orig

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [p0 * res / 1e6, p1 * res / 1e6, p1 * res / 1e6, p0 * res / 1e6]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    vmax = np.quantile(d_orig[d_orig > 0], 0.999) if (d_orig > 0).any() else 1
    for ax, (name, m, cmap, vmin, vm) in zip(axes, [
        ("original prediction", d_orig, "RdYlBu_r", 0, vmax),
        (f"perturbed ({'+'.join(tracks)} deleted)", d_pert, "RdYlBu_r", 0, vmax),
        ("difference (perturbed - original)", diff, "seismic",
         -np.abs(diff).max() or 1, np.abs(diff).max() or 1),
    ]):
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vm, extent=ext)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(f"{chrom} (Mb)")
        ax.axvspan(rb0 * res / 1e6, rb1 * res / 1e6, color="k", alpha=0.06)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"in-silico deletion of {'+'.join(tracks)}"
        f"{' + motif sites' if zero_motifs else ''} in {region} "
        f"(sample {sample.name}, log1p)", fontsize=12)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    log.info("wrote %s", out_png)

    lost = float((d_orig - d_pert).clip(0).sum())
    gained = float((d_pert - d_orig).clip(0).sum())
    log.info("signal lost=%.1f gained=%.1f (log1p units, plotted area)", lost, gained)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="In-silico protein deletion.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--region", required=True, help="e.g. chr2:20000000-22000000")
    ap.add_argument("--tracks", required=True,
                    help="comma-separated protein tracks to delete, e.g. CTCF,RAD21,SMC3")
    ap.add_argument("--zero-motifs", action="store_true",
                    help="also zero the proteins' motif channels (delete the sites)")
    ap.add_argument("--out", required=True, help="output .png")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    perturb(Config.load(args.config), args.region,
            [t.strip() for t in args.tracks.split(",")], args.out,
            args.sample, args.zero_motifs, args.checkpoint)


if __name__ == "__main__":
    main()
