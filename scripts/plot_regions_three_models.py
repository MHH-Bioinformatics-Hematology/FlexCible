"""Region panels: measured above the diagonal, prediction below, one row per model.

Akita's and Hi-cGAN's predictions are read from the rasters the Hi-cGAN project
already produced (akita_testset_rasters.npz) -- nothing is re-run. Only our own
prediction is computed here, and it is cropped to Akita's own 448-bin span so
all three models are drawn on identical pixels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import torch                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config                          # noqa: E402
from flexcible.genome import match_name                      # noqa: E402
from flexcible.matrix import MatrixNormalizer, open_cool     # noqa: E402
from flexcible.model import build_spatial_cond               # noqa: E402
from flexcible.precompute import (assemble_chrom, chrom_nbins_of,
                                    sample_track_files)        # noqa: E402
from flexcible.tracks import TrackNormalizer                 # noqa: E402
from scripts.score_like_hicgan import load_model               # noqa: E402

RAST = ("/mnt/processed/wolffjoa/restart_of_prediction_project/"
        "akita_normalization/akita_testset_rasters.npz")
GM = 1                      # cells = ['HFF', 'GM12878']; index 1 is GM12878


def dense(vals: np.ndarray, side: int, offset: int) -> np.ndarray:
    r, c = np.triu_indices(side, offset)
    M = np.full((side, side), np.nan, np.float32)
    M[r, c] = vals
    return np.where(np.isnan(M), M.T, M)


def our_prediction(cfg, ckpt, chrom, start, end, side, device="cpu"):
    model, channels = load_model(cfg, ckpt, device)
    size, res = cfg.window.size, cfg.data.resolution
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    mat_norm = MatrixNormalizer.load(
        ckpt / f"norm_matrix_{cfg.get_sample(None).name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    clr = open_cool(cfg.get_sample(None).cool_path)
    name = match_name(chrom, list(clr.chromnames))
    nb = chrom_nbins_of(clr, name)
    sym = track_norm.apply(assemble_chrom(
        cfg, cfg.get_sample(None), sample_track_files(cfg, cfg.get_sample(None)),
        name, nb))
    centre = (start + end) // 2 // res
    st = max(0, min(nb - size, centre - size // 2))
    ct = sym[:, st:st + size]
    if ct.shape[1] < size:
        ct = np.concatenate(
            [ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
    cond = torch.from_numpy(ct[None]).to(device)
    pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
             if n_motifs else None)
    with torch.no_grad():
        out = model(build_spatial_cond(cond, pairs, None, cfg.model.cond_mode))
    P = mat_norm.decode(out[0, 0].float().cpu().numpy())
    P = 0.5 * (P + P.T)
    # Akita crops its 512-bin window to the central `side` bins; match that
    lo = (size - side) // 2
    return P[lo:lo + side, lo:lo + side]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/akita_native.yaml")
    ap.add_argument("--checkpoint", default="runs/akita_native/checkpoint-best")
    ap.add_argument("--scores", default="runs/akita_native/score_like_hicgan.csv")
    ap.add_argument("--n", type=int, default=6)
    # HiCExplorer's hicPlotMatrix default colour map
    ap.add_argument("--cmap", default="RdYlBu_r")
    ap.add_argument("--vmax", type=float, default=2.0)
    ap.add_argument("--out", default="runs/regions_three_models.png")
    a = ap.parse_args()

    z = np.load(RAST, allow_pickle=True)
    side, off = int(z["side"]), int(z["offset"])
    regions = list(z["regions"])
    print(f"rasters: {len(regions)} regions, {side}x{side}, offset {off}, "
          f"cells {list(z['cells'])}")

    sc = pd.read_csv(a.scores).dropna(subset=["ours"])
    sc = sc[sc.region.isin(regions)].sort_values("ours")
    # across the difficulty range, not the flattering end
    picks = sc.iloc[np.linspace(0, len(sc) - 1, a.n).astype(int)]

    cfg = Config.load(a.config)
    ckpt = Path(a.checkpoint)

    fig, axes = plt.subplots(3, a.n, figsize=(3.1 * a.n, 9.8))
    axes = np.atleast_2d(axes)
    rows = ["Akita", "Hi-cGAN", "ours (multi-omic U-Net)"]

    for col, (_, r) in enumerate(picks.iterrows()):
        k = regions.index(r.region)
        chrom, span = r.region.split(":")
        s, e = (int(x) for x in span.split("-"))
        truth = dense(z["akita_target"][k, :, GM], side, off)
        preds = [dense(z["akita_pred"][k, :, GM], side, off),
                 dense(z["hicgan"][k], side, off),
                 our_prediction(cfg, ckpt, chrom, s, e, side)]
        scores = [r.akita_gm12878, r.hicgan_fixed, r.ours]

        for row in range(3):
            ax = axes[row, col]
            keep = np.triu(np.ones((side, side), bool), 1)
            mat = np.where(keep, truth, preds[row])
            ax.imshow(mat, cmap=a.cmap, vmin=-a.vmax, vmax=a.vmax,
                      interpolation="nearest")
            ax.plot([0, side - 1], [0, side - 1], color="#444", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.text(0.03, 0.97, f"r = {scores[row]:.3f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                              boxstyle="round,pad=0.25"))
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=11)
            if row == 0:
                ax.set_title(f"{chrom}:{s/1e6:.2f}-{e/1e6:.2f} Mb",
                             fontsize=9)

    im = axes[0, 0].images[0]
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("log observed/expected (Akita space)")
    fig.suptitle(
        "Above the diagonal: measured Hi-C.  Below: prediction.  "
        "Regions span the range of our per-region Pearson, worst to best.\n"
        "Akita and Hi-cGAN predictions are the ones already produced by the "
        "Hi-cGAN project; only our row is computed here.", fontsize=10)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out, "and", out.with_suffix(".pdf"))
    for _, r in picks.iterrows():
        print(f"  {r.region}: akita {r.akita_gm12878:.3f}  "
              f"hicgan {r.hicgan_fixed:.3f}  ours {r.ours:.3f}")


if __name__ == "__main__":
    main()
