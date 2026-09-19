"""Region-example figure: three models against the measured map, split on the diagonal.

Three genomic regions of Akita's published GM12878 test set are shown as columns,
one row per model. In every panel the upper triangle is the measured matrix and
the lower triangle is that model's prediction of the same pixels. All three
predictions live in Akita space, so one diverging scale centred at zero, clipped
at +/-2, serves the whole figure.

Akita's predictions, the measured target it was scored against and Hi-cGAN's
predictions all come from akita_testset_rasters.npz, where they are already
reduced to identical pixels: the upper triangle of a 448 x 448 matrix above
diagonal offset 2, 99,681 values per region. 448 bins of 2048 bp is 917,504 bp
centred in each 1,048,576 bp test region, which is what Akita emits after its
target crop. Our model is predicted here, on CPU, and resampled onto exactly that
crop and reduced to exactly those pixels, so all nine panels are drawn on one
pixel set.

The last axis of akita_pred / akita_target is the cell type; index 1 is GM12878.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import score_like_hicgan as slh  # noqa: E402

from hicmultiome.config import Config  # noqa: E402
from hicmultiome.genome import match_name  # noqa: E402
from hicmultiome.matrix import MatrixNormalizer, open_cool  # noqa: E402
from hicmultiome.model import build_spatial_cond  # noqa: E402
from hicmultiome.precompute import (assemble_chrom, chrom_nbins_of,  # noqa: E402
                                    sample_track_files)
from hicmultiome.tracks import TrackNormalizer  # noqa: E402

RASTERS = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/"
               "akita_normalization/akita_testset_rasters.npz")
OUTDIR = REPO / "runs/region_figure"
AKITA_RES = 2048

# Chosen to span the range of difficulty rather than the flattering end. Ranking
# the 222 test regions all three models cover by the mean of their three recorded
# per-region Pearsons puts these at roughly the 1st, 50th and 97th percentile.
REGIONS = [
    ("chr4:33370112-34418688", "hardest decile"),
    ("chr8:126361600-127410176", "median difficulty"),
    ("chr20:47089664-48138240", "easiest decile"),
]

ROWS = ["Akita", "Hi-cGAN", "multi-omic U-Net"]
TAGS = ["a", "b", "c"]
VMIN, VMAX = -2.0, 2.0
CMAP = "RdBu_r"
BAD = "#dcdcdc"


class OurModel:
    """Our multi-omic U-Net, loaded once and asked for one region at a time."""

    def __init__(self, config: str, checkpoint: str):
        self.cfg = Config.load(config)
        self.sample = self.cfg.get_sample(None)
        ckpt = Path(checkpoint)
        self.device = "cpu"
        self.model, self.channels = slh.load_model(self.cfg, ckpt, self.device)
        self.size, self.res = self.cfg.window.size, self.cfg.data.resolution
        self.n_motifs = len(self.channels["motif_ids"])
        self.offset = len(self.channels["tracks"]) + len(self.channels["seq_features"])
        self.mat_norm = MatrixNormalizer.load(
            ckpt / f"norm_matrix_{self.sample.name}.json")
        self.track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
        self.src = open_cool(self.sample.cool_path)
        self.track_files = sample_track_files(self.cfg, self.sample)
        self._cache: dict[str, np.ndarray] = {}

    def _cond(self, chrom: str) -> np.ndarray:
        if chrom not in self._cache:
            name = match_name(chrom, list(self.src.chromnames))
            nb = chrom_nbins_of(self.src, name)
            self._cache[chrom] = self.track_norm.apply(
                assemble_chrom(self.cfg, self.sample, self.track_files, name, nb))
        return self._cache[chrom]

    def predict(self, chrom: str, start: int, end: int, crop_start: int,
                side: int) -> np.ndarray:
        """Predict the region, resampled onto Akita's cropped 2048 bp grid."""
        sym_full = self._cond(chrom)
        nb = sym_full.shape[1]
        centre_bin = (start + end) // 2 // self.res
        st = max(0, min(nb - self.size, centre_bin - self.size // 2))
        ct = sym_full[:, st:st + self.size]
        if ct.shape[1] < self.size:
            ct = np.concatenate(
                [ct, np.zeros((ct.shape[0], self.size - ct.shape[1]), np.float32)], 1)
        cond = torch.from_numpy(ct[None]).to(self.device)
        pairs = (cond[:, self.offset:self.offset + 2 * self.n_motifs]
                 .reshape(1, self.n_motifs, 2, self.size) if self.n_motifs else None)
        with torch.no_grad():
            out = self.model(build_spatial_cond(cond, pairs, None,
                                                self.cfg.model.cond_mode))
        P = self.mat_norm.decode(out[0, 0].float().cpu().numpy())
        P = 0.5 * (P + P.T)
        return slh.resample(P, st * self.res, self.res, crop_start, side, AKITA_RES)


def dense(vals: np.ndarray, side: int, offset: int) -> np.ndarray:
    r, c = np.triu_indices(side, offset)
    m = np.full((side, side), np.nan)
    m[r, c] = vals
    m[c, r] = vals
    return m


def score(pred: np.ndarray, meas: np.ndarray) -> float:
    """Their metric, on the shared pixel set: zeros in the target are missing
    values rather than measurements, and so are non-finite pixels on either side."""
    ok = (np.isfinite(pred) & np.isfinite(meas) & (meas != 0))
    if ok.sum() < 100 or pred[ok].std() == 0 or meas[ok].std() == 0:
        return float("nan")
    return float(pearsonr(pred[ok], meas[ok]).statistic)


def split_panel(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    n = upper.shape[0]
    out = np.full((n, n), np.nan)
    iu, il = np.triu_indices(n, 1), np.tril_indices(n, -1)
    out[iu] = upper[iu]
    out[il] = lower[il]
    return out


def mb(x: int) -> str:
    return f"{x / 1e6:.2f}"


def main():
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/akita_native.yaml"))
    ap.add_argument("--checkpoint",
                    default=str(REPO / "runs/akita_native/checkpoint-best"))
    ap.add_argument("--outstem", default=str(OUTDIR / "region_examples"))
    ap.add_argument("--vmax", type=float, default=2.0,
                    help="symmetric colour limit; the target's own clip is 2")
    a = ap.parse_args()
    global VMIN, VMAX
    VMAX = a.vmax
    VMIN = -a.vmax

    OUTDIR.mkdir(parents=True, exist_ok=True)
    z = np.load(RASTERS, allow_pickle=True)
    side, offset = int(z["side"]), int(z["offset"])
    cells = list(z["cells"])
    gm = cells.index("GM12878")
    print(f"rasters: {len(z['regions'])} regions, {side} bins, offset {offset}, "
          f"GM12878 at cell index {gm} of {cells}")
    index = {str(r): i for i, r in enumerate(z["regions"])}

    ours = OurModel(a.config, a.checkpoint)

    panels, scores = [], {}
    for key, note in REGIONS:
        i = index[key]
        chrom, span = key.split(":")
        start, end = (int(v) for v in span.split("-"))
        crop_bins = ((end - start) // AKITA_RES - side) // 2
        crop_start = start + crop_bins * AKITA_RES
        crop_end = crop_start + side * AKITA_RES

        meas = z["akita_target"][i][:, gm].astype(float)
        pred = {
            "Akita": z["akita_pred"][i][:, gm].astype(float),
            "Hi-cGAN": z["hicgan"][i].astype(float),
            "multi-omic U-Net": ours.predict(chrom, start, end, crop_start,
                                             side)[np.triu_indices(side, offset)],
        }
        for name, P in pred.items():
            scores[(key, name)] = score(P, meas)
        panels.append({"key": key, "chrom": chrom, "note": note,
                       "crop_start": crop_start, "crop_end": crop_end,
                       "meas": dense(meas, side, offset),
                       "pred": {n: dense(P, side, offset) for n, P in pred.items()}})
        print(f"{key}  ({note})  {(meas == 0).mean() * 100:.1f}% of target pixels zero")
        for n in ROWS:
            print(f"    {n:<18s} r = {scores[(key, n)]:.3f}")

    # ---- draw -------------------------------------------------------------
    mpl.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "font.family": "sans-serif",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(BAD)

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.5),
                             gridspec_kw={"wspace": 0.07, "hspace": 0.07})
    im = None
    for j, p in enumerate(panels):
        for i, name in enumerate(ROWS):
            ax = axes[i, j]
            P = p["pred"][name]
            im = ax.imshow(split_panel(p["meas"], P), cmap=cmap, vmin=VMIN,
                           vmax=VMAX, interpolation="nearest", origin="upper")
            ax.add_line(Line2D([0, side - 1], [0, side - 1], color="0.3", lw=0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.975, 0.955, "measured", ha="right", va="top",
                    transform=ax.transAxes, fontsize=6.5, color="0.15")
            ax.text(0.025, 0.035, "prediction", ha="left", va="bottom",
                    transform=ax.transAxes, fontsize=6.5, color="0.15")
            r = scores[(p["key"], name)]
            ax.text(0.025, 0.955, "r = n/a" if not np.isfinite(r) else f"r = {r:.2f}",
                    ha="left", va="top", transform=ax.transAxes, fontsize=8,
                    weight="bold", color="0.1",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                              alpha=0.8))
            if j == 0:
                ax.set_ylabel(f"({TAGS[i]}) {name}", fontsize=9, labelpad=6)
            if i == len(ROWS) - 1:
                bar = 200_000 / AKITA_RES          # 200 kb in bins
                x0, y = side * 0.045, side * 0.895
                ax.plot([x0, x0 + bar], [y, y], color="0.2", lw=1.4,
                        solid_capstyle="butt")
                ax.text(x0 + bar / 2, y - side * 0.018, "200 kb", ha="center",
                        va="bottom", fontsize=6.5, color="0.2")
        axes[0, j].set_title(
            f"{p['chrom']}:{mb(p['crop_start'])}–{mb(p['crop_end'])} Mb\n"
            f"{p['note']}", fontsize=8.5, pad=5)

    fig.subplots_adjust(left=0.075, right=0.875, top=0.905, bottom=0.02)
    cax = fig.add_axes([0.893, 0.31, 0.014, 0.34])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"log(observed / expected), clipped at ±{VMAX:g}", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2.5)
    cb.outline.set_linewidth(0.6)

    for ext in ("png", "pdf"):
        out = f"{a.outstem}.{ext}"
        fig.savefig(out, dpi=400 if ext == "png" else None, facecolor="white")
        print(f"wrote {out}")

    tab = pd.DataFrame([{"region": p["key"],
                         "shown": f"{p['chrom']}:{p['crop_start']}-{p['crop_end']}",
                         "note": p["note"],
                         **{n: scores[(p["key"], n)] for n in ROWS}} for p in panels])
    tab.to_csv(f"{a.outstem}_scores.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
