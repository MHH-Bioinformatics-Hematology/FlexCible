"""Reading and normalising bigwig tracks into per-bin signal vectors.

These are the model's "words": for every genomic bin (matching the Hi-C bin
size) we read the mean signal of each histone/protein track. A window of W bins
therefore yields a (n_tracks, W) array, which the TrackEncoder later turns into
conditioning tokens.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def discover_tracks(bigwig_dir: str, names: list[str] | None) -> list[tuple[str, str]]:
    """Return an ordered list of (track_name, file_path).

    If `names` is given, files are matched by stem (CTCF -> CTCF.bigwig); the
    order of `names` is preserved so token positions are deterministic.
    """
    files = {}
    for path in sorted(glob.glob(os.path.join(bigwig_dir, "*"))):
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() in (".bigwig", ".bw"):
            files[stem] = path
    if names is None:
        return sorted(files.items())
    out = []
    for n in names:
        if n not in files:
            raise FileNotFoundError(
                f"Track {n!r} not found in {bigwig_dir} "
                f"(have: {sorted(files)})"
            )
        out.append((n, files[n]))
    return out


def read_track_binned(path: str, chrom: str, n_bins: int, binsize: int) -> np.ndarray:
    """Mean bigwig signal per bin over [0, n_bins*binsize) of `chrom`.

    Uses pyBigWig's `stats(..., type="mean", nBins=...)`, which is exact and
    fast. NaNs (no coverage) become 0.
    """
    import pyBigWig

    from .genome import match_name

    bw = pyBigWig.open(path)
    try:
        # cool uses bare names ('1'); bigwigs use 'chr1'. Resolve to this file's.
        bw_chrom = match_name(chrom, list(bw.chroms().keys()))
        if bw_chrom is None:
            raise KeyError(f"{chrom} absent from {path} (have {list(bw.chroms())[:5]}...)")
        chrom = bw_chrom
        chrom_len = bw.chroms()[chrom]
        span = min(n_bins * binsize, chrom_len)
        # nBins covering [0, span); request exactly the bins we have data for.
        nb = max(1, span // binsize)
        vals = bw.stats(chrom, 0, nb * binsize, type="mean", nBins=nb)
        arr = np.array([v if v is not None else 0.0 for v in vals], dtype=np.float32)
    finally:
        bw.close()
    out = np.zeros(n_bins, dtype=np.float32)
    out[: arr.shape[0]] = arr[:n_bins]
    return np.nan_to_num(out, nan=0.0)


def read_all_tracks(
    track_files: list[tuple[str, str]], chrom: str, n_bins: int, binsize: int
) -> np.ndarray:
    """Stack every track for one chromosome into (n_tracks, n_bins).

    A track whose path is None (declared missing/zeroed for this sample)
    contributes a raw-zero row."""
    rows = [
        np.zeros(n_bins, dtype=np.float32) if p is None
        else read_track_binned(p, chrom, n_bins, binsize)
        for _, p in track_files
    ]
    return np.stack(rows, axis=0)


@dataclass
class TrackNormalizer:
    """Per-track signal transform + standardisation.

    Stats are fit ONLY on the training chromosomes (avoids leakage), then frozen
    and reused for val/test/prediction. Persisted as JSON next to checkpoints.
    """

    transform: str = "log1p"
    norm: str = "zscore"
    smooth_bins: int = 0
    mean: list[float] | None = None
    std: list[float] | None = None
    vmin: list[float] | None = None
    vmax: list[float] | None = None

    def _pre(self, x: np.ndarray) -> np.ndarray:
        if self.transform == "log1p":
            x = np.log1p(np.clip(x, 0, None))
        elif self.transform != "none":
            raise ValueError(self.transform)
        if self.smooth_bins and self.smooth_bins > 1:
            k = self.smooth_bins
            kern = np.ones(k, dtype=np.float32) / k
            x = np.stack([np.convolve(row, kern, mode="same") for row in x])
        return x

    def fit(self, per_chrom_tracks: list[np.ndarray],
            valid_rows: list[np.ndarray] | None = None) -> "TrackNormalizer":
        """Fit on a list of (n_tracks, n_bins) arrays (one per train chrom).

        `valid_rows` marks, per entry, which rows carry a real measurement.
        Samples that simply lack an assay contribute all-zero rows, and
        folding those into the mean and standard deviation would shift the
        scale of every sample that DOES have it -- so they are excluded.
        """
        n_rows = per_chrom_tracks[0].shape[0]
        pre = [self._pre(t) for t in per_chrom_tracks]
        if valid_rows is None:
            valid_rows = [np.ones(n_rows, dtype=bool)] * len(pre)
        mean, std, vmin, vmax = [], [], [], []
        for r in range(n_rows):
            vals = [p[r] for p, m in zip(pre, valid_rows) if m[r]]
            if not vals:                      # measured nowhere: neutral scale
                mean.append(0.0); std.append(1.0); vmin.append(0.0); vmax.append(1.0)
                continue
            cat = np.concatenate(vals)
            mean.append(float(cat.mean()))
            std.append(float(cat.std() + 1e-6))
            vmin.append(float(cat.min()))
            vmax.append(float(cat.max() + 1e-6))
        self.mean, self.std, self.vmin, self.vmax = mean, std, vmin, vmax
        return self

    def apply(self, tracks: np.ndarray) -> np.ndarray:
        """Normalise a (n_tracks, n_bins) array using the fitted stats."""
        x = self._pre(tracks)
        if self.norm == "zscore":
            m = np.asarray(self.mean)[:, None]
            s = np.asarray(self.std)[:, None]
            x = (x - m) / s
        elif self.norm == "minmax":
            lo = np.asarray(self.vmin)[:, None]
            hi = np.asarray(self.vmax)[:, None]
            x = (x - lo) / (hi - lo)
        elif self.norm != "none":
            raise ValueError(self.norm)
        return x.astype(np.float32)

    def absent_values(self) -> np.ndarray:
        """Per-row normalised value of a raw-zero ("assay absent") signal --
        what apply() yields for an all-zero track. Used by track dropout."""
        zero = np.zeros((len(self.mean), 1), dtype=np.float32)
        return self.apply(zero)[:, 0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrackNormalizer":
        return cls(**json.loads(Path(path).read_text()))
