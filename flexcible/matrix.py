"""Hi-C matrix I/O: cool reading, diagonal-window extraction, and merge-to-cool.

We only model the main diagonal for now, so for a window of W bins starting at
bin ``s`` the target is the dense WxW intra-window block ``M[s:s+W, s:s+W]``.
Predictions for overlapping windows are blended back into a *banded* full-genome
matrix (only entries with |i-j| < W are ever populated) and written out as a
valid .cool via cooler.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# cool access
# --------------------------------------------------------------------------- #
def open_cool(path: str):
    import cooler

    return cooler.Cooler(path)


def cool_info(path: str) -> dict:
    clr = open_cool(path)
    return {
        "binsize": clr.binsize,
        "chromnames": list(clr.chromnames),
        "chromsizes": {k: int(v) for k, v in clr.chromsizes.items()},
        "nbins": int(clr.bins().shape[0]),
    }


def chrom_nbins(path: str, chrom: str) -> int:
    clr = open_cool(path)
    return int(clr.extent(chrom)[1] - clr.extent(chrom)[0])


def fetch_chrom_matrix(path: str, chrom: str, balance: bool = False) -> np.ndarray:
    """Dense intra-chromosomal matrix for one chromosome (counts by default)."""
    clr = open_cool(path)
    m = clr.matrix(balance=balance).fetch(chrom)
    return np.nan_to_num(np.asarray(m, dtype=np.float32), nan=0.0)


def fetch_diagonal_window(clr, chrom: str, start_bin: int, size: int) -> np.ndarray:
    """WxW diagonal block via a cooler *region* query -- O(window) memory, so
    high-resolution chromosomes never need a dense nbins x nbins matrix."""
    binsize = clr.binsize
    chromlen = int(clr.chromsizes[chrom])
    s = start_bin * binsize
    e = min((start_bin + size) * binsize, chromlen)
    m = clr.matrix(balance=False).fetch(f"{chrom}:{s}-{e}")
    m = np.nan_to_num(np.asarray(m, dtype=np.float32), nan=0.0)
    if m.shape != (size, size):
        out = np.zeros((size, size), dtype=np.float32)
        out[: m.shape[0], : m.shape[1]] = m
        return out
    return m


def extract_diagonal_window(
    full: np.ndarray, start: int, size: int
) -> np.ndarray:
    """WxW block on the diagonal of a dense chromosome matrix, zero-padded at the
    chromosome end so every window is exactly ``size`` x ``size``."""
    n = full.shape[0]
    end = min(start + size, n)
    block = full[start:end, start:end]
    if block.shape != (size, size):
        out = np.zeros((size, size), dtype=np.float32)
        out[: block.shape[0], : block.shape[1]] = block
        return out
    return block.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# window plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    chrom: str
    start: int      # start bin (inclusive)
    size: int       # window side in bins


def plan_windows(nbins: int, size: int, stride: int, chrom: str) -> list[Window]:
    """Tiling of [0, nbins) by overlapping windows; the last window is clamped to
    end exactly at ``nbins`` so the chromosome end is fully covered."""
    if nbins <= size:
        return [Window(chrom, 0, size)]
    starts = list(range(0, nbins - size + 1, stride))
    if starts[-1] != nbins - size:
        starts.append(nbins - size)
    return [Window(chrom, s, size) for s in starts]


# --------------------------------------------------------------------------- #
# contact-value normalisation (global, fit on train windows)
# --------------------------------------------------------------------------- #
@dataclass
class MatrixNormalizer:
    """Counts -> [-1, 1] image with an exact inverse.

    transform:
      log1p   : log1p + quantile-clip scale (a single global log-space value
                maps to +1) -- the original scheme.
      none    : raw counts, same scaling.
      prescaled : the data is ALREADY in a signed transformed space (e.g. an
                Akita-normalised cool: log observed/expected, clipped), so no
                further transform is applied -- only a symmetric scale to
                [-1, 1]. Negative values are preserved, which log1p/none
                would destroy.
      obs_exp : log1p(obs) - log1p(expected(distance)). The per-distance
                expected profile is fit on training windows, so the target is
                distance-decay-free -- the whole loss lands on the structure
                that stratum-adjusted metrics (SCC) actually score. `scale`
                is a quantile of |log-oe| mapping to +/-1.
    """

    transform: str = "log1p"
    clip_quantile: float = 0.999
    scale: float | None = None     # log-space value mapped to +1 (or +/-1)
    headroom: float = 1.0          # prescaled: clip bound maps here, not to 1
    expected: list[float] | None = None    # obs_exp: mean count per distance

    def fit(self, sample_values: np.ndarray) -> "MatrixNormalizer":
        if self.transform == "obs_exp":
            raise ValueError("obs_exp needs fit_obs_exp(blocks)")
        if self.transform == "prescaled":
            q = float(np.quantile(np.abs(sample_values), self.clip_quantile) + 1e-6)
            # keep the extremes off the tanh asymptote
            self.scale = q / max(min(self.headroom, 1.0), 1e-3)
            return self
        x = self._fwd_log(sample_values)
        self.scale = float(np.quantile(x, self.clip_quantile) + 1e-6)
        return self

    def fit_obs_exp(self, blocks: list[np.ndarray]) -> "MatrixNormalizer":
        """Fit the per-distance expected profile + oe scale on sampled
        diagonal windows (all of size W x W)."""
        w = blocks[0].shape[0]
        sums = np.zeros(w)
        cnts = np.zeros(w)
        for b in blocks:
            for d in range(w):
                v = np.diagonal(b, d)
                sums[d] += v.sum()
                cnts[d] += v.size
        self.expected = (sums / np.maximum(cnts, 1)).tolist()
        loge = np.log1p(np.asarray(self.expected))
        oes = []
        for b in blocks[: 200]:
            oe = np.log1p(np.clip(b, 0, None)) - loge[self._dist(w)]
            oes.append(np.abs(oe).ravel())
        self.scale = float(np.quantile(np.concatenate(oes), self.clip_quantile) + 1e-6)
        return self

    @staticmethod
    def _dist(w: int) -> np.ndarray:
        idx = np.arange(w)
        return np.abs(idx[:, None] - idx[None, :])

    def _loge(self, w: int) -> np.ndarray:
        e = np.asarray(self.expected, dtype=np.float64)
        if e.shape[0] < w:
            e = np.pad(e, (0, w - e.shape[0]), mode="edge")
        return np.log1p(e)[self._dist(w)]

    def _fwd_log(self, x: np.ndarray) -> np.ndarray:
        if self.transform == "log1p":
            return np.log1p(np.clip(x, 0, None))
        if self.transform == "none":
            return np.clip(x, 0, None)
        raise ValueError(self.transform)

    def encode(self, mat: np.ndarray) -> np.ndarray:
        """Counts -> image in [-1, 1]."""
        if self.transform == "prescaled":
            return np.clip(mat / self.scale, -1.0, 1.0).astype(np.float32)
        if self.transform == "obs_exp":
            oe = np.log1p(np.clip(mat, 0, None)) - self._loge(mat.shape[0])
            return np.clip(oe / self.scale, -1.0, 1.0).astype(np.float32)
        x = self._fwd_log(mat)
        x = np.clip(x / self.scale, 0.0, 1.0)
        return (x * 2.0 - 1.0).astype(np.float32)

    def decode_oe(self, img: np.ndarray) -> np.ndarray:
        """Image in [-1, 1] -> linear obs/expected ratio (~1 = expected),
        WITHOUT re-applying the distance decay. Only for transform=obs_exp;
        useful to write an obs/exp cool directly (compartments, saddle plots,
        loop calling on decay-free signal)."""
        if self.transform != "obs_exp":
            raise ValueError("decode_oe requires transform=obs_exp")
        oe = np.clip(img, -1.0, 1.0) * self.scale
        return np.exp(oe).astype(np.float64)

    def decode(self, img: np.ndarray) -> np.ndarray:
        """Image in [-1, 1] -> counts (inverse of `encode`)."""
        if self.transform == "prescaled":
            return (np.clip(img, -1.0, 1.0) * self.scale).astype(np.float64)
        if self.transform == "obs_exp":
            oe = np.clip(img, -1.0, 1.0) * self.scale
            return np.maximum(np.expm1(oe + self._loge(img.shape[0])), 0.0)
        x = (np.clip(img, -1.0, 1.0) + 1.0) / 2.0
        x = x * self.scale
        if self.transform == "log1p":
            return np.expm1(x)
        return x

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MatrixNormalizer":
        return cls(**json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# merge predicted windows -> banded chromosome -> cool
# --------------------------------------------------------------------------- #
def hann2d(size: int) -> np.ndarray:
    """2-D Hann taper used to feather overlapping windows (avoids seams)."""
    w = np.hanning(size + 2)[1:-1]      # strictly positive interior weights
    w = np.clip(w, 1e-3, None)
    return np.outer(w, w).astype(np.float32)


class BandedAccumulator:
    """Weighted accumulation of diagonal-block predictions for one chromosome.

    Only entries with 0 <= (j - i) < band are stored, as two (nbins, band)
    arrays (value*weight and weight), so memory is O(nbins * band) rather than
    O(nbins^2).
    """

    def __init__(self, nbins: int, band: int):
        self.nbins = nbins
        self.band = band
        self.num = np.zeros((nbins, band), dtype=np.float64)
        self.den = np.zeros((nbins, band), dtype=np.float64)

    def add_block(self, start: int, block: np.ndarray, weight: np.ndarray) -> None:
        size = block.shape[0]
        for i in range(size):
            gi = start + i
            if gi >= self.nbins:
                break
            # upper triangle within the block: j >= i
            jmax = min(size, self.nbins - start)
            d = np.arange(i, jmax) - i          # offsets 0..
            keep = d < self.band
            d = d[keep]
            cols = np.arange(i, jmax)[keep]
            self.num[gi, d] += block[i, cols] * weight[i, cols]
            self.den[gi, d] += weight[i, cols]

    def to_pixels(self, bin_offset: int):
        """Yield (bin1_id, bin2_id, count) upper-triangle pixels (global ids)."""
        merged = np.where(self.den > 0, self.num / np.maximum(self.den, 1e-9), 0.0)
        rows, offs = np.nonzero(merged)
        bin1 = rows + bin_offset
        bin2 = rows + offs + bin_offset
        counts = merged[rows, offs]
        return bin1, bin2, counts


def write_cool(
    out_path: str,
    template_cool: str,
    chrom_pixels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    integer: bool = False,
) -> None:
    """Create a .cool reusing the bin table of ``template_cool``.

    ``chrom_pixels`` maps chrom -> (bin1_id, bin2_id, count) with *global* bin
    ids. Pixels from all chromosomes are concatenated, sorted, and written.
    """
    import cooler
    import pandas as pd

    clr = open_cool(template_cool)
    bins = clr.bins()[:][["chrom", "start", "end"]].copy()

    b1, b2, cnt = [], [], []
    for _, (x1, x2, c) in chrom_pixels.items():
        b1.append(x1)
        b2.append(x2)
        cnt.append(c)
    if not b1:
        raise ValueError("no pixels to write")
    pixels = pd.DataFrame(
        {
            "bin1_id": np.concatenate(b1).astype(np.int64),
            "bin2_id": np.concatenate(b2).astype(np.int64),
            "count": np.concatenate(cnt).astype(np.float64),
        }
    )
    # cooler requires bin1_id <= bin2_id and sorted order.
    swap = pixels["bin1_id"] > pixels["bin2_id"]
    pixels.loc[swap, ["bin1_id", "bin2_id"]] = pixels.loc[
        swap, ["bin2_id", "bin1_id"]
    ].values
    pixels = pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    dtype = "float64"
    if integer:
        # opt-in: emit observation-like integer counts. Note this discards
        # sub-1 expectations, so never use it for scoring
        pixels["count"] = np.rint(pixels["count"]).astype(np.int64)
        pixels = pixels[pixels["count"] > 0].reset_index(drop=True)
        dtype = "int64"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(
        out_path,
        bins=bins,
        pixels=pixels,
        dtypes={"count": dtype},
        symmetric_upper=True,
        ordered=True,
    )


def write_cool_custom(out_path: str, chrom: str, n_bins: int, binsize: int,
                      bin1: np.ndarray, bin2: np.ndarray,
                      counts: np.ndarray) -> str:
    """Write a one-chromosome cool with a freshly built bin table.

    Needed when an edit changes the genome: the edited chromosome has its own
    length, so no reference bin table applies.
    """
    import cooler
    import pandas as pd

    name = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    bins = pd.DataFrame({"chrom": [name] * n_bins,
                         "start": np.arange(n_bins) * binsize,
                         "end": (np.arange(n_bins) + 1) * binsize})
    px = pd.DataFrame({"bin1_id": bin1.astype(np.int64),
                       "bin2_id": bin2.astype(np.int64),
                       "count": counts.astype(np.float64)})
    swap = px.bin1_id > px.bin2_id
    px.loc[swap, ["bin1_id", "bin2_id"]] = px.loc[swap, ["bin2_id", "bin1_id"]].values
    px = px.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(out_path, bins=bins, pixels=px,
                         dtypes={"count": "float64"},
                         symmetric_upper=True, ordered=True)
    return out_path
