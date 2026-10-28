"""Capture Hi-C interactions as an auxiliary training target.

Parses significant promoter-capture Hi-C interactions (Mifsud et al. 2015,
E-MTAB-2323 format), bins them onto the Hi-C bin grid, and provides per-window
target blocks plus a bait-coverage mask.

The capture design only observes pairs where at least one end is a baited
promoter fragment, so the auxiliary loss is masked to rows/columns that contain
a bait. The model's second output channel therefore learns "how strongly would
a capture experiment link these two bins", which concentrates gradient signal
exactly on enhancer-promoter geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .genome import match_name
from .utils import get_logger

log = get_logger()


def parse_mifsud(path: str, min_score: float):
    """Yield (bait_chrom, bait_mid, other_chrom, other_mid, score).

    Columns (tab-separated, 1 header line):
      chr bait, start bait, end bait, Symbol, Ensembl Gene ID, expression
      quartile, chr, start, end, raw count, log(observed/expected)
    Only cis (same-chromosome) interactions are used.
    """
    with open(path) as f:
        header = f.readline()
        if "bait" not in header:
            raise ValueError(f"{path}: unexpected header {header[:80]!r}")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 11:
                continue
            bc, oc = p[0], p[6]
            if bc != oc:
                continue
            score = float(p[10])
            if score < min_score:
                continue
            bmid = (int(p[1]) + int(p[2])) // 2
            omid = (int(p[7]) + int(p[8])) // 2
            yield bc, bmid, oc, omid, score


def read_bait_bed(path: str) -> dict[str, set[int]]:
    """chrom -> set of bait fragment midpoints from a BED (chrom, start, end, ...)."""
    out: dict[str, set[int]] = {}
    with open(Path(path).expanduser()) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track")):
                continue
            p = line.split("\t")
            out.setdefault(p[0], set()).add((int(p[1]) + int(p[2])) // 2)
    return out


@dataclass
class ChromCapture:
    """Binned capture data for one chromosome."""

    scores: np.ndarray       # (n_bins, n_bins) float32, symmetric, 0 = none
    bait_bins: np.ndarray    # (n_bins,) uint8, 1 where a bait fragment falls


class CaptureData:
    """All cis capture interactions, binned per chromosome on demand."""

    def __init__(self, path: str, fmt: str, min_score: float,
                 baits: str | None = None):
        if fmt != "mifsud":
            raise ValueError(fmt)
        self.by_chrom: dict[str, list[tuple[int, int, float]]] = {}
        self.baits: dict[str, set[int]] = {}
        n = 0
        for bc, bmid, _oc, omid, score in parse_mifsud(str(Path(path).expanduser()), min_score):
            self.by_chrom.setdefault(bc, []).append((bmid, omid, score))
            self.baits.setdefault(bc, set()).add(bmid)
            n += 1
        log.info(
            "capture: %d cis interactions on %d chroms (min_score=%.1f)",
            n, len(self.by_chrom), min_score,
        )
        if baits:
            # the full design: every baited fragment is observed, also those
            # without a significant interaction (their rows hold no contact)
            n_seen = sum(len(v) for v in self.baits.values())
            self.baits = read_bait_bed(baits)
            for c in self.by_chrom:
                self.baits.setdefault(c, set())
            log.info("capture: %d designed baits from %s (%d baits carry an "
                     "interaction)", sum(len(v) for v in self.baits.values()),
                     baits, n_seen)

    def window_block(
        self, chrom: str, start_bin: int, size: int, binsize: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(size x size scores, size bait flags) for one diagonal window --
        O(window) memory (no dense chromosome matrix)."""
        scores = np.zeros((size, size), dtype=np.float32)
        bait = np.zeros(size, dtype=np.uint8)
        name = match_name(chrom, list(self.by_chrom.keys()))
        if name is None:
            return scores, bait
        if not hasattr(self, "_arrays"):
            self._arrays = {}
        if name not in self._arrays:
            a = np.asarray(self.by_chrom[name], dtype=np.float64)
            self._arrays[name] = (
                a[:, 0].astype(np.int64), a[:, 1].astype(np.int64),
                a[:, 2].astype(np.float32),
                np.fromiter(self.baits[name], dtype=np.int64),
            )
        bmid, omid, sc, bts = self._arrays[name]
        lo, hi = start_bin * binsize, (start_bin + size) * binsize
        keep = (bmid >= lo) & (bmid < hi) & (omid >= lo) & (omid < hi)
        i = (bmid[keep] - lo) // binsize
        j = (omid[keep] - lo) // binsize
        np.maximum.at(scores, (i, j), sc[keep])
        np.maximum.at(scores, (j, i), sc[keep])
        b = bts[(bts >= lo) & (bts < hi)]
        bait[(b - lo) // binsize] = 1
        return scores, bait

    MAX_DENSE_BINS = 60000       # ~14 GB float32; beyond this use window_block

    def chrom(self, chrom: str, n_bins: int, binsize: int) -> ChromCapture:
        if n_bins > self.MAX_DENSE_BINS:
            raise MemoryError(
                f"dense capture matrix for {chrom} would be {n_bins}x{n_bins} "
                f"({n_bins**2 * 4 / 2**30:.0f} GiB). Use window_block() or the "
                "interaction-level scorers instead of a whole-chromosome matrix."
            )
        name = match_name(chrom, list(self.by_chrom.keys()))
        scores = np.zeros((n_bins, n_bins), dtype=np.float32)
        bait = np.zeros(n_bins, dtype=np.uint8)
        if name is None:
            return ChromCapture(scores, bait)
        for bmid, omid, score in self.by_chrom[name]:
            i, j = bmid // binsize, omid // binsize
            if i >= n_bins or j >= n_bins:
                continue
            if score > scores[i, j]:
                scores[i, j] = score
                scores[j, i] = score
        for bmid in self.baits[name]:
            b = bmid // binsize
            if b < n_bins:
                bait[b] = 1
        return ChromCapture(scores, bait)

    def train_score_sample(
        self, chroms: list[str], n_bins_of, binsize: int, cap: int = 500000
    ) -> np.ndarray:
        """Positive interaction scores from the training chromosomes.

        Taken straight from the interaction list: binning only ever assigns
        these same values to pixels (collisions keep the max), so the quantile
        is the same and no dense n_bins x n_bins matrix is built -- which at
        1 kb would be hundreds of GiB per chromosome.
        """
        vals = []
        for c in chroms:
            name = match_name(c, list(self.by_chrom.keys()))
            if name is None:
                continue
            v = np.asarray([r[2] for r in self.by_chrom[name]], dtype=np.float32)
            vals.append(v[v > 0][:cap])
        return np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32)


@dataclass
class CaptureNormalizer:
    """Score -> [-1, 1] image encoding with an exact inverse.

    Scores are already log-scale (log(observed/expected)); a single global
    clip value fit on the training chromosomes maps to +1, absence maps to -1.
    """

    clip_quantile: float = 0.999
    scale: float | None = None

    def fit(self, sample_scores: np.ndarray) -> "CaptureNormalizer":
        if sample_scores.size == 0:
            self.scale = 1.0
        else:
            self.scale = float(np.quantile(sample_scores, self.clip_quantile) + 1e-6)
        return self

    def encode(self, scores: np.ndarray) -> np.ndarray:
        x = np.clip(scores / self.scale, 0.0, 1.0)
        return (x * 2.0 - 1.0).astype(np.float32)

    def decode(self, img: np.ndarray) -> np.ndarray:
        x = (np.clip(img, -1.0, 1.0) + 1.0) / 2.0
        return x * self.scale

    def save(self, path: str | Path) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CaptureNormalizer":
        import json

        return cls(**json.loads(Path(path).read_text()))
