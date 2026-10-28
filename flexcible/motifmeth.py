"""Methylation-aware directed motif strength (config motifs.motif_meth).

The CpG methylation inside a CTCF motif lowers the CTCF height beyond what the
motif score explains (registry section 41), and the 10 kb bin-averaged Meth
track dilutes it. This module builds, per cell, the CTCF forward and reverse
hit planes with every site weighted by its own methylation:

    weight = 1 - methylated fraction of the CpGs wholly inside the motif
             (reads pooled over the CpGs, at least MIN_SITE_COV reads);
    weight = 1 for a site without covered motif CpG (no CpG in the motif, or
             coverage below MIN_SITE_COV): the unmodified hit.

A bin holds the sum of the weights of the hits starting in it, so for a cell
without CpG data the planes equal the unmodified hit counts of
flexcible.motifs.cached_motif_tracks. The sites are the PWM hits at
motifs.score_frac (the channel threshold of the model), taken from
motifs.cached_motif_positions; the CpG file is the per-chromosome npz written
by scripts/chip_seq_meth_relationships.py (cpg_build): <chrom>_pos (C position
of the CpG, both strands summed), <chrom>_cov, <chrom>_meth.

Because a weight of 1 means both "unmethylated" and "no information", a third
row, ``motifmeth:covered``, carries the fraction of the bin's hits (both
strands) that have a measured motif methylation (0 without CpG data, in bins
without hits, and when the Meth track is dropped). The planes enter the model
as one more directed motif pair (pseudo motif id "<ID>:meth") next to the
unmodified pair of the same motif; the covered row is a conditioning row in the
sequence-feature block.

Cells without a CpG.npz in one of their extra_bigwig_dirs (HUVEC, NHEK, PC-9,
the RRBS cells HCT116, MCF-7, HFF, patients without WGBS) get the unmodified
planes and covered = 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SUFFIX = ":meth"
CPG_FILE = "CpG.npz"
COVERED_ROW = "motifmeth:covered"       # sym row name (seq-feature block)
COVERED_FEATURE = "motifmeth_covered"   # channels["seq_features"] name
MIN_SITE_COV = 10                       # reads pooled over the motif's CpGs


def is_meth_id(mid: str) -> bool:
    return mid.endswith(SUFFIX)


def base_id(mid: str) -> str:
    return mid[: -len(SUFFIX)] if is_meth_id(mid) else mid


def meth_id(mid: str) -> str:
    return mid + SUFFIX


def find_cpg_file(dirs) -> Path | None:
    """First CpG.npz among a sample's extra_bigwig_dirs (earlier folders win)."""
    for d in dirs or []:
        p = Path(d) / CPG_FILE
        if p.exists():
            return p
    return None


_CPG_CACHE: dict = {}


def load_cpg_chrom(path: Path, chrom: str):
    """(pos, cov, meth) of one chromosome, or None when the file lacks it."""
    key = str(path)
    if key not in _CPG_CACHE:
        _CPG_CACHE.clear()               # one cell at a time
        _CPG_CACHE[key] = np.load(path)
    z = _CPG_CACHE[key]
    if chrom + "_pos" not in z.files:
        return None
    return z[chrom + "_pos"], z[chrom + "_cov"], z[chrom + "_meth"]


def site_methylation(starts: np.ndarray, length: int, cpg) -> tuple[np.ndarray, np.ndarray]:
    """Methylated fraction of the CpGs wholly inside each motif site and a
    covered mask. A site covers C positions start <= p < start + length - 1
    (the CpG's G is the next base)."""
    pos, cov, meth = cpg
    a = np.searchsorted(pos, starts)
    b = np.searchsorted(pos, starts + length - 1)
    cc = np.concatenate([[0], np.cumsum(cov, dtype=np.int64)])
    cm = np.concatenate([[0], np.cumsum(meth, dtype=np.int64)])
    reads, mreads = cc[b] - cc[a], cm[b] - cm[a]
    covered = reads >= MIN_SITE_COV
    frac = np.where(covered, mreads / np.maximum(reads, 1), 0.0)
    return frac.astype(np.float32), covered


def bin_weights(starts: np.ndarray, w: np.ndarray, n_bins: int, binsize: int) -> np.ndarray:
    if starts.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    b = starts // binsize
    keep = b < n_bins
    return np.bincount(b[keep], weights=w[keep], minlength=n_bins).astype(np.float32)


def methylated_motif_tracks(
    fasta: str, chrom: str, motif_id: str, motif_dir: str, n_bins: int, binsize: int,
    score_frac: float, cache_dir: str, cpg_file: Path | None,
) -> np.ndarray:
    """(3, n_bins) float32: forward and reverse methylation-weighted hit
    counts and the covered fraction of the bin's hits."""
    from .motifs import cached_motif_positions, pwm_logodds, read_jaspar

    fwd, rev = cached_motif_positions(fasta, chrom, motif_id, motif_dir, score_frac, cache_dir)
    length = read_jaspar(Path(motif_dir).expanduser() / f"{motif_id}.jaspar").shape[1]
    cpg = load_cpg_chrom(cpg_file, chrom) if cpg_file is not None else None
    out = np.zeros((3, n_bins), dtype=np.float32)
    n_all = np.zeros(n_bins, dtype=np.float32)
    n_cov = np.zeros(n_bins, dtype=np.float32)
    for k, starts in enumerate((fwd.astype(np.int64), rev.astype(np.int64))):
        if cpg is None:
            w = np.ones(starts.shape[0], dtype=np.float32)
            covered = np.zeros(starts.shape[0], dtype=bool)
        else:
            frac, covered = site_methylation(starts, length, cpg)
            w = 1.0 - frac                       # frac is 0 where not covered
        out[k] = bin_weights(starts, w, n_bins, binsize)
        n_all += bin_weights(starts, np.ones_like(w), n_bins, binsize)
        n_cov += bin_weights(starts, covered.astype(np.float32), n_bins, binsize)
    out[2] = np.where(n_all > 0, n_cov / np.maximum(n_all, 1), 0.0)
    return out


def coupling_rows(sym_rows: list[str], mid: str) -> dict:
    """Row indices (into the full sym rows) of the unmodified pair, the
    methylation-weighted pair and the covered row of motif `mid`."""
    u = [sym_rows.index(f"motif:{mid}:fwd"), sym_rows.index(f"motif:{mid}:rev")]
    m = [sym_rows.index(f"motif:{meth_id(mid)}:fwd"), sym_rows.index(f"motif:{meth_id(mid)}:rev")]
    return {"unmod": u, "meth": m, "covered": sym_rows.index(COVERED_ROW)}


def absent_transform(track_norm, rows: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """The methylation-free value of the weighted rows from the unmodified
    rows: z_meth = a * z_unmod + b per plane (both rows share the log1p
    transform; only their z-score statistics differ), and the covered row's
    absent value. Returns (a (2,), b (2,), covered_absent)."""
    if track_norm.transform != "log1p" or track_norm.norm != "zscore" or track_norm.smooth_bins:
        raise ValueError("motif_meth needs track.transform log1p, track.norm zscore, no smoothing")
    mean, std = np.asarray(track_norm.mean), np.asarray(track_norm.std)
    a = np.array([std[u] / std[m] for u, m in zip(rows["unmod"], rows["meth"])], np.float32)
    b = np.array([(mean[u] - mean[m]) / std[m] for u, m in zip(rows["unmod"], rows["meth"])],
                 np.float32)
    cov_absent = float(track_norm.absent_values()[rows["covered"]])
    return a, b, cov_absent


def drop_methylation(sym: np.ndarray, track_norm, rows: dict) -> np.ndarray:
    """Normalised sym rows (R, n) with the methylation of the motif planes
    removed in place: weighted planes := unmodified planes (same units),
    covered row := its absent value. Returns sym."""
    a, b, cov_absent = absent_transform(track_norm, rows)
    for k in range(2):
        sym[rows["meth"][k]] = a[k] * sym[rows["unmod"][k]] + b[k]
    sym[rows["covered"]] = cov_absent
    return sym
