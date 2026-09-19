"""PWM motif scanning: per-bin, strand-separated motif-hit tracks.

Every DNA-binding protein track must name its JASPAR motif (config enforces
this). The genome is scanned once per motif with the log-odds PWM on both
strands; positions scoring >= score_frac * max_score are hits. Hits are
counted per Hi-C bin, giving two per-bin channels per motif (forward and
reverse strand). Keeping strands separate lets the model see *convergent*
motif orientation -- the geometry CTCF loop anchors actually have -- via the
directed fwd_i * rev_j conditioning channels built in `model.py`.

Scans are cached per (genome, chrom, motif, resolution, threshold).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .sequence import genome_key, read_chrom_codes

BG = 0.25            # uniform background
PSEUDO = 0.8         # pseudocount added per base


def read_jaspar(path: str | Path) -> np.ndarray:
    """Parse a JASPAR .jaspar PFM -> (4, L) float count matrix (A,C,G,T)."""
    rows: dict[str, list[float]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        base, rest = line.split(None, 1)
        nums = rest.replace("[", " ").replace("]", " ").split()
        rows[base.upper()] = [float(x) for x in nums]
    try:
        pfm = np.array([rows[b] for b in "ACGT"], dtype=np.float64)
    except KeyError as e:
        raise ValueError(f"{path}: missing base row {e}") from e
    if pfm.shape[0] != 4 or pfm.shape[1] < 4:
        raise ValueError(f"{path}: bad PFM shape {pfm.shape}")
    return pfm


def pwm_logodds(pfm: np.ndarray) -> np.ndarray:
    """(4, L) counts -> (4, L) log-odds vs uniform background."""
    p = (pfm + PSEUDO) / (pfm.sum(axis=0, keepdims=True) + 4 * PSEUDO)
    return np.log2(p / BG)


def revcomp_pwm(pwm: np.ndarray) -> np.ndarray:
    return pwm[::-1, ::-1]


def scan_hits(codes: np.ndarray, pwm: np.ndarray, score_frac: float) -> np.ndarray:
    """Positions (motif start) whose window scores >= score_frac * max score.

    codes: uint8 A=0 C=1 G=2 T=3 N=4. N gets the worst log-odds of its column
    so windows containing N essentially never pass the threshold.
    """
    L = pwm.shape[1]
    n = codes.shape[0] - L + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    lut = np.vstack([pwm, pwm.min(axis=0, keepdims=True)])   # row 4 = N penalty
    score = np.zeros(n, dtype=np.float32)
    for k in range(L):
        score += lut[codes[k : k + n], k].astype(np.float32)
    thresh = score_frac * float(pwm.max(axis=0).sum())
    return np.nonzero(score >= thresh)[0]


def bin_hit_counts(positions: np.ndarray, n_bins: int, binsize: int) -> np.ndarray:
    """Count motif hits per Hi-C bin (by motif start position)."""
    if positions.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    bins = positions // binsize
    bins = bins[bins < n_bins]
    return np.bincount(bins, minlength=n_bins).astype(np.float32)


def cached_motif_positions(
    fasta: str, chrom: str, motif_id: str, motif_dir: str,
    score_frac: float, cache_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Base-pair positions of forward and reverse motif hits (cached).

    Bin-free: used by fragment-level models that need where a site actually
    sits inside a restriction fragment, not just how many fall in a bin.
    """
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    key = f"motifpos_{genome_key(fasta)}_{chrom}_{motif_id}_f{score_frac:.2f}"
    path = cache / f"{key}.npz"
    if path.exists():
        z = np.load(path)
        return z["fwd"], z["rev"]
    pfm = read_jaspar(Path(motif_dir).expanduser() / f"{motif_id}.jaspar")
    pwm = pwm_logodds(pfm)
    codes = read_chrom_codes(fasta, chrom)
    fwd = scan_hits(codes, pwm, score_frac)
    rev = scan_hits(codes, revcomp_pwm(pwm), score_frac)
    np.savez_compressed(path, fwd=fwd, rev=rev)
    return fwd, rev


def cached_motif_tracks(
    fasta: str, chrom: str, motif_id: str, motif_dir: str,
    n_bins: int, binsize: int, score_frac: float, cache_dir: str,
) -> np.ndarray:
    """(2, n_bins) float32: forward- and reverse-strand hit counts per bin."""
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    key = (
        f"motif_{genome_key(fasta)}_{chrom}_{motif_id}_{binsize}"
        f"_f{score_frac:.2f}"
    )
    path = cache / f"{key}.npz"
    if path.exists():
        arr = np.load(path)["tracks"]
        if arr.shape == (2, n_bins):
            return arr
    pfm = read_jaspar(Path(motif_dir).expanduser() / f"{motif_id}.jaspar")
    pwm = pwm_logodds(pfm)
    codes = read_chrom_codes(fasta, chrom)
    fwd = bin_hit_counts(scan_hits(codes, pwm, score_frac), n_bins, binsize)
    rev = bin_hit_counts(
        scan_hits(codes, revcomp_pwm(pwm), score_frac), n_bins, binsize
    )
    tracks = np.stack([fwd, rev])
    np.savez_compressed(path, tracks=tracks)
    return tracks
