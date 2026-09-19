"""Per-bin DNA sequence features from the genome FASTA.

The Hi-C bins are large (10-25 kb), so the sequence enters the model two ways:
scalar per-bin composition features here (GC fraction, CpG density), and
per-bin motif-hit tracks from PWM scanning in `motifs.py`. Both are computed
once per (genome, chromosome, resolution) and cached as .npz.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_CODE = np.full(256, 4, dtype=np.uint8)          # 4 = N/other
for i, b in enumerate(b"ACGT"):
    _CODE[b] = i
    _CODE[b + 32] = i                             # lower-case


def genome_key(fasta: str) -> str:
    """Stable short key for the FASTA (path + size), used in cache filenames."""
    p = Path(fasta).expanduser()
    raw = f"{p.resolve()}:{p.stat().st_size}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def read_chrom_codes(fasta: str, chrom: str) -> np.ndarray:
    """Chromosome sequence as uint8 codes A=0 C=1 G=2 T=3 N=4."""
    from pyfaidx import Fasta

    fa = Fasta(str(Path(fasta).expanduser()), rebuild=False)
    from .genome import match_name

    name = match_name(chrom, list(fa.keys()))
    if name is None:
        raise KeyError(f"{chrom} not in {fasta} (have {list(fa.keys())[:5]}...)")
    seq = np.frombuffer(fa[name][:].seq.encode("ascii"), dtype=np.uint8)
    return _CODE[seq]


# A=0 C=1 G=2 T=3 N=4 -> reverse complement swaps A<->T, C<->G, leaves N.
_COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
_chrom_code_cache: dict[tuple[str, str], np.ndarray] = {}


def chrom_codes_cached(fasta: str, chrom: str) -> np.ndarray:
    """Per-chromosome base codes, cached in memory (whole chromosome, uint8).

    The learned sequence encoder needs the raw bases of each window, fetched by
    slicing; reading the chromosome once and slicing is far cheaper than
    re-reading the FASTA per window."""
    key = (str(fasta), str(chrom))
    if key not in _chrom_code_cache:
        if len(_chrom_code_cache) > 64:       # bounded; chromosomes are large
            _chrom_code_cache.pop(next(iter(_chrom_code_cache)))
        _chrom_code_cache[key] = read_chrom_codes(fasta, chrom)
    return _chrom_code_cache[key]


def window_codes(fasta: str, chrom: str, start_bin: int, n_bins: int,
                 binsize: int, revcomp: bool = False) -> np.ndarray:
    """Base codes (uint8, A=0..T=3, N=4) for one window's genomic span,
    [start_bin*binsize, (start_bin+n_bins)*binsize), zero-padded past the
    chromosome end with N. `revcomp` returns the reverse complement, matching
    the reverse-strand augmentation used on the rest of the conditioning."""
    codes = chrom_codes_cached(fasta, chrom)
    lo = int(start_bin) * binsize
    hi = lo + n_bins * binsize
    seg = codes[lo:hi]
    if seg.shape[0] < hi - lo:
        seg = np.concatenate(
            [seg, np.full(hi - lo - seg.shape[0], 4, dtype=np.uint8)])
    if revcomp:
        seg = _COMPLEMENT[seg[::-1]]
    return seg


def _binned_sum(x: np.ndarray, n_bins: int, binsize: int) -> np.ndarray:
    """Sum of x per bin over [0, n_bins*binsize), zero-padded past the end."""
    out = np.zeros(n_bins, dtype=np.float64)
    span = min(x.shape[0], n_bins * binsize)
    full = span // binsize
    if full:
        out[:full] = x[: full * binsize].reshape(full, binsize).sum(axis=1)
    if full < n_bins and span > full * binsize:
        out[full] = x[full * binsize : span].sum()
    return out


def sequence_features(
    codes: np.ndarray, features: list[str], n_bins: int, binsize: int
) -> np.ndarray:
    """(n_features, n_bins) float32: composition features per bin.

    gc  : (G+C) / non-N bases in the bin
    cpg : CpG dinucleotides / non-N bases in the bin
    """
    if not features:
        return np.zeros((0, n_bins), dtype=np.float32)
    valid = (codes != 4).astype(np.float32)
    denom = np.maximum(_binned_sum(valid, n_bins, binsize), 1.0)
    rows = []
    for f in features:
        if f == "gc":
            gc = ((codes == 1) | (codes == 2)).astype(np.float32)
            rows.append(_binned_sum(gc, n_bins, binsize) / denom)
        elif f == "cpg":
            cpg = np.zeros_like(valid)
            cg = (codes[:-1] == 1) & (codes[1:] == 2)
            cpg[:-1] = cg.astype(np.float32)
            rows.append(_binned_sum(cpg, n_bins, binsize) / denom)
        else:
            raise ValueError(f)
    return np.stack(rows).astype(np.float32)


def cached_sequence_features(
    fasta: str, chrom: str, features: list[str], n_bins: int, binsize: int,
    cache_dir: str,
) -> np.ndarray:
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    key = f"seq_{genome_key(fasta)}_{chrom}_{binsize}_{'-'.join(features)}"
    path = cache / f"{key}.npz"
    if path.exists():
        arr = np.load(path)["feats"]
        if arr.shape == (len(features), n_bins):
            return arr
    codes = read_chrom_codes(fasta, chrom)
    feats = sequence_features(codes, features, n_bins, binsize)
    np.savez_compressed(path, feats=feats)
    return feats
