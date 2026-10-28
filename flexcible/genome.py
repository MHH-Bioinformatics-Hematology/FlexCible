"""Genome / chromosome bookkeeping: the train/test split lives here.

The first experiment trains on odd-numbered chromosomes and tests on even ones
(a clean held-out split that also generalises across chromosome size). Helpers
here turn a parity choice into a concrete chromosome list given what the cool
file actually contains.
"""
from __future__ import annotations

import re

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def chrom_number(name: str) -> int | None:
    """Integer index of an autosome, accepting both 'chr1' and bare '1' naming.
    Returns None for X/Y/MT and any non-autosome."""
    m = re.fullmatch(r"(?:chr)?(\d+)", name)
    return int(m.group(1)) if m else None


def match_name(chrom: str, available: list[str]) -> str | None:
    """Resolve a chromosome name against a file's naming convention.

    Hi-C cool files here use bare names ('1'), bigwigs use 'chr1'. This matches
    a canonical name to whatever a given file actually uses (exact, with-, or
    without- 'chr' prefix)."""
    avail = set(available)
    bare = chrom[3:] if chrom.startswith("chr") else chrom
    for cand in (chrom, bare, f"chr{bare}"):
        if cand in avail:
            return cand
    return None


def select_by_parity(available: list[str], parity: str) -> list[str]:
    """Filter chromosome names by odd/even autosome number, preserving order.

    Non-autosomes (chrX, chrY, ...) are never selected by a parity filter.
    """
    if parity not in ("odd", "even", "all"):
        raise ValueError(f"parity must be odd|even|all, got {parity!r}")
    out = []
    for name in available:
        n = chrom_number(name)
        if n is None:
            continue
        if parity == "all" or (parity == "odd" and n % 2 == 1) or (
            parity == "even" and n % 2 == 0
        ):
            out.append(name)
    return out


def resolve_chroms(cfg_chroms: list[str] | None, available: list[str]) -> list[str]:
    """Return canonical chromosome names (as they appear in the cool file).

    If `cfg_chroms` is None, use every autosome present, ordered by number. If it
    is given, each entry is matched naming-insensitively ('chr1' <-> '1')."""
    if cfg_chroms is None:
        auto = [(chrom_number(c), c) for c in available if chrom_number(c) is not None]
        return [c for _, c in sorted(auto)]
    out, missing = [], []
    for c in cfg_chroms:
        m = match_name(c, available)
        (out if m else missing).append(m or c)
    if missing:
        raise ValueError(
            f"Chromosomes not present in cool file: {missing}. "
            f"Available: {sorted(available)[:30]}"
        )
    return out


def split_chroms(
    cfg_chroms: list[str] | None,
    available: list[str],
    train_parity: str,
    test_parity: str,
) -> tuple[list[str], list[str]]:
    """Return (train_chroms, test_chroms) honouring the configured parities."""
    pool = resolve_chroms(cfg_chroms, available)
    train = select_by_parity(pool, train_parity)
    test = select_by_parity(pool, test_parity)
    return train, test


def read_bed_regions(path: str) -> dict[str, list[tuple[int, int]]]:
    """chrom -> sorted [(start, end)] from a BED (comment lines ignored)."""
    out: dict[str, list[tuple[int, int]]] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            f = line.split()
            out.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    for c in out:
        out[c].sort()
    return out


def merge_regions(regions: dict[str, list[tuple[int, int]]]
                  ) -> dict[str, list[tuple[int, int]]]:
    """Merge overlapping or touching intervals per chromosome, so "wholly
    inside one region" (window_region_state) means inside their union."""
    out: dict[str, list[tuple[int, int]]] = {}
    for c, ivs in regions.items():
        merged: list[list[int]] = []
        for s, e in sorted(ivs):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[c] = [(s, e) for s, e in merged]
    return out


def window_region_state(regions, chrom: str, start_bp: int, end_bp: int) -> str:
    """"inside" if the window lies wholly in a region, "overlap" if it touches
    one, else "outside"."""
    from .genome import match_name as _mn

    name = _mn(chrom, list(regions.keys())) if regions else None
    if name is None:
        return "outside"
    touched = False
    for s, e in regions[name]:
        if e <= start_bp:
            continue
        if s >= end_bp:
            break
        touched = True
        if s <= start_bp and e >= end_bp:
            return "inside"
    return "overlap" if touched else "outside"
