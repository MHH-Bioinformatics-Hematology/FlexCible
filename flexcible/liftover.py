"""Minimal UCSC-chain liftover (no external tool needed).

Only what a locus comparison needs: map a genomic position, and map a whole
interval by lifting both ends and requiring them to land on the same
chromosome in the same orientation with a comparable span. Chains are read
from the standard UCSC .over.chain(.gz) format.
"""
from __future__ import annotations

import gzip
from bisect import bisect_right
from pathlib import Path


class LiftOver:
    def __init__(self, chain_path: str | Path):
        self.blocks: dict[str, list[tuple[int, int, str, int, int]]] = {}
        op = gzip.open if str(chain_path).endswith(".gz") else open
        with op(chain_path, "rt") as fh:
            src = dst = None
            s_pos = d_pos = 0
            d_strand = "+"
            d_size = 0
            for line in fh:
                line = line.strip()
                if line.startswith("chain"):
                    f = line.split()
                    src, s_pos = f[2], int(f[5])
                    dst, d_strand, d_size, d_pos = f[7], f[9], int(f[8]), int(f[10])
                    continue
                if not line:
                    src = None
                    continue
                if src is None:
                    continue
                parts = line.split()
                size = int(parts[0])
                self.blocks.setdefault(src, []).append(
                    (s_pos, s_pos + size, dst, d_pos, 1 if d_strand == "+" else -1)
                    if d_strand == "+" else
                    (s_pos, s_pos + size, dst, d_size - d_pos - size, -1))
                if len(parts) == 3:
                    s_pos += size + int(parts[1])
                    d_pos += size + int(parts[2])
                else:
                    src = None
        for c in self.blocks:
            self.blocks[c].sort()
        self._starts = {c: [b[0] for b in v] for c, v in self.blocks.items()}

    def convert(self, chrom: str, pos: int):
        """-> (chrom, pos) in the target build, or None if unmapped."""
        v = self.blocks.get(chrom)
        if not v:
            return None
        i = bisect_right(self._starts[chrom], pos) - 1
        if i < 0:
            return None
        s, e, dst, dstart, strand = v[i]
        if not (s <= pos < e):
            return None
        off = pos - s
        return (dst, dstart + off) if strand == 1 else (dst, dstart + (e - s) - off - 1)

    def convert_interval(self, chrom: str, start: int, end: int,
                         max_len_ratio: float = 1.1):
        """Lift an interval; None unless both ends map to one chromosome with a
        span within `max_len_ratio` of the original (rearranged regions are
        dropped rather than silently distorted)."""
        a = self.convert(chrom, start)
        b = self.convert(chrom, end - 1)
        if a is None or b is None or a[0] != b[0]:
            return None
        lo, hi = sorted((a[1], b[1]))
        hi += 1
        span, orig = hi - lo, end - start
        if not (1 / max_len_ratio <= span / max(orig, 1) <= max_len_ratio):
            return None
        return a[0], lo, hi
