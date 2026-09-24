"""Torch dataset over the precomputed window cache.

Each item:
  target : (1|2, W, W) float32 in [-1, 1]  -- Hi-C [+ capture-Hi-C] windows
  cond   : (R, W) float32                  -- all normalised sym rows (ChIP,
                                              seq features, motif fwd/rev)
  bait   : (W,) float32                    -- bait coverage (all-zero if the
                                              cache has no capture data)
  meta   : (chrom, start)

The manifest's channel bookkeeping (how many ChIP/seq rows, which motif rows
form fwd/rev pairs) lives in `.channels`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, cache_dir: str, split: str, val_fraction: float = 0.0,
                 want_val: bool = False, seed: int = 0, chroms=None):
        self.cache_dir = Path(cache_dir)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.channels = {
            k: manifest[k]
            for k in ("sym_rows", "tracks", "seq_features", "motif_ids", "has_capture")
        }
        self.window_size = manifest["window_size"]

        # optional chromosome restriction (fast proxy search on 1-2 chroms)
        chroms = set(chroms) if chroms else None
        self._index: list[tuple[str, int, str, int, str]] = []
        self._shard_cache: dict[str, dict] = {}
        for s in [s for s in manifest["shards"]
                  if s["split"] == split and (chroms is None or s["chrom"] in chroms)]:
            data = np.load(s["path"])
            for i, st in enumerate(data["start"]):
                self._index.append(
                    (s["path"], int(i), s["chrom"], int(st), s.get("sample", "default"))
                )

        if split == "train" and val_fraction > 0:
            rng = np.random.default_rng(seed)
            mask = rng.random(len(self._index)) < val_fraction
            keep = mask if want_val else ~mask
            self._index = [x for x, m in zip(self._index, keep) if m]

    @property
    def n_sym(self) -> int:
        return len(self.channels["sym_rows"])

    @property
    def n_motifs(self) -> int:
        return len(self.channels["motif_ids"])

    @property
    def n_out(self) -> int:
        return 2 if self.channels["has_capture"] else 1

    def motif_row_offset(self) -> int:
        """Index of the first motif row within the sym rows."""
        return len(self.channels["tracks"]) + len(self.channels["seq_features"])

    def grouped_index(self):
        """{shard_path: [(row_in_output, local_index), ...]} so a caller can
        decompress each shard ONCE. Indexing an NpzFile re-decompresses the
        whole array on every access, which at fine resolution means hundreds of
        MB per window; bulk loading is the difference between seconds and
        tens of minutes."""
        out: dict[str, list[tuple[int, int]]] = {}
        for row, (path, local, *_rest) in enumerate(self._index):
            out.setdefault(path, []).append((row, local))
        return out

    def _shard(self, path: str) -> dict:
        if path not in self._shard_cache:
            if len(self._shard_cache) > 4:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            self._shard_cache[path] = np.load(path)
        return self._shard_cache[path]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        path, local, chrom, start, sample = self._index[idx]
        shard = self._shard(path)
        target = shard["target"][local].astype(np.float32)     # (C, W, W)
        cond = shard["cond"][local].astype(np.float32)         # (R, W)
        if "bait" in shard:
            bait = shard["bait"][local].astype(np.float32)
        else:
            bait = np.zeros(self.window_size, dtype=np.float32)
        return {
            "target": torch.from_numpy(target),
            "cond": torch.from_numpy(cond),
            "bait": torch.from_numpy(bait),
            "chrom": chrom,
            "start": start,
            "sample": sample,
        }


def reverse_cond(cond: torch.Tensor, offset: int, n_motifs: int) -> torch.Tensor:
    """Genomic reversal of the 1-D conditioning: flip every row along the
    genome axis and swap each motif's fwd/rev rows (a forward site read
    backwards is a reverse site)."""
    c = torch.flip(cond, dims=[-1]).clone()
    for m in range(n_motifs):
        f = offset + 2 * m
        c[:, [f, f + 1]] = c[:, [f + 1, f]]
    return c


def split_cond(cond: torch.Tensor, offset: int, n_motifs: int):
    """(B, R, W) -> sym (B, R, W) and pairs (B, M, 2, W) views for the model."""
    if n_motifs == 0:
        return cond, None
    b, _, w = cond.shape
    pairs = cond[:, offset : offset + 2 * n_motifs].reshape(b, n_motifs, 2, w)
    return cond, pairs
