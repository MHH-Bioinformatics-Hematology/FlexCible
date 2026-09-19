"""Streaming window dataset: fetch each training window from the cooler and the
tracks on the fly, instead of precomputing every window into a resident cache.

This removes the two limits of the precompute path -- disk (a dense target per
window) and GPU memory (all windows held resident) -- so training can use an
arbitrary stride, including the dense, high-overlap regime (down to stride 1,
as Hi-cGAN used). Storage becomes O(genome) rather than O(windows), and only a
batch lives on the GPU at a time.

The per-window computation is identical to `precompute.process`, so a streamed
window equals the cached one for the same (sample, chromosome, start): the
matrix window is fetched, smoothed and encoded with the same fitted
MatrixNormalizer; the conditioning is the normalised per-bin rows sliced from
the chromosome. The fitted normalisers and the channel layout are loaded from a
cache directory that a `precompute` run already produced (the fit is cheap and
shared); only the per-window slicing is done here, live.

Windows are filtered by the same region holdout and mappability rules. The
empty-window rule (max_zero_fraction) is the one difference: it needs the
fetched target, so applying it up front would defeat streaming; empties (a
small fraction over mappable sequence) are left in and contribute ~zero loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .capture import CaptureData, CaptureNormalizer
from .config import Config
from .genome import read_bed_regions, resolve_chroms, split_chroms, window_region_state
from .matrix import MatrixNormalizer, open_cool, plan_windows
from .precompute import (
    _prep_target,
    assemble_chrom,
    chrom_nbins_of,
    load_mappable,
    mappable_fraction,
    sample_track_files,
)
from .tracks import TrackNormalizer
from .utils import get_logger

log = get_logger()


class StreamWindowDataset(Dataset):
    def __init__(self, cfg: Config, split: str, cache_dir: str,
                 stride: int | None = None):
        self.cfg = cfg
        self.split = split
        self.size = cfg.window.size
        self.res = cfg.data.resolution
        self.stride = int(stride) if stride else int(cfg.window.stride)
        cache = Path(cache_dir)
        self.channels = json.loads((cache / "channels.json").read_text())
        self.has_capture = self.channels["has_capture"]
        self.n_out = 2 if self.has_capture else 1

        self.track_norm = TrackNormalizer.load(cache / "norm_tracks.json")
        self.samples = {s.name: s for s in cfg.data.samples}
        self.track_files = {s.name: sample_track_files(cfg, s) for s in cfg.data.samples}
        self.mat_norm: dict[str, MatrixNormalizer] = {}
        self.cap_norm: dict[str, CaptureNormalizer] = {}
        self.captures: dict[str, CaptureData] = {}
        for s in cfg.data.samples:
            p = cache / f"norm_matrix_{s.name}.json"
            if not p.exists():
                p = cache / "norm_matrix.json"
            self.mat_norm[s.name] = MatrixNormalizer.load(p)
            if self.has_capture and s.capture_interactions:
                self.captures[s.name] = CaptureData(
                    s.capture_interactions, cfg.capture.format, cfg.capture.min_score)
                cp = cache / f"norm_capture_{s.name}.json"
                if cp.exists():
                    self.cap_norm[s.name] = CaptureNormalizer.load(cp)

        self.holdout = (read_bed_regions(cfg.data.holdout_bed)
                        if cfg.data.holdout_bed else None)

        # window index: (sample, chrom, start), filtered by holdout + mappability
        self.index: list[tuple[str, str, int]] = []
        for s in cfg.data.samples:
            clr = open_cool(s.cool_path)
            names = list(clr.chromnames)
            if self.holdout is not None:
                chroms = resolve_chroms(cfg.data.chroms, names)
            else:
                tr, te = split_chroms(cfg.data.chroms, names,
                                      cfg.data.train_parity, cfg.data.test_parity)
                chroms = tr if split == "train" else te
            mappable = load_mappable(s.cool_path)
            for chrom in chroms:
                nb = chrom_nbins_of(clr, chrom)
                for w in plan_windows(nb, self.size, self.stride, chrom):
                    if self.holdout is not None:
                        state = window_region_state(
                            self.holdout, chrom, w.start * self.res,
                            (w.start + w.size) * self.res)
                        if split == "train" and state != "outside":
                            continue
                        if split == "test" and state != "inside":
                            continue
                    if (mappable_fraction(mappable, chrom, w.start, w.size)
                            < cfg.window.min_mappable_fraction):
                        continue
                    self.index.append((s.name, chrom, w.start))

        # per-worker lazy caches (each DataLoader worker is its own process)
        self._clr: dict[str, object] = {}
        self._sym: dict[tuple[str, str], np.ndarray] = {}
        log.info("stream[%s]: %d windows, stride %d (%.1f%% overlap)",
                 split, len(self.index), self.stride,
                 max(0.0, 100 * (1 - self.stride / self.size)))

    def __len__(self) -> int:
        return len(self.index)

    def _clr_for(self, name: str):
        if name not in self._clr:
            self._clr[name] = open_cool(self.samples[name].cool_path)
        return self._clr[name]

    def _sym_for(self, name: str, chrom: str) -> np.ndarray:
        key = (name, chrom)
        if key not in self._sym:
            if len(self._sym) > 3:                 # bound worker memory
                self._sym.pop(next(iter(self._sym)))
            s = self.samples[name]
            nb = chrom_nbins_of(self._clr_for(name), chrom)
            self._sym[key] = self.track_norm.apply(
                assemble_chrom(self.cfg, s, self.track_files[name], chrom, nb))
        return self._sym[key]

    def __getitem__(self, i: int) -> dict:
        name, chrom, start = self.index[i]
        clr = self._clr_for(name)
        from .matrix import fetch_diagonal_window
        block = fetch_diagonal_window(clr, chrom, start, self.size)
        imgs = [self.mat_norm[name].encode(
            _prep_target(block, self.cfg.window.smooth_sigma))]
        bait = np.zeros(self.size, dtype=np.float32)
        if self.has_capture:
            cap = self.captures.get(name)
            if cap is not None:
                cb, bt = cap.window_block(chrom, start, self.size, self.res)
                imgs.append(self.cap_norm[name].encode(cb))
                bait = bt.astype(np.float32)
            else:
                imgs.append(np.full_like(imgs[0], -1.0))
        sym = self._sym_for(name, chrom)
        ct = sym[:, start:start + self.size]
        if ct.shape[1] < self.size:
            ct = np.concatenate(
                [ct, np.zeros((ct.shape[0], self.size - ct.shape[1]), np.float32)], 1)
        return {
            "target": torch.from_numpy(np.stack(imgs).astype(np.float32)),
            "cond": torch.from_numpy(ct.astype(np.float32)),
            "bait": torch.from_numpy(bait),
            "chrom": chrom,
            "start": int(start),
            "sample": name,
        }


def collate(batch):
    """Stack a batch; keep coords as lists for the sequence encoder fetch."""
    return {
        "target": torch.stack([b["target"] for b in batch]),
        "cond": torch.stack([b["cond"] for b in batch]),
        "bait": torch.stack([b["bait"] for b in batch]),
        "chrom": [b["chrom"] for b in batch],
        "start": [b["start"] for b in batch],
        "sample": [b["sample"] for b in batch],
    }
