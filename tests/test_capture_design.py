"""Capture design options: full bait design from a BED, and capture-input
training with Akita-style validation and reverse-strand augmentation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline import (BINSIZE, MOTIF, _write_bigwig, _write_capture,  # noqa: E402
                           _write_cool, _write_fasta, _write_jaspar)


@pytest.fixture(scope="module")
def wd(tmp_path_factory):
    root = tmp_path_factory.mktemp("capdesign")
    _write_fasta(root / "genome.fa")
    _write_jaspar(root / f"{MOTIF}.jaspar")
    _write_cool(root / "target.cool")
    _write_bigwig(root / "CTCF.bigwig",
                  {"chr1": [5, 12, 20, 30, 40, 55], "chr2": [5, 12, 20, 30]})
    _write_bigwig(root / "H3K27ac.bigwig", {"chr1": [10, 11, 12], "chr2": [8, 9]})
    _write_capture(root / "capture.txt")
    # the design: the three interacting baits plus bait bin 60, which has no
    # significant interaction
    (root / "baits.bed").write_text("".join(
        f"{c}\t{b * BINSIZE}\t{b * BINSIZE + 900}\t{i}\tG\n"
        for c in ("chr1", "chr2") for i, b in enumerate((5, 20, 40, 60))))
    return root


def test_bait_bed_adds_baits_without_interactions(wd):
    from hicmultiome.capture import CaptureData

    old = CaptureData(str(wd / "capture.txt"), "mifsud", 0.0)
    new = CaptureData(str(wd / "capture.txt"), "mifsud", 0.0,
                      baits=str(wd / "baits.bed"))
    s_old, b_old = old.window_block("chr1", 0, 64, BINSIZE)
    s_new, b_new = new.window_block("chr1", 0, 64, BINSIZE)
    assert np.array_equal(s_old, s_new)                 # scores unchanged
    assert set(np.flatnonzero(b_old)) == {5, 20, 40}    # original behaviour
    assert set(np.flatnonzero(b_new)) == {5, 20, 40, 60}


def test_capture_planes_absent_encoding():
    from hicmultiome.train import capture_planes

    tgt = torch.zeros(2, 2, 8, 8)
    tgt[:, 1] = 0.3
    bait = torch.zeros(2, 8)
    bait[:, 2] = 1
    p = capture_planes(tgt, bait, torch.tensor([1.0, 0.0]))
    assert torch.all(p[1, 0] == -1) and torch.all(p[1, 1] == 0)   # absent
    assert torch.all(p[0, 0] == 0.3)
    assert p[0, 1, 2, 5] == 1 and p[0, 1, 5, 2] == 1 and p[0, 1, 4, 5] == 0


def test_capin_trains_with_val_regions_and_reverse(wd):
    """Capture-input model with Akita-style validation (val_regions) and
    reverse augmentation: validation feeds absent capture planes."""
    from hicmultiome import precompute, train
    from hicmultiome.config import Config

    raw = {
        "data": {"samples": [{"name": "s", "cool_path": str(wd / "target.cool"),
                              "bigwig_dir": str(wd),
                              "capture_interactions": str(wd / "capture.txt"),
                              "capture_baits": str(wd / "baits.bed")}],
                 "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                            {"name": "H3K27ac", "type": "histone"}],
                 "resolution": BINSIZE},
        "sequence": {"fasta": str(wd / "genome.fa"), "cache_dir": str(wd / "gcache")},
        "motifs": {"dir": str(wd), "score_frac": 0.8},
        "capture": {"use_as_input": True, "input_dropout": 0.5},
        "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99},
        "model": {"base_channels": 16, "depth": 2},
        "train": {"output_dir": str(wd / "run"), "cache_dir": str(wd / "cache"),
                  "batch_size": 4, "max_steps": 20, "eval_every": 10,
                  "val_regions": 2, "augment_reverse": True, "lr": 1e-3},
    }
    cfg = Config.from_dict(raw)
    precompute.build(cfg)
    train.train(cfg)
    assert (wd / "run" / "best").exists()
