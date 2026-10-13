"""Shared constants and helpers of the Akita retraining (registry section 73).

Akita (Fudenberg et al. 2020): 1 048 576 bp input, 2048 bp bins, 512 output bins
of which the 32 outermost on each side are cropped, so a window predicts the
upper triangle (diagonal offset 2) of a 448 x 448 map = 99 681 values per
output head. One head per cell (the original has five), original architecture
and hyper-parameters (manuscripts/akita/params.json of ~/src/basenji, copied
unchanged except for the number of output units).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

W = Path(os.environ.get("AKITA_RETRAIN_W", "/mnt/processed/wolffjoa/hic-multiome/akita_retrain"))
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
OE = HM / "crosscell_oe"
FASTA = Path("/home/mh-hannover.local/wolffjoa/data/hic-multiome/genome/hg38.fa")
HOLDOUT_BED = Path("/home/mh-hannover.local/wolffjoa/data/hic-multiome/akita_holdout_valid_test.bed")
TRAIN_REGIONS_BED = Path("/home/mh-hannover.local/wolffjoa/data/hic-multiome/akita/akita_train_regions.bed")
BASENJI_SRC = W / "basenji_src"          # copy of ~/src/basenji/basenji (commit in COMMIT), unmodified
PYLIB = W / "pylib"                      # natsort (pure python) for the TensorFlow env

SEQ_LEN, BIN, NB, DIAG, CROP = 1_048_576, 2048, 448, 2, 65_536
NV = NB * (NB - 1) // 2 - (NB - 1)       # 99 681
STRIDE_TRAIN = 112 * BIN                 # 229 376 bp: a quarter of the 917 504 bp target span
STRIDE_EVAL = 224 * BIN                  # 458 752 bp: half of it (validation windows, tiling)

CELLS = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
         "gm23248", "mcf7", "cd14mono", "pc9"]          # TRAIN12 of hic_measures.py

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
SPLITS = {
    # A: the chromosome-held-out split of our chrho models (data.chroms of
    # all12*_drop_chrho_*.yaml = all but 2, 3, 11, 14, 17, X; validation 4, 12)
    "A": dict(test=["chr2", "chr3", "chr11", "chr14", "chr17"], val=["chr4", "chr12"],
              train=[c for c in AUTOSOMES if c not in
                     ("chr2", "chr3", "chr11", "chr14", "chr17", "chr4", "chr12")]),
    # B: odd chromosomes train, even test; validation = two odd chromosomes
    "B": dict(test=[f"chr{i}" for i in range(2, 23, 2)], val=["chr5", "chr13"],
              train=[f"chr{i}" for i in range(1, 23, 2) if i not in (5, 13)]),
}


def iu():
    return np.triu_indices(NB, DIAG)


def band_dir() -> Path:
    """AKITA_DATA_LOCAL: a local-disk copy of data/ (faster than the share)."""
    return Path(os.environ.get("AKITA_DATA_LOCAL", str(W / "data"))) / "bands"


def seq_dir() -> Path:
    return Path(os.environ.get("AKITA_DATA_LOCAL", str(W / "data"))) / "seq"


def setup_basenji_path():
    import sys
    for p in (str(BASENJI_SRC), str(PYLIB)):
        if p not in sys.path:
            sys.path.insert(0, p)


def window_target(band: np.ndarray, start: int) -> np.ndarray:
    """(NV, ncell) float32 target of the window starting at `start` bp, from a
    chromosome band array (nbins, 448, ncell) (index [bin, distance, cell])."""
    b0 = (start + CROP) // BIN
    r, c = iu()
    return np.asarray(band[b0 + r, c - r], dtype=np.float32)
