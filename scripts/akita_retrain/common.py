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

# Resolution configurations (registry 79b). "2kb" is the original Akita geometry. "8kb" and
# "32kb" add pooling (first block pool 8, tower 10 / 12 blocks) so that the output bin is
# 2^13 / 2^15 bp, on a 4 194 304 bp input; at 32 kb 128 bins remain, 8 are cropped per side
# (112 target bins = 3.67 Mb). Targets come from our 10 kb / 25 kb O/E maps (SRC_RES).
CFGS = {
    "2kb": dict(SEQ_LEN=1_048_576, BIN=2048, NB=448, CROP=65_536, SRC_RES=2000),
    "8kb": dict(SEQ_LEN=4_194_304, BIN=8192, NB=448, CROP=262_144, SRC_RES=10000),
    "32kb": dict(SEQ_LEN=4_194_304, BIN=32768, NB=112, CROP=262_144, SRC_RES=25000),
}
CFG = os.environ.get("AKITA_CFG", "2kb")
SEQ_LEN, BIN, NB, CROP, SRC_RES = (CFGS[CFG][k] for k in ("SEQ_LEN", "BIN", "NB", "CROP", "SRC_RES"))
DIAG = 2
NV = NB * (NB - 1) // 2 - (NB - 1)       # 99 681 at 2 kb and 8 kb, 6 105 at 32 kb
STRIDE_TRAIN = NB // 4 * BIN             # a quarter of the target span (229 376 bp at 2 kb)
STRIDE_EVAL = NB // 2 * BIN              # half of it (validation windows, tiling)
SUFFIX = "" if CFG == "2kb" else f"_{CFG}"


def windows_file(split: str) -> Path:
    return W / "data" / f"windows{SUFFIX}_{split}.tsv"

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
    return Path(os.environ.get("AKITA_DATA_LOCAL", str(W / "data"))) / f"bands{SUFFIX}"


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
