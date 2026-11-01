"""crosscell_predict.py reading the binned bigWig signal from a store (one_model_remaining_mouse.py predict).

The store holds, per bigWig folder, track, chromosome, number of bins and bin size, the array that
flexcible.tracks.read_track_binned returns (built by `one_model_remaining_mouse.py store` from the real bigWigs on
the machine that holds them); on the GPU host the folders contain empty placeholder files only. Optional marks
shuffle (MT_SHUFFLE=1): every ChIP track of a chromosome is rolled circularly by one common offset of 25 to 75
percent of its length, as in mouse_transfer_predict.py.

    MT_STORE=~/scratch/mt_store python scripts/one_model_remaining_mouse_predict.py <crosscell_predict args>
"""
import hashlib
import os
import runpy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import flexcible.tracks as pc                                          # noqa: E402

STORE = Path(os.environ["MT_STORE"])
SHUF = os.environ.get("MT_SHUFFLE") == "1"


def _from_store(path, chrom, n_bins, binsize):
    p = Path(path)
    f = STORE / p.parent.name / f"{p.stem}__{chrom}__{n_bins}__{binsize}.npy"
    a = np.load(f)
    if SHUF:
        h = int(hashlib.md5(f"mt-shuffle|{chrom}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        a = np.roll(a, int(len(a) * (0.25 + 0.5 * h)), axis=-1)
    return a


pc.read_track_binned = _from_store
sys.argv = [str(REPO / "scripts/crosscell_predict.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
