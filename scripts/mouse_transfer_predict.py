"""crosscell_predict.py with an on-disk memo of the binned bigWig signal and an optional marks shuffle
(mouse_transfer.py predict).

Marks shuffle (environment MT_SHUFFLE=1): every ChIP track of a chromosome is rolled circularly along the
chromosome by the same offset (a fraction 0.25 to 0.75 of its length, fixed by the chromosome name), so
the marks stay internally coherent and keep their genome-wide statistics, but no longer sit at their own
locus; sequence, GC and motif planes are untouched.

    IV_TRACK_CACHE=~/scratch/mt_tracks python scripts/mouse_transfer_predict.py <crosscell_predict args>
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

CACHE = Path(os.environ.get("IV_TRACK_CACHE", str(Path.home() / "scratch/mt_tracks")))
CACHE.mkdir(parents=True, exist_ok=True)
SHUF = os.environ.get("MT_SHUFFLE") == "1"
_orig = pc.read_track_binned


def _cached(path, chrom, n_bins, binsize):
    st = os.stat(path)
    key = hashlib.md5(f"{path}|{st.st_size}|{st.st_mtime_ns}|{chrom}|{n_bins}|{binsize}".encode()).hexdigest()
    f = CACHE / f"{key}.npy"
    a = None
    if f.exists():
        try:
            a = np.load(f)
        except Exception:                                                    # noqa: BLE001
            a = None
    if a is None:
        a = _orig(path, chrom, n_bins, binsize)
        tmp = f.with_name(f"{key}.{os.getpid()}.tmp.npy")
        np.save(tmp, a)
        os.replace(tmp, f)
    if SHUF:
        h = int(hashlib.md5(f"mt-shuffle|{chrom}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        a = np.roll(a, int(len(a) * (0.25 + 0.5 * h)), axis=-1)
    return a


pc.read_track_binned = _cached
sys.argv = [str(REPO / "scripts/crosscell_predict.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
