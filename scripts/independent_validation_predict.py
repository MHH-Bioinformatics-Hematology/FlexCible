"""crosscell_predict.py with an on-disk memo of the binned bigWig signal (independent_validation.py predict).

Reading the 7 to 13 bigWigs of a cell for every chromosome takes minutes per model on the share and is
identical for every model of one resolution. The binned arrays are stored once per (file, chromosome, bins,
bin size) in $IV_TRACK_CACHE; nothing else of crosscell_predict.py changes, and the library is not touched.

    IV_TRACK_CACHE=~/scratch/iv_tracks python scripts/independent_validation_predict.py <crosscell_predict args>
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

CACHE = Path(os.environ.get("IV_TRACK_CACHE", str(Path.home() / "scratch/iv_tracks")))
CACHE.mkdir(parents=True, exist_ok=True)
_orig = pc.read_track_binned


def _cached(path, chrom, n_bins, binsize):
    st = os.stat(path)
    key = hashlib.md5(f"{path}|{st.st_size}|{st.st_mtime_ns}|{chrom}|{n_bins}|{binsize}".encode()).hexdigest()
    f = CACHE / f"{key}.npy"
    if f.exists():
        try:
            return np.load(f)
        except Exception:                                                    # noqa: BLE001
            pass
    a = _orig(path, chrom, n_bins, binsize)
    tmp = f.with_name(f"{key}.{os.getpid()}.tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, f)
    return a


pc.read_track_binned = _cached
sys.argv = [str(REPO / "scripts/crosscell_predict.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
