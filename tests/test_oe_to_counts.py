"""Round trip: oe_proper.py then oe_to_counts.py reproduces the balanced counts
up to the Gaussian smoothing and clipping of the O/E space."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cooler
import numpy as np

from test_pipeline import BINSIZE, workdir  # noqa: F401  (fixture)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BAND = 30


def _run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def test_oe_roundtrip(workdir):  # noqa: F811
    src = workdir / "target.cool"
    oe = workdir / "target_oe.cool"
    back = workdir / "target_counts.cool"
    _run(SCRIPTS / "oe_proper.py", "--input", src, "--output", oe,
         "--binsize", BINSIZE, "--max-dist", BAND * BINSIZE)
    _run(SCRIPTS / "oe_to_counts.py", "--oe", oe, "--measured", src,
         "--output", back, "--binsize", BINSIZE, "--max-dist", BAND * BINSIZE,
         "--oe-is-target")
    a = cooler.Cooler(str(src)).pixels()[:]
    b = cooler.Cooler(str(back)).pixels()[:]
    m = a.merge(b, on=["bin1_id", "bin2_id"], suffixes=("_obs", "_back"))
    m = m[(m.bin2_id - m.bin1_id <= BAND) & (m.count_obs > 0) & (m.count_back > 0)]
    assert len(m) > 1000
    r = np.corrcoef(np.log(m.count_obs), np.log(m.count_back))[0, 1]
    assert r > 0.9, r
    # region option keeps only pixels inside the region
    reg = workdir / "target_counts_reg.cool"
    _run(SCRIPTS / "oe_to_counts.py", "--oe", oe, "--measured", src, "--output", reg,
         "--binsize", BINSIZE, "--max-dist", BAND * BINSIZE, "--region", "chr1:10000-50000")
    p = cooler.Cooler(str(reg)).pixels()[:]
    assert len(p) > 0 and p.bin2_id.max() < 50 and p.bin1_id.min() >= 10
