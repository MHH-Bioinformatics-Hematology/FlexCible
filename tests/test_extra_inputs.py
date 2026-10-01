"""Extra per-cell input folders (samples[].extra_bigwig_dirs) and the
methylation channel's binning (scripts/methylation.py)."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _bw(path: Path, value: float):
    import pyBigWig
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", 10_000)])
    bw.addEntries("chr1", 0, values=[value] * 10, span=1000, step=1000)
    bw.close()


def _cfg(tracks, sample):
    from hicmultiome.config import Config, SampleSpec, TrackSpec
    cfg = Config.__new__(Config)
    from hicmultiome.config import DataConfig
    cfg.data = DataConfig(tracks=[TrackSpec(**t) for t in tracks], samples=[SampleSpec(**sample)])
    return cfg


def test_extra_bigwig_dirs_resolution(tmp_path):
    from hicmultiome.precompute import sample_track_files
    chip, meth, other = tmp_path / "chip", tmp_path / "meth", tmp_path / "other"
    for d in (chip, meth, other):
        d.mkdir()
    _bw(chip / "H3K27ac.bigwig", 2.0)
    _bw(meth / "Meth.bigwig", 1.5)
    _bw(meth / "H3K27ac.bigwig", 9.0)      # shadowed by bigwig_dir
    _bw(other / "Meth.bigwig", 1.9)        # shadowed by the first extra folder
    tracks = [{"name": "H3K27ac"}, {"name": "Meth", "type": "methylation"}]
    base = {"name": "s", "cool_path": "x.cool", "bigwig_dir": str(chip)}
    with pytest.raises(FileNotFoundError):
        sample_track_files(_cfg(tracks, base), _cfg(tracks, base).data.samples[0])
    cfg = _cfg(tracks, {**base, "extra_bigwig_dirs": [str(meth), str(other)]})
    files = dict(sample_track_files(cfg, cfg.data.samples[0]))
    assert files["H3K27ac"] == str(chip / "H3K27ac.bigwig")
    assert files["Meth"] == str(meth / "Meth.bigwig")
    # declared missing: no file needed, a raw-zero row
    cfg = _cfg(tracks, {**base, "missing_tracks": ["Meth"]})
    files = dict(sample_track_files(cfg, cfg.data.samples[0]))
    assert files["Meth"] is None


def test_methylation_binning(tmp_path):
    import methylation as me
    bed = tmp_path / "m.bed.gz"
    rows = [  # chrom start end name score strand ts te rgb coverage percent
        ("chr1", 100, 101, 10, 80), ("chr1", 500, 501, 10, 40),   # bin 0: 60 %
        ("chr1", 1500, 1501, 2, 100),                               # bin 1: below MIN_COV
        ("chr2", 50, 51, 20, 0),                                    # bin 0 of chr2: 0 %
        ("chrUn_x", 10, 11, 50, 100)]                               # not a main chromosome
    with gzip.open(bed, "wt") as fh:
        for c, s, e, cov, pct in rows:
            fh.write(f"{c}\t{s}\t{e}\t.\t0\t+\t{s}\t{e}\t0\t{cov}\t{pct}\n")
    sizes = {"chr1": 3000, "chr2": 2000}
    vals = me.bin_values(*me.bin_counts(bed, sizes))
    mean = (8 + 4 + 0) / 40                 # methylated / total reads of covered bins
    assert vals["chr1"][0] == pytest.approx(1.6)
    assert vals["chr1"][1] == pytest.approx(1 + mean)   # low coverage -> genome mean
    assert vals["chr2"][0] == pytest.approx(1.0)
    assert np.all(vals["chr1"] >= 1.0)                  # raw zero stays "no data"
