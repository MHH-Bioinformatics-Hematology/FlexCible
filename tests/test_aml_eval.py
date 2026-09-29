"""AML evaluation helpers: STAG2-KO input rewriting, insulation, loop grid."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
aml = pytest.importorskip("aml_eval")
pyBigWig = pytest.importorskip("pyBigWig")


def _bigwig(path, values, step=None):
    step = step or aml.KO_STEP
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", len(values) * step)])
    bw.addEntries("chr1", 0, values=[float(v) for v in values], span=step, step=step)
    bw.close()


def test_ko_ctcfonly_keeps_ctcf_sites_and_flattens_the_rest(tmp_path):
    v = np.ones(400)
    v[100] = 50.0                      # cohesin at a CTCF site
    v[300] = 40.0                      # cohesin at an enhancer
    v[350:360] = 0.0                   # unmappable
    src, dst = tmp_path / "in.bw", tmp_path / "out.bw"
    _bigwig(src, v)
    peaks = pd.DataFrame({"chrom": ["chr1"], "start": [100 * aml.KO_STEP], "end": [101 * aml.KO_STEP]})
    aml.ko_bigwig(src, dst, "ctcfonly", peaks)
    out = np.array(pyBigWig.open(str(dst)).values("chr1", 0, 400 * aml.KO_STEP))[::aml.KO_STEP]
    assert out[100] == pytest.approx(50.0)
    mean = v.mean()
    assert out[300] == pytest.approx(mean, rel=1e-5)
    assert np.all(out[350:360] == 0)


def test_ko_half_halves_enrichment():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        v = np.full(200, 2.0)
        v[50] = 10.0
        src, dst = Path(d) / "in.bw", Path(d) / "out.bw"
        _bigwig(src, v)
        aml.ko_bigwig(src, dst, "half", None)
        out = np.array(pyBigWig.open(str(dst)).values("chr1", 0, 200 * aml.KO_STEP))[::aml.KO_STEP]
        mean = v.mean()
        assert out[50] == pytest.approx(mean + 0.5 * (10.0 - mean), rel=1e-5)


def test_insulation_dips_at_a_boundary():
    n, w = 60, 5
    L = np.full((n, n), -0.5)
    L[:30, :30] = 0.5
    L[30:, 30:] = 0.5
    prof = aml.insulation_log(L, w)
    assert np.nanargmin(prof) in (29, 30)
    assert np.isnan(prof[:w]).all()


def test_loop_params_use_a_10kb_grid():
    for res in (2000, 5000, 10000, 25000):
        p = aml.loop_params(res)
        cres = res * p.loop_coarsen
        assert cres in (10000, 25000)
        assert p.loop_max_dist * cres == 600000 or cres == 25000
