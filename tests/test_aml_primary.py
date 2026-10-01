"""Primary-AML helpers: quantile matching of input tracks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ap = pytest.importorskip("aml_primary")


def _q(ref):
    return np.quantile(ref, np.linspace(0, 1, ap.NQ))


def test_quantile_map_is_monotone_and_matches_the_reference():
    rng = np.random.default_rng(0)
    ref = rng.lognormal(0, 1, 200_000)
    src = rng.gamma(2.0, 3.0, 100_000)                 # another scale entirely
    f = ap.quantile_map(src, _q(ref))
    out = f(src)
    order = np.argsort(src)
    assert np.all(np.diff(out[order]) >= -1e-9)        # order kept
    assert np.median(out) == pytest.approx(np.median(ref), rel=0.02)
    assert np.mean(out) == pytest.approx(np.mean(ref), rel=0.02)


def test_quantile_map_ties_get_the_block_mean():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 10, 100_000)                  # mean of lower half = 2.5
    src = np.concatenate([np.zeros(50_000), rng.uniform(1, 2, 50_000)])
    f = ap.quantile_map(src, _q(ref))
    assert f(np.array([0.0]))[0] == pytest.approx(2.5, abs=0.05)
    assert f(src).mean() == pytest.approx(ref.mean(), rel=0.01)
