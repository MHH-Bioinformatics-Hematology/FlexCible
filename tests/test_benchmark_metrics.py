"""scripts/benchmark_metrics.py: region metrics of the competitor benchmark."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_metrics as bm  # noqa: E402


def _map(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    a = 0.5 * (a + a.T)
    return a


def test_identical_maps_score_one():
    m = _map()
    mask = bm.triu_mask(60, 2)
    assert abs(bm.pearson(m, m, mask) - 1) < 1e-9
    assert abs(bm.scc(m, m, mask) - 1) < 1e-9


def test_scc_ignores_per_diagonal_offsets():
    """A per-distance shift (a different expected) leaves SCC unchanged but
    lowers the pooled Pearson."""
    m = _map()
    n = m.shape[0]
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    shifted = m + 0.2 * d
    mask = bm.triu_mask(n, 2)
    assert abs(bm.scc(shifted, m, mask) - 1) < 1e-9
    assert bm.pearson(shifted, m, mask) < 0.5


def test_mask_and_reach():
    mask = bm.triu_mask(10, 2, max_d=4)
    i, j = np.nonzero(mask)
    assert ((j - i) >= 2).all() and ((j - i) <= 4).all()


def test_boundary_recall_finds_domain_edges():
    n = 90
    m = np.full((n, n), -1.0)
    for s, e in ((0, 30), (30, 60), (60, 90)):
        m[s:e, s:e] = 1.0
    hits, nref = bm.boundary_recall(m, np.array([30, 60]), window=5, tol=1)
    assert nref == 2 and hits == 2
    flat = np.zeros((n, n))
    hits_flat, _ = bm.boundary_recall(flat, np.array([30, 60]), window=5, tol=1)
    assert hits_flat == 0


def test_per_distance_pools_pairs():
    m = _map(20)
    out = bm.per_distance(m, m, bm.triu_mask(20, 2))
    assert min(out) == 2 and max(out) == 19
    a, b = out[5]
    assert np.allclose(a, b) and a.size == 15


def test_summarize_full_grid():
    s = bm.summarize([0.1, 0.2, 0.3, np.nan], "x")
    assert s["x_n"] == 3 and abs(s["x_median"] - 0.2) < 1e-12
    assert set(s) == {"x_median", "x_mean", "x_q25", "x_q75", "x_sd", "x_n"}


def test_local_e1_recovers_compartments():
    import hicreg_audit as ha
    n = 40
    s = np.where(np.arange(n) % 10 < 5, 1.0, -1.0)
    m = 0.5 * np.outer(s, s) + 0.01 * np.random.default_rng(0).normal(size=(n, n))
    e = ha.local_e1(m)
    assert abs(np.corrcoef(e, s)[0, 1]) > 0.99
