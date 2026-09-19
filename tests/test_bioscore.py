"""Tests for the feature-level Hi-C score.

The point of a metric is that it separates a right answer from a wrong one, so
these tests are mostly of the form "build a map whose answer is known, check
the score says so". A synthetic map with planted domains and planted loops
stands in for real Hi-C; the score has to give a perfect answer for the map
against itself, roughly nothing for a shuffled or structureless map, and
something in between for a degraded one.
"""
from __future__ import annotations

import numpy as np
import pytest

from hicmultiome import bioscore as bs


# --------------------------------------------------------------------------- #
def synthetic(n: int = 448, seed: int = 0, loops=((60, 140), (200, 300),
                                                  (120, 260)),
              edges=(0, 70, 150, 230, 310, 380, None)) -> np.ndarray:
    """A log(o/e) map with domains and point loops, in Akita space."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n, n))
    edges = [n if e is None else e for e in edges]
    for a, b in zip(edges[:-1], edges[1:]):
        M[a:b, a:b] += 0.55
    for i, j in loops:
        y, x = np.mgrid[0:n, 0:n]
        M += 1.5 * np.exp(-(((y - i) ** 2 + (x - j) ** 2) / (2 * 6.0 ** 2)))
        M += 1.5 * np.exp(-(((y - j) ** 2 + (x - i) ** 2) / (2 * 6.0 ** 2)))
    M += 0.05 * rng.standard_normal((n, n))
    return 0.5 * (M + M.T)


@pytest.fixture(scope="module")
def truth():
    return synthetic()


@pytest.fixture(scope="module")
def params():
    return bs.BioParams()


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def test_dense_from_triu_round_trip():
    side, off = 12, 2
    r, c = np.triu_indices(side, off)
    vals = np.arange(r.size, dtype=np.float32)
    M = bs.dense_from_triu(vals, side, off)
    assert np.allclose(M, M.T, equal_nan=True)
    assert M[r[0], c[0]] == vals[0]
    # the excluded near-diagonal band stays NaN
    assert np.isnan(M[0, 1])


def test_oe_from_log_is_exp_and_clipped():
    v = np.array([[0.0, 1.0], [-50.0, 50.0]])
    o = bs.oe_from_log(v)
    assert o[0, 0] == pytest.approx(1.0)
    assert o[0, 1] == pytest.approx(np.e)
    assert o[1, 0] == pytest.approx(np.exp(-8.0))
    assert o[1, 1] == pytest.approx(np.exp(8.0))


def test_coarsen_block_averages():
    M = np.arange(16, dtype=float).reshape(4, 4)
    C = bs.coarsen(M, 2)
    assert C.shape == (2, 2)
    assert C[0, 0] == pytest.approx(M[:2, :2].mean())
    assert bs.coarsen(M, 1) is M


def test_coarsen_drops_the_ragged_edge():
    M = np.ones((5, 5))
    assert bs.coarsen(M, 2).shape == (2, 2)


def test_donut_kernel_excludes_core_and_cross():
    k = bs.donut_kernel(1, 4)
    c = 4
    assert k[c, c] == 0                      # centre
    assert k[c, c + 3] == 0                  # on the cross
    assert k[c + 3, c] == 0
    assert k[c + 3, c + 3] == 1              # in the donut
    assert np.array_equal(k, k.T)
    assert np.array_equal(k, k[::-1, ::-1])


def test_local_enrichment_sees_a_planted_peak():
    n = 60
    M = np.ones((n, n))
    y, x = np.mgrid[0:n, 0:n]
    M += 3.0 * np.exp(-(((y - 20) ** 2 + (x - 40) ** 2) / (2 * 1.5 ** 2)))
    e = bs.local_enrichment(M, bs.BioParams(donut_inner=1, donut_outer=4))
    assert e[20, 40] > 2.0
    assert e[50, 55] == pytest.approx(1.0, abs=0.05)


def test_local_enrichment_is_flat_on_a_flat_map():
    e = bs.local_enrichment(np.full((40, 40), 3.0),
                            bs.BioParams(donut_inner=1, donut_outer=4))
    assert np.nanmax(np.abs(e - 1.0)) < 1e-9


# --------------------------------------------------------------------------- #
# loop calling
# --------------------------------------------------------------------------- #
def test_call_loops_finds_planted_peaks_and_suppresses_duplicates():
    n = 80
    M = np.ones((n, n))
    y, x = np.mgrid[0:n, 0:n]
    for i, j in ((15, 40), (30, 65)):
        M += 4.0 * np.exp(-(((y - i) ** 2 + (x - j) ** 2) / (2 * 1.5 ** 2)))
    p = bs.BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4,
                     loop_min_dist=5, loop_max_dist=60, loop_min_enrich=1.5,
                     loop_nms=3)
    a, v = bs.call_loops(M, p)
    assert len(a) == 2
    found = {tuple(x) for x in a}
    for want in ((15, 40), (30, 65)):
        assert min(max(abs(i - want[0]), abs(j - want[1]))
                   for i, j in found) <= 1
    assert (v > 1.5).all()


def test_call_loops_top_k_ignores_the_absolute_thresholds():
    rng = np.random.default_rng(1)
    M = 1.0 + 0.01 * rng.standard_normal((80, 80))
    M = 0.5 * (M + M.T)
    p = bs.BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4,
                     loop_min_dist=5, loop_max_dist=60, loop_min_enrich=5.0,
                     loop_nms=3)
    assert len(bs.call_loops(M, p)[0]) == 0          # nothing clears 5.0
    a, _ = bs.call_loops(M, p, top_k=7)
    assert len(a) == 7                               # rank cut still returns k
    assert (a[:, 1] - a[:, 0] >= 5).all()


def test_call_loops_respects_the_distance_window():
    n = 80
    M = np.ones((n, n))
    y, x = np.mgrid[0:n, 0:n]
    M += 4.0 * np.exp(-(((y - 30) ** 2 + (x - 32) ** 2) / (2 * 1.5 ** 2)))
    p = bs.BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4,
                     loop_min_dist=10, loop_max_dist=60, loop_min_enrich=1.5)
    assert len(bs.call_loops(M, p)[0]) == 0      # 2 bins apart, below min_dist


def test_match_loops_counts():
    t = np.array([[10, 20], [30, 40], [50, 60]])
    p = np.array([[10, 21], [70, 80]])
    m = bs.match_loops(t, p, tol=1)
    assert m == {"tp_true": 1, "tp_pred": 1, "n_true": 3, "n_pred": 2}
    assert bs.match_loops(t, np.zeros((0, 2), int), 1)["n_pred"] == 0
    assert bs.match_loops(np.zeros((0, 2), int), p, 1)["tp_pred"] == 0


def test_apa_pair_uses_only_anchors_valid_in_both_maps():
    n = 40
    A = np.ones((n, n))
    B = np.ones((n, n))
    B[5, 20] = np.nan
    anchors = np.array([[5, 20], [15, 30]])
    st, sp, cnt = bs.apa_pair(A, B, anchors, half=3)
    assert cnt == 1                       # the NaN anchor is dropped from both
    assert st.sum() == pytest.approx(sp.sum())


def test_apa_pair_drops_anchors_at_the_edge():
    A = np.ones((20, 20))
    st, sp, cnt = bs.apa_pair(A, A, np.array([[1, 18]]), half=3)
    assert cnt == 0


def test_apa_score_is_peak_over_corner():
    A = np.ones((11, 11))
    A[5, 5] = 3.0
    assert bs.apa_score(A, 1, corner=2) == pytest.approx(3.0)
    assert np.isnan(bs.apa_score(A, 0, corner=2))


# --------------------------------------------------------------------------- #
# region-level behaviour
# --------------------------------------------------------------------------- #
def test_perfect_prediction_scores_near_one(truth, params):
    r = bs.region_metrics(truth, truth, params)
    out = bs.pool([r], params)
    assert out["pearson_mean"] == pytest.approx(1.0)
    assert out["insulation_pearson"] == pytest.approx(1.0)
    assert out["boundary_f1"] == pytest.approx(1.0)
    assert out["loop_f1"] == pytest.approx(1.0)
    assert out["apa_ratio"] == pytest.approx(1.0)
    assert out["bioscore"] > 0.99


def test_a_different_region_scores_near_zero(truth, params):
    """The region-swap control: a map that looks like Hi-C, in the wrong place."""
    other = synthetic(seed=7, loops=((30, 190), (250, 350), (100, 330)),
                      edges=(0, 45, 175, 205, 290, 400, None))
    out = bs.pool([bs.region_metrics(truth, other, params)], params)
    assert out["bioscore"] < 0.5
    assert out["bioscore"] < 0.99
    assert abs(out["apa_ratio"]) < 0.6


def test_a_structureless_prediction_gets_zero_not_nan(truth, params):
    """The distance-only baseline is a flat map; it must land in the table."""
    flat = np.zeros_like(truth)
    out = bs.pool([bs.region_metrics(truth, flat, params)], params)
    assert out["insulation_pearson"] == 0.0        # not NaN
    assert out["boundary_f1"] == 0.0
    assert out["boundary_n_pred"] == 0
    assert np.isfinite(out["bioscore"])
    assert out["bioscore"] < 0.05


def test_blurring_degrades_the_score_monotonically(truth, params):
    from scipy.ndimage import gaussian_filter

    scores = [bs.pool([bs.region_metrics(truth, gaussian_filter(truth, s),
                                         params)], params)["bioscore"]
              for s in (0.5, 2.0, 6.0)]
    assert scores[0] > scores[1] > scores[2]


def test_components_are_reported_separately(truth, params):
    out = bs.pool([bs.region_metrics(truth, truth, params)], params)
    for key in ("insulation_pearson", "boundary_f1", "loop_f1", "apa_ratio",
                "anchor_spearman", "anchor_auroc", "tad_score", "loop_score",
                "bioscore", "loop_n_invented", "loopthr_n_pred"):
        assert key in out


def test_invented_loops_are_counted(truth, params):
    """A prediction with extra peaks where the measurement has none."""
    n = truth.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    extra = truth.copy()
    for i, j in ((40, 330), (170, 400)):
        blob = 2.0 * np.exp(-(((y - i) ** 2 + (x - j) ** 2) / (2 * 10.0 ** 2)))
        extra += blob + blob.T
    out = bs.pool([bs.region_metrics(truth, extra, params)], params)
    assert out["loopthr_n_pred"] > out["loop_n_true"]
    assert out["loopthr_n_invented"] > 0


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def test_pool_micro_averages_counts(truth, params):
    a = bs.region_metrics(truth, truth, params)
    b = bs.region_metrics(truth, np.zeros_like(truth), params, seed_offset=1)
    both = bs.pool([a, b], params)
    assert both["boundary_n_true"] == a.b_n_true + b.b_n_true
    assert both["loop_n_true"] == a.l_n_true + b.l_n_true
    assert both["n_regions"] == 2
    # a perfect and a hopeless region average to something in between
    assert 0.0 < both["bioscore"] < 1.0


def test_pool_of_nothing_is_empty():
    assert bs.pool([]) == {}


# --------------------------------------------------------------------------- #
# baselines and grid helpers
# --------------------------------------------------------------------------- #
def test_distance_baseline_is_toeplitz_and_matches_diagonal_means():
    T = np.stack([synthetic(64, seed=s) for s in range(4)])
    B = bs.distance_baseline(T, 64)
    assert np.allclose(np.diagonal(B, 5), np.diagonal(B, 5)[0])
    assert np.diagonal(B, 5)[0] == pytest.approx(
        np.diagonal(T, 5, axis1=1, axis2=2).mean())


def test_resample_square_is_the_identity_on_a_matched_grid():
    M = synthetic(64, seed=2)
    out = bs.resample_square(M, 0.0, 100.0, 0.0, 64, 100.0)
    assert np.allclose(out, M, atol=1e-6)


def test_resample_square_changes_resolution():
    M = synthetic(64, seed=2)
    out = bs.resample_square(M, 0.0, 100.0, 0.0, 32, 200.0)
    assert out.shape == (32, 32)
    assert np.isfinite(out).all()


def test_params_round_trip():
    p = bs.BioParams(loop_coarsen=4)
    assert p.as_dict()["loop_coarsen"] == 4
    assert bs.BioParams(**p.as_dict()) == p


# --------------------------------------------------------------------------- #
# the down-sampling experiment's own helpers (Part 3)
# --------------------------------------------------------------------------- #
def test_block_sum_is_exact_for_counts():
    from scripts.bioscore_depth import block_sum

    C = np.arange(36, dtype=float).reshape(6, 6)
    B = block_sum(C, 3)
    assert B.shape == (2, 2)
    assert B.sum() == pytest.approx(C.sum())
    assert B[0, 0] == pytest.approx(C[:3, :3].sum())


def test_ice_equalises_marginals():
    from scripts.bioscore_depth import ice

    rng = np.random.default_rng(0)
    n = 40
    bias = rng.uniform(0.4, 2.5, n)
    true = np.ones((n, n))
    C = true * bias[:, None] * bias[None, :]
    usable = np.ones(n, bool)
    M = ice(C, usable)
    s = np.nansum(M, axis=1)
    assert s.std() / s.mean() < 0.01


def test_ice_masks_unusable_bins():
    from scripts.bioscore_depth import ice

    C = np.ones((10, 10))
    usable = np.ones(10, bool)
    usable[3] = False
    M = ice(C, usable)
    assert np.isnan(M[3]).all()
    assert np.isnan(M[:, 3]).all()
    assert np.isfinite(M[0, 1])


def test_destripe_removes_the_per_diagonal_mean():
    from scripts.bioscore_depth import destripe

    n = 30
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    M = 5.0 - 0.1 * d + 0.01 * np.random.default_rng(0).standard_normal((n, n))
    M = 0.5 * (M + M.T)
    out = destripe(M)
    for k in range(1, 20):
        assert np.diagonal(out, k).mean() == pytest.approx(0.0, abs=1e-9)


def test_obs_over_expected_centres_each_diagonal_on_one():
    from scripts.bioscore_depth import obs_over_expected

    n = 25
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    bal = np.exp(-0.05 * d)
    oe = obs_over_expected(bal)
    for k in range(0, 15):
        assert np.diagonal(oe, k).mean() == pytest.approx(1.0)


def test_smooth_nan_ignores_missing_pixels():
    from scripts.bioscore_depth import _smooth_nan

    M = np.ones((20, 20))
    M[10, 10] = np.nan
    out = _smooth_nan(M, 1.0)
    assert np.isnan(out[10, 10])
    assert out[5, 5] == pytest.approx(1.0, abs=1e-6)
    assert out[9, 10] == pytest.approx(1.0, abs=1e-6)
