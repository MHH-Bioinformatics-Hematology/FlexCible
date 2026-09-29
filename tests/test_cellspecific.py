"""Cell-specific structure helpers: loop and boundary strength, compartment
flip score, difference-map correlation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
cs = pytest.importorskip("cellspecific")


def _loop_map(n=60, i=20, j=40, amp=1.0):
    L = np.zeros((n, n))
    ii, jj = np.indices((n, n))
    L += amp * np.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (2 * 1.5 ** 2))
    return np.maximum(L, L.T)


def test_loop_strength_detects_a_dot_and_ignores_flat_maps():
    assert cs.loop_strength(_loop_map(), 20, 40) > 0.5
    assert abs(cs.loop_strength(np.zeros((60, 60)), 20, 40)) < 1e-9
    # too close to the edge for the donut ring
    assert np.isnan(cs.loop_strength(_loop_map(), 3, 40))


def test_boundary_strength_and_calls():
    n = 80
    L = np.full((n, n), -0.5)
    L[:40, :40] = 0.5                               # two domains, boundary at bin 40
    L[40:, 40:] = 0.5
    prof = cs.insulation(L)
    b = int(np.nanargmin(prof))
    assert abs(b - 40) <= 1
    assert cs.boundary_strength(prof, b) > 0.3
    calls = cs.boundary_calls(prof, 0.2)
    assert len(calls) and np.min(np.abs(calls - 40)) <= 1
    flat = cs.insulation(np.zeros((n, n)))
    assert len(cs.boundary_calls(flat, 0.2)) == 0


def test_flip_score_sign_follows_compartment_of_flipping_bins():
    n = 40
    comp_a = np.array([1] * 10 + [1] * 10 + [-1] * 20)      # bins 10-19 A in cell A
    comp_b = np.array([1] * 10 + [-1] * 10 + [-1] * 20)     # and B in cell B
    Ma = 0.5 * np.outer(comp_a, comp_a)
    Mb = 0.5 * np.outer(comp_b, comp_b)
    flip = (comp_a > 0) & (comp_b < 0)
    sign = np.where((comp_a > 0) & (comp_b > 0), 1, np.where((comp_a < 0) & (comp_b < 0), -1, 0))
    assert cs.flip_score(Ma, flip, sign) > 0
    assert cs.flip_score(Mb, flip, sign) < 0


def test_diff_scores_perfect_and_flat_predictions():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(50, 50)), rng.normal(size=(50, 50))
    A, B = A + A.T, B + B.T
    flat = np.zeros_like(A)
    r = cs.diff_scores(A, B, {"good": (A, B), "swap": (B, A), "flat": (flat, flat + 1e-9)})
    assert r["r__good"] > 0.999 and r["r__swap"] < -0.999
    assert r["sd__flat"] < 1e-6


def test_pair_kind_and_row_order():
    assert cs.pair_kind("k562", "gm12878") == "train-heldout"
    assert cs.pair_kind("u937_ko", "u937") == "u937_wt_vs_ko"
    assert cs.pair_kind("molm13", "hmec_deep") == "heldout-aml"
    rows = cs.row_cells("loop", "a", "b", {"a": 1, "b": 0, "c": 0.9, "d": -0.2, "e": 0.5})
    assert rows[:2] == ["a", "b"] and rows[2] == "c" and rows[3] == "d"
