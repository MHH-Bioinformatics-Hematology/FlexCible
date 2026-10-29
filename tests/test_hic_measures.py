"""scripts/hic_measures.py: pure functions of the multi-measure evaluation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hic_measures as hm  # noqa: E402


def _band(n=120, D=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, D + 1))


def test_band_round_trip_through_dense():
    B = _band()
    n, Dp = B.shape
    for d in range(Dp):                     # pixels past the chromosome end
        B[n - d:, d] = np.nan
    M = hm.band_to_dense(B, 0, n)
    i, j = np.triu_indices(n)
    k = (j - i) < Dp
    B2 = hm.band_from_pixels(i[k], j[k], M[i[k], j[k]], n, Dp - 1)
    assert np.allclose(B, B2, equal_nan=True)
    assert np.allclose(M, M.T, equal_nan=True)
    assert np.isnan(M[0, Dp + 1])


def test_counts_conversion_inverts():
    B = _band(D=10)
    exp_ = np.linspace(100, 1, 11)
    off = np.linspace(-0.2, 0.1, 11)
    C = hm.to_counts(B, exp_, off, 3.0)
    assert np.allclose(hm.to_oe(C, exp_, off, 3.0), B)


def test_hicgan_pearson_auc_matches_pandas_definition():
    """Reference: Hi-cGAN computePearsonCorrelationSparse + getCorrelation
    (pandas groupby corr per distance, NaN dropped, AUC over distance /
    max distance)."""
    from sklearn import metrics
    rng = np.random.default_rng(1)
    res, n = 100_000, 60
    K = int(np.round(1_000_000 / res))
    T = rng.gamma(2.0, 10.0, size=(n, K + 3))
    P = T + rng.normal(scale=8.0, size=T.shape)
    P[:, 4] = 5.0                                  # constant diagonal -> NaN, dropped
    rows = []
    for d in range(K):
        for i in range(n - d):
            rows.append((d, T[i, d], P[i, d]))
    df = pd.DataFrame(rows, columns=["distance", "reads1", "reads2"])
    cm = df.groupby("distance")[["reads1", "reads2"]].corr(method="pearson")
    vals = cm.iloc[0::2, -1].reset_index(drop=True).dropna()
    x = np.array(vals.index, dtype=np.int32) / df.distance.max()
    ref = metrics.auc(x, vals.values)
    assert abs(hm.hicgan_pearson_auc(T, P, res) - ref) < 1e-9


def test_identical_maps_score_one():
    B = _band(n=200, D=30)
    assert abs(hm.band_pearson(B, B) - 1) < 1e-12
    assert abs(hm.band_scc(B, B) - 1) < 1e-12
    tp = hm.tile_pearsons(B, B, 40)
    assert tp.size == 5 and np.allclose(tp, 1)
    r = hm.per_distance_pearson(B, B)
    assert np.allclose(r, 1)
    assert abs(hm.auc01(r) - 1) < 1e-12


def test_insulation_dips_at_boundary():
    n, D, w = 100, 12, 3
    dom = np.repeat([0, 1], 50)
    B = np.zeros((n, D + 1))
    for d in range(D + 1):
        i = np.arange(n - d)
        B[i, d] = np.where(dom[i] == dom[i + d], 1.0, -1.0)
    prof = hm.insulation_band(B, w)
    assert np.nanargmin(prof) in (49, 50)


def test_match_points_and_loops():
    hr, hp = hm.match_points(np.array([10, 50, 90]), np.array([11, 70]), 1)
    assert (hr, hp) == (1, 1)
    ref = np.array([[10, 20], [30, 45]])
    pred = np.array([[11, 21], [60, 70], [30, 47]])
    assert hm.match_loops(ref, pred, 1) == (1, 1)


def test_compartment_scores_follow_labels():
    n, D = 200, 30
    lab = np.sign(np.sin(np.arange(n) / 9.0) + 1e-9)
    B = np.zeros((n, D + 1))
    for d in range(D + 1):
        i = np.arange(n - d)
        B[i, d] = 0.5 * lab[i] * lab[i + d]
    cp = hm.contact_preference(B, lab, 2)
    ok = np.isfinite(cp)
    assert np.mean(np.sign(cp[ok]) == lab[ok]) > 0.95
    e1 = lab * (1 + 0.1 * np.random.default_rng(0).random(n))
    assert hm.saddle_strength(B, e1, 2) > 1.5
    flat = np.zeros_like(B)
    assert abs(hm.saddle_strength(flat, e1, 2) - 1) < 1e-9


def test_parse_subset_handles_tad_names():
    f = ("best_model_gradientBoosting_QuASAR-Rep HiC-Spector GenomeDISCO Pearson TAD Score MSE "
         "TAD Fraction TAD Fraction Exact Match_degree_1.pkl")
    assert hm.parse_subset(f) == ["QuASAR-Rep", "HiC-Spector", "GenomeDISCO", "Pearson",
                                  "TAD Score MSE", "TAD Fraction", "TAD Fraction Exact Match"]
    g = "best_model_random forest_GenomeDISCO TAD Fraction HiCRep_degree_1.pkl"
    assert hm.parse_subset(g) == ["GenomeDISCO", "TAD Fraction", "HiCRep"]


def test_blocks_are_unique_and_cover_sets():
    b = hm.blocks()
    ids = [x["id"] for x in b]
    assert len(ids) == len(set(ids))
    assert {x["set"] for x in b} >= {"heldout", "trainchrom", "trainstretch"}


def _windows(k=6, n=80, seed=0):
    """Smooth synthetic log O/E windows with domains, loops and a checkerboard."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        i = np.arange(n)
        dom = np.cumsum(rng.random(n) < 0.08)
        comp = np.sign(np.sin(i / rng.uniform(6, 12)))
        T = 0.6 * (dom[:, None] == dom[None, :]) + 0.3 * np.outer(comp, comp)
        for _ in range(4):
            a, b = sorted(rng.integers(0, n, 2))
            if b - a > 6:
                T[a, b] = T[b, a] = T[a, b] + 2.0
        T = T + 0.05 * rng.normal(size=(n, n))
        T = 0.5 * (T + T.T)
        out.append(T)
    return out


def test_combined_score_orders_predictions():
    T = _windows()
    rng = np.random.default_rng(3)
    perfect = [(t, t) for t in T]
    noisy = [(t + 0.4 * rng.normal(size=t.shape), t) for t in T]
    noisy = [(0.5 * (p + p.T), t) for p, t in noisy]
    flat = [(np.zeros_like(t) + 0.01 * rng.normal(size=t.shape), t) for t in T]
    s1 = hm.combined_score(perfect, 25000)
    s2 = hm.combined_score(noisy, 25000)
    s3 = hm.combined_score(flat, 25000)
    assert s1 > s2 > s3
    assert 0.95 < s1 <= 1.0 + 1e-9
    # affine encoding of the maps does not change the score
    enc = [(2.0 * p + 1.0, 2.0 * t + 1.0) for p, t in noisy]
    assert abs(hm.combined_score(enc, 25000) - s2) < 0.05
    parts = hm.combined_parts(noisy, 25000)
    for keys in hm.COMBINED_GROUPS.values():
        for k in keys:
            assert k in parts


def test_combined_score_handles_nan_pixels():
    T = _windows(k=3)
    pairs = []
    for t in T:
        t = t.copy()
        t[:5, :] = np.nan
        t[:, :5] = np.nan
        pairs.append((t + 0.0, t))
    assert np.isfinite(hm.combined_score(pairs, 10000))


def test_resample_band_identity_and_shift():
    rng = np.random.default_rng(0)
    n, D = 50, 8
    R = rng.normal(size=(n, D + 1))
    for d in range(D + 1):
        R[n - d:, d] = np.nan
    same = hm.resample_band(R, 25000, n, D, 25000)
    ok = np.isfinite(R)
    assert np.allclose(same[ok], R[ok])
    # a smooth map sampled on a 24 kb grid reproduces the 25 kb values closely
    f = lambda a, b: np.sin(a / 7e5) + np.cos(b / 9e5)
    def band(res, n, D):
        B = np.full((n, D + 1), np.nan)
        for d in range(D + 1):
            i = np.arange(n - d)
            B[i, d] = f((i + 0.5) * res, (i + d + 0.5) * res)
        return B
    R24 = band(24000, 220, 12)
    B25 = band(25000, 200, 10)
    got = hm.resample_band(R24, 24000, 200, 10, 25000)
    m = np.isfinite(got) & np.isfinite(B25)
    assert m.sum() > 1500
    assert np.abs(got[m] - B25[m]).max() < 0.02


def test_rank_score_prefers_better_models():
    rows = []
    for k, m in enumerate(["all12honest", "all12refres", "all12refin"]):
        q = 0.5 + 0.2 * k                   # all12refin best on every measure
        r = dict(block="b", set="heldout", source=m, apa_ratio=q, saddle_ratio=q,
                 **{"TAD Score MSE": 1 - q, "TAD Fraction": 1 + (1 - q)})
        for lv, ms in hm.MMRS_LEVELS.items():
            for x in ms:
                r.setdefault(x, q)
        rows.append(r)
    rows.append(dict(block="b", set="heldout", source="consensus", oe_scc=0.99))
    d = hm.add_rank_score(pd.DataFrame(rows))
    s = d.set_index("source").mmrs
    assert s["all12refin"] == 1.0 and s["all12honest"] == 0.0
    assert np.isnan(s["consensus"])            # baselines are not ranked
    o = hm.oriented(pd.DataFrame({"apa_ratio": [0.5, 2.0, np.nan]}), "apa_agreement")
    assert np.allclose(o.values[:2], [0.5, 0.5]) and np.isnan(o.values[2])


def test_noref_variant_shares_the_checkpoint_of_the_one_model(tmp_path, monkeypatch):
    """all12one_chrho_noref reads the run folder and config of all12one_chrho, predicts with
    --no-reference, and the --chrho-models blocks carry their own ids."""
    monkeypatch.setattr(hm, "MR", tmp_path)
    (tmp_path / "all12one_chrho_10kb" / "checkpoint-best").mkdir(parents=True)
    (tmp_path / "all12one_chrho_10kb" / "checkpoint-best" / "model.pt").write_bytes(b"x")
    (tmp_path / "all12one_chrho_10kb.yaml").write_text("{}")
    assert hm.run_dir("all12one_chrho_noref", 10000) == hm.run_dir("all12one_chrho", 10000)
    assert hm.model_available("all12one_chrho_noref", 10000)
    assert not hm.model_available("all12one_chrho_noref", 25000)
    assert "all12one_chrho_noref" in hm.NO_REFERENCE and "all12one_chrho" not in hm.NO_REFERENCE
    plain = {b["id"] for b in hm.chrho_blocks()}
    monkeypatch.setattr(hm, "CHRHO_ONE", True)
    one = {b["id"] for b in hm.chrho_blocks()}
    assert one == {i + "__onemodel" for i in plain} and not (one & plain)
