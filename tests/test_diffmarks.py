"""Differential mark channels and all-cell objects (registry section 66).

Channel construction on a tiny synthetic genome (reference = mean of the others, the sample itself never
in its reference, shapes, scales, per-sample standardisation, missing marks), the cache layout and the
normaliser pass-through, track-dropout semantics, the regression that a difference channel reaches the
network (trained model, select_tracks in force), the all-cell objects table (definitions on synthetic
maps, leakage), run configs, and a 60-step CPU smoke with every flag."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hicmultiome import diffmarks as dm  # noqa: E402
from hicmultiome.config import Config  # noqa: E402
from hicmultiome.tracks import TrackNormalizer  # noqa: E402
from test_pipeline import BINSIZE, CHROMS, workdir  # noqa: E402,F401  (fixture)

MARKS = ["H3K27ac", "H3K4me1"]
SAMPLES = ["s0", "s1", "s2", "s3"]


# --------------------------------------------------------------------------- fixtures
def _write_bw(path: Path, vals: dict) -> None:
    import pyBigWig

    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(c, ln) for c, ln in CHROMS.items()])
    for chrom, ln in CHROMS.items():
        nb = ln // BINSIZE
        bw.addEntries([chrom] * nb, [i * BINSIZE for i in range(nb)],
                      ends=[(i + 1) * BINSIZE for i in range(nb)], values=[float(v) for v in vals[chrom]])
    bw.close()


def _tracks(seed: int, depth: float = 1.0) -> dict:
    """Per mark and chromosome a positive signal: noise plus a few planted peaks, scaled by depth."""
    rng = np.random.default_rng(seed)
    out = {}
    for mi, m in enumerate(MARKS):
        out[m] = {}
        for chrom, ln in CHROMS.items():
            nb = ln // BINSIZE
            v = rng.gamma(2.0, 0.5, size=nb)
            for p in rng.choice(nb, 6, replace=False):
                v[max(0, p - 1):p + 2] += rng.uniform(5, 20)
            out[m][chrom] = v * depth
    return out


@pytest.fixture(scope="module")
def world(workdir, tmp_path_factory):
    root = tmp_path_factory.mktemp("diffworld")
    dirs = {}
    for k, s in enumerate(SAMPLES):
        d = root / s
        d.mkdir()
        t = _tracks(100 + k, depth=4.0 if s == "s3" else 1.0)        # s3: a deep sample
        for m in MARKS:
            _write_bw(d / f"{m}.bigwig", t[m])
        dirs[s] = d
    return {"root": root, "dirs": dirs, "work": workdir}


def raw_cfg(world, name="a", dirs=None, diff=True, missing=None, extra_train=None, extra_data=None,
            reference=False, scales=(0, 2000, 6000), samples=SAMPLES):
    dirs = dirs or world["dirs"]
    missing = missing or {}
    w = world["work"]
    raw = {
        "data": {
            "samples": [{"name": s, "cool_path": str(w / "target.cool"), "bigwig_dir": str(dirs[s]),
                         "missing_tracks": missing.get(s, []),
                         **({"reference_cool": str(w / "target.cool")} if reference else {})}
                        for s in samples],
            "tracks": [{"name": m, "type": "histone"} for m in MARKS],
            "resolution": BINSIZE,
            **({"diff_marks": True, "diff_scales": list(scales),
                "diff_reference_sets": {"consensus": list(samples)}} if diff else {}),
            **(extra_data or {}),
        },
        "window": {"size": 32, "stride": 4, "max_zero_fraction": 0.99,
                   "transform": "prescaled" if reference else "obs_exp"},
        "model": {"base_channels": 16, "depth": 2, **({"reference": "residual"} if reference else {})},
        "train": {"output_dir": str(world["root"] / f"run_{name}"), "cache_dir": str(world["root"] / f"cache_{name}"),
                  "batch_size": 4, "max_steps": 60, "eval_every": 30, "val_regions": 1,
                  "val_exclude_overlap": False, "lr": 1e-3, "augment_reverse": True, **(extra_train or {})},
    }
    return raw


def cfg_of(world, **kw) -> Config:
    return Config.from_dict(raw_cfg(world, **kw))


# --------------------------------------------------------------------------- reference sets
def test_reference_members_exclude_self_and_same_cell():
    pool = ["gm12878", "imr90", "hmec", "k562"]
    assert dm.reference_members(pool, "gm12878") == ["imr90", "hmec", "k562"]
    assert dm.reference_members(pool, "hmec_deep") == ["gm12878", "imr90", "k562"]     # same cell identity
    assert dm.reference_members(pool, "huvec") == pool                                  # held-out cell: all of them


def test_set_keys_and_sets():
    assert dm.set_key("mean_aml", "cell") == "consensus" and dm.set_key("hspc", "cell") == "consensus"
    assert dm.set_key("mean_aml", "aml") == "mean_aml" and dm.set_key("hspc", "aml") == "hspc"
    assert dm.set_key("consensus", "aml") == "consensus" and dm.set_key("mean_aml", "hspc") == "hspc"
    with pytest.raises(ValueError):
        dm.set_key("bogus", "aml")
    s = dm.build_sets(["a", "b"], ["p1", "p2"], ["h1"])
    assert s == {"consensus": ["a", "b"], "mean_aml": ["p1", "p2"], "hspc": ["h1"]}


# --------------------------------------------------------------------------- channel construction
def test_reference_is_mean_of_the_others(world):
    cfg = cfg_of(world, name="ref")
    inst = dm.DiffMarks.get(cfg)
    spec = {s.name: s for s in cfg.data.samples}
    z = {n: inst.sample_marks(spec[n]).z["chr1"] for n in SAMPLES}
    for me in SAMPLES:
        others = [n for n in SAMPLES if n != me]
        ref, cnt = inst.reference(spec[me], "chr1")
        assert np.allclose(ref, np.mean([z[n] for n in others], axis=0), atol=1e-6)
        assert (cnt == len(others)).all()
        rows = inst.rows(spec[me], "chr1", z[me].shape[1])
        names = inst.names
        for k, m in enumerate(MARKS):
            assert np.allclose(rows[names.index(f"dref:{m}")], ref[k], atol=1e-6)


def test_own_track_never_enters_its_reference(world, tmp_path):
    """Replace s0's tracks: the reference rows of s0 stay put, its difference rows and every other
    sample's reference rows move."""
    cfgA = cfg_of(world, name="leakA")
    d0 = tmp_path / "s0new"
    d0.mkdir()
    t = _tracks(999)
    for m in MARKS:
        _write_bw(d0 / f"{m}.bigwig", t[m])
    cfgB = cfg_of(world, name="leakB", dirs={**world["dirs"], "s0": d0})
    ia, ib = dm.DiffMarks.get(cfgA), dm.DiffMarks.get(cfgB)
    sa, sb = {s.name: s for s in cfgA.data.samples}, {s.name: s for s in cfgB.data.samples}
    nb = CHROMS["chr1"] // BINSIZE
    ra, rb = ia.rows(sa["s0"], "chr1", nb), ib.rows(sb["s0"], "chr1", nb)
    names = ia.names
    dref = [i for i, n in enumerate(names) if n.startswith("dref:")]
    diff = [i for i, n in enumerate(names) if n.startswith("diff:")]
    assert np.array_equal(ra[dref], rb[dref])                      # s0's reference does not see s0
    assert not np.allclose(ra[diff], rb[diff])
    r1a, r1b = ia.rows(sa["s1"], "chr1", nb), ib.rows(sb["s1"], "chr1", nb)
    assert not np.allclose(r1a[dref], r1b[dref])                   # s1's reference does contain s0


def test_shapes_scales_and_standardisation(world):
    cfg = cfg_of(world, name="shape")
    inst = dm.DiffMarks.get(cfg)
    assert inst.names == ["dref:H3K27ac", "dref:H3K4me1"] + [
        f"diff:{m}:s{r}" for m in MARKS for r in (0, 2000, 6000)]
    for s in cfg.data.samples:                                      # incl. the deep sample s3
        rows = np.concatenate([inst.rows(s, c, n // BINSIZE) for c, n in CHROMS.items()], axis=1)
        assert rows.shape == (len(inst.names), sum(n // BINSIZE for n in CHROMS.values()))
        for i, nm in enumerate(inst.names):
            if nm.startswith("diff:"):
                v = rows[i]
                assert abs(np.median(v)) < 0.15, (s.name, nm)
                assert 0.7 < 1.4826 * np.median(np.abs(v - np.median(v))) < 1.3, (s.name, nm)
                assert np.abs(v).max() <= dm.CLIP + 1e-6
    s0 = cfg.data.samples[0]
    r = inst.rows(s0, "chr1", 96)
    tv = [np.abs(np.diff(r[inst.names.index(f"diff:H3K27ac:s{k}")])).mean() for k in (0, 2000, 6000)]
    assert tv[0] > tv[1] > tv[2]                                   # wider scale = smoother


def test_radius_and_smoothing():
    assert dm.radius_bins(0, 10000) == 0 and dm.radius_bins(20000, 10000) == 2
    assert dm.radius_bins(100000, 25000) == 4 and dm.radius_bins(20000, 25000) == 1
    x = np.zeros(21)
    x[10] = 1.0
    y = dm.smooth_box(x, 2)
    assert np.allclose(y[8:13], 0.2) and y[7] == 0 and np.isclose(y.sum(), 1.0)
    assert dm.smooth_box(x, 0) is x


def test_missing_marks_give_zero_rows(world):
    cfg = cfg_of(world, name="miss", missing={"s1": ["H3K4me1"], "s2": ["H3K4me1"], "s3": ["H3K4me1"]})
    inst = dm.DiffMarks.get(cfg)
    spec = {s.name: s for s in cfg.data.samples}
    nb = 96
    # the sample lacks the mark: all its rows are 0; the other mark is untouched
    r1 = inst.rows(spec["s1"], "chr1", nb)
    for i, n in enumerate(inst.names):
        if dm.row_mark(n) == "H3K4me1":
            assert not r1[i].any(), n
        else:
            assert r1[i].any(), n
    # no member of the reference has the mark: s0 has it but its reference does not -> 0 as well
    r0 = inst.rows(spec["s0"], "chr1", nb)
    assert not any(r0[i].any() for i, n in enumerate(inst.names) if dm.row_mark(n) == "H3K4me1")
    assert any(r0[i].any() for i, n in enumerate(inst.names) if dm.row_mark(n) == "H3K27ac")
    # a member lacking the mark drops out of the mean, it does not count as zero signal
    cfg2 = cfg_of(world, name="miss2", missing={"s3": ["H3K27ac"]})
    i2 = dm.DiffMarks.get(cfg2)
    sp2 = {s.name: s for s in cfg2.data.samples}
    ref, cnt = i2.reference(sp2["s0"], "chr1")
    z = {n: i2.sample_marks(sp2[n]).z["chr1"] for n in ("s1", "s2")}
    assert cnt[0] == 2 and np.allclose(ref[0], (z["s1"][0] + z["s2"][0]) / 2, atol=1e-6)
    # zeroed tracks behave as missing ones
    cfg3 = cfg_of(world, name="zero")
    sp3 = cfg3.data.samples[1]
    sp3.zeroed_tracks = ["H3K27ac"]
    assert not dm.DiffMarks.get(cfg3).rows(sp3, "chr1", nb)[:1].any()


def test_unseen_cell_configuration(world):
    cfg = cfg_of(world, name="unseen", extra_data={"diff_default_set": "consensus"})
    spec = cfg.data.samples[0]
    import copy
    new = copy.copy(spec)
    dm.configure_unseen(cfg, new, "k562", None)
    assert new.diff_identity == "k562" and new.diff_set == "consensus"
    # a copy standing for an unseen cell reads ALL pool members, also the configured sample it is a copy of
    assert dm.DiffMarks.get(cfg).members(new) == SAMPLES
    assert dm.DiffMarks.get(cfg).members(spec) == SAMPLES[1:]


def test_config_validation(world):
    with pytest.raises(ValueError, match="diff_reference_sets"):
        Config.from_dict(raw_cfg(world, diff=False, extra_data={"diff_marks": True}))
    r = raw_cfg(world)
    r["data"]["diff_reference_sets"] = {"consensus": ["s0", "nope"]}
    with pytest.raises(ValueError, match="not samples"):
        Config.from_dict(r)
    r = raw_cfg(world)
    r["data"]["diff_reference"] = "bogus"
    with pytest.raises(ValueError, match="diff_reference"):
        Config.from_dict(r)
    r = raw_cfg(world)
    r["data"]["diff_reference_sets"] = {"consensus": ["s0"]}
    with pytest.raises(ValueError, match="empty reference set"):
        Config.from_dict(r)
    off = cfg_of(world, diff=False)                                # defaults off
    d = off.data
    assert not d.diff_marks and d.diff_reference_sets == {} and dm.diff_row_names(off) == []
    assert off.train.object_scope == "aml" and d.cell_objects_table is None


# --------------------------------------------------------------------------- normaliser
def test_track_normalizer_passthrough(tmp_path):
    rng = np.random.default_rng(0)
    arrs = [np.abs(rng.normal(size=(4, 50))) * 5 for _ in range(3)]
    plain = TrackNormalizer().fit(arrs)
    pt = TrackNormalizer(passthrough=[2, 3]).fit(arrs)
    x = rng.normal(size=(4, 20)).astype(np.float32)
    y = pt.apply(x)
    assert np.array_equal(y[2:], x[2:])                            # untouched, negative values included
    assert np.allclose(y[:2], plain.apply(x)[:2])
    assert np.allclose(pt.absent_values()[2:], 0.0)
    pt.save(tmp_path / "n.json")
    assert np.array_equal(TrackNormalizer.load(tmp_path / "n.json").apply(x), y)
    # a normaliser without pass-through writes the file it always wrote
    plain.save(tmp_path / "p.json")
    assert "passthrough" not in json.loads((tmp_path / "p.json").read_text())
    assert np.array_equal(TrackNormalizer.load(tmp_path / "p.json").apply(np.abs(x)), plain.apply(np.abs(x)))


# --------------------------------------------------------------------------- cache layout
def test_cache_layout_and_other_rows_unchanged(world):
    from hicmultiome import precompute as hp

    c_diff = cfg_of(world, name="lay_d", reference=True)
    c_off = cfg_of(world, name="lay_o", reference=True, diff=False)
    m_d, m_o = hp.build(c_diff), hp.build(c_off)
    names = dm.diff_row_names(c_diff)
    assert m_d["seq_features"] == names and m_o["seq_features"] == []
    assert m_d["sym_rows"] == MARKS + names and m_o["sym_rows"] == MARKS
    assert m_d["diff"]["members"]["s0"] == SAMPLES[1:]
    nt = TrackNormalizer.load(Path(c_diff.train.cache_dir) / "norm_tracks.json")
    assert nt.passthrough == list(range(len(MARKS), len(MARKS) + len(names)))
    # the shards: same windows, same targets, same ChIP rows; the diff rows are the DiffMarks rows
    inst = dm.DiffMarks.get(c_diff)
    spec = {s.name: s for s in c_diff.data.samples}
    for a, b in zip(m_d["shards"], m_o["shards"]):
        assert (a["sample"], a["chrom"], a["split"]) == (b["sample"], b["chrom"], b["split"])
        za, zb = np.load(a["path"]), np.load(b["path"])
        assert np.array_equal(za["target"], zb["target"]) and np.array_equal(za["start"], zb["start"])
        assert np.array_equal(za["cond"][:, :len(MARKS)], zb["cond"][:, :len(MARKS)])
        assert za["cond"].shape[1] == len(MARKS) + len(names)
        nb = int(za["nbins"][0])
        full = inst.rows(spec[a["sample"]], a["chrom"], nb)
        st = int(za["start"][0])
        assert np.allclose(za["cond"][0, len(MARKS):].astype(np.float32), full[:, st:st + 32], atol=2e-2)


def test_select_tracks_keeps_only_the_diff_rows_of_kept_marks(world):
    from hicmultiome import precompute as hp
    from hicmultiome.train import _select_rows, cond_row_map

    cfg = cfg_of(world, name="sel", extra_train={"select_tracks": ["H3K27ac"]})
    man = hp.build(cfg)
    rows, offset, n_motifs, n_sym = _select_rows(cfg, man)
    kept = [man["sym_rows"][r] for r in rows]
    assert "H3K27ac" in kept and "H3K4me1" not in kept
    assert all(dm.row_mark(n) in (None, "H3K27ac") for n in kept)
    assert "dref:H3K27ac" in kept and "diff:H3K27ac:s6000" in kept and "diff:H3K4me1:s0" not in kept
    assert n_sym == len(rows) == offset and n_motifs == 0
    assert cond_row_map(cfg, man) == {r: i for i, r in enumerate(rows)}


# --------------------------------------------------------------------------- dropout semantics
def test_track_dropout_blanks_reference_and_difference_rows(world):
    from hicmultiome import precompute as hp
    from hicmultiome.train import apply_diff_drop, diff_coupling

    cfg = cfg_of(world, name="drop")
    man = hp.build(cfg)
    chip_rows = [0, 1]
    rows, mat = diff_coupling(cfg, man, chip_rows)
    names = man["sym_rows"]
    assert [names[r] for r in rows] == dm.diff_row_names(cfg)
    owner = [[dm.row_mark(names[r]) for r in rows if mat[j, rows.index(r)]] for j in range(2)]
    assert set(owner[0]) == {"H3K27ac"} and set(owner[1]) == {"H3K4me1"}
    assert mat.sum() == len(rows)                                  # every diff row has exactly one owner
    cond = torch.randn(3, len(names), 8) + 5.0
    ref = cond.clone()
    drop = torch.tensor([[True, False], [False, False], [True, True]])
    apply_diff_drop(cond, drop, (torch.as_tensor(rows), torch.from_numpy(mat)))
    for b in range(3):
        for j, m in enumerate(MARKS):
            blank = drop[b, j].item()
            for r in rows:
                if dm.row_mark(names[r]) == m:
                    assert (cond[b, r] == 0).all() == blank, (b, names[r])
                    if not blank:
                        assert torch.equal(cond[b, r], ref[b, r])
        assert torch.equal(cond[b, :2], ref[b, :2])                # the ChIP rows are the loop's business
    # no diff rows in the cache: no coupling
    off = cfg_of(world, name="drop_off", diff=False)
    assert diff_coupling(off, hp.build(off), [0, 1]) is None


def test_blank_diff_rows_at_inference(world):
    cfg = cfg_of(world, name="blank")
    tracks, seqf = MARKS, dm.diff_row_names(cfg)
    sym = np.ones((len(tracks) + len(seqf), 5), np.float32)
    dm.blank_diff_rows(sym, cfg, tracks, seqf, "H3K4me1")
    for k, n in enumerate(seqf):
        assert (sym[len(tracks) + k] == 0).all() == (dm.row_mark(n) == "H3K4me1")
    assert sym[:2].all()


# --------------------------------------------------------------------------- reaches the network
def test_difference_channel_reaches_the_network(world):
    """Regression of the select_tracks lesson: after training with select_tracks in force, changing any
    difference or reference row the model reads changes its output; the rows of an unselected mark are not
    inputs at all."""
    from hicmultiome import precompute as hp
    from hicmultiome import train as htrain
    from hicmultiome.dataset import WindowDataset
    from hicmultiome.predict import build_model
    from hicmultiome.train import _select_rows

    cfg = cfg_of(world, name="reach", reference=True,
                 extra_train={"select_tracks": ["H3K27ac"], "track_dropout": 0.5,
                              "track_dropout_mode": "uniform_count"})
    hp.build(cfg)
    htrain.train(cfg)
    out = Path(cfg.train.output_dir)
    ck = out / (out / "best").read_text().strip()
    model, channels = build_model(cfg, ck, torch.device("cpu"))
    rows, offset, n_motifs, n_sym = _select_rows(cfg, channels)
    ds = WindowDataset(cfg.train.cache_dir, "train")
    item = ds[0]
    cond = item["cond"][rows][None]
    ref = item["ref"][None]
    names = [channels["sym_rows"][r] for r in rows]
    assert not any(n.startswith(("dref:H3K4me1", "diff:H3K4me1")) for n in names)

    def run(c):
        with torch.no_grad():
            return model(c, None, None, None, ref)
    base = run(cond)
    tested = 0
    for i, n in enumerate(names):
        if not dm.is_diff_row(n):
            continue
        c2 = cond.clone()
        c2[0, i] += 2.5
        assert (run(c2) - base).abs().max() > 1e-6, f"{n} does not reach the output"
        tested += 1
    assert tested == 1 + 3                                         # dref + three scales of the kept mark
    # and the difference rows carry data: they are not all zero in the cache
    assert any(item["cond"][len(MARKS) + k].abs().max() > 0 for k in range(4))


# --------------------------------------------------------------------------- objects
def _planted(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.05, size=(n, n))


def _sym(M):
    M = np.triu(M)
    return M + np.triu(M, 1).T


def test_loop_definitions_on_synthetic_maps():
    import cell_objects_table as cot

    n = 200
    Mc, Mr = _planted(n, 1), _planted(n, 2)
    for (i, j) in [(60, 110), (80, 150)]:
        Mc[i - 1:i + 2, j - 1:j + 2] += 1.2
    Mr[79:82, 149:152] += 1.2                                     # (80, 150) is in the reference as well
    Mc, Mr = _sym(Mc), _sym(Mr)
    P = cot.loop_peaks(cot.donut_log2(Mc), cot.donut_log2(Mr), 0, n, True)
    P = cot.classify_loops(P)
    spec = P[P.cls == "specific"]
    shared = P[P.cls == "shared"]
    assert any((abs(spec.i - 60) <= 1) & (abs(spec.j - 110) <= 1))
    assert any((abs(shared.i - 80) <= 1) & (abs(shared.j - 150) <= 1))
    assert not any((abs(spec.i - 80) <= 1) & (abs(spec.j - 150) <= 1))
    assert ((P.j - P.i) >= cot.MIN_BINS).all() and ((P.j - P.i) <= cot.MAX_BINS).all()


def test_stripe_definitions_on_synthetic_maps():
    import cell_objects_table as cot

    n = 200
    Mc, Mr = _planted(n, 3), _planted(n, 4)
    Mc[40:44, 44:64] += 0.7                                        # right stripe of anchor 40, cell only
    Mc[100:104, 104:124] += 0.7                                    # right stripe of anchor 100, also in the reference
    Mr[100:104, 104:124] += 0.7
    Mc[120:140, 140:144] += 0.7                                    # left stripe of anchor 140 (mirror), cell only
    Mc, Mr = _sym(Mc), _sym(Mr)
    rc, lc = cot.stripe_scores(Mc)
    rr, lr = cot.stripe_scores(Mr)
    assert rc[40] > 0.6 and rr[40] < 0.2 and lc[140] > 0.6
    cls = cot.classify_stripes(rc, rr)
    assert cls[40] == "specific" and cls[100] == "shared"
    assert cot.classify_stripes(lc, lr)[140] == "specific"
    assert np.isnan(rc[n - 5]) and np.isnan(lc[3])                 # rectangles leaving the window
    # peak-only: a neighbour of the maximum is not an object
    assert (cls[38:43] == "specific").sum() == 1


def test_match_controls_one_to_one_and_forbidden():
    import cell_objects_table as cot

    spec_f = np.column_stack([np.arange(5) * 10.0, np.arange(5) * 10.0])
    pool_f = np.vstack([spec_f + 0.01, np.full((5, 2), 1000.0)])
    spec_d = np.log10(np.array([50, 100, 200, 400, 800.0]))
    pool_d = np.concatenate([spec_d + 0.01, np.log10(np.full(5, 1300.0))])
    forb = np.zeros(10, bool)
    forb[0] = True
    pick = cot.match_controls(spec_f, pool_f, spec_d, pool_d, forb, np.random.default_rng(1))
    assert pick[0] == -1                                           # its only candidate is forbidden
    assert pick[1:] == [1, 2, 3, 4]
    assert len(set(p for p in pick if p >= 0)) == len([p for p in pick if p >= 0])
    assert (cot.near_any(np.arange(20), np.array([5, 12]), 2) == np.isin(np.arange(20), [3, 4, 5, 6, 7, 10, 11, 12, 13, 14])).all()


def _synthetic_calls(seed, extra=None):
    rng = np.random.default_rng(seed)
    loops = []
    for k in range(30):
        i = int(rng.integers(10, 400))
        loops.append({"chrom": "chr1", "i": i, "j": i + int(rng.integers(8, 100)),
                      "e_cell": 1.0 + rng.random(), "e_ref": 0.0 if k < 15 else 1.0})
    n = 600
    st = np.full((4, n), np.nan, np.float32)
    st[:, 20:560] = 0.0
    for a in (50, 150, 250):
        st[0, a] = 0.8                                             # specific right stripes (ref 0)
    for a in (350, 450):
        st[0, a] = 0.8
        st[2, a] = 0.8                                             # shared right stripes
    return {"loops": pd.DataFrame(loops), "stripes": {"chr1": st}}


def test_build_cell_table(tmp_path):
    import cell_objects_table as cot

    c = _synthetic_calls(0)
    others = {"o1": _synthetic_calls(1), "o2": _synthetic_calls(2)}
    d = cot.build_cell("gm12878", c, others, tmp_path)             # no ChIP files: distance-only matching
    pos, ctrl = d[d.role == "pos"], d[d.role == "ctrl"]
    assert set(d.cell) == {"gm12878"} and set(pos.set) == {"pos"} and set(ctrl.set) == {"A_ref"}
    assert (pos[pos.kind == "loop"].shape[0], pos[pos.kind == "stripe"].shape[0]) == (15, 3)
    assert ctrl.pair_id.isin(pos.pair_id).all() and not ctrl.pair_id.duplicated().any()
    assert pos.n_train_other.notna().all() and ctrl.n_train_other.isna().all()
    # controls are other places than the specific objects (not within two bins)
    for (kind, chrom), g in d.groupby(["kind", "chrom"]):
        a_pos = g[g.role == "pos"].a_start.to_numpy() // cot.RES
        a_ctl = g[g.role == "ctrl"].a_start.to_numpy() // cot.RES
        if kind == "stripe":
            assert not np.isin(a_ctl, a_pos).any()
    cot.assert_no_leak(d)
    # a stripe object geometry: anchor RA bins, extent EB bins adjacent to it
    s = pos[pos.kind == "stripe"].iloc[0]
    assert s.a_end - s.a_start == cot.RA * cot.RES and s.b_end - s.b_start == cot.EB * cot.RES
    assert s.b_start == s.a_end
    # depth cap
    cot.TOP_LOOPS, old = 5, cot.TOP_LOOPS
    try:
        d5 = cot.build_cell("gm12878", c, others, tmp_path)
    finally:
        cot.TOP_LOOPS = old
    assert (d5[(d5.kind == "loop") & (d5.role == "pos")].shape[0]) == 5


def test_objects_table_leakage_for_all_cells(tmp_path):
    import cell_objects_table as cot
    import multicell_expanded as mx

    from hicmultiome import aml_objects as ao

    c = _synthetic_calls(0)
    for cell in ("gm12878", "pc9"):
        cot.assert_no_leak(cot.build_cell(cell, c, {}, tmp_path))
    for held in ("k562", "hmec", "hmec_deep", "huvec_deep"):
        bad = cot.build_cell(held, c, {}, tmp_path)
        with pytest.raises(AssertionError):                        # a held-out cell is not a training table row
            cot.assert_no_leak(bad)
        cot.assert_no_leak(bad, heldout=True)                      # but the evaluation table is exactly those
        p = tmp_path / f"{held}.csv"
        bad.to_csv(p, index=False)
        with pytest.raises(ValueError, match="not training samples"):
            ao.load_objects(str(p), set(mx.TRAIN12))               # the trainer refuses it
    train = cot.build_cell("gm12878", c, {}, tmp_path)
    with pytest.raises(AssertionError):
        cot.assert_no_leak(train, heldout=True)
    real = [q for q in (REPO / "results/cell_objects/cell_objects.csv.gz",) if q.exists()]
    if real:
        d = pd.read_csv(real[0])
        cot.assert_no_leak(d)
        assert set(d.cell) <= set(mx.TRAIN12)
        assert set(d.cell) == set(mx.TRAIN12) and set(d["set"]) == {"pos", "A_ref"}
        pos_ids = set(zip(d[d.role == "pos"].cell, d[d.role == "pos"].kind, d[d.role == "pos"].pair_id))
        assert all(k in pos_ids for k in zip(d[d.role == "ctrl"].cell, d[d.role == "ctrl"].kind, d[d.role == "ctrl"].pair_id))
        assert len(ao.load_objects(str(real[0]), set(mx.TRAIN12), ("A_hspc", "A_ref"))) == len(d)
        h = REPO / "results/cell_objects/cell_objects_heldout.csv.gz"
        if h.exists():
            cot.assert_no_leak(pd.read_csv(h), heldout=True)
            with pytest.raises(ValueError, match="not training samples"):
                ao.load_objects(str(h), set(mx.TRAIN12))


def test_conversion_of_the_rule_transfer_objects(tmp_path):
    """crosscell_rule_transfer's cell objects become the training table (12 training cells, controls A_ref)
    and the evaluation table (held-out cells, deep maps renamed); n_train_other counts other training cells."""
    import cell_objects_table as cot
    import multicell_expanded as mx

    from hicmultiome import aml_objects as ao

    rows = []

    def add(cell, kind, st, pid, a, b, chrom="chr1"):
        rows.append({"cell": cell, "kind": kind, "set": st, "pair_id": pid, "y": int(st == "pos"), "chrom": chrom,
                     "a_lo": a[0], "a_hi": a[1], "b_lo": b[0], "b_hi": b[1], "e_cell": 1.0, "e_cons": 0.0, "gain": 1.0})
    for cell in ("gm12878", "imr90", "k562", "hmec", "huvec"):
        add(cell, "loop", "pos", f"{cell}:L0", (100, 101), (120, 121))
        add(cell, "loop", "A_ref", f"{cell}:L0", (300, 301), (320, 321))
        add(cell, "loop", "A_full", f"{cell}:L0", (400, 401), (420, 421))               # dropped
    add("gm12878", "loop", "pos", "gm12878:L1", (200, 201), (230, 231))                 # private
    add("imr90", "loop", "pos", "imr90:L1", (101, 102), (121, 122))                     # one bin off gm12878:L0
    add("gm12878", "stripe", "pos", "gm12878:S0", (50, 54), (100, 140))
    add("imr90", "stripe", "pos", "imr90:S0", (51, 55), (101, 141))
    add("imr90", "stripe", "pos", "imr90:S1", (80, 84), (10, 50))                       # left, private
    d = pd.DataFrame(rows)
    tr = cot.from_rule_transfer(d, list(mx.TRAIN12), heldout=False)
    assert set(tr.cell) == {"gm12878", "imr90"} and set(tr["set"]) == {"pos", "A_ref"}
    assert list(tr.columns) == cot.COLS and set(tr.role) == {"pos", "ctrl"}
    cot.assert_no_leak(tr)
    n = tr[tr.role == "pos"].set_index("pair_id").n_train_other
    assert n["gm12878:L0"] == 1 and n["gm12878:L1"] == 0 and n["gm12878:S0"] == 1 and n["imr90:S1"] == 0
    assert tr[tr.role == "ctrl"].n_train_other.isna().all()
    assert tr[tr.pair_id == "gm12878:S0"].a_start.tolist() == [50 * cot.RES]
    ev = cot.from_rule_transfer(d, list(mx.TRAIN12), heldout=True)
    assert set(ev.cell) == {"k562", "hmec_deep", "huvec_deep"}
    cot.assert_no_leak(ev, heldout=True)
    assert ev[ev.role == "pos"].n_train_other.eq(2).all()                               # both training cells carry (chr1, 100, 120)
    p = tmp_path / "t.csv"
    tr.to_csv(p, index=False)
    got = ao.load_objects(str(p), set(mx.TRAIN12), ("A_ref",))
    assert set(got["set"]) == {"pos", "A_ref"} and len(got) == len(tr)


def test_cell_objects_eval_on_synthetic_maps():
    """Window assignment, object scoring against synthetic maps and the AUROC summary of cell_objects_eval."""
    import cell_objects_eval as ev

    rng = np.random.default_rng(0)
    rows = []
    for k in range(40):
        for role, st in (("pos", "pos"), ("ctrl", "A_ref")):
            i = 20 + 3 * k
            rows.append({"cell": "k562", "kind": "loop", "role": role, "set": st, "pair_id": f"k562:L{k}",
                         "chrom": "chr1", "a_start": i * ev.RES, "a_end": (i + 1) * ev.RES,
                         "b_start": (i + (40 if role == "pos" else 55)) * ev.RES,
                         "b_end": (i + (41 if role == "pos" else 56)) * ev.RES,
                         "n_train_other": 0 if role == "pos" else np.nan})
    tab = pd.DataFrame(rows)
    sizes = {"chr1": 400 * ev.RES}
    items = ev.build_items(tab, sizes)
    assert items.win_bin.between(0, 400 - ev.WIN).all() and (items.cls.isin(["private", "control"])).all()
    assert items.win.str.startswith("chr1:").all()

    def maps_of(win):
        ws = int(win.split(":")[1].split("-")[0]) // ev.RES
        base = rng.normal(0, 0.02, size=(ev.WIN, ev.WIN))
        meas, model = base.copy(), base.copy()
        for r in items[items.win == win].itertuples():
            i, j = r.a_lo - ws, r.b_lo - ws
            if r.role == "pos":                                    # the model gets half of the measured gain right
                meas[i - 1:i + 2, j - 1:j + 2] += 1.0
                model[i - 1:i + 2, j - 1:j + 2] += 0.5
        sym = lambda M: np.triu(M) + np.triu(M, 1).T            # noqa: E731
        return {"measured": sym(meas), "reference": sym(base * 0), "m1": sym(model)}
    D = ev.object_scores(items, maps_of)
    assert len(D) == len(items)
    S = ev.summarise(D, ["m1"], nboot=50)
    row = S[(S.model == "m1") & (S.cell == "all") & (S["class"] == "all") & (S.kind == "loop")].iloc[0]
    assert row.auroc_gain > 0.9 and 0.2 < row.amplitude < 0.9
    meas = S[(S.model == "measured") & (S.cell == "all") & (S["class"] == "all")].iloc[0]
    assert meas.auroc_gain >= row.auroc_gain - 1e-9


def test_aml_leakage_still_holds_in_the_joint_table_set(tmp_path):
    """The AML table builder's safeguards are untouched: test patients and same-donor partners of the
    fold never enter, whatever the object scope."""
    import aml_objects_table as aot
    import aml_train as at

    for fold in (10, 11, 12):
        assert not set(aot.train_donors(fold)) & {x for t in at.test_patients(fold) for x in at.donor_group(t)}


def test_object_tables_and_scope_validation(world):
    from hicmultiome.train import object_tables

    cfg = cfg_of(world, name="scope", diff=False)
    assert object_tables(cfg) == []
    with pytest.raises(ValueError, match="cell_objects_table"):
        Config.from_dict(raw_cfg(world, diff=False, extra_train={"object_scope": "all"}))
    with pytest.raises(ValueError, match="object_scope 'all' only"):
        Config.from_dict(raw_cfg(world, diff=False, extra_data={"cell_objects_table": "x.csv"}))
    with pytest.raises(ValueError, match="aml' or 'all"):
        Config.from_dict(raw_cfg(world, diff=False, extra_train={"object_scope": "none"}))
    c = Config.from_dict(raw_cfg(world, diff=False, reference=True,
                                 extra_data={"cell_objects_table": "c.csv", "aml_objects_table": "a.csv"},
                                 extra_train={"object_scope": "all", "aml_object_weight": 1.0}))
    assert object_tables(c) == ["a.csv", "c.csv"]
    c = Config.from_dict(raw_cfg(world, diff=False, extra_data={"cell_objects_table": "c.csv"},
                                 extra_train={"object_scope": "all", "aml_object_oversample": 3}))
    assert object_tables(c) == ["c.csv"]                           # cell-only objects are allowed


# --------------------------------------------------------------------------- run configs
FOLDS = (10, 11, 12)


@pytest.mark.parametrize("fold", FOLDS)
@pytest.mark.parametrize("res", [10000, 25000])
def test_run_config_of_diff_loopfocus(fold, res):
    import aml_train as at
    import multicell_expanded as mx

    if not mx.HM.exists():
        pytest.skip("share not mounted")
    name = f"all12diff_loopfocus_refres_drop_f{fold}_{res // 1000}kb"
    cfg = mx.run_config(name, 100, 10, 500)
    ref = mx.run_config(f"all12aml_loopfocus_refres_drop_f{fold}_{res // 1000}kb", 100, 10, 500)
    d, t = cfg["data"], cfg["train"]
    assert d["diff_marks"] and d["diff_scales"] == [0, 20000, 100000] and d["diff_reference"] == "mean_aml"
    assert cfg["train"]["cache_dir"].endswith(f"mc_all12aml_ref_{res // 1000}kb_f{fold}_diff")
    assert t["object_scope"] == "all" and d["cell_objects_table"].endswith("cell_objects.csv.gz")
    assert d["aml_objects_table"] == ref["data"]["aml_objects_table"]
    # the joint composition of the all12aml runs is kept: same samples, references, oversampling, validation
    assert [s["name"] for s in d["samples"]] == [s["name"] for s in ref["data"]["samples"]]
    for a, b in zip(d["samples"], ref["data"]["samples"]):
        assert {k: v for k, v in a.items() if k != "diff_set"} == b
    for k in set(ref["train"]) - {"output_dir", "cache_dir", "aml_object_controls"}:
        assert t[k] == ref["train"][k], k
    assert t["aml_object_controls"] == ["A_hspc", "A_ref"]          # the AML table's and the cell table's controls
    assert cfg["model"] == ref["model"] and d["lab_classes"] == ["xu"]
    # reference sets: LOO-ready pools, nothing of the test patients or their partners
    sets = d["diff_reference_sets"]
    assert sets["consensus"] == list(mx.TRAIN12)
    assert sets["mean_aml"] == [at.guard(p) for p in at.train_patients(fold)]
    assert sets["hspc"] == [at.hspc_guard(x) for x in at.HSPC_TRAIN]
    test_side = {at.guard(x) for p in at.test_patients(fold) for x in at.donor_group(p)}
    assert not test_side & {n for v in sets.values() for n in v}
    assert not test_side & {s["name"] for s in d["samples"]}
    by = {s["name"]: s["diff_set"] for s in d["samples"]}
    assert all(by[c] == "consensus" for c in mx.TRAIN12)
    assert all(by[at.guard(p)] == "mean_aml" for p in at.train_patients(fold)) and by["molm13"] == "mean_aml"
    assert all(by[at.hspc_guard(x)] == "hspc" for x in at.HSPC_TRAIN)
    Config.from_dict(cfg)                                           # validates (sets, objects, scope)


@pytest.mark.parametrize("name", ["all12diff_refres_drop_f10_10kb", "all12diffmeth_loopfocus_refres_drop_f11_10kb",
                                  "all12diffhspc_loopfocus_refres_drop_f12_10kb", "all12allobj_refres_drop_f10_10kb",
                                  "all12diffcells_refres_drop_25kb", "all12diffcells_loopfocus_refres_drop_10kb"])
def test_run_config_variants(name):
    import multicell_expanded as mx

    if not mx.HM.exists():
        pytest.skip("share not mounted")
    cfg = mx.run_config(name, 100, 10, 500)
    d, t = cfg["data"], cfg["train"]
    Config.from_dict(cfg)
    if "allobj" in name:
        assert "diff_marks" not in d and t["object_scope"] == "all"
        assert cfg["train"]["cache_dir"].endswith("mc_all12aml_ref_10kb_f10")          # the existing AML reference cache
    if "hspc" in name:
        assert d["diff_reference"] == "hspc" and cfg["train"]["cache_dir"].endswith("_diffhspc")
        assert all(s["diff_set"] == "hspc" for s in d["samples"] if s["name"].startswith("amlxu"))
    if "meth" in name:
        assert any(x["name"] == "Meth" for x in d["tracks"]) and "Meth" in t["select_tracks"]
        assert "meth" in t["cache_dir"]
    if "diffcells" in name:
        import multicell_expanded as mx_
        assert [x["name"] for x in d["samples"]] == list(mx_.TRAIN12) and d["diff_marks"]
        assert set(d["diff_reference_sets"]) == {"consensus"} and not any("lab" in x for x in d["samples"])
        assert t["cache_dir"].endswith(f"mc_all12ref_diff_{name.split('_')[-1]}")
        if "loopfocus" in name:
            assert "aml_objects_table" not in d and d["cell_objects_table"].endswith("cell_objects.csv.gz")
            assert t["object_scope"] == "all" and t["aml_object_controls"] == ["A_ref"]
        else:
            assert "object_scope" not in t
    if name.startswith("all12diff_refres"):
        assert "object_scope" not in t and "cell_objects_table" not in d and "aml_object_oversample" not in t


def test_existing_run_configs_unchanged():
    """Runs that existed before section 66 get no diff / scope keys."""
    import multicell_expanded as mx

    if not mx.HM.exists():
        pytest.skip("share not mounted")
    for name in ("all12aml_loopfocus_refres_drop_f10_10kb", "all12aml_refres_drop_f11_25kb",
                 "all12refres_drop_10kb", "all12honest_drop_meth_10kb"):
        cfg = mx.run_config(name, 100, 10, 500)
        assert not any(k.startswith("diff_") for k in cfg["data"]), name
        assert "cell_objects_table" not in cfg["data"] and "object_scope" not in cfg["train"], name
        assert "diff_set" not in cfg["data"]["samples"][0]


# --------------------------------------------------------------------------- smoke
def _object_table(path, cell, shift=0):
    rows = []

    def add(kind, st, pid, a, b):
        rows.append({"patient": cell, "cell": cell, "kind": kind, "role": "pos" if st == "pos" else "ctrl",
                     "set": st, "pair_id": pid, "chrom": "chr1", "a_start": a[0] * BINSIZE, "a_end": a[1] * BINSIZE,
                     "b_start": b[0] * BINSIZE, "b_end": b[1] * BINSIZE})
    add("loop", "pos", "L0", (10 + shift, 11 + shift), (24 + shift, 25 + shift))
    add("loop", "A_hspc" if cell == "s0" else "A_ref", "L0", (40, 41), (54, 55))
    add("loop", "pos", "L1", (50, 51), (66, 67))
    add("loop", "A_hspc" if cell == "s0" else "A_ref", "L1", (14, 15), (30, 31))
    add("stripe", "pos", "S0", (30, 32), (40, 52))
    add("stripe", "A_hspc" if cell == "s0" else "A_ref", "S0", (60, 62), (70, 82))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_smoke_training_with_all_flags(world, tmp_path, caplog):
    from hicmultiome import precompute as hp
    from hicmultiome import train as htrain

    ta, tc = tmp_path / "aml.csv", tmp_path / "cells.csv"
    _object_table(ta, "s0")
    _object_table(tc, "s1", shift=3)
    obj = {"aml_object_oversample": 4, "aml_object_pair_batches": True, "aml_object_weight": 1.0,
           "aml_object_rank": 0.5, "aml_object_rank_consistent": False, "aml_object_donut": [1, 4],
           "aml_object_controls": ["A_hspc", "A_ref"], "object_scope": "all", "track_dropout": 0.5,
           "track_dropout_mode": "uniform_count"}
    cfg = cfg_of(world, name="smoke", reference=True, extra_train=obj,
                 extra_data={"aml_objects_table": str(ta), "cell_objects_table": str(tc)})
    hp.build(cfg)
    with caplog.at_level(logging.INFO):
        htrain.train(cfg)
    txt = caplog.text
    assert (Path(cfg.train.output_dir) / "best").exists()
    assert "diff marks: dropout of a track blanks its 8 reference/difference rows" in txt
    assert f"{ta},{tc}" in txt and "obj_pix" in txt and "obj_rank" in txt
    import re
    m = re.search(r"aml objects .*: (\d+) objects of (\d+) samples", txt)
    assert m and int(m.group(2)) == 2                              # the objects of both tables, two cells
    # the same run with the flags off: neither the diff coupling nor the objects code is entered
    caplog.clear()
    plain = cfg_of(world, name="smoke_off", reference=True, diff=False, extra_train={"track_dropout": 0.5})
    hp.build(plain)
    with caplog.at_level(logging.INFO):
        htrain.train(plain)
    assert "diff marks" not in caplog.text and "aml objects" not in caplog.text
