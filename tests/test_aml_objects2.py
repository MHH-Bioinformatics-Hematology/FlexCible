"""Loop-centric recipe version 2 (registry section 70): controls matched on the reference and HSPC enrichment
(A_refrand), weak shared-loop controls, the marks-dependence margin, run families loopfocus2."""
from __future__ import annotations

import re
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

from flexcible import aml_objects as ao  # noqa: E402
from flexcible.config import Config  # noqa: E402
from test_pipeline import BINSIZE, workdir  # noqa: E402,F401  (fixture)
from test_aml_objects import _smoke_raw  # noqa: E402


def auroc(a, b):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([a, b]))
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


# --------------------------------------------------------------------------- matching
def _synthetic_maps(n=600, seed=0):
    rng = np.random.default_rng(seed)
    Er = rng.normal(0.3, 0.4, (n, 151)).astype(np.float32)
    Eh = (0.5 * Er + rng.normal(0.2, 0.3, (n, 151))).astype(np.float32)
    strand = rng.choice([-1, 0, 1], n).astype(np.int8)
    score = np.where(strand == 0, 0.0, 0.8).astype(np.float32)
    return Er, Eh, strand, score


def test_match_loops_calipers_and_shortcut():
    import refrand_controls as rc
    Er, Eh, strand, score = _synthetic_maps()
    n = Er.shape[0]
    rng = np.random.default_rng(1)
    # AML-specific loops: weak reference and HSPC (the pattern that makes shared loops separable)
    cand = [(i, i + d) for i in range(20, 400, 7) for d in (20, 40, 80)
            if Er[i, d] < 0.2 and Eh[i, d] < 0.4]
    pos = pd.DataFrame(cand[:120], columns=["i", "j"])
    valid = np.ones(n, bool)
    valid[300:310] = False
    avoid = pos[["i", "j"]].to_numpy()
    res, rv, hv = rc.match_loops(pos, Er, Eh, strand, score, valid, avoid, rng)
    assert len(res) > 0.5 * len(pos)
    oc = rc.orient_codes(strand, score, pos.i.to_numpy(), pos.j.to_numpy())
    seen = set()
    from scipy.spatial import cKDTree
    tree = cKDTree(avoid.astype(float))
    for r, i, j in res:
        d0, d = pos.j[r] - pos.i[r], j - i
        assert abs(Er[i, d] - rv[r]) <= rc.CAL_ENR + 1e-6 and abs(Eh[i, d] - hv[r]) <= rc.CAL_ENR + 1e-6
        assert abs(np.log10(d / d0)) <= rc.CAL_DIST + 1e-9
        assert rc.orient_codes(strand, score, np.array([i]), np.array([j]))[0] == oc[r]
        assert valid[i] and valid[j]
        assert tree.query([i, j], p=np.inf)[0] > rc.NEAR
        assert (i, j) not in seen
        seen.add((i, j))
    # the matched controls cannot be separated by either enrichment; random controls could
    pr = np.array([rv[r] for r, _, _ in res])
    cr = np.array([Er[i, j - i] for _, i, j in res])
    ph = np.array([hv[r] for r, _, _ in res])
    ch = np.array([Eh[i, j - i] for _, i, j in res])
    assert abs(auroc(-pr, -cr) - 0.5) < 0.1 and abs(auroc(-ph, -ch) - 0.5) < 0.1
    anyc = np.array([Er[i, d] for i in range(20, 500, 5) for d in (20, 40, 80)])
    assert auroc(-pr, -anyc) > 0.7                       # an unmatched pool is separable


def test_match_stripes_geometry_and_values():
    import aml_loop_classifier as lc
    import refrand_controls as rc
    rng = np.random.default_rng(2)
    n = 700
    Dr = rng.normal(0.0, 0.3, (n, lc.NMAX + 1)).astype(np.float32)
    Dh = rng.normal(0.0, 0.3, (n, lc.NMAX + 1)).astype(np.float32)
    strand = rng.choice([-1, 0, 1], n).astype(np.int8)
    score = np.where(strand == 0, 0.0, 0.7).astype(np.float32)
    pos = pd.DataFrame({"a_lo": [100, 250, 400], "ra": [4, 4, 4], "eb": [20, 20, 20], "o": [10, -30, 40]})
    res, rv, hv = rc.match_stripes(pos, Dr, Dh, strand, score, None, [100, 250, 400], rng, lc.shift_scores)
    assert len(res) >= 2
    for r, a in res:
        q = pos.iloc[r]
        sr = lc.shift_scores(Dr, int(q.ra), int(q.eb), int(q.o))
        sh = lc.shift_scores(Dh, int(q.ra), int(q.eb), int(q.o))
        assert abs(sr[a] - rv[r]) <= rc.CAL_ENR and abs(sh[a] - hv[r]) <= rc.CAL_ENR
        assert abs(a - [100, 250, 400][r]) > rc.NEAR_ANCHOR


# --------------------------------------------------------------------------- roles and loss
def test_weak_controls_roles_and_geometry():
    d = pd.DataFrame({"cell": ["s"] * 3, "kind": ["loop"] * 3, "role": ["pos", "ctrl", "ctrl"],
                      "set": ["pos", "A_refrand", "A_hspc"], "pair_id": ["a", "a", "a"], "chrom": ["chr1"] * 3,
                      "a_start": [10000, 40000, 60000], "a_end": [11000, 41000, 61000],
                      "b_start": [24000, 55000, 75000], "b_end": [25000, 56000, 76000]})
    p = Path(REPO / "results") / "_tmp_objects_test.csv"
    d.to_csv(p, index=False)
    try:
        o = ao.load_objects(str(p), {"s"}, ("A_refrand",), ("A_hspc",))
    finally:
        p.unlink()
    assert o.role_id.tolist() == [1, 2, 3]
    rows = [[(0, 3, 5, 6, 15, 16, 2)], [(0, 1, 5, 6, 15, 16, 0)], []]
    assert ao.row_roles(rows).tolist() == [0, 1, 0]            # weak-only rows are not sampled
    geo = ao.batch_geometry([[(0, 3, 10, 11, 20, 21, 0)]], None, 32, 4, "cpu", weak_weight=0.2)
    assert geo.mask.max().item() == pytest.approx(0.2) and not geo.has_pairs()
    geo = ao.batch_geometry([[(0, 1, 10, 11, 20, 21, 0), (0, 2, 12, 13, 22, 23, 1)]], None, 32, 4, "cpu")
    assert geo.has_pairs()


def _pair_batch(W=32):
    T = torch.zeros(2, W, W)
    T[0, 10, 20] = 0.6
    T[1, 12, 22] = 0.1
    T = torch.maximum(T, T.transpose(1, 2))
    ref = torch.zeros(2, W, W)
    objs = [[(0, 1, 10, 11, 20, 21, 0)], [(0, 2, 12, 13, 22, 23, 1)]]
    return T, ref, ao.batch_geometry(objs, None, W, 4, "cpu")


def test_weak_objects_never_enter_the_ranking():
    T, ref, geo = _pair_batch()
    W = 32
    geo_w = ao.batch_geometry([[(0, 1, 10, 11, 20, 21, 0)], [(0, 3, 12, 13, 22, 23, 1)]], None, W, 4, "cpu")
    pred = torch.zeros(2, W, W)
    assert ao.ObjectLoss(rank=1.0)(pred, T, ref, geo_w).item() == 0.0       # no control: no pair
    assert ao.ObjectLoss(rank=1.0)(pred, T, ref, geo).item() > 0


def test_marks_margin_penalises_a_shortcut_and_passes_a_marks_gain():
    T, ref, geo = _pair_batch()
    W = 32
    loss = ao.ObjectLoss(rank=0.0, marks_margin=1.0, delta=0.02, temp=0.1, consistent=False)
    with_m = torch.zeros(2, W, W)
    with_m[0, 10, 20] = 0.3           # positive above control in the pass with marks
    with_m = torch.maximum(with_m, with_m.transpose(1, 2))
    shortcut = with_m.clone()          # the same margin without marks: reference / HSPC alone does it
    v1 = loss(with_m.clone().requires_grad_(True), T, ref, geo, shortcut).item()
    gain = torch.zeros(2, W, W)
    gain[0, 10, 20] = 0.05
    gain = torch.maximum(gain, gain.transpose(1, 2))
    v2 = loss(with_m.clone(), T, ref, geo, gain).item()
    assert v1 > 0.0 and v2 == 0.0                              # marks add 0.25 over the no-marks margin
    assert loss.last["obj_gap"] == pytest.approx(0.25, abs=1e-5)
    w = with_m.clone().requires_grad_(True)
    loss(w, T, ref, geo, shortcut).backward()
    assert w.grad[0, 10, 20] < 0 and w.grad[1, 12, 22] > 0   # raises the AML-specific object, lowers the control
    # without the margin weight the gap is only reported
    diag = ao.ObjectLoss(marks_margin=0.0)
    assert diag(with_m, T, ref, geo, shortcut).item() == 0.0 and diag.last["obj_gap"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- config and run families
def test_config_marks_pass_needs_track_dropout(workdir):
    tab = workdir / "t.csv"
    tab.write_text("x\n")
    with pytest.raises(ValueError, match="track_dropout"):
        Config.from_dict(_smoke_raw(workdir, "m", table=tab, aml_object_marks_pass=True))
    with pytest.raises(ValueError, match="marks_pass"):
        Config.from_dict(_smoke_raw(workdir, "m", table=tab, aml_object_marks_margin=0.5,
                                    track_dropout=0.5, track_dropout_mode="uniform_count"))
    c = Config.from_dict(_smoke_raw(workdir, "m", table=tab, aml_object_marks_pass=True, aml_object_marks_margin=0.5,
                                    track_dropout=0.5, track_dropout_mode="uniform_count"))
    assert c.train.aml_object_marks_margin == 0.5 and c.train.aml_object_weak_controls == []


@pytest.mark.parametrize("fold", [10, 11, 12])
@pytest.mark.parametrize("res", [10000, 25000])
def test_run_config_of_loopfocus2(fold, res):
    import multicell_expanded as mx
    if not mx.HM.exists():
        pytest.skip("share not mounted")
    k = res // 1000
    cfg = mx.run_config(f"all12aml_loopfocus2_refres_drop_f{fold}_{k}kb", 100, 10, 500)
    old = mx.run_config(f"all12aml_loopfocus_refres_drop_f{fold}_{k}kb", 100, 10, 500)
    t = cfg["train"]
    assert cfg["data"]["aml_objects_table"].endswith(f"objects2_f{fold}.csv.gz")
    assert t["aml_object_controls"] == ["A_refrand"] and "A_hspc" in t["aml_object_weak_controls"]
    assert t["aml_object_marks_pass"] and t["aml_object_marks_margin"] > 0 and t["track_dropout"] > 0
    assert t["aml_object_oversample"] == old["train"]["aml_object_oversample"]
    assert cfg["train"]["cache_dir"] == old["train"]["cache_dir"]
    assert old["train"]["aml_object_controls"] == ["A_hspc"] and "aml_object_marks_pass" not in old["train"]
    assert cfg["model"]["reference"] == "residual"
    from flexcible.config import Config as C
    C.from_dict(cfg)


@pytest.mark.parametrize("kind", ["diff", "diffmeth", "diffhspc"])
def test_diff_loopfocus2_variants(kind):
    import multicell_expanded as mx
    if not mx.HM.exists():
        pytest.skip("share not mounted")
    for suffix, scope_all in (("_loopfocus2", True), ("_loopfocus2aml", False)):
        name = f"all12{kind}{suffix}_refres_drop_f10_10kb"
        assert (name.replace("_10kb", "") in mx.RUNS) and mx.diff_parts(name.replace("_10kb", ""))
        cfg = mx.run_config(name, 100, 10, 500)
        old = mx.run_config(name.replace("loopfocus2", "loopfocus"), 100, 10, 500)
        assert cfg["data"]["aml_objects_table"].endswith("objects2_f10.csv.gz")
        assert cfg["train"]["aml_object_controls"] == ["A_refrand"]
        assert cfg["data"]["diff_marks"] and cfg["data"]["diff_reference"] == old["data"]["diff_reference"]
        assert cfg["train"]["cache_dir"] == old["train"]["cache_dir"]
        if scope_all:
            assert cfg["train"]["object_scope"] == "all" and cfg["data"]["cell_objects_table"].endswith("cell_objects_refrand.csv.gz")
        else:
            assert cfg["train"].get("object_scope", "aml") == "aml" and not cfg["data"].get("cell_objects_table")
        assert old["train"]["aml_object_controls"] != ["A_refrand"]
    cell = mx.run_config("all12diffcells_loopfocus2_refres_drop_10kb", 100, 10, 500)
    assert cell["train"]["aml_object_controls"] == ["A_refrand"] and cell["data"]["cell_objects_table"].endswith("refrand.csv.gz")


# --------------------------------------------------------------------------- written tables
@pytest.mark.parametrize("fold", [10, 11, 12])
def test_written_v2_tables(fold):
    import aml_objects_table as aot
    import aml_train as at
    path = next((q for q in (REPO / f"results/aml_objects/objects2_f{fold}.csv.gz", aot.table_path(fold, 2))
                 if q.exists()), None)
    if path is None:
        pytest.skip("v2 table not generated (python scripts/aml_objects_table.py --refrand)")
    d = pd.read_csv(path, dtype={"patient": str})
    aot.assert_no_leak(d, fold)
    bad = set()
    for t in at.test_patients(fold):
        bad |= at.donor_group(t)
    assert not set(d.patient) & bad
    P, C = d[d["set"] == "pos"], d[d["set"] == "A_refrand"]
    assert len(C) > 0.8 * len(P)
    m = P.merge(C, on=["patient", "kind", "pair_id"], suffixes=("_p", "_c"))
    assert len(m) == len(C)                                    # one control per object, inheriting its pair_id
    ok = m.ref_val_p.notna() & m.ref_val_c.notna()
    assert (np.abs(m.ref_val_p - m.ref_val_c)[ok] <= 0.15 + 1e-4).all()
    okh = m.hspc_val_p.notna() & m.hspc_val_c.notna()
    assert (np.abs(m.hspc_val_p - m.hspc_val_c)[okh] <= 0.15 + 1e-4).all()
    lp = m[m.kind == "loop"]
    dp = (lp.b_start_p - lp.a_start_p).to_numpy() / 1e4
    dc = (lp.b_start_c - lp.a_start_c).to_numpy() / 1e4
    assert (np.abs(np.log10(dc / dp)) <= 0.10 + 1e-6).all()
    assert not C.duplicated(["patient", "kind", "chrom", "a_start", "b_start"]).any()
    assert {"A_hspc"} <= set(d["set"])                         # shared loops stay for the weak term


# --------------------------------------------------------------------------- smoke
def _smoke_table_v2(path):
    rows = []

    def add(kind, st, pid, a, b):
        rows.append({"patient": "018", "cell": "s", "kind": kind, "role": "pos" if st == "pos" else "ctrl",
                     "set": st, "pair_id": pid, "chrom": "chr1", "a_start": a[0] * BINSIZE, "a_end": a[1] * BINSIZE,
                     "b_start": b[0] * BINSIZE, "b_end": b[1] * BINSIZE})
    add("loop", "pos", "L0", (10, 11), (24, 25))
    add("loop", "A_refrand", "L0", (40, 41), (54, 55))
    add("loop", "pos", "L1", (50, 51), (66, 67))
    add("loop", "A_refrand", "L1", (14, 15), (30, 31))
    add("loop", "A_hspc", "L1", (28, 29), (44, 45))
    add("stripe", "pos", "S0", (30, 32), (40, 52))
    add("stripe", "A_refrand", "S0", (60, 62), (70, 82))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_smoke_training_marks_pass(workdir, caplog):
    import logging
    from flexcible import precompute as hprep
    from flexcible import train as htrain

    tab = workdir / "objects_smoke2.csv"
    _smoke_table_v2(tab)
    hprep.build(Config.from_dict(_smoke_raw(workdir, "pre2")))
    c = Config.from_dict(_smoke_raw(
        workdir, "loopfocus2", table=tab, aml_object_oversample=4, aml_object_pair_batches=True,
        aml_object_weight=1.0, aml_object_rank=0.5, aml_object_rank_consistent=False,
        aml_object_controls=["A_refrand"], aml_object_weak_controls=["A_hspc"],
        aml_object_marks_pass=True, aml_object_marks_margin=0.5, aml_object_donut=[1, 4],
        track_dropout=0.5, track_dropout_mode="uniform_count"))
    with caplog.at_level(logging.INFO):
        htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()
    txt = caplog.text
    assert "aml objects" in txt and "loop/A_refrand 2" in txt and "loop/A_hspc 1" in txt
    gaps = [float(x) for x in re.findall(r"obj_gap=(-?[0-9.]+)", txt)]
    margins = [float(x) for x in re.findall(r"obj_margin=(-?[0-9.]+)", txt)]
    assert gaps and any(abs(g) > 0 for g in gaps) and margins and max(margins) >= 0
