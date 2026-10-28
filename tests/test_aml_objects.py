"""Loop-centric AML recipe (registry section 63): objects table leakage, window selection, batch
composition, the object loss, config validation, run configs of folds 10 to 12 and a 60-step CPU
smoke on the synthetic genome of test_pipeline."""
from __future__ import annotations

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
from test_pipeline import BINSIZE, MOTIF, workdir  # noqa: E402,F401  (fixture)

FOLD_TEST = {10: ["018", "027", "270"], 11: ["168", "546", "629", "1360"], 12: ["472", "773", "1916"]}


# --------------------------------------------------------------------------- objects table
def _fake_features():
    rows = []
    pats = sorted({p for v in FOLD_TEST.values() for p in v} | {"496", "424"})
    for p in pats:
        for k in range(4):
            for st in ("pos", "A_hspc", "A_full"):
                rows.append({"patient": p, "kind": "loop", "set": st, "pair_id": f"{p}:L{k}",
                             "chrom": "chr1", "a_lo": 100 + 40 * k, "a_hi": 101 + 40 * k,
                             "b_lo": 115 + 40 * k, "b_hi": 116 + 40 * k, "pair_type": "other"})
        rows.append({"patient": p, "kind": "stripe", "set": "pos", "pair_id": f"{p}:S0",
                     "chrom": "chr2", "a_lo": 50, "a_hi": 52, "b_lo": 60, "b_hi": 100, "pair_type": "other"})
        rows.append({"patient": p, "kind": "loop", "set": "pos", "pair_id": f"{p}:Lbig", "chrom": "chr3",
                     "a_lo": 10, "a_hi": 11, "b_lo": 400, "b_hi": 401, "pair_type": "other"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fold", [10, 11, 12])
def test_table_has_no_test_or_partner_objects(fold):
    import aml_objects_table as aot
    import aml_train as at

    F = _fake_features()
    d = aot.build_table(fold, F)
    test = set(FOLD_TEST[fold])
    partners = {"496"} if "168" in test else set()      # 168 and 496: one donor
    assert not set(d.patient) & (test | partners | {"424"} if "270" in test else test | partners)
    assert set(d.patient) == set(at.train_patients(fold))
    assert not set(d.cell) & {at.guard(p) for p in test}
    assert d.cell.str.startswith("amlxu").all()
    assert (d.groupby(["kind", "set"]).size() > 0).all()
    assert not ((d.kind == "loop") & (d.b_end - d.a_start > 1_500_000)).any()      # span filter
    # a leaking table is rejected
    bad = pd.concat([d, d.iloc[:1].assign(patient=sorted(test)[0], cell=at.guard(sorted(test)[0]))])
    with pytest.raises(AssertionError):
        aot.assert_no_leak(bad, fold)


@pytest.mark.parametrize("fold", [10, 11, 12])
def test_real_table_is_leak_free(fold):
    import aml_objects_table as aot
    import aml_train as at

    path = next((q for q in (REPO / f"results/aml_objects/objects_f{fold}.csv.gz", aot.table_path(fold))
                 if q.exists()), None)
    if path is None:
        pytest.skip("objects table not generated (python scripts/aml_objects_table.py)")
    d = pd.read_csv(path, dtype={"patient": str})
    bad = set()
    for t in at.test_patients(fold):
        bad |= at.donor_group(t)
    assert not set(d.patient) & bad
    assert set(d.patient) <= set(at.train_patients(fold))
    aot.assert_no_leak(d, fold)
    assert {"pos", "A_hspc"} <= set(d["set"])
    assert d.n_train_other[d.role == "pos"].notna().all()


def test_load_objects_rejects_foreign_cells(tmp_path):
    d = pd.DataFrame({"cell": ["amlxu018", "amlxu999"], "kind": ["loop", "loop"], "role": ["pos", "pos"],
                      "set": ["pos", "pos"], "pair_id": ["a", "b"], "chrom": ["chr1"] * 2,
                      "a_start": [0, 0], "a_end": [1000, 1000], "b_start": [5000, 5000], "b_end": [6000, 6000]})
    p = tmp_path / "t.csv"
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="not training samples"):
        ao.load_objects(str(p), {"amlxu018"})
    assert len(ao.load_objects(str(p), {"amlxu018", "amlxu999"})) == 2


# --------------------------------------------------------------------------- windows
def _objs():
    d = pd.DataFrame({
        "cell": ["s"] * 6, "kind": ["loop", "loop", "loop", "stripe", "stripe", "loop"],
        "role": ["pos", "ctrl", "pos", "pos", "ctrl", "ctrl"],
        "set": ["pos", "A_hspc", "pos", "pos", "A_hspc", "A_full"],
        "pair_id": ["s:L0", "s:L0", "s:L1", "s:S0", "s:S0", "s:L1"],
        "chrom": ["chr1"] * 6,
        "a_start": [10000, 40000, 60000, 20000, 70000, 5000], "a_end": [11000, 41000, 61000, 22000, 72000, 6000],
        "b_start": [24000, 55000, 75000, 30000, 80000, 9000], "b_end": [25000, 56000, 76000, 40000, 90000, 10000]})
    d["role_id"] = np.where(d["set"] == "pos", 1, 2)
    return ao.to_bins(d, 1000)


def test_to_bins_and_rows_of_objects():
    o = _objs()
    assert (o.a1 - o.a0).min() >= 1
    # windows of 32 bins at stride 16 on chr1, sample s
    starts = list(range(0, 65, 2))
    coords = (["chr1"] * len(starts), np.array(starts), ["s"] * len(starts))
    rows = ao.rows_of_objects(o, coords, 32, 5)
    for r, s in enumerate(starts):
        for (k, role, a0, a1, b0, b1, oi) in rows[r]:
            assert 5 <= min(a0, b0) and max(a1, b1) <= 32 - 5          # margin inside the window
            assert o.a0[oi] - s == a0 and o.b1[oi] - s == b1            # window-local coordinates
    held = {x[6] for rr in rows for x in rr}
    assert {0, 1, 2, 3, 4} <= held
    # other sample / chromosome: nothing
    other = ao.rows_of_objects(o, (["chr2"] * 3, np.array([0, 16, 32]), ["s"] * 3), 32, 5)
    assert all(not r for r in other)
    other = ao.rows_of_objects(o, (["chr1"] * 3, np.array([0, 16, 32]), ["t"] * 3), 32, 5)
    assert all(not r for r in other)


def test_select_pairs_one_to_one_matched():
    o = _objs()
    starts = list(range(0, 65, 2))
    coords = (["chr1"] * len(starts), np.array(starts), ["s"] * len(starts))
    rows = ao.rows_of_objects(o, coords, 32, 5)
    pos, ctl = ao.select_pairs(rows, o, np.random.default_rng(0), fallback=False)
    assert len(pos) == len(ctl) > 0
    assert len(set(pos)) == len(pos) and len(set(ctl)) == len(ctl) and not set(pos) & set(ctl)
    key = (o.cell + "|" + o.kind + "|" + o.pair_id).to_numpy()
    for p, c in zip(pos, ctl):
        kp = {key[x[6]] for x in rows[p] if x[1] == 1}
        kc = {key[x[6]] for x in rows[c] if x[1] == 2}
        assert kp & kc                                              # a shared pair_id
    roles = ao.row_roles(rows)
    assert (roles[pos] == 1).all() and (roles[ctl] >= 1).all()
    p2, c2 = ao.select_pairs(rows, o, np.random.default_rng(0), fallback=True)     # never fewer pairs
    assert len(p2) == len(c2) >= len(pos) and not set(p2) & set(c2)
    assert len(set(p2) | set(c2)) == len(p2) + len(c2)


def test_object_batches_compose_and_keep_rows():
    rng = np.random.default_rng(1)
    roles = np.array([1] * 12 + [2] * 12 + [0] * 40, dtype=np.int8)
    perm = rng.permutation(len(roles))
    out = ao.object_batches(perm, roles, 6, rng)
    assert sorted(out) == sorted(perm)
    nb = 0
    for k in range(0, len(out) - 5, 6):
        r = roles[out[k:k + 6]]
        if (r == 1).sum() == 3 and (r == 2).sum() == 3:
            nb += 1
    assert nb == 4                                                  # 12 pos / 3 per batch
    assert np.array_equal(ao.object_batches(perm, roles, 1, rng), perm)


def test_flip_object_matches_tensor_flip():
    W = 32
    for o in [(1, 1, 5, 7, 12, 20, 0), (0, 1, 5, 6, 15, 16, 0)]:
        plain = ao.batch_geometry([[o]], None, W, 4, "cpu")
        rev = ao.batch_geometry([[o]], [True], W, 4, "cpu")
        assert torch.equal(rev.mask, torch.flip(plain.mask, dims=[-1, -2]))
    # object score at the flipped position equals the score on the flipped map
    M = torch.zeros(1, W, W)
    M[0, 5:7, 12:20] = 0.4
    M = torch.maximum(M, M.transpose(1, 2))
    o = (1, 1, 5, 7, 12, 20, 0)
    _, s0 = ao.object_scores(M, ao.batch_geometry([[o]], None, W, 4, "cpu"), 1, 4)
    _, s1 = ao.object_scores(torch.flip(M, dims=[-1, -2]), ao.batch_geometry([[o]], [True], W, 4, "cpu"), 1, 4)
    assert s0.item() == pytest.approx(s1.item()) and s0.item() == pytest.approx(0.4)
    L = (0, 1, 5, 6, 15, 16, 0)
    P = torch.zeros(1, W, W)
    P[0, 5, 15] = 1.0
    P = torch.maximum(P, P.transpose(1, 2))
    l0, _ = ao.object_scores(P, ao.batch_geometry([[L]], None, W, 4, "cpu"), 1, 4)
    l1, _ = ao.object_scores(torch.flip(P, dims=[-1, -2]), ao.batch_geometry([[L]], [True], W, 4, "cpu"), 1, 4)
    assert l0.item() == pytest.approx(l1.item()) and l0.item() > 0.5


# --------------------------------------------------------------------------- loss
def _planted(amp=0.5, W=32):
    T = torch.zeros(1, W, W)
    T[0, 10, 20] = amp
    T = torch.maximum(T, T.transpose(1, 2))
    return T


def test_object_loss_values_and_gradient():
    W = 32
    T = _planted()
    ref = torch.zeros(1, W, W)
    objs = [[(0, 1, 10, 11, 20, 21, 0)]]
    geo = ao.batch_geometry(objs, None, W, 4, "cpu")
    assert geo.n_objects == 1 and geo.mask.sum() > 0
    loss = ao.ObjectLoss(weight=1.0)
    perfect = loss(T.clone(), T, ref, geo).item()
    pred = torch.zeros(1, W, W, requires_grad=True)
    flat = loss(pred, T, ref, geo)
    assert perfect == pytest.approx(0.0, abs=1e-6) and flat.item() > 0.05
    flat.backward()
    assert pred.grad[0, 10, 20] < 0                       # raises the planted pixel
    # no objects: zero and no graph
    assert ao.ObjectLoss(weight=1.0, rank=1.0)(pred, T, ref, ao.batch_geometry([[]], None, W, 4, "cpu")).item() == 0


def test_rank_term_separates_positive_from_control():
    W = 32
    T = torch.zeros(2, W, W)
    T[0, 10, 20] = 0.6                                    # AML-specific loop, measured enrichment
    T[1, 12, 22] = 0.1                                    # shared control, weaker
    T = torch.maximum(T, T.transpose(1, 2))
    ref = torch.zeros(2, W, W)
    objs = [[(0, 1, 10, 11, 20, 21, 0)], [(0, 2, 12, 13, 22, 23, 1)]]
    geo = ao.batch_geometry(objs, None, W, 4, "cpu")
    loss = ao.ObjectLoss(rank=1.0, temp=0.1)
    same = torch.zeros(2, W, W)
    same[:, 10, 20] = same[:, 12, 22] = 0.3
    right, wrong = same.clone(), same.clone()
    right[0, 10, 20] += 0.3
    wrong[1, 12, 22] += 0.3
    right.requires_grad_(True)
    lr, lw, ls = loss(right, T, ref, geo).item(), loss(wrong, T, ref, geo).item(), loss(same, T, ref, geo).item()
    assert lr < ls < lw
    loss(right, T, ref, geo).backward()
    assert right.grad[0, 10, 20] < 0 and right.grad[1, 12, 22] > 0   # push apart
    # a pair the measurement does not separate is ignored when consistent
    T2 = T.clone()
    T2[0, 10, 20] = 0.05
    assert ao.ObjectLoss(rank=1.0, consistent=True)(same, T2, ref, geo).item() == 0.0
    assert ao.ObjectLoss(rank=1.0, consistent=False)(same, T2, ref, geo).item() > 0


def test_stripe_scores():
    W = 32
    M = torch.zeros(1, W, W)
    M[0, 8:10, 14:24] = 0.4
    geo = ao.batch_geometry([[(1, 1, 8, 10, 14, 24, 0)]], None, W, 4, "cpu")
    ls, ss = ao.object_scores(M, geo, 1, 4)
    assert ls.numel() == 0 and ss.item() == pytest.approx(0.4)


# --------------------------------------------------------------------------- config
def _smoke_raw(workdir, name, table=None, **train):
    raw = {
        "data": {
            "samples": [{"name": "s", "cool_path": str(workdir / "target.cool"),
                         "bigwig_dir": str(workdir), "reference_cool": str(workdir / "target.cool")}],
            "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF}, {"name": "H3K27ac", "type": "histone"}],
            "resolution": BINSIZE,
        },
        "sequence": {"fasta": str(workdir / "genome.fa"), "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "window": {"size": 32, "stride": 4, "max_zero_fraction": 0.99, "transform": "prescaled"},
        "model": {"base_channels": 16, "depth": 2, "reference": "residual"},
        "train": {"output_dir": str(workdir / f"run_{name}"), "cache_dir": str(workdir / "cache_loop"),
                  "batch_size": 4, "max_steps": 60, "eval_every": 30, "val_regions": 1,
                  "val_exclude_overlap": False, "lr": 1e-3, "augment_reverse": True, **train},
    }
    if table is not None:
        raw["data"]["aml_objects_table"] = str(table)
    return raw


def test_config_defaults_off_and_validation(workdir):
    c = Config.from_dict(_smoke_raw(workdir, "d"))
    t = c.train
    assert (t.aml_object_oversample, t.aml_object_weight, t.aml_object_rank) == (1, 0.0, 0.0)
    assert not t.aml_object_pair_batches and c.data.aml_objects_table is None
    with pytest.raises(ValueError, match="aml_objects_table"):
        Config.from_dict(_smoke_raw(workdir, "d", aml_object_weight=1.0))
    raw = _smoke_raw(workdir, "d", table="x.csv", aml_object_weight=1.0)
    raw["model"]["reference"] = "none"
    raw["data"]["samples"][0].pop("reference_cool")
    with pytest.raises(ValueError, match="model.reference"):
        Config.from_dict(raw)
    with pytest.raises(ValueError, match="donut"):
        Config.from_dict(_smoke_raw(workdir, "d", table="x.csv", aml_object_rank=1.0, aml_object_donut=[4, 2]))
    with pytest.raises(ValueError, match="val_regions"):
        Config.from_dict(_smoke_raw(workdir, "d", table="x.csv", aml_object_rank=1.0, val_regions=0))


@pytest.mark.parametrize("fold", [10, 11, 12])
@pytest.mark.parametrize("res", [10000, 25000])
def test_run_config_of_loopfocus(fold, res):
    import multicell_expanded as mx

    if not mx.HM.exists():
        pytest.skip("share not mounted")
    name = f"all12aml_loopfocus_refres_drop_f{fold}_{res // 1000}kb"
    cfg = mx.run_config(name, 100, 10, 500)
    ref = mx.run_config(f"all12aml_refres_drop_f{fold}_{res // 1000}kb", 100, 10, 500)
    assert cfg["train"]["cache_dir"] == ref["train"]["cache_dir"]        # the existing AML reference cache
    assert cfg["model"]["reference"] == "residual"
    assert cfg["data"]["aml_objects_table"].endswith(f"objects_f{fold}.csv.gz")
    t = cfg["train"]
    assert t["aml_object_oversample"] == 5 and t["aml_object_weight"] > 0 and t["aml_object_rank"] > 0
    assert t["aml_object_donut"] == ([1, 4] if res == 10000 else [1, 3])
    # everything else equals the residual-reference run
    for k in set(ref["train"]) - {"output_dir"}:
        assert cfg["train"][k] == ref["train"][k], k
    assert {s["name"] for s in cfg["data"]["samples"]} == {s["name"] for s in ref["data"]["samples"]}
    import aml_train as at
    test_cells = {at.guard(p) for p in at.test_patients(fold)}
    assert not {s["name"] for s in cfg["data"]["samples"]} & test_cells
    assert all(k not in ref["train"] for k in ("aml_object_oversample", "aml_object_rank"))


# --------------------------------------------------------------------------- smoke
def _smoke_table(path):
    rows = []
    def add(kind, st, pid, a, b):
        rows.append({"patient": "018", "cell": "s", "kind": kind, "role": "pos" if st == "pos" else "ctrl",
                     "set": st, "pair_id": pid, "chrom": "chr1", "a_start": a[0] * BINSIZE, "a_end": a[1] * BINSIZE,
                     "b_start": b[0] * BINSIZE, "b_end": b[1] * BINSIZE})
    add("loop", "pos", "L0", (10, 11), (24, 25))
    add("loop", "A_hspc", "L0", (40, 41), (54, 55))
    add("loop", "pos", "L1", (50, 51), (66, 67))
    add("loop", "A_hspc", "L1", (14, 15), (30, 31))
    add("stripe", "pos", "S0", (30, 32), (40, 52))
    add("stripe", "A_hspc", "S0", (60, 62), (70, 82))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_smoke_training_with_all_flags(workdir, caplog):
    import logging

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    tab = workdir / "objects_smoke.csv"
    _smoke_table(tab)
    hprep.build(Config.from_dict(_smoke_raw(workdir, "pre")))
    c = Config.from_dict(_smoke_raw(
        workdir, "loopfocus", table=tab, aml_object_oversample=4, aml_object_pair_batches=True,
        aml_object_weight=1.0, aml_object_rank=0.5, aml_object_rank_consistent=False,
        aml_object_donut=[1, 4]))
    with caplog.at_level(logging.INFO):
        htrain.train(c)
    out = Path(c.train.output_dir)
    assert (out / "best").exists()
    txt = caplog.text
    assert "aml objects" in txt and "obj_rank" in txt and "obj_pix" in txt
    assert "step 60" in txt or "done." in txt
    import re
    ranks = [float(x) for x in re.findall(r"obj_rank=([0-9.]+)", txt)]
    assert ranks and max(ranks) > 0                       # half-and-half batches give the ranking pairs
    # flags off: the recipe code is not entered
    caplog.clear()
    c0 = Config.from_dict(_smoke_raw(workdir, "plain", table=tab))
    with caplog.at_level(logging.INFO):
        htrain.train(c0)
    assert "aml objects" not in caplog.text and "obj_pix" not in caplog.text
