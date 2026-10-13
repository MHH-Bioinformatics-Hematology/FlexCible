"""Window prior network (registry section 76, hicmultiome/windowprior.py).

Position-free (architecture, rolled signal, relabelled chromosomes), leakage (the reference of a sample
never holds the sample, cooler metadata, cross-fitting folds), dref construction (identical to the
reference rows of hicmultiome.diffmarks, own-marks mode, missing marks), production (blend, cooler
round trip), the run configs of scripts/multicell_expanded.py (new families only, nothing else changes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hicmultiome import diffmarks as dm  # noqa: E402
from hicmultiome import windowprior as wp  # noqa: E402
from test_diffmarks import MARKS, SAMPLES, cfg_of, world, _tracks, _write_bw  # noqa: E402,F401
from test_pipeline import BINSIZE, CHROMS, workdir  # noqa: E402,F401


# --------------------------------------------------------------------------- position-free
def _net(m=3, k=0, seed=0):
    torch.manual_seed(seed)
    return wp.make_prior_net(m, k, base=8, depth=2)


def test_architecture_has_no_positional_input():
    net = _net(3, 1)
    assert net.reference == "none" and net.prior is False and net.encoder is None
    assert net.dist_bias is None and net.res_embed is None
    assert not any(isinstance(m, torch.nn.Embedding) for m in net.modules())
    import inspect
    names = list(inspect.signature(wp.prior_forward).parameters)
    assert names == ["net", "marks", "pairs"]          # nothing but the content of the window
    # the same parameters run on windows of any width: no parameter depends on a length
    a = wp.prior_forward(net, torch.randn(2, 3, 32), torch.randn(2, 1, 2, 32))
    b = wp.prior_forward(net, torch.randn(2, 3, 48), torch.randn(2, 1, 2, 48))
    assert a.shape == (2, 32, 32) and b.shape == (2, 48, 48)
    assert a.abs().max() <= 1.0


def test_rolled_signal_gives_the_same_map():
    """A function of the content only: the chromosome rolled by whole windows changes every coordinate and
    no value of the (aligned) blended map."""
    net = _net(3, 1)
    rng = np.random.default_rng(0)
    nb, W = 400, 32
    x = rng.gamma(2.0, 0.5, size=(3, nb)).astype(np.float32)
    pr = rng.normal(size=(1, 2, nb)).astype(np.float32)
    for k in (1, 3, 5):
        d, n = wp.shift_difference(net, x, pr, W, k)
        assert n > 1000
        assert d < 1e-5, (k, d)


def test_relabelled_chromosomes_give_the_same_map():
    net = _net(2, 0)
    rng = np.random.default_rng(1)
    chroms = {"chr1": rng.random((2, 150)).astype(np.float32), "chr2": rng.random((2, 130)).astype(np.float32)}
    out = {c: wp.predict_chrom(net, x, None, 32) for c, x in chroms.items()}
    swapped = {"chr2": chroms["chr1"], "chr1": chroms["chr2"]}          # labels permuted
    out2 = {c: wp.predict_chrom(net, x, None, 32) for c, x in swapped.items()}
    assert np.array_equal(out["chr1"].num, out2["chr2"].num)
    assert np.array_equal(out["chr2"].num, out2["chr1"].num)


def test_positional_consensus_changes_under_the_same_shift(workdir):
    """The control is sensitive: a lookup by coordinates (the positional consensus) is not invariant."""
    import cooler
    clr = cooler.Cooler(str(workdir / "target.cool"))
    a = np.asarray(clr.matrix(balance=False).fetch("chr1:10000-42000"))
    b = np.asarray(clr.matrix(balance=False).fetch("chr1:26000-58000"))
    assert a.shape == b.shape and np.abs(a - b).max() > 0.0


def test_reversal_is_an_involution_and_swaps_strands():
    x = torch.randn(2, 3, 16)
    p = torch.randn(2, 2, 2, 16)
    xr, pr = wp.reverse_inputs(x, p)
    assert torch.equal(xr[..., 0], x[..., -1])
    assert torch.equal(pr[:, :, 0, 0], p[:, :, 1, -1])               # fwd of the reversed window = rev read backwards
    x2, p2 = wp.reverse_inputs(xr, pr)
    assert torch.equal(x2, x) and torch.equal(p2, p)


def test_mark_dropout_counts_and_zeros_whole_marks():
    gen = torch.Generator().manual_seed(0)
    x = torch.ones(64, 13, 8)
    y = wp.drop_marks(x, gen, 0.5)
    per = (y.sum(-1) == 0).sum(1)                                     # marks dropped per window
    assert ((y == 0) | (y == 1)).all()
    assert per.max() <= 12 and per.min() >= 0
    assert 10 < int((per > 0).sum()) < 54                              # about half the windows
    assert wp.drop_marks(x, gen, 0.0) is x


# --------------------------------------------------------------------------- dref construction and leakage
def test_ref_rows_are_the_dref_rows_of_diffmarks(world):
    cfg = cfg_of(world, name="wp_dref")
    inst = dm.DiffMarks.get(cfg)
    spec = {s.name: s for s in cfg.data.samples}
    M = len(MARKS)
    for me in SAMPLES:
        for chrom, ln in CHROMS.items():
            nb = ln // BINSIZE
            full = inst.rows(spec[me], chrom, nb)
            mine = wp.mark_rows(inst, spec[me], chrom, nb, "ref")
            assert mine.shape == (M, nb)
            assert np.array_equal(mine, full[:M])                      # identical construction
            assert [inst.names[i] for i in range(M)] == [f"dref:{t}" for t in MARKS]


def test_own_rows_are_the_sample_itself_and_missing_marks_are_zero(world):
    cfg = cfg_of(world, name="wp_own", missing={"s1": ["H3K4me1"]})
    inst = dm.DiffMarks.get(cfg)
    spec = {s.name: s for s in cfg.data.samples}
    nb = CHROMS["chr1"] // BINSIZE
    own = wp.mark_rows(inst, spec["s0"], "chr1", nb, "own")
    sm = inst.sample_marks(spec["s0"])
    assert np.array_equal(own, sm.z["chr1"][:, :nb] * sm.valid[:, None])
    o1 = wp.mark_rows(inst, spec["s1"], "chr1", nb, "own")
    assert not o1[MARKS.index("H3K4me1")].any() and o1[MARKS.index("H3K27ac")].any()
    r1 = wp.mark_rows(inst, spec["s1"], "chr1", nb, "ref")
    assert not r1[MARKS.index("H3K4me1")].any()                        # the sample lacks it: 0 in dref as well
    with pytest.raises(ValueError):
        wp.mark_rows(inst, spec["s0"], "chr1", nb, "bogus")


def test_reference_marks_never_contain_the_sample_or_its_cell(world, tmp_path):
    """Replace the sample's own bigWigs: its reference input stays put; the others' reference moves."""
    cfgA = cfg_of(world, name="wp_leakA")
    d0 = tmp_path / "s0new"
    d0.mkdir()
    t = _tracks(4242)
    for m in MARKS:
        _write_bw(d0 / f"{m}.bigwig", t[m])
    cfgB = cfg_of(world, name="wp_leakB", dirs={**world["dirs"], "s0": d0})
    ia, ib = dm.DiffMarks.get(cfgA), dm.DiffMarks.get(cfgB)
    sa, sb = {s.name: s for s in cfgA.data.samples}, {s.name: s for s in cfgB.data.samples}
    nb = CHROMS["chr1"] // BINSIZE
    assert np.array_equal(wp.mark_rows(ia, sa["s0"], "chr1", nb), wp.mark_rows(ib, sb["s0"], "chr1", nb))
    assert not np.allclose(wp.mark_rows(ia, sa["s1"], "chr1", nb), wp.mark_rows(ib, sb["s1"], "chr1", nb))
    # the own-marks mode, by contrast, is the sample's track and moves
    assert not np.allclose(wp.mark_rows(ia, sa["s0"], "chr1", nb, "own"),
                           wp.mark_rows(ib, sb["s0"], "chr1", nb, "own"))
    # a held-out identity (hmec_deep stands for hmec) is left out of the pool
    mem = dm.reference_members(["gm12878", "hmec", "k562"], "hmec_deep")
    assert mem == ["gm12878", "k562"]


def test_explicit_members_replace_the_pool(world):
    """Another organism: the reference marks are those of the given samples (here two others)."""
    cfg = cfg_of(world, name="wp_members")
    inst = dm.DiffMarks.get(cfg)
    spec = {s.name: s for s in cfg.data.samples}
    nb = CHROMS["chr1"] // BINSIZE
    z = {n: inst.sample_marks(spec[n]).z["chr1"][:, :nb] for n in SAMPLES}
    got = wp.mark_rows(inst, spec["s0"], "chr1", nb, "ref", members=[spec["s1"], spec["s2"]])
    assert np.allclose(got, (z["s1"] + z["s2"]) / 2, atol=1e-6)


def test_crossfitting_folds_never_train_on_the_predicted_chromosome():
    for c in wp.AUTOSOMES:
        f = wp.predictor_fold(c)
        assert f in ("odd", "even") and f != wp.fold_of_chrom(c)
        assert c not in wp.train_chroms_of(f)
        assert wp.predictor_fold(c, "full") == "full"
    assert set(wp.train_chroms_of("odd")) | set(wp.train_chroms_of("even")) == set(wp.AUTOSOMES)
    assert not set(wp.train_chroms_of("odd")) & set(wp.train_chroms_of("even"))
    held = ("chr2", "chr3", "chr11", "chr14", "chr17")
    for f in wp.FOLDS:
        assert not set(held) & set(wp.train_chroms_of(f, held))
    with pytest.raises(ValueError):
        wp.train_chroms_of("bogus")


# --------------------------------------------------------------------------- production
def test_cooler_round_trip_and_leakage_guard(workdir, tmp_path):
    import cooler

    from hicmultiome.reference import assert_reference_excludes, reference_block
    net = _net(2, 0, seed=3)
    rng = np.random.default_rng(2)
    accs, sizes = {}, {}
    for chrom, ln in CHROMS.items():
        nb = ln // BINSIZE
        sizes[chrom] = nb
        accs[chrom] = wp.predict_chrom(net, rng.random((2, nb)).astype(np.float32), None, 16)
    out = wp.write_prior_cool(tmp_path / "p.cool", workdir / "target.cool", accs,
                              {"cells": ["s1", "s2"], "mode": "ref"})
    clr = cooler.Cooler(str(out))
    from hicmultiome.matrix import MatrixNormalizer
    mn = MatrixNormalizer(transform="prescaled", scale=2.0)
    blk = reference_block(clr, "chr1", 20, 16, mn)
    acc = accs["chr1"]
    num = acc.num[20:36, :16] / np.maximum(acc.den[20:36, :16], 1e-9)
    want = np.zeros((16, 16), np.float32)
    for i in range(16):
        for d in range(16 - i):
            want[i, i + d] = want[i + d, i] = num[i, d]
    assert np.allclose(blk, np.clip(want / 2.0, -1, 1), atol=1e-5)
    assert_reference_excludes(out, "s0")                               # the cell is not among the inputs
    with pytest.raises(ValueError):
        assert_reference_excludes(out, "s1")


def test_short_chromosome_is_padded():
    net = _net(2, 0)
    acc = wp.predict_chrom(net, np.random.default_rng(0).random((2, 10)).astype(np.float32), None, 32)
    assert acc.nbins == 10 and np.isfinite(acc.num).all()


# --------------------------------------------------------------------------- run configs
def test_run_configs_new_families_only(monkeypatch):
    import multicell_expanded as me
    base = {"all12honest_drop", "all12refres_drop", "all12diffcells_refres_drop"}
    for b in base:
        assert b in me.RUNS and b not in me.WP_BASE
    assert set(me.WP_BASE) == {"all12wp_ref_drop", "all12wp_refm_drop", "all12wp_self_drop", "all12wp_both_drop"}
    for n in me.WP_BASE:
        assert me.is_ref(f"{n}_10kb") and me.out_of(f"{n}_10kb") == me.REF_OUT
    assert me.reference_of("all12refres_drop_10kb", "gm12878") == me.consensus_loo("gm12878", 10000)
    assert me.reference_of("all12wp_ref_drop_10kb", "k562") == me.wp_cool("all12wp_ref_drop_10kb", "k562")
    assert me.reference_of("all12wp_both_drop_10kb", "k562") == me.consensus12(10000)   # reference stays positional
    assert str(me.wp_cool("all12wp_self_drop_10kb", "pc9")).endswith("window_prior/10kb/own/cools/pc9.cool")
    assert str(me.wp_cool("all12wp_refm_drop_25kb", "pc9")).endswith("window_prior/25kb/ref_motif/cools/pc9.cool")
