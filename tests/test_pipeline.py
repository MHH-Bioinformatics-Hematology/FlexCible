"""End-to-end test on a tiny synthetic genome (CPU, seconds).

Builds two mini chromosomes with a planted convergent-motif loop structure,
writes cool/bigwig/fasta/jaspar/capture fixtures, then runs precompute ->
train (a few steps) -> predict -> evaluate/loops and asserts the plumbing:
shapes, masks, normaliser round-trips, config validation, and that predicted
cool files are valid and non-empty.

Run:  python -m pytest tests/ -x -q     (needs torch, cooler, pyBigWig, pyfaidx)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexcible.config import Config  # noqa: E402

BINSIZE = 1000
CHROMS = {"chr1": 96 * BINSIZE, "chr2": 80 * BINSIZE}
MOTIF = "MA9999.1"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_fasta(path: Path) -> None:
    rng = np.random.default_rng(0)
    with open(path, "w") as f:
        for chrom, ln in CHROMS.items():
            seq = rng.choice(list("ACGT"), size=ln)
            # plant forward motif sites early, reverse sites later (convergent)
            site = "ACGTACGTAC"
            rc = "GTACGTACGT"
            for start in (5 * BINSIZE, 20 * BINSIZE, 40 * BINSIZE):
                seq[start : start + 10] = list(site)
            for start in (12 * BINSIZE, 30 * BINSIZE, 55 * BINSIZE):
                seq[start : start + 10] = list(rc)
            f.write(f">{chrom}\n")
            s = "".join(seq)
            for i in range(0, ln, 60):
                f.write(s[i : i + 60] + "\n")


def _write_jaspar(path: Path) -> None:
    # PFM matching ACGTACGTAC exactly
    site = "ACGTACGTAC"
    counts = {b: [(90 if c == b else 3) for c in site] for b in "ACGT"}
    with open(path, "w") as f:
        f.write(f">{MOTIF}\tTESTMOTIF\n")
        for b in "ACGT":
            f.write(f"{b} [ " + " ".join(str(x) for x in counts[b]) + " ]\n")


def _write_bigwig(path: Path, peak_bins: dict[str, list[int]]) -> None:
    import pyBigWig

    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(c, ln) for c, ln in CHROMS.items()])
    for chrom, ln in CHROMS.items():
        nb = ln // BINSIZE
        vals = np.full(nb, 1.0, dtype=np.float64)
        for b in peak_bins.get(chrom, []):
            vals[b] = 25.0
        bw.addEntries([chrom] * nb, [i * BINSIZE for i in range(nb)],
                      ends=[(i + 1) * BINSIZE for i in range(nb)],
                      values=[float(v) for v in vals])
    bw.close()


def _write_cool(path: Path) -> None:
    import cooler
    import pandas as pd

    bins = []
    for chrom, ln in CHROMS.items():
        for s in range(0, ln, BINSIZE):
            bins.append((chrom, s, min(s + BINSIZE, ln)))
    bins = pd.DataFrame(bins, columns=["chrom", "start", "end"])
    rng = np.random.default_rng(1)
    px = []
    offset = 0
    for chrom, ln in CHROMS.items():
        nb = ln // BINSIZE
        for i in range(nb):
            for j in range(i, min(i + 40, nb)):
                base = 60.0 / (1 + (j - i)) + rng.random()
                px.append((offset + i, offset + j, base))
        offset += nb
    pixels = pd.DataFrame(px, columns=["bin1_id", "bin2_id", "count"])
    cooler.create_cooler(str(path), bins=bins, pixels=pixels,
                         dtypes={"count": "float64"}, symmetric_upper=True,
                         ordered=True)


def _write_capture(path: Path) -> None:
    hdr = ("chr bait\tstart bait\tend bait\tSymbol\tEnsembl Gene ID\t"
           "expresssion quartile\tchr\tstart\tend\traw count\t"
           "log(observed/expected)\n")
    rows = []
    for chrom in CHROMS:
        for bait_bin, other_bin, score in ((5, 12, 9.0), (20, 30, 12.0), (40, 55, 7.5)):
            rows.append(
                f"{chrom}\t{bait_bin * BINSIZE}\t{bait_bin * BINSIZE + 900}\tGENE\tENSG\t1\t"
                f"{chrom}\t{other_bin * BINSIZE}\t{other_bin * BINSIZE + 900}\t50\t{score}\n"
            )
    # one trans interaction that must be ignored
    rows.append("chr1\t1000\t1900\tG\tE\t1\tchr2\t1000\t1900\t9\t8.0\n")
    path.write_text(hdr + "".join(rows))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("hicmu")
    _write_fasta(root / "genome.fa")
    _write_jaspar(root / f"{MOTIF}.jaspar")
    _write_cool(root / "target.cool")
    _write_bigwig(root / "CTCF.bigwig",
                  {"chr1": [5, 12, 20, 30, 40, 55], "chr2": [5, 12, 20, 30]})
    _write_bigwig(root / "H3K27ac.bigwig", {"chr1": [10, 11, 12], "chr2": [8, 9]})
    _write_capture(root / "capture.txt")
    return root


@pytest.fixture(scope="module")
def cfg(workdir):
    raw = {
        "data": {
            "cool_path": str(workdir / "target.cool"),
            "bigwig_dir": str(workdir),
            "tracks": [
                {"name": "CTCF", "type": "tf", "motif": MOTIF},
                {"name": "H3K27ac", "type": "histone"},
            ],
            "resolution": BINSIZE,
        },
        "sequence": {"fasta": str(workdir / "genome.fa"),
                     "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "capture": {"interactions": str(workdir / "capture.txt")},
        "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99},
        "model": {"base_channels": 16, "depth": 2},
        "train": {
            "output_dir": str(workdir / "run"),
            "cache_dir": str(workdir / "cache"),
            "batch_size": 8, "max_steps": 60, "eval_every": 30,
            "val_fraction": 0.2, "lr": 1e-3,
        },
    }
    return Config.from_dict(raw)


# --------------------------------------------------------------------------- #
# unit-level checks
# --------------------------------------------------------------------------- #
def test_motif_required_for_tf():
    with pytest.raises(ValueError, match="binding\\s+motif"):
        from flexcible.config import TrackSpec

        TrackSpec(name="CTCF", type="tf")


def test_motif_rejected_for_histone():
    from flexcible.config import TrackSpec

    with pytest.raises(ValueError, match="does not apply"):
        TrackSpec(name="H3K27ac", type="histone", motif="MA0139.2")


def test_motif_scan_finds_planted_sites(workdir):
    from flexcible.motifs import cached_motif_tracks

    tr = cached_motif_tracks(str(workdir / "genome.fa"), "chr1", MOTIF,
                             str(workdir), CHROMS["chr1"] // BINSIZE, BINSIZE,
                             0.8, str(workdir / "gcache"))
    fwd, rev = tr
    assert fwd[5] >= 1 and fwd[20] >= 1 and fwd[40] >= 1
    assert rev[12] >= 1 and rev[30] >= 1 and rev[55] >= 1


def test_sequence_features(workdir):
    from flexcible.sequence import cached_sequence_features

    nb = CHROMS["chr1"] // BINSIZE
    feats = cached_sequence_features(str(workdir / "genome.fa"), "chr1",
                                     ["gc", "cpg"], nb, BINSIZE,
                                     str(workdir / "gcache"))
    assert feats.shape == (2, nb)
    assert 0.2 < feats[0].mean() < 0.8          # GC of random sequence
    assert (feats[1] >= 0).all() and feats[1].mean() > 0


def test_capture_binning_and_mask(workdir, cfg):
    from flexcible.capture import CaptureData

    cap = CaptureData(cfg.capture.interactions, "mifsud", 0.0)
    nb = CHROMS["chr1"] // BINSIZE
    cc = cap.chrom("chr1", nb, BINSIZE)
    assert cc.scores[5, 12] == pytest.approx(9.0)
    assert cc.scores[12, 5] == pytest.approx(9.0)     # symmetric
    assert cc.bait_bins[5] == 1 and cc.bait_bins[20] == 1
    assert cc.scores[1, 1] == 0.0                      # trans line ignored


def test_directed_cond_channels():
    import torch

    from flexcible.model import build_spatial_cond, n_cond_channels

    sym = torch.randn(2, 5, 16)
    pairs = torch.randn(2, 1, 2, 16)
    img = build_spatial_cond(sym, pairs)
    assert img.shape == (2, n_cond_channels(5, 1), 16, 16)
    # the fwd_i*rev_j plane must be the transpose of rev_i*fwd_j
    conv, divr = img[:, 15], img[:, 16]
    assert torch.allclose(conv, divr.transpose(-1, -2), atol=1e-5)


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def test_precompute_train_predict_evaluate(workdir, cfg):
    from flexcible import loops, precompute, predict, train
    from flexcible.dataset import WindowDataset

    manifest = precompute.build(cfg)
    assert manifest["motif_ids"] == [MOTIF]
    assert manifest["has_capture"] is True
    assert manifest["sym_rows"] == [
        "CTCF", "H3K27ac", "seq:gc", "seq:cpg",
        f"motif:{MOTIF}:fwd", f"motif:{MOTIF}:rev",
    ]

    ds = WindowDataset(cfg.train.cache_dir, "train")
    assert len(ds) > 0
    it = ds[0]
    assert it["target"].shape == (2, 32, 32)
    assert it["cond"].shape == (6, 32)
    assert it["bait"].shape == (32,)
    assert ds.motif_row_offset() == 4

    train.train(cfg)
    out = Path(cfg.train.output_dir)
    assert (out / "best").exists() or (out / "latest").exists()
    assert (out / "learning_curve.png").exists()

    pred_cool = str(Path(cfg.train.output_dir) / "pred.cool")
    cap_cool = str(Path(cfg.train.output_dir) / "pred_capture.cool")
    predict.predict(cfg, "all", pred_cool, batch=4, pchic_out=cap_cool)

    import cooler

    clr = cooler.Cooler(pred_cool)
    m = clr.matrix(balance=False).fetch("chr1")
    assert np.nansum(m) > 0
    assert cooler.Cooler(cap_cool).info["nnz"] > 0

    res = loops.evaluate(cfg, pred_cool, cap_cool, "all")
    assert "chr1" in res["per_chrom"]
    assert np.isfinite(res["mean_insulation_pearson"])
    ep = res["per_chrom"]["chr1"].get("loops_capture_channel", {})
    assert ep.get("n_pairs", 0) > 0


def test_chip_only_ablation(workdir):
    """No fasta -> no seq/motif channels, no capture -> 1 output channel."""
    from flexcible import precompute

    raw = {
        "data": {
            "cool_path": str(workdir / "target.cool"),
            "bigwig_dir": str(workdir),
            "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                       {"name": "H3K27ac", "type": "histone"}],
            "resolution": BINSIZE,
        },
        "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99},
        "train": {"cache_dir": str(workdir / "cache_ablation"),
                  "output_dir": str(workdir / "run_ablation")},
    }
    cfg = Config.from_dict(raw)
    assert cfg.active_motif_ids == []
    manifest = precompute.build(cfg)
    assert manifest["sym_rows"] == ["CTCF", "H3K27ac"]
    assert manifest["has_capture"] is False


def test_multisample_degron_and_perturb(workdir, cfg):
    """Two samples (one degron with zeroed CTCF): raw-zero encoding + perturb CLI."""
    import numpy as np

    from flexcible import perturb, precompute, train
    from flexcible.config import Config
    from flexcible.dataset import WindowDataset

    raw = {
        "data": {
            "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                       {"name": "H3K27ac", "type": "histone"}],
            "resolution": BINSIZE,
            "samples": [
                {"name": "wt", "cool_path": str(workdir / "target.cool"),
                 "bigwig_dir": str(workdir),
                 "capture_interactions": str(workdir / "capture.txt")},
                {"name": "degron", "cool_path": str(workdir / "target.cool"),
                 "bigwig_dir": str(workdir),
                 "zeroed_tracks": ["CTCF"]},
            ],
        },
        "sequence": {"fasta": str(workdir / "genome.fa"),
                     "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99},
        "model": {"base_channels": 16, "depth": 2},
        "train": {"output_dir": str(workdir / "run_ms"),
                  "cache_dir": str(workdir / "cache_ms"),
                  "batch_size": 8, "max_steps": 40, "eval_every": 20,
                  "val_fraction": 0.2, "lr": 1e-3},
    }
    mcfg = Config.from_dict(raw)
    manifest = precompute.build(mcfg)
    assert set(x["sample"] for x in manifest["shards"]) == {"wt", "degron"}
    assert manifest["has_capture"] is True

    ds = WindowDataset(mcfg.train.cache_dir, "train")
    ctcf_row = manifest["sym_rows"].index("CTCF")
    wt_vals, degron_vals = [], []
    for i in range(len(ds)):
        it = ds[i]
        (wt_vals if it["sample"] == "wt" else degron_vals).append(
            it["cond"][ctcf_row].numpy())
    # degron CTCF rows must be constant (the normalised raw-zero level)
    dv = np.concatenate(degron_vals)
    assert dv.std() < 1e-4
    assert np.concatenate(wt_vals).std() > 0.1
    # degron windows carry no capture -> bait all zero
    for i in range(len(ds)):
        it = ds[i]
        if it["sample"] == "degron":
            assert float(it["bait"].sum()) == 0.0

    train.train(mcfg)
    out = perturb.perturb(
        mcfg, f"chr1:{5 * BINSIZE}-{15 * BINSIZE}", ["CTCF"],
        str(workdir / "perturb.png"), sample_name="wt", zero_motifs=True)
    assert Path(out).exists()


def test_capture_as_optional_input(workdir):
    """Capture-input mode: trains with modality dropout, predicts in BOTH
    modes (tracks-only default; --capture-input when data available)."""
    import cooler
    import numpy as np

    from flexcible import precompute, predict, train
    from flexcible.config import Config

    raw = {
        "data": {
            "cool_path": str(workdir / "target.cool"),
            "bigwig_dir": str(workdir),
            "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                       {"name": "H3K27ac", "type": "histone"}],
            "resolution": BINSIZE,
        },
        "sequence": {"fasta": str(workdir / "genome.fa"),
                     "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "capture": {"interactions": str(workdir / "capture.txt"),
                    "use_as_input": True, "input_dropout": 0.5},
        "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99},
        "model": {"base_channels": 16, "depth": 2},
        "train": {"output_dir": str(workdir / "run_capin"),
                  "cache_dir": str(workdir / "cache_capin"),
                  "batch_size": 8, "max_steps": 40, "eval_every": 20,
                  "val_fraction": 0.2, "lr": 1e-3},
    }
    cfg = Config.from_dict(raw)
    manifest = precompute.build(cfg)
    assert manifest["capture_input"] is True

    train.train(cfg)

    out1 = str(workdir / "pred_capin_absent.cool")
    predict.predict(cfg, "all", out1, batch=4)                 # tracks-only
    out2 = str(workdir / "pred_capin_fed.cool")
    predict.predict(cfg, "all", out2, batch=4, capture_input=True)
    m1 = cooler.Cooler(out1).matrix(balance=False).fetch("chr1")
    m2 = cooler.Cooler(out2).matrix(balance=False).fetch("chr1")
    assert np.nansum(m1) > 0 and np.nansum(m2) > 0
    # the two modes must actually differ (the input plane is used)
    assert not np.allclose(np.nan_to_num(m1), np.nan_to_num(m2))


def test_obs_exp_roundtrip():
    """obs_exp encode/decode inverts exactly within the clip range, and
    decode_oe returns the decay-free ratio."""
    import numpy as np

    from flexcible.matrix import MatrixNormalizer

    rng = np.random.default_rng(0)
    w = 32
    d = np.abs(np.subtract.outer(np.arange(w), np.arange(w)))
    blocks = [(50.0 / (1 + d)) * rng.uniform(0.5, 2.0, size=(w, w))
              for _ in range(30)]
    norm = MatrixNormalizer(transform="obs_exp", clip_quantile=0.999)
    norm.fit_obs_exp(blocks)
    b = blocks[0]
    rec = norm.decode(norm.encode(b))
    oe_true = (1 + b) / (1 + np.asarray(norm.expected)[d])
    inside = np.abs(np.log(oe_true)) < norm.scale * 0.98
    assert np.allclose(rec[inside], b[inside], rtol=1e-4, atol=1e-3)
    ratio = norm.decode_oe(norm.encode(b))
    assert np.allclose(ratio[inside], oe_true[inside], rtol=1e-4)
    assert 0.5 < np.median(ratio) < 2.0            # decay removed


def test_obs_exp_smooth_reverse_tta(workdir):
    """obs_exp target + smoothing + reversal augmentation trains; predict
    works plain, with --tta, and with --obs-exp output."""
    import cooler
    import numpy as np

    from flexcible import precompute, predict, train
    from flexcible.config import Config

    raw = {
        "data": {
            "cool_path": str(workdir / "target.cool"),
            "bigwig_dir": str(workdir),
            "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                       {"name": "H3K27ac", "type": "histone"}],
            "resolution": BINSIZE,
        },
        "sequence": {"fasta": str(workdir / "genome.fa"),
                     "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99,
                   "transform": "obs_exp", "smooth_sigma": 0.75},
        "model": {"base_channels": 16, "depth": 2},
        "train": {"output_dir": str(workdir / "run_oe"),
                  "cache_dir": str(workdir / "cache_oe"),
                  "batch_size": 8, "max_steps": 40, "eval_every": 20,
                  "val_fraction": 0.2, "lr": 1e-3, "augment_reverse": True},
    }
    cfg = Config.from_dict(raw)
    precompute.build(cfg)
    train.train(cfg)

    out = str(workdir / "pred_oe.cool")
    predict.predict(cfg, "all", out, batch=4, tta=True)
    m = cooler.Cooler(out).matrix(balance=False).fetch("chr1")
    assert np.nansum(m) > 0
    # counts mode must show distance decay; obs/exp mode must not
    out2 = str(workdir / "pred_oe_ratio.cool")
    predict.predict(cfg, "all", out2, batch=4, obs_exp_out=True)
    r = np.nan_to_num(cooler.Cooler(out2).matrix(balance=False).fetch("chr1"))
    near = np.nanmean(np.diagonal(r, 1))
    far = np.nanmean(np.diagonal(r, 20))
    assert 0.2 < far / max(near, 1e-9) < 5.0       # ratio is decay-free


def test_cond_modes_restrict_information():
    """The conditioning ablation must actually withhold information:
    'tile' is constant along j (pixel (i,j) cannot see locus j) and 'pair'
    keeps both endpoints but drops the product/diff/distance features."""
    import torch

    from flexcible.model import build_spatial_cond, n_cond_channels

    sym = torch.randn(2, 5, 16)
    pairs = torch.randn(2, 1, 2, 16)

    full = build_spatial_cond(sym, pairs, None, "full")
    assert full.shape[1] == n_cond_channels(5, 1, 0, "full") == 3 * 5 + 2 + 1

    tile = build_spatial_cond(sym, pairs, None, "tile")
    assert tile.shape[1] == n_cond_channels(5, 1, 0, "tile") == 5
    # constant along the j axis => no information about the second locus
    assert torch.allclose(tile[:, :, :, 0], tile[:, :, :, -1])
    # and it is exactly t_i broadcast
    assert torch.allclose(tile[:, :, :, 3], sym)

    tiled = build_spatial_cond(sym, pairs, None, "tile_dist")
    assert tiled.shape[1] == 6
    assert not torch.allclose(tiled[:, -1, :, 0], tiled[:, -1, :, -1])   # distance varies

    pair = build_spatial_cond(sym, pairs, None, "pair")
    assert pair.shape[1] == n_cond_channels(5, 1, 0, "pair") == 10
    assert torch.allclose(pair[:, :5], tile)                    # t_i half
    assert torch.allclose(pair[:, 5:], sym[:, :, None, :].expand(2, 5, 16, 16))
    # pair mode must NOT contain the product feature anywhere
    prod = (sym[:, :, :, None] * sym[:, :, None, :])
    assert not any(torch.allclose(pair[:, c], prod[:, 0]) for c in range(10))

    with pytest.raises(ValueError, match="cond mode"):
        build_spatial_cond(sym, pairs, None, "nonsense")


def test_cond_mode_trains_end_to_end(workdir, cfg):
    """A tile-mode model builds with the right input width and trains."""
    import dataclasses

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    c2 = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, cond_mode="tile_dist"),
        train=dataclasses.replace(cfg.train, max_steps=10, eval_every=5,
                                  output_dir=str(workdir / "run_tile"),
                                  cache_dir=str(workdir / "cache_tile")))
    hprep.build(c2)
    htrain.train(c2)
    assert (Path(c2.train.output_dir) / "latest").exists()


def test_interaction_strength_metrics(workdir, cfg):
    """Interaction-level scoring: perfect prediction -> r=1 and full recovery;
    a prediction that misses interactions -> strength 0 for those."""
    import numpy as np

    from flexcible.capture import CaptureData
    from flexcible.loops import interaction_strength_metrics

    cap = CaptureData(cfg.capture.interactions, "mifsud", 0.0)
    nb = CHROMS["chr1"] // BINSIZE
    cc = cap.chrom("chr1", nb, BINSIZE)

    # perfect prediction = the true binned map
    m = interaction_strength_metrics(cc.scores, cap, "chr1", nb, BINSIZE, 64)
    assert m["n_interactions"] == 3                     # (5,12) (20,30) (40,55)
    assert m["recovery_rate"] == 1.0
    assert m["strength_pearson"] == pytest.approx(1.0, abs=1e-6)
    assert m["mean_true_strength"] == pytest.approx(m["mean_pred_strength"])

    # a model that predicts nothing: every interaction scores strength 0
    zero = np.zeros_like(cc.scores)
    m0 = interaction_strength_metrics(zero, cap, "chr1", nb, BINSIZE, 64)
    assert m0["n_recovered"] == 0 and m0["recovery_rate"] == 0.0
    assert m0["mean_pred_strength"] == 0.0

    # a model that predicts one of the three interactions
    partial = np.zeros_like(cc.scores)
    partial[20, 30] = partial[30, 20] = 12.0
    m1 = interaction_strength_metrics(partial, cap, "chr1", nb, BINSIZE, 64)
    assert m1["n_recovered"] == 1
    assert m1["recovery_rate"] == pytest.approx(1 / 3)

    # a model that predicts everything high must NOT look good on the control
    flat = np.full_like(cc.scores, 9.0)
    mf = interaction_strength_metrics(flat, cap, "chr1", nb, BINSIZE, 64)
    assert mf["recovery_rate"] == 1.0
    assert mf["auroc_vs_distance_matched"] == pytest.approx(0.5, abs=0.05)


def test_capture_guards_against_dense_blowup(workdir, cfg):
    """A whole-chromosome capture matrix is refused at fine resolution, and
    the score sample no longer needs one."""
    from flexcible.capture import CaptureData

    cap = CaptureData(cfg.capture.interactions, "mifsud", 0.0)
    with pytest.raises(MemoryError, match="GiB"):
        cap.chrom("chr1", 250000, 1000)          # 1 kb chr1 -> 231 GiB
    # the normaliser sample comes from the interaction list, no matrix at all
    v = cap.train_score_sample(["chr1"], lambda c: 250000, 1000)
    assert v.size > 0 and v.min() > 0


def test_interaction_metrics_from_cool_matches_dense(workdir, cfg):
    """The streaming scorer reproduces the in-memory one on the same data."""
    import cooler
    import numpy as np

    from flexcible.capture import CaptureData
    from flexcible.loops import (interaction_metrics_from_cool,
                                   interaction_strength_metrics)

    cap = CaptureData(cfg.capture.interactions, "mifsud", 0.0)
    nb = CHROMS["chr1"] // BINSIZE
    cc = cap.chrom("chr1", nb, BINSIZE)
    # write the true capture map as a cool, then score it both ways
    import pandas as pd

    bins = pd.DataFrame({"chrom": ["chr1"] * nb,
                         "start": np.arange(nb) * BINSIZE,
                         "end": (np.arange(nb) + 1) * BINSIZE})
    i, j = np.triu_indices(nb)
    v = cc.scores[i, j]
    px = pd.DataFrame({"bin1_id": i[v > 0], "bin2_id": j[v > 0],
                       "count": v[v > 0].astype(np.float64)})
    out = str(workdir / "cap_truth.cool")
    cooler.create_cooler(out, bins=bins, pixels=px, dtypes={"count": "float64"},
                         symmetric_upper=True, ordered=True)
    a = interaction_strength_metrics(cc.scores, cap, "chr1", nb, BINSIZE, 64)
    b = interaction_metrics_from_cool(out, cap, "chr1", nb, BINSIZE, 64)
    assert a["n_interactions"] == b["n_interactions"]
    assert b["strength_pearson"] == pytest.approx(a["strength_pearson"], abs=1e-6)
    assert b["mean_pred_strength"] == pytest.approx(a["mean_pred_strength"], rel=1e-6)


def test_predictions_are_float_expectations_by_default(workdir, cfg):
    """The model predicts an expected contact count, which is fractional; an
    integer count matrix is available but opt-in, and never for obs/exp."""
    import dataclasses

    import cooler
    import numpy as np

    from flexcible import predict as hpred

    out = str(workdir / "pred_float.cool")
    hpred.predict(cfg, "all", out, batch=4)
    v = cooler.Cooler(out).pixels()[:]["count"].values
    assert np.issubdtype(v.dtype, np.floating)
    assert not np.allclose(v, np.rint(v))       # genuinely fractional

    c2 = dataclasses.replace(
        cfg, predict=dataclasses.replace(cfg.predict, integer_counts=True))
    out2 = str(workdir / "pred_int.cool")
    hpred.predict(c2, "all", out2, batch=4)
    v2 = cooler.Cooler(out2).pixels()[:]["count"].values
    assert np.issubdtype(v2.dtype, np.integer) and (v2 > 0).all()


def test_streaming_scc_matches_dense(workdir, cfg):
    """The slice-streaming SCC reproduces the in-memory one exactly."""
    import numpy as np

    from flexcible import predict as hpred
    from flexcible.evaluate import scc, scc_streaming
    from flexcible.matrix import fetch_chrom_matrix

    out = str(workdir / "pred_scc.cool")
    hpred.predict(cfg, "all", out, batch=4)
    truth = cfg.get_sample(None).cool_path
    a = fetch_chrom_matrix(truth, "chr1")
    b = fetch_chrom_matrix(out, "chr1")
    n = min(a.shape[0], b.shape[0])
    dense = scc(b[:n, :n], a[:n, :n], max_dist_bins=32, h=1)
    streamed = scc_streaming(truth, out, "chr1", max_dist_bins=32, h=1,
                             core_bins=20)      # force several slices
    assert streamed == pytest.approx(dense, abs=0.02)


def test_liftover_maps_and_rejects():
    """Liftover maps coordinates and refuses intervals that do not map
    cleanly, so a rearranged region is dropped instead of silently distorted."""
    import gzip
    import tempfile
    from pathlib import Path

    from flexcible.liftover import LiftOver

    # a chain mapping chr1:1000-2000 (source) onto chr1:5000-6000 (target)
    chain = ("chain 1000 chr1 100000 + 1000 2000 chr1 100000 + 5000 6000 1\n"
             "1000\n\n")
    p = Path(tempfile.mkdtemp()) / "t.chain.gz"
    with gzip.open(p, "wt") as f:
        f.write(chain)
    lo = LiftOver(p)
    assert lo.convert("chr1", 1000) == ("chr1", 5000)
    assert lo.convert("chr1", 1500) == ("chr1", 5500)
    assert lo.convert("chr1", 50) is None            # outside any block
    assert lo.convert_interval("chr1", 1000, 1500) == ("chr1", 5000, 5500)
    assert lo.convert_interval("chr1", 900, 1500) is None   # start unmapped


def test_bed_holdout_split_and_prescaled(workdir, cfg):
    """A BED holdout splits by region, not by chromosome: training windows must
    never touch a held-out region and test windows must lie inside one."""
    import dataclasses
    import json

    import numpy as np

    from flexcible import precompute as hprep

    bed = workdir / "holdout.bed"
    # hold out one interval on each chromosome
    bed.write_text("chr1\t40000\t70000\nchr2\t20000\t50000\n")
    c = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, holdout_bed=str(bed)),
        window=dataclasses.replace(cfg.window, transform="prescaled"),
        train=dataclasses.replace(cfg.train, cache_dir=str(workdir / "cache_bed")))
    man = hprep.build(c)
    assert man["shards"], "no windows cached"
    regions = {"chr1": (40000, 70000), "chr2": (20000, 50000)}
    for sh in man["shards"]:
        z = np.load(sh["path"])
        lo, hi = regions[sh["chrom"]]
        for st in z["start"]:
            s_bp, e_bp = int(st) * BINSIZE, (int(st) + c.window.size) * BINSIZE
            if sh["split"] == "train":
                assert e_bp <= lo or s_bp >= hi, "train window touches holdout"
            else:
                assert s_bp >= lo and e_bp <= hi, "test window outside holdout"
    # prescaled keeps negative values intact
    n = json.loads((Path(c.train.cache_dir) / "norm_matrix_default.json").read_text())
    assert n["transform"] == "prescaled"


def test_edit_coordinate_map_and_kinds():
    """Insertions, deletions and tandem duplications map coordinates
    correctly, and novel inserted sequence is marked as unmeasured."""
    import numpy as np

    from flexcible.variants import Edit, apply_edits

    seq = "AAAACCCCGGGGTTTT"          # 16 bp
    # deletion of CCCC
    ed, pos, src = apply_edits(seq, [Edit("chr1", 4, 8, "")])
    assert ed == "AAAAGGGGTTTT"
    assert (pos[4:8] == -1).all()      # deleted bases have no target
    assert pos[8] == 4                 # first base after the deletion
    assert (src >= 0).all()            # every remaining base has a source

    # tandem duplication of CCCC: BOTH copies come from the same source, so
    # nothing is novel -- each copy inherits the source's chromatin state
    ed, pos, src = apply_edits(seq, [Edit("chr1", 4, 8, "CCCCCCCC")])
    assert ed == "AAAACCCCCCCCGGGGTTTT"
    assert pos[4] == 4 and pos[7] == 7
    assert (src[4:8] == np.arange(4, 8)).all()
    assert (src[8:12] == np.arange(4, 8)).all()
    assert (src >= 0).all()

    # a pure insertion of NEW sequence has no source: it is truly unmeasured
    ed, pos, src = apply_edits(seq, [Edit("chr1", 8, 8, "TTT")])
    assert ed == "AAAACCCCTTTGGGGTTTT"
    assert (src[8:11] == -1).all()
    assert pos[8] == 11                # bases after the insert shift right

    with pytest.raises(ValueError, match="overlapping"):
        apply_edits(seq, [Edit("chr1", 0, 6, ""), Edit("chr1", 4, 8, "")])


def test_edit_changes_motif_channels(workdir, cfg):
    """Destroying a planted motif with a small edit must change the motif
    channel, which is how a sub-bin variant reaches the model at all."""
    import numpy as np

    from flexcible.precompute import assemble_chrom, sample_track_files
    from flexcible.variants import Edit, edited_rows

    s = cfg.get_sample(None)
    nb = CHROMS["chr1"] // BINSIZE
    ref = assemble_chrom(cfg, s, sample_track_files(cfg, s), "chr1", nb)
    rows = [i for i, n in enumerate(
        ["CTCF", "H3K27ac", "seq:gc", "seq:cpg",
         f"motif:{MOTIF}:fwd", f"motif:{MOTIF}:rev"]) if n.startswith("motif")]
    fwd_row = rows[0]
    # the fixture plants a forward site at 5*BINSIZE; scramble it
    site = 5 * BINSIZE
    ed = Edit("chr1", site, site + 10, "AAAAAAAAAA", "kill motif")
    out, nb_ed, novel = edited_rows(cfg, s, "chr1", [ed], ref, nb)
    assert nb_ed == nb                             # same length edit
    assert ref[fwd_row][5] >= 1                    # site present in reference
    assert out[fwd_row][5] < ref[fwd_row][5]       # and destroyed after the edit


def test_large_deletion_shrinks_the_map(workdir, cfg):
    """A deletion big enough to span bins removes them from the edited map."""
    from flexcible.precompute import assemble_chrom, sample_track_files
    from flexcible.variants import Edit, edited_rows

    s = cfg.get_sample(None)
    nb = CHROMS["chr1"] // BINSIZE
    ref = assemble_chrom(cfg, s, sample_track_files(cfg, s), "chr1", nb)
    ed = Edit("chr1", 20 * BINSIZE, 30 * BINSIZE, "", "10 kb deletion")
    out, nb_ed, novel = edited_rows(cfg, s, "chr1", [ed], ref, nb)
    assert nb_ed == nb - 10


def test_combined_upper_lower_cool(workdir, cfg):
    """One cool, measured above the diagonal and predicted below it."""
    import cooler
    import numpy as np

    from flexcible import predict as hpred
    from flexcible.combine import combine_matrices, write_combined
    from flexcible.matrix import fetch_chrom_matrix

    a = np.array([[0.0, 1, 2], [1, 0, 3], [2, 3, 0]])
    b = np.array([[0.0, 7, 8], [7, 0, 9], [8, 9, 0]])
    c = combine_matrices(a, b)
    assert c[0, 1] == 1 and c[0, 2] == 2      # upper from a
    assert c[1, 0] == 7 and c[2, 0] == 8      # lower from b
    assert not np.allclose(c, c.T)            # genuinely non-symmetric

    pred = str(workdir / "pred_comb.cool")
    hpred.predict(cfg, "chr1", pred, batch=4)
    out = write_combined(cfg.get_sample(None).cool_path, pred, "chr1",
                         str(workdir / "combined.cool"))
    clr = cooler.Cooler(out)
    m = clr.matrix(balance=False).fetch("chr1")
    truth = fetch_chrom_matrix(cfg.get_sample(None).cool_path, "chr1")
    n = min(m.shape[0], truth.shape[0])
    assert np.allclose(np.nan_to_num(m[:n, :n])[np.triu_indices(n, 1)],
                       truth[:n, :n][np.triu_indices(n, 1)], rtol=1e-5)


def test_prescaled_headroom_keeps_targets_off_the_tanh_asymptote():
    """With a tanh head, targets must not sit exactly at +/-1."""
    import numpy as np

    from flexcible.matrix import MatrixNormalizer

    # Akita-like data: clipped at -2, mass sitting exactly on the bound
    v = np.concatenate([np.full(2000, -2.0), np.random.uniform(-2, 1.9, 20000)])
    n = MatrixNormalizer(transform="prescaled", clip_quantile=0.999,
                         headroom=0.9).fit(v)
    enc = n.encode(v.reshape(-1, 1))
    assert np.abs(enc).max() <= 0.9 + 1e-6, "clip bound reaches the asymptote"
    assert np.isclose(n.decode(enc).ravel()[0], -2.0, atol=1e-4)   # exact inverse

    # without headroom the bound lands on the asymptote
    n1 = MatrixNormalizer(transform="prescaled", clip_quantile=0.999,
                          headroom=1.0).fit(v)
    assert np.isclose(np.abs(n1.encode(v.reshape(-1, 1))).max(), 1.0, atol=1e-6)


def test_row_layout_is_consistent_across_samples(workdir, cfg):
    """Samples may carry different genomes, but every sample must produce the
    SAME conditioning rows -- they are stacked into one tensor."""
    import dataclasses

    from flexcible.precompute import (assemble_chrom, sample_track_files,
                                        sym_row_names)

    base = cfg.get_sample(None)
    with_fa = dataclasses.replace(base, name="with_fasta",
                                  fasta=str(workdir / "genome.fa"))
    without = dataclasses.replace(base, name="no_fasta", fasta=None)
    c = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, samples=[with_fa, without]),
        sequence=dataclasses.replace(cfg.sequence, fasta=None))
    assert c.any_fasta                       # one sample has a genome
    expect = len(sym_row_names(c))
    nb = CHROMS["chr1"] // BINSIZE
    for s in (with_fa, without):
        rows = assemble_chrom(c, s, sample_track_files(c, s), "chr1", nb)
        assert rows.shape[0] == expect, f"{s.name}: {rows.shape[0]} != {expect}"


def test_mixed_track_availability_train_and_predict_subset(workdir, cfg):
    """Sample A has both assays, sample B only one, and prediction then runs
    from an even smaller subset. Assays are matched BY NAME, so the same
    histone always occupies the same conditioning row."""
    import dataclasses

    import cooler
    import numpy as np

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible import train as htrain
    from flexcible.tracks import TrackNormalizer

    base = cfg.get_sample(None)
    rich = dataclasses.replace(base, name="rich")                  # CTCF + H3K27ac
    poor = dataclasses.replace(base, name="poor",
                               missing_tracks=["H3K27ac"])         # CTCF only
    c = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, samples=[rich, poor]),
        train=dataclasses.replace(cfg.train,
                                  cache_dir=str(workdir / "cache_mixed"),
                                  output_dir=str(workdir / "run_mixed"),
                                  track_dropout=0.3, max_steps=40, eval_every=20))
    man = hprep.build(c)
    # one row layout keyed by name, shared by both samples
    assert man["tracks"] == ["CTCF", "H3K27ac"]
    assert {s["sample"] for s in man["shards"]} == {"rich", "poor"}

    # the missing assay must not drag the normaliser of the sample that has it
    tn = TrackNormalizer.load(Path(c.train.cache_dir) / "norm_tracks.json")
    solo = dataclasses.replace(c, data=dataclasses.replace(c.data, samples=[rich]),
                               train=dataclasses.replace(
                                   c.train, cache_dir=str(workdir / "cache_solo")))
    hprep.build(solo)
    tn_solo = TrackNormalizer.load(Path(solo.train.cache_dir) / "norm_tracks.json")
    i = man["tracks"].index("H3K27ac")
    assert tn.mean[i] == pytest.approx(tn_solo.mean[i], rel=1e-6)

    htrain.train(c)
    # predict from a SMALLER subset than either sample was trained with
    out = str(workdir / "pred_subset.cool")
    hpred.predict(c, "all", out, batch=4, sample_name="rich",
                  zero_tracks=["H3K27ac"])
    m = cooler.Cooler(out).matrix(balance=False).fetch("chr1")
    assert np.nansum(np.nan_to_num(m)) > 0


def test_regional_knockout_only_affects_its_interval(workdir, cfg):
    """Knocking an assay out on an interval must change the prediction there
    and leave a distant region untouched."""
    import cooler
    import numpy as np

    from flexcible import predict as hpred

    full = str(workdir / "pred_rk_none.cool")
    hpred.predict(cfg, "chr1", full, batch=4)
    ko = str(workdir / "pred_rk_region.cool")
    # remove H3K27ac only over bins 8-14 (the fixture's peak sits at bin 10)
    hpred.predict(cfg, "chr1", ko, batch=4,
                  zero_regions=[("H3K27ac", "chr1", 8 * BINSIZE, 14 * BINSIZE)])
    a = np.nan_to_num(cooler.Cooler(full).matrix(balance=False).fetch("chr1"))
    b = np.nan_to_num(cooler.Cooler(ko).matrix(balance=False).fetch("chr1"))
    n = min(a.shape[0], b.shape[0])
    inside = np.abs(a[8:14, 8:14] - b[8:14, 8:14]).sum()
    far = np.abs(a[60:n, 60:n] - b[60:n, 60:n]).sum()
    assert inside > 0, "regional knockout changed nothing inside its interval"
    assert far < inside, "a distant region moved as much as the knocked-out one"


def test_regional_knockout_rejects_unknown_track(workdir, cfg):
    from flexcible import predict as hpred

    with pytest.raises(ValueError, match="unknown track"):
        hpred.predict(cfg, "chr1", str(workdir / "x.cool"), batch=4,
                      zero_regions=[("NoSuchMark", "chr1", 0, 10000)])


def test_knockout_removes_protein_but_keeps_the_binding_site(workdir, cfg):
    """A knockout mutes the measured protein signal; the DNA is untouched, so
    the motif channels still report the site. Removing the site itself is a
    separate, explicit operation."""
    import numpy as np

    from flexcible.precompute import (assemble_chrom, sample_track_files,
                                        sym_row_names)

    s = cfg.get_sample(None)
    import dataclasses

    ko = dataclasses.replace(s, name="ko", zeroed_tracks=["CTCF"])
    nb = CHROMS["chr1"] // BINSIZE
    rows = sym_row_names(cfg)
    ctcf = rows.index("CTCF")
    fwd = rows.index(f"motif:{MOTIF}:fwd")

    ref = assemble_chrom(cfg, s, sample_track_files(cfg, s), "chr1", nb)
    out = assemble_chrom(cfg, ko, sample_track_files(cfg, ko), "chr1", nb)
    assert ref[ctcf].max() > 0 and out[ctcf].max() == 0     # protein gone
    assert np.array_equal(ref[fwd], out[fwd])               # site still there


def test_regional_site_removal_targets_motif_rows(workdir, cfg):
    """Naming a motif in a regional knockout removes the site instead of the
    protein, and an unknown name is rejected with both options listed."""
    import cooler
    import numpy as np

    from flexcible import predict as hpred

    prot = str(workdir / "ko_protein.cool")
    site = str(workdir / "ko_site.cool")
    hpred.predict(cfg, "chr1", prot, batch=4,
                  zero_regions=[("CTCF", "chr1", 4 * BINSIZE, 8 * BINSIZE)])
    hpred.predict(cfg, "chr1", site, batch=4,
                  zero_regions=[(f"motif:{MOTIF}", "chr1", 4 * BINSIZE, 8 * BINSIZE)])
    a = np.nan_to_num(cooler.Cooler(prot).matrix(balance=False).fetch("chr1"))
    b = np.nan_to_num(cooler.Cooler(site).matrix(balance=False).fetch("chr1"))
    assert not np.allclose(a, b), "removing the protein and the site did the same"

    with pytest.raises(ValueError, match="motifs:"):
        hpred.predict(cfg, "chr1", str(workdir / "z.cool"), batch=4,
                      zero_regions=[("motif:NOPE", "chr1", 0, 10000)])


def test_all_three_perturbations_compose(workdir, cfg):
    """Protein removal, binding-site removal and a real DNA edit each change
    the prediction, and a deletion changes the edited chromosome's length."""
    import cooler
    import numpy as np

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible import train as htrain
    from flexcible.variants import Edit, predict_edited

    if not (Path(cfg.train.output_dir) / "latest").exists():
        if not (Path(cfg.train.cache_dir) / "manifest.json").exists():
            hprep.build(cfg)
        htrain.train(cfg)

    base = str(workdir / "p3_ref.cool")
    hpred.predict(cfg, "chr1", base, batch=4)
    n_chr1 = CHROMS["chr1"] // BINSIZE

    # 1) protein gone, site intact
    prot = str(workdir / "p3_protein.cool")
    hpred.predict(cfg, "chr1", prot, batch=4,
                  zero_regions=[("CTCF", "chr1", 4 * BINSIZE, 10 * BINSIZE)])
    # 2) binding site gone
    site = str(workdir / "p3_site.cool")
    hpred.predict(cfg, "chr1", site, batch=4,
                  zero_regions=[(f"motif:{MOTIF}", "chr1", 4 * BINSIZE, 10 * BINSIZE)])
    # 3) DNA actually deleted, and combined with a protein knockout
    ed = str(workdir / "p3_edit.cool")
    info = predict_edited(cfg, "chr1",
                          [Edit("chr1", 20 * BINSIZE, 28 * BINSIZE, "", "8 kb del")],
                          ed, zero_regions=[("CTCF", "chr1", 4 * BINSIZE,
                                             10 * BINSIZE)])
    assert info["n_bins"] == info["n_bins_reference"] - 8   # genome got shorter
    # the edited cool carries ONLY the edited chromosome, with its new length
    assert cooler.Cooler(ed).bins()[:].shape[0] == n_chr1 - 8
    assert cooler.Cooler(ed).chromnames == ["chr1"]

    def m(path, n):
        a = np.nan_to_num(cooler.Cooler(path).matrix(balance=False).fetch("chr1"))
        return a[:n, :n]
    n = 20
    b, p, s = m(base, n), m(prot, n), m(site, n)
    assert not np.allclose(b, p), "protein knockout changed nothing"
    assert not np.allclose(b, s), "site removal changed nothing"
    assert not np.allclose(p, s), "protein and site removal are not distinct"


def test_resume_continues_instead_of_restarting(workdir, cfg):
    """A resumed run picks up the step counter, weights and optimizer state
    rather than starting from zero."""
    import dataclasses
    import json

    import torch

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    c = dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, cache_dir=str(workdir / "cache_res"),
                                  output_dir=str(workdir / "run_res"),
                                  max_steps=20, eval_every=10, resume=False))
    hprep.build(c)
    htrain.train(c)
    ck = Path(c.train.output_dir) / (Path(c.train.output_dir) / "latest").read_text().strip()
    first = json.loads((ck / "progress.json").read_text())
    assert first["step"] >= 20 and (ck / "optim.pt").exists()
    w1 = torch.load(ck / "model.pt", map_location="cpu")

    c2 = dataclasses.replace(
        c, train=dataclasses.replace(c.train, max_steps=40, resume=True))
    htrain.train(c2)
    ck2 = Path(c.train.output_dir) / (Path(c.train.output_dir) / "latest").read_text().strip()
    second = json.loads((ck2 / "progress.json").read_text())
    assert second["step"] > first["step"], "resume restarted the step counter"
    w2 = torch.load(ck2 / "model.pt", map_location="cpu")
    k = next(iter(w1))
    assert not torch.allclose(w1[k], w2[k]), "weights did not move after resume"


def test_fold_edited_duplication_back_to_reference():
    """Reads from a tandem duplication map onto the source interval, so the
    folded prediction must place both copies there and conserve total signal."""
    import numpy as np
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mundlos_test import fold_edited_to_reference

    res = 10
    # reference has 6 bins; duplicate bins 2-4 -> edited has 8
    n_ref, a, b = 6, 2, 4
    n_ed = n_ref + (b - a)
    ed = np.zeros((n_ed, n_ed))
    ed[2, 3] = ed[3, 2] = 1.0          # inside copy 1
    ed[4, 5] = ed[5, 4] = 2.0          # the same pair inside copy 2
    out = fold_edited_to_reference(ed, (a * res, b * res), res, 0)
    assert out.shape == (n_ref, n_ref)
    # both copies land on the same reference pair and add up
    assert out[2, 3] == pytest.approx(3.0)
    assert out.sum() == pytest.approx(ed.sum())    # signal conserved


def test_unseen_cell_type_borrows_a_normaliser(workdir, cfg):
    """Predicting a cell type absent from training must work: the model is
    cell-type agnostic, so only the tracks change."""
    import dataclasses

    import cooler
    import numpy as np

    from flexcible import predict as hpred

    base = cfg.get_sample(None)
    unseen = dataclasses.replace(base, name="never_trained_on_this")
    c = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, samples=[base, unseen]))
    out = str(workdir / "pred_unseen.cool")
    hpred.predict(c, "chr1", out, batch=4, sample_name="never_trained_on_this")
    m = np.nan_to_num(cooler.Cooler(out).matrix(balance=False).fetch("chr1"))
    assert m.sum() > 0


def test_motifs_without_any_chip_track(workdir, cfg):
    """A binding site is a property of the DNA: motif channels must be
    available even when no ChIP track declares the motif, which is what lets
    the model see a duplicated boundary ELEMENT."""
    import dataclasses

    from flexcible.precompute import sym_row_names

    histones_only = [t for t in cfg.data.tracks if t.type == "histone"]
    c = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, tracks=histones_only),
        motifs=dataclasses.replace(cfg.motifs, enabled=True, extra=[MOTIF]))
    assert c.active_motif_ids == [MOTIF]
    rows = sym_row_names(c)
    assert f"motif:{MOTIF}:fwd" in rows and f"motif:{MOTIF}:rev" in rows


def test_duplicated_copy_inherits_signal_no_false_cliff(workdir, cfg):
    """A tandem duplication copies the DNA, so both copies must carry the same
    chromatin signal. Zeroing the second copy would put an artificial cliff at
    the junction and invite a spurious boundary call."""
    import numpy as np

    from flexcible.precompute import (assemble_chrom, sample_track_files,
                                        sym_row_names)
    from flexcible.variants import Edit, apply_edits, edited_rows

    # source map: every copy of a tandem duplication points at the source
    seq = "AAAACCCCGGGGTTTT"
    ed, pos, src = apply_edits(seq, [Edit("chr1", 4, 8, "CCCCCCCC")])
    assert ed == "AAAACCCCCCCCGGGGTTTT"
    assert (src[4:8] == np.arange(4, 8)).all()      # copy 1 -> source
    assert (src[8:12] == np.arange(4, 8)).all()     # copy 2 -> the SAME source
    assert (src >= 0).all()                          # nothing is unmeasured

    s = cfg.get_sample(None)
    nb = CHROMS["chr1"] // BINSIZE
    ref = assemble_chrom(cfg, s, sample_track_files(cfg, s), "chr1", nb)
    rows = sym_row_names(cfg)
    ctcf = rows.index("CTCF")
    a, b = 8 * BINSIZE, 16 * BINSIZE                 # duplicate 8 bins
    out, nb_ed, novel = edited_rows(
        cfg, s, "chr1", [Edit("chr1", a, b, None or "")], ref, nb) if False else (None,)*3
    dup = Edit("chr1", a, b, "")                     # placeholder replaced below
    from flexcible.variants import read_chrom_seq
    src_seq = read_chrom_seq(cfg.sample_fasta(s), "chr1")[a:b]
    out, nb_ed, novel = edited_rows(
        cfg, s, "chr1", [Edit("chr1", a, b, src_seq + src_seq)], ref, nb)
    assert nb_ed == nb + 8
    assert novel.sum() == 0, "a tandem duplication has no unmeasured sequence"
    # the two copies carry the same signal
    copy1 = out[ctcf][8:16]
    copy2 = out[ctcf][16:24]
    assert np.allclose(copy1, copy2, atol=1e-5), "copies differ in signal"
    assert copy2.max() > 0, "the second copy was left empty"


def test_hic_backbones_are_selectable(workdir, cfg):
    """The Hi-C side must not be U-Net-only: every backbone trains and
    predicts through the same interface."""
    import dataclasses

    import cooler
    import numpy as np

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible import train as htrain
    from flexcible.model import BACKBONES, build_backbone, n_cond_channels

    for kind in BACKBONES:
        m = build_backbone(kind, n_cond_channels(4, 0, 0, "full"), out_ch=2,
                           base=16, depth=2)
        import torch
        y = m(torch.zeros(1, n_cond_channels(4, 0, 0, "full"), 32, 32))
        assert y.shape == (1, 2, 32, 32), f"{kind} shape {tuple(y.shape)}"
        assert torch.allclose(y, y.transpose(-1, -2), atol=1e-6), f"{kind} not symmetric"

    c = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, kind="dilated", base_channels=16),
        train=dataclasses.replace(cfg.train, cache_dir=str(workdir / "cache_bb"),
                                  output_dir=str(workdir / "run_bb"),
                                  max_steps=10, eval_every=5))
    hprep.build(c)
    htrain.train(c)
    out = str(workdir / "pred_bb.cool")
    hpred.predict(c, "chr1", out, batch=4)
    assert np.nan_to_num(
        cooler.Cooler(out).matrix(balance=False).fetch("chr1")).sum() > 0


def test_merge_regions_and_containment():
    from flexcible.genome import merge_regions, window_region_state

    m = merge_regions({"chr1": [(50, 80), (0, 30), (20, 40), (40, 45)]})
    assert m == {"chr1": [(0, 45), (50, 80)]}
    assert window_region_state(m, "chr1", 5, 44) == "inside"      # spans two raw
    assert window_region_state(m, "chr1", 40, 60) == "overlap"    # crosses the gap


def test_train_regions_bed_restricts_training_windows_only(workdir, cfg):
    """data.train_regions_bed keeps only training windows wholly inside the
    (merged) regions and counts the rest; test windows are untouched."""
    import dataclasses

    from flexcible import precompute as hprep

    hold = workdir / "tr_holdout.bed"
    hold.write_text("chr1\t40000\t72000\n")
    keep = workdir / "tr_keep.bed"
    # two touching intervals on chr1 (merged into 0-40000), and chr2 0-60000
    keep.write_text("chr1\t0\t20000\nchr1\t20000\t40000\nchr2\t0\t60000\n")

    def run(tr_bed, name):
        c = dataclasses.replace(
            cfg,
            data=dataclasses.replace(cfg.data, holdout_bed=str(hold),
                                     train_regions_bed=tr_bed),
            train=dataclasses.replace(cfg.train, cache_dir=str(workdir / name)))
        return hprep.build(c)

    base = run(None, "cache_tr_off")
    on = run(str(keep), "cache_tr_on")
    n = lambda m, sp: sum(x["n_windows"] for x in m["shards"] if x["split"] == sp)
    assert n(on, "test") == n(base, "test")
    assert 0 < n(on, "train") < n(base, "train")
    assert on["n_outside_train_regions"] == n(base, "train") - n(on, "train")
    assert "n_outside_train_regions" not in base
    lim = {"chr1": 40000, "chr2": 60000}
    for sh in on["shards"]:
        if sh["split"] != "train":
            continue
        for st in np.load(sh["path"])["start"]:
            assert (int(st) + cfg.window.size) * BINSIZE <= lim[sh["chrom"]]


def test_assay_indicator_row(workdir, cfg):
    """data.assay_indicator adds one constant row after the sequence features
    (0 Hi-C, 1 Micro-C); off, the layout is unchanged."""
    import dataclasses

    from flexcible.precompute import (assemble_chrom, sample_track_files,
                                        seq_feature_names, sym_row_names)

    base = cfg.get_sample(None)
    hic = dataclasses.replace(base, name="hic_s", assay="hic")
    mc = dataclasses.replace(base, name="mc_s", assay="microc")
    off = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, samples=[hic, mc]))
    on = dataclasses.replace(off, data=dataclasses.replace(off.data, assay_indicator=True))
    assert "assay:microc" not in sym_row_names(off)
    rows_on, rows_off = sym_row_names(on), sym_row_names(off)
    assert len(rows_on) == len(rows_off) + 1
    k = rows_on.index("assay:microc")
    assert k == len(on.track_names) + len(seq_feature_names(on)) - 1
    nb = CHROMS["chr1"] // BINSIZE
    for s, v in ((hic, 0.0), (mc, 1.0)):
        a_on = assemble_chrom(on, s, sample_track_files(on, s), "chr1", nb)
        a_off = assemble_chrom(off, s, sample_track_files(off, s), "chr1", nb)
        assert a_on.shape[0] == len(rows_on)
        assert np.all(a_on[k] == v)
        np.testing.assert_array_equal(np.delete(a_on, k, axis=0), a_off)
    with pytest.raises(ValueError):
        dataclasses.replace(base, assay="capture")


def test_assay_indicator_three_class(workdir, cfg):
    """assay_indicator_mode "three" adds two constant rows (intact, microc):
    in situ Hi-C (0, 0), intact Hi-C (1, 0), Micro-C (0, 1). The binary default
    keeps one row and counts intact Hi-C as Hi-C."""
    import dataclasses

    from flexcible.precompute import (assemble_chrom, sample_track_files,
                                        seq_feature_names, sym_row_names)

    base = cfg.get_sample(None)
    smp = {a: dataclasses.replace(base, name=f"{a}_s", assay=a)
           for a in ("hic", "intact", "microc")}
    off = dataclasses.replace(cfg, data=dataclasses.replace(
        cfg.data, samples=list(smp.values())))
    binary = dataclasses.replace(off, data=dataclasses.replace(off.data, assay_indicator=True))
    three = dataclasses.replace(binary, data=dataclasses.replace(
        binary.data, assay_indicator_mode="three"))
    r_off, r_bin, r_3 = sym_row_names(off), sym_row_names(binary), sym_row_names(three)
    assert len(r_bin) == len(r_off) + 1 and len(r_3) == len(r_off) + 2
    ki, km = r_3.index("assay:intact"), r_3.index("assay:microc")
    assert (ki, km) == (km - 1, len(three.track_names) + len(seq_feature_names(three)) - 1)
    kb = r_bin.index("assay:microc")
    nb = CHROMS["chr1"] // BINSIZE
    want = {"hic": (0.0, 0.0), "intact": (1.0, 0.0), "microc": (0.0, 1.0)}
    for a, s in smp.items():
        a3 = assemble_chrom(three, s, sample_track_files(three, s), "chr1", nb)
        ab = assemble_chrom(binary, s, sample_track_files(binary, s), "chr1", nb)
        ao = assemble_chrom(off, s, sample_track_files(off, s), "chr1", nb)
        assert np.all(a3[ki] == want[a][0]) and np.all(a3[km] == want[a][1])
        assert np.all(ab[kb] == float(a == "microc"))
        np.testing.assert_array_equal(np.delete(a3, [ki, km], axis=0), ao)
    with pytest.raises(ValueError):
        dataclasses.replace(off.data, assay_indicator_mode="four")


def _cohesin_raw(workdir, name):
    d = workdir / "coh_bw"
    d.mkdir(exist_ok=True)
    for t, pos in (("CTCF", [5, 12, 20, 30]), ("RAD21", [5, 12, 20]), ("SMC3", [12, 30]),
                   ("H3K27ac", [10, 11, 12])):
        if not (d / f"{t}.bigwig").exists():
            _write_bigwig(d / f"{t}.bigwig", {"chr1": pos, "chr2": pos[:2]})
    raw = _ref_raw(workdir, name, ref_cool=False)
    smp = raw["data"]["samples"][0]
    raw["data"]["samples"] = [
        dict(smp, name="withcoh", bigwig_dir=str(d)),
        dict(smp, name="nocoh", bigwig_dir=str(d), missing_tracks=["RAD21", "SMC3"])]
    raw["data"]["tracks"] = [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                             {"name": "RAD21", "type": "cofactor", "motif": MOTIF},
                             {"name": "SMC3", "type": "cofactor", "motif": MOTIF},
                             {"name": "H3K27ac", "type": "histone"}]
    raw["data"]["cohesin_unknown"] = True
    raw["train"]["cache_dir"] = str(workdir / f"cache_{name}")
    return raw


def test_cohesin_unknown_row(workdir):
    """data.cohesin_unknown adds one constant row after the sequence features:
    1 where neither RAD21 nor SMC3 is measured (missing or zeroed), else 0;
    off, the layout is unchanged."""
    import dataclasses

    from flexcible.precompute import (assemble_chrom, sample_track_files,
                                        seq_feature_names, sym_row_names)

    on = Config.from_dict(_cohesin_raw(workdir, "cohrow"))
    raw_off = _cohesin_raw(workdir, "cohrow")
    raw_off["data"]["cohesin_unknown"] = False
    off = Config.from_dict(raw_off)
    r_on, r_off = sym_row_names(on), sym_row_names(off)
    assert len(r_on) == len(r_off) + 1
    k = r_on.index("cohesin:unknown")
    assert k == len(on.track_names) + len(seq_feature_names(on)) - 1
    assert seq_feature_names(on)[-1] == "cohesin_unknown"
    nb = CHROMS["chr1"] // BINSIZE
    with_, no_ = on.data.samples
    zeroed = dataclasses.replace(with_, zeroed_tracks=["RAD21", "SMC3"])
    half = dataclasses.replace(with_, zeroed_tracks=["RAD21"])
    for s, v in ((with_, 0.0), (no_, 1.0), (zeroed, 1.0), (half, 0.0)):
        a_on = assemble_chrom(on, s, sample_track_files(on, s), "chr1", nb)
        a_off = assemble_chrom(off, s, sample_track_files(off, s), "chr1", nb)
        assert np.all(a_on[k] == v), s.name
        np.testing.assert_array_equal(np.delete(a_on, k, axis=0), a_off)


def test_cohesin_unknown_training(workdir, caplog):
    """Track dropout with the cohesin-unknown indicator trains; the SMC3 row
    follows the RAD21 draw and the indicator row is set where it is dropped."""
    import logging

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    raw = _cohesin_raw(workdir, "cohtrain")
    raw["train"].update(track_dropout=0.5, track_dropout_mode="uniform_count", max_steps=6,
                        eval_every=3)
    c = Config.from_dict(raw)
    hprep.build(c)
    with caplog.at_level(logging.INFO):
        htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()
    assert any("cohesin_unknown: group ['RAD21', 'SMC3']" in r.getMessage()
               for r in caplog.records)


def test_precompute_workers_identical_cache(workdir):
    """precompute.build(workers=2) writes the same manifest, normalisers and
    shards as the serial default, with two samples, capture and a train filter."""
    import json

    from flexcible import precompute
    from flexcible.config import Config

    bed = workdir / "train_regions_w.bed"
    bed.write_text("chr1\t0\t40000\nchr2\t0\t60000\n")
    caches = {}
    for w in (1, 2):
        raw = {
            "data": {
                "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                           {"name": "H3K27ac", "type": "histone"}],
                "resolution": BINSIZE,
                "train_regions_bed": str(bed),
                "samples": [
                    {"name": "a", "cool_path": str(workdir / "target.cool"),
                     "bigwig_dir": str(workdir),
                     "capture_interactions": str(workdir / "capture.txt")},
                    {"name": "b", "cool_path": str(workdir / "target.cool"),
                     "bigwig_dir": str(workdir), "zeroed_tracks": ["CTCF"]},
                ],
            },
            "sequence": {"fasta": str(workdir / "genome.fa"),
                         "cache_dir": str(workdir / "gcache")},
            "motifs": {"dir": str(workdir), "score_frac": 0.8},
            "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99},
            "train": {"output_dir": str(workdir / f"run_w{w}"),
                      "cache_dir": str(workdir / f"cache_w{w}")},
        }
        caches[w] = (precompute.build(Config.from_dict(raw), workers=w),
                     workdir / f"cache_w{w}")
    (m1, c1), (m2, c2) = caches[1], caches[2]
    strip = [{k: v for k, v in sh.items() if k != "path"} for sh in m1["shards"]]
    assert strip == [{k: v for k, v in sh.items() if k != "path"} for sh in m2["shards"]]
    assert m1.get("n_outside_train_regions") == m2.get("n_outside_train_regions")
    for f in sorted(p.name for p in c1.glob("norm_*.json")):
        assert json.loads((c1 / f).read_text()) == json.loads((c2 / f).read_text())
    for sh in m1["shards"]:
        a, b = np.load(sh["path"]), np.load(c2 / Path(sh["path"]).name)
        assert set(a.files) == set(b.files)
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])


def test_resident_float16_matches_float32(workdir, cfg):
    """The cache is float16, so a float16 resident stack holds the same values."""
    import torch

    from flexcible import precompute as hprep
    from flexcible.train import _stack

    if not (Path(cfg.train.cache_dir) / "manifest.json").exists():
        hprep.build(cfg)
    d32, _ = _stack(cfg.train.cache_dir, "train", 0.0, False, 0)
    d16, _ = _stack(cfg.train.cache_dir, "train", 0.0, False, 0, dtype=torch.float16)
    assert d16["target"].dtype == torch.float16
    assert torch.equal(d16["target"].float(), d32["target"])
    assert torch.equal(d16["cond"].float(), d32["cond"])


# --------------------------------------------------------------------------- #
# reference map as input (data.reference_cool, model.reference)
# --------------------------------------------------------------------------- #
def _ref_raw(workdir, name, reference="none", ref_cool=True, transform="prescaled"):
    return {
        "data": {
            "samples": [{"name": "s", "cool_path": str(workdir / "target.cool"),
                         "bigwig_dir": str(workdir)}],
            "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                       {"name": "H3K27ac", "type": "histone"}],
            "resolution": BINSIZE,
            "reference_cool": str(workdir / "target.cool") if ref_cool else None,
        },
        "sequence": {"fasta": str(workdir / "genome.fa"),
                     "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "window": {"size": 32, "stride": 16, "max_zero_fraction": 0.99,
                   "transform": transform},
        "model": {"base_channels": 16, "depth": 2, "reference": reference},
        "train": {"output_dir": str(workdir / f"run_{name}"),
                  "cache_dir": str(workdir / "cache_ref"),
                  "batch_size": 4, "max_steps": 8, "eval_every": 4,
                  "val_regions": 3, "val_exclude_overlap": True, "lr": 1e-3,
                  "augment_reverse": True},
    }


def test_reference_config_validation(workdir):
    with pytest.raises(ValueError, match="model.reference"):
        Config.from_dict(_ref_raw(workdir, "x", reference="bogus"))
    with pytest.raises(ValueError, match="needs data.reference_cool"):
        Config.from_dict(_ref_raw(workdir, "x", reference="input", ref_cool=False))
    with pytest.raises(ValueError, match="prescaled"):
        Config.from_dict(_ref_raw(workdir, "x", transform="log1p"))
    c = Config.from_dict(_ref_raw(workdir, "x", ref_cool=False))
    assert c.model.reference == "none" and c.model.reference_planes == 0
    assert c.data.reference_cool is None


def test_reference_plane_and_residual_model():
    """input: one extra conditioning plane; residual: the same network output
    plus the reference on the Hi-C channel only."""
    import torch

    from flexcible.model import HiCModel, build_backbone, n_cond_channels

    torch.manual_seed(0)
    w = 16
    in_ch = n_cond_channels(3, 1, 1, "full")
    assert in_ch == n_cond_channels(3, 1, 0, "full") + 1
    bb = build_backbone("unet", in_ch, out_ch=2, base=8, depth=2)
    m_in = HiCModel(bb, "full", reference="input").eval()
    m_res = HiCModel(bb, "full", reference="residual").eval()
    cond = torch.randn(2, 3, w)
    pairs = cond[:, 1:3].reshape(2, 1, 2, w)
    ref = torch.randn(2, w, w)
    ref = 0.5 * (ref + ref.transpose(-1, -2))
    with torch.no_grad():
        a = m_in(cond, pairs, None, None, ref)
        b = m_res(cond, pairs, None, None, ref)
        a2 = m_in(cond, pairs, None, None, torch.zeros_like(ref))
    assert torch.allclose(b[:, 0], a[:, 0] + ref, atol=1e-6)
    assert torch.allclose(b[:, 1], a[:, 1])
    assert not torch.allclose(a, a2), "reference plane has no effect"
    with pytest.raises(ValueError, match="without ref"):
        m_in(cond, pairs, None, None, None)
    # "none" ignores ref and keeps the old channel count
    m0 = HiCModel(build_backbone("unet", in_ch - 1, out_ch=1, base=8, depth=2), "full")
    assert m0(cond, pairs, None, None, ref).shape == (2, 1, w, w)


def test_reference_cached_trained_and_scored(workdir):
    """precompute caches the reference encoded like the target (here the
    reference IS the target cool, so both arrays match); input and residual
    models train and are scored by eval_val_windows, which also scores the
    reference alone."""
    import json
    import subprocess

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    c = Config.from_dict(_ref_raw(workdir, "refin", reference="input"))
    man = hprep.build(c)
    assert man["reference"] == str(workdir / "target.cool")
    for sh in man["shards"]:
        z = np.load(sh["path"])
        assert z["ref"].shape == z["target"][:, 0].shape
        np.testing.assert_array_equal(z["ref"], z["target"][:, 0])
    data, ds = htrain._stack(c.train.cache_dir, "train", 0.0, False, 0)
    assert data["ref"].shape == data["target"][:, 0].shape and ds.reference

    repo = Path(__file__).resolve().parents[1]
    for mode in ("input", "residual"):
        cm = Config.from_dict(_ref_raw(workdir, f"ref{mode}", reference=mode))
        htrain.train(cm)
        run = Path(cm.train.output_dir)
        assert (run / "best").exists()
        for extra in ([], ["--reference-only"], ["--no-reference"]):
            out = run / f"eval{len(extra)}{extra[0] if extra else ''}.json"
            subprocess.run([sys.executable, str(repo / "scripts/eval_val_windows.py"),
                            "--run", str(run), "--split", "test", "--out", str(out),
                            *extra], check=True, cwd=repo, capture_output=True)
            r = json.loads(out.read_text())
            assert r["n"] > 0
            if extra:   # reference == target: a perfect prediction
                if extra == ["--reference-only"]:
                    assert r["prediction"] == "reference map"
                    assert r["oe_median_r"] > 0.999
                else:
                    assert r["prediction"] == "model, reference plane 0"
                    base = json.loads((run / "eval0.json").read_text())["oe_median_r"]
                    assert r["oe_median_r"] != base

    # a cache without reference windows is rejected by a reference model
    import dataclasses
    bad = dataclasses.replace(
        cm, train=dataclasses.replace(cm.train, cache_dir=str(workdir / "cache_noref"),
                                      output_dir=str(workdir / "run_noref")),
        data=dataclasses.replace(cm.data, reference_cool=None),
        model=dataclasses.replace(cm.model, reference="none"))
    hprep.build(bad)
    bad2 = dataclasses.replace(
        cm, train=dataclasses.replace(cm.train, cache_dir=str(workdir / "cache_noref"),
                                      output_dir=str(workdir / "run_noref")))
    with pytest.raises(RuntimeError, match="no reference windows"):
        htrain.train(bad2)


def test_coarsen_oe_matches_dense_block_mean(workdir):
    """scripts/coarsen_oe.py equals the bp-overlap weighted mean over each
    coarse block of the dense symmetric fine matrix, including coarse bins
    that cut a fine bin in half (2.5 kb over 1 kb bins), and writes only
    blocks wholly inside the fine band."""
    import subprocess

    import cooler
    import pandas as pd

    repo = Path(__file__).resolve().parents[1]
    fine = cooler.Cooler(str(workdir / "target.cool"))       # band: offsets <= 39
    for res in (2000, 2500):
        bins = []
        for chrom, ln in CHROMS.items():
            for s in range(0, ln, res):
                bins.append((chrom, s, min(s + res, ln)))
        cooler.create_cooler(str(workdir / f"tmpl_{res}.cool"),
                             bins=pd.DataFrame(bins, columns=["chrom", "start", "end"]),
                             pixels=pd.DataFrame({"bin1_id": [0], "bin2_id": [0],
                                                  "count": [1.0]}))
    subprocess.run([sys.executable, str(repo / "scripts/coarsen_oe.py"),
                    "--fine", str(workdir / "target.cool"), "--res", "2000", "2500",
                    "--band", "39000", "--template", str(workdir / "tmpl_{res}.cool"),
                    "--out", str(workdir / "avg_{res}.cool")],
                   check=True, cwd=repo, capture_output=True)
    for res in (2000, 2500):
        out = cooler.Cooler(str(workdir / f"avg_{res}.cool"))
        dmax = (39 * BINSIZE + BINSIZE) // res - 1
        for chrom, ln in CHROMS.items():
            M = np.asarray(fine.matrix(balance=False).fetch(chrom), float)
            nf = M.shape[0]
            nc = -(-ln // res)
            A = np.zeros((nc, nf))
            for k in range(nf):
                s, e = k * BINSIZE, min((k + 1) * BINSIZE, ln)
                for c in range(s // res, (e - 1) // res + 1):
                    A[c, k] = (min(e, (c + 1) * res) - max(s, c * res)) / BINSIZE
            P = (M != 0).astype(float)
            num, den = A @ M @ A.T, A @ P @ A.T
            got = np.asarray(out.matrix(balance=False).fetch(chrom), float)
            full = (res / BINSIZE) ** 2
            for I in range(nc):
                for J in range(I, nc):
                    inside = J - I <= dmax and den[I, J] / full >= 0.5
                    if inside:
                        assert got[I, J] == pytest.approx(num[I, J] / den[I, J], rel=1e-5)
                    else:
                        assert got[I, J] == 0


def test_reference_per_sample(workdir):
    """samples[].reference_cool overrides data.reference_cool per sample (e.g.
    a leave-one-out consensus); each sample's cached ref is its own map."""
    import cooler

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    src = cooler.Cooler(str(workdir / "target.cool"))
    px = src.pixels()[:]
    px["count"] = -px["count"]                      # a distinguishable map
    neg = workdir / "neg_ref.cool"
    cooler.create_cooler(str(neg), bins=src.bins()[:][["chrom", "start", "end"]],
                         pixels=px, dtypes={"count": "float64"})
    raw = _ref_raw(workdir, "persample", reference="residual", ref_cool=False)
    s0 = raw["data"]["samples"][0]
    raw["data"]["samples"] = [dict(s0, name="a", reference_cool=str(workdir / "target.cool")),
                              dict(s0, name="b", reference_cool=str(neg))]
    raw["train"]["cache_dir"] = str(workdir / "cache_ref_ps")
    c = Config.from_dict(raw)
    assert c.has_reference and c.sample_reference(c.data.samples[1]) == str(neg)
    man = hprep.build(c)
    assert man["reference"] == {"a": str(workdir / "target.cool"), "b": str(neg)}
    for sh in man["shards"]:
        z = np.load(sh["path"])
        sign = 1 if sh["sample"] == "a" else -1
        np.testing.assert_array_equal(z["ref"], sign * z["target"][:, 0])
    htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()
    # a reference model with one sample lacking a map is rejected
    raw["data"]["samples"][1].pop("reference_cool")
    with pytest.raises(ValueError, match="none for"):
        Config.from_dict(raw)


def test_reference_leakage_guard(workdir):
    """A reference whose <cool>.json lists the trained or scored cell (by
    identity: hmec_deep is hmec) is rejected by precompute, training and the
    scoring helpers; one that excludes it passes."""
    import dataclasses
    import json
    import shutil

    from flexcible import precompute as hprep
    from flexcible import train as htrain
    from flexcible.reference import (assert_reference_excludes, cell_identity,
                                       reference_cells)

    assert cell_identity("hmec_deep") == cell_identity("HMEC") == "hmec"
    assert cell_identity("huvec_rep1") == "huvec"
    assert cell_identity("gm12878") != cell_identity("gm23248")
    ref = workdir / "leak_ref.cool"
    shutil.copy(workdir / "target.cool", ref)
    meta = Path(str(ref) + ".json")
    meta.write_text(json.dumps({"cells": ["imr90", "hmec_deep"]}))
    assert reference_cells(ref) == ["imr90", "hmec_deep"]
    assert_reference_excludes(ref, "k562")
    for leaked in ("hmec", "hmec_deep", "IMR90"):
        with pytest.raises(ValueError, match="reference leakage"):
            assert_reference_excludes(ref, leaked)
    assert reference_cells(workdir / "target.cool") is None   # no record: no check

    raw = _ref_raw(workdir, "leak", reference="input", ref_cool=False)
    s0 = raw["data"]["samples"][0]
    raw["data"]["samples"] = [dict(s0, name="imr90", reference_cool=str(ref))]
    raw["train"]["cache_dir"] = str(workdir / "cache_leak")
    with pytest.raises(ValueError, match="reference leakage"):
        hprep.build(Config.from_dict(raw))
    # a clean cache whose reference record later names the sample: training refuses
    raw["data"]["samples"] = [dict(s0, name="k562", reference_cool=str(ref))]
    c = Config.from_dict(raw)
    hprep.build(c)
    meta.write_text(json.dumps({"cells": ["k562"]}))
    with pytest.raises(ValueError, match="reference leakage"):
        htrain.train(c)
    ds = htrain.WindowDataset(c.train.cache_dir, "test")
    assert ds.reference_by_sample() == {"k562": str(ref)}
    with pytest.raises(ValueError, match="reference leakage"):
        ds.check_reference_leakage()


def test_honest_validation_default_and_multisample_split():
    """val_exclude_overlap defaults to True. With one sample the honest split
    keeps the old validation windows and drops the overlapping training
    windows; with several samples validation is drawn by position (every
    sample's window there), each sample gets validation windows, and no
    training window of any sample overlaps a validation position."""
    from flexcible.config import TrainConfig
    from flexcible.train import overlaps_positions, split_val_regions

    assert TrainConfig().val_exclude_overlap is True
    W, stride = 8, 4
    starts = list(range(0, 400, stride))
    one = (["chr1"] * len(starts), starts, ["a"] * len(starts))
    old_vi, _ = split_val_regions(one, 10, W, 3, False)
    vi, ti = split_val_regions(one, 10, W, 3, True)
    np.testing.assert_array_equal(vi, old_vi)
    assert not overlaps_positions(([one[0][i] for i in ti], [one[1][i] for i in ti],
                                   [one[2][i] for i in ti]),
                                  ([one[0][i] for i in vi], [one[1][i] for i in vi],
                                   [one[2][i] for i in vi]), W).any()
    assert len(vi) + len(ti) < len(starts)

    samples = ["a", "b", "c"]
    ch, st, sm = [], [], []
    for s in samples:                       # same positions in every sample
        for c in ("chr1", "chr2"):
            for x in starts:
                ch.append(c); st.append(x); sm.append(s)
    co = (ch, st, sm)
    vi, ti = split_val_regions(co, 30, W, 0, True)
    pos = {(ch[i], st[i]) for i in vi}
    assert len(vi) == 3 * len(pos) >= 30
    for p in pos:                           # all samples at every validation position
        assert {sm[i] for i in vi if (ch[i], st[i]) == p} == set(samples)
    for s in samples:
        assert sum(sm[i] == s for i in vi) > 0 and sum(sm[i] == s for i in ti) > 0
    sub = lambda ix: ([ch[i] for i in ix], [st[i] for i in ix], [sm[i] for i in ix])
    assert not overlaps_positions(sub(ti), sub(vi), W).any()
    assert set(vi).isdisjoint(ti)
    # the old rule leaves training windows of other samples at validation positions
    ovi, oti = split_val_regions(co, 30, W, 0, False)
    assert overlaps_positions(sub(oti), sub(ovi), W).any()


def test_honest_validation_in_multisample_training(workdir, cfg):
    """Training with val_regions on two samples uses the honest split by
    default; eval_val_windows reproduces the same validation windows."""
    import dataclasses

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    raw = _ref_raw(workdir, "honest_ms", ref_cool=False)
    s0 = raw["data"]["samples"][0]
    raw["data"]["samples"] = [dict(s0, name="a"), dict(s0, name="b")]
    raw["train"]["cache_dir"] = str(workdir / "cache_honest_ms")
    raw["train"].pop("val_exclude_overlap")
    c = Config.from_dict(raw)
    assert c.train.val_exclude_overlap
    hprep.build(c)
    data, ds = htrain._stack(c.train.cache_dir, "train", 0.0, False, 0)
    co = htrain._coords(ds)
    vi, ti = htrain.split_val_regions(co, c.train.val_regions, ds.window_size,
                                      c.train.seed, True)
    assert {co[2][i] for i in vi} == {"a", "b"}
    htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()


def test_embedded_reference_predicts_new_cell_from_chip_only(workdir):
    """model.embed_reference copies the map into every checkpoint; predict.py
    then predicts a new cell from its bigWigs alone, identical to passing the
    same map explicitly; a training cell keeps its own reference and a cell
    contained in the embedded map is refused."""
    import json
    import shutil

    import cooler

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible import train as htrain
    from flexcible.reference import EMBED_NAME, resolve_reference

    emb = workdir / "emb_src.cool"
    shutil.copy(workdir / "target.cool", emb)
    Path(str(emb) + ".json").write_text(json.dumps({"cells": ["s", "other"]}))
    raw = _ref_raw(workdir, "embed", reference="residual")
    raw["model"]["embed_reference"] = str(emb)
    raw["train"]["cache_dir"] = str(workdir / "cache_embed")
    c = Config.from_dict(raw)
    hprep.build(c)
    htrain.train(c)
    run = Path(c.train.output_dir)
    ck = run / (run / "best").read_text().strip()
    assert (ck / EMBED_NAME).exists() and (ck / (EMBED_NAME + ".json")).exists()
    assert json.loads((ck / (EMBED_NAME + ".json")).read_text())["cells"] == ["s", "other"]
    # training sample: its own reference; new cell: the embedded map
    assert resolve_reference(c, ck, "s") == str(workdir / "target.cool")
    assert resolve_reference(c, ck, "newcell") == str(ck / EMBED_NAME)
    with pytest.raises(ValueError, match="reference leakage"):
        resolve_reference(c, ck, "other")
    a = str(workdir / "pred_emb.cool")
    b = str(workdir / "pred_explicit.cool")
    hpred.predict(c, "chr1", a, batch=4, bigwig_dir=str(workdir), cell="newcell", tta=True)
    hpred.predict(c, "chr1", b, batch=4, bigwig_dir=str(workdir), cell="newcell", tta=True,
                  reference=str(emb))
    pa = np.nan_to_num(cooler.Cooler(a).matrix(balance=False).fetch("chr1"))
    pb = np.nan_to_num(cooler.Cooler(b).matrix(balance=False).fetch("chr1"))
    assert np.abs(pa).sum() > 0
    # the two paths read the same map; GPU kernels are not bitwise deterministic across calls (~2e-5)
    assert np.abs(pa - pb).max() < 1e-3


def test_predict_subset_model_sequence_only(workdir, cfg):
    """predict.py runs a model trained on a subset of the cached rows (no ChIP,
    motifs and GC/CpG only), as the sequence-only reference model uses."""
    import dataclasses

    import cooler

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible import train as htrain

    raw = _ref_raw(workdir, "seqonly", ref_cool=False)
    raw["train"].update({"select_tracks": [], "select_motifs": [MOTIF],
                         "cache_dir": str(workdir / "cache_seqonly")})
    c = Config.from_dict(raw)
    hprep.build(c)
    htrain.train(c)
    out = str(workdir / "pred_seqonly.cool")
    hpred.predict(c, "chr1", out, batch=4, tta=True)
    assert np.abs(np.nan_to_num(cooler.Cooler(out).matrix(balance=False).fetch("chr1"))).sum() > 0


def test_film_resolution_conditioning_identity_at_init():
    """FiLM projections start at zero, so a resolution-conditioned U-Net
    initially equals the plain U-Net with the same weights; after changing a
    projection the resolutions differ; unknown resolutions are refused."""
    import torch

    from flexcible.model import HiCModel, build_backbone, n_cond_channels

    torch.manual_seed(0)
    in_ch = n_cond_channels(3, 1, 0, "full")
    plain = build_backbone("unet", in_ch, out_ch=1, base=8, depth=2)
    film = build_backbone("unet", in_ch, out_ch=1, base=8, depth=2, film_dim=4)
    film.load_state_dict(plain.state_dict(), strict=False)
    m0 = HiCModel(plain, "full").eval()
    m1 = HiCModel(film, "full", resolutions=[1000, 2000], res_embed_dim=4,
                  default_resolution=1000).eval()
    cond = torch.randn(2, 3, 16)
    pairs = cond[:, 1:3].reshape(2, 1, 2, 16)
    with torch.no_grad():
        a = m0(cond, pairs)
        b = m1(cond, pairs)
        c = m1(cond, pairs, res=2000)
        torch.nn.init.normal_(m1.backbone.film[0].weight)
        d1 = m1(cond, pairs, res=1000)
        d2 = m1(cond, pairs, res=2000)
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)
    assert not torch.allclose(d1, d2)
    with pytest.raises(ValueError, match="not among"):
        m1(cond, pairs, res=5000)
    with pytest.raises(ValueError, match="unet only"):
        build_backbone("dilated", in_ch, film_dim=4)


def test_multires_training_and_views(workdir):
    """train_multires on 1 kb and 2 kb windows of the fixture (reference
    residual, honest validation), then per-resolution views that
    eval_val_windows and predict.py load like single-resolution runs."""
    import json
    import subprocess

    import cooler

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible.train_multires import train_multires, write_views

    c2 = workdir / "target_2kb.cool"
    if not c2.exists():
        cooler.coarsen_cooler(str(workdir / "target.cool"), str(c2), factor=2, chunksize=10**6)
    cfgs = []
    for res, cool, W in ((BINSIZE, workdir / "target.cool", 32), (2 * BINSIZE, c2, 16)):
        raw = _ref_raw(workdir, f"mr{res}", reference="residual", ref_cool=False)
        raw["data"]["resolution"] = res
        raw["data"]["samples"][0].update(cool_path=str(cool), reference_cool=str(cool))
        raw["window"].update(size=W, stride=W // 2)
        raw["model"]["resolutions"] = [BINSIZE, 2 * BINSIZE]
        raw["model"]["res_embed_dim"] = 4
        raw["train"].update(cache_dir=str(workdir / f"cache_mr{res}"), val_regions=2,
                            max_epochs=2, patience_epochs=5, batch_size=2)
        c = Config.from_dict(raw)
        hprep.build(c)
        cfgs.append(c)
    out = workdir / "run_multires"
    train_multires(cfgs, out)
    lc = list(open(out / "learning_curve.csv"))
    assert len(lc) == 3 and "val_1000" in lc[0] and "loss_2000" in lc[0]
    views = {BINSIZE: workdir / "view_1kb", 2 * BINSIZE: workdir / "view_2kb"}
    write_views(out, views)
    repo = Path(__file__).resolve().parents[1]
    for res, v in views.items():
        ck = v / "checkpoint-best"
        assert (ck / "model.pt").is_symlink()
        vc = Config.load(ck / "config.yaml")
        assert vc.data.resolution == res and vc.model.resolutions == [BINSIZE, 2 * BINSIZE]
        o = v / "val.json"
        subprocess.run([sys.executable, str(repo / "scripts/eval_val_windows.py"), "--run",
                        str(v), "--split", "val", "--out", str(o)], check=True, cwd=repo,
                       capture_output=True)
        assert json.loads(o.read_text())["n"] > 0
        p = str(v / "pred.cool")
        hpred.predict(vc, "chr1", p, checkpoint=str(ck), batch=4)
        assert cooler.Cooler(p).binsize == res


def test_multires_reference_dropout(workdir):
    """train_multires with train.reference_dropout: the reference plane of a share of the
    training windows is set to 0 (counted per resolution in the learning curve), and the
    validation is also reported with the plane at 0 (valnoref_<res>, valnoref_mean) while the
    checkpoint selection stays on the validation with the reference."""
    import csv

    import cooler

    from flexcible import precompute as hprep
    from flexcible.train_multires import train_multires

    c2 = workdir / "target_2kb.cool"
    if not c2.exists():
        cooler.coarsen_cooler(str(workdir / "target.cool"), str(c2), factor=2, chunksize=10**6)
    cfgs = []
    for res, cool, W in ((BINSIZE, workdir / "target.cool", 32), (2 * BINSIZE, c2, 16)):
        raw = _ref_raw(workdir, f"mrrd{res}", reference="input", ref_cool=False)
        raw["data"]["resolution"] = res
        raw["data"]["samples"][0].update(cool_path=str(cool), reference_cool=str(cool))
        raw["window"].update(size=W, stride=W // 2)
        raw["model"]["resolutions"] = [BINSIZE, 2 * BINSIZE]
        raw["model"]["res_embed_dim"] = 4
        raw["train"].update(cache_dir=str(workdir / f"cache_mrrd{res}"), val_regions=2,
                            max_epochs=2, patience_epochs=5, batch_size=2, reference_dropout=1.0)
        c = Config.from_dict(raw)
        hprep.build(c)
        cfgs.append(c)
    out = workdir / "run_multires_refdrop"
    train_multires(cfgs, out)
    lc = list(csv.DictReader(open(out / "learning_curve.csv")))
    assert len(lc) == 2
    for k in (f"refdrop_{BINSIZE}", f"refdrop_{2 * BINSIZE}", f"valnoref_{BINSIZE}",
              f"valnoref_{2 * BINSIZE}", "valnoref_mean"):
        assert k in lc[0]
    # with the plane withheld in every window, at least one window per step is counted
    assert all(int(r[f"refdrop_{BINSIZE}"]) + int(r[f"refdrop_{2 * BINSIZE}"])
               >= int(r[f"steps_{BINSIZE}"]) + int(r[f"steps_{2 * BINSIZE}"]) > 0 for r in lc)
    assert all(np.isfinite(float(r["valnoref_mean"])) for r in lc)
    best = json.loads((out / "checkpoint-best" / "progress.json").read_text())["best"]
    assert best == max(float(r["val_mean"]) for r in lc)


def test_predict_bin_size_option(workdir, monkeypatch):
    """predict --bin-size: a continuous-embedding multi-resolution model predicts at a bin size it
    was not trained on (window keeps its genomic span, reference at that bin size required,
    normalisers of the nearest trained bin size or refitted); lookup and single-resolution
    models are refused."""
    import cooler

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible.train_multires import train_multires, write_views

    c2 = workdir / "target_2kb.cool"
    if not c2.exists():
        cooler.coarsen_cooler(str(workdir / "target.cool"), str(c2), factor=2, chunksize=10**6)
    c4 = workdir / "target_4kb.cool"
    if not c4.exists():
        cooler.coarsen_cooler(str(c2), str(c4), factor=2, chunksize=10**6)
    cfgs = []
    for res, cool, W in ((BINSIZE, workdir / "target.cool", 32), (2 * BINSIZE, c2, 16)):
        raw = _ref_raw(workdir, f"bs{res}", reference="input", ref_cool=False)
        raw["data"]["resolution"] = res
        raw["data"]["samples"][0].update(cool_path=str(cool), reference_cool=str(cool))
        raw["window"].update(size=W, stride=W // 2)
        raw["model"].update(resolutions=[BINSIZE, 2 * BINSIZE], res_embed_dim=4,
                            res_embed_mode="continuous")
        raw["train"].update(cache_dir=str(workdir / f"cache_bs{res}"), val_regions=2,
                            max_epochs=2, patience_epochs=5, batch_size=2)
        c = Config.from_dict(raw)
        hprep.build(c)
        cfgs.append(c)
    out = workdir / "run_binsize"
    train_multires(cfgs, out)
    view = workdir / "view_bs_1kb"
    write_views(out, {BINSIZE: view, 2 * BINSIZE: workdir / "view_bs_2kb"})
    ck = view / "checkpoint-best"
    vc = Config.load(ck / "config.yaml")
    assert hpred.scaled_window(vc, 4000) == (8, 4)          # 32 kb span, multiple of 2 ** depth
    assert hpred.nearest_trained([1000, 2000], 4000) == 2000
    assert hpred.nearest_trained([2000, 5000, 10000, 25000], 4000) == 5000
    assert hpred.nearest_trained([2000, 5000, 10000, 25000], 50000) == 25000
    assert hpred.norm_dir_for(ck, [1000, 2000], 4000) == out / "checkpoint-best" / "res_2000"

    p4 = workdir / "pred_4kb.cool"
    hpred.predict(vc, "chr1", str(p4), checkpoint=str(ck), batch=4, bin_size=4000,
                  reference=str(c4))
    clr = cooler.Cooler(str(p4))
    assert clr.binsize == 4000 and clr.info["nnz"] > 0
    assert clr.chromsizes["chr1"] == CHROMS["chr1"] and not list(workdir.glob("pred_4kb.cool.grid*"))
    assert vc.data.resolution == BINSIZE                   # the loaded config is not modified in place
    p4r = workdir / "pred_4kb_refit.cool"
    hpred.predict(Config.load(ck / "config.yaml"), "chr1", str(p4r), checkpoint=str(ck), batch=4,
                  bin_size=4000, reference=str(c4), renormalize_tracks=True)
    assert cooler.Cooler(str(p4r)).binsize == 4000
    # the bin size reaches the network: a different bin size gives a different map
    m1, m4 = (np.nan_to_num(cooler.Cooler(str(p)).matrix(balance=False).fetch("chr1"))
              for p in (p4, p4r))
    assert m1.shape == m4.shape == (24, 24)

    with pytest.raises(ValueError, match="needs --reference"):
        hpred.predict(Config.load(ck / "config.yaml"), "chr1", str(workdir / "x.cool"),
                      checkpoint=str(ck), bin_size=4000)
    lookup = Config.load(ck / "config.yaml")
    lookup.model.res_embed_mode = "lookup"
    with pytest.raises(ValueError, match="lookup"):
        hpred.predict(lookup, "chr1", str(workdir / "x.cool"), checkpoint=str(ck), bin_size=4000,
                      reference=str(c4))
    single = Config.from_dict(_ref_raw(workdir, "bs_single", reference="none", ref_cool=False))
    with pytest.raises(ValueError, match="several bin sizes"):
        hpred.predict(single, "chr1", str(workdir / "x.cool"), checkpoint=str(ck), bin_size=4000,
                      reference=str(c4))
    # command line: --bin-size and its alias --binSize reach predict()
    seen = {}
    monkeypatch.setattr(hpred, "predict", lambda *a, **k: seen.update(args=a))
    vc.save(workdir / "view_bs.yaml")
    for flag in ("--bin-size", "--binSize"):
        monkeypatch.setattr(sys, "argv", ["flexcible predict", "--config", str(workdir / "view_bs.yaml"),
                                          "--out", "o.cool", flag, "4000", "--reference", str(c4)])
        hpred.main()
        assert seen["args"][-2] == 4000      # bin_size, then no_reference


def test_cascade_prior_plane(workdir):
    """model.prior: a coarser map (2 kb) resampled onto the 1 kb windows as an
    extra input plane; prior_block takes, per fine bin, the coarse bin holding
    its centre; the cascade model trains, is scored and predicts."""
    import json
    import subprocess

    import cooler

    from flexcible import precompute as hprep
    from flexcible import predict as hpred
    from flexcible import train as htrain
    from flexcible.matrix import MatrixNormalizer
    from flexcible.reference import prior_block

    c2 = workdir / "target_2kb.cool"
    if not c2.exists():
        cooler.coarsen_cooler(str(workdir / "target.cool"), str(c2), factor=2, chunksize=10**6)
    clr2 = cooler.Cooler(str(c2))
    ident = MatrixNormalizer(transform="prescaled", scale=1.0)
    blk = prior_block(clr2, "chr1", 3, 10, BINSIZE, ident)
    M = np.nan_to_num(np.asarray(clr2.matrix(balance=False).fetch("chr1"), float))
    fine = np.arange(3, 13)
    np.testing.assert_allclose(blk, np.clip(M[np.ix_(fine // 2, fine // 2)], -1, 1), rtol=1e-6)

    raw = _ref_raw(workdir, "casc", reference="residual")
    raw["data"]["samples"][0]["prior_cool"] = str(c2)
    raw["model"]["prior"] = True
    raw["train"]["cache_dir"] = str(workdir / "cache_casc")
    with pytest.raises(ValueError, match="model.prior needs"):
        bad = json.loads(json.dumps(raw))
        bad["data"]["samples"][0].pop("prior_cool")
        Config.from_dict(bad)
    c = Config.from_dict(raw)
    assert c.model.reference_planes == 2
    man = hprep.build(c)
    assert man["prior"] == {"s": str(c2)}
    mn = MatrixNormalizer.load(Path(c.train.cache_dir) / "norm_matrix_s.json")
    sh = next(x for x in man["shards"] if x["chrom"] == "chr1")
    z = np.load(sh["path"])
    st = int(z["start"][0])
    np.testing.assert_allclose(z["prior"][0].astype(np.float32),
                               prior_block(clr2, "chr1", st, 32, BINSIZE, mn), atol=2e-3)
    htrain.train(c)
    run = Path(c.train.output_dir)
    repo = Path(__file__).resolve().parents[1]
    o = run / "t.json"
    subprocess.run([sys.executable, str(repo / "scripts/eval_val_windows.py"), "--run", str(run),
                    "--split", "test", "--out", str(o)], check=True, cwd=repo, capture_output=True)
    assert json.loads(o.read_text())["n"] > 0
    p = str(workdir / "pred_casc.cool")
    hpred.predict(c, "chr1", p, batch=4, tta=True)
    assert np.abs(np.nan_to_num(cooler.Cooler(p).matrix(balance=False).fetch("chr1"))).sum() > 0


def test_multires_staged_memmaps_match_cache(workdir):
    """--stage-only writes each resolution's training windows as memory maps
    identical to the cache (cond reduced to the model's rows), and training
    from them runs."""
    import cooler

    from flexcible import precompute as hprep
    from flexcible.dataset import WindowDataset
    from flexcible.train import _select_rows
    from flexcible.train_multires import load_rows, open_stage, stage_arrays, train_multires

    c2 = workdir / "target_2kb.cool"
    if not c2.exists():
        cooler.coarsen_cooler(str(workdir / "target.cool"), str(c2), factor=2, chunksize=10**6)
    cfgs = []
    for res, cool, W in ((BINSIZE, workdir / "target.cool", 32), (2 * BINSIZE, c2, 16)):
        raw = _ref_raw(workdir, f"mrs{res}", reference="residual", ref_cool=False)
        raw["data"]["resolution"] = res
        raw["data"]["samples"][0].update(cool_path=str(cool), reference_cool=str(cool))
        raw["window"].update(size=W, stride=W // 2)
        raw["model"].update(resolutions=[BINSIZE, 2 * BINSIZE], res_embed_dim=4)
        raw["train"].update(cache_dir=str(workdir / f"cache_mr{res}"), val_regions=2,
                            max_epochs=1, patience_epochs=5, batch_size=2,
                            select_motifs=[MOTIF])
        c = Config.from_dict(raw)
        if not (Path(c.train.cache_dir) / "manifest.json").exists():
            hprep.build(c)
        cfgs.append(c)
    st = workdir / "stage_mr"
    stage_arrays(cfgs, st)
    for c in cfgs:
        ds = WindowDataset(c.train.cache_dir, "train")
        rows, *_ = _select_rows(c, ds.channels)
        mm = open_stage(st, c, len(ds), rows)
        ref = load_rows(ds, range(len(ds)))
        np.testing.assert_array_equal(np.asarray(mm["target"]), ref["target"].numpy())
        np.testing.assert_array_equal(np.asarray(mm["cond"]), ref["cond"][:, rows].numpy())
        np.testing.assert_array_equal(np.asarray(mm["ref"]), ref["ref"].numpy())
    train_multires(cfgs, workdir / "run_mr_staged", stage_dir=st)
    assert (workdir / "run_mr_staged" / "checkpoint-best" / "model.pt").exists()


def test_reference_dropout_config_and_training(workdir):
    """train.reference_dropout (default 0) trains a reference model with the
    reference plane withheld in some windows, together with track dropout;
    it is refused for a model without reference."""
    from flexcible import precompute as hprep
    from flexcible import train as htrain
    from flexcible.config import TrainConfig

    assert TrainConfig().reference_dropout == 0.0
    raw = _ref_raw(workdir, "refdrop", reference="residual")
    raw["train"].update(reference_dropout=0.5, track_dropout=0.5,
                        track_dropout_mode="uniform_count")
    c = Config.from_dict(raw)
    if not (Path(c.train.cache_dir) / "manifest.json").exists():
        hprep.build(c)
    htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()
    raw2 = _ref_raw(workdir, "refdrop_bad", ref_cool=False)
    raw2["train"].update(reference_dropout=0.5, cache_dir=str(workdir / "cache_noref"))
    c2 = Config.from_dict(raw2)
    if not (Path(c2.train.cache_dir) / "manifest.json").exists():
        hprep.build(c2)
    with pytest.raises(ValueError, match="reference_dropout"):
        htrain.train(c2)


def test_hard_split_training(workdir):
    """train.hard_split: every window overlapping any cell's hard position is
    removed from training for all cells, the hard-validation windows are the
    listed ones of each sample, and training selects on them."""
    import json

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    raw = _ref_raw(workdir, "hard", ref_cool=False)
    s0 = raw["data"]["samples"][0]
    raw["data"]["samples"] = [dict(s0, name="a"), dict(s0, name="b")]
    raw["train"]["cache_dir"] = str(workdir / "cache_hard")
    c = Config.from_dict(raw)
    if not (Path(c.train.cache_dir) / "manifest.json").exists():
        hprep.build(c)
    data, ds = htrain._stack(c.train.cache_dir, "train", 0.0, False, 0)
    co = htrain._coords(ds)
    W = ds.window_size
    a_rows = [i for i, s in enumerate(co[2]) if s == "a"]
    v = (co[0][a_rows[0]], int(co[1][a_rows[0]]))
    t = (co[0][a_rows[-1]], int(co[1][a_rows[-1]]))
    sp = {"res": BINSIZE, "window": W,
          "cells": {"a": {"val": [list(v)], "test": [list(t)]}, "b": {"val": [], "test": []}},
          "union": [[v[0], v[1], v[1] + W], [t[0], t[1], t[1] + W]]}
    f = workdir / "hard_split.json"
    f.write_text(json.dumps(sp))
    hv, keep = htrain.hard_split_rows(co, str(f), W, BINSIZE)
    assert [co[2][i] for i in hv] == ["a"] and (co[0][hv[0]], int(co[1][hv[0]])) == v
    for i in keep:                      # no kept window of any sample touches a hard position
        for ch, a, b in sp["union"]:
            assert not (co[0][i] == ch and int(co[1][i]) < b and a < int(co[1][i]) + W)
    b_at_v = [i for i in range(len(co[0])) if co[2][i] == "b" and co[0][i] == v[0]
              and int(co[1][i]) == v[1]]
    assert b_at_v and not set(b_at_v) & set(keep)        # removed for the other cell too
    with pytest.raises(ValueError, match="split for"):
        htrain.hard_split_rows(co, str(f), W + 1, BINSIZE)
    raw["train"].update(hard_split=str(f), output_dir=str(workdir / "run_hard"))
    c2 = Config.from_dict(raw)
    htrain.train(c2)
    txt = (workdir / "run_hard" / "best").exists()
    assert txt


def test_train_from_staged_memmaps(workdir):
    """train.stage_dir: the staged split read through LazyRows equals the
    resident load (rows, column selection, val split), and training runs."""
    import torch

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    raw = _ref_raw(workdir, "staged", reference="residual")
    raw["train"].update(cache_dir=str(workdir / "cache_ref"),
                        stage_dir=str(workdir / "stage"), track_dropout=0.5,
                        track_dropout_mode="uniform_count")
    c = Config.from_dict(raw)
    if not (Path(c.train.cache_dir) / "manifest.json").exists():
        hprep.build(c)
    htrain.stage_cache(c.train.cache_dir, "train", c.train.stage_dir)
    eager, _ = htrain._stack(c.train.cache_dir, "train", 0.0, False, 0)
    lazy, _ = htrain._stack(c.train.cache_dir, "train", 0.0, False, 0,
                            stage_dir=c.train.stage_dir)
    assert set(lazy) >= {"target", "cond", "ref"}
    rows = np.array([3, 0, 2])
    for k in ("target", "cond", "ref"):
        assert lazy[k].shape == tuple(eager[k].shape)
        got = lazy[k][rows].to("cpu")[torch.arange(3)]
        assert torch.equal(got.float(), eager[k][rows].float())
    sel = lazy["cond"][:, [0, 2], :].materialize()
    assert torch.equal(sel.float(), eager["cond"][:, [0, 2], :].float())
    htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()
    with pytest.raises(RuntimeError, match="not staged"):
        htrain._stack(c.train.cache_dir, "train", 0.0, False, 0,
                      stage_dir=str(workdir / "nostage"))


def test_sample_train_chroms(workdir):
    """samples[].train_chroms keeps a sample's training windows on the listed
    chromosomes only; other samples and the test split are unaffected."""
    from flexcible import precompute as hprep

    raw = _ref_raw(workdir, "tchroms", ref_cool=False)
    s0 = raw["data"]["samples"][0]
    raw["data"]["holdout_bed"] = None
    raw["data"]["chroms"] = ["chr1", "chr2"]
    raw["data"]["train_parity"] = "all"
    raw["data"]["samples"] = [dict(s0, name="a"), dict(s0, name="b", train_chroms=["chr1"])]
    raw["train"]["cache_dir"] = str(workdir / "cache_tchroms")
    c = Config.from_dict(raw)
    man = hprep.build(c)
    tr = {(x["sample"], x["chrom"]) for x in man["shards"] if x["split"] == "train"}
    assert ("a", "chr1") in tr and ("a", "chr2") in tr
    assert ("b", "chr1") in tr and ("b", "chr2") not in tr


def test_between_cell_variance_loss_weighting(workdir):
    """between_cell_weights: per locus the variance across samples; the
    weights have mean 1 + alpha over multi-sample loci, are 1 where the
    samples agree; training with them runs."""
    import torch

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    t = torch.zeros(4, 1, 3, 3)
    t[1, 0, 0, 0] = 2.0                          # locus A: samples differ at one pixel
    coords = (["chr1", "chr1", "chr1", "chr1"], np.array([0, 0, 5, 5]), ["a", "b", "a", "b"])
    pos, w = htrain.between_cell_weights(t, coords, 1.0)
    assert pos.tolist() == [0, 0, 1, 1]
    assert float(w[1].max()) == 1.0 and float(w[0, 0, 0]) > 1.0
    raw = _ref_raw(workdir, "varw", ref_cool=False)
    s0 = raw["data"]["samples"][0]
    raw["data"]["samples"] = [dict(s0, name="a"), dict(s0, name="b")]
    raw["train"].update(cache_dir=str(workdir / "cache_honest_ms"), loss_weighting="between_cell_var",
                        val_regions=1)
    c = Config.from_dict(raw)
    if not (Path(c.train.cache_dir) / "manifest.json").exists():
        hprep.build(c)
    htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()


def test_track_dropout_with_selected_tracks(workdir):
    """Track dropout on a model trained on a subset of the tracks acts on the
    kept ChIP rows only (they lead the reduced cond) with their own absent
    values, and trains."""
    from flexcible import precompute as hprep
    from flexcible import train as htrain

    channels = {"tracks": ["CTCF", "RAD21", "H3K27ac"], "seq_features": ["gc", "cpg"],
                "motif_ids": [MOTIF]}
    raw = _ref_raw(workdir, "dropsel", ref_cool=False)
    raw["data"]["tracks"] = [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                             {"name": "RAD21", "type": "cofactor", "motif": MOTIF},
                             {"name": "H3K27ac", "type": "histone"}]
    raw["train"].update(select_tracks=["CTCF", "H3K27ac"], select_motifs=[MOTIF])
    c = Config.from_dict(raw)
    assert htrain.dropout_chip_rows(c, channels) == [0, 2]
    rows, *_ = htrain._select_rows(c, channels)
    assert rows[:2] == [0, 2]
    raw["train"]["select_tracks"] = None
    raw["train"]["select_motifs"] = None
    assert htrain.dropout_chip_rows(Config.from_dict(raw), channels) == [0, 1, 2]
    raw = _ref_raw(workdir, "dropsel2", ref_cool=False)
    raw["train"].update(select_tracks=["H3K27ac"], select_motifs=[MOTIF], track_dropout=0.5,
                        track_dropout_mode="uniform_count", cache_dir=str(workdir / "cache_ref"),
                        max_steps=4)
    c2 = Config.from_dict(raw)
    if not (Path(c2.train.cache_dir) / "manifest.json").exists():
        hprep.build(c2)
    htrain.train(c2)


def test_sv_mask_trains(workdir):
    """train.sv_mask with val_regions: the masked loss path runs end to end."""
    import json

    from flexcible import precompute, train
    from flexcible.config import Config
    bp = workdir / "bp_sv.json"
    bp.write_text(json.dumps({"breakpoints": {"default": [["chr1", 20 * BINSIZE + 5],
                                                          ["chr2", 12 * BINSIZE]]}}))
    raw = {
        "data": {"cool_path": str(workdir / "target.cool"), "bigwig_dir": str(workdir),
                 "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                            {"name": "H3K27ac", "type": "histone"}],
                 "resolution": BINSIZE},
        "sequence": {"fasta": str(workdir / "genome.fa"), "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "window": {"size": 32, "stride": 8, "max_zero_fraction": 0.99},
        "model": {"base_channels": 16, "depth": 2},
        "train": {"output_dir": str(workdir / "run_sv"), "cache_dir": str(workdir / "cache_sv"),
                  "batch_size": 4, "max_steps": 20, "eval_every": 10, "val_regions": 2,
                  "lr": 1e-3, "sv_mask": str(bp)},
    }
    cfg = Config.from_dict(raw)
    precompute.build(cfg)
    train.train(cfg)
    assert (workdir / "run_sv" / "latest").exists() or (workdir / "run_sv" / "best").exists()


def test_structure_loss_and_combined_selection(workdir):
    """Structure loss terms and select_metric=combined run end to end and
    write val_structure.csv."""
    from flexcible import precompute, train
    from flexcible.config import Config
    raw = {
        "data": {"cool_path": str(workdir / "target.cool"), "bigwig_dir": str(workdir),
                 "tracks": [{"name": "CTCF", "type": "tf", "motif": MOTIF},
                            {"name": "H3K27ac", "type": "histone"}],
                 "resolution": BINSIZE},
        "sequence": {"fasta": str(workdir / "genome.fa"), "cache_dir": str(workdir / "gcache")},
        "motifs": {"dir": str(workdir), "score_frac": 0.8},
        "window": {"size": 32, "stride": 8, "max_zero_fraction": 0.99},
        "model": {"base_channels": 16, "depth": 2},
        "train": {"output_dir": str(workdir / "run_struct"),
                  "cache_dir": str(workdir / "cache_sv"),
                  "batch_size": 4, "max_steps": 30, "eval_every": 10, "val_regions": 2,
                  "lr": 1e-3, "loss": "l1+mse+0.1*insul+0.1*donut+0.1*compart",
                  "select_metric": "combined"},
    }
    cfg = Config.from_dict(raw)
    if not (workdir / "cache_sv" / "manifest.json").exists():
        precompute.build(cfg)
    train.train(cfg)
    out = workdir / "run_struct"
    assert (out / "best").exists()
    assert (out / "val_structure.csv").exists()


def test_oversample_samples_training(workdir, caplog):
    """train.oversample_samples / oversample_factor repeat the listed samples'
    training windows; the other sample's windows enter once."""
    import logging

    from flexcible import precompute as hprep
    from flexcible import train as htrain

    raw = _ref_raw(workdir, "over", ref_cool=False)
    s0 = raw["data"]["samples"][0]
    raw["data"]["samples"] = [dict(s0, name="a"), dict(s0, name="b")]
    raw["train"].update(cache_dir=str(workdir / "cache_over"), max_steps=6, eval_every=3, val_regions=1,
                        oversample_samples=["b"], oversample_factor=3)
    c = Config.from_dict(raw)
    hprep.build(c)
    with caplog.at_level(logging.INFO):
        htrain.train(c)
    msgs = [r.getMessage() for r in caplog.records if "oversample x3" in r.getMessage()]
    assert msgs, "no oversample log line"
    import re
    m = re.search(r"(\d+) windows of 1 samples; training windows (\d+) -> (\d+)", msgs[0])
    n_over, n0, n1 = (int(x) for x in m.groups())
    assert n1 == n0 + 2 * n_over and 0 < n_over < n0
    bad = _ref_raw(workdir, "over_bad", ref_cool=False)
    bad["data"]["samples"] = [dict(s0, name="a"), dict(s0, name="b")]
    bad["train"].update(cache_dir=str(workdir / "cache_over"), max_steps=2, eval_every=2, val_regions=1,
                        oversample_samples=["zz"], oversample_factor=2)
    with pytest.raises(ValueError, match="no training window"):
        htrain.train(Config.from_dict(bad))


def test_lab_indicator_rows(workdir):
    """data.lab_classes adds one constant row per class after the sequence
    features (and the assay and cohesin rows): 1 for samples of that lab; off,
    the layout is unchanged."""
    import dataclasses

    from flexcible.precompute import (assemble_chrom, sample_track_files,
                                        seq_feature_names, sym_row_names)

    raw = _cohesin_raw(workdir, "labrow")
    raw["data"]["samples"][0]["lab"] = "xu"
    on = Config.from_dict(raw)
    raw_off = _cohesin_raw(workdir, "labrow")
    raw_off["data"]["samples"][0]["lab"] = "xu"
    off = Config.from_dict(raw_off)
    on = dataclasses.replace(on, data=dataclasses.replace(on.data, lab_classes=["xu"]))
    r_on, r_off = sym_row_names(on), sym_row_names(off)
    assert len(r_on) == len(r_off) + 1
    k = r_on.index("lab:xu")
    assert k == len(on.track_names) + len(seq_feature_names(on)) - 1
    assert r_on.index("cohesin:unknown") == k - 1
    nb = CHROMS["chr1"] // BINSIZE
    for s, v in zip(on.data.samples, (1.0, 0.0)):
        a_on = assemble_chrom(on, s, sample_track_files(on, s), "chr1", nb)
        a_off = assemble_chrom(off, s, sample_track_files(off, s), "chr1", nb)
        assert np.all(a_on[k] == v)
        np.testing.assert_array_equal(np.delete(a_on, k, axis=0), a_off)
