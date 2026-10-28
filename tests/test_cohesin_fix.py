"""Cohesin-fraction input and cohesin-loss augmentation (registry section 55).

CPU only:  CUDA_VISIBLE_DEVICES= nice -n 10 python -m pytest tests/test_cohesin_fix.py -q
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flexcible.cohesin import (CohesinAugmenter, DegronOperator, fraction_z,   # noqa: E402
                                 load_operator, save_operator)
from flexcible.config import Config                                            # noqa: E402
from flexcible.precompute import (FRACTION_ROW, assemble_chrom, fix_fraction_norm,   # noqa: E402
                                    fraction_row_value, sample_track_files,
                                    seq_feature_names, sym_row_names)
from flexcible.tracks import TrackNormalizer                                   # noqa: E402
from test_pipeline import BINSIZE, CHROMS, _cohesin_raw, workdir                 # noqa: E402,F401


class HalfOp(torch.nn.Module):
    """Stand-in operator: the change is -0.5 times the map (log O/E units)."""

    def forward(self, x):
        n = x.shape[-1]
        far = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs() >= 2
        return -0.5 * x * far.to(x.dtype)[None]


def test_fraction_value():
    assert fraction_row_value(1.0) == pytest.approx(0.0)
    assert fraction_row_value(0.0) == pytest.approx(-1.0)
    assert fraction_row_value(0.26) == pytest.approx(fraction_z(0.26))
    assert float(fraction_z(torch.tensor(0.5))) == pytest.approx(math.log1p(0.5) / math.log(2) - 1)
    tn = TrackNormalizer(mean=[0.3, 0.2], std=[0.1, 0.4])
    fix_fraction_norm(tn, 1)
    z = tn.apply(np.array([[1.0], [0.26]], dtype=np.float32))
    assert z[1, 0] == pytest.approx(fraction_row_value(0.26), abs=1e-6)
    assert tn.apply(np.array([[1.0], [1.0]], dtype=np.float32))[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert tn.absent_values()[1] == pytest.approx(-1.0, abs=1e-6)


def test_operator_symmetry_zero_band_roundtrip(tmp_path):
    torch.manual_seed(0)
    op = DegronOperator(width=4, dilations=(1, 2)).eval()
    x = torch.randn(3, 16, 16)
    x = 0.5 * (x + x.transpose(-1, -2))
    y = op(x)
    assert y.shape == x.shape
    assert torch.allclose(y, y.transpose(-1, -2), atol=1e-6)
    near = (torch.arange(16)[:, None] - torch.arange(16)[None, :]).abs() < 2
    assert float(y.detach()[:, near].abs().max()) == 0.0
    save_operator(tmp_path / "op.pt", op, 0.26, 10000, meta={"a": 1})
    op2, f = load_operator(tmp_path / "op.pt")
    assert f == pytest.approx(0.26)
    assert torch.allclose(op2(x), y, atol=1e-6)
    assert not any(p.requires_grad for p in op2.parameters())


def _aug(prob=1.0, fractions=(0.0,), p_absent=0.0, n_rows=6):
    # cond rows: 0 RAD21, 1 SMC3, 2 filler, 3 fraction row; absent codes -2
    mean, std = [0.4, 0.4], [0.3, 0.3]
    return CohesinAugmenter(HalfOp(), 0.26, prob, fractions, p_absent, [0, 1], mean, std,
                            [-2.0, -2.0], 3, "cpu"), mean, std


def _batch(bs=4, w=12, intact=(True, True, False, True)):
    g = torch.Generator().manual_seed(1)
    cond = torch.randn(bs, 6, w, generator=g)
    cond[:, 3] = torch.tensor([0.0 if i else fraction_z(0.26) for i in intact])[:, None]
    x = torch.randn(bs, w, w, generator=g) * 0.4
    x = 0.5 * (x + x.transpose(-1, -2))
    scale = torch.tensor([1.9, 1.3, 2.1, 1.9])
    tgt = (x / scale[:, None, None]).clamp(-1, 1)[:, None]
    return cond, tgt, scale


def test_augmenter_changes_only_intact_windows():
    aug, mean, std = _aug(fractions=(0.0,), p_absent=0.0)
    cond, tgt, scale = _batch()
    c2, t2, n = aug(cond, tgt, scale)
    assert n == 3
    # window 2 carries the auxin fraction: untouched
    assert torch.equal(c2[2], cond[2]) and torch.equal(t2[2], tgt[2])
    for i in (0, 1, 3):
        assert float(c2[i, 3].mean()) == pytest.approx(-1.0, abs=1e-6)       # f = 0
        # f = 0: the cohesin signal is flat at the window median
        raw = np.expm1(c2[i, 0].numpy() * std[0] + mean[0])
        assert raw.std() == pytest.approx(0.0, abs=1e-4)
        assert torch.equal(c2[i, 2], cond[i, 2])                              # other rows
    # target: x + (1 - f) / (1 - f_ref) * op(x) = x * (1 - 0.5 / 0.74), in the sample's units
    x0 = tgt[0, 0] * scale[0]
    far = (torch.arange(12)[:, None] - torch.arange(12)[None, :]).abs() >= 2
    want = torch.where(far, x0 * (1 - 0.5 / 0.74), x0) / scale[0]
    assert torch.allclose(t2[0, 0], want.clamp(-1, 1), atol=1e-5)


def test_augmenter_fraction_contrast_absent_and_identity():
    aug, mean, std = _aug(fractions=(0.5,), p_absent=0.0)
    cond, tgt, scale = _batch(intact=(True,) * 4)
    c2, t2, _ = aug(cond, tgt, scale)
    raw0 = np.expm1(cond[0, 0].numpy() * std[0] + mean[0]).clip(0)
    med = float(torch.from_numpy(raw0).median())          # torch: lower of the two middle values
    raw2 = np.expm1(c2[0, 0].numpy() * std[0] + mean[0])
    np.testing.assert_allclose(raw2, np.clip(med + 0.5 * (raw0 - med), 0, None), atol=1e-4)
    assert float(c2[0, 3].mean()) == pytest.approx(fraction_z(0.5), abs=1e-6)
    # absent share 1: the tracks take the absent code, the fraction row still says 0.5
    aug2, *_ = _aug(fractions=(0.5,), p_absent=1.0)
    c3, _, _ = aug2(cond, tgt, scale)
    assert torch.all(c3[:, 0] == -2.0) and torch.all(c3[:, 1] == -2.0)
    assert float(c3[0, 3].mean()) == pytest.approx(fraction_z(0.5), abs=1e-6)
    # probability 0: nothing changes
    aug0, *_ = _aug(prob=0.0)
    c4, t4, n0 = aug0(cond, tgt, scale)
    assert n0 == 0 and torch.equal(c4, cond) and torch.equal(t4, tgt)
    # empty pixels (target exactly 0) stay empty
    tgt0 = tgt.clone()
    tgt0[0, 0, 3, :] = 0.0
    tgt0[0, 0, :, 3] = 0.0
    _, t5, _ = aug(cond, tgt0, scale)
    assert torch.all(t5[0, 0, 3, :] == 0.0)


def test_augmenter_requires_depleted_operator():
    with pytest.raises(ValueError, match="depleted"):
        CohesinAugmenter(HalfOp(), 1.0, 0.5, [0.0], 0.0, [0], [0.0], [1.0], [-1.0], 1, "cpu")


def _cohfix_raw(workdir, name, aug=False, op_path=None):
    raw = _cohesin_raw(workdir, name)
    raw["data"]["cohesin_unknown"] = False
    raw["data"]["cohesin_fraction"] = True
    smp = raw["data"]["samples"][0]
    raw["data"]["samples"] = [
        dict(smp, name="withcoh"),
        dict(smp, name="depl", cohesin_fraction=0.26, norm_like="withcoh")]
    raw["window"]["stride"] = 8                      # more windows than the 3 of the base fixture
    raw["train"].update(max_steps=60, eval_every=30, batch_size=2, track_dropout=0.5,
                        track_dropout_mode="uniform_count", val_regions=1)
    if aug:
        raw["train"].update(cohesin_aug=0.5, cohesin_aug_operator=str(op_path),
                            cohesin_aug_fractions=[0.0, 0.5], cohesin_aug_absent=0.5)
    return raw


def test_config_validation(workdir, tmp_path):
    with pytest.raises(ValueError, match="needs data.cohesin_fraction"):
        raw = _cohfix_raw(workdir, "cfgv")
        raw["data"]["cohesin_fraction"] = False
        Config.from_dict(raw)
    with pytest.raises(ValueError, match="norm_like"):
        raw = _cohfix_raw(workdir, "cfgv")
        raw["data"]["samples"][1]["norm_like"] = "nobody"
        Config.from_dict(raw)
    with pytest.raises(ValueError, match="cohesin_aug_operator"):
        raw = _cohfix_raw(workdir, "cfgv")
        raw["train"]["cohesin_aug"] = 0.3
        Config.from_dict(raw)
    with pytest.raises(ValueError, match="outside"):
        raw = _cohfix_raw(workdir, "cfgv")
        raw["data"]["samples"][1]["cohesin_fraction"] = 1.5
        Config.from_dict(raw)


def test_fraction_row_layout_and_values(workdir):
    on = Config.from_dict(_cohfix_raw(workdir, "rowv"))
    raw_off = _cohfix_raw(workdir, "rowv")
    raw_off["data"]["cohesin_fraction"] = False
    raw_off["data"]["samples"][1]["cohesin_fraction"] = 1.0
    off = Config.from_dict(raw_off)
    r_on, r_off = sym_row_names(on), sym_row_names(off)
    assert len(r_on) == len(r_off) + 1
    k = r_on.index("cohesin:fraction")
    assert k == len(on.track_names) + len(seq_feature_names(on)) - 1
    assert seq_feature_names(on)[-1] == FRACTION_ROW
    nb = CHROMS["chr1"] // BINSIZE
    for s, v in zip(on.data.samples, (1.0, 0.26)):
        a = assemble_chrom(on, s, sample_track_files(on, s), "chr1", nb)
        assert np.allclose(a[k], v)


def test_precompute_shares_scale_and_trains_with_augmentation(workdir, tmp_path, caplog):
    from flexcible import precompute as hprep
    from flexcible import train as htrain

    op = DegronOperator(width=4, dilations=(1, 2))
    save_operator(tmp_path / "op.pt", op, 0.26, BINSIZE)
    raw = _cohfix_raw(workdir, "cohfixtrain", aug=True, op_path=tmp_path / "op.pt")
    c = Config.from_dict(raw)
    hprep.build(c)
    cache = Path(c.train.cache_dir)
    import json
    sc_a = json.loads((cache / "norm_matrix_withcoh.json").read_text())["scale"]
    sc_b = json.loads((cache / "norm_matrix_depl.json").read_text())["scale"]
    assert sc_a == sc_b
    tn = TrackNormalizer.load(cache / "norm_tracks.json")
    ch = json.loads((cache / "channels.json").read_text())
    r = ch["sym_rows"].index("cohesin:fraction")
    assert tn.mean[r] == pytest.approx(math.log(2)) and tn.std[r] == pytest.approx(math.log(2))
    shard = sorted(cache.glob("depl_train_chr*.npz"))[0]
    cond = np.load(shard)["cond"]
    assert np.allclose(cond[:, r], fraction_row_value(0.26), atol=2e-3)       # float16
    shard = sorted(cache.glob("withcoh_train_chr*.npz"))[0]
    assert np.allclose(np.load(shard)["cond"][:, r], 0.0, atol=1e-6)
    with caplog.at_level(logging.INFO):
        htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()
    msgs = [m.getMessage() for m in caplog.records]
    assert any("cohesin_aug: p=0.50" in m for m in msgs)
    assert any("cohesin-augmented windows" in m and not m.endswith("windows 0") for m in msgs)
