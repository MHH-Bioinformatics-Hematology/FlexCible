"""Resolution conditioning: the lookup embedding accepts only trained bin sizes, the continuous
embedding (log2 of the bin size) accepts any bin size and changes the output with it."""
import pytest
import torch

from flexcible.model import HiCModel, build_backbone, n_cond_channels


def _model(mode):
    bb = build_backbone("unet", in_ch=n_cond_channels(2, 0, 0, "full"), out_ch=1, base=8, depth=2, film_dim=8)
    return HiCModel(bb, cond_mode="full", resolutions=[2000, 10000], res_embed_dim=8,
                    res_embed_mode=mode).eval()


def _inputs(w=32):
    torch.manual_seed(0)
    cond = torch.randn(2, 2, w)
    return cond, None


def test_lookup_rejects_untrained_resolution():
    m = _model("lookup")
    cond, pairs = _inputs()
    m(cond, pairs, None, None, None, res=2000)
    with pytest.raises(ValueError, match="not among the trained"):
        m(cond, pairs, None, None, None, res=4000)


def test_continuous_accepts_any_resolution_and_depends_on_it():
    m = _model("continuous")
    # FiLM starts at zero (no effect at initialisation); give it weights so the bin size matters
    torch.manual_seed(1)
    for lin in m.backbone.film:
        torch.nn.init.normal_(lin.weight, std=0.5)
    cond, pairs = _inputs()
    a = m(cond, pairs, None, None, None, res=2000)
    b = m(cond, pairs, None, None, None, res=4000)       # never trained, must still run
    c = m(cond, pairs, None, None, None, res=2000)
    assert a.shape == b.shape == (2, 1, 32, 32)
    assert torch.allclose(a, c)
    assert not torch.allclose(a, b)
    e = m.res_log2(torch.tensor([2000.0, 8000.0]), 2, cond.device)
    assert torch.allclose(e[:, 0], torch.tensor([1.0, 3.0]))


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="res_embed_mode"):
        _model("other")
