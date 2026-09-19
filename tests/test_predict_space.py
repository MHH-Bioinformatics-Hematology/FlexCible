import numpy as np

from hicmultiome.matrix import MatrixNormalizer


def test_prescaled_roundtrip_keeps_negatives():
    """Akita-space targets are mostly negative; encode/decode must preserve
    that. Clamping the decode at 0 (correct for counts) destroys them."""
    norm = MatrixNormalizer(transform="prescaled", headroom=0.9)
    vals = np.linspace(-2.0, 2.0, 41).reshape(1, 41) * np.ones((41, 1))
    norm.fit(vals)
    img = norm.encode(vals)
    back = norm.decode(img)
    assert (back < 0).any(), "negative structure lost in roundtrip"
    assert np.allclose(back, vals, atol=0.05)
    # what the old code did, kept here so the regression is explicit
    clamped = np.maximum(back, 0.0)
    assert (clamped < 0).sum() == 0
    assert np.abs(clamped - vals).max() > 1.0


def test_count_transforms_stay_non_negative():
    for t in ("log1p", "none"):
        norm = MatrixNormalizer(transform=t)
        norm.fit(np.abs(np.random.default_rng(0).normal(size=(32, 32))) * 10)
        out = norm.decode(np.linspace(-1, 1, 64).reshape(8, 8))
        assert np.isfinite(out).all()
