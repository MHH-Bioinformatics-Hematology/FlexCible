import numpy as np
import pytest

from hicmultiome.precompute import load_mappable, mappable_fraction


def test_missing_mask_is_permissive(tmp_path):
    assert load_mappable(str(tmp_path / "nope.cool")) is None
    assert mappable_fraction(None, "chr1", 0, 512) == 1.0


def test_mask_is_read_and_windows_scored(tmp_path):
    cool = tmp_path / "m.cool"
    v = np.ones(1000, dtype=bool)
    v[200:700] = False                       # a big unmappable stretch
    np.savez_compressed(str(cool) + ".mappable.npz", chr1=v)
    m = load_mappable(str(cool))
    assert m is not None
    assert mappable_fraction(m, "chr1", 0, 100) == 1.0
    assert mappable_fraction(m, "chr1", 300, 100) == 0.0
    assert mappable_fraction(m, "chr1", 150, 100) == pytest.approx(0.5)


def test_chrom_name_variants_resolve(tmp_path):
    cool = tmp_path / "n.cool"
    np.savez_compressed(str(cool) + ".mappable.npz", chr7=np.ones(50, dtype=bool))
    m = load_mappable(str(cool))
    assert mappable_fraction(m, "7", 0, 50) == 1.0      # '7' vs 'chr7'
    assert mappable_fraction(m, "chrZZ", 0, 50) == 1.0  # unknown -> permissive
