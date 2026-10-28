"""Primary-AML helpers: quantile matching of input tracks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ap = pytest.importorskip("aml_primary")


def _q(ref):
    return np.quantile(ref, np.linspace(0, 1, ap.NQ))


def test_quantile_map_is_monotone_and_matches_the_reference():
    rng = np.random.default_rng(0)
    ref = rng.lognormal(0, 1, 200_000)
    src = rng.gamma(2.0, 3.0, 100_000)                 # another scale entirely
    f = ap.quantile_map(src, _q(ref))
    out = f(src)
    order = np.argsort(src)
    assert np.all(np.diff(out[order]) >= -1e-9)        # order kept
    assert np.median(out) == pytest.approx(np.median(ref), rel=0.02)
    assert np.mean(out) == pytest.approx(np.mean(ref), rel=0.02)


def test_quantile_map_ties_get_the_block_mean():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 10, 100_000)                  # mean of lower half = 2.5
    src = np.concatenate([np.zeros(50_000), rng.uniform(1, 2, 50_000)])
    f = ap.quantile_map(src, _q(ref))
    assert f(np.array([0.0]))[0] == pytest.approx(2.5, abs=0.05)
    assert f(src).mean() == pytest.approx(ref.mean(), rel=0.01)


def test_aml_consensus_leakage_guard(tmp_path):
    from flexcible.reference import assert_reference_excludes
    import json
    ref = tmp_path / "amlcons_lopo_xu_103_25kb.cool"
    ref.write_text("")
    cells = [ap.guard_name(p) for p in ["xu_018", "xu_673", "xu_HSPC-050"]]
    (tmp_path / (ref.name + ".json")).write_text(json.dumps({"cells": cells}))
    assert ap.guard_name("xu_103") == "amlxu103"
    assert_reference_excludes(ref, ap.guard_name("xu_103"))          # left out: allowed
    with pytest.raises(ValueError):
        assert_reference_excludes(ref, ap.guard_name("xu_673"))      # contained: rejected
    assert ap.guard_name("xu_10") != ap.guard_name("xu_103")


def test_xu_methylation_channel_follows_the_encode_convention(tmp_path, monkeypatch):
    pyBigWig = pytest.importorskip("pyBigWig")
    import aml_primary_meth as am
    f = tmp_path / "GSM1_allc_5.CGN.10bp.bw"
    bw = pyBigWig.open(str(f), "w")
    bw.addHeader([("chr1", 5000)])
    # kb 0: five covered 10 bp bins at 0.5; kb 1: two covered bins (below MIN_BINS)
    bw.addEntries(["chr1"] * 7, [0, 100, 200, 300, 400, 1000, 1100],
                  ends=[10, 110, 210, 310, 410, 1010, 1110], values=[0.5] * 5 + [1.0, 1.0])
    bw.close()
    monkeypatch.setattr(am.me, "chrom_sizes", lambda: {"chr1": 5000})
    monkeypatch.setattr(am, "MD", tmp_path / "meth")
    assert am.build_one(("xu_005", f)) == "xu_005"
    out = pyBigWig.open(str(tmp_path / "meth" / "xu_005" / "Meth.bigwig"))
    v = out.values("chr1", 0, 3000)
    assert v[0] == pytest.approx(1.5)           # 1 + mean of the covered bins
    assert v[1500] == pytest.approx(1.5)        # too few bins: 1 + genome mean (0.5)
    assert min(v) >= 1.0                        # a raw zero stays "no data"
    assert am.wgbs_files.__doc__
