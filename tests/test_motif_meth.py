"""Methylation-aware directed CTCF motif planes (motifs.motif_meth, registry 43).

Tiny synthetic genome of tests/test_pipeline.py; CPU, seconds. The planted
motif ACGTACGTAC (10 bp) holds two CpGs at offsets 1 and 5; GTACGTACGT (the
reverse-strand sites) holds two at offsets 3 and 7.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_pipeline import (BINSIZE, CHROMS, MOTIF, _ref_raw, _write_bigwig,  # noqa: E402,F401
                           workdir)

from hicmultiome import motifmeth as mm  # noqa: E402
from hicmultiome.config import Config  # noqa: E402

NB = {c: n // BINSIZE for c, n in CHROMS.items()}


def _cpg_file(path: Path, meth_by_site: dict[int, float], cov: int = 20) -> Path:
    """CpG table for chr1/chr2: the two CpGs of every planted site get the
    methylated fraction meth_by_site[site start]; all other positions absent."""
    out = {}
    for chrom in CHROMS:
        pos, c, m = [], [], []
        for start, frac in sorted(meth_by_site.items()):
            offs = (1, 5) if start in (5000, 20000, 40000) else (3, 7)
            for o in offs:
                pos.append(start + o); c.append(cov); m.append(int(round(cov * frac)))
        out[chrom + "_pos"] = np.array(pos, np.int32)
        out[chrom + "_cov"] = np.array(c, np.int32)
        out[chrom + "_meth"] = np.array(m, np.int32)
    np.savez_compressed(path, **out)
    return path


def _expected(fasta, chrom, workdir, cpg_file, nb):
    """Independent per-site loop (no searchsorted/cumsum) for the three rows."""
    from hicmultiome.motifs import cached_motif_positions
    fwd, rev = cached_motif_positions(fasta, chrom, MOTIF, str(workdir), 0.8,
                                      str(workdir / "gcache"))
    z = np.load(cpg_file)
    pos, cov, meth = z[chrom + "_pos"], z[chrom + "_cov"], z[chrom + "_meth"]
    out = np.zeros((3, nb))
    n_all = np.zeros(nb)
    n_cov = np.zeros(nb)
    for k, starts in enumerate((fwd, rev)):
        for s in starts:
            r = m = 0
            for p, cc, mc in zip(pos, cov, meth):
                if s <= p < s + 10 - 1:
                    r += cc; m += mc
            ok = r >= mm.MIN_SITE_COV
            out[k, s // BINSIZE] += 1.0 - m / r if ok else 1.0
            n_all[s // BINSIZE] += 1
            n_cov[s // BINSIZE] += ok
    out[2] = np.where(n_all > 0, n_cov / np.maximum(n_all, 1), 0)
    return out


def test_site_weights_and_covered_fraction(workdir):
    """Weight = 1 - methylated fraction of the motif CpGs; sites without
    covered CpG keep weight 1 and are not counted as covered."""
    f = _cpg_file(workdir / "cpg_a.npz",
                  {5000: 1.0, 20000: 0.5, 12000: 0.25, 30000: 0.0})   # 40000, 55000: no CpG
    nb = NB["chr1"]
    got = mm.methylated_motif_tracks(str(workdir / "genome.fa"), "chr1", MOTIF, str(workdir),
                                     nb, BINSIZE, 0.8, str(workdir / "gcache"), f)
    want = _expected(str(workdir / "genome.fa"), "chr1", workdir, f, nb)
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert got[0, 5] < 1.0 and got[0, 20] < got[0, 40] <= 1.0    # methylated sites are weaker
    assert got[2, 5] > 0 and got[2, 40] == 0.0                   # no CpG data at site 40
    # below the coverage floor a site stays unmodified
    f_low = _cpg_file(workdir / "cpg_low.npz", {5000: 1.0}, cov=3)       # 2 CpGs x 3 reads = 6 < 10
    low = mm.methylated_motif_tracks(str(workdir / "genome.fa"), "chr1", MOTIF, str(workdir),
                                     nb, BINSIZE, 0.8, str(workdir / "gcache"), f_low)
    assert low[2].sum() == 0


def test_no_cpg_file_equals_unmodified_hits(workdir):
    """A cell without CpG data: weighted planes equal the plain hit counts,
    covered = 0 (the state the model sees for HUVEC, NHEK, PC-9)."""
    from hicmultiome.motifs import cached_motif_tracks
    nb = NB["chr1"]
    got = mm.methylated_motif_tracks(str(workdir / "genome.fa"), "chr1", MOTIF, str(workdir),
                                     nb, BINSIZE, 0.8, str(workdir / "gcache"), None)
    plain = cached_motif_tracks(str(workdir / "genome.fa"), "chr1", MOTIF, str(workdir), nb,
                                BINSIZE, 0.8, str(workdir / "gcache"))
    np.testing.assert_array_equal(got[:2], plain)
    assert not got[2].any()


def _cfg(workdir, name, motif_meth, with_meth_track=True, **train):
    raw = _ref_raw(workdir, name, ref_cool=False)
    if with_meth_track:
        raw["data"]["tracks"].append({"name": "Meth", "type": "methylation"})
    raw["motifs"]["motif_meth"] = motif_meth
    raw["motifs"]["motif_meth_id"] = MOTIF
    raw["train"]["cache_dir"] = str(workdir / f"cache_{name}")
    raw["train"].update(train)
    return raw


def _samples(workdir, raw):
    """Three cells: 'cpg' (Meth + CpG.npz), 'meth_nocpg' (Meth bigwig, no CpG file),
    'nometh' (Meth declared missing)."""
    d1, d2 = workdir / "mm_cpg", workdir / "mm_nocpg"
    for d in (d1, d2):
        d.mkdir(exist_ok=True)
        _write_bigwig(d / "Meth.bigwig", {"chr1": [3, 6, 20, 41], "chr2": [4, 9]})
    _cpg_file(d1 / mm.CPG_FILE, {5000: 1.0, 20000: 0.5, 12000: 0.2, 30000: 0.0, 55000: 0.7})
    base = raw["data"]["samples"][0]
    raw["data"]["samples"] = [
        dict(base, name="cpg", extra_bigwig_dirs=[str(d1)]),
        dict(base, name="meth_nocpg", extra_bigwig_dirs=[str(d2)]),
        dict(base, name="nometh", missing_tracks=["Meth"])]
    return raw


def test_layout_default_off_and_on(workdir):
    from hicmultiome.precompute import seq_feature_names, sym_row_names
    off = Config.from_dict(_cfg(workdir, "lay_off", False))
    on = Config.from_dict(_cfg(workdir, "lay_on", True))
    assert off.active_motif_ids == [MOTIF] and not off.motif_meth_active
    assert on.active_motif_ids == [MOTIF, mm.meth_id(MOTIF)]
    r_off, r_on = sym_row_names(off), sym_row_names(on)
    assert len(r_on) == len(r_off) + 3                       # covered row + one fwd/rev pair
    assert r_on[-4:] == [f"motif:{MOTIF}:fwd", f"motif:{MOTIF}:rev",
                         f"motif:{MOTIF}:meth:fwd", f"motif:{MOTIF}:meth:rev"]
    assert seq_feature_names(on)[-1] == mm.COVERED_FEATURE
    assert r_on.index(mm.COVERED_ROW) == len(on.track_names) + len(seq_feature_names(on)) - 1
    with pytest.raises(ValueError, match="motif_meth_id"):
        bad = _cfg(workdir, "lay_bad", True)
        bad["motifs"]["motif_meth_id"] = "MA0000.0"
        Config.from_dict(bad).motif_ids


def test_assemble_rows_and_off_is_unchanged(workdir):
    """The three new rows come out of assemble_chrom; every other row equals
    the motif_meth-off assembly."""
    from hicmultiome.precompute import assemble_chrom, sample_track_files, sym_row_names
    raw_on = _samples(workdir, _cfg(workdir, "asm_on", True))
    raw_off = _samples(workdir, _cfg(workdir, "asm_off", False))
    on, off = Config.from_dict(raw_on), Config.from_dict(raw_off)
    names = sym_row_names(on)
    new = [names.index(n) for n in (mm.COVERED_ROW, f"motif:{MOTIF}:meth:fwd",
                                    f"motif:{MOTIF}:meth:rev")]
    nb = NB["chr1"]
    for s_on, s_off in zip(on.data.samples, off.data.samples):
        a = assemble_chrom(on, s_on, sample_track_files(on, s_on), "chr1", nb)
        b = assemble_chrom(off, s_off, sample_track_files(off, s_off), "chr1", nb)
        np.testing.assert_array_equal(np.delete(a, new, axis=0), b)
        fwd, rev = a[names.index(f"motif:{MOTIF}:fwd")], a[names.index(f"motif:{MOTIF}:rev")]
        if s_on.name == "cpg":
            assert a[new[0]].any() and (a[new[1]] < fwd).any() or (a[new[2]] < rev).any()
        else:                                                  # no CpG data: unmodified, covered 0
            assert not a[new[0]].any()
            np.testing.assert_array_equal(a[new[1]], fwd)
            np.testing.assert_array_equal(a[new[2]], rev)


def test_meth_dropout_equals_cell_without_cpg(workdir):
    """drop_methylation of a normalised CpG cell reproduces the rows of the same
    cell without CpG data (unmodified planes, covered absent) in the model's
    normalisation, for the numpy path (inference) and the torch path (training)
    including reverse-augmented windows."""
    from hicmultiome import train as htrain
    from hicmultiome.dataset import reverse_cond
    from hicmultiome.precompute import assemble_chrom, sample_track_files, sym_row_names
    from hicmultiome.tracks import TrackNormalizer
    raw = _samples(workdir, _cfg(workdir, "drop_eq", True))
    c = Config.from_dict(raw)
    names = sym_row_names(c)
    nb = NB["chr1"]
    arrs = [assemble_chrom(c, s, sample_track_files(c, s), "chr1", nb) for s in c.data.samples]
    tn = TrackNormalizer(transform="log1p", norm="zscore").fit(arrs, None)
    a_cpg, a_no = tn.apply(arrs[0]), tn.apply(arrs[1])      # same tracks, the second has no CpG file
    rows = mm.coupling_rows(names, MOTIF)
    # compare only the motif-meth rows: a_no has no CpG data
    d = mm.drop_methylation(a_cpg.copy(), tn, rows)
    keep = rows["meth"] + [rows["covered"]]
    np.testing.assert_allclose(d[keep], a_no[keep], atol=1e-5)
    assert not np.allclose(a_cpg[keep], a_no[keep])         # the CpG cell did differ before

    # torch path on a selected cond (Meth kept), window 0 dropped, window 1 kept,
    # window 2 dropped and reverse-augmented
    channels = {"sym_rows": names, "tracks": c.track_names,
                "seq_features": [n for n in names if n.startswith(("seq:", "motifmeth"))],
                "motif_ids": c.active_motif_ids}
    chip_rows = htrain.dropout_chip_rows(c, channels)
    mmc = htrain.motif_meth_coupling(c, channels, chip_rows, tn)
    assert mmc is not None and mmc[0] == c.track_names.index("Meth")
    n_motifs, offset = len(channels["motif_ids"]), len(c.track_names) + len(channels["seq_features"])
    cond = torch.from_numpy(np.stack([a_cpg, a_cpg, a_cpg]))
    rev = torch.tensor([False, False, True])
    cond = torch.where(rev[:, None, None], reverse_cond(cond, offset, n_motifs), cond)
    out = htrain.apply_motif_meth_drop(cond.clone(), torch.tensor([True, False, True]), mmc, rev)
    out = torch.where(rev[:, None, None], reverse_cond(out, offset, n_motifs), out).numpy()
    np.testing.assert_allclose(out[0][keep], a_no[keep], atol=1e-5)
    np.testing.assert_allclose(out[2][keep], a_no[keep], atol=1e-5)
    np.testing.assert_array_equal(out[1], a_cpg)


def test_select_tracks_keeps_the_methylation_rows(workdir):
    """Regression for the select_tracks bug of 2026-10-02 (Meth missing from
    the list dropped its row): with Meth in the list the methylation-weighted
    pair and the covered row reach the model input; without it the covered row
    remains (sequence-feature block) and the Meth track row is gone."""
    from hicmultiome import train as htrain
    from hicmultiome.precompute import sym_row_names, seq_feature_names
    raw = _cfg(workdir, "sel", True, select_tracks=["CTCF", "H3K27ac", "Meth"])
    raw["train"]["select_motifs"] = [MOTIF]
    c = Config.from_dict(raw)
    names = sym_row_names(c)
    channels = {"sym_rows": names, "tracks": c.track_names, "seq_features": seq_feature_names(c),
                "motif_ids": c.active_motif_ids}
    rows, offset, n_motifs, n_sym = htrain._select_rows(c, channels)
    for n in ("Meth", mm.COVERED_ROW, f"motif:{MOTIF}:meth:fwd", f"motif:{MOTIF}:meth:rev",
              f"motif:{MOTIF}:fwd"):
        assert names.index(n) in rows, n
    assert n_motifs == 2
    raw["train"]["select_tracks"] = ["CTCF", "H3K27ac"]       # the 2026-10-02 mistake
    rows2, *_ = htrain._select_rows(Config.from_dict(raw), channels)
    assert names.index("Meth") not in rows2


def _random_model(c, channels):
    from hicmultiome import train as htrain
    from hicmultiome.model import make_model, n_cond_channels
    rows, offset, n_motifs, n_sym = htrain._select_rows(c, channels)
    torch.manual_seed(0)
    model = make_model(c, n_cond_channels(n_sym, n_motifs, 0, c.model.cond_mode), 1, None)
    with torch.no_grad():                                    # non-trivial weights everywhere
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    return model.eval(), rows, offset, n_motifs


def test_methylation_channels_change_model_input_and_output(workdir):
    """Changing the CpG-derived rows (motif pair + covered) changes the cond the
    model receives and its prediction; changing the Meth track row does too; a
    model built with motif_meth off has fewer input channels."""
    from hicmultiome.dataset import split_cond
    from hicmultiome.model import n_cond_channels
    from hicmultiome.precompute import assemble_chrom, sample_track_files, sym_row_names, seq_feature_names
    from hicmultiome.tracks import TrackNormalizer
    raw = _samples(workdir, _cfg(workdir, "net", True, select_tracks=["CTCF", "H3K27ac", "Meth"]))
    raw["train"]["select_motifs"] = [MOTIF]
    c = Config.from_dict(raw)
    names = sym_row_names(c)
    channels = {"sym_rows": names, "tracks": c.track_names, "seq_features": seq_feature_names(c),
                "motif_ids": c.active_motif_ids}
    arrs = [assemble_chrom(c, s, sample_track_files(c, s), "chr1", NB["chr1"])
            for s in c.data.samples]
    tn = TrackNormalizer(transform="log1p", norm="zscore").fit(arrs, None)
    model, rows, offset, n_motifs = _random_model(c, channels)
    W = c.window.size
    sym = torch.from_numpy(tn.apply(arrs[0])[rows][:, :W])[None]       # CpG cell, first window
    sym_no = sym.clone()
    r_mm = [rows.index(names.index(n)) for n in (f"motif:{MOTIF}:meth:fwd", f"motif:{MOTIF}:meth:rev",
                                                 mm.COVERED_ROW)]
    # the CpG cell differs from its CpG-free version in those rows
    cond_nocpg = torch.from_numpy(tn.apply(arrs[1])[rows][:, :W])[None]
    assert (sym[:, r_mm] - cond_nocpg[:, r_mm]).abs().max() > 0.05
    # and the model sees a different input channel count than the off model
    off_in = n_cond_channels(len(rows) - 3, n_motifs - 1, 0, c.model.cond_mode)
    assert n_cond_channels(len(rows), n_motifs, 0, c.model.cond_mode) > off_in

    def run(cond):
        s, p = split_cond(cond, offset, n_motifs)
        with torch.no_grad():
            return model(cond, p, None, None, None)
    base = run(sym)
    assert not torch.allclose(base, run(cond_nocpg), atol=1e-6)         # CpG rows reach the output
    sym_no[:, r_mm] = cond_nocpg[:, r_mm]
    assert not torch.allclose(base, run(sym_no), atol=1e-6)
    sym_meth = sym.clone()
    sym_meth[:, rows.index(names.index("Meth"))] += 1.0                 # the Meth track row
    assert not torch.allclose(base, run(sym_meth), atol=1e-6)


def test_precompute_and_train_with_motif_meth(workdir, caplog):
    """Cache with three cells (CpG, Meth without CpG, no Meth), track dropout 0.5,
    select_tracks including Meth: trains, logs the coupling, and the cache holds
    the extra rows."""
    import json
    import logging

    from hicmultiome import precompute as hprep
    from hicmultiome import train as htrain
    raw = _samples(workdir, _cfg(workdir, "e2e", True, select_tracks=["CTCF", "H3K27ac", "Meth"],
                                 track_dropout=0.5, track_dropout_mode="uniform_count",
                                 max_steps=6, eval_every=3, val_regions=2))
    raw["train"]["select_motifs"] = [MOTIF]
    c = Config.from_dict(raw)
    hprep.build(c)
    ch = json.loads((Path(c.train.cache_dir) / "channels.json").read_text())
    assert mm.COVERED_ROW in ch["sym_rows"] and mm.meth_id(MOTIF) in ch["motif_ids"]
    with caplog.at_level(logging.INFO):
        htrain.train(c)
    assert (Path(c.train.output_dir) / "best").exists()
    assert any("motif_meth: Meth dropout removes" in r.getMessage() for r in caplog.records)
