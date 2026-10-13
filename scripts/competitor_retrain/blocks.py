"""Scoring blocks and competitor registration of the retrained C.Origami and Epiphany
(registry section 77), plugged into scripts/hic_measures.py (--chrho-competitor-retrain).

A block is one cell, one grid (10 or 25 kb), one test chromosome (2, 3, 11, 14, 17) and one competitor
family; its sources are our chrho models, the baselines of hic_measures, and for the family the
pretrained model on the cell's inputs, the retrained model, and where it exists the released
predictions of the original papers:

  chrho_rtco   C.Origami, band = our model window (1.91 Mb at 10 kb, 1.975 Mb at 25 kb; reach 2.1 Mb)
  chrho_rtep   Epiphany, band 99 bins at 10 kb, 39 at 25 kb (reach 990 kb)

Competitor names: corigami_pre_<cell>, corigami_re_<cell>, epiphany_pre_<cell>, epiphany_re_<cell>
(per-chromosome O/E cools of runs/benchmark/competitor_oe, built by predict_*.py and finalize.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

RES = (10000, 25000)


def comps(family: str, cell: str) -> list[str]:
    pre = f"{family}_pre_{cell}"
    out = [pre, f"{family}_re_{cell}"]
    if family == "corigami" and cell == "imr90":
        out.append("corigami_imr90_8kb")          # released predictions on the authors' own IMR-90 tracks
    if family == "epiphany" and cell == "gm12878":
        out.append("epiphany_gm12878_10kb")       # released predictions on the authors' GM12878 tracks
    return out


def rows(res_list=RES, cells=None, chroms=None):
    """(set, cell, res, chrom, band bins, competitors) as hic_measures.chrho_blocks()."""
    out = []
    for res in res_list:
        for cell in (cells or C.ALL_CELLS):
            for ch in (chroms or C.HELD):
                out.append(("chrho_rtco", cell, res, ch, None, comps("corigami", cell)))
                out.append(("chrho_rtep", cell, res, ch, 990_000 // res, comps("epiphany", cell)))
    return out


def specs():
    """benchmark_competitors.COMPETITORS entries of every name used by rows()."""
    d = {}
    for cell in C.ALL_CELLS:
        for tag, lab in (("pre", "pretrained"), ("re", "retrained (our split)")):
            d[f"corigami_{tag}_{cell}"] = dict(
                label=f"C.Origami {lab}", path=C.COMP_OE / f"corigami_{tag}_{cell}_{{chrom}}.cool",
                per_chrom=True, build="hg38", res=C.PIX, reach=2_097_152, space="counts",
                heldout=C.HELD, cell=cell, chroms=C.HELD)
            d[f"epiphany_{tag}_{cell}"] = dict(
                label=f"Epiphany {lab}", path=C.COMP_OE / f"epiphany_{tag}_{cell}_{{chrom}}.cool",
                per_chrom=True, build="hg38", res=10000, reach=990_000, space="counts",
                heldout=C.HELD, cell=cell, chroms=C.HELD)
    return d


def register():
    sys.path.insert(0, str(HERE.parent))
    import benchmark_competitors as bc
    bc.COMPETITORS.update(specs())
    return bc
