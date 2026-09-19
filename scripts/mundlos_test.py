"""Reproduce the Franke 2016 neo-TAD result with a cell-type-agnostic model.

The model holds no cell-type identity: the only thing that changes between a
stem-cell prediction and a limb-bud prediction is which ChIP tracks are fed in.
This script therefore does two things:

1. Cell-type transfer: predict the Sox9 landscape from E12.5 LIMB tracks and
   ask whether the predicted TAD structure matches limb-bud measurement better
   than the same model fed stem-cell tracks.

2. The perturbation: predict the Dup-L allele and fold the edited genome back
   onto reference coordinates the way sequencing reads from a duplication
   actually map (both copies land on the source interval), so the prediction
   and the published matrix live in the same coordinate system.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome.loops import boundaries, insulation_profile   # noqa: E402

MUN = Path.home() / "data/hic-multiome/mundlos"
RES_M, LO_M9 = 10000, 109010000          # measured: mm9, 10 kb
LO_M10 = 109148686                       # same locus lifted to mm10
DUP10 = (111098686, 112618686)           # Dup-L in mm10


def fold_edited_to_reference(pred: np.ndarray, dup: tuple[int, int],
                             res: int, lo_bp: int) -> np.ndarray:
    """Project an edited-genome matrix back onto reference coordinates.

    Reads from a tandem duplication map to the SAME reference interval, so the
    published matrix is the sum over both copies. Folding the prediction the
    same way is what makes the two comparable.
    """
    a, b = dup[0] // res, dup[1] // res
    dup_len = b - a
    n_ed = pred.shape[0]
    n_ref = n_ed - dup_len
    idx = np.empty(n_ed, dtype=np.int64)
    idx[:b] = np.arange(b)                       # up to and including copy 1
    idx[b:b + dup_len] = np.arange(a, b)         # copy 2 -> the source interval
    idx[b + dup_len:] = np.arange(b, n_ref)      # everything downstream
    out = np.zeros((n_ref, n_ref))
    np.add.at(out, (idx[:, None], idx[None, :]), pred)
    return out


def insulation_agreement(a: np.ndarray, b: np.ndarray, res_a: int, res_b: int,
                         window_bp: int = 200000) -> dict:
    """Compare TAD structure across matrices of different resolution by
    resampling both insulation profiles onto a common grid."""
    pa = insulation_profile(a, max(2, window_bp // res_a))
    pb = insulation_profile(b, max(2, window_bp // res_b))
    xa = np.arange(len(pa)) * res_a
    xb = np.arange(len(pb)) * res_b
    grid = np.arange(0, min(xa[-1], xb[-1]), 25000)
    ia = np.interp(grid, xa, np.nan_to_num(pa))
    ib = np.interp(grid, xb, np.nan_to_num(pb))
    ok = np.isfinite(ia) & np.isfinite(ib)
    r = float(np.corrcoef(ia[ok], ib[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    ba = set(np.round(np.array(boundaries(pa)) * res_a / 25000).astype(int))
    bb = set(np.round(np.array(boundaries(pb)) * res_b / 25000).astype(int))
    tol = 2
    matched = sum(1 for x in ba if any(abs(x - y) <= tol for y in bb))
    prec = matched / max(1, len(ba))
    rec = sum(1 for y in bb if any(abs(x - y) <= tol for x in ba)) / max(1, len(bb))
    return {"insulation_pearson": r, "n_boundaries_a": len(ba),
            "n_boundaries_b": len(bb),
            "boundary_f1": 2 * prec * rec / max(1e-9, prec + rec)}
