"""Window prior network P: Hi-C of a window from the CONTENT of its marks, no coordinates (registry 76).

The positional consensus input of the reference models is the mean of other cells' Hi-C at the same
genomic coordinates; it cannot move to another organism, whose syntenic regions differ. P is a function
learned across the whole genome that maps what a window looks like (per-mark signal along the window) to
the typical Hi-C of such a window (the distance-normalised O/E map the consensus holds):

    P(marks of the window) -> log O/E (W x W)

Position-free by construction and by test (tests/test_windowprior.py):
  * the module sees (marks (B, M, W), optional motif pairs (B, K, 2, W)) and nothing else: no chromosome
    id, no start bin, no learned positional embedding, no per-locus table, no parameter whose shape
    depends on the chromosome or genome size;
  * the only coordinates inside a window are relative (the |i-j|/W distance plane of the U-Net family,
    zero padding at the window edges), identical for every genomic position;
  * production over a chromosome (predict_chrom) slides windows on a fixed grid and blends the overlaps
    with a Hann taper; the result is the same function of the signal when the signal is rolled by whole
    windows or the chromosome is renamed (controls in scripts/window_prior.py, max abs difference 0).

Inputs (the same representation as the dref channels of registry 66, flexcible.diffmarks):
  ref   mode "ref": per mark the mean over the reference set (without the sample and its cell
        identity) of the per-sample quantile-anchored track (q50 -> 0, q99 -> 1), 0 for a mark the sample
        or its reference lacks. The reference of a training cell is its leave-one-out set, that of a
        held-out cell the 12 training cells.
  own   mode "own": the quantile-anchored track of the sample itself, 0 for a missing mark. Needs no other
        cell, so the function transfers to an organism with no reference Hi-C (and no reference marks).
Optional (ablation switch): the CTCF motif planes (fwd / rev of the four motifs, normalised as the model's
cond rows), a function of sequence content, not of position.

Target: the leave-one-out consensus Hi-C window in raw log O/E (consensus_loo_<cell>_proper_<k>kb.cool),
divided by PRIOR_SCALE and clipped to [-1, 1]. The output of P is written as a cooler (raw log O/E, band of
one window) that replaces the positional consensus as `samples[].reference_cool` (model.reference) or enters
as `samples[].prior_cool` (model.prior); no module of the training or prediction code changes.

Cross-fitting: a chromosome is predicted by a P that was trained without that chromosome (odd / even
autosomes), so a training window never receives a prior fitted on its own region.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

PRIOR_SCALE = 2.0                      # raw log O/E value mapped to +-1 (the cells' own scales are 1.9 to 2.0)
MODES = ("ref", "own")
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
FOLDS = ("odd", "even", "full")


# --------------------------------------------------------------------------- network
def make_prior_net(n_marks: int, n_motifs: int = 0, base: int = 24, depth: int = 4):
    """HiCModel of the U-Net family without reference, prior, encoder, distance bias or resolution
    embedding: forward(marks, pairs) only. n_motifs: number of motif fwd/rev pairs (0 = marks only)."""
    from .model import HiCModel, HiCUNet, n_cond_channels
    in_ch = n_cond_channels(n_marks, n_motifs, 0, "full")
    return HiCModel(HiCUNet(in_ch, out_ch=1, base=base, depth=depth), "full")


def prior_forward(net, marks: torch.Tensor, pairs: torch.Tensor | None = None) -> torch.Tensor:
    """(B, M, W) marks [+ (B, K, 2, W) motif pairs] -> (B, W, W) in [-1, 1] (raw log O/E / PRIOR_SCALE)."""
    return net(marks, pairs)[:, 0]


# --------------------------------------------------------------------------- chromosome folds
def fold_of_chrom(chrom: str) -> str:
    """'odd' or 'even' autosome number (chrX and others: 'even')."""
    c = str(chrom)[3:] if str(chrom).startswith("chr") else str(chrom)
    return "odd" if c.isdigit() and int(c) % 2 == 1 else "even"


def train_chroms_of(fold: str, held: tuple = ()) -> list[str]:
    """Chromosomes a P of `fold` is trained on: odd, even, or all autosomes, minus `held` (the
    held-out chromosomes of the chrho models)."""
    if fold not in FOLDS:
        raise ValueError(f"fold {fold!r} not in {FOLDS}")
    out = [c for c in AUTOSOMES if fold == "full" or fold_of_chrom(c) == fold]
    return [c for c in out if c not in set(held)]


def predictor_fold(chrom: str, mode: str = "oof") -> str:
    """Fold of the P that predicts `chrom`: the one trained WITHOUT it (cross-fitting), or 'full'."""
    if mode == "full":
        return "full"
    return "even" if fold_of_chrom(chrom) == "odd" else "odd"


# --------------------------------------------------------------------------- window inputs
def reference_z(dm, members: list, chrom: str):
    """(mean over `members` (SampleSpec) of the quantile-anchored marks on `chrom` where a member has the
    mark, count of such members (M,)); (None, counts) when no member has the chromosome."""
    from .diffmarks import chrom_key
    c = chrom_key(chrom)
    acc, cnt = None, np.zeros(len(dm.marks))
    for s in members:
        sm = dm.sample_marks(s)
        if c not in sm.z:
            continue
        z = sm.z[c] * sm.valid[:, None]
        acc = z if acc is None else acc + z
        cnt += sm.valid
    if acc is None:
        return None, cnt
    return acc / np.maximum(cnt, 1)[:, None], cnt


def mark_rows(dm, sample, chrom: str, nb: int, mode: str = "ref", members: list | None = None) -> np.ndarray:
    """(M, nb) float32 input rows of one sample chromosome.

    mode 'ref': the dref rows of flexcible.diffmarks (identical values: the same reference set, the same
    zero for a mark the sample or the whole reference lacks). `members` overrides the reference set
    (SampleSpec list, e.g. the other samples of another organism); default dm.members(sample).
    mode 'own': the sample's own anchored marks, 0 for a missing mark."""
    from .diffmarks import chrom_key
    c = chrom_key(chrom)
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    sm = dm.sample_marks(sample)
    out = np.zeros((len(dm.marks), nb), np.float32)
    if c not in sm.z:
        return out
    n = min(nb, sm.z[c].shape[1])
    if mode == "own":
        out[:, :n] = sm.z[c][:, :n] * sm.valid[:, None]
        return out
    if members is None:
        members = [dm._byname[x] for x in dm.members(sample)]
    ref, cnt = reference_z(dm, members, c)
    if ref is None:
        return out
    ok = sm.valid & (cnt > 0)
    out[:, :n] = ref[:, :n] * ok[:, None]
    return out


def drop_marks(x: torch.Tensor, gen: torch.Generator, p: float = 0.5) -> torch.Tensor:
    """Mark dropout of a batch (B, M, W): with probability p a window loses a uniformly drawn number
    (1 .. M-1) of marks, set to 0 ("absent"), the uniform_count mode of the main models."""
    b, m, _ = x.shape
    if p <= 0 or m < 2:
        return x
    apply = torch.rand(b, generator=gen) < p
    k = torch.randint(1, m, (b,), generator=gen)
    keep = torch.ones(b, m)
    for i in torch.nonzero(apply).flatten().tolist():
        idx = torch.randperm(m, generator=gen)[: int(k[i])]
        keep[i, idx] = 0.0
    return x * keep[:, :, None].to(x.device, x.dtype)


def reverse_inputs(marks: torch.Tensor, pairs: torch.Tensor | None):
    """Genomic reversal of a window: flip along the window, swap each motif's fwd / rev strand."""
    m = torch.flip(marks, dims=[-1])
    p = None if pairs is None else torch.flip(pairs, dims=[-1]).flip(dims=[2])
    return m, p


# --------------------------------------------------------------------------- production
def chrom_starts(nb: int, size: int, stride: int) -> list[int]:
    """Window starts of the production grid: stride steps, the last window ends at the chromosome end."""
    if nb <= size:
        return [0]
    st = list(range(0, nb - size + 1, stride))
    if st[-1] != nb - size:
        st.append(nb - size)
    return st


@torch.no_grad()
def predict_chrom(net, marks: np.ndarray, pairs: np.ndarray | None, size: int, stride: int | None = None,
                  device: str = "cpu", batch: int = 32, tta: bool = True):
    """Blend P over one chromosome. marks (M, nb) [, pairs (K, 2, nb)] -> BandedAccumulator of raw log O/E
    (band = size). A function of the signal only: same values for a rolled or relabelled chromosome."""
    from .matrix import BandedAccumulator, hann2d
    stride = stride or size // 2
    nb = marks.shape[1]
    pad = max(0, size - nb)
    if pad:
        marks = np.concatenate([marks, np.zeros((marks.shape[0], pad), marks.dtype)], 1)
        if pairs is not None:
            pairs = np.concatenate([pairs, np.zeros((*pairs.shape[:2], pad), pairs.dtype)], 2)
    acc = BandedAccumulator(nb, size)
    wt = hann2d(size)
    starts = chrom_starts(marks.shape[1], size, stride)
    net.eval()
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        x = torch.from_numpy(np.stack([marks[:, s:s + size] for s in ss])).float().to(device)
        pr = (torch.from_numpy(np.stack([pairs[:, :, s:s + size] for s in ss])).float().to(device)
              if pairs is not None else None)
        o = prior_forward(net, x, pr)
        if tta:
            xr, pr_r = reverse_inputs(x, pr)
            o = 0.5 * (o + torch.flip(prior_forward(net, xr, pr_r), dims=[-1, -2]))
        o = (o.float().cpu().numpy() * PRIOR_SCALE)
        for s, blk in zip(ss, o):
            acc.add_block(s, 0.5 * (blk + blk.T), wt)
    return acc


def write_prior_cool(out: str | Path, template: str | Path, accs: dict, meta: dict) -> Path:
    """Cooler of the blended prior maps (chrom -> BandedAccumulator), bin table of `template`, float32
    counts, written to <out>.part and renamed; <out>.json = meta (key 'cells': the cells whose Hi-C is in
    the map's INPUT, for flexcible.reference.assert_reference_excludes)."""
    import cooler
    import pandas as pd

    grid = cooler.Cooler(str(template))
    bins = grid.bins()[:][["chrom", "start", "end"]]

    def chunks():
        for chrom in grid.chromnames:
            if chrom not in accs:
                continue
            go = int(grid.offset(chrom))
            b1, b2, v = accs[chrom].to_pixels(go)
            if len(b1):
                yield pd.DataFrame({"bin1_id": b1, "bin2_id": b2, "count": v.astype(np.float32)})

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    cooler.create_cooler(str(part), bins=bins, pixels=chunks(), dtypes={"count": np.float32}, ordered=True)
    part.rename(out)
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    return out


def shift_difference(net, marks: np.ndarray, pairs: np.ndarray | None, size: int, shift_strides: int,
                     device: str = "cpu") -> tuple[float, int]:
    """Position-free control. The signal of a chromosome is rolled by `shift_strides` window strides (a whole
    number of windows of the production grid: the grid stays aligned, only the genomic coordinates of
    every window change), P is run on both, and the maps are compared after undoing the roll, away from
    the roll seam and the chromosome ends. Returns (maximum absolute difference of the log O/E maps,
    number of pixels compared); a function of the content gives 0 up to float rounding."""
    stride = size // 2
    shift = int(shift_strides) * stride
    nb = marks.shape[1]
    A = predict_chrom(net, marks, pairs, size, stride, device)
    B = predict_chrom(net, np.roll(marks, shift, axis=1),
                      None if pairs is None else np.roll(pairs, shift, axis=2), size, stride, device)
    lo, hi = size, nb - shift - size
    if hi - lo < 2 * size:
        return 0.0, 0
    mx, n = 0.0, 0
    for off in range(size):
        i = np.arange(lo, hi - off)
        a = A.num[i, off] / np.maximum(A.den[i, off], 1e-9)
        b = B.num[i + shift, off] / np.maximum(B.den[i + shift, off], 1e-9)
        mx = max(mx, float(np.abs(a - b).max()))
        n += len(i)
    return mx, n
