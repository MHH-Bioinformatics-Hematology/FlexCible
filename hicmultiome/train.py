"""Train the multi-omic Hi-C U-Net -- single GPU or multi-GPU (DDP).

Multi-GPU: launch with torchrun, one process per GPU; the window cache is small
enough that every rank keeps its own stripe fully resident on its GPU:

    torchrun --nproc_per_node=2 -m hicmultiome.train --config configs/multiome_25kb.yaml

Single GPU:  python -m hicmultiome.train --config configs/multiome_25kb.yaml

Loss: L1+MSE on the Hi-C channel, plus (when capture data is configured) a
bait-masked L1+MSE on the capture-Hi-C channel weighted by
capture.loss_weight. Validation metric (checkpoint selection): mean
per-genomic-distance Pearson of the Hi-C channel on held-out windows.
A learning-curve CSV and PNG are written to the output dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import Config
from .dataset import WindowDataset, reverse_cond, split_cond
from . import motifmeth as mm
from .losses import build_loss
from .model import (HiCModel, HiCUNet, SeqEncoder, build_backbone,
                    build_spatial_cond, make_model, n_cond_channels)
from .motifmeth import meth_id
from .sequence import window_codes
from .utils import get_logger, set_seed

log = get_logger()


def ddp_env():
    """(rank, world_size, local_rank); (0, 1, 0) when not under torchrun."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


class LazyRows:
    """Rows of a memory-mapped array (train.stage_dir), indexed like a tensor
    without loading it: row selections compose, `[:, rows, :]` selects
    columns (the cond rows of select_tracks), `.to(device)` only records the
    device, and indexing with a torch tensor after `.to` loads those rows and
    returns a tensor on that device (one batch at a time)."""

    def __init__(self, arr, rows=None, cols=None, device=None, dtype=None):
        self.arr, self.cols, self.device, self.dtype = arr, cols, device, dtype
        self.rows = np.arange(arr.shape[0]) if rows is None else np.asarray(rows)

    @property
    def shape(self):
        s = list(self.arr.shape)
        s[0] = len(self.rows)
        if self.cols is not None:
            s[1] = len(self.cols)
        return tuple(s)

    def _new(self, rows=None, cols=None, device=None):
        return LazyRows(self.arr, self.rows if rows is None else rows,
                        self.cols if cols is None else cols,
                        self.device if device is None else device, self.dtype)

    def materialize(self, rows=None) -> torch.Tensor:
        r = self.rows if rows is None else self.rows[rows]
        order = np.argsort(r, kind="stable")
        x = np.asarray(self.arr[r[order]])
        if self.cols is not None:
            x = x[:, self.cols]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        t = torch.from_numpy(np.ascontiguousarray(x[inv]))
        return t.to(self.dtype) if self.dtype is not None else t

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) >= 2 and key[0] == slice(None) and all(k == slice(None) for k in key[2:]):
                return self._new(cols=list(key[1]))
            raise IndexError("LazyRows supports [rows] and [:, cols, :]")
        if isinstance(key, torch.Tensor):
            t = self.materialize(key.detach().cpu().numpy())
            return t.to(self.device) if self.device is not None else t
        if isinstance(key, slice):
            return self._new(rows=self.rows[key])
        return self._new(rows=self.rows[np.asarray(key)])

    def to(self, device):
        return self._new(device=device)


def _dense(v, device):
    """A validation tensor on device (LazyRows loaded in full)."""
    return (v.materialize() if isinstance(v, LazyRows) else v).to(device)


STAGE_KEYS = ("target", "cond", "bait", "ref", "prior")


def stage_cache(cache_dir, split: str, stage_dir) -> Path:
    """Write the split of a cache uncompressed as float16 .npy memory maps
    (bait uint8) in WindowDataset index order, one shard decompressed at a
    time, into stage_dir/<cache name>/<split>/ (restartable; meta.json last)."""
    ds = WindowDataset(cache_dir, split)
    d = Path(stage_dir) / Path(cache_dir).name / split
    man = Path(cache_dir) / "manifest.json"
    meta = {"cache": str(cache_dir), "split": split, "n": len(ds),
            "manifest_mtime": man.stat().st_mtime,
            "index": [[x[2], x[3], x[4]] for x in ds._index]}
    if (d / "meta.json").exists():
        old = json.loads((d / "meta.json").read_text())
        if old["n"] == meta["n"] and old["manifest_mtime"] == meta["manifest_mtime"]:
            return d
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").unlink(missing_ok=True)
    n, w = len(ds), ds.window_size
    shapes = {"target": ((n, ds.n_out, w, w), np.float16), "cond": ((n, ds.n_sym, w), np.float16),
              "bait": ((n, w), np.uint8)}
    if ds.reference:
        shapes["ref"] = ((n, w, w), np.float16)
    if ds.prior:
        shapes["prior"] = ((n, w, w), np.float16)
    mm = {k: np.lib.format.open_memmap(d / f"{k}.npy.part", "w+", dt, sh)
          for k, (sh, dt) in shapes.items()}
    for path, pairs in ds.grouped_index().items():
        z = np.load(path)
        rows = np.array([r for r, _ in pairs])
        loc = np.array([l for _, l in pairs])
        for k in mm:
            if k == "bait":
                mm[k][rows] = z["bait"][loc] if "bait" in z else 0
            else:
                mm[k][rows] = z[k][loc]
        log.info("staged %s", Path(path).name)
    for k, m in mm.items():
        m.flush()
        del m
        (d / f"{k}.npy.part").rename(d / f"{k}.npy")
    (d / "meta.json").write_text(json.dumps(meta))
    log.info("staged %s %s: %d windows -> %s", Path(cache_dir).name, split, n, d)
    return d


def _stack(cache_dir, split, val_fraction, want_val, seed, chroms=None,
           dtype=torch.float32, stage_dir=None):
    ds = WindowDataset(cache_dir, split, val_fraction, want_val, seed, chroms=chroms)
    if len(ds) == 0:
        return None, ds
    if stage_dir is not None:
        d = Path(stage_dir) / Path(cache_dir).name / split
        if not (d / "meta.json").exists():
            raise RuntimeError(f"{d}: not staged; run python -m hicmultiome.train "
                               "--config <cfg> --stage-only first")
        meta = json.loads((d / "meta.json").read_text())
        man = Path(cache_dir) / "manifest.json"
        if meta["manifest_mtime"] != man.stat().st_mtime:
            raise RuntimeError(f"{d} is older than {man}; restage")
        pos = {tuple(x): i for i, x in enumerate(meta["index"])}
        rows = np.array([pos[(x[2], x[3], x[4])] for x in ds._index])
        out = {}
        for k in STAGE_KEYS:
            f = d / f"{k}.npy"
            if f.exists():
                out[k] = LazyRows(np.load(f, mmap_mode="r"), rows,
                                  dtype=torch.float32 if k == "bait" else dtype)
        return out, ds
    n, w = len(ds), ds.window_size
    tg = torch.empty((n, ds.n_out, w, w), dtype=dtype)
    cd = torch.empty((n, ds.n_sym, w), dtype=dtype)
    bt = torch.zeros((n, w), dtype=torch.float32)
    rf = torch.empty((n, w, w), dtype=dtype) if ds.reference else None
    pf = torch.empty((n, w, w), dtype=dtype) if ds.prior else None
    for path, pairs in ds.grouped_index().items():        # one decompress each
        z = np.load(path)
        rows = np.array([r for r, _ in pairs])
        loc = np.array([l for _, l in pairs])
        npdt = np.float16 if dtype == torch.float16 else np.float32
        tg[rows] = torch.from_numpy(z["target"][loc].astype(npdt))
        cd[rows] = torch.from_numpy(z["cond"][loc].astype(npdt))
        if "bait" in z:
            bt[rows] = torch.from_numpy(z["bait"][loc].astype(np.float32))
        if rf is not None:
            rf[rows] = torch.from_numpy(z["ref"][loc].astype(npdt))
        if pf is not None:
            pf[rows] = torch.from_numpy(z["prior"][loc].astype(npdt))
    out = {"target": tg, "cond": cd, "bait": bt}
    if rf is not None:
        out["ref"] = rf
    if pf is not None:
        out["prior"] = pf
    return out, ds


def _ref_batch(data, i, n, device, use_ref, key="ref"):
    """Encoded reference (or prior) rows [i:i+n] on device, None when unused."""
    if not use_ref or key not in data:
        return None
    return data[key][i : i + n].to(device).float()


def _coords(ds):
    """Per-window (chrom, start_bin, sample) aligned to the stacked tensors.
    `_stack` fills row r from ds._index[r], so this order matches."""
    chroms = [c for _, _, c, _, _ in ds._index]
    starts = np.array([s for _, _, _, s, _ in ds._index], dtype=np.int64)
    samples = [sm for _, _, _, _, sm in ds._index]
    return chroms, starts, samples


def overlaps_positions(coords, val_coords, window: int) -> np.ndarray:
    """Per window of `coords`: True when it shares a bin with any window of
    `val_coords` on the same chromosome, in ANY sample. Across samples because
    windows of different cells at one position share the sequence, the motifs
    and the cell-invariant structure (and any reference map), so a training
    window of one cell at a validation position of another is not held out."""
    ch, st, _ = coords
    vch, vst, _ = val_coords
    by = {}
    for c, s in zip(vch, vst):
        by.setdefault(c, []).append(int(s))
    by = {c: np.sort(np.asarray(v)) for c, v in by.items()}
    out = np.zeros(len(ch), dtype=bool)
    for i, (c, s) in enumerate(zip(ch, st)):
        v = by.get(c)
        if v is None:
            continue
        k = np.searchsorted(v, int(s))
        for j in (k - 1, k):
            if 0 <= j < len(v) and abs(int(v[j]) - int(s)) < window:
                out[i] = True
                break
    return out


def split_val_regions(coords, n_val: int, window: int, seed: int,
                      exclude_overlap: bool):
    """(vi, ti) row indices for train.val_regions.

    exclude_overlap False (runs before 2026-09-28): the first n_val windows of
    a seeded permutation are validation, all others train.
    exclude_overlap True: genomic positions are drawn in the order of the same
    permutation; every sample's window at a drawn position is validation, until
    n_val validation windows are reached, and every training window overlapping
    a validation position in any sample is dropped (overlaps_positions). With
    one sample this is the same validation set as above, minus the overlapping
    training windows."""
    ch, st, sm = coords
    n = len(ch)
    perm = np.random.default_rng(seed).permutation(n)
    if not exclude_overlap:
        return perm[:n_val], perm[n_val:]
    at = {}
    for i in range(n):
        at.setdefault((ch[i], int(st[i])), []).append(i)
    chosen, vlist = set(), []
    for i in perm:
        if len(vlist) >= n_val:
            break
        key = (ch[i], int(st[i]))
        if key in chosen:
            continue
        chosen.add(key)
        # the drawn window first, so one sample reproduces perm[:n_val]
        vlist += [i] + [j for j in at[key] if j != i]
    vi = np.asarray(vlist, dtype=perm.dtype)
    vset = set(vlist)
    rest = np.asarray([i for i in perm if i not in vset], dtype=perm.dtype)
    vcoords = ([ch[i] for i in vi], [st[i] for i in vi], [sm[i] for i in vi])
    rcoords = ([ch[i] for i in rest], [st[i] for i in rest], [sm[i] for i in rest])
    ti = rest[~overlaps_positions(rcoords, vcoords, window)]
    return vi, ti


def between_cell_weights(target, coords, alpha: float):
    """(position index per row, weight map per position) for the
    between_cell_var loss weighting: rows at one (chrom, start) are the
    samples' windows at that locus; v = variance across them per pixel."""
    ch, st, _ = coords
    key = {}
    pos = np.empty(len(ch), dtype=np.int64)
    for i, (c, s) in enumerate(zip(ch, st)):
        pos[i] = key.setdefault((c, int(s)), len(key))
    W = target.shape[-1]
    s1 = torch.zeros((len(key), W, W), dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    n = torch.zeros(len(key), dtype=torch.float64)
    p = torch.from_numpy(pos)
    for i in range(0, len(ch), 256):
        t = target[i:i + 256]
        t = (t.materialize() if hasattr(t, "materialize") else t)[:, 0].double()
        s1.index_add_(0, p[i:i + 256], t)
        s2.index_add_(0, p[i:i + 256], t * t)
        n.index_add_(0, p[i:i + 256], torch.ones(len(t), dtype=torch.float64))
    m = s1 / n[:, None, None]
    var = (s2 / n[:, None, None] - m * m).clamp_min(0)
    var[n < 2] = 0
    w = 1 + alpha * var / var[n >= 2].mean().clamp_min(1e-12)
    return p, w.float()


def sv_mask_rows(coords, mask_path: str, window: int, res: int):
    """(per-row index into the masks or -1, (n, W, W) float masks) for
    train.sv_mask: in a window holding a breakpoint at bin o, pixels (r, c)
    with min(r, c) <= o <= max(r, c) get weight 0."""
    bp = json.loads(Path(mask_path).read_text())["breakpoints"]
    by = {}
    for smp, lst in bp.items():
        for c, pos in lst:
            by.setdefault((smp, str(c)), []).append(int(pos) // int(res))
    ch, st, sm = coords
    idx = np.full(len(ch), -1, dtype=np.int64)
    masks = []
    ar = np.arange(window)
    for i, (c, s, smp) in enumerate(zip(ch, st, sm)):
        offs = [b - int(s) for b in by.get((smp, c), ()) if 0 <= b - int(s) < window]
        if not offs:
            continue
        m = np.ones((window, window), dtype=np.float32)
        for o in offs:
            m[np.ix_(ar <= o, ar >= o)] = 0.0
            m[np.ix_(ar >= o, ar <= o)] = 0.0
        idx[i] = len(masks)
        masks.append(m)
    out = (torch.from_numpy(np.stack(masks)) if masks
           else torch.ones((0, window, window), dtype=torch.float32))
    return torch.from_numpy(idx), out


def hard_split_rows(coords, split_path: str, window: int, res: int):
    """(hard-validation rows, rows outside every hard position) of a window
    list for a hard split JSON (scripts/hard_split.py): a row is hard-validation
    when its (chrom, start) is a validation window of its own sample; a row is
    removed from training when it shares a bin with any cell's hard window."""
    sp = json.loads(Path(split_path).read_text())
    if int(sp["res"]) != int(res) or int(sp["window"]) != int(window):
        raise ValueError(f"{split_path}: split for {sp['res']} bp / {sp['window']} bins, "
                         f"run is {res} bp / {window} bins")
    ch, st, sm = coords
    hv = {(s, c, int(b)) for s, v in sp["cells"].items() for c, b in v.get("val", [])}
    by = {}
    for c, a, b in sp["union"]:
        by.setdefault(c, []).append((int(a), int(b)))
    hv_rows, keep = [], []
    for i, (c, s, smp) in enumerate(zip(ch, st, sm)):
        s = int(s)
        if (smp, c, s) in hv:
            hv_rows.append(i)
        if not any(s < b and a < s + window for a, b in by.get(c, ())):
            keep.append(i)
    return np.asarray(hv_rows, dtype=np.int64), np.asarray(keep, dtype=np.int64)


def log_val_split(coords, vi, ti, exclude_overlap: bool):
    """Per sample: validation windows and training windows kept; warns when a
    sample has no validation window."""
    sm = coords[2]
    n = len(sm)
    samples = sorted(set(sm))
    nv = {s: 0 for s in samples}
    nt = {s: 0 for s in samples}
    for i in vi:
        nv[sm[i]] += 1
    for i in ti:
        nt[sm[i]] += 1
    log.info("validation split (%s): %d validation windows, %d training windows, "
             "%d dropped for overlapping a validation position",
             "honest, no overlap" if exclude_overlap else "OVERLAPPING training windows",
             len(vi), len(ti), n - len(vi) - len(ti))
    for s in samples:
        log.info("  sample %s: %d validation, %d training windows", s, nv[s], nt[s])
        if nv[s] == 0:
            log.warning("  sample %s has NO validation window", s)


def apply_motif_meth_drop(cond: torch.Tensor, dropped: torch.Tensor, mmc, rev_mask=None):
    """In place: the windows with `dropped` (bool, B) lose the methylation of
    the motif planes (see motif_meth_coupling). rev_mask marks the windows whose
    fwd/rev rows were swapped by the reverse augmentation: the row at a fwd
    position then holds the rev plane's normalisation, so the other plane's
    statistics apply."""
    bs, dm = cond.shape[0], dropped[:, None]
    for k, (dst, src, ca, cb) in enumerate(mmc[1]):
        ca_t = torch.full((bs, 1), ca, device=cond.device)
        cb_t = torch.full((bs, 1), cb, device=cond.device)
        if rev_mask is not None:
            oa, ob = mmc[1][1 - k][2], mmc[1][1 - k][3]
            ca_t = torch.where(rev_mask[:, None], ca_t.new_full((bs, 1), oa), ca_t)
            cb_t = torch.where(rev_mask[:, None], cb_t.new_full((bs, 1), ob), cb_t)
        cond[:, dst] = torch.where(dm, ca_t * cond[:, src] + cb_t, cond[:, dst])
    cond[:, mmc[2]] = torch.where(dm, torch.full_like(cond[:, mmc[2]], mmc[3]), cond[:, mmc[2]])
    return cond


METH_TRACK = "Meth"      # scripts/methylation.py TRACK


def dropout_chip_rows(cfg: Config, channels: dict) -> list[int]:
    """Cache rows of the ChIP tracks the model reads, in the order they lead
    its (possibly reduced) cond: track dropout acts on cond rows
    [0, len(result)) and uses these rows' absent values."""
    n = len(channels["tracks"])
    if cfg.train.select_tracks is None and cfg.train.select_motifs is None:
        return list(range(n))
    rows, *_ = _select_rows(cfg, channels)
    return [r for r in rows if r < n]


def _select_rows(cfg: Config, channels: dict):
    """Row indices into the cached cond for cfg.train.select_tracks: the chosen
    ChIP tracks, all sequence features, and each motif whose track is kept.
    Returns (rows, motif_offset, n_motifs, n_sym) for the reduced conditioning."""
    tracks, seqf, motifs = channels["tracks"], channels["seq_features"], channels["motif_ids"]
    sel = set(cfg.train.select_tracks) if cfg.train.select_tracks is not None else set(tracks)
    keep_t = [i for i, t in enumerate(tracks) if t in sel]
    if cfg.train.select_motifs is not None:
        used = set(cfg.train.select_motifs)
    else:
        track_motif = {t.name: t.motif for t in cfg.data.tracks}
        used = {track_motif.get(tracks[i]) for i in keep_t} | set(cfg.motifs.extra)
    # the methylation-weighted pair of a kept motif (motifs.motif_meth) travels with it
    used |= {meth_id(m) for m in list(used) if m is not None}
    keep_m = [m for m, mid in enumerate(motifs) if mid in used]
    nt, ns = len(tracks), len(seqf)
    rows = (keep_t
            + list(range(nt, nt + ns))
            + [nt + ns + 2 * m + k for m in keep_m for k in (0, 1)])
    return rows, len(keep_t) + ns, len(keep_m), len(rows)


def motif_meth_coupling(cfg: Config, channels: dict, chip_rows: list[int], track_norm):
    """Track dropout of the Meth track also removes the methylation of the
    CTCF motif planes (motifs.motif_meth): in a window that drops Meth the
    weighted pair is replaced by the unmodified pair and the covered row by
    its absent value, the state of a cell without methylation. Returns None
    when the model has no such planes or no Meth track among the dropout rows,
    else (draw position of Meth, [(dst, src, a, b)] per plane, covered row,
    covered absent value), all as indices into the model's (reduced) cond."""
    if not cfg.motif_meth_active or METH_TRACK not in channels["tracks"]:
        return None
    t = channels["tracks"].index(METH_TRACK)
    if t not in chip_rows:
        return None
    rows_full = mm.coupling_rows(channels["sym_rows"], cfg.motifs.motif_meth_id)
    full_to_cond = {r: i for i, r in enumerate(_select_rows(cfg, channels)[0])} if (
        cfg.train.select_tracks is not None or cfg.train.select_motifs is not None) else None

    def cond_row(r):
        if full_to_cond is None:
            return r
        if r not in full_to_cond:
            raise ValueError(f"motif_meth: cond row {r} is not kept by select_tracks/select_motifs")
        return full_to_cond[r]
    a, b, cov_absent = mm.absent_transform(track_norm, rows_full)
    planes = [(cond_row(rows_full["meth"][k]), cond_row(rows_full["unmod"][k]),
               float(a[k]), float(b[k])) for k in range(2)]
    return chip_rows.index(t), planes, cond_row(rows_full["covered"]), cov_absent



def _fasta_map(cfg: Config) -> dict:
    """sample name -> FASTA path (per-sample override, else sequence.fasta).
    Includes a 'default' entry for caches whose windows carry no sample name."""
    m = {s.name: (s.fasta or cfg.sequence.fasta) for s in cfg.data.samples}
    m.setdefault("default", cfg.sequence.fasta)
    return m


def batch_onehot(coords, idx, rmask, W, binsize, fasta_map, device):
    """One-hot DNA (len(idx), 4, W*binsize) for the given rank-local rows.
    `rmask` (bool per row) requests the reverse complement, matching the
    reverse-strand augmentation applied to the rest of the conditioning.

    The CPU only slices raw base codes (cheap); the small uint8 buffer is sent to
    the GPU and the one-hot scatter runs there, so the GPU is not stalled waiting
    on a Python one-hot loop."""
    chroms, starts, samples = coords
    L = W * binsize
    codes = np.empty((len(idx), L), dtype=np.uint8)
    for k, i in enumerate(idx):
        codes[k] = window_codes(fasta_map[samples[i]], chroms[i], int(starts[i]),
                                W, binsize, revcomp=bool(rmask[k]))
    c = torch.from_numpy(codes).to(device, non_blocking=True).long()   # (B, L)
    oh = torch.zeros(len(idx), 4, L, device=device)
    oh.scatter_(1, c.clamp(max=3).unsqueeze(1), 1.0)   # 1 at [b, base, l]
    oh.mul_((c < 4).unsqueeze(1).to(oh.dtype))         # N (code 4) -> all zero
    return oh


def masked_pair_loss(pred, tgt, bait):
    """L1+MSE over pixels where either bin has bait coverage."""
    mask = torch.maximum(bait[:, :, None], bait[:, None, :])   # (B, W, W)
    denom = mask.sum().clamp_min(1.0)
    d = (pred - tgt) * mask
    return (d.abs().sum() + d.pow(2).sum()) / denom


def capture_planes(tgt, bait, present):
    """(B,2,W,W) capture-input planes: encoded capture map + bait-pair mask.

    present: (B,) float 0/1 -- windows with 0 get the 'absent' encoding
    (plane=-1, mask=0), which is also the inference default without capture."""
    p = present[:, None, None]
    plane = tgt[:, 1] * p + (-1.0) * (1 - p)
    mask = torch.maximum(bait[:, :, None], bait[:, None, :]) * p
    return torch.stack([plane, mask], dim=1)


def _aml_object_rows(cfg: Config, ds, ti, os_idx, data, is_main: bool):
    """Loop-centric recipe (hicmultiome/aml_objects.py): the objects of the table in every
    training window row, the matched-pair oversampling rows appended to `data`. Returns
    (data, per-row object lists, per-row roles) aligned to the returned data."""
    from . import aml_objects as ao

    tc = cfg.train
    rows_ti = np.asarray(ti, dtype=np.int64)
    ch, st, sm = _coords(ds)
    coords = ([ch[i] for i in rows_ti], st[rows_ti], [sm[i] for i in rows_ti])
    train_samples = {s.name for s in cfg.data.samples}
    objs = ao.load_objects(cfg.data.aml_objects_table, train_samples, tuple(tc.aml_object_controls))
    objs = ao.to_bins(objs, int(cfg.data.resolution))
    margin = int(tc.aml_object_donut[1]) + 1
    row0 = ao.rows_of_objects(objs, coords, ds.window_size, margin)     # rows of ti
    n_ti = len(row0)
    base = np.arange(n_ti) if os_idx is None else np.asarray(os_idx, dtype=np.int64)
    row_objs = [row0[i] for i in base]                  # aligned to data (first n_ti rows = ti)
    extra = np.zeros(0, dtype=np.int64)
    if tc.aml_object_oversample > 1:
        rng = np.random.default_rng(tc.seed + 7)
        pos_r, ctl_r = ao.select_pairs(row0, objs, rng)
        if len(pos_r) == 0:
            raise ValueError("train.aml_object_oversample: no training window holds a matched "
                             "pair of an AML-specific object and a control")
        extra = np.concatenate([np.concatenate([pos_r, ctl_r])] * (int(tc.aml_object_oversample) - 1))
        idx = np.concatenate([np.arange(len(base)), extra])
        data = {k: v[idx] for k, v in data.items()}
        row_objs += [row0[i] for i in extra]
    roles = ao.row_roles(row_objs)
    if is_main:
        log.info("aml objects %s: %d objects of %d samples (%s); %d of %d training windows "
                 "hold objects; oversample x%d adds %d windows (%d matched pairs); rows "
                 "AML-specific %d, control-only %d",
                 cfg.data.aml_objects_table, len(objs), objs.cell.nunique(),
                 ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in objs.groupby(["kind", "set"]).size().items()),
                 sum(1 for o in row0 if o), n_ti, tc.aml_object_oversample, len(extra),
                 len(extra) // 2 // max(1, tc.aml_object_oversample - 1),
                 int((roles == 1).sum()), int((roles == 2).sum()))
    return data, row_objs, roles


@torch.no_grad()
def val_distance_pearson(model, data, device, batch, offset, n_motifs, maxd=64,
                         capin=False, cond_mode="full", seq_fn=None, use_ref=False):
    """Validation in the tracks-only mode (capture plane absent) -- the honest
    default, since prediction must not require capture data. `seq_fn(i, n)`, when
    given, returns the one-hot DNA for val rows [i:i+n] for the sequence encoder."""
    if data is None or data["target"].shape[0] == 0:
        return float("nan")
    model.eval()
    preds = []
    for i in range(0, data["target"].shape[0], batch):
        tgt_b = data["target"][i : i + batch].to(device).float()
        sym, pairs = split_cond(data["cond"][i : i + batch].to(device).float(),
                                offset, n_motifs)
        planes = None
        if capin:
            absent = torch.zeros(tgt_b.shape[0], device=device)
            planes = capture_planes(tgt_b, data["bait"][i : i + batch].to(device), absent)
        seq_oh = seq_fn(i, tgt_b.shape[0]) if seq_fn is not None else None
        preds.append(model(sym, pairs, planes, seq_oh,
                           _ref_batch(data, i, tgt_b.shape[0], device, use_ref),
                           prior=_ref_batch(data, i, tgt_b.shape[0], device, True,
                                            "prior")).cpu())
    pred = torch.cat(preds)[:, 0].numpy()
    true = data["target"][:, 0].float().cpu().numpy()
    w = true.shape[-1]
    rs = []
    for d in range(1, min(maxd, w)):
        x = np.concatenate([np.diagonal(true[k], d) for k in range(true.shape[0])])
        y = np.concatenate([np.diagonal(pred[k], d) for k in range(pred.shape[0])])
        if x.std() > 1e-8 and y.std() > 1e-8:
            rs.append(np.corrcoef(x, y)[0, 1])
    model.train()
    return float(np.nanmean(rs)) if rs else float("nan")


@torch.no_grad()
def val_akita_median(model, data, device, batch, offset, n_motifs,
                     cond_mode="full", seq_fn=None, capin=False, use_ref=False):
    """Akita-style validation: the median over the held-out regions of each
    region's Pearson between prediction and target over its finite upper-triangle
    pixels (log obs/exp). Pearson is affine-invariant, so comparing in the
    encoded space the model outputs equals comparing decoded O/E.
    With capin (capture-input model) the capture planes are fed as absent, the
    same tracks-only mode as val_distance_pearson."""
    rs = []
    for pk_full, tk_full in val_maps(model, data, device, batch, offset, n_motifs,
                                     cond_mode, seq_fn, capin, use_ref):
        iu = np.triu_indices(pk_full.shape[-1], 1)
        pk, tk = pk_full[iu], tk_full[iu]
        m = np.isfinite(pk) & np.isfinite(tk)
        if m.sum() > 100 and pk[m].std() > 1e-8 and tk[m].std() > 1e-8:
            rs.append(np.corrcoef(pk[m], tk[m])[0, 1])
    return float(np.median(rs)) if rs else float("nan")


@torch.no_grad()
def val_structure(model, data, device, batch, offset, n_motifs, res: int,
                  cond_mode="full", seq_fn=None, capin=False, use_ref=False,
                  hic_measures=None) -> dict:
    """Median over the validation windows of every hicmultiome.structure
    measure, plus the per-window maps' selection values; with hic_measures
    (a callable on the list of (pred, target) maps and res) also its
    combined score as 'hic_measures'."""
    from .structure import selection_values, structure_metrics
    rows, maps = [], []
    for pk, tk in val_maps(model, data, device, batch, offset, n_motifs,
                           cond_mode, seq_fn, capin, use_ref):
        m = structure_metrics(pk, tk, res)
        m.update({f"sel_{k}": v for k, v in selection_values(m).items()})
        rows.append(m)
        if hic_measures is not None:
            maps.append((pk, tk))
    out = {}
    for k in (rows[0] if rows else {}):
        v = np.array([r[k] for r in rows], float)
        out[k] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
    if hic_measures is not None and maps:
        out["hic_measures"] = float(hic_measures(maps, res))
    return out


@torch.no_grad()
def val_maps(model, data, device, batch, offset, n_motifs, cond_mode="full",
             seq_fn=None, capin=False, use_ref=False):
    """(prediction, target) per validation window, encoded space, numpy."""
    if data is None or data["target"].shape[0] == 0:
        return
    model.eval()
    N = data["target"].shape[0]
    for i in range(0, N, batch):
        tgt = data["target"][i : i + batch].to(device).float()
        sym, pairs = split_cond(data["cond"][i : i + batch].to(device).float(),
                                offset, n_motifs)
        seq_oh = seq_fn(i, tgt.shape[0]) if seq_fn is not None else None
        planes = None
        if capin:
            nb_ = tgt.shape[0]
            planes = capture_planes(tgt, torch.zeros(nb_, tgt.shape[-1], device=device),
                                    torch.zeros(nb_, device=device))
        p = model(sym, pairs, planes, seq_oh,
                  _ref_batch(data, i, tgt.shape[0], device, use_ref),
                  prior=_ref_batch(data, i, tgt.shape[0], device, True,
                                   "prior"))[:, 0].float().cpu().numpy()
        t = tgt[:, 0].float().cpu().numpy()
        for k in range(p.shape[0]):
            yield p[k], t[k]
    model.train()


def _plot_curve(rows, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(steps, [r["loss"] for r in rows], color="#888", lw=1, label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    vs = [(r["step"], r["val_pearson"]) for r in rows if r["val_pearson"] == r["val_pearson"]]
    if vs:
        ax2.plot(*zip(*vs), color="#2c7fb8", lw=2, marker="o", ms=3,
                 label="val dist-Pearson")
        ax2.set_ylabel("val per-distance Pearson")
    fig.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _build_cohesin_aug(cfg: Config, ds, ti, data, chip_rows, offset, sl, device, cache_dir,
                       is_main):
    """CohesinAugmenter (train.cohesin_aug) and the contact scale of every
    resident training window of this rank (aligned with the window stripe)."""
    from .cohesin import CohesinAugmenter, load_operator
    from .precompute import COHESIN_TRACKS, FRACTION_ROW
    from .tracks import TrackNormalizer

    if (cfg.train.window_subsample > 1 or cfg.train.oversample_factor > 1
            or cfg.train.hard_split):
        raise ValueError("train.cohesin_aug is not supported with window_subsample, "
                         "oversample_* or hard_split")
    seqf = ds.channels["seq_features"]
    if FRACTION_ROW not in seqf:
        raise RuntimeError("train.cohesin_aug but the cache has no cohesin_fraction row: "
                           "set data.cohesin_fraction and re-run hicmultiome.precompute")
    if ds.channels.get("has_capture") or ds.channels.get("capture_input"):
        raise ValueError("train.cohesin_aug is not supported with capture data")
    tracks_all = ds.channels["tracks"]
    tn = TrackNormalizer.load(Path(cache_dir) / "norm_tracks.json")
    absent_all = tn.absent_values()
    keep = [t for t in COHESIN_TRACKS if t in tracks_all and tracks_all.index(t) in chip_rows]
    if not keep:
        raise ValueError("train.cohesin_aug needs RAD21 or SMC3 among the model tracks")
    rows = [chip_rows.index(tracks_all.index(t)) for t in keep]
    op, f_ref = load_operator(cfg.train.cohesin_aug_operator, device)
    aug = CohesinAugmenter(
        op, f_ref, cfg.train.cohesin_aug, cfg.train.cohesin_aug_fractions,
        cfg.train.cohesin_aug_absent, rows,
        [tn.mean[tracks_all.index(t)] for t in keep], [tn.std[tracks_all.index(t)] for t in keep],
        [float(absent_all[tracks_all.index(t)]) for t in keep],
        offset - len(seqf) + seqf.index(FRACTION_ROW), device)
    names_all = _coords(ds)[2]
    names = [names_all[i] for i in ti] if ti is not None else list(names_all)
    scale_of = {n: float(json.loads((Path(cache_dir) / f"norm_matrix_{n}.json").read_text())["scale"])
                for n in set(names)}
    sc_all = torch.tensor([scale_of[n] for n in names], dtype=torch.float32)
    if len(sc_all) != data["target"].shape[0]:
        raise RuntimeError("cohesin_aug: window names do not align with the training windows")
    if is_main:
        log.info("cohesin_aug: p=%.2f fractions %s absent-track share %.2f, operator %s "
                 "(fitted at cohesin fraction %.2f), tracks %s", cfg.train.cohesin_aug,
                 cfg.train.cohesin_aug_fractions, cfg.train.cohesin_aug_absent,
                 cfg.train.cohesin_aug_operator, f_ref, keep)
    return aug, sc_all[sl].to(device) if sl is not None else sc_all.to(device)


def train(cfg: Config):
    if cfg.train.stream:
        return train_stream(cfg)
    rank, world, local_rank = ddp_env()
    is_ddp = world > 1
    if is_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    set_seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(cfg.train.output_dir)
    cache_dir = Path(cfg.train.cache_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    chroms = cfg.train.chroms                    # optional fast-proxy chrom restriction
    res_dt = {"float32": torch.float32, "float16": torch.float16}[cfg.train.resident_dtype]
    vi = ti = None                               # val_regions row split of ds
    hdata = None                                 # hard-validation windows (train.hard_split)
    if cfg.train.hard_split and not cfg.train.val_regions:
        raise ValueError("train.hard_split needs train.val_regions")
    if cfg.train.val_regions:
        # hold out a fixed set of whole windows as an Akita-style validation set
        data, ds = _stack(str(cache_dir), "train", 0.0, False, cfg.train.seed, chroms=chroms,
                          dtype=res_dt, stage_dir=cfg.train.stage_dir)
        vds = ds
        if data is not None:
            co = _coords(ds)
            base = np.arange(len(co[0]))
            if cfg.train.hard_split:
                hv_rows, base = hard_split_rows(co, cfg.train.hard_split, ds.window_size,
                                                cfg.data.resolution)
                hdata = {k: v[hv_rows] for k, v in data.items()}
                if rank == 0:
                    per = {}
                    for i in hv_rows:
                        per[co[2][i]] = per.get(co[2][i], 0) + 1
                    log.info("hard split %s: %d hard-validation windows %s; %d of %d "
                             "training-split windows removed (any cell's hard position)",
                             cfg.train.hard_split, len(hv_rows), per,
                             len(co[0]) - len(base), len(co[0]))
                co = ([co[0][i] for i in base], co[1][base], [co[2][i] for i in base])
            vi, ti = split_val_regions(co, cfg.train.val_regions, ds.window_size,
                                       cfg.train.seed, cfg.train.val_exclude_overlap)
            if rank == 0:
                log_val_split(co, vi, ti, cfg.train.val_exclude_overlap)
            vi, ti = base[vi], base[ti]
            vdata = {k: v[vi] for k, v in data.items()}
            data = {k: v[ti] for k, v in data.items()}
        else:
            vdata = None
    else:
        data, ds = _stack(str(cache_dir), "train", cfg.train.val_fraction, False, cfg.train.seed, chroms=chroms,
                          dtype=res_dt, stage_dir=cfg.train.stage_dir)
        vdata, vds = _stack(str(cache_dir), "train", cfg.train.val_fraction, True, cfg.train.seed, chroms=chroms,
                            dtype=res_dt, stage_dir=cfg.train.stage_dir)
        if cfg.train.val_exclude_overlap and data is not None and vdata is not None:
            keep = ~overlaps_positions(_coords(ds), _coords(vds), ds.window_size)
            if rank == 0:
                log.info("val_exclude_overlap: %d of %d training windows kept (%d overlap "
                         "a validation position in any sample)", int(keep.sum()),
                         len(keep), int((~keep).sum()))
            idx = np.nonzero(keep)[0]
            data = {k: v[idx] for k, v in data.items()}
            ds._index = [ds._index[i] for i in idx]
    if data is None:
        raise RuntimeError("no training windows; run hicmultiome.precompute first")
    use_ref = cfg.model.reference != "none"
    if use_ref:
        ds.check_reference_leakage()
    if cfg.train.reference_dropout > 0 and not use_ref:
        raise ValueError("train.reference_dropout needs a model.reference model")
    if cfg.model.prior and "prior" not in data:
        raise RuntimeError(f"model.prior but the cache {cache_dir} holds no prior "
                           "windows: set data.prior_cool / samples[].prior_cool and "
                           "re-run hicmultiome.precompute")
    if not cfg.model.prior:
        data.pop("prior", None)
    if use_ref and "ref" not in data:
        raise RuntimeError(f"model.reference={cfg.model.reference!r} but the cache "
                           f"{cache_dir} holds no reference windows: set "
                           "data.reference_cool and re-run hicmultiome.precompute")

    # greedy feature selection: keep only the chosen ChIP tracks' cond rows
    n_sym_eff = ds.n_sym
    if cfg.train.select_tracks is not None or cfg.train.select_motifs is not None:
        rows, offset, n_motifs, n_sym_eff = _select_rows(cfg, ds.channels)
        data["cond"] = data["cond"][:, rows, :]
        if vdata is not None:
            vdata["cond"] = vdata["cond"][:, rows, :]
        if hdata is not None:
            hdata["cond"] = hdata["cond"][:, rows, :]
        if is_main:
            log.info("select_tracks=%s -> %d tracks, %d motifs, %d sym rows",
                     cfg.train.select_tracks,
                     len(rows) - n_motifs * 2 - len(ds.channels["seq_features"]), n_motifs, n_sym_eff)
    else:
        offset, n_motifs = ds.motif_row_offset(), ds.n_motifs

    use_seq = cfg.sequence.uses_encoder
    binsize = int(cfg.data.resolution)
    fasta_map = _fasta_map(cfg)
    W = ds.window_size
    # full-set window coordinates (aligned to the stacked rows), for the encoder
    full_coords = _coords(ds) if use_seq else None
    val_coords_all = None
    if full_coords is not None and ti is not None:
        # val_regions split the stacked rows by perm (vi/ti); split the
        # coordinates the same way so the encoder reads each window's own DNA
        ch, st, sm = full_coords
        val_coords_all = ([ch[i] for i in vi], st[vi], [sm[i] for i in vi])
        full_coords = ([ch[i] for i in ti], st[ti], [sm[i] for i in ti])

    # data-efficiency knob: keep every k-th window (effective stride =
    # window.stride * k). A cheap way to sweep training-set size without
    # re-precomputing the cache -- every k-th of the 50%-overlap windows is
    # exactly the coarser-stride tiling.
    k = max(1, int(cfg.train.window_subsample))
    if k > 1 and (cfg.train.loss_weighting != "none" or cfg.train.sv_mask):
        raise ValueError("train.loss_weighting / train.sv_mask with window_subsample "
                         "is not supported")
    if k > 1:
        keep = list(range(0, data["target"].shape[0], k))
        data = {kk: v[keep] for kk, v in data.items()}
        if full_coords is not None:
            ch, st, sm = full_coords
            full_coords = ([ch[i] for i in keep], st[keep], [sm[i] for i in keep])
        if is_main:
            log.info("window_subsample=%d: %d training windows (effective "
                     "stride %d bins)", k, len(keep), cfg.window.stride * k)
    os_idx = None                                # rows of ti after train.oversample_*
    if cfg.train.oversample_factor > 1 and cfg.train.oversample_samples:
        if (ti is None or cfg.train.loss_weighting != "none" or cfg.train.sv_mask or use_seq
                or cfg.train.window_subsample > 1):
            raise ValueError("train.oversample_* needs val_regions and no loss_weighting, "
                             "sv_mask or sequence encoder")
        names_all = _coords(ds)[2]
        names = [names_all[i] for i in ti]
        want = set(cfg.train.oversample_samples)
        over = np.array([i for i, n in enumerate(names) if n in want], dtype=np.int64)
        if len(over) == 0:
            raise ValueError(f"train.oversample_samples {cfg.train.oversample_samples}: "
                             f"no training window of these samples (training samples: {sorted(set(names))})")
        idx = np.concatenate([np.arange(len(names))] + [over] * (cfg.train.oversample_factor - 1))
        data = {kk: v[idx] for kk, v in data.items()}
        os_idx = idx
        if is_main:
            log.info("oversample x%d: %d windows of %d samples; training windows %d -> %d",
                     cfg.train.oversample_factor, len(over),
                     len({names[i] for i in over}), len(names), len(idx))
    obj_rows = obj_roles = obj_loss = None
    if cfg.data.aml_objects_table and (
            cfg.train.aml_object_oversample > 1 or cfg.train.aml_object_weight > 0
            or cfg.train.aml_object_rank > 0 or cfg.train.aml_object_pair_batches):
        if (ti is None or cfg.train.loss_weighting != "none" or cfg.train.sv_mask or use_seq
                or cfg.train.window_subsample > 1 or cfg.train.cohesin_aug > 0):
            raise ValueError("train.aml_object_* needs val_regions and no loss_weighting, "
                             "sv_mask, sequence encoder, window_subsample or cohesin_aug")
        data, obj_rows, obj_roles = _aml_object_rows(cfg, ds, ti, os_idx, data, is_main)
    N_total = data["target"].shape[0]

    # Each rank keeps its stripe of the windows resident on its GPU. Stripes are
    # trimmed to equal length so every rank runs the same number of steps --
    # the per-step stop_flag all_reduce would deadlock otherwise.
    n_even = (N_total // world) * world if world > 1 else N_total
    sl = slice(rank, n_even, world)
    tg = data["target"][sl].to(device)
    cd = data["cond"][sl].to(device)
    bt = data["bait"][sl].to(device)
    rf = data["ref"][sl].to(device) if use_ref else None
    pw_pos = pw_map = None
    if cfg.train.loss_weighting == "between_cell_var":
        tco = ([ds._index[i][2] for i in ti], np.array([ds._index[i][3] for i in ti]),
               [ds._index[i][4] for i in ti]) if ti is not None else _coords(ds)
        pos_all, pw_map = between_cell_weights(data["target"], tco, cfg.train.loss_weight_alpha)
        pw_pos = pos_all[:n_even][rank::world].to(device) if world > 1 else pos_all.to(device)
        pw_map = pw_map.to(device)
        if is_main:
            log.info("loss weighting between_cell_var: %d loci, weight mean %.3f max %.2f",
                     pw_map.shape[0], float(pw_map.mean()), float(pw_map.max()))
    elif cfg.train.loss_weighting != "none":
        raise ValueError(f"unknown train.loss_weighting {cfg.train.loss_weighting!r}")
    sv_idx = sv_maps = None
    if cfg.train.sv_mask:
        sco = ([ds._index[i][2] for i in ti], np.array([ds._index[i][3] for i in ti]),
               [ds._index[i][4] for i in ti]) if ti is not None else _coords(ds)
        sv_all, sv_maps = sv_mask_rows(sco, cfg.train.sv_mask, ds.window_size,
                                       cfg.data.resolution)
        if is_main:
            per = {}
            for i in np.nonzero(sv_all.numpy() >= 0)[0]:
                per[sco[2][i]] = per.get(sco[2][i], 0) + 1
            frac = float(1 - sv_maps.mean()) if len(sv_maps) else 0.0
            log.info("sv_mask %s: %d of %d training windows hold a breakpoint %s; "
                     "%.1f%% of their pixels masked", cfg.train.sv_mask, len(sv_maps),
                     len(sv_all), per, 100 * frac)
        sv_idx = (sv_all[:n_even][rank::world] if world > 1 else sv_all).to(device)
        sv_maps = sv_maps.to(device)
    pf = data["prior"][sl].to(device) if cfg.model.prior else None
    vdata_d = hdata_d = None
    if is_main and vdata is not None:
        vdata_d = {k2: _dense(v, device) for k2, v in vdata.items()}
    if is_main and hdata is not None:
        hdata_d = {k2: _dense(v, device) for k2, v in hdata.items()}
    N = tg.shape[0]
    if obj_rows is not None:
        from .aml_objects import ObjectLoss
        obj_rows = [obj_rows[i] for i in range(rank, n_even, world)]
        obj_roles = obj_roles[rank:n_even:world]
        tc = cfg.train
        if tc.aml_object_weight > 0 or tc.aml_object_rank > 0:
            obj_loss = ObjectLoss(tc.aml_object_weight, tc.aml_object_rank, tc.aml_object_rank_temp,
                                  tc.aml_object_rank_consistent, tc.aml_object_rank_space,
                                  tc.aml_object_donut)

    tr_coords = val_coords = None
    if use_seq:
        ch, st, sm = full_coords                  # rank stripe, aligned to tg/cd
        rows = list(range(rank, n_even, world))
        tr_coords = ([ch[i] for i in rows], st[rows], [sm[i] for i in rows])
        if is_main and val_coords_all is not None:
            val_coords = val_coords_all
        elif is_main and vds is not None:
            val_coords = _coords(vds)

    chip_rows = dropout_chip_rows(cfg, ds.channels)
    n_chip = len(chip_rows)
    absent_t = None
    if cfg.train.track_dropout > 0 and n_chip > 0:
        from .tracks import TrackNormalizer

        tn = TrackNormalizer.load(cache_dir / "norm_tracks.json")
        absent_t = torch.from_numpy(tn.absent_values()[chip_rows]).to(device)

    # cohesin-unknown indicator (data.cohesin_unknown): the SMC3 row follows
    # the RAD21 dropout draw and the indicator row (normalised "1") is set in
    # every window that loses the cohesin group
    mmc = None
    if absent_t is not None:
        mmc = motif_meth_coupling(cfg, ds.channels, chip_rows, tn)
        if mmc is not None and is_main:
            log.info("motif_meth: Meth dropout removes the methylation of the motif planes "
                     "(cond rows %s from %s, covered row %d)", [p[0] for p in mmc[1]],
                     [p[1] for p in mmc[1]], mmc[2])
    coh = None
    if cfg.data.cohesin_unknown and absent_t is not None:
        from .precompute import COHESIN_ROW, COHESIN_TRACKS

        tracks_all = ds.channels["tracks"]
        if COHESIN_ROW not in ds.channels["seq_features"]:
            raise RuntimeError("data.cohesin_unknown but the cache has no cohesin_unknown "
                               "row: re-run hicmultiome.precompute")
        keep = [t for t in COHESIN_TRACKS if t in tracks_all]
        pos = [chip_rows.index(tracks_all.index(t)) for t in keep
               if tracks_all.index(t) in chip_rows]
        if pos:
            # row of the indicator in the (possibly reduced) cond: the last
            # sequence-feature row, directly before the motif rows
            seqf = ds.channels["seq_features"]
            r_ind = offset - len(seqf) + seqf.index(COHESIN_ROW)
            sym_i = ds.channels["sym_rows"].index("cohesin:unknown")
            one = tn.apply(np.ones((len(tn.mean), 1), dtype=np.float32))[sym_i, 0]
            coh = (pos[0], pos[1:], r_ind, float(one))
            if is_main:
                log.info("cohesin_unknown: group %s dropped jointly, indicator cond row %d "
                         "set to %.3f", keep, r_ind, one)

    aug = win_scale = None
    n_aug_epoch = 0
    if cfg.train.cohesin_aug > 0:
        aug, win_scale = _build_cohesin_aug(cfg, ds, ti, data, chip_rows, offset, sl, device,
                                            cache_dir, is_main)

    capin = bool(cfg.capture.use_as_input and ds.n_out == 2)
    cond_mode = cfg.model.cond_mode
    emb = cfg.sequence.embed_dim if use_seq else 0
    in_ch = n_cond_channels(n_sym_eff + emb, n_motifs,
                            (2 if capin else 0) + cfg.model.reference_planes, cond_mode)
    encoder = (SeqEncoder(embed_dim=emb, width=cfg.sequence.enc_width)
               if use_seq else None)
    model = make_model(cfg, in_ch, ds.n_out, encoder).to(device)
    if use_ref and is_main:
        log.info("reference map: %s (%s)", cfg.model.reference, ds.reference)
    if use_seq and is_main:
        log.info("sequence encoder: dilated, embed_dim=%d width=%d (bp->%d bins)",
                 emb, cfg.sequence.enc_width, W)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    core = model.module if is_ddp else model
    if cfg.train.compile:
        # `core` deliberately keeps pointing at the uncompiled module: the
        # compiled wrapper prefixes every state-dict key with "_orig_mod.",
        # which would save checkpoints that no plain run could load back.
        # Parameters are shared, so training the wrapper trains `core`.
        model = torch.compile(model)
        if is_main:
            log.info("torch.compile enabled (first steps include compilation)")
    if is_main:
        log.info(
            "hic-multiome-unet: %d train windows (%d on this rank, world=%d), "
            "%d val, %d sym rows, %d motifs, %d cond channels, %d out channels, "
            "%.2fM params",
            N_total, N, world, 0 if vdata is None else vdata["target"].shape[0],
            n_sym_eff, n_motifs, in_ch, ds.n_out,
            sum(p.numel() for p in core.parameters()) / 1e6,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    batch = cfg.train.batch_size
    cap_w = cfg.capture.loss_weight if ds.n_out == 2 else 0.0
    hic_loss = build_loss(cfg.train.loss, max_dist=cfg.train.loss_max_dist,
                          stride=cfg.train.loss_stride)
    if is_main:
        log.info("objective: %s", cfg.train.loss)
    step = seen = epoch = 0
    best = float("-inf")

    if cfg.train.resume:
        src = Path(cfg.train.resume_from) if cfg.train.resume_from else None
        if src is None:
            for ptr in ("best", "latest"):
                q = out_dir / ptr
                if q.exists():
                    src = out_dir / q.read_text().strip()
                    break
        if src is None or not (src / "model.pt").exists():
            log.warning("resume requested but no checkpoint found; "
                        "starting from scratch")
        else:
            core.load_state_dict(torch.load(src / "model.pt", map_location=device))
            if (src / "optim.pt").exists():
                opt.load_state_dict(torch.load(src / "optim.pt",
                                               map_location=device))
            if (src / "progress.json").exists():
                pr = json.loads((src / "progress.json").read_text())
                step, epoch, best = pr["step"], pr["epoch"], pr["best"]
            log.info("resumed from %s at step %d (epoch %d, best val %.4f)",
                     src, step, epoch, best)
    t0 = last_log = time.time()
    last_eval = 0
    stop = False
    curve: list[dict] = []

    def save(name, ptr):
        if not is_main or not cfg.train.save_checkpoints:
            return
        ck = out_dir / name
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(core.state_dict(), ck / "model.pt")
        torch.save(opt.state_dict(), ck / "optim.pt")
        (ck / "progress.json").write_text(json.dumps(
            {"step": step, "epoch": epoch, "best": best}))
        for s in list(cache_dir.glob("norm_*.json")) + [cache_dir / "channels.json"]:
            if s.exists():
                (ck / s.name).write_text(s.read_text())
        if cfg.model.embed_reference:
            from .reference import embed_reference
            embed_reference(cfg.model.embed_reference, ck)
        cfg.save(ck / "config.yaml")
        (out_dir / ptr).write_text(name)

    struct_rows: list[dict] = []
    sel_hist: list = []
    hic_measures_fn = None
    if cfg.train.select_metric not in ("dist_pearson", "combined", "hic_measures"):
        raise ValueError(f"unknown train.select_metric {cfg.train.select_metric!r}")
    if cfg.train.select_metric != "dist_pearson":
        if not cfg.train.val_regions:
            raise ValueError("train.select_metric other than dist_pearson needs "
                             "train.val_regions")
        if hdata is not None:
            raise ValueError("train.select_metric with train.hard_split is not supported")
    if cfg.train.select_metric == "hic_measures":
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        hic_measures_fn = importlib.import_module("hic_measures").combined_score
    from .structure import rank_combined

    def validate(cur_loss):
        nonlocal best, last_eval
        last_eval = step
        save("checkpoint-latest", "latest")
        if not is_main:
            return
        r = float("nan")
        if vdata_d is not None:
            seq_fn = None
            if use_seq and val_coords is not None:
                seq_fn = lambda i, n: batch_onehot(
                    val_coords, list(range(i, i + n)), [False] * n, W, binsize,
                    fasta_map, device)
            vb = max(1, min(batch, vdata_d["target"].shape[0]))
            sel = cfg.train.select_metric
            if cfg.train.val_regions and (sel != "dist_pearson" or cfg.train.log_structure):
                sm = val_structure(model, vdata_d, device, vb, offset, n_motifs,
                                   int(cfg.data.resolution), cond_mode=cond_mode,
                                   seq_fn=seq_fn, capin=capin, use_ref=use_ref,
                                   hic_measures=hic_measures_fn)
                log.info("  [val-structure] epoch %d %s", epoch,
                         " ".join(f"{k}={v:.4f}" for k, v in sm.items()))
                struct_rows.append({"step": step, "epoch": epoch, **sm})
                if sel != "dist_pearson":
                    if sel == "combined":
                        sel_hist.append({k[4:]: v for k, v in sm.items()
                                         if k.startswith("sel_")})
                        sc = rank_combined(sel_hist)
                        cur = sc[-1]
                        imp_c = len(sc) == 1 or cur > max(sc[:-1])
                    else:
                        cur = sm["hic_measures"]
                        imp_c = not sel_hist or cur > max(sel_hist)
                        sel_hist.append(cur)
                    log.info("  [val] epoch %d step %d  %s=%.4f%s (pearson %.4f)",
                             epoch, step, sel, cur, "  <- best" if imp_c else "",
                             sm["pearson"])
                    if imp_c:
                        # `best` counts the selected checkpoints, so early
                        # stopping still sees an increase at every new best
                        best = best + 1 if best > float("-inf") else 1.0
                        save("checkpoint-best", "best")
                    curve.append({"step": step, "loss": cur_loss, "val_pearson": sm["pearson"]})
                    return
            if cfg.train.val_regions:
                r = val_akita_median(model, vdata_d, device, vb, offset, n_motifs,
                                     cond_mode=cond_mode, seq_fn=seq_fn, capin=capin,
                                     use_ref=use_ref)
                if hdata_d is not None:
                    # selection on the hard-validation windows; honest logged
                    r_h = val_akita_median(model, hdata_d, device, vb, offset, n_motifs,
                                           cond_mode=cond_mode, capin=capin, use_ref=use_ref)
                    log.info("  [val] epoch %d honest-val %.4f hard-val %.4f",
                             epoch, r, r_h)
                    r = r_h
            else:
                r = val_distance_pearson(model, vdata_d, device, vb, offset, n_motifs,
                                         capin=capin, cond_mode=cond_mode, seq_fn=seq_fn,
                                         use_ref=use_ref)
            imp = r > best
            log.info("  [val] epoch %d step %d  dist-pearson=%.4f%s",
                     epoch, step, r, "  <- best" if imp else "")
            if imp:
                best = r
                save("checkpoint-best", "best")
        curve.append({"step": step, "loss": cur_loss, "val_pearson": r})

    def stop_flag(val: bool) -> bool:
        """Agree on stopping across ranks so DDP never deadlocks."""
        if not is_ddp:
            return val
        t = torch.tensor([1.0 if val else 0.0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.train.amp)
    scaler = torch.amp.GradScaler("cuda") if cfg.train.amp == "fp16" else None
    if amp_dtype is not None and is_main:
        log.info("mixed precision: %s autocast", cfg.train.amp)

    loss_val = float("nan")
    epochs_no_improve = 0
    prev_best = best
    while step < cfg.train.max_steps and not stop:
        timed_out = bool(
            cfg.train.max_minutes and (time.time() - t0) / 60.0 >= cfg.train.max_minutes
        )
        if stop_flag(timed_out):
            if is_main:
                log.info("reached max_minutes=%.1f at step %d", cfg.train.max_minutes, step)
            break
        g = torch.Generator(device="cpu").manual_seed(cfg.train.seed * 100003 + epoch)
        perm = torch.randperm(N, generator=g).to(device)
        if obj_roles is not None and cfg.train.aml_object_pair_batches:
            from .aml_objects import object_batches
            perm = torch.from_numpy(object_batches(
                perm.cpu().numpy(), obj_roles, min(batch, N),
                np.random.default_rng(cfg.train.seed * 100003 + epoch))).to(device)
        epoch += 1
        bs = min(batch, N)
        ep_loss = nb = 0
        ep_terms: dict = {}
        b = 0
        while b + bs <= N:
            idx = perm[b : b + bs]
            cond_b, tgt, bait_b = cd[idx].float(), tg[idx].float(), bt[idx]
            ref_b = rf[idx].float() if rf is not None else None
            prior_b = pf[idx].float() if pf is not None else None
            if ref_b is not None and cfg.train.reference_dropout > 0:
                keep = (torch.rand(bs, device=device) >= cfg.train.reference_dropout)
                ref_b = ref_b * keep[:, None, None].to(ref_b.dtype)
            rev_mask = None
            if cfg.train.augment_reverse:
                rmask = torch.rand(bs, device=device) < 0.5
                rev_mask = rmask
                if rmask.any():
                    cond_r = reverse_cond(cond_b, offset, n_motifs)
                    tgt_r = torch.flip(tgt, dims=[-1, -2])
                    bait_r = torch.flip(bait_b, dims=[-1])
                    m1 = rmask[:, None, None]
                    cond_b = torch.where(m1, cond_r, cond_b)
                    tgt = torch.where(rmask[:, None, None, None], tgt_r, tgt)
                    bait_b = torch.where(m1[:, :, 0], bait_r, bait_b)
                    if ref_b is not None:
                        ref_b = torch.where(m1, torch.flip(ref_b, dims=[-1, -2]), ref_b)
                    if prior_b is not None:
                        prior_b = torch.where(m1, torch.flip(prior_b, dims=[-1, -2]), prior_b)
            if aug is not None:
                cond_b, tgt, n_a = aug(cond_b, tgt, win_scale[idx])
                n_aug_epoch += n_a
            if absent_t is not None:
                if cfg.train.track_dropout_mode == "uniform_count":
                    # per window keep k ~ Uniform{1..n_chip} tracks, drop the rest:
                    # rank tracks by random key, keep the k smallest -> drop the rest
                    k = torch.randint(1, n_chip + 1, (bs, 1), device=device)
                    rank = torch.rand(bs, n_chip, device=device).argsort(1).argsort(1)
                    drop = rank >= k
                else:
                    drop = (torch.rand(bs, n_chip, device=device)
                            < cfg.train.track_dropout)
                if coh is not None:
                    for q in coh[1]:             # SMC3 follows the RAD21 draw
                        drop[:, q] = drop[:, coh[0]]
                if drop.any():
                    cond_b = cond_b.clone()
                    cond_b[:, :n_chip] = torch.where(
                        drop[:, :, None],
                        absent_t[None, :, None].expand(bs, n_chip, cond_b.shape[-1]),
                        cond_b[:, :n_chip])
                    if coh is not None:
                        cond_b[:, coh[2]] = torch.where(
                            drop[:, coh[0], None],
                            torch.full_like(cond_b[:, coh[2]], coh[3]), cond_b[:, coh[2]])
                    if mmc is not None:
                        apply_motif_meth_drop(cond_b, drop[:, mmc[0]], mmc, rev_mask)
            sym, pairs = split_cond(cond_b, offset, n_motifs)
            planes = None
            dropped = None
            if capin:
                present = (torch.rand(bs, device=device)
                           >= cfg.capture.input_dropout).float()
                # bait_b: the bait flags flipped with reverse-augmented windows
                planes = capture_planes(tgt, bait_b, present)
                dropped = 1.0 - present
            seq_oh = None
            if use_seq:
                idx_cpu = idx.detach().cpu().tolist()
                rmask_cpu = (rev_mask.detach().cpu().tolist()
                             if rev_mask is not None else [False] * bs)
                seq_oh = batch_onehot(tr_coords, idx_cpu, rmask_cpu, W, binsize,
                                      fasta_map, device)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                pred = model(cond_b, pairs, planes, seq_oh, ref_b, prior=prior_b)
            pred = pred.float()
            pwb = None
            if pw_map is not None:
                pwb = pw_map[pw_pos[idx]]
            if sv_maps is not None:
                si = sv_idx[idx]
                if bool((si >= 0).any()):
                    m = torch.ones((bs, W, W), device=device)
                    hit = si >= 0
                    m[hit] = sv_maps[si[hit]]
                    pwb = m if pwb is None else pwb * m
            if pwb is not None:
                if rev_mask is not None:
                    pwb = torch.where(rev_mask[:, None, None], torch.flip(pwb, dims=[-1, -2]), pwb)
                pwb = pwb / pwb.mean()
            loss = hic_loss(pred[:, 0], tgt[:, 0], pwb) if pwb is not None else \
                hic_loss(pred[:, 0], tgt[:, 0])
            if obj_loss is not None:
                from .aml_objects import batch_geometry
                geo = batch_geometry([obj_rows[i] for i in idx.tolist()],
                                     rev_mask.tolist() if rev_mask is not None else None,
                                     W, obj_loss.outer, device)
                loss = loss + obj_loss(pred[:, 0], tgt[:, 0], ref_b, geo)
                for tn_, tv_ in obj_loss.last.items():
                    ep_terms[tn_] = ep_terms.get(tn_, 0.0) + tv_
            if cap_w > 0:
                bait_eff = bait_b
                if dropped is not None:
                    # aux target only where the input plane was blanked --
                    # otherwise the task degenerates to copying the input
                    bait_eff = bait_eff * dropped[:, None]
                loss = loss + cap_w * masked_pair_loss(pred[:, 1], tgt[:, 1], bait_eff)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            step += 1
            seen += bs * world
            b += bs
            loss_val = loss.item()
            ep_loss += loss_val
            nb += 1
            if len(getattr(hic_loss, "term_names", ())) > 1:
                for tn, tv in hic_loss.last_terms.items():
                    ep_terms[tn] = ep_terms.get(tn, 0.0) + tv
            now = time.time()
            timed_out = bool(
                cfg.train.max_minutes and (now - t0) / 60.0 >= cfg.train.max_minutes
            )
            if step >= cfg.train.max_steps or stop_flag(timed_out):
                stop = True
                break
            if is_main and now - last_log >= 15.0:
                log.info("epoch %d step %d  loss=%.4f  %.0f win/s  %.1fmin",
                         epoch, step, loss_val, seen / (now - t0), (now - t0) / 60.0)
                last_log = now
        if is_main and nb:
            log.info("epoch %d done  avg_loss=%.4f (step %d)%s%s", epoch, ep_loss / nb, step,
                     "".join(f"  {k}={float(v) / nb:.4f}" for k, v in ep_terms.items()),
                     f"  cohesin-augmented windows {n_aug_epoch}" if aug is not None else "")
        n_aug_epoch = 0
        if step - last_eval >= cfg.train.eval_every or stop:
            validate(loss_val)
        # epoch-based early stopping: max_epochs cap, or no val improvement for
        # patience_epochs (validate once per epoch so the count is per-epoch)
        if cfg.train.patience_epochs or cfg.train.max_epochs:
            if step != last_eval:
                validate(loss_val)
            if best > prev_best:
                prev_best = best
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            es = is_main and (
                (cfg.train.max_epochs and epoch >= cfg.train.max_epochs)
                or (cfg.train.patience_epochs
                    and epochs_no_improve >= cfg.train.patience_epochs))
            if stop_flag(bool(es)):
                if is_main:
                    log.info("early stop at epoch %d (best val %.4f, "
                             "%d epochs no improvement)", epoch, best, epochs_no_improve)
                stop = True
    if step != last_eval:
        validate(loss_val)

    if is_main:
        log.info("done. %d epochs, %d steps, best val dist-pearson=%.4f",
                 epoch, step, best)
        if curve:
            with open(out_dir / "learning_curve.csv", "w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=["step", "loss", "val_pearson"])
                wtr.writeheader()
                wtr.writerows(curve)
            _plot_curve(curve, out_dir / "learning_curve.png")
            log.info("wrote %s", out_dir / "learning_curve.png")
        if struct_rows:
            with open(out_dir / "val_structure.csv", "w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=list(struct_rows[0]))
                wtr.writeheader()
                wtr.writerows(struct_rows)
    if is_ddp:
        dist.destroy_process_group()


def train_stream(cfg: Config):
    """Train from windows streamed off the cooler on the fly (any stride,
    including the dense high-overlap regime). Reuses the fitted normalisers in
    cache_dir; supports the Hi-C channel with the sequence encoder and DDP."""
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.distributed import DistributedSampler

    from .stream import StreamWindowDataset, collate

    rank, world, local_rank = ddp_env()
    is_ddp = world > 1
    if is_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    set_seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.train.output_dir)
    cache_dir = Path(cfg.train.cache_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    if (cfg.model.reference != "none" or cfg.model.prior or cfg.train.hard_split
            or cfg.train.sv_mask or cfg.train.loss_weighting != "none"):
        raise NotImplementedError("model.reference / model.prior / train.hard_split / "
                                  "train.sv_mask / train.loss_weighting are supported on the "
                                  "resident cache path only (train.stream: false)")
    ds = StreamWindowDataset(cfg, "train", str(cache_dir), stride=cfg.train.stream_stride)
    if ds.n_out != 1:
        raise NotImplementedError(
            "streaming path handles the Hi-C channel only (no capture) for now")
    channels = ds.channels
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    n_sym = len(channels["sym_rows"])

    rng = np.random.default_rng(cfg.train.seed)
    mask = rng.random(len(ds)) < cfg.train.val_fraction
    all_idx = np.arange(len(ds))
    val_idx = all_idx[mask][:400]              # bounded held-out monitor
    tr_idx = all_idx[~mask]
    if cfg.train.val_exclude_overlap and len(val_idx):
        def _c(ix):
            e = [ds.index[i] for i in ix]
            return ([c for _, c, _ in e], [s for _, _, s in e], [n for n, _, _ in e])
        tr_idx = tr_idx[~overlaps_positions(_c(tr_idx), _c(val_idx), ds.size)]
        if is_main:
            log.info("val_exclude_overlap: %d training windows (none overlaps a "
                     "validation position)", len(tr_idx))
    train_ds = Subset(ds, tr_idx.tolist())

    use_seq = cfg.sequence.uses_encoder
    binsize = int(cfg.data.resolution)
    fasta_map = _fasta_map(cfg)
    W = ds.size
    emb = cfg.sequence.embed_dim if use_seq else 0
    in_ch = n_cond_channels(n_sym + emb, n_motifs, 0, cfg.model.cond_mode)
    encoder = (SeqEncoder(embed_dim=emb, width=cfg.sequence.enc_width)
               if use_seq else None)
    model = make_model(cfg, in_ch, 1, encoder).to(device)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    core = model.module if is_ddp else model
    if cfg.train.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    hic_loss = build_loss(cfg.train.loss, max_dist=cfg.train.loss_max_dist,
                          stride=cfg.train.loss_stride)

    vdata_d = None
    val_coords = None
    if is_main and len(val_idx):
        vt = torch.stack([ds[i]["target"] for i in val_idx])
        vc = torch.stack([ds[i]["cond"] for i in val_idx])
        vdata_d = {"target": vt.to(device), "cond": vc.to(device),
                   "bait": torch.zeros(len(val_idx), W, device=device)}
        if use_seq:
            vi = [ds.index[i] for i in val_idx]
            val_coords = ([c for _, c, _ in vi],
                          np.array([s for _, _, s in vi]),
                          [n for n, _, _ in vi])

    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
               if is_ddp else None)
    loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=cfg.train.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
        prefetch_factor=4 if cfg.train.num_workers > 0 else None)
    if is_main:
        log.info("stream training: %d train, %d val, in_ch %d, %.2fM params, "
                 "stride %d (%.1f%% overlap), %d workers",
                 len(train_ds), len(val_idx), in_ch,
                 sum(p.numel() for p in core.parameters()) / 1e6, ds.stride,
                 max(0.0, 100 * (1 - ds.stride / W)), cfg.train.num_workers)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.train.amp)
    step = epoch = 0
    best = float("-inf")
    curve: list[dict] = []
    t0 = last_log = time.time()
    last_eval = 0
    seen = 0

    def save(name, ptr):
        if not is_main or not cfg.train.save_checkpoints:
            return
        ck = out_dir / name
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(core.state_dict(), ck / "model.pt")
        for s in list(cache_dir.glob("norm_*.json")) + [cache_dir / "channels.json"]:
            if s.exists():
                (ck / s.name).write_text(s.read_text())
        cfg.save(ck / "config.yaml")
        (out_dir / ptr).write_text(name)

    def stop_flag(val: bool) -> bool:
        if not is_ddp:
            return val
        t = torch.tensor([1.0 if val else 0.0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    struct_rows: list[dict] = []
    sel_hist: list = []
    hic_measures_fn = None
    if cfg.train.select_metric not in ("dist_pearson", "combined", "hic_measures"):
        raise ValueError(f"unknown train.select_metric {cfg.train.select_metric!r}")
    if cfg.train.select_metric != "dist_pearson":
        if not cfg.train.val_regions:
            raise ValueError("train.select_metric other than dist_pearson needs "
                             "train.val_regions")
        if hdata is not None:
            raise ValueError("train.select_metric with train.hard_split is not supported")
    if cfg.train.select_metric == "hic_measures":
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        hic_measures_fn = importlib.import_module("hic_measures").combined_score
    from .structure import rank_combined

    def validate(cur_loss):
        nonlocal best, last_eval
        last_eval = step
        save("checkpoint-latest", "latest")
        if not is_main:
            return
        r = float("nan")
        if vdata_d is not None:
            seq_fn = None
            if use_seq and val_coords is not None:
                seq_fn = lambda i, n: batch_onehot(
                    val_coords, list(range(i, i + n)), [False] * n, W, binsize,
                    fasta_map, device)
            r = val_distance_pearson(
                model, vdata_d, device,
                max(1, min(cfg.train.batch_size, vdata_d["target"].shape[0])),
                offset, n_motifs, capin=False, cond_mode=cfg.model.cond_mode,
                seq_fn=seq_fn)
            if r > best:
                best = r
                save("checkpoint-best", "best")
            log.info("  [val] epoch %d step %d  dist-pearson=%.4f%s",
                     epoch, step, r, "  <- best" if r >= best else "")
        curve.append({"step": step, "loss": cur_loss, "val_pearson": r})

    stop = False
    loss_val = float("nan")
    while step < cfg.train.max_steps and not stop:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            timed_out = bool(cfg.train.max_minutes
                             and (time.time() - t0) / 60.0 >= cfg.train.max_minutes)
            if stop_flag(timed_out):
                stop = True
                break
            cond_b = batch["cond"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            bs = cond_b.shape[0]
            rev_mask = None
            if cfg.train.augment_reverse:
                rev_mask = torch.rand(bs, device=device) < 0.5
                if rev_mask.any():
                    cond_r = reverse_cond(cond_b, offset, n_motifs)
                    m1 = rev_mask[:, None, None]
                    cond_b = torch.where(m1, cond_r, cond_b)
                    tgt = torch.where(rev_mask[:, None, None, None],
                                      torch.flip(tgt, dims=[-1, -2]), tgt)
            sym, pairs = split_cond(cond_b, offset, n_motifs)
            seq_oh = None
            if use_seq:
                coords = (batch["chrom"], np.asarray(batch["start"]), batch["sample"])
                rmask_cpu = (rev_mask.detach().cpu().tolist()
                             if rev_mask is not None else [False] * bs)
                seq_oh = batch_onehot(coords, list(range(bs)), rmask_cpu, W, binsize,
                                      fasta_map, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(cond_b, pairs, None, seq_oh)
            loss = hic_loss(pred[:, 0].float(), tgt[:, 0])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            seen += bs * world
            loss_val = loss.item()
            now = time.time()
            if is_main and now - last_log >= 15.0:
                log.info("epoch %d step %d  loss=%.4f  %.0f win/s  %.1fmin",
                         epoch, step, loss_val, seen / (now - t0), (now - t0) / 60.0)
                last_log = now
            if step >= cfg.train.max_steps:
                stop = True
                break
            if step - last_eval >= cfg.train.eval_every:
                validate(loss_val)
        if not stop and is_main:
            log.info("epoch %d done (step %d)", epoch, step)
    if step != last_eval:
        validate(loss_val)
    if is_main and curve:
        with open(out_dir / "learning_curve.csv", "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=["step", "loss", "val_pearson"])
            wtr.writeheader()
            wtr.writerows(curve)
        _plot_curve(curve, out_dir / "learning_curve.png")
        log.info("done. %d steps, best val dist-pearson=%.4f", step, best)
    if is_ddp:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="Train the multi-omic Hi-C U-Net.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage-only", action="store_true",
                    help="write the train split of the cache as memory maps into "
                         "train.stage_dir and exit (single process, before torchrun)")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.stage_only:
        if not cfg.train.stage_dir:
            raise SystemExit("--stage-only needs train.stage_dir")
        stage_cache(cfg.train.cache_dir, "train", cfg.train.stage_dir)
        return
    train(cfg)


if __name__ == "__main__":
    main()
