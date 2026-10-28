"""Window prior network P (registry section 76; flexcible/windowprior.py).

P maps the CONTENT of a window (reference marks, no coordinates) to the typical Hi-C of such a window
(the leave-one-out consensus O/E map). Its output replaces the positional consensus of the reference
models (all12wp_* runs of scripts/multicell_expanded.py).

Stages
  train     one P: --mode ref|own, --fold odd|even|full, [--motifs] [--chrho]; training windows from the
            cache of all12diffcells_refres_drop (train split, the 12 cells, the chromosomes of the fold),
            input = the dref rows (ref) or the sample's own anchored marks (own), target = the cell's
            leave-one-out consensus window; honest validation windows (no overlap with training windows)
  predict   cooler of P(marks) per sample: a chromosome is predicted by the P trained without it
            (odd / even cross-fitting) or by the full P (--source full)
  ponly     P alone as a predictor on the 413 Akita regions: Pearson with the cell's own Hi-C, with the
            leave-one-out consensus, and the consensus with the cell's own Hi-C
  controls  position-free control: the signal rolled by whole windows and the chromosome labels permuted;
            maximum absolute difference of P (and, as the positive control, of the positional consensus)

    python scripts/window_prior.py train --res 10000 --mode ref --fold odd
    python scripts/window_prior.py predict --res 10000 --mode ref --cells gm12878 k562
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
WP = HM / "window_prior"
OE = HM / "crosscell_oe"
MR = HM / "runs/multicell_reference"
CACHE = HM / "cache_expanded"
VN = HM / "validation_new"
CHRHO_HELD = ("chr2", "chr3", "chr11", "chr14", "chr17")
NEW_SAMPLES = ["pc3", "panc1", "nk", "caco2"]
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248", "mcf7",
           "cd14mono", "pc9"]
HELD_OUT = ["k562", "hmec", "huvec", "hmec_deep", "huvec_deep"]
IDENT = {"hmec_deep": "hmec", "huvec_deep": "huvec"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- names and paths
def tag_of(mode: str, motifs: bool, chrho: bool) -> str:
    return mode + ("_motif" if motifs else "") + ("_chrho" if chrho else "")


def model_dir(res: int, tag: str, fold: str) -> Path:
    return WP / f"{res // 1000}kb" / tag / f"P_{fold}"


def cool_path(res: int, tag: str, cell: str, source: str = "oof") -> Path:
    return WP / f"{res // 1000}kb" / tag / ("cools" if source == "oof" else "cools_full") / f"{cell}.cool"


def base_run(res: int) -> str:
    return f"all12diffcells_refres_drop_{res // 1000}kb"


def base_cfg(res: int):
    from flexcible.config import Config
    return Config.load(str(MR / f"{base_run(res)}.yaml"))


# --------------------------------------------------------------------------- DiffMarks helpers
def local_marks_cache(cfg, res: int):
    """Point the diffmarks disk cache at a folder of this feature (the 12 cells' arrays copied once from
    the cache of the diff run), so that new samples never write into a running job's cache."""
    import shutil
    src = Path(cfg.train.cache_dir) / "diffmarks"
    dst = WP / f"{res // 1000}kb" / "diffmarks"
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for f in src.glob("*.npz"):
            if not (dst / f.name).exists():
                shutil.copyfile(f, dst / (f.name + ".part"))
                (dst / (f.name + ".part")).rename(dst / f.name)
    cfg.train.cache_dir = str(WP / f"{res // 1000}kb")
    return cfg


def cell_sample(cfg, cell: str):
    """SampleSpec of `cell` for the mark rows: the configured sample of a training cell; a copy of the
    first sample pointed at the cell's bigWigs for a held-out cell or a new sample (the reference of
    which is the 12-cell pool without the cell's identity, as flexcible.diffmarks.configure_unseen)."""
    import copy

    import multicell_expanded as me
    by = {s.name: s for s in cfg.data.samples}
    if cell in by:
        return by[cell]
    s = copy.deepcopy(cfg.data.samples[0])
    s.name = cell
    if cell in me.CELLS:
        bw = me.CELLS[cell][1]
        s.missing_tracks = list(me.missing_marks(bw))
        s.assay = me.CELLS[cell][2]
    else:
        bw = Path(os.environ.get("IV_BW", str(Path.home() / "scratch/iv_bw"))) / cell
        if not (bw / "COPIED").exists():
            bw = VN / f"hg38_{cell}"
        have = {p.stem.upper() for p in Path(bw).glob("*.bigwig")}
        s.missing_tracks = [t for t in cfg.track_names if t.upper() not in have]
    s.bigwig_dir = str(bw)
    s.extra_bigwig_dirs = []
    s.zeroed_tracks = []
    s.diff_identity = IDENT.get(cell, cell)
    s.diff_set = "consensus"
    return s


# --------------------------------------------------------------------------- training data
def load_windows(res: int, mode: str, chroms: list[str], motifs: bool):
    """Training-split windows of the 12 cells on `chroms`: marks (n, M, W) float16, target (n, W, W)
    float16 = raw leave-one-out consensus / PRIOR_SCALE clipped, pairs (n, K, 2, W) or None,
    coords (chrom list, start array, sample list)."""
    from flexcible import diffmarks as dmk
    from flexcible.matrix import MatrixNormalizer
    from flexcible.windowprior import PRIOR_SCALE, mark_rows

    cfg = local_marks_cache(base_cfg(res), res)
    cache = CACHE / f"mc_all12ref_diff_{res // 1000}kb"
    man = json.loads((cache / "manifest.json").read_text())
    ck = MR / base_run(res) / "checkpoint-best"
    W = int(man["window_size"])
    inst = dmk.DiffMarks.get(cfg)
    spec = {s.name: s for s in cfg.data.samples}
    norms = {s: MatrixNormalizer.load(ck / f"norm_matrix_{s}.json") for s in spec}
    sym = man["sym_rows"]
    dref_idx = [sym.index(f"dref:{t}") for t in inst.marks]
    mot_idx = [i for i, r in enumerate(sym) if r.startswith("motif:")]
    X, T, P, ch, st, sm = [], [], [], [], [], []
    want = set(chroms)
    cache_arr = {}
    for sh in man["shards"]:
        if sh["split"] != "train" or sh["chrom"] not in want:
            continue
        z = np.load(sh["path"])
        starts = z["start"].astype(int)
        nb = int(z["nbins"][0])
        key = (sh["sample"], sh["chrom"])
        if key not in cache_arr:
            cache_arr = {key: mark_rows(inst, spec[sh["sample"]], sh["chrom"], nb, mode)}
        arr = cache_arr[key]
        x = np.stack([arr[:, s:s + W] for s in starts]).astype(np.float16)
        if mode == "ref" and len(starts):          # the rows are the dref rows of the cache
            c = z["cond"][:, dref_idx, :].astype(np.float32)
            err = float(np.abs(c - x.astype(np.float32)).max())
            if err > 5e-3:
                raise RuntimeError(f"{sh['path']}: dref rows differ from the cache by {err}")
        t = np.clip(z["ref"].astype(np.float32) * norms[sh["sample"]].scale / PRIOR_SCALE, -1, 1)
        X.append(x)
        T.append(t.astype(np.float16))
        if motifs:
            P.append(z["cond"][:, mot_idx, :].reshape(len(starts), -1, 2, W).astype(np.float16))
        ch += [sh["chrom"]] * len(starts)
        st += starts.tolist()
        sm += [sh["sample"]] * len(starts)
    if not X:
        raise RuntimeError("no training windows")
    return (np.concatenate(X), np.concatenate(T), np.concatenate(P) if motifs else None,
            (ch, np.asarray(st), sm), W)


def window_pearson(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Per-window Pearson over the pixels >= 2 bins off the diagonal with a non-zero target."""
    W = pred.shape[-1]
    iu = np.triu_indices(W, 2)
    out = np.full(len(pred), np.nan)
    for i in range(len(pred)):
        x, y = pred[i][iu].astype(np.float64), tgt[i][iu].astype(np.float64)
        m = y != 0
        if m.sum() > 100 and x[m].std() > 0 and y[m].std() > 0:
            out[i] = np.corrcoef(x[m], y[m])[0, 1]
    return out


# --------------------------------------------------------------------------- train
def stage_train(a):
    import torch
    import torch.nn.functional as F

    from flexcible.train import split_val_regions
    from flexcible.windowprior import (drop_marks, make_prior_net, prior_forward, reverse_inputs,
                                         train_chroms_of)

    tag = tag_of(a.mode, a.motifs, a.chrho)
    out = model_dir(a.res, tag, a.fold)
    if (out / "DONE").exists():
        log(f"[have] {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    chroms = train_chroms_of(a.fold, CHRHO_HELD if a.chrho else ())
    log(f"P {tag} {a.fold}: {len(chroms)} chromosomes {chroms}")
    X, T, Pm, co, W = load_windows(a.res, a.mode, chroms, a.motifs)
    n = len(X)
    n_val = a.val_per_cell * len(TRAIN12) // (1 if a.fold == "full" else 2)
    vi, ti = split_val_regions(co, n_val, W, a.seed, True)
    log(f"windows {n}: train {len(ti)}, validation {len(vi)} (no overlap), W={W}, marks {X.shape[1]}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    net = make_prior_net(X.shape[1], 0 if Pm is None else Pm.shape[1], a.base, a.depth).to(dev)
    log(f"parameters {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    gen = torch.Generator().manual_seed(a.seed)
    Xt, Tt = torch.from_numpy(X), torch.from_numpy(T)
    Pt = torch.from_numpy(Pm) if Pm is not None else None

    def batch(idx, train):
        x = Xt[idx].float()
        t = Tt[idx].float()
        p = Pt[idx].float() if Pt is not None else None
        if train:
            x = drop_marks(x, gen, a.dropout)
            if torch.rand(1, generator=gen).item() < 0.5:        # genomic reversal
                x, p = reverse_inputs(x, p)
                t = torch.flip(t, dims=[-1, -2])
        return x.to(dev), t.to(dev), (p.to(dev) if p is not None else None)

    def evaluate():
        net.eval()
        L, outs = 0.0, []
        with torch.no_grad():
            for i in range(0, len(vi), 64):
                x, t, p = batch(vi[i:i + 64], False)
                o = prior_forward(net, x, p)
                L += (F.l1_loss(o, t, reduction="sum") + F.mse_loss(o, t, reduction="sum")).item()
                outs.append(o.float().cpu().numpy())
        r = window_pearson(np.concatenate(outs), Tt[vi].float().numpy())
        return L / (len(vi) * W * W), float(np.nanmedian(r))

    best, bad, rows = 1e9, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = ti[np.random.default_rng(a.seed + ep).permutation(len(ti))]
        tl, nb_ = 0.0, 0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            x, t, p = batch(perm[i:i + a.batch], True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = prior_forward(net, x, p)
            o = o.float()
            loss = F.l1_loss(o, t) + F.mse_loss(o, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item()
            nb_ += 1
        vl, vr = evaluate()
        sched.step(vl)
        rows.append({"epoch": ep, "train_loss": tl / max(nb_, 1), "val_loss": vl, "val_median_pearson": vr,
                     "minutes": (time.time() - t0) / 60})
        log(f"epoch {ep}: train {tl / max(nb_, 1):.4f} val {vl:.4f} median r {vr:.4f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            torch.save({"state": net.state_dict(), "n_marks": int(X.shape[1]),
                        "n_motifs": 0 if Pm is None else int(Pm.shape[1]), "base": a.base, "depth": a.depth,
                        "window": W, "res": a.res, "mode": a.mode, "motifs": a.motifs, "fold": a.fold,
                        "chroms": chroms, "epoch": ep, "val_loss": vl, "val_median_pearson": vr,
                        "marks": list(base_cfg(a.res).track_names)}, out / "model.pt")
        else:
            bad += 1
            if bad >= a.patience:
                log(f"early stop at epoch {ep}")
                break
    import csv
    with open(out / "log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "DONE").write_text(json.dumps({"epochs": len(rows), "best_val_loss": best,
                                          "minutes": (time.time() - t0) / 60,
                                          "n_train": int(len(ti)), "n_val": int(len(vi))}))
    log(f"done {out}: {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- production
def load_net(path: Path, device: str):
    import torch

    from flexcible.windowprior import make_prior_net
    ck = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    net = make_prior_net(ck["n_marks"], ck["n_motifs"], ck["base"], ck["depth"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck


def motif_pairs(cfg, res: int, chrom: str, nb: int, memo: dict):
    """Normalised motif fwd / rev rows (K, 2, nb) of a chromosome, as the model's cond rows."""
    from flexcible.motifs import cached_motif_tracks
    from flexcible.tracks import TrackNormalizer
    if chrom in memo:
        return memo[chrom]
    ck = MR / base_run(res) / "checkpoint-best"
    tn = TrackNormalizer.load(ck / "norm_tracks.json")
    chan = json.loads((ck / "channels.json").read_text())
    rows = chan["sym_rows"]
    fasta = cfg.sample_fasta(cfg.data.samples[0])
    full = np.zeros((len(rows), nb), np.float32)
    idx = []
    for mid in cfg.active_motif_ids:
        t = cached_motif_tracks(fasta, chrom, mid, cfg.motifs.dir, nb, res, cfg.motifs.score_frac,
                                cfg.sequence.cache_dir)
        for k, nm in enumerate((f"motif:{mid}:fwd", f"motif:{mid}:rev")):
            full[rows.index(nm)] = t[k]
            idx.append(rows.index(nm))
    z = tn.apply(full)[idx].reshape(-1, 2, nb)
    memo[chrom] = z.astype(np.float32)
    return memo[chrom]


def stage_predict(a):
    import cooler
    import torch

    from flexcible import diffmarks as dmk
    from flexcible.windowprior import mark_rows, predict_chrom, predictor_fold, write_prior_cool

    tag = tag_of(a.mode, a.motifs, a.chrho)
    cfg = local_marks_cache(base_cfg(a.res), a.res)
    inst = dmk.DiffMarks.get(cfg)
    tmpl = OE / f"consensus12_proper_{a.res // 1000}kb.cool"
    grid = cooler.Cooler(str(tmpl))
    dev = a.device if torch.cuda.is_available() else "cpu"
    nets = {}
    memo = {}
    chroms = [c for c in grid.chromnames if c in {f"chr{i}" for i in range(1, 23)}]
    for cell in a.cells:
        outp = cool_path(a.res, tag, cell, a.source)
        if outp.exists():
            log(f"[have] {outp}")
            continue
        sample = cell_sample(cfg, cell)
        accs = {}
        for chrom in chroms:
            fold = predictor_fold(chrom, a.source)
            if a.chrho and chrom in CHRHO_HELD:
                fold = "full"                     # trained on the other chromosomes only: never saw these
            if fold not in nets:
                nets[fold] = load_net(model_dir(a.res, tag, fold), dev)
            net, ck = nets[fold]
            nb = int(grid.extent(chrom)[1] - grid.offset(chrom))
            x = mark_rows(inst, sample, chrom, nb, a.mode)
            pr = motif_pairs(cfg, a.res, chrom, nb, memo) if ck["n_motifs"] else None
            accs[chrom] = predict_chrom(net, x, pr, ck["window"], device=dev)
        pool = [n for n in dmk.reference_members(list(cfg.data.diff_reference_sets["consensus"]),
                                                 sample.diff_identity or sample.name)]
        meta = {"kind": "window_prior", "mode": a.mode, "motifs": a.motifs, "chrho": a.chrho,
                "source": a.source, "res": a.res, "cell": cell,
                # cells whose Hi-C or marks enter the INPUT of the map (leakage guard: never the cell)
                "cells": pool if a.mode == "ref" else [],
                "input_marks": ("mean of the reference pool" if a.mode == "ref" else "the sample itself"),
                "trained_with_hic_of": TRAIN12, "folds": "odd/even cross-fitted" if a.source == "oof" else "full",
                "prior_scale": 2.0}
        write_prior_cool(outp, tmpl, accs, meta)
        log(f"wrote {outp}")


# --------------------------------------------------------------------------- P alone
def stage_ponly(a):
    import cooler
    import pandas as pd
    from scipy.stats import pearsonr

    import score_like_hicgan as sc
    import multicell_expanded as me

    tag = tag_of(a.mode, a.motifs, a.chrho)
    reg = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    rows = []
    for cell in a.cells:
        p_path = cool_path(a.res, tag, cell, a.source)
        tgt = me.target(cell, a.res)
        cons = me.measured_reference(f"all12refres_drop_{a.res // 1000}kb", cell)
        if not (p_path.exists() and tgt.exists() and cons.exists()):
            log(f"[skip] {cell}")
            continue
        cP, cT, cC = (cooler.Cooler(str(p)) for p in (p_path, tgt, cons))
        for _, r in reg.iterrows():
            def f(c):
                try:
                    return np.asarray(c.matrix(balance=False).fetch(f"{r.chrom}:{int(r.start)}-{int(r.end)}"), float)
                except Exception:                                                # noqa: BLE001
                    return None
            P, T, C = f(cP), f(cT), f(cC)
            if P is None or T is None or C is None or not (P.shape == T.shape == C.shape):
                continue
            iu = np.triu_indices(T.shape[0], 2)

            def rr(x, y):
                x, y = x[iu], y[iu]
                m = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
                if m.sum() > 100 and x[m].std() > 0 and y[m].std() > 0:
                    return float(pearsonr(x[m], y[m]).statistic)
                return np.nan
            rows.append({"res": a.res, "tag": tag, "cell": cell, "region": f"{r.chrom}:{int(r.start)}-{int(r.end)}",
                         "r_prior_own": rr(P, T), "r_consensus_own": rr(C, T), "r_prior_consensus": rr(P, C)})
        log(f"ponly {cell}: {sum(1 for x in rows if x['cell'] == cell)} regions")
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"ponly_regions_{tag}_{a.res // 1000}kb_{a.source}.csv", index=False)
    summ = df.groupby("cell")[["r_prior_own", "r_consensus_own", "r_prior_consensus"]].median().round(4)
    summ["n_regions"] = df.groupby("cell").size()
    summ["trained_on_cell"] = [c in TRAIN12 for c in summ.index]
    summ.to_csv(out / f"ponly_{tag}_{a.res // 1000}kb_{a.source}.csv")
    log("\n" + summ.to_string())


# --------------------------------------------------------------------------- controls
def stage_controls(a):
    """Position-free control. For each cell and chromosome the signal is rolled by a random whole number
    of windows (the grid stays aligned), the chromosome labels are permuted over the cells' chromosomes,
    and P is run through the production path; the maps are compared after undoing the roll, away from
    the roll seam and the chromosome ends. Positive control: the positional consensus read at the same
    shifted coordinates."""
    import cooler
    import torch

    from flexcible import diffmarks as dmk
    from flexcible.windowprior import mark_rows, shift_difference

    tag = tag_of(a.mode, a.motifs, a.chrho)
    cfg = local_marks_cache(base_cfg(a.res), a.res)
    inst = dmk.DiffMarks.get(cfg)
    tmpl = OE / f"consensus12_proper_{a.res // 1000}kb.cool"
    grid = cooler.Cooler(str(tmpl))
    dev = a.device if torch.cuda.is_available() else "cpu"
    net, ck = load_net(model_dir(a.res, tag, "full"), dev)
    W = ck["window"]
    rng = np.random.default_rng(a.seed)
    chroms = [c for c in grid.chromnames if c in {f"chr{i}" for i in range(1, 23)}]
    memo = {}
    rows = []
    for cell in a.cells:
        sample = cell_sample(cfg, cell)
        arrs = {}
        for c in chroms:
            nb = int(grid.extent(c)[1] - grid.offset(c))
            arrs[c] = (mark_rows(inst, sample, c, nb, a.mode),
                       motif_pairs(cfg, a.res, c, nb, memo) if ck["n_motifs"] else None)
        perm = rng.permutation(len(chroms))
        # same-length chromosomes are not required: the labels are permuted over the data dict, the
        # signal of chromosome A is run under the label of chromosome B (labels never reach P)
        relabel = {chroms[i]: chroms[j] for i, j in enumerate(perm)}
        mx, n_cmp = 0.0, 0
        ref_cons = cooler.Cooler(str(OE / f"consensus_loo_{cell}_proper_{a.res // 1000}kb.cool")) \
            if cell in TRAIN12 else cooler.Cooler(str(tmpl))
        pos_diff, pos_n = 0.0, 0
        for c in chroms[: a.n_chroms]:
            x, pr = arrs[c]
            nb = x.shape[1]
            k = int(rng.integers(1, max(2, (nb // W) // 2)))
            shift = k * (W // 2)                       # whole strides: the grid stays aligned
            d, npx = shift_difference(net, x, pr, W, k, dev)
            if npx == 0:
                continue
            mx = max(mx, d)
            n_cmp += 1
            # positive control: the positional consensus looked up at the shifted coordinates
            bsz = a.res
            s0 = 3 * W * bsz
            try:
                M0 = np.asarray(ref_cons.matrix(balance=False).fetch(f"{c}:{s0}-{s0 + W * bsz}"), float)
                M1 = np.asarray(ref_cons.matrix(balance=False).fetch(f"{c}:{s0 + shift * bsz}-{s0 + (shift + W) * bsz}"), float)
                if M0.shape == M1.shape:
                    pos_diff = max(pos_diff, float(np.nanmax(np.abs(M0 - M1))))
                    pos_n += 1
            except Exception:                                                    # noqa: BLE001
                pass
        rows.append({"res": a.res, "tag": tag, "cell": cell, "chromosomes_compared": n_cmp,
                     "label_permutation_example": json.dumps(dict(list(relabel.items())[:4])),
                     "max_abs_diff_prior": mx, "max_abs_diff_positional_consensus": pos_diff,
                     "positional_pairs": pos_n})
        log(f"control {cell}: P max abs diff {mx:.3g}; positional consensus {pos_diff:.3g}")
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"controls_position_{tag}_{a.res // 1000}kb.csv", index=False)


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["train", "predict", "ponly", "controls"])
    ap.add_argument("--res", type=int, default=10000, choices=[10000, 25000])
    ap.add_argument("--mode", default="ref", choices=["ref", "own"])
    ap.add_argument("--fold", default="full", choices=["odd", "even", "full"])
    ap.add_argument("--motifs", action="store_true", help="add the CTCF motif planes (ablation)")
    ap.add_argument("--chrho", action="store_true", help="train without chr2, 3, 11, 14, 17")
    ap.add_argument("--source", default="oof", choices=["oof", "full"])
    ap.add_argument("--cells", nargs="*", default=TRAIN12 + HELD_OUT)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-chroms", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "results/window_prior"))
    a = ap.parse_args()
    {"train": stage_train, "predict": stage_predict, "ponly": stage_ponly,
     "controls": stage_controls}[a.stage](a)


if __name__ == "__main__":
    main()
