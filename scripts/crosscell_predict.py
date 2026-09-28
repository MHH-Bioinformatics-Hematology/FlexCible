"""Predict a cell's Hi-C from its own ChIP and score against its real map.

Runs the (dropout-trained) model on one cell's ChIP bigwigs, predicts each test
region at 2 kb, and scores against that cell's measured O/E cool on the SAME 2 kb
grid (no resampling). Missing marks (e.g. HMEC/HUVEC have no cohesin) are filled
with the absent value, so the dropout model predicts from whatever subset exists.
Also scores every OTHER cell's real map as a prediction for this cell (the
measured-map baseline), so model-vs-baseline is on identical regions/pixels.

    python scripts/crosscell_predict.py --config configs/akita_native_trackdrop.yaml \
        --checkpoint runs/akita_native_trackdrop/checkpoint-best \
        --cell imr90 --bigwig-dir /mnt/processed/wolffjoa/hic-multiome/hg38_imr90 \
        --target crosscell_oe/imr90_2kb_oe.cool \
        --others gm12878=.../gm12878_2kb_oe.cool k562=.../k562_2kb_oe.cool
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_like_hicgan as sc
from hicmultiome.config import Config
from hicmultiome.genome import match_name
from hicmultiome.matrix import MatrixNormalizer, open_cool
from hicmultiome.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from hicmultiome.tracks import TrackNormalizer


def coarse_score(M, start, end, fine_res, cres, cclr, chrom):
    """Pearson of the fine prediction M (region grid, fine_res bins starting at
    `start`), averaged into the bins of the coarser cooler `cclr`, against that
    cooler's values on the same region. None if too few bins or pixels."""
    name = match_name(chrom, list(cclr.chromnames))
    if name is None:
        return None
    c0 = (start // cres) * cres
    c1 = -(-end // cres) * cres
    try:
        T = np.asarray(cclr.matrix(balance=False).fetch(f"{name}:{c0}-{c1}"), float)
    except Exception:
        return None
    m = T.shape[0]
    if m < 20:
        return None
    centres = start + (np.arange(M.shape[0]) + 0.5) * fine_res
    idx = ((centres - c0) // cres).astype(int)
    keep = (idx >= 0) & (idx < m)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    ok = np.isfinite(M) & keep[:, None] & keep[None, :]
    s = np.zeros((m, m)); n = np.zeros((m, m))
    np.add.at(s, (ii[ok], jj[ok]), M[ok])
    np.add.at(n, (ii[ok], jj[ok]), 1.0)
    P = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    iu = np.triu_indices(m, k=2)
    x, y = P[iu], np.where(T == 0, np.nan, T)[iu]
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 100 or x[good].std() == 0 or y[good].std() == 0:
        return None
    return float(pearsonr(x[good], y[good]).statistic)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--bigwig-dir", required=True)
    ap.add_argument("--target", required=True, help="this cell's measured O/E cool (2kb)")
    ap.add_argument("--others", nargs="*", default=[], help="cell=cool of OTHER cells")
    ap.add_argument("--binsize", type=int, default=2000)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--assay", choices=["hic", "intact", "microc"], default=None,
                    help="contact assay of this cell's map, for models trained "
                         "with data.assay_indicator (default: keep the config's)")
    ap.add_argument("--coarse", nargs="*", default=[],
                    help="res=cool: also block-average the native prediction onto "
                         "that coarser grid and score it against that target")
    ap.add_argument("--regions", default=None,
                    help="chrom/start/end TSV of regions to score (default: the "
                         "413 Akita test regions)")
    ap.add_argument("--reference", default=None,
                    help="reference-map models (model.reference): the reference O/E "
                         "cool at the model resolution (default: the config's "
                         "data.reference_cool). For a held-out cell it must not "
                         "contain that cell's map")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="/mnt/processed/wolffjoa/hic-multiome/runs/crosscell")
    a = ap.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    cfg = Config.load(a.config)
    ckpt = Path(a.checkpoint)
    model, channels = sc.load_model(cfg, ckpt, device)
    size = cfg.window.size
    res = cfg.data.resolution                    # model native res == target binsize (2000)
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    n_chip = len(channels["tracks"])
    row_of = {t: i for i, t in enumerate(channels["tracks"])}
    sel_rows = None                              # model trained on a subset of the cond rows
    if cfg.train.select_tracks is not None or cfg.train.select_motifs is not None:
        from hicmultiome.train import _select_rows
        sel_rows, offset, n_motifs, _ = _select_rows(cfg, channels)
    mat_norm = MatrixNormalizer.load(ckpt / f"norm_matrix_{cfg.get_sample(None).name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    absent = track_norm.absent_values()
    use_seq = cfg.sequence.uses_encoder
    fasta = cfg.get_sample(None).fasta or cfg.sequence.fasta

    # this cell's sample: point tracks at its bigwig dir; which marks it has
    sample = cfg.get_sample(None)
    sample.bigwig_dir = a.bigwig_dir
    if a.assay is not None:
        sample.assay = a.assay
    have = {p.stem.upper(): p for p in Path(a.bigwig_dir).glob("*.bigwig")}
    missing = [t for t in channels["tracks"] if t.upper() not in have]
    print(f"[{a.cell}] {n_chip-len(missing)}/{n_chip} marks; missing={missing}", flush=True)
    sample.missing_tracks = list(missing)        # declared absent -> raw zero -> absent value
    track_files = sample_track_files(cfg, sample)

    ref_clr = None
    if cfg.model.reference != "none":
        # default: the cell's own reference when it is a training sample
        # (samples[].reference_cool), else data.reference_cool
        own = next((s for s in cfg.data.samples if s.name == a.cell), None)
        ref_path = a.reference or (cfg.sample_reference(own) if own else
                                   cfg.data.reference_cool)
        if not ref_path:
            raise SystemExit(f"{a.cell}: reference-map model needs --reference")
        # a training cell's reference was encoded with its own normaliser
        own_norm = ckpt / f"norm_matrix_{a.cell}.json"
        if own_norm.exists():
            mat_norm = MatrixNormalizer.load(own_norm)
        ref_clr = cooler.Cooler(ref_path)
        if ref_clr.binsize != res:
            raise SystemExit(f"reference {ref_path}: binsize {ref_clr.binsize} != {res}")
        print(f"[{a.cell}] reference map {ref_path} ({cfg.model.reference})", flush=True)

    tgt = cooler.Cooler(a.target)
    src = tgt                                    # chrom sizes at the model/target resolution
    coarse = {}
    for kv in a.coarse:
        k, v = kv.split("=", 1)
        coarse[int(k)] = cooler.Cooler(v)
    others = {}
    for kv in a.others:
        k, v = kv.split("=", 1)
        others[k] = cooler.Cooler(v)

    regions = pd.read_csv(a.regions or sc.REGIONS, sep="\t", usecols=[0, 1, 2],
                          names=["chrom", "start", "end"])
    rows = []
    for chrom, grp in regions.groupby("chrom"):
        tname = match_name(chrom, list(tgt.chromnames))     # target fetch (2 kb grid)
        sname = match_name(chrom, list(src.chromnames))     # nb + ChIP binning (2048)
        if tname is None or sname is None:
            continue
        nb = chrom_nbins_of(src, sname)
        sym_full = track_norm.apply(assemble_chrom(cfg, sample, track_files, sname, nb))
        # fill any missing mark rows with the absent value
        for t in missing:
            sym_full[row_of[t]] = absent[row_of[t]]
        if sel_rows is not None:
            sym_full = sym_full[sel_rows]
        print(f"  {chrom}: {len(grp)} regions", flush=True)
        for _, r in grp.iterrows():
            try:
                T = np.asarray(tgt.matrix(balance=False).fetch(
                    f"{tname}:{int(r.start)}-{int(r.end)}"), float)
            except Exception:
                continue
            n_out = T.shape[0]
            centre = (int(r.start) + int(r.end)) // 2 // res
            st = max(0, min(nb - size, centre - size // 2))
            ct = sym_full[:, st:st + size]
            if ct.shape[1] < size:
                ct = np.concatenate([ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
            cond = torch.from_numpy(ct[None]).to(device)
            pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
                     if n_motifs else None)
            seq = (sc.region_onehot(fasta, tname, st, size, res, device) if use_seq else None)
            ref = None
            if ref_clr is not None:
                from hicmultiome.matrix import fetch_diagonal_window
                rn = match_name(chrom, list(ref_clr.chromnames))
                rb = (fetch_diagonal_window(ref_clr, rn, st, size) if rn is not None
                      else np.zeros((size, size), np.float32))
                ref = torch.from_numpy(mat_norm.encode(rb)[None]).to(device)
            with torch.no_grad():
                o = model(cond, pairs, None, seq, ref)
                if a.tta:
                    from hicmultiome.dataset import reverse_cond, split_cond
                    cr = reverse_cond(cond, offset, n_motifs)
                    srr, prr = split_cond(cr, offset, n_motifs)
                    seqr = (sc.region_onehot(fasta, tname, st, size, res, device, revcomp=True)
                            if use_seq else None)
                    refr = torch.flip(ref, dims=[-1, -2]) if ref is not None else None
                    o = 0.5 * (o + torch.flip(model(srr, prr, None, seqr, refr),
                                              dims=[-1, -2]))
            P = mat_norm.decode(o[0, 0].float().cpu().numpy())
            P = 0.5 * (P + P.T)
            P = sc.resample(P, st * res, res, int(r.start), n_out, tgt.binsize)
            live = np.abs(P).sum(1) > 0
            if live.mean() < 0.5:
                continue
            M = np.where(live[:, None] & live[None, :], P, np.nan)
            Tm = np.where(T == 0, np.nan, T)
            ok = np.isfinite(M) & np.isfinite(Tm)
            rec = {"region": f"{chrom}:{int(r.start)}-{int(r.end)}"}
            if ok.sum() > 100 and M[ok].std() > 0 and Tm[ok].std() > 0:
                rec["model"] = float(pearsonr(M[ok], Tm[ok]).statistic)
            # the same prediction averaged onto coarser grids, scored there; only
            # bins inside the model window count (resample extends the edges of
            # a window shorter than the region by nearest-value copies)
            if coarse:
                cen = int(r.start) + (np.arange(n_out) + 0.5) * tgt.binsize
                cov = (cen >= st * res) & (cen < (st + size) * res)
                Mc = np.where(cov[:, None] & cov[None, :], M, np.nan)
            for cres, cclr in coarse.items():
                r_c = coarse_score(Mc, int(r.start), int(r.end), tgt.binsize, cres, cclr, chrom)
                if r_c is not None:
                    rec[f"model_{cres // 1000}kb"] = r_c
            # measured-map baselines: other cells' REAL map vs this cell's real map
            iu = np.triu_indices(n_out, k=2)
            for oc, oclr in others.items():
                onm = match_name(chrom, list(oclr.chromnames))
                try:
                    O = np.asarray(oclr.matrix(balance=False).fetch(
                        f"{onm}:{int(r.start)}-{int(r.end)}"), float)
                except Exception:
                    continue
                if O.shape == T.shape:
                    x, y = O[iu], T[iu]
                    m2 = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
                    if m2.sum() > 100 and x[m2].std() > 0 and y[m2].std() > 0:
                        rec[f"base_{oc}"] = float(pearsonr(x[m2], y[m2]).statistic)
            rows.append(rec)

    df = pd.DataFrame(rows)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"predict_{a.cell}.csv", index=False)
    print(f"\n=== {a.cell}: model prediction vs measured-map baselines ===", flush=True)
    cols = [c for c in df.columns if c != "region"]
    summ = df[cols].median().sort_values(ascending=False)
    for k, v in summ.items():
        tag = ("MODEL (from ChIP)" if k == "model" else
               f"MODEL averaged to {k[6:]}" if k.startswith("model_") else
               k.replace("base_", "real ") + " -> " + a.cell)
        print(f"  {v:.4f}  {tag}", flush=True)
    print(f"  n={df['model'].notna().sum()} regions; wrote predict_{a.cell}.csv")


if __name__ == "__main__":
    main()
