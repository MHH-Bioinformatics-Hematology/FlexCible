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

    tgt = cooler.Cooler(a.target)
    src = tgt                                    # chrom sizes at the model/target resolution
    others = {}
    for kv in a.others:
        k, v = kv.split("=", 1)
        others[k] = cooler.Cooler(v)

    regions = pd.read_csv(sc.REGIONS, sep="\t", names=["chrom", "start", "end"])
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
            with torch.no_grad():
                o = model(cond, pairs, None, seq)
                if a.tta:
                    from hicmultiome.dataset import reverse_cond, split_cond
                    cr = reverse_cond(cond, offset, n_motifs)
                    srr, prr = split_cond(cr, offset, n_motifs)
                    seqr = (sc.region_onehot(fasta, tname, st, size, res, device, revcomp=True)
                            if use_seq else None)
                    o = 0.5 * (o + torch.flip(model(srr, prr, None, seqr), dims=[-1, -2]))
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
        tag = "MODEL (from ChIP)" if k == "model" else k.replace("base_", "real ") + " -> " + a.cell
        print(f"  {v:.4f}  {tag}", flush=True)
    print(f"  n={df['model'].notna().sum()} regions; wrote predict_{a.cell}.csv")


if __name__ == "__main__":
    main()
