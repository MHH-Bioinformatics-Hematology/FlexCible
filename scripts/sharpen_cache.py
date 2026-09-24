"""Cache per-region TTA predictions and matching targets for post-hoc sharpening.

Same prediction path as crosscell_predict.py; writes one npz per cell with
P (raw prediction on the target grid), T (target), region names.

    python scripts/sharpen_cache.py --cell gm12878 --bigwig-dir ... --target ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_like_hicgan as sc
from hicmultiome.config import Config
from hicmultiome.dataset import reverse_cond, split_cond
from hicmultiome.genome import match_name
from hicmultiome.matrix import MatrixNormalizer
from hicmultiome.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from hicmultiome.tracks import TrackNormalizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full2k_proper.yaml")
    ap.add_argument("--checkpoint", default="runs/full2k_proper/checkpoint-best")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--bigwig-dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(3.5 / 24.0)
    cfg = Config.load(a.config)
    ckpt = Path(a.checkpoint)
    model, channels = sc.load_model(cfg, ckpt, device)
    size, res = cfg.window.size, cfg.data.resolution
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    n_chip = len(channels["tracks"])
    row_of = {t: i for i, t in enumerate(channels["tracks"])}
    mat_norm = MatrixNormalizer.load(ckpt / f"norm_matrix_{cfg.get_sample(None).name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    absent = track_norm.absent_values()
    use_seq = cfg.sequence.uses_encoder
    fasta = cfg.get_sample(None).fasta or cfg.sequence.fasta

    sample = cfg.get_sample(None)
    sample.bigwig_dir = a.bigwig_dir
    have = {p.stem.upper() for p in Path(a.bigwig_dir).glob("*.bigwig")}
    missing = [t for t in channels["tracks"] if t.upper() not in have]
    print(f"[{a.cell}] {n_chip-len(missing)}/{n_chip} marks; missing={missing}", flush=True)
    sample.missing_tracks = list(missing)
    track_files = sample_track_files(cfg, sample)
    tgt = cooler.Cooler(a.target)

    regions = pd.read_csv(sc.REGIONS, sep="\t", names=["chrom", "start", "end"])
    Ps, Ts, names = [], [], []
    for chrom, grp in regions.groupby("chrom"):
        tname = match_name(chrom, list(tgt.chromnames))
        if tname is None:
            continue
        nb = chrom_nbins_of(tgt, tname)
        sym_full = track_norm.apply(assemble_chrom(cfg, sample, track_files, tname, nb))
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
            seq = sc.region_onehot(fasta, tname, st, size, res, device) if use_seq else None
            with torch.no_grad():
                o = model(cond, pairs, None, seq)
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
            Ps.append(P.astype(np.float32)); Ts.append(T.astype(np.float32))
            names.append(f"{chrom}:{int(r.start)}-{int(r.end)}")
    shapes = {p.shape for p in Ps}
    print(f"n={len(Ps)} shapes={shapes}", flush=True)
    np.savez_compressed(a.out, P=np.array(Ps, dtype=object) if len(shapes) > 1 else np.stack(Ps),
                        T=np.array(Ts, dtype=object) if len(shapes) > 1 else np.stack(Ts),
                        names=np.array(names))


if __name__ == "__main__":
    main()
