"""Region examples: ground-truth vs our prediction for good / medium / bad cases.

Ranks the Akita test regions by our per-region Pearson (from an akita score CSV),
picks the top N (good), middle N (medium) and bottom N (bad), predicts each with
the model, and draws a grid of target-vs-prediction log(obs/exp) maps. Each panel
title shows our Pearson and Akita's recorded Pearson for that region, so the
comparison to Akita is on the figure even though Akita's own maps are not
available (only their published per-region scores).

    python scripts/plot_region_examples.py --config configs/combo_rad21_full.yaml \
        --checkpoint <run>/checkpoint-best --csv <run>/akita_final.csv \
        --out <run>/region_examples.png --n 3 --tta
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
import score_like_hicgan as sc          # load_model, resample, region_onehot, TARGET

from flexcible.config import Config
from flexcible.genome import match_name
from flexcible.matrix import MatrixNormalizer, open_cool
from flexcible.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from flexcible.tracks import TrackNormalizer
from flexcible.train import _select_rows

load_model, resample, region_onehot = sc.load_model, sc.resample, sc.region_onehot


def predict_region(cfg, model, channels, sel_rows, offset, n_motifs, use_seq,
                   fasta, mat_norm, track_norm, track_files, sample, src,
                   chrom, start, end, n_out, tgt_binsize, device, tta):
    size, res = cfg.window.size, cfg.data.resolution
    name = match_name(chrom, list(src.chromnames))
    nb = chrom_nbins_of(src, name)
    sym_full = track_norm.apply(assemble_chrom(cfg, sample, track_files, name, nb))
    if sel_rows is not None:
        sym_full = sym_full[sel_rows]
    centre_bin = (start + end) // 2 // res
    st = max(0, min(nb - size, centre_bin - size // 2))
    ct = sym_full[:, st:st + size]
    if ct.shape[1] < size:
        ct = np.concatenate([ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
    cond = torch.from_numpy(ct[None]).to(device)
    pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
             if n_motifs else None)
    seq = region_onehot(fasta, name, st, size, res, device) if use_seq else None
    with torch.no_grad():
        out = model(cond, pairs, None, seq)
        if tta:
            from flexcible.dataset import reverse_cond, split_cond
            cr = reverse_cond(cond, offset, n_motifs)
            sr, pr = split_cond(cr, offset, n_motifs)
            seqr = region_onehot(fasta, name, st, size, res, device, revcomp=True) if use_seq else None
            out = 0.5 * (out + torch.flip(model(sr, pr, None, seqr), dims=[-1, -2]))
    P = mat_norm.decode(out[0, 0].float().cpu().numpy())
    P = 0.5 * (P + P.T)
    return resample(P, st * res, res, start, n_out, tgt_binsize)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TARGET = sc.TARGET                      # Akita target cool

    cfg = Config.load(a.config)
    sample = cfg.get_sample(None)
    use_seq = cfg.sequence.uses_encoder
    fasta = sample.fasta or cfg.sequence.fasta
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, channels = load_model(cfg, Path(a.checkpoint), device)
    if cfg.train.select_tracks:
        sel_rows, offset, n_motifs, _ = _select_rows(cfg, channels)
    else:
        sel_rows = None
        offset = len(channels["tracks"]) + len(channels["seq_features"])
        n_motifs = len(channels["motif_ids"])
    mat_norm = MatrixNormalizer.load(Path(a.checkpoint) / f"norm_matrix_{sample.name}.json")
    track_norm = TrackNormalizer.load(Path(a.checkpoint) / "norm_tracks.json")
    track_files = sample_track_files(cfg, sample)
    tgt = cooler.Cooler(str(TARGET))
    src = open_cool(sample.cool_path)

    df = pd.read_csv(a.csv).dropna(subset=["ours"]).sort_values("ours", ascending=False)
    df = df.reset_index(drop=True)
    n = a.n
    mid = len(df) // 2
    picks = [("good", df.iloc[i]) for i in range(n)] \
        + [("medium", df.iloc[mid - n // 2 + i]) for i in range(n)] \
        + [("bad", df.iloc[len(df) - n + i]) for i in range(n)]

    # 3x3 grid: rows = good/medium/bad, each a split map (upper=truth, lower=ours)
    bands = ["good", "medium", "bad"]
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 3.4 * 3))
    kw = dict(cmap="RdBu_r", vmin=-2, vmax=2)
    for idx, (band, row) in enumerate(picks):
        br, bc = bands.index(band), idx % n
        ax = axes[br, bc] if n > 1 else axes[br]
        chrom, span = row["region"].split(":")
        start, end = (int(x) for x in span.split("-"))
        tname = match_name(chrom, list(tgt.chromnames))
        T = np.asarray(tgt.matrix(balance=False).fetch(f"{tname}:{start}-{end}"), dtype=float)
        n_out = T.shape[0]
        P = predict_region(cfg, model, channels, sel_rows, offset, n_motifs, use_seq,
                           fasta, mat_norm, track_norm, track_files, sample, src,
                           chrom, start, end, n_out, tgt.binsize, device, a.tta)
        # split map: upper triangle (incl. diagonal) = original, lower = prediction
        M = P.astype(float).copy()
        ut = np.triu_indices(n_out, 0)
        M[ut] = np.where(T[ut] == 0, np.nan, T[ut])
        ax.imshow(M, **kw)
        ar = row.get("akita_gm12878", float("nan"))
        ax.set_title(f"{row['region']}\nours r={row['ours']:.2f} | Akita r={ar:.2f}",
                     fontsize=7)
        if bc == 0:
            ax.set_ylabel(band, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("upper = original Hi-C  |  lower = flexcible "
                 "(RAD21+H3K4me1+H3K4me2), log obs/exp", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
