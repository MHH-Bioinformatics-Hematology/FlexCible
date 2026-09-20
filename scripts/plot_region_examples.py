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

from hicmultiome.config import Config
from hicmultiome.genome import match_name
from hicmultiome.matrix import MatrixNormalizer, open_cool
from hicmultiome.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from hicmultiome.tracks import TrackNormalizer
from hicmultiome.train import _select_rows

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
            from hicmultiome.dataset import reverse_cond, split_cond
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

    rows = len(picks)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))
    for r, (band, row) in enumerate(picks):
        chrom, span = row["region"].split(":")
        start, end = (int(x) for x in span.split("-"))
        tname = match_name(chrom, list(tgt.chromnames))
        T = np.asarray(tgt.matrix(balance=False).fetch(f"{tname}:{start}-{end}"), dtype=float)
        n_out = T.shape[0]
        P = predict_region(cfg, model, channels, sel_rows, offset, n_motifs, use_seq,
                           fasta, mat_norm, track_norm, track_files, sample, src,
                           chrom, start, end, n_out, tgt.binsize, device, a.tta)
        Tm = np.where(T == 0, np.nan, T)
        kw = dict(cmap="RdBu_r", vmin=-2, vmax=2)
        axes[r, 0].imshow(Tm, **kw)
        axes[r, 1].imshow(P, **kw)
        ar = row.get("akita_gm12878", float("nan"))
        axes[r, 0].set_ylabel(f"{band}\n{row['region']}", fontsize=7)
        axes[r, 0].set_title("ground truth" if r == 0 else "", fontsize=9)
        axes[r, 1].set_title("our prediction" if r == 0 else "", fontsize=9)
        axes[r, 1].text(0.02, 0.06, f"ours r={row['ours']:.2f}\nAkita r={ar:.2f}",
                        transform=axes[r, 1].transAxes, fontsize=7,
                        bbox=dict(fc="white", alpha=0.7, lw=0))
        for c in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("RAD21+H3K4me1+H3K4me2 vs Akita target (log obs/exp)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
