"""Three-model split-map figure on Akita's test regions.

Rows = models (Akita, hic-multiome-unet 3-track, hic-multiome-unet all-tracks);
columns = a good / medium / bad region (by the all-tracks model's per-region
Pearson). Each panel is one region as a split map: upper triangle = original
Hi-C (Akita's target), lower triangle = that model's prediction, log(obs/exp),
same +/-2 scale. Akita's predicted maps come from akita_testset_rasters.npz;
our models are run here. Panel titles show each model's per-region Pearson.

    python scripts/plot_three_models.py --out <path>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from hicmultiome.train import _select_rows

NPZ = "/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/akita_testset_rasters.npz"
CELL = 1  # cells = ['HFF', 'GM12878']

# our models: (row label, config, checkpoint)
MODELS = [
    ("hic-multiome-unet (RAD21+H3K4me1+H3K4me2)",
     "configs/combo_rad21_full.yaml",
     "/mnt/processed/wolffjoa/hic-multiome/runs/combo_rad21_full/checkpoint-best"),
    ("hic-multiome-unet (all 13 ChIP)",
     "configs/akita_native.yaml",
     "runs/akita_native/checkpoint-best"),
]
NATIVE_CSV = "runs/akita_native/score_like_hicgan.csv"   # for ranking regions


def load_our_model(config, ckpt, device):
    cfg = Config.load(config)
    model, channels = sc.load_model(cfg, Path(ckpt), device)
    if cfg.train.select_tracks:
        sel_rows, offset, n_motifs, _ = _select_rows(cfg, channels)
    else:
        sel_rows = None
        offset = len(channels["tracks"]) + len(channels["seq_features"])
        n_motifs = len(channels["motif_ids"])
    sample = cfg.get_sample(None)
    if not Path(sample.cool_path).exists():          # training cool moved; chrom sizes only
        sample.cool_path = ("/mnt/processed/wolffjoa/restart_of_prediction_project/"
                            "akita_normalization/akita_published_GM12878_test_2048bp.cool")
    return dict(cfg=cfg, model=model, channels=channels, sel_rows=sel_rows,
                offset=offset, n_motifs=n_motifs, sample=sample,
                use_seq=cfg.sequence.uses_encoder,
                fasta=sample.fasta or cfg.sequence.fasta,
                mat_norm=MatrixNormalizer.load(Path(ckpt) / f"norm_matrix_{sample.name}.json"),
                track_norm=TrackNormalizer.load(Path(ckpt) / "norm_tracks.json"),
                track_files=sample_track_files(cfg, sample),
                src=open_cool(sample.cool_path))


def predict_map(m, chrom, reg_start, n_out, out_res, device, tta=True):
    cfg = m["cfg"]; size, res = cfg.window.size, cfg.data.resolution
    name = match_name(chrom, list(m["src"].chromnames))
    nb = chrom_nbins_of(m["src"], name)
    sym = m["track_norm"].apply(assemble_chrom(cfg, m["sample"], m["track_files"], name, nb))
    if m["sel_rows"] is not None:
        sym = sym[m["sel_rows"]]
    centre_bin = (reg_start + n_out * out_res // 2) // res
    st = max(0, min(nb - size, centre_bin - size // 2))
    ct = sym[:, st:st + size]
    if ct.shape[1] < size:
        ct = np.concatenate([ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
    cond = torch.from_numpy(ct[None]).to(device)
    pairs = (cond[:, m["offset"]:m["offset"] + 2 * m["n_motifs"]].reshape(1, m["n_motifs"], 2, size)
             if m["n_motifs"] else None)
    seq = sc.region_onehot(m["fasta"], name, st, size, res, device) if m["use_seq"] else None
    with torch.no_grad():
        out = m["model"](cond, pairs, None, seq)
        if tta:
            from hicmultiome.dataset import reverse_cond, split_cond
            cr = reverse_cond(cond, m["offset"], m["n_motifs"])
            sr, pr = split_cond(cr, m["offset"], m["n_motifs"])
            seqr = sc.region_onehot(m["fasta"], name, st, size, res, device, revcomp=True) if m["use_seq"] else None
            out = 0.5 * (out + torch.flip(m["model"](sr, pr, None, seqr), dims=[-1, -2]))
    P = m["mat_norm"].decode(out[0, 0].float().cpu().numpy())
    P = 0.5 * (P + P.T)
    return sc.resample(P, st * res, res, reg_start, n_out, out_res)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/processed/wolffjoa/hic-multiome/runs/three_models.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(NPZ, allow_pickle=True)
    side, off = int(z["side"]), int(z["offset"])
    iu = np.triu_indices(side, k=off)
    regions = list(z["regions"])
    ridx = {r: i for i, r in enumerate(regions)}

    def recon(vec):
        M = np.full((side, side), np.nan, np.float32)
        M[iu] = vec
        return np.where(np.isnan(M), M.T, M)

    def split(upper_full, lower_full):
        ut = np.triu(np.ones((side, side), bool), 0)
        return np.where(ut, upper_full, lower_full)

    # rank regions by the all-tracks model's per-region Pearson (in the npz set)
    nat = pd.read_csv(NATIVE_CSV).dropna(subset=["ours"])
    nat = nat[nat["region"].isin(ridx)].sort_values("ours", ascending=False).reset_index(drop=True)
    picks = {"good": nat.iloc[0]["region"],
             "medium": nat.iloc[len(nat) // 2]["region"],
             "bad": nat.iloc[-1]["region"]}
    print("picked regions:", picks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ours = [load_our_model(c, k, device) for _, c, k in MODELS]

    cols = ["good", "medium", "bad"]
    row_labels = ["Akita"] + [lbl for lbl, _, _ in MODELS]
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    kw = dict(cmap="RdBu_r", vmin=-2, vmax=2)
    for cj, band in enumerate(cols):
        reg = picks[band]
        gi = ridx[reg]
        tvec = z["akita_target"][gi, :, CELL]
        Mt = recon(tvec)
        chrom, span = reg.split(":")
        start = int(span.split("-")[0])
        # akita output covers the central `side` bins of the region
        full_bins = (int(span.split("-")[1]) - start) // 2048
        reg_start = start + ((full_bins - side) // 2) * 2048
        # predictions per model
        preds = [recon(z["akita_pred"][gi, :, CELL])]
        for m in ours:
            preds.append(predict_map(m, chrom, reg_start, side, 2048, device))
        for ri, Mp in enumerate(preds):
            ax = axes[ri, cj]
            ax.imshow(split(Mt, Mp), **kw)
            fin = np.isfinite(tvec) & np.isfinite(Mp[iu])
            r = (pearsonr(Mp[iu][fin], tvec[fin])[0]
                 if fin.sum() > 100 and tvec[fin].std() > 0 and Mp[iu][fin].std() > 0
                 else float("nan"))
            ax.set_title(f"r={r:.2f}", fontsize=9)
            if cj == 0:
                ax.set_ylabel(row_labels[ri], fontsize=9, fontweight="bold")
            if ri == 0:
                ax.text(0.5, 1.18, f"{band}\n{reg}", transform=ax.transAxes,
                        ha="center", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("GM12878 2 kb, log(obs/exp). Upper = original Hi-C, "
                 "lower = model prediction", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
