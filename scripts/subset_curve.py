"""Accuracy vs number of tracks, on a track-dropout-trained model.

For a model trained with track dropout, predict from a shrinking set of tracks by
filling every non-kept slot with its absent value, and score on Akita's test
target (Hi-cGAN metric). Produces the accuracy-vs-#tracks curve that answers
"trained on 13, predict on 2 -- does it still work, and how well?".

    python scripts/subset_curve.py --config configs/akita_native_trackdrop.yaml \
        --checkpoint runs/akita_native_trackdrop/checkpoint-best --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

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

# correlation-informed keep-sets, shrinking; one representative per group as it shrinks
KEEP_SETS = {
    "all_13":  ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
                "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"],
    "keep_8":  ["CTCF", "RAD21", "H3K27ac", "H3K27me3", "H3K36me3", "H3K79me2",
                "H3K9me3", "H4K20me1"],
    "keep_5":  ["CTCF", "RAD21", "H3K27ac", "H3K27me3", "H3K36me3"],
    "keep_3":  ["CTCF", "RAD21", "H3K27ac"],
    "keep_2":  ["CTCF", "H3K27ac"],
    "keep_1":  ["CTCF"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--minCovered", type=float, default=0.5)
    ap.add_argument("--out-dir",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/subset_curve")
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = a.device
    cfg = Config.load(a.config)
    sample = cfg.get_sample(None)
    ckpt = Path(a.checkpoint)
    model, channels = sc.load_model(cfg, ckpt, device)
    size, res = cfg.window.size, cfg.data.resolution
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    n_chip = len(channels["tracks"])
    row_of = {t: i for i, t in enumerate(channels["tracks"])}
    mat_norm = MatrixNormalizer.load(ckpt / f"norm_matrix_{sample.name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    absent = track_norm.absent_values()
    use_seq = cfg.sequence.uses_encoder
    fasta = sample.fasta or cfg.sequence.fasta

    regions = pd.read_csv(sc.REGIONS, sep="\t", names=["chrom", "start", "end"]).reset_index(drop=True)
    tgt = cooler.Cooler(str(sc.TARGET))
    if not Path(sample.cool_path).exists():
        sample.cool_path = str(sc.TARGET)
    src = open_cool(sample.cool_path)
    track_files = sample_track_files(cfg, sample)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    keep_masks = {name: [row_of[t] for t in ts] for name, ts in KEEP_SETS.items()}
    per_region = {}
    for chrom, grp in regions.groupby("chrom"):
        if match_name(chrom, list(tgt.chromnames)) is None:
            continue
        try:
            name = match_name(chrom, list(src.chromnames))
        except Exception:
            continue
        nb = chrom_nbins_of(src, name)
        sym_full = track_norm.apply(assemble_chrom(cfg, sample, track_files, name, nb))
        print(f"{chrom}: {len(grp)} regions", flush=True)

        for _, r in grp.iterrows():
            tname = match_name(chrom, list(tgt.chromnames))
            try:
                T = np.asarray(tgt.matrix(balance=False).fetch(
                    f"{tname}:{int(r.start)}-{int(r.end)}"), dtype=float)
            except Exception:
                continue
            n_out = T.shape[0]
            centre_bin = (int(r.start) + int(r.end)) // 2 // res
            st = max(0, min(nb - size, centre_bin - size // 2))
            base_ct = sym_full[:, st:st + size]
            if base_ct.shape[1] < size:
                base_ct = np.concatenate(
                    [base_ct, np.zeros((base_ct.shape[0], size - base_ct.shape[1]), np.float32)], 1)
            reg = f"{chrom}:{int(r.start)}-{int(r.end)}"
            per_region.setdefault(reg, {})
            seq_oh = (sc.region_onehot(fasta, name, st, size, res, device)
                      if use_seq else None)

            for kname, keep_rows in keep_masks.items():
                ct = base_ct.copy()
                mask = np.ones(n_chip, bool)
                mask[keep_rows] = False           # True => drop to absent
                ct[:n_chip][mask] = absent[:n_chip][mask][:, None]
                cond = torch.from_numpy(ct[None]).to(device)
                pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
                         if n_motifs else None)
                with torch.no_grad():
                    o = model(cond, pairs, None, seq_oh)
                    if a.tta:
                        from hicmultiome.dataset import reverse_cond, split_cond
                        cr = reverse_cond(cond, offset, n_motifs)
                        sr, pr = split_cond(cr, offset, n_motifs)
                        seqr = (sc.region_onehot(fasta, name, st, size, res, device, revcomp=True)
                                if use_seq else None)
                        o = 0.5 * (o + torch.flip(model(sr, pr, None, seqr), dims=[-1, -2]))
                P = mat_norm.decode(o[0, 0].float().cpu().numpy())
                P = 0.5 * (P + P.T)
                P = sc.resample(P, st * res, res, int(r.start), n_out, tgt.binsize)
                live = np.abs(P).sum(1) > 0
                if live.mean() < a.minCovered:
                    continue
                M = np.where(live[:, None] & live[None, :], P, np.nan)
                Tm = np.where(T == 0, np.nan, T)
                ok = np.isfinite(M) & np.isfinite(Tm)
                if ok.sum() >= 100 and M[ok].std() > 0 and Tm[ok].std() > 0:
                    per_region[reg][kname] = float(pearsonr(M[ok], Tm[ok]).statistic)

        df = pd.DataFrame(per_region).T
        df.to_csv(out / "per_region.csv")
        _summary(df, out)

    df = pd.DataFrame(per_region).T
    df.to_csv(out / "per_region.csv")
    _summary(df, out, final=True)


def _summary(df, out, final=False):
    rows = []
    for k, ts in KEEP_SETS.items():
        if k not in df:
            continue
        col = df[k].dropna()
        rows.append({"keep_set": k, "n_tracks": len(ts), "n_regions": len(col),
                     "median_r": round(col.median(), 4), "mean_r": round(col.mean(), 4)})
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    if final:
        print("\n=== accuracy vs #tracks (median per-region Pearson, Akita test) ===", flush=True)
        print(s.to_string(index=False), flush=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 5))
            plt.plot(s["n_tracks"], s["median_r"], "o-")
            for _, r in s.iterrows():
                plt.annotate(r["keep_set"], (r["n_tracks"], r["median_r"]), fontsize=7)
            plt.xlabel("number of tracks given at prediction")
            plt.ylabel("median per-region Pearson (Akita test)")
            plt.title("Track-dropout model: accuracy vs #tracks")
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(out / "subset_curve.png", dpi=150)
            print("wrote", out / "subset_curve.png", flush=True)
        except Exception as e:
            print("plot failed:", e)


if __name__ == "__main__":
    main()
