"""Score our 2 kb model exactly the way Hi-cGAN and Akita were scored.

The recorded akita_gm12878 / hicgan_fixed columns were produced by
Hi-cGAN/manuscript_gigascience/scripts/score_akita_regions.py against
akita_published_GM12878_test_2048bp.cool -- Akita's OWN targets, in which
mean(exp(v)) sits at 1.0 at every genomic distance. Scoring against
gm12878_2kb_akita_normalized.cool instead is not the same measurement: that
file keeps a residual distance dependence, and on it a distance-only
prediction already scores ~0.74, so any model looks good.

The metric is theirs, verbatim:
  - regions from sequences_test_list.tsv
  - rows the generator never reached are excluded (covered >= minCovered)
  - zero pixels in the target are missing, not values
  - space "akita": neither side is transformed again
  - Pearson over all finite pixels of the region (not just the triangle)

One unavoidable difference, stated rather than hidden: Akita's grid is 2048 bp
and ours is 2000 bp, so our prediction is resampled onto their bin centres by
bilinear interpolation before scoring. Comparing 512x512 blocks directly
without that would misalign the two grids by up to 24 kb at the region edge.
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
from scipy.ndimage import map_coordinates
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome.config import Config                                # noqa: E402
from hicmultiome.genome import match_name                            # noqa: E402
from hicmultiome.matrix import MatrixNormalizer, open_cool           # noqa: E402
from hicmultiome.model import (build_backbone, build_spatial_cond,
                               n_cond_channels)                      # noqa: E402
from hicmultiome.precompute import (assemble_chrom, chrom_nbins_of,
                                    sample_track_files)              # noqa: E402
from hicmultiome.tracks import TrackNormalizer                       # noqa: E402
from hicmultiome.utils import get_logger                             # noqa: E402

AK = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization")
REGIONS = AK / "akita_validation/sequences_test_list.tsv"
TARGET = AK / "akita_published_GM12878_test_2048bp.cool"
TABLE = Path.home() / "src/Hi-cGAN/manuscript_gigascience/data/akita_vs_hicgan_heldout.csv"
log = get_logger("score-like-hicgan")


def load_model(cfg, ckpt, device):
    channels = json.loads((ckpt / "channels.json").read_text())
    in_ch = n_cond_channels(len(channels["sym_rows"]), len(channels["motif_ids"]),
                            2 if channels.get("capture_input") else 0,
                            cfg.model.cond_mode)
    model = build_backbone(cfg.model.kind, in_ch,
                           out_ch=2 if channels.get("has_capture") else 1,
                           base=cfg.model.base_channels,
                           depth=cfg.model.depth).to(device)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
    model.eval()
    return model, channels


def resample(pred: np.ndarray, our_start_bp: int, our_res: int,
             reg_start: int, n_out: int, out_res: int) -> np.ndarray:
    """Our 2000 bp prediction sampled at Akita's 2048 bp bin centres."""
    centres = reg_start + (np.arange(n_out) + 0.5) * out_res
    coord = (centres - our_start_bp) / our_res - 0.5
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    return map_coordinates(pred, [yy.ravel(), xx.ravel()], order=1,
                           mode="nearest").reshape(n_out, n_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--minCovered", type=float, default=0.5)
    ap.add_argument("--even-only", action="store_true",
                    help="restrict to even chromosomes, our held-out set")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    sample = cfg.get_sample(None)
    ckpt = Path(a.checkpoint or (Path(cfg.train.output_dir) / "checkpoint-best"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, channels = load_model(cfg, ckpt, device)
    size, res = cfg.window.size, cfg.data.resolution
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    mat_norm = MatrixNormalizer.load(ckpt / f"norm_matrix_{sample.name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")

    regions = pd.read_csv(REGIONS, sep="\t", names=["chrom", "start", "end"])
    if a.even_only:
        num = regions.chrom.str.replace("chr", "", regex=False)
        regions = regions[num.str.isdigit()]
        regions = regions[num[num.str.isdigit()].astype(int) % 2 == 0]
    regions = regions.reset_index(drop=True)
    log.info("%d Akita test regions; target %s", len(regions), TARGET.name)

    tgt = cooler.Cooler(str(TARGET))
    src = open_cool(sample.cool_path)          # only for chromosome sizes
    track_files = sample_track_files(cfg, sample)
    rows = []

    for chrom, grp in regions.groupby("chrom"):
        if match_name(chrom, list(tgt.chromnames)) is None:
            continue
        try:
            name = match_name(chrom, list(src.chromnames))
        except Exception:
            continue
        nb = chrom_nbins_of(src, name)
        sym_full = track_norm.apply(
            assemble_chrom(cfg, sample, track_files, name, nb))

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
            ct = sym_full[:, st:st + size]
            if ct.shape[1] < size:
                ct = np.concatenate(
                    [ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
            cond = torch.from_numpy(ct[None]).to(device)
            pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
                     if n_motifs else None)
            with torch.no_grad():
                out = model(build_spatial_cond(cond, pairs, None, cfg.model.cond_mode))
                if a.tta:
                    from hicmultiome.dataset import reverse_cond, split_cond
                    cr = reverse_cond(cond, offset, n_motifs)
                    sr, pr = split_cond(cr, offset, n_motifs)
                    out_r = model(build_spatial_cond(sr, pr, None, cfg.model.cond_mode))
                    out = 0.5 * (out + torch.flip(out_r, dims=[-1, -2]))
            P = mat_norm.decode(out[0, 0].float().cpu().numpy())
            P = 0.5 * (P + P.T)
            P = resample(P, st * res, res, int(r.start), n_out, tgt.binsize)

            # --- their metric, verbatim ---
            live = np.abs(P).sum(axis=1) > 0
            covered = float(live.mean())
            rec = {"region": f"{chrom}:{int(r.start)}-{int(r.end)}",
                   "covered": covered}
            if covered >= a.minCovered:
                M = np.where(live[:, None] & live[None, :], P, np.nan)
                Tm = np.where(T == 0, np.nan, T)
                ok = np.isfinite(M) & np.isfinite(Tm)
                if ok.sum() >= 100 and M[ok].std() > 0 and Tm[ok].std() > 0:
                    rec["ours"] = float(pearsonr(M[ok], Tm[ok]).statistic)
            rows.append(rec)
        log.info("%s: %d regions", chrom, len(grp))

    df = pd.DataFrame(rows)
    rec_tab = pd.read_csv(TABLE)
    df = df.merge(rec_tab[["region", "akita_gm12878", "hicgan_fixed"]],
                  on="region", how="left")
    out = a.out or str(Path(cfg.train.output_dir) / "score_like_hicgan.csv")
    df.to_csv(out, index=False)

    ok = df.dropna(subset=["ours"])
    both = ok.dropna(subset=["akita_gm12878"])
    print(f"\nscored {len(ok)} of {len(df)} regions "
          f"(Akita's own target, their metric)")
    print(f"  Akita   (recorded): median {both.akita_gm12878.median():.3f}")
    hg = both.hicgan_fixed.dropna()
    print(f"  Hi-cGAN (recorded): median {hg.median():.3f}")
    print(f"  ours              : median {ok.ours.median():.3f}  "
          f"mean {ok.ours.mean():.3f}")
    if len(both):
        print(f"  ours beats Akita in "
              f"{100*(both.ours > both.akita_gm12878).mean():.0f}% of "
              f"{len(both)} paired regions")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
