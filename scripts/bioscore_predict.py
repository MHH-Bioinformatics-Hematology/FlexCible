"""Cache our prediction for every region of the Akita test rasters.

Nothing about Akita or Hi-cGAN is recomputed here: their maps are read from
akita_testset_rasters.npz. This script only runs OUR model, once per region,
and stores

  ours_native  (N, 512, 512)  our own 2000 bp grid, the model's native output
  win_start    (N,)           first bin of that window, in 2000 bp bins
  chrom/start/end              the Akita region, hg38 bp

so that everything downstream (Akita-grid scoring, the 2000 bp down-sampling
experiment) can resample from one cached forward pass instead of re-running
the network. The raw (un-normalised) mean of every input track over the
region is written to the same file -- that is the only thing Part 2's
stratification is allowed to use, and it must never see a model score.

CPU only by construction: CUDA_VISIBLE_DEVICES is cleared before torch loads.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config                               # noqa: E402
from flexcible.genome import match_name                           # noqa: E402
from flexcible.matrix import MatrixNormalizer, open_cool          # noqa: E402
from flexcible.model import build_spatial_cond                    # noqa: E402
from flexcible.precompute import (assemble_chrom, chrom_nbins_of,
                                    sample_track_files)             # noqa: E402
from flexcible.tracks import TrackNormalizer                      # noqa: E402
from flexcible.utils import get_logger                            # noqa: E402
from scripts.score_like_hicgan import load_model                    # noqa: E402

RAST = ("/mnt/processed/wolffjoa/restart_of_prediction_project/"
        "akita_normalization/akita_testset_rasters.npz")
log = get_logger("bioscore-predict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/akita_native.yaml")
    ap.add_argument("--checkpoint", default="runs/akita_native/checkpoint-best")
    ap.add_argument("--rasters", default=RAST)
    ap.add_argument("--out", default="runs/bioscore/our_preds.npz")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    z = np.load(a.rasters, allow_pickle=True)
    regions = [str(r) for r in z["regions"]]
    if a.limit:
        regions = regions[: a.limit]
    log.info("%d regions", len(regions))

    cfg = Config.load(a.config)
    sample = cfg.get_sample(None)
    ckpt = Path(a.checkpoint)
    model, channels = load_model(cfg, ckpt, "cpu")
    size, res = cfg.window.size, cfg.data.resolution
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    mat_norm = MatrixNormalizer.load(ckpt / f"norm_matrix_{sample.name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    track_names = list(cfg.track_names)

    clr = open_cool(sample.cool_path)
    track_files = sample_track_files(cfg, sample)

    parsed = []
    for r in regions:
        c, span = r.split(":")
        s, e = (int(x) for x in span.split("-"))
        parsed.append((r, c, s, e))

    N = len(parsed)
    native = np.zeros((N, size, size), np.float32)
    win_start = np.zeros(N, np.int64)
    tmeans = np.full((N, len(track_names)), np.nan, np.float64)
    order = {r: i for i, (r, *_ ) in enumerate(parsed)}

    by_chrom: dict[str, list] = {}
    for rec in parsed:
        by_chrom.setdefault(rec[1], []).append(rec)

    for chrom in sorted(by_chrom):
        name = match_name(chrom, list(clr.chromnames))
        nb = chrom_nbins_of(clr, name)
        raw = assemble_chrom(cfg, sample, track_files, name, nb)
        sym = track_norm.apply(raw)
        for (rname, _c, s, e) in by_chrom[chrom]:
            k = order[rname]
            centre = (s + e) // 2 // res
            st = max(0, min(nb - size, centre - size // 2))
            ct = sym[:, st:st + size]
            if ct.shape[1] < size:
                ct = np.concatenate(
                    [ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
            cond = torch.from_numpy(ct[None])
            pairs = (cond[:, offset:offset + 2 * n_motifs]
                     .reshape(1, n_motifs, 2, size) if n_motifs else None)
            with torch.no_grad():
                out = model(build_spatial_cond(cond, pairs, None,
                                               cfg.model.cond_mode))
            P = mat_norm.decode(out[0, 0].float().numpy())
            native[k] = 0.5 * (P + P.T)
            win_start[k] = st
            lo, hi = s // res, e // res
            seg = raw[:len(track_names), lo:min(hi, nb)]
            if seg.size:
                tmeans[k] = np.nanmean(seg, axis=1)
        log.info("%s: %d regions", chrom, len(by_chrom[chrom]))
        del raw, sym

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             regions=np.array([p[0] for p in parsed], dtype=object),
             chrom=np.array([p[1] for p in parsed], dtype=object),
             start=np.array([p[2] for p in parsed], np.int64),
             end=np.array([p[3] for p in parsed], np.int64),
             ours_native=native, win_start=win_start,
             resolution=np.int64(res), window=np.int64(size),
             track_names=np.array(track_names, dtype=object),
             track_means=tmeans)
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
