"""Render a region with pyGenomeTracks: CTCF track + true vs predicted Hi-C.

Builds a tracks.ini on the fly with three panels for the requested region:
  1. the conditioning track (e.g. CTCF), binned to the matrix resolution,
  2. the ground-truth Hi-C matrix (cool), and
  3. the predicted Hi-C matrix (cool),
then calls pyGenomeTracks. Handles the cool('2') vs bigwig('chr2') naming
mismatch by resolving each file's convention and writing the track as a bedgraph
with the cool's chromosome name.

CLI:  python -m flexcible.plot --config configs/ctcf_256.yaml \
          --pred runs/ctcf_256/pred_even.cool \
          --region chr2:18000000-25000000 --out runs/ctcf_256/chr2_18-25Mb.png
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from .genome import match_name
from .tracks import discover_tracks


def parse_region(region: str) -> tuple[str, int, int]:
    region = region.replace(",", "").strip()
    chrom, span = region.split(":")
    start, end = span.split("-")
    return chrom, int(start), int(end)


def write_track_bedgraph(bw_path: str, chrom_out: str, chrom_bw: str,
                         start: int, end: int, binsize: int, out_path: Path) -> None:
    """Bin a bigwig over [start,end) to `binsize` and write a bedgraph using the
    cool's chromosome name (`chrom_out`)."""
    import pyBigWig

    bw = pyBigWig.open(bw_path)
    try:
        n = max(1, (end - start) // binsize)
        vals = bw.stats(chrom_bw, start, start + n * binsize, type="mean", nBins=n)
    finally:
        bw.close()
    with open(out_path, "w") as f:
        for i, v in enumerate(vals):
            s = start + i * binsize
            f.write(f"{chrom_out}\t{s}\t{s + binsize}\t{0.0 if v is None else v}\n")


def build_ini(track_name, track_bedgraph, truth_cool, pred_cool, depth,
              ini_path: Path) -> None:
    ini = f"""
[ctcf]
file = {track_bedgraph}
title = {track_name}
height = 2.5
color = #2c7fb8
file_type = bedgraph

[spacer]
height = 0.3

[truth]
file = {truth_cool}
title = Ground truth Hi-C
depth = {depth}
transform = log1p
file_type = hic_matrix
colormap = RdYlBu_r
show_masked_bins = false

[spacer]
height = 0.3

[pred]
file = {pred_cool}
title = Predicted Hi-C
depth = {depth}
transform = log1p
file_type = hic_matrix
colormap = RdYlBu_r
show_masked_bins = false

[x-axis]
where = bottom
"""
    ini_path.write_text(ini)


def plot(cfg: Config, pred_cool: str, region: str, out_path: str,
         track_name: str | None = None, depth: int | None = None) -> str:
    import cooler

    chrom, start, end = parse_region(region)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # resolve cool chromosome naming ('2'); predicted + truth share the bin table
    smp = cfg.get_sample(None)
    clr = cooler.Cooler(smp.cool_path)
    chrom_cool = match_name(chrom, list(clr.chromnames))
    if chrom_cool is None:
        raise ValueError(f"{chrom} not in cool (have {list(clr.chromnames)[:5]})")

    # pick the conditioning track (default: first configured, e.g. CTCF)
    from .precompute import sample_track_files as _stf
    track_files = [(n, p) for n, p in _stf(cfg, smp) if p]
    name_to_path = dict(track_files)
    track_name = track_name or track_files[0][0]
    bw_path = name_to_path[track_name]
    import pyBigWig

    bw = pyBigWig.open(bw_path)
    chrom_bw = match_name(chrom, list(bw.chroms().keys()))
    bw.close()

    bedgraph = out.with_suffix(".track.bedgraph")
    write_track_bedgraph(bw_path, chrom_cool, chrom_bw, start, end,
                         cfg.data.resolution, bedgraph)

    if depth is None:
        # focus the triangle on the reliably-covered band (size - stride bins),
        # capped by the region length
        reliable = (cfg.window.size - cfg.window.stride) * cfg.data.resolution
        depth = int(min(reliable, (end - start) // 2))

    ini = out.with_suffix(".tracks.ini")
    build_ini(track_name, bedgraph, smp.cool_path, pred_cool, depth, ini)

    region_str = f"{chrom_cool}:{start}-{end}"
    cmd = ["pyGenomeTracks", "--tracks", str(ini), "--region", region_str,
           "--dpi", "150", "-o", str(out)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Plot a region: track + true/pred Hi-C.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred", required=True, help="predicted .cool")
    ap.add_argument("--region", required=True, help="e.g. chr2:18000000-25000000")
    ap.add_argument("--out", required=True, help="output .png")
    ap.add_argument("--track", default=None, help="track name (default: first in config)")
    ap.add_argument("--depth", type=int, default=None, help="Hi-C triangle depth in bp")
    args = ap.parse_args()
    plot(Config.load(args.config), args.pred, args.region, args.out, args.track, args.depth)


if __name__ == "__main__":
    main()
