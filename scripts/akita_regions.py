"""Akita's training regions as a BED, and how many of our windows they remove.

Akita (Fudenberg et al. 2020) publishes every sequence it used, with its fold,
in the basenji data release (`sequences.bed`, 7008 train, 419 valid, 413 test
sequences of 1,048,576 bp at a 262,144 bp stride). Its filters (assembly gaps,
blacklist, Umap unmappable fraction, fraction of missing target pixels) are
already applied to that list, so the genome no training sequence covers is
exactly what Akita left out of training. The targets are the central
448 x 2048 bp of each sequence (32 bins cropped per side), which is also how
`akita_holdout_valid_test.bed` was built from the valid and test sequences.

This writes the merged target spans of the training sequences (for
`data.train_regions_bed`) and counts, on the window grid of a config, how many
training windows (outside the held-out regions) are not wholly inside them.

    python scripts/akita_regions.py --out ~/data/hic-multiome/akita/akita_train_regions.bed \
        --config configs/full2k_motifs.yaml
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome.config import Config                                  # noqa: E402
from hicmultiome.genome import (merge_regions, read_bed_regions,
                                resolve_chroms, window_region_state)   # noqa: E402

URL = "https://storage.googleapis.com/basenji_hic/1m/data/sequences.bed"
CROP_BP = 32 * 2048


def load_sequences(path: Path) -> list[tuple[str, int, int, str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    out = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) >= 4:
            out.append((f[0], int(f[1]), int(f[2]), f[3]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", default=str(Path.home() / "data/hic-multiome/akita/sequences.bed"),
                    help=f"Akita sequences.bed (downloaded from {URL} if absent)")
    ap.add_argument("--out", required=True, help="merged training target spans (BED)")
    ap.add_argument("--crop-bp", type=int, default=CROP_BP,
                    help="bp cropped from each sequence end to get the target span")
    ap.add_argument("--folds", default="train",
                    help="comma-separated folds whose sequences form the regions; "
                         "'train' = Akita's training set (default), "
                         "'train,valid,test' = every sequence Akita kept, i.e. only "
                         "the filtered-out genome is removed, not the gaps Akita "
                         "left between folds")
    ap.add_argument("--config", default=None,
                    help="count this config's training windows removed by the BED")
    a = ap.parse_args()

    seqs = load_sequences(Path(a.sequences).expanduser())
    folds = {}
    for *_, f in seqs:
        folds[f] = folds.get(f, 0) + 1
    want = set(a.folds.split(","))
    tr = {}
    for c, s, e, f in seqs:
        if f in want and e - s > 2 * a.crop_bp:
            tr.setdefault(c, []).append((s + a.crop_bp, e - a.crop_bp))
    merged = merge_regions(tr)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = sum(len(v) for v in merged.values())
    mb = sum(e - s for v in merged.values() for s, e in v) / 1e6
    with out.open("w") as fh:
        fh.write(f"# Akita {a.folds} target spans (sequence +/- {a.crop_bp} bp), merged\n")
        fh.write(f"# source: {URL}; folds {folds}\n")
        for c in sorted(merged, key=lambda x: (len(x), x)):
            for s, e in merged[c]:
                fh.write(f"{c}\t{s}\t{e}\t{a.folds}\n")
    print(f"folds: {folds}")
    print(f"wrote {out}: {n} merged regions, {mb:.1f} Mb")

    if a.config:
        import cooler

        from hicmultiome.matrix import open_cool, plan_windows

        cfg = Config.load(a.config)
        hold = read_bed_regions(cfg.data.holdout_bed) if cfg.data.holdout_bed else None
        clr = open_cool(cfg.data.samples[0].cool_path)
        res, size, stride = cfg.data.resolution, cfg.window.size, cfg.window.stride
        tot = kept = 0
        per = []
        for chrom in resolve_chroms(cfg.data.chroms, list(clr.chromnames)):
            lo, hi = clr.extent(chrom)
            n_all = n_keep = 0
            for w in plan_windows(int(hi - lo), size, stride, chrom):
                s_bp, e_bp = w.start * res, (w.start + w.size) * res
                if hold is not None and window_region_state(hold, chrom, s_bp, e_bp) != "outside":
                    continue
                n_all += 1
                n_keep += window_region_state(merged, chrom, s_bp, e_bp) == "inside"
            per.append((chrom, n_all, n_keep))
            tot += n_all
            kept += n_keep
        del cooler
        for c, n_all, n_keep in per:
            print(f"  {c:6s} {n_all:5d} grid windows outside holdout, {n_all - n_keep:4d} removed")
        print(f"grid windows outside holdout: {tot}; inside Akita training regions: "
              f"{kept}; removed: {tot - kept} ({100 * (tot - kept) / max(tot, 1):.1f}%) "
              "(before the empty-window filter of precompute)")


if __name__ == "__main__":
    main()
