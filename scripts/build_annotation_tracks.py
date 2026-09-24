"""Cell-invariant genome-annotation tracks as bigWigs, one value per bin.

Histone marks have no binding motif, but their placement has sequence /
annotation priors: CpG islands (CFP1 -> H3K4me3/me2, Polycomb -> H3K27me3),
TSSs (H3K4me3), transcribed gene bodies and exons (H3K36me3, H3K79me2,
H4K20me1), and repeat classes (KRAB-ZNF silencing -> H3K9me3). Each channel
is written at the model's bin size so the per-bin mean read by
hicmultiome.tracks equals the stored value:

    ANN_CGI        fraction of the bin covered by CpG islands (cpgIslandExt)
    ANN_TSS        number of distinct transcript TSSs in the bin (knownGene)
    ANN_GeneBody   fraction covered by any transcript span (knownGene)
    ANN_Exon       fraction covered by any exon (knownGene)
    ANN_LTR / ANN_LINE / ANN_SINE / ANN_Satellite
                   fraction covered by that RepeatMasker class (rmsk)

Inputs are the UCSC hg38 tables (*.txt.gz) in --annot-dir.

    python scripts/build_annotation_tracks.py \
        --annot-dir ~/data/hic-multiome/annotation/hg38 \
        --fai ~/data/hic-multiome/genome/hg38.fa.fai --binsize 2000 \
        --out-dir ~/data/hic-multiome/annotation/hg38/bw_2kb
"""
from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyBigWig

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
RMSK_CLASSES = {"LTR": "ANN_LTR", "LINE": "ANN_LINE", "SINE": "ANN_SINE",
                "Satellite": "ANN_Satellite"}


def read_fai(path):
    return {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(path)}


def frac_cover(intervals, length, binsize):
    """Per-bin covered fraction of the union of [s, e) intervals."""
    diff = np.zeros(length + 1, dtype=np.int32)
    if intervals:
        iv = np.asarray(intervals, dtype=np.int64)
        s, e = np.clip(iv[:, 0], 0, length), np.clip(iv[:, 1], 0, length)
        np.add.at(diff, s, 1)
        np.add.at(diff, e, -1)
    cov = (np.cumsum(diff[:-1]) > 0).astype(np.float32)
    nb = -(-length // binsize)
    pad = np.zeros(nb * binsize, dtype=np.float32)
    pad[:length] = cov
    return pad.reshape(nb, binsize).mean(1)


def count_points(points, length, binsize):
    nb = -(-length // binsize)
    out = np.zeros(nb, dtype=np.float32)
    if points:
        p = np.clip(np.asarray(sorted(set(points)), dtype=np.int64), 0, length - 1)
        np.add.at(out, p // binsize, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annot-dir", required=True)
    ap.add_argument("--fai", required=True)
    ap.add_argument("--binsize", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    d = Path(a.annot_dir).expanduser()
    out = Path(a.out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    sizes = read_fai(Path(a.fai).expanduser())
    chroms = [c for c in CHROMS if c in sizes]

    iv = defaultdict(lambda: defaultdict(list))     # channel -> chrom -> [(s, e)]
    tss = defaultdict(list)
    with gzip.open(d / "cpgIslandExt.txt.gz", "rt") as f:
        for l in f:
            x = l.split("\t")
            iv["ANN_CGI"][x[1]].append((int(x[2]), int(x[3])))
    with gzip.open(d / "knownGene.txt.gz", "rt") as f:
        for l in f:
            x = l.split("\t")
            c, strand, s, e = x[1], x[2], int(x[3]), int(x[4])
            iv["ANN_GeneBody"][c].append((s, e))
            tss[c].append(s if strand == "+" else e - 1)
            for es, ee in zip(x[8].rstrip(",").split(","), x[9].rstrip(",").split(",")):
                iv["ANN_Exon"][c].append((int(es), int(ee)))
    with gzip.open(d / "rmsk.txt.gz", "rt") as f:
        for l in f:
            x = l.split("\t")
            ch = RMSK_CLASSES.get(x[11])
            if ch:
                iv[ch][x[5]].append((int(x[6]), int(x[7])))

    channels = ["ANN_CGI", "ANN_TSS", "ANN_GeneBody", "ANN_Exon"] + list(RMSK_CLASSES.values())
    for ch in channels:
        bw = pyBigWig.open(str(out / f"{ch}.bigwig"), "w")
        bw.addHeader([(c, sizes[c]) for c in chroms])
        tot = 0.0
        for c in chroms:
            L = sizes[c]
            v = (count_points(tss[c], L, a.binsize) if ch == "ANN_TSS"
                 else frac_cover(iv[ch][c], L, a.binsize))
            starts = np.arange(len(v), dtype=np.int64) * a.binsize
            ends = np.minimum(starts + a.binsize, L)
            bw.addEntries([c] * len(v), starts.tolist(), ends=ends.tolist(),
                          values=v.astype(float).tolist())
            tot += float(v.mean()) * len(v)
        bw.close()
        print(f"{ch}: mean per bin {tot / sum(-(-sizes[c] // a.binsize) for c in chroms):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
