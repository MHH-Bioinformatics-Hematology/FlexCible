"""Toy data set for the FlexCible quickstart: two cells on a 3-chromosome synthetic genome.

Writes, into the folder given as the first argument (default ./toy):
  genome.fa            three chromosomes of 120, 100 and 90 kb of random sequence with planted CTCF-like
                       motif sites in convergent orientation (forward sites, then reverse sites)
  MA9999.1.jaspar      the position frequency matrix of that planted motif
  cellA/, cellB/       one folder per cell with CTCF.bigwig and H3K27ac.bigwig (1 kb bins, peaks at the
                       planted sites; cell B lacks two of the loops and their peaks)
  cellA.cool, cellB.cool
                       contact maps at 1 kb with a distance decay and loops between the motif pairs;
                       cell B without the two loops that it lacks
  consensus.cool       the mean of both maps after the log O/E transform of the pipeline, on the same bin
                       grid, usable as data.reference_cool (window.transform prescaled)
  quickstart.yaml      a configuration that trains a small model on chromosomes 1 and 3 (odd) and tests on
                       chromosome 2 (even), from the two tracks of cell A and B with the consensus as reference

The point is the plumbing, not accuracy: precompute, train, predict and evaluate run on a laptop CPU
in a few minutes. Real data replace every file here (see the tutorial).

    python examples/quickstart/make_toy_data.py toy
"""
from __future__ import annotations

import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import pyBigWig

BIN = 1000
CHROMS = {"chr1": 120 * BIN, "chr2": 100 * BIN, "chr3": 90 * BIN}
MOTIF = "MA9999.1"
SITE, SITE_RC = "ACGTACGTAC", "GTACGTACGT"
# loop anchors per chromosome as (forward bin, reverse bin); cell B lacks the loops marked False
LOOPS = {"chr1": [(5, 12, True), (20, 30, True), (40, 55, False), (70, 85, True)],
         "chr2": [(8, 18, True), (30, 45, False), (60, 75, True)],
         "chr3": [(6, 15, True), (25, 40, True), (55, 70, True)]}


def write_fasta(path: Path) -> None:
    rng = np.random.default_rng(0)
    with open(path, "w") as f:
        for chrom, ln in CHROMS.items():
            seq = rng.choice(list("ACGT"), size=ln)
            for fwd, rev, _ in LOOPS[chrom]:
                seq[fwd * BIN:fwd * BIN + 10] = list(SITE)
                seq[rev * BIN:rev * BIN + 10] = list(SITE_RC)
            s = "".join(seq)
            f.write(f">{chrom}\n" + "\n".join(s[i:i + 60] for i in range(0, ln, 60)) + "\n")


def write_jaspar(path: Path) -> None:
    counts = {b: [(90 if c == b else 3) for c in SITE] for b in "ACGT"}
    with open(path, "w") as f:
        f.write(f">{MOTIF}\tTOYCTCF\n")
        for b in "ACGT":
            f.write(f"{b} [ " + " ".join(str(x) for x in counts[b]) + " ]\n")


def write_bigwig(path: Path, peaks: dict[str, list[int]], height: float) -> None:
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(list(CHROMS.items()))
    rng = np.random.default_rng(len(path.name))
    for chrom, ln in CHROMS.items():
        nb = ln // BIN
        vals = 1.0 + 0.3 * rng.random(nb)
        for b in peaks.get(chrom, []):
            vals[b] = height
        bw.addEntries([chrom] * nb, [i * BIN for i in range(nb)], ends=[(i + 1) * BIN for i in range(nb)],
                      values=[float(v) for v in vals])
    bw.close()


def contact_matrices(in_cell_b: bool, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for chrom, ln in CHROMS.items():
        nb = ln // BIN
        i, j = np.indices((nb, nb))
        d = np.abs(i - j)
        m = 80.0 / (1.0 + d) ** 0.9 + 0.5 * rng.random((nb, nb))
        for fwd, rev, in_b in LOOPS[chrom]:
            if in_cell_b or in_b:
                # a domain between the anchors and a loop peak at the anchor pair
                m[fwd:rev + 1, fwd:rev + 1] *= 1.6
                m[fwd, rev] += 25.0
                m[rev, fwd] += 25.0
        out[chrom] = np.triu(m)
    return out


def write_cool(path: Path, mats: dict[str, np.ndarray], max_dist_bins: int = 60) -> None:
    bins, px, offset = [], [], 0
    for chrom, ln in CHROMS.items():
        nb = ln // BIN
        bins += [(chrom, s, min(s + BIN, ln)) for s in range(0, ln, BIN)]
        m = mats[chrom]
        for a in range(nb):
            for b in range(a, min(a + max_dist_bins, nb)):
                px.append((offset + a, offset + b, float(m[a, b])))
        offset += nb
    bins = pd.DataFrame(bins, columns=["chrom", "start", "end"])
    pixels = pd.DataFrame(px, columns=["bin1_id", "bin2_id", "count"])
    cooler.create_cooler(str(path), bins=bins, pixels=pixels, dtypes={"count": "float64"},
                         symmetric_upper=True, ordered=True)


def log_oe(mats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Distance-normalised log O/E of a dense upper-triangular count map (toy version of oe_proper.py)."""
    out = {}
    for chrom, m in mats.items():
        nb = m.shape[0]
        full = m + np.triu(m, 1).T
        i, j = np.indices((nb, nb))
        d = np.abs(i - j)
        exp = np.array([full[d == k].mean() for k in range(nb)])
        oe = np.log((full + 1e-6) / (exp[d] + 1e-6))
        out[chrom] = np.triu(np.clip(oe, -2, 2))
    return out


def main(root: str = "toy") -> None:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_fasta(root / "genome.fa")
    write_jaspar(root / f"{MOTIF}.jaspar")
    peaks_a = {c: [b for f, r, _ in L for b in (f, r)] for c, L in LOOPS.items()}
    peaks_b = {c: [b for f, r, in_b in L if in_b for b in (f, r)] for c, L in LOOPS.items()}
    for cell, peaks, h in (("cellA", peaks_a, 25.0), ("cellB", peaks_b, 22.0)):
        (root / cell).mkdir(exist_ok=True)
        write_bigwig(root / cell / "CTCF.bigwig", peaks, h)
        write_bigwig(root / cell / "H3K27ac.bigwig", {c: [b + 1 for b in v] for c, v in peaks.items()}, 12.0)
    mats_a = contact_matrices(in_cell_b=True, seed=1)      # every loop
    mats_b = contact_matrices(in_cell_b=False, seed=2)     # loops of cell B only
    write_cool(root / "cellA.cool", mats_a)
    write_cool(root / "cellB.cool", mats_b)
    oe_a, oe_b = log_oe(mats_a), log_oe(mats_b)
    write_cool(root / "consensus.cool", {c: (oe_a[c] + oe_b[c]) / 2 for c in CHROMS})
    cfg = f"""# FlexCible quickstart on toy data (written by make_toy_data.py; every path is absolute)
data:
  resolution: {BIN}
  chroms: [chr1, chr2, chr3]
  tracks:
    - {{name: CTCF, type: tf, motif: {MOTIF}}}
    - {{name: H3K27ac, type: histone}}
  samples:
    - name: cellA
      cool_path: {root}/cellA.cool
      bigwig_dir: {root}/cellA
    - name: cellB
      cool_path: {root}/cellB.cool
      bigwig_dir: {root}/cellB
  # consensus of the two cells as reference map (on real data: the mean of the OTHER cells)
  reference_cool: {root}/consensus.cool
  # chromosome split by parity of the chromosome number: chr1 and chr3 train, chr2 is the test chromosome
  train_parity: odd
  test_parity: even
sequence:
  fasta: {root}/genome.fa
  cache_dir: {root}/cache/genome
motifs:
  dir: {root}
  score_frac: 0.8
window:
  size: 32
  stride: 16
  transform: prescaled
  max_zero_fraction: 0.99
model:
  base_channels: 16
  depth: 2
  reference: input
  embed_reference: {root}/consensus.cool
train:
  output_dir: {root}/run
  cache_dir: {root}/cache/windows
  batch_size: 8
  lr: 0.001
  max_epochs: 40
  patience_epochs: 8
  val_fraction: 0.2
  track_dropout: 0.5
  track_dropout_mode: uniform_count
  num_workers: 0
  compile: false
  amp: none
"""
    (root / "quickstart.yaml").write_text(cfg)
    print(f"toy data written to {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy")
