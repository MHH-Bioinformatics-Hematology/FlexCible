"""Check of predict_epiphany.py: the pretrained model on the authors' own GM12878 bigWigs
(epiphany/bigwig/GM12878_*_merge.bigWig, scaled to a genome mean of 1) must reproduce the released
predictions (assemble_matrix_<chrom>.cool of the authors' prediction run). CPU, one chromosome.

    python check_epiphany_pretrained.py --chrom chr17
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
import predict_epiphany as pe  # noqa: E402

BW = Path.home() / "src/epiphany/bigwig"
REL = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/epiphany_pretrained")


def main():
    import cooler
    import pyBigWig
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="chr17")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    net = pe.load_model("pretrained", None, dev)
    bws = [pyBigWig.open(str(BW / f"GM12878_{t}_merge.bigWig")) for t in
           ("DNaseI", "H3K27ac", "H3K4me3", "H3K27me3", "CTCF")]
    size = bws[0].chroms()[a.chrom]
    n100 = size // 100
    X = np.zeros((5, n100), np.float32)
    for k, bw in enumerate(bws):
        h = bw.header()
        v = np.array(bw.values(a.chrom, 0, n100 * 100), np.float32).reshape(-1, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[k] = np.nan_to_num(np.nanmean(v, axis=1)) / (h["sumData"] / h["nBasesCovered"])
    n = size // 10000 + 1
    M = pe.predict_chrom(net, dev, X, n, True)               # log2(count + 1)
    rel = cooler.Cooler(str(REL / f"assemble_matrix_{a.chrom}.cool"))
    px = rel.matrix(balance=False, as_pixels=True).fetch(a.chrom)
    i, d = px.bin1_id.values, px.bin2_id.values - px.bin1_id.values
    ok = (d >= 0) & (d < 100)
    mine = M[i[ok], d[ok]]
    ref = px["count"].values[ok]
    g = np.isfinite(mine)
    print(f"{a.chrom}: {g.sum()} pixels, Pearson mine vs released {np.corrcoef(mine[g], ref[g])[0, 1]:.4f}, "
          f"max abs diff {np.abs(mine[g] - ref[g]).max():.3f}")


if __name__ == "__main__":
    main()
