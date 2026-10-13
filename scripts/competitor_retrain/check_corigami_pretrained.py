"""Check of predict_corigami.py: the pretrained model on the authors' own IMR-90 tracks
(ctcf_log2fc.bw, atac.bw with norm 'log', hg38 sequence), without the quantile map, must
reproduce the released prediction (C.Origami testNovember/IMR90/<chrom>.cool, log(count + 1)
on the 8192 bp grid) in a window.

    python check_corigami_pretrained.py --chrom chr2 --start 100000000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
import common as C                   # noqa: E402
import predict_corigami as pc        # noqa: E402

REL = Path.home() / "src/C.Origami/testNovember/IMR90"


def main():
    import cooler
    import pyBigWig
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="chr2")
    ap.add_argument("--start", type=int, default=100_000_000)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pc.load_model("pretrained", None, dev)
    s = a.start // C.PIX * C.PIX
    seq = np.asarray(C.hg38_seq(a.chrom)[s:s + C.WIN], np.int64)
    ct = C.bw_values(pyBigWig.open(str(C.REF_CTCF)), a.chrom, s, s + C.WIN)
    at = np.log(C.bw_values(pyBigWig.open(str(C.REF_ATAC)), a.chrom, s, s + C.WIN) + 1)
    x = np.zeros((1, C.WIN, 7), np.float32)
    x[0, np.arange(C.WIN), seq] = 1
    x[0, :, 5], x[0, :, 6] = ct, at
    with torch.no_grad():
        P = model(torch.from_numpy(x).to(dev))[0].cpu().numpy()
    P = (P + P.T) / 2
    rel = cooler.Cooler(str(REL / f"{a.chrom}.cool"))
    b0 = s // C.PIX
    R = rel.matrix(balance=False, sparse=False).fetch(f"{a.chrom}:{s}-{s + C.WIN}")
    n = min(R.shape[0], C.NPIX)
    i, j = np.triu_indices(n, 1)
    ok = (j - i) < 200
    print(f"{a.chrom}:{s} released {R.shape}, Pearson {np.corrcoef(P[:n, :n][i, j][ok], R[:n, :n][i, j][ok])[0, 1]:.4f}, "
          f"max abs diff {np.abs(P[:n, :n][i, j][ok] - R[:n, :n][i, j][ok]).max():.3f}")


if __name__ == "__main__":
    main()
