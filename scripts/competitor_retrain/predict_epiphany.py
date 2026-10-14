"""Epiphany predictions of whole chromosomes at 10 kb: pretrained (released 10 kb weights, ReLU
output = log2(count + 1), pixels of row t at (t + 1 - k//2 - k%2, +k) as in the pretrained
model's own data preparation) or retrained (O/E output, pixels at (t - k//2 - k%2, +k)).

Chunks of 200 rows, step 100 rows, the central 100 rows of each chunk kept (every pixel is
predicted by exactly one row).
  retrained   competitor_oe/epiphany_re_<cell>_<chrom>.cool   (O/E)
  pretrained  pre_counts/epiphany_pre_<cell>_<chrom>.cool     (2**output - 1; oe_proper.py follows)

    python predict_epiphany.py --mode retrained --ckpt RUN/best.pt --cells k562
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
import common as C                      # noqa: E402
import prep                             # noqa: E402
import train_epiphany as te             # noqa: E402
from epiphany_model_10kb import Net     # noqa: E402

PRETRAINED = Path("/mnt/processed/wolffjoa/galaxy-hic-models/epiphany/pretrained_10kb.pt_model")


def load_model(mode, ckpt, dev):
    net = Net(1, 5, te.WINDOW)
    net.stride_units = te.STRIDE_U
    if mode == "pretrained":
        sd = torch.load(PRETRAINED, map_location="cpu")
    else:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd)
    net.load_state_dict(sd)
    return net.float().eval().to(dev)


def predict_chrom(net, dev, X, n, pretrained):
    shift = 1 if pretrained else 0
    M = np.full((n, te.NDIAG), np.nan, np.float32)
    s = 0
    # chunk rows s .. s + 199, rows s + 50 .. s + 149 kept; row t -> pixel (t + shift - HALF[k], +k)
    first = -50
    for s in range(first, n + 100, 100):
        x = torch.from_numpy(te.chunk_input(X, s)).to(dev)
        with torch.no_grad():
            y = net(x, seq_length=te.SEQ)[0].view(te.SEQ, te.NDIAG)
            if pretrained:
                y = torch.relu(y)
            y = y.cpu().numpy()
        t = np.arange(s + 50, s + 150)
        i = t[:, None] + shift - te.HALF[None, :]
        v = y[50:150]
        ok = (i >= 0) & (i < n) & (t[:, None] >= 0)
        kk = np.broadcast_to(te.K[None, :], i.shape)
        M[i[ok], kk[ok]] = v[ok]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrained", "retrained"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cells", nargs="*", default=C.ALL_CELLS)
    ap.add_argument("--chroms", nargs="*", default=C.HELD)
    ap.add_argument("--tag", default="re")
    a = ap.parse_args()
    dev = torch.device("cuda")
    net = load_model(a.mode, a.ckpt, dev)
    pre = a.mode == "pretrained"
    for cell in a.cells:
        for ch in a.chroms:
            name = f"epiphany_{'pre' if pre else a.tag}_{cell}"
            dst = (C.PRE_COUNTS / f"{name}_{ch}.cool") if pre else (C.COMP_OE / f"{name}_{ch}.cool")
            if dst.exists():
                continue
            t0 = time.time()
            X = np.load(prep.epi_file(cell, ch))
            n = C.chrom_size(ch) // te.RES + 1
            M = predict_chrom(net, dev, X, n, pre)
            if pre:
                M = np.where(np.isfinite(M), 2.0 ** M - 1.0, np.nan)
            C.band_to_cool(M, ch, te.RES, dst)
            print(f"{name} {ch}: {time.time() - t0:.0f} s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
