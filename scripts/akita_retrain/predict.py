"""Whole-chromosome predictions of a retrained or the released Akita.

Windows of 1 048 576 bp at a stride of 224 bins (half the 448-bin target
span, the half-window stride of hicmultiome.predict) cover a chromosome; each
window is predicted forward and reverse-complemented (averaged), and the
windows are merged per pixel with a 2-D Hann weight (product of the 1-D Hann
weights of the two bins). Output: preds/<run>/<chrom>.npy, float16 band
(bin on the 2048 bp grid, distance 0..447, head), NaN where no window covers
the pixel (the first and last 32 bins of a chromosome), plus cells.json.

kinds
  retrained  weights of runs/<run>/best.weights.h5, heads = args.json cells;
             the output is already our proper O/E (clipped, smoothed target
             space), written unchanged.
  pretrained the released Akita (galaxy-hic-models/akita, model_best.h5; heads
             HFF, H1, GM12878, IMR90, HCT116 = our hff, h1, gm12878, imr90,
             hct116). Its target space is log2(obs/exp); as in registry
             sections 18, 58 and 72 the window map is converted to natural
             log, centred per diagonal, clipped to +-2 and Gaussian smoothed
             (sigma 2 bins) per window (oe_proper.py's last passes) before the
             windows are merged.

    python predict.py --kind retrained --run akita_A_all12 --chroms chr2 chr3
    python predict.py --kind pretrained --run akita_pretrained --chroms chr2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C.setup_basenji_path()

PRETRAINED = Path("/mnt/processed/wolffjoa/galaxy-hic-models/akita")
PRETRAINED_CELLS = ["hff", "h1", "gm12878", "imr90", "hct116"]
CLIP, SIGMA = 2.0, 2.0


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def rc_order() -> np.ndarray:
    """Index into a triu vector that maps the prediction of the reverse
    complemented window back (basenji layers.SwitchReverseTriu)."""
    r, c = C.iu()
    mat = np.zeros((C.NB, C.NB), int)
    mat[r, c] = np.arange(C.NV)
    full = mat.copy()                      # the lower triangle mirrors the upper
    low = np.tril_indices(C.NB, -1)
    full[low] = mat.T[low]
    return full[::-1, ::-1][r, c]


def log_to_final(L: np.ndarray) -> np.ndarray:
    """Per-diagonal centring, clip, Gaussian smoothing of a dense natural-log
    O/E block (registry 72, aml_competitors_loops.log_to_final)."""
    from scipy.ndimage import gaussian_filter
    n = L.shape[0]
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    off = np.zeros(n)
    for k in range(n):
        v = np.diagonal(L, k)
        v = v[np.isfinite(v)]
        off[k] = v.mean() if v.size else 0.0
    val = np.clip(L - off[d], -CLIP, CLIP)
    val = np.where(np.isfinite(val), val, 0.0)
    return gaussian_filter(val, sigma=SIGMA, truncate=1.0, mode="nearest")


def window_starts(size: int) -> np.ndarray:
    s = list(range(0, size - C.SEQ_LEN + 1, C.STRIDE_EVAL))
    last = ((size - C.SEQ_LEN) // C.BIN) * C.BIN
    if s and s[-1] != last:
        s.append(last)
    return np.array(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["retrained", "pretrained"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--chroms", nargs="+", required=True)
    ap.add_argument("--weights", default="best.weights.h5")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    from basenji import seqnn

    if a.kind == "retrained":
        out = C.W / "runs" / a.run
        cells = json.loads((out / "args.json").read_text())["cells_list"]
        p = json.load(open(C.W / "params_akita_original.json"))
        p["model"]["head_hic"][-1]["units"] = len(cells)
        m = seqnn.SeqNN(p["model"])
        m.model.load_weights(str(out / a.weights))
    else:
        cells = PRETRAINED_CELLS
        p = json.load(open(PRETRAINED / "params.json"))
        m = seqnn.SeqNN(p["model"])
        m.restore(str(PRETRAINED / "model_best.h5"))
    model = m.model
    K = len(cells)
    odir = C.W / "preds" / a.run
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "cells.json").write_text(json.dumps(cells))

    @tf.function
    def fwd(codes):
        return model(tf.one_hot(tf.cast(codes, tf.int32), 4), training=False)

    r, c = C.iu()
    ro = rc_order()
    h1 = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(C.NB) + 0.5) / C.NB)
    w2 = np.maximum(h1[r] * h1[c], 1e-4).astype(np.float32)           # (NV,)
    fai = {}
    for ln in open(str(C.FASTA) + ".fai"):
        f = ln.split("\t")
        fai[f[0]] = int(f[1])

    for chrom in a.chroms:
        of = odir / f"{chrom}.npy"
        if of.exists():
            log(f"{chrom}: exists")
            continue
        t0 = time.time()
        seq = np.load(C.seq_dir() / f"{chrom}.npy")
        size = fai[chrom]
        n = int(np.ceil(size / C.BIN))
        acc = np.zeros((n, C.NB, K), np.float32)
        wsum = np.zeros((n, C.NB), np.float32)
        starts = window_starts(size)
        for i in range(0, len(starts), a.batch):
            ss = starts[i:i + a.batch]
            x = np.stack([seq[s:s + C.SEQ_LEN] for s in ss]).astype(np.int8)
            xr = np.where(x[:, ::-1] < 4, 3 - x[:, ::-1], 4).astype(np.int8)
            both = np.concatenate([x, xr])
            y = fwd(both).numpy()                                    # (2B, NV, K)
            B = len(ss)
            pf, pr = y[:B], y[B:][:, ro]
            pred = 0.5 * (pf + pr)
            for j, s in enumerate(ss):
                v = pred[j]
                if a.kind == "pretrained":
                    v2 = np.empty_like(v)
                    for k in range(K):
                        Mx = np.full((C.NB, C.NB), np.nan)
                        Mx[r, c] = v[:, k] * np.log(2.0)             # log2 -> natural log
                        Mx = np.where(np.isfinite(Mx), Mx, Mx.T)
                        # diagonals 0 and 1 are not predicted: left out of the centring
                        np.fill_diagonal(Mx, np.nan)
                        Mx[np.arange(C.NB - 1), np.arange(1, C.NB)] = np.nan
                        Mx[np.arange(1, C.NB), np.arange(C.NB - 1)] = np.nan
                        v2[:, k] = log_to_final(Mx)[r, c]
                    v = v2
                b0 = (int(s) + C.CROP) // C.BIN
                rows, d = b0 + r, c - r
                keep = rows < n
                acc[rows[keep], d[keep]] += v[keep] * w2[keep, None]
                wsum[rows[keep], d[keep]] += w2[keep]
            if (i // a.batch) % 25 == 0:
                log(f"{chrom}: {i + len(ss)}/{len(starts)} windows, {time.time() - t0:.0f} s")
        res = np.where(wsum[:, :, None] > 0, acc / np.maximum(wsum[:, :, None], 1e-12), np.nan)
        np.save(str(of) + ".tmp.npy", res.astype(np.float16))
        Path(str(of) + ".tmp.npy").rename(of)
        log(f"{chrom}: done {len(starts)} windows, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
