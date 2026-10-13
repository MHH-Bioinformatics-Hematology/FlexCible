"""C.Origami predictions of whole chromosomes: pretrained (released IMR-90 model, log(count + 1)
output) or retrained (our O/E output), on the cell's own inputs (CTCF and DNase mapped onto the
IMR-90 reference distributions, the "q" variant of registry section 72).

Windows of 2 097 152 bp every 524 288 bp on the 8192 bp grid, prediction symmetrised, windows
combined by a weighted mean (weight 1 at the window centre, 0.1 at its edge, along the pixel
centre). Output: a band array and a cool at 8192 bp per chromosome.
  retrained   competitor_oe/corigami_re_<cell>_<chrom>.cool   (O/E, read by the scoring directly)
  pretrained  pre_counts/corigami_pre_<cell>_<chrom>.cool     (expm1 of the output; oe_proper.py
                                                              turns it into competitor_oe/...)

    python predict_corigami.py --mode retrained --ckpt RUN/best.pt --cells k562 imr90
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
import corigami_models                  # noqa: E402

STRIDE = 64 * C.PIX
PRETRAINED = Path("/mnt/processed/wolffjoa/galaxy-hic-models/corigami/corigami_base.ckpt")


def _stub_omegaconf():
    """The released checkpoint pickles omegaconf config objects next to the weights; omegaconf is not
    needed for inference, so the classes are replaced by inert stubs while loading."""
    import types
    try:
        import omegaconf  # noqa: F401
        return
    except ImportError:
        pass

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, st):
            self.__dict__["_state"] = st

    def mk(name):
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: type(attr, (_Stub,), {})
        sys.modules[name] = m
        return m
    root = mk("omegaconf")
    for sub in ("dictconfig", "listconfig", "base", "nodes", "basecontainer", "omegaconf", "errors"):
        setattr(root, sub, mk(f"omegaconf.{sub}"))


def load_model(mode, ckpt, dev):
    m = corigami_models.ConvTransModel(2, mid_hidden=256)
    if mode == "pretrained":
        _stub_omegaconf()
        sd = torch.load(PRETRAINED, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    else:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd)
    m.load_state_dict(sd)
    return m.to(dev).eval()


def chrom_inputs(cell, chrom):
    return np.load(C.cofeat_file(cell, chrom)), C.hg38_seq(chrom)


def predict_chrom(model, dev, feats, seq, batch=8):
    size = seq.shape[0]
    nb = -(-size // C.PIX)
    starts = list(range(0, size - C.WIN + 1, STRIDE))
    last = (size - C.WIN) // C.PIX * C.PIX
    if starts[-1] != last:
        starts.append(last)
    num = np.zeros((nb, C.NPIX), np.float32)
    den = np.zeros((nb, C.NPIX), np.float32)
    k = np.arange(C.NPIX)
    pos = np.arange(C.NPIX)
    for b0 in range(0, len(starts), batch):
        ss = starts[b0:b0 + batch]
        x = np.zeros((len(ss), C.WIN, 7), np.float32)
        for q, s in enumerate(ss):
            x[q, np.arange(C.WIN), np.asarray(seq[s:s + C.WIN], np.int64)] = 1.0
            x[q, :, 5:] = feats[:, s:s + C.WIN].T.astype(np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(torch.from_numpy(x).to(dev)).float().cpu().numpy()
        for q, s in enumerate(ss):
            P = (y[q] + y[q].T) / 2
            r0 = s // C.PIX
            for d in range(C.NPIX):
                r = np.arange(C.NPIX - d)
                cen = r + d / 2.0
                w = 1.0 - 0.9 * np.abs(cen - (C.NPIX - 1) / 2) / ((C.NPIX - 1) / 2)
                num[r0 + r, d] += P[r, r + d] * w
                den[r0 + r, d] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(den > 0, num / den, np.nan)
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
    model = load_model(a.mode, a.ckpt, dev)
    for cell in a.cells:
        for ch in a.chroms:
            name = f"corigami_{'pre' if a.mode == 'pretrained' else a.tag}_{cell}"
            dst = (C.PRE_COUNTS / f"{name}_{ch}.cool") if a.mode == "pretrained" else (C.COMP_OE / f"{name}_{ch}.cool")
            if dst.exists():
                continue
            t0 = time.time()
            feats, seq = chrom_inputs(cell, ch)
            M = predict_chrom(model, dev, feats, seq)
            if a.mode == "pretrained":
                M = np.expm1(M)
                M = np.where(np.isfinite(M), np.maximum(M, 0.0), np.nan)
            C.band_to_cool(M, ch, C.PIX, dst)
            print(f"{name} {ch}: {time.time() - t0:.0f} s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
