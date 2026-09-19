"""Where does the conditioning tensor's cost go, and can it be cut?

build_spatial_cond materialises (B, C, W, W) every forward pass from 1-D
signals. At 2 kb that is 54 channels at 512x512. Two things look wasteful:

  1. It is built in fp32 even under bf16 autocast. Autocast only rewrites
     conv/matmul, not the elementwise add/mul/abs/cat that build the planes,
     so a 170 MB fp32 tensor is written and then read back by a conv that
     immediately casts it down. Building in bf16 halves that traffic.
  2. torch.cat allocates the full tensor and copies every plane into it, so
     each plane is written twice. Writing planes straight into a preallocated
     buffer removes one of those passes.

Reported: build time alone, and build + stem conv forward/backward, which is
what the training step actually pays.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from hicmultiome.model import build_spatial_cond, n_cond_channels


def build_preallocated(sym, pairs, planes, out):
    """Same tensor as build_spatial_cond(mode='full'), written in place."""
    b, t, w = sym.shape
    ti = sym[:, :, :, None]
    tj = sym[:, :, None, :]
    k = 0
    torch.add(ti, tj, out=out[:, k:k + t]);      out[:, k:k + t] *= 0.5;  k += t
    torch.mul(ti, tj, out=out[:, k:k + t]);      k += t
    torch.sub(ti, tj, out=out[:, k:k + t])
    out[:, k:k + t].abs_();                      k += t
    if pairs is not None and pairs.shape[1] > 0:
        m = pairs.shape[1]
        fwd, rev = pairs[:, :, 0, :], pairs[:, :, 1, :]
        torch.mul(fwd[:, :, :, None], rev[:, :, None, :], out=out[:, k:k + m]); k += m
        torch.mul(rev[:, :, :, None], fwd[:, :, None, :], out=out[:, k:k + m]); k += m
    if planes is not None and planes.shape[1] > 0:
        p = planes.shape[1]
        out[:, k:k + p] = planes
        k += p
    idx = torch.arange(w, device=sym.device, dtype=out.dtype)
    out[:, k] = (idx[:, None] - idx[None, :]).abs() / w
    return out


def timeit(fn, n=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--tracks", type=int, default=17)
    ap.add_argument("--motifs", type=int, default=1)
    ap.add_argument("--base", type=int, default=48)
    a = ap.parse_args()

    dev = "cuda"
    b, t, w, m = a.batch, a.tracks, a.w, a.motifs
    ch = n_cond_channels(t, m, 0, "full")
    sym32 = torch.randn(b, t, w, device=dev)
    prs32 = torch.rand(b, m, 2, w, device=dev)
    sym16, prs16 = sym32.bfloat16(), prs32.bfloat16()
    stem = nn.Conv2d(ch, a.base, 3, padding=1).to(dev)
    buf = torch.empty(b, ch, w, w, device=dev, dtype=torch.bfloat16)

    print(f"W={w} batch={b} tracks={t} motifs={m} -> {ch} cond channels")
    print(f"  tensor: {b*ch*w*w*4/1e6:.0f} MB fp32 / {b*ch*w*w*2/1e6:.0f} MB bf16\n")

    build32 = lambda: build_spatial_cond(sym32, prs32, None, "full")
    build16 = lambda: build_spatial_cond(sym16, prs16, None, "full")
    buildpre = lambda: build_preallocated(sym16, prs16, None, buf)

    print("build only")
    for name, fn in (("fp32 cat (current)", build32), ("bf16 cat", build16),
                     ("bf16 preallocated", buildpre)):
        print(f"  {name:22s} {timeit(fn):7.2f} ms")

    def step(build):
        x = build()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = stem(x)
        y.float().sum().backward()

    print("\nbuild + stem conv fwd/bwd")
    for name, fn in (("fp32 cat (current)", build32), ("bf16 cat", build16),
                     ("bf16 preallocated", buildpre)):
        print(f"  {name:22s} {timeit(lambda f=fn: step(f), n=20):7.2f} ms")

    # correctness: the fast paths must reproduce the current tensor
    ref = build_spatial_cond(sym32, prs32, None, "full")
    for name, got in (("bf16 cat", build16()), ("bf16 preallocated", buildpre())):
        err = (got.float() - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"\n  {name}: max abs deviation from fp32 build {err:.4g} "
              f"({rel:.2%} of range)")


if __name__ == "__main__":
    main()
