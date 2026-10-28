"""Does lower precision actually speed up THIS model on THIS hardware?

Measures a real training step (forward + backward + optimizer) for the Hi-C
U-Net at the shapes we train with, across precisions and memory formats.
Reported per step and as a speedup over the current setting (fp32 + TF32).

Note on FP8: Ada (RTX 4090) has FP8 tensor cores, but they are exposed for
matrix multiplication, not for the 2-D convolutions this model is built from,
so the win that FP8 gives transformers does not transfer here. The check below
reports whether a usable FP8 path exists at all rather than assuming one.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.model import build_backbone, build_spatial_cond, n_cond_channels


def bench(kind, n_sym, n_motifs, window, batch, mode, steps=12, channels_last=False):
    dev = torch.device("cuda")
    in_ch = n_cond_channels(n_sym, n_motifs, 0, "full")
    model = build_backbone(kind, in_ch, out_ch=1, base=48, depth=4).to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sym = torch.randn(batch, n_sym, window, device=dev)
    pairs = torch.randn(batch, n_motifs, 2, window, device=dev) if n_motifs else None
    tgt = torch.randn(batch, 1, window, window, device=dev)
    amp = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[mode]
    scaler = torch.amp.GradScaler("cuda") if mode == "fp16" else None

    def step():
        x = build_spatial_cond(sym, pairs, None, "full")
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        if amp is None:
            loss = torch.nn.functional.l1_loss(model(x), tgt)
            loss.backward(); opt.step()
        else:
            with torch.autocast("cuda", dtype=amp):
                loss = torch.nn.functional.l1_loss(model(x), tgt)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    mem = torch.cuda.max_memory_allocated() / 2**30
    del model, opt
    torch.cuda.empty_cache()
    return dt, mem


def main():
    print("FP8 availability:")
    print("  torch.float8_e4m3fn dtype:", hasattr(torch, "float8_e4m3fn"))
    try:
        import transformer_engine  # noqa: F401
        print("  transformer-engine: present")
    except Exception:
        print("  transformer-engine: absent (FP8 conv path unavailable)")
    print(f"  device: {torch.cuda.get_device_name(0)}\n")

    for label, n_sym, n_motifs, window, batch in (
            ("25 kb  (18 rows, W=128, b=16)", 18, 1, 128, 16),
            ("2 kb   (17 rows, W=512, b=3)", 17, 1, 512, 3)):
        print(label)
        base = None
        for mode, cl in (("fp32", False), ("bf16", False), ("fp16", False),
                         ("bf16", True)):
            try:
                dt, mem = bench("unet", n_sym, n_motifs, window, batch, mode,
                                channels_last=cl)
            except RuntimeError as e:
                print(f"  {mode}{'+chlast' if cl else '':9s} failed: {str(e)[:60]}")
                continue
            base = base or dt
            tag = f"{mode}{'+channels_last' if cl else ''}"
            print(f"  {tag:20s} {dt*1000:7.1f} ms/step  {mem:5.2f} GB  "
                  f"{base/dt:4.2f}x")
        print()


if __name__ == "__main__":
    main()
