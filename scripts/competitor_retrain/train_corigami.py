"""C.Origami (ConvTransModel, original architecture and optimiser settings) trained on our cells,
chromosome split and O/E targets. No Lightning: the loop below replaces training/main.py.

Original recipe kept: Adam lr 2e-4 without weight decay, gradient clipping 1, warm-up 10 of 80
epochs then cosine, MSE on the 256 x 256 map, 5 Mb intervals every 500 kb with a random 2 097 152 bp
window inside the interval, Gaussian noise 0.1 on sequence and features, reverse (and complement)
augmentation, intervals overlapping centromeres or telomeres dropped, batch 8.
Changed: inputs and targets (our cells, DNase, O/E), validation = chr18 and chr21 of every training
cell instead of chr3, masked MSE (pixels without a measured O/E excluded), bf16 autocast.

    python train_corigami.py --out RUN --steps 60000
    python train_corigami.py --out RUN --steps 3000 --max-minutes 30        # pilot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
import common as C                                     # noqa: E402
import corigami_models                                 # noqa: E402

STRIDE, SPAN = 500_000, 5_000_000


def omit_regions():
    import pandas as pd
    df = pd.read_csv(HERE / "vendor/centrotelo_hg38.bed", sep="\t", names=["c", "s", "e"])
    return {c: g[["s", "e"]].to_numpy() for c, g in df.groupby("c")}


class DS(torch.utils.data.Dataset):
    def __init__(self, cells, chroms, mode, seed=0):
        self.cells, self.mode = list(cells), mode
        om = omit_regions()
        self.items = []
        for cell in self.cells:
            for ch in chroms:
                size = C.chrom_size(ch)
                if mode == "train":
                    starts = [k * STRIDE for k in range(int((size - SPAN) / STRIDE) + 1)]
                    ends = [s + SPAN for s in starts]
                else:                        # consecutive windows
                    starts = list(range(0, size - C.WIN + 1, C.WIN))
                    ends = [s + C.WIN for s in starts]
                for s, e in zip(starts, ends):
                    o = om.get(ch)
                    if o is not None and ((s <= o[:, 1]) & (o[:, 0] <= e)).any():
                        continue
                    self.items.append((cell, ch, s, e))
        self.feat = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        cell, ch, s0, e0 = self.items[k]
        rng = np.random.default_rng((k * 7919 + int.from_bytes(os.urandom(4), "little")) & 0xFFFFFFFF)
        s = int(rng.integers(s0, e0 - C.WIN + 1)) if self.mode == "train" else s0
        for _ in range(5):                    # a window without a single measured pixel is redrawn
            tgt, ok = C.co_target(C.load_band(cell, ch), s)
            if ok.mean() > 0.2 or self.mode != "train":
                break
            s = int(rng.integers(s0, e0 - C.WIN + 1))
        seq = np.asarray(C.hg38_seq(ch)[s:s + C.WIN], np.int8)
        F = self.feat.get((cell, ch))
        if F is None:
            F = self.feat[(cell, ch)] = np.load(C.cofeat_file(cell, ch), mmap_mode="r")
        feats = np.asarray(F[:, s:s + C.WIN], np.float32)
        if self.mode == "train" and rng.random() < 0.5:
            seq, feats = seq[::-1], feats[:, ::-1]
            tgt, ok = tgt[::-1, ::-1], ok[::-1, ::-1]
            if rng.random() < 0.5:                       # complement of the original reverse()
                lut = np.array([1, 0, 3, 2, 4], np.int8)
                seq = lut[seq]
        return (torch.from_numpy(np.ascontiguousarray(seq)), torch.from_numpy(np.ascontiguousarray(feats)),
                torch.from_numpy(np.ascontiguousarray(tgt)), torch.from_numpy(np.ascontiguousarray(ok)))


def build_input(seq, feats, noise: float):
    """GPU: one-hot (B, L, 5) + features (B, L, 2) -> (B, L, 7), Gaussian noise on all of it."""
    oh = torch.nn.functional.one_hot(seq.long(), 5).float()
    x = torch.cat([oh, feats.transpose(1, 2)], 2)
    if noise > 0:
        x = x + noise * torch.randn_like(x)
    return x


def lr_at(step, total, base, warm):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def validate(model, loader, dev):
    model.eval()
    se, n, rs = 0.0, 0, []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for seq, feats, tgt, ok in loader:
            x = build_input(seq.to(dev), feats.to(dev), 0.0)
            y = model(x).float()
            t, m = tgt.to(dev), ok.to(dev)
            se += (((y - t) ** 2) * m).sum().item()
            n += m.sum().item()
            for b in range(y.shape[0]):
                mm = m[b]
                if mm.sum() > 100:
                    a, c = y[b][mm], t[b][mm]
                    a, c = a - a.mean(), c - c.mean()
                    rs.append(float((a * c).sum() / (a.norm() * c.norm() + 1e-9)))
    model.train()
    return se / max(n, 1), float(np.mean(rs)) if rs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=54000, help="optimizer steps (original: 80 epochs x ~5400 / 8)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4, help="micro-batch (gradient accumulation to --batch)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--cells", nargs="*", default=C.TRAIN_CELLS)
    ap.add_argument("--max-val-windows", type=int, default=12, help="validation windows per cell")
    ap.add_argument("--seed", type=int, default=2077)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    model = corigami_models.ConvTransModel(2, mid_hidden=256).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0)
    tr = DS(a.cells, C.TRAIN, "train", a.seed)
    va = DS(a.cells, C.VALID, "val")
    # validation subsample: evenly spaced windows per cell
    byc = {}
    for it in va.items:
        byc.setdefault(it[0], []).append(it)
    va.items = [it for c, v in byc.items() for it in v[::max(1, len(v) // a.max_val_windows)][:a.max_val_windows]]
    print(f"train intervals {len(tr)} ({len(a.cells)} cells), validation windows {len(va)}", flush=True)
    mb = a.micro
    acc = a.batch // mb
    dl = torch.utils.data.DataLoader(tr, batch_size=mb, shuffle=True, num_workers=a.workers, drop_last=True,
                                     persistent_workers=True, prefetch_factor=2, pin_memory=True)
    vl = torch.utils.data.DataLoader(va, batch_size=mb, shuffle=False, num_workers=min(4, a.workers))
    state_f = out / "latest.pt"
    step, best, hist = 0, float("inf"), []
    if state_f.exists():
        st = torch.load(state_f, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, best, hist = st["step"], st["best"], st["hist"]
        print(f"resumed at step {step}", flush=True)
    warm = int(0.125 * a.steps)
    t0 = time.time()
    t_start = t0
    seen = 0
    run_loss = []
    model.train()
    it = iter(dl)
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, warm)
        tot = 0.0
        for _ in range(acc):
            try:
                seq, feats, tgt, ok = next(it)
            except StopIteration:
                it = iter(dl)
                seq, feats, tgt, ok = next(it)
            x = build_input(seq.to(dev, non_blocking=True), feats.to(dev, non_blocking=True), 0.1)
            t, m = tgt.to(dev, non_blocking=True), ok.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(x)
            loss = (((y.float() - t) ** 2) * m).sum() / m.sum().clamp(min=1)
            (loss / acc).backward()
            tot += loss.item() / acc
            seen += mb
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run_loss.append(tot)
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {np.mean(run_loss[-50:]):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{seen / el:.1f} samples/s elapsed {(time.time() - t_start) / 60:.1f} min "
                  f"gpu_mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        stop = a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes
        if step % a.val_every == 0 or step == a.steps or stop:
            vloss, vr = validate(model, vl, dev)
            hist.append(dict(step=step, train_loss=float(np.mean(run_loss[-200:])), val_mse=vloss, val_pearson=vr,
                             minutes=(time.time() - t_start) / 60, samples=seen))
            print("VAL", json.dumps(hist[-1]), flush=True)
            if vloss < best:
                best = vloss
                torch.save(model.state_dict(), out / "best.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, hist=hist),
                       str(state_f) + ".tmp")
            os.replace(str(state_f) + ".tmp", state_f)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
        if stop:
            print(f"stopped at step {step} after {a.max_minutes} min (pilot)", flush=True)
            break
    (out / "done.json").write_text(json.dumps(dict(steps=step, minutes=(time.time() - t_start) / 60,
                                                   samples=seen, best_val_mse=best, cells=a.cells)))


if __name__ == "__main__":
    main()
