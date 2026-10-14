"""Epiphany (10 kb model of epiphany/epiphany/model_10kb.py: Net generator with the Disc of
adversarial.py) trained on our cells, chromosome split and O/E targets.

Original recipe kept: one 200-row chunk per step (batch 1), 14 000 x 100 bp window per row, 100
diagonals, Adam weight decay 5e-4, lr 1e-4 halved every 16 epochs, generator loss
lam * MSE + (1 - lam) * adversarial with lam 0.95, 55 epochs (an epoch = 1250 chunks of the
19 training chromosomes of the original; here 69 000 steps).
Changed: inputs (our cells, DNase), target (our 10 kb O/E instead of the HiC-DC+ z-score), 15
training and 2 validation chromosomes per cell, masked MSE, chunks drawn at random offsets, the
cells of C.TRAIN_CELLS drawn uniformly.

Label geometry (data_preparation of the original, 0-based bins): the window of row t is
[100 t - 7000, 100 t + 7000) in 100 bp units, output k of row t is the pixel (t - k//2 - k%2, +k).

    python train_epiphany.py --out RUN --steps 69000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
import common as C                          # noqa: E402
import prep                                 # noqa: E402
from epiphany_model_10kb import Net, Disc   # noqa: E402

import os as _os
RES = int(_os.environ.get("COMPRT_EPI_RES", 10000))     # row grid (registry 77b): same 200 rows, 100 diagonals, reach 100 bins
UNIT = 100                                              # bp of one input bin
STRIDE_U = RES // UNIT                                  # input bins per row
SEQ, NDIAG = 200, 100
WINDOW = 14000 * RES // 10000                           # window per row in input bins: 1.4 Mb at 10 kb, scales with the row grid
K = np.arange(NDIAG)
HALF = K // 2 + K % 2                       # pixel (t - HALF[k], t - HALF[k] + k)


def chunk_label(band, s: int):
    """(200, 100) O/E of the pixels of rows s .. s + 199 (NaN where missing or off the chromosome)."""
    n = band.shape[0]
    t = np.arange(s, s + SEQ)
    i = t[:, None] - HALF[None, :]
    ok = (i >= 0) & (i < n)
    v = np.full((SEQ, NDIAG), np.nan, np.float32)
    ii = np.clip(i, 0, n - 1)
    v[ok] = np.asarray(band[ii[ok], np.broadcast_to(K[None, :], i.shape)[ok]], np.float32)
    return v


def chunk_input(X, s: int):
    """(5, 34000) 100 bp means for the windows of rows s .. s + 199, zero outside the chromosome."""
    a = STRIDE_U * s - WINDOW // 2
    L = (SEQ - 1) * STRIDE_U + WINDOW + STRIDE_U
    out = np.zeros((X.shape[0], L), np.float32)
    lo, hi = max(a, 0), min(a + L, X.shape[1])
    if hi > lo:
        out[:, lo - a:hi - a] = X[:, lo:hi]
    return out


class Chunks(torch.utils.data.Dataset):
    def __init__(self, cells, chroms, mode, n, seed=0, per_cell_val=40):
        self.cells, self.chroms, self.mode, self.n, self.seed = list(cells), list(chroms), mode, n, seed
        self.size = {ch: C.chrom_size(ch) // RES for ch in self.chroms}
        self.w = np.array([self.size[c] for c in self.chroms], float)
        self.w /= self.w.sum()
        self.cache = {}
        if mode == "val":
            self.items = []
            for cell in self.cells:
                cand = [(cell, ch, s) for ch in self.chroms for s in range(100, self.size[ch] - 300, SEQ)]
                self.items += cand[::max(1, len(cand) // per_cell_val)][:per_cell_val]

    def __len__(self):
        return len(self.items) if self.mode == "val" else self.n

    def get(self, cell, ch):
        k = (cell, ch)
        if k not in self.cache:
            self.cache[k] = (np.load(prep.epi_file(cell, ch), mmap_mode="r"), C.load_band(cell, ch))
        return self.cache[k]

    def __getitem__(self, idx):
        if self.mode == "val":
            cell, ch, s = self.items[idx]
        else:
            rng = np.random.default_rng((self.seed, idx, int.from_bytes(os.urandom(4), "little")))
            for _ in range(20):
                cell = self.cells[int(rng.integers(len(self.cells)))]
                ch = self.chroms[int(rng.choice(len(self.chroms), p=self.w))]
                s = int(rng.integers(0, self.size[ch] - SEQ))
                X, band = self.get(cell, ch)
                y = chunk_label(band, s)
                if np.isfinite(y).mean() > 0.2:
                    break
        X, band = self.get(cell, ch)
        y = chunk_label(band, s)
        x = chunk_input(X, s)
        m = np.isfinite(y)
        return (torch.from_numpy(x), torch.from_numpy(np.nan_to_num(y)), torch.from_numpy(m))


def validate(model, loader, dev):
    model.eval()
    se, n, ys, ts = 0.0, 0, [], []
    with torch.no_grad():
        for x, y, m in loader:
            out, _ = model(x[0].to(dev), seq_length=SEQ)
            out = out.view(SEQ, NDIAG)
            y, m = y[0].to(dev), m[0].to(dev)
            se += (((out - y) ** 2) * m).sum().item()
            n += m.sum().item()
            ys.append(out[m].cpu().numpy())
            ts.append(y[m].cpu().numpy())
    model.train()
    a, b = np.concatenate(ys), np.concatenate(ts)
    return se / max(n, 1), float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=69000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--steps-per-epoch", type=int, default=1250)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cells", nargs="*", default=C.TRAIN_CELLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0,
                    help="original rule (registry 77b): epochs of --steps-per-epoch chunks (original 55, lr halved "
                         "every 16), validation every epoch, patience --patience, best-loss weights kept")
    ap.add_argument("--patience", type=int, default=12)
    a = ap.parse_args()
    if a.epochs:
        a.steps, a.val_every = a.epochs * a.steps_per_epoch, a.steps_per_epoch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    model = Net(1, 5, WINDOW)
    model.stride_units = STRIDE_U
    model = model.to(dev)
    disc = Disc().to(dev)
    params = list(model.parameters())
    tr = Chunks(a.cells, C.TRAIN, "train", a.steps * 1, a.seed)
    va = Chunks(a.cells, C.VALID, "val", 0)
    dl = torch.utils.data.DataLoader(tr, batch_size=None, shuffle=False, num_workers=a.workers,
                                     prefetch_factor=4, pin_memory=True)
    vl = torch.utils.data.DataLoader(va, batch_size=1, shuffle=False, num_workers=2)
    print(f"{len(a.cells)} cells, {len(va)} validation chunks", flush=True)
    step, best, hist, since_best = 0, float("inf"), [], 0
    state_f = out / "latest.pt"
    cur_lr = None
    opt = dopt = None

    def make_opts(lr):
        return (torch.optim.Adam(params, lr=lr, weight_decay=0.0005),
                torch.optim.Adam(disc.parameters(), lr=lr, weight_decay=0.0005))

    if state_f.exists():
        st = torch.load(state_f, map_location="cpu")
        model.load_state_dict(st["model"])
        disc.load_state_dict(st["disc"])
        step, best, hist = st["step"], st["best"], st["hist"]
        since_best = hist[-1].get("since_best", 0) if hist else 0
        print(f"resumed at step {step}", flush=True)
    t0 = time.time()
    run, rd = [], []
    model.train()
    disc.train()
    it = iter(dl)
    # the loader yields a stream; resumed runs simply draw new random chunks
    while step < a.steps:
        lr = max(a.lr * 0.5 ** (step // (16 * a.steps_per_epoch)), 1e-6)
        if lr != cur_lr:
            opt, dopt = make_opts(lr)
            cur_lr = lr
        x, y, m = next(it)
        x, y, m = x.to(dev), y.to(dev), m.to(dev)
        opt.zero_grad()
        out_, _ = model(x, seq_length=SEQ)
        pred = out_.view(SEQ, NDIAG)
        mse = (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1)
        d_out = disc(pred.view(1, 1, SEQ, NDIAG))
        adv = F.binary_cross_entropy_with_logits(d_out.view(1), torch.ones(1, device=dev))
        loss = a.lam * mse + (1 - a.lam) * adv
        loss.backward()
        opt.step()
        dopt.zero_grad()
        t_pred = disc(y.view(1, 1, SEQ, NDIAG))
        f_pred = disc(pred.detach().view(1, 1, SEQ, NDIAG))
        dl_ = F.binary_cross_entropy_with_logits(torch.cat([t_pred, f_pred]), torch.tensor([[1.0], [0.0]], device=dev))
        dl_.backward()
        dopt.step()
        step += 1
        run.append(mse.item())
        rd.append(torch.sigmoid(f_pred).item())
        if step % 100 == 0:
            print(f"step {step}/{a.steps} mse {np.mean(run[-100:]):.4f} D(fake) {np.mean(rd[-100:]):.2f} lr {lr:.1e} "
                  f"{step / (time.time() - t0):.1f} steps/s elapsed {(time.time() - t0) / 60:.1f} min "
                  f"gpu_mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        stop = a.max_minutes and (time.time() - t0) / 60 > a.max_minutes
        if step % a.val_every == 0 or step == a.steps or stop:
            vm, vr = validate(model, vl, dev)
            hist.append(dict(step=step, train_mse=float(np.mean(run[-500:])), val_mse=vm, val_pearson=vr,
                             minutes=(time.time() - t0) / 60))
            print("VAL", json.dumps(hist[-1]), flush=True)
            if vm < best:
                best = vm
                since_best = 0
                torch.save(model.state_dict(), out / "best.pt")
            else:
                since_best += 1
            hist[-1]["since_best"] = since_best
            torch.save(dict(model=model.state_dict(), disc=disc.state_dict(), step=step, best=best, hist=hist),
                       str(state_f) + ".tmp")
            os.replace(str(state_f) + ".tmp", state_f)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
        if a.epochs and since_best >= a.patience and step % a.steps_per_epoch == 0:
            print(f"early stop at epoch {step // a.steps_per_epoch}", flush=True)
            break
        if stop:
            print(f"stopped at step {step} after {a.max_minutes} min (pilot)", flush=True)
            break
    (out / "done.json").write_text(json.dumps(dict(steps=step, minutes=(time.time() - t0) / 60,
                                                   best_val_mse=best, cells=a.cells, lam=a.lam, res=RES,
                                                   epochs_run=step / a.steps_per_epoch, cap_reached=bool(a.epochs and step >= a.steps))))


if __name__ == "__main__":
    main()
