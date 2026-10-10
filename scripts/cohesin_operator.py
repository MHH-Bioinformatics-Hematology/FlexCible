"""Fit the degron operator for the cohesin-loss augmentation (registry section 55).

The operator (hicmultiome.cohesin.DegronOperator) maps a control log O/E window
to the change that cohesin depletion causes. It is fitted on the measured
HCT116 RAD21-mAC pair (untreated and 6 h auxin Hi-C) and nothing else:

  fit      windows of the ODD chromosomes (the training shards of the degron cache,
           the same positions in both conditions; targets decoded with each
           sample's own contact scale to log O/E units)
  report   the Akita test windows of the EVEN chromosomes (never used for the fit
           or for any choice: architecture and number of epochs are fixed)

Reported on the even windows: median per-window Pearson of the predicted with the
measured change, the same for the high-pass part (5x5 background removed: loops,
stripes), the spread of the predicted and the measured change, and two baselines
(the per-diagonal mean change; a single linear high-pass factor).

    CUDA_VISIBLE_DEVICES= nice -n 10 python scripts/cohesin_operator.py --res 10000
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from hicmultiome.cohesin import DegronOperator, save_operator            # noqa: E402

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
AUX_FRACTION = 0.26          # scripts/multicell_expanded.COHFIX_AUX_FRACTION


def load_pairs(cache: Path, split: str, parity: str | None, ctrl="hct116", aux="hct116_auxin"):
    """(control, auxin) log O/E windows at the same positions: (N, W, W) each.
    parity 'odd' / 'even' keeps those chromosomes, None all."""
    def scale(c):
        return json.loads((cache / f"norm_matrix_{c}.json").read_text())["scale"]
    su, sa = scale(ctrl), scale(aux)
    X, Y, names = [], [], []
    for f in sorted(glob.glob(str(cache / f"{aux}_{split}_chr*.npz"))):
        ch = int(f.rsplit("chr", 1)[-1][:-4])
        if parity == "odd" and ch % 2 == 0 or parity == "even" and ch % 2:
            continue
        a, u = np.load(f), np.load(f.replace(aux, ctrl))
        pos = {int(s): i for i, s in enumerate(u["start"])}
        for i, s in enumerate(a["start"]):
            if int(s) in pos:
                X.append(u["target"][pos[int(s)], 0].astype(np.float32) * su)
                Y.append(a["target"][i, 0].astype(np.float32) * sa)
                names.append(f"chr{ch}:{int(s)}")
    return np.stack(X), np.stack(Y), names


def upper_mask(n: int) -> np.ndarray:
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :]) >= 2


def highpass(d: np.ndarray, k: int = 5) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    return d - uniform_filter(d, k, mode="nearest")


def evaluate(pred: np.ndarray, meas: np.ndarray) -> dict:
    n = meas.shape[-1]
    m = upper_mask(n)

    def med_corr(a, b):
        r = [np.corrcoef(x[m], y[m])[0, 1] for x, y in zip(a, b)
             if x[m].std() > 0 and y[m].std() > 0]
        return float(np.median(r)) if r else float("nan")
    hp_p = np.stack([highpass(x) for x in pred])
    hp_m = np.stack([highpass(y) for y in meas])
    return {"pearson": med_corr(pred, meas), "pearson_highpass": med_corr(hp_p, hp_m),
            "sd_pred": float(pred[:, m].std()), "sd_meas": float(meas[:, m].std()),
            "mean_pred": float(pred[:, m].mean()), "mean_meas": float(meas[:, m].mean()),
            "sd_highpass_pred": float(hp_p[:, m].std()), "sd_highpass_meas": float(hp_m[:, m].std())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=int, required=True, help="bin size: 10000 or 25000")
    ap.add_argument("--cache", default=None,
                    help="degron cache (default cache_expanded/mc_all12deg_refres_drop_<k>kb)")
    ap.add_argument("--fraction", type=float, default=AUX_FRACTION,
                    help="cohesin fraction of the auxin condition")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    k = a.res // 1000
    cache = Path(a.cache) if a.cache else HM / f"cache_expanded/mc_all12deg_refres_drop_{k}kb"
    out = Path(a.out) if a.out else REPO / f"results/cohesin_fix/operator_{k}kb.pt"
    Xtr, Ytr, _ = load_pairs(cache, "train", "odd")
    Xte, Yte, _ = load_pairs(cache, "test", "even")
    n = Xtr.shape[-1]
    m = torch.from_numpy(upper_mask(n))
    print(f"{k} kb: {len(Xtr)} odd-chromosome training pairs, {len(Xte)} even-chromosome test "
          f"pairs, window {n}", flush=True)
    op = DegronOperator(a.width)
    opt = torch.optim.Adam(op.parameters(), a.lr)
    Xt, Dt = torch.from_numpy(Xtr), torch.from_numpy(Ytr - Xtr)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt))
        tot = nb = 0
        for b in range(0, len(Xt) - a.batch + 1, a.batch):
            i = perm[b:b + a.batch]
            x, d = Xt[i], Dt[i]
            if ep % 2:                                   # genomic reversal
                x, d = torch.flip(x, [-1, -2]), torch.flip(d, [-1, -2])
            loss = ((op(x) - d) ** 2)[:, m].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"  epoch {ep + 1}  train mse {tot / nb:.5f}  {time.time() - t0:.0f}s", flush=True)
    op.eval()
    with torch.no_grad():
        pred = op(torch.from_numpy(Xte)).numpy()
    meas = Yte - Xte
    res = {"model": evaluate(pred, meas)}
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    dd = np.zeros_like(meas)
    for kk in range(2, n):
        dd[:, dist == kk] = (Ytr - Xtr)[:, dist == kk].mean()
    res["baseline_distance_mean"] = evaluate(dd, meas)
    hp_tr = np.stack([highpass(x) for x in Xtr])
    alpha = float(((Ytr - Xtr) * hp_tr)[:, upper_mask(n)].sum() / (hp_tr[:, upper_mask(n)] ** 2).sum())
    res["baseline_linear_highpass"] = evaluate(alpha * np.stack([highpass(x) for x in Xte]), meas)
    res["baseline_linear_highpass"]["alpha"] = alpha
    res.update({"n_train": len(Xtr), "n_test_even": len(Xte), "epochs": a.epochs,
                "fraction_ref": a.fraction, "resolution": a.res})
    save_operator(out, op, a.fraction, a.res, meta=res)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
