"""Post-hoc sharpening of cached predictions: tune on 1/3 of regions, report on 2/3.

Reads cache_<cell>.npz from sharpen_cache.py. Split is by chromosome (tune set =
chromosomes greedily filled to ~1/3 of GM12878 regions), identical for both cells.
Distance-dependent variants (contrast / quantile matching) learn their tables on
the GM12878 tuning split only, then are applied unchanged to both cells.

    python scripts/sharpen_eval.py --dir /mnt/processed/wolffjoa/hic-multiome/runs/sharpen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.fft import irfft2, rfft2
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible import bioscore as bs  # noqa: E402

KMIN, KMAX = 2, 250          # distances used for curves / SCC (bins, 2 kb) -> 4-500 kb
INS_W = (25, 50)             # 50 kb, 100 kb diamond


# ----------------------------------------------------------------- utilities
def masks(P, T):
    live = np.abs(P).sum(1) > 0
    M = np.where(live[:, None] & live[None, :], P, np.nan)
    Tm = np.where(T == 0, np.nan, T)
    return M, Tm


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def pixel_r(M, Tm):
    iu = np.triu_indices(M.shape[0], k=2)
    x, y = M[iu], Tm[iu]
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    return pearson(x[ok], y[ok]) if ok.sum() > 100 else np.nan


def per_diag(M, Tm):
    """Pearson r per diagonal k, plus SCC (hicrep weights, rank-variance, h=0)."""
    n = M.shape[0]
    rs = np.full(KMAX + 1, np.nan)
    num = den = 0.0
    for k in range(KMIN, min(KMAX, n - 3) + 1):
        x, y = np.diagonal(M, k), np.diagonal(Tm, k)
        ok = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
        N = ok.sum()
        if N < 10:
            continue
        x, y = x[ok], y[ok]
        r = pearson(x, y)
        if not np.isfinite(r):
            continue
        rs[k] = r
        w = N * np.sqrt(np.var(rankdata(x) / N) * np.var(rankdata(y) / N))
        num += w * r; den += w
    return rs, (num / den if den > 0 else np.nan)


def insulation(A, w):
    """Diamond insulation: mean of A[i-w:i, i+1:i+w+1] (nan-aware)."""
    n = A.shape[0]
    v = np.nan_to_num(A); c = np.isfinite(A).astype(float)
    S = np.pad(v, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    C = np.pad(c, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    out = np.full(n, np.nan)
    for i in range(w, n - w - 1):
        r0, r1, c0, c1 = i - w, i, i + 1, i + w + 1
        s = S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]
        m = C[r1, c1] - C[r0, c1] - C[r1, c0] + C[r0, c0]
        if m > 0.5 * w * w:
            out[i] = s / m
    return out


def boundaries(ins, n=None, prom=0.1):
    x = -ins.copy()
    ok = np.isfinite(x)
    if ok.sum() < 20:
        return np.array([], int)
    x[~ok] = np.nanmin(x[ok])
    pk, pr = find_peaks(x, prominence=0)
    pr = pr["prominences"]
    if n is None:
        return pk[pr > prom]
    return pk[np.argsort(-pr)[:n]]


def ins_metrics(M, Tm, w):
    ip, it = insulation(M, w), insulation(Tm, w)
    ok = np.isfinite(ip) & np.isfinite(it)
    r = pearson(ip[ok], it[ok]) if ok.sum() > 20 else np.nan
    bt = boundaries(it)
    if len(bt) == 0:
        return r, np.nan
    bp = boundaries(ip, n=len(bt))                  # top-|bt| predicted minima: scale-free
    hit = sum(np.any(np.abs(bp - b) <= 3) for b in bt)
    return r, hit / len(bt)                          # = precision = recall (same count)


# ----------------------------------------------------------------- variants
def unsharp(P, s, a):
    return P + a * (P - gaussian_filter(P, s, mode="nearest"))


def gauss_otf(shape, s):
    n, m = shape
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.rfftfreq(m)[None, :]
    return np.exp(-2 * (np.pi * s) ** 2 * (fx ** 2 + fy ** 2))


def wiener(P, s, K):
    n = P.shape[0]
    pad = np.pad(P, n // 4, mode="reflect")
    H = gauss_otf(pad.shape, s)
    X = rfft2(pad) * H / (H * H + K)
    out = irfft2(X, s=pad.shape)
    return out[n // 4:n // 4 + n, n // 4:n // 4 + n]


def richardson_lucy(P, s, it):
    """RL in the positive domain exp(P), then log back."""
    y = np.exp(P)
    x = y.copy()
    for _ in range(it):
        est = gaussian_filter(x, s, mode="nearest")
        x = x * gaussian_filter(y / np.maximum(est, 1e-9), s, mode="nearest")
    return np.log(np.maximum(x, 1e-9))


def diag_apply(P, fn):
    """Apply fn(values, k) to each diagonal k>=0 (and mirror)."""
    n = P.shape[0]
    Q = P.copy()
    for k in range(n):
        idx = np.arange(n - k)
        v = fn(P[idx, idx + k], k)
        Q[idx, idx + k] = v; Q[idx + k, idx] = v
    return Q


class DiagStats:
    """Per-distance std / quantile tables of P and T, learned on the tuning split."""
    QS = np.linspace(0, 1, 201)

    def __init__(self, Ps, Ts, kmax=600):
        pv = [[] for _ in range(kmax)]; tv = [[] for _ in range(kmax)]
        for P, T in zip(Ps, Ts):
            M, Tm = masks(P, T)
            for k in range(min(kmax, M.shape[0])):
                x, y = np.diagonal(M, k), np.diagonal(Tm, k)
                ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
                pv[k].append(x[ok]); tv[k].append(y[ok])
        self.sp = np.ones(kmax); self.st = np.ones(kmax)
        self.qp = np.zeros((kmax, len(self.QS))); self.qt = np.zeros((kmax, len(self.QS)))
        for k in range(kmax):
            x = np.concatenate(pv[k]) if pv[k] else np.array([0.0])
            y = np.concatenate(tv[k]) if tv[k] else np.array([0.0])
            if len(x) < 10:
                x, y = np.array([0.0]), np.array([0.0])
            self.sp[k], self.st[k] = max(x.std(), 1e-6), max(y.std(), 1e-6)
            self.qp[k] = np.quantile(x, self.QS); self.qt[k] = np.quantile(y, self.QS)
        self.kmax = kmax

    def contrast(self, P):
        def f(v, k):
            k = min(k, self.kmax - 1)
            return v.mean() + (v - v.mean()) * self.st[k] / self.sp[k]
        return diag_apply(P, f)

    def quantile(self, P):
        def f(v, k):
            k = min(k, self.kmax - 1)
            qp = np.maximum.accumulate(self.qp[k] + 1e-9 * np.arange(len(self.QS)))
            return np.interp(v, qp, self.qt[k])
        return diag_apply(P, f)


def contrast_region(P, stt):
    """Per-region: each diagonal's own std rescaled to the typical target std."""
    def f(v, k):
        k2 = min(k, stt.kmax - 1)
        sd = v.std()
        return v if sd < 1e-6 else v.mean() + (v - v.mean()) * stt.st[k2] / sd
    return diag_apply(P, f)


def build_variants(stt):
    V = {"identity": lambda P: P}
    for s in (1, 2, 3):
        for a in (0.25, 0.5, 1, 2):
            V[f"unsharp s={s} a={a}"] = (lambda P, s=s, a=a: unsharp(P, s, a))
    for s in (1, 2):
        for K in (0.01, 0.05, 0.2):
            V[f"wiener s={s} K={K}"] = (lambda P, s=s, K=K: wiener(P, s, K))
        for it in (5, 20):
            V[f"RL s={s} it={it}"] = (lambda P, s=s, it=it: richardson_lucy(P, s, it))
    V["diag-contrast (global)"] = stt.contrast
    V["diag-contrast (per-region)"] = lambda P: contrast_region(P, stt)
    V["diag-quantile"] = stt.quantile
    return V


LINEAR = {"identity"}
DESC = {
    "unsharp": "spatial, linear filter (changes Pearson)",
    "wiener": "spatial, linear filter",
    "RL": "spatial, nonlinear",
    "diag-contrast (global)": "per-distance affine (non-uniform across k)",
    "diag-contrast (per-region)": "per-distance affine, per region",
    "diag-quantile": "per-distance monotone nonlinear",
}


def evaluate(P, T):
    M, Tm = masks(P, T)
    rec = {"pixel_r": pixel_r(M, Tm)}
    rs, scc = per_diag(M, Tm)
    rec["scc"] = scc
    for lo, hi, nm in ((2, 10, "r_4-20kb"), (10, 50, "r_20-100kb"),
                       (50, 150, "r_100-300kb"), (150, 250, "r_300-500kb")):
        rec[nm] = float(np.nanmean(rs[lo:hi]))
    for w in INS_W:
        r, f = ins_metrics(M, Tm, w)
        rec[f"ins_r_{2*w}kb"] = r; rec[f"bnd_hit_{2*w}kb"] = f
    rec["_rs"] = rs
    rec["_bio"] = bs.region_metrics(Tm, M)             # project bioscore: loops on 16 kb, TAD calls
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/mnt/processed/wolffjoa/hic-multiome/runs/sharpen")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    d = Path(a.dir)
    data = {}
    for cell in ("gm12878", "imr90"):
        z = np.load(d / f"cache_{cell}.npz", allow_pickle=True)
        data[cell] = (list(z["P"]), list(z["T"]), list(z["names"]))

    # chromosome split, fixed, from GM12878 region list
    names = data["gm12878"][2]
    chroms = pd.Series([n.split(":")[0] for n in names])
    cnt = chroms.value_counts()
    order = sorted(cnt.index, key=lambda c: (int(c[3:]) if c[3:].isdigit() else 99))
    tune_chr, tot = set(), 0
    for i, c in enumerate(order):                  # every 3rd chromosome until ~1/3
        if i % 3 == 0 and tot < len(names) / 3:
            tune_chr.add(c); tot += cnt[c]
    print(f"tune chromosomes: {sorted(tune_chr, key=order.index)} ({tot}/{len(names)} regions)", flush=True)

    Pt, Tt, _ = data["gm12878"]
    tmask = [n.split(":")[0] in tune_chr for n in names]
    stt = DiagStats([p for p, m in zip(Pt, tmask) if m], [t for t, m in zip(Tt, tmask) if m])
    V = build_variants(stt)

    from concurrent.futures import ProcessPoolExecutor
    global _V
    _V = V
    rows, curves, bio = [], {}, {}
    for cell, (Ps, Ts, nms) in data.items():
        jobs = [(cell, i) for i in range(len(Ps))]
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(Ps, Ts)) as ex:
            for cell_i, recs in zip(jobs, ex.map(_run_one, [j[1] for j in jobs], chunksize=4)):
                i = cell_i[1]
                split = "tune" if nms[i].split(":")[0] in tune_chr else "test"
                for vn, rec in recs.items():
                    curves.setdefault((cell, split, vn), []).append(rec.pop("_rs"))
                    bio.setdefault((cell, split, vn), []).append(rec.pop("_bio"))
                    rows.append({"cell": cell, "split": split, "region": nms[i], "variant": vn, **rec})
        print(f"{cell}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(d / "sharpen_per_region.csv", index=False)
    med = df.groupby(["cell", "split", "variant"]).median(numeric_only=True).reset_index()
    keys = ["insulation_pearson", "boundary_f1", "loop_f1", "loopthr_f1", "loopthr_n_pred",
            "apa_ratio", "anchor_auroc", "anchor_spearman", "loop_n_true"]
    brows = [{"cell": c, "split": s_, "variant": v, **{k: bs.pool(r)[k] for k in keys}}
             for (c, s_, v), r in bio.items()]
    med = med.merge(pd.DataFrame(brows), on=["cell", "split", "variant"])
    med.to_csv(d / "sharpen_median.csv", index=False)
    cur = {f"{c}|{s}|{v}": np.nanmedian(np.array(x), 0) for (c, s, v), x in curves.items()}
    np.savez(d / "sharpen_curves.npz", **cur)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    for split in ("tune", "test"):
        for cell in ("gm12878", "imr90"):
            print(f"\n=== {cell} / {split} ===")
            print(med[(med.cell == cell) & (med.split == split)].drop(columns=["cell", "split"])
                  .set_index("variant").round(4).to_string())


_P = _T = None


def _init(Ps, Ts):
    global _P, _T
    _P, _T = Ps, Ts


def _run_one(i):
    P, T = _P[i].astype(np.float64), _T[i].astype(np.float64)
    out = {}
    for vn, fn in _V.items():
        Q = fn(P)
        live = np.abs(P).sum(1) > 0                   # keep original dead rows dead
        Q = np.where(live[:, None] & live[None, :], Q, 0.0)
        out[vn] = evaluate(Q, T)
    return out


if __name__ == "__main__":
    main()
