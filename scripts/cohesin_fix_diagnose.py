"""Diagnosis of the cohesin-loss response (registry section 55), CPU only.

Feeds the cached HCT116 test windows (Akita test regions, ENCODE-style 10 kb
cache of a run) to a trained model with the RAD21 row changed, and compares
the predicted change with the measured RAD21-degron change (auxin minus
untreated target of the same windows, even chromosomes: never trained under
auxin).

Scenarios (all on the untreated ChIP, motif channels unchanged; aux_<set>_only
  swaps just that set of tracks to the auxin ChIP; contrast=<f>: RAD21 minus its window
  median multiplied by f, the shape of the auxin track; scaled=<f>: every RAD21 value
  multiplied by f):
  aux_chip   the measured 6 h auxin ChIP of the degron cells
  s=<f>      the untreated RAD21 track multiplied by f in raw space
  absent     RAD21 at its absent value (the track-dropout / missing code)

    CUDA_VISIBLE_DEVICES= nice -n 10 python scripts/cohesin_fix_diagnose.py \
        --run all12deg_refres_drop_10kb [--out results/cohesin_fix/diagnose_<run>.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flexcible.config import Config                                    # noqa: E402
from flexcible.matrix import MatrixNormalizer                          # noqa: E402
from flexcible.predict import build_model                              # noqa: E402
from flexcible.tracks import TrackNormalizer                           # noqa: E402

HM = Path("/mnt/processed/wolffjoa/hic-multiome")


def box(x: np.ndarray, k: int = 5) -> np.ndarray:
    """k x k mean filter (NaN-aware), the local background for the high-pass part."""
    from scipy.ndimage import uniform_filter
    f = np.isfinite(x).astype(float)
    return uniform_filter(np.nan_to_num(x), k, mode="nearest") / np.maximum(
        uniform_filter(f, k, mode="nearest"), 1e-9)


def parts(d: np.ndarray, live: np.ndarray):
    """Split a change map into its per-diagonal mean (distance decay) and the
    remainder; the remainder into a 5x5 low-pass and a high-pass (loops, stripes)."""
    n = d.shape[0]
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    dd = np.zeros_like(d)
    for k in range(2, n):
        m = (dist == k) & live
        if m.any():
            dd[m] = d[m].mean()
    rest = np.where(live, d - dd, np.nan)
    lp = box(rest)
    return dd, lp, rest - lp


def pear(a, b, m):
    a, b = a[m], b[m]
    if a.size < 50 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. all12deg_refres_drop_10kb")
    ap.add_argument("--cache", default=None, help="cache dir with hct116 / hct116_auxin test shards")
    ap.add_argument("--scales", default="0.75,0.5,0.26,0.1,0")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)

    run = HM / "runs/multicell_reference" / a.run
    ck = run / (run / "best").read_text().strip()
    cfg = Config.load(ck / "config.yaml")
    model, channels = build_model(cfg, ck, "cpu")
    cache = Path(a.cache) if a.cache else HM / "cache_expanded" / ("mc_" + a.run)
    if not cache.exists():       # cache of the degron base (same windows)
        cache = HM / "cache_expanded" / ("mc_all12deg_refres_drop_" + a.run.rsplit("_", 1)[1])
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    sel_rows = None
    if cfg.train.select_tracks is not None or cfg.train.select_motifs is not None:
        from flexcible.train import _select_rows
        sel_rows, offset, n_motifs, _ = _select_rows(cfg, channels)
    mat = MatrixNormalizer.load(ck / "norm_matrix_hct116.json")
    tn = TrackNormalizer.load(ck / "norm_tracks.json")
    rad = channels["tracks"].index("RAD21")
    mean, std = tn.mean[rad], tn.std[rad]
    z_abs = float(tn.absent_values()[rad])
    coh_i = None
    if "cohesin_unknown" in channels["seq_features"]:
        coh_i = len(channels["tracks"]) + channels["seq_features"].index("cohesin_unknown")
        one = float(tn.apply(np.ones((len(tn.mean), 1), np.float32))[coh_i, 0])
    scales = [float(s) for s in a.scales.split(",")]
    frac_i = None
    if "cohesin_fraction" in channels["seq_features"]:
        frac_i = len(channels["tracks"]) + channels["seq_features"].index("cohesin_fraction")

    def load(cell):
        out = {}
        for f in sorted(cache.glob(f"{cell}_test_chr*.npz")):
            chrom = f.stem.split("_test_")[1]
            if int(chrom[3:]) % 2:
                continue                                  # odd: auxin was trained on them
            z = np.load(f)
            for i in range(len(z["start"])):
                out[(chrom, int(z["start"][i]))] = (z["cond"][i].astype(np.float32),
                                                    z["target"][i, 0].astype(np.float32),
                                                    z["ref"][i].astype(np.float32))
        return out

    U, A = load("hct116"), load("hct116_auxin")
    keys = sorted(set(U) & set(A))
    if a.max_windows:
        keys = keys[:a.max_windows]
    print(f"{a.run}: {len(keys)} even-chromosome test windows", flush=True)

    mat_aux = MatrixNormalizer.load(ck / "norm_matrix_hct116_auxin.json")
    print(f"matrix scale: hct116 {mat.scale:.3f}, hct116_auxin {mat_aux.scale:.3f}", flush=True)

    def predict(cond, ref, norm=None):
        c = cond if sel_rows is None else cond[sel_rows]
        c = torch.from_numpy(c[None])
        pairs = c[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, -1) if n_motifs else None
        with torch.no_grad():
            o = model(c, pairs, None, None, torch.from_numpy(ref[None]))
        P = (norm or mat).decode(o[0, 0].numpy())
        return 0.5 * (P + P.T)

    rows = []
    for ki, k in enumerate(keys):
        cu, tu, ru = U[k]
        ca, ta, ra = A[k]
        n = tu.shape[0]
        live = np.triu(np.ones((n, n), bool), 2)
        Tu, Ta = mat.decode(tu), mat.decode(ta)
        live &= np.isfinite(Tu) & np.isfinite(Ta) & (np.abs(Tu) + np.abs(Ta) > 0)
        if live.sum() < 200:
            continue
        P0 = predict(cu, ru)
        scen = {"aux_chip": ca, "aux_chip_auxscale": ca}
        tr = channels["tracks"]
        ctcf_i = tr.index("CTCF")
        hist = [i for i, t in enumerate(tr) if t not in ("CTCF", "RAD21", "SMC3")]
        for nm, idx in (("aux_RAD21_only", [rad]), ("aux_CTCF_only", [ctcf_i]),
                        ("aux_histones_only", hist),
                        ("aux_all_but_RAD21", [i for i in range(len(tr)) if i != rad])):
            c = cu.copy()
            c[idx] = ca[idx]
            scen[nm] = c
        for s in scales:
            c = cu.copy()
            raw = np.expm1(c[rad] * std + mean)
            med = np.median(raw)             # contrast around the window median (peaks shrink)
            c[rad] = (np.log1p(np.clip(med + s * (raw - med), 0, None)) - mean) / std
            scen[f"contrast={s:g}"] = c
            c = cu.copy()
            c[rad] = (np.log1p(s * raw) - mean) / std
            scen[f"scaled={s:g}"] = c        # every value multiplied (background too)
        c = cu.copy()
        c[rad] = z_abs
        if coh_i is not None:
            c[coh_i] = one
        scen["absent"] = c
        if coh_i is not None:
            c = c.copy()
            c[coh_i] = cu[coh_i]
            scen["absent_ind0"] = c
        if frac_i is not None:       # models with the cohesin:fraction row (section 55)
            fz = lambda f: float(tn.apply(np.full((len(tn.mean), 1), f, np.float32))[frac_i, 0])  # noqa: E731
            for f in (0.0, 0.26, 0.5, 0.75):
                c = cu.copy()
                c[frac_i] = fz(f)
                scen[f"wt_f{f:g}"] = c                       # untouched ChIP, only the row says less cohesin
            for f in (0.0, 0.5):
                c = scen["absent"].copy()
                c[frac_i] = fz(f)
                scen[f"absent_f{f:g}"] = c                   # RAD21 absent and the row says so
            c = scen["aux_RAD21_only"].copy()
            c[frac_i] = fz(0.26)
            scen["aux_RAD21_only_f0.26"] = c
            c = scen["aux_chip"].copy()
            c[frac_i] = fz(0.26)
            scen["aux_chip_f0.26"] = c
            c = scen["contrast=0.26"].copy()
            c[frac_i] = fz(0.26)
            scen["contrast=0.26_f0.26"] = c
        dM = Ta - Tu
        dm_dd, dm_lp, dm_hp = parts(dM, live)
        for name, c in scen.items():
            if name == "aux_chip_auxscale":   # the training view of the auxin sample: its own scale
                dP = predict(c, ra, mat_aux) - P0
            else:
                dP = predict(c, ru) - P0
            dp_dd, dp_lp, dp_hp = parts(dP, live)
            L = live & np.isfinite(dm_hp) & np.isfinite(dp_hp)
            d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            rows.append({
                "window": f"{k[0]}:{k[1]}", "scenario": name,
                "pearson": pear(dP, dM, live),
                "pearson_distance": pear(dp_dd, dm_dd, live),
                "pearson_lowpass": pear(np.nan_to_num(dp_lp), np.nan_to_num(dm_lp), L),
                "pearson_highpass": pear(np.nan_to_num(dp_hp), np.nan_to_num(dm_hp), L),
                "meas_mean": float(dM[live].mean()), "pred_mean": float(dP[live].mean()),
                "meas_mean_le100kb": float(dM[live & (d * cfg.data.resolution <= 1e5)].mean()),
                "pred_mean_le100kb": float(dP[live & (d * cfg.data.resolution <= 1e5)].mean()),
                "meas_mean_gt250kb": float(dM[live & (d * cfg.data.resolution > 2.5e5)].mean()),
                "pred_mean_gt250kb": float(dP[live & (d * cfg.data.resolution > 2.5e5)].mean()),
                "meas_sd": float(dM[live].std()), "pred_sd": float(dP[live].std()),
                "meas_sd_highpass": float(np.nanstd(dm_hp[L])), "pred_sd_highpass": float(np.nanstd(dp_hp[L])),
                "meas_sd_distance": float(dm_dd[live].std()), "pred_sd_distance": float(dp_dd[live].std()),
            })
        if (ki + 1) % 10 == 0:
            print(f"  {ki + 1}/{len(keys)}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else REPO / "results/cohesin_fix" / f"diagnose_{a.run}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    g = df.groupby("scenario", sort=False).median(numeric_only=True).round(3)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(g.to_string())


if __name__ == "__main__":
    main()
