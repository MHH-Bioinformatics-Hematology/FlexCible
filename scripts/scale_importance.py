"""Scale-dependent input importance of a model, per resolution (analysis C).

For each per-resolution view of a (multi-resolution) run, on every disjoint
test window of the 12 training cells (its cache's test split):

drop        the median per-window Pearson (upper triangle, the metric of
            eval_val_windows) with all inputs, and with one input group set to
            its absent value (the normalised value of a raw-zero track, as in
            track dropout): CTCF, cohesin (RAD21, SMC3), active (H3K27ac,
            H3K4me1/2/3, H3K9ac), gene body (H3K36me3, H3K79me2, H4K20me1),
            repressive (H3K27me3, H3K9me3), motifs (4 motifs, both strands),
            sequence (GC, CpG) and, for reference-map models, the reference
            plane set to 0 (the expected level). Median over windows per cell,
            then median over cells; loss = full minus dropped.
attribution gradient x input on a seeded sample of windows (20 per cell): the
            scalar per distance band is sum(pred * z(target)) over the band's
            upper-triangle pixels (z: target standardised in the band), i.e.
            the agreement of prediction and target there; per group the sum
            of |grad * (input - absent)| over its rows and positions, the
            reference plane |grad * ref|; shares of the total per window,
            averaged. Bands < 100 kb, 100 to 500 kb, > 500 kb.

Outputs <out>/importance_drop.csv, importance_drop_cells.csv,
importance_attr.csv and importance.png (heatmaps groups x resolutions).

    python scripts/scale_importance.py --views runs/.../all12multires \
        runs/.../all12multires_5kb ... --out runs/.../scale_importance
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome.config import Config                                  # noqa: E402
from hicmultiome.dataset import split_cond                             # noqa: E402
from hicmultiome.model import make_model, n_cond_channels              # noqa: E402
from hicmultiome.tracks import TrackNormalizer                         # noqa: E402
from hicmultiome.train import _coords, _select_rows, _stack            # noqa: E402

GROUPS = {
    "CTCF": ["CTCF"],
    "cohesin": ["RAD21", "SMC3"],
    "active": ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac"],
    "gene body": ["H3K36me3", "H3K79me2", "H4K20me1"],
    "repressive": ["H3K27me3", "H3K9me3"],
}
BANDS = [("<100 kb", 0, 100_000), ("100-500 kb", 100_000, 500_000),
         (">500 kb", 500_000, 10**12)]


def region_r(p, t):
    iu = np.triu_indices(p.shape[-1], 1)
    a, b = p[iu], t[iu]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() > 100 and a[m].std() > 1e-8 and b[m].std() > 1e-8:
        return float(np.corrcoef(a[m], b[m])[0, 1])
    return float("nan")


def group_rows(names: list[str]) -> dict[str, list[int]]:
    g = {k: [i for i, n in enumerate(names) if n in v] for k, v in GROUPS.items()}
    g["motifs"] = [i for i, n in enumerate(names) if n.startswith("motif:")]
    g["sequence"] = [i for i, n in enumerate(names) if n.startswith("seq:")]
    return {k: v for k, v in g.items() if v}


def load(view: Path, device):
    ck = view / (view / "best").read_text().strip()
    cfg = Config.load(ck / "config.yaml")
    data, ds = _stack(cfg.train.cache_dir, "test", 0.0, False, cfg.train.seed)
    rows, offset, n_motifs, n_sym = _select_rows(cfg, ds.channels)
    names = [ds.channels["sym_rows"][r] for r in rows]
    model = make_model(cfg, n_cond_channels(n_sym, n_motifs, cfg.model.reference_planes,
                                            cfg.model.cond_mode), ds.n_out).to(device)
    model.load_state_dict(torch.load(ck / "model.pt", map_location=device))
    model.eval()
    absent = torch.from_numpy(TrackNormalizer.load(ck / "norm_tracks.json")
                              .absent_values()[rows].astype(np.float32))
    return cfg, data, ds, rows, offset, n_motifs, names, model, absent


def predict(model, cond, ref, offset, n_motifs, device, res, batch=8):
    out = []
    with torch.no_grad():
        for i in range(0, cond.shape[0], batch):
            c = cond[i:i + batch].to(device).float()
            sym, pairs = split_cond(c, offset, n_motifs)
            r = ref[i:i + batch].to(device).float() if ref is not None else None
            out.append(model(sym, pairs, None, None, r, res=res)[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def run(views, out: Path, n_attr: int, device):
    out.mkdir(parents=True, exist_ok=True)
    drop_rows, cell_rows, attr_rows = [], [], []
    for view in views:
        cfg, data, ds, rows, offset, n_motifs, names, model, absent = load(Path(view), device)
        res = int(cfg.data.resolution)
        W = ds.window_size
        cond = data["cond"][:, rows, :]
        tgt = data["target"][:, 0].float().numpy()
        ref = data.get("ref")
        cells = np.array(_coords(ds)[2])
        groups = group_rows(names)
        conds = {"all inputs": None, **{k: k for k in groups}}
        if ref is not None:
            conds["reference map"] = "ref"
        scores = {}
        for label, g in conds.items():
            c, r = cond, ref
            if g == "ref":
                r = torch.zeros_like(ref)
            elif g is not None:
                c = cond.clone()
                c[:, groups[g], :] = absent[groups[g]][None, :, None].to(c.dtype)
            p = predict(model, c, r, offset, n_motifs, device, res)
            rr = np.array([region_r(p[k], tgt[k]) for k in range(len(p))])
            per_cell = pd.Series(rr).groupby(cells).median()
            scores[label] = per_cell
            for cell, v in per_cell.items():
                cell_rows.append({"res_kb": res // 1000, "dropped": label, "cell": cell,
                                  "median_r": round(float(v), 4)})
            print(f"{res // 1000} kb  {label:14s} {float(per_cell.median()):.4f}", flush=True)
        full = scores["all inputs"]
        for label, s in scores.items():
            drop_rows.append({"res_kb": res // 1000, "dropped": label,
                              "median_over_cells": round(float(s.median()), 4),
                              "loss_vs_all_inputs": round(float((full - s).median()), 4),
                              "n_windows": len(cells)})
        # gradient x input on a seeded sample, 20 windows per cell
        rng = np.random.default_rng(0)
        pick = np.concatenate([rng.choice(np.nonzero(cells == c)[0],
                                          min(n_attr, int((cells == c).sum())), replace=False)
                               for c in np.unique(cells)])
        idx = np.arange(W)
        D = np.abs(idx[:, None] - idx[None, :]) * res
        up = np.triu(np.ones((W, W), bool), 1)
        acc = {b[0]: {g: [] for g in list(groups) + (["reference map"] if ref is not None else [])}
               for b in BANDS}
        for k in pick:
            c = cond[k:k + 1].to(device).float().requires_grad_(True)
            r = (ref[k:k + 1].to(device).float().requires_grad_(True)
                 if ref is not None else None)
            sym, pairs = split_cond(c, offset, n_motifs)
            p = model(sym, pairs, None, None, r, res=res)[0, 0]
            t = tgt[k]
            for bname, lo, hi in BANDS:
                m = up & (D >= lo) & (D < hi)
                if m.sum() < 50:
                    continue
                z = (t[m] - t[m].mean()) / (t[m].std() + 1e-8)
                zt = torch.zeros(W, W, device=device)
                zt[torch.from_numpy(m).to(device)] = torch.from_numpy(z.astype(np.float32)).to(device)
                s = (p * zt).sum()
                ins = [c] + ([r] if r is not None else [])
                grads = torch.autograd.grad(s, ins, retain_graph=True)
                gc, gr = grads[0], (grads[1] if r is not None else None)
                a = (gc * (c - absent.to(device)[None, :, None])).abs()[0].sum(1).detach()
                vals = {g: float(a[rows_].sum()) for g, rows_ in groups.items()}
                if r is not None:
                    vals["reference map"] = float((gr * r).abs().sum().detach())
                tot = sum(vals.values()) or 1.0
                for g, v in vals.items():
                    acc[bname][g].append(v / tot)
        for bname, d in acc.items():
            for g, v in d.items():
                if v:
                    attr_rows.append({"res_kb": res // 1000, "band": bname, "group": g,
                                      "share": round(float(np.mean(v)), 4), "n_windows": len(v)})
        del data, cond, ref, model
        torch.cuda.empty_cache()
    for fn, rows_ in (("importance_drop.csv", drop_rows), ("importance_drop_cells.csv", cell_rows),
                      ("importance_attr.csv", attr_rows)):
        pd.DataFrame(rows_).to_csv(out / fn, index=False)
    figure(out)


def figure(out: Path):
    """Heatmaps: score loss per dropped group (diverging blue - gray - red,
    centred on 0) and gradient x input share per band (one-hue ramp)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    ink, muted = "#1f1f1e", "#6b6a66"
    div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#2a78d6"])
    d = pd.read_csv(out / "importance_drop.csv")
    d = d[d.dropped != "all inputs"]
    M = d.pivot(index="dropped", columns="res_kb", values="loss_vs_all_inputs")
    order = list(GROUPS) + ["motifs", "sequence", "reference map"]
    M = M.reindex([g for g in order if g in M.index])
    M = M[sorted(M.columns)]
    a = pd.read_csv(out / "importance_attr.csv")
    bands = [b[0] for b in BANDS]
    fig, axes = plt.subplots(1, 1 + len(bands), figsize=(4 + 3 * len(bands), 4.2),
                             gridspec_kw={"width_ratios": [1.3] + [1] * len(bands)})
    lim = max(abs(M.values).max(), 1e-3)
    ax = axes[0]
    ax.imshow(M.values, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(M.shape[1]), [f"{c} kb" for c in M.columns])
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.set_title("score loss when dropped", color=ink, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:+.3f}", ha="center", va="center", fontsize=7, color=ink)
    for ax, b in zip(axes[1:], bands):
        S = a[a.band == b].pivot(index="group", columns="res_kb", values="share")
        S = S.reindex(M.index)
        S = S[sorted(S.columns)]
        ax.imshow(S.values, cmap=seq, vmin=0, vmax=max(a.share.max(), 1e-3), aspect="auto")
        ax.set_xticks(range(S.shape[1]), [f"{c} kb" for c in S.columns])
        ax.set_yticks(range(S.shape[0]), S.index if ax is axes[1] else [""] * S.shape[0])
        ax.set_title(f"attribution share, {b}", color=ink, fontsize=10)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                v = S.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=ink)
    for ax in axes:
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    tmp = out / "importance.part.png"
    fig.savefig(tmp, dpi=160, facecolor="#fcfcfb")
    tmp.rename(out / "importance.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attr-per-cell", type=int, default=20)
    ap.add_argument("--figure-only", action="store_true")
    a = ap.parse_args()
    if a.figure_only:
        figure(Path(a.out))
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(a.views, Path(a.out), a.attr_per_cell, device)
    (Path(a.out) / "views.json").write_text(json.dumps(a.views, indent=1))
    print("SCALE_IMPORTANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
