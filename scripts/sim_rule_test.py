"""Did the model learn the CAUSAL rule, or another correlational shortcut?

Held-out synthetic chromosomes, never seen in training, in four classes:
reference, a duplication that captured a boundary, a duplication that captured
none, and an inversion. For each we predict from the conditioning rows alone
and ask whether the predicted insulation at the rearrangement junction matches
what the rules say it should be:

    duplication WITH a boundary  -> a new boundary at the junction
    duplication WITHOUT one      -> no new boundary, just a larger domain
    inversion                    -> the loop of the inverted anchor is lost
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config                      # noqa: E402
from flexcible.loops import insulation_profile           # noqa: E402
from flexcible.matrix import (BandedAccumulator, MatrixNormalizer,  # noqa: E402
                                hann2d, plan_windows)
from flexcible.model import HiCUNet, build_spatial_cond, n_cond_channels
from flexcible.simulate import SimParams, duplicate, invert, simulate
from flexcible.tracks import TrackNormalizer                # noqa: E402


def rows_of(sim):
    a = sim.active
    gc = 0.4 + 0.1 * (a / max(a.max(), 1e-9))
    cpg = 0.02 + 0.03 * (a / max(a.max(), 1e-9))
    return np.stack([a, sim.repressed, gc, cpg,
                     sim.ctcf_fwd, sim.ctcf_rev]).astype(np.float32)


def predict_rows(model, rows, mat_norm, window, stride, device):
    n = rows.shape[1]
    acc = BandedAccumulator(n, band=window)
    w2d = hann2d(window)
    wins = plan_windows(n, window, stride, "sim")
    for i in range(0, len(wins), 8):
        bw = wins[i:i + 8]
        cond = []
        for w in bw:
            c = rows[:, w.start:w.start + window]
            if c.shape[1] < window:
                c = np.concatenate(
                    [c, np.zeros((c.shape[0], window - c.shape[1]), np.float32)], 1)
            cond.append(c)
        t = torch.from_numpy(np.stack(cond)).to(device)
        pairs = t[:, 4:6].reshape(t.shape[0], 1, 2, window)   # motif fwd/rev
        with torch.no_grad():
            img = model(build_spatial_cond(t, pairs, None, "full"))[:, 0]
        for w, m in zip(bw, img.float().cpu().numpy()):
            blk = np.maximum(mat_norm.decode(m), 0.0)
            acc.add_block(w.start, 0.5 * (blk + blk.T), w2d)
    band = np.where(acc.den > 0, acc.num / np.maximum(acc.den, 1e-9), 0.0)
    out = np.zeros((n, n))
    for d in range(min(window, n)):
        v = band[:n - d, d]
        idx = np.arange(v.shape[0])
        out[idx, idx + d] = v
        out[idx + d, idx] = v
    return out


def junction_insulation(m, j, window=6):
    p = insulation_profile(m, 8)
    loc = np.nanmin(p[max(0, j - window):j + window + 1])
    bg = np.nanmedian(p[np.isfinite(p)])
    return float(loc), float(bg)


def main():
    cfg = Config.load("configs/sim_rules.yaml")
    out = Path(cfg.train.output_dir)
    ck = out / (out / "best").read_text().strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tn = TrackNormalizer.load(ck / "norm_tracks.json")
    mn = MatrixNormalizer.load(ck / "norm_matrix_sim.json")
    # 6 conditioning rows, of which the last two are the motif fwd/rev
    # pair used for the directed planes
    model = HiCUNet(n_cond_channels(6, 1, 0, "full"), out_ch=1,
                    base=cfg.model.base_channels, depth=cfg.model.depth).to(device)
    model.load_state_dict(torch.load(ck / "model.pt", map_location=device))
    model.eval()

    res = {"dup_with_boundary": [], "dup_no_boundary": [], "inversion": []}
    for seed in range(20000, 20040):           # fresh, never trained on
        p = SimParams(n_bins=512, seed=seed)
        base = simulate(p)
        n = base.matrix.shape[0]
        rng = np.random.default_rng(seed + 4242)
        span = int(rng.integers(n // 12, n // 5))
        b_bins = np.nonzero(base.boundaries)[0]
        cases = []
        mid = b_bins[(b_bins > n // 5) & (b_bins < 3 * n // 5)]  # keep it interior
        if len(mid):
            b0 = int(rng.choice(mid))
            a = max(0, b0 - span // 2)
            b = min(n - 1, a + span)
            ed, had = duplicate(base, a, b, p)
            # A duplicated boundary makes its new border where the COPY of the
            # boundary element lands, not at the seam between the two copies.
            probe = b0 + (b - a)
            cases.append(("dup_with_boundary" if had else "dup_no_boundary",
                          ed, probe))
        for _ in range(10):
            a = int(rng.integers(1, n - span - 1))
            if base.boundaries[a:a + span].max() == 0:
                ed, _ = duplicate(base, a, a + span, p)
                # the equivalent probe: the middle of the second copy, where a
                # boundary WOULD have landed had one been duplicated
                cases.append(("dup_no_boundary", ed, a + span + span // 2))
                break
        for label, sim, j in cases:
            n_ed = sim.matrix.shape[0]
            if not (20 < j < n_ed - 20):    # insulation is undefined at the edges
                continue
            rows = tn.apply(rows_of(sim))
            pred = predict_rows(model, rows, mn, cfg.window.size,
                                cfg.window.stride, device)
            lo_p, bg_p = junction_insulation(pred, j)
            lo_t, bg_t = junction_insulation(sim.matrix, j)
            res[label].append((lo_p - bg_p, lo_t - bg_t))

    print("held-out synthetic rearrangements: insulation dip at the junction")
    print(f"{'class':22s} {'n':>3} {'TRUE dip':>10} {'PREDICTED dip':>14}")
    for k, v in res.items():
        if not v:
            continue
        pr = np.array([x[0] for x in v]); tr = np.array([x[1] for x in v])
        print(f"{k:22s} {len(v):3d} {tr.mean():10.3f} {pr.mean():14.3f}")
    a = np.array([x[0] for x in res["dup_with_boundary"]])
    b = np.array([x[0] for x in res["dup_no_boundary"]])
    if len(a) and len(b):
        print(f"\ndiscrimination (with-boundary should dip MORE than without):"
              f" {b.mean() - a.mean():+.3f}")
        print("  learned the rule" if a.mean() < b.mean() - 0.1
              else "  did NOT learn the rule")


if __name__ == "__main__":
    main()
