"""How much Hi-C information the capture input carries, and how much a
capture-input model (capture_experiment.py variant B) changes its prediction
when the capture map is given.

On the test windows of a capture cache:
  - mean encoded Hi-C O/E at bait pixels with a significant capture
    interaction vs the other bait pixels (offset >= 2 bins), and the median
    per-window Pearson between the capture plane and Hi-C O/E on bait pixels;
  - for each --run: mean |prediction(capture given) - prediction(absent)| on
    bait pixels, other pixels and significant capture pixels, in the encoded
    target scale, next to the standard deviation of the target.
Runs on CPU (--device) so it can run while the GPUs train.

    python scripts/capture_input_diagnostics.py --cache cache/full2k_capture \
        --runs runs/capture/B__seed0 runs/capture/B__seed1 --out diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config                                  # noqa: E402
from flexcible.dataset import split_cond                             # noqa: E402
from flexcible.model import HiCModel, build_backbone, n_cond_channels  # noqa: E402
from flexcible.train import _select_rows, _stack, capture_planes     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--n-windows", type=int, default=48,
                    help="windows with the most bait bins used for the model check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, ds = _stack(a.cache, "test", 0.0, False, 0)
    t = data["target"].numpy()
    b = data["bait"].numpy() > 0
    W = t.shape[-1]
    idx = np.arange(W)
    far = np.abs(idx[:, None] - idx[None, :]) >= 2
    pos, neg, rs, frac = [], [], [], []
    for k in range(len(t)):
        touch = (b[k][:, None] | b[k][None, :]) & far
        if not touch.any():
            continue
        cap, oe = t[k, 1][touch], t[k, 0][touch]
        p = cap > -1
        frac.append(p.mean())
        if p.sum() > 5:
            pos.append(oe[p].mean())
            neg.append(oe[~p].mean())
        if p.sum() > 20 and cap.std() > 0:
            rs.append(np.corrcoef(cap, oe)[0, 1])
    res = {"cache": a.cache, "n_windows": len(t),
           "target_oe_std_encoded": float(t[:, 0].std()),
           "significant_fraction_of_bait_pixels": float(np.mean(frac)),
           "oe_at_significant_capture_pixels": float(np.mean(pos)),
           "oe_at_other_bait_pixels": float(np.mean(neg)),
           "median_window_r_capture_vs_oe_bait_pixels": float(np.median(rs)),
           "n_windows_r": len(rs), "runs": {}}
    sel = np.argsort(-b.sum(1))[: a.n_windows]
    dev = torch.device(a.device)
    for r in a.runs:
        run = Path(r)
        ck = run / (run / "best").read_text().strip()
        cfg = Config.load(ck / "config.yaml")
        rows, off, nm, ns = _select_rows(cfg, ds.channels)
        m = HiCModel(build_backbone(cfg.model.kind, n_cond_channels(
            ns, nm, 2, cfg.model.cond_mode), out_ch=2, base=cfg.model.base_channels,
            depth=cfg.model.depth), cfg.model.cond_mode, None, 0).to(dev)
        m.load_state_dict(torch.load(ck / "model.pt", map_location=dev))
        m.eval()
        pa, pp = [], []
        with torch.no_grad():
            for i in range(0, len(sel), 4):
                s = sel[i:i + 4]
                c = data["cond"][s][:, rows].float().to(dev)
                sym, pairs = split_cond(c, off, nm)
                tg, bt = data["target"][s].float().to(dev), data["bait"][s].to(dev)
                n = len(s)
                pa.append(m(sym, pairs, capture_planes(tg, bt, torch.zeros(n, device=dev)),
                            None)[:, 0].cpu().numpy())
                pp.append(m(sym, pairs, capture_planes(tg, bt, torch.ones(n, device=dev)),
                            None)[:, 0].cpu().numpy())
        pa, pp = np.concatenate(pa), np.concatenate(pp)
        bb = b[sel]
        touch = bb[:, :, None] | bb[:, None, :]
        sig = t[sel, 1] > -1
        d = np.abs(pp - pa)
        res["runs"][str(run)] = {
            "n_windows": len(sel),
            "mean_abs_change_bait_pixels": float(d[touch].mean()),
            "mean_abs_change_other_pixels": float(d[~touch].mean()),
            "mean_abs_change_significant_capture_pixels": float(d[sig].mean()),
            "mean_signed_change_significant_capture_pixels": float((pp - pa)[sig].mean()),
        }
    print(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
