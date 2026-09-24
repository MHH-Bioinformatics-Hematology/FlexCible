"""Score a checkpoint on the disjoint Akita test windows of a capture cache,
overall and split by pixels touching a capture bait.

Same metric as scripts/eval_val_windows.py (median over windows of the
per-window Pearson of predicted vs measured log(obs/exp) over the upper
triangle), computed three times per window:

  all       every upper-triangle pixel (reproduces eval_val_windows' oe score)
  bait      pixels whose row or column bin holds a baited fragment of the
            capture design (the pixels a promoter capture experiment measures)
  nonbait   all other pixels

The windows, their Hi-C target and the bait flags come from --cache (a cache
built with capture data, e.g. cache/full2k_capture), so models trained without
capture are scored on exactly the same windows and pixel sets. For a model
trained with capture.use_as_input, --capture present feeds the measured
capture map and bait mask as input planes; absent (default) feeds the absent
encoding (plane -1, mask 0), the inference default without capture data.
Models with a capture output channel also get the median per-window Pearson
of that channel against the capture target on bait pixels (capture_bait_r).

    python scripts/eval_capture_windows.py --run <output_dir> \
        --cache cache/full2k_capture [--capture present]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome.config import Config                                  # noqa: E402
from hicmultiome.dataset import split_cond                             # noqa: E402
from hicmultiome.model import (HiCModel, SeqEncoder, build_backbone,
                               n_cond_channels)                        # noqa: E402
from hicmultiome.train import (_coords, _select_rows, _stack,
                               capture_planes)                         # noqa: E402


def masked_r(p, t, m):
    a, b = p[m], t[m]
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() > 100 and a[f].std() > 1e-8 and b[f].std() > 1e-8:
        return float(np.corrcoef(a[f], b[f])[0, 1])
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--cache", required=True,
                    help="window cache with capture data (bait flags, capture target)")
    ap.add_argument("--capture", choices=["absent", "present"], default="absent")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", default=None,
                    help="JSON output (default <run>/capture_<split>_<capture>.json)")
    a = ap.parse_args()
    run = Path(a.run)
    ck = run / (run / a.ckpt).read_text().strip()
    cfg = Config.load(ck / "config.yaml")
    ck_ch = json.loads((ck / "channels.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capin = bool(ck_ch.get("capture_input"))
    if a.capture == "present" and not capin:
        raise SystemExit("--capture present needs a model trained with capture.use_as_input")

    if a.split == "test":
        data, ds = _stack(a.cache, "test", 0.0, False, cfg.train.seed, chroms=cfg.train.chroms)
        vi = np.arange(data["target"].shape[0])
    else:
        data, ds = _stack(a.cache, "train", 0.0, False, cfg.train.seed, chroms=cfg.train.chroms)
        vi = np.random.default_rng(cfg.train.seed).permutation(
            data["target"].shape[0])[: cfg.train.val_regions]
    if not ds.channels["has_capture"]:
        raise SystemExit(f"{a.cache} holds no capture data (no bait flags)")
    for k in ("tracks", "seq_features", "motif_ids"):
        if ck_ch[k] != ds.channels[k]:
            raise SystemExit(f"cache and checkpoint differ in {k}")
    tgt_all = data["target"][vi]
    bait = data["bait"][vi]
    rows, offset, n_motifs, n_sym = _select_rows(cfg, ds.channels)
    cond = data["cond"][vi][:, rows, :]

    if cfg.sequence.uses_encoder:
        raise SystemExit("sequence-encoder checkpoints are not supported here")
    n_out = 2 if ck_ch["has_capture"] else 1
    backbone = build_backbone(cfg.model.kind, n_cond_channels(
        n_sym, n_motifs, 2 if capin else 0, cfg.model.cond_mode), out_ch=n_out,
        base=cfg.model.base_channels, depth=cfg.model.depth)
    model = HiCModel(backbone, cfg.model.cond_mode, None,
                     cfg.window.size if cfg.model.dist_bias else 0).to(device)
    model.load_state_dict(torch.load(ck / "model.pt", map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(vi), 4):
            c = cond[i:i + 4].to(device).float()
            n = c.shape[0]
            sym, pairs = split_cond(c, offset, n_motifs)
            planes = None
            if capin:
                present = torch.full((n,), 1.0 if a.capture == "present" else 0.0,
                                     device=device)
                planes = capture_planes(tgt_all[i:i + 4].to(device).float(),
                                        bait[i:i + 4].to(device), present)
            preds.append(model(sym, pairs, planes, None).float().cpu().numpy())
    pred = np.concatenate(preds)

    W = pred.shape[-1]
    iu = np.triu(np.ones((W, W), dtype=bool), 1)
    per = []
    for k in range(len(vi)):
        b = bait[k].numpy() > 0
        touch = b[:, None] | b[None, :]
        t = np.clip(tgt_all[k, 0].float().numpy(), -1, 1)
        p = np.clip(pred[k, 0], -1, 1)
        row = {"window": int(vi[k]), "n_bait_bins": int(b.sum()),
               "r_all": masked_r(p, t, iu),
               "r_bait": masked_r(p, t, iu & touch),
               "r_nonbait": masked_r(p, t, iu & ~touch)}
        if n_out == 2:
            row["r_capture_bait"] = masked_r(
                pred[k, 1], tgt_all[k, 1].float().numpy(), iu & touch)
        per.append(row)
    ch, st, _ = _coords(ds)
    for row in per:
        row["chrom"], row["start_bin"] = ch[row["window"]], int(st[row["window"]])

    def med(key):
        v = np.array([r[key] for r in per], dtype=float)
        return float(np.nanmedian(v)), int(np.isfinite(v).sum())

    res = {"run": str(run), "ckpt": str(ck), "cache": a.cache, "split": a.split,
           "capture": a.capture if capin else "none (model has no capture input)",
           "n": len(vi)}
    for key, name in (("r_all", "oe_median_r"), ("r_bait", "bait_median_r"),
                      ("r_nonbait", "nonbait_median_r"),
                      ("r_capture_bait", "capture_channel_bait_median_r")):
        if key in per[0]:
            res[name], res[f"{name}_n"] = med(key)
    tri = W * (W - 1) / 2
    res["bait_pixel_fraction"] = float(np.mean(
        [(np.triu((bait[k].numpy() > 0)[:, None] | (bait[k].numpy() > 0)[None, :], 1)).sum()
         / tri for k in range(len(vi))]))
    print(json.dumps(res, indent=1))
    stem = f"capture_{a.split}_{a.capture if capin else 'none'}"
    out = Path(a.out or run / f"{stem}.json")
    out.write_text(json.dumps(res, indent=1))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)


if __name__ == "__main__":
    main()
