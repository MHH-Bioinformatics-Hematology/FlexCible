"""Re-score a checkpoint on the held-out validation windows of its training run
(same cache, chroms, seed and val_regions -> the identical windows), or with
--split test on every test-region window of its chromosomes, in two spaces:

  oe       median per-region Pearson of log(obs/exp) over the upper triangle --
           the training/early-stopping metric (distance-free target)
  logobs   the same after adding the distance decay back to BOTH prediction
           and target: log expected(d) + per-diagonal offset(d) of the
           window's chromosome (scripts/expected_profile.py). DECAY-INFLATED:
           the shared distance trend alone correlates strongly, which is why
           the distance-only row is reported next to it.

    python scripts/eval_val_windows.py --run <output_dir> \
        --expected runs/motif_grid/expected_gm12878_2kb.npz [--ckpt best]
"""
from __future__ import annotations

import argparse
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
from hicmultiome.train import (_coords, _fasta_map, _select_rows, _stack,
                               batch_onehot)                           # noqa: E402


def region_r(p, t):
    iu = np.triu_indices(p.shape[-1], 1)
    a, b = p[iu], t[iu]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() > 100 and a[m].std() > 1e-8 and b[m].std() > 1e-8:
        return float(np.corrcoef(a[m], b[m])[0, 1])
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--expected", default=None,
                    help="expected-profile npz for the log-obs rows; without it "
                         "only the O/E score is computed")
    ap.add_argument("--sample", default=None,
                    help="multi-sample caches: score only this sample's windows "
                         "(default: all windows pooled)")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="val: the run's held-out validation windows (they overlap "
                         "training windows by up to half a window); test: all "
                         "test-region windows on the run's chromosomes (disjoint)")
    ap.add_argument("--out", default=None, help="JSON output (default <run>/<split>_eval.json)")
    ap.add_argument("--cache-dir", default=None,
                    help="score on this cache's windows and targets instead of the "
                         "run's own (same bin grid and inputs, e.g. another target "
                         "map or the half-stride cache of a quarter-stride run). "
                         "--split val on another cache is only honest when its "
                         "window list equals the training cache's")
    ap.add_argument("--reference-only", action="store_true",
                    help="score the cached reference map (data.reference_cool) "
                         "itself as the prediction, on the same windows and pixels")
    a = ap.parse_args()
    run = Path(a.run)
    ck = run / (run / a.ckpt).read_text().strip()
    cfg = Config.load(ck / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = a.cache_dir or cfg.train.cache_dir

    if a.split == "val":
        data, ds = _stack(cache_dir, "train", 0.0, False, cfg.train.seed,
                          chroms=cfg.train.chroms)
        from hicmultiome.train import split_val_regions
        vi, _ = split_val_regions(_coords(ds), cfg.train.val_regions, ds.window_size,
                                  cfg.train.seed, cfg.train.val_exclude_overlap)
    else:   # every test window (held-out regions, disjoint from all training windows)
        data, ds = _stack(cache_dir, "test", 0.0, False, cfg.train.seed,
                          chroms=cfg.train.chroms)
        vi = np.arange(data["target"].shape[0])
    use_ref = cfg.model.reference != "none"
    if (use_ref or a.reference_only) and "ref" not in data:
        raise SystemExit(f"{cache_dir} holds no reference windows")
    if use_ref or a.reference_only:
        ds.check_reference_leakage()
    ref = data["ref"][vi] if "ref" in data else None
    if a.sample is not None:
        smp = _coords(ds)[2]
        vi = np.array([i for i in vi if smp[i] == a.sample], dtype=np.int64)
        if not len(vi):
            raise SystemExit(f"no {a.split} windows for sample {a.sample!r}")
    tgt = data["target"][vi][:, 0].numpy()
    cond = data["cond"][vi]
    ch, st, sm = _coords(ds)
    vcoords = ([ch[i] for i in vi], st[vi], [sm[i] for i in vi])
    rows, offset, n_motifs, n_sym = _select_rows(cfg, ds.channels)
    cond = cond[:, rows, :]

    use_seq = cfg.sequence.uses_encoder
    emb = cfg.sequence.embed_dim if use_seq else 0
    W = ds.window_size
    backbone = build_backbone(cfg.model.kind, n_cond_channels(
                              n_sym + emb, n_motifs, cfg.model.reference_planes,
                              cfg.model.cond_mode), out_ch=ds.n_out,
                              base=cfg.model.base_channels, depth=cfg.model.depth)
    enc = SeqEncoder(embed_dim=emb, width=cfg.sequence.enc_width) if use_seq else None
    model = HiCModel(backbone, cfg.model.cond_mode, enc,
                     cfg.window.size if cfg.model.dist_bias else 0,
                     reference=cfg.model.reference).to(device)
    model.load_state_dict(torch.load(ck / "model.pt", map_location=device))
    model.eval()
    fm = _fasta_map(cfg)
    preds = []
    if a.reference_only:
        pred = ref.float().numpy()
    else:
        with torch.no_grad():
            for i in range(0, len(vi), 4):
                c = cond[i:i + 4].to(device)
                sym, pairs = split_cond(c, offset, n_motifs)
                n = c.shape[0]
                oh = (batch_onehot(vcoords, list(range(i, i + n)), [False] * n, W,
                                   int(cfg.data.resolution), fm, device) if use_seq else None)
                rb = ref[i:i + n].to(device).float() if use_ref else None
                preds.append(model(sym, pairs, None, oh, rb)[:, 0].float().cpu().numpy())
        pred = np.concatenate(preds)

    nm = Path(cache_dir) / f"norm_matrix_{a.sample}.json"
    if a.sample is None or not nm.exists():
        nm = next(Path(cache_dir).glob("norm_matrix_*.json"))
    scale = json.loads(nm.read_text())["scale"]
    E = np.load(a.expected) if a.expected else None
    idx = np.arange(W)
    D = np.abs(idx[:, None] - idx[None, :])
    r_oe, r_lo, r_dist = [], [], []
    for k in range(len(vi)):
        c = vcoords[0][k]
        t_oe = np.clip(tgt[k], -1, 1) * scale
        p_oe = np.clip(pred[k], -1, 1) * scale
        r_oe.append(region_r(p_oe, t_oe))
        if E is None:
            continue
        decay = np.log(E[f"{c}:exp"])[D] + E[f"{c}:offset"][D]
        r_lo.append(region_r(p_oe + decay, t_oe + decay))
        r_dist.append(region_r(decay, t_oe + decay))
    res = {"run": str(run), "ckpt": str(ck), "split": a.split, "n": len(vi),
           "oe_median_r": float(np.nanmedian(r_oe)), "cache_dir": str(cache_dir),
           "prediction": "reference map" if a.reference_only else "model"}
    if a.sample is not None:
        res["sample"] = a.sample
    if E is not None:
        res["logobs_median_r_decay_inflated"] = float(np.nanmedian(r_lo))
        res["distance_only_logobs_median_r"] = float(np.nanmedian(r_dist))
    print(json.dumps(res, indent=1))
    Path(a.out or run / (f"{a.split}_eval.json" if a.sample is None
                         else f"{a.split}_eval_{a.sample}.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
