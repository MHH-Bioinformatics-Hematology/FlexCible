"""Greedy minimal-input search on a track-dropout-trained model.

Because the model was trained seeing random track subsets (absent = the
raw-zero encoding), any subset can be evaluated at inference without
retraining. Greedy forward selection: start from no assays (sequence/motif
channels stay on -- they cost nothing experimentally), each round add the
track that best improves the score on the search chromosome, stop when the
gain drops below --eps. The result is the assay cost/performance curve.

Usage:
  python scripts/minimal_set.py --config configs/dropout_25kb.yaml \
      --mode hic --chrom 2 --out runs/dropout_25kb/minimal_set.json
  python scripts/minimal_set.py --config ../pchic-unet/configs/gm12878_25kb.dropout.yaml \
      --mode pchic --chrom 2 --out .../minimal_set.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "src/pchic-unet"))

from flexcible.config import Config
from flexcible.evaluate import scc
from flexcible.matrix import fetch_chrom_matrix


def score_hic(cfg, chrom, zero_tracks, tmp):
    from flexcible.predict import predict

    out = tmp / "pred.cool"
    predict(cfg, chrom, str(out), batch=16, zero_tracks=zero_tracks or None)
    truth = fetch_chrom_matrix(cfg.get_sample(None).cool_path, chrom)
    pred = fetch_chrom_matrix(str(out), chrom)
    n = min(truth.shape[0], pred.shape[0])
    out.unlink()
    return scc(pred[:n, :n], truth[:n, :n], max_dist_bins=cfg.window.size, h=1)


def score_pchic(cfg, chrom, zero_tracks, tmp):
    import dataclasses

    from pchicunet.evaluate import evaluate
    from pchicunet.predict import predict as ppredict

    sample = cfg.get_sample(None)
    cfg2 = cfg
    if zero_tracks:
        s2 = dataclasses.replace(
            sample, zeroed_tracks=sorted(set(sample.zeroed_tracks) | set(zero_tracks)))
        cfg2 = dataclasses.replace(cfg, data=dataclasses.replace(
            cfg.data, samples=[s2]))
    out = tmp / "pred_cap.cool"
    ppredict(cfg2, chrom, str(out), batch=16)
    res = evaluate(cfg2, str(out), chrom)
    out.unlink()
    # interaction-level strength correlation is the objective; the
    # distance-matched AUROC is logged so specificity stays visible
    print(f"      [pchic] strength_rho={res['mean_strength_spearman']:.4f} "
          f"AUROC={res['mean_auroc_vs_distance_matched']:.4f}", flush=True)
    return res["mean_strength_spearman"]


def main():
    ap = argparse.ArgumentParser(description="Greedy minimal assay-set search.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["hic", "pchic"], required=True)
    ap.add_argument("--chrom", default="2", help="held-out search chromosome")
    ap.add_argument("--eps", type=float, default=0.003,
                    help="stop when the best gain drops below this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    all_tracks = cfg.track_names
    scorer = score_hic if args.mode == "hic" else score_pchic
    tmp = Path(tempfile.mkdtemp(prefix="minset_"))

    full = scorer(cfg, args.chrom, [], tmp)
    print(f"ALL {len(all_tracks)} assays: {full:.4f}", flush=True)
    none = scorer(cfg, args.chrom, list(all_tracks), tmp)
    print(f"NO assays (sequence/motifs only): {none:.4f}", flush=True)

    chosen: list[str] = []
    curve = [{"n": 0, "tracks": [], "score": none}]
    best_prev = none
    while len(chosen) < len(all_tracks):
        results = {}
        for t in all_tracks:
            if t in chosen:
                continue
            zero = [x for x in all_tracks if x not in chosen + [t]]
            results[t] = scorer(cfg, args.chrom, zero, tmp)
        t_best = max(results, key=results.get)
        gain = results[t_best] - best_prev
        print(f"round {len(chosen)+1}: +{t_best} -> {results[t_best]:.4f} "
              f"(gain {gain:+.4f})", flush=True)
        if gain < args.eps:
            print("gain below eps; stopping")
            break
        chosen.append(t_best)
        best_prev = results[t_best]
        curve.append({"n": len(chosen), "tracks": list(chosen),
                      "score": best_prev, "round_scores": results})

    summary = {
        "mode": args.mode, "chrom": args.chrom,
        "score_all": full, "score_none": none,
        "minimal_set": chosen, "curve": curve,
        "fraction_of_full": (best_prev - none) / max(full - none, 1e-9),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nminimal set ({len(chosen)} assays): {chosen}")
    print(f"reaches {best_prev:.4f} vs full {full:.4f} "
          f"({100*summary['fraction_of_full']:.0f}% of the assay-driven gain)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
