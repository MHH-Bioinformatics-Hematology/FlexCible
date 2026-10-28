"""Leave-one-protein-out at inference: zero each track genome-wide on chr2,
re-predict with the trained chip_only_25kb model, score SCC vs truth.

Measures how much the model relies on each protein without retraining
(inference-time marginal importance)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexcible.config import Config
from flexcible.evaluate import scc
from flexcible.matrix import fetch_chrom_matrix
from flexcible.predict import predict

cfg = Config.load("configs/chip_only_25kb.yaml")
truth = fetch_chrom_matrix(cfg.get_sample(None).cool_path, "2")
out_dir = Path("runs/leave_one_out")
out_dir.mkdir(parents=True, exist_ok=True)
results = {}

variants = [("none", None)] + [(t.name, [t.name]) for t in cfg.data.tracks]
for label, zt in variants:
    out = out_dir / f"pred_chr2_zero_{label}.cool"
    predict(cfg, "2", str(out), batch=16, zero_tracks=zt)
    pred = fetch_chrom_matrix(str(out), "2")
    n = min(truth.shape[0], pred.shape[0])
    s = scc(pred[:n, :n], truth[:n, :n], max_dist_bins=128, h=1)
    results[label] = s
    print(f"zero={label:10s} SCC(chr2)={s:.4f}", flush=True)
    out.unlink()                                   # keep disk clean

base = results["none"]
ranked = sorted(
    ((k, v, base - v) for k, v in results.items() if k != "none"),
    key=lambda x: -x[2],
)
print("\nprotein importance (SCC drop when zeroed, chr2):")
for k, v, d in ranked:
    print(f"  {k:10s} {v:.4f}  drop {d:+.4f}")
json.dump(results, open(out_dir / "loo_scc_chr2.json", "w"), indent=2)
