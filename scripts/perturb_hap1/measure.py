"""Measured log2 O/E maps (WT, each clone) of the three captured regions at 10 and 5 kb."""
import sys, os, csv; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from lib import *
(RES / "maps").mkdir(exist_ok=True, parents=True)
summary = []
for rn, r in REGIONS.items():
    for rl in ("10kb", "5kb"):
        res = RESOLUTION[rl]
        arrs = {}
        cnt = {}
        for es in [WT_SET] + [s for d in DELETIONS.values() if d["region"] == rn for s in d["sets"]]:
            c = region_counts(mcool_path(es), res, r["chrom"], r["start"], r["end"])
            cnt[es] = c
            arrs[es] = region_logoe(c)
            summary.append((rn, rl, es, int(np.triu(c).sum()), int((c.sum(1) > 0).sum())))
        np.savez_compressed(RES / "maps" / f"measured_{rn}_{rl}.npz", **arrs, **{f"counts_{k}": v.astype(np.float32) for k, v in cnt.items()})
with open(RES / "measured_depth.csv", "w") as f:
    w = csv.writer(f); w.writerow(["region", "res", "set", "counts_in_region", "covered_bins"]); w.writerows(summary)
print(open(RES / "measured_depth.csv").read())
