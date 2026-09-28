"""Check that no training window touches the test data, per cell.

For every cache given, and every sample in it: the number of training windows
(split "train" of the manifest, i.e. before the validation split, which only
removes windows) that share at least one base pair with (a) any of Akita's 413
test regions (sequences_test_list.tsv) or (b) any window of the cache's test
split (the disjoint test windows scored by eval_val_windows --split test), on
the same chromosome. Both counts must be 0. Writes one CSV row per (cache,
sample) and exits 1 when any count is nonzero.

    python scripts/check_split_disjoint.py --caches mc_all12ref_25kb ... \
        --out runs/multicell_reference/split_disjoint.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("/mnt/processed/wolffjoa/hic-multiome/cache_expanded")
TEST_REGIONS = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/"
                    "akita_normalization/akita_validation/sequences_test_list.tsv")


def intervals_overlap(starts, ends, qs, qe) -> np.ndarray:
    """Per query [qs, qe): True if it overlaps any [starts, ends) (sorted by start)."""
    if len(starts) == 0:
        return np.zeros(len(qs), dtype=bool)
    order = np.argsort(starts)
    s, e = np.asarray(starts)[order], np.asarray(ends)[order]
    emax = np.maximum.accumulate(e)
    k = np.searchsorted(s, qe, side="left")        # intervals starting before qe
    out = np.zeros(len(qs), dtype=bool)
    has = k > 0
    out[has] = emax[k[has] - 1] > np.asarray(qs)[has]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--regions", default=str(TEST_REGIONS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reg = pd.read_csv(a.regions, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    rows, bad = [], 0
    for c in a.caches:
        cache = Path(c) if Path(c).is_absolute() else CACHE / c
        m = json.loads((cache / "manifest.json").read_text())
        res, W = int(m["config_resolution"]), int(m["window_size"])
        per = {}
        for sh in m["shards"]:
            st = np.load(sh["path"])["start"].astype(np.int64)
            d = per.setdefault(sh.get("sample", "default"), {"train": {}, "test": {}})
            d[sh["split"]].setdefault(sh["chrom"], []).append(st)
        for smp, d in sorted(per.items()):
            n_tr = n_reg = n_tw = 0
            for chrom, parts in d["train"].items():
                st = np.concatenate(parts)
                qs, qe = st * res, (st + W) * res
                n_tr += len(st)
                r = reg[reg["chrom"] == chrom]
                n_reg += int(intervals_overlap(r["start"].values, r["end"].values, qs, qe).sum())
                tw = np.concatenate(d["test"].get(chrom, [np.zeros(0, np.int64)]))
                n_tw += int(intervals_overlap(tw * res, (tw + W) * res, qs, qe).sum())
            n_test = sum(len(x) for p in d["test"].values() for x in p)
            rows.append({"cache": cache.name, "res_kb": res // 1000, "window": W,
                         "sample": smp, "train_windows": n_tr, "test_windows": n_test,
                         "train_overlapping_akita_test_regions": n_reg,
                         "train_overlapping_test_windows": n_tw})
            bad += n_reg + n_tw
    out = Path(a.out)
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.rename(out)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"total overlaps: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
