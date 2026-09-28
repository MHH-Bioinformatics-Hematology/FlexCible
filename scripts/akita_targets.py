"""Akita's published targets of all five cells on the 2048 bp grid, and their
consensus without GM12878 (registry section 1).

Akita's 1m data (basenji_hic bucket, fetched by runs akita_tfr/fetch.sh) holds
for each of 7,840 windows of 1,048,576 bp (7,008 train, 419 valid, 413 test)
the targets of HFF (index 0), H1hESC (1), GM12878 (2), IMR90 (3) and HCT116
(4): the upper triangle (offset >= 2) of the central 448 x 448 bins at 2048
bp, clipped log(obs/exp), as float16. The GM12878 cools used in section 1 were
made from these records by ~/src/Hi-cGAN/manuscript_gigascience/scripts/
fetch_akita_targets.py (map rebuilt per window, placed at window start +
65,536 bp, duplicated pixels averaged); this script applies the same rule to
all five cells in one pass.

  records   (TensorFlow env, e.g. envs/basenji) every record of every fold,
            in sequences.bed order, into records.f16 (7840 x 99681 x 5)
  cools     (hicdiff env) per cell akita_<cell>_all_2048bp.cool (all folds),
            akita_gm12878_train_2048bp.cool (train fold only, the section 1
            training target), akita_consensus4_2048bp.cool: per window the
            mean over HFF, H1, IMR90 and HCT116 of the finite values, then
            duplicated pixels averaged; .json lists the four cells. Checks:
            the rebuilt GM12878 test fold against the section 1 test cool.

    envs/basenji/bin/python scripts/akita_targets.py --stage records
    envs/hicdiff/bin/python scripts/akita_targets.py --stage cools
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/mnt/processed/wolffjoa/hic-multiome/akita_tfr")
OUTD = Path("/mnt/processed/wolffjoa/hic-multiome/akita2kb")
PUBLISHED_TEST = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/"
                      "akita_normalization/akita_published_GM12878_test_2048bp.cool")
CHROM_SIZES = Path.home() / "src/epiphany/hg38.chrom.sizes"
SEQ_LEN, BINSIZE, NB, DIAG = 1_048_576, 2048, 448, 2
CELLS = ["hff", "h1", "gm12878", "imr90", "hct116"]          # target index order
CONS4 = ["hff", "h1", "imr90", "hct116"]
NV = NB * (NB - 1) // 2 - (NB - 1)        # 99,681 upper-triangle values, offset >= 2
CROP = (SEQ_LEN - NB * BINSIZE) // 2      # 65,536 bp


def seqs():
    s = pd.read_csv(D / "sequences.bed", sep="\t", header=None,
                    names=["chrom", "start", "end", "fold"])
    # records are read fold by fold, in sequences.bed order within a fold
    order = pd.concat([s[s.fold == f] for f in ("train", "valid", "test")])
    return order.reset_index().rename(columns={"index": "bed_row"})


def records():
    import tensorflow as tf

    order = seqs()
    mm = np.lib.format.open_memmap(D / "records.f16.npy.part", mode="w+",
                                   dtype=np.float16, shape=(len(order), NV, 5))
    feat = {"target": tf.io.FixedLenFeature([], tf.string),
            "sequence": tf.io.FixedLenFeature([], tf.string),
            "genome": tf.io.FixedLenFeature([], tf.int64)}
    k = 0
    for fold in ("train", "valid", "test"):
        shards = sorted(D.glob(f"{fold}-*.tfr"), key=lambda p: int(p.stem.split("-")[1]))
        n_fold = int((order.fold == fold).sum())
        n = 0
        for raw in tf.data.TFRecordDataset([str(p) for p in shards], compression_type="ZLIB"):
            ex = tf.io.parse_single_example(raw, feat)
            v = tf.io.decode_raw(ex["target"], tf.float16).numpy().reshape(NV, 5)
            mm[k] = v
            k += 1
            n += 1
        print(f"{fold}: {n} records ({n_fold} windows in sequences.bed)", flush=True)
        if n != n_fold:
            raise SystemExit(f"{fold}: record count {n} != {n_fold}")
    mm.flush()
    del mm
    (D / "records.f16.npy.part").rename(D / "records.f16.npy")
    order.to_csv(D / "records_order.tsv", sep="\t", index=False)
    print("RECORDS_DONE", flush=True)


def cools():
    """One output at a time, streamed chromosome by chromosome (all outputs'
    pixels together do not fit in memory)."""
    import cooler

    order = pd.read_csv(D / "records_order.tsv", sep="\t")
    rec = np.load(D / "records.f16.npy", mmap_mode="r")
    sizes = pd.read_csv(CHROM_SIZES, sep="\t", header=None, names=["chrom", "length"])
    sizes = sizes[sizes.chrom.isin(order.chrom.unique())]
    bins = cooler.binnify(sizes.set_index("chrom")["length"], BINSIZE)
    first = {c: int(g.index[0]) for c, g in bins.groupby("chrom", sort=False)}
    nbins = {c: len(g) for c, g in bins.groupby("chrom", sort=False)}
    iu = np.triu_indices(NB, DIAG)
    outs = {f"akita_{c}_all_2048bp.cool": (c, None) for c in CELLS}
    outs["akita_gm12878_train_2048bp.cool"] = ("gm12878", "train")
    outs["akita_consensus4_2048bp.cool"] = ("consensus4", None)
    chrom_order = [c for c in bins["chrom"].unique() if c in set(order.chrom)]

    def pixels(cell, fold):
        for chrom in chrom_order:
            rows = order.index[(order.chrom == chrom)
                               & ((order.fold == fold) if fold else True)].values
            n = nbins[chrom]
            S = np.zeros((n, NB))
            C = np.zeros((n, NB), np.int32)
            for r in rows:
                if (int(order.start[r]) + CROP) % BINSIZE:
                    raise SystemExit("window not on the 2048 bp grid")
                b0 = (int(order.start[r]) + CROP) // BINSIZE
                v = np.asarray(rec[r], dtype=np.float32)             # (NV, 5)
                if cell == "consensus4":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        x = np.nanmean(v[:, [0, 1, 3, 4]], axis=1)  # NaN where all absent
                else:
                    x = v[:, CELLS.index(cell)]
                ok = np.isfinite(x)
                gi = b0 + iu[0][ok]
                off = iu[1][ok] - iu[0][ok]
                keep = gi < n
                # (bin, offset) pairs are unique within one window
                S[gi[keep], off[keep]] += x[ok][keep]
                C[gi[keep], off[keep]] += 1
            gi, off = np.nonzero(C)
            if len(gi):
                yield pd.DataFrame({"bin1_id": gi + first[chrom],
                                    "bin2_id": gi + off + first[chrom],
                                    "count": S[gi, off] / C[gi, off]})

    for k, (cell, fold) in outs.items():
        path = OUTD / k
        if path.exists():
            continue
        meta = {"source": "Akita 1m data, gs://basenji_hic/1m/data (targets of "
                          "Fudenberg et al. 2020)", "binsize": BINSIZE,
                "folds": [fold] if fold else ["train", "valid", "test"],
                "rule": "per-window 448-bin map at window start + 65,536 bp, "
                        "duplicated pixels averaged"}
        if cell == "consensus4":
            meta.update({"cells": CONS4,
                         "method": "per window mean of the finite HFF, H1, IMR90 "
                                   "and HCT116 targets, then duplicates averaged"})
        else:
            meta["cells"] = [cell]
        part = path.with_name(path.name + ".part")
        part.unlink(missing_ok=True)
        cooler.create_cooler(str(part), bins, pixels(cell, fold),
                             dtypes={"count": np.float64}, ordered=True,
                             symmetric_upper=True)
        part.rename(path)
        Path(str(path) + ".json").write_text(json.dumps(meta, indent=1))
        print(f"-> {path}", flush=True)
    check()


def check():
    """Rebuilt GM12878 test fold against the section 1 published test cool."""
    import cooler

    a = cooler.Cooler(str(OUTD / "akita_gm12878_all_2048bp.cool"))
    b = cooler.Cooler(str(PUBLISHED_TEST))
    reg = pd.read_csv(Path("/mnt/processed/wolffjoa/restart_of_prediction_project/"
                           "akita_normalization/akita_validation/sequences_test_list.tsv"),
                      sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    diffs = []
    for r in reg.itertuples():
        q = f"{r.chrom}:{r.start + CROP}-{r.end - CROP}"
        A = np.asarray(a.matrix(balance=False).fetch(q), float)
        B = np.asarray(b.matrix(balance=False).fetch(q), float)
        m = (A != 0) & (B != 0)
        diffs.append(float(np.abs(A[m] - B[m]).max()) if m.any() else 0.0)
    res = {"regions": len(diffs), "max_abs_diff_vs_published_test_cool": max(diffs),
           "note": "all-fold GM12878 map vs the section 1 test-fold cool, on the "
                   "central 448 bins of the 413 test windows; differences can only "
                   "come from pixels also covered by valid or train windows"}
    (OUTD / "akita_targets_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def cachecheck(old: str, new: str):
    """The section 1 cache (built from the original train-fold cool by an
    earlier precompute that kept near-empty windows) against a cache built
    from the rebuilt train-fold cool: per training window present in both,
    the encoded targets must agree; windows only in the old cache are
    counted with their fraction of zero pixels."""
    import json as _j

    mo = _j.loads((Path(old) / "manifest.json").read_text())
    mn = _j.loads((Path(new) / "manifest.json").read_text())
    so = {x["chrom"]: x for x in mo["shards"] if x["split"] == "train"}
    sn = {x["chrom"]: x for x in mn["shards"] if x["split"] == "train"}
    res = {"train_windows_old": sum(x["n_windows"] for x in so.values()),
           "train_windows_new": sum(x["n_windows"] for x in sn.values()),
           "common_windows": 0, "only_old": 0, "only_new": 0,
           "max_abs_diff_target_common": 0.0, "only_old_min_zero_fraction": 1.0}
    for chrom in sorted(set(so) | set(sn)):
        # load each array once (indexing an NpzFile decompresses it every time)
        a = dict(np.load(so[chrom]["path"])) if chrom in so else None
        b = dict(np.load(sn[chrom]["path"])) if chrom in sn else None
        sa = {int(x): i for i, x in enumerate(a["start"])} if a is not None else {}
        sb = {int(x): i for i, x in enumerate(b["start"])} if b is not None else {}
        for st in set(sa) & set(sb):
            d = float(np.abs(a["target"][sa[st]].astype(np.float32)
                             - b["target"][sb[st]].astype(np.float32)).max())
            res["max_abs_diff_target_common"] = max(res["max_abs_diff_target_common"], d)
            res["common_windows"] += 1
        for st in set(sa) - set(sb):
            res["only_old"] += 1
            # the zero-fraction filter works on the raw block; here the encoded
            # target, where an unmeasured pixel encodes as exactly 0
            z = float((a["target"][sa[st]][0] == 0).mean())
            res["only_old_min_zero_fraction"] = min(res["only_old_min_zero_fraction"], z)
        res["only_new"] += len(set(sb) - set(sa))
    (OUTD / "akita_cache_check.json").write_text(_j.dumps(res, indent=1))
    print(_j.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["records", "cools", "check", "cachecheck"],
                    required=True)
    ap.add_argument("--old-cache", default=None)
    ap.add_argument("--new-cache", default=None)
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if a.stage == "cachecheck":
        cachecheck(a.old_cache, a.new_cache)
        return
    {"records": records, "cools": cools, "check": check}[a.stage]()


if __name__ == "__main__":
    main()
