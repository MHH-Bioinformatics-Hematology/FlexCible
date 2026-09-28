"""Consensus carried by the model: embedded map and learned map (registry 16).

At prediction time the only cell-specific input shall be the cell's ChIP
(plus sequence and motif channels). Two ways to carry the consensus Hi-C:

embedcheck  EMBEDDED consensus. consensus12 (all 12 training cells) is
            copied into the best checkpoint of all12refres_<k>kb
            (scripts/embed_reference.py, model.embed_reference) and
            predict.py / crosscell_predict.py look it up by chromosome and
            bin. Training cells keep their leave-one-out map. Verified: the
            predictions for K562, HMEC and HUVEC with the embedded map equal
            those with --reference consensus12 (crosscell_predict --save-pred
            on the 413 regions, and predict.py on one chromosome of K562);
            max abs difference in embedcheck_<k>kb.json.

stage1      LEARNED consensus from sequence. A model with GC/CpG and the four
            motif planes only (no ChIP, no reference) is trained on
            consensus12 as its target, in the all12 configuration (windows,
            training regions, honest validation, patience 10), single GPU.
            Predictions for training windows would be in-sample, so the map
            is built out of fold by chromosome: model "odd" trains on chr1,
            3, ..., 21 and predicts chr2, 4, ..., 22 and chrX; model "even"
            trains on chr2, 4, ..., 22 and chrX and predicts the odd
            chromosomes. The merged prediction is
            crosscell_oe/learned_cons12_oof_<k>kb.cool (+ .json). Model "all"
            (every training chromosome) is trained only for comparison on
            the test regions, which none of the three models trains on.
            Reported: the learned map against consensus12 on the 413 Akita
            test regions and on the disjoint test windows.
            Stage 2 is the 12-cell residual run all12reflearn
            (scripts/multicell_expanded.py) with this map as every cell's
            reference. The map derives from the maps of all 12 training
            cells (on the other chromosome half) and holds none of their
            values at the scored positions; its .json lists no cells, so the
            leakage guard does not apply to it.

    python scripts/learned_reference.py --stage stage1 --res 25000 --gpu 0
    python scripts/learned_reference.py --stage embedcheck --res 25000 --gpu 0
    python scripts/learned_reference.py --stage summary
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                      # noqa: E402

REPO = mx.REPO
PY = mx.PY
OE, CACHE, HM = mx.OE, mx.CACHE, mx.HM
OUT = HM / "runs/learned_reference"
MOTIFS = ["MA0139.2", "MA0095.3", "MA0088.2", "MA1522.1"]
ODD = [f"chr{i}" for i in range(1, 23, 2)]
EVEN = [f"chr{i}" for i in range(2, 23, 2)] + ["chrX"]
FOLDS = {"odd": ODD, "even": EVEN, "all": None}
PREDICTS = {"odd": EVEN, "even": ODD}          # out-of-fold chromosomes
HELDOUT = ["k562", "hmec_deep", "huvec_deep"]


def sh(cmd, log: Path, env=None):
    import subprocess
    print("[cmd]", " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as fh:
        return subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=REPO).returncode


def kb(res):
    return res // 1000


def stage1_name(res, fold):
    return f"stage1_{kb(res)}kb_{fold}"


def stage1_config(res, fold) -> dict:
    """all12 configuration (honest validation), one sample: consensus12 as the
    target, ChIP rows unused (select_tracks []), GC/CpG and 4 motifs."""
    cfg = copy.deepcopy(mx.run_config(mx.run_name("all12honest", res), 100, 10, 500))
    gm = next(s for s in cfg["data"]["samples"] if s["name"] == "gm12878")
    cfg["data"]["samples"] = [{"name": "consensus12", "cool_path": str(mx.consensus12(res)),
                               "bigwig_dir": gm["bigwig_dir"], "missing_tracks": [],
                               "assay": "hic"}]
    cfg["train"].update({
        "output_dir": str(OUT / stage1_name(res, fold)),
        "cache_dir": str(CACHE / f"lr_cons12_{kb(res)}kb"),
        "val_regions": 100, "val_exclude_overlap": True, "seed": 0,
        "select_tracks": [], "select_motifs": MOTIFS, "chroms": FOLDS[fold],
        "patience_epochs": 10, "max_epochs": 500, "resume": False,
    })
    return cfg


def finished(log: Path) -> bool:
    t = log.read_text(errors="replace") if log.exists() else ""
    return "early stop at epoch" in t or "done." in t


def stage1(res, gpu, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cache = Path(stage1_config(res, "all")["train"]["cache_dir"])
    if not (cache / "manifest.json").exists():
        cp = OUT / f"{cache.name}.precompute.yaml"
        yaml.safe_dump(stage1_config(res, "all"), open(cp, "w"), sort_keys=False)
        if sh([PY, "-m", "hicmultiome.precompute", "--config", cp, "--workers", workers],
              OUT / f"{cache.name}.precompute.log"):
            raise SystemExit("precompute failed")
    for fold in FOLDS:
        name = stage1_name(res, fold)
        log = OUT / f"{name}.log"
        if not finished(log):
            cp = OUT / f"{name}.yaml"
            yaml.safe_dump(stage1_config(res, fold), open(cp, "w"), sort_keys=False)
            log.write_text("")
            sh([PY, "-m", "hicmultiome.train", "--config", cp], log, env)
            if not finished(log):
                raise SystemExit(f"training failed: {name}")
    # predictions: each fold model on the other chromosomes, "all" everywhere
    preds = {}
    for fold in FOLDS:
        run = OUT / stage1_name(res, fold)
        ck = run / (run / "best").read_text().strip()
        out = OUT / f"{stage1_name(res, fold)}_pred.cool"
        chroms = ",".join(PREDICTS.get(fold, ODD + EVEN))
        if not out.exists():
            part = out.with_name(out.name + ".part.cool")
            part.unlink(missing_ok=True)
            if sh([PY, "-m", "hicmultiome.predict", "--config", ck / "config.yaml",
                   "--checkpoint", ck, "--chroms", chroms, "--out", part, "--tta"],
                  OUT / f"{stage1_name(res, fold)}.predict.log", env):
                raise SystemExit(f"prediction failed: {fold}")
            part.rename(out)
        preds[fold] = out
    merge_oof(res, preds["odd"], preds["even"])
    quality(res)


def merge_oof(res, pred_odd: Path, pred_even: Path):
    """Union of the two out-of-fold predictions (disjoint chromosomes)."""
    import cooler
    import pandas as pd

    out = mx.learned_map(res)
    if out.exists():
        return out
    a, b = cooler.Cooler(str(pred_odd)), cooler.Cooler(str(pred_even))
    px = pd.concat([a.pixels()[:], b.pixels()[:]], ignore_index=True)
    px = px.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    if px.duplicated(["bin1_id", "bin2_id"]).any():
        raise SystemExit("out-of-fold predictions overlap")
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    cooler.create_cooler(str(part), bins=a.bins()[:][["chrom", "start", "end"]],
                         pixels=px, dtypes={"count": np.float32}, ordered=True)
    part.rename(out)
    cells = json.loads(Path(str(mx.consensus12(res)) + ".json").read_text())["cells"]
    Path(str(out) + ".json").write_text(json.dumps({
        "res": res, "cells": [],
        "learned_from": str(mx.consensus12(res)), "learned_from_cells": cells,
        "inputs": "GC, CpG, 4 motif planes (no ChIP, no reference)",
        "out_of_fold": {"model_odd_trained_on": ODD, "predicts": EVEN,
                        "model_even_trained_on": EVEN, "predicts_odd": ODD},
        "models": [str(OUT / stage1_name(res, f)) for f in ("odd", "even")],
        "note": "holds no measured value at any scored position; cells [] so the "
                "leakage guard does not apply"}, indent=1))
    print(f"-> {out}", flush=True)
    return out


def region_r(A, B):
    n = A.shape[0]
    iu = np.triu_indices(n, k=2)
    x, y = A[iu], B[iu]
    m = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    if m.sum() < 100 or x[m].std() == 0 or y[m].std() == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def quality(res):
    """Median per-region Pearson of the learned maps against consensus12 on
    the 413 Akita test regions and on the disjoint test windows."""
    import cooler
    import pandas as pd

    tgt = cooler.Cooler(str(mx.consensus12(res)))
    maps = {"learned_oof": cooler.Cooler(str(mx.learned_map(res))),
            "stage1_all": cooler.Cooler(str(OUT / f"{stage1_name(res, 'all')}_pred.cool"))}
    import score_like_hicgan as sc
    reg = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    out = {"res": res}
    for name, clr in maps.items():
        rs = []
        for r in reg.itertuples():
            q = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
            T = np.asarray(tgt.matrix(balance=False).fetch(q), float)
            P = np.asarray(clr.matrix(balance=False).fetch(q), float)
            rs.append(region_r(P, T))
        out[f"{name}_akita_test_regions"] = round(float(np.nanmedian(rs)), 4)
        out[f"{name}_akita_test_n"] = int(np.isfinite(rs).sum())
        m = json.loads((CACHE / f"lr_cons12_{kb(res)}kb" / "manifest.json").read_text())
        W = m["window_size"]
        ws = []
        for shd in m["shards"]:
            if shd["split"] != "test":
                continue
            for st in np.load(shd["path"])["start"]:
                q = f"{shd['chrom']}:{int(st) * res}-{min((int(st) + W) * res, int(tgt.chromsizes[shd['chrom']]))}"
                ws.append(region_r(np.asarray(clr.matrix(balance=False).fetch(q), float),
                                   np.asarray(tgt.matrix(balance=False).fetch(q), float)))
        out[f"{name}_test_windows"] = round(float(np.nanmedian(ws)), 4)
        out[f"{name}_test_windows_n"] = int(np.isfinite(ws).sum())
    for fold in FOLDS:
        t = (OUT / f"{stage1_name(res, fold)}.log").read_text(errors="replace")
        m2 = re.findall(r"early stop at epoch (\d+) \(best val ([-0-9.]+)", t)
        if m2:
            out[f"stage1_{fold}_epochs"], out[f"stage1_{fold}_honest_val"] = int(m2[-1][0]), float(m2[-1][1])
    (OUT / f"quality_{kb(res)}kb.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


def embedcheck(res, gpu):
    """Embed consensus12 into all12refres_<k>kb and compare predictions with
    the embedded map against --reference consensus12 for the held-out cells."""
    import cooler

    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    name = mx.run_name("all12refres", res)
    run = mx.REF_OUT / name
    ck = run / (run / "best").read_text().strip()
    d = OUT / f"embedcheck_{kb(res)}kb"
    d.mkdir(exist_ok=True)
    log = d / "embedcheck.log"
    if sh([PY, "scripts/embed_reference.py", "--run", run, "--reference",
           mx.consensus12(res)], log, env):
        raise SystemExit("embedding failed")
    rec = {"res": res, "run": str(run), "embedded": str(ck / "embedded_reference.cool")}
    for cell in HELDOUT:
        _, bw, assay, _ = mx.CELLS[cell]
        npz = {}
        for mode in ("embedded", "explicit"):
            f = d / f"{cell}_{mode}.npz"
            if not f.exists():
                cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
                       "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
                       "--target", mx.target(cell, res), "--tta", "--assay", assay,
                       "--binsize", res, "--out-dir", d / mode, "--save-pred", f]
                if mode == "explicit":
                    cmd += ["--reference", mx.consensus12(res)]
                if sh(cmd, log, env):
                    raise SystemExit(f"crosscell failed {cell} {mode}")
            npz[mode] = np.load(f)
        keys = sorted(npz["embedded"].files)
        assert keys == sorted(npz["explicit"].files)
        diff = max(float(np.nanmax(np.abs(npz["embedded"][k] - npz["explicit"][k])))
                   for k in keys)
        rec[f"{cell}_regions"] = len(keys)
        rec[f"{cell}_max_abs_diff_413_regions"] = diff
    # genome-wide path: predict.py, K562, one chromosome
    _, bw, _, _ = mx.CELLS["k562"]
    cools = {}
    for mode in ("embedded", "explicit"):
        out = d / f"k562_chr2_{mode}.cool"
        if not out.exists():
            cmd = [PY, "-m", "hicmultiome.predict", "--config", ck / "config.yaml",
                   "--checkpoint", ck, "--chroms", "chr2", "--bigwig-dir", bw,
                   "--cell", "k562", "--out", out, "--tta"]
            if mode == "explicit":
                cmd += ["--reference", mx.consensus12(res)]
            if sh(cmd, log, env):
                raise SystemExit(f"predict failed {mode}")
        cools[mode] = np.nan_to_num(np.asarray(
            cooler.Cooler(str(out)).matrix(balance=False).fetch("chr2"), np.float64))
    rec["k562_chr2_predict_max_abs_diff"] = float(np.abs(cools["embedded"] - cools["explicit"]).max())
    rec["k562_chr2_predict_nonzero_pixels"] = int((cools["embedded"] != 0).sum())
    (OUT / f"embedcheck_{kb(res)}kb.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


def summary():
    rows = []
    for res in (25000, 10000):
        q = OUT / f"quality_{kb(res)}kb.json"
        e = OUT / f"embedcheck_{kb(res)}kb.json"
        r = {"res_kb": kb(res)}
        if q.exists():
            r.update({k: v for k, v in json.loads(q.read_text()).items() if k != "res"})
        if e.exists():
            r.update({k: v for k, v in json.loads(e.read_text()).items()
                      if "diff" in k or k.endswith("_regions")})
        if len(r) > 1:
            rows.append(r)
    if rows:
        import csv
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        tmp = OUT / "learned_reference.csv.part"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        tmp.rename(OUT / "learned_reference.csv")
        print(f"wrote {OUT / 'learned_reference.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["stage1", "embedcheck", "summary"], required=True)
    ap.add_argument("--res", nargs="*", type=int, default=[25000, 10000])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for res in a.res:
        if a.stage == "stage1":
            stage1(res, a.gpu, a.workers)
        elif a.stage == "embedcheck":
            embedcheck(res, a.gpu)
    summary()
    print("LEARNED_REFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
