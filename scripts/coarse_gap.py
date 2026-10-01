"""Closing the coarse-resolution gap (GM12878 only, registry section 15).

At 10 and 25 kb the GM12878 model (section 14, "all" inputs) scores about as
well as, or below, the measured map of another cell, while at 2 kb it wins
(section 13). Three ideas, each trained with the protocol of
scripts/resolution_inputs.py (all 13 ChIP tracks + 4 motifs, the default
window per resolution, 100 honest validation windows with every overlapping
training window removed, patience 10 epochs, seeds 0 and 1, one GPU per run):

  A  reference map as input. Reference = consensus11: the pixel-wise mean of
     the proper O/E maps of IMR90, HFF, H1, HCT116, HepG2, A549, NHEK,
     GM23248, MCF-7, CD14+ monocytes and PC-9 (scripts/consensus_oe.py); it
     excludes GM12878 (training cell) and K562, HMEC and HUVEC (held-out
     cells), so the same reference serves the cross-cell evaluation.
       refin    reference as one extra input plane (model.reference: input)
       refres   input plane and residual output (model.reference: residual)
     At 5, 10 and 25 kb. Also scored: the reference map alone on the same
     windows and regions, and the A models on held-out K562, HMEC and HUVEC
     against every measured map of another cell.
  B  more windows and a smaller network (10 and 25 kb):
       q4       quarter-stride windows (window.stride = size / 4), about
                four times the half-stride windows
       narrow   model.base_channels 24 (default 48)
       q4narrow both
     q4 runs pick their 100 validation windows from the quarter-stride
     windows and drop every training window overlapping one; their test
     scores use the half-stride test windows of the baseline cache.
  C  coarse targets from the fine map (10 and 25 kb):
       avg2kb   target = the 2 kb GM12878 O/E map block-averaged onto the
                coarse grid (scripts/coarsen_oe.py), scored against both the
                averaged and the native target. The baseline models are also
                scored against the averaged target.

Outputs under runs/coarse_gap/: <res>_w<size>__<variant>__s<seed>/ with
test_eval.json (native target, half-stride test windows), crosscell_valid/
and crosscell_test/ (Akita's 419 validation and 413 test regions, native
target, consensus11 as a measured-map column), and per variant: ref_*_eval.json
(reference alone), crosscell_heldout/ (A), test_eval_avg.json,
val_eval_native.json and crosscell_test_avg/ (C). baseline/ holds the extra
scores of the section 14 "all" runs. STATUS.md is rewritten after every run.

    python scripts/coarse_gap.py --stage train --slots 0,1
    python scripts/coarse_gap.py --stage summary
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                      # noqa: E402
import resolution_inputs as ri                                       # noqa: E402

REPO = ri.REPO
HM, OE, CACHE, PY = ri.HM, ri.OE, ri.CACHE, ri.PY
OUT = HM / "runs/coarse_gap"
CONS_CELLS = ["imr90", "hff", "h1", "hct116", "hepg2", "a549", "nhek", "gm23248",
              "mcf7", "cd14mono", "pc9"]
HELDOUT = ["k562", "hmec_deep", "huvec_deep"]
SEEDS = [0, 1]
VARIANTS = {
    # name: (cache kind, model overrides, resolutions)
    "refin": ("ref", {"reference": "input"}, [25000, 10000, 5000]),
    "refres": ("ref", {"reference": "residual"}, [25000, 10000, 5000]),
    "avg2kb": ("avg", {}, [25000, 10000]),
    "q4": ("q4", {}, [25000, 10000]),
    "narrow": ("base", {"base_channels": 24}, [25000, 10000]),
    "q4narrow": ("q4", {"base_channels": 24}, [25000, 10000]),
}
BASELINE_SET = "all"            # section 14 runs used as the baseline


def kb(res):
    return res // 1000


def consensus(res) -> Path:
    return OE / f"consensus11_proper_{kb(res)}kb.cool"


def avg_target(res) -> Path:
    return OE / f"gm12878_avg2kb_{kb(res)}kb.cool"


def native_target(res) -> Path:
    return OE / f"gm12878_proper_{kb(res)}kb.cool"


# ---------------------------------------------------------------- caches
def cache_config(res: int, kind: str) -> dict:
    size = ri.DEFAULT_WINDOW[res]
    cfg = ri.base_config(res, size)
    name = ri.cache_name(res, size)
    if kind == "ref":
        cfg["data"]["reference_cool"] = str(consensus(res))
        name += "_ref"
    elif kind == "q4":
        cfg["window"]["stride"] = size // 4
        name += "_q4"
    elif kind == "avg":
        cfg["data"]["samples"][0]["cool_path"] = str(avg_target(res))
        name += "_avg"
    elif kind != "base":
        raise ValueError(kind)
    cfg["train"]["cache_dir"] = str(CACHE / name)
    return cfg


def cache_input(res, kind) -> Path | None:
    return {"ref": consensus(res), "avg": avg_target(res)}.get(kind)


_cache_locks: dict = {}
_cache_guard = threading.Lock()


def ensure_cache(res, kind, workers=8) -> Path:
    """Precompute the cache once (waits for its input map to exist)."""
    cfg = cache_config(res, kind)
    cache = Path(cfg["train"]["cache_dir"])
    with _cache_guard:
        lk = _cache_locks.setdefault(cache.name, threading.Lock())
    with lk:
        if (cache / "manifest.json").exists():
            return cache
        need = cache_input(res, kind)
        while need is not None and not need.exists():
            print(f"[wait] {cache.name}: {need.name} not built yet", flush=True)
            time.sleep(120)
        cp = OUT / f"{cache.name}.precompute.yaml"
        yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        print(f"[precompute] {cache.name}", flush=True)
        if ri.sh([PY, "-m", "hicmultiome.precompute", "--config", cp, "--workers", workers],
                 OUT / f"{cache.name}.precompute.log", mode="w"):
            raise RuntimeError(f"precompute failed for {cache.name}")
    return cache


def windows_of(cache: Path):
    m = json.loads((cache / "manifest.json").read_text())
    import numpy as np
    out = {}
    for s in m["shards"]:
        out[(s["split"], s["chrom"])] = tuple(int(x) for x in np.load(s["path"])["start"])
    return out


def same_windows(c1: Path, c2: Path) -> bool:
    return windows_of(c1) == windows_of(c2)


# ---------------------------------------------------------------- runs
def run_name(res, variant, seed):
    return f"{kb(res)}kb_w{ri.DEFAULT_WINDOW[res]}__{variant}__s{seed}"


def parse(name):
    m = re.fullmatch(r"(\d+)kb_w(\d+)__(.+)__s(\d+)", name)
    return int(m.group(1)) * 1000, int(m.group(2)), m.group(3), int(m.group(4))


def log_of(name):
    return OUT / f"{name}.log"


def finished(name) -> bool:
    p = log_of(name)
    t = p.read_text(errors="replace") if p.exists() else ""
    return "early stop at epoch" in t or "done." in t


def stop_info(name):
    p = log_of(name)
    m = ri.STOP_RE.findall(p.read_text(errors="replace")) if p.exists() else []
    if not m:
        return None
    ep, best, noimp = m[-1]
    return int(ep), float(best), int(noimp)


def run_config(name, a) -> dict:
    res, size, variant, seed = parse(name)
    kind, model_over, _ = VARIANTS[variant]
    tracks, motifs = ri.input_sets()[BASELINE_SET]
    cfg = ri.run_config(res, size, BASELINE_SET, seed, tracks, motifs, a.patience,
                        a.max_epochs)
    cc = cache_config(res, kind)
    cfg["data"] = copy.deepcopy(cc["data"])
    cfg["window"] = copy.deepcopy(cc["window"])
    cfg["train"]["cache_dir"] = cc["train"]["cache_dir"]
    cfg["train"]["output_dir"] = str(OUT / name)
    cfg["train"]["val_regions"] = 100
    cfg["model"].update(model_over)
    return cfg


def train_one(name, gpu, a):
    res, size, variant, seed = parse(name)
    kind = VARIANTS[variant][0]
    ensure_cache(res, kind, a.workers)
    if kind in ("q4", "avg"):
        ensure_cache(res, "base", a.workers)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if not finished(name):
        cp = OUT / f"{name}.yaml"
        yaml.safe_dump(run_config(name, a), open(cp, "w"), sort_keys=False)
        print(f"[train] {name} on GPU {gpu}", flush=True)
        ri.sh([PY, "-m", "hicmultiome.train", "--config", cp], log_of(name), env, mode="w")
        if not finished(name):
            print(f"[FAILED] {name}", flush=True)
            return
    si = stop_info(name)
    if si and si[2] < a.patience:
        print(f"[WARN] {name} stopped by the epoch cap at {si[0]}", flush=True)
    score_one(name, gpu)


def other_maps(cell, res) -> list[str]:
    """cell=cool of every measured map of another cell, plus consensus11."""
    out = [f"{c}={mx.target(c, res)}" for c in mx.BASELINE_MAPS
           if c != cell and mx.target(c, res).exists()]
    if consensus(res).exists():
        out.append(f"consensus11={consensus(res)}")
    return out


def crosscell(run: Path, ck: Path, cell: str, res: int, target: Path, regions: Path,
              out_dir: Path, log: Path, env, others=(), reference=None):
    if (out_dir / f"predict_{cell}.csv").exists():
        return
    bw = mx.CELLS[cell][1]
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
           "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw, "--target", target,
           "--tta", "--binsize", res, "--regions", regions, "--out-dir", out_dir]
    if others:
        cmd += ["--others", *others]
    if reference is not None:
        cmd += ["--reference", reference]
    ri.sh(cmd, log, env)


def evalw(run: Path, out: Path, log: Path, env, split="test", cache=None, ref_only=False):
    if out.exists():
        return
    cmd = [PY, "scripts/eval_val_windows.py", "--run", run, "--split", split, "--out", out]
    if cache is not None:
        cmd += ["--cache-dir", cache]
    if ref_only:
        cmd.append("--reference-only")
    ri.sh(cmd, log, env)


def score_one(name, gpu):
    res, size, variant, seed = parse(name)
    kind = VARIANTS[variant][0]
    run = OUT / name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    ck = run / (run / "best").read_text().strip()
    elog = OUT / f"{name}.eval.log"
    clog = OUT / f"{name}.crosscell.log"
    base_cache = Path(cache_config(res, "base")["train"]["cache_dir"])
    own_cache = Path(cache_config(res, kind)["train"]["cache_dir"])
    # test windows: always the half-stride windows with the native target
    evalw(run, run / "test_eval.json", elog, env,
          cache=None if kind in ("base", "ref") else base_cache)
    others = [f"consensus11={consensus(res)}"] if kind == "ref" else []
    for tag, regions in (("valid", ri.VALID_REGIONS), ("test", ri.TEST_REGIONS)):
        crosscell(run, ck, "gm12878", res, native_target(res), regions,
                  run / f"crosscell_{tag}", clog, env, others)
    if kind == "ref":
        # the reference map alone, on the same windows
        evalw(run, run / "ref_test_eval.json", elog, env, ref_only=True)
        evalw(run, run / "ref_val_eval.json", elog, env, split="val", ref_only=True)
        for cell in HELDOUT:
            crosscell(run, ck, cell, res, mx.target(cell, res), ri.TEST_REGIONS,
                      run / "crosscell_heldout", clog, env, other_maps(cell, res),
                      reference=consensus(res))
    if kind == "avg":
        evalw(run, run / "test_eval_avg.json", elog, env)
        if same_windows(own_cache, base_cache):
            evalw(run, run / "val_eval_native.json", elog, env, split="val", cache=base_cache)
        else:
            print(f"[note] {name}: averaged and native caches differ in windows; "
                  "no native validation score", flush=True)
        crosscell(run, ck, "gm12878", res, avg_target(res), ri.TEST_REGIONS,
                  run / "crosscell_test_avg", clog, env)


def score_baseline(name, gpu, a):
    """Extra scores of a section 14 run (GM12878 only, all inputs): held-out
    cells with every other measured map, and (10, 25 kb) the averaged target."""
    res, size, _, seed = ri.parse_run(name)
    src = ri.OUT / name
    ck = src / (src / "best").read_text().strip()
    out = OUT / "baseline" / name
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = OUT / "baseline" / f"{name}.log"
    for cell in HELDOUT:
        crosscell(src, ck, cell, res, mx.target(cell, res), ri.TEST_REGIONS,
                  out / "crosscell_heldout", log, env, other_maps(cell, res))
    if res in VARIANTS["avg2kb"][2] and (not a.variants or "avg2kb" in a.variants):
        avg_cache = ensure_cache(res, "avg", a.workers)
        base_cache = ensure_cache(res, "base", a.workers)
        evalw(src, out / "test_eval_avg.json", log, env, cache=avg_cache)
        if same_windows(avg_cache, base_cache):
            evalw(src, out / "val_eval_avg.json", log, env, split="val", cache=avg_cache)
        crosscell(src, ck, "gm12878", res, avg_target(res), ri.TEST_REGIONS,
                  out / "crosscell_test_avg", log, env)
    # consensus11 alone on Akita's test regions, and every other cell's map
    crosscell(src, ck, "gm12878", res, native_target(res), ri.TEST_REGIONS,
              out / "crosscell_test_maps", log, env, other_maps("gm12878", res))


# ---------------------------------------------------------------- queue
def run_queue(jobs, slots, a):
    """jobs: [(kind, name)], kind 'train' or 'baseline'."""
    lock = threading.Lock()
    todo = list(jobs)

    def worker(gpu):
        while True:
            with lock:
                if not todo:
                    return
                kind, name = todo.pop(0)
            try:
                if kind == "train":
                    train_one(name, gpu, a)
                else:
                    score_baseline(name, gpu, a)
            except Exception as e:                   # noqa: BLE001
                print(f"[ERROR] {name}: {e}", flush=True)
            with lock:
                try:
                    summarize(quiet=True)
                except Exception as e:               # noqa: BLE001
                    print(f"[summary] {e}", flush=True)

    ths = [threading.Thread(target=worker, args=(g,)) for g in slots]
    for i, t in enumerate(ths):
        t.start()
        time.sleep(20 if i < len(ths) - 1 else 0)
    for t in ths:
        t.join()


def all_jobs(res_filter=None, variants=None):
    order = [("refin", 25000), ("refres", 25000), ("refin", 10000), ("refres", 10000),
             ("avg2kb", 25000), ("avg2kb", 10000),
             ("refin", 5000), ("refres", 5000),
             ("q4", 25000), ("narrow", 25000), ("q4narrow", 25000),
             ("q4", 10000), ("narrow", 10000), ("q4narrow", 10000)]
    jobs = []
    for v, r in order:
        if (res_filter and r not in res_filter) or (variants and v not in variants):
            continue
        jobs += [("train", run_name(r, v, s)) for s in SEEDS]
        if v == "refres" and r == 10000:
            # baseline extras once the A runs at 25 and 10 kb are queued
            jobs += [("baseline", ri.run_name(rr, ri.DEFAULT_WINDOW[rr], BASELINE_SET, s))
                     for rr in (25000, 10000, 5000) for s in SEEDS
                     if not res_filter or rr in res_filter]
    return jobs


# ---------------------------------------------------------------- results
def _median(f: Path, col: str):
    return ri._median(f, col)


def _json(f: Path, key="oe_median_r"):
    return round(json.loads(f.read_text())[key], 4) if f.exists() else None


def result_row(name):
    res, size, variant, seed = parse(name)
    run = OUT / name
    si = stop_info(name)
    r = {"res_kb": kb(res), "window": size, "variant": variant, "seed": seed,
         "honest_val": si[1] if si else None, "stopped_at_epoch": si[0] if si else None,
         "stopped_by": ("patience" if si and si[2] >= 10 else "cap") if si else None}
    t = OUT / f"{name}.log"
    m = re.search(r"(\d+) train windows \(", t.read_text(errors="replace")) if t.exists() else None
    r["train_windows"] = int(m.group(1)) if m else None
    m = re.search(r"([0-9.]+)M params", t.read_text(errors="replace")) if t.exists() else None
    r["params_M"] = float(m.group(1)) if m else None
    r["test_windows"] = _json(run / "test_eval.json")
    r["akita_valid_regions"] = _median(run / "crosscell_valid/predict_gm12878.csv", "model")
    r["akita_test_regions"] = _median(run / "crosscell_test/predict_gm12878.csv", "model")
    if variant == "avg2kb":
        r["val_native"] = _json(run / "val_eval_native.json")
        r["test_windows_avg"] = _json(run / "test_eval_avg.json")
        r["akita_test_regions_avg"] = _median(run / "crosscell_test_avg/predict_gm12878.csv",
                                              "model")
    if VARIANTS[variant][0] == "ref":
        r["ref_alone_val"] = _json(run / "ref_val_eval.json")
        r["ref_alone_test_windows"] = _json(run / "ref_test_eval.json")
        r["ref_alone_akita_valid_regions"] = _median(
            run / "crosscell_valid/predict_gm12878.csv", "base_consensus11")
        r["ref_alone_akita_test_regions"] = _median(
            run / "crosscell_test/predict_gm12878.csv", "base_consensus11")
    return r


def baseline_rows():
    rows = []
    for res in (25000, 10000, 5000):
        for seed in SEEDS:
            name = ri.run_name(res, ri.DEFAULT_WINDOW[res], BASELINE_SET, seed)
            if not ri.finished(name):
                continue
            b = ri.result_row(name)
            out = OUT / "baseline" / name
            r = {"res_kb": kb(res), "window": b["window"], "variant": "baseline",
                 "seed": seed, "honest_val": b["honest_val"],
                 "stopped_at_epoch": b["stopped_at_epoch"], "stopped_by": b["stopped_by"],
                 "test_windows": b.get("test_windows"),
                 "akita_valid_regions": b.get("akita_valid_regions"),
                 "akita_test_regions": b.get("akita_test_regions"),
                 "val_avg": _json(out / "val_eval_avg.json"),
                 "test_windows_avg": _json(out / "test_eval_avg.json"),
                 "akita_test_regions_avg": _median(
                     out / "crosscell_test_avg/predict_gm12878.csv", "model")}
            t = ri.log_of(name)
            m = re.search(r"(\d+) train windows \(", t.read_text(errors="replace"))
            r["train_windows"] = int(m.group(1)) if m else None
            m = re.search(r"([0-9.]+)M params", t.read_text(errors="replace"))
            r["params_M"] = float(m.group(1)) if m else None
            rows.append(r)
    return rows


def map_rows():
    """Measured maps of other cells on GM12878's 413 test regions (from the
    baseline run's crosscell_test_maps; identical for both seeds)."""
    import pandas as pd
    rows = []
    for res in (25000, 10000, 5000):
        name = ri.run_name(res, ri.DEFAULT_WINDOW[res], BASELINE_SET, 0)
        f = OUT / "baseline" / name / "crosscell_test_maps/predict_gm12878.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for c in df.columns:
            if c.startswith("base_"):
                rows.append({"res_kb": kb(res), "map": c[5:],
                             "akita_test_regions": round(float(df[c].median()), 4),
                             "n": int(df[c].notna().sum())})
    return rows


def heldout_rows():
    """Held-out cells, 413 regions: model (A variants and the baseline) and
    every measured map of another cell incl. consensus11."""
    import pandas as pd
    rows = []
    srcs = []
    for res in (25000, 10000, 5000):
        for seed in SEEDS:
            for v in ("refin", "refres"):
                srcs.append((res, v, seed, OUT / run_name(res, v, seed) / "crosscell_heldout"))
            bn = ri.run_name(res, ri.DEFAULT_WINDOW[res], BASELINE_SET, seed)
            srcs.append((res, "baseline", seed, OUT / "baseline" / bn / "crosscell_heldout"))
    for res, v, seed, d in srcs:
        for cell in HELDOUT:
            f = d / f"predict_{cell}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            base = {c[5:]: float(df[c].median()) for c in df if c.startswith("base_")}
            single = {k: x for k, x in base.items() if k != "consensus11"}
            b = max(single, key=single.get) if single else None
            rows.append({"res_kb": kb(res), "variant": v, "seed": seed, "cell": cell,
                         "model": round(float(df["model"].median()), 4),
                         "n": int(df["model"].notna().sum()),
                         "consensus11": round(base["consensus11"], 4) if "consensus11" in base else None,
                         "best_other_map": round(single[b], 4) if b else None,
                         "best_other_map_cell": b})
    return rows


def write_csv(fn, rows):
    ri_out = ri.OUT
    try:
        ri.OUT = OUT
        ri.write_csv(fn, rows)
    finally:
        ri.OUT = ri_out


def summarize(quiet=False):
    rows = baseline_rows()
    for log in sorted(OUT.glob("*kb_w*__*__s*.log")):
        name = log.stem
        if re.fullmatch(r"\d+kb_w\d+__.+__s\d+", name) and finished(name):
            rows.append(result_row(name))
    write_csv("runs.csv", rows)
    keys = ["honest_val", "test_windows", "akita_valid_regions", "akita_test_regions",
            "val_native", "val_avg", "test_windows_avg", "akita_test_regions_avg",
            "ref_alone_val", "ref_alone_test_windows", "ref_alone_akita_valid_regions",
            "ref_alone_akita_test_regions", "train_windows", "params_M"]
    groups = {}
    for r in rows:
        groups.setdefault((r["res_kb"], r["variant"]), []).append(r)
    agg = []
    for (rk, v), rs in sorted(groups.items(), key=lambda x: (-x[0][0], x[0][1])):
        rec = {"res_kb": rk, "variant": v, "n_seeds": len(rs),
               "epochs": ";".join(str(r["stopped_at_epoch"]) for r in rs)}
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if not vals:
                continue
            rec[k] = round(sum(vals) / len(vals), 4)
            rec[k + "_seeds"] = ";".join(f"{x:.4f}" if isinstance(x, float) else str(x)
                                         for x in vals)
        agg.append(rec)
    write_csv("summary.csv", agg)
    write_csv("heldout.csv", heldout_rows())
    write_csv("maps_gm12878.csv", map_rows())
    status(agg, quiet)
    return rows, agg


def status(agg, quiet=False):
    lines = ["# coarse_gap status", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}. Mean of finished seeds; "
             "seed values in runs/coarse_gap/summary.csv.", "",
             "| res | variant | seeds | val | test windows | Akita test | extra |",
             "|---|---|---|---|---|---|---|"]
    for r in agg:
        extra = []
        for k, t in (("val_native", "val native"), ("val_avg", "val avg"),
                     ("test_windows_avg", "test avg"),
                     ("akita_test_regions_avg", "Akita avg"),
                     ("ref_alone_val", "ref val"),
                     ("ref_alone_test_windows", "ref test"),
                     ("ref_alone_akita_test_regions", "ref Akita")):
            if r.get(k) is not None:
                extra.append(f"{t} {r[k]:.3f}")
        f = lambda k: f"{r[k]:.3f}" if r.get(k) is not None else ""
        lines.append(f"| {r['res_kb']} kb | {r['variant']} | {r['n_seeds']} | {f('honest_val')} "
                     f"| {f('test_windows')} | {f('akita_test_regions')} | {'; '.join(extra)} |")
    running = [p.stem for p in OUT.glob("*kb_w*__*__s*.log")
               if re.fullmatch(r"\d+kb_w\d+__.+__s\d+", p.stem) and not finished(p.stem)]
    if running:
        lines += ["", "Training now: " + ", ".join(sorted(running))]
    # 12-cell reference runs (scripts/multicell_expanded.py --stage refsummary)
    rs = HM / "runs/multicell_reference/reference_summary.csv"
    if rs.exists():
        import pandas as pd
        d = pd.read_csv(rs)
        lines += ["", "## 12-cell training with reference map (runs/multicell_reference)", "",
                  "Median over cells, 413 Akita test regions (test windows for training "
                  "cells in the last columns).", "",
                  "| res | run | cells | model | reference alone | best other map | "
                  "all12honest (no ref) | beats ref | beats best map | "
                  "test win model / ref / all12honest |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        g = lambda r, c: "" if c not in r or pd.isna(r[c]) else (
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]))
        for _, r in d.iterrows():
            lines.append(
                f"| {r['res_kb']} kb | {r['run']} | {r['cells']} | {g(r, 'median_model')} "
                f"| {g(r, 'median_reference_alone')} | {g(r, 'median_best_other_map')} "
                f"| {g(r, 'median_all12honest_no_reference')} | {g(r, 'beats_reference')} "
                f"| {g(r, 'beats_best_other_map')} | {g(r, 'median_test_windows')} / "
                f"{g(r, 'median_test_windows_reference_alone')} / "
                f"{g(r, 'median_test_windows_all12honest_no_reference')} |")
    lr = HM / "runs/learned_reference/learned_reference.csv"
    if lr.exists():
        import pandas as pd
        lines += ["", "## Section 16: embedded and learned consensus (runs/learned_reference)", ""]
        for _, r in pd.read_csv(lr).iterrows():
            items = [f"{k} {v}" for k, v in r.items() if k != "res_kb" and pd.notna(v)]
            lines.append(f"- {r['res_kb']} kb: " + "; ".join(items))
    drivers = [HM / "runs/multicell_reference/driver.log",
               HM / "runs/multicell_reference/all12multires_train.log",
               HM / "runs/akita_ref/driver.log",
               HM / "runs/multicell_reference/driver_5_2kb.log",
               HM / "runs/learned_reference/driver.log"]
    q = []
    for d in drivers:
        if d.exists():
            last = [x for x in d.read_text(errors="replace").splitlines()
                    if x[:4].isdigit()]
            if last:
                q.append(f"- {d.parent.name}/{d.name}: {last[-1]}")
    if q:
        lines += ["", "## Queue (last milestone per driver)", "",
                  "scripts/gpu_scheduler.py: GPU jobs in priority order, each "
                  "started the moment the GPUs are free and its inputs exist; caches, "
                  "priors, staging, evaluations and analyses run on the CPU ahead of "
                  "time. Done: multi-resolution model, dropout and hard-split runs, "
                  "cascade, 5 and 2 kb pairs, all12refin. Now: the dedicated "
                  "minimal-input models (runs/benchmark/QUEUE_REQUEST.md), then "
                  "Track 1b RAD21 degron in training (all12deg_refres_drop, "
                  "all12deg_honest_drop, 25/10 kb; section 25), Track 2 "
                  "(all12refin_drop, all12refin_drop_ctcfac, all12refres_drop_varw, "
                  "25/10 kb; section 26), Track 1a HFF with SMC1A as cohesin "
                  "(all12hffcoh_refres_drop 25/10 kb), each with the perturbation, "
                  "AML and hard-region re-evaluations, then the washout test. "
                  "2 kb trainings read memory-mapped windows on local disk "
                  "(~/scratch/hicmx_stage); no CPU job starts while one loads.", ""] + q
    tmp = OUT / "STATUS.md.part"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(OUT / "STATUS.md")
    if not quiet:
        print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["train", "summary"], required=True)
    ap.add_argument("--slots", default="0,1", help="GPU id per concurrent run")
    ap.add_argument("--res", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=1000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        run_queue(all_jobs(a.res, a.variants), [s for s in a.slots.split(",") if s], a)
    summarize()
    print("COARSE_GAP_DONE", flush=True)


if __name__ == "__main__":
    main()
