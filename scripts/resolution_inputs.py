"""Input selection per resolution at the level of chromatin groups (GM12878).

The 13 ChIP marks and 4 motif planes were chosen at 2 kb. At 5, 10 and 25 kb
the target is dominated by compartments and TAD-scale structure, so the useful
inputs may differ. Single-mark swaps inside a group cost almost nothing
(registry section 3), so the search drops whole groups:

  ctcf        CTCF ChIP
  cohesin     RAD21, SMC3
  active      H3K27ac, H3K4me1, H3K4me2, H3K4me3, H3K9ac
  genebody    H3K36me3, H3K79me2, H4K20me1
  repressive  H3K27me3, H3K9me3
  motifs      MA0139.2 CTCF, MA0095.3 YY1, MA0088.2 ZNF143, MA1522.1 MAZ
  (GC and CpG are always kept)

Input sets per resolution: all, each group dropped (no_<group>), repressive
plus gene body ChIP only (repr_genebody), no ChIP at all (seqOnly: GC/CpG and
the 4 motifs), and combo: every group whose removal raised the mean honest
validation score dropped together (only when at least two did).

Every run: GM12878 only, one cache per (resolution, window) with Akita's
training regions (data.train_regions_bed), 100 validation windows (other
windows: the same fraction of the training windows) with every overlapping
training window removed (train.val_exclude_overlap), patience 10
epochs, the multi-cell configuration otherwise (scripts/multicell_expanded.py),
seeds 0 and 1 (the seed also picks the validation windows), one GPU per run.
Selection uses the honest validation score only (mean over seeds). Reported:
eval_val_windows --split test (every disjoint test window of the cache) and
crosscell_predict --tta on Akita's 413 test regions; crosscell_predict on
Akita's 419 validation regions (held out of training, disjoint from test)
gives a score on identical pixels for every window size and serves as the
selection score between window sizes.

Windows: the multi-cell windows (384 / 192 / 80 bins at 5 / 10 / 25 kb) for
the group search; then, with each resolution's selected inputs, 256 bins at 5
and 10 kb and 160 and 256 bins at 25 kb. Windows spanning more than 2 Mb use
targets from oe_proper.py --max-dist wide enough for the window.

    python scripts/resolution_inputs.py --stage precompute
    python scripts/resolution_inputs.py --stage groups --slots 0,0,1,1
    python scripts/resolution_inputs.py --stage windows --slots 0,0,1,1
    python scripts/resolution_inputs.py --stage summary
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
MX = HM / "runs/multicell_expanded"
OUT = HM / "runs/resolution_inputs"
CACHE = HM / "cache_expanded"
PY = sys.executable
AKITA = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization")
TEST_REGIONS = AKITA / "akita_validation/sequences_test_list.tsv"
VALID_REGIONS = Path.home() / "data/hic-multiome/akita/sequences_valid_list.tsv"
GM_BW = Path.home() / "data/hic-multiome/hg38_gm12878"

GROUPS = {
    "ctcf": ["CTCF"],
    "cohesin": ["RAD21", "SMC3"],
    "active": ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac"],
    "genebody": ["H3K36me3", "H3K79me2", "H4K20me1"],
    "repressive": ["H3K27me3", "H3K9me3"],
}
MOTIFS = ["MA0139.2", "MA0095.3", "MA0088.2", "MA1522.1"]
ALL13 = [t for g in GROUPS.values() for t in g]
GROUP_NAMES = list(GROUPS) + ["motifs"]
RES = [5000, 10000, 25000]
DEFAULT_WINDOW = {5000: 384, 10000: 192, 25000: 80}
EXTRA_WINDOWS = {5000: [256], 10000: [256], 25000: [160, 256]}
SEEDS = [0, 1]
MAX_DIST = 2_000_000


def inputs(dropped: set[str]) -> tuple[list[str], list[str]]:
    tracks = [t for g, ts in GROUPS.items() if g not in dropped for t in ts]
    return tracks, ([] if "motifs" in dropped else list(MOTIFS))


def input_sets() -> dict[str, tuple[list[str], list[str]]]:
    s = {"all": inputs(set())}
    for g in GROUP_NAMES:
        s[f"no_{g}"] = inputs({g})
    s["repr_genebody"] = (GROUPS["repressive"] + GROUPS["genebody"], [])
    s["seqOnly"] = ([], list(MOTIFS))
    return s


def band_tag(res: int, size: int) -> str | None:
    """Wider oe_proper band for windows spanning more than 2 Mb (None: default)."""
    span = res * size
    return None if span <= MAX_DIST else f"{span // 1000}k"


def gm_target(res: int, size: int) -> Path:
    t = band_tag(res, size)
    k = res // 1000
    if t is None:
        return OE / f"gm12878_proper_{k}kb.cool"
    # the widest band built for this resolution that covers the window
    have = sorted(OE.glob(f"gm12878_proper_{k}kb_band*k.cool"),
                  key=lambda p: int(re.search(r"band(\d+)k", p.name).group(1)))
    for p in have:
        if int(re.search(r"band(\d+)k", p.name).group(1)) * 1000 >= res * size:
            return p
    return OE / f"gm12878_proper_{k}kb_band{t}.cool"


def cache_name(res: int, size: int) -> str:
    return f"ri_gm_{res // 1000}kb_w{size}"


def base_config(res: int, size: int) -> dict:
    """The multi-cell configuration at res, GM12878 only, window `size`."""
    cfg = copy.deepcopy(yaml.safe_load(open(MX / f"all12_{res // 1000}kb.yaml")))
    gm = next(s for s in cfg["data"]["samples"] if s["name"] == "gm12878")
    gm["cool_path"] = str(gm_target(res, size))
    cfg["data"]["samples"] = [gm]
    cfg["window"]["size"], cfg["window"]["stride"] = size, size // 2
    cfg["train"]["cache_dir"] = str(CACHE / cache_name(res, size))
    return cfg


def run_name(res, size, iset, seed):
    return f"{res // 1000}kb_w{size}__{iset}__s{seed}"


def parse_run(name):
    m = re.fullmatch(r"(\d+)kb_w(\d+)__(.+)__s(\d+)", name)
    return int(m.group(1)) * 1000, int(m.group(2)), m.group(3), int(m.group(4))


def n_train_windows(res, size) -> int:
    m = json.loads((CACHE / cache_name(res, size) / "manifest.json").read_text())
    return sum(s["n_windows"] for s in m["shards"] if s["split"] == "train")


def val_regions(res, size) -> int:
    """100 validation windows at the default window; other windows hold out the
    same fraction of their training windows (e.g. 18 of 257 at 25 kb, 256 bins),
    so overlap removal leaves the same share of training windows."""
    if size == DEFAULT_WINDOW[res]:
        return 100
    return max(10, round(100 * n_train_windows(res, size) / n_train_windows(res, DEFAULT_WINDOW[res])))


def run_config(res, size, iset, seed, tracks, motifs, patience, max_epochs):
    cfg = base_config(res, size)
    cfg["train"].update({
        "output_dir": str(OUT / run_name(res, size, iset, seed)),
        "select_tracks": tracks, "select_motifs": motifs,
        "val_regions": val_regions(res, size), "val_exclude_overlap": True, "seed": seed,
        "patience_epochs": patience, "max_epochs": max_epochs,
        "resume": False, "save_checkpoints": True, "track_dropout": 0.0,
    })
    return cfg


STOP_RE = re.compile(r"early stop at epoch (\d+) \(best val ([-0-9.]+), (\d+) epochs no improvement\)")


def log_of(name):
    return OUT / f"{name}.log"


def finished(name) -> bool:
    p = log_of(name)
    t = p.read_text(errors="replace") if p.exists() else ""
    return "early stop at epoch" in t or "done." in t


def stop_info(name):
    m = STOP_RE.findall(log_of(name).read_text(errors="replace")) if log_of(name).exists() else []
    if not m:
        return None
    ep, best, noimp = m[-1]
    return int(ep), float(best), int(noimp)


def sh(cmd, log: Path, env=None, mode="a"):
    with open(log, mode) as fh:
        return subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=REPO).returncode


# ---------------------------------------------------------------- precompute
def precompute(workers):
    combos = [(r, DEFAULT_WINDOW[r]) for r in RES] + [(r, w) for r in RES for w in EXTRA_WINDOWS[r]]
    for res, size in combos:
        cfg = base_config(res, size)
        cache = Path(cfg["train"]["cache_dir"])
        if (cache / "manifest.json").exists():
            continue
        tgt = Path(cfg["data"]["samples"][0]["cool_path"])
        if not tgt.exists():
            print(f"[skip] {cache.name}: target {tgt} missing", flush=True)
            continue
        cp = OUT / f"{cache.name}.precompute.yaml"
        yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        print(f"[precompute] {cache.name}", flush=True)
        if sh([PY, "-m", "hicmultiome.precompute", "--config", cp, "--workers", workers],
              OUT / f"{cache.name}.precompute.log"):
            raise SystemExit(f"precompute failed for {cache.name}")


# ---------------------------------------------------------------- training + scoring
def train_one(name, tracks, motifs, gpu, a):
    res, size, iset, seed = parse_run(name)
    if not (Path(base_config(res, size)["train"]["cache_dir"]) / "manifest.json").exists():
        print(f"[skip] {name}: no cache", flush=True)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if not finished(name):
        cfg = run_config(res, size, iset, seed, tracks, motifs, a.patience, a.max_epochs)
        cp = OUT / f"{name}.yaml"
        yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        print(f"[train] {name} on GPU {gpu}", flush=True)
        sh([PY, "-m", "hicmultiome.train", "--config", cp], log_of(name), env, mode="w")
        if not finished(name):
            print(f"[FAILED] {name}", flush=True)
            return
    si = stop_info(name)
    if si and si[2] < a.patience:
        print(f"[WARN] {name} stopped by the epoch cap at {si[0]}", flush=True)
    score_one(name, gpu)


def score_one(name, gpu):
    run = OUT / name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if not (run / "test_eval.json").exists():
        sh([PY, "scripts/eval_val_windows.py", "--run", run, "--split", "test",
            "--out", run / "test_eval.json"], OUT / f"{name}.eval.log", env)
    ck = run / (run / "best").read_text().strip()
    res = parse_run(name)[0]
    k = res // 1000
    for tag, regions in (("valid", VALID_REGIONS), ("test", TEST_REGIONS)):
        d = run / f"crosscell_{tag}"
        if (d / "predict_gm12878.csv").exists():
            continue
        sh([PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
            "--checkpoint", ck, "--cell", "gm12878", "--bigwig-dir", GM_BW,
            "--target", OE / f"gm12878_proper_{k}kb.cool", "--tta", "--assay", "hic",
            "--binsize", res, "--regions", regions, "--out-dir", d],
           OUT / f"{name}.crosscell.log", env)


def run_queue(jobs, slots, a):
    """jobs: [(name, tracks, motifs)]; slots: GPU id per concurrent worker."""
    lock = threading.Lock()
    todo = list(jobs)

    def worker(gpu):
        while True:
            with lock:
                if not todo:
                    return
                name, tr, mo = todo.pop(0)
            try:
                train_one(name, tr, mo, gpu, a)
            except Exception as e:                   # noqa: BLE001
                print(f"[ERROR] {name}: {e}", flush=True)

    ths = [threading.Thread(target=worker, args=(g,)) for g in slots]
    for i, t in enumerate(ths):
        t.start()
        time.sleep(20 if i < len(ths) - 1 else 0)   # stagger the cache loads
    for t in ths:
        t.join()


# ---------------------------------------------------------------- averaged 2 kb model
def avg2kb(cells, device, threads, jobs):
    """crosscell_predict of the 2 kb 12-cell model (all12) with --coarse: its
    native prediction block-averaged onto the 5, 10 and 25 kb grids and scored
    against those targets on the 413 Akita test regions. Output
    OUT/avg2kb/predict_<cell>.csv (columns model, model_5kb, model_10kb, model_25kb)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import multicell_expanded as mx
    run = MX / "all12"
    ck = run / (run / "best").read_text().strip()
    out = OUT / "avg2kb"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    todo = [c for c in cells if not (out / f"predict_{c}.csv").exists()]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                cell = todo.pop(0)
            _, bw, assay, _ = mx.CELLS[cell]
            coarse = [f"{r}={mx.target(cell, r)}" for r in RES if mx.target(cell, r).exists()]
            print(f"[avg2kb] {cell}", flush=True)
            rc = sh([PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
                     "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
                     "--target", mx.target(cell, 2000), "--tta", "--assay", assay,
                     "--binsize", 2000, "--device", device, "--coarse", *coarse,
                     "--out-dir", out], out / f"predict_{cell}.log", env, mode="w")
            if rc:
                print(f"[FAILED] avg2kb {cell}", flush=True)

    ths = [threading.Thread(target=worker) for _ in range(jobs)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


# ---------------------------------------------------------------- results
def result_row(name):
    res, size, iset, seed = parse_run(name)
    si = stop_info(name)
    r = {"res_kb": res // 1000, "window": size, "input_set": iset, "seed": seed,
         "honest_val": si[1] if si else None, "stopped_at_epoch": si[0] if si else None,
         "stopped_by": ("patience" if si and si[2] >= 10 else "cap") if si else None}
    tj = OUT / name / "test_eval.json"
    if tj.exists():
        j = json.loads(tj.read_text())
        r["test_windows_n"], r["test_windows"] = j["n"], round(j["oe_median_r"], 4)
    import pandas as pd
    for tag in ("valid", "test"):
        f = OUT / name / f"crosscell_{tag}" / "predict_gm12878.csv"
        if f.exists():
            df = pd.read_csv(f)
            r[f"akita_{tag}_regions"] = round(float(df["model"].median()), 4)
            r[f"akita_{tag}_n"] = int(df["model"].notna().sum())
    return r


def all_rows():
    rows = []
    for log in sorted(OUT.glob("*kb_w*__*__s*.log")):
        name = log.stem
        if re.fullmatch(r"\d+kb_w\d+__.+__s\d+", name) and finished(name):
            rows.append(result_row(name))
    return rows


def mean_by_set(rows, res, size, key="honest_val"):
    out = {}
    for r in rows:
        if r["res_kb"] * 1000 == res and r["window"] == size and r.get(key) is not None:
            out.setdefault(r["input_set"], []).append(r[key])
    return {k: sum(v) / len(v) for k, v in out.items() if len(v) == len(SEEDS)}


def select(rows):
    """Per resolution: the input set with the highest mean honest validation
    score at the default window (all seeds finished)."""
    sel = {}
    for res in RES:
        m = mean_by_set(rows, res, DEFAULT_WINDOW[res])
        if "all" not in m:
            continue
        best = max(m, key=m.get)
        sets = {**input_sets(), **combo_set(rows, res)}
        tr, mo = sets[best]
        sel[res] = {"input_set": best, "honest_val": round(m[best], 4),
                    "honest_val_all": round(m["all"], 4), "tracks": tr, "motifs": mo,
                    "window": DEFAULT_WINDOW[res]}
        # window: the highest mean score on Akita's validation regions (identical
        # pixels for every window size), once every window has all seeds
        wv = {}
        for w in [DEFAULT_WINDOW[res]] + EXTRA_WINDOWS[res]:
            v = mean_by_set(rows, res, w, "akita_valid_regions").get(best)
            if v is not None:
                wv[w] = v
        sel[res]["akita_valid_by_window"] = {str(w): round(v, 4) for w, v in wv.items()}
        if len(wv) == 1 + len(EXTRA_WINDOWS[res]):
            sel[res]["window"] = max(wv, key=wv.get)
    return sel


def combo_set(rows, res):
    """Every group whose removal raised the mean honest validation score,
    dropped together; only defined when at least two groups qualify."""
    m = mean_by_set(rows, res, DEFAULT_WINDOW[res])
    if "all" not in m or not all(f"no_{g}" in m for g in GROUP_NAMES):
        return {}
    up = {g for g in GROUP_NAMES if m[f"no_{g}"] > m["all"]}
    return {"combo": inputs(up)} if len(up) >= 2 else {}


def write_csv(fn, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    tmp = OUT / (fn + ".part")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.rename(OUT / fn)
    print(f"wrote {OUT / fn}", flush=True)


def _median(f: Path, col: str, regions=None):
    import pandas as pd
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if regions is not None:
        df = df[df["region"].isin(regions)]
    return round(float(df[col].median()), 4) if col in df and df[col].notna().any() else None


def compare_resolutions():
    """Per resolution and cell, on the 413 Akita test regions: the dedicated
    all-input 12-cell model (all12_<k>kb), the 2 kb model averaged onto the
    coarse grid (avg2kb), the 12-cell model with the selected inputs
    (all12sel_<k>kb, where one was trained), the best measured map of another
    cell, and replicate agreement; plus the disjoint test windows of the
    training cells. 'paired' columns restrict both models to the regions both
    scored."""
    import pandas as pd
    sys.path.insert(0, str(REPO / "scripts"))
    import multicell_expanded as mx
    rep = pd.read_csv(MX / "resolution_vs_replicates.csv")
    cells = mx.TRAIN12 + ["k562", "hmec_deep", "huvec_deep"]
    rows = []
    for res in [2000] + RES:
        k = res // 1000
        ded = MX / mx.run_name("all12", res)
        sel = MX / mx.run_name("all12sel", res)
        for cell in cells:
            f_ded = ded / "crosscell" / f"predict_{cell}.csv"
            f_avg = OUT / "avg2kb" / f"predict_{cell}.csv"
            f_sel = sel / "crosscell" / f"predict_{cell}.csv"
            rec = {"res_kb": k, "cell": cell, "trained_on_cell": cell in mx.TRAIN12,
                   "dedicated_all_inputs": _median(f_ded, "model")}
            if res > 2000:
                rec["averaged_2kb"] = _median(f_avg, f"model_{k}kb")
                if f_ded.exists() and f_avg.exists():
                    a_ = pd.read_csv(f_avg).dropna(subset=[f"model_{k}kb"])
                    d_ = pd.read_csv(f_ded).dropna(subset=["model"])
                    common = set(a_["region"]) & set(d_["region"])
                    rec["n_paired"] = len(common)
                    rec["dedicated_paired"] = _median(f_ded, "model", common)
                    rec["averaged_2kb_paired"] = _median(f_avg, f"model_{k}kb", common)
                rec["best_inputs"] = _median(f_sel, "model")
            if f_ded.exists():
                df = pd.read_csv(f_ded)
                base = {c[5:]: float(df[c].median()) for c in df if c.startswith("base_")}
                if base:
                    b = max(base, key=base.get)
                    rec["best_measured_map"], rec["best_measured_map_cell"] = round(base[b], 4), b
            for tag, run in (("test_windows_dedicated", ded), ("test_windows_best_inputs", sel)):
                j = run / f"test_eval_{cell}.json"
                if j.exists():
                    rec[tag] = round(json.loads(j.read_text())["oe_median_r"], 4)
            r = rep[(rep["res_kb"] == k) & (rep["cell"] == cell)]
            if len(r):
                for c in ("rep_insitu_single_lane", "rep_intact_deep"):
                    v = r.iloc[0].get(c)
                    rec[c] = None if pd.isna(v) else float(v)
            rows.append(rec)
    write_csv("resolution_compare.csv", rows)
    df = pd.DataFrame(rows)
    summ = []
    cols = [c for c in ("dedicated_all_inputs", "averaged_2kb", "best_inputs",
                        "best_measured_map", "test_windows_dedicated",
                        "test_windows_best_inputs", "rep_insitu_single_lane",
                        "rep_intact_deep") if c in df]
    for k, g in df.groupby("res_kb"):
        tr = g[g["trained_on_cell"]]
        rec = {"res_kb": k, "cells": "median of 12 training cells"}
        for c in cols:
            v = tr[c].dropna()
            rec[c] = round(float(v.median()), 4) if len(v) else None
        if "best_measured_map" in tr:
            both = tr.dropna(subset=["dedicated_all_inputs", "best_measured_map"])
            rec["dedicated_beats_map"] = f"{int((both['dedicated_all_inputs'] > both['best_measured_map']).sum())}/{len(both)}"
            for c, t in (("averaged_2kb", "averaged_beats_map"), ("best_inputs", "best_inputs_beats_map")):
                if c in tr:
                    b2 = tr.dropna(subset=[c, "best_measured_map"])
                    if len(b2):
                        rec[t] = f"{int((b2[c] > b2['best_measured_map']).sum())}/{len(b2)}"
        summ.append(rec)
        for cell in ("k562", "hmec_deep", "huvec_deep"):
            r = g[g["cell"] == cell]
            if len(r):
                summ.append({"res_kb": k, "cells": cell,
                             **{c: r.iloc[0][c] for c in cols if pd.notna(r.iloc[0][c])}})
    write_csv("resolution_compare_summary.csv", summ)


def summarize():
    try:
        compare_resolutions()
    except Exception as e:                      # noqa: BLE001
        print(f"[compare] skipped: {e}", flush=True)
    rows = all_rows()
    write_csv("runs.csv", rows)
    agg = []
    keys = ["honest_val", "test_windows", "akita_valid_regions", "akita_test_regions"]
    groups = {}
    for r in rows:
        groups.setdefault((r["res_kb"], r["window"], r["input_set"]), []).append(r)
    for (rk, w, s), rs in sorted(groups.items()):
        rec = {"res_kb": rk, "window": w, "input_set": s, "n_seeds": len(rs),
               "epochs": ";".join(str(r["stopped_at_epoch"]) for r in rs)}
        for k in keys:
            v = [r[k] for r in rs if r.get(k) is not None]
            rec[k] = round(sum(v) / len(v), 4) if v else None
            rec[k + "_seeds"] = ";".join(f"{x:.4f}" for x in v)
        agg.append(rec)
    write_csv("summary.csv", agg)
    sel = select(rows)
    if sel:
        tmp = OUT / "selection.json.part"
        tmp.write_text(json.dumps({str(k): v for k, v in sel.items()}, indent=1))
        tmp.rename(OUT / "selection.json")
        print(json.dumps({str(k): v["input_set"] for k, v in sel.items()}), flush=True)
    return rows, sel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["precompute", "avg2kb", "groups", "combo", "windows",
                                        "summary"], required=True)
    ap.add_argument("--cells", nargs="*", default=None, help="avg2kb: cells (default all)")
    ap.add_argument("--device", default="cuda", help="avg2kb: cuda or cpu")
    ap.add_argument("--threads", type=int, default=8, help="avg2kb: torch threads per job")
    ap.add_argument("--jobs", type=int, default=1, help="avg2kb: concurrent cells")
    ap.add_argument("--slots", default="0,0,1,1",
                    help="GPU id per concurrent run (two runs per GPU by default)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--res", nargs="*", type=int, default=RES)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=1000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    slots = [s for s in a.slots.split(",") if s]
    if a.stage == "precompute":
        precompute(a.workers)
    elif a.stage == "avg2kb":
        sys.path.insert(0, str(REPO / "scripts"))
        import multicell_expanded as mx
        avg2kb(a.cells or mx.TRAIN12 + ["k562", "hmec_deep", "huvec_deep"], a.device,
               a.threads, a.jobs)
        return
    elif a.stage == "groups":
        jobs = [(run_name(r, DEFAULT_WINDOW[r], s, seed), tr, mo)
                for seed in SEEDS for r in a.res for s, (tr, mo) in input_sets().items()]
        run_queue(jobs, slots, a)
    elif a.stage == "combo":
        rows = all_rows()
        jobs = []
        for r in a.res:
            for s, (tr, mo) in combo_set(rows, r).items():
                jobs += [(run_name(r, DEFAULT_WINDOW[r], s, seed), tr, mo) for seed in SEEDS]
        print(f"[combo] {[j[0] for j in jobs]}", flush=True)
        run_queue(jobs, slots, a)
    elif a.stage == "windows":
        _, sel = summarize()
        jobs = []
        for r in a.res:
            if r not in sel:
                print(f"[skip] windows at {r}: no selection yet", flush=True)
                continue
            s = sel[r]
            for w in EXTRA_WINDOWS[r]:
                jobs += [(run_name(r, w, s["input_set"], seed), s["tracks"], s["motifs"])
                         for seed in SEEDS]
        run_queue(jobs, slots, a)
    summarize()
    print("RESOLUTION_INPUTS_DONE", flush=True)


if __name__ == "__main__":
    main()
