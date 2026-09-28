"""Multi-cell retraining on Akita's five cell types, outside Akita's excluded regions.

Akita trains one network on HFF and H1-hESC Micro-C and GM12878, IMR90 and
HCT116 in situ Hi-C. Here the best single-cell configuration (the resolved
config of runs/motif_grid/full__all13__4motifs: 13 ChIP tracks, CTCF, YY1,
ZNF143 and MAZ motif planes, GC/CpG, no DNA encoder, no distance bias, 2 kb,
oe_proper.py targets) is retrained on several cells at once, each cell with
its own ChIP. Training windows are restricted to Akita's training sequences
(`data.train_regions_bed`, built by scripts/akita_regions.py); validation is
100 held-out windows per cell with patience-based early stopping, both GPUs.

Runs (names select cells and options):
  gm1         GM12878 alone with the region filter (isolates the filter's effect)
  hic3        GM12878, IMR90, HCT116 (in situ Hi-C only)
  all5        the three Hi-C cells plus HFF and H1 Micro-C
  all5assay   all5 with the assay indicator row (0 Hi-C, 1 Micro-C)

Stages, each skipped when its output exists:
  precompute  window cache per run (CPU; several can run in parallel)
  train       torchrun on --gpus
  eval        per training cell: scripts/eval_val_windows.py --split test --sample
              (the disjoint Akita test windows of that cell's own target), and
              scripts/crosscell_predict.py --tta on the 413 Akita test regions for
              every cell with a target (training and unseen cells), with the
              measured maps of the other cells as baselines
  The single-cell reference checkpoint is scored with crosscell_predict under
  the run name "ref".

    python scripts/multicell_akita.py --runs gm1 hic3 all5 all5assay --stage all
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
REF_RUN = HM / "runs/motif_grid/full__all13__4motifs"
REF_CFG = HM / "runs/motif_grid/full__all13__4motifs.yaml"
TRAIN_BED = Path.home() / "data/hic-multiome/akita/akita_train_regions.bed"
OUT = HM / "runs/multicell"
CACHE = REPO / "cache"
PY = sys.executable
TORCHRUN = str(Path(sys.executable).parent / "torchrun")

ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]

# name: (target O/E cool, ChIP bigwig dir, assay)
CELLS = {
    "gm12878": (OE / "gm12878_proper_2kb.cool",
                Path.home() / "data/hic-multiome/hg38_gm12878", "hic"),
    "imr90": (OE / "imr90_proper_2kb.cool", HM / "hg38_imr90", "hic"),
    "hct116": (OE / "hct116_proper_2kb.cool", HM / "hg38_hct116", "hic"),
    "hff": (OE / "hff_proper_2kb.cool", HM / "hg38_hff", "microc"),
    "h1": (OE / "h1_proper_2kb.cool", HM / "hg38_h1", "microc"),
    # never trained on; scored with crosscell_predict only
    "k562": (OE / "k562_proper_2kb.cool", HM / "hg38_k562", "hic"),
    "hmec": (OE / "hmec_rep1_ice_2kb.cool", HM / "hg38_hmec", "hic"),
    "huvec": (OE / "huvec_rep1_ice_2kb.cool", HM / "hg38_huvec", "hic"),
}
BASELINE_CELLS = ["gm12878", "imr90", "hct116", "hff", "h1", "k562"]
RUNS = {
    "gm1": (["gm12878"], False),
    "hic3": (["gm12878", "imr90", "hct116"], False),
    "all5": (["gm12878", "imr90", "hct116", "hff", "h1"], False),
    "all5assay": (["gm12878", "imr90", "hct116", "hff", "h1"], True),
}


def missing_marks(bw_dir: Path) -> list[str]:
    have = {p.stem.upper() for p in bw_dir.glob("*.bigwig")}
    return [t for t in ALL13 if t.upper() not in have]


def run_config(name: str, val_per_cell: int, patience: int, max_epochs: int) -> dict:
    cells, assay = RUNS[name]
    cfg = copy.deepcopy(yaml.safe_load(open(REF_CFG)))
    cfg["data"]["samples"] = [
        {"name": c, "cool_path": str(CELLS[c][0]), "bigwig_dir": str(CELLS[c][1]),
         "missing_tracks": missing_marks(CELLS[c][1]), "assay": CELLS[c][2]}
        for c in cells]
    cfg["data"]["train_regions_bed"] = str(TRAIN_BED)
    cfg["data"]["assay_indicator"] = assay
    cfg["train"].update({
        "val_exclude_overlap": False,   # as run, before the honest default
        "output_dir": str(OUT / name), "cache_dir": str(CACHE / f"mc_{name}"),
        "val_regions": val_per_cell * len(cells), "patience_epochs": patience,
        "max_epochs": max_epochs, "resident_dtype": "float16", "resume": False,
        "save_checkpoints": True, "track_dropout": 0.0,
    })
    return cfg


def done(log: Path) -> bool:
    t = log.read_text(errors="replace") if log.exists() else ""
    return "early stop at epoch" in t or "done." in t


STOP_RE = re.compile(r"early stop at epoch (\d+) \(best val ([-0-9.]+), (\d+) epochs no improvement\)")


def stop_info(name: str, patience: int):
    """(epoch, best_val, 'patience' | 'cap') of the LAST stop of a run, over
    the first training log and its continuation log, or None."""
    t = ""
    for f in (OUT / f"{name}.log", OUT / f"{name}.cont.log"):
        if f.exists():
            t += f.read_text(errors="replace")
    m = STOP_RE.findall(t)
    if not m:
        return None
    ep, best, noimp = m[-1]
    return int(ep), float(best), "patience" if int(noimp) >= patience else "cap"


def sh(cmd, log: Path, env=None):
    print("[cmd]", " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as fh:
        return subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=REPO).returncode


def crosscell(tag: str, cfg_path: Path, ckpt: Path, out: Path, gpu: str, cells=None):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    for cell, (cool, bw, assay) in CELLS.items():
        if cells and cell not in cells:
            continue
        if not cool.exists() or not bw.exists():
            print(f"[skip] {tag}/{cell}: target or ChIP missing", flush=True)
            continue
        if (out / f"predict_{cell}.csv").exists():
            continue
        others = [f"{o}={CELLS[o][0]}" for o in BASELINE_CELLS
                  if o != cell and CELLS[o][0].exists()]
        sh([PY, "scripts/crosscell_predict.py", "--config", cfg_path,
            "--checkpoint", ckpt, "--cell", cell, "--bigwig-dir", bw,
            "--target", cool, "--tta", "--assay", assay, "--out-dir", out,
            "--others", *others], out / f"predict_{cell}.log", env)


def summarize():
    """Tables from all finished runs: test730.csv (eval_val_windows per training
    cell) and crosscell.csv (crosscell_predict medians with the best baseline),
    plus train_windows.csv (cache sizes and windows removed by the region filter)."""
    import csv

    import pandas as pd

    names = ["ref"] + list(RUNS)
    t_rows, c_rows, w_rows = [], [], []
    for tag, rdir in (("ref", REF_RUN), ("ref_seed1", Path(f"{REF_RUN}__seed1"))):
        if (rdir / "test_eval.json").exists():
            r = json.loads((rdir / "test_eval.json").read_text())
            t_rows.append({"run": tag, "cell": "gm12878", "n": r["n"],
                           "oe_median_r": round(r["oe_median_r"], 4)})
    for name in names:
        run = OUT / name
        for j in sorted(run.glob("test_eval_*.json")):
            r = json.loads(j.read_text())
            t_rows.append({"run": name, "cell": r["sample"], "n": r["n"],
                           "oe_median_r": round(r["oe_median_r"], 4)})
        # best checkpoint as it stood at epoch 100 (snapshot), where taken
        for j in sorted(run.glob("ep100_test_*.json")):
            r = json.loads(j.read_text())
            t_rows.append({"run": f"{name}@ep100", "cell": r["sample"], "n": r["n"],
                           "oe_median_r": round(r["oe_median_r"], 4)})
        for f in sorted((run / "crosscell").glob("predict_*.csv")):
            df = pd.read_csv(f)
            cell = f.stem.replace("predict_", "")
            base = {c.replace("base_", ""): df[c].median() for c in df.columns
                    if c.startswith("base_")}
            best = max(base, key=base.get) if base else ""
            c_rows.append({"run": name, "cell": cell,
                           "trained_on_cell": name != "ref" and cell in RUNS[name][0]
                           or (name == "ref" and cell == "gm12878"),
                           "n": int(df["model"].notna().sum()),
                           "model": round(float(df["model"].median()), 4),
                           "best_baseline_cell": best,
                           "best_baseline": round(float(base[best]), 4) if best else None,
                           **{f"base_{k}": round(float(v), 4) for k, v in base.items()}})
        man = CACHE / f"mc_{name}" / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            per = {}
            for sh_ in m["shards"]:
                d = per.setdefault(sh_["sample"], {"train": 0, "test": 0})
                d[sh_["split"]] += sh_["n_windows"]
            for smp, d in per.items():
                w_rows.append({"run": name, "cell": smp, "train_windows": d["train"],
                               "test_windows": d["test"]})
            w_rows.append({"run": name, "cell": "ALL",
                           "train_windows": sum(d["train"] for d in per.values()),
                           "test_windows": sum(d["test"] for d in per.values()),
                           "removed_by_train_regions": m.get("n_outside_train_regions")})
    s_rows = []
    for name in RUNS:
        si = stop_info(name, 10)
        if si:
            t = "".join(f.read_text(errors="replace") for f in
                        (OUT / f"{name}.log", OUT / f"{name}.cont.log") if f.exists())
            hist = "; ".join(f"epoch {e} {'patience' if int(n) >= 10 else 'cap'}"
                             for e, _, n in STOP_RE.findall(t))
            s_rows.append({"run": name, "stopped_at_epoch": si[0], "best_val": si[1],
                           "stopped_by": si[2], "all_stops": hist})
    for fn, rows in (("test730.csv", t_rows), ("crosscell.csv", c_rows),
                     ("train_windows.csv", w_rows), ("training_stop.csv", s_rows)):
        if not rows:
            continue
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(OUT / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {OUT / fn}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS) + ["ref"])
    ap.add_argument("--stage", choices=["precompute", "train", "eval", "all", "summary"],
                    default="all")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--val-per-cell", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="eval stage: only these cells (split the work across GPUs)")
    ap.add_argument("--continue-capped", action="store_true",
                    help="train stage: resume a run whose last stop was the "
                         "max_epochs cap (not patience) up to --max-epochs")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "summary":
        summarize()
        return
    gpus = [g for g in a.gpus.split(",") if g]

    for name in a.runs:
        if name == "ref":
            if a.stage in ("eval", "all"):
                ck = REF_RUN / (REF_RUN / "best").read_text().strip()
                crosscell("ref", ck / "config.yaml", ck, OUT / "ref" / "crosscell", gpus[0],
                          a.cells)
            continue
        cfg = run_config(name, a.val_per_cell, a.patience, a.max_epochs)
        cp = OUT / f"{name}.yaml"
        if not done(OUT / f"{name}.log"):       # a finished run keeps its config
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        cache = Path(cfg["train"]["cache_dir"])
        if a.stage in ("precompute", "all") and not (cache / "manifest.json").exists():
            print(f"[precompute] {name}", flush=True)
            if sh([PY, "-m", "hicmultiome.precompute", "--config", cp],
                  OUT / f"{name}.precompute.log"):
                raise SystemExit(f"precompute failed for {name}")
        log = OUT / f"{name}.log"
        if a.stage in ("train", "all") and not done(log):
            if not (cache / "manifest.json").exists():
                raise SystemExit(f"{name}: no cache; run --stage precompute first")
            log.write_text("")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
            sh([TORCHRUN, "--standalone", f"--nproc_per_node={len(gpus)}",
                "-m", "hicmultiome.train", "--config", cp], log, env)
            if not done(log):
                raise SystemExit(f"training failed for {name}; see {log}")
        # a run cut off by the max_epochs cap while still improving continues
        # from its latest checkpoint (weights, optimizer, step/epoch counters,
        # best validation score), with --max-epochs as the new cap and
        # patience as the stopping rule; repeated until patience stops it
        while (a.stage in ("train", "all") and a.continue_capped
               and (si := stop_info(name, a.patience)) and si[2] == "cap"
               and si[0] < a.max_epochs):
            ccfg = yaml.safe_load(open(cp))
            ccfg["train"].update({
                "resume": True,
                "resume_from": str(OUT / name / "checkpoint-latest"),
                "max_epochs": a.max_epochs, "patience_epochs": a.patience})
            # checkpoint-latest is written before that epoch's validation, so
            # its stored best can lag by the last improvement; carry the true
            # best over so a worse model cannot replace checkpoint-best
            pj = OUT / name / "checkpoint-latest" / "progress.json"
            pr = json.loads(pj.read_text())
            if si[1] > pr["best"]:
                pr["best"] = si[1]
                pj.write_text(json.dumps(pr))
            ccp = OUT / f"{name}.cont.yaml"
            yaml.safe_dump(ccfg, open(ccp, "w"), sort_keys=False)
            print(f"[continue] {name} from epoch {si[0]} (cap) to max {a.max_epochs}",
                  flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
            n0 = len(STOP_RE.findall((OUT / f"{name}.cont.log").read_text(errors="replace"))
                     if (OUT / f"{name}.cont.log").exists() else [])
            sh([TORCHRUN, "--standalone", f"--nproc_per_node={len(gpus)}",
                "-m", "hicmultiome.train", "--config", ccp], OUT / f"{name}.cont.log", env)
            if len(STOP_RE.findall((OUT / f"{name}.cont.log").read_text(errors="replace"))) <= n0:
                raise SystemExit(f"continuation failed for {name}")
        if a.stage in ("eval", "all"):
            run = OUT / name
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0])
            for cell in RUNS[name][0]:
                if a.cells and cell not in a.cells:
                    continue
                if not (run / f"test_eval_{cell}.json").exists():
                    sh([PY, "scripts/eval_val_windows.py", "--run", run, "--split", "test",
                        "--sample", cell], OUT / f"{name}.eval.log", env)
            ck = run / (run / "best").read_text().strip()
            crosscell(name, ck / "config.yaml", ck, run / "crosscell", gpus[0], a.cells)
        print(f"[done] {name} ({a.stage})", flush=True)
    print("MULTICELL_DONE", flush=True)


if __name__ == "__main__":
    main()
