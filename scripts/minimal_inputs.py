"""Practical minimal input sets (registry section 20).

Sets (every set keeps all motif channels of its base model; the sequence
features GC and CpG are always present):
  a  ctcf_h3k27ac   CTCF + H3K27ac: the C.Origami-like analogue (accessibility is
                    not a track here; H3K27ac stands in for ATAC)
  b  ctcf_rad21     CTCF + RAD21
  c  one_per_group  CTCF, RAD21, H3K27ac, H3K4me3, H3K27me3 (one mark per group)
  d  motifs_only    no ChIP
  full              all 13 ChIP tracks (reference)

Setups: the 12-cell reference-residual model at 25 and 10 kb (all12refres,
consensus leave-one-out reference) and GM12878 at 2 kb (akita_native setup,
honest validation, Akita's target).

Stages:
  absent   CPU: the existing models predicted with the other tracks set to their
           absent value (crosscell_predict --keep-tracks). The 2 kb
           akita_native_trackdrop model was trained with track dropout; the
           12-cell and akita_native_honest/refres models were not (track_dropout
           0), so their rows measure robustness outside the training
           distribution.
  configs  training configs for the dedicated minimal-set models (configs/minimal/)
  train    one dedicated training (GPU; run from the queue, never by hand here)
  eval     crosscell_predict of a trained dedicated model (CPU or GPU)
  summary  tables results/minimal_inputs/*.csv

    python scripts/minimal_inputs.py --stage absent
    python scripts/minimal_inputs.py --stage configs
    python scripts/minimal_inputs.py --stage train --run min_ctcf_rad21_refres_25kb --gpus 0,1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import multicell_expanded as mx                                      # noqa: E402

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OUT = HM / "runs/minimal_inputs"
RES_OUT = REPO / "results/minimal_inputs"
CFG_OUT = REPO / "configs/minimal"
PY = sys.executable
TORCHRUN = str(Path(sys.executable).parent / "torchrun")
THREADS = int(os.environ.get("MIN_THREADS", "4"))
AK2 = HM / "akita2kb"
AKITA_TARGET = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/"
                    "akita_published_GM12878_test_2048bp.cool")

SETS = {
    "full": mx.ALL13,
    "ctcf_h3k27ac": ["CTCF", "H3K27ac"],
    "ctcf_rad21": ["CTCF", "RAD21"],
    "one_per_group": ["CTCF", "RAD21", "H3K27ac", "H3K4me3", "H3K27me3"],
    "motifs_only": [],
}
SET_LETTER = {"ctcf_h3k27ac": "a", "ctcf_rad21": "b", "one_per_group": "c",
              "motifs_only": "d", "full": "full"}
# existing models for the absent-track evaluation: name -> (run dir, res, cells, dropout)
EXISTING = {
    "all12refres_25kb": (mx.REF_OUT / "all12refres_25kb", 25000, ["gm12878", "imr90", "k562"], False),
    "all12refres_10kb": (mx.REF_OUT / "all12refres_10kb", 10000, ["gm12878", "imr90", "k562"], False),
    "all12honest_25kb": (mx.REF_OUT / "all12honest_25kb", 25000, ["gm12878", "imr90", "k562"], False),
    "all12honest_10kb": (mx.REF_OUT / "all12honest_10kb", 10000, ["gm12878", "imr90", "k562"], False),
    "akita_native_trackdrop": (REPO / "runs/akita_native_trackdrop", 2048, ["gm12878"], True),
    "akita_native_honest": (HM / "runs/akita_ref/akita_native_honest", 2048, ["gm12878"], False),
    "akita_native_refres": (HM / "runs/akita_ref/akita_native_refres", 2048, ["gm12878"], False),
}
# dedicated trainings: base config per setup
BASE = {
    "refres_25kb": mx.REF_OUT / "all12refres_25kb.yaml",
    "refres_10kb": mx.REF_OUT / "all12refres_10kb.yaml",
    "gm2kb": REPO / "configs/akita_native_honest.yaml",
}
TRAIN_SETS = ["ctcf_h3k27ac", "ctcf_rad21", "one_per_group", "motifs_only"]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "minimal_inputs.log", "a") as fh:
        fh.write(line + "\n")


def sh(cmd, logf: Path, env=None) -> int:
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def cpu_env():
    n = str(THREADS)
    return dict(os.environ, OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n,
                CUDA_VISIBLE_DEVICES="")


def best_ckpt(run: Path) -> Path:
    b = run / "best"
    return run / (b.read_text().strip() if b.exists() else "checkpoint-best")


def predict(run: Path, res: int, cell: str, keep: list[str] | None, out_dir: Path,
            ref_run: str | None, env=None, device="cpu", keep_sets: dict | None = None,
            with_baselines: bool = False) -> int:
    """crosscell_predict for one cell; 2048 bp models score against Akita's
    GM12878 target (the section 1 metric), the others against the proper O/E
    target of the cell at res with the measured-map baselines."""
    ck = best_ckpt(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    if keep_sets:
        if all((out_dir / k / f"predict_{cell}.csv").exists() for k in keep_sets):
            return 0
    elif (out_dir / f"predict_{cell}.csv").exists():
        return 0
    extra = []
    if keep is not None:
        extra += ["--keep-tracks", ",".join(keep)]
    if keep_sets:
        extra += ["--keep-sets", *[f"{k}={','.join(v)}" for k, v in keep_sets.items()]]
    if res == 2048:
        cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
               "--checkpoint", ck, "--cell", cell, "--bigwig-dir",
               mx.CELLS[cell][1], "--target", AKITA_TARGET, "--binsize", 2048, "--tta",
               "--device", device, "--out-dir", out_dir, *extra,
               "--others", f"consensus4={AK2}/akita_consensus4_2048bp.cool",
               f"hff={AK2}/akita_hff_all_2048bp.cool"]
    else:
        _, bw, assay, ident = mx.CELLS[cell]
        # measured-map baselines only with the dedicated models' evaluation;
        # each cooler fetch over the share costs ~0.8 s per region
        others = ([f"{o}={mx.target(o, res)}" for o in mx.BASELINE_MAPS
                   if mx.CELLS[o][3] != ident and mx.target(o, res).exists()]
                  if with_baselines else [])
        if ref_run is not None:
            rf = mx.reference_of(mx.run_name(ref_run, res), cell)
            tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
            if with_baselines:
                others.append(f"{tag}={rf}")
            extra += ["--reference", rf]
        cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
               "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw, "--target",
               mx.target(cell, res), "--tta", "--assay", assay, "--binsize", res,
               "--device", device, "--out-dir", out_dir, *extra]
        if others:
            cmd += ["--others", *others]
    return sh(cmd, out_dir / f"predict_{cell}.log", env)


def stage_absent(models=None, sets=None):
    for name, (run, res, cells, _) in EXISTING.items():
        if models and name not in models:
            continue
        ref = "all12refres" if name.startswith("all12refres") else None
        ks = {k: v for k, v in SETS.items() if not sets or k in sets}
        for cell in cells:
            # one pass per model and cell; every keep set in the same pass
            rc = predict(run, res, cell, None, OUT / "absent" / name, ref, cpu_env(),
                         keep_sets=ks)
            log(f"absent {name} {cell} rc={rc}")


# --------------------------------------------------------------------------- #
def run_name(setname: str, setup: str) -> str:
    return f"min_{setname}_{setup}"


def stage_configs():
    CFG_OUT.mkdir(parents=True, exist_ok=True)
    for setup, base in BASE.items():
        cfg = yaml.safe_load(open(base))
        motifs = ([t["motif"] for t in cfg["data"]["tracks"] if t.get("motif")]
                  + list(cfg.get("motifs", {}).get("extra", []) or []))
        motifs = list(dict.fromkeys(motifs))
        for s in TRAIN_SETS:
            c = json.loads(json.dumps(cfg))
            name = run_name(s, setup)
            c["train"]["output_dir"] = str(OUT / name)
            c["train"]["select_tracks"] = list(SETS[s])
            c["train"]["select_motifs"] = motifs
            c["train"]["val_exclude_overlap"] = True
            c["train"]["track_dropout"] = 0.0
            if setup == "gm2kb":
                # the akita_native cache path is relative to the repository
                c["train"]["cache_dir"] = str(REPO / c["train"]["cache_dir"])
            hdr = (f"# registry section 20, minimal input set ({SET_LETTER[s]}) {s}: "
                   f"{', '.join(SETS[s]) or 'no ChIP'} + motifs {', '.join(motifs)}\n"
                   f"# base config {base}; generated by scripts/minimal_inputs.py --stage configs\n")
            (CFG_OUT / f"{name}.yaml").write_text(hdr + yaml.safe_dump(c, sort_keys=False))
            log(f"config {name}")


def stage_train(run: str, gpus: str):
    cp = CFG_OUT / f"{run}.yaml"
    if not cp.exists():
        raise SystemExit(f"no config {cp}; run --stage configs")
    lf = OUT / f"{run}.train.log"
    g = gpus.split(",")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    if len(g) > 1:
        cmd = [TORCHRUN, "--standalone", f"--nproc_per_node={len(g)}", "-m",
               "hicmultiome.train", "--config", cp]
    else:
        cmd = [PY, "-m", "hicmultiome.train", "--config", cp]
    rc = sh(cmd, lf, env)
    log(f"train {run} rc={rc}")
    if rc:
        raise SystemExit(rc)


def stage_eval(run: str, device: str = "cpu"):
    setup = run.split("_")[-1] if not run.endswith("gm2kb") else "gm2kb"
    if run.endswith("refres_25kb"):
        res, ref, cells = 25000, "all12refres", ["gm12878", "imr90", "k562", "hmec_deep",
                                                  "huvec_deep"]
    elif run.endswith("refres_10kb"):
        res, ref, cells = 10000, "all12refres", ["gm12878", "imr90", "k562", "hmec_deep",
                                                  "huvec_deep"]
    else:
        res, ref, cells = 2048, None, ["gm12878"]
    env = cpu_env() if device == "cpu" else dict(os.environ)
    for cell in cells:
        rc = predict(OUT / run, res, cell, None, OUT / "dedicated" / run, ref, env,
                     device="cpu" if device == "cpu" else "cuda")
        log(f"eval {run} {cell} rc={rc} ({setup})")


# --------------------------------------------------------------------------- #
def _read(d: Path, cell: str) -> pd.DataFrame | None:
    f = d / f"predict_{cell}.csv"
    return pd.read_csv(f) if f.exists() else None


def stage_summary():
    RES_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (run, res, cells, dropout) in EXISTING.items():
        for sname in SETS:
            for cell in cells:
                df = _read(OUT / "absent" / name / sname, cell)
                if df is None or "model" not in df:
                    continue
                r = {"kind": "absent_at_inference", "model": name,
                     "trained_with_track_dropout": dropout, "set": sname,
                     "set_letter": SET_LETTER[sname], "tracks": "+".join(SETS[sname]) or "none",
                     "res": res, "cell": cell, "n_regions": int(df.model.notna().sum()),
                     "median": df.model.median(), "mean": df.model.mean(),
                     "q25": df.model.quantile(0.25), "q75": df.model.quantile(0.75)}
                base = [c for c in df.columns if c.startswith("base_")]
                if base:
                    meds = df[base].median()
                    r["best_other_map"] = meds.idxmax()[5:]
                    r["best_other_map_median"] = meds.max()
                    cons = [c for c in base if c.startswith("base_consensus")]
                    if cons:
                        r["consensus_median"] = df[cons[0]].median()
                rows.append(r)
    for f in sorted((OUT / "dedicated").glob("*/predict_*.csv")) if (OUT / "dedicated").exists() else []:
        run, cell = f.parent.name, f.stem.replace("predict_", "")
        df = pd.read_csv(f)
        sname = next(s for s in TRAIN_SETS if f"min_{s}_" in run)
        res = 25000 if run.endswith("25kb") else 10000 if run.endswith("10kb") else 2048
        r = {"kind": "dedicated_training", "model": run, "trained_with_track_dropout": False,
             "set": sname, "set_letter": SET_LETTER[sname], "tracks": "+".join(SETS[sname]) or "none",
             "res": res, "cell": cell, "n_regions": int(df.model.notna().sum()),
             "median": df.model.median(), "mean": df.model.mean(),
             "q25": df.model.quantile(0.25), "q75": df.model.quantile(0.75)}
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RES_OUT / "minimal_inputs_summary.csv", index=False, float_format="%.4f")
    log(f"summary: {len(S)} rows")
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    for st in a.stage:
        if st == "absent":
            stage_absent(a.models, a.sets)
        elif st == "configs":
            stage_configs()
        elif st == "train":
            stage_train(a.run, a.gpus)
        elif st == "eval":
            stage_eval(a.run, a.device)
        elif st == "summary":
            stage_summary()


if __name__ == "__main__":
    main()
