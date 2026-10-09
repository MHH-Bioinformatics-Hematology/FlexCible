"""Sparse real-world input sets without the consensus, with and without Meth
(registry section 44).

No reference map enters any model of this benchmark: the base model is
all12honest_drop_meth (12 training cells, honest validation, track dropout 0.5,
Meth as a 14th track, hg19 RRBS cells lifted as in section 35 v2), trained at
25 and 10 kb with two seeds (_s1 is the second seed on the same cache).

Scenario sets (tracks given to the model, everything else absent; motifs and
GC/CpG always present):

  meth_only            Meth
  ctcf_meth            CTCF + Meth
  ctcf_h3k27ac         CTCF + H3K27ac                       (+ Meth: ctcf_h3k27ac_meth)
  ctcf_ac_k4me3        CTCF + H3K27ac + H3K4me3             (+ Meth: ctcf_ac_k4me3_meth)
  five_marks           CTCF + H3K27ac + H3K4me3 + H3K27me3 + H3K36me3   (+ Meth: five_marks_meth)
  noctcf_ac_k4me3_meth H3K27ac + H3K4me3 + Meth (no CTCF track), scored with and without
                       motif_meth planes (second base model, below)
  reference rows       chip13 (all 13 ChIP tracks), full (13 + Meth), seq_only

Two ways to score a set, on the same regions and metric:
  infer      the base model with the other tracks set to their absent value
             (crosscell_predict --keep-sets; in distribution because of track
             dropout), as section 35 part A (scripts/meth_replace.py)
  dedicated  a model trained on the set alone (select_tracks = the set, motifs
             kept, track dropout 0.5 over the kept tracks, honest validation, the
             base model's cache and cell set), the machinery of the section 20
             and 35 part C minimal-input runs (scripts/minimal_inputs.py) without
             the consensus input

Motif_meth rows. A second honest base model, all12honest_drop_motifmeth (25 kb seeds
0 and 1, 10 kb seed 0), has the methylation-aware CTCF motif planes (registry 43). On it
the sets noctcf_ac_k4me3_meth, meth_only and ctcf_h3k27ac_meth are scored inference-only
with the planes (set name) and with their methylation removed (suffix _nomm); the track
dropout of the model covers these sets. scenario_motifmeth_vs_off.csv also compares with
the model trained without planes.

Evaluation: the 413 Akita test regions ("akita413") and the disjoint test
windows of the 12-cell cache ("testwin", the gm12878 test shard windows of
mc_all12ref_<k>kb, as scripts/consensus_variance_diag.py), for the held-out K562,
HMEC (deep) and HUVEC (deep) and the training cells with WGBS. HUVEC has no
methylation data: only the sets without Meth are scored there. Median per-region
Pearson of the O/E maps (crosscell_predict --tta); the measured consensus is
scored alongside as a floor but never given to a model.

    python scripts/scenario_bench.py --stage configs
    python scripts/scenario_bench.py --stage train --run scen_meth_only_25kb_s0 --gpus 0,1
    python scripts/scenario_bench.py --stage infer --res 25000 --seed 0 --gpu 0
    python scripts/scenario_bench.py --stage dedicated --run scen_meth_only_25kb_s0 --gpu 1
    python scripts/scenario_bench.py --stage summary
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
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

OUT = mx.HM / "runs/scenario_bench"
CFG_OUT = REPO / "configs/scenario"
RES_OUT = REPO / "results/scenario_bench"
PY = sys.executable
TORCHRUN = str(Path(sys.executable).parent / "torchrun")

CHIP13 = mx.ALL13
SETS = {
    "meth_only": ["Meth"],
    "ctcf_meth": ["CTCF", "Meth"],
    "ctcf_h3k27ac": ["CTCF", "H3K27ac"],
    "ctcf_h3k27ac_meth": ["CTCF", "H3K27ac", "Meth"],
    "ctcf_ac_k4me3": ["CTCF", "H3K27ac", "H3K4me3"],
    "ctcf_ac_k4me3_meth": ["CTCF", "H3K27ac", "H3K4me3", "Meth"],
    "five_marks": ["CTCF", "H3K27ac", "H3K4me3", "H3K27me3", "H3K36me3"],
    "five_marks_meth": ["CTCF", "H3K27ac", "H3K4me3", "H3K27me3", "H3K36me3", "Meth"],
}
REF_ROWS = {"chip13": list(CHIP13), "full": list(CHIP13) + ["Meth"], "seq_only": []}
# rows crossing the motif_meth planes with a missing CTCF track (registry section 44, part 2)
NOCTCF = {"noctcf_ac_k4me3_meth": ["H3K27ac", "H3K4me3", "Meth"]}
INFER_SETS = {**SETS, **NOCTCF, **REF_ROWS}
# scored on the motif_meth base models; "<name>_nomm" gives the same tracks with the
# methylation removed from the motif planes (crosscell_predict pseudo track NOMM)
MM_BASE_SETS = {**NOCTCF, "meth_only": ["Meth"], "ctcf_h3k27ac_meth": ["CTCF", "H3K27ac", "Meth"]}
MM_SETS = {**MM_BASE_SETS, **{f"{k}_nomm": v + ["NOMM"] for k, v in MM_BASE_SETS.items()}}
MM_RUNS = [(25000, 0), (25000, 1), (10000, 0)]
HELD_OUT = ["k562", "hmec_deep", "huvec_deep"]
TRAIN_CELLS = ["gm12878", "imr90", "h1", "hepg2", "a549", "gm23248", "cd14mono"]
CELLS = HELD_OUT + TRAIN_CELLS
REGION_SETS = ["akita413", "testwin"]
RESOLUTIONS = (25000, 10000)
SEEDS = (0, 1)

# dedicated trainings (GPU budget in runs/coarse_gap/QUEUE.md): every set at 25 kb
# with both seeds; at 10 kb the key sets with seed 0
KEY_10KB = ["meth_only", "ctcf_meth", "ctcf_h3k27ac_meth"]


def dedicated_runs() -> list[str]:
    runs = [ded_name(s, 25000, sd) for sd in SEEDS for s in SETS]
    return runs + [ded_name(s, 10000, 0) for s in KEY_10KB]


def ded_name(setname: str, res: int, seed: int) -> str:
    return f"scen_{setname}_{res // 1000}kb_s{seed}"


def parse_name(run: str) -> tuple[str, int, int]:
    body = run[len("scen_"):]
    rest, seed = body.rsplit("_s", 1)
    setname, k = rest.rsplit("_", 1)
    return setname, int(k[:-2]) * 1000, int(seed)


def base_name(res: int, seed: int) -> str:
    return mx.run_name("all12honest_drop_meth" + ("_s1" if seed == 1 else ""), res)


def mm_base_name(res: int, seed: int) -> str:
    return mx.run_name("all12honest_drop_motifmeth" + ("_s1" if seed == 1 else ""), res)


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "scenario_bench.log", "a") as fh:
        fh.write(line + "\n")


def has_meth(cell: str) -> bool:
    return me.has_meth(cell, True)


def cells_for(keep: list[str]) -> list[str]:
    """HUVEC has no methylation: sets that need Meth are not scored there."""
    return [c for c in CELLS if "Meth" not in keep or has_meth(c)]


def regions_file(res: int, kind: str) -> Path | None:
    """TSV of the scored regions (no header): the Akita 413 or the disjoint test
    windows of the 12-cell cache."""
    if kind == "akita413":
        return None                                   # crosscell_predict default
    f = OUT / f"regions_{kind}_{res // 1000}kb.tsv"
    if not f.exists():
        m = json.loads((mx.CACHE / f"mc_all12ref_{res // 1000}kb" / "manifest.json").read_text())
        W = int(m["window_size"])
        rows = []
        for sh in m["shards"]:
            if sh["sample"] == "gm12878" and sh["split"] == "test":
                for st in np.load(sh["path"])["start"].astype(np.int64):
                    rows.append((sh["chrom"], int(st) * res, int(st + W) * res))
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f, sep="\t", header=False, index=False)
    return f


def base_trained(res: int, seed: int) -> bool:
    si = mx.stop_info(base_name(res, seed), 10)
    return bool(si and si[2] == "patience")


def dedicated_done(run: str) -> bool:
    f = OUT / f"{run}.train.log"
    return (OUT / run / "best").exists() and f.exists() and "done." in f.read_text(errors="replace")


# --------------------------------------------------------------------------- configs
def stage_configs():
    CFG_OUT.mkdir(parents=True, exist_ok=True)
    for run in dedicated_runs():
        setname, res, seed = parse_name(run)
        base = base_name(res, seed)
        c = mx.run_config(base, 100, 10, 500)
        motifs = list(dict.fromkeys([t["motif"] for t in c["data"]["tracks"] if t.get("motif")]
                                    + list(c.get("motifs", {}).get("extra", []) or [])))
        c["train"]["output_dir"] = str(OUT / run)
        c["train"]["select_tracks"] = list(SETS[setname])
        c["train"]["select_motifs"] = motifs
        c["train"]["val_exclude_overlap"] = True
        c["train"]["track_dropout"] = 0.5
        c["train"]["track_dropout_mode"] = "uniform_count"
        c["train"]["seed"] = seed
        hdr = (f"# registry section 44, dedicated scenario model {setname} ({'+'.join(SETS[setname])}), "
               f"seed {seed}, no consensus; cache {Path(c['train']['cache_dir']).name}\n"
               f"# generated by scripts/scenario_bench.py --stage configs\n")
        (CFG_OUT / f"{run}.yaml").write_text(hdr + yaml.safe_dump(c, sort_keys=False))
        log(f"config {run}")


def stage_train(run: str, gpus: str):
    cp = CFG_OUT / f"{run}.yaml"
    if not cp.exists():
        raise SystemExit(f"no config {cp}; run --stage configs")
    cache = Path(yaml.safe_load(open(cp))["train"]["cache_dir"])
    if not (cache / "manifest.json").exists():
        raise SystemExit(f"{run}: cache {cache} missing")
    lf = OUT / f"{run}.train.log"
    g = gpus.split(",")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    cmd = ([TORCHRUN, "--standalone", f"--nproc_per_node={len(g)}", "-m", "hicmultiome.train",
            "--config", cp] if len(g) > 1 else [PY, "-m", "hicmultiome.train", "--config", cp])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(lf, "w") as fh:
        rc = subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                             cwd=REPO, env=env)
    log(f"train {run} rc={rc}")
    if rc:
        raise SystemExit(rc)


# --------------------------------------------------------------------------- prediction
def predict(model_dir: Path, cfg_ckpt: Path, res: int, cell: str, region_kind: str,
            keep_sets: dict | None, out: Path, gpu: str) -> int:
    if keep_sets is None:
        if (out / f"predict_{cell}.csv").exists():
            return 0
    elif all((out / k / f"predict_{cell}.csv").exists() for k in keep_sets):
        return 0
    _, bw, assay, ident = mx.CELLS[cell]
    cons = consensus_for(cell, res)
    tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
    cmd = [PY, "scripts/crosscell_predict.py", "--config", cfg_ckpt / "config.yaml",
           "--checkpoint", cfg_ckpt, "--cell", cell, "--bigwig-dir", bw,
           "--target", mx.target(cell, res), "--tta", "--assay", assay, "--binsize", res,
           "--device", "cuda", "--out-dir", out, "--others", f"{tag}={cons}"]
    rf = regions_file(res, region_kind)
    if rf is not None:
        cmd += ["--regions", rf]
    if keep_sets is not None:
        cmd += ["--keep-sets", *[f"{k}={','.join(v)}" for k, v in keep_sets.items()]]
    if has_meth(cell):
        cmd += ["--extra-bigwig-dir", me.meth_dir(cell, True)]
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3",
               CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"predict_{cell}.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def consensus_for(cell: str, res: int) -> Path:
    """Measured consensus as a scored floor only (never a model input)."""
    return mx.consensus_loo(mx.CELLS[cell][3], res) if cell in mx.TRAIN12 else mx.consensus12(res)


def stage_infer(res: int, seed: int, gpu: str):
    name = base_name(res, seed)
    ck = mx.best_ckpt(mx.out_of(name) / name)
    for kind in REGION_SETS:
        for cell in CELLS:
            sets = {k: v for k, v in INFER_SETS.items() if cell in cells_for(v)}
            rc = predict(None, ck, res, cell, kind, sets, OUT / "infer" / name / kind, gpu)
            log(f"infer {name} {kind} {cell} gpu{gpu} rc={rc}")


def stage_infer_mm(res: int, seed: int, gpu: str):
    name = mm_base_name(res, seed)
    ck = mx.best_ckpt(mx.out_of(name) / name)
    for kind in REGION_SETS:
        for cell in CELLS:
            sets = {k: v for k, v in MM_SETS.items() if cell in cells_for(v)}
            rc = predict(None, ck, res, cell, kind, sets, OUT / "infer_mm" / name / kind, gpu)
            log(f"infer_mm {name} {kind} {cell} gpu{gpu} rc={rc}")


def stage_dedicated(run: str, gpu: str):
    setname, res, seed = parse_name(run)
    ck = mx.best_ckpt(OUT / run)
    for kind in REGION_SETS:
        for cell in cells_for(SETS[setname]):
            rc = predict(None, ck, res, cell, kind, None, OUT / "dedicated" / run / kind, gpu)
            log(f"dedicated {run} {kind} {cell} gpu{gpu} rc={rc}")


# --------------------------------------------------------------------------- summary
def _med(f: Path):
    if not f.exists():
        return None
    df = pd.read_csv(f)
    cons = [c for c in df.columns if c.startswith("base_consensus")]
    return (int(df.model.notna().sum()), float(df.model.median()),
            float(df[cons[0]].median()) if cons else np.nan)


def stage_summary():
    RES_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in RESOLUTIONS:
        for seed in SEEDS:
            name = base_name(res, seed)
            for kind in REGION_SETS:
                for st in INFER_SETS:
                    for cell in CELLS:
                        m = _med(OUT / "infer" / name / kind / st / f"predict_{cell}.csv")
                        if m:
                            rows.append(dict(res_kb=res // 1000, seed=seed, mode="infer", set=st,
                                             regions=kind, cell=cell, n=m[0], pearson=m[1],
                                             consensus=m[2]))
    for res, seed in MM_RUNS:
        name = mm_base_name(res, seed)
        for kind in REGION_SETS:
            for st in MM_SETS:
                for cell in CELLS:
                    m = _med(OUT / "infer_mm" / name / kind / st / f"predict_{cell}.csv")
                    if m:
                        rows.append(dict(res_kb=res // 1000, seed=seed, mode="infer_mm", set=st,
                                         regions=kind, cell=cell, n=m[0], pearson=m[1],
                                         consensus=m[2]))
    for run in dedicated_runs():
        setname, res, seed = parse_name(run)
        for kind in REGION_SETS:
            for cell in CELLS:
                m = _med(OUT / "dedicated" / run / kind / f"predict_{cell}.csv")
                if m:
                    rows.append(dict(res_kb=res // 1000, seed=seed, mode="dedicated", set=setname,
                                     regions=kind, cell=cell, n=m[0], pearson=m[1],
                                     consensus=m[2]))
    S = pd.DataFrame(rows)
    if S.empty:
        log("summary: no predictions yet")
        return S
    S["group"] = np.where(S.cell.isin(HELD_OUT), "held_out", "training")
    S["meth"] = S.set.str.contains("meth")
    S.to_csv(RES_OUT / "scenario_cells.csv", index=False, float_format="%.4f")
    per_seed = (S.groupby(["res_kb", "mode", "set", "regions", "group", "seed"])
                .agg(pearson=("pearson", "median"), consensus=("consensus", "median"),
                     n_cells=("cell", "nunique")).reset_index())
    P = (per_seed.groupby(["res_kb", "mode", "set", "regions", "group"])
         .agg(pearson=("pearson", "mean"), consensus=("consensus", "mean"),
              seeds=("seed", "nunique"), n_cells=("n_cells", "max")).reset_index())
    P.to_csv(RES_OUT / "scenario_summary.csv", index=False, float_format="%.4f")
    # what Meth adds to each ChIP set, by mode, and what dedicated training adds to inference-only
    wide = P.pivot_table(index=["res_kb", "regions", "group", "set"], columns="mode",
                         values="pearson").reset_index()
    wide["dedicated_minus_infer"] = wide.get("dedicated") - wide.get("infer")
    wide.to_csv(RES_OUT / "scenario_infer_vs_dedicated.csv", index=False, float_format="%.4f")
    # motif_meth planes on versus off, same model and tracks; and versus the model trained without planes
    mm = P[P["mode"] == "infer_mm"].copy()
    if len(mm):
        mm["base_set"] = mm.set.str.removesuffix("_nomm")
        mm["planes"] = np.where(mm.set.str.endswith("_nomm"), "off", "on")
        w2 = mm.pivot_table(index=["res_kb", "regions", "group", "base_set"], columns="planes",
                            values="pearson").reset_index()
        w2["planes_on_minus_off"] = w2["on"] - w2["off"]
        plain = P[P["mode"] == "infer"].rename(columns={"set": "base_set", "pearson": "plain_model"})
        w2 = w2.merge(plain[["res_kb", "regions", "group", "base_set", "plain_model"]], how="left",
                      on=["res_kb", "regions", "group", "base_set"])
        w2["on_minus_plain_model"] = w2["on"] - w2["plain_model"]
        w2.to_csv(RES_OUT / "scenario_motifmeth_vs_off.csv", index=False, float_format="%.4f")
    log(f"summary: {len(S)} cell rows, {len(P)} pooled rows")
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["configs", "train", "infer", "infer_mm", "dedicated", "summary"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--res", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    for st in a.stage:
        if st == "configs":
            stage_configs()
        elif st == "train":
            stage_train(a.run, a.gpus)
        elif st == "infer":
            stage_infer(a.res, a.seed, a.gpu)
        elif st == "infer_mm":
            stage_infer_mm(a.res, a.seed, a.gpu)
        elif st == "dedicated":
            stage_dedicated(a.run, a.gpu)
        elif st == "summary":
            stage_summary()


if __name__ == "__main__":
    main()
