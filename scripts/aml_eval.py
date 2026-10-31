"""Evaluate the trained models on AML cell lines never seen in training.

Cells: MOLM-13 (in situ Hi-C, KMT2A::MLLT3) and U-937 (Micro-C, wild type and
STAG2 knockout). Inputs: ChIP-Atlas bigWigs rescaled to the training scale
(scripts/aml_fetch.py); marks without a usable experiment are declared absent.

Stages
  targets   measured maps -> one cooler per level -> cooler ICE (--ignore-diags
            2 --max-iters 500, the ENCODE path of scripts/encode_expand.py; the
            GEO weights are replaced so every map is balanced alike) ->
            scripts/oe_proper.py -> aml/oe/<map>_proper_<k>kb.cool (temporary
            name, renamed when complete). Clones are summed (cooler merge) into
            one map per genotype; single clones stay available as replicates.
  predict   scripts/crosscell_predict.py --tta --save-pred on the CPU (threads
            <= 8 in total) for every model x input set x resolution (MODELS,
            INPUTS); reference models read consensus12, which holds none of the
            AML maps (flexcible.reference guard).
  score     per region of the 413 Akita test regions, on one pixel mask shared
            by every method of a (cell, resolution) block: Pearson and HiCRep
            SCC (scripts/benchmark_metrics.py) of each model prediction, each
            training cell's measured map, consensus12 and the replicate maps
            against the target; per-distance Pearson pooled over regions.
            Tables results/aml/scores_long.csv, summary.csv, per_distance.csv.
  perturb   U-937 STAG2 knockout (see perturb()).
  figures   per-distance Pearson, region panels, perturbation aggregates.

    python scripts/aml_eval.py --stage targets
    python scripts/aml_eval.py --stage predict --jobs 3 --threads 2
    python scripts/aml_eval.py --stage score perturb figures
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
AML = HM / "aml"
OE = HM / "crosscell_oe"
AOE = AML / "oe"
WORK = AML / "work"
PRED = AML / "pred"
RES_DIR = REPO / "results/aml"
ENV = Path(sys.executable).parent
PY = sys.executable
import score_like_hicgan as _sc                      # noqa: E402
REGIONS = _sc.REGIONS                                # the 413 Akita test regions
RESOLUTIONS = [2000, 5000, 10000, 25000]
H2C = Path.home() / "src/hic2cool/build/hic2cool"

# ---------------------------------------------------------------- measured maps
# label: (source mcools (summed when several), assay, genotype)
MAPS = {
    "molm13": ([AML / "molm13/GSM9769670_MOLM13-WT-HiC.mcool"], "hic", "wt"),
    # Xu et al. 2022 (GSE147123): its mcool starts at 10 kb in 2x steps, the .hic
    # holds 5, 10 and 25 kb
    "molm13_xu": ([AML / "molm13/GSM4417606_MOLM13_b38d5.hic"], "hic", "wt"),
    "u937_wt": ([AML / "u937/GSM9916468_WT_dmso_Rep1_all.250.mcool",
                 AML / "u937/GSM9916469_WT_dmso_Rep2_all.250.mcool",
                 AML / "u937/GSM9916470_WT_dmso_Rep3_all.250.mcool"], "microc", "wt"),
    "u937_wt_c1": ([AML / "u937/GSM9916468_WT_dmso_Rep1_all.250.mcool"], "microc", "wt"),
    "u937_wt_c2": ([AML / "u937/GSM9916469_WT_dmso_Rep2_all.250.mcool"], "microc", "wt"),
    "u937_wt_c3": ([AML / "u937/GSM9916470_WT_dmso_Rep3_all.250.mcool"], "microc", "wt"),
    "u937_ko": ([AML / "u937/GSM9916471_STAG2_KO_dmso_Rep1_all.250.mcool",
                 AML / "u937/GSM9916472_STAG2_KO_dmso_Rep2_all.250.mcool"], "microc", "ko"),
    "u937_ko_c1": ([AML / "u937/GSM9916471_STAG2_KO_dmso_Rep1_all.250.mcool"], "microc", "ko"),
    "u937_ko_c2": ([AML / "u937/GSM9916472_STAG2_KO_dmso_Rep2_all.250.mcool"], "microc", "ko"),
}
# single clones read their GEO weights (verified against a fresh ICE at 25 kb)
STORED_WEIGHTS = {"u937_wt_c1", "u937_wt_c2", "u937_wt_c3", "u937_ko_c1", "u937_ko_c2"}
STORED_CHECK_RES = 25000
# the main target of each test cell and its replicate maps (ceiling)
CELLS = {
    # both MOLM-13 maps start at 5 kb
    "molm13": {"target": "molm13", "reps": ["molm13_xu"], "inputs": "molm13", "assay": "hic",
               "min_res": 5000},
    # U-937: clone 1 of each genotype is the target (3.26 B WT and the deeper KO
    # clone), the other clones the replicates. The clone sums u937_wt/u937_ko
    # stay selectable; ICE of the 6.2 B-contact WT sum took over 2 h at 25 kb
    # alone, so they were not built.
    "u937": {"target": "u937_wt_c1", "reps": ["u937_wt_c2", "u937_wt_c3"],
             "inputs": "u937", "assay": "microc"},
    "u937_ko": {"target": "u937_ko_c1", "reps": ["u937_ko_c2"],
                "inputs": "u937", "assay": "microc"},
    # processing control: K562 (held out from every model) from ChIP-Atlas inputs
    "k562ca": {"target": None, "reps": [], "inputs": "k562ca", "assay": "hic"},
}
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248",
           "mcf7", "cd14mono", "pc9"]


def oe_path(label: str, res: int) -> Path:
    return AOE / f"{label}_proper_{res // 1000}kb.cool"


def target_of(cell: str, res: int) -> Path | None:
    t = CELLS[cell]["target"]
    if t is None:                                     # k562ca: the K562 target
        p = OE / f"k562_proper_{res // 1000}kb.cool"
        return p if p.exists() else None
    p = oe_path(t, res)
    return p if p.exists() else None


def train_map(cell: str, res: int) -> Path:
    return OE / f"{cell}_proper_{res // 1000}kb.cool"


def consensus12(res: int) -> Path:
    return OE / f"consensus12_proper_{res // 1000}kb.cool"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def sh(cmd: str, logf: Path | None = None) -> None:
    log(f"[cmd] {cmd}")
    if logf is None:
        subprocess.run(cmd, shell=True, check=True, cwd=REPO)
    else:
        with open(logf, "a") as fh:
            subprocess.run(cmd, shell=True, check=True, cwd=REPO, stdout=fh,
                           stderr=subprocess.STDOUT)


def levels(mcool: Path) -> dict[int, str]:
    import cooler
    return {int(u.rstrip("/").rsplit("/", 1)[-1]): u
            for u in cooler.fileops.list_coolers(str(mcool))}


def level_cool(src: Path, res: int, dest: Path) -> bool:
    """One resolution of an mcool as a standalone cooler (copied, or coarsened
    from the finest level that divides res). False if neither exists."""
    tmp = dest.with_name(dest.stem + ".tmp.cool")   # hic2cool appends .cool otherwise
    if tmp.exists():
        tmp.unlink()
    if src.suffix == ".hic":                          # in-house C++ hic2cool, raw counts
        import hicstraw
        if res not in hicstraw.HiCFile(str(src)).getResolutions():
            return False
        sh(f"{H2C} convert -r {res} -p 8 -s '{src}' '{tmp}'")
        tmp.rename(dest)
        return True
    lv = levels(src)
    if res in lv:
        sh(f"{ENV}/cooler cp '{src}::{lv[res]}' '{tmp}'")
    else:
        base = [r for r in sorted(lv, reverse=True) if r < res and res % r == 0]
        if not base:
            return False
        sh(f"{ENV}/cooler coarsen -k {res // base[0]} -p 8 -o '{tmp}' '{src}::{lv[base[0]]}'")
    tmp.rename(dest)
    return True


def build_target(label: str, res: int) -> None:
    """Balanced level cooler and proper O/E for one map at one resolution."""
    o = oe_path(label, res)
    if o.exists():
        return
    srcs, _, _ = MAPS[label]
    if not all(s.exists() for s in srcs):
        log(f"[skip] {label} {res}: source not downloaded")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    AOE.mkdir(parents=True, exist_ok=True)
    k = res // 1000
    bal = WORK / f"{label}_{k}kb.cool"
    lv = levels(srcs[0]) if srcs[0].suffix == ".mcool" else {}
    if label in STORED_WEIGHTS and res in lv and res != STORED_CHECK_RES:
        # single U-937 clone: the GEO weights (cooler zoomify --balance, ICE with
        # the same defaults) are used read-only; STORED_CHECK_RES verifies them
        bal = Path(f"{srcs[0]}::{lv[res]}")
        stats_target(label, res, bal)
    elif not bal.exists():
        tmp = WORK / f"{label}_{k}kb.bal.tmp.cool"
        if tmp.exists():
            tmp.unlink()
        parts = []
        if len(srcs) > 1 and all(s.suffix == ".mcool" and res in levels(s) for s in srcs):
            # clones summed straight from their mcool levels
            sh(f"{ENV}/cooler merge '{tmp}' "
               + " ".join(f"'{s}::{levels(s)[res]}'" for s in srcs))
        else:
            for i, s in enumerate(srcs):
                p = WORK / f"{label}_{k}kb.part{i}.cool"
                if not p.exists() and not level_cool(s, res, p):
                    log(f"[skip] {label} {res}: no level {res} or divisor in {s.name}")
                    return
                parts.append(p)
            if len(parts) == 1:
                sh(f"cp '{parts[0]}' '{tmp}'")
            else:
                sh(f"{ENV}/cooler merge '{tmp}' " + " ".join(f"'{p}'" for p in parts))
        # the stored GEO weights are replaced by ICE with the ENCODE parameters
        sh(f"{ENV}/cooler balance --force -p 8 --ignore-diags 2 --max-iters 500 '{tmp}'",
           AML / "logs" / f"balance_{label}_{k}kb.log")
        tmp.rename(bal)
        for p in parts:
            p.unlink()
        stats_target(label, res, bal)
    tmp = AOE / f"{label}_proper_{k}kb.tmp.cool"
    if tmp.exists():
        tmp.unlink()
    sh(f"{PY} scripts/oe_proper.py --input '{bal}' --output '{tmp}' --binsize {res}",
       AML / "logs" / f"oe_{label}_{k}kb.log")
    tmp.rename(o)
    log(f"[oe ok] {o.name}")


def stats_target(label: str, res: int, bal: Path) -> None:
    """Contacts, masked-bin fraction and, for a single-source map with stored
    weights, the agreement of the new ICE weights with the GEO weights."""
    import cooler
    c = cooler.Cooler(str(bal))
    w = c.bins()[:]["weight"]
    if "::" in str(bal):
        out = RES_DIR / "targets" / f"{label}_{res // 1000}kb.json"
        if out.exists():
            return
    tot = c.info.get("sum")
    if not tot:                                        # hic2cool writes no sum
        tot = sum(float(c.pixels()[i:i + 50_000_000]["count"].sum())
                  for i in range(0, int(c.info["nnz"]), 50_000_000))
    st = {"label": label, "res": res, "contacts": float(tot),
          "nnz": int(c.info.get("nnz") or 0), "ice_masked_frac": float(w.isna().mean()),
          "sources": [str(s) for s in MAPS[label][0]]}
    srcs = MAPS[label][0]
    if len(srcs) == 1 and srcs[0].suffix == ".mcool" and "::" not in str(bal):
        lv = levels(srcs[0])
        if res in lv:
            g = cooler.Cooler(f"{srcs[0]}::{lv[res]}").bins()[:]
            if "weight" in g.columns:
                a, b = g["weight"].values, w.values
                ok = np.isfinite(a) & np.isfinite(b)
                st["geo_weight_pearson"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
                st["geo_weight_masked_frac"] = float(np.isnan(a).mean())
    out = RES_DIR / "targets"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}_{res // 1000}kb.json").write_text(json.dumps(st, indent=1))
    log(f"[stats] {label} {res}: {st['contacts'] / 1e6:.1f} M contacts, "
        f"masked {100 * st['ice_masked_frac']:.1f}%")


def targets(labels: list[str], res_list: list[int]) -> None:
    for label in labels:
        for res in res_list:
            try:
                build_target(label, res)
            except subprocess.CalledProcessError as e:
                log(f"[FAILED] {label} {res}: {e}")


# ---------------------------------------------------------------- models
RUNS = HM / "runs"
# name: (resolution, config, run dir, family, reference kind)
# family: all12 (section 13, overlapping validation), all12honest (honest
# validation retrain, section 15), all12refres (residual on consensus12),
# gm12878 (GM12878 only: section 6 model at 2 kb, section 14 "all" seed 0 at
# 5 to 25 kb), gm12878_refres (section 15 A2 seed 0, reference consensus11).
MODELS = {
    "all12_2kb": (2000, RUNS / "multicell_expanded/all12.yaml",
                  RUNS / "multicell_expanded/all12", "all12", None),
    "gm12878_2kb": (2000, REPO / "configs/full2k_proper.yaml",
                    REPO / "runs/full2k_proper", "gm12878", None),
}
for _k in (5, 10, 25):
    MODELS[f"all12_{_k}kb"] = (_k * 1000, RUNS / f"multicell_expanded/all12_{_k}kb.yaml",
                               RUNS / f"multicell_expanded/all12_{_k}kb", "all12", None)
    MODELS[f"all12honest_{_k}kb"] = (
        _k * 1000, RUNS / f"multicell_reference/all12honest_{_k}kb.yaml",
        RUNS / f"multicell_reference/all12honest_{_k}kb", "all12honest", None)
    MODELS[f"all12refres_{_k}kb"] = (
        _k * 1000, RUNS / f"multicell_reference/all12refres_{_k}kb.yaml",
        RUNS / f"multicell_reference/all12refres_{_k}kb", "all12refres", "consensus12")
for _k, _w in ((5, 384), (10, 192), (25, 80)):
    MODELS[f"gm12878_{_k}kb"] = (_k * 1000, RUNS / f"resolution_inputs/{_k}kb_w{_w}__all__s0.yaml",
                                 RUNS / f"resolution_inputs/{_k}kb_w{_w}__all__s0",
                                 "gm12878", None)
for _k, _w in ((10, 192), (25, 80)):
    MODELS[f"gm12878refres_{_k}kb"] = (
        _k * 1000, RUNS / f"coarse_gap/{_k}kb_w{_w}__refres__s0.yaml",
        RUNS / f"coarse_gap/{_k}kb_w{_w}__refres__s0", "gm12878_refres", "consensus11")
for _k in (2, 5):   # 12-cell honest/reference runs at 5 and 2 kb once they exist
    MODELS[f"all12honest_{_k}kb"] = (
        _k * 1000, RUNS / f"multicell_reference/all12honest_{_k}kb.yaml",
        RUNS / f"multicell_reference/all12honest_{_k}kb", "all12honest", None)
    MODELS[f"all12refres_{_k}kb"] = (
        _k * 1000, RUNS / f"multicell_reference/all12refres_{_k}kb.yaml",
        RUNS / f"multicell_reference/all12refres_{_k}kb", "all12refres", "consensus12")

# 12-cell models trained with track dropout (registry section 22)
for _k in (10, 25):
    for _b, _ref in (("all12refres_drop", "consensus12"), ("all12honest_drop", None),
                     ("all12refres_refdrop", "consensus12"),
                     ("all12refres_drop_hard", "consensus12"), ("all12honest_drop_hard", None),
                     ("all12deg_refres_drop", "consensus12"), ("all12deg_honest_drop", None),
                     ("all12refin_drop", "consensus12"), ("all12refin_drop_ctcfac", "consensus12"),
                     ("all12refres_drop_varw", "consensus12"), ("all12refin", "consensus12"),
                     ("all12hffcoh_refres_drop", "consensus12"),
                     # cohesin-unknown indicator (section 25, 1c)
                     ("all12cohunk_refres_drop", "consensus12"),
                     ("all12cohunk_honest_drop", None),
                     # cohesin-fraction input and cohesin-loss augmentation (section 55)
                     ("all12cohfix_refres_drop", "consensus12"),
                     ("all12cohfixna_refres_drop", "consensus12")):
        MODELS[f"{_b}_{_k}kb"] = (
            _k * 1000, RUNS / f"multicell_reference/{_b}_{_k}kb.yaml",
            RUNS / f"multicell_reference/{_b}_{_k}kb", _b, _ref)
# 5 kb: the track-dropout recipes (AML maps exist from 5 kb; at 2 kb only the
# training-cell and held-out ENCODE evaluation applies)
for _b, _ref in (("all12refres_drop", "consensus12"), ("all12honest_drop", None),
                 ("all12deg_refres_drop", "consensus12"), ("all12refin_drop", "consensus12"),
                 ("all12cohunk_refres_drop", "consensus12"),
                 ("all12refin_drop_struct3", "consensus12")):
    MODELS[f"{_b}_5kb"] = (5000, RUNS / f"multicell_reference/{_b}_5kb.yaml",
                           RUNS / f"multicell_reference/{_b}_5kb", _b, _ref)
for _b in ("all12refin_drop_struct3",):          # chosen structure weights, 25 and 10 kb
    for _k in (10, 25):
        MODELS[f"{_b}_{_k}kb"] = (_k * 1000, RUNS / f"multicell_reference/{_b}_{_k}kb.yaml",
                                  RUNS / f"multicell_reference/{_b}_{_k}kb", _b, "consensus12")
# the one model (registry section 95): per-resolution views of the shared checkpoint (config, norm files and
# model.pt in checkpoint-best), consensus plane as input; trained with the fraction row and the augmentation
for _b in ("all12one_full", "all12one_fullrd"):
    for _k in (10, 25):
        _v = RUNS / f"multicell_reference/{_b}_{_k}kb"
        MODELS[f"{_b}_{_k}kb"] = (_k * 1000, _v / "checkpoint-best/config.yaml", _v, _b, "consensus12")
_DROP = {"all12one_full", "all12one_fullrd", "all12refres_drop", "all12honest_drop", "all12refres_refdrop",
         "all12refres_drop_hard", "all12honest_drop_hard",
         "all12deg_refres_drop", "all12deg_honest_drop", "all12hffcoh_refres_drop",
         "all12cohunk_refres_drop", "all12cohunk_honest_drop", "all12refin_drop",
         "all12refin_drop_struct3", "all12cohfix_refres_drop", "all12cohfixna_refres_drop"}
COHUNK_FAMILIES = {"all12cohunk_refres_drop", "all12cohunk_honest_drop"}
# models with the cohesin-fraction row: the knockout is also asked as a fraction
# (keep-set name suffix _f<value>, read by crosscell_predict): cohesin absent and the
# fraction at 0 or 0.5, and the unchanged wild-type ChIP with the fraction at 0.5 or 0.75
COHFIX_FAMILIES = {"all12cohfix_refres_drop", "all12cohfixna_refres_drop",
                   "all12one_full", "all12one_fullrd"}

# input sets: ChIP folder and, for the perturbation, the keep sets
INPUTS = {
    "molm13": AML / "hg38_molm13",
    "u937": AML / "hg38_u937",
    "k562ca": AML / "hg38_k562ca",
    "k562enc": HM / "hg38_k562",                   # ENCODE fold change (training scale)
    # STAG2-KO input variants (perturb_inputs)
    "u937_ko_ctcfonly": AML / "hg38_u937_ko_ctcfonly",
    "u937_ko_half": AML / "hg38_u937_ko_half",
}
# (input set, cell whose grid/target is used, model families, priority)
PRED_JOBS = [
    ("molm13", "molm13", None, 0),
    ("u937", "u937", None, 0),
    ("u937_ko_ctcfonly", "u937", {"all12honest", "all12refres", "all12"} | _DROP, 1),
    ("u937_ko_half", "u937", {"all12honest", "all12refres", "all12"} | _DROP, 1),
    # processing control: K562 from ChIP-Atlas inputs and, on the same pixels,
    # from the ENCODE fold-change inputs the models were built on
    ("k562ca", "k562ca", {"all12honest", "all12refres", "all12"}, 2),
    ("k562enc", "k562ca", {"all12honest", "all12refres"}, 3),
]
# the cohesin-absent KO variant: the u937 input with RAD21 and SMC3 dropped
KO_ABSENT_KEEP = ["CTCF", "H3K27ac", "H3K4me3", "H3K9ac", "H3K9me3"]


def model_ready(name: str) -> bool:
    res, cfg, run, _, _ = MODELS[name]
    return cfg.exists() and (run / "best").exists()


def ckpt_of(name: str) -> Path:
    run = MODELS[name][2]
    return run / (run / "best").read_text().strip()


def pred_file(model: str, inp: str) -> Path:
    return PRED / model / f"{inp}.npz"


def predict_one(model: str, inp: str, cell: str, threads: int, extra_sets=None) -> None:
    res, cfg, run, fam, ref = MODELS[model]
    if not (Path(run) / "best").exists():            # not trained yet
        return
    # the prediction grid depends only on the bin size and the region, so a
    # cell whose map is not built yet predicts on a training cell's grid (its
    # crosscell csv score is then not used)
    if res < CELLS[cell].get("min_res", 0):
        return
    tgt = target_of(cell, res) or train_map("imr90", res)
    out = PRED / model
    done = pred_file(model, inp)
    if done.exists() or not INPUTS[inp].exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "scripts/crosscell_predict.py", "--config", str(cfg),
           "--checkpoint", str(ckpt_of(model)), "--cell", f"aml_{inp}",
           "--bigwig-dir", str(INPUTS[inp]), "--target", str(tgt), "--tta",
           "--assay", CELLS[cell]["assay"], "--binsize", str(res), "--device", "cpu",
           "--out-dir", str(out / f"{inp}_csv")]
    if ref == "consensus12":
        cmd += ["--reference", str(consensus12(res))]
    if extra_sets:
        cmd += ["--keep-sets", *extra_sets, "--save-pred", str(out / f"{inp}__{{set}}.npz")]
    else:
        cmd += ["--save-pred", str(done) + ".tmp.npz"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES="")
    if os.environ.get("AML_EVAL_GPU") is not None:     # one GPU of a shared machine (default: CPU)
        env["CUDA_VISIBLE_DEVICES"] = os.environ["AML_EVAL_GPU"]
        cmd[cmd.index("--device") + 1] = "cuda"
    logf = out / f"{inp}.log"
    log(f"[predict] {model} {inp}")
    with open(logf, "w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        log(f"[FAILED] {model} {inp} rc={rc}, see {logf}")
        return
    if extra_sets:
        # keep-set outputs: <inp>__<set>.npz; the full set is the plain input
        Path(out / f"{inp}__all.npz").rename(done)
        for s in extra_sets:
            nm = s.split("=", 1)[0]
            if nm != "all":
                p = out / f"{inp}__{nm}.npz"
                p.rename(out / f"{inp}_{nm}.npz")
    else:
        Path(str(done) + ".tmp.npz").rename(done)
    log(f"[predict ok] {model} {inp}")


def predict(models: list[str], jobs: int, threads: int) -> None:
    todo = []
    u937_all = ",".join(json.loads((INPUTS["u937"] / "picks.json").read_text())
                        if (INPUTS["u937"] / "picks.json").exists() else
                        [p.stem for p in INPUTS["u937"].glob("*.bigwig")])
    for m in models:
        if not model_ready(m):
            log(f"[skip] {m}: no checkpoint")
            continue
        for inp, cell, fams, prio in PRED_JOBS:
            if fams is not None and MODELS[m][3] not in fams:
                continue
            if inp == "k562ca" and MODELS[m][3] == "all12" and MODELS[m][0] > 5000:
                continue          # all12 at 10 and 25 kb: all12honest is the control
            sets = None
            if inp == "u937":      # WT and the cohesin-absent KO variant in one pass
                sets = [f"all={u937_all}", "ko_absent=" + ",".join(KO_ABSENT_KEEP)]
                if MODELS[m][3] in COHUNK_FAMILIES:     # cohesin absent, indicator forced to 0
                    sets.append("ko_absent_ind0=" + ",".join(KO_ABSENT_KEEP))
                if MODELS[m][3] in COHFIX_FAMILIES:
                    sets += [f"ko_absent_f0={','.join(KO_ABSENT_KEEP)}",
                             f"ko_absent_f0.5={','.join(KO_ABSENT_KEEP)}",
                             f"ko_f0.5={u937_all}", f"ko_f0.75={u937_all}"]
            todo.append((prio, m, inp, cell, sets))
    # inputs by priority, coarse resolutions first within a priority
    todo.sort(key=lambda t: (t[0], -MODELS[t[1]][0]))
    todo = [t[1:] for t in todo]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(predict_one, m, i, c, threads, s) for m, i, c, s in todo]:
            f.result()


# ---------------------------------------------------------------- STAG2-KO inputs
# No ChIP of the U-937 STAG2 knockout is public, so the knockout is predicted
# from the wild-type ChIP with the cohesin tracks (RAD21, SMC3; both measure the
# whole ring, STAG1- plus STAG2-cohesin) changed. STAG2 loss removes most
# cohesin from non-CTCF sites (enhancers, promoters), while STAG1-cohesin stays
# at CTCF sites (Kojic et al. 2018; Viny et al. 2019; Casa et al. 2020). Variants:
#   ko_ctcfonly  cohesin kept within 1 kb of a U-937 CTCF peak (ChIP-Atlas
#                bed05 of the CTCF pick, q < 1e-5); outside these windows the
#                track is set to its genome-wide mean (enrichment removed)
#   ko_half      enrichment above the track's mean halved everywhere
#                (v -> mean + 0.5 (v - mean)), reduced cohesin at all sites
#   ko_absent    RAD21 and SMC3 declared absent (crosscell_predict keep set),
#                the loss of all cohesin signal as the extreme case
KO_STEP = 200                                           # bp, rewritten bigWig grid
KO_FLANK = 1000


def ctcf_peaks() -> Path:
    srx = next(l.split("\t")[1] for l in open(AML / "picks_u937.tsv") if l.startswith("CTCF\t"))
    return AML / "peaks" / f"{srx}.05.bed"


def ko_bigwig(src: Path, dest: Path, mode: str, peaks: pd.DataFrame | None) -> None:
    import pyBigWig
    bi = pyBigWig.open(str(src))
    chroms = [(c, L) for c, L in bi.chroms().items() if c.startswith("chr") and "_" not in c]
    main = [c for c, _ in chroms if c in {f"chr{i}" for i in range(1, 23)} | {"chrX"}]
    tot = n = 0.0
    for c in main:
        L = bi.chroms()[c]
        tot += bi.stats(c, 0, L, type="sum", exact=True)[0] or 0.0
        n += L
    mean = tot / n
    tmp = Path(str(dest) + ".tmp")
    bo = pyBigWig.open(str(tmp), "w")
    bo.addHeader(chroms, maxZooms=10)
    for c, L in chroms:
        nb = L // KO_STEP
        if nb == 0:
            continue
        from aml_fetch import binned_mean
        v = np.nan_to_num(binned_mean(bi, c, KO_STEP))
        if mode == "ctcfonly":
            keep = np.zeros(nb, bool)
            if peaks is not None:
                pk = peaks[peaks.chrom == c]
                for s, e in zip((pk.start - KO_FLANK) // KO_STEP, (pk.end + KO_FLANK) // KO_STEP):
                    keep[max(0, s):min(nb, e + 1)] = True
            v = np.where(keep | (v == 0), v, mean)     # unmappable zeros stay zero
        elif mode == "half":
            v = np.where(v == 0, 0.0, mean + 0.5 * (v - mean))
        bo.addEntries(c, 0, values=v.astype(np.float32).tolist(), span=KO_STEP, step=KO_STEP)
    bo.close()
    bi.close()
    tmp.rename(dest)


def perturb_inputs() -> None:
    src = INPUTS["u937"]
    peaks = pd.read_csv(ctcf_peaks(), sep="\t", header=None, usecols=[0, 1, 2],
                        names=["chrom", "start", "end"])
    for mode in ("ctcfonly", "half"):
        out = INPUTS[f"u937_ko_{mode}"]
        out.mkdir(parents=True, exist_ok=True)
        for bw in src.glob("*.bigwig"):
            dest = out / bw.name
            if dest.exists() or dest.is_symlink():
                continue
            if bw.stem in ("RAD21", "SMC3"):
                log(f"[ko input] {mode} {bw.stem}")
                ko_bigwig(bw, dest, mode, peaks)
            else:                                  # the share has no symlinks
                shutil.copyfile(bw, str(dest) + ".tmp")
                Path(str(dest) + ".tmp").rename(dest)
        for f in ("missing.json", "picks.csv"):
            if (src / f).exists() and not (out / f).exists():
                shutil.copyfile(src / f, out / f)


# ---------------------------------------------------------------- scoring
# score blocks: target map -> (prediction inputs, extra measured maps). Every
# prediction lies on the target grid of its resolution, which depends only on
# the region and the bin size, so one prediction is scored against several
# targets of the same cell.
BLOCKS = {
    "molm13": (["molm13"], ["molm13_xu"]),
    "molm13_xu": (["molm13"], ["molm13"]),
    "u937_wt_c1": (["u937"], ["u937_ko_c1", "u937_wt_c2", "u937_wt_c3"]),
    "u937_ko_c1": (["u937", "u937_ko_absent", "u937_ko_ctcfonly", "u937_ko_half",
                   "u937_ko_absent_ind0"],
                   ["u937_wt_c1", "u937_ko_c2"]),
    "k562": (["k562ca", "k562enc"], []),
}
# replicate pairs scored against each other (ceiling), per target block
REP_PAIRS = {
    "molm13": [("molm13", "molm13_xu")],
    "u937_wt_c1": [("u937_wt_c1", "u937_wt_c2"), ("u937_wt_c1", "u937_wt_c3"),
                   ("u937_wt_c2", "u937_wt_c3")],
    "u937_ko_c1": [("u937_ko_c1", "u937_ko_c2")],
}
BASE_CELLS = TRAIN12 + ["k562"]


def map_path(label: str, res: int) -> Path | None:
    if label == "consensus12":
        p = consensus12(res)
    elif label == "k562":
        p = OE / f"k562_proper_{res // 1000}kb.cool"
    elif label in MAPS:
        p = oe_path(label, res)
    else:
        p = train_map(label, res)
    return p if p.exists() else None


def load_preds(inp: str, res: int) -> dict[str, dict]:
    """model name -> {region: prediction} for one input set at one resolution."""
    out = {}
    for m, spec in MODELS.items():
        if spec[0] != res:
            continue
        f = pred_file(m, inp)
        if f.exists():
            z = np.load(f)
            out[m] = {k: z[k] for k in z.files}
    return out


def fetch_region(clr, region: str, cache: dict) -> np.ndarray | None:
    from flexcible.genome import match_name
    key = (clr.filename, region)
    if key in cache:
        return cache[key]
    chrom, se = region.split(":")
    s, e = map(int, se.split("-"))
    name = match_name(chrom, list(clr.chromnames))
    try:
        M = np.asarray(clr.matrix(balance=False).fetch(f"{name}:{s}-{e}"), float)
    except Exception:                                  # noqa: BLE001
        M = None
    if M is not None:
        M = np.where(M == 0, np.nan, M)                # 0 = unmappable / not measured
    cache[key] = M
    return M


def pred_matrix(P: np.ndarray) -> np.ndarray:
    """A saved prediction with crosscell_predict's 'live' rule (rows the model
    never covered are missing)."""
    live = np.abs(np.nan_to_num(P)).sum(1) > 0
    return np.where(live[:, None] & live[None, :], P, np.nan)


def score_res(res: int) -> Path | None:
    """Every block (BLOCKS) and replicate pair (REP_PAIRS) at one resolution.
    Per region, every map is read once and shared by the blocks; per block,
    Pearson and SCC of each method on one shared pixel mask (upper triangle,
    offset >= 2, measured in the target and defined for every method covering
    the region) and pixel pairs per diagonal for the per-distance Pearson.
    Written to WORK/score_parts/<k>kb.pkl."""
    import pickle
    import cooler
    import benchmark_metrics as bm
    k = res // 1000
    blocks = {}
    for target, (inputs, extra) in BLOCKS.items():
        if map_path(target, res) is None:
            continue
        meth = {}
        for inp in inputs:
            for m in load_preds(inp, res):
                meth[f"model:{m}:{inp}"] = ("pred", (m, inp))
        if not meth:
            continue
        for c in BASE_CELLS + ["consensus12"] + list(extra):
            if c != target and (map_path(c, res) is not None or c == "consensus12"):
                meth[f"map:{c}"] = ("map", c)
        blocks[target] = meth
    pairs = [(t, a, b) for t in REP_PAIRS for a, b in REP_PAIRS[t]
             if map_path(a, res) is not None and map_path(b, res) is not None]
    if not blocks and not pairs:
        return None
    labels = set(blocks) | {x for t, a, b in pairs for x in (a, b)} | {
        v[1] for meth in blocks.values() for v in meth.values() if v[0] == "map"}
    clrs = {}
    for lab in labels:
        path = consensus12(res) if lab == "consensus12" else map_path(lab, res)
        if path is not None and Path(path).exists():
            clrs[lab] = cooler.Cooler(str(path))
    preds = {}
    for meth in blocks.values():
        for kind, v in meth.values():
            if kind == "pred" and v not in preds:
                preds[v] = dict(np.load(pred_file(*v)))
    regions = pd.read_csv(REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    rows, reps = [], []
    dist = {(t, m): {} for t, meth in blocks.items() for m in meth}
    for r in regions.itertuples():
        reg = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
        cache: dict = {}
        get = {lab: fetch_region(c, reg, cache) for lab, c in clrs.items()}
        for target, meth in blocks.items():
            T = get.get(target)
            if T is None:
                continue
            mats = {}
            for m, (kind, v) in meth.items():
                if kind == "pred":
                    P = preds[v].get(reg)
                    M = pred_matrix(P) if P is not None and P.shape == T.shape else None
                else:
                    M = get.get(v)
                    M = M if M is not None and M.shape == T.shape else None
                mats[m] = M
            mask = bm.triu_mask(T.shape[0], 2) & np.isfinite(T)
            for M in mats.values():
                if M is not None:
                    mask &= np.isfinite(M)
            if mask.sum() < 100:
                continue
            for m, M in mats.items():
                if M is None:
                    continue
                rows.append({"target": target, "res_kb": k, "method": m, "region": reg,
                             "pearson": bm.pearson(M, T, mask), "scc": bm.scc(M, T, mask),
                             "n_pixels": int(mask.sum())})
                for d, (a, b) in bm.per_distance(M, T, mask).items():
                    # running sums n, x, y, xx, yy, xy: the pooled Pearson per
                    # diagonal without holding every pixel pair
                    a, b = a.astype(np.float64), b.astype(np.float64)
                    acc = dist[(target, m)].setdefault(d, np.zeros(6))
                    acc += (a.size, a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum())
        for t, a, b in pairs:
            A, B = get.get(a), get.get(b)
            if A is None or B is None or A.shape != B.shape:
                continue
            mask = bm.triu_mask(A.shape[0], 2) & np.isfinite(A) & np.isfinite(B)
            if mask.sum() >= 100:
                reps.append({"target": t, "res_kb": k, "pair": f"{a}~{b}", "region": reg,
                             "pearson": bm.pearson(A, B, mask), "scc": bm.scc(A, B, mask)})
    drows = []
    for (t, m), dd in dist.items():
        for d, (n, sx, sy, sxx, syy, sxy) in sorted(dd.items()):
            vx, vy = sxx - sx * sx / max(n, 1), syy - sy * sy / max(n, 1)
            if n >= 50 and vx > 0 and vy > 0:
                r = (sxy - sx * sy / n) / np.sqrt(vx * vy)
                drows.append({"target": t, "res_kb": k, "method": m, "distance_kb": d * res / 1000,
                              "pearson": float(r), "n_pixels": int(n)})
    out = WORK / "score_parts" / f"{k}kb.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".tmp", "wb") as fh:
        pickle.dump({"rows": rows, "dist": drows, "reps": reps}, fh)
    Path(str(out) + ".tmp").rename(out)
    log(f"[score] {k} kb: {len(rows)} rows")
    return out


def score(res_list: list[int]) -> None:
    """Score the given resolutions (one process each), then merge every
    resolution scored so far into the tables."""
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    RES_DIR.mkdir(parents=True, exist_ok=True)
    if res_list:
        with ProcessPoolExecutor(max_workers=len(res_list)) as ex:
            list(ex.map(score_res, res_list))
    long_rows, dist_rows, rep_rows = [], [], []
    for f in sorted((WORK / "score_parts").glob("*kb.pkl")):
        z = pickle.load(open(f, "rb"))
        long_rows += z["rows"]
        dist_rows += z["dist"]
        rep_rows += z["reps"]
    L = pd.DataFrame(long_rows)
    L.to_csv(WORK / "scores_long.csv.gz", index=False)      # per region, large
    pd.DataFrame(dist_rows).to_csv(RES_DIR / "per_distance.csv", index=False)
    R = pd.DataFrame(rep_rows)
    R.to_csv(RES_DIR / "replicates_per_region.csv", index=False)
    summ = (L.groupby(["target", "res_kb", "method"])
             .agg(pearson_median=("pearson", "median"), scc_median=("scc", "median"),
                  pearson_mean=("pearson", "mean"), n_regions=("pearson", "count"))
             .reset_index())
    summ.to_csv(RES_DIR / "summary.csv", index=False)
    if len(R):
        (R.groupby(["target", "res_kb", "pair"])
          .agg(pearson_median=("pearson", "median"), scc_median=("scc", "median"),
               n_regions=("pearson", "count")).reset_index()
          .to_csv(RES_DIR / "replicates.csv", index=False))
    grid(summ)


def grid(summ: pd.DataFrame) -> None:
    """Compact table per target and resolution: every model, the best
    training-cell map, consensus12 and the replicate columns."""
    rows = []
    for (t, k), g in summ.groupby(["target", "res_kb"]):
        g = g.set_index("method")
        maps = g[g.index.str.startswith("map:") & g.index.str[4:].isin(TRAIN12)]
        rec = {"target": t, "res_kb": k, "n_regions": int(g["n_regions"].max())}
        if len(maps):
            best = maps["pearson_median"].idxmax()
            rec.update({"best_train_map": best[4:],
                        "best_train_map_pearson": maps.loc[best, "pearson_median"],
                        "best_train_map_scc": maps.loc[best, "scc_median"],
                        "median_train_map_pearson": maps["pearson_median"].median()})
        for m in g.index:
            if m.startswith("model:") or not m[4:] in TRAIN12:
                rec[f"{m}|pearson"] = g.loc[m, "pearson_median"]
                rec[f"{m}|scc"] = g.loc[m, "scc_median"]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(RES_DIR / "grid.csv", index=False)


# ---------------------------------------------------------------- perturbation
KO_VARIANTS = ["u937_ko_ctcfonly", "u937_ko_half", "u937_ko_absent", "u937_ko_absent_ind0",
               "u937_ko_absent_f0", "u937_ko_absent_f0.5", "u937_ko_f0.5", "u937_ko_f0.75"]
# measured change between independent clones: (KO clone - WT clone) pairs that
# share no map, scored against each other (the ceiling of a change prediction)
DELTA_REP = [(("u937_ko_c1", "u937_wt_c1"), ("u937_ko_c2", "u937_wt_c2")),
             (("u937_ko_c1", "u937_wt_c1"), ("u937_ko_c2", "u937_wt_c3")),
             (("u937_ko_c1", "u937_wt_c2"), ("u937_ko_c2", "u937_wt_c3"))]


def loop_params(res: int):
    """Loop calling on a ~10 kb grid (bioscore.call_loops), 60 to 600 kb."""
    from flexcible.bioscore import BioParams
    f = max(1, 10000 // res)
    cres = res * f
    return BioParams(loop_coarsen=f, donut_inner=1, donut_outer=4,
                     loop_min_dist=max(3, 60000 // cres), loop_max_dist=600000 // cres,
                     loop_max_peaks=25, apa_half=5)


def insulation_log(L: np.ndarray, w: int) -> np.ndarray:
    """Diamond mean of log O/E (w bins each side) per bin; NaN at edges."""
    n = L.shape[0]
    prof = np.full(n, np.nan)
    for b in range(w, n - w):
        blk = L[b - w:b, b + 1:b + w + 1]
        if np.isfinite(blk).mean() > 0.5:
            prof[b] = np.nanmean(blk)
    return prof


def perturb(res_list: list[int], only: list[str] | None = None, tag: str | None = None) -> None:
    """Predicted versus measured WT -> STAG2-KO change in U-937."""
    import cooler
    import benchmark_metrics as bm
    from scipy.stats import pearsonr, spearmanr
    from flexcible.bioscore import call_loops, coarsen, local_enrichment
    from flexcible.loops import boundaries
    regions = pd.read_csv(REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    region_rows, loop_rows, bnd_rows, agg = [], [], [], {}
    for res in res_list:
        pw, pk = map_path("u937_wt_c1", res), map_path("u937_ko_c1", res)
        if pw is None or pk is None:
            continue
        cw, ck = cooler.Cooler(str(pw)), cooler.Cooler(str(pk))
        reps = {lab: cooler.Cooler(str(map_path(lab, res))) for pair in DELTA_REP
                for d in pair for lab in d if map_path(lab, res) is not None}
        models = [m for m, s in MODELS.items() if s[0] == res
                  and pred_file(m, "u937").exists() and (not only or m in only)]
        preds = {m: {v: dict(np.load(pred_file(m, v))) for v in ["u937"] + KO_VARIANTS
                     if pred_file(m, v).exists()} for m in models}
        prm = loop_params(res)
        w = max(3, 100000 // res)
        cache: dict = {}
        for r in regions.itertuples():
            reg = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
            W, K = fetch_region(cw, reg, cache), fetch_region(ck, reg, cache)
            if W is None or K is None or W.shape != K.shape:
                continue
            n = W.shape[0]
            dT = K - W
            base = bm.triu_mask(n, 2) & np.isfinite(dT)
            # measured change between independent clone pairs
            for i, ((k1, w1), (k2, w2)) in enumerate(DELTA_REP):
                mats = [fetch_region(reps[x], reg, cache) if x in reps else None
                        for x in (k1, w1, k2, w2)]
                if any(m is None or m.shape != W.shape for m in mats):
                    continue
                a, b = mats[0] - mats[1], mats[2] - mats[3]
                mk = base & np.isfinite(a) & np.isfinite(b)
                if mk.sum() >= 100:
                    region_rows.append({"res_kb": res // 1000, "method": f"clones_{i}",
                                        "region": reg, "delta_pearson": bm.pearson(a, b, mk)})
            # loops called on the measured WT and KO maps (union), ~10 kb grid
            f = prm.loop_coarsen
            Wc, Kc = coarsen(np.exp(W), f), coarsen(np.exp(K), f)
            aw, _ = call_loops(Wc, prm)
            ak, _ = call_loops(Kc, prm)
            anchors = {tuple(x) for x in aw} | {tuple(x) for x in ak}
            anchors = np.array(sorted(anchors), np.int64).reshape(-1, 2)
            ew, ek = local_enrichment(Wc, prm), local_enrichment(Kc, prm)
            # boundaries of the measured WT insulation profile
            iw, ik = insulation_log(W, w), insulation_log(K, w)
            bnd = boundaries(iw, 0.1)
            for m in models:
                pv = preds[m]
                if "u937" not in pv or reg not in pv["u937"]:
                    continue
                PW = pred_matrix(pv["u937"][reg])
                if PW.shape != W.shape:
                    continue
                pwc = coarsen(np.exp(PW), f)
                epw = local_enrichment(pwc, prm)
                ipw = insulation_log(PW, w)
                for v in KO_VARIANTS:
                    if v not in pv or reg not in pv[v]:
                        continue
                    PK = pred_matrix(pv[v][reg])
                    dP = PK - PW
                    mk = base & np.isfinite(dP)
                    if mk.sum() >= 100:
                        region_rows.append({"res_kb": res // 1000, "method": f"{m}:{v}",
                                            "region": reg, "delta_pearson": bm.pearson(dP, dT, mk),
                                            "delta_scc": bm.scc(dP, dT, mk)})
                    epk = local_enrichment(coarsen(np.exp(PK), f), prm)
                    ipk = insulation_log(PK, w)
                    for i, j in anchors:
                        loop_rows.append({
                            "res_kb": res // 1000, "method": f"{m}:{v}", "region": reg,
                            "i": int(i), "j": int(j), "dist_kb": (j - i) * res * f / 1000,
                            "meas_wt": np.log2(ew[i, j]), "meas_ko": np.log2(ek[i, j]),
                            "pred_wt": np.log2(epw[i, j]), "pred_ko": np.log2(epk[i, j])})
                    for b in bnd:
                        bnd_rows.append({"res_kb": res // 1000, "method": f"{m}:{v}",
                                         "region": reg, "bin": int(b), "meas_wt": iw[b],
                                         "meas_ko": ik[b], "pred_wt": ipw[b], "pred_ko": ipk[b]})
                    # aggregates: APA around the loops and insulation around boundaries
                    key = (res // 1000, m, v)
                    A = agg.setdefault(key, {"apa": [], "ins": []})
                    h = prm.apa_half
                    Wl, Kl = np.log2(Wc), np.log2(Kc)
                    Pl, Ql = np.log2(pwc), np.log2(coarsen(np.exp(PK), f))
                    for i, j in anchors:
                        if i - h >= 0 and j + h < Wc.shape[0]:
                            sl = (slice(i - h, i + h + 1), slice(j - h, j + h + 1))
                            A["apa"].append(np.stack([Wl[sl], Kl[sl], Pl[sl], Ql[sl]]))
                    for b in bnd:
                        if b - 2 * w >= 0 and b + 2 * w < n:
                            s = slice(b - 2 * w, b + 2 * w + 1)
                            A["ins"].append(np.stack([iw[s], ik[s], ipw[s], ipk[s]]))
    RES_DIR.mkdir(parents=True, exist_ok=True)
    # one part per resolution, merged with the parts of earlier runs
    sfx = f"_{tag}" if tag else ""             # a tag keeps the tables of earlier models untouched
    parts = WORK / f"perturb_parts{sfx}"
    parts.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(r // 1000) for r in res_list)
    for name, rows_ in (("regions", region_rows), ("loops", loop_rows), ("boundaries", bnd_rows)):
        pd.DataFrame(rows_).to_csv(parts / f"{name}_{tag}kb.csv.gz", index=False)
    np.savez_compressed(parts / f"aggregates_{tag}kb.npz",
                        **{f"{k}kb|{m}|{v}|{t}": np.nanmean(np.stack(A[t]), 0)
                           for (k, m, v), A in agg.items() for t in ("apa", "ins") if A[t]},
                        **{f"{k}kb|{m}|{v}|n_{t}": np.array(len(A[t]))
                           for (k, m, v), A in agg.items() for t in ("apa", "ins")})

    def merged(name):
        fs = sorted(parts.glob(f"{name}_*kb.csv.gz"))
        dfs = [pd.read_csv(f) for f in fs if f.stat().st_size > 30]
        D = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if len(D):     # a resolution rerun later replaces the earlier part's rows
            D = D.drop_duplicates()
        return D
    R, Lp, Bp = merged("regions"), merged("loops"), merged("boundaries")
    R.to_csv(WORK / f"perturb_regions{sfx}.csv.gz", index=False)
    Lp.to_csv(WORK / f"perturb_loops{sfx}.csv.gz", index=False)
    Bp.to_csv(WORK / f"perturb_boundaries{sfx}.csv.gz", index=False)
    agg_all = {}
    for f in sorted(parts.glob("aggregates_*kb.npz")):
        agg_all.update(dict(np.load(f)))
    np.savez_compressed(WORK / f"perturb_aggregates{sfx}.npz", **agg_all)
    out = []
    for (k, m), g in R.groupby(["res_kb", "method"]):
        rec = {"res_kb": k, "method": m, "n_regions": len(g),
               "delta_pearson_median": g["delta_pearson"].median()}
        if "delta_scc" in g:
            rec["delta_scc_median"] = g["delta_scc"].median()
        for name, D in (("loops", Lp), ("boundaries", Bp)):
            if not len(D):
                continue
            d = D[(D.res_kb == k) & (D.method == m)].replace([np.inf, -np.inf], np.nan).dropna(
                subset=["meas_wt", "meas_ko", "pred_wt", "pred_ko"])
            if len(d) < 10:
                continue
            dm, dp = d.meas_ko - d.meas_wt, d.pred_ko - d.pred_wt
            rec.update({f"{name}_n": len(d),
                        f"{name}_meas_mean_change": dm.mean(),
                        f"{name}_pred_mean_change": dp.mean(),
                        f"{name}_change_pearson": pearsonr(dm, dp).statistic,
                        f"{name}_change_spearman": spearmanr(dm, dp).statistic,
                        f"{name}_sign_agreement": float((np.sign(dm) == np.sign(dp)).mean())})
            if name == "loops":
                for lab, sel in (("short", d.dist_kb < 200), ("long", d.dist_kb >= 200)):
                    rec[f"loops_{lab}_n"] = int(sel.sum())
                    rec[f"loops_{lab}_meas_mean_change"] = dm[sel].mean()
                    rec[f"loops_{lab}_pred_mean_change"] = dp[sel].mean()
        out.append(rec)
    pd.DataFrame(out).to_csv(RES_DIR / f"perturb_summary{sfx}.csv", index=False)
    log(f"[perturb] wrote perturb_summary{sfx}.csv")


# ---------------------------------------------------------------- figures
FAMILY_COLOR = {"all12refres": "#2a78d6", "all12honest": "#eb6834", "all12": "#1baf7a",
                "gm12878": "#eda100", "gm12878_refres": "#e87ba4",
                # track-dropout models (section 22), palette slots 6 to 8
                "all12refres_drop": "#008300", "all12honest_drop": "#e34948",
                "all12refres_refdrop": "#4a3aa7",
                # hard-split runs (section 24): the dropout models' hues, reused
                "all12refres_drop_hard": "#008300", "all12honest_drop_hard": "#e34948",
                # the eight palette hues are taken: the degron-trained models
                # share their non-degron counterpart's hue (compare them in
                # separate panels or tables, never on one axis)
                "all12deg_refres_drop": "#008300", "all12deg_honest_drop": "#e34948",
                "all12refin_drop": "#4a3aa7", "all12refin_drop_ctcfac": "#4a3aa7",
                "all12refres_drop_varw": "#008300", "all12refin": "#4a3aa7",
                "all12hffcoh_refres_drop": "#008300",
                "all12cohunk_refres_drop": "#008300", "all12cohunk_honest_drop": "#e34948",
                "all12refin_drop_struct3": "#4a3aa7"}
MAP_STYLE = {"consensus12": ("#52514e", "--"), "best_train_map": ("#0b0b0b", ":"),
             "replicate": ("#4a3aa7", "-.")}


def _fam(method: str) -> str | None:
    if not method.startswith("model:"):
        return None
    return MODELS[method.split(":")[1]][3]


def fig_per_distance() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = pd.read_csv(RES_DIR / "per_distance.csv")
    S = pd.read_csv(RES_DIR / "summary.csv")
    targets = [t for t in BLOCKS if t in set(D.target)]
    ress = sorted(D.res_kb.unique())
    fig, axes = plt.subplots(len(targets), len(ress), figsize=(3.4 * len(ress), 2.7 * len(targets)),
                             squeeze=False, sharey=True)
    for r, t in enumerate(targets):
        for c, k in enumerate(ress):
            ax = axes[r, c]
            d = D[(D.target == t) & (D.res_kb == k)]
            s = S[(S.target == t) & (S.res_kb == k)].set_index("method")
            if not len(d):
                ax.set_axis_off()
                continue
            for m, g in d.groupby("method"):
                fam = _fam(m)
                inp = m.split(":")[2] if fam else None
                if fam and inp in ("molm13", "u937", "k562ca"):
                    ax.plot(g.distance_kb, g.pearson, color=FAMILY_COLOR[fam], lw=1.6,
                            label=m.split(":")[1].rsplit("_", 1)[0])
            maps = s[s.index.str[4:].isin(TRAIN12)]
            if len(maps):
                best = maps.pearson_median.idxmax()
                g = d[d.method == best]
                col, ls = MAP_STYLE["best_train_map"]
                ax.plot(g.distance_kb, g.pearson, color=col, ls=ls, lw=1.4,
                        label=f"best other cell ({best[4:]})")
            for lab, key in (("map:consensus12", "consensus12"),
                             ("map:molm13_xu", "replicate"), ("map:molm13", "replicate")):
                g = d[d.method == lab]
                if len(g) and not (t == "molm13_xu" and lab == "map:molm13_xu"):
                    col, ls = MAP_STYLE[key]
                    ax.plot(g.distance_kb, g.pearson, color=col, ls=ls, lw=1.4, label=lab[4:])
            ax.set_title(f"{t}, {k} kb", fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            if r == len(targets) - 1:
                ax.set_xlabel("distance (kb)")
            if c == 0:
                ax.set_ylabel("Pearson (pooled)")
            ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(RES_DIR / "per_distance.png", dpi=130)
    plt.close(fig)


def fig_regions() -> None:
    """Target, best model, consensus12 and the best other cell's map for the
    regions at the 10th, 50th and 90th percentile of the best model's score."""
    import cooler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = pd.read_csv(WORK / "scores_long.csv.gz")
    S = pd.read_csv(RES_DIR / "summary.csv")
    for (t, k), s in S.groupby(["target", "res_kb"]):
        mods = s[s.method.str.startswith("model:")]
        if not len(mods) or t == "k562":
            continue
        best = mods.loc[mods.pearson_median.idxmax(), "method"]
        maps = s[s.method.str[4:].isin(TRAIN12)]
        bmap = maps.loc[maps.pearson_median.idxmax(), "method"]
        l = L[(L.target == t) & (L.res_kb == k) & (L.method == best)].sort_values("pearson")
        picks = [l.iloc[int(q * (len(l) - 1))] for q in (0.1, 0.5, 0.9)]
        res = k * 1000
        _, mname, inp = best.split(":")
        preds = dict(np.load(pred_file(mname, inp)))
        tclr = cooler.Cooler(str(map_path(t, res)))
        cclr = cooler.Cooler(str(consensus12(res)))
        mclr = cooler.Cooler(str(map_path(bmap[4:], res)))
        fig, axes = plt.subplots(3, 4, figsize=(11, 8.4))
        cache: dict = {}
        for r, row in enumerate(picks):
            reg = row.region
            mats = [fetch_region(tclr, reg, cache), pred_matrix(preds[reg]),
                    fetch_region(cclr, reg, cache), fetch_region(mclr, reg, cache)]
            sc = L[(L.target == t) & (L.res_kb == k) & (L.region == reg)].set_index("method")
            titles = [f"measured {t}", f"{mname} ({sc.loc[best, 'pearson']:.2f})",
                      f"consensus12 ({sc.loc['map:consensus12', 'pearson']:.2f})"
                      if "map:consensus12" in sc.index else "consensus12",
                      f"{bmap[4:]} map ({sc.loc[bmap, 'pearson']:.2f})"]
            for c, (M, ti) in enumerate(zip(mats, titles)):
                ax = axes[r, c]
                ax.imshow(M, cmap="RdBu_r", vmin=-1.5, vmax=1.5, interpolation="none")
                ax.set_title(ti, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(reg, fontsize=7)
        fig.suptitle(f"{t}, {k} kb: log O/E, regions at the 10th, 50th, 90th percentile "
                     f"of {mname} (Pearson in brackets)", fontsize=9)
        fig.tight_layout()
        fig.savefig(RES_DIR / f"regions_{t}_{k}kb.png", dpi=110)
        plt.close(fig)


def fig_perturb() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = WORK / "perturb_aggregates.npz"
    if not f.exists():
        return
    Z = np.load(f)
    P = pd.read_csv(RES_DIR / "perturb_summary.csv")
    # 10 kb (the loop grid of the caller without coarsening) and 2 kb
    keys = sorted({k.rsplit("|", 1)[0] for k in Z.files if k.endswith("|apa")
                   and k.split("|")[0] in ("10kb", "2kb")})
    for key in keys:
        kb, m, v = key.split("|")
        apa, ins = Z[f"{key}|apa"], Z.get(f"{key}|ins")
        fig, axes = plt.subplots(1, 7, figsize=(17, 2.8),
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1.6]})
        tt = ["measured WT", "measured KO", "predicted WT", f"predicted KO", "measured KO - WT",
              "predicted KO - WT"]
        mats = [apa[0], apa[1], apa[2], apa[3], apa[1] - apa[0], apa[3] - apa[2]]
        for ax, M, ti in zip(axes[:6], mats, tt):
            lim = 1.0 if "-" not in ti else max(0.1, float(np.nanmax(np.abs(mats[4]))))
            ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(ti, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        n = int(Z[f"{key}|n_apa"])
        axes[0].set_ylabel(f"APA, {n} loops\nlog2 O/E", fontsize=8)
        ax = axes[6]
        if ins is not None and ins.ndim == 2:
            x = np.arange(ins.shape[1]) - ins.shape[1] // 2
            for i, (lab, col, ls) in enumerate((("measured WT", "#0b0b0b", "-"),
                                                ("measured KO", "#0b0b0b", "--"),
                                                ("predicted WT", "#2a78d6", "-"),
                                                ("predicted KO", "#2a78d6", "--"))):
                ax.plot(x, ins[i], color=col, ls=ls, lw=1.5, label=lab)
            ax.set_title(f"insulation at {int(Z[f'{key}|n_ins'])} WT boundaries", fontsize=8)
            ax.set_xlabel("bins from boundary")
            ax.legend(fontsize=6, frameon=False)
        row = P[(P.res_kb == int(kb[:-2])) & (P.method == f"{m}:{v}")]
        sub = ""
        if len(row):
            r0 = row.iloc[0]
            sub = (f"; change map Pearson (median over regions) {r0.delta_pearson_median:.3f}"
                   f", loop change Pearson {r0.get('loops_change_pearson', np.nan):.3f}"
                   f", boundary change Pearson {r0.get('boundaries_change_pearson', np.nan):.3f}")
        fig.suptitle(f"U-937 STAG2 KO, {kb}, {m}, input {v}{sub}", fontsize=9)
        fig.tight_layout()
        fig.savefig(RES_DIR / f"perturb_{kb}_{m}_{v}.png", dpi=110)
        plt.close(fig)


def figures() -> None:
    fig_per_distance()
    fig_regions()
    fig_perturb()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["targets", "perturb_inputs", "predict", "score", "perturb", "figures"])
    ap.add_argument("--maps", nargs="*", default=list(MAPS))
    ap.add_argument("--res", nargs="*", type=int, default=RESOLUTIONS,
                    help="resolutions to (re)compute; score with none merges the parts")
    ap.add_argument("--cells", nargs="*", default=list(CELLS))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tag", default=None,
                    help="stage perturb: score only --models and write perturb_summary_<tag>.csv "
                         "(the tables of earlier models stay untouched)")
    a = ap.parse_args()
    if "targets" in a.stage:
        targets(a.maps, a.res)
    if "predict" in a.stage:
        predict(a.models, a.jobs, a.threads)
    if "perturb_inputs" in a.stage:
        perturb_inputs()
    if "score" in a.stage:
        score(a.res)
    if "perturb" in a.stage:
        perturb(a.res, a.models if a.tag else None, a.tag)
    if "figures" in a.stage:
        figures()


if __name__ == "__main__":
    main()
