"""Cell-type-specific structure: are differences between cells predicted?

Held-out regions only: the 413 Akita test regions, which no training window of
any cell overlaps (results/multicell_reference/split_disjoint.csv, checked in
stage `split`). Cells: the 12 training cells (each read with its leave-one-out
consensus), the held-out K562, HMEC deep and HUVEC deep (consensus12) and the
AML lines MOLM-13, U-937 wild type and U-937 STAG2 knockout (consensus12; the
knockout is predicted from the wild-type ChIP with cohesin declared absent,
registry section 19). Models: all12refres and all12honest at 10 and 25 kb,
and the track-dropout variants all12refres_drop / all12honest_drop once their
checkpoints exist (MODELS; stages skip a model without `best`).

Stages (outputs under --out, default runs/cellspecific; tables in
results/cellspecific):
  split     confirm that no training window overlaps a test region.
  predict   CPU predictions with saved maps (scripts/crosscell_predict.py
            --tta --save-pred, threads <= 8 in total). Existing dumps of the
            same call (runs/benchmark/pred, runs/worst_regions/pred_ref25,
            aml/pred) are linked instead of recomputed.
  pairref   control for the residual reference models: both cells of a pair
            predicted with one common reference (consensus without both
            cells), so the reference carries no difference (PAIR_CELLS, the
            showcase and the review pairs).
  wide      predictions over the 2 Mb (80-bin) window centred on each 25 kb
            figure region, for the figures only.
  calls     loops per measured map at 10 kb, hicDetectLoops of the HiCExplorer
            3.7.6 release with the settings of scripts/benchmark_competitors.py
            (the v4 C++ hicDetectLoops aborts with a double free); calls that
            exist in runs/benchmark/calls are reused.
  e1        genome-wide cis E1 (cooltools, 100 kb, phased by GC content) per
            measured map, for the compartment state of each bin.
  load      per resolution one cache of every measured map, prediction and
            consensus on the 413 regions (npz per source).
  diffmap   per region and cell pair: Pearson of the predicted difference
            (pred_A - pred_B) with the measured difference (meas_A - meas_B),
            per model and for the consensus; replicate-difference noise
            floor from independent replicate pairs.
  features  differential loops (log O/E 3x3 centre minus donut ring, called
            in one cell and absent in the other), differential boundaries
            (insulation minima with a large insulation difference) and
            compartment flips (E1 sign change between cells; orientation-free
            contact score of the flipping bins with stable A and B bins);
            per feature the direction predicted by each model.
  showcase  rank candidate regions and pick the showcase set.
  figures   one figure per showcase region and per review-set region
            (showcase80_*), PNG + npz with every panel; 80 bins per panel at
            every resolution (10 kb: 800 kb crop inside the test region;
            25 kb: 2 Mb window with the test region framed); every metric on
            the test region only.
  summary   tables and STATUS.md numbers.

    python scripts/cellspecific.py --stage split predict pairref wide --jobs 2 --threads 3
    python scripts/cellspecific.py --stage predict pairref wide --device cuda --gpu 0 --jobs 1
    python scripts/cellspecific.py --stage calls e1 --jobs 2 --threads 4
    python scripts/cellspecific.py --stage load diffmap features showcase figures summary

Prediction, pair consensus, loop-call and E1 outputs are claimed with lock
files, so several CPU and GPU worker processes can share one queue.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
RUNS = HM / "runs"
MR = RUNS / "multicell_reference"
AML = HM / "aml"
OUT = RUNS / "cellspecific"
TABLES = REPO / "results/cellspecific"
PY = sys.executable
HX376 = Path.home() / "miniconda3/envs/__hicexplorer@3.7.6/bin"
PP = HM.parent / "restart_of_prediction_project"

TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
           "gm23248", "mcf7", "cd14mono", "pc9"]
HELDOUT = ["k562", "hmec_deep", "huvec_deep"]
AMLC = ["molm13", "u937", "u937_ko"]
CELLS = TRAIN12 + HELDOUT + AMLC
LABEL = {"gm12878": "GM12878", "imr90": "IMR-90", "hct116": "HCT116", "hff": "HFF",
         "h1": "H1", "hepg2": "HepG2", "a549": "A549", "nhek": "NHEK", "gm23248": "GM23248",
         "mcf7": "MCF-7", "cd14mono": "CD14+ mono", "pc9": "PC-9", "k562": "K562",
         "hmec_deep": "HMEC", "huvec_deep": "HUVEC", "molm13": "MOLM-13",
         "u937": "U-937 WT", "u937_ko": "U-937 STAG2-KO"}
RESOLUTIONS = [10000, 25000]
# model family -> uses the consensus reference plane
MODEL_REF = {"all12refres": True, "all12honest": False,
             "all12refres_drop": True, "all12honest_drop": False}
# AML: target map label (aml/oe) and prediction input set (aml/pred)
AML_TARGET = {"molm13": "molm13", "u937": "u937_wt_c1", "u937_ko": "u937_ko_c1"}
AML_INPUT = {"molm13": "molm13", "u937": "u937", "u937_ko": "u937_ko_absent"}


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "cellspecific.log", "a") as fh:
        fh.write(line + "\n")


def run_dir(model: str, res: int) -> Path:
    return MR / f"{model}_{res // 1000}kb"


def model_ready(model: str, res: int) -> bool:
    """Trained to the end (early stop by patience), not a checkpoint of a run
    still in progress."""
    lg = MR / f"{model}_{res // 1000}kb.log"
    return ((run_dir(model, res) / "best").exists() and lg.exists()
            and "early stop at epoch" in lg.read_text(errors="replace"))


def models_at(res: int) -> list[str]:
    return [m for m in MODEL_REF if model_ready(m, res)]


def target(cell: str, res: int) -> Path:
    k = res // 1000
    if cell in AML_TARGET:
        return AML / "oe" / f"{AML_TARGET[cell]}_proper_{k}kb.cool"
    return OE / f"{cell}_proper_{k}kb.cool"


def consensus_path(cell: str, res: int) -> Path:
    """The reference map the residual model reads for `cell`."""
    k = res // 1000
    if cell in TRAIN12:
        return OE / f"consensus_loo_{cell}_proper_{k}kb.cool"
    return OE / f"consensus12_proper_{k}kb.cool"


def pred_path(model: str, res: int, cell: str) -> Path:
    """Saved prediction of `model` for `cell`; the first existing dump of the
    identical crosscell_predict call, else this analysis' own file."""
    k = res // 1000
    cands = [RUNS / "benchmark/pred" / f"{model}_{k}kb" / f"pred_{cell}.npz"]
    if cell in AML_INPUT:                  # scripts/aml_eval.py, else our own call
        cands = [AML / "pred" / f"{model}_{k}kb" / f"{AML_INPUT[cell]}.npz"]
    if model == "all12refres" and res == 25000:
        cands.append(RUNS / "worst_regions/pred_ref25" / f"pred_{cell}.npz")
    for c in cands:
        if c.exists():
            return c
    return OUT / "pred" / f"{model}_{k}kb" / f"pred_{cell}.npz"


def regions_df() -> pd.DataFrame:
    import score_like_hicgan as sc
    r = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    r["region"] = r.chrom + ":" + r.start.astype(str) + "-" + r.end.astype(str)
    return r


# ------------------------------------------------------------------ split
def stage_split() -> None:
    """Every training cache at 10 and 25 kb: zero training windows overlapping
    an Akita test region or a test window (scripts/check_split_disjoint.py)."""
    df = pd.read_csv(REPO / "results/multicell_reference/split_disjoint.csv")
    df = df[df.cache.str.startswith("mc_all12")]
    bad = df[(df.train_overlapping_akita_test_regions > 0) | (df.train_overlapping_test_windows > 0)]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = df.groupby(["cache", "res_kb"]).agg(
        cells=("sample", "nunique"), train_windows=("train_windows", "sum"),
        overlapping_test_regions=("train_overlapping_akita_test_regions", "sum"),
        overlapping_test_windows=("train_overlapping_test_windows", "sum")).reset_index()
    out.to_csv(TABLES / "split_check.csv", index=False)
    log(f"[split] {len(df)} cache x cell rows, {len(bad)} with overlap")
    if len(bad):
        raise SystemExit("training windows overlap test regions")


# ---------------------------------------------------------------- predict
DEVICE = {"device": "cpu", "gpu": None}


def claim(out: Path) -> bool:
    """Atomic per-output lock, so several predict processes (CPU and GPU
    workers) share one queue without running a job twice."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(out) + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()} {DEVICE['device']}{DEVICE['gpu'] or ''}".encode())
    os.close(fd)
    return True


def release(out: Path) -> None:
    Path(str(out) + ".lock").unlink(missing_ok=True)


def predict_env(threads: int) -> dict:
    gpu = DEVICE["gpu"] if DEVICE["device"] == "cuda" else ""
    return dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                OPENBLAS_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES=str(gpu or ""))


def run_predict(cmd: list, out: Path, threads: int, what: str) -> None:
    if out.exists() or not claim(out):
        return
    try:
        cmd = [str(c) for c in cmd] + ["--device", DEVICE["device"], "--save-pred",
                                        str(out) + ".tmp.npz"]
        log(f"[predict] {what} on {DEVICE['device']}{DEVICE['gpu'] or ''}")
        with open(out.parent / f"predict_{out.stem.replace('pred_', '')}.log", "w") as fh:
            rc = subprocess.run(cmd, cwd=REPO, env=predict_env(threads), stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc:
            log(f"[FAILED] {what} rc={rc}")
            return
        Path(str(out) + ".tmp.npz").rename(out)
    finally:
        release(out)


def predict_one(model: str, res: int, cell: str, threads: int) -> None:
    out = pred_path(model, res, cell)
    if out.exists():
        return
    cmd = predict_cmd(model, res, cell, consensus_path(cell, res), out.parent)
    run_predict(cmd, out, threads, f"{model} {res} {cell}")


def stage_predict(jobs: int, threads: int) -> None:
    todo = [(m, r, c) for r in sorted(RESOLUTIONS, reverse=True) for m in models_at(r)
            for c in CELLS if not pred_path(m, r, c).exists()]
    log(f"[predict] {len(todo)} predictions to run")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(predict_one, m, r, c, threads) for m, r, c in todo]:
            f.result()
    miss = [(m, r, c) for r in RESOLUTIONS for m in models_at(r) for c in CELLS
            if not pred_path(m, r, c).exists()]
    log(f"[predict] missing after run: {miss}")


# ---------------------------------------------------------------- pairref
# Control for the residual reference models: for two training cells the
# leave-one-out references differ by (map_B - map_A) / 11, so the reference
# itself carries the inverted true difference. Here both cells of a pair are
# predicted with the SAME reference, the consensus of the training cells other
# than A and B (10 cells for two training cells; the training cell's
# leave-one-out map for a training / held-out pair; consensus12 for two
# held-out cells, which is the standard setting), so the reference contributes
# no difference and a predicted difference comes from the cells' ChIP alone.
PAIR_CELLS = ["gm12878", "imr90", "hct116", "h1", "hepg2", "cd14mono"]
# the review set (runs/cellspecific/review/review.py, list C): feature, res,
# region, A, B, bin i, bin j (region-local), and the context cells shown
REVIEW_SET = [
    ("loop", 10, "chr7:21997568-23046144", "imr90", "hmec_deep", 11, 50),
    ("loop", 10, "chr12:114114560-115163136", "imr90", "hmec_deep", 29, 56),
    ("loop", 10, "chr11:74381312-75429888", "huvec_deep", "u937", 9, 50),
    ("loop", 10, "chr16:51652608-52701184", "hmec_deep", "huvec_deep", 16, 32),
    ("boundary", 10, "chr6:117059584-118108160", "huvec_deep", "hmec_deep", 38, 38),
    ("boundary", 10, "chr4:7170048-8218624", "imr90", "gm12878", 59, 59),
    ("boundary", 10, "chr7:37795840-38844416", "u937", "huvec_deep", 40, 40),
    ("compartment", 25, "chr9:134488064-135536640", "gm12878", "imr90", 25, 36),
    ("compartment", 25, "chr10:16289792-17338368", "imr90", "gm12878", 25, 32),
    ("compartment", 25, "chr7:54032384-55080960", "imr90", "gm12878", 31, 38)]
REVIEW_CTX = ["gm12878", "imr90", "k562", "hmec_deep", "huvec_deep", "molm13", "u937"]


def pair_supported(a: str, b: str) -> bool:
    """A pair-common prediction needs a training cell in the pair (else both
    cells already read consensus12) and no AML cell next to a training cell
    (the AML inputs are predicted by scripts/aml_eval.py with consensus12)."""
    n_train = (a in TRAIN12) + (b in TRAIN12)
    return n_train >= 1 and not ({a, b} & set(AMLC))


def pair_list() -> list[tuple[str, str]]:
    pairs = [tuple(sorted(p)) for p in combinations(PAIR_CELLS, 2)]
    f = TABLES / "showcase.csv"
    if f.exists():
        sc = pd.read_csv(f)
        pairs += [tuple(sorted((r.a, r.b))) for r in sc.itertuples()]
    pairs += [tuple(sorted((r[3], r[4]))) for r in REVIEW_SET]
    pairs = [p for p in pairs if pair_supported(*p)]
    return sorted(set(pairs), key=pairs.index)


def pair_consensus(a: str, b: str, res: int) -> Path:
    """The reference both cells of the pair read."""
    tr = [c for c in (a, b) if c in TRAIN12]
    if len(tr) == 2:
        x, y = sorted((a, b))
        return OUT / "pair_consensus" / f"consensus_excl_{x}_{y}_proper_{res // 1000}kb.cool"
    if len(tr) == 1:
        return consensus_path(tr[0], res)
    return consensus_path(a, res)


def pair_pred_path(model: str, res: int, a: str, b: str, cell: str) -> Path:
    """Prediction of `cell` with the pair-common reference; the standard
    prediction where that reference is the cell's own."""
    if pair_consensus(a, b, res) == consensus_path(cell, res):
        return pred_path(model, res, cell)
    x, y = sorted((a, b))
    return OUT / "pred_pair" / f"{model}_{res // 1000}kb" / f"{x}__{y}" / f"pred_{cell}.npz"


def build_pair_consensus(a: str, b: str, res: int) -> None:
    out = pair_consensus(a, b, res)
    if out.exists():
        return
    if not claim(out):                      # another worker builds it
        while not out.exists() and Path(str(out) + ".lock").exists():
            time.sleep(10)
        return
    others = [c for c in TRAIN12 if c not in (a, b)]
    subprocess.run([PY, "scripts/consensus_oe.py", "--res", str(res), "--cells", *others,
                    "--out", str(out)], cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    release(out)
    log(f"[pairref] consensus without {a}, {b} at {res}")


def predict_cmd(model: str, res: int, cell: str, reference: Path | None,
                out_dir: Path, regions: Path | None = None) -> list:
    """crosscell_predict call for a training or held-out cell, or an AML input."""
    import multicell_expanded as mx
    run = run_dir(model, res)
    ck = run / (run / "best").read_text().strip()
    if cell in AML_INPUT:
        bw = AML / f"hg38_{'u937' if cell == 'u937_ko' else cell}"
        assay = "hic" if cell == "molm13" else "microc"
        tgt = OE / f"imr90_proper_{res // 1000}kb.cool"       # prediction grid only
    else:
        _, bw, assay, _ = mx.CELLS[cell]
        tgt = target(cell, res)
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
           "--checkpoint", ck, "--cell", cell if cell not in AML_INPUT else f"aml_{cell}",
           "--bigwig-dir", bw, "--target", tgt, "--tta", "--assay", assay,
           "--out-dir", out_dir, "--binsize", res]
    if cell == "u937_ko":               # cohesin declared absent (registry section 19)
        import aml_eval
        cmd += ["--keep-tracks", ",".join(aml_eval.KO_ABSENT_KEEP)]
    if MODEL_REF[model] and reference is not None:
        cmd += ["--reference", reference]
    if regions is not None:
        cmd += ["--regions", regions]
    return cmd


def predict_pair_one(model: str, res: int, a: str, b: str, cell: str, threads: int) -> None:
    out = pair_pred_path(model, res, a, b, cell)
    if out.exists():
        return
    build_pair_consensus(a, b, res)
    cmd = predict_cmd(model, res, cell, pair_consensus(a, b, res), out.parent)
    run_predict(cmd, out, threads, f"{model} {res} {cell} (reference without {a}, {b})")


def stage_pairref(jobs: int, threads: int) -> None:
    """Consensus maps are built on demand by the first worker that needs one;
    each worker walks the queue in its own order (25 kb first), so parallel
    workers build different maps instead of waiting for the same one."""
    import random
    pairs = pair_list()
    todo = []
    for r in sorted(RESOLUTIONS, reverse=True):
        part = [(m, r, a, b, c) for m in models_at(r) if MODEL_REF[m] for a, b in pairs
                for c in (a, b) if not pair_pred_path(m, r, a, b, c).exists()]
        random.Random(os.getpid()).shuffle(part)
        todo += part
    log(f"[pairref] {len(pairs)} pairs, {len(todo)} predictions to run")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(predict_pair_one, *t, threads) for t in todo]:
            f.result()


# ------------------------------------------------------------------ calls
# source count maps (balanced) behind each proper O/E target
MEASURED = {
    "gm12878": PP / "originalData/4Dnucleosome/GM12878_4DNFIXP4QG5B.mcool",
    "h1": PP / "originalData/4Dnucleosome/H1_4DNFI6HDY7WZ.mcool",
    "k562": PP / "originalData/4Dnucleosome/K562_4DNFI18UHVRO.mcool",
    "imr90": HM.parent / "hyperparameter_publication/imr90.mcool",
    "hff": HM / "matrices/hff_microc.mcool",
    "hct116": HM / "matrices/hct116_untreated.mcool",
    "molm13": AML / "molm13/GSM9769670_MOLM13-WT-HiC.mcool",
    "u937": AML / "u937/GSM9916468_WT_dmso_Rep1_all.250.mcool",
    "u937_ko": AML / "u937/GSM9916471_STAG2_KO_dmso_Rep1_all.250.mcool",
}
for _c in ["hepg2", "a549", "nhek", "gm23248", "mcf7", "cd14mono", "pc9", "hmec_deep",
           "huvec_deep"]:
    MEASURED[_c] = HM / "encode_hic" / f"{_c}_{{k}}kb.cool"
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def measured_uri(cell: str, res: int) -> str:
    p = str(MEASURED[cell])
    return p.format(k=res // 1000) if "{k}" in p else f"{p}::/resolutions/{res}"


def loops_file(cell: str) -> Path:
    b = RUNS / "benchmark/calls" / f"{cell}_10kb_loops.bedgraph"
    return b if b.exists() else OUT / "calls" / f"{cell}_10kb_loops.bedgraph"


def extract_cool(cell: str, res: int, band_bp: int = 3_000_000) -> Path:
    """Single-weight, band-limited cis copy (as benchmark_competitors.extract;
    mcools with several weight columns trip the tools)."""
    import cooler
    out = OUT / "calls" / f"{cell}_{res // 1000}kb.cool"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(measured_uri(cell, res))
    bins = clr.bins()[:][["chrom", "start", "end", "weight"]]
    band = band_bp // res
    keep = [c for c in clr.chromnames if c in CHROMS]
    step = 10_000_000

    def gen():
        mat = clr.matrix(balance=False, as_pixels=True)
        for ch in keep:
            size = int(clr.chromsizes[ch])
            for s in range(0, size, step):
                e = min(s + step, size)
                p = mat.fetch(f"{ch}:{s}-{e}", f"{ch}:{s}-{min(e + band_bp, size)}")
                p = p[(p.bin2_id >= p.bin1_id) & (p.bin2_id - p.bin1_id <= band)]
                p = p.drop_duplicates(["bin1_id", "bin2_id"])
                yield p[["bin1_id", "bin2_id", "count"]].sort_values(["bin1_id", "bin2_id"])
    kind = clr.pixels()[:1]["count"].dtype.kind
    tmp = out.with_suffix(".tmp.cool")
    cooler.create_cooler(str(tmp), bins=bins, pixels=gen(), ordered=True,
                         dtypes={"count": np.float64 if kind == "f" else np.int32})
    tmp.rename(out)
    return out


def call_one(cell: str, threads: int) -> None:
    lf = loops_file(cell)
    if lf.exists() or not claim(lf):
        return
    try:
        _call_one(cell, threads, lf)
    finally:
        release(lf)


def _call_one(cell: str, threads: int, lf: Path) -> None:
    uri = str(extract_cool(cell, 10000))
    log(f"[calls] hicDetectLoops {cell}")
    with open(OUT / "calls" / f"{cell}_10kb.calls.log", "w") as fh:
        rc = subprocess.run([str(HX376 / "hicDetectLoops"), "-m", uri, "-o", str(lf) + ".tmp",
                             "--maxLoopDistance", "2000000", "-pit", "0", "-t", str(threads),
                             "-tpc", "1", "--chromosomes", *CHROMS],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc == 0 and Path(str(lf) + ".tmp").exists():
        Path(str(lf) + ".tmp").rename(lf)
    log(f"[calls] {cell} rc={rc}")


def stage_calls(jobs: int, threads: int) -> None:
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(call_one, c, threads) for c in CELLS]:
            f.result()


def read_loops(cell: str) -> pd.DataFrame:
    f = loops_file(cell)
    if not f.exists():
        return pd.DataFrame(columns=["chrom", "s1", "s2", "p"])
    d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 4, 6],
                    names=["chrom", "s1", "s2", "p"])
    return d


# --------------------------------------------------------------------- e1
E1_RES = 100000
FASTA = Path.home() / "data/hic-multiome/genome/hg38.fa"


def e1_file(cell: str) -> Path:
    return OUT / "e1" / f"{cell}_e1_100kb.tsv"


def e1_cool(cell: str) -> str:
    """100 kb balanced cooler: the mcool level where one exists, else the 25 kb
    ENCODE cooler coarsened x4 and re-balanced (cooler ICE)."""
    p = str(MEASURED[cell])
    if "{k}" not in p:
        return f"{p}::/resolutions/{E1_RES}"
    out = OUT / "e1" / f"{cell}_100kb.cool"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        bind = Path(PY).parent
        tmp = out.with_suffix(".tmp.cool")
        subprocess.run([str(bind / "cooler"), "coarsen", "-k", "4", "-o", str(tmp),
                        p.format(k=25)], check=True)
        subprocess.run([str(bind / "cooler"), "balance", "--force", "--ignore-diags", "2", "-p", "2",
                        "--max-iters", "500", str(tmp)], check=True)
        tmp.rename(out)
    return str(out)


def gc_track(bins: pd.DataFrame) -> np.ndarray:
    import pysam
    fh = pysam.FastaFile(str(FASTA))
    out = np.full(len(bins), np.nan)
    for i, (c, s, e) in enumerate(zip(bins.chrom, bins.start, bins.end)):
        if c not in fh.references:
            continue
        seq = fh.fetch(c, int(s), int(e)).upper()
        acgt = sum(seq.count(x) for x in "ACGT")
        if acgt > 0.5 * len(seq):
            out[i] = (seq.count("G") + seq.count("C")) / acgt
    return out


def stage_e1() -> None:
    import bioframe
    import cooler
    import cooltools
    gc_cache = OUT / "e1" / "gc_100kb.tsv"
    for cell in CELLS:
        if e1_file(cell).exists() or not claim(e1_file(cell)):
            continue
        try:
            e1_one(cell, gc_cache, bioframe, cooler, cooltools)
        finally:
            release(e1_file(cell))


def e1_one(cell, gc_cache, bioframe, cooler, cooltools) -> None:
    clr = cooler.Cooler(e1_cool(cell))
    bins = clr.bins()[:][["chrom", "start", "end"]]
    bins = bins[bins.chrom.isin(CHROMS)].reset_index(drop=True)
    if gc_cache.exists():
        gc = pd.read_csv(gc_cache, sep="\t")
    else:
        gc = bins.copy()
        gc["GC"] = gc_track(gc)
        gc_cache.parent.mkdir(parents=True, exist_ok=True)
        gc.to_csv(gc_cache, sep="\t", index=False)
    ph = bins.merge(gc, on=["chrom", "start", "end"], how="left")
    chroms = [c for c in clr.chromnames if c in CHROMS]
    view = bioframe.make_viewframe([(c, 0, int(clr.chromsizes[c])) for c in chroms])
    _, eig = cooltools.eigs_cis(clr, phasing_track=ph[["chrom", "start", "end", "GC"]],
                                view_df=view, n_eigs=1)
    eig[["chrom", "start", "end", "E1"]].to_csv(e1_file(cell), sep="\t", index=False)
    log(f"[e1] {cell}: {np.isfinite(eig.E1).sum()} bins, A share "
        f"{(eig.E1 > 0).mean():.3f}")


# ------------------------------------------------------------------- load
# independent replicate pairs (proper O/E of each replicate) per cell: single
# lane in situ Hi-C (registry section 7), deep intact Hi-C (section 12), the
# second MOLM-13 map and the U-937 clones (section 19)
REPLICATES = {
    "insitu_lane": {c: (f"{c.replace('_deep', '')}_rep1", f"{c.replace('_deep', '')}_rep2")
                    for c in ["gm12878", "imr90", "k562", "hmec_deep", "huvec_deep"]},
    "intact_deep": {"gm12878": ("intact_gm12878_ENCSR965OVH_r75", "intact_gm12878_ENCSR965OVH_r76"),
                    "k562": ("intact_k562_ENCSR676BEP_r5", "intact_k562_ENCSR676BEP_r7")},
    "aml": {"molm13": ("molm13", "molm13_xu"), "u937": ("u937_wt_c2", "u937_wt_c3"),
            "u937_ko": ("u937_ko_c1", "u937_ko_c2")},
}


def noise_pairs() -> list[tuple]:
    """(set, cell A, cell B, (A map 1, B map 1), (A map 2, B map 2)): two
    differences A - B from maps that share nothing. The second difference
    never contains the target map of A or B."""
    out = []
    lane = REPLICATES["insitu_lane"]
    for a, b in combinations(list(lane), 2):
        out.append(("insitu_lane", a, b, (lane[a][0], lane[b][0]), (lane[a][1], lane[b][1])))
    d = REPLICATES["intact_deep"]
    out.append(("intact_deep", "gm12878", "k562", (d["gm12878"][0], d["k562"][0]),
                (d["gm12878"][1], d["k562"][1])))
    out.append(("aml", "molm13", "u937", ("molm13", "u937_wt_c2"), ("molm13_xu", "u937_wt_c3")))
    out.append(("aml", "u937", "u937_ko", ("u937_wt_c2", "u937_ko_c1"),
                ("u937_wt_c3", "u937_ko_c2")))
    return out


# replicate maps independent of the target, per cell (direction checks of the
# features use replicate k of both cells)
INDEP_REPS = {c: pair for c, pair in REPLICATES["insitu_lane"].items()}
INDEP_REPS.update({"molm13": ("molm13_xu",), "u937": ("u937_wt_c2", "u937_wt_c3"),
                   "u937_ko": ("u937_ko_c2",)})


def rep_path(label: str, res: int) -> Path:
    k = res // 1000
    if label.startswith(("molm13", "u937")):
        return AML / "oe" / f"{label}_proper_{k}kb.cool"
    if label.startswith("intact_"):
        return OE / f"{label}_ice_{k}kb.cool"
    return OE / f"{label}_ice_{k}kb.cool"


def local_copy(p: Path, local: Path) -> Path:
    """Cooler fetches over the CIFS share cost ~0.8 s each; read a local copy."""
    import shutil
    dst = local / (p.parent.name + "__" + p.name)
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        local.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, str(dst) + ".tmp")
        os.replace(str(dst) + ".tmp", dst)
    return dst


def nbins(res: int) -> int:
    return 106 if res == 10000 else 44


def fetch_stack(path: Path, regions: pd.DataFrame, res: int, local: Path) -> np.ndarray:
    """(regions, N, N) float32 of a proper O/E cool, zero (unmeasured) -> NaN,
    NaN-padded to N bins."""
    import cooler
    from hicmultiome.genome import match_name
    clr = cooler.Cooler(str(local_copy(path, local)))
    N = nbins(res)
    out = np.full((len(regions), N, N), np.nan, np.float32)
    for i, r in enumerate(regions.itertuples()):
        nm = match_name(r.chrom, list(clr.chromnames))
        if nm is None:
            continue
        try:
            M = np.asarray(clr.matrix(balance=False).fetch(f"{nm}:{r.start}-{r.end}"), float)
        except Exception:                                        # noqa: BLE001
            continue
        n = min(N, M.shape[0])
        out[i, :n, :n] = np.where(M[:n, :n] == 0, np.nan, M[:n, :n])
    return out


def pred_stack(npz: Path, regions: pd.DataFrame, res: int) -> np.ndarray:
    """Saved predictions with crosscell_predict's live-row rule, NaN-padded."""
    z = np.load(npz)
    N = nbins(res)
    out = np.full((len(regions), N, N), np.nan, np.float32)
    for i, reg in enumerate(regions.region):
        if reg not in z.files:
            continue
        P = z[reg].astype(np.float32)
        live = np.abs(np.nan_to_num(P)).sum(1) > 0
        P = np.where(live[:, None] & live[None, :], P, np.nan)
        n = min(N, P.shape[0])
        out[i, :n, :n] = P[:n, :n]
    return out


def cache_file(res: int, source: str) -> Path:
    return OUT / "cache" / f"{res // 1000}kb" / f"{source}.npy"


def load(res: int, source: str) -> np.ndarray | None:
    f = cache_file(res, source)
    return np.load(f, mmap_mode="r") if f.exists() else None


def stage_load(local: Path) -> None:
    regions = regions_df()
    for res in RESOLUTIONS:
        jobs = {}
        for c in CELLS:
            jobs[f"meas__{c}"] = ("cool", target(c, res))
            jobs[f"cons__{c}"] = ("cool", consensus_path(c, res))
            for m in models_at(res):
                jobs[f"pred__{m}__{c}"] = ("npz", pred_path(m, res, c))
        for grp in REPLICATES.values():
            for c, (r1, r2) in grp.items():
                for lab in (r1, r2):
                    jobs[f"rep__{lab}"] = ("cool", rep_path(lab, res))
        for m in models_at(res):
            if not MODEL_REF[m]:
                continue
            for a, b in pair_list():
                for c in (a, b):
                    p = pair_pred_path(m, res, a, b, c)
                    if p.exists():
                        jobs[f"pairref__{m}__{a}__{b}__{c}"] = ("npz", p)
        for src, (kind, p) in jobs.items():
            f = cache_file(res, src)
            if f.exists():
                continue
            if not Path(p).exists():
                log(f"[load] {res} {src}: {p} missing")
                continue
            arr = (fetch_stack(Path(p), regions, res, local) if kind == "cool"
                   else pred_stack(Path(p), regions, res))
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(f) + ".tmp.npy", arr)
            os.replace(str(f) + ".tmp.npy", f)
            log(f"[load] {res} {src}: {np.isfinite(arr).mean():.3f} finite")


def sources_of(res: int) -> list[str]:
    """Prediction sources compared per cell: the models whose predictions of
    every cell are loaded, then the consensus."""
    return [f"pred__{m}" for m in models_at(res)
            if all(cache_file(res, f"pred__{m}__{c}").exists() for c in CELLS)] + ["cons"]


def pairref_maps(res: int, a: str, b: str) -> dict[str, tuple]:
    """Pair-common reference predictions for cells a and b, per reference
    model: {'pairref__<model>': (map stack of a, map stack of b)}."""
    x, y = sorted((a, b))
    out = {}
    for m in models_at(res):
        if not MODEL_REF[m]:
            continue
        pa = load(res, f"pairref__{m}__{x}__{y}__{a}")
        pb = load(res, f"pairref__{m}__{x}__{y}__{b}")
        if pa is not None and pb is not None:
            out[f"pairref__{m}"] = (pa, pb)
    return out


# ----------------------------------------------------------------- diffmap
def pair_kind(a: str, b: str) -> str:
    grp = lambda c: "train" if c in TRAIN12 else "heldout" if c in HELDOUT else "aml"
    if {a, b} == {"u937", "u937_ko"}:
        return "u937_wt_vs_ko"
    return "-".join(sorted([grp(a), grp(b)], key=["train", "heldout", "aml"].index))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return np.nan
    a, b = x[ok], y[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def diff_scores(mA, mB, preds: dict[str, tuple[np.ndarray, np.ndarray]], k: int = 2) -> dict:
    """Pearson of predicted difference with measured difference on one pixel
    mask (upper triangle, offset >= k, finite in both measured maps and in
    every prediction), plus the SD ratio predicted / measured difference."""
    n = mA.shape[0]
    iu = np.triu_indices(n, k)
    d = (mA - mB)[iu]
    ok = np.isfinite(d)
    for pa, pb in preds.values():
        ok &= np.isfinite(pa[iu]) & np.isfinite(pb[iu])
    out = {"n_px": int(ok.sum()), "meas_sd": float(d[ok].std()) if ok.sum() else np.nan}
    for s, (pa, pb) in preds.items():
        pd_ = (pa - pb)[iu]
        out[f"r__{s}"] = corr(pd_[ok], d[ok]) if ok.sum() >= 100 else np.nan
        out[f"sd__{s}"] = float(pd_[ok].std() / d[ok].std()) if ok.sum() >= 100 and d[ok].std() > 0 else np.nan
    return out


def stage_diffmap() -> None:
    regions = regions_df()
    rows, noise = [], []
    for res in RESOLUTIONS:
        srcs = sources_of(res)
        meas = {c: load(res, f"meas__{c}") for c in CELLS}
        pred = {s: {c: load(res, f"{s}__{c}") for c in CELLS} for s in srcs}
        cells = [c for c in CELLS if meas[c] is not None
                 and all(pred[s][c] is not None for s in srcs)]
        for a, b in combinations(cells, 2):
            pr = pairref_maps(res, a, b)
            for i, reg in enumerate(regions.region):
                pp = {s: (pred[s][a][i], pred[s][b][i]) for s in srcs}
                rec = diff_scores(meas[a][i], meas[b][i], pp)
                if pr:
                    # the same pixels with the pair-common reference predictions
                    pp.update({s: (x[i], y[i]) for s, (x, y) in pr.items()})
                    rec2 = diff_scores(meas[a][i], meas[b][i], pp)
                    rec.update({f"pc_{k}": v for k, v in rec2.items()
                                if k.startswith(("r__", "sd__"))})
                rows.append({"res_kb": res // 1000, "region": reg, "a": a, "b": b,
                             "kind": pair_kind(a, b), **rec})
        # noise floor: two measurements of the same difference that share no map
        # (r_rep), and the target difference against one independent
        # replicate difference (r_rep_vs_target)
        for set_, a, b, (a1, b1), (a2, b2) in noise_pairs():
            reps = {lab: load(res, f"rep__{lab}") for lab in (a1, b1, a2, b2)}
            if any(v is None for v in reps.values()):
                continue
            for i, reg in enumerate(regions.region):
                x = reps[a1][i] - reps[b1][i]
                y = reps[a2][i] - reps[b2][i]
                iu = np.triu_indices(x.shape[0], 2)
                noise.append({"res_kb": res // 1000, "region": reg, "set": set_,
                              "a": a, "b": b, "r_rep": corr(x[iu], y[iu]),
                              "r_rep_vs_target": corr(y[iu], (meas[a][i] - meas[b][i])[iu])
                              if meas.get(a) is not None and meas.get(b) is not None else np.nan})
    df = pd.DataFrame(rows)
    nz = pd.DataFrame(noise)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "diffmap_region.csv.gz", index=False)
    nz.to_csv(OUT / "diffmap_noise_region.csv.gz", index=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    rcols = [c for c in df.columns if c.startswith(("r__", "sd__"))]
    pccols = [c for c in df.columns if c.startswith("pc_")]
    pairs = df.groupby(["res_kb", "a", "b", "kind"])[rcols + pccols + ["meas_sd"]].median().reset_index()
    if pccols:
        # pair-common reference control: the pairs that have it, all columns on
        # the joint pixel mask of the pair-common predictions
        pc = pairs.dropna(subset=[c for c in pccols if c.startswith("pc_r__pairref")], how="all")
        pc.to_csv(TABLES / "diffmap_pairref.csv", index=False, float_format="%.4f")
        agg = pc.groupby("res_kb")[pccols].median()
        agg["pairs"] = pc.groupby("res_kb").size()
        agg.reset_index().to_csv(TABLES / "diffmap_pairref_summary.csv", index=False,
                                 float_format="%.4f")
        log("[diffmap] pair-common reference\n" + agg.round(3).to_string())
    nzs = nz.groupby(["res_kb", "set", "a", "b"])[["r_rep", "r_rep_vs_target"]].median().reset_index()
    pairs = pairs.merge(nzs.drop(columns="set").groupby(["res_kb", "a", "b"]).max().reset_index(),
                        on=["res_kb", "a", "b"], how="left")
    pairs.to_csv(TABLES / "diffmap_pairs.csv", index=False, float_format="%.4f")
    nzs.to_csv(TABLES / "diffmap_noise.csv", index=False, float_format="%.4f")
    kinds = pairs.groupby(["res_kb", "kind"])[rcols].median()
    kinds["pairs"] = pairs.groupby(["res_kb", "kind"]).size()
    kinds.reset_index().to_csv(TABLES / "diffmap_kinds.csv", index=False, float_format="%.4f")
    log("[diffmap]\n" + kinds.round(3).to_string())


# ---------------------------------------------------------------- features
LOOP_MIN_D = 5          # bins (10 kb): loops from 50 kb
LOOP_MARGIN = 7         # donut ring reach
INS_W = 10              # insulation diamond, 100 kb at 10 kb


def loop_strength(L: np.ndarray, i: int, j: int) -> float:
    """log O/E at the loop pixel (3 x 3 mean) minus its donut ring (Chebyshev
    distance 4 to 7 bins); the maps are distance-normalised, so the ring is
    not biased by decay."""
    n = L.shape[0]
    if i - 7 < 0 or j + 8 > n or j - i < LOOP_MIN_D:
        return np.nan
    c = L[i - 1:i + 2, j - 1:j + 2]
    blk = L[i - 7:i + 8, j - 7:j + 8]
    di, dj = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8), indexing="ij")
    ring = np.maximum(np.abs(di), np.abs(dj)) >= 4
    ring &= (j + dj - (i + di)) >= 2          # keep the ring off the diagonal
    if np.isfinite(c).mean() < 0.6 or np.isfinite(blk[ring]).mean() < 0.5:
        return np.nan
    return float(np.nanmean(c) - np.nanmean(blk[ring]))


def insulation(L: np.ndarray, w: int = INS_W) -> np.ndarray:
    """Diamond mean of log O/E (w bins each side); NaN within w of the edge."""
    n = L.shape[0]
    prof = np.full(n, np.nan)
    for b in range(w, n - w):
        blk = L[b - w:b, b + 1:b + w + 1]
        if np.isfinite(blk).mean() > 0.5:
            prof[b] = np.nanmean(blk)
    return prof


def boundary_strength(prof: np.ndarray, b: int) -> float:
    """Depth of the insulation dip at b: mean of the flanks 4 to 8 bins away
    minus the profile at b (positive = insulating boundary)."""
    n = prof.size
    if b - 8 < 0 or b + 9 > n or not np.isfinite(prof[b]):
        return np.nan
    fl = np.concatenate([prof[b - 8:b - 3], prof[b + 4:b + 9]])
    if np.isfinite(fl).sum() < 6:
        return np.nan
    return float(np.nanmean(fl) - prof[b])


def boundary_calls(prof: np.ndarray, min_depth: float) -> np.ndarray:
    """Local minima (within +-3 bins) with a dip of at least min_depth."""
    out = []
    for b in range(3, prof.size - 3):
        v = prof[b]
        if not np.isfinite(v):
            continue
        win = prof[b - 3:b + 4]
        if v <= np.nanmin(win) and boundary_strength(prof, b) >= min_depth:
            out.append(b)
    return np.asarray(out, int)


def flip_score(M: np.ndarray, flip: np.ndarray, stable_sign: np.ndarray,
               min_sep: int = 3) -> float:
    """Contact of the flipping bins with the stable bins, signed by the stable
    bins' compartment (+1 A, -1 B): mean of M[f, s] * sign(s) over pairs at
    least min_sep bins apart. Higher = the flipping bins behave as A."""
    fi = np.where(flip)[0]
    si = np.where(stable_sign != 0)[0]
    if len(fi) == 0 or len(si) == 0:
        return np.nan
    F, S = np.meshgrid(fi, si, indexing="ij")
    keep = np.abs(F - S) >= min_sep
    v = M[F[keep], S[keep]] * stable_sign[S[keep]]
    ok = np.isfinite(v)
    return float(v[ok].mean()) if ok.sum() >= 10 else np.nan


def e1_region(e1: pd.DataFrame, chrom: str, start: int, n: int, res: int) -> np.ndarray:
    """E1 (z-scored genome-wide) of each res bin of a region, from its 100 kb bin."""
    sub = e1[e1.chrom == chrom]
    lut = dict(zip(sub.start // E1_RES, sub.z))
    b0 = start // res
    return np.array([lut.get(((b0 + i) * res) // E1_RES, np.nan) for i in range(n)])


def load_e1() -> dict[str, pd.DataFrame]:
    out = {}
    for c in CELLS:
        if e1_file(c).exists():
            e = pd.read_csv(e1_file(c), sep="\t")
            e["z"] = e.E1 / np.nanstd(e.E1)
            out[c] = e
    return out


def union_loops(regions: pd.DataFrame, res: int = 10000) -> pd.DataFrame:
    """Every called loop (any cell) inside a test region, as region-local bins;
    calls of different cells within 1 bin merged; `called_<cell>` flags."""
    recs = {}
    for c in CELLS:
        lp = read_loops(c)
        if lp.empty:
            continue
        for ri, r in enumerate(regions.itertuples()):
            sub = lp[(lp.chrom == r.chrom) & (lp.s1 >= r.start) & (lp.s2 < r.end)]
            b0 = r.start // res
            for s1, s2 in zip(sub.s1, sub.s2):
                i, j = s1 // res - b0, s2 // res - b0
                key = None
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if (ri, i + di, j + dj) in recs:
                            key = (ri, i + di, j + dj)
                if key is None:
                    key = (ri, i, j)
                    recs[key] = set()
                recs[key].add(c)
    rows = []
    for (ri, i, j), cs in recs.items():
        rows.append({"ri": ri, "region": regions.region.iloc[ri], "i": i, "j": j,
                     **{f"called_{c}": c in cs for c in CELLS}})
    return pd.DataFrame(rows)


def near_call(lp: pd.DataFrame, chrom: str, s1: int, s2: int, tol: int) -> bool:
    sub = lp[lp.chrom == chrom]
    return bool(((np.abs(sub.s1 - s1) <= tol) & (np.abs(sub.s2 - s2) <= tol)).any())


def direction_rows(kind_of_feature: str, meas_a, meas_b, preds_a: dict, preds_b: dict,
                   base: dict) -> dict:
    """Measured difference and, per source, the predicted difference and
    whether its sign matches (tie = 0.5)."""
    d = meas_a - meas_b
    rec = {**base, "feature": kind_of_feature, "meas_a": meas_a, "meas_b": meas_b, "meas_diff": d}
    for s in preds_a:
        pdiff = preds_a[s] - preds_b[s]
        rec[f"diff__{s}"] = pdiff
        rec[f"ok__{s}"] = (np.nan if not np.isfinite(pdiff) else
                           0.5 if pdiff == 0 else float(np.sign(pdiff) == np.sign(d)))
    return rec


def genomic(regions: pd.DataFrame, ri: int, i: int, j: int, res: int, snap: int = 0) -> dict:
    """Genomic position of region-local bins i, j (snapped to `snap` bp)."""
    r = regions.iloc[ri]
    g1, g2 = (r.start // res + i) * res, (r.start // res + j) * res
    if snap:
        g1, g2 = g1 // snap * snap, g2 // snap * snap
    return {"chrom": r.chrom, "g1": int(g1), "g2": int(g2)}


def add_pairref(rec: dict, maps: dict, fn) -> None:
    """Predicted difference and direction under the pair-common reference;
    maps: {'pairref__<model>': (stack of rec['a'], stack of rec['b'])}."""
    for s, (xa, xb) in maps.items():
        d = fn(xa) - fn(xb)
        rec[f"diff__{s}"] = d
        rec[f"ok__{s}"] = (np.nan if not np.isfinite(d) else 0.5 if d == 0
                           else float(np.sign(d) == np.sign(rec["meas_diff"])))


def stage_features() -> None:
    regions = regions_df()
    rows = []
    # ---- loops and boundaries at 10 kb
    res = 10000
    srcs = sources_of(res)
    meas = {c: load(res, f"meas__{c}") for c in CELLS}
    pred = {s: {c: load(res, f"{s}__{c}") for c in CELLS} for s in srcs}
    reps = {lab: load(res, f"rep__{lab}") for g in REPLICATES.values()
            for pair in g.values() for lab in pair}
    rep_of = INDEP_REPS
    cells = [c for c in CELLS if meas[c] is not None and all(pred[s][c] is not None for s in srcs)]
    ul = union_loops(regions, res)
    log(f"[features] {len(ul)} loops (union of calls) in test regions")
    strength = {}                                    # (source, cell) -> array per loop
    for c in cells:
        strength[("meas", c)] = np.array([loop_strength(meas[c][r.ri], r.i, r.j)
                                          for r in ul.itertuples()])
        for s in srcs:
            strength[(s, c)] = np.array([loop_strength(pred[s][c][r.ri], r.i, r.j)
                                         for r in ul.itertuples()])
        if c in rep_of:
            for k, lab in enumerate(rep_of[c]):
                if reps.get(lab) is not None:
                    strength[(f"rep{k + 1}", c)] = np.array(
                        [loop_strength(reps[lab][r.ri], r.i, r.j) for r in ul.itertuples()])
    lt = ul.copy()
    for (s, c), v in strength.items():
        lt[f"S__{s}__{c}"] = v
    lt.to_csv(OUT / "loops_strength.csv.gz", index=False)
    for a, b in combinations(cells, 2):
        sa, sb = strength[("meas", a)], strength[("meas", b)]
        for li, r in enumerate(ul.itertuples()):
            ca, cb = getattr(r, f"called_{a}"), getattr(r, f"called_{b}")
            if ca == cb or not (np.isfinite(sa[li]) and np.isfinite(sb[li])):
                continue
            # A = the cell with the call
            hi, lo = (a, b) if ca else (b, a)
            s_hi, s_lo = strength[("meas", hi)][li], strength[("meas", lo)][li]
            if s_hi < 0.3 or s_hi - s_lo < 0.3:
                continue
            base = {"res_kb": 10, "region": r.region, "a": hi, "b": lo, "kind": pair_kind(a, b),
                    "i": r.i, "j": r.j, **genomic(regions, r.ri, r.i, r.j, res)}
            rec = direction_rows("loop", s_hi, s_lo, {s: strength[(s, hi)][li] for s in srcs},
                                 {s: strength[(s, lo)][li] for s in srcs}, base)
            add_pairref(rec, pairref_maps(res, hi, lo),
                        lambda M, r=r: loop_strength(M[r.ri], r.i, r.j))
            for k in (1, 2):
                x, y = strength.get((f"rep{k}", hi)), strength.get((f"rep{k}", lo))
                if x is not None and y is not None and np.isfinite(x[li] - y[li]):
                    rec[f"ok__rep{k}"] = float(x[li] - y[li] > 0)
            rows.append(rec)
    log(f"[features] {sum(r['feature'] == 'loop' for r in rows)} differential loop x pair")
    # boundaries: insulation dips
    prof = {}
    for c in cells:
        prof[("meas", c)] = [insulation(meas[c][i]) for i in range(len(regions))]
        for s in srcs:
            prof[(s, c)] = [insulation(pred[s][c][i]) for i in range(len(regions))]
        if c in rep_of:
            for k, lab in enumerate(rep_of[c]):
                if reps.get(lab) is not None:
                    prof[(f"rep{k + 1}", c)] = [insulation(reps[lab][i]) for i in range(len(regions))]
    calls = {c: [boundary_calls(p, 0.2) for p in prof[("meas", c)]] for c in cells}
    bs = lambda key, i, b: boundary_strength(prof[key][i], b)
    btab = []
    for i, reg in enumerate(regions.region):
        for c in cells:
            for b in calls[c][i]:
                btab.append({"region": reg, "ri": i, "cell": c, "bin": int(b),
                             "depth": bs(("meas", c), i, b)})
    pd.DataFrame(btab).to_csv(OUT / "boundaries_meas.csv.gz", index=False)
    n_bnd = 0
    for a, b in combinations(cells, 2):
        for hi, lo in ((a, b), (b, a)):
            for i, reg in enumerate(regions.region):
                for bb in calls[hi][i]:
                    d_hi = bs(("meas", hi), i, bb)
                    d_lo = bs(("meas", lo), i, bb)
                    if not (np.isfinite(d_hi) and np.isfinite(d_lo)):
                        continue
                    if d_hi < 0.3 or d_hi - d_lo < 0.25:
                        continue
                    if len(calls[lo][i]) and np.min(np.abs(calls[lo][i] - bb)) <= 2:
                        continue
                    base = {"res_kb": 10, "region": reg, "a": hi, "b": lo,
                            "kind": pair_kind(hi, lo), "i": int(bb), "j": int(bb),
                            **genomic(regions, i, int(bb), int(bb), res)}
                    rec = direction_rows("boundary", d_hi, d_lo,
                                         {s: bs((s, hi), i, bb) for s in srcs},
                                         {s: bs((s, lo), i, bb) for s in srcs}, base)
                    add_pairref(rec, pairref_maps(res, hi, lo),
                                lambda M, i=i, bb=bb: boundary_strength(insulation(M[i]), bb))
                    for k in (1, 2):
                        if (f"rep{k}", hi) in prof and (f"rep{k}", lo) in prof:
                            x = bs((f"rep{k}", hi), i, bb) - bs((f"rep{k}", lo), i, bb)
                            if np.isfinite(x):
                                rec[f"ok__rep{k}"] = float(x > 0)
                    rows.append(rec)
                    n_bnd += 1
    log(f"[features] {n_bnd} differential boundary x pair")
    # ---- compartment flips at 25 kb
    res = 25000
    srcs25 = sources_of(res)
    meas = {c: load(res, f"meas__{c}") for c in CELLS}
    pred = {s: {c: load(res, f"{s}__{c}") for c in CELLS} for s in srcs25}
    reps = {lab: load(res, f"rep__{lab}") for g in REPLICATES.values()
            for pair in g.values() for lab in pair}
    e1 = load_e1()
    cells = [c for c in CELLS if meas[c] is not None and c in e1
             and all(pred[s][c] is not None for s in srcs25)]
    N = nbins(res)
    E = {c: np.array([e1_region(e1[c], r.chrom, r.start, N, res) for r in regions.itertuples()])
         for c in cells}
    n_flip = 0
    for a, b in combinations(cells, 2):
        for hi, lo in ((a, b), (b, a)):
            for i, reg in enumerate(regions.region):
                ea, eb = E[hi][i], E[lo][i]
                flip = (ea > 0.5) & (eb < -0.5)
                if flip.sum() < 4:
                    continue
                sign = np.where((ea > 0.5) & (eb > 0.5), 1, np.where((ea < -0.5) & (eb < -0.5), -1, 0))
                if (sign != 0).sum() < 4:
                    continue
                f = lambda M: flip_score(M, flip, sign)
                ma, mb = f(meas[hi][i]), f(meas[lo][i])
                if not (np.isfinite(ma) and np.isfinite(mb)):
                    continue
                base = {"res_kb": 25, "region": reg, "a": hi, "b": lo, "kind": pair_kind(hi, lo),
                        "i": int(np.where(flip)[0][0]), "j": int(np.where(flip)[0][-1]),
                        "e1_diff": float(np.nanmean(ea[flip] - eb[flip])),
                        **genomic(regions, i, int(np.where(flip)[0][0]),
                                  int(np.where(flip)[0][-1]), res, E1_RES)}
                rec = direction_rows("compartment", ma, mb,
                                     {s: f(pred[s][hi][i]) for s in srcs25},
                                     {s: f(pred[s][lo][i]) for s in srcs25}, base)
                # the E1 call says A in `hi`: the measured contact score agrees?
                rec["ok__meas_vs_e1"] = float(ma - mb > 0)
                add_pairref(rec, pairref_maps(res, hi, lo), lambda M, i=i, f=f: f(M[i]))
                for k in (1, 2):
                    rh, rl = rep_of.get(hi), rep_of.get(lo)
                    if (rh and rl and len(rh) >= k and len(rl) >= k
                            and reps.get(rh[k - 1]) is not None and reps.get(rl[k - 1]) is not None):
                        x = f(reps[rh[k - 1]][i]) - f(reps[rl[k - 1]][i])
                        if np.isfinite(x):
                            rec[f"ok__rep{k}"] = float(x > 0)
                rows.append(rec)
                n_flip += 1
    log(f"[features] {n_flip} compartment flip x pair")
    df = pd.DataFrame(rows)
    # the Akita test regions overlap each other: one row per genomic feature
    n0 = len(df)
    df = df.drop_duplicates(["feature", "a", "b", "chrom", "g1", "g2"]).reset_index(drop=True)
    log(f"[features] {n0} rows, {len(df)} after merging overlapping regions")
    df.to_csv(OUT / "features_diff.csv.gz", index=False)
    summarize_features(df)


def balanced_direction(df: pd.DataFrame, src: str, min_n: int = 3) -> tuple[float, float, int]:
    """Direction accuracy that a per-cell offset cannot inflate. Per unordered
    pair, the features stronger in the first cell and those stronger in the
    second form two classes; balanced accuracy = mean of the two class
    accuracies (pairs with >= min_n features in each class), averaged over
    pairs. Returned with the same statistic for the pair-constant predictor
    (the sign of the pair's median predicted difference), which scores 0.5
    by construction of the balance, and the number of pairs."""
    col = f"diff__{src}"
    if col not in df:
        return np.nan, np.nan, 0
    d = df[np.isfinite(df[col])].copy()
    first = d.a < d.b
    d["p"] = np.where(first, d.a, d.b) + "|" + np.where(first, d.b, d.a)
    d["cls"] = first                         # True: stronger in the alphabetically first cell
    d["pd"] = np.where(first, d[col], -d[col])   # predicted first minus second
    bal, off = [], []
    for _, g in d.groupby("p"):
        n1, n0 = g.cls.sum(), (~g.cls).sum()
        if n1 < min_n or n0 < min_n:
            continue
        acc1 = (g.pd[g.cls] > 0).mean() + 0.5 * (g.pd[g.cls] == 0).mean()
        acc0 = (g.pd[~g.cls] < 0).mean() + 0.5 * (g.pd[~g.cls] == 0).mean()
        bal.append(0.5 * (acc1 + acc0))
    # unbalanced accuracy of the pair-constant predictor, over every pair
    for _, g in d.groupby("p"):
        s = np.sign(g.pd.median())
        off.extend(list(np.where(g.cls, s > 0, s < 0).astype(float)))
    return (float(np.mean(bal)) if bal else np.nan, float(np.mean(off)) if off else np.nan,
            len(bal))


def centred_r(g: pd.DataFrame, col: str) -> float:
    """Pearson of predicted with measured difference after removing each
    ordered pair's mean (the within-pair, feature-level agreement)."""
    ok = np.isfinite(g[col]) & np.isfinite(g.meas_diff)
    h = g[ok]
    if len(h) < 10:
        return np.nan
    key = h.a + "|" + h.b
    x = h[col] - h.groupby(key)[col].transform("mean")
    y = h.meas_diff - h.groupby(key).meas_diff.transform("mean")
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan


def summarize_features(df: pd.DataFrame, name: str = "features_direction") -> None:
    pcol = [c for c in df.columns if c.startswith("ok__pairref__")]
    if name == "features_direction" and pcol:
        # the pair-common reference control on its own rows (training pairs of
        # PAIR_CELLS and of the showcase), every source on the same features
        summarize_features(df[df[pcol].notna().any(axis=1)], "features_direction_pairref")
    okc = [c for c in df.columns if c.startswith("ok__")]
    dfc = [c for c in df.columns if c.startswith("diff__")]
    rec = []
    for (feat, kind), g in list(df.groupby(["feature", "kind"])) + [
            ((f, "all"), g) for f, g in df.groupby("feature")]:
        r = {"feature": feat, "kind": kind, "n": len(g), "pairs": g[["a", "b"]].drop_duplicates().shape[0],
             "regions": g.region.nunique(), "meas_diff_median": g.meas_diff.median()}
        for c in okc:
            v = g[c].dropna()
            r[f"acc_{c[4:]}"] = v.mean() if len(v) else np.nan
            r[f"n_{c[4:]}"] = len(v)
        for c in dfc:
            ok = np.isfinite(g[c]) & np.isfinite(g.meas_diff)
            r[f"r_{c[6:]}"] = (np.corrcoef(g[c][ok], g.meas_diff[ok])[0, 1]
                               if ok.sum() > 10 and g[c][ok].std() > 0 else np.nan)
            r[f"ratio_{c[6:]}"] = (g[c][ok] / g.meas_diff[ok]).median() if ok.sum() else np.nan
            r[f"rc_{c[6:]}"] = centred_r(g, c)
            b, o, npairs = balanced_direction(g, c[6:])
            r[f"bal_{c[6:]}"] = b
            r[f"balpairs_{c[6:]}"] = npairs
            r[f"constacc_{c[6:]}"] = o
        rec.append(r)
    out = pd.DataFrame(rec)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.4f")
    # per pair (for cell-pair level: which contrasts are learned)
    pp = df.groupby(["feature", "a", "b", "kind"])[okc].mean()
    pp["n"] = df.groupby(["feature", "a", "b", "kind"]).size()
    pp.reset_index().to_csv(TABLES / f"{name}_pairs.csv", index=False,
                            float_format="%.4f")
    show = [c for c in out.columns if c.startswith(("acc_", "bal_", "constacc_", "rc_"))]
    log("[features]\n" + out[["feature", "kind", "n"] + show].round(3).to_string())


# ---------------------------------------------------------------- showcase
def feature_value(feature: str, M: np.ndarray, i: int, j: int, extra=None) -> float:
    if feature == "loop":
        return loop_strength(M, i, j)
    if feature == "boundary":
        return boundary_strength(insulation(M), i)
    if feature == "compartment":
        flip, sign = extra
        return flip_score(M, flip, sign)
    return np.nan


def overlaps(region: str, used: set) -> bool:
    c, se = region.split(":")
    s, e = map(int, se.split("-"))
    return any(c == c2 and s < e2 and s2 < e for c2, s2, e2 in used)


def use(region: str, used: set) -> None:
    c, se = region.split(":")
    used.add((c, *map(int, se.split("-"))))


def pick(df: pd.DataFrame, used: set, n: int, by: str = "meas_diff") -> list[dict]:
    out = []
    for r in df.sort_values(by, ascending=False).itertuples(index=False):
        if overlaps(r.region, used):
            continue
        use(r.region, used)
        out.append(r._asdict())
        if len(out) == n:
            break
    return out


def stage_showcase() -> None:
    """Showcase regions, selected on the measured maps only (effect size of the
    measured difference, replicated where replicates exist): per feature type
    the largest differences among training pairs, pairs with a held-out cell
    and AML pairs, the U-937 WT/KO contrast with the largest loop change, and
    the regions with the largest share of between-cell variance."""
    df = pd.read_csv(OUT / "features_diff.csv.gz")
    for c in ("ok__rep1", "ok__rep2"):
        if c not in df:
            df[c] = np.nan
    # a measured difference that an independent replicate pair contradicts is
    # not showcased
    rep_bad = (df.ok__rep1 == 0) | (df.ok__rep2 == 0)
    df = df[~rep_bad]
    used: set = set()
    picks = []
    # (feature, pair kinds, number, a cell the pair must contain); the AML
    # entries name the cell so MOLM-13 and U-937 both appear, and the STAG2
    # knockout enters only through its wild-type contrast
    aml = ("train-aml", "heldout-aml", "aml-aml")
    plan = [("loop", ("train-train",), 2, None),
            ("loop", ("train-heldout", "heldout-heldout"), 1, None),
            ("loop", aml, 1, "molm13"), ("loop", ("u937_wt_vs_ko",), 1, None),
            ("boundary", ("train-train",), 2, None),
            ("boundary", ("train-heldout", "heldout-heldout"), 1, None),
            ("boundary", aml, 1, "u937"),
            ("compartment", ("train-train",), 2, None),
            ("compartment", ("train-heldout", "heldout-heldout"), 1, None),
            ("compartment", aml, 1, "molm13")]
    for feat, kinds, n, need in plan:
        sub = df[(df.feature == feat) & df.kind.isin(kinds)]
        if "u937_wt_vs_ko" not in kinds:
            sub = sub[(sub.a != "u937_ko") & (sub.b != "u937_ko")]
        if need:
            sub = sub[(sub.a == need) | (sub.b == need)]
        if feat == "compartment":
            sub = sub[sub.ok__meas_vs_e1 == 1]
        for p in pick(sub, used, n):
            p["why"] = f"{feat} ({'/'.join(kinds)}{', ' + LABEL[need] if need else ''})"
            picks.append(p)
    # between-cell variance share (runs/worst_regions/features_region.csv)
    fr = pd.read_csv(RUNS / "worst_regions/features_region.csv")
    dm = pd.read_csv(OUT / "diffmap_region.csv.gz")
    dm = dm[(dm.res_kb == 25) & (dm.kind == "train-train")]
    for r in fr.sort_values("between_cell_var_share", ascending=False).itertuples():
        if overlaps(r.region, used):
            continue
        g = dm[dm.region == r.region].sort_values("meas_sd", ascending=False)
        if g.empty:
            continue
        top = g.iloc[0]
        use(r.region, used)
        picks.append({"region": r.region, "feature": "map", "a": top.a, "b": top.b,
                      "kind": top.kind, "res_kb": 25, "i": -1, "j": -1,
                      "meas_diff": top.meas_sd,
                      "why": f"between-cell variance share {r.between_cell_var_share:.2f}"})
        if sum(p["feature"] == "map" for p in picks) == 2:
            break
    sc = pd.DataFrame(picks)
    keep = [c for c in ["region", "feature", "why", "kind", "a", "b", "res_kb", "i", "j",
                        "meas_a", "meas_b", "meas_diff", "e1_diff"] if c in sc]
    sc = sc[keep + [c for c in sc.columns if c.startswith(("diff__", "ok__"))]]
    sc.insert(0, "id", [f"S{k + 1:02d}" for k in range(len(sc))])
    TABLES.mkdir(parents=True, exist_ok=True)
    sc.to_csv(TABLES / "showcase.csv", index=False, float_format="%.4f")
    log("[showcase]\n" + sc[["id", "region", "feature", "kind", "a", "b", "meas_diff"]]
        .to_string(index=False))


# ----------------------------------------------------------------- figures
FIGS = OUT / "figures"
PANEL = 80              # bins per map panel at every resolution
SRC_LABEL = {"meas": "measured", "pred__all12refres": "all12refres",
             "pred__all12honest": "all12honest (no reference)",
             "pred__all12refres_drop": "all12refres_drop",
             "pred__all12honest_drop": "all12honest_drop", "cons": "consensus read",
             "pairref__all12refres": "all12refres, pair-common reference",
             "pairref__all12refres_drop": "all12refres_drop, pair-common reference"}
SRC_COLOR = {"pred__all12refres": "#2a78d6", "pred__all12honest": "#eb6834",
             "pred__all12refres_drop": "#1baf7a", "pred__all12honest_drop": "#a05ad0",
             "cons": "#52514e", "pairref__all12refres": "#0b0b0b",
             "pairref__all12refres_drop": "#7a7a7a"}


def per_distance_r(P: np.ndarray, T: np.ndarray, dmin: int = 2) -> np.ndarray:
    n = T.shape[0]
    out = np.full(n, np.nan)
    for d in range(dmin, n):
        a, b = np.diagonal(P, d), np.diagonal(T, d)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 10 and a[ok].std() > 0 and b[ok].std() > 0:
            out[d] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def region_r(P: np.ndarray, T: np.ndarray) -> float:
    iu = np.triu_indices(T.shape[0], 2)
    return corr(P[iu], T[iu])


def row_cells(feat: str, a: str, b: str, vals: dict[str, float], n_extra: int = 3) -> list[str]:
    """Contrast cells first, then the other cells at both extremes of the
    measured feature value (the highest, the lowest, the next highest)."""
    rest = sorted([c for c in vals if c not in (a, b) and np.isfinite(vals[c])],
                  key=lambda c: -vals[c])
    extra = []
    while rest and len(extra) < n_extra:
        extra.append(rest.pop(0))
        if rest and len(extra) < n_extra:
            extra.append(rest.pop(-1))
    return [a, b] + extra


# 25 kb panels: the model's native 80-bin (2 Mb) window centred on the test
# region; predictions over these windows (stage `wide`) serve the figures only
def wide_span(region: str, res: int = 25000) -> tuple[str, int, int]:
    c, se = region.split(":")
    s, e = map(int, se.split("-"))
    centre = (s + e) // 2 // res
    ws = (centre - PANEL // 2) * res
    return c, ws, ws + PANEL * res


def figure_sets() -> list[tuple]:
    """(id, feature, res_kb, region, A, B, i, j, why) of every figure."""
    out = []
    f = TABLES / "showcase.csv"
    if f.exists():
        for r in pd.read_csv(f).itertuples():
            out.append((r.id, r.feature, int(r.res_kb), r.region, r.a, r.b, int(r.i), int(r.j),
                        r.why))
    for k, (feat, res_kb, reg, a, b, i, j) in enumerate(REVIEW_SET):
        out.append((f"R{k:02d}", feat, res_kb, reg, a, b, i, j, "review set"))
    return out


def wide_dir() -> Path:
    return OUT / "pred_wide"


def wide_regions_file() -> Path:
    """TSV of the 2 Mb windows of every 25 kb figure; the prediction folder
    is keyed by its content, so a changed set re-predicts."""
    import hashlib
    spans = sorted({wide_span(f[3]) for f in figure_sets() if f[2] == 25})
    txt = "".join(f"{c}\t{s}\t{e}\n" for c, s, e in spans)
    h = hashlib.sha1(txt.encode()).hexdigest()[:10]
    p = wide_dir() / h / "regions.tsv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    return p


def wide_pred_path(model: str, cell: str, pair: tuple[str, str] | None = None) -> Path:
    d = wide_regions_file().parent / f"{model}_25kb"
    if pair is not None and pair_consensus(*pair, 25000) != consensus_path(cell, 25000):
        d = d / f"pair_{'__'.join(sorted(pair))}"
    return d / f"pred_{cell}.npz"


def stage_wide(jobs: int, threads: int) -> None:
    res = 25000
    reg = wide_regions_file()
    todo = []
    for m in models_at(res):
        for c in CELLS:
            todo.append((m, c, None))
        if MODEL_REF[m]:
            for f in figure_sets():
                if f[2] == 25 and pair_supported(f[4], f[5]):
                    for c in (f[4], f[5]):
                        todo.append((m, c, (f[4], f[5])))
    todo = list(dict.fromkeys(t for t in todo if not wide_pred_path(*t).exists()))

    def one(m, c, pair):
        out = wide_pred_path(m, c, pair)
        refp = pair_consensus(*pair, res) if pair else consensus_path(c, res)
        if pair:
            build_pair_consensus(*pair, res)
        run_predict(predict_cmd(m, res, c, refp, out.parent, regions=reg), out, threads,
                    f"wide {m} {c}" + (f" pair {pair}" if pair else ""))
    log(f"[wide] {len(todo)} predictions of the 2 Mb windows")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(one, *t) for t in todo]:
            f.result()


class Panels:
    """80-bin display maps of one figure. 10 kb: an 800 kb crop of the test
    region centred on the feature (every pixel held out). 25 kb: the 2 Mb
    window centred on the test region, whose 1 Mb is framed; metrics always
    use the test region only (the cache)."""

    def __init__(self, res: int, region: str, i: int, j: int, local: Path):
        self.res, self.region = res, region
        regions = regions_df()
        self.ri = int(np.where(regions.region == region)[0][0])
        r = regions.iloc[self.ri]
        n = int(np.isfinite(np.asarray(load(res, "meas__gm12878")[self.ri])).any(1).sum())
        self.n_region = n
        self.local = local
        if res == 25000:
            self.chrom, self.ws, self.we = wide_span(region, res)
            self.shift = int(r.start // res - self.ws // res)   # region bin 0 in the panel
            self.frame = (self.shift, int(self.shift + n))
        else:
            mid = (i + j) // 2
            off = int(min(max(mid - PANEL // 2, 0), max(n - PANEL, 0)))
            self.shift, self.frame = -off, None
        self._wide: dict = {}

    def crop(self, M: np.ndarray) -> np.ndarray:
        o = -self.shift
        out = np.full((PANEL, PANEL), np.nan, np.float32)
        sub = M[o:o + PANEL, o:o + PANEL]
        out[:sub.shape[0], :sub.shape[1]] = sub
        return out

    def _wide_cool(self, path: Path) -> np.ndarray:
        import cooler
        from hicmultiome.genome import match_name
        clr = cooler.Cooler(str(local_copy(path, self.local)))
        nm = match_name(self.chrom, list(clr.chromnames))
        M = np.asarray(clr.matrix(balance=False).fetch(f"{nm}:{self.ws}-{self.we}"), float)
        return np.where(M == 0, np.nan, M)[:PANEL, :PANEL]

    def _wide_npz(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        key = f"{self.chrom}:{self.ws}-{self.we}"
        z = np.load(path)
        if key not in z.files:
            return None
        P = z[key].astype(float)
        live = np.abs(np.nan_to_num(P)).sum(1) > 0
        return np.where(live[:, None] & live[None, :], P, np.nan)[:PANEL, :PANEL]

    def get(self, src: str, cell: str, pair: tuple | None = None) -> np.ndarray:
        """Display map of source `src` (meas, cons, pred__<m>, pairref__<m>)."""
        nan = np.full((PANEL, PANEL), np.nan)
        if self.res == 25000:
            if src == "meas":
                return self._wide_cool(target(cell, self.res))
            if src == "cons":
                return self._wide_cool(consensus_path(cell, self.res))
            m = src.split("__", 1)[1]
            if src.startswith("pairref__"):
                if pair is None or cell not in pair or not pair_supported(*pair):
                    return nan
                P = self._wide_npz(wide_pred_path(m, cell, pair))
            else:
                P = self._wide_npz(wide_pred_path(m, cell))
            return nan if P is None else P
        if src.startswith("pairref__"):
            if pair is None or cell not in pair:
                return nan
            pm = pairref_maps(self.res, *pair).get(src)
            if pm is None:
                return nan
            return self.crop(np.asarray((pm[0] if cell == pair[0] else pm[1])[self.ri]))
        arr = load(self.res, f"{src}__{cell}")
        return nan if arr is None else self.crop(np.asarray(arr[self.ri]))

    def heldout(self, src: str, cell: str, pair: tuple | None = None) -> np.ndarray | None:
        """The test-region map (cache) for metrics."""
        if src.startswith("pairref__"):
            if pair is None or cell not in pair:
                return None
            pm = pairref_maps(self.res, *pair).get(src)
            return None if pm is None else np.asarray(
                (pm[0] if cell == pair[0] else pm[1])[self.ri])
        arr = load(self.res, f"{src}__{cell}")
        return None if arr is None else np.asarray(arr[self.ri])


def draw_marks(ax, feature: str, i: int, j: int, pn: Panels, flip=None) -> None:
    from matplotlib.patches import Circle, Rectangle
    s = pn.shift
    if pn.frame is not None:
        a, b = pn.frame
        ax.add_patch(Rectangle((a, a), b - a, b - a, fill=False, ec="#333", lw=0.6, ls="--"))
    if feature == "loop":
        ax.add_patch(Circle((j + s + 0.5, i + s + 0.5), 3.5, fill=False, ec="k", lw=0.8))
    elif feature == "boundary":
        b = i + s + 0.5
        ax.plot([b, b], [b - 8, b], color="k", lw=0.8)
        ax.plot([b, b + 8], [b, b], color="k", lw=0.8)
    elif feature == "compartment":
        f = np.where(flip)[0] + s if flip is not None else np.array([i + s, j + s])
        ax.plot([f.min(), f.max() + 1], [PANEL - 0.4, PANEL - 0.4], color="k", lw=2.5,
                solid_capstyle="butt")
    ax.set_xlim(0, PANEL)
    ax.set_ylim(PANEL, 0)


def flip_extra(feature, res, region, a, b, e1):
    if feature != "compartment" or a not in e1 or b not in e1:
        return None
    c, se = region.split(":")
    st = int(se.split("-")[0])
    ea = e1_region(e1[a], c, st, nbins(res), res)
    eb = e1_region(e1[b], c, st, nbins(res), res)
    flip = (ea > 0.5) & (eb < -0.5)
    sign = np.where((ea > 0.5) & (eb > 0.5), 1, np.where((ea < -0.5) & (eb < -0.5), -1, 0))
    return flip, sign


CAPTION_25 = ("dashed frame: the held-out 1 Mb test region; pixels outside the frame "
              "may overlap training windows. Metrics use the framed region only.")
CAPTION_10 = "800 kb crop inside the held-out test region; every pixel is held out."


def fig_showcase(fid, feature, res_kb, region, a, b, i, j, why, local: Path, e1) -> None:
    """Rows = cells (A, B, then cells at both extremes of the measured
    feature), columns = measured, every model, consensus read, pair-common
    reference, per-distance Pearson; last row A - B for every source."""
    import matplotlib.pyplot as plt
    res = res_kb * 1000
    pn = Panels(res, region, i, j, local)
    pair = (a, b)
    srcs = sources_of(res) + [f"pairref__{m}" for m in models_at(res) if MODEL_REF[m]
                              and pair_supported(a, b)]
    cols = ["meas"] + srcs
    extra = flip_extra(feature, res, region, a, b, e1)
    held = {c: pn.heldout("meas", c) for c in CELLS}
    held = {c: v for c, v in held.items() if v is not None}
    if feature == "map":
        iu = np.triu_indices(nbins(res), 2)
        vals = {c: -np.nanstd((held[c] - held[a])[iu]) if c != a else 0.0 for c in held}
    else:
        vals = {c: feature_value(feature, held[c], i, j, extra) for c in held}
    rows = row_cells(feature, a, b, vals, 2 if feature == "map" else 3)
    disp = {(src, c): pn.get(src, c, pair) for src in cols for c in rows}
    hmaps = {(src, c): pn.heldout(src, c, pair) for src in srcs for c in rows}
    vmax = float(np.nanpercentile(np.abs(np.stack([disp[("meas", c)] for c in rows])), 98))
    dA = {src: disp[(src, a)] - disp[(src, b)] for src in cols}
    dmax = float(np.nanpercentile(np.abs(dA["meas"]), 98))
    rr, pdr, fv = {}, {}, {}
    for (src, c), H in hmaps.items():
        if H is None or not np.isfinite(H).any():
            continue
        rr[(src, c)] = region_r(H, held[c])
        pdr[(src, c)] = per_distance_r(H, held[c])
    for src in cols:
        for c in rows:
            H = held[c] if src == "meas" else hmaps.get((src, c))
            if H is not None and feature != "map":
                fv[(src, c)] = feature_value(feature, H, i, j, extra)
    rd = {}
    iu = np.triu_indices(nbins(res), 2)
    dm = (held[a] - held[b])[iu]
    for src in srcs:
        ha, hb = hmaps.get((src, a)), hmaps.get((src, b))
        if ha is not None and hb is not None:
            rd[src] = corr((ha - hb)[iu], dm)
    nr, nc = len(rows) + 1, len(cols) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(2.2 * nc, 2.3 * nr), squeeze=False)
    for r_, c in enumerate(rows):
        for k, src in enumerate(cols):
            ax = axes[r_, k]
            M = disp[(src, c)]
            ax.set_xticks([]); ax.set_yticks([])
            if not np.isfinite(M).any():
                ax.axis("off")
                continue
            ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
                      extent=(0, PANEL, PANEL, 0))
            sub = []
            if (src, c) in rr and np.isfinite(rr[(src, c)]):
                sub.append(f"r={rr[(src, c)]:.2f}")
            if np.isfinite(fv.get((src, c), np.nan)):
                sub.append(f"f={fv[(src, c)]:.2f}")
            ax.set_title((SRC_LABEL[src] + "\n" if r_ == 0 else "") + "  ".join(sub), fontsize=7)
            if k == 0:
                ax.set_ylabel(LABEL[c] + (" (A)" if c == a else " (B)" if c == b else ""),
                              fontsize=8)
            draw_marks(ax, feature, i, j, pn, extra[0] if extra else None)
        ax = axes[r_, -1]
        for src in srcs:
            if (src, c) in pdr:
                ax.plot(np.arange(len(pdr[(src, c)])) * res / 1000, pdr[(src, c)],
                        color=SRC_COLOR[src], lw=1, label=SRC_LABEL[src])
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_ylim(-0.6, 1.0)
        ax.tick_params(labelsize=6)
        if r_ == 0:
            ax.set_title("per-distance Pearson\n(test region)", fontsize=7)
            ax.legend(fontsize=5, loc="lower left", frameon=False)
        if r_ == len(rows) - 1:
            ax.set_xlabel("distance (kb)", fontsize=7)
    for k, src in enumerate(cols):
        ax = axes[-1, k]
        ax.set_xticks([]); ax.set_yticks([])
        if not np.isfinite(dA[src]).any():
            ax.axis("off")
            continue
        ax.imshow(dA[src], cmap="PuOr_r", vmin=-dmax, vmax=dmax, interpolation="nearest",
                  extent=(0, PANEL, PANEL, 0))
        ttl = f"{SRC_LABEL[src]}: A - B"
        if src in rd:
            ttl += f"\nr={rd[src]:.2f} with measured"
        ax.set_title(ttl, fontsize=7)
        if k == 0:
            ax.set_ylabel(f"{LABEL[a]} - {LABEL[b]}", fontsize=8)
        draw_marks(ax, feature, i, j, pn, extra[0] if extra else None)
    axes[-1, -1].axis("off")
    fig.suptitle(f"{fid} {region} ({res_kb} kb, {PANEL} bins per panel): {why}; A = {LABEL[a]}, "
                 f"B = {LABEL[b]}; map scale +-{vmax:.2f}, difference scale +-{dmax:.2f} "
                 f"log O/E\n{CAPTION_25 if res == 25000 else CAPTION_10}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = FIGS / f"{fid}_{feature}_{region.replace(':', '_')}"
    fig.savefig(f"{stem}.png", dpi=130)
    plt.close(fig)
    np.savez_compressed(
        f"{stem}.npz",
        **{f"{src}__{c}": disp[(src, c)].astype(np.float32) for (src, c) in disp},
        **{f"diff__{src}": dA[src].astype(np.float32) for src in cols},
        **{f"perdist__{src}__{c}": v.astype(np.float32) for (src, c), v in pdr.items()},
        meta=json.dumps({"id": fid, "region": region, "res": res, "feature": feature,
                         "why": why, "a": a, "b": b, "rows": rows, "sources": cols,
                         "panel_bins": PANEL, "panel_shift": int(pn.shift),
                         "test_frame": pn.frame, "i": i, "j": j, "vmax": vmax, "dmax": dmax,
                         "region_r": {f"{k[0]}__{k[1]}": v for k, v in rr.items()},
                         "diff_r": rd,
                         "feature_value": {f"{k[0]}__{k[1]}": v for k, v in fv.items()}}))
    log(f"[figures] {stem.name}.png")


def fig_review80(k, feature, res_kb, region, a, b, i, j, local: Path, e1) -> None:
    """The review set in the layout of runs/cellspecific/review/predplot.py:
    rows = sources (measured, with consensus, without, consensus read,
    pair-common reference, dropout models when trained), columns = the
    context cells and A - B; 80 bins per panel."""
    import matplotlib.pyplot as plt
    res = res_kb * 1000
    pn = Panels(res, region, i, j, local)
    pair = (a, b)
    rows = ["meas"] + [s for s in sources_of(res) if s != "cons"] + ["cons"]
    rows += [f"pairref__{m}" for m in models_at(res) if MODEL_REF[m]]
    extra = flip_extra(feature, res, region, a, b, e1)
    M = {(src, c): pn.get(src, c, pair) for src in rows for c in REVIEW_CTX}
    lim = float(np.nanpercentile(np.abs(np.stack([M[("meas", a)], M[("meas", b)]])), 98))
    held = {c: pn.heldout("meas", c) for c in REVIEW_CTX}
    fig, ax = plt.subplots(len(rows), len(REVIEW_CTX) + 1,
                           figsize=(1.9 * (len(REVIEW_CTX) + 1), 2.0 * len(rows)), squeeze=False)
    iu = np.triu_indices(nbins(res), 2)
    for r_, src in enumerate(rows):
        for q, c in enumerate(REVIEW_CTX):
            x = ax[r_, q]
            x.set_xticks([]); x.set_yticks([])
            m = M[(src, c)]
            if not np.isfinite(m).any():
                if src.startswith("pairref"):
                    x.axis("off")
                else:
                    x.text(0.5, 0.5, "n/a", ha="center", transform=x.transAxes, fontsize=7)
            else:
                x.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                         extent=(0, PANEL, PANEL, 0))
                draw_marks(x, feature, i, j, pn, extra[0] if extra else None)
                H = pn.heldout(src, c, pair) if src != "meas" else None
                if H is not None and held.get(c) is not None:
                    x.set_title(("" if r_ else LABEL[c] + (" (A)" if c == a else " (B)"
                                                           if c == b else "") + "\n")
                                + f"r={region_r(H, held[c]):.2f}", fontsize=6)
            if r_ == 0:
                x.set_title(LABEL[c] + (" (A)" if c == a else " (B)" if c == b else ""),
                            fontsize=7, fontweight="bold" if c in (a, b) else "normal")
            if q == 0:
                x.set_ylabel(SRC_LABEL[src].replace(", ", ",\n"), fontsize=7)
        x = ax[r_, -1]
        x.set_xticks([]); x.set_yticks([])
        A, B = M[(src, a)], M[(src, b)]
        if not (np.isfinite(A).any() and np.isfinite(B).any()):
            x.axis("off")
        if np.isfinite(A).any() and np.isfinite(B).any():
            x.imshow(A - B, cmap="PuOr_r", vmin=-lim, vmax=lim, interpolation="nearest",
                     extent=(0, PANEL, PANEL, 0))
            draw_marks(x, feature, i, j, pn, extra[0] if extra else None)
            if src != "meas":
                ha, hb = pn.heldout(src, a, pair), pn.heldout(src, b, pair)
                if ha is not None and hb is not None and held.get(a) is not None:
                    x.set_title(("A - B\n" if r_ == 0 else "")
                                + f"r={corr((ha - hb)[iu], (held[a] - held[b])[iu]):.2f}",
                                fontsize=6)
        if r_ == 0:
            x.set_title("A - B", fontsize=7, fontweight="bold")
    fig.suptitle(f"{feature} {res_kb} kb {region}  A={LABEL[a]}  B={LABEL[b]}  "
                 f"({PANEL} bins per panel; one colour scale per region; r = Pearson on the "
                 f"test region)\n{CAPTION_25 if res == 25000 else CAPTION_10}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = FIGS / f"showcase80_{k:02d}_{feature}_{region.replace(':', '_')}"
    fig.savefig(f"{stem}.png", dpi=110)
    plt.close(fig)
    np.savez_compressed(f"{stem}.npz", **{f"{s}__{c}": v.astype(np.float32)
                                          for (s, c), v in M.items()},
                        meta=json.dumps({"region": region, "res": res, "feature": feature,
                                         "a": a, "b": b, "rows": rows, "cells": REVIEW_CTX,
                                         "panel_shift": int(pn.shift), "test_frame": pn.frame,
                                         "vmax": lim}))
    log(f"[figures] {stem.name}.png")


def stage_figures(local: Path, which: str = "all") -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    e1 = load_e1()
    for f in figure_sets():
        fid, feature, res_kb, region, a, b, i, j, why = f
        if fid.startswith("R"):
            if which in ("all", "review"):
                fig_review80(int(fid[1:]), feature, res_kb, region, a, b, i, j, local, e1)
        elif which in ("all", "showcase"):
            fig_showcase(fid, feature, res_kb, region, a, b, i, j, why, local, e1)


# ----------------------------------------------------------------- summary
def md_table(d: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    rows = ["| " + " | ".join(map(str, d.columns)) + " |",
            "|" + "---|" * len(d.columns)]
    for r in d.itertuples(index=False):
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(rows)


def stage_summary() -> None:
    """STATUS.md: the tables of this analysis and the figure paths."""
    def tab(name, cols=None):
        f = TABLES / name
        if not f.exists():
            return "(missing)"
        d = pd.read_csv(f)
        if cols:
            d = d[[c for c in d.columns if c in cols or any(c.startswith(x) for x in cols)]]
        return md_table(d.round(3))
    keep = ["feature", "kind", "n", "acc_", "bal_", "constacc_", "rc_"]
    figs = sorted(FIGS.glob("*.png"))
    lines = ["# Cell-type-specific structure: status", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M')}. Script scripts/cellspecific.py; tables "
             "results/cellspecific; figures runs/cellspecific/figures (PNG + npz per figure).",
             f"Models: {', '.join(sorted(set(models_at(10000)) | set(models_at(25000))))}.", "",
             "## Difference maps: Pearson of pred_A - pred_B with meas_A - meas_B "
             "(median over regions, then over pairs)", "", tab("diffmap_kinds.csv"), "",
             "## Pair-common reference control (same pixels)", "",
             tab("diffmap_pairref_summary.csv"), "",
             "## Replicate-difference noise floor", "", tab("diffmap_noise.csv"), "",
             "## Feature direction (plain / balanced over pairs / pair-constant / "
             "pair-centred r)", "", tab("features_direction.csv", keep), "",
             "## Feature direction on the pair-common reference pairs", "",
             tab("features_direction_pairref.csv", keep), "",
             "## Showcase", "", tab("showcase.csv", ["id", "region", "feature", "kind", "a", "b",
                                                    "meas_diff"]), "",
             "## Figures", ""] + [f"- {f}" for f in figs]
    lines += ["", "Progress notes with intermediate numbers: runs/cellspecific/STATUS_progress.md.", ""]
    (OUT / "STATUS.md").write_text("\n".join(lines))
    log("[summary] STATUS.md written")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--local", default=str(Path.home() / ".cache/cellspecific_cools"),
                    help="local disk folder for cooler copies (the share is slow to fetch)")
    ap.add_argument("--out", default=None, help="output folder (default runs/cellspecific)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="prediction device (inference only)")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for --device cuda")
    ap.add_argument("--which", default="all", choices=["all", "showcase", "review"],
                    help="figures: the showcase set, the review set or both")
    a = ap.parse_args()
    DEVICE.update(device=a.device, gpu=a.gpu)
    if a.out:
        global OUT
        OUT = Path(a.out)
    for st in a.stage:
        if st == "split":
            stage_split()
        elif st == "predict":
            stage_predict(a.jobs, a.threads)
        elif st == "pairref":
            stage_pairref(a.jobs, a.threads)
        elif st == "calls":
            stage_calls(a.jobs, a.threads)
        elif st == "e1":
            stage_e1()
        elif st == "load":
            stage_load(Path(a.local))
        elif st == "diffmap":
            stage_diffmap()
        elif st == "features":
            stage_features()
        elif st == "showcase":
            stage_showcase()
        elif st == "figures":
            stage_figures(Path(a.local), a.which)
        elif st == "wide":
            stage_wide(a.jobs, a.threads)
        elif st == "summary":
            stage_summary()
        else:
            raise SystemExit(f"unknown stage {st}")


if __name__ == "__main__":
    main()
