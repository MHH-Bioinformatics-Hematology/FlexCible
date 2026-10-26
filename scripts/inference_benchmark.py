"""Prediction time and memory of our models and the released competitors (registry 82).

Every method predicts the whole of GM12878 chromosome 2 into a cool file, one job at a time, on one RTX 4090
(CUDA device 1) or on 8 CPU threads. Each tool runs through its own inference code; for the competitors this is
the whole-chromosome cool writer we added on their hic-prediction-workbench branches (the released code predicts
single windows only). Per job: wall time (process start to exit, including model loading and writing the cool),
peak GPU memory (nvidia-smi, summed over the process tree) and peak resident CPU memory (psutil, process tree).
"Default" uses each tool's default batch setting; "lowest" its smallest batch.

    ~/miniconda3/envs/hicdiff/bin/python scripts/inference_benchmark.py [--only REGEX]
Output: results/inference_benchmark/runs.csv (one row per job, resumable)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

import psutil

HOME = Path.home()
ENV = HOME / "miniconda3/envs"
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
MR = HM / "runs/multicell_reference"
OE = HM / "crosscell_oe"
WORK = HM / "runs/inference_benchmark"
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/inference_benchmark/runs.csv"
GPU = "1"
CHROM, CHROM_LEN = "chr2", 242193529
FIELDS = ["job", "method", "res", "device", "setting", "wall_s", "peak_gpu_mib", "peak_rss_mib",
          "exit", "out_mb", "cmd"]


def ours(res_kb: int, device: str, batch: int, setting: str):
    run = "all12refin_drop" if res_kb == 2 else f"all12refin_drop_{res_kb}kb"
    out = WORK / f"ours_{res_kb}kb_{device}_{setting}.cool"
    cmd = [str(ENV / "hicdiff/bin/python"), "-m", "hicmultiome.predict",
           "--config", str(MR / f"{run}.yaml"), "--checkpoint", str(MR / run / "checkpoint-best"),
           "--chroms", CHROM, "--tta", "--sample", "gm12878", "--batch-size", str(batch),
           "--reference", str(OE / f"consensus_loo_gm12878_proper_{res_kb}kb.cool"), "--out", str(out)]
    return dict(job=f"ours_{res_kb}kb_{device}_{setting}", method="ours (reference input)", res=f"{res_kb} kb",
                device=device, setting=setting, cmd=cmd, cwd=REPO, out=out)


def corigami(setting: str, nb: int):
    sizes = WORK / "chr2.sizes"
    sizes.write_text(f"{CHROM}\t{CHROM_LEN}\n")
    d = HOME / "src/C.Origami/corigami_data"
    out = WORK / f"corigami_{setting}"
    cmd = [str(ENV / "corigami/bin/python"), "corigami/inference/prediction.py", "--out", str(out),
           "--celltype", "gm12878", "--chr", CHROM, "--start", "0", "--full-chr", "--chr-len", str(sizes),
           "--gpu", "0", "--number-of-batches", str(nb),
           "--model", str(d / "model_weights/corigami_base.ckpt"), "--seq", str(d / "data/hg38/dna_sequence"),
           "--ctcf", str(d / "data/hg38/gm12878/genomic_features/ctcf_log2fc.bw"),
           "--atac", str(d / "data/hg38/gm12878/genomic_features/atac.bw")]
    return dict(job=f"corigami_gpu_{setting}", method="C.Origami", res="8192 bp", device="gpu", setting=setting,
                cmd=cmd, cwd=HOME / "src/C.Origami/src", out=out / "gm12878" / f"{CHROM}.cool")


def epiphany(setting: str, batch: int):
    bw = HOME / "src/epiphany/bigwig"
    out = WORK / f"epiphany_{setting}.cool"
    cmd = [str(ENV / "epiphany/bin/python"), "epiphany/predict_cool.py",
           "--weights", "/mnt/processed/wolffjoa/galaxy-hic-models/epiphany/pretrained_10kb.pt_model",
           "--tracks", *[str(bw / f"GM12878_{t}_merge.bigWig") for t in
                         ("DNaseI", "H3K27ac", "H3K4me3", "H3K27me3", "CTCF")],
           "--chromosomes", CHROM, "--batch", str(batch), "--out", str(out)]
    return dict(job=f"epiphany_gpu_{setting}", method="Epiphany", res="10 kb", device="gpu", setting=setting,
                cmd=cmd, cwd=HOME / "src/epiphany-workbench", out=out)


def akita(setting: str, batch: int):
    td = HOME / "src/tools-iuc/tools/akita/tool-data"
    out = WORK / f"akita_{setting}.cool"
    cmd = [str(ENV / "basenji/bin/python"), "bin/akita_predict_cool.py", "--model", str(td / "model_best.h5"),
           "--params", str(td / "params.json"), "--fasta", str(HOME / "data/hic-multiome/genome/hg38.fa"),
           "--target-index", "2", "--chromosomes", CHROM, "--batch-size", str(batch), "--out", str(out)]
    # TensorFlow reserves the whole card unless memory growth is on; with growth the allocator high-water mark is
    # what nvidia-smi reports
    return dict(job=f"akita_gpu_{setting}", method="Akita", res="2048 bp", device="gpu", setting=setting,
                cmd=cmd, cwd=HOME / "src/basenji", out=out, env={"TF_FORCE_GPU_ALLOW_GROWTH": "true"})


def jobs():
    j = []
    for r in (25, 10, 5, 2):
        j.append(ours(r, "gpu", 8, "default"))
        j.append(ours(r, "gpu", 1, "lowest"))
    for r in (25, 10):
        j.append(ours(r, "cpu", 1, "lowest"))
    # Epiphany predicts fixed 2 Mb chunks one at a time; --batch only sets the progress step, so one setting
    j += [epiphany("default", 16), akita("default", 4), akita("lowest", 1),
          corigami("default", 16), corigami("lowest", 452)]
    return j


def gpu_mem(pids: set[int]) -> int:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    tot = 0
    for line in r.stdout.strip().splitlines():
        p, m = [x.strip() for x in line.split(",")]
        if p.isdigit() and int(p) in pids:
            tot += int(m)
    return tot


def run(job: dict) -> dict:
    env = dict(os.environ, PYTHONPATH=str(REPO), **job.get("env", {}))
    if job["device"] == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = GPU
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[v] = "8"
    job["out"].parent.mkdir(parents=True, exist_ok=True)
    log = open(WORK / f"{job['job']}.log", "w")
    t0 = time.time()
    p = subprocess.Popen(job["cmd"], cwd=job["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    proc = psutil.Process(p.pid)
    peak_gpu = peak_rss = 0
    while p.poll() is None:
        try:
            tree = [proc] + proc.children(recursive=True)
            peak_rss = max(peak_rss, sum(x.memory_info().rss for x in tree if x.is_running()))
            peak_gpu = max(peak_gpu, gpu_mem({x.pid for x in tree}))
        except psutil.Error:
            pass
        time.sleep(0.25)
    wall = time.time() - t0
    out_mb = job["out"].stat().st_size / 2**20 if job["out"].exists() else float("nan")
    return dict(job=job["job"], method=job["method"], res=job["res"], device=job["device"],
                setting=job["setting"], wall_s=round(wall, 1), peak_gpu_mib=peak_gpu,
                peak_rss_mib=round(peak_rss / 2**20), exit=p.returncode, out_mb=round(out_mb, 1),
                cmd=" ".join(map(str, job["cmd"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="regex of job names")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {r["job"] for r in csv.DictReader(open(OUT)) if r["exit"] == "0"}
    for job in jobs():
        if job["job"] in done or (a.only and not re.search(a.only, job["job"])):
            continue
        row = run(job)
        print(f"{row['job']}: {row['wall_s']} s, GPU {row['peak_gpu_mib']} MiB, RSS {row['peak_rss_mib']} MiB, "
              f"exit {row['exit']}", flush=True)
        new = not OUT.exists()
        with open(OUT, "a", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
