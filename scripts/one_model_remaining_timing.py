"""Prediction time and memory of the release checkpoint all12one_fullrd (registry sections 82 and 98).

Same measurement as scripts/inference_benchmark.py (registry section 82): GM12878 chromosome 2 (242 Mb) into one cool
file, one job at a time, wall time from process start to exit (model and input loading and cool writing included),
peak GPU memory (nvidia-smi over the process tree, 0.25 s) and peak resident memory (psutil), "default" batch 8
and "lowest" batch 1, test-time reverse-complement averaging, reference input (leave-one-out consensus of GM12878), on
GPU and, for 25 and 10 kb, on 8 CPU threads at batch 1. The release rows read 13 marks + Meth of GM12878 and the
cohesin fraction 1; the paper rows (dedicated reference-input models all12refin_drop, 13 marks) are measured in the
same session on the same host so that the release/paper ratio does not depend on the machine.

The GPU host is not the machine of section 82 (RTX 4090 and Ryzen 9 7950X): the absolute values of both model rows
belong to the host named in the `host` column.

    python scripts/one_model_remaining_timing.py --models release paper [--only REGEX]
Environment: HM_ROOT (data root with runs/multicell_reference, crosscell_oe, meth), TIMING_WORK (scratch folder).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
HM = Path(os.environ.get("HM_ROOT", "/mnt/processed/wolffjoa/hic-multiome"))
WORK = Path(os.environ.get("TIMING_WORK", str(HM / "runs/inference_benchmark_one_model")))
OUT = Path(os.environ.get("TIMING_OUT", str(REPO / "results/one_model/remaining/timing/runs.csv")))
MR = HM / "runs/multicell_reference"
OE = HM / "crosscell_oe"
GPU = os.environ.get("TIMING_GPU", "0")
PY = os.environ.get("HICMULTI_PY", sys.executable)
CHROM = "chr2"
FIELDS = ["job", "host", "model", "res", "device", "setting", "wall_s", "peak_gpu_mib", "peak_rss_mib", "exit",
          "out_mb", "cmd"]


def cfg_for(ck: Path, tag: str) -> Path:
    """checkpoint config with the data paths of this host (only the GM12878 sample is opened)"""
    p = WORK / f"cfg_{tag}.yaml"
    txt = ck.joinpath("config.yaml").read_text()
    txt = re.sub(r"/mnt/processed/wolffjoa/hic-multiome/crosscell_oe", str(OE), txt)
    txt = re.sub(r"/mnt/processed/wolffjoa/hic-multiome/meth", str(HM / "meth"), txt)
    p.write_text(txt)
    return p


def job(model: str, k: int, device: str, batch: int, setting: str):
    run = f"all12one_fullrd_{k}kb" if model == "release" else (f"all12refin_drop_{k}kb" if k != 2 else "all12refin_drop")
    ck = MR / run / "checkpoint-best"
    tag = f"{model}_{k}kb"
    out = WORK / f"{tag}_{device}_{setting}.cool"
    cmd = [PY, "-m", "flexcible.predict", "--config", str(cfg_for(ck, tag)), "--checkpoint", str(ck),
           "--chroms", CHROM, "--tta", "--sample", "gm12878", "--batch-size", str(batch),
           "--reference", str(OE / f"consensus_loo_gm12878_proper_{k}kb.cool"), "--out", str(out)]
    return dict(job=f"{tag}_{device}_{setting}", model=model, res=f"{k} kb", device=device, setting=setting,
                cmd=cmd, out=out)


def jobs(models):
    j = []
    for m in models:
        for r in (25, 10, 5, 2):
            j.append(job(m, r, "gpu", 8, "default"))
            j.append(job(m, r, "gpu", 1, "lowest"))
        for r in (25, 10):
            j.append(job(m, r, "cpu", 1, "lowest"))
    return j


def gpu_mem(pids):
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    tot = 0
    for line in r.stdout.strip().splitlines():
        p, m = [x.strip() for x in line.split(",")]
        if p.isdigit() and int(p) in pids:
            tot += int(m)
    return tot


def run(j):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    if j["device"] == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = GPU
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[v] = "8"
    j["out"].parent.mkdir(parents=True, exist_ok=True)
    log = open(WORK / f"{j['job']}.log", "w")
    t0 = time.time()
    p = subprocess.Popen(j["cmd"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    proc = psutil.Process(p.pid)
    pg = pr = 0
    while p.poll() is None:
        try:
            tree = [proc] + proc.children(recursive=True)
            pr = max(pr, sum(x.memory_info().rss for x in tree if x.is_running()))
            pg = max(pg, gpu_mem({x.pid for x in tree}))
        except psutil.Error:
            pass
        time.sleep(0.25)
    wall = time.time() - t0
    mb = j["out"].stat().st_size / 2**20 if j["out"].exists() else float("nan")
    return dict(job=j["job"], host=socket.gethostname(), model=j["model"], res=j["res"], device=j["device"],
                setting=j["setting"], wall_s=round(wall, 1), peak_gpu_mib=pg, peak_rss_mib=round(pr / 2**20),
                exit=p.returncode, out_mb=round(mb, 1), cmd=" ".join(map(str, j["cmd"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["release", "paper"], choices=["release", "paper"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {r["job"] for r in csv.DictReader(open(OUT)) if r["exit"] == "0"}
    for j in jobs(a.models):
        if j["job"] in done or (a.only and not re.search(a.only, j["job"])):
            continue
        for _ in range(a.repeat):
            row = run(j)
            print(f"{row['job']}: {row['wall_s']} s, GPU {row['peak_gpu_mib']} MiB, RSS {row['peak_rss_mib']} MiB, "
                  f"exit {row['exit']}", flush=True)
            new = not OUT.exists()
            with open(OUT, "a", newline="") as fh:
                w = csv.DictWriter(fh, FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            if j["out"].exists():
                j["out"].unlink()


if __name__ == "__main__":
    main()
