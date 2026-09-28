"""Multi-resolution training (registry section "Multi-resolution training").

A  all12multires: ONE resolution-conditioned residual reference model on the
   12 training cells at 2, 5, 10 and 25 kb (hicmultiome.train_multires; FiLM
   from a learned resolution embedding; windows of 512 / 384 / 192 / 80
   bins as the dedicated models; one resolution per step, drawn uniformly;
   per-cell leave-one-out consensus as reference, consensus12 for held-out
   cells; honest validation per resolution). Its per-resolution views
   runs/multicell_reference/all12multires[_<k>kb] are evaluated by
   scripts/multicell_expanded.py --stage eval exactly like the dedicated
   all12refres / all12honest runs, and appear in reference_summary.csv.

Stages: precompute (the shared mc_all12ref_<k>kb caches), stage (the training
windows uncompressed as memory maps under ~/scratch/hicmx_multires_stage, so
both DDP ranks share one page cache), train (torchrun on both GPUs), views,
eval.

    python scripts/multires.py --stage precompute --res 5000 2000
    python scripts/multires.py --stage train
    python scripts/multires.py --stage views
    python scripts/multires.py --stage eval
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                      # noqa: E402

REPO, PY, TORCHRUN = mx.REPO, mx.PY, mx.TORCHRUN
OUT = mx.REF_OUT
RUN = "all12multires"
TRAIN_DIR = OUT / f"{RUN}_train"
# memory-mapped training windows on local disk (not /tmp): both DDP ranks map
# the same files instead of each decompressing the caches from the share
STAGE = Path.home() / "scratch/hicmx_multires_stage"
LOG = OUT / f"{RUN}_train.log"
A = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "k562", "hmec_deep"]
B = ["nhek", "gm23248", "mcf7", "cd14mono", "pc9", "hmec", "huvec", "huvec_deep"]
STOP_RE = re.compile(r"early stop at epoch (\d+) \(best val ([-0-9.]+), (\d+) epochs")


def sh(cmd, log: Path, env=None):
    print("[cmd]", " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as fh:
        return subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=REPO).returncode


def view(res):
    return OUT / mx.run_name(RUN, res)


def config_paths():
    paths = []
    for r in mx.MULTIRES:
        cfg = mx.run_config(mx.run_name(RUN, r), 100, 10, 500)
        p = OUT / f"{mx.run_name(RUN, r)}.yaml"
        yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
        paths.append(p)
    return paths


def precompute(res_list, workers):
    for r in res_list:
        if sh([PY, "scripts/multicell_expanded.py", "--runs", RUN, "--res", r,
               "--stage", "precompute", "--workers", workers],
              OUT / f"{RUN}_precompute_{r // 1000}kb.log"):
            raise SystemExit(f"precompute failed at {r}")


def train(max_epochs):
    paths = config_paths()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")
    t = LOG.read_text(errors="replace") if LOG.exists() else ""
    resume = "checkpoint-latest" in {p.name for p in TRAIN_DIR.glob("*")} and "done." not in t
    if "done." not in t:
        cmd = [TORCHRUN, "--standalone", "--nproc_per_node=2", "-m",
               "hicmultiome.train_multires", "--configs", *paths, "--out", TRAIN_DIR,
               "--max-epochs", max_epochs, "--stage-dir", STAGE] + (["--resume"] if resume else [])
        if sh(cmd, LOG, env):
            raise SystemExit("multi-resolution training failed")
    # continue while stopped by the epoch cap
    m = STOP_RE.findall(LOG.read_text(errors="replace"))
    if m and int(m[-1][2]) < 10:
        raise SystemExit(f"stopped by the epoch cap at {m[-1][0]}; rerun with a larger "
                         "--max-epochs (resumes from checkpoint-latest)")


def stage():
    STAGE.mkdir(parents=True, exist_ok=True)
    if sh([PY, "-m", "hicmultiome.train_multires", "--configs", *config_paths(),
           "--out", TRAIN_DIR, "--stage-dir", STAGE, "--stage-only"],
          OUT / f"{RUN}_stage.log"):
        raise SystemExit("staging failed")


def staged() -> bool:
    return all((STAGE / f"res_{r}" / "meta.json").exists() for r in mx.MULTIRES)


def views():
    from hicmultiome.train_multires import write_views
    write_views(TRAIN_DIR, {r: view(r) for r in mx.MULTIRES})


def evaluate():
    procs = []
    for gpu, cells in (("0", A), ("1", B)):
        cmd = []
        for r in mx.MULTIRES:
            cmd.append(f"{PY} scripts/multicell_expanded.py --runs {RUN} --res {r} "
                       f"--stage eval --gpus {gpu} --cells {' '.join(cells)}")
        procs.append(subprocess.Popen(" ; ".join(cmd), shell=True, cwd=REPO,
                                      stdout=open(OUT / f"{RUN}_eval_gpu{gpu}.log", "a"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    sh([PY, "scripts/multicell_expanded.py", "--stage", "refsummary"], OUT / f"{RUN}_eval.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["precompute", "stage", "train", "views", "eval"],
                    required=True)
    ap.add_argument("--res", nargs="*", type=int, default=mx.MULTIRES)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=500)
    a = ap.parse_args()
    if a.stage == "precompute":
        precompute(a.res, a.workers)
    elif a.stage == "stage":
        stage()
    elif a.stage == "train":
        train(a.max_epochs)
    elif a.stage == "views":
        views()
    else:
        evaluate()
    print("MULTIRES_DONE", flush=True)


if __name__ == "__main__":
    main()
