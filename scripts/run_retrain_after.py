"""Wait for the backward search + combo queue to release the GPUs, then retrain
the all-13 model WITH uniform-keep-count track dropout, then score the
accuracy-vs-#tracks curve. Sits at the end of the queue.

    python scripts/run_retrain_after.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIG = "configs/akita_native_trackdrop.yaml"
CKPT = "runs/akita_native_trackdrop/checkpoint-best"
GPUS = "0,1"
WAIT_FOR = ["backward_tracks.py", "queue_combos.py"]


def busy():
    for pat in WAIT_FOR:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        pids = [p for p in r.stdout.split() if p.strip()]
        if pids:
            return f"{pat} {pids}"
    return None


def main():
    print("[retrain] waiting for backward search + combo queue to finish", flush=True)
    while True:
        b = busy()
        if not b:
            break
        print(f"  still running: {b}; waiting...", flush=True)
        time.sleep(120)
    print("[retrain] GPUs free; starting track-dropout retrain", flush=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS)
    torchrun = str(Path(PY).parent / "torchrun")
    cmd = [torchrun, "--standalone", "--nproc_per_node=2",
           "-m", "flexcible.train", "--config", CONFIG]
    print("  " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    print(f"[retrain] training exit {rc}", flush=True)
    if rc != 0:
        sys.exit(rc)

    print("[retrain] scoring accuracy-vs-#tracks curve (CPU)", flush=True)
    rc = subprocess.run(
        [PY, "scripts/subset_curve.py", "--config", CONFIG,
         "--checkpoint", CKPT, "--device", "cpu"],
        cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="8")).returncode
    print(f"[retrain] subset_curve exit {rc}; done.", flush=True)


if __name__ == "__main__":
    main()
