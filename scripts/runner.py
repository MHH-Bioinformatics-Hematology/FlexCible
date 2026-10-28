"""Dependency-aware runner that keeps the GPUs busy.

The serial scripts leave both GPUs idle through every precompute (CPU-only)
and every evaluation (numpy/cooler). This scheduler splits each experiment
into typed tasks and overlaps them:

    precompute (CPU) -> train (GPU) -> predict (GPU) -> evaluate (CPU)

with two GPU slots (one experiment per GPU, pinned via CUDA_VISIBLE_DEVICES)
and several CPU slots. While experiment A trains on GPU0, experiment B trains
on GPU1 and experiments C..E precompute on CPU, so a GPU only idles if no
trained-and-ready job exists at all.

Note on comparability: every training task gets exactly one GPU and the same
max_minutes, so time-boxed runs stay directly comparable to each other (they
are NOT comparable to earlier 2-GPU DDP runs at the same wall-clock budget).

Usage:
    python scripts/runner.py --spec scripts/queue_now.json
    python scripts/runner.py --spec ... --gpu-slots 2 --cpu-slots 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("HICMULTI_PY", str(Path.home() / "miniconda3/envs/hicdiff/bin/python"))


@dataclass
class Task:
    tid: str
    cmd: list[str]
    resource: str                     # "gpu" | "cpu"
    deps: list[str] = field(default_factory=list)
    cwd: str = str(ROOT)
    log: str = ""
    skip_if: str = ""                 # path that, if non-empty, skips the task


def build_tasks(spec: dict) -> list[Task]:
    tasks: list[Task] = []
    seen_pre: dict[str, str] = {}     # cache_dir -> task id (shared precompute)
    logdir = spec.get("logdir", "runs/runner")

    for job in spec["jobs"]:
        name = job["name"]
        cwd = str((ROOT / job.get("cwd", ".")).resolve())
        mod = job.get("module", "flexcible")
        cfg = job["config"]
        cache = job.get("cache_dir") or f"{cwd}:{cfg}"
        log = f"{logdir}/{name}"

        pre_id = seen_pre.get(cache)
        if pre_id is None:
            pre_id = f"{name}.pre"
            tasks.append(Task(
                pre_id, [PY, "-m", f"{mod}.precompute", "--config", cfg],
                "cpu", [], cwd, f"{log}.precompute.log",
                skip_if=job.get("cache_manifest", "")))
            seen_pre[cache] = pre_id

        tr_id = f"{name}.train"
        tasks.append(Task(
            tr_id, [PY, "-m", f"{mod}.train", "--config", cfg],
            "gpu", [pre_id], cwd, f"{log}.train.log"))

        if job.get("train_only"):
            # throughput benchmarks: prediction at fine resolution costs more
            # than the training window being measured
            continue

        pr_id = f"{name}.predict"
        pred_cmd = [PY, "-m", f"{mod}.predict", "--config", cfg,
                    "--chroms", "test", "--out", job["pred_out"]]
        pred_cmd += job.get("predict_extra", [])
        tasks.append(Task(pr_id, pred_cmd, "gpu", [tr_id], cwd,
                          f"{log}.predict.log"))

        ev_cmd = [PY, "-m", f"{mod}.evaluate", "--config", cfg,
                  "--pred", job["pred_out"], "--chroms", "test",
                  "--out", job["eval_out"]]
        ev_cmd += job.get("evaluate_extra", [])
        tasks.append(Task(f"{name}.eval", ev_cmd, "cpu", [pr_id], cwd,
                          f"{log}.evaluate.log"))

    for extra in spec.get("post", []):
        tasks.append(Task(
            extra["name"], extra["cmd"], extra.get("resource", "gpu"),
            extra.get("deps", []),
            str((ROOT / extra.get("cwd", ".")).resolve()),
            f"{logdir}/{extra['name']}.log"))
    return tasks


class GpuSampler(threading.Thread):
    """Background nvidia-smi sampler so the run reports real utilisation."""

    def __init__(self, period=5.0):
        super().__init__(daemon=True)
        self.period = period
        self.samples: list[list[int]] = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.period):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                self.samples.append([int(x) for x in out.split()])
            except Exception:
                pass

    def stop(self):
        self._stop.set()

    def report(self) -> str:
        if not self.samples:
            return "no GPU samples"
        import statistics

        cols = list(zip(*self.samples))
        lines = []
        for i, col in enumerate(cols):
            busy = sum(1 for v in col if v >= 50) / len(col) * 100
            lines.append(f"GPU{i}: mean {statistics.mean(col):.0f}%  "
                         f"busy(>=50%) {busy:.0f}% of {len(col)} samples")
        return "\n".join(lines)


def run(spec: dict, gpu_slots: int, cpu_slots: int,
        gpu_ids: list[int] | None = None):
    tasks = {t.tid: t for t in build_tasks(spec)}
    pending = dict(tasks)
    done: set[str] = set()
    failed: set[str] = set()
    running: dict[str, tuple] = {}
    free_gpu = list(gpu_ids) if gpu_ids else list(range(gpu_slots))
    free_cpu = cpu_slots
    logdir = ROOT / spec.get("logdir", "runs/runner")
    logdir.mkdir(parents=True, exist_ok=True)
    sampler = GpuSampler()
    sampler.start()
    t_start = time.time()
    timeline: list[dict] = []

    def launch(t: Task):
        nonlocal free_cpu
        env = dict(os.environ)
        slot = None
        if t.resource == "gpu":
            slot = free_gpu.pop(0)
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        else:
            free_cpu -= 1
            env["CUDA_VISIBLE_DEVICES"] = ""
        logp = ROOT / (t.log or f"{logdir}/{t.tid}.log")
        logp.parent.mkdir(parents=True, exist_ok=True)
        fh = open(logp, "w")
        p = subprocess.Popen(t.cmd, cwd=t.cwd, env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
        running[t.tid] = (p, t, slot, fh, time.time())
        print(f"[{time.strftime('%H:%M:%S')}] START {t.tid} "
              f"({t.resource}{'' if slot is None else slot})", flush=True)

    while pending or running:
        # reap finished
        for tid in list(running):
            p, t, slot, fh, t0 = running[tid]
            if p.poll() is None:
                continue
            fh.close()
            del running[tid]
            if t.resource == "gpu":
                free_gpu.append(slot)
                free_gpu.sort()
            else:
                free_cpu += 1
            dur = time.time() - t0
            ok = p.returncode == 0
            (done if ok else failed).add(tid)
            timeline.append({"task": tid, "resource": t.resource,
                             "start": t0 - t_start, "seconds": dur,
                             "ok": ok})
            print(f"[{time.strftime('%H:%M:%S')}] {'DONE ' if ok else 'FAIL '}"
                  f"{tid} in {dur/60:.1f} min", flush=True)

        # launch whatever is ready
        for tid in list(pending):
            t = pending[tid]
            if any(d in failed for d in t.deps):
                del pending[tid]
                failed.add(tid)
                print(f"SKIP {tid} (dependency failed)", flush=True)
                continue
            if not all(d in done for d in t.deps):
                continue
            if t.skip_if and Path(t.cwd, t.skip_if).exists():
                del pending[tid]
                done.add(tid)
                print(f"SKIP {tid} (cache present)", flush=True)
                continue
            if t.resource == "gpu" and not free_gpu:
                continue
            if t.resource == "cpu" and free_cpu <= 0:
                continue
            del pending[tid]
            launch(t)
        time.sleep(2)

    sampler.stop()
    total = time.time() - t_start
    print(f"\nall tasks finished in {total/60:.1f} min "
          f"({len(done)} ok, {len(failed)} failed)")
    print(sampler.report())
    json.dump({"total_minutes": total / 60, "timeline": timeline,
               "gpu_report": sampler.report(),
               "failed": sorted(failed)},
              open(logdir / "runner_summary.json", "w"), indent=2)
    print(f"wrote {logdir / 'runner_summary.json'}")


def main():
    ap = argparse.ArgumentParser(description="GPU-saturating experiment runner.")
    ap.add_argument("--spec", required=True, help="JSON job specification")
    ap.add_argument("--gpu-slots", type=int, default=2)
    ap.add_argument("--gpu-ids", default=None,
                    help="comma-separated physical GPU ids to use "
                         "(default: 0..gpu-slots-1). Lets two runners share a "
                         "machine without both grabbing device 0.")
    ap.add_argument("--cpu-slots", type=int, default=4)
    args = ap.parse_args()
    ids = ([int(x) for x in args.gpu_ids.split(',')]
           if args.gpu_ids else list(range(args.gpu_slots)))
    run(json.load(open(args.spec)), len(ids), args.cpu_slots, ids)


if __name__ == "__main__":
    main()
