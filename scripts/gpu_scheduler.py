"""Dependency-driven job scheduler for the 12-cell experiments (both GPUs).

GPU jobs (two-GPU trainings, one at a time) are started in priority order,
the first whose inputs exist, the moment the GPUs are free; nothing waits on
a precompute or a consensus build: CPU preparation jobs (caches, cascade
priors, evaluations, analyses) run concurrently as soon as their inputs exist,
up to --cpu-slots at a time. Every job is skipped when its output exists, so
the scheduler can be restarted at any time. Milestones go to
runs/multicell_reference/driver.log; STATUS.md is refreshed after each job.

Priority (GPU): all12reflearn 25/10 kb, the multi-resolution model
all12multires, the cascade (fold models 25 kb, final 10 kb, fold models
10 kb, final 5 kb, fold models 5 kb, final 2 kb), the 5 and 2 kb pairs
(all12refres, all12honest), all12refin 25/10 kb. A lower-priority job runs
only while every higher one still waits for its inputs.

    python scripts/gpu_scheduler.py [--wait-pid PID]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade as cs                                                 # noqa: E402
import multicell_expanded as mx                                      # noqa: E402

REPO, PY = mx.REPO, mx.PY
MR = mx.REF_OUT
CACHE = mx.CACHE
LOG = MR / "driver.log"
OUTLOG = MR / "gpu_scheduler.out"
A_CELLS = "gm12878 imr90 hct116 hff h1 hepg2 a549 k562 hmec_deep"
B_CELLS = "nhek gm23248 mcf7 cd14mono pc9 hmec huvec huvec_deep"
M = f"{PY} scripts/multicell_expanded.py"


def note(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def cache_ok(name) -> bool:
    return (CACHE / name / "manifest.json").exists()


def trained(run: str, res: int) -> bool:
    name = mx.run_name(run, res)
    si = mx.stop_info(name, 10)
    return bool(si and si[2] == "patience")


def mark(name) -> Path:
    return MR / f".done_{name}"


def is_marked(name) -> bool:
    return mark(name).exists()


STATUS = (f"{M} --stage refsummary; {PY} scripts/learned_reference.py --stage summary; "
          f"{PY} scripts/coarse_gap.py --stage summary")


def eval_cmd(run, res):
    return (f"{M} --runs {run} --res {res} --stage eval --gpus 0 --cells {A_CELLS} & "
            f"{M} --runs {run} --res {res} --stage eval --gpus 1 --cells {B_CELLS} & wait")


class Job:
    def __init__(self, name, cmd, ready, done, gpu=False, after=None):
        self.name, self.cmd, self.ready, self.done, self.gpu = name, cmd, ready, done, gpu
        self.after = after          # name of the job whose completion adds this one's marker

    def __repr__(self):
        return self.name


def jobs():
    J = []

    def train12(run, res, cache):
        n = f"train_{mx.run_name(run, res)}"
        J.append(Job(n, f"{M} --runs {run} --res {res} --stage precompute --workers 8 && "
                        f"{M} --runs {run} --res {res} --stage train --gpus 0,1",
                     lambda: cache_ok(cache), lambda: trained(run, res), gpu=True))
        J.append(Job(f"eval_{mx.run_name(run, res)}", eval_cmd(run, res),
                     lambda: trained(run, res), lambda: is_marked(f"eval_{mx.run_name(run, res)}")))

    def prep_cache(run, res, cache, needs=lambda: True):
        J.append(Job(f"cache_{cache}",
                     f"{M} --runs {run} --res {res} --stage precompute --workers 8",
                     needs, lambda: cache_ok(cache)))

    # --- caches (CPU, ahead of time)
    prep_cache("all12reflearn", 10000, "mc_all12reflearn_10kb",
               lambda: mx.learned_map(10000).exists())
    prep_cache("all12reflearn", 25000, "mc_all12reflearn_25kb",
               lambda: mx.learned_map(25000).exists())
    prep_cache("all12refres", 2000, "mc_all12ref_2kb",
               lambda: mx.consensus_loo("pc9", 2000).exists())
    prep_cache("all12refres", 5000, "mc_all12ref_5kb",
               lambda: mx.consensus_loo("pc9", 5000).exists())
    # --- GPU jobs in priority order, with their evaluations
    train12("all12reflearn", 25000, "mc_all12reflearn_25kb")
    train12("all12reflearn", 10000, "mc_all12reflearn_10kb")
    a_ready = lambda: all(cache_ok(f"mc_all12ref_{k}kb") for k in (2, 5, 10, 25))
    a_log = MR / "all12multires_train.log"
    a_done = lambda: "done." in (a_log.read_text(errors="replace") if a_log.exists() else "")
    J.append(Job("train_all12multires", f"{PY} scripts/multires.py --stage train",
                 a_ready, a_done, gpu=True))
    J.append(Job("eval_all12multires", f"{PY} scripts/multires.py --stage views && "
                                       f"{PY} scripts/multires.py --stage eval",
                 a_done, lambda: is_marked("eval_all12multires")))
    views = " ".join(str(MR / mx.run_name("all12multires", r)) for r in mx.MULTIRES)
    J.append(Job("analysis_c", f"CUDA_VISIBLE_DEVICES=1 {PY} scripts/scale_importance.py "
                               f"--views {views} --out {MR / 'scale_importance'}",
                 lambda: is_marked("eval_all12multires"),
                 lambda: (MR / "scale_importance" / "importance.png").exists()))
    # --- cascade
    for res in cs.COARSE_LEVELS:
        fine = {25000: 10000, 10000: 5000, 5000: 2000}[res]
        cache_res = "mc_all12ref_25kb" if res == 25000 else f"mc_all12casc_{res // 1000}kb"
        for f in cs.FOLDS:
            log = MR / f"{cs.fold_name(res, f)}.log"
            J.append(Job(f"train_{cs.fold_name(res, f)}",
                         f"{PY} scripts/cascade.py --stage fold --res {res} --fold {f}",
                         (lambda c=cache_res: cache_ok(c)),
                         (lambda l=log: cs.done(l)), gpu=True))
        fold_logs = [MR / f"{cs.fold_name(res, f)}.log" for f in cs.FOLDS]
        J.append(Job(f"priors_{res // 1000}kb", f"{PY} scripts/cascade.py --stage priors --res {res}",
                     (lambda ls=fold_logs: all(cs.done(l) for l in ls)),
                     (lambda r=res: all(mx.cascade_prior(c, r).exists() for c in cs.IDENT_CELLS))))
        J.append(Job(f"cache_mc_all12casc_{fine // 1000}kb",
                     f"{PY} scripts/cascade.py --stage precompute --res {fine}",
                     (lambda r=res: all(mx.cascade_prior(c, r).exists() for c in cs.IDENT_CELLS)
                      and (fine != 2000 or cache_ok("mc_all12ref_2kb"))),
                     (lambda fr=fine: cache_ok(f"mc_all12casc_{fr // 1000}kb"))))
        train12("all12casc", fine, f"mc_all12casc_{fine // 1000}kb")
    # --- pairs and the input-only reference model
    train12("all12refres", 5000, "mc_all12ref_5kb")
    train12("all12honest", 5000, "mc_all12_5kb")
    train12("all12refres", 2000, "mc_all12ref_2kb")
    train12("all12honest", 2000, "mc_all12")
    train12("all12refin", 25000, "mc_all12ref_25kb")
    train12("all12refin", 10000, "mc_all12ref_10kb")
    return J


def launch(job: Job):
    note(f"start {job.name}")
    wrapped = (f"cd {REPO} && ( {job.cmd} ) >> {OUTLOG} 2>&1; rc=$?; "
               f"[ $rc -eq 0 ] && touch {mark(job.name)}; "
               f"({STATUS}) > /dev/null 2>&1; exit $rc")
    return subprocess.Popen(["bash", "-c", wrapped], start_new_session=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-slots", type=int, default=3)
    ap.add_argument("--wait-pid", type=int, nargs="*", default=[],
                    help="GPU trainings started before the scheduler: no GPU job "
                         "starts while one of these runs")
    a = ap.parse_args()
    for h in MR.glob("HOLD_*"):
        h.unlink()
    note("gpu_scheduler started (replaces the shell drivers)")
    all_jobs = jobs()
    running: dict[str, subprocess.Popen] = {}
    failed: set[str] = set()
    while True:
        for n, p in list(running.items()):
            if p.poll() is not None:
                note(f"{'done' if p.returncode == 0 else 'FAILED'} {n}")
                if p.returncode != 0:
                    failed.add(n)
                del running[n]
        pending = [j for j in all_jobs if j.name not in running and j.name not in failed
                   and not j.done()]
        if not pending and not running:
            note("GPU_SCHEDULER_DONE")
            return
        external = any(_alive(pid) for pid in a.wait_pid)
        gpu_busy = external or any(j.gpu for j in all_jobs if j.name in running)
        if not gpu_busy:
            for j in pending:
                if j.gpu and j.ready():
                    running[j.name] = launch(j)
                    break
        cpu_running = sum(1 for j in all_jobs if j.name in running and not j.gpu)
        for j in pending:
            if cpu_running >= a.cpu_slots:
                break
            if not j.gpu and j.name not in running and j.ready():
                running[j.name] = launch(j)
                cpu_running += 1
        time.sleep(30)


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
