"""Second remote runner: starts after remote_queue.py has finished its list (REMOTE_QUEUE_DONE in
remote_queue.log), then trains on the single GPU, one after another, with the same claim logic,
recipe (scripts/run_single_gpu.py, doubled per-rank batch) and share outputs as the local queue:

  1. the base models of gpu_queue_extra.py block b (multicell_expanded --stage train --gpus 0):
     all12honest_drop_meth 25 kb s0, s1, 10 kb s0, s1 and all12honest_drop_motifmeth 25 kb s0, s1,
     10 kb s0 (their inference scoring stays local and waits for these models)
  2. the remaining dedicated scenario trainings in the local order (s0 25 kb x8, s1 meth_only 25 kb)

A job is appended only when its cache manifest exists. Stop: kill $(cat remote_queue_ext.pid).
Restart after an interruption: same command, finished jobs are skipped, an interrupted training
restarts from scratch.
"""
import os
import signal
import subprocess
import sys
import time

import remote_queue as rq

PIDF = rq.REPO / "remote_queue_ext.pid"
OUTQ = rq.HM / "runs/multicell_reference"
import multicell_expanded as mx   # noqa: E402
import scenario_bench as sb       # noqa: E402

BASES = [("all12honest_drop_meth", 25000), ("all12honest_drop_meth_s1", 25000),
         ("all12honest_drop_meth", 10000), ("all12honest_drop_meth_s1", 10000),
         ("all12honest_drop_motifmeth", 25000), ("all12honest_drop_motifmeth_s1", 25000),
         ("all12honest_drop_motifmeth", 10000)]


def base_cache(run, res):
    b = "all12honest_drop_motifmeth" if "motifmeth" in run else "all12honest_drop_meth"
    return rq.HM / f"cache_expanded/mc_{b}_{res // 1000}kb"


def base_done(name):
    si = mx.stop_info(name, 10)
    return bool(si and si[2] == "patience")


def main():
    if PIDF.exists():
        try:
            os.kill(int(PIDF.read_text()), 0)
            raise SystemExit("already running")
        except (OSError, ValueError):
            pass
    PIDF.write_text(str(os.getpid()))
    rq.PIDF = PIDF
    signal.signal(signal.SIGTERM, rq.stop)
    signal.signal(signal.SIGINT, rq.stop)
    first = rq.REPO / "remote_queue.pid"
    rq.log(f"remote_queue_ext started pid {os.getpid()}, waiting for remote_queue.py to finish")
    while first.exists():
        try:
            os.kill(int(first.read_text()), 0)
        except (OSError, ValueError):
            break
        time.sleep(30)
    time.sleep(5)
    if "REMOTE_QUEUE_DONE" not in rq.LOG.read_text().splitlines()[-3:].__str__() and \
            "all jobs finished" not in rq.LOG.read_text():
        rq.log("remote_queue.py did not end with REMOTE_QUEUE_DONE (failures or stop): ext not started")
        PIDF.unlink(missing_ok=True)
        return
    jobs, skipped = [], []
    for run, res in BASES:
        name = mx.run_name(run, res)
        if (base_cache(run, res) / "manifest.json").exists():
            jobs.append((name, "base", run, res))
        else:
            skipped.append(name)
    for n in sb.dedicated_runs():
        if n not in rq.JOBS:
            jobs.append((n, "ded", None, None))
    rq.log(f"ext jobs: {[j[0] for j in jobs]}; skipped (cache missing): {skipped}")
    while rq.gpu_mib() >= 1024:
        time.sleep(30)
    fails = 0
    for name, kind, run, res in jobs:
        done = base_done(name) if kind == "base" else sb.dedicated_done(name)
        if done:
            rq.log(f"skip {name}: finished")
            continue
        r = rq.sh([rq.PYB, "scripts/claim_job.py", "claim", name, "--dir", str(rq.CLAIMS), "--cmd",
                   "remote_queue_ext"], capture_output=True, text=True)
        if r.returncode == 1 and rq.my_claim(name):
            rq.log(f"{name}: reusing own claim of an interrupted run")
        elif r.returncode != 0:
            rq.log(f"skip {name}: {r.stdout.strip()} (claim exit {r.returncode})")
            continue
        rq.current = name
        if kind == "base":
            cmd = (f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {run} --res {res} "
                   f"--stage train --gpus 0")
            out = OUTQ / f"{name}.remote.out"
        else:
            cmd = (f"{rq.PYB} scripts/run_single_gpu.py scenario_bench --stage configs && "
                   f"{rq.PYB} scripts/run_single_gpu.py scenario_bench --stage train --run {name} --gpus 0")
            out = sb.OUT / f"{name}.remote.out"
        rq.log(f"start {name}")
        t0 = time.time()
        rq.child = subprocess.Popen(["bash", "-c", cmd], cwd=rq.REPO, env=rq.env, start_new_session=True,
                                    stdout=open(out, "a"), stderr=subprocess.STDOUT)
        rc = rq.child.wait()
        ok = rc == 0 and (base_done(name) if kind == "base" else sb.dedicated_done(name))
        rq.log(f"{'done' if ok else 'FAILED'} {name} rc={rc} {(time.time() - t0) / 3600:.2f} h")
        rq.current = None
        if ok:
            fails = 0
        else:
            rq.release(name)
            fails += 1
            if fails >= 2:
                rq.log("2 consecutive failures: ext stopping")
                break
    else:
        rq.log("REMOTE_QUEUE_EXT_DONE")
    PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
