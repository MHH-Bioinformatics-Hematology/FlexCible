"""Sequential runner for the dedicated scenario trainings moved to this machine.

Waits until the sequence CNN is gone (no seq_cnn_chip.py / run_seq_cnn_remote.sh process and
GPU memory below 1 GiB for 2 consecutive minutes), runs a 60-step GPU smoke, then trains the
jobs below one after another on the single GPU through scripts/run_single_gpu.py (per-rank batch
doubled, global batch 6 as in the local 2-GPU recipe). Each job is claimed with
scripts/claim_job.py (claim name = run name, claims folder on the share). Same commands as
gpu_queue_extra.py: `scenario_bench --stage configs && scenario_bench --stage train --run R`.
Inference scoring (scen_ded_*) stays in the local queue.

Resumable: finished jobs (scenario_bench.dedicated_done) are skipped; an interrupted training
restarts from scratch (train.resume is false in these configs); a claim of this host left by an
interrupted run is reused. Stops after 2 consecutive failures. Stop: kill $(cat remote_queue.pid)
(SIGTERM ends the running job's process group and releases its claim).

    setsid nohup ~/hicenv/bin/python remote_queue.py >> remote_queue.out 2>&1 &
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
CLAIMS = HM / "runs/multicell_reference/claims"
SMOKE = HM / "remote_smoke"
PYB = str(Path.home() / "hicenv/bin/python")
LOG = REPO / "remote_queue.log"
PIDF = REPO / "remote_queue.pid"
JOBS = [f"scen_{s}_25kb_s1" for s in ("ctcf_meth", "ctcf_h3k27ac", "ctcf_h3k27ac_meth",
        "ctcf_ac_k4me3", "ctcf_ac_k4me3_meth", "five_marks", "five_marks_meth")] + \
       [f"scen_{s}_10kb_s0" for s in ("meth_only", "ctcf_meth", "ctcf_h3k27ac_meth")]
HOST = os.environ.get("HICMULTI_HOST") or os.uname().nodename.split(".")[0]
env = dict(os.environ, HICMULTI_PY=PYB, PYTHONUNBUFFERED="1")
sys.path.insert(0, str(REPO / "scripts"))
child = None
current = None


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO, env=env, **kw)


def cnn_alive():
    r = subprocess.run("pgrep -f '[s]eq_cnn_chip.py|[r]un_seq_cnn_remote.sh'", shell=True,
                       capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if int(p) != os.getpid()]
    return bool(pids)


def gpu_mib():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 10**9


def wait_free():
    quiet_since = None
    last = 0
    while True:
        busy = cnn_alive() or gpu_mib() >= 1024
        if busy:
            quiet_since = None
            if time.time() - last > 1800:
                last = time.time()
                log(f"waiting: CNN alive={cnn_alive()} gpu_mib={gpu_mib()}")
        else:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= 120:
                log("GPU free: no CNN process, GPU memory below 1 GiB for 2 minutes")
                return
        time.sleep(15)


def smoke():
    """60 steps on the GPU of a 25 kb scenario config, config built through the wrapper."""
    import yaml
    SMOKE.mkdir(parents=True, exist_ok=True)
    if sh([PYB, "scripts/run_single_gpu.py", "scenario_bench", "--stage", "configs"],
          capture_output=True).returncode:
        return "stage configs failed"
    c = yaml.safe_load(open(REPO / "configs/scenario/scen_ctcf_meth_25kb_s1.yaml"))
    if c["train"]["batch_size"] != 6:
        return f"wrapper batch size {c['train']['batch_size']} != 6"
    c["train"].update(output_dir=str(SMOKE / "smoke_run"), max_steps=60, save_checkpoints=False,
                      max_epochs=None, patience_epochs=None)
    cp = SMOKE / "smoke.yaml"
    yaml.safe_dump(c, open(cp, "w"), sort_keys=False)
    t0 = time.time()
    r = sh([PYB, "-m", "hicmultiome.train", "--config", str(cp)], stdout=open(SMOKE / "smoke.log", "w"),
           stderr=subprocess.STDOUT, timeout=1200)
    txt = (SMOKE / "smoke.log").read_text(errors="replace")
    if r.returncode or "done." not in txt:
        return f"smoke training failed rc={r.returncode}, log {SMOKE / 'smoke.log'}"
    log(f"smoke ok: 60 steps in {time.time() - t0:.0f} s wall incl. start, compile={c['train'].get('compile')}")
    shutil.rmtree(SMOKE)
    return None


def my_claim(run):
    try:
        return json.loads((CLAIMS / f"{run}.claim").read_text()).get("host") == HOST
    except (OSError, ValueError):
        return False


def release(run):
    sh([PYB, "scripts/claim_job.py", "release", run, "--dir", str(CLAIMS)], capture_output=True)


def stop(signum, frame):
    log(f"signal {signum}: stopping")
    if child and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=60)
        except Exception:
            pass
    if current:
        release(current)
    PIDF.unlink(missing_ok=True)
    sys.exit(0)


def main():
    global child, current
    if PIDF.exists():
        try:
            os.kill(int(PIDF.read_text()), 0)
            raise SystemExit(f"already running: pid {PIDF.read_text().strip()}")
        except (OSError, ValueError):
            pass
    PIDF.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    import scenario_bench as sb
    log(f"remote_queue started pid {os.getpid()} host {HOST}; {len(JOBS)} jobs: {JOBS}")
    todo = [j for j in JOBS if not sb.dedicated_done(j)]
    if not todo:
        log("all jobs finished")
        PIDF.unlink(missing_ok=True)
        return
    wait_free()
    err = smoke()
    if err:
        log(f"SMOKE FAILED, no real job started: {err}")
        PIDF.unlink(missing_ok=True)
        sys.exit(1)
    fails = 0
    for run in JOBS:
        if sb.dedicated_done(run):
            log(f"skip {run}: finished")
            continue
        r = sh([PYB, "scripts/claim_job.py", "claim", run, "--dir", str(CLAIMS), "--cmd",
                "remote_queue run_single_gpu scenario_bench --stage train"], capture_output=True, text=True)
        if r.returncode == 1 and my_claim(run):
            log(f"{run}: reusing own claim of an interrupted run")
        elif r.returncode != 0:
            log(f"skip {run}: {r.stdout.strip()} (claim exit {r.returncode})")
            continue
        current = run
        out = sb.OUT / f"{run}.remote.out"
        sb.OUT.mkdir(parents=True, exist_ok=True)
        log(f"start {run}")
        t0 = time.time()
        cmd = (f"{PYB} scripts/run_single_gpu.py scenario_bench --stage configs && "
               f"{PYB} scripts/run_single_gpu.py scenario_bench --stage train --run {run} --gpus 0")
        child = subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=env, start_new_session=True,
                                 stdout=open(out, "a"), stderr=subprocess.STDOUT)
        rc = child.wait()
        ok = rc == 0 and sb.dedicated_done(run)
        log(f"{'done' if ok else 'FAILED'} {run} rc={rc} {(time.time() - t0) / 3600:.2f} h")
        current = None
        if ok:
            fails = 0
        else:
            release(run)
            fails += 1
            if fails >= 2:
                log("2 consecutive failures: stopping")
                break
    else:
        log("REMOTE_QUEUE_DONE")
    PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
