"""Priority runner (2026-10-11): the loop-centric AML pilot, then the rest of the remote_queue_more.py queue.

Replaces remote_queue_more.py between two jobs: the old runner process is killed by explicit PID (SIGKILL,
no handler) while its last child (an evaluation) keeps running as an orphan; this runner waits until no
training/evaluation process of the repo is left and the GPU has been quiet for 2 minutes, then:
  1. 60-step smoke of the loop-centric config (new code path),
  2. pilot all12aml_loopfocus_refres_drop_f10_10kb (claim by run name, run_single_gpu.py),
  3. cheap pilot evaluation: aml_loop_detection.py --recipe loopfocus, test donors of fold 10 (018 027 270):
     predict (GPU), score, stats, inc, criteria; tables copied to runs/aml_loopfocus/,
  4. hands over to remote_queue_more.main() (seed replicates, AML Meth/residual variants; finished jobs skipped).
Stops after 2 consecutive failures. Stop: kill $(cat remote_queue_loop.pid).
"""
import os
import shutil
import signal
import subprocess
import sys
import time

import remote_queue as rq
import remote_queue_more as m
import multicell_expanded as mx

PIDF = rq.REPO / "remote_queue_loop.pid"
rq.PIDF = PIDF
BASE, RES = "all12aml_loopfocus_refres_drop_f10", 10000
NAME = mx.run_name(BASE, RES)
ONLY = ["018", "027", "270"]
DEST = rq.HM / "runs/aml_loopfocus"


def busy_procs():
    r = subprocess.run("pgrep -f '[r]un_single_gpu.py|[h]icmultiome.train|[c]rosscell_predict.py|"
                       "[e]val_val_windows.py|[a]ml_train_eval.py'", shell=True, capture_output=True, text=True)
    return [p for p in r.stdout.split() if int(p) != os.getpid()]


def wait_idle():
    quiet = None
    while True:
        if busy_procs() or rq.cnn_alive() or rq.gpu_mib() >= 1024:
            quiet = None
        else:
            quiet = quiet or time.time()
            if time.time() - quiet >= 120:
                return
        time.sleep(15)


class PreWatch:
    """Stand-in for the precompute Popen of remote_queue_more: running while a precompute process lives."""
    def poll(self):
        r = subprocess.run("pgrep -f '[s]tage precompute'", shell=True, capture_output=True)
        return None if r.stdout.strip() else 0


def main():
    if PIDF.exists():
        try:
            os.kill(int(PIDF.read_text()), 0)
            raise SystemExit("already running")
        except (OSError, ValueError):
            pass
    PIDF.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, rq.stop)
    signal.signal(signal.SIGINT, rq.stop)
    rq.log(f"remote_queue_loop started pid {os.getpid()}; waiting for the previous job's processes to end")
    wait_idle()
    err = m.smoke(BASE, RES, "loop_smoke")
    if err:
        rq.log(f"LOOP SMOKE FAILED, nothing started: {err}")
        PIDF.unlink(missing_ok=True)
        sys.exit(1)
    rq.log("loop smoke ok")
    fails = 0
    if not m.trained(NAME):
        if m.claim(NAME):
            ok = m.run_cmd(NAME, f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {BASE} "
                                 f"--res {RES} --stage train --gpus 0", mx.REF_OUT / f"{NAME}.remote.out",
                           lambda: m.trained(NAME))
            if not ok:
                fails += 1
    if m.trained(NAME):
        DEST.mkdir(parents=True, exist_ok=True)
        cmds = [
            f"{rq.PYB} scripts/aml_loop_detection.py --recipe loopfocus --stage predict score --gpu 0 --only {' '.join(ONLY)}",
            "cp results/aml_loop_detection_loopfocus/object_scores_partial.csv.gz "
            "results/aml_loop_detection_loopfocus/object_scores.csv.gz",
            f"{rq.PYB} scripts/aml_loop_detection.py --recipe loopfocus --stage stats inc criteria",
        ]
        ok = m.run_cmd("loopfocus_pilot_eval", " && ".join(f"( {c} )" for c in cmds),
                       DEST / "pilot_eval_f10.out") if m.claim("loopfocus_pilot_eval_f10") else True
        if ok:
            for p in (rq.REPO / "results/aml_loop_detection_loopfocus").glob("*.csv*"):
                shutil.copyfile(p, DEST / (p.stem.replace(".csv", "") + "_f10_pilot" + "".join(p.suffixes)))
            rq.log(f"pilot tables copied to {DEST}")
        else:
            fails += 1
    if fails >= 2:
        rq.log("2 failures: loop runner stopping")
        PIDF.unlink(missing_ok=True)
        return
    rq.log("pilot done, handing over to remote_queue_more")
    PIDF.unlink(missing_ok=True)
    m.start_precompute = lambda: PreWatch()
    m.main()


if __name__ == "__main__":
    main()
