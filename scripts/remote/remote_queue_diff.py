"""Remote runner for the differential mark channels and all-cell objects (registry section 66,
results/diff_marks/REMOTE_JOB.md), ahead of the seed replicates and the AML Meth/residual jobs.

Lock: writes its PID into remote_queue_more.pid, so that remote_queue_loop.py (whose last step is
remote_queue_more.main()) exits with "already running" after the loop-focus pilot instead of starting the
older queue. This runner waits for remote_queue_loop.py to be gone, then for 2 quiet GPU minutes, and
  1. smokes the tier-0 loop-focus config (60 steps),
  2. tier 0 at 10 kb: all12diffcells_refres_drop, all12diffcells_loopfocus_refres_drop (train, standard eval),
     refsummary, cell-line criterion: median Pearson (12 training / 5 held-out cells) against
     all12refres_drop_10kb must not fall by more than 0.006; otherwise the primary runs are skipped
     (STOP_TIER0 in the log),
  3. primary all12diff_loopfocus_refres_drop_f10/f11/f12 at 10 kb (stop rule: honest validation Pearson
     more than 0.01 below the AML residual baseline of the fold (all12aml_refres_drop, else refin) stops
     the remaining folds),
  4. evaluations: aml_train_eval --fold N --variant extra, aml_loop_detection --recipe diff_loopfocus
     (predict, score stats inc criteria), cell_objects_eval (windows, predict, score); tables copied to
     runs/diff_marks/ on the share,
  5. hands over to remote_queue_more.main() (seed replicates, AML Meth/residual jobs).
Claims by run name; stops after 2 consecutive failures. Stop: kill $(cat remote_queue_diff.pid).
"""
import csv
import os
import shutil
import signal
import subprocess
import sys
import time

import remote_queue as rq
import remote_queue_more as m
import multicell_expanded as mx

PIDF = rq.REPO / "remote_queue_diff.pid"
LOCK = rq.REPO / "remote_queue_more.pid"
rq.PIDF = PIDF
DEST = rq.HM / "runs/diff_marks"
RES = 10000
T0 = ["all12diffcells_refres_drop", "all12diffcells_loopfocus_refres_drop"]
T1 = [f"all12diff_loopfocus_refres_drop_f{f}" for f in (10, 11, 12)]
BASE0 = "all12refres_drop_10kb"
fails = 0


def release_lock():
    if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
        LOCK.unlink()


def stop(signum, frame):
    release_lock()
    rq.stop(signum, frame)


def cache_ready(base):
    c = mx.run_config(mx.run_name(base, RES), 100, 10, 500)
    return (os.path.exists(os.path.join(c["train"]["cache_dir"], "manifest.json")))


def wait_cache(base):
    t0 = time.time()
    while not cache_ready(base):
        if time.time() - t0 > 6 * 3600:
            return False
        time.sleep(60)
    return True


def wait_idle():
    quiet = None
    while True:
        loop = subprocess.run("pgrep -f '[r]emote_queue_loop.py'", shell=True, capture_output=True).stdout.strip()
        busy = loop or subprocess.run("pgrep -f '[r]un_single_gpu.py|[h]icmultiome.train|[c]rosscell_predict.py|"
                                      "[a]ml_loop_detection.py|[a]ml_train_eval.py'", shell=True,
                                      capture_output=True).stdout.strip()
        if busy or rq.cnn_alive() or rq.gpu_mib() >= 1024:
            quiet = None
        else:
            quiet = quiet or time.time()
            if time.time() - quiet >= 120:
                return
        time.sleep(20)


def summary_row(run, group):
    with open(mx.REF_OUT / "reference_summary.csv") as fh:
        for r in csv.DictReader(fh):
            if r["run"] == run and r["cells"].startswith(group) and r["res_kb"] == str(RES // 1000):
                return float(r["median_model"])
    return None


def train(base, res=RES):
    global fails
    name = mx.run_name(base, res)
    if m.trained(name):
        return True
    if not m.claim(name):
        return False
    ok = m.run_cmd(name, f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} --res {res} "
                         f"--stage train --gpus 0", mx.REF_OUT / f"{name}.remote.out", lambda: m.trained(name))
    fails = 0 if ok else fails + 1
    return ok


def run_step(name, cmd, out):
    global fails
    if not m.claim(name):
        return True
    ok = m.run_cmd(name, cmd, out)
    fails = 0 if ok else fails + 1
    return ok


def copy_dir(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*"):
        if p.is_file():
            shutil.copyfile(p, dst / (p.name + ".part"))
            os.replace(dst / (p.name + ".part"), dst / p.name)


def main():
    global fails
    if PIDF.exists():
        try:
            os.kill(int(PIDF.read_text()), 0)
            raise SystemExit("already running")
        except (OSError, ValueError):
            pass
    PIDF.write_text(str(os.getpid()))
    if LOCK.exists():
        try:
            os.kill(int(LOCK.read_text()), 0)
            raise SystemExit("remote_queue_more.pid belongs to a live process")
        except (OSError, ValueError):
            pass
    LOCK.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    DEST.mkdir(parents=True, exist_ok=True)
    rq.log(f"remote_queue_diff started pid {os.getpid()}; waiting for the loop-focus pilot to finish")
    wait_idle()
    if not wait_cache(T0[1]):
        rq.log("tier-0 cache never became ready; stopping")
        release_lock(); PIDF.unlink(missing_ok=True); return
    err = m.smoke(T0[1], RES, "diff_smoke")
    if err:
        rq.log(f"DIFF SMOKE FAILED, nothing started: {err}")
        release_lock(); PIDF.unlink(missing_ok=True); sys.exit(1)
    rq.log("diff smoke ok")
    # ---- tier 0
    for base in T0:
        name = mx.run_name(base, RES)
        if train(base):
            if not (mx.REF_OUT / name / "test_eval_gm12878.json").exists():
                run_step(f"{name}_eval", f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} "
                                         f"--res {RES} --stage eval --gpus 0", mx.REF_OUT / f"{name}.remote.out")
        if fails >= 2:
            rq.log("2 consecutive failures: diff stopping")
            release_lock(); PIDF.unlink(missing_ok=True); return
    subprocess.run(f"{rq.PYB} scripts/multicell_expanded.py --stage refsummary", shell=True, cwd=rq.REPO, env=rq.env,
                   stdout=open(DEST / "refsummary.out", "a"), stderr=subprocess.STDOUT)
    stop_t0 = False
    for base in T0:
        name = mx.run_name(base, RES)
        for grp in ("12 training", "held-out"):
            v, b = summary_row(name, grp), summary_row(BASE0, grp)
            rq.log(f"tier0 criterion {name} [{grp}]: median Pearson {v} against {BASE0} {b}")
            if v is None or b is None or v - b < -0.006:
                stop_t0 = True
    if stop_t0:
        rq.log("STOP_TIER0: cell-line Pearson fell by more than 0.006 (or is missing); primary AML runs skipped")
    else:
        rq.log("tier 0 criterion passed")
        # ---- tier 1
        trained_ok = []
        for base in T1:
            if not wait_cache(base):
                rq.log(f"{base}: cache not ready, skipped")
                continue
            if fails >= 2:
                break
            if not train(base):
                continue
            name = mx.run_name(base, RES)
            fold = int(base.rsplit("_f", 1)[1])
            si = mx.stop_info(name, 10)
            ref = None
            for b in (f"all12aml_refres_drop_f{fold}", f"all12aml_refin_drop_f{fold}"):
                s2 = mx.stop_info(mx.run_name(b, RES), 10)
                if s2:
                    ref = (b, s2[1])
                    break
            rq.log(f"{name}: best honest validation {si[1] if si else None}; baseline {ref}")
            trained_ok.append(fold)
            if si and ref and si[1] - ref[1] < -0.01:
                rq.log("STOP_TIER1: honest validation more than 0.01 below the AML baseline; other folds skipped")
                break
        for fold in trained_ok:
            if run_step(f"diff_aml_eval_f{fold}", f"HICMULTI_GPUS=0 {rq.PYB} scripts/aml_train_eval.py --fold {fold} "
                        f"--variant extra", DEST / f"aml_eval_f{fold}.out"):
                for p in (rq.REPO / "results/aml_train").glob(f"eval_f{fold}_extra_*.csv"):
                    DEST.joinpath(f"fold{fold}").mkdir(exist_ok=True)
                    shutil.copyfile(p, DEST / f"fold{fold}" / p.name)
            if fails >= 2:
                break
        if trained_ok and fails < 2:
            run_step("diff_loop_detection", f"{rq.PYB} scripts/aml_loop_detection.py --recipe diff_loopfocus --stage predict "
                     f"--gpu 0 && {rq.PYB} scripts/aml_loop_detection.py --recipe diff_loopfocus --stage score stats inc "
                     f"criteria", DEST / "loop_detection.out")
            copy_dir(rq.REPO / "results/aml_loop_detection_diff_loopfocus", DEST / "aml_loop_detection_diff_loopfocus")
            models = "base_refres " + " ".join(f"diff_loopfocus_f{f}" for f in trained_ok)
            run_step("diff_cell_objects_eval",
                     f"{rq.PYB} scripts/cell_objects_eval.py --stage windows && "
                     f"{rq.PYB} scripts/cell_objects_eval.py --stage predict --models {models} --gpu 0 && "
                     f"{rq.PYB} scripts/cell_objects_eval.py --stage score --models {models}",
                     DEST / "cell_objects_eval.out")
            copy_dir(rq.REPO / "results/cell_objects_eval", DEST / "cell_objects_eval")
    if fails >= 2:
        rq.log("2 consecutive failures: diff stopping")
        release_lock(); PIDF.unlink(missing_ok=True); return
    rq.log("diff queue done, handing over to remote_queue_more")
    release_lock()
    PIDF.unlink(missing_ok=True)
    class PreWatch:
        def poll(self):
            r = subprocess.run("pgrep -f '[s]tage precompute'", shell=True, capture_output=True)
            return None if r.stdout.strip() else 0
    m.start_precompute = lambda: PreWatch()
    m.main()


if __name__ == "__main__":
    main()
