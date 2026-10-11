"""Remote runner for the differential mark channels (registry section 66, results/diff_marks/REMOTE_JOB.md),
second order (2026-10-11): channels first, objects later.

Lock: holds remote_queue_more.pid so that remote_queue_loop.py (last step remote_queue_more.main()) exits after the
loop-focus pilot instead of starting the older queue. Waits for remote_queue_loop.py to be gone and 2 quiet GPU
minutes, smokes the tier-0 config (60 steps), then at 10 kb, claims by run name:
  T0a  all12diffcells_refres_drop (channels only, 12 cell lines): train, standard eval, refsummary, cell-line
       criterion (median Pearson on the 12 training / 5 held-out cells within 0.006 of all12refres_drop_10kb,
       else STOP_TIER0 and the AML diff stages are skipped)
  T1a  all12diff_refres_drop_f10/f11/f12      (AML folds, channels only, reference = training patients)
  T1b  all12diffhspc_refres_drop_f10/f11/f12  (channels only, reference marks = HSPC donors)
  T0b  all12diffcells_loopfocus_refres_drop   (12 cell lines, channels + cell objects)
  T2a  all12diff_loopfocusaml_refres_drop_fN  (channels + AML objects only)
  T2b  all12diff_loopfocus_refres_drop_fN     (channels + all-cell objects, ablation)
After each AML stage: aml_train_eval --fold N --variant extra, aml_loop_detection --recipe diff_loopfocus (predict
of the finished diff models, score stats inc criteria); after T2b also cell_objects_eval. Tables are copied to
runs/diff_marks/ on the share. STOP rule per AML stage: best honest validation Pearson more than 0.01 below the
baseline (T1: all12aml_refres_drop, else all12aml_refin_drop; T2: the channels-only run of the fold) skips the
remaining folds of that stage. Stops after 2 consecutive failures. Finally hands over to remote_queue_more.main().
Stop: kill $(cat remote_queue_diff.pid).
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
BASE0 = "all12refres_drop_10kb"
FOLDS = (10, 11, 12)
fails = 0


def release_lock():
    if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
        LOCK.unlink()


def stop(signum, frame):
    release_lock()
    rq.stop(signum, frame)


def cache_ready(base):
    c = mx.run_config(mx.run_name(base, RES), 100, 10, 500)
    return os.path.exists(os.path.join(c["train"]["cache_dir"], "manifest.json"))


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
        busy = subprocess.run("pgrep -f '[r]emote_queue_loop.py|[r]un_single_gpu.py|[h]icmultiome.train|"
                              "[c]rosscell_predict.py|[a]ml_loop_detection.py|[a]ml_train_eval.py'", shell=True,
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


def train(base):
    global fails
    name = mx.run_name(base, RES)
    if m.trained(name):
        return True
    if not m.claim(name):
        return False
    ok = m.run_cmd(name, f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} --res {RES} "
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
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*"):
        if p.is_file():
            shutil.copyfile(p, dst / (p.name + ".part"))
            os.replace(dst / (p.name + ".part"), dst / p.name)


def cell_eval(base):
    name = mx.run_name(base, RES)
    if m.trained(name) and not (mx.REF_OUT / name / "test_eval_gm12878.json").exists():
        run_step(f"{name}_eval", f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} "
                                 f"--res {RES} --stage eval --gpus 0", mx.REF_OUT / f"{name}.remote.out")


def criterion(base):
    subprocess.run(f"{rq.PYB} scripts/multicell_expanded.py --stage refsummary", shell=True, cwd=rq.REPO, env=rq.env,
                   stdout=open(DEST / "refsummary.out", "a"), stderr=subprocess.STDOUT)
    name = mx.run_name(base, RES)
    bad = False
    for grp in ("12 training", "held-out"):
        v, b = summary_row(name, grp), summary_row(BASE0, grp)
        rq.log(f"cell-line criterion {name} [{grp}]: median Pearson {v} against {BASE0} {b}")
        if v is None or b is None or v - b < -0.006:
            bad = True
    return not bad


def best_val(name):
    si = mx.stop_info(name, 10)
    return si[1] if si else None


def aml_stage(tag, bases, baselines):
    """Train the AML folds of one family, stop rule on the honest validation, evaluate."""
    done = []
    for base in bases:
        if fails >= 2:
            break
        if not wait_cache(base):
            rq.log(f"{base}: cache not ready, skipped")
            continue
        if not train(base):
            continue
        fold = int(base.rsplit("_f", 1)[1])
        v = best_val(mx.run_name(base, RES))
        ref = next(((b, best_val(mx.run_name(b, RES))) for b in baselines(fold)
                    if best_val(mx.run_name(b, RES)) is not None), None)
        rq.log(f"{tag} {base}: best honest validation {v}; baseline {ref}")
        done.append(fold)
        if v is not None and ref and v - ref[1] < -0.01:
            rq.log(f"STOP_{tag}: honest validation more than 0.01 below {ref[0]}; remaining folds of {tag} skipped")
            break
    for fold in done:
        if fails >= 2:
            return
        if run_step(f"diff_aml_eval_f{fold}_{tag}", f"HICMULTI_GPUS=0 {rq.PYB} scripts/aml_train_eval.py --fold {fold} "
                    f"--variant extra", DEST / f"aml_eval_f{fold}_{tag}.out"):
            for p in (rq.REPO / "results/aml_train").glob(f"eval_f{fold}_extra_*.csv"):
                (DEST / f"fold{fold}").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, DEST / f"fold{fold}" / p.name)
    if done and fails < 2:
        run_step(f"diff_loop_detection_{tag}", f"{rq.PYB} scripts/aml_loop_detection.py --recipe diff_loopfocus --stage "
                 f"predict --gpu 0 && {rq.PYB} scripts/aml_loop_detection.py --recipe diff_loopfocus --stage score "
                 f"stats inc criteria", DEST / f"loop_detection_{tag}.out")
        copy_dir(rq.REPO / "results/aml_loop_detection_diff_loopfocus", DEST / f"aml_loop_detection_diff_loopfocus_after_{tag}")
    return done


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
    rq.log(f"remote_queue_diff (order 2: channels first) started pid {os.getpid()}; waiting for the loop-focus pilot")
    wait_idle()
    T0a, T0b = "all12diffcells_refres_drop", "all12diffcells_loopfocus_refres_drop"
    if not wait_cache(T0a):
        rq.log("tier-0 cache never became ready; stopping")
        release_lock(); PIDF.unlink(missing_ok=True); return
    err = m.smoke(T0a, RES, "diff_smoke")
    if err:
        rq.log(f"DIFF SMOKE FAILED, nothing started: {err}")
        release_lock(); PIDF.unlink(missing_ok=True); sys.exit(1)
    rq.log("diff smoke ok")
    t0_pass = False
    if train(T0a):
        cell_eval(T0a)
        t0_pass = criterion(T0a)
    if fails >= 2:
        rq.log("2 consecutive failures: diff stopping")
        release_lock(); PIDF.unlink(missing_ok=True); return
    if not t0_pass:
        rq.log("STOP_TIER0: cell-line Pearson fell by more than 0.006 (or is missing); AML diff stages skipped")
    else:
        rq.log("tier 0 criterion passed")
        aml_base = lambda f: (f"all12aml_refres_drop_f{f}", f"all12aml_refin_drop_f{f}")
        aml_stage("T1a", [f"all12diff_refres_drop_f{f}" for f in FOLDS], aml_base)
        aml_stage("T1b", [f"all12diffhspc_refres_drop_f{f}" for f in FOLDS], aml_base)
        if fails < 2 and train(T0b):
            cell_eval(T0b)
            criterion(T0b)
        chan = lambda f: (f"all12diff_refres_drop_f{f}",) + aml_base(f)
        aml_stage("T2a", [f"all12diff_loopfocusaml_refres_drop_f{f}" for f in FOLDS], chan)
        done = aml_stage("T2b", [f"all12diff_loopfocus_refres_drop_f{f}" for f in FOLDS], chan) or []
        if done and fails < 2:
            models = "base_refres " + " ".join(f"diff_loopfocus_f{f}" for f in done)
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
