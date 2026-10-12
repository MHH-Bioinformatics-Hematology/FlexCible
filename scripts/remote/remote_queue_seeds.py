"""Seed replicates of the AML folds (registry section 72): all12aml_refin_drop_fN, all12diff_refres_drop_fN
(channels only) and all12aml_honest_drop_fN, folds 10, 11, 12, 10 kb, seeds s1 and s2 (18 trainings).

Waits until meth_resid.sh has written METH_RESID_DONE and no other repo job runs. Batches (all three folds of one
family and seed), in this order: refin s1, diff s1, refin s2, diff s2, honest s1, honest s2. Before each job the
GPU is checked (wait for < 1 GiB used by others); claims by run name; trained runs are skipped (resumable).
After each batch: aml_train_eval --fold N --variant seeds (new labels only, predictions cached), then for the
refin and diff batches aml_loop_detection --recipe seeds (predict score stats inc criteria) and criteria2
(--tag seeds); tables to runs/diff_marks/ and the tidy seed table (scripts/aml_seeds_table.py) to
runs/diff_marks/aml_seeds/. Evaluation failures are logged and never count; 2 consecutive training failures stop.
Stop: kill $(cat remote_queue_seeds.pid).
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

PIDF = rq.REPO / "remote_queue_seeds.pid"
rq.PIDF = PIDF
DEST = rq.HM / "runs/diff_marks"
RES = 10000
FOLDS = (10, 11, 12)
FAM = {"refin": "all12aml_refin_drop", "diff": "all12diff_refres_drop", "honest": "all12aml_honest_drop"}
BATCHES = [("refin", 1), ("diff", 1), ("refin", 2), ("diff", 2), ("honest", 1), ("honest", 2)]
fails = 0


def run_name(fam, fold, seed):
    return mx.run_name(f"{FAM[fam]}_f{fold}_s{seed}", RES)


def wait_gpu(tag):
    t0 = time.time()
    last = 0
    while True:
        if rq.gpu_mib() < 1024 and not rq.cnn_alive():
            return True
        if time.time() - last > 1800:
            last = time.time()
            rq.log(f"seeds: waiting for the GPU before {tag} (used {rq.gpu_mib()} MiB)")
        time.sleep(30)


def eval_step(name, cmd, out):
    if not m.claim(name):
        return True
    ok = m.run_cmd(name, cmd, out)
    if not ok:
        rq.log(f"EVAL STEP FAILED (continuing): {name}")
    return ok


def copy(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def main():
    global fails
    if PIDF.exists():
        try:
            os.kill(int(PIDF.read_text()), 0)
            raise SystemExit("already running")
        except (OSError, ValueError):
            pass
    PIDF.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, rq.stop)
    signal.signal(signal.SIGINT, rq.stop)
    rq.log(f"remote_queue_seeds started pid {os.getpid()}")
    while "METH_RESID_DONE" not in (DEST / "meth_resid.log").read_text() if (DEST / "meth_resid.log").exists() else False:
        time.sleep(60)
    for fam, seed in BATCHES:
        trained_now = False
        for fold in FOLDS:
            name = run_name(fam, fold, seed)
            base = f"{FAM[fam]}_f{fold}_s{seed}"
            if m.trained(name):
                continue
            wait_gpu(name)
            if not m.claim(name):
                continue
            t0 = time.time()
            ok = m.run_cmd(name, f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} --res {RES} "
                                 f"--stage train --gpus 0", mx.REF_OUT / f"{name}.remote.out", lambda n=name: m.trained(n))
            fails = 0 if ok else fails + 1
            if ok:
                trained_now = True
            if fails >= 2:
                rq.log("2 consecutive training failures: seeds stopping")
                PIDF.unlink(missing_ok=True)
                return
        tag = f"{fam}_s{seed}"
        if not all(m.trained(run_name(fam, f, seed)) for f in FOLDS):
            rq.log(f"seeds batch {tag} incomplete, evaluation skipped")
            continue
        for fold in FOLDS:
            wait_gpu(f"eval {tag} f{fold}")
            eval_step(f"seeds_eval_f{fold}_{tag}", f"HICMULTI_GPUS=0 {rq.PYB} scripts/aml_train_eval.py --fold {fold} "
                      f"--variant seeds", DEST / f"seeds_eval_f{fold}_{tag}.out")
            for p in (rq.REPO / "results/aml_train").glob(f"eval_f{fold}_seeds_*.csv"):
                d = rq.HM / f"runs/aml_train_remote/fold{fold}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, d / p.name)
        if fam in ("refin", "diff"):
            wait_gpu(f"loop detection {tag}")
            ok = eval_step(f"seeds_loop_detection_{tag}",
                           f"{rq.PYB} scripts/aml_loop_detection.py --recipe seeds --stage predict score stats inc "
                           f"criteria --gpu 0 && {rq.PYB} scripts/aml_loop_detection.py --stage criteria2 --scores "
                           f"results/aml_loop_detection_seeds/object_scores.csv.gz --tag seeds --nboot 1000",
                           DEST / f"seeds_loop_detection_{tag}.out")
            src = rq.REPO / "results/aml_loop_detection_seeds"
            for p in src.glob("*.csv*"):
                copy(p, DEST / "aml_loop_detection_seeds" / p.name)
            copy(rq.REPO / "results/aml_loop_detection_criteria2/seeds_criteria2.csv", DEST / "seeds_criteria2.csv")
        # tidy table from the share tables
        r = subprocess.run([rq.PYB, "scripts/aml_seeds_table.py"], cwd=rq.REPO, env=rq.env, capture_output=True, text=True)
        (DEST / "aml_seeds_table.out").write_text(r.stdout + r.stderr)
        for p in (rq.REPO / "results/aml_seeds").glob("*.csv"):
            copy(p, DEST / "aml_seeds" / p.name)
        rq.log(f"seeds batch {tag} evaluated; table in {DEST / 'aml_seeds'}")
    rq.log("REMOTE_QUEUE_SEEDS_DONE")
    PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
