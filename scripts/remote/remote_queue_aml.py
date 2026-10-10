"""Third remote runner: AML-inclusive cross-validation folds 10, 11 and 12 (registry section 36 recipe).

Starts when no seq_cnn_chip.py / run_seq_cnn_remote.sh process exists, remote_queue.pid and
remote_queue_ext.pid are gone, and GPU memory is below 1 GiB for 2 minutes. Then a 60-step GPU
smoke on the fold-10 25 kb config, then per fold (same order as gpu_scheduler.aml_block):
  all12aml_refin_drop_fN 25 kb, all12aml_honest_drop_fN 25 kb, the same two at 10 kb
  (multicell_expanded --stage train through scripts/run_single_gpu.py, doubled per-rank batch),
  then scripts/aml_train_eval.py --fold N (HICMULTI_GPUS=0).
Preparation (references, caches, hard split) is done by aml_prep.py; the runner waits for it.
Claims: run name (eval: aml_eval_fN) in runs/multicell_reference/claims. Stops after 2
consecutive failures. After each fold, results/aml_train/eval_fN_*.csv and hard_fN_*.json are
copied to runs/aml_train_remote/foldN/ on the share.
Stop: kill $(cat remote_queue_aml.pid).
"""
import os
import shutil
import signal
import subprocess
import sys
import time

import remote_queue as rq
import aml_train as at
import multicell_expanded as mx

PIDF = rq.REPO / "remote_queue_aml.pid"
DEST = rq.HM / "runs/aml_train_remote"
BASES = [("all12aml_refin_drop", 25000), ("all12aml_honest_drop", 25000),
         ("all12aml_refin_drop", 10000), ("all12aml_honest_drop", 10000)]
rq.PIDF = PIDF


def name_of(base, res, fold):
    return mx.run_name(f"{base}_f{fold}", res)


def trained(name):
    si = mx.stop_info(name, 10)
    return bool(si and si[2] == "patience")


def prep_ready(fold):
    ok = True
    for res in (25000, 10000):
        k = res // 1000
        for c in (f"mc_all12aml_ref_{k}kb_f{fold}", f"mc_all12aml_{k}kb_f{fold}"):
            ok = ok and (mx.CACHE / c / "manifest.json").exists()
        ok = ok and at.refs_done(fold, res) and at.hard_split_path(fold, res).exists()
    return ok


def wait_prep(fold):
    while not prep_ready(fold):
        alive = subprocess.run("pgrep -f '[a]ml_prep.py'", shell=True, capture_output=True).stdout
        if not alive:
            rq.log(f"preparation of fold {fold} incomplete and aml_prep.py is not running")
            return False
        time.sleep(60)
    return True


def wait_free():
    quiet = None
    last = 0
    while True:
        busy = (rq.cnn_alive() or (rq.REPO / "remote_queue.pid").exists()
                or (rq.REPO / "remote_queue_ext.pid").exists() or rq.gpu_mib() >= 1024)
        if busy:
            quiet = None
            if time.time() - last > 1800:
                last = time.time()
                rq.log(f"aml: waiting (cnn={rq.cnn_alive()} gpu_mib={rq.gpu_mib()})")
        else:
            quiet = quiet or time.time()
            if time.time() - quiet >= 120:
                rq.log("aml: GPU free")
                return
        time.sleep(15)


def smoke():
    import yaml
    S = rq.SMOKE
    S.mkdir(parents=True, exist_ok=True)
    cfg = mx.run_config(name_of("all12aml_refin_drop", 25000, 10), 100, 10, 500)
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"]) * 2
    cfg["train"].update(output_dir=str(S / "aml_smoke_run"), max_steps=60, save_checkpoints=False,
                        max_epochs=None, patience_epochs=None)
    cp = S / "aml_smoke.yaml"
    yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
    r = rq.sh([rq.PYB, "-m", "hicmultiome.train", "--config", str(cp)],
              stdout=open(S / "aml_smoke.log", "w"), stderr=subprocess.STDOUT, timeout=1800)
    if r.returncode or "done." not in (S / "aml_smoke.log").read_text(errors="replace"):
        return f"smoke failed rc={r.returncode}, log {S / 'aml_smoke.log'}"
    shutil.rmtree(S)
    return None


def claim(name):
    r = rq.sh([rq.PYB, "scripts/claim_job.py", "claim", name, "--dir", str(rq.CLAIMS), "--cmd",
               "remote_queue_aml"], capture_output=True, text=True)
    if r.returncode == 1 and rq.my_claim(name):
        rq.log(f"{name}: reusing own claim")
        return True
    if r.returncode:
        rq.log(f"skip {name}: {r.stdout.strip()} (claim exit {r.returncode})")
        return False
    return True


def run_cmd(name, cmd, out, done):
    rq.current = name
    rq.log(f"start {name}")
    t0 = time.time()
    rq.child = subprocess.Popen(["bash", "-c", cmd], cwd=rq.REPO, env=rq.env, start_new_session=True,
                                stdout=open(out, "a"), stderr=subprocess.STDOUT)
    rc = rq.child.wait()
    ok = rc == 0 and done()
    rq.log(f"{'done' if ok else 'FAILED'} {name} rc={rc} {(time.time() - t0) / 3600:.2f} h")
    rq.current = None
    if not ok:
        rq.release(name)
    return ok


def copy_results(fold):
    d = DEST / f"fold{fold}"
    d.mkdir(parents=True, exist_ok=True)
    for p in list((rq.REPO / "results/aml_train").glob(f"eval_f{fold}_*.csv")) + \
            list((rq.REPO / "results/aml_train").glob(f"hard_f{fold}_*.json")):
        shutil.copyfile(p, d / (p.name + ".part"))
        os.replace(d / (p.name + ".part"), d / p.name)
    rq.log(f"copied fold {fold} results to {d}")


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
    env_gpus = dict(rq.env, HICMULTI_GPUS="0")
    rq.log(f"remote_queue_aml started pid {os.getpid()}")
    wait_free()
    if not wait_prep(10):
        PIDF.unlink(missing_ok=True)
        return
    err = smoke()
    if err:
        rq.log(f"AML SMOKE FAILED, no job started: {err}")
        PIDF.unlink(missing_ok=True)
        sys.exit(1)
    rq.log("aml smoke ok")
    fails = 0
    for fold in (10, 11, 12):
        if not wait_prep(fold):
            break
        copy_results(fold)                               # hard-split jsons
        for base, res in BASES:
            name = name_of(base, res, fold)
            if trained(name):
                rq.log(f"skip {name}: finished")
                continue
            if not claim(name):
                continue
            cmd = (f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base}_f{fold} "
                   f"--res {res} --stage train --gpus 0")
            ok = run_cmd(name, cmd, mx.REF_OUT / f"{name}.remote.out", lambda n=name: trained(n))
            fails = 0 if ok else fails + 1
            if fails >= 2:
                rq.log("2 consecutive failures: aml stopping")
                PIDF.unlink(missing_ok=True)
                return
        names = [name_of(b, r, fold) for b, r in BASES]
        ev = f"aml_eval_f{fold}"
        done_ev = lambda f=fold: (rq.REPO / f"results/aml_train/eval_f{f}_summary.csv").exists()
        if all(trained(n) for n in names) and not done_ev():
            if claim(ev):
                cmd = f"HICMULTI_GPUS=0 {rq.PYB} scripts/aml_train_eval.py --fold {fold}"
                ok = run_cmd(ev, cmd, DEST / f"eval_f{fold}.remote.out", done_ev)
                fails = 0 if ok else fails + 1
                if fails >= 2:
                    rq.log("2 consecutive failures: aml stopping")
                    break
        if done_ev():
            copy_results(fold)
    else:
        rq.log("REMOTE_QUEUE_AML_DONE")
    PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    main()
