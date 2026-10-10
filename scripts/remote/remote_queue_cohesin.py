"""Fourth remote runner: cohesin-fraction input and cohesin-loss augmentation (registry section 55,
results/cohesin_fix/REMOTE_JOB.md). Same pattern as remote_queue_aml.py.

Waits until remote_queue_aml.py has finished (remote_queue_aml.pid gone and its log ends with
REMOTE_QUEUE_AML_DONE, or every AML f10 to f12 training is finished/claimed), no seq_cnn_chip.py
process exists and GPU memory has been below 1 GiB for 2 minutes. Then a 60-step smoke on the 25 kb
config and, in this order (claims by run name, run_single_gpu.py multicell_expanded --stage train
--gpus 0, then --stage eval --gpus 0 per run):
  all12cohfix_refres_drop 25 kb, 10 kb, all12cohfixna_refres_drop 25 kb, 10 kb
A job whose cache manifest or operator file is missing is skipped for now and retried after the other
jobs (the 10 kb cache is built on the local machine). Then the prediction stages of the evaluation
(perturb_validate predict predict_chip, washout predict, aml_eval predict for the four models; all on
the CPU, predictions on the share, scoring stays local). Stops after 2 consecutive failures.
Stop: kill $(cat remote_queue_cohesin.pid).
"""
import os
import shutil
import signal
import subprocess
import sys
import time

import remote_queue as rq
import multicell_expanded as mx

PIDF = rq.REPO / "remote_queue_cohesin.pid"
rq.PIDF = PIDF
AMLLOG = rq.LOG
OPS = rq.HM / "cohesin_fix"
JOBS = [("all12cohfix_refres_drop", 25000), ("all12cohfix_refres_drop", 10000),
        ("all12cohfixna_refres_drop", 25000), ("all12cohfixna_refres_drop", 10000)]
MODELS = [mx.run_name(b, r) for b, r in JOBS]


def trained(name):
    si = mx.stop_info(name, 10)
    return bool(si and si[2] == "patience")


def inputs_ready(base, res):
    k = res // 1000
    return ((rq.HM / f"cache_expanded/mc_all12cohfix_refres_drop_{k}kb/manifest.json").exists()
            and (OPS / f"operator_{k}kb.pt").exists())


def aml_finished():
    pid = rq.REPO / "remote_queue_aml.pid"
    if pid.exists():
        try:
            os.kill(int(pid.read_text()), 0)
            return False
        except (OSError, ValueError):
            pass
    txt = AMLLOG.read_text(errors="replace")
    if "REMOTE_QUEUE_AML_DONE" in txt:
        return True
    names = [mx.run_name(f"{b}_f{f}", r) for f in (10, 11, 12)
             for b in ("all12aml_refin_drop", "all12aml_honest_drop") for r in (25000, 10000)]
    for n in names:
        if not (trained(n) or (rq.CLAIMS / f"{n}.claim").exists()):
            return False
    ev = [(rq.CLAIMS / f"aml_eval_f{f}.claim").exists() or
          (rq.REPO / f"results/aml_train/eval_f{f}_summary.csv").exists() for f in (10, 11, 12)]
    return all(ev)


def wait_free():
    quiet = None
    last = 0
    while True:
        busy = (not aml_finished()) or rq.cnn_alive() or rq.gpu_mib() >= 1024
        if busy:
            quiet = None
            if time.time() - last > 1800:
                last = time.time()
                rq.log(f"cohesin: waiting (aml_finished={aml_finished()} cnn={rq.cnn_alive()} gpu_mib={rq.gpu_mib()})")
        else:
            quiet = quiet or time.time()
            if time.time() - quiet >= 120:
                rq.log("cohesin: GPU free, AML runner finished")
                return
        time.sleep(15)


def smoke():
    import yaml
    S = rq.SMOKE
    S.mkdir(parents=True, exist_ok=True)
    cfg = mx.run_config(mx.run_name("all12cohfix_refres_drop", 25000), 100, 10, 500)
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"]) * 2
    cfg["train"].update(output_dir=str(S / "cohfix_smoke_run"), max_steps=60, save_checkpoints=False,
                        max_epochs=None, patience_epochs=None)
    cp = S / "cohfix_smoke.yaml"
    yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
    r = rq.sh([rq.PYB, "-m", "hicmultiome.train", "--config", str(cp)],
              stdout=open(S / "cohfix_smoke.log", "w"), stderr=subprocess.STDOUT, timeout=1800)
    if r.returncode or "done." not in (S / "cohfix_smoke.log").read_text(errors="replace"):
        return f"smoke failed rc={r.returncode}, log {S / 'cohfix_smoke.log'}"
    shutil.rmtree(S)
    return None


def claim(name):
    r = rq.sh([rq.PYB, "scripts/claim_job.py", "claim", name, "--dir", str(rq.CLAIMS), "--cmd",
               "remote_queue_cohesin"], capture_output=True, text=True)
    if r.returncode == 1 and rq.my_claim(name):
        rq.log(f"{name}: reusing own claim")
        return True
    if r.returncode:
        rq.log(f"skip {name}: {r.stdout.strip()} (claim exit {r.returncode})")
        return False
    return True


def run_cmd(name, cmd, out, done=lambda: True):
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
    rq.log(f"remote_queue_cohesin started pid {os.getpid()}")
    wait_free()
    err = smoke()
    if err:
        rq.log(f"COHESIN SMOKE FAILED, no job started: {err}")
        PIDF.unlink(missing_ok=True)
        sys.exit(1)
    rq.log("cohesin smoke ok")
    fails = 0
    pending = [(b, r) for b, r in JOBS if not trained(mx.run_name(b, r))]
    waited = 0
    while pending:
        job = next(((b, r) for b, r in pending if inputs_ready(b, r)), None)
        if job is None:
            if waited % 30 == 0:
                rq.log(f"cohesin: inputs missing for {[mx.run_name(b, r) for b, r in pending]}, waiting")
            waited += 1
            time.sleep(60)
            continue
        pending.remove(job)
        base, res = job
        name = mx.run_name(base, res)
        if not claim(name):
            continue
        out = mx.REF_OUT / f"{name}.remote.out"
        ok = run_cmd(name, f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} "
                           f"--res {res} --stage train --gpus 0", out, lambda n=name: trained(n))
        if ok:
            ok = run_cmd(f"{name}_eval", f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} "
                                         f"--res {res} --stage eval --gpus 0", out)
        fails = 0 if ok else fails + 1
        if fails >= 2:
            rq.log("2 consecutive failures: cohesin stopping")
            PIDF.unlink(missing_ok=True)
            return
    models = " ".join(MODELS)
    steps = [("cohfix_perturb_predict", f"{rq.PYB} scripts/perturb_validate.py --stage predict predict_chip"),
             ("cohfix_washout_predict", f"{rq.PYB} scripts/washout.py --stage predict"),
             ("cohfix_aml_predict", f"{rq.PYB} scripts/aml_eval.py --stage predict --models {models}")]
    for n, cmd in steps:
        if not claim(n):
            continue
        ok = run_cmd(n, "CUDA_VISIBLE_DEVICES= nice -n 10 " + cmd, mx.REF_OUT / f"{n}.remote.out")
        fails = 0 if ok else fails + 1
        if fails >= 2:
            rq.log("2 consecutive failures: cohesin stopping")
            break
    else:
        rq.log("REMOTE_QUEUE_COHESIN_DONE")
    PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
