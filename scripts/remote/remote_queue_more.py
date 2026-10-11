"""Fifth remote runner: (B) honest seed replicates of the headline 12-cell models, then (A) AML variants
on the folds 10, 11, 12 (Meth and residual-reference models), registry sections 51, 52, 61.

Order: a 60-step smoke on the first B config; the CPU precompute of the A caches
(mc_all12aml_meth_ref_{25,10}kb_f{10,11,12}, nice 10, 8 workers) starts at once in the background;
B jobs in the order of the local queue (seed 1 then 2, 25 kb then 10 kb, refin_drop, refres_drop,
honest_drop; each trained and evaluated with `--stage eval`), then A: per fold all12aml_meth_refin_drop_fN
25 kb and 10 kb, all12aml_refres_drop_fN 10 kb, then `aml_train_eval.py --fold N --variant extra`
(HICMULTI_GPUS=0) with eval_fN_extra_*.csv copied to runs/aml_train_remote/fold<N>/. Every job is claimed by
run name; a job claimed elsewhere or finished is skipped; a job whose cache is not ready is retried
after the others. Stops after 2 consecutive failures. Stop: kill $(cat remote_queue_more.pid).
"""
import os
import shutil
import signal
import subprocess
import sys
import time

import remote_queue as rq
import multicell_expanded as mx

PIDF = rq.REPO / "remote_queue_more.pid"
rq.PIDF = PIDF
DEST = rq.HM / "runs/aml_train_remote"
FOLDS = (10, 11, 12)
B = [(f"{fam}_s{seed}", res) for res in (25000, 10000) for seed in (1, 2)
     for fam in ("all12refin_drop", "all12refres_drop", "all12honest_drop")]
CACHE_OF = {"all12refin_drop": "mc_all12ref_{k}kb", "all12refres_drop": "mc_all12ref_{k}kb",
            "all12honest_drop": "mc_all12_{k}kb"}
A = []
for f in FOLDS:
    A += [(f"all12aml_meth_refin_drop_f{f}", 25000), (f"all12aml_meth_refin_drop_f{f}", 10000),
          (f"all12aml_refres_drop_f{f}", 10000)]


def cache_of(base, res):
    k = res // 1000
    if base.startswith("all12aml_"):
        meth = "meth_" if "meth" in base else ""
        return f"mc_all12aml_{meth}ref_{k}kb_f{base.rsplit('_f', 1)[1]}"
    return CACHE_OF[base.rsplit("_s", 1)[0]].format(k=k)


def trained(name):
    si = mx.stop_info(name, 10)
    return bool(si and si[2] == "patience")


def ready(base, res):
    return (mx.CACHE / cache_of(base, res) / "manifest.json").exists()


def smoke(base, res, tag):
    import yaml
    S = rq.SMOKE
    S.mkdir(parents=True, exist_ok=True)
    cfg = mx.run_config(mx.run_name(base, res), 100, 10, 500)
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"]) * 2
    cfg["train"].update(output_dir=str(S / f"{tag}_run"), max_steps=60, save_checkpoints=False,
                        max_epochs=None, patience_epochs=None)
    cp = S / f"{tag}.yaml"
    yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
    r = rq.sh([rq.PYB, "-m", "hicmultiome.train", "--config", str(cp)],
              stdout=open(S / f"{tag}.log", "w"), stderr=subprocess.STDOUT, timeout=1800)
    if r.returncode or "done." not in (S / f"{tag}.log").read_text(errors="replace"):
        return f"smoke failed rc={r.returncode}, log {S / (tag + '.log')}"
    shutil.rmtree(S, ignore_errors=True)
    return None


def claim(name):
    r = rq.sh([rq.PYB, "scripts/claim_job.py", "claim", name, "--dir", str(rq.CLAIMS), "--cmd",
               "remote_queue_more"], capture_output=True, text=True)
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


def start_precompute():
    """A caches on the CPU, one after the other, nice 10, 8 workers."""
    cmds = []
    for base, res in A:
        c = cache_of(base, res)
        if "meth" in base and not (mx.CACHE / c / "manifest.json").exists():
            cmds.append(f"{rq.PYB} scripts/multicell_expanded.py --runs {base} --res {res} "
                        f"--stage precompute --workers 8")
    if cmds:
        rq.log(f"A precompute in the background: {len(cmds)} caches")
        return subprocess.Popen(["bash", "-c", " ; ".join(f"nice -n 10 {c}" for c in cmds)], cwd=rq.REPO,
                                env=rq.env, start_new_session=True,
                                stdout=open(mx.REF_OUT / "remote_more_precompute.out", "a"),
                                stderr=subprocess.STDOUT)
    return None


def copy_results(fold):
    d = DEST / f"fold{fold}"
    d.mkdir(parents=True, exist_ok=True)
    for p in (rq.REPO / "results/aml_train").glob(f"eval_f{fold}_extra_*.csv"):
        shutil.copyfile(p, d / (p.name + ".part"))
        os.replace(d / (p.name + ".part"), d / p.name)
    rq.log(f"copied eval_f{fold}_extra_* to {d}")


def wait_gpu():
    quiet = None
    while True:
        if rq.cnn_alive() or rq.gpu_mib() >= 1024:
            quiet = None
        else:
            quiet = quiet or time.time()
            if time.time() - quiet >= 120:
                return
        time.sleep(15)


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
    rq.log(f"remote_queue_more started pid {os.getpid()}")
    wait_gpu()
    err = smoke(*B[0], "more_smoke")
    if err:
        rq.log(f"MORE SMOKE FAILED, no job started: {err}")
        PIDF.unlink(missing_ok=True)
        sys.exit(1)
    rq.log("more smoke ok")
    pre = start_precompute()
    fails = 0
    a_smoke_done = False
    pending = [("B", b, r) for b, r in B] + [("A", b, r) for b, r in A]
    idle = 0
    while pending:
        job = next((j for j in pending if ready(j[1], j[2]) and not (j[0] == "A" and False)), None)
        if job is None:
            if pre is not None and pre.poll() is not None:
                rq.log(f"inputs missing for {[mx.run_name(b, r) for _, b, r in pending]} and precompute ended")
                break
            idle += 1
            time.sleep(60)
            continue
        pending.remove(job)
        kind, base, res = job
        name = mx.run_name(base, res)
        if trained(name) and kind == "B" and (mx.REF_OUT / name / "test_eval_gm12878.json").exists():
            rq.log(f"skip {name}: finished")
            continue
        if kind == "A" and not a_smoke_done:
            err = smoke(base, res, "more_smoke_a")
            if err:
                rq.log(f"A SMOKE FAILED, A jobs skipped: {err}")
                pending = [j for j in pending if j[0] == "B"]
                continue
            a_smoke_done = True
            rq.log("A smoke ok")
        if not trained(name):
            if not claim(name):
                continue
            out = mx.REF_OUT / f"{name}.remote.out"
            ok = run_cmd(name, f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} "
                               f"--res {res} --stage train --gpus 0", out, lambda n=name: trained(n))
        else:
            ok = True
            out = mx.REF_OUT / f"{name}.remote.out"
        if ok and kind == "B":
            ok = run_cmd(f"{name}_eval", f"{rq.PYB} scripts/run_single_gpu.py multicell_expanded --runs {base} "
                                         f"--res {res} --stage eval --gpus 0", out)
        fails = 0 if ok else fails + 1
        if fails >= 2:
            rq.log("2 consecutive failures: more stopping")
            PIDF.unlink(missing_ok=True)
            return
        if kind == "A":
            f = int(base.rsplit("_f", 1)[1])
            names = [mx.run_name(b, r) for b, r in A if b.endswith(f"_f{f}")]
            if all(trained(n) for n in names):
                ev = f"aml_extra_eval_f{f}"
                done = lambda f=f: (rq.REPO / f"results/aml_train/eval_f{f}_extra_summary.csv").exists()
                if not done() and claim(ev):
                    ok = run_cmd(ev, f"HICMULTI_GPUS=0 {rq.PYB} scripts/aml_train_eval.py --fold {f} --variant extra",
                                 DEST / f"eval_f{f}_extra.remote.out", done)
                    fails = 0 if ok else fails + 1
                    if fails >= 2:
                        rq.log("2 consecutive failures: more stopping")
                        PIDF.unlink(missing_ok=True)
                        return
                if done():
                    copy_results(f)
    rq.log("REMOTE_QUEUE_MORE_DONE")
    PIDF.unlink(missing_ok=True)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    main()
