"""Extra GPU queue behind scripts/gpu_scheduler.py (runs/coarse_gap/QUEUE.md).

A separate process that never touches the scheduler: it imports gpu_scheduler
for its Job class, its trained() / eval_cmd() / marker helpers and its
read-only job list, and waits until the scheduler's GPUs are free. Its jobs run
one GPU training at a time, in this order:

  a  all12refres_drop_motifmeth 25 kb, 10 kb (methylation-aware CTCF motif
     planes, registry section 43) with the scheduler's standard evaluation
  b  scenario benchmark without the consensus (scripts/scenario_bench.py,
     registry section 44): base models all12honest_drop_meth (2 seeds, 25 and
     10 kb) with inference-only sets, then dedicated trainings per set
  c  all12aml_refres_drop 10 kb: AML-inclusive recipe of section 36 with the
     reference as residual (patient minus AML reference), fold 0
  f  cohesin fix: all12cohfix_refres_drop and the ablation all12cohfixna_refres_drop, 25 and 10 kb
     (registry 55), claimed by run name, with evaluation, perturbation, washout, AML and diagnosis
  g  honest seed replicates (seeds 1, 2) of all12refin_drop, all12refres_drop, all12honest_drop,
     25 kb then 10 kb, claimed by run name, standard evaluation (registry 51 audit)
  h  differential-channel 12-cell family: all12diffcells_refres_drop seeds s1, s2 at 10 kb, the run and
     seeds at 25 kb (registry 66), claimed by run name, standard evaluation, reference summary refresh
  d  (removed 2026-10-09: the sequence CNN runs on another computer)
  e  chromosome-held-out 5 kb models all12refin_drop_chrho_5kb, all12honest_drop_chrho_5kb and
     (optional, last) all12refres_drop_chrho_5kb with the standard evaluation (registry 58),
     claimed like the dedicated trainings, started once their caches exist

CPU preparation (the six window caches) starts at once
at nice 19. Every job is skipped when its output exists, so the queue can be
stopped and started again; an interrupted training restarts from scratch.

GPU start condition (either one):
  --wait-pid PID      the scheduler process has exited
  --idle-minutes N    the scheduler has run no GPU job and has no GPU job ready
                      for N minutes, and runs/multicell_reference/AML_CV does
                      not exist (the scheduler never exits while the AML_CV
                      jobs of folds 1 and 2 wait for that flag)

Reload: `touch runs/multicell_reference/.extra_reload`; when no job of the queue is running it
re-executes itself (same PID, same arguments) and reads the current code and job list.

Claims: every dedicated scenario training is claimed on the shared drive
(scripts/claim_job.py, runs/multicell_reference/claims/<run>.claim) before a local start. A job
claimed by another host is skipped and re-checked at every poll until its output exists or the
claim is released; a local failure releases the claim, a success keeps it.

Stop: kill <PID in runs/multicell_reference/.extra_queue.pid> (SIGTERM; the
queue then ends its running jobs' process groups and exits). Nothing here uses
killall, pkill or the scheduler's pid files for anything but reading.

    python scripts/gpu_queue_extra.py --wait-pid 1902496 --idle-minutes 60
    python scripts/gpu_queue_extra.py --list
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claim_job as cj                                               # noqa: E402
import gpu_scheduler as gs                                           # noqa: E402
import multicell_expanded as mx                                      # noqa: E402

REPO, PY, MR = mx.REPO, mx.PY, mx.REF_OUT
LOG = MR / "extra_queue.log"
OUTLOG = MR / "extra_queue.out"
PIDFILE = MR / ".extra_queue.pid"
RELOAD = MR / ".extra_reload"
M = gs.M
SB = f"{PY} scripts/scenario_bench.py"
NICE = "nice -n 19"
CNN_DATA = Path.home() / "data/seq_cnn_data"      # symlinks to the data folders, see QUEUE.md
CNN_OUT = mx.HM / "runs/seq_cnn"
CNN_MODES = ("seq", "seq_meth", "meth")
CPU_SLOTS = 2
# 5 kb chromosome-held-out models: (run, cache); refres is optional and last
CHRHO5 = [("all12refin_drop_chrho", "mc_all12ref_chrho_5kb"),
          ("all12honest_drop_chrho", "mc_all12chrho_5kb"),
          ("all12refres_drop_chrho", "mc_all12ref_chrho_5kb")]


def note(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def X(name) -> str:
    return "x_" + name


def marked(name) -> bool:
    return gs.is_marked(X(name))


def cache_ok(name) -> bool:
    return gs.cache_ok(name)


def jobs():
    import scenario_bench as sb
    J = []

    def cpu(name, cmd, ready, done, lane=None):
        j = gs.Job(name, f"{NICE} {cmd}", ready, done)
        j.lane = lane            # jobs of one lane never run together (shared motif-scan cache)
        J.append(j)

    def gpu(name, cmd, ready, done):
        J.append(gs.Job(name, cmd, ready, done, gpu=True))

    def aux(name, cmd, ready, gpu_use=True):
        j = gs.Job(name, cmd, ready, (lambda n=name: marked(n)))
        j.uses_gpu = gpu_use
        J.append(j)

    # ---- CPU preparation (starts at once)
    cpg_ok = lambda: all((me_dir / "CpG.npz").exists() for me_dir in cpg_dirs())
    for run, res, cache in (("all12refres_drop_motifmeth", 25000, "mc_all12refres_drop_motifmeth_25kb"),
                            ("all12refres_drop_motifmeth", 10000, "mc_all12refres_drop_motifmeth_10kb"),
                            ("all12honest_drop_meth", 25000, "mc_all12honest_drop_meth_25kb"),
                            ("all12honest_drop_meth", 10000, "mc_all12honest_drop_meth_10kb"),
                            ("all12honest_drop_motifmeth", 25000, "mc_all12honest_drop_motifmeth_25kb"),
                            ("all12honest_drop_motifmeth", 10000, "mc_all12honest_drop_motifmeth_10kb")):
        cpu(f"cache_{cache}", f"{M} --runs {run} --res {res} --stage precompute --workers 3",
            cpg_ok, (lambda c=cache: cache_ok(c)), lane="cache")

    def train(run, res, cache):
        gpu(f"train_{mx.run_name(run, res)}",
            f"{M} --runs {run} --res {res} --stage precompute --workers 3 && "
            f"{M} --runs {run} --res {res} --stage train --gpus 0,1",
            (lambda c=cache: cache_ok(c)), (lambda r=run, k=res: gs.trained(r, k)))

    # ---- (a) methylation-aware motif planes
    for res in (25000, 10000):
        cache = f"mc_all12refres_drop_motifmeth_{res // 1000}kb"
        train("all12refres_drop_motifmeth", res, cache)
        n = mx.run_name("all12refres_drop_motifmeth", res)
        aux(f"eval_{n}", f"{gs.eval_cmd('all12refres_drop_motifmeth', res)} && "
                         f"( {gs.STATUS} || true )",
            (lambda r=res: gs.trained("all12refres_drop_motifmeth", r)))

    # ---- (b) scenario benchmark without the consensus
    aux("scen_configs", f"{SB} --stage configs", lambda: True, gpu_use=False)
    for res in (25000, 10000):
        cache = f"mc_all12honest_drop_meth_{res // 1000}kb"
        for seed in (0, 1):
            run = "all12honest_drop_meth" + ("_s1" if seed else "")
            train(run, res, cache)
            aux(f"scen_infer_{res // 1000}kb_s{seed}",
                f"{SB} --stage infer --res {res} --seed {seed} --gpu {seed}",
                (lambda r=run, k=res: gs.trained(r, k)))
    # base models with the motif_meth planes (registry 44, part 2): inference-only rows crossing the
    # planes with a missing CTCF track
    for res, seed in sb.MM_RUNS:
        run = "all12honest_drop_motifmeth" + ("_s1" if seed else "")
        train(run, res, f"mc_all12honest_drop_motifmeth_{res // 1000}kb")
        aux(f"scen_infer_mm_{res // 1000}kb_s{seed}",
            f"{SB} --stage infer_mm --res {res} --seed {seed} --gpu {seed}",
            (lambda r=run, k=res: gs.trained(r, k)))
    for run_name in sb.dedicated_runs():
        setname, res, seed = sb.parse_name(run_name)
        gpu(f"train_{run_name}", f"{SB} --stage configs && {SB} --stage train --run {run_name} --gpus 0,1",
            (lambda r=res, s=seed: gs.trained("all12honest_drop_meth" + ("_s1" if s else ""), r)),
            (lambda n=run_name: sb.dedicated_done(n)))
        J[-1].claim = run_name      # claimed on the shared drive before a local start (claim_job.py)
        aux(f"scen_ded_{run_name}", f"{SB} --stage dedicated --run {run_name} --gpu {seed}",
            (lambda n=run_name: sb.dedicated_done(n)))
    aux("scen_summary", f"{SB} --stage summary",
        lambda: all(sb.dedicated_done(n) for n in sb.dedicated_runs())
        and all(marked(f"scen_infer_{k}kb_s{s}") for k in (25, 10) for s in (0, 1))
        and all(marked(f"scen_infer_mm_{r // 1000}kb_s{s}") for r, s in sb.MM_RUNS), gpu_use=False)

    # ---- (c) AML residual-only training, fold 0, 10 kb
    import aml_train as at
    train("all12aml_refres_drop", 10000, "mc_all12aml_ref_10kb")
    J[-1].ready = (lambda b=J[-1].ready: b() and at.hard_split_path(0, 10000).exists())
    aux("aml_refres_eval_f0", gs.locked("amleval", f"{PY} scripts/aml_train_eval.py --fold 0"),
        lambda: gs.trained("all12aml_refres_drop", 10000))

    # ---- (e) chromosome-held-out 5 kb models for the competitor comparison (registry 58,
    # results/benchmark/REMOTE_JOB_chrho_5kb.md), last, claimed like the dedicated trainings. The
    # caches are built by the competitor agent's own precompute (nice 19): wait for the manifests,
    # never start a second precompute (the train stage skips it when the manifest exists).
    for run, cache in CHRHO5:
        gpu(f"train_{mx.run_name(run, 5000)}",
            f"{M} --runs {run} --res 5000 --stage precompute --workers 3 && "
            f"{M} --runs {run} --res 5000 --stage train --gpus 0,1",
            (lambda c=cache: cache_ok(c)), (lambda r=run: gs.trained(r, 5000)))
        J[-1].claim = mx.run_name(run, 5000)
        J[-1].external_wait = True
        aux(f"eval_{mx.run_name(run, 5000)}", f"{gs.eval_cmd(run, 5000)} && ( {gs.STATUS} || true )",
            (lambda r=run: gs.trained(r, 5000)))
    # ---- (f) cohesin-fraction input and cohesin-loss augmentation (registry 55,
    # results/cohesin_fix/REMOTE_JOB.md): main run and no-augmentation ablation, 25 and 10 kb, claimed
    # by run name (the remote runner remote_queue_cohesin.py claims the same names), one cache
    # (mc_all12cohfix_refres_drop_<k>kb, built by the cohesin agent: wait for the manifest)
    coh_done = {}
    for fam in ("all12cohfix_refres_drop", "all12cohfixna_refres_drop"):
        for res in (25000, 10000):
            k = res // 1000
            op = mx.HM / f"cohesin_fix/operator_{k}kb.pt"
            gpu(f"train_{mx.run_name(fam, res)}",
                f"{M} --runs {fam} --res {res} --stage precompute --workers 3 && "
                f"{M} --runs {fam} --res {res} --stage train --gpus 0,1",
                (lambda c=f"mc_all12cohfix_refres_drop_{k}kb", o=op: cache_ok(c) and o.exists()),
                (lambda r=fam, x=res: gs.trained(r, x)))
            J[-1].claim = mx.run_name(fam, res)
            J[-1].external_wait = True
            aux(f"eval_{mx.run_name(fam, res)}", f"{gs.eval_cmd(fam, res)} && ( {gs.STATUS} || true )",
                (lambda r=fam, x=res: gs.trained(r, x)))
            aux(f"diagnose_{mx.run_name(fam, res)}",
                f"{PY} scripts/cohesin_fix_diagnose.py --run {mx.run_name(fam, res)}",
                (lambda r=fam, x=res: gs.trained(r, x)), gpu_use=False)
    names = [mx.run_name(f, r) for f in ("all12cohfix_refres_drop", "all12cohfixna_refres_drop")
             for r in (10000, 25000)]
    coh_ready = lambda: all(gs.trained(*mx.split_name(n)) for n in names)
    aux("cohfix_perturb", gs.locked("perturb", f"{PY} scripts/perturb_validate.py --stage predict "
                                    "predict_chip score"), coh_ready, gpu_use=False)
    aux("cohfix_washout", f"CUDA_VISIBLE_DEVICES=1 {PY} scripts/washout.py --stage predict score",
        coh_ready, gpu_use=False)
    aux("cohfix_aml", gs.locked("aml", f"{PY} scripts/aml_eval.py --stage predict --models "
                                f"{' '.join(names)} --jobs 2 --threads 3 && {PY} scripts/aml_eval.py "
                                "--stage perturb --res 10000 25000"), coh_ready, gpu_use=False)
    # ---- (g) honest seed replicates of the headline 12-cell models (registry 51 audit): seeds 1 and
    # 2 of all12refin_drop, all12refres_drop, all12honest_drop, 25 kb first, then 10 kb; same recipe
    # and caches as the seed-0 runs; claimed by run name (the remote runner takes jobs from the same
    # list through the same claims), standard evaluation after each (reference_summary.csv)
    for res in (25000, 10000):
        for seed in (1, 2):
            for fam, cache in (("all12refin_drop", "mc_all12ref_{k}kb"),
                               ("all12refres_drop", "mc_all12ref_{k}kb"),
                               ("all12honest_drop", "mc_all12_{k}kb")):
                run = f"{fam}_s{seed}"
                train(run, res, cache.format(k=res // 1000))
                J[-1].claim = mx.run_name(run, res)
                aux(f"eval_{mx.run_name(run, res)}", f"{gs.eval_cmd(run, res)} && ( {gs.STATUS} || true )",
                    (lambda r=run, x=res: gs.trained(r, x)))
    # ---- (h) differential-channel 12-cell family (registry 66, results/diff_marks/REMOTE_JOB.md):
    # seed replicates of all12diffcells_refres_drop at 10 kb first (cache exists), then the 25 kb
    # run (its cache is built here on the CPU, nice 19, 12 workers), then the 25 kb replicates.
    # Claims by run name; standard evaluation after each; the reference summary is refreshed at the end.
    fam = "all12diffcells_refres_drop"
    cpu("cache_mc_all12ref_diff_25kb", f"{M} --runs {fam} --res 25000 --stage precompute --workers 12",
        lambda: True, (lambda: cache_ok("mc_all12ref_diff_25kb")), lane="cache")
    diff_runs = [(f"{fam}_s1", 10000), (f"{fam}_s2", 10000), (fam, 25000),
                 (f"{fam}_s1", 25000), (f"{fam}_s2", 25000)]
    for run, res in diff_runs:
        train(run, res, f"mc_all12ref_diff_{res // 1000}kb")
        J[-1].claim = mx.run_name(run, res)
        aux(f"eval_{mx.run_name(run, res)}", f"{gs.eval_cmd(run, res)} && ( {gs.STATUS} || true )",
            (lambda r=run, x=res: gs.trained(r, x)))
    need = [mx.run_name(r, k) for r, k in diff_runs] + [mx.run_name(fam, 10000)]
    chk = ("import pandas as pd,sys; d=pd.read_csv(%r); m=[n for n in %r if n not in set(d.run)]; "
           "print('reference_summary rows missing for', m); sys.exit(1 if m else 0)"
           % (str(MR / "reference_summary.csv"), need))
    aux("diffcells_refsummary", f"{M} --stage refsummary && {PY} -c {shlex.quote(chk)}",
        lambda: all(marked(f"eval_{mx.run_name(r, k)}") for r, k in diff_runs), gpu_use=False)
    # (d) the sequence CNN runs on another computer (2026-10-09), not in this queue;
    # cnn_cmd() below still builds the one-command local run if it is needed again
    return J


def cpg_dirs():
    import methylation as me
    return [me.meth_dir(c) for c in me.FILES]


def cnn_data_ok() -> bool:
    d = CNN_DATA
    cells = ("gm12878 imr90 h1 hepg2 a549 gm23248 cd14mono k562 hmec").split()
    return ((d / "genome/hg38.fa").exists() and (d / "genome/hg38.fa.fai").exists()
            and all((d / f"hg38_{c}").is_dir() and (d / f"meth/{c}/Meth.bigwig").exists()
                    for c in cells))


def cnn_cmd(what: str) -> str:
    env = f"DATA_ROOT={CNN_DATA} OUT_ROOT={CNN_OUT} PYTHON={PY} CUDA_VISIBLE_DEVICES=0"
    if what == "cache":
        return (f"env {env} {PY} scripts/seq_cnn_chip.py --stage cache --mode seq "
                f"--data-root {CNN_DATA} --out-root {CNN_OUT} --build-workers 3")
    return f"env {env} bash scripts/run_seq_cnn_remote.sh"


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def scheduler_gpu_busy() -> bool:
    """True when the scheduler runs a GPU job or has one ready (read-only)."""
    for pf in MR.glob(".run_train_*.pid"):
        try:
            if alive(pf.read_text().strip()):
                return True
        except OSError:
            return True
    import yaml
    real = yaml.safe_dump
    yaml.safe_dump = lambda *a, **k: None       # jobs() writes stage configs: not from here
    try:
        return any(j.gpu and not j.done() and j.ready() for j in gs.jobs())
    except Exception as e:                      # unknown: assume busy
        note(f"scheduler state unreadable ({e!r}), assuming busy")
        return True
    finally:
        yaml.safe_dump = real


class Gate:
    """Opens when the scheduler pid is gone, or after N idle minutes."""

    def __init__(self, wait_pids, idle_min):
        self.pids, self.idle_min = wait_pids, idle_min
        self.idle_since = None
        self.last_check = 0.0
        self.last_note = 0.0
        self.open_ = False

    def __call__(self) -> bool:
        if self.open_:
            return True
        if not any(alive(p) for p in self.pids):
            note(f"gate open: scheduler pid(s) {self.pids} have exited")
            self.open_ = True
            return True
        if not self.idle_min or time.time() - self.last_check < 120:
            return False
        self.last_check = time.time()
        if (MR / "AML_CV").exists() or scheduler_gpu_busy():
            self.idle_since = None
        else:
            self.idle_since = self.idle_since or time.time()
            if time.time() - self.idle_since >= 60 * self.idle_min:
                note(f"gate open: scheduler idle on the GPUs for {self.idle_min} min "
                     "(it is still alive, waiting for the AML_CV flag)")
                self.open_ = True
                return True
        if time.time() - self.last_note > 3600:
            self.last_note = time.time()
            note("waiting for the scheduler: " + (
                "idle since " + time.strftime("%H:%M", time.localtime(self.idle_since))
                if self.idle_since else "its GPU work is not finished"))
        return False


CLAIMS = cj.DEFAULT_DIR


def claimed_elsewhere(job, claims_dir=None) -> bool:
    """The job's claim is held by another host: it runs (or ran) there."""
    run = getattr(job, "claim", None)
    if not run:
        return False
    c = cj.read_claim(Path(claims_dir or CLAIMS), run)
    return c is not None and c.get("host") != cj.host()


def try_claim(job, claims_dir=None) -> bool:
    """Claim the job's run name for this host. True: start it. A claim this host
    already holds (an earlier run of this queue that is not running now) counts as ours."""
    run = getattr(job, "claim", None)
    if not run:
        return True
    d = Path(claims_dir or CLAIMS)
    if cj.claim(d, run, None, f"gpu_queue_extra {job.name}") == 0:
        return True
    c = cj.read_claim(d, run)
    return c is not None and c.get("host") == cj.host()


def release_claim(job, claims_dir=None):
    run = getattr(job, "claim", None)
    if run:
        cj.release(Path(claims_dir or CLAIMS), run)


def launch(job):
    note(f"start {job.name}")
    wrapped = (f"cd {REPO} && ( {job.cmd} ) >> {OUTLOG} 2>&1; rc=$?; "
               f"[ $rc -eq 0 ] && touch {gs.mark(X(job.name))}; exit $rc")
    p = subprocess.Popen(["bash", "-c", wrapped], start_new_session=True)
    (MR / f".extra_run_{job.name}.pid").write_text(str(p.pid))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait-pid", type=int, nargs="*", default=[])
    ap.add_argument("--idle-minutes", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="print the job list and exit")
    a = ap.parse_args()
    all_jobs = jobs()
    if a.list:
        for j in all_jobs:
            state = "done" if (j.done() or marked(j.name)) else (
                "ready" if j.ready() else "wait")
            print(f"{'GPU' if j.gpu else 'aux'} {j.name:55s} {state}")
        return
    if (PIDFILE.exists() and PIDFILE.read_text().strip() != str(os.getpid())
            and alive(PIDFILE.read_text().strip() or 0)):
        raise SystemExit(f"already running: pid {PIDFILE.read_text().strip()}")
    PIDFILE.write_text(str(os.getpid()))
    running: dict[str, subprocess.Popen] = {}
    failed: set[str] = set()

    def stop(signum, frame):
        note(f"signal {signum}: ending {list(running)} and exiting")
        for p in running.values():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass
        for n in running:
            (MR / f".extra_run_{n}.pid").unlink(missing_ok=True)
        PIDFILE.unlink(missing_ok=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    gate = Gate(a.wait_pid, a.idle_minutes)
    note(f"gpu_queue_extra started (pid {os.getpid()}), waiting for pid(s) {a.wait_pid}, "
         f"idle minutes {a.idle_minutes}; {len(all_jobs)} jobs")
    stalled = 0
    while True:
        for n, p in list(running.items()):
            if p.poll() is not None:
                note(f"{'done' if p.returncode == 0 else 'FAILED'} {n}")
                (MR / f".extra_run_{n}.pid").unlink(missing_ok=True)
                if p.returncode != 0:
                    failed.add(n)
                    release_claim(next(j for j in all_jobs if j.name == n))   # failure only
                del running[n]
        pending = [j for j in all_jobs if j.name not in running and j.name not in failed
                   and not j.done() and not marked(j.name)]
        if RELOAD.exists() and not running:      # touch RELOAD while idle: same PID, new job list
            RELOAD.unlink()
            note("reload requested: re-executing with the current code and job list")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        if not pending and not running:
            note("EXTRA_QUEUE_DONE")
            break
        gpu_busy = any(j.gpu for j in all_jobs if j.name in running)
        opened = gate()
        # strict order; a job claimed by another host is running elsewhere: skip it, re-check later
        gpu_pending = [j for j in pending if j.gpu]
        elsewhere = [j for j in gpu_pending if claimed_elsewhere(j)]
        head = next((j for j in gpu_pending if j not in elsewhere), None)
        if opened and not gpu_busy and head is not None and head.ready():
            if try_claim(head):
                running[head.name] = launch(head)
            else:
                note(f"{head.name}: claimed elsewhere just now, skipped")
        cpu_running = sum(1 for j in all_jobs if j.name in running and not j.gpu)
        for j in pending:
            if cpu_running >= CPU_SLOTS:
                break
            if j.gpu or j.name in running or not j.ready():
                continue
            if getattr(j, "uses_gpu", False) and not opened:
                continue
            lane = getattr(j, "lane", None)
            if lane and any(getattr(k, "lane", None) == lane and k.name in running for k in all_jobs):
                continue
            running[j.name] = launch(j)
            cpu_running += 1
        # nothing can start and nothing runs although the gate is open: blocked by failures
        ext = [j for j in pending if getattr(j, "external_wait", False) and not j.ready()]
        if opened and not running and not elsewhere and not ext and not any(
                j.ready() for j in pending if not j.gpu or j is head):
            stalled += 1
            if stalled > 10:
                note("EXTRA_QUEUE_STALLED, blocked: " + ", ".join(j.name for j in pending[:20])
                     + (f"; failed: {sorted(failed)}" if failed else ""))
                break
        else:
            stalled = 0
        time.sleep(30)
    PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
