"""Test-window and 413-region evaluation of the release candidates all12one_full and
all12one_fullrd (registry section 93), the same commands as for all12one (registry section 88:
all12one_eval_gpu*.log next to the runs), with and without the reference plane.

Per view <model>_<k>kb (k = 2, 5, 10, 25), on the GPUs given by --gpus (--per-gpu workers each):
  - eval_val_windows.py --split test for the 12 training cells: test_eval_<cell>.json (consensus
    as input), refalone_test_eval_<cell>.json (the consensus alone), noref_test_eval_<cell>.json
    (reference plane 0, --no-reference);
  - crosscell_predict.py --tta on the 413 Akita test regions for the 12 training cells and the held-out
    cells (k562, hmec, huvec and the deep hmec / huvec maps), every other cell's measured map as a
    baseline, the cell's methylation track when it has one. The consensus input and the plane at 0
    are two keep sets (all marks kept; `full`, `full_noref`) of ONE pass; the `full` tables are copied to
    <view>/crosscell/predict_<cell>.csv (the layout of the all12one views) and the plane-at-0 tables
    to <view>/crosscell_noref/predict_<cell>.csv.

The crosscell command lines are those of the all12one logs (cell maps, consensus, baselines) with
the view replaced. Everything is resumable (a task is skipped when its output exists).

    python scripts/one_model_release_eval.py --models all12one_full all12one_fullrd --gpus 0 1
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
REF = Path("/mnt/processed/wolffjoa/hic-multiome/runs/multicell_reference")
PY = os.environ.get("HICMULTI_PY", sys.executable)
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248", "mcf7",
           "cd14mono", "pc9"]
RES = [2, 5, 10, 25]


def view_of(model, k):
    return REF / f"{model}_{k}kb"


def template_commands():
    """(k, cell) -> argument list of the all12one crosscell_predict call, from the logs."""
    t = {}
    for f in sorted(REF.glob("all12one_eval_gpu*.log")):
        for ln in f.read_text().splitlines():
            if not ln.startswith("[cmd]") or "crosscell_predict.py" not in ln:
                continue
            args = ln.split()[2:]
            k = int(re.search(r"all12one_(\d+)kb/checkpoint-best", ln).group(1))
            t[(k, args[args.index("--cell") + 1])] = args
    return t


def crosscell_cmd(model, k, cell, args, tracks):
    import methylation as me
    view = view_of(model, k)
    a = [x.replace(f"/all12one_{k}kb/", f"/{view.name}/") for x in args]
    i = a.index("--others")
    extra = []
    if me.has_meth(cell):
        extra = ["--extra-bigwig-dir", str(me.meth_dir(cell))]
    j = a.index("--out-dir")
    a[j + 1] = str(view / "crosscell_sets")
    t = ",".join(tracks)
    return a[:i] + extra + ["--keep-sets", f"full={t}", f"full_noref={t}"] + a[i:]


def tasks(models):
    tmpl = template_commands()
    out = []
    for k in sorted(RES, key=lambda r: -1 / r):               # 2 kb first: the longest
        for m in models:
            view = view_of(m, k)
            tracks = json.loads((view / "checkpoint-best/channels.json").read_text())["tracks"]
            for (kk, cell), args in sorted(tmpl.items()):
                if kk != k:
                    continue
                done = (view / "crosscell" / f"predict_{cell}.csv").exists() and \
                       (view / "crosscell_noref" / f"predict_{cell}.csv").exists()
                if not done:
                    out.append(("crosscell", m, k, cell, crosscell_cmd(m, k, cell, args, tracks)))
            for cell in TRAIN12:
                for tag, extra in (("test_eval", []),
                                   ("refalone_test_eval", ["--reference-only"]),
                                   ("noref_test_eval", ["--no-reference"])):
                    o = view / f"{tag}_{cell}.json"
                    if not o.exists():
                        out.append(("eval", m, k, cell, [
                            str(REPO / "scripts/eval_val_windows.py"), "--run", str(view), "--split",
                            "test", "--sample", cell, *extra, "--out", str(o)]))
    return out


def run_task(t, gpu, logdir):
    kind, m, k, cell, cmd = t
    view = view_of(m, k)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    line = " ".join([PY, *cmd])
    with open(logdir / f"{m}_eval_gpu{gpu}.log", "a") as lg:
        lg.write(f"[cmd] {line}\n")
        lg.flush()
        rc = subprocess.run([PY, *cmd], stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
        if rc:
            lg.write(f"[FAILED] {m} {k}kb {kind} {cell} rc={rc}\n")
            return
    if kind == "crosscell":
        for s, d in (("full", "crosscell"), ("full_noref", "crosscell_noref")):
            src = view / "crosscell_sets" / s / f"predict_{cell}.csv"
            if src.exists():
                (view / d).mkdir(exist_ok=True)
                shutil.copyfile(src, view / d / f"predict_{cell}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["all12one_full", "all12one_fullrd"])
    ap.add_argument("--gpus", nargs="+", default=["0", "1"])
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ts = tasks(a.models)
    print(f"{len(ts)} tasks", flush=True)
    if a.dry_run:
        for t in ts[:3] + ts[-3:]:
            print(t[:4], " ".join(t[4])[:1500])
        return
    q = queue.Queue()
    for t in ts:
        q.put(t)
    logdir = REF

    def worker(g):
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            run_task(t, g, logdir)
    th = [threading.Thread(target=worker, args=(g,)) for g in a.gpus for _ in range(a.per_gpu)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    print("ONE_MODEL_RELEASE_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
