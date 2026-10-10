"""CPU preparation of the AML cross-validation folds 1 and 2 (same commands and outputs as
gpu_scheduler.add_aml_and_chrho): references, window caches, hard-split windows."""
import os, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
import aml_train as at
import multicell_expanded as mx
PYB = str(Path.home() / "hicenv/bin/python")
env = dict(os.environ, HICMULTI_PY=PYB)
LOG = REPO / "aml_prep.log"
def log(m):
    with open(LOG, "a") as f: f.write(f"{time.strftime('%F %T')} {m}\n")
def run(cmd):
    log("run " + cmd)
    r = subprocess.run(f"nice -n 10 {cmd}", shell=True, cwd=REPO, env=env)
    log(f"rc={r.returncode}")
    return r.returncode == 0
for fold in (10, 11, 12):
    sfx = f"_f{fold}"
    for res in (25000, 10000):
        k = res // 1000
        if not at.refs_done(fold, res):
            if not run(f"{PYB} scripts/aml_train.py --stage split refs --fold {fold} --res {res}"): sys.exit(1)
        for r, cache in ((f"all12aml_refin_drop{sfx}", f"mc_all12aml_ref_{k}kb{sfx}"),
                         (f"all12aml_honest_drop{sfx}", f"mc_all12aml_{k}kb{sfx}")):
            if not (mx.CACHE / cache / "manifest.json").exists():
                if not run(f"{PYB} scripts/multicell_expanded.py --runs {r} --res {res} --stage precompute --workers 16"): sys.exit(1)
        if not at.hard_split_path(fold, res).exists():
            if not run(f"{PYB} scripts/aml_train.py --stage hard --fold {fold} --res {res}"): sys.exit(1)
log("AML_PREP_DONE")
