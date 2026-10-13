"""Window-prior models on the four independent validation samples (registry section 76).

PC-3, Panc1, NK, Caco-2 (scripts/independent_validation.py; seven marks, 10 / 25 kb O/E targets): the wp
models predict the 413 Akita test windows with the prior cooler of each sample as reference (wp_ref,
wp_self) or prior plane (wp_both), the existing predictions of honest / refres / refin / diff are read from
validation_new/pred/, every predictor is scored on identical pixels with the measures of
independent_validation.py (_score_chunk).

    python scripts/window_prior_indep.py predict --res 10000 --gpu 0
    python scripts/window_prior_indep.py score --res 10000 --procs 6
    python scripts/window_prior_indep.py summary --res 10000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import independent_validation as iv  # noqa: E402

HM, VN = iv.HM, iv.VN
WP = HM / "window_prior"
OUTP = VN / "pred_wp"
WPM = {"wp_ref": "all12wp_ref_drop", "wp_self": "all12wp_self_drop", "wp_both": "all12wp_both_drop",
       "wp_refm": "all12wp_refm_drop"}
TAG = {"wp_ref": "ref", "wp_self": "own", "wp_both": "ref", "wp_refm": "ref_motif"}
OLD = ["honest", "refres", "refin", "diff"]
CELLS = ["pc3", "panc1", "nk", "caco2"]
RESULTS = REPO / "results/window_prior"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def stage_predict(a):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    py = os.environ.get("HICMULTI_PY", sys.executable)
    for res in a.res:
        k = res // 1000
        regs = VN / "regions" / f"crop_{k}kb_test.tsv"
        for cell in a.cells:
            bw, assay = iv.CELLS[cell]
            lb = Path(os.environ.get("IV_BW", str(Path.home() / "scratch/iv_bw"))) / cell
            if (lb / "COPIED").exists():
                bw = lb
            have = {t for t in iv.CORE7 if (bw / f"{t}.bigwig").exists()}
            for m in a.models:
                run = HM / "runs/multicell_reference" / f"{WPM[m]}_{k}kb"
                ck = run / "checkpoint-best"
                if not (ck / "model.pt").exists():
                    log(f"[skip] {run.name}: no checkpoint")
                    continue
                pcool = WP / f"{k}kb" / TAG[m] / "cools" / f"{cell}.cool"
                if not pcool.exists():
                    log(f"[skip] {m} {cell}: no prior cooler {pcool}")
                    continue
                out = OUTP / f"{k}kb" / m / cell
                if (out / "DONE").exists():
                    continue
                job = f"windowprior_iv_{cell}_{m}_{k}kb"
                rc = subprocess.run([sys.executable, str(REPO / "scripts/claim_job.py"), "claim", job,
                                     "--done-path", str(out / "DONE"), "--cmd", "window_prior_indep predict"]).returncode
                if rc != 0:
                    log(f"[claim] {job} rc={rc}")
                    continue
                out.mkdir(parents=True, exist_ok=True)
                cmd = [py, "scripts/independent_validation_predict.py", "--config", str(ck / "config.yaml"),
                       "--checkpoint", str(ck), "--cell", cell, "--bigwig-dir", str(bw),
                       "--target", str(iv.cool_of(cell, res)), "--tta", "--assay", assay, "--binsize", str(res),
                       "--device", a.device, "--regions", str(regs), "--out-dir", str(out),
                       "--save-pred", str(out / "{set}.npz"),
                       "--keep-sets", f"all={','.join(t for t in iv.CORE7 if t in have)}"]
                if m == "wp_both":                       # positional consensus stays the reference
                    cmd += ["--reference", str(iv.OE / f"consensus12_proper_{k}kb.cool"), "--prior", str(pcool)]
                else:
                    cmd += ["--reference", str(pcool)]
                if m in ("wp_ref", "wp_both", "wp_refm"):
                    cmd += ["--diff-group", "cell"]
                log(f"[run] {job}")
                with open(out / "predict.log", "w") as fh:
                    fh.write("[cmd] " + " ".join(cmd) + "\n")
                    fh.flush()
                    r = subprocess.run(["nice", "-n", "10"] + cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=env)
                if r.returncode == 0 and (out / "all.npz").exists():
                    (out / "DONE").write_text(time.strftime("%F %T"))
                else:
                    log(f"[FAILED] {job} rc={r.returncode}")
                    subprocess.run([sys.executable, str(REPO / "scripts/claim_job.py"), "release", job])


def stage_score(a):
    for res in a.res:
        k = res // 1000
        A = iv.aligned(pd.read_csv(VN / "regions" / "sequence_regions.tsv", sep="\t"), res).drop_duplicates("key")
        A = A[A.set == "test"]
        for cell in a.cells:
            outf = RESULTS / f"indep_scores_{cell}_{k}kb.csv.gz"
            if outf.exists():
                log(f"[have] {outf.name}")
                continue
            methods = {}
            for m in OLD:
                f = VN / "pred" / f"{k}kb" / m / cell / "all.npz"
                if f.exists() and (f.parent / "DONE").exists():
                    methods[f"{m}:all"] = [f]
            for m in WPM:
                f = OUTP / f"{k}kb" / m / cell / "all.npz"
                if f.exists() and (f.parent / "DONE").exists():
                    methods[f"{m}:all"] = [f]
            if not any(x.startswith("wp_") for x in methods):
                log(f"[skip] {cell} {k}kb: no wp predictions")
                continue
            rows = list(zip(A.key, A.chrom, A.b_lo, A.b_hi, A.seq_key, A.set))
            chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
            log(f"score {cell} {k}kb: {len(rows)} windows, methods {sorted(methods)}")
            res_l = iv._run_pool(a, cell, res, chunks, methods, [], [], False)
            df = pd.DataFrame(res_l)
            outf.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outf, index=False, float_format="%.5g")
            log(df.groupby("method").pearson.median().sort_values(ascending=False).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["predict", "score"])
    ap.add_argument("--res", nargs="+", type=int, default=[10000])
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--models", nargs="+", default=["wp_ref", "wp_self", "wp_both"], choices=list(WPM))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    {"predict": stage_predict, "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
