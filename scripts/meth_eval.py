"""Mean DNA methylation as an input (registry section 27): evaluation.

The model all12refres_drop_meth_10kb (scripts/multicell_expanded.py; the
14th track Meth from scripts/methylation.py) against all12refres_drop_10kb,
the same model without the channel. Both are predicted (crosscell_predict,
consensus reference, TTA) with keep sets, every other track at its absent
value:

  full            the 13 ChIP tracks + Meth
  full_nometh     the 13 ChIP tracks
  patient         CTCF, H3K27ac, H3K27me3 + Meth (the marks of the Xu 2022
                  primary AML samples, which also have WGBS)
  patient_nometh  CTCF, H3K27ac, H3K27me3

Cells: the held-out K562 and HMEC (deep map) and the training cells GM12878
and IMR-90 on their test regions, all with WGBS.

    python scripts/meth_eval.py --stage predict summary
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

OUT = mx.HM / "runs/meth_eval"
RES_OUT = REPO / "results/methylation"
PY = sys.executable
THREADS = int(os.environ.get("METH_THREADS", "4"))
PATIENT = ["CTCF", "H3K27ac", "H3K27me3"]
SETS = {"full": mx.ALL13 + [me.TRACK], "full_nometh": list(mx.ALL13),
        "patient": PATIENT + [me.TRACK], "patient_nometh": list(PATIENT)}
MODELS = {"all12refres_drop_meth_10kb": list(SETS),
          "all12refres_drop_10kb": ["full_nometh", "patient_nometh"]}
CELLS = ["k562", "hmec_deep", "gm12878", "imr90"]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "meth_eval.log", "a") as fh:
        fh.write(line + "\n")


def predict(model: str, cell: str) -> int:
    run = mx.out_of(model) / model
    ck = mx.best_ckpt(run)
    out = OUT / model
    sets = MODELS[model]
    if all((out / s / f"predict_{cell}.csv").exists() for s in sets):
        return 0
    res = mx.split_name(model)[1]
    _, bw, assay, _ = mx.CELLS[cell]
    ref = mx.reference_of(model, cell)
    tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
           "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
           "--target", mx.target(cell, res), "--tta", "--assay", assay, "--binsize", res,
           "--device", "cpu", "--out-dir", out, "--reference", ref,
           "--keep-sets", *[f"{s}={','.join(SETS[s])}" for s in sets],
           "--others", f"{tag}={ref}"]
    if me.has_meth(cell):
        cmd += ["--extra-bigwig-dir", me.meth_dir(cell)]
    n = str(THREADS)
    env = dict(os.environ, OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n,
               CUDA_VISIBLE_DEVICES="")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"predict_{cell}.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def stage_predict():
    for model in MODELS:
        for cell in CELLS:
            log(f"predict {model} {cell} rc={predict(model, cell)}")


def stage_summary() -> pd.DataFrame:
    rows = []
    for model, sets in MODELS.items():
        for s in sets:
            for cell in CELLS:
                f = OUT / model / s / f"predict_{cell}.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                cons = [c for c in df.columns if c.startswith("base_consensus")]
                r = {"model": model, "set": s, "tracks": "+".join(SETS[s]),
                     "cell": cell, "held_out": cell not in mx.TRAIN12,
                     "n_regions": int(df.model.notna().sum()),
                     "median": df.model.median(), "mean": df.model.mean()}
                if cons:
                    r["consensus_median"] = df[cons[0]].median()
                    r["beats_consensus_frac"] = float((df.model > df[cons[0]]).mean())
                rows.append(r)
    S = pd.DataFrame(rows)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(RES_OUT / "meth_eval_summary.csv", index=False, float_format="%.4f")
    log(f"summary: {len(S)} rows")
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=["predict", "summary"], required=True)
    a = ap.parse_args()
    if "predict" in a.stage:
        stage_predict()
    if "summary" in a.stage:
        print(stage_summary().to_string(index=False))


if __name__ == "__main__":
    main()
