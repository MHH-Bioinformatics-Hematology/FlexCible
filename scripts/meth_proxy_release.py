"""Methylation as a proxy that replaces missing tracks, on the release candidate all12one_fullrd
(registry section 95; the analysis, group sets, replacement fraction and bootstrap of section 78,
scripts/meth_proxy_analysis.py, unchanged).

The release candidate is one checkpoint for every bin size with Meth as the 14th input track and
the leave-one-out consensus map as an input plane; its per-resolution views are
runs/multicell_reference/all12one_fullrd_<k>kb (flexcible.train_multires.write_views). Per view and
cell the 413 Akita test regions are predicted (TTA, --structure) from the cell's ChIP tracks with each
group set of section 35 removed, with and without the cell's methylation track, in ONE pass:

  <set>, <set>+meth                  consensus plane as input (the family "consensus" of section 78)
  <set>_noref, <set>+meth_noref      reference plane at 0 (the ChIP-only use of the checkpoint; the
                                     family "honest" of section 78 is the dedicated counterpart)

Cells: the 10 training cells with methylation (7 WGBS, 3 RRBS) and the held-out K562 and HMEC
(deep map). The 413 regions are held out of the training of the training cells.

The noise threshold of a replacement fraction is that of section 78: 0.0062 for the models with the
consensus (section 51), and for the plane at 0 the honest two-seed spread of the dedicated model at
that bin size (results/meth_proxy/seed_noise.csv).

    python scripts/meth_proxy_release.py --stage predict --gpu 1
    python scripts/meth_proxy_release.py --stage summary

Single marks (registry section 97; option --sets): the question for which individual mark methylation
can act as a proxy. The sets are

  minus <mark>   m1_<mark>: all 13 ChIP tracks except that mark, with and without Meth
  alone <mark>   o1_<mark>: only that mark, with and without Meth
  each with the consensus plane and with the plane at 0 (suffix _noref); the reference sets "full" and
  "full+meth" come from the group pass above. RF = gain from Meth / loss against the full set, as above.

    python scripts/meth_proxy_release.py --sets minus alone --stage predict --gpu 1 --set-batch 8
    python scripts/meth_proxy_release.py --sets minus alone --stage summary_single
"""
from __future__ import annotations

import argparse
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
import meth_proxy_analysis as mp                                    # noqa: E402
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

HM = mx.HM
OUT = HM / "runs/meth_proxy"
RES_OUT = REPO / "results/meth_proxy"
PY = os.environ.get("HICMULTI_PY", sys.executable)
BASE = "all12one_fullrd"
RES = [10, 25]
CELLS = mp.WGBS_TRAIN + mp.RRBS_TRAIN + mp.HELD
CONSENSUS_NOISE = mp.SEC51_CONSENSUS_NOISE
SET_BATCH = 1


def view(res_kb: int) -> Path:
    return mx.REF_OUT / f"{BASE}_{res_kb}kb"


def model_name(res_kb: int, noref: bool) -> str:
    return f"{BASE}_noref_{res_kb}kb" if noref else f"{BASE}_{res_kb}kb"


MARKS = list(mx.ALL13)
SINGLE_ORDER = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K36me3",
                "H3K79me2", "H4K20me1", "H3K27me3", "H3K9me3"]
SELECT = ["groups"]                     # set families of this call (--sets)


def single_base(kind: str) -> dict[str, list[str]]:
    """Single-mark keep sets without Meth: m1_<mark> (that mark removed), o1_<mark> (that mark alone)."""
    if kind == "minus":
        return {f"m1_{m}": [x for x in MARKS if x != m] for m in MARKS}
    return {f"o1_{m}": [m] for m in MARKS}


def family_sets(kind: str) -> dict[str, list[str]]:
    """Keep sets of one family with and without Meth (consensus plane)."""
    base = mp.keep_sets() if kind == "groups" else single_base(kind)
    if kind != "groups":
        base = {**base, **{f"{k}+meth": v + [me.TRACK] for k, v in base.items()}}
    return base


def sets_of(kind: str) -> dict[str, list[str]]:
    base = family_sets(kind)
    return {**base, **{f"{k}_noref": v for k, v in base.items()}}


def sets_all() -> dict[str, list[str]]:
    """Keep sets of one pass: the 13 group sets with and without Meth, plane present and plane at 0
    (--sets groups, default), or the single-mark sets of the chosen families (--sets minus alone)."""
    out: dict[str, list[str]] = {}
    for k in SELECT:
        out.update(sets_of(k))
    return out


def pred_csv(res_kb: int, noref: bool, st: str, cell: str) -> Path:
    s = f"{st}_noref" if noref else st
    return OUT / f"{BASE}_{res_kb}kb" / s / f"predict_{cell}.csv"


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "meth_proxy_release.log", "a") as fh:
        fh.write(line + "\n")


def done(res_kb: int, cell: str) -> bool:
    return all(pred_csv(res_kb, nr, st, cell).exists() for nr in (False, True)
               for k in SELECT for st in family_sets(k))


def command(res_kb: int, cell: str) -> list:
    res = res_kb * 1000
    ck = view(res_kb) / "checkpoint-best"
    _, bw, assay, _ = mx.CELLS[cell]
    ref = mp.consensus_floor(cell, res)
    tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
           "--cell", cell, "--bigwig-dir", bw, "--target", mx.target(cell, res), "--tta", "--assay", assay,
           "--binsize", res, "--device", "cuda", "--out-dir", OUT / f"{BASE}_{res_kb}kb", "--structure",
           "--set-batch", SET_BATCH, "--keep-sets", *[f"{s}={','.join(t)}" for s, t in sets_all().items()],
           "--reference", ref, "--others", f"{tag}={ref}"]
    if me.has_meth(cell, True):
        cmd += ["--extra-bigwig-dir", me.meth_dir(cell, True)]
    return cmd


def stage_predict(a) -> None:
    tasks = [(k, c) for k in RES for c in CELLS if not done(k, c)]
    log(f"predict: {len(tasks)} view-cell tasks on gpu{a.gpu}")
    for k, c in tasks:
        cmd = command(k, c)
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   CUDA_VISIBLE_DEVICES=str(a.gpu))
        d = OUT / f"{BASE}_{k}kb"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"predict_{c}.log", "a") as fh:
            fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
            fh.flush()
            rc = subprocess.call(["nice", "-n", "10", *map(str, cmd)], stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=REPO, env=env)
        log(f"predict {BASE} {k}kb {c} rc={rc}")


# --------------------------------------------------------------------------- tables
def noise_of_model(model: str) -> float:
    if "_noref_" not in model:
        return CONSENSUS_NOISE
    res = int(model.rsplit("_", 1)[1].removesuffix("kb"))
    t = pd.read_csv(RES_OUT / "seed_noise.csv")
    t = t[(t.res_kb == res) & (t.family == "honest meth")]
    return float(t.median_abs_diff.median())


def load_regions() -> pd.DataFrame:
    cols = {v: k for k, v in mp.METRICS.items()}
    parts = []
    for k in RES:
        for noref in (False, True):
            for cell in CELLS:
                for st in mp.keep_sets():
                    f = pred_csv(k, noref, st, cell)
                    if not f.exists():
                        continue
                    d = pd.read_csv(f)
                    d = d[["region"] + [c for c in cols if c in d]].rename(columns=cols)
                    d.insert(0, "meth", st.endswith("+meth"))
                    d.insert(0, "set", st.removesuffix("+meth"))
                    d.insert(0, "cell", cell)
                    d.insert(0, "model", model_name(k, noref))
                    parts.append(d)
    return pd.concat(parts, ignore_index=True)


def stage_summary(a) -> None:
    R = load_regions()
    log(f"regions table {R.shape}")
    models = {model_name(k, nr): dict(family="plane at 0" if nr else "consensus", res=k, planes=False, seed=0)
              for k in RES for nr in (False, True) if (R.model == model_name(k, nr)).any()}
    metrics = mp.MAIN_METRICS + ["scc", "eigenvector"]
    C, P = mp.analyse(R, models, lambda m: CELLS, mp.cell_groups,
                      ["training WGBS", "training RRBS", "held-out", "training all"],
                      metrics, mp.SET_ORDER, noise_of_model, nb=a.nboot)
    for df in (C, P):
        df.insert(1, "family", df.model.map(lambda m: models[m]["family"]))
        df.insert(2, "res_kb", df.model.map(lambda m: models[m]["res"]))
    C.to_csv(RES_OUT / f"{BASE}_replacement_by_cell.csv", index=False, float_format="%.5f")
    P.to_csv(RES_OUT / f"{BASE}_replacement_pooled.csv", index=False, float_format="%.5f")
    log("summary written")


def main() -> None:
    np.seterr(divide="ignore", invalid="ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True, choices=["predict", "summary", "summary_single"])
    ap.add_argument("--sets", nargs="+", default=["groups"], choices=["groups", "minus", "alone"],
                    help="keep-set families: the 13 group sets of section 95 (default) or the single marks")
    ap.add_argument("--set-batch", type=int, default=1, help="keep sets per forward pass (outputs equal to 1e-4)")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--nboot", type=int, default=500)
    a = ap.parse_args()
    global SET_BATCH
    SELECT[:] = a.sets
    SET_BATCH = a.set_batch
    for st in a.stage:
        globals()[f"stage_{st}"](a)


if __name__ == "__main__":
    main()
