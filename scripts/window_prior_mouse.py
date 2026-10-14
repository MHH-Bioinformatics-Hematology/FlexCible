"""Window prior on mouse (mm10), registry section 76 secondary (human-trained models and P, unchanged).

The prior of a mouse sample is P applied to its marks: wp_ref with the reference marks = the other mouse
samples (mesc, ch12, heart; leave-one-out, no human cell), wp_self with the sample's own marks. P is the full
human-trained function (P_full: chromosome numbers carry no meaning across species); no retraining. The main
models all12wp_{ref,self,both}_drop are the human-trained checkpoints read through the config views of
scripts/mouse_transfer.py (reference set = the mouse pool); predictions go to validation_mouse/pred_wp/ and
are scored with the scoring functions of mouse_transfer.py next to the honest, refres and diff models.

    python scripts/window_prior_mouse.py cools --res 10000
    python scripts/window_prior_mouse.py predict --res 10000 --gpu 0
    python scripts/window_prior_mouse.py score --res 10000 --procs 6
"""
from __future__ import annotations

import argparse
import copy
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
import mouse_transfer as mt  # noqa: E402
import window_prior as wpr  # noqa: E402

VM, HM = mt.VM, mt.HM
OUTP = VM / "pred_wp"
RESULTS = REPO / "results/window_prior"
WPM = {"wp_ref": ("all12wp_ref_drop", "ref"), "wp_self": ("all12wp_self_drop", "own"),
       "wp_both": ("all12wp_both_drop", "ref")}
OLD = {"honest": "all12honest_drop", "refin": "all12refin_drop", "refres": "all12refres_drop",
       "diff": "all12diffcells_refres_drop"}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def pcool(res: int, tag: str, cell: str) -> Path:
    return wpr.WP / "mouse" / f"{res // 1000}kb" / tag / f"{cell}.cool"


def stage_cools(a):
    import cooler

    from hicmultiome import diffmarks as dmk
    from hicmultiome.windowprior import mark_rows, predict_chrom, write_prior_cool
    cfg = wpr.base_cfg(a.res)
    base = copy.deepcopy(cfg.data.samples[0])
    samples = []
    for c in mt.POOL:
        d = VM / mt.MARKS[c]
        have = {p.stem for p in d.glob("*.bigwig")}
        s = copy.deepcopy(base)
        s.name, s.bigwig_dir, s.extra_bigwig_dirs = c, str(d), []
        s.missing_tracks = [t for t in cfg.track_names if t not in have]
        s.zeroed_tracks, s.diff_set, s.diff_identity = [], "consensus", c
        samples.append(s)
    cfg.data.samples = samples
    cfg.data.diff_reference_sets = {"consensus": list(mt.POOL)}
    cfg.data.diff_default_set = "consensus"
    cfg.train.cache_dir = str(wpr.WP / "mouse" / f"{a.res // 1000}kb")
    inst = dmk.DiffMarks.get(cfg)
    tmpl = next(p for p in [mt.oe_cool(c, a.res) for c in mt.POOL] if Path(p).exists())
    grid = cooler.Cooler(str(tmpl))
    dev = a.device
    chroms = [c for c in mt.CHRS + mt.CHRX if c in grid.chromnames]
    for tag in ("ref", "own"):
        net, ck = wpr.load_net(wpr.model_dir(a.res, tag, "full"), dev)
        for s in samples:
            out = pcool(a.res, tag, s.name)
            if out.exists():
                continue
            accs = {}
            for chrom in chroms:
                nb = int(grid.extent(chrom)[1] - grid.offset(chrom))
                x = mark_rows(inst, s, chrom, nb, tag)
                accs[chrom] = predict_chrom(net, x, None, ck["window"], device=dev)
            pool = dmk.reference_members(list(mt.POOL), s.name) if tag == "ref" else []
            write_prior_cool(out, tmpl, accs, {"kind": "window_prior_mouse", "mode": tag, "cell": s.name,
                                               "cells": pool, "source": "P_full, human-trained, unchanged"})
            log(f"wrote {out}")


def stage_predict(a):
    import yaml
    # config views of the wp models (the views of mouse_transfer.py for other run names)
    saved = dict(mt.MODELS)
    mt.MODELS.clear()
    mt.MODELS.update({m: r for m, (r, _) in WPM.items()})
    try:
        mt.stage_views(a)
    finally:
        mt.MODELS.clear()
        mt.MODELS.update(saved)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS=str(a.threads))
    py = os.environ.get("HICMULTI_PY", sys.executable)
    for res in a.res:
        k = mt.kb(res)
        for m, (run, tag) in WPM.items():
            ck = VM / "views" / f"{run}_{k}" / "checkpoint-best"
            if not (ck / "model.pt").exists():
                log(f"[skip] {m} {k}: no view")
                continue
            for hic in mt.POOL:
                if a.only and hic not in a.only:
                    continue
                tgt, bw = mt.oe_cool(hic, res), VM / mt.MARKS[hic]
                pc = pcool(res, tag, hic)
                out = OUTP / k / m / hic
                if (out / "DONE").exists() or not Path(tgt).exists() or not pc.exists():
                    continue
                job = f"windowprior_mouse_{m}_{hic}_{k}"
                if mt.claim(job, out / "DONE") != 0:
                    continue
                out.mkdir(parents=True, exist_ok=True)
                have = {p.stem for p in bw.glob("*.bigwig")}
                ks = f"all7={','.join(t for t in mt.SETS['all7'] if t in have)}"
                cmd = [py, "scripts/mouse_transfer_predict.py", "--config", str(ck / "config.yaml"),
                       "--checkpoint", str(ck), "--cell", hic, "--bigwig-dir", str(bw), "--target", str(tgt),
                       "--tta", "--assay", mt.ASSAY[hic], "--binsize", str(res), "--device", a.device,
                       "--regions", str(VM / "regions" / f"crop_{k}_auto.tsv"), "--out-dir", str(out),
                       "--save-pred", str(out / "{set}.npz"), "--keep-sets", ks]
                if m == "wp_both":
                    cmd += ["--reference", str(mt.loo_cool(hic, res)), "--prior", str(pc)]
                else:
                    cmd += ["--reference", str(pc)]
                if m != "wp_self":
                    cmd += ["--diff-group", "cell"]
                log(f"[run] {job}")
                with open(out / "predict.log", "w") as fh:
                    r = subprocess.run(["nice", "-n", "10"] + cmd, stdout=fh, stderr=subprocess.STDOUT,
                                       cwd=REPO, env=env)
                if r.returncode == 0 and (out / "all7.npz").exists():
                    (out / "DONE").write_text(time.strftime("%F %T"))
                else:
                    log(f"[FAILED] {job}")
                    subprocess.run([sys.executable, str(REPO / "scripts/claim_job.py"), "release", job])


def stage_score(a):
    import multiprocessing as mp
    for res in a.res:
        k = mt.kb(res)
        Wn = mt.load_windows(res, ("auto",))
        for cell in a.cells:
            outf = RESULTS / f"mouse_scores_{cell}_{k}.csv.gz"
            if outf.exists():
                continue
            methods = {}
            for pdir in sorted((VM / "pred" / k).glob("*")):
                if pdir.name in OLD and (pdir / cell / "DONE").exists():
                    for npz in (pdir / cell).glob("all7.npz"):
                        methods[f"{pdir.name}:all7"] = npz
            for m in WPM:
                f = OUTP / k / m / cell / "all7.npz"
                if f.exists() and (f.parent / "DONE").exists():
                    methods[f"{m}:all7"] = f
            if not any(x.startswith("wp_") for x in methods):
                continue
            rows = list(zip(Wn.key, Wn.chrom, Wn.b_lo, Wn.b_hi, Wn.seq_key, Wn.set))
            chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
            log(f"score {cell} {k}: {len(rows)} windows, {sorted(methods)}")
            res_l = []
            with mp.Pool(a.procs, initializer=mt._init_worker, initargs=(cell, res, methods, [], None)) as pool:
                for r in pool.imap_unordered(mt._score_chunk, chunks):
                    res_l.extend(r)
            df = pd.DataFrame(res_l)
            outf.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outf, index=False, float_format="%.5g")
            log(df[df.set == "auto"].groupby("method").pearson.median().sort_values(ascending=False).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["cools", "predict", "score"])
    ap.add_argument("--res", nargs="+", type=int, default=[10000])
    ap.add_argument("--cells", nargs="+", default=list(mt.POOL))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "cools":
        a.res = a.res[0]
    {"cools": stage_cools, "predict": stage_predict, "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
