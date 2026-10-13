"""Independent validation of the unseen-cell-type claim on cells no part of the pipeline has touched
(registry section 74; pre-registration in results/independent_validation/PREREGISTRATION.md).

Samples (inputs of scripts/independent_validation_fetch.py): pc3 (PC-3), panc1 (Panc1), nk (natural killer
cell, primary), caco2 (Caco-2). Seven marks each (CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3,
H3K9me3), 10 kb and 25 kb O/E targets built as the training cells' (oe_proper). The trained checkpoints are
used unchanged: no retraining, no tuning, consensus12 (the 12 training cells) as Hi-C reference input.

Stages
  regions   region tables: the 413 Akita test windows and a whole-genome tiling (1 048 576 bp sequences
            every 917 504 bp, chr1-22); the scored pixels of every window are the central 917 504 bp
            (Akita's 448 x 2048 bp target crop), whole 10 / 25 kb bins inside it
  predict   crosscell_predict.py of every model x resolution x sample, four mark sets (all seven, CTCF +
            H3K27ac, CTCF, none), predictions saved on the target grid (GPU, claims by job name)
  akita     Akita (GM12878 head, sequence only) on every window (basenji environment, CPU)
  score     per-region measures of every predictor on identical pixels (hicmultiome.structure:
            Pearson, SCC, insulation r, boundary F1, loop enrichment r, APA ratio, compartment r)
  ceiling   split-half (binomial) reproducibility ceiling of every target (Spearman-Brown)
  summary   tables (medians, paired bootstrap), endpoint verdicts, figure

    python scripts/independent_validation.py regions
    python scripts/independent_validation.py predict --cells pc3 panc1 --res 10000 25000 --gpu 0
    ~/miniconda3/envs/basenji/bin/python scripts/independent_validation.py akita
    python scripts/independent_validation.py score --cells pc3 --res 10000 --procs 7
    python scripts/independent_validation.py summary
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

HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
VN = HM / "validation_new"
OE = HM / "crosscell_oe"
MR = HM / "runs/multicell_reference"
RESULTS = REPO / "results/independent_validation"
AK_REGIONS = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/"
                  "akita_validation/sequences_test_list.tsv")
SEQ, CROP_PAD, STEP = 1_048_576, 65_536, 917_504          # sequence, crop margin, tile step (bp)
CHRS = [f"chr{i}" for i in range(1, 23)]
CHRHO_HELD = ["chr2", "chr3", "chr11", "chr14", "chr17"]
CORE7 = ["CTCF", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]

NEW = ["pc3", "panc1", "nk", "caco2"]
# cell: (bigWig dir, assay of the Hi-C map); k562 is only the pipeline check against the registry numbers
CELLS = {c: (VN / f"hg38_{c}", a) for c, a in
         [("pc3", "intact"), ("panc1", "hic"), ("nk", "intact"), ("caco2", "intact")]}
CELLS["k562"] = (HM / "hg38_k562", "hic")


LOCAL = Path(os.environ.get("IV_CACHE", str(Path.home() / "scratch/iv_cache")))   # local copies (the share is slow)


def cool_of(cell: str, res: int, local: bool = True) -> Path:
    k = res // 1000
    p = OE / f"{cell}_proper_{k}kb.cool" if cell not in CELLS or cell == "k562" \
        else VN / f"oe/{cell}_proper_{k}kb.cool"
    q = LOCAL / p.name
    return q if (local and q.exists()) else p


# measured maps of other cells (the baselines of the earlier tables)
OTHERS = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248", "mcf7", "cd14mono",
          "pc9", "k562", "hmec_deep", "huvec_deep"]
# key: (run base, chrho)  -> run folder <base>_<k>kb
MODELS = {
    "refin": "all12refin_drop", "refres": "all12refres_drop", "honest": "all12honest_drop",
    "diff": "all12diffcells_refres_drop",
    "refin_chrho": "all12refin_drop_chrho", "refres_chrho": "all12refres_drop_chrho",
    "honest_chrho": "all12honest_drop_chrho",
    "refin_s1": "all12refin_drop_s1", "refin_s2": "all12refin_drop_s2",
    "honest_s1": "all12honest_drop_s1", "honest_s2": "all12honest_drop_s2",
}
MAINS = ["refin", "refres", "honest", "diff"]
SETS = {"all": CORE7, "ctcf_k27ac": ["CTCF", "H3K27ac"], "ctcf": ["CTCF"], "none": []}
SETS_BY_MODEL = {"refin_chrho": ["all"], "refres_chrho": ["all"], "honest_chrho": ["all"],
                 "refin_s1": ["all"], "refin_s2": ["all"], "honest_s1": ["all"], "honest_s2": ["all"]}


def run_dir(model: str, res: int) -> Path:
    return MR / f"{MODELS[model]}_{res // 1000}kb"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ================================================================== regions
def crop_bins(start: int, end: int, res: int) -> tuple[int, int]:
    """First and one-past-last 10/25 kb bin lying wholly inside the central 917 504 bp of a sequence."""
    c0, c1 = start + CROP_PAD, end - CROP_PAD
    return -(-c0 // res), c1 // res


def sequence_regions() -> pd.DataFrame:
    """test windows (Akita's 413) and genome tiling; columns chrom, start, end (sequence), set."""
    import cooler
    t = pd.read_csv(AK_REGIONS, sep="\t", header=None, names=["chrom", "start", "end"])
    t["set"] = "test"
    cs = cooler.Cooler(str(OE / "consensus12_proper_10kb.cool")).chromsizes
    rows = []
    for c in CHRS:
        L = int(cs[c])
        for s in range(0, L - SEQ + 1, STEP):
            rows.append((c, s, s + SEQ, "tile"))
    til = pd.DataFrame(rows, columns=["chrom", "start", "end", "set"])
    return pd.concat([t, til], ignore_index=True)


def aligned(df: pd.DataFrame, res: int) -> pd.DataFrame:
    b = [crop_bins(s, e, res) for s, e in zip(df.start, df.end)]
    out = df.copy()
    out["b_lo"], out["b_hi"] = [x[0] for x in b], [x[1] for x in b]
    out["a_start"], out["a_end"] = out.b_lo * res, out.b_hi * res
    out["key"] = out.chrom + ":" + out.a_start.astype(str) + "-" + out.a_end.astype(str)
    out["seq_key"] = out.chrom + ":" + out.start.astype(str)
    return out


def stage_regions():
    d = VN / "regions"
    d.mkdir(parents=True, exist_ok=True)
    R = sequence_regions()
    R.to_csv(d / "sequence_regions.tsv", sep="\t", index=False)
    for res in (10000, 25000):
        A = aligned(R, res)
        A.to_csv(d / f"aligned_{res // 1000}kb.tsv", sep="\t", index=False)
        held = A.chrom.isin(CHRHO_HELD)
        for nm, sub in (("all", A), ("chrho", A[held]), ("test", A[A.set == "test"]),
                        ("test_chrho", A[(A.set == "test") & held]), ("tiles", A[A.set == "tile"]),
                        ("tiles_chrho", A[(A.set == "tile") & held])):
            sub.drop_duplicates("key")[["chrom", "a_start", "a_end"]].to_csv(
                d / f"crop_{res // 1000}kb_{nm}.tsv", sep="\t", header=False, index=False)
    log(f"regions: {int((R.set == 'test').sum())} test windows, {int((R.set == 'tile').sum())} tiles")


# ================================================================== predict
def claim(name: str, done: Path) -> int:
    r = subprocess.run([sys.executable, str(REPO / "scripts/claim_job.py"), "claim", name,
                        "--done-path", str(done), "--cmd", f"independent_validation.py predict {name}"])
    return r.returncode


def stage_predict(a):
    pdir = a.pred_dir or {"all": "pred_all", "test": "pred", "tiles": "pred_tiles"}[a.regionset]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads))
    py = os.environ.get("HICMULTI_PY", sys.executable)
    c12 = lambda res: _loc(OE / f"consensus12_proper_{res // 1000}kb.cool")      # noqa: E731
    for res in a.res:
        for cell in a.cells:
            bw, assay = CELLS[cell]
            lb = Path(os.environ.get("IV_BW", str(Path.home() / "scratch/iv_bw"))) / cell
            if (lb / "COPIED").exists():                    # local copy of the bigWigs (the share reads at KB/s)
                bw = lb
            if not cool_of(cell, res).exists() or not bw.exists():
                log(f"[skip] {cell} {res}: target or bigWigs missing")
                continue
            have = {t for t in CORE7 if (bw / f"{t}.bigwig").exists()}
            for m in a.models:
                run = run_dir(m, res)
                ck = run / "checkpoint-best"
                if not (ck / "model.pt").exists():
                    log(f"[skip] {run.name}: no checkpoint")
                    continue
                if a.regionset == "tiles" and m not in MAINS and not m.endswith("chrho"):
                    continue                               # whole-genome tiles: the four models and the chrho models
                out = VN / pdir / f"{res // 1000}kb" / m / cell
                done = out / "DONE"
                if done.exists():
                    continue
                job = f"iv_{cell}_{m}_{res // 1000}kb" + ("" if pdir == "pred" else "_" + pdir)
                rc = claim(job, done)
                if rc != 0:
                    log(f"[claim] {job} rc={rc}, skipped")
                    continue
                out.mkdir(parents=True, exist_ok=True)
                sets = ["all"] if a.regionset == "tiles" else SETS_BY_MODEL.get(m, list(SETS))
                ks = [f"{n}={','.join(t for t in SETS[n] if t in have)}" for n in sets]
                kind = {"all": "all", "test": "test", "tiles": "tiles"}[a.regionset] + \
                    ("_chrho" if m.endswith("chrho") and a.regionset != "all" else "")
                if a.regionset == "all" and m.endswith("chrho"):
                    kind = "chrho"
                regs = VN / "regions" / f"crop_{res // 1000}kb_{kind}.tsv"
                if a.limit:
                    regs = out / "regions_limited.tsv"
                    pd.read_csv(VN / "regions" / f"crop_{res // 1000}kb_all.tsv", sep="\t",
                                header=None).sample(a.limit, random_state=0).to_csv(
                        regs, sep="\t", header=False, index=False)
                cmd = [py, "scripts/independent_validation_predict.py", "--config", str(ck / "config.yaml"),
                       "--checkpoint", str(ck), "--cell", cell, "--bigwig-dir", str(bw),
                       "--target", str(cool_of(cell, res)), "--tta", "--assay", assay,
                       "--binsize", str(res), "--device", "cuda", "--regions", str(regs),
                       "--out-dir", str(out), "--save-pred", str(out / "{set}.npz"),
                       "--keep-sets", *ks]
                if not m.startswith("honest"):
                    cmd += ["--reference", str(c12(res))]
                if m == "diff":
                    cmd += ["--diff-group", "cell"]
                log(f"[run] {job}")
                with open(out / "predict.log", "w") as fh:
                    fh.write("[cmd] " + " ".join(cmd) + "\n")
                    fh.flush()
                    r = subprocess.run(["nice", "-n", "10"] + cmd, stdout=fh, stderr=subprocess.STDOUT,
                                       cwd=REPO, env=env)
                if r.returncode == 0 and all((out / f"{s}.npz").exists() for s in sets):
                    done.write_text(time.strftime("%F %T"))
                else:
                    log(f"[FAILED] {job} rc={r.returncode}")
                    subprocess.run([sys.executable, str(REPO / "scripts/claim_job.py"), "release", job])


# ================================================================== akita
def stage_akita(a):
    import aml_competitors_loops as acl
    import json as _json
    from basenji import seqnn
    R = pd.read_csv(VN / "regions" / "sequence_regions.tsv", sep="\t")
    if a.limit:
        R = R.sample(a.limit, random_state=0)
    d = acl.GALAXY_MODELS / "akita"
    params = _json.load(open(d / "params.json"))
    m = seqnn.SeqNN(params["model"])
    m.restore(str(d / "model_best.h5"))
    NB, HEAD, ASZ = 448, 2, 2048
    iu = np.triu_indices(NB, 2)
    out = VN / "akita"
    out.mkdir(parents=True, exist_ok=True)
    f_out = out / ("akita_limited.npz" if a.limit else (f"akita_part{a.part}.npz" if a.part >= 0 else "akita.npz"))
    res_d = {}
    for f in ([f_out] if not (out / "akita.npz").exists() or f_out == out / "akita.npz" else [f_out, out / "akita.npz"]):
        if f.exists():                                   # resume; windows of the first single-process run are kept
            res_d.update(dict(np.load(f)))
    t0, cur, seq = time.time(), None, None
    R = R.sort_values(["chrom", "start"]).drop_duplicates(["chrom", "start"])
    if a.part >= 0:
        R = R.iloc[a.part::a.nparts]
    n = 0
    for _, r in R.iterrows():
        key = f"{r.chrom}:{r.start}"
        if f"10000:{key}" in res_d:
            continue
        if r.chrom != cur:
            seq, cur = acl.hg38_seq(r.chrom), r.chrom
        s0 = int(r.start)
        codes = np.full(SEQ, 4, np.uint8)
        lo, hi = max(s0, 0), min(s0 + SEQ, len(seq))
        codes[lo - s0:hi - s0] = seq[lo:hi]
        oh = np.zeros((SEQ, 4), np.float32)
        for code, col in ((0, 0), (1, 3), (2, 1), (3, 2)):          # a0 t1 c2 g3 -> columns A C G T
            oh[codes == code, col] = 1
        x = np.stack([oh, oh[::-1, ::-1]])
        y = m.model.predict(x, verbose=0)[..., HEAD]
        mats = []
        for rr in range(2):
            M = np.full((NB, NB), np.nan)
            M[iu] = y[rr]
            M = np.where(np.isfinite(M), M, M.T)
            mats.append(M if rr == 0 else M[::-1, ::-1])
        Mm = np.nanmean(mats, 0) * np.log(2.0)
        Lf = acl.log_to_final(Mm, NB)
        x0 = s0 + SEQ // 2 - NB * ASZ // 2                           # = start + 65 536
        for res in (10000, 25000):
            lo_b, hi_b = crop_bins(s0, s0 + SEQ, res)
            res_d[f"{res}:{key}"] = acl.sample_grid(Lf, ASZ, x0, lo_b, n=hi_b - lo_b, res=res).astype(np.float32)
        n += 1
        if n % 100 == 0:
            log(f"akita {n}/{len(R)} windows, {time.time() - t0:.0f} s")
            np.savez(f_out.with_suffix(".tmp.npz"), **res_d)
            os.replace(f_out.with_suffix(".tmp.npz"), f_out)
    np.savez(f_out.with_suffix(".tmp.npz"), **res_d)
    os.replace(f_out.with_suffix(".tmp.npz"), f_out)
    log(f"akita written {len(res_d) // 2} windows")


# ================================================================== scoring
METRICS = ["pearson", "scc", "insulation_corr", "boundary_f1", "loop_enrich_corr", "loop_apa_ratio",
           "compartment_corr", "eigenvector_abs_corr"]
_W = {}


def _fetch(clr, chrom, lo, hi, res):
    try:
        return np.asarray(clr.matrix(balance=False).fetch((chrom, lo * res, hi * res)), float)
    except Exception:                                                         # noqa: BLE001
        return None


def _mask_zero(T):
    return None if T is None else np.where(T == 0, np.nan, T)


def _loc(p: Path) -> Path:
    q = LOCAL / p.name
    return q if q.exists() else p


def stage_cache(a):
    """copy the O/E cools the scoring reads (3 to 5 GB) to the local disk; the share answers 0.7 s per fetch"""
    import shutil
    LOCAL.mkdir(parents=True, exist_ok=True)
    for res in a.res:
        k = res // 1000
        files = [OE / f"consensus12_proper_{k}kb.cool"] + \
                ([cool_of(o, res, False) for o in OTHERS] if a.with_others else []) + \
                [cool_of(c, res, False) for c in a.cells]
        for f in files:
            d = LOCAL / f.name
            if d.exists() or not f.exists():
                continue
            tmp = d.with_name(d.name + ".part")
            shutil.copyfile(f, tmp)
            tmp.rename(d)
            log(f"cached {f.name}")


def _init_worker(cell, res, sets, methods, akita_file, others, full_others=False, only_others=False):
    import cooler
    _W.clear()
    _W.update(cell=cell, res=res, tgt=cooler.Cooler(str(cool_of(cell, res))),
              c12=cooler.Cooler(str(_loc(OE / f"consensus12_proper_{res // 1000}kb.cool"))),
              npz={m: [np.load(f) for f in fl] for m, fl in methods.items()},
              others={o: cooler.Cooler(str(cool_of(o, res))) for o in others},
              akita=[np.load(f) for f in akita_file] if akita_file else None, full_others=full_others,
              only_others=only_others)


def _pearson(x, y):
    from hicmultiome.structure import _corr
    return _corr(x, y)


def _score_chunk(rows):
    """rows: list of (key, chrom, lo, hi, seqkey, set). Returns metric rows of the common-pixel scoring."""
    from hicmultiome.structure import structure_metrics
    W = _W
    res = W["res"]
    out = []
    for key, chrom, lo, hi, seqkey, st in rows:
        T = _mask_zero(_fetch(W["tgt"], chrom, lo, hi, res))
        if T is None or T.shape[0] != hi - lo or T.shape[0] < 20:
            continue
        maps = {}
        c = _mask_zero(_fetch(W["c12"], chrom, lo, hi, res))
        if c is not None and c.shape == T.shape:
            maps["consensus12"] = c
        for m, zl in W["npz"].items():
            for z in zl:
                if key in z.files:
                    P = np.asarray(z[key], float)
                    if P.shape != T.shape:
                        continue
                    live = np.abs(np.nan_to_num(P)).sum(1) > 0
                    maps[m] = np.where(live[:, None] & live[None, :], P, np.nan)
                    break
        for z in (W["akita"] or []):
            if f"{res}:{seqkey}" in z.files:
                maps["akita"] = np.asarray(z[f"{res}:{seqkey}"], float)
                break
        if "consensus12" not in maps:
            continue
        mask = np.isfinite(T)
        for m, M in maps.items():
            mask &= np.isfinite(M)
        iu = np.triu_indices(T.shape[0], 2)
        n_px = int(mask[iu].sum())
        if n_px < 100:
            continue
        Tm = np.where(mask, T, np.nan)
        base = {"cell": W["cell"], "res": res, "key": key, "chrom": chrom, "set": st, "n_px": n_px}
        for m, M in maps.items():
            if W["only_others"]:
                break
            r = structure_metrics(np.where(mask, M, np.nan), Tm, res)
            out.append({**base, "method": m, **{k: r.get(k, np.nan) for k in METRICS}})
        for o, clr in W["others"].items():            # measured maps of the other cells (own finite pixels)
            O = _mask_zero(_fetch(clr, chrom, lo, hi, res))
            if O is None or O.shape != T.shape:
                continue
            mk = mask & np.isfinite(O)
            n_o = int(mk[iu].sum())
            if n_o < 100:
                continue
            if W["full_others"]:
                r = structure_metrics(np.where(mk, O, np.nan), np.where(mk, T, np.nan), res)
            else:                                     # first pass: Pearson only (choice of the best cell)
                r = {"pearson": _pearson(O[iu][mk[iu]], T[iu][mk[iu]])}
            out.append({**base, "method": f"cell:{o}", "n_px": n_o,
                        **{k: r.get(k, np.nan) for k in METRICS}})
    return out


def _run_pool(a, cell, res, chunks, methods, akf, others, full, only_others=False):
    import multiprocessing as mp
    out = []
    with mp.Pool(a.procs, initializer=_init_worker,
                 initargs=(cell, res, None, methods, akf if akf else None, others, full,
                           only_others)) as pool:
        for i, r in enumerate(pool.imap_unordered(_score_chunk, chunks)):
            out.extend(r)
            if i % 25 == 0:
                log(f"  {i + 1}/{len(chunks)} chunks")
    return out


def stage_score(a):
    import multiprocessing as mp
    for res in a.res:
        k = res // 1000
        A = aligned(pd.read_csv(VN / "regions" / "sequence_regions.tsv", sep="\t"), res).drop_duplicates("key")
        for cell in a.cells:
            outf = VN / ("scores" if a.pred_dir is None else a.pred_dir + "_scores") / f"{cell}_{k}kb.csv.gz"
            if outf.exists() and not a.limit:
                log(f"[have] {outf.name}")
                continue
            methods = {}
            dirs = ["pred", "pred_tiles"] if a.pred_dir is None else [a.pred_dir]
            for dname in dirs:
                for m in MODELS:
                    for sname in list(SETS):
                        f = VN / dname / f"{k}kb" / m / cell / f"{sname}.npz"
                        if f.exists() and (f.parent / "DONE").exists():
                            methods.setdefault(f"{m}:{sname}", []).append(f)
            akf = sorted((VN / "akita").glob("akita*.npz")) if (VN / "akita").exists() else []
            akf = [f for f in akf if "limited" not in f.name and "tmp" not in f.name]
            sub = A if not a.limit else A.sample(a.limit, random_state=0)
            if a.test_only:
                sub = sub[sub.set == "test"]

            rows = list(zip(sub.key, sub.chrom, sub.b_lo, sub.b_hi, sub.seq_key, sub.set))
            chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
            log(f"score {cell} {k}kb: {len(rows)} windows, {len(methods)} model outputs, akita files={len(akf)}")
            others = [o for o in OTHERS if o != cell and cool_of(o, res).exists()]
            res_l = _run_pool(a, cell, res, chunks, methods, akf, others, False)
            first = pd.DataFrame(res_l)
            ot = first[(first.set == "test") & first.method.str.startswith("cell:")]
            best = ot.groupby("method").pearson.median().idxmax().split(":", 1)[1]
            log(f"best other cell for {cell} at {k} kb on the test windows: {best}")
            # second pass: all measures of the model outputs and of the best other cell's map
            keep = first[~first.method.str.startswith("cell:")]
            res_b = _run_pool(a, cell, res, chunks, methods, akf, [best], True, True)
            bdf = pd.DataFrame(res_b)
            bdf = bdf[bdf.method == f"cell:{best}"].assign(method="best_cell")
            res_l = pd.concat([keep, first[first.method.str.startswith("cell:")], bdf]).to_dict("records")
            df = pd.DataFrame(res_l)
            outf.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outf if not a.limit else outf.with_suffix(".limited.gz"), index=False, float_format="%.5g")
            t = df[df.set == "test"].groupby("method").pearson.median().sort_values(ascending=False)
            log(f"median Pearson on the test windows ({cell}, {k} kb):\n{t.head(40).to_string()}")


def ceiling_one(cell: str, res: int, n_windows: int = 0, seed: int = 0):
    """Split-half reproducibility ceiling of a target on the test windows: the raw counts of every pixel are
    split with Binomial(c, 0.5) into two disjoint halves, each balanced with the map's weights and
    transformed as the replicate ceiling of the registry (window-local obs/exp, log, clip, Gaussian
    smoothing, replicate_ceiling.akita_space); Pearson of the halves over the pixels >= 2 bins off the
    diagonal, Spearman-Brown to full depth. Median over the windows."""
    import cooler
    import replicate_ceiling as rc
    k = res // 1000
    f = VN / "balanced" / f"{cell}_{k}kb.cool"
    if not f.exists():
        return None
    clr = cooler.Cooler(str(f))
    A = aligned(pd.read_csv(VN / "regions" / "sequence_regions.tsv", sep="\t"), res)
    A = A[A.set == "test"].drop_duplicates("key")
    if n_windows:
        A = A.sample(min(n_windows, len(A)), random_state=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in A.iterrows():
        reg = (r.chrom, int(r.b_lo) * res, int(r.b_hi) * res)
        try:
            C = np.asarray(clr.matrix(balance=False).fetch(reg), float)
            w = clr.bins().fetch(reg)["weight"].values.astype(float)
        except Exception:                                                     # noqa: BLE001
            continue
        n = C.shape[0]
        if n < 20 or not np.isfinite(w).any():
            continue
        U = np.triu(np.rint(C).astype(np.int64))
        H1 = rng.binomial(U, 0.5)
        H2 = U - H1
        H1 = H1 + np.triu(H1, 1).T
        H2 = H2 + np.triu(H2, 1).T
        outs = []
        for H in (H1, H2):
            B = H * w[:, None] * w[None, :]
            B = np.where(np.isfinite(B), B, np.nan)
            outs.append(rc.akita_space(B, max_d=n, smooth=True))
        iu = np.triu_indices(n, 2)
        x, y = outs[0][iu], outs[1][iu]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 100 or x[ok].std() < 1e-9 or y[ok].std() < 1e-9:
            continue
        rh = float(np.corrcoef(x[ok], y[ok])[0, 1])
        rows.append({"cell": cell, "res": res, "key": r.key, "r_half": rh, "r_full": 2 * rh / (1 + rh),
                     "n_px": int(ok.sum())})
    return pd.DataFrame(rows)


def stage_ceiling(a):
    out = VN / "scores"
    out.mkdir(parents=True, exist_ok=True)
    for cell in a.cells:
        for res in a.res:
            f = out / f"ceiling_{cell}_{res // 1000}kb.csv"
            if f.exists() and not a.limit:
                continue
            df = ceiling_one(cell, res, a.limit)
            if df is None or df.empty:
                log(f"[skip] ceiling {cell} {res}: no balanced cool")
                continue
            df.to_csv(f if not a.limit else f.with_suffix('.limited.csv'), index=False)
            log(f"ceiling {cell} {res // 1000} kb: median half-depth r {df.r_half.median():.3f}, "
                f"full-depth {df.r_full.median():.3f} ({len(df)} windows)")


# ================================================================== summary
LABEL = {"consensus12": "consensus12 alone", "best_cell": "best other cell (measured)", "akita": "Akita (sequence only)",
         "refin:all": "refin, 7 marks", "refres:all": "refres, 7 marks", "honest:all": "honest (no consensus), 7 marks",
         "diff:all": "differential channels, 7 marks"}


def _boot_diff(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    d = np.median(x[idx], 1) - np.median(y[idx], 1)
    return float(np.median(x) - np.median(y)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _load_scores(cells, res_l):
    parts = []
    for res in res_l:
        for c in cells:
            f = VN / "scores" / f"{c}_{res // 1000}kb.csv.gz"
            if f.exists():
                parts.append(pd.read_csv(f))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stage_summary(a):
    RESULTS.mkdir(parents=True, exist_ok=True)
    D = _load_scores(a.cells, a.res)
    if D.empty:
        raise SystemExit("no score tables")
    D["held"] = D.chrom.isin(CHRHO_HELD)
    # ---- per-method medians, all measures
    def table(sub, name):
        g = sub.groupby(["cell", "res", "method"])
        t = g[METRICS].median()
        t["n_windows"] = g.size()
        t.reset_index().to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.4f")
        return t.reset_index()
    T_test = table(D[D.set == "test"], "summary_test_windows")
    table(D, "summary_all_windows")                                   # test + tiles, every window once
    table(D[D.set == "tile"], "summary_whole_genome_tiles")
    table(D[D.held & (D.set == "tile")], "summary_chrho_chromosomes_tiles")
    table(D[D.held & (D.set == "test")], "summary_chrho_chromosomes_test_windows")
    # ---- primary endpoint, paired bootstrap (test windows) and the same contrasts on the whole-genome tiles
    rows = []
    for stn in ("test", "tile"):
        for c in a.cells:
            for res in a.res:
                S = D[(D.cell == c) & (D.res == res) & (D.set == stn)]
                piv = S.pivot_table(index="key", columns="method", values="pearson")
                need = ["consensus12", "refin:all", "honest:all"]
                if not all(m in piv.columns for m in need):
                    continue
                piv = piv.dropna(subset=need)
                if piv.empty:
                    continue
                others = [("consensus12", "A"), ("honest:all", "B")]
                others += [(m, "") for m in piv.columns if m not in ("refin:all", "consensus12", "honest:all")
                           and not m.startswith("cell:") and (stn == "test" or m.endswith(":all") or m == "akita")]
                for other, label in others:
                    pr = piv[["refin:all", other]].dropna()
                    if len(pr) < 10:
                        continue
                    d, lo, hi = _boot_diff(pr["refin:all"].values, pr[other].values)
                    rows.append({"set": stn, "cell": c, "res": res,
                                 "contrast": (f"{label}: " if label else "") + f"refin:all - {other}",
                                 "n_windows": len(pr), "median_refin": pr["refin:all"].median(),
                                 "median_other": pr[other].median(), "diff": d, "ci_lo": lo, "ci_hi": hi,
                                 "above": bool(lo > 0), "windows_refin_better": float((pr["refin:all"] > pr[other]).mean())})
    # chromosome-held-out models on the held-out chromosomes (same windows for every predictor)
    for stn in ("test", "tile"):
        for c in a.cells:
            for res in a.res:
                S = D[(D.cell == c) & (D.res == res) & (D.set == stn) & D.held]
                piv = S.pivot_table(index="key", columns="method", values="pearson")
                for mod in ("refin_chrho:all", "refres_chrho:all"):
                    for other in ("consensus12", "honest_chrho:all", "honest:all"):
                        if mod in piv.columns and other in piv.columns:
                            pr = piv[[mod, other]].dropna()
                            if len(pr) < 10:
                                continue
                            d, lo, hi = _boot_diff(pr[mod].values, pr[other].values)
                            rows.append({"set": stn + "_chrho_chromosomes", "cell": c, "res": res,
                                         "contrast": f"{mod} - {other}", "n_windows": len(pr),
                                         "median_refin": pr[mod].median(), "median_other": pr[other].median(),
                                         "diff": d, "ci_lo": lo, "ci_hi": hi, "above": bool(lo > 0),
                                         "windows_refin_better": float((pr[mod] > pr[other]).mean())})
    P = pd.DataFrame(rows)
    P.to_csv(RESULTS / "paired_contrasts.csv", index=False, float_format="%.4f")
    prim = P[(P.set == "test") & (P.res == 10000) & P.contrast.str.startswith(("A:", "B:"))]
    A_ok = prim[prim.contrast.str.startswith("A:")].set_index("cell").above
    B_ok = prim[prim.contrast.str.startswith("B:")].set_index("cell").above
    n = len(A_ok)
    verdict = {"samples": list(A_ok.index), "A_above_consensus12": {k: bool(v) for k, v in A_ok.items()},
               "B_above_honest": {k: bool(v) for k, v in B_ok.items()},
               "A_count": int(A_ok.sum()), "n": n,
               "A_success": bool(n and A_ok.sum() >= int(np.ceil(2 * n / 3))),
               "B_success": bool(n and B_ok.all())}
    (RESULTS / "primary_endpoint_verdict.json").write_text(json.dumps(verdict, indent=1))
    log(json.dumps(verdict))
    # ---- ceilings
    cl = []
    for c in a.cells:
        for res in a.res:
            f = VN / "scores" / f"ceiling_{c}_{res // 1000}kb.csv"
            if f.exists():
                d = pd.read_csv(f)
                cl.append({"cell": c, "res": res, "n_windows": len(d), "median_r_half": d.r_half.median(),
                           "median_r_full_SB": d.r_full.median()})
    if cl:
        pd.DataFrame(cl).to_csv(RESULTS / "ceilings.csv", index=False, float_format="%.4f")
    # ---- sample table (accessions, assay, depth, marks, size)
    inv = pd.read_csv(VN / "inventory.csv")
    rows = []
    for c in a.cells:
        if c == "k562":
            continue
        i = inv[inv["sample"] == c]
        h = i[i.kind == "hic"].iloc[0]
        chips = i[i.kind == "chip"]
        st = {}
        for res in (10000, 25000):
            f = VN / "stats" / f"{c}_{res // 1000}kb.json"
            if f.exists():
                st.update(json.loads(f.read_text()))
        rows.append({"sample": c, "Hi-C experiment": h.experiment, "Hi-C file": h.accession,
                     "assay": CELLS[c][1], "contacts_M": round(st.get("contacts", np.nan) / 1e6, 0),
                     "cis_contacts_M": round(st.get("cis_contacts", np.nan) / 1e6, 0),
                     "ice_masked_frac_25kb": round(st.get("ice_masked_frac", np.nan), 3),
                     "marks": ",".join(chips.mark), "chip_experiments": ",".join(chips.experiment),
                     "chip_files": ",".join(chips.accession), "genome": "GRCh38",
                     "hic_gb": h.size_gb, "chip_gb": round(chips.size_gb.sum(), 1)})
    pd.DataFrame(rows).to_csv(RESULTS / "samples.csv", index=False)
    # ---- markdown tables
    names = {"pc3": "PC-3", "panc1": "Panc1", "nk": "NK cell", "caco2": "Caco-2", "k562": "K562 (pipeline check)"}
    cells = [c for c in a.cells if c in set(D.cell)]
    ceil = pd.DataFrame(cl) if cl else pd.DataFrame(columns=["cell", "res", "median_r_full_SB"])

    def md(rowsl, header):
        out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rowsl]
        return out

    def med(sub, cell, res, m, col="pearson"):
        v = sub[(sub.cell == cell) & (sub.res == res) & (sub.method == m)][col]
        return f"{v.median():.3f}" if v.notna().any() else ""

    L = []
    for res in a.res:
        Tt = D[(D.set == "test") & (D.res == res)]
        L.append(f"\n### Median per-window Pearson, {res // 1000} kb, Akita test windows\n")
        order = ["consensus12", "best_cell", "akita"] + [f"{m}:{s_}" for m in MAINS for s_ in SETS]
        rr = []
        for m in order:
            v = [med(Tt, c, res, m) for c in cells]
            if any(v):
                rr.append([LABEL.get(m, m)] + v)
        nw = [str(int(Tt[(Tt.cell == c) & (Tt.method == "consensus12")].shape[0])) for c in cells]
        rr.append(["windows"] + nw)
        cc = [(f"{ceil[(ceil.cell == c) & (ceil.res == res)].median_r_full_SB.iloc[0]:.3f}"
               if len(ceil[(ceil.cell == c) & (ceil.res == res)]) else "") for c in cells]
        rr.append(["split-half ceiling (Spearman-Brown)"] + cc)
        L += md(rr, ["predictor"] + [names.get(c, c) for c in cells])
        L.append(f"\n### Structure measures, {res // 1000} kb, Akita test windows (median over windows)\n")
        rr = []
        for m in ["consensus12", "best_cell", "akita", "honest:all", "refin:all", "refres:all", "diff:all"]:
            for c in cells:
                v = [med(Tt, c, res, m, k) for k in METRICS]
                if any(v):
                    rr.append([LABEL.get(m, m), names.get(c, c)] + v)
        L += md(rr, ["predictor", "sample"] + METRICS)
    L.append("\n### Chromosomes 2, 3, 11, 14, 17 (held out of the chrho models), 10 kb\n")
    H = D[(D.res == 10000) & D.held]
    rr = []
    for st in ("test", "tile"):
        for m in ["consensus12", "honest_chrho:all", "refin_chrho:all", "refres_chrho:all", "honest:all", "refin:all",
                  "refres:all", "diff:all"]:
            v = [med(H[H.set == st], c, 10000, m) for c in cells]
            if any(v):
                rr.append([("test windows" if st == "test" else "1 Mb tiles"), LABEL.get(m, m)] + v)
    L += md(rr, ["windows", "predictor"] + [names.get(c, c) for c in cells])
    L.append("\n### Whole genome, 1 Mb tiles (chr1-22), median Pearson\n")
    rr = []
    for res in a.res:
        for m in ["consensus12", "honest:all", "refin:all", "refres:all", "diff:all", "akita"]:
            v = [med(D[(D.set == "tile")], c, res, m) for c in cells]
            if any(v):
                rr.append([f"{res // 1000} kb", LABEL.get(m, m)] + v)
    L += md(rr, ["resolution", "predictor"] + [names.get(c, c) for c in cells])
    L.append("\n### Paired contrasts on the test windows (median difference of per-window Pearson, 95% bootstrap interval)\n")
    rr = []
    for _, r in P[P.contrast.str.contains("refin:all") & (P.set == "test")].iterrows():
        rr.append([names.get(r.cell, r.cell), f"{r.res // 1000} kb", r.contrast, int(r.n_windows),
                   f"{r['diff']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", f"{r.windows_refin_better:.2f}"])
    L += md(rr, ["sample", "res", "contrast", "windows", "difference", "share of windows refin better"])
    L.append("\n### Paired contrasts on the whole-genome 1 Mb tiles\n")
    rr = []
    for _, r in P[P.contrast.str.contains("refin:all") & (P.set == "tile")].iterrows():
        rr.append([names.get(r.cell, r.cell), f"{r.res // 1000} kb", r.contrast, int(r.n_windows),
                   f"{r['diff']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", f"{r.windows_refin_better:.2f}"])
    L += md(rr, ["sample", "res", "contrast", "tiles", "difference", "share of tiles refin better"])
    L.append("\n### Chromosome-held-out models on chromosomes 2, 3, 11, 14, 17: paired contrasts\n")
    rr = []
    for _, r in P[P.set.str.endswith("chrho_chromosomes")].iterrows():
        rr.append([names.get(r.cell, r.cell), f"{r.res // 1000} kb", r.set.split("_")[0], r.contrast, int(r.n_windows),
                   f"{r['diff']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"])
    L += md(rr, ["sample", "res", "windows", "contrast", "n", "difference"])
    (RESULTS / "tables.md").write_text("\n".join(L) + "\n")
    _figure(D, P, a)


def _figure(D, P, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [c for c in a.cells if c in set(D.cell)]
    names = {"pc3": "PC-3", "panc1": "Panc1", "nk": "NK cell", "caco2": "Caco-2", "k562": "K562 (check)"}
    cols = {"consensus12": "#7f7f7f", "best_cell": "#bcbd22", "akita": "#8c564b", "honest:all": "#1f77b4",
            "refin:all": "#d62728", "refres:all": "#ff7f0e", "diff:all": "#9467bd"}
    S = D[(D.res == 10000) & (D.set == "test")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    # A: median Pearson per sample and predictor
    a0 = ax[0, 0]
    meths = ["consensus12", "best_cell", "akita", "honest:all", "refin:all", "refres:all", "diff:all"]
    w = 0.8 / len(meths)
    for j, m in enumerate(meths):
        v, e = [], []
        for c in cells:
            x = S[(S.cell == c) & (S.method == m)].pearson.dropna().values
            v.append(np.median(x) if len(x) else np.nan)
            if len(x) > 5:
                rng = np.random.default_rng(1)
                bs = np.median(x[rng.integers(0, len(x), (1000, len(x)))], 1)
                e.append((np.median(x) - np.percentile(bs, 2.5), np.percentile(bs, 97.5) - np.median(x)))
            else:
                e.append((0, 0))
        a0.bar(np.arange(len(cells)) + j * w, v, w, yerr=np.array(e).T, color=cols[m], label=LABEL.get(m, m),
               capsize=1.5, error_kw={"lw": 0.7})
    a0.set_xticks(np.arange(len(cells)) + 0.4 - w / 2)
    a0.set_xticklabels([names.get(c, c) for c in cells])
    lo = np.nanmin([S[(S.method == m)].groupby("cell").pearson.median().min() for m in meths]) - 0.05
    a0.set_ylim(max(0, lo), 1.0)
    a0.set_ylabel("median per-window Pearson, 10 kb")
    a0.set_title("A  Test windows, identical pixels", loc="left", fontsize=10)
    a0.legend(fontsize=7, loc="lower right", frameon=False, ncol=2)
    # B: paired difference refin - consensus12 and refin - honest per window
    a1 = ax[0, 1]
    pos = 0
    ticks, tl = [], []
    for c in cells:
        sub = S[S.cell == c].pivot_table(index="key", columns="method", values="pearson")
        for k, (o, col) in enumerate((("consensus12", cols["consensus12"]), ("honest:all", cols["honest:all"]))):
            if "refin:all" in sub and o in sub:
                dd = (sub["refin:all"] - sub[o]).dropna().values
                bp = a1.boxplot([dd], positions=[pos], widths=0.7, showfliers=False, patch_artist=True)
                for b in bp["boxes"]:
                    b.set_facecolor(col)
                    b.set_alpha(0.6)
                for mdn in bp["medians"]:
                    mdn.set_color("black")
            pos += 1
        ticks.append(pos - 1.5)
        tl.append(names.get(c, c))
        pos += 1
    a1.axhline(0, color="black", lw=0.7)
    a1.set_xticks(ticks)
    a1.set_xticklabels(tl)
    a1.set_ylabel("per-window difference in Pearson (refin)")
    a1.set_title("B  refin minus consensus12 (grey), minus honest (blue)", loc="left", fontsize=10)
    # C: structure measures
    a2 = ax[1, 0]
    mets = ["scc", "insulation_corr", "boundary_f1", "loop_enrich_corr", "compartment_corr"]
    meths2 = ["consensus12", "akita", "honest:all", "refin:all", "diff:all"]
    w2 = 0.8 / len(meths2)
    for j, m in enumerate(meths2):
        v = [S[(S.method == m)][mm].median() for mm in mets]
        a2.bar(np.arange(len(mets)) + j * w2, v, w2, color=cols[m], label=LABEL.get(m, m))
    a2.set_xticks(np.arange(len(mets)) + 0.4 - w2 / 2)
    a2.set_xticklabels(["SCC", "insulation r", "boundary F1", "loop enrich r", "compartment r"], fontsize=8)
    a2.set_ylabel("median over windows and samples, 10 kb")
    a2.set_title("C  Structure measures (pooled over the samples)", loc="left", fontsize=10)
    a2.legend(fontsize=7, frameon=False)
    # D: whole-genome tiles vs held-out chromosomes with the chrho models
    a3 = ax[1, 1]
    H = D[(D.res == 10000) & (D.set == "tile") & D.chrom.isin(CHRHO_HELD)]
    meths3 = ["consensus12", "honest_chrho:all", "refin_chrho:all", "refres_chrho:all"]
    w3 = 0.8 / len(meths3)
    for j, m in enumerate(meths3):
        v = [H[(H.cell == c) & (H.method == m)].pearson.median() for c in cells]
        a3.bar(np.arange(len(cells)) + j * w3, v, w3, color={"consensus12": cols["consensus12"],
               "honest_chrho:all": cols["honest:all"], "refin_chrho:all": cols["refin:all"],
               "refres_chrho:all": cols["refres:all"]}[m], label=m.replace(":all", ""))
    a3.set_xticks(np.arange(len(cells)) + 0.4 - w3 / 2)
    a3.set_xticklabels([names.get(c, c) for c in cells])
    a3.set_ylim(max(0, np.nanmin([H[H.method == m].groupby("cell").pearson.median().min() for m in meths3]) - 0.05), 1)
    a3.set_ylabel("median per-tile Pearson, 10 kb")
    a3.set_title("D  Chromosomes 2, 3, 11, 14, 17 (chrho models), 1 Mb tiles", loc="left", fontsize=10)
    a3.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(RESULTS / "fig_independent_validation.png", dpi=200)
    log("figure written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["regions", "predict", "akita", "cache", "score", "ceiling", "summary"])
    ap.add_argument("--cells", nargs="*", default=NEW)
    ap.add_argument("--res", nargs="*", type=int, default=[10000, 25000])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: random subset of regions")
    ap.add_argument("--with-others", action="store_true", help="cache: also the other cells' maps")
    ap.add_argument("--test-only", action="store_true", help="score: only the 413 test windows")
    ap.add_argument("--part", type=int, default=-1, help="akita: this part of --nparts")
    ap.add_argument("--nparts", type=int, default=2)
    ap.add_argument("--regionset", choices=["all", "test", "tiles"], default="test")
    ap.add_argument("--pred-dir", default=None)
    a = ap.parse_args()
    if a.stage == "regions":
        stage_regions()
    elif a.stage == "predict":
        stage_predict(a)
    elif a.stage == "akita":
        stage_akita(a)
    else:
        globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
