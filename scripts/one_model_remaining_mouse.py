"""Transfer to mouse (mm10) of the release checkpoint all12one_fullrd (registry sections 75 and 98).

Protocol of section 75 (scripts/mouse_transfer.py, unchanged functions where possible): the 6 038 Akita-v2 tiles,
central 1 048 576 bp, 5 629 autosome windows at 10 kb, ENCODE blacklist bins excluded, identical pixels, per-window
Pearson and the structure measures of flexcible.structure, medians over windows, paired bootstrap over windows.
mESC E14, CH12.LX and adult heart with the seven marks CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3,
H3K9me3; human JASPAR motifs on mm10, human normalisers; the leave-one-out mean of the other mouse samples as
reference. The checkpoint is used unchanged (13 marks + Meth, cohesin fraction 1): the six marks and Meth that mouse
does not have are absent.

One pass per sample, size and variant with keep sets; every set exists twice, with the mouse consensus (`<set>`)
and with the reference plane at 0 (`<set>_noref`, `flexcible predict --no-reference`, the use that corresponds to the
paper's model without consensus):
  rel        marks of the sample (all7, ctcf_k27ac, ctcf, none)
  rel_shuf   marks rolled by one common offset of 25 to 75 percent of each chromosome (as in section 75)
  rel_xm     marks of another mouse sample (mesc <- heart, ch12 <- mesc, heart <- ch12)
The predictions of section 75 (honest, refin, Akita heads, replicate experiment) are re-scored in the same pass so
that every method shares the pixels of the release methods.

Stages: views (on the machine that predicts), store (local: binned bigWig signal, so that the multi-GB
bigWigs need not be copied to the GPU host), predict (GPU host), score, summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import mouse_transfer as mt                                              # noqa: E402

HM = mt.HM
VM = mt.VM
MR = mt.MR
REL = "all12one_fullrd"
OUT = REPO / "results/one_model/remaining/mouse"
STORE = Path(os.environ.get("MT_STORE", str(Path.home() / "scratch/mt_store")))
PRED = "pred_rel"
SCORES = "scores_rel"
SETS = {"all7": mt.CORE7, "ctcf_k27ac": ["CTCF", "H3K27ac"], "ctcf": ["CTCF"], "none": []}
VARIANTS = {"rel": ("main", ["all7", "ctcf_k27ac", "ctcf", "none"]), "rel_shuf": ("shuf", ["all7"]),
            "rel_xm": ("xm", ["all7"])}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def view_dir(res):
    return VM / "views_rel" / f"{REL}_{mt.kb(res)}"


# ---------------------------------------------------------------- views (as mouse_transfer.stage_views)
def stage_views(a):
    import yaml
    for res in a.res:
        src = MR / f"{REL}_{mt.kb(res)}" / "checkpoint-best"
        dst = view_dir(res) / "checkpoint-best"
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("model.pt", "norm_tracks.json", "channels.json"):
            if not (dst / fn).exists():
                shutil.copyfile(src / fn, dst / fn) if not a.link else (dst / fn).symlink_to(src / fn)
        nm = dst / "norm_matrix_mt_target.json"
        if not nm.exists():
            shutil.copyfile(src / "norm_matrix_gm12878.json", nm)
        cfg = yaml.safe_load((src / "config.yaml").read_text())
        first = dict(cfg["data"]["samples"][0])
        first["reference_cool"] = None
        first["extra_bigwig_dirs"] = []
        tg = dict(first, name="mt_target", cool_path=str(mt.oe_cool("heart", res)), bigwig_dir=str(VM / mt.MARKS["heart"]),
                  missing_tracks=[], zeroed_tracks=[], assay="hic", fasta=None,
                  reference_cool=str(mt.loo_cool("heart", res)))
        samples = [tg]
        for c in mt.POOL:
            d = VM / mt.MARKS[c]
            have = {p.stem for p in d.glob("*.bigwig")}
            samples.append(dict(first, name=c, cool_path=str(mt.oe_cool(c, res)), bigwig_dir=str(d),
                                missing_tracks=[t for t in mt.ALL13 if t not in have], zeroed_tracks=[],
                                assay=mt.ASSAY[c], fasta=None, reference_cool=str(mt.loo_cool(c, res))))
        cfg["data"]["samples"] = samples
        cfg["sequence"]["fasta"] = str(VM / "ref/mm10.fa")
        cfg["sequence"]["cache_dir"] = str(VM / "seqcache")
        cfg["train"]["cache_dir"] = str(VM / "cfgcache" / f"{REL}_{mt.kb(res)}")
        cfg["train"]["output_dir"] = str(VM / "cfgcache" / f"{REL}_{mt.kb(res)}_out")
        (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        log(f"view {dst}")


# ---------------------------------------------------------------- store (local)
def store_key(dirname, track, chrom, n_bins, binsize):
    return STORE / dirname / f"{track}__{chrom}__{n_bins}__{binsize}.npy"


def stage_store(a):
    import cooler
    import flexcible.tracks as pc
    for res in a.res:
        for cell in mt.POOL:
            clr = cooler.Cooler(str(mt.oe_cool(cell, res)))
            d = VM / mt.MARKS[cell]
            for chrom in mt.CHRS:
                nb = int(clr.extent(chrom)[1] - clr.extent(chrom)[0])
                for t in mt.CORE7:
                    if not (d / f"{t}.bigwig").exists():
                        continue                              # absent mark of this sample
                    f = store_key(d.name, t, chrom, nb, res)
                    if f.exists():
                        continue
                    f.parent.mkdir(parents=True, exist_ok=True)
                    arr = pc.read_track_binned(str(d / f"{t}.bigwig"), chrom, nb, res)
                    np.save(f.with_name(f.name + ".tmp.npy"), arr)
                    os.replace(f.with_name(f.name + ".tmp.npy"), f)
            log(f"store {cell} {res}")


# ---------------------------------------------------------------- predict (GPU host)
def stage_predict(a):
    py = os.environ.get("HICMULTI_PY", sys.executable)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               MT_STORE=str(STORE))
    for res in a.res:
        k = mt.kb(res)
        ck = view_dir(res) / "checkpoint-best"
        for cell in a.cells:
            for vname, (var, sets) in VARIANTS.items():
                if var not in a.variants:
                    continue
                marks = cell if var != "xm" else mt.XM_NEXT[cell]
                bw = VM / mt.MARKS[marks]
                out = VM / PRED / k / vname / cell
                if (out / "DONE").exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                have = {p.stem for p in bw.glob("*.bigwig")}
                ks = []
                for n in sets:
                    t = ",".join(x for x in SETS[n] if x in have)
                    ks += [f"{n}={t}", f"{n}_noref={t}"]
                cmd = [py, "scripts/one_model_remaining_mouse_predict.py", "--config", str(ck / "config.yaml"),
                       "--checkpoint", str(ck), "--cell", cell, "--bigwig-dir", str(bw),
                       "--target", str(mt.oe_cool(cell, res)), "--tta", "--assay", mt.ASSAY[cell],
                       "--binsize", str(res), "--device", a.device,
                       "--regions", str(VM / "regions" / f"crop_{k}_auto.tsv"), "--out-dir", str(out),
                       "--save-pred", str(out / "{set}.npz"), "--keep-sets", *ks,
                       "--reference", str(mt.loo_cool(cell, res)), "--set-batch", str(a.set_batch)]
                e = dict(env)
                if var == "shuf":
                    e["MT_SHUFFLE"] = "1"
                log(f"[run] {vname} {cell} {k}")
                with open(out / "predict.log", "w") as fh:
                    fh.write("[cmd] " + " ".join(cmd) + "\n")
                    fh.flush()
                    r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=e)
                want = [f"{n}{s}" for n in sets for s in ("", "_noref")]
                if r.returncode == 0 and all((out / f"{s}.npz").exists() for s in want):
                    (out / "DONE").write_text(time.strftime("%F %T"))
                else:
                    log(f"[FAILED] {vname} {cell} {k} rc={r.returncode}")


# ---------------------------------------------------------------- score (local)
OLD = ["honest:all7", "honest:ctcf_k27ac", "honest:ctcf", "honest:none", "honest_shuf:all7", "honest_xm:all7",
       "refin:all7"]


def stage_score(a):
    import multiprocessing as mp
    for res in a.res:
        k = mt.kb(res)
        Wn = mt.load_windows(res, ("auto",))
        for cell in a.cells:
            outf = VM / SCORES / f"{cell}_{k}.csv.gz"
            if outf.exists() and not a.force:
                log(f"[have] {outf.name}")
                continue
            methods = {}
            for m in OLD:                                                    # predictions of section 75
                d, _, s = m.partition(":")
                f = VM / "pred" / k / d / cell / f"{s}.npz"
                if f.exists() and (f.parent / "DONE").exists():
                    methods[m] = f
            for vname in VARIANTS:
                p = VM / PRED / k / vname / cell
                if (p / "DONE").exists():
                    for npz in sorted(p.glob("*.npz")):
                        methods[f"{vname}:{npz.stem}"] = npz
            akf = []
            rep = None
            if cell == "mesc":
                akf = [p for p in sorted((VM / "akita").glob("akita*.npz")) if "tmp" not in p.name]
                p = VM / "oe" / f"mesc_rep_proper_{k}.cool"
                rep = p if p.exists() else None
            sub = Wn if not a.limit else Wn.sample(a.limit, random_state=0)
            rows = list(zip(sub.key, sub.chrom, sub.b_lo, sub.b_hi, sub.seq_key, sub.set))
            chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
            log(f"score {cell} {k}: {len(rows)} windows, {len(methods)} model outputs")
            res_l = []
            with mp.Pool(a.procs, initializer=mt._init_worker, initargs=(cell, res, methods, akf, rep)) as pool:
                for i, r in enumerate(pool.imap_unordered(mt._score_chunk, chunks)):
                    res_l.extend(r)
                    if i % 25 == 0:
                        log(f"  {i + 1}/{len(chunks)} chunks")
            df = pd.DataFrame(res_l)
            outf.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outf, index=False, float_format="%.5g")
            t = df.groupby("method").pearson.median().sort_values(ascending=False)
            log(f"median Pearson ({cell}, {k}):\n{t.to_string()}")


# ---------------------------------------------------------------- summary
def stage_summary(a):
    import mouse_transfer_report as rep
    OUT.mkdir(parents=True, exist_ok=True)
    fr = [pd.read_csv(VM / SCORES / f"{c}_{mt.kb(r)}.csv.gz") for c in mt.POOL for r in a.res
          if (VM / SCORES / f"{c}_{mt.kb(r)}.csv.gz").exists()]
    D = pd.concat(fr, ignore_index=True)
    D = D[D.set == "auto"]
    g = D.groupby(["cell", "res", "method"])
    S = g[mt.METRICS].median().add_suffix("_median").join(g[["boundary_f1"]].mean().add_suffix("_mean")) \
        .join(g.size().rename("n_windows")).reset_index()
    S.to_csv(OUT / "summary_autosomes.csv", index=False, float_format="%.4f")
    C = pd.read_csv(REPO / "results/mouse_transfer/ceilings.csv")
    pairs = []
    contrasts = [("rel:all7_noref", "distance_decay"), ("rel:all7_noref", "rel_shuf:all7_noref"),
                 ("rel:all7_noref", "rel_xm:all7_noref"), ("rel:all7_noref", "rel:none_noref"),
                 ("rel:all7_noref", "rel:ctcf_noref"), ("rel:all7_noref", "rel:ctcf_k27ac_noref"),
                 ("rel:all7_noref", "consensus"), ("rel:all7_noref", "distance_oracle"),
                 ("rel:all7", "consensus"), ("rel:all7", "rel:all7_noref"), ("rel:all7", "rel_shuf:all7"),
                 ("rel:all7", "rel_xm:all7"), ("rel:all7", "rel:none"),
                 ("rel:all7_noref", "honest:all7"), ("rel:all7", "refin:all7"),
                 ("honest:all7", "distance_decay"), ("honest:all7", "honest_shuf:all7"),
                 ("honest:all7", "honest_xm:all7"), ("honest:all7", "consensus"), ("refin:all7", "consensus"),
                 ("refin:all7", "honest:all7"),
                 ("rel:all7_noref", "akita:Hsieh_mESC_uC"), ("rel:all7_noref", "akita:Bonev_ES"),
                 ("rel:all7", "akita:Bonev_ES"), ("honest:all7", "akita:Hsieh_mESC_uC"),
                 ("honest:all7", "akita:Bonev_ES"), ("rel:all7_noref", "replicate_experiment"),
                 ("honest:all7", "replicate_experiment")]
    for res in a.res:
        for c in mt.POOL:
            for m1, m2 in contrasts:
                r = rep.paired(D, c, res, m1, m2)
                if r:
                    pairs.append(r)
    P = pd.DataFrame(pairs)
    P.to_csv(OUT / "paired_contrasts.csv", index=False, float_format="%.4f")
    rows = []
    for c in mt.POOL:
        for metric in rep.STRUCT:
            fn = np.mean if metric == "boundary_f1" else np.median
            for m in ("rel:all7_noref", "rel_shuf:all7_noref", "rel:none_noref", "rel_xm:all7_noref", "rel:all7",
                      "honest:all7", "honest_shuf:all7", "honest_xm:all7", "refin:all7", "consensus",
                      "distance_oracle", "akita:Bonev_ES"):
                v = rep.window_vec(D, c, 10000, m, metric).dropna()
                if len(v):
                    rows.append({"cell": c, "metric": metric, "method": m, "stat": float(fn(v.values)), "n": len(v)})
    pd.DataFrame(rows).to_csv(OUT / "structure_classes.csv", index=False, float_format="%.4f")
    cls = []
    for c in mt.POOL:
        for metric in rep.STRUCT:
            fn = np.mean if metric == "boundary_f1" else np.median
            for own, refs in (("rel:all7_noref", ("rel_shuf:all7_noref", "rel:none_noref", "rel_xm:all7_noref")),
                              ("honest:all7", ("honest_shuf:all7", "honest:none", "honest_xm:all7"))):
                for ref in refs:
                    r = rep.paired(D, c, 10000, own, ref, metric, fn=fn)
                    if r:
                        r["ref"] = ref
                        cls.append(r)
    pd.DataFrame(cls).to_csv(OUT / "structure_class_contrasts.csv", index=False, float_format="%.4f")
    verdict = {"definition": "rel:all7_noref (release checkpoint, 10 kb, autosome windows) above distance_decay and "
                             "above rel_shuf:all7_noref in every sample (difference of medians, 95% paired "
                             "bootstrap interval excludes 0)", "samples": {}}
    ok_all = True
    for c in mt.POOL:
        e = {}
        for ref in ("distance_decay", "rel_shuf:all7_noref"):
            r = rep.paired(D, c, 10000, "rel:all7_noref", ref)
            e[ref] = r
            ok_all &= bool(r and r["lo"] > 0)
        cc = C[(C.cell == c) & (C.res == 10000)]
        med = float(rep.window_vec(D, c, 10000, "rel:all7_noref").median())
        e["release_noref_median"] = med
        if len(cc):
            e["ceiling_full_depth_median"] = float(cc.full_depth_median.iloc[0])
            e["fraction_of_ceiling"] = med / float(cc.full_depth_median.iloc[0])
        verdict["samples"][c] = e
    verdict["endpoint_met"] = bool(ok_all)
    (OUT / "primary_endpoint_verdict.json").write_text(json.dumps(verdict, indent=1))
    log(f"endpoint met: {ok_all}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["views", "store", "predict", "score", "summary"])
    ap.add_argument("--res", type=int, nargs="+", default=[10000, 25000])
    ap.add_argument("--cells", nargs="+", default=mt.POOL)
    ap.add_argument("--variants", nargs="+", default=["main", "shuf", "xm"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--set-batch", type=int, default=8)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()
    {"views": stage_views, "store": stage_store, "predict": stage_predict, "score": stage_score,
     "summary": stage_summary}[a.stage](a)


if __name__ == "__main__":
    main()
