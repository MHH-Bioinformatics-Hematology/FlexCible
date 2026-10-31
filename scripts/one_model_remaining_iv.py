"""Pre-registered new cell types (registry section 74) on the release checkpoint all12one_fullrd (section 98).

The protocol of scripts/independent_validation.py is kept: PC-3, Panc1, NK, Caco-2, the seven marks, the
413 Akita test windows (central 917 504 bp), consensus12 as Hi-C reference, identical pixels, per-window
Pearson, paired bootstrap over windows (2000, seed 0). The release checkpoint is one network for all
bin sizes; its views `runs/multicell_reference/all12one_fullrd_<k>kb` (k = 10, 25) are used unchanged. Two uses
of the same checkpoint are predicted in ONE pass per cell and size (keep sets): with the consensus12 plane
(`rel:<set>`) and with the reference plane at 0, `flexcible predict --no-reference` (`rel:<set>_noref`);
the four mark sets are those of section 74 (all seven marks, CTCF + H3K27ac, CTCF, none). Methylation is
not available for the new cells (absent), the cohesin-fraction row is 1 (intact).

Existing predictions, scores and tables of section 74 are not touched; everything new is written to
`validation_new/pred_rel`, `validation_new/scores_rel` and `results/one_model/remaining/new_cell_types/`.

    # on the remote GPU host (HICMULTI_HM = data root, IV_BW / IV_CACHE = local copies):
    python scripts/one_model_remaining_iv.py predict --cells pc3 panc1 nk caco2 --res 10000 25000
    # locally, after the predictions were copied to validation_new/pred_rel:
    python scripts/one_model_remaining_iv.py score --cells pc3 panc1 nk caco2 --res 10000 25000 --procs 6
    python scripts/one_model_remaining_iv.py summary
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
import independent_validation as iv                                     # noqa: E402

HM = iv.HM
VN = iv.VN
MR = iv.MR
OUT = REPO / "results/one_model/remaining/new_cell_types"
VIEW = os.environ.get("REL_VIEW", "all12one_fullrd")
SETS = iv.SETS
REL_SETS = [s for n in SETS for s in (n, n + "_noref")]
PRED = "pred_rel"
SCORES = "scores_rel"
NAMES = {"pc3": "PC-3", "panc1": "Panc1", "nk": "NK cell", "caco2": "Caco-2"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- predict (remote GPU)
def stage_predict(a):
    py = os.environ.get("HICMULTI_PY", sys.executable)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads))
    for res in a.res:
        k = res // 1000
        view = MR / f"{VIEW}_{k}kb"
        ck = view / "checkpoint-best"
        c12 = iv._loc(iv.OE / f"consensus12_proper_{k}kb.cool")
        for cell in a.cells:
            bw, assay = iv.CELLS[cell]
            lb = Path(os.environ.get("IV_BW", str(Path.home() / "scratch/iv_bw"))) / cell
            if (lb / "COPIED").exists():
                bw = lb
            tgt = iv.cool_of(cell, res)
            if not tgt.exists() or not bw.exists():
                log(f"[skip] {cell} {k}kb: target or bigWigs missing")
                continue
            have = {t for t in iv.CORE7 if (bw / f"{t}.bigwig").exists()}
            out = VN / PRED / f"{k}kb" / "rel" / cell
            if (out / "DONE").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            ks = []
            for n, tr in SETS.items():
                t = ",".join(x for x in tr if x in have)
                ks += [f"{n}={t}", f"{n}_noref={t}"]
            regs = VN / "regions" / f"crop_{k}kb_test.tsv"
            cmd = [py, "scripts/independent_validation_predict.py", "--config", str(ck / "config.yaml"),
                   "--checkpoint", str(ck), "--cell", cell, "--bigwig-dir", str(bw), "--target", str(tgt),
                   "--tta", "--assay", assay, "--binsize", str(res), "--device", a.device,
                   "--regions", str(regs), "--out-dir", str(out), "--save-pred", str(out / "{set}.npz"),
                   "--keep-sets", *ks, "--reference", str(c12), "--set-batch", str(a.set_batch)]
            log(f"[run] {cell} {k}kb")
            with open(out / "predict.log", "w") as fh:
                fh.write("[cmd] " + " ".join(cmd) + "\n")
                fh.flush()
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=env)
            if r.returncode == 0 and all((out / f"{s}.npz").exists() for s in REL_SETS):
                (out / "DONE").write_text(time.strftime("%F %T"))
            else:
                log(f"[FAILED] {cell} {k}kb rc={r.returncode}")


# ---------------------------------------------------------------- score (local CPU)
def stage_score(a):
    for res in a.res:
        k = res // 1000
        A = iv.aligned(pd.read_csv(VN / "regions" / "sequence_regions.tsv", sep="\t"), res).drop_duplicates("key")
        A = A[A.set == "test"]
        for cell in a.cells:
            outf = VN / SCORES / f"{cell}_{k}kb.csv.gz"
            if outf.exists():
                log(f"[have] {outf.name}")
                continue
            methods = {}
            for m in iv.MAINS:                                           # the models of section 74, same pixels
                for s in SETS:
                    f = VN / "pred" / f"{k}kb" / m / cell / f"{s}.npz"
                    if f.exists() and (f.parent / "DONE").exists():
                        methods.setdefault(f"{m}:{s}", []).append(f)
            for s in REL_SETS:
                f = VN / PRED / f"{k}kb" / "rel" / cell / f"{s}.npz"
                if f.exists() and (f.parent / "DONE").exists():
                    methods.setdefault(f"rel:{s}", []).append(f)
            akf = [f for f in sorted((VN / "akita").glob("akita*.npz")) if "limited" not in f.name
                   and "tmp" not in f.name]
            rows = list(zip(A.key, A.chrom, A.b_lo, A.b_hi, A.seq_key, A.set))
            chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
            log(f"score {cell} {k}kb: {len(rows)} windows, {len(methods)} model outputs")
            others = [o for o in iv.OTHERS if o != cell and iv.cool_of(o, res).exists()]
            first = pd.DataFrame(iv._run_pool(a, cell, res, chunks, methods, akf, others, False))
            ot = first[first.method.str.startswith("cell:")]
            best = ot.groupby("method").pearson.median().idxmax().split(":", 1)[1]
            log(f"best other cell for {cell} at {k} kb: {best}")
            keep = first[~first.method.str.startswith("cell:")]
            bdf = pd.DataFrame(iv._run_pool(a, cell, res, chunks, methods, akf, [best], True, True))
            bdf = bdf[bdf.method == f"cell:{best}"].assign(method="best_cell")
            df = pd.concat([keep, first[first.method.str.startswith("cell:")], bdf])
            outf.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outf, index=False, float_format="%.5g")
            t = df.groupby("method").pearson.median().sort_values(ascending=False)
            log(f"median Pearson ({cell}, {k} kb):\n{t.head(30).to_string()}")


# ---------------------------------------------------------------- summary
def stage_summary(a):
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.concat([pd.read_csv(VN / SCORES / f"{c}_{r // 1000}kb.csv.gz") for r in a.res for c in a.cells
                   if (VN / SCORES / f"{c}_{r // 1000}kb.csv.gz").exists()], ignore_index=True)
    g = D.groupby(["cell", "res", "method"])
    T = g[iv.METRICS].median()
    T["n_windows"] = g.size()
    T = T.reset_index()
    T.to_csv(OUT / "summary_test_windows.csv", index=False, float_format="%.4f")
    rows = []
    for c in a.cells:
        for res in a.res:
            S = D[(D.cell == c) & (D.res == res)]
            piv = S.pivot_table(index="key", columns="method", values="pearson")
            need = ["consensus12", "rel:all", "rel:all_noref", "refin:all", "honest:all"]
            if not all(m in piv.columns for m in need):
                continue
            piv = piv.dropna(subset=need)
            pairs = [("A: rel:all - consensus12", "rel:all", "consensus12"),
                     ("B: rel:all - rel:all_noref", "rel:all", "rel:all_noref"),
                     ("rel:all - refin:all", "rel:all", "refin:all"),
                     ("rel:all_noref - honest:all", "rel:all_noref", "honest:all"),
                     ("rel:all_noref - consensus12", "rel:all_noref", "consensus12"),
                     ("rel:all - akita", "rel:all", "akita"), ("rel:all_noref - akita", "rel:all_noref", "akita"),
                     ("rel:all - best_cell", "rel:all", "best_cell"),
                     ("refin:all - consensus12 (section 74 A)", "refin:all", "consensus12"),
                     ("refin:all - honest:all (section 74 B)", "refin:all", "honest:all")]
            for name, x, y in pairs:
                if x not in piv or y not in piv:
                    continue
                pr = piv[[x, y]].dropna()
                d, lo, hi = iv._boot_diff(pr[x].values, pr[y].values)
                rows.append({"cell": c, "res": res, "contrast": name, "n_windows": len(pr),
                             "median_x": pr[x].median(), "median_y": pr[y].median(), "diff": d, "ci_lo": lo,
                             "ci_hi": hi, "above": bool(lo > 0), "share_x_better": float((pr[x] > pr[y]).mean())})
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "paired_contrasts.csv", index=False, float_format="%.4f")
    A_ok = P[(P.res == 10000) & P.contrast.str.startswith("A: ")].set_index("cell").above
    B_ok = P[(P.res == 10000) & P.contrast.str.startswith("B: ")].set_index("cell").above
    verdict = {"samples": list(A_ok.index), "A_above_consensus12": {k: bool(v) for k, v in A_ok.items()},
               "B_above_noref": {k: bool(v) for k, v in B_ok.items()}, "A_count": int(A_ok.sum()), "n": len(A_ok),
               "A_success": bool(len(A_ok) and A_ok.sum() >= int(np.ceil(2 * len(A_ok) / 3))),
               "B_success": bool(len(B_ok) and B_ok.all())}
    (OUT / "primary_endpoint_verdict.json").write_text(json.dumps(verdict, indent=1))
    # ceilings are those of section 74 (targets unchanged)
    ce = pd.read_csv(REPO / "results/independent_validation/ceilings.csv")
    # markdown
    L = []
    for res in a.res:
        L.append(f"\n### Median per-window Pearson, {res // 1000} kb, 413 Akita test windows\n")
        L.append("| predictor | " + " | ".join(NAMES[c] for c in a.cells) + " |")
        L.append("|---|" + "---|" * len(a.cells))
        order = ["consensus12", "best_cell", "akita", "honest:all", "refin:all", "refres:all", "diff:all",
                 "rel:all", "rel:ctcf_k27ac", "rel:ctcf", "rel:none", "rel:all_noref", "rel:ctcf_k27ac_noref",
                 "rel:ctcf_noref", "rel:none_noref"]
        for m in order:
            v = []
            for c in a.cells:
                x = T[(T.cell == c) & (T.res == res) & (T.method == m)].pearson
                v.append(f"{x.iloc[0]:.3f}" if len(x) else "")
            if any(v):
                L.append(f"| {m} | " + " | ".join(v) + " |")
        cc = [f"{ce[(ce.cell == c) & (ce.res == res)].median_r_full_SB.iloc[0]:.3f}" for c in a.cells]
        L.append("| split-half ceiling | " + " | ".join(cc) + " |")
        L.append(f"\n### Structure measures, {res // 1000} kb (median over windows)\n")
        L.append("| predictor | sample | " + " | ".join(iv.METRICS) + " |")
        L.append("|---|---|" + "---|" * len(iv.METRICS))
        for m in ["consensus12", "best_cell", "akita", "refin:all", "honest:all", "rel:all", "rel:all_noref"]:
            for c in a.cells:
                x = T[(T.cell == c) & (T.res == res) & (T.method == m)]
                if len(x):
                    L.append(f"| {m} | {NAMES[c]} | " + " | ".join(f"{x[k_].iloc[0]:.3f}" for k_ in iv.METRICS) + " |")
    L.append("\n### Paired contrasts (median difference of per-window Pearson, 95% bootstrap interval over windows)\n")
    L.append("| sample | res | contrast | windows | difference | share of windows x better |")
    L.append("|---|---|---|---|---|---|")
    for _, r in P.iterrows():
        L.append(f"| {NAMES[r.cell]} | {r.res // 1000} kb | {r.contrast} | {int(r.n_windows)} | "
                 f"{r['diff']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.share_x_better:.2f} |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    log(json.dumps(verdict))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["predict", "score", "summary"])
    ap.add_argument("--cells", nargs="+", default=iv.NEW)
    ap.add_argument("--res", nargs="+", type=int, default=[10000, 25000])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--set-batch", type=int, default=8)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    {"predict": stage_predict, "score": stage_score, "summary": stage_summary}[a.stage](a)


if __name__ == "__main__":
    main()
