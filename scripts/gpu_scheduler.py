"""Dependency-driven job scheduler for the 12-cell experiments (both GPUs).

GPU jobs (two-GPU trainings, one at a time) are started in priority order,
the first whose inputs exist, the moment the GPUs are free; nothing waits on
a precompute or a consensus build: CPU preparation jobs (caches, cascade
priors, evaluations, analyses) run concurrently as soon as their inputs exist,
up to --cpu-slots at a time. Every job is skipped when its output exists, so
the scheduler can be restarted at any time. Milestones go to
runs/multicell_reference/driver.log; STATUS.md is refreshed after each job.

Priority (GPU): all12reflearn 25/10 kb, the multi-resolution model
all12multires, the cascade (fold models 25 kb, final 10 kb, fold models
10 kb, final 5 kb, fold models 5 kb, final 2 kb), the 5 and 2 kb pairs
(all12refres, all12honest), all12refin 25/10 kb. A lower-priority job runs
only while every higher one still waits for its inputs.

    python scripts/gpu_scheduler.py [--wait-pid PID]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade as cs                                                 # noqa: E402
import multicell_expanded as mx                                      # noqa: E402

REPO, PY = mx.REPO, mx.PY
MR = mx.REF_OUT
CACHE = mx.CACHE
LOG = MR / "driver.log"
OUTLOG = MR / "gpu_scheduler.out"
A_CELLS = "gm12878 imr90 hct116 hff h1 hepg2 a549 k562 hmec_deep"
B_CELLS = "nhek gm23248 mcf7 cd14mono pc9 hmec huvec huvec_deep"
M = f"{PY} scripts/multicell_expanded.py"


def note(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def cache_ok(name) -> bool:
    return (CACHE / name / "manifest.json").exists()


def trained(run: str, res: int) -> bool:
    name = mx.run_name(run, res)
    si = mx.stop_info(name, 10)
    return bool(si and si[2] == "patience")


def mark(name) -> Path:
    return MR / f".done_{name}"


def is_marked(name) -> bool:
    return mark(name).exists()


STATUS = (f"{M} --stage refsummary; {PY} scripts/learned_reference.py --stage summary; "
          f"{PY} scripts/coarse_gap.py --stage summary")


def eval_cmd(run, res):
    return (f"{M} --runs {run} --res {res} --stage eval --gpus 0 --cells {A_CELLS} & "
            f"{M} --runs {run} --res {res} --stage eval --gpus 1 --cells {B_CELLS} & wait")


class Job:
    def __init__(self, name, cmd, ready, done, gpu=False, after=None, loaded=None):
        self.name, self.cmd, self.ready, self.done, self.gpu = name, cmd, ready, done, gpu
        self.after = after          # name of the job whose completion adds this one's marker
        # loaded(): False while the job is still loading its data; no CPU job
        # starts meanwhile (None: no load phase to protect)
        self.loaded = loaded

    def __repr__(self):
        return self.name


def jobs():
    J = []

    def train12(run, res, cache):
        n = f"train_{mx.run_name(run, res)}"
        cfg = mx.run_config(mx.run_name(run, res), 100, 10, 500)
        ready = lambda: cache_ok(cache)
        loaded = None
        if cfg["train"].get("stage_dir"):
            st = Path(cfg["train"]["stage_dir"]) / cache / "train" / "meta.json"
            J.append(Job(f"stage_{cache}", f"{PY} -m hicmultiome.train --config "
                                           f"{MR / (mx.run_name(run, res) + '.stage.yaml')} --stage-only",
                         lambda: cache_ok(cache),
                         (lambda st=st, r=run, k=res: st.exists() or trained(r, k))))
            import yaml
            yaml.safe_dump(cfg, open(MR / f"{mx.run_name(run, res)}.stage.yaml", "w"),
                           sort_keys=False)
            ready = (lambda st=st: cache_ok(cache) and st.exists())
            lg = MR / f"{mx.run_name(run, res)}.log"
            loaded = (lambda lg=lg: lg.exists() and "train windows (" in
                      lg.read_text(errors="replace"))
        J.append(Job(n, f"{M} --runs {run} --res {res} --stage precompute --workers 8 && "
                        f"{M} --runs {run} --res {res} --stage train --gpus 0,1",
                     ready, lambda: trained(run, res), gpu=True, loaded=loaded))
        J.append(Job(f"eval_{mx.run_name(run, res)}", eval_cmd(run, res),
                     lambda: trained(run, res), lambda: is_marked(f"eval_{mx.run_name(run, res)}")))

    def prep_cache(run, res, cache, needs=lambda: True):
        J.append(Job(f"cache_{cache}",
                     f"{M} --runs {run} --res {res} --stage precompute --workers 8",
                     needs, lambda: cache_ok(cache)))

    # --- caches (CPU, ahead of time)
    prep_cache("all12reflearn", 10000, "mc_all12reflearn_10kb",
               lambda: mx.learned_map(10000).exists())
    prep_cache("all12reflearn", 25000, "mc_all12reflearn_25kb",
               lambda: mx.learned_map(25000).exists())
    prep_cache("all12refres", 2000, "mc_all12ref_2kb",
               lambda: mx.consensus_loo("pc9", 2000).exists())
    prep_cache("all12refres", 5000, "mc_all12ref_5kb",
               lambda: mx.consensus_loo("pc9", 5000).exists())
    # --- GPU jobs in priority order, with their evaluations
    train12("all12reflearn", 25000, "mc_all12reflearn_25kb")
    train12("all12reflearn", 10000, "mc_all12reflearn_10kb")
    import multires as mr
    caches_ready = lambda: all(cache_ok(f"mc_all12ref_{k}kb") for k in (2, 5, 10, 25))
    J.append(Job("stage_all12multires", f"{PY} scripts/multires.py --stage stage",
                 caches_ready, mr.staged))
    a_ready = lambda: caches_ready() and mr.staged()
    a_log = MR / "all12multires_train.log"
    a_done = lambda: "done." in (a_log.read_text(errors="replace") if a_log.exists() else "")
    J.append(Job("train_all12multires", f"{PY} scripts/multires.py --stage train",
                 a_ready, a_done, gpu=True))
    J.append(Job("eval_all12multires", f"{PY} scripts/multires.py --stage views && "
                                       f"{PY} scripts/multires.py --stage eval",
                 a_done, lambda: is_marked("eval_all12multires")))
    # --- track dropout (registry section 22), right after model A
    drop = [("all12refres_drop", 25000, "mc_all12ref_25kb"),
            ("all12honest_drop", 25000, "mc_all12_25kb"),
            ("all12refres_drop", 10000, "mc_all12ref_10kb"),
            ("all12honest_drop", 10000, "mc_all12_10kb"),
            ("all12refres_refdrop", 25000, "mc_all12ref_25kb"),
            ("all12refres_refdrop", 10000, "mc_all12ref_10kb")]
    for run, res, cache in drop:
        train12(run, res, cache)
    drop_names = [mx.run_name(r, k) for r, k, _ in drop]
    for wave, n in (("main", 4), ("all", 6)):
        ready = (lambda n=n: all(trained(r, k) for r, k, _ in drop[:n]))
        models = " ".join(drop_names[:n])
        J.append(Job(f"drop_minimal_{wave}",
                     f"{PY} scripts/minimal_inputs.py --stage absent --models {models} && "
                     f"{PY} scripts/minimal_inputs.py --stage summary",
                     ready, (lambda w=wave: is_marked(f"drop_minimal_{w}"))))
        J.append(Job(f"drop_perturb_{wave}",
                     f"{PY} scripts/perturb_validate.py --stage predict score figures",
                     ready, (lambda w=wave: is_marked(f"drop_perturb_{w}"))))
        J.append(Job(f"drop_aml_{wave}",
                     f"{PY} scripts/aml_eval.py --stage predict --models {models} "
                     f"--jobs 2 --threads 3 && {PY} scripts/aml_eval.py --stage score perturb figures",
                     ready, (lambda w=wave: is_marked(f"drop_aml_{w}"))))
    # --- hard split (scripts/hard_split.py), before the cascade
    split_ok = {r: (lambda r=r: (mx.HARD_SPLIT_DIR / f"{r // 1000}kb" / "split.json").exists())
                for r in (25000, 10000)}
    hard = [("all12refres_drop_hard", 25000, "mc_all12ref_25kb"),
            ("all12honest_drop_hard", 25000, "mc_all12_25kb"),
            ("all12refres_drop_hard", 10000, "mc_all12ref_10kb"),
            ("all12honest_drop_hard", 10000, "mc_all12_10kb")]
    for run, res, cache in hard:
        n0 = len(J)
        train12(run, res, cache)
        base_ready = J[n0].ready
        J[n0].ready = (lambda b=base_ready, r=res: b() and split_ok[r]())
    for res in (25000, 10000):          # only if reference dropout wins (flag file)
        n0 = len(J)
        train12("all12refres_refdrop_hard", res, f"mc_all12ref_{res // 1000}kb")
        base_ready = J[n0].ready
        J[n0].ready = (lambda b=base_ready, r=res: b() and split_ok[r]()
                       and (MR / "REFDROP_WINS").exists())
    J.append(Job("hardeval", f"{PY} scripts/hard_split.py --stage hardeval --jobs 4",
                 lambda: all(trained(r, k) for r, k, _ in hard),
                 lambda: is_marked("hardeval")))
    hard_names = " ".join(mx.run_name(r, k) for r, k, _ in hard)
    hard_ready = lambda: all(trained(r, k) for r, k, _ in hard)
    J.append(Job("hard_minimal", f"{PY} scripts/minimal_inputs.py --stage absent --models "
                                 f"{hard_names} && {PY} scripts/minimal_inputs.py --stage summary",
                 hard_ready, lambda: is_marked("hard_minimal")))
    J.append(Job("hard_perturb", f"{PY} scripts/perturb_validate.py --stage predict score figures",
                 hard_ready, lambda: is_marked("hard_perturb")))
    J.append(Job("hard_aml", f"{PY} scripts/aml_eval.py --stage predict --models {hard_names} "
                             f"--jobs 2 --threads 3 && {PY} scripts/aml_eval.py --stage score "
                             "perturb figures", hard_ready, lambda: is_marked("hard_aml")))
    gate_from = len(J)          # later GPU jobs wait for the hard split to exist
    views = " ".join(str(MR / mx.run_name("all12multires", r)) for r in mx.MULTIRES)
    J.append(Job("analysis_c", f"CUDA_VISIBLE_DEVICES=1 {PY} scripts/scale_importance.py "
                               f"--views {views} --out {MR / 'scale_importance'}",
                 lambda: is_marked("eval_all12multires"),
                 lambda: (MR / "scale_importance" / "importance.png").exists()))
    # --- cascade
    for res in cs.COARSE_LEVELS:
        fine = {25000: 10000, 10000: 5000, 5000: 2000}[res]
        cache_res = "mc_all12ref_25kb" if res == 25000 else f"mc_all12casc_{res // 1000}kb"
        for f in cs.FOLDS:
            log = MR / f"{cs.fold_name(res, f)}.log"
            J.append(Job(f"train_{cs.fold_name(res, f)}",
                         f"{PY} scripts/cascade.py --stage fold --res {res} --fold {f}",
                         (lambda c=cache_res: cache_ok(c)),
                         (lambda l=log: cs.done(l)), gpu=True))
        fold_logs = [MR / f"{cs.fold_name(res, f)}.log" for f in cs.FOLDS]
        J.append(Job(f"priors_{res // 1000}kb", f"{PY} scripts/cascade.py --stage priors --res {res}",
                     (lambda ls=fold_logs: all(cs.done(l) for l in ls)),
                     (lambda r=res: all(mx.cascade_prior(c, r).exists() for c in cs.IDENT_CELLS))))
        J.append(Job(f"cache_mc_all12casc_{fine // 1000}kb",
                     f"{PY} scripts/cascade.py --stage precompute --res {fine}",
                     (lambda r=res: all(mx.cascade_prior(c, r).exists() for c in cs.IDENT_CELLS)
                      and (fine != 2000 or cache_ok("mc_all12ref_2kb"))),
                     (lambda fr=fine: cache_ok(f"mc_all12casc_{fr // 1000}kb"))))
        train12("all12casc", fine, f"mc_all12casc_{fine // 1000}kb")
    # --- pairs and the input-only reference model
    train12("all12refres", 5000, "mc_all12ref_5kb")
    train12("all12honest", 5000, "mc_all12_5kb")
    train12("all12refres", 2000, "mc_all12ref_2kb")
    train12("all12honest", 2000, "mc_all12")
    train12("all12refin", 25000, "mc_all12ref_25kb")
    train12("all12refin", 10000, "mc_all12ref_10kb")
    # --- dedicated minimal-input models (runs/benchmark/QUEUE_REQUEST.md, section 20)
    gate_minimal = len(J)
    import minimal_inputs as mi

    def mdone(run):
        f = mi.OUT / f"{run}.train.log"
        return "done." in (f.read_text(errors="replace") if f.exists() else "")
    MI = f"{PY} scripts/minimal_inputs.py"
    groups = [["min_ctcf_h3k27ac_refres_25kb"], ["min_ctcf_rad21_refres_25kb"],
              ["min_one_per_group_refres_25kb"], ["min_ctcf_h3k27ac_refres_10kb"],
              ["min_ctcf_rad21_refres_10kb"], ["min_one_per_group_refres_10kb"],
              ["min_ctcf_h3k27ac_gm2kb", "min_ctcf_rad21_gm2kb"],
              ["min_one_per_group_gm2kb", "min_motifs_only_gm2kb"],
              ["min_motifs_only_refres_25kb"], ["min_motifs_only_refres_10kb"]]
    for g in groups:
        if len(g) == 1:
            cmd = f"{MI} --stage train --run {g[0]} --gpus 0,1"
        else:              # two single-GPU trainings side by side
            cmd = (f"{MI} --stage train --run {g[0]} --gpus 0 & "
                   f"{MI} --stage train --run {g[1]} --gpus 1 & wait")
        J.append(Job("train_" + "+".join(g), cmd, lambda: True,
                     (lambda g=g: all(mdone(r) for r in g)), gpu=True))
        for r in g:
            J.append(Job(f"eval_{r}", f"{MI} --stage eval --run {r} --device cpu && "
                                      f"{MI} --stage summary",
                         (lambda r=r: mdone(r)), (lambda r=r: is_marked(f"eval_{r}"))))
    # --- Track 1b: RAD21 degron pair in training (registry section 25)
    deg = [("all12deg_refres_drop", 25000), ("all12deg_honest_drop", 25000),
           ("all12deg_refres_drop", 10000), ("all12deg_honest_drop", 10000)]
    for run, res in deg:
        prep_cache(run, res, f"mc_{run}_{res // 1000}kb")       # CPU, ahead of time
        train12(run, res, f"mc_{run}_{res // 1000}kb")
    deg_names = " ".join(mx.run_name(r, k) for r, k in deg)
    deg_ready = lambda: all(trained(r, k) for r, k in deg)
    J.append(Job("deg_perturb", f"{PY} scripts/perturb_validate.py --stage predict "
                                "predict_chip score figures",
                 deg_ready, lambda: is_marked("deg_perturb")))
    J.append(Job("deg_aml", f"{PY} scripts/aml_eval.py --stage predict --models {deg_names} "
                            f"--jobs 2 --threads 3 && {PY} scripts/aml_eval.py --stage score "
                            "perturb figures", deg_ready, lambda: is_marked("deg_aml")))
    J.append(Job("deg_minimal", f"{PY} scripts/minimal_inputs.py --stage absent --models "
                                f"{deg_names} && {PY} scripts/minimal_inputs.py --stage summary",
                 deg_ready, lambda: is_marked("deg_minimal")))
    # --- Track 2: gain beyond the consensus (registry section 26)
    t2 = [("all12refin_drop", 25000, "mc_all12ref_25kb"),
          ("all12refin_drop_ctcfac", 25000, "mc_all12ref_25kb"),
          ("all12refres_drop_varw", 25000, "mc_all12ref_25kb"),
          ("all12refin_drop", 10000, "mc_all12ref_10kb"),
          ("all12refin_drop_ctcfac", 10000, "mc_all12ref_10kb"),
          ("all12refres_drop_varw", 10000, "mc_all12ref_10kb")]
    for run, res, cache in t2:
        train12(run, res, cache)
    t2_names = " ".join(mx.run_name(r, k) for r, k, _ in t2)
    t2_ready = lambda: all(trained(r, k) for r, k, _ in t2)
    J.append(Job("t2_aml", f"{PY} scripts/aml_eval.py --stage predict --models {t2_names} "
                           f"--jobs 2 --threads 3 && {PY} scripts/aml_eval.py --stage score "
                           "figures", t2_ready, lambda: is_marked("t2_aml")))
    J.append(Job("t2_hardregions", f"{PY} scripts/hard_split.py --stage quick --jobs 4 "
                                   "--families all12refin_drop all12refin_drop_ctcfac "
                                   "all12refres_drop_varw all12refres_drop all12refin "
                                   "--tag t2", t2_ready, lambda: is_marked("t2_hardregions")))
    # --- mean DNA methylation as a 14th track (registry section 27; needs
    # scripts/methylation.py --stage fetch build)
    import methylation as me
    meth_ok = lambda: all(me.has_meth(c) for c in me.FILES)
    prep_cache("all12refres_drop_meth", 10000, "mc_all12refres_drop_meth_10kb", meth_ok)
    train12("all12refres_drop_meth", 10000, "mc_all12refres_drop_meth_10kb")
    J.append(Job("meth_eval", f"{PY} scripts/meth_eval.py --stage predict summary",
                 lambda: trained("all12refres_drop_meth", 10000) and trained("all12refres_drop", 10000),
                 lambda: is_marked("meth_eval")))
    # --- copy number as two tracks + structural-variant loss mask (registry
    # section 28; inputs from scripts/cnv_sv.py)
    import cnv_sv as cv
    # the training mask needs the five cancer training lines only; K562 (held
    # out) refreshes breakpoints.json for the analysis once its calls exist
    sv_train = [c for c in cv.SV_CELLS if c in mx.TRAIN12]
    J.append(Job("sv_breakpoints", f"{PY} scripts/cnv_sv.py --stage breakpoints summary",
                 lambda: cv.eaglec_finished(sv_train) and (cv.RES_OUT / "cnv_segments.csv").exists(),
                 lambda: (cv.SV / "breakpoints.json").exists()))
    J.append(Job("sv_breakpoints_all", f"{PY} scripts/cnv_sv.py --stage breakpoints summary",
                 lambda: cv.eaglec_finished() and (cv.SV / "breakpoints.json").exists(),
                 lambda: is_marked("sv_breakpoints_all")))
    cnv_ok = lambda: (cv.SV / "breakpoints.json").exists() and all(
        cv.has_cnv(c) for c in mx.TRAIN12)
    prep_cache("all12refres_drop_cnv", 10000, "mc_all12refres_drop_cnv_10kb", cnv_ok)
    train12("all12refres_drop_cnv", 10000, "mc_all12refres_drop_cnv_10kb")
    J.append(Job("cnv_compare", f"{PY} scripts/cnv_sv.py --stage compare",
                 lambda: is_marked("eval_all12refres_drop_cnv_10kb"),
                 lambda: is_marked("cnv_compare")))
    # --- Track 1a: HFF with cohesin ChIP (needs aml_fetch.py --cells hffcoh first)
    try:
        coh = [("all12hffcoh_refres_drop", 25000), ("all12hffcoh_refres_drop", 10000)]
        for run, res in coh:
            mx.run_config(mx.run_name(run, res), 100, 10, 500)      # raises until fetched
        for run, res in coh:
            prep_cache(run, res, f"mc_{run}_{res // 1000}kb")
            train12(run, res, f"mc_{run}_{res // 1000}kb")
        coh_names = " ".join(mx.run_name(r, k) for r, k in coh)
        coh_ready = lambda: all(trained(r, k) for r, k in coh)
        J.append(Job("coh_perturb", f"{PY} scripts/perturb_validate.py --stage predict "
                                    "predict_chip score figures",
                     coh_ready, lambda: is_marked("coh_perturb")))
        J.append(Job("coh_aml", f"{PY} scripts/aml_eval.py --stage predict --models {coh_names} "
                                f"--jobs 2 --threads 3 && {PY} scripts/aml_eval.py --stage score "
                                "perturb figures", coh_ready, lambda: is_marked("coh_aml")))
    except SystemExit:
        pass
    # washout test (registry section 25) once the degron and HFF-cohesin runs exist
    import washout as wo
    wo_ready = lambda: deg_ready() and all(
        trained("all12hffcoh_refres_drop", k) for k in (25000, 10000)) and all(
        wo.oe_path(t, r).exists() for t in wo.TIMES for r in wo.RES)
    J.append(Job("washout", f"CUDA_VISIBLE_DEVICES=1 {PY} scripts/washout.py --stage predict score",
                 wo_ready, lambda: is_marked("washout")))
    # --- structure-aware learning (registry section 30): weight grid at 25 kb,
    # weights chosen on honest validation, then the chosen-weight runs
    chosen_ok = lambda: mx.STRUCT_CHOSEN.exists()
    hm = mx.REPO / "scripts/hic_measures.py"
    hm_ok = lambda: hm.exists() and "def combined_score" in hm.read_text()
    grid = [(f"all12refin_drop_{t}_{mx.wtag(w)}", 25000)
            for t in mx.STRUCT_TERMS for w in mx.STRUCT_GRID]
    for run, res in grid:
        train12(run, res, "mc_all12ref_25kb")
    J.append(Job("struct_choose", f"{PY} scripts/structure_loss.py --stage choose",
                 lambda: all(trained(r, k) for r, k in grid), chosen_ok))
    later = [("all12refin_drop_struct3", 25000, chosen_ok),
             ("all12refin_drop_selcomb", 25000, lambda: True)]
    later += [(f"all12refin_drop_{t}", 10000, chosen_ok) for t in mx.STRUCT_TERMS]
    later += [("all12refin_drop_struct3", 10000, chosen_ok),
              ("all12refin_drop_selcomb", 10000, lambda: True),
              ("all12refin_drop_selhm", 25000, hm_ok),
              ("all12refin_drop_selhm", 10000, hm_ok)]
    for run, res, gate in later:
        n0 = len(J)
        train12(run, res, f"mc_all12ref_{res // 1000}kb")
        base_ready = J[n0].ready
        J[n0].ready = (lambda b=base_ready, g=gate: b() and g())
    for res in (25000, 10000):
        done_set = [(r, k) for r, k, _ in later if k == res and not r.endswith("selhm")]
        if res == 25000:
            done_set += grid
        J.append(Job(f"struct_eval_{res // 1000}kb",
                     f"{PY} scripts/structure_loss.py --stage evaluate summary --res {res}",
                     (lambda ds=done_set: all(trained(r, k) for r, k in ds)),
                     (lambda r=res: is_marked(f"struct_eval_{r // 1000}kb"))))
    for j in J[gate_from:]:
        if j.gpu:
            j.ready = (lambda b=j.ready: b() and split_ok[25000]() and split_ok[10000]())
    add_meth_replacement(J, train12, prep_cache)
    add_cohunk_and_fine(J, train12, prep_cache, chosen_ok)
    return J


def add_meth_replacement(J, train12, prep_cache):
    """Registry section 35, part C: how far methylation replaces ChIP. The hg19
    RRBS cells (HCT116, MCF-7, HFF) are added to the methylation channel
    (scripts/methylation.py), the cache of all12refres_drop_meth_v2 is built,
    then the dedicated 10 kb runs with Meth and the v2 model train. Queued
    after the earlier jobs and before the cohesin-unknown block."""
    import methylation as me
    import minimal_inputs as mi
    v2_ok = lambda: all(me.has_meth(c, True) for c in list(me.FILES) + list(me.HG19))
    cache = "mc_all12refres_drop_meth_v2_10kb"
    prep_cache("all12refres_drop_meth_v2", 10000, cache, v2_ok)
    runs = [mi.run_name(s, "refres_10kb") for s in mi.METH_SETS]

    def mdone(run):
        f = mi.OUT / f"{run}.train.log"
        return "done." in (f.read_text(errors="replace") if f.exists() else "")
    MI = f"{PY} scripts/minimal_inputs.py"
    for r in runs:
        J.append(Job("train_" + r, f"{MI} --stage configs_meth && {MI} --stage train --run {r} "
                                   "--gpus 0,1",
                     lambda: cache_ok(cache), (lambda r=r: mdone(r)), gpu=True))
    train12("all12refres_drop_meth_v2", 10000, cache)
    J.append(Job("meth_dedicated_eval", locked("methded",
                 f"{PY} scripts/meth_replace.py --stage dedicated dedicated_summary"),
                 lambda: all(mdone(r) for r in runs), lambda: is_marked("meth_dedicated_eval")))
    v2n = mx.run_name("all12refres_drop_meth_v2", 10000)
    J.append(Job("meth_replace_v2", locked("methrep",
                 f"{PY} scripts/meth_replace.py --stage predict summary --models {v2n} "
                 "all12refres_drop_meth_10kb all12refres_drop_10kb"),
                 lambda: trained("all12refres_drop_meth_v2", 10000),
                 lambda: is_marked("meth_replace_v2")))


def add_aml_and_chrho(J, train12, prep_cache):
    """Registry sections 36 and 37, queued after the cohesin-unknown block and
    before the 5 and 2 kb blocks: (A) AML-inclusive training, test patients of
    fold 0 (the other two folds wait for the flag file runs/multicell_reference/
    AML_CV, written when the fold-0 result is promising); (B) chromosome-held-out
    final models."""
    import aml_train as at

    def refs_done(f, k):
        return all(at.ref_path(f, k, p).exists() for p in [None] + at.train_patients(f))

    def aml_block(fold, flag=None):
        sfx = f"_f{fold}" if fold else ""
        gate = (lambda: flag is None or (MR / flag).exists())
        for res in (25000, 10000):
            k = res // 1000
            J.append(Job(f"aml_refs_f{fold}_{k}kb",
                         f"{PY} scripts/aml_train.py --stage split refs --fold {fold} --res {res}",
                         gate, lambda f=fold, k=k: refs_done(f, k * 1000)))
        order = [("all12aml_refin_drop", 25000), ("all12aml_honest_drop", 25000),
                 ("all12aml_refin_drop", 10000), ("all12aml_honest_drop", 10000)]
        for res in (25000, 10000):
            k = res // 1000
            rbase = f"all12aml_refin_drop{sfx}"
            cache = f"mc_all12aml_ref_{k}kb{sfx}"
            prep_cache(rbase, res, cache, lambda f=fold, r=res: gate() and refs_done(f, r))
            prep_cache(f"all12aml_honest_drop{sfx}", res, f"mc_all12aml_{k}kb{sfx}",
                       lambda f=fold, r=res: gate() and refs_done(f, r))
            J.append(Job(f"aml_hard_f{fold}_{k}kb",
                         f"{PY} scripts/aml_train.py --stage hard --fold {fold} --res {res}",
                         lambda c=cache: cache_ok(c),
                         lambda f=fold, r=res: at.hard_split_path(f, r).exists()))
        for run, res in order:
            k = res // 1000
            r = f"{run}{sfx}"
            cache = f"mc_all12aml_{'ref_' if 'refin' in run else ''}{k}kb{sfx}"
            n0 = len(J)
            train12(r, res, cache)
            tr = next(j for j in J[n0:] if j.gpu)
            tr.ready = (lambda b=tr.ready, f=fold, rr=res: b() and at.hard_split_path(f, rr).exists())
        names = [mx.run_name(f"{b}{sfx}", r) for b in ("all12aml_refin_drop", "all12aml_honest_drop")
                 for r in (25000, 10000)]
        J.append(Job(f"aml_eval_f{fold}",
                     locked("amleval", f"{PY} scripts/aml_train.py --stage eval --fold {fold}"),
                     lambda ns=names: all(trained(*mx.split_name(n)) for n in ns),
                     lambda f=fold: is_marked(f"aml_eval_f{f}")))
    aml_block(0)

    # --- (B) chromosome-held-out final models
    chr_order = [("all12refin_drop_chrho", 25000, "mc_all12ref_chrho_25kb"),
                 ("all12honest_drop_chrho", 25000, "mc_all12chrho_25kb"),
                 ("all12refin_drop_chrho", 10000, "mc_all12ref_chrho_10kb"),
                 ("all12honest_drop_chrho", 10000, "mc_all12chrho_10kb"),
                 ("all12refres_drop_chrho", 25000, "mc_all12ref_chrho_25kb"),
                 ("all12refres_drop_chrho", 10000, "mc_all12ref_chrho_10kb")]
    seen = set()
    for run, res, cache in chr_order:
        if cache not in seen:
            prep_cache(run, res, cache)
            seen.add(cache)
    for run, res, cache in chr_order:
        train12(run, res, cache)
    chr_names = [(r, k) for r, k, _ in chr_order]
    J.append(Job("chrho_predict_score",
                 locked("chrho", f"{PY} scripts/hic_measures.py --stage chrho_predict chrho_score "
                                 "--threads 8 --jobs 4"),
                 lambda: all(trained(r, k) for r, k in chr_names[:4]),
                 lambda: is_marked("chrho_predict_score")))
    J.append(Job("chrho_predict_score_refres",
                 locked("chrho", f"{PY} scripts/hic_measures.py --stage chrho_predict chrho_score "
                                 "--threads 8 --jobs 4"),
                 lambda: all(trained(r, k) for r, k in chr_names),
                 lambda: is_marked("chrho_predict_score_refres")))
    # 3-fold patient cross-validation, after the chromosome-held-out models and
    # only once runs/multicell_reference/AML_CV exists
    aml_block(1, "AML_CV")
    aml_block(2, "AML_CV")


def locked(name, cmd):
    """Serialise jobs that rewrite shared result tables (flock on a per-table lock)."""
    return f"flock {MR / ('.lock_' + name)} bash -c {cmd!r}"


def add_cohunk_and_fine(J, train12, prep_cache, chosen_ok):
    """Registry section 25 (1c) cohesin-unknown models at 25/10 kb, then the
    track-dropout, degron, input-only reference, cohesin-unknown and structure
    recipes at 5 kb and 2 kb (section 32). Appended after every earlier job."""
    import perturb_validate as pv
    # 5 and 2 kb need the auxin Hi-C of the degron pair at that resolution
    def aux_ok(res):
        lg = pv.OUT / "oe" / f"rad21_treated_{res // 1000}kb.log"
        return (pv.oe_path("rad21", "treated", res).exists() and lg.exists()
                and "range" in lg.read_text(errors="replace"))

    def block(run, res, cache, needs=lambda: True, own_cache=False, cache_needs=None):
        if own_cache:
            prep_cache(run, res, cache, cache_needs or needs)
        n0 = len(J)
        train12(run, res, cache)
        tr = next(j for j in J[n0:] if j.gpu)
        base = tr.ready
        tr.ready = (lambda b=base, nd=needs: b() and nd())

    def post(names, tag, aml=True, perturb=True, hard=False):
        ready = lambda ns=names: all(trained(*mx.split_name(n)) for n in ns)
        if perturb:
            J.append(Job(f"{tag}_perturb", locked("perturb",
                         f"{PY} scripts/perturb_validate.py --stage predict predict_chip score figures"),
                         ready, lambda: is_marked(f"{tag}_perturb")))
        if aml:
            ms = " ".join(names)
            J.append(Job(f"{tag}_aml", locked("aml",
                         f"{PY} scripts/aml_eval.py --stage predict --models {ms} --jobs 2 "
                         f"--threads 3 && {PY} scripts/aml_eval.py --stage score perturb figures"),
                         ready, lambda: is_marked(f"{tag}_aml")))

    # --- 1c: cohesin-unknown indicator on the degron recipe, 25 and 10 kb
    coh = [("all12cohunk_refres_drop", 25000), ("all12cohunk_honest_drop", 25000),
           ("all12cohunk_refres_drop", 10000), ("all12cohunk_honest_drop", 10000)]
    for run, res in coh:
        block(run, res, f"mc_{run}_{res // 1000}kb", own_cache=True)
    coh_names = [mx.run_name(r, k) for r, k in coh]
    post(coh_names, "cohunk")
    cready = lambda: all(trained(r, k) for r, k in coh)
    J.append(Job("cohunk_washout", f"CUDA_VISIBLE_DEVICES=1 {PY} scripts/washout.py "
                                   "--stage predict score", cready,
                 lambda: is_marked("cohunk_washout")))
    J.append(Job("cohunk_hardregions", locked("hard",
                 f"{PY} scripts/hard_split.py --stage quick --jobs 4 --families "
                 "all12refres_drop all12deg_refres_drop all12deg_honest_drop all12honest_drop "
                 "all12cohunk_refres_drop all12cohunk_honest_drop --tag cohunk"),
                 cready, lambda: is_marked("cohunk_hardregions")))

    add_aml_and_chrho(J, train12, prep_cache)

    # --- section 32: 5 kb, then 2 kb; per resolution a, c, b, d, e
    for res in (5000, 2000):
        k = res // 1000
        sfx = f"_{k}kb"
        ref_cache = f"mc_all12ref_{k}kb"
        hon_cache = f"mc_all12_{k}kb" if res == 5000 else "mc_all12"
        # (a) track dropout
        block("all12refres_drop", res, ref_cache)
        block("all12honest_drop", res, hon_cache)
        # (c) RAD21 degron in training
        block("all12deg_refres_drop", res, f"mc_all12deg_refres_drop{sfx}",
              lambda r=res: aux_ok(r), own_cache=True)
        # (b) input-only reference with track dropout
        block("all12refin_drop", res, ref_cache)
        # (d) cohesin-unknown (5 kb; 2 kb last, optional)
        if res == 5000:
            block("all12cohunk_refres_drop", res, f"mc_all12cohunk_refres_drop{sfx}",
                  lambda r=res: aux_ok(r), own_cache=True)
        # (e) structure losses (insulation + donut + compartment, chosen weights)
        block("all12refin_drop_struct3", res, ref_cache, chosen_ok)
        names = [mx.run_name(b, res) for b in (
            "all12refres_drop", "all12honest_drop", "all12deg_refres_drop", "all12refin_drop")]
        if res == 5000:
            names.append(mx.run_name("all12cohunk_refres_drop", res))
        names.append(mx.run_name("all12refin_drop_struct3", res))
        for n in names:
            post([n], f"f_{n}", aml=(res == 5000),
                 perturb=("deg" in n or "cohunk" in n or "refres_drop" in n))
        sready = lambda r=res: all(trained(b, r) for b in
                                   ("all12refin_drop", "all12refin_drop_struct3"))
        J.append(Job(f"struct_eval_{k}kb", locked("struct",
                     f"{PY} scripts/structure_loss.py --stage evaluate summary --res {res}"),
                     sready, lambda r=res: is_marked(f"struct_eval_{r // 1000}kb")))
    # optional last job: the cohesin-unknown model at 2 kb
    block("all12cohunk_refres_drop", 2000, "mc_all12cohunk_refres_drop_2kb",
          lambda: aux_ok(2000), own_cache=True,
          cache_needs=lambda: aux_ok(2000) and trained("all12deg_refres_drop", 2000))


def launch(job: Job):
    note(f"start {job.name}")
    wrapped = (f"cd {REPO} && ( {job.cmd} ) >> {OUTLOG} 2>&1; rc=$?; "
               f"[ $rc -eq 0 ] && touch {mark(job.name)}; "
               f"({STATUS}) > /dev/null 2>&1; exit $rc")
    return subprocess.Popen(["bash", "-c", wrapped], start_new_session=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-slots", type=int, default=3)
    ap.add_argument("--external", nargs="*", default=[],
                    help="jobs still running from an earlier scheduler (their wrapper "
                         "writes the done marker); not started again")
    ap.add_argument("--wait-pid", type=int, nargs="*", default=[],
                    help="GPU trainings started before the scheduler: no GPU job "
                         "starts while one of these runs")
    a = ap.parse_args()
    for h in MR.glob("HOLD_*"):
        h.unlink()
    note("gpu_scheduler started (replaces the shell drivers)")
    all_jobs = jobs()
    running: dict[str, subprocess.Popen] = {}
    failed: set[str] = set()
    while True:
        for n, p in list(running.items()):
            if p.poll() is not None:
                note(f"{'done' if p.returncode == 0 else 'FAILED'} {n}")
                if p.returncode != 0:
                    failed.add(n)
                del running[n]
        pending = [j for j in all_jobs if j.name not in running and j.name not in failed
                   and j.name not in a.external and not j.done()]
        if not pending and not running:
            note("GPU_SCHEDULER_DONE")
            return
        external = any(_alive(pid) for pid in a.wait_pid)
        gpu_busy = external or any(j.gpu for j in all_jobs if j.name in running)
        if not gpu_busy:
            for j in pending:
                if j.gpu and j.ready():
                    running[j.name] = launch(j)
                    break
        cpu_running = sum(1 for j in all_jobs if j.name in running and not j.gpu)
        if any(j.loaded is not None and not j.loaded() for j in all_jobs if j.name in running):
            cpu_running = a.cpu_slots            # a training is loading: start no CPU job
        for j in pending:
            if cpu_running >= a.cpu_slots:
                break
            if not j.gpu and j.name not in running and j.ready():
                running[j.name] = launch(j)
                cpu_running += 1
        time.sleep(30)


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
