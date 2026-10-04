"""Multi-cell training on every usable ENCODE cell plus Akita's five.

The configuration of scripts/multicell_akita.py (13 ChIP tracks, CTCF, YY1,
ZNF143 and MAZ motif planes, GC/CpG, no DNA encoder, 2 kb, oe_proper targets,
training windows inside Akita's training regions, 100 validation windows per
cell, patience 10, float16 resident windows) on up to twelve training cells:
Akita's GM12878, IMR90, HCT116 (in situ Hi-C), HFF and H1 (Micro-C), and seven
ENCODE cells prepared by scripts/encode_expand.py: HepG2, A549, NHEK, GM23248
(in situ Hi-C), MCF-7, CD14+ monocyte and PC-9 (intact Hi-C). K562, HMEC and
HUVEC are never trained on.

Runs
  all12        the twelve training cells, no assay indicator
  all12assay3  the twelve cells with the three-class assay indicator
               (data.assay_indicator_mode "three": in situ, intact, Micro-C)
  insitu7      the seven in situ Hi-C cells only, no indicator

Stages (each skipped when its output exists)
  precompute  window cache (CPU, python -m hicmultiome.precompute --workers N)
  train       torchrun on --gpus; a run stopped by the max_epochs cap while
              still improving continues from checkpoint-latest (--max-epochs)
  eval        eval_val_windows --split test per training cell (730 disjoint
              Akita test windows of the cell's own target) and crosscell_predict
              --tta on the 413 Akita test regions for every training and test
              target, with every other cell's measured map as a baseline
  rescore     crosscell_predict of the earlier models (ref, gm1, hic3, all5,
              all5assay) on the targets they were not scored on (the deep
              HMEC/HUVEC maps), so the scaling curve uses one set of targets
  ready       exit 0 when all targets and ChIP of the runs exist
  decide      all12 or all12assay3 from all5 vs all5assay (see decide())
  summary     tables and plots in runs/multicell_expanded/: test730.csv,
              crosscell.csv, scaling.csv/.png (unseen-cell score vs training
              cell types, 2 kb), resolution_vs_replicates.csv/.png (model vs
              replicate agreement at 2, 5, 10, 25 kb)

Resolutions (--res): 2 kb as above; 5, 10 and 25 kb runs (<run>_<k>kb) use the
same configuration with data.resolution and the window of WINDOW.

    python scripts/multicell_expanded.py --runs all12 --stage precompute
    python scripts/multicell_expanded.py --runs all12 --stage all --gpus 0,1
    python scripts/multicell_expanded.py --runs all12 --res 5000 10000 25000 --stage all
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
REF_RUN = HM / "runs/motif_grid/full__all13__4motifs"
REF_CFG = HM / "runs/motif_grid/full__all13__4motifs.yaml"
TRAIN_BED = Path.home() / "data/hic-multiome/akita/akita_train_regions.bed"
OLD = HM / "runs/multicell"
OUT = HM / "runs/multicell_expanded"
CACHE = HM / "cache_expanded"
PY = sys.executable
TORCHRUN = str(Path(sys.executable).parent / "torchrun")

ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]

# name: (target O/E cool, ChIP bigwig dir, assay, cell identity for baselines)
# assay: hic = in situ Hi-C, intact = intact Hi-C, microc = Micro-C
CELLS = {
    "gm12878": (OE / "gm12878_proper_2kb.cool", Path.home() / "data/hic-multiome/hg38_gm12878",
                "hic", "gm12878"),
    "imr90": (OE / "imr90_proper_2kb.cool", HM / "hg38_imr90", "hic", "imr90"),
    "hct116": (OE / "hct116_proper_2kb.cool", HM / "hg38_hct116", "hic", "hct116"),
    "hff": (OE / "hff_proper_2kb.cool", HM / "hg38_hff", "microc", "hff"),
    "h1": (OE / "h1_proper_2kb.cool", HM / "hg38_h1", "microc", "h1"),
    "hepg2": (OE / "hepg2_proper_2kb.cool", HM / "hg38_hepg2", "hic", "hepg2"),
    "a549": (OE / "a549_proper_2kb.cool", HM / "hg38_a549", "hic", "a549"),
    "nhek": (OE / "nhek_proper_2kb.cool", HM / "hg38_nhek", "hic", "nhek"),
    "gm23248": (OE / "gm23248_proper_2kb.cool", HM / "hg38_gm23248", "hic", "gm23248"),
    "mcf7": (OE / "mcf7_proper_2kb.cool", HM / "hg38_mcf7", "intact", "mcf7"),
    "cd14mono": (OE / "cd14mono_proper_2kb.cool", HM / "hg38_cd14mono", "intact", "cd14mono"),
    "pc9": (OE / "pc9_proper_2kb.cool", HM / "hg38_pc9", "intact", "pc9"),
    # held-out test cells, never trained on
    "k562": (OE / "k562_proper_2kb.cool", HM / "hg38_k562", "hic", "k562"),
    "hmec": (OE / "hmec_rep1_ice_2kb.cool", HM / "hg38_hmec", "hic", "hmec"),
    "huvec": (OE / "huvec_rep1_ice_2kb.cool", HM / "hg38_huvec", "hic", "huvec"),
    "hmec_deep": (OE / "hmec_deep_proper_2kb.cool", HM / "hg38_hmec", "hic", "hmec"),
    "huvec_deep": (OE / "huvec_deep_proper_2kb.cool", HM / "hg38_huvec", "intact", "huvec"),
}
# (window bins, stride) per resolution. At 2 kb the reference 512-bin window
# (1.02 Mb). Coarser: the largest span divisible by 16 (four U-Net poolings)
# that stays inside oe_proper's 2 Mb distance band, so every target pixel is a
# measured O/E value and no window corner is an unmeasured zero: 384 bins at
# 5 kb (1.92 Mb), 192 at 10 kb (1.92 Mb), 80 at 25 kb (2.0 Mb). Held-out
# regions are merged Akita valid+test stretches (median 5.1 Mb), so ~2 Mb test
# windows still lie wholly inside them. Stride = half a window, as at 2 kb.
WINDOW = {2000: (512, 256), 5000: (384, 192), 10000: (192, 96), 25000: (80, 40)}


def target(cell: str, res: int) -> Path:
    """A cell's proper O/E target at res (2 kb paths as in CELLS)."""
    if res == 2000:
        return CELLS[cell][0]
    k = res // 1000
    if cell in ("hmec", "huvec"):                 # single-lane in situ (replicates_all_cells)
        return OE / f"{cell}_rep1_ice_{k}kb.cool"
    return OE / f"{cell}_proper_{k}kb.cool"


def run_name(base: str, res: int) -> str:
    return base if res == 2000 else f"{base}_{res // 1000}kb"


def split_name(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(.+?)_(5|10|25)kb", name)
    return (m.group(1), int(m.group(2)) * 1000) if m else (name, 2000)


AKITA5 = ["gm12878", "imr90", "hct116", "hff", "h1"]
TRAIN12 = AKITA5 + ["hepg2", "a549", "nhek", "gm23248", "mcf7", "cd14mono", "pc9"]
INSITU7 = ["gm12878", "imr90", "hct116", "hepg2", "a549", "nhek", "gm23248"]
TEST = ["k562", "hmec", "huvec", "hmec_deep", "huvec_deep"]
# baselines of the earlier tables (multicell_akita.BASELINE_CELLS)
BASE6 = ["gm12878", "imr90", "hct116", "hff", "h1", "k562"]
# one measured map per cell identity serves as a baseline (the deep one where both exist)
BASELINE_MAPS = [c for c in CELLS if c not in ("hmec", "huvec")]
RUNS = {  # name: (cells, assay indicator mode or None)
    "all12": (TRAIN12, None),
    "all12assay3": (TRAIN12, "three"),
    "insitu7": (INSITU7, None),
    # all12 plus a reference map (model.reference): every training cell reads a
    # leave-one-out consensus (mean of the other 11 training cells' maps), the
    # held-out cells the consensus of all 12. Outputs in runs/multicell_reference.
    "all12refres": (TRAIN12, None),
    "all12refin": (TRAIN12, None),
    # all12 retrained with honest validation: the no-reference comparison
    "all12honest": (TRAIN12, None),
    # residual model on the LEARNED consensus (scripts/learned_reference.py):
    # a sequence-only model's out-of-fold prediction of consensus12 replaces
    # the measured consensus for every cell
    "all12reflearn": (TRAIN12, None),
    # one resolution-conditioned residual model for 2, 5, 10 and 25 kb
    # (hicmultiome.train_multires, scripts/multires.py); all12multires_<k>kb
    # are its per-resolution views, evaluated like the dedicated runs
    "all12multires": (TRAIN12, None),
    # coarse-to-fine cascade (scripts/cascade.py): the residual reference model
    # plus the coarser level's out-of-fold prediction as a prior plane
    "all12casc": (TRAIN12, None),
    # with track dropout (uniform_count, as configs/full2k_proper.yaml): a
    # missing track means "unknown"; refdrop adds reference dropout 0.3
    "all12refres_drop": (TRAIN12, None),
    "all12honest_drop": (TRAIN12, None),
    "all12refres_refdrop": (TRAIN12, None),
    # the dropout runs on the hard split (scripts/hard_split.py): every cell's
    # hard windows removed from training, selection on hard-validation
    "all12refres_drop_hard": (TRAIN12, None),
    "all12honest_drop_hard": (TRAIN12, None),
    "all12refres_refdrop_hard": (TRAIN12, None),
    # with the HCT116 RAD21-mAC degron (Rao et al. 2017) as training data:
    # hct116 reads the matched untreated degron ChIP (perturb_chip/untreated),
    # hct116_auxin the 6 h auxin Hi-C and ChIP, on odd chromosomes only
    "all12deg_refres_drop": (TRAIN12, None),
    "all12deg_honest_drop": (TRAIN12, None),
    # the degron recipe plus the cohesin-unknown indicator (registry section 25,
    # 1c; data.cohesin_unknown): absent cohesin is "unknown", not "HFF-like"
    "all12cohunk_refres_drop": (TRAIN12, None),
    "all12cohunk_honest_drop": (TRAIN12, None),
    # AML-inclusive training (registry section 36; scripts/aml_train.py): the 12
    # cells plus MOLM-13 and eight Xu 2022 patients (fold 0; _f1, _f2 the other
    # test folds), AML windows oversampled, AML reference map, patient-specific
    # hard validation
    **{f"all12aml_{k}{f}": (TRAIN12, None) for k in ("refin_drop", "honest_drop")
       for f in ("", "_f1", "_f2")},
    # gain beyond the consensus on new cells (registry section 26): input-only
    # reference with track dropout, the same on CTCF + H3K27ac only, and the
    # residual dropout model with between-cell-variance loss weighting
    "all12refin_drop": (TRAIN12, None),
    "all12refin_drop_ctcfac": (TRAIN12, None),
    "all12refres_drop_varw": (TRAIN12, None),
    # HFF with cohesin ChIP (HFFc6 SMC1A from ChIP-Atlas in the RAD21 and SMC3
    # slots; registry section 25): no training cell then lacks cohesin
    "all12hffcoh_refres_drop": (TRAIN12, None),
    # mean DNA methylation (ENCODE WGBS, scripts/methylation.py) as a 14th
    # track, a proxy for marks primary samples lack (registry section 27)
    "all12refres_drop_meth": (TRAIN12, None),
    # the same with the larger cell set: the hg19 RRBS of HCT116, MCF-7 and HFF
    # lifted to hg38 (scripts/methylation.py, registry section 35)
    "all12refres_drop_meth_v2": (TRAIN12, None),
    # copy number (CNVgain, CNVloss from Hi-C coverage, scripts/cnv_sv.py) as
    # two more tracks and the structural-variant loss mask (registry section 28)
    "all12refres_drop_cnv": (TRAIN12, None),
}
# structure-aware learning (registry section 30): all12refin_drop plus one
# structure loss term (insulation = TADs, donut = loops, compart = within-
# window compartment proxy) on a weight grid at 25 kb, the chosen weight per
# term (scripts/structure_loss.py --stage choose) at 10 kb, all three
# together, and checkpoint selection on the combined structure score
STRUCT_TERMS = ["insul", "donut", "compart"]
STRUCT_GRID = [0.1, 0.3, 1.0]
STRUCT_CHOSEN = REPO / "results/structure_loss/chosen_weights.json"


def wtag(w: float) -> str:
    return "w" + f"{w:g}".replace(".", "p")


STRUCT_RUNS = ([f"all12refin_drop_{t}_{wtag(w)}" for t in STRUCT_TERMS for w in STRUCT_GRID]
               + [f"all12refin_drop_{t}" for t in STRUCT_TERMS]
               + ["all12refin_drop_struct3", "all12refin_drop_selcomb",
                  "all12refin_drop_selhm"])
for _r in STRUCT_RUNS:
    RUNS[_r] = (TRAIN12, None)


def struct_loss(base: str) -> str | None:
    """train.loss of a structure run (None: the default l1+mse)."""
    m = re.fullmatch(r"all12refin_drop_(insul|donut|compart)_w([0-9p]+)", base)
    if m:
        return f"l1+mse+{float(m.group(2).replace('p', '.')):g}*{m.group(1)}"
    chosen = (json.loads(STRUCT_CHOSEN.read_text()) if STRUCT_CHOSEN.exists()
              else {t: 0.3 for t in STRUCT_TERMS})      # placeholder until chosen
    m = re.fullmatch(r"all12refin_drop_(insul|donut|compart)", base)
    if m:
        return f"l1+mse+{chosen[m.group(1)]:g}*{m.group(1)}"
    if base == "all12refin_drop_struct3":
        return "l1+mse+" + "+".join(f"{chosen[t]:g}*{t}" for t in STRUCT_TERMS)
    return None
DROPOUT_RUNS = {"all12refres_drop": (0.5, 0.0), "all12honest_drop": (0.5, 0.0),
                "all12refres_refdrop": (0.5, 0.3)}   # (track_dropout, reference_dropout)
DROPOUT_RUNS.update({f"{k}_hard": v for k, v in list(DROPOUT_RUNS.items())})
DROPOUT_RUNS.update({"all12deg_refres_drop": (0.5, 0.0), "all12deg_honest_drop": (0.5, 0.0),
                     "all12cohunk_refres_drop": (0.5, 0.0), "all12cohunk_honest_drop": (0.5, 0.0)})
DROPOUT_RUNS.update({"all12refin_drop": (0.5, 0.0), "all12refin_drop_ctcfac": (0.5, 0.0),
                     "all12refres_drop_varw": (0.5, 0.0)})
DROPOUT_RUNS["all12hffcoh_refres_drop"] = (0.5, 0.0)
DROPOUT_RUNS["all12refres_drop_meth"] = (0.5, 0.0)
DROPOUT_RUNS["all12refres_drop_meth_v2"] = (0.5, 0.0)
DROPOUT_RUNS["all12refres_drop_cnv"] = (0.5, 0.0)
DROPOUT_RUNS.update({r: (0.5, 0.0) for r in STRUCT_RUNS})
# chromosome-held-out final models (registry section 37): the union of the
# chromosomes the competitors hold out (C.Origami chr2 test and chr3
# validation, Epiphany chr3, 11, 17, HiC-Reg chr14 and chr17), absent from the
# data of every cell (data.chroms), so no training window and no normaliser
# statistic sees them; the Akita valid+test regions stay excluded as before
CHRHO_RUNS = ["all12refin_drop_chrho", "all12honest_drop_chrho", "all12refres_drop_chrho"]
for _r in CHRHO_RUNS:
    RUNS[_r] = (TRAIN12, None)
    DROPOUT_RUNS[_r] = (0.5, 0.0)
CHRHO_HELD = ["chr2", "chr3", "chr11", "chr14", "chr17"]
CHRHO_CHROMS = [f"chr{i}" for i in range(1, 23) if f"chr{i}" not in CHRHO_HELD]
AML_RUNS = {k for k in RUNS if k.startswith("all12aml_")}
DROPOUT_RUNS.update({k: (0.5, 0.0) for k in AML_RUNS})
METH_RUNS = {"all12refres_drop_meth", "all12refres_drop_meth_v2"}
CNV_RUNS = {"all12refres_drop_cnv"}
COHUNK_RUNS = {"all12cohunk_refres_drop", "all12cohunk_honest_drop"}
DEGRON_RUNS = {"all12deg_refres_drop", "all12deg_honest_drop"} | COHUNK_RUNS
HFF_COHESIN_DIR = HM / "hg38_hff_cohesin"


def hff_cohesin_dir() -> Path:
    """HFF's ChIP folder plus RAD21 and SMC3 from the HFFc6 SMC1A replicates
    (aml_fetch.py --cells hffcoh), as copies, with missing.json and the
    picks of the SMC1A tracks."""
    src, coh = HM / "hg38_hff", HM / "aml/hg38_hffcoh"
    if not (coh / "RAD21.bigwig").exists() or not (coh / "SMC3.bigwig").exists():
        raise SystemExit(f"{coh}: run scripts/aml_fetch.py --cells hffcoh first")
    d = HFF_COHESIN_DIR
    d.mkdir(exist_ok=True)
    import shutil
    for f in list(src.glob("*.bigwig")) + [coh / "RAD21.bigwig", coh / "SMC3.bigwig"]:
        dst = d / f.name                      # copies: the share refuses symlinks
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(f, tmp)
            tmp.rename(dst)
    have = {p.stem.upper() for p in d.glob("*.bigwig")}
    (d / "missing.json").write_text(json.dumps([t for t in ALL13 if t.upper() not in have]))
    (d / "picks.csv").write_text((coh / "picks.csv").read_text())
    return d
PERTURB_CHIP = Path.home() / "data/hic-multiome/perturb_chip"
DEGRON_TRAIN_CHROMS = [f"chr{i}" for i in range(1, 23, 2)]     # odd autosomes
# the auxin condition's identity (consensus, leakage guard): HCT116
SAMPLE_IDENT = {"hct116_auxin": "hct116"}
HARD_SPLIT_DIR = REPO / "results/hard_split"
STAGE_ROOT = Path.home() / "scratch/hicmx_stage"      # local disk, not /tmp
REF_RUNS = {"all12refres": "residual", "all12refin": "input", "all12reflearn": "residual",
            "all12multires": "residual", "all12casc": "residual",
            "all12refres_drop": "residual", "all12refres_refdrop": "residual",
            "all12refres_drop_hard": "residual", "all12refres_refdrop_hard": "residual",
            "all12deg_refres_drop": "residual", "all12cohunk_refres_drop": "residual",
            "all12refin_drop": "input", "all12refin_drop_ctcfac": "input",
            "all12refres_drop_varw": "residual", "all12hffcoh_refres_drop": "residual",
            "all12refres_drop_meth": "residual",
            "all12refres_drop_meth_v2": "residual", "all12refres_drop_cnv": "residual",
            **{r: "input" for r in STRUCT_RUNS},
            **{k: "input" for k in AML_RUNS if "refin" in k},
            "all12refin_drop_chrho": "input", "all12refres_drop_chrho": "residual"}
CASCADE_FROM = {10000: 25000, 5000: 10000, 2000: 5000}   # fine -> coarser level
MULTIRES = [2000, 5000, 10000, 25000]
# runs trained with train.val_exclude_overlap (honest validation)
HONEST_RUNS = set(REF_RUNS) | {"all12honest", "all12honest_drop", "all12honest_drop_hard",
                               "all12deg_honest_drop", "all12cohunk_honest_drop"} | AML_RUNS | {"all12honest_drop_chrho"}
REF_OUT = HM / "runs/multicell_reference"


def consensus_loo(cell: str, res: int) -> Path:
    """Mean of the proper O/E maps of the 11 training cells other than `cell`
    (scripts/consensus_oe.py; runs/multicell_reference/consensus.sh)."""
    return OE / f"consensus_loo_{cell}_proper_{res // 1000}kb.cool"


def consensus12(res: int) -> Path:
    """Mean of all 12 training cells' maps: the reference for held-out cells."""
    return OE / f"consensus12_proper_{res // 1000}kb.cool"


def is_ref(name: str) -> bool:
    return split_name(name)[0] in REF_RUNS


def out_of(name: str) -> Path:
    """Folder of a run's directory, config and logs."""
    return REF_OUT if split_name(name)[0] in HONEST_RUNS else OUT


def learned_map(res: int) -> Path:
    """Out-of-fold prediction of consensus12 by the sequence-only model
    (scripts/learned_reference.py)."""
    return OE / f"learned_cons12_oof_{res // 1000}kb.cool"


def measured_reference(name: str, cell: str) -> Path:
    """Measured consensus for `cell`: leave-one-out for a training cell,
    consensus12 for a held-out cell."""
    res = split_name(name)[1]
    return consensus_loo(cell, res) if cell in RUNS[split_name(name)[0]][0] else consensus12(res)


def cascade_prior(cell: str, coarse_res: int) -> Path:
    """Out-of-fold prediction of the coarser cascade level for a cell
    (scripts/cascade.py), at the coarse resolution; named by cell identity."""
    return OE / "cascade" / f"prior_{CELLS[cell][3]}_{coarse_res // 1000}kb.cool"


def reference_of(name: str, cell: str) -> Path:
    """The reference map a reference run reads for `cell`."""
    if split_name(name)[0] == "all12reflearn":
        return learned_map(split_name(name)[1])
    return measured_reference(name, SAMPLE_IDENT.get(cell, cell))
# earlier models: name -> (run dir, cells trained on)
OLD_RUNS = {"ref": (REF_RUN, ["gm12878"]), "gm1": (OLD / "gm1", ["gm12878"]),
            "hic3": (OLD / "hic3", ["gm12878", "imr90", "hct116"]),
            "all5": (OLD / "all5", AKITA5), "all5assay": (OLD / "all5assay", AKITA5)}
STOP_RE = re.compile(r"early stop at epoch (\d+) \(best val ([-0-9.]+), (\d+) epochs no improvement\)")


def missing_marks(bw_dir: Path) -> list[str]:
    """Marks without a bigWig. Where encode_expand wrote missing.json, the marks
    it lists are the missing ones and every other must be downloaded already."""
    have = {p.stem.upper() for p in bw_dir.glob("*.bigwig")}
    mj = bw_dir / "missing.json"
    if mj.exists() and (bw_dir / "picks.csv").exists():
        miss = json.loads(mj.read_text())
        absent = [t for t in ALL13 if t not in miss and t.upper() not in have]
        if absent:
            raise SystemExit(f"{bw_dir}: ChIP not complete, still missing {absent}")
        return miss
    return [t for t in ALL13 if t.upper() not in have]


def run_config(name, val_per_cell, patience, max_epochs):
    base, res = split_name(name)
    cells, mode = RUNS[base]
    cfg = copy.deepcopy(yaml.safe_load(open(REF_CFG)))
    cfg["data"]["samples"] = [
        {"name": c, "cool_path": str(target(c, res)), "bigwig_dir": str(CELLS[c][1]),
         "missing_tracks": missing_marks(CELLS[c][1]), "assay": CELLS[c][2]}
        for c in cells]
    if base == "all12hffcoh_refres_drop":
        hff = next(s for s in cfg["data"]["samples"] if s["name"] == "hff")
        d = hff_cohesin_dir()
        hff["bigwig_dir"] = str(d)
        hff["missing_tracks"] = missing_marks(d)
    if base in METH_RUNS:
        import methylation as me
        cfg["data"]["tracks"].append({"name": me.TRACK, "type": "methylation"})
        for smp in cfg["data"]["samples"]:
            v2 = base.endswith("_v2")
            if me.has_meth(smp["name"], v2):
                smp["extra_bigwig_dirs"] = [str(me.meth_dir(smp["name"], v2))]
            else:
                smp["missing_tracks"] = smp["missing_tracks"] + [me.TRACK]
    if base in CNV_RUNS:
        import cnv_sv as cs_
        for t in cs_.TRACKS:
            cfg["data"]["tracks"].append({"name": t, "type": "copynumber"})
        for smp in cfg["data"]["samples"]:        # written by cnv_sv.py --stage cnv
            smp["extra_bigwig_dirs"] = [str(cs_.cnv_dir(smp["name"]))]
        cfg["train"]["sv_mask"] = str(cs_.SV / "breakpoints.json")
    # tracks added on top of the base config's 13 ChIP tracks must also be in
    # its train.select_tracks, which otherwise drops their rows from the model
    # input (all12refres_drop_meth_10kb was first trained that way, 2026-10-02)
    added = [t["name"] for t in cfg["data"]["tracks"] if t["name"] not in ALL13]
    if added and cfg["train"].get("select_tracks") is not None:
        cfg["train"]["select_tracks"] = list(cfg["train"]["select_tracks"]) + [
            t for t in added if t not in cfg["train"]["select_tracks"]]
    if base in DEGRON_RUNS:
        import perturb_validate as pv
        hct = next(s for s in cfg["data"]["samples"] if s["name"] == "hct116")
        hct["bigwig_dir"] = str(PERTURB_CHIP / "untreated")
        hct["missing_tracks"] = missing_marks(PERTURB_CHIP / "untreated")
        cfg["data"]["samples"].append({
            "name": "hct116_auxin", "cool_path": str(pv.oe_path("rad21", "treated", res)),
            "bigwig_dir": str(PERTURB_CHIP / "auxin6h"),
            "missing_tracks": missing_marks(PERTURB_CHIP / "auxin6h"), "assay": "hic",
            "train_chroms": list(DEGRON_TRAIN_CHROMS)})
        cells = cells + ["hct116_auxin"]
    if base in COHUNK_RUNS:
        cfg["data"]["cohesin_unknown"] = True
    aml_set = set()
    if base in AML_RUNS:
        import aml_train as at
        fold = at.fold_of(base)
        cfg["data"]["samples"] += at.aml_samples(fold, res, with_ref=base in REF_RUNS)
        aml_set = set(at.aml_names(fold))
        cells = cells + at.aml_names(fold)
    if base in CHRHO_RUNS:
        cfg["data"]["chroms"] = list(CHRHO_CHROMS)
    cfg["data"]["resolution"] = res
    cfg["window"]["size"], cfg["window"]["stride"] = WINDOW[res]
    cfg["data"]["train_regions_bed"] = str(TRAIN_BED)
    cfg["data"]["assay_indicator"] = mode is not None
    if mode is not None:
        cfg["data"]["assay_indicator_mode"] = mode
    if base in REF_RUNS:
        for smp in cfg["data"]["samples"]:
            if smp["name"] not in aml_set:       # AML samples read the AML reference
                smp["reference_cool"] = str(reference_of(name, smp["name"]))
        cfg.setdefault("model", {})["reference"] = REF_RUNS[base]
        if base == "all12casc":
            for smp in cfg["data"]["samples"]:
                smp["prior_cool"] = str(cascade_prior(smp["name"], CASCADE_FROM[res]))
            cfg["model"]["prior"] = True
        if base == "all12multires":
            cfg["model"]["resolutions"] = list(MULTIRES)
            cfg["model"]["res_embed_dim"] = 16
            cfg["train"]["compile"] = False      # four window shapes
    cache_name = (f"mc_all12aml_{'ref_' if base in REF_RUNS else ''}{res // 1000}kb"
                  f"{'_f' + str(fold) if fold else ''}" if base in AML_RUNS
                  else f"mc_all12ref_chrho_{res // 1000}kb" if base in CHRHO_RUNS[::2] + ["all12refres_drop_chrho"]
                  else f"mc_all12chrho_{res // 1000}kb" if base == "all12honest_drop_chrho"
                  else f"mc_{base}_{res // 1000}kb" if base in DEGRON_RUNS | METH_RUNS | CNV_RUNS
                  | {"all12hffcoh_refres_drop"}
                  else f"mc_all12reflearn_{res // 1000}kb" if base == "all12reflearn"
                  else f"mc_all12casc_{res // 1000}kb" if base == "all12casc"
                  else f"mc_{run_name('all12', res)}" if base in ("all12honest_drop",
                                                                   "all12honest_drop_hard")
                  else f"mc_all12ref_{res // 1000}kb" if base in REF_RUNS
                  else f"mc_{run_name('all12', res)}" if base == "all12honest"   # same windows
                  else f"mc_{name}")
    cfg["train"].update({
        "output_dir": str(out_of(name) / name), "cache_dir": str(CACHE / cache_name),
        "val_regions": val_per_cell * len(cells), "patience_epochs": patience,
        "max_epochs": max_epochs, "resident_dtype": "float16", "resume": False,
        "save_checkpoints": True, "track_dropout": 0.0,
        # honest validation (no training window at a validation position, in
        # any cell) for the runs made after 2026-09-28; the earlier runs as run
        "val_exclude_overlap": base in HONEST_RUNS,
    })
    if base in DROPOUT_RUNS:
        td, rd = DROPOUT_RUNS[base]
        cfg["train"].update({"track_dropout": td, "track_dropout_mode": "uniform_count",
                             "reference_dropout": rd})
    if res == 2000 and base in HONEST_RUNS:
        # 2 kb with a reference or prior plane does not fit resident in RAM on
        # both DDP ranks (earlyoom killed all12refres and all12casc on
        # 2026-09-30): read the train split from memory maps on local disk
        cfg["train"]["stage_dir"] = str(STAGE_ROOT)
    if base == "all12refin_drop_ctcfac":
        cfg["train"]["select_tracks"] = ["CTCF", "H3K27ac"]
    if base == "all12refres_drop_varw":
        cfg["train"].update({"loss_weighting": "between_cell_var", "loss_weight_alpha": 1.0})
    if base in STRUCT_RUNS:
        cfg["train"]["log_structure"] = True
        if struct_loss(base):
            cfg["train"]["loss"] = struct_loss(base)
        if base == "all12refin_drop_selcomb":
            cfg["train"]["select_metric"] = "combined"
        if base == "all12refin_drop_selhm":
            cfg["train"]["select_metric"] = "hic_measures"
    if base in AML_RUNS:
        cfg["train"].update({"oversample_samples": at.aml_names(fold),
                             "oversample_factor": at.OVERSAMPLE,
                             "hard_split": str(at.hard_split_path(fold, res))})
    if base.endswith("_hard"):
        cfg["train"]["hard_split"] = str(HARD_SPLIT_DIR / f"{res // 1000}kb" / "split.json")
    return cfg


def logtext(name):
    return "".join(f.read_text(errors="replace") for f in
                   (out_of(name) / f"{name}.log", out_of(name) / f"{name}.cont.log")
                   if f.exists())


def done(log: Path) -> bool:
    t = log.read_text(errors="replace") if log.exists() else ""
    return "early stop at epoch" in t or "done." in t


def stop_info(name, patience):
    m = STOP_RE.findall(logtext(name))
    if not m:
        return None
    ep, best, noimp = m[-1]
    return int(ep), float(best), "patience" if int(noimp) >= patience else "cap"


def sh(cmd, log: Path, env=None):
    print("[cmd]", " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as fh:
        return subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=REPO).returncode


def best_ckpt(run: Path) -> Path:
    return run / (run / "best").read_text().strip()


def crosscell(cfg_path, ckpt, out: Path, gpu: str, cells, res: int = 2000, ref_run=None):
    """ref_run: name of a reference-map run; each cell then gets its reference
    (reference_of) as model input and as an extra measured-map column
    (base_consensus_loo for training cells, base_consensus12 for held-out)."""
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    for cell in cells:
        _, bw, assay, ident = CELLS[cell]
        cool = target(cell, res)
        if not cool.exists() or not bw.exists():
            print(f"[skip] {out}/{cell}: target or ChIP missing", flush=True)
            continue
        if (out / f"predict_{cell}.csv").exists():
            continue
        others = [f"{o}={target(o, res)}" for o in BASELINE_MAPS
                  if CELLS[o][3] != ident and target(o, res).exists()]
        extra = []
        if ref_run is not None:
            rf = reference_of(ref_run, cell)
            meas = measured_reference(ref_run, cell)
            tag = "consensus_loo" if cell in RUNS[split_name(ref_run)[0]][0] else "consensus12"
            others.append(f"{tag}={meas}")
            if rf != meas:
                others.append(f"learned={rf}")
            extra = ["--reference", rf]
            if split_name(ref_run)[0] == "all12casc":
                extra += ["--prior", cascade_prior(cell, CASCADE_FROM[res])]
        if ref_run is not None and split_name(ref_run)[0] in METH_RUNS:
            import methylation as me
            v2 = split_name(ref_run)[0].endswith("_v2")
            if me.has_meth(cell, v2):
                extra += ["--extra-bigwig-dir", me.meth_dir(cell, v2)]
        if ref_run is not None and split_name(ref_run)[0] in CNV_RUNS:
            import cnv_sv as cs_
            if cs_.has_cnv(cell):
                extra += ["--extra-bigwig-dir", cs_.cnv_dir(cell)]
        rc = sh([PY, "scripts/crosscell_predict.py", "--config", cfg_path,
                 "--checkpoint", ckpt, "--cell", cell, "--bigwig-dir", bw,
                 "--target", cool, "--tta", "--assay", assay, "--out-dir", out,
                 "--binsize", res, *extra, "--others", *others],
                out / f"predict_{cell}.log", env)
        if rc:
            print(f"[FAILED] crosscell {out}/{cell} rc={rc}", flush=True)


def train(name, a, gpus):
    cfg = run_config(name, a.val_per_cell, a.patience, a.max_epochs)
    O = out_of(name)
    cp = O / f"{name}.yaml"
    log = O / f"{name}.log"
    cache = Path(cfg["train"]["cache_dir"])
    if not (cache / "manifest.json").exists():
        raise SystemExit(f"{name}: no cache; run --stage precompute first")
    if not done(log):
        yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        log.write_text("")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
        if cfg["train"].get("stage_dir"):
            # memory-mapped training windows on local disk, written once
            if sh([PY, "-m", "hicmultiome.train", "--config", cp, "--stage-only"],
                  O / f"{name}.stage.log"):
                raise SystemExit(f"staging failed for {name}")
        sh([TORCHRUN, "--standalone", f"--nproc_per_node={len(gpus)}",
            "-m", "hicmultiome.train", "--config", cp], log, env)
        if not done(log):
            raise SystemExit(f"training failed for {name}; see {log}")
    # never let the epoch cap stop a run that is still improving
    while (si := stop_info(name, a.patience)) and si[2] == "cap" and si[0] < a.max_epochs:
        ccfg = yaml.safe_load(open(cp))
        ccfg["train"].update({"resume": True,
                              "resume_from": str(O / name / "checkpoint-latest"),
                              "max_epochs": a.max_epochs, "patience_epochs": a.patience})
        pj = O / name / "checkpoint-latest" / "progress.json"
        pr = json.loads(pj.read_text())
        if si[1] > pr["best"]:
            pr["best"] = si[1]
            pj.write_text(json.dumps(pr))
        ccp = O / f"{name}.cont.yaml"
        yaml.safe_dump(ccfg, open(ccp, "w"), sort_keys=False)
        print(f"[continue] {name} from epoch {si[0]} (cap) to max {a.max_epochs}", flush=True)
        n0 = len(STOP_RE.findall(logtext(name)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
        sh([TORCHRUN, "--standalone", f"--nproc_per_node={len(gpus)}",
            "-m", "hicmultiome.train", "--config", ccp], O / f"{name}.cont.log", env)
        if len(STOP_RE.findall(logtext(name))) <= n0:
            raise SystemExit(f"continuation failed for {name}")


def evaluate(name, gpu, cells):
    run = out_of(name) / name
    base, res = split_name(name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    for cell in RUNS[base][0]:
        if cells and cell not in cells:
            continue
        if not (run / f"test_eval_{cell}.json").exists():
            sh([PY, "scripts/eval_val_windows.py", "--run", run, "--split", "test",
                "--sample", cell], out_of(name) / f"{name}.eval.log", env)
        # the cell's leave-one-out reference alone, on the same windows
        if is_ref(name) and not (run / f"refalone_test_eval_{cell}.json").exists():
            sh([PY, "scripts/eval_val_windows.py", "--run", run, "--split", "test",
                "--sample", cell, "--reference-only",
                "--out", run / f"refalone_test_eval_{cell}.json"],
               out_of(name) / f"{name}.eval.log", env)
    ck = best_ckpt(run)
    crosscell(ck / "config.yaml", ck, run / "crosscell", gpu,
              [c for c in RUNS[base][0] + TEST if not cells or c in cells], res,
              ref_run=name if is_ref(name) else None)


def rescore(gpu, cells):
    for name, (rdir, _) in OLD_RUNS.items():
        if not (rdir / "best").exists():
            print(f"[skip] rescore {name}: no best checkpoint", flush=True)
            continue
        ck = best_ckpt(rdir)
        old_cc = (OLD / "ref" / "crosscell") if name == "ref" else (rdir / "crosscell")
        # the deep targets, plus any earlier target this model was never scored on
        want = ["hmec_deep", "huvec_deep"] + [
            c for c in ("k562", "hmec", "huvec") if not (old_cc / f"predict_{c}.csv").exists()]
        crosscell(ck / "config.yaml", ck, OUT / "rescore" / name, gpu,
                  [c for c in want if not cells or c in cells])


def _write(fn, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    tmp = OUT / (fn + ".part")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.rename(OUT / fn)
    print(f"wrote {OUT / fn}")


ALL_RUNS = [run_name(b, r) for r in WINDOW for b in RUNS if b not in HONEST_RUNS]


def summarize():
    import pandas as pd

    t_rows, c_rows, w_rows, s_rows = [], [], [], []
    # crosscell tables of every model: new runs, rescored and earlier ones
    sources = []
    for name in ALL_RUNS:
        base, res = split_name(name)
        sources.append((name, res, OUT / name / "crosscell", RUNS[base][0]))
    for name, (rdir, cells) in OLD_RUNS.items():
        sources.append((name, 2000, OUT / "rescore" / name, cells))
        old_cc = (OLD / "ref" / "crosscell") if name == "ref" else (rdir / "crosscell")
        sources.append((name, 2000, old_cc, cells))
    seen = set()
    for name, res, d, trained in sources:
        for f in sorted(d.glob("predict_*.csv")):
            cell = f.stem.replace("predict_", "")
            if (name, cell) in seen or cell not in CELLS:
                continue
            seen.add((name, cell))
            df = pd.read_csv(f)
            base = {c.replace("base_", ""): float(df[c].median()) for c in df.columns
                    if c.startswith("base_")}
            b6 = {k: v for k, v in base.items() if k in BASE6}
            best = max(base, key=base.get) if base else ""
            best6 = max(b6, key=b6.get) if b6 else ""
            c_rows.append({
                "run": name, "res_kb": res // 1000, "n_train_cells": len(trained),
                "cell": cell, "trained_on_cell": cell in trained,
                "n": int(df["model"].notna().sum()),
                "model": round(float(df["model"].median()), 4),
                "best_baseline6_cell": best6,
                "best_baseline6": round(b6[best6], 4) if best6 else None,
                "best_baseline_all_cell": best,
                "best_baseline_all": round(base[best], 4) if best else None,
                **{f"base_{k}": round(v, 4) for k, v in sorted(base.items())}})
    for name in ALL_RUNS:
        run = OUT / name
        for j in sorted(run.glob("test_eval_*.json")):
            r = json.loads(j.read_text())
            t_rows.append({"run": name, "res_kb": split_name(name)[1] // 1000,
                           "cell": r["sample"], "assay": CELLS[r["sample"]][2],
                           "n": r["n"], "oe_median_r": round(r["oe_median_r"], 4)})
        man = CACHE / f"mc_{name}" / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            per = {}
            for s in m["shards"]:
                d = per.setdefault(s["sample"], {"train": 0, "test": 0})
                d[s["split"]] += s["n_windows"]
            for smp, d in per.items():
                w_rows.append({"run": name, "cell": smp, "train_windows": d["train"],
                               "test_windows": d["test"]})
            w_rows.append({"run": name, "cell": "ALL",
                           "train_windows": sum(d["train"] for d in per.values()),
                           "test_windows": sum(d["test"] for d in per.values()),
                           "removed_by_train_regions": m.get("n_outside_train_regions")})
        si = stop_info(name, 10)
        if si:
            hist = "; ".join(f"epoch {e} {'patience' if int(n) >= 10 else 'cap'}"
                             for e, _, n in STOP_RE.findall(logtext(name)))
            s_rows.append({"run": name, "stopped_at_epoch": si[0], "best_val": si[1],
                           "stopped_by": si[2], "all_stops": hist})
    _write("test730.csv", t_rows)
    _write("crosscell.csv", c_rows)
    _write("train_windows.csv", w_rows)
    _write("training_stop.csv", s_rows)
    scaling([r for r in c_rows if r["res_kb"] == 2])
    resolution_table(c_rows, t_rows)


def scaling(c_rows):
    """Unseen-cell score against the number of training cell types (2 kb)."""
    order = ["ref", "gm1", "hic3", "all5", "all5assay", "insitu7", "all12", "all12assay3"]
    unseen = ["k562", "hmec", "huvec", "hmec_deep", "huvec_deep"]
    by = {(r["run"], r["cell"]): r for r in c_rows}
    rows = []
    for run in order:
        if not any((run, c) in by for c in unseen):
            continue
        n = next(r["n_train_cells"] for r in c_rows if r["run"] == run)
        rec = {"run": run, "n_train_cells": n}
        for c in unseen:
            if (run, c) in by:
                rec[c] = by[(run, c)]["model"]
        rows.append(rec)
    for c in unseen:                           # measured-map baselines (model-free)
        r = next((by[k] for k in by if k[1] == c), None)
        if r:
            for tag in ("best_baseline6", "best_baseline_all"):
                rows.append({"run": f"{tag}:{c}", c: r[tag],
                             "n_train_cells": r[f"{tag}_cell"]})
    _write("scaling.csv", rows)
    plt = _plt()
    if plt is None:
        return
    runs = [r for r in rows if not r["run"].startswith("best_")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (cell, deep) in zip(axes, (("k562", None), ("hmec", "hmec_deep"),
                                       ("huvec", "huvec_deep"))):
        for key, style, lab in ((cell, "o-", "single-lane target" if deep else "target"),
                                (deep, "s--", "deep target")):
            if key is None:
                continue
            pts = sorted((r["n_train_cells"], r[key], r["run"]) for r in runs
                         if key in r and r["run"] not in ("gm1", "all5assay", "all12assay3"))
            if not pts:
                continue
            line, = ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=lab)
            for x, y, name in pts:
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, -10),
                            fontsize=7)
            b = next((r for r in rows if r["run"] == f"best_baseline_all:{key}"), None)
            if b:
                ax.axhline(b[key], color=line.get_color(), lw=0.8, ls=":",
                           label=f"best measured map ({b['n_train_cells']})")
        ax.set_title(cell.upper())
        ax.set_xlabel("training cell types")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 3, 5, 7, 12])
        ax.set_xticklabels(["1", "3", "5", "7", "12"])
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("median per-region Pearson (O/E), 413 regions")
    _save(fig, "scaling.png")


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:                          # noqa: BLE001
        return None


def _save(fig, fn):
    tmp = OUT / (fn + ".part.png")
    fig.tight_layout()
    fig.savefig(tmp, dpi=150)
    tmp.rename(OUT / fn)
    print(f"wrote {OUT / fn}")


def resolution_table(c_rows, t_rows, base_run="all12"):
    """Model vs replicate agreement per resolution, on the 413 Akita regions.

    model_413: crosscell_predict median (training cells: own-ChIP prediction of
    a cell trained on; K562/HMEC/HUVEC: unseen). model_test_windows: the
    disjoint test windows (training cells). rep_insitu: single-lane in situ
    rep1 vs rep2 (results/replicates/replicate_corr.csv); rep_intact_deep: the
    deep intact pairs (replicate_corr_intact_deep.csv), mean over pairs of a cell."""
    import pandas as pd

    rep = REPO / "results/replicates/replicate_corr.csv"
    deep = REPO / "results/replicates/replicate_corr_intact_deep.csv"
    rep = pd.read_csv(rep).set_index("cell") if rep.exists() else None
    deep = pd.read_csv(deep) if deep.exists() else None
    rows = []
    for res in WINDOW:
        k = res // 1000
        name = run_name(base_run, res)
        for cell in TRAIN12 + TEST:
            r = next((x for x in c_rows if x["run"] == name and x["cell"] == cell), None)
            t = next((x for x in t_rows if x["run"] == name and x["cell"] == cell), None)
            ident = CELLS[cell][3]
            rec = {"res_kb": k, "run": name, "cell": cell,
                   "trained_on_cell": cell in RUNS[base_run][0],
                   "model_413": r["model"] if r else None,
                   "best_measured_map_413": r["best_baseline_all"] if r else None,
                   "model_test_windows": t["oe_median_r"] if t else None}
            if rep is not None and ident in rep.index and cell not in ("hmec_deep", "huvec_deep"):
                rec["rep_insitu_single_lane"] = rep.loc[ident, f"{k}kb"]
            if deep is not None and ident in set(deep["cell"]):
                v = deep.loc[deep["cell"] == ident, f"{k}kb"]
                rec["rep_intact_deep"] = round(float(v.mean()), 3)
                rec["rep_intact_deep_pairs"] = int(v.notna().sum())
            if any(rec.get(x) is not None for x in ("model_413", "model_test_windows")):
                rows.append(rec)
    _write("resolution_vs_replicates.csv", rows)
    plt = _plt()
    if plt is None or not rows:
        return
    df = pd.DataFrame(rows)
    cells = ["gm12878", "imr90", "k562", "hmec", "huvec"]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 3.6), sharey=True)
    for ax, cell in zip(axes, cells):
        d = df[df["cell"] == cell].sort_values("res_kb")
        dd = df[df["cell"] == f"{cell}_deep"].sort_values("res_kb")
        tag = "trained on" if cell in RUNS[base_run][0] else "unseen"
        if len(d):
            ax.plot(d["res_kb"], d["model_413"], "o-", label=f"model ({tag})")
            if "rep_insitu_single_lane" in d:
                ax.plot(d["res_kb"], d["rep_insitu_single_lane"], "^:",
                        label="replicates, single-lane in situ")
            if "rep_intact_deep" in d and d["rep_intact_deep"].notna().any():
                ax.plot(d["res_kb"], d["rep_intact_deep"], "v:",
                        label="replicates, deep intact")
            ax.plot(d["res_kb"], d["best_measured_map_413"], "x--", lw=0.8,
                    label="best other cell's map")
        if len(dd):
            ax.plot(dd["res_kb"], dd["model_413"], "s-", label="model, deep target")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 25])
        ax.set_xticklabels(["2", "5", "10", "25"])
        ax.set_xlabel("resolution (kb)")
        ax.set_title(cell.upper())
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("median per-region Pearson (O/E), 413 regions")
    _save(fig, "resolution_vs_replicates.png")


def _write_to(folder: Path, fn, rows):
    global OUT
    keep = OUT
    try:
        OUT = folder
        _write(fn, rows)
    finally:
        OUT = keep


def reference_summary():
    """Per cell and resolution, 413 Akita test regions and the disjoint test
    windows: each reference run, its reference alone (leave-one-out consensus
    for training cells, consensus12 for held-out cells), the best measured map
    of another cell, and the no-reference all12 run. Written to
    runs/multicell_reference/reference_compare.csv, reference_summary.csv and
    training_stop.csv."""
    import pandas as pd

    rows, stops = [], []
    for res in (25000, 10000, 5000, 2000):
        base_name = run_name("all12", res)          # overlapping validation (section 13)
        honest_name = run_name("all12honest", res)  # the same, honest validation
        for rb in list(REF_RUNS) + ["all12honest", "all12honest_drop", "all12honest_drop_hard",
                                    "all12deg_honest_drop", "all12cohunk_honest_drop",
                                    "all12aml_honest_drop"]:
            refres_name = run_name("all12refres", res)
            name = run_name(rb, res)
            run = REF_OUT / name
            if not run.exists():
                continue
            si = stop_info(name, 10)
            if si:
                hist = "; ".join(f"epoch {e} {'patience' if int(n) >= 10 else 'cap'}"
                                 for e, _, n in STOP_RE.findall(logtext(name)))
                stops.append({"run": name, "stopped_at_epoch": si[0], "best_val": si[1],
                              "stopped_by": si[2], "all_stops": hist})
            for cell in TRAIN12 + TEST:
                f = run / "crosscell" / f"predict_{cell}.csv"
                rec = {"res_kb": res // 1000, "run": name, "cell": cell,
                       "trained_on_cell": cell in TRAIN12}
                if f.exists():
                    df = pd.read_csv(f)
                    base = {c[5:]: float(df[c].median()) for c in df if c.startswith("base_")}
                    cons = next((k for k in base if k.startswith("consensus")), None)
                    ref_col = "learned" if "learned" in base else cons
                    single = {k: v for k, v in base.items()
                              if not k.startswith("consensus") and k != "learned"}
                    b = max(single, key=single.get) if single else None
                    rec.update({"n": int(df["model"].notna().sum()),
                                "model": round(float(df["model"].median()), 4),
                                "reference_alone": round(base[ref_col], 4) if ref_col else None,
                                "reference": ref_col,
                                "measured_consensus_alone": (round(base[cons], 4)
                                                             if cons else None),
                                "best_other_map": round(single[b], 4) if b else None,
                                "best_other_map_cell": b})
                    if cons:   # paired per region, against the measured consensus
                        d = df.dropna(subset=["model", f"base_{cons}"])
                        rec["frac_regions_model_above_measured_consensus"] = round(
                            float((d["model"] > d[f"base_{cons}"]).mean()), 3)
                for tag, d in (("all12refres_measured_reference", REF_OUT / refres_name),
                               ("all12honest_no_reference", REF_OUT / honest_name),
                               ("all12_overlapval_no_reference", OUT / base_name)):
                    fb = d / "crosscell" / f"predict_{cell}.csv"
                    if fb.exists() and d != run:
                        rec[tag] = round(float(pd.read_csv(fb)["model"].median()), 4)
                for tag, j in (("test_windows", run / f"test_eval_{cell}.json"),
                               ("test_windows_reference_alone",
                                run / f"refalone_test_eval_{cell}.json"),
                               ("test_windows_all12honest_no_reference",
                                REF_OUT / honest_name / f"test_eval_{cell}.json"),
                               ("test_windows_all12_overlapval_no_reference",
                                OUT / base_name / f"test_eval_{cell}.json")):
                    if tag.startswith("test_windows_all12honest") and name == honest_name:
                        continue
                    if j.exists():
                        rec[tag] = round(json.loads(j.read_text())["oe_median_r"], 4)
                if len(rec) > 4:
                    rows.append(rec)
    _write_to(REF_OUT, "reference_compare.csv", rows)
    _write_to(REF_OUT, "training_stop.csv", stops)
    if not rows:
        return
    df = pd.DataFrame(rows)
    summ = []
    for (k, run), g in df.groupby(["res_kb", "run"], sort=False):
        for label, sub in (("12 training cells", g[g["trained_on_cell"]]),
                           ("held-out cells", g[~g["trained_on_cell"]])):
            rec = {"res_kb": k, "run": run, "cells": label, "n_cells": len(sub)}
            for c in ("model", "reference_alone", "measured_consensus_alone",
                      "best_other_map", "all12refres_measured_reference",
                      "all12honest_no_reference", "all12_overlapval_no_reference",
                      "test_windows", "test_windows_reference_alone",
                      "test_windows_all12honest_no_reference",
                      "test_windows_all12_overlapval_no_reference"):
                if c in sub and sub[c].notna().any():
                    rec[f"median_{c}"] = round(float(sub[c].median()), 4)
            for c, t in (("reference_alone", "beats_reference"),
                         ("measured_consensus_alone", "beats_measured_consensus"),
                         ("best_other_map", "beats_best_other_map"),
                         ("all12honest_no_reference", "beats_all12honest_no_reference")):
                if c in sub:
                    s2 = sub.dropna(subset=["model", c])
                    rec[t] = f"{int((s2['model'] > s2[c]).sum())}/{len(s2)}"
            summ.append(rec)
    _write_to(REF_OUT, "reference_summary.csv", summ)


def decide():
    """all12 or all12assay3 from the five-cell pair all5 / all5assay: the
    indicator is used when it raises the mean unseen-cell score (K562, HMEC,
    HUVEC, 413 regions) by >= 0.005 without lowering the mean test-window score
    of the training cells by more than 0.005. Written to assay_decision.json."""
    import pandas as pd

    def unseen(run):
        v = {}
        for c in ("k562", "hmec", "huvec"):
            for d in (OLD / run / "crosscell", OUT / "rescore" / run):
                f = d / f"predict_{c}.csv"
                if f.exists():
                    v[c] = float(pd.read_csv(f)["model"].median())
                    break
        return v

    def test730(run):
        return {json.loads(j.read_text())["sample"]: json.loads(j.read_text())["oe_median_r"]
                for j in (OLD / run).glob("test_eval_*.json")}

    u0, u1 = unseen("all5"), unseen("all5assay")
    t0, t1 = test730("all5"), test730("all5assay")
    common_u = sorted(set(u0) & set(u1))
    common_t = sorted(set(t0) & set(t1))
    if len(common_u) < 3:
        print(f"[decide] all5assay unseen-cell scores incomplete: {u1}", flush=True)
        return None
    du = sum(u1[c] - u0[c] for c in common_u) / len(common_u)
    dt = (sum(t1[c] - t0[c] for c in common_t) / len(common_t)) if common_t else 0.0
    use = du >= 0.005 and dt >= -0.005
    out = {"all5_unseen": u0, "all5assay_unseen": u1, "mean_unseen_delta": round(du, 4),
           "all5_test730": t0, "all5assay_test730": t1, "mean_test730_delta": round(dt, 4),
           "use_indicator": use, "run": "all12assay3" if use else "all12"}
    (OUT / "assay_decision.json").write_text(json.dumps(out, indent=1))
    print(f"[decide] {out}", flush=True)
    return out["run"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["all12"], choices=list(RUNS))
    ap.add_argument("--res", nargs="*", type=int, default=[2000], choices=list(WINDOW),
                    help="resolutions; a run at res != 2000 is named <run>_<k>kb")
    ap.add_argument("--stage", choices=["precompute", "train", "eval", "rescore", "all",
                                        "summary", "refsummary", "decide", "ready"],
                    default="all")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers", type=int, default=8, help="precompute processes")
    ap.add_argument("--val-per-cell", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="eval/rescore: only these cells (split the work across GPUs)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REF_OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.stage == "summary":
        summarize()
        return
    if a.stage == "refsummary":
        reference_summary()
        return
    if a.stage == "decide":
        r = decide()
        print(f"DECISION {r}", flush=True)
        return
    gpus = [g for g in a.gpus.split(",") if g]
    if a.stage == "rescore":
        rescore(gpus[0], a.cells)
        print("MULTICELL_EXPANDED_DONE", flush=True)
        return
    names = [run_name(r, res) for res in a.res for r in a.runs]
    if a.stage == "ready":                      # exit 0 when every input exists
        for name in names:
            base, res = split_name(name)
            try:
                missing = [c for c in RUNS[base][0] if not target(c, res).exists()]
                if base in REF_RUNS:
                    missing += [str(reference_of(name, c)) for c in RUNS[base][0] + TEST
                                if not reference_of(name, c).exists()]
                for c in RUNS[base][0]:
                    missing_marks(CELLS[c][1])
            except SystemExit as e:
                missing = [str(e)]
            if missing:
                print(f"NOT READY {name}: {missing}", flush=True)
                raise SystemExit(1)
        print("READY", flush=True)
        return
    for name in names:
        # queue control: a run waits while <out>/HOLD_<name> exists
        hold = out_of(name) / f"HOLD_{name}"
        if hold.exists() and a.stage in ("precompute", "train", "all"):
            print(f"[hold] {name}: waiting for {hold} to be removed", flush=True)
            while hold.exists():
                import time
                time.sleep(60)
        cfg = run_config(name, a.val_per_cell, a.patience, a.max_epochs)
        cp = out_of(name) / f"{name}.yaml"
        if not done(out_of(name) / f"{name}.log"):  # a finished run keeps its config
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
        cache = Path(cfg["train"]["cache_dir"])
        if a.stage in ("precompute", "all") and not (cache / "manifest.json").exists():
            missing = [s["cool_path"] for s in cfg["data"]["samples"]
                       if not Path(s["cool_path"]).exists()]
            if missing:
                raise SystemExit(f"{name}: targets not ready: {missing}")
            print(f"[precompute] {name}", flush=True)
            if sh([PY, "-m", "hicmultiome.precompute", "--config", cp,
                   "--workers", a.workers], out_of(name) / f"{name}.precompute.log"):
                raise SystemExit(f"precompute failed for {name}")
        if a.stage in ("train", "all"):
            train(name, a, gpus)
        if a.stage in ("eval", "all"):
            evaluate(name, gpus[0], a.cells)
        print(f"[done] {name} ({a.stage})", flush=True)
    print("MULTICELL_EXPANDED_DONE", flush=True)


if __name__ == "__main__":
    main()
