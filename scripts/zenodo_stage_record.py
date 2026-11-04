"""Assemble the single Zenodo record of the FlexCible manuscript (DOI 10.5281/zenodo.23922797).

Writes a flat folder (default /mnt/processed/wolffjoa/hic-multiome/zenodo_record_v1) with one zip per
topic and the documents, as Zenodo stores files without folders:

  flexcible_released_model.zip   the released model (one network for 2 to 25 kb, any subset of marks)
                                 and its chromosome-held-out twin, with a prediction configuration per
                                 bin size, the normalisers and the depletion operators
  benchmark_models.zip           the dedicated per-bin-size models (consensus input, consensus residual,
                                 marks alone; chromosome-held-out versions and seeds), the two Akita-grid
                                 models and the AML fold models and seed runs
  consensus12_maps.zip           the 12-cell consensus at 25, 10, 5 and 2 kb (the reference for new cells)
  loo_maps_25_10_5kb.zip         the 12 leave-one-out consensus maps at 25, 10 and 5 kb
  loo_maps_2kb.zip               the 12 leave-one-out consensus maps at 2 kb
  other_maps.zip                 the four-cell consensus on Akita's grid, the AML fold and HSPC
                                 references, the mouse leave-one-out maps
  tables.zip                     split files, AML object tables, every result table, the generated
                                 supplementary tables, numbers_provenance.csv, DATA_DICTIONARY.md
  README.md, LICENSE.txt, NOTICE_DATA.md, SOURCES.csv, manifest.csv (sha256), metadata.json

Nothing under the source trees is modified. Checkpoints hold the weights, the configuration and the
normalisers, never the optimiser state. In the released model the four bin-size views share the weights
of the shared checkpoint (multires_source.json); the prediction configurations refer to the genome, the
motifs, a cache folder and the consensus maps through ${FLEXCIBLE_GENOME}, ${FLEXCIBLE_MOTIFS},
${FLEXCIBLE_CACHE} and ${FLEXCIBLE_MAPS}, which flexcible expands from the environment. Training
configurations of every model are included with the local paths replaced by ${HICMULTI_HM} and
${HICMULTI_DATA}. The 2 kb maps and the Akita-grid map are the band-cut copies of zenodo_trim_cool.py.

    python scripts/zenodo_stage_record.py [--out DIR] [--dry-run] [--replace]
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import io
import json
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import yaml

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
DATA = Path.home() / "data/hic-multiome"
RUNS = HM / "runs/multicell_reference"
OE = HM / "crosscell_oe"
TRIM = HM / "zenodo_maps_trimmed"
MOUSE_OE = HM / "validation_mouse/oe"
OLD = HM / "zenodo_staging"
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
MANUSCRIPT = Path.home() / "src/manuscripts/hic-multiome-natcomms"

CELLS = ["a549", "cd14mono", "gm12878", "gm23248", "h1", "hct116", "hepg2", "hff", "imr90", "mcf7", "nhek", "pc9"]
BINS = ["25", "10", "5", "2"]
DOI = "10.5281/zenodo.23922797"
REPO_URL = "https://github.com/MHH-Bioinformatics-Hematology/FlexCible"
ZENODO_GB = 50.0
ZENODO_FILES = 100
STORED = {".cool", ".pt", ".gz", ".tar", ".zip"}     # already compressed or incompressible

MANIFEST: list[tuple[str, str]] = []   # (file name, description)


def subst(text: str) -> str:
    return text.replace(str(HM), "${HICMULTI_HM}").replace(str(DATA), "${HICMULTI_DATA}")


class Zip:
    """One topic archive. In a dry run only the existence of the sources is checked."""

    def __init__(self, out: Path, name: str, desc: str, dry: bool):
        MANIFEST.append((name, desc))
        self.dry = dry
        self.zf = None if dry else zipfile.ZipFile(out / name, "w", zipfile.ZIP_DEFLATED, allowZip64=True)
        self.n = 0

    @staticmethod
    def _ctype(arc: str) -> int:
        return zipfile.ZIP_STORED if Path(arc).suffix in STORED else zipfile.ZIP_DEFLATED

    def add_bytes(self, arc: str, data: bytes, mtime: float | None = None):
        self.n += 1
        if self.dry:
            return
        zi = zipfile.ZipInfo(arc, date_time=time.localtime(mtime or time.time())[:6])
        zi.compress_type = self._ctype(arc)
        zi.external_attr = 0o644 << 16
        self.zf.writestr(zi, data)

    def add_path(self, src: Path, arc: str, text_subst: bool = False):
        assert src.is_file(), src
        if text_subst:
            self.add_bytes(arc, subst(src.read_text()).encode(), src.stat().st_mtime)
            return
        self.n += 1
        if not self.dry:
            self.zf.write(src, arc, compress_type=self._ctype(arc))

    def add_tree(self, src: Path, arc: str, skip: tuple[str, ...] = ()):
        assert src.is_dir(), src
        for p in sorted(src.rglob("*")):
            if p.is_file() and p.name not in skip:
                self.add_path(p, f"{arc}/{p.relative_to(src)}")

    def add_tar_members(self, tar_path: Path, arc: str):
        """The files of a tar, placed under arc/ (the nested tars of the earlier staging)."""
        assert tar_path.is_file(), tar_path
        if self.dry:
            self.n += 1
            return
        with tarfile.open(tar_path) as tar:
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                zi = zipfile.ZipInfo(f"{arc}/{m.name}", date_time=time.localtime(m.mtime)[:6])
                zi.compress_type = self._ctype(m.name)
                zi.external_attr = 0o644 << 16
                with tar.extractfile(m) as f, self.zf.open(zi, "w", force_zip64=True) as w:
                    shutil.copyfileobj(f, w, 1 << 24)
                self.n += 1

    def close(self):
        if self.zf is not None:
            self.zf.close()


def add_checkpoint_dir(z: Zip, ckpt: Path, arc: str, weights: bool = True):
    """checkpoint-best without optim.pt; yaml/json with path substitution; res_<bp> normaliser
    folders of a shared multi-resolution checkpoint included; weights=False leaves out model.pt
    (a view that shares the weights of its shared checkpoint)."""
    for f in sorted(ckpt.iterdir()):
        if f.name == "optim.pt" or (f.name == "model.pt" and not weights):
            continue
        if f.is_dir():
            if f.name.startswith("res_"):
                add_checkpoint_dir(z, f, f"{arc}/{f.name}")
            continue
        z.add_path(f, f"{arc}/{f.name}", text_subst=f.suffix in (".yaml", ".json"))


def prediction_config(view_yaml: Path, k: str) -> str:
    """The training configuration of a view reduced to what a prediction of a new cell needs: one
    template sample on the consensus grid, the consensus as reference, and the genome, motifs and cache
    through environment variables."""
    c = yaml.safe_load(view_yaml.read_text())
    cons = f"${{FLEXCIBLE_MAPS}}/consensus12_proper_{k}kb.cool"
    c["data"]["samples"] = [{"name": "template", "cool_path": cons, "bigwig_dir": "${FLEXCIBLE_TRACKS}",
                             "missing_tracks": [], "assay": "hic"}]
    c["data"]["reference_cool"] = cons
    for key in ("holdout_bed", "train_regions_bed"):
        c["data"].pop(key, None)
    c["sequence"]["fasta"] = "${FLEXCIBLE_GENOME}"
    c["sequence"]["cache_dir"] = "${FLEXCIBLE_CACHE}/genome"
    c["motifs"]["dir"] = "${FLEXCIBLE_MOTIFS}"
    c["train"]["output_dir"] = "${FLEXCIBLE_CACHE}/run"
    c["train"]["cache_dir"] = "${FLEXCIBLE_CACHE}/windows"
    c["train"]["cohesin_aug"] = 0.0
    c["train"]["cohesin_aug_operator"] = None
    head = ("# FlexCible prediction configuration, %s kb view of the released model.\n"
            "# Set FLEXCIBLE_GENOME (hg38 FASTA), FLEXCIBLE_MOTIFS (folder with the JASPAR files),\n"
            "# FLEXCIBLE_CACHE (writable folder) and FLEXCIBLE_MAPS (folder with the consensus12 maps);\n"
            "# the tracks of the cell are given with --bigwig-dir. The template sample sets the bin grid\n"
            "# and the output scale (normaliser of GM12878, as in the manuscript).\n" % k)
    return head + yaml.safe_dump(c, sort_keys=False)


def release(z: Zip, run: str, name: str):
    """Shared weights, four views without weights, prediction and training configurations."""
    top = "flexcible_release"
    shared = RUNS / run / "checkpoint-best"
    add_checkpoint_dir(z, shared, f"{top}/{name}/checkpoint-best")
    for extra in ("learning_curve.csv", "best"):
        if (RUNS / run / extra).exists():
            z.add_path(RUNS / run / extra, f"{top}/{name}/{extra}")
    for k in BINS:
        view = RUNS / f"{run}_{k}kb"
        vck = view / "checkpoint-best"
        assert filecmp.cmp(vck / "model.pt", shared / "model.pt", shallow=False), f"{view}: weights differ from the shared checkpoint"
        arc = f"{top}/{name}_{k}kb"
        add_checkpoint_dir(z, vck, f"{arc}/checkpoint-best", weights=False)
        # the template sample of the prediction configuration uses the GM12878 output normaliser
        z.add_path(vck / "norm_matrix_gm12878.json", f"{arc}/checkpoint-best/norm_matrix_template.json")
        z.add_bytes(f"{arc}/multires_source.json",
                    json.dumps({"shared_checkpoint": f"../{name}/checkpoint-best", "resolution": int(k) * 1000}, indent=1).encode())
        z.add_bytes(f"{top}/{name}_{k}kb.yaml", prediction_config(RUNS / f"{run}_{k}kb.yaml", k).encode())
        z.add_path(RUNS / f"{run}_{k}kb.yaml", f"{arc}/training_config.yaml", text_subst=True)
        for f in sorted(view.glob("*test_eval_*.json")):
            z.add_path(f, f"{arc}/evaluation/{f.name}")


def operators(z: Zip):
    """Depletion-augmentation operators referenced by the training configurations."""
    for k in BINS:
        op = HM / "cohesin_fix" / f"operator_{k}kb.pt"
        if op.exists():
            z.add_path(op, f"flexcible_release/cohesin_fix/operator_{k}kb.pt")


def run_dir(z: Zip, run: str, arc: str):
    """One dedicated run from its folder: checkpoint-best without optim.pt, learning curve, evaluations."""
    d = RUNS / run
    add_checkpoint_dir(z, d / "checkpoint-best", f"{arc}/{run}/checkpoint-best")
    for f in sorted(d.iterdir()):
        if f.is_file() and (f.name in ("learning_curve.csv", "best") or f.name.endswith(".json")):
            z.add_path(f, f"{arc}/{run}/{f.name}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


README = """# FlexCible: released model, consensus maps and result tables

Data record of the manuscript "FlexCible: flexible-input prediction of chromatin architecture from chromatin
marks and a consensus matrix" (Joachim Wolff, Hannover Medical School). Code and documentation:
{repo}. DOI of this record: {doi}. One zip per topic; `manifest.csv` lists every file with its size
and SHA-256.

## flexcible_released_model.zip

The released model: one network for 2, 5, 10 and 25 kb and for any subset of the 13 chromatin marks and
DNA methylation (`flexcible/`), and the twin trained with chromosomes 2, 3, 11, 14 and 17 held out
(`flexcible_chrho/`, used only for the comparison with the published predictors on those chromosomes).

    flexcible_release/
      flexcible/checkpoint-best/          shared weights (model.pt), res_<bp>/ normalisers per trained bin size
      flexcible_<k>kb/checkpoint-best/    view for k = 25, 10, 5, 2: configuration, channels, normalisers
                                          (multires_source.json points to the shared weights)
      flexcible_<k>kb.yaml                prediction configuration of that bin size
      flexcible_<k>kb/training_config.yaml, evaluation/   training configuration, test evaluations
      flexcible_chrho*, flexcible_chrho_<k>kb*            the same for the twin
      cohesin_fix/operator_<k>kb.pt       depletion-augmentation operators used in training

The prediction configurations take four folders from the environment, so no path has to be edited:

    export FLEXCIBLE_GENOME=/path/hg38.fa          # hg38 FASTA (see the tutorial)
    export FLEXCIBLE_MOTIFS=/path/motifs           # MA0139.2, MA0095.3, MA0088.2, MA1522.1 .jaspar from JASPAR
    export FLEXCIBLE_CACHE=/path/cache             # any writable folder (genome features are cached once)
    export FLEXCIBLE_MAPS=/path/consensus12        # the folder unpacked from consensus12_maps.zip
    unzip flexcible_released_model.zip
    flexcible predict --config flexcible_release/flexcible_10kb.yaml \\
        --checkpoint flexcible_release/flexcible_10kb/checkpoint-best \\
        --bigwig-dir <folder with CTCF.bigwig, H3K27ac.bigwig, ...> --cell <name> --chroms all --tta \\
        --out <name>_10kb.cool

Marks that were not measured are left out of the bigWig folder or declared with `--absent-track`;
`--no-reference` predicts from the marks alone; `--reference` replaces the 12-cell consensus by another
log O/E map on the same grid (a cohort consensus); `--renormalize-tracks` refits the track statistics on a
sample whose signal scale differs from ENCODE fold change (CUT&Tag); `--bin-size` predicts at another bin
size and needs a reference at that bin size. The output is a cool file with log O/E values within 2 Mb of
the diagonal, read by HiCExplorer, cooltools and HiGlass.

## benchmark_models.zip

The models of the comparisons in the manuscript, one folder per run with `checkpoint-best/` (weights,
configuration, normalisers), the learning curve and the evaluations; the configurations keep the training
paths with `${{HICMULTI_HM}}` and `${{HICMULTI_DATA}}` for the data roots.

    benchmark_models/
      dedicated_refin/     dedicated per-bin-size models, consensus as input
      dedicated_refres/    consensus added to the output
      dedicated_honest/    marks alone
      akita_grid/          the two GM12878 models trained on Akita's targets and 2,048 bp grid
      aml/                 AML fold models (f10, f11, f12: consensus input, marks alone, difference tracks)
                           and the seed runs of the consensus-input models

Run names: `all12` trained on the 12 cell lines; `refin` consensus input; `refres` consensus residual;
`honest` marks alone; `drop` track dropout; `chrho` chromosomes held out; `s1`, `s2` further seeds; `aml`
AML-inclusive; `f10`.. fold; `_<k>kb` bin size.

## Maps

Distance-normalised log O/E maps (cool format, hg38, 2 Mb band from the diagonal) as read by the models.
A `.json` next to a consensus map names its cells. The 2 kb maps and the Akita-grid map hold a 1 Mb band
(pixels up to 511 bins from the diagonal), the whole span of the 512-bin window the model reads at these
bin sizes; every value in the band equals the original (`scripts/zenodo_trim_cool.py --check`).

* `consensus12_maps.zip`: `consensus12/consensus12_proper_<k>kb.cool`, the consensus of the 12 training
  cell lines at 25, 10, 5 and 2 kb; the reference for predicting a cell without Hi-C data.
* `loo_maps_25_10_5kb.zip`, `loo_maps_2kb.zip`: `consensus_loo_<k>kb/consensus_loo_<cell>_proper_<k>kb.cool`,
  the leave-one-out consensus read by each training cell (the 11 others), for reproducing the benchmarks.
* `other_maps.zip`: `akita_grid/akita_consensus4_2048bp.cool` (four-cell consensus on Akita's grid),
  `aml_hspc_references/` (mean map of the training patients of each AML fold, and the HSPC reference, at
  25 and 10 kb), `mouse_consensus_loo/` (mESC, CH12.LX and heart, mm10, 25 and 10 kb).

## tables.zip

`splits/` (validation positions, test windows, held-out chromosomes, AML folds, same-donor checks),
`objects/` (AML-specific loops and stripes with matched controls, derived from the published
Supplementary Tables of Xu et al.; cell-specific object tables), `results/` (every result table of the
working repository: csv, json, md, txt), `supp_tables/` (the generated bodies of the supplementary
tables), `numbers_provenance.csv` (every number of the manuscript with its source file) and
`DATA_DICTIONARY.md`.

Licence: CC BY 4.0 (`LICENSE.txt`); the underlying public data and their terms are listed in
`NOTICE_DATA.md` and `SOURCES.csv`.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HM / "zenodo_record_v1"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace", action="store_true", help="remove an existing out folder first")
    a = ap.parse_args()
    out = Path(a.out)
    dry = a.dry_run
    if not dry:
        if out.exists():
            if not a.replace:
                raise SystemExit(f"{out} exists; pass --replace")
            shutil.rmtree(out)
        out.mkdir(parents=True)

    # --- released model
    z = Zip(out, "flexcible_released_model.zip", "released model (one network, 2 to 25 kb, any subset of marks) and its chromosome-held-out twin, with prediction configurations", dry)
    release(z, "all12one_fullrd", "flexcible")
    release(z, "all12one_fullrd_chrho", "flexcible_chrho")
    operators(z)
    z.close(); print(f"  flexcible_released_model.zip: {z.n} members")

    # --- benchmark models
    old1 = OLD / "record1_models_12cell/checkpoints"
    old2 = OLD / "record2_models_aml/checkpoints"
    z = Zip(out, "benchmark_models.zip", "dedicated per-bin-size models (consensus input, residual, marks alone; chrho, seeds), Akita-grid models, AML fold models and seed runs", dry)
    for key, arc in (("all12refin_drop", "dedicated_refin"), ("all12refres_drop", "dedicated_refres"),
                     ("all12honest_drop", "dedicated_honest"), ("akita_native", "akita_grid")):
        for t in sorted(p for p in old1.glob("*.tar") if p.name.startswith(key)):
            z.add_tar_members(t, f"benchmark_models/{arc}")
    for t in sorted(old2.glob("*.tar")):
        z.add_tar_members(t, "benchmark_models/aml")
    seeds = [f"all12aml_refin_drop_f{f}_s{s}_10kb" for f in (10, 11, 12) for s in (1, 2)]
    missing = [r for r in seeds if not (RUNS / r / "checkpoint-best" / "model.pt").exists()]
    if missing:   # fold 10 has two seeds only (an empty run directory was left behind, never trained)
        print("  seed runs without checkpoint, skipped:", " ".join(missing))
    for r in seeds:
        if r not in missing:
            run_dir(z, r, "benchmark_models/aml")
    z.close(); print(f"  benchmark_models.zip: {z.n} members")

    # --- maps
    z = Zip(out, "consensus12_maps.zip", "12-cell consensus at 25, 10, 5 kb (2 Mb band) and 2 kb (1 Mb band), with cell lists", dry)
    for k in ("25", "10", "5"):
        for suf in (".cool", ".cool.json"):
            z.add_path(OE / f"consensus12_proper_{k}kb{suf}", f"consensus12/consensus12_proper_{k}kb{suf}")
    for suf in (".cool", ".cool.json"):
        z.add_path(TRIM / f"consensus12_proper_2kb{suf}", f"consensus12/consensus12_proper_2kb{suf}")
    z.close()
    z = Zip(out, "loo_maps_25_10_5kb.zip", "12 leave-one-out consensus maps at 25, 10 and 5 kb, 2 Mb band, with cell lists", dry)
    for k in ("25", "10", "5"):
        for c in CELLS:
            for suf in (".cool", ".cool.json"):
                z.add_path(OE / f"consensus_loo_{c}_proper_{k}kb{suf}", f"consensus_loo_{k}kb/consensus_loo_{c}_proper_{k}kb{suf}")
    z.close()
    z = Zip(out, "loo_maps_2kb.zip", "12 leave-one-out consensus maps at 2 kb, 1 Mb band, with cell lists", dry)
    for c in CELLS:
        for suf in (".cool", ".cool.json"):
            z.add_path(TRIM / f"consensus_loo_{c}_proper_2kb{suf}", f"consensus_loo_2kb/consensus_loo_{c}_proper_2kb{suf}")
    z.close()
    z = Zip(out, "other_maps.zip", "four-cell consensus on Akita's 2,048 bp grid (1 Mb band); AML fold and HSPC references (25, 10 kb); mouse leave-one-out maps (mm10, 25, 10 kb)", dry)
    z.add_path(TRIM / "akita_consensus4_2048bp.cool", "akita_grid/akita_consensus4_2048bp.cool")
    for f in sorted((OLD / "record3_maps/maps").iterdir()):
        if f.name.startswith(("amltrain_", "hspccons_")):
            z.add_path(f, f"aml_hspc_references/{f.name}")
    for c in ("mesc", "ch12", "heart"):
        for k in ("10", "25"):
            z.add_path(MOUSE_OE / f"consensus_loo_{c}_proper_{k}kb.cool", f"mouse_consensus_loo/consensus_loo_{c}_proper_{k}kb.cool")
    z.close()

    # --- tables
    old4 = OLD / "record4_tables"
    z = Zip(out, "tables.zip", "split files, AML object tables, every result table, generated supplementary tables, numbers_provenance.csv, DATA_DICTIONARY.md", dry)
    z.add_tree(old4 / "splits", "splits")
    z.add_tree(old4 / "objects", "objects")
    exts = (".csv", ".json", ".md", ".txt", ".csv.gz")
    for p in sorted(RESULTS.rglob("*")):
        if p.is_file() and p.name.endswith(exts):
            z.add_path(p, f"results/{p.relative_to(RESULTS)}")
    z.add_tree(MANUSCRIPT / "supp_tables", "supp_tables")
    z.add_path(MANUSCRIPT / "numbers_provenance.csv", "numbers_provenance.csv")
    z.add_path(old4 / "DATA_DICTIONARY.md", "DATA_DICTIONARY.md")
    z.close(); print(f"  tables.zip: {z.n} members")

    # --- documents
    for name, desc in (("LICENSE.txt", "licence of the derived data (CC BY 4.0)"),
                       ("SOURCES.csv", "accessions of the underlying public data")):
        MANIFEST.append((name, desc))
        if not dry:
            shutil.copyfile(OLD / "record1_models_12cell" / name, out / name)
    MANIFEST.append(("NOTICE_DATA.md", "sources, accessions and terms of the underlying data"))
    MANIFEST.append(("README.md", "contents and use of this record"))
    MANIFEST.append(("metadata.json", "Zenodo metadata"))
    if not dry:
        notice = (OLD / "record1_models_12cell/NOTICE_DATA.md").read_text()
        notice = (notice.replace("(records 1 and 2)", "(`flexcible_released_model.zip`, `benchmark_models.zip`)")
                  .replace("(record 3)", "(the map zips)").replace("(record 4)", "(`tables.zip`)"))
        (out / "NOTICE_DATA.md").write_text(notice)
        (out / "README.md").write_text(README.format(doi=DOI, repo=REPO_URL))
        meta = {"metadata": {
            "title": "FlexCible: released model, consensus maps and result tables",
            "upload_type": "dataset",
            "description": "Trained checkpoints (released FlexCible model for 2 to 25 kb and any subset of chromatin marks, its chromosome-held-out twin, the dedicated per-bin-size models, the Akita-grid models and the AML fold models), the consensus Hi-C maps (12-cell and leave-one-out, log observed/expected), split files and every result table of the manuscript 'FlexCible: flexible-input prediction of chromatin architecture from chromatin marks and a consensus matrix'.",
            "creators": [{"name": "Wolff, Joachim", "affiliation": "Department of Hematology, Hemostasis, Oncology and Cell Therapy, Hannover Medical School", "orcid": "ORCID_TO_FILL"}],
            "license": "cc-by-4.0", "access_right": "open", "doi": DOI,
            "keywords": ["Hi-C", "chromatin architecture", "deep learning", "model checkpoints", "consensus contact map"],
            "related_identifiers": [{"identifier": REPO_URL, "relation": "isSupplementTo", "scheme": "url", "resource_type": "software"}],
            "version": "1.0.0", "language": "eng"}}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # --- manifest and limits
    if dry:
        print(f"dry run: {len(MANIFEST) + 1} files planned")
        return
    rows, total = [], 0
    for name, desc in MANIFEST:
        size = (out / name).stat().st_size
        total += size
        rows.append(f'{name},{size},{sha256(out / name)},"{desc}"')
    (out / "manifest.csv").write_text("path,size_bytes,sha256,description\n" + "\n".join(rows) + "\n")
    n = len(rows) + 1
    print(f"{n} files, {total / 1e9:.2f} GB; Zenodo default limits {ZENODO_FILES} files, {ZENODO_GB} GB -> "
          f"{'OK' if n <= ZENODO_FILES and total / 1e9 <= ZENODO_GB else 'OVER LIMIT'}")


if __name__ == "__main__":
    main()
