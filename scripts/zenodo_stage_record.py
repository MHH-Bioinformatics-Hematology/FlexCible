"""Assemble the single Zenodo record of the FlexCible manuscript (DOI 10.5281/zenodo.23922797).

Writes a self-contained folder (default /mnt/processed/wolffjoa/hic-multiome/zenodo_record_v1) with

  checkpoints/   flexcible_release.tar            released model (one network, 2 to 25 kb) with its four
                                                  per-bin-size views, configurations and normalisers
                 flexcible_release_chrho.tar      twin trained with chromosomes 2, 3, 11, 14, 17 held out
                 models_dedicated_refin.tar       dedicated per-bin-size models, consensus input (+ chrho, seeds)
                 models_dedicated_refres.tar      the same with the consensus added to the output
                 models_dedicated_honest.tar      the same without consensus (marks alone)
                 models_akita_grid.tar            the two GM12878 models of the Akita comparison
                 models_aml.tar                   AML fold models (consensus input, marks alone, difference
                                                  tracks) and the seed runs of the consensus-input models
  maps/          consensus12_proper_{25,10,5}kb.cool    12-cell consensus, 2 Mb band
                 consensus12_proper_2kb.cool            12-cell consensus, 1 Mb band (see README)
                 consensus_loo_{25,10,5}kb.tar          12 leave-one-out maps per bin size, 2 Mb band
                 consensus_loo_<cell>_proper_2kb.cool   12 leave-one-out maps at 2 kb, 1 Mb band
                 akita_consensus4_2048bp.cool           four-cell consensus on Akita's grid, 1 Mb band
                 amltrain_f{10,11,12}_all_{10,25}kb.cool, hspccons_{10,25}kb.cool   AML fold references
                 mouse_consensus_loo.tar                mouse leave-one-out maps (mESC, CH12.LX, heart)
  tables/        splits.tar, objects.tar, results_tables.tar.gz, supp_tables.tar, numbers_provenance.csv,
                 DATA_DICTIONARY.md
  README.md, LICENSE.txt, NOTICE_DATA.md, SOURCES.csv, manifest.csv (sha256), metadata.json

Nothing under the source trees is modified. The old four-record staging folder is read for the tars and
documents it already holds. Checkpoint tars contain the weights, the configuration and the normalisers,
never the optimiser state; absolute paths in configurations are replaced by ${HICMULTI_HM} and
${HICMULTI_DATA} as in the earlier staging.

    python scripts/zenodo_stage_record.py [--out DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

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
ZENODO_GB, ZENODO_FILES = 50.0, 100

MANIFEST: list[tuple[str, str]] = []   # (relative path, description)


def subst(text: str) -> str:
    return text.replace(str(HM), "${HICMULTI_HM}").replace(str(DATA), "${HICMULTI_DATA}")


def add_file(tar: tarfile.TarFile, src: Path, arcname: str, text_subst: bool = False):
    if text_subst:
        data = subst(src.read_text()).encode()
        ti = tarfile.TarInfo(arcname)
        ti.size = len(data)
        ti.mtime = int(src.stat().st_mtime)
        tar.addfile(ti, io.BytesIO(data))
    else:
        tar.add(src, arcname=arcname, recursive=False)


def add_checkpoint_dir(tar: tarfile.TarFile, ckpt: Path, arc: str):
    """checkpoint-best without optim.pt; yaml/json with path substitution."""
    for f in sorted(ckpt.iterdir()):
        if f.name == "optim.pt" or f.is_dir():
            continue
        add_file(tar, f, f"{arc}/{f.name}", text_subst=f.suffix in (".yaml", ".json"))


def release_tar(out: Path, name: str, run: str, dry: bool):
    """Shared weights + the four per-bin-size views."""
    path = out / "checkpoints" / f"{name}.tar"
    MANIFEST.append((str(path.relative_to(out)), f"{run}: shared weights, four bin-size views, configurations, normalisers"))
    if dry:
        return
    with tarfile.open(path, "w") as tar:
        shared = RUNS / run
        add_checkpoint_dir(tar, shared / "checkpoint-best", f"{name}/checkpoint-best")
        for extra in ("learning_curve.csv", "best"):
            if (shared / extra).exists():
                add_file(tar, shared / extra, f"{name}/{extra}")
        for k in BINS:
            view = RUNS / f"{run}_{k}kb"
            add_checkpoint_dir(tar, view / "checkpoint-best", f"{name}/{run}_{k}kb/checkpoint-best")
            src_json = {"shared_checkpoint": f"../checkpoint-best", "resolution": int(k) * 1000}
            data = json.dumps(src_json, indent=1).encode()
            ti = tarfile.TarInfo(f"{name}/{run}_{k}kb/multires_source.json")
            ti.size = len(data); ti.mtime = int(time.time())
            tar.addfile(ti, io.BytesIO(data))
            add_file(tar, RUNS / f"{run}_{k}kb.yaml", f"{name}/{run}_{k}kb.yaml", text_subst=True)
            for f in sorted(view.glob("*test_eval_*.json")):
                add_file(tar, f, f"{name}/{run}_{k}kb/{f.name}")
        for k in BINS:   # depletion-augmentation operators referenced by the training configuration
            op = HM / "cohesin_fix" / f"operator_{k}kb.pt"
            if op.exists():
                add_file(tar, op, f"{name}/cohesin_fix/operator_{k}kb.pt")


def run_tar_bytes(run: str) -> bytes:
    """One dedicated run (checkpoint-best without optim.pt, learning curve, evaluations) as an in-memory tar."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        d = RUNS / run
        add_checkpoint_dir(tar, d / "checkpoint-best", f"{run}/checkpoint-best")
        for f in sorted(d.iterdir()):
            if f.is_file() and (f.name in ("learning_curve.csv", "best") or f.name.endswith(".json")):
                add_file(tar, f, f"{run}/{f.name}")
    return buf.getvalue()


def bundle(out: Path, name: str, old_tars: list[Path], new_runs: list[str], desc: str, dry: bool):
    """A tar of per-run tars: existing ones copied from the old staging, new ones built from the run folders."""
    path = out / "checkpoints" / f"{name}.tar"
    MANIFEST.append((str(path.relative_to(out)), desc))
    if dry:
        print(f"  {name}: {len(old_tars)} staged tars + {len(new_runs)} new runs")
        return
    with tarfile.open(path, "w") as tar:
        for t in old_tars:
            tar.add(t, arcname=f"{name}/{t.name}")
        for run in new_runs:
            data = run_tar_bytes(run)
            ti = tarfile.TarInfo(f"{name}/{run}.tar")
            ti.size = len(data); ti.mtime = int(time.time())
            tar.addfile(ti, io.BytesIO(data))


def copy(out: Path, src: Path, rel: str, desc: str, dry: bool):
    MANIFEST.append((rel, desc))
    if dry:
        assert src.exists(), src
        return
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def tar_of(out: Path, rel: str, items: list[tuple[Path, str]], desc: str, dry: bool, gz: bool = False):
    MANIFEST.append((rel, desc))
    if dry:
        for p, _ in items:
            assert p.exists(), p
        return
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out / rel, "w:gz" if gz else "w") as tar:
        for p, arc in items:
            tar.add(p, arcname=arc)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


README = """# FlexCible: released model, consensus maps and result tables

Data record of the manuscript "FlexCible: flexible-input prediction of chromatin architecture from chromatin
marks and a consensus matrix" (Joachim Wolff, Hannover Medical School). Code: the `flexcible` package
(repository named in the manuscript). DOI of this record: {doi}.

## checkpoints/

* `flexcible_release.tar`: the released model, one network for 2, 5, 10 and 25 kb and for any subset of the
  13 chromatin marks and DNA methylation. Unpacks to `flexcible_release/checkpoint-best/` (shared weights
  `model.pt`, no optimiser state) and four views `all12one_fullrd_<k>kb/` with the configuration, the
  normalisers and `multires_source.json` pointing to the shared weights. Predict with the view of the bin
  size you want:

      tar xf flexcible_release.tar
      flexcible predict --config flexcible_release/all12one_fullrd_10kb.yaml \\
          --checkpoint flexcible_release/all12one_fullrd_10kb/checkpoint-best \\
          --bigwig-dir <bigwigs of your cell> --cell <name> --chroms all --tta \\
          --reference consensus12_proper_10kb.cool --out prediction_10kb.cool

  Marks you do not have are left out of the bigWig folder or declared with `--absent-track`;
  `--no-reference` predicts from the marks alone; `--bin-size` predicts at other bin sizes (needs a
  reference at that bin size). Configurations use `${{HICMULTI_HM}}` and `${{HICMULTI_DATA}}` for the
  data roots. `cohesin_fix/operator_<k>kb.pt` are the depletion-augmentation operators used in training.
* `flexcible_release_chrho.tar`: the same recipe trained with chromosomes 2, 3, 11, 14 and 17 held out;
  used only for the comparison with the published predictors on those chromosomes.
* `models_dedicated_refin.tar`, `models_dedicated_refres.tar`, `models_dedicated_honest.tar`: the dedicated
  per-bin-size models of the supplementary comparisons (consensus as input, consensus added to the output,
  marks alone), with the chromosome-held-out versions (`_chrho`) and further seeds (`_s1`, `_s2`) where
  trained. Each is a tar of per-run tars with the layout `<run>/checkpoint-best/`.
* `models_akita_grid.tar`: the two GM12878 models trained on Akita's targets and 2,048 bp grid.
* `models_aml.tar`: the AML fold models (`f10`, `f11`, `f12`: consensus input, marks alone, difference
  tracks) and the seed runs of the consensus-input models.

Model names: `all12` trained on the 12 cell lines; `one` one network for all bin sizes; `fullrd` all inputs
with reference dropout; `refin` consensus input; `refres` consensus residual; `honest` marks alone; `drop`
track dropout; `chrho` chromosomes held out; `s1`, `s2` seeds; `aml` AML-inclusive; `_<k>kb` bin size.

## maps/

Distance-normalised log O/E maps (cool format, hg38) as used by the models: consensus of the 12 training
cell lines (`consensus12_proper_<k>kb.cool`), the leave-one-out consensus read by each training cell
(`consensus_loo_<cell>_proper_<k>kb.cool`, 25, 10 and 5 kb bundled per bin size), the four-cell consensus
of the Akita comparison on Akita's grid, the AML fold references (mean of the training patients of a fold)
and the HSPC reference, and the mouse leave-one-out maps. The 25, 10 and 5 kb maps hold the 2 Mb band
from the diagonal. The 2 kb maps and the Akita-grid map hold a 1 Mb band (pixels up to 511 bins from the
diagonal), which is the whole span of the 512-bin window the model reads at these bin sizes; every value in
the band is identical to the original 2 Mb-band map (`scripts/zenodo_trim_cool.py --check`). A `.json`
next to a consensus map names its cells.

## tables/

`splits.tar` (validation positions, test windows, held-out chromosomes, AML folds, same-donor checks),
`objects.tar` (AML-specific loops and stripes with matched controls, derived from the published
Supplementary Tables of Xu et al.; cell-specific object tables), `results_tables.tar.gz` (every result
table of the working repository: csv, json, md, txt), `supp_tables.tar` (the generated bodies of the
supplementary tables), `numbers_provenance.csv` (every number of the manuscript with its source file) and
`DATA_DICTIONARY.md`.

`manifest.csv` lists every file with its size and SHA-256. Licence: CC BY 4.0 (`LICENSE.txt`); the
underlying public data and their terms are listed in `NOTICE_DATA.md` and `SOURCES.csv`.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HM / "zenodo_record_v1"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    dry = a.dry_run
    if not dry:
        if out.exists():
            raise SystemExit(f"{out} exists; remove it first")
        for d in ("checkpoints", "maps", "tables"):
            (out / d).mkdir(parents=True)

    # --- checkpoints
    release_tar(out, "flexcible_release", "all12one_fullrd", dry)
    release_tar(out, "flexcible_release_chrho", "all12one_fullrd_chrho", dry)
    old1 = OLD / "record1_models_12cell/checkpoints"
    old2 = OLD / "record2_models_aml/checkpoints"
    fam = lambda key: sorted(p for p in old1.glob("*.tar") if p.name.startswith(key))
    bundle(out, "models_dedicated_refin", fam("all12refin_drop"), [], "dedicated per-bin-size models, consensus input (incl. chrho and seeds)", dry)
    bundle(out, "models_dedicated_refres", fam("all12refres_drop"), [], "dedicated per-bin-size models, consensus residual (incl. chrho and seeds)", dry)
    bundle(out, "models_dedicated_honest", fam("all12honest_drop"), [], "dedicated per-bin-size models, marks alone (incl. chrho and seeds)", dry)
    bundle(out, "models_akita_grid", fam("akita_native"), [], "GM12878 models of the Akita comparison (Akita grid and targets)", dry)
    seeds = [f"all12aml_refin_drop_f{f}_s{s}_10kb" for f in (10, 11, 12) for s in (1, 2)]
    bundle(out, "models_aml", sorted(old2.glob("*.tar")), seeds, "AML fold models and seed runs", dry)

    # --- maps
    for k in ("25", "10", "5"):
        copy(out, OE / f"consensus12_proper_{k}kb.cool", f"maps/consensus12_proper_{k}kb.cool", f"12-cell consensus, {k} kb, 2 Mb band", dry)
        copy(out, OE / f"consensus12_proper_{k}kb.cool.json", f"maps/consensus12_proper_{k}kb.cool.json", "cells of the consensus", dry)
        items = []
        for c in CELLS:
            items.append((OE / f"consensus_loo_{c}_proper_{k}kb.cool", f"consensus_loo_{k}kb/consensus_loo_{c}_proper_{k}kb.cool"))
            items.append((OE / f"consensus_loo_{c}_proper_{k}kb.cool.json", f"consensus_loo_{k}kb/consensus_loo_{c}_proper_{k}kb.cool.json"))
        tar_of(out, f"maps/consensus_loo_{k}kb.tar", items, f"12 leave-one-out consensus maps, {k} kb, 2 Mb band", dry)
    copy(out, TRIM / "consensus12_proper_2kb.cool", "maps/consensus12_proper_2kb.cool", "12-cell consensus, 2 kb, 1 Mb band", dry)
    copy(out, TRIM / "consensus12_proper_2kb.cool.json", "maps/consensus12_proper_2kb.cool.json", "cells of the consensus", dry)
    for c in CELLS:
        copy(out, TRIM / f"consensus_loo_{c}_proper_2kb.cool", f"maps/consensus_loo_{c}_proper_2kb.cool", f"leave-one-out consensus for {c}, 2 kb, 1 Mb band", dry)
        copy(out, TRIM / f"consensus_loo_{c}_proper_2kb.cool.json", f"maps/consensus_loo_{c}_proper_2kb.cool.json", "cells of the consensus", dry)
    copy(out, TRIM / "akita_consensus4_2048bp.cool", "maps/akita_consensus4_2048bp.cool", "four-cell consensus on Akita's 2,048 bp grid, 1 Mb band", dry)
    old3 = OLD / "record3_maps/maps"
    for f in sorted(old3.iterdir()):
        if f.name.startswith(("amltrain_", "hspccons_")):
            copy(out, f, f"maps/{f.name}", "AML fold reference (mean of the training patients) or HSPC reference" if f.suffix == ".cool" else "cells of the map", dry)
    items = []
    for c in ("mesc", "ch12", "heart"):
        for k in ("10", "25"):
            items.append((MOUSE_OE / f"consensus_loo_{c}_proper_{k}kb.cool", f"mouse_consensus_loo/consensus_loo_{c}_proper_{k}kb.cool"))
    tar_of(out, "maps/mouse_consensus_loo.tar", items, "mouse leave-one-out consensus maps (mm10), 10 and 25 kb", dry)

    # --- tables
    old4 = OLD / "record4_tables"
    tar_of(out, "tables/splits.tar", [(old4 / "splits", "splits")], "split files: validation positions, test windows, held-out chromosomes, AML folds, same-donor checks", dry)
    tar_of(out, "tables/objects.tar", [(old4 / "objects", "objects")], "AML-specific loop and stripe tables with controls; cell-specific objects", dry)
    exts = (".csv", ".json", ".md", ".txt", ".csv.gz")
    items = [(p, f"results/{p.relative_to(RESULTS)}") for p in sorted(RESULTS.rglob("*")) if p.is_file() and p.name.endswith(exts)]
    tar_of(out, "tables/results_tables.tar.gz", items, "all result tables of the working repository", dry, gz=True)
    tar_of(out, "tables/supp_tables.tar", [(MANUSCRIPT / "supp_tables", "supp_tables")], "generated bodies of the supplementary tables", dry)
    copy(out, MANUSCRIPT / "numbers_provenance.csv", "tables/numbers_provenance.csv", "every number of the manuscript with its source file", dry)
    copy(out, old4 / "DATA_DICTIONARY.md", "tables/DATA_DICTIONARY.md", "columns of the key tables", dry)

    # --- documents
    copy(out, OLD / "record1_models_12cell/LICENSE.txt", "LICENSE.txt", "licence of the derived data (CC BY 4.0)", dry)
    copy(out, OLD / "record1_models_12cell/SOURCES.csv", "SOURCES.csv", "accessions of the underlying public data", dry)
    MANIFEST.append(("NOTICE_DATA.md", "sources, accessions and terms of the underlying data"))
    MANIFEST.append(("README.md", "contents and use of this record"))
    MANIFEST.append(("metadata.json", "Zenodo metadata"))
    if not dry:
        notice = (OLD / "record1_models_12cell/NOTICE_DATA.md").read_text()
        notice = notice.replace("(records 1 and 2)", "(`checkpoints/`)").replace("(record 3)", "(`maps/`)").replace("(record 4)", "(`tables/`)")
        (out / "NOTICE_DATA.md").write_text(notice)
        (out / "README.md").write_text(README.format(doi=DOI))
        meta = {"metadata": {
            "title": "FlexCible: released model, consensus maps and result tables",
            "upload_type": "dataset",
            "description": "Trained checkpoints (released FlexCible model for 2 to 25 kb and any subset of chromatin marks, its chromosome-held-out twin, the dedicated per-bin-size models, the Akita-grid models and the AML fold models), the consensus Hi-C maps (12-cell and leave-one-out, log observed/expected), split files and every result table of the manuscript 'FlexCible: flexible-input prediction of chromatin architecture from chromatin marks and a consensus matrix'.",
            "creators": [{"name": "Wolff, Joachim", "affiliation": "Department of Hematology, Hemostasis, Oncology and Cell Therapy, Hannover Medical School", "orcid": "ORCID_TO_FILL"}],
            "license": "cc-by-4.0", "access_right": "open", "doi": DOI,
            "keywords": ["Hi-C", "chromatin architecture", "deep learning", "model checkpoints", "consensus contact map"],
            "related_identifiers": [{"identifier": "https://github.com/MHH-Bioinformatics-Hematology/FlexCible", "relation": "isSupplementTo", "scheme": "url", "resource_type": "software"}],
            "version": "1.0.0", "language": "eng"}}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # --- manifest and limits
    rows, total = [], 0
    for rel, desc in MANIFEST:
        p = out / rel
        if dry:
            continue
        size = p.stat().st_size
        total += size
        rows.append(f'{rel},{size},{sha256(p)},"{desc}"')
    if not dry:
        (out / "manifest.csv").write_text("path,size_bytes,sha256,description\n" + "\n".join(rows) + "\n")
        n = len(rows) + 1
        print(f"{n} files, {total / 1e9:.2f} GB; Zenodo default limits {ZENODO_FILES} files, {ZENODO_GB} GB -> "
              f"{'OK' if n <= ZENODO_FILES and total / 1e9 <= ZENODO_GB else 'OVER LIMIT'}")
    else:
        print(f"dry run: {len(MANIFEST) + 1} files planned")


if __name__ == "__main__":
    main()
