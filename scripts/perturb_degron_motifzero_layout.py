"""Registry section 100: the encoding "CTCF signal zero + motif rows zero" (zero_CTCF_motif of the encodings set) in the layout that
figures/scripts/fig_perturb_degron.py of the manuscript reads for results/perturb_degron_zero/all12one_fullrd_noref.

Writes results/perturb_degron_zero/all12one_fullrd_{noref,cons}_motifzero/{change_per_region.csv.gz, loops_per_anchor.csv.gz,
boundaries_per_bin.csv.gz, change_per_distance.csv, perturb_summary.csv} with model all12one_fullrd_{noref,cons}_motifzero_<k>kb and
the perturbation names of that script (zero_CTCF = CTCF signal zero + motif rows zero, zero_H3K36me3 = control, full), and copies
the cool files into /mnt/processed/wolffjoa/hic-multiome/perturb/degron_zero/cool/all12one_fullrd_{noref,cons}_motifzero_<k>kb/
{full,zero_CTCF}.cool (copies of the predictions of the encodings set). The figure is switched by
RELEASE-model name and result folder: MODEL = "all12one_fullrd_noref_motifzero_10kb", R = .../all12one_fullrd_noref_motifzero.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
E = REPO / "results/one_model/ctcf_encoding"
COOL = Path("/mnt/processed/wolffjoa/hic-multiome/perturb/degron_zero/cool")
VARIANTS = {"motifzero": ("zero_CTCF_motif", "CTCF signal zero + CTCF motif rows zero", "the raw CTCF signal and the rows motif:MA0139.2:fwd and :rev"),
            "cohabsent": ("zero_CTCF_RAD21", "CTCF and RAD21 signal zero (SMC3 is absent in this ChIP set)", "the raw CTCF and RAD21 signals")}
VAR = sys.argv[1] if len(sys.argv) > 1 else "motifzero"
SRCPERT, VTITLE, VDESC = VARIANTS[VAR]
MAP = {"full": "full", SRCPERT: "zero_CTCF", "zero_H3K36me3": "zero_H3K36me3"}
COOLS_COPIED = ("full", SRCPERT)        # the cool files the figure reads (the share has no symbolic links)
TEXT = {"noref": "no consensus (plane at 0)", "cons": "consensus (leave-one-out HCT116 map)"}

for use in ("noref", "cons"):
    out = REPO / f"results/perturb_degron_zero/all12one_fullrd_{use}_{VAR}"
    out.mkdir(parents=True, exist_ok=True)
    for src, dst, kind in (("change_per_region.csv.gz", "change_per_region.csv.gz", "csv"),
                           ("loops_per_anchor.csv.gz", "loops_per_anchor.csv.gz", "csv"),
                           ("boundaries_per_bin.csv.gz", "boundaries_per_bin.csv.gz", "csv"),
                           ("change_per_distance.csv", "change_per_distance.csv", "csv"),
                           ("perturb_summary.csv", "perturb_summary.csv", "csv")):
        d = pd.read_csv(E / src)
        d = d[d.model.str.contains(f"fullrd_{use}_")].copy()
        d = d[d.perturbation.isin(MAP)]
        d["encoding"] = d.perturbation
        d["perturbation"] = d.perturbation.map(MAP)
        d["model"] = d.model.str.replace(f"fullrd_{use}_", f"fullrd_{use}_{VAR}_", regex=False)
        d.to_csv(out / dst, index=False, float_format="%.5f")
    for k in (10, 25):
        sd = COOL / f"all12one_fullrd_{use}_{k}kb"
        dd = COOL / f"all12one_fullrd_{use}_{VAR}_{k}kb"
        dd.mkdir(parents=True, exist_ok=True)
        for a in COOLS_COPIED:
            dst = dd / f"{MAP[a]}.cool"
            if (sd / f"{a}.cool").exists() and not dst.exists():
                shutil.copy2(sd / f"{a}.cool", dst)
    (out / "README.md").write_text(
        f"# {VTITLE}, release checkpoint, {TEXT[use]} (2026-11-01)\n\n"
        "Registry section 100. The layout of `results/perturb_degron_zero/all12one_fullrd_noref`, so that "
        "`figures/scripts/fig_perturb_degron.py` switches by `MODEL` and the result folder. "
        f"`perturbation` zero_CTCF here means {VDESC} set to zero on chr1-chr21 "
        f"(encoding {SRCPERT} of `scripts/perturb_degron_zero.py --set encodings`), zero_H3K36me3 is the unchanged control, full the "
        "unperturbed prediction; the column `encoding` holds the name in `results/one_model/ctcf_encoding`. "
        f"Models all12one_fullrd_{use}_{VAR}_{{10,25}}kb. Cool files: `{COOL}/all12one_fullrd_{use}_{VAR}_<k>kb/"
        "{full,zero_CTCF}.cool` (copies of the predictions of `scripts/perturb_degron_zero.py --set encodings`). "
        "Input: the ChIP of the untreated degron cells (config sample hct116), as in the folder all12one_fullrd_noref.\n")
    print("wrote", out)
