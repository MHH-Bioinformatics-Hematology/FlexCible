"""Per-cell CpG files for the methylation-aware CTCF motif planes
(motifs.motif_meth, hicmultiome.motifmeth; registry section 43).

The planes read <HM>/meth/<cell>/CpG.npz next to the cell's Meth.bigwig (a
sample's extra_bigwig_dirs). The file is the CpG-level methylation table that
scripts/chip_seq_meth_relationships.py builds (--stage cpg, ENCODE WGBS beds:
<chrom>_pos, <chrom>_cov, <chrom>_meth), copied here so that the training
config finds it through the folder it already passes.

    python scripts/motif_meth.py --stage cpg      # copy the CpG tables into the meth folders
    python scripts/motif_meth.py --stage check    # list cells with and without a CpG.npz
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402
from hicmultiome.motifmeth import CPG_FILE                          # noqa: E402

SRC = mx.HM / "runs/chip_meth_rel/cpg"


def stage_cpg():
    for cell in me.FILES:
        src, dst = SRC / f"{cell}.npz", me.meth_dir(cell) / CPG_FILE
        if dst.exists():
            print(f"{cell}: {dst} exists")
            continue
        if not src.exists():
            print(f"{cell}: {src} missing; run chip_seq_meth_relationships.py --stage cpg")
            continue
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, tmp)
        os.rename(tmp, dst)
        print(f"{cell}: copied {src.stat().st_size / 1e6:.0f} MB")


def stage_check():
    for cell in mx.CELLS:
        d = me.meth_dir(cell)
        ok = d is not None and (d / CPG_FILE).exists()
        print(f"{cell:10s} Meth={'yes' if me.has_meth(cell) else 'no ':3s} CpG.npz={'yes' if ok else 'no'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", nargs="+", choices=["cpg", "check"], required=True)
    a = ap.parse_args()
    if "cpg" in a.stage:
        stage_cpg()
    if "check" in a.stage:
        stage_check()


if __name__ == "__main__":
    main()
