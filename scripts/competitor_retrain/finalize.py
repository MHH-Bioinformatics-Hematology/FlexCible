"""oe_proper.py on the count cools of the pretrained predictions (the conversion of registry
sections 58 and 67): pre_counts/<name>_<chrom>.cool -> runs/benchmark/competitor_oe/<name>_<chrom>.cool.
The retrained predictions are O/E already and are written to competitor_oe by predict_*.py.

    python finalize.py [--names corigami_pre_k562 ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C          # noqa: E402

REPO = HERE.parents[1]
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--chroms", nargs="*", default=C.HELD)
    a = ap.parse_args()
    for f in sorted(C.PRE_COUNTS.glob("*_pre_*.cool")):
        stem = f.stem
        name, chrom = stem.rsplit("_chr", 1)[0], "chr" + stem.rsplit("_chr", 1)[1]
        if (a.names and name not in a.names) or chrom not in a.chroms:
            continue
        dst = C.COMP_OE / f"{name}_{chrom}.cool"
        if dst.exists():
            continue
        res, reach = (C.PIX, 2_097_152) if name.startswith("corigami") else (10000, 990_000)
        cmd = [PY, str(REPO / "scripts/oe_proper.py"), "--input", str(f), "--output", str(dst) + ".tmp",
               "--binsize", str(res), "--chroms", chrom, "--max-dist", str(reach)]
        rc = subprocess.call(cmd, env=dict(__import__("os").environ, OMP_NUM_THREADS="2"))
        if rc == 0:
            Path(str(dst) + ".tmp").rename(dst)
        print(name, chrom, "rc", rc, flush=True)


if __name__ == "__main__":
    main()
