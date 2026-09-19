"""Turn the Galaxy 1 kb cool matrices into the resolution ladder to score on.

hicBuildMatrix is asked for 1 kb bins only. Every coarser resolution is made
here by summing 1 kb bins, which is the same matrix hicBuildMatrix would have
produced at that bin size -- the read filtering it applies does not depend on
bin size -- so the coarse rungs cost nothing on the server.

Pooling for the depth series works the same way: summing the contact matrices
of k replicates is exactly pooling their reads, so `cooler merge` replaces
re-mapping and re-merging BAMs.

Balancing (ICE) is applied per resolution because the ceiling script reads
balanced counts, and an unbalanced matrix would make the comparison depend on
per-bin coverage differences between replicates rather than on structure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gxhic  # noqa: E402

RESOLUTIONS = [1000, 2000, 5000, 10000, 25000, 40000]
OUT = Path(__file__).parent.parent / "runs" / "hic_matrices"
COOLER = str(Path(sys.executable).parent / "cooler")


def sh(*cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def fetch(label: str, hda_id: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{label}.1000.cool"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {label}: already downloaded ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    n = gxhic.download(hda_id, str(dest))
    print(f"  {label}: {n/1e6:.0f} MB")
    return dest


def merge(label: str, parts: list[Path]) -> Path:
    dest = OUT / f"{label}.1000.cool"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".part")
    sh(COOLER, "merge", tmp, *parts)
    tmp.rename(dest)
    return dest


def zoomify(cool: Path, nproc: int = 8) -> Path:
    dest = cool.with_name(cool.name.replace(".1000.cool", ".mcool"))
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {dest.name}: already built")
        return dest
    tmp = dest.with_suffix(".mcool.part")
    sh(COOLER, "zoomify", "--balance", "--nproc", nproc,
       "--resolutions", ",".join(str(r) for r in RESOLUTIONS),
       "--out", tmp, cool)
    tmp.rename(dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", nargs="*", default=[],
                    help="label=hda_id, downloads the 1 kb cool")
    ap.add_argument("--pool", nargs="*", default=[],
                    help="label=memberA+memberB+... sums 1 kb cools")
    ap.add_argument("--zoomify", nargs="*", default=[],
                    help="labels to coarsen+balance (default: all touched)")
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()

    touched = []
    for spec in a.fetch:
        label, hda = spec.split("=", 1)
        fetch(label, hda)
        touched.append(label)
    for spec in a.pool:
        label, members = spec.split("=", 1)
        parts = [OUT / f"{m}.1000.cool" for m in members.split("+")]
        missing = [p for p in parts if not p.exists()]
        if missing:
            print(f"  {label}: SKIPPED, missing {[p.name for p in missing]}")
            continue
        merge(label, parts)
        touched.append(label)

    for label in (a.zoomify or touched):
        cool = OUT / f"{label}.1000.cool"
        if cool.exists():
            zoomify(cool, a.nproc)

    print(json.dumps(sorted(p.name for p in OUT.glob("*.mcool")), indent=1))


if __name__ == "__main__":
    main()
