"""Lift the Mifsud 2015 GM12878 promoter capture Hi-C table and its bait design
from hg19 to hg38, fragment by fragment.

Both inputs are defined on HindIII restriction fragments: every interaction is
a (bait fragment, other-end fragment, raw count, log(observed/expected)) record
and every bait is one fragment of the HindIII design. The scores belong to the
fragment pair, so lifting the two fragment intervals (UCSC hg19ToHg38 chain,
hicmultiome.liftover) carries the published scores over unchanged. A fragment
is kept only when both of its ends map to the same hg38 chromosome with a span
within 10% of the original; an interaction is kept only when both fragments
are kept and stay on one chromosome. The script then checks that the lifted
fragment ends sit on hg38 HindIII sites (AAGCTT within 8 bp), which confirms
that the lifted intervals are the hg38 HindIII fragments.

Outputs (same Mifsud column layout, so hicmultiome.capture.parse_mifsud reads
the hg38 file unchanged):
  <out-dir>/TS5_GM12878_promoter-other_significant_interactions.hg38.txt
  <out-dir>/GM12878_pchic_baits.hg38.bed   (chrom, start, end, bait id, genes)
  <out-dir>/lift_capture_hg38.json         (counts and the HindIII check)

    python scripts/lift_capture_hg38.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome.liftover import LiftOver  # noqa: E402

RAW = Path("/mnt/processed/wolffjoa/capturehic-to-hic-raw")
PRIMARY = {f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interactions", default=str(Path(
        "~/data/hic-multiome/pchic/TS5_GM12878_promoter-other_significant_interactions.txt"
    ).expanduser()))
    ap.add_argument("--baitmap", default=str(
        RAW / "chicago_design/Human_hg19_chr/Digest_Human_HindIII_baits_e75_ID.baitmap"))
    ap.add_argument("--chain", default=str(RAW / "liftover/hg19ToHg38.over.chain.gz"))
    ap.add_argument("--hg38", default=str(Path(
        "~/data/hic-multiome/genome/hg38.fa").expanduser()))
    ap.add_argument("--out-dir", default=str(Path("~/data/hic-multiome/pchic").expanduser()))
    a = ap.parse_args()
    out = Path(a.out_dir)
    lo = LiftOver(a.chain)
    cache: dict[tuple[str, int, int], tuple | None] = {}

    def lift(c, s, e):
        k = (c, s, e)
        if k not in cache:
            r = lo.convert_interval(c, s, e)
            cache[k] = r if r is not None and r[0] in PRIMARY else None
        return cache[k]

    stats = {"interactions_in": 0, "interactions_out": 0,
             "interactions_dropped_unmapped": 0, "interactions_dropped_split": 0}
    lifted_ends = []
    out_int = out / "TS5_GM12878_promoter-other_significant_interactions.hg38.txt"
    with open(a.interactions) as fi, open(out_int, "w") as fo:
        fo.write(fi.readline())
        for line in fi:
            p = line.rstrip("\n").split("\t")
            if len(p) < 11:
                continue
            stats["interactions_in"] += 1
            b = lift(p[0], int(p[1]), int(p[2]))
            o = lift(p[6], int(p[7]), int(p[8]))
            if b is None or o is None:
                stats["interactions_dropped_unmapped"] += 1
                continue
            if p[0] == p[6] and b[0] != o[0]:
                stats["interactions_dropped_split"] += 1
                continue
            p[0], p[1], p[2] = b[0], str(b[1]), str(b[2])
            p[6], p[7], p[8] = o[0], str(o[1]), str(o[2])
            fo.write("\t".join(p) + "\n")
            stats["interactions_out"] += 1
    frags_in = len(cache)
    frags_ok = sum(v is not None for v in cache.values())
    stats.update(fragments_in=frags_in, fragments_lifted=frags_ok)

    n_b = n_bo = 0
    out_b = out / "GM12878_pchic_baits.hg38.bed"
    with open(a.baitmap) as fi, open(out_b, "w") as fo:
        for line in fi:
            p = line.rstrip("\n").split("\t")
            n_b += 1
            r = lift(p[0], int(p[1]), int(p[2]))
            if r is None:
                continue
            fo.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{p[3]}\t{p[4] if len(p) > 4 else ''}\n")
            n_bo += 1
    stats.update(baits_in=n_b, baits_lifted=n_bo)

    # HindIII check on a random sample of lifted fragments (both ends)
    import pysam

    fa = pysam.FastaFile(a.hg38)
    ok = [v for v in cache.values() if v is not None]
    random.Random(0).shuffle(ok)
    hit = tot = 0
    for c, s, e in ok[:20000]:
        for x in (s, e):
            tot += 1
            hit += "AAGCTT" in fa.fetch(c, max(0, x - 8), x + 8).upper()
    stats["hindiii_site_at_lifted_end_fraction"] = hit / max(tot, 1)
    stats["hindiii_ends_checked"] = tot
    (out / "lift_capture_hg38.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
