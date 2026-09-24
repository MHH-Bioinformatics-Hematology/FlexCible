"""Build a 2 kb KR-balanced cooler from a (possibly remote) .hic via straw.

Avoids downloading the whole multi-resolution .hic: hicstraw random-accesses the
2 kb KR-normalized intra-chromosomal records directly from the ENCODE URL. Keeps
only the <=2 Mb band (all the model/O-E pipeline uses). The KR values are written
as 'count' with NO weight column, so oe_proper.py treats them as already balanced.

    python scripts/straw_to_cool.py --hic <url|path> --output out.cool \
        --binsize 2000 --band-mb 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cooler
import hicstraw
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hic", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--binsize", type=int, default=2000)
    ap.add_argument("--band-mb", type=float, default=2.0)
    ap.add_argument("--norm", default="KR")
    ap.add_argument("--chroms", nargs="*", default=None)
    a = ap.parse_args()

    hic = hicstraw.HiCFile(a.hic)
    chrom_objs = [c for c in hic.getChromosomes() if c.name.lower() != "all"]
    want = a.chroms or [f"chr{i}" for i in list(range(1, 23)) + ["X"]]
    def norm_name(n): return n if n.startswith("chr") else f"chr{n}"
    keep = [c for c in chrom_objs if norm_name(c.name) in want]

    # cooler bins over the chroms we keep, in hic order
    binsize = a.binsize
    band = int(a.band_mb * 1_000_000)
    bin_rows, offsets, cum = [], {}, 0
    for c in keep:
        nb = int(np.ceil(c.length / binsize))
        starts = np.arange(nb) * binsize
        ends = np.minimum(starts + binsize, c.length)
        bin_rows.append(pd.DataFrame({"chrom": norm_name(c.name), "start": starts, "end": ends}))
        offsets[c.name] = cum
        cum += nb
    bins = pd.concat(bin_rows, ignore_index=True)

    def pixels():
        for c in keep:
            off = offsets[c.name]
            mzd = hic.getMatrixZoomData(c.name, c.name, "observed", a.norm, "BP", binsize)
            recs = mzd.getRecords(0, c.length, 0, c.length)   # whole chrom, sparse
            if not recs:
                print(f"  {c.name}: 0 records", flush=True); continue
            bx = np.fromiter((r.binX for r in recs), np.int64, len(recs)) // binsize
            by = np.fromiter((r.binY for r in recs), np.int64, len(recs)) // binsize
            cnt = np.fromiter((r.counts for r in recs), np.float64, len(recs))
            i = np.minimum(bx, by); j = np.maximum(bx, by)
            d = (j - i) * binsize
            m = np.isfinite(cnt) & (cnt > 0) & (d <= band)
            i, j, cnt = i[m], j[m], cnt[m]
            order = np.lexsort((j, i))                        # sort by bin1 then bin2
            i, j, cnt = i[order], j[order], cnt[order]
            print(f"  {c.name}: {m.sum():,} banded pixels", flush=True)
            yield pd.DataFrame({"bin1_id": off + i, "bin2_id": off + j, "count": cnt.astype(np.float32)})

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(a.output, bins=bins, pixels=pixels(),
                         dtypes={"count": np.float32}, ordered=False)
    print("wrote", a.output, flush=True)


if __name__ == "__main__":
    main()
