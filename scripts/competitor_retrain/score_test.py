"""Registry 77b smoke test: score the one-epoch C.Origami of a 5 or 2 kb bin on held-out blocks with the
identical-pixel scoring of hic_measures (set chrho_rtco). COMPRT_BIN=5000|2000 selects the grid.

    COMPRT_BIN=5000 python score_test.py gm12878:chr14 imr90:chr2
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import blocks as rb                       # noqa: E402
import hic_measures as hm                 # noqa: E402

assert rb.TEST_BIN, "set COMPRT_BIN=5000 or 2000"
hm.CHRHO_COMPRT = True
hm.chrho_use_root()
rb.register()
want = set(sys.argv[1:])
for b in hm._finish_blocks(rb.rows()):
    if f"{b['cell']}:{b['chrom']}" in want:
        print(b["id"], b["D"], flush=True)
        print(hm.score_block(b, 4), flush=True)
