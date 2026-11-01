"""Measured maps of every other cell type, and the consensus, scored on the pixels of the
published-predictor blocks (registry section 103). Only released predictions of the published
methods are used; the retrained C.Origami, Epiphany and Akita (sections 77, 79) are not part of this analysis.

For every block of the benchmark tables in which a published predictor was scored (released
predictions in their own cells at 25, 10 and 5 kb on the held-out chromosomes; C.Origami and
the C.Origami de novo block; Akita on the 413 regions is in motivation_akita413.py), the block machinery of scripts/hic_measures.py (score_block, unchanged)
is run with the sources

  the competitor(s) of the block          (stored predictions, never rerun)
  consensus                               (leave-one-out mean map of the other training cells,
                                           the 12-cell consensus for K562, HMEC, HUVEC)
  other_<cell>                            (the measured O/E map crosscell_oe/<cell>_proper_<k>kb.cool
                                           of EVERY other cell that has a map at this bin size)

so that all sources are scored on the same pixels (finite in the target and in every source),
with the same measures as the tables of sections 37, 58, 73, 77, 79 and 81. The models of
the chrho tables are not sources here (they are not part of the question); the pixel set can
therefore differ from the older blocks by the pixels that only a model left out. The scored
competitor rows are compared with the stored rows by the report stage.

Differences to the standard score_block call (cost only, no definition changes): ENT3C and the
3DChromatin measures are not run (slow_tools False), hicDetectLoops is not run (the rank-matched
loop recall of the tables is a native measure).

    python scripts/motivation_other_cells.py --family released25 --jobs 4
    families: released (25/10 kb), released5, denovo
Blocks are written to <root>/blocks/<id>__motiv/measures.csv (__motivnat with --native-only).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("BENCH_5KB", "1")
os.environ.setdefault("BENCH_HICGAN_IMR90_25KB", "1")
os.environ.setdefault("BENCH_CORIGAMI_DENOVO", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import hic_measures as H  # noqa: E402

ALL_CELLS = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
             "gm23248", "mcf7", "cd14mono", "pc9", "k562", "hmec_deep", "huvec_deep"]
ROOT = H.HM / "runs/motivation"
TAG = "__motiv"


def other_maps(cell: str, res: int) -> list[dict]:
    out = []
    for o in ALL_CELLS:
        p = H.target_cool(o, res)
        if o != cell and p.exists():
            out.append(dict(name=f"other_{o}", kind="cool", path=p, flag=f"other:{o}"))
    return out


def source_list(b: dict) -> list[dict]:
    base = H.block_sources(b["set"], b["cell"], b["res"], b["comps"], models=[])
    keep = [s for s in base if s["kind"] == "competitor" or s["name"] == "consensus"]
    return keep + other_maps(b["cell"], b["res"])


def setup() -> None:
    (ROOT / "blocks").mkdir(parents=True, exist_ok=True)
    H.OUT = ROOT
    _exp = H.expected_file

    def expected_file(cell, res, chrom, D=200):          # cache of the balanced expected curves, shared
        old = H.OUT
        H.OUT = H.CHRHO_ROOT
        try:
            return _exp(cell, res, chrom, D)
        finally:
            H.OUT = old
    H.expected_file = expected_file
    H.block_source_list = source_list
    H.slow_tools = lambda b: False
    H.run_detectloops = lambda *a, **k: None


def family_blocks(fam: str) -> list[dict]:
    if fam == "released":
        H.CHRHO_25KB_COMPS = True
        bl = H.chrho_blocks()
    elif fam == "released5":
        H.CHRHO_RES = 5000
        bl = H.chrho_blocks()
    elif fam == "hicreg10":
        # the released HiC-Reg variants at 10 kb (chr14, chr17; band 1 Mb), the three released names of
        # the retrained-pipeline blocks of registry 63; the retrained variants are not used
        H.CHRHO_HICREG_RE = True
        bl = [b for b in H.chrho_blocks() if b["res"] == 10000 and b["set"] == "chrho_hicregre"
              and b["chrom"] in ("chr14", "chr17")]
        for b in bl:
            b["comps"] = [c for c in b["comps"] if c.startswith(("hicregwindowcrosschrom", "hicregmulticellcrosschrom",
                                                                 "hicregwindowcv_"))]
            b["set"] = "chrho_hicreg"
            b["id"] = f"chrho_hicreg_{b['cell']}_10kb_{b['chrom']}"
    elif fam == "denovo":
        H.CHRHO_CO_DENOVO = True
        bl = H.chrho_blocks()
    else:
        raise SystemExit(fam)
    for b in bl:
        b["id"] = b["id"] + TAG
    return bl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True,
                    choices=["released", "released5", "denovo", "hicreg10"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="regexes of block ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--native-only", action="store_true",
                    help="native measures only (Pearson, SCC, insulation r, rank-matched loop recall, "
                         "compartment preference); blocks get the suffix __motivnat")
    a = ap.parse_args()
    import hicreg_audit
    hicreg_audit.register()
    setup()
    global TAG
    if a.native_only:
        H.NATIVE_ONLY = True
        TAG = "__motivnat"
    bl = family_blocks(a.family)
    if a.only:
        bl = [b for b in bl if any(re.search(o, b["id"]) for o in a.only)]
    if a.list:
        for b in bl:
            print(b["id"], b["comps"])
        return
    for b in bl:
        f = H.OUT / "blocks" / b["id"] / "measures.csv"
        if f.exists() or not H.claim(f):
            continue
        try:
            H.score_block(b, a.jobs)
        except Exception as e:                       # noqa: BLE001
            import traceback
            H.log(f"[motivation] FAILED {b['id']}: {e!r}\n{traceback.format_exc()}")
        finally:
            H.release(f)


if __name__ == "__main__":
    main()
