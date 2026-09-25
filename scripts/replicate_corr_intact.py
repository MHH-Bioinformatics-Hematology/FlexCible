"""Replicate A vs replicate B for the deep intact Hi-C pairs (ENCODE, Aiden lab).

Same measure as scripts/replicate_corr.py (the single-lane in situ table):
median per-region Pearson on ICE-balanced proper O/E over the 413 Akita test
regions, a region needs >= 20 bins, at 2, 5, 10 and 25 kb. Maps come from
scripts/encode_expand.py (crosscell_oe/intact_<cell>_<exp>_<rep>_ice_<res>kb.cool);
contacts per replicate are the pixel sums of the converted .hic at 25 kb
(encode_hic/stats) and the Juicer header count.

    python scripts/replicate_corr_intact.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_like_hicgan as sc                       # noqa: E402
from encode_expand import ENC, OE, PAIRS             # noqa: E402
from replicate_corr import region_rs                 # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/replicates/replicate_corr_intact_deep.csv"
RES = [2, 5, 10, 25]


def stats(label):
    p = ENC / "stats" / f"{label}_25kb.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    regions = pd.read_csv(sc.REGIONS, sep="\t", names=["chrom", "start", "end"])
    rows = []
    for cell, exp, (ra, acc_a), (rb, acc_b) in PAIRS:
        la, lb = f"{cell}_{exp}_{ra}", f"{cell}_{exp}_{rb}"
        sa, sb = stats(la), stats(lb)
        rec = {"cell": cell, "experiment": exp, "assay": "intact Hi-C",
               "repA": f"{ra} {acc_a}", "repB": f"{rb} {acc_b}",
               "repA_Mcontacts": round(sa.get("contacts", float("nan")) / 1e6, 1),
               "repB_Mcontacts": round(sb.get("contacts", float("nan")) / 1e6, 1),
               "repA_header_Mcontacts": round((sa.get("header_contacts") or [np.nan])[0] / 1e6, 1),
               "repB_header_Mcontacts": round((sb.get("header_contacts") or [np.nan])[0] / 1e6, 1)}
        for res in RES:
            pa = OE / f"intact_{la}_ice_{res}kb.cool"
            pb = OE / f"intact_{lb}_ice_{res}kb.cool"
            if pa.exists() and pb.exists():
                rs = region_rs(cooler.Cooler(str(pa)), cooler.Cooler(str(pb)), regions)
                rec[f"{res}kb"] = round(float(np.median(rs)), 3) if len(rs) else float("nan")
                rec[f"{res}kb_n"] = len(rs)
        rows.append(rec)
        print(rec, flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    df.to_csv(tmp, index=False)
    tmp.rename(OUT)
    print("replicate A vs B, deep intact Hi-C, median per-region Pearson on "
          "ICE-balanced proper O/E (413 Akita regions)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
