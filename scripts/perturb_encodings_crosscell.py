"""Encodings of CTCF removal in the pipeline of registry sections 95 and 99 (registry section 100).

The release checkpoint all12one_fullrd (views at 10 and 25 kb) predicts one window centred on each of the 413 Akita test
regions from the HCT116 ChIP (scripts/crosscell_predict.py, TTA), without the consensus (plane at 0, models *_enc_noref_*; *_encu_* = ChIP of the untreated degron cells instead of the ENCODE set) and
with the leave-one-out consensus (*_enc_cons_*). Keep sets (scripts/perturb_validate.py perts_of): full; minus_CTCF (the
CTCF track absent = raw zero, the same normalised input: TrackNormalizer.absent_values is the value of a raw-zero track);
minus_CTCF_nomotif (and the CTCF motif rows MA0139.2 fwd and rev at the raw-zero level); minus_nomotif (motif rows only);
minus_H3K36me3 (control). Scoring is that of perturb_validate (same regions, loops, boundaries) for the CTCF-AID experiment.

    python scripts/perturb_encodings_crosscell.py --stage predict score      # predict needs GPU 1 (PERT_GPU=1)
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import perturb_validate as pv                                        # noqa: E402

MODELS = [f"all12one_fullrd_{e}_{u}_{k}kb" for e in ("enc", "encu", "enck", "encc", "encd") for k in (10, 25) for u in ("noref", "cons")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    a = ap.parse_args()
    pv._ONLY.update(MODELS)
    for st in a.stage:
        if st == "predict":
            pv.stage_predict()
        elif st == "score":
            pv.RES_OUT = REPO / "results/one_model/ctcf_encoding"
            pv.EXPERIMENTS = {"ctcf": pv.EXPERIMENTS["ctcf"]}
            pv.stage_score(MODELS, "crosscell")


if __name__ == "__main__":
    main()
