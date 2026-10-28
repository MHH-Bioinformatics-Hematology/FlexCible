"""Run configs of the differential-mark / all-cell-object runs (registry section 66), written for review to
results/diff_marks/configs/ and loaded through Config.from_dict (the training stage of
scripts/multicell_expanded.py writes the same dicts next to each run on the share).

    python scripts/diff_marks_configs.py            # primary runs (diff_loopfocus, folds 10-12) and the 12-cell pilot
    python scripts/diff_marks_configs.py --all      # also the ablations (diff, allobj, diffmeth_loopfocus, diffhspc_loopfocus)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import multicell_expanded as mx  # noqa: E402
from flexcible.config import Config  # noqa: E402

FOLDS = (10, 11, 12)
PRIMARY = [f"all12diff_loopfocus_refres_drop_f{f}" for f in FOLDS]
PILOT = ["all12diffcells_refres_drop", "all12diffcells_loopfocus_refres_drop"]
ABLATIONS = ([f"all12diff_refres_drop_f{f}" for f in FOLDS] + [f"all12allobj_refres_drop_f{f}" for f in FOLDS]
             + [f"all12diffmeth_loopfocus_refres_drop_f{f}" for f in FOLDS]
             + [f"all12diffhspc_loopfocus_refres_drop_f{f}" for f in FOLDS])


def run_names(all_runs: bool = False) -> list[str]:
    bases = PILOT + PRIMARY + (ABLATIONS if all_runs else [])
    return [mx.run_name(b, res) for b in bases for res in (10000, 25000)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    out = REPO / "results/diff_marks/configs"
    out.mkdir(parents=True, exist_ok=True)
    for name in run_names(a.all):
        cfg = mx.run_config(name, 100, 10, 500)
        Config.from_dict(cfg)                          # validates
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(name, cfg["train"]["cache_dir"], flush=True)


if __name__ == "__main__":
    main()
