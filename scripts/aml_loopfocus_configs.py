"""Run configs of the loop-centric AML recipe (registry section 63) for folds 10, 11, 12 at 10 and
25 kb, written for review to results/aml_loopfocus/configs/ and loaded through Config.from_dict (the
training stage of scripts/multicell_expanded.py writes the same dicts next to the run on the share).

    python scripts/aml_loopfocus_configs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import multicell_expanded as mx  # noqa: E402
from hicmultiome.config import Config  # noqa: E402


def main():
    out = REPO / "results/aml_loopfocus/configs"
    out.mkdir(parents=True, exist_ok=True)
    for res in (10000, 25000):
        for f in (10, 11, 12):
            name = mx.run_name(f"all12aml_loopfocus_refres_drop_f{f}", res)
            cfg = mx.run_config(name, 100, 10, 500)
            Config.from_dict(cfg)                      # validates
            (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(name, cfg["train"]["cache_dir"], flush=True)


if __name__ == "__main__":
    main()
