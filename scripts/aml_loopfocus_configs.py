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
from flexcible.config import Config  # noqa: E402


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
    # registry section 70: loopfocus2 (A_refrand controls, weak shared loops, marks margin)
    out2 = REPO / "results/aml_loopfocus/configs2"
    out2.mkdir(parents=True, exist_ok=True)
    names = [mx.run_name(f"all12aml_loopfocus2_refres_drop_f{f}", res) for res in (10000, 25000) for f in (10, 11, 12)]
    names += [mx.run_name(f"all12{k}{o}_refres_drop_f{f}", 10000) for k in ("diff", "diffmeth", "diffhspc")
              for o in ("_loopfocus2", "_loopfocus2aml") for f in (10, 11, 12)]
    for name in names:
        cfg = mx.run_config(name, 100, 10, 500)
        Config.from_dict(cfg)
        (out2 / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(name, cfg["data"]["aml_objects_table"], flush=True)


if __name__ == "__main__":
    main()
