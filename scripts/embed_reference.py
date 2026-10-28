"""Embed a reference map into a trained reference-map checkpoint.

Copies the map into the checkpoint folder as embedded_reference.cool (with a
.json recording its cells, source, bin size and chromosome sizes) and sets
model.embed_reference in the checkpoint's config.yaml. predict.py and
crosscell_predict.py then predict a NEW cell from its ChIP bigWigs alone: the
reference windows are looked up in the embedded map by chromosome name and
bin, so the bigWigs must be on the map's assembly (hg38 here). Training cells
keep their own leave-one-out reference; the leakage guard refuses the
embedded map for any cell it contains.

    python scripts/embed_reference.py --run runs/multicell_reference/all12refres_25kb \
        --reference crosscell_oe/consensus12_proper_25kb.cool
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config                                  # noqa: E402
from flexcible.reference import embed_reference                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run folder (uses its 'best' pointer)")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--reference", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    ck = run / (run / a.ckpt).read_text().strip()
    cfg = Config.load(ck / "config.yaml")
    if cfg.model.reference == "none":
        raise SystemExit(f"{ck}: not a reference-map model")
    dest = embed_reference(a.reference, ck)
    cfg.model.embed_reference = str(Path(a.reference).resolve())
    cfg.save(ck / "config.yaml")
    print(f"embedded {a.reference} -> {dest}")


if __name__ == "__main__":
    main()
