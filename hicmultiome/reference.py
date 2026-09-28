"""Leakage guard for reference contact maps (data.reference_cool,
samples[].reference_cool).

A reference built by scripts/consensus_oe.py records its input cells in
<cool>.json ("cells"). A reference that contains the map of the cell being
trained on or scored would hand the model (or the reference-alone baseline)
the answer, so every place that reads a reference for a cell checks it here.
Cells match by identity: the name up to the first underscore, lower case,
without '-' (hmec, hmec_deep and hmec_rep1 are one cell).
"""
from __future__ import annotations

import json
from pathlib import Path


def cell_identity(name: str) -> str:
    return str(name).lower().replace("-", "").split("_")[0]


def reference_cells(path: str | Path) -> list[str] | None:
    """Cells averaged into a reference map, or None when it has no record."""
    meta = Path(str(path) + ".json")
    if not meta.exists():
        return None
    cells = json.loads(meta.read_text()).get("cells")
    return list(cells) if cells is not None else None


def assert_reference_excludes(path: str | Path, cell: str) -> None:
    """Raise ValueError when the reference at `path` contains `cell`."""
    cells = reference_cells(path)
    if not cells:
        return
    ident = cell_identity(cell)
    hit = [c for c in cells if cell_identity(c) == ident]
    if hit:
        raise ValueError(f"reference leakage: {path} averages {hit}, the same cell "
                         f"as {cell!r}; use a reference that excludes this cell")


# --------------------------------------------------------------------------- #
# embedded reference (model.embed_reference): the map travels with the model
# --------------------------------------------------------------------------- #
EMBED_NAME = "embedded_reference.cool"


def embed_reference(src: str | Path, ckpt: Path) -> Path:
    """Copy the reference map `src` into the checkpoint folder as
    embedded_reference.cool (+ .json with its cell list, source and bin grid),
    once; an existing identical copy is kept."""
    import shutil

    import cooler

    ckpt = Path(ckpt)
    dest = ckpt / EMBED_NAME
    src = Path(src)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return dest
    part = ckpt / (EMBED_NAME + ".part")
    shutil.copyfile(src, part)
    part.rename(dest)
    meta = {}
    if Path(str(src) + ".json").exists():
        meta = json.loads(Path(str(src) + ".json").read_text())
    clr = cooler.Cooler(str(dest))
    meta.update({"embedded_from": str(src), "binsize": int(clr.binsize),
                 "chromsizes": {k: int(v) for k, v in clr.chromsizes.items()},
                 "coordinates": "bins of the source map (hg38 for every map of "
                                "this project); predictions look windows up by "
                                "chromosome name and bin, so the input tracks "
                                "must be on the same assembly"})
    Path(str(dest) + ".json").write_text(json.dumps(meta, indent=1))
    return dest


def resolve_reference(cfg, ckpt: Path, cell: str, override: str | None = None,
                      prefer_embedded: bool = False) -> str:
    """Reference map for predicting `cell` with a reference-map model.

    Order: `override`; the cell's own training reference (samples[].
    reference_cool, a leave-one-out map) unless prefer_embedded; the map
    embedded in the checkpoint; data.reference_cool. The result must not
    contain `cell` (assert_reference_excludes), so the embedded map of all
    training cells serves new cells only."""
    path = override
    if path is None and not prefer_embedded:
        own = next((s for s in cfg.data.samples if s.name == cell), None)
        if own is not None and cfg.sample_reference(own):
            path = cfg.sample_reference(own)
    if path is None and (Path(ckpt) / EMBED_NAME).exists():
        path = str(Path(ckpt) / EMBED_NAME)
    if path is None:
        path = cfg.data.reference_cool
    if not path:
        raise ValueError(f"{cell}: reference-map model and no reference map "
                         "(pass one, or embed one with model.embed_reference)")
    assert_reference_excludes(path, cell)
    return str(path)


def open_reference(path: str, resolution: int, chromsizes: dict | None = None):
    """Cooler of a reference map, checked against the model's bin size and,
    when given, the chromosome sizes of the prediction grid (same assembly)."""
    import cooler

    clr = cooler.Cooler(str(path))
    if clr.binsize != resolution:
        raise ValueError(f"reference {path}: binsize {clr.binsize} != {resolution}")
    if chromsizes:
        rs = {k: int(v) for k, v in clr.chromsizes.items()}
        bad = [c for c, n in chromsizes.items() if c in rs and rs[c] != int(n)]
        if bad:
            raise ValueError(f"reference {path}: chromosome sizes differ from the "
                             f"prediction grid for {bad[:3]} (other assembly?)")
    return clr


def reference_block(clr, chrom: str, start: int, size: int, mat_norm):
    """Encoded (size, size) reference window at bins [start, start + size) of
    `chrom` (0 = expected where the map has no such chromosome)."""
    import numpy as np

    from .genome import match_name
    from .matrix import fetch_diagonal_window

    name = match_name(chrom, list(clr.chromnames))
    if name is None:
        return np.zeros((size, size), np.float32)
    return mat_norm.encode(fetch_diagonal_window(clr, name, start, size))
