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
