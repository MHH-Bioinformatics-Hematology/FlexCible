"""Interactive prediction server: measured vs predicted, reference vs edited.

Endpoints
  GET  /                     the UI
  GET  /api/config           samples, tracks and knockout-able assays
  POST /api/predict          {region, edits[], knockouts[]} -> matrices + stats

A prediction returns both triangles of one square: the upper triangle is the
comparison baseline (measured Hi-C, or the reference-sequence prediction) and
the lower is the prediction for the requested edit/knockout, which is the
layout `flexcible.combine` writes for HiGlass. The response also carries the
edited-coordinate bookkeeping so the UI can draw the inserted or deleted
stretch as a gap on the unedited side.

Run:  uvicorn webapp.server:app --host 127.0.0.1 --port 8055
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexcible.config import Config          # noqa: E402
from flexcible.matrix import fetch_chrom_matrix  # noqa: E402
from flexcible.utils import get_logger        # noqa: E402

log = get_logger("webapp")
app = FastAPI(title="Hi-C prediction explorer")

CONFIG = ROOT / "configs/multicell_25kb.yaml"
_cache: dict = {}


def cfg() -> Config:
    if "cfg" not in _cache:
        _cache["cfg"] = Config.load(CONFIG)
    return _cache["cfg"]


class EditIn(BaseModel):
    chrom: str
    start: int
    end: int
    kind: Literal["del", "dup", "ins", "sub"] = "del"
    sequence: str = ""


class RegionalKO(BaseModel):
    """Knock an assay out only inside an interval, e.g. no H3K27ac on
    chr1:10-50 Mb, leaving it intact everywhere else."""

    track: str
    chrom: str
    start: int
    end: int


class PredictIn(BaseModel):
    region: str                       # chrom:start-end
    sample: str | None = None
    edits: list[EditIn] = []
    knockouts: list[str] = []         # track names to zero genome-wide
    regional_knockouts: list[RegionalKO] = []
    baseline: Literal["measured", "reference"] = "measured"


def _parse_region(region: str):
    chrom, span = region.replace(",", "").split(":")
    s, e = (int(x) for x in span.split("-"))
    return chrom, s, e


@app.get("/api/config")
def api_config():
    c = cfg()
    return {
        "config": str(CONFIG.name),
        "resolution": c.data.resolution,
        "samples": [s.name for s in c.data.samples],
        "tracks": [{"name": t.name, "type": t.type, "motif": t.motif}
                   for t in c.data.tracks],
        "knockoutable": [t.name for t in c.data.tracks],
        "window_bp": c.window.size * c.data.resolution,
    }


@app.post("/api/predict")
def api_predict(req: PredictIn):
    from flexcible.combine import combine_matrices
    from flexcible.predict import predict as run_predict

    c = cfg()
    chrom, s, e = _parse_region(req.region)
    res = c.data.resolution
    if (e - s) // res < 8:
        raise HTTPException(400, "region too small for this resolution")
    out_dir = ROOT / "runs/webapp"
    out_dir.mkdir(parents=True, exist_ok=True)
    rk = "_".join(f"{k.track}@{k.start}" for k in req.regional_knockouts)
    tag = f"{chrom}_{s}_{e}_{'-'.join(req.knockouts) or 'wt'}{'_' + rk if rk else ''}"

    # lower triangle: the perturbed prediction (knockouts and/or edits)
    pred_cool = str(out_dir / f"pred_{tag}.cool")
    zr = [(k.track, k.chrom, k.start, k.end) for k in req.regional_knockouts]
    run_predict(c, chrom, pred_cool, batch=8, sample_name=req.sample,
                zero_tracks=req.knockouts or None, zero_regions=zr or None)
    lower = fetch_chrom_matrix(pred_cool, chrom)

    # upper triangle: measured Hi-C, or the unperturbed prediction
    if req.baseline == "measured":
        upper = fetch_chrom_matrix(c.get_sample(req.sample).cool_path, chrom)
    else:
        ref_cool = str(out_dir / f"ref_{chrom}_{s}_{e}.cool")
        if not Path(ref_cool).exists():
            run_predict(c, chrom, ref_cool, batch=8, sample_name=req.sample)
        upper = fetch_chrom_matrix(ref_cool, chrom)

    b0, b1 = s // res, min(e // res, min(upper.shape[0], lower.shape[0]))
    up, lo = upper[b0:b1, b0:b1], lower[b0:b1, b0:b1]
    comb = combine_matrices(np.log1p(np.maximum(up, 0)),
                            np.log1p(np.maximum(lo, 0)))
    gaps = [{"start": ed.start, "end": ed.end, "kind": ed.kind}
            for ed in req.edits]
    return {
        "region": {"chrom": chrom, "start": b0 * res, "end": b1 * res,
                   "bins": int(b1 - b0), "resolution": res},
        "matrix": np.round(comb, 4).tolist(),
        "vmax": float(np.percentile(comb[comb > 0], 99)) if (comb > 0).any() else 1.0,
        "upper_label": "measured" if req.baseline == "measured"
                       else "reference prediction",
        "lower_label": ("prediction"
                        + (f" | KO: {', '.join(req.knockouts)}" if req.knockouts else "")
                        + (f" | {len(req.edits)} edit(s)" if req.edits else "")
                        + (f" | regional KO: {', '.join(k.track for k in req.regional_knockouts)}"
                           if req.regional_knockouts else "")),
        "edit_gaps": gaps,
        "ko_regions": [{"track": k.track, "start": k.start, "end": k.end}
                       for k in req.regional_knockouts],
        "knockouts": req.knockouts,
    }


@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "static/index.html"))


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")),
          name="static")
