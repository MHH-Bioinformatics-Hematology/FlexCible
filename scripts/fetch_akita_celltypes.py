"""Find (and optionally download) ENCODE hg38 ChIP tracks for Akita's cell types.

Akita trains on five targets -- HFF and H1hESC Micro-C, GM12878, IMR90 and
HCT116 Hi-C -- but it is sequence-only, so it needs no per-cell-type inputs at
all: one genome feeds five output heads. This model is multiome-conditioned,
so each added cell type needs its own CTCF/cohesin/histone tracks. That is the
extra cost of letting the cell type come from the data instead of the model.

Track set matches what GM12878 already has, so every sample presents identical
rows to the network. Missing assays are reported rather than silently skipped:
a sample with a hole in it must be declared in `missing_tracks` so the absent
value is encoded properly, not left as a misleading zero.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.encodeproject.org/search/"

# the 13 assays GM12878 has, so all samples share one row layout
TARGETS = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3",
           "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3",
           "H4K20me1"]

# ENCODE biosample term names for Akita's other four cell types
CELLS = {
    "h1":     ["H1"],
    "imr90":  ["IMR-90"],
    "hff":    ["HFFc6", "foreskin fibroblast"],
    "hct116": ["HCT116"],
}


def _get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def find_bigwig(cell_terms: list[str], target: str) -> dict | None:
    """Best released hg38 fold-change bigWig for this cell type and target."""
    for term in cell_terms:
        params = [
            ("type", "File"), ("file_format", "bigWig"),
            ("assembly", "GRCh38"), ("status", "released"),
            ("output_type", "fold change over control"),
            ("biosample_ontology.term_name", term),
            ("target.label", target),
            ("limit", "25"), ("format", "json"),
        ]
        url = API + "?" + urllib.parse.urlencode(params)
        try:
            hits = _get(url).get("@graph", [])
        except Exception as exc:                      # noqa: BLE001
            print(f"    ! query failed for {term}/{target}: {exc}", file=sys.stderr)
            continue
        if not hits:
            continue
        # prefer ENCODE's own default pick, then the largest (deepest) file
        hits.sort(key=lambda f: (not f.get("preferred_default", False),
                                 -(f.get("file_size") or 0)))
        f = hits[0]
        return {"accession": f["accession"], "term": term,
                "size_mb": (f.get("file_size") or 0) / 1e6,
                "url": "https://www.encodeproject.org" + f["href"]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["h1", "imr90", "hff"],
                    choices=sorted(CELLS))
    ap.add_argument("--out-root", default=str(Path.home() / "data/hic-multiome"))
    ap.add_argument("--download", action="store_true",
                    help="actually fetch; without this it only reports")
    a = ap.parse_args()

    grand = 0.0
    for cell in a.cells:
        print(f"\n{cell}  (ENCODE terms: {', '.join(CELLS[cell])})")
        outdir = Path(a.out_root) / f"hg38_{cell}"
        rows, missing = [], []
        for t in TARGETS:
            hit = find_bigwig(CELLS[cell], t)
            if hit is None:
                missing.append(t)
                print(f"  {t:10s} MISSING")
            else:
                rows.append((t, hit))
                grand += hit["size_mb"]
                print(f"  {t:10s} {hit['accession']}  {hit['size_mb']:7.1f} MB"
                      f"  [{hit['term']}]")
        print(f"  -> {len(rows)}/{len(TARGETS)} present"
              + (f", missing: {', '.join(missing)}" if missing else ""))

        if a.download and rows:
            outdir.mkdir(parents=True, exist_ok=True)
            with (outdir / "dl.tsv").open("w") as fh:
                for t, hit in rows:
                    fh.write(f"{t}\t{hit['url']}\n")
            (outdir / "missing.json").write_text(json.dumps(missing, indent=2))
            for t, hit in rows:
                dest = outdir / f"{t}.bigwig"
                if dest.exists() and dest.stat().st_size > 0:
                    print(f"  have {dest.name}")
                    continue
                print(f"  fetching {t} ...", flush=True)
                subprocess.run(["curl", "-fsSL", "--retry", "3", "-o",
                                str(dest) + ".part", hit["url"]], check=True)
                Path(str(dest) + ".part").rename(dest)
    print(f"\ntotal to fetch: {grand/1000:.1f} GB")


if __name__ == "__main__":
    main()
