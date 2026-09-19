"""fastp (paired) -> minibwa (per mate) on Galaxy, for local matrix building.

Matrix building moved off Galaxy because __hicexplorer@3.7.6 there has a
truncated libgomp.so.1 and cannot be repaired remotely. Trimming and mapping
stay on Galaxy's compute; the BAMs are pulled down and hicBuildMatrix runs
locally.

WHY fastp RUNS PAIRED AND MAPPING RUNS PER MATE
hicBuildMatrix pairs the two SAM files BY POSITION, not by read name. fastp
must therefore see both mates in one job (paired collection mode), because it
then discards BOTH mates whenever either fails a filter and the two outputs
stay in lockstep. Trimming each mate in a separate job would drop different
reads from each file, silently breaking the 1:1 correspondence -- the matrix
would still build, and would be wrong.

Mapping is the opposite: each trimmed mate is mapped INDEPENDENTLY, because
hicBuildMatrix classifies ligation junctions itself and a paired aligner's
insert-size model does not hold for Hi-C. Nothing is ever coordinate-sorted.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gxhic  # noqa: E402

ROOT = Path(__file__).parent.parent
STATE = ROOT / "runs" / "gx_trimmap_state.json"
BAMDIR = Path("/mnt/processed/wolffjoa/hic-multiome/local/bam")

FASTP = "toolshed.g2.bx.psu.edu/repos/iuc/fastp/fastp/0.24.3+galaxy0"
MINIBWA = "toolshed.g2.bx.psu.edu/repos/iuc/minibwa/minibwa/0.5+galaxy0"

H_HCT1 = "2810f54e3c952a81"
H_HCT2 = "03694320c2305563"
H_MOLM = "01ce4583c5019a62"
H_GM = "b510be579c1fa8fe"

# Priority order: HCT116 is the replicate pair we most want and is one of
# Akita's five cell types; MOLM-13 second because its enzyme differs and so
# exercises the HindIII path; GM13977's depth series last.
SAMPLES = [
    # (label, history, R1 hda, R2 hda)
    ("HCT116_rep1", H_HCT1, "f788d2c55dd7d1f2", "2516b6627b80e903"),
    ("HCT116_rep2", H_HCT2, "1c437f54f8067477", "1116230d1ef01441"),
    # MOLM-13 rep1 is the CONCATENATION of SRR11250154 + SRR11250155, so the
    # replicate is 118M pairs rather than a 59M half of itself.
    ("MOLM13_rep1", H_MOLM, "54a3715b5b459479", "97c584e53b65d0e0"),
    ("MOLM13_rep2", H_MOLM, "21ebbf67a1fe7e35", "4921d3e474f26a89"),
]

# minibwa name-sorts through samtools, which spills temp BAMs to /tmp on the
# compute node. Six concurrent large jobs (plus leftover untrimmed mapping)
# exhausted /tmp on 2026-09-02 and killed two jobs with "No space left on
# device" mid-sort. /tmp free space cannot be inspected from here, so the only
# lever available is concurrency; keep it low rather than guess at headroom.
MAX_INFLIGHT = 3


def log(*a):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]", *a, flush=True)


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"steps": {}}


def save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def ds_state(did):
    try:
        d = gxhic.dataset(did)
    except Exception:
        return "unknown"
    return "deleted" if (d.get("deleted") or d.get("purged")) else d.get("state")


def all_ok(ids):
    return bool(ids) and all(i and ds_state(i) == "ok" for i in ids)


def record(st, key, **kw):
    st["steps"][key] = {**kw, "at": datetime.now().isoformat()}
    save(st)


def make_pair_collection(st, key, hist, label, r1, r2):
    rec = st["steps"].get(key)
    if rec:
        return rec.get("hdca")
    body = {"type": "dataset_collection", "collection_type": "paired",
            "name": f"{label} raw pair",
            "element_identifiers": [
                {"name": "forward", "src": "hda", "id": r1},
                {"name": "reverse", "src": "hda", "id": r2}]}
    try:
        r = gxhic.post(f"/api/histories/{hist}/contents", body)
    except Exception as e:
        log(f"collection FAILED {key}: {str(e)[:300]}")
        return None
    record(st, key, hdca=r["id"])
    log(f"collection {key} -> {r['id']}")
    return r["id"]


def run_fastp(st, key, hist, hdca):
    rec = st["steps"].get(key)
    if rec:
        if rec.get("failed"):
            return None
        return rec
    inputs = {
        "single_paired|single_paired_selector": "paired_collection",
        "single_paired|paired_input": {"src": "hdca", "id": hdca},
        "single_paired|adapter_trimming_options|detect_adapter_for_pe": True,
        "filter_options|quality_filtering_options|qualified_quality_phred": 30,
        "filter_options|quality_filtering_options|unqualified_percent_limit": 40,
        "filter_options|length_filtering_options|length_required": 25,
        "output_options|report_html": True,
        "output_options|report_json": True,
    }
    try:
        r = gxhic.run_tool(hist, FASTP, inputs)
    except Exception as e:
        record(st, key, failed=True, reason=str(e)[:1500])
        log(f"fastp SUBMIT FAILED {key}: {str(e)[:300]}")
        return None
    colls = [c["id"] for c in r.get("output_collections", [])]
    outs = {o.get("output_name") or f"out{i}": o["id"]
            for i, o in enumerate(r.get("outputs", []))}
    record(st, key, collections=colls, outputs=outs,
           jobs=[j["id"] for j in r.get("jobs", [])])
    log(f"fastp {key} -> colls={colls} outs={list(outs)}")
    return st["steps"][key]


def collection_elements(hdca_id, hist):
    """forward/reverse HDA ids of a paired collection."""
    c = gxhic.get(f"/api/dataset_collections/{hdca_id}", instance_type="history")
    out = {}
    for el in c.get("elements", []):
        obj = el.get("object") or {}
        out[el["element_identifier"]] = obj.get("id")
    return out


def map_mate(st, key, hist, fastq_id):
    rec = st["steps"].get(key)
    if rec:
        if rec.get("failed"):
            return None
        o = rec.get("outputs") or []
        return o if all_ok(o) else None
    try:
        r = gxhic.run_tool(hist, MINIBWA, {
            "reference_source|reference_source_selector": "cached",
            "reference_source|ref_file": "hg19",
            "fastq_input|fastq_input_selector": "single",
            "fastq_input|fastq_input1": {"src": "hda", "id": fastq_id},
            "mode": "hic",
            "output_format|output_format_selector": "bam",
            "output_format|output_sort": "name",
        })
    except Exception as e:
        record(st, key, failed=True, reason=str(e)[:1500])
        log(f"minibwa SUBMIT FAILED {key}: {str(e)[:300]}")
        return None
    record(st, key, outputs=[o["id"] for o in r.get("outputs", [])],
           jobs=[j["id"] for j in r.get("jobs", [])])
    log(f"minibwa {key} -> {st['steps'][key]['outputs']}")
    return None


def inflight(st, kind):
    n = 0
    for k, rec in st["steps"].items():
        if kind in k and not rec.get("failed"):
            outs = rec.get("outputs")
            if isinstance(outs, list) and outs and not all_ok(outs):
                n += 1
    return n


def download(st, label, mate, did):
    BAMDIR.mkdir(parents=True, exist_ok=True)
    dest = BAMDIR / f"{label}_{mate}.bam"
    want = (gxhic.dataset(did).get("file_size") or 0)
    if dest.exists() and dest.stat().st_size == want and want > 0:
        return dest
    if dest.exists():
        log(f"re-fetching {dest.name}: {dest.stat().st_size} != {want}")
        dest.unlink()
    log(f"downloading {dest.name} ({want/1e9:.2f} GB)")
    gxhic.download(did, str(dest), timeout=7200)
    got = dest.stat().st_size
    if got != want:
        log(f"TRUNCATED {dest.name}: {got} != {want}; removing")
        dest.unlink()
        return None
    record(st, f"dl_{label}_{mate}", path=str(dest), size=got)
    log(f"downloaded {dest.name} OK ({got/1e9:.2f} GB)")
    return dest


def gm_samples():
    cs = gxhic.contents(H_GM, v="dev", keys="id,name,state")
    fq = {c["name"]: c for c in cs if c["name"].endswith(".fastq.gz")}
    out = []
    for run in sorted({n.rsplit("_", 1)[0] for n in fq}):
        a, b = fq.get(f"{run}_1.fastq.gz"), fq.get(f"{run}_2.fastq.gz")
        if a and b and a["state"] == "ok" and b["state"] == "ok":
            out.append((f"GM13977_{run}", H_GM, a["id"], b["id"]))
    return out


def advance(st, samples):
    for label, hist, r1, r2 in samples:
        hdca = make_pair_collection(st, f"coll_{label}", hist, label, r1, r2)
        if not hdca:
            continue
        f = run_fastp(st, f"fastp_{label}", hist, hdca)
        if not f or not f.get("collections"):
            continue
        try:
            els = collection_elements(f["collections"][0], hist)
        except Exception as e:
            log(f"{label}: cannot read fastp collection yet ({str(e)[:120]})")
            continue
        pair = {}
        for ident, mate in (("forward", "R1"), ("reverse", "R2")):
            did = els.get(ident)
            if not did or ds_state(did) != "ok":
                continue
            key = f"map_{label}_{mate}"
            if key not in st["steps"] and inflight(st, "map_") >= MAX_INFLIGHT:
                continue
            got = map_mate(st, key, hist, did)
            if got:
                pair[mate] = got[0]
        for mate, bam in pair.items():
            download(st, label, mate, bam)


def main():
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    interval = int(nums[0]) if nums else 300
    import fcntl
    lock = open(ROOT / "runs" / ".gx_trimmap.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another gx_trim_map is already running")
    once = "--once" in sys.argv
    st = load()
    while True:
        try:
            samples = list(SAMPLES)
            # GM13977 only after the four priority samples have their BAMs.
            if all(f"dl_{l}_R2" in st["steps"] for l, *_ in SAMPLES):
                samples += gm_samples()
            advance(st, samples)
        except Exception:
            log("tick error:\n" + traceback.format_exc())
        if once:
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
