"""Drive the Hi-C fetch -> map -> build pipeline on galaxy-heidel to completion.

The pipeline is long (a 35 GB ENA fetch alone takes ~50 min; mapping and matrix
building take hours), so it is written as an idempotent state machine polled on
a slow cadence rather than as a script that blocks step by step. Every rule
checks whether its output already exists before submitting, so the driver can
be killed and restarted at any point without duplicating work.

Failures are recorded and then skipped, never retried in a loop, so a bad
configuration cannot be resubmitted repeatedly.

Matrices are built at 1 kb ONLY. Coarser resolutions are produced locally with
cooler zoomify instead of by asking hicBuildMatrix for extra bin sizes: binning
the same set of valid pairs at 40 kb and summing 1 kb bins into 40 kb bins give
the identical matrix, so the coarse rungs cost nothing on the server and the
submitted parameter set stays minimal.
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
STATE = ROOT / "runs" / "gx_pipeline_state.json"

MINIBWA = "toolshed.g2.bx.psu.edu/repos/iuc/minibwa/minibwa/0.5+galaxy0"
BUILD = ("toolshed.g2.bx.psu.edu/repos/bgruening/hicexplorer_hicbuildmatrix/"
         "hicexplorer_hicbuildmatrix/3.7.6+galaxy1")

H_HCT1 = "2810f54e3c952a81"
H_HCT2 = "03694320c2305563"
H_MOLM = "01ce4583c5019a62"
H_GM = "b510be579c1fa8fe"

CUT_GATC = "ffe621e91c5cb8b9"      # MboI   -> HCT116, GM13977
CUT_AAGCTT = "ef1c387f9103c6d3"    # HindIII -> MOLM-13

# HCT116 rep2 mates were mapped before this driver existed.
HCT2_BAM = ["03f22b6f8bd401c8", "682a7666adaf64d2"]
# MOLM-13: rep1 is two ENA runs of one library, concatenated per mate as FASTQ
# so that minibwa name-sorts the merged file itself (the available samtools
# merge wrapper exposes no -n, and Picard MergeSamFiles coordinate-sorts, so
# neither can merge name-sorted BAMs without destroying the mate ordering
# hicBuildMatrix depends on) and so duplicate detection sees across both runs.
MOLM_R1_PARTS = ("42ff68ba8d9cf477", "77682e8a1cdd6ead")   # 154_1, 155_1
MOLM_R2_PARTS = ("7c4e5c7168519ae9", "683b56a9e6c42fd2")   # 154_2, 155_2
MOLM_REP2_FQ = ("21ebbf67a1fe7e35", "4921d3e474f26a89")


def log(*a):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]", *a, flush=True)


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"steps": {}}


def save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


_cache = {}


def ds_state(did, fresh=False):
    if not fresh and did in _cache and _cache[did] == "ok":
        return "ok"
    try:
        d = gxhic.dataset(did)
    except Exception:
        return "unknown"
    s = "deleted" if (d.get("deleted") or d.get("purged")) else d.get("state")
    _cache[did] = s
    return s


def all_ok(ids):
    return bool(ids) and all(i and ds_state(i) == "ok" for i in ids)


def submit(st, key, hist, tool_id, inputs):
    """Run a tool at most once. Returns output ids when they are all ok."""
    steps = st["steps"]
    rec = steps.get(key)
    if rec is not None:
        if rec.get("failed"):
            return None
        outs = rec.get("outputs") or []
        # hicBuildMatrix declares an optional BAM output that is filtered out
        # when outBam is false; a filtered output never leaves 'discarded', so
        # it must not hold the step open or be read as a failure.
        states = [ds_state(o) for o in outs]
        live = [s for s in states if s != "discarded"]
        if live and all(s == "ok" for s in live):
            return [o for o, s in zip(outs, states) if s == "ok"]
        if any(s in ("error", "empty") for s in states):
            rec["failed"] = True
            rec["reason"] = f"output state {states}"
            log(f"FAILED {key}: output state {states}")
            save(st)
        return None
    try:
        r = gxhic.run_tool(hist, tool_id, inputs)
    except Exception as e:
        steps[key] = {"failed": True, "reason": str(e)[:1500], "outputs": []}
        log(f"SUBMIT FAILED {key}: {str(e)[:300]}")
        save(st)
        return None
    steps[key] = {"outputs": [o["id"] for o in r.get("outputs", [])],
                  "jobs": [j["id"] for j in r.get("jobs", [])],
                  "hist": hist, "tool": tool_id.rsplit("/", 2)[-2:][0],
                  "submitted": datetime.now().isoformat()}
    log(f"submitted {key} -> {steps[key]['outputs']}")
    save(st)
    return None


def cat_fastq(st, key, hist, a, b):
    return submit(st, key, hist, "cat1",
                  {"input1": {"src": "hda", "id": a},
                   "queries_0|input2": {"src": "hda", "id": b}})


def map_reads(st, key, hist, fastq_id):
    return submit(st, key, hist, MINIBWA, {
        "reference_source|reference_source_selector": "cached",
        "reference_source|ref_file": "hg19",
        "fastq_input|fastq_input_selector": "single",
        "fastq_input|fastq_input1": {"src": "hda", "id": fastq_id},
        "mode": "hic",
        "output_format|output_format_selector": "bam",
        "output_format|output_sort": "name",
    })


# hicBuildMatrix 3.7.6 is unusable on this server: its conda environment has a
# truncated libgomp.so.1, so the tool dies on `import sklearn` before reading
# any input (confirmed three times, including a deliberate retry). Mapping and
# fetching still stage the data, so those keep running; matrix building is held
# behind this sentinel and resumes the moment the environment is repaired:
#     touch runs/.builds_enabled
BUILD_GATE = ROOT / "runs" / ".builds_enabled"
_gate_logged = False


def builds_enabled():
    global _gate_logged
    if BUILD_GATE.exists():
        return True
    if not _gate_logged:
        log("matrix builds HELD: __hicexplorer@3.7.6 has a truncated "
            "libgomp.so.1; touch runs/.builds_enabled once it is rebuilt")
        _gate_logged = True
    return False


def build_matrix(st, key, hist, sam_a, sam_b, restr, dangl, cutfile):
    if not builds_enabled():
        return None
    return submit(st, key, hist, BUILD, {
        "samFiles_0|samFile": {"src": "hda", "id": sam_a},
        "samFiles_1|samFile": {"src": "hda", "id": sam_b},
        "restrictionCutFile": {"src": "hda", "id": cutfile},
        "restrictionSequence": restr,
        "danglingSequence": dangl,
        "binSizes_0|binSize": 1000,
        "outputFormat": "cool",
    })


def adopt(st, key, hist, outputs):
    if key not in st["steps"]:
        st["steps"][key] = {"outputs": outputs, "hist": hist,
                            "adopted": datetime.now().isoformat()}
        save(st)


def outs(st, key):
    return (st["steps"].get(key) or {}).get("outputs") or []


def in_flight(st, prefix):
    """Submitted steps whose outputs are not yet all ok."""
    n = 0
    for key, rec in st["steps"].items():
        if key.startswith(prefix) and not rec.get("failed"):
            if not all_ok(rec.get("outputs") or []):
                n += 1
    return n


def tick(st):
    # work submitted by hand before the driver existed
    adopt(st, "map_hct1_R1", H_HCT1, ["204042af9980a3ac"])
    adopt(st, "map_hct1_R2", H_HCT1, ["38b1d59aa4bc141e"])
    adopt(st, "cat_molm_rep1_R1", H_MOLM, ["54a3715b5b459479"])

    cuts_ready = all_ok([CUT_GATC]), all_ok([CUT_AAGCTT])

    # ---- HCT116 rep2 (mates already mapped) --------------------------------
    if cuts_ready[0] and all_ok(HCT2_BAM):
        build_matrix(st, "matrix_HCT116_rep2", H_HCT2, HCT2_BAM[0],
                     HCT2_BAM[1], "GATC", "GATC", CUT_GATC)

    # ---- HCT116 rep1 -------------------------------------------------------
    r1 = [outs(st, "map_hct1_R1")[0], outs(st, "map_hct1_R2")[0]]
    if cuts_ready[0] and all_ok(r1):
        build_matrix(st, "matrix_HCT116_rep1", H_HCT1, r1[0], r1[1],
                     "GATC", "GATC", CUT_GATC)

    # ---- MOLM-13 rep1: concatenate the two runs, then map ------------------
    cat_fastq(st, "cat_molm_rep1_R1", H_MOLM, *MOLM_R1_PARTS)
    cat_fastq(st, "cat_molm_rep1_R2", H_MOLM, *MOLM_R2_PARTS)
    mrep1 = []
    for mate in ("R1", "R2"):
        c = outs(st, f"cat_molm_rep1_{mate}")
        if all_ok(c):
            map_reads(st, f"map_molm_rep1_{mate}", H_MOLM, c[0])
        mrep1.append((outs(st, f"map_molm_rep1_{mate}") or [None])[0])
    if cuts_ready[1] and all_ok(mrep1):
        build_matrix(st, "matrix_MOLM13_rep1", H_MOLM, mrep1[0], mrep1[1],
                     "AAGCTT", "AGCT", CUT_AAGCTT)

    # ---- MOLM-13 rep2: one 328M-pair run ----------------------------------
    mrep2 = []
    for mate, fq in zip(("R1", "R2"), MOLM_REP2_FQ):
        map_reads(st, f"map_molm_rep2_{mate}", H_MOLM, fq)
        mrep2.append((outs(st, f"map_molm_rep2_{mate}") or [None])[0])
    if cuts_ready[1] and all_ok(mrep2):
        build_matrix(st, "matrix_MOLM13_rep2", H_MOLM, mrep2[0], mrep2[1],
                     "AAGCTT", "AGCT", CUT_AAGCTT)

    # ---- GM13977: 16 replicates -------------------------------------------
    # Mapping writes roughly 250 GB of BAM to a server whose free space cannot
    # be checked from here. It was originally gated behind the build sentinel,
    # but mapping is useful work on its own -- the BAMs are exactly what the
    # builds will need once the hicexplorer conda env is repaired, and mapping
    # takes hours, so doing it now shortens the critical path. Only the matrix
    # BUILD stays gated, since that tool is known to fail on import.
    gm = gxhic.contents(H_GM, v="dev", keys="id,hid,name,state")
    fq = {c["name"]: c for c in gm if c["name"].endswith(".fastq.gz")}
    if fq and all(c["state"] == "ok" for c in fq.values()):
        for run in sorted({n.rsplit("_", 1)[0] for n in fq}):
            b = []
            for mate in ("1", "2"):
                d = fq.get(f"{run}_{mate}.fastq.gz")
                key = f"map_gm_{run}_R{mate}"
                # Submitting all 32 at once would swamp the queue and starve
                # the HCT116 and MOLM-13 builds, which are the results the
                # depth series is compared against. Keep a bounded number of
                # GM mapping jobs in flight and let the rest wait.
                if d and (key in st["steps"] or in_flight(st, "map_gm_") < 8):
                    map_reads(st, key, H_GM, d["id"])
                b.append((outs(st, key) or [None])[0])
            if cuts_ready[0] and all_ok(b) and builds_enabled():
                build_matrix(st, f"matrix_GM13977_{run}", H_GM, b[0], b[1],
                             "GATC", "GATC", CUT_GATC)
    save(st)


def main():
    # Exactly one driver may run: two would race on the state file and could
    # submit the same expensive job twice.
    import fcntl
    lock = open(ROOT / "runs" / ".gx_drive.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another gx_drive is already running")

    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    st = load()
    while True:
        try:
            tick(st)
        except Exception:
            log("tick error:\n" + traceback.format_exc())
        time.sleep(interval)


if __name__ == "__main__":
    main()
