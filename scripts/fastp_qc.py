"""Summarise the fastp JSON reports: how much Q30 filtering actually removed.

This matters beyond housekeeping. Discarding a large fraction of read pairs
lowers the effective sequencing depth, and the reproducibility ceiling at fine
bins is depth-limited -- so a heavy trim would itself push the 1 kb agreement
down and must not be mistaken for irreproducible signal.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gxhic  # noqa: E402

REPORTS = {
    "HCT116_rep1": "a6576b69d0a40dfc",
    "HCT116_rep2": "715ede122fd5e655",
    "MOLM13_rep1": "f1e8311da0ae17ca",
    "MOLM13_rep2": "0c9ac71492d18572",
}


def summarise(label, did):
    d = gxhic.dataset(did)
    if d.get("state") != "ok":
        return f"{label:14s} fastp {d.get('state')}"
    raw = gxhic.get(f"/api/datasets/{did}/display")
    if isinstance(raw, (dict, list)):
        j = raw
    else:
        j = json.loads(raw)
    b, a = j["summary"]["before_filtering"], j["summary"]["after_filtering"]
    fr = j.get("filtering_result", {})
    # fastp counts READS; paired data has two reads per pair.
    bp, ap = b["total_reads"] / 2, a["total_reads"] / 2
    kept = 100.0 * ap / bp if bp else float("nan")
    return (f"{label:14s} {bp/1e6:8.1f}M -> {ap/1e6:8.1f}M pairs "
            f"({kept:5.1f}% kept, {100-kept:4.1f}% discarded)  "
            f"Q30 {100*b['q30_rate']:.1f}->{100*a['q30_rate']:.1f}%  "
            f"len {b['read1_mean_length']}->{a['read1_mean_length']}  "
            f"lowqual={fr.get('low_quality_reads',0)/2/1e6:.1f}M "
            f"short={fr.get('too_short_reads',0)/2/1e6:.1f}M "
            f"manyN={fr.get('too_many_N_reads',0)/2/1e6:.1f}M")


def main():
    for label, did in REPORTS.items():
        try:
            print(summarise(label, did), flush=True)
        except Exception as e:  # noqa
            print(f"{label:14s} ERROR {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
