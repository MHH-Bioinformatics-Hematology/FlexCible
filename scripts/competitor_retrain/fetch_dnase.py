"""ENCODE DNase-seq signal bigWigs (GRCh38, read-depth normalized signal) for the cells of the
competitor retraining (registry section 73). One unperturbed experiment and one file per cell;
size and md5 are checked against the ENCODE file record before the .part file is renamed.

    python scripts/competitor_retrain/fetch_dnase.py [--only k562 ...]

Output: /mnt/processed/wolffjoa/hic-multiome/competitor_retrain/dnase/<cell>.bigWig and
dnase_picks.tsv (cell, experiment, file, size, md5, biosample summary, treatments).
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

OUT = Path("/mnt/processed/wolffjoa/hic-multiome/competitor_retrain/dnase")
# cell: (experiment, file). Experiments are the unperturbed ENCODE DNase-seq experiments with
# two released replicates (UW / Duke classic series where available); the file is the
# replicate-level read-depth normalized signal of largest size.
PICKS = {
    "gm12878": ("ENCSR000EMT", "ENCFF743ULW"),
    "imr90": ("ENCSR000EKG", "ENCFF323RPM"),
    "hct116": ("ENCSR000ENM", "ENCFF225OLI"),
    "hff": ("ENCSR000ENQ", "ENCFF111BCZ"),
    "h1": ("ENCSR000EJN", "ENCFF153XQL"),
    "hepg2": ("ENCSR000EJV", "ENCFF136DBS"),
    "a549": ("ENCSR000EIE", "ENCFF124QWS"),
    "nhek": ("ENCSR000ELH", "ENCFF690MIA"),
    "mcf7": ("ENCSR000EKZ", "ENCFF382YNA"),
    "cd14mono": ("ENCSR000ELE", "ENCFF678LXL"),
    "pc9": ("ENCSR781TYE", "ENCFF482ZNF"),
    "k562": ("ENCSR000EKS", "ENCFF972GVB"),
    "hmec": ("ENCSR000ENV", "ENCFF263BZU"),
    "huvec": ("ENCSR000EOQ", "ENCFF752DYE"),
}


def api(path):
    r = urllib.request.Request("https://www.encodeproject.org" + path + ("&" if "?" in path else "?") + "format=json",
                               headers={"Accept": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    picks = OUT.parent / "dnase_picks.tsv"
    rows = []
    for cell, (exp, acc) in PICKS.items():
        if a.only and cell not in a.only:
            continue
        f = api(f"/files/{acc}/")
        e = api(f"/experiments/{exp}/")
        assert f["file_format"] == "bigWig" and f["assembly"] == "GRCh38", (cell, acc)
        assert f["dataset"].strip("/").split("/")[-1] == exp, (cell, acc, f["dataset"])
        treat = e.get("perturbed") or any(r["library"]["biosample"].get("treatments")
                                          for r in e.get("replicates", []))
        dst = OUT / f"{cell}.bigWig"
        if not (dst.exists() and dst.stat().st_size == f["file_size"]):
            part = dst.with_suffix(".part")
            for attempt in range(8):
                rc = subprocess.call(["curl", "-sS", "-L", "-C", "-", "-o", str(part),
                                      "https://www.encodeproject.org" + f["href"]])
                if rc == 0 and part.exists() and part.stat().st_size == f["file_size"]:
                    break
                time.sleep(10)
            assert part.stat().st_size == f["file_size"], f"{cell}: size"
            assert md5(part) == f["md5sum"], f"{cell}: md5"
            os.replace(part, dst)
        print(cell, exp, acc, f["file_size"], "treated" if treat else "untreated", flush=True)
        rows.append((cell, exp, acc, f["file_size"], f["md5sum"], e.get("biosample_summary", ""),
                     "treated" if treat else "untreated"))
    if rows and not a.only:
        picks.write_text("cell\texperiment\tfile\tsize\tmd5\tbiosample\ttreatments\n" +
                         "\n".join("\t".join(map(str, r)) for r in rows) + "\n")


if __name__ == "__main__":
    main()
