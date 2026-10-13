"""Inventory, download and input build of the independent validation samples (registry section 74).

Four samples (HeLa-S3 was planned: its only processed Hi-C is the 4DN mcool 4DNFIBM9QCFG, which the 4DN
portal serves only to logged-in users, and ENCODE holds only its fastq files, so it is not used) that no part of the pipeline has touched (not among K562, HMEC, HUVEC, GM12878, IMR-90,
H1, HepG2, A549, GM23248, CD14+ monocyte, HCT116, MCF-7, HFF, NHEK, PC-9, the AML patients):

  caco2  Caco-2 colorectal adenocarcinoma. Hi-C: ENCODE intact (MNase) ENCSR820DYU, ENCFF234MDO (MAPQ >= 30).
  pc3    PC-3 prostate adenocarcinoma. Hi-C: ENCODE intact (MNase) ENCSR038TZA, ENCFF322IBI (MAPQ >= 30).
  panc1  Panc1 pancreatic carcinoma. Hi-C: ENCODE in situ HindIII ENCSR440CTR, ENCFF168OXT (2 replicates).
  nk     natural killer cell, primary, male adult (33 y). Hi-C: ENCODE intact ENCSR971CJS, ENCFF197OWW.

Inputs are built exactly as for the training cells (scripts/encode_expand.py): ENCODE MAPQ30 .hic ->
hicxchange hic2cool at one resolution -> cooler balance --ignore-diags 2 --max-iters 500 ->
scripts/oe_proper.py (the 4DN mcool: oe_proper on its stored weights, as the LOCAL cells of
encode_expand), and ChIP as the released GRCh38 fold-change-over-control bigWig of each mark into
hg38_<cell>/<MARK>.bigwig. Resolutions 10 kb and 25 kb (the two the evaluated models use).

Every download goes to a .part name with curl -C - and is checked against the portal's size and md5
before it is renamed; derived files are written under a temporary name and renamed when complete, so a
rerun resumes at the first missing output.

    python scripts/independent_validation_fetch.py inventory          # picks + table, no download
    python scripts/independent_validation_fetch.py chip  [--only pc3]
    python scripts/independent_validation_fetch.py hic   [--only pc3] [--threads 6]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HOME = Path.home()
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
VN = HM / "validation_new"
ENV = HOME / "miniconda3/envs/hicdiff/bin"
H2C = HOME / "src/hicxchange/build/hicxchange"
CODE = Path(__file__).resolve().parents[1]
CHR1 = 248956422
RES = [10000, 25000]
ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
# the seven marks every sample has: the same inputs for all samples (PC-3 has four more, not downloaded:
# H3K4me2, H3K9ac, H3K79me2, H4K20me1; the disk budget of the download is 150 GB)
CORE7 = ["CTCF", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]
STAGE = Path(os.environ.get("IV_STAGE", "/tmp/iv_stage"))

# label: dict(term = ENCODE biosample term of the ChIP, hic = (source, accession), assay, donor hint)
SAMPLES = {
    "pc3": dict(term="PC-3", hic=("encode", "ENCFF322IBI"), assay="intact", exp="ENCSR038TZA",
                desc="PC-3, intact MNase", donor=""),
    "panc1": dict(term="Panc1", hic=("encode", "ENCFF168OXT"), assay="hic", exp="ENCSR440CTR",
                  desc="Panc1, in situ HindIII, 2 replicates", donor=""),
    "nk": dict(term="natural killer cell", hic=("encode", "ENCFF197OWW"), assay="intact", exp="ENCSR971CJS",
               desc="NK cell, primary, intact MNase", donor="male adult (33 years)"),
    "caco2": dict(term="Caco-2", hic=("encode", "ENCFF234MDO"), assay="intact", exp="ENCSR820DYU",
                  desc="Caco-2, intact MNase", donor=""),
}
URL_4DN = "https://data.4dnucleome.org/files-processed/{a}/@@download/{a}.mcool"
API = "https://www.encodeproject.org"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sh(cmd, check=True):
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"FAILED ({r.returncode}): {cmd}\n{r.stderr[-3000:]}")
    return r


def get_json(url, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as e:
            if e.code == 404 and "/search/" in url:      # the portal answers 404 to an empty search
                return {"@graph": []}
            log(f"[retry {i}] {url[:100]}: {e}")
            time.sleep(10)
        except Exception as e:                          # noqa: BLE001
            log(f"[retry {i}] {url[:100]}: {e}")
            time.sleep(10)
    raise RuntimeError(url)


def cached(name, fn):
    p = VN / "meta" / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    v = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v))
    return v


# ------------------------------------------------------------------ ChIP picks
def candidates(term, target):
    q = urllib.parse.urlencode([
        ("type", "File"), ("file_format", "bigWig"), ("assembly", "GRCh38"), ("status", "released"),
        ("output_type", "fold change over control"), ("biosample_ontology.term_name", term),
        ("target.label", target), ("limit", "all"), ("format", "json"), ("frame", "object")])
    return get_json(f"{API}/search/?{q}")["@graph"]


def experiment(acc):
    def f():
        e = get_json(f"{API}/experiments/{acc}/?format=json&frame=embedded")
        keep = {k: e.get(k) for k in ("accession", "assay_title", "biosample_summary", "perturbed",
                                      "date_released", "description")}
        keep["lab"] = (e.get("lab") or {}).get("title")
        keep["treated"] = any(r.get("library", {}).get("biosample", {}).get("treatments")
                              for r in e.get("replicates", []))
        keep["modified"] = any(r.get("library", {}).get("biosample", {}).get("applied_modifications")
                               or r.get("library", {}).get("biosample", {}).get("genetic_modifications")
                               for r in e.get("replicates", []))
        keep["donors"] = sorted({str(r.get("library", {}).get("biosample", {}).get("donor", {}).get("accession"))
                                 for r in e.get("replicates", [])
                                 if isinstance(r.get("library", {}).get("biosample", {}).get("donor"), dict)})
        keep["biosample_age"] = sorted({str(r.get("library", {}).get("biosample", {}).get("age"))
                                        for r in e.get("replicates", [])})
        return keep
    return cached(f"exp_{acc}", f)


def pick_chip(label):
    """Rule: released GRCh38 fold-change bigWig of an unperturbed, untreated, unmodified ChIP-seq
    experiment (Histone / TF / Mint ChIP-seq) of the biosample term; the donor of the Hi-C when one is
    given; no cell-cycle-phase sample unless that is all there is; the file that pools both
    biological replicates, then the largest file."""
    s = SAMPLES[label]
    out = {}
    for t in CORE7:
        c = cached(f"cand_{label}_{t}", lambda: [
            {k: f.get(k) for k in ("accession", "file_size", "md5sum", "href", "dataset",
                                   "biological_replicates", "technical_replicates", "date_created")}
            for f in candidates(s["term"], t)])
        rows = []
        for f in c:
            ex = experiment(f["dataset"].strip("/").split("/")[-1])
            if ex["assay_title"] not in ("Histone ChIP-seq", "TF ChIP-seq", "Mint-ChIP-seq"):
                continue
            if ex["perturbed"] or ex["treated"] or ex["modified"]:
                continue
            summ = (ex.get("biosample_summary") or "")
            if "treated" in summ:
                continue
            phase = "phase" in summ or "G1" in summ or "S phase" in summ
            donor_ok = (not s["donor"]) or (s["donor"] in summ) or not summ
            # the pooled file, unless it is larger than 2.5 GB (disk budget of the validation download)
            pooled = sorted(f.get("biological_replicates") or []) == [1, 2] and f["file_size"] < 2.5e9
            rows.append((not donor_ok, phase, not pooled, -f["file_size"], f, ex))
        small = [r for r in rows if r[4]["file_size"] < 2.5e9]
        rows = small or rows                             # disk budget: no file above 2.5 GB if avoidable
        if rows:
            rows.sort(key=lambda r: r[:4])
            _, phase, _, _, f, ex = rows[0]
            out[t] = dict(mark=t, accession=f["accession"], file_size=f["file_size"], md5sum=f["md5sum"],
                          url=API + f["href"], experiment=ex["accession"],
                          biosample_summary=ex.get("biosample_summary") or "", lab=ex.get("lab"),
                          replicates=f.get("biological_replicates"), phase_sample=bool(phase),
                          n_candidates=len(c), n_eligible=len(rows))
    return out


# ------------------------------------------------------------------ download
def md5(path, bs=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(bs):
            h.update(b)
    return h.hexdigest()


def download(url, dest: Path, size: int, md5sum: str):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dest) + ".part")
    for full in range(3):
        for i in range(80):
            have = part.stat().st_size if part.exists() else 0
            if have == size:
                break
            if have > size:
                part.unlink()
            if i % 5 == 0:
                log(f"[dl] {dest.name} ({have / 1e9:.2f}/{size / 1e9:.2f} GB)")
            r = sh(f"curl -fsSL -C - --speed-limit 100000 --speed-time 120 --connect-timeout 60 "
                   f"-o '{part}' '{url}'", check=False)
            if r.returncode != 0:
                time.sleep(15)
        if not part.exists() or part.stat().st_size != size:
            raise RuntimeError(f"{dest.name}: size never matched")
        got = md5(part)
        if got == md5sum:
            part.rename(dest)
            log(f"[dl ok] {dest.name} md5 {got}")
            return dest
        log(f"[dl BAD md5] {dest.name} {got} != {md5sum}; refetching")
        part.unlink()
    raise RuntimeError(f"{dest.name}: md5 mismatch")


def hic_source(label):
    """(url, size, md5, filename) of the Hi-C file of a sample."""
    kind, acc = SAMPLES[label]["hic"]
    if kind == "4dn":
        def f():
            d = get_json(f"https://data.4dnucleome.org/files-processed/{acc}/?format=json&frame=object")
            return {"file_size": d["file_size"], "md5sum": d["md5sum"], "genome_assembly": d["genome_assembly"]}
        m = cached(f"hic_{acc}", f)
        return URL_4DN.format(a=acc), m["file_size"], m["md5sum"], f"{acc}.mcool"

    def f():
        d = get_json(f"{API}/files/{acc}/?format=json&frame=object")
        return {k: d.get(k) for k in ("accession", "md5sum", "file_size", "href", "assembly", "output_type",
                                      "dataset", "filter_value")}
    m = cached(f"hic_{acc}", f)
    assert m["assembly"] == "GRCh38", m
    return API + m["href"], m["file_size"], m["md5sum"], f"{acc}.hic"


# ------------------------------------------------------------------ build
def cool_check(path: Path, res: int):
    import cooler
    c = cooler.Cooler(str(path))
    names = list(c.chromnames)
    chr1 = int(c.chromsizes.get("chr1", c.chromsizes.get("1", -1)))
    if chr1 != CHR1:
        raise RuntimeError(f"{path}: chr1 length {chr1}, not hg38")
    if c.binsize != res:
        raise RuntimeError(f"{path}: binsize {c.binsize} != {res}")
    return {"chroms": names[:3], "nnz": int(c.info["nnz"]), "binsize": res}


def total_contacts(path: Path):
    import cooler
    import numpy as np
    c = cooler.Cooler(str(path))
    px = c.pixels()
    n = c.info["nnz"]
    tot = cis = 0.0
    b = c.bins()[:]["chrom"].cat.codes.values
    for i in range(0, n, 25_000_000):
        p = px[i:i + 25_000_000]
        v = p["count"].values.astype(np.float64)
        tot += v.sum()
        cis += v[b[p["bin1_id"].values] == b[p["bin2_id"].values]].sum()
    return {"contacts": float(tot), "cis_contacts": float(cis)}


def build(label, threads):
    s = SAMPLES[label]
    url, size, md, fn = hic_source(label)
    OEd = VN / "oe"
    OEd.mkdir(parents=True, exist_ok=True)
    (VN / "stats").mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    todo = [r for r in RES if not (OEd / f"{label}_proper_{r // 1000}kb.cool").exists()]
    if not todo:
        log(f"[have] {label}")
        return
    src = download(url, VN / "hic" / fn, size, md)
    import fcntl                                        # one CPU-heavy conversion at a time (<= 8 threads)
    lockf = open(VN / ".build.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    for res in todo:
        k = res // 1000
        out = OEd / f"{label}_proper_{k}kb.cool"
        if src.suffix == ".mcool":
            loc_in = f"{src}::/resolutions/{res}"
            wdir = loc_in
            bal = STAGE / f"{label}_{k}kb.cool"
            sh(f"rm -f '{bal}' && {ENV}/cooler cp '{wdir}' '{bal}'")   # the 4DN level, stored weights
        else:
            raw = STAGE / f"{label}_{k}kb.raw.cool"
            bal = STAGE / f"{label}_{k}kb.cool"
            sh(f"rm -f '{raw}' '{bal}'")
            log(f"[hic2cool] {label} {res}")
            sh(f"nice -n 10 {H2C} hic2cool convert -r {res} -p {threads} -s '{src}' '{raw}'")
            cool_check(raw, res)
            log(f"[balance] {label} {res}")
            sh(f"mv '{raw}' '{bal}' && nice -n 10 {ENV}/cooler balance --force -p {threads} "
               f"--ignore-diags 2 --max-iters 500 '{bal}'")
        st = cool_check(bal, res)
        import cooler
        w = cooler.Cooler(str(bal)).bins()[:]["weight"]
        st["ice_masked_frac"] = float(w.isna().mean())
        if res == max(RES) or True:
            st.update(total_contacts(bal) if res == 25000 else {})
        st.update({"label": label, "hic_file": fn, "assay": s["assay"], "experiment": s["exp"]})
        (VN / "stats" / f"{label}_{k}kb.json").write_text(json.dumps(st, indent=1))
        tmp = STAGE / f"oe_{label}_{k}kb.cool"
        log(f"[oe_proper] {label} {res}")
        sh(f"rm -f '{tmp}' && cd {CODE} && nice -n 10 {ENV}/python scripts/oe_proper.py --input '{bal}' "
           f"--output '{tmp}' --binsize {res} > '{VN}/stats/oe_{label}_{k}kb.log' 2>&1")
        part = out.with_name(out.name + ".part")
        sh(f"cp '{tmp}' '{part}'")
        assert part.stat().st_size == tmp.stat().st_size
        part.rename(out)
        # keep the balanced raw-count cool (split-half ceiling stage of independent_validation.py)
        keep = VN / "balanced" / f"{label}_{k}kb.cool"
        keep.parent.mkdir(parents=True, exist_ok=True)
        kp = keep.with_name(keep.name + ".part")
        sh(f"cp '{bal}' '{kp}'")
        assert kp.stat().st_size == bal.stat().st_size
        kp.rename(keep)
        for p in (tmp, bal):
            p.unlink(missing_ok=True)
        log(f"[oe ok] {label} {res} -> {out.name}")


def chip(label):
    picks = pick_chip(label)
    d = VN / f"hg38_{label}"
    d.mkdir(parents=True, exist_ok=True)
    for t, r in picks.items():
        download(r["url"], d / f"{t}.bigwig", r["file_size"], r["md5sum"])
    missing = [t for t in ALL13 if t not in picks]      # models see these as absent
    (d / "missing.json").write_text(json.dumps(missing))
    log(f"[chip ok] {label}: {len(picks)}/13, missing {missing}")


def inventory(only):
    rows = []
    for lab in (only or SAMPLES):
        s = SAMPLES[lab]
        picks = pick_chip(lab)
        url, size, md, fn = hic_source(lab)
        for t, r in picks.items():
            rows.append({"sample": lab, "kind": "chip", "mark": t, "accession": r["accession"],
                         "experiment": r["experiment"], "size_gb": round(r["file_size"] / 1e9, 3),
                         "biosample_summary": r["biosample_summary"], "lab": r["lab"],
                         "replicates": r["replicates"], "cell_cycle_phase": r["phase_sample"],
                         "n_candidates": r["n_candidates"], "n_eligible": r["n_eligible"]})
        rows.append({"sample": lab, "kind": "hic", "mark": "Hi-C", "accession": fn.split(".")[0],
                     "experiment": s["exp"], "size_gb": round(size / 1e9, 3), "biosample_summary": s["desc"]})
    out = VN / "inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["sample", "kind", "mark", "accession", "experiment", "size_gb", "biosample_summary", "lab",
            "replicates", "cell_cycle_phase", "n_candidates", "n_eligible"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    import collections
    tot = collections.defaultdict(float)
    for r in rows:
        tot[(r["sample"], r["kind"])] += r["size_gb"]
        print(r["sample"], r["kind"], r["mark"], r["accession"], r["experiment"], r["size_gb"], r.get("replicates"),
              r.get("cell_cycle_phase"), (r.get("biosample_summary") or "")[:50])
    for k, v in sorted(tot.items()):
        print(k, round(v, 1), "GB")
    print("TOTAL", round(sum(tot.values()), 1), "GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["inventory", "chip", "hic"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    labs = a.only or list(SAMPLES)
    VN.mkdir(parents=True, exist_ok=True)
    if a.stage == "inventory":
        inventory(a.only)
    elif a.stage == "chip":
        for lab in labs:
            chip(lab)
    else:
        for lab in labs:
            build(lab, a.threads)


if __name__ == "__main__":
    main()
