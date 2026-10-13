"""Inventory, download and input build of the mouse transfer samples (registry section 75).

The human-trained models are applied unchanged to mouse. Samples (ENCODE, assembly mm10; every input of
one sample is mm10, nothing is lifted):

  mesc   mouse ES cell E14. Hi-C ENCSR051JZL (in situ, Aiden lab, MAPQ >= 30); replicate experiment of the
         same cell line ENCSR819XFA (ceiling). ChIP: ES-E14 experiments (H3K27ac, H3K4me1, H3K4me3,
         H3K36me3, H3K9me3) and E14TG2a.4 (CTCF, H3K27me3; the E14 subclone), the E14 lineage of the Hi-C.
  mesc_b4  the same Hi-C, ChIP of ES-Bruce4 (all seven marks from one lab): marks-from-another-line control.
  ch12   CH12.LX B-cell lymphoma. Hi-C ENCSR005UON (in situ, 3 replicates, MAPQ >= 30). ChIP: CH12.LX, seven
         marks plus RAD21 and SMC3.
  heart  adult heart (2 months). Hi-C ENCSR997TTD (intact, B6CASTF1/J). ChIP of heart tissue, adult
         2 months (CTCF, H3K27ac, H3K4me3 of B6CASTF1/J female; the other four of B6NCrl male).
  (cortex: Hi-C ENCSR639NRL (52.8 GB) and three marks exist; not downloaded, disk budget.)

Inputs are built as for the human cells (scripts/independent_validation_fetch.py): ENCODE MAPQ30 .hic ->
hicxchange hic2cool at one resolution -> cooler balance --ignore-diags 2 --max-iters 500 -> oe_proper.py;
ChIP as the released mm10 fold-change-over-control bigWig into <dir>/<MARK>.bigwig. Resolutions 10 and
25 kb. Downloads go to .part names with curl -C - and are checked against the portal size and md5.

    python scripts/mouse_transfer_fetch.py inventory
    python scripts/mouse_transfer_fetch.py ref                   # mm10.fa, blacklist, Akita v2 files
    python scripts/mouse_transfer_fetch.py chip [--only heart]
    python scripts/mouse_transfer_fetch.py hic  [--only heart] [--threads 6] [--drop-hic]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HOME = Path.home()
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
VM = HM / "validation_mouse"
ENV = HOME / "miniconda3/envs/hicdiff/bin"
H2C = HOME / "src/hicxchange/build/hicxchange"
CODE = Path(__file__).resolve().parents[1]
CHR1_MM10 = 195471971
RES = [10000, 25000]
CORE7 = ["CTCF", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]
EXTRA = ["RAD21", "SMC3"]
STAGE = Path(os.environ.get("MT_STAGE", str(HOME / "scratch/mt_stage")))
API = "https://www.encodeproject.org"

# label -> Hi-C file, ChIP term preference per mark (first term that has an eligible experiment wins),
# prefer = substring of the experiment's biosample summary that ranks first (same animal / age as the Hi-C)
SAMPLES = {
    "mesc": dict(hic="ENCFF092WXU", exp="ENCSR051JZL", assay="hic", desc="ES-E14 mESC, in situ Hi-C",
                 terms={"CTCF": ["E14TG2a.4", "ES-E14"], "H3K27me3": ["E14TG2a.4", "ES-E14"],
                        **{m: ["ES-E14"] for m in ["H3K27ac", "H3K4me1", "H3K4me3", "H3K36me3", "H3K9me3"]}},
                 prefer=""),
    "mesc_b4": dict(hic="ENCFF092WXU", exp="ENCSR051JZL", assay="hic", desc="ES-E14 Hi-C, ES-Bruce4 ChIP",
                    terms={m: ["ES-Bruce4"] for m in CORE7}, prefer="", hic_of="mesc"),
    "ch12": dict(hic="ENCFF076GYR", exp="ENCSR005UON", assay="hic", desc="CH12.LX, in situ Hi-C, 3 replicates",
                 terms={m: ["CH12.LX"] for m in CORE7 + EXTRA}, prefer=""),
    "heart": dict(hic="ENCFF651SGY", exp="ENCSR997TTD", assay="intact",
                  desc="heart adult (2 months) B6CASTF1/J, intact Hi-C",
                  terms={m: ["heart"] for m in CORE7}, prefer="B6CASTF1/J", adult=True),
}
HIC_EXTRA = {"mesc_rep": dict(hic="ENCFF079MKO", exp="ENCSR819XFA", assay="hic", desc="ES-E14 replicate experiment")}
REF = {
    "mm10.fa.gz": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
    "mm10.chrom.sizes": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes",
    "mm10-blacklist.v2.bed.gz": "https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz",
}
AKITA_GCS = "https://storage.googleapis.com/basenji_hic/3-2021"


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
            if e.code == 404 and "/search/" in url:
                return {"@graph": []}
            log(f"[retry {i}] {url[:100]}: {e}")
            time.sleep(10)
        except Exception as e:                                             # noqa: BLE001
            log(f"[retry {i}] {url[:100]}: {e}")
            time.sleep(10)
    raise RuntimeError(url)


def cached(name, fn):
    p = VM / "meta" / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    v = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v))
    return v


def md5(path, bs=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(bs):
            h.update(b)
    return h.hexdigest()


def download(url, dest: Path, size: int | None = None, md5sum: str | None = None):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dest) + ".part")
    for full in range(3):
        for i in range(80):
            have = part.stat().st_size if part.exists() else 0
            if size is not None and have == size:
                break
            if size is not None and have > size:
                part.unlink()
            if i % 5 == 0:
                log(f"[dl] {dest.name} ({have / 1e9:.2f}/{(size or 0) / 1e9:.2f} GB)")
            r = sh(f"curl -fsSL -C - --speed-limit 100000 --speed-time 120 --connect-timeout 60 "
                   f"-o '{part}' '{url}'", check=False)
            if r.returncode == 0 and size is None:
                break
            if r.returncode != 0:
                time.sleep(15)
        if not part.exists() or (size is not None and part.stat().st_size != size):
            raise RuntimeError(f"{dest.name}: size never matched")
        got = md5(part) if md5sum else None
        if md5sum is None or got == md5sum:
            part.rename(dest)
            log(f"[dl ok] {dest.name} md5 {got}")
            return dest
        log(f"[dl BAD md5] {dest.name}; refetching")
        part.unlink()
    raise RuntimeError(f"{dest.name}: md5 mismatch")


# ------------------------------------------------------------------ ChIP picks
def candidates(term, target):
    q = urllib.parse.urlencode([
        ("type", "File"), ("file_format", "bigWig"), ("assembly", "mm10"), ("status", "released"),
        ("output_type", "fold change over control"), ("biosample_ontology.term_name", term),
        ("target.label", target), ("limit", "all"), ("format", "json"), ("frame", "object")])
    return get_json(f"{API}/search/?{q}")["@graph"]


def experiment(acc):
    def f():
        e = get_json(f"{API}/experiments/{acc}/?format=json&frame=embedded")
        keep = {k: e.get(k) for k in ("accession", "assay_title", "biosample_summary", "perturbed", "date_released")}
        keep["lab"] = (e.get("lab") or {}).get("title")
        keep["treated"] = any(r.get("library", {}).get("biosample", {}).get("treatments")
                              for r in e.get("replicates", []))
        keep["modified"] = any(r.get("library", {}).get("biosample", {}).get("genetic_modifications")
                               for r in e.get("replicates", []))
        return keep
    return cached(f"exp_{acc}", f)


def pick_chip(label):
    """Rule: released mm10 fold-change bigWig of an unperturbed, untreated, unmodified ChIP-seq experiment of
    the listed biosample term (terms in preference order); adult (2 months) for tissues; the strain of the
    Hi-C when the sample asks for it; the file that pools both biological replicates; then the largest
    file below 2.5 GB."""
    s = SAMPLES[label]
    out = {}
    for t, terms in s["terms"].items():
        for term in terms:
            c = cached(f"cand_{term.replace(' ', '_').replace('.', '')}_{t}", lambda: [
                {k: f.get(k) for k in ("accession", "file_size", "md5sum", "href", "dataset",
                                       "biological_replicates")} for f in candidates(term, t)])
            rows = []
            for f in c:
                ex = experiment(f["dataset"].strip("/").split("/")[-1])
                if ex["assay_title"] not in ("Histone ChIP-seq", "TF ChIP-seq"):
                    continue
                if ex["perturbed"] or ex["treated"] or ex["modified"]:
                    continue
                summ = ex.get("biosample_summary") or ""
                if s.get("adult") and not ("adult" in summ and "2 months" in summ):
                    continue
                pref_bad = bool(s["prefer"]) and s["prefer"] not in summ
                pooled = sorted(f.get("biological_replicates") or []) == [1, 2]
                rows.append((pref_bad, not pooled, f["file_size"] > 2.5e9, -f["file_size"], f, ex))
            if rows:
                rows.sort(key=lambda r: r[:4])
                _, _, _, _, f, ex = rows[0]
                out[t] = dict(mark=t, accession=f["accession"], file_size=f["file_size"], md5sum=f["md5sum"],
                              url=API + f["href"], experiment=ex["accession"], term=term,
                              biosample_summary=ex.get("biosample_summary") or "", lab=ex.get("lab"),
                              replicates=f.get("biological_replicates"), n_eligible=len(rows))
                break
    return out


def hic_meta(acc):
    def f():
        d = get_json(f"{API}/files/{acc}/?format=json&frame=object")
        return {k: d.get(k) for k in ("accession", "md5sum", "file_size", "href", "assembly", "output_type",
                                      "dataset", "filter_value")}
    m = cached(f"hic_{acc}", f)
    assert m["assembly"] == "mm10", m
    return m


# ------------------------------------------------------------------ build
def cool_check(path: Path, res: int):
    import cooler
    c = cooler.Cooler(str(path))
    chr1 = int(c.chromsizes.get("chr1", c.chromsizes.get("1", -1)))
    if chr1 != CHR1_MM10:
        raise RuntimeError(f"{path}: chr1 length {chr1}, not mm10")
    if c.binsize != res:
        raise RuntimeError(f"{path}: binsize {c.binsize} != {res}")
    return {"chroms": list(c.chromnames)[:3], "nnz": int(c.info["nnz"]), "binsize": res}


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


def build(label, spec, threads, drop_hic):
    m = hic_meta(spec["hic"])
    OEd = VM / "oe"
    OEd.mkdir(parents=True, exist_ok=True)
    (VM / "stats").mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    todo = [r for r in RES if not (OEd / f"{label}_proper_{r // 1000}kb.cool").exists()]
    if not todo:
        log(f"[have] {label}")
        return
    src = download(API + m["href"], VM / "hic" / f"{spec['hic']}.hic", m["file_size"], m["md5sum"])
    import fcntl                                        # one CPU-heavy conversion at a time (<= 8 threads)
    lockf = open(VM / ".build.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    for res in todo:
        k = res // 1000
        out = OEd / f"{label}_proper_{k}kb.cool"
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
        if res == 25000:
            st.update(total_contacts(bal))
        st.update({"label": label, "hic_file": src.name, "assay": spec["assay"], "experiment": spec["exp"],
                   "hic_bytes": m["file_size"]})
        (VM / "stats" / f"{label}_{k}kb.json").write_text(json.dumps(st, indent=1))
        tmp = STAGE / f"oe_{label}_{k}kb.cool"
        log(f"[oe_proper] {label} {res}")
        sh(f"rm -f '{tmp}' && cd {CODE} && nice -n 10 {ENV}/python scripts/oe_proper.py --input '{bal}' "
           f"--output '{tmp}' --binsize {res} > '{VM}/stats/oe_{label}_{k}kb.log' 2>&1")
        part = out.with_name(out.name + ".part")
        sh(f"cp '{tmp}' '{part}'")
        assert part.stat().st_size == tmp.stat().st_size
        part.rename(out)
        keep = VM / "balanced" / f"{label}_{k}kb.cool"
        keep.parent.mkdir(parents=True, exist_ok=True)
        kp = keep.with_name(keep.name + ".part")
        sh(f"cp '{bal}' '{kp}'")
        assert kp.stat().st_size == bal.stat().st_size
        kp.rename(keep)
        for p in (tmp, bal):
            p.unlink(missing_ok=True)
        log(f"[oe ok] {label} {res} -> {out.name}")
    if drop_hic:
        src.unlink()
        log(f"[hic removed after the 10 and 25 kb cools were built] {src.name}")


def chip(label):
    picks = pick_chip(label)
    d = VM / f"mm10_{label}"
    d.mkdir(parents=True, exist_ok=True)
    for t, r in picks.items():
        download(r["url"], d / f"{t}.bigwig", r["file_size"], r["md5sum"])
    miss = [t for t in CORE7 + EXTRA if t not in picks]
    (d / "missing.json").write_text(json.dumps(miss))
    log(f"[chip ok] {label}: {len(picks)} marks, missing {miss}")


def ref():
    d = VM / "ref"
    for fn, url in REF.items():
        download(url, d / fn)
    fa = d / "mm10.fa"
    if not fa.exists():
        sh(f"gunzip -c '{d / 'mm10.fa.gz'}' > '{fa}.part' && mv '{fa}.part' '{fa}'")
    if not (d / "mm10.fa.fai").exists():
        sh(f"{ENV}/python -c \"from pyfaidx import Fasta; Fasta('{fa}')\"")
    a = VM / "akita_v2"
    for fn in ("data/mm10/sequences.bed", "data/mm10/targets.txt", "data/mm10/contigs.bed",
               "models/f0c0/train/params.json"):
        download(f"{AKITA_GCS}/{fn}", a / Path(fn).name)
    for k in range(8):
        download(f"{AKITA_GCS}/models/f{k}c0/train/model1_best.h5", a / f"f{k}c0_model1_best.h5")
    log("[ref ok]")


def inventory():
    rows = []
    for lab in SAMPLES:
        s = SAMPLES[lab]
        for t, r in pick_chip(lab).items():
            rows.append({"sample": lab, "kind": "chip", "mark": t, "accession": r["accession"],
                         "experiment": r["experiment"], "size_gb": round(r["file_size"] / 1e9, 3),
                         "biosample_summary": r["biosample_summary"], "lab": r["lab"],
                         "replicates": r["replicates"], "n_eligible": r["n_eligible"]})
    for lab, s in {**SAMPLES, **HIC_EXTRA}.items():
        if s.get("hic_of"):
            continue
        m = hic_meta(s["hic"])
        rows.append({"sample": lab, "kind": "hic", "mark": "Hi-C", "accession": s["hic"], "experiment": s["exp"],
                     "size_gb": round(m["file_size"] / 1e9, 3), "biosample_summary": s["desc"], "lab": "",
                     "replicates": "", "n_eligible": ""})
    keys = ["sample", "kind", "mark", "accession", "experiment", "size_gb", "biosample_summary", "lab",
            "replicates", "n_eligible"]
    VM.mkdir(parents=True, exist_ok=True)
    with open(VM / "inventory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tot = {}
    for r in rows:
        tot[(r["sample"], r["kind"])] = tot.get((r["sample"], r["kind"]), 0) + r["size_gb"]
        print(r["sample"], r["kind"], r["mark"], r["accession"], r["experiment"], r["size_gb"], r["replicates"],
              (r["biosample_summary"] or "")[:60])
    for k, v in sorted(tot.items()):
        print(k, round(v, 1), "GB")
    print("TOTAL", round(sum(tot.values()), 1), "GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["inventory", "ref", "chip", "hic"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--drop-hic", action="store_true", help="delete the .hic after both cools exist")
    a = ap.parse_args()
    VM.mkdir(parents=True, exist_ok=True)
    if a.stage == "inventory":
        inventory()
    elif a.stage == "ref":
        ref()
    elif a.stage == "chip":
        for lab in (a.only or list(SAMPLES)):
            chip(lab)
    else:
        allhic = {k: v for k, v in {**SAMPLES, **HIC_EXTRA}.items() if not v.get("hic_of")}
        for lab in (a.only or list(allhic)):
            build(lab, allhic[lab], a.threads, a.drop_hic)


if __name__ == "__main__":
    main()
