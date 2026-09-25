"""Acquire and prepare every usable ENCODE cell (runs/encode_survey) plus the deep
intact Hi-C replicate pairs.

Hi-C: ENCODE MAPQ30 .hic -> in-house C++ hic2cool (2, 5, 10 and 25 kb) -> checks (hg38 chr1 length, bin size, total contacts) -> cooler ICE
(--ignore-diags 2 --max-iters 500) -> scripts/oe_proper.py, exactly the path of
scripts/replicates_all_cells.py. A549 uses the untreated in situ MboI experiment
ENCSR662QKG, whose four biological-replicate .hic files are converted and summed
(cooler merge) into one map, since ENCODE has no merged file for it.

ChIP: the untreated/unmodified fold-change bigWig picks of
runs/encode_survey/recommended_bigwigs.csv, into hg38_<cell>/<MARK>.bigwig with
dl.tsv, missing.json and picks.csv (accession, md5, biosample summary).

Every download goes to a .part name with `curl -C -` in a retry loop and is
checked against ENCODE's size and md5 before it is renamed. Every derived file is
written under a temporary name and renamed when complete, so a rerun resumes at
the first missing output. A .hic is deleted once all its cools are verified.

Outputs
  crosscell_oe/<cell>_proper_<res>kb.cool                training/test targets
  crosscell_oe/intact_<pair>_<rep>_ice_<res>kb.cool       replicate pairs
  encode_hic/<label>_<res>kb.cool                        balanced raw maps
  encode_hic/stats/<label>_<res>kb.json                  checks and contacts

    python scripts/encode_expand.py [--only hepg2 a549 ...] [--no-chip]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path("/home/mh-hannover.local/wolffjoa")
ENV = HOME / "miniconda3/envs/hicdiff/bin"
H2C = HOME / "src/hic2cool/build/hic2cool"
CODE = HOME / "src/hic-multiome-unet"
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
ENC = HM / "encode_hic"
OE = HM / "crosscell_oe"
SURVEY = HM / "runs/encode_survey"
CHR1 = 248956422
REP_RES = [2000, 5000, 10000, 25000]

# label: (.hic accessions, assay, experiment, role, resolutions)
# assay: insitu = in situ Hi-C, intact = intact Hi-C (MNase and/or DNaseI)
HIC = {
    "hepg2": (["ENCFF306VTV"], "insitu", "ENCSR194SRI", "train", REP_RES),
    "a549": (["ENCFF121YPY", "ENCFF876LAW", "ENCFF675SJE", "ENCFF740KVX"],
             "insitu", "ENCSR662QKG", "train", REP_RES),
    "mcf7": (["ENCFF420JTA"], "intact", "ENCSR660LPJ", "train", REP_RES),
    "nhek": (["ENCFF148AQB"], "insitu", "ENCSR869CSI", "train", REP_RES),
    "gm23248": (["ENCFF702IFC"], "insitu", "ENCSR220MNC", "train", REP_RES),
    "cd14mono": (["ENCFF684YCT"], "intact", "ENCSR444SKT", "train", REP_RES),
    "pc9": (["ENCFF181ROW"], "intact", "ENCSR859DRK", "train", REP_RES),
    # deep targets for held-out test cells (HUVEC has no merged in situ file)
    "hmec_deep": (["ENCFF943JRY"], "insitu", "ENCSR711AVS", "test", REP_RES),
    "huvec_deep": (["ENCFF783KQI"], "intact", "ENCSR788FBI", "test", REP_RES),
}
# deep intact Hi-C replicate pairs (runs/encode_survey/replicate_pairs.csv)
PAIRS = [
    ("k562", "ENCSR676BEP", ("r5", "ENCFF740BHI"), ("r7", "ENCFF891MHD")),
    ("gm12878", "ENCSR965OVH", ("r75", "ENCFF159BIG"), ("r76", "ENCFF389TAA")),
    ("gm12878", "ENCSR653JLK", ("r7", "ENCFF358MKI"), ("r9", "ENCFF871WRP")),
    ("gm12878", "ENCSR946AKI", ("r8", "ENCFF619PDD"), ("r12", "ENCFF380XFP")),
    ("k562", "ENCSR178ENS", ("r2", "ENCFF411LXX"), ("r3", "ENCFF943XAA")),
]
for _cell, _exp, *_reps in PAIRS:
    for _r, _acc in _reps:
        HIC[f"{_cell}_{_exp}_{_r}"] = ([_acc], "intact", _exp, "pair", REP_RES)
# a verified copy already on the share (md5 checked before use, never deleted)
REUSE = {"ENCFF943JRY": HM / "crosscell_hic/hmec_rep1.hic"}
# Akita's five cells and K562: 4DN mcools (ICE weights stored by the 4DN
# pipeline at every level); --local builds their proper O/E at 5, 10, 25 kb
LOCAL = {
    "gm12878": HM.parent / "restart_of_prediction_project/originalData/4Dnucleosome/GM12878_4DNFIXP4QG5B.mcool",
    "h1": HM.parent / "restart_of_prediction_project/originalData/4Dnucleosome/H1_4DNFI6HDY7WZ.mcool",
    "k562": HM.parent / "restart_of_prediction_project/originalData/4Dnucleosome/K562_4DNFI18UHVRO.mcool",
    "imr90": HM.parent / "hyperparameter_publication/imr90.mcool",
    "hff": HM / "matrices/hff_microc.mcool",
    "hct116": HM / "matrices/hct116_untreated.mcool",
}
# survey biosample term -> cell label
CHIP_CELLS = {"HepG2": "hepg2", "A549": "a549", "MCF-7": "mcf7", "keratinocyte": "nhek",
              "GM23248": "gm23248", "CD14-positive monocyte": "cd14mono", "PC-9": "pc9"}
ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]

_lock = threading.Lock()
SEM_DL_HIC = threading.Semaphore(4)
SEM_DL_BW = threading.Semaphore(3)
# replicate-pair downloads start once the training/test .hic files are in
NONPAIR_LEFT = [0]
NONPAIR_DONE = threading.Event()
SEM_CONV = threading.Semaphore(2)
SEM_OE = threading.Semaphore(3)


def log(msg):
    with _lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def sh(cmd, check=True):
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"FAILED ({r.returncode}): {cmd}\n{r.stderr[-3000:]}")
    return r


def meta(acc):
    """ENCODE file record (md5sum, file_size, href), cached on disk."""
    cache = ENC / "meta" / f"{acc}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for i in range(10):
        try:
            req = urllib.request.Request(
                f"https://www.encodeproject.org/files/{acc}/?format=json&frame=object",
                headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                d = json.load(fh)
            keep = {k: d.get(k) for k in ("accession", "md5sum", "file_size", "href",
                                          "assembly", "status", "output_type", "dataset",
                                          "file_format")}
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".part")
            tmp.write_text(json.dumps(keep))
            tmp.rename(cache)
            return keep
        except Exception as e:                          # noqa: BLE001
            log(f"[meta retry {i}] {acc}: {e}")
            time.sleep(20)
    raise RuntimeError(f"no metadata for {acc}")


def md5(path, bs=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(bs):
            h.update(b)
    return h.hexdigest()


def download(acc, dest: Path):
    """curl -C - retry loop into a .part file, size + md5 check, rename.

    A .part file the share stops accepting writes on (curl exit 23, a stale
    SMB handle) is abandoned for a fresh .partN name."""
    m = meta(acc)
    if dest.exists():
        return dest
    url = "https://www.encodeproject.org" + m["href"]
    parts = sorted(dest.parent.glob(dest.name + ".part*"),
                   key=lambda p: p.stat().st_size if p.exists() else 0)
    part = parts[-1] if parts else Path(str(dest) + ".part")
    gen = len(parts)
    with (SEM_DL_HIC if dest.suffix == ".hic" else SEM_DL_BW):
        for full in range(3):
            n23 = 0
            for i in range(60):
                have = part.stat().st_size if part.exists() else 0
                if have == m["file_size"]:
                    break
                if have > m["file_size"]:
                    part.unlink()
                if i == 0 or i % 5 == 0:
                    log(f"[dl] {acc} -> {part.name} ({have / 1e9:.2f}/{m['file_size'] / 1e9:.2f} GB)")
                r = sh(f"curl -fsSL -C - --speed-limit 100000 --speed-time 120 "
                       f"--connect-timeout 60 -o '{part}' '{url}'", check=False)
                if r.returncode == 23:
                    n23 += 1
                    if n23 >= 2:
                        log(f"[dl] {acc}: share refuses writes to {part.name}; new file")
                        try:
                            part.unlink()
                        except OSError:
                            pass
                        gen += 1
                        part = Path(f"{dest}.part{gen}")
                        n23 = 0
                elif r.returncode != 0:
                    n23 = 0
                    time.sleep(15)
            if not part.exists() or part.stat().st_size != m["file_size"]:
                raise RuntimeError(f"{acc}: size never matched")
            got = md5(part)
            if got == m["md5sum"]:
                part.rename(dest)
                for p in dest.parent.glob(dest.name + ".part*"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                log(f"[dl ok] {acc} md5 {got}")
                return dest
            log(f"[dl BAD md5] {acc} {got} != {m['md5sum']}; refetching")
            part.unlink()
    raise RuntimeError(f"{acc}: md5 mismatch after 3 full downloads")


def hic_file(acc):
    src = REUSE.get(acc)
    if src and src.exists():
        ok = ENC / "meta" / f"{acc}.reuse_ok"
        if ok.exists() or md5(src) == meta(acc)["md5sum"]:
            ok.write_text(str(src))
            return src, False                            # not ours: never delete
        log(f"[reuse] {src} md5 differs from {acc}; downloading")
    return download(acc, ENC / f"{acc}.hic"), True


def cool_stats(path: Path, res: int, count=True) -> dict:
    """hg38 check, bin size, and (count=True) total and cis contacts (streamed
    pixel sum; resolution independent, so only taken at the coarsest level)."""
    import cooler
    import numpy as np
    c = cooler.Cooler(str(path))
    cs = c.chromsizes
    chr1 = int(cs.get("chr1", cs.get("1", -1)))
    if chr1 != CHR1:
        raise RuntimeError(f"{path}: chr1 length {chr1}, not hg38")
    if c.binsize != res:
        raise RuntimeError(f"{path}: binsize {c.binsize} != {res}")
    bins = c.bins()[:]["chrom"].cat.codes.values if hasattr(
        c.bins()[:]["chrom"], "cat") else None
    px = c.pixels()
    tot = cis = 0.0
    n = c.info["nnz"]
    if not count:
        return {"path": str(path), "binsize": int(c.binsize), "chr1": chr1, "nnz": int(n),
                "assembly": c.info.get("genome-assembly")}
    for i in range(0, n, 25_000_000):
        p = px[i:i + 25_000_000]
        v = p["count"].values.astype(np.float64)
        tot += float(v.sum())
        if bins is not None:
            same = bins[p["bin1_id"].values] == bins[p["bin2_id"].values]
            cis += float(v[same].sum())
    return {"path": str(path), "binsize": int(c.binsize), "chr1": chr1, "nnz": int(n),
            "contacts": tot, "cis_contacts": cis, "assembly": c.info.get("genome-assembly")}


def header_contacts(path: Path):
    import cooler
    st = cooler.Cooler(str(path)).info.get("statistics") or ""
    for line in st.splitlines():
        if line.startswith("Hi-C Contacts:"):
            return int(line.split(":")[1].split("(")[0].replace(",", "").strip())
    return None


def convert(acc, hic: Path, res: int) -> Path:
    """Unbalanced single-resolution cool of one .hic, verified."""
    out = ENC / f"{acc}_{res // 1000}kb.raw.cool"
    if out.exists():
        return out
    tmp = ENC / f"{acc}_{res // 1000}kb.raw.part.cool"
    with SEM_CONV:
        log(f"[hic2cool] {acc} {res}")
        sh(f"rm -f '{tmp}' && {H2C} convert -r {res} -p 12 -s '{hic}' '{tmp}'")
        cool_stats(tmp, res, count=False)                 # hg38 + binsize checks
        tmp.rename(out)
    log(f"[hic2cool ok] {acc} {res}")
    return out


def final_path(label, res):
    return ENC / f"{label}_{res // 1000}kb.cool"


def oe_path(label, res):
    role = HIC[label][3]
    if role == "pair":
        return OE / f"intact_{label}_ice_{res // 1000}kb.cool"
    return OE / f"{label}_proper_{res // 1000}kb.cool"


def build(label):
    accs, assay, exp, role, resl = HIC[label]
    if all(oe_path(label, r).exists() and (ENC / "stats" / f"{label}_{r // 1000}kb.json").exists()
           for r in resl):
        log(f"[have] {label}")
        return
    hics = {}
    try:
        if role == "pair" and not all(final_path(label, r).exists() for r in resl):
            NONPAIR_DONE.wait()
        for acc in accs:
            if all(final_path(label, r).exists() for r in resl):
                break
            hics[acc] = hic_file(acc)
    finally:
        if role != "pair":
            with _lock:
                NONPAIR_LEFT[0] -= 1
                if NONPAIR_LEFT[0] <= 0:
                    NONPAIR_DONE.set()
    for res in resl:
        fin = final_path(label, res)
        if not fin.exists():
            raws = [convert(acc, hics[acc][0], res) for acc in accs]
            part = ENC / f"{label}_{res // 1000}kb.part.cool"
            sh(f"rm -f '{part}'")
            if len(raws) == 1:
                raws[0].rename(part)
            else:                                        # sum biological replicates
                log(f"[merge] {label} {res} from {len(raws)} replicates")
                sh(f"{ENV}/cooler merge '{part}' " + " ".join(f"'{r}'" for r in raws))
            with SEM_OE:
                log(f"[balance] {label} {res}")
                sh(f"{ENV}/cooler balance --force -p 8 --ignore-diags 2 --max-iters 500 '{part}'")
            part.rename(fin)
            for r in raws:
                if r.exists():
                    r.unlink()
        sp = ENC / "stats" / f"{label}_{res // 1000}kb.json"
        if not sp.exists():
            st = cool_stats(fin, res, count=(res == max(resl)))
            st.update({"label": label, "accessions": accs, "assay": assay,
                       "experiment": exp, "role": role,
                       "header_contacts": [header_contacts(fin)] if len(accs) == 1 else None,
                       "encode_md5": {a: meta(a)["md5sum"] for a in accs}})
            import cooler
            w = cooler.Cooler(str(fin)).bins()[:]["weight"]
            st["ice_masked_frac"] = float(w.isna().mean())
            sp.parent.mkdir(parents=True, exist_ok=True)
            tmp = sp.with_suffix(".part")
            tmp.write_text(json.dumps(st, indent=1, default=float))
            tmp.rename(sp)
            log(f"[stats] {label} {res}: {st.get('contacts', float('nan')) / 1e6:.1f} M contacts, "
                f"masked {100 * st['ice_masked_frac']:.1f}%")
        o = oe_path(label, res)
        if not o.exists():
            tmp = o.with_suffix(".part.cool")
            with SEM_OE:
                log(f"[oe_proper] {label} {res}")
                sh(f"rm -f '{tmp}' && cd {CODE} && {ENV}/python scripts/oe_proper.py "
                   f"--input '{fin}' --output '{tmp}' --binsize {res} "
                   f"> '{ENC}/logs/oe_{label}_{res // 1000}kb.log' 2>&1")
            tmp.rename(o)
            log(f"[oe ok] {label} {res} -> {o.name}")
    # all cools verified: the .hic files are no longer needed
    for acc in accs:
        for path in [ENC / f"{acc}.hic", *ENC.glob(f"{acc}.hic.part*")]:
            if path.exists():
                path.unlink()
                log(f"[rm] {path.name}")
    log(f"[built] {label}")


def build_local(cell, res):
    """Proper O/E of a 4DN mcool level with its stored weights (read-only)."""
    o = OE / f"{cell}_proper_{res // 1000}kb.cool"
    if o.exists():
        return
    tmp = o.with_suffix(".part.cool")
    with SEM_OE:
        log(f"[oe_proper] {cell} {res} from {LOCAL[cell].name}")
        sh(f"rm -f '{tmp}' && cd {CODE} && {ENV}/python scripts/oe_proper.py "
           f"--input '{LOCAL[cell]}' --output '{tmp}' --binsize {res} "
           f"> '{ENC}/logs/oe_{cell}_{res // 1000}kb.log' 2>&1")
    tmp.rename(o)
    log(f"[oe ok] {cell} {res} -> {o.name}")


def chip(term):
    cell = CHIP_CELLS[term]
    out = HM / f"hg38_{cell}"
    out.mkdir(parents=True, exist_ok=True)
    picks = [r for r in csv.DictReader(open(SURVEY / "recommended_bigwigs.csv"))
             if r["biosample"] == term]
    got = {r["target"]: r for r in picks}
    missing = [t for t in ALL13 if t not in got]
    rows = []
    for t in ALL13:
        if t not in got:
            continue
        m = meta(got[t]["accession"])
        if m.get("assembly") != "GRCh38" or m.get("status") != "released":
            raise RuntimeError(f"{cell} {t}: {m}")
        rows.append({"mark": t, "accession": m["accession"], "md5sum": m["md5sum"],
                     "file_size": m["file_size"], "dataset": m["dataset"],
                     "biosample_summary": got[t]["simple_biosample_summary"],
                     "url": "https://www.encodeproject.org" + m["href"]})
    for fn, text in (("dl.tsv", "".join(f"{r['mark']}\t{r['url']}\n" for r in rows)),
                     ("missing.json", json.dumps(missing, indent=2))):
        tmp = out / (fn + ".part")
        tmp.write_text(text)
        tmp.rename(out / fn)
    with open(out / "picks.csv.part", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "picks.csv.part").rename(out / "picks.csv")
    for r in rows:
        download(r["accession"], out / f"{r['mark']}.bigwig")
    log(f"[chip ok] {cell}: {len(rows)}/13, missing {missing}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="labels (default: all)")
    ap.add_argument("--no-chip", action="store_true")
    ap.add_argument("--no-hic", action="store_true")
    ap.add_argument("--local", action="store_true",
                    help="also build the 5/10/25 kb proper O/E of the 4DN mcools (LOCAL)")
    ap.add_argument("--hic-streams", type=int, default=8,
                    help="parallel .hic downloads (one stream runs at about 5 MB/s)")
    a = ap.parse_args()
    global SEM_DL_HIC
    SEM_DL_HIC = threading.Semaphore(a.hic_streams)
    for d in (ENC, ENC / "stats", ENC / "logs", ENC / "meta"):
        d.mkdir(parents=True, exist_ok=True)
    labels = a.only or list(HIC)
    NONPAIR_LEFT[0] = sum(HIC[lab][3] != "pair" for lab in labels)
    if NONPAIR_LEFT[0] == 0:
        NONPAIR_DONE.set()
    futs = []
    pool = ThreadPoolExecutor(len(labels) + len(CHIP_CELLS) + 3 * len(LOCAL))
    if not a.no_chip:
        futs += [(f"chip {t}", pool.submit(chip, t)) for t in CHIP_CELLS
                 if a.only is None or CHIP_CELLS[t] in a.only]
    if not a.no_hic:
        futs += [(lab, pool.submit(build, lab)) for lab in labels]
    if a.local:
        futs += [(f"local {c} {r}", pool.submit(build_local, c, r))
                 for c in LOCAL for r in (5000, 10000, 25000)]
    bad = 0
    for name, f in futs:
        try:
            f.result()
        except Exception as e:                          # keep the rest going
            bad += 1
            log(f"[ERROR] {name}: {e}")
    log(f"all finished, {bad} errors")
    print("ENCODE_EXPAND_DONE" if bad == 0 else "ENCODE_EXPAND_ERRORS", flush=True)


if __name__ == "__main__":
    main()
