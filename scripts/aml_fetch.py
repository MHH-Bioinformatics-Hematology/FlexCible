"""ChIP inputs for the AML test cells from ChIP-Atlas (hg38 bigWig per SRX).

ChIP-Atlas realigns every public SRX to hg38 with one pipeline and serves an
RPM-normalised coverage bigWig per experiment. The training cells use ENCODE
"fold change over control" bigWigs instead, so each ChIP-Atlas track is
rescaled before use (stage convert):

  genome_mean  divide by the track's genome-wide mean (chr1-22, X, all bases),
               then multiply by the genome-wide mean of the training cells'
               fold-change bigWigs of the same mark (the scale the network's
               TrackNormalizer was fitted on; 0.28 to 0.68 per track). A
               genome-wide constant, so relative signal is unchanged.
               Written on a 200 bp grid with uncovered bases as 0 (see
               scale_bigwig).

The K562 set (`picks_k562ca.tsv`) takes the same ENCODE experiments as
`hg38_k562` for 11 of 13 marks through ChIP-Atlas, so K562 predicted from
both input sets measures the effect of the input processing alone.

Picks (one experiment per mark, untreated or control arm, deepest with the
most peaks) are TSVs `mark srx source note` in the AML data folder; marks
without a pick are written to missing.json and the model fills them with the
absent value.

    python scripts/aml_fetch.py --cells molm13 u937 k562ca --stage chip convert
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
AML = HM / "aml"
URL = "https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/{srx}.bw"
ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
MAIN = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
STEP = 200                         # bp grid of the rescaled bigWigs
TRAIN_DIRS = [Path.home() / "data/hic-multiome/hg38_gm12878"] + [
    HM / f"hg38_{c}" for c in ("imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
                               "gm23248", "mcf7", "cd14mono", "pc9")]


def picks(cell: str) -> list[dict]:
    with open(AML / f"picks_{cell}.tsv") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def remote_size(url: str) -> int | None:
    out = subprocess.run(["curl", "-sIL", "--max-time", "120", url], capture_output=True,
                         text=True).stdout
    sizes = [int(l.split()[1]) for l in out.splitlines()
             if l.lower().startswith("content-length:")]
    return sizes[-1] if sizes else None


def check_bigwig(path: Path) -> bool:
    import pyBigWig
    try:
        bw = pyBigWig.open(str(path))
        ok = "chr1" in bw.chroms() and bw.header()["nBasesCovered"] > 0
        ok = ok and bw.stats("chr1", 0, 10_000_000, type="mean") is not None
        bw.close()
        return ok
    except Exception:                                   # noqa: BLE001
        return False


def fetch(srx: str) -> Path:
    """Download one ChIP-Atlas bigWig: resumable, retried, stall-aborted, size
    and bigWig-structure checked, renamed into place only when complete."""
    dest = AML / "bw" / f"{srx}.bw"
    if dest.exists() and check_bigwig(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(srx=srx)
    want = remote_size(url)
    part = Path(str(dest) + ".part")
    for _ in range(6):
        subprocess.run(["curl", "-fL", "-sS", "--retry", "10", "--retry-delay", "20",
                        "--retry-all-errors", "--speed-limit", "20000", "--speed-time", "120",
                        "-C", "-", "-o", str(part), url])
        if part.exists() and (want is None or part.stat().st_size == want):
            break
    if want is not None and (not part.exists() or part.stat().st_size != want):
        raise SystemExit(f"{srx}: incomplete download")
    if not check_bigwig(part):
        raise SystemExit(f"{srx}: not a readable bigWig")
    part.rename(dest)
    return dest


def genome_mean(path: Path) -> float:
    """Mean signal over every base of chr1-22 and X (uncovered bases count 0),
    from the bigWig's whole-chromosome sums."""
    import pyBigWig
    bw = pyBigWig.open(str(path))
    chroms = bw.chroms()
    tot = n = 0.0
    for c in MAIN:
        if c not in chroms:
            continue
        L = chroms[c]
        # pyBigWig's zoom-level "sum" is unreliable; mean x covered fraction x
        # length reproduces the exact sum
        m = bw.stats(c, 0, L, type="mean")[0]
        cov = bw.stats(c, 0, L, type="coverage")[0]
        tot += (m or 0.0) * (cov or 0.0) * L
        n += L
    bw.close()
    return tot / n


def train_mean(mark: str, cache: dict) -> float:
    """Mean over training cells of the genome-wide mean of their fold-change
    bigWig for this mark (the scale the network was trained on)."""
    if mark not in cache:
        vals = [genome_mean(d / f"{mark}.bigwig") for d in TRAIN_DIRS
                if (d / f"{mark}.bigwig").exists()]
        cache[mark] = float(np.mean(vals))
        print(f"  training mean {mark}: {cache[mark]:.4f} over {len(vals)} cells", flush=True)
    return cache[mark]


def binned_mean(bw, chrom: str, step: int, all_bases: bool = False) -> np.ndarray:
    """Mean over the covered bases of each `step` bp bin of one chromosome (NaN
    where no base is covered), from the bigWig's intervals via prefix
    integrals; equivalent to bw.stats(type="mean", nBins=...) and much faster.
    all_bases: mean over every base of the bin (uncovered bases count 0)."""
    L = bw.chroms()[chrom]
    nb = L // step
    iv = bw.intervals(chrom)
    if nb == 0:
        return np.zeros(0)
    if not iv:
        return np.full(nb, np.nan)
    a = np.asarray(iv, dtype=np.float64)
    s, e, v = a[:, 0], a[:, 1], a[:, 2]
    cum_f = np.concatenate([[0.0], np.cumsum(v * (e - s))])
    cum_c = np.concatenate([[0.0], np.cumsum(e - s)])
    x = np.arange(nb + 1, dtype=np.float64) * step

    def integral(cum, val):
        i = np.searchsorted(e, x, side="right")         # intervals fully left of x
        part = np.zeros_like(x)
        inside = (i < len(s)) & (s[np.minimum(i, len(s) - 1)] < x)
        j = i[inside]
        part[inside] = (val[j] if val is not None else 1.0) * (x[inside] - s[j])
        return cum[i] + part

    F = integral(cum_f, v)
    C = integral(cum_c, None)
    num, den = np.diff(F), np.diff(C)
    if all_bases:
        return num / step
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def scale_bigwig(src: Path, dest: Path, factor: float, step: int = STEP) -> None:
    """Write src * factor to dest on a `step` bp grid, via a temporary file.
    Each step holds the mean over all its bases, uncovered bases counting 0:
    the ChIP-Atlas coverage bigWigs omit zero-coverage runs (15 to 80% of the
    genome is uncovered), while the ENCODE fold-change bigWigs cover every
    base, and the model bins with the mean over covered bases, so without the
    zero fill a sparsely covered bin would read as enriched. The model bins
    tracks at >= 2 kb, a multiple of the step."""
    import pyBigWig
    bi = pyBigWig.open(str(src))
    chroms = [(c, L) for c, L in bi.chroms().items() if c.startswith("chr") and "_" not in c]
    tmp = Path(str(dest) + ".tmp")
    bo = pyBigWig.open(str(tmp), "w")
    bo.addHeader(chroms, maxZooms=10)
    for c, L in chroms:
        if L // step == 0:
            continue
        v = binned_mean(bi, c, step, all_bases=True) * factor
        bo.addEntries(c, 0, values=v.astype(np.float64), span=step, step=step)
    bo.close()
    bi.close()
    if not check_bigwig(tmp):
        raise SystemExit(f"{dest}: written bigWig unreadable")
    tmp.rename(dest)


def _scale_job(args):
    src, dest, factor = args
    if not dest.exists():
        scale_bigwig(src, dest, factor)
    print(f"  wrote {dest}", flush=True)


def convert(cell: str, cache: dict, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    out = AML / f"hg38_{cell}"
    out.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for p in picks(cell):
        src = AML / "bw" / f"{p['srx']}.bw"
        dest = out / f"{p['mark']}.bigwig"
        gm = genome_mean(src)
        tm = train_mean(p["mark"], cache)
        factor = tm / gm
        print(f"[{cell}] {p['mark']} {p['srx']}: x{factor:.4g}", flush=True)
        jobs.append((src, dest, factor))
        rows.append({**p, "chipatlas_genome_mean": gm, "train_genome_mean": tm,
                     "factor": factor, "url": URL.format(srx=p["srx"])})
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_scale_job, jobs))
    have = {r["mark"] for r in rows}
    (out / "missing.json").write_text(json.dumps([m for m in ALL13 if m not in have], indent=2))
    with open(out / "picks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "dl.tsv", "w") as fh:
        for r in rows:
            fh.write(f"{r['mark']}\t{r['url']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=["molm13", "u937", "k562ca"])
    ap.add_argument("--stage", nargs="+", default=["chip", "convert"],
                    choices=["chip", "convert"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if "chip" in a.stage:
        for cell in a.cells:
            for p in picks(cell):
                print(f"[{cell}] {p['mark']} {p['srx']}", flush=True)
                fetch(p["srx"])
    if "convert" in a.stage:
        cache: dict = {}
        tm_file = AML / "train_genome_means.json"
        if tm_file.exists():
            cache.update(json.loads(tm_file.read_text()))
        for cell in a.cells:
            convert(cell, cache, a.workers)
            tm_file.write_text(json.dumps(cache, indent=2))


if __name__ == "__main__":
    main()
