"""Biological-replicate Hi-C for 5 cell lines, one identical pipeline, 4 resolutions.

ENCODE in situ Hi-C (all MboI) fastq -> minibwa map --hic (per mate, per lane)
-> HiCExplorer v4 C++ hicBuildMatrix at 2/5/10/25 kb (one mcool per replicate)
-> cooler ICE balance -> oe_proper (distance-normalized O/E) -> per-region Pearson
rep1 vs rep2 on the 413 Akita test regions, per cell and resolution.

One sequencing lane-pair per replicate (K562 rep2: all 8 small lanes, joined) so
the depth is comparable across cells. Every stage is resumable: finished files are
skipped, and each fastq is deleted once its BAM exists.

    python scripts/replicates_all_cells.py
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path("/home/mh-hannover.local/wolffjoa")
ENV = HOME / "miniconda3/envs/hicdiff/bin"
MINIBWA = HOME / "src/bwa-mem3/bench/minibwa"
HBM = HOME / "src/HiCExplorer-v4/cpp/build/tools/hicBuildMatrix"
CODE = HOME / "src/hic-multiome-unet"
IDX = "/mnt/processed/wolffjoa/hic-multiome/minibwa_index/hg38"
W = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_raw")
OUT = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_oe")
RES = [2000, 5000, 10000, 25000]

K562_R2 = [("ENCFF017YRC", "ENCFF740JAL"), ("ENCFF077DJW", "ENCFF942LOO"),
           ("ENCFF299EZU", "ENCFF272RRK"), ("ENCFF307HYX", "ENCFF570ZJK"),
           ("ENCFF402DLV", "ENCFF403MXC"), ("ENCFF416MGD", "ENCFF749NNL"),
           ("ENCFF681AQK", "ENCFF375MOP"), ("ENCFF825VCF", "ENCFF991GET")]
REPS = {  # (cell, rep): ([R1 lanes], [R2 lanes]) -- same lane order for both mates
    ("hmec", "rep1"): (["ENCFF055UHA"], ["ENCFF416YDZ"]),
    ("hmec", "rep2"): (["ENCFF126WCO"], ["ENCFF244GNV"]),
    ("imr90", "rep1"): (["ENCFF346EOS"], ["ENCFF983FKT"]),
    ("imr90", "rep2"): (["ENCFF110XAL"], ["ENCFF490HFV"]),
    ("huvec", "rep1"): (["ENCFF678PIY"], ["ENCFF411KFA"]),
    ("huvec", "rep2"): (["ENCFF765YLB"], ["ENCFF544HRQ"]),
    ("k562", "rep1"): (["ENCFF245VNR"], ["ENCFF887BPM"]),
    ("k562", "rep2"): ([a for a, _ in K562_R2], [b for _, b in K562_R2]),
    ("gm12878", "rep1"): (["ENCFF991GHV"], ["ENCFF075YWD"]),
    ("gm12878", "rep2"): (["ENCFF416CVP"], ["ENCFF997OUB"]),
}
_lock = threading.Lock()


def log(msg):
    with _lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def sh(cmd):
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED ({r.returncode}): {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def fq(acc): return W / f"{acc}.fastq.gz"
def bam(acc): return W / f"{acc}.bam"


def download(acc):
    if fq(acc).exists() or bam(acc).exists():
        return
    log(f"[dl] {acc}")
    sh(f"curl -fsSL -C - --retry 8 --retry-delay 15 -o {fq(acc)}.part "
       f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.fastq.gz")
    sh(f"mv {fq(acc)}.part {fq(acc)}")
    log(f"[dl ok] {acc}")


def mapacc(acc):
    if bam(acc).exists():
        return
    while not fq(acc).exists():
        time.sleep(30)
    log(f"[map] {acc}")
    sh(f"{MINIBWA} map --hic -t 15 {IDX} {fq(acc)} 2>/dev/null "
       f"| {ENV}/samtools view -b -@ 2 -o {bam(acc)}.part -")
    sh(f"mv {bam(acc)}.part {bam(acc)} && rm -f {fq(acc)}")
    log(f"[map ok] {acc}")


def oe_path(cell, rep, res):
    return OUT / f"{cell}_{rep}_ice_{res // 1000}kb.cool"


def build(cell, rep):
    if all(oe_path(cell, rep, r).exists() for r in RES):
        log(f"[have] {cell} {rep}"); return
    r1s, r2s = REPS[(cell, rep)]
    for a in r1s + r2s:
        while not bam(a).exists():
            time.sleep(60)
    if len(r1s) == 1:
        b1, b2 = bam(r1s[0]), bam(r2s[0])
    else:                                    # join lanes, same order for both mates
        b1, b2 = W / f"{cell}_{rep}_R1.bam", W / f"{cell}_{rep}_R2.bam"
        for dst, lanes in ((b1, r1s), (b2, r2s)):
            if not dst.exists():
                sh(f"{ENV}/samtools cat -o {dst}.part " + " ".join(str(bam(a)) for a in lanes)
                   + f" && mv {dst}.part {dst}")
    mc = W / f"{cell}_{rep}.mcool"
    if cell == "hmec" and rep == "rep1" and not mc.exists() and (W / "test_multires.mcool").exists():
        sh(f"mv {W/'test_multires.mcool'} {mc}")
    if not mc.exists():
        log(f"[matrix] {cell} {rep}")
        sh(f"{HBM} --samFiles {b1} {b2} -o {mc}.part.mcool --QCfolder {W}/qc_{cell}_{rep} "
           f"--binSize {' '.join(map(str, RES))} --restrictionCutFile {W}/gatc_sites.bed "
           f"--restrictionSequence GATC --danglingSequence GATC --minMappingQuality 15 "
           f"--threads 8 --skipDuplicationCheck --noPlot")
        sh(f"mv {mc}.part.mcool {mc}")
    for r in RES:
        if oe_path(cell, rep, r).exists():
            continue
        log(f"[balance+oe] {cell} {rep} {r}")
        sh(f"{ENV}/cooler balance --force -p 8 --ignore-diags 2 --max-iters 500 {mc}::/resolutions/{r}")
        tmp = oe_path(cell, rep, r).with_suffix(".part.cool")
        sh(f"rm -f {tmp} && cd {CODE} && {ENV}/python scripts/oe_proper.py --input {mc} "
           f"--output {tmp} --binsize {r} && mv {tmp} {oe_path(cell, rep, r)}")
    log(f"[built] {cell} {rep}")


def main():
    order = list(REPS)
    accs = [a for k in order for pair in zip(*REPS[k]) for a in pair]
    dl_pool, map_pool, build_pool = (ThreadPoolExecutor(3), ThreadPoolExecutor(2),
                                     ThreadPoolExecutor(3))
    futs = [dl_pool.submit(download, a) for a in accs]
    futs += [map_pool.submit(mapacc, a) for a in accs]
    futs += [build_pool.submit(build, c, r) for c, r in order]
    bad = 0
    for f in futs:
        try:
            f.result()
        except Exception as e:                # keep the rest going, report at end
            bad += 1; log(f"[ERROR] {e}")
    log(f"all stages finished, {bad} errors")
    sh(f"cd {CODE} && {ENV}/python scripts/replicate_corr.py > {W}/replicate_corr.txt")
    print(open(W / "replicate_corr.txt").read(), flush=True)
    print("REPLICATES_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
