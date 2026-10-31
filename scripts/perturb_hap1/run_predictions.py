"""Run flexcible predict (new-cell mode, ChIP only, no consensus) for WT, each deletion and each control site.
Every command is appended to results/perturb_hap1/commands_predict.sh."""
import sys, os, subprocess, shlex; sys.path.insert(0, os.path.dirname(__file__))
from common import *
MODELS_TO_RUN = sys.argv[1].split(",") if len(sys.argv) > 1 else ["10kb", "5kb"]
PY = str(Path.home() / "miniconda3/envs/hicdiff/bin/flexcible")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
HAVE = {p.stem for p in (OUT / "tracks").glob("*.bigwig")}
ALL = ["Meth", "CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
MISSING = [t for t in ALL if t not in HAVE]
CONS12 = {k: Path(f"/mnt/processed/wolffjoa/hic-multiome/crosscell_oe/consensus12_proper_{k}.cool") for k in ("10kb", "5kb")}  # the 12 training cells; HAP1 is none of them
ctrl = {}
for l in list(open(RES / "control_sites.tsv"))[1:]:
    p = l.rstrip().split("\t"); ctrl[p[0]] = (p[1], int(p[2]), int(p[3]))

def zr(ch, s, e, rad21=False):
    a, b = max(0, s - PAD), e + PAD
    z = [f"CTCF:{ch}:{a}-{b}", f"motif:MA0139.2:{ch}:{a}-{b}"]
    if "RAD21" in HAVE: z.append(f"RAD21:{ch}:{a}-{b}")
    if "SMC3" in HAVE: z.append(f"SMC3:{ch}:{a}-{b}")
    return z

jobs = []
for rl in MODELS_TO_RUN:
    for rn, r in REGIONS.items():
        base = ["--config", str(MODELS[rl]), "--bigwig-dir", str(OUT / "tracks"), "--cell", "hap1", "--chroms", r["chrom"], "--tta",
                "--zero-tracks", ",".join(MISSING), "--batch-size", "4"] + (["--no-reference"] if NOREF else []) + ["--reference", str(CONS12[rl])]
        od = COOLS / rl; od.mkdir(parents=True, exist_ok=True)
        jobs.append((od / f"wt_{rn}.cool", base, []))
        for dn, d in DELETIONS.items():
            if d["region"] != rn: continue
            z = [x for s in d["sites"] for x in zr(SITES[s]["chrom"], SITES[s]["start"], SITES[s]["end"])]
            jobs.append((od / f"del_{dn}.cool", base, z))
            c = ctrl[dn]
            jobs.append((od / f"ctrl_{dn}.cool", base, zr(*c)))
log = open(RES / f"{PFX}commands_predict.sh", "a")
for out, base, z in jobs:
    if out.exists() and out.stat().st_size > 0: continue
    cmd = [PY, "predict", *base, *sum([["--zero-region", x] for x in z], []), "--out", str(out)]
    line = "CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 " + "nice -n 10 " + " ".join(shlex.quote(c) for c in cmd)
    log.write(line + "\n"); log.flush()
    r = subprocess.run(["nice","-n","10",*cmd], env=ENV, capture_output=True, text=True)
    (out.with_suffix(".log")).write_text(r.stdout + r.stderr)
    print(out.name, r.returncode, flush=True)
    if r.returncode: print(r.stderr[-1500:]); sys.exit(1)
