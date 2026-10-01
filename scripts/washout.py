"""Cohesin recovery after auxin washout as a held-out perturbation test.

Rao et al. 2017 (HCT116 RAD21-mAC) measured Hi-C 20, 40, 60 and 180 min after
withdrawing auxin from 6 h treated cells (4DN 4DNESCOJ3ADI, 4DNES7RDRS69,
4DNES4KDIQNN, 4DNESOGIV7NG; GRCh38 mcools of the 4DN pipeline). No ChIP
exists at these time points: Kim et al. 2024 measured RAD21 and CTCF only
untreated, after 6 h auxin and at 9 h and 12 h washout. The washout maps are
therefore never trained on; the models predict them only approximately:

  RAD21 interpolated   every input from the untreated degron ChIP, except
                       RAD21 = (1 - f) * auxin + f * untreated in raw signal,
                       for f in 0, 0.25, 0.5, 0.75, 1 (f = 0: the 6 h auxin
                       RAD21 track; f = 1: the untreated inputs)
and compare the predicted loop strength at the loops of the untreated map
with the measured loop strength at each time point. Reported per model: the
measured recovery of loop strength (as a fraction of the auxin-to-untreated
difference) at 20, 40, 60, 180 min, the predicted recovery as a function of
f, and the f whose prediction matches each time point; this is an
approximation (the real RAD21 occupancy at each time point is unknown).

Stages: fetch (download, md5), oe (single-resolution copies at 10 and 25 kb,
oe_proper.py), predict, score.

    python scripts/washout.py --stage fetch oe
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                      # noqa: E402

REPO, PY, HM = mx.REPO, mx.PY, mx.HM
DATA = HM / "perturb/hct116_rad21_washout"
OUT = HM / "runs/washout"
TIMES = {20: ("4DNFI2NUHMIM", "627bb8ab-3a86-4d99-859e-9cee8850e46c",
              "3035e14735e5d4369ff84357b89cf893"),
         40: ("4DNFINKNVML6", "25590bcb-a616-4ab1-bd45-d5346a973ff9",
              "e12a38e5c894251f190e86671c308997"),
         60: ("4DNFIACWCWO1", "7adb3ed0-7066-4148-9cfe-55ff13630fd4",
              "da72f2d00680e8d36c740c1454702145"),
         180: ("4DNFIBSLD9S9", "cafe7c52-40b1-4ccf-81b7-34b135c4ff5e",
               "69b02f9635ca80f38e5bc6fd95e361bf")}
URL = "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/{uuid}/{acc}.mcool"
RES = [25000, 10000]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "washout.log", "a") as fh:
        fh.write(line + "\n")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def mcool(t: int) -> Path:
    return DATA / f"hct116_rad21mac_washout{t}min.mcool"


def oe_path(t: int, res: int) -> Path:
    return OUT / "oe" / f"washout{t}min_proper_{res // 1000}kb.cool"


def stage_fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for t, (acc, uuid, m) in TIMES.items():
        dst = mcool(t)
        if dst.exists():
            continue
        part = dst.with_name(dst.name + ".part")
        log(f"fetch {t} min {acc}")
        rc = subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "5", "-o", str(part),
                             URL.format(uuid=uuid, acc=acc)]).returncode
        if rc or md5(part) != m:
            raise SystemExit(f"download of {acc} failed (rc {rc}) or md5 mismatch")
        part.rename(dst)
        log(f"fetched {dst.name} md5 ok")


def stage_oe():
    for t in TIMES:
        for res in RES:
            out = oe_path(t, res)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part.cool")
            tmp.unlink(missing_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="4")
            with open(OUT / "oe" / f"washout{t}min_{res // 1000}kb.log", "w") as fh:
                rc = subprocess.run([PY, "scripts/oe_proper.py", "--input", str(mcool(t)),
                                     "--output", str(tmp), "--binsize", str(res)],
                                    cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc:
                raise SystemExit(f"oe_proper failed for {t} min at {res}")
            tmp.rename(out)
            log(f"oe {t} min {res}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", nargs="+", required=True)
    a = ap.parse_args()
    for st in a.stage:
        globals()[f"stage_{st}"]()


if __name__ == "__main__":
    main()
