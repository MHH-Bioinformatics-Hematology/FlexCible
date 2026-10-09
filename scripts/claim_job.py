"""Claim a training job on the shared drive so that two machines never run it twice.

Both computers write to the same runs/ folders on the shared drive. A job (a run
name such as all12honest_drop_meth_25kb_s0) is claimed before it starts:

    python scripts/claim_job.py claim  <run>   exit 0: claimed by this host, start the job
                                               exit 1: already claimed by another host (or this one), skip it
                                               exit 2: the run is finished (--done-path exists), skip it
    python scripts/claim_job.py status <run>   prints the claim (host, pid, time) or "free"; exit 0 if free
    python scripts/claim_job.py release <run>  removes this host's claim (failed or interrupted job)
    python scripts/claim_job.py list           prints all claims

Files, in <claims dir> (default $HICMULTI_HM/runs/multicell_reference/claims):
    <run>.claim                 the arbiter, created with O_CREAT|O_EXCL (atomic on SMB: the
                                server refuses the second create); content is JSON
                                {host, pid, time, cmd}
    <run>.claimed_by_<host>     the visible marker named by the convention, created after the
                                arbiter by the winner only

A claim is never removed automatically on success: a finished job stays claimed, so neither
queue starts it again. A job that failed is released by hand (release) or by the queue that
owns it. A claim older than --stale-hours (default 0 = never) is reported by status/list as
STALE but is still never taken over automatically.

The local scheduler does not call this script. Use it in the wrapper of every job that
the second machine starts, and in the local queue only when the local queue is changed to
honour it (see REMOTE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
DEFAULT_DIR = HM / "runs/multicell_reference/claims"


def host() -> str:
    return os.environ.get("HICMULTI_HOST") or socket.gethostname().split(".")[0]


def read_claim(d: Path, run: str):
    p = d / f"{run}.claim"
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"host": "?", "pid": 0, "time": "unreadable"}


def claim(d: Path, run: str, done_path: str | None, cmd: str) -> int:
    if done_path and Path(done_path).exists():
        print(f"{run}: finished ({done_path} exists)")
        return 2
    d.mkdir(parents=True, exist_ok=True)
    arb = d / f"{run}.claim"
    try:
        fd = os.open(arb, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        c = read_claim(d, run) or {}
        print(f"{run}: already claimed by {c.get('host')} at {c.get('time')}")
        return 1
    info = {"host": host(), "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd}
    with os.fdopen(fd, "w") as fh:
        json.dump(info, fh)
        fh.flush()
        os.fsync(fh.fileno())
    (d / f"{run}.claimed_by_{host()}").write_text(info["time"] + "\n")
    print(f"{run}: claimed by {host()}")
    return 0


def release(d: Path, run: str) -> int:
    c = read_claim(d, run)
    if c is None:
        print(f"{run}: no claim")
        return 0
    if c.get("host") != host():
        print(f"{run}: claimed by {c.get('host')}, not released by {host()}")
        return 1
    (d / f"{run}.claimed_by_{host()}").unlink(missing_ok=True)
    (d / f"{run}.claim").unlink(missing_ok=True)
    print(f"{run}: released")
    return 0


def age_hours(c) -> float | None:
    try:
        return (time.time() - time.mktime(time.strptime(c["time"], "%Y-%m-%d %H:%M:%S"))) / 3600
    except (KeyError, ValueError, TypeError):
        return None


def describe(c, stale_h) -> str:
    a = age_hours(c)
    tag = " STALE" if stale_h and a is not None and a > stale_h else ""
    return f"{c.get('host')} since {c.get('time')} (pid {c.get('pid')}){tag}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["claim", "status", "release", "list"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--dir", default=os.environ.get("HICMULTI_CLAIMS", str(DEFAULT_DIR)))
    ap.add_argument("--done-path", default=None,
                    help="claim exits 2 when this path exists (e.g. <run dir>/checkpoint-best/progress.json)")
    ap.add_argument("--cmd", default="", help="free text stored in the claim")
    ap.add_argument("--stale-hours", type=float, default=0.0)
    a = ap.parse_args()
    d = Path(a.dir)
    if a.action == "list":
        for p in sorted(d.glob("*.claim")):
            print(f"{p.name[:-6]:60s} {describe(read_claim(d, p.name[:-6]) or {}, a.stale_hours)}")
        return 0
    if not a.run:
        ap.error("run name required")
    if a.action == "claim":
        return claim(d, a.run, a.done_path, a.cmd)
    if a.action == "release":
        return release(d, a.run)
    c = read_claim(d, a.run)
    print(f"{a.run}: free" if c is None else f"{a.run}: claimed by {describe(c, a.stale_hours)}")
    return 0 if c is None else 1


if __name__ == "__main__":
    sys.exit(main())
