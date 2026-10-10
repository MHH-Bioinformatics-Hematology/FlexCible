"""Scan every training log under the runs folder and record, per run, whether the
validation split was honest (train.val_exclude_overlap).

Evidence per run (read only):
  yaml_val_exclude_overlap   value of train.val_exclude_overlap in <run>.yaml
                             ('absent' when the key is missing; before 2026-09-28
                             the default was false)
  log_split_line             'honest, no overlap' when the log carries the line
                             'validation split (honest, no overlap)' written by the
                             honest split; 'val_exclude_overlap: N training windows
                             left' (older honest logs) is reported as 'windows-left';
                             'no-line' otherwise
  log_mtime                  last write of the training log (end of training)
  verdict                    HONEST when the yaml key is true or an honest line is
                             present; LEAKY when the key is false, or absent without
                             an honest line and the log ends before 2026-09-28;
                             UNKNOWN otherwise (driver and scoring logs)

    python results/number_audit/scan_runs.py \
        --runs /mnt/processed/wolffjoa/hic-multiome/runs \
        --out results/number_audit/run_honesty_scan.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re

SKIP = (".eval.", ".precompute.", ".stage.", ".crosscell.", ".score.", ".remote.")
CUT = dt.datetime(2026, 9, 28)


def scan(root: str):
    rows = []
    for log in sorted(glob.glob(os.path.join(root, "*", "*.log"))):
        if any(s in log for s in SKIP):
            continue
        base = log[:-4]
        yaml_path = base + ".yaml"
        yaml_val = ""
        if os.path.exists(yaml_path):
            m = re.search(r"^\s*val_exclude_overlap:\s*(\S+)", open(yaml_path).read(), re.M)
            yaml_val = m.group(1) if m else "absent"
        with open(log, errors="ignore") as f:
            head = f.read(400000)
        if re.search(r"validation split \(honest", head):
            line = "honest, no overlap"
        elif re.search(r"val_exclude_overlap: \d+ (?:of \d+ )?training windows", head):
            line = "windows-left"
        else:
            line = "no-line"
        mt = dt.datetime.fromtimestamp(os.path.getmtime(log))
        if yaml_val == "true" or line != "no-line":
            verdict = "HONEST"
        elif yaml_val == "false" or (yaml_val == "absent" and mt < CUT):
            verdict = "LEAKY"
        else:
            verdict = "UNKNOWN"
        rows.append([os.path.relpath(base, root), yaml_val, line,
                     mt.strftime("%Y-%m-%d %H:%M"), verdict])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/mnt/processed/wolffjoa/hic-multiome/runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = scan(a.runs)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "yaml_val_exclude_overlap", "log_split_line", "log_mtime", "verdict"])
        w.writerows(rows)
    from collections import Counter
    print(len(rows), "runs", dict(Counter(r[4] for r in rows)))


if __name__ == "__main__":
    main()
