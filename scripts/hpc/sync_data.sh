#!/usr/bin/env bash
# Push the code + every data file the configs reference to the HPC.
#   bash scripts/hpc/sync_data.sh <user@host> <remote-base-dir> [config ...]
# With no configs given, syncs the inputs of every configs/*.yaml.
# Re-runs are cheap (rsync). Paths are recreated 1:1 on the remote side, so
# the configs work unchanged.
set -euo pipefail
HOST="${1:?usage: sync_data.sh <user@host> <remote-base> [config ...]}"
BASE="${2:?usage: sync_data.sh <user@host> <remote-base> [config ...]}"
shift 2
cd "$(dirname "$0")/../.."
CFGS=("$@")
[ "${#CFGS[@]}" -gt 0 ] || CFGS=(configs/*.yaml)

mapfile -t FILES < <(python3 - "${CFGS[@]}" <<'EOF'
import sys, yaml, os
paths = set()
for cfg in sys.argv[1:]:
    c = yaml.safe_load(open(cfg))
    d = c.get("data", {})
    samples = d.get("samples") or [{"cool_path": d.get("cool_path"),
                                    "bigwig_dir": d.get("bigwig_dir")}]
    for s in samples:
        for k in ("cool_path", "bigwig_dir", "fasta", "capture_interactions"):
            if s.get(k):
                paths.add(s[k])
    seq = c.get("sequence") or {}
    if seq.get("fasta"):
        paths.add(seq["fasta"])
        fai = seq["fasta"] + ".fai"
        if os.path.exists(fai):
            paths.add(fai)
    mot = c.get("motifs") or {}
    if mot.get("dir"):
        paths.add(mot["dir"])
for p in sorted(paths):
    print(p)
EOF
)

echo "syncing repo -> $HOST:$BASE/hic-multiome-unet"
rsync -a --exclude cache --exclude runs --exclude '__pycache__' \
    ./ "$HOST:$BASE/hic-multiome-unet/"

for p in "${FILES[@]}"; do
    [ -e "$p" ] || { echo "MISSING locally: $p"; continue; }
    echo "syncing $p"
    rsync -aR "$p" "$HOST:/"
done
echo "done. On the HPC: cd $BASE/hic-multiome-unet && bash scripts/hpc/setup_env.sh"
