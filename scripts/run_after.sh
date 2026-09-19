#!/usr/bin/env bash
# Start a runner spec once a specific other runner has finished.
#   nohup setsid bash scripts/run_after.sh <spec.json> <wait-pattern> <gpu-ids> &
set -uo pipefail
cd "$(dirname "$0")/.."
SPEC="${1:?spec}"; WAIT="${2:?wait pattern}"; GPUS="${3:-0}"
PY=~/miniconda3/envs/hicdiff/bin/python
while pgrep -f "runner\.py --spec $WAIT" > /dev/null; do sleep 60; done
echo "starting $SPEC on GPU $GPUS at $(date +%H:%M:%S)"
exec $PY scripts/runner.py --spec "$SPEC" --gpu-ids "$GPUS" --cpu-slots 2
