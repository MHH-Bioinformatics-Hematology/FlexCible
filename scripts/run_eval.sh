#!/usr/bin/env bash
# Full held-out evaluation for one config: predict -> SCC -> loops/TADs -> plots.
# Usage: scripts/run_eval.sh configs/multiome_25kb.yaml [--with-capture]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/hicdiff/bin/python

CFG="$1"
WITH_CAP="${2:-}"
RUN_DIR=$($PY - "$CFG" <<'EOF'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["train"]["output_dir"])
EOF
)

PRED="$RUN_DIR/pred_even.cool"
CAP_ARGS=()
LOOP_CAP_ARGS=()
if [ "$WITH_CAP" = "--with-capture" ]; then
    CAP_ARGS=(--pchic-out "$RUN_DIR/pred_even_capture.cool")
    LOOP_CAP_ARGS=(--pred-capture "$RUN_DIR/pred_even_capture.cool")
fi

$PY -m hicmultiome.predict  --config "$CFG" --chroms test --out "$PRED" "${CAP_ARGS[@]}"
$PY -m hicmultiome.evaluate --config "$CFG" --pred "$PRED" --chroms test \
    --out "$RUN_DIR/scc.json"
$PY -m hicmultiome.loops    --config "$CFG" --pred "$PRED" "${LOOP_CAP_ARGS[@]}" \
    --chroms test --out "$RUN_DIR/loops_tads.json"
$PY -m hicmultiome.diagnose --config "$CFG" --chrom 2 --pred "$PRED" \
    --out-dir "$RUN_DIR"
echo "=== $CFG done -> $RUN_DIR ==="
