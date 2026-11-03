#!/usr/bin/env bash
# FlexCible quickstart: toy data -> precompute -> train -> predict -> evaluate -> plot, on the CPU in minutes.
#   bash examples/quickstart/run.sh [folder]      (default folder: ./toy)
set -euo pipefail
ROOT=$(realpath "${1:-toy}")
HERE=$(cd "$(dirname "$0")" && pwd)
CFG=$ROOT/quickstart.yaml

python "$HERE/make_toy_data.py" "$ROOT"
flexcible precompute --config "$CFG"
flexcible train --config "$CFG"
# chromosome 2 (the test chromosome: even chromosome numbers) of cell A
flexcible predict --config "$CFG" --sample cellA --chroms test --out "$ROOT/pred_cellA_chr2.cool" --tta
flexcible evaluate --config "$CFG" --sample cellA --pred "$ROOT/pred_cellA_chr2.cool" --chroms test \
    --max-dist-mb 0.05 --out "$ROOT/eval_cellA.json"
flexcible plot --config "$CFG" --pred "$ROOT/pred_cellA_chr2.cool" --region chr2:0-100000 \
    --out "$ROOT/pred_cellA_chr2.png"
# a "new" cell: only its bigWigs are given; the reference comes from the embedded consensus
flexcible predict --config "$CFG" --bigwig-dir "$ROOT/cellB" --cell newcell --chroms test \
    --absent-track H3K27ac --out "$ROOT/pred_newcell_ctcf_only_chr2.cool"
echo "done: $ROOT"
