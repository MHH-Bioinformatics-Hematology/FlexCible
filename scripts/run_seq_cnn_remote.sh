#!/usr/bin/env bash
# One-command run of the sequence CNN study (registry section 42) on a machine with one GPU.
#
#   DATA_ROOT=/path/to/hic-multiome bash scripts/run_seq_cnn_remote.sh
#
# Environment variables
#   DATA_ROOT   required. Layout of /mnt/processed/wolffjoa/hic-multiome:
#                 hg38_<cell>/<MARK>.bigwig   (gm12878 imr90 h1 hepg2 a549 gm23248 cd14mono k562 hmec)
#                 meth/<cell>/Meth.bigwig
#                 genome/hg38.fa              (+ hg38.fa.fai, created if the folder is writable)
#                 annotation/hg38/hg38-blacklist.v2.bed.gz   (optional)
#               hg38_gm12878 sits under ~/data/hic-multiome on the origin machine: copy or link it in.
#   OUT_ROOT    run folders, cache, logs (default ./runs/seq_cnn)
#   PYTHON      interpreter (default python)
#   MODES       modes to run, in order (default "seq seq_meth meth")
#   EXTRA       extra arguments for seq_cnn_chip.py, e.g. "--steps 30000" or "--config my.yaml"
#   CUDA_VISIBLE_DEVICES  choose the GPU
#
# Resumable: the cache, finished trainings and finished evaluations are skipped; an interrupted
# training resumes from OUT_ROOT/<mode>/ckpt_last.pt (written every 500 steps). Start the same
# command again after any interruption.
#
# Expected runtime on one RTX 4090: cache about 1 h (CPU, bigWig reads), then about 1.5 to 2.5 h per
# mode with the default 20000 steps at batch 32; total roughly 5 to 8 h. RAM below 8 GB, disk 2 GB cache.
#
# Copy back (from the folder holding this script's repository copy):
#   scp -r user@host:<repo>/results/seq_cnn ./results/
# files: seq_cnn_results.csv (all rows), seq_cnn_summary.csv, seq_cnn_partition.csv;
# optional: OUT_ROOT/*/train_log.csv, OUT_ROOT/*/log.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "$HERE")}"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-./runs/seq_cnn}"
MODES="${MODES:-seq seq_meth meth}"
EXTRA="${EXTRA:-}"
: "${DATA_ROOT:?set DATA_ROOT to the data folder (see header)}"
export DATA_ROOT OUT_ROOT
SCRIPT="$HERE/seq_cnn_chip.py"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== environment check"
"$PYTHON" - <<'PY'
import importlib, sys
bad = []
for m in ("numpy", "scipy", "pandas", "yaml", "torch", "pyBigWig", "pyfaidx"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad.append(f"{m} ({e})")
if bad:
    sys.exit("missing python modules: " + ", ".join(bad) +
             "\n  pip install numpy scipy pandas pyyaml torch pyBigWig pyfaidx")
import torch
print("  cuda available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if not torch.cuda.is_available():
    print("  WARNING: no GPU, training on CPU is far too slow for the default settings")
PY
echo "== data check ($DATA_ROOT)"
fail=0
for p in genome/hg38.fa; do [ -e "$DATA_ROOT/$p" ] || { echo "  missing $p"; fail=1; }; done
for c in gm12878 imr90 h1 hepg2 a549 gm23248 cd14mono k562 hmec; do
  [ -d "$DATA_ROOT/hg38_$c" ] || { echo "  missing hg38_$c/"; fail=1; }
  [ -e "$DATA_ROOT/meth/$c/Meth.bigwig" ] || { echo "  missing meth/$c/Meth.bigwig"; fail=1; }
done
[ -e "$DATA_ROOT/annotation/hg38/hg38-blacklist.v2.bed.gz" ] || \
  echo "  note: no blacklist file, only N-containing bins are excluded"
[ "$fail" = 0 ] || { echo "data check failed"; exit 1; }
echo "  ok"
echo "== cache"
"$PYTHON" "$SCRIPT" --stage cache --mode seq $EXTRA
for m in $MODES; do
  echo "== mode $m"
  "$PYTHON" "$SCRIPT" --stage train eval --mode "$m" $EXTRA
done
echo "== summary"
"$PYTHON" "$SCRIPT" --stage summarize $EXTRA
echo "done. Copy back: $REPO_ROOT/results/seq_cnn/seq_cnn_results.csv (and seq_cnn_summary.csv, seq_cnn_partition.csv)"
