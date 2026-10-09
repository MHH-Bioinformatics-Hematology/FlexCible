"""Run scripts/multicell_expanded.py or scripts/scenario_bench.py on one GPU with the
recipe of the two-GPU DDP runs.

The training batch size is per rank (train.batch_size = 3, two ranks, global batch 6).
On one GPU the same global batch needs train.batch_size = 6, with the same learning rate,
the same number of optimizer steps per epoch and the same early-stopping rule. This
wrapper multiplies the batch size of every config built by multicell_expanded.run_config
by HICMULTI_BATCH_SCALE (default 2) and then calls the module's main() with the
remaining arguments:

    python scripts/run_single_gpu.py multicell_expanded --runs all12honest_drop_meth --res 25000 --stage train --gpus 0
    python scripts/run_single_gpu.py scenario_bench --stage configs
    python scripts/run_single_gpu.py scenario_bench --stage train --run scen_meth_only_25kb_s0 --gpus 0

Set HICMULTI_BATCH_SCALE=1 to keep the per-rank batch (global batch 3).
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if len(sys.argv) < 2 or sys.argv[1] not in ("multicell_expanded", "scenario_bench"):
    raise SystemExit(__doc__)
mod = sys.argv[1]
sys.argv = [mod + ".py"] + sys.argv[2:]
scale = int(os.environ.get("HICMULTI_BATCH_SCALE", "2"))
mx = importlib.import_module("multicell_expanded")
_orig = mx.run_config


def run_config(*a, **k):
    cfg = _orig(*a, **k)
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"]) * scale
    return cfg


mx.run_config = run_config
importlib.import_module(mod).main()
