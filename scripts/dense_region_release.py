"""Dense chr2 predictions of the release twin all12one_fullrd_chrho at 10 kb for the count-space figures (registry 101).

Reproduces runs/dense_region (all12one_chrho_10kb_s24: chr2, window stride 24 instead of 96 so that every pixel
up to 168 bins from the diagonal is covered, no TTA, checkpoint-best, consensus as reference input) with the twin,
and writes the same file names into runs/dense_region_fullrd_chrho:
    <cell>.cool, <cell>_counts.log, <cell>_counts_chr2.cool  (cells gm12878, imr90, k562)
and copies gm12878_consensus_counts_chr2.cool (the consensus, no model) so that the folder is complete.
Inputs as in scripts/hic_measures.py chrho_predict_one: the training cells' tracks and Meth through the config sample,
K562 from hg38_k562 with meth/k562; leave-one-out consensus (K562: the 12-cell consensus) as reference.
The back-transform is scripts/oe_to_counts.py --chroms chr2 with the measured map of the same cell.
    python scripts/dense_region_release.py [--gpu 0]
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SH = Path("/mnt/processed/wolffjoa/hic-multiome")
MR = SH / "runs/multicell_reference"
OLD = SH / "runs/dense_region"
NEW = SH / "runs/dense_region_fullrd_chrho"
OE = SH / "crosscell_oe"
HM = SH.parent
M4 = HM / "restart_of_prediction_project/originalData/4Dnucleosome"
MEAS = {"gm12878": M4 / "GM12878_4DNFIXP4QG5B.mcool", "imr90": HM / "hyperparameter_publication/imr90.mcool",
        "k562": M4 / "K562_4DNFI18UHVRO.mcool"}
CONS = {"gm12878": OE / "consensus_loo_gm12878_proper_10kb.cool", "imr90": OE / "consensus_loo_imr90_proper_10kb.cool",
        "k562": OE / "consensus12_proper_10kb.cool"}
MODEL = "all12one_fullrd_chrho"
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="")
    ap.add_argument("--cells", nargs="*", default=["gm12878", "imr90", "k562"])
    a = ap.parse_args()
    NEW.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((MR / f"{MODEL}_10kb.yaml").read_text())
    assert cfg["window"]["stride"] == 96
    cfg["window"]["stride"] = 24
    cfg_s24 = NEW / f"{MODEL}_10kb_s24.yaml"
    cfg_s24.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="8")
    for cell in a.cells:
        out = NEW / f"{cell}.cool"
        if not out.exists():
            cmd = [PY, "-m", "flexcible.predict", "--config", str(cfg_s24),
                   "--checkpoint", str(MR / f"{MODEL}_10kb/checkpoint-best"), "--chroms", "chr2",
                   "--out", str(out) + ".tmp", "--reference", str(CONS[cell])]
            if cell == "k562":
                cmd += ["--sample", "gm12878", "--bigwig-dir", str(SH / "hg38_k562"), "--cell", "k562",
                        "--extra-bigwig-dir", str(SH / "meth/k562")]
            else:
                cmd += ["--sample", cell]
            with open(NEW / f"{cell}.log", "w") as fh:
                fh.write(" ".join(cmd) + "\n")
                fh.flush()
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=env, check=True)
            os.replace(str(out) + ".tmp", out)
        cc = NEW / f"{cell}_counts_chr2.cool"
        if not cc.exists():
            with open(NEW / f"{cell}_counts.log", "w") as fh:
                subprocess.run([PY, "scripts/oe_to_counts.py", "--oe", str(out), "--measured", str(MEAS[cell]),
                                "--output", str(cc), "--binsize", "10000", "--chroms", "chr2"],
                               stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, check=True)
    for f in ("gm12878_consensus_counts_chr2.cool", "gm12878_consensus_counts.log"):
        if not (NEW / f).exists():
            shutil.copyfile(OLD / f, NEW / f)


if __name__ == "__main__":
    main()
