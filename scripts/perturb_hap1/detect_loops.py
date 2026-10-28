"""Region-only cooler of the WT capture map (10 kb, raw counts), ICE balanced inside the region, then hicDetectLoops."""
import sys, os, subprocess; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, cooler
from scipy.sparse import coo_matrix
from common import *
from lib import region_counts
HE = Path.home() / "miniconda3/envs/__hicexplorer@3.7.6/bin"
D = OUT / "work" / "loops"; D.mkdir(parents=True, exist_ok=True)
for rn, r in REGIONS.items():
    c = region_counts(mcool_path(WT_SET), 10000, r["chrom"], r["start"], r["end"])
    n = c.shape[0]; st = 10000 * np.arange(n)   # region-local coordinates; add the region start to the loop positions
    bins = pd.DataFrame(dict(chrom=r["chrom"], start=st, end=st + 10000))
    iu = np.triu_indices(n); v = c[iu]; k = v > 0
    px = pd.DataFrame(dict(bin1_id=iu[0][k], bin2_id=iu[1][k], count=v[k].astype(int)))
    f = str(D / f"wt_{rn}.cool"); cooler.create_cooler(f, bins, px, ordered=True)
    cooler.balance_cooler(cooler.Cooler(f), store=True, mad_max=0, min_nnz=1, ignore_diags=1)
    subprocess.run([str(HE / "hicDetectLoops"), "-m", f, "-o", str(D / f"loops_{rn}.bedgraph"), "--peakWidth", "6", "--windowSize", "10",
                    "--pValuePreselection", "0.1", "--pValue", "0.1", "--peakInteractionsThreshold", "5", "--obsExpThreshold", "1.0", "--maxLoopDistance", "1500000", "--chromosomes", r["chrom"], "--threads", "2"],
                   check=False)
