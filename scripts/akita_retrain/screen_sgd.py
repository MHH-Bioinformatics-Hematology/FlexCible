"""Short optimizer screen of the Akita retraining (registry section 79b).

One head (gm12878), split A, original architecture, batch 2, clip 10. Trains for
--minutes with SGD(lr, momentum) or Adam and prints the validation Pearson (60
validation windows) every --every steps, so that "trains" can be read off the curve.

    python screen_sgd.py --lr 0.0065 --momentum 0.99575 --minutes 12 --tag orig
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
C.setup_basenji_path()
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
import train as T

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--momentum", type=float, default=0.0)
ap.add_argument("--adam", action="store_true")
ap.add_argument("--clip", type=float, default=10.0)
ap.add_argument("--warmup", type=int, default=0, help="linear warm-up steps")
ap.add_argument("--minutes", type=float, default=12)
ap.add_argument("--every", type=int, default=500)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
p = json.load(open(C.W / "params_akita_original.json"))
p["model"]["head_hic"][-1]["units"] = 1
from basenji import seqnn
model = seqnn.SeqNN(p["model"]).model
opt = (tf.keras.optimizers.Adam(a.lr, clipnorm=a.clip) if a.adam else
       tf.keras.optimizers.SGD(a.lr, momentum=a.momentum, clipnorm=a.clip))
step, ev = T.make_step(model, opt, 0)
d, seq, band = T.load_split("A", ["gm12878"])
tr = T.Batches(d[d.role == "train"].reset_index(drop=True), seq, band, 2, True, 1)
dv = d[d.role == "val"].reset_index(drop=True)
dv = dv.iloc[np.linspace(0, len(dv) - 1, 60).astype(int)].reset_index(drop=True)
va = T.Batches(dv, seq, band, 2, False, 0)
t0, k, ls = time.time(), 0, []
print(f"{a.tag} lr {a.lr} mom {a.momentum} adam {a.adam} warmup {a.warmup}", flush=True)
while time.time() - t0 < a.minutes * 60:
    for codes, y in tr:
        if a.warmup and k < a.warmup:
            opt.learning_rate.assign(a.lr * (k + 1) / a.warmup)
        ls.append(float(step(codes, y))); k += 1
        if k % a.every == 0:
            vl, vr = T.evaluate(ev, va)
            print(f"{a.tag} step {k} train {np.mean(ls[-a.every:]):.4f} val_loss {vl:.4f} "
                  f"val_r {float(np.mean(vr)):.4f} t {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > a.minutes * 60:
            break
