"""Train Akita (original architecture and hyper-parameters) on our targets.

Model: basenji SeqNN built from the unchanged manuscripts/akita/params.json
(trunk of 96-filter convolutions and 8 dilated residual blocks, 2-D head with
6 dilated residual blocks, SGD lr 0.0065, momentum 0.99575, per-variable
gradient clip 10, batch 2, reverse-complement and +-11 bp shift augmentation of
the model, batch-norm momentum 0.9265, patience 12 on the validation Pearson),
with the number of output units set to the number of cells (one head per cell).

Differences from the original training script, all forced by the data:
  * targets are our proper O/E maps (build_data.py), not Akita's TFRecords;
    unmappable pixels (NaN) are masked out of the loss and the metrics,
  * the loss is the MSE over the finite pixels of all heads,
  * an epoch is one pass over the training windows of build_data.py,
  * a wall-clock budget (--max-minutes) ends training at the end of an epoch
    (inside the epoch with --pilot),
  * optional (--lr-cosine) learning-rate decay over that budget, and Adam instead of
    SGD where the parameter file says so (params_adam_*.json; see the registry).

    python train.py --split A --cells all --run akita_A_all12 --max-minutes 420
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C.setup_basenji_path()


def log(m):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)


def load_split(split: str, cells: list[str], roles=("train", "val")):
    """windows table (kept windows) and in-memory chromosome arrays."""
    d = pd.read_csv(C.windows_file(split), sep="\t")
    d = d[d.keep & d.role.isin(roles)].reset_index(drop=True)
    ci = [C.CELLS.index(c) for c in cells]
    seq, band = {}, {}
    for ch in sorted(d.chrom.unique()):
        seq[ch] = np.load(C.seq_dir() / f"{ch}.npy")
        band[ch] = np.ascontiguousarray(np.load(C.band_dir() / f"{ch}.npy", mmap_mode="r")[:, :, ci])
        log(f"loaded {ch}: band {band[ch].shape} {band[ch].nbytes / 1e9:.2f} GB")
    return d, seq, band


class Batches:
    """Background thread producing (codes int8 (B, L), target float32 (B, NV, K))."""

    def __init__(self, d, seq, band, batch, shuffle, seed, depth=6):
        self.d, self.seq, self.band, self.batch = d, seq, band, batch
        self.shuffle, self.rng = shuffle, np.random.default_rng(seed)
        self.depth = depth
        self.r, self.c = C.iu()

    def n_batches(self):
        return int(np.ceil(len(self.d) / self.batch)) if not self.shuffle else len(self.d) // self.batch

    def _one(self, i):
        row = self.d.iloc[i]
        s = int(row.start)
        codes = self.seq[row.chrom][s:s + C.SEQ_LEN]
        b0 = (s + C.CROP) // C.BIN
        y = self.band[row.chrom][b0 + self.r, self.c - self.r]       # (NV, K) float16
        return codes, y

    def __iter__(self):
        idx = np.arange(len(self.d))
        if self.shuffle:
            self.rng.shuffle(idx)
        q: queue.Queue = queue.Queue(self.depth)
        nb = self.n_batches()

        def work():
            for b in range(nb):
                ii = idx[b * self.batch:(b + 1) * self.batch]
                items = [self._one(i) for i in ii]
                q.put((np.stack([x[0] for x in items]).astype(np.int8),
                       np.stack([x[1] for x in items]).astype(np.float32)))
            q.put(None)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while True:
            x = q.get()
            if x is None:
                break
            yield x


def patch_mixed_precision():
    """basenji's ConcatDist2D concatenates a float32 distance channel to the
    (bfloat16) features; cast it to the layer dtype (no other change)."""
    import tensorflow as tf
    from basenji import layers

    def call(self, inputs):
        shp = tf.shape(inputs)
        batch_size, seq_len = shp[0], shp[1]
        pos = tf.expand_dims(tf.range(0, seq_len), axis=-1)
        m1 = tf.tile(pos, [1, seq_len])
        dist = tf.math.abs(m1 - tf.transpose(m1, [1, 0]))
        dist = tf.cast(dist, inputs.dtype)
        dist = tf.tile(tf.expand_dims(tf.expand_dims(dist, -1), 0), [batch_size, 1, 1, 1])
        return tf.concat([inputs, dist], axis=-1)

    layers.ConcatDist2D.call = call


def build_model(n_out: int, params_path: Path):
    import tensorflow as tf
    from basenji import seqnn
    p = json.load(open(params_path))
    p["model"]["head_hic"][-1]["units"] = n_out
    m = seqnn.SeqNN(p["model"])
    return m, p


def make_step(model, optimizer, micro: int = 0):
    """train_step(codes, y) -> mean loss. With micro > 0 a batch is processed in
    micro-batches whose gradients are averaged before one optimizer update
    (same update as the full batch except that batch-norm statistics are those
    of the micro-batch; used when the shared GPU has too little free memory)."""
    import tensorflow as tf

    tv = model.trainable_variables
    accum = [tf.Variable(tf.zeros_like(v), trainable=False) for v in tv]

    @tf.function(jit_compile=False)
    def grad_step(codes, y):
        x = tf.one_hot(tf.cast(codes, tf.int32), 4)               # N (4) -> all zero
        m = tf.math.is_finite(y)
        yz = tf.where(m, y, 0.0)
        mf = tf.cast(m, tf.float32)
        with tf.GradientTape() as tape:
            p = tf.cast(model(x, training=True), tf.float32)
            loss = tf.reduce_sum(mf * tf.square(p - yz)) / tf.maximum(tf.reduce_sum(mf), 1.0)
        g = tape.gradient(loss, tv)
        for a_, g_ in zip(accum, g):
            a_.assign_add(g_)
        return loss

    @tf.function(jit_compile=False)
    def apply_step(n):
        optimizer.apply_gradients([(a_ / n, v) for a_, v in zip(accum, tv)])
        for a_ in accum:
            a_.assign(tf.zeros_like(a_))

    def train_step(codes, y):
        mb = micro if micro else len(codes)
        n = int(np.ceil(len(codes) / mb))
        loss = 0.0
        for i in range(n):
            loss += float(grad_step(codes[i * mb:(i + 1) * mb], y[i * mb:(i + 1) * mb]))
        apply_step(tf.constant(float(n)))
        return loss / n

    @tf.function(jit_compile=False)
    def eval_step(codes, y):
        x = tf.one_hot(tf.cast(codes, tf.int32), 4)
        p = tf.cast(model(x, training=False), tf.float32)
        m = tf.math.is_finite(y)
        yz = tf.where(m, y, 0.0)
        mf = tf.cast(m, tf.float32)
        pm = p * mf                                               # per head sums over pixels
        # sums per head: n, sum x, sum y, sum xx, sum yy, sum xy, squared error
        n = tf.reduce_sum(mf, axis=[0, 1])
        sx = tf.reduce_sum(pm, axis=[0, 1])
        sy = tf.reduce_sum(yz * mf, axis=[0, 1])
        sxx = tf.reduce_sum(pm * p, axis=[0, 1])
        syy = tf.reduce_sum(yz * yz * mf, axis=[0, 1])
        sxy = tf.reduce_sum(pm * yz, axis=[0, 1])
        se = tf.reduce_sum(mf * tf.square(p - yz), axis=[0, 1])
        return tf.stack([n, sx, sy, sxx, syy, sxy, se])

    return train_step, eval_step


def evaluate(eval_step, val: Batches):
    S = None
    for codes, y in val:
        s = eval_step(codes, y).numpy().astype(np.float64)
        S = s if S is None else S + s
    n, sx, sy, sxx, syy, sxy, se = S
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sxx / n - (sx / n) ** 2
    vy = syy / n - (sy / n) ** 2
    r = cov / np.sqrt(np.maximum(vx * vy, 1e-12))
    return float(se.sum() / n.sum()), r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=list(C.SPLITS))
    ap.add_argument("--cells", default="all", help="'all' or a comma list")
    ap.add_argument("--run", required=True)
    ap.add_argument("--params", default=str(C.W / "params_akita_original.json"))
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-epochs", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=None, help="default: params train.patience (12)")
    ap.add_argument("--val-windows", type=int, default=None, help="cap on validation windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=["loss", "pearson"], default="loss",
                    help="early stopping and best weights on the validation loss (original rule, "
                         "default) or on the mean validation Pearson (section 79)")
    ap.add_argument("--warmup", type=int, default=0, help="linear learning-rate warm-up steps")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--micro", type=int, default=0, help="micro-batch size (0 = whole batch)")
    ap.add_argument("--pilot", action="store_true", help="stop at --max-minutes inside an epoch")
    ap.add_argument("--bf16", action="store_true", help="mixed precision (not the original)")
    ap.add_argument("--lr-cosine", type=float, default=0.0, metavar="FINAL_FRACTION",
                    help="learning rate decays on a cosine of the elapsed wall-clock fraction of "
                         "--max-minutes down to this fraction of the initial rate (0 = constant, "
                         "as in the original)")
    a = ap.parse_args()

    import tensorflow as tf
    if os.environ.get("AKITA_TF_THREADS"):
        k = int(os.environ["AKITA_TF_THREADS"])
        tf.config.threading.set_intra_op_parallelism_threads(k)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    tf.random.set_seed(a.seed)
    if a.bf16:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        patch_mixed_precision()

    cells = C.CELLS if a.cells == "all" else a.cells.split(",")
    out = C.W / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a) | {"cells_list": cells}, indent=1))

    m, p = build_model(len(cells), Path(a.params))
    model = m.model
    tp = p["train"]
    patience = a.patience if a.patience is not None else tp.get("patience", 12)
    if tp.get("optimizer", "sgd").lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=tp["learning_rate"],
                                       beta_1=tp.get("adam_beta1", 0.9),
                                       beta_2=tp.get("adam_beta2", 0.999),
                                       clipnorm=tp.get("clip_norm"))
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=tp["learning_rate"],
                                      momentum=tp["momentum"], clipnorm=tp.get("clip_norm"))
    train_step, eval_step = make_step(model, opt, a.micro)

    d, seq, band = load_split(a.split, cells)
    tr = Batches(d[d.role == "train"].reset_index(drop=True), seq, band, tp["batch_size"],
                 True, a.seed)
    dv = d[d.role == "val"].reset_index(drop=True)
    if a.val_windows and len(dv) > a.val_windows:
        dv = dv.iloc[np.linspace(0, len(dv) - 1, a.val_windows).astype(int)].reset_index(drop=True)
    va = Batches(dv, seq, band, 2, False, 0)
    log(f"{a.run}: train windows {len(tr.d)} ({tr.n_batches()} steps/epoch), "
        f"val windows {len(dv)}, heads {len(cells)}, patience {patience}")

    lc = out / "learning_curve.csv"
    rows, best, bad, epoch0 = [], np.inf if a.select == "loss" else -np.inf, 0, 0
    best_loss_r, best_r = np.inf, -np.inf
    if a.resume and (out / "latest.weights.h5").exists():
        model.load_weights(str(out / "latest.weights.h5"))
        st = json.loads((out / "state.json").read_text())
        best, bad, epoch0 = st["best"], st["bad"], st["epoch"]
        best_loss_r, best_r = st.get("best_loss", np.inf), st.get("best_r", -np.inf)
        rows = pd.read_csv(lc).to_dict("records")
        log(f"resumed at epoch {epoch0}, best {best:.4f}")

    t_start = time.time()
    steps_total = 0
    for ep in range(epoch0 + 1, a.max_epochs + 1):
        t0 = time.time()
        losses, tl = [], time.time()
        for k, (codes, y) in enumerate(tr):
            if a.lr_cosine and a.max_minutes:
                f = min(1.0, (time.time() - t_start) / (a.max_minutes * 60))
                opt.learning_rate.assign(tp["learning_rate"] * (
                    a.lr_cosine + (1 - a.lr_cosine) * 0.5 * (1 + np.cos(np.pi * f))))
            if a.warmup and steps_total < a.warmup:
                opt.learning_rate.assign(tp["learning_rate"] * (steps_total + 1) / a.warmup)
            losses.append(float(train_step(codes, y)))
            steps_total += 1
            if (k + 1) % 200 == 0:
                log(f"ep {ep} step {k + 1}/{tr.n_batches()} loss {np.mean(losses[-200:]):.4f} "
                    f"{200 / (time.time() - tl):.2f} steps/s")
                tl = time.time()
            if a.pilot and a.max_minutes and (time.time() - t_start) > a.max_minutes * 60:
                break                                 # pilot: stop inside the epoch
        t_train = time.time() - t0
        vloss, vr = evaluate(eval_step, va)
        vmean = float(np.mean(vr))
        row = dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=vloss, val_pearson=vmean,
                   train_steps=len(losses), seconds_train=round(t_train, 1),
                   seconds_total=round(time.time() - t0, 1),
                   steps_per_s=round(len(losses) / t_train, 3))
        for c, r in zip(cells, vr):
            row[f"val_r_{c}"] = float(r)
        rows.append(row)
        pd.DataFrame(rows).to_csv(lc, index=False, float_format="%.5f")
        log(f"epoch {ep}: train {row['train_loss']:.4f} val {vloss:.4f} r {vmean:.4f} "
            f"({row['seconds_total']:.0f} s, {row['steps_per_s']:.2f} steps/s)")
        model.save_weights(str(out / "latest.weights.h5"))
        best_loss_r, best_r = min(best_loss_r, vloss), max(best_r, vmean)
        score = vloss if a.select == "loss" else vmean
        if (score < best) if a.select == "loss" else (score > best):
            best, bad = score, 0
            model.save_weights(str(out / "best.weights.h5"))
            (out / "best_epoch.txt").write_text(f"{ep}\n")
        else:
            bad += 1
        (out / "state.json").write_text(json.dumps(dict(best=best, bad=bad, epoch=ep,
                                                        best_loss=best_loss_r, best_r=best_r)))
        if bad >= patience:
            log(f"early stop: no improvement for {patience} epochs")
            break
        if a.max_minutes and (time.time() - t_start) > a.max_minutes * 60:
            log(f"time budget of {a.max_minutes} min reached")
            break
    log(f"done: best val {a.select} {best:.4f}, {steps_total} steps, "
        f"{(time.time() - t_start) / 3600:.2f} h")
    (out / "DONE").write_text(f"{best}\n")


if __name__ == "__main__":
    main()
