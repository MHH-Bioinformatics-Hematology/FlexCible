"""predict --no-reference: the reference plane of a reference-input model is 0, so the prediction
no longer depends on the reference map; with the reference it does."""
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline import _write_cool, cfg, workdir  # noqa: E402,F401
from flexcible.config import Config  # noqa: E402


def _dense(path):
    import cooler
    return cooler.Cooler(str(path)).matrix(balance=False).fetch("chr1")


def test_no_reference_ignores_the_reference_map(workdir, cfg, tmp_path):  # noqa: F811
    from flexcible import precompute, predict, train

    d = tmp_path
    # two reference maps at the data scale of the (prescaled) target: noise around zero
    import cooler
    import pandas as pd
    src = cooler.Cooler(str(workdir / "target.cool"))
    px = src.pixels()[:]
    refs = []
    for seed in (0, 1):
        r = np.random.default_rng(seed)
        q = px.copy()
        q["count"] = r.normal(0, 1, len(q))
        p = d / f"ref{seed}.cool"
        cooler.create_cooler(str(p), bins=src.bins()[:], pixels=q, dtypes={"count": "float64"},
                             symmetric_upper=True, ordered=True)
        refs.append(p)
    c = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, reference_cool=str(refs[0])),
        window=dataclasses.replace(cfg.window, transform="prescaled"),
        model=dataclasses.replace(cfg.model, reference="input"),
        train=dataclasses.replace(cfg.train, output_dir=str(d / "run"), cache_dir=str(d / "cache"),
                                  max_steps=20, eval_every=10))
    precompute.build(c)
    train.train(c)
    out = {}
    for name, ref, flag in (("a", refs[0], False), ("b", refs[1], False),
                            ("a0", refs[0], True), ("b0", refs[1], True)):
        p = d / f"{name}.cool"
        predict.predict(c, "all", str(p), batch=4, reference=str(ref), no_reference=flag)
        out[name] = _dense(p)
    assert not np.allclose(out["a"], out["b"])          # the reference matters
    assert np.allclose(out["a0"], out["b0"], atol=1e-3)          # and is gone with --no-reference
    assert not np.allclose(out["a"], out["a0"])
