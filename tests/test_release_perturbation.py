"""The perturbation, washout, AML knockout and methylation scripts accept the per-resolution views of the
one model all12one_fullrd (registry section 95): Meth in every keep set, fraction row, consensus plane,
result-file tags that leave the tables of earlier models untouched.

CPU only:  CUDA_VISIBLE_DEVICES= python -m pytest tests/test_release_perturbation.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import aml_eval as ae                                                # noqa: E402
import meth_proxy_release as mpr                                     # noqa: E402
import perturb_validate as pv                                        # noqa: E402
import washout as wo                                                 # noqa: E402


def test_perturbation_sets_of_the_view_keep_meth():
    P = pv.perts_of("all12one_fullrd_10kb")
    assert set(P) == {"full", "minus_RAD21", "minus_CTCF", "minus_H3K36me3",
                      "minus_RAD21_f0", "minus_RAD21_f0.5"}
    assert P["full"] is None and pv.all_tracks("all12one_fullrd_10kb")[-1] == "Meth"
    for k, keep in P.items():
        if keep is not None:
            assert "Meth" in keep and len(keep) == 13
    assert "RAD21" not in P["minus_RAD21_f0"] and "CTCF" not in P["minus_CTCF"]


def test_old_models_keep_their_perturbations():
    assert list(pv.perts_of("all12deg_refres_drop_10kb")) == list(pv.PERTS)
    assert "minus_RAD21_f0" in pv.perts_of("all12cohfix_refres_drop_25kb")
    assert pv.meth_args("all12cohfix_refres_drop_25kb") == []
    assert pv.aux_fraction("all12deg_refres_drop_10kb") == []
    assert pv.aux_fraction("all12one_fullrd_25kb") == ["--cohesin-fraction", "0.26"]
    assert all("Meth" not in (k or []) for k in pv.perts_of("all12cohfix_refres_drop_10kb").values())


def test_view_models_registered_and_finished():
    for k in (10, 25):
        run, res, ref = pv.MODELS[f"all12one_fullrd_{k}kb"]
        assert res == k * 1000 and ref and run.name == f"all12one_fullrd_{k}kb"
        assert not pv.unfinished(f"all12one_fullrd_{k}kb", run, res)
    assert pv.ref_of("all12one_fullrd_10kb", "all12one_fullrd", 10000).endswith(
        "consensus_loo_hct116_proper_10kb.cool")
    assert pv.ref_of("all12honest_10kb", None, 10000) is None
    assert pv.meth_args("all12one_fullrd_10kb")[1].endswith("meth/hct116")


def test_prediction_device(monkeypatch):
    monkeypatch.delenv("PERT_GPU", raising=False)
    assert pv.pred_device()[0] == "cpu"
    monkeypatch.setenv("PERT_GPU", "1")
    dev, env = pv.pred_device()
    assert dev == "cuda" and env["CUDA_VISIBLE_DEVICES"] == "1"


def test_washout_and_aml_know_the_views():
    assert "all12one_fullrd" in wo.FAMILIES
    assert wo._trained("all12one_fullrd", 10000) in (True, False)      # no stop_info lookup for a view
    for k in (10, 25):
        res, cfg, run, fam, ref = ae.MODELS[f"all12one_fullrd_{k}kb"]
        assert res == k * 1000 and fam in ae.COHFIX_FAMILIES and fam in ae._DROP and ref == "consensus12"
        assert cfg.name == "config.yaml" and cfg.parent.name == "checkpoint-best"


def test_meth_proxy_release_sets():
    S = mpr.sets_all()
    assert "full" in S and "full+meth" in S and "full_noref" in S and "full+meth_noref" in S
    assert S["minus_CTCF+meth"][-1] == "Meth" and "Meth" not in S["minus_CTCF"]
    assert len(S) == 4 * len(mpr.mp.SET_ORDER)
    assert mpr.pred_csv(10, True, "full+meth", "k562").parent.name == "full+meth_noref"
    assert mpr.noise_of_model("all12one_fullrd_10kb") == 0.0062
