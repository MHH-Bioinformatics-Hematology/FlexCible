"""Claim logic of scripts/gpu_queue_extra.py (throwaway claims folder, no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def test_claim_skip_and_release(tmp_path, monkeypatch):
    import gpu_queue_extra as q
    job = SimpleNamespace(claim="scen_test_throwaway_25kb_s9", name="train_x")
    plain = SimpleNamespace(name="scen_ded_x")
    d = tmp_path / "claims"
    monkeypatch.setenv("HICMULTI_HOST", "local")
    assert q.try_claim(plain, d) and not q.claimed_elsewhere(plain, d)    # no claim name: always ours
    assert not q.claimed_elsewhere(job, d)
    assert q.try_claim(job, d)                                            # free: claimed
    assert (d / "scen_test_throwaway_25kb_s9.claim").exists()
    assert q.try_claim(job, d) and not q.claimed_elsewhere(job, d)        # own claim of an earlier run
    monkeypatch.setenv("HICMULTI_HOST", "remote")
    assert q.claimed_elsewhere(job, d) and not q.try_claim(job, d)        # seen from the other host
    q.release_claim(job, d)                                               # not the owner: kept
    assert (d / "scen_test_throwaway_25kb_s9.claim").exists()
    monkeypatch.setenv("HICMULTI_HOST", "local")
    q.release_claim(job, d)                                               # owner (failure): removed
    assert not (d / "scen_test_throwaway_25kb_s9.claim").exists()
    assert not q.claimed_elsewhere(job, d)
    monkeypatch.setenv("HICMULTI_HOST", "remote")                         # now free for the other host
    assert q.try_claim(job, d)
    monkeypatch.setenv("HICMULTI_HOST", "local")
    assert q.claimed_elsewhere(job, d) and not q.try_claim(job, d)


def test_only_dedicated_trainings_carry_a_claim():
    import gpu_queue_extra as q
    import scenario_bench as sb
    J = q.jobs()
    claimed = {j.name: j.claim for j in J if getattr(j, "claim", None)}
    chrho = {q.mx.run_name(r, 5000) for r, _ in q.CHRHO5}
    assert set(claimed.values()) == set(sb.dedicated_runs()) | chrho | {f"{b}_{k}kb" for b in ("all12cohfix_refres_drop", "all12cohfixna_refres_drop") for k in (25, 10)}
    assert all(n == f"train_{c}" for n, c in claimed.items())
    assert not any(getattr(j, "claim", None) for j in J if j.name.startswith(("scen_ded_", "scen_infer")))
