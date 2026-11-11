import json
import os

import numpy as np

from v81 import release as RL
from v81 import spec as S
from v81 import stats as ST


def _prim(Ew, Ec, Dw, Dc, sw=0, sc=0):
    cl = np.repeat(np.arange(40), 8)
    return ST.primary_endpoints(Ew, Ec, Dw, Dc, cl, ST.boot_matrix(S.BOOT_SEEDS["primary"], 40), sw, sc)


def test_advance_when_all_pass():
    rng = np.random.default_rng(1)
    Ec = rng.uniform(20, 30, 320)
    p = _prim(0.8 * Ec, Ec, np.zeros(320), np.zeros(320))
    assert p["P1"] and p["P2"] and p["P3"] and p["P4"]
    assert ST.outcome(p, True) == "ADVANCE"
    assert ST.outcome(p, False) == "INVALID"
    assert ST.outcome(p, True, launched=False) == "NOT_LAUNCHED"


def test_p1_threshold_and_p3_margin():
    Ec = np.full(320, 10.0)
    p = _prim(0.96 * Ec, Ec, np.zeros(320), np.zeros(320))
    assert not p["P1"] and ST.outcome(p, True) == "NO_ADVANCE"
    p = _prim(0.8 * Ec, Ec, np.full(320, 0.3), np.zeros(320))
    assert not p["P3"]
    p = _prim(0.8 * Ec, Ec, np.full(320, 0.2), np.zeros(320))
    assert p["P3"]                                    # worse by less than the 0.25 margin is allowed


def test_p4_safety_gate():
    Ec = np.full(320, 10.0)
    p = _prim(0.5 * Ec, Ec, np.zeros(320), np.zeros(320), sw=3, sc=2)
    assert not p["P4"] and ST.outcome(p, True) == "NO_ADVANCE"
    p = _prim(0.5 * Ec, Ec, np.zeros(320), np.zeros(320), sw=2, sc=2)
    assert p["P4"]


def test_draft_release_is_not_launched(tmp_path):
    """v8.3: a DRAFT release without CODE_PASS / probe / CONFIG_v8.3 never launches."""
    rel = dict(status="DRAFT", plan_bundle_v8_2={S.HASHES_FILE: S.HASHES_FILE_SHA256},
               plan_pass_v8_2={"sha256": S.PLAN_PASS_SHA256}, implementer="TMbot · Research (non-author)",
               code_review_pass=None, code_hashes={"manifest_sha256": "x"})
    p = tmp_path / "RELEASE_v8.3.json"
    p.write_text(json.dumps(rel))
    ok, reasons, _, _ = RL.check_start(str(p))
    assert not ok
    assert any("code-review PASS missing" in r for r in reasons)
    assert any("FINAL" in r for r in reasons)
    assert not any("v8.2 plan PASS" in r for r in reasons)          # real dahill PASS 8ad13f89 verifies
    assert any("grid_override" in r for r in reasons)
    assert any("diff v8.2 -> v8.3" in r for r in reasons)
    assert any("plan r3" in r for r in reasons)
    assert any("PLAN_REVIEW_v8.3_r3" in r for r in reasons)
    assert any("conditions not closed" in r for r in reasons)
    assert any("SEED_EVIDENCE" in r for r in reasons)
    assert any("sidecar" in r for r in reasons)
    assert any("probe file" in r for r in reasons)


def test_missing_release_is_not_launched(tmp_path):
    ok, reasons, _, _ = RL.check_start(str(tmp_path / "nope.json"))
    assert not ok and reasons == ["RELEASE_v8.3.json missing"]


def test_production_entry_writes_not_launched_without_release(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp", os.path.join(S.CODE_DIR, "run_production.py"))
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    ok_rule = lambda: dict(ok=True, reasons=[])
    st = rp.main(release_path=str(tmp_path / "RELEASE_missing.json"), out_dir=str(tmp_path / "run_output"),
                 start_rule_fn=ok_rule, with_sidecar=False)
    assert st["outcome"] == "NOT_LAUNCHED"
    assert not (tmp_path / "run_output").exists()
    rec = (tmp_path / "NOT_LAUNCHED_run_output.jsonl").read_text().splitlines()
    assert len(rec) == 1
    bad_rule = lambda: dict(ok=False, reasons=["load1 2.00 >= 1.0"])
    st = rp.main(release_path=str(tmp_path / "RELEASE_missing.json"), out_dir=str(tmp_path / "run_output"),
                 start_rule_fn=bad_rule, with_sidecar=False)
    assert any("C4 start rule: load1" in r for r in st["reasons"])
    assert len((tmp_path / "NOT_LAUNCHED_run_output.jsonl").read_text().splitlines()) == 2    # append-only


def test_budget_cap_raises():
    from v81.budget import Budget, CapExceeded
    t = [0.0]
    b = Budget(caps={k: 1.0 for k in S.CAPS}, global_cap=100.0, clock=lambda: t[0])
    try:
        with b.phase("fits"):
            t[0] = 2.0
            b.check()
        raise AssertionError("no raise")
    except CapExceeded:
        pass
