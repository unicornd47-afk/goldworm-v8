"""IMPL_ERRATUM_v8.2_W1: warm-up of every structure, plausibility gate, ladder (a) only, bindings in check_start."""
import inspect
import json
import os

from v81 import preflight as PF
from v81 import release as RL
from v81 import spec as S


def _m(wm32=0.010, wm64=0.013):
    return {"unit": {"WM_32": {"per_update_cpu_s": wm32}, "WM_64": {"per_update_cpu_s": wm64}}}


def _proj(over):
    def project(m, setup, ladder=()):
        proj = {k: 0.5 * S.GATE_FRAC * v for k, v in S.CAPS.items()}
        for k, v in over.items():
            proj[k] = v(ladder) if callable(v) else v
        return proj, sum(proj.values()), {k: proj[k] <= S.GATE_FRAC * S.CAPS[k] for k in proj}
    return project


def test_warmup_covers_every_arm_and_structure():
    src = inspect.getsource(PF.measure)
    w = src.index("warmup_all_structures_cpu_s")
    first_timed = src.index("for arm in S.ARMS:\n        for st in S.structures(arm):\n            u = dict(")
    assert src.index("IMPL_ERRATUM_v8.2_W1") < w < first_timed          # warm-up precedes all timed measurements
    assert "model_lin.fit(train, ns, order=st)" in src[:first_timed] and "MT.build(arm, st, ns.to_dict(), 0)" in src[:first_timed]


def test_plausibility_gate(monkeypatch):
    monkeypatch.setattr(PF, "project", _proj({}))
    d = PF.decide(_m(0.010, 0.013), 1.0)
    assert d["all_gates_pass"] and d["plausibility"]["ok"]
    d = PF.decide(_m(0.013, 0.013), 1.0)
    assert d["all_gates_pass"]                                                   # equality allowed (<=)
    d = PF.decide(_m(0.0805, 0.0133), 1.0)
    assert not d["all_gates_pass"] and not d["plausibility"]["ok"]


def test_fits_fail_is_not_launched_no_ladder(monkeypatch):
    monkeypatch.setattr(PF, "project", _proj({"fits": 815.4, "adaptation": 537.4}))
    d = PF.decide(_m(), 1.0)
    assert not d["all_gates_pass"] and d["ladder_applied"] == [] and d["failing_non_ladder"] == ["fits"]


def test_adaptation_fail_ladder_a_only(monkeypatch):
    monkeypatch.setattr(PF, "project", _proj({"adaptation": lambda L: 300.0 if "a" in L else 537.4}))
    d = PF.decide(_m(), 1.0)
    assert d["all_gates_pass"] and d["ladder_applied"] == ["a"]
    monkeypatch.setattr(PF, "project", _proj({"adaptation": 500.0}))           # still failing after (a)
    d = PF.decide(_m(), 1.0)
    assert not d["all_gates_pass"] and d["ladder_applied"] == ["a"]
    monkeypatch.setattr(PF, "project", _proj({"tuning": 1000.0}))              # (b)/(c) not available under W1
    d = PF.decide(_m(), 1.0)
    assert not d["all_gates_pass"] and d["ladder_applied"] == []


def test_w1_and_attempt1_bound():
    assert S.W1_ALLOWED_LADDER == ("a",)
    for name, h in S.BUNDLE["impl_errata"].items():
        assert S.sha256_file(os.path.join(S.PLAN_DIR, name)) == h
    a1 = S.BUNDLE["preflight_attempt1"]
    p = os.path.join(S.PLAN_DIR, a1["file"])
    assert S.sha256_file(p) == a1["sha256"] == "f2a0e6b2a608d71bd0f1d554a2aca481ac97621f2be08985a565bdf73c4b17f7"
    assert not os.access(p, os.W_OK)
    with open(p) as f:
        old = json.load(f)
    assert not old["all_gates_pass"] and old["measurements"]["unit"]["WM_32"]["per_update_cpu_s"] > \
        old["measurements"]["unit"]["WM_64"]["per_update_cpu_s"]                # the attempt would fail plausibility too


def test_check_start_w1_reasons(tmp_path):
    rel = dict(status="DRAFT", plan_bundle={S.HASHES_FILE: S.HASHES_FILE_SHA256}, plan_pass={"sha256": S.PLAN_PASS_SHA256},
               grid_override=S.GRID_OVERRIDE, implementer="TMbot · Research (non-author)", code_review_pass=None,
               code_hashes={"manifest_sha256": "x"}, preflight={"all_gates_pass": True})
    p = tmp_path / "RELEASE_v8.2.json"
    p.write_text(json.dumps(rel))
    r = RL.check_start(str(p))[1]
    assert any("implementation erratum" in x for x in r) and any("attempt 1" in x for x in r)
    rel.update(impl_errata=S.BUNDLE["impl_errata"], preflight_attempts=[dict(S.BUNDLE["preflight_attempt1"])])
    p.write_text(json.dumps(rel))
    r = RL.check_start(str(p))[1]
    assert not any("implementation erratum" in x for x in r) and not any("attempt 1" in x for x in r)
