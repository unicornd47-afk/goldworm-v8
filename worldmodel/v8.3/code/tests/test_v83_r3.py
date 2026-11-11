"""v8.3 code revision r3: R_fresh/drift, production verdict, ERRATUM E1 + E2 (plan r4), dahill requirements 1-7."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v81 import baseline as BL  # noqa: E402
from v81 import budget as BU  # noqa: E402
from v81 import gates as GT  # noqa: E402
from v81 import hostenv as HE  # noqa: E402
from v81 import models_torch as MT  # noqa: E402
from v81 import probe as PB  # noqa: E402
from v81 import release as RL  # noqa: E402
from v81 import spec as S  # noqa: E402
from v81 import warmup as WU  # noqa: E402

LAP = dict(hostname="unicorn", cpu_model="13th Gen Intel(R) Core(TM) i5-13500H", cpu_count=16)
BOX = dict(S.BOX_FINGERPRINT)


def _mod(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(S.CODE_DIR, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _toy_file(d, host, n=1, fits=400.0, pinning=None):
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, "TOY_PREFLIGHT_attempt%d.json" % n)
    open(f, "w").write(json.dumps(dict(preflight_within_cap=True, projection_cpu_s=dict(fits=fits))))
    GT.append(os.path.join(d, GT.TOY_LEDGER), "attempt_end", attempt=n, outcome="TOY_OK", file=os.path.basename(f),
              sha256=S.sha256_file(f), host=host, pinning=pinning)


# ---------------- bundle: plan r4 = r3 + E1 + E2 ----------------
def test_bundle_binds_both_errata():
    h = S.verify_v83_bundle()
    assert h["ERRATUM_v8.3_E1.md"] == S.V83["erratum_sha256"] and S.V83["erratum_sha256"].startswith("ea642114")
    assert h["ERRATUM_v8.3_E2.md"] == S.V83["erratum2_sha256"] and S.V83["erratum2_sha256"].startswith("3d6f8c0a")
    assert h["PLAN_v8.3.md"] == S.V83["plan_sha256"]


# ---------------- item 1: R_fresh + drift (box only, descriptive) ----------------
def test_k2_fresh_on_box_and_unchanged_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WM_ROOT", str(tmp_path))
    d = str(tmp_path / "probe")
    r = GT.k2_fresh(BOX, d, 500.0)
    assert r["R_fresh"] is None and r["drift_flag"] is None and r["reference_toy_fits"] == 336.8
    _toy_file(d, BOX, fits=400.0)
    r = GT.k2_fresh(BOX, d, 500.0)
    assert r["R_fresh"] == 500.0 / 400.0 and r["R_gate"] == 500.0 / 336.8
    assert r["fresh_toy_fits"] == 400.0 and r["drift_flag"] is True and abs(r["drift_rel"] - (400 / 336.8 - 1)) < 1e-12
    _toy_file(d, BOX, n=2, fits=350.0)
    r = GT.k2_fresh(BOX, d, None)
    assert r["drift_flag"] is False and r["R_fresh"] is None and r["fresh_toy"]["attempt"] == 2
    assert GT.k2_fresh(LAP, d, 500.0) is None                                # non-box: unchanged (no R_fresh)


def test_box_k2_reference_recorded_as_assumed_same_host():
    ref, r = GT.k2_reference(BOX, "/nonexistent")
    assert r == [] and ref["toy_fits"] == 336.8 and ref["host_assumption"].startswith("assumed same host")


def test_box_gate_R_still_uses_336_8():
    src = open(os.path.join(S.CODE_DIR, "run_probe.py")).read()
    assert 'PB.decide(pm, toy, aborted=aborted, toy_fits=ref["toy_fits"])' in src
    assert GT.k2_reference(BOX, "/nonexistent")[0]["toy_fits"] == S.K2_TOY_FITS


# ---------------- item 2: production accepts only CODE_PASS ----------------
def _release_with_cp(tmp_path, verdict):
    cp = tmp_path / "CP.json"
    cp.write_text(json.dumps(dict(reviewer="dahill", verdict=verdict, manifest_sha256="m" * 64)))
    rel = dict(code_review_pass=dict(file=str(cp), sha256=S.sha256_file(str(cp)), verdict=verdict),
               code_hashes=dict(manifest_sha256="m" * 64))
    p = tmp_path / "REL.json"
    p.write_text(json.dumps(rel))
    return str(p)


def test_production_requires_unconditional_code_pass(tmp_path):
    _, r, _, _ = RL.check_start(_release_with_cp(tmp_path, "CODE_PASS_CONDITIONAL"))
    assert any("production requires an unconditional CODE_PASS" in x for x in r)
    (tmp_path / "x").mkdir()
    _, r, _, _ = RL.check_start(_release_with_cp(tmp_path / "x", "CODE_PASS"))
    assert not any("unconditional CODE_PASS" in x for x in r) and not any("code-review PASS missing" in x for x in r)
    assert S.CODE_PASS_VERDICTS_PRODUCTION == ("CODE_PASS",)


def test_toy_and_probe_still_accept_conditional(tmp_path):
    (tmp_path / "CODE_PASS_v8.3_rX.json").write_text(json.dumps(dict(
        reviewer="dahill", verdict="CODE_PASS_CONDITIONAL", plan_sha256=S.V83["plan_sha256"], manifest_sha256="q" * 64)))
    assert GT.code_pass_reasons("q" * 64, str(tmp_path)) == []
    assert "CODE_PASS_CONDITIONAL" in S.CODE_PASS_VERDICTS


# ---------------- item 4 / dahill req 7: checksum list excludes itself ----------------
def test_write_sha256sums_excludes_itself(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    out = GT.write_sha256sums(str(tmp_path), "SUMS.txt")
    lines = open(out).read().splitlines()
    assert [l.split()[1] for l in lines] == ["./a.txt", "./sub/b.txt"] and "SUMS" not in open(out).read()
    with pytest.raises(FileExistsError):
        GT.write_sha256sums(str(tmp_path), "SUMS.txt")


# ---------------- E1.1 / E2.1 warm-up ----------------
def _fit(arm, st, ctx, fut, tgt, n=6):
    mdl = MT.build(arm, st, _NORM, 11)
    bidx = np.random.default_rng([11, 2]).integers(0, len(ctx), size=(n, S.BATCH))
    mdl, _ = MT.train(mdl, ctx, fut, tgt, bidx, n, lambda t: MT.lr_cosine(t, n, S.LR0, S.LR_MIN), log_every=0)
    return {k: v.detach().clone() for k, v in mdl.state_dict().items()}


_NORM = None


def _windows():
    """Small real train windows (preflight seeds) for the warm-up tests."""
    global _NORM
    from v81 import features as Fe
    from v81 import plant_iv as P
    from v81 import seeds as SD
    G = SD.SeedGuard("probe")
    comp = SD.train_composition(2)
    tr = [P.data_episode(s, i, comp[i][0], comp[i][1], G, "train")[0] for i, s in enumerate(list(S.TRAIN_SEEDS[0])[:2])]
    train = P.stack_obs(tr)
    ns = Fe.NormStats.from_train(train)
    _NORM = ns.to_dict()
    return Fe.window_arrays(train, Fe.train_pairs(train["u"].shape[0]), ns)


def test_warmup_is_bit_identical_and_rng_isolated():
    ctx, fut, tgt = _windows()
    for arm, st in (("WM", 32), ("GRU", 32), ("RC_only", 3)):
        torch.manual_seed(123)
        a = _fit(arm, st, ctx, fut, tgt)
        s0 = torch.get_rng_state()
        w = WU.warmup(arm, st, _NORM, ctx, fut, tgt, n_updates=100)
        assert torch.equal(torch.get_rng_state(), s0)                       # torch global RNG restored
        b = _fit(arm, st, ctx, fut, tgt)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)   # bit-identical fit results
        assert w["updates"] == 100 and w["seed"] == S.WARMUP_SEED and len(w["block_per_update_cpu_s"]) == 2
        assert (w["arm"], w["structure"]) == (arm, st)                     # same architecture as the next segment


def test_warmup_seed_not_in_any_seed_set():
    allseeds = set(S.TUNING_SEEDS) | set(S.TEST_SEEDS) | set(S.ADAPT_SEEDS) | set(S.INIT_SEEDS.values())
    for r in range(S.REPLICATES):
        allseeds |= set(S.TRAIN_SEEDS[r]) | set(S.VAL_SEEDS[r])
    assert S.WARMUP_SEED not in allseeds and S.WARMUP_UPDATES == 200


def test_budget_warmup_counts_inside_phase_and_k1():
    t = [0.0]
    b = BU.Budget(caps=dict(S.CAPS, fits=10.0), clock=lambda: t[0], t0=0.0, k1_cap=3600.0)
    with b.phase("fits"):
        with b.warmup():
            t[0] += 4.0
        t[0] += 5.0
        b.check()
    assert b.used["fits"] == 9.0 and b.warmup_cpu_s == 4.0 and b.n_warmups == 1   # E2.1: inside the fits phase
    with pytest.raises(BU.CapExceeded):
        with b.phase("tuning"):
            with b.warmup():
                t[0] += 5000.0
            b.check()                                                        # K1/global see the warm-up CPU


# ---------------- probe measurement: E2.2 + dahill reqs 1-2 ----------------
TINY = dict(n_updates=20, fraction=0.1, n_train=4, n_val=2,
            structures={"WM": (32,), "GRU": (32,), "LIN": (3,), "RC_only": (3,)}, full_fit=("WM", 32), warmup_updates=50)


def test_measure_probe_warmups_two_full_fits_blocks(monkeypatch):
    monkeypatch.setattr(PB, "linearity_ok", lambda r: True)
    monkeypatch.setattr(S, "PROBE_BLOCK_UPDATES", 5)
    ev = []
    pm = PB.measure_probe(PB.ProbeBudget(cap=1e9), event=lambda e, **k: ev.append((e, k)), **TINY)
    names = [e for e, _ in ev]
    wu = [k for e, k in ev if e == "warmup"]
    assert [(w["arm"], w["structure"]) for w in wu] == [("WM", 32), ("GRU", 32), ("RC_only", 3)]
    # warm-up precedes each timed neural segment; the two WM fits follow ONE warm-up back-to-back
    fe = [(e, k.get("arm")) for e, k in ev if e in ("warmup", "probe_fit_end")]
    assert fe[:3] == [("warmup", "WM"), ("probe_fit_end", "WM"), ("probe_fit_end", "WM")]
    assert len(pm["full_fit_repeats"]) == 2 and all(u["updates_run"] == 20 for u in pm["full_fit_repeats"])
    for u in pm["full_fit_repeats"] + [pm["unit"]["GRU_32"]]:
        assert u["block_updates"] == 5 and len(u["block_per_update_cpu_s"]) == u["updates_run"] // 5
    assert all("block_per_update_cpu_s" in k for e, k in ev if e == "probe_fit_end")      # ledger too
    assert pm["linearity"]["gating_fit"] == 2 and len(pm["linearity"]["per_fit"]) == 2
    cs = pm["cold_start"]
    assert cs["mode"] in ("none", "per_process", "per_fit") and cs["warmup_cpu_s"] > 0 and cs["n_process_starts"] == 1
    assert "probe_cold_start" in names and pm["warmup_total_cpu_s"] == sum(w["cpu_s"] for w in pm["warmups"])


def _full(first, train, build=0.1, ev=0.2):
    return dict(first_mark_train_cpu_s=first, train_cpu_s=train, build_cpu_s=build, eval_cpu_s=ev, failed=False,
                fit_cpu_s=build + train + ev)


def test_cold_start_modes_and_projection():
    n, n10 = 2000, 200
    w = [dict(updates=200, cpu_s=2.5), dict(updates=200, cpu_s=1.0)]
    # e_k = first - 200 x (train - first)/1800
    none = PB.cold_start([_full(2.0, 20.0), _full(2.0, 20.0)], n, n10, warmups=w)       # rate 0.01 -> e = 0
    assert none["mode"] == "none" and none["W_cpu_s"] == 2.5 and none["n_cold_starts"] == 1
    pp = PB.cold_start([_full(5.0, 23.0), _full(2.1, 20.1)], n, n10, warmups=w, n_process_starts=1)
    assert pp["mode"] == "per_process" and abs(pp["W_cpu_s"] - (2.5 + pp["excess1_cpu_s"])) < 1e-12
    pf = PB.cold_start([_full(5.0, 23.0), _full(5.0, 23.0)], n, n10, warmups=w, n_process_starts=1)
    assert pf["mode"] == "per_fit" and pf["n_cold_starts"] == 27 and pf["W_cpu_s"] == max(2.5, pf["excess1_cpu_s"])
    assert abs(pf["linear_full_fit_cpu_s"] - (0.3 + 2000 * (18.0 / 1800))) < 1e-9
    # projection = 3 x per replicate (linear part) + W x n_coldstarts
    from tests.test_v83_mechanisms import _pm
    pm = _pm(per_upd=0.005, ep=0.03)
    pm["cold_start"] = pf
    f = PB.fits_projection(pm)
    assert abs(f["total"] - (3 * f["per_replicate"] + pf["W_cpu_s"] * 27)) < 1e-9
    assert f["terms"]["full_fit"] == pf["linear_full_fit_cpu_s"] and f["terms"]["cold_start_mode"] == "per_fit"


def test_linearity_gates_on_fit_2_only():
    good, bad = _full(2.0, 20.0), _full(4.0, 20.0)                           # ratios 1.0 and 2.0
    assert PB.linearity_record([bad, good], 0.1, 2000, 200)["ok"] is True
    rec = PB.linearity_record([good, bad], 0.1, 2000, 200)
    assert rec["ok"] is False and rec["ratio"] == 2.0 and [x["ratio"] for x in rec["per_fit"]] == [1.0, 2.0]
    d = PB.decide(dict(linearity=rec), None)
    assert d["outcome"] == "PROBE_DIAGNOSIS" and d["diagnosis_kind"] == "linearity"


# ---------------- E1.2 / E2.5 pinning + dahill req 4 ----------------
def _sysfs(tmp_path, core=None, atom=None, caps=None, freqs=None, n=4):
    root = tmp_path / "sys"
    if core is not None:
        (root / "devices/cpu_core").mkdir(parents=True)
        (root / "devices/cpu_core/cpus").write_text(core)
        (root / "devices/cpu_atom").mkdir(parents=True)
        (root / "devices/cpu_atom/cpus").write_text(atom)
    for c in range(n):
        d = root / ("devices/system/cpu/cpu%d" % c)
        (d / "cpufreq").mkdir(parents=True)
        if caps:
            (d / "cpu_capacity").write_text(str(caps[c]))
        if freqs:
            (d / "cpufreq/cpuinfo_max_freq").write_text(str(freqs[c]))
            (d / "cpufreq/scaling_cur_freq").write_text(str(freqs[c] // 2))
    return str(root)


def test_cpu_topology_detection(tmp_path):
    ci = tmp_path / "cpuinfo"
    ci.write_text("flags\t: fpu sse\n")
    t = HE.cpu_topology(_sysfs(tmp_path / "a", core="0-1", atom="2-3"), str(ci), ncpu=4)
    assert t["hybrid"] and t["p_cores"] == [0, 1] and t["method"].startswith("pmu")
    t = HE.cpu_topology(_sysfs(tmp_path / "b", caps=[1024, 1024, 512, 512]), str(ci), ncpu=4)
    assert t["hybrid"] and t["p_cores"] == [0, 1] and t["method"] == "cpu_capacity"
    t = HE.cpu_topology(_sysfs(tmp_path / "c", freqs=[4700000, 4700000, 3500000, 3500000]), str(ci), ncpu=4)
    assert t["hybrid"] and t["p_cores"] == [0, 1] and t["method"] == "cpuinfo_max_freq"
    t = HE.cpu_topology(_sysfs(tmp_path / "d", freqs=[3000000] * 4), str(ci), ncpu=4)
    assert t["hybrid"] is False                                              # box-like: not hybrid, no pinning
    ci.write_text("flags\t: fpu hybrid\n")
    t = HE.cpu_topology(_sysfs(tmp_path / "e"), str(ci), ncpu=4)
    assert t["hybrid"] and t["p_cores"] is None
    pin, r = HE.plan_pinning(t)
    assert pin is None and "not determinable" in r[0]                       # fail closed


def test_plan_apply_and_match_pinning(tmp_path):
    sysfs = _sysfs(tmp_path, core="0-1", atom="2-3", freqs=[4700000, 4700000, 3500000, 3500000])
    pin, r = HE.plan_pinning(HE.cpu_topology(sysfs, "/nonexistent", ncpu=4), sysfs=sysfs)
    assert r == [] and pin["pinned"] and pin["core"] == 1 and pin["p_cores"] == [0, 1] and pin["core_max_mhz"] == 4700.0
    aff = {}
    assert HE.apply_pinning(pin, setter=lambda p, s: aff.update(s=set(s)), getter=lambda p: aff["s"], sysfs=sysfs) == []
    assert aff["s"] == {1} and pin["core_cur_mhz_at_pin"] == 2350.0
    assert HE.apply_pinning(pin, setter=lambda p, s: None, getter=lambda p: {0, 1, 2, 3})[0].startswith("E1.2: affinity")
    assert HE.pinning_matches(pin, dict(pin))[0]
    assert not HE.pinning_matches(pin, dict(pin, core=0))[0] and not HE.pinning_matches(pin, dict(pin, p_cores=[0]))[0]
    np_, r = HE.plan_pinning(dict(hybrid=False))
    assert r == [] and np_["pinned"] is False and HE.apply_pinning(np_) == []


def test_release_refuses_pinning_mismatch(tmp_path, monkeypatch):
    pj = dict(outcome="GATE_PASS", pinning=dict(pinned=True, hybrid=True, core=7, p_cores=list(range(8))))
    pp = tmp_path / "P.json"
    pp.write_text(json.dumps(pj))
    monkeypatch.setattr(S, "V83_DIR", str(tmp_path))
    rel = dict(probe=dict(file="P.json", sha256=S.sha256_file(str(pp))), pinning=dict(pj["pinning"], core=6))
    monkeypatch.setattr(RL, "CONFIG_FILE", str(tmp_path / "C.json"))
    (tmp_path / "C.json").write_text(json.dumps(dict(probe=dict(sha256=rel["probe"]["sha256"]), pinning=pj["pinning"])))
    rel["config_v8_3"] = dict(sha256=S.sha256_file(str(tmp_path / "C.json")))
    r, _ = RL.check_probe_config(rel)
    assert any(x.startswith("E1.2: pinning") for x in r)
    r, _ = RL.check_probe_config(dict(rel, pinning=pj["pinning"]))
    assert not any(x.startswith("E1.2: pinning") for x in r)


def test_production_refuses_when_core_list_differs(tmp_path, monkeypatch):
    rpr = _mod("run_production")
    bound = dict(pinned=True, hybrid=True, core=7, p_cores=list(range(8)))
    monkeypatch.setattr(RL, "check_start", lambda *a, **k: (True, [], dict(pinning=bound), {}))
    monkeypatch.setattr(GT, "start_threshold", lambda *a, **k: (99.0, None, None, []))
    monkeypatch.setattr(HE, "thread_readback_ok", lambda t: (True, []))
    monkeypatch.setattr(HE, "sidecar_capable", lambda: (True, []))
    ok = lambda: dict(ok=True, reasons=[])
    out = str(tmp_path / "run_output")
    r = rpr.main(release_path=str(tmp_path / "R.json"), out_dir=out, start_rule_fn=ok, with_sidecar=False,
                 topology=dict(hybrid=True, p_cores=[0, 1, 2, 3], method="pmu cpu_core/cpu_atom"))
    assert r["outcome"] == "NOT_LAUNCHED" and "E1.2 pinning differs" in r["reasons"][0] and not os.path.exists(out)
    r = rpr.main(release_path=str(tmp_path / "R.json"), out_dir=out, start_rule_fn=ok, with_sidecar=False,
                 topology=dict(hybrid=True, p_cores=list(range(8)), method="pmu cpu_core/cpu_atom"),
                 pin_setter=lambda p, s: None, pin_getter=lambda p: {0, 1})
    assert r["outcome"] == "NOT_LAUNCHED" and "affinity read-back" in r["reasons"][0] and not os.path.exists(out)


def test_toy_k2_reference_must_be_pinned_identically(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WM_ROOT", str(tmp_path))
    d = str(tmp_path / "probe")
    pin = dict(pinned=True, hybrid=True, core=7, p_cores=list(range(8)))
    _toy_file(d, LAP, n=1, fits=369.6, pinning=None)                        # unpinned attempt 1 (as on unicorn)
    assert GT.k2_reference(LAP, d, pin)[0] is None                         # E2.5: does not count when pinned
    assert GT.k2_reference(LAP, d, None)[0]["attempt"] == 1
    _toy_file(d, LAP, n=2, fits=380.0, pinning=dict(pin, core=6))
    assert GT.k2_reference(LAP, d, pin)[0] is None                         # other core
    _toy_file(d, LAP, n=3, fits=390.0, pinning=dict(pin))
    ref, r = GT.k2_reference(LAP, d, pin)
    assert r == [] and ref["attempt"] == 3 and ref["toy_fits"] == 390.0 and ref["pinning"]["core"] == 7


# ---------------- E2.4 baseline + dahill req 5 ----------------
def _fake_baseline(tmp_path, loads, hits_at=None, duration=20.0, interval=2.0, host=LAP, monkeypatch=None):
    t = [1000.0]
    it = iter(loads)
    procs = lambda: ([(99999, "python run_arm.py exp001")] if hits_at is not None and int(t[0] - 1000) == hits_at else [])
    monkeypatch.setattr(HE, "host_fingerprint", lambda: dict(host))
    return BL.run_baseline(probe_dir=str(tmp_path / "probe"), duration_s=duration, interval_s=interval,
                           load_fn=lambda: next(it), procs_fn=procs, sleep=lambda s: t.__setitem__(0, t[0] + s),
                           wall=lambda: t[0])


def test_baseline_attempt_and_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WM_ROOT", str(tmp_path))
    loads = [1.0 + 0.05 * i for i in range(11)]
    r = _fake_baseline(tmp_path, loads, monkeypatch=monkeypatch)
    assert r["outcome"] == "BASELINE_OK" and r["attempt"] == 1 and r["summary"]["n_samples"] == 11
    p95 = float(np.percentile(loads, 95))
    assert r["summary"]["load1_p95"] == p95 and r["summary"]["threshold_load1_max_exclusive"] == 1.5 * p95
    r2 = _fake_baseline(tmp_path, loads, hits_at=4, monkeypatch=monkeypatch)
    assert r2["outcome"] == "BASELINE_INVALID" and r2["attempt"] == 2 and "experiment" in r2["reasons"][0]
    ev = GT.events(str(tmp_path / "probe" / GT.BASELINE_LEDGER))
    assert [e["event"] for e in ev] == ["attempt_start", "attempt_end"] * 2
    # fixed once in the RELEASE draft; no later adjustment
    dp = tmp_path / "DRAFT.json"
    dp.write_text(json.dumps(dict(status="DRAFT")))
    os.chmod(str(dp), 0o444)
    bfile = str(tmp_path / "probe" / "BASELINE_attempt1.json")
    with pytest.raises(ValueError):
        BL.set_start_threshold(str(tmp_path / "probe" / "BASELINE_attempt2.json"), str(dp))   # INVALID baseline
    f = BL.set_start_threshold(bfile, str(dp))
    assert json.load(open(str(dp)))[S.START_RULE_FIELD] == f and (os.stat(str(dp)).st_mode & 0o777) == 0o444
    with pytest.raises(SystemExit):
        BL.set_start_threshold(bfile, str(dp))
    l1max, field, sha, rr = GT.start_threshold(LAP, str(dp))
    assert rr == [] and l1max == 1.5 * p95 and field == f and len(sha) == 64
    # start rule uses it
    assert HE.start_rule(patterns=[], load_fn=lambda: l1max - 1e-9, mem_fn=lambda: 8 << 20, procs_fn=lambda: [],
                         load_max=l1max)["ok"]
    assert not HE.start_rule(patterns=[], load_fn=lambda: l1max, mem_fn=lambda: 8 << 20, procs_fn=lambda: [],
                             load_max=l1max)["ok"]


def test_start_threshold_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WM_ROOT", str(tmp_path))
    _fake_baseline(tmp_path, [1.0] * 11, monkeypatch=monkeypatch)
    f = GT.threshold_field_from_baseline(str(tmp_path / "probe" / "BASELINE_attempt1.json"))
    assert GT.start_threshold(LAP, field=f)[3] == []
    assert GT.start_threshold(dict(LAP, hostname="other"), field=f)[3]                  # other host
    assert GT.start_threshold(LAP, field=dict(f, load1_max_exclusive=2.0))[3]           # changed value
    assert GT.start_threshold(LAP, field=dict(f, factor=2.0))[3]
    os.chmod(str(tmp_path / "probe" / "BASELINE_attempt1.json"), 0o644)
    open(str(tmp_path / "probe" / "BASELINE_attempt1.json"), "a").write(" ")
    assert any("hash mismatch" in x for x in GT.start_threshold(LAP, field=f)[3])       # tampered baseline
    dp = tmp_path / "D.json"
    dp.write_text("{}")
    l1, _, _, r = GT.start_threshold(LAP, str(dp))
    assert l1 is None and "absent for non-box host" in r[0]                            # non-box: refused
    l1, fld, _, r = GT.start_threshold(BOX, str(dp))
    assert l1 == 1.0 and fld is None and r == []                                       # box: old < 1.0 (E2.4)


def test_start_rule_default_box_threshold():
    G = 1024 * 1024
    assert not HE.start_rule(patterns=[], load_fn=lambda: 1.0, mem_fn=lambda: 5 * G, procs_fn=lambda: [])["ok"]
    assert HE.start_rule(patterns=[], load_fn=lambda: 1.0, mem_fn=lambda: 5 * G, procs_fn=lambda: [], load_max=2.0)["ok"]


def test_process_starts_field(tmp_path):
    # hermetic: never fall back to the real RELEASE draft (value=None reads draft_path)
    missing = str(tmp_path / "no_draft.json")
    assert GT.process_starts(value=1) == (1, [])
    for bad in (None, 0, True, 1.0):
        assert GT.process_starts(draft_path=missing, value=bad)[0] is None
    assert "absent" in GT.process_starts("/nonexistent/D.json")[1][-1]


# ---------------- E2.3 sidecar ----------------
def test_probe_sidecar_interval_and_summary(tmp_path):
    assert S.PROBE_SIDECAR_INTERVAL_S <= 2.0 and S.SIDECAR_INTERVAL_S == 30.0
    src = open(os.path.join(S.CODE_DIR, "run_probe.py")).read()
    assert "interval=S.PROBE_SIDECAR_INTERVAL_S" in src
    assert "interval=S.PROBE_SIDECAR_INTERVAL_S" in open(os.path.join(S.CODE_DIR, "run_preflight.py")).read()
    p = tmp_path / "L.txt"
    rows = [dict(event="sidecar_start", wall_unix=0)]
    for i, (l1, mhz, st) in enumerate([(0.5, 3000, None), (1.7, 2100, 0.0), (0.9, 2500, 0.01), (9.0, 100, 0.5)]):
        rows.append(dict(event="sample", wall_unix=10 + 2 * i, load_1_5_15=[l1, 0, 0], cpu_mhz_mean=mhz,
                         steal_frac_since_prev=st, steal_ticks=5 * i))
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = GT.sidecar_summary(str(p), 10, 14)                                   # last sample outside the window
    assert s["n_samples"] == 3 and s["load1_max"] == 1.7 and s["cpu_mhz_mean_min"] == 2100 and s["steal_frac_max"] == 0.01
    assert s["steal_ticks_delta"] == 10


# ---------------- E1.5 / E2.7 ----------------
def test_e1_5_blocks_after_linearity_failure_under_erratum(tmp_path, monkeypatch):
    rpb = _mod("run_probe")
    d = tmp_path / "probe"
    d.mkdir()
    led = str(d / GT.PROBE_LEDGER)
    GT.append(led, "attempt_start", attempt=1)
    GT.append(led, "attempt_end", attempt=1, outcome="PROBE_DIAGNOSIS")     # pre-erratum (no erratum field)
    assert GT.e1_5_blocked(led) == []
    GT.append(led, "attempt_start", attempt=2, erratum=S.ERRATUM)
    GT.append(led, "attempt_end", attempt=2, outcome="PROBE_DIAGNOSIS", diagnosis_kind="cap", erratum=S.ERRATUM)
    assert GT.e1_5_blocked(led) == []                                        # a cap diagnosis does not block
    GT.append(led, "attempt_start", attempt=3, erratum=S.ERRATUM)
    GT.append(led, "attempt_end", attempt=3, outcome="PROBE_DIAGNOSIS", diagnosis_kind="linearity", erratum=S.ERRATUM)
    assert GT.e1_5_blocked(led)
    ok = lambda: dict(ok=True, reasons=[], load1=0.1)
    r = rpb.main(probe_dir=str(d), config_path=str(tmp_path / "C.json"), start_rule_fn=ok, with_sidecar=False, t0=None,
                 topology=dict(hybrid=False))
    assert r["outcome"] == "PROBE_BLOCKED_E1_5" and r["attempt"] == 4 and r["v8_3_status"].startswith("NOT_LAUNCHED")
    assert GT.events(led)[-1]["outcome"] == "PROBE_BLOCKED_E1_5"


def test_run_probe_records_erratum_and_refuses_without_release_fields(tmp_path, monkeypatch):
    rpb = _mod("run_probe")
    monkeypatch.setattr(HE, "host_fingerprint", lambda: dict(LAP))
    dp = tmp_path / "D.json"
    dp.write_text("{}")
    ok = lambda: dict(ok=True, reasons=[], load1=0.1)
    r = rpb.main(probe_dir=str(tmp_path / "probe"), config_path=str(tmp_path / "C.json"), start_rule_fn=ok,
                 with_sidecar=False, t0=None, draft_path=str(dp), topology=dict(hybrid=True, p_cores=None, reason="x"))
    assert r["outcome"] == "PROBE_NOT_STARTED" and r["erratum"] == S.ERRATUM
    assert r["erratum_sha256"] == S.V83["erratum_sha256"] and r["erratum2_sha256"] == S.V83["erratum2_sha256"]
    rs = " | ".join(r["reasons"])
    assert "start-rule threshold" in rs and "production_process_starts" in rs and "P-cores not determinable" in rs
