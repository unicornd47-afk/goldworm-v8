"""v8.3 mechanisms (PLAN_v8.3 r3 C1-C7, K1, K2; dahill PLAN_REVIEW_v8.3_r3 A1, A2). No probe, no production run:
only tiny CPU work on train/val data seeds (probe mode) or preflight seeds 1-4 (smoke)."""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from v81 import hostenv as HE
from v81 import models_torch as MT
from v81 import pipeline as PP
from v81 import preflight as PF
from v81 import probe as PB
from v81 import release as RL
from v81 import seeds as SD
from v81 import spec as S
from v81 import stats as ST
from v81.budget import Budget, CapExceeded
from v81.ledger import Ledger, write_once

SIDECAR_DIR = os.path.join(S.V83_DIR, "sidecar")


def _toy():
    with open(os.path.join(S.PLAN_DIR, S.V83["toy_preflight_file"])) as f:
        return json.load(f)


# ---------------- bundle ----------------
def test_v83_bundle_and_constants():
    h = S.verify_v83_bundle()
    assert h["PLAN_v8.3.md"] == "e2debca54bb2d303b910eb5d32d691b14652bd61422b53e3eadf757d4c5784ec"
    assert h["PLAN_REVIEW_v8.3_r3.json"] == "43a6e207bc104097c302b652fdb6e7124cba068d876da8d976455137ae698b3f"
    S.verify_plan_hashes()                                                  # science = v8.2 bundle, unchanged
    assert S.sha256_file(os.path.join(S.V83_DIR, S.V83["seed_evidence_file"])) == S.V83["seed_evidence_sha256"]
    assert S.sha256_file(os.path.join(S.PLAN_DIR, S.V83["toy_preflight_file"])) == S.V83["toy_preflight_sha256"]
    assert S.PROBE_CAP_CPU_S == 450 and S.LINEARITY_WINDOW == (0.87, 1.15) and S.CAP_FACTOR == 1.25
    assert S.K1_GLOBAL_CAP == 3600 and sum(S.FIXED_CAPS.values()) == 260 and S.K2_TOY_FITS == 336.8
    assert (3600 - 260) / 1.25 == 2672
    assert S.FIT_CHECK_EVERY == 10 and S.LEDGER_UPDATE_EVERY == 100 and S.BOOT_BLOCK == 1000
    assert S.V83_LADDER == ("a", "b", "c")


# ---------------- K1 ----------------
def test_k1_global_abort_at_3600_even_if_every_phase_under_cap():
    t = [0.0]
    b = Budget(caps={k: 1e9 for k in S.CAPS}, global_cap=1e9, clock=lambda: t[0], t0=0.0)
    with pytest.raises(CapExceeded, match="K1"):
        with b.phase("fits"):
            t[0] = 3600.0
            b.check()                       # exactly 3600 is not > 3600
            t[0] = 3600.01
            b.check()
    assert b.used["fits"] < b.caps["fits"]   # the phase itself was under its cap


def test_k1_static_check_at_phase_end_and_finalize():
    t = [0.0]
    b = Budget(caps={k: 1e9 for k in S.CAPS}, global_cap=1e9, clock=lambda: t[0], t0=0.0)
    with pytest.raises(CapExceeded, match="K1"):
        with b.phase("data"):
            t[0] = 3700.0                   # no check() inside the phase: caught by the static check at phase end
    t2 = [0.0]
    b2 = Budget(caps={k: 1e9 for k in S.CAPS}, global_cap=1e9, clock=lambda: t2[0], t0=0.0)
    t2[0] = 3601.0
    f = b2.finalize()
    assert f["k1_respected"] is False and f["all_caps_respected"] is False and f["k1_cap"] == 3600.0


def test_global_cap_is_sum_of_caps_and_k1_both_enforced():
    t = [0.0]
    caps = dict(S.FIXED_CAPS, fits=500, tuning=500, test=500, in_distribution=100, adaptation=300)
    b = Budget(caps=caps, global_cap=sum(caps.values()), clock=lambda: t[0], t0=0.0)
    assert b.global_cap == 2160 and b.k1_cap == 3600
    with pytest.raises(CapExceeded, match="global"):
        with b.phase("overhead"):           # overhead is not checked live as a phase here; global sum is
            t[0] = 2160.5
            b._check_global(b.total())


# ---------------- C6 / C6b ----------------
def test_budget_events_and_jump_tracking():
    t = [0.0]
    ev = []
    b = Budget(caps={k: 100.0 for k in S.CAPS}, global_cap=3600, clock=lambda: t[0], t0=0.0,
               on_event=lambda e, **k: ev.append((e, k)), thread_readout=lambda: {"torch_num_threads": 1})
    with b.phase("fits"):
        t[0] = 1.0
        b.check()
        t[0] = 4.6
        b.check()
    assert [e for e, _ in ev] == ["phase_start", "phase_end"]
    end = ev[1][1]
    assert end["phase"] == "fits" and end["phase_cpu_s"] == 4.6 and end["n_checks"] == 2
    assert abs(end["max_check_jump_cpu_s"] - 3.6) < 1e-12 and end["threads"] == {"torch_num_threads": 1}
    assert "wall_unix" in end and "process_cpu_s" in ev[0][1]


def test_train_checks_every_10_updates_and_progress_every_update():
    m = MT.build("GRU", 32, None, 0)
    ctx, fut, tgt = torch.zeros(20, 48, 6).numpy(), torch.zeros(20, 12, 3).numpy(), torch.zeros(20, 12).numpy()
    bidx = np.zeros((25, S.BATCH), dtype=np.int64)
    calls, prog = [], []
    MT.train(m, ctx, fut, tgt, bidx, 25, lambda t: 1e-3, budget_check=lambda: calls.append(1), check_every=10,
             progress_cb=prog.append, log_every=0)
    assert len(calls) == 3 and prog == list(range(1, 26))            # at 10, 20 and the final update 25


def test_bootstrap_blocks_of_1000_identical_results():
    rng = np.random.default_rng(3)
    E1, E2, D1, D2 = (rng.uniform(1, 5, 320) for _ in range(4))
    cl = np.repeat(np.arange(40), 8)
    idx = ST.boot_matrix(S.BOOT_SEEDS["primary"], 40)
    n = []
    a = ST.boot_S_dD(E1, E2, D1, D2, cl, idx, check=lambda: n.append(1))
    b = ST.boot_S_dD(E1, E2, D1, D2, cl, idx)
    assert len(n) == 10 and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    # identical to the single vectorised v8.2 pass
    ii = idx.astype(np.int64)
    cE_w, cE_c = ST.cluster_sums(E1, cl, 40), ST.cluster_sums(E2, cl, 40)
    assert np.array_equal(a[0], 1.0 - cE_w[ii].sum(1) / cE_c[ii].sum(1))
    per = {k: rng.uniform(0, 3, 40) for k in S.ARMS}
    m = []
    c1 = ST.mse_contrast_ci(per, idx, check=lambda: m.append(1))
    c2 = ST.mse_contrast_ci(per, idx)
    assert c1 == c2 and len(m) == 10 * 6
    d = per["WM"] - per["GRU"]
    assert c1["WM-GRU"]["ci"] == list(ST.ci(d[ii].mean(1)))


# ---------------- C7 ----------------
def test_write_once(tmp_path):
    L = Ledger(str(tmp_path))
    L.write_sealed("X.json", {"a": 1})
    with pytest.raises(FileExistsError):
        L.write_sealed("X.json", {"a": 2})
    p = str(tmp_path / "normalization.json")
    write_once(p, "{}")
    with pytest.raises(FileExistsError):
        write_once(p, "{}")
    assert open(p).read() == "{}"


# ---------------- probe seed guard ----------------
def test_probe_seed_guard_train_val_only():
    G = SD.SeedGuard("probe")
    G.check(S.TRAIN_SEEDS[0][0], "train")
    G.check(S.VAL_SEEDS[0][0], "val")
    G.check(S.TRAIN_SEEDS[0][1], "probe_train_entry")
    for seed in (S.TUNING_SEEDS[0], S.TEST_SEEDS[0], S.ADAPT_SEEDS[0], 1):
        with pytest.raises(SD.SeedGuardError):
            G.check(seed, "probe_train_entry")
    for purpose in ("tuning", "test_heldout", "adapt_eval", "adaptation_data"):
        with pytest.raises(SD.SeedGuardError):
            G.check(S.TRAIN_SEEDS[0][0], purpose)


# ---------------- C1 ----------------
def test_thread_readback():
    r = HE.thread_readout()
    assert HE.thread_readback_ok(r)[0]                                   # conftest pins 1; torch set to 1 below
    bad = dict(r, OMP_NUM_THREADS="4", torch_num_threads=8)
    ok, which = HE.thread_readback_ok(bad)
    assert not ok and set(which) == {"OMP_NUM_THREADS", "torch_num_threads"}


# ---------------- C4 ----------------
def test_start_rule_conditions():
    G = 1024 * 1024
    ok = HE.start_rule(patterns=["exp001"], load_fn=lambda: 0.5, mem_fn=lambda: 5 * G, procs_fn=lambda: [(1, "init")])
    assert ok["ok"] and ok["load1"] == 0.5 and ok["mem_available_kb"] == 5 * G
    r = HE.start_rule(patterns=["exp001"], load_fn=lambda: 1.0, mem_fn=lambda: 5 * G, procs_fn=lambda: [])
    assert not r["ok"] and "load1" in r["reasons"][0]                    # strict < 1.0
    r = HE.start_rule(patterns=["exp001"], load_fn=lambda: 0.1, mem_fn=lambda: 4 * G - 1, procs_fn=lambda: [])
    assert not r["ok"] and "MemAvailable" in r["reasons"][0]
    r = HE.start_rule(patterns=["exp001", "run_arm.py"], load_fn=lambda: 0.1, mem_fn=lambda: 8 * G,
                      procs_fn=lambda: [(99999, "python run_arm.py --arm G --seed 11")])
    assert not r["ok"] and r["excluded_hits"][0][0] == 99999
    r = HE.start_rule(patterns=["x"], load_fn=lambda: None, mem_fn=lambda: None, procs_fn=lambda: None)
    assert not r["ok"] and len(r["reasons"]) == 3                        # unreadable -> fail closed
    assert S.START_MEMAVAIL_MIN_KB == 4 * G


def test_start_rule_ignores_own_process():
    r = HE.start_rule(patterns=["pytest"], load_fn=lambda: 0.1, mem_fn=lambda: 8 * 1024 * 1024,
                      procs_fn=lambda: [(os.getpid(), "python -m pytest")])
    assert r["ok"]


# ---------------- A2 ----------------
def test_host_fingerprint_and_same_host():
    h = HE.host_fingerprint()
    assert set(HE.FINGERPRINT_KEYS) <= set(h) and h["cpu_count"] == os.cpu_count()
    assert HE.same_host(h, dict(h))[0]
    ok, diff = HE.same_host(h, dict(h, hostname="unicorn-other"))
    assert not ok and diff == ["hostname"]
    assert not HE.same_host(h, dict(h, cpu_count=(h["cpu_count"] or 0) + 1))[0]
    assert not HE.same_host(None, h)[0]


# ---------------- C5 sidecar ----------------
def _sidecar():
    sys.path.insert(0, SIDECAR_DIR)
    import load_sidecar
    return load_sidecar


def test_sidecar_samples_fields_and_stops_with_pid(tmp_path):
    sc = _sidecar()
    child = subprocess.Popen([sys.executable, "-c", "import time; t=time.time()\nwhile time.time()-t<0.6: pass"])
    out = str(tmp_path / "LOAD.txt")
    n = sc.run(child.pid, out, interval=0.05)
    child.wait()
    recs = [json.loads(l) for l in open(out)]
    assert recs[0]["event"] == "sidecar_start" and recs[-1]["event"] == "sidecar_end" and n >= 2
    s = [r for r in recs if r["event"] == "sample"]
    for k in ("wall_unix", "load_1_5_15", "runnable", "mem_available_kb", "steal_ticks", "steal_frac_since_prev",
              "cpu_mhz_mean", "run_proc_cpu_s"):
        assert k in s[0], k
    if os.path.exists("/proc/stat"):
        assert s[-1]["run_proc_cpu_s"] is not None and s[1]["steal_frac_since_prev"] is not None


def test_sidecar_gap_report(tmp_path):
    sc = _sidecar()
    p = tmp_path / "L.txt"
    ts = [0, 30, 60, 200, 230, 330.5]
    p.write_text("\n".join(json.dumps(dict(event="sample", wall_unix=t)) for t in ts) + "\n")
    g = sc.gaps(str(p))
    assert [x["gap_s"] for x in g] == [140, 100.5]


def test_sidecar_launcher_is_separate_session(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.8)"])
    out = str(tmp_path / "L.txt")
    p = HE.start_sidecar(child.pid, out, interval=0.1)
    assert os.getsid(p.pid) != os.getsid(0)                           # setsid: own session
    child.wait()
    p.wait(timeout=10)
    recs = [json.loads(l) for l in open(out)]
    assert recs[-1]["event"] == "sidecar_end" and recs[-1]["samples"] >= 1


# ---------------- C3 / K2 / A1 decisions (pure) ----------------
def test_episode_projection_equals_v82_preflight_formulas():
    toy = _toy()
    m = toy["measurements"]
    for ladder in ((), ("a",), ("a", "b"), ("a", "b", "c")):
        mine = PB.toy_episode_projection(toy, ladder)
        ref, _, _ = PF.project(m, toy["setup_cpu_s"], ladder)
        for k in S.K2_PHASES:
            assert mine[k] == ref[k], (ladder, k)
    p0 = PB.toy_episode_projection(toy, ())
    for k in S.K2_PHASES:
        assert p0[k] == toy["projection_cpu_s"][k]                       # reproduces the bound v8.2 file


def _pm(per_upd=0.01, ep=0.05, lin=0.5, ratio=1.0, full=None):
    unit = {}
    for a in S.ARMS:
        for s in S.structures(a):
            if a == "LIN":
                unit["LIN_%s" % s] = dict(fit_cpu_s=lin, refit_cpu_s_scaled_42ep=lin * 42 / 32, episode_cpu_s=ep,
                                          per_update_cpu_s=None)
            else:
                n = 2000 if (a, s) == ("WM", 96) else 200
                unit["%s_%s" % (a, s)] = dict(fit_cpu_s=per_upd * n + 0.1, per_update_cpu_s=per_upd, episode_cpu_s=ep)
    if full is not None:
        unit["WM_96"]["fit_cpu_s"] = full
    return dict(full_fit=["WM", 96], fraction=0.1, unit=unit, windows_per_replicate_cpu_s=0.2,
                scenario_build_cpu_s=0.01, pi_episode_cpu_s=0.01, data_episode_cpu_s=0.005, complete=True,
                linearity=dict(ratio=ratio, ok=PB.linearity_ok(ratio)))


def test_a1_linearity_window_and_cap_abort_give_diagnosis_without_projection():
    assert PB.linearity_ok(0.87) and PB.linearity_ok(1.15) and PB.linearity_ok(1.0)
    assert not PB.linearity_ok(0.8699) and not PB.linearity_ok(1.1501) and not PB.linearity_ok(None)
    toy = _toy()
    d = PB.decide(_pm(ratio=1.2), toy)
    assert d["outcome"] == "PROBE_DIAGNOSIS" and d["projection"] is None and d["caps"] is None
    d = PB.decide(_pm(), toy, aborted="probe 450.10 > cap 450.0 CPU-s")
    assert d["outcome"] == "PROBE_DIAGNOSIS" and d["projection"] is None
    pm = _pm()
    pm["complete"] = False
    assert PB.decide(pm, toy)["outcome"] == "PROBE_DIAGNOSIS"


def test_c3_caps_one_budget_and_gate():
    toy = _toy()
    d = PB.decide(_pm(per_upd=0.005, ep=0.03), toy)
    assert d["outcome"] == "GATE_PASS" and d["ladder_applied"] == []
    caps, proj = d["caps"], d["projection"]
    for k in S.MEASURED_PHASES:
        assert caps[k] == math.ceil(1.25 * proj[k])
    for k, v in S.FIXED_CAPS.items():
        assert caps[k] == v
    assert d["global_cap"] == sum(caps.values()) <= 3600 and d["k1_global_cap"] == 3600
    f = d["fits"]
    assert f["replicates"] == 3 and abs(f["total"] - 3 * f["per_replicate"]) < 1e-9
    t = f["terms"]
    assert abs(t["full_fit"] - (0.005 * 2000 + 0.1)) < 1e-12
    assert abs(t["other_neural_x10"] - 8 * 10 * (0.005 * 200 + 0.1)) < 1e-9 and abs(t["lin"] - 1.5) < 1e-12


def test_c3_ladder_order_and_not_launched():
    toy = _toy()
    # episodes expensive enough that the unladdered sum fails but some ladder step passes
    for ep in np.linspace(0.05, 0.4, 36):
        d = PB.decide(_pm(per_upd=0.004, ep=float(ep)), toy)
        lad = d["ladder_applied"]
        assert lad == list(S.V83_LADDER[:len(lad)])                       # a, then b, then c - in order
        if d["outcome"] == "GATE_PASS":
            assert d["tries"][-1]["sum_of_caps"] <= 3600
            assert all(not t["gate_pass"] for t in d["tries"][:-1])
    d = PB.decide(_pm(per_upd=0.004, ep=0.03, full=2000.0), toy)            # fits alone too big: ladder cannot help
    assert d["outcome"] == "NOT_LAUNCHED" and d["caps"] is None and d["ladder_applied"] == ["a", "b", "c"]


def test_k2_rule():
    toy = {"tuning": 800.0, "test": 500.0, "in_distribution": 60.0, "adaptation": 250.0}
    meas = {"tuning": 900.0, "test": 700.0, "in_distribution": 100.0, "adaptation": 300.0}
    out, rec = PB.apply_k2(meas, toy, R=1.5)                                # R not > 1.5: never replaced
    assert out == meas and not any(r["replaced"] for r in rec.values())
    out, rec = PB.apply_k2(meas, toy, R=2.0)                                # toy x R = 1600/1000/120/500
    assert out["tuning"] == 1600.0 and rec["tuning"]["replaced"]            # 900 < 0.8 x 1600
    assert out["test"] == 1000.0 and rec["test"]["replaced"]               # 700 < 800
    assert out["in_distribution"] == 100.0 and not rec["in_distribution"]["replaced"]   # 100 >= 96
    assert out["adaptation"] == 500.0 and rec["adaptation"]["replaced"]
    assert rec["tuning"]["toy"] == 800.0 and rec["tuning"]["toy_x_R"] == 1600.0


def test_probe_report_contains_R_and_toy_projections():
    toy = _toy()
    pm = _pm(per_upd=0.02, ep=0.03)                                         # fits ~ 3 x (40.3 + 8 x 41 + 1.5)
    d = PB.decide(pm, toy)
    assert abs(d["R"] - d["fits"]["total"] / 336.8) < 1e-12 and d["R"] > 1.5
    t0 = d["tries"][0]
    for k in S.K2_PHASES:
        assert t0["toy_projection"][k] == PB.toy_episode_projection(toy, ())[k]
        assert t0["k2"][k]["toy_x_R"] == t0["toy_projection"][k] * d["R"]
    assert any(t0["k2"][k]["replaced"] for k in S.K2_PHASES)               # cheap measured episodes -> toy x R used


# ---------------- probe measurement (tiny, train/val seeds only) ----------------
TINY = dict(n_updates=20, fraction=0.1, n_train=4, n_val=2,
            structures={"WM": (32,), "GRU": (32,), "LIN": (3,), "RC_only": (3,)}, full_fit=("WM", 32))


def test_measure_probe_stops_on_linearity_failure(monkeypatch):
    monkeypatch.setattr(PB, "linearity_ok", lambda r: False)
    pm = PB.measure_probe(PB.ProbeBudget(cap=1e9), **TINY)
    assert "complete" not in pm and list(pm["unit"]) == ["WM_32"] and pm["linearity"]["ok"] is False
    assert PB.decide(pm, _toy())["outcome"] == "PROBE_DIAGNOSIS"


def test_measure_probe_tiny(monkeypatch):
    monkeypatch.setattr(PB, "linearity_ok", lambda r: True)   # 20-update toy timings are noise; window tested above
    b = PB.ProbeBudget(cap=1e9)
    ev = []
    pm = PB.measure_probe(b, event=lambda e, **k: ev.append(e), n_updates=20, fraction=0.1, n_train=4, n_val=2,
                          structures={"WM": (32,), "GRU": (32,), "LIN": (3,), "RC_only": (3,)}, full_fit=("WM", 32))
    assert pm["complete"] and pm["linearity"]["ratio"] is not None
    assert pm["unit"]["WM_32"]["updates_run"] == 20 and pm["unit"]["GRU_32"]["updates_run"] == 2
    assert pm["unit"]["LIN_3"]["kind"] == "full_closed_form"
    for k in ("WM_32", "GRU_32", "LIN_3", "RC_only_3"):
        assert pm["unit"][k]["episodes"] == 2 and pm["unit"][k]["episode_cpu_s"] > 0
    seeds = {sc["seed"] for sc in pm["episode_scenarios"]}
    assert seeds <= set(S.TRAIN_SEEDS[0]) and all(sc["purpose"] == "probe_train_entry" for sc in pm["episode_scenarios"])
    assert "probe_linearity" in ev and "probe_fit_end" in ev and b.n_checks > 0


def test_measure_probe_cap_abort_keeps_partial_data():
    t = [0.0]

    def clk():
        t[0] += 1.0
        return t[0]
    b = PB.ProbeBudget(cap=5.0, clock=clk, t0=0.0)
    pm = {}
    with pytest.raises(PB.ProbeCapExceeded):
        PB.measure_probe(b, out=pm, n_updates=20, n_train=4, n_val=2,
                         structures={"WM": (32,), "GRU": (32,), "LIN": (3,), "RC_only": (3,)}, full_fit=("WM", 32))
    assert pm["train_seeds"] and "complete" not in pm
    assert PB.decide(pm, _toy(), aborted="cap")["outcome"] == "PROBE_DIAGNOSIS"


# ---------------- run_probe attempts (A1) ----------------
def _run_probe_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rpb", os.path.join(S.CODE_DIR, "run_probe.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_run_probe_attempt_counter_never_overwrites(tmp_path):
    rpb = _run_probe_module()
    bad = lambda: dict(ok=False, reasons=["load1 3.00 >= 1.0"], load1=3.0)
    d, cfg = str(tmp_path / "probe"), str(tmp_path / "CONFIG_v8.3.json")
    r1 = rpb.main(probe_dir=d, config_path=cfg, start_rule_fn=bad, with_sidecar=False, t0=None)
    first = open(os.path.join(d, "PROBE_v8.3_attempt1.json")).read()
    r2 = rpb.main(probe_dir=d, config_path=cfg, start_rule_fn=bad, with_sidecar=False, t0=None)
    assert r1["outcome"] == r2["outcome"] == "PROBE_NOT_STARTED"
    assert r1["attempt"] == 1 and r2["attempt"] == 2
    assert open(os.path.join(d, "PROBE_v8.3_attempt1.json")).read() == first
    ev = [json.loads(l) for l in open(os.path.join(d, "PROBE_LEDGER.jsonl"))]
    assert [e["event"] for e in ev] == ["attempt_start", "attempt_end"] * 2
    assert ev[0]["host"]["hostname"] and ev[0]["start_rule"]["load1"] == 3.0 and "threads" in ev[0]
    assert any("C4 start rule" in x for x in r1["reasons"])
    assert r1["host_fingerprint"] == HE.host_fingerprint() and "A1" in r1["rules"] and "A2" in r1["rules"]
    open(cfg, "w").write("{}")
    with pytest.raises(SystemExit):
        rpb.main(probe_dir=d, config_path=cfg, start_rule_fn=bad, with_sidecar=False, t0=None)


# ---------------- RELEASE: probe/CONFIG binding and A2 host refusal ----------------
def test_release_probe_config_binding_and_host_refusal(tmp_path, monkeypatch):
    from v81.ledger import canon
    monkeypatch.setattr(S, "V83_DIR", str(tmp_path))
    cfgp = str(tmp_path / "CONFIG_v8.3.json")
    monkeypatch.setattr(RL, "CONFIG_FILE", cfgp)
    host = HE.host_fingerprint()
    pm = _pm(per_upd=0.005, ep=0.03)
    dec = PB.decide(pm, _toy())
    (tmp_path / "probe").mkdir()
    pp = tmp_path / "probe" / "PROBE_v8.3_attempt1.json"
    from v81 import gates as GT
    ref = dict(kind="box_v8.2", file="v8.2/PREFLIGHT_PROJECTION.json", sha256=S.V83["toy_preflight_sha256"],
               toy_fits=336.8, attempt=None)
    host = dict(S.BOX_FINGERPRINT, platform="x")
    pin = dict(pinned=False, hybrid=False, core=None, p_cores=None)
    e1 = dict(erratum_sha256=S.V83["erratum_sha256"], erratum2_sha256=S.V83["erratum2_sha256"])
    pp.write_text(canon(dict(outcome="GATE_PASS", measurements=pm, host_fingerprint=host, k2_reference=ref, pinning=pin,
                             start_rule_threshold=None, production_process_starts=1, **e1)))
    psha = S.sha256_file(str(pp))
    cfg = dict(probe=dict(file="probe/PROBE_v8.3_attempt1.json", sha256=psha),
               caps_cpu_s=dict(per_phase=dec["caps"], global_cap=dec["global_cap"], k1_global_cap=3600.0),
               ladder_applied=dec["ladder_applied"], host_fingerprint=host, pinning=pin)
    open(cfgp, "w").write(canon(cfg))
    rel = dict(probe=dict(file="probe/PROBE_v8.3_attempt1.json", sha256=psha),
               config_v8_3=dict(sha256=S.sha256_file(cfgp)), host_fingerprint=host, k2_reference=ref, pinning=pin,
               production_process_starts=1)
    r, c = RL.check_probe_config(rel, current_host=host)
    assert r == [] and c["caps_cpu_s"]["per_phase"] == dec["caps"]
    r, _ = RL.check_probe_config(dict(rel, k2_reference=None), current_host=host)
    assert any("K2 reference" in x for x in r)                              # K2 reference must be bound in RELEASE
    r, _ = RL.check_probe_config(rel, current_host=dict(host, hostname="unicorn"))
    assert any(x.startswith("A2: this host") for x in r)
    bad = dict(cfg, caps_cpu_s=dict(cfg["caps_cpu_s"], per_phase=dict(dec["caps"], fits=dec["caps"]["fits"] + 1)))
    open(cfgp, "w").write(canon(bad))
    rel["config_v8_3"]["sha256"] = S.sha256_file(cfgp)
    r, _ = RL.check_probe_config(rel, current_host=host)
    assert any("differ from the decision recomputed" in x for x in r)


# ---------------- pipeline ledger events (smoke, preflight seeds 1-4) ----------------
def test_smoke_ledger_phase_fit_and_update_events(tmp_path):
    out = tmp_path / "smoke"
    run = PP.Run(PP.smoke_design(n_updates=100, adapt_updates=2), str(out))
    final = run.run_all()
    ev = [json.loads(l) for l in (out / "LEDGER.jsonl").read_text().splitlines()]
    names = [e["event"] for e in ev]
    assert names[0] == "run_start" and names[-1] == "final" and "process_cpu_s" in ev[-1]
    assert ev[0]["threads"]["torch_num_threads"] == 1
    starts = [e["phase"] for e in ev if e["event"] == "phase_start"]
    ends = [e for e in ev if e["event"] == "phase_end"]
    assert starts == [e["phase"] for e in ends] and {"data", "fits", "tuning", "test", "audit"} <= set(starts)
    for e in ends:
        assert {"phase_cpu_s", "process_cpu_s", "wall_unix", "threads", "n_checks", "max_check_jump_cpu_s"} <= set(e)
    fe = [e for e in ev if e["event"] == "fit_end"]
    assert len(fe) == 4 * 3 and all("fit_cpu_s" in e and "updates" in e for e in fe)
    fp = [e for e in ev if e["event"] == "fit_progress"]
    assert len(fp) == 3 * 3 and all(e["updates"] == 100 for e in fp)        # 3 neural fits x 3 reps, every 100
    fits_end = [e for e in ends if e["phase"] == "fits"][0]
    assert fits_end["n_checks"] == 3 * 3 * 10 + 3 + 1                       # every 10 updates + 1 per LIN fit + warm-up
    wu = [e for e in ev if e["event"] == "warmup"]
    assert len(wu) == 1 and wu[0]["updates"] == 5 and wu[0]["rep"] == 0         # E2.1: once per process start
    assert final["budget"]["k1_cap"] == 1e9 and "max_check_jump_cpu_s" in final["budget"]
    with pytest.raises(FileExistsError):
        run.save_json("normalization.json", {}, once=True)


# =====================================================================================================================
# uni 11:12 interpretations + dahill CODE_PASS (19bea022) rulings
# =====================================================================================================================
from v81 import gates as GT  # noqa: E402


def _nan_toy():
    """A toy reference whose every per-unit number is NaN (and fits = 336.8): must not leak into the projection."""
    toy = json.loads(json.dumps(_toy()))
    m = toy["measurements"]
    for u in m["unit"].values():
        for k in list(u):
            if isinstance(u[k], (int, float)) and not isinstance(u[k], bool):
                u[k] = float("nan")
    for k in ("scenario_build_cpu_s", "pi_episode_cpu_s", "data_episode_cpu_s"):
        m[k] = float("nan")
    return toy


def test_projection_uses_no_toy_input():
    """uni (4): every per-unit cost in the projection comes from the probe; the toy only feeds the K2 comparison."""
    pm = _pm(per_upd=0.005, ep=0.03)
    a = PB.decide(pm, _toy())
    b = PB.decide(pm, _nan_toy())
    assert a["fits"] == b["fits"]
    for ta, tb in zip(a["tries"], b["tries"]):
        assert ta["measured_projection"] == tb["measured_projection"]
        assert ta["measured_projection_detail"] == tb["measured_projection_detail"]
    assert a["R"] <= 1.5 and a["projection"] == b["projection"] and a["caps"] == b["caps"]   # no K2 -> identical
    # the projection functions take only probe fields
    import inspect
    src = inspect.getsource(PB.decide)
    assert "measured_episode_phases(pm, ladder)" in src and "fits_projection(pm)" in src
    assert "toy_episode_projection(toy, ())" in src
    for fn in (PB.measured_episode_phases, PB.fits_projection, PB.episode_count_projection):
        assert "toy" not in inspect.getsource(fn)


def test_projection_inputs_all_produced_by_probe(monkeypatch):
    """Every per-unit key the projection reads is produced by measure_probe (tiny, all 12 structures)."""
    monkeypatch.setattr(PB, "linearity_ok", lambda r: True)
    pm = PB.measure_probe(PB.ProbeBudget(cap=1e9), n_updates=20, fraction=0.1, n_train=4, n_val=2, full_fit=("WM", 96))
    d = PB.decide(pm, _nan_toy())
    for k in S.MEASURED_PHASES:
        assert np.isfinite(d["projection"][k]), k
    f2 = pm["full_fit_repeats"][-1]["fit_cpu_s"]
    assert d["full_fit_total_cpu_s"] == f2 > 0 and pm["linearity"]["full_fit_total_cpu_s"] == f2   # ruling 3 (gating fit)
    assert d["full_fit_total_cpu_s_per_fit"] == [u["fit_cpu_s"] for u in pm["full_fit_repeats"]]
    assert len(pm["full_fit_repeats"]) == 2 and pm["unit"]["WM_96"] is pm["full_fit_repeats"][0]


def test_episode_phase_max_rule_reports_both():
    """dahill ruling 4: max(v8.2 formula, episodes x episode cost), both values and the winner reported."""
    pm = _pm(per_upd=0.005, ep=0.03)
    proj, rec = PB.measured_episode_phases(pm)
    f = PB.episode_phase_projection(pm["unit"], pm["scenario_build_cpu_s"], pm["pi_episode_cpu_s"], pm["data_episode_cpu_s"])
    c = PB.episode_count_projection(pm["unit"])
    for k in S.K2_PHASES:
        assert proj[k] == max(f[k], c[k]) and rec[k]["v82_formula"] == f[k] and rec[k]["episodes_x_cost"] == c[k]
        assert rec[k]["winner"] == ("v82_formula" if f[k] >= c[k] else "episodes_x_cost")
    assert c["tuning"] == 12 * 0.03 * 3 * 5 * 40 and c["test"] == 4 * 0.03 * 3 * 320
    assert c["adaptation"] == 4 * 0.03 * 3 * 3 * 40
    pm2 = dict(pm, scenario_build_cpu_s=-1000.0)              # formula forced below the episode count -> other wins
    _, rec2 = PB.measured_episode_phases(pm2)
    assert rec2["test"]["winner"] == "episodes_x_cost"
    d = PB.decide(pm, _toy())
    assert d["tries"][-1]["measured_projection_detail"]["tuning"]["used"] == d["projection"]["tuning"] or \
        d["tries"][-1]["k2"]["tuning"]["replaced"]


def test_adaptation_projected_max_jump_reported():
    pm = _pm(per_upd=0.01, ep=0.05)
    d = PB.decide(pm, _toy())
    assert abs(d["adaptation_projected_max_jump_cpu_s"] - (3 * 100 * 0.01 + 0.05)) < 1e-12


def test_k2_stricter_toy_reference_without_ladder():
    pm = _pm(per_upd=0.02, ep=0.03)
    toy = _toy()
    d = PB.decide(pm, toy)
    full = PB.toy_episode_projection(toy, ())
    for t in d["tries"]:
        for k in S.K2_PHASES:
            exp = 0.0 if (k == "in_distribution" and "b" in t["ladder"]) else full[k]
            assert t["toy_projection"][k] == exp


def test_r_uses_given_toy_fits():
    pm = _pm(per_upd=0.005, ep=0.03)
    a = PB.decide(pm, _toy(), toy_fits=336.8)
    b = PB.decide(pm, _toy(), toy_fits=100.0)
    assert abs(a["R"] * 336.8 - b["R"] * 100.0) < 1e-9 and b["k2_toy_fits"] == 100.0


# ---------------- (9) CODE_PASS bound to the current manifest ----------------
def _cp(tmp_path, name, **kw):
    j = dict(reviewer="dahill", verdict="CODE_PASS", plan_sha256=S.V83["plan_sha256"], manifest_sha256="m" * 64)
    j.update(kw)
    (tmp_path / name).write_text(json.dumps(j))


def test_code_pass_binding(tmp_path):
    msha = "m" * 64
    assert GT.code_pass_reasons(msha, str(tmp_path))                         # none -> fail closed
    _cp(tmp_path, "CODE_PASS_v8.3.json", manifest_sha256="a9841b4517c3fb3ef1fe48c68c3ffb5acc60cdff97c55dcbb4721f1549df2ed5")
    assert GT.code_pass_reasons(msha, str(tmp_path))                         # PASS for an older manifest: refused
    _cp(tmp_path, "CODE_PASS_v8.3_r2.json", reviewer="uni")
    assert GT.code_pass_reasons(msha, str(tmp_path))
    _cp(tmp_path, "CODE_PASS_v8.3_r3.json", verdict="CODE_BLOCKED")
    assert GT.code_pass_reasons(msha, str(tmp_path))
    _cp(tmp_path, "CODE_PASS_v8.3_r4.json", plan_sha256="0" * 64)
    assert GT.code_pass_reasons(msha, str(tmp_path))
    assert GT.code_pass_reasons(None, str(tmp_path))
    _cp(tmp_path, "CODE_PASS_v8.3_r5.json", verdict="CODE_PASS_CONDITIONAL")
    assert GT.code_pass_reasons(msha, str(tmp_path)) == []
    _cp(tmp_path, "CODE_PASS_v8.3_r6.json", manifest_sha256=None, code_manifest_sha256="n" * 64)
    assert GT.code_pass_reasons("n" * 64, str(tmp_path)) == []               # v8.2-style key also read


def _hermetic_cp_dir(tmp_path, monkeypatch, copy_real=()):
    """r3 item 3 / dahill req (6): an isolated CODE_PASS dir (copies only; never deletes or edits real files)."""
    import shutil
    d = tmp_path / "audit_dahill_v8.3"
    d.mkdir(exist_ok=True)
    for name in copy_real:
        src = os.path.join(S.CODE_PASS_DIR, name)
        if os.path.exists(src):
            shutil.copy(src, str(d / name))
    monkeypatch.setattr(S, "CODE_PASS_DIR", str(d))
    return d


def test_real_code_pass_a9841b45_does_not_satisfy_current_manifest(tmp_path, monkeypatch):
    old = "a9841b4517c3fb3ef1fe48c68c3ffb5acc60cdff97c55dcbb4721f1549df2ed5"
    d = _hermetic_cp_dir(tmp_path, monkeypatch, copy_real=("CODE_PASS_v8.3.json",))
    if not (d / "CODE_PASS_v8.3.json").exists():                           # real file absent: same content, synthetic
        _cp(d, "CODE_PASS_v8.3.json", manifest_sha256=old, verdict="CODE_PASS_CONDITIONAL")
    assert GT.find_code_pass(old)[0] is not None                            # dahill's file binds a9841b45 ...
    ok, cur = GT.manifest_ok()
    for m in ([cur] if ok else []) + ["c" * 64]:
        assert m != old and GT.find_code_pass(m)[0] is None                 # ... never another manifest


# ---------------- (5) K2 host rule ----------------
def test_is_box_rule():
    box = dict(S.BOX_FINGERPRINT, platform="p")
    assert HE.is_box(box)
    for k, v in (("hostname", "unicorn"), ("cpu_model", "AMD"), ("cpu_count", 16)):
        assert not HE.is_box(dict(box, **{k: v}))


def _toy_attempt(tmp_path, host, n=1, outcome="TOY_OK", within=True, fits=500.0, tamper=False):
    d = tmp_path / "probe"
    d.mkdir(exist_ok=True)
    f = d / ("TOY_PREFLIGHT_attempt%d.json" % n)
    toy = json.loads(json.dumps(_toy()))
    toy["projection_cpu_s"]["fits"] = fits
    toy["preflight_within_cap"] = within
    f.write_text(json.dumps(toy))
    sha = S.sha256_file(str(f))
    GT.append(str(d / GT.TOY_LEDGER), "attempt_start", attempt=n, host=host)
    GT.append(str(d / GT.TOY_LEDGER), "attempt_end", attempt=n, outcome=outcome, file=f.name, sha256=sha, host=host)
    if tamper:
        f.write_text(json.dumps(dict(toy, x=1)))
    return str(d)


def test_k2_reference_box_and_host(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WM_ROOT", S.WM_ROOT)
    ref, r = GT.k2_reference(dict(S.BOX_FINGERPRINT), str(tmp_path))
    assert r == [] and ref["kind"] == "box_v8.2" and ref["toy_fits"] == 336.8
    lap = dict(hostname="unicorn", cpu_model="Laptop CPU", cpu_count=12)
    ref, r = GT.k2_reference(lap, str(tmp_path / "none"))
    assert ref is None and "no valid same-host toy" in r[0]
    # other-host attempt does not count; tampered file does not count; cap-exceeded does not count
    monkeypatch.setattr(S, "WM_ROOT", str(tmp_path))
    d = _toy_attempt(tmp_path, dict(lap, hostname="other"), n=1)
    assert GT.k2_reference(lap, d)[0] is None
    _toy_attempt(tmp_path, lap, n=2, within=False)
    assert GT.k2_reference(lap, d)[0] is None
    _toy_attempt(tmp_path, lap, n=3, tamper=True)
    assert GT.k2_reference(lap, d)[0] is None
    _toy_attempt(tmp_path, lap, n=4, fits=512.5)
    ref, r = GT.k2_reference(lap, d)
    assert r == [] and ref["kind"] == "host_toy" and ref["attempt"] == 4 and ref["toy_fits"] == 512.5
    assert GT.load_k2_reference(ref)["projection_cpu_s"]["fits"] == 512.5


def test_run_probe_refuses_k2_on_non_box_host_without_toy(tmp_path, monkeypatch):
    _hermetic_cp_dir(tmp_path, monkeypatch)                                 # no CODE_PASS at all (hermetic)
    rpb = _run_probe_module()
    monkeypatch.setattr(HE, "host_fingerprint", lambda: dict(hostname="unicorn", cpu_model="Laptop CPU", cpu_count=12))
    ok = lambda: dict(ok=True, reasons=[], load1=0.1)
    r = rpb.main(probe_dir=str(tmp_path / "probe"), config_path=str(tmp_path / "C.json"), start_rule_fn=ok,
                 with_sidecar=False, t0=None)
    assert r["outcome"] == "PROBE_NOT_STARTED" and not r["is_box"] and r["k2_reference"] is None
    assert any("no valid same-host toy" in x for x in r["reasons"])
    assert any("CODE_PASS" in x for x in r["reasons"])                     # hermetic dir: no dahill PASS


def test_sidecar_capable_on_this_host_and_missing_script(tmp_path):
    ok, bad = HE.sidecar_capable()
    if os.path.exists("/proc/stat") and any(l.lower().startswith("cpu mhz") for l in open("/proc/cpuinfo")):
        assert ok and bad == []
    assert HE.sidecar_capable(str(tmp_path / "nope.py")) == (False, ["sidecar script missing"])


# ---------------- restored v8.2 toy preflight ----------------
def _rp():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rpf", os.path.join(S.CODE_DIR, "run_preflight.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_toy_preflight_main_is_v82_logic():
    import inspect
    v82 = os.path.join(S.PLAN_DIR, "code", "run_preflight.py")
    if not os.path.exists(v82):
        pytest.skip("v8.2 code not present")
    import ast
    old_src = open(v82).read()
    old_main = [n for n in ast.parse(old_src).body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    old_lines = old_src.splitlines()[old_main.lineno - 1:old_main.end_lineno]
    new_lines = inspect.getsource(_rp().main).splitlines()
    import difflib
    changed = [l for l in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0)
               if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    assert changed == ["-def main():", "+def main(out=None):",
                       "-    out = RL.PREFLIGHT_FILE                                # v8.2: v8.2/PREFLIGHT_PROJECTION.json",
                       "+    out = out or os.environ.get(\"WM_TOY_PREFLIGHT_OUT\")   # v8.3: path config only (v8.2: RL.PREFLIGHT_FILE)",
                       "+    if not out:", "+        raise SystemExit(\"no output path (argument or WM_TOY_PREFLIGHT_OUT)\")"]
    v82_mod = open(os.path.join(S.PLAN_DIR, "code", "v81", "preflight.py")).read()
    assert v82_mod == open(os.path.join(S.CODE_DIR, "v81", "preflight.py")).read()   # measurement module unchanged


def test_toy_attempts_numbered_append_only(tmp_path, monkeypatch):
    _hermetic_cp_dir(tmp_path, monkeypatch)                                 # no CODE_PASS at all (hermetic)
    rp = _rp()
    monkeypatch.delenv("WM_TOY_PREFLIGHT_OUT", raising=False)
    with pytest.raises(SystemExit):
        rp.main()                                                            # no output path
    bad = lambda: dict(ok=False, reasons=["load1 2.00 >= 1.0"])
    d = str(tmp_path / "probe")
    a = rp.attempt_main(probe_dir=d, start_rule_fn=bad, with_sidecar=False)
    b = rp.attempt_main(probe_dir=d, start_rule_fn=bad, with_sidecar=False)
    assert a["attempt"] == 1 and b["attempt"] == 2 and a["outcome"] == "TOY_NOT_STARTED"
    ev = GT.events(os.path.join(d, GT.TOY_LEDGER))
    assert [e["event"] for e in ev] == ["attempt_start", "attempt_end"] * 2 and ev[0]["host"]["hostname"]
    assert any("C4 start rule" in r for r in ev[1]["reasons"]) and any("CODE_PASS" in r for r in ev[1]["reasons"])
