"""v8.2-specific tests: energy-under-comfort-bound selection (PLAN §8), comparator rule, fresh seeds disjoint from
v8.1 (incl. SeedGuard), bundle/config numbers, MSE contrast CI (PLAN §9)."""
import json
import os

import numpy as np
import pytest

from v81 import pipeline as PP
from v81 import seeds as SD
from v81 import spec as S
from v81 import stats as ST

V81_DIR = os.path.normpath(os.path.join(S.PLAN_DIR, "..", "v8.1"))
DMAX = 0.3269812989590207


def _t(E, D, J=0.0):
    return dict(mean_E=E, mean_D=D, mean_J=J)


# ---------------- bundle / CONFIG numbers ----------------
def test_bundle_and_config_numbers():
    S.verify_plan_hashes()
    assert S.HASHES_FILE == "HASHES_v8.2_E1.sha256"
    assert S.HASHES_FILE_SHA256 == "304bc28de338c92ee72817e7ed4ad498c367165e7f3ca47eed38e1732c636ff3"
    assert S.BOUND_FILES["ERRATUM_v8.2_E1.md"] == "67b4ecdb9bd5a31a91ef2e603fe15350b7048fb98a05f8e3168e99f152938c47"
    assert S.sha256_file(os.path.join(S.PLAN_DIR, S.HASHES_FILE)) == S.HASHES_FILE_SHA256
    assert S.LAMBDAS == (0.5, 0.625, 0.75, 0.875, 1.0)                      # ERRATUM_v8.2_E1 §E1.1
    assert 1.0 in S.LAMBDAS                                                  # D_max reference point is on the grid
    assert len(S.configs("WM")) == 15 and all(len(S.configs(a)) == 15 for a in S.ARMS)
    assert S.COMFORT_BOUND == dict(form="absolute", D_max=DMAX)
    assert S.TUNING_SEEDS == tuple(range(7300, 7320)) and S.TEST_SEEDS == tuple(range(9500, 9540))
    assert S.ADAPT_SEEDS == tuple(range(4100, 4110))
    assert S.BOOT_SEEDS == dict(primary=9921, adapt=9922, secondary=9923, mse=9924)
    assert S.CAPS["tuning"] == 1100 and S.CAPS["test"] == 800 and S.GLOBAL_CAP == 3600
    assert S.PLAN_PASS_SHA256 == "8ad13f896e2677498048f864ee3abed6e6534c4dcb75c889b59643de34bd49fe"
    assert S.sha256_file(S.PLAN_PASS_FILE) == S.PLAN_PASS_SHA256
    with open(S.PLAN_PASS_FILE) as f:
        pp = json.load(f)
    assert pp["verdict"] == "PLAN_PASS" and pp["reviewer"] == "dahill"
    assert pp["hashes_v8.2_sha256"] == S.HASHES_FILE_SHA256


def test_grid_override_record():
    g = S.GRID_OVERRIDE
    assert g["parameter"] == "planner.lambda_grid"
    assert g["config_value"] == [0.5, 1.0, 2.0, 4.0, 8.0] == S.CFG["planner"]["lambda_grid"]
    assert g["effective_value"] == [0.5, 0.625, 0.75, 0.875, 1.0] == list(S.LAMBDAS)
    assert g["source_sha256"] == S.BOUND_FILES["ERRATUM_v8.2_E1.md"] and "E1.1" in g["source"]
    json.dumps(g)                                                              # machine-readable


def _release(tmp_path, **kw):
    from v81 import release as RL
    rel = dict(status="DRAFT", plan_bundle_v8_2={S.HASHES_FILE: S.HASHES_FILE_SHA256},     # v8.3 release field names
               plan_pass_v8_2={"sha256": S.PLAN_PASS_SHA256}, grid_override=S.GRID_OVERRIDE,
               implementer="TMbot · Research (non-author)", code_review_pass=None,
               code_hashes={"manifest_sha256": "x"})
    rel.update(kw)
    p = tmp_path / "RELEASE_v8.3.json"
    p.write_text(json.dumps(rel))
    return RL.check_start(str(p))[1]


def test_check_start_plan_pass_and_grid(tmp_path, monkeypatch):
    r = _release(tmp_path)
    assert not any("plan PASS" in x for x in r) and not any("grid_override" in x for x in r)
    assert any("plan PASS" in x for x in _release(tmp_path, plan_pass_v8_2={"sha256": "0" * 64}))
    g = dict(S.GRID_OVERRIDE, effective_value=[0.5, 1.0, 2.0, 4.0, 8.0])
    assert any("grid_override" in x for x in _release(tmp_path, grid_override=g))
    # a PASS file with the wrong reviewer / bundle is rejected even if its own hash is "right"
    for bad in (dict(verdict="PLAN_PASS", reviewer="uni", **{"hashes_v8.2_sha256": S.HASHES_FILE_SHA256}),
                dict(verdict="PLAN_PASS", reviewer="dahill", **{"hashes_v8.2_sha256": "1" * 64})):
        f = tmp_path / "pp.json"
        f.write_text(json.dumps(bad))
        monkeypatch.setattr(S, "PLAN_PASS_FILE", str(f))
        monkeypatch.setattr(S, "PLAN_PASS_SHA256", S.sha256_file(str(f)))
        assert any("plan PASS" in x for x in _release(tmp_path, plan_pass_v8_2={"sha256": S.sha256_file(str(f))}))


def test_unchanged_vs_v81_config():
    with open(os.path.join(V81_DIR, "CONFIG_v8.1_r3.json")) as f:
        old = json.load(f)
    for k in ("time", "inputs_all_arms", "context_steps", "horizon_steps", "PI", "training", "initial_states"):
        assert S.CFG[k] == old[k], k
    for k in ("train_data", "val_data", "model_init"):
        assert S.CFG["seeds"][k] == old["seeds"][k], k
    assert S.CFG["seeds"]["preflight"] == old["preflight"]["seeds"] == S.CFG["preflight"]["seeds"]
    assert S.CFG["preflight"]["cap_cpu_s"] == old["preflight"]["cap_cpu_s"]
    for a in S.ARMS:
        for k, v in old["arms"][a].items():
            if k != "desc":
                assert S.CFG["arms"][a][k] == v, (a, k)


# ---------------- seeds ----------------
def test_seeds_disjoint_from_v81():
    with open(os.path.join(V81_DIR, "CONFIG_v8.1_r3.json")) as f:
        old = json.load(f)["seeds"]
    old_ep = set(old["tuning"]) | set(old["test"]) | set(old["adaptation"])
    new_ep = set(S.TUNING_SEEDS) | set(S.TEST_SEEDS) | set(S.ADAPT_SEEDS)
    assert not (old_ep & new_ep)
    assert not (set(old["bootstrap"].values()) & set(S.BOOT_SEEDS.values()))
    assert old_ep <= S.EXCLUDED_SEEDS and set(old["bootstrap"].values()) <= S.EXCLUDED_SEEDS
    # new lists pairwise disjoint and disjoint from train/val/init/preflight
    tv = set(sum((list(v) for v in S.TRAIN_SEEDS.values()), [])) | set(sum((list(v) for v in S.VAL_SEEDS.values()), []))
    lists = [set(S.TUNING_SEEDS), set(S.TEST_SEEDS), set(S.ADAPT_SEEDS), tv, set(S.INIT_SEEDS.values()),
             set(S.PREFLIGHT_SEEDS), set(S.BOOT_SEEDS.values())]
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            assert not (lists[i] & lists[j]), (i, j)


@pytest.mark.parametrize("seed,purpose", [(7000, "tuning"), (7019, "tuning"), (9100, "test_heldout"),
                                          (9139, "test_id"), (8000, "adaptation_data"), (8009, "adaptation_data"),
                                          (9100, "adapt_eval"), (5000, "train")])
def test_seedguard_rejects_v81_seeds(seed, purpose):
    G = SD.SeedGuard("production")
    G.selection_sealed = G.primary_sealed = True
    with pytest.raises(SD.SeedGuardError):
        G.check(seed, purpose)


def test_seedguard_accepts_v82_seeds_in_order():
    G = SD.SeedGuard("production")
    for s in S.TUNING_SEEDS:
        G.check(s, "tuning")
    with pytest.raises(SD.SeedGuardError):
        G.check(S.TEST_SEEDS[0], "test_heldout")
    G.selection_sealed = True
    for s in S.TEST_SEEDS:
        G.check(s, "test_heldout")
    with pytest.raises(SD.SeedGuardError):
        G.check(S.ADAPT_SEEDS[0], "adaptation_data")
    G.primary_sealed = True
    for s in S.ADAPT_SEEDS:
        G.check(s, "adaptation_data")
    for s in range(9500, 9510):
        G.check(s, "adapt_eval")


def test_adaptation_composition_uses_new_seeds():
    assert [c[0] for c in SD.adaptation_composition("H1", 10)] == list(range(4100, 4110))


def test_ladder_c_tuning_seeds():
    d = PP.Design().apply_ladder(["c"])
    assert d.tuning_seeds == list(range(7300, 7315))


# ---------------- within-arm selection ----------------
def test_feasible_min_energy():
    tab = {(32, 0.5): _t(10.0, 0.10), (32, 0.875): _t(6.0, 0.50), (64, 0.625): _t(8.0, 0.30), (96, 0.75): _t(7.0, 0.32)}
    assert ST.select_within_arm("GRU", tab, DMAX) == ((96, 0.75), True)


def test_boundary_is_inclusive():
    tab = {(32, 1.0): _t(9.0, 0.1), (64, 0.75): _t(5.0, DMAX)}
    assert ST.select_within_arm("GRU", tab, DMAX) == ((64, 0.75), True)
    tab[(64, 0.75)] = _t(5.0, np.nextafter(DMAX, 1.0))
    assert ST.select_within_arm("GRU", tab, DMAX) == ((32, 1.0), True)


def test_feasible_ties_lower_D_then_structure_then_lambda():
    tab = {(64, 1.0): _t(5.0, 0.2), (32, 0.625): _t(5.0, 0.3)}
    assert ST.select_within_arm("GRU", tab, DMAX)[0] == (64, 1.0)                 # lower D
    tab = {(64, 1.0): _t(5.0, 0.2), (32, 0.625): _t(5.0, 0.2)}
    assert ST.select_within_arm("GRU", tab, DMAX)[0] == (32, 0.625)                 # smaller structure
    tab = {(32, 0.75): _t(5.0, 0.2), (32, 0.5): _t(5.0, 0.2), (32, 1.0): _t(5.0, 0.2)}
    assert ST.select_within_arm("GRU", tab, DMAX)[0] == (32, 0.5)                 # smaller lambda (not 1.0 first)


def test_rc_and_lin_structure_order():
    tab = {(3, 1.0): _t(5.0, 0.2), (1, 1.0): _t(5.0, 0.2), (2, 1.0): _t(5.0, 0.2)}
    assert ST.select_within_arm("RC_only", tab, DMAX)[0] == (1, 1.0)
    tab = {(8, 1.0): _t(5.0, 0.2), (5, 1.0): _t(5.0, 0.2)}
    assert ST.select_within_arm("LIN", tab, DMAX)[0] == (5, 1.0)


def test_infeasible_fallback_lowest_D():
    tab = {(32, 0.5): _t(10.0, 0.40), (32, 0.875): _t(2.0, 0.90), (64, 1.0): _t(9.0, 0.35), (96, 0.5): _t(8.0, 0.35)}
    sel, feas = ST.select_within_arm("GRU", tab, DMAX)
    assert not feas and sel == (96, 0.5)                                         # tie on D -> lower E
    tab[(32, 1.0)] = _t(8.0, 0.35)
    assert ST.select_within_arm("GRU", tab, DMAX)[0] == (32, 1.0)                # then smaller structure
    tab[(32, 0.5)] = _t(8.0, 0.35)
    assert ST.select_within_arm("GRU", tab, DMAX)[0] == (32, 0.5)                # then smaller lambda


# ---------------- comparator ----------------
def _s(E, D, f):
    return dict(mean_E=E, mean_D=D, feasible=f)


def test_comparator_feasible_lowest_E():
    sel = {"GRU": _s(5.0, 0.3, True), "LIN": _s(4.0, 0.1, True), "RC_only": _s(6.0, 0.0, True)}
    assert ST.select_comparator(sel) == "LIN"


def test_comparator_feasible_before_infeasible():
    sel = {"GRU": _s(1.0, 0.9, False), "LIN": _s(9.0, 0.3, True), "RC_only": _s(0.5, 0.4, False)}
    assert ST.select_comparator(sel) == "LIN"


def test_comparator_none_feasible_lowest_D():
    sel = {"GRU": _s(1.0, 0.9, False), "LIN": _s(9.0, 0.5, False), "RC_only": _s(0.5, 0.6, False)}
    assert ST.select_comparator(sel) == "LIN"


def test_comparator_ties():
    sel = {"GRU": _s(5.0, 0.2, True), "LIN": _s(5.0, 0.1, True), "RC_only": _s(5.0, 0.2, True)}
    assert ST.select_comparator(sel) == "LIN"                                   # lower D
    sel["LIN"] = _s(5.0, 0.2, True)
    assert ST.select_comparator(sel) == "RC_only"                               # order RC_only, LIN, GRU
    sel = {"GRU": _s(4.0, 0.5, False), "LIN": _s(3.0, 0.5, False), "RC_only": _s(3.0, 0.5, False)}
    assert ST.select_comparator(sel) == "RC_only"
    sel["RC_only"] = _s(3.5, 0.5, False)
    assert ST.select_comparator(sel) == "LIN"                                   # infeasible tie on D -> lower E


def test_comparator_ignores_wm():
    sel = {"GRU": _s(5.0, 0.2, True), "LIN": _s(5.0, 0.2, True), "RC_only": _s(5.0, 0.2, True),
           "WM": _s(0.0, 0.0, True)}
    assert ST.select_comparator(sel) in ("GRU", "LIN", "RC_only")


# ---------------- full-table selection + recompute gate ----------------
def _full_table(seed=0):
    rng = np.random.default_rng(seed)
    t = {}
    for a in S.ARMS:
        for st in S.structures(a):
            for lam in S.LAMBDAS:
                t[(a, st, lam)] = dict(mean_E=float(rng.uniform(5, 15) / lam ** 0.3), mean_D=float(rng.uniform(0, 0.6) * lam ** 0.5),
                                       mean_J=float(rng.uniform(10, 20)), n_rows=120)
    return t


def test_select_from_table_and_recompute(tmp_path):
    for seed in range(20):
        t = _full_table(seed)
        sel = PP.select_from_table(t)
        assert sel["comfort_bound"]["value"] == DMAX
        for a in S.ARMS:
            x = sel["selected"][a]
            feas = [k for k in t if k[0] == a and t[k]["mean_D"] <= DMAX]
            if feas:
                best = min(t[k]["mean_E"] for k in feas)
                assert x["feasible"] and x["mean_E"] == best and x["mean_D"] <= DMAX
            else:
                assert not x["feasible"] and x["mean_D"] == min(t[k]["mean_D"] for k in t if k[0] == a)
        p = tmp_path / ("sel%d.json" % seed)
        p.write_text(json.dumps(sel))
        assert PP.recompute_selection_ok(str(p))
        bad = json.loads(p.read_text())
        others = [c for c in ("GRU", "LIN", "RC_only") if c != bad["comparator"]]
        bad["comparator"] = others[0]
        p.write_text(json.dumps(bad))
        assert not PP.recompute_selection_ok(str(p))


def test_table_matches_rows():
    rows = [dict(arm="GRU", structure=32, lam=1.0, E=float(e), D=0.1 * e, J=2.0 * e) for e in range(6)]
    tt = [dict(arm="GRU", structure=32, lam=1.0, n_rows=6, mean_E=float(np.mean([r["E"] for r in rows])),
               mean_D=float(np.mean([r["D"] for r in rows])), mean_J=float(np.mean([r["J"] for r in rows])))]
    assert PP.table_matches_rows(tt, rows)
    tt[0]["mean_E"] += 1e-9
    assert not PP.table_matches_rows(tt, rows)


# ---------------- comfort bound: absolute D_max only ----------------
def test_comfort_bound_absolute_only():
    assert S.D_MAX_EXPECTED == DMAX == S.CFG["selection"]["D_max_K_h_per_day"]
    v, rec = ST.comfort_bound(_full_table(1))
    assert v == DMAX and rec == dict(form="absolute", D_max=DMAX, value=DMAX)
    assert S.parse_comfort_bound({"D_max_K_h_per_day": 0.3}) == dict(form="absolute", D_max=0.3)
    for bad in ({}, {"D_margin_K_h_per_day": 0.25}, {"D_max_K_h_per_day": 0.3, "D_margin": 0.25}, None):
        with pytest.raises(S.SpecError):
            S.parse_comfort_bound(bad)
    for bad in (dict(form="relative", D_max=DMAX), dict(form="absolute", D_max=DMAX, margin=0.25), dict(form="absolute")):
        with pytest.raises(S.SpecError):
            ST.comfort_bound({}, bad)


# ---------------- E1.6 flag-only fallback ----------------
def _sel(wm, gru, lin, rc):
    return {"WM": dict(feasible=wm), "GRU": dict(feasible=gru), "LIN": dict(feasible=lin), "RC_only": dict(feasible=rc)}


def test_fallback_flags():
    f = ST.fallback_flags(_sel(False, True, False, True), "GRU")
    assert f == dict(WM_fallback=True, comparator_fallback=False, arms_fallback=["WM", "LIN"])
    f = ST.fallback_flags(_sel(True, False, False, False), "RC_only")
    assert f == dict(WM_fallback=False, comparator_fallback=True, arms_fallback=["GRU", "LIN", "RC_only"])


def test_outcome_precedence_v81_restored():
    adv = dict(P1=True, P2=True, P3=True, P4=True)
    assert ST.outcome(adv, True) == "ADVANCE"
    assert ST.outcome(dict(adv, P3=False), True) == "NO_ADVANCE"
    assert ST.outcome(adv, False) == "INVALID"
    assert ST.outcome(adv, False, launched=False) == "NOT_LAUNCHED"
    import inspect
    assert list(inspect.signature(ST.outcome).parameters) == ["primary", "validity_ok", "launched", "start_ok"]


def _table_with(wm_feasible, comp_feasible):
    t = {}
    for a in S.ARMS:
        ok = wm_feasible if a == "WM" else (comp_feasible if a == "RC_only" else False)
        for st in S.structures(a):
            for lam in S.LAMBDAS:
                t[(a, st, lam)] = dict(mean_E=10.0 / lam, mean_D=0.1 if (ok and lam == 1.0) else 0.5 + lam, mean_J=1.0)
    return t


@pytest.mark.parametrize("wm,comp", [(True, True), (False, True), (True, False), (False, False)])
def test_fallback_flags_sealed_and_recomputed(tmp_path, wm, comp):
    sel = PP.select_from_table(_table_with(wm, comp))
    f = sel["fallback_flags"]
    assert f["WM_fallback"] == (not wm) and f["comparator_fallback"] == (not sel["selected"][sel["comparator"]]["feasible"])
    if comp:
        assert sel["comparator"] == "RC_only" and not f["comparator_fallback"]  # feasible arm ranks first
    else:
        assert f["comparator_fallback"]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(sel))
    assert PP.recompute_selection_ok(str(p))
    bad = dict(sel, fallback_flags=dict(f, WM_fallback=not f["WM_fallback"]))
    p.write_text(json.dumps(bad))
    assert not PP.recompute_selection_ok(str(p))                              # tampered flag is detected


# ---------------- MSE contrast CI ----------------
def test_mse_contrast_ci_matches_loop():
    rng = np.random.default_rng(5)
    per = {a: rng.uniform(1, 3, 40) for a in S.ARMS}
    per["WM"] = per["GRU"] - 0.5
    idx = ST.boot_matrix(S.BOOT_SEEDS["mse"], 40)
    assert idx.shape == (10000, 40)
    out = ST.mse_contrast_ci(per, idx)
    assert set(out) == {"WM-GRU", "WM-RC_only", "WM-LIN", "GRU-RC_only", "GRU-LIN", "RC_only-LIN"}
    assert np.isclose(out["WM-GRU"]["diff"], -0.5) and np.allclose(out["WM-GRU"]["ci"], [-0.5, -0.5])
    d = per["RC_only"] - per["LIN"]
    vals = np.array([d[idx[b]].mean() for b in range(10000)])
    assert out["RC_only-LIN"]["ci"] == list(ST.ci(vals))
    assert np.isclose(out["RC_only-LIN"]["diff"], d.mean())


def test_mse_contrast_ci_nan_and_missing():
    per = {a: np.ones(40) for a in S.ARMS}
    per["LIN"] = np.ones(40)
    per["LIN"][3] = np.nan
    per["GRU"] = None
    out = ST.mse_contrast_ci(per, ST.boot_matrix(S.BOOT_SEEDS["mse"], 40))
    assert out["WM-LIN"]["ci"] is None and out["WM-LIN"]["n_nan_seeds"] == 1
    assert out["WM-GRU"]["diff"] is None and out["WM-GRU"]["ci"] is None
    assert out["WM-RC_only"]["ci"] == [0.0, 0.0]
