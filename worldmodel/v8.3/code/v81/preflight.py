"""Toy preflight (PLAN §10 fix 2, CONFIG preflight): seeds 1-4 only, cap 120 CPU-s. Measures per-update
(20 updates; LIN: 1 fit) and per-episode (2 toy episodes x 96 decisions) cost for EVERY arm x structure,
projects every phase against 0.9 x cap and the total against 3240 s, applies the shrink ladder only to
adaptation / in-distribution / tuning. Touches no test, tuning, training or adaptation seed."""
import os
import tempfile
import time

import numpy as np
import torch

from . import closedloop as CL
from . import features as Fe
from . import model_lin
from . import models_torch as MT
from . import plant_iv as P
from . import predictors as PR
from . import seeds as SD
from . import spec as S
from . import stage0
from . import stats as ST

clock = time.process_time
N_TOY_UPDATES = 20
N_TOY_EPISODES = 2
SUPERVISOR_RESERVE_S = 15.0     # Option1 constants.PROD_SUPERVISOR_RESERVE_S (conservative reserve in overhead)


class PreflightCapExceeded(RuntimeError):
    pass


def _timed(fn, *a, **kw):
    t = clock()
    r = fn(*a, **kw)
    return r, clock() - t


def measure(t_start, report=print):
    G = SD.SeedGuard("preflight")
    m = dict(seeds=list(S.PREFLIGHT_SEEDS))

    def guard_cap():
        if clock() - t_start > S.PREFLIGHT_CAP:
            raise PreflightCapExceeded("preflight > %.0f s" % S.PREFLIGHT_CAP)

    # one throwaway torch forward/backward so lazy library initialisation is not charged to the first
    # structure (its CPU still counts toward the 120 s preflight cap; IMPL_NOTES D11)
    t = clock()
    _w = MT.build("GRU", 32, None, 0)
    _w(torch.zeros(16, 48, 6), torch.zeros(16, 12, 3)).sum().backward()
    m["torch_warmup_cpu_s"] = clock() - t
    _, m["stage0_cpu_s"] = _timed(stage0.model_import_check)
    # data: 4 toy data episodes (288 transitions), seeds 1-4, TRAIN composition head (incl. overrides)
    comp = [("N0", None), ("T1a", 72), ("T3b", 120), ("T2a", 168)]
    t = clock()
    obs = [P.data_episode(s, i, comp[i][0], comp[i][1], G, "preflight_toy")[0] for i, s in enumerate(S.PREFLIGHT_SEEDS)]
    m["data_episode_cpu_s"] = (clock() - t) / 4
    train = P.stack_obs(obs)
    ns = Fe.NormStats.from_train(train)
    pairs = Fe.train_pairs(4)
    (ctx, fut, tgt), m["windows_4ep_cpu_s"] = _timed(Fe.window_arrays, train, pairs, ns)
    bidx = np.random.default_rng([1, 2]).integers(0, len(pairs), size=(N_TOY_UPDATES, S.BATCH))
    # IMPL_ERRATUM_v8.2_W1 §W1.1: untimed warm-up of EVERY arm x structure before any timed update/episode
    # (separate instances, init seed 0; measured models use seed 3101). CPU counts toward the 120 s cap.
    t = clock()
    lr_w = lambda tt: MT.lr_cosine(tt, S.N_UPDATES, S.LR0, S.LR_MIN)
    for arm in S.ARMS:
        for st in S.structures(arm):
            if arm == "LIN":
                model_lin.fit(train, ns, order=st)
            else:
                MT.train(MT.build(arm, st, ns.to_dict(), 0), ctx, fut, tgt, bidx[:2], 2, lr_w, log_every=0)
    m["warmup_all_structures_cpu_s"] = clock() - t
    guard_cap()
    # toy closed-loop scenarios (warm-up + 96 decisions); purpose preflight_toy (never test)
    t = clock()
    scens = [P.Scenario(1, 0, "N0", G, "preflight_toy"), P.Scenario(2, 1, "T1a", G, "preflight_toy")]
    m["scenario_build_cpu_s"] = (clock() - t) / len(scens)
    unit = {}
    for arm in S.ARMS:
        for st in S.structures(arm):
            u = dict(arm=arm, structure=st)
            if arm == "LIN":
                (mdl, _), c = _timed(model_lin.fit, train, ns, order=st)
                u["fit_cpu_s_4ep"] = c
                u["fit_cpu_s_scaled_32ep"] = c * 32 / 4
                u["refit_cpu_s_scaled_42ep"] = c * 42 / 4
                u["per_update_cpu_s"] = None
            else:
                mdl = MT.build(arm, st, ns.to_dict(), 3101)
                u["n_params"] = MT.n_params(mdl)
                lr = lambda tt: MT.lr_cosine(tt, S.N_UPDATES, S.LR0, S.LR_MIN)
                (mdl, info), c = _timed(MT.train, mdl, ctx, fut, tgt, bidx, N_TOY_UPDATES, lr, log_every=0)
                u["per_update_cpu_s"] = c / N_TOY_UPDATES
                u["toy_fit_failed"] = info["failed"]
                if info["failed"]:
                    mdl = None
                vctx, vfut, vtgt = ctx[:64], fut[:64], tgt[:64]
                if mdl is not None:
                    _, u["eval_64win_cpu_s"] = _timed(MT.eval_loss, mdl, vctx, vfut, vtgt)
            pred = PR.make(arm, mdl, ns)
            t = clock()
            rows = [CL.run_episode(sc, pred, 1.0) for sc in scens]
            c = clock() - t
            dec = sum(r["n_mpc_decisions"] for r in rows)
            u["toy_episodes"] = len(rows)
            u["toy_decisions"] = dec
            u["toy_fallbacks"] = int(sum(r["fallback"] for r in rows))
            u["episode_cpu_s_measured"] = c / len(rows)
            # conservative: if fallback cut MPC decisions short, scale to 96 decisions per episode
            u["episode_cpu_s"] = u["episode_cpu_s_measured"] * (max(1.0, S.N_CONTROL * len(rows) / max(dec, 1)))
            # 12-step single-plan prediction cost (held-out MSE secondary)
            if pred is not None:
                tj = rows[0]["traj"]
                view = Fe.ObservableView(120, tj["y"], tj["To"], tj["u"], scens[0].d.s_occ, scens[0].d.s_night)
                t = clock()
                for _ in range(5):
                    pred(view, tj["u"][120:132][None, :])
                u["single_pred_cpu_s"] = (clock() - t) / 5
            else:
                u["single_pred_cpu_s"] = 0.0
            unit["%s_%s" % (arm, st)] = u
            report("  %-10s %-4s per_update=%s episode=%.4f s" % (arm, st, u["per_update_cpu_s"], u["episode_cpu_s"]))
            guard_cap()
    m["unit"] = unit
    t = clock()
    for sc in scens:
        CL.run_episode(sc, None, 1.0, mode="pi")
    m["pi_episode_cpu_s"] = (clock() - t) / len(scens)
    t = clock()
    for sc in scens:
        CL.run_episode(sc, None, 1.0, mode="oracle")
    m["oracle_episode_cpu_s"] = (clock() - t) / len(scens)
    # bootstrap/stats at production size on synthetic numbers (no data from any experiment seed)
    t = clock()
    rng = np.random.default_rng(12345)
    with tempfile.TemporaryDirectory() as td:
        mats = [ST.boot_matrix(S.BOOT_SEEDS["primary"], 40), ST.boot_matrix(S.BOOT_SEEDS["adapt"], 10),
                ST.boot_matrix(S.BOOT_SEEDS["secondary"], 40), ST.boot_matrix(S.BOOT_SEEDS["mse"], 40)]
        for i, mm in enumerate(mats):
            p = os.path.join(td, "b%d.npy" % i)
            np.save(p, mm)
            S.sha256_file(p)
        cl = np.repeat(np.arange(40), 8)
        E1, E2, D1, D2 = (rng.uniform(10, 50, 320) for _ in range(4))
        ST.primary_endpoints(E1, E2, D1, D2, cl, mats[0], 0, 0)
        for _ in range(2):                      # secondary window + in-distribution
            ST.boot_S_dD(E1, E2, D1, D2, cl, mats[2])
        cla = np.tile(np.arange(10), 4)
        for _ in range(3):                      # adaptation K=1,3,10
            ST.boot_S_dD(E1[:40], E2[:40], D1[:40], D2[:40], cla, mats[1])
        ST.mse_contrast_ci({a: rng.uniform(0, 3, 40) for a in S.ARMS}, mats[3])     # v8.2: 6 MSE contrasts
    m["bootstrap_stats_cpu_s"] = clock() - t
    # audit: stage0 + plan hash verify + code hashing + HELDOUT scan scaled to the production record count
    t = clock()
    stage0.model_import_check()
    S.verify_plan_hashes()
    for root, _, files in os.walk(S.CODE_DIR):
        for f in files:
            if f.endswith(".py"):
                S.sha256_file(os.path.join(root, f))
    m["audit_static_cpu_s"] = clock() - t
    recs = [dict(arm="GRU", structure=64, lam=1.0, rep=0, seed=S.TUNING_SEEDS[0] + i, iv_id="T1a", E=1.0, D=0.1, J=2.0,
                 safety=False, fallback=False) for i in range(500)]
    _, c = _timed(stage0.heldout_scan, recs)
    m["heldout_scan_per_record_cpu_s"] = c / 500
    m["n_records_pre_test_production"] = 3 * 40 + 40 + 4 * len(S.configs("WM")) * 3 * 40 + 40   # v8.2: 15 configs
    guard_cap()
    return m


def project(m, setup_cpu_s, ladder=()):
    """Phase projections from measured unit costs (fits/tuning: per-structure; test/ID/adaptation: per-arm max)."""
    u = m["unit"]
    neural = ("WM", "GRU", "RC_only")
    tun_eps = 30 if "c" in ladder else 40
    adapt_eval = 5 if "a" in ladder else 10
    ep = lambda a, s: u["%s_%s" % (a, s)]["episode_cpu_s"]
    ep_max = {a: max(ep(a, s) for s in S.structures(a)) for a in S.ARMS}
    upd_max = {a: max(u["%s_%s" % (a, s)]["per_update_cpu_s"] for s in S.structures(a)) for a in neural}
    pred_max = {a: max(u["%s_%s" % (a, s)]["single_pred_cpu_s"] for s in S.structures(a)) for a in S.ARMS}
    sc = m["scenario_build_cpu_s"]
    win_rep = m["windows_4ep_cpu_s"] * (32 + 8) / 4
    proj = {}
    proj["overhead"] = setup_cpu_s + SUPERVISOR_RESERVE_S
    proj["data"] = m["stage0_cpu_s"] + 3 * 40 * m["data_episode_cpu_s"]
    fits = 3 * win_rep
    for a in neural:
        for s in S.structures(a):
            uu = u["%s_%s" % (a, s)]
            fits += 3 * (uu["per_update_cpu_s"] * S.N_UPDATES + uu.get("eval_64win_cpu_s", 0.0) * (8 * 229 / 64))
    for s in S.LIN_ORDERS:
        fits += 3 * u["LIN_%s" % s]["fit_cpu_s_scaled_32ep"]
    proj["fits"] = fits
    # v8.2: 3 replicates x len(LAMBDAS) (=5) per structure (v8.1 hard-coded 3 x 3)
    proj["tuning"] = tun_eps * sc + sum(ep(a, s) * 3 * len(S.LAMBDAS) * tun_eps for a in S.ARMS for s in S.structures(a))
    proj["test"] = 320 * sc + sum(ep_max[a] * 3 * 320 for a in S.ARMS)
    proj["in_distribution"] = 0.0 if "b" in ladder else 40 * sc + sum(ep_max[a] * 3 * 40 for a in S.ARMS)
    proj["pi_oracle"] = m["pi_episode_cpu_s"] * (360 if "b" not in ladder else 320) + \
        m["oracle_episode_cpu_s"] * (360 if "b" not in ladder else 320)
    n_eval = 4 * adapt_eval
    proj["adaptation"] = (40 * m["data_episode_cpu_s"] + n_eval * (sc + m["pi_episode_cpu_s"])
                          + sum(upd_max[a] * S.ADAPT_UPDATES * 36 for a in neural)
                          + max(u["LIN_%s" % s]["refit_cpu_s_scaled_42ep"] for s in S.LIN_ORDERS) * 36
                          + sum(ep_max[a] * 3 * 3 * n_eval for a in S.ARMS))
    proj["bootstrap_stats"] = (m["bootstrap_stats_cpu_s"] + 320 * (sc + m["pi_episode_cpu_s"])
                               + sum(pred_max[a] * 3 * 320 for a in S.ARMS))
    proj["audit"] = m["audit_static_cpu_s"] + m["heldout_scan_per_record_cpu_s"] * m["n_records_pre_test_production"]
    total = sum(proj.values())
    gates = {k: proj[k] <= S.GATE_FRAC * S.CAPS[k] for k in proj}
    return proj, total, gates


def plausibility(m):
    """IMPL_ERRATUM_v8.2_W1 §W1.3(4): WM_32 per-update cost must be <= WM_64 per-update cost."""
    a = m["unit"]["WM_32"]["per_update_cpu_s"]
    b = m["unit"]["WM_64"]["per_update_cpu_s"]
    return dict(rule="WM_32 per_update_cpu_s <= WM_64 per_update_cpu_s", WM_32=a, WM_64=b,
                ok=bool(a is not None and b is not None and a <= b))


def decide(m, setup_cpu_s):
    """W1 rules: ladder only step (a) and only if adaptation fails; any other failing phase, the total gate or the
    plausibility gate -> all_gates_pass False (NOT_LAUNCHED)."""
    applied = []
    pl = plausibility(m)
    while True:
        proj, total, gates = project(m, setup_cpu_s, tuple(applied))
        failing = [k for k, ok in gates.items() if not ok]
        total_ok = total <= S.TOTAL_GATE
        if not failing and total_ok:
            return dict(projection=proj, total=total, gates=gates, total_gate=total_ok, ladder_applied=applied,
                        plausibility=pl, all_gates_pass=bool(pl["ok"]), failing_non_ladder=[])
        non_ladder = [k for k in failing if k not in S.LADDER_PHASES]
        can_a = ("adaptation" in failing and "a" in S.W1_ALLOWED_LADDER and "a" not in applied
                 and not non_ladder)
        if not can_a:
            return dict(projection=proj, total=total, gates=gates, total_gate=total_ok, ladder_applied=applied,
                        plausibility=pl, all_gates_pass=False, failing_non_ladder=non_ladder)
        applied.append("a")
