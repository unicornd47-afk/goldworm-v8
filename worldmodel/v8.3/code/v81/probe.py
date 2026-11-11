"""v8.3 full-size probe (PLAN_v8.3 r3 C2, C3, K1, K2; dahill PLAN_REVIEW r3 A1, A2).

Measurement (measure_probe), all on replicate r1's train/val data seeds only (SeedGuard mode 'probe'):
  data      32 train + 8 val data episodes of replicate r1 (needed at full size for the fits)
  fits (a)  WM h96: one complete fit at full size (2000 updates); the other 11 arm x structure fits: the first
            10 % of their updates (200; same batch indices and cosine schedule as production), each timed incl.
            build + val eval (conservative, dahill N5). LIN (closed form, no updates): one full fit each.
  (b)       linearity: 10 x CPU of the first 10 % of the WM h96 training loop / CPU of its full training loop.
  v8.3 r3 (ERRATUM E1.1 + dahill reqs 1-2): a 200-update throwaway warm-up (same architecture) before each timed
            neural fit segment; WM h96 is fitted TWICE back-to-back after one warm-up (E2.2: linearity on fit 2;
            W x n_coldstarts, per fit or per process, enters the fits projection, E2.1); per-update CPU in blocks
            of 50 updates for every measured fit and every warm-up.
  (c)       per arm x structure: 2 full-size train-entry closed-loop episodes (96 decisions, 64 candidates,
            lambda = 1.0) on train seeds; plus 2 PI episodes on the same scenarios (adaptation term).
  hard cap  450 CPU-s (process CPU since t0), checked every 10 updates and after every episode.
Decision (decide, pure): A1 -> cap hit or ratio outside [0.87, 1.15]: PROBE_DIAGNOSIS, no projection.
  Otherwise projections from measured per-unit costs, K2, caps = ceil(1.25 x projection), ONE budget
  (sum of caps incl. the fixed 260 <= 3600), v8.2 ladder a, b, c in order, else NOT_LAUNCHED."""
import math
import time

import numpy as np

from . import closedloop as CL
from . import features as Fe
from . import model_lin
from . import models_torch as MT
from . import plant_iv as P
from . import predictors as PR
from . import seeds as SD
from . import spec as S
from . import warmup as WU

NEURAL = ("WM", "GRU", "RC_only")


class ProbeCapExceeded(RuntimeError):
    pass


class ProbeBudget:
    """Hard probe cap on process CPU-s since t0; records every check and the largest jump between checks."""

    def __init__(self, cap=S.PROBE_CAP_CPU_S, clock=time.process_time, t0=None):
        self.cap, self.clock = float(cap), clock
        self.t0 = clock() if t0 is None else float(t0)
        self.last = self.clock()
        self.n_checks, self.max_jump = 0, 0.0

    def used(self):
        return self.clock() - self.t0

    def check(self):
        now = self.clock()
        self.max_jump = max(self.max_jump, now - self.last)
        self.last = now
        self.n_checks += 1
        if now - self.t0 > self.cap:
            raise ProbeCapExceeded("probe %.2f > cap %.1f CPU-s" % (now - self.t0, self.cap))


def _units():
    return [(a, s) for a in S.ARMS for s in S.structures(a)]


def measure_probe(budget, event=lambda *a, **k: None, n_updates=S.N_UPDATES, fraction=S.PROBE_FRACTION,
                  structures=None, full_fit=S.PROBE_FULL_FIT, n_train=None, n_val=None, out=None,
                  warmup_updates=S.WARMUP_UPDATES, full_repeats=S.PROBE_FULL_FIT_REPEATS, n_process_starts=1):
    """Fills and returns dict `out` (partial on ProbeCapExceeded: the caller keeps it for the diagnosis)."""
    out = {} if out is None else out
    clock = budget.clock
    structures = structures or {a: tuple(S.structures(a)) for a in S.ARMS}
    G = SD.SeedGuard("probe")
    r = 0
    tr_seeds = list(S.TRAIN_SEEDS[r])[:n_train] if n_train else list(S.TRAIN_SEEDS[r])
    va_seeds = list(S.VAL_SEEDS[r])[:n_val] if n_val else list(S.VAL_SEEDS[r])
    out.update(replicate=r + 1, n_updates=n_updates, fraction=fraction, full_fit=list(full_fit),
               train_seeds=tr_seeds, val_seeds=va_seeds, unit={}, linearity=None)
    event("probe_step", step="data_start", probe_cpu_s=budget.used())
    t = clock()
    comp, vcomp = SD.train_composition(len(tr_seeds)), SD.val_composition(len(va_seeds))
    tr = []
    for i, s in enumerate(tr_seeds):
        tr.append(P.data_episode(s, i, comp[i][0], comp[i][1], G, "train")[0])
        budget.check()
    va = []
    for i, s in enumerate(va_seeds):
        va.append(P.data_episode(s, i, vcomp[i][0], vcomp[i][1], G, "val")[0])
        budget.check()
    out["data_episode_cpu_s"] = (clock() - t) / (len(tr_seeds) + len(va_seeds))
    train, val = P.stack_obs(tr), P.stack_obs(va)
    ns = Fe.NormStats.from_train(train)
    t = clock()
    pairs = Fe.train_pairs(train["u"].shape[0])
    ctx, fut, tgt = Fe.window_arrays(train, pairs, ns)
    vctx, vfut, vtgt = Fe.window_arrays(val, Fe.train_pairs(val["u"].shape[0]), ns)
    out["windows_per_replicate_cpu_s"] = clock() - t
    bidx = np.random.default_rng([S.INIT_SEEDS[r], 2]).integers(0, len(pairs), size=(n_updates, S.BATCH))
    n10 = int(round(fraction * n_updates))
    lr = lambda tt: MT.lr_cosine(tt, n_updates, S.LR0, S.LR_MIN)
    models = {}

    def do_warmup(arm, st):
        """ERRATUM E1.1: throwaway warm-up (same architecture as the following segment); counts toward the 450 s cap
        (process CPU since t0) via budget.check; its CPU-s enter the fits projection as W (E2.1)."""
        w = WU.warmup(arm, st, ns.to_dict(), ctx, fut, tgt, n_updates=warmup_updates, clock=clock, check=budget.check)
        out.setdefault("warmups", []).append(w)
        event("warmup", probe_cpu_s=budget.used(), **w)
        return w

    def neural_fit(arm, st, n_run, mark=None):
        u = dict(arm=arm, structure=st)
        blocks, last = [], [None]
        t0 = clock()
        mdl = MT.build(arm, st, ns.to_dict(), S.INIT_SEEDS[r])
        u["build_cpu_s"] = clock() - t0
        marks = {}

        def cb(n):
            if mark is not None and n == mark:
                marks["first_mark_train_cpu_s"] = clock() - t1
            if n % S.PROBE_BLOCK_UPDATES == 0:            # dahill req (1): CPU-s per update, blocks of 50 updates
                now = clock()
                blocks.append((now - (last[0] if last[0] is not None else t1)) / S.PROBE_BLOCK_UPDATES)
                last[0] = now
            if n % S.LEDGER_UPDATE_EVERY == 0:
                event("probe_fit_progress", arm=arm, structure=st, updates=n, probe_cpu_s=budget.used())
        t1 = clock()
        mdl, info = MT.train(mdl, ctx, fut, tgt, bidx[:n_run], n_run, lr, budget_check=budget.check,
                             check_every=S.FIT_CHECK_EVERY, log_every=0, progress_cb=cb)
        u["train_cpu_s"] = clock() - t1
        u["updates_run"] = info["updates_done"]
        u["failed"] = bool(info["failed"])
        u["per_update_cpu_s"] = u["train_cpu_s"] / max(1, info["updates_done"])
        t2 = clock()
        if not info["failed"]:
            MT.eval_loss(mdl, vctx, vfut, vtgt)
        u["eval_cpu_s"] = clock() - t2
        u["fit_cpu_s"] = clock() - t0
        u["block_updates"] = S.PROBE_BLOCK_UPDATES
        u["block_per_update_cpu_s"] = blocks
        u.update(marks)
        event("probe_fit_end", arm=arm, structure=st, updates=info["updates_done"], fit_cpu_s=u["fit_cpu_s"],
              block_updates=S.PROBE_BLOCK_UPDATES, block_per_update_cpu_s=blocks, probe_cpu_s=budget.used())
        return (None if info["failed"] else mdl), u

    # (a) E1.1 warm-up, then dahill req (2): TWO full-size WM h96 fits back-to-back in this process, each with its
    #     own first-10 % window; (b) linearity on both (post-warm-up), cold-start residual W per fit
    fa, fs = full_fit
    out["erratum"] = S.ERRATUM
    do_warmup(fa, fs)
    fulls = []
    for k in range(full_repeats):
        mdl_k, u = neural_fit(fa, fs, n_updates, mark=n10)
        u["kind"] = "full" if k == 0 else "full_repeat_%d" % (k + 1)
        fulls.append(u)
        if k == 0:
            models[(fa, fs)] = mdl_k
            out["unit"]["%s_%s" % (fa, fs)] = u            # fit 1 = the unit used for episodes (as before)
    out["full_fit_repeats"] = fulls
    out["linearity"] = linearity_record(fulls, fraction, n_updates, n10)
    out["cold_start"] = cold_start(fulls, n_updates, n10, warmups=out.get("warmups", []), n_process_starts=n_process_starts)
    event("probe_linearity", **out["linearity"])
    event("probe_cold_start", **out["cold_start"])
    if not out["linearity"]["ok"]:
        return out                                             # A1: stop, diagnosis (no further measurement)
    for arm, st in [(a, s) for a in S.ARMS for s in structures[a]]:
        if (arm, st) == (fa, fs):
            continue
        if arm == "LIN":
            t0 = clock()
            try:
                m, _ = model_lin.fit(train, ns, order=st)
                failed = False
            except model_lin.LinFailure:
                m, failed = None, True
            c = clock() - t0
            budget.check()
            models[(arm, st)] = m
            out["unit"]["LIN_%s" % st] = dict(arm=arm, structure=st, kind="full_closed_form", fit_cpu_s=c,
                                              failed=failed, per_update_cpu_s=None,
                                              refit_cpu_s_scaled_42ep=c * 42 / max(1, train["u"].shape[0]))
        else:
            do_warmup(arm, st)                                  # E1.1: before each timed fit segment
            models[(arm, st)], u = neural_fit(arm, st, n10)
            u["kind"] = "partial_%d_of_%d" % (n10, n_updates)
            out["unit"]["%s_%s" % (arm, st)] = u
    # (c) episodes: 2 full-size train-entry episodes per arm x structure, lambda = 1.0
    t = clock()
    scens = [P.Scenario(tr_seeds[i], i, S.TRAIN_IDS[i], G, "probe_train_entry") for i in range(S.PROBE_EPISODES_PER_CONFIG)]
    out["scenario_build_cpu_s"] = (clock() - t) / len(scens)
    out["episode_scenarios"] = [sc.meta() for sc in scens]
    budget.check()
    for arm in S.ARMS:
        for st in structures[arm]:
            pred = PR.make(arm, models[(arm, st)], ns)
            t = clock()
            rows = []
            for sc in scens:
                rows.append(CL.run_episode(sc, pred, S.PROBE_EPISODE_LAMBDA))
                budget.check()
            c = clock() - t
            dec = sum(x["n_mpc_decisions"] for x in rows)
            u = out["unit"]["%s_%s" % (arm, st)]
            u["episodes"] = len(rows)
            u["decisions"] = int(dec)
            u["fallbacks"] = int(sum(x["fallback"] for x in rows))
            u["episode_cpu_s_measured"] = c / len(rows)
            # as in v8.2: if fallback cut MPC decisions short, scale to 96 decisions per episode (conservative)
            u["episode_cpu_s"] = u["episode_cpu_s_measured"] * max(1.0, S.N_CONTROL * len(rows) / max(dec, 1))
            event("probe_episodes", arm=arm, structure=st, episode_cpu_s=u["episode_cpu_s"], probe_cpu_s=budget.used())
    t = clock()
    for sc in scens:
        CL.run_episode(sc, None, 1.0, mode="pi")
        budget.check()
    out["pi_episode_cpu_s"] = (clock() - t) / len(scens)
    ws = out.get("warmups", [])
    out["cold_start"] = cold_start(fulls, n_updates, n10, warmups=ws, n_process_starts=n_process_starts)  # all warm-ups
    out["warmup_total_cpu_s"] = sum(w["cpu_s"] for w in ws)
    out["complete"] = True
    return out


def _ratio(u, fraction):
    if "first_mark_train_cpu_s" in u and u["train_cpu_s"] > 0 and not u["failed"]:
        return (u["first_mark_train_cpu_s"] / fraction) / u["train_cpu_s"]
    return None


def linearity_record(fulls, fraction, n_updates, n10):
    """A1 linearity, ERRATUM E2.2: bounds [0.87, 1.15] applied to the post-warm-up window of the LAST full fit (fit 2)
    vs that fit's full training loop. Fit 1 is reported (per_fit) but does not gate. Top-level first_10pct / full_* /
    ratio = the gating fit (fit 2); full_fit_total_cpu_s per fit in per_fit (dahill ruling 3)."""
    per = []
    for u in fulls:
        r = _ratio(u, fraction)
        per.append(dict(ratio=r, ok=linearity_ok(r), first_10pct_train_cpu_s=u.get("first_mark_train_cpu_s"),
                        full_train_cpu_s=u["train_cpu_s"], full_fit_total_cpu_s=u["fit_cpu_s"],
                        full_fit_build_cpu_s=u["build_cpu_s"], full_fit_eval_cpu_s=u["eval_cpu_s"]))
    g = per[-1]
    return dict(ratio=g["ratio"], window=list(S.LINEARITY_WINDOW), gating_fit=len(per),
                first_10pct_train_cpu_s=g["first_10pct_train_cpu_s"], full_train_cpu_s=g["full_train_cpu_s"],
                full_fit_total_cpu_s=g["full_fit_total_cpu_s"], full_fit_build_cpu_s=g["full_fit_build_cpu_s"],
                full_fit_eval_cpu_s=g["full_fit_eval_cpu_s"], per_fit=per, ok=g["ok"], after_warmup=True,
                rule="E2.2: fit %d (post-warm-up) first-10%% window x 10 / its full loop in [0.87, 1.15]" % len(per))


def cold_start(fulls, n_updates, n10, warmups=(), n_process_starts=1, n_neural_fits=None):
    """ERRATUM E2.1/E2.2 + dahill req (2). Per full fit k: post-window rate r_k = (train_k - first_k)/(n - n10) and
    first-window excess e_k = first_k - n10 x r_k. W_warmup = measured warm-up CPU-s (max over the probe's warm-ups).
    Recurrence: e_1, e_2 <= 0 -> 'none'; e_2 < 0.5 x e_1 -> 'per_process'; otherwise 'per_fit'.
    Fixed cost in the fits projection (conservative reading, IMPL_NOTES):
      none        W_warmup x n_process_starts
      per_process (W_warmup + e_1) x n_process_starts
      per_fit     max(W_warmup, e_1, e_2) x n_neural_fits     (n_coldstarts := number of fits, E2.2)
    Linear part of a full fit = build + eval + max_k r_k x n_updates."""
    n_neural_fits = n_neural_fits or S.REPLICATES * sum(len(S.structures(a)) for a in NEURAL)
    per = []
    for u in fulls:
        f = u.get("first_mark_train_cpu_s")
        if f is None or u["failed"] or n_updates <= n10:
            per.append(dict(post_window_rate_cpu_s=None, excess_cpu_s=None))
            continue
        rate = (u["train_cpu_s"] - f) / (n_updates - n10)
        per.append(dict(post_window_rate_cpu_s=rate, excess_cpu_s=f - n10 * rate))
    ex = [p["excess_cpu_s"] for p in per]
    w_warm = max([w["cpu_s"] for w in warmups if w.get("updates")] or [0.0])
    if any(e is None for e in ex):
        return dict(per_fit=per, mode=None, W_cpu_s=None, n_cold_starts=None, linear_full_fit_cpu_s=None,
                    warmup_cpu_s=w_warm, rule="incomplete")
    e1, e2 = ex[0], ex[-1]
    if e1 <= 0 and e2 <= 0:
        mode, W, n_cold = "none", w_warm, n_process_starts
    elif len(ex) > 1 and e2 < S.COLD_START_PER_PROCESS_FRAC * e1:
        mode, W, n_cold = "per_process", w_warm + e1, n_process_starts
    else:
        mode, W, n_cold = "per_fit", max(w_warm, e1, e2), n_neural_fits
    rates = [p["post_window_rate_cpu_s"] for p in per]
    u0 = fulls[0]
    lin_full = u0["build_cpu_s"] + u0["eval_cpu_s"] + max(rates) * n_updates
    return dict(per_fit=per, excess1_cpu_s=e1, excess2_cpu_s=e2, warmup_cpu_s=w_warm, mode=mode, W_cpu_s=W,
                n_cold_starts=n_cold, n_process_starts=n_process_starts, n_neural_fits_production=n_neural_fits,
                cold_start_total_cpu_s=W * n_cold, linear_full_fit_cpu_s=lin_full,
                rule="e_k = first_k - n10 x post-window rate_k; none / per_process (e2 < %.2f x e1) / per_fit; "
                     "W = W_warmup | W_warmup + e1 | max(W_warmup, e1, e2)" % S.COLD_START_PER_PROCESS_FRAC)


def linearity_ok(ratio):
    lo, hi = S.LINEARITY_WINDOW
    return ratio is not None and lo <= ratio <= hi


# ---------------- projections (pure) ----------------
def episode_phase_projection(unit, sc, pi_ep, data_ep, ladder=()):
    """tuning / test / in_distribution / adaptation exactly as v8.2 preflight.project (same counts from the v8.2
    CONFIG, test/ID/adaptation at the per-arm max over structures), with the given per-unit costs."""
    tun_eps = 30 if "c" in ladder else 40
    adapt_eval = 5 if "a" in ladder else 10
    ep = lambda a, s: unit["%s_%s" % (a, s)]["episode_cpu_s"]
    ep_max = {a: max(ep(a, s) for s in S.structures(a)) for a in S.ARMS}
    upd_max = {a: max(unit["%s_%s" % (a, s)]["per_update_cpu_s"] for s in S.structures(a)) for a in NEURAL}
    proj = {}
    proj["tuning"] = tun_eps * sc + sum(ep(a, s) * 3 * len(S.LAMBDAS) * tun_eps for a in S.ARMS for s in S.structures(a))
    proj["test"] = 320 * sc + sum(ep_max[a] * 3 * 320 for a in S.ARMS)
    proj["in_distribution"] = 0.0 if "b" in ladder else 40 * sc + sum(ep_max[a] * 3 * 40 for a in S.ARMS)
    n_eval = 4 * adapt_eval
    proj["adaptation"] = (40 * data_ep + n_eval * (sc + pi_ep)
                          + sum(upd_max[a] * S.ADAPT_UPDATES * 36 for a in NEURAL)
                          + max(unit["LIN_%s" % s]["refit_cpu_s_scaled_42ep"] for s in S.LIN_ORDERS) * 36
                          + sum(ep_max[a] * 3 * 3 * n_eval for a in S.ARMS))
    return proj


def episode_count_projection(unit, ladder=()):
    """dahill CODE_PASS ruling 4: 'episodes x measured episode cost' per episode phase (closed-loop MPC episodes only;
    test/ID/adaptation at the per-arm max over structures, as in C3). Probe-measured episode costs only."""
    tun_eps = 30 if "c" in ladder else 40
    adapt_eval = 5 if "a" in ladder else 10
    ep = lambda a, s: unit["%s_%s" % (a, s)]["episode_cpu_s"]
    ep_max = {a: max(ep(a, s) for s in S.structures(a)) for a in S.ARMS}
    return dict(tuning=sum(ep(a, s) * 3 * len(S.LAMBDAS) * tun_eps for a in S.ARMS for s in S.structures(a)),
                test=sum(ep_max[a] * 3 * 320 for a in S.ARMS),
                in_distribution=0.0 if "b" in ladder else sum(ep_max[a] * 3 * 40 for a in S.ARMS),
                adaptation=sum(ep_max[a] * 3 * 3 * 4 * adapt_eval for a in S.ARMS))


def measured_episode_phases(pm, ladder=()):
    """Per episode phase: max(v8.2 formula with probe unit costs, episodes x measured episode cost) (ruling 4).
    Inputs: pm only (the probe). Returns (projection, record with both values and the winner)."""
    f = episode_phase_projection(pm["unit"], pm["scenario_build_cpu_s"], pm["pi_episode_cpu_s"],
                                 pm["data_episode_cpu_s"], ladder)
    c = episode_count_projection(pm["unit"], ladder)
    proj, rec = {}, {}
    for k in S.K2_PHASES:
        proj[k] = max(f[k], c[k])
        rec[k] = dict(v82_formula=f[k], episodes_x_cost=c[k], used=proj[k],
                      winner="v82_formula" if f[k] >= c[k] else "episodes_x_cost")
    return proj, rec


def adaptation_max_jump(pm):
    """dahill ruling 7: projected largest CPU between two adaptation cap checks = fine-tuning one arm's 3 replicates
    (3 x 100 updates at the largest per-update cost) + one evaluation episode (largest)."""
    neural = [pm["unit"]["%s_%s" % (a, s)]["per_update_cpu_s"] for a in NEURAL for s in S.structures(a)]
    lin = [pm["unit"]["LIN_%s" % s]["refit_cpu_s_scaled_42ep"] for s in S.LIN_ORDERS]
    ep = max(pm["unit"]["%s_%s" % (a, s)]["episode_cpu_s"] for a, s in _units())
    return max(S.REPLICATES * S.ADAPT_UPDATES * max(neural), S.REPLICATES * max(lin)) + ep


def toy_episode_projection(toy, ladder=()):
    """K2 toy reference: the same formulas on the bound v8.2 toy preflight measurements."""
    m = toy["measurements"]
    return episode_phase_projection(m["unit"], m["scenario_build_cpu_s"], m["pi_episode_cpu_s"],
                                    m["data_episode_cpu_s"], ladder)


def fits_projection(pm):
    """C3: fits = WM h96 full cost + 10 x the 10 % costs of the other neural fits + LIN full closed-form fits,
    plus the per-replicate window construction, times REPLICATES (3; production fits 3 replicates)."""
    fa, fs = pm["full_fit"]
    u = pm["unit"]
    cs = pm.get("cold_start")
    if cs and cs.get("linear_full_fit_cpu_s") is not None:
        full = cs["linear_full_fit_cpu_s"]              # dahill req (2): linear part from the post-window rate
        cold = cs["W_cpu_s"] * cs["n_cold_starts"]       # + W x number of cold starts (per fit or per process)
    else:
        full, cold, cs = u["%s_%s" % (fa, fs)]["fit_cpu_s"], 0.0, None
    other = 0.0
    lin = 0.0
    for a, s in _units():
        if (a, s) == (fa, fs):
            continue
        if a == "LIN":
            lin += u["LIN_%s" % s]["fit_cpu_s"]
        else:
            other += u["%s_%s" % (a, s)]["fit_cpu_s"] / pm["fraction"]
    per_rep = pm["windows_per_replicate_cpu_s"] + full + other + lin
    return dict(per_replicate=per_rep, replicates=S.REPLICATES, total=S.REPLICATES * per_rep + cold,
                terms=dict(windows=pm["windows_per_replicate_cpu_s"], full_fit=full, other_neural_x10=other, lin=lin,
                           cold_start_total=cold, cold_start_mode=(cs or {}).get("mode")))


def caps_from(proj):
    caps = dict(S.FIXED_CAPS)
    for k in S.MEASURED_PHASES:
        caps[k] = int(math.ceil(S.CAP_FACTOR * proj[k]))
    return caps


def apply_k2(measured, toy, R):
    """K2: if R > 1.5 and a phase's measured projection is more than 20 % below toy x R, use toy x R (larger)."""
    out, rec = dict(measured), {}
    for k in S.K2_PHASES:
        tr = toy[k] * R
        trig = R > S.K2_R_THRESHOLD and measured[k] < (1.0 - S.K2_BELOW_FRAC) * tr
        rec[k] = dict(measured=measured[k], toy=toy[k], toy_x_R=tr, replaced=bool(trig))
        if trig:
            out[k] = max(measured[k], tr)
    return out, rec


def decide(pm, toy, aborted=None, toy_fits=S.K2_TOY_FITS):
    """A1 + C3 + K1/K2. pm: probe measurements (the ONLY source of every per-unit cost in the projection);
    toy: the K2 toy reference dict (box: v8.2 PREFLIGHT_PROJECTION.json; other host: that host's toy attempt),
    used ONLY for the K2 comparison; toy_fits: its fits projection (box: 336.8 plan literal)."""
    if aborted:
        return dict(outcome="PROBE_DIAGNOSIS", diagnosis_kind="cap", reason="probe cap: %s" % aborted, projection=None,
                    caps=None)
    lin = pm.get("linearity") or {}
    if not lin.get("ok"):
        per = [x.get("ratio") for x in lin.get("per_fit", [])] or [lin.get("ratio")]
        return dict(outcome="PROBE_DIAGNOSIS", diagnosis_kind="linearity",
                    reason="linearity ratio %r outside %r (per fit %r)" % (lin.get("ratio"), list(S.LINEARITY_WINDOW), per),
                    projection=None, caps=None)
    if not pm.get("complete"):
        return dict(outcome="PROBE_DIAGNOSIS", diagnosis_kind="incomplete", reason="probe incomplete", projection=None,
                    caps=None)
    fits = fits_projection(pm)
    R = fits["total"] / float(toy_fits)
    toy_full = toy_episode_projection(toy, ())         # stricter reading S1: K2 toy reference WITHOUT ladder reduction
    tries = []
    for n in range(len(S.V83_LADDER) + 1):
        ladder = tuple(S.V83_LADDER[:n])
        measured, mrec = measured_episode_phases(pm, ladder)
        toy_p = {k: (0.0 if (k == "in_distribution" and "b" in ladder) else toy_full[k]) for k in S.K2_PHASES}
        used, k2 = apply_k2(measured, toy_p, R)
        proj = dict(used, fits=fits["total"])
        caps = caps_from(proj)
        total_caps = sum(caps.values())
        ok = total_caps <= S.K1_GLOBAL_CAP
        tries.append(dict(ladder=list(ladder), measured_projection=measured, measured_projection_detail=mrec,
                          toy_projection=toy_p, k2=k2,
                          projection=proj, measured_projection_sum=sum(proj.values()), caps=caps,
                          sum_of_caps=total_caps, gate_pass=ok))
        if ok:
            break
    last = tries[-1]
    return dict(outcome="GATE_PASS" if last["gate_pass"] else "NOT_LAUNCHED",
                reason=None if last["gate_pass"] else "sum of caps %d > %d after ladder %s" % (
                    last["sum_of_caps"], S.K1_GLOBAL_CAP, last["ladder"]),
                R=R, R_threshold=S.K2_R_THRESHOLD, k2_toy_fits=float(toy_fits), fits=fits,
                full_fit_total_cpu_s=lin.get("full_fit_total_cpu_s"),
                full_fit_total_cpu_s_per_fit=[x.get("full_fit_total_cpu_s") for x in lin.get("per_fit", [])],
                linearity_per_fit=[x.get("ratio") for x in lin.get("per_fit", [])], cold_start=pm.get("cold_start"),
                warmup_total_cpu_s=pm.get("warmup_total_cpu_s"),
                warmup_rule="E2.1 (replaces E1.1 'not part of any projection'): W x n_coldstarts is a fixed cost in "
                            "the fits projection; probe warm-ups count toward the 450 s cap",
                adaptation_projected_max_jump_cpu_s=adaptation_max_jump(pm),
                ladder_applied=last["ladder"], projection=last["projection"], caps=last["caps"] if last["gate_pass"] else None,
                global_cap=last["sum_of_caps"] if last["gate_pass"] else None, k1_global_cap=S.K1_GLOBAL_CAP,
                gate_rule="sum(fixed 260 + ceil(1.25 x projection) of the 5 measured phases) <= 3600 "
                          "(<=> projections <= 2672 up to ceil rounding)", tries=tries)
