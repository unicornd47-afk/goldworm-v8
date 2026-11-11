"""Closed-loop episode execution (scorer side). Controllers see only an ObservableView; the Oracle
receives privileged state through a separate path. Scoring per PLAN §6/§7:
 step k in 96..191: E_k = 2.5*u_cmd_k; d_k = 0.25*min(v_{k+1},11.5); J_k = 2.5u + 0.25(4v+8v^2).
 Non-finite prediction -> frozen PI for the rest of the episode (flagged, counted).
 Safety exit (true T_r outside [10,30]) at step i -> E=2.5, d=2.875, J=J_MAX for steps i..191."""
import numpy as np

from . import plant_iv as P
from . import planner as PL
from . import spec as S
from .features import ObservableView, schedule


def _pi_controller(view, u_prev):
    pi = P.PI(*S.PI_CFG)
    sp = lambda t: float(schedule(np.array([t]), view.s_occ, view.s_night)[0])
    pi.handover(u_prev, pi.error(view.k, view.y_r[view.k], sp))
    return pi, sp


def run_episode(sc, predictor, lam, mode="mpc", budget_check=None):
    """mode: 'mpc' (predictor may be None = failed fit -> PI fallback), 'pi', 'oracle'."""
    n = S.N_CONTROL
    E = np.zeros(n)
    dd = np.zeros(n)
    J = np.zeros(n)
    res = dict(meta=sc.meta(), lam=lam, mode=mode, fallback=False, fallback_k=None, fallback_reason=None,
               safety_exit=False, safety_k=None, n_mpc_decisions=0)
    if sc.warm_failed_at is not None:
        E[:], dd[:], J[:] = 2.5, PL.D_STEP_MAX, PL.J_MAX
        res.update(safety_exit=True, safety_k=int(sc.warm_failed_at), warmup_exit=True)
        return _finish(res, E, dd, J, None, sc)
    d, sch = sc.d, sc.sch
    y, To, u = sc.y.copy(), sc.To.copy(), sc.u.copy()
    x = sc.x96.copy()
    Tr_true = np.full(S.EP_TRANSITIONS_SCEN + 1, np.nan)
    Tr_true[S.WARMUP] = x[0]
    pi = sp = None
    if mode == "pi":
        view = ObservableView(S.CONTROL_FIRST, y, To, u, d.s_occ, d.s_night)
        pi, sp = _pi_controller(view, float(u[S.CONTROL_FIRST - 1]))
    elif mode == "mpc" and predictor is None:
        res.update(fallback=True, fallback_k=S.CONTROL_FIRST, fallback_reason="failed_fit")
        view = ObservableView(S.CONTROL_FIRST, y, To, u, d.s_occ, d.s_night)
        pi, sp = _pi_controller(view, float(u[S.CONTROL_FIRST - 1]))
    for k in range(S.CONTROL_FIRST, S.CONTROL_LAST + 1):
        i = k - S.CONTROL_FIRST
        view = ObservableView(k, y, To, u, d.s_occ, d.s_night)
        if pi is not None:
            act, _ = pi.step(pi.error(k, view.y_r[k], sp))
        elif mode == "oracle":
            act = _oracle_decide(sc, view, x, lam, u)
        else:
            plans = PL.library(sc.episode_id, k)
            L = view.lower(np.arange(k + 1, k + PL.H + 1))
            yhat = predictor(view, plans)
            idx = None
            if yhat is not None:
                idx, _ = PL.plan(yhat, plans, L, lam)
            res["n_mpc_decisions"] += 1
            if idx is None:
                res.update(fallback=True, fallback_k=k, fallback_reason="nonfinite_prediction")
                pi, sp = _pi_controller(view, float(u[k - 1]))
                act, _ = pi.step(pi.error(k, view.y_r[k], sp))
            else:
                act = float(plans[idx, 0])
        if not (0.0 <= act <= 1.0):
            raise AssertionError("unclipped action")
        u[k] = act
        p = sch.params(k)
        x = P.step(x, P.applied_u(u, k, p), k, d, p)
        Tr_true[k + 1] = x[0]
        if not P.in_safety(x[0]):
            E[i:], dd[i:], J[i:] = 2.5, PL.D_STEP_MAX, PL.J_MAX
            res.update(safety_exit=True, safety_k=k)
            break
        L_next = P.setpoint(k + 1, d.s_occ, d.s_night) - P.COMFORT_LOWER_OFFSET
        v = min(max(0.0, L_next - x[0]) + max(0.0, x[0] - P.U_COMFORT), PL.V_MAX)
        E[i] = 2.5 * act
        dd[i] = 0.25 * v
        J[i] = 2.5 * act + 0.25 * (4.0 * v + 8.0 * v * v)
        y[k + 1], To[k + 1] = P.measure(x, k + 1, d, sch.params(k + 1))
        if budget_check is not None:
            budget_check()
    res["traj"] = dict(y=y, To=To, u=u)
    return _finish(res, E, dd, J, Tr_true, sc)


def settling_time(Tr_true, sc):
    """First k >= 120 with |T_r - setpoint| < 0.5 K for 8 consecutive steps (log only)."""
    if Tr_true is None:
        return None
    run = 0
    for k in range(120, S.EP_TRANSITIONS_SCEN + 1):
        if not np.isfinite(Tr_true[k]):
            return None
        if abs(Tr_true[k] - P.setpoint(k, sc.d.s_occ, sc.d.s_night)) < 0.5:
            run += 1
            if run == 8:
                return k - 7
        else:
            run = 0
    return None


def window_sum(a, win):
    lo, hi = win
    return float(np.sum(a[lo - S.CONTROL_FIRST: hi - S.CONTROL_FIRST + 1]))


def _finish(res, E, dd, J, Tr_true, sc):
    res.update(E_step=E, d_step=dd, J_step=J,
               E=window_sum(E, S.WIN_PRIMARY), D=window_sum(dd, S.WIN_PRIMARY), J=window_sum(J, S.WIN_PRIMARY),
               E2=window_sum(E, S.WIN_SECONDARY), D2=window_sum(dd, S.WIN_SECONDARY),
               J2=window_sum(J, S.WIN_SECONDARY), settling_k=settling_time(Tr_true, sc))
    return res


# ---------------- Oracle-MPC (descriptive only, privileged) ----------------
def _oracle_decide(sc, view, x_k, lam, u_cmd):
    """True dynamics incl. the known intervention schedule, conditional-mean exogenous (no innovations)."""
    k = view.k
    plans = PL.library(sc.episode_id, k)
    L = view.lower(np.arange(k + 1, k + PL.H + 1))
    d, sch = sc.d, sc.sch
    M = plans.shape[0]
    X = np.repeat(x_k[None, :], M, 0)
    out = np.empty((M, PL.H))
    for j in range(PL.H):
        kk = k + j
        p = sch.params(kk)
        if p.dead == 0:
            ua = plans[:, j]
        else:
            src = kk - p.dead
            ua = plans[:, src - k] if src >= k else np.full(M, u_cmd[src])
        Tr, Tw, q, dl, eo = X.T
        To = P.outdoor_mean(kk, d.mu_o, d.A_o) + p.To_off + eo
        a, b = p.A_TR, p.A_TW
        X = np.stack([a[0] * Tr + a[1] * Tw + a[2] * To + a[3] * q + a[4] * (p.D_BAR + dl),
                      b[0] * Tr + b[1] * Tw + b[2] * To, p.A_QH[0] * q + p.A_QH[1] * ua,
                      P.RHO_D * dl, P.RHO_O * eo], -1)
        out[:, j] = X[:, 0]
    idx, _ = PL.plan(out, plans, L, lam)
    return float(plans[idx, 0])
