"""Scorer-side plant: Option1 thermal plant (thermal5/plant.py, constants.py; ported, numerically
identical under nominal parameters, see tests/test_plant_equivalence.py) extended ONLY by the
time-windowed parameter overrides of INTERVENTIONS_v8.1.json. Model modules never import this file."""
import math

import numpy as np

from . import seeds as SD
from . import spec as S

# ---- Option1 constants (thermal5/constants.py §3) ----
STEPS_PER_DAY = 96
D_BAR = 0.5
A_TR = (0.925, 0.0625, 0.0125, 0.125, 0.125)
A_TW = (0.0125, 0.98125, 0.00625)
A_QH = (0.60, 4.0)
RHO_D, SIG_D = 0.95, 0.10
RHO_O, SIG_O = 0.99, 0.15
SIG_Y = 0.10
S_OCC_LEVELS = (20.0, 20.5, 21.0, 21.5, 22.0)
S_NIGHT_LEVELS = (16.0, 17.0, 18.0)
OCC_START_H, OCC_END_H = 6.0, 22.0
COMFORT_LOWER_OFFSET = 0.5
U_COMFORT = 24.0
SAFE_LO, SAFE_HI = 10.0, 30.0
INIT_SD_EO, INIT_SD_DELTA = 1.06, 0.32
EXC_GEOM_P = 1.0 / 8.0
EXC_HOLD_CAP = 32
EXC_NOISE_SD = 0.15
EXC_HOLD_PROB = 0.05
EXC_HOLD_LEN = 4
BASE_PI = (0.4, 2.0, 0)


def hour(k):
    return (k % STEPS_PER_DAY) / 4.0


def setpoint(k, s_occ, s_night):
    h = hour(k)
    return s_occ if (OCC_START_H <= h < OCC_END_H) else s_night


def setpoint_array(ks, s_occ, s_night):
    ks = np.asarray(ks)
    h = (ks % STEPS_PER_DAY) / 4.0
    return np.where((h >= OCC_START_H) & (h < OCC_END_H), s_occ, s_night).astype(np.float64)


def outdoor_mean(k, mu_o, A_o):
    return mu_o - A_o * math.cos(2.0 * math.pi * (hour(k) - 5.0) / 24.0)


class Draw:
    __slots__ = ("mu_o", "A_o", "e_o0", "delta0", "T_w0", "T_r0", "q_h0", "s_occ", "s_night",
                 "eps_d", "eps_o", "nu", "n")


def draw_episode(rng, N):
    """Option1 draw_episode order (§3): mu_o, A_o, e_o0, delta0, T_w0, T_r0, q_h0, s_occ, s_night, eps_d, eps_o, nu."""
    d = Draw()
    d.mu_o = float(rng.uniform(-3.0, 10.0))
    d.A_o = float(rng.uniform(2.0, 5.0))
    d.e_o0 = float(rng.normal(0.0, INIT_SD_EO))
    d.delta0 = float(rng.normal(0.0, INIT_SD_DELTA))
    d.T_w0 = float(rng.uniform(15.0, 20.0))
    d.T_r0 = d.T_w0 + float(rng.uniform(-0.5, 1.5))
    d.q_h0 = float(rng.uniform(0.0, 5.0))
    d.s_occ = S_OCC_LEVELS[int(rng.integers(0, len(S_OCC_LEVELS)))]
    d.s_night = S_NIGHT_LEVELS[int(rng.integers(0, len(S_NIGHT_LEVELS)))]
    d.eps_d = rng.standard_normal(N)
    d.eps_o = rng.standard_normal(N)
    d.nu = rng.standard_normal(N + 1)
    d.n = N
    return d


def apply_initial_state(d, position):
    """PLAN §5 fix 3: (T_r0, T_w0, q_h0) := S[i mod 4]; delta0/e_o0/weather/schedule stay from the seed draw."""
    d.T_r0, d.T_w0, d.q_h0 = (float(v) for v in SD.initial_state(position))
    return d


# ---------------- interventions ----------------
class Params:
    """Per-step plant parameters (nominal unless an override window is active)."""
    __slots__ = ("A_TR", "A_TW", "A_QH", "D_BAR", "To_off", "dead")

    def __init__(self):
        self.A_TR = A_TR
        self.A_TW = A_TW
        self.A_QH = A_QH
        self.D_BAR = D_BAR
        self.To_off = 0.0
        self.dead = 0


NOMINAL = Params()


def override_params(iv_id):
    """Params during the active window of entry iv_id (INTERVENTIONS_v8.1.json)."""
    p = Params()
    if iv_id == "N0":
        return p
    e = S.IV_BY_ID[iv_id]
    t = e["type"]
    if t == "internal_gain_step":
        p.D_BAR = float(e["value"])
    elif t == "ambient_offset_in_range":
        p.To_off = float(e["value"])
    elif t == "heater_time_constant_const_gain":
        p.A_QH = (float(e["value"][0]), float(e["value"][1]))
    elif t == "heater_derating":
        p.A_QH = (A_QH[0], float(e["value"]))
    elif t == "envelope_loss_step_window_open":
        a2 = float(e["value"][1])
        a0 = 1.0 - A_TR[1] - a2
        if abs(a0 - float(e["value"][0])) > 1e-12:
            raise AssertionError("H2 A_TR[0] inconsistent with file")
        p.A_TR = (a0, A_TR[1], a2, A_TR[3], A_TR[4])
    elif t == "wall_thermal_mass_change":
        b = tuple(float(v) for v in e["value"])
        if abs(sum(b) - 1.0) > 1e-12:
            raise AssertionError("H3 b does not sum to 1")
        p.A_TW = b
    elif t == "actuator_dead_time":
        p.dead = int(e["value"])
    else:
        raise AssertionError("unknown intervention type %r" % t)
    return p


class Schedule:
    """Override active for onset <= k < onset + duration, nominal otherwise."""

    def __init__(self, iv_id, onset):
        self.iv_id = iv_id
        self.onset = None if iv_id == "N0" else int(onset)
        self.dur = S.IV_DURATION
        self.p_on = override_params(iv_id)

    def active(self, k):
        return self.onset is not None and self.onset <= k < self.onset + self.dur

    def params(self, k):
        return self.p_on if self.active(k) else NOMINAL


def step(x, u_applied, k, d, p):
    """x_k -> x_{k+1} (Option1 §3 step with per-step parameters p). u_applied in [0,1]."""
    if not (0.0 <= u_applied <= 1.0):
        raise AssertionError("unclipped action %r" % (u_applied,))
    T_r, T_w, q, dl, eo = x
    To = outdoor_mean(k, d.mu_o, d.A_o) + p.To_off + eo
    a, b = p.A_TR, p.A_TW
    return np.array([
        a[0] * T_r + a[1] * T_w + a[2] * To + a[3] * q + a[4] * (p.D_BAR + dl),
        b[0] * T_r + b[1] * T_w + b[2] * To,
        p.A_QH[0] * q + p.A_QH[1] * u_applied,
        RHO_D * dl + SIG_D * d.eps_d[k],
        RHO_O * eo + SIG_O * d.eps_o[k],
    ], dtype=np.float64)


def measure(x, k, d, p):
    """(y_r[k], T_o[k]); the ambient offset (T2) shifts the measured T_o as well (IMPL_NOTES D6)."""
    return x[0] + SIG_Y * d.nu[k], outdoor_mean(k, d.mu_o, d.A_o) + p.To_off + x[4]


def in_safety(T_r):
    return SAFE_LO <= T_r <= SAFE_HI


def applied_u(u_cmd_hist, k, p):
    """H4: applied u_k = commanded u_{k-d} from the real commanded history (d=0 -> u_k)."""
    if p.dead == 0:
        return float(u_cmd_hist[k])
    j = k - p.dead
    return float(u_cmd_hist[j]) if j >= 0 else 0.0


class PI:
    """Option1 PI law (§4.1-§4.4)."""

    def __init__(self, Kp, Ti, ell):
        self.Kp, self.Ti, self.ell = float(Kp), float(Ti), int(ell)
        self.I = None

    def error(self, k, y_k, sp):
        return sp(k + self.ell) - y_k

    def start_base(self, e0):
        self.I = -self.Kp * e0

    def handover(self, u_prev, e_k):
        self.I = u_prev - self.Kp * e_k

    def step(self, e):
        v = self.Kp * e + self.I
        u = 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)
        skip = (v >= 1.0 and e > 0.0) or (v <= 0.0 and e < 0.0)
        if not skip:
            self.I = self.I + self.Kp * (S.DT / self.Ti) * e
        return u, v


def random_hold_actions(exc, N):
    u = np.empty(N)
    k = 0
    while k < N:
        G = int(exc.geometric(EXC_GEOM_P)) - 1
        L = min(EXC_HOLD_CAP, 1 + G)
        level = float(exc.uniform(0.0, 1.0))
        u[k:min(N, k + L)] = level
        k += L
    return u


def data_episode(seed, position, iv_id, onset, guard, purpose, N=None):
    """Option1-style data episode (288 transitions) for training/validation/adaptation data.
    position = 0-based index in its seed list (initial state rule and even/odd excitation rule)."""
    N = S.EP_TRANSITIONS_DATA if N is None else N
    guard.check(seed, purpose)
    d = apply_initial_state(draw_episode(SD.draw_rng(seed), N), position)
    exc = SD.excitation_rng(seed)
    sch = Schedule(iv_id, onset)
    x = np.array([d.T_r0, d.T_w0, d.q_h0, d.delta0, d.e_o0])
    y = np.empty(N + 1)
    To = np.empty(N + 1)
    u = np.empty(N)
    sp = lambda k: setpoint(k, d.s_occ, d.s_night)
    holds = random_hold_actions(exc, N) if position % 2 == 0 else None
    pi = PI(*BASE_PI) if position % 2 == 1 else None
    hold_left, hold_level = 0, 0.0
    for k in range(N):
        p = sch.params(k)
        y[k], To[k] = measure(x, k, d, p)
        if holds is not None:
            uk = float(holds[k])
        else:
            ek = pi.error(k, y[k], sp)
            if k == 0:
                pi.start_base(ek)
            _, vk = pi.step(ek)
            nk = float(exc.normal(0.0, EXC_NOISE_SD))
            if hold_left == 0 and float(exc.uniform(0.0, 1.0)) < EXC_HOLD_PROB:
                hold_level = float(exc.uniform(0.0, 1.0))
                hold_left = EXC_HOLD_LEN
            if hold_left > 0:
                uk = hold_level
                hold_left -= 1
            else:
                uk = min(1.0, max(0.0, vk + nk))
        u[k] = uk
        x = step(x, applied_u(u, k, p), k, d, p)
    y[N], To[N] = measure(x, N, d, sch.params(N))
    obs = dict(y_r=y, T_o=To, u=u, s=setpoint_array(np.arange(N + 1), d.s_occ, d.s_night),
               s_occ=d.s_occ, s_night=d.s_night)
    meta = dict(seed=int(seed), position=int(position), iv_id=iv_id, onset=onset, purpose=purpose,
                init_state=SD.initial_state_index(position))
    return obs, meta


def stack_obs(obs_list):
    return {k: (np.stack([o[k] for o in obs_list]) if np.ndim(obs_list[0][k]) else np.array([o[k] for o in obs_list]))
            for k in obs_list[0]}


class Scenario:
    """Tuning/test/adaptation-eval episode: 192 transitions, warm-up k=0..95 under the frozen PI
    (identical for every arm), intervention window per INTERVENTIONS (onset 120)."""

    def __init__(self, seed, position, iv_id, guard, purpose, onset=S.IV_ONSET_TEST):
        guard.check(seed, purpose)
        N = S.EP_TRANSITIONS_SCEN
        self.seed, self.position, self.iv_id, self.purpose = int(seed), int(position), iv_id, purpose
        self.episode_id = SD.episode_id(seed, iv_id)
        self.d = d = apply_initial_state(draw_episode(SD.draw_rng(seed), N), position)
        self.sch = Schedule(iv_id, onset)
        self.y = np.full(N + 1, np.nan)
        self.To = np.full(N + 1, np.nan)
        self.u = np.full(N, np.nan)
        x = np.array([d.T_r0, d.T_w0, d.q_h0, d.delta0, d.e_o0])
        pi = PI(*S.PI_CFG)
        sp = lambda k: setpoint(k, d.s_occ, d.s_night)
        self.warm_failed_at = None
        for k in range(S.WARMUP):
            p = self.sch.params(k)
            self.y[k], self.To[k] = measure(x, k, d, p)
            ek = pi.error(k, self.y[k], sp)
            if k == 0:
                pi.start_base(ek)
            uk, _ = pi.step(ek)
            self.u[k] = uk
            x = step(x, applied_u(self.u, k, p), k, d, p)
            if not in_safety(x[0]):
                self.warm_failed_at = k
                break
        self.x96 = x if self.warm_failed_at is None else None
        if self.warm_failed_at is None:
            self.y[S.WARMUP], self.To[S.WARMUP] = measure(x, S.WARMUP, d, self.sch.params(S.WARMUP))

    def meta(self):
        return dict(seed=self.seed, position=self.position, iv_id=self.iv_id, purpose=self.purpose,
                    episode_id=self.episode_id, init_state=SD.initial_state_index(self.position),
                    warm_failed_at=self.warm_failed_at)
