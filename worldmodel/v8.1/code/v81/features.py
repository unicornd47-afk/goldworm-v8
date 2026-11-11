"""Observable-only feature construction (port of Option1 thermal5/features.py). Imports nothing from
plant/intervention/scorer modules. Every arm receives exactly these 6 observables (PLAN §3):
y_r, T_o, setpoint s, commanded u, clock sin, clock cos. No indicator channel exists."""
import math

import numpy as np

CONTEXT = 48
H = 12
STEPS_PER_DAY = 96
INPUT_NAMES = ("y_r", "T_o", "s", "u_cmd_prev", "clock_sin", "clock_cos")
N_INPUTS = 6


def clock(ks):
    ks = np.asarray(ks)
    a = 2.0 * np.pi * ((ks % STEPS_PER_DAY) / 4.0) / 24.0
    return np.stack([np.sin(a), np.cos(a)], axis=-1)


def schedule(ks, s_occ, s_night):
    ks = np.asarray(ks)
    h = (ks % STEPS_PER_DAY) / 4.0
    return np.where((h >= 6.0) & (h < 22.0), s_occ, s_night).astype(np.float64)


class ObservableView:
    """Information at decision k: y_r[0..k], T_o[0..k], commanded u[0..k-1], public schedule, clock."""

    __slots__ = ("k", "y_r", "T_o", "u", "s_occ", "s_night")

    def __init__(self, k, y_r, T_o, u, s_occ, s_night):
        self.k = int(k)
        self.y_r = np.array(y_r[: k + 1], dtype=np.float64, copy=True)
        self.T_o = np.array(T_o[: k + 1], dtype=np.float64, copy=True)
        self.u = np.array(u[:k], dtype=np.float64, copy=True)
        self.s_occ = float(s_occ)
        self.s_night = float(s_night)
        if self.y_r.shape != (k + 1,) or self.T_o.shape != (k + 1,) or self.u.shape != (k,):
            raise AssertionError("observable view shape mismatch")
        if not (np.all(np.isfinite(self.y_r)) and np.all(np.isfinite(self.T_o)) and np.all(np.isfinite(self.u))):
            raise AssertionError("observable view contains non-finite entries")

    def lower(self, ks):
        return schedule(ks, self.s_occ, self.s_night) - 0.5


class NormStats:
    """Training mean/sd (ddof=0) for y_r, T_o, s, u of one replicate's training set."""

    FIELDS = ("y_r", "T_o", "s", "u")

    def __init__(self, means, sds):
        self.mean = {k: float(means[k]) for k in self.FIELDS}
        self.sd = {k: float(sds[k]) for k in self.FIELDS}
        for k in self.FIELDS:
            if not (math.isfinite(self.sd[k]) and self.sd[k] > 0):
                raise ValueError("degenerate normalization for %s" % k)

    @classmethod
    def from_train(cls, obs_train):
        means, sds = {}, {}
        for k in cls.FIELDS:
            a = np.asarray(obs_train[k], dtype=np.float64).ravel()
            means[k] = a.mean()
            sds[k] = a.std()
        return cls(means, sds)

    def to_dict(self):
        return {"mean": self.mean, "sd": self.sd}

    def n(self, field, a):
        return (np.asarray(a, dtype=np.float64) - self.mean[field]) / self.sd[field]

    def denorm(self, field, a):
        return np.asarray(a, dtype=np.float64) * self.sd[field] + self.mean[field]


def context_indices(k):
    return np.arange(k - CONTEXT + 1, k + 1)


def ctx_features(y_r, T_o, s, u, k, ns):
    """(48,6) f_t = [y_r[t], T_o[t], s_t, u_{t-1}, sin, cos] for t = k-47..k (normalized; clock raw)."""
    t = context_indices(k)
    if t[0] - 1 < 0:
        raise AssertionError("context needs k >= 48")
    f = np.empty((CONTEXT, N_INPUTS))
    f[:, 0] = ns.n("y_r", y_r[t])
    f[:, 1] = ns.n("T_o", T_o[t])
    f[:, 2] = ns.n("s", s[t])
    f[:, 3] = ns.n("u", u[t - 1])
    f[:, 4:6] = clock(t)
    return f


def ctx_from_view(view, ns):
    t_all = np.arange(view.k + 1)
    s = schedule(t_all, view.s_occ, view.s_night)
    return ctx_features(view.y_r, view.T_o, s, view.u, view.k, ns)


def fut_inputs(plans, k, ns):
    """(M,12,3) decoder inputs: step j gets [u_{k+j-1} (normalized), c_{k+j}]."""
    plans = np.atleast_2d(np.asarray(plans, dtype=np.float64))
    M = plans.shape[0]
    out = np.empty((M, H, 3))
    out[:, :, 0] = ns.n("u", plans)
    out[:, :, 1:3] = clock(np.arange(k + 1, k + H + 1))[None, :, :]
    return out


def window_arrays(obs, pairs, ns):
    """ctx (B,48,6), fut (B,12,3) from logged commanded actions, tgt (B,12) normalized y_r."""
    pairs = np.asarray(pairs, dtype=np.int64)
    e, k = pairs[:, 0], pairs[:, 1]
    B = len(pairs)
    t = k[:, None] + np.arange(-CONTEXT + 1, 1)[None, :]
    ctx = np.empty((B, CONTEXT, N_INPUTS))
    ctx[:, :, 0] = ns.n("y_r", obs["y_r"][e[:, None], t])
    ctx[:, :, 1] = ns.n("T_o", obs["T_o"][e[:, None], t])
    ctx[:, :, 2] = ns.n("s", obs["s"][e[:, None], t])
    ctx[:, :, 3] = ns.n("u", obs["u"][e[:, None], t - 1])
    ctx[:, :, 4:6] = clock(t)
    j = np.arange(1, H + 1)
    tf = k[:, None] + j[None, :]
    fut = np.empty((B, H, 3))
    fut[:, :, 0] = ns.n("u", obs["u"][e[:, None], tf - 1])
    fut[:, :, 1:3] = clock(tf)
    tgt = ns.n("y_r", obs["y_r"][e[:, None], tf])
    return ctx, fut, tgt


def train_pairs(n_ep, origin_min=48, origin_max=276):
    n_or = origin_max - origin_min + 1
    i = np.arange(n_ep * n_or)
    return np.stack([i // n_or, origin_min + i % n_or], axis=1)
