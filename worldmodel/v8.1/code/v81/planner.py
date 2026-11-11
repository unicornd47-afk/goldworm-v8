"""Option1 64-candidate library MPC (thermal5/planner.py, scoring.py) with the energy weight lambda.
Library child key (10, episode_id, k) under the Option1 root entropy: identical for every arm."""
import numpy as np

ROOT_ENTROPY = int("3ce616721340579be585abd64c5c9915fa294b9cc0c80e057f122db5459185df"[:32], 16)   # Option1 §5.1
CAND_STREAM = 10
H = 12
N_CAND = 64
CONST_PLANS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_RANDOM_PLANS = 59
N_BLOCKS, BLOCK_LEN = 3, 4
N_LEVELS = 11
V_MAX = 11.5
U_COMFORT = 24.0
SAFE_LO, SAFE_HI = 10.0, 30.0
J_MAX = 2.5 + 0.25 * (4 * V_MAX + 8 * V_MAX ** 2)     # 278.5
D_STEP_MAX = 0.25 * V_MAX                             # 2.875


def library(episode_id, k):
    ss = np.random.SeedSequence(entropy=ROOT_ENTROPY, spawn_key=(CAND_STREAM, int(episode_id), int(k)))
    g = np.random.Generator(np.random.PCG64(ss))
    lv = g.integers(0, N_LEVELS, size=(N_RANDOM_PLANS, N_BLOCKS)) / 10.0
    const = np.repeat(np.array(CONST_PLANS)[:, None], H, axis=1)
    lib = np.vstack([const, np.repeat(lv, BLOCK_LEN, axis=1)])
    assert lib.shape == (N_CAND, H)
    return lib


def candidate_costs(yhat_r, plans, L_future, lam):
    """Planner cost: lam*2.5u + 0.25(4v+8v^2), v=min(violation,11.5); predicted exit of [10,30] -> J_MAX onward."""
    yhat_r = np.asarray(yhat_r, dtype=np.float64)
    finite = bool(np.all(np.isfinite(yhat_r)))
    v = np.maximum(0.0, L_future[None, :] - yhat_r) + np.maximum(0.0, yhat_r - U_COMFORT)
    v = np.minimum(v, V_MAX)
    J = lam * 2.5 * plans + 0.25 * (4.0 * v + 8.0 * v * v)
    out = (yhat_r < SAFE_LO) | (yhat_r > SAFE_HI)
    J = np.where(np.cumsum(out, axis=1) > 0, J_MAX, J)
    return J.sum(axis=1), finite


def plan(yhat_r, plans, L_future, lam):
    costs, finite = candidate_costs(yhat_r, plans, L_future, lam)
    if not finite:
        return None, None
    return int(np.argmin(costs)), costs
