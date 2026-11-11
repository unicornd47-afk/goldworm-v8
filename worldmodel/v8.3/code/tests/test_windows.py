import numpy as np

from v81 import closedloop as CL
from v81 import planner as PL
from v81 import plant_iv as P
from v81 import seeds as SD
from v81 import spec as S


def test_window_definitions():
    a = np.arange(S.N_CONTROL, dtype=float)       # value = k - 96
    assert CL.window_sum(np.ones(96), S.WIN_PRIMARY) == 96
    assert CL.window_sum(np.ones(96), S.WIN_SECONDARY) == 72
    assert CL.window_sum(a, S.WIN_SECONDARY) == sum(range(24, 96))
    assert S.WIN_PRIMARY == (96, 191) and S.WIN_SECONDARY == (120, 191)


def test_episode_metrics_consistent_and_windows_identical_across_arms():
    G = SD.SeedGuard("smoke")
    sc = P.Scenario(1, 0, "T1a", G, "tuning")
    r_pi = CL.run_episode(sc, None, 1.0, mode="pi")
    r_or = CL.run_episode(sc, None, 1.0, mode="oracle")
    for r in (r_pi, r_or):
        assert len(r["E_step"]) == 96
        assert np.isclose(r["E"], r["E_step"].sum()) and np.isclose(r["E2"], r["E_step"][24:].sum())
        assert np.isclose(r["D"], r["d_step"].sum())
        assert np.allclose(r["J_step"], 2.5 * r["traj"]["u"][96:192] + 0.25 * (4 * r["d_step"] / 0.25
                                                                              + 8 * (r["d_step"] / 0.25) ** 2))
    # E is charged on the commanded u
    assert np.allclose(r_pi["E_step"], 2.5 * r_pi["traj"]["u"][96:192])


def test_safety_exit_charges_remaining_steps():
    class Hot:
        kind = "x"

        def __call__(self, view, plans):
            return np.full(plans.shape, 20.0) - 5 * plans     # predicts heating cools -> always u=1
    G = SD.SeedGuard("smoke")
    sc = P.Scenario(2, 3, "N0", G, "tuning")
    r = CL.run_episode(sc, Hot(), 1.0)
    if r["safety_exit"]:
        i = r["safety_k"] - 96
        assert np.all(r["E_step"][i:] == 2.5) and np.all(r["d_step"][i:] == PL.D_STEP_MAX)


def test_nonfinite_prediction_falls_back_to_pi():
    class Bad:
        kind = "x"

        def __call__(self, view, plans):
            return None if view.k >= 130 else np.full(plans.shape, 21.0)
    G = SD.SeedGuard("smoke")
    sc = P.Scenario(1, 0, "N0", G, "tuning")
    r = CL.run_episode(sc, Bad(), 1.0)
    assert r["fallback"] and r["fallback_k"] == 130 and r["n_mpc_decisions"] == 35
    assert len(r["E_step"]) == 96 and np.all(np.isfinite(r["E_step"]))


def test_settling_is_log_only():
    G = SD.SeedGuard("smoke")
    sc = P.Scenario(1, 0, "N0", G, "tuning")
    r = CL.run_episode(sc, None, 1.0, mode="pi")
    assert r["settling_k"] is None or r["settling_k"] >= 120
    assert np.isclose(r["E"], r["E_step"].sum())        # window never depends on settling
