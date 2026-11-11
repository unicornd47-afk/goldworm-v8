import importlib
import sys

import numpy as np
import torch

from v81 import model_lin
from v81 import models_torch as MT
from v81 import planner as PL
from v81 import plant_iv as P
from v81 import seeds as SD
from v81 import spec as S


def _option1():
    import os
    import pytest
    if not os.path.isdir(S.OPTION1_DIR):          # v8.3 portability: v7 reference tree may not exist on another host
        pytest.skip("Option1 reference not present (set WM_OPTION1_DIR)")
    sys.dont_write_bytecode = True
    if S.OPTION1_DIR not in sys.path:
        sys.path.insert(0, S.OPTION1_DIR)
    return importlib.import_module("thermal5.plant"), importlib.import_module("thermal5.rng"), \
        importlib.import_module("thermal5.planner"), importlib.import_module("thermal5.model_lin")


def test_nominal_plant_equals_option1():
    op, orng, oplan, olin = _option1()
    d1 = P.draw_episode(np.random.default_rng(5), 192)
    d2 = op.draw_episode(np.random.default_rng(5), 192)
    assert d1.mu_o == d2.mu_o and np.array_equal(d1.nu, d2.nu)
    x = np.array([18.0, 17.0, 2.0, 0.1, -0.3])
    for k in range(0, 150, 7):
        u = (k % 11) / 10
        a = P.step(x, u, k, d1, P.NOMINAL)
        b = op.step(x, u, k, d2.mu_o, d2.A_o, d2.eps_d[k], d2.eps_o[k])
        assert np.allclose(a, b, rtol=0, atol=1e-12)
        assert np.allclose(P.measure(x, k, d1, P.NOMINAL), op.measure(x, k, d2))


def test_library_equals_option1_child_key():
    op, orng, oplan, olin = _option1()
    try:
        orng.set_mode("production")
    except orng.StreamModeError:
        pass
    for eid, k in ((SD.episode_id(7000, "N0"), 96), (SD.episode_id(9139, "H4b"), 191)):
        assert np.array_equal(PL.library(eid, k), oplan.library(eid, k))


def test_lin_order5_equals_option1():
    op, orng, oplan, olin = _option1()
    from v81 import features as Fe
    G = SD.SeedGuard("smoke")
    obs = P.stack_obs([P.data_episode(s, i, "N0", None, G, "train")[0] for i, s in enumerate((1, 2, 3, 4))])
    ns = Fe.NormStats.from_train(obs)
    m1, _ = model_lin.fit(obs, ns, order=5)
    m2, _ = olin.fit(obs, ns, 276)
    for key in m1:
        assert np.allclose(m1[key], m2[key])
    m3, i3 = model_lin.fit(obs, ns, order=3)
    assert m3["A"].shape == (3, 3) and i3["order"] == 3


def test_interventions_semantics():
    p = P.override_params("H2b")
    assert np.isclose(sum(p.A_TR[:3]), 1.0) and p.A_TR[2] == 0.0375
    assert P.override_params("H3a").A_TW == (0.00625, 0.990625, 0.003125)
    assert P.override_params("H1b").A_QH == (0.6, 2.4)
    assert P.override_params("T3a").A_QH == (0.45, 5.5)
    assert P.override_params("T1b").D_BAR == 1.5
    sch = P.Schedule("H4b", 120)
    assert not sch.active(119) and sch.active(120) and sch.active(167) and not sch.active(168)
    u = np.arange(200) / 200.0
    assert P.applied_u(u, 130, sch.params(130)) == u[126]
    assert P.applied_u(u, 170, sch.params(170)) == u[170]
    d = P.draw_episode(np.random.default_rng(1), 192)
    x = np.array([18.0, 17.0, 2.0, 0.0, 0.0])
    assert np.isclose(P.measure(x, 130, d, P.override_params("T2a"))[1] - P.measure(x, 130, d, P.NOMINAL)[1], -2.0)


def test_rc_closed_form_filter_equals_sequential():
    norm = {"mean": {"y_r": 19.0, "T_o": 4.0, "s": 19.0, "u": 0.4}, "sd": {"y_r": 2.0, "T_o": 4.0, "s": 2.0, "u": 0.3}}
    rng = np.random.default_rng(0)
    ctx = rng.standard_normal((48, 6))
    fut = rng.standard_normal((3, 12, 3))
    for st in (1, 2, 3):
        m = MT.build("RC_only", st, norm, 0)
        parts = m.rc.numpy_parts()
        A, G, L = parts["A"], parts["G"], parts["L"]
        y = ctx[:, 0] * 2 + 19
        To = ctx[:, 1] * 4 + 4
        up = ctx[:, 3] * 0.3 + 0.4
        z = np.zeros(A.shape[0])
        z[0] = y[0]
        if st >= 2:
            z[1] = y[0]
        for i in range(1, 48):
            z = A @ z + G @ np.array([up[i], up[i - 1], To[i - 1], 1.0])
            z = z + L * (y[i] - z[0])
        uf = fut[0, :, 0] * 0.3 + 0.4
        out = []
        prev = up[-1]
        for j in range(12):
            z = A @ z + G @ np.array([uf[j], prev, To[-1], 1.0])
            out.append(z[0])
            prev = uf[j]
        ref = np.array(out)
        got = MT.rc_predict_numpy(parts, ctx, fut)[0]
        assert np.allclose(got, ref, atol=1e-9)
        with torch.no_grad():
            tt = m.rc.forward_raw(torch.tensor(ctx[None], dtype=torch.float32),
                                  torch.tensor(fut[:1], dtype=torch.float32)).numpy()[0]
        assert np.allclose(tt, ref, atol=1e-3)


def test_same_inputs_all_arms():
    """All learned arms are called with the identical (ctx (B,48,6), fut (B,12,3)) tensors."""
    norm = {"mean": {"y_r": 19.0, "T_o": 4.0, "s": 19.0, "u": 0.4}, "sd": {"y_r": 2.0, "T_o": 4.0, "s": 2.0, "u": 0.3}}
    ctx = torch.zeros(2, 48, 6)
    fut = torch.zeros(2, 12, 3)
    for arm, st in (("WM", 32), ("GRU", 32), ("RC_only", 3)):
        assert MT.build(arm, st, norm, 0)(ctx, fut).shape == (2, 12)
