from v81 import models_torch as MT
from v81 import spec as S

NORM = {"mean": {"y_r": 19.0, "T_o": 4.0, "s": 19.0, "u": 0.4}, "sd": {"y_r": 2.0, "T_o": 4.0, "s": 2.0, "u": 0.3}}


def test_gru_wm_counts_match_config():
    for h, pg, pw in zip(S.GRU_HIDDEN, S.GRU_PARAMS, S.WM_PARAMS):
        g = MT.n_params(MT.build("GRU", h, NORM, 3101))
        w = MT.n_params(MT.build("WM", h, NORM, 3101))
        assert g == pg and w == pw
        assert w - g == 13
        assert abs(w - g) / g < 0.002          # matched within 0.2 %


def test_rc_counts_match_config():
    for st, p in zip(S.RC_STRUCTS, S.RC_PARAMS):
        assert MT.n_params(MT.build("RC_only", st, NORM, 1)) == p == MT.RC_NPARAM[st]


def test_wm_residual_same_architecture_as_gru():
    for h in S.GRU_HIDDEN:
        g = MT.build("GRU", h, NORM, 3101)
        w = MT.build("WM", h, NORM, 3101)
        assert [p.shape for p in g.core.parameters()] == [p.shape for p in w.core.parameters()]
        assert w.rc.theta.numel() == 13
