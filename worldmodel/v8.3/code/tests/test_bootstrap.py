import numpy as np
import pytest

from v81 import spec as S
from v81 import stats as ST


def test_ci_uses_0based_249_9750():
    x = np.random.default_rng(0).permutation(np.arange(10000)).astype(float)
    lo, hi = ST.ci(x)
    assert lo == 249.0 and hi == 9750.0
    assert np.sum(x < lo) == 249 and np.sum(x > hi) == 249     # symmetric


def test_ci_requires_B_10000():
    with pytest.raises(AssertionError):
        ST.ci(np.arange(9999.0))


def test_boot_matrix_shape_dtype_determinism():
    a = ST.boot_matrix(S.BOOT_SEEDS["primary"], 40)
    b = ST.boot_matrix(S.BOOT_SEEDS["primary"], 40)
    assert a.shape == (10000, 40) and a.dtype == np.int16
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 39
    assert ST.boot_matrix(S.BOOT_SEEDS["adapt"], 10).shape == (10000, 10)


def test_cluster_bootstrap_matches_loop():
    rng = np.random.default_rng(3)
    cl = np.repeat(np.arange(40), 8)
    Ew, Ec, Dw, Dc = (rng.uniform(1, 5, 320) for _ in range(4))
    idx = ST.boot_matrix(S.BOOT_SEEDS["primary"], 40)
    Sb, dDb, _ = ST.boot_S_dD(Ew, Ec, Dw, Dc, cl, idx)
    for b in (0, 17, 9999):
        sel = np.concatenate([np.where(cl == c)[0] for c in idx[b]])
        assert np.isclose(Sb[b], 1 - Ew[sel].sum() / Ec[sel].sum())
        assert np.isclose(dDb[b], np.mean(Dw[sel] - Dc[sel]))
