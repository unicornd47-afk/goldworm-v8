"""Endpoints P1-P4, bootstrap (B=10000, 0-based order statistics [249]/[9750]), selection, gates."""
import numpy as np

from . import spec as S


def boot_matrix(seed, n_clusters, B=S.N_BOOT):
    """(B, n_clusters) int16 cluster-index matrix from default_rng(seed) (IMPL_NOTES D9)."""
    return np.random.default_rng(int(seed)).integers(0, n_clusters, size=(B, n_clusters)).astype(np.int16)


def ci(x):
    """95% percentile CI: lower = np.sort(x)[249], upper = np.sort(x)[9750] (0-based, no interpolation)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) != S.N_BOOT:
        raise AssertionError("expected %d bootstrap values" % S.N_BOOT)
    v = np.sort(x, kind="mergesort")
    return float(v[S.BOOT_LO_IDX]), float(v[S.BOOT_HI_IDX])


def saving(E_wm, E_comp):
    return 1.0 - float(np.sum(E_wm)) / float(np.sum(E_comp))


def cluster_sums(values, cluster_of, n_clusters):
    out = np.zeros(n_clusters)
    np.add.at(out, np.asarray(cluster_of), np.asarray(values, dtype=np.float64))
    return out


def boot_S_dD(E_wm, E_comp, D_wm, D_comp, cluster_of, idx):
    """Paired cluster bootstrap. E/D: per-episode values (mean over replicates). idx (B, n_clusters).
    S* = 1 - sum E_wm / sum E_comp over resampled clusters; dD* = mean(D_wm - D_comp) over resampled episodes."""
    n_cl = idx.shape[1]
    cE_w = cluster_sums(E_wm, cluster_of, n_cl)
    cE_c = cluster_sums(E_comp, cluster_of, n_cl)
    cD = cluster_sums(np.asarray(D_wm) - np.asarray(D_comp), cluster_of, n_cl)
    cN = cluster_sums(np.ones(len(E_wm)), cluster_of, n_cl)
    ii = idx.astype(np.int64)
    num, den = cE_w[ii].sum(1), cE_c[ii].sum(1)
    S_b = np.full(len(ii), -np.inf)
    nz = den != 0
    S_b[nz] = 1.0 - num[nz] / den[nz]
    dD_b = cD[ii].sum(1) / cN[ii].sum(1)
    return S_b, dD_b, int((~nz).sum())


def primary_endpoints(E_wm, E_comp, D_wm, D_comp, cluster_of, idx, safety_wm, safety_comp):
    S_hat = saving(E_wm, E_comp)
    dD_hat = float(np.mean(np.asarray(D_wm) - np.asarray(D_comp)))
    S_b, dD_b, nzero = boot_S_dD(E_wm, E_comp, D_wm, D_comp, cluster_of, idx)
    S_lo, S_hi = ci(S_b)
    dD_lo, dD_hi = ci(dD_b)
    P1 = S_hat >= S.P1_MIN_SAVING
    P2 = S_lo > 0.0
    P3 = dD_hi < S.P3_MARGIN
    P4 = not (int(safety_wm) > int(safety_comp))
    return dict(S=S_hat, S_ci=[S_lo, S_hi], dD=dD_hat, dD_ci=[dD_lo, dD_hi], zero_den_resamples=nzero,
                safety_exits_WM=int(safety_wm), safety_exits_comp=int(safety_comp),
                P1=bool(P1), P2=bool(P2), P3=bool(P3), P4=bool(P4))


def outcome(primary, validity_ok, launched=True, start_ok=True):
    """PLAN §11."""
    if not (launched and start_ok):
        return "NOT_LAUNCHED"
    if not validity_ok:
        return "INVALID"
    if primary["P1"] and primary["P2"] and primary["P3"] and primary["P4"]:
        return "ADVANCE"
    return "NO_ADVANCE"


def struct_rank(arm, structure):
    return list(S.structures(arm)).index(structure)


def select_within_arm(arm, table):
    """table: {(structure, lam): (mean_J, mean_E)} over 3 reps x n episodes. Lowest J; ties: lower E,
    smaller structure, lam = 1.0 (then the remaining lambda order is the grid order)."""
    def key(c):
        s, lam = c
        J, E = table[c]
        return (J, E, struct_rank(arm, s), 0 if lam == 1.0 else 1, S.LAMBDAS.index(lam))
    return min(table, key=key)


def select_comparator(sel):
    """sel: {arm: (config, mean_J, mean_E)} for GRU, LIN, RC_only. Ties: lower E, then RC_only, LIN, GRU."""
    arms = [a for a in S.COMPARATOR_ARMS_TIEBREAK]
    return min(arms, key=lambda a: (sel[a][1], sel[a][2], S.COMPARATOR_ARMS_TIEBREAK.index(a)))
