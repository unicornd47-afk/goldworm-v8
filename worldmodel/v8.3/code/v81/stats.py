"""Endpoints P1-P4, bootstrap (B=10000, 0-based order statistics [249]/[9750]), selection, gates.
v8.2: energy-targeted selection under a comfort bound (PLAN §8), MSE contrast CIs (PLAN §9)."""
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


def _blocks(n, check):
    """v8.3 C6b: row blocks of S.BOOT_BLOCK (1000) resamples; check() after every block. Rows are independent,
    so block-wise evaluation gives bit-identical values to the single vectorised pass."""
    for a in range(0, n, S.BOOT_BLOCK):
        yield slice(a, min(n, a + S.BOOT_BLOCK))
        if check is not None:
            check()


def boot_S_dD(E_wm, E_comp, D_wm, D_comp, cluster_of, idx, check=None):
    """Paired cluster bootstrap. E/D: per-episode values (mean over replicates). idx (B, n_clusters).
    S* = 1 - sum E_wm / sum E_comp over resampled clusters; dD* = mean(D_wm - D_comp) over resampled episodes.
    v8.3: evaluated in blocks of 1000 resamples with check() after each block (C6b)."""
    n_cl = idx.shape[1]
    cE_w = cluster_sums(E_wm, cluster_of, n_cl)
    cE_c = cluster_sums(E_comp, cluster_of, n_cl)
    cD = cluster_sums(np.asarray(D_wm) - np.asarray(D_comp), cluster_of, n_cl)
    cN = cluster_sums(np.ones(len(E_wm)), cluster_of, n_cl)
    ii_all = idx.astype(np.int64)
    S_b = np.full(len(ii_all), -np.inf)
    dD_b = np.empty(len(ii_all))
    nzero = 0
    for sl in _blocks(len(ii_all), check):
        ii = ii_all[sl]
        num, den = cE_w[ii].sum(1), cE_c[ii].sum(1)
        nz = den != 0
        sb = np.full(len(ii), -np.inf)
        sb[nz] = 1.0 - num[nz] / den[nz]
        S_b[sl] = sb
        dD_b[sl] = cD[ii].sum(1) / cN[ii].sum(1)
        nzero += int((~nz).sum())
    return S_b, dD_b, nzero


def primary_endpoints(E_wm, E_comp, D_wm, D_comp, cluster_of, idx, safety_wm, safety_comp, check=None):
    S_hat = saving(E_wm, E_comp)
    dD_hat = float(np.mean(np.asarray(D_wm) - np.asarray(D_comp)))
    S_b, dD_b, nzero = boot_S_dD(E_wm, E_comp, D_wm, D_comp, cluster_of, idx, check=check)
    S_lo, S_hi = ci(S_b)
    dD_lo, dD_hi = ci(dD_b)
    P1 = S_hat >= S.P1_MIN_SAVING
    P2 = S_lo > 0.0
    P3 = dD_hi < S.P3_MARGIN
    P4 = not (int(safety_wm) > int(safety_comp))
    return dict(S=S_hat, S_ci=[S_lo, S_hi], dD=dD_hat, dD_ci=[dD_lo, dD_hi], zero_den_resamples=nzero,
                safety_exits_WM=int(safety_wm), safety_exits_comp=int(safety_comp),
                P1=bool(P1), P2=bool(P2), P3=bool(P3), P4=bool(P4))


def fallback_flags(selected, comparator):
    """ERRATUM_v8.2_E1 §E1.6 (flag-only): which selections took the infeasible (lowest-D) fallback path.
    Reported only; the run stays valid and the outcome is decided by P1-P4 as predeclared."""
    return dict(WM_fallback=not selected["WM"]["feasible"], comparator_fallback=not selected[comparator]["feasible"],
                arms_fallback=[a for a in S.ARMS if not selected[a]["feasible"]])


def outcome(primary, validity_ok, launched=True, start_ok=True):
    """PLAN §11. Precedence: NOT_LAUNCHED > INVALID > ADVANCE/NO_ADVANCE."""
    if not (launched and start_ok):
        return "NOT_LAUNCHED"
    if not validity_ok:
        return "INVALID"
    if primary["P1"] and primary["P2"] and primary["P3"] and primary["P4"]:
        return "ADVANCE"
    return "NO_ADVANCE"


def struct_rank(arm, structure):
    return list(S.structures(arm)).index(structure)


def comfort_bound(table, bound=None):
    """Numeric comfort bound for the selection (K h/day): ONLY the absolute D_max from CONFIG (S.COMFORT_BOUND).
    Any other form raises SpecError. Returns (value, record). `table` is unused (bound is fixed before data)."""
    b = dict(S.COMFORT_BOUND if bound is None else bound)
    if set(b) != {"form", "D_max"} or b["form"] != "absolute":
        raise S.SpecError("comfort bound must be the absolute D_max form, got %r" % (b,))
    return float(b["D_max"]), dict(b, value=float(b["D_max"]))


def select_within_arm(arm, table, d_bound):
    """v8.2 PLAN §8. table: {(structure, lam): {"mean_E", "mean_D", ...}} (means over 3 reps x n episodes).
    F = {c: mean_D <= d_bound}. F non-empty: argmin mean_E over F; ties lower D, smaller structure, smaller lambda.
    F empty: argmin mean_D over all; ties lower E, smaller structure, smaller lambda. Returns (config, feasible)."""
    rank = lambda c: (struct_rank(arm, c[0]), S.LAMBDAS.index(c[1]))
    F = [c for c in table if table[c]["mean_D"] <= d_bound]
    if F:
        return min(F, key=lambda c: (table[c]["mean_E"], table[c]["mean_D"]) + rank(c)), True
    return min(table, key=lambda c: (table[c]["mean_D"], table[c]["mean_E"]) + rank(c)), False


def select_comparator(sel):
    """sel: {arm: {"mean_E", "mean_D", "feasible"}} for GRU, LIN, RC_only (their selected configs).
    Feasible arms rank before infeasible ones; feasible: lowest E, ties lower D; none feasible: lowest D, ties
    lower E; remaining ties: order RC_only, LIN, GRU."""
    def key(a):
        x = sel[a]
        m = (x["mean_E"], x["mean_D"]) if x["feasible"] else (x["mean_D"], x["mean_E"])
        return (0 if x["feasible"] else 1,) + m + (S.COMPARATOR_ARMS_TIEBREAK.index(a),)
    return min(S.COMPARATOR_ARMS_TIEBREAK, key=key)


MSE_CONTRASTS = (("WM", "GRU"), ("WM", "RC_only"), ("WM", "LIN"), ("GRU", "RC_only"), ("GRU", "LIN"),
                 ("RC_only", "LIN"))


def mse_contrast_ci(per_seed, idx, check=None):
    """v8.2 secondary (CONFIG endpoints.secondary_mse_ci). per_seed: {arm: (n_clusters,) per-test-seed mean
    12-step MSE (NaN = no prediction for that seed)}. For each contrast a-b: d = per_seed[a] - per_seed[b];
    point = mean(d); bootstrap value b = mean(d[idx[b]]) over the resampled seed clusters; CI by ci().
    A contrast with any NaN seed is reported with ci = None (no ranking claim possible)."""
    out = {}
    ii = idx.astype(np.int64)
    for a, b in MSE_CONTRASTS:
        key = "%s-%s" % (a, b)
        if per_seed.get(a) is None or per_seed.get(b) is None:
            out[key] = dict(diff=None, ci=None, n_nan_seeds=None)
            continue
        d = np.asarray(per_seed[a], dtype=np.float64) - np.asarray(per_seed[b], dtype=np.float64)
        nn = int(np.isnan(d).sum())
        if nn:
            out[key] = dict(diff=float(np.nanmean(d)) if nn < len(d) else None, ci=None, n_nan_seeds=nn)
            continue
        bvals = np.empty(len(ii))
        for sl in _blocks(len(ii), check):                  # v8.3 C6b: blocks of 1000 resamples
            bvals[sl] = d[ii[sl]].mean(1)
        out[key] = dict(diff=float(d.mean()), ci=list(ci(bvals)), n_nan_seeds=0)
    return out
