"""LIN arm: port of Option1 thermal5/model_lin.py (deterministic subspace identification + finite-history
Kalman predictor, ridge rel 1e-6), with the state order as the structure parameter {3,5,8} (CONFIG arms.LIN;
IMPL_NOTES Q1: CONFIG calls it 'ARX ridge', Option1's actual LIN is this subspace/ridge model).
numpy float64 only; observable data only (y_r, T_o, commanded u, clock)."""
import numpy as np

from .features import clock

ORDER_DEFAULT = 5
T_MAX = 276                  # Option1 B1 erratum (LIN_T_MAX_ERRATUM)
P = F = 12
RIDGE_REL = 1e-6
RANK_TOL = 1e-8
R_TOL = 1e-10
CONTEXT = 48
H = 12


class LinFailure(Exception):
    """Intrinsic (counted) baseline failure: rank, R, or non-finite fit quantities."""


def _ridge(Y, Phi):
    """[coef] = Y Phiᵀ (Phi Phiᵀ + λI)⁻¹ with λ = 1e-6·trace(Phi Phiᵀ)/dim(Phi)."""
    G = Phi @ Phi.T
    lam = RIDGE_REL * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), Phi @ Y.T).T




def check_t_max(t_max, n_actions):
    """Pure index check (no data, no fit): W_f(t_max) = w_{t_max..t_max+11} needs u up to
    t_max+11 <= n_actions-1. Literal 277 fails for 288 actions (B1); erratum 276 passes."""
    if t_max + F - 1 > n_actions - 1:
        raise AssertionError("W_f(t) would need u beyond index %d" % (n_actions - 1))


def fit(obs, ns, t_max=T_MAX, order=ORDER_DEFAULT):
    ORDER = int(order)
    """obs: training observable set (E episodes). t_max: last stack column index.
    The protocol literal t_max = 277 needs u_288 which does not exist (KNOWN_ISSUES B1);
    the caller must pass an explicitly released resolution."""
    y_r = ns.n("y_r", obs["y_r"])
    T_o = ns.n("T_o", obs["T_o"])
    y = np.stack([y_r, T_o], axis=-1)                    # (E, 289, 2)
    u = ns.n("u", obs["u"])                              # (E, 288)
    E, Np1, _ = y.shape
    N = Np1 - 1
    check_t_max(t_max, N)
    ts = np.arange(P, t_max + 1)
    c_all = clock(np.arange(Np1))                        # (289, 2)
    w = np.concatenate([u[:, :, None], np.broadcast_to(c_all[None, :N, :], (E, N, 2))], -1)   # (E,288,3)
    zp, yf, wf, col_ep, col_t = [], [], [], [], []
    for e in range(E):
        for t in ts:
            zp.append(np.concatenate([y[e, t - P:t].ravel(), w[e, t - P:t].ravel()]))
            yf.append(y[e, t:t + F].ravel())
            wf.append(w[e, t:t + F].ravel())
            col_ep.append(e)
            col_t.append(t)
    Zp = np.array(zp).T          # (60, Ncol)
    Yf = np.array(yf).T          # (24, Ncol)
    Wf = np.array(wf).T          # (36, Ncol)
    col_ep = np.array(col_ep)
    col_t = np.array(col_t)
    Ncol = Zp.shape[1]
    # step 2
    Phi = np.vstack([Zp, Wf])
    Lfull = _ridge(Yf, Phi)
    Lp = Lfull[:, :Zp.shape[0]]
    # step 3
    Sigma = (Zp @ Zp.T) / Ncol
    ev, V = np.linalg.eigh(Sigma)
    ev = np.clip(ev, 0.0, None)
    Sig_half = (V * np.sqrt(ev)) @ V.T
    Mmat = Lp @ Sig_half
    U, s, _ = np.linalg.svd(Mmat, full_matrices=False)
    if not np.all(np.isfinite(s)):
        raise LinFailure("nonfinite_singular_values")
    if not (s[ORDER - 1] > RANK_TOL * s[0]):
        raise LinFailure("rank_check_failed s_order=%r s1=%r" % (float(s[ORDER - 1]), float(s[0])))
    Gpinv = np.diag(s[:ORDER] ** -0.5) @ U[:, :ORDER].T
    X = Gpinv @ Lp @ Zp          # (order, Ncol)
    # step 4: aligned pairs (t, t+1) within the same episode
    nxt = np.arange(1, Ncol)
    prv = np.arange(0, Ncol - 1)
    ok = (col_ep[prv] == col_ep[nxt]) & (col_t[nxt] == col_t[prv] + 1)
    prv, nxt = prv[ok], nxt[ok]
    Npairs = len(prv)
    e_idx, t_idx = col_ep[prv], col_t[prv]
    c_t = c_all[t_idx]                                    # (Npairs, 2)
    u_t = u[e_idx, t_idx]
    ones = np.ones(Npairs)
    Phi_s = np.vstack([X[:, prv], u_t[None, :], c_t.T, ones[None, :]])      # (9, Np)
    Phi_m = np.vstack([X[:, prv], c_t.T, ones[None, :]])                    # (8, Np) — no u column
    AB = _ridge(X[:, nxt], Phi_s)
    y_t = y[e_idx, t_idx].T                                                 # (2, Np)
    CD = _ridge(y_t, Phi_m)
    A, B = AB[:, :ORDER], AB[:, ORDER:]                  # B columns: u, sin, cos, 1
    Cm, D = CD[:, :ORDER], CD[:, ORDER:]                 # D columns: sin, cos, 1  (D_u = 0 exactly)
    v = X[:, nxt] - AB @ Phi_s
    er = y_t - CD @ Phi_m
    Q = v @ v.T / Npairs
    R = er @ er.T / Npairs
    S = v @ er.T / Npairs
    evR = np.linalg.eigvalsh(R)
    if not (evR.min() > R_TOL * np.trace(R)):
        raise LinFailure("R_not_positive_definite")
    m0 = X.mean(axis=1)
    Xc = X - m0[:, None]
    P0 = Xc @ Xc.T / Ncol
    model = dict(A=A, B=B, C=Cm, D=D, Q=Q, R=R, S=S, m0=m0, P0=P0)
    for k_, v_ in model.items():
        if not np.all(np.isfinite(v_)):
            raise LinFailure("nonfinite_%s" % k_)
    info = dict(order=ORDER, n_columns=int(Ncol), n_pairs=int(Npairs), singular_values=s.tolist(), t_max=int(t_max),
                eig_A=np.abs(np.linalg.eigvals(A)).tolist())
    return model, info


def predict(model, y_ctx, u_ctx, k, u_cand, with_var=False):
    """Finite-history Kalman predictor at origin k.
    y_ctx (48,2) normalized y_t for t = k-47..k; u_ctx (47,) normalized logged u_t for t = k-47..k-1;
    u_cand (M,12) normalized candidate u_k..u_{k+11}. Returns yhat (M,12,2) normalized
    [and y_r predictive variance (12,) normalized units], or raises LinFailure (intrinsic)."""
    A, B, Cm, D, Q, R, S = (model[n] for n in ("A", "B", "C", "D", "Q", "R", "S"))
    m = model["m0"].copy()
    Pm = model["P0"].copy()
    u_cand = np.atleast_2d(u_cand)
    M = u_cand.shape[0]
    ts = np.arange(k - CONTEXT + 1, k + 1)
    cks = clock(np.arange(k - CONTEXT + 1, k + H + 1))   # clock for k-47..k+12
    cb = lambda t: np.concatenate([cks[t - (k - CONTEXT + 1)], [1.0]])
    mc = None
    try:
        for i, t in enumerate(ts):
            Om = Cm @ Pm @ Cm.T + R
            r = y_ctx[i] - Cm @ m - D @ cb(t)
            G = np.linalg.solve(Om.T, (A @ Pm @ Cm.T + S).T).T
            base = A @ m + B[:, 1:] @ cb(t) + G @ r
            if t < k:
                m = base + B[:, 0] * u_ctx[i]
            else:
                mc = base[None, :] + np.outer(u_cand[:, 0], B[:, 0])          # (M,5), candidate u_k
            Pm = A @ Pm @ A.T + Q - G @ (Cm @ Pm @ A.T + S.T)
    except np.linalg.LinAlgError as ex:
        raise LinFailure("singular_Omega: %s" % ex)
    yhat = np.empty((M, H, 2))
    var = np.empty(H)
    yhat[:, 0, :] = mc @ Cm.T + D @ cb(k + 1)
    var[0] = (Cm @ Pm @ Cm.T + R)[0, 0]
    for j in range(2, H + 1):
        mc = mc @ A.T + np.outer(u_cand[:, j - 1], B[:, 0]) + (B[:, 1:] @ cb(k + j - 1))[None, :]
        Pm = A @ Pm @ A.T + Q
        yhat[:, j - 1, :] = mc @ Cm.T + D @ cb(k + j)
        var[j - 1] = (Cm @ Pm @ Cm.T + R)[0, 0]
    if not (np.all(np.isfinite(yhat)) and np.all(np.isfinite(var))):
        raise LinFailure("nonfinite_prediction")
    return (yhat, var) if with_var else yhat


def predict_loop(model, y_ctx, u_ctx, k, u_cand):
    """Reference per-candidate implementation for preflight equivalence checks."""
    return np.concatenate([predict(model, y_ctx, u_ctx, k, u_cand[i:i + 1]) for i in range(len(u_cand))], 0)
