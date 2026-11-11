"""Learned models WM (GreyBox), GRU and RC_only plus training/adaptation (PLAN §2).
Imports only torch, numpy, math and the observable-only features module (Stage-0 static check).

All three arms consume the identical normalized inputs ctx (B,48,6) and fut (B,12,3) and are trained
with the identical loss: multi-step H=12 MSE on normalized y_r (CONFIG training.optimizer).

RC core (IMPL_NOTES D7 for the parameterization behind CONFIG's 7/11/13 counts):
  state z = [physical states..., b] (b = disturbance bias, AR(rho)); input w_t = [u_t, u_{t-1}, T_o_t, 1]
  z_{t+1} = A z_t + G w_t; y_t = z_t[0] (T_r); steady-state Kalman correction z_t += L (y_t - z_t[0]).
  3-state (T_r,T_w,q_h): a1 a2 a3 a4 b1 b2 c1 c2 rho L0..L3                  = 13
  2-state (T_r,T_w):     a1 a2 a3 a4 a5 b1 b2 rho L0..L2 (u, u_{t-1} on T_r)  = 11
  1-state (T_r lumped):  a2 a3 a4 a5 rho L0 L1                                 = 7
Over the horizon the unknown future T_o is held at the last measured T_o[k] (observable-only).
WM = 3-state RC core + GRU encoder-decoder residual on T_r (same GRUCore as the GRU arm; head zero-init).
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from . import features as Fe

CONTEXT = Fe.CONTEXT
H = Fe.H
GRAD_CLIP = 1.0

RC_INIT = {
    3: [0.05, 0.02, 0.1, 0.05, 0.01, 0.01, 0.5, 3.0, 0.9, 0.5, 0.05, 0.0, 0.05],
    2: [0.05, 0.02, 0.3, 0.05, 0.3, 0.01, 0.01, 0.9, 0.5, 0.05, 0.05],
    1: [0.02, 0.3, 0.05, 0.3, 0.9, 0.5, 0.05],
}
RC_NPARAM = {3: 13, 2: 11, 1: 7}


def rc_matrices(theta, nstate, xp):
    """Build (A, G, L) from theta. xp = torch or numpy. A (n,n), G (n,4), L (n,)."""
    t = theta
    if xp is torch:
        z = t.new_zeros(())
        one = t.new_ones(())
        st = torch.stack
    else:
        z, one = 0.0, 1.0
        st = lambda rows: np.array(rows, dtype=np.float64)
    if nstate == 3:
        a1, a2, a3, a4, b1, b2, c1, c2, rho = (t[i] for i in range(9))
        A = [[1 - a1 - a2, a1, a3, one], [b1, 1 - b1 - b2, z, z], [z, z, c1, z], [z, z, z, rho]]
        G = [[z, z, a2, a4], [z, z, b2, z], [c2, z, z, z], [z, z, z, z]]
        L = [t[9], t[10], t[11], t[12]]
    elif nstate == 2:
        a1, a2, a3, a4, a5, b1, b2, rho = (t[i] for i in range(8))
        A = [[1 - a1 - a2, a1, one], [b1, 1 - b1 - b2, z], [z, z, rho]]
        G = [[a3, a5, a2, a4], [z, z, b2, z], [z, z, z, z]]
        L = [t[8], t[9], t[10]]
    elif nstate == 1:
        a2, a3, a4, a5, rho = (t[i] for i in range(5))
        A = [[1 - a2, one], [z, rho]]
        G = [[a3, a5, a2, a4], [z, z, z, z]]
        L = [t[5], t[6]]
    else:
        raise ValueError(nstate)
    if xp is torch:
        A = st([st(r) for r in A])
        G = st([st(r) for r in G])
        L = st(L)
    else:
        A, G, L = st(A), st(G), np.array(L, dtype=np.float64)
    return A, G, L


def _powers_torch(M, n_pow):
    """[M^0, ..., M^{n_pow-1}] by doubling."""
    n = M.shape[0]
    P = torch.stack([torch.eye(n, dtype=M.dtype), M])
    while P.shape[0] < n_pow:
        Mm = P[-1] @ M                       # M^{len(P)}
        P = torch.cat([P, P @ Mm], 0)
    return P[:n_pow]


class RCCore(nn.Module):
    """RC core + steady-state Kalman filter over the 48-step context. Works in raw units internally
    using the replicate's normalization constants (buffers, not parameters)."""

    def __init__(self, nstate, norm):
        super().__init__()
        self.nstate = int(nstate)
        self.theta = nn.Parameter(torch.tensor(RC_INIT[self.nstate], dtype=torch.float32))
        m, s = norm["mean"], norm["sd"]
        self.register_buffer("nm", torch.tensor([m["y_r"], m["T_o"], m["u"]], dtype=torch.float32))
        self.register_buffer("ns", torch.tensor([s["y_r"], s["T_o"], s["u"]], dtype=torch.float32))

    def n_state(self):
        return self.nstate + 1

    def forward_raw(self, ctx, fut):
        """ctx (B,48,6), fut (B,12,3) normalized -> T_r predictions (B,12) in raw °C."""
        A, G, L = rc_matrices(self.theta, self.nstate, torch)
        n = A.shape[0]
        y = ctx[:, :, 0] * self.ns[0] + self.nm[0]
        To = ctx[:, :, 1] * self.ns[1] + self.nm[1]
        up = ctx[:, :, 3] * self.ns[2] + self.nm[2]          # u_{t-1} for t = k-47..k
        uf = fut[:, :, 0] * self.ns[2] + self.nm[2]          # u_k..u_{k+11}
        B = ctx.shape[0]
        # filter: z_0 = [y_0, (y_0), 0..., b=0]; z_i = Mz_{i-1} + N w_{i-1} + L y_i, i = 1..47
        z0 = ctx.new_zeros(B, n)
        z0[:, 0] = y[:, 0]
        if self.nstate >= 2:
            z0[:, 1] = y[:, 0]
        Ic = torch.eye(n, dtype=A.dtype) - torch.outer(L, torch.eye(n, dtype=A.dtype)[0])
        M = Ic @ A
        Nm = Ic @ G
        # w_{i-1} for i = 1..47: [u_{t_{i-1}} = up[:, i], u_{t_{i-1}-1} = up[:, i-1], To[:, i-1], 1]
        w = torch.stack([up[:, 1:], up[:, :-1], To[:, :-1], torch.ones_like(To[:, :-1])], -1)   # (B,47,4)
        g = w @ Nm.T + y[:, 1:, None] * L[None, None, :]                                       # (B,47,n)
        P = _powers_torch(M, CONTEXT)                                                            # (48,n,n)
        # z_47 = M^47 z0 + sum_{i=1}^{47} M^{47-i} g_i
        Prev = P[:CONTEXT - 1].flip(0)                                                           # M^{46}..M^{0}
        zk = z0 @ P[47].T + torch.einsum("ijk,bik->bj", Prev, g)
        # rollout with T_o held at T_o[k]
        To_k = To[:, -1]
        u_prev = up[:, -1]
        outs = []
        z = zk
        for j in range(H):
            wj = torch.stack([uf[:, j], u_prev, To_k, torch.ones_like(To_k)], -1)
            z = z @ A.T + wj @ G.T
            outs.append(z[:, 0])
            u_prev = uf[:, j]
        return torch.stack(outs, 1)

    def numpy_parts(self):
        A, G, L = rc_matrices(self.theta.detach().double().numpy(), self.nstate, np)
        n = A.shape[0]
        Ic = np.eye(n) - np.outer(L, np.eye(n)[0])
        M = Ic @ A
        P = [np.eye(n)]
        for _ in range(CONTEXT - 1):
            P.append(P[-1] @ M)
        P = np.stack(P)
        return dict(A=A, G=G, L=L, M=M, N=Ic @ G, P=P, nm=self.nm.double().numpy(), ns=self.ns.double().numpy(),
                    nstate=self.nstate)


def rc_predict_numpy(parts, ctx1, fut):
    """numpy RC prediction for one context (48,6) and M candidates (M,12,3) -> raw T_r (M,12)."""
    nm, ns = parts["nm"], parts["ns"]
    y = ctx1[:, 0] * ns[0] + nm[0]
    To = ctx1[:, 1] * ns[1] + nm[1]
    up = ctx1[:, 3] * ns[2] + nm[2]
    uf = fut[:, :, 0] * ns[2] + nm[2]
    A, G, L, Nm, P = parts["A"], parts["G"], parts["L"], parts["N"], parts["P"]
    n = A.shape[0]
    z0 = np.zeros(n)
    z0[0] = y[0]
    if parts["nstate"] >= 2:
        z0[1] = y[0]
    w = np.stack([up[1:], up[:-1], To[:-1], np.ones(CONTEXT - 1)], -1)
    g = w @ Nm.T + y[1:, None] * L[None, :]
    zk = P[47] @ z0 + np.einsum("ijk,ik->j", P[:CONTEXT - 1][::-1], g)     # M^{46}..M^{0}
    Mc = fut.shape[0]
    Z = np.repeat(zk[None, :], Mc, 0)
    u_prev = np.full(Mc, up[-1])
    out = np.empty((Mc, H))
    To_k = To[-1]
    for j in range(H):
        W = np.stack([uf[:, j], u_prev, np.full(Mc, To_k), np.ones(Mc)], -1)
        Z = Z @ A.T + W @ G.T
        out[:, j] = Z[:, 0]
        u_prev = uf[:, j]
    return out


class GRUCore(nn.Module):
    """Option1 GRU enc-dec: enc GRU(6->h), dec GRU(3->h) from encoder state, Linear(h->2) head."""

    def __init__(self, h):
        super().__init__()
        self.enc = nn.GRU(6, h, batch_first=True)
        self.dec = nn.GRU(3, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def heads(self, ctx, fut):
        _, hs = self.enc(ctx)
        if hs.shape[1] != fut.shape[0]:
            hs = hs.expand(1, fut.shape[0], hs.shape[-1]).contiguous()
        out, _ = self.dec(fut, hs)
        return self.head(out)


class GRUArm(nn.Module):
    kind = "GRU"

    def __init__(self, h, norm=None):
        super().__init__()
        self.core = GRUCore(h)

    def forward(self, ctx, fut):
        """Normalized y_r predictions (B,12) (increment head, channel 0)."""
        return ctx[:, -1:, 0] + self.core.heads(ctx, fut)[:, :, 0]


class RCArm(nn.Module):
    kind = "RC_only"

    def __init__(self, nstate, norm):
        super().__init__()
        self.rc = RCCore(nstate, norm)

    def forward(self, ctx, fut):
        return (self.rc.forward_raw(ctx, fut) - self.rc.nm[0]) / self.rc.ns[0]


class WMArm(nn.Module):
    kind = "WM"

    def __init__(self, h, norm):
        super().__init__()
        self.core = GRUCore(h)                   # built first: same default-init RNG draw as the GRU arm
        self.rc = RCCore(3, norm)
        nn.init.zeros_(self.core.head.weight)    # residual starts at zero (reference ew fit_rc, IMPL_NOTES D8)
        nn.init.zeros_(self.core.head.bias)

    def forward(self, ctx, fut):
        rc = (self.rc.forward_raw(ctx, fut) - self.rc.nm[0]) / self.rc.ns[0]
        return rc + self.core.heads(ctx, fut)[:, :, 0]


def build(arm, structure, norm, init_seed):
    torch.manual_seed(int(init_seed))
    if arm == "GRU":
        return GRUArm(int(structure))
    if arm == "WM":
        return WMArm(int(structure), norm)
    if arm == "RC_only":
        return RCArm(int(structure), norm)
    raise ValueError(arm)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def lr_cosine(t, n_updates, lr0, lr_min):
    return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * t / n_updates))


def params_finite(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def train(model, ctx, fut, tgt, batch_idx, n_updates, lr_fn, budget_check=None, log_every=100):
    """Adam (0.9,0.999,1e-8), grad clip 1.0, batch rows from batch_idx (n_updates,16); final weights.
    Loss: mean squared error over H=12 on normalized y_r (same for WM, GRU, RC_only)."""
    if batch_idx.shape[0] != n_updates:
        raise AssertionError("batch index shape")
    opt = torch.optim.Adam(model.parameters(), lr=lr_fn(0), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ctx_t = torch.as_tensor(ctx, dtype=torch.float32)
    fut_t = torch.as_tensor(fut, dtype=torch.float32)
    tgt_t = torch.as_tensor(tgt, dtype=torch.float32)
    info = dict(failed=False, fail_reason=None, fail_update=None, updates_done=0, train_loss_log=[])
    model.train()
    for t in range(n_updates):
        for gr in opt.param_groups:
            gr["lr"] = lr_fn(t)
        idx = torch.as_tensor(batch_idx[t], dtype=torch.long)
        loss = ((model(ctx_t[idx], fut_t[idx]) - tgt_t[idx]) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            info.update(failed=True, fail_reason="nonfinite_loss", fail_update=t)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if not params_finite(model):
            info.update(failed=True, fail_reason="nonfinite_parameter", fail_update=t)
            break
        info["updates_done"] = t + 1
        if log_every and (t + 1) % log_every == 0:
            info["train_loss_log"].append([t + 1, float(loss.detach())])
        if budget_check is not None:
            budget_check()
    model.eval()
    return model, info


@torch.no_grad()
def eval_loss(model, ctx, fut, tgt):
    model.eval()
    p = model(torch.as_tensor(ctx, dtype=torch.float32), torch.as_tensor(fut, dtype=torch.float32))
    return float(((p - torch.as_tensor(tgt, dtype=torch.float32)) ** 2).mean())


def clone(model):
    return copy.deepcopy(model)


class Predictor:
    """Normalized y_r predictions for one context and M candidates; numpy RC part for speed."""

    def __init__(self, model):
        self.model = model.eval()
        self.kind = model.kind
        self.parts = model.rc.numpy_parts() if hasattr(model, "rc") else None

    @torch.no_grad()
    def __call__(self, ctx1, fut):
        if self.kind == "GRU":
            c = torch.as_tensor(ctx1, dtype=torch.float32).unsqueeze(0)
            f = torch.as_tensor(fut, dtype=torch.float32)
            return (c[:, -1:, 0] + self.model.core.heads(c, f)[:, :, 0]).double().numpy()
        rc = (rc_predict_numpy(self.parts, ctx1, fut) - self.parts["nm"][0]) / self.parts["ns"][0]
        if self.kind == "RC_only":
            return rc
        c = torch.as_tensor(ctx1, dtype=torch.float32).unsqueeze(0)
        f = torch.as_tensor(fut, dtype=torch.float32)
        return rc + self.model.core.heads(c, f)[:, :, 0].double().numpy()
