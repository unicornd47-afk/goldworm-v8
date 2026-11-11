"""Observable-only predictors used by the MPC. Every predictor has the identical signature
(view: ObservableView, plans (M,12)) -> raw y_r (M,12) or None (non-finite). Identical-input gate:
the ObservableView carries exactly y_r, T_o, commanded u, public schedule and k (PLAN §3)."""
import numpy as np

from . import features as Fe
from . import model_lin
from . import models_torch as MT


class NeuralPredictor:
    def __init__(self, model, ns):
        self.p = MT.Predictor(model)
        self.ns = ns
        self.kind = model.kind

    def __call__(self, view, plans):
        if type(view) is not Fe.ObservableView:
            raise AssertionError("identical-input violation")
        ctx = Fe.ctx_from_view(view, self.ns)
        fut = Fe.fut_inputs(plans, view.k, self.ns)
        z = self.p(ctx, fut)
        if not np.all(np.isfinite(z)):
            return None
        return self.ns.denorm("y_r", z)


class LinPredictor:
    kind = "LIN"

    def __init__(self, model, ns):
        self.model, self.ns = model, ns

    def __call__(self, view, plans):
        if type(view) is not Fe.ObservableView:
            raise AssertionError("identical-input violation")
        k = view.k
        t = Fe.context_indices(k)
        y_ctx = np.stack([self.ns.n("y_r", view.y_r[t]), self.ns.n("T_o", view.T_o[t])], -1)
        u_ctx = self.ns.n("u", view.u[k - 47:k])
        try:
            z = model_lin.predict(self.model, y_ctx, u_ctx, k, self.ns.n("u", plans))
        except model_lin.LinFailure:
            return None
        return self.ns.denorm("y_r", z[:, :, 0])


def make(arm, model, ns):
    if model is None:
        return None              # failed fit -> PI fallback for every decision (flagged)
    return LinPredictor(model, ns) if arm == "LIN" else NeuralPredictor(model, ns)
