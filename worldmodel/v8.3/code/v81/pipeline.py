"""Full-run orchestration (PLAN_v8.2 §2-§11). The same code runs the production Design and the toy smoke
Design (seeds 1-4, tiny counts) used only by tests; preflight measures unit costs separately."""
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import closedloop as CL
from . import features as Fe
from . import model_lin
from . import models_torch as MT
from . import plant_iv as P
from . import predictors as PR
from . import seeds as SD
from . import spec as S
from . import stage0
from . import stats as ST
from . import warmup as WU
from .budget import Budget
from .ledger import Ledger, canon, write_once


@dataclass
class Design:
    mode: str = "production"
    train_seeds: dict = field(default_factory=lambda: {r: list(S.TRAIN_SEEDS[r]) for r in range(S.REPLICATES)})
    val_seeds: dict = field(default_factory=lambda: {r: list(S.VAL_SEEDS[r]) for r in range(S.REPLICATES)})
    init_seeds: dict = field(default_factory=lambda: dict(S.INIT_SEEDS))
    tuning_seeds: list = field(default_factory=lambda: list(S.TUNING_SEEDS))
    test_seeds: list = field(default_factory=lambda: list(S.TEST_SEEDS))
    adapt_seeds: list = field(default_factory=lambda: list(S.ADAPT_SEEDS))
    adapt_eval_n: int = S.ADAPT_EVAL_SEEDS_N
    adapt_K: tuple = S.ADAPT_K
    n_updates: int = S.N_UPDATES
    adapt_updates: int = S.ADAPT_UPDATES
    structures: dict = field(default_factory=lambda: {a: tuple(S.structures(a)) for a in S.ARMS})
    lambdas: tuple = S.LAMBDAS
    include_in_dist: bool = True
    ladder_applied: tuple = ()
    caps: dict = field(default_factory=lambda: dict(S.CAPS))
    global_cap: float = S.GLOBAL_CAP
    k1_cap: float = S.K1_GLOBAL_CAP          # v8.3 K1: explicit 3600 CPU-s abort, independent of the phase caps
    warmup_updates: int = S.WARMUP_UPDATES   # ERRATUM E1.1/E2.1: once per process start, before the first neural fit

    def apply_ladder(self, steps):
        for st in steps:
            if st == "a":
                self.adapt_eval_n = 5
            elif st == "b":
                self.include_in_dist = False
            elif st == "c":
                self.tuning_seeds = list(S.TUNING_SEEDS[:15])
            else:
                raise ValueError(st)
        self.ladder_applied = tuple(steps)
        return self


def _sha(path):
    return S.sha256_file(path)


class Run:
    def __init__(self, design, out_dir, release_hashes=None):
        self.D = design
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=False)
        self.L = Ledger(out_dir)
        # v8.3 C6: phase_start/phase_end events with process CPU-s, wall time and thread readout -> LEDGER
        from .hostenv import thread_readout
        self.B = Budget(design.caps, design.global_cap, t0=0.0 if design.mode == "production" else None,
                        k1_cap=design.k1_cap, on_event=self.L.append, thread_readout=thread_readout)
        self.G = SD.SeedGuard(design.mode)
        self.release_hashes = release_hashes
        self.logs_pre_test = []          # training/tuning/selection records (HELDOUT scan)
        self.results = {}
        self.validity = {}
        self.input_signatures = set()

    # ---------------- helpers ----------------
    def save_json(self, name, obj, once=False):
        path = os.path.join(self.out, name)
        if once:                                   # v8.3 C7: exclusive create
            write_once(path, canon(obj))
            return _sha(path)
        with open(path, "w") as f:
            f.write(canon(obj))
        return _sha(path)

    def _scenarios(self, seeds, entries_fn, purpose):
        out = []
        for i, s in enumerate(seeds):
            for iid in entries_fn(i):
                out.append(P.Scenario(s, i, iid, self.G, purpose))
        return out

    def _predictor(self, arm, rep, structure):
        m = self.models[(arm, structure, rep)]
        pr = PR.make(arm, m, self.ns[rep])
        if pr is not None:
            self.input_signatures.add(type(pr).__call__.__code__.co_varnames[:3])
        return pr

    def _run_set(self, scens, pred, lam, mode="mpc"):
        rows = []
        for sc in scens:
            r = CL.run_episode(sc, pred, lam, mode=mode)
            rows.append(r)
            self.B.check()                         # v8.3 C6b: cap check after every episode
        return rows

    @staticmethod
    def _summ(rows):
        keys = ("E", "D", "J", "E2", "D2", "J2")
        out = {k: np.array([r[k] for r in rows]) for k in keys}
        out["safety"] = np.array([r["safety_exit"] for r in rows])
        out["fallback"] = np.array([r["fallback"] for r in rows])
        out["settling"] = np.array([-1 if r["settling_k"] is None else r["settling_k"] for r in rows])
        out["decisions"] = int(sum(r["n_mpc_decisions"] for r in rows))
        return out

    # ---------------- phases ----------------
    def phase_data(self):
        with self.B.phase("data"):
            s0 = stage0.model_import_check()
            self.validity["stage0_model_imports"] = s0["valid"]
            self.train, self.val, self.ns = {}, {}, {}
            for r in range(S.REPLICATES):
                tr, vm = [], []
                comp = SD.train_composition(len(self.D.train_seeds[r]))
                for i, s in enumerate(self.D.train_seeds[r]):
                    o, m = P.data_episode(s, i, comp[i][0], comp[i][1], self.G, "train")
                    self.B.check()                 # v8.3 C6b: after every episode
                    tr.append(o)
                    self.logs_pre_test.append(m)
                vcomp = SD.val_composition(len(self.D.val_seeds[r]))
                va = []
                for i, s in enumerate(self.D.val_seeds[r]):
                    o, m = P.data_episode(s, i, vcomp[i][0], vcomp[i][1], self.G, "val")
                    self.B.check()
                    va.append(o)
                    self.logs_pre_test.append(m)
                self.train[r] = P.stack_obs(tr)
                self.val[r] = P.stack_obs(va)
                self.ns[r] = Fe.NormStats.from_train(self.train[r])
            self.save_json("normalization.json", {r: self.ns[r].to_dict() for r in self.ns}, once=True)

    def phase_fits(self):
        self.models, self.fit_info = {}, {}
        self.warmed_up = False
        with self.B.phase("fits"):
            for r in range(S.REPLICATES):
                ns = self.ns[r]
                n_ep = self.train[r]["u"].shape[0]
                pairs = Fe.train_pairs(n_ep)
                ctx, fut, tgt = Fe.window_arrays(self.train[r], pairs, ns)
                vctx, vfut, vtgt = Fe.window_arrays(self.val[r], Fe.train_pairs(self.val[r]["u"].shape[0]), ns)
                bidx = np.random.default_rng([self.D.init_seeds[r], 2]).integers(0, len(pairs), size=(self.D.n_updates, S.BATCH))
                for arm in S.ARMS:
                    for st in self.D.structures[arm]:
                        key = (arm, st, r)
                        c_fit0 = self.B.clock()
                        if arm == "LIN":
                            try:
                                m, info = model_lin.fit(self.train[r], ns, order=st)
                                info = dict(failed=False, order=st, n_columns=info["n_columns"])
                            except model_lin.LinFailure as ex:
                                m, info = None, dict(failed=True, fail_reason=str(ex))
                            self.B.check()         # LIN is closed form (no updates): one check after the fit
                        else:
                            # ERRATUM E1.1/E2.1: ONE throwaway warm-up per production process start (n_coldstarts =
                            # process starts, fixed in RELEASE), before the first neural fit, same architecture as that
                            # fit, train windows only, own RNG; inside the fits phase (projected as W x n_coldstarts)
                            if not self.warmed_up:
                                with self.B.warmup():
                                    wu = WU.warmup(arm, st, ns.to_dict(), ctx, fut, tgt, n_updates=self.D.warmup_updates,
                                                   clock=self.B.clock, check=self.B.check)
                                self.warmed_up = True
                                self.L.append("warmup", rep=r, process_cpu_s=self.B.total(), **wu)
                            c_fit0 = self.B.clock()
                            mdl = MT.build(arm, st, ns.to_dict(), self.D.init_seeds[r])
                            lr = lambda t: MT.lr_cosine(t, self.D.n_updates, S.LR0, S.LR_MIN)
                            mdl, info = MT.train(mdl, ctx, fut, tgt, bidx, self.D.n_updates, lr, budget_check=self.B.check,
                                                 check_every=S.FIT_CHECK_EVERY,
                                                 progress_cb=self._fit_progress(arm, st, r, c_fit0))
                            info["n_params"] = MT.n_params(mdl)
                            info["val_mse_final"] = None if info["failed"] else MT.eval_loss(mdl, vctx, vfut, vtgt)
                            m = None if info["failed"] else mdl
                        self.models[key] = m
                        self.fit_info["%s_%s_r%d" % key] = info
                        # v8.3 C6: per-fit ledger event (arm, structure, updates, CPU-s)
                        self.L.append("fit_end", arm=arm, structure=st, rep=r, updates=info.get("updates_done"),
                                      failed=bool(info.get("failed")), fit_cpu_s=self.B.clock() - c_fit0,
                                      process_cpu_s=self.B.total())
            self.save_json("fit_info.json", self.fit_info)
            self.logs_pre_test.append(dict(fit_info=self.fit_info))

    def _fit_progress(self, arm, st, r, c_fit0):
        """v8.3 C6: ledger event every 100 updates inside fits (updates, fit CPU-s, process CPU-s; time_unix = wall)."""
        def cb(n):
            if n % S.LEDGER_UPDATE_EVERY == 0:
                self.L.append("fit_progress", arm=arm, structure=st, rep=r, updates=n,
                              fit_cpu_s=self.B.clock() - c_fit0, process_cpu_s=self.B.total())
        return cb

    def phase_tuning(self):
        D = self.D
        with self.B.phase("tuning"):
            scens = self._scenarios(D.tuning_seeds, lambda i: ["N0", SD.tuning_entry(i)], "tuning")
            self.logs_pre_test += [sc.meta() for sc in scens]
            table, rows_all, effort = {}, [], {}
            for arm in S.ARMS:
                tab = {}
                effort[arm] = dict(configs=0, episodes=0, decisions=0)
                for st in D.structures[arm]:
                    for lam in D.lambdas:
                        Js, Es, Ds, saf, fb = [], [], [], 0, 0
                        for r in range(S.REPLICATES):
                            pred = self._predictor(arm, r, st)
                            rows = self._run_set(scens, pred, lam)
                            sm = self._summ(rows)
                            Js.append(sm["J"]); Es.append(sm["E"]); Ds.append(sm["D"])
                            saf += int(sm["safety"].sum()); fb += int(sm["fallback"].sum())
                            effort[arm]["episodes"] += len(rows)
                            effort[arm]["decisions"] += len(rows) * S.N_CONTROL
                            rows_all += [dict(arm=arm, structure=st, lam=lam, rep=r, seed=x["meta"]["seed"],
                                              iv_id=x["meta"]["iv_id"], E=x["E"], D=x["D"], J=x["J"],
                                              safety=x["safety_exit"], fallback=x["fallback"]) for x in rows]
                        effort[arm]["configs"] += 1
                        # means over all 3 reps x n episodes (equal counts per rep)
                        table[(arm, st, lam)] = dict(mean_J=float(np.mean(np.concatenate(Js))),
                                                     mean_E=float(np.mean(np.concatenate(Es))),
                                                     mean_D=float(np.mean(np.concatenate(Ds))),
                                                     rep_mean_E=[float(np.mean(x)) for x in Es],
                                                     rep_mean_D=[float(np.mean(x)) for x in Ds],
                                                     n_rows=int(sum(len(x) for x in Es)),
                                                     safety_exits=saf, fallbacks=fb)
            self.selection = select_from_table(table, D.structures, D.lambdas)
            comp = self.selection["comparator"]
            self.selection.update(effort=effort, tuning_seeds=list(D.tuning_seeds), lambda_grid=list(D.lambdas),
                                  ladder_applied=list(D.ladder_applied))
            self.logs_pre_test += rows_all
            self.logs_pre_test.append(self.selection)
            exp_eps = len(D.structures["WM"]) * len(D.lambdas) * S.REPLICATES * len(scens)
            self.validity["equal_search_effort"] = all(
                e["configs"] == len(D.structures["WM"]) * len(D.lambdas) and e["episodes"] == exp_eps
                and e["decisions"] == exp_eps * S.N_CONTROL for e in effort.values())
            np.savez(os.path.join(self.out, "tuning_rows.npz"), rows=np.array([json.dumps(r, default=float) for r in rows_all]))
            # HELDOUT scan of everything before test, then seal the selection BEFORE any test-seed episode
            scan = stage0.heldout_scan(self.logs_pre_test)
            self.validity["no_heldout_pre_test"] = scan["valid"]
            h = self.L.write_sealed("COMPARATOR_SELECTION.json", self.selection)
            self.results["comparator_selection_sha256"] = h
            # v8.2 gate (PLAN §9): the sealed selection must follow §8 exactly; recomputed from the sealed file,
            # and the sealed table means are recomputed from the saved tuning rows
            self.validity["selection_recomputed_from_sealed_table"] = (
                recompute_selection_ok(os.path.join(self.out, "COMPARATOR_SELECTION.json"))
                and table_matches_rows(self.selection["tuning_table"], rows_all))
            self.G.selection_sealed = True

    def _instances(self, arm):
        st = self.selection["selected"][arm]["structure"]
        lam = self.selection["selected"][arm]["lam"]
        return st, lam

    def _arm_eval(self, arm, scens, models_override=None):
        st, lam = self._instances(arm)
        per_rep = []
        for r in range(S.REPLICATES):
            if models_override is not None:
                pred = PR.make(arm, models_override[r], self.ns[r])
            else:
                pred = self._predictor(arm, r, st)
            per_rep.append(self._summ(self._run_set(scens, pred, lam)))
        return per_rep

    def phase_test(self):
        self.L.append("test_begin")
        with self.B.phase("test"):
            self.test_scens = self._scenarios(self.D.test_seeds, lambda i: list(S.HELDOUT_IDS), "test_heldout")
            self.cluster_of = np.array([sc.position for sc in self.test_scens])
            self.test = {arm: self._arm_eval(arm, self.test_scens) for arm in S.ARMS}
        n = len(self.test_scens) * S.REPLICATES
        self.validity["fallback_le_10pct"] = {a: int(sum(x["fallback"].sum() for x in self.test[a])) <= S.FALLBACK_MAX_FRAC * n
                                              for a in S.ARMS}

    def _mean_rep(self, per_rep, key):
        return np.mean(np.stack([x[key] for x in per_rep]), 0)

    def phase_primary(self):
        with self.B.phase("bootstrap_stats"):
            nC = len(self.D.test_seeds)
            idx = ST.boot_matrix(S.BOOT_SEEDS["primary"], nC)
            p = os.path.join(self.out, "BOOT_INDEX_primary.npy")
            np.save(p, idx)
            self.L.append("boot_matrix_saved", file="BOOT_INDEX_primary.npy", sha256=_sha(p))
            comp = self.selection["comparator"]
            saf = {a: int(sum(x["safety"].sum() for x in self.test[a])) for a in S.ARMS}
            prim = ST.primary_endpoints(self._mean_rep(self.test["WM"], "E"), self._mean_rep(self.test[comp], "E"),
                                        self._mean_rep(self.test["WM"], "D"), self._mean_rep(self.test[comp], "D"),
                                        self.cluster_of, idx, saf["WM"], saf[comp], check=self.B.check)
            prim["comparator"] = comp
            prim["safety_exits_test_all_arms"] = saf
            prim["fallback_test_all_arms"] = {a: int(sum(x["fallback"].sum() for x in self.test[a])) for a in S.ARMS}
            np.savez(os.path.join(self.out, "test_results.npz"),
                     **{"%s_%s" % (a, k): np.stack([x[k] for x in self.test[a]]) for a in S.ARMS
                        for k in ("E", "D", "J", "E2", "D2", "J2", "safety", "fallback", "settling")},
                     cluster_of=self.cluster_of,
                     iv_ids=np.array([sc.iv_id for sc in self.test_scens]))
            self.primary = prim
            self.results["primary_sha256"] = self.L.write_sealed("PRIMARY_RESULT.json", prim)
            self.G.primary_sealed = True

    def phase_in_dist(self):
        if not self.D.include_in_dist:
            return
        with self.B.phase("in_distribution"):
            self.id_scens = self._scenarios(self.D.test_seeds, lambda i: [S.TRAIN_IDS[i % 6]], "test_id")
            self.id_res = {arm: self._arm_eval(arm, self.id_scens) for arm in S.ARMS}

    def phase_pi_oracle(self):
        with self.B.phase("pi_oracle"):
            self.pi_test = self._summ(self._run_set(self.test_scens, None, 1.0, mode="pi"))
            self.pi_test_rows_traj = None
            self.oracle_test = self._summ(self._run_set(self.test_scens, None, 1.0, mode="oracle"))
            if self.D.include_in_dist:
                self.pi_id = self._summ(self._run_set(self.id_scens, None, 1.0, mode="pi"))
                self.oracle_id = self._summ(self._run_set(self.id_scens, None, 1.0, mode="oracle"))

    def phase_adaptation(self):
        D = self.D
        with self.B.phase("adaptation"):
            Kmax = max(D.adapt_K)
            self.adapt = {arm: {} for arm in S.ARMS}
            self.pi_adapt = {}
            eval_seeds = D.test_seeds[:D.adapt_eval_n]
            for ti, ht in enumerate(S.HELDOUT_TYPES):
                comp = SD.adaptation_composition(ht, Kmax)
                if D.mode != "production":
                    comp = [(D.adapt_seeds[j % len(D.adapt_seeds)], j, iid, on) for (_, j, iid, on) in comp]
                data = []
                for (s, j, iid, on) in comp:
                    data.append(P.data_episode(s, j, iid, on, self.G, "adaptation_data")[0])
                    self.B.check()                 # v8.3 C6b: after every episode
                ev = [P.Scenario(s, i, ht + "b", self.G, "adapt_eval") for i, s in enumerate(eval_seeds)]
                self.pi_adapt[ht] = self._summ(self._run_set(ev, None, 1.0, mode="pi"))
                for K in D.adapt_K:
                    obsK = P.stack_obs(data[:K])
                    for arm in S.ARMS:
                        st, lam = self._instances(arm)
                        models = []
                        for r in range(S.REPLICATES):
                            base = self.models[(arm, st, r)]
                            if base is None:
                                models.append(None)
                                continue
                            if arm == "LIN":
                                both = {k: np.concatenate([self.train[r][k], obsK[k]]) for k in ("y_r", "T_o", "u", "s")}
                                try:
                                    m, _ = model_lin.fit(both, self.ns[r], order=st)
                                except model_lin.LinFailure:
                                    m = None
                            else:
                                pairs = Fe.train_pairs(K)
                                ctx, fut, tgt = Fe.window_arrays(obsK, pairs, self.ns[r])
                                bidx = np.random.default_rng([D.init_seeds[r], 3, ti, K]).integers(
                                    0, len(pairs), size=(D.adapt_updates, S.BATCH))
                                m, info = MT.train(MT.clone(base), ctx, fut, tgt, bidx, D.adapt_updates,
                                                   lambda t: S.ADAPT_LR, budget_check=None, log_every=0)
                                # v8.3 C6b: adaptation is not 'fits' -> checks after every episode only (see IMPL_NOTES_v8.3)
                                m = None if info["failed"] else m
                            models.append(m)
                        self.adapt[arm][(ht, K)] = self._arm_eval(arm, ev, models_override=models)

    def phase_secondary(self):
        D = self.D
        with self.B.phase("bootstrap_stats"):
            comp = self.selection["comparator"]
            nA = D.adapt_eval_n
            idx_a = ST.boot_matrix(S.BOOT_SEEDS["adapt"], nA)
            idx_s = ST.boot_matrix(S.BOOT_SEEDS["secondary"], len(D.test_seeds))
            for nm, m in (("adapt", idx_a), ("secondary", idx_s)):
                p = os.path.join(self.out, "BOOT_INDEX_%s.npy" % nm)
                np.save(p, m)
                self.L.append("boot_matrix_saved", file=os.path.basename(p), sha256=_sha(p))
            sec = {}

            def sd(a_rep, c_rep, ek, dk, cl, idx):
                Ew, Ec = self._mean_rep(a_rep, ek), self._mean_rep(c_rep, ek)
                Dw, Dc = self._mean_rep(a_rep, dk), self._mean_rep(c_rep, dk)
                Sb, dDb, _ = ST.boot_S_dD(Ew, Ec, Dw, Dc, cl, idx, check=self.B.check)
                return dict(S=ST.saving(Ew, Ec), S_ci=list(ST.ci(Sb)), dD=float(np.mean(Dw - Dc)), dD_ci=list(ST.ci(dDb)))

            sec["secondary_window"] = sd(self.test["WM"], self.test[comp], "E2", "D2", self.cluster_of, idx_s)
            if D.include_in_dist:
                cl_id = np.array([sc.position for sc in self.id_scens])
                sec["in_distribution"] = sd(self.id_res["WM"], self.id_res[comp], "E", "D", cl_id, idx_s)
                sec["safety_exits_in_dist"] = {a: int(sum(x["safety"].sum() for x in self.id_res[a])) for a in S.ARMS}
            ivs = np.array([sc.iv_id for sc in self.test_scens])
            sec["per_type"] = {}
            for ht in S.HELDOUT_TYPES:
                msk = np.array([i.startswith(ht) for i in ivs])
                Ew, Ec = self._mean_rep(self.test["WM"], "E")[msk], self._mean_rep(self.test[comp], "E")[msk]
                Dw, Dc = self._mean_rep(self.test["WM"], "D")[msk], self._mean_rep(self.test[comp], "D")[msk]
                sec["per_type"][ht] = dict(S=ST.saving(Ew, Ec), dD=float(np.mean(Dw - Dc)))
            ad = {}
            for K in D.adapt_K:
                wm = [self.adapt["WM"][(ht, K)] for ht in S.HELDOUT_TYPES]
                cp = [self.adapt[comp][(ht, K)] for ht in S.HELDOUT_TYPES]
                cat = lambda lst: [{k: np.concatenate([lst[t][r][k] for t in range(len(lst))]) for k in ("E", "D", "safety")}
                                   for r in range(S.REPLICATES)]
                cl = np.tile(np.arange(nA), len(S.HELDOUT_TYPES))
                ad["K%d" % K] = sd(cat(wm), cat(cp), "E", "D", cl, idx_a)
                ad["K%d" % K]["safety_exits"] = {a: int(sum(self.adapt[a][(ht, K)][r]["safety"].sum()
                                                            for ht in S.HELDOUT_TYPES for r in range(S.REPLICATES)))
                                                 for a in S.ARMS}
            sec["adaptation"] = ad
            sec["settling_mean_k"] = {a: float(np.mean([x["settling"][x["settling"] >= 0].mean() if (x["settling"] >= 0).any()
                                                        else np.nan for x in self.test[a]])) for a in S.ARMS}
            sec["oracle_test"] = dict(E=float(self.oracle_test["E"].sum()), J=float(self.oracle_test["J"].sum()))
            sec["pi_test"] = dict(E=float(self.pi_test["E"].sum()), J=float(self.pi_test["J"].sum()),
                                  safety_exits=int(self.pi_test["safety"].sum()))
            # v8.2 PLAN §9 "other secondaries": S and dD of WM against every non-WM arm (descriptive, point only)
            sec["wm_vs_each_arm_descriptive"] = {
                a: dict(S=ST.saving(self._mean_rep(self.test["WM"], "E"), self._mean_rep(self.test[a], "E")),
                        dD=float(np.mean(self._mean_rep(self.test["WM"], "D") - self._mean_rep(self.test[a], "D"))))
                for a in S.ARMS if a != "WM"}
            sec["arms_test_totals"] = {a: dict(E=float(self._mean_rep(self.test[a], "E").sum()),
                                               J=float(self._mean_rep(self.test[a], "J").sum())) for a in S.ARMS}
            sec["tuning_safety_exits"] = {a: int(sum(t["safety_exits"] for t in self.selection["tuning_table"] if t["arm"] == a))
                                          for a in S.ARMS}
            sec["heldout_mse_12step"] = self._heldout_mse()
            sec["heldout_mse_12step_ci"] = self._heldout_mse_ci()
            self.secondary = sec
            self.save_json("SECONDARY_RESULTS.json", sec)

    def _heldout_mse(self):
        """12-step y_r MSE at origin k=120 on the PI-reference trajectories of the HELDOUT test episodes
        (logged commanded actions), selected config x 3 replicates (IMPL_NOTES D10).
        v8.2: also keeps the per-(replicate, episode) errors (NaN = no prediction) so that per-test-seed means
        can be paired across arms for the MSE contrast CIs (bootstrap seed 9924)."""
        out = {}
        trajs = []
        for sc in self.test_scens:
            r = CL.run_episode(sc, None, 1.0, mode="pi")
            self.B.check()                         # v8.3 C6b: after every episode
            if "traj" in r and not r["safety_exit"]:
                trajs.append((sc, r["traj"]))
        k = 120
        self.mse_err = {}
        for arm in S.ARMS:
            st, _ = self._instances(arm)
            errs = []
            E = np.full((S.REPLICATES, len(trajs)), np.nan)
            for rep in range(S.REPLICATES):
                pred = self._predictor(arm, rep, st)
                if pred is None:
                    continue
                for j, (sc, tj) in enumerate(trajs):
                    view = Fe.ObservableView(k, tj["y"], tj["To"], tj["u"], sc.d.s_occ, sc.d.s_night)
                    yh = pred(view, tj["u"][k:k + S.H][None, :])
                    if yh is not None:
                        errs.append(float(np.mean((yh[0] - tj["y"][k + 1:k + S.H + 1]) ** 2)))
                        E[rep, j] = errs[-1]
                self.B.check()
            out[arm] = float(np.mean(errs)) if errs else None
            self.mse_err[arm] = E
        self.mse_cluster = np.array([sc.position for sc, _ in trajs], dtype=np.int64)
        return out

    def _heldout_mse_ci(self):
        """v8.2 secondary: paired cluster-bootstrap CIs of per-test-seed mean MSE differences (CONFIG
        endpoints.secondary_mse_ci). Per-seed mean = nanmean over that seed's (replicate, episode) errors."""
        nC = len(self.D.test_seeds)
        idx = ST.boot_matrix(S.BOOT_SEEDS["mse"], nC)
        p = os.path.join(self.out, "BOOT_INDEX_mse.npy")
        np.save(p, idx)
        self.L.append("boot_matrix_saved", file="BOOT_INDEX_mse.npy", sha256=_sha(p))
        per_seed = {}
        for arm in S.ARMS:
            E = self.mse_err[arm]
            v = np.full(nC, np.nan)
            for c in range(nC):
                x = E[:, self.mse_cluster == c]
                if np.isfinite(x).any():
                    v[c] = float(np.nanmean(x))
            per_seed[arm] = v
        np.savez(os.path.join(self.out, "heldout_mse_rows.npz"), cluster_of=self.mse_cluster,
                 **{"%s_err" % a: self.mse_err[a] for a in S.ARMS}, **{"%s_per_seed" % a: per_seed[a] for a in S.ARMS})
        return dict(per_seed_mean={a: [None if np.isnan(x) else float(x) for x in per_seed[a]] for a in S.ARMS},
                    contrasts=ST.mse_contrast_ci(per_seed, idx, check=self.B.check), boot_seed=S.BOOT_SEEDS["mse"])

    def phase_audit(self, extra_validity=None):
        with self.B.phase("audit"):
            if self.release_hashes is not None:
                ok = True
                try:
                    S.verify_plan_hashes()
                except S.SpecError:
                    ok = False
                for path, h in self.release_hashes.items():
                    ok = ok and os.path.exists(path) and _sha(path) == h
                self.validity["frozen_hashes_match_release"] = ok
            ev = self.L.events()
            order = [e.get("file") or e["event"] for e in ev]
            self.validity["selection_sealed_before_test"] = (
                "COMPARATOR_SELECTION.json" in order and "test_begin" in order
                and order.index("COMPARATOR_SELECTION.json") < order.index("test_begin"))
            self.validity["identical_inputs"] = (len(self.input_signatures) <= 1
                                                 and stage0.model_import_check()["valid"])
            if extra_validity:
                self.validity.update(extra_validity)
        bud = self.B.finalize()
        self.validity["all_caps_respected"] = bud["all_caps_respected"]
        return bud

    def validity_ok(self):
        v = dict(self.validity)
        fb = v.pop("fallback_le_10pct", {})
        return all(bool(x) for x in v.values()) and all(fb.values()), v, fb

    def run_all(self):
        from .budget import CapExceeded
        status = "RUNNING"
        from .hostenv import thread_readout
        self.L.append("run_start", process_cpu_s=self.B.total(), threads=thread_readout(), caps=self.D.caps,
                      global_cap=self.D.global_cap, k1_cap=self.D.k1_cap, ladder_applied=list(self.D.ladder_applied))
        try:
            self.phase_data()
            self.phase_fits()
            self.phase_tuning()
            self.phase_test()
            self.phase_primary()
            self.phase_in_dist()
            self.phase_pi_oracle()
            self.phase_adaptation()
            self.phase_secondary()
            bud = self.phase_audit()
            ok, v, fb = self.validity_ok()
            status = ST.outcome(self.primary, ok)
        except CapExceeded as ex:
            bud = dict(cap_exceeded=str(ex), used=self.B.used, total=self.B.total(), caps=self.B.caps,
                       global_cap=self.B.global_cap, k1_cap=self.B.k1_cap, n_checks=self.B.n_checks,
                       max_check_jump_cpu_s=self.B.max_jump)
            v, fb, status = self.validity, {}, "INVALID"
        final = dict(outcome=status, validity=v, fallback_gate=fb, budget=bud, design_mode=self.D.mode,
                     grid_override=S.GRID_OVERRIDE,
                     selection_fallback_flags=(getattr(self, "selection", None) or {}).get("fallback_flags"),
                     ladder_applied=list(self.D.ladder_applied),
                     primary=getattr(self, "primary", None), selection=getattr(self, "selection", None))
        self.save_json("FINAL_STATUS.json", final)
        self.L.append("final", outcome=status, process_cpu_s=self.B.total())     # v8.3: run end = this event
        return final


def select_from_table(table, structures=None, lambdas=None):
    """PLAN §8 applied to a full tuning table {(arm, structure, lam): {"mean_E", "mean_D", "mean_J", ...}}.
    Pure function of the table (used for sealing AND for the recompute gate)."""
    bound, brec = ST.comfort_bound(table)
    selected, tab_rows = {}, []
    for arm in S.ARMS:
        tab = {(k[1], k[2]): v for k, v in table.items() if k[0] == arm}
        (st, lam), feas = ST.select_within_arm(arm, tab, bound)
        v = tab[(st, lam)]
        selected[arm] = dict(structure=st, lam=lam, mean_E=v["mean_E"], mean_D=v["mean_D"], mean_J=v["mean_J"],
                             feasible=bool(feas))
    comp = ST.select_comparator({a: selected[a] for a in S.COMPARATOR_ARMS_TIEBREAK})
    for k, v in table.items():
        tab_rows.append(dict(arm=k[0], structure=k[1], lam=k[2], feasible=bool(v["mean_D"] <= bound), **v))
    return dict(selected=selected, comparator=comp, comfort_bound=brec, selection_rule="v8.2 PLAN §8 min E s.t. D<=bound",
                fallback_flags=ST.fallback_flags(selected, comp),
                tuning_table=tab_rows)


def recompute_selection_ok(sealed_path):
    """Recompute selection + comparator + feasibility flags from the sealed COMPARATOR_SELECTION.json table."""
    with open(sealed_path) as f:
        sel = json.load(f)
    keep = ("mean_E", "mean_D", "mean_J")
    table = {(t["arm"], t["structure"], float(t["lam"])): {k: t[k] for k in keep} for t in sel["tuning_table"]}
    re = select_from_table(table)
    if re["comparator"] != sel["comparator"] or re["comfort_bound"]["value"] != sel["comfort_bound"]["value"]:
        return False
    if re["fallback_flags"] != sel["fallback_flags"]:
        return False
    for a in S.ARMS:
        x, y = re["selected"][a], sel["selected"][a]
        if (x["structure"], float(x["lam"]), x["feasible"]) != (y["structure"], float(y["lam"]), y["feasible"]):
            return False
    flags = {(t["arm"], t["structure"], float(t["lam"])): t["feasible"] for t in sel["tuning_table"]}
    return all(flags[(t["arm"], t["structure"], float(t["lam"]))] == t["feasible"] for t in re["tuning_table"])


def table_matches_rows(tuning_table, rows):
    """Sealed per-config means equal the means recomputed from the per-episode tuning rows (tuning_rows.npz)."""
    for t in tuning_table:
        rr = [r for r in rows if r["arm"] == t["arm"] and r["structure"] == t["structure"] and r["lam"] == t["lam"]]
        if len(rr) != t["n_rows"]:
            return False
        for k in ("E", "D", "J"):
            if float(np.mean([r[k] for r in rr])) != t["mean_" + k]:
                return False
    return True


def smoke_design(n_updates=20, adapt_updates=5):
    """Toy design on preflight seeds 1-4 only (tests). Never used for production."""
    tr = {r: [1, 2, 3, 4] for r in range(S.REPLICATES)}
    return Design(mode="smoke", train_seeds=tr, val_seeds={r: [1, 2] for r in range(S.REPLICATES)},
                  tuning_seeds=[1, 2], test_seeds=[3, 4], adapt_seeds=[1, 2, 3, 4], adapt_eval_n=2,
                  adapt_K=(1, 3), n_updates=n_updates, adapt_updates=adapt_updates,
                  structures={"WM": (32,), "GRU": (32,), "LIN": (3,), "RC_only": (3,)},
                  caps={k: 1e9 for k in S.CAPS}, global_cap=1e9, k1_cap=1e9, warmup_updates=5)
