"""v8.3 probe entry point (PLAN_v8.3 r3 C2/C3/C4/C5, K1/K2; dahill PLAN_REVIEW r3 A1/A2). CPU only, 1 thread.
Every invocation is a new, numbered attempt in probe/PROBE_LEDGER.jsonl (A1); outputs are exclusive-create
(PROBE_v8.3_attempt<N>.json, LOAD_SAMPLES_probe_attempt<N>.txt) and never overwritten. CONFIG_v8.3.json is
written ONLY on GATE_PASS (exclusive create); once it exists, no further probe attempt is accepted.
Touches no tuning, test or adaptation seed. Does NOT start the production run.
Requires (fail closed): dahill CODE_PASS_v8.3*.json on the current manifest; on a non-box host a valid same-host toy
preflight attempt (run_preflight.py) for K2; the start-rule threshold field in RELEASE_v8.3_DRAFT.json (dahill req 5);
on a hybrid CPU a determinable P-core (E1.2). Blocked for good after a linearity diagnosis under ERRATUM E1 (E1.5).
Usage (from any cwd):  WM_ROOT=<dir holding v8.1 v8.2 v8.3> WM_AUDIT_ROOT=<dir holding audit_dahill_v8.*> \\
                       python <WM_ROOT>/v8.3/code/run_probe.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

torch.set_num_threads(1)
from v81 import gates as GT  # noqa: E402
from v81 import hostenv as HE  # noqa: E402
from v81 import probe as PB  # noqa: E402
from v81 import spec as S  # noqa: E402
from v81.ledger import canon, write_once  # noqa: E402

A1_RULE = ("A1: probe cap 450 CPU-s hit OR linearity ratio outside [0.87, 1.15] -> PROBE_DIAGNOSIS: no freeze, no "
           "projection from partial data; a new probe is a new attempt (counter in PROBE_LEDGER.jsonl, never overwrite).")
A2_RULE = ("A2: probe and production on the same host under the same C4 start rule; hostname, CPU model and thread "
           "count recorded here and bound in RELEASE; production refuses on a different fingerprint. On a host "
           "switch the C2 probe is repeated there (the C4 'toy preflight' sentence is superseded).")
K2_HOST_RULE = ("uni 11:12 (5): K2 R across hosts is invalid. On the v8.2 box (spec.BOX_FINGERPRINT) the toy reference is "
                "v8.2/PREFLIGHT_PROJECTION.json (fits 336.8); on any other host the unchanged v8.2 toy preflight must have "
                "run there (run_preflight.py, 120 s cap, numbered attempt in probe/TOY_LEDGER.jsonl) and R = probe P_fits / "
                "that attempt's fits projection. Without a valid same-host toy attempt the probe does not start.")


def _code_pass_record():
    path, info = GT.find_code_pass(GT.manifest_sha())
    return dict(file=path, sha256=info) if path else dict(file=None, sha256=None)


E1_RULE = ("plan r4 = PLAN r3 + ERRATUM_v8.3_E1 (ea642114) + ERRATUM_v8.3_E2 (3d6f8c0a; E2 wins): E1.1 200-update "
           "throwaway warm-up (same architecture) before each timed fit segment, logged as 'warmup', counts toward the "
           "450 s cap; E2.1 W x n_coldstarts is a fixed cost in the fits projection (n_coldstarts = production process "
           "starts fixed in RELEASE, or the number of fits if the excess recurs per fit); E2.2 two full h96 fits, "
           "blocks of 50 updates, linearity on fit 2; E2.3 sidecar <= 2 s, max load1 / min MHz / steal reported; "
           "E2.4 start threshold = 1.5 x p95 of a 10-min baseline fixed in RELEASE (box: < 1.0 unless measured); "
           "E1.2/E2.5 hybrid CPU -> one P-core, core list bound in RELEASE, toy reference = pinned toy attempt on the "
           "same core, production identical; E1.5/E2.7 a linearity diagnosis under E1/E2 blocks every further probe.")


def main(probe_dir=None, config_path=None, start_rule_fn=None, measure_kw=None, with_sidecar=True, t0=0.0,
         draft_path=None, pin_setter=None, pin_getter=None, topology=None):
    probe_dir = probe_dir or os.path.join(S.V83_DIR, S.V83["probe_dir"])
    config_path = config_path or os.path.join(S.V83_DIR, S.V83["config_file"])
    os.makedirs(probe_dir, exist_ok=True)
    led = os.path.join(probe_dir, GT.PROBE_LEDGER)
    if os.path.exists(config_path):
        raise SystemExit("%s exists: the probe already passed; no further attempt" % config_path)
    n = GT.count_attempts(led) + 1
    out_path = os.path.join(probe_dir, "PROBE_v8.3_attempt%d.json" % n)
    if os.path.exists(out_path):
        raise SystemExit("%s exists; refusing to overwrite" % out_path)
    host = HE.host_fingerprint()
    threads = HE.thread_readout()
    # dahill req (5) / E2.4: threshold fixed in the RELEASE draft (measured baseline on this host; box default 1.0)
    l1max, thr, thr_sha, tr = GT.start_threshold(host, draft_path)
    start = (start_rule_fn or (lambda: HE.start_rule(load_max=l1max if l1max is not None else 0.0)))()
    GT.append(led, "attempt_start", attempt=n, pid=os.getpid(), host=host, is_box=HE.is_box(host), threads=threads,
              start_rule=start, process_cpu_s=time.process_time(), erratum=S.ERRATUM)
    reasons = GT.e1_5_blocked(led)                      # E1.5: blocked after a linearity failure under E1
    hashes = {}
    try:
        hashes["v8.2_bundle"] = S.verify_plan_hashes()
        hashes["v8.3_bundle"] = S.verify_v83_bundle()
    except S.SpecError as ex:
        reasons.append("bundle: %s" % ex)
    se = os.path.join(S.V83_DIR, S.V83["seed_evidence_file"])
    if not os.path.exists(se) or S.sha256_file(se) != S.V83["seed_evidence_sha256"]:
        reasons.append("SEED_EVIDENCE_v8.3.md missing or hash mismatch")
    # C1, C4, frozen manifest, dahill CODE_PASS on it (uni 11:12 (9)), sidecar fields readable (stricter S5)
    reasons += GT.common_gate_reasons(threads, start)
    reasons += tr
    n_starts, ps = GT.process_starts(draft_path)        # E2.1: production process starts fixed in RELEASE
    reasons += ps
    # E1.2: P-core pinning on a hybrid CPU (fail closed if P-cores not determinable)
    pin, pr = HE.plan_pinning(topology)
    reasons += pr
    # K2 reference for THIS host (uni 11:12 (5)): box -> v8.2 file; other host -> valid same-host toy attempt,
    # pinned identically if pinned (E2.5)
    ref, rr = GT.k2_reference(host, probe_dir, pin)
    reasons += rr
    toy = GT.load_k2_reference(ref) if ref else None
    if ref and toy is None:
        reasons.append("K2 reference file unreadable or hash mismatch")
    sc_path = HE.sidecar_path()
    sc_sha = S.sha256_file(sc_path) if os.path.exists(sc_path) else None
    res = dict(kind="v8.3 full-size probe (PLAN r3 C2)", attempt=n, time_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               host_fingerprint=host, is_box=HE.is_box(host), threads=threads, start_rule=start, plan_hashes=hashes,
               code_manifest_sha256=GT.manifest_sha(), code_pass=_code_pass_record(),
               sidecar_sha256=sc_sha, k2_reference=ref, probe_cap_cpu_s=S.PROBE_CAP_CPU_S, erratum=S.ERRATUM,
               erratum_sha256=S.V83["erratum_sha256"], erratum2_sha256=S.V83["erratum2_sha256"],
               start_rule_threshold=thr, start_rule_threshold_sha256=thr_sha, start_load1_max=l1max,
               production_process_starts=n_starts, pinning=pin, rules=dict(A1=A1_RULE, A2=A2_RULE, K2_HOST=K2_HOST_RULE, E1=E1_RULE))
    if not reasons and pin and pin.get("pinned"):
        reasons += HE.apply_pinning(pin, setter=pin_setter, getter=pin_getter)
    if reasons:
        blocked = any(r.startswith("ERRATUM E1.5") for r in reasons)
        res.update(outcome="PROBE_BLOCKED_E1_5" if blocked else "PROBE_NOT_STARTED", reasons=reasons,
                   v8_3_status="NOT_LAUNCHED (E1.5)" if blocked else None)
        write_once(out_path, canon(res))
        GT.append(led, "attempt_end", attempt=n, outcome=res["outcome"], file=os.path.basename(out_path),
                  sha256=S.sha256_file(out_path), erratum=S.ERRATUM)
        print(json.dumps(dict(outcome=res["outcome"], reasons=reasons), indent=1))
        return res
    if with_sidecar:
        samples = os.path.join(probe_dir, "LOAD_SAMPLES_probe_attempt%d.txt" % n)
        sc_proc = HE.start_sidecar(os.getpid(), samples, interval=S.PROBE_SIDECAR_INTERVAL_S)   # dahill req (3)
        res["sidecar"] = dict(file=os.path.basename(samples), pid=sc_proc.pid, interval_s=S.PROBE_SIDECAR_INTERVAL_S)
    budget = PB.ProbeBudget(t0=t0)
    pm, aborted = {}, None
    ev = lambda e, **k: GT.append(led, e, attempt=n, **k)
    w0 = time.time()
    try:
        PB.measure_probe(budget, event=ev, out=pm, n_process_starts=n_starts, **(measure_kw or {}))
    except PB.ProbeCapExceeded as ex:
        aborted = str(ex)
    w1 = time.time()
    if with_sidecar:                                    # E2.3: load / MHz / steal over the timed segments
        time.sleep(min(2.5, S.PROBE_SIDECAR_INTERVAL_S + 0.5))
        res["sidecar_summary"] = GT.sidecar_summary(samples, w0, w1 + S.PROBE_SIDECAR_INTERVAL_S)
    dec = PB.decide(pm, toy, aborted=aborted, toy_fits=ref["toy_fits"])
    fresh = GT.k2_fresh(host, probe_dir, (dec.get("fits") or {}).get("total"))   # box only, descriptive
    if pin and pin.get("pinned"):
        pin["core_cur_mhz_at_end"] = HE.core_mhz(int(pin["core"]))[1]
    res.update(outcome=dec["outcome"], measurements=pm, decision=dec, probe_cpu_s=budget.used(), k2_fresh=fresh,
               pinning=pin,
               n_checks=budget.n_checks, max_check_jump_cpu_s=budget.max_jump,
               wall_end_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_once(out_path, canon(res))
    psha = S.sha256_file(out_path)
    cfg_sha = None
    if dec["outcome"] == "GATE_PASS":
        cfg = dict(version="v8.3", status="derived from probe; binding once bound in RELEASE_v8.3",
                   probe=dict(file=os.path.relpath(out_path, S.V83_DIR), sha256=psha, attempt=n),
                   caps_cpu_s=dict(per_phase=dec["caps"], global_cap=dec["global_cap"], k1_global_cap=S.K1_GLOBAL_CAP),
                   ladder_applied=dec["ladder_applied"], projection_cpu_s=dec["projection"], R=dec["R"],
                   k2_reference=ref, host_fingerprint=host, pinning=pin, start_rule_threshold=thr,
                   start_rule_threshold_sha256=thr_sha, erratum=S.ERRATUM, erratum_sha256=S.V83["erratum_sha256"],
                   erratum2_sha256=S.V83["erratum2_sha256"], production_process_starts=n_starts,
                   rules=dict(A1=A1_RULE, A2=A2_RULE, K2_HOST=K2_HOST_RULE, E1=E1_RULE))
        write_once(config_path, canon(cfg))
        cfg_sha = S.sha256_file(config_path)
    GT.append(led, "attempt_end", attempt=n, outcome=dec["outcome"], file=os.path.basename(out_path), sha256=psha,
              config_sha256=cfg_sha, probe_cpu_s=budget.used(), erratum=S.ERRATUM, sidecar_summary=res.get("sidecar_summary"),
              diagnosis_kind=dec.get("diagnosis_kind"))
    print(json.dumps(dict(outcome=dec["outcome"], reason=dec.get("reason"), R=dec.get("R"),
                          caps=dec.get("caps"), probe_cpu_s=budget.used()), indent=1))
    return res


if __name__ == "__main__":
    main()
