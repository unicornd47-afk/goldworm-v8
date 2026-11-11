"""v8.3 production entry point. Order (PLAN_v8.3 r3 C1, C4, C5; A2): C1 thread readback -> C4 start rule (logged)
-> RELEASE_v8.3 start rule incl. A2 host fingerprint == probe host -> sidecar -> run. Any failure writes a
NOT_LAUNCHED record (append-only) and exits without generating data. One run only: refuses if run_output exists.
Caps come from the bound CONFIG_v8.3.json (from the probe); global cap = sum of phase caps; K1 abort at 3600.
Usage: WM_ROOT=<dir holding v8.1 v8.2 v8.3> WM_AUDIT_ROOT=<dir holding audit_dahill_v8.*> python <WM_ROOT>/v8.3/code/run_production.py"""
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
from v81 import hostenv as HE  # noqa: E402
from v81 import pipeline as PP  # noqa: E402
from v81 import release as RL  # noqa: E402
from v81 import spec as S  # noqa: E402


def _not_launched(out_dir, st):
    nl = os.path.join(os.path.dirname(out_dir), "NOT_LAUNCHED_%s.jsonl" % os.path.basename(out_dir))
    with open(nl, "a") as f:                                      # append-only record of every refused start
        f.write(json.dumps(dict(st, time_unix=time.time()), sort_keys=True, default=str) + "\n")
    print(json.dumps(st, indent=1, default=str))
    return st


def main(release_path=None, out_dir=None, start_rule_fn=None, with_sidecar=True, topology=None, pin_setter=None,
         pin_getter=None):
    release_path = release_path or os.path.join(S.V83_DIR, S.V83["release_file"])
    out_dir = out_dir or os.path.join(S.V83_DIR, "run_output")
    threads = HE.thread_readout()
    ok_t, bad_t = HE.thread_readback_ok(threads)
    host = HE.host_fingerprint()
    # dahill req (5) / E2.4: the start threshold bound in RELEASE (checked against the probe's in check_start)
    rel0 = {}
    if os.path.exists(release_path):
        with open(release_path) as f:
            rel0 = json.load(f)
    from v81 import gates as GT
    l1max, _, _, tr = GT.start_threshold(host, field=rel0.get(S.START_RULE_FIELD))
    start = (start_rule_fn or (lambda: HE.start_rule(load_max=l1max if l1max is not None else 0.0)))()
    reasons = [] if ok_t else ["C1 thread readback mismatch: %s" % bad_t]
    reasons += ["C4 start rule: %s" % r for r in start["reasons"]] + tr
    ok_s, bad_s = HE.sidecar_capable()                 # stricter S5: every C5 field readable on this host
    if not ok_s:
        reasons.append("C5 sidecar fields unreadable on this host: %s" % bad_s)
    ok, rr, rel, cfg = RL.check_start(release_path, current_host=host)
    reasons += rr
    if reasons:
        return _not_launched(out_dir, dict(outcome="NOT_LAUNCHED", reasons=reasons, threads=threads, start_rule=start,
                                           host=host))
    # ERRATUM E1.2 / dahill req (4): pinned identically to the probe (same P-core list and core), else NOT_LAUNCHED
    cur_pin, pr = HE.plan_pinning(topology)
    bound_pin = dict(rel.get("pinning") or {})
    ok_p, diff_p = HE.pinning_matches(bound_pin, cur_pin) if not pr else (False, pr)
    if not ok_p:
        return _not_launched(out_dir, dict(outcome="NOT_LAUNCHED", reasons=["E1.2 pinning differs from RELEASE: %s"
                                                                            % diff_p], host=host, pinning=cur_pin))
    pr = HE.apply_pinning(bound_pin, setter=pin_setter, getter=pin_getter)
    if pr:
        return _not_launched(out_dir, dict(outcome="NOT_LAUNCHED", reasons=pr, host=host, pinning=bound_pin))
    if os.path.exists(out_dir):
        raise SystemExit("run_output exists: one production run only (no retry, no resume)")
    caps = cfg["caps_cpu_s"]
    design = PP.Design(caps=dict(caps["per_phase"]), global_cap=float(caps["global_cap"]),
                       k1_cap=float(caps["k1_global_cap"])).apply_ladder(cfg.get("ladder_applied", []))
    bound = {os.path.normpath(os.path.join(S.PLAN_DIR, k)): v for k, v in S.BOUND_FILES.items()}
    bound[os.path.join(S.CODE_DIR, RL.CODE_MANIFEST)] = rel["code_hashes"]["manifest_sha256"]
    bound[RL.CONFIG_FILE] = rel["config_v8_3"]["sha256"]
    run = PP.Run(design, out_dir, release_hashes=bound)
    run.L.append("start_rule", threads=threads, start_rule=start, host=host, pinning=bound_pin)          # C4: logged right before launch
    if with_sidecar:
        samples = os.path.join(out_dir, "LOAD_SAMPLES_production_v8.3.txt")
        p = HE.start_sidecar(os.getpid(), samples)
        run.L.append("sidecar_started", pid=p.pid, file=os.path.basename(samples),
                     sha256=S.sha256_file(HE.sidecar_path()))
    final = run.run_all()
    print(json.dumps(dict(outcome=final["outcome"]), indent=1))
    return final


if __name__ == "__main__":
    main()
