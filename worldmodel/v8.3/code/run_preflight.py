"""Toy preflight entry point (CPU only, seeds 1-4, cap 120 CPU-s). Writes ../preflight/PREFLIGHT_PROJECTION.json.
Does NOT start the production run and touches no test/tuning/training/adaptation seed.

v8.3 (uni 11:12, interpretation 5; dahill A2): this is the UNCHANGED v8.2 toy preflight. main() is the v8.2 logic
byte for byte except the output path (argument or env WM_TOY_PREFLIGHT_OUT; never v8.2/PREFLIGHT_PROJECTION.json).
On a host other than the v8.2 box it must run there before the probe, as a numbered, append-only attempt
(attempt_main(): probe/TOY_LEDGER.jsonl, probe/TOY_PREFLIGHT_attempt<N>.json, same C1/C4/CODE_PASS gates as the probe).
Usage: WM_ROOT=... WM_AUDIT_ROOT=... python v8.3/code/run_preflight.py"""
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
from v81 import preflight as PF  # noqa: E402
from v81 import release as RL  # noqa: E402
from v81 import spec as S  # noqa: E402


def main(out=None):
    hashes = S.verify_plan_hashes()                       # STOP on mismatch (raises)
    setup_cpu = time.process_time()                      # imports + hash verification = overhead proxy
    out = out or os.environ.get("WM_TOY_PREFLIGHT_OUT")   # v8.3: path config only (v8.2: RL.PREFLIGHT_FILE)
    if not out:
        raise SystemExit("no output path (argument or WM_TOY_PREFLIGHT_OUT)")
    if os.path.exists(out):
        raise SystemExit("PREFLIGHT_PROJECTION.json exists; refusing to overwrite")
    man = os.path.join(S.CODE_DIR, RL.CODE_MANIFEST)
    man_sha = S.sha256_file(man) if os.path.exists(man) else None
    t0 = 0.0
    load_before = list(os.getloadavg())                  # IMPL_ERRATUM_v8.2_W1: log machine load
    m = PF.measure(t0)
    load_after = list(os.getloadavg())
    dec = PF.decide(m, setup_cpu)
    cpu = time.process_time()
    res = dict(kind="%s toy preflight" % S.BUNDLE["version"], device="cpu", torch_threads=torch.get_num_threads(),
               time_berlin=time.strftime("%Y-%m-%dT%H:%M:%S%z"), plan_hashes=hashes,
               code_manifest_sha256=man_sha, setup_cpu_s=setup_cpu, measurements=m,
               lambda_grid=list(S.LAMBDAS), grid_override=S.GRID_OVERRIDE,
               projection_cpu_s=dec["projection"], total_projection_cpu_s=dec["total"],
               caps_cpu_s=S.CAPS, gate_limits_cpu_s={k: S.GATE_FRAC * v for k, v in S.CAPS.items()},
               total_gate_cpu_s=S.TOTAL_GATE, phase_gates=dec["gates"], total_gate=dec["total_gate"],
               ladder_applied=dec["ladder_applied"], all_gates_pass=dec["all_gates_pass"],
               plausibility=dec["plausibility"], loadavg_1_5_15_before=load_before, loadavg_1_5_15_after=load_after,
               impl_errata=S.BUNDLE["impl_errata"],
               failing_non_ladder=dec.get("failing_non_ladder", []),
               preflight_cpu_s_used=cpu, preflight_cap_cpu_s=S.PREFLIGHT_CAP,
               preflight_within_cap=cpu <= S.PREFLIGHT_CAP,
               projection_rules="fits/tuning per-structure measured cost; test/in-distribution/adaptation per-arm max over "
                                "structures; overhead = measured setup CPU + 15 s reserve (Option1 PROD_SUPERVISOR_RESERVE_S); "
                                "LIN fit cost scaled linearly 4->32 episodes (refit 4->42)")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps(dict(projection=dec["projection"], total=dec["total"], gates=dec["gates"],
                          ladder=dec["ladder_applied"], all_pass=dec["all_gates_pass"], preflight_cpu=cpu,
                          plausibility=dec["plausibility"], load_before=load_before, load_after=load_after), indent=1))


def attempt_main(probe_dir=None, start_rule_fn=None, with_sidecar=True, draft_path=None, topology=None, pin_setter=None,
                 pin_getter=None):
    """v8.3: one numbered, logged toy-preflight attempt on THIS host (append-only; never overwrites)."""
    from v81 import gates as GT
    from v81 import hostenv as HE
    probe_dir = probe_dir or os.path.join(S.V83_DIR, S.V83["probe_dir"])
    os.makedirs(probe_dir, exist_ok=True)
    led = os.path.join(probe_dir, GT.TOY_LEDGER)
    n = GT.count_attempts(led) + 1
    out = os.path.join(probe_dir, "TOY_PREFLIGHT_attempt%d.json" % n)
    if os.path.exists(out):
        raise SystemExit("%s exists; refusing to overwrite" % out)
    host, threads = HE.host_fingerprint(), HE.thread_readout()
    l1max, thr, thr_sha, tr = GT.start_threshold(host, draft_path)              # dahill req (5) / E2.4
    start = (start_rule_fn or (lambda: HE.start_rule(load_max=l1max if l1max is not None else 0.0)))()
    pin, pr = HE.plan_pinning(topology)                                         # E2.5: pinned toy reference
    GT.append(led, "attempt_start", attempt=n, pid=os.getpid(), host=host, threads=threads, start_rule=start,
              is_box=HE.is_box(host), process_cpu_s=time.process_time(), start_rule_threshold_sha256=thr_sha,
              pinning=pin, erratum=S.ERRATUM)
    reasons = GT.common_gate_reasons(threads, start)
    reasons += tr + pr
    if not reasons and pin and pin.get("pinned"):
        reasons += HE.apply_pinning(pin, setter=pin_setter, getter=pin_getter)
    if reasons:
        GT.append(led, "attempt_end", attempt=n, outcome="TOY_NOT_STARTED", reasons=reasons, host=host, pinning=pin)
        print(json.dumps(dict(outcome="TOY_NOT_STARTED", reasons=reasons), indent=1))
        return dict(outcome="TOY_NOT_STARTED", reasons=reasons, attempt=n)
    if with_sidecar:
        HE.start_sidecar(os.getpid(), os.path.join(probe_dir, "LOAD_SAMPLES_toy_attempt%d.txt" % n),
                         interval=S.PROBE_SIDECAR_INTERVAL_S)                 # dahill req (3), stricter: toy too
    try:
        main(out)
    except PF.PreflightCapExceeded as ex:
        GT.append(led, "attempt_end", attempt=n, outcome="TOY_CAP_EXCEEDED", reason=str(ex), host=host, pinning=pin,
                  process_cpu_s=time.process_time())
        return dict(outcome="TOY_CAP_EXCEEDED", attempt=n)
    with open(out) as f:
        res = json.load(f)
    ok = bool(res.get("preflight_within_cap")) and res.get("projection_cpu_s", {}).get("fits") is not None
    outcome = "TOY_OK" if ok else "TOY_CAP_EXCEEDED"
    GT.append(led, "attempt_end", attempt=n, outcome=outcome, file=os.path.basename(out), sha256=S.sha256_file(out),
              host=host, pinning=pin, toy_fits_cpu_s=res.get("projection_cpu_s", {}).get("fits"),
              preflight_cpu_s_used=res.get("preflight_cpu_s_used"), process_cpu_s=time.process_time())
    return dict(outcome=outcome, attempt=n, file=out)


if __name__ == "__main__":
    attempt_main()
