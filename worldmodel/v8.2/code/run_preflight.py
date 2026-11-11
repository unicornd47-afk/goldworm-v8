"""Toy preflight entry point (CPU only, seeds 1-4, cap 120 CPU-s). Writes ../preflight/PREFLIGHT_PROJECTION.json.
Does NOT start the production run and touches no test/tuning/training/adaptation seed."""
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


def main():
    hashes = S.verify_plan_hashes()                       # STOP on mismatch (raises)
    setup_cpu = time.process_time()                      # imports + hash verification = overhead proxy
    out = RL.PREFLIGHT_FILE                                # v8.2: v8.2/PREFLIGHT_PROJECTION.json
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


if __name__ == "__main__":
    main()
