"""Production entry point. Checks the start rule FIRST; if any condition is missing it writes
NOT_LAUNCHED and exits without generating any data. One run only: refuses if run_output exists."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

torch.set_num_threads(1)
from v81 import pipeline as PP  # noqa: E402
from v81 import release as RL  # noqa: E402
from v81 import spec as S  # noqa: E402


def main(release_path=None, out_dir=None):
    release_path = release_path or os.path.join(S.PLAN_DIR, "RELEASE_v8.1.json")
    out_dir = out_dir or os.path.join(S.PLAN_DIR, "run_output")
    ok, reasons, rel = RL.check_start(release_path)
    if not ok:
        st = dict(outcome="NOT_LAUNCHED", reasons=reasons)
        nl = os.path.join(os.path.dirname(out_dir), "NOT_LAUNCHED_%s.json" % os.path.basename(out_dir))
        with open(nl, "w") as f:
            json.dump(st, f, indent=1)
        print(json.dumps(st, indent=1))
        return st
    if os.path.exists(out_dir):
        raise SystemExit("run_output exists: one production run only (no retry, no resume)")
    design = PP.Design().apply_ladder(rel["preflight"].get("ladder_applied", []))
    bound = {os.path.join(S.PLAN_DIR, k): v for k, v in S.BOUND_FILES.items()}
    bound[os.path.join(S.CODE_DIR, RL.CODE_MANIFEST)] = rel["code_hashes"]["manifest_sha256"]
    run = PP.Run(design, out_dir, release_hashes=bound)
    final = run.run_all()
    print(json.dumps(dict(outcome=final["outcome"]), indent=1))
    return final


if __name__ == "__main__":
    main()
