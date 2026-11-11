"""v8.3 dahill req (5) / ERRATUM E2.4: fix the start-rule threshold (1.5 x p95 of a BASELINE_OK attempt) in
RELEASE_v8.3_DRAFT.json once, before the toy attempt and probe attempt 3. Refuses if already set.
Usage: WM_ROOT=... python <WM_ROOT>/v8.3/code/set_start_threshold.py <WM_ROOT>/v8.3/probe/BASELINE_attempt<N>.json"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v81 import baseline as BL  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(BL.set_start_threshold(os.path.abspath(sys.argv[1])), indent=1))
