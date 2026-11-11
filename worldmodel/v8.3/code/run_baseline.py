"""v8.3 ERRATUM E2.4 entry point: one numbered 10-minute load1 baseline attempt on THIS host (no experiment running),
before the toy attempt. Then fix the threshold in the RELEASE draft:  python set_start_threshold.py <BASELINE file>
Usage: WM_ROOT=<dir holding v8.1 v8.2 v8.3> WM_AUDIT_ROOT=<audit root> python <WM_ROOT>/v8.3/code/run_baseline.py"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v81 import baseline as BL  # noqa: E402

if __name__ == "__main__":
    BL.run_baseline()
