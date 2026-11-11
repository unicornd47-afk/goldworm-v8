# Dynamic harness results

REJECT

Historical v3 cache reused by exact body key; no newly executed hidden tests. v3 new_16 frozen selector includes its original heal step.

## eval
Means: {"raw": 0.07083333333333333, "first": 0.1, "oracle16": 0.16666666666666666, "v3": 0.16666666666666666, "selected": 0.13333333333333333, "n_tests": 4.4, "discriminating_tests": 1.6666666666666667}
vs_raw: {"base": 0.07083333333333333, "new": 0.13333333333333333, "diff": 0.0625, "ci95": [0.0, 0.14375]}
vs_v3: {"base": 0.16666666666666666, "new": 0.13333333333333333, "diff": -0.03333333333333333, "ci95": [-0.1, 0.0]}
vs_first: {"base": 0.1, "new": 0.13333333333333333, "diff": 0.03333333333333333, "ci95": [0.0, 0.1]}
Diagnostics: {"n": 30, "fallback_tasks": 19, "tasks_with_tests": 19, "tasks_with_discrimination": 11, "runner_errors": 0}

## eval2
Means: {"raw": 0.004166666666666667, "first": 0.0, "oracle16": 0.06666666666666667, "v3": 0.0, "selected": 0.0, "n_tests": 4.533333333333333, "discriminating_tests": 0.4}
vs_raw: {"base": 0.004166666666666667, "new": 0.0, "diff": -0.004166666666666667, "ci95": [-0.010416666666666666, 0.0]}
vs_v3: {"base": 0.0, "new": 0.0, "diff": 0.0, "ci95": [0.0, 0.0]}
vs_first: {"base": 0.0, "new": 0.0, "diff": 0.0, "ci95": [0.0, 0.0]}
Diagnostics: {"n": 30, "fallback_tasks": 25, "tasks_with_tests": 21, "tasks_with_discrimination": 5, "runner_errors": 0}

## pooled
Means: {"raw": 0.0375, "first": 0.05, "oracle16": 0.11666666666666667, "v3": 0.08333333333333333, "selected": 0.06666666666666667, "n_tests": 4.466666666666667, "discriminating_tests": 1.0333333333333334}
vs_raw: {"base": 0.0375, "new": 0.06666666666666667, "diff": 0.029166666666666667, "ci95": [-0.0020833333333333333, 0.07291666666666667]}
vs_v3: {"base": 0.08333333333333333, "new": 0.06666666666666667, "diff": -0.016666666666666666, "ci95": [-0.05, 0.0]}
vs_first: {"base": 0.05, "new": 0.06666666666666667, "diff": 0.016666666666666666, "ci95": [0.0, 0.05]}
Diagnostics: {"n": 60, "fallback_tasks": 44, "tasks_with_tests": 40, "tasks_with_discrimination": 16, "runner_errors": 0}

Gates: {"G0": "PARTIAL: exact archive reproduction and current hash verified; original immutable checkpoint hash unavailable", "G1": false, "G2": "PASS procedural isolation: teacher and candidate runners lack hidden-test/label mounts; suite/index hashes frozen before cache scoring. Does not rule out pretraining contamination."}

Limitations: tiny repeated evaluation sets; noisy teacher oracles; no guarantee that consensus is correctness. No training or publication.