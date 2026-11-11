# MBR-EXEC-style cached-v8 experiment
REJECT
Same historical v3 evaluator caches keyed by exact body; no fresh hidden-test execution. v3 baseline retains heal. Reused evals, uncertain inferred types/domains; no oracle-correctness guarantee from agreement. Historical cache lacks original immutable checkpoint SHA. This is a simplified MBR-EXEC-style largest-output-cluster selector, not an exact reproduction of all Shi et al. settings.

## eval
{
  "n": 30,
  "means": {
    "raw": 0.07083333333333333,
    "v3": 0.16666666666666666,
    "oracle": 0.16666666666666666,
    "mbr": 0.1,
    "visible": 0.1
  },
  "diagnostics": {
    "nontrivial_cluster": 22,
    "discriminating_cluster": 21,
    "any_output_discrimination": 25,
    "fallback": 1,
    "has_visible": 1,
    "runner_errors": 0
  },
  "mbr_vs_raw": {
    "diff": 0.029166666666666667,
    "ci95": [
      -0.00625,
      0.07916666666666666
    ]
  },
  "mbr_vs_v3": {
    "diff": -0.06666666666666667,
    "ci95": [
      -0.16666666666666666,
      0.0
    ]
  },
  "G1_mbr": false,
  "visible_vs_raw": {
    "diff": 0.029166666666666667,
    "ci95": [
      -0.00625,
      0.07916666666666666
    ]
  },
  "visible_vs_v3": {
    "diff": -0.06666666666666667,
    "ci95": [
      -0.16666666666666666,
      0.0
    ]
  },
  "G1_visible": false
}

## eval2
{
  "n": 30,
  "means": {
    "raw": 0.004166666666666667,
    "v3": 0.0,
    "oracle": 0.06666666666666667,
    "mbr": 0.0,
    "visible": 0.0
  },
  "diagnostics": {
    "nontrivial_cluster": 22,
    "discriminating_cluster": 20,
    "any_output_discrimination": 25,
    "fallback": 1,
    "has_visible": 3,
    "runner_errors": 0
  },
  "mbr_vs_raw": {
    "diff": -0.004166666666666667,
    "ci95": [
      -0.010416666666666666,
      0.0
    ]
  },
  "mbr_vs_v3": {
    "diff": 0.0,
    "ci95": [
      0.0,
      0.0
    ]
  },
  "G1_mbr": false,
  "visible_vs_raw": {
    "diff": -0.004166666666666667,
    "ci95": [
      -0.010416666666666666,
      0.0
    ]
  },
  "visible_vs_v3": {
    "diff": 0.0,
    "ci95": [
      0.0,
      0.0
    ]
  },
  "G1_visible": false
}

## pooled
{
  "n": 60,
  "means": {
    "raw": 0.0375,
    "v3": 0.08333333333333333,
    "oracle": 0.11666666666666667,
    "mbr": 0.05,
    "visible": 0.05
  },
  "diagnostics": {
    "nontrivial_cluster": 44,
    "discriminating_cluster": 41,
    "any_output_discrimination": 50,
    "fallback": 2,
    "has_visible": 4,
    "runner_errors": 0
  },
  "mbr_vs_raw": {
    "diff": 0.0125,
    "ci95": [
      -0.005208333333333333,
      0.03854166666666667
    ]
  },
  "mbr_vs_v3": {
    "diff": -0.03333333333333333,
    "ci95": [
      -0.08333333333333333,
      0.0
    ]
  },
  "G1_mbr": false,
  "visible_vs_raw": {
    "diff": 0.0125,
    "ci95": [
      -0.005208333333333333,
      0.03854166666666667
    ]
  },
  "visible_vs_v3": {
    "diff": -0.03333333333333333,
    "ci95": [
      -0.08333333333333333,
      0.0
    ]
  },
  "G1_visible": false
}

Gates: {"G1_primary_pooled": false, "G1_visible_pooled": false, "G2": "Procedural PASS: selector namespace has no hidden tests or labels; selection lock written before separate historical-cache scoring."}