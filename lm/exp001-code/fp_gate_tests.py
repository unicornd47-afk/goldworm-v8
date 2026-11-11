#!/usr/bin/env python3
"""Negative tests for the torch FP gate (fail-closed).

Each of NaN, sequence-count mismatch, and shape mismatch must FAIL the gate.
A finite identity pair must pass. Writes results/fp_gate_negative_tests.json.
Not a training arm.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import FP32_TOL_MAX_ABS, RESULTS
from grow import fp_gate_torch, max_abs_logit_diff
from model_io import empty_v8, unique_params
from grow import grow_depth, load_fp_sequences


def _run(name: str, fn) -> dict:
    rec = {"name": name, "raised": False, "error": None, "stats": None}
    try:
        rec["stats"] = fn()
        rec["raised"] = False
    except Exception as e:
        rec["raised"] = True
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback_tail"] = traceback.format_exc()[-800:]
    return rec


def test_nan_must_fail() -> dict:
    a = [torch.zeros(1, 4, 8)]
    b = [torch.zeros(1, 4, 8)]
    b[0][0, 0, 0] = float("nan")
    return max_abs_logit_diff(a, b)


def test_inf_must_fail() -> dict:
    a = [torch.zeros(1, 4, 8)]
    b = [torch.zeros(1, 4, 8)]
    b[0][0, 0, 0] = float("inf")
    return max_abs_logit_diff(a, b)


def test_length_mismatch_must_fail() -> dict:
    a = [torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)]
    b = [torch.zeros(1, 4, 8)]
    return max_abs_logit_diff(a, b)


def test_shape_mismatch_must_fail() -> dict:
    a = [torch.zeros(1, 4, 8)]
    b = [torch.zeros(1, 5, 8)]
    return max_abs_logit_diff(a, b)


def test_empty_must_fail() -> dict:
    return max_abs_logit_diff([], [])


def test_identity_finite_must_pass() -> dict:
    x = torch.randn(1, 4, 8)
    return max_abs_logit_diff([x], [x.clone()])


def test_fp_gate_identity_tiny() -> dict:
    """End-to-end gate on a proxy-size random model, 2→4 identity grow, cropped to ctx."""
    torch.manual_seed(0)
    orig = empty_v8(n_layer=2, d_model=64, n_head=4, ctx=256)
    grown = empty_v8(n_layer=2, d_model=64, n_head=4, ctx=256)
    grown.load_state_dict(orig.state_dict())
    grow_depth(grown, 4, mode="identity_copy_inproj", seed=0)
    seqs = [s[:256] for s in load_fp_sequences()]
    return fp_gate_torch(orig, grown, seqs, "cpu", tol=FP32_TOL_MAX_ABS)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = [
        ("nan", test_nan_must_fail, True),
        ("inf", test_inf_must_fail, True),
        ("length_mismatch", test_length_mismatch_must_fail, True),
        ("shape_mismatch", test_shape_mismatch_must_fail, True),
        ("empty_lists", test_empty_must_fail, True),
        ("identity_finite", test_identity_finite_must_pass, False),
        ("fp_gate_identity_tiny_2to4", test_fp_gate_identity_tiny, False),
    ]
    rows = []
    all_ok = True
    for name, fn, must_raise in cases:
        rec = _run(name, fn)
        rec["must_raise"] = must_raise
        if must_raise:
            rec["ok"] = bool(rec["raised"]) and "FP_GATE_FAIL" in (rec["error"] or "")
            rec["expected"] = "FAIL the gate (raise FP_GATE_FAIL)"
        else:
            passed = (not rec["raised"]) and bool((rec["stats"] or {}).get("pass", True) if "pass" in (rec["stats"] or {}) else True)
            if rec["stats"] and "max_abs_logit_diff" in rec["stats"]:
                passed = passed and (rec["stats"]["max_abs_logit_diff"] <= FP32_TOL_MAX_ABS)
            rec["ok"] = passed
            rec["expected"] = "PASS"
        if not rec["ok"]:
            all_ok = False
        rows.append(rec)
        print(f"{name}: ok={rec['ok']} raised={rec['raised']} err={rec['error']}")

    # Document the Python-max fail-open pitfall that this gate must not use.
    py_max_nan = max(0.0, float("nan"))
    report = {
        "tolerance": FP32_TOL_MAX_ABS,
        "fail_closed": True,
        "python_max_0_nan": py_max_nan,  # 0.0 — why Python max must not be used
        "python_max_is_fail_open_on_nan": py_max_nan == 0.0,
        "cases": rows,
        "PASS": all_ok,
        "n_params_tiny": unique_params(empty_v8(n_layer=2, d_model=64, n_head=4, ctx=256)),
    }
    out = RESULTS / "fp_gate_negative_tests.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out, "PASS", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
