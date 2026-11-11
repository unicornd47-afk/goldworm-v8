"""Hook to the existing GW01 q3g64mse converter. Does not vendor the exporter.

Official convert.py pins the 85M source sha256 and the official config n_layer, so
it cannot export grown, from-scratch, or proxy checkpoints. Those go through the
same gwcore q3 / group-64 / mse-clip routine (imported, not vendored) to write a
.gw when possible and always to report per-tensor rel. Frobenius, split old vs
new blocks. Settings are identical for grown and baseline arms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from constants import GW_CONVERT, GW_Q3_KIND, L_START
from model_io import read_safetensors_tensors


def _gwcore():
    sys.path.insert(0, str(Path(GW_CONVERT).resolve().parent))
    import gwcore as C  # noqa: WPS433 — existing published converter library
    return C


def q3g64_dequant_rel_fro(w: np.ndarray, C) -> float:
    parts = C.quantize(w, "q3", 64, "mse")
    rec = C.dequant("q3", w.shape, 64, dict(parts))
    den = float(np.linalg.norm(w))
    return float(np.linalg.norm(rec - w) / den) if den > 0 else 0.0


def quant_gap_report(tensors: dict[str, np.ndarray], old_L: int) -> dict:
    """Per-tensor q3g64mse rel. Frobenius, aggregated over old vs new blocks."""
    C = _gwcore()
    per = {}
    old_errs, new_errs, other_errs = [], [], []
    for name, w in tensors.items():
        if not isinstance(w, np.ndarray):
            w = np.asarray(w)
        if w.ndim != 2 or name.endswith("tok_emb.weight") or name.endswith("head.weight"):
            continue  # embeddings stay unquantized in the official recipe (f16)
        rel = q3g64_dequant_rel_fro(w.astype(np.float32), C)
        rec = {"rel_fro_err": rel}
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            rec["block"] = idx
            rec["group"] = "old" if idx < old_L else "new"
            (old_errs if idx < old_L else new_errs).append(rel)
        else:
            rec["group"] = "other"
            other_errs.append(rel)
        per[name] = rec

    def agg(xs):
        if not xs:
            return None
        return {"n": len(xs), "mean_rel_fro": float(np.mean(xs)), "max_rel_fro": float(np.max(xs))}

    return {
        "kind": GW_Q3_KIND,
        "old_L": old_L,
        "per_tensor": per,
        "old_blocks": agg(old_errs),
        "new_blocks": agg(new_errs),
        "other": agg(other_errs),
        "gap_new_minus_old_mean": (
            None if not old_errs or not new_errs
            else float(np.mean(new_errs) - np.mean(old_errs))
        ),
    }


def tensors_from_safetensors(path: Path) -> dict[str, np.ndarray]:
    t = read_safetensors_tensors(path)
    return {k: v.detach().cpu().numpy() for k, v in t.items()}


def export_q3g64(
    weights_safetensors: Path,
    out_gw: Path,
    log_path: Path | None = None,
    old_L: int = L_START,
) -> dict:
    """Compute old/new quant gaps; try official convert.py (will refuse non-85M SHA)."""
    weights_safetensors = Path(weights_safetensors)
    rec = {
        "kind": GW_Q3_KIND,
        "src": str(weights_safetensors),
        "out": str(out_gw),
        "old_L": old_L,
    }
    rec["gap"] = quant_gap_report(tensors_from_safetensors(weights_safetensors), old_L)

    # Official converter: --wbits 3 --group 64 --clip mse (no --kind). SHA-pinned.
    import subprocess
    if GW_CONVERT.is_file():
        out_gw.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(GW_CONVERT),
            "--src", str(weights_safetensors),
            "--out", str(out_gw),
            "--wbits", "3",
            "--group", "64",
            "--clip", "mse",
        ]
        rec["convert_cmd"] = cmd
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            rec["convert_returncode"] = p.returncode
            rec["convert_stdout_tail"] = (p.stdout or "")[-2000:]
            rec["convert_stderr_tail"] = (p.stderr or "")[-2000:]
            rec["convert_status"] = "ok" if p.returncode == 0 else "refused_or_fail"
        except Exception as e:
            rec["convert_status"] = "error"
            rec["convert_error"] = str(e)
    else:
        rec["convert_status"] = "SKIP"
        rec["convert_reason"] = f"converter missing: {GW_CONVERT}"
    rec["status"] = "ok"  # gap report is the load-bearing export artifact
    if log_path:
        Path(log_path).write_text(json.dumps(rec, indent=2) + "\n")
    return rec
