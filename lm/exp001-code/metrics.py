"""Primary bpb (256-byte chunks), 2048-context secondary, divergence, knockout."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from constants import (
    COSINE_TO_SELF0_MAX, DATA, DEV_NPY_SHA256, HELDOUT_SHA256,
    KNOCKOUT_BYTES, KNOCKOUT_DEV_WINDOWS, KNOCKOUT_MIN_BPB,
    PRODUCTIVE_FRACTION, RMS_FLOOR, RMS_RATIO, TRAIN_T,
)
from model_io import sha256_file


LN2 = math.log(2.0)


def _require_sha(path: Path, expected: str, label: str):
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"{label} sha256 mismatch: {got} != {expected}")


def load_heldout_windows() -> tuple[list[bytes], list[tuple[str, str]]]:
    path = DATA / "heldout_exp001.bin"
    _require_sha(path, HELDOUT_SHA256, "heldout")
    blob = path.read_bytes()
    windows = [blob[i:i + 2048] for i in range(0, len(blob), 2048)]
    import json
    man = json.loads((DATA / "heldout_exp001.manifest.json").read_text())
    files = [(w["package"], w["path"]) for w in man["windows"]]
    assert len(files) == len(windows)
    return windows, files


def load_dev_windows() -> np.ndarray:
    path = DATA / "dev_exp001.npy"
    _require_sha(path, DEV_NPY_SHA256, "dev")
    return np.load(path)  # (N, 2049) uint8


def bits_per_byte(nll_nats: float, n_bytes: int) -> float:
    if n_bytes <= 0:
        return float("nan")
    return (nll_nats / n_bytes) / LN2


@torch.no_grad()
def nll_on_sequence(model, raw: bytes, device, ctx_len: int | None = None) -> tuple[float, int]:
    """Teacher-forcing NLL (nats) over predicted bytes. If ctx_len is set, crop."""
    ids = list(raw)
    if ctx_len is not None:
        ids = ids[:ctx_len]
    if len(ids) < 2:
        return 0.0, 0
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x[:, :-1])
    target = x[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="sum"
    )
    return float(loss.item()), int(target.numel())


@torch.no_grad()
def eval_bpb_256(model, windows: list[bytes], device) -> dict:
    """Primary: non-overlapping 256-byte chunks, teacher-forcing inside each chunk."""
    per_chunk = []
    per_window = []
    total_nll = 0.0
    total_n = 0
    for wi, w in enumerate(windows):
        w_nll = 0.0
        w_n = 0
        chunks = [w[i:i + 256] for i in range(0, len(w) // 256 * 256, 256)]
        chunk_bpb = []
        for ch in chunks:
            nll, n = nll_on_sequence(model, ch, device)
            total_nll += nll
            total_n += n
            w_nll += nll
            w_n += n
            chunk_bpb.append(bits_per_byte(nll, n) if n else float("nan"))
        per_chunk.extend(chunk_bpb)
        per_window.append(bits_per_byte(w_nll, w_n) if w_n else float("nan"))
    return {
        "bpb": bits_per_byte(total_nll, total_n),
        "n_pred_bytes": total_n,
        "per_window_bpb": per_window,
        "per_chunk_bpb": per_chunk,
    }


@torch.no_grad()
def eval_bpb_2048(model, windows: list[bytes], device) -> dict:
    """Descriptive secondary: full-window 2048-context teacher forcing."""
    total_nll = 0.0
    total_n = 0
    per_window = []
    pos_nll = None
    pos_n = None
    for w in windows:
        nll, n = nll_on_sequence(model, w, device)
        total_nll += nll
        total_n += n
        per_window.append(bits_per_byte(nll, n) if n else float("nan"))
        # position curve for S-above-256 reporting
        ids = torch.tensor(list(w), dtype=torch.long, device=device).unsqueeze(0)
        logits = model(ids[:, :-1])
        logp = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none"
        )
        if pos_nll is None:
            pos_nll = logp.double().cpu()
            pos_n = torch.ones_like(pos_nll)
        else:
            L = min(pos_nll.numel(), logp.numel())
            pos_nll[:L] += logp.double().cpu()[:L]
            pos_n[:L] += 1
    pos_bpb = (pos_nll / pos_n / LN2).tolist() if pos_nll is not None else []
    return {
        "bpb": bits_per_byte(total_nll, total_n),
        "n_pred_bytes": total_n,
        "per_window_bpb": per_window,
        "per_position_bpb": pos_bpb,
        "mean_bpb_pos_gt_256": float(sum(pos_bpb[256:]) / max(1, len(pos_bpb[256:]))) if len(pos_bpb) > 256 else None,
    }


@torch.no_grad()
def capture_block_traces(model, sequences: list[bytes], device) -> list[dict]:
    """For each block, pre-outproj activations (attn wo input, mlp fc2 input) and residual branch outputs.

    Projection-invariant cosine uses the pre-outproj tensors. Residual branch is wo(x)+fc2(x).
    """
    model.eval()
    handles = []
    bag = {i: {"pre_wo": [], "pre_fc2": [], "wo_out": [], "fc2_out": []} for i in range(len(model.blocks))}

    def make_hooks(i, blk):
        def hook_wo(_m, inp, out):
            bag[i]["pre_wo"].append(inp[0].detach().float().cpu())
            bag[i]["wo_out"].append(out.detach().float().cpu())
        def hook_fc2(_m, inp, out):
            bag[i]["pre_fc2"].append(inp[0].detach().float().cpu())
            bag[i]["fc2_out"].append(out.detach().float().cpu())
        handles.append(blk.attn.wo.register_forward_hook(hook_wo))
        handles.append(blk.mlp.fc2.register_forward_hook(hook_fc2))

    for i, blk in enumerate(model.blocks):
        make_hooks(i, blk)
    try:
        for raw in sequences:
            ids = torch.tensor(list(raw), dtype=torch.long, device=device).unsqueeze(0)
            model(ids)
    finally:
        for h in handles:
            h.remove()

    traces = []
    for i in range(len(model.blocks)):
        pre_wo = torch.cat(bag[i]["pre_wo"], 1) if bag[i]["pre_wo"] else torch.zeros(1, 0)
        pre_fc2 = torch.cat(bag[i]["pre_fc2"], 1) if bag[i]["pre_fc2"] else torch.zeros(1, 0)
        wo_out = torch.cat(bag[i]["wo_out"], 1) if bag[i]["wo_out"] else torch.zeros(1, 0)
        fc2_out = torch.cat(bag[i]["fc2_out"], 1) if bag[i]["fc2_out"] else torch.zeros(1, 0)
        residual = wo_out + fc2_out
        traces.append({
            "pre_wo": pre_wo, "pre_fc2": pre_fc2,
            "residual": residual,
        })
    return traces


def _cos_rms(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Mean cosine over positions (last dim is features) and RMS of a, b, a-b."""
    a = a.reshape(-1, a.shape[-1]).double()
    b = b.reshape(-1, b.shape[-1]).double()
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    an = a.norm(dim=-1).clamp_min(1e-12)
    bn = b.norm(dim=-1).clamp_min(1e-12)
    cos = float((a * b).sum(-1).div(an * bn).mean())
    return {
        "cosine": cos,
        "rms_a": float(a.pow(2).mean().sqrt()),
        "rms_b": float(b.pow(2).mean().sqrt()),
        "rms_diff": float((a - b).pow(2).mean().sqrt()),
        "n_pos": int(n),
    }


def divergence_report(traces, traces_ref, sources: list[int], old_L: int, tag: str) -> dict:
    """Per new block: cosine/rms vs mapped source and vs traces_ref (step 0) if given."""
    blocks = []
    for t, src in enumerate(sources):
        i = old_L + t
        rec = {"new_index": i, "source_index": src, "tag": tag}
        rec["vs_source_residual"] = _cos_rms(traces[i]["residual"], traces[src]["residual"])
        rec["vs_source_pre_wo"] = _cos_rms(traces[i]["pre_wo"], traces[src]["pre_wo"])
        rec["vs_source_pre_fc2"] = _cos_rms(traces[i]["pre_fc2"], traces[src]["pre_fc2"])
        if traces_ref is not None:
            rec["vs_self0_residual"] = _cos_rms(traces[i]["residual"], traces_ref[i]["residual"])
            rec["vs_self0_pre_wo"] = _cos_rms(traces[i]["pre_wo"], traces_ref[i]["pre_wo"])
            rec["vs_self0_pre_fc2"] = _cos_rms(traces[i]["pre_fc2"], traces_ref[i]["pre_fc2"])
            # RMS test on the learned update of the residual branch vs step-0 (noise-scale) output
            d = rec["vs_self0_residual"]
            rec["rms_test_pass"] = d["rms_diff"] >= max(RMS_RATIO * d["rms_b"], RMS_FLOOR)
        else:
            rec["rms_test_pass"] = False  # step 0 itself must not count as productive
        blocks.append(rec)
    return {"tag": tag, "blocks": blocks}


def knockout_dev_bpb(model, device, new_indices: list[int]) -> dict:
    """Per-new-block knockout on a frozen DEV subset (first K windows, first 256 bytes).

    A block passes if zeroing its residual contribution raises bpb by >= KNOCKOUT_MIN_BPB.
    """
    dev = load_dev_windows()
    k = min(KNOCKOUT_DEV_WINDOWS, len(dev))
    seqs = [bytes(dev[i, :KNOCKOUT_BYTES].tolist()) for i in range(k)]

    def bpb_now():
        r = eval_bpb_256(model, seqs, device)
        return r["bpb"], r["n_pred_bytes"]

    base, n = bpb_now()
    per = []
    # temporarily zero wo and fc2 of one new block
    for i in new_indices:
        saved = (
            model.blocks[i].attn.wo.weight.data.clone(),
            model.blocks[i].mlp.fc2.weight.data.clone(),
        )
        with torch.no_grad():
            model.blocks[i].attn.wo.weight.zero_()
            model.blocks[i].mlp.fc2.weight.zero_()
        ko, _ = bpb_now()
        with torch.no_grad():
            model.blocks[i].attn.wo.weight.copy_(saved[0])
            model.blocks[i].mlp.fc2.weight.copy_(saved[1])
        delta = ko - base
        per.append({
            "new_index": i,
            "base_bpb": base,
            "knockout_bpb": ko,
            "delta_bpb": delta,
            "pass": delta >= KNOCKOUT_MIN_BPB,
        })
    return {
        "base_bpb": base,
        "n_pred_bytes": n,
        "n_windows": k,
        "chunk_bytes": KNOCKOUT_BYTES,
        "min_delta": KNOCKOUT_MIN_BPB,
        "blocks": per,
    }


def productive_blocks(div_at_n, knockout, sources, old_L) -> dict:
    """A new block is productive only if RMS test (vs step 0) AND knockout both pass.

    A gate that would pass at step 0 does not count: rms_test_pass is False at step 0
    by construction, and this function is only applied to the N_div report.
    """
    ko = {b["new_index"]: b for b in knockout["blocks"]}
    rows = []
    for b in div_at_n["blocks"]:
        i = b["new_index"]
        rms_ok = bool(b.get("rms_test_pass"))
        ko_ok = bool(ko[i]["pass"]) if i in ko else False
        rows.append({
            "new_index": i,
            "source_index": b["source_index"],
            "rms_test_pass": rms_ok,
            "knockout_pass": ko_ok,
            "knockout_delta_bpb": ko[i]["delta_bpb"] if i in ko else None,
            "rms_diff_vs_step0": b.get("vs_self0_residual", {}).get("rms_diff"),
            "productive": rms_ok and ko_ok,
        })
    n = len(rows)
    frac = sum(r["productive"] for r in rows) / n if n else 0.0
    return {
        "blocks": rows,
        "n_new": n,
        "n_productive": int(sum(r["productive"] for r in rows)),
        "fraction": frac,
        "arm_pass": frac >= PRODUCTIVE_FRACTION,
        "threshold": PRODUCTIVE_FRACTION,
        "note": "step-0 satisfaction is not counted; rms test is vs step-0 residual, not vs source.",
    }
