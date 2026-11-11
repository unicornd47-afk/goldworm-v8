#!/usr/bin/env python3
"""Single CLI entry point for exp001 arms and the proxy-size smoke test.

This frozen module is exactly what runs — no post-freeze edits.

  python run_arm.py --smoke-proxy
      Tiny end-to-end path on the proxy-size model (seconds). NOT an arm.

  python run_arm.py --arm G|C|S|G_zeroall|G_signflip --ratio A|B|proxy \\
      --seed 11 --i-understand-this-starts-an-arm
      Real arms (also --arm proxy_pretrain). Locked to box CPU / fp32.

Device/precision: EXP001_DEVICE=cpu, EXP001_DTYPE=fp32. CUDA or bf16 is refused.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (
    CPU_BOX_BUDGETS, DATA, DEFAULT_PILOT_CAP_S, EVAL_RESERVE_S, EXP001_DEVICE,
    EXP001_DTYPE, FP32_TOL_MAX_ABS, GROWTH_FP_RESERVE_S, L_RATIO_A, L_RATIO_B,
    L_START, N_DIV_FRAC, N_DIV_MAX, N_DIV_MIN, NOISE_SIGMA, PREFLIGHT_ABORT_FRAC,
    PROXY_B1_BYTES, PROXY_D, PROXY_HEADS, PROXY_L0, PROXY_L1, PROXY_N,
    PROXY_PRETRAIN_BYTES, PROXY_S_BYTES, RESULTS, SEEDS, THREAD_COUNT, TRAIN_T,
    WEIGHTS,
)
from export_hook import export_q3g64
from grow import apply_noise_new_blocks, fp_gate_torch, grow_depth, load_fp_sequences
from metrics import (
    capture_block_traces, divergence_report, eval_bpb_2048, eval_bpb_256,
    knockout_dev_bpb, load_heldout_windows, productive_blocks,
)
from model_io import empty_v8, load_state_dict_fail_closed, load_v8, unique_params, write_safetensors
from train import flops_fwd_bwd, train_loop


def n_div_for(steps: int) -> int:
    return min(N_DIV_MAX, max(N_DIV_MIN, int(N_DIV_FRAC * steps)))


def lock_runtime(device: str, dtype: torch.dtype = torch.float32) -> str:
    if str(device) != EXP001_DEVICE:
        raise RuntimeError(
            f"exp001 is locked to {EXP001_DEVICE} / {EXP001_DTYPE}; "
            f"got device={device}. GPU and bf16 are out of scope for exp001 "
            "(a new prereg with a measured identity baseline is required)."
        )
    if dtype != torch.float32:
        raise RuntimeError(
            f"exp001 is locked to {EXP001_DTYPE}; got dtype={dtype}. "
            "bf16 is out of scope for exp001."
        )
    os.environ["OMP_NUM_THREADS"] = str(THREAD_COUNT)
    os.environ["MKL_NUM_THREADS"] = str(THREAD_COUNT)
    torch.set_num_threads(THREAD_COUNT)
    return "cpu"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k not in ("pre_wo", "pre_fc2", "residual")}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (torch.Tensor,)):
        return None
    return obj


def crop_seqs(seqs: list[bytes], ctx: int) -> list[bytes]:
    return [s[:ctx] for s in seqs]


def eval_heldout(model, device) -> dict:
    windows, files = load_heldout_windows()
    ctx = int(model.cfg.ctx)
    primary = eval_bpb_256(model, windows, device)
    if ctx < 2048:
        cropped = [w[:ctx] for w in windows]
        secondary = eval_bpb_2048(model, cropped, device)
        secondary["note"] = f"cropped to model ctx={ctx}; not full 2048"
    else:
        secondary = eval_bpb_2048(model, windows, device)
    return {
        "primary_256": primary,
        "descriptive_2048": secondary,
        "n_windows": len(windows),
        "file_tags": [f"{p}/{Path(q).name}" for p, q in files],
    }


def load_g_log(g_outdir: Path) -> dict:
    path = Path(g_outdir) / "train_log.json"
    if not path.is_file():
        raise RuntimeError(f"G train log missing: {path}")
    return json.loads(path.read_text())


def outdir_for(arm: str, ratio: str, seed: int, smoke: bool) -> Path:
    if smoke:
        return RESULTS / "smoke_proxy" / arm
    return RESULTS / f"{arm}_{ratio}_seed{seed}"


def _make_empty(n_layer: int, d_model: int, n_head: int, ctx: int, seed: int):
    torch.manual_seed(seed)
    return empty_v8(n_layer=n_layer, d_model=d_model, n_head=n_head, ctx=ctx)


def load_or_empty(spec: dict):
    """Load official / checkpoint weights, or a random-init model for S / proxy."""
    device = spec["device"]
    if spec.get("weights") and Path(spec["weights"]).is_file():
        if spec.get("official"):
            m = load_v8(spec["weights"], n_layer=spec["load_L"], map_location=device)
        else:
            m = empty_v8(
                n_layer=spec["load_L"], d_model=spec["d_model"],
                n_head=spec["n_head"], ctx=spec["ctx"],
            )
            from model_io import read_safetensors_tensors
            tensors = read_safetensors_tensors(Path(spec["weights"]))
            # tied embeddings / pos_emb / rope_freq, then fail-closed on any
            # remaining missing or unexpected key.
            load_state_dict_fail_closed(m, tensors)
            m = m.to(device)
        return m
    return _make_empty(spec["load_L"], spec["d_model"], spec["n_head"], spec["ctx"], spec["seed"]).to(spec["device"])


def run_eval_export(model, spec: dict, outdir: Path, sources=None) -> dict:
    device = spec["device"]
    old_L = spec["old_L"]
    new_indices = list(range(old_L, len(model.blocks)))
    ev = eval_heldout(model, device)
    (outdir / "eval_heldout.json").write_text(json.dumps(_jsonable(ev), indent=2) + "\n")
    ko = None
    prod = None
    if new_indices:
        ko = knockout_dev_bpb(model, device, new_indices)
        (outdir / "knockout.json").write_text(json.dumps(ko, indent=2) + "\n")
    weights = outdir / "model.safetensors"
    write_safetensors(weights, model)
    gap = export_q3g64(weights, outdir / "model.q3g64.gw", outdir / "export_q3g64.json", old_L=old_L)
    return {"eval": ev, "knockout": ko, "export": gap, "productive": prod}


def arm_reserves_s(arm: str, apply_reserves: bool) -> float:
    """Growth arms: 300+600 s. C/S/pretrain: 600 s eval only. Smoke may disable."""
    if not apply_reserves:
        return 0.0
    r = float(EVAL_RESERVE_S)
    if arm in ("G", "G_zeroall", "G_signflip"):
        r += float(GROWTH_FP_RESERVE_S)
    return r


def planned_train_steps(spec: dict, n_params: int) -> int:
    if spec.get("max_steps") is not None:
        cap = int(spec["max_steps"])
    else:
        cap = 10**12
    if spec.get("target_bytes") is not None:
        return min(cap, max(1, int(spec["target_bytes"]) // TRAIN_T))
    if spec.get("target_flops") is not None:
        return min(cap, max(1, int(spec["target_flops"]) // max(1, 6 * n_params * TRAIN_T)))
    if spec.get("max_steps") is not None:
        return int(spec["max_steps"])
    raise RuntimeError("no planned steps: need target_bytes, target_flops, or max_steps")


def measure_step_s(n_layer: int, d_model: int, n_head: int, ctx: int, device, warmup=1, timed=3) -> dict:
    """A few fwd+bwd+AdamW steps on a throwaway same-size model. Does not touch the arm."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    m = empty_v8(n_layer=n_layer, d_model=d_model, n_head=n_head, ctx=ctx).to(device)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    def step():
        x = torch.randint(0, 256, (1, TRAIN_T + 1), device=device)
        opt.zero_grad(set_to_none=True)
        logits = m(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        return float(loss.item())
    rss_before = peak_rss_kb()
    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(timed):
        step()
    dt = time.perf_counter() - t0
    rss_after = peak_rss_kb()
    return {
        "sec_per_step_wall": dt / timed,
        "timed_steps": timed,
        "warmup_steps": warmup,
        "peak_rss_kb_before": rss_before,
        "peak_rss_kb_after": rss_after,
        "peak_rss_kb": rss_after,
    }


def peak_rss_kb() -> int:
    """Linux ru_maxrss is in kilobytes (process high-water mark)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def read_mem_available_kb(path: Path | str = "/proc/meminfo") -> int:
    """MemAvailable from /proc/meminfo, in kB. Missing field is fail-closed."""
    text = Path(path).read_text()
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) < 2:
                break
            return int(parts[1])
    raise PreflightAbort(
        "PREFLIGHT_ABORT MemAvailable missing from /proc/meminfo (fail-closed)."
    )


def project_peak_rss_kb(peak_rss_kb: int, n_params: int, n_layer: int,
                        d_model: int, n_head: int, ctx: int) -> dict:
    """Conservative full-run peak: measured preflight high-water plus extras.

    Extras (all in kB, integer arithmetic):
      - one extra parameter copy (growth / a second resident model);
      - eval attention scores for one layer at full ctx (n_head * ctx * ctx * 4 B);
      - four residual working buffers through depth (ctx * d_model * n_layer * 16 B).
    """
    param_kb = int((n_params * 4 + 1023) // 1024)
    attn_kb = int((n_head * ctx * ctx * 4 + 1023) // 1024)
    residual_kb = int((ctx * d_model * n_layer * 16 + 1023) // 1024)
    extras = param_kb + attn_kb + residual_kb
    projected = int(peak_rss_kb) + extras
    return {
        "peak_rss_kb": int(peak_rss_kb),
        "extra_param_copy_kb": param_kb,
        "eval_attn_kb": attn_kb,
        "eval_residual_kb": residual_kb,
        "projected_peak_rss_kb": projected,
    }


class PreflightAbort(RuntimeError):
    """Projected train+reserves or projected peak RAM exceeds 0.9 × cap. Fail-closed."""


def arm_preflight(spec: dict, n_layer: int, n_params: int, outdir: Path,
                  mem_available_kb: int | None = None) -> dict:
    """Measured per-arm preflight. Raises PreflightAbort (PREFLIGHT_ABORT) if over budget.

    Aborts if projected train wall plus reserves > 0.9 × wall_cap_s, or if the
    conservative projected peak RSS > 0.9 × MemAvailable. mem_available_kb may
    be injected for the negative test; otherwise it is read from /proc/meminfo.
    """
    apply = bool(spec.get("apply_reserves", not spec.get("smoke")))
    reserves = arm_reserves_s(spec["arm"], apply)
    wall = float(spec["wall_cap_s"])
    abort_line = PREFLIGHT_ABORT_FRAC * wall
    meas = measure_step_s(
        n_layer, spec["d_model"], spec["n_head"], spec["ctx"], spec["device"],
    )
    steps = planned_train_steps(spec, n_params)
    proj_train = steps * meas["sec_per_step_wall"]
    proj_total = proj_train + reserves
    if mem_available_kb is None:
        avail_kb = read_mem_available_kb()
        mem_source = "/proc/meminfo"
    else:
        avail_kb = int(mem_available_kb)
        mem_source = "injected"
    mem = project_peak_rss_kb(
        meas["peak_rss_kb"], n_params, n_layer,
        int(spec["d_model"]), int(spec["n_head"]), int(spec["ctx"]),
    )
    abort_line_kb = PREFLIGHT_ABORT_FRAC * avail_kb
    rec = {
        "arm": spec["arm"],
        "n_layer": n_layer,
        "n_params": n_params,
        "planned_steps": steps,
        "sec_per_step_wall": meas["sec_per_step_wall"],
        "projected_train_s": proj_train,
        "reserves_s": reserves,
        "projected_train_plus_reserves_s": proj_total,
        "wall_cap_s": wall,
        "abort_line_s": abort_line,
        "abort_frac": PREFLIGHT_ABORT_FRAC,
        "apply_reserves": apply,
        "peak_rss_kb_before": meas["peak_rss_kb_before"],
        "peak_rss_kb": mem["peak_rss_kb"],
        "extra_param_copy_kb": mem["extra_param_copy_kb"],
        "eval_attn_kb": mem["eval_attn_kb"],
        "eval_residual_kb": mem["eval_residual_kb"],
        "projected_peak_rss_kb": mem["projected_peak_rss_kb"],
        "mem_available_kb": avail_kb,
        "mem_available_source": mem_source,
        "abort_line_kb": abort_line_kb,
        "status": "ok",
    }
    outdir.mkdir(parents=True, exist_ok=True)
    reasons = []
    if proj_total > abort_line:
        reasons.append(
            f"projected train+reserves {proj_total:.2f}s > 0.9×cap {abort_line:.2f}s "
            f"(train {proj_train:.2f}s + reserves {reserves:.2f}s, cap {wall:.2f}s)"
        )
    if mem["projected_peak_rss_kb"] > abort_line_kb:
        reasons.append(
            f"projected peak RSS {mem['projected_peak_rss_kb']} kB > "
            f"0.9×MemAvailable {abort_line_kb:.2f} kB "
            f"(available {avail_kb} kB, source {mem_source})"
        )
    if reasons:
        rec["status"] = "PREFLIGHT_ABORT"
        rec["abort_reasons"] = reasons
        (outdir / "preflight.json").write_text(json.dumps(rec, indent=2) + "\n")
        raise PreflightAbort("PREFLIGHT_ABORT " + "; ".join(reasons) + ". Fail-closed.")
    (outdir / "preflight.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def test_ram_preflight_abort(outdir: Path | None = None) -> dict:
    """Negative test: fake a tiny MemAvailable; must PREFLIGHT_ABORT on RAM, not time."""
    outdir = Path(outdir or (RESULTS / "smoke_proxy" / "ram_preflight_abort"))
    rec = {"raised": False, "error": None, "status": None, "abort_reasons": None}
    try:
        arm_preflight(
            {
                "arm": "G", "d_model": PROXY_D, "n_head": PROXY_HEADS, "ctx": 256,
                "device": "cpu", "wall_cap_s": 1e9, "target_bytes": TRAIN_T,
                "apply_reserves": False, "smoke": True, "max_steps": 1,
            },
            n_layer=PROXY_L1,
            n_params=PROXY_N[4],
            outdir=outdir,
            mem_available_kb=1,  # 1 kB available → 0.9 kB abort line
        )
        rec["status"] = "FAILED_TO_ABORT"
    except PreflightAbort as e:
        rec["raised"] = True
        rec["error"] = str(e)
        rec["status"] = "PREFLIGHT_ABORT"
        rec["abort_reasons"] = getattr(e, "args", None)
        pf = outdir / "preflight.json"
        if pf.is_file():
            rec["preflight"] = json.loads(pf.read_text())
    rec["ok"] = (
        rec["raised"]
        and rec["status"] == "PREFLIGHT_ABORT"
        and rec["error"] is not None
        and "MemAvailable" in rec["error"]
        and "projected peak RSS" in rec["error"]
    )
    return rec


def resolve_cs_budget(spec: dict) -> tuple[int, int | None]:
    """C/S stop at G's logged FLOPs (K7). Bytes are only an upper bound. Never more than G."""
    g = load_g_log(spec["g_outdir"])
    g_flops = int(g["flops"])
    if (
        spec.get("arm") == "S"
        and spec.get("ratio") == "proxy"
        and not spec.get("smoke")
        and spec.get("pretrain_outdir")
    ):
        pre = json.loads((Path(spec["pretrain_outdir"]) / "train_log.json").read_text())
        target_flops = int(pre["flops"]) + g_flops
        bytes_upper = PROXY_S_BYTES
    else:
        target_flops = g_flops
        bytes_upper = spec.get("bytes_upper_bound")
    return target_flops, bytes_upper


def run_G(spec: dict) -> dict:
    """Grow, FP gate (before noise), noise, step-0 traces, train with N_div snapshot,
    divergence vs step 0, knockout, held-out eval, q3g64 gap."""
    device = spec["device"]
    outdir = Path(spec["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    old_L = spec["old_L"]
    new_L = spec["new_L"]
    mode = spec.get("mode", "identity_copy_inproj")
    original = load_or_empty({**spec, "load_L": old_L})
    grown = load_or_empty({**spec, "load_L": old_L})
    sources = grow_depth(grown, new_L, mode=mode, seed=spec["seed"])
    if mode == "identity_zero_all":
        with torch.no_grad():
            for i, blk in enumerate(grown.blocks[old_L:], start=old_L):
                for tensor in (blk.attn.wq.weight, blk.attn.wk.weight, blk.attn.wv.weight,
                               blk.mlp.fc1.weight, blk.attn.wo.weight, blk.mlp.fc2.weight):
                    if float(tensor.abs().max()) != 0.0:
                        raise RuntimeError(
                            f"K4: G_zeroall in-projection/out-projection not exactly zero "
                            f"at block {i} (fail-closed)"
                        )
    seqs = crop_seqs(load_fp_sequences(), int(grown.cfg.ctx))
    if mode != "signflip_copy":
        fp = fp_gate_torch(original, grown, seqs, device, tol=FP32_TOL_MAX_ABS)
    else:
        try:
            fp_gate_torch(original, grown, seqs, device, tol=FP32_TOL_MAX_ABS)
            fp = {"pass": True, "expected_fail": True, "skipped_abort": True, "unexpected_pass": True}
        except RuntimeError as e:
            fp = {"pass": False, "expected_fail": True, "skipped_abort": True, "error": str(e)}
    apply_noise_new_blocks(grown, old_L, sigma=NOISE_SIGMA, seed=1000 + spec["seed"])
    tr0 = capture_block_traces(grown, seqs, device)
    div0 = divergence_report(tr0, None, sources, old_L, tag="step0_post_noise")
    (outdir / "fp_gate.json").write_text(json.dumps(fp, indent=2) + "\n")
    (outdir / "div_step0.json").write_text(json.dumps(_jsonable(div0), indent=2) + "\n")

    if spec.get("target_bytes") is None and spec.get("max_steps") is None:
        raise RuntimeError("K2: G must get target_bytes from the registered budget")
    planned_steps = planned_train_steps(spec, unique_params(grown))
    n_div = n_div_for(planned_steps)
    if spec.get("n_div_override") is not None:
        n_div = int(spec["n_div_override"])
    snap = {"n_div": n_div, "captured": False}

    def cb(step, model, rec):
        if rec["step"] + 1 == n_div and not snap["captured"]:
            tr_n = capture_block_traces(model, seqs, device)
            divn = divergence_report(tr_n, tr0, sources, old_L, tag="N_div")
            (outdir / "div_Ndiv.json").write_text(json.dumps(_jsonable(divn), indent=2) + "\n")
            snap["captured"] = True
            snap["div"] = divn
            snap["step"] = rec["step"] + 1

    apply = bool(spec.get("apply_reserves", not spec.get("smoke")))
    pre = arm_preflight(spec, n_layer=new_L, n_params=unique_params(grown), outdir=outdir)
    reserve = arm_reserves_s(spec.get("arm", "G"), apply)
    log = train_loop(
        grown, device=device, seed=spec["seed"], old_L=old_L,
        target_flops=spec.get("target_flops"), target_bytes=spec.get("target_bytes"),
        wall_cap_s=spec["wall_cap_s"], log_path=outdir / "train_log.json",
        scratch=False, max_steps=spec.get("max_steps"), callback=cb,
        train_reserve_s=reserve,
    )
    if not snap["captured"]:
        tr_n = capture_block_traces(grown, seqs, device)
        divn = divergence_report(tr_n, tr0, sources, old_L, tag="end_before_N_div")
        (outdir / "div_Ndiv.json").write_text(json.dumps(_jsonable(divn), indent=2) + "\n")
        snap["div"] = divn

    post = run_eval_export(grown, spec, outdir, sources=sources)
    if post["knockout"] is not None and snap.get("div") is not None:
        prod = productive_blocks(snap["div"], post["knockout"], sources, old_L)
        (outdir / "productive.json").write_text(json.dumps(prod, indent=2) + "\n")
        post["productive"] = prod
    summary = {
        "arm": spec.get("arm", "G"),
        "mode": mode,
        "preflight": pre,
        "train_reserve_s": reserve,
        "seed": spec["seed"],
        "smoke": bool(spec.get("smoke")),
        "n_params": unique_params(grown),
        "old_L": old_L, "new_L": new_L,
        "fp_gate": fp,
        "n_div": n_div,
        "n_div_captured": snap["captured"],
        "presented_bytes": log["presented_bytes"],
        "flops": log["flops"],
        "steps_run": log["steps_run"],
        "stopped": log["stopped"],
        "eval_bpb_256": post["eval"]["primary_256"]["bpb"],
        "eval_bpb_2048": post["eval"]["descriptive_2048"]["bpb"],
        "productive": post.get("productive"),
        "export_gap": post["export"].get("gap"),
        "outdir": str(outdir),
        "cosine_steps": log.get("cosine_steps"),
        "zeroall_expected_near_zero_learning": mode == "identity_zero_all",
    }
    (outdir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary



def run_C(spec: dict) -> dict:
    """Continue the un-grown model. Stop at G's logged FLOPs (never more)."""
    device = spec["device"]
    outdir = Path(spec["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    target_flops, bytes_upper = resolve_cs_budget({**spec, "arm": "C"})
    model = load_or_empty({**spec, "load_L": spec["old_L"]})
    apply = bool(spec.get("apply_reserves", not spec.get("smoke")))
    spec = {**spec, "arm": "C", "target_flops": target_flops, "target_bytes": bytes_upper}
    pre = arm_preflight(spec, n_layer=spec["old_L"], n_params=unique_params(model), outdir=outdir)
    reserve = arm_reserves_s("C", apply)
    log = train_loop(
        model, device=device, seed=spec["seed"], old_L=spec["old_L"],
        target_flops=target_flops, target_bytes=bytes_upper,
        wall_cap_s=spec["wall_cap_s"], log_path=outdir / "train_log.json",
        scratch=False, max_steps=spec.get("max_steps"),
        train_reserve_s=reserve,
    )
    if log["flops"] > target_flops:
        raise RuntimeError(
            f"K7: C flops {log['flops']} exceeded G's logged {target_flops} (fail-closed)"
        )
    post = run_eval_export(model, {**spec, "old_L": len(model.blocks)}, outdir)
    summary = {
        "arm": "C",
        "seed": spec["seed"],
        "smoke": bool(spec.get("smoke")),
        "n_params": unique_params(model),
        "target_flops_from_G": target_flops,
        "bytes_upper_bound": bytes_upper,
        "presented_bytes": log["presented_bytes"],
        "flops": log["flops"],
        "steps_run": log["steps_run"],
        "stopped": log["stopped"],
        "preflight": pre,
        "train_reserve_s": reserve,
        "eval_bpb_256": post["eval"]["primary_256"]["bpb"],
        "eval_bpb_2048": post["eval"]["descriptive_2048"]["bpb"],
        "export_gap": post["export"].get("gap"),
        "g_outdir": str(spec["g_outdir"]),
        "outdir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary


def run_S(spec: dict) -> dict:
    """From-scratch at grown depth. Stop at G's logged FLOPs (never more)."""
    device = spec["device"]
    outdir = Path(spec["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    target_flops, bytes_upper = resolve_cs_budget({**spec, "arm": "S"})
    model = _make_empty(spec["new_L"], spec["d_model"], spec["n_head"], spec["ctx"], spec["seed"]).to(device)
    apply = bool(spec.get("apply_reserves", not spec.get("smoke")))
    spec = {**spec, "arm": "S", "target_flops": target_flops, "target_bytes": bytes_upper}
    pre = arm_preflight(spec, n_layer=spec["new_L"], n_params=unique_params(model), outdir=outdir)
    reserve = arm_reserves_s("S", apply)
    log = train_loop(
        model, device=device, seed=spec["seed"], old_L=0,
        target_flops=target_flops, target_bytes=bytes_upper,
        wall_cap_s=spec["wall_cap_s"], log_path=outdir / "train_log.json",
        scratch=True, max_steps=spec.get("max_steps"),
        train_reserve_s=reserve,
    )
    if log["flops"] > target_flops:
        raise RuntimeError(
            f"K7: S flops {log['flops']} exceeded G's logged {target_flops} (fail-closed)"
        )
    post = run_eval_export(model, {**spec, "old_L": spec["new_L"]}, outdir)
    summary = {
        "arm": "S",
        "seed": spec["seed"],
        "smoke": bool(spec.get("smoke")),
        "n_params": unique_params(model),
        "target_flops_from_G": target_flops,
        "bytes_upper_bound": bytes_upper,
        "presented_bytes": log["presented_bytes"],
        "flops": log["flops"],
        "steps_run": log["steps_run"],
        "stopped": log["stopped"],
        "preflight": pre,
        "train_reserve_s": reserve,
        "eval_bpb_256": post["eval"]["primary_256"]["bpb"],
        "eval_bpb_2048": post["eval"]["descriptive_2048"]["bpb"],
        "export_gap": post["export"].get("gap"),
        "outdir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary


def run_proxy_pretrain(spec: dict) -> dict:
    """Pretrain the L=PROXY_L0 proxy; budget PROXY_PRETRAIN_BYTES (or smoke max_steps)."""
    device = spec["device"]
    outdir = Path(spec["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    model = _make_empty(spec["old_L"], spec["d_model"], spec["n_head"], spec["ctx"], spec["seed"]).to(device)
    apply = bool(spec.get("apply_reserves", not spec.get("smoke")))
    pre = arm_preflight(
        {**spec, "arm": "proxy_pretrain"},
        n_layer=spec["old_L"], n_params=unique_params(model), outdir=outdir,
    )
    reserve = arm_reserves_s("proxy_pretrain", apply)
    log = train_loop(
        model, device=device, seed=spec["seed"], old_L=0,
        target_flops=None, target_bytes=spec.get("target_bytes", PROXY_PRETRAIN_BYTES),
        wall_cap_s=spec["wall_cap_s"], log_path=outdir / "train_log.json",
        scratch=True, max_steps=spec.get("max_steps"),
        train_reserve_s=reserve,
    )
    ckpt = outdir / "model.safetensors"
    write_safetensors(ckpt, model)
    ev = eval_heldout(model, device)
    (outdir / "eval_heldout.json").write_text(json.dumps(_jsonable(ev), indent=2) + "\n")
    summary = {
        "arm": "proxy_pretrain",
        "seed": spec["seed"],
        "smoke": bool(spec.get("smoke")),
        "n_params": unique_params(model),
        "presented_bytes": log["presented_bytes"],
        "flops": log["flops"],
        "steps_run": log["steps_run"],
        "stopped": log["stopped"],
        "preflight": pre,
        "eval_bpb_256": ev["primary_256"]["bpb"],
        "checkpoint": str(ckpt),
        "outdir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary


def ratio_spec(ratio: str, arm: str, seed: int, wall_cap_s: float, smoke: bool = False) -> dict:
    device = lock_runtime(EXP001_DEVICE)
    if ratio == "proxy":
        d, h, ctx = PROXY_D, PROXY_HEADS, 256
        old_L, new_L = PROXY_L0, PROXY_L1
        budgets = {
            "G": PROXY_B1_BYTES,
            "G_zeroall": PROXY_B1_BYTES,
            "G_signflip": PROXY_B1_BYTES,
            "S": PROXY_S_BYTES,
            "proxy_pretrain": PROXY_PRETRAIN_BYTES,
        }
        official = False
        weights = None
    elif ratio in ("A", "B"):
        d, h, ctx = 768, 12, 2048
        old_L, new_L = L_START, (L_RATIO_A if ratio == "A" else L_RATIO_B)
        budgets = dict(CPU_BOX_BUDGETS[ratio])
        budgets["G_zeroall"] = budgets["G"]
        budgets["G_signflip"] = budgets["G"]
        official = True
        weights = WEIGHTS
    else:
        raise ValueError(ratio)
    base = RESULTS / ("smoke_proxy" if smoke else "arms")
    outdir = base / f"{arm}_{ratio}_seed{seed}"
    g_outdir = base / f"G_{ratio}_seed{seed}"
    return {
        "device": device,
        "d_model": d, "n_head": h, "ctx": ctx,
        "old_L": old_L, "new_L": new_L, "load_L": old_L,
        "seed": seed, "ratio": ratio, "arm": arm,
        "wall_cap_s": wall_cap_s,
        "target_bytes": budgets.get(arm) if arm in ("G", "G_zeroall", "G_signflip", "S", "proxy_pretrain") else None,
        "bytes_upper_bound": budgets.get("C") if arm == "C" else (budgets.get("S") if arm == "S" else None),
        "official": official and not smoke,
        "weights": None if smoke else weights,
        "outdir": outdir,
        "g_outdir": g_outdir,
        "smoke": smoke,
        "mode": {
            "G": "identity_copy_inproj",
            "G_zeroall": "identity_zero_all",
            "G_signflip": "signflip_copy",
        }.get(arm, "identity_copy_inproj"),
    }


def run_arm(arm: str, ratio: str, seed: int, wall_cap_s: float, smoke: bool = False, **extra) -> dict:
    if not smoke and seed not in SEEDS and arm != "smoke":
        raise SystemExit(f"seed {seed} not in {SEEDS}")
    spec = ratio_spec(ratio, arm, seed, wall_cap_s, smoke=smoke)
    spec.update(extra)
    if arm in ("G", "G_zeroall", "G_signflip"):
        return run_G(spec)
    if arm == "C":
        return run_C(spec)
    if arm == "S":
        return run_S(spec)
    if arm == "proxy_pretrain":
        return run_proxy_pretrain(spec)
    raise SystemExit(f"unknown arm {arm}")


def _path_ok(summary: dict, need: tuple[str, ...]) -> list[str]:
    out = Path(summary["outdir"])
    return [n for n in need if not (out / n).is_file()]


def run_smoke_proxy() -> dict:
    """Full path on the proxy-size model. Not an official arm.

    Exercises EVERY arm type (pretrain, G, C, S, G_zeroall, G_signflip) plus
    K5 PREFLIGHT_ABORT and K7 early-stop FLOP match. Official seeds unused.
    """
    t0 = time.perf_counter()
    lock_runtime(EXP001_DEVICE)
    seed = 7
    wall = 180.0
    needed_g = (
        "fp_gate.json", "div_step0.json", "div_Ndiv.json", "knockout.json",
        "productive.json", "eval_heldout.json", "export_q3g64.json",
        "train_log.json", "preflight.json", "summary.json",
    )
    needed_cs = ("eval_heldout.json", "export_q3g64.json", "train_log.json",
                 "preflight.json", "summary.json")

    pre = run_arm(
        "proxy_pretrain", "proxy", seed, wall, smoke=True,
        max_steps=8, target_bytes=8 * TRAIN_T,
    )
    g = run_arm(
        "G", "proxy", seed, wall, smoke=True,
        max_steps=60, target_bytes=60 * TRAIN_T, n_div_override=50,
        weights=pre["checkpoint"], official=False,
    )
    c = run_arm("C", "proxy", seed, wall, smoke=True, weights=pre["checkpoint"], official=False)
    s = run_arm("S", "proxy", seed, wall, smoke=True, target_bytes=None)
    z = run_arm(
        "G_zeroall", "proxy", seed, wall, smoke=True,
        max_steps=16, target_bytes=16 * TRAIN_T, n_div_override=8,
        weights=pre["checkpoint"], official=False,
    )
    sf = run_arm(
        "G_signflip", "proxy", seed, wall, smoke=True,
        max_steps=16, target_bytes=16 * TRAIN_T, n_div_override=8,
        weights=pre["checkpoint"], official=False,
    )

    g_early = run_arm(
        "G", "proxy", seed, wall, smoke=True,
        max_steps=5, target_bytes=5 * TRAIN_T, n_div_override=3,
        weights=pre["checkpoint"], official=False,
        outdir=RESULTS / "smoke_proxy" / "G_early_proxy_seed7",
    )
    c_early = run_arm(
        "C", "proxy", seed, wall, smoke=True, weights=pre["checkpoint"], official=False,
        g_outdir=Path(g_early["outdir"]),
        outdir=RESULTS / "smoke_proxy" / "C_early_proxy_seed7",
    )
    s_early = run_arm(
        "S", "proxy", seed, wall, smoke=True, target_bytes=None,
        g_outdir=Path(g_early["outdir"]),
        outdir=RESULTS / "smoke_proxy" / "S_early_proxy_seed7",
    )
    k7_ok = (
        c_early["flops"] <= g_early["flops"]
        and s_early["flops"] <= g_early["flops"]
        and s_early["flops"] == g_early["flops"]
    )

    k5 = {"raised": False, "error": None, "status": None}
    try:
        arm_preflight(
            {
                "arm": "G", "d_model": PROXY_D, "n_head": PROXY_HEADS, "ctx": 256,
                "device": "cpu", "wall_cap_s": 10.0, "target_bytes": 60 * TRAIN_T,
                "apply_reserves": True, "smoke": True, "max_steps": 60,
            },
            n_layer=PROXY_L1,
            n_params=PROXY_N[4],
            outdir=RESULTS / "smoke_proxy" / "k5_preflight_abort",
        )
        k5["status"] = "FAILED_TO_ABORT"
    except PreflightAbort as e:
        k5["raised"] = True
        k5["error"] = str(e)
        k5["status"] = "PREFLIGHT_ABORT"
    k5_ok = k5["raised"] and k5["status"] == "PREFLIGHT_ABORT"

    errors = []
    if c["flops"] > g["flops"] or s["flops"] > g["flops"]:
        errors.append({"budget": {"G": g["flops"], "C": c["flops"], "S": s["flops"]}})
    if s["flops"] != g["flops"]:
        errors.append({"S_flops_not_exact": {"G": g["flops"], "S": s["flops"]}})
    if g.get("fp_gate", {}).get("pass") is not True:
        errors.append({"fp_error": g.get("fp_gate")})
    if z.get("fp_gate", {}).get("pass") is not True:
        errors.append({"fp_zeroall": z.get("fp_gate")})
    if z.get("mode") != "identity_zero_all":
        errors.append({"zeroall_mode": z.get("mode")})
    if sf.get("fp_gate", {}).get("pass") is True:
        errors.append({"signflip_unexpected_fp_pass": sf.get("fp_gate")})
    for name, rec, need in (
        ("G", g, needed_g), ("C", c, needed_cs), ("S", s, needed_cs),
        ("G_zeroall", z, needed_g), ("G_signflip", sf, needed_g),
        ("proxy_pretrain", pre, ("train_log.json", "summary.json", "preflight.json")),
    ):
        miss = _path_ok(rec, need)
        if miss:
            errors.append({name: {"missing": miss}})
        if name in ("G", "G_zeroall", "G_signflip") and not rec.get("n_div_captured"):
            errors.append({name: "n_div_not_captured"})
    if not k7_ok:
        errors.append({"K7": {"G": g_early["flops"], "C": c_early["flops"], "S": s_early["flops"]}})
    ram = test_ram_preflight_abort(RESULTS / "smoke_proxy" / "ram_preflight_abort")
    if not ram.get("ok"):
        errors.append({"RAM_PREFLIGHT_ABORT": ram})

    if not k5_ok:
        errors.append({"K5": k5})

    summary = {
        "role": "smoke_proxy",
        "is_arm": False,
        "note": "Proxy-size end-to-end path. Not an exp001 arm. Covers every arm type plus K5 abort and K7 early-stop.",
        "seed": seed,
        "device": EXP001_DEVICE,
        "dtype": EXP001_DTYPE,
        "thread_count": THREAD_COUNT,
        "expected_n_params": {"L2": PROXY_N[2], "L4": PROXY_N[4]},
        "pretrain": pre,
        "G": g, "C": c, "S": s,
        "G_zeroall": z, "G_signflip": sf,
        "K7_early": {"G": g_early, "C": c_early, "S": s_early, "ok": k7_ok},
        "K5_preflight_abort": k5,
        "RAM_preflight_abort": ram,
        "wall_s": time.perf_counter() - t0,
        "ok": len(errors) == 0,
        "errors": errors,
    }
    out = RESULTS / "smoke_proxy" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps({
        "ok": summary["ok"], "wall_s": summary["wall_s"], "errors": errors,
        "per_arm": {
            "pretrain": {"steps": pre["steps_run"], "flops": pre["flops"], "bpb256": pre["eval_bpb_256"]},
            "G": {"steps": g["steps_run"], "flops": g["flops"], "bpb256": g["eval_bpb_256"],
                  "fp": g["fp_gate"].get("pass"), "n_div": g["n_div_captured"], "n_params": g["n_params"]},
            "C": {"steps": c["steps_run"], "flops": c["flops"], "bpb256": c["eval_bpb_256"]},
            "S": {"steps": s["steps_run"], "flops": s["flops"], "bpb256": s["eval_bpb_256"]},
            "G_zeroall": {"steps": z["steps_run"], "fp": z["fp_gate"].get("pass"),
                          "n_div": z["n_div_captured"], "mode": z.get("mode")},
            "G_signflip": {"steps": sf["steps_run"], "fp": sf["fp_gate"].get("pass"),
                           "expected_fail": sf["fp_gate"].get("expected_fail"),
                           "n_div": sf["n_div_captured"]},
        },
        "K7": {"ok": k7_ok, "G": g_early["flops"], "C": c_early["flops"], "S": s_early["flops"]},
        "K5": k5,
        "RAM": {"ok": ram.get("ok"), "status": ram.get("status")},
    }, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-proxy", action="store_true",
                    help="Run the proxy-size end-to-end smoke test (not an arm).")
    ap.add_argument("--test-ram-abort", action="store_true",
                    help="Run only the MemAvailable negative preflight test (not an arm).")
    ap.add_argument("--arm", choices=("G", "C", "S", "G_zeroall", "G_signflip", "proxy_pretrain"))
    ap.add_argument("--ratio", default="A", choices=("A", "B", "proxy"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=EXP001_DEVICE)
    ap.add_argument("--wall-cap-s", type=float, default=DEFAULT_PILOT_CAP_S)
    ap.add_argument("--i-understand-this-starts-an-arm", action="store_true")
    args = ap.parse_args()
    if args.smoke_proxy:
        rec = run_smoke_proxy()
        sys.exit(0 if rec.get("ok") else 1)
    if args.test_ram_abort:
        rec = test_ram_preflight_abort()
        print(json.dumps({k: rec[k] for k in rec if k != "preflight"} | {"ok": rec["ok"], "projected_peak_rss_kb": (rec.get("preflight") or {}).get("projected_peak_rss_kb"), "mem_available_kb": (rec.get("preflight") or {}).get("mem_available_kb")}, indent=2))
        sys.exit(0 if rec.get("ok") else 1)
    if not args.arm or args.seed is None:
        print("Need --smoke-proxy, or --arm and --seed.", file=sys.stderr)
        sys.exit(2)
    if not args.i_understand_this_starts_an_arm:
        print("Refusing to start an arm without --i-understand-this-starts-an-arm", file=sys.stderr)
        sys.exit(2)
    if args.device != EXP001_DEVICE:
        raise RuntimeError(
            f"exp001 is locked to {EXP001_DEVICE}; refusing --device {args.device}. "
            "GPU is out of scope for exp001."
        )
    run_arm(args.arm, args.ratio, args.seed, args.wall_cap_s, smoke=False)
    return 0


if __name__ == "__main__":
    main()
