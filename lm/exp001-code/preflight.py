#!/usr/bin/env python3
"""Measured torch preflight: fwd+bwd+AdamW, T=256, batch=1, for 85M/170M/255M.

CPU is the only device this script should be invoked with until a room lock
clears the unicorn GPU. Use --device cuda later, on machine 698f92ba-... only
after that lock is in place. Do not start training arms from this file.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch

# Make the code dir importable when run as a script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (
    ADAMW_BETAS, BATCH, HARD_CAP_S, L_RATIO_A, L_RATIO_B, L_START,
    LR_OLD, N_PARAMS, PREFLIGHT_ABORT_FRAC, DEFAULT_PILOT_CAP_S,
    RESULTS, THREAD_COUNT, TRAIN_T, WD_OLD,
)
from model_io import empty_v8, unique_params
from train import flops_fwd_bwd, make_optimizer


def peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return float(r.ru_utime + r.ru_stime)


def pin_threads(n: int):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)


def time_size(n_layer: int, device: torch.device, warmup=2, timed=5) -> dict:
    torch.manual_seed(0)
    model = empty_v8(n_layer).to(device)
    model.train()
    n_params = unique_params(model)
    # old_L = n_layer so a single LR group (preflight is throughput, not the G schedule)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_OLD, betas=ADAMW_BETAS, weight_decay=WD_OLD)
    B, T = BATCH, TRAIN_T
    # (B, T+1) -> T predicted bytes
    def batch():
        x = torch.randint(0, 256, (B, T + 1), device=device)
        return x[:, :-1], x[:, 1:]

    def step():
        x, y = batch()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        return float(loss.item())

    # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    rss0 = peak_rss_kb()
    cpu0 = cpu_seconds()
    t0 = time.perf_counter()
    last_loss = None
    for _ in range(warmup):
        last_loss = step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_warm = time.perf_counter()
    cpu_warm = cpu_seconds()
    for _ in range(timed):
        last_loss = step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    cpu_end = cpu_seconds()
    rss1 = peak_rss_kb()
    wall_timed = t_end - t_warm
    cpu_timed = cpu_end - cpu_warm
    per_step_wall = wall_timed / timed
    per_step_cpu = cpu_timed / timed
    presented = T  # targets per step (batch=1)
    bps = presented / per_step_wall
    out = {
        "n_layer": n_layer,
        "n_params": n_params,
        "n_params_expected": N_PARAMS.get(n_layer),
        "device": str(device),
        "thread_count": THREAD_COUNT if device.type == "cpu" else None,
        "seq_len_input": T + 1,
        "n_targets_per_step": presented,
        "batch": B,
        "warmup_steps": warmup,
        "timed_steps": timed,
        "last_loss": last_loss,
        "warmup_wall_s": t_warm - t0,
        "timed_wall_s": wall_timed,
        "timed_cpu_s": cpu_timed,
        "sec_per_step_wall": per_step_wall,
        "sec_per_step_cpu": per_step_cpu,
        "cpu_over_wall": (cpu_timed / wall_timed) if wall_timed else None,
        "presented_bytes_per_s_wall": bps,
        "flops_per_step": flops_fwd_bwd(n_params, presented),
        "peak_rss_kb_before": rss0,
        "peak_rss_kb_after": rss1,
        "definitions": {
            "wall": "time.perf_counter elapsed for the timed steps, CUDA-synchronized if cuda",
            "cpu_s": "resource.ru_utime + ru_stime of this process over the same window",
            "peak_rss_kb": "Linux ru_maxrss, kilobytes, process high-water mark",
        },
    }
    if device.type == "cuda":
        out["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        out["cuda_max_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # free
    del opt, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def derive_caps(rows: list[dict], pilot_cap=DEFAULT_PILOT_CAP_S, hard_cap=HARD_CAP_S) -> dict:
    """Per-size projected bytes/FLOPs at the pilot and hard caps, and 0.9x abort thresholds."""
    caps = {}
    for r in rows:
        wall = r["sec_per_step_wall"]
        bps = r["presented_bytes_per_s_wall"]
        n = r["n_params"]
        for name, cap in (("pilot", pilot_cap), ("hard", hard_cap)):
            abort = cap * PREFLIGHT_ABORT_FRAC
            caps.setdefault(r["n_layer"], {})[name] = {
                "wall_cap_s": cap,
                "abort_if_projected_wall_gt_s": abort,
                "bytes_at_cap": int(bps * cap),
                "bytes_at_0.9_cap": int(bps * abort),
                "steps_at_cap": int(cap / wall) if wall else None,
                "flops_at_cap": int(bps * cap * 6 * n),
                "sec_per_step_wall": wall,
            }
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--layers", default="12,24,36")
    args = ap.parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("cuda requested but torch.cuda.is_available() is False")
        # Safety: this script may be run on unicorn only AFTER a room lock.
        print("WARNING: GPU preflight — confirm the unicorn GPU is locked for this project.", flush=True)
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
    else:
        pin_threads(THREAD_COUNT)
        device = torch.device("cpu")
    layers = [int(x) for x in args.layers.split(",") if x]
    rows = []
    for L in layers:
        print(f"preflight L={L} on {device} ...", flush=True)
        row = time_size(L, device)
        print(json.dumps({k: row[k] for k in (
            "n_layer", "n_params", "sec_per_step_wall", "sec_per_step_cpu",
            "presented_bytes_per_s_wall", "peak_rss_kb_after", "cpu_over_wall"
        )}), flush=True)
        rows.append(row)
    caps = derive_caps(rows)
    report = {
        "device": str(device),
        "thread_count": THREAD_COUNT if device.type == "cpu" else None,
        "sizes": rows,
        "caps": caps,
        "pilot_cap_s": DEFAULT_PILOT_CAP_S,
        "hard_cap_s": HARD_CAP_S,
        "abort_frac": PREFLIGHT_ABORT_FRAC,
        "note": "Arms of a comparison must share this device. Projected wall > 0.9*cap aborts the arm before it starts. Training arms are not started here.",
    }
    out = args.out or (RESULTS / f"preflight_torch_{args.device}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
