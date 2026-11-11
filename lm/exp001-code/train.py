"""Training loop: two LR groups, FLOP/byte logging, G-first budget consumption."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from constants import (
    ADAMW_BETAS, BATCH, DATA, LR_NEW_MIN_FRAC, LR_NEW_PEAK, LR_OLD,
    N_PARAMS, TRAIN_NPY_SHA256, TRAIN_T, WARMUP_STEPS, WD_NEW, WD_OLD,
)
from model_io import sha256_file, unique_params


def flops_fwd_bwd(n_params: int, presented_bytes: int) -> int:
    """6 * n_unique_params * presented_bytes (tied embeddings counted once)."""
    return 6 * int(n_params) * int(presented_bytes)


def load_train_array() -> np.ndarray:
    path = DATA / "train_exp001.npy"
    got = sha256_file(path)
    if got != TRAIN_NPY_SHA256:
        raise RuntimeError(f"train npy sha256 mismatch: {got}")
    return np.load(path)  # (N, 2049) uint8


def param_groups(model, old_L: int | None):
    """G: old blocks + shared tensors at LR_OLD; new blocks at LR_NEW_PEAK (scheduled).

    C: old_L is None or equals n_layer — everything at LR_OLD.
    S: old_L == 0 — everything at the new/pretrain schedule.
    """
    if old_L is None:
        old_L = len(model.blocks)
    shared, old_p, new_p = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            (old_p if idx < old_L else new_p).append(p)
        else:
            shared.append(p)  # tok_emb, head (tied), ln_final
    groups = []
    if old_p or shared:
        groups.append({"params": old_p + shared, "lr": LR_OLD, "weight_decay": WD_OLD, "tag": "old"})
    if new_p:
        groups.append({"params": new_p, "lr": LR_NEW_PEAK, "weight_decay": WD_NEW, "tag": "new"})
    if not groups:
        raise RuntimeError("no param groups")
    return groups


def set_lrs(opt, step: int, total_new_steps: int, has_new: bool, scratch: bool):
    """Old group stays at LR_OLD. New group (and all of S) uses warmup+cosine to 0.1*peak."""
    for g in opt.param_groups:
        if g.get("tag") == "old" and not scratch:
            g["lr"] = LR_OLD
        else:
            # pretrain schedule
            warm = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
            if total_new_steps <= 0:
                decay = 1.0
            else:
                prog = min(1.0, step / max(1, total_new_steps))
                decay = LR_NEW_MIN_FRAC + (1 - LR_NEW_MIN_FRAC) * 0.5 * (1 + math.cos(math.pi * prog))
            g["lr"] = LR_NEW_PEAK * warm * decay


def make_optimizer(model, old_L: int | None):
    groups = param_groups(model, old_L)
    opt = torch.optim.AdamW(groups, betas=ADAMW_BETAS)
    return opt


def sample_batch(data: np.ndarray, rng: np.random.Generator, device):
    """One 257-byte slice (256 predicted positions) from a random train window."""
    row = int(rng.integers(0, len(data)))
    w = data[row]
    # 2049-byte windows; need 257 consecutive bytes
    max_off = len(w) - (TRAIN_T + 1)
    off = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
    sl = np.ascontiguousarray(w[off:off + TRAIN_T + 1])
    x = torch.from_numpy(sl[:-1].copy()).long().unsqueeze(0).to(device)
    y = torch.from_numpy(sl[1:].copy()).long().unsqueeze(0).to(device)
    return x, y, int(y.numel())


def train_loop(
    model,
    *,
    device,
    seed: int,
    old_L: int | None,
    target_flops: int | None,
    target_bytes: int | None,
    wall_cap_s: float,
    log_path: Path,
    scratch: bool = False,
    max_steps: int | None = None,
    callback=None,
    train_reserve_s: float = 0.0,
):
    """Train until FLOP budget or the reserved train wall.

    target_flops is authoritative for C/S (G's logged FLOPs). A step is not
    taken if it would make logged FLOPs exceed the target (never more than G).
    target_bytes is only an upper bound. The cosine schedule length is derived
    from the registered budget (target_bytes or the FLOP-implied step count),
    never a 1e9 placeholder.

    wall_cap_s is the raw arm cap. Training stops at
    train_wall = wall_cap_s - train_reserve_s (K6: 300+600 s reserves subtracted
    in code).
    """
    model.train()
    n_params = unique_params(model)
    data = load_train_array()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = make_optimizer(model, old_L if not scratch else 0)
    has_new = any(g.get("tag") == "new" for g in opt.param_groups)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    presented = 0
    flops = 0
    t0 = time.perf_counter()
    step = 0
    if target_bytes is not None:
        total_new_steps = max(1, int(target_bytes) // TRAIN_T)
    elif target_flops is not None:
        total_new_steps = max(1, int(target_flops) // max(1, 6 * n_params * TRAIN_T))
    elif max_steps is not None:
        total_new_steps = max(1, int(max_steps))
    else:
        raise RuntimeError(
            "K3: cosine schedule length requires target_bytes, target_flops, or max_steps "
            "(no 1e9 default)"
        )
    if max_steps is not None:
        total_new_steps = min(total_new_steps, int(max_steps))
    train_wall = float(wall_cap_s) - float(train_reserve_s)
    if train_wall <= 0:
        raise RuntimeError(
            f"K6: train wall non-positive (wall_cap_s={wall_cap_s} reserve={train_reserve_s})"
        )
    meta = {
        "n_params": n_params, "seed": seed, "old_L": old_L, "scratch": scratch,
        "target_flops": target_flops, "target_bytes": target_bytes,
        "wall_cap_s": wall_cap_s, "train_reserve_s": train_reserve_s,
        "train_wall_cap_s": train_wall, "cosine_steps": total_new_steps,
        "rng_seed": seed,
        "steps": [],
    }
    stopped = "running"
    while True:
        # Do not start a step that would exceed G's logged FLOPs (K7, never more).
        n_t_planned = TRAIN_T * 1  # batch=1, TRAIN_T predicted bytes
        if target_flops is not None and flops + flops_fwd_bwd(n_params, n_t_planned) > target_flops:
            stopped = "target_flops"
            break
        if target_bytes is not None and presented + n_t_planned > target_bytes:
            stopped = "target_bytes"
            break
        if max_steps is not None and step >= max_steps:
            stopped = "max_steps"
            break
        if (time.perf_counter() - t0) >= train_wall:
            stopped = "wall_cap"
            break
        set_lrs(opt, step, total_new_steps, has_new, scratch)
        x, y, n_t = sample_batch(data, rng, device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        presented += n_t
        flops += flops_fwd_bwd(n_params, n_t)
        wall = time.perf_counter() - t0
        rec = {
            "step": step, "loss": float(loss.item()), "presented_bytes": presented,
            "flops": flops, "wall_s": wall,
            "lrs": {g.get("tag", "?"): g["lr"] for g in opt.param_groups},
        }
        meta["steps"].append(rec)
        if callback is not None:
            callback(step, model, rec)
        step += 1
        # Stop conditions are evaluated at the top of the next iteration so a
        # step cannot start if it would exceed G's logged FLOPs or the reserved train wall.
    meta["stopped"] = stopped
    meta["presented_bytes"] = presented
    meta["flops"] = flops
    meta["steps_run"] = step
    meta["wall_s"] = time.perf_counter() - t0
    with open(log_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta
