"""Depth-identity growth, fail-closed torch FP gate, symmetry-breaking noise."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from goldworm_v8 import RopeAttention
from gwlm_v6 import Block
from constants import FP32_TOL_MAX_ABS, L_START, NOISE_SIGMA
from model_io import unique_params


def _rope_block_like(src: nn.Module) -> nn.Module:
    d_model = src.attn.wq.in_features
    n_head = src.attn.n_head
    dropout = src.attn.dropout
    ffn_mult = src.mlp.fc1.out_features // d_model
    # Use a tiny stub cfg-like object
    class _C:
        pass
    c = _C()
    c.d_model = d_model
    c.n_head = n_head
    c.dropout = dropout
    c.ffn_mult = ffn_mult
    blk = Block(c)
    attn = RopeAttention(d_model, n_head, dropout)
    blk.attn = attn
    return blk


def load_block_state(blk: nn.Module, sd: dict, where: str) -> None:
    """Copy a source block. strict=True; any missing/unexpected key is fail-closed."""
    try:
        missing, unexpected = blk.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"GROW_COPY_FAIL {where}: {e} (fail-closed)") from e
    if missing or unexpected:
        raise RuntimeError(
            f"GROW_COPY_FAIL {where}: missing={list(missing)} unexpected={list(unexpected)} (fail-closed)"
        )


def identity_block(src: nn.Module, copy_inproj: bool, rng: torch.Generator, device, dtype) -> nn.Module:
    blk = _rope_block_like(src)
    if copy_inproj:
        load_block_state(blk, src.state_dict(), "identity_copy_inproj")
    else:
        # G_zeroall: every in-projection is exactly 0 (not N(0,1e-3)).
        # LN gamma=1, beta=0. wo/fc2 are zeroed below. Near-zero learning of
        # the new residual is the expected outcome (zero in-proj ⇒ zero branch).
        with torch.no_grad():
            for p in blk.parameters():
                p.zero_()
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
    with torch.no_grad():
        blk.attn.wo.weight.zero_()
        blk.mlp.fc2.weight.zero_()
    return blk.to(device=device, dtype=dtype)


def grow_depth(model, new_L: int, mode: str = "identity_copy_inproj", seed: int = 11):
    """Append identity residual blocks. Mapping: new index old_L+t -> source t % old_L.

    mode: identity_copy_inproj | identity_zero_all | signflip_copy (NOT function-preserving).
    Returns source index list for the new blocks.
    """
    old_L = len(model.blocks)
    if new_L <= old_L:
        raise ValueError(f"new_L {new_L} must exceed {old_L}")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    sources = []
    for t in range(new_L - old_L):
        src_i = t % old_L
        sources.append(src_i)
        src = model.blocks[src_i]
        if mode == "signflip_copy":
            blk = _rope_block_like(src)
            load_block_state(blk, src.state_dict(), "signflip_copy")
            with torch.no_grad():
                blk.attn.wo.weight.mul_(-1)
                blk.mlp.fc2.weight.mul_(-1)
            blk = blk.to(device=device, dtype=dtype)
        else:
            copy_inproj = mode != "identity_zero_all"
            blk = identity_block(src, copy_inproj=copy_inproj, rng=rng, device=device, dtype=dtype)
        model.blocks.append(blk)
    model.cfg.n_layer = new_L
    return sources


def apply_noise_new_blocks(model, old_L: int, sigma: float = NOISE_SIGMA, seed: int = 1011):
    """Symmetry break: Gaussian noise on new blocks' wo and fc2 only. After the FP gate."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    with torch.no_grad():
        for blk in model.blocks[old_L:]:
            for p in (blk.attn.wo.weight, blk.mlp.fc2.weight):
                noise = torch.randn(p.shape, generator=rng, dtype=torch.float32) * sigma
                p.add_(noise.to(device=p.device, dtype=p.dtype))


@torch.no_grad()
def logits_on(model, sequences: list[bytes], device) -> list[torch.Tensor]:
    model.eval()
    out = []
    for b in sequences:
        ids = torch.tensor(list(b), dtype=torch.long, device=device).unsqueeze(0)
        out.append(model(ids).float().cpu())
    return out


def max_abs_logit_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> dict:
    """Fail-closed comparison. Raises on count/shape mismatch or any non-finite logit.

    Must not use Python max() (NaN compared with a finite value is fail-open:
    max(0.0, nan) == 0.0). Check finiteness first, then reduce with torch.maximum.
    """
    if len(a) != len(b):
        raise RuntimeError(
            f"FP_GATE_FAIL sequence count mismatch: {len(a)} vs {len(b)} (fail-closed)"
        )
    if len(a) == 0:
        raise RuntimeError("FP_GATE_FAIL empty logit lists (fail-closed)")
    mx_t = None
    ss = torch.zeros((), dtype=torch.float64)
    n = 0
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if tuple(x.shape) != tuple(y.shape):
            raise RuntimeError(
                f"FP_GATE_FAIL shape mismatch at seq {i}: {tuple(x.shape)} vs {tuple(y.shape)} (fail-closed)"
            )
        if (not torch.isfinite(x).all()) or (not torch.isfinite(y).all()):
            raise RuntimeError(
                f"FP_GATE_FAIL non-finite logits at seq {i} (fail-closed)"
            )
        d = x.double() - y.double()
        this = d.abs().max()  # 0-dim torch tensor
        mx_t = this if mx_t is None else torch.maximum(mx_t, this)
        ss = ss + (d * d).sum()
        n += int(d.numel())
    mx = float(mx_t)
    return {
        "max_abs_logit_diff": mx,
        "rms_logit_diff": float((ss / n).sqrt()) if n else 0.0,
        "n_logits": n,
    }


def load_fp_sequences():
    """Frozen FP input set (v1 held-out prefixes + held-out A + add prompt)."""
    from pathlib import Path
    from constants import DATA, HELDOUT_V1_SHA256
    import hashlib
    v1 = DATA / "heldout_exp001_v1_superseded.bin"
    a = DATA / "heldout_A_antigoldworm.bin"
    primary = v1.read_bytes()
    got = hashlib.sha256(primary).hexdigest()
    if got != HELDOUT_V1_SHA256:
        raise RuntimeError(f"FP set v1 held-out sha256 mismatch: {got}")
    seqs = [primary[i * 2048: i * 2048 + 256] for i in (0, 8, 16, 24)]
    if a.is_file():
        seqs.append(a.read_bytes())
    seqs.append(b'def add(a, b):\n    """Return the sum of a and b."""\n')
    return seqs


def fp_gate_torch(original, grown, sequences, device, tol=FP32_TOL_MAX_ABS) -> dict:
    """Fail-closed: abort (raise) if max abs logit diff exceeds tol. Call BEFORE noise."""
    # v2.4.2: first CPU forward can differ from later ones (kernel warm-up; observed
    # orig-vs-orig 9.6e-5 on call 1, 0.0 after). Warm both models, then require the
    # original to reproduce itself bit-exactly before comparing (fail-closed).
    for _ in range(2):
        logits_on(original, sequences, device)
        logits_on(grown, sequences, device)
    z0 = logits_on(original, sequences, device)
    z0b = logits_on(original, sequences, device)
    self_noise = max_abs_logit_diff(z0, z0b)["max_abs_logit_diff"]
    if not (self_noise == 0.0):
        raise RuntimeError(
            f"FP_GATE_NONDETERMINISTIC orig_vs_orig={self_noise} (fail-closed, no override)"
        )
    z1 = logits_on(grown, sequences, device)
    # count/shape/non-finite failures raise inside max_abs_logit_diff (fail-closed)
    stats = max_abs_logit_diff(z0, z1)
    stats["tolerance"] = tol
    stats["orig_vs_orig_max_abs"] = self_noise
    stats["warmup_passes"] = 2
    stats["n_params_grown"] = unique_params(grown)
    # `not (mx <= tol)` is True for NaN, unlike a boolean we might default to pass
    if not (stats["max_abs_logit_diff"] <= tol):
        stats["pass"] = False
        raise RuntimeError(
            f"FP_GATE_FAIL max_abs_logit_diff={stats['max_abs_logit_diff']} tol={tol} "
            "(fail-closed, no override)"
        )
    stats["pass"] = True
    return stats
