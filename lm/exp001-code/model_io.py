"""Load GoldWormV8 from a safetensors file without the safetensors package."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch

from goldworm_v8 import GoldWormV8
from gwlm_v6 import GoldWormV6Config
from constants import CTX, D_MODEL, FFN_MULT, L_START, N_HEAD, VOCAB, WEIGHTS, WEIGHTS_SHA256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_params(model: torch.nn.Module) -> int:
    """Count unique parameter elements (tied tok_emb / head once)."""
    seen = set()
    n = 0
    for p in model.parameters():
        i = id(p)
        if i in seen:
            continue
        seen.add(i)
        n += p.numel()
    return n


def make_cfg(n_layer: int, d_model: int = D_MODEL, n_head: int = N_HEAD, ctx: int = CTX) -> GoldWormV6Config:
    return GoldWormV6Config(
        vocab_size=VOCAB, ctx=ctx, d_model=d_model, n_layer=n_layer,
        n_head=n_head, ffn_mult=FFN_MULT, dropout=0.0, tie_embeddings=True,
    )


def read_safetensors_tensors(path: Path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        body = f.read()
    base = 0
    out = {}
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dt = dtypes[meta["dtype"]]
        buf = bytearray(body[start:end])
        t = torch.frombuffer(buf, dtype=dt).clone().reshape(meta["shape"])
        out[name] = t
    return out


def _ignorable_key(name: str) -> bool:
    """pos_emb was deleted; rope_freq is generated and non-persistent."""
    return "pos_emb" in name or "rope_freq" in name


def apply_tied_embedding_aliases(tensors: dict) -> dict:
    """File may contain only tok_emb.weight or only head.weight."""
    out = dict(tensors)
    if "tok_emb.weight" in out and "head.weight" not in out:
        out["head.weight"] = out["tok_emb.weight"]
    elif "head.weight" in out and "tok_emb.weight" not in out:
        out["tok_emb.weight"] = out["head.weight"]
    return out


def load_state_dict_fail_closed(model: torch.nn.Module, tensors: dict) -> tuple[list[str], list[str]]:
    """Load weights; missing or unexpected keys fail closed after documented aliases.

    Allowed exceptions: pos_emb (deleted in v8) and rope_freq (generated).
    Tied embeddings are aliased first so a file with only one of tok_emb/head is OK.
    An unexpected key in the file or from load_state_dict raises.
    """
    tensors = apply_tied_embedding_aliases(tensors)
    sd_keys = set(model.state_dict())
    unexpected_file = sorted(k for k in tensors if k not in sd_keys and not _ignorable_key(k))
    if unexpected_file:
        raise RuntimeError(f"checkpoint unexpected keys: {unexpected_file}")
    # Do not feed ignorable extras (pos_emb, rope_freq) into load_state_dict.
    filtered = {k: v for k, v in tensors.items() if not _ignorable_key(k)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [m for m in missing if not _ignorable_key(m)]
    unexpected = [u for u in unexpected if not _ignorable_key(u)]
    if missing:
        raise RuntimeError(f"checkpoint missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"checkpoint unexpected keys: {unexpected}")
    if getattr(getattr(model, "cfg", None), "tie_embeddings", False):
        model.head.weight = model.tok_emb.weight
    return missing, unexpected


def load_v8(path: Path | None = None, n_layer: int = L_START, map_location="cpu", verify_sha256: bool = True) -> GoldWormV8:
    path = Path(path or WEIGHTS)
    if verify_sha256 and path.resolve() == Path(WEIGHTS).resolve():
        got = sha256_file(path)
        if got != WEIGHTS_SHA256:
            raise RuntimeError(f"weights sha256 mismatch: {got} != {WEIGHTS_SHA256}")
    tensors = read_safetensors_tensors(path)
    cfg = make_cfg(n_layer)
    model = GoldWormV8(cfg)
    load_state_dict_fail_closed(model, tensors)
    model.eval()
    return model.to(map_location)


def empty_v8(n_layer: int, d_model: int = D_MODEL, n_head: int = N_HEAD, ctx: int = CTX) -> GoldWormV8:
    """Random-init model of a given depth (for preflight / from-scratch / proxy)."""
    return GoldWormV8(make_cfg(n_layer, d_model=d_model, n_head=n_head, ctx=ctx))


def write_safetensors(path: Path, model: torch.nn.Module) -> Path:
    """Write unique parameters (tied head/tok_emb once) as an F32 safetensors file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        i = id(p)
        if i in seen:
            continue
        seen.add(i)
        tensors[name] = p.detach().to("cpu").contiguous().float()
    header = {}
    offset = 0
    raw = []
    for name, arr in tensors.items():
        b = arr.numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + len(b)]}
        raw.append(b)
        offset += len(b)
    header["__metadata__"] = {"format": "pt"}
    h = json.dumps(header).encode("utf-8")
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(raw))
    return path
