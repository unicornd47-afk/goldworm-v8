"""Minimal CPU inference for GoldWormV8 (byte-level, no tokenizer).
Usage: CUDA_VISIBLE_DEVICES='' python inference_cpu.py
"""
import json, torch
from pathlib import Path
from safetensors.torch import load_model
from goldworm_v8 import GoldWormV8, GoldWormV6Config

HERE = Path(__file__).resolve().parent

def load(device="cpu"):
    cfg = json.load(open(HERE / "config.json"))
    keys = ["vocab_size", "ctx", "d_model", "n_layer", "n_head", "ffn_mult", "dropout", "tie_embeddings"]
    m = GoldWormV8(GoldWormV6Config(**{k: cfg[k] for k in keys}))
    load_model(m, str(HERE / "model.safetensors"))
    return m.to(device).eval()

@torch.no_grad()
def complete(m, prompt, max_new=160, temperature=0.8, top_k=40, seed=0):
    """Sample one completion; stops at the first dedented non-blank line (same rule as the eval)."""
    g = torch.Generator().manual_seed(seed)
    ids = list(prompt.encode("utf-8"))[-(m.cfg.ctx - 1):]
    cache = [[] for _ in m.blocks]
    x = torch.tensor([ids]); out = []
    for _ in range(min(max_new, m.cfg.ctx - len(ids))):
        logits = m(x, cache=cache)[:, -1] / temperature
        v, _ = logits.topk(top_k); logits[logits < v[:, -1:]] = -float("inf")
        n = int(torch.multinomial(logits.softmax(-1), 1, generator=g)[0, 0])
        out.append(n); s = bytes(out)
        if len(s) >= 2 and s[-2:-1] == b"\n" and s[-1:] not in (b" ", b"\t", b"\n"):
            out.pop(); break
        x = torch.tensor([[n]])
    return bytes(out).decode("utf-8", "replace")

if __name__ == "__main__":
    torch.set_num_threads(4)
    m = load()
    prompt = 'def add(a, b):\n    """Return the sum of a and b."""\n'
    print(prompt + complete(m, prompt, seed=0))
