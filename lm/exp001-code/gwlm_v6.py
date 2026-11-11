"""
GoldWorm v6 — byte-level causal transformer
Architecture matches eval_results.json model_config and F1 parameter reconciliation:
  vocab_size=256, ctx=256, d_model=192, n_layer=6, n_head=4, ffn_mult=4
  tie_embeddings=True, learned absolute positional embedding (256×192)
  Pre-LN blocks: QKVO (no linear biases) + FFN (no linear biases) + affine LayerNorms
Total trainable params = 2,757,504 exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GoldWormV6Config:
    vocab_size: int = 256
    ctx: int = 256
    d_model: int = 192
    n_layer: int = 6
    n_head: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    """Pre-LN causal self-attention with QKVO projections and no linear biases."""
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout

        # Q, K, V, O projections: each d_model × d_model, no bias
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # causal mask, built lazily to the largest T seen (identical values to the
        # old fixed 256x256 buffer; persistent=False so checkpoints are unaffected)
        if not hasattr(self, "_mask") or self._mask.shape[-1] < T or \
                self._mask.device != x.device:
            self.register_buffer("_mask",
                                 torch.tril(torch.ones(1, 1, T, T, dtype=torch.bool,
                                                       device=x.device)),
                                 persistent=False)
        q = self.wq(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        k = self.wk(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.d_head ** 0.5))
        att = att.masked_fill(~self._mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.wo(y))
        return y


class MLP(nn.Module):
    """FFN with GELU, no linear biases."""
    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * ffn_mult, bias=False)
        self.fc2 = nn.Linear(d_model * ffn_mult, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Pre-LN transformer block: LN → Attention → +x → LN → FFN → +x"""
    def __init__(self, cfg: GoldWormV6Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.ffn_mult, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GoldWormV6(nn.Module):
    def __init__(self, cfg: GoldWormV6Config = None):
        super().__init__()
        self.cfg = cfg or GoldWormV6Config()

        # token embeddings
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)

        # learned absolute positional embedding (proven by F1 param count)
        self.pos_emb = nn.Embedding(self.cfg.ctx, self.cfg.d_model)

        self.drop = nn.Dropout(self.cfg.dropout)

        # transformer blocks
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layer)])

        self.ln_final = nn.LayerNorm(self.cfg.d_model)

        # output head (tied to token embeddings per config)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.ctx, f"Sequence length {T} exceeds context {self.cfg.ctx}"

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device, dtype=torch.long))
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
            return logits, loss
        return logits

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_params():
    """Return exact parameter count for the published architecture."""
    m = GoldWormV6()
    return m.n_params


if __name__ == "__main__":
    cfg = GoldWormV6Config()
    model = GoldWormV6(cfg)
    print(f"GoldWorm v6 — {model.n_params:,} trainable parameters")
    print(f"Config: {cfg}")
    # quick forward test
    x = torch.randint(0, 256, (2, 64), dtype=torch.long)
    logits = model(x)
    print(f"Input {x.shape} -> logits {logits.shape}")