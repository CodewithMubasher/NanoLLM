"""
NanoLLM v2 - Model Architecture
~50M parameter decoder-only Transformer with RoPE, RMSNorm, SwiGLU.

Architecture:
    Token Embedding -> [Transformer Block x N] -> RMSNorm -> LM Head

Each Transformer Block:
    RMSNorm -> Causal Self-Attention (RoPE) -> Residual
    RMSNorm -> SwiGLU FFN -> Residual
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ============================================================
# Normalization
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ============================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================

def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to x."""
    B, T, n_heads, head_dim = x.shape
    half = head_dim // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    cos = cos[:T, :half].unsqueeze(0).unsqueeze(2)
    sin = sin[:T, :half].unsqueeze(0).unsqueeze(2)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


# ============================================================
# Causal Self-Attention with RoPE
# ============================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length),
        )

        cos, sin = precompute_rope(config.head_dim, config.context_length)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        y = self.resid_dropout(self.o_proj(y))
        return y


# ============================================================
# SwiGLU Feed-Forward Network
# ============================================================

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Language Model
# ============================================================

class NanoLLM(nn.Module):
    """Decoder-only language model with RoPE, RMSNorm, SwiGLU."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tied_weights:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

        n_params = self.get_num_params()
        print(f"Model Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.config.context_length, f"Sequence length {T} > context_length {self.config.context_length}"

        x = self.drop(self.token_emb(idx))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay = set()
        no_decay = set()

        for pn, p in param_dict.items():
            if pn.endswith(".weight") and p.dim() == 2:
                decay.add(pn)
            else:
                no_decay.add(pn)

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters not assigned: {param_dict.keys() - union_params}"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        use_fused = device_type == "cuda" and torch.cuda.is_available()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer


def count_parameters(model: NanoLLM) -> dict:
    """Count parameters by component."""
    config = model.config
    counts = {
        "Token Embeddings": config.vocab_size * config.hidden_size,
        "Attention": 0,
        "SwiGLU": 0,
        "Norms": 0,
        "LM Head": 0,
    }

    for block in model.blocks:
        counts["Attention"] += (
            block.attn.q_proj.weight.numel() +
            block.attn.k_proj.weight.numel() +
            block.attn.v_proj.weight.numel() +
            block.attn.o_proj.weight.numel()
        )
        counts["SwiGLU"] += (
            block.ffn.gate_proj.weight.numel() +
            block.ffn.up_proj.weight.numel() +
            block.ffn.down_proj.weight.numel()
        )
        counts["Norms"] += block.norm1.weight.numel() + block.norm2.weight.numel()

    counts["Norms"] += model.norm.weight.numel()

    if not config.tied_weights:
        counts["LM Head"] = config.hidden_size * config.vocab_size

    counts["Total"] = sum(counts.values())
    return counts
