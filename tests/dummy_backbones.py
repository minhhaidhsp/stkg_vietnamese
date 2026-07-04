"""Backbone giả lập nhỏ dùng cho test MultimodalEncoder — mô phỏng đủ cấu
trúc đặt tên (q_proj/v_proj) để peft LoRA có thể gắn vào, KHÔNG phải bản
sao thật của Qwen2/InternViT. Chạy CPU, vài mili-giây."""

import torch
import torch.nn as nn


class DummyAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5), dim=-1)
        return self.o_proj(scores @ v)


class DummyTextBackbone(nn.Module):
    """Mô phỏng vài lớp attention của Qwen2-0.5B (chỉ để test LoRA/freeze),
    hidden_dim mặc định = 32 (nhỏ, KHÔNG phải 896 thật của Qwen2 — số 896
    dùng ở lớp chiếu W_q của MultimodalEncoder, không phải hidden_dim nội
    bộ của backbone giả lập này)."""

    def __init__(self, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([DummyAttention(hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class DummyVisionBackbone(nn.Module):
    """Mô phỏng InternViT: nhận patch feature thô, trả về patch embedding
    cùng chiều (không LoRA, theo đúng manuscript — LoRA chỉ gắn LLM)."""

    def __init__(self, hidden_dim: int = 24):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
