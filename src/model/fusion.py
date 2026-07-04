"""
CT (7) — Chú ý chéo (luồng hướng vào, chuẩn, KHÔNG phải đóng góp mới).

r_m = Σ_i softmax((q_m^Z)^T k_i^X / sqrt(d_head)) · v_i^X

PHÁT HIỆN THẬT (tải Vintern-1B-v2 thật, xác nhận qua model.language_model.
model.layers[0].self_attn): Qwen2-0.5B dùng GQA (Grouped Query Attention),
KHÔNG phải MHA chuẩn — k_proj/v_proj chiếu 896 -> 128 (chỉ 2 KV-head × 64),
không phải 896 như giả định ban đầu ở Bước 3. Vì vậy:
  - q_m^Z: chiếu từ g(f) (d=512) sang 896 qua query_projection (PHÉP CHIẾU
    MỚI, cần huấn luyện), rồi reshape thành 14 query-head × 64.
  - k_i^X, v_i^X: LẤY THẲNG từ k_proj/v_proj THẬT của backbone (896->128,
    KHÔNG đổi chiều, không phát sinh tham số mới), reshape 2 KV-head × 64,
    rồi repeat_interleave theo group_size=num_heads/num_kv_heads=7 để khớp
    14 query-head (đúng cách Qwen2 tự tính attention nội bộ).
  - Attention từng head, scale = sqrt(head_dim=64) — KHÔNG phải sqrt(896).
  - Nối 14 head lại (896), qua o_proj THẬT của backbone (896->896, TÁI SỬ
    DỤNG, không phát sinh tham số mới) -> r_m ở đúng 896 để cộng dư vào X.

Toàn bộ num_heads/num_kv_heads/head_dim đọc từ config.yaml (khớp config.json
thật của model), không hardcode 14/2/64 trong code.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        qp = cfg["model"]["query_projection"]
        self.in_dim = qp["in_dim"]     # 512 = spatiotemporal_grid.embedding_dim
        self.out_dim = qp["out_dim"]   # 896 = model.backbone.llm_hidden_size

        backbone_cfg = cfg["model"]["backbone"]
        expected_out = backbone_cfg["llm_hidden_size"]
        if self.out_dim != expected_out:
            raise ValueError(
                f"query_projection.out_dim ({self.out_dim}) phải khớp "
                f"model.backbone.llm_hidden_size ({expected_out}) — không được lệch."
            )

        self.num_heads = backbone_cfg["llm_num_attention_heads"]
        self.num_kv_heads = backbone_cfg["llm_num_key_value_heads"]
        self.head_dim = backbone_cfg["llm_head_dim"]

        if self.num_heads * self.head_dim != self.out_dim:
            raise ValueError(
                f"num_heads*head_dim ({self.num_heads}*{self.head_dim}="
                f"{self.num_heads * self.head_dim}) phải bằng llm_hidden_size ({self.out_dim})."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) phải chia hết cho num_kv_heads "
                f"({self.num_kv_heads}) để repeat_interleave đúng nhóm GQA."
            )
        self.group_size = self.num_heads // self.num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim   # 128 thật của Qwen2-0.5B, KHÔNG phải out_dim

        self.query_projection = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def project_query(self, z_m: torch.Tensor) -> torch.Tensor:
        """z_m: (..., in_dim) -> q_m^Z: (..., out_dim)."""
        if z_m.shape[-1] != self.in_dim:
            raise ValueError(f"z_m có chiều cuối {z_m.shape[-1]}, kỳ vọng {self.in_dim}")
        q = self.query_projection(z_m)
        assert q.shape[-1] == self.out_dim
        return q

    def forward(
        self,
        z_m: torch.Tensor,
        k_x_raw: torch.Tensor,
        v_x_raw: torch.Tensor,
        o_proj: nn.Module,
    ) -> torch.Tensor:
        """
        z_m:     (batch, n_queries, in_dim=512) — nguồn của q_m^Z.
        k_x_raw: (batch, n_tokens, kv_dim=128)  — output THẬT của backbone.k_proj (tái sử dụng).
        v_x_raw: (batch, n_tokens, kv_dim=128)  — output THẬT của backbone.v_proj (tái sử dụng).
        o_proj:  nn.Linear(out_dim, out_dim)     — o_proj THẬT của backbone (tái sử dụng).
        Trả về r_m: (batch, n_queries, out_dim=896).
        """
        if k_x_raw.shape[-1] != self.kv_dim or v_x_raw.shape[-1] != self.kv_dim:
            raise ValueError(
                f"K/V phải có chiều cuối = num_kv_heads*head_dim ({self.kv_dim}), "
                f"tức chiều output THẬT của k_proj/v_proj Qwen2 (GQA) — "
                f"nhận được k={k_x_raw.shape[-1]} v={v_x_raw.shape[-1]}"
            )

        q = self.project_query(z_m)                # (batch, n_q, out_dim)
        batch, n_q, _ = q.shape
        n_tok = k_x_raw.shape[1]

        q = q.view(batch, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, n_q, head_dim)
        k = k_x_raw.view(batch, n_tok, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_x_raw.view(batch, n_tok, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # (batch, num_kv_heads, n_tok, head_dim)

        # GQA: mỗi KV-head dùng chung cho group_size query-head liền kề (đúng cách Qwen2 tính attention).
        k = k.repeat_interleave(self.group_size, dim=1)   # (batch, num_heads, n_tok, head_dim)
        v = v.repeat_interleave(self.group_size, dim=1)

        scale = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # (batch, num_heads, n_q, n_tok)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                              # (batch, num_heads, n_q, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch, n_q, self.out_dim)  # nối lại 14 head -> 896
        r_m = o_proj(out)   # tái sử dụng o_proj thật của backbone, không tham số mới
        assert r_m.shape[-1] == self.out_dim
        return r_m
