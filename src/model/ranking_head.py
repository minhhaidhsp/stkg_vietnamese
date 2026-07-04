"""
CT (8) — Đầu ra: xếp hạng thực thể (Phương án A, ĐÃ CHỐT).

p(e | Q, I, G) = softmax(u^T · STE(e))    với e ∈ toàn bộ tập thực thể E

KHÔNG sinh văn bản tự do / autoregressive generation dưới bất kỳ hình thức
nào — không gọi .generate()/beam search, chỉ có một forward pass tạo phân
phối xác suất trên tập thực thể.

u được suy ra từ luồng ẩn cuối cùng của LLM (chiều llm_hidden_size=896,
sau khi cộng dư X ⊕ R_final), CHIẾU VỀ d=512 để khớp chiều STE(e) — lớp
ranking_projection (896->512, cfg model.ranking_projection) là tham số
MỚI, không có sẵn trong backbone. TÁCH RIÊNG với retrieval_query_projection
(SubgraphRetriever, CT6) dù cùng in/out dim — vai trò khác nhau (CT6 chiếu
x̄ để truy xuất subgraph, CT8 chiếu u để xếp hạng thực thể cuối cùng),
không dùng chung 1 lớp cho 2 việc.
"""

import torch
import torch.nn as nn


class EntityRankingHead(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        rp = cfg["model"]["ranking_projection"]
        self.d = rp["out_dim"]
        expected_d = cfg["spatiotemporal_grid"]["embedding_dim"]
        if rp["out_dim"] != expected_d:
            raise ValueError(
                f"ranking_projection.out_dim ({rp['out_dim']}) phải khớp "
                f"spatiotemporal_grid.embedding_dim ({expected_d})."
            )
        expected_in = cfg["model"]["backbone"]["llm_hidden_size"]
        if rp["in_dim"] != expected_in:
            raise ValueError(
                f"ranking_projection.in_dim ({rp['in_dim']}) phải khớp "
                f"model.backbone.llm_hidden_size ({expected_in})."
            )
        self.ranking_projection = nn.Linear(rp["in_dim"], rp["out_dim"], bias=False)

    def compute_u(self, final_hidden: torch.Tensor) -> torch.Tensor:
        """final_hidden: (batch, llm_hidden_size) -> u: (batch, d)."""
        return self.ranking_projection(final_hidden)

    def forward(self, final_hidden: torch.Tensor, ste_entities: torch.Tensor) -> torch.Tensor:
        """
        final_hidden: (batch, llm_hidden_size)
        ste_entities: (num_entities, d) — STE(e) của TOÀN BỘ tập thực thể E.
        Trả về p(e|Q,I,G): (batch, num_entities), là phân phối xác suất hợp lệ
        (không âm, tổng theo hàng = 1) — KHÔNG dùng generate()/beam search.
        """
        u = self.compute_u(final_hidden)              # (batch, d)
        logits = u @ ste_entities.t()                   # (batch, num_entities)
        return torch.softmax(logits, dim=-1)
