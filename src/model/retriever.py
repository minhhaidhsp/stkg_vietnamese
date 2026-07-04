"""
CT (6) — Truy xuất có nhận thức không gian thời gian (luồng hướng ra).

rel(f) = alpha * cos(x_bar, g(f)) - (1 - alpha) * s_hat(f)

  - x_bar_raw: gộp trung bình các hàng của X (đã ở llm_hidden_size=896 vì X
    qua W_q/W_v — xác nhận thật khi wire MultimodalEncoder, xem
    docs/DECISIONS.md). PHẢI chiếu qua `retrieval_query_projection` (896->512,
    MỚI, học được, TÁCH RIÊNG với query_projection của CrossAttentionFusion
    dù cùng cặp chiều — vai trò khác nhau) để về cùng không gian d=512 với
    g(f) trước khi tính cosine.
  - g(f)  = trung bình STE(h), RE(r), STE(t) — đã ở d=512, không cần chiếu.
  - s_hat(f): s(f) (CT 2) chuẩn hóa min-max về [0,1] TRÊN TẬP ỨNG VIÊN hiện
    tại (không phải toàn cục), đúng mô tả manuscript ("trên tập ứng viên").
  - alpha=1 => rel(f) suy biến thành cos(x_bar_projected, g(f)) thuần túy —
    đây là ablation quan trọng nhất (tương đương retrieval ngữ nghĩa thuần
    của KG-Attention gốc [21]), PHẢI cho thứ hạng giống hệt cosine đơn thuần
    (trong không gian ĐÃ chiếu, vì phép chiếu là bước bắt buộc trước cosine,
    không phải một phần của trọng số alpha).
  - Không dùng ngưỡng cứng — chỉ chọn top-K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgraphRetriever(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        r_cfg = cfg["retrieval"]
        self.alpha = r_cfg["alpha"]
        self.top_k = r_cfg["top_k"]

        rqp = cfg["model"]["retrieval_query_projection"]
        self.in_dim = rqp["in_dim"]
        self.out_dim = rqp["out_dim"]

        expected_in = cfg["model"]["backbone"]["llm_hidden_size"]
        expected_out = cfg["spatiotemporal_grid"]["embedding_dim"]
        if self.in_dim != expected_in:
            raise ValueError(
                f"retrieval_query_projection.in_dim ({self.in_dim}) phải khớp "
                f"model.backbone.llm_hidden_size ({expected_in})."
            )
        if self.out_dim != expected_out:
            raise ValueError(
                f"retrieval_query_projection.out_dim ({self.out_dim}) phải khớp "
                f"spatiotemporal_grid.embedding_dim ({expected_out})."
            )

        self.retrieval_query_projection = nn.Linear(self.in_dim, self.out_dim, bias=False)

    @staticmethod
    def event_embedding(ste_h: torch.Tensor, re_r: torch.Tensor, ste_t: torch.Tensor) -> torch.Tensor:
        """g(f) = mean(STE(h), RE(r), STE(t)), shape (n_candidates, d)."""
        return (ste_h + re_r + ste_t) / 3.0

    @staticmethod
    def _min_max_normalize(s: torch.Tensor) -> torch.Tensor:
        s_min, s_max = s.min(), s.max()
        denom = (s_max - s_min).clamp(min=1e-12)
        return (s - s_min) / denom

    def project_x_bar(self, x_bar_raw: torch.Tensor) -> torch.Tensor:
        """x_bar_raw: (..., in_dim=896) -> x_bar: (..., out_dim=512)."""
        if x_bar_raw.shape[-1] != self.in_dim:
            raise ValueError(
                f"x_bar_raw có chiều cuối {x_bar_raw.shape[-1]}, kỳ vọng {self.in_dim} "
                f"(llm_hidden_size — x_bar_raw phải là X CHƯA chiếu, không phải g(f))."
            )
        return self.retrieval_query_projection(x_bar_raw)

    def relevance(
        self,
        x_bar_raw: torch.Tensor,       # (in_dim=896,) hoặc (batch, 896) — X gộp trung bình, CHƯA chiếu
        g_f: torch.Tensor,              # (n_candidates, d) hoặc (batch, n_candidates, d)
        s_f: torch.Tensor,              # (n_candidates,) hoặc (batch, n_candidates)
        alpha: float | None = None,
    ) -> torch.Tensor:
        alpha = self.alpha if alpha is None else alpha
        s_hat = self._min_max_normalize(s_f)
        x_bar = self.project_x_bar(x_bar_raw)

        if x_bar.dim() == 1:
            cos = F.cosine_similarity(x_bar.unsqueeze(0).expand_as(g_f), g_f, dim=-1)
        else:
            cos = F.cosine_similarity(x_bar.unsqueeze(1).expand_as(g_f), g_f, dim=-1)
        return alpha * cos - (1 - alpha) * s_hat

    def top_k_indices(self, rel: torch.Tensor, k: int | None = None) -> torch.Tensor:
        k = self.top_k if k is None else k
        k = min(k, rel.shape[-1])
        return torch.topk(rel, k, dim=-1).indices
