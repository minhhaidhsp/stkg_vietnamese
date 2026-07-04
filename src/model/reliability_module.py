"""
Module đánh giá độ tin cậy bộ ba trực quan (s, r, o) — PHIÊN BẢN HỌC ĐƯỢC,
thay thế step3_visual/reliability_scorer.py (công thức tuyến tính cố định
trọng số 0.35/0.30/0.20/0.15, không có tham số huấn luyện).

Nhận vào biểu diễn vector của 3 thành phần bộ ba (subject, relation,
object) — mỗi vector chiều feature_dim (mặc định = embedding_dim của lưới
STKG để dùng chung không gian đặc trưng, xem model.visual_reliability_module
trong config.yaml) — nối lại, qua MLP 1 lớp ẩn, sigmoid ra điểm tin cậy
trong [0,1]. Tham số của MLP được tối ưu qua L_VG (CT 9), KHÔNG cố định.
"""

import torch
import torch.nn as nn


class VisualReliabilityModule(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        vg = cfg["model"]["visual_reliability_module"]
        self.feature_dim = vg["feature_dim"]
        hidden_dim = vg["hidden_dim"]

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """s, r, o: (batch, feature_dim) — vector đặc trưng của bộ ba trực
        quan. Trả về (batch,) điểm tin cậy trong [0,1]."""
        for name, t in [("s", s), ("r", r), ("o", o)]:
            if t.shape[-1] != self.feature_dim:
                raise ValueError(f"Thành phần '{name}' có chiều {t.shape[-1]}, kỳ vọng {self.feature_dim}")
        x = torch.cat([s, r, o], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logits)
