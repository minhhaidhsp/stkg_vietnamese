"""
CT (9) — Mục tiêu đa nhiệm.

L_total = lambda1 * L_QA + lambda2 * L_STKG + lambda3 * L_VG
lambda1=1.0 (loss.lambda_qa), lambda2=0.5 (loss.lambda_stkg), lambda3=0.3 (loss.lambda_vg)

  - L_QA:   cross-entropy trên phân phối CT(8) (EntityRankingHead) so với
            thực thể đúng.
  - L_STKG: margin ranking loss trên s(f) (CT 2) — fact dương (thật) phải
            có s(f) THẤP hơn fact âm (negative sampling bằng thay thế thực
            thể ngẫu nhiên) tối thiểu 1 khoảng margin (loss.stkg_margin).
  - L_VG:   binary cross-entropy giữa điểm tin cậy dự đoán
            (VisualReliabilityModule) và nhãn mục tiêu — CHỈ trên các mẫu
            THẬT SỰ có ảnh/scene-graph triplet (mask `has_image`). Đa số
            facts (~90.7%) không có ảnh (Wikimedia matching mới phủ 652/6.980
            facts) nên một batch hoàn toàn không có mẫu nào có ảnh là tình
            huống BÌNH THƯỜNG, không phải hiếm — GUARD BẮT BUỘC: khi mask
            rỗng, trả về 0.0 tường minh, KHÔNG được chia 0/NaN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        loss_cfg = cfg["loss"]
        self.lambda_qa = loss_cfg["lambda_qa"]
        self.lambda_stkg = loss_cfg["lambda_stkg"]
        self.lambda_vg = loss_cfg["lambda_vg"]
        self.margin = loss_cfg["stkg_margin"]

    def qa_loss(self, p: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """p: (batch, num_entities) phân phối xác suất từ EntityRankingHead
        (đã softmax). target_idx: (batch,) chỉ số thực thể đúng."""
        return F.nll_loss(torch.log(p.clamp_min(1e-12)), target_idx)

    def stkg_loss(self, s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
        """s_pos, s_neg: (batch,) hoặc (batch, num_negatives) — s(f) CT(2)
        của fact dương/âm. Margin ranking: max(0, margin + s_pos - s_neg)."""
        return F.relu(self.margin + s_pos - s_neg).mean()

    def vg_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """mask: (batch,) bool — True = mẫu có ảnh/scene-graph triplet thật,
        đưa vào tính L_VG. False = không có ảnh, LOẠI KHỎI trung bình (không
        phải ép điểm về 0). Nếu mask is None, giả định pred/target đã được
        lọc sẵn (chỉ chứa mẫu có ảnh) trước khi gọi.

        GUARD: nếu không còn mẫu nào sau khi lọc (mask toàn False, hoặc
        pred/target rỗng ngay từ đầu — vd cả batch không có ảnh), trả về
        0.0 TƯỜNG MINH thay vì để binary_cross_entropy trên tensor rỗng gây
        NaN/lỗi chia 0."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        if pred.numel() == 0:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)
        return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target)

    def forward(
        self,
        p: torch.Tensor, qa_target: torch.Tensor,
        s_pos: torch.Tensor, s_neg: torch.Tensor,
        vg_pred: torch.Tensor, vg_target: torch.Tensor,
        vg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_qa = self.qa_loss(p, qa_target)
        l_stkg = self.stkg_loss(s_pos, s_neg)
        l_vg = self.vg_loss(vg_pred, vg_target, mask=vg_mask)
        total = self.lambda_qa * l_qa + self.lambda_stkg * l_stkg + self.lambda_vg * l_vg
        return {"loss_qa": l_qa, "loss_stkg": l_stkg, "loss_vg": l_vg, "loss_total": total}
