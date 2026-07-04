"""
B1 (bắt buộc) — SubGTR retrain trên ViSTQAD. CHƯA triển khai — cần clone
repo gốc SubGTR (subgraph-based temporal KGQA), thay bộ mã hóa câu hỏi
tiếng Anh bằng PhoBERT/đa ngữ, và viết adapter đọc data/vistqad/*.csv theo
đúng định dạng input mà code gốc SubGTR yêu cầu (thường là (h,r,t,τ) rời
rạc, không có l_h/l_t — cần quyết định cách nhúng thêm thông tin không
gian, hoặc bỏ qua l_h/l_t cho baseline này vì bản gốc không hỗ trợ, và ghi
rõ hạn chế này trong bảng kết quả).

Repo gốc: CHƯA XÁC ĐỊNH URL chính xác — cần tìm link paper SubGTR trong
danh mục tham khảo của bản thảo trước khi clone (tránh clone nhầm repo).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.baselines.base import BaselineWrapper


class SubGTR(BaselineWrapper):
    name = "subgtr_retrain"
    output_type = "ranked_entities"
    group = "1_embedding"

    def __init__(self, cfg: dict):
        raise NotImplementedError(
            "SubGTR cần clone repo gốc + viết adapter dữ liệu — chưa xác định "
            "URL repo chính xác, chưa làm ở phiên này."
        )

    def predict(self, question: str, image_path: str | None, candidates: list[str]) -> list[str]:
        raise NotImplementedError
