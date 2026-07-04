"""
B1 (bắt buộc) — Think-on-Graph: agent LLM training-free duyệt đồ thị sáu
ngôi đã tuyến tính hóa (không retrain, chỉ cần LLM + prompt duyệt đồ thị).

CHƯA TRIỂN KHAI ở phiên này. Cần:
  1. Tuyến tính hóa đồ thị con quanh câu hỏi thành text (vd liệt kê các
     fact (h,r,t,τ,l_h,l_t) liên quan dưới dạng câu, giới hạn độ dài prompt).
  2. Một LLM để đóng vai "agent" duyệt/suy luận (bài gốc ToG dùng GPT-4;
     cần quyết định dùng model nào — Qwen2.5-VL-7B đã có trong danh sách
     B1 hay 1 API khác — CHƯA CHỐT, cần hỏi trước khi chọn để tránh phát
     sinh chi phí API ngoài dự tính).
  3. Prompt template ToG gốc (beam search trên đồ thị, chọn nhánh theo LLM
     tự đánh giá) — cần đọc paper gốc [tham chiếu 21 trong bản thảo] để
     bám đúng thuật toán, không tự chế phiên bản khác rồi gọi là "ToG".
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.baselines.base import BaselineWrapper


class ThinkOnGraph(BaselineWrapper):
    name = "think_on_graph"
    output_type = "free_text"
    group = "2_llm_kg"

    def __init__(self, cfg: dict):
        raise NotImplementedError(
            "Think-on-Graph CHƯA triển khai — cần chốt LLM agent dùng và đọc "
            "đúng thuật toán duyệt đồ thị gốc trước khi viết code (xem docstring)."
        )

    def predict(self, question: str, image_path: str | None, candidates: list[str]) -> str:
        raise NotImplementedError
