"""
Giao diện chung cho baseline (Mục 4.2 manuscript_methodology.md, 3 nhóm).

predict() trả về MỘT TRONG HAI dạng tuỳ nhóm:
  - ranked_entities: list[str] (thực thể xếp hạng, cho nhóm 1 — nhúng đồ
    thị) -> tính được Hit@1/3/10, MRR trực tiếp.
  - free_text: str (văn bản tự do, cho nhóm 2/3 — LLM+KG, VLM) -> CHUẨN
    HÓA CHUỖI (NFC, thường hóa, bỏ dấu câu) rồi so khớp CHÍNH XÁC để tính
    Hit@1; KHÔNG suy ra được Hit@3/10/MRR (không có xếp hạng) — đúng ghi
    chú Mục 4.2 manuscript.
"""

import re
import unicodedata
from abc import ABC, abstractmethod


class BaselineWrapper(ABC):
    name: str = "base"
    output_type: str = "ranked_entities"  # hoặc "free_text"
    group: str = "1_embedding"  # "1_embedding" | "2_llm_kg" | "3_vlm"

    @abstractmethod
    def predict(self, question: str, image_path: str | None, candidates: list[str]) -> list[str] | str:
        """candidates: danh sách thực thể ứng viên (cho nhóm 1 xếp hạng lại;
        nhóm 2/3 có thể bỏ qua nếu mô hình tự sinh câu trả lời không ràng buộc)."""
        raise NotImplementedError


def normalize_text(s: str) -> str:
    """NFC + thường hóa + bỏ dấu câu — dùng so khớp chính xác cho nhóm 2/3 (Mục 4.2)."""
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def free_text_hit_at_1(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)
