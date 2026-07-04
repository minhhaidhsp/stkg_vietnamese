"""
B1 (bắt buộc) — Qwen2.5-VL-7B zero-shot. CHƯA triển khai ở phiên này —
model ~7B, cần tải qua HuggingFace (`Qwen/Qwen2.5-VL-7B-Instruct`), cần
GPU đủ VRAM (~16GB+ ở fp16) — KHÔNG thử tải/chạy khi chưa có môi trường
Colab GPU thật (tránh tải nhầm hàng chục GB không cần thiết ở máy local).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.baselines.base import BaselineWrapper


class Qwen25VLZeroShot(BaselineWrapper):
    name = "qwen2.5_vl_7b_zeroshot"
    output_type = "free_text"
    group = "3_vlm"

    def __init__(self, cfg: dict):
        raise NotImplementedError(
            "Qwen2.5-VL-7B cần GPU thật (Colab) để tải/chạy — chưa thử ở máy local. "
            "Dùng transformers.Qwen2VLForConditionalGeneration.from_pretrained("
            "'Qwen/Qwen2.5-VL-7B-Instruct') khi có GPU."
        )

    def predict(self, question: str, image_path: str | None, candidates: list[str]) -> str:
        raise NotImplementedError
