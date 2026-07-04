"""
B1 (bắt buộc) — Vintern-1B-v2 zero-shot: hỏi trực tiếp bằng prompt tiếng
Việt + ảnh (nếu có), KHÔNG qua STE/RE/retrieval/fusion của ta — dùng
nguyên bản khả năng chat đa phương thức có sẵn của model.

TRẠNG THÁI: sườn đã viết, DÙNG ĐƯỢC NGAY vì tái sử dụng
MultimodalEncoder.from_pretrained() đã xác nhận tải thành công thật (xem
docs/DECISIONS.md) — nhưng CHƯA chạy thật trên tập test (cần GPU đủ nhanh
để chạy hết ~1.746 câu test, và cần gọi đúng API .chat()/.generate() riêng
của InternVLChatModel — CHƯA xác minh chữ ký hàm thật, chỉ mới xác nhận
.from_pretrained() và cấu trúc submodule, KHÔNG phải toàn bộ API generate).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.baselines.base import BaselineWrapper


class VinternZeroShot(BaselineWrapper):
    name = "vintern_1b_v2_zeroshot"
    output_type = "free_text"
    group = "3_vlm"

    def __init__(self, cfg: dict):
        from transformers import AutoModel, AutoTokenizer
        name = cfg["model"]["backbone"]["name"]
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)

    def predict(self, question: str, image_path: str | None, candidates: list[str]) -> str:
        # TODO xác minh chữ ký .chat() thật của InternVLChatModel (thường có
        # dạng model.chat(tokenizer, pixel_values, question, generation_config))
        # trước khi chạy thật trên tập test — KHÔNG đoán mò tham số.
        raise NotImplementedError(
            "Cần xác minh API .chat()/.generate() thật của InternVLChatModel "
            "trên môi trường có GPU trước khi hoàn thiện predict() — chưa làm ở phiên này."
        )
