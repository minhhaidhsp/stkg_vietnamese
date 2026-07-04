"""
B2 (best-effort, tối đa 3 giờ sửa dependency/repo mỗi cái — quá thì BỎ, ghi
log lý do vào docs/DECISIONS.md, KHÔNG điền số từ paper gốc của họ vào
bảng kết quả vì khác dataset).

CẢ 4 BASELINE DƯỚI ĐÂY CHƯA TRIỂN KHAI — mỗi cái cần clone 1 repo GitHub
riêng (chưa xác định URL chính xác), cài dependency Python khác nhau (có
thể xung đột phiên bản với môi trường .venv hiện tại, tương tự vụ
transformers/peft/tokenizers đã gặp — nên rất có thể cần venv RIÊNG cho
từng baseline, không dùng chung .venv với Vintern), và viết adapter đọc
data/vistqad/*.csv theo đúng format input của từng repo.

  - EmbedKGQA: embedding-based KGQA, dùng ComplEx pretrained + LSTM câu hỏi.
  - CronKGQA: temporal KGQA trên TempQuestions/CronQuestions.
  - TempoQR: temporal-aware KGQA, có module thời gian tường minh (gần nhất
    với thiết kế STE/RE của ta — ưu tiên thử trước trong nhóm B2 nếu có
    thời gian).
  - GenTKGQA: temporal KGQA dạng sinh (generative), retrain trên ViSTQAD.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.baselines.base import BaselineWrapper


class _NotImplementedBaseline(BaselineWrapper):
    repo_url_placeholder = "CHƯA XÁC ĐỊNH"

    def __init__(self, cfg: dict):
        raise NotImplementedError(
            f"{self.name}: chưa clone repo ({self.repo_url_placeholder}), chưa viết adapter dữ liệu. "
            f"Nếu clone+setup dependency vượt quá 3 giờ, BỎ và ghi log lý do vào docs/DECISIONS.md "
            f"thay vì cố hoàn thiện — theo đúng quy tắc B2."
        )

    def predict(self, question: str, image_path: str | None, candidates: list[str]) -> list[str]:
        raise NotImplementedError


class EmbedKGQA(_NotImplementedBaseline):
    name = "embedkgqa"
    output_type = "ranked_entities"
    group = "1_embedding"


class CronKGQA(_NotImplementedBaseline):
    name = "cronkgqa"
    output_type = "ranked_entities"
    group = "1_embedding"


class TempoQR(_NotImplementedBaseline):
    name = "tempoqr"
    output_type = "ranked_entities"
    group = "1_embedding"


class GenTKGQA(_NotImplementedBaseline):
    name = "gentkgqa"
    output_type = "free_text"
    group = "2_llm_kg"
