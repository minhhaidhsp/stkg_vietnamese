"""
Template sinh câu hỏi tiếng Việt từ fact 6-ngôi (h, r, t, τ, l_h, l_t).

Mỗi template che (mask) một phía của fact — luôn là một THỰC THỂ (h hoặc t),
không bao giờ che năm — để đáp án luôn nằm trong tập thực thể E, khớp thiết
kế "entity ranking" của mô hình (CT 8), không sinh câu hỏi trả lời bằng số.

Số lượng template CỐ Ý KHÔNG ĐỀU giữa các quan hệ: bornIn/diedIn (đã chiếm
71.9% số facts) chỉ có 2 template/fact, còn occurredAt/locatedIn (chỉ 4.0%
và 24.1% số facts) có 4 template/fact — để phần nào cân bằng lại PHÂN BỐ SỐ
CÂU HỎI (không phải phân bố facts, vốn không đổi được bằng sinh câu hỏi).
"""

from dataclasses import dataclass


@dataclass
class Template:
    template_id: str
    relation: str
    mask: str          # "t" hoặc "h" — phía bị che, cũng là đáp án
    question_type: str  # nhãn loại câu hỏi
    requires: tuple[str, ...]  # các trường bắt buộc phải có giá trị (ngoài h_label/t_label)
    format_fn: "callable"      # (row) -> câu hỏi tiếng Việt


def _bornIn_t1(row):
    return f"{row['h_label']} sinh ra ở đâu?"


def _bornIn_t2(row):
    return f"Nơi sinh của {row['h_label']} là gì?"


def _diedIn_t1(row):
    return f"{row['h_label']} mất ở đâu?"


def _diedIn_t2(row):
    return f"{row['h_label']} qua đời tại đâu?"


def _occurredAt_t1(row):
    return f"Sự kiện {row['h_label']} xảy ra ở đâu?"


def _occurredAt_t2(row):
    return f"{row['h_label']} diễn ra tại địa điểm nào?"


def _occurredAt_t3(row):
    return f"Địa điểm gắn liền với sự kiện {row['h_label']} là gì?"


def _occurredAt_t4_reverse(row):
    year = int(row["tau_start"])
    return f"Vào năm {year}, sự kiện nào đã xảy ra tại {row['t_label']}?"


def _locatedIn_t1(row):
    return f"{row['h_label']} thuộc tỉnh/thành nào?"


def _locatedIn_t2(row):
    return f"Di tích {row['h_label']} nằm ở đâu?"


def _locatedIn_t3(row):
    return f"Địa bàn hành chính của {row['h_label']} là gì?"


def _locatedIn_t4(row):
    year = int(row["tau_start"])
    return f"{row['h_label']} được công nhận di tích quốc gia vào năm {year}, tại tỉnh/thành nào?"


TEMPLATES: list[Template] = [
    Template("bornIn_t1", "bornIn", "t", "location_query", (), _bornIn_t1),
    Template("bornIn_t2", "bornIn", "t", "location_query", (), _bornIn_t2),

    Template("diedIn_t1", "diedIn", "t", "location_query", (), _diedIn_t1),
    Template("diedIn_t2", "diedIn", "t", "location_query", (), _diedIn_t2),

    Template("occurredAt_t1", "occurredAt", "t", "location_query", (), _occurredAt_t1),
    Template("occurredAt_t2", "occurredAt", "t", "location_query", (), _occurredAt_t2),
    Template("occurredAt_t3", "occurredAt", "t", "location_query", (), _occurredAt_t3),
    # T4 đảo chiều (che h): chỉ áp dụng khi (t_label, tau_start) là duy nhất
    # trong toàn bộ tập occurredAt (kiểm tra ở generate_questions.py), tránh
    # đáp án nhập nhằng khi nhiều sự kiện cùng địa điểm + năm.
    Template("occurredAt_t4_reverse", "occurredAt", "h", "entity_query", ("tau_start",), _occurredAt_t4_reverse),

    Template("locatedIn_t1", "locatedIn", "t", "location_query", (), _locatedIn_t1),
    Template("locatedIn_t2", "locatedIn", "t", "location_query", (), _locatedIn_t2),
    Template("locatedIn_t3", "locatedIn", "t", "location_query", (), _locatedIn_t3),
    Template("locatedIn_t4", "locatedIn", "t", "location_query", ("tau_start",), _locatedIn_t4),
]


def templates_for_relation(relation: str) -> list[Template]:
    return [t for t in TEMPLATES if t.relation == relation]


_TEMPLATE_BY_ID = {t.template_id: t for t in TEMPLATES}

ENTITY_PLACEHOLDER = "X"


def canonical_form(template_id: str, h_label: str, t_label: str, tau_start=None) -> str:
    """"Khuôn mẫu gốc" của 1 câu hỏi đã sinh: cùng template, nhưng thay tên
    thực thể (h_label/t_label) bằng token trung tính "X" — τ (nếu template
    có dùng) giữ nguyên vì đó là nội dung sự kiện, không phải slot paraphrase.

    Dùng để đo BLEU/ROUGE-L giữa câu hỏi sinh ra và khuôn mẫu gốc (Mục 3.4
    manuscript_methodology.md): vì pipeline KHÔNG có bước viết lại bằng LLM
    (đã xác nhận — generate_questions.py chỉ f-string điền template), phép
    đo này cho biết mức paraphrase THẬT sự đã xảy ra — dự kiến gần như tối
    đa (câu hỏi sinh ra == khuôn mẫu gốc, chỉ khác đúng token thực thể),
    không phải một chỉ số "đa dạng" giữa các câu hỏi với nhau.
    """
    template = _TEMPLATE_BY_ID[template_id]
    row = {"h_label": ENTITY_PLACEHOLDER, "t_label": ENTITY_PLACEHOLDER, "tau_start": tau_start}
    return template.format_fn(row)
