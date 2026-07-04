"""
L3 — Mapping template_id -> loại câu hỏi cho Bảng 4 (phân loại template) và
Bảng 7 (phân tích lỗi theo loại câu hỏi).

Phân loại theo NỘI DUNG NGÔN NGỮ của câu hỏi (có nhắc τ/năm hay không):
  - "thuan_khong_gian": chỉ hỏi vị trí, KHÔNG nhắc năm trong câu hỏi.
  - "ket_hop": câu hỏi có nhắc năm (τ) VÀ hỏi/cho vị trí — cả 2 chiều
    không gian + thời gian cùng xuất hiện.
  - "thuan_thoi_gian": CHỈ hỏi về thời gian, không có yếu tố không gian.
    ⚠️ KHÔNG có template nào thuộc loại này trong 12 template hiện tại —
    vì thiết kế "entity ranking" (CT8) yêu cầu đáp án luôn là một THỰC
    THỂ, không phải một con số/năm, nên không có câu hỏi "xảy ra năm
    nào?" kiểu trả lời bằng số. Ghi rõ để không bị hiểu nhầm là thiếu sót
    khi review Bảng 4 — đây là hệ quả tất yếu của quyết định kiến trúc.

"cần ảnh" là thuộc tính CẤP FACT (has_image, đã tính ở
build_training_manifest.py), KHÔNG phải cấp template — join riêng ở cấp
câu hỏi, không trộn vào bảng mapping template.

Output:
  - results/dataset/table4_template_category.csv (12 dòng, 1/template)
  - results/dataset/table7_question_type_x_image.csv (câu hỏi theo
    category x has_image, dùng lọc lỗi theo loại cho Bảng 7)
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.export_showcase import load_image_lookup
from src.data.question_templates import TEMPLATES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Template nào có nhắc năm (τ) trong câu hỏi -> "ket_hop", còn lại -> "thuan_khong_gian".
TEMPORAL_TEMPLATE_IDS = {"occurredAt_t4_reverse", "locatedIn_t4"}


def categorize_template(template_id: str) -> str:
    return "ket_hop" if template_id in TEMPORAL_TEMPLATE_IDS else "thuan_khong_gian"


def main():
    rows = []
    for t in TEMPLATES:
        rows.append({
            "template_id": t.template_id,
            "relation": t.relation,
            "mask": t.mask,
            "question_type_ct": t.question_type,          # location_query / entity_query (Bước 2)
            "spatiotemporal_category": categorize_template(t.template_id),  # Bảng 4/7
        })
    template_table = pd.DataFrame(rows)
    template_table.loc[len(template_table)] = {
        "template_id": "(không có)", "relation": "-", "mask": "-",
        "question_type_ct": "-", "spatiotemporal_category": "thuan_thoi_gian",
    }

    out_dir = os.path.join(ROOT, "results", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    t4_path = os.path.join(out_dir, "table4_template_category.csv")
    template_table.to_csv(t4_path, index=False, encoding="utf-8-sig")

    # --- Bảng 7: câu hỏi theo category x has_image ---
    q_path = os.path.join(ROOT, "data", "vistqad", "questions.csv")
    questions = pd.read_csv(q_path)
    cat_map = {t.template_id: categorize_template(t.template_id) for t in TEMPLATES}
    questions["spatiotemporal_category"] = questions["template_id"].map(cat_map)

    image_lookup = load_image_lookup()
    questions["has_image"] = questions["h"].map(image_lookup).fillna("") != ""

    cross = (
        questions.groupby(["spatiotemporal_category", "has_image"])
        .size()
        .reset_index(name="n_questions")
    )
    t7_path = os.path.join(out_dir, "table7_question_type_x_image.csv")
    cross.to_csv(t7_path, index=False, encoding="utf-8-sig")

    print("--- Bảng 4: template -> category ---")
    print(template_table.to_string(index=False))
    print("\n--- Bảng 7: câu hỏi theo category x has_image ---")
    print(cross.to_string(index=False))
    print(f"\nĐã lưu -> {t4_path}\nĐã lưu -> {t7_path}")
    return template_table, cross


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
