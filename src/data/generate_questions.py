"""
Sinh câu hỏi ViSTQAD từ data/spatial/enriched.csv (6.980 facts thật).

Mỗi fact có thể sinh NHIỀU câu hỏi (1 câu / template áp dụng được), việc này
tăng tổng số câu hỏi và — vì occurredAt/locatedIn có nhiều template hơn
bornIn/diedIn (xem question_templates.py) — giúp cân bằng lại phân bố SỐ
CÂU HỎI theo quan hệ so với phân bố facts gốc.

Log rõ số facts/template bị loại và lý do (thiếu trường bắt buộc, nhãn
rỗng, hoặc đáp án nhập nhằng đối với template đảo chiều).

Output: data/vistqad/questions.csv
"""

import csv
import logging
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import load_config
from src.data.question_templates import TEMPLATES, templates_for_relation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROMPT_WRAPPER = (
    "Dựa trên hình ảnh và tri thức không gian thời gian liên quan, hãy trả lời "
    "câu hỏi sau bằng MỘT thực thể duy nhất: {question}"
)


def _row_ok(row: pd.Series, template) -> tuple[bool, str]:
    if not str(row.get("h_label", "")).strip() or not str(row.get("t_label", "")).strip():
        return False, "thieu_nhan_h_hoac_t"
    for field in template.requires:
        val = row.get(field)
        if pd.isna(val):
            return False, f"thieu_truong_{field}"
    return True, ""


def generate(enriched: pd.DataFrame) -> tuple[list[dict], Counter]:
    skip_reasons: Counter = Counter()
    rows = []
    qid = 0

    # Tính uniqueness cho template đảo chiều occurredAt_t4_reverse:
    # chỉ sinh câu hỏi "năm X tại địa điểm Y -> sự kiện nào?" khi (t_label,
    # tau_start) xác định DUY NHẤT một fact occurredAt trong toàn bộ dữ liệu.
    occ = enriched[enriched["r"] == "occurredAt"]
    key_counts = occ.groupby(["t_label", "tau_start"]).size()
    unique_keys = set(key_counts[key_counts == 1].index)

    for fact_id, row in enriched.iterrows():
        relation = row["r"]
        for template in templates_for_relation(relation):
            ok, reason = _row_ok(row, template)
            if not ok:
                skip_reasons[f"{template.template_id}:{reason}"] += 1
                continue

            if template.template_id == "occurredAt_t4_reverse":
                key = (row["t_label"], row["tau_start"])
                if key not in unique_keys:
                    skip_reasons[f"{template.template_id}:dap_an_nhap_nhang"] += 1
                    continue

            question = template.format_fn(row)
            if template.mask == "t":
                answer_entity, answer_label = row["t"], row["t_label"]
            else:
                answer_entity, answer_label = row["h"], row["h_label"]

            qid += 1
            rows.append({
                "qid": f"q{qid:06d}",
                "fact_id": fact_id,
                "relation": relation,
                "template_id": template.template_id,
                "question_type": template.question_type,
                "question": question,
                "prompt": PROMPT_WRAPPER.format(question=question),
                "answer_entity": answer_entity,
                "answer_label": answer_label,
                "h": row["h"], "h_label": row["h_label"],
                "t": row["t"], "t_label": row["t_label"],
                "tau_start": row.get("tau_start"),
                "l_h_lat": row.get("l_h_lat"), "l_h_lon": row.get("l_h_lon"),
                "l_t_lat": row.get("l_t_lat"), "l_t_lon": row.get("l_t_lon"),
                "source": row.get("source") or row.get("__file", ""),
            })

    return rows, skip_reasons


def main():
    cfg = load_config()
    enriched_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        cfg["data"]["enriched_csv"],
    )
    enriched = pd.read_csv(enriched_path)
    logger.info(f"Đọc {len(enriched)} facts từ {enriched_path}")

    rows, skip_reasons = generate(enriched)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "vistqad",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "questions.csv")
    cols = ["qid", "fact_id", "relation", "template_id", "question_type", "question", "prompt",
            "answer_entity", "answer_label", "h", "h_label", "t", "t_label", "tau_start",
            "l_h_lat", "l_h_lon", "l_t_lat", "l_t_lon", "source"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    total_skipped = sum(skip_reasons.values())
    logger.info(f"Sinh được {len(rows)} câu hỏi từ {len(enriched)} facts -> {out_path}")
    logger.info(f"Tổng lượt template bị bỏ qua: {total_skipped}")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  - {reason}: {count}")

    df = pd.DataFrame(rows)
    print("\n--- Phân bố câu hỏi theo quan hệ ---")
    print(df["relation"].value_counts())
    print("\n--- Phân bố câu hỏi theo loại (question_type) ---")
    print(df["question_type"].value_counts())

    return df, skip_reasons


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
