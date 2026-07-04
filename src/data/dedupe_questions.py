"""
Xử lý 8,2% câu hỏi trùng lặp (auto_checks phát hiện: text giống hệt nhau,
khác fact_id, do 1 sự kiện có nhiều địa điểm hợp lệ trong Wikidata P276).

PHƯƠNG ÁN ĐÃ CHỐT: với mỗi nhóm câu hỏi trùng text, GIỮ bản ghi có fact_id
thuộc NGUỒN ưu tiên cao nhất theo thứ tự Wikidata > Wikipedia timeline >
Wikipedia heritage (dùng cột __file của data/spatial/enriched.csv để xác
định nguồn — đáng tin hơn cột "source" vì "source" chỉ được set cho facts
gốc Wikipedia, facts Wikidata có source=NaN). Trong cùng 1 nguồn, giữ
fact_id nhỏ nhất (thứ tự thu thập gốc). LOẠI các bản còn lại khỏi TẬP CÂU
HỎI — KHÔNG loại fact khỏi data/spatial/enriched.csv (đồ thị G giữ nguyên).

Output: ghi đè data/vistqad/questions.csv (bản đã làm sạch), log số liệu +
5 ví dụ vào docs/DECISIONS.md.
"""

import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Thứ tự ưu tiên nguồn (số nhỏ hơn = ưu tiên giữ trước)
SOURCE_PRIORITY = {
    "historical_figures.csv": 1,
    "landmarks.csv": 1,
    "historical_events.csv": 1,
    "historical_events_timeline.csv": 2,
    "heritage_sites.csv": 3,
}


def main():
    q_path = os.path.join(ROOT, "data", "vistqad", "questions.csv")
    enriched_path = os.path.join(ROOT, "data", "spatial", "enriched.csv")

    questions = pd.read_csv(q_path)
    enriched = pd.read_csv(enriched_path)
    n_before = len(questions)

    fact_source = enriched["__file"].to_dict()  # fact_id (index) -> __file
    questions["_source_file"] = questions["fact_id"].map(fact_source)
    questions["_priority"] = questions["_source_file"].map(SOURCE_PRIORITY).fillna(99).astype(int)

    dup_mask = questions.duplicated(subset=["question"], keep=False)
    dup_groups = questions[dup_mask].groupby("question")

    # 5 ví dụ cụ thể trước khi loại (để ghi log)
    examples = []
    for question_text, group in dup_groups:
        if len(examples) >= 5:
            break
        examples.append({
            "question": question_text,
            "fact_ids": group["fact_id"].tolist(),
            "sources": group["_source_file"].tolist(),
            "relation": group["relation"].iloc[0],
        })

    # Sắp theo priority rồi fact_id tăng dần, giữ dòng ĐẦU TIÊN mỗi nhóm trùng text
    questions_sorted = questions.sort_values(["_priority", "fact_id"])
    kept = questions_sorted.drop_duplicates(subset=["question"], keep="first")
    kept = kept.sort_values("qid").reset_index(drop=True)

    removed = questions[~questions["qid"].isin(kept["qid"])]
    n_removed = len(removed)
    removed_by_relation = removed["relation"].value_counts().to_dict()

    kept_final = kept.drop(columns=["_source_file", "_priority"])
    kept_final.to_csv(q_path, index=False, encoding="utf-8-sig")

    logger.info(f"Trước: {n_before} câu hỏi. Loại: {n_removed}. Còn lại: {len(kept_final)}.")
    logger.info(f"Loại theo quan hệ: {removed_by_relation}")

    # --- Ghi log vào docs/DECISIONS.md ---
    lines = []
    lines.append(
        f"- 2026-07-04 (xử lý trùng lặp câu hỏi, phương án đã chốt): loại "
        f"{n_removed}/{n_before} câu hỏi trùng text (giữ bản ghi nguồn ưu tiên "
        f"cao nhất: Wikidata > Wikipedia timeline > Wikipedia heritage, trong "
        f"cùng nguồn giữ fact_id nhỏ nhất). KHÔNG loại fact khỏi "
        f"data/spatial/enriched.csv (đồ thị G giữ nguyên {len(enriched)} facts). "
        f"Loại theo quan hệ: {removed_by_relation}. Còn lại {len(kept_final)} câu hỏi."
    )
    lines.append("  5 ví dụ cụ thể (câu hỏi + fact_id trùng + nguồn):")
    for ex in examples:
        lines.append(f"    - \"{ex['question']}\" ({ex['relation']}): fact_id={ex['fact_ids']}, nguồn={ex['sources']}")

    with open(os.path.join(ROOT, "docs", "DECISIONS.md"), "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")
    logger.info("Đã ghi log vào docs/DECISIONS.md")

    print(f"\nTổng câu hỏi cuối cùng: {len(kept_final)}")
    print(kept_final["relation"].value_counts())
    return kept_final, removed_by_relation, examples


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
