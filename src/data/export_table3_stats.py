"""
L2 — Thống kê chia tập cho Bảng 3 bản thảo: facts/thực thể/câu hỏi theo
train/val/test, câu hỏi theo 4 quan hệ theo tập, độ dài trung bình câu hỏi
theo tập (số từ, tách theo khoảng trắng — tiếng Việt không cần tokenizer
phức tạp cho mục đích thống kê mô tả này).

Output: results/dataset/table3_split_stats.json + .csv (dễ dán vào LaTeX/Word)
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def word_count(text: str) -> int:
    return len(str(text).split())


def main():
    rows = []
    per_relation_rows = []

    for split in ["train", "val", "test"]:
        path = os.path.join(ROOT, "data", "vistqad", f"{split}_manifest.csv")
        df = pd.read_csv(path)

        n_questions = len(df)
        n_facts = df["fact_id"].nunique()
        n_entities = len(set(df["h"].astype(str)) | set(df["t"].astype(str)))
        avg_len = df["question"].apply(word_count).mean()

        rows.append({
            "split": split,
            "n_facts": n_facts,
            "n_entities": n_entities,
            "n_questions": n_questions,
            "avg_question_len_words": round(avg_len, 2),
        })

        rel_counts = df["relation"].value_counts()
        for rel in ["bornIn", "diedIn", "occurredAt", "locatedIn"]:
            per_relation_rows.append({
                "split": split,
                "relation": rel,
                "n_questions": int(rel_counts.get(rel, 0)),
                "pct_of_split": round(rel_counts.get(rel, 0) / n_questions * 100, 2) if n_questions else 0.0,
            })

    overview = pd.DataFrame(rows)
    per_relation = pd.DataFrame(per_relation_rows)

    out_dir = os.path.join(ROOT, "results", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    overview.to_csv(os.path.join(out_dir, "table3_split_overview.csv"), index=False, encoding="utf-8-sig")
    per_relation.to_csv(os.path.join(out_dir, "table3_split_by_relation.csv"), index=False, encoding="utf-8-sig")

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "overview": overview.to_dict("records"),
        "by_relation": per_relation.to_dict("records"),
    }
    with open(os.path.join(out_dir, "table3_split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("--- Bảng 3: Tổng quan theo tập ---")
    print(overview.to_string(index=False))
    print("\n--- Bảng 3: Theo quan hệ ---")
    print(per_relation.to_string(index=False))
    print(f"\nĐã lưu -> {out_dir}/table3_*.csv, table3_split_stats.json")
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
