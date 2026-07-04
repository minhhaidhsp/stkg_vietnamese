"""
Lấy mẫu 500 câu hỏi (stratified theo relation, tỷ lệ tương ứng với phân bố
thật trong data/vistqad/questions.csv) để con người thẩm định thủ công.

Output: data/vistqad/validation_sample_500.csv — có 2 cột rỗng
"annotator1_valid" / "annotator2_valid" (1 = câu hỏi + đáp án hợp lý/tự
nhiên, 0 = không) để 2 người đánh giá độc lập điền tay. Sau khi điền xong,
chạy compute_kappa.py để tính Cohen's κ — KHÔNG mô phỏng/tạo sẵn giá trị
annotator ở đây, vì đó sẽ là bịa số liệu thực nghiệm.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 500


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    relations = df["relation"].value_counts(normalize=True)
    quotas = {r: min(int(round(frac * n)), (df["relation"] == r).sum()) for r, frac in relations.items()}
    # Bù phần chênh lệch do làm tròn vào quan hệ có nhiều câu hỏi nhất, để tổng đúng n.
    diff = n - sum(quotas.values())
    if diff != 0:
        biggest = relations.idxmax()
        quotas[biggest] += diff

    parts = []
    for relation, k in quotas.items():
        group = df[df["relation"] == relation]
        idx = rng.choice(group.index, size=k, replace=False)
        parts.append(df.loc[idx])
    sample = pd.concat(parts)
    return sample.sample(frac=1, random_state=seed)  # xáo trộn thứ tự


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    q_path = os.path.join(root, "data", "vistqad", "questions.csv")
    df = pd.read_csv(q_path)
    logger.info(f"Đọc {len(df)} câu hỏi từ {q_path}")

    sample = stratified_sample(df, SAMPLE_SIZE, seed=42)
    sample = sample[["qid", "relation", "template_id", "question", "answer_label"]].copy()
    sample["annotator1_valid"] = ""
    sample["annotator1_notes"] = ""
    sample["annotator2_valid"] = ""
    sample["annotator2_notes"] = ""

    out_path = os.path.join(root, "data", "vistqad", "validation_sample_500.csv")
    sample.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lấy mẫu {len(sample)} câu hỏi -> {out_path}")
    logger.info("CẦN 2 người thẩm định độc lập điền cột annotator1_valid / annotator2_valid "
                "(1=hợp lý, 0=không hợp lý) trước khi chạy compute_kappa.py.")

    print("\n--- Phân bố mẫu 500 theo quan hệ ---")
    print(sample["relation"].value_counts())
    return sample


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
