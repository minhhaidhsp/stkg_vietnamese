"""
Chia data/vistqad/questions.csv thành train/val/test theo tỷ lệ trong
config.yaml (data.split), stratified theo `relation` để mỗi tập giữ đúng tỷ
lệ 4 quan hệ. Chia theo fact_id (không theo qid) để các câu hỏi sinh từ CÙNG
một fact luôn nằm chung một tập — tránh rò rỉ dữ liệu (data leakage) giữa
train/val/test khi 1 fact sinh ra nhiều câu hỏi.

Output: data/vistqad/{train,val,test}.csv
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_by_fact(questions: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    facts = questions[["fact_id", "relation"]].drop_duplicates("fact_id")
    rng = np.random.RandomState(seed)

    assign = {}
    for relation, group in facts.groupby("relation"):
        fact_ids = group["fact_id"].to_numpy().copy()
        rng.shuffle(fact_ids)
        n = len(fact_ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for fid in fact_ids[:n_train]:
            assign[fid] = "train"
        for fid in fact_ids[n_train:n_train + n_val]:
            assign[fid] = "val"
        for fid in fact_ids[n_train + n_val:]:
            assign[fid] = "test"

    questions = questions.copy()
    questions["split"] = questions["fact_id"].map(assign)
    return questions


def main():
    cfg = load_config()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    q_path = os.path.join(root, "data", "vistqad", "questions.csv")
    questions = pd.read_csv(q_path)
    logger.info(f"Đọc {len(questions)} câu hỏi từ {q_path}")

    split_cfg = cfg["data"]["split"]
    questions = split_by_fact(questions, split_cfg["train"], split_cfg["val"], split_cfg["seed"])

    out_dir = os.path.join(root, "data", "vistqad")
    for name in ["train", "val", "test"]:
        sub = questions[questions["split"] == name].drop(columns=["split"])
        path = os.path.join(out_dir, f"{name}.csv")
        sub.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"{name}: {len(sub)} câu hỏi -> {path}")

    print("\n--- Kiểm tra tỷ lệ theo quan hệ trong từng tập ---")
    print(pd.crosstab(questions["relation"], questions["split"], normalize="index").round(3))

    n_facts = questions["fact_id"].nunique()
    n_facts_leak = 0
    for name_a, name_b in [("train", "val"), ("train", "test"), ("val", "test")]:
        a = set(questions[questions["split"] == name_a]["fact_id"])
        b = set(questions[questions["split"] == name_b]["fact_id"])
        n_facts_leak += len(a & b)
    print(f"\nSố fact_id bị rò rỉ giữa các tập (phải = 0): {n_facts_leak}")

    return questions


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
