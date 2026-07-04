"""
C5 — Eval đầy đủ (Bảng 5-7, bootstrap p-value, latency, danh sách lỗi).
CHẠY TRÊN COLAB GPU (không chạy được ở đây — cần checkpoint thật từ C3).

Phần CHƯA VIẾT ở bản này (ghi rõ, không giả vờ đã xong):
  - Hình 4 (attention weights) / Hình 5 (phân phối không gian) / Hình 6
    (data scaling 20-100%): cần script vẽ riêng trong src/viz/ (Bước 7),
    đọc CSV/JSON do script này xuất ra — CHƯA viết src/viz/ ở phiên này.
  - So sánh nhiều checkpoint/seed cho bootstrap p-value: cần >= 2 checkpoint
    thật (từ C3, 3 seed) mới chạy được — hiện chưa có checkpoint thật nào.

Đã viết trong bản này:
  - Hit@K/MRR trên tập test (tái sử dụng train.py::evaluate).
  - Đo latency suy luận (thời gian/câu hỏi, trung bình + p50/p95).
  - Xuất danh sách ~100 mẫu dự đoán sai (kèm câu hỏi, đáp án đúng, top-5 dự
    đoán) cho phân loại lỗi thủ công (Bảng 8) -> results/error_analysis/.
  - Bootstrap p-value GHÉP CẶP giữa 2 checkpoint (hàm sẵn có, cần >=2
    checkpoint thật để gọi).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import add_config_args, load_config_from_args
from src.eval.metrics import compute_rank, summarize_ranks
from src.train.train import Modules, build_entity_universe, forward_one

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_checkpoint_into(m: Modules, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=m.device)
    m.ste.load_state_dict(ckpt["ste"])
    m.fusion.load_state_dict(ckpt["fusion"])
    m.ranking_head.load_state_dict(ckpt["ranking_head"])
    m.retriever.load_state_dict(ckpt["retriever"])
    m.reliability.load_state_dict(ckpt["reliability"])
    m.encoder.w_q.load_state_dict(ckpt["w_q"])
    m.encoder.w_v.load_state_dict(ckpt["w_v"])
    with torch.no_grad():
        m.encoder.null_image_embedding.copy_(ckpt["null_image_embedding"])
    lora_sd = m.encoder.text_backbone.state_dict()
    lora_sd.update(ckpt["lora"])
    m.encoder.text_backbone.load_state_dict(lora_sd)
    return ckpt.get("val_mrr")


def eval_test_set(m: Modules, test_df: pd.DataFrame, entity_universe: pd.DataFrame) -> tuple[dict, list[dict]]:
    entity_to_idx = {e: i for i, e in enumerate(entity_universe["entity"])}
    entity_lat = torch.tensor(entity_universe["lat"].values, dtype=torch.float32, device=m.device)
    entity_lon = torch.tensor(entity_universe["lon"].values, dtype=torch.float32, device=m.device)
    entity_tau = torch.tensor(entity_universe["tau"].values, dtype=torch.float32, device=m.device)
    idx_to_entity = {i: e for e, i in entity_to_idx.items()}

    ranks, errors, latencies = [], [], []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            if row["answer_entity"] not in entity_to_idx:
                continue
            t0 = time.time()
            out = forward_one(m, row, test_df, entity_to_idx, entity_lat, entity_lon, entity_tau)
            latencies.append(time.time() - t0)

            p = out["p"].squeeze(0)
            target_idx = out["qa_target"].item()
            rank = compute_rank(p, target_idx)
            ranks.append(rank)

            if rank > 1:
                top5_idx = torch.topk(p, min(5, len(p))).indices.tolist()
                errors.append({
                    "qid": row.get("qid"), "question": row["question"],
                    "correct_answer": row["answer_label"],
                    "top5_predictions": [idx_to_entity[i] for i in top5_idx],
                    "rank_of_correct": rank,
                })

    metrics = summarize_ranks(ranks)
    metrics["latency_ms_mean"] = round(np.mean(latencies) * 1000, 2) if latencies else None
    metrics["latency_ms_p50"] = round(np.percentile(latencies, 50) * 1000, 2) if latencies else None
    metrics["latency_ms_p95"] = round(np.percentile(latencies, 95) * 1000, 2) if latencies else None
    return metrics, errors


def bootstrap_p_value(ranks_a: list[int], ranks_b: list[int], metric_fn, n_boot: int = 1000, seed: int = 0) -> float:
    """Bootstrap ghép cặp: xác suất (dưới H0 hoán vị) mà chênh lệch metric
    giữa A và B lớn bằng hoặc hơn quan sát thật — dùng cho Bảng 5."""
    rng = np.random.RandomState(seed)
    ranks_a, ranks_b = np.array(ranks_a), np.array(ranks_b)
    n = len(ranks_a)
    observed_diff = metric_fn(ranks_b.tolist()) - metric_fn(ranks_a.tolist())

    count = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diff = metric_fn(ranks_b[idx].tolist()) - metric_fn(ranks_a[idx].tolist())
        if abs(diff) >= abs(observed_diff):
            count += 1
    return count / n_boot


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    device = cfg["train"]["device"]

    ckpt_path = os.path.join(args.checkpoint_dir, f"seed{args.seed}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"LỖI: chưa có checkpoint {ckpt_path} — chạy src/train/train.py (C3) trước.")
        sys.exit(1)

    m = Modules(cfg, device)
    val_mrr_at_save = load_checkpoint_into(m, ckpt_path)
    print(f"Đã nạp checkpoint {ckpt_path} (val_mrr lúc lưu: {val_mrr_at_save})")

    train_df = pd.read_csv(os.path.join(ROOT, "data", "vistqad", "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(ROOT, "data", "vistqad", "val_manifest.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "data", "vistqad", "test_manifest.csv"))
    entity_universe = build_entity_universe(pd.concat([train_df, val_df, test_df]))

    metrics, errors = eval_test_set(m, test_df, entity_universe)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    out_dir = os.path.join(ROOT, "results", "eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"test_metrics_seed{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    error_dir = os.path.join(ROOT, "results", "error_analysis")
    os.makedirs(error_dir, exist_ok=True)
    sample_errors = errors[:100]
    with open(os.path.join(error_dir, f"top100_errors_seed{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(sample_errors, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(sample_errors)}/{len(errors)} mẫu lỗi -> {error_dir}")

    return metrics


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
