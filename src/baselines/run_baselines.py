"""
Runner tổng hợp B1 (bắt buộc) + B2 (best-effort) — chạy từng baseline trên
tập test, tính Hit@1 (mọi nhóm) + Hit@3/10/MRR (chỉ nhóm 1, có xếp hạng).

Baseline nào raise NotImplementedError hoặc lỗi runtime -> GHI LOG lý do,
KHÔNG có dòng trong bảng kết quả cuối (không suy diễn/điền số thay).
"""

import json
import os
import sys
import time
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.baselines.base import free_text_hit_at_1
from src.baselines.group2_best_effort import CronKGQA, EmbedKGQA, GenTKGQA, TempoQR
from src.baselines.qwen25vl_zeroshot import Qwen25VLZeroShot
from src.baselines.subgtr import SubGTR
from src.baselines.think_on_graph import ThinkOnGraph
from src.baselines.vintern_zeroshot import VinternZeroShot
from src.config import load_config
from src.eval.metrics import summarize_ranks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

B1_REQUIRED = [ThinkOnGraph, VinternZeroShot, Qwen25VLZeroShot, SubGTR]
B2_BEST_EFFORT = [EmbedKGQA, CronKGQA, TempoQR, GenTKGQA]


def run_baseline(cls, cfg: dict, test_df: pd.DataFrame, max_samples: int = 200) -> dict:
    log = {"name": cls.name, "group": cls.group, "output_type": cls.output_type}
    t0 = time.time()
    try:
        wrapper = cls(cfg)
    except NotImplementedError as e:
        log["status"] = "SKIPPED_NOT_IMPLEMENTED"
        log["reason"] = str(e)
        return log
    except Exception as e:
        log["status"] = "SKIPPED_SETUP_ERROR"
        log["reason"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return log

    sub = test_df.sample(n=min(max_samples, len(test_df)), random_state=0)
    hits, ranks = [], []
    for _, row in sub.iterrows():
        try:
            pred = wrapper.predict(row["question"], None, [])
        except Exception as e:
            log["status"] = "FAILED_DURING_PREDICT"
            log["reason"] = f"{type(e).__name__}: {e}"
            return log
        if cls.output_type == "free_text":
            hits.append(free_text_hit_at_1(pred, row["answer_label"]))
        else:
            ranks.append(pred.index(row["answer_entity"]) + 1 if row["answer_entity"] in pred else len(pred) + 1)

    log["status"] = "OK"
    log["n_samples"] = len(sub)
    log["elapsed_s"] = round(time.time() - t0, 1)
    if cls.output_type == "free_text":
        log["hit@1"] = round(sum(hits) / len(hits), 4) if hits else None
        log["note"] = "Chỉ có Hit@1 (không có xếp hạng đầy đủ) — đúng Mục 4.2 manuscript."
    else:
        log.update(summarize_ranks(ranks))
    return log


def main():
    cfg = load_config()
    test_df = pd.read_csv(os.path.join(ROOT, "data", "vistqad", "test_manifest.csv"))

    results = []
    print("=== B1 (bắt buộc) ===")
    for cls in B1_REQUIRED:
        r = run_baseline(cls, cfg, test_df)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        results.append(r)

    print("\n=== B2 (best-effort) ===")
    for cls in B2_BEST_EFFORT:
        r = run_baseline(cls, cfg, test_df)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        results.append(r)

    out_path = os.path.join(ROOT, "results", "baselines", "baseline_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu -> {out_path}")

    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\nTổng kết: {n_ok}/{len(results)} baseline chạy được thật. "
          f"Baseline KHÔNG chạy được sẽ KHÔNG có dòng trong bảng kết quả cuối.")
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
