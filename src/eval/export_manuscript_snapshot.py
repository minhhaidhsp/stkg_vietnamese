"""
Tổng hợp TOÀN BỘ dữ liệu cần cho bản thảo vào MỘT file
results/manuscript_snapshot.json — CHẠY ĐƯỢC BẤT KỲ LÚC NÀO trong quá
trình train, kể cả khi training đang chạy dở ở nơi khác (Colab). Script
CHỈ ĐỌC checkpoint/file kết quả đã lưu trên đĩa (.pt, *_seed*_result.json,
results/dataset/*, results/baselines/*), KHÔNG cần training dừng lại,
KHÔNG tự chạy training/eval.

MỌI trường CHƯA CÓ dữ liệu THẬT = null — KHÔNG suy diễn/nội suy/điền số
giả (nguyên tắc bất biến Mục 6 manuscript_methodology.md).

Cấu trúc output:
  metadata          — timestamp, git commit, is_provisional, missing[]
  training_status   — theo (tag, seed): epoch đã chạy, MRR/Hit@1 tốt nhất
                       hiện có, đã hội tụ (early_stop/max_epochs_reached)
                       hay còn dở (in_progress_interrupted/not_started)
  main_results      — mô hình đề xuất (tag=full_run) + baseline đã chạy
  ablation          — 8 kịch bản Bảng 6 (theo src/eval/run_ablations.SCENARIOS)
  dataset_stats     — Bảng 3/4 (đã có sẵn từ Bước 2/L2/L3)
  figure_data       — Hình 2/4/5/6, null nếu script xuất dữ liệu chưa chạy
                       (Hình 4/5 hiện CHƯA có script xuất — ghi rõ, không giả vờ)
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import load_config
from src.eval.run_ablations import SCENARIOS as ABLATION_SCENARIOS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _read_json(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def unit_status(checkpoint_path: str, result_path: str) -> dict:
    """Trạng thái 1 đơn vị (tag, seed) độc lập, dựa THUẦN vào file trên đĩa."""
    ckpt_exists = os.path.exists(checkpoint_path)
    result = _read_json(result_path)

    if result is not None:
        status = "completed" if result.get("converged") else "in_progress_interrupted"
        return {
            "status": status,
            "checkpoint_path": checkpoint_path if ckpt_exists else None,
            "result_path": result_path,
            "n_epochs_trained": result.get("n_epochs_trained"),
            "max_epochs": result.get("max_epochs"),
            "patience": result.get("patience"),
            "stopped_reason": result.get("stopped_reason"),
            "converged": result.get("converged"),
            "best_val_mrr": result.get("best_val_mrr"),
            "best_val_hit1": result.get("best_val_hit1"),
            "best_val_hit3": result.get("best_val_hit3"),
            "best_val_hit10": result.get("best_val_hit10"),
        }
    if ckpt_exists:
        # Checkpoint tồn tại nhưng KHÔNG có file kết quả -> train_one_seed()
        # chưa chạy xong vòng lặp (bị ngắt giữa chừng, vd Colab mất kết nối) —
        # KHÔNG suy diễn MRR (file kết quả chỉ ghi ở cuối vòng lặp).
        return {
            "status": "in_progress_interrupted", "checkpoint_path": checkpoint_path,
            "result_path": None, "n_epochs_trained": None, "max_epochs": None,
            "patience": None, "stopped_reason": None, "converged": False,
            "best_val_mrr": None, "best_val_hit1": None, "best_val_hit3": None, "best_val_hit10": None,
        }
    return {
        "status": "not_started", "checkpoint_path": None, "result_path": None,
        "n_epochs_trained": None, "max_epochs": None, "patience": None,
        "stopped_reason": None, "converged": None,
        "best_val_mrr": None, "best_val_hit1": None, "best_val_hit3": None, "best_val_hit10": None,
    }


def build_training_status(cfg: dict, checkpoint_root: str, results_dir: str) -> dict:
    out = {}

    # --- C2: grid search alpha (1 seed = 42 theo notebook) ---
    for alpha in cfg["retrieval"]["alpha_grid_search"]:
        tag = f"alpha_{alpha}"
        ckpt = os.path.join(checkpoint_root, f"alpha_grid_{alpha}", "seed42_best.pt")
        res = os.path.join(results_dir, f"{tag}_seed42_result.json")
        out.setdefault(tag, {})["42"] = unit_status(ckpt, res)

    # --- C3: huấn luyện đầy đủ, 3 seed ---
    for seed in cfg["train"]["seeds"]:
        ckpt = os.path.join(checkpoint_root, "full_run", f"seed{seed}_best.pt")
        res = os.path.join(results_dir, f"full_run_seed{seed}_result.json")
        out.setdefault("full_run", {})[str(seed)] = unit_status(ckpt, res)

    return out


def build_main_results(training_status: dict) -> dict:
    full_run = training_status.get("full_run", {})
    seeds_with_data = [s for s, v in full_run.items() if v["best_val_mrr"] is not None]
    best_mrr_per_seed = {s: v["best_val_mrr"] for s, v in full_run.items() if v["best_val_mrr"] is not None}

    proposed = None
    if best_mrr_per_seed:
        best_seed = max(best_mrr_per_seed, key=best_mrr_per_seed.get)
        best = full_run[best_seed]
        proposed = {
            "tag": "full_run", "best_seed": int(best_seed),
            "hit@1": best["best_val_hit1"], "hit@3": best["best_val_hit3"],
            "hit@10": best["best_val_hit10"], "mrr": best["best_val_mrr"],
            "n_seeds_with_data": len(seeds_with_data),
            "n_seeds_expected": len(full_run),
            "note": "MRR/Hit@K trên VALIDATION (checkpoint tốt nhất hiện có), "
                    "KHÔNG phải test set — số liệu test cần src/eval/run_full_eval.py.",
        }

    baseline_path = os.path.join(ROOT, "results", "baselines", "baseline_results.json")
    baselines = _read_json(baseline_path) or []

    return {"proposed_model_val": proposed, "baselines": baselines}


def build_ablation(results_dir: str) -> dict:
    out = {}
    for name, priority, overrides in ABLATION_SCENARIOS:
        res_path = os.path.join(results_dir, f"{name}_seed42_result.json")
        result = _read_json(res_path)
        if result is not None:
            out[name] = {
                "status": "completed" if result.get("converged") else "in_progress_interrupted",
                "priority": priority, "overrides": overrides,
                "best_val_mrr": result.get("best_val_mrr"),
                "best_val_hit1": result.get("best_val_hit1"),
                "best_val_hit3": result.get("best_val_hit3"),
                "best_val_hit10": result.get("best_val_hit10"),
            }
        else:
            out[name] = {
                "status": "not_started", "priority": priority, "overrides": overrides,
                "best_val_mrr": None, "best_val_hit1": None, "best_val_hit3": None, "best_val_hit10": None,
            }
    return out


def build_dataset_stats() -> dict:
    dataset_dir = os.path.join(ROOT, "results", "dataset")
    table3 = _read_json(os.path.join(dataset_dir, "table3_split_stats.json"))

    def _read_csv_records(name):
        p = os.path.join(dataset_dir, name)
        if not os.path.exists(p):
            return None
        return pd.read_csv(p).to_dict("records")

    return {
        "table3_split_stats": table3,
        "table4_template_category": _read_csv_records("table4_template_category.csv"),
        "table7_question_type_x_image": _read_csv_records("table7_question_type_x_image.csv"),
        "auto_checks": _read_json(os.path.join(dataset_dir, "auto_checks.json")),
        "human_validation": _read_json(os.path.join(dataset_dir, "human_validation.json")),
    }


def build_figure_data() -> dict:
    eval_dir = os.path.join(ROOT, "results", "eval")
    latency_files = sorted(glob.glob(os.path.join(eval_dir, "test_metrics_seed*.json")))
    figure2 = None
    if latency_files:
        metrics = [_read_json(p) for p in latency_files]
        metrics = [m for m in metrics if m]
        if metrics:
            figure2 = {
                "source_files": [os.path.basename(p) for p in latency_files],
                "latency_ms_mean": [m.get("latency_ms_mean") for m in metrics],
                "latency_ms_p50": [m.get("latency_ms_p50") for m in metrics],
                "latency_ms_p95": [m.get("latency_ms_p95") for m in metrics],
            }

    return {
        "figure2_latency": figure2,
        "figure2_note": None if figure2 else "Chưa có — chạy src/eval/run_full_eval.py trước (cần checkpoint C3).",
        "figure4_attention_by_layer": None,
        "figure4_note": "CHƯA CÓ SCRIPT xuất attention weights theo layer cho 1 câu hỏi ví dụ — "
                         "cần viết thêm trong src/eval/ hoặc src/viz/ trước khi có dữ liệu.",
        "figure5_spatial_distribution": None,
        "figure5_note": "CHƯA CÓ SCRIPT xuất phân phối không gian cho 1 câu hỏi ví dụ — "
                         "cần viết thêm trong src/eval/ hoặc src/viz/ trước khi có dữ liệu.",
        "figure6_data_scaling": None,
        "figure6_note": "CHƯA chạy — cần train nhiều lần với tỷ lệ dữ liệu 20/40/60/80/100% "
                         "(C5 mục 5, đã duyệt trước nhưng chưa có script con riêng cho scaling).",
    }


def build_missing_list(training_status: dict, main_results: dict, ablation: dict, figure_data: dict) -> list[str]:
    missing = []

    grid_missing = [tag for tag, seeds in training_status.items()
                     if tag.startswith("alpha_") for s, v in seeds.items() if v["status"] != "completed"]
    if grid_missing:
        missing.append(f"C2 (grid search alpha): {len(grid_missing)}/5 điểm chưa hoàn tất — {sorted(set(grid_missing))}")

    full_run = training_status.get("full_run", {})
    incomplete_seeds = [s for s, v in full_run.items() if v["status"] != "completed"]
    if incomplete_seeds:
        missing.append(f"C3 (huấn luyện đầy đủ): {len(incomplete_seeds)}/{len(full_run)} seed chưa hoàn tất "
                        f"(seed {sorted(incomplete_seeds)})")

    baselines = main_results.get("baselines", [])
    not_ok = [b["name"] for b in baselines if b.get("status") != "OK"]
    if not_ok:
        missing.append(f"Baseline: {len(not_ok)}/{len(baselines)} chưa chạy được — {not_ok}")
    elif not baselines:
        missing.append("Baseline: chưa có kết quả nào (chạy src/baselines/run_baselines.py)")

    ablation_not_done = [name for name, v in ablation.items() if v["status"] != "completed"]
    if ablation_not_done:
        missing.append(f"Ablation (Bảng 6): {len(ablation_not_done)}/{len(ablation)} kịch bản chưa chạy xong "
                        f"— {ablation_not_done}")

    for fig_key, note_key in [("figure2_latency", "figure2_note"), ("figure4_attention_by_layer", "figure4_note"),
                                ("figure5_spatial_distribution", "figure5_note"), ("figure6_data_scaling", "figure6_note")]:
        if figure_data.get(fig_key) is None:
            missing.append(f"{note_key.replace('_note','')}: {figure_data[note_key]}")

    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", default=os.path.join(ROOT, "results", "training", "checkpoints"),
                         help="Thư mục gốc checkpoint C2/C3 (Drive khi Colab), khớp CHECKPOINT_ROOT trong notebook")
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "results", "training"),
                         help="Thư mục kết quả (Drive khi Colab), khớp RESULTS_ROOT trong notebook")
    args = parser.parse_args()

    cfg = load_config()

    training_status = build_training_status(cfg, args.checkpoint_root, args.results_dir)
    main_results = build_main_results(training_status)
    ablation = build_ablation(args.results_dir)
    dataset_stats = build_dataset_stats()
    figure_data = build_figure_data()
    missing = build_missing_list(training_status, main_results, ablation, figure_data)

    snapshot = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "git_commit": get_git_commit(),
            "checkpoint_root_scanned": args.checkpoint_root,
            "results_dir_scanned": args.results_dir,
            "is_provisional": len(missing) > 0,
            "missing": missing,
        },
        "training_status": training_status,
        "main_results": main_results,
        "ablation": ablation,
        "dataset_stats": dataset_stats,
        "figure_data": figure_data,
    }

    out_path = os.path.join(ROOT, "results", "manuscript_snapshot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def count_nulls(obj, path=""):
        n_null, n_data = 0, 0
        if isinstance(obj, dict):
            for k, v in obj.items():
                nn, nd = count_nulls(v, f"{path}.{k}")
                n_null += nn
                n_data += nd
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                nn, nd = count_nulls(v, f"{path}[{i}]")
                n_null += nn
                n_data += nd
        else:
            if obj is None:
                n_null += 1
            else:
                n_data += 1
        return n_null, n_data

    n_null, n_data = count_nulls(snapshot)
    print(f"Đã lưu snapshot -> {out_path}")
    print(f"Tổng số giá trị lá (leaf values): {n_null + n_data} — {n_data} có dữ liệu, {n_null} null")
    print(f"is_provisional = {snapshot['metadata']['is_provisional']}")
    print("\nCòn thiếu (missing):")
    for m in missing:
        print(f"  - {m}")

    return snapshot


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
