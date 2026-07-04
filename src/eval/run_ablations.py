"""
C4 — Ablation runner (Mục 5 manuscript_methodology.md, 8 kịch bản + kịch
bản 1 baseline = 9 dòng lệnh). Mỗi kịch bản train 1 seed (không phải 3 seed
đầy đủ như C3) để tiết kiệm GPU, theo đúng "ĐƯỢC DUYỆT TRƯỚC" trong yêu cầu.

CHẠY TRÊN COLAB GPU (KHÔNG chạy được ở đây — không có quyền truy cập GPU
thật). Script này chỉ ĐỊNH NGHĨA 8 kịch bản qua override config.yaml và gọi
src/train/train.py cho từng kịch bản — đúng yêu cầu "notebook Colab chỉ gọi
script trong src/, không viết logic riêng".

Ưu tiên chạy trước theo yêu cầu: alpha=1 (kịch bản 5) và no-image (kịch bản 6).
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (tên kịch bản, priority, list override "--set key=value")
SCENARIOS = [
    ("5_alpha_1_pure_semantic", 1, ["retrieval.alpha=1.0"]),
    ("6_no_image_channel", 1, ["ablation.disable_image=true"]),
    ("1_full_model_baseline", 2, []),
    ("2_no_inward_attention", 2, ["ablation.disable_inward_attention=true"]),
    ("3_no_spatial", 2, ["ablation.disable_spatial=true"]),
    ("4_no_temporal", 2, ["ablation.disable_temporal=true"]),
    ("7_no_visual_reliability", 2, ["loss.lambda_vg=0.0", "model.visual_reliability_module.enabled=false"]),
    ("8_qa_only", 2, ["loss.lambda_stkg=0.0", "loss.lambda_vg=0.0"]),
    # Kịch bản 9 (RAG nối chuỗi thay chú ý 2 chiều): kiến trúc khác hẳn CT(7),
    # KHÔNG cài được qua cờ config đơn giản — cần viết forward_one riêng.
    # Xem docs/DECISIONS.md — CHƯA triển khai, cần quyết định thiết kế trước.
]


def run_all(seed: int = 42, dry_run: bool = True, checkpoint_root: str | None = None, results_dir: str | None = None):
    checkpoint_root = checkpoint_root or os.path.join(ROOT, "results", "training", "checkpoints_ablation")
    results_dir = results_dir or os.path.join(ROOT, "results", "training")

    SCENARIOS_sorted = sorted(SCENARIOS, key=lambda s: s[1])
    results = []
    for name, priority, overrides in SCENARIOS_sorted:
        # --tag={name} bắt buộc — nếu không, MỌI kịch bản dùng tag mặc định
        # "default" và GHI ĐÈ LẪN NHAU lên cùng 1 file kết quả (bug đã sửa,
        # xem docs/DECISIONS.md). checkpoint_dir cũng phải tách riêng theo
        # kịch bản vì cùng seed=42 -> cùng tên file seed42_best.pt.
        cmd = [
            sys.executable, "-m", "src.train.train", "--seeds", str(seed), "--tag", name,
            "--set", f"train.checkpoint_dir={os.path.join(checkpoint_root, name)}",
            "--set", f"train.results_dir={results_dir}",
        ]
        for ov in overrides:
            cmd += ["--set", ov]
        print(f"[priority {priority}] {name}: {' '.join(cmd)}")
        if dry_run:
            results.append({"scenario": name, "priority": priority, "cmd": cmd, "status": "DRY_RUN_NOT_EXECUTED"})
            continue
        env = dict(os.environ)
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        results.append({"scenario": name, "priority": priority, "cmd": cmd,
                         "status": "OK" if proc.returncode == 0 else f"FAILED({proc.returncode})"})

    out_path = os.path.join(ROOT, "results", "eval", "ablation_run_log.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu log -> {out_path}")
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true",
                         help="Thực sự chạy (mặc định chỉ in lệnh — dry run, vì cần GPU Colab)")
    parser.add_argument("--checkpoint_root", default=None,
                         help="Thư mục gốc checkpoint (Drive khi chạy Colab) — mỗi kịch bản 1 thư mục con")
    parser.add_argument("--results_dir", default=None,
                         help="Thư mục kết quả (Drive khi chạy Colab), CHUNG cho mọi kịch bản (phân biệt qua --tag)")
    args = parser.parse_args()
    run_all(seed=args.seed, dry_run=not args.execute,
             checkpoint_root=args.checkpoint_root, results_dir=args.results_dir)
