"""
Đọc toàn bộ results/training/train_results_alpha_*.json (sinh ra từ C2 —
grid search alpha) và chọn alpha có best_val_mrr cao nhất, ghi ra
results/training/best_alpha.txt (1 số duy nhất) để C3 đọc tự động — KHÔNG
cần người copy/dán thủ công giữa các cell notebook.
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    pattern = os.path.join(ROOT, "results", "training", "train_results_alpha_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"LỖI: không tìm thấy file nào khớp {pattern} — chạy C2 (grid search) trước.")
        sys.exit(1)

    best_alpha, best_mrr = None, -1.0
    summary = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            results = json.load(f)
        alpha = results[0]["alpha"]
        mrr = max(r["best_val_mrr"] for r in results)
        summary.append({"alpha": alpha, "best_val_mrr": mrr, "file": os.path.basename(fpath)})
        if mrr > best_mrr:
            best_mrr, best_alpha = mrr, alpha

    print("==RESULT== Bảng grid search alpha:")
    for s in sorted(summary, key=lambda x: x["alpha"]):
        marker = " <-- BEST" if s["alpha"] == best_alpha else ""
        print(f"  alpha={s['alpha']}: val_MRR={s['best_val_mrr']:.4f}{marker}")

    out_path = os.path.join(ROOT, "results", "training", "best_alpha.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(best_alpha))
    print(f"\n==RESULT== BEST_ALPHA={best_alpha} (val_MRR={best_mrr:.4f}) -> ghi vào {out_path}")
    return best_alpha


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
