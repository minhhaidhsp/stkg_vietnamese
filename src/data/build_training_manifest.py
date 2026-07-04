"""
Gắn cờ has_image (khớp logic export_showcase.py::load_image_lookup) vào
từng câu hỏi trong data/vistqad/{train,val,test}.csv, lưu ra
data/vistqad/{train,val,test}_manifest.csv — dùng trực tiếp bởi train loop
(Bước 4) để quyết định mẫu nào dùng ảnh thật / null_image_embedding.

Log tỷ lệ có ảnh/không ảnh mỗi tập (yêu cầu bắt buộc trước smoke test).
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.export_showcase import load_image_lookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    image_lookup = load_image_lookup()
    report_rows = []

    for split in ["train", "val", "test"]:
        path = os.path.join(ROOT, "data", "vistqad", f"{split}.csv")
        df = pd.read_csv(path)
        df["image_url"] = df["h"].map(image_lookup).fillna("")
        df["has_image"] = df["image_url"] != ""

        out_path = os.path.join(ROOT, "data", "vistqad", f"{split}_manifest.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        n_total = len(df)
        n_with_image = int(df["has_image"].sum())
        pct = n_with_image / n_total * 100 if n_total else 0.0
        report_rows.append({"split": split, "n_total": n_total, "n_with_image": n_with_image,
                             "pct_with_image": round(pct, 2)})
        logger.info(f"{split}: {n_total} câu hỏi, {n_with_image} có ảnh ({pct:.2f}%) -> {out_path}")

    report = pd.DataFrame(report_rows)
    report_path = os.path.join(ROOT, "data", "vistqad", "manifest_image_coverage_report.csv")
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
