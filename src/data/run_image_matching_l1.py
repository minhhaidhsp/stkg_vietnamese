"""
L1 (autonomous mode) — Wikimedia image matching cho các thực thể MỚI
(~4.898 QID chưa có trong data/visual/url_cache.json, tức chưa được xử lý
ở vòng 652 facts cũ). ĐÓNG KHUNG 24 GIỜ kể từ lúc bắt đầu chạy script này:
hết giờ thì dừng tải ảnh, đóng băng manifest has_image với kết quả đã có.

Có thể dừng giữa chừng (Ctrl-C / mất phiên) và CHẠY LẠI: url_cache.json và
file ảnh local đã tải đều được tái sử dụng (không tải lại), nên tổng thời
gian tích lũy qua nhiều lần chạy vẫn tính đúng nếu resume trong vòng 24h kể
từ SESSION_START ghi trong results/dataset/l1_session_start.txt (không tự
đặt lại đồng hồ mỗi lần resume).

Output:
  - data/visual/images/<QID>.jpg cho các thực thể tải được ảnh
  - data/visual/url_cache.json cập nhật (như cũ)
  - results/dataset/l1_image_matching_report.json (tỷ lệ phủ ảnh cuối)
  - Cập nhật has_image trong data/vistqad/{train,val,test}_manifest.csv
    (chạy lại build_training_manifest.py sau khi L1 xong/hết giờ)
"""

import json
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SPATIAL_DIR  # noqa: E402  (cần sys.path ở trên)
from step3_visual.image_collector import ImageCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEADLINE_HOURS = 24
SESSION_START_PATH = os.path.join(ROOT, "results", "dataset", "l1_session_start.txt")


def _qid_like(s) -> bool:
    return isinstance(s, str) and s.startswith("Q") and s[1:].isdigit()


def get_session_start() -> float:
    os.makedirs(os.path.dirname(SESSION_START_PATH), exist_ok=True)
    if os.path.exists(SESSION_START_PATH):
        with open(SESSION_START_PATH) as f:
            return float(f.read().strip())
    now = time.time()
    with open(SESSION_START_PATH, "w") as f:
        f.write(str(now))
    return now


def build_entity_list(enriched: pd.DataFrame) -> list[dict]:
    label_map = {}
    for _, row in enriched.iterrows():
        if _qid_like(row["h"]):
            label_map.setdefault(row["h"], row["h_label"])
        if _qid_like(row["t"]):
            label_map.setdefault(row["t"], row["t_label"])
    return [{"qid": q, "label": lbl} for q, lbl in label_map.items()]


def main():
    session_start = get_session_start()
    deadline = session_start + DEADLINE_HOURS * 3600
    remaining_h = (deadline - time.time()) / 3600
    logger.info(f"L1 image matching — session bắt đầu lúc {time.ctime(session_start)}, "
                f"còn lại {remaining_h:.2f}h trước deadline 24h.")
    if remaining_h <= 0:
        logger.warning("ĐÃ HẾT 24h — không chạy thêm, chỉ đóng băng báo cáo từ trạng thái hiện có.")

    enriched = pd.read_csv(os.path.join(SPATIAL_DIR, "enriched.csv"))
    entities = build_entity_list(enriched)
    logger.info(f"Tổng số thực thể (QID) trong dataset: {len(entities)}")

    collector = ImageCollector()
    if remaining_h > 0:
        result = collector.collect_batch(entities, skip_existing=True, deadline=deadline)
    else:
        result = {}

    # Tính tỷ lệ phủ ảnh cuối cùng (từ url_cache, không phụ thuộc file local có tải kịp hay không —
    # "has_image" cho manifest huấn luyện dùng khi CÓ URL, khớp logic export_showcase.load_image_lookup)
    with open(os.path.join(ROOT, "data", "visual", "url_cache.json"), encoding="utf-8") as f:
        url_cache = json.load(f)
    all_qids = {e["qid"] for e in entities}
    with_url = {q for q in all_qids if url_cache.get(q)}
    with_local_file = {q for q in all_qids if os.path.exists(collector._local_path(q))}

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "session_start": time.ctime(session_start),
        "deadline_hours": DEADLINE_HOURS,
        "stopped_by_deadline": getattr(collector, "_last_run_stopped_by_deadline", False),
        "n_entities_total": len(all_qids),
        "n_entities_with_image_url": len(with_url),
        "pct_with_image_url": round(len(with_url) / len(all_qids) * 100, 2) if all_qids else 0.0,
        "n_entities_with_local_file_downloaded": len(with_local_file),
        "pct_with_local_file": round(len(with_local_file) / len(all_qids) * 100, 2) if all_qids else 0.0,
    }
    out_path = os.path.join(ROOT, "results", "dataset", "l1_image_matching_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu báo cáo -> {out_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
