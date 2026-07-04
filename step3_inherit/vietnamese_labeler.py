"""
Việt hóa nhãn cho các thực thể: điền nhãn tiếng Việt từ Wikidata.

Chỉ áp dụng cho các QID thực (Q...), bỏ qua ICEWS_* và các ID tổng hợp.
"""

import logging
import os
import sys
import time

import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REQUEST_DELAY, TIMEOUT

logger = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
BATCH_SIZE   = 50


def _is_qid(val: str) -> bool:
    return isinstance(val, str) and val.startswith("Q") and val[1:].isdigit()


def fetch_vi_labels(qids: list[str]) -> dict[str, str]:
    """
    Lấy nhãn tiếng Việt cho danh sách QIDs từ Wikidata API.

    Returns:
        dict {qid: vi_label}
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "stkg_vietnamese/1.0 (research; contact: stkg@example.com)"
    })

    result: dict[str, str] = {}
    for i in range(0, len(qids), BATCH_SIZE):
        batch = [q for q in qids[i: i + BATCH_SIZE] if _is_qid(q)]
        if not batch:
            continue
        params = {
            "action":      "wbgetentities",
            "ids":         "|".join(batch),
            "props":       "labels",
            "languages":   "vi|en",
            "format":      "json",
        }
        try:
            resp = session.get(WIKIDATA_API, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            entities = resp.json().get("entities", {})
            for qid, entity in entities.items():
                labels = entity.get("labels", {})
                vi = labels.get("vi", {}).get("value", "")
                en = labels.get("en", {}).get("value", "")
                result[qid] = vi or en
        except Exception as e:
            logger.warning(f"Label fetch error batch {i}: {e}")
        time.sleep(REQUEST_DELAY)

    return result


def fill_vietnamese_labels(df: pd.DataFrame,
                           label_cache: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Điền nhãn tiếng Việt cho cột h_label và t_label còn trống.

    Args:
        df: DataFrame 6-tuple
        label_cache: Cache nhãn đã có (sẽ được cập nhật in-place)

    Returns:
        DataFrame đã cập nhật nhãn.
    """
    if label_cache is None:
        label_cache = {}

    df = df.copy()

    # Tìm các QID cần lấy nhãn
    qids_need = set()
    for col_id, col_label in [("h", "h_label"), ("t", "t_label")]:
        mask_empty = df[col_label].isna() | (df[col_label] == "")
        qids_need.update(
            df.loc[mask_empty & df[col_id].apply(_is_qid), col_id].tolist()
        )

    # Bỏ qua QID đã có trong cache
    qids_to_fetch = [q for q in qids_need if q not in label_cache]
    if qids_to_fetch:
        logger.info(f"Lay nhan VI cho {len(qids_to_fetch)} QIDs...")
        new_labels = fetch_vi_labels(qids_to_fetch)
        label_cache.update(new_labels)
        logger.info(f"  {len(new_labels)} nhan moi")
    else:
        logger.info("Tat ca nhan da co trong cache")

    # Điền nhãn vào DataFrame
    for col_id, col_label in [("h", "h_label"), ("t", "t_label")]:
        mask_empty = df[col_label].isna() | (df[col_label] == "")
        df.loc[mask_empty, col_label] = df.loc[mask_empty, col_id].map(
            lambda q: label_cache.get(q, "")
        )

    # Xóa dòng vẫn không có nhãn
    before = len(df)
    df = df[df["h_label"].notna() & (df["h_label"] != "")].copy()
    df = df[df["t_label"].notna() & (df["t_label"] != "")].copy()
    logger.info(f"Sau Viet hoa: {len(df)}/{before} rows con lai (co nhan)")

    return df


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_qids = ["Q36014", "Q1858", "Q178561", "Q9199", "Q200538"]
    labels = fetch_vi_labels(test_qids)
    for qid, label in labels.items():
        print(f"  {qid}: {label}")
