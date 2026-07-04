"""
Mở rộng KG bằng cách thêm các quan hệ mới cho các thực thể đã có trong Bước 1+2.

Dùng Wikidata REST API (wbgetentities) thay vì SPARQL để tránh timeout/429.

Quan hệ bổ sung:
  - educatedAt   : học tại (P69)
  - memberOf     : thành viên của tổ chức (P463)
  - heldPosition : chức vụ (P39)
  - receivedAward: giải thưởng (P166)
  - partOf       : thuộc về (P361)
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
BATCH_SIZE   = 50  # wbgetentities hỗ trợ tối đa 50 QID/request

PROPERTY_MAP = {
    "P39":  "heldPosition",    # chức vụ — phong phú nhất
    "P106": "hasOccupation",   # nghề nghiệp
    "P102": "memberOfParty",   # đảng phái
    "P108": "workedFor",       # nơi làm việc
    "P69":  "educatedAt",      # học tại
    "P463": "memberOf",        # thành viên tổ chức
    "P166": "receivedAward",   # giải thưởng
    "P361": "partOf",          # thuộc về
    "P17":  "inCountry",       # thuộc quốc gia (cho di tích/sự kiện)
}


def _get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "stkg_vietnamese/1.0 (research; contact: stkg@example.com)"
    })
    return s


def _vi_label(entity: dict) -> str:
    """Lấy nhãn tiếng Việt, fallback sang tiếng Anh."""
    labels = entity.get("labels", {})
    return (labels.get("vi") or labels.get("en") or {}).get("value", "")


def _extract_qid(snak: dict) -> str | None:
    try:
        return snak["datavalue"]["value"]["id"]
    except (KeyError, TypeError):
        return None


def expand_entities(qids: list[str]) -> pd.DataFrame:
    """
    Lấy thêm quan hệ cho danh sách QIDs qua Wikidata REST API.

    Returns:
        DataFrame với cột: h, h_label, r, t, t_label,
                           tau_start, tau_end, l_h_lat, l_h_lon, l_t_lat, l_t_lon
    """
    session = _get_session()

    # ── Bước 1: Lấy claims cho tất cả QIDs ──────────────────────────
    prop_ids = "|".join(PROPERTY_MAP.keys())
    all_entities: dict[str, dict] = {}

    for i in range(0, len(qids), BATCH_SIZE):
        batch = qids[i: i + BATCH_SIZE]
        params = {
            "action":    "wbgetentities",
            "ids":       "|".join(batch),
            "props":     "claims|labels",
            "languages": "vi|en",
            "format":    "json",
        }
        try:
            resp = session.get(WIKIDATA_API, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            entities = resp.json().get("entities", {})
            all_entities.update(entities)
            logger.info(f"  Batch {i//BATCH_SIZE + 1}/{(len(qids)-1)//BATCH_SIZE + 1}: "
                        f"{len(entities)} entities")
        except Exception as e:
            logger.warning(f"  API error batch {i}: {e}")
        time.sleep(REQUEST_DELAY)

    # ── Bước 2: Trích xuất relations ─────────────────────────────────
    rows: list[dict] = []
    t_qids: set[str] = set()

    for qid, entity in all_entities.items():
        h_label = _vi_label(entity)
        claims  = entity.get("claims", {})

        for prop, rel in PROPERTY_MAP.items():
            for claim in claims.get(prop, []):
                mainsnak = claim.get("mainsnak", {})
                t_qid = _extract_qid(mainsnak)
                if not t_qid:
                    continue

                # Lấy năm từ qualifiers (start time P580) nếu có
                tau = None
                qualifiers = claim.get("qualifiers", {})
                for p580_claim in qualifiers.get("P580", []):
                    try:
                        time_val = p580_claim["datavalue"]["value"]["time"]
                        tau = int(time_val[1:5])
                        break
                    except Exception:
                        pass

                rows.append({
                    "h":         qid,
                    "h_label":   h_label,
                    "r":         rel,
                    "t":         t_qid,
                    "t_label":   "",     # điền sau
                    "tau_start": tau,
                    "tau_end":   tau,
                    "l_h_lat":   None,
                    "l_h_lon":   None,
                    "l_t_lat":   None,
                    "l_t_lon":   None,
                })
                t_qids.add(t_qid)

    if not rows:
        logger.info("  Khong co du lieu expand")
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["h", "r", "t"])
    logger.info(f"  {len(df)} relations, can lay nhan cho {len(t_qids)} t-entities")

    # ── Bước 3: Lấy nhãn cho t-entities ─────────────────────────────
    t_labels: dict[str, str] = {}
    t_list = list(t_qids)
    for i in range(0, len(t_list), BATCH_SIZE):
        batch = t_list[i: i + BATCH_SIZE]
        params = {
            "action":    "wbgetentities",
            "ids":       "|".join(batch),
            "props":     "labels",
            "languages": "vi|en",
            "format":    "json",
        }
        try:
            resp = session.get(WIKIDATA_API, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            for qid, ent in resp.json().get("entities", {}).items():
                t_labels[qid] = _vi_label(ent)
        except Exception as e:
            logger.warning(f"  Label fetch error: {e}")
        time.sleep(REQUEST_DELAY)

    df["t_label"] = df["t"].map(t_labels).fillna("")

    # Bỏ dòng cả h_label lẫn t_label đều rỗng
    df = df[(df["h_label"] != "") | (df["t_label"] != "")].copy()
    logger.info(f"  Wikidata expand xong: {len(df)} rows")
    return df


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_qids = ["Q36014", "Q1858", "Q178561", "Q9199"]
    df = expand_entities(test_qids)
    print(f"Expanded: {len(df)} rows")
    if not df.empty:
        print(df[["h_label", "r", "t_label"]].head(10).to_string(index=False))
