"""
Load và lọc dữ liệu sự kiện ICEWS cho Việt Nam.

Download tại: https://dataverse.harvard.edu/dataverse/icews
Tìm: "ICEWS Coded Event Data" → chọn năm (vd: 2015) → Download .tab

File .tab đặt vào: data/inherited/

CAMEO code → relation mapping (rút gọn cho Việt Nam):
  https://eventdata.utdallas.edu/cameo.html
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INHERITED_DIR

logger = logging.getLogger(__name__)

# Cột trong file ICEWS .tab
ICEWS_COLS = {
    "Event Date":       "date",
    "Source Name":      "h_label",
    "Source Country":   "h_country",
    "Event Text":       "event_text",
    "CAMEO Code":       "cameo",
    "Intensity":        "intensity",
    "Target Name":      "t_label",
    "Target Country":   "t_country",
    "Country":          "country",
    "Latitude":         "l_t_lat",
    "Longitude":        "l_t_lon",
    "Story ID":         "story_id",
}

# CAMEO root code → quan hệ KG
CAMEO_TO_REL = {
    "01": "makeStatement",
    "02": "appeal",
    "03": "express_intent",
    "04": "consult",
    "05": "engage_diplomacy",
    "06": "cooperate_materially",
    "07": "provide_aid",
    "08": "yield",
    "09": "investigate",
    "10": "demand",
    "11": "disapprove",
    "12": "reject",
    "13": "threaten",
    "14": "protest",
    "15": "exhibit_force",
    "16": "reduce_relations",
    "17": "coerce",
    "18": "assault",
    "19": "fight",
    "20": "mass_violence",
}

VIETNAM_KEYWORDS = {"VIE", "VIET", "VIETNAM", "VIET NAM"}


def _cameo_to_rel(code: str) -> str:
    root = str(code)[:2] if code else ""
    return CAMEO_TO_REL.get(root, "interact")


def load_icews_files(data_dir: str = INHERITED_DIR) -> pd.DataFrame:
    """
    Load tất cả file .tab trong INHERITED_DIR, lọc sự kiện liên quan Việt Nam.

    Returns:
        DataFrame 6-tuple: h, h_label, r, t, t_label, tau_start, tau_end,
                           l_h_lat, l_h_lon, l_t_lat, l_t_lon
        Trả về DataFrame rỗng nếu không có file nào.
    """
    tab_files = [f for f in os.listdir(data_dir) if f.endswith(".tab") or f.endswith(".csv")]
    if not tab_files:
        logger.warning(f"Khong tim thay file ICEWS trong {data_dir}")
        logger.warning("Download tai: https://dataverse.harvard.edu/dataverse/icews")
        return pd.DataFrame()

    all_dfs = []
    for fname in sorted(tab_files):
        fpath = os.path.join(data_dir, fname)
        logger.info(f"  Doc {fname}...")
        try:
            df_raw = pd.read_csv(fpath, sep="\t", low_memory=False, encoding="utf-8",
                                 on_bad_lines="skip")

            # Chuẩn hóa tên cột
            rename = {}
            for orig, new in ICEWS_COLS.items():
                match = [c for c in df_raw.columns if orig.lower() in c.lower()]
                if match:
                    rename[match[0]] = new
            df_raw = df_raw.rename(columns=rename)

            # Lọc Việt Nam
            mask = pd.Series(False, index=df_raw.index)
            for col in ["h_country", "t_country", "country"]:
                if col in df_raw.columns:
                    mask |= df_raw[col].astype(str).str.upper().str.contains(
                        "|".join(VIETNAM_KEYWORDS), na=False
                    )
            df_vn = df_raw[mask].copy()
            logger.info(f"    {len(df_raw)} events -> {len(df_vn)} Vietnam events")
            all_dfs.append(df_vn)

        except Exception as e:
            logger.warning(f"  Loi doc {fname}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Chuyển sang 6-tuple
    result_rows = []
    for _, row in df.iterrows():
        h_label = str(row.get("h_label", "")).strip()
        t_label = str(row.get("t_label", "")).strip()
        if not h_label or not t_label or h_label == t_label:
            continue

        date_str = str(row.get("date", ""))
        try:
            year = int(date_str[:4])
        except Exception:
            year = None

        rel = _cameo_to_rel(str(row.get("cameo", "")))

        # Dùng h_label làm QID tạm (ICEWS không có QID)
        h_id = "ICEWS_" + h_label.replace(" ", "_")[:30]
        t_id = "ICEWS_" + t_label.replace(" ", "_")[:30]

        result_rows.append({
            "h":         h_id,
            "h_label":   h_label,
            "r":         rel,
            "t":         t_id,
            "t_label":   t_label,
            "tau_start": year,
            "tau_end":   year,
            "l_h_lat":   None,
            "l_h_lon":   None,
            "l_t_lat":   row.get("l_t_lat"),
            "l_t_lon":   row.get("l_t_lon"),
            "source":    "ICEWS",
        })

    if not result_rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(result_rows)
    df_out = df_out.drop_duplicates(subset=["h_label", "r", "t_label", "tau_start"])
    logger.info(f"ICEWS: {len(df_out)} unique Vietnam facts")
    return df_out


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df = load_icews_files()
    if df.empty:
        print("Chua co file ICEWS. Dat file .tab vao data/inherited/")
    else:
        print(f"ICEWS Vietnam: {len(df)} facts")
        print(df[["h_label", "r", "t_label", "tau_start"]].head(10).to_string(index=False))
