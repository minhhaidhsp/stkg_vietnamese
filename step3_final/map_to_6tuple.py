"""
Map Visual Triplets → 6-ngôi (h, r, t, τ, l_h, l_t).
Merge với bộ facts từ Bước 1+2 (data/spatial/enriched.csv).

Output: data/vivqa/vi_stkg_multimodal.csv
"""

import json
import logging
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPATIAL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIVQA_DIR     = os.path.join(BASE_DIR, "data", "vivqa")
TRIPLET_PATH  = os.path.join(VIVQA_DIR, "visual_triplets.json")
SPATIAL_PATH  = os.path.join(SPATIAL_DIR, "enriched.csv")
OUTPUT_PATH   = os.path.join(VIVQA_DIR, "vi_stkg_multimodal.csv")

# Tọa độ mặc định trung tâm Việt Nam
VN_DEFAULT = (14.0583, 108.2772)


class Mapper:
    def __init__(self):
        self.stkg = pd.read_csv(SPATIAL_PATH)
        logger.info(f"STKG Buoc 1+2: {len(self.stkg)} facts")

        # Build index tìm kiếm tọa độ theo label
        self._coord_index: dict[str, tuple] = {}
        for _, row in self.stkg.iterrows():
            for col_id, lat_col, lon_col in [
                ("h_label", "l_h_lat", "l_h_lon"),
                ("t_label", "l_t_lat", "l_t_lon"),
            ]:
                label = str(row.get(col_id, "")).lower().strip()
                if label and pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)):
                    self._coord_index[label] = (float(row[lat_col]), float(row[lon_col]))

    def _get_coords(self, entity: str) -> tuple[float, float]:
        """Tìm tọa độ gần đúng cho entity, fallback VN trung tâm."""
        key = entity.lower().strip()
        if key in self._coord_index:
            return self._coord_index[key]
        # Partial match
        for label, coords in self._coord_index.items():
            if key in label or label in key:
                return coords
        return VN_DEFAULT

    def _get_timestamp(self, img_id: str, caption: str = "") -> int | None:
        """Lấy năm: tìm trong caption, fallback 2014 (COCO val2014)."""
        text = caption + " " + img_id
        years = re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", text)
        if years:
            return int(years[0])
        return 2014  # COCO val2014 default

    def map_triplets_to_6tuple(self) -> pd.DataFrame:
        if not os.path.exists(TRIPLET_PATH):
            logger.error(f"Chua co {TRIPLET_PATH}. Chay scene_graph_extractor.py truoc.")
            sys.exit(1)

        with open(TRIPLET_PATH, encoding="utf-8") as f:
            triplets = json.load(f)
        logger.info(f"Map {len(triplets)} visual triplets -> 6-ngoi...")

        records = []
        for t in triplets:
            subj    = t.get("subject", "")
            pred    = t.get("predicate", "")
            obj     = t.get("object", "")
            img_id  = t.get("img_id", "")
            caption = t.get("caption", "")

            lh_lat, lh_lon = self._get_coords(subj)
            lt_lat, lt_lon = self._get_coords(obj)
            tau = self._get_timestamp(img_id, caption)

            records.append({
                "h":         f"img_{img_id}",
                "h_label":   subj,
                "r":         pred,
                "t":         obj,
                "t_label":   obj,
                "tau_start": tau,
                "tau_end":   tau,
                "l_h_lat":   lh_lat,
                "l_h_lon":   lh_lon,
                "l_t_lat":   lt_lat,
                "l_t_lon":   lt_lon,
                "image_file": t.get("image_file", ""),
                "vi_context": caption,
                "source":    t.get("source", "ViVQA"),
            })

        df_visual = pd.DataFrame(records)
        logger.info(f"Visual facts: {len(df_visual)}")

        # Merge với STKG Bước 1+2
        stkg = self.stkg.copy()
        for col in ["image_file", "vi_context"]:
            stkg[col] = ""
        stkg["source"] = "Wikidata-VN"

        combined = pd.concat([stkg, df_visual], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["h", "r", "t", "tau_start"], keep="first"
        )

        os.makedirs(VIVQA_DIR, exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"Saved -> {OUTPUT_PATH} ({len(combined)} facts)")

        # Thống kê
        has_img = (combined["image_file"] != "").sum()
        logger.info(f"Facts co anh : {has_img} ({has_img/len(combined)*100:.1f}%)")
        logger.info(f"Nguon Wikidata: {(combined['source']=='Wikidata-VN').sum()}")
        logger.info(f"Nguon ViVQA  : {combined['source'].str.startswith('ViVQA').sum()}")
        return combined


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    mapper = Mapper()
    df = mapper.map_triplets_to_6tuple()
    print(f"\nDone: {len(df)} facts -> {OUTPUT_PATH}")
