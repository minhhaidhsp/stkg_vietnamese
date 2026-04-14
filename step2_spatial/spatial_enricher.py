"""
Bước 2: Làm giàu tọa độ không gian cho combined_raw.csv.

Chiến lược theo loại quan hệ:
  1. bornIn / diedIn  → l_h = l_t  (người ở nơi sinh/mất)
  2. locatedIn        → l_h geocode từ h_label (tên di tích/công trình)
  3. occurredAt       → l_h geocode từ h_label, fallback l_t (địa điểm sự kiện)

Output: data/spatial/enriched.csv
"""

import csv
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, SPATIAL_DIR, VIETNAM_LAT, VIETNAM_LON

from step2_spatial.osm_resolver import OSMResolver
from step2_spatial.geonames_resolver import GeoNamesResolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENRICHED_CSV = os.path.join(SPATIAL_DIR, "enriched.csv")
GEO_CACHE_CSV = os.path.join(SPATIAL_DIR, "geocode_cache.csv")


class SpatialEnricher:
    """Làm giàu tọa độ không gian cho dữ liệu 6-ngôi."""

    def __init__(self, geonames_username: str = ""):
        self.osm = OSMResolver()
        self.geonames = GeoNamesResolver(geonames_username)
        self._cache: dict[str, tuple[float, float] | None] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Cache geocoding (tránh gọi API lặp lại giữa các lần chạy)
    # ------------------------------------------------------------------

    def _load_cache(self):
        if not os.path.exists(GEO_CACHE_CSV):
            return
        with open(GEO_CACHE_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lat = row["lat"]
                lon = row["lon"]
                self._cache[row["name"]] = (float(lat), float(lon)) if lat else None
        logger.info(f"Loaded geocoding cache: {len(self._cache)} entries")

    def _save_cache(self):
        os.makedirs(SPATIAL_DIR, exist_ok=True)
        with open(GEO_CACHE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "lat", "lon"])
            writer.writeheader()
            for name, coords in self._cache.items():
                writer.writerow({
                    "name": name,
                    "lat": coords[0] if coords else "",
                    "lon": coords[1] if coords else "",
                })

    # ------------------------------------------------------------------
    # Geocoding với cache + fallback chain OSM → GeoNames
    # ------------------------------------------------------------------

    def _geocode(self, place_name: str) -> tuple[float, float] | None:
        if not place_name or (isinstance(place_name, float)):
            return None
        place_name = str(place_name).strip()
        if not place_name:
            return None

        if place_name in self._cache:
            return self._cache[place_name]

        coords = self.osm.resolve(place_name)
        if not coords and self.geonames.available:
            coords = self.geonames.resolve(place_name)

        self._cache[place_name] = coords
        return coords

    def _in_vietnam(self, lat: float, lon: float) -> bool:
        return (VIETNAM_LAT[0] <= lat <= VIETNAM_LAT[1] and
                VIETNAM_LON[0] <= lon <= VIETNAM_LON[1])

    # ------------------------------------------------------------------
    # Pipeline làm giàu tọa độ
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        total = len(df)

        # --- Chiến lược 1: bornIn/diedIn — người ở nơi sinh/mất ---
        mask_person = df["r"].isin(["bornIn", "diedIn"]) & df["l_h_lat"].isna()
        df.loc[mask_person, "l_h_lat"] = df.loc[mask_person, "l_t_lat"]
        df.loc[mask_person, "l_h_lon"] = df.loc[mask_person, "l_t_lon"]
        filled_s1 = mask_person.sum()
        logger.info(f"Strategy 1 (copy l_t->l_h for persons): {filled_s1} rows")

        # --- Chiến lược 2: geocode l_h còn thiếu từ h_label ---
        need_h = df["l_h_lat"].isna()
        labels_h = df.loc[need_h, "h_label"].unique().tolist()
        logger.info(f"Strategy 2 (geocode h_label): {need_h.sum()} rows, {len(labels_h)} unique names")

        geo_h: dict[str, tuple[float, float] | None] = {}
        for i, name in enumerate(labels_h, 1):
            geo_h[name] = self._geocode(name)
            if i % 10 == 0:
                hits = sum(v is not None for v in geo_h.values())
                logger.info(f"  h geocoding: {i}/{len(labels_h)} ({hits} hits)")

        filled_s2 = 0
        for idx in df[need_h].index:
            coords = geo_h.get(df.at[idx, "h_label"])
            if coords:
                df.at[idx, "l_h_lat"] = coords[0]
                df.at[idx, "l_h_lon"] = coords[1]
                filled_s2 += 1
        logger.info(f"Strategy 2 filled: {filled_s2}/{need_h.sum()} rows")

        # --- Chiến lược 3: geocode l_t còn thiếu từ t_label ---
        need_t = df["l_t_lat"].isna() & df["t_label"].notna()
        labels_t = df.loc[need_t, "t_label"].unique().tolist()
        logger.info(f"Strategy 3 (geocode t_label): {need_t.sum()} rows, {len(labels_t)} unique names")

        geo_t: dict[str, tuple[float, float] | None] = {}
        for name in labels_t:
            geo_t[name] = self._geocode(name)

        filled_s3 = 0
        for idx in df[need_t].index:
            coords = geo_t.get(df.at[idx, "t_label"])
            if coords:
                df.at[idx, "l_t_lat"] = coords[0]
                df.at[idx, "l_t_lon"] = coords[1]
                filled_s3 += 1
        logger.info(f"Strategy 3 filled: {filled_s3}/{need_t.sum()} rows")

        # --- Chiến lược 4: occurredAt còn thiếu l_h → dùng l_t làm proxy ---
        mask_event = (df["r"] == "occurredAt") & df["l_h_lat"].isna() & df["l_t_lat"].notna()
        df.loc[mask_event, "l_h_lat"] = df.loc[mask_event, "l_t_lat"]
        df.loc[mask_event, "l_h_lon"] = df.loc[mask_event, "l_t_lon"]
        filled_s4 = mask_event.sum()
        logger.info(f"Strategy 4 (copy l_t->l_h for events): {filled_s4} rows")

        self._save_cache()
        logger.info(
            f"Enrichment xong: {total} rows. "
            f"l_h_lat coverage: {df['l_h_lat'].notna().mean()*100:.1f}% | "
            f"l_t_lat coverage: {df['l_t_lat'].notna().mean()*100:.1f}%"
        )
        return df

    def validate_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Đánh dấu tọa độ nằm ngoài bbox Việt Nam."""
        def check(row, col_lat, col_lon):
            lat, lon = row.get(col_lat), row.get(col_lon)
            if pd.isna(lat) or pd.isna(lon):
                return True  # null = chưa có, không phải sai
            return self._in_vietnam(float(lat), float(lon))

        df["h_in_vn"] = df.apply(lambda r: check(r, "l_h_lat", "l_h_lon"), axis=1)
        df["t_in_vn"] = df.apply(lambda r: check(r, "l_t_lat", "l_t_lon"), axis=1)
        return df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    input_path = os.path.join(RAW_DIR, "combined_raw.csv")
    df_raw = pd.read_csv(input_path)
    print(f"Đọc {len(df_raw)} facts từ {input_path}")

    enricher = SpatialEnricher()
    df_enriched = enricher.enrich(df_raw)
    df_enriched = enricher.validate_coords(df_enriched)

    os.makedirs(SPATIAL_DIR, exist_ok=True)
    df_enriched.to_csv(ENRICHED_CSV, index=False, encoding="utf-8")
    print(f"Đã lưu -> {ENRICHED_CSV}")

    # Báo cáo nhanh
    print("\n--- Kết quả ---")
    for col in ["l_h_lat", "l_h_lon", "l_t_lat", "l_t_lon"]:
        pct = df_enriched[col].notna().mean() * 100
        raw_pct = df_raw[col].notna().mean() * 100
        delta = pct - raw_pct
        arrow = "+" if delta > 0 else ""
        print(f"  {col}: {raw_pct:.1f}% -> {pct:.1f}%  ({arrow}{delta:.1f}%)")
    out_of_vn_h = (~df_enriched["h_in_vn"]).sum()
    out_of_vn_t = (~df_enriched["t_in_vn"]).sum()
    print(f"\n  Toa do ngoai Viet Nam (h): {out_of_vn_h}")
    print(f"  Toa do ngoai Viet Nam (t): {out_of_vn_t}")
