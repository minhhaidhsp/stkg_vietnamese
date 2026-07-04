"""
Pipeline chính Bước 3: Kế thừa + Mở rộng + Việt hóa.

Luồng:
  data/spatial/enriched.csv  (652 facts từ B1+B2)
      ↓ wikidata_expander     → thêm quan hệ mới (educatedAt, memberOf, ...)
      ↓ icews_loader          → thêm sự kiện ICEWS (nếu có file)
      ↓ vietnamese_labeler    → đảm bảo nhãn tiếng Việt
      ↓ merge + dedup         → data/step3/merged.csv
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPATIAL_DIR, STEP3_DIR, INHERITED_DIR

from step3_inherit.wikidata_expander  import expand_entities
from step3_inherit.icews_loader       import load_icews_files
from step3_inherit.vietnamese_labeler import fill_vietnamese_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_CSV   = os.path.join(SPATIAL_DIR, "enriched.csv")
OUTPUT_CSV = os.path.join(STEP3_DIR, "merged.csv")

REQUIRED_COLS = ["h", "h_label", "r", "t", "t_label",
                 "tau_start", "tau_end",
                 "l_h_lat", "l_h_lon", "l_t_lat", "l_t_lon"]


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None
    return df[REQUIRED_COLS + [c for c in df.columns if c not in REQUIRED_COLS]]


def run() -> pd.DataFrame:
    os.makedirs(STEP3_DIR, exist_ok=True)

    # ── 1. Load base facts ────────────────────────────────────────────
    logger.info(f"Load base: {BASE_CSV}")
    base_df = pd.read_csv(BASE_CSV)
    if "tau_end" not in base_df.columns:
        base_df["tau_end"] = base_df.get("tau_start")
    base_df["source"] = "wikidata_base"
    logger.info(f"  Base: {len(base_df)} facts, {base_df['h'].nunique()} entities")

    # ── 2. Mở rộng Wikidata ──────────────────────────────────────────
    logger.info("Wikidata expand...")
    base_qids = base_df["h"].dropna().unique().tolist()
    # Lọc chỉ QID thực (Q...)
    base_qids = [q for q in base_qids if str(q).startswith("Q")]
    expand_df = expand_entities(base_qids)
    if not expand_df.empty:
        expand_df["source"] = "wikidata_expand"
        logger.info(f"  Wikidata expand: +{len(expand_df)} rows")
    else:
        logger.warning("  Wikidata expand: khong co du lieu")

    # ── 3. ICEWS ──────────────────────────────────────────────────────
    logger.info("ICEWS load...")
    icews_df = load_icews_files(INHERITED_DIR)
    if icews_df.empty:
        logger.info("  ICEWS: khong co file (bo qua)")
    else:
        logger.info(f"  ICEWS: +{len(icews_df)} rows")

    # ── 4. Merge ──────────────────────────────────────────────────────
    parts = [base_df]
    if not expand_df.empty:
        parts.append(_ensure_cols(expand_df))
    if not icews_df.empty:
        parts.append(_ensure_cols(icews_df))

    merged = pd.concat(parts, ignore_index=True)
    logger.info(f"Truoc dedup: {len(merged)} rows")

    # Dedup trên (h, r, t, tau_start)
    merged["_key"] = (
        merged["h"].astype(str) + "|" +
        merged["r"].astype(str) + "|" +
        merged["t"].astype(str) + "|" +
        merged["tau_start"].astype(str)
    )
    merged = merged.drop_duplicates("_key").drop(columns="_key")
    logger.info(f"Sau dedup: {len(merged)} rows")

    # ── 5. Việt hóa nhãn ─────────────────────────────────────────────
    logger.info("Viet hoa nhan...")
    label_cache: dict[str, str] = {}
    merged = fill_vietnamese_labels(merged, label_cache)

    # ── 6. Lưu ───────────────────────────────────────────────────────
    merged = _ensure_cols(merged)
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved -> {OUTPUT_CSV} ({len(merged)} facts)")

    # ── 7. Tóm tắt ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info(f"Tong facts    : {len(merged)}")
    logger.info(f"Unique h      : {merged['h'].nunique()}")
    logger.info(f"Unique r      : {merged['r'].nunique()}")
    logger.info(f"Unique t      : {merged['t'].nunique()}")
    logger.info(f"Co l_h_lat    : {merged['l_h_lat'].notna().sum()} ({merged['l_h_lat'].notna().mean()*100:.1f}%)")
    logger.info(f"Co tau_start  : {merged['tau_start'].notna().sum()} ({merged['tau_start'].notna().mean()*100:.1f}%)")
    sources = merged.get("source", pd.Series()).value_counts()
    logger.info(f"Theo nguon:\n{sources.to_string()}")

    return merged


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    df = run()
    print(f"\nDone: {len(df)} facts -> {OUTPUT_CSV}")
