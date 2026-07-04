"""
Download ViVQA annotations + subset ảnh MS-COCO cần thiết.

Annotations: CSV trên GitHub (train.csv, test.csv) — tải tự động.
Ảnh COCO   : 2 cách tùy lựa chọn:
  - Cách 1 (mặc định): Tải val2014.zip (~6GB) → giải nén lấy ảnh cần
  - Cách 2: Bỏ qua ảnh, dùng text-only mode

Cách dùng:
  python step3_final/download_vivqa.py            # tải ZIP (~6GB)
  python step3_final/download_vivqa.py --no-images # text-only
"""

import argparse
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TIMEOUT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIVQA_DIR   = os.path.join(BASE_DIR, "data", "vivqa")
IMAGE_DIR   = os.path.join(VIVQA_DIR, "images")
ZIP_PATH    = os.path.join(VIVQA_DIR, "val2014.zip")
COCO_ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"

# ViVQA CSV files trên GitHub
ANNOTATION_URLS = {
    "train": "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/train.csv",
    "test":  "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/test.csv",
}

session = requests.Session()
session.headers["User-Agent"] = "STKG-VN-Research/1.0"


def download_annotations() -> dict[str, pd.DataFrame]:
    """Download CSV annotations từ GitHub."""
    os.makedirs(VIVQA_DIR, exist_ok=True)
    dfs: dict[str, pd.DataFrame] = {}

    for split, url in ANNOTATION_URLS.items():
        save_path = os.path.join(VIVQA_DIR, f"{split}.csv")
        if os.path.exists(save_path):
            logger.info(f"  [CACHE] {split}.csv da co san")
        else:
            logger.info(f"  [DOWN] Downloading {split}.csv...")
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

        df = pd.read_csv(save_path, index_col=0)
        dfs[split] = df
        logger.info(f"  {split}: {len(df)} QA pairs, cols={list(df.columns)}")

    return dfs


def get_needed_image_ids(dfs: dict[str, pd.DataFrame]) -> list[int]:
    """Lấy danh sách img_id cần thiết từ annotations."""
    ids: set[int] = set()
    for df in dfs.values():
        if "img_id" in df.columns:
            ids.update(df["img_id"].dropna().astype(int).tolist())
    return sorted(ids)


def copy_from_local_folder(image_ids: list[int], val2014_dir: str) -> int:
    """
    Copy ảnh cần thiết từ thư mục val2014 đã có sẵn.
    """
    import shutil
    os.makedirs(IMAGE_DIR, exist_ok=True)
    needed_set = {f"COCO_val2014_{i:012d}.jpg" for i in image_ids}

    existing = {f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")}
    missing  = needed_set - existing
    if not missing:
        logger.info(f"  Tat ca {len(needed_set)} anh da co, bo qua copy")
        return len(needed_set)

    logger.info(f"  Copy {len(missing)} anh tu {val2014_dir}...")
    copied = 0
    not_found = 0
    for fname in tqdm(sorted(missing), desc="Copying images"):
        src = os.path.join(val2014_dir, fname)
        dst = os.path.join(IMAGE_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            not_found += 1

    logger.info(f"  Copy xong: {copied} anh (khong tim thay: {not_found})")
    return len(existing) + copied


def download_coco_zip_and_extract(image_ids: list[int]) -> int:
    """
    Tải val2014.zip (~6GB) rồi giải nén chỉ những ảnh cần thiết.
    Nếu ZIP đã có, bỏ qua tải.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    needed_set = {f"COCO_val2014_{i:012d}.jpg" for i in image_ids}

    # Kiểm tra ảnh đã có
    existing = {f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")}
    missing  = needed_set - existing
    if not missing:
        logger.info(f"  Tat ca {len(needed_set)} anh da co, bo qua download")
        return len(needed_set)

    # Tải ZIP nếu chưa có
    if not os.path.exists(ZIP_PATH):
        logger.info(f"  Tai val2014.zip (~6GB) tu {COCO_ZIP_URL}")
        logger.info("  Co the mat 20-60 phut tuy toc do mang...")
        resp = session.get(COCO_ZIP_URL, stream=True, timeout=3600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(ZIP_PATH, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="val2014.zip"
        ) as pbar:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
                pbar.update(len(chunk))
        logger.info(f"  ZIP da tai: {ZIP_PATH}")
    else:
        logger.info(f"  ZIP da co: {ZIP_PATH}")

    # Giải nén chỉ ảnh cần thiết
    logger.info(f"  Giai nen {len(missing)} anh can thiet...")
    extracted = 0
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        all_names = set(zf.namelist())
        for fname in tqdm(missing, desc="Extracting"):
            zpath = f"val2014/{fname}"
            if zpath in all_names:
                data = zf.read(zpath)
                with open(os.path.join(IMAGE_DIR, fname), "wb") as f:
                    f.write(data)
                extracted += 1

    logger.info(f"  Giai nen xong: {extracted}/{len(missing)} anh")
    return len(existing) + extracted


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-images", action="store_true",
                        help="Chi tai annotations, bo qua anh")
    parser.add_argument("--val2014-dir", type=str, default=None,
                        help="Duong dan thu muc val2014 da co san (bo qua download ZIP)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("DOWNLOAD ViVQA + COCO IMAGES")
    logger.info("=" * 50)

    # 1. Annotations
    logger.info("\nDownload annotations...")
    dfs = download_annotations()
    total_qa = sum(len(df) for df in dfs.values())
    logger.info(f"Tong: {total_qa} QA pairs")

    if args.no_images:
        logger.info("\n--no-images: Bo qua download anh")
        logger.info("Buoc tiep theo: python step3_final/scene_graph_extractor.py --text-only")
        sys.exit(0)

    # 2. Ảnh COCO
    image_ids = get_needed_image_ids(dfs)
    logger.info(f"\n{len(image_ids)} image IDs can xu ly")

    if args.val2014_dir:
        val_dir = os.path.abspath(args.val2014_dir)
        if not os.path.isdir(val_dir):
            logger.error(f"Thu muc khong ton tai: {val_dir}")
            sys.exit(1)
        logger.info(f"\nDung thu muc local: {val_dir}")
        n_ok = copy_from_local_folder(image_ids, val_dir)
    else:
        n_ok = download_coco_zip_and_extract(image_ids)

    logger.info("\n=== Ket qua ===")
    logger.info(f"Annotations : {total_qa} QA pairs")
    logger.info(f"Anh co san  : {n_ok} files in {IMAGE_DIR}")
    logger.info("\nBuoc tiep theo:")
    logger.info("  python step3_final/scene_graph_extractor.py")
    logger.info("  python step3_final/map_to_6tuple.py")
