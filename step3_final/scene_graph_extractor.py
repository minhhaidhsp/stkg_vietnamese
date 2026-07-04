"""
Trích xuất Visual Triplets từ ảnh dùng BLIP + rule-based regex.

Input : data/vivqa/images/*.jpg
Output: data/vivqa/visual_triplets.json

Mode:
  --text-only : Không dùng BLIP, sinh triplets từ ViVQA Q&A text
  (default)   : Dùng BLIP captioning model (cần ảnh thực)
"""

import argparse
import json
import logging
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIVQA_DIR     = os.path.join(BASE_DIR, "data", "vivqa")
IMAGE_DIR     = os.path.join(VIVQA_DIR, "images")
TRIPLET_PATH  = os.path.join(VIVQA_DIR, "visual_triplets.json")

# ── Regex patterns trích xuất triplet từ BLIP caption ───────────────
PATTERNS = [
    # "a man riding a horse"
    r"(?:a |an |the )?(\w+)\s+(riding|holding|carrying|wearing|eating|standing near|sitting on|looking at|walking through|running in)\s+(?:a |an |the )?(.+?)(?:\s+in|\s+at|\s+on|$)",
    # "a woman in front of a temple"
    r"(?:a |an |the )?(\w+)\s+(in front of|next to|beside|inside|outside|on top of)\s+(?:a |an |the )?(.+?)(?:\.|$)",
    # "a dog on a green field"
    r"(?:a |an |the )?(\w+)\s+(on|in|at|near|by)\s+(?:a |an |the )?(.+?)(?:\.|$)",
]

# ── Question type → relation (cho text-only mode) ───────────────────
TYPE_TO_REL = {
    1: "hasObject",
    2: "hasColor",
    3: "hasCount",
    4: "isPresent",
    5: "hasActivity",
    6: "hasLocation",
}


def extract_triplets_from_caption(caption: str, image_file: str) -> list[dict]:
    """Trích xuất triplets từ BLIP caption bằng regex."""
    triplets = []
    caption_lower = caption.lower()

    for pat in PATTERNS:
        for m in re.finditer(pat, caption_lower):
            triplets.append({
                "subject":   m.group(1).strip(),
                "predicate": m.group(2).strip(),
                "object":    m.group(3).strip(),
                "caption":   caption,
                "image_file": image_file,
                "source":    "BLIP",
            })
        if triplets:
            break

    # Fallback: không match → triplet đơn giản từ đầu caption
    if not triplets:
        words = caption_lower.split()
        if len(words) >= 3:
            triplets.append({
                "subject":   words[0],
                "predicate": words[1],
                "object":    " ".join(words[2:5]),
                "caption":   caption,
                "image_file": image_file,
                "source":    "BLIP-fallback",
            })

    # Thêm img_id
    img_id = image_file.replace("COCO_val2014_", "").replace(".jpg", "").lstrip("0")
    for t in triplets:
        t["img_id"] = img_id

    return triplets[:5]


def extract_triplets_from_qa(row: pd.Series, split: str) -> list[dict]:
    """
    Text-only mode: tạo triplets từ ViVQA Q&A pairs.
    (h=img_id, r=relation_từ_type, t=answer)
    """
    img_id   = str(int(row["img_id"]))
    question = str(row.get("question", ""))
    answer   = str(row.get("answer", ""))
    qa_type  = int(row.get("type", 1))
    rel      = TYPE_TO_REL.get(qa_type, "hasAttribute")

    return [{
        "subject":   f"image_{img_id}",
        "predicate": rel,
        "object":    answer,
        "caption":   question,
        "img_id":    img_id,
        "image_file": f"COCO_val2014_{int(img_id):012d}.jpg",
        "source":    f"ViVQA-{split}",
    }]


def run_blip_mode(max_images: int = 200) -> list[dict]:
    """Chạy BLIP captioning trên ảnh thực."""
    from PIL import Image
    from transformers import pipeline

    logger.info("Dang tai BLIP captioning model...")
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        max_new_tokens=60,
    )
    logger.info("BLIP san sang")

    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")][:max_images]
    logger.info(f"Xu ly {len(files)} anh...")

    all_triplets: list[dict] = []
    for fname in tqdm(files, desc="BLIP captioning"):
        path = os.path.join(IMAGE_DIR, fname)
        try:
            img    = Image.open(path).convert("RGB")
            result = captioner(img)
            cap    = result[0]["generated_text"].strip()
            all_triplets.extend(extract_triplets_from_caption(cap, fname))
        except Exception as e:
            logger.warning(f"  {fname}: {e}")

    return all_triplets


def run_text_only_mode() -> list[dict]:
    """Text-only: sinh triplets từ ViVQA CSV annotations."""
    all_triplets: list[dict] = []

    for split in ["train", "test"]:
        csv_path = os.path.join(VIVQA_DIR, f"{split}.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"  {csv_path} chua co, chay download_vivqa.py truoc")
            continue

        df = pd.read_csv(csv_path, index_col=0)
        logger.info(f"  {split}: {len(df)} QA pairs")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ViVQA {split}"):
            all_triplets.extend(extract_triplets_from_qa(row, split))

    return all_triplets


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--text-only", action="store_true",
                        help="Dung ViVQA Q&A text, khong can BLIP/anh")
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(VIVQA_DIR, exist_ok=True)

    if args.text_only:
        logger.info("=== Text-only mode: ViVQA Q&A → Triplets ===")
        triplets = run_text_only_mode()
    else:
        logger.info("=== BLIP mode: anh → caption → Triplets ===")
        n_imgs = len([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]) if os.path.exists(IMAGE_DIR) else 0
        if n_imgs == 0:
            logger.error("Chua co anh. Chay download_vivqa.py truoc, hoac dung --text-only")
            sys.exit(1)
        triplets = run_blip_mode(args.max_images)

    with open(TRIPLET_PATH, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{len(triplets)} triplets -> {TRIPLET_PATH}")
    logger.info("Buoc tiep: python step3_final/map_to_6tuple.py")
