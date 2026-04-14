"""
Sinh Visual Triplets từ ảnh dùng CLIP zero-shot.
Model: openai/clip-vit-base-patch32

Visual triplet format:
    {"subject": str, "predicate": str, "object": str, "confidence": float}

Ví dụ:
    {"subject": "Q36014", "predicate": "depicts", "object": "historical leader", "confidence": 0.82}
"""

import json
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLIP_MODEL, VISUAL_DIR

logger = logging.getLogger(__name__)

TRIPLETS_PATH = os.path.join(VISUAL_DIR, "visual_triplets.json")

# ------------------------------------------------------------------
# Kho khái niệm thị giác cho STKG Việt Nam
# ------------------------------------------------------------------
VISUAL_CONCEPTS: dict[str, list[str]] = {
    # Nhân vật / người
    "historical leader":     ["a portrait of a Vietnamese leader or president",
                               "a photograph of a Vietnamese historical figure"],
    "military figure":       ["a Vietnamese general or military officer in uniform",
                               "a portrait of a Vietnamese soldier"],
    "scholar or intellectual":["a portrait of a Vietnamese scholar, writer, or poet",
                               "a Vietnamese intellectual or artist"],
    # Địa điểm tôn giáo
    "temple or pagoda":      ["a Vietnamese Buddhist pagoda or temple",
                               "a traditional Vietnamese religious building"],
    # Địa điểm lịch sử
    "citadel or fortress":   ["a Vietnamese citadel, fortress, or ancient city wall",
                               "Vietnamese ancient military architecture"],
    "museum":                ["a Vietnamese museum building or exhibition hall"],
    # Cảnh quan
    "landscape":             ["a Vietnamese landscape, mountain, or river scenery",
                               "aerial view of Vietnamese countryside or city"],
    # Sự kiện / hoạt động
    "battle or conflict":    ["a historical battle scene or military conflict",
                               "war memorial or battlefield in Vietnam"],
    "ceremony":              ["a Vietnamese traditional ceremony or festival",
                               "Vietnamese cultural performance or ritual"],
    # Hiện vật
    "artifact or document":  ["a historical Vietnamese document or artifact",
                               "ancient Vietnamese writing or seal"],
}

# Vị từ (predicate) theo loại quan hệ
PREDICATE_MAP = {
    "bornIn":     "depicts birthplace of",
    "diedIn":     "depicts place of death of",
    "locatedIn":  "depicts",
    "occurredAt": "depicts location of",
}


class SceneGraphGenerator:
    """Sinh visual triplets từ ảnh dùng CLIP zero-shot matching."""

    def __init__(self, model_name: str = CLIP_MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        # Pre-compute text embeddings cho tất cả concepts
        self._text_features, self._concept_names = self._encode_concepts()

    def _encode_concepts(self) -> tuple[torch.Tensor, list[str]]:
        """Encode tất cả text descriptions thành CLIP embeddings."""
        names, texts = [], []
        for concept, descriptions in VISUAL_CONCEPTS.items():
            for desc in descriptions:
                names.append(concept)
                texts.append(desc)

        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats, names

    def get_image_features(self, image_path: str) -> torch.Tensor | None:
        """Encode 1 ảnh thành CLIP image embedding (normalized)."""
        try:
            img    = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.squeeze(0)
        except Exception as e:
            logger.warning(f"CLIP encode error for {image_path}: {e}")
            return None

    def clip_text_similarity(self, image_path: str, text: str) -> float:
        """
        Tính cosine similarity giữa ảnh và một đoạn text bất kỳ.
        Dùng trong reliability_scorer để đánh giá sự khớp ảnh-nhãn.

        Returns:
            float trong [-1, 1], thường [0, 1] khi đã normalize
        """
        img_feat = self.get_image_features(image_path)
        if img_feat is None:
            return 0.0
        text_inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_feat = self.model.get_text_features(**text_inputs)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ text_feat.T).item()
        return float(np.clip((sim + 1) / 2, 0.0, 1.0))  # map [-1,1] → [0,1]

    def generate(
        self,
        image_path: str,
        subject_qid: str,
        subject_label: str,
        relation: str,
        top_k: int = 3,
        threshold: float = 0.20,
    ) -> list[dict]:
        """
        Sinh visual triplets cho 1 ảnh.

        Args:
            image_path: đường dẫn ảnh local
            subject_qid: QID của thực thể (head)
            subject_label: nhãn tiếng Việt
            relation: loại quan hệ (bornIn/diedIn/locatedIn/occurredAt)
            top_k: số triplet trả về
            threshold: ngưỡng tối thiểu của confidence

        Returns:
            list of {"subject", "predicate", "object", "confidence"}
        """
        img_feat = self.get_image_features(image_path)
        if img_feat is None:
            return []

        # Tính similarity với tất cả concept descriptions
        sims = (img_feat @ self._text_features.T).cpu().numpy()  # shape (N,)
        sims_mapped = (sims + 1) / 2  # → [0, 1]

        # Aggregate: lấy max similarity per concept
        concept_scores: dict[str, float] = {}
        for concept, score in zip(self._concept_names, sims_mapped):
            concept_scores[concept] = max(concept_scores.get(concept, 0.0), float(score))

        # Filter và sort
        predicate = PREDICATE_MAP.get(relation, "depicts")
        triplets = [
            {
                "subject":    subject_qid,
                "predicate":  predicate,
                "object":     concept,
                "confidence": round(score, 4),
            }
            for concept, score in sorted(concept_scores.items(), key=lambda x: -x[1])
            if score >= threshold
        ]
        return triplets[:top_k]

    def generate_batch(
        self,
        rows: list[dict],           # [{"qid", "label", "relation", "image_path"}, ...]
        show_progress: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Sinh triplets cho nhiều thực thể.

        Returns:
            dict {qid: [triplets]}
        """
        from tqdm import tqdm

        results: dict[str, list[dict]] = {}
        it = tqdm(rows, desc="Scene graph", unit="entity") if show_progress else rows

        for row in it:
            qid   = row["qid"]
            trips = self.generate(
                image_path    = row["image_path"],
                subject_qid   = qid,
                subject_label = row["label"],
                relation      = row.get("relation", "locatedIn"),
            )
            results[qid] = trips

        return results

    @staticmethod
    def save(triplets: dict[str, list[dict]], path: str = TRIPLETS_PATH):
        """Lưu toàn bộ visual triplets ra JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)
        total = sum(len(v) for v in triplets.values())
        logger.info(f"Saved {total} visual triplets ({len(triplets)} entities) -> {path}")

    @staticmethod
    def load(path: str = TRIPLETS_PATH) -> dict[str, list[dict]]:
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    from step3_visual.image_collector import ImageCollector

    collector = ImageCollector()
    images = collector.collect_batch([{"qid": "Q36014", "label": "Ho Chi Minh"}])

    if images:
        gen = SceneGraphGenerator()
        for qid, path in images.items():
            trips = gen.generate(path, qid, "Ho Chi Minh", "bornIn")
            print(f"\n{qid} visual triplets:")
            for t in trips:
                print(f"  {t}")
    else:
        print("Khong co anh de generate")
