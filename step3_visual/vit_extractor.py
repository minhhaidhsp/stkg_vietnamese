"""
Trích xuất đặc trưng ảnh dùng Vision Transformer (ViT).
Model: google/vit-base-patch16-224  →  vector 768 chiều (CLS token).
Lưu features vào data/visual/vit_features.npz (key = QID).
"""

import logging
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIT_MODEL, VISUAL_DIR

logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(VISUAL_DIR, "vit_features.npz")


class ViTExtractor:
    """Trích xuất CLS embedding từ ảnh bằng ViT."""

    def __init__(self, model_name: str = VIT_MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading ViT model: {model_name} on {self.device}")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, image_path: str) -> np.ndarray | None:
        """
        Trích xuất CLS embedding từ 1 ảnh.

        Returns:
            ndarray shape (768,) hoặc None nếu lỗi
        """
        try:
            img    = Image.open(image_path).convert("RGB")
            inputs = self.extractor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return cls_vec
        except Exception as e:
            logger.warning(f"ViT extract failed for {image_path}: {e}")
            return None

    def extract_batch(
        self,
        image_map: dict[str, str],   # {qid: local_image_path}
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Trích xuất features cho nhiều ảnh.

        Returns:
            dict {qid: ndarray[768]}
        """
        from tqdm import tqdm

        results: dict[str, np.ndarray] = {}
        items = list(image_map.items())
        it = tqdm(items, desc="ViT extract", unit="img") if show_progress else items

        for qid, path in it:
            vec = self.extract(path)
            if vec is not None:
                results[qid] = vec

        logger.info(f"ViT: extracted {len(results)}/{len(image_map)} features")
        return results

    # ------------------------------------------------------------------
    # Persist / load
    # ------------------------------------------------------------------

    @staticmethod
    def save(features: dict[str, np.ndarray], path: str = FEATURES_PATH):
        """Lưu dict {qid: ndarray} vào file .npz."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **features)
        logger.info(f"Saved {len(features)} ViT features -> {path}")

    @staticmethod
    def load(path: str = FEATURES_PATH) -> dict[str, np.ndarray]:
        """Tải features từ .npz."""
        if not os.path.exists(path):
            return {}
        data = np.load(path, allow_pickle=False)
        features = {k: data[k] for k in data.files}
        logger.info(f"Loaded {len(features)} ViT features from {path}")
        return features

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity giữa 2 vector."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    from step3_visual.image_collector import ImageCollector

    collector = ImageCollector()
    images = collector.collect_batch([
        {"qid": "Q36014", "label": "Ho Chi Minh"},
        {"qid": "Q1858",  "label": "Ha Noi"},
    ])

    if images:
        vit = ViTExtractor()
        features = vit.extract_batch(images)
        ViTExtractor.save(features)
        for qid, vec in features.items():
            print(f"{qid}: shape={vec.shape}, norm={np.linalg.norm(vec):.3f}")
    else:
        print("Khong co anh nao de extract")
