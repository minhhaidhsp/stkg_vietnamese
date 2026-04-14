"""
Bước 3: Pipeline làm giàu thị giác.

Luồng xử lý:
  enriched.csv (step2)
      ↓  image_collector    → data/visual/images/{qid}.jpg
      ↓  vit_extractor      → data/visual/vit_features.npz
      ↓  scene_graph_gen    → data/visual/visual_triplets.json
      ↓  reliability_scorer → data/visual/visual_enriched.csv
"""

import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPATIAL_DIR, VISUAL_DIR

from step3_visual.image_collector    import ImageCollector
from step3_visual.vit_extractor      import ViTExtractor
from step3_visual.scene_graph_generator import SceneGraphGenerator
from step3_visual.reliability_scorer import ReliabilityScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV   = os.path.join(SPATIAL_DIR, "enriched.csv")
OUTPUT_CSV  = os.path.join(VISUAL_DIR,  "visual_enriched.csv")


class VisualEnricher:
    """Orchestrator cho toàn bộ pipeline bước 3."""

    def __init__(self):
        self.collector = ImageCollector()
        self.scorer    = ReliabilityScorer()
        self._vit: ViTExtractor | None = None
        self._sgg: SceneGraphGenerator | None = None

    def _get_vit(self) -> ViTExtractor:
        if self._vit is None:
            self._vit = ViTExtractor()
        return self._vit

    def _get_sgg(self) -> SceneGraphGenerator:
        if self._sgg is None:
            self._sgg = SceneGraphGenerator()
        return self._sgg

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        input_csv:  str = INPUT_CSV,
        output_csv: str = OUTPUT_CSV,
        extract_vit: bool = True,
        gen_scene_graph: bool = True,
    ) -> pd.DataFrame:
        """
        Chạy toàn bộ pipeline.

        Args:
            extract_vit: Có dùng ViT để trích xuất features không
            gen_scene_graph: Có dùng CLIP để sinh visual triplets không

        Returns:
            DataFrame đã làm giàu (cũng được lưu ra file)
        """
        os.makedirs(VISUAL_DIR, exist_ok=True)

        # ── 1. Load dữ liệu ──────────────────────────────────────────
        logger.info(f"Load input: {input_csv}")
        df = pd.read_csv(input_csv)
        logger.info(f"  {len(df)} facts, {df['h'].nunique()} unique entities")

        # ── 2. Thu thập ảnh ──────────────────────────────────────────
        logger.info("Step 3.1: Thu thap anh...")
        unique_entities = (
            df[["h", "h_label"]]
            .drop_duplicates("h")
            .rename(columns={"h": "qid", "h_label": "label"})
            .to_dict("records")
        )
        image_map = self.collector.collect_batch(unique_entities)
        logger.info(f"  {len(image_map)}/{len(unique_entities)} entities co anh")

        # ── 3. ViT features ──────────────────────────────────────────
        vit_features: dict = {}
        if extract_vit and image_map:
            logger.info("Step 3.2: Trich xuat ViT features...")
            # Tải cache nếu có
            vit_features = ViTExtractor.load()
            missing_vit  = {q: p for q, p in image_map.items() if q not in vit_features}
            if missing_vit:
                new_feats = self._get_vit().extract_batch(missing_vit)
                vit_features.update(new_feats)
                ViTExtractor.save(vit_features)
            logger.info(f"  {len(vit_features)} ViT feature vectors")

        # ── 4. CLIP score (image-label similarity) ────────────────────
        clip_map: dict[str, float] = {}
        if gen_scene_graph and image_map:
            logger.info("Step 3.3: Tinh CLIP score va sinh scene graphs...")
            # Tải cache triplets nếu có
            triplets_all = SceneGraphGenerator.load()

            # Chỉ xử lý entity chưa có triplets
            label_map = dict(zip(df["h"].astype(str), df["h_label"]))
            rel_map   = dict(zip(df["h"].astype(str), df["r"]))
            sgg = self._get_sgg()

            rows_to_process = [
                {
                    "qid":        qid,
                    "label":      label_map.get(qid, ""),
                    "relation":   rel_map.get(qid, "locatedIn"),
                    "image_path": path,
                }
                for qid, path in image_map.items()
                if qid not in triplets_all
            ]

            if rows_to_process:
                new_trips = sgg.generate_batch(rows_to_process)
                triplets_all.update(new_trips)
                SceneGraphGenerator.save(triplets_all)

            # Tính CLIP score (image ↔ entity label)
            from tqdm import tqdm
            for qid, path in tqdm(image_map.items(), desc="CLIP score", unit="img"):
                label = label_map.get(qid, "")
                if label:
                    clip_map[qid] = sgg.clip_text_similarity(path, label)

            logger.info(f"  CLIP scores computed: {len(clip_map)}")
        else:
            triplets_all = SceneGraphGenerator.load()

        # ── 5. Reliability scoring ────────────────────────────────────
        logger.info("Step 3.4: Tinh Reliability Score...")
        df = self.scorer.score_dataframe(df, image_map, clip_map)

        # ── 6. Thêm cột visual metadata ───────────────────────────────
        df["has_image"]      = df["h"].astype(str).map(lambda q: q in image_map)
        df["image_path"]     = df["h"].astype(str).map(lambda q: image_map.get(q, ""))
        df["visual_triplets"] = df["h"].astype(str).map(
            lambda q: json.dumps(triplets_all.get(q, []), ensure_ascii=False)
        )
        df["vit_feature_saved"] = df["h"].astype(str).map(lambda q: q in vit_features)

        # ── 7. Lưu kết quả ───────────────────────────────────────────
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info(f"Saved -> {output_csv}")

        return df


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")

    enricher = VisualEnricher()
    df = enricher.run()

    print("\n--- Ket qua Buoc 3 ---")
    print(f"Total facts: {len(df)}")
    print(f"Has image:   {df['has_image'].sum()} ({df['has_image'].mean()*100:.1f}%)")
    print(f"Reliability: mean={df['reliability_score'].mean():.3f}, "
          f">=0.5: {(df['reliability_score']>=0.5).mean()*100:.1f}%")
    print(f"\nTop 5 facts (reliability cao nhat):")
    cols = ["h_label", "r", "t_label", "tau_start", "reliability_score"]
    print(df.nlargest(5, "reliability_score")[cols].to_string(index=False))
