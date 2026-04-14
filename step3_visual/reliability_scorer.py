"""
Tính Reliability Score cho từng fact trong knowledge graph.

Công thức:
    reliability = w_vis  * visual_score       (0.35)
                + w_clip * clip_score          (0.30)
                + w_sp   * spatial_score       (0.20)
                + w_temp * temporal_score      (0.15)

Giải thích:
    visual_score   : 1.0 nếu thực thể có ảnh, 0.0 nếu không
    clip_score     : CLIP cosine sim(ảnh, nhãn) ∈ [0,1], 0 nếu không có ảnh
    spatial_score  : 1.0 nếu có tọa độ hợp lệ trong bbox VN
    temporal_score : 1.0 nếu tau_start ∈ [800, 2024]
                     0.5 nếu tau_start null
                     0.3 nếu ngoài khoảng trên
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIETNAM_LAT, VIETNAM_LON

logger = logging.getLogger(__name__)

# Trọng số
W_VISUAL   = 0.35
W_CLIP     = 0.30
W_SPATIAL  = 0.20
W_TEMPORAL = 0.15

TAU_MIN = 800    # Lịch sử Việt Nam từ khoảng thế kỷ 9
TAU_MAX = 2024


class ReliabilityScorer:
    """Tính reliability score cho từng hàng trong dataframe 6-ngôi."""

    # ------------------------------------------------------------------
    # Các thành phần con
    # ------------------------------------------------------------------

    @staticmethod
    def _visual_score(qid: str, image_map: dict[str, str]) -> float:
        return 1.0 if qid in image_map else 0.0

    @staticmethod
    def _clip_score(qid: str, clip_map: dict[str, float]) -> float:
        return float(clip_map.get(qid, 0.0))

    @staticmethod
    def _spatial_score(row: pd.Series) -> float:
        lat = row.get("l_h_lat")
        lon = row.get("l_h_lon")
        if pd.isna(lat) or pd.isna(lon):
            return 0.0
        lat, lon = float(lat), float(lon)
        in_vn = (VIETNAM_LAT[0] <= lat <= VIETNAM_LAT[1] and
                 VIETNAM_LON[0] <= lon <= VIETNAM_LON[1])
        return 1.0 if in_vn else 0.5

    @staticmethod
    def _temporal_score(row: pd.Series) -> float:
        tau = row.get("tau_start")
        if pd.isna(tau):
            return 0.5
        tau = float(tau)
        if TAU_MIN <= tau <= TAU_MAX:
            return 1.0
        return 0.3

    def score_row(
        self,
        row: pd.Series,
        image_map: dict[str, str],
        clip_map:  dict[str, float],
    ) -> float:
        """Tính reliability score cho 1 hàng."""
        qid = str(row["h"])
        v = self._visual_score(qid, image_map)
        c = self._clip_score(qid, clip_map)
        s = self._spatial_score(row)
        t = self._temporal_score(row)
        score = W_VISUAL * v + W_CLIP * c + W_SPATIAL * s + W_TEMPORAL * t
        return round(float(score), 4)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        image_map: dict[str, str],
        clip_map:  dict[str, float],
    ) -> pd.DataFrame:
        """
        Thêm cột reliability_score và các component score vào dataframe.

        Args:
            df: dataframe 6-ngôi từ step2
            image_map: {qid: local_image_path}
            clip_map:  {qid: clip_similarity_score}

        Returns:
            df mới với thêm các cột:
              vis_score, clip_score, spatial_score, temporal_score, reliability_score
        """
        df = df.copy()
        qids = df["h"].astype(str)

        df["vis_score"]      = qids.map(lambda q: self._visual_score(q, image_map))
        df["clip_score"]     = qids.map(lambda q: self._clip_score(q, clip_map))
        df["spatial_score"]  = df.apply(self._spatial_score, axis=1)
        df["temporal_score"] = df.apply(self._temporal_score, axis=1)
        df["reliability_score"] = (
            W_VISUAL   * df["vis_score"]      +
            W_CLIP     * df["clip_score"]      +
            W_SPATIAL  * df["spatial_score"]   +
            W_TEMPORAL * df["temporal_score"]
        ).round(4)

        logger.info(
            f"Reliability scores: "
            f"mean={df['reliability_score'].mean():.3f}, "
            f"median={df['reliability_score'].median():.3f}, "
            f">=0.5: {(df['reliability_score']>=0.5).mean()*100:.1f}%"
        )
        return df

    def summary(self, df: pd.DataFrame) -> dict:
        """Tóm tắt thống kê reliability score theo relation."""
        if "reliability_score" not in df.columns:
            return {}
        stats = (
            df.groupby("r")["reliability_score"]
            .agg(["mean", "min", "max", "count"])
            .round(3)
            .to_dict("index")
        )
        return stats


if __name__ == "__main__":
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    # Demo với dữ liệu giả
    data = {
        "h": ["Q1", "Q2", "Q3"],
        "r": ["bornIn", "locatedIn", "occurredAt"],
        "l_h_lat": [21.0, None, 16.5],
        "l_h_lon": [105.8, None, 107.6],
        "tau_start": [1890, 1070, 1954],
    }
    df = pd.DataFrame(data)
    scorer = ReliabilityScorer()
    image_map = {"Q1": "path/to/img.jpg"}          # Q1 có ảnh
    clip_map  = {"Q1": 0.75}                        # CLIP score cho Q1
    result = scorer.score_dataframe(df, image_map, clip_map)
    print(result[["h", "r", "vis_score", "clip_score", "spatial_score", "temporal_score", "reliability_score"]])
