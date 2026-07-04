"""
Các module mô hình ViSTKG-QA (Bước 3), khớp Mục 1 docs/manuscript_methodology.md:
  - SpatioTemporalEmbedding: STE/RE + s(f) (CT 1, 2)
  - MultimodalEncoder: backbone đóng băng + LoRA + W_q/W_v (CT 3-5)
  - SubgraphRetriever: rel(f), top-K (CT 6)
  - CrossAttentionFusion: query_projection 512->896 + chú ý chéo (CT 7)
  - EntityRankingHead: xếp hạng thực thể, không sinh văn bản tự do (CT 8)
  - VisualReliabilityModule: module tin cậy bộ ba trực quan HỌC ĐƯỢC
  - MultiTaskLoss: L_total đa nhiệm (CT 9)
"""

from src.model.fusion import CrossAttentionFusion
from src.model.losses import MultiTaskLoss
from src.model.multimodal_encoder import MultimodalEncoder
from src.model.ranking_head import EntityRankingHead
from src.model.reliability_module import VisualReliabilityModule
from src.model.retriever import SubgraphRetriever
from src.model.spatiotemporal import SpatioTemporalEmbedding

__all__ = [
    "SpatioTemporalEmbedding",
    "MultimodalEncoder",
    "SubgraphRetriever",
    "CrossAttentionFusion",
    "EntityRankingHead",
    "VisualReliabilityModule",
    "MultiTaskLoss",
]
