"""
C1 — Smoke test: forward + backward thật trên 50 mẫu thật (ViSTQAD), 1
epoch, batch=1 (tránh phức tạp padding độ dài câu hỏi khác nhau — hợp lý
cho smoke test, KHÔNG phải batch=32 của huấn luyện đầy đủ CT9).

MỤC TIÊU: xác nhận wiring thật (MultimodalEncoder.from_pretrained + GQA
cross-attention + ranking head + multi-task loss) không NaN/lỗi shape, và
loss có xu hướng giảm — KHÔNG phải huấn luyện có ý nghĩa (50 mẫu, 1 epoch).

QUAN TRỌNG — giới hạn môi trường: script này chạy được cả CPU (local, chậm)
lẫn GPU (Colab, nhanh). Bản ghi trong docs/DECISIONS.md đã nêu rõ các lựa
chọn: Enc_LM(Q)=embed_tokens, K/V/O lấy từ layer 0, R_final broadcast-add
toàn bộ token X, TẤT CẢ mẫu dùng null_image_embedding (chưa tải ảnh Wikimedia
thật về máy nên chưa nối ảnh thật vào pipeline — L1 đang chạy nền để tải,
sẽ nối ảnh thật ở vòng huấn luyện đầy đủ sau khi có tỷ lệ phủ ảnh cuối).
"""

import logging
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import load_config
from src.model.fusion import CrossAttentionFusion
from src.model.losses import MultiTaskLoss
from src.model.multimodal_encoder import MultimodalEncoder
from src.model.ranking_head import EntityRankingHead
from src.model.retriever import SubgraphRetriever
from src.model.spatiotemporal import SpatioTemporalEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VN_DEFAULT = (14.0583, 108.2772)


def load_50_samples(cfg, seed: int = 42) -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "vistqad", "train_manifest.csv")
    df = pd.read_csv(path)
    return df.sample(n=50, random_state=seed).reset_index(drop=True)


def build_entity_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Vũ trụ thực thể cho ranking (smoke test): entity + toạ độ/tau đại
    diện lấy từ CHÍNH fact mà entity đó là đáp án — đơn giản hoá hợp lý
    cho quy mô 50 mẫu (không phải toàn bộ 7.218 thực thể thật của Bước 4)."""
    entities = {}
    for _, row in df.iterrows():
        entities[row["answer_entity"]] = {
            "lat": row["l_t_lat"] if pd.notna(row["l_t_lat"]) else VN_DEFAULT[0],
            "lon": row["l_t_lon"] if pd.notna(row["l_t_lon"]) else VN_DEFAULT[1],
            "tau": row["tau_start"] if pd.notna(row["tau_start"]) else 2000.0,
        }
    return pd.DataFrame([{"entity": k, **v} for k, v in entities.items()])


def main():
    cfg = load_config()
    torch.manual_seed(cfg["data"]["split"]["seed"])

    logger.info("Đọc 50 mẫu thật từ data/vistqad/train_manifest.csv")
    samples = load_50_samples(cfg)
    logger.info(f"Tỷ lệ mẫu có ảnh trong 50 mẫu smoke test: {samples['has_image'].mean()*100:.1f}%")

    entity_universe = build_entity_universe(samples)
    entity_to_idx = {e: i for i, e in enumerate(entity_universe["entity"])}
    logger.info(f"Vũ trụ thực thể (smoke test): {len(entity_universe)} thực thể duy nhất")

    logger.info("Tải Vintern-1B-v2 thật (MultimodalEncoder.from_pretrained)...")
    t0 = time.time()
    encoder = MultimodalEncoder.from_pretrained(cfg)
    logger.info(f"Tải xong trong {time.time()-t0:.1f}s")

    ste = SpatioTemporalEmbedding(cfg)
    retriever = SubgraphRetriever(cfg)
    fusion = CrossAttentionFusion(cfg)
    ranking_head = EntityRankingHead(cfg)
    loss_fn = MultiTaskLoss(cfg)

    layer0_attn = encoder.text_backbone.base_model.model.model.layers[0].self_attn

    trainable = (
        [p for p in encoder.parameters() if p.requires_grad]
        + list(fusion.parameters())
        + list(ranking_head.parameters())
        + list(retriever.parameters())
    )
    opt_cfg = cfg["optimizer"]
    optimizer_adapter = torch.optim.AdamW(trainable, lr=opt_cfg["lr_adapter"], weight_decay=opt_cfg["weight_decay"])
    optimizer_embedding = torch.optim.AdamW(ste.parameters(), lr=opt_cfg["lr_embedding"], weight_decay=opt_cfg["weight_decay"])

    bbox = cfg["spatiotemporal_grid"]["spatial_bbox"]
    relations = cfg["data"]["relations"]

    entity_lat = torch.tensor(entity_universe["lat"].values, dtype=torch.float32)
    entity_lon = torch.tensor(entity_universe["lon"].values, dtype=torch.float32)
    entity_tau = torch.tensor(entity_universe["tau"].values, dtype=torch.float32)
    # ste_entities_full PHẢI tính lại MỖI step (không cache tensor kết quả) —
    # STE.encode_entity có tham số huấn luyện (E_x/E_y/E_t), tái sử dụng cùng
    # 1 tensor kết quả qua nhiều lần backward() gây "backward qua graph lần
    # 2" (graph của lần tính trước đã bị giải phóng) — xác nhận thật qua
    # torch.autograd.set_detect_anomaly khi debug smoke test.

    losses = []
    n_nan = 0
    rng = torch.Generator().manual_seed(123)

    for step, row in samples.iterrows():
        optimizer_adapter.zero_grad()
        optimizer_embedding.zero_grad()

        # --- CT(3): E_Q = Enc_LM(Q) = embed_tokens ---
        input_ids = encoder.tokenizer(row["question"], return_tensors="pt").input_ids
        e_q = encoder.embed_tokens(input_ids)                    # (1, n_text, 896)

        # --- CT(4): E_I (mẫu này KHÔNG có ảnh thật nối vào — xem docstring) ---
        e_i = encoder.null_image_features(1)                        # (1, n_null_patches, vision_hidden)

        # --- CT(5): X = [E_Q W_q ; E_I W_v] ---
        x = encoder.fuse(e_q, e_i)                                   # (1, n_tok, 896)

        # --- CT(2) + CT(6): STE/RE cho candidate facts (chính 50 mẫu làm ứng viên) + retrieval ---
        cand = samples.sample(n=min(10, len(samples)), random_state=int(step))
        lat_h = torch.tensor(cand["l_h_lat"].fillna(VN_DEFAULT[0]).values, dtype=torch.float32)
        lon_h = torch.tensor(cand["l_h_lon"].fillna(VN_DEFAULT[1]).values, dtype=torch.float32)
        lat_t = torch.tensor(cand["l_t_lat"].fillna(VN_DEFAULT[0]).values, dtype=torch.float32)
        lon_t = torch.tensor(cand["l_t_lon"].fillna(VN_DEFAULT[1]).values, dtype=torch.float32)
        tau = torch.tensor(cand["tau_start"].fillna(2000.0).values, dtype=torch.float32)
        rel_ids = ste.relation_ids_from_names(cand["relation"].tolist())

        ste_h = ste.encode_entity(lat_h, lon_h, tau)
        re_r = ste.encode_relation(rel_ids)
        ste_t = ste.encode_entity(lat_t, lon_t, tau)
        g_f = retriever.event_embedding(ste_h, re_r, ste_t)          # (n_cand, 512)
        s_f = ste.score(lat_h, lon_h, tau, rel_ids, lat_t, lon_t)     # (n_cand,)

        x_bar_raw = x.mean(dim=1).squeeze(0)                          # (896,) — X CHƯA chiếu
        rel_scores = retriever.relevance(x_bar_raw, g_f, s_f)          # (n_cand,)
        top_k = min(retriever.top_k, len(cand))
        top_idx = retriever.top_k_indices(rel_scores, k=top_k)

        # --- CT(7): cross-attention hướng vào, dùng K/V/O thật của layer 0 ---
        k_x_raw = layer0_attn.k_proj(x)                                # (1, n_tok, 128)
        v_x_raw = layer0_attn.v_proj(x)
        z_m = g_f[top_idx].unsqueeze(0)                                 # (1, top_k, 512) — q_m^Z nguồn
        r_m = fusion(z_m, k_x_raw, v_x_raw, layer0_attn.o_proj)          # (1, top_k, 896)

        w_m = torch.softmax(rel_scores[top_idx], dim=0)                 # trọng số kết hợp theo rel(f)
        r_final = (r_m.squeeze(0) * w_m.unsqueeze(-1)).sum(dim=0)        # (896,)

        x_final = x + r_final.view(1, 1, -1)                             # cộng dư broadcast toàn bộ token
        ste_entities_full = ste.encode_entity(entity_lat, entity_lon, entity_tau)  # tính LẠI mỗi step

        # --- "Các lớp còn lại của LLM": chạy lại toàn bộ Qwen2Model trên X đã cộng dư ---
        # use_cache=False: tránh Qwen2Model tạo past_key_values (legacy tuple cache,
        # xác nhận thật gây RuntimeError "backward qua graph lần 2" khi kết hợp với
        # peft — không cần cache vì đây là 1 forward pass đơn, không sinh chuỗi).
        lm_out = encoder.text_backbone.base_model.model.model(inputs_embeds=x_final, use_cache=False)
        final_hidden = lm_out.last_hidden_state.mean(dim=1)              # (1, 896) — pool trung bình token

        # --- CT(8): entity ranking ---
        p = ranking_head(final_hidden, ste_entities_full)                 # (1, num_entities)
        qa_target = torch.tensor([entity_to_idx[row["answer_entity"]]])

        # --- CT(9): loss đa nhiệm (mẫu này không có ảnh -> L_VG=0, guard đã test ở Bước 3) ---
        s_pos = s_f[:1]
        s_neg = s_f[1:2] if len(s_f) > 1 else s_f[:1] + 1.0
        vg_pred = torch.zeros(1)
        vg_target = torch.zeros(1)
        vg_mask = torch.zeros(1, dtype=torch.bool)

        out = loss_fn(p, qa_target, s_pos, s_neg, vg_pred, vg_target, vg_mask=vg_mask)
        loss = out["loss_total"]

        if not torch.isfinite(loss):
            n_nan += 1
            logger.warning(f"Step {step}: loss KHÔNG hữu hạn ({loss.item()}) — {row['question']}")
            continue

        loss.backward()
        optimizer_adapter.step()
        optimizer_embedding.step()

        losses.append(loss.item())
        if step % 10 == 0 or step == len(samples) - 1:
            logger.info(f"Step {step:2d}/{len(samples)}: loss_total={loss.item():.4f} "
                        f"(qa={out['loss_qa'].item():.4f} stkg={out['loss_stkg'].item():.4f} vg={out['loss_vg'].item():.4f})")

    logger.info(f"Hoàn tất smoke test: {len(losses)}/{len(samples)} step thành công, {n_nan} step NaN/lỗi.")
    if len(losses) >= 10:
        first10 = sum(losses[:10]) / 10
        last10 = sum(losses[-10:]) / 10
        logger.info(f"Loss trung bình 10 step đầu: {first10:.4f} | 10 step cuối: {last10:.4f} "
                    f"({'GIẢM' if last10 < first10 else 'KHÔNG giảm'})")

    return {"losses": losses, "n_nan": n_nan, "n_total": len(samples)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
