"""
Bước 4 — Vòng lặp huấn luyện đầy đủ (CT9), theo đúng config.yaml.

CHẠY ĐƯỢC TRÊN CPU (chậm, dùng debug/mock) LẪN GPU (Colab, thật) — script
giống hệt nhau, chỉ khác `--set train.device=cuda`. Notebook Colab CHỈ gọi
script này (không viết logic riêng trong notebook), đúng yêu cầu ban đầu.

Thiết kế (kế thừa từ src/train/smoke_test.py đã smoke-test thành công 50
mẫu thật, xem docs/DECISIONS.md để biết các quyết định wiring):
  - Enc_LM(Q) = embed_tokens (không chạy qua toàn bộ LLM trước W_q).
  - K/V/O tái sử dụng từ layer 0 của Qwen2 (self_attn.k_proj/v_proj/o_proj).
  - R_final cộng dư (broadcast) vào mọi token của X.
  - "Các lớp còn lại của LLM" = chạy lại TOÀN BỘ Qwen2Model trên X_final.
  - batch_size=32 (config.yaml) thực hiện bằng GRADIENT ACCUMULATION (mỗi
    mẫu forward+backward riêng lẻ do độ dài token câu hỏi khác nhau, cộng
    dồn gradient qua 32 mẫu rồi mới optimizer.step() — tương đương batch
    thật về mặt cập nhật trọng số, khác về chuẩn hóa batch-norm (không có
    lớp batch-norm nào trong kiến trúc nên tương đương chính xác)).
  - Ảnh: dùng has_image (đã tính ở build_training_manifest.py) — mẫu có
    ảnh dùng đặc trưng ảnh thật NẾU đã tải (data/visual/images/<QID>.jpg
    tồn tại), còn lại dùng null_image_embedding (cơ chế đã anh duyệt).
  - Early stopping theo val MRR, patience theo config (early_stopping).
  - Checkpoint lưu vào train.checkpoint_dir (đường dẫn Google Drive khi
    mount trên Colab), tên file gồm seed + epoch + MRR để dễ so sánh.
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import add_config_args, load_config_from_args
from src.eval.metrics import compute_rank, summarize_ranks
from src.model.fusion import CrossAttentionFusion
from src.model.losses import MultiTaskLoss
from src.model.multimodal_encoder import MultimodalEncoder
from src.model.ranking_head import EntityRankingHead
from src.model.reliability_module import VisualReliabilityModule
from src.model.retriever import SubgraphRetriever
from src.model.spatiotemporal import SpatioTemporalEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VN_DEFAULT = (14.0583, 108.2772)


class Modules:
    """Gói toàn bộ module + state dùng chung giữa train/eval, tránh truyền quá nhiều tham số."""

    def __init__(self, cfg: dict, device: str):
        self.cfg = cfg
        self.device = device
        logger.info("Tải Vintern-1B-v2 thật (MultimodalEncoder.from_pretrained)...")
        self.encoder = MultimodalEncoder.from_pretrained(cfg).to(device)
        self.ste = SpatioTemporalEmbedding(cfg).to(device)
        self.retriever = SubgraphRetriever(cfg).to(device)
        self.fusion = CrossAttentionFusion(cfg).to(device)
        self.ranking_head = EntityRankingHead(cfg).to(device)
        self.reliability = VisualReliabilityModule(cfg).to(device)
        self.loss_fn = MultiTaskLoss(cfg)
        self.layer0_attn = self.encoder.text_backbone.base_model.model.model.layers[0].self_attn

    def trainable_adapter_params(self):
        return (
            [p for p in self.encoder.parameters() if p.requires_grad]
            + list(self.fusion.parameters())
            + list(self.ranking_head.parameters())
            + list(self.retriever.parameters())
            + list(self.reliability.parameters())
        )

    def trainable_embedding_params(self):
        return list(self.ste.parameters())


def build_entity_universe(df: pd.DataFrame) -> pd.DataFrame:
    entities = {}
    for _, row in df.iterrows():
        for id_col, label_col, lat_col, lon_col in [
            ("h", "h_label", "l_h_lat", "l_h_lon"), ("t", "t_label", "l_t_lat", "l_t_lon"),
        ]:
            eid = row[id_col]
            if eid not in entities:
                entities[eid] = {
                    "lat": row[lat_col] if pd.notna(row[lat_col]) else VN_DEFAULT[0],
                    "lon": row[lon_col] if pd.notna(row[lon_col]) else VN_DEFAULT[1],
                    "tau": row["tau_start"] if pd.notna(row["tau_start"]) else 2000.0,
                }
    return pd.DataFrame([{"entity": k, **v} for k, v in entities.items()])


def forward_one(m: Modules, row: pd.Series, candidate_pool: pd.DataFrame,
                 entity_to_idx: dict, entity_lat, entity_lon, entity_tau) -> dict:
    """1 forward pass đầy đủ CT(3)-(9) cho 1 câu hỏi. Trả về dict loss + rank
    (rank chỉ tính nếu caller truyền no_grad context, xem evaluate()).

    Đọc cfg["ablation"] (Mục 5 manuscript, 8 kịch bản) — RỖNG/không có key
    nào = mô hình đầy đủ (kịch bản 1, baseline). Mỗi cờ tương ứng 1 kịch bản:
      - disable_inward_attention (kịch bản 2): bỏ CT(7), R_final=0.
      - disable_spatial (kịch bản 3): ép lat/lon hằng số (bỏ định vị không gian).
      - disable_temporal (kịch bản 4): ép tau hằng số (bỏ nhúng thời gian).
      - disable_image (kịch bản 6): luôn dùng null_image_embedding dù có ảnh thật.
    alpha=1 (kịch bản 5) và chỉ L_QA (kịch bản 8, lambda_stkg=lambda_vg=0) đã
    có sẵn qua retrieval.alpha/loss.lambda_* trong config.yaml, không cần cờ
    riêng. Kịch bản 7 (bỏ trọng số tin cậy) qua model.visual_reliability_module.enabled=false
    (chưa đọc ở đây vì L_VG luôn mask=False khi chưa có ảnh thật — xem ghi chú dưới).
    Kịch bản 9 (RAG nối chuỗi thay chú ý 2 chiều) cần kiến trúc khác hẳn,
    CHƯA cài — xem docs/DECISIONS.md."""
    cfg = m.cfg
    device = m.device
    ablation = cfg.get("ablation", {})

    input_ids = m.encoder.tokenizer(row["question"], return_tensors="pt").input_ids.to(device)
    e_q = m.encoder.embed_tokens(input_ids)

    image_path = os.path.join(ROOT, "data", "visual", "images", f"{row['h']}.jpg")
    if row.get("has_image") and os.path.exists(image_path):
        # Ảnh thật đã tải (L1) — chạy qua vision_backbone thật (frozen InternViT).
        # Tiền xử lý tối giản: resize + to-tensor, KHÔNG dùng bộ tiền xử lý
        # chuẩn của InternViT (dynamic patch) — cần hoàn thiện khi có ảnh
        # thật số lượng đủ lớn để kiểm chứng (xem docs/DECISIONS.md).
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        pixel_values = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 224, 224, 3).permute(0, 3, 1, 2) / 255.0
        vision_out = m.encoder.encode_image(pixel_values.to(device))
        e_i = vision_out if vision_out.dim() == 3 else vision_out.unsqueeze(1)
        has_real_image = True
    else:
        e_i = m.encoder.null_image_features(1)
        has_real_image = False

    if ablation.get("disable_image", False):
        e_i = m.encoder.null_image_features(1)
        has_real_image = False

    x = m.encoder.fuse(e_q, e_i)

    cand = candidate_pool.sample(n=min(10, len(candidate_pool)))
    if ablation.get("disable_spatial", False):
        lat_h = torch.full((len(cand),), VN_DEFAULT[0], device=device)
        lon_h = torch.full((len(cand),), VN_DEFAULT[1], device=device)
        lat_t = torch.full((len(cand),), VN_DEFAULT[0], device=device)
        lon_t = torch.full((len(cand),), VN_DEFAULT[1], device=device)
    else:
        lat_h = torch.tensor(cand["l_h_lat"].fillna(VN_DEFAULT[0]).values, dtype=torch.float32, device=device)
        lon_h = torch.tensor(cand["l_h_lon"].fillna(VN_DEFAULT[1]).values, dtype=torch.float32, device=device)
        lat_t = torch.tensor(cand["l_t_lat"].fillna(VN_DEFAULT[0]).values, dtype=torch.float32, device=device)
        lon_t = torch.tensor(cand["l_t_lon"].fillna(VN_DEFAULT[1]).values, dtype=torch.float32, device=device)
    if ablation.get("disable_temporal", False):
        tau = torch.full((len(cand),), 2000.0, device=device)
    else:
        tau = torch.tensor(cand["tau_start"].fillna(2000.0).values, dtype=torch.float32, device=device)
    rel_ids = m.ste.relation_ids_from_names(cand["relation"].tolist()).to(device)

    ste_h = m.ste.encode_entity(lat_h, lon_h, tau)
    re_r = m.ste.encode_relation(rel_ids)
    ste_t = m.ste.encode_entity(lat_t, lon_t, tau)
    g_f = m.retriever.event_embedding(ste_h, re_r, ste_t)
    s_f = m.ste.score(lat_h, lon_h, tau, rel_ids, lat_t, lon_t)

    x_bar_raw = x.mean(dim=1).squeeze(0)
    rel_scores = m.retriever.relevance(x_bar_raw, g_f, s_f)
    top_k = min(m.retriever.top_k, len(cand))
    top_idx = m.retriever.top_k_indices(rel_scores, k=top_k)

    if ablation.get("disable_inward_attention", False):
        # Kịch bản 2 (Mục 5 manuscript): bỏ hẳn CT(7), R_final=0 -> X không đổi.
        x_final = x
    else:
        k_x_raw = m.layer0_attn.k_proj(x)
        v_x_raw = m.layer0_attn.v_proj(x)
        z_m = g_f[top_idx].unsqueeze(0)
        r_m = m.fusion(z_m, k_x_raw, v_x_raw, m.layer0_attn.o_proj)

        w_m = torch.softmax(rel_scores[top_idx], dim=0)
        r_final = (r_m.squeeze(0) * w_m.unsqueeze(-1)).sum(dim=0)
        x_final = x + r_final.view(1, 1, -1)

    lm_out = m.encoder.text_backbone.base_model.model.model(inputs_embeds=x_final, use_cache=False)
    final_hidden = lm_out.last_hidden_state.mean(dim=1)

    ste_entities_full = m.ste.encode_entity(entity_lat, entity_lon, entity_tau)
    p = m.ranking_head(final_hidden, ste_entities_full)
    qa_target = torch.tensor([entity_to_idx[row["answer_entity"]]], device=device)

    s_pos = s_f[:1]
    s_neg = s_f[1:2] if len(s_f) > 1 else s_f[:1] + 1.0

    if has_real_image:
        # L_VG thật cần scene-graph triplet (s,r,o) trích từ ảnh — CHƯA nối
        # (step3_visual/scene_graph_generator.py chạy riêng, ngoài phạm vi
        # smoke test/Bước 4 khởi tạo). Dùng placeholder 0 + mask=True tạm
        # thời SẼ SAI (không nên bật mask=True khi chưa có triplet thật) —
        # vì vậy vẫn giữ mask=False cho tới khi nối scene graph thật.
        vg_pred, vg_target, vg_mask = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, dtype=torch.bool, device=device)
    else:
        vg_pred, vg_target, vg_mask = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, dtype=torch.bool, device=device)

    out = m.loss_fn(p, qa_target, s_pos, s_neg, vg_pred, vg_target, vg_mask=vg_mask)
    out["p"] = p
    out["qa_target"] = qa_target
    return out


def evaluate(m: Modules, df: pd.DataFrame, entity_universe: pd.DataFrame, max_samples: int | None = None) -> dict:
    entity_to_idx = {e: i for i, e in enumerate(entity_universe["entity"])}
    entity_lat = torch.tensor(entity_universe["lat"].values, dtype=torch.float32, device=m.device)
    entity_lon = torch.tensor(entity_universe["lon"].values, dtype=torch.float32, device=m.device)
    entity_tau = torch.tensor(entity_universe["tau"].values, dtype=torch.float32, device=m.device)

    sub = df if max_samples is None else df.sample(n=min(max_samples, len(df)), random_state=0)
    ranks = []
    with torch.no_grad():
        for _, row in sub.iterrows():
            if row["answer_entity"] not in entity_to_idx:
                continue
            out = forward_one(m, row, df, entity_to_idx, entity_lat, entity_lon, entity_tau)
            rank = compute_rank(out["p"].squeeze(0), out["qa_target"].item())
            ranks.append(rank)
    return summarize_ranks(ranks)


def _resolve_dir(path: str) -> str:
    """os.path.join(ROOT, path) giữ nguyên path nếu đã tuyệt đối (vd Drive
    /content/drive/...) — os.path.join bỏ ROOT khi path thứ 2 tuyệt đối."""
    return os.path.join(ROOT, path)


def train_one_seed(cfg: dict, seed: int, device: str, tag: str = "default") -> dict:
    """Mỗi seed là 1 ĐƠN VỊ ĐỘC LẬP resume/skip được: nếu checkpoint tốt nhất
    VÀ file kết quả (MRR) của đúng seed+tag này đã tồn tại, bỏ qua huấn
    luyện (không tải Vintern, không train lại), load MRR từ file kết quả có
    sẵn và trả về ngay — quan trọng cho C3 (3 seed, tốn thời gian nhất, rủi
    ro runtime Colab bị ngắt giữa chừng cao nhất): xong seed nào lưu kết quả
    seed đó ngay, không chờ xong cả 3 seed."""
    ckpt_dir = _resolve_dir(cfg["train"]["checkpoint_dir"])
    results_dir = _resolve_dir(cfg["train"].get("results_dir", "results/training"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_best.pt")
    result_path = os.path.join(results_dir, f"{tag}_seed{seed}_result.json")

    if os.path.exists(ckpt_path) and os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            existing = json.load(f)
        logger.info(f"[seed={seed}, tag={tag}] ĐÃ CÓ kết quả đầy đủ (checkpoint + result json) "
                    f"-> bỏ qua huấn luyện, dùng lại best_val_mrr={existing['best_val_mrr']:.4f} "
                    f"từ {result_path}")
        return existing

    torch.manual_seed(seed)
    m = Modules(cfg, device)

    train_df = pd.read_csv(os.path.join(ROOT, "data", "vistqad", "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(ROOT, "data", "vistqad", "val_manifest.csv"))
    entity_universe = build_entity_universe(pd.concat([train_df, val_df]))
    entity_to_idx = {e: i for i, e in enumerate(entity_universe["entity"])}
    entity_lat = torch.tensor(entity_universe["lat"].values, dtype=torch.float32, device=device)
    entity_lon = torch.tensor(entity_universe["lon"].values, dtype=torch.float32, device=device)
    entity_tau = torch.tensor(entity_universe["tau"].values, dtype=torch.float32, device=device)

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW([
        {"params": m.trainable_adapter_params(), "lr": opt_cfg["lr_adapter"]},
        {"params": m.trainable_embedding_params(), "lr": opt_cfg["lr_embedding"]},
    ], weight_decay=opt_cfg["weight_decay"])

    batch_size = opt_cfg["batch_size"]
    max_epochs = opt_cfg["max_epochs"]
    patience = cfg["early_stopping"]["patience"]

    best_mrr = -1.0
    epochs_without_improve = 0
    history = []

    nan_max_ratio = cfg.get("train", {}).get("nan_stop_ratio", 0.05)  # >5% mẫu NaN trong 1 epoch -> DỪNG THẬT (không âm thầm bỏ qua)

    for epoch in range(max_epochs):
        t0 = time.time()
        shuffled = train_df.sample(frac=1, random_state=seed * 1000 + epoch).reset_index(drop=True)
        optimizer.zero_grad()
        running_loss = 0.0
        n_nan_epoch = 0
        n_seen = 0
        for i, row in shuffled.iterrows():
            if row["answer_entity"] not in entity_to_idx:
                continue
            n_seen += 1
            out = forward_one(m, row, train_df, entity_to_idx, entity_lat, entity_lon, entity_tau)
            loss = out["loss_total"] / batch_size
            if not torch.isfinite(loss):
                n_nan_epoch += 1
                logger.warning(f"Epoch {epoch} sample {i}: loss NaN/Inf ({n_nan_epoch} lần trong epoch này).")
                if n_nan_epoch / n_seen > nan_max_ratio and n_seen >= 20:
                    raise RuntimeError(
                        f"DỪNG THẬT: {n_nan_epoch}/{n_seen} mẫu NaN/Inf trong epoch {epoch} "
                        f"(> {nan_max_ratio*100:.0f}%) — có lỗi thật trong wiring, không phải nhiễu vặt."
                    )
                continue
            loss.backward()
            running_loss += out["loss_total"].item()
            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()

        val_metrics = evaluate(m, val_df, entity_universe, max_samples=cfg.get("eval_max_samples"))
        elapsed = time.time() - t0
        logger.info(f"[seed={seed}] Epoch {epoch}: loss={running_loss/len(shuffled):.4f} "
                    f"val_MRR={val_metrics['mrr']:.4f} val_Hit@1={val_metrics['hit@1']:.4f} ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": running_loss / len(shuffled), **val_metrics})

        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            epochs_without_improve = 0
            ckpt_path = os.path.join(ckpt_dir, f"seed{seed}_best.pt")
            torch.save({
                "epoch": epoch, "seed": seed, "val_mrr": best_mrr,
                "ste": m.ste.state_dict(), "fusion": m.fusion.state_dict(),
                "ranking_head": m.ranking_head.state_dict(), "retriever": m.retriever.state_dict(),
                "reliability": m.reliability.state_dict(),
                "lora": {k: v for k, v in m.encoder.text_backbone.state_dict().items() if "lora_" in k},
                "w_q": m.encoder.w_q.state_dict(), "w_v": m.encoder.w_v.state_dict(),
                "null_image_embedding": m.encoder.null_image_embedding,
            }, ckpt_path)
            logger.info(f"  -> checkpoint mới tốt nhất lưu tại {ckpt_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                logger.info(f"Early stop tại epoch {epoch} (patience={patience}).")
                break

    result = {
        "seed": seed, "best_val_mrr": best_mrr, "history": history,
        "alpha": cfg["retrieval"]["alpha"], "tag": tag,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"[seed={seed}, tag={tag}] Đã lưu kết quả -> {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default="default",
                         help="Nhãn phân biệt lần chạy (vd alpha_0.5) — tránh ghi đè "
                              "results/training/train_results_<tag>.json giữa các lần grid search.")
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    device = args.device or cfg["train"]["device"]
    seeds = args.seeds or cfg["train"]["seeds"]

    all_results = []
    for seed in seeds:
        logger.info(f"=== Bắt đầu huấn luyện seed={seed} trên device={device} (tag={args.tag}) ===")
        # train_one_seed tự resume/skip nếu seed này đã có đủ checkpoint+kết
        # quả, và tự ghi kết quả của NÓ ngay khi xong — không chờ các seed
        # khác trong vòng lặp này.
        result = train_one_seed(cfg, seed, device, tag=args.tag)
        all_results.append(result)

    # File tổng hợp (tiện xem nhanh cả batch seeds) — mỗi seed đã có file
    # riêng {tag}_seed{seed}_result.json ghi ngay sau khi xong, đây chỉ là
    # bản gộp, ghi lại vào ĐÚNG results_dir (có thể là Drive) như các file seed.
    results_dir = _resolve_dir(cfg["train"].get("results_dir", "results/training"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"train_results_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu kết quả gộp -> {out_path}")

    best = max(all_results, key=lambda r: r["best_val_mrr"])
    print(f"\n==RESULT== tag={args.tag} alpha={cfg['retrieval']['alpha']} "
          f"best_val_mrr_across_seeds={best['best_val_mrr']:.4f} n_seeds={len(all_results)}")
    return all_results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
