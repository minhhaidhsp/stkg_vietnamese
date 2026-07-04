import math

import torch

from src.model.losses import MultiTaskLoss
from src.model.ranking_head import EntityRankingHead
from src.model.reliability_module import VisualReliabilityModule
from src.model.spatiotemporal import SpatioTemporalEmbedding


def test_individual_losses_are_finite_and_nonnegative(cfg):
    loss_fn = MultiTaskLoss(cfg)
    batch, num_entities = 4, 10
    p = torch.softmax(torch.randn(batch, num_entities), dim=-1)
    target = torch.randint(0, num_entities, (batch,))
    s_pos = torch.rand(batch)
    s_neg = torch.rand(batch) + 0.5
    vg_pred = torch.sigmoid(torch.randn(batch))
    vg_target = torch.randint(0, 2, (batch,)).float()

    out = loss_fn(p, target, s_pos, s_neg, vg_pred, vg_target)
    for key in ["loss_qa", "loss_stkg", "loss_vg", "loss_total"]:
        assert torch.isfinite(out[key])
        assert out[key] >= 0


def test_total_loss_matches_weighted_sum(cfg):
    loss_fn = MultiTaskLoss(cfg)
    batch, num_entities = 3, 6
    p = torch.softmax(torch.randn(batch, num_entities), dim=-1)
    target = torch.randint(0, num_entities, (batch,))
    s_pos, s_neg = torch.rand(batch), torch.rand(batch) + 1.0
    vg_pred = torch.sigmoid(torch.randn(batch))
    vg_target = torch.randint(0, 2, (batch,)).float()

    out = loss_fn(p, target, s_pos, s_neg, vg_pred, vg_target)
    expected = (loss_fn.lambda_qa * out["loss_qa"]
                + loss_fn.lambda_stkg * out["loss_stkg"]
                + loss_fn.lambda_vg * out["loss_vg"])
    assert torch.allclose(out["loss_total"], expected, atol=1e-6)


def test_gradient_flows_through_all_three_branches(cfg):
    """Xây đồ thị tính toán thật qua 3 module (STE/RE, EntityRankingHead,
    VisualReliabilityModule) rồi kiểm tra gradient của loss_total chảy tới
    tham số của CẢ BA nhánh — đúng yêu cầu Bước 3."""
    torch.manual_seed(0)
    d = cfg["spatiotemporal_grid"]["embedding_dim"]
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]
    feat_dim = cfg["model"]["visual_reliability_module"]["feature_dim"]
    batch, num_entities = 4, 8

    ste = SpatioTemporalEmbedding(cfg)
    ranking_head = EntityRankingHead(cfg)
    reliability = VisualReliabilityModule(cfg)
    loss_fn = MultiTaskLoss(cfg)

    # --- Nhánh L_QA: EntityRankingHead ---
    final_hidden = torch.randn(batch, llm_hidden)
    ste_entities = torch.randn(num_entities, d)
    p = ranking_head(final_hidden, ste_entities)
    qa_target = torch.randint(0, num_entities, (batch,))

    # --- Nhánh L_STKG: SpatioTemporalEmbedding ---
    bbox = cfg["spatiotemporal_grid"]["spatial_bbox"]
    trange = cfg["spatiotemporal_grid"]["temporal_range"]
    lat_h = torch.rand(batch) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_h = torch.rand(batch) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    lat_t = torch.rand(batch) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_t = torch.rand(batch) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    lat_t_neg = torch.rand(batch) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_t_neg = torch.rand(batch) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    tau = torch.rand(batch) * (trange["tau_max"] - trange["tau_min"]) + trange["tau_min"]
    rel_ids = torch.zeros(batch, dtype=torch.long)

    s_pos = ste.score(lat_h, lon_h, tau, rel_ids, lat_t, lon_t)
    s_neg = ste.score(lat_h, lon_h, tau, rel_ids, lat_t_neg, lon_t_neg)  # negative sampling: thay t ngẫu nhiên

    # --- Nhánh L_VG: VisualReliabilityModule ---
    s_feat = torch.randn(batch, feat_dim)
    r_feat = torch.randn(batch, feat_dim)
    o_feat = torch.randn(batch, feat_dim)
    vg_pred = reliability(s_feat, r_feat, o_feat)
    vg_target = torch.randint(0, 2, (batch,)).float()

    out = loss_fn(p, qa_target, s_pos, s_neg, vg_pred, vg_target)
    out["loss_total"].backward()

    assert ranking_head.ranking_projection.weight.grad is not None
    assert torch.any(ranking_head.ranking_projection.weight.grad != 0)

    assert ste.E_x.weight.grad is not None
    assert torch.any(ste.E_x.weight.grad != 0)

    for name, param in reliability.named_parameters():
        assert param.grad is not None, f"Không có gradient cho {name} (nhánh L_VG)"
        assert torch.any(param.grad != 0)


def test_vg_loss_empty_mask_returns_zero_not_nan(cfg):
    """GUARD bắt buộc (yêu cầu duyệt trước smoke test): batch không có mẫu
    ảnh nào (mask toàn False) — tình huống BÌNH THƯỜNG vì ~90.7% facts
    không có ảnh — phải trả về 0.0 tường minh, KHÔNG NaN."""
    loss_fn = MultiTaskLoss(cfg)
    batch = 8
    vg_pred = torch.sigmoid(torch.randn(batch))
    vg_target = torch.randint(0, 2, (batch,)).float()
    mask = torch.zeros(batch, dtype=torch.bool)  # không mẫu nào có ảnh

    l_vg = loss_fn.vg_loss(vg_pred, vg_target, mask=mask)
    assert not torch.isnan(l_vg)
    assert torch.equal(l_vg, torch.zeros(()))


def test_vg_loss_empty_tensor_returns_zero_not_nan(cfg):
    """Trường hợp pred/target rỗng ngay từ đầu (đã lọc sẵn trước khi gọi)."""
    loss_fn = MultiTaskLoss(cfg)
    vg_pred = torch.empty(0)
    vg_target = torch.empty(0)
    l_vg = loss_fn.vg_loss(vg_pred, vg_target)
    assert not torch.isnan(l_vg)
    assert torch.equal(l_vg, torch.zeros(()))


def test_vg_loss_partial_mask_only_averages_masked_samples(cfg):
    loss_fn = MultiTaskLoss(cfg)
    vg_pred = torch.tensor([0.9, 0.1, 0.9, 0.1])
    vg_target = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mask = torch.tensor([True, False, True, False])  # chỉ 2 mẫu đầu/thứ 3 có ảnh, đều dự đoán đúng (0.9 gần 1)

    l_vg = loss_fn.vg_loss(vg_pred, vg_target, mask=mask)
    expected = torch.nn.functional.binary_cross_entropy(
        torch.tensor([0.9, 0.9]).clamp(1e-6, 1 - 1e-6), torch.tensor([1.0, 1.0])
    )
    assert torch.allclose(l_vg, expected, atol=1e-6)


def test_total_loss_finite_when_batch_has_no_images(cfg):
    """loss_total của CẢ BATCH vẫn phải hữu hạn (không NaN) khi L_VG=0 do
    không mẫu nào có ảnh — L_QA/L_STKG không bị ảnh hưởng."""
    loss_fn = MultiTaskLoss(cfg)
    batch, num_entities = 8, 12
    p = torch.softmax(torch.randn(batch, num_entities), dim=-1)
    target = torch.randint(0, num_entities, (batch,))
    s_pos, s_neg = torch.rand(batch), torch.rand(batch) + 1.0
    vg_pred = torch.sigmoid(torch.randn(batch))
    vg_target = torch.randint(0, 2, (batch,)).float()
    vg_mask = torch.zeros(batch, dtype=torch.bool)

    out = loss_fn(p, target, s_pos, s_neg, vg_pred, vg_target, vg_mask=vg_mask)
    assert torch.isfinite(out["loss_total"])
    assert torch.equal(out["loss_vg"], torch.zeros(()))
    assert not math.isnan(out["loss_total"].item())


def test_gradient_still_flows_to_qa_and_stkg_when_batch_has_no_images(cfg):
    """Khi L_VG=0 (batch toàn không ảnh), gradient của L_QA/L_STKG vẫn phải
    chảy bình thường qua các module tương ứng — không bị "đóng băng" lây."""
    torch.manual_seed(0)
    d = cfg["spatiotemporal_grid"]["embedding_dim"]
    llm_hidden = cfg["model"]["backbone"]["llm_hidden_size"]
    batch, num_entities = 5, 8

    ranking_head = EntityRankingHead(cfg)
    ste = SpatioTemporalEmbedding(cfg)
    loss_fn = MultiTaskLoss(cfg)

    final_hidden = torch.randn(batch, llm_hidden)
    ste_entities = torch.randn(num_entities, d)
    p = ranking_head(final_hidden, ste_entities)
    qa_target = torch.randint(0, num_entities, (batch,))

    bbox = cfg["spatiotemporal_grid"]["spatial_bbox"]
    trange = cfg["spatiotemporal_grid"]["temporal_range"]
    lat_h = torch.rand(batch) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_h = torch.rand(batch) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    lat_t = torch.rand(batch) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_t = torch.rand(batch) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    lat_t_neg = torch.rand(batch) * (bbox["lat_max"] - bbox["lat_min"]) + bbox["lat_min"]
    lon_t_neg = torch.rand(batch) * (bbox["lon_max"] - bbox["lon_min"]) + bbox["lon_min"]
    tau = torch.rand(batch) * (trange["tau_max"] - trange["tau_min"]) + trange["tau_min"]
    rel_ids = torch.zeros(batch, dtype=torch.long)
    s_pos = ste.score(lat_h, lon_h, tau, rel_ids, lat_t, lon_t)
    s_neg = ste.score(lat_h, lon_h, tau, rel_ids, lat_t_neg, lon_t_neg)

    vg_pred = torch.sigmoid(torch.randn(batch))  # không nối vào graph của module nào (mô phỏng batch không ảnh)
    vg_target = torch.randint(0, 2, (batch,)).float()
    vg_mask = torch.zeros(batch, dtype=torch.bool)  # cả batch không có ảnh

    out = loss_fn(p, qa_target, s_pos, s_neg, vg_pred, vg_target, vg_mask=vg_mask)
    out["loss_total"].backward()

    assert ranking_head.ranking_projection.weight.grad is not None
    assert torch.any(ranking_head.ranking_projection.weight.grad != 0)
    assert ste.E_x.weight.grad is not None
    assert torch.any(ste.E_x.weight.grad != 0)
